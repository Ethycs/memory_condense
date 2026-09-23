"""Compact every over-budget summary in bounded, deterministic batches.

The complete namespace audit determines the population. Each Qwen request has
at most eight summary-only jobs. All batches must validate before an aggregate
repair artifact can be admitted; failed reservations are never cleared.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.spine_merge_batch import merge_batch_messages, pack_merge_batches, parse_merge_batch
from memory_condense.search.spine_quote_json_repair_v3 import repair_support_list_closures
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest
from tools.build_spine_corpus_hierarchy import GATEWAY, MODEL, restore_request
from tools.execute_spine_corpus import prepare as corpus_prepare
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.repair_spine_summary_budget import apply_repairs
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized


IMPLEMENTATION = ("tools/repair_spine_summary_budget_v4.py", "tools/repair_spine_summary_budget.py",
    "src/memory_condense/search/spine_merge_batch.py", "src/memory_condense/search/spine_summary.py",
    "src/memory_condense/search/spine_quote_json_repair.py", "src/memory_condense/search/spine_quote_json_repair_v2.py", "src/memory_condense/search/spine_quote_json_repair_v3.py")


def batches_for(rows):
    keys = [(row["raw_request_sha256"], row["label"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("summary repair contains duplicate attributed jobs")
    jobs = tuple(restore_request(row["job"]) for row in rows)
    if not jobs:
        raise ValueError("summary repair requires a nonempty audited population")
    batches, start = [], 0
    for index, batch in enumerate(pack_merge_batches(jobs)):
        body = {"batch_index": index, "row_indices": list(range(start, start + len(batch))),
                "messages": merge_batch_messages(batch)}
        batches.append({**body, "batch_sha256": identity_sha256(body)})
        start += len(batch)
    if start != len(rows):
        raise ValueError("summary batching lost an audited job")
    return batches


def prepare(root, corpus_root, audit_path, offset, limit):
    corpus, namespace, requests, execution = corpus_prepare(corpus_root, offset, limit)
    if namespace["request_count"] != limit:
        raise ValueError("multi-batch repair requires a complete namespace audit")
    audit = read_sealed_json(audit_path)
    if audit.payload["execution_preflight_sha256"] != execution.sha256 or audit.payload["request_count"] != limit:
        raise ValueError("summary audit belongs to another execution or prefix")
    by_sha = {r.sha256: r for r in requests}
    rows, seen = [], set()
    for failure in audit.payload["failures"]:
        if failure["unresolved_schema_failure"] or not failure["invalid_summaries"]:
            raise ValueError("schema or attribution failures cannot be repaired as summary length")
        request = by_sha[failure["request_sha256"]]
        p = request.payload
        runtime = FastCompletionRuntime(checkpoint_dir=corpus_root / f"offset-{offset:03d}/raw-checkpoints" / request.sha256,
            prompt_population=[p["messages"]], model=p["model"], client=None,
            max_prompt_tokens=7000, max_new_tokens=3072, max_concurrency=1, retries=0,
            benchmark_provenance={"raw_request_sha256": request.sha256})
        try:
            response = runtime.run().logical_completions[0]
        finally:
            runtime.close()
        parsed, _ = repair_support_list_closures(response)
        atoms = json.loads(parsed)["atoms"]
        for bad in failure["invalid_summaries"]:
            index = int(bad["label"][1:])
            atom, span = atoms[index], p["raw_spans"][index]
            key = (request.sha256, bad["label"])
            if (key in seen or atom["label"] != bad["label"] or atom["summary"] != bad["summary"] or
                    count_tokens(atom["summary"]) <= 128):
                raise ValueError("only each exact audited over-budget summary may be compacted")
            seen.add(key)
            job = SpineSummaryRequest("user_spine" if span["role"] == "user" else "attached_context",
                (SpineSummaryFragment(span["role"], span["created_at"], atom["summary"]),), max_output_tokens=128)
            rows.append({"raw_request_sha256": request.sha256, "label": atom["label"],
                "original_summary_sha256": quote_sha256(atom["summary"]), "role": span["role"], "job": asdict(job)})
    batches = batches_for(rows)
    result, _ = publish_sealed_json(root / "preflight.json", {
        "format": "memory-condense-spine-summary-budget-repair-v4",
        "corpus_root": str(corpus_root.resolve()), "audit_path": str(audit_path.resolve()),
        "shard_offset": offset, "request_limit": limit,
        "corpus_preflight_sha256": corpus.sha256, "execution_preflight_sha256": execution.sha256,
        "audit_sha256": audit.sha256, "rows": rows, "batches": batches,
        "model": MODEL, "gateway": GATEWAY, "maximum_provider_calls": len(batches),
        "maximum_jobs_per_batch": 8, "retries": 0, "raw_qwen_inputs": False,
        "complete_namespace": True,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}})
    print({"preflight_sha256": result.sha256, "summary_jobs": len(rows),
           "provider_batches": len(batches), "provider_calls": 0}, flush=True)
    return result


def run(root, enable=False):
    preflight = read_sealed_json(root / "preflight.json")
    p = preflight.payload
    # Reconstruct every model input from the saved original summaries before
    # allowing calls; a resealed edited job is not source authentication.
    prepare(root, Path(p["corpus_root"]), Path(p["audit_path"]), p["shard_offset"], p["request_limit"])
    if (p["format"] != "memory-condense-spine-summary-budget-repair-v4" or
            p["implementation"] != {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION} or
            p["batches"] != batches_for(p["rows"]) or p["maximum_provider_calls"] != len(p["batches"]) or
            p["maximum_jobs_per_batch"] != 8 or p["retries"] != 0 or p["raw_qwen_inputs"] is not False or
            p["model"] != MODEL or p["gateway"] != GATEWAY or p["complete_namespace"] is not True):
        raise ValueError("multi-batch summary compaction protocol changed")
    values, receipts, calls, hits = [], [], 0, 0
    for batch_row in p["batches"]:
        jobs = tuple(restore_request(p["rows"][i]["job"]) for i in batch_row["row_indices"])
        def factory(client):
            return FastCompletionRuntime(checkpoint_dir=root / "checkpoints" / batch_row["batch_sha256"],
                prompt_population=[batch_row["messages"]], model=MODEL, client=client,
                max_prompt_tokens=7000, max_new_tokens=2048, max_concurrency=1, retries=0,
                request_options={"temperature": 0, "extra_body": {"enable_thinking": False}},
                benchmark_provenance={"summary_budget_preflight_sha256": preflight.sha256,
                                      "summary_budget_batch_sha256": batch_row["batch_sha256"]})
        runtime = factory(None)
        try:
            remaining = 1 - len(_authenticated_records(runtime))
        finally:
            runtime.close()
        result, used, reused, _ = _run_exactly_authorized(runtime_factory=factory,
            authorized_provider_calls=remaining, enable_provider=enable,
            client_factory=lambda: _completion_client("LITELLM_KEY", GATEWAY))
        values.extend(parse_merge_batch(result.logical_completions[0], jobs))
        receipts.extend(r.response_journal_sha256 for r in result.unique_records)
        calls += used
        hits += reused
        print({"batch": batch_row["batch_index"], "summary_jobs": len(jobs), "new_calls": used, "replay_hits": reused}, flush=True)
    rows = [{k: row[k] for k in ("raw_request_sha256", "label", "original_summary_sha256", "role")} |
            {"summary": value, "summary_sha256": quote_sha256(value)}
            for row, value in zip(p["rows"], values, strict=True)]
    artifact, _ = publish_sealed_json(root / "repairs.json", {"preflight_sha256": preflight.sha256,
        "corpus_preflight_sha256": p["corpus_preflight_sha256"], "rows": rows,
        "response_journal_shas": receipts, "raw_qwen_inputs": False,
        "successful_batch_count": len(p["batches"])})
    print({"repairs_sha256": artifact.sha256, "compacted_summaries": len(rows),
           "new_calls": calls, "replay_hits": hits}, flush=True)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--shard-offset", type=int)
    parser.add_argument("--request-limit", type=int)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output_root, args.corpus_root, args.audit, args.shard_offset, args.request_limit)
    else:
        run(args.output_root, args.enable_provider)
