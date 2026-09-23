"""Compact only audited over-budget ingest summaries through summary-only Qwen."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.spine_merge_batch import merge_batch_messages, parse_merge_batch
from memory_condense.search.spine_quote_json_repair import repair_support_list_closures
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest
from tools.build_spine_corpus_hierarchy import restore_request, GATEWAY, MODEL
from tools.execute_spine_corpus import prepare as corpus_prepare
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized


IMPLEMENTATION = ("tools/repair_spine_summary_budget.py", "src/memory_condense/search/spine_merge_batch.py",
                  "src/memory_condense/search/spine_summary.py", "src/memory_condense/search/spine_quote_json_repair.py")


def prepare(root, corpus_root, audit_path, offset, limit):
    corpus, _, requests, execution = corpus_prepare(corpus_root, offset, limit)
    audit = read_sealed_json(audit_path)
    if audit.payload["execution_preflight_sha256"] != execution.sha256:
        raise ValueError("summary budget audit belongs to another execution")
    by_sha = {r.sha256: r for r in requests}
    rows = []
    for failure in audit.payload["failures"]:
        request = by_sha[failure["request_sha256"]]
        p = request.payload
        runtime = FastCompletionRuntime(checkpoint_dir=corpus_root / f"offset-{offset:03d}" / "raw-checkpoints" / request.sha256,
            prompt_population=[p["messages"]], model=p["model"], client=None,
            max_prompt_tokens=7000, max_new_tokens=3072, max_concurrency=1, retries=0,
            benchmark_provenance={"raw_request_sha256": request.sha256})
        try:
            original = runtime.run().logical_completions[0]
        finally:
            runtime.close()
        parsed, _ = repair_support_list_closures(original)
        atoms = json.loads(parsed)["atoms"]
        for bad in failure["invalid_summaries"]:
            index = int(bad["label"][1:])
            atom, span = atoms[index], p["raw_spans"][index]
            if atom["label"] != bad["label"] or atom["summary"] != bad["summary"] or count_tokens(atom["summary"]) <= 128:
                raise ValueError("only the exact audited over-budget summary may be compacted")
            job = SpineSummaryRequest("user_spine" if span["role"] == "user" else "attached_context",
                (SpineSummaryFragment(span["role"], span["created_at"], atom["summary"]),), max_output_tokens=128)
            rows.append({"raw_request_sha256": request.sha256, "label": atom["label"],
                "original_summary_sha256": quote_sha256(atom["summary"]), "role": span["role"], "job": asdict(job)})
    jobs = tuple(restore_request(row["job"]) for row in rows)
    messages = merge_batch_messages(jobs)
    result, _ = publish_sealed_json(root / "preflight.json", {"format": "memory-condense-spine-summary-budget-repair-v1",
        "corpus_preflight_sha256": corpus.sha256, "execution_preflight_sha256": execution.sha256,
        "audit_sha256": audit.sha256, "rows": rows, "messages": messages, "model": MODEL, "gateway": GATEWAY,
        "maximum_provider_calls": 1, "retries": 0, "raw_qwen_inputs": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}})
    print({"preflight_sha256": result.sha256, "summary_jobs": len(rows), "provider_calls": 0}, flush=True)


def run(root, enable=False):
    preflight = read_sealed_json(root / "preflight.json")
    p = preflight.payload
    if p["implementation"] != {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}:
        raise ValueError("summary compaction implementation changed")
    jobs = tuple(restore_request(row["job"]) for row in p["rows"])
    if merge_batch_messages(jobs) != p["messages"]:
        raise ValueError("summary compaction prompt changed")
    def factory(client):
        return FastCompletionRuntime(checkpoint_dir=root / "checkpoints", prompt_population=[p["messages"]],
            model=MODEL, client=client, max_prompt_tokens=7000, max_new_tokens=2048, max_concurrency=1, retries=0,
            request_options={"temperature": 0, "extra_body": {"enable_thinking": False}},
            benchmark_provenance={"preflight_sha256": preflight.sha256})
    audit = factory(None)
    try:
        remaining = 1 - len(_authenticated_records(audit))
    finally:
        audit.close()
    batch, calls, hits, _ = _run_exactly_authorized(runtime_factory=factory, authorized_provider_calls=remaining,
        enable_provider=enable, client_factory=lambda: _completion_client("LITELLM_KEY", GATEWAY))
    values = parse_merge_batch(batch.logical_completions[0], jobs)
    rows = [{k: row[k] for k in ("raw_request_sha256", "label", "original_summary_sha256", "role")} |
            {"summary": value, "summary_sha256": quote_sha256(value)} for row, value in zip(p["rows"], values, strict=True)]
    result, _ = publish_sealed_json(root / "repairs.json", {"preflight_sha256": preflight.sha256,
        "corpus_preflight_sha256": p["corpus_preflight_sha256"], "rows": rows,
        "response_journal_shas": [r.response_journal_sha256 for r in batch.unique_records], "raw_qwen_inputs": False})
    print({"repairs_sha256": result.sha256, "compacted_summaries": len(rows), "new_calls": calls, "replay_hits": hits}, flush=True)
    return result


def apply_repairs(response, raw_request_sha, rows):
    body = json.loads(response)
    applied = []
    for repair in rows:
        if repair["raw_request_sha256"] != raw_request_sha:
            continue
        index = int(repair["label"][1:])
        atom = body["atoms"][index]
        if (atom["label"] != repair["label"] or quote_sha256(atom["summary"]) != repair["original_summary_sha256"] or
            count_tokens(atom["summary"]) <= 128 or quote_sha256(repair["summary"]) != repair["summary_sha256"] or
            not repair["summary"].strip() or count_tokens(repair["summary"]) > 128):
            raise ValueError("summary compaction attribution, input or budget changed")
        atom["summary"] = repair["summary"]
        applied.append(repair)
    return json.dumps(body, ensure_ascii=False), applied


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--shard-offset", type=int, default=0)
    parser.add_argument("--request-limit", type=int, default=850)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output_root, args.corpus_root, args.audit, args.shard_offset, args.request_limit)
    else:
        run(args.output_root, args.enable_provider)
