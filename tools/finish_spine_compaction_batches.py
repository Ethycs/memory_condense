"""Finish unstarted compaction batches while retaining completed invalid outputs.

This only executes original, already prepared summary-only requests. Invalid
completed slots are reported for separate bounded recovery; no original batch
is resent to improve its content. Unacknowledged reservations fail closed.
"""
import argparse
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from tools import repair_spine_summary_budget_v4 as parent
from tools.build_spine_corpus_hierarchy import restore_request
from tools.build_spine_corpus_hierarchy_resilient import recoverable_slots
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized


IMPLEMENTATION = (*parent.IMPLEMENTATION, "tools/finish_spine_compaction_batches.py",
    "tools/build_spine_corpus_hierarchy_resilient.py", "src/memory_condense/eval/fast_completion_runtime.py")


def original_context(parent_root):
    original = read_sealed_json(parent_root / "preflight.json")
    p = original.payload
    rebuilt = parent.prepare(parent_root, Path(p["corpus_root"]), Path(p["audit_path"]),
        p["shard_offset"], p["request_limit"])
    if (rebuilt.sha256 != original.sha256 or p["format"] != "memory-condense-spine-summary-budget-repair-v4" or
            p["model"] != parent.MODEL or p["gateway"] != parent.GATEWAY or
            p["raw_qwen_inputs"] is not False or p["retries"] != 0 or
            p["batches"] != parent.batches_for(p["rows"])):
        raise ValueError("original summary compaction protocol changed")
    return original


def runtime_for(parent_root, original, batch, client=None):
    return FastCompletionRuntime(checkpoint_dir=parent_root / "checkpoints" / batch["batch_sha256"],
        prompt_population=[batch["messages"]], model=parent.MODEL, client=client,
        max_prompt_tokens=7000, max_new_tokens=2048, max_concurrency=1, retries=0,
        request_options={"temperature": 0, "extra_body": {"enable_thinking": False}},
        benchmark_provenance={"summary_budget_preflight_sha256": original.sha256,
            "summary_budget_batch_sha256": batch["batch_sha256"]})


def inventory(parent_root, original):
    rows = []
    for batch in original.payload["batches"]:
        runtime = runtime_for(parent_root, original, batch)
        try:
            records = _authenticated_records(runtime)
        finally:
            runtime.close()
        if len(records) > 1:
            raise ValueError("original batch has more than one completion")
        rows.append({"batch_sha256": batch["batch_sha256"],
            "response_journal_shas": [r.response_journal_sha256 for r in records.values()]})
    return rows


def prepare(root, parent_root):
    if root.resolve() == parent_root.resolve():
        raise ValueError("completion accounting requires a separate output root")
    original = original_context(parent_root)
    rows = inventory(parent_root, original)
    artifact, _ = publish_sealed_json(root / "preflight.json", {
        "format": "memory-condense-finish-original-compaction-batches-v1",
        "parent_root": str(parent_root.resolve()), "parent_preflight_sha256": original.sha256,
        "batches": rows, "maximum_new_calls": sum(not r["response_journal_shas"] for r in rows),
        "raw_qwen_inputs": False, "automatic_retries": 0,
        "completed_originals_must_be_preserved": True,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}})
    print({"completion_preflight_sha256": artifact.sha256,
        "preserved_batches": sum(bool(r["response_journal_shas"]) for r in rows),
        "unstarted_batches": artifact.payload["maximum_new_calls"], "new_provider_calls": 0}, flush=True)
    return artifact


def run(root, enable=False):
    preflight = read_sealed_json(root / "preflight.json")
    p = preflight.payload
    parent_root = Path(p["parent_root"])
    original = original_context(parent_root)
    current = inventory(parent_root, original)
    if (p["format"] != "memory-condense-finish-original-compaction-batches-v1" or
            p["parent_preflight_sha256"] != original.sha256 or p["raw_qwen_inputs"] is not False or
            p["automatic_retries"] != 0 or p["completed_originals_must_be_preserved"] is not True or
            p["implementation"] != {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION} or
            len(current) != len(p["batches"]) or
            p["maximum_new_calls"] != sum(not r["response_journal_shas"] for r in p["batches"])):
        raise ValueError("original completion plan changed")
    for before, after in zip(p["batches"], current, strict=True):
        if (before["batch_sha256"] != after["batch_sha256"] or
                (before["response_journal_shas"] and before != after)):
            raise ValueError("a completed original batch was replaced")
    remaining = sum(not r["response_journal_shas"] for r in current)
    if remaining:
        if not enable or remaining != p["maximum_new_calls"]:
            raise ValueError("only the complete prepared unstarted batch population may be released")
        with (root / "execution.reserved").open("x", encoding="utf-8") as handle:
            handle.write(preflight.sha256 + "\n")
    elif p["maximum_new_calls"]:
        marker = root / "execution.reserved"
        if not marker.is_file() or marker.read_text(encoding="utf-8") != preflight.sha256 + "\n":
            raise ValueError("completed new batches have no matching recorded release")
    rows, calls, hits = [], 0, 0
    for batch, observed in zip(original.payload["batches"], current, strict=True):
        result, used, reused, _ = _run_exactly_authorized(
            runtime_factory=lambda client: runtime_for(parent_root, original, batch, client),
            authorized_provider_calls=int(not observed["response_journal_shas"]), enable_provider=enable,
            client_factory=lambda: _completion_client("LITELLM_KEY", parent.GATEWAY))
        jobs = tuple(restore_request(original.payload["rows"][i]["job"]) for i in batch["row_indices"])
        values, missing = recoverable_slots(result.logical_completions[0], jobs)
        rows.append({"batch_sha256": batch["batch_sha256"], "row_indices": batch["row_indices"],
            "response_sha256": quote_sha256(result.logical_completions[0]),
            "response_journal_shas": [r.response_journal_sha256 for r in result.unique_records],
            "valid_summaries": len(values) - len(missing), "invalid_slots": missing})
        calls += used
        hits += reused
    if calls != remaining:
        raise ValueError("original batch completion call accounting changed")
    artifact, _ = publish_sealed_json(root / "complete.json", {
        "preflight_sha256": preflight.sha256, "parent_preflight_sha256": original.sha256,
        "rows": rows, "all_original_batches_completed": True,
        "original_batch_count": len(rows), "raw_qwen_inputs": False})
    print({"original_completion_sha256": artifact.sha256, "new_calls": calls, "replay_hits": hits,
        "invalid_summaries": sum(len(r["invalid_slots"]) for r in rows)}, flush=True)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.parent_root is None or args.enable_provider:
            parser.error("prepare requires --parent-root and no provider flag")
        prepare(args.output_root, args.parent_root)
    else:
        run(args.output_root, args.enable_provider)
