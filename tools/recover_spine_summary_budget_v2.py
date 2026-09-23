"""Preserve completed compaction batches and recover only invalid summary slots."""
import argparse
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from tools import repair_spine_summary_budget_v4 as parent
from tools.build_spine_corpus_hierarchy import restore_request
from tools.build_spine_corpus_hierarchy_resilient import IMPLEMENTATION as RECOVERY_IMPLEMENTATION, RecoveryJournal, recoverable_slots
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


IMPLEMENTATION = tuple(dict.fromkeys((*parent.IMPLEMENTATION, *RECOVERY_IMPLEMENTATION,
    "tools/recover_spine_summary_budget_v2.py")))


def prepare(root, parent_root):
    original = read_sealed_json(parent_root / "preflight.json")
    p = original.payload
    parent.prepare(parent_root, Path(p["corpus_root"]), Path(p["audit_path"]), p["shard_offset"], p["request_limit"])
    values, failures, responses = [], [], []
    for batch in p["batches"]:
        jobs = tuple(restore_request(p["rows"][i]["job"]) for i in batch["row_indices"])
        runtime = FastCompletionRuntime(checkpoint_dir=parent_root / "checkpoints" / batch["batch_sha256"],
            prompt_population=[batch["messages"]], model=parent.MODEL, client=None,
            max_prompt_tokens=7000, max_new_tokens=2048, max_concurrency=1, retries=0,
            request_options={"temperature": 0, "extra_body": {"enable_thinking": False}},
            benchmark_provenance={"summary_budget_preflight_sha256": original.sha256,
                                  "summary_budget_batch_sha256": batch["batch_sha256"]})
        try:
            result = runtime.run()
        finally:
            runtime.close()
        response = result.logical_completions[0]
        slots, missing = recoverable_slots(response, jobs)
        values.extend(slots)
        failures.extend({"row_index": batch["row_indices"][i], "slot": i,
            "failed_response_sha256": quote_sha256(response)} for i in missing)
        responses.extend(r.response_journal_sha256 for r in result.unique_records)
    if not failures:
        raise ValueError("summary recovery requires an invalid completed compaction slot")
    artifact, _ = publish_sealed_json(root / "preflight.json", {
        "format": "memory-condense-spine-summary-budget-recovery-v2",
        "parent_root": str(parent_root.resolve()), "parent_preflight_sha256": original.sha256,
        "corpus_preflight_sha256": p["corpus_preflight_sha256"], "execution_preflight_sha256": p["execution_preflight_sha256"],
        "rows": p["rows"], "original_values": values, "failures": failures,
        "original_response_journal_shas": responses, "original_batch_count": len(p["batches"]),
        "model": parent.MODEL, "gateway": parent.GATEWAY, "raw_qwen_inputs": False,
        "maximum_recovery_calls": 2 * len(failures), "recovery_words": [48, 24], "retries": 0,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}})
    print({"recovery_preflight_sha256": artifact.sha256, "preserved_valid_summaries": len(values) - len(failures),
        "failed_summaries": len(failures), "maximum_recovery_calls": 2 * len(failures), "new_calls": 0}, flush=True)
    return artifact


def run(root, enable=False, budget=0):
    preflight = read_sealed_json(root / "preflight.json")
    p = preflight.payload
    prepare(root, Path(p["parent_root"]))
    if type(budget) is not int or not 0 <= budget <= p["maximum_recovery_calls"]:
        raise ValueError("recovery call allowance exceeds the complete failed-slot population")
    journal = RecoveryJournal(root, preflight, enable, budget)
    values = list(p["original_values"])
    for failure in p["failures"]:
        index = failure["row_index"]
        values[index] = journal.recover(restore_request(p["rows"][index]["job"]),
            failure["failed_response_sha256"], failure["slot"])
    if any(before is not None and before != after for before, after in zip(p["original_values"], values, strict=True)):
        raise ValueError("a previously valid original summary was changed")
    rows = [{k: row[k] for k in ("raw_request_sha256", "label", "original_summary_sha256", "role")} |
            {"summary": value, "summary_sha256": quote_sha256(value)}
            for row, value in zip(p["rows"], values, strict=True)]
    artifact, _ = publish_sealed_json(root / "repairs.json", {
        "preflight_sha256": preflight.sha256, "corpus_preflight_sha256": p["corpus_preflight_sha256"],
        "rows": rows, "raw_qwen_inputs": False, "original_response_journal_shas": p["original_response_journal_shas"],
        "original_batch_count": p["original_batch_count"], "recovered_summary_count": len(p["failures"]),
        "recovery_attempts": journal.calls + journal.hits, "recovery_receipts": journal.recoveries})
    print({"repairs_sha256": artifact.sha256, "compacted_summaries": len(rows),
        "new_calls": journal.calls, "replay_hits": journal.hits, "original_batch_hits": p["original_batch_count"]}, flush=True)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--max-new-calls", type=int, default=0)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output_root, args.parent_root)
    else:
        run(args.output_root, args.enable_provider, args.max_new_calls)
