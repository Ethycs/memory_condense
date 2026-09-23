"""Stage one explicit transport recovery without clearing failed reservations.

This tool makes no provider calls. It copies verified completed journals into a
separate execution root and binds every omitted reservation to the earlier
terminal-run inventory. The staged root is not yet approved for the existing
full100 admission certificate: that certificate permits only one compaction
attempt per namespace, whereas this recovery needs a versioned attempt audit.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime, _read_journal
from tools.execute_spine_corpus import prepare
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def copy_exact(source: Path, target: Path):
    """Copy immutable evidence, refusing changed or aliased destinations."""
    if source.is_symlink() or not source.is_file() or target.is_symlink():
        raise ValueError("recovery inputs and outputs must be regular files")
    content = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.read_bytes() != content:
            raise ValueError("recovery destination differs from its original evidence")
    else:
        with target.open("xb") as handle:
            handle.write(content)
    return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def journal_state(checkpoint: Path):
    requests = sorted(checkpoint.glob("*.request.json"))
    responses = sorted(checkpoint.glob("*.response.json"))
    if len(requests) > 1 or len(responses) > 1 or (responses and not requests):
        raise ValueError("unexpected single-request journal population")
    return {
        "state": "completed" if responses else "reserved_without_response" if requests else "unstarted",
        "reservation_journal_shas": [_read_journal(p)[1] for p in requests],
        "response_journal_shas": [_read_journal(p)[1] for p in responses],
    }


def copy_completed_checkpoint(source: Path, target: Path, request):
    """Replay the actual runtime contract before copying a completed response."""
    p = request.payload
    runtime = FastCompletionRuntime(checkpoint_dir=source, prompt_population=[p["messages"]],
        model=p["model"], client=None, max_prompt_tokens=7000, max_new_tokens=3072,
        max_concurrency=1, retries=0, benchmark_provenance={"raw_request_sha256": request.sha256})
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    if batch.usage.physical_calls != 0 or batch.usage.checkpoint_hits != 1:
        raise ValueError("completed recovery inputs must replay without provider calls")
    files = {}
    for path in sorted(source.glob("*.json")):
        if not path.name.endswith((".request.json", ".response.json")):
            raise ValueError("unexpected journal file")
        files[path.name] = copy_exact(path, target / path.name)
    return files


def validate_inventory(plan_path: Path):
    plan = read_sealed_json(plan_path)
    p = plan.payload
    if p["format"] != "memory-condense-spine-transport-recovery-plan-v1":
        raise ValueError("unknown recovery plan")
    original = plan_path.parent.resolve()
    inventory = read_sealed_json(original / "gateway-timeout-inventory-20260910-r1.json")
    if inventory.sha256 != p["timeout_inventory_sha256"]:
        raise ValueError("recovery inventory changed")
    observed = inventory.payload
    if (observed["corpus_preflight_sha256"] != p["corpus_preflight_sha256"] or
            observed["offset020"]["execution_preflight_sha256"] != p["offset020_execution_preflight_sha256"] or
            {(r["session_id"], r["exit_code"], r["error_type"]) for r in observed["terminal_process_observations"]} !=
            {(43793, 1, "APITimeoutError"), (91664, 1, "APITimeoutError")}):
        raise ValueError("recovery must bind the recorded terminal failures")
    expected_work = []
    corpus = read_sealed_json(original / "preflight.json")
    namespace = next(n for n in corpus.payload["namespaces"] if n["shard_offset"] == 20)
    if len(observed["offset020"]["rows"]) != len(namespace["requests"]):
        raise ValueError("recovery inventory omits source requests")
    for ordinal, row in enumerate(observed["offset020"]["rows"]):
        binding = namespace["requests"][ordinal]
        if row["batch_index"] != ordinal or row["raw_request_sha256"] != binding["sha256"]:
            raise ValueError("recovery request order or source binding changed")
        state = journal_state(original / "offset-020/raw-checkpoints" / binding["sha256"])
        if any(row[k] != state[k] for k in state):
            raise ValueError("original request state changed; reassess the recovery allowance")
        if state["state"] != "completed":
            expected_work.append({"batch_index": ordinal, "raw_request_path": binding["path"],
                "raw_request_sha256": binding["sha256"], "prior_state": state["state"],
                "prior_reservation_journal_shas": state["reservation_journal_shas"],
                "maximum_additional_attempts": 1})
    if p["raw_work"] != expected_work:
        raise ValueError("recovery may include every missing request exactly once and no completed request")
    new = sum(r["prior_state"] == "unstarted" for r in expected_work)
    repeated = len(expected_work) - new
    if (p["first_attempt_raw_requests"] != new or p["maximum_reissued_raw_requests"] != repeated or
            p["retained_completed_raw_responses"] != len(namespace["requests"]) - len(expected_work) or
            p["maximum_new_provider_calls"] != len(expected_work) + 1):
        raise ValueError("recovery call counts do not match the original journal population")
    repair = p["summary_compaction"]
    original_repair = Path(repair["original_root"])
    prior = observed["offset010_compaction"]
    state = journal_state(original_repair / "checkpoints")
    preflight = read_sealed_json(original_repair / "preflight.json")
    if (original_repair.resolve() != Path(prior["root"]).resolve() or
            preflight.sha256 != repair["preflight_sha256"] or preflight.sha256 != prior["preflight_sha256"] or
            state["state"] != "reserved_without_response" or
            state["reservation_journal_shas"] != [prior["request_journal_sha256"]] or
            repair["original_reservation_journal_sha256"] != prior["request_journal_sha256"] or
            repair["maximum_additional_attempts"] != 1 or repair["raw_qwen_inputs"] is not False or
            repair["model"] != preflight.payload["model"] or
            repair["summary_jobs"] != len(preflight.payload["rows"])):
        raise ValueError("summary compaction recovery binding changed")
    return original, plan, inventory, original_repair


def stage(plan_path: Path, root: Path):
    original, plan, inventory, original_repair = validate_inventory(plan_path)
    root = root.resolve()
    workspace = Path.cwd().resolve()
    root.relative_to(workspace)
    if root == workspace or root == original or original.is_relative_to(root) or root.is_relative_to(original):
        raise ValueError("recovery requires a separate workspace directory")
    # Authenticate complete source populations before writing their successor.
    populations = [prepare(original, offset, limit) for offset, limit in ((10, 801), (20, 817))]
    corpus = populations[0][0]
    if corpus.sha256 != plan.payload["corpus_preflight_sha256"]:
        raise ValueError("recovery corpus changed")
    preflight, _ = publish_sealed_json(root / "stage-preflight.json", {
        "format": "memory-condense-spine-transport-recovery-stage-v1",
        "recovery_plan_sha256": plan.sha256, "timeout_inventory_sha256": inventory.sha256,
        "original_corpus_root": str(original), "original_compaction_root": str(original_repair.resolve()),
        "maximum_new_provider_calls": plan.payload["maximum_new_provider_calls"],
        "provider_execution_enabled": False, "full100_method_verification_required": True,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    target_corpus = root / "corpus"
    publish_sealed_json(target_corpus / "preflight.json", corpus.payload)
    copied = []
    for _, namespace, requests, execution in populations:
        offset = namespace["shard_offset"]
        for binding, request in zip(namespace["requests"], requests, strict=True):
            relative = Path(binding["path"])
            (target_corpus / relative).resolve().relative_to(target_corpus.resolve())
            publish_sealed_json(target_corpus / relative, request.payload)
            source = original / f"offset-{offset:03d}" / "raw-checkpoints" / request.sha256
            state = journal_state(source)
            if offset == 10 and state["state"] != "completed":
                raise ValueError("the completed second memory changed")
            if state["state"] == "completed":
                target = target_corpus / f"offset-{offset:03d}" / "raw-checkpoints" / request.sha256
                files = copy_completed_checkpoint(source, target, request)
                copied.append({"offset": offset, "raw_request_sha256": request.sha256,
                    "files": files, **state})
        successor = prepare(target_corpus, offset, len(requests))[3]
        if successor.sha256 != execution.sha256:
            raise ValueError("staging changed the original request protocol")
    repair_preflight = read_sealed_json(original_repair / "preflight.json")
    publish_sealed_json(root / "summary-repair-offset010/preflight.json", repair_preflight.payload)
    # Re-check every unresolved original after copying; a newly recovered
    # response must reduce the allowance, never be silently discarded.
    validate_inventory(plan_path)
    result, _ = publish_sealed_json(root / "stage.json", {
        "stage_preflight_sha256": preflight.sha256, "recovery_plan_sha256": plan.sha256,
        "copied_completed_requests": copied, "copied_completed_request_count": len(copied),
        "original_failed_reservations_preserved": True, "new_provider_calls": 0,
        "staged_corpus_root": str(target_corpus),
        "staged_compaction_root": str(root / "summary-repair-offset010"),
        "execution_implemented": False, "existing_full100_certificate_eligible": False})
    print({"stage_sha256": result.sha256, "copied_completed_requests": len(copied),
        "maximum_future_calls": plan.payload["maximum_new_provider_calls"], "new_provider_calls": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    stage(args.plan, args.output_root)
