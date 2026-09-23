"""Authenticate one compaction-only transport reissue without changing summaries."""
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval.fast_completion_runtime import _read_journal
from tools import finish_spine_compaction_batches as completion
from tools.matched_eval.artifacts import read_sealed_json
from tools.stage_spine_transport_recovery import journal_state


FORMAT = "memory-condense-compaction-only-transport-stage-v1"


def original_batches_root(repair_root):
    if repair_root is None:
        return None
    repair_root = Path(repair_root).resolve()
    p = read_sealed_json(repair_root / "preflight.json").payload
    if p.get("format") == "memory-condense-spine-summary-budget-recovery-v2":
        return Path(p["parent_root"]).resolve()
    return repair_root


def declared_stage(repair_root):
    parent = original_batches_root(repair_root)
    if parent is None or not (parent.parent / "stage.json").exists():
        return None
    stage = read_sealed_json(parent.parent / "stage.json")
    return parent.parent if stage.payload.get("format") == FORMAT else None


def verify(root, repair_root, admission):
    root = Path(root).resolve()
    stage = read_sealed_json(root / "stage.json")
    p = stage.payload
    original_root, successor = Path(p["original_root"]).resolve(), Path(p["successor_root"]).resolve()
    if (p["format"] != FORMAT or p["maximum_additional_attempts"] != 1 or p["automatic_retries"] != 0 or
            p["raw_inputs_to_qwen"] is not False or p["original_reservation_preserved"] is not True or
            p["original_attempt_counted_conservatively"] is not True or original_root == successor or
            successor.parent != root or successor != original_batches_root(repair_root)):
        raise ValueError("compaction-only transport policy or repair root changed")
    original = completion.original_context(original_root)
    if (original.sha256 != p["original_preflight_sha256"] or
            read_sealed_json(successor / "preflight.json").sha256 != original.sha256 or
            original.payload["corpus_preflight_sha256"] != admission["corpus_preflight_sha256"] or
            original.payload["execution_preflight_sha256"] != admission["execution_preflight_sha256"]):
        raise ValueError("compaction transport belongs to another admitted namespace")
    attempts, preserved = [], []
    expected = {b["batch_sha256"] for b in original.payload["batches"]}
    for base in (original_root, successor):
        observed = {path.name for path in (base / "checkpoints").glob("*")
            if path.is_dir() and any(path.glob("*.json"))}
        if observed != expected:
            raise ValueError("compaction transport omitted or added a batch")
    for batch in original.payload["batches"]:
        old_dir = original_root / "checkpoints" / batch["batch_sha256"]
        new_dir = successor / "checkpoints" / batch["batch_sha256"]
        old, new = journal_state(old_dir), journal_state(new_dir)
        if new["state"] != "completed" or old["state"] not in ("completed", "reserved_without_response"):
            raise ValueError("compaction transport requires every original and completed successor")
        old_request, old_sha = _read_journal(next(old_dir.glob("*.request.json")))
        new_request, new_sha = _read_journal(next(new_dir.glob("*.request.json")))
        if old_request != new_request or old_sha != new_sha:
            raise ValueError("compaction reissue changed the original request")
        runtime = completion.runtime_for(successor, original, batch)
        try:
            protocol = runtime._request_body(identity_sha256(batch["messages"]))
        finally:
            runtime.close()
        if {k: v for k, v in old_request.items() if k != "journal_sha256"} != protocol:
            raise ValueError("compaction request does not authenticate its original protocol")
        if old["state"] == "completed":
            if old != new:
                raise ValueError("compaction transport replaced a completed original response")
            preserved.extend(old["response_journal_shas"])
        else:
            if (batch["batch_sha256"] != p["original_batch_sha256"] or old != p["original_state"] or
                    old_sha != p["original_request_journal_sha256"]):
                raise ValueError("compaction transport changed the declared unresolved population")
            attempts.append({"original_request_journal_sha256": old_sha,
                "successor_response_journal_sha256": new["response_journal_shas"][0]})
    if len(attempts) != 1:
        raise ValueError("compaction transport must count exactly one additional attempt")
    execution = read_sealed_json(root / "execution/preflight.json")
    if execution.sha256 != p["execution_preflight_sha256"] or execution.payload["maximum_new_calls"] != 1:
        raise ValueError("compaction reissue release changed its attempt budget")
    replay = completion.run(root / "execution", False)
    if replay.payload["preflight_sha256"] != execution.sha256:
        raise ValueError("compaction reissue completion binding changed")
    return {"stage_sha256": stage.sha256, "execution_preflight_sha256": execution.sha256,
        "execution_complete_sha256": replay.sha256, "original_preflight_sha256": original.sha256,
        "preserved_response_journal_shas": preserved, "attempts": attempts,
        "additional_compaction_attempts": 1, "new_provider_calls": 0}
