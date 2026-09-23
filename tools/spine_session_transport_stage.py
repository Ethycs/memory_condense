"""Authenticate a staged raw recovery bound to an observed terminal exec session."""
import hashlib
from pathlib import Path

import psutil

from tools.execute_spine_corpus import prepare as prepare_corpus
from tools.matched_eval.artifacts import read_sealed_json
from tools.stage_spine_transport_recovery import journal_state


def require_terminal(observation):
    p = observation.payload
    if (p.get("format") != "memory-condense-terminal-ingest-session-observation-v1" or
            type(p.get("session_id")) is not int or p["session_id"] <= 0 or
            not isinstance(p.get("terminal_tool_chunk_id"), str) or not p["terminal_tool_chunk_id"] or
            p.get("ingest_exit_code") != 1 or p.get("original_process_absent") is not True or
            p.get("observed_exception") != "openai.APITimeoutError caused by httpx.ReadTimeout" or
            p.get("reservations_cleared") is not False or p.get("process_termination_performed") is not False):
        raise ValueError("a terminal exec timeout observation with preserved journals is required")
    parent = observation.path.parent
    plan = read_sealed_json(parent / "preflight.json")
    failure = read_sealed_json(parent / "failure.json")
    release = read_sealed_json(parent / "release.json")
    if (plan.sha256 != p["scheduler_preflight_sha256"] or failure.sha256 != p["scheduler_failure_sha256"] or
            release.sha256 != p["scheduler_release_sha256"] or
            failure.payload["preflight_sha256"] != plan.sha256 or
            release.payload["preflight_sha256"] != plan.sha256 or
            failure.payload["exception_type"] != "APITimeoutError" or
            failure.payload["automatic_retry_performed"] is not False or
            failure.payload["original_reservations_preserved"] is not True):
        raise ValueError("terminal observation is not bound to its failed scheduler execution")
    # The operator-observed terminal tool session is the exit evidence. Check
    # that its scheduler command has not since been started again as well.
    for process in psutil.process_iter(["name"]):
        if not (process.info["name"] or "").lower().startswith("python"):
            continue
        try:
            if any(arg.replace("\\", "/").rsplit("/", 1)[-1].lower() == "run_remaining.py"
                   for arg in process.cmdline()):
                raise ValueError("the original remaining-ingest scheduler command is live")
        except psutil.NoSuchProcess:
            pass


def validate(root):
    root = Path(root).resolve()
    plan = read_sealed_json(root / "stage-preflight.json")
    p = plan.payload
    stage_source = root / "stage_from_terminal.py"
    if (p["format"] != "memory-condense-spine-transport-recovery-stage-v3" or
            p["terminal_identity_kind"] != "recorded exec session exit plus scheduler failure" or
            p["implementation_sha256"] != hashlib.sha256(stage_source.read_bytes()).hexdigest() or
            p["automatic_retries"] != 0 or p["provider_execution_enabled"] is not False or
            p["full100_method_verification_required"] is not True):
        raise ValueError("the terminal-session staging implementation or policy changed")
    original = Path(p["original_corpus_root"]).resolve()
    if original == root or original.is_relative_to(root) or root.is_relative_to(original):
        raise ValueError("transport recovery requires separate original and successor roots")
    observation = read_sealed_json(Path(p["terminal_observation_path"]))
    if observation.sha256 != p["terminal_observation_sha256"]:
        raise ValueError("terminal observation changed after staging")
    require_terminal(observation)
    offset, count = p["shard_offset"], p["full_request_count"]
    corpus, namespace, requests, execution = prepare_corpus(original, offset, count)
    observed = observation.payload
    if (namespace["request_count"] != count or corpus.sha256 != p["corpus_preflight_sha256"] or
            execution.sha256 != p["execution_preflight_sha256"] or
            observed["corpus_preflight_sha256"] != corpus.sha256 or observed["shard_offset"] != offset or
            observed["full_request_count"] != count):
        raise ValueError("terminal staging must cover the exact complete source population")
    rows = [{"batch_index": i, "raw_request_sha256": request.sha256, "raw_request_path": binding["path"],
        **journal_state(original / f"offset-{offset:03d}/raw-checkpoints" / request.sha256)}
        for i, (request, binding) in enumerate(zip(requests, namespace["requests"], strict=True))]
    counts = {state: sum(r["state"] == state for r in rows)
        for state in ("completed", "reserved_without_response", "unstarted")}
    if (rows != observed["rows"] or counts != observed["counts"] or
            observed["response_files"] != counts["completed"] or
            observed["request_reservations"] != counts["completed"] + counts["reserved_without_response"] or
            observed["unacknowledged"] != [r for r in rows if r["state"] == "reserved_without_response"]):
        raise ValueError("original response or reservation population changed after the terminal observation")
    snapshot = read_sealed_json(root / "original-inventory.json")
    if snapshot.sha256 != p["original_inventory_sha256"] or snapshot.payload != {
        "corpus_preflight_sha256": corpus.sha256, "execution_preflight_sha256": execution.sha256,
        "terminal_observation_sha256": observation.sha256, "shard_offset": offset,
        "full_request_count": count, "rows": rows, "counts": counts, "new_provider_calls": 0}:
        raise ValueError("the staged original inventory changed")
    work = [{"raw_request_sha256": r["raw_request_sha256"], "raw_request_path": r["raw_request_path"],
        "batch_index": r["batch_index"], "prior_state": r["state"],
        "prior_reservation_journal_shas": r["reservation_journal_shas"], "maximum_additional_attempts": 1}
        for r in rows if r["state"] != "completed"]
    if (p["raw_work"] != work or p["retained_completed_raw_responses"] != counts["completed"] or
            p["first_attempt_raw_requests"] != counts["unstarted"] or
            p["maximum_reissued_raw_requests"] != counts["reserved_without_response"] or
            p["maximum_new_provider_calls"] != len(work)):
        raise ValueError("the recovery allowance differs from the complete terminal inventory")
    return plan
