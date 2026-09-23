"""Stage a complete namespace after a bound terminal raw-ingest failure.

No provider calls. Preserve completed journals byte for byte and keep all
original unresolved reservations. Every non-completed request is listed once;
the successor requires a transport-aware admission verifier before full100 use.
"""
import argparse
import hashlib
from pathlib import Path

import psutil

from tools.execute_spine_corpus import prepare as corpus_prepare
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.stage_spine_transport_recovery import copy_completed_checkpoint, journal_state


def require_terminal(observation):
    p = observation.payload
    if (p.get("ingest_exit_code") != 1 or p.get("original_process_absent") is not True or
            p.get("observed_exception") != "openai.APITimeoutError caused by httpx.ReadTimeout" or
            p.get("reservations_cleared") is not False or p.get("process_termination_performed") is not False):
        raise ValueError("recovery requires the observed terminal timeout with original journals retained")
    try:
        process = psutil.Process(p["ingest_pid"])
        if process.create_time() == p["ingest_process_create_time"]:
            raise ValueError("the original ingest process is still live")
    except psutil.NoSuchProcess:
        pass


def inventory(corpus_root, offset, observation):
    require_terminal(observation)
    manifest = read_sealed_json(corpus_root / "preflight.json")
    namespace = next(n for n in manifest.payload["namespaces"] if n["shard_offset"] == offset)
    corpus, namespace, requests, execution = corpus_prepare(corpus_root, offset, namespace["request_count"])
    rows = []
    for request, binding in zip(requests, namespace["requests"], strict=True):
        state = journal_state(corpus_root / f"offset-{offset:03d}/raw-checkpoints" / request.sha256)
        rows.append({"batch_index": request.payload["batch_index"], "raw_request_sha256": request.sha256,
            "raw_request_path": binding["path"], **state})
    observed = observation.payload
    counts = {state: sum(row["state"] == state for row in rows)
              for state in ("completed", "reserved_without_response", "unstarted")}
    unresolved = {row["raw_request_sha256"] for row in rows if row["state"] == "reserved_without_response"}
    if (counts["completed"] != observed["response_files"] or
            counts["completed"] + counts["reserved_without_response"] != observed["request_reservations"] or
            unresolved != {row["raw_request_sha256"] for row in observed["unacknowledged"]}):
        raise ValueError("original response/reservation population changed after the terminal observation")
    return corpus, namespace, requests, execution, rows, counts


def prepare(root, corpus_root, offset, terminal_path):
    root, corpus_root = root.resolve(), corpus_root.resolve()
    root.relative_to(Path.cwd().resolve())
    if (root == Path.cwd().resolve() or root == corpus_root or
            root.is_relative_to(corpus_root) or corpus_root.is_relative_to(root)):
        raise ValueError("recovery must use a separate directory inside the workspace")
    observation = read_sealed_json(terminal_path)
    corpus, namespace, _, execution, rows, counts = inventory(corpus_root, offset, observation)
    snapshot, _ = publish_sealed_json(root / "original-inventory.json", {
        "corpus_preflight_sha256": corpus.sha256, "execution_preflight_sha256": execution.sha256,
        "terminal_observation_sha256": observation.sha256, "shard_offset": offset,
        "full_request_count": namespace["request_count"], "rows": rows, "counts": counts,
        "new_provider_calls": 0})
    work = [{"raw_request_sha256": row["raw_request_sha256"], "raw_request_path": row["raw_request_path"],
        "batch_index": row["batch_index"], "prior_state": row["state"],
        "prior_reservation_journal_shas": row["reservation_journal_shas"], "maximum_additional_attempts": 1}
        for row in rows if row["state"] != "completed"]
    artifact, _ = publish_sealed_json(root / "stage-preflight.json", {
        "format": "memory-condense-spine-transport-recovery-stage-v2",
        "original_corpus_root": str(corpus_root), "shard_offset": offset,
        "terminal_observation_path": str(terminal_path.resolve()), "terminal_observation_sha256": observation.sha256,
        "original_inventory_sha256": snapshot.sha256, "corpus_preflight_sha256": corpus.sha256,
        "execution_preflight_sha256": execution.sha256, "full_request_count": namespace["request_count"],
        "raw_work": work, "retained_completed_raw_responses": counts["completed"],
        "first_attempt_raw_requests": counts["unstarted"],
        "maximum_reissued_raw_requests": counts["reserved_without_response"],
        "maximum_new_provider_calls": len(work), "automatic_retries": 0,
        "original_outcomes_without_responses": "unknown; count all original attempts conservatively",
        "provider_execution_enabled": False, "full100_method_verification_required": True,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print({"stage_preflight_sha256": artifact.sha256, "preserved_responses": counts["completed"],
        "first_attempt_requests": counts["unstarted"], "maximum_reissued_requests": counts["reserved_without_response"],
        "maximum_future_calls": len(work), "new_provider_calls": 0}, flush=True)
    return artifact


def validate(root):
    plan = read_sealed_json(root / "stage-preflight.json")
    p = plan.payload
    if (p["format"] != "memory-condense-spine-transport-recovery-stage-v2" or
            p["implementation_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError("staging implementation or format changed")
    replay = prepare(root, Path(p["original_corpus_root"]), p["shard_offset"], Path(p["terminal_observation_path"]))
    if replay.sha256 != plan.sha256:
        raise ValueError("staging population changed")
    return plan


def stage(root):
    root = root.resolve()
    plan = validate(root)
    p = plan.payload
    original, offset = Path(p["original_corpus_root"]), p["shard_offset"]
    corpus, namespace, requests, execution = corpus_prepare(original, offset, p["full_request_count"])
    target = root / "corpus"
    publish_sealed_json(target / "preflight.json", corpus.payload)
    copied = []
    for request, binding in zip(requests, namespace["requests"], strict=True):
        destination = target / binding["path"]
        destination.resolve().relative_to(target.resolve())
        publish_sealed_json(destination, request.payload)
        old_checkpoint = original / f"offset-{offset:03d}/raw-checkpoints" / request.sha256
        new_checkpoint = target / f"offset-{offset:03d}/raw-checkpoints" / request.sha256
        prior = journal_state(old_checkpoint)
        if prior["state"] == "completed":
            files = copy_completed_checkpoint(old_checkpoint, new_checkpoint, request)
            copied.append({"offset": offset, "raw_request_sha256": request.sha256, "files": files, **prior})
        elif journal_state(new_checkpoint)["state"] != "unstarted":
            raise ValueError("staging cannot overwrite or clear a successor attempt")
    if (len(copied) != p["retained_completed_raw_responses"] or
            corpus_prepare(target, offset, p["full_request_count"])[3].sha256 != execution.sha256):
        raise ValueError("staging changed the retained responses or execution protocol")
    validate(root)
    artifact, _ = publish_sealed_json(root / "stage.json", {
        "format": "memory-condense-spine-transport-recovery-stage-result-v2",
        "stage_preflight_sha256": plan.sha256, "original_inventory_sha256": p["original_inventory_sha256"],
        "copied_completed_requests": copied, "copied_completed_request_count": len(copied),
        "staged_corpus_root": str(target), "shard_offset": offset,
        "original_failed_reservations_preserved": True, "new_provider_calls": 0,
        "execution_implemented": False, "existing_full100_certificate_eligible": False})
    print({"stage_sha256": artifact.sha256, "copied_completed_requests": len(copied),
        "maximum_future_calls": p["maximum_new_provider_calls"], "new_provider_calls": 0}, flush=True)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "stage"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--shard-offset", type=int)
    parser.add_argument("--terminal-observation", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output_root, args.corpus_root, args.shard_offset, args.terminal_observation)
    else:
        stage(args.output_root)
