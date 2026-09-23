"""Execute a bounded namespace recovery after fresh successful model readiness."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from tools.execute_spine_corpus import execute as execute_corpus
from tools.execute_spine_transport_recovery import require_readiness
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.spine_transport_lineage_v2 import stage_state


IMPLEMENTATION = ("tools/execute_spine_transport_recovery_v2.py", "tools/spine_transport_lineage_v2.py",
    "tools/stage_spine_transport_recovery_v2.py", "tools/execute_spine_corpus.py",
    "tools/execute_spine_transport_recovery.py", "tools/probe_spine_gateway_readiness.py",
    "src/memory_condense/eval/fast_completion_runtime.py")


def execution_payload(state):
    p = state["plan"].payload
    return {"format": "memory-condense-spine-transport-successor-execution-v2",
        "stage_sha256": state["stage"].sha256, "stage_preflight_sha256": state["plan"].sha256,
        "shard_offset": p["shard_offset"], "full_request_count": p["full_request_count"],
        "maximum_new_calls": p["maximum_new_provider_calls"],
        "maximum_reissued_requests": p["maximum_reissued_raw_requests"],
        "maximum_additional_attempts_per_unresolved_request": 1, "automatic_retries": 0,
        "original_failed_reservations_preserved": True, "max_concurrency": 4,
        "requires_admission_verifier": "tools.verify_spine_admission_method_v7",
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}}


def prepare(root):
    stage = read_sealed_json(root / "stage.json")
    state = stage_state(Path(stage.payload["staged_corpus_root"]), stage.payload["shard_offset"])
    artifact, _ = publish_sealed_json(root / "execution-raw-v2.json", execution_payload(state))
    print({"execution_preflight_sha256": artifact.sha256, "remaining_requests": len(state["pending"]),
        "preserved_responses": state["preserved_completed_raw_responses"], "new_provider_calls": 0}, flush=True)
    return artifact, state


def run(root, readiness_path=None, enable=False):
    execution, state = prepare(root)
    p = state["plan"].payload
    remaining = len(state["pending"])
    marker = root / "execution-raw-v2.reserved"
    if remaining:
        if not enable or readiness_path is None:
            raise ValueError("new recovery calls require explicit execution and fresh successful readiness")
        if state["completed_new"] or remaining != p["maximum_new_provider_calls"]:
            raise ValueError("a partially executed successor cannot receive a second release")
        readiness = require_readiness(readiness_path)
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(execution.sha256 + "\n")
        publish_sealed_json(root / "readiness" / f"raw-v2-{readiness.sha256}.json", {
            "execution_preflight_sha256": execution.sha256, "readiness_report_sha256": readiness.sha256,
            "readiness_report_path": str(readiness_path.resolve()),
            "started_utc": datetime.now(timezone.utc).isoformat(), "additional_calls_at_start": remaining})
    elif not marker.exists():
        raise ValueError("completed successor has no authorized release receipt")
    corpus_root = Path(state["stage"].payload["staged_corpus_root"])
    try:
        execute_corpus(corpus_root, p["shard_offset"], p["full_request_count"], bool(remaining))
    except Exception as exc:
        publish_sealed_json(root / "execution-failure.json", {
            "execution_preflight_sha256": execution.sha256, "exception_type": type(exc).__name__,
            "original_failed_reservations_preserved": True, "automatic_retry_performed": False})
        raise
    from tools.spine_transport_lineage_v2 import verify_transport_lineage
    lineage = verify_transport_lineage(corpus_root, p["shard_offset"], None)
    artifact, _ = publish_sealed_json(root / "execution-complete.json", {
        "execution_preflight_sha256": execution.sha256, "transport_lineage": lineage,
        "complete_raw_namespace": True, "source_admission_claimed": False})
    print({"execution_complete_sha256": artifact.sha256, "new_calls": remaining,
        "additional_original_unknown_attempts": lineage["additional_raw_attempts"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output_root)
    else:
        run(args.output_root, args.readiness_report, args.enable_provider)
