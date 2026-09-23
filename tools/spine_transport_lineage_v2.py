"""Verify a complete raw namespace's explicit terminal-run recovery lineage."""
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime, _read_journal
from tools import spine_transport_lineage as previous
from tools.execute_spine_corpus import prepare as prepare_corpus
from tools.matched_eval.artifacts import read_sealed_json
from tools.stage_spine_transport_recovery import journal_state
from tools.stage_spine_transport_recovery_v2 import validate


TRANSPORT_POLICY = previous.TRANSPORT_POLICY


def runtime_for(corpus_root, offset, request):
    p = request.payload
    return FastCompletionRuntime(
        checkpoint_dir=corpus_root / f"offset-{offset:03d}/raw-checkpoints" / request.sha256,
        prompt_population=[p["messages"]], model=p["model"], client=None,
        max_prompt_tokens=7000, max_new_tokens=3072, max_concurrency=1, retries=0,
        benchmark_provenance={"raw_request_sha256": request.sha256})


def authenticate_reservation(corpus_root, offset, request, *, expected_corpus_root=None):
    checkpoint = corpus_root / f"offset-{offset:03d}/raw-checkpoints" / request.sha256
    body, digest = _read_journal(next(checkpoint.glob("*.request.json")))
    # An unresolved original reservation must remain non-replayable. Derive the
    # path-independent request protocol from the checked successor instead.
    runtime = runtime_for(expected_corpus_root or corpus_root, offset, request)
    try:
        expected = runtime._request_body(identity_sha256(request.payload["messages"]))
    finally:
        runtime.close()
    plain = dict(body)
    plain.pop("journal_sha256")
    if plain != expected:
        raise ValueError("transport attempt changed its original request protocol")
    return digest


def stage_state(corpus_root, offset, *, complete=False):
    """Authenticate all old states and all current successes before execution."""
    corpus_root = corpus_root.resolve()
    root = corpus_root.parent
    stage = read_sealed_json(root / "stage.json")
    s = stage.payload
    plan = validate(root)
    p = plan.payload
    if (s["format"] != "memory-condense-spine-transport-recovery-stage-result-v2" or
            s["stage_preflight_sha256"] != plan.sha256 or s["original_inventory_sha256"] != p["original_inventory_sha256"] or
            Path(s["staged_corpus_root"]).resolve() != corpus_root or corpus_root != root / "corpus" or
            s["shard_offset"] != offset or p["shard_offset"] != offset):
        raise ValueError("transport stage belongs to a different corpus or namespace")
    original = Path(p["original_corpus_root"])
    if (original.parent / "stage.json").exists():
        raise ValueError("nested transport recovery requires separate cumulative attempt accounting")
    corpus, namespace, requests, execution = prepare_corpus(corpus_root, offset, p["full_request_count"])
    if (corpus.sha256 != p["corpus_preflight_sha256"] or execution.sha256 != p["execution_preflight_sha256"] or
            namespace["request_count"] != p["full_request_count"]):
        raise ValueError("transport recovery changed the full raw population")
    copied = {row["raw_request_sha256"]: row for row in s["copied_completed_requests"]}
    work = {row["raw_request_sha256"]: row for row in p["raw_work"]}
    expected = {r.sha256 for r in requests}
    if (len(copied) != len(s["copied_completed_requests"]) or len(work) != len(p["raw_work"]) or
            set(copied) & set(work) or set(copied) | set(work) != expected or
            len(copied) != s["copied_completed_request_count"] or len(work) != p["maximum_new_provider_calls"]):
        raise ValueError("transport stage omitted, duplicated or added a request")
    observed_directories = {d.name for d in (corpus_root / f"offset-{offset:03d}/raw-checkpoints").glob("*")
                            if d.is_dir() and any(d.glob("*.json"))}
    if not observed_directories <= expected:
        raise ValueError("transport successor has foreign request journals")
    pending, additional, completed_new = [], [], []
    for request in requests:
        sha = request.sha256
        old_dir = original / f"offset-{offset:03d}/raw-checkpoints" / sha
        new_dir = corpus_root / f"offset-{offset:03d}/raw-checkpoints" / sha
        old_state, new_state = journal_state(old_dir), journal_state(new_dir)
        if sha in copied:
            row = copied[sha]
            if row["offset"] != offset or any(row[k] != old_state[k] or old_state[k] != new_state[k] for k in old_state):
                raise ValueError("a retained successful response was dropped or replaced")
            actual_files = {f.name for f in old_dir.glob("*.json")}
            if actual_files != set(row["files"]):
                raise ValueError("retained response file inventory changed")
            for name, receipt in row["files"].items():
                for directory in (old_dir, new_dir):
                    content = (directory / name).read_bytes()
                    if len(content) != receipt["bytes"] or hashlib.sha256(content).hexdigest() != receipt["sha256"]:
                        raise ValueError("a retained successful response was dropped or replaced")
        else:
            row = work[sha]
            if (row["prior_state"] != old_state["state"] or
                    row["prior_reservation_journal_shas"] != old_state["reservation_journal_shas"] or
                    row["maximum_additional_attempts"] != 1):
                raise ValueError("original unresolved request state or attempt allowance changed")
            if new_state["state"] == "reserved_without_response":
                raise RuntimeError("successor request is unacknowledged; it cannot be retried")
            if new_state["state"] == "unstarted":
                if complete:
                    raise ValueError("transport verification requires the complete successor population")
                pending.append(sha)
            else:
                completed_new.append(sha)
            if old_state["state"] == "reserved_without_response":
                old_digest = authenticate_reservation(original, offset, request, expected_corpus_root=corpus_root)
                new_digest = authenticate_reservation(corpus_root, offset, request) if new_state["state"] == "completed" else None
                if new_digest is not None and new_digest != old_digest:
                    raise ValueError("the transport reissue changed its request protocol")
                additional.append({"raw_request_sha256": sha, "prior_request_journal_sha256": old_digest,
                    "successor_response_journal_sha256": new_state["response_journal_shas"][0] if new_digest else None,
                    "additional_physical_attempts": int(new_digest is not None)})
        if new_state["state"] == "completed":
            runtime = runtime_for(corpus_root, offset, request)
            try:
                batch = runtime.run()
            finally:
                runtime.close()
            if batch.usage.physical_calls or batch.usage.checkpoint_hits != 1:
                raise ValueError("transport response verification must replay without calls")
    if len(additional) != p["maximum_reissued_raw_requests"]:
        raise ValueError("transport reissue population differs from the original inventory")
    return {"plan": plan, "stage": stage, "pending": pending, "completed_new": completed_new,
        "additional": additional, "preserved_completed_raw_responses": len(copied)}


def verify_transport_lineage(corpus_root, offset, repair_root):
    stage_path = corpus_root.resolve().parent / "stage.json"
    if not stage_path.exists() or read_sealed_json(stage_path).payload.get("format") != "memory-condense-spine-transport-recovery-stage-result-v2":
        return previous.verify_transport_lineage(corpus_root, offset, repair_root)
    state = stage_state(corpus_root, offset, complete=True)
    from tools.execute_spine_transport_recovery_v2 import execution_payload
    execution = read_sealed_json(stage_path.parent / "execution-raw-v2.json")
    if execution.payload != execution_payload(state):
        raise ValueError("transport executor protocol or call budget changed")
    marker = stage_path.parent / "execution-raw-v2.reserved"
    if not marker.is_file() or marker.read_text(encoding="utf-8") != execution.sha256 + "\n":
        raise ValueError("transport executor has no recorded release")
    starts = list((stage_path.parent / "readiness").glob("raw-v2-*.json"))
    if len(starts) != 1:
        raise ValueError("expected exactly one bounded transport release")
    release = read_sealed_json(starts[0]).payload
    from datetime import datetime
    from tools.execute_spine_transport_recovery import require_readiness
    readiness = require_readiness(Path(release["readiness_report_path"]), now=datetime.fromisoformat(release["started_utc"]))
    if (release["execution_preflight_sha256"] != execution.sha256 or
            release["readiness_report_sha256"] != readiness.sha256 or
            release["additional_calls_at_start"] != state["plan"].payload["maximum_new_provider_calls"]):
        raise ValueError("transport release changed its readiness or request population")
    return {"recovery_used": True, "stage_sha256": state["stage"].sha256,
        "stage_preflight_sha256": state["plan"].sha256,
        "preserved_completed_raw_responses": state["preserved_completed_raw_responses"],
        "additional_raw_attempts": len(state["additional"]), "raw_attempts": state["additional"],
        "first_attempt_raw_requests": state["plan"].payload["first_attempt_raw_requests"],
        "additional_compaction_attempts": 0, "compaction_attempt": None, "new_provider_calls": 0}
