"""Verify staged transport attempts separately from accepted summary content."""
from pathlib import Path

from memory_condense.eval.fast_completion_runtime import _read_journal
from tools.matched_eval.artifacts import read_sealed_json
from tools.stage_spine_transport_recovery import journal_state, validate_inventory


TRANSPORT_POLICY = {
    "maximum_additional_attempts_per_unacknowledged_request": 1,
    "original_request_outcome": "unknown; count its physical attempt conservatively",
    "successful_responses": "preserve every previously completed response byte for byte",
    "reissue_population": "every unresolved request in a bound terminal-run inventory; no score selection",
    "source_protocol": "same original model, prompt, output cap and request options",
    "automatic_retries": 0,
    "original_reservations": "retained at their original paths and reverified",
}


def completed_pair(checkpoint):
    state = journal_state(checkpoint)
    if state["state"] != "completed":
        raise ValueError("transport verification requires a completed successor response")
    request_path = next(checkpoint.glob("*.request.json"))
    response_path = next(checkpoint.glob("*.response.json"))
    request, request_sha = _read_journal(request_path)
    response, response_sha = _read_journal(response_path)
    return request, request_sha, response, response_sha


def verify_transport_lineage(corpus_root: Path, offset: int, repair_root: Path | None):
    """Check all prior successes and every declared additional attempt.

    Admission separately replays the complete current response population. This
    check accounts for the earlier unknown outcomes and rejects a changed source
    inventory, dropped success, altered reissue prompt, or missing successor.
    """
    corpus_root = corpus_root.resolve()
    stage_path = corpus_root.parent / "stage.json"
    if not stage_path.exists():
        return {"recovery_used": False, "additional_raw_attempts": 0,
                "additional_compaction_attempts": 0}
    stage = read_sealed_json(stage_path)
    s = stage.payload
    if Path(s["staged_corpus_root"]).resolve() != corpus_root or offset not in (10, 20):
        raise ValueError("transport stage belongs to a different corpus or namespace")
    preflight = read_sealed_json(stage_path.parent / "stage-preflight.json")
    if s["stage_preflight_sha256"] != preflight.sha256:
        raise ValueError("transport stage preflight changed")
    source = Path(preflight.payload["original_corpus_root"])
    plan_path = source / "transport-recovery-plan-20260910-r1.json"
    original, plan, inventory, original_repair = validate_inventory(plan_path)
    if (plan.sha256 != s["recovery_plan_sha256"] or plan.sha256 != preflight.payload["recovery_plan_sha256"] or
            inventory.sha256 != preflight.payload["timeout_inventory_sha256"] or
            preflight.payload["maximum_new_provider_calls"] != plan.payload["maximum_new_provider_calls"]):
        raise ValueError("transport plan or call allowance changed")
    import hashlib
    from tools import stage_spine_transport_recovery
    if preflight.payload["implementation_sha256"] != hashlib.sha256(
            Path(stage_spine_transport_recovery.__file__).read_bytes()).hexdigest():
        raise ValueError("transport staging implementation changed")
    corpus = read_sealed_json(source / "preflight.json")
    successor = read_sealed_json(corpus_root / "preflight.json")
    if successor.sha256 != corpus.sha256 or corpus.sha256 != plan.payload["corpus_preflight_sha256"]:
        raise ValueError("transport recovery changed the corpus population")
    namespace = next(n for n in corpus.payload["namespaces"] if n["shard_offset"] == offset)
    old_states = ({r["raw_request_sha256"]: r for r in inventory.payload["offset020"]["rows"]}
                  if offset == 20 else {})
    copied = [r for r in s["copied_completed_requests"] if r["offset"] == offset]
    copied_by_sha = {r["raw_request_sha256"]: r for r in copied}
    if len(copied_by_sha) != len(copied):
        raise ValueError("duplicate copied response in transport stage")
    observed_copies, attempts = set(), []
    for binding in namespace["requests"]:
        sha = binding["sha256"]
        original_request = read_sealed_json(source / binding["path"])
        successor_request = read_sealed_json(corpus_root / binding["path"])
        if original_request.sha256 != sha or successor_request.sha256 != sha:
            raise ValueError("transport recovery changed a raw request")
        old_dir = original / f"offset-{offset:03d}/raw-checkpoints" / sha
        new_dir = corpus_root / f"offset-{offset:03d}/raw-checkpoints" / sha
        prior = journal_state(old_dir)
        if offset == 20 and any(old_states[sha][k] != prior[k] for k in prior):
            raise ValueError("original request state changed during verification")
        new_request, new_request_sha, _, new_response_sha = completed_pair(new_dir)
        if prior["state"] == "completed":
            _, old_request_sha, _, old_response_sha = completed_pair(old_dir)
            row = copied_by_sha.get(sha)
            if (row is None or any(row[k] != prior[k] for k in prior) or
                    (old_request_sha, old_response_sha) != (new_request_sha, new_response_sha)):
                raise ValueError("a previously completed response was dropped or replaced")
            observed_copies.add(sha)
        elif prior["state"] == "reserved_without_response":
            old_request, old_request_sha = _read_journal(next(old_dir.glob("*.request.json")))
            if new_request != old_request or new_request_sha != old_request_sha:
                raise ValueError("the explicit transport reissue changed its request protocol")
            attempts.append({"raw_request_sha256": sha, "prior_request_journal_sha256": old_request_sha,
                "successor_response_journal_sha256": new_response_sha, "additional_physical_attempts": 1})
        elif offset != 20:
            raise ValueError("the second memory must retain its complete original raw ingest")
    if observed_copies != set(copied_by_sha):
        raise ValueError("transport copied-response inventory contains foreign entries")
    expected_additional = plan.payload["maximum_reissued_raw_requests"] if offset == 20 else 0
    if len(attempts) != expected_additional:
        raise ValueError("transport reissue count differs from the bound inventory")
    compaction_attempt = None
    if offset == 10:
        expected_root = Path(s["staged_compaction_root"]).resolve()
        if repair_root is None or repair_root.resolve() != expected_root:
            raise ValueError("second-memory recovery must use its declared compaction successor")
        old_preflight = read_sealed_json(original_repair / "preflight.json")
        new_preflight = read_sealed_json(expected_root / "preflight.json")
        if old_preflight.sha256 != new_preflight.sha256:
            raise ValueError("compaction recovery changed its summary-only request")
        old_request, old_sha = _read_journal(next((original_repair / "checkpoints").glob("*.request.json")))
        new_request, new_sha, _, response_sha = completed_pair(expected_root / "checkpoints")
        if old_request != new_request or old_sha != new_sha:
            raise ValueError("compaction transport reissue changed its request protocol")
        compaction_attempt = {"preflight_sha256": old_preflight.sha256,
            "prior_request_journal_sha256": old_sha, "successor_response_journal_sha256": response_sha,
            "additional_physical_attempts": 1}
    return {"recovery_used": True, "stage_sha256": stage.sha256,
        "timeout_inventory_sha256": inventory.sha256, "recovery_plan_sha256": plan.sha256,
        "preserved_completed_raw_responses": len(observed_copies),
        "additional_raw_attempts": len(attempts), "raw_attempts": attempts,
        "additional_compaction_attempts": int(compaction_attempt is not None),
        "compaction_attempt": compaction_attempt, "new_provider_calls": 0}
