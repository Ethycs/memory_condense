import hashlib
from pathlib import Path

import pytest

from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime, _canonical_bytes, _read_journal, _sealed
from tools import spine_transport_lineage as lineage
from tools import stage_spine_transport_recovery as staging
from tools.matched_eval.artifacts import publish_sealed_json
from tests.test_spine_transport_recovery_stage import Client


def record(checkpoint, messages, model, provenance, *, fail=False, max_new_tokens=3072):
    runtime = FastCompletionRuntime(checkpoint_dir=checkpoint, prompt_population=[messages],
        model=model, client=Client(fail=fail), max_prompt_tokens=7000,
        max_new_tokens=max_new_tokens, max_concurrency=1, retries=0, benchmark_provenance=provenance)
    try:
        if fail:
            with pytest.raises(TimeoutError):
                runtime.run()
        else:
            runtime.run()
    finally:
        runtime.close()


def fixture(tmp_path, monkeypatch, offset=20):
    original = tmp_path / "original"
    stage_root = tmp_path / "stage"
    successor = stage_root / "corpus"
    original_repair = original / "summary-repair"
    successor_repair = stage_root / "summary-repair-offset010"
    bindings, states, copied = [], [], []
    for i in range(3 if offset == 20 else 1):
        relative = f"offset-{offset:03d}/requests/{i:04d}.json"
        request, _ = publish_sealed_json(original / relative, {
            "model": "codex_sdk/gpt-5.6-terra", "messages": [{"role": "user", "content": f"Raw fragment {i}."}]})
        publish_sealed_json(successor / relative, request.payload)
        sha = request.sha256
        bindings.append({"path": relative, "sha256": sha})
        old = original / f"offset-{offset:03d}/raw-checkpoints" / sha
        new = successor / f"offset-{offset:03d}/raw-checkpoints" / sha
        if i < 2:
            record(old, request.payload["messages"], request.payload["model"],
                {"raw_request_sha256": sha}, fail=i == 1)
        state = staging.journal_state(old)
        states.append({"raw_request_sha256": sha, **state})
        if i == 0:
            files = staging.copy_completed_checkpoint(old, new, request)
            copied.append({"offset": offset, "raw_request_sha256": sha, "files": files, **state})
        else:
            record(new, request.payload["messages"], request.payload["model"], {"raw_request_sha256": sha})
    corpus, _ = publish_sealed_json(original / "preflight.json", {
        "namespaces": [{"shard_offset": offset, "requests": bindings}]})
    publish_sealed_json(successor / "preflight.json", corpus.payload)
    plan, _ = publish_sealed_json(original / "transport-recovery-plan-20260910-r1.json", {
        "maximum_new_provider_calls": 3, "maximum_reissued_raw_requests": int(offset == 20),
        "corpus_preflight_sha256": corpus.sha256})
    inventory, _ = publish_sealed_json(original / "inventory.json", {"offset020": {"rows": states}})
    preflight, _ = publish_sealed_json(stage_root / "stage-preflight.json", {
        "original_corpus_root": str(original), "recovery_plan_sha256": plan.sha256,
        "timeout_inventory_sha256": inventory.sha256, "maximum_new_provider_calls": 3,
        "implementation_sha256": hashlib.sha256(Path(staging.__file__).read_bytes()).hexdigest()})
    publish_sealed_json(stage_root / "stage.json", {
        "staged_corpus_root": str(successor), "staged_compaction_root": str(successor_repair),
        "stage_preflight_sha256": preflight.sha256, "recovery_plan_sha256": plan.sha256,
        "copied_completed_requests": copied})
    monkeypatch.setattr(lineage, "validate_inventory", lambda _: (original, plan, inventory, original_repair))
    if offset == 10:
        messages = [{"role": "user", "content": "Synthetic summary: User prefers concise explanations."}]
        repair, _ = publish_sealed_json(original_repair / "preflight.json", {"messages": messages})
        publish_sealed_json(successor_repair / "preflight.json", repair.payload)
        for directory, fail in ((original_repair, True), (successor_repair, False)):
            record(directory / "checkpoints", messages, "qwen3-8b",
                {"preflight_sha256": repair.sha256}, fail=fail, max_new_tokens=2048)
    return original, successor, successor_repair, bindings


def test_raw_recovery_counts_prior_unknown_attempt_and_preserves_success(tmp_path, monkeypatch):
    original, successor, _, bindings = fixture(tmp_path, monkeypatch)
    result = lineage.verify_transport_lineage(successor, 20, None)
    assert result["additional_raw_attempts"] == 1
    assert result["preserved_completed_raw_responses"] == 1
    assert result["additional_compaction_attempts"] == result["new_provider_calls"] == 0
    assert staging.journal_state(original / "offset-020/raw-checkpoints" / bindings[1]["sha256"])["state"] == "reserved_without_response"


def test_compaction_recovery_counts_both_attempts_without_removing_original(tmp_path, monkeypatch):
    original, successor, repair, _ = fixture(tmp_path, monkeypatch, offset=10)
    result = lineage.verify_transport_lineage(successor, 10, repair)
    assert result["additional_compaction_attempts"] == 1
    assert result["additional_raw_attempts"] == result["new_provider_calls"] == 0
    assert staging.journal_state(original / "summary-repair/checkpoints")["state"] == "reserved_without_response"


def test_missing_successor_compaction_is_not_certified(tmp_path, monkeypatch):
    _, successor, _, _ = fixture(tmp_path, monkeypatch, offset=10)
    with pytest.raises(ValueError, match="declared compaction successor"):
        lineage.verify_transport_lineage(successor, 10, None)


def test_resealed_replacement_of_prior_success_is_rejected(tmp_path, monkeypatch):
    _, successor, _, bindings = fixture(tmp_path, monkeypatch)
    path = next((successor / "offset-020/raw-checkpoints" / bindings[0]["sha256"]).glob("*.response.json"))
    body, _ = _read_journal(path)
    body.pop("journal_sha256")
    body["completion"] = "A replacement result."
    path.write_bytes(_canonical_bytes(_sealed(body)))
    with pytest.raises(ValueError, match="dropped or replaced"):
        lineage.verify_transport_lineage(successor, 20, None)


def test_foreign_raw_request_after_staging_is_rejected(tmp_path, monkeypatch):
    _, successor, _, bindings = fixture(tmp_path, monkeypatch)
    path = successor / bindings[1]["path"]
    from tools.matched_eval.contracts import canonical_json_bytes
    body = {"model": "other", "messages": []}
    encoded = canonical_json_bytes(body)
    path.write_bytes(encoded)
    path.with_name(path.name + ".sha256").write_bytes((hashlib.sha256(encoded).hexdigest() + "  " + path.name + "\n").encode("ascii"))
    with pytest.raises(ValueError, match="changed a raw request"):
        lineage.verify_transport_lineage(successor, 20, None)


def test_unstaged_memory_records_no_additional_transport_attempts(tmp_path):
    assert lineage.verify_transport_lineage(tmp_path / "original", 0, None) == {
        "recovery_used": False, "additional_raw_attempts": 0, "additional_compaction_attempts": 0}
