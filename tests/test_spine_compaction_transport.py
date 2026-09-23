import hashlib
import json

import pytest

from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime, _read_journal
from tools import admit_spine_corpus_v6 as admission
from tools import finish_spine_compaction_batches as completion
from tools import repair_spine_summary_budget_v4 as parent
from tools import spine_compaction_transport as transport
from tools import verify_spine_admission_method_v9 as verification
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json
from tools.matched_eval.contracts import canonical_json_bytes
from tools.stage_spine_transport_recovery import copy_exact, journal_state
from tests import test_spine_compaction_batch_completion as fixtures
from tests import test_spine_summary_budget_multibatch as batches
from tests.test_spine_admission_method_verification import Client


def fixture(tmp_path, monkeypatch):
    corpus, original_root, original, second, _ = fixtures.fixture(tmp_path, monkeypatch, invalid=False)
    class UnknownClient(Client):
        def create(self, **kwargs):
            raise RuntimeError("terminal connection failure")
    runtime = FastCompletionRuntime(client=UnknownClient(""), **batches.runtime_kwargs(original_root, original, second))
    try:
        with pytest.raises(RuntimeError, match="terminal connection failure"):
            runtime.run()
    finally:
        runtime.close()
    root = tmp_path / "transport"
    successor = root / "summary-repair"
    publish_sealed_json(successor / "preflight.json", original.payload)
    first = original.payload["batches"][0]
    for path in (original_root / "checkpoints" / first["batch_sha256"]).glob("*.json"):
        copy_exact(path, successor / "checkpoints" / first["batch_sha256"] / path.name)
    execution = completion.prepare(root / "execution", successor)
    old_dir = original_root / "checkpoints" / second["batch_sha256"]
    _, request_sha = _read_journal(next(old_dir.glob("*.request.json")))
    publish_sealed_json(root / "stage.json", {
        "format": transport.FORMAT, "maximum_additional_attempts": 1, "automatic_retries": 0,
        "raw_inputs_to_qwen": False, "original_reservation_preserved": True,
        "original_attempt_counted_conservatively": True,
        "original_root": str(original_root.resolve()), "successor_root": str(successor.resolve()),
        "original_preflight_sha256": original.sha256, "original_batch_sha256": second["batch_sha256"],
        "original_state": journal_state(old_dir), "original_request_journal_sha256": request_sha,
        "execution_preflight_sha256": execution.sha256})
    captured = []
    class CapturingClient(Client):
        def create(self, **kwargs):
            captured.append(kwargs["messages"])
            return super().create(**kwargs)
    monkeypatch.setattr(completion, "_completion_client", lambda *args: CapturingClient(batches.output(second)))
    completion.run(root / "execution", True)
    assert captured == [second["messages"]] and "RAW_CANARY" not in json.dumps(captured)
    monkeypatch.setattr(completion, "_completion_client", lambda *args: pytest.fail("transport verifier called a provider"))
    parent.run(successor, False)
    admission.admit(corpus, 0, 1, successor)
    return corpus / "offset-000/source-bound-atoms-prefix-0001.json", successor, root, original_root, original


def test_all_originals_preserved_and_additional_attempt_counted_with_common_method(tmp_path, monkeypatch):
    atoms, successor, stage, original_root, _ = fixture(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in original_root.rglob("*.json")}
    atoms_before = atoms.read_bytes()
    result = verification.verify(atoms, successor, stage)
    assert result.payload["compaction_provider_attempts"] == 3
    assert result.payload["compaction_only_transport"]["additional_compaction_attempts"] == 1
    assert len(result.payload["compaction_only_transport"]["preserved_response_journal_shas"]) == 1
    assert verification.load_verified_method(atoms, read_sealed_json(atoms))[1] == result.sha256
    assert atoms.read_bytes() == atoms_before
    assert all(p.read_bytes() == content for p, content in before.items())
    corpus, normal, preflight, second, _ = fixtures.fixture(tmp_path / "normal", monkeypatch, invalid=False)
    batches.record(batches.output(second), **batches.runtime_kwargs(normal, preflight, second))
    parent.run(normal, False)
    admission.admit(corpus, 0, 1, normal)
    regular = verification.verify(corpus / "offset-000/source-bound-atoms-prefix-0001.json", normal)
    assert regular.payload["method_sha256"] == result.payload["method_sha256"]
    assert regular.payload["compaction_provider_attempts"] == 2


@pytest.mark.parametrize("change", ["omitted_stage", "wrong_stage", "missing_original", "pending_successor",
    "dropped_preserved_success", "extra_attempt", "changed_original_request"])
def test_changed_or_incomplete_transport_cannot_enter_the_method(tmp_path, monkeypatch, change):
    atoms, successor, stage, original_root, original = fixture(tmp_path, monkeypatch)
    first, second = original.payload["batches"]
    argument = stage
    if change == "omitted_stage":
        argument = None
    elif change == "wrong_stage":
        argument = tmp_path / "foreign"
    elif change == "missing_original":
        next((original_root / "checkpoints" / second["batch_sha256"]).glob("*.request.json")).unlink()
    elif change == "pending_successor":
        next((successor / "checkpoints" / second["batch_sha256"]).glob("*.response.json")).unlink()
    elif change == "dropped_preserved_success":
        next((original_root / "checkpoints" / first["batch_sha256"]).glob("*.response.json")).unlink()
    elif change == "extra_attempt":
        path = stage / "stage.json"
        p = read_sealed_json(path).payload
        raw = canonical_json_bytes({**p, "maximum_additional_attempts": 2})
        path.write_bytes(raw)
        path.with_name(path.name + ".sha256").write_text(hashlib.sha256(raw).hexdigest() + "  " + path.name + "\n")
    else:
        path = next((original_root / "checkpoints" / second["batch_sha256"]).glob("*.request.json"))
        path.write_text(path.read_text().replace("qwen3-8b", "other-model"))
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        verification.verify(atoms, successor, argument)
    assert not atoms.with_name("conditional-method-v9-prefix-0001.json").exists()
