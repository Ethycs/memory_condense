import pytest

from tests import test_native_spine_repair as original_fixture
from tests.test_native_spine_direct_repairs import SectionClient
from memory_condense.search.native_spine_batch import restore
from tools import repair_native_json_batches as repair
from tools.matched_eval.artifacts import read_sealed_json


def malformed_source(root, monkeypatch):
    monkeypatch.setattr(original_fixture, "original_response", lambda: '{"atoms":[{"label":"T0" "summary":"broken"}]}')
    return original_fixture.original_checkpoint(root, monkeypatch)


def test_json_recovery_preserves_original_failed_journals_and_exact_fragment_population(tmp_path, monkeypatch):
    source, root = tmp_path/"source", tmp_path/"recovery"
    malformed_source(source, monkeypatch)
    frozen = {p: p.read_bytes() for p in source.rglob("*.json")}
    plan = repair.prepare(source, root, [0])
    assert plan.payload["format"] == "native-spine-malformed-json-repair-v1"
    snapshot = read_sealed_json(root/"source-snapshot.json")
    assert snapshot.payload["originals"][0]["valid_atom_indices"] == []
    assert snapshot.payload["originals"][0]["invalid_atom_indices"] == [0, 1]
    fragments = tuple(f for binding in plan.payload["jobs"] for f in restore(read_sealed_json(root/binding["path"]).payload))
    assert fragments == original_fixture.inputs()
    client = SectionClient()
    monkeypatch.setattr(repair, "_completion_client", lambda *a: client)
    result = repair.execute(root, True)
    assert result.payload["complete_repair_snapshot"] and client.calls == 1
    admitted = read_sealed_json(root/"admitted-batches"/"000000.json")
    assert admitted.payload["json_repair_preflight_sha256"] == plan.sha256
    assert [a["pointer"] for a in admitted.payload["summaries"]] == [f.pointer() for f in fragments]
    monkeypatch.setattr(repair, "_completion_client", lambda *a: pytest.fail("replay called a provider"))
    assert repair.execute(root, False).sha256 == result.sha256
    assert all(p.read_bytes() == raw for p, raw in frozen.items())


def test_attributable_overlong_json_must_use_targeted_repair_instead(tmp_path, monkeypatch):
    source = tmp_path/"source"
    original_fixture.original_checkpoint(source, monkeypatch)
    with pytest.raises(ValueError, match="malformed-JSON"):
        repair.prepare(source, tmp_path/"wrong-recovery", [0])


def test_json_recovery_refinement_keeps_accepted_outputs_and_subdivides_only_rejected_pieces(tmp_path, monkeypatch):
    source, root, refined = tmp_path/"source", tmp_path/"recovery", tmp_path/"refined"
    malformed_source(source, monkeypatch)
    repair.prepare(source, root, [0])
    client = SectionClient(invalid_first=True)
    monkeypatch.setattr(repair, "_completion_client", lambda *a: client)
    first = repair.execute(root, True)
    assert not first.payload["complete_repair_snapshot"]
    next_plan = repair.prepare(source, refined, [0], previous=root)
    fragments = tuple(f for binding in next_plan.payload["jobs"] for f in restore(read_sealed_json(refined/binding["path"]).payload))
    assert "".join(f.text for f in fragments) == original_fixture.inputs()[0].text
    ready = read_sealed_json(refined/"source-snapshot.json").payload["ready"]
    assert len(ready) == 1
    monkeypatch.setattr(repair, "_completion_client", lambda *a: SectionClient())
    result = repair.execute(refined, True)
    assert result.payload["complete_repair_snapshot"]
    summaries = read_sealed_json(refined/"admitted-batches"/"000000.json").payload["summaries"]
    assert all(row["atom"] in summaries for row in ready)


def test_json_recovery_cannot_implicitly_retry_an_unacknowledged_call(tmp_path, monkeypatch):
    source, root = tmp_path/"source", tmp_path/"recovery"
    malformed_source(source, monkeypatch)
    repair.prepare(source, root, [0])
    client = SectionClient(fail=True)
    monkeypatch.setattr(repair, "_completion_client", lambda *a: client)
    with pytest.raises(ConnectionError):
        repair.execute(root, True)
    with pytest.raises(RuntimeError, match="refusing an unsafe retry"):
        repair.execute(root, True)
    assert client.calls == 1
