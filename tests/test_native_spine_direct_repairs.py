import json

import pytest

from memory_condense.domain._discourse_identity import canonical_json
from memory_condense.search.native_spine_batch import restore
from tools import repair_native_spine_sections as repair
from tools.matched_eval.artifacts import read_sealed_json
from tests.test_native_spine_repair import Client, original_checkpoint, original_response


class SectionClient(Client):
    def __init__(self, *, invalid_first=False, fail=False):
        super().__init__("", fail=fail)
        self.invalid_first = invalid_first

    def create(self, **kwargs):
        wire = json.loads(kwargs["messages"][1]["content"])
        fragments = [f for b in wire["transcripts"] for f in b["fragments"]]
        self.response = canonical_json({"atoms": [{"label": f["label"],
            "summary": "word " * 150 if self.invalid_first and i == 0 else "A source detail."}
            for i, f in enumerate(fragments)]})
        return super().create(**kwargs)


def test_direct_subdivision_preserves_valid_originals_and_replays_without_calls(tmp_path, monkeypatch):
    source, root = tmp_path/"source", tmp_path/"repair"
    original = original_checkpoint(source, monkeypatch)
    plan = repair.prepare(source, root, [0])
    snapshot = read_sealed_json(root/"source-snapshot.json")
    assert len(snapshot.payload["pieces"]) >= 2
    assert snapshot.payload["originals"][0]["valid_atom_indices"] == [1]
    client = SectionClient()
    monkeypatch.setattr(repair, "_completion_client", lambda *args: client)
    result = repair.execute(root, True)
    assert result.payload["complete_repair_snapshot"]
    assert client.calls == plan.payload["maximum_new_provider_calls"]
    admitted = read_sealed_json(root/"admitted-batches"/"000000.json").payload
    assert admitted["summaries"][-1]["summary"] == json.loads(original_response())["atoms"][1]["summary"]
    assert admitted["unchanged_valid_atom_indices"] == [1]
    assert read_sealed_json(original.path).sha256 == original.sha256
    def forbidden(*args):
        raise AssertionError("replay must not create a provider")
    monkeypatch.setattr(repair, "_completion_client", forbidden)
    assert repair.execute(root, False).sha256 == result.sha256


def test_refinement_reuses_good_sections_and_sends_only_the_still_invalid_raw_piece(tmp_path, monkeypatch):
    source, root, next_root = tmp_path/"source", tmp_path/"repair", tmp_path/"refined"
    original_checkpoint(source, monkeypatch)
    repair.prepare(source, root, [0])
    first_client = SectionClient(invalid_first=True)
    monkeypatch.setattr(repair, "_completion_client", lambda *args: first_client)
    first = repair.execute(root, True)
    assert not first.payload["complete_repair_snapshot"]
    first_job = read_sealed_json(root/"requests"/"000000.json")
    invalid_piece = restore(first_job.payload)[0]
    repair.prepare(source, next_root, [0], previous=root)
    next_plan = read_sealed_json(next_root/"preflight.json")
    pieces = tuple(f for binding in next_plan.payload["jobs"]
                   for f in restore(read_sealed_json(next_root/binding["path"]).payload))
    assert "".join(f.text for f in pieces) == invalid_piece.text
    saved = read_sealed_json(next_root/"source-snapshot.json").payload["ready"]
    assert len(saved) == len(first.payload["accepted_section_positions"])
    client = SectionClient()
    monkeypatch.setattr(repair, "_completion_client", lambda *args: client)
    result = repair.execute(next_root, True)
    assert result.payload["complete_repair_snapshot"]
    assert result.payload["kept_prior_sections"] == len(saved)
    merged = read_sealed_json(next_root/"admitted-batches"/"000000.json").payload["summaries"]
    assert all(r["atom"] in merged for r in saved)
    assert repair.execute(next_root, False).sha256 == result.sha256


def test_unanswered_transport_cannot_be_implicitly_reissued(tmp_path, monkeypatch):
    source, root = tmp_path/"source", tmp_path/"repair"
    original_checkpoint(source, monkeypatch)
    repair.prepare(source, root, [0])
    client = SectionClient(fail=True)
    monkeypatch.setattr(repair, "_completion_client", lambda *args: client)
    with pytest.raises(ConnectionError):
        repair.execute(root, True)
    with pytest.raises(RuntimeError, match="refusing an unsafe retry"):
        repair.execute(root, True)
    assert client.calls == 1 and not (root/"result.json").exists()


def test_complete_repairs_cannot_create_another_refinement_wave(tmp_path, monkeypatch):
    source, root = tmp_path/"source", tmp_path/"repair"
    original_checkpoint(source, monkeypatch)
    repair.prepare(source, root, [0])
    client = SectionClient()
    monkeypatch.setattr(repair, "_completion_client", lambda *args: client)
    repair.execute(root, True)
    with pytest.raises(ValueError, match="needs no further subdivision"):
        repair.prepare(source, tmp_path/"unneeded", [0], previous=root)
    assert not (tmp_path/"unneeded"/"preflight.json").exists()


def test_duplicate_or_out_of_range_original_population_is_rejected_before_provider_use(tmp_path, monkeypatch):
    source = tmp_path/"source"
    original_checkpoint(source, monkeypatch)
    for ordinals in ([0, 0], [1], [], [-1]):
        with pytest.raises(ValueError, match="explicit rejected ordinals"):
            repair.prepare(source, tmp_path/"invalid", ordinals)
    assert not (tmp_path/"invalid"/"preflight.json").exists()
