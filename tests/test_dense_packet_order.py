from __future__ import annotations

import numpy as np
import pytest

from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from tools import assay_hot_dense_order_reduced30 as assay
from tools.matched_eval.artifacts import publish_sealed_json
from tests.test_hot_temporal_reference_chain import _sealed_population


def _fixture(tmp_path, monkeypatch):
    parent = _sealed_population(tmp_path, monkeypatch)
    monkeypatch.setattr(assay, "DEFAULT_PARENT", parent.path)
    monkeypatch.setattr(assay, "DEFAULT_PARENT_SHA256", parent.sha256)
    _, inputs = assay._inputs()
    rows = inputs[0][-1]
    # Prefer G3, then G2, then G1, regardless of upstream order.
    raw = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    queries = np.array([[-1, 0], [-1, 0]], dtype=np.float32)
    np.save(tmp_path / "raw-vectors.npy", raw, allow_pickle=False)
    np.save(tmp_path / "query-vectors.npy", queries, allow_pickle=False)
    publish_sealed_json(tmp_path / "addresses.json", {
        "parent_selection_sha256": parent.sha256, "implementation": assay._implementation(),
        "raw_text_sha256s": [row["raw_text_sha256"] for row in rows],
        "query_sha256s": [quote_sha256(item[5]) for item in inputs],
        "raw_vectors_sha256": file_sha256(tmp_path / "raw-vectors.npy"),
        "query_vectors_sha256": file_sha256(tmp_path / "query-vectors.npy"),
    })
    return parent, inputs


def test_dense_order_replays_and_changes_only_complete_global_block_order(tmp_path, monkeypatch):
    _, inputs = _fixture(tmp_path, monkeypatch)
    first, changed = assay.build_selection(tmp_path)
    second, _ = assay.build_selection(tmp_path)
    assert first == second
    assert changed == 2
    for candidate, source in zip(first["questions"], inputs, strict=True):
        _, arm = assay.harness.find_provider_arm(candidate["source_row"], candidate["telemetry"]["arm_path"])
        original = source[1]
        assert arm["provider_messages"][0] == original["provider_messages"][0]
        assert [row["citation"] for row in arm["dense_packet_order"]["ordered_rows"]] == ["G3", "G2", "G1"]
        assert arm["rendered_parent_evidence_ids"] == original["rendered_parent_evidence_ids"]
        for raw in source[-1]:
            assert arm["provider_messages"][1]["content"].count(raw["text"]) == 1
        assert arm["provider_messages"][1]["content"].split("\n\nQuestion: ")[-1] == original["provider_messages"][1]["content"].split("\n\nQuestion: ")[-1]
        assert arm["dense_packet_order"]["frontier_closed"] is False


@pytest.mark.parametrize("name", ["raw", "query"])
def test_changed_vectors_are_rejected_before_ordering(tmp_path, monkeypatch, name):
    _fixture(tmp_path, monkeypatch)
    np.save(tmp_path / f"{name}-vectors.npy", np.ones((3, 2), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="vector file changed"):
        assay.build_selection(tmp_path)


def test_changed_parent_is_rejected_before_reading_vectors(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(assay, "DEFAULT_PARENT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="conventional parent changed"):
        assay.build_selection(tmp_path)
