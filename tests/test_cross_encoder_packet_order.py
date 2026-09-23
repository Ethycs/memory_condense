import pytest

from memory_condense.domain._tokenizer import truncate_to_tokens
from memory_condense.domain.discourse import quote_sha256
from tools import assay_hot_cross_encoder_order_reduced30 as assay
from tools.matched_eval.artifacts import publish_sealed_json
from tests.test_hot_temporal_reference_chain import _sealed_population


def _fixture(tmp_path, monkeypatch, *, corrupt=False):
    parent = _sealed_population(tmp_path, monkeypatch)
    monkeypatch.setattr(assay.dense, "DEFAULT_PARENT", parent.path)
    monkeypatch.setattr(assay.dense, "DEFAULT_PARENT_SHA256", parent.sha256)
    _, items = assay.dense._inputs()
    questions = []
    for row, _, _, _, _, query, raw in items:
        scores = [{"evidence_id": g["evidence_id"], "raw_text_sha256": g["raw_text_sha256"],
                   "scored_text_sha256": quote_sha256(truncate_to_tokens(g["text"], 320)), "score": i}
                  for i, g in enumerate(raw)]
        questions.append({"global_ordinal": row["global_ordinal"], "query_sha256": quote_sha256(query), "scores": scores})
    if corrupt:
        questions[0]["scores"][0]["raw_text_sha256"] = "0" * 64
    publish_sealed_json(tmp_path / "scores.json", {"parent_selection_sha256": parent.sha256,
        "implementation": assay.implementation(), "questions": questions})
    return items


def test_order_only_replay_preserves_all_raw_and_the_original_prompt_policy(tmp_path, monkeypatch):
    items = _fixture(tmp_path, monkeypatch)
    first = assay.build_selection(tmp_path)
    assert first == assay.build_selection(tmp_path)
    for row, parent in zip(first["questions"], items, strict=True):
        _, arm = assay.harness.find_provider_arm(row["source_row"], row["telemetry"]["arm_path"])
        assert arm["cross_encoder_packet_order"]["ordered_citations"] == ["G3", "G2", "G1"]
        assert arm["provider_messages"][0] == parent[1]["provider_messages"][0]
        assert arm["rendered_parent_evidence_ids"] == parent[1]["rendered_parent_evidence_ids"]
        assert arm["context_token_proxy"] == parent[1]["context_token_proxy"]
        for raw in parent[-1]:
            assert arm["provider_messages"][1]["content"].count(raw["text"]) == 1


def test_outer_seal_cannot_hide_scores_bound_to_different_raw_text(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch, corrupt=True)
    with pytest.raises(ValueError, match="raw-evidence binding"):
        assay.build_selection(tmp_path)
