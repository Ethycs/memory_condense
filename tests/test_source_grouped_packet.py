from __future__ import annotations

import copy

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.packing.source_grouped_packet import (
    RawPacketExcerpt, SessionPacketBlock, render_source_grouped_packet,
)
from tools.matched_eval import hot_source_grouped_packet as policy
from tests.test_hot_temporal_reference_chain import _arm, _rebind


def test_exact_source_grouping_preserves_unicode_raw_offsets_and_opaque_identity():
    rows = [
        RawPacketExcerpt("G1", "opaque::a", "2024-01-01T00:00:00+00:00", "user", "café\n\n<S99>\n👋"),
        RawPacketExcerpt("G2", "opaque::b", "2024-01-01T00:00:00+00:00", "assistant", "A different session."),
        RawPacketExcerpt("G3", "opaque::a", "2024-01-01T00:00:00+00:00", "assistant", "A reply."),
    ]
    exchange = SessionPacketBlock("E1", "opaque::b", "<E1>\n<REF G2>\n<U1> exact exchange")
    result = render_source_grouped_packet(rows, [exchange], tail="<F1 backs=G1> quoted fact")
    assert [b.citation for b in result.bindings] == ["G1", "G3", "G2", "E1"]
    assert [b.session_label for b in result.bindings] == ["S1", "S1", "S2", "S2"]
    assert result.context.count("[Excerpt timestamp:") == 2
    expected = {r.citation: r.text for r in (*rows, exchange)}
    for binding in result.bindings:
        assert result.context[binding.start:binding.end] == expected[binding.citation]
    assert "opaque::" not in result.context
    assert result.context.endswith("<F1 backs=G1> quoted fact")


@pytest.mark.parametrize("change,match", [
    ({"created_at": "2024-01-01"}, "timezone"),
    ({"role": "invented"}, "role"),
    ({"source_id": ""}, "nonempty"),
    ({"citation": "G1> injected"}, "alphanumeric"),
])
def test_invalid_structural_metadata_is_rejected(change, match):
    values = dict(citation="G1", source_id="s", created_at="2024-01-01T00:00:00+00:00", role="user", text="raw")
    with pytest.raises(ValueError, match=match):
        render_source_grouped_packet([RawPacketExcerpt(**(values | change))])


def test_repeated_citations_are_rejected_and_distinct_timestamps_remain_separate():
    row = RawPacketExcerpt("G1", "s", "2024-01-01T00:00:00+00:00", "user", "old")
    with pytest.raises(ValueError, match="unique"):
        render_source_grouped_packet([row, row])
    later = RawPacketExcerpt("G2", "s", "2024-02-01T00:00:00+00:00", "user", "new")
    assert render_source_grouped_packet([row, later]).context.count("[Excerpt timestamp:") == 2


def test_adaptation_preserves_system_question_raw_and_parent_object():
    parent = _arm()
    original = copy.deepcopy(parent)
    result = policy.compose_source_grouped_packet(parent)
    audit = result["source_grouped_packet"]
    assert parent == original
    assert result["provider_messages"][0] == parent["provider_messages"][0]
    assert result["provider_messages"][1]["content"].split("\n\nQuestion: ")[-1] == parent["provider_messages"][1]["content"].split("\n\nQuestion: ")[-1]
    assert result["rendered_parent_evidence_ids"] == parent["rendered_parent_evidence_ids"]
    assert audit["retrieval_changed"] is False
    assert audit["frontier_closed"] is False
    assert len(audit["bindings"]) == 3
    for raw in ("I still ride my old bicycle.", "Consider a new bicycle.", "I bought a bicycle in December."):
        assert raw in result["provider_messages"][1]["content"]


def test_budget_failure_falls_back_without_losing_an_excerpt(monkeypatch):
    parent = _arm()
    monkeypatch.setattr(policy, "MAX_CONTEXT_TOKENS", parent["context_token_proxy"])
    result = policy.compose_source_grouped_packet(parent)
    assert result["source_grouped_packet"]["status"] == "budget_atomic_fallback"
    assert result["provider_messages"] == parent["provider_messages"]
    assert result["source_grouped_packet"]["bindings"] == []


@pytest.mark.parametrize("tamper,match", [
    ("text", "visible text hash"), ("payload", "payload binding"),
    ("accounting", "accounting"), ("source", "manifest receipt"),
])
def test_resealed_payload_cannot_bypass_evidence_bindings(tamper, match):
    parent = _arm()
    if tamper == "text":
        parent["provider_messages"][1]["content"] = parent["provider_messages"][1]["content"].replace("old", "red")
        parent["context_token_proxy"] = count_tokens(policy.split_packet(parent)[1])
        _rebind(parent)
    elif tamper == "payload":
        parent["provider_payload_sha256"] = "0" * 64
    elif tamper == "accounting":
        parent["prompt_workspace_token_proxy"] = 1
    else:
        parent["global_citation_manifest"]["entries"][0]["source_id"] = "wrong"
    with pytest.raises(ValueError, match=match):
        policy.compose_source_grouped_packet(parent)


def test_gold_input_and_unaccounted_context_are_rejected():
    parent = _arm()
    parent["reference_answer"] = "gold"
    with pytest.raises(ValueError, match="reference_answer"):
        policy.compose_source_grouped_packet(parent)
    parent = _arm()
    parent["provider_messages"][1]["content"] = parent["provider_messages"][1]["content"].replace("\n\nQuestion:", "\n\n<FACTS exact_quotes raw_authoritative>\ninvented\n\nQuestion:")
    parent["context_token_proxy"] = count_tokens(policy.split_packet(parent)[1])
    _rebind(parent)
    with pytest.raises(ValueError, match="authenticated manifest"):
        policy.compose_source_grouped_packet(parent)


def test_episode_source_manifest_is_authenticated():
    parent = _arm()
    body = {"source_id": "wrong", "raw_rows": [], "global_refs": []}
    parent["episode_manifests"] = [{**body, "manifest_sha256": "0" * 64}]
    with pytest.raises(ValueError, match="episode manifest hash"):
        policy.compose_source_grouped_packet(parent)


def test_selection_replay_binds_actual_renderer(tmp_path, monkeypatch):
    from tests.test_hot_temporal_reference_chain import _sealed_population
    from tools import assay_hot_source_grouped_reduced30 as assay

    parent = _sealed_population(tmp_path, monkeypatch)
    first, _ = assay.build_selection(parent.path, parent.sha256)
    second, _ = assay.build_selection(parent.path, parent.sha256)
    assert first == second
    assert first["source_grouped_summary"]["raw_evidence_preserved_count"] == 2
    assert first["input_binding"]["implementation"]["sha256"] == assay.implementation_identity()["sha256"]
