from dataclasses import replace

import pytest

from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from memory_condense.search.episodes.user_spine_hierarchy import build_user_spine_hierarchy
from memory_condense.search.spine_parent_hierarchy import SourceSpineParentPlan
from tests.test_attention_summary_sections import SummaryLinker
from tests.test_user_spine_hierarchy import make_exchanges, Summarizer
from tools.spine_leaf_projection import project_source_leaves


def fixture():
    _, _, _, exchanges = make_exchanges()
    scorer = QwenAttentionHeadSurpriseScorer(SummaryLinker(), max_spans=8, span_token_cap=128)
    leaves, cuts, _ = project_source_leaves(exchanges, scorer=scorer,
        summarize=Summarizer(), summarizer_identity="fixture")
    spans = tuple(s for e in exchanges for s in e.section.spans)
    return exchanges, leaves, cuts, spans


def test_restores_full_attention_tree_without_changing_leaves_or_reading_raw():
    exchanges, leaves, cuts, spans = fixture()
    plan = SourceSpineParentPlan(leaves[::-1], spans, cuts)
    seen = []
    def summarize(request):
        seen.append(request)
        assert "RAW_CANARY" not in str(request.messages)
        if request.kind == "user_spine":
            assert all(f.role == "user_summary" for f in request.fragments)
        return Summarizer()(request)
    result = plan.compile(summarize=summarize, summarizer_identity="fixture")
    full = build_user_spine_hierarchy(exchanges,
        scorer=QwenAttentionHeadSurpriseScorer(SummaryLinker(), max_spans=8, span_token_cap=128),
        summarize=Summarizer(), summarizer_identity="fixture", max_channel_tokens=128,
        max_leaf_exchanges=2, window_exchange_cap=8)
    assert result.to_json() == full.summary_index().to_json()
    assert tuple(s for s in result.sections if not s.child_section_ids) == tuple(sorted(leaves, key=lambda s: s.section_id))
    assert plan.root_section_id == full.root_section_ids[0]
    assert len(plan.parents) == len(cuts) > 0
    assert seen


@pytest.mark.parametrize("change", ["missing_cut", "extra_cut", "bad_cut", "missing_span", "duplicate_leaf", "reversed_spans"])
def test_rejects_changed_topology_before_any_summary_call(change):
    _, leaves, cuts, spans = fixture()
    if change == "missing_cut":
        cuts = cuts[:-1]
    elif change == "extra_cut":
        cuts = (*cuts, {**cuts[0], "section_id": "foreign-parent"})
    elif change == "bad_cut":
        cuts = (*cuts[:-1], {**cuts[-1], "split_atom": 10000})
    elif change == "missing_span":
        spans = spans[:-1]
    elif change == "duplicate_leaf":
        leaves = (*leaves, leaves[0])
    else:
        spans = spans[::-1]
    with pytest.raises(ValueError):
        SourceSpineParentPlan(leaves, spans, cuts)


def test_rejects_untyped_or_extra_raw_summary_channels():
    _, leaves, cuts, spans = fixture()
    import json
    body = json.loads(leaves[0].summary)
    body["raw_support_quote"] = "RAW_CANARY"
    changed = replace(leaves[0], summary=json.dumps(body), receipt_sha256="")
    with pytest.raises(ValueError, match="explicit user-spine"):
        SourceSpineParentPlan((changed, *leaves[1:]), spans, cuts)
