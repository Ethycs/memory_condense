from dataclasses import replace
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain.schemas import Turn
from memory_condense.search.episodes.attention_hierarchy import compile_attention_atoms
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from memory_condense.search.episodes.user_spine_hierarchy import (
    build_user_spine_hierarchy, compile_user_spine_exchanges,
)
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.spine_summary import (
    QwenSpineSummarizer, SpineSummaryFragment, SpineSummaryRequest, parse_spine_summary,
)
from tests.test_attention_summary_sections import SummaryLinker


def records():
    return [Turn(turn_id=f"private-{i}", source_id="private-source", role=role,
                 text=f"  RAW_CANARY_{i} τ\r\n", created_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
            for i, role in enumerate(("system", "user", "assistant", "user", "assistant", "user", "assistant"))]


class Summarizer:
    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        # Test double only. Preserve topics so routing can test exact hydration.
        return " ".join(dict.fromkeys(f.summary for f in request.fragments))


def make_exchanges(turns=None):
    turns = records() if turns is None else turns
    summaries = iter(("prelude", "orchard harvest", "machine suggestion", "orchard irrigation",
                      "machine claim", "observatory reservations", "machine answer"))
    atoms = compile_attention_atoms(turns, summarize_raw=lambda _: next(summaries),
                                    summarizer_identity="non-qwen-fixture", atom_token_cap=32)
    summarizer = Summarizer()
    exchanges = compile_user_spine_exchanges(atoms, summarize=summarizer, summarizer_identity="summary-fixture")
    return turns, atoms, summarizer, exchanges


def build(exchanges, summarize, **kwargs):
    linker = SummaryLinker()
    hierarchy = build_user_spine_hierarchy(exchanges,
        scorer=QwenAttentionHeadSurpriseScorer(linker, max_spans=3, span_token_cap=64),
        summarize=summarize, summarizer_identity="summary-fixture", window_exchange_cap=3,
        max_leaf_exchanges=1, **kwargs)
    return linker, hierarchy


def test_user_leads_own_responses_and_prelude_stays_unowned():
    turns, atoms, summarizer, exchanges = make_exchanges()
    assert [e.lead_turn_id for e in exchanges] == [None, "private-1", "private-3", "private-5"]
    assert [len(e.section.spans) for e in exchanges] == [1, 2, 2, 2]
    assert tuple(s for e in exchanges for s in e.section.spans) == tuple(a.spans[0] for a in atoms)
    for request in summarizer.requests:
        if request.kind == "user_spine":
            assert all(f.role == "user" for f in request.fragments)
            assert "machine" not in json.dumps(request.messages)
        elif request.user_spine is not None:
            assert request.user_spine in {"orchard harvest", "orchard irrigation", "observatory reservations"}
    assert "RAW_CANARY" not in json.dumps([r.messages for r in summarizer.requests])
    assert "private-source" not in json.dumps([r.messages for r in summarizer.requests])


def test_attention_uses_only_user_channel_and_splits_between_whole_exchanges():
    turns, atoms, summarizer, exchanges = make_exchanges()
    linker, hierarchy = build(exchanges, summarizer)
    observed = [text for _, texts in linker.inputs for text in texts]
    assert set(observed) == {"Unowned prelude.", "orchard harvest", "orchard irrigation", "observatory reservations"}
    assert [(w.atom_start, w.atom_end) for w in hierarchy.windows] == [(0, 3), (2, 4)]
    assert all("machine" not in text and "RAW_CANARY" not in text for text in observed)
    leaves = [s for s in hierarchy.sections if not s.child_section_ids]
    assert [s.spans for s in leaves] == [e.section.spans for e in exchanges]
    assert len(hierarchy.splits) == len(exchanges) - 1
    assert all("exchange_boundary" in s.reason for s in hierarchy.splits)
    root_split = next(s for s in hierarchy.splits if s.section_id == hierarchy.root_section_ids[0])
    assert root_split.split_atom == 1 and root_split.attention_change == 1.0
    # Two sources of identical shape but changed assistant topics keep the cuts.
    changed = tuple(replace(e, section=replace(e.section,
        summary=e.section.summary.replace("machine", "observatory"), receipt_sha256=""),
        attached_context=e.attached_context.replace("machine", "observatory") if e.attached_context else None,
        receipt_sha256="") for e in exchanges)
    _, second = build(changed, Summarizer())
    assert [(s.split_atom, s.attention_change) for s in hierarchy.splits] == [
        (s.split_atom, s.attention_change) for s in second.splits]


def test_summary_index_roundtrip_routes_and_hydrates_complete_user_exchange():
    turns, _, summarizer, exchanges = make_exchanges()
    _, hierarchy = build(exchanges, summarizer)
    index = SectionSummaryIndex.from_json(hierarchy.summary_index().to_json())
    plan = index.route("observatory reservations", max_sections=1)
    reads = []
    by_id = {t.turn_id: t for t in turns}
    def load(identity):
        reads.append(identity)
        return by_id[identity]
    result = hydrate_section_plan(plan, load_turn=load)
    assert reads == ["private-5", "private-6"]
    assert not result.requires_raw_fallback
    assert [row.text for row in result.sections[0].evidence] == [t.text for t in turns[-2:]]
    assert "observatory reservations" not in result.render_context()
    assert all(t.text in result.render_context() for t in turns[-2:])


def test_oversized_exchange_is_preserved_and_atomically_rejected_at_hydration():
    turns, _, summarizer, exchanges = make_exchanges()
    _, hierarchy = build(exchanges, summarizer, leaf_token_cap=1)
    assert hierarchy.oversized_exchange_ids == tuple(e.section.section_id for e in exchanges)
    plan = hierarchy.summary_index().route("observatory reservations", max_sections=1)
    reads = []
    result = hydrate_section_plan(plan, load_turn=lambda tid: reads.append(tid), max_context_tokens=1)
    assert reads == [] and result.requires_raw_fallback
    assert result.diagnostics[0].reason == "context_budget"


def test_long_user_turn_fragments_remain_one_lead_with_exact_unicode_coverage():
    turn = Turn(turn_id="long", source_id="source", role="user", text="long statement τ " * 30,
                created_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
    atoms = compile_attention_atoms([turn], summarize_raw=lambda _: "orchard", summarizer_identity="non-qwen", atom_token_cap=16)
    summarizer = Summarizer()
    exchanges = compile_user_spine_exchanges(atoms, summarize=summarizer, summarizer_identity="summary")
    assert len(exchanges) == 1 and exchanges[0].lead_turn_id == "long"
    assert len(exchanges[0].section.spans) == len(atoms) > 1
    assert all(len(r.fragments) <= 2 for r in summarizer.requests)


@pytest.mark.parametrize("bad", ["raw", records()[0]])
def test_raw_objects_rejected_before_summary_or_attention_calls(bad):
    calls = []
    with pytest.raises(TypeError, match="summary descriptors"):
        compile_user_spine_exchanges([bad], summarize=calls.append, summarizer_identity="test")
    assert calls == []


def test_malformed_population_rejected_before_model_work():
    _, atoms, _, _ = make_exchanges()
    calls = []
    with pytest.raises(ValueError, match="duplicate"):
        compile_user_spine_exchanges([*atoms, atoms[0]], summarize=calls.append, summarizer_identity="test")
    with pytest.raises(ValueError, match="revisit"):
        compile_user_spine_exchanges([atoms[1], atoms[3], replace(atoms[1], section_id="repeated", receipt_sha256="")],
                                    summarize=calls.append, summarizer_identity="test")
    assert calls == []


def test_separate_sources_never_share_an_exchange_or_attention_window():
    turns, atoms, _, _ = make_exchanges()
    second = [t.model_copy(update={"turn_id": "second-" + t.turn_id, "source_id": "second-source"}) for t in turns]
    _, second_atoms, _, _ = make_exchanges(second)
    summarizer = Summarizer()
    exchanges = compile_user_spine_exchanges([*atoms, *second_atoms], summarize=summarizer, summarizer_identity="summary")
    _, hierarchy = build(exchanges, summarizer)
    assert len(hierarchy.root_section_ids) == 2
    assert {w.source_id for w in hierarchy.windows} == {"private-source", "second-source"}
    assert all(len({s.source_id for s in node.spans}) == 1 for node in hierarchy.sections)
    for request in summarizer.requests:
        if request.kind == "user_spine":
            assert all(f.role in {"user", "user_summary"} for f in request.fragments)
            assert "machine" not in " ".join(f.summary for f in request.fragments)


def test_stale_response_rejects_whole_exchange_including_valid_user_lead():
    turns, _, summarizer, exchanges = make_exchanges()
    _, hierarchy = build(exchanges, summarizer)
    plan = hierarchy.summary_index().route("observatory reservations", max_sections=1)
    by_id = {t.turn_id: t for t in turns}
    by_id["private-6"] = by_id["private-6"].model_copy(update={"text": "changed assistant text"})
    result = hydrate_section_plan(plan, load_turn=by_id.get)
    assert result.sections == () and result.requires_raw_fallback
    assert result.diagnostics[0].reason == "raw_turn_identity_changed"


def test_channel_overflow_is_not_silently_truncated():
    _, atoms, _, _ = make_exchanges()
    with pytest.raises(ValueError, match="output budget"):
        compile_user_spine_exchanges(atoms, summarize=lambda _: "oversized " * 100,
                                    summarizer_identity="bad", max_channel_tokens=8)


def test_scorer_truncation_is_rejected_before_model_work():
    _, _, summarizer, exchanges = make_exchanges()
    linker = SummaryLinker()
    with pytest.raises(ValueError, match="truncated"):
        build_user_spine_hierarchy(exchanges, summarize=summarizer, summarizer_identity="summary",
            scorer=QwenAttentionHeadSurpriseScorer(linker, max_spans=8, span_token_cap=8),
            max_channel_tokens=64, window_exchange_cap=8)
    assert linker.inputs == []


def test_response_cannot_be_promoted_to_spine_and_budget_is_fail_closed():
    with pytest.raises(ValueError, match="machine material"):
        SpineSummaryRequest("user_spine", (SpineSummaryFragment("assistant", "2026-09-09", "suggestion"),))
    request = SpineSummaryRequest("user_spine", (SpineSummaryFragment("user", "2026-09-09", "orchard"),), max_output_tokens=2)
    for invalid in ('{"summary": "too many words in this response"}', '{"summary":"ok","answer":"invented"}', 'not JSON'):
        with pytest.raises(ValueError):
            parse_spine_summary(invalid, request)
    with pytest.raises(ValueError, match="prompt budget"):
        replace(request, max_prompt_tokens=1)


def test_qwen_adapter_sends_only_summary_requests_and_disables_retries():
    calls = []
    class Client:
        def with_options(self, **kwargs):
            assert kwargs == {"max_retries": 0}
            return self
        chat = property(lambda self: SimpleNamespace(completions=SimpleNamespace(create=self.complete)))
        def complete(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                message=SimpleNamespace(content='{"summary":"orchard harvest"}'))])
    summarizer = QwenSpineSummarizer(Client())
    request = SpineSummaryRequest("user_spine", (SpineSummaryFragment("user", "2026-09-09", "orchard"),))
    assert summarizer(request) == "orchard harvest"
    assert calls[0]["messages"] == request.messages
    assert calls[0]["extra_body"] == {"enable_thinking": False}
    with pytest.raises(TypeError):
        summarizer(records()[0])
    assert len(calls) == 1


@pytest.mark.parametrize("support", [["I recommend an invented campsite"], [], [""], ["trip"] * 5])
def test_raw_summary_adapter_rejects_invented_or_invalid_support(support):
    from tools.assay_user_spine_hierarchy import parse_raw_summary
    with pytest.raises(ValueError, match="exact fragment quotes"):
        parse_raw_summary(json.dumps({"summary": "User requests campsite advice.", "support": support}),
                          "I am planning a trip. Can you recommend campsites?")


def test_raw_summary_quotes_stay_in_audit_not_qwen_spine_inputs():
    from tools.assay_user_spine_hierarchy import parse_raw_summary
    raw = "RAW_SUPPORT_CANARY: Can you recommend campsites?"
    summary, support = parse_raw_summary(json.dumps({"summary": "User requests campsite recommendations.",
                                                     "support": [raw]}), raw)
    request = SpineSummaryRequest("user_spine", (SpineSummaryFragment("user", "2026-09-09", summary),))
    assert support == [raw]
    assert "RAW_SUPPORT_CANARY" not in json.dumps(request.messages)
