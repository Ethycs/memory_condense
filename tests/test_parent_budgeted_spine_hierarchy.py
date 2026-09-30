from dataclasses import replace

import pytest

from memory_condense.search.episodes.parent_budgeted_spine_hierarchy import build_parent_budgeted_spine_hierarchy as build
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from memory_condense.search.episodes.user_spine_hierarchy import build_user_spine_hierarchy, _render_channels
from memory_condense.search.spine_summary_reuse import ReusingSpineSummarizer
from tests.test_attention_summary_sections import SummaryLinker
from tests.test_user_spine_hierarchy import make_exchanges, Summarizer


def scorer(cap=64):
    linker = SummaryLinker()
    return linker, QwenAttentionHeadSurpriseScorer(linker, max_spans=3, span_token_cap=cap)


def enrich(exchanges, words=20):
    result = []
    for e in exchanges:
        spine = e.user_spine + " detail"*words if e.user_spine is not None else None
        section = replace(e.section, summary=_render_channels(spine, e.attached_context, e.section.spans), receipt_sha256="")
        result.append(replace(e, section=section, user_spine=spine, receipt_sha256=""))
    return result


def test_equal_budgets_reproduce_existing_hierarchy_and_attention_receipts_exactly():
    _, _, _, exchanges = make_exchanges()
    first, old_scorer = scorer()
    second, new_scorer = scorer()
    kwargs = dict(summarizer_identity="same", leaf_token_cap=512, max_leaf_exchanges=1, window_exchange_cap=3)
    old = build_user_spine_hierarchy(exchanges, scorer=old_scorer, summarize=Summarizer(), max_channel_tokens=64, **kwargs)
    new = build(exchanges, scorer=new_scorer, summarize=Summarizer(), max_exchange_channel_tokens=64,
                max_parent_channel_tokens=64, **kwargs)
    assert old == new and old.receipt_sha256 == new.receipt_sha256
    assert first.inputs == second.inputs


def test_larger_parents_reduce_generation_without_changing_attention_cuts_or_raw_coverage():
    _, _, _, exchanges = make_exchanges()
    exchanges = enrich(exchanges)
    old_linker, old_scorer = scorer(32)
    new_linker, new_scorer = scorer(32)
    requests = []
    def fallback(request):
        requests.append(request)
        return "Condensed fixture."
    old_summary, new_summary = ReusingSpineSummarizer(fallback), ReusingSpineSummarizer(fallback)
    kwargs = dict(summarizer_identity="same", leaf_token_cap=512, max_leaf_exchanges=1, window_exchange_cap=3)
    old = build_user_spine_hierarchy(exchanges, scorer=old_scorer, summarize=old_summary, max_channel_tokens=32, **kwargs)
    new = build(exchanges, scorer=new_scorer, summarize=new_summary, max_exchange_channel_tokens=32,
                max_parent_channel_tokens=128, **kwargs)
    assert old_summary.generated_requests > 0 and new_summary.generated_requests == 0
    assert old_linker.inputs == new_linker.inputs
    assert old.windows == new.windows and old.splits == new.splits
    assert [(s.section_id, s.spans, s.child_section_ids) for s in old.sections] == [
        (s.section_id, s.spans, s.child_section_ids) for s in new.sections]
    assert all("RAW_CANARY" not in str(r.messages) for r in requests)
    assert new.max_channel_tokens == 128 and new_scorer.span_token_cap == 32


def test_parent_allowance_cannot_hide_an_oversized_exchange_attention_input():
    _, _, _, exchanges = make_exchanges()
    linker, signal = scorer(32)
    with pytest.raises(ValueError, match="attention input"):
        build(enrich(exchanges, words=80), scorer=signal, summarize=lambda _: pytest.fail("summary call"),
              summarizer_identity="fixture", max_exchange_channel_tokens=32, max_parent_channel_tokens=512,
              window_exchange_cap=3)
    assert not linker.inputs


def test_parallel_branches_preserve_attention_receipts_order_and_hydration():
    from threading import Barrier
    from datetime import datetime, timezone
    from memory_condense.domain.schemas import Turn
    from memory_condense.search.episodes.attention_hierarchy import compile_attention_atoms
    from memory_condense.search.episodes.user_spine_hierarchy import compile_user_spine_exchanges
    turns=[Turn(turn_id=str(i),source_id='same',role='user',text=f'RAW_CANARY {i}',
                created_at=datetime(2026,9,29,tzinfo=timezone.utc)) for i in range(8)]
    leaves={f'orchard leaf {i}' for i in range(8)}
    atoms=compile_attention_atoms(turns,summarize_raw=lambda text:'orchard leaf '+text.split()[-1],
                                   summarizer_identity='fixture',atom_token_cap=32)
    exchanges=compile_user_spine_exchanges(atoms,summarize=Summarizer(),summarizer_identity='fixture')
    barrier=Barrier(2)
    def concurrent(request):
        if len(request.fragments)==2 and all(f.summary in leaves for f in request.fragments):
            barrier.wait(timeout=5)
        return Summarizer()(request)
    kwargs=dict(summarizer_identity='same',leaf_token_cap=512,max_leaf_exchanges=1,
                window_exchange_cap=3,max_exchange_channel_tokens=64,max_parent_channel_tokens=128)
    linker1,signal1=scorer()
    linker2,signal2=scorer()
    sequential=build(exchanges,scorer=signal1,summarize=Summarizer(),**kwargs)
    parallel=build(exchanges,scorer=signal2,summarize=concurrent,max_workers=3,**kwargs)
    assert parallel==sequential
    assert parallel.receipt_sha256==sequential.receipt_sha256
    assert linker1.inputs==linker2.inputs


@pytest.mark.parametrize("kwargs", [{"max_exchange_channel_tokens": 128},
                                   {"max_parent_channel_tokens": 16}, {"window_exchange_cap": 4}])
def test_invalid_input_or_parent_budget_fails_before_attention(kwargs):
    _, _, _, exchanges = make_exchanges()
    linker, signal = scorer(64)
    options = dict(max_exchange_channel_tokens=64, max_parent_channel_tokens=128, window_exchange_cap=3)
    options.update(kwargs)
    with pytest.raises(ValueError):
        build(exchanges, scorer=signal, summarize=lambda _: pytest.fail("summary call"),
              summarizer_identity="fixture", **options)
    assert not linker.inputs
