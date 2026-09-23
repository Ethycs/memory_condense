from datetime import datetime, timezone

import pytest

from memory_condense.domain.schemas import Turn
from memory_condense.search.episodes.attention_hierarchy import compile_attention_atoms
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from memory_condense.search.episodes.user_spine_hierarchy import build_user_spine_hierarchy, compile_user_spine_exchanges
from tests.test_attention_summary_sections import SummaryLinker
from tests.test_user_spine_hierarchy import make_exchanges, Summarizer
from tools.spine_leaf_projection import project_source_leaves


def test_partition_validation_uses_transcript_order_before_sorting_index_ids():
    from dataclasses import replace
    from memory_condense.domain._discourse_identity import identity_sha256
    from tools.compile_spine_leaf_projection import validate_leaf_partition
    _, _, _, exchanges = make_exchanges()
    sections = tuple(replace(e.section, section_id=f"section-{9-i}", receipt_sha256="") for i, e in enumerate(exchanges))
    expected = identity_sha256([span.receipt_sha256 for s in sections for span in s.spans])
    index = validate_leaf_partition({"source": (sections,)}, expected)
    assert [s.section_id for s in index.sections] == sorted(s.section_id for s in sections)
    assert index.sections != sections
    with pytest.raises(ValueError, match="reordered"):
        validate_leaf_partition({"source": (sections[::-1],)}, expected)
    with pytest.raises(ValueError, match="lost"):
        validate_leaf_partition({"source": (sections[:-1],)}, expected)


@pytest.mark.parametrize("population", [0, 1, 7, 8, 9, 19])
def test_projection_matches_full_hierarchy_leaf_bytes_and_every_attention_cut(population):
    if population == 0:
        _, _, _, exchanges = make_exchanges()
    else:
        turns = [Turn(turn_id=str(i), source_id="fixture-source", role="user",
            created_at=datetime(2026, 9, 9, tzinfo=timezone.utc), text="RAW_CANARY " * (600 if i % 4 == 0 else 2))
            for i in range(population)]
        summaries = iter("orchard harvest" if i % 3 else "observatory reservations" for i in range(population))
        atoms = compile_attention_atoms(turns, summarize_raw=lambda _: next(summaries),
                                        summarizer_identity="raw-fixture", atom_token_cap=2048)
        exchanges = compile_user_spine_exchanges(atoms, summarize=Summarizer(), summarizer_identity="fixture",
                                                 max_channel_tokens=128)
    def scorer():
        return QwenAttentionHeadSurpriseScorer(SummaryLinker(), max_spans=8, span_token_cap=128)
    full = build_user_spine_hierarchy(exchanges, scorer=scorer(), summarize=Summarizer(),
        summarizer_identity="fixture", max_channel_tokens=128, max_leaf_exchanges=2, window_exchange_cap=8)
    leaves, splits, receipts = project_source_leaves(exchanges, scorer=scorer(), summarize=Summarizer(),
        summarizer_identity="fixture")
    assert leaves == tuple(s for s in full.sections if not s.child_section_ids)
    assert splits == tuple({"section_id": s.section_id, "split_atom": s.split_atom,
                           "attention_change": s.attention_change} for s in full.splits)
    assert receipts == tuple(w.signal.receipt_sha256 for w in full.windows)
