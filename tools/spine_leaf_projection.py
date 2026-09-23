"""Project exact query-addressable leaves without generating unused parents.

The attention partition and leaf channel merges match the complete hierarchy.
Only summaries enter the scorer and merger. Raw span metadata controls the same
whole-exchange size limits; raw text is unavailable here.
"""
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.episodes.user_spine_hierarchy import _fold, _render_channels
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.spine_summary import SpineSummaryFragment


def project_source_leaves(exchanges, *, scorer, summarize, summarizer_identity):
    """Fixed production policy: 512 raw tokens, two exchanges, 128-token channels."""
    rows = tuple(exchanges)
    SectionSummaryIndex(tuple(e.section for e in rows))
    sources = {e.section.source_id for e in rows}
    if not rows or len(sources) != 1:
        raise ValueError("leaf projection requires exactly one nonempty source")
    changes = [0.0] * len(rows)
    attention = []
    start = 0
    while start < len(rows):
        end = min(len(rows), start + 8)
        texts = tuple(e.user_spine if e.user_spine is not None else "Unowned prelude." for e in rows[start:end])
        signal = scorer.score_sequence(texts)
        signal.validate_inputs(texts)
        attention.append(signal.receipt.receipt_sha256)
        for offset, score in enumerate(signal.scores):
            if offset or start == 0:
                changes[start + offset] = score
        if end == len(rows):
            break
        start = end - 1

    leaves, splits = [], []
    def descend(left, right):
        spans = tuple(s for e in rows[left:right] for s in e.section.spans)
        section_id = "spine-section-" + identity_sha256([s.receipt_sha256 for s in spans])
        if right - left > 1 and (sum(s.token_count for s in spans) > 512 or right - left > 2):
            margin = max(1, (right - left) // 4)
            cut = max(range(left + margin, right - margin + 1),
                      key=lambda i: (changes[i], -abs(2 * i - left - right), -i))
            descend(left, cut)
            descend(cut, right)
            splits.append({"section_id": section_id, "split_atom": cut, "attention_change": changes[cut]})
            return
        group = rows[left:right]
        if len(group) == 1:
            spine, context = group[0].user_spine, group[0].attached_context
        else:
            spine = _fold([SpineSummaryFragment("user_summary", e.section.spans[0].created_at, e.user_spine)
                           for e in group if e.user_spine is not None], kind="user_spine", user_spine=None,
                          summarize=summarize, cap=128, prompt_cap=2048)
            context = _fold([SpineSummaryFragment("attached_summary", e.section.spans[0].created_at, e.attached_context)
                             for e in group if e.attached_context is not None], kind="attached_context", user_spine=spine,
                            summarize=summarize, cap=128, prompt_cap=2048)
        leaves.append(SectionSummary(section_id, rows[0].section.source_id,
            _render_channels(spine, context, spans), spans, summarizer_identity))
    descend(0, len(rows))
    return tuple(leaves), tuple(splits), tuple(attention)
