"""Keep attention inputs short while independently bounding parent summaries."""
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.episodes.attention_hierarchy import AttentionHierarchySplit, AttentionHierarchyWindow
from memory_condense.search.episodes.surprise_models import ScoredSurpriseSequence
from memory_condense.search.episodes.user_spine_hierarchy import (
    UserSpineExchange, UserSpineHierarchy, _fold, _render_channels,
)
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary, bound_int, exact_text
from memory_condense.search.spine_summary import SpineSummaryFragment


def build_parent_budgeted_spine_hierarchy(
    exchanges, *, scorer, summarize, summarizer_identity, leaf_token_cap=512,
    max_leaf_exchanges=2, max_exchange_channel_tokens=128, max_parent_channel_tokens=512,
    window_exchange_cap=8, max_prompt_tokens=2048,
):
    """Use the original exchange-only attention signal and exact balanced cuts.

    Parents never enter the boundary scorer. Their separate budget permits exact
    concatenation of more existing summaries before a Qwen merge is necessary.
    The returned hierarchy's max_channel_tokens bounds all rendered channels;
    the caller must also bind the smaller exchange input budget in its policy.
    """
    for name, value in (("leaf_token_cap", leaf_token_cap), ("max_leaf_exchanges", max_leaf_exchanges),
                       ("max_exchange_channel_tokens", max_exchange_channel_tokens),
                       ("max_parent_channel_tokens", max_parent_channel_tokens), ("max_prompt_tokens", max_prompt_tokens)):
        bound_int(value, name, 1)
    bound_int(window_exchange_cap, "window_exchange_cap", 2)
    exact_text(summarizer_identity, "summarizer_identity")
    if max_parent_channel_tokens < max_exchange_channel_tokens:
        raise ValueError("parent channel budget must also accommodate unchanged exchange channels")
    if max_exchange_channel_tokens > getattr(scorer, "span_token_cap", max_exchange_channel_tokens):
        raise ValueError("exchange user spine would be truncated by the attention scorer")
    if window_exchange_cap > getattr(scorer, "max_spans", window_exchange_cap):
        raise ValueError("user spine exceeds the attention window cap")
    sources = {}
    for exchange in exchanges:
        if type(exchange) is not UserSpineExchange:
            raise TypeError("hierarchy accepts compiled user-spine exchanges only")
        if any(count_tokens(value) > max_exchange_channel_tokens for value in
               (exchange.user_spine, exchange.attached_context) if value is not None):
            raise ValueError("exchange channel exceeds the attention input summary budget")
        sources.setdefault(exchange.section.source_id, []).append(exchange)
    ordered = tuple(e for group in sources.values() for e in group)
    SectionSummaryIndex(tuple(e.section for e in ordered))
    for source, group in sources.items():
        SectionSummary("validate", source, "validate", tuple(s for e in group for s in e.section.spans), summarizer_identity)
    sections, roots, windows, splits = [], [], [], []
    binding = None
    for source, group in sources.items():
        changes = [0.0]*len(group)
        start = 0
        while start < len(group):
            end = min(len(group), start+window_exchange_cap)
            texts = tuple(e.user_spine if e.user_spine is not None else "Unowned prelude." for e in group[start:end])
            signal = scorer.score_sequence(texts)
            if type(signal) is not ScoredSurpriseSequence:
                raise TypeError("user-spine cuts require authenticated Qwen attention")
            signal.validate_inputs(texts)
            current = tuple(getattr(signal.receipt, name) for name in (
                "model_id", "model_revision", "checkpoint_sha256", "prefix_layers", "attention_layer",
                "implementation_sha256", "span_token_cap"))
            if binding is not None and binding != current:
                raise ValueError("attention identity changed between spine windows")
            binding = current
            for offset, score in enumerate(signal.scores):
                if offset or start == 0:
                    changes[start+offset] = score
            windows.append(AttentionHierarchyWindow(source, start, end,
                identity_sha256([s.identity_payload() for e in group[start:end] for s in e.section.spans]),
                identity_sha256([e.receipt_sha256 for e in group[start:end]]), signal.receipt))
            del signal
            if end == len(group):
                break
            start = end-1

        def merge_channels(nodes):
            spine = _fold([SpineSummaryFragment("user_summary", n[0].spans[0].created_at, n[1])
                           for n in nodes if n[1] is not None], kind="user_spine", user_spine=None,
                          summarize=summarize, cap=max_parent_channel_tokens, prompt_cap=max_prompt_tokens)
            context = _fold([SpineSummaryFragment("attached_summary", n[0].spans[0].created_at, n[2])
                             for n in nodes if n[2] is not None], kind="attached_context", user_spine=spine,
                            summarize=summarize, cap=max_parent_channel_tokens, prompt_cap=max_prompt_tokens)
            return spine, context

        def build(left, right):
            spans = tuple(s for e in group[left:right] for s in e.section.spans)
            section_id = "spine-section-"+identity_sha256([s.receipt_sha256 for s in spans])
            children = ()
            if right-left > 1 and (sum(s.token_count for s in spans) > leaf_token_cap or right-left > max_leaf_exchanges):
                margin = max(1, (right-left)//4)
                cut = max(range(left+margin, right-margin+1),
                          key=lambda i: (changes[i], -abs(2*i-left-right), -i))
                nodes = (build(left, cut), build(cut, right))
                children = tuple(n[0].section_id for n in nodes)
                spine, context = merge_channels(nodes)
                splits.append(AttentionHierarchySplit(section_id, cut, changes[cut],
                    "user_spine_attention_change_at_exchange_boundary"))
            elif right-left == 1:
                spine, context = group[left].user_spine, group[left].attached_context
            else:
                spine, context = merge_channels([(e.section, e.user_spine, e.attached_context) for e in group[left:right]])
            section = SectionSummary(section_id, source, _render_channels(spine, context, spans),
                                     spans, summarizer_identity, child_section_ids=children)
            sections.append(section)
            return section, spine, context

        roots.append(build(0, len(group))[0].section_id)
    oversized = tuple(e.section.section_id for e in ordered if sum(s.token_count for s in e.section.spans) > leaf_token_cap)
    return UserSpineHierarchy(tuple(sections), tuple(roots), ordered, tuple(windows), tuple(splits), oversized,
                             leaf_token_cap, max_leaf_exchanges, max_parent_channel_tokens, window_exchange_cap)
