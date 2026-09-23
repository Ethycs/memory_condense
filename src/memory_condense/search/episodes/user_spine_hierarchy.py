"""Attention partitions user-led exchanges; summaries route exact raw sections.

User leads determine the boundary signal. Responses stay attached to their lead
and cannot enter the user-spine summary channel. Whole exchanges are indivisible:
an oversized exchange is retained and diagnosed, never silently truncated.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.episodes.attention_hierarchy import (
    AttentionHierarchySplit, AttentionHierarchyWindow,
)
from memory_condense.search.episodes.surprise_models import ScoredSurpriseSequence, SurpriseSequenceScorer
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary, bound_int, exact_text
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest


FORMAT = "memory-condense-user-spine-attention-hierarchy-v1"
Summarize = Callable[[SpineSummaryRequest], str]


@dataclass(frozen=True, slots=True)
class UserSpineExchange(SealedIdentity):
    section: SectionSummary
    user_spine: str | None
    attached_context: str | None
    lead_turn_id: str | None
    receipt_sha256: str = ""

    def __post_init__(self):
        if type(self.section) is not SectionSummary or self.section.child_section_ids:
            raise ValueError("an exchange must be one exact leaf section")
        user_turns = tuple(dict.fromkeys(s.turn_id for s in self.section.spans if s.role == "user"))
        if user_turns != (() if self.lead_turn_id is None else (self.lead_turn_id,)):
            raise ValueError("an exchange has exactly one user lead, or an orphan prelude")
        if self.lead_turn_id is not None and self.section.spans[0].turn_id != self.lead_turn_id:
            raise ValueError("the user lead must precede attached responses")
        if (self.user_spine is not None) != (self.lead_turn_id is not None):
            raise ValueError("user spine and lead ownership disagree")
        for value in (self.user_spine, self.attached_context):
            if value is not None:
                exact_text(value, "summary channel")
        if self.section.summary != _render_channels(self.user_spine, self.attached_context, self.section.spans):
            raise ValueError("exchange summary channels changed")
        self._seal()


def _render_channels(spine, context, spans):
    return canonical_json({
        "user_spine": spine, "attached_context_not_user_assertions": context,
        "transcript_date_range": [spans[0].created_at, spans[-1].created_at],
    })


def _fold(fragments, *, kind, user_spine, summarize, cap, prompt_cap):
    """Bound every merge to two summaries; no prefix truncation or raw callback."""
    rows = list(fragments)
    if not rows:
        return None
    while True:
        merged = []
        for start in range(0, len(rows), 2):
            pair = rows[start:start + 2]
            request = SpineSummaryRequest(kind, tuple(pair), user_spine, cap, prompt_cap)
            summary = exact_text(summarize(request), "compiled spine summary")
            if count_tokens(summary) > cap:
                raise ValueError("compiled spine summary exceeds its output budget")
            merged.append(SpineSummaryFragment(
                "user_summary" if kind == "user_spine" else "attached_summary",
                pair[0].transcript_date if len(pair) == 1 else
                pair[0].transcript_date.split(" through ")[0] + " through " +
                pair[-1].transcript_date.split(" through ")[-1], summary,
            ))
        if len(merged) == 1:
            return merged[0].summary
        rows = merged


def compile_user_spine_exchanges(
    atoms: Sequence[SectionSummary], *, summarize: Summarize, summarizer_identity: str,
    max_channel_tokens: int = 64, max_prompt_tokens: int = 2048,
) -> tuple[UserSpineExchange, ...]:
    """Consume ordered atomic summaries, never raw turns, queries or references.

    Compile atoms separately with a non-Qwen raw summarizer. Input order is
    authoritative transcript order; timestamps are preserved, not used to sort.
    Validate the whole population before invoking the summary callback.
    """
    bound_int(max_channel_tokens, "max_channel_tokens", 1)
    bound_int(max_prompt_tokens, "max_prompt_tokens", 1)
    exact_text(summarizer_identity, "summarizer_identity")
    sources: dict[str, list[SectionSummary]] = {}
    seen, turn_sources = set(), {}
    for atom in atoms:
        if type(atom) is not SectionSummary or len(atom.spans) != 1 or atom.child_section_ids:
            raise TypeError("user spine accepts atomic summary descriptors only")
        if atom.section_id in seen:
            raise ValueError("duplicate user-spine atom")
        seen.add(atom.section_id)
        span = atom.spans[0]
        if span.role not in {"user", "assistant", "system"}:
            raise ValueError("unsupported transcript role")
        if turn_sources.setdefault(span.turn_id, span.source_id) != span.source_id:
            raise ValueError("one turn ID cannot belong to multiple sources")
        sources.setdefault(atom.source_id, []).append(atom)
    for source, rows in sources.items():
        SectionSummary("validate", source, "validate", tuple(a.spans[0] for a in rows), summarizer_identity)
    output = []
    for source, rows in sources.items():
        groups: list[list[SectionSummary]] = []
        for atom in rows:
            span = atom.spans[0]
            if not groups or (span.role == "user" and groups[-1][-1].spans[0].turn_id != span.turn_id):
                groups.append([])
            groups[-1].append(atom)
        for group in groups:
            fragments = [SpineSummaryFragment(a.spans[0].role, a.spans[0].created_at, a.summary) for a in group]
            spine = _fold([f for f in fragments if f.role == "user"], kind="user_spine", user_spine=None,
                          summarize=summarize, cap=max_channel_tokens, prompt_cap=max_prompt_tokens)
            context = _fold([f for f in fragments if f.role != "user"], kind="attached_context", user_spine=spine,
                            summarize=summarize, cap=max_channel_tokens, prompt_cap=max_prompt_tokens)
            spans = tuple(a.spans[0] for a in group)
            section_id = "spine-exchange-" + identity_sha256([s.receipt_sha256 for s in spans])
            section = SectionSummary(section_id, source, _render_channels(spine, context, spans),
                                     spans, summarizer_identity)
            output.append(UserSpineExchange(section, spine, context,
                                            spans[0].turn_id if spine is not None else None))
    return tuple(output)


@dataclass(frozen=True, slots=True)
class UserSpineHierarchy(SealedIdentity):
    sections: tuple[SectionSummary, ...]
    root_section_ids: tuple[str, ...]
    exchanges: tuple[UserSpineExchange, ...]
    windows: tuple[AttentionHierarchyWindow, ...]
    splits: tuple[AttentionHierarchySplit, ...]
    oversized_exchange_ids: tuple[str, ...]
    leaf_token_cap: int
    max_leaf_exchanges: int
    max_channel_tokens: int
    window_exchange_cap: int
    format: str = FORMAT
    receipt_sha256: str = ""

    def __post_init__(self):
        for name in ("sections", "root_section_ids", "exchanges", "windows", "splits", "oversized_exchange_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.format != FORMAT:
            raise ValueError("unsupported user-spine hierarchy format")
        for name in ("leaf_token_cap", "max_leaf_exchanges", "max_channel_tokens", "window_exchange_cap"):
            bound_int(getattr(self, name), name, 2 if name == "window_exchange_cap" else 1)
        index = self.summary_index()
        children = {child for s in index.sections for child in s.child_section_ids}
        roots = {s.section_id: s for s in index.sections if s.section_id not in children}
        if set(roots) != set(self.root_section_ids) or len(roots) != len(self.root_section_ids):
            raise ValueError("user-spine roots changed")
        expected = tuple(span for e in self.exchanges for span in e.section.spans)
        observed = tuple(span for rid in self.root_section_ids for span in roots[rid].spans)
        if observed != expected:
            raise ValueError("hierarchy must retain every exchange span in order")
        oversized = tuple(e.section.section_id for e in self.exchanges
                          if sum(s.token_count for s in e.section.spans) > self.leaf_token_cap)
        if oversized != self.oversized_exchange_ids:
            raise ValueError("oversized exchange diagnostics changed")
        leaves = [s for s in self.sections if not s.child_section_ids]
        cursor = 0
        # Tree validation above proves child coverage. Every leaf must also end
        # at an exchange boundary; source roots retain input source order.
        by_first = {e.section.spans[0].receipt_sha256: i for i, e in enumerate(self.exchanges)}
        for leaf in leaves:
            first = by_first.get(leaf.spans[0].receipt_sha256)
            if first is None:
                raise ValueError("a leaf split an exchange")
            covered = []
            cursor = first
            while len(covered) < len(leaf.spans) and cursor < len(self.exchanges):
                covered.extend(self.exchanges[cursor].section.spans)
                cursor += 1
            if tuple(covered) != leaf.spans or cursor - first > self.max_leaf_exchanges:
                raise ValueError("a leaf split or overfilled its exchange partition")
            if sum(s.token_count for s in leaf.spans) > self.leaf_token_cap and cursor - first != 1:
                raise ValueError("only indivisible exchanges may exceed the leaf cap")
        self._seal()

    def summary_index(self):
        return SectionSummaryIndex(self.sections)


def build_user_spine_hierarchy(
    exchanges: Sequence[UserSpineExchange], *, scorer: SurpriseSequenceScorer,
    summarize: Summarize, summarizer_identity: str, leaf_token_cap: int = 512,
    max_leaf_exchanges: int = 4, max_channel_tokens: int = 64,
    window_exchange_cap: int = 32, max_prompt_tokens: int = 2048,
) -> UserSpineHierarchy:
    """Use Qwen head-transport change between user summaries for balanced cuts.

    Attached context is available to routing summaries, but never controls the
    attention boundary signal. Source boundaries remain hard. Oversized single
    exchanges remain intact for atomic hydration or an explicit budget fallback.
    """
    for name, value in (("leaf_token_cap", leaf_token_cap), ("max_leaf_exchanges", max_leaf_exchanges),
                        ("max_channel_tokens", max_channel_tokens), ("max_prompt_tokens", max_prompt_tokens)):
        bound_int(value, name, 1)
    bound_int(window_exchange_cap, "window_exchange_cap", 2)
    exact_text(summarizer_identity, "summarizer_identity")
    if max_channel_tokens > getattr(scorer, "span_token_cap", max_channel_tokens):
        raise ValueError("user spine would be truncated by the attention scorer")
    if window_exchange_cap > getattr(scorer, "max_spans", window_exchange_cap):
        raise ValueError("user spine exceeds the attention window cap")
    sources: dict[str, list[UserSpineExchange]] = {}
    for exchange in exchanges:
        if type(exchange) is not UserSpineExchange:
            raise TypeError("hierarchy accepts compiled user-spine exchanges only")
        if any(count_tokens(value) > max_channel_tokens for value in
               (exchange.user_spine, exchange.attached_context) if value is not None):
            raise ValueError("exchange channel exceeds the summary budget")
        sources.setdefault(exchange.section.source_id, []).append(exchange)
    ordered = tuple(e for group in sources.values() for e in group)
    # Reject overlaps, duplicate IDs and invalid order before any model call.
    SectionSummaryIndex(tuple(e.section for e in ordered))
    for source, group in sources.items():
        SectionSummary("validate", source, "validate", tuple(s for e in group for s in e.section.spans), summarizer_identity)
    sections, roots, windows, splits = [], [], [], []
    binding = None
    for source, group in sources.items():
        changes = [0.0] * len(group)
        start = 0
        while start < len(group):
            end = min(len(group), start + window_exchange_cap)
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
                    changes[start + offset] = score
            windows.append(AttentionHierarchyWindow(source, start, end,
                identity_sha256([s.identity_payload() for e in group[start:end] for s in e.section.spans]),
                identity_sha256([e.receipt_sha256 for e in group[start:end]]), signal.receipt))
            del signal
            if end == len(group):
                break
            start = end - 1

        def merge_channels(nodes):
            spine = _fold([SpineSummaryFragment("user_summary", n[0].spans[0].created_at, n[1])
                           for n in nodes if n[1] is not None], kind="user_spine", user_spine=None,
                          summarize=summarize, cap=max_channel_tokens, prompt_cap=max_prompt_tokens)
            context = _fold([SpineSummaryFragment("attached_summary", n[0].spans[0].created_at, n[2])
                             for n in nodes if n[2] is not None], kind="attached_context", user_spine=spine,
                            summarize=summarize, cap=max_channel_tokens, prompt_cap=max_prompt_tokens)
            return spine, context

        def build(left, right):
            spans = tuple(s for e in group[left:right] for s in e.section.spans)
            section_id = "spine-section-" + identity_sha256([s.receipt_sha256 for s in spans])
            children = ()
            if right - left > 1 and (sum(s.token_count for s in spans) > leaf_token_cap or right - left > max_leaf_exchanges):
                margin = max(1, (right - left) // 4)
                cut = max(range(left + margin, right - margin + 1),
                          key=lambda i: (changes[i], -abs(2 * i - left - right), -i))
                nodes = (build(left, cut), build(cut, right))
                children = tuple(n[0].section_id for n in nodes)
                spine, context = merge_channels(nodes)
                splits.append(AttentionHierarchySplit(section_id, cut, changes[cut],
                              "user_spine_attention_change_at_exchange_boundary"))
            elif right - left == 1:
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
                              leaf_token_cap, max_leaf_exchanges, max_channel_tokens, window_exchange_cap)
