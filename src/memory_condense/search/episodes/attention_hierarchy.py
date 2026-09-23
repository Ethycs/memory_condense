"""Attention-guided hierarchical sections with exact lossless raw leaves.

Qwen OV-transport change over atomic SUMMARIES guides contiguous splits. Raw
atoms are summarized separately, before this model boundary. Small fixed token
atoms only bound compilation work; they are not the retrieved sections.
Each bounded scoring window overlaps its predecessor by one atom so no seam
loses its attention-change measurement. Only scalar cut/receipt metadata lives
past construction, never transport vectors or transformer token state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import _get_encoder, count_tokens
from memory_condense.domain.schemas import Turn
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.episodes.surprise_models import (
    AttentionHeadSurpriseReceipt,
    ScoredSurpriseSequence,
    SurpriseSequenceScorer,
)
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import (
    RawSectionSpan, SectionSummary, bound_int, exact_text,
)


FORMAT = "memory-condense-attention-section-hierarchy-v1"


@dataclass(frozen=True, slots=True)
class AttentionHierarchyWindow(SealedIdentity):
    source_id: str
    atom_start: int
    atom_end: int
    raw_spans_sha256: str
    summary_atoms_sha256: str
    signal: AttentionHeadSurpriseReceipt
    receipt_sha256: str = ""

    def __post_init__(self):
        if self.atom_end - self.atom_start != self.signal.input_spans:
            raise ValueError("attention window does not bind all its atoms")
        self._seal()


@dataclass(frozen=True, slots=True)
class AttentionHierarchySplit(SealedIdentity):
    section_id: str
    split_atom: int
    attention_change: float
    reason: str
    receipt_sha256: str = ""

    def __post_init__(self):
        if not 0 <= self.attention_change <= 1:
            raise ValueError("attention boundary change must lie in [0,1]")
        self._seal()


@dataclass(frozen=True, slots=True)
class AttentionSectionHierarchy(SealedIdentity):
    sections: tuple[SectionSummary, ...]
    root_section_ids: tuple[str, ...]
    windows: tuple[AttentionHierarchyWindow, ...]
    splits: tuple[AttentionHierarchySplit, ...]
    atom_token_cap: int
    leaf_token_cap: int
    window_atom_cap: int
    max_summary_tokens: int = 64
    retained_transformer_token_state_bytes: int = 0
    format: str = FORMAT
    receipt_sha256: str = ""

    def __post_init__(self):
        for name in ("sections", "root_section_ids", "windows", "splits"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.format != FORMAT or self.retained_transformer_token_state_bytes != 0:
            raise ValueError("attention hierarchy state/format changed")
        bound_int(self.atom_token_cap, "atom_token_cap", 1)
        bound_int(self.leaf_token_cap, "leaf_token_cap", self.atom_token_cap)
        bound_int(self.window_atom_cap, "window_atom_cap", 2)
        bound_int(self.max_summary_tokens, "max_summary_tokens", 1)
        if any(count_tokens(section.summary) > self.max_summary_tokens for section in self.sections):
            raise ValueError("hierarchy summary exceeds its token budget")
        index = SectionSummaryIndex(self.sections)
        children = {child for section in self.sections for child in section.child_section_ids}
        roots = {section.section_id for section in self.sections} - children
        if roots != set(self.root_section_ids) or len(roots) != len(self.root_section_ids):
            raise ValueError("hierarchy root membership changed")
        if any(sum(span.token_count for span in section.spans) > self.leaf_token_cap
               for section in self.sections if not section.child_section_ids):
            raise ValueError("attention leaf exceeds its hard token cap")
        if len(index.sections) != 2 * len(self.splits) + len(roots):
            raise ValueError("attention hierarchy must be a binary partition forest")
        self._seal()

    def summary_index(self) -> SectionSummaryIndex:
        return SectionSummaryIndex(self.sections)


def _atoms(turn: Turn, token_cap: int) -> tuple[tuple[RawSectionSpan, str], ...]:
    """Encode once, then form UTF-8-safe atom slices without dropping whitespace."""
    if not turn.source_id or not turn.text:
        raise ValueError("attention sections require nonempty turns with exact sources")
    encoder = _get_encoder()
    encoded = encoder.encode(turn.text, disallowed_special=())
    token_bytes = [encoder.decode_single_token_bytes(token) for token in encoded]
    digest = quote_sha256(turn.text)
    output = []
    start_token = start_char = 0
    while start_token < len(encoded):
        end_token = min(len(encoded), start_token + token_cap)
        while end_token > start_token:
            try:
                text = b"".join(token_bytes[start_token:end_token]).decode("utf-8")
                break
            except UnicodeDecodeError:
                end_token -= 1
        if end_token == start_token:
            raise ValueError("attention atom cap cannot hold one complete Unicode character")
        end_char = start_char + len(text)
        span = RawSectionSpan(
            turn.turn_id, turn.source_id, turn.role, turn.created_at.isoformat(),
            start_char, end_char, digest, quote_sha256(text), count_tokens(text),
        )
        if span.token_count > token_cap:
            raise ValueError("retokenized attention atom exceeds its hard cap")
        output.append((span, text))
        start_token, start_char = end_token, end_char
    if "".join(text for _, text in output) != turn.text:
        raise ValueError("attention atoms failed exact raw coverage")
    return tuple(output)


def compile_attention_atoms(
    turns: Sequence[Turn], *, summarize_raw: Callable[[str], str],
    summarizer_identity: str, atom_token_cap: int = 64, max_summary_tokens: int = 64,
) -> tuple[SectionSummary, ...]:
    """Compile query-independent atomic summaries with a separate raw summarizer.

    Use a non-Qwen summarizer (e.g. the local LFM compiler) or load precomputed
    SectionSummary objects instead. The Qwen hierarchy builder accepts only the
    resulting summary descriptors, never these raw turns or this callback.
    """
    bound_int(atom_token_cap, "atom_token_cap", 1)
    bound_int(max_summary_tokens, "max_summary_tokens", 1)
    exact_text(summarizer_identity, "summarizer_identity")
    output, seen = [], set()
    for turn in turns:
        if turn.turn_id in seen:
            raise ValueError("attention input turn IDs must be unique")
        seen.add(turn.turn_id)
        for span, raw in _atoms(turn, atom_token_cap):
            summary = exact_text(summarize_raw(raw), "atomic summary")
            if count_tokens(summary) > max_summary_tokens:
                raise ValueError("atomic summary exceeds its token budget")
            output.append(SectionSummary(
                "attention-atom-" + span.receipt_sha256, turn.source_id, summary,
                (span,), summarizer_identity,
            ))
    return tuple(output)


def build_attention_section_hierarchy(
    atoms: Sequence[SectionSummary], *, scorer: SurpriseSequenceScorer,
    summarize_summaries: Callable[[str], str], summarizer_identity: str,
    atom_token_cap: int = 64, leaf_token_cap: int = 512,
    window_atom_cap: int = 128, max_summary_tokens: int = 64,
) -> AttentionSectionHierarchy:
    """Build from precompiled summaries; Qwen never receives raw transcript text.

    Pass ``QwenAttentionHeadSurpriseScorer`` for live attention. A lexical or
    embedding-only scorer cannot satisfy the required head-signal receipt.
    Leaf and parent summaries consume only lower-level summaries. The neutral
    attention probe is independent of any future retrieval question.
    """
    bound_int(atom_token_cap, "atom_token_cap", 1)
    bound_int(leaf_token_cap, "leaf_token_cap", atom_token_cap)
    bound_int(window_atom_cap, "window_atom_cap", 2)
    bound_int(max_summary_tokens, "max_summary_tokens", 1)
    exact_text(summarizer_identity, "summarizer_identity")
    if not callable(getattr(scorer, "score_sequence", None)):
        raise TypeError("attention hierarchy requires a sequence attention scorer")
    if max_summary_tokens > getattr(scorer, "span_token_cap", max_summary_tokens):
        raise ValueError("attention summaries would be truncated by the scorer")
    if window_atom_cap > getattr(scorer, "max_spans", window_atom_cap):
        raise ValueError("attention windows exceed the scorer's span cap")
    sources: dict[str, list[SectionSummary]] = {}
    seen = set()
    for atom in atoms:
        if type(atom) is not SectionSummary or len(atom.spans) != 1 or atom.child_section_ids:
            raise ValueError("hierarchy inputs must be atomic summary descriptors")
        if atom.section_id in seen:
            raise ValueError("attention atom IDs must be unique")
        seen.add(atom.section_id)
        if atom.spans[0].token_count > atom_token_cap or count_tokens(atom.summary) > max_summary_tokens:
            raise ValueError("attention atom or summary exceeds its token budget")
        sources.setdefault(atom.source_id, []).append(atom)
    sections, roots, windows, splits = [], [], [], []
    model_binding = None
    for source_id, atoms in sources.items():
        # Validate continuity before any model call, including loaded snapshots.
        SectionSummary("validation", source_id, "validation",
                       tuple(atom.spans[0] for atom in atoms), summarizer_identity)
        changes = [0.0] * len(atoms)
        start = 0
        while start < len(atoms):
            end = min(len(atoms), start + window_atom_cap)
            texts = tuple(atom.summary for atom in atoms[start:end])
            signal = scorer.score_sequence(texts)
            if type(signal) is not ScoredSurpriseSequence:
                raise TypeError("chunking requires an authenticated attention-head signal")
            signal.validate_inputs(texts)
            binding = tuple(getattr(signal.receipt, name) for name in (
                "model_id", "model_revision", "checkpoint_sha256", "prefix_layers",
                "attention_layer", "implementation_sha256", "span_token_cap",
            ))
            if model_binding is not None and binding != model_binding:
                raise ValueError("attention scorer identity changed between windows")
            model_binding = binding
            for offset, score in enumerate(signal.scores):
                if offset or start == 0:
                    changes[start + offset] = score
            windows.append(AttentionHierarchyWindow(
                source_id, start, end,
                identity_sha256([atom.spans[0].identity_payload() for atom in atoms[start:end]]),
                identity_sha256([atom.receipt_sha256 for atom in atoms[start:end]]),
                signal.receipt,
            ))
            del signal  # Similarity matrices die with each bounded window.
            if end == len(atoms):
                break
            start = end - 1
        cumulative = [0]
        for atom in atoms:
            cumulative.append(cumulative[-1] + atom.spans[0].token_count)

        def build(left: int, right: int) -> SectionSummary:
            spans = tuple(atom.spans[0] for atom in atoms[left:right])
            section_id = "attention-section-" + identity_sha256({
                "format": FORMAT, "source_id": source_id,
                "spans": [span.receipt_sha256 for span in spans],
            })
            children = ()
            if cumulative[right] - cumulative[left] > leaf_token_cap:
                # Balance bounds prevent pathological linear-depth trees. Within
                # the middle half, the strongest attention change owns the cut.
                margin = max(1, (right - left) // 4)
                positions = range(left + margin, right - margin + 1)
                cut = max(positions, key=lambda i: (changes[i], -abs(2 * i - left - right), -i))
                first, second = build(left, cut), build(cut, right)
                children = (first.section_id, second.section_id)
                material = first.summary + "\n\n" + second.summary
                splits.append(AttentionHierarchySplit(section_id, cut, changes[cut],
                              "maximum_attention_change_within_balanced_range"))
            else:
                material = "\n\n".join(atom.summary for atom in atoms[left:right])
            summary = exact_text(summarize_summaries(material), "summary")
            if count_tokens(summary) > max_summary_tokens:
                raise ValueError("hierarchy summary exceeds its token budget")
            node = SectionSummary(section_id, source_id, summary, spans,
                                  summarizer_identity, child_section_ids=children)
            sections.append(node)
            return node

        roots.append(build(0, len(atoms)).section_id)
    return AttentionSectionHierarchy(tuple(sections), tuple(roots), tuple(windows), tuple(splits),
                                     atom_token_cap, leaf_token_cap, window_atom_cap, max_summary_tokens)
