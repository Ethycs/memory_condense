"""Query-independent summaries bound to exact raw transcript sections.

Summaries are routing text only. They carry no factual authority and retain no
raw transcript text; source coordinates and hashes are the hydration contract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Turn
from memory_condense.domain.sealed import SealedIdentity


SECTION_SUMMARY_FORMAT = "memory-condense-section-summary-v1"


def exact_text(value: str, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def bound_int(value: int, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class RawSectionSpan(SealedIdentity):
    turn_id: str
    source_id: str
    role: str
    created_at: str
    start_char: int
    end_char: int
    turn_text_sha256: str
    span_text_sha256: str
    token_count: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("turn_id", "source_id", "role", "created_at"):
            exact_text(getattr(self, name), name)
        bound_int(self.start_char, "start_char")
        bound_int(self.end_char, "end_char", self.start_char + 1)
        bound_int(self.token_count, "token_count")
        for digest in (self.turn_text_sha256, self.span_text_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("raw section hashes must be lowercase SHA-256")
        self._seal()

    @classmethod
    def from_turn(cls, turn: Turn, *, start_char: int = 0,
                  end_char: int | None = None) -> RawSectionSpan:
        end = len(turn.text) if end_char is None else end_char
        bound_int(start_char, "start_char")
        bound_int(end, "end_char", start_char + 1)
        if end > len(turn.text) or not turn.source_id:
            raise ValueError("section span needs an exact source and valid raw bounds")
        text = turn.text[start_char:end]
        return cls(
            turn.turn_id, turn.source_id, turn.role, turn.created_at.isoformat(),
            start_char, end, quote_sha256(turn.text), quote_sha256(text), count_tokens(text),
        )


@dataclass(frozen=True, slots=True)
class SectionSummary(SealedIdentity):
    section_id: str
    source_id: str
    summary: str
    spans: tuple[RawSectionSpan, ...]
    summarizer_identity: str
    child_section_ids: tuple[str, ...] = ()
    format: str = SECTION_SUMMARY_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("section_id", "source_id", "summary", "summarizer_identity"):
            exact_text(getattr(self, name), name)
        if self.format != SECTION_SUMMARY_FORMAT:
            raise ValueError("unsupported section summary format")
        object.__setattr__(self, "spans", tuple(self.spans))
        object.__setattr__(self, "child_section_ids", tuple(self.child_section_ids))
        if len(set(self.child_section_ids)) != len(self.child_section_ids) or any(
            type(child) is not str or not child.strip() or child == self.section_id
            for child in self.child_section_ids
        ):
            raise ValueError("child section IDs must be unique exact distinct IDs")
        if not self.spans or any(type(span) is not RawSectionSpan for span in self.spans):
            raise ValueError("a section requires exact raw span references")
        previous = None
        seen_turns: set[str] = set()
        for span in self.spans:
            if span.source_id != self.source_id:
                raise ValueError("a section cannot cross source boundaries")
            if previous is not None and previous.turn_id == span.turn_id:
                if span.start_char != previous.end_char or (
                    span.turn_text_sha256, span.role, span.created_at
                ) != (previous.turn_text_sha256, previous.role, previous.created_at):
                    raise ValueError("same-turn section spans must be contiguous and consistent")
            elif span.turn_id in seen_turns:
                raise ValueError("a section cannot revisit an earlier turn")
            seen_turns.add(span.turn_id)
            previous = span
        self._seal()

    @classmethod
    def from_dict(cls, payload: dict) -> SectionSummary:
        body = dict(payload)
        body["spans"] = tuple(RawSectionSpan(**span) for span in body["spans"])
        return cls(**body)


def summarize_section(
    section_id: str, turns: Sequence[Turn], *,
    summarize: Callable[[tuple[Turn, ...]], str], summarizer_identity: str,
    max_summary_tokens: int = 128,
) -> SectionSummary:
    """Summarize one explicit whole-turn section once, before query time.

    The caller supplies section membership in transcript order. This supports
    user-led episodes, larger topical sections, and explicit ingest boundaries.
    The callback may use a local model or a previously computed summary.
    """
    bound_int(max_summary_tokens, "max_summary_tokens", 1)
    ordered = tuple(turns)
    if not ordered or len({turn.turn_id for turn in ordered}) != len(ordered):
        raise ValueError("section turns must be nonempty and unique")
    spans = tuple(RawSectionSpan.from_turn(turn) for turn in ordered)
    if len({span.source_id for span in spans}) != 1:
        raise ValueError("a section cannot cross source boundaries")
    summary = exact_text(summarize(ordered), "summary")
    if count_tokens(summary) > max_summary_tokens:
        raise ValueError("section summary exceeds its token budget")
    return SectionSummary(section_id, spans[0].source_id, summary, spans, summarizer_identity)


def summarize_user_sections(
    turns: Sequence[Turn], *, summarize: Callable[[tuple[Turn, ...]], str],
    summarizer_identity: str, max_summary_tokens: int = 128,
) -> tuple[SectionSummary, ...]:
    """Default boundary adapter: one user lead and its following machine turns.

    Source-local input order is preserved even if sources are interleaved. An
    orphan prelude remains a separate section; it is never attached forward.
    An appended assistant turn produces a new snapshot receipt on recompilation.
    """
    groups: list[list[Turn]] = []
    active: dict[str, list[Turn]] = {}
    seen: set[str] = set()
    for turn in turns:
        if not turn.source_id or turn.turn_id in seen:
            raise ValueError("user sections require unique turns with exact sources")
        seen.add(turn.turn_id)
        if turn.role == "user" or turn.source_id not in active:
            group: list[Turn] = []
            groups.append(group)
            active[turn.source_id] = group
        active[turn.source_id].append(turn)
    return tuple(summarize_section(
        "section-" + identity_sha256({"source_id": group[0].source_id,
                                       "lead_turn_id": group[0].turn_id}),
        group, summarize=summarize, summarizer_identity=summarizer_identity,
        max_summary_tokens=max_summary_tokens,
    ) for group in groups)
