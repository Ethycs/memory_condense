"""Hydrate only summary-selected sections, preserving exact raw span bytes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Turn
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.section_routing import SectionRoutePlan
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary, bound_int


@dataclass(frozen=True, slots=True)
class HydratedSectionSpan(SealedIdentity):
    span: RawSectionSpan
    text: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if quote_sha256(self.text) != self.span.span_text_sha256:
            raise ValueError("hydrated raw span hash changed")
        if len(self.text) != self.span.end_char - self.span.start_char:
            raise ValueError("hydrated raw span length changed")
        if count_tokens(self.text) != self.span.token_count:
            raise ValueError("hydrated raw span token count changed")
        self._seal()


@dataclass(frozen=True, slots=True)
class HydratedSection(SealedIdentity):
    section: SectionSummary
    evidence: tuple[HydratedSectionSpan, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if tuple(row.span for row in self.evidence) != self.section.spans:
            raise ValueError("hydration must contain the entire exact section in order")
        self._seal()

    def render_raw(self, label: str) -> str:
        # The labels are prompt-local. Exact section/source IDs and generated
        # summaries stay in the audit objects; only raw evidence goes to readers.
        lines = [f"<SECTION {label}>"]
        # Atoms are compilation details. Rejoin adjacent same-turn slices before
        # framing so an attention cut never inserts whitespace inside raw text.
        groups: list[tuple[RawSectionSpan, str]] = []
        for row in self.evidence:
            if groups and groups[-1][0].turn_id == row.span.turn_id:
                first, text = groups[-1]
                groups[-1] = (first, text + row.text)
            else:
                groups.append((row.span, row.text))
        for ordinal, (span, text) in enumerate(groups, 1):
            lines.append(f"<{label}.{ordinal} [{span.created_at} | {span.role}]>\n{text}")
        lines.append(f"</SECTION {label}>")
        return "\n".join(lines)


def _render(sections: tuple[HydratedSection, ...]) -> str:
    return "\n\n".join(section.render_raw(f"S{i}") for i, section in enumerate(sections, 1))


@dataclass(frozen=True, slots=True)
class SectionHydrationDiagnostic:
    section_id: str
    reason: str

    def identity_payload(self) -> dict:
        return {"section_id": self.section_id, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class SectionRetrievalResult(SealedIdentity):
    plan: SectionRoutePlan
    sections: tuple[HydratedSection, ...]
    diagnostics: tuple[SectionHydrationDiagnostic, ...]
    raw_turn_read_count: int
    max_raw_spans: int
    max_context_tokens: int
    context_token_count: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        bound_int(self.raw_turn_read_count, "raw_turn_read_count")
        bound_int(self.max_raw_spans, "max_raw_spans")
        bound_int(self.max_context_tokens, "max_context_tokens")
        if sum(len(section.evidence) for section in self.sections) > self.max_raw_spans:
            raise ValueError("hydrated sections exceed the raw span budget")
        if count_tokens(_render(self.sections)) != self.context_token_count or (
            self.context_token_count > self.max_context_tokens
        ):
            raise ValueError("hydrated section context exceeds or misstates its budget")
        routes = {route.section.section_id: route.section for route in self.plan.routes}
        if any(routes.get(section.section.section_id) != section.section for section in self.sections):
            raise ValueError("hydration escaped the selected summary sections")
        admitted = [section.section.section_id for section in self.sections]
        rejected = [row.section_id for row in self.diagnostics]
        if len(set(admitted + rejected)) != len(admitted + rejected) or set(admitted + rejected) != set(routes):
            raise ValueError("every selected section needs one hydration outcome")
        self._seal()

    @property
    def requires_raw_fallback(self) -> bool:
        """A missing/invalid/over-budget route cannot establish evidence absence."""
        return not self.sections or bool(self.diagnostics)

    def render_context(self) -> str:
        """Return the budgeted raw-only packet; never render routing summaries."""
        return _render(self.sections)


def hydrate_section_plan(
    plan: SectionRoutePlan, *, load_turn: Callable[[str], Turn | None],
    max_raw_spans: int = 32, max_context_tokens: int = 4096,
) -> SectionRetrievalResult:
    """Exact, atomic hydration after summary selection, with bounded raw I/O.

    Missing/stale/foreign text rejects the whole affected section. A context
    overflow never silently truncates a selected raw section. Remaining selected
    sections can still fit; diagnostics tell the caller to retain its raw lane.
    """
    if type(plan) is not SectionRoutePlan:
        raise TypeError("plan must be a SectionRoutePlan")
    bound_int(max_raw_spans, "max_raw_spans")
    bound_int(max_context_tokens, "max_context_tokens")
    cache: dict[str, Turn | None] = {}
    failed_reads: set[str] = set()
    sections: list[HydratedSection] = []
    diagnostics: list[SectionHydrationDiagnostic] = []
    read_count = 0
    admitted_spans = 0
    attempted_spans = 0
    for route in plan.routes:
        section = route.section
        reason = None
        if attempted_spans + len(section.spans) > max_raw_spans:
            reason = "raw_span_budget"
        elif sum(span.token_count for span in section.spans) > max_context_tokens:
            reason = "context_budget"
        if reason is not None:
            diagnostics.append(SectionHydrationDiagnostic(section.section_id, reason))
            continue
        attempted_spans += len(section.spans)
        prepared: list[HydratedSectionSpan] = []
        for span in section.spans:
            if span.turn_id not in cache:
                read_count += 1
                try:
                    cache[span.turn_id] = load_turn(span.turn_id)
                except Exception:
                    cache[span.turn_id] = None
                    failed_reads.add(span.turn_id)
            turn = cache[span.turn_id]
            if turn is None:
                reason = "raw_read_failed" if span.turn_id in failed_reads else "raw_turn_missing"
                break
            if not isinstance(turn, Turn) or (
                turn.turn_id, turn.source_id, turn.role, turn.created_at.isoformat(), quote_sha256(turn.text)
            ) != (span.turn_id, span.source_id, span.role, span.created_at, span.turn_text_sha256):
                reason = "raw_turn_identity_changed"
                break
            if span.end_char > len(turn.text):
                reason = "raw_span_bounds_changed"
                break
            try:
                prepared.append(HydratedSectionSpan(span, turn.text[span.start_char:span.end_char]))
            except ValueError:
                reason = "raw_span_identity_changed"
                break
        if reason is None:
            hydrated = HydratedSection(section, tuple(prepared))
            proposed = tuple([*sections, hydrated])
            if admitted_spans + len(prepared) > max_raw_spans or count_tokens(_render(proposed)) > max_context_tokens:
                reason = "context_budget"
            else:
                sections.append(hydrated)
                admitted_spans += len(prepared)
        if reason is not None:
            diagnostics.append(SectionHydrationDiagnostic(section.section_id, reason))
    result = tuple(sections)
    return SectionRetrievalResult(plan, result, tuple(diagnostics), read_count,
                                  max_raw_spans, max_context_tokens, count_tokens(_render(result)))
