"""Gold-blind routing for implicit temporal event-set questions.

The ordinary lexical and dense lanes optimize top-ranked relevance.  Questions
such as "what is the order of the six museums I visited" instead need breadth:
one dated, user-owned event assertion for every member of a set.  This module
contains the small deterministic policy used to open that separate lane.  It
does not search, hydrate, or summarize evidence itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from memory_condense.search.indexes.lexical import tokenize


TEMPORAL_EVENT_LANE_BUDGET = 24

_DATED_HEADER_RE = re.compile(r"^\s*\[[^\]\r\n]*\]\s*")
_QUERY_EVENT_VERB_RE = re.compile(r"\bI\s+([A-Za-z]+ed)\b", re.IGNORECASE)
_LOOKBACK_MONTHS_RE = re.compile(
    r"\bpast\s+(?P<count>\d+|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|eleven|twelve)\s+months?\b",
    re.IGNORECASE,
)
_QUESTION_TIMESTAMP_RE = re.compile(
    r"\[Question asked at\s+(?P<date>\d{4}[/-]\d{1,2}[/-]\d{1,2})"
    r"(?:\s+\([^\]]+?\))?\s+(?P<time>\d{1,2}:\d{2})\]",
    re.IGNORECASE,
)
_FIRST_PERSON_RE = re.compile(r"\b(?:i|we|my|our)\b", re.IGNORECASE)
_ED_VERB_RE = re.compile(r"\b([A-Za-z]+ed)\b", re.IGNORECASE)
_COMPLETED_EVENT_RE = re.compile(
    r"\b(?:attended|visited|went|saw|seen|took|participated|"
    r"got\s+back|came\s+back|been\s+to|enjoyed)\b",
    re.IGNORECASE,
)
_ORDER_TERMS = frozenset({"order"})
_BOUNDARY_TERMS = frozenset({"earliest", "latest", "first", "last"})
_FOCUS_STOP_TERMS = frozenset(
    {
        "order",
        "ordered",
        "earliest",
        "latest",
        "first",
        "last",
        "starting",
        "start",
        "past",
        "present",
        "month",
        "months",
        "week",
        "weeks",
        "day",
        "days",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    }
)
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


@dataclass(frozen=True, slots=True)
class TemporalEnumerationPlan:
    """One query-only decision to open or skip the temporal event lane."""

    active: bool
    body: str
    event_verb: str
    search_terms: tuple[str, ...]
    budget: int
    reason: str
    lookback_months: int | None

    @property
    def search_query(self) -> str:
        return " ".join(self.search_terms)


@dataclass(frozen=True, slots=True)
class EventChunkFeatures:
    """Query-independent eligibility fields compiled beside a chunk address."""

    first_person: bool
    fixed_completed_event: bool
    ed_verbs: tuple[str, ...]

    def admits(self, plan: TemporalEnumerationPlan) -> bool:
        if not plan.active or not self.first_person:
            return False
        return self.fixed_completed_event or plan.event_verb in self.ed_verbs


@dataclass(frozen=True, slots=True)
class TemporalEvidenceWindow:
    """Inclusive wall-clock bounds derived from a dated lookback question."""

    start: datetime
    end: datetime
    lookback_months: int

    def __post_init__(self) -> None:
        if self.start.tzinfo is not None or self.end.tzinfo is not None:
            raise ValueError("temporal evidence bounds must be timezone-naive")
        if self.lookback_months < 1 or self.start > self.end:
            raise ValueError("temporal evidence window has invalid bounds")

    def admits(self, created_at: object) -> bool:
        if not isinstance(created_at, str) or not created_at.strip():
            return False
        try:
            observed = datetime.fromisoformat(
                created_at.strip().replace("Z", "+00:00")
            )
        except ValueError:
            return False
        # The benchmark question date has no timezone. Compare local wall
        # clocks rather than silently converting source offsets to UTC.
        observed = observed.replace(tzinfo=None)
        return self.start <= observed <= self.end

    def model_dump(self) -> dict[str, object]:
        return {
            "start": self.start.isoformat(timespec="minutes"),
            "end": self.end.isoformat(timespec="minutes"),
            "lookback_months": self.lookback_months,
            "inclusive": True,
            "timestamp_policy": "source_local_wall_clock",
        }


def _question_body(query: str) -> str:
    return _DATED_HEADER_RE.sub("", str(query), count=1).strip()


def _expand_focus_term(term: str) -> tuple[str, ...]:
    """Return the stable, intentionally tiny morphology used by the lane."""

    values = [term]
    if len(term) > 3 and term.endswith("ies"):
        values.append(term[:-3] + "y")
    elif len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        values.append(term[:-1])
    if len(term) > 4 and term.endswith("al"):
        values.append(term[:-2])
    return tuple(dict.fromkeys(values))


def _lookback_months(body: str) -> int | None:
    match = _LOOKBACK_MONTHS_RE.search(body)
    if match is None:
        return None
    raw = match.group("count").casefold()
    value = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw]
    return value if value > 0 else None


def _subtract_calendar_months(value: datetime, months: int) -> datetime:
    total = value.year * 12 + (value.month - 1) - months
    year, month_zero = divmod(total, 12)
    month = month_zero + 1
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    last_day = (next_month - datetime.resolution).day
    return value.replace(year=year, month=month, day=min(value.day, last_day))


def resolve_temporal_evidence_window(
    plan: TemporalEnumerationPlan,
    dated_question: str,
) -> TemporalEvidenceWindow | None:
    """Resolve an active plan's lookback against its sealed question date."""

    if not plan.active or plan.lookback_months is None:
        return None
    match = _QUESTION_TIMESTAMP_RE.search(str(dated_question))
    if match is None:
        raise ValueError("dated lookback question omitted its question timestamp")
    normalized = match.group("date").replace("-", "/")
    end = datetime.strptime(
        f"{normalized} {match.group('time')}",
        "%Y/%m/%d %H:%M",
    )
    return TemporalEvidenceWindow(
        start=_subtract_calendar_months(end, plan.lookback_months),
        end=end,
        lookback_months=plan.lookback_months,
    )


def plan_temporal_enumeration(query: str) -> TemporalEnumerationPlan:
    """Derive the fixed event-set lane from the question alone.

    The narrow trigger is deliberate.  It recognizes implicit ordered-set
    questions while leaving explicit operand lists and ordinary fact questions
    on the protected base route.
    """

    body = _question_body(query)
    lookback_months = _lookback_months(body)
    terms = tuple(tokenize(body))
    event_match = _QUERY_EVENT_VERB_RE.search(body)
    active = bool(
        not ":" in body
        and event_match is not None
        and _ORDER_TERMS.intersection(terms)
        and _BOUNDARY_TERMS.intersection(terms)
    )
    if not active:
        return TemporalEnumerationPlan(
            active=False,
            body=body,
            event_verb="",
            search_terms=(),
            budget=0,
            reason="not_implicit_temporal_event_set",
            lookback_months=lookback_months,
        )

    assert event_match is not None
    event_verb = event_match.group(1).casefold()
    focus = body[: event_match.start()]
    expanded: list[str] = []
    for term in tokenize(focus):
        if term in _FOCUS_STOP_TERMS:
            continue
        expanded.extend(_expand_focus_term(term))
    search_terms = tuple(dict.fromkeys(expanded))
    if not search_terms:
        return TemporalEnumerationPlan(
            active=False,
            body=body,
            event_verb=event_verb,
            search_terms=(),
            budget=0,
            reason="empty_event_focus",
            lookback_months=lookback_months,
        )
    return TemporalEnumerationPlan(
        active=True,
        body=body,
        event_verb=event_verb,
        search_terms=search_terms,
        budget=TEMPORAL_EVENT_LANE_BUDGET,
        reason="implicit_temporal_event_set",
        lookback_months=lookback_months,
    )


def compile_event_chunk_features(*, role: str, text: str) -> EventChunkFeatures:
    """Compile the tiny query-independent user-event eligibility sidecar."""

    if str(role).strip().casefold() != "user":
        return EventChunkFeatures(False, False, ())
    raw = str(text)
    first_person = _FIRST_PERSON_RE.search(raw) is not None
    if not first_person:
        return EventChunkFeatures(False, False, ())
    verbs = tuple(dict.fromkeys(match.casefold() for match in _ED_VERB_RE.findall(raw)))
    return EventChunkFeatures(
        first_person=True,
        fixed_completed_event=_COMPLETED_EVENT_RE.search(raw) is not None,
        ed_verbs=verbs,
    )


__all__ = [
    "EventChunkFeatures",
    "TEMPORAL_EVENT_LANE_BUDGET",
    "TemporalEvidenceWindow",
    "TemporalEnumerationPlan",
    "compile_event_chunk_features",
    "plan_temporal_enumeration",
    "resolve_temporal_evidence_window",
]
