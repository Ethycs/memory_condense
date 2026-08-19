"""Conservative evidence filtering for derived scalar questions.

The coverage selector is intentionally bypassed for ordinary scalar questions.
That restores the normal information-gain packet, but can also expose an
approximate duration recap beside the two provenance-bearing boundary events
from which the requested duration should be derived.  This module removes only
the narrow case where those three claims are jointly sufficient to prove that
the approximate recap is inconsistent with an explicit onset.

No answer is computed here, no excerpt is rewritten, and no text is retained in
the returned diagnostics.  Callers receive the exact ``RetrievalResult``
objects they supplied, in their original order, minus proven conflicts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

from memory_condense.search.indexes.lexical import tokenize
from memory_condense.domain.schemas import RetrievalResult

__all__ = [
    "TemporalConflictDecision",
    "filter_conflicting_approximate_duration_recaps",
]


@dataclass(frozen=True, slots=True)
class TemporalConflictDecision:
    """Text-free provenance for one conservatively suppressed recap."""

    reason: str
    onset_chunk_id: str
    endpoint_chunk_id: str


@dataclass(frozen=True, slots=True)
class _DerivedDurationSpec:
    activity_terms: frozenset[str]
    endpoint_terms: frozenset[str]


@dataclass(frozen=True, slots=True)
class _TimedEvidence:
    result: RetrievalResult
    timestamp: float


_QUERY_PREFIX_RE = re.compile(
    r"^\[Question asked at .+?\]\s*",
    re.IGNORECASE | re.DOTALL,
)
_DERIVED_DURATION_RE = re.compile(
    r"\b(?:for\s+)?how\s+(?:"
    r"many\s+(?:minutes?|hours?|days?|weeks?)|long"
    r")\b"
    r"(?P<activity>.{1,180}?)"
    r"\b(?:when|by\s+the\s+time|at\s+the\s+time)\b"
    r"(?P<endpoint>.{1,180}?)(?:[?.]|$)",
    re.IGNORECASE | re.DOTALL,
)
_DIRECT_REPORT_RE = re.compile(
    r"\b(?:say|said|mention(?:ed)?|report(?:ed)?|tell|told)\b",
    re.IGNORECASE,
)
_ONSET_RE = re.compile(
    r"\b(?:i|we)\s+(?:had\s+)?(?:just\s+)?"
    r"(?:started|began|commenced)\b",
    re.IGNORECASE,
)
_COMPLETED_ENDPOINT_RE = re.compile(
    r"\b(?:i|we)\s+(?:actually\s+)?(?:"
    r"got|bought|purchased|invested|completed|finished|attended|received"
    r")\b",
    re.IGNORECASE,
)
_DEICTIC_RE = re.compile(r"\b(?P<when>today|yesterday)\b", re.IGNORECASE)
_RESTART_RE = re.compile(
    r"\b(?:again|restart(?:ed|ing)?|resum(?:ed|ing)|returned|back\s+to)\b",
    re.IGNORECASE,
)
_NONFACTUAL_RE = re.compile(
    r"\b(?:if|might|may|could|would|plan(?:ned|ning)?|intend(?:ed|ing)?)\b",
    re.IGNORECASE,
)
_NEGATED_BOUNDARY_RE = re.compile(
    r"\b(?:did\s+not|didn't|never)\b[^.!?]{0,32}"
    r"\b(?:start(?:ed)?|begin|began|buy|bought|purchase|invest)\b",
    re.IGNORECASE,
)
_CORRECTION_RE = re.compile(
    r"\b(?:correction|i\s+was\s+wrong|to\s+clarify|actually)\b",
    re.IGNORECASE,
)
_APPROXIMATE_DURATION_RE = re.compile(
    r"\b(?:i(?:'ve|\s+have|'d|\s+had)|"
    r"we(?:'ve|\s+have|'d|\s+had))\s+been\b"
    r"[^.!?]{0,240}?"
    r"\b(?:for\s+)?(?:about|around|roughly|approximately)\s+"
    r"(?P<count>\d+(?:\.\d+)?|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
    r"\s*(?P<unit>minutes?|hours?|days?|weeks?)\b",
    re.IGNORECASE,
)
_TIMESTAMP_RE = re.compile(
    r"\b(?P<year>(?:19|20)\d{2})[/-](?P<month>\d{1,2})"
    r"[/-](?P<day>\d{1,2})(?:\D+(?P<hour>\d{1,2})"
    r":(?P<minute>\d{2}))?"
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")

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
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_UNIT_SECONDS = {
    "minute": 60.0,
    "hour": 60.0 * 60.0,
    "day": 24.0 * 60.0 * 60.0,
    "week": 7.0 * 24.0 * 60.0 * 60.0,
}
_CONFLICT_REASON = "approximate_duration_conflicts_with_explicit_onset"


def filter_conflicting_approximate_duration_recaps(
    candidates: Sequence[RetrievalResult],
    *,
    query: str,
    source_timestamps: Mapping[str, str],
) -> tuple[list[RetrievalResult], dict[str, TemporalConflictDecision]]:
    """Suppress only approximate duration recaps contradicted by boundaries.

    The function fails open unless the query requests a duration at a distinct
    endpoint and the candidate packet contains one unambiguous, user-authored
    onset and endpoint.  Exact duration claims, corrections, restarts,
    conditionals, missing timestamps, and ambiguous boundaries are retained.
    """

    original = list(candidates)
    spec = _parse_derived_duration_query(query)
    if spec is None or not original:
        return original, {}

    onsets = _find_onsets(original, spec, source_timestamps)
    endpoints = _find_endpoints(original, spec, source_timestamps)
    onset = _one_timestamp(onsets)
    endpoint = _one_timestamp(endpoints)
    if onset is None or endpoint is None or endpoint.timestamp <= onset.timestamp:
        return original, {}

    anchor_ids = {onset.result.chunk.chunk_id, endpoint.result.chunk.chunk_id}
    decisions: dict[str, TemporalConflictDecision] = {}
    retained: list[RetrievalResult] = []
    for result in original:
        chunk_id = result.chunk.chunk_id
        if chunk_id in anchor_ids or not _is_user_evidence(result):
            retained.append(result)
            continue
        recap_timestamp = _source_timestamp(result, source_timestamps)
        if recap_timestamp is None:
            retained.append(result)
            continue
        claims = _approximate_duration_claims(result.chunk.text, spec)
        if len(claims) != 1:
            retained.append(result)
            continue
        claimed_seconds = claims[0]
        observed_seconds = recap_timestamp - onset.timestamp
        if observed_seconds < 0.0:
            retained.append(result)
            continue
        unit_seconds, duration_seconds = claimed_seconds
        tolerance = max(unit_seconds, 0.25 * duration_seconds)
        if abs(observed_seconds - duration_seconds) <= tolerance:
            retained.append(result)
            continue
        decisions[chunk_id] = TemporalConflictDecision(
            reason=_CONFLICT_REASON,
            onset_chunk_id=onset.result.chunk.chunk_id,
            endpoint_chunk_id=endpoint.result.chunk.chunk_id,
        )

    if not decisions:
        return original, {}
    retained = [
        result
        for result in original
        if result.chunk.chunk_id not in decisions
    ]
    return retained, decisions


def _parse_derived_duration_query(query: str) -> _DerivedDurationSpec | None:
    body = _QUERY_PREFIX_RE.sub("", query.strip())
    if _DIRECT_REPORT_RE.search(body) is not None:
        return None
    match = _DERIVED_DURATION_RE.search(body)
    if match is None:
        return None
    activity_terms = frozenset(tokenize(match.group("activity")))
    endpoint_terms = frozenset(tokenize(match.group("endpoint")))
    if not activity_terms or not endpoint_terms:
        return None
    return _DerivedDurationSpec(
        activity_terms=activity_terms,
        endpoint_terms=endpoint_terms,
    )


def _find_onsets(
    candidates: Sequence[RetrievalResult],
    spec: _DerivedDurationSpec,
    source_timestamps: Mapping[str, str],
) -> list[_TimedEvidence]:
    found: list[_TimedEvidence] = []
    for result in candidates:
        if not _is_user_evidence(result):
            continue
        source_timestamp = _source_timestamp(result, source_timestamps)
        if source_timestamp is None:
            continue
        for sentence in _sentences(result.chunk.text):
            if (
                _ONSET_RE.search(sentence) is None
                or not _term_match(sentence, spec.activity_terms)
                or _RESTART_RE.search(sentence) is not None
                or _NONFACTUAL_RE.search(sentence) is not None
                or _NEGATED_BOUNDARY_RE.search(sentence) is not None
                or re.search(r"\bago\b", sentence, re.IGNORECASE) is not None
            ):
                continue
            timestamp = _event_timestamp(sentence, source_timestamp)
            if timestamp is not None:
                found.append(_TimedEvidence(result=result, timestamp=timestamp))
                break
    return found


def _find_endpoints(
    candidates: Sequence[RetrievalResult],
    spec: _DerivedDurationSpec,
    source_timestamps: Mapping[str, str],
) -> list[_TimedEvidence]:
    found: list[_TimedEvidence] = []
    for result in candidates:
        if not _is_user_evidence(result):
            continue
        source_timestamp = _source_timestamp(result, source_timestamps)
        if source_timestamp is None:
            continue
        for sentence in _sentences(result.chunk.text):
            if (
                _COMPLETED_ENDPOINT_RE.search(sentence) is None
                or not _term_match(sentence, spec.endpoint_terms)
                or _NONFACTUAL_RE.search(sentence) is not None
                or _NEGATED_BOUNDARY_RE.search(sentence) is not None
            ):
                continue
            timestamp = _event_timestamp(sentence, source_timestamp)
            if timestamp is not None:
                found.append(_TimedEvidence(result=result, timestamp=timestamp))
                break
    return found


def _approximate_duration_claims(
    text: str,
    spec: _DerivedDurationSpec,
) -> list[tuple[float, float]]:
    claims: list[tuple[float, float]] = []
    for sentence in _sentences(text):
        if (
            _CORRECTION_RE.search(sentence) is not None
            or not _term_match(sentence, spec.activity_terms)
        ):
            continue
        for match in _APPROXIMATE_DURATION_RE.finditer(sentence):
            raw_count = match.group("count").casefold()
            count = (
                float(raw_count)
                if re.fullmatch(r"\d+(?:\.\d+)?", raw_count)
                else float(_NUMBER_WORDS[raw_count])
            )
            if count <= 0.0:
                continue
            unit = match.group("unit").casefold().rstrip("s")
            unit_seconds = _UNIT_SECONDS.get(unit)
            if unit_seconds is not None:
                claims.append((unit_seconds, count * unit_seconds))
    return claims


def _one_timestamp(values: Sequence[_TimedEvidence]) -> _TimedEvidence | None:
    if not values:
        return None
    timestamps = {round(value.timestamp, 6) for value in values}
    if len(timestamps) != 1:
        return None
    # Stable upstream order is the deterministic tie-break for duplicate
    # descriptions of the same boundary.
    return values[0]


def _is_user_evidence(result: RetrievalResult) -> bool:
    return bool(
        result.turn is not None
        and result.turn.role.strip().casefold() == "user"
    )


def _source_id(result: RetrievalResult) -> str:
    if result.memory_source_id:
        return result.memory_source_id
    if result.turn is not None:
        return str(result.turn.source_id or result.turn.turn_id)
    return result.chunk.turn_id


def _source_timestamp(
    result: RetrievalResult,
    source_timestamps: Mapping[str, str],
) -> float | None:
    return _parse_timestamp(source_timestamps.get(_source_id(result)))


def _parse_timestamp(value: str | None) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        match = _TIMESTAMP_RE.search(cleaned)
        if match is None:
            return None
        try:
            return datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
                int(match.group("hour") or 0),
                int(match.group("minute") or 0),
                tzinfo=timezone.utc,
            ).timestamp()
        except (OverflowError, ValueError):
            return None


def _event_timestamp(sentence: str, source_timestamp: float) -> float | None:
    explicit = _parse_timestamp(sentence)
    if explicit is not None:
        return explicit
    deictic = _DEICTIC_RE.search(sentence)
    if deictic is None:
        return None
    if deictic.group("when").casefold() == "yesterday":
        return source_timestamp - _UNIT_SECONDS["day"]
    return source_timestamp


def _term_match(sentence: str, target: frozenset[str]) -> bool:
    overlap = target.intersection(tokenize(sentence))
    required = 1 if len(target) == 1 else 2
    return len(overlap) >= required


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY_RE.split(text.strip())
        if sentence.strip()
    ]
