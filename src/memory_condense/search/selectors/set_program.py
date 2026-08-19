"""Deterministic query-to-set-program compilation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

class SetOperator(str, Enum):
    """Deterministic answer-set operation requested by the query."""

    SINGLE = "single"
    ALL = "all"
    COUNT = "count"
    FIXED = "fixed_cardinality"
    ORDERED = "ordered"
    EARLIEST = "earliest"
    LATEST = "latest"


class SetQuantifier(str, Enum):
    """How many answer members the deterministic reducer must retain."""

    SINGLE = "single"
    ALL = "all"
    COUNT = "count"
    FIXED = "fixed_cardinality"


class SetOrdering(str, Enum):
    """Independent ordering clause in a compositional set query."""

    NONE = "none"
    ASCENDING = "ascending"
    DESCENDING = "descending"


@dataclass(frozen=True, slots=True)
class SetProgram:
    """Small, inspectable query program used by the neural classifier."""

    operator: SetOperator
    cardinality: int | None
    requires_completeness: bool
    identity_rule: str
    quantifier: SetQuantifier = SetQuantifier.SINGLE
    ordering: SetOrdering = SetOrdering.NONE
    preferred_evidence_role: str | None = None
    # ``preferred_evidence_role`` is a broad, soft ranking hint.  These two
    # fields are intentionally narrower: only explicit retrospective
    # attribution may authorize role-aligned FIXED-K reservation.
    required_evidence_role: str | None = None
    required_evidence_role_basis: str | None = None
    query_timestamp: str | None = None
    temporal_window_days: int | None = None


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
_DATED_QUESTION_RE = re.compile(
    r"^\[Question asked at (?P<asked_at>.+?)\]\s*",
    re.DOTALL,
)
_COUNT_PATTERN = r"\d+|" + "|".join(_NUMBER_WORDS)
_FIXED_CARDINALITY_RE = re.compile(
    r"(?:\b(?:name|list|give|identify|which|what\s+are|order|ordering|arrange)\b"
    r"[^?.!]{0,48}?\b(?:the\s+)?(?P<command_count>"
    + _COUNT_PATTERN
    + r")\s+(?:museums?|concerts?|events?|places?|items?|visits?)\b|"
    + r"\b(?:the\s+)?(?P<bare_count>"
    + _COUNT_PATTERN
    + r")\s+(?:museums?|concerts?|events?|places?|items?|visits?)\b)",
    re.IGNORECASE,
)

# ``how many`` is ambiguous: it can request the cardinality of a memory set
# (``how many museums did I visit?``), or a scalar derived from a small number
# of facts (``how many pages are left?`` / ``how many days before X was Y?``).
# Only the former needs an exhaustive coverage frontier.  Treating the latter
# as COUNT causes the packer to reserve a fragment for every transient cluster,
# which both wastes the packet and can cut the operands out of otherwise-correct
# chunks.  Keep this deliberately narrow: a unit plus a relational cue is a
# derived scalar; an unqualified ``how many <objects>`` remains a set count.
_DERIVED_SCALAR_COUNT_RE = re.compile(
    r"\b(?:how many|number of)\s+"
    r"(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?|pages?)\b"
    r"[^?.!]{0,96}\b(?:before|after|since|until|when|left|remaining)\b",
    re.IGNORECASE,
)

# A present-tense possession question with an explicit current-state deictic
# asks for one scalar observation, not the cardinality of every matching memory
# event.  Keep the grammar intentionally narrow: it must be the whole question,
# use first-person ``do I have``, and say ``now``/``currently``/``at present``.
# Historical, third-person, non-deictic, compound, and obligation (``have to``)
# questions continue through the ordinary set compiler.
_FIRST_PERSON_POSSESSION_COUNT_RE = re.compile(
    r"^how many\s+(?!of\b)[^?.!\n]{1,64}?\s+do\s+i\s+"
    r"(?P<pre_deictic>currently\s+|presently\s+)?have\b"
    r"(?P<tail>[^?.!\n]*)\??$",
    re.IGNORECASE,
)
_CURRENT_STATE_DEICTIC_RE = re.compile(
    r"\b(?:right\s+now|now|currently|presently|at\s+present|"
    r"at\s+the\s+moment)\b",
    re.IGNORECASE,
)
_NON_SCALAR_CURRENT_POSSESSION_RE = re.compile(
    r"^\s*to\b|\b(?:and|or)\s+how many\b|"
    r"\b(?:compared\s+(?:with|to)|versus|vs\.?|before|after|than)\b",
    re.IGNORECASE,
)


def _is_first_person_current_possessed_scalar(query: str) -> bool:
    """Return whether *query* asks for one explicitly current owned value."""

    match = _FIRST_PERSON_POSSESSION_COUNT_RE.fullmatch(query.strip())
    if match is None:
        return False
    tail = match.group("tail") or ""
    if _NON_SCALAR_CURRENT_POSSESSION_RE.search(tail):
        return False
    return bool(
        match.group("pre_deictic") or _CURRENT_STATE_DEICTIC_RE.search(tail)
    )


_RETROSPECTIVE_ASSISTANT_ROLE_RE = re.compile(
    r"(?:\b(?:what|which|where|when|who|how)\b[^?.!\n]{0,80}?"
    r"\bdid\s+(?:you|the\s+assistant)\s+"
    r"(?:say|provide|give|mention|recommend|suggest|tell|advise|list|share)\b|"
    r"\b(?:you|the\s+assistant)\s+(?:had\s+)?"
    r"(?:said|provided|gave|mentioned|recommended|suggested|told|advised|"
    r"listed|shared)\b[^?.!\n]{0,64}?"
    r"\b(?:earlier|before|previously|last\s+time)\b|"
    r"\b(?:earlier|previously|last\s+time)\b[^?.!\n]{0,64}?"
    r"\b(?:you|the\s+assistant)\s+(?:had\s+)?"
    r"(?:said|provided|gave|mentioned|recommended|suggested|told|advised|"
    r"listed|shared)\b)",
    re.IGNORECASE,
)

# Keep this vocabulary intentionally concrete.  A generic first-person
# pronoun ("what should I ...") is not evidence that the answer was authored
# by the user.  These are completed actions commonly used to refer back to a
# remembered episode; regular expansions should be added only with tests that
# distinguish them from current requests.
_FIRST_PERSON_PAST_ACTION_RE = re.compile(
    r"\b(?:i|we)\s+(?:had\s+|just\s+)?(?:"
    r"attended|bought|completed|created|did|drove|flew|gave|helped|joined|"
    r"left|made|met|moved|ordered|participated|picked|played|prepared|"
    r"said|saw|sent|started|stayed|told|took|traveled|travelled|visited|"
    r"watched|went|worked|wrote"
    r")\b",
    re.IGNORECASE,
)


def _required_evidence_role(query: str) -> tuple[str | None, str | None]:
    """Return a role only for explicit retrospective authorship evidence.

    This must remain stricter than the soft preferred-role heuristic.  In
    particular, current requests such as ``can you recommend three places``
    and ambiguous first-person needs must abstain instead of hard-gating a
    source role.
    """

    if _RETROSPECTIVE_ASSISTANT_ROLE_RE.search(query):
        return "assistant", "explicit_retrospective_assistant_attribution"
    if _FIRST_PERSON_PAST_ACTION_RE.search(query):
        return "user", "explicit_first_person_past_action"
    return None, None


def compile_set_program(query: str) -> SetProgram:
    """Compile explicit quantifier/order language without guessing answers."""

    stripped = query.strip()
    dated_match = _DATED_QUESTION_RE.match(stripped)
    query_timestamp = (
        dated_match.group("asked_at").strip() if dated_match is not None else None
    )
    body = _DATED_QUESTION_RE.sub("", stripped)
    lowered = body.casefold()
    fixed_match = _FIXED_CARDINALITY_RE.search(lowered)
    cardinality: int | None = None
    if fixed_match is not None:
        raw_count = fixed_match.group("command_count") or fixed_match.group(
            "bare_count"
        )
        cardinality = (
            int(raw_count) if raw_count.isdigit() else _NUMBER_WORDS[raw_count]
        )

    derived_scalar_count = bool(_DERIVED_SCALAR_COUNT_RE.search(lowered))
    current_possessed_scalar = _is_first_person_current_possessed_scalar(lowered)
    count_query = bool(
        re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", lowered)
    ) and not (derived_scalar_count or current_possessed_scalar)
    explicit_all = bool(re.search(r"\ball\b|\beach\b|\bevery\b", lowered))
    set_order = bool(
        re.search(
            r"\bchronological(?:ly)?\b|\bin\s+(?:chronological\s+)?order\b|"
            r"\border\s+of\b|\bsequence\b|\bstarting\s+(?:with|from)\b|"
            r"\bfrom\s+(?:the\s+)?(?:earliest|oldest|latest|newest)\b|"
            r"\b(?:earliest|oldest|latest|newest)\s+to\s+"
            r"(?:latest|newest|earliest|oldest)\b",
            lowered,
        )
    )
    descending = bool(
        re.search(
            r"\b(?:latest|newest)\s+to\s+(?:earliest|oldest)\b|"
            r"\b(?:reverse\s+chronological|descending)\b|"
            r"\bstarting\s+(?:with|from)\s+(?:the\s+)?(?:latest|newest)\b",
            lowered,
        )
    )
    ascending = bool(
        re.search(
            r"\b(?:earliest|oldest)\s+to\s+(?:latest|newest)\b|"
            r"\bchronological(?:ly)?\b|\bascending\b|"
            r"\bstarting\s+(?:with|from)\s+(?:the\s+)?(?:earliest|oldest|first)\b",
            lowered,
        )
    )
    earliest = bool(re.search(r"\bearliest\b|\boldest\b|\bfirst\b(?!\s+name\b)", lowered))
    latest = bool(re.search(r"\blatest\b|\bmost recent\b|\bnewest\b", lowered))

    if descending:
        ordering = SetOrdering.DESCENDING
    elif ascending or set_order or earliest:
        ordering = SetOrdering.ASCENDING
    elif latest:
        ordering = SetOrdering.DESCENDING
    else:
        ordering = SetOrdering.NONE

    if count_query:
        quantifier = SetQuantifier.COUNT
    elif cardinality is not None:
        quantifier = SetQuantifier.FIXED
    elif explicit_all or set_order:
        # Definite plural ordering ("order of the concerts", "concerts in
        # chronological order") requests the complete set even when "all" is
        # omitted.  This is the LongMemEval ordered-all form.
        quantifier = SetQuantifier.ALL
    else:
        quantifier = SetQuantifier.SINGLE

    if quantifier is SetQuantifier.COUNT:
        operator = SetOperator.COUNT
    elif quantifier is SetQuantifier.FIXED:
        operator = SetOperator.FIXED
    elif quantifier is SetQuantifier.ALL and ordering is not SetOrdering.NONE:
        operator = SetOperator.ORDERED
    elif quantifier is SetQuantifier.ALL:
        operator = SetOperator.ALL
    elif ordering is SetOrdering.ASCENDING:
        operator = SetOperator.EARLIEST
    elif ordering is SetOrdering.DESCENDING:
        operator = SetOperator.LATEST
    else:
        operator = SetOperator.SINGLE

    if "museum" in lowered:
        identity_rule = (
            "Group by canonical museum venue. Different excerpts about the same "
            "venue are one answer member; different venues are distinct."
        )
    elif "concert" in lowered or "performance" in lowered:
        identity_rule = (
            "Group by performance occurrence, not merely artist or topic. "
            "Different dates or venues are distinct occurrences."
        )
    else:
        identity_rule = (
            "Use the identity relation implied by the requested answer: merge "
            "paraphrases of one object/event, but keep distinct occurrences, "
            "dates, venues, or answer values separate when the query requires them."
        )

    assistant_request = bool(
        re.search(
            r"\b(?:you|assistant)\b[^?.!]{0,36}\b"
            r"(?:recommend|suggest|tell|told|advise|said|mention)",
            lowered,
        )
        or re.search(
            r"\bwhat\b[^?.!]{0,20}\b(?:recommend|suggest|tell|advise)\w*\b",
            lowered,
        )
    )
    autobiographical_request = bool(re.search(r"\b(?:i|me|my|mine)\b", lowered))
    preferred_evidence_role = (
        "assistant"
        if assistant_request
        else "user"
        if autobiographical_request
        else None
    )
    required_evidence_role, required_evidence_role_basis = (
        _required_evidence_role(lowered)
    )
    temporal_window_days: int | None = None
    temporal_window = re.search(
        r"\b(?:past|last|previous)\s+"
        r"(?P<count>an?|\d+|"
        + "|".join(_NUMBER_WORDS)
        + r")?\s*(?P<unit>days?|weeks?|months?|years?)\b",
        lowered,
    )
    if temporal_window is not None:
        raw_count = temporal_window.group("count") or "one"
        count = (
            1
            if raw_count in {"a", "an"}
            else int(raw_count)
            if raw_count.isdigit()
            else _NUMBER_WORDS[raw_count]
        )
        unit = temporal_window.group("unit")
        unit_days = (
            1
            if unit.startswith("day")
            else 7
            if unit.startswith("week")
            else 31
            if unit.startswith("month")
            else 366
        )
        temporal_window_days = count * unit_days

    return SetProgram(
        operator=operator,
        cardinality=cardinality,
        requires_completeness=(
            quantifier is not SetQuantifier.SINGLE
            or ordering is not SetOrdering.NONE
        ),
        identity_rule=identity_rule,
        quantifier=quantifier,
        ordering=ordering,
        preferred_evidence_role=preferred_evidence_role,
        required_evidence_role=required_evidence_role,
        required_evidence_role_basis=required_evidence_role_basis,
        query_timestamp=query_timestamp,
        temporal_window_days=temporal_window_days,
    )
