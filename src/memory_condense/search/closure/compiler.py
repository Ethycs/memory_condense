"""Deterministic, domain-neutral compilation of evidence obligations.

The compiler is intentionally conservative.  It recognizes a small grammar
of answer intents and emits inspectable routing obligations; it does not
generate facts and nothing it emits may be treated as answer evidence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from memory_condense.domain.discourse import EvidenceObligation, QueryProgram
from memory_condense.domain.text_numbers import NUMBER_WORDS as _NUMBER_WORDS
from memory_condense.search.closure.semantics import (
    CONFLICT_RELATIONS,
    DEPENDENCY_RELATIONS,
    RESOLUTION_RELATIONS,
    REVISION_RELATIONS,
    TEST_RESULT_RELATIONS,
)


_INTENTS = {
    "lookup",
    "enumerate",
    "compare",
    "explain",
    "diagnose",
    "recommend",
    "plan",
    "status",
}
_INTENT_ALIASES = {
    "diagnosis": "diagnose",
    "explanation": "explain",
    "improve": "recommend",
    "improvement": "recommend",
    "planning": "plan",
    "recommendation": "recommend",
}

_TOKEN_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_COUNT_TOKEN = (
    r"\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty"
)
_EXPLICIT_COUNT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"\b(?:list|give|name|identify|enumerate|show)(?:\s+me)?(?:\s+the)?\s+(?P<count>{_COUNT_TOKEN})\b",
        rf"\b(?:top|first|last)\s+(?P<count>{_COUNT_TOKEN})\b",
        rf"\b(?:what|which)\s+are(?:\s+the)?\s+(?P<count>{_COUNT_TOKEN})\b",
        rf"(?<![/\.\w])(?P<count>{_COUNT_TOKEN})\s+(?:items?|entries|members|examples|options|issues|features|results|events|steps)\b",
    )
)

# These words express query grammar, not a content domain.  Removing them
# leaves stable content-bearing terms for deterministic store-side matching.
_QUERY_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "before",
    "best",
    "between",
    "by",
    "can",
    "compare",
    "current",
    "currently",
    "did",
    "do",
    "does",
    "each",
    "explain",
    "for",
    "from",
    "give",
    "had",
    "has",
    "have",
    "how",
    "i",
    "improve",
    "in",
    "is",
    "it",
    "its",
    "latest",
    "list",
    "me",
    "most",
    "my",
    "next",
    "now",
    "of",
    "on",
    "our",
    "plan",
    "please",
    "recommend",
    "should",
    "status",
    "than",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "to",
    "troubleshoot",
    "versus",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
}


_OBJECTIVE_KINDS = (
    "objective",
    "goal",
    "success_criterion",
    "desired_outcome",
    "requirement",
)
_STATE_KINDS = (
    "current_state",
    "state",
    "status",
    "baseline",
    "condition",
)
_CONSTRAINT_KINDS = (
    "constraint",
    "requirement",
    "limitation",
    "policy",
    "preference",
)
_DECISION_KINDS = (
    "decision",
    "choice",
    "action",
    "intervention",
    "commitment",
)
_OBSERVATION_KINDS = (
    "observation",
    "measurement",
    "outcome",
    "result",
    "finding",
    "evidence",
)
_FAILURE_KINDS = (
    "failure",
    "error",
    "counterevidence",
    "adverse_outcome",
    "risk",
    "blocker",
)
_DEPENDENCY_KINDS = (
    "dependency",
    "prerequisite",
    "resource",
    "component",
    "requirement",
)
_ISSUE_KINDS = (
    "unresolved_issue",
    "issue",
    "open_question",
    "alternative",
    "risk",
    "blocker",
)
_REVISION_KINDS = (
    "revision",
    "correction",
    "conflict",
    "contradiction",
    "amendment",
)

# Relation families come from the shared closure ontology in ``semantics``.
# Obligation ``relation_types`` are ordered tuples inside a sealed
# ``QueryProgram``, so each family is frozen into one deterministic sorted
# tuple here rather than re-declared.
_REVISION_RELATIONS = tuple(sorted(REVISION_RELATIONS))
_CONFLICT_RELATIONS = tuple(sorted(CONFLICT_RELATIONS))
_RESOLUTION_RELATIONS = tuple(sorted(RESOLUTION_RELATIONS))
_DEPENDENCY_RELATIONS = tuple(sorted(DEPENDENCY_RELATIONS))
# Compiler-only family: has no counterpart in the shared ontology.
_EXPLANATION_RELATIONS = (
    "causes",
    "explains",
    "supports",
    "qualifies",
    "depends_on",
    "requires",
)
# Walk-time semantics credit "causes" as a test-result relation, but compiled
# obligations deliberately exclude it (open author decision).  The explicit
# subtraction preserves that compiled behavior.
_TEST_RESULT_RELATIONS = tuple(sorted(TEST_RESULT_RELATIONS - {"causes"}))


def compile_query_program(
    query: str | QueryProgram,
    *,
    intent: str | None = None,
    subject_terms: Iterable[str] | None = None,
    as_of_ordinal: int | None = None,
) -> QueryProgram:
    """Compile one query, or return an explicitly supplied manual program.

    Passing a :class:`QueryProgram` as ``query`` is the fail-closed manual
    path used by evaluation fixtures.  Overrides are rejected on that path so
    the program identity cannot be silently changed.
    """

    if isinstance(query, QueryProgram):
        if intent is not None or subject_terms is not None or as_of_ordinal is not None:
            raise ValueError("manual query programs cannot be partially overridden")
        return query

    body = str(query).strip()
    if not body:
        raise ValueError("query must be non-empty")
    normalized_intent = _normalize_intent(intent) if intent else infer_intent(body)
    subjects = (
        _normalize_subject_terms(subject_terms)
        if subject_terms is not None
        else extract_subject_terms(body)
    )
    cardinality = _cardinality(body) if normalized_intent == "enumerate" else None
    ordering = _ordering(body) if normalized_intent == "enumerate" else "none"
    obligations = _compile_obligations(
        normalized_intent,
        subjects,
        cardinality,
        query=body,
    )
    return QueryProgram(
        query=body,
        intent=normalized_intent,
        subject_terms=subjects,
        obligations=obligations,
        as_of_ordinal=as_of_ordinal,
        ordering=ordering,
        cardinality=cardinality,
    )


def infer_intent(query: str) -> str:
    """Infer one of eight bounded intents using deterministic query grammar."""

    text = " ".join(str(query).casefold().split())
    if re.search(
        r"\b(?:recommend|suggest|improv(?:e|ement|ing)|optimi[sz]e|"
        r"what should (?:i|we|they)|better approach|best (?:way|option))\b",
        text,
    ):
        return "recommend"
    if re.search(r"\b(?:diagnos(?:e|is)|troubleshoot|root cause)\b", text) or (
        re.search(r"\bwhy\b", text)
        and re.search(r"\b(?:fail(?:ed|ure)?|error|wrong|broken|problem)\b", text)
    ):
        return "diagnose"
    if re.search(r"\b(?:make|create|draft|develop|outline)\b.{0,32}\bplan\b", text) or re.search(
        r"(?:^\s*plan\b|\bplanning\b|\broadmap\b|\baction plan\b|\bnext steps\b|\bplan for\b)", text
    ):
        return "plan"
    if re.search(
        r"\b(?:status|progress|where (?:do|does|are|is) .{0,24} stand|"
        r"how (?:are|is) .{0,24} doing|current state)\b",
        text,
    ):
        return "status"
    if re.search(r"\b(?:compare|versus|vs\.?|difference between|contrast)\b", text):
        return "compare"
    if re.search(
        r"\b(?:list|enumerate|name|identify|give|show)\b|\b(?:all|each|every)\b|"
        r"\bhow many\b|\bwhat are\b|\bwhich are\b",
        text,
    ):
        return "enumerate"
    if re.search(
        r"\b(?:explain|why|reason|caused?|how (?:did|does|do|can|could|was|were))\b",
        text,
    ):
        return "explain"
    return "lookup"


def extract_subject_terms(query: str, *, max_terms: int = 12) -> tuple[str, ...]:
    """Return stable content terms without depending on a domain vocabulary."""

    if max_terms < 1:
        raise ValueError("max_terms must be positive")
    values: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(str(query).casefold()):
        value = token.strip("-'")
        if len(value) < 2 or value in _QUERY_STOPWORDS or value.isdigit():
            continue
        if value not in seen:
            values.append(value)
            seen.add(value)
        if len(values) >= max_terms:
            break
    return tuple(values)


def _normalize_intent(intent: str) -> str:
    normalized = str(intent).strip().casefold().replace("-", "_").replace(" ", "_")
    normalized = _INTENT_ALIASES.get(normalized, normalized)
    if normalized not in _INTENTS:
        raise ValueError(f"unsupported query intent: {intent!r}")
    return normalized


def _normalize_subject_terms(values: Iterable[str]) -> tuple[str, ...]:
    subjects: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = " ".join(str(item).casefold().split())
        if not value:
            raise ValueError("subject terms must be non-empty")
        if value not in seen:
            subjects.append(value)
            seen.add(value)
    return tuple(subjects)


def _cardinality(query: str) -> int | None:
    for pattern in _EXPLICIT_COUNT_PATTERNS:
        match = pattern.search(query)
        if match is None:
            continue
        raw = match.group("count").casefold()
        value = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw]
        # Very large values are overwhelmingly dates, versions, or identifiers,
        # not a realistic bounded answer cardinality.
        return value if 0 < value <= 100 else None
    return None


def _ordering(query: str) -> str:
    text = query.casefold()
    if re.search(r"\b(?:reverse chronological|descending|latest to earliest)\b", text):
        return "descending"
    if re.search(r"\b(?:chronological|ascending|earliest to latest|in order)\b", text):
        return "ascending"
    return "none"


def _obligation(
    obligation_id: str,
    *,
    kinds: tuple[str, ...],
    subjects: tuple[str, ...],
    relations: tuple[str, ...] = (),
    required: bool = True,
    weight: float = 1.0,
    min_count: int = 1,
    max_count: int | None = None,
    temporal_stance: str = "any",
    dependencies: tuple[str, ...] = (),
) -> EvidenceObligation:
    return EvidenceObligation(
        obligation_id=obligation_id,
        kind=obligation_id,
        required=required,
        weight=weight,
        unit_kinds=kinds,
        relation_types=relations,
        subject_terms=subjects,
        dependencies=dependencies,
        min_count=min_count,
        max_count=max_count,
        temporal_stance=temporal_stance,
    )


def _compile_obligations(
    intent: str,
    subjects: tuple[str, ...],
    cardinality: int | None,
    *,
    query: str,
) -> tuple[EvidenceObligation, ...]:
    if intent == "recommend":
        return (
            _obligation("objective", kinds=_OBJECTIVE_KINDS, subjects=subjects, weight=1.3),
            _obligation("current_state", kinds=_STATE_KINDS, subjects=subjects, weight=1.3, temporal_stance="latest"),
            _obligation("constraints", kinds=_CONSTRAINT_KINDS, subjects=subjects, weight=1.2),
            _obligation("decisions", kinds=_DECISION_KINDS, subjects=subjects, weight=1.1, temporal_stance="terminal"),
            _obligation("observations", kinds=_OBSERVATION_KINDS, subjects=subjects, relations=_TEST_RESULT_RELATIONS, weight=1.2),
            _obligation("failures", kinds=_FAILURE_KINDS, subjects=subjects, relations=_CONFLICT_RELATIONS, weight=1.2),
            _obligation("dependencies", kinds=_DEPENDENCY_KINDS, subjects=subjects, relations=_DEPENDENCY_RELATIONS, weight=1.0),
            _obligation("unresolved_issues", kinds=_ISSUE_KINDS, subjects=subjects, relations=_CONFLICT_RELATIONS, weight=1.1),
            _obligation(
                "revisions_conflicts",
                kinds=_REVISION_KINDS,
                subjects=subjects,
                relations=_REVISION_RELATIONS + _CONFLICT_RELATIONS + _RESOLUTION_RELATIONS,
                weight=1.3,
                temporal_stance="terminal",
            ),
        )
    if intent == "diagnose":
        return (
            _obligation("observations", kinds=_OBSERVATION_KINDS + _STATE_KINDS, subjects=subjects, weight=1.3),
            _obligation("failures", kinds=_FAILURE_KINDS, subjects=subjects, relations=_CONFLICT_RELATIONS, weight=1.4),
            _obligation("causes", kinds=("cause", "hypothesis", "explanation"), subjects=subjects, relations=_EXPLANATION_RELATIONS, weight=1.3),
            _obligation("dependencies", kinds=_DEPENDENCY_KINDS, subjects=subjects, relations=_DEPENDENCY_RELATIONS, weight=1.0),
            _obligation("tests_results", kinds=("test", "experiment") + _OBSERVATION_KINDS, subjects=subjects, relations=_TEST_RESULT_RELATIONS, weight=1.2),
            _obligation("unresolved_issues", kinds=_ISSUE_KINDS, subjects=subjects, relations=_CONFLICT_RELATIONS, required=False),
        )
    if intent == "plan":
        return (
            _obligation("objective", kinds=_OBJECTIVE_KINDS, subjects=subjects, weight=1.3),
            _obligation("current_state", kinds=_STATE_KINDS, subjects=subjects, temporal_stance="latest", weight=1.2),
            _obligation("constraints", kinds=_CONSTRAINT_KINDS, subjects=subjects, weight=1.2),
            _obligation("dependencies", kinds=_DEPENDENCY_KINDS, subjects=subjects, relations=_DEPENDENCY_RELATIONS),
            _obligation("decisions_steps", kinds=_DECISION_KINDS + ("step", "task", "plan"), subjects=subjects, relations=("sequence", "implements", "addresses"), weight=1.2),
            _obligation("unresolved_issues", kinds=_ISSUE_KINDS, subjects=subjects, relations=_CONFLICT_RELATIONS, required=False),
            _obligation("revisions_conflicts", kinds=_REVISION_KINDS, subjects=subjects, relations=_REVISION_RELATIONS + _CONFLICT_RELATIONS + _RESOLUTION_RELATIONS, required=False, temporal_stance="terminal"),
        )
    if intent == "status":
        return (
            _obligation("current_state", kinds=_STATE_KINDS, subjects=subjects, temporal_stance="latest", weight=1.4),
            _obligation("observations", kinds=_OBSERVATION_KINDS, subjects=subjects, relations=_TEST_RESULT_RELATIONS, weight=1.2),
            _obligation("failures", kinds=_FAILURE_KINDS, subjects=subjects, required=False),
            _obligation("unresolved_issues", kinds=_ISSUE_KINDS, subjects=subjects, relations=_CONFLICT_RELATIONS, required=False),
            _obligation("revisions_conflicts", kinds=_REVISION_KINDS, subjects=subjects, relations=_REVISION_RELATIONS + _CONFLICT_RELATIONS + _RESOLUTION_RELATIONS, required=False, temporal_stance="terminal"),
        )
    if intent == "compare":
        return (
            _obligation("comparison_items", kinds=("entity", "event", "option", "state", "fact", "decision", "outcome"), subjects=subjects, min_count=2, weight=1.4),
            _obligation("comparison_basis", kinds=("criterion", "attribute", "constraint", "measurement", "observation"), subjects=subjects, relations=("compares", "contrasts", "qualifies"), weight=1.2),
            _obligation("differences", kinds=("difference", "observation", "result", "finding"), subjects=subjects, relations=("contrasts", "differs_from", "contradicts"), weight=1.2),
        )
    if intent == "explain":
        return (
            _obligation("subject", kinds=("fact", "event", "state", "decision", "outcome", "observation"), subjects=subjects, weight=1.2),
            _obligation("explanation", kinds=("cause", "reason", "rationale", "explanation", "dependency"), subjects=subjects, relations=_EXPLANATION_RELATIONS, weight=1.4),
            _obligation("qualifications", kinds=("constraint", "qualification", "exception", "limitation", "counterevidence"), subjects=subjects, relations=("qualifies", "contradicts"), required=False),
        )
    if intent == "enumerate":
        count = cardinality or 1
        members = _obligation(
            "members",
            kinds=(
                "entity",
                "event",
                "item",
                "fact",
                "decision",
                "observation",
                "outcome",
                "task",
            ),
            subjects=subjects,
            relations=("member_of", "instance_of", "sequence"),
            min_count=count,
            max_count=cardinality,
            weight=1.4,
            temporal_stance="ordered" if cardinality else "any",
        )
        if cardinality is not None:
            return (members,)
        # Open-ended enumeration cannot honestly claim completeness merely
        # because a bounded frontier happened to end.  It needs an evidenced
        # collection/scope boundary in addition to the members themselves.
        return (
            members,
            _obligation(
                "enumeration_scope",
                kinds=("scope_boundary", "collection_boundary", "completeness_receipt"),
                subjects=subjects,
                relations=("exhausts", "covers", "bounds"),
                weight=1.4,
            ),
        )
    return (
        _obligation(
            "answer_fact",
            kinds=("fact", "value", "entity", "event", "state", "decision", "observation", "outcome"),
            subjects=subjects,
            relations=("refers_to", "supports", "produces"),
            weight=1.4,
            temporal_stance=(
                "latest"
                if re.search(r"\b(?:current|latest|now)\b", query, re.IGNORECASE)
                else "any"
            ),
        ),
        # The rule linker deliberately falls back to ``claim``/``question``
        # for ordinary conversation without an ontology cue. Keep that exact
        # evidence eligible for lookup packets, but do not let an unverified
        # claim or a question satisfy the required answer-fact obligation.
        _obligation(
            "lookup_context",
            kinds=("claim", "question"),
            subjects=subjects,
            required=False,
            weight=0.7,
        ),
    )


__all__ = ["compile_query_program", "extract_subject_terms", "infer_intent"]
