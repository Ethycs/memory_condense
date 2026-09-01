"""Conservative, source-grounded discourse annotations for the offline path.

The rule linker is intentionally modest.  It supplies reconstructible base
annotations and explicit-cue relations; an injected semantic linker may add
stronger annotations later.  Every emitted unit and relation cites exact raw
spans, and generated keys remain routing metadata rather than answer evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from memory_condense.domain.discourse import (
    DiscourseRelation,
    DiscourseUnit,
    EvidenceAtom,
    RelationMember,
    evidence_span_sort_key,
    identity_sha256,
)


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", re.IGNORECASE)
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "our",
        "that",
        "the",
        "this",
        "to",
        "we",
        "with",
    }
)

_KIND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "current_state",
        re.compile(
            r"\b(?:current state|currently|at present|baseline|status is|stands at)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "goal",
        re.compile(
            r"\b(?:goal|objective|target|success criterion|we need to achieve|aim)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dependency",
        re.compile(
            r"\b(?:depends? on|dependency|prerequisite|blocked by|component|resource)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "constraint",
        re.compile(
            r"\b(?:must|cannot|can't|constraint|requirement|limited to|hard cap|budget)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "decision",
        re.compile(
            r"\b(?:decided|decision|we will|we'll use|selected|chose|adopted)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "issue",
        re.compile(
            r"\b(?:bug|problem|blocker|unresolved|open issue|risk)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "failure",
        re.compile(
            r"\b(?:failure|failed|error|broken|regression|adverse outcome)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "observation",
        re.compile(
            r"\b(?:observed|measured|measurement|result|accuracy|latency|passed|showed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "action",
        re.compile(
            r"\b(?:implemented|changed|tried|ran|deployed|tested|configured|built)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "option",
        re.compile(
            r"\b(?:option|alternative|could|might|consider|proposal|propose)\b",
            re.IGNORECASE,
        ),
    ),
)

_REVISION_RE = re.compile(
    r"\b(?:instead|revis(?:e|ed|ing)|replace(?:d)?|supersed(?:e|ed)|no longer)\b",
    re.IGNORECASE,
)
_CONTRADICTION_RE = re.compile(
    r"\b(?:however|but|contradict(?:s|ed)?|conflict(?:s|ed)?|on the other hand)\b",
    re.IGNORECASE,
)
_DEPENDENCY_RE = re.compile(
    r"\b(?:depends? on|requires?|blocked by|only if|prerequisite)\b",
    re.IGNORECASE,
)
_CAUSE_RE = re.compile(
    r"\b(?:because|caused by|therefore|so that|led to|results? from)\b",
    re.IGNORECASE,
)
_RESOLUTION_RE = re.compile(
    r"\b(?:fixed|resolved|addressed|mitigated|unblocked)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class LinkerInput:
    atom: EvidenceAtom
    episode_id: str | None = None


@dataclass(frozen=True, slots=True)
class LinkerOutput:
    units: tuple[DiscourseUnit, ...]
    relations: tuple[DiscourseRelation, ...]
    retained_request_token_state_bytes: int = 0

    def __post_init__(self) -> None:
        if self.retained_request_token_state_bytes != 0:
            raise ValueError("discourse linker cannot retain request token state")


def _unit_kind(text: str) -> str:
    if text.rstrip().endswith("?"):
        return "question"
    for kind, pattern in _KIND_PATTERNS:
        if pattern.search(text):
            return kind
    return "claim"


def _canonical_key(text: str) -> str:
    terms = [
        token.casefold()
        for token in _TOKEN_RE.findall(text)
        if token.casefold() not in _STOP
    ]
    return " ".join(terms[:12]) or "unkeyed"


def _unit_id(*, artifact_id: str, item: LinkerInput, kind: str, key: str) -> str:
    return "unit-" + identity_sha256(
        {
            "artifact_id": artifact_id,
            "kind": kind,
            "canonical_key": key,
            "evidence": item.atom.span.identity_payload(),
            "episode_id": item.episode_id,
        }
    )[:24]


def _relation(
    *,
    artifact_id: str,
    relation_type: str,
    left: DiscourseUnit,
    right: DiscourseUnit,
    left_role: str,
    right_role: str,
    confidence: float,
) -> DiscourseRelation:
    members = (
        RelationMember(left.unit_id, left_role, 0),
        RelationMember(right.unit_id, right_role, 1),
    )
    evidence_by_identity = {
        identity_sha256(item.identity_payload()): item
        for item in (*left.evidence, *right.evidence)
    }
    evidence = tuple(
        sorted(
            evidence_by_identity.values(),
            key=evidence_span_sort_key,
        )
    )
    body = {
        "artifact_id": artifact_id,
        "relation_type": relation_type,
        "members": [
            {"unit_id": item.unit_id, "role": item.role, "ordinal": item.ordinal}
            for item in members
        ],
        "evidence": [item.identity_payload() for item in evidence],
    }
    return DiscourseRelation(
        relation_id="relation-" + identity_sha256(body)[:24],
        artifact_id=artifact_id,
        relation_type=relation_type,
        members=members,
        evidence=evidence,
        confidence=confidence,
        created_ordinal=max(left.asserted_ordinal, right.asserted_ordinal),
        metadata={"linker": "explicit-cue-v1"},
    )


def _nearest_prior(
    units: Sequence[DiscourseUnit],
    current: DiscourseUnit,
    *,
    kinds: tuple[str, ...] | None = None,
) -> DiscourseUnit | None:
    source_id = current.evidence[0].source_id
    preferred = kinds or ("*",)
    for kind in preferred:
        for candidate in reversed(units):
            if source_id is not None and candidate.evidence[0].source_id != source_id:
                continue
            if kind == "*" or candidate.kind == kind:
                return candidate
    return None


class RuleBasedDiscourseLinker:
    """Emit domain-neutral base units and only explicitly cued semantic links."""

    def link(
        self,
        artifact_id: str,
        inputs: Sequence[LinkerInput],
    ) -> LinkerOutput:
        ordered = tuple(
            sorted(
                inputs,
                key=lambda item: (
                    *evidence_span_sort_key(item.atom.span),
                    item.atom.atom_id,
                ),
            )
        )
        units: list[DiscourseUnit] = []
        for item in ordered:
            kind = _unit_kind(item.atom.text)
            key = _canonical_key(item.atom.text)
            units.append(
                DiscourseUnit(
                    unit_id=_unit_id(
                        artifact_id=artifact_id,
                        item=item,
                        kind=kind,
                        key=key,
                    ),
                    artifact_id=artifact_id,
                    kind=kind,
                    canonical_key=key,
                    asserted_ordinal=item.atom.span.ordinal,
                    confidence=1.0 if kind in {"question", "claim"} else 0.75,
                    evidence=(item.atom.span,),
                    metadata=(
                        {"episode_id": item.episode_id, "linker": "rules-v1"}
                        if item.episode_id is not None
                        else {"linker": "rules-v1"}
                    ),
                )
            )

        relations: list[DiscourseRelation] = []
        for index, current in enumerate(units):
            if index == 0:
                continue
            current_text = ordered[index].atom.text
            # Sequence is reconstructible source-local structure, not a
            # semantic claim.  Interleaved source histories therefore connect
            # to the nearest prior unit in the same source, while evidence
            # without an authenticated source never acquires sequence edges.
            prior = (
                _nearest_prior(units[:index], current)
                if current.evidence[0].source_id is not None
                else None
            )
            if prior is not None:
                relations.append(
                    _relation(
                        artifact_id=artifact_id,
                        relation_type="sequence",
                        left=prior,
                        right=current,
                        left_role="previous",
                        right_role="next",
                        confidence=1.0,
                    )
                )

            semantic: tuple[str, str, str, float, tuple[str, ...] | None] | None = None
            if _REVISION_RE.search(current_text):
                semantic = (
                    "revises",
                    "predecessor",
                    "successor",
                    0.80,
                    ("decision", "option", "claim"),
                )
            elif _CONTRADICTION_RE.search(current_text):
                semantic = (
                    "contradicts",
                    "side_a",
                    "side_b",
                    0.65,
                    tuple(dict.fromkeys((current.kind, "claim", "decision", "observation"))),
                )
            elif _DEPENDENCY_RE.search(current_text):
                semantic = (
                    "depends_on",
                    "requirement",
                    "dependent",
                    0.60,
                    ("constraint", "goal", "claim"),
                )
            elif _CAUSE_RE.search(current_text):
                semantic = ("causes", "cause", "effect", 0.60, None)
            elif _RESOLUTION_RE.search(current_text):
                semantic = (
                    "resolves",
                    "issue",
                    "resolution",
                    0.70,
                    ("issue",),
                )
            elif current.kind == "observation":
                semantic = (
                    "evaluates",
                    "action",
                    "result",
                    0.60,
                    ("action",),
                )
            if semantic is not None:
                relation_type, left_role, right_role, confidence, kinds = semantic
                antecedent = _nearest_prior(units[:index], current, kinds=kinds)
                if antecedent is not None:
                    relations.append(
                        _relation(
                            artifact_id=artifact_id,
                            relation_type=relation_type,
                            left=antecedent,
                            right=current,
                            left_role=left_role,
                            right_role=right_role,
                            confidence=confidence,
                        )
                    )

        deduplicated = {
            relation.relation_id: relation for relation in relations
        }
        return LinkerOutput(
            units=tuple(units),
            relations=tuple(deduplicated[key] for key in sorted(deduplicated)),
        )


__all__ = ["LinkerInput", "LinkerOutput", "RuleBasedDiscourseLinker"]
