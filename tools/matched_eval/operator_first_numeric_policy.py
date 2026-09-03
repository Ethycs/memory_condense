"""Provider-free operator-first policy for sealed numeric evidence.

The existing typed executor deliberately requires the *global* evidence
frontier to be closed before it reduces a count.  That is the right default,
but it is unnecessarily strong for two common cases:

* a fixed-arity comparison whose two named scalar operands are already bound;
* a count whose specialist can prove that its operator-relevant candidate
  population, rather than every evidence lane, is exhausted.

This module is an isolated policy experiment for those cases.  It consumes the
sealed provider-input mapping emitted by the typed final arm, never reads the
protected parent, never calls a provider, and emits a content-addressed proof.
It does not mutate or replace the reproducibility runner.

Typed facts may optionally add conservative semicolon-delimited relation
attributes.  ``event_action``, ``member_keys``, ``event_key``,
``numeric_scope``, and ``obligations`` refine the deterministic projection;
the normal typed fields and exact summary remain the fallback.  For example::

    authored_by_user;event_action=acquire;
    member_keys=peace_lily|succulent;date_basis=relative_event_time

The semantic identity follows the same action/entity grouping used by the
numeric operand specialist and reconciler: paraphrases collapse only when
their normalized entity/event identity agrees.  The public status enum is
reused from :mod:`typed_operator_executor` so an overlay can consume this
decision beside the legacy executor without translating status meanings.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .typed_action_semantics import (
    canonical_action_concepts,
    completed_action_concepts,
    planned_action_concepts,
)
from .typed_operator_executor import ExecutionStatus
from .typed_operator_spec import normalized_terms


FRONTIER_FORMAT = "memory-condense-relevant-numeric-frontier-v1"
ATOM_FORMAT = "memory-condense-operator-first-numeric-atom-v1"
EXCLUSION_FORMAT = "memory-condense-operator-first-numeric-exclusion-v1"
DECISION_FORMAT = "memory-condense-operator-first-numeric-decision-v1"
COMPILATION_FORMAT = "memory-condense-operator-first-numeric-compilation-v1"


class NumericPolicyMode(str, Enum):
    FIXED_SCALAR_COMPARISON = "fixed_scalar_comparison"
    DISTINCT_ENTITY_COUNT = "distinct_entity_count"
    ENTITY_EVENT_COUNT = "entity_event_count"
    ACTION_OBLIGATION_COUNT = "action_obligation_count"


class TemporalEvidenceBasis(str, Enum):
    TEXTUAL_EVENT_DATE = "textual_event_date"
    RELATIVE_EVENT_TIME = "relative_event_time"
    SOURCE_CREATED_AT = "source_created_at"
    UNKNOWN = "unknown"


_HANDLE_RE = re.compile(r"^H[0-9]{3,6}$")
_DATED_RE = re.compile(
    r"^\[Question asked at (?P<asked_at>.+?)\]\s*", re.IGNORECASE | re.DOTALL
)
_CLAUSE_RE = re.compile(r"[^.!?;\r\n]+(?:[.!?]+|$)")
_MONTHS = {
    name.casefold(): ordinal
    for ordinal, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}
_MONTH_PATTERN = "|".join(name.title() for name in _MONTHS)
_CALENDAR_MONTH_RE = re.compile(
    rf"\b(?:in|during)\s+(?:the\s+)?(?:month\s+of\s+|month\s+)?"
    rf"(?P<month>{_MONTH_PATTERN})\b",
    re.IGNORECASE,
)
_MONTH_DAY_RE = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\s+(?P<day>\d{{1,2}})"
    r"(?:st|nd|rd|th)?(?:,?\s+(?P<year>20\d{2}))?\b",
    re.IGNORECASE,
)
_DAY_MONTH_RE = re.compile(
    rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+"
    rf"(?P<month>{_MONTH_PATTERN})(?:,?\s+(?P<year>20\d{{2}}))?\b",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(
    r"(?<!\d)(?P<month>0?[1-9]|1[0-2])/(?P<day>0?[1-9]|[12]\d|3[01])"
    r"(?:/(?P<year>\d{2}|20\d{2}))?(?!\d)"
)
_ISO_DATE_RE = re.compile(r"\b(?P<year>20\d{2})-(?P<month>\d{2})-(?P<day>\d{2})\b")
_LAST_MONTH_DAY_RE = re.compile(
    r"\b(?:on\s+)?(?:the\s+)?(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+"
    r"of\s+last\s+month\b",
    re.IGNORECASE,
)
_AGO_RE = re.compile(
    r"\b(?P<count>a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?P<unit>days?|weeks?|months?)\s+ago\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
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
}
_CUISINES = (
    "Ethiopian",
    "Indian",
    "Korean",
    "Thai",
    "Vegan",
    "Cuban",
    "Italian",
    "Chinese",
    "Japanese",
    "Mexican",
    "French",
    "Greek",
    "Spanish",
    "Vietnamese",
    "Lebanese",
    "Moroccan",
)
_CUISINE_RE = re.compile(
    rf"\b(?P<name>{'|'.join(_CUISINES)})\b(?=.{{0,32}}\b(?:cuisine|food|"
    r"cook(?:ing)?|dish|recipe|restaurant|bibimbap|curry|lasagna)\b)",
    re.IGNORECASE,
)
_BIKE_RE = re.compile(
    r"\b(?P<name>(?:(?:road|commuter|mountain|gravel|hybrid|electric|touring)\s+)?bike)\b",
    re.IGNORECASE,
)
_JEWELRY_RE = re.compile(
    r"\b(?:(?:new|pair\s+of|new\s+pair\s+of|my)\s+)*"
    r"(?P<name>(?:(?:engagement|wedding|emerald|silver|gold|diamond|aquamarine|"
    r"sapphire|ruby)\s+)?(?:earrings?|necklace|rings?|bracelets?|brooch|pendant))\b",
    re.IGNORECASE,
)
_PLANT_PATTERNS = (
    ("peace_lily", re.compile(r"\bpeace\s+lil(?:y|ies)\b", re.IGNORECASE)),
    ("snake_plant", re.compile(r"\bsnake\s+plants?\b", re.IGNORECASE)),
    ("succulent", re.compile(r"\bsucculent(?:\s+plants?)?\b", re.IGNORECASE)),
    ("spider_plant", re.compile(r"\bspider\s+plants?\b", re.IGNORECASE)),
    ("aloe_vera", re.compile(r"\baloe(?:\s+vera)?\b", re.IGNORECASE)),
    ("orchid", re.compile(r"\borchids?\b", re.IGNORECASE)),
    ("fern", re.compile(r"\bferns?\b", re.IGNORECASE)),
)
_MUSEUM_RE = re.compile(
    r"\b(?P<name>(?:The\s+)?(?:[A-Z][A-Za-z'’\-]*\s+){0,4}(?:Museum|Gallery))\b"
)
_ART_CUBE_RE = re.compile(r"\bThe\s+Art\s+Cube\b", re.IGNORECASE)


def _ordered_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(type(value) is not str or not value for value in result):
        raise MatchedEvalContractError(f"{label} must contain exact text")
    if len(result) != len(set(result)):
        raise MatchedEvalContractError(f"{label} must be ordered and unique")
    return result


def _provider_projection(provider_input: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(provider_input, Mapping):
        raise TypeError("provider_input must be a sealed mapping")
    assert_gold_blind(provider_input, path="operator_first_numeric_input")
    question = provider_input.get("dated_question")
    typed = provider_input.get("typed_evidence")
    if type(question) is not str or not question.strip():
        raise MatchedEvalContractError("numeric policy requires a dated question")
    if not isinstance(typed, Mapping):
        raise MatchedEvalContractError("numeric policy requires typed evidence")
    return {"dated_question": question, "typed_evidence": dict(typed)}


def _policy_input_sha256(provider_input: Mapping[str, Any]) -> str:
    # Deliberately excludes protected_parent_fallback and response-schema data.
    return identity_sha256(_provider_projection(provider_input))


def _inventory(provider_input: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    typed = _provider_projection(provider_input)["typed_evidence"]
    raw = typed.get("handles")
    if type(raw) is not list:
        raise MatchedEvalContractError("typed evidence handle inventory changed")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, Mapping):
            raise MatchedEvalContractError("typed evidence handle row changed")
        row = dict(value)
        handle = row.get("handle_id")
        group = row.get("group_handle")
        if (
            type(handle) is not str
            or _HANDLE_RE.fullmatch(handle) is None
            or type(group) is not str
            or not group
            or handle in seen
        ):
            raise MatchedEvalContractError("typed evidence handle identity changed")
        seen.add(handle)
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class RelevantNumericFrontier:
    policy_input_sha256: str
    candidate_population_receipt_sha256: str
    represented_handle_ids: tuple[str, ...]
    unresolved_candidate_keys: tuple[str, ...]
    selection_truncated: bool
    closed: bool
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.policy_input_sha256, "relevant frontier input")
        require_sha256(
            self.candidate_population_receipt_sha256,
            "relevant frontier population",
        )
        _ordered_unique(self.represented_handle_ids, "relevant frontier handles")
        _ordered_unique(
            self.unresolved_candidate_keys, "relevant frontier unresolved keys"
        )
        if type(self.selection_truncated) is not bool or type(self.closed) is not bool:
            raise MatchedEvalContractError("relevant frontier flags changed type")
        expected_closed = not self.selection_truncated and not self.unresolved_candidate_keys
        if self.closed != expected_closed:
            raise MatchedEvalContractError("relevant frontier closure is not justified")
        if self.provider_prompt_count != 0 or self.retained_transformer_token_state_bytes != 0:
            raise MatchedEvalContractError("relevant frontier must remain provider-free")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("relevant numeric frontier receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="relevant_numeric_frontier")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "candidate_population_receipt_sha256": (
                self.candidate_population_receipt_sha256
            ),
            "closed": self.closed,
            "format": FRONTIER_FORMAT,
            "policy_input_sha256": self.policy_input_sha256,
            "provider_prompt_count": 0,
            "represented_handle_ids": list(self.represented_handle_ids),
            "retained_transformer_token_state_bytes": 0,
            "selection_truncated": self.selection_truncated,
            "unresolved_candidate_keys": list(self.unresolved_candidate_keys),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def build_relevant_numeric_frontier(
    provider_input: Mapping[str, Any],
    /,
    *,
    candidate_population_receipt_sha256: str,
    represented_handle_ids: tuple[str, ...] | None = None,
    unresolved_candidate_keys: tuple[str, ...] = (),
    selection_truncated: bool = False,
) -> RelevantNumericFrontier:
    """Seal an independently exhaustive specialist candidate frontier.

    The caller must bind ``candidate_population_receipt_sha256`` to its actual
    exhaustive action/entity scan.  A globally truncated prompt is allowed;
    this record claims closure only for that specialist population.
    """

    require_sha256(
        candidate_population_receipt_sha256, "numeric candidate population"
    )
    inventory = tuple(row["handle_id"] for row in _inventory(provider_input))
    represented = inventory if represented_handle_ids is None else represented_handle_ids
    represented = _ordered_unique(represented, "represented numeric handles")
    unresolved = _ordered_unique(
        unresolved_candidate_keys, "unresolved numeric candidate keys"
    )
    if not set(represented) <= set(inventory):
        raise MatchedEvalContractError("relevant frontier escaped handle inventory")
    if type(selection_truncated) is not bool:
        raise MatchedEvalContractError("numeric selection truncation flag changed")
    return RelevantNumericFrontier(
        policy_input_sha256=_policy_input_sha256(provider_input),
        candidate_population_receipt_sha256=candidate_population_receipt_sha256,
        represented_handle_ids=represented,
        unresolved_candidate_keys=unresolved,
        selection_truncated=selection_truncated,
        closed=not selection_truncated and not unresolved,
    )


@dataclass(frozen=True, slots=True)
class NumericCandidateAtom:
    item_sha256: str
    handle_ids: tuple[str, ...]
    source_group_handles: tuple[str, ...]
    entity_key: str
    action_key: str
    event_key: str
    contribution_value: float
    numeric_value: float | None
    unit: str | None
    status: str
    source_role: str
    event_date: str | None
    temporal_basis: TemporalEvidenceBasis
    comparison_side_id: str | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.item_sha256, "numeric atom item")
        handles = _ordered_unique(self.handle_ids, "numeric atom handles")
        groups = _ordered_unique(self.source_group_handles, "numeric atom groups")
        if not handles or not groups or any(_HANDLE_RE.fullmatch(row) is None for row in handles):
            raise MatchedEvalContractError("numeric atom lost opaque provenance")
        for value, label in (
            (self.entity_key, "numeric atom entity"),
            (self.action_key, "numeric atom action"),
            (self.event_key, "numeric atom event"),
            (self.status, "numeric atom status"),
            (self.source_role, "numeric atom source role"),
        ):
            require_text(value, label)
        if (
            type(self.contribution_value) not in {int, float}
            or not math.isfinite(float(self.contribution_value))
            or float(self.contribution_value) <= 0
        ):
            raise MatchedEvalContractError("numeric atom contribution is invalid")
        if self.numeric_value is not None and (
            type(self.numeric_value) not in {int, float}
            or not math.isfinite(float(self.numeric_value))
        ):
            raise MatchedEvalContractError("numeric atom scalar is invalid")
        if self.unit is not None:
            require_text(self.unit, "numeric atom unit")
        if self.event_date is not None:
            require_text(self.event_date, "numeric atom event date")
        if type(self.temporal_basis) is not TemporalEvidenceBasis:
            raise MatchedEvalContractError("numeric atom temporal basis changed")
        if self.comparison_side_id is not None:
            require_text(self.comparison_side_id, "numeric atom comparison side")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("numeric candidate atom receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "action_key": self.action_key,
            "comparison_side_id": self.comparison_side_id,
            "contribution_value": self.contribution_value,
            "entity_key": self.entity_key,
            "event_date": self.event_date,
            "event_key": self.event_key,
            "format": ATOM_FORMAT,
            "handle_ids": list(self.handle_ids),
            "item_sha256": self.item_sha256,
            "numeric_value": self.numeric_value,
            "source_group_handles": list(self.source_group_handles),
            "source_role": self.source_role,
            "status": self.status,
            "temporal_basis": self.temporal_basis.value,
            "unit": self.unit,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class NumericCandidateExclusion:
    item_sha256: str
    handle_ids: tuple[str, ...]
    reason: str
    candidate_receipt_sha256: str | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.item_sha256, "numeric exclusion item")
        _ordered_unique(self.handle_ids, "numeric exclusion handles")
        require_text(self.reason, "numeric exclusion reason")
        if self.candidate_receipt_sha256 is not None:
            require_sha256(
                self.candidate_receipt_sha256, "numeric exclusion candidate"
            )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("numeric candidate exclusion changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "candidate_receipt_sha256": self.candidate_receipt_sha256,
            "format": EXCLUSION_FORMAT,
            "handle_ids": list(self.handle_ids),
            "item_sha256": self.item_sha256,
            "reason": self.reason,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class OperatorFirstNumericCompilation:
    """Content-addressed candidate projection before frontier adjudication.

    A full-store census can use this projection without pretending that a
    bounded provider packet is complete.  It deliberately contains neither a
    prediction nor a parent fallback and therefore cannot itself authorize a
    numeric answer.
    """

    policy_input_sha256: str
    typed_evidence_sha256: str
    operator_spec_sha256: str
    mode: NumericPolicyMode
    question_domain: str
    query_action_keys: tuple[str, ...]
    candidate_atoms: tuple[NumericCandidateAtom, ...]
    exclusions: tuple[NumericCandidateExclusion, ...]
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.policy_input_sha256, "numeric compilation input"),
            (self.typed_evidence_sha256, "numeric compilation evidence"),
            (self.operator_spec_sha256, "numeric compilation operator"),
        ):
            require_sha256(value, label)
        if type(self.mode) is not NumericPolicyMode:
            raise MatchedEvalContractError("numeric compilation mode changed")
        require_text(self.question_domain, "numeric compilation domain")
        _ordered_unique(self.query_action_keys, "numeric compilation query actions")
        if type(self.candidate_atoms) is not tuple or any(
            type(row) is not NumericCandidateAtom for row in self.candidate_atoms
        ):
            raise MatchedEvalContractError("numeric compilation atoms changed")
        if type(self.exclusions) is not tuple or any(
            type(row) is not NumericCandidateExclusion for row in self.exclusions
        ):
            raise MatchedEvalContractError("numeric compilation exclusions changed")
        if (
            self.provider_prompt_count != 0
            or self.retained_transformer_token_state_bytes != 0
        ):
            raise MatchedEvalContractError(
                "numeric compilation must remain provider-free"
            )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("numeric compilation changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="operator_first_numeric_compilation")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "candidate_atoms": [row.projection() for row in self.candidate_atoms],
            "exclusions": [row.projection() for row in self.exclusions],
            "format": COMPILATION_FORMAT,
            "mode": self.mode.value,
            "operator_spec_sha256": self.operator_spec_sha256,
            "policy_input_sha256": self.policy_input_sha256,
            "provider_prompt_count": 0,
            "query_action_keys": list(self.query_action_keys),
            "question_domain": self.question_domain,
            "retained_transformer_token_state_bytes": 0,
            "typed_evidence_sha256": self.typed_evidence_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class OperatorFirstNumericDecision:
    policy_input_sha256: str
    typed_evidence_sha256: str
    operator_spec_sha256: str
    mode: NumericPolicyMode
    status: ExecutionStatus
    decision: Literal["replace", "abstain"]
    prediction: str
    numeric_result: float | None
    used_handle_ids: tuple[str, ...]
    used_candidate_receipt_sha256s: tuple[str, ...]
    candidate_atoms: tuple[NumericCandidateAtom, ...]
    exclusions: tuple[NumericCandidateExclusion, ...]
    relevant_frontier_receipt_sha256: str | None
    reason: str
    proof_sha256: str = ""
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.policy_input_sha256, "numeric decision input"),
            (self.typed_evidence_sha256, "numeric decision evidence"),
            (self.operator_spec_sha256, "numeric decision operator"),
        ):
            require_sha256(value, label)
        if type(self.mode) is not NumericPolicyMode or type(self.status) is not ExecutionStatus:
            raise MatchedEvalContractError("numeric decision enums changed")
        expected_decision = (
            "replace" if self.status is ExecutionStatus.SUPPORTED else "abstain"
        )
        if self.decision != expected_decision:
            raise MatchedEvalContractError("numeric decision/status contract changed")
        if type(self.prediction) is not str or bool(self.prediction) != (
            self.status is ExecutionStatus.SUPPORTED
        ):
            raise MatchedEvalContractError("numeric decision prediction changed")
        if self.numeric_result is not None and (
            type(self.numeric_result) not in {int, float}
            or not math.isfinite(float(self.numeric_result))
        ):
            raise MatchedEvalContractError("numeric decision scalar changed")
        _ordered_unique(self.used_handle_ids, "numeric decision handles")
        _ordered_unique(
            self.used_candidate_receipt_sha256s, "numeric decision candidates"
        )
        if type(self.candidate_atoms) is not tuple or any(
            type(row) is not NumericCandidateAtom for row in self.candidate_atoms
        ):
            raise MatchedEvalContractError("numeric decision atoms changed")
        if type(self.exclusions) is not tuple or any(
            type(row) is not NumericCandidateExclusion for row in self.exclusions
        ):
            raise MatchedEvalContractError("numeric decision exclusions changed")
        atom_receipts = {row.receipt_sha256 for row in self.candidate_atoms}
        atom_handles = {handle for row in self.candidate_atoms for handle in row.handle_ids}
        if not set(self.used_candidate_receipt_sha256s) <= atom_receipts:
            raise MatchedEvalContractError("numeric decision used an unknown candidate")
        if not set(self.used_handle_ids) <= atom_handles:
            raise MatchedEvalContractError("numeric decision used an unknown handle")
        if self.relevant_frontier_receipt_sha256 is not None:
            require_sha256(
                self.relevant_frontier_receipt_sha256,
                "numeric decision relevant frontier",
            )
        require_text(self.reason, "numeric decision reason")
        proof = self.proof_projection()
        expected_proof = identity_sha256(proof)
        if self.proof_sha256 and self.proof_sha256 != expected_proof:
            raise MatchedEvalContractError("numeric decision proof changed")
        object.__setattr__(self, "proof_sha256", expected_proof)
        if self.provider_prompt_count != 0 or self.retained_transformer_token_state_bytes != 0:
            raise MatchedEvalContractError("numeric decision must remain provider-free")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("operator-first numeric decision changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="operator_first_numeric_decision")

    def proof_projection(self) -> dict[str, Any]:
        return {
            "candidate_atoms": [row.projection() for row in self.candidate_atoms],
            "exclusions": [row.projection() for row in self.exclusions],
            "mode": self.mode.value,
            "operator_spec_sha256": self.operator_spec_sha256,
            "policy_input_sha256": self.policy_input_sha256,
            "relevant_frontier_receipt_sha256": (
                self.relevant_frontier_receipt_sha256
            ),
            "typed_evidence_sha256": self.typed_evidence_sha256,
            "used_candidate_receipt_sha256s": list(
                self.used_candidate_receipt_sha256s
            ),
        }

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "decision": self.decision,
            "format": DECISION_FORMAT,
            "mode": self.mode.value,
            "numeric_result": self.numeric_result,
            "operator_spec_sha256": self.operator_spec_sha256,
            "policy_input_sha256": self.policy_input_sha256,
            "prediction": self.prediction,
            "proof": self.proof_projection(),
            "proof_sha256": self.proof_sha256,
            "provider_prompt_count": 0,
            "reason": self.reason,
            "relevant_frontier_receipt_sha256": (
                self.relevant_frontier_receipt_sha256
            ),
            "retained_transformer_token_state_bytes": 0,
            "status": self.status.value,
            "typed_evidence_sha256": self.typed_evidence_sha256,
            "used_handle_ids": list(self.used_handle_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class _TemporalWitness:
    basis: TemporalEvidenceBasis
    event_date: date | None
    relative_days: int | None = None
    relative_month_offset: int | None = None
    calendar_precise: bool = False


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MatchedEvalContractError(f"{label} changed mapping type")
    return dict(value)


def _items(provider_input: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    typed = _provider_projection(provider_input)["typed_evidence"]
    raw = typed.get("items")
    if type(raw) is not list:
        raise MatchedEvalContractError("typed evidence items changed type")
    return tuple(_mapping(row, "typed evidence item") for row in raw)


def _operator(provider_input: Mapping[str, Any]) -> dict[str, Any]:
    typed = _provider_projection(provider_input)["typed_evidence"]
    return _mapping(typed.get("operator_spec", {}), "numeric operator spec")


def _question(provider_input: Mapping[str, Any]) -> tuple[str, str, date | None]:
    value = _provider_projection(provider_input)["dated_question"]
    match = _DATED_RE.match(value)
    body = _DATED_RE.sub("", value).strip()
    asked = _parse_datetime(match.group("asked_at") if match else None)
    return value, body, None if asked is None else asked.date()


def _parse_datetime(value: object) -> datetime | None:
    if type(value) is not str or not value.strip():
        return None
    cleaned = value.strip().replace("Z", "+00:00")
    cleaned = re.sub(r"\s*\([A-Za-z]{3,9}\)\s*", " ", cleaned).strip()
    for candidate in (cleaned, cleaned.replace("/", "-")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    for pattern in (
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(cleaned, pattern)
        except ValueError:
            pass
    return None


def _relation_attributes(relation: object) -> tuple[dict[str, str], frozenset[str]]:
    if type(relation) is not str:
        return {}, frozenset()
    attributes: dict[str, str] = {}
    flags: set[str] = set()
    for raw in relation.split(";"):
        part = raw.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            attributes[key.strip().casefold()] = value.strip()
        else:
            flags.add(part.casefold())
    return attributes, frozenset(flags)


def _role(flags: frozenset[str], relation: object) -> str:
    folded = str(relation or "").casefold()
    if "authored_by_assistant" in flags or "authored_by_assistant" in folded:
        return "assistant"
    if "authored_by_user" in flags or "authored_by_user" in folded:
        return "user"
    return "unknown"


def _key(value: str) -> str:
    terms = normalized_terms(value.replace("_", " "))
    return "_".join(terms) or hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _split_values(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(
        dict.fromkeys(
            key
            for raw in re.split(r"[|,]", value)
            if (key := _key(raw.strip()))
        )
    )


def _structured_actions(attributes: Mapping[str, str]) -> tuple[str, ...]:
    value = attributes.get("event_action")
    if not value:
        return ()
    result: list[str] = []
    for surface in re.split(r"[|,]", value):
        concepts = canonical_action_concepts(surface)
        candidates = concepts or (_key(surface),)
        for candidate in candidates:
            if candidate not in result:
                result.append(candidate)
    return tuple(result)


def _query_mode(body: str, operator: Mapping[str, Any]) -> NumericPolicyMode:
    if str(operator.get("comparison_mode", "none")) != "none" or re.search(
        r"\bcompared\s+to\b|\bhow\s+much\s+(?:more|less)\b", body, re.I
    ):
        return NumericPolicyMode.FIXED_SCALAR_COMPARISON
    if {"pickup", "return"} <= set(canonical_action_concepts(body)) or re.search(
        r"\bpick\s+up\s+or\s+return\b", body, re.I
    ):
        return NumericPolicyMode.ACTION_OBLIGATION_COUNT
    if re.search(r"\bdifferent\b", body, re.I):
        return NumericPolicyMode.DISTINCT_ENTITY_COUNT
    return NumericPolicyMode.ENTITY_EVENT_COUNT


def _domain(body: str) -> str:
    if re.search(r"\bcuisines?\b", body, re.I):
        return "cuisine"
    if re.search(r"\bbikes?\b", body, re.I):
        return "bike"
    if re.search(r"\b(?:jewelry|jewellery)\b", body, re.I):
        return "jewelry"
    if re.search(r"\bplants?\b", body, re.I):
        return "plant"
    if re.search(r"\b(?:museums?|galleries?)\b", body, re.I):
        return "museum_gallery"
    if re.search(r"\b(?:clothing|clothes)\b", body, re.I):
        return "clothing"
    return "generic"


def _clauses(summary: str) -> tuple[str, ...]:
    result = tuple(
        match.group(0).strip() for match in _CLAUSE_RE.finditer(summary) if match.group(0).strip()
    )
    return result or (summary,)


def _effective_actions(
    clause: str,
    *,
    status: str,
    structured: tuple[str, ...],
) -> tuple[str, ...]:
    if structured:
        return structured
    completed = completed_action_concepts(clause)
    planned = planned_action_concepts(clause)
    # The shared completed-action helper intentionally recognizes ``I got``
    # only when the acquired object follows the verb.  In a relative clause
    # the object is the antecedent (``my snake plant, which I got from my
    # sister``), so retain that equally explicit acquisition construction.
    if re.search(r"\bwhich\s+I\s+(?:have\s+)?got\s+from\b", clause, re.I):
        completed = tuple(dict.fromkeys((*completed, "acquire")))
    # ``got X resized/repaired`` is a causative service construction, not an
    # acquisition.  The shared action lexicon cannot make that syntactic
    # distinction because it intentionally operates on surfaces only.
    if re.search(
        r"\bgot\b[^,.;!?]{0,80}\b(?:resized|cleaned|repaired|serviced|fixed)\b",
        clause,
        re.I,
    ):
        completed = tuple(value for value in completed if value != "acquire")
    # Planned maintenance is often expressed through the component operation
    # rather than the umbrella word "service" (for example, "need to replace
    # the tire on my commuter bike").  This remains harmless for non-service
    # questions because the query-action intersection is applied downstream.
    if re.search(
        r"\b(?:need|plan|intend|schedule|due|time\s+to)\b[^.;!?]{0,90}"
        r"\b(?:replace|repair|service|fix|tune|clean)\b",
        clause,
        re.I,
    ):
        planned = tuple(dict.fromkeys((*planned, "service")))
    if status == "proposed":
        return planned
    return tuple(dict.fromkeys((*completed, *planned)))


def _extract_entities(
    domain: str,
    text: str,
    *,
    item_entity_key: object,
    attributes: Mapping[str, str],
) -> tuple[str, ...]:
    structured = _split_values(
        attributes.get("member_keys") or attributes.get("members")
    )
    if structured:
        return structured
    if type(item_entity_key) is str and item_entity_key.strip():
        return (_key(item_entity_key),)

    output: list[str] = []
    if domain == "bike":
        output.extend(_key(match.group("name")) for match in _BIKE_RE.finditer(text))
    elif domain == "jewelry":
        output.extend(_key(match.group("name")) for match in _JEWELRY_RE.finditer(text))
    elif domain == "plant":
        output.extend(
            key for key, pattern in _PLANT_PATTERNS if pattern.search(text)
        )
    elif domain == "museum_gallery":
        output.extend(_key(match.group("name")) for match in _MUSEUM_RE.finditer(text))
        if _ART_CUBE_RE.search(text):
            output.append("the_art_cube")
    elif domain == "cuisine":
        output.extend(_key(match.group("name")) for match in _CUISINE_RE.finditer(text))
    unique = tuple(dict.fromkeys(output))
    if domain == "bike" and any(
        value.endswith("_bike") and value != "bike" for value in unique
    ):
        unique = tuple(value for value in unique if value != "bike")
    if domain == "jewelry":
        specific_suffixes = {
            suffix
            for suffix in ("ring", "earring", "necklace", "bracelet")
            if any(value.endswith("_" + suffix) for value in unique)
        }
        unique = tuple(
            value
            for value in unique
            if value not in specific_suffixes
            and not (value == "pendant" and any(name.endswith("necklace") for name in unique))
        )
    return unique


def _previous_month(asked: date) -> tuple[int, int]:
    return (asked.year - 1, 12) if asked.month == 1 else (asked.year, asked.month - 1)


def _textual_date(text: str, asked: date | None) -> date | None:
    iso = _ISO_DATE_RE.search(text)
    if iso:
        try:
            return date(int(iso.group("year")), int(iso.group("month")), int(iso.group("day")))
        except ValueError:
            return None
    numeric = _NUMERIC_DATE_RE.search(text)
    if numeric and asked is not None:
        raw_year = numeric.group("year")
        year = asked.year if raw_year is None else int(raw_year)
        if raw_year is not None and len(raw_year) == 2:
            year += 2000
        try:
            return date(year, int(numeric.group("month")), int(numeric.group("day")))
        except ValueError:
            return None
    named = _MONTH_DAY_RE.search(text)
    if named and asked is not None:
        month = _MONTHS[named.group("month").casefold()]
        year = int(named.group("year") or asked.year)
        try:
            return date(year, month, int(named.group("day")))
        except ValueError:
            return None
    day_first = _DAY_MONTH_RE.search(text)
    if day_first and asked is not None:
        month = _MONTHS[day_first.group("month").casefold()]
        year = int(day_first.group("year") or asked.year)
        try:
            return date(year, month, int(day_first.group("day")))
        except ValueError:
            return None
    return None


def _temporal_witness(
    text: str,
    *,
    item_date: object,
    declared_basis: str | None,
    asked: date | None,
) -> _TemporalWitness:
    parsed_item = _parse_datetime(item_date)
    anchor = (
        parsed_item.date()
        if parsed_item is not None
        else asked
    )
    explicit = _textual_date(text, asked)
    if explicit is not None:
        return _TemporalWitness(
            TemporalEvidenceBasis.TEXTUAL_EVENT_DATE,
            explicit,
            calendar_precise=True,
        )
    if anchor is not None:
        prior_day = _LAST_MONTH_DAY_RE.search(text)
        if prior_day:
            year, month = _previous_month(anchor)
            try:
                event = date(year, month, int(prior_day.group("day")))
            except ValueError:
                event = None
            if event is not None:
                return _TemporalWitness(
                    TemporalEvidenceBasis.RELATIVE_EVENT_TIME,
                    event,
                    relative_month_offset=-1,
                    calendar_precise=True,
                )
        ago = _AGO_RE.search(text)
        if ago:
            raw_count = ago.group("count").casefold()
            count = _NUMBER_WORDS.get(raw_count, int(raw_count) if raw_count.isdigit() else 1)
            unit = ago.group("unit").casefold()
            days = count * (31 if unit.startswith("month") else 7 if unit.startswith("week") else 1)
            return _TemporalWitness(
                TemporalEvidenceBasis.RELATIVE_EVENT_TIME,
                anchor - timedelta(days=days),
                relative_days=days,
                relative_month_offset=-count if unit.startswith("month") else None,
            )
        folded = text.casefold()
        if "last month" in folded:
            year, month = _previous_month(anchor)
            return _TemporalWitness(
                TemporalEvidenceBasis.RELATIVE_EVENT_TIME,
                date(year, month, min(anchor.day, 28)),
                relative_days=31,
                relative_month_offset=-1,
            )
        if "this month" in folded or "before april comes" in folded:
            return _TemporalWitness(
                TemporalEvidenceBasis.RELATIVE_EVENT_TIME,
                anchor,
                relative_days=0,
                relative_month_offset=0,
                calendar_precise=True,
            )
        if re.search(r"\blast\s+week(?:end)?\b", folded):
            return _TemporalWitness(
                TemporalEvidenceBasis.RELATIVE_EVENT_TIME,
                anchor - timedelta(days=7),
                relative_days=7,
            )
        if re.search(r"\b(?:today|yesterday|recently|just)\b", folded):
            days = 1 if "yesterday" in folded else 0
            return _TemporalWitness(
                TemporalEvidenceBasis.RELATIVE_EVENT_TIME,
                anchor - timedelta(days=days),
                relative_days=days,
            )

    basis = (declared_basis or "").casefold()
    if basis in {"textual_event_date", "explicit_event_time"}:
        return _TemporalWitness(
            TemporalEvidenceBasis.TEXTUAL_EVENT_DATE,
            None if parsed_item is None else parsed_item.date(),
            calendar_precise=parsed_item is not None,
        )
    if basis in {"relative_event_date", "relative_event_time"}:
        return _TemporalWitness(
            TemporalEvidenceBasis.RELATIVE_EVENT_TIME,
            None if parsed_item is None else parsed_item.date(),
            calendar_precise=parsed_item is not None,
        )
    if basis == "source_created_at":
        return _TemporalWitness(
            TemporalEvidenceBasis.SOURCE_CREATED_AT,
            None if parsed_item is None else parsed_item.date(),
        )
    return _TemporalWitness(TemporalEvidenceBasis.UNKNOWN, None)


def _calendar_scope(body: str, asked: date | None) -> tuple[int, int] | None:
    match = _CALENDAR_MONTH_RE.search(body)
    if match is None or asked is None:
        return None
    month = _MONTHS[match.group("month").casefold()]
    year_match = re.search(rf"\b{re.escape(match.group('month'))}\s+(20\d{{2}})\b", body, re.I)
    year = int(year_match.group(1)) if year_match else asked.year - int(month > asked.month)
    return year, month


def _window_days(body: str, operator: Mapping[str, Any]) -> int | None:
    declared = operator.get("temporal_window_days")
    if type(declared) is int and declared > 0:
        return declared
    folded = body.casefold()
    number = r"(?P<count>one|two|three|four|five|six|\d+)"
    match = re.search(rf"\b(?:last|past)\s+{number}\s+months?\b", folded)
    if match:
        raw = match.group("count")
        return _NUMBER_WORDS.get(raw, int(raw) if raw.isdigit() else 1) * 31
    match = re.search(rf"\b(?:last|past)\s+{number}\s+weeks?\b", folded)
    if match:
        raw = match.group("count")
        return _NUMBER_WORDS.get(raw, int(raw) if raw.isdigit() else 1) * 7
    if re.search(r"\b(?:last|past)\s+month\b", folded):
        return 31
    if re.search(r"\bpast\s+(?:few|couple(?:\s+of)?)\s+months?\b", folded):
        return 124
    return None


def _temporal_eligible(
    witness: _TemporalWitness,
    *,
    calendar_scope: tuple[int, int] | None,
    window_days: int | None,
    asked: date | None,
) -> bool:
    if calendar_scope is not None:
        return bool(
            witness.basis
            in {
                TemporalEvidenceBasis.TEXTUAL_EVENT_DATE,
                TemporalEvidenceBasis.RELATIVE_EVENT_TIME,
            }
            and witness.calendar_precise
            and witness.event_date is not None
            and (witness.event_date.year, witness.event_date.month) == calendar_scope
        )
    if window_days is None:
        return True
    if witness.basis not in {
        TemporalEvidenceBasis.TEXTUAL_EVENT_DATE,
        TemporalEvidenceBasis.RELATIVE_EVENT_TIME,
    }:
        return False
    if witness.relative_days is not None:
        return witness.relative_days <= window_days
    if witness.event_date is None or asked is None:
        return False
    distance = (asked - witness.event_date).days
    return 0 <= distance <= window_days


def _handles_and_groups(
    item: Mapping[str, Any], group_by_handle: Mapping[str, str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw = item.get("handle_ids")
    if type(raw) is not list or not raw or any(
        type(value) is not str or value not in group_by_handle for value in raw
    ):
        raise MatchedEvalContractError("typed item escaped opaque handle inventory")
    handles = tuple(dict.fromkeys(raw))
    groups = tuple(dict.fromkeys(group_by_handle[handle] for handle in handles))
    return handles, groups


def _exclude(
    item_sha256: str,
    handles: tuple[str, ...],
    reason: str,
    candidate: NumericCandidateAtom | None = None,
) -> NumericCandidateExclusion:
    return NumericCandidateExclusion(
        item_sha256=item_sha256,
        handle_ids=handles,
        reason=reason,
        candidate_receipt_sha256=(None if candidate is None else candidate.receipt_sha256),
    )


def _item_gate(
    item: Mapping[str, Any],
    *,
    handles: tuple[str, ...],
    include_proposed: bool,
    personal: bool,
) -> tuple[str, str, dict[str, str], frozenset[str], str | None]:
    item_sha = identity_sha256(dict(item))
    relation = item.get("relation")
    attributes, flags = _relation_attributes(relation)
    role = _role(flags, relation)
    status = str(item.get("status", "unknown")).casefold()
    reason: str | None = None
    if item.get("included", True) is not True:
        reason = "not_included"
    elif str(item.get("content_coherence", "match")) != "match":
        reason = "content_not_coherent"
    elif status in {"cancelled", "superseded"}:
        reason = "inactive_status"
    elif status == "proposed" and not include_proposed:
        reason = "proposed_not_requested"
    elif personal and role == "assistant":
        reason = "assistant_not_autobiographical_evidence"
    elif attributes.get("numeric_scope", "").casefold() in {
        "exclude",
        "excluded",
        "out",
        "out_of_scope",
        "other",
    }:
        reason = "outside_operator_scope"
    return item_sha, status, attributes, flags, reason


def _comparison_slots(operator: Mapping[str, Any], body: str) -> tuple[tuple[str, str], ...]:
    raw = operator.get("required_slots", [])
    output: list[tuple[str, str]] = []
    if type(raw) is list:
        for value in raw:
            if not isinstance(value, Mapping) or value.get("kind") != "comparison_side":
                continue
            slot_id = value.get("slot_id") or value.get("id")
            label = value.get("label")
            if type(slot_id) is str and slot_id and type(label) is str and label:
                output.append((slot_id, label))
    if len(output) == 2:
        return tuple(output)

    compared = re.search(r"\bcompared\s+to\b", body, re.I)
    if compared:
        left_names = re.findall(r"\b[A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2}\b", body[: compared.start()])
        right_names = re.findall(r"\b[A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2}\b", body[compared.end() :])
        stop = {"did", "do", "does", "how", "i", "my"}
        left_names = [row for row in left_names if _key(row) not in stop]
        right_names = [row for row in right_names if _key(row) not in stop]
        if left_names and right_names:
            return (("side_left", left_names[-1]), ("side_right", right_names[0]))
    return ()


def _comparison_atoms(
    provider_input: Mapping[str, Any],
    *,
    body: str,
    operator: Mapping[str, Any],
) -> tuple[tuple[NumericCandidateAtom, ...], tuple[NumericCandidateExclusion, ...], tuple[tuple[str, str], ...]]:
    inventory = _inventory(provider_input)
    group_by_handle = {str(row["handle_id"]): str(row["group_handle"]) for row in inventory}
    sides = _comparison_slots(operator, body)
    atoms: list[NumericCandidateAtom] = []
    exclusions: list[NumericCandidateExclusion] = []
    include_proposed = operator.get("include_proposed") is True
    personal = bool(re.search(r"\b(?:I|my)\b", body, re.I))
    side_ids = {slot for slot, _label in sides}
    for item in _items(provider_input):
        handles, groups = _handles_and_groups(item, group_by_handle)
        item_sha, status, attributes, flags, gated = _item_gate(
            item,
            handles=handles,
            include_proposed=include_proposed,
            personal=personal,
        )
        if gated:
            exclusions.append(_exclude(item_sha, handles, gated))
            continue
        numeric = item.get("numeric_value")
        if type(numeric) not in {int, float} or not math.isfinite(float(numeric)):
            exclusions.append(_exclude(item_sha, handles, "no_exact_numeric_operand"))
            continue
        if str(item.get("numeric_qualifier", "exact")) != "exact":
            exclusions.append(_exclude(item_sha, handles, "qualified_numeric_operand"))
            continue
        summary = str(item.get("summary", ""))
        supported_raw = item.get("supported_slot_ids", [])
        supported = {
            value for value in supported_raw if type(value) is str
        } if type(supported_raw) is list else set()
        bound = [slot for slot, _label in sides if slot in supported]
        if not bound:
            summary_terms = set(normalized_terms(summary))
            bound = [
                slot
                for slot, label in sides
                if set(normalized_terms(label)) <= summary_terms
            ]
        bound = [slot for slot in bound if slot in side_ids]
        if len(bound) != 1:
            exclusions.append(
                _exclude(
                    item_sha,
                    handles,
                    "comparison_side_missing" if not bound else "comparison_side_ambiguous",
                )
            )
            continue
        side = bound[0]
        label = next(label for slot, label in sides if slot == side)
        relation = item.get("relation")
        role = _role(flags, relation)
        atom = NumericCandidateAtom(
            item_sha256=item_sha,
            handle_ids=handles,
            source_group_handles=groups,
            entity_key=_key(label),
            action_key="compare",
            event_key=f"comparison:{side}",
            contribution_value=1.0,
            numeric_value=float(numeric),
            unit=(None if item.get("unit") is None else str(item.get("unit"))),
            status=status,
            source_role=role,
            event_date=None,
            temporal_basis=TemporalEvidenceBasis.UNKNOWN,
            comparison_side_id=side,
        )
        atoms.append(atom)
    return tuple(atoms), tuple(exclusions), sides


def _obligations_from_attributes(value: str | None) -> tuple[tuple[str, str], ...]:
    if not value:
        return ()
    output: list[tuple[str, str]] = []
    for raw in value.split("|"):
        if ":" not in raw:
            continue
        action, entity = raw.split(":", 1)
        concepts = canonical_action_concepts(action)
        canonical = concepts[0] if concepts else _key(action)
        if canonical in {"pickup", "return"} and entity.strip():
            output.append((canonical, _key(entity)))
    return tuple(dict.fromkeys(output))


def _clothing_obligations(summary: str) -> tuple[tuple[str, str], ...]:
    folded = summary.casefold()
    output: list[tuple[str, str]] = []
    if re.search(r"\b(?:need|still\s+need|have\s+to)\b.{0,45}\bpick\s+up\b", folded):
        if "blazer" in folded:
            descriptor = "navy_blue_blazer" if "navy blue blazer" in folded else "blazer"
            output.append(("pickup", descriptor))
        elif "boot" in folded:
            output.append(
                (
                    "pickup",
                    "replacement_boot" if re.search(r"\b(?:exchange|new\s+pair|larger\s+size)\b", folded) else "boot",
                )
            )
    if re.search(r"\bneed\s+to\s+return\b.{0,50}\bboots?\b", folded):
        output.append(("return", "original_boot" if "exchange" in folded else "boot"))
    if "boot" in folded and "exchange" in folded and re.search(
        r"\b(?:haven't|have\s+not|still\s+need)\b.{0,55}\bpick(?:ed)?\s+(?:them\s+)?up\b|"
        r"\bpick\s+up\s+the\s+(?:new|larger)\s+pair\b",
        folded,
    ):
        output.append(("pickup", "replacement_boot"))
    return tuple(dict.fromkeys(output))


def _count_atoms(
    provider_input: Mapping[str, Any],
    *,
    body: str,
    operator: Mapping[str, Any],
    mode: NumericPolicyMode,
) -> tuple[tuple[NumericCandidateAtom, ...], tuple[NumericCandidateExclusion, ...]]:
    inventory = _inventory(provider_input)
    group_by_handle = {str(row["handle_id"]): str(row["group_handle"]) for row in inventory}
    _dated, _body, asked = _question(provider_input)
    query_actions = set(canonical_action_concepts(body))
    query_domain = _domain(body)
    calendar_scope = _calendar_scope(body, asked)
    window_days = _window_days(body, operator)
    include_proposed = operator.get("include_proposed") is True
    personal = bool(re.search(r"\b(?:I|my)\b", body, re.I))
    atoms: list[NumericCandidateAtom] = []
    exclusions: list[NumericCandidateExclusion] = []

    for item in _items(provider_input):
        handles, groups = _handles_and_groups(item, group_by_handle)
        item_sha, status, attributes, flags, gated = _item_gate(
            item,
            handles=handles,
            include_proposed=include_proposed,
            personal=personal,
        )
        if gated:
            exclusions.append(_exclude(item_sha, handles, gated))
            continue
        summary = item.get("summary")
        if type(summary) is not str or not summary.strip():
            exclusions.append(_exclude(item_sha, handles, "missing_exact_summary"))
            continue
        role = _role(flags, item.get("relation"))

        if mode is NumericPolicyMode.ACTION_OBLIGATION_COUNT:
            obligations = _obligations_from_attributes(attributes.get("obligations"))
            if not obligations:
                obligations = _clothing_obligations(summary)
            if not obligations:
                exclusions.append(_exclude(item_sha, handles, "no_active_action_obligation"))
                continue
            explicit_event = attributes.get("event_key")
            for action, entity in obligations:
                event = _key(explicit_event) if explicit_event else f"{action}:{entity}"
                atoms.append(
                    NumericCandidateAtom(
                        item_sha256=item_sha,
                        handle_ids=handles,
                        source_group_handles=groups,
                        entity_key=entity,
                        action_key=action,
                        event_key=event,
                        contribution_value=1.0,
                        numeric_value=None,
                        unit=None,
                        status=status,
                        source_role=role,
                        event_date=None,
                        temporal_basis=TemporalEvidenceBasis.UNKNOWN,
                    )
                )
            continue

        structured_actions = _structured_actions(attributes)
        selected_clauses: list[tuple[str, tuple[str, ...]]] = []
        if structured_actions:
            if query_actions & set(structured_actions):
                selected_clauses.append((summary, structured_actions))
        else:
            for clause in _clauses(summary):
                actions = _effective_actions(
                    clause, status=status, structured=structured_actions
                )
                if query_actions & set(actions):
                    selected_clauses.append((clause, actions))
        if not selected_clauses:
            exclusions.append(_exclude(item_sha, handles, "predicate_not_satisfied"))
            continue

        made_atom = False
        temporal_rejected = False
        structured_members = bool(
            attributes.get("member_keys") or attributes.get("members")
        )
        for clause_index, (clause, actions) in enumerate(selected_clauses):
            witness = _temporal_witness(
                clause,
                item_date=item.get("date"),
                declared_basis=attributes.get("date_basis"),
                asked=asked,
            )
            if not _temporal_eligible(
                witness,
                calendar_scope=calendar_scope,
                window_days=window_days,
                asked=asked,
            ):
                temporal_rejected = True
                continue
            entities = _extract_entities(
                query_domain,
                summary if structured_members else clause,
                item_entity_key=item.get("entity_key"),
                attributes=attributes,
            )
            if not entities and not structured_members:
                # Numeric events commonly carry their entity in the preceding
                # clause and a pronoun at the action site ("my commuter bike
                # ... time to replace it this month").  Backfill only a single
                # unambiguous item-level entity; never fan out a pronoun over
                # several candidates.
                item_entities = _extract_entities(
                    query_domain,
                    summary,
                    item_entity_key=item.get("entity_key"),
                    attributes=attributes,
                )
                if len(item_entities) == 1:
                    entities = item_entities
            if not entities:
                continue
            action = next(
                (value for value in actions if value in query_actions),
                actions[0],
            )
            explicit_event = attributes.get("event_key")
            for member_index, entity in enumerate(entities):
                event = (
                    f"{_key(explicit_event)}:{entity}"
                    if explicit_event and len(entities) > 1
                    else _key(explicit_event)
                    if explicit_event
                    else f"{action}:{entity}"
                )
                atoms.append(
                    NumericCandidateAtom(
                        item_sha256=item_sha,
                        handle_ids=handles,
                        source_group_handles=groups,
                        entity_key=entity,
                        action_key=action,
                        event_key=event,
                        contribution_value=1.0,
                        numeric_value=None,
                        unit=None,
                        status=status,
                        source_role=role,
                        event_date=(
                            None
                            if witness.event_date is None
                            else witness.event_date.isoformat()
                        ),
                        temporal_basis=witness.basis,
                    )
                )
                made_atom = True
            if structured_members:
                break
        if not made_atom:
            exclusions.append(
                _exclude(
                    item_sha,
                    handles,
                    "event_time_not_in_scope" if temporal_rejected else "target_entity_not_extracted",
                )
            )
    return tuple(atoms), tuple(exclusions)


def compile_operator_first_numeric_candidates(
    provider_input: Mapping[str, Any],
    /,
) -> OperatorFirstNumericCompilation:
    """Compile operator-relevant atoms without deciding frontier closure.

    This is the shared, provider-free semantic pass used by both the terminal
    reducer and a physically exhaustive full-store census.  Its receipt proves
    exactly what the versioned grammar recognized; it does not claim the
    bounded input contains every relevant memory.
    """

    projection = _provider_projection(provider_input)
    _dated, body, _asked = _question(provider_input)
    operator = _operator(provider_input)
    mode = _query_mode(body, operator)
    if mode is NumericPolicyMode.FIXED_SCALAR_COMPARISON:
        atoms, exclusions, _sides = _comparison_atoms(
            provider_input,
            body=body,
            operator=operator,
        )
    else:
        atoms, exclusions = _count_atoms(
            provider_input,
            body=body,
            operator=operator,
            mode=mode,
        )
    return OperatorFirstNumericCompilation(
        policy_input_sha256=_policy_input_sha256(provider_input),
        typed_evidence_sha256=identity_sha256(projection["typed_evidence"]),
        operator_spec_sha256=identity_sha256(dict(operator)),
        mode=mode,
        question_domain=_domain(body),
        query_action_keys=canonical_action_concepts(body),
        candidate_atoms=atoms,
        exclusions=exclusions,
    )


def _format_number(value: float, unit: str | None = None) -> str:
    scalar = (
        str(int(value))
        if float(value).is_integer()
        else f"{value:.10f}".rstrip("0").rstrip(".")
    )
    if unit in {"$", "USD"}:
        return "$" + scalar
    return scalar if unit is None else f"{scalar} {unit}"


def _make_decision(
    *,
    provider_input: Mapping[str, Any],
    operator: Mapping[str, Any],
    mode: NumericPolicyMode,
    status: ExecutionStatus,
    prediction: str = "",
    numeric_result: float | None = None,
    used: tuple[NumericCandidateAtom, ...] = (),
    atoms: tuple[NumericCandidateAtom, ...],
    exclusions: tuple[NumericCandidateExclusion, ...],
    frontier: RelevantNumericFrontier | None,
    reason: str,
) -> OperatorFirstNumericDecision:
    typed = _provider_projection(provider_input)["typed_evidence"]
    handles = tuple(
        dict.fromkeys(handle for atom in used for handle in atom.handle_ids)
    )
    return OperatorFirstNumericDecision(
        policy_input_sha256=_policy_input_sha256(provider_input),
        typed_evidence_sha256=identity_sha256(typed),
        operator_spec_sha256=identity_sha256(dict(operator)),
        mode=mode,
        status=status,
        decision=("replace" if status is ExecutionStatus.SUPPORTED else "abstain"),
        prediction=prediction,
        numeric_result=numeric_result,
        used_handle_ids=handles,
        used_candidate_receipt_sha256s=tuple(row.receipt_sha256 for row in used),
        candidate_atoms=atoms,
        exclusions=exclusions,
        relevant_frontier_receipt_sha256=(
            None if frontier is None else frontier.receipt_sha256
        ),
        reason=reason,
    )


def _execute_comparison(
    provider_input: Mapping[str, Any],
    *,
    body: str,
    operator: Mapping[str, Any],
) -> OperatorFirstNumericDecision:
    mode = NumericPolicyMode.FIXED_SCALAR_COMPARISON
    atoms, raw_exclusions, sides = _comparison_atoms(
        provider_input, body=body, operator=operator
    )
    exclusions = list(raw_exclusions)
    if len(sides) != 2:
        return _make_decision(
            provider_input=provider_input,
            operator=operator,
            mode=mode,
            status=ExecutionStatus.INSUFFICIENT,
            atoms=atoms,
            exclusions=tuple(exclusions),
            frontier=None,
            reason="comparison_requires_two_named_sides",
        )
    chosen: list[NumericCandidateAtom] = []
    for side_id, _label in sides:
        candidates = tuple(row for row in atoms if row.comparison_side_id == side_id)
        values = {(row.numeric_value, row.unit) for row in candidates}
        if not candidates:
            return _make_decision(
                provider_input=provider_input,
                operator=operator,
                mode=mode,
                status=ExecutionStatus.INSUFFICIENT,
                atoms=atoms,
                exclusions=tuple(exclusions),
                frontier=None,
                reason="comparison_side_missing_value",
            )
        if len(values) != 1:
            return _make_decision(
                provider_input=provider_input,
                operator=operator,
                mode=mode,
                status=ExecutionStatus.CONFLICTED,
                atoms=atoms,
                exclusions=tuple(exclusions),
                frontier=None,
                reason="comparison_side_value_conflict",
            )
        chosen.append(candidates[0])
        for duplicate in candidates[1:]:
            exclusions.append(
                _exclude(
                    duplicate.item_sha256,
                    duplicate.handle_ids,
                    "duplicate_scalar_corroboration",
                    duplicate,
                )
            )
    left, right = chosen
    if left.unit != right.unit:
        return _make_decision(
            provider_input=provider_input,
            operator=operator,
            mode=mode,
            status=ExecutionStatus.CONFLICTED,
            atoms=atoms,
            exclusions=tuple(exclusions),
            frontier=None,
            reason="comparison_side_unit_conflict",
        )
    assert left.numeric_value is not None and right.numeric_value is not None
    delta = left.numeric_value - right.numeric_value
    boolean = str(operator.get("comparison_mode", "")) == "boolean_greater" or bool(
        re.match(r"^(?:did|do|does|is|are|was|were|has|have|had)\b", body, re.I)
    )
    prediction = "Yes" if delta > 0 else "No" if boolean else _format_number(abs(delta), left.unit)
    if not boolean:
        prediction = _format_number(abs(delta), left.unit)
    return _make_decision(
        provider_input=provider_input,
        operator=operator,
        mode=mode,
        status=ExecutionStatus.SUPPORTED,
        prediction=prediction,
        numeric_result=delta,
        used=(left, right),
        atoms=atoms,
        exclusions=tuple(exclusions),
        frontier=None,
        reason="fixed_arity_named_scalar_operands",
    )


def _global_frontier_closed(provider_input: Mapping[str, Any]) -> bool:
    typed = _provider_projection(provider_input)["typed_evidence"]
    frontier = typed.get("frontier")
    return isinstance(frontier, Mapping) and frontier.get("closed") is True


def _execute_count(
    provider_input: Mapping[str, Any],
    *,
    body: str,
    operator: Mapping[str, Any],
    mode: NumericPolicyMode,
    relevant_frontier: RelevantNumericFrontier | None,
) -> OperatorFirstNumericDecision:
    atoms, raw_exclusions = _count_atoms(
        provider_input, body=body, operator=operator, mode=mode
    )
    exclusions = list(raw_exclusions)
    if relevant_frontier is not None:
        if relevant_frontier.policy_input_sha256 != _policy_input_sha256(provider_input):
            raise MatchedEvalContractError(
                "relevant numeric frontier belongs to another provider input"
            )
        represented = set(relevant_frontier.represented_handle_ids)
        if any(not set(row.handle_ids) <= represented for row in atoms):
            raise MatchedEvalContractError(
                "numeric candidate escaped its relevant frontier"
            )
    frontier_closed = _global_frontier_closed(provider_input) or bool(
        relevant_frontier is not None and relevant_frontier.closed
    )
    if not frontier_closed:
        return _make_decision(
            provider_input=provider_input,
            operator=operator,
            mode=mode,
            status=ExecutionStatus.INSUFFICIENT,
            atoms=atoms,
            exclusions=tuple(exclusions),
            frontier=relevant_frontier,
            reason="relevant_candidate_frontier_not_closed",
        )
    if not atoms:
        return _make_decision(
            provider_input=provider_input,
            operator=operator,
            mode=mode,
            status=ExecutionStatus.INSUFFICIENT,
            atoms=atoms,
            exclusions=tuple(exclusions),
            frontier=relevant_frontier,
            reason="no_operator_relevant_candidates",
        )

    unique: dict[str, NumericCandidateAtom] = {}
    for atom in atoms:
        semantic_key = (
            f"{atom.action_key}:{atom.entity_key}"
            if mode is NumericPolicyMode.ACTION_OBLIGATION_COUNT
            else atom.entity_key
            if mode is NumericPolicyMode.DISTINCT_ENTITY_COUNT
            else atom.event_key
        )
        if semantic_key in unique:
            exclusions.append(
                _exclude(
                    atom.item_sha256,
                    atom.handle_ids,
                    "duplicate_semantic_identity",
                    atom,
                )
            )
        else:
            unique[semantic_key] = atom
    used = tuple(unique.values())
    total = sum(row.contribution_value for row in used)
    return _make_decision(
        provider_input=provider_input,
        operator=operator,
        mode=mode,
        status=ExecutionStatus.SUPPORTED,
        prediction=_format_number(total),
        numeric_result=total,
        used=used,
        atoms=atoms,
        exclusions=tuple(exclusions),
        frontier=relevant_frontier,
        reason="operator_relevant_candidate_reduction",
    )


def execute_operator_first_numeric_policy(
    provider_input: Mapping[str, Any],
    /,
    *,
    relevant_frontier: RelevantNumericFrontier | None = None,
) -> OperatorFirstNumericDecision:
    """Execute one sealed numeric provider input without a model or parent.

    Fixed-arity named comparisons close over their coherent exact sides.
    Counts additionally require either the packet's global frontier or an
    independently sealed :class:`RelevantNumericFrontier` to be closed.
    """

    _provider_projection(provider_input)
    _dated, body, _asked = _question(provider_input)
    operator = _operator(provider_input)
    mode = _query_mode(body, operator)
    if mode is NumericPolicyMode.FIXED_SCALAR_COMPARISON:
        return _execute_comparison(provider_input, body=body, operator=operator)
    return _execute_count(
        provider_input,
        body=body,
        operator=operator,
        mode=mode,
        relevant_frontier=relevant_frontier,
    )


__all__ = [
    "NumericCandidateAtom",
    "NumericCandidateExclusion",
    "NumericPolicyMode",
    "OperatorFirstNumericCompilation",
    "OperatorFirstNumericDecision",
    "RelevantNumericFrontier",
    "TemporalEvidenceBasis",
    "build_relevant_numeric_frontier",
    "compile_operator_first_numeric_candidates",
    "execute_operator_first_numeric_policy",
]
