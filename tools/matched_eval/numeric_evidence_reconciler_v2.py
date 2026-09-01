"""Actual-schema numeric reconciliation over sealed compact provider input.

V2 preserves the V1 result verbatim when V1 is decisive.  When V1 is
insufficient, it can additionally project two deliberately narrow compiler
schemas already present in sealed typed summaries:

* explicit distinct-category rows; and
* an explicit recurring aggregate plus named recurring additions.

The projection is provider-free, gold-blind, question-ID-blind, and emits only
hashed semantic identities and opaque evidence handles in its proof receipt.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import assert_gold_blind, identity_sha256, require_sha256, require_text
from .numeric_evidence_reconciler import (
    NumericEvidenceReconcilerError,
    NumericEvidenceReconciliationReceipt,
    ReconciliationMode,
    ReconciliationStatus,
    RejectedNumericOperandProof,
    reconcile_sealed_numeric_evidence,
)


FORMAT = "memory-condense-sealed-numeric-evidence-reconciliation-v2"
PROJECTED_ITEM_FORMAT = f"{FORMAT}-projected-item-v1"
CONTRIBUTION_FORMAT = f"{FORMAT}-contribution-v1"
POLICY_ID = "sealed_typed_numeric_reconciliation_v2_actual_schema"

_HANDLE_RE = re.compile(r"^H[0-9]{3,6}$")
_DISTINCT_QUESTION_RE = re.compile(
    r"\bhow\s+many\s+(?:different|distinct)\s+types?\s+of\s+"
    r"(?P<target>.+?)\s+(?:have|do|did|are|were)\b",
    re.I,
)
_DISTINCT_SUMMARY_RE = re.compile(
    r"^(?P<value>[1-9][0-9]*)\s+distinct\s+"
    r"(?P<category>[A-Za-z][^:;]{1,100}):\s*"
    r"(?P<label>[^;]{1,160});\s*positive;\s*eligible;\s*"
    r"recent-use\s+context(?:\s*\([^)]*\))?\.$",
    re.I,
)
_RECURRING_QUESTION_RE = re.compile(
    r"\bhow\s+many\s+(?P<target>.+?)\s+do\s+I\s+attend\s+in\s+a\s+"
    r"typical\s+(?P<period>day|week|month|year)\b",
    re.I,
)
_RECURRING_BASE_RE = re.compile(
    r"^(?P<value>[1-9][0-9]*)\s+(?P<category>[A-Za-z][^;]{1,100}?)\s+"
    r"per\s+(?P<period>day|week|month|year);\s*explicit\s+"
    r"(?P<period_word>daily|weekly|monthly|yearly)\s+attendance\s+frequency;\s*"
    r"eligible\s+aggregate\s+operand\.$",
    re.I,
)
_RECURRING_ADDITION_RE = re.compile(
    r"^(?P<value>[1-9][0-9]*)\s+(?P<label>[A-Za-z][^;]{1,100}?)\s+"
    r"class(?:es)?\s+per\s+(?P<period>day|week|month|year),\s*inferred\s+from\s+a\s+"
    r"recurring\s+(?P<schedule>[^;]{1,100}?)\s+schedule;\s*"
    r"(?P<disposition>eligible|potentially\s+corroborative)\s+"
    r"(?P<period_word>daily|weekly|monthly|yearly)-class\s+operand\.$",
    re.I,
)
_PERIOD_WORD = {
    "day": "daily",
    "week": "weekly",
    "month": "monthly",
    "year": "yearly",
}
_TERM_RE = re.compile(r"[a-z0-9]+", re.I)
_GENERIC_CATEGORY_TERMS = frozenset({"different", "distinct", "type"})
_SEMANTIC_RELATION_KEYS = (
    "corroborates",
    "event_key",
    "dedup_key",
    "cardinality_key",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NumericEvidenceReconcilerError(message)


def _terms(value: str) -> tuple[str, ...]:
    output: list[str] = []
    for raw in _TERM_RE.findall(value.casefold()):
        token = raw
        if token.endswith("sses"):
            token = token[:-2]
        elif token.endswith("ies") and len(token) > 3:
            token = f"{token[:-3]}y"
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            token = token[:-1]
        if token not in output:
            output.append(token)
    return tuple(output)


def _category_matches(left: str, right: str) -> bool:
    left_terms = set(_terms(left)) - _GENERIC_CATEGORY_TERMS
    right_terms = set(_terms(right)) - _GENERIC_CATEGORY_TERMS
    return bool(left_terms) and (
        left_terms <= right_terms or right_terms <= left_terms
    )


def _sealed_relation_identity(item: Mapping[str, Any]) -> str | None:
    relation = item.get("relation")
    if type(relation) is not str:
        return None
    fields: dict[str, str] = {}
    for part in relation.split(";"):
        key, separator, value = part.partition("=")
        if not separator:
            continue
        normalized_key = key.strip().casefold().replace("-", "_")
        normalized_value = value.strip().casefold()
        if normalized_key and normalized_value:
            fields[normalized_key] = normalized_value
    return next(
        (fields[key] for key in _SEMANTIC_RELATION_KEYS if key in fields),
        None,
    )


def _finite_exact_number(value: object) -> float | None:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        return None
    return float(value)


def _item_handles(item: Mapping[str, Any]) -> tuple[str, ...]:
    raw = item.get("handle_ids")
    _require(
        type(raw) is list
        and bool(raw)
        and len(set(raw)) == len(raw)
        and all(type(value) is str and _HANDLE_RE.fullmatch(value) for value in raw),
        "V2 projected item handles changed schema",
    )
    return tuple(raw)


def _basic_projection_eligible(item: Mapping[str, Any]) -> bool:
    return (
        item.get("included") is True
        and item.get("content_coherence") == "match"
        and item.get("status") not in {"cancelled", "proposed"}
        and item.get("value_authority") == "explicit"
        and item.get("numeric_qualifier", "exact") == "exact"
        and item.get("numeric_role", "operand") == "operand"
        and item.get("kind") == "operand"
    )


@dataclass(frozen=True, slots=True)
class _ProjectedCandidate:
    item_projection_sha256: str
    handle_ids: tuple[str, ...]
    projection_rule: str
    semantic_key_sha256: str
    numeric_role: str
    numeric_value: float
    unit: str
    disposition: str


@dataclass(frozen=True, slots=True)
class ProjectedNumericItemProof:
    item_projection_sha256: str
    handle_ids: tuple[str, ...]
    projection_rule: str
    semantic_key_sha256: str
    numeric_role: str
    numeric_value: float
    unit: str
    disposition: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.item_projection_sha256, "V2 projected numeric item")
        require_sha256(self.semantic_key_sha256, "V2 projected semantic key")
        _require(
            type(self.handle_ids) is tuple
            and bool(self.handle_ids)
            and len(set(self.handle_ids)) == len(self.handle_ids)
            and all(_HANDLE_RE.fullmatch(value) for value in self.handle_ids),
            "V2 projected handles changed",
        )
        require_text(self.projection_rule, "V2 projection rule")
        _require(
            self.numeric_role in {"baseline", "delta", "operand"},
            "V2 projected numeric role changed",
        )
        _require(
            math.isfinite(self.numeric_value),
            "V2 projected numeric value changed",
        )
        require_text(self.unit, "V2 projected unit")
        _require(
            self.disposition in {"primary", "corroborating"},
            "V2 projected disposition changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise NumericEvidenceReconcilerError("V2 projected item receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="numeric_v2_projected_item")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "disposition": self.disposition,
            "format": PROJECTED_ITEM_FORMAT,
            "handle_ids": list(self.handle_ids),
            "item_projection_sha256": self.item_projection_sha256,
            "numeric_role": self.numeric_role,
            "numeric_value": self.numeric_value,
            "projection_rule": self.projection_rule,
            "semantic_key_sha256": self.semantic_key_sha256,
            "unit": self.unit,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ProjectedNumericContributionProof:
    semantic_key_sha256: str
    projection_rule: str
    numeric_role: str
    numeric_value: float
    unit: str
    handle_ids: tuple[str, ...]
    item_projection_sha256s: tuple[str, ...]
    corroborated_duplicate_count: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.semantic_key_sha256, "V2 contribution semantic key")
        require_text(self.projection_rule, "V2 contribution projection rule")
        _require(
            self.numeric_role in {"baseline", "delta", "operand"},
            "V2 contribution numeric role changed",
        )
        _require(math.isfinite(self.numeric_value), "V2 contribution value changed")
        require_text(self.unit, "V2 contribution unit")
        _require(
            type(self.handle_ids) is tuple
            and bool(self.handle_ids)
            and len(set(self.handle_ids)) == len(self.handle_ids)
            and all(_HANDLE_RE.fullmatch(value) for value in self.handle_ids),
            "V2 contribution handles changed",
        )
        _require(
            type(self.item_projection_sha256s) is tuple
            and bool(self.item_projection_sha256s)
            and len(set(self.item_projection_sha256s))
            == len(self.item_projection_sha256s),
            "V2 contribution item identities changed",
        )
        for digest in self.item_projection_sha256s:
            require_sha256(digest, "V2 contribution item")
        _require(
            type(self.corroborated_duplicate_count) is int
            and self.corroborated_duplicate_count
            == len(self.item_projection_sha256s) - 1,
            "V2 contribution deduplication accounting changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise NumericEvidenceReconcilerError("V2 contribution receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="numeric_v2_contribution")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "corroborated_duplicate_count": self.corroborated_duplicate_count,
            "format": CONTRIBUTION_FORMAT,
            "handle_ids": list(self.handle_ids),
            "item_projection_sha256s": list(self.item_projection_sha256s),
            "numeric_role": self.numeric_role,
            "numeric_value": self.numeric_value,
            "projection_rule": self.projection_rule,
            "semantic_key_sha256": self.semantic_key_sha256,
            "unit": self.unit,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class NumericEvidenceReconciliationV2Receipt:
    base_reconciliation: NumericEvidenceReconciliationReceipt
    status: ReconciliationStatus
    mode: ReconciliationMode
    reason: str
    numeric_result: float | None
    unit: str | None
    comparison_relation: str | None
    boolean_result: bool | None
    projection_rule: str
    projected_items: tuple[ProjectedNumericItemProof, ...]
    contributions: tuple[ProjectedNumericContributionProof, ...]
    projection_rejections: tuple[RejectedNumericOperandProof, ...]
    used_handle_ids: tuple[str, ...]
    deduplicated_item_count: int
    provider_prompt_count: int = 0
    retained_transformer_token_state_bytes: int = 0
    gold_loaded: bool = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.base_reconciliation) is NumericEvidenceReconciliationReceipt,
            "V2 base reconciliation changed type",
        )
        _require(type(self.status) is ReconciliationStatus, "V2 status changed")
        _require(type(self.mode) is ReconciliationMode, "V2 mode changed")
        require_text(self.reason, "V2 reconciliation reason")
        require_text(self.projection_rule, "V2 reconciliation projection rule")
        if self.numeric_result is not None:
            _require(math.isfinite(self.numeric_result), "V2 result changed")
        if self.unit is not None:
            require_text(self.unit, "V2 result unit")
        _require(
            type(self.projected_items) is tuple
            and all(type(row) is ProjectedNumericItemProof for row in self.projected_items),
            "V2 projected item proof changed",
        )
        _require(
            type(self.contributions) is tuple
            and all(
                type(row) is ProjectedNumericContributionProof
                for row in self.contributions
            ),
            "V2 contribution proof changed",
        )
        _require(
            type(self.projection_rejections) is tuple
            and all(
                type(row) is RejectedNumericOperandProof
                for row in self.projection_rejections
            ),
            "V2 projection rejections changed",
        )
        expected_handles = (
            self.base_reconciliation.used_handle_ids
            if self.projection_rule == "sealed_typed_fields"
            else tuple(
                dict.fromkeys(
                    handle for row in self.contributions for handle in row.handle_ids
                )
            )
        )
        _require(
            self.used_handle_ids == expected_handles,
            "V2 used handles do not reconcile to proof",
        )
        expected_dedup = (
            self.base_reconciliation.deduplicated_item_count
            if self.projection_rule == "sealed_typed_fields"
            else sum(row.corroborated_duplicate_count for row in self.contributions)
        )
        _require(
            self.deduplicated_item_count == expected_dedup,
            "V2 deduplication accounting changed",
        )
        _require(
            self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "V2 provider-free/gold-blind invariants changed",
        )
        if self.status is ReconciliationStatus.SUPPORTED:
            _require(self.numeric_result is not None, "V2 supported result disappeared")
            _require(
                self.base_reconciliation.supported or bool(self.contributions),
                "V2 supported proof disappeared",
            )
        else:
            _require(
                self.numeric_result is None and not self.contributions,
                "V2 unsupported receipt carried a result",
            )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise NumericEvidenceReconcilerError("V2 reconciliation receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="numeric_reconciliation_v2")

    @property
    def supported(self) -> bool:
        return self.status is ReconciliationStatus.SUPPORTED

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "base_reconciliation": self.base_reconciliation.projection(),
            "boolean_result": self.boolean_result,
            "comparison_relation": self.comparison_relation,
            "contributions": [row.projection() for row in self.contributions],
            "deduplicated_item_count": self.deduplicated_item_count,
            "format": FORMAT,
            "gold_loaded": False,
            "mode": self.mode.value,
            "numeric_result": self.numeric_result,
            "policy_id": POLICY_ID,
            "projected_items": [row.projection() for row in self.projected_items],
            "projection_rejections": [
                row.projection() for row in self.projection_rejections
            ],
            "projection_rule": self.projection_rule,
            "provider_prompt_count": 0,
            "reason": self.reason,
            "retained_transformer_token_state_bytes": 0,
            "status": self.status.value,
            "unit": self.unit,
            "used_handle_ids": list(self.used_handle_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _candidate_proofs(
    candidates: Sequence[_ProjectedCandidate],
) -> tuple[ProjectedNumericItemProof, ...]:
    return tuple(
        ProjectedNumericItemProof(
            row.item_projection_sha256,
            row.handle_ids,
            row.projection_rule,
            row.semantic_key_sha256,
            row.numeric_role,
            row.numeric_value,
            row.unit,
            row.disposition,
        )
        for row in candidates
    )


def _group_contributions(
    candidates: Sequence[_ProjectedCandidate],
    *,
    require_primary: bool,
) -> tuple[str, tuple[ProjectedNumericContributionProof, ...]]:
    grouped: dict[str, list[_ProjectedCandidate]] = {}
    for row in candidates:
        grouped.setdefault(row.semantic_key_sha256, []).append(row)
    output: list[ProjectedNumericContributionProof] = []
    for key, members in grouped.items():
        first = members[0]
        if any(
            row.numeric_value != first.numeric_value
            or row.numeric_role != first.numeric_role
            or row.unit != first.unit
            or row.projection_rule != first.projection_rule
            for row in members[1:]
        ):
            return "projected_semantic_numeric_conflict", ()
        if require_primary and not any(row.disposition == "primary" for row in members):
            return "projected_corroboration_without_primary", ()
        item_hashes = tuple(
            dict.fromkeys(row.item_projection_sha256 for row in members)
        )
        output.append(
            ProjectedNumericContributionProof(
                key,
                first.projection_rule,
                first.numeric_role,
                first.numeric_value,
                first.unit,
                tuple(
                    dict.fromkeys(
                        handle for row in members for handle in row.handle_ids
                    )
                ),
                item_hashes,
                len(item_hashes) - 1,
            )
        )
    return "supported", tuple(output)


def _h700_rejections(items: Sequence[Mapping[str, Any]]) -> tuple[RejectedNumericOperandProof, ...]:
    output: list[RejectedNumericOperandProof] = []
    for item in items:
        handles = _item_handles(item)
        if any(handle.startswith("H700") for handle in handles):
            output.append(
                RejectedNumericOperandProof(
                    identity_sha256(item),
                    handles,
                    "generic_lexical_h700_operand",
                )
            )
    return tuple(output)


def _distinct_projection(
    question: str,
    items: Sequence[Mapping[str, Any]],
) -> tuple[
    str,
    tuple[_ProjectedCandidate, ...],
    tuple[ProjectedNumericContributionProof, ...],
    float | None,
    str | None,
]:
    question_match = _DISTINCT_QUESTION_RE.search(question)
    if question_match is None:
        return "not_applicable", (), (), None, None
    candidates: list[_ProjectedCandidate] = []
    unit: str | None = None
    for item in items:
        if not _basic_projection_eligible(item):
            continue
        handles = _item_handles(item)
        if any(handle.startswith("H700") for handle in handles):
            continue
        summary = item.get("summary")
        if type(summary) is not str:
            continue
        match = _DISTINCT_SUMMARY_RE.fullmatch(summary)
        if match is None or not _category_matches(
            match.group("category"), question_match.group("target")
        ):
            continue
        parsed = float(match.group("value"))
        typed = item.get("numeric_value")
        if typed is not None and _finite_exact_number(typed) != parsed:
            return "projected_summary_numeric_conflict", tuple(candidates), (), None, None
        category_terms = tuple(
            term for term in _terms(match.group("category")) if term != "type"
        )
        candidate_unit = " ".join(category_terms) + " type"
        if unit is not None and candidate_unit != unit:
            return "projected_distinct_category_conflict", tuple(candidates), (), None, None
        unit = candidate_unit
        explicit_identity = _sealed_relation_identity(item)
        semantic_key = identity_sha256(
            {
                "category_terms": list(category_terms),
                "identity": (
                    {"sealed_relation_key": explicit_identity}
                    if explicit_identity is not None
                    else {"label_terms": list(_terms(match.group("label")))}
                ),
                "projection_rule": "strict_distinct_summary",
            }
        )
        candidates.append(
            _ProjectedCandidate(
                identity_sha256(item),
                handles,
                "strict_distinct_summary",
                semantic_key,
                "operand",
                parsed,
                candidate_unit,
                "primary",
            )
        )
    if not candidates:
        return "strict_distinct_summary_operands_missing", (), (), None, None
    status, contributions = _group_contributions(candidates, require_primary=False)
    if status != "supported":
        return status, tuple(candidates), (), None, None
    _require(unit is not None, "projected distinct unit disappeared")
    return (
        "supported",
        tuple(candidates),
        contributions,
        sum(row.numeric_value for row in contributions),
        unit,
    )


def _recurring_projection(
    question: str,
    items: Sequence[Mapping[str, Any]],
) -> tuple[
    str,
    tuple[_ProjectedCandidate, ...],
    tuple[ProjectedNumericContributionProof, ...],
    float | None,
    str | None,
]:
    question_match = _RECURRING_QUESTION_RE.search(question)
    if question_match is None:
        return "not_applicable", (), (), None, None
    period = question_match.group("period").casefold()
    base_candidates: list[_ProjectedCandidate] = []
    addition_candidates: list[_ProjectedCandidate] = []
    unit: str | None = None
    for item in items:
        if not _basic_projection_eligible(item):
            continue
        handles = _item_handles(item)
        if any(handle.startswith("H700") for handle in handles):
            continue
        summary = item.get("summary")
        typed = _finite_exact_number(item.get("numeric_value"))
        if type(summary) is not str or typed is None:
            continue
        base_match = _RECURRING_BASE_RE.fullmatch(summary)
        if base_match is not None:
            if (
                base_match.group("period").casefold() != period
                or base_match.group("period_word").casefold() != _PERIOD_WORD[period]
                or not _category_matches(
                    base_match.group("category"), question_match.group("target")
                )
            ):
                continue
            parsed = float(base_match.group("value"))
            if parsed != typed:
                return "projected_summary_numeric_conflict", (), (), None, None
            category_terms = _terms(base_match.group("category"))
            unit = f"{'_'.join(category_terms)}/{period}"
            base_candidates.append(
                _ProjectedCandidate(
                    identity_sha256(item),
                    handles,
                    "strict_recurring_summary_base",
                    identity_sha256(
                        {
                            "category_terms": list(category_terms),
                            "period": period,
                            "projection_rule": "strict_recurring_summary_base",
                        }
                    ),
                    "baseline",
                    parsed,
                    unit,
                    "primary",
                )
            )
            continue
        addition_match = _RECURRING_ADDITION_RE.fullmatch(summary)
        if addition_match is None:
            continue
        if (
            addition_match.group("period").casefold() != period
            or addition_match.group("period_word").casefold() != _PERIOD_WORD[period]
            or "class" not in _terms(question_match.group("target"))
        ):
            continue
        parsed = float(addition_match.group("value"))
        if parsed != typed:
            return "projected_summary_numeric_conflict", (), (), None, None
        explicit_identity = _sealed_relation_identity(item)
        semantic_key = identity_sha256(
            {
                "identity": (
                    {"sealed_relation_key": explicit_identity}
                    if explicit_identity is not None
                    else {
                        "label_terms": list(_terms(addition_match.group("label"))),
                        "schedule_terms": list(
                            _terms(addition_match.group("schedule"))
                        ),
                    }
                ),
                "period": period,
                "projection_rule": "strict_recurring_summary_addition",
            }
        )
        addition_candidates.append(
            _ProjectedCandidate(
                identity_sha256(item),
                handles,
                "strict_recurring_summary_addition",
                semantic_key,
                "delta",
                parsed,
                f"class/{period}",
                (
                    "primary"
                    if addition_match.group("disposition").casefold() == "eligible"
                    else "corroborating"
                ),
            )
        )
    all_candidates = (*base_candidates, *addition_candidates)
    if not base_candidates:
        return "strict_recurring_summary_base_missing", all_candidates, (), None, None
    if unit is None:
        return "strict_recurring_summary_unit_missing", all_candidates, (), None, None
    # Normalize addition units to the aggregate's answer unit after the strict
    # question/period contract has established dimensional compatibility.
    addition_candidates = [
        _ProjectedCandidate(
            row.item_projection_sha256,
            row.handle_ids,
            row.projection_rule,
            row.semantic_key_sha256,
            row.numeric_role,
            row.numeric_value,
            unit,
            row.disposition,
        )
        for row in addition_candidates
    ]
    all_candidates = (*base_candidates, *addition_candidates)
    base_status, bases = _group_contributions(base_candidates, require_primary=False)
    if base_status != "supported" or len(bases) != 1:
        return "projected_recurring_base_conflict", all_candidates, (), None, None
    addition_status, additions = _group_contributions(
        addition_candidates,
        require_primary=True,
    )
    if addition_status != "supported":
        return addition_status, all_candidates, (), None, None
    if not additions:
        return "strict_recurring_summary_additions_missing", all_candidates, (), None, None
    contributions = (*bases, *additions)
    return (
        "supported",
        all_candidates,
        contributions,
        sum(row.numeric_value for row in contributions),
        unit,
    )


def _receipt(
    base: NumericEvidenceReconciliationReceipt,
    *,
    status: ReconciliationStatus | None = None,
    mode: ReconciliationMode | None = None,
    reason: str | None = None,
    numeric_result: float | None = None,
    unit: str | None = None,
    projection_rule: str = "sealed_typed_fields",
    candidates: Sequence[_ProjectedCandidate] = (),
    contributions: Sequence[ProjectedNumericContributionProof] = (),
    rejections: Sequence[RejectedNumericOperandProof] = (),
) -> NumericEvidenceReconciliationV2Receipt:
    delegated = projection_rule == "sealed_typed_fields"
    exact_contributions = tuple(contributions)
    return NumericEvidenceReconciliationV2Receipt(
        base,
        base.status if status is None else status,
        base.mode if mode is None else mode,
        base.reason if reason is None else reason,
        base.numeric_result if delegated else numeric_result,
        base.unit if delegated else unit,
        base.comparison_relation if delegated else None,
        base.boolean_result if delegated else None,
        projection_rule,
        _candidate_proofs(candidates),
        exact_contributions,
        tuple(rejections),
        (
            base.used_handle_ids
            if delegated
            else tuple(
                dict.fromkeys(
                    handle
                    for row in exact_contributions
                    for handle in row.handle_ids
                )
            )
        ),
        (
            base.deduplicated_item_count
            if delegated
            else sum(row.corroborated_duplicate_count for row in exact_contributions)
        ),
    )


def reconcile_sealed_numeric_evidence_v2(
    provider_input: Mapping[str, Any],
    *,
    sealed_provider_input_sha256: str,
) -> NumericEvidenceReconciliationV2Receipt:
    """Reconcile V1 fields plus strictly recognized actual-schema summaries."""

    base = reconcile_sealed_numeric_evidence(
        provider_input,
        sealed_provider_input_sha256=sealed_provider_input_sha256,
    )
    if base.status is not ReconciliationStatus.INSUFFICIENT:
        return _receipt(base)
    question = provider_input["dated_question"]
    typed = provider_input["typed_evidence"]
    operator = typed["operator_spec"]
    if not (
        operator.get("answer_shape") == "number"
        and operator.get("comparison_mode") == "none"
        and operator.get("operation") == "count_or_aggregate"
        and operator.get("required_slots") == []
    ):
        return _receipt(base)
    items = typed["items"]
    rejections = _h700_rejections(items)
    status, candidates, contributions, result, unit = _recurring_projection(
        question, items
    )
    if status == "not_applicable":
        status, candidates, contributions, result, unit = _distinct_projection(
            question, items
        )
        rule = "strict_distinct_summary"
        mode = ReconciliationMode.CARDINALITY_SUM
        success_reason = "strict_distinct_summary_cardinality_projection"
    else:
        rule = "strict_recurring_summary"
        mode = ReconciliationMode.RECURRING_PLUS_ADDITIONS
        success_reason = (
            "strict_recurring_summary_base_plus_deduplicated_additions"
        )
    if status == "not_applicable":
        return _receipt(base)
    if status != "supported":
        return _receipt(
            base,
            status=ReconciliationStatus.CONFLICTED,
            mode=mode,
            reason=status,
            projection_rule=rule,
            candidates=candidates,
            rejections=rejections,
        )
    _require(result is not None and unit is not None, "V2 projected result disappeared")
    return _receipt(
        base,
        status=ReconciliationStatus.SUPPORTED,
        mode=mode,
        reason=success_reason,
        numeric_result=result,
        unit=unit,
        projection_rule=rule,
        candidates=candidates,
        contributions=contributions,
        rejections=rejections,
    )


__all__ = [
    "CONTRIBUTION_FORMAT",
    "FORMAT",
    "NumericEvidenceReconciliationV2Receipt",
    "POLICY_ID",
    "PROJECTED_ITEM_FORMAT",
    "ProjectedNumericContributionProof",
    "ProjectedNumericItemProof",
    "reconcile_sealed_numeric_evidence_v2",
]
