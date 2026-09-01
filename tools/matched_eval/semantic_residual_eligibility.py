"""Gold-blind eligibility gate for the terminal semantic residual lane.

The semantic residual tree is intentionally the last retrieval mechanism, not
an always-on second copy of the primary retriever.  This module admits a
question only when sealed answer/route/sufficiency state exposes a concrete
unresolved condition.  A merely bounded upstream frontier is not sufficient:
it must coincide with an unresolved specialist-shaped need.

The gate consumes runtime projections only.  It does not accept benchmark
labels, references, judge verdicts, target ordinals, or target question IDs.
Every decision is replayable from its exact source projections.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)


MECHANISM_ID = "semantic_residual_gold_blind_eligibility_gate_v1"
POLICY_FORMAT = "memory-condense-semantic-residual-eligibility-policy-v1"
SIGNALS_FORMAT = "memory-condense-semantic-residual-eligibility-signals-v1"
DECISION_FORMAT = "memory-condense-semantic-residual-eligibility-decision-v1"

DEFAULT_RESIDUAL_PAYLOAD_TOKEN_CAP = 2_400
DEFAULT_HARD_COMPLETE_CHAT_TOKEN_CAP = 8_000
DEFAULT_OUTPUT_TOKEN_RESERVE = 768

_SPECIALIZED_STYLES = frozenset(
    {
        "compare",
        "comparison",
        "frequency",
        "latest_state",
        "numeric_reduce",
        "profile",
        "set_list",
        "state_chain",
        "synthesis",
        "synthesize",
        "temporal_timeline",
    }
)
_CLOSED_WORLD_AGGREGATION_STYLES = frozenset(
    {
        "profile",
        "set_join",
        "set_list",
        "state_chain",
        "synthesis",
        "synthesize",
    }
)
_ABSTENTION_RE = re.compile(
    r"(?:\b(?:i\s+do\s*n['’]?t\s+know|unknown|cannot\s+determine|"
    r"unable\s+to\s+determine|not\s+enough\s+(?:memory\s+)?(?:evidence|"
    r"information)|insufficient\s+(?:memory\s+)?(?:evidence|information))\b)",
    re.IGNORECASE,
)
_UNRESOLVED_WORDS = frozenset(
    {
        "abstain",
        "abstained",
        "conflicted",
        "fallback",
        "failed",
        "insufficient",
        "invalid",
        "no_resolution",
        "open",
        "unresolved",
    }
)
_WEAK_DECISIONS = frozenset(
    {
        "abstain",
        "fallback",
        "invalid_keep_parent",
        "keep_parent",
        "parent_passthrough",
        "unresolved",
    }
)


class SemanticResidualEligibilityError(MatchedEvalContractError):
    """A residual gate input, policy, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise SemanticResidualEligibilityError(message)


def _receipt(value: Mapping[str, object], declared: str, label: str) -> str:
    expected = identity_sha256(value)
    if declared:
        _require(require_sha256(declared, label) == expected, f"{label} changed")
    return expected


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


@dataclass(frozen=True, slots=True)
class SemanticResidualEligibilityPolicy:
    """Frozen gate and non-borrowable residual-lane budget."""

    residual_payload_token_cap: int = DEFAULT_RESIDUAL_PAYLOAD_TOKEN_CAP
    hard_complete_chat_token_cap: int = DEFAULT_HARD_COMPLETE_CHAT_TOKEN_CAP
    output_token_reserve: int = DEFAULT_OUTPUT_TOKEN_RESERVE
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.residual_payload_token_cap, "residual payload token cap"),
            (self.hard_complete_chat_token_cap, "hard chat token cap"),
            (self.output_token_reserve, "output token reserve"),
        ):
            _require(type(value) is int and value > 0, f"{label} changed")
        _require(
            self.residual_payload_token_cap + self.output_token_reserve
            < self.hard_complete_chat_token_cap,
            "residual lane consumed the complete prompt envelope",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic residual eligibility policy receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="semantic_residual_eligibility_policy")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "format": POLICY_FORMAT,
            "gold_loaded": False,
            "hard_complete_chat_token_cap": self.hard_complete_chat_token_cap,
            "mechanism_id": MECHANISM_ID,
            "new_provider_calls": 0,
            "non_borrowable_residual_budget": True,
            "output_token_reserve": self.output_token_reserve,
            "residual_payload_token_cap": self.residual_payload_token_cap,
            "retained_transformer_token_state_bytes": 0,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SemanticResidualEligibilitySignals:
    """Question-local facts derived only from sealed runtime state."""

    answer_abstained: bool
    answer_invalid: bool
    answer_weak: bool
    bounded_or_truncated_frontier: bool
    combined_decision_lane: str
    combined_resolution_applied: bool
    construction_mode: str
    decision: str
    frontier_receipt_sha256s: tuple[str, ...]
    operator_style: str
    prior_answer_bound: bool
    reconciliation_unresolved: bool
    route_has_applicable_specialist: bool
    route_specialized: bool
    specialist_route_gap: bool
    sufficiency_incomplete: bool
    unresolved_slot_ids: tuple[str, ...]
    used_handle_count: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.answer_abstained, "answer abstention"),
            (self.answer_invalid, "answer invalidity"),
            (self.answer_weak, "weak-answer flag"),
            (self.bounded_or_truncated_frontier, "frontier incompleteness"),
            (self.combined_resolution_applied, "combined resolution flag"),
            (self.reconciliation_unresolved, "reconciliation unresolved flag"),
            (self.prior_answer_bound, "prior answer binding flag"),
            (self.route_has_applicable_specialist, "applicable specialist flag"),
            (self.route_specialized, "specialized route flag"),
            (self.specialist_route_gap, "specialist route gap"),
            (self.sufficiency_incomplete, "sufficiency flag"),
        ):
            _require(type(value) is bool, f"{label} changed")
        _require(
            type(self.construction_mode) is str
            and type(self.combined_decision_lane) is str
            and type(self.decision) is str
            and type(self.operator_style) is str,
            "residual signal text changed",
        )
        _require(
            type(self.used_handle_count) is int and self.used_handle_count >= 0,
            "residual used-handle count changed",
        )
        _require(
            type(self.frontier_receipt_sha256s) is tuple
            and len(set(self.frontier_receipt_sha256s))
            == len(self.frontier_receipt_sha256s)
            and all(
                require_sha256(value, "frontier receipt") == value
                for value in self.frontier_receipt_sha256s
            ),
            "frontier receipt population changed",
        )
        _require(
            type(self.unresolved_slot_ids) is tuple
            and len(set(self.unresolved_slot_ids)) == len(self.unresolved_slot_ids)
            and all(type(value) is str and value for value in self.unresolved_slot_ids),
            "unresolved slot population changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic residual eligibility signal receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "answer_abstained": self.answer_abstained,
            "answer_invalid": self.answer_invalid,
            "answer_weak": self.answer_weak,
            "bounded_or_truncated_frontier": self.bounded_or_truncated_frontier,
            "combined_decision_lane": self.combined_decision_lane,
            "combined_resolution_applied": self.combined_resolution_applied,
            "construction_mode": self.construction_mode,
            "decision": self.decision,
            "format": SIGNALS_FORMAT,
            "frontier_receipt_sha256s": list(self.frontier_receipt_sha256s),
            "operator_style": self.operator_style,
            "prior_answer_bound": self.prior_answer_bound,
            "reconciliation_unresolved": self.reconciliation_unresolved,
            "route_has_applicable_specialist": self.route_has_applicable_specialist,
            "route_specialized": self.route_specialized,
            "specialist_route_gap": self.specialist_route_gap,
            "sufficiency_incomplete": self.sufficiency_incomplete,
            "unresolved_slot_ids": list(self.unresolved_slot_ids),
            "used_handle_count": self.used_handle_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


EligibilityReason = Literal[
    "answer_abstained",
    "answer_invalid",
    "reconciliation_unresolved",
    "specialist_route_gap",
    "specialized_frontier_unresolved",
    "specialized_sufficiency_unresolved",
    "aggregation_frontier_open",
    "synthesis_or_profile_unresolved",
]


@dataclass(frozen=True, slots=True)
class SemanticResidualEligibilityDecision:
    policy_receipt_sha256: str
    source_answer_row_sha256: str
    source_construction_row_sha256: str
    source_prior_answer_row_sha256: str | None
    source_reconciliation_row_sha256: str | None
    signals: SemanticResidualEligibilitySignals
    eligible: bool
    reasons: tuple[EligibilityReason, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.policy_receipt_sha256, "residual gate policy")
        require_sha256(self.source_answer_row_sha256, "residual gate answer row")
        require_sha256(
            self.source_construction_row_sha256, "residual gate construction row"
        )
        if self.source_reconciliation_row_sha256 is not None:
            require_sha256(
                self.source_reconciliation_row_sha256,
                "residual gate reconciliation row",
            )
        if self.source_prior_answer_row_sha256 is not None:
            require_sha256(
                self.source_prior_answer_row_sha256,
                "residual gate prior answer row",
            )
        _require(
            type(self.signals) is SemanticResidualEligibilitySignals,
            "residual gate signals changed type",
        )
        _require(
            type(self.eligible) is bool
            and type(self.reasons) is tuple
            and len(set(self.reasons)) == len(self.reasons)
            and self.eligible == bool(self.reasons),
            "residual gate decision/reason relation changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "semantic residual eligibility decision receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="semantic_residual_eligibility")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "eligible": self.eligible,
            "format": DECISION_FORMAT,
            "gold_loaded": False,
            "new_provider_calls": 0,
            "policy_receipt_sha256": self.policy_receipt_sha256,
            "reasons": list(self.reasons),
            "retained_transformer_token_state_bytes": 0,
            "signals": self.signals.projection(),
            "source_answer_row_sha256": self.source_answer_row_sha256,
            "source_construction_row_sha256": (
                self.source_construction_row_sha256
            ),
            "source_prior_answer_row_sha256": (
                self.source_prior_answer_row_sha256
            ),
            "source_reconciliation_row_sha256": (
                self.source_reconciliation_row_sha256
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _exact_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _route_style(construction_row: Mapping[str, Any]) -> str:
    route = _exact_mapping(construction_row.get("route"))
    legacy = _exact_mapping(route.get("legacy_route"))
    terminal = _exact_mapping(construction_row.get("terminal_prompt"))
    provider_input = _exact_mapping(terminal.get("provider_input"))
    typed = _exact_mapping(provider_input.get("typed_evidence"))
    operator = _exact_mapping(typed.get("operator_spec"))
    for value in (
        route.get("style"),
        legacy.get("style"),
        operator.get("style"),
        route.get("temporal_mode"),
    ):
        if type(value) is str and value.strip():
            return value.strip().casefold()
    return ""


def _applicable_specialists(construction_row: Mapping[str, Any]) -> tuple[str, ...]:
    route = _exact_mapping(construction_row.get("route"))
    candidates: list[str] = []
    for raw in (
        construction_row.get("applicable_specialist_ids"),
        route.get("applicable_specialist_ids"),
        route.get("applicable_mechanism_ids"),
    ):
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            candidates.extend(value for value in raw if type(value) is str and value)
    return _ordered_unique(candidates)


def _frontier_projections(
    construction_row: Mapping[str, Any],
    reconciliation_row: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    terminal = _exact_mapping(construction_row.get("terminal_prompt"))
    provider_input = _exact_mapping(terminal.get("provider_input"))
    typed = _exact_mapping(provider_input.get("typed_evidence"))
    frontier = _exact_mapping(typed.get("frontier"))
    if frontier:
        rows.append(frontier)
    methods = construction_row.get("methods")
    if isinstance(methods, Sequence) and not isinstance(
        methods, (str, bytes, bytearray)
    ):
        for raw_method in methods:
            method = _exact_mapping(raw_method)
            contributions = method.get("typed_contributions")
            if not isinstance(contributions, Sequence) or isinstance(
                contributions, (str, bytes, bytearray)
            ):
                continue
            rows.extend(
                contribution
                for raw in contributions
                if (contribution := _exact_mapping(raw))
            )
    if reconciliation_row is not None and _is_combined_reconciliation(
        reconciliation_row
    ):
        for key in (
            "frontier",
            "classified_frontier",
            "sufficiency",
            "sufficiency_gate",
        ):
            value = _exact_mapping(reconciliation_row.get(key))
            if value:
                rows.append(value)
    return tuple(rows)


def _frontier_state(
    projections: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool, tuple[str, ...], tuple[str, ...]]:
    incomplete = False
    sufficiency_incomplete = False
    unresolved: list[str] = []
    receipts: list[str] = []
    for row in projections:
        declared = row.get("receipt_sha256")
        receipts.append(
            declared
            if type(declared) is str and re.fullmatch(r"[0-9a-f]{64}", declared)
            else identity_sha256(dict(row))
        )
        mode = row.get("mode", row.get("frontier_mode"))
        truncated = row.get("truncated", row.get("selection_truncated"))
        closed = row.get("closed")
        sufficient = row.get("sufficient")
        raw_unresolved = row.get(
            "unresolved_slot_ids", row.get("missing_slot_ids", ())
        )
        if isinstance(raw_unresolved, Sequence) and not isinstance(
            raw_unresolved, (str, bytes, bytearray)
        ):
            unresolved.extend(
                value for value in raw_unresolved if type(value) is str and value
            )
        incomplete = incomplete or (
            truncated is True
            or closed is False
            or (type(mode) is str and mode.casefold() in {"bounded", "open"})
        )
        sufficiency_incomplete = sufficiency_incomplete or (
            sufficient is False or bool(raw_unresolved)
        )
    return (
        incomplete,
        sufficiency_incomplete,
        _ordered_unique(unresolved),
        tuple(sorted(set(receipts))),
    )


def _is_combined_reconciliation(row: Mapping[str, Any]) -> bool:
    """Reject arbitrary per-lane status as a global residual admission signal."""

    for key in ("decision_scope", "reconciliation_scope", "scope"):
        value = row.get(key)
        if type(value) is str and value.casefold() in {
            "applicable_specialist",
            "combined",
            "final",
            "question",
            "whole_question",
        }:
            return True
    if any(key in row for key in ("combined_decision", "final_decision")):
        return True
    value = row.get("format")
    if type(value) is str:
        normalized = value.casefold()
        return any(
            marker in normalized
            for marker in (
                "final-reconcile",
                "final-reconciliation",
                "reconcile-v3",
                "reconciliation-v3",
            )
        )
    return False


def _reconciliation_unresolved(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    if not _is_combined_reconciliation(row):
        return False
    for key in ("resolved", "sufficient", "solver_valid"):
        if row.get(key) is False:
            return True
    for key in (
        "decision",
        "disposition",
        "outcome",
        "resolution_status",
        "status",
    ):
        value = row.get(key)
        if type(value) is str and value.casefold() in _UNRESOLVED_WORDS:
            return True
    return False


def _combined_resolution_state(
    answer_row: Mapping[str, Any],
) -> tuple[str, bool, bool]:
    """Return lane, V3-schema flag, and whether a combined repair won.

    Missing V2 fields must not be interpreted as weak merely because V3 has a
    different schema.  A non-fallback V3 decision is a completed combined
    resolution and therefore terminal for this gate.
    """

    format_value = answer_row.get("format")
    lane_value = answer_row.get("decision_lane")
    source_value = answer_row.get("prediction_source")
    is_v3 = (
        type(format_value) is str
        and "locked-specialist-final-reconciliation-v3-result-row" in format_value
    ) or (
        type(source_value) is str and source_value.startswith("locked_v3_")
    )
    lane = lane_value.casefold() if type(lane_value) is str else ""
    applied = is_v3 and lane not in {"", "fallback", "none", "v2_fallback"}
    return lane, is_v3, applied


def evaluate_semantic_residual_eligibility(
    answer_row: Mapping[str, Any],
    construction_row: Mapping[str, Any],
    /,
    *,
    prior_answer_row: Mapping[str, Any] | None = None,
    reconciliation_row: Mapping[str, Any] | None = None,
    policy: SemanticResidualEligibilityPolicy = (
        SemanticResidualEligibilityPolicy()
    ),
) -> SemanticResidualEligibilityDecision:
    """Evaluate the terminal gate from sealed, gold-blind runtime rows."""

    _require(isinstance(answer_row, Mapping), "residual gate answer row changed type")
    _require(
        isinstance(construction_row, Mapping),
        "residual gate construction row changed type",
    )
    _require(
        reconciliation_row is None or isinstance(reconciliation_row, Mapping),
        "residual gate reconciliation row changed type",
    )
    _require(
        prior_answer_row is None or isinstance(prior_answer_row, Mapping),
        "residual gate prior answer row changed type",
    )
    _require(type(policy) is SemanticResidualEligibilityPolicy, "gate policy changed")
    assert_gold_blind(answer_row, path="semantic_residual_gate.answer_row")
    assert_gold_blind(
        construction_row, path="semantic_residual_gate.construction_row"
    )
    if reconciliation_row is not None:
        assert_gold_blind(
            reconciliation_row,
            path="semantic_residual_gate.reconciliation_row",
        )
    if prior_answer_row is not None:
        assert_gold_blind(
            prior_answer_row,
            path="semantic_residual_gate.prior_answer_row",
        )

    prediction = require_text(answer_row.get("prediction"), "residual answer prediction")
    construction_mode = str(construction_row.get("mode") or "").casefold()
    combined_lane, is_v3, combined_applied = _combined_resolution_state(answer_row)
    answer_basis = (
        prior_answer_row
        if is_v3 and not combined_applied and prior_answer_row is not None
        else answer_row
    )
    decision = str(answer_basis.get("decision") or "").casefold()
    used_handles = answer_basis.get("used_handle_ids", ())
    _require(
        isinstance(used_handles, Sequence)
        and not isinstance(used_handles, (str, bytes, bytearray)),
        "residual answer used handles changed type",
    )
    used_handle_count = len(tuple(used_handles))
    answer_invalid = not combined_applied and (
        answer_basis.get("solver_valid") is False
        or answer_basis.get("parse_error_code") not in {None, "", "none"}
        or decision in {"invalid", "invalid_keep_parent"}
    )
    answer_abstained = _ABSTENTION_RE.search(prediction) is not None
    if combined_applied:
        answer_weak = False
    elif is_v3 and prior_answer_row is None:
        # A V3 fallback preserves the prior answer but intentionally omits V2
        # parser/handle fields.  Do not turn that schema omission into a weak
        # numeric/temporal signal.  Route gaps and profile/synthesis demand are
        # evaluated explicitly below.
        answer_weak = construction_mode == "parent_passthrough"
    else:
        answer_weak = (
            decision in _WEAK_DECISIONS
            or construction_mode == "parent_passthrough"
            or (used_handle_count == 0 and decision != "replace")
        )
    operator_style = _route_style(construction_row)
    specialized = operator_style in _SPECIALIZED_STYLES
    applicable = _applicable_specialists(construction_row)
    frontier_rows = _frontier_projections(construction_row, reconciliation_row)
    (
        frontier_incomplete,
        sufficiency_incomplete,
        unresolved_slots,
        frontier_receipts,
    ) = _frontier_state(frontier_rows)
    route_gap = not combined_applied and specialized and (
        construction_mode == "parent_passthrough"
        or (bool(applicable) and not construction_row.get("methods"))
    )
    reconciliation_open = _reconciliation_unresolved(reconciliation_row)
    signals = SemanticResidualEligibilitySignals(
        answer_abstained=answer_abstained,
        answer_invalid=answer_invalid,
        answer_weak=answer_weak,
        bounded_or_truncated_frontier=frontier_incomplete,
        combined_decision_lane=combined_lane,
        combined_resolution_applied=combined_applied,
        construction_mode=construction_mode,
        decision=decision,
        frontier_receipt_sha256s=frontier_receipts,
        operator_style=operator_style,
        prior_answer_bound=prior_answer_row is not None,
        reconciliation_unresolved=reconciliation_open,
        route_has_applicable_specialist=bool(applicable),
        route_specialized=specialized,
        specialist_route_gap=route_gap,
        sufficiency_incomplete=sufficiency_incomplete,
        unresolved_slot_ids=unresolved_slots,
        used_handle_count=used_handle_count,
    )
    reasons: list[EligibilityReason] = []
    if answer_abstained and not combined_applied:
        reasons.append("answer_abstained")
    if answer_invalid:
        reasons.append("answer_invalid")
    if reconciliation_open and not combined_applied:
        reasons.append("reconciliation_unresolved")
    if route_gap:
        reasons.append("specialist_route_gap")
    if (
        specialized
        and not combined_applied
        and frontier_incomplete
        and (reconciliation_open or sufficiency_incomplete or unresolved_slots)
    ):
        reasons.append("specialized_frontier_unresolved")
    if (
        specialized
        and not combined_applied
        and (sufficiency_incomplete or unresolved_slots)
    ):
        reasons.append("specialized_sufficiency_unresolved")
    if (
        operator_style in _CLOSED_WORLD_AGGREGATION_STYLES
        and not combined_applied
        and frontier_incomplete
    ):
        # Closed-world list, profile, state-chain, and synthesis questions ask
        # for global coverage.  A bounded/open selection frontier therefore
        # leaves a real completeness obligation even when the current answer
        # is cited and confident.  Direct extraction deliberately does not
        # inherit this rule.
        reasons.append("aggregation_frontier_open")
    if (
        operator_style in _CLOSED_WORLD_AGGREGATION_STYLES
        and not combined_applied
        and answer_weak
        and not frontier_incomplete
    ):
        reasons.append("synthesis_or_profile_unresolved")
    frozen_reasons = tuple(dict.fromkeys(reasons))
    return SemanticResidualEligibilityDecision(
        policy_receipt_sha256=policy.receipt_sha256,
        source_answer_row_sha256=identity_sha256(dict(answer_row)),
        source_construction_row_sha256=identity_sha256(dict(construction_row)),
        source_prior_answer_row_sha256=(
            None
            if prior_answer_row is None
            else identity_sha256(dict(prior_answer_row))
        ),
        source_reconciliation_row_sha256=(
            None
            if reconciliation_row is None
            else identity_sha256(dict(reconciliation_row))
        ),
        signals=signals,
        eligible=bool(frozen_reasons),
        reasons=frozen_reasons,
    )


def replay_semantic_residual_eligibility(
    answer_row: Mapping[str, Any],
    construction_row: Mapping[str, Any],
    sealed: SemanticResidualEligibilityDecision,
    /,
    *,
    prior_answer_row: Mapping[str, Any] | None = None,
    reconciliation_row: Mapping[str, Any] | None = None,
    policy: SemanticResidualEligibilityPolicy = (
        SemanticResidualEligibilityPolicy()
    ),
) -> SemanticResidualEligibilityDecision:
    """Recompute a sealed gate decision and require byte-identical projection."""

    _require(
        type(sealed) is SemanticResidualEligibilityDecision,
        "sealed residual eligibility changed type",
    )
    replayed = evaluate_semantic_residual_eligibility(
        answer_row,
        construction_row,
        prior_answer_row=prior_answer_row,
        reconciliation_row=reconciliation_row,
        policy=policy,
    )
    _require(
        replayed.receipt_sha256 == sealed.receipt_sha256
        and replayed.projection() == sealed.projection(),
        "semantic residual eligibility replay changed",
    )
    return replayed


__all__ = [
    "DECISION_FORMAT",
    "DEFAULT_HARD_COMPLETE_CHAT_TOKEN_CAP",
    "DEFAULT_OUTPUT_TOKEN_RESERVE",
    "DEFAULT_RESIDUAL_PAYLOAD_TOKEN_CAP",
    "MECHANISM_ID",
    "POLICY_FORMAT",
    "SIGNALS_FORMAT",
    "SemanticResidualEligibilityDecision",
    "SemanticResidualEligibilityError",
    "SemanticResidualEligibilityPolicy",
    "SemanticResidualEligibilitySignals",
    "evaluate_semantic_residual_eligibility",
    "replay_semantic_residual_eligibility",
]
