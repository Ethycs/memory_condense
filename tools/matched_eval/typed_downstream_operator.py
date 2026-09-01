"""Versioned question-only overlays for downstream typed execution.

Legacy typed operator specifications are content-addressed inputs to locked
artifacts.  New question semantics therefore live in this additive overlay
instead of mutating those specifications or their receipts.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from memory_condense.domain.text_numbers import NUMBER_WORDS

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
)
from .typed_operator_adapter import TypedEvidencePacket
from .typed_operator_executor import (
    ExecutionStatus,
    ExecutorKind,
    OperatorExecutionReceipt,
    execute_typed_operator,
)
from .typed_operator_spec import AnswerShape, TypedOperatorSpec


FORMAT = "memory-condense-typed-downstream-operator-overlay-v1"
_CARDINAL_TOKEN = r"\d+|" + "|".join(NUMBER_WORDS)
_EXPLICIT_SET_CARDINALITY_RE = re.compile(
    r"\b(?:what|which)\s+are\s+(?:the\s+)?(?P<count>"
    + _CARDINAL_TOKEN
    + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DownstreamOperatorOverlay:
    """Additive constraints compiled after a legacy operator spec is sealed."""

    question_sha256: str
    legacy_operator_spec_receipt_sha256: str
    effective_set_cardinality: int | None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.question_sha256, "downstream-overlay question")
        require_sha256(
            self.legacy_operator_spec_receipt_sha256,
            "downstream-overlay legacy operator spec",
        )
        cardinality = self.effective_set_cardinality
        if cardinality is not None and (
            type(cardinality) is not int or cardinality < 1
        ):
            raise MatchedEvalContractError(
                "downstream-overlay set cardinality must be positive"
            )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("downstream-overlay receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="typed_downstream_operator_overlay")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "effective_set_cardinality": self.effective_set_cardinality,
            "format": FORMAT,
            "legacy_operator_spec_receipt_sha256": (
                self.legacy_operator_spec_receipt_sha256
            ),
            "question_sha256": self.question_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _question_set_cardinality(
    dated_question: str,
    operator_spec: TypedOperatorSpec,
) -> int | None:
    if operator_spec.cardinality is not None:
        return operator_spec.cardinality
    if operator_spec.answer_shape is not AnswerShape.SET_LIST:
        return None
    match = _EXPLICIT_SET_CARDINALITY_RE.search(dated_question)
    if match is None:
        return None
    raw = match.group("count").casefold()
    cardinality = int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
    return cardinality if cardinality >= 1 else None


def compile_downstream_operator_overlay(
    dated_question: str,
    operator_spec: TypedOperatorSpec,
) -> DownstreamOperatorOverlay:
    """Compile additive constraints without changing the legacy spec bytes."""

    if type(dated_question) is not str or not dated_question:
        raise TypeError("dated_question must be exact non-empty text")
    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be an exact TypedOperatorSpec")
    question_sha256 = hashlib.sha256(dated_question.encode("utf-8")).hexdigest()
    if question_sha256 != operator_spec.question_sha256:
        raise MatchedEvalContractError(
            "downstream-overlay question/spec binding changed"
        )
    return DownstreamOperatorOverlay(
        question_sha256=question_sha256,
        legacy_operator_spec_receipt_sha256=operator_spec.receipt_sha256,
        effective_set_cardinality=_question_set_cardinality(
            dated_question,
            operator_spec,
        ),
    )


def execute_downstream_typed_operator(
    operator_spec: TypedOperatorSpec,
    packet: TypedEvidencePacket,
    overlay: DownstreamOperatorOverlay,
) -> OperatorExecutionReceipt:
    """Execute legacy semantics, then fail closed on additive set cardinality."""

    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be an exact TypedOperatorSpec")
    if type(packet) is not TypedEvidencePacket:
        raise TypeError("packet must be an exact TypedEvidencePacket")
    if type(overlay) is not DownstreamOperatorOverlay:
        raise TypeError("overlay must be an exact DownstreamOperatorOverlay")
    if (
        overlay.question_sha256 != operator_spec.question_sha256
        or overlay.legacy_operator_spec_receipt_sha256
        != operator_spec.receipt_sha256
    ):
        raise MatchedEvalContractError(
            "downstream-overlay execution binding changed"
        )
    base = execute_typed_operator(operator_spec, packet)
    cardinality = overlay.effective_set_cardinality
    if (
        cardinality is None
        or operator_spec.answer_shape is not AnswerShape.SET_LIST
        or base.executor is not ExecutorKind.SET
        or base.status is not ExecutionStatus.SUPPORTED
        or len(base.used_item_receipt_sha256s) == cardinality
    ):
        return base
    return OperatorExecutionReceipt(
        operator_spec_receipt_sha256=base.operator_spec_receipt_sha256,
        evidence_packet_receipt_sha256=base.evidence_packet_receipt_sha256,
        closure_receipt_sha256=base.closure_receipt_sha256,
        consensus_receipt_sha256=base.consensus_receipt_sha256,
        executor=ExecutorKind.SET,
        status=ExecutionStatus.INSUFFICIENT,
        prediction="",
        numeric_result=None,
        used_item_receipt_sha256s=(),
        used_handle_ids=(),
        missing_slot_ids=base.missing_slot_ids,
        reason=(
            "downstream_set_cardinality_not_closed:"
            f"{overlay.receipt_sha256}"
        ),
    )


__all__ = [
    "DownstreamOperatorOverlay",
    "FORMAT",
    "compile_downstream_operator_overlay",
    "execute_downstream_typed_operator",
]
