"""Compact provider advisory over a final hot-v6 fact-ledger slice.

This is the packet-composer seam for :mod:`hot_v6_typed_reducer`.  It accepts
the already compiled ledger and the *final* provider-visible fact slice, runs
the deterministic reducer, and emits text only when that reducer has a
supported result.  Exact fact and backing IDs remain in the local audit
projection; provider text uses slice-local ``F`` ordinals only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from memory_condense.domain.discourse import quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .hot_v6_query_fact_ledger import FactLedgerSlice, QueryFactLedger
from .hot_v6_typed_reducer import (
    HotV6TypedReducerInput,
    HotV6TypedReduction,
    build_hot_v6_typed_reducer_input,
    reduce_hot_v6_typed_ledger,
)
from .operator_first_numeric_policy import RelevantNumericFrontier


FORMAT = "memory-condense-hot-v6-typed-reducer-advisory-v2"
PROVIDER_FORMAT = "memory-condense-typed-reduction-v2"


class HotV6TypedReducerAdvisoryError(MatchedEvalContractError):
    """The final slice, local citations, or provider boundary changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise HotV6TypedReducerAdvisoryError(message)


def _ordered_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    _require(
        all(type(value) is str and bool(value) for value in result),
        f"{label} must contain exact text",
    )
    _require(len(result) == len(set(result)), f"{label} repeat")
    return result


@dataclass(frozen=True, slots=True)
class LocalFactOrdinalBinding:
    """Prompt-local fact label bound to exact local-only provenance."""

    ordinal: int
    local_label: str
    fact_id: str
    fact_receipt_sha256: str
    backing_evidence_id: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.ordinal) is int and self.ordinal >= 1,
            "local fact ordinal changed",
        )
        _require(
            self.local_label == f"F{self.ordinal}"
            and bool(re.fullmatch(r"F[1-9]\d*", self.local_label)),
            "local fact label changed",
        )
        require_sha256(self.fact_id, "local fact ID")
        require_sha256(self.fact_receipt_sha256, "local fact receipt")
        require_text(self.backing_evidence_id, "local fact backing evidence")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "local fact binding changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "backing_evidence_id": self.backing_evidence_id,
            "fact_id": self.fact_id,
            "fact_receipt_sha256": self.fact_receipt_sha256,
            "local_label": self.local_label,
            "ordinal": self.ordinal,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class HotV6TypedReducerAdvisory:
    """Provider-safe text paired with its complete local audit projection."""

    question_sha256: str
    ledger_receipt_sha256: str
    fact_slice_receipt_sha256: str
    represented_backing_evidence_ids: tuple[str, ...]
    local_fact_bindings: tuple[LocalFactOrdinalBinding, ...]
    support_fact_labels: tuple[str, ...]
    reducer_input: HotV6TypedReducerInput
    reduction: HotV6TypedReduction
    provider_advisory_text: str
    provider_advisory_text_sha256: str
    provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.question_sha256, "typed advisory question")
        require_sha256(self.ledger_receipt_sha256, "typed advisory ledger")
        require_sha256(self.fact_slice_receipt_sha256, "typed advisory fact slice")
        represented = _ordered_unique(
            self.represented_backing_evidence_ids,
            "typed advisory represented backing evidence IDs",
        )
        _require(
            type(self.local_fact_bindings) is tuple
            and all(
                type(row) is LocalFactOrdinalBinding
                for row in self.local_fact_bindings
            ),
            "typed advisory local fact bindings changed",
        )
        expected_ordinals = tuple(range(1, len(self.local_fact_bindings) + 1))
        _require(
            tuple(row.ordinal for row in self.local_fact_bindings)
            == expected_ordinals
            and len({row.fact_id for row in self.local_fact_bindings})
            == len(self.local_fact_bindings),
            "typed advisory local fact order changed",
        )
        _require(
            all(
                row.backing_evidence_id in set(represented)
                for row in self.local_fact_bindings
            ),
            "typed advisory fact backing is not represented",
        )
        support = _ordered_unique(
            self.support_fact_labels,
            "typed advisory support fact labels",
        )
        available_labels = {row.local_label for row in self.local_fact_bindings}
        _require(
            set(support) <= available_labels,
            "typed advisory support escaped the selected fact slice",
        )
        _require(
            type(self.reducer_input) is HotV6TypedReducerInput
            and type(self.reduction) is HotV6TypedReduction,
            "typed advisory reducer objects changed",
        )
        _require(
            self.reducer_input.ledger_receipt_sha256
            == self.ledger_receipt_sha256
            and self.reducer_input.fact_population_receipt_sha256
            == self.fact_slice_receipt_sha256
            and self.reducer_input.represented_backing_evidence_ids == represented,
            "typed advisory reducer input escaped the final slice",
        )
        _require(
            self.reduction.input_receipt_sha256
            == self.reducer_input.receipt_sha256,
            "typed advisory reduction belongs to another input",
        )
        used_fact_ids = set(self.reduction.used_fact_ids)
        expected_support = tuple(
            row.local_label
            for row in self.local_fact_bindings
            if row.fact_id in used_fact_ids
        )
        _require(
            len(expected_support) == len(self.reduction.used_fact_ids)
            and support == expected_support,
            "typed advisory support disagrees with reducer proof",
        )
        _require(
            type(self.provider_advisory_text) is str,
            "typed advisory provider text changed",
        )
        require_sha256(
            self.provider_advisory_text_sha256,
            "typed advisory provider text",
        )
        _require(
            self.provider_advisory_text_sha256
            == quote_sha256(self.provider_advisory_text),
            "typed advisory provider text digest changed",
        )
        if self.reduction.status == "supported":
            _require(
                bool(self.provider_advisory_text) and bool(support),
                "supported typed reduction lost its provider advisory",
            )
        else:
            _require(
                self.provider_advisory_text == "" and not support,
                "unsupported typed reduction emitted provider text",
            )
        forbidden_ids = {
            self.ledger_receipt_sha256,
            self.fact_slice_receipt_sha256,
            self.reducer_input.receipt_sha256,
            self.reduction.receipt_sha256,
            *represented,
            *(row.fact_id for row in self.local_fact_bindings),
            *(row.fact_receipt_sha256 for row in self.local_fact_bindings),
        }
        _require(
            not any(value in self.provider_advisory_text for value in forbidden_ids),
            "typed reducer provider advisory exposed a local raw identifier",
        )
        _require(
            self.provider_calls == self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "typed advisory escaped the zero-call gold-free boundary",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "typed advisory changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_v6_typed_reducer_advisory")

    @property
    def emitted(self) -> bool:
        return bool(self.provider_advisory_text)

    @property
    def audit_projection(self) -> dict[str, Any]:
        """Return the complete gold-blind local audit record."""

        return self.projection()

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "fact_slice_receipt_sha256": self.fact_slice_receipt_sha256,
            "format": FORMAT,
            "gold_loaded": False,
            "ledger_receipt_sha256": self.ledger_receipt_sha256,
            "local_fact_bindings": [
                row.projection() for row in self.local_fact_bindings
            ],
            "provider_advisory": {
                "emitted": self.emitted,
                "support_fact_labels": list(self.support_fact_labels),
                "text": self.provider_advisory_text,
                "text_sha256": self.provider_advisory_text_sha256,
            },
            "provider_calls": 0,
            "question_sha256": self.question_sha256,
            "reducer_input": self.reducer_input.projection(),
            "reduction": self.reduction.projection(),
            "represented_backing_evidence_ids": list(
                self.represented_backing_evidence_ids
            ),
            "retained_transformer_token_state_bytes": 0,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _provider_text(
    reduction: HotV6TypedReduction,
    support_fact_labels: Sequence[str],
) -> str:
    if reduction.status != "supported":
        return ""
    labels = ",".join(support_fact_labels)
    return "\n".join(
        (
            f"<TYPED_REDUCTION_ADVISORY format={PROVIDER_FORMAT}>",
            "prediction="
            + json.dumps(reduction.prediction, ensure_ascii=False),
            f"support={labels}",
            "</TYPED_REDUCTION_ADVISORY>",
        )
    )


def compile_hot_v6_typed_reducer_advisory(
    dated_question: str,
    ledger: QueryFactLedger,
    fact_slice: FactLedgerSlice,
    *,
    represented_backing_evidence_ids: Sequence[str],
    relevant_frontier: RelevantNumericFrontier | None = None,
) -> HotV6TypedReducerAdvisory:
    """Run a final-slice reducer and compile a provider-safe advisory.

    No candidate or parent prediction is accepted at this pre-reader seam.
    Consequently, a bounded approximate temporal interval such as Q25 remains
    insufficient unless the underlying reducer can compute one exact value.
    Set/count reductions likewise remain insufficient unless the caller passes
    a separately constructed and closed relevant frontier.
    """

    require_text(dated_question, "typed advisory dated question")
    if type(ledger) is not QueryFactLedger:
        raise TypeError("ledger must be an exact QueryFactLedger")
    if type(fact_slice) is not FactLedgerSlice:
        raise TypeError("fact_slice must be an exact FactLedgerSlice")
    represented = _ordered_unique(
        represented_backing_evidence_ids,
        "typed advisory represented backing evidence IDs",
    )
    reducer_input = build_hot_v6_typed_reducer_input(
        dated_question,
        ledger,
        represented_backing_evidence_ids=represented,
        fact_slice=fact_slice,
    )
    reduction = reduce_hot_v6_typed_ledger(
        reducer_input,
        relevant_frontier=relevant_frontier,
    )
    local_fact_bindings = tuple(
        LocalFactOrdinalBinding(
            ordinal=ordinal,
            local_label=f"F{ordinal}",
            fact_id=fact.fact_id,
            fact_receipt_sha256=fact.receipt_sha256,
            backing_evidence_id=fact.backing_evidence_id,
        )
        for ordinal, fact in enumerate(fact_slice.facts, 1)
    )
    used_fact_ids = set(reduction.used_fact_ids)
    _require(
        used_fact_ids <= {row.fact_id for row in local_fact_bindings},
        "typed reduction used a fact outside the final slice",
    )
    support_fact_labels = tuple(
        row.local_label
        for row in local_fact_bindings
        if row.fact_id in used_fact_ids
    )
    provider_advisory_text = _provider_text(reduction, support_fact_labels)
    return HotV6TypedReducerAdvisory(
        question_sha256=ledger.question_sha256,
        ledger_receipt_sha256=ledger.receipt_sha256,
        fact_slice_receipt_sha256=fact_slice.receipt_sha256,
        represented_backing_evidence_ids=represented,
        local_fact_bindings=local_fact_bindings,
        support_fact_labels=support_fact_labels,
        reducer_input=reducer_input,
        reduction=reduction,
        provider_advisory_text=provider_advisory_text,
        provider_advisory_text_sha256=quote_sha256(provider_advisory_text),
    )


__all__ = [
    "FORMAT",
    "PROVIDER_FORMAT",
    "HotV6TypedReducerAdvisory",
    "HotV6TypedReducerAdvisoryError",
    "LocalFactOrdinalBinding",
    "compile_hot_v6_typed_reducer_advisory",
]
