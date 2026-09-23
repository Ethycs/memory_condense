"""Gold-blind typed reducers over the hot-v6 exact fact ledger.

The hot-v6 packet compiler already turns selected raw rows into
``QueryFactLedger`` facts.  This module is the deliberately small adapter that
lets the established temporal and numeric reducers consume those facts.  It
does not retrieve, select, or silently certify a candidate frontier.

Provider-visible evidence uses opaque handles.  The exact fact and backing
evidence IDs stay in a local receipt so a deterministic answer can be traced
back to raw G or E evidence.  A caller must pass the complete set of rendered
backing evidence IDs; an absent backing row fails closed before any reducer is
run.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain.discourse import quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .hot_v6_query_fact_ledger import (
    FactLedgerSlice,
    FactStatus,
    QueryFactLedger,
    TypedExactQuoteFact,
)
from .numeric_evidence_reconciler import ReconciliationStatus
from .numeric_evidence_reconciler import NumericEvidenceReconcilerError
from .numeric_evidence_reconciler_v2 import (
    reconcile_sealed_numeric_evidence_v2,
)
from .operator_first_numeric_policy import (
    RelevantNumericFrontier,
    build_relevant_numeric_frontier,
    execute_operator_first_numeric_policy,
)
from .temporal_event_reconciler import (
    TemporalEventReconcilerError,
    reconcile_temporal_events,
)
from .typed_memory_final_arm import PROMPT_ROW_FORMAT
from .typed_operator_adapter import COMPACT_FINAL_PROVIDER_FORMAT
from .typed_operator_executor import ExecutionStatus
from .typed_operator_spec import (
    AnswerShape,
    ComparisonMode,
    TemporalMode,
    TypedOperatorSpec,
    normalized_terms,
)


FORMAT = "memory-condense-hot-v6-typed-reducer-input-v2"
REDUCTION_FORMAT = "memory-condense-hot-v6-typed-reduction-v2"
_NO_CANDIDATE = "No candidate prediction was supplied."
_NO_PARENT = "No protected parent prediction was supplied."

_AGO_RE = re.compile(
    r"\b(?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|\d+(?:\.\d+)?)\s+"
    r"(?P<unit>days?|weeks?|months?|years?)\s+ago\b",
    re.IGNORECASE,
)

_OPERATOR_FIELDS = (
    "absence_decision_requires_closed_frontier",
    "answer_shape",
    "cardinality",
    "comparison_mode",
    "include_proposed",
    "operation",
    "ordering",
    "personalization_required",
    "query_timestamp",
    "required_evidence_role",
    "required_slots",
    "requires_all_slots",
    "requires_complete_frontier",
    "specificity_required",
    "style",
    "temporal_mode",
    "temporal_window_days",
)


class HotV6TypedReducerError(MatchedEvalContractError):
    """The exact ledger/reducer provenance boundary changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise HotV6TypedReducerError(message)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _ordered_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    output = tuple(values)
    _require(
        all(type(value) is str and bool(value) for value in output),
        f"{label} must contain exact text",
    )
    _require(len(output) == len(set(output)), f"{label} repeat")
    return output


def _source_day(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().replace("Z", "+00:00")
    for candidate in (cleaned, cleaned.replace("/", "-")):
        try:
            return datetime.fromisoformat(candidate).date().isoformat()
        except ValueError:
            pass
    for pattern in (
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(cleaned, pattern).date().isoformat()
        except ValueError:
            pass
    return None


def _temporal_summary(fact: TypedExactQuoteFact) -> str:
    """Retain the quote and expose an ``ago`` relation to the V3 parser.

    The temporal reconciler understands ``N weeks before YYYY-MM-DD``.  Raw
    conversational evidence normally says ``N weeks ago`` and carries its
    reference clock in source metadata.  This deterministic append preserves
    the quote verbatim while making that already-authenticated clock explicit.
    """

    source_day = _source_day(fact.source_created_at)
    match = _AGO_RE.search(fact.exact_quote)
    if source_day is None or match is None:
        return fact.exact_quote
    relation = f"{match.group('value')} {match.group('unit')} before {source_day}"
    return f"{fact.exact_quote} [normalized temporal relation: {relation}]"


def _status(value: FactStatus) -> str:
    return {
        FactStatus.ASSERTED: "unknown",
        FactStatus.MENTIONED: "unknown",
        FactStatus.COMPLETED: "completed",
        FactStatus.PLANNED: "proposed",
        FactStatus.FAILED: "cancelled",
        FactStatus.NEGATED: "cancelled",
    }[value]


def _operator_projection(spec: TypedOperatorSpec) -> dict[str, Any]:
    projection = spec.projection()
    result = {field: projection[field] for field in _OPERATOR_FIELDS}
    # The established numeric reconciler predates RequiredSlot's version tag
    # and accepts the exact seven semantic fields below.  Strip only that
    # transport tag at this compatibility boundary; retain every operand,
    # relation, threshold, and stable slot ID unchanged.
    result["required_slots"] = [
        {key: value for key, value in slot.items() if key != "format"}
        for slot in projection["required_slots"]
    ]
    return result


def _relation(fact: TypedExactQuoteFact) -> str:
    fields = [f"authored_by_{fact.source_role.casefold()}"]
    if fact.action_concepts:
        fields.append(f"event_action={'|'.join(fact.action_concepts)}")
    if fact.time_basis == "source_created_at_fallback":
        fields.append("date_basis=source_created_at")
    elif fact.time_basis == "explicit_event_time_in_quote":
        fields.append("date_basis=relative_event_time")
    return ";".join(fields)


@dataclass(frozen=True, slots=True)
class HotV6FactHandleBinding:
    handle_id: str
    group_handle: str
    backing_evidence_id: str
    backing_source_sha256: str
    fact_ids: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(bool(re.fullmatch(r"H\d{3,6}", self.handle_id)), "handle changed")
        _require(bool(re.fullmatch(r"G\d{3,6}", self.group_handle)), "group changed")
        require_text(self.backing_evidence_id, "typed reducer backing evidence")
        require_sha256(self.backing_source_sha256, "typed reducer backing source")
        _ordered_unique(self.fact_ids, "typed reducer binding facts")
        for fact_id in self.fact_ids:
            require_sha256(fact_id, "typed reducer fact")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "fact handle binding changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "backing_evidence_id": self.backing_evidence_id,
            "backing_source_sha256": self.backing_source_sha256,
            "fact_ids": list(self.fact_ids),
            "group_handle": self.group_handle,
            "handle_id": self.handle_id,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class HotV6ItemFactBinding:
    item_sha256: str
    fact_id: str
    backing_evidence_id: str
    operand_id: str | None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.item_sha256, "typed reducer provider item")
        require_sha256(self.fact_id, "typed reducer item fact")
        require_text(self.backing_evidence_id, "typed reducer item backing")
        if self.operand_id is not None:
            require_sha256(self.operand_id, "typed reducer item operand")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "item/fact binding changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "backing_evidence_id": self.backing_evidence_id,
            "fact_id": self.fact_id,
            "item_sha256": self.item_sha256,
            "operand_id": self.operand_id,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class HotV6TypedReducerInput:
    ledger_receipt_sha256: str
    fact_population_receipt_sha256: str
    represented_backing_evidence_ids: tuple[str, ...]
    represented_population_sha256: str
    provider_input_json: str
    provider_input_sha256: str
    handle_bindings: tuple[HotV6FactHandleBinding, ...]
    item_bindings: tuple[HotV6ItemFactBinding, ...]
    provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.ledger_receipt_sha256, "typed reducer ledger")
        require_sha256(
            self.fact_population_receipt_sha256,
            "typed reducer selected fact population",
        )
        represented = _ordered_unique(
            self.represented_backing_evidence_ids,
            "typed reducer represented backing evidence IDs",
        )
        require_sha256(self.represented_population_sha256, "represented population")
        _require(
            self.represented_population_sha256 == identity_sha256(list(represented)),
            "represented backing population changed",
        )
        require_text(self.provider_input_json, "typed reducer provider input")
        require_sha256(self.provider_input_sha256, "typed reducer provider input")
        _require(
            self.provider_input_json == _canonical(self.provider_input),
            "typed reducer provider input encoding changed",
        )
        _require(
            self.provider_input_sha256 == identity_sha256(self.provider_input),
            "typed reducer provider input digest changed",
        )
        _require(
            type(self.handle_bindings) is tuple
            and all(type(row) is HotV6FactHandleBinding for row in self.handle_bindings)
            and len({row.handle_id for row in self.handle_bindings})
            == len(self.handle_bindings),
            "typed reducer handle bindings changed",
        )
        _require(
            type(self.item_bindings) is tuple
            and all(type(row) is HotV6ItemFactBinding for row in self.item_bindings)
            and len({row.item_sha256 for row in self.item_bindings})
            == len(self.item_bindings),
            "typed reducer item bindings changed",
        )
        represented_set = set(represented)
        _require(
            all(
                row.backing_evidence_id in represented_set
                for row in (*self.handle_bindings, *self.item_bindings)
            ),
            "typed reducer evidence escaped rendered backing rows",
        )
        _require(
            self.provider_calls == self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "typed reducer input escaped zero-call gold-free boundary",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "typed reducer input changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_v6_typed_reducer_input")

    @property
    def provider_input(self) -> dict[str, Any]:
        value = json.loads(self.provider_input_json)
        _require(type(value) is dict, "typed reducer provider input type changed")
        return dict(value)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": FORMAT,
            "gold_loaded": False,
            "handle_bindings": [row.projection() for row in self.handle_bindings],
            "item_bindings": [row.projection() for row in self.item_bindings],
            "fact_population_receipt_sha256": (
                self.fact_population_receipt_sha256
            ),
            "ledger_receipt_sha256": self.ledger_receipt_sha256,
            "provider_calls": 0,
            "provider_input_sha256": self.provider_input_sha256,
            "represented_backing_evidence_ids": list(
                self.represented_backing_evidence_ids
            ),
            "represented_population_sha256": self.represented_population_sha256,
            "retained_transformer_token_state_bytes": 0,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class HotV6TypedReduction:
    input_receipt_sha256: str
    mechanism: str
    status: Literal[
        "supported", "insufficient", "conflicted", "not_applicable"
    ]
    prediction: str
    reason: str
    used_fact_ids: tuple[str, ...]
    used_backing_evidence_ids: tuple[str, ...]
    proof_json: str
    proof_sha256: str
    provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.input_receipt_sha256, "typed reduction input")
        require_text(self.mechanism, "typed reduction mechanism")
        _require(
            self.status
            in {"supported", "insufficient", "conflicted", "not_applicable"},
            "typed reduction status changed",
        )
        _require(
            bool(self.prediction) == (self.status == "supported"),
            "typed reduction prediction/status changed",
        )
        require_text(self.reason, "typed reduction reason")
        _ordered_unique(self.used_fact_ids, "typed reduction used facts")
        _ordered_unique(
            self.used_backing_evidence_ids, "typed reduction used backing evidence"
        )
        for fact_id in self.used_fact_ids:
            require_sha256(fact_id, "typed reduction used fact")
        require_text(self.proof_json, "typed reduction proof")
        require_sha256(self.proof_sha256, "typed reduction proof")
        _require(self.proof_json == _canonical(self.proof), "typed proof encoding changed")
        _require(identity_sha256(self.proof) == self.proof_sha256, "typed proof changed")
        _require(
            self.provider_calls == self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "typed reduction escaped zero-call gold-free boundary",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "typed reduction changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_v6_typed_reduction")

    @property
    def proof(self) -> dict[str, Any]:
        value = json.loads(self.proof_json)
        _require(type(value) is dict, "typed reduction proof type changed")
        return dict(value)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": REDUCTION_FORMAT,
            "gold_loaded": False,
            "input_receipt_sha256": self.input_receipt_sha256,
            "mechanism": self.mechanism,
            "prediction": self.prediction,
            "proof": self.proof,
            "proof_sha256": self.proof_sha256,
            "provider_calls": 0,
            "reason": self.reason,
            "retained_transformer_token_state_bytes": 0,
            "status": self.status,
            "used_backing_evidence_ids": list(self.used_backing_evidence_ids),
            "used_fact_ids": list(self.used_fact_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _provider_item(
    fact: TypedExactQuoteFact,
    *,
    handle_id: str,
    numeric_index: int | None,
) -> tuple[dict[str, Any], str | None]:
    item: dict[str, Any] = {
        "content_coherence": "match",
        "handle_ids": [handle_id],
        "included": True,
        "kind": "operand" if numeric_index is not None else "event",
        "status": _status(fact.status),
        "summary": _temporal_summary(fact),
        "supported_slot_ids": list(fact.bound_slot_ids),
        "value_authority": "explicit",
    }
    source_day = _source_day(fact.source_created_at)
    if source_day is not None:
        item["date"] = source_day
    relation = _relation(fact)
    if relation:
        item["relation"] = relation
    operand_id: str | None = None
    if numeric_index is not None:
        operand = fact.numeric_operands[numeric_index]
        operand_id = operand.operand_id
        item.update(
            {
                "numeric_qualifier": operand.qualifier,
                "numeric_role": "operand",
                "numeric_value": float(operand.value),
            }
        )
        if operand.unit is not None:
            item["unit"] = operand.unit
    return item, operand_id


def build_hot_v6_typed_reducer_input(
    dated_question: str,
    ledger: QueryFactLedger,
    *,
    represented_backing_evidence_ids: Sequence[str],
    fact_slice: FactLedgerSlice | None = None,
) -> HotV6TypedReducerInput:
    """Adapt provider-visible exact G/E facts without upgrading their frontier.

    Pass the final ``FactLedgerSlice`` when packing selected only part of a
    compiled ledger.  Omitted ledger candidates are then intentionally outside
    the reducer.  Their unrendered backing rows neither fail the adapter nor
    become hidden reducer inputs.
    """

    require_text(dated_question, "typed reducer dated question")
    if type(ledger) is not QueryFactLedger:
        raise TypeError("ledger must be an exact QueryFactLedger")
    question_sha = hashlib.sha256(dated_question.encode("utf-8")).hexdigest()
    _require(question_sha == ledger.question_sha256, "ledger belongs to another question")
    facts = ledger.facts
    fact_population_receipt_sha256 = ledger.receipt_sha256
    if fact_slice is not None:
        if type(fact_slice) is not FactLedgerSlice:
            raise TypeError("fact_slice must be an exact FactLedgerSlice")
        _require(
            fact_slice.question_sha256 == ledger.question_sha256
            and fact_slice.ledger_receipt_sha256 == ledger.receipt_sha256,
            "fact slice belongs to another ledger",
        )
        ledger_fact_ids = tuple(fact.fact_id for fact in ledger.facts)
        _require(
            fact_slice.ledger_fact_ids == ledger_fact_ids,
            "fact slice ledger population changed",
        )
        ledger_by_id = {fact.fact_id: fact for fact in ledger.facts}
        _require(
            all(
                fact.fact_id in ledger_by_id
                and fact == ledger_by_id[fact.fact_id]
                for fact in fact_slice.facts
            ),
            "fact slice escaped its sealed ledger",
        )
        facts = fact_slice.facts
        fact_population_receipt_sha256 = fact_slice.receipt_sha256
    represented = _ordered_unique(
        represented_backing_evidence_ids,
        "typed reducer represented backing evidence IDs",
    )
    cited_backing_evidence_ids = {
        fact.backing_evidence_id for fact in facts
    }
    missing = cited_backing_evidence_ids - set(represented)
    _require(
        not missing,
        "typed reducer ledger fact backing evidence is not rendered: "
        + ", ".join(sorted(missing)),
    )

    evidence_order = tuple(
        dict.fromkeys(fact.backing_evidence_id for fact in facts)
    )
    source_order = tuple(dict.fromkeys(fact.backing_source_id for fact in facts))
    source_group = {
        source_id: f"G{index:03d}"
        for index, source_id in enumerate(source_order, 1)
    }
    handle_by_evidence = {
        evidence_id: f"H{index:03d}"
        for index, evidence_id in enumerate(evidence_order, 1)
    }
    facts_by_evidence: dict[str, list[TypedExactQuoteFact]] = {}
    for fact in facts:
        facts_by_evidence.setdefault(fact.backing_evidence_id, []).append(fact)
    handle_bindings = tuple(
        HotV6FactHandleBinding(
            handle_id=handle_by_evidence[evidence_id],
            group_handle=source_group[facts_by_evidence[evidence_id][0].backing_source_id],
            backing_evidence_id=evidence_id,
            backing_source_sha256=quote_sha256(
                facts_by_evidence[evidence_id][0].backing_source_id
            ),
            fact_ids=tuple(fact.fact_id for fact in facts_by_evidence[evidence_id]),
        )
        for evidence_id in evidence_order
    )
    handles = [
        {
            "group_handle": row.group_handle,
            "handle_id": row.handle_id,
            "origin": "direct_pointer",
            "provenance_grade": "direct_pointer",
        }
        for row in handle_bindings
    ]
    items: list[dict[str, Any]] = []
    item_bindings: list[HotV6ItemFactBinding] = []
    for fact in facts:
        indices: tuple[int | None, ...] = (
            tuple(range(len(fact.numeric_operands)))
            if fact.numeric_operands
            else (None,)
        )
        for numeric_index in indices:
            item, operand_id = _provider_item(
                fact,
                handle_id=handle_by_evidence[fact.backing_evidence_id],
                numeric_index=numeric_index,
            )
            item_sha = identity_sha256(item)
            # Two clauses with byte-identical typed projections and the same
            # exact backing row are one provider item, but both local facts
            # would make the item-to-fact mapping ambiguous.  Retain the first
            # projection only; exact backing authority is unchanged.
            if any(row.item_sha256 == item_sha for row in item_bindings):
                continue
            items.append(item)
            item_bindings.append(
                HotV6ItemFactBinding(
                    item_sha256=item_sha,
                    fact_id=fact.fact_id,
                    backing_evidence_id=fact.backing_evidence_id,
                    operand_id=operand_id,
                )
            )

    operator = _operator_projection(ledger.operator_spec)
    handle_ids = [row["handle_id"] for row in handles]
    typed = {
        "conflict_policy": "quarantine",
        "format": COMPACT_FINAL_PROVIDER_FORMAT,
        "frontier": {
            "available_handle_ids": handle_ids,
            "closed": False,
            "mode": "bounded",
            "omitted_handle_ids": [],
            "rejected_item_count": 0,
            "represented_handle_ids": handle_ids,
            "truncated": True,
            "unresolved_slot_ids": list(ledger.unresolved_slot_ids),
        },
        "handles": handles,
        "items": items,
        "operator_spec": operator,
    }
    provider_input = {
        "dated_question": dated_question,
        "format": PROMPT_ROW_FORMAT,
        "typed_evidence": typed,
    }
    assert_gold_blind(provider_input, path="hot_v6_typed_reducer_provider_input")
    return HotV6TypedReducerInput(
        ledger_receipt_sha256=ledger.receipt_sha256,
        fact_population_receipt_sha256=fact_population_receipt_sha256,
        represented_backing_evidence_ids=represented,
        represented_population_sha256=identity_sha256(list(represented)),
        provider_input_json=_canonical(provider_input),
        provider_input_sha256=identity_sha256(provider_input),
        handle_bindings=handle_bindings,
        item_bindings=tuple(item_bindings),
    )


def build_hot_v6_relevant_frontier(
    reducer_input: HotV6TypedReducerInput,
    *,
    candidate_population_receipt_sha256: str,
    unresolved_candidate_keys: Sequence[str] = (),
    selection_truncated: bool = False,
) -> RelevantNumericFrontier:
    """Bind an independently exhaustive section/full-store candidate census."""

    if type(reducer_input) is not HotV6TypedReducerInput:
        raise TypeError("reducer_input must be exact")
    return build_relevant_numeric_frontier(
        reducer_input.provider_input,
        candidate_population_receipt_sha256=candidate_population_receipt_sha256,
        unresolved_candidate_keys=_ordered_unique(
            unresolved_candidate_keys, "typed reducer unresolved candidate keys"
        ),
        selection_truncated=selection_truncated,
    )


def _used_bindings(
    reducer_input: HotV6TypedReducerInput,
    handle_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selected = set(handle_ids)
    facts = tuple(
        dict.fromkeys(
            fact_id
            for row in reducer_input.handle_bindings
            if row.handle_id in selected
            for fact_id in row.fact_ids
        )
    )
    backing = tuple(
        dict.fromkeys(
            row.backing_evidence_id
            for row in reducer_input.handle_bindings
            if row.handle_id in selected
        )
    )
    _require(
        set(backing) <= set(reducer_input.represented_backing_evidence_ids),
        "typed reduction used an unrendered backing row",
    )
    return facts, backing


def _reduction(
    reducer_input: HotV6TypedReducerInput,
    *,
    mechanism: str,
    status: str,
    prediction: str,
    reason: str,
    handle_ids: Sequence[str],
    proof: Mapping[str, Any],
) -> HotV6TypedReduction:
    facts, backing = _used_bindings(reducer_input, handle_ids)
    body = dict(proof)
    assert_gold_blind(body, path="hot_v6_typed_reducer_proof")
    return HotV6TypedReduction(
        input_receipt_sha256=reducer_input.receipt_sha256,
        mechanism=mechanism,
        status=status,  # type: ignore[arg-type]
        prediction=prediction,
        reason=reason,
        used_fact_ids=facts,
        used_backing_evidence_ids=backing,
        proof_json=_canonical(body),
        proof_sha256=identity_sha256(body),
    )


def _numeric_prediction(value: float, unit: str | None) -> str:
    scalar = str(int(value)) if float(value).is_integer() else f"{value:g}"
    if unit in {"$", "USD"}:
        return "$" + scalar
    if unit and "/" in unit:
        noun, period = unit.split("/", 1)
        noun = noun.replace("_", " ")
        if value != 1 and not noun.endswith("s"):
            noun += "s"
        return f"{scalar} {noun} per {period}"
    if unit:
        return f"{scalar} {unit}"
    return scalar


def _contract_non_applicability_proof(
    *,
    stage: str,
    error: MatchedEvalContractError | TemporalEventReconcilerError,
    provider_input_sha256: str,
) -> dict[str, Any]:
    """Seal one recognized reducer-domain refusal without hiding other errors."""

    body = {
        "contract_error_message_sha256": quote_sha256(str(error)),
        "contract_error_type": type(error).__name__,
        "format": f"{REDUCTION_FORMAT}-contract-non-applicability-v1",
        "provider_calls": 0,
        "provider_input_sha256": provider_input_sha256,
        "retained_transformer_token_state_bytes": 0,
        "stage": stage,
        "status": "insufficient",
    }
    assert_gold_blind(body, path="hot_v6_typed_reducer_contract_refusal")
    return body


def reduce_hot_v6_typed_ledger(
    reducer_input: HotV6TypedReducerInput,
    *,
    relevant_frontier: RelevantNumericFrontier | None = None,
    candidate_prediction: str | None = None,
    parent_prediction: str | None = None,
) -> HotV6TypedReduction:
    """Run the applicable proven reducer, otherwise return an explicit abstention."""

    if type(reducer_input) is not HotV6TypedReducerInput:
        raise TypeError("reducer_input must be exact")
    provider_input = reducer_input.provider_input
    operator = provider_input["typed_evidence"]["operator_spec"]
    temporal_mode = TemporalMode(operator["temporal_mode"])
    if temporal_mode is not TemporalMode.NONE:
        inventory = provider_input["typed_evidence"]["handles"]
        items = provider_input["typed_evidence"]["items"]
        item_bindings = {row.item_sha256: row for row in reducer_input.item_bindings}
        by_handle: dict[str, dict[str, Any]] = {}
        for handle in inventory:
            handle_id = handle["handle_id"]
            matching = [item for item in items if handle_id in item["handle_ids"]]
            fact_terms = tuple(
                dict.fromkeys(
                    term
                    for item in matching
                    for term in normalized_terms(str(item["summary"]))
                )
            )
            by_handle[handle_id] = {
                "answer_anchor_terms": list(fact_terms),
                "usable_item_receipt_sha256s": [
                    identity_sha256(item) for item in matching
                ],
            }
        try:
            result = reconcile_temporal_events(
                dated_question=provider_input["dated_question"],
                candidate_prediction=candidate_prediction or _NO_CANDIDATE,
                parent_prediction=parent_prediction or _NO_PARENT,
                provider_input=provider_input,
                validation_contract={"by_handle": by_handle},
                allowed_handle_ids=[row["handle_id"] for row in inventory],
                source_receipt_sha256=reducer_input.receipt_sha256,
            )
        except TemporalEventReconcilerError as error:
            return _reduction(
                reducer_input,
                mechanism="temporal_event_reconciler_v1",
                status="insufficient",
                prediction="",
                reason="temporal_reconciler_contract_non_applicable",
                handle_ids=(),
                proof=_contract_non_applicability_proof(
                    stage="temporal_event_reconciler_v1",
                    error=error,
                    provider_input_sha256=reducer_input.provider_input_sha256,
                ),
            )
        if result is not None:
            # Assert that every proof item still has a local exact-fact edge.
            for evidence in result.proof["evidence"]:
                _require(
                    evidence["provider_item_sha256"] in item_bindings,
                    "temporal proof item lost its exact fact binding",
                )
            return _reduction(
                reducer_input,
                mechanism="temporal_event_reconciler_v1",
                status="supported",
                prediction=result.prediction,
                reason=result.operation,
                handle_ids=result.proof_handle_ids,
                proof=result.projection(),
            )
        return _reduction(
            reducer_input,
            mechanism="temporal_event_reconciler_v1",
            status="insufficient",
            prediction="",
            reason="question_bound_temporal_evidence_not_resolved",
            handle_ids=(),
            proof={
                "candidate_prediction_sha256": quote_sha256(
                    candidate_prediction or _NO_CANDIDATE
                ),
                "parent_prediction_sha256": quote_sha256(
                    parent_prediction or _NO_PARENT
                ),
                "provider_input_sha256": reducer_input.provider_input_sha256,
            },
        )

    answer_shape = AnswerShape(operator["answer_shape"])
    comparison = ComparisonMode(operator["comparison_mode"])
    if (
        answer_shape not in {AnswerShape.NUMBER, AnswerShape.BOOLEAN}
        and comparison is ComparisonMode.NONE
        and operator["operation"] != "count_or_aggregate"
    ):
        return _reduction(
            reducer_input,
            mechanism="none",
            status="not_applicable",
            prediction="",
            reason="question_has_no_deterministic_temporal_or_numeric_operation",
            handle_ids=(),
            proof={"provider_input_sha256": reducer_input.provider_input_sha256},
        )

    # The operator-first policy is the proven path for fixed named scalar sides
    # and for set/count questions with a separately certified candidate census.
    operator_first = None
    operator_first_contract_refusal: dict[str, Any] | None = None
    try:
        operator_first = execute_operator_first_numeric_policy(
            provider_input,
            relevant_frontier=relevant_frontier,
        )
    except MatchedEvalContractError as error:
        if relevant_frontier is not None:
            # A supplied frontier is a separate authority object.  Its binding
            # failures remain fatal rather than being relabeled as domain
            # non-applicability.
            raise
        operator_first_contract_refusal = _contract_non_applicability_proof(
            stage="operator_first_numeric_policy_v1",
            error=error,
            provider_input_sha256=reducer_input.provider_input_sha256,
        )
    if (
        operator_first is not None
        and operator_first.status is ExecutionStatus.SUPPORTED
    ):
        return _reduction(
            reducer_input,
            mechanism="operator_first_numeric_policy_v1",
            status="supported",
            prediction=operator_first.prediction,
            reason=operator_first.reason,
            handle_ids=operator_first.used_handle_ids,
            proof=operator_first.projection(),
        )

    # V2 retains the earlier direct/recurring reducers.  It is intentionally a
    # fallback: set closure still belongs to the independently certified
    # operator-first frontier above.
    try:
        reconciled = reconcile_sealed_numeric_evidence_v2(
            provider_input,
            sealed_provider_input_sha256=reducer_input.provider_input_sha256,
        )
    except NumericEvidenceReconcilerError as error:
        reconciler_contract_refusal = _contract_non_applicability_proof(
            stage="numeric_evidence_reconciler_v2",
            error=error,
            provider_input_sha256=reducer_input.provider_input_sha256,
        )
        operator_status = (
            operator_first.status
            if operator_first is not None
            else ExecutionStatus.INSUFFICIENT
        )
        return _reduction(
            reducer_input,
            mechanism=(
                "operator_first_numeric_policy_v1+"
                "numeric_evidence_reconciler_v2"
            ),
            status=(
                "conflicted"
                if operator_status is ExecutionStatus.CONFLICTED
                else "insufficient"
            ),
            prediction="",
            reason=(
                f"{operator_first.reason};numeric_reconciler_contract_non_applicable"
                if operator_first is not None
                else "operator_first_contract_non_applicable;"
                "numeric_reconciler_contract_non_applicable"
            ),
            handle_ids=(),
            proof={
                "numeric_reconciliation": reconciler_contract_refusal,
                "operator_first": (
                    operator_first.projection()
                    if operator_first is not None
                    else operator_first_contract_refusal
                ),
            },
        )
    if reconciled.status is ReconciliationStatus.SUPPORTED:
        prediction = (
            "Yes"
            if reconciled.boolean_result is True
            else "No"
            if reconciled.boolean_result is False
            else _numeric_prediction(
                float(reconciled.numeric_result or 0.0), reconciled.unit
            )
        )
        proof = reconciled.projection()
        mechanism = "numeric_evidence_reconciler_v2"
        if operator_first_contract_refusal is not None:
            proof = {
                "numeric_reconciliation": proof,
                "operator_first": operator_first_contract_refusal,
            }
            mechanism += "_after_operator_first_contract_non_applicability"
        return _reduction(
            reducer_input,
            mechanism=mechanism,
            status="supported",
            prediction=prediction,
            reason=reconciled.reason,
            handle_ids=reconciled.used_handle_ids,
            proof=reconciled.projection(),
        )
    final_status = (
        "conflicted"
        if (
            operator_first is not None
            and operator_first.status is ExecutionStatus.CONFLICTED
        )
        or reconciled.status is ReconciliationStatus.CONFLICTED
        else "insufficient"
    )
    return _reduction(
        reducer_input,
        mechanism="operator_first_numeric_policy_v1+numeric_evidence_reconciler_v2",
        status=final_status,
        prediction="",
        reason=(
            f"{operator_first.reason};{reconciled.reason}"
            if operator_first is not None
            else "operator_first_contract_non_applicable;"
            f"{reconciled.reason}"
        ),
        handle_ids=(),
        proof={
            "numeric_reconciliation": reconciled.projection(),
            "operator_first": (
                operator_first.projection()
                if operator_first is not None
                else operator_first_contract_refusal
            ),
        },
    )


__all__ = [
    "FORMAT",
    "REDUCTION_FORMAT",
    "HotV6FactHandleBinding",
    "HotV6ItemFactBinding",
    "HotV6TypedReducerError",
    "HotV6TypedReducerInput",
    "HotV6TypedReduction",
    "build_hot_v6_relevant_frontier",
    "build_hot_v6_typed_reducer_input",
    "reduce_hot_v6_typed_ledger",
]
