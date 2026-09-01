"""Gold-blind local operand proof for two-event ``... first, X or Y`` queries.

The global temporal bundle is useful for discovery, but it is not an entity
scope.  This module deliberately resolves the two named operands only from the
target-local exact-citation map lane, compares conservative date intervals,
and emits a receipt-bound proof.  It never accepts a global/direct-pointer
event as a substitute for one of the local operands.
"""

from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain.discourse import quote_sha256

from .contracts import assert_gold_blind, identity_sha256, require_sha256, require_text
from .typed_operator_spec import normalized_terms


FORMAT = "memory-condense-local-temporal-pair-proof-v1"
RESOLUTION_FORMAT = f"{FORMAT}-resolution-v1"

_FIRST_PAIR_RE = re.compile(
    r"\bfirst\s*,\s*(?P<left>.+?)\s+or\s+(?P<right>.+?)\s*[?.!]*$",
    re.IGNORECASE,
)
_FIRST_RE = re.compile(r"\bfirst\b", re.IGNORECASE)
_DAY_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
_QUESTION_DATE_RE = re.compile(r"^\[Question asked at (\d{4}/\d{1,2}/\d{1,2})\b")
_RELATIVE_APPROX_DAY_RE = re.compile(
    r"\bdate\s+relative\s+to\s+(?:the\s+)?(?P<query>\d{4}[-/]\d{1,2}[-/]\d{1,2})"
    r"\s+question\s*:\s*(?:about|around|approx(?:imately)?|approximate(?:ly)?)"
    r"\s+(?P<event>\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
    re.IGNORECASE,
)
_MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(calendar.month_name[1:]) + r")\s+(\d{4})\b",
    re.IGNORECASE,
)
_PAIR_META_TERMS = frozenset(
    {
        "ask",
        "date",
        "event",
        "first",
        "happen",
        "question",
        "relative",
        "state",
    }
)


class LocalTemporalPairError(ValueError):
    """Raised when a constructed local temporal proof changes identity."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LocalTemporalPairError(message)


def _exact_dict(value: object) -> dict[str, Any] | None:
    return dict(value) if type(value) is dict else None


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _content_terms(value: str) -> tuple[str, ...]:
    return tuple(term for term in normalized_terms(value) if term not in _PAIR_META_TERMS)


def _date_interval(value: str, *, approximate: bool = False) -> tuple[date, date] | None:
    text = value.strip()
    day = _DAY_RE.fullmatch(text)
    if day is not None:
        try:
            exact = date(*(int(day.group(index)) for index in range(1, 4)))
        except ValueError:
            return None
        if approximate:
            return exact - timedelta(days=3), exact + timedelta(days=3)
        return exact, exact
    month = _MONTH_YEAR_RE.fullmatch(text)
    if month is None:
        return None
    month_number = tuple(name.casefold() for name in calendar.month_name).index(
        month.group(1).casefold()
    )
    year = int(month.group(2))
    return (
        date(year, month_number, 1),
        date(year, month_number, calendar.monthrange(year, month_number)[1]),
    )


def _event_interval(
    item: Mapping[str, Any],
    *,
    question_date: str,
) -> tuple[date, date] | None:
    explicit = item.get("date")
    if type(explicit) is str and explicit:
        parsed = _date_interval(explicit)
        if parsed is not None:
            return parsed
    summary = item.get("summary")
    if type(summary) is not str:
        return None
    approximate = _RELATIVE_APPROX_DAY_RE.search(summary)
    if (
        approximate is not None
        and approximate.group("query").replace("-", "/") == question_date
    ):
        return _date_interval(approximate.group("event"), approximate=True)
    # A summary-derived exact day is allowed only when it is explicitly
    # introduced as the event date.  This avoids accidentally consuming the
    # question timestamp from text such as "relative to the 2023/05/25 query".
    introduced = re.search(
        r"\b(?:event\s+date|date)\s*:\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
        summary,
        re.IGNORECASE,
    )
    return _date_interval(introduced.group(1)) if introduced is not None else None


def _match_score(terms: Sequence[str], anchors: Sequence[str]) -> int:
    return len(set(terms) & set(anchors))


def _matches_one_side(
    anchors: Sequence[str],
    left_terms: Sequence[str],
    right_terms: Sequence[str],
) -> Literal["left", "right"] | None:
    left = _match_score(left_terms, anchors)
    right = _match_score(right_terms, anchors)
    left_min = min(2, len(left_terms))
    right_min = min(2, len(right_terms))
    if left >= left_min and right == 0:
        return "left"
    if right >= right_min and left == 0:
        return "right"
    return None


@dataclass(frozen=True, slots=True)
class LocalTemporalPairResolution:
    prediction: str
    earlier_side: Literal["left", "right"]
    proof_handle_ids: tuple[str, ...]
    proof_json: str
    proof_receipt_sha256: str
    source_completion_sha256: str
    scope_receipt_sha256: str
    receipt_sha256: str = ""
    provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        require_text(self.prediction, "local temporal prediction")
        _require(self.earlier_side in {"left", "right"}, "local temporal side changed")
        _require(
            bool(self.proof_handle_ids)
            and len(self.proof_handle_ids) == len(set(self.proof_handle_ids)),
            "local temporal proof handles changed",
        )
        for value in (
            self.proof_receipt_sha256,
            self.source_completion_sha256,
            self.scope_receipt_sha256,
        ):
            require_sha256(value, "local temporal receipt")
        proof = self.proof
        _require(
            self.proof_json == _canonical_json(proof)
            and self.proof_receipt_sha256 == identity_sha256(proof),
            "local temporal proof changed",
        )
        _require(
            self.provider_calls == 0 and self.retained_transformer_token_state_bytes == 0,
            "local temporal proof escaped zero-call zero-state boundary",
        )
        computed = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == computed, "local temporal resolution changed")
        object.__setattr__(self, "receipt_sha256", computed)
        assert_gold_blind(self.projection(), path="local_temporal_pair_resolution")

    @property
    def proof(self) -> dict[str, Any]:
        try:
            value = json.loads(self.proof_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LocalTemporalPairError("local temporal proof changed encoding") from exc
        _require(type(value) is dict, "local temporal proof changed type")
        return dict(value)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "earlier_side": self.earlier_side,
            "format": RESOLUTION_FORMAT,
            "prediction": self.prediction,
            "prediction_sha256": quote_sha256(self.prediction),
            "proof_handle_ids": list(self.proof_handle_ids),
            "proof": self.proof,
            "proof_receipt_sha256": self.proof_receipt_sha256,
            "provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
            "scope_receipt_sha256": self.scope_receipt_sha256,
            "source_completion_sha256": self.source_completion_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def resolve_parent_from_local_temporal_pair(
    *,
    dated_question: str,
    parent_prediction: str,
    provider_input: Mapping[str, Any],
    validation_contract: Mapping[str, Any],
    allowed_handle_ids: Sequence[str],
    answer_plan_receipt_sha256: str,
    base_scope_receipt_sha256: str,
    source_completion_sha256: str,
) -> LocalTemporalPairResolution | None:
    """Return a sealed keep-parent proof, or ``None`` when any fact is unsafe."""

    require_text(dated_question, "local temporal dated question")
    require_text(parent_prediction, "local temporal parent")
    for value in (
        answer_plan_receipt_sha256,
        base_scope_receipt_sha256,
        source_completion_sha256,
    ):
        require_sha256(value, "local temporal binding")
    allowed = tuple(allowed_handle_ids)
    if (
        not allowed
        or len(allowed) != len(set(allowed))
        or any(type(value) is not str or not value for value in allowed)
        or validation_contract.get("temporal_mode") != "order"
    ):
        return None
    body = dated_question.rsplit("]\n", 1)[-1]
    pair = _FIRST_PAIR_RE.search(body)
    question_date_match = _QUESTION_DATE_RE.search(dated_question)
    if (
        pair is None
        or question_date_match is None
        or len(re.findall(r"\s+or\s+", body, re.IGNORECASE)) != 1
        or _FIRST_RE.search(parent_prediction) is None
    ):
        return None
    left_terms = _content_terms(pair.group("left"))
    right_terms = _content_terms(pair.group("right"))
    if not left_terms or not right_terms:
        return None
    parent_side = _matches_one_side(
        _content_terms(parent_prediction), left_terms, right_terms
    )
    if parent_side is None:
        return None

    typed = _exact_dict(provider_input.get("typed_evidence"))
    contract_by_handle = _exact_dict(validation_contract.get("by_handle"))
    if typed is None or contract_by_handle is None:
        return None
    inventory = typed.get("handles")
    items = typed.get("items")
    if type(inventory) is not list or type(items) is not list:
        return None
    local: dict[str, dict[str, Any]] = {}
    for raw in inventory:
        row = _exact_dict(raw)
        if (
            row is None
            or row.get("origin") != "map"
            or row.get("provenance_grade") != "exact_citation"
            or row.get("handle_id") not in allowed
            or type(row.get("group_handle")) is not str
        ):
            continue
        local[str(row["handle_id"])] = row

    candidates: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for raw in items:
        item = _exact_dict(raw)
        if item is None:
            continue
        handle_ids = item.get("handle_ids")
        if (
            type(handle_ids) is not list
            or len(handle_ids) != 1
            or handle_ids[0] not in local
            or item.get("included") is not True
            or item.get("kind") != "event"
            or item.get("content_coherence") != "match"
            or item.get("value_authority") != "explicit"
            or item.get("status") in {"cancelled", "proposed"}
            or "authored_by_assistant" in str(item.get("relation", ""))
            or "date_basis:row_created_at" in str(item.get("relation", ""))
        ):
            continue
        handle_id = str(handle_ids[0])
        contract = _exact_dict(contract_by_handle.get(handle_id))
        if contract is None:
            continue
        anchors = contract.get("answer_anchor_terms")
        receipts = contract.get("usable_item_receipt_sha256s")
        if (
            type(anchors) is not list
            or type(receipts) is not list
            or not receipts
            or any(type(value) is not str for value in anchors)
            or any(type(value) is not str for value in receipts)
        ):
            continue
        side = _matches_one_side(anchors, left_terms, right_terms)
        interval = _event_interval(
            item,
            question_date=question_date_match.group(1),
        )
        if side is None or interval is None:
            continue
        candidates[side].append(
            {
                "contract_item_receipt_sha256s": list(receipts),
                "date_end": interval[1].isoformat(),
                "date_start": interval[0].isoformat(),
                "group_handle": local[handle_id]["group_handle"],
                "handle_id": handle_id,
                "provider_item_sha256": identity_sha256(item),
            }
        )
    if not candidates["left"] or not candidates["right"]:
        return None
    for side in ("left", "right"):
        candidates[side].sort(key=lambda row: allowed.index(row["handle_id"]))
        if (
            len(candidates[side]) != 1
            or len({row["handle_id"] for row in candidates[side]}) != 1
            or len({row["group_handle"] for row in candidates[side]}) != 1
        ):
            return None
    left_latest = max(date.fromisoformat(row["date_end"]) for row in candidates["left"])
    left_earliest = min(date.fromisoformat(row["date_start"]) for row in candidates["left"])
    right_latest = max(date.fromisoformat(row["date_end"]) for row in candidates["right"])
    right_earliest = min(date.fromisoformat(row["date_start"]) for row in candidates["right"])
    if left_latest < right_earliest:
        earlier: Literal["left", "right"] = "left"
    elif right_latest < left_earliest:
        earlier = "right"
    else:
        return None
    if parent_side != earlier:
        return None

    proof_body = {
        "allowed_handle_ids_sha256": identity_sha256(list(allowed)),
        "answer_plan_receipt_sha256": answer_plan_receipt_sha256,
        "base_scope_receipt_sha256": base_scope_receipt_sha256,
        "earlier_side": earlier,
        "format": FORMAT,
        "left_evidence": candidates["left"],
        "left_terms": list(left_terms),
        "parent_prediction_sha256": quote_sha256(parent_prediction),
        "provider_input_sha256": identity_sha256(provider_input),
        "provider_calls": 0,
        "question_sha256": quote_sha256(dated_question),
        "retained_transformer_token_state_bytes": 0,
        "right_evidence": candidates["right"],
        "right_terms": list(right_terms),
        "source_completion_sha256": source_completion_sha256,
        "validation_contract_sha256": identity_sha256(validation_contract),
    }
    assert_gold_blind(proof_body, path="local_temporal_pair_proof")
    proof_receipt = identity_sha256(proof_body)
    proof_handles = tuple(
        row["handle_id"]
        for side in (earlier, "right" if earlier == "left" else "left")
        for row in candidates[side]
    )
    return LocalTemporalPairResolution(
        prediction=parent_prediction,
        earlier_side=earlier,
        proof_handle_ids=proof_handles,
        proof_json=_canonical_json(proof_body),
        proof_receipt_sha256=proof_receipt,
        source_completion_sha256=source_completion_sha256,
        scope_receipt_sha256=base_scope_receipt_sha256,
    )


__all__ = [
    "FORMAT",
    "RESOLUTION_FORMAT",
    "LocalTemporalPairError",
    "LocalTemporalPairResolution",
    "resolve_parent_from_local_temporal_pair",
]
