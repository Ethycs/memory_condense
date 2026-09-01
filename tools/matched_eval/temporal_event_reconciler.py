"""Provider-free, gold-blind reconciliation for question-bound temporal facts.

Unlike a temporal retrieval lane, this module does no discovery.  It accepts a
sealed provider projection, binds evidence to the events or state named in the
question, and only then performs temporal arithmetic.  A result can validate
the candidate or parent, or emit a canonical deterministic answer whose full
inputs and computation are sealed into the proof receipt.
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


FORMAT = "memory-condense-temporal-event-reconciler-proof-v1"
RESOLUTION_FORMAT = f"{FORMAT}-resolution-v1"

_QUESTION_DATE_RE = re.compile(r"^\[Question asked at (\d{4})/(\d{1,2})/(\d{1,2})\b")
_DAY_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
_MONTH_RE = re.compile(
    r"\b(" + "|".join(calendar.month_name[1:]) + r")\s+(\d{4})\b", re.I
)
_DURATION_RE = re.compile(
    r"\b(?P<value>\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve)\s*(?P<unit>days?|weeks?|months?|years?)\b", re.I
)
_NUMBER_WORDS = {word: index for index, word in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
     "nine", "ten", "eleven", "twelve")
)}
_RELATIVE_OFFSET_RE = re.compile(
    r"\b(?P<value>\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|eleven|twelve)\s*(?P<unit>days?|weeks?|months?|years?)\s+"
    r"(?P<direction>before|earlier(?:\s+than)?)\b", re.I,
)
_FIRST_PAIR_RE = re.compile(
    r"\bfirst\s*,\s*(?P<left>.+?)\s+or\s+(?P<right>.+?)\s*[?.!]*$", re.I
)
_HOW_LONG_WHEN_RE = re.compile(
    r"\bhow\s+long\s+(?P<left>.+?)\s+when\s+(?P<right>.+?)\s*[?.!]*$", re.I
)
_HOW_LONG_RE = re.compile(r"\bhow\s+long\b", re.I)
_DAYS_AGO_RE = re.compile(r"\bhow\s+many\s+days\s+ago\b", re.I)
_COMPOSITE_DURATION_RE = re.compile(r"\b(?:combined|in\s+total|altogether)\b", re.I)
_META = frozenset(
    {"ago", "ask", "date", "day", "event", "first", "happen", "how", "long",
     "month", "question", "state", "time", "week", "when", "year"}
)


class TemporalEventReconcilerError(ValueError):
    """Raised when a sealed temporal reconciliation changes identity."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TemporalEventReconcilerError(message)


def _dict(value: object) -> dict[str, Any] | None:
    return dict(value) if type(value) is dict else None


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":"))


def _terms(text: str) -> tuple[str, ...]:
    return tuple(term for term in normalized_terms(text) if term not in _META)


def _interval(text: str) -> tuple[date, date] | None:
    day = _DAY_RE.fullmatch(text.strip())
    if day:
        try:
            value = date(*(int(day.group(i)) for i in range(1, 4)))
        except ValueError:
            return None
        return value, value
    month = _MONTH_RE.fullmatch(text.strip())
    if not month:
        return None
    month_number = tuple(x.casefold() for x in calendar.month_name).index(
        month.group(1).casefold()
    )
    year = int(month.group(2))
    return date(year, month_number, 1), date(
        year, month_number, calendar.monthrange(year, month_number)[1]
    )


def _item_interval(item: Mapping[str, Any]) -> tuple[date, date] | None:
    explicit = item.get("date")
    if type(explicit) is str:
        parsed = _interval(explicit)
        if parsed:
            return parsed
    summary = item.get("summary")
    if type(summary) is not str:
        return None
    leading = re.match(
        r"\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})(?:\s+\d{1,2}:\d{2})?\s*:", summary
    )
    if leading:
        return _interval(leading.group(1))
    introduced = re.search(
        r"\b(?:(?:event\s+)?date\s*:|approximate\s+(?:setup\s+)?date\s*)"
        r"\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
        summary, re.I,
    )
    return _interval(introduced.group(1)) if introduced else None


def _relative_event_interval(
    item: Mapping[str, Any], *, question_date: date
) -> tuple[tuple[date, date], date, dict[str, Any]] | None:
    """Parse a past offset and conservatively project its event-date interval."""

    summary = item.get("summary")
    if type(summary) is not str:
        return None
    match = _RELATIVE_OFFSET_RE.search(summary)
    if not match:
        return None
    raw = match.group("value").casefold()
    value = float(1 if raw in {"a", "an"} else _NUMBER_WORDS.get(raw, raw))
    unit = match.group("unit").casefold().rstrip("s")
    explicit_anchor = _DAY_RE.search(summary[match.end():])
    leading_anchor = re.match(
        r"\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})(?:\s+\d{1,2}:\d{2})?\s*:", summary
    )
    if explicit_anchor:
        anchor_interval = _interval(explicit_anchor.group(0))
        anchor_basis = "explicit_after_relation"
    elif leading_anchor:
        anchor_interval = _interval(leading_anchor.group(1))
        anchor_basis = "leading_statement_date"
    elif re.search(r"\b(?:question|conversation)\b", summary, re.I):
        anchor_interval = (question_date, question_date)
        anchor_basis = "sealed_question_context"
    elif type(item.get("date")) is str:
        anchor_interval = _interval(str(item["date"]))
        anchor_basis = "item_date"
    else:
        return None
    if anchor_interval is None or anchor_interval[0] != anchor_interval[1]:
        return None
    anchor = anchor_interval[0]
    if unit == "day":
        low = high = int(value)
    elif unit == "week":
        low = high = int(value * 7)
    elif unit == "month":
        center = int(value * 30)
        low, high = center - int(value * 3), center + int(value * 3)
    elif unit == "year":
        center = int(value * 365)
        low, high = center - int(value * 7), center + int(value * 7)
    else:  # pragma: no cover - regex restricts the units
        return None
    return (
        (anchor - timedelta(days=high), anchor - timedelta(days=low)),
        anchor,
        {
            "anchor_basis": anchor_basis,
            "anchor_date": anchor.isoformat(),
            "direction": "before",
            "offset_unit": unit,
            "offset_value": value,
            "projected_offset_days_max": high,
            "projected_offset_days_min": low,
        },
    )


def _duration(value: object) -> tuple[float, str] | None:
    if type(value) is not str:
        return None
    match = _DURATION_RE.search(value)
    if not match:
        return None
    unit = match.group("unit").casefold().rstrip("s")
    raw_value = match.group("value").casefold()
    return float(_NUMBER_WORDS.get(raw_value, raw_value)), unit


def _same_duration(left: tuple[float, str], right: tuple[float, str]) -> bool:
    days = {"day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0}
    return abs(left[0] * days[left[1]] - right[0] * days[right[1]]) < 0.01


def _side(anchors: Sequence[str], left: Sequence[str], right: Sequence[str]) -> Literal["left", "right"] | None:
    anchor_set = set(anchors)
    lscore, rscore = len(anchor_set & set(left)), len(anchor_set & set(right))
    lmin, rmin = min(2, len(left)), min(2, len(right))
    if lscore >= lmin and rscore == 0:
        return "left"
    if rscore >= rmin and lscore == 0:
        return "right"
    return None


def _answer_side(answer: str, left: Sequence[str], right: Sequence[str]) -> Literal["left", "right"] | None:
    terms = _terms(answer)
    return _side(terms, left, right)


@dataclass(frozen=True, slots=True)
class TemporalEventResolution:
    prediction: str
    prediction_source: Literal["candidate", "parent", "computed"]
    operation: Literal["direct_duration", "event_interval", "event_order"]
    proof_handle_ids: tuple[str, ...]
    proof_json: str
    proof_receipt_sha256: str
    source_receipt_sha256: str
    receipt_sha256: str = ""
    provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        require_text(self.prediction, "temporal reconciler prediction")
        require_sha256(self.proof_receipt_sha256, "temporal proof")
        require_sha256(self.source_receipt_sha256, "temporal source")
        _require(bool(self.proof_handle_ids), "temporal proof has no handles")
        _require(len(set(self.proof_handle_ids)) == len(self.proof_handle_ids),
                 "temporal proof repeats handles")
        proof = self.proof
        _require(self.proof_json == _canonical(proof), "temporal proof encoding changed")
        _require(identity_sha256(proof) == self.proof_receipt_sha256,
                 "temporal proof receipt changed")
        _require(self.provider_calls == self.retained_transformer_token_state_bytes == 0,
                 "temporal reconciliation escaped zero-call zero-state boundary")
        receipt = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(receipt == self.receipt_sha256, "temporal resolution changed")
        object.__setattr__(self, "receipt_sha256", receipt)
        assert_gold_blind(self.projection(), path="temporal_event_reconciliation")

    @property
    def proof(self) -> dict[str, Any]:
        try:
            value = json.loads(self.proof_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise TemporalEventReconcilerError("temporal proof is not JSON") from exc
        _require(type(value) is dict, "temporal proof type changed")
        return dict(value)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "format": RESOLUTION_FORMAT,
            "operation": self.operation,
            "prediction": self.prediction,
            "prediction_sha256": quote_sha256(self.prediction),
            "prediction_source": self.prediction_source,
            "proof": self.proof,
            "proof_handle_ids": list(self.proof_handle_ids),
            "proof_receipt_sha256": self.proof_receipt_sha256,
            "provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
            "source_receipt_sha256": self.source_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def reconcile_temporal_events(
    *, dated_question: str, candidate_prediction: str, parent_prediction: str,
    provider_input: Mapping[str, Any], validation_contract: Mapping[str, Any],
    allowed_handle_ids: Sequence[str], source_receipt_sha256: str,
) -> TemporalEventResolution | None:
    """Validate candidate or parent from question-bound sealed temporal evidence."""

    require_text(dated_question, "temporal question")
    require_text(candidate_prediction, "temporal candidate")
    require_text(parent_prediction, "temporal parent")
    require_sha256(source_receipt_sha256, "temporal source")
    question_date_match = _QUESTION_DATE_RE.search(dated_question)
    allowed = tuple(allowed_handle_ids)
    if (not question_date_match or not allowed or len(set(allowed)) != len(allowed)
            or any(type(x) is not str or not x for x in allowed)):
        return None
    question_date = date(*(int(question_date_match.group(i)) for i in range(1, 4)))
    question = dated_question.rsplit("]\n", 1)[-1]
    typed = _dict(provider_input.get("typed_evidence"))
    by_handle = _dict(validation_contract.get("by_handle"))
    if typed is None or by_handle is None:
        return None
    inventory = typed.get("handles")
    raw_items = typed.get("items")
    if type(inventory) is not list or type(raw_items) is not list:
        return None
    inventory_by_id = {
        str(row["handle_id"]): dict(row) for raw in inventory
        if (row := _dict(raw)) is not None and row.get("handle_id") in allowed
        and row.get("provenance_grade") in {"exact_citation", "direct_pointer"}
    }

    def evidence(target_terms: Sequence[str], *, dated: bool,
                 exclude_terms: Sequence[str] = ()) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for raw in raw_items:
            item = _dict(raw)
            if item is None or item.get("included") is not True or item.get("content_coherence") != "match":
                continue
            handles = item.get("handle_ids")
            if type(handles) is not list or len(handles) != 1 or handles[0] not in inventory_by_id:
                continue
            handle = str(handles[0])
            contract = _dict(by_handle.get(handle))
            anchors = contract.get("answer_anchor_terms") if contract else None
            receipts = contract.get("usable_item_receipt_sha256s") if contract else None
            if type(anchors) is not list or type(receipts) is not list or not receipts:
                continue
            summary_terms = _terms(str(item.get("summary", "")))
            needed = min(2, len(target_terms))
            # Both the contract and the actual sealed item must bind the named target.
            target_score = len(set(summary_terms) & set(target_terms))
            exclude_score = len(set(summary_terms) & set(exclude_terms))
            if target_score < needed or (exclude_terms and target_score <= exclude_score):
                continue
            relative = _relative_event_interval(item, question_date=question_date)
            interval = relative[0] if relative else _item_interval(item)
            if dated and interval is None:
                continue
            result.append({
                "contract_item_receipt_sha256s": list(receipts),
                "date_end": interval[1].isoformat() if interval else None,
                "date_start": interval[0].isoformat() if interval else None,
                "group_handle": inventory_by_id[handle].get("group_handle"),
                "handle_id": handle,
                "provider_item_sha256": identity_sha256(item),
                "relative_relation": relative[2] if relative else None,
                "relation_anchor_distance_days": (
                    abs((question_date - relative[1]).days) if relative else None
                ),
                "summary_sha256": quote_sha256(str(item.get("summary", ""))),
                "_item": item,
            })
        return result

    def unique_event(rows: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Select one relation-consistent event, retaining corroborating receipts."""

        relative = [row for row in rows if row["relative_relation"] is not None]
        if relative:
            nearest = min(int(row["relation_anchor_distance_days"]) for row in relative)
            rows = [row for row in relative if row["relation_anchor_distance_days"] == nearest]
        if not rows:
            return None
        intervals = {(row["date_start"], row["date_end"]) for row in rows}
        if len(intervals) != 1:
            return None
        return rows

    operation: Literal["direct_duration", "event_interval", "event_order"]
    selected: tuple[str, Literal["candidate", "parent", "computed"]] | None = None
    proof_evidence: list[dict[str, Any]] = []
    computed: dict[str, Any] = {}
    order = _FIRST_PAIR_RE.search(question)
    interval_query = _HOW_LONG_WHEN_RE.search(question)
    if order:
        operation = "event_order"
        left_terms, right_terms = _terms(order.group("left")), _terms(order.group("right"))
        left = evidence(left_terms, dated=True, exclude_terms=right_terms)
        right = evidence(right_terms, dated=True, exclude_terms=left_terms)
        left, right = unique_event(left), unique_event(right)
        if left is None or right is None:
            return None
        li = (date.fromisoformat(left[0]["date_start"]), date.fromisoformat(left[0]["date_end"]))
        ri = (date.fromisoformat(right[0]["date_start"]), date.fromisoformat(right[0]["date_end"]))
        earlier = "left" if li[1] < ri[0] else "right" if ri[1] < li[0] else None
        if earlier is None:
            return None
        for prediction, source in ((candidate_prediction, "candidate"), (parent_prediction, "parent")):
            if _answer_side(prediction, left_terms, right_terms) == earlier:
                selected = prediction, source  # type: ignore[assignment]
                break
        if selected is None:
            named_side = order.group(earlier).strip(" \t\r\n?.!")
            selected = named_side, "computed"
        proof_evidence = left + right
        computed = {"earlier_side": earlier, "left_terms": list(left_terms), "right_terms": list(right_terms)}
    elif interval_query:
        operation = "event_interval"
        left_terms, right_terms = _terms(interval_query.group("left")), _terms(interval_query.group("right"))
        left = evidence(left_terms, dated=True, exclude_terms=right_terms)
        right = evidence(right_terms, dated=True, exclude_terms=left_terms)
        left, right = unique_event(left), unique_event(right)
        if left is None or right is None:
            return None
        left_start = date.fromisoformat(left[0]["date_start"])
        left_end = date.fromisoformat(left[0]["date_end"])
        right_start = date.fromisoformat(right[0]["date_start"])
        right_end = date.fromisoformat(right[0]["date_end"])
        minimum_days = (right_start - left_end).days
        maximum_days = (right_end - left_start).days
        if minimum_days < 0 or maximum_days < minimum_days:
            return None
        expected_min = (float(minimum_days), "day")
        expected_max = (float(maximum_days), "day")
        for prediction, source in ((candidate_prediction, "candidate"), (parent_prediction, "parent")):
            parsed = _duration(prediction)
            if parsed:
                parsed_days = parsed[0] * {"day": 1, "week": 7, "month": 30, "year": 365}[parsed[1]]
            else:
                parsed_days = -1
            if minimum_days <= parsed_days <= maximum_days:
                selected = prediction, source  # type: ignore[assignment]
                break
        if selected is None and minimum_days == maximum_days:
            count = minimum_days
            selected = f"{count} day" + ("" if count == 1 else "s"), "computed"
        proof_evidence = left + right
        computed = {"duration_days_max": maximum_days, "duration_days_min": minimum_days,
                    "left_terms": list(left_terms), "right_terms": list(right_terms)}
    elif _DAYS_AGO_RE.search(question):
        operation = "event_interval"
        target_terms = _terms(question)
        rows = evidence(target_terms, dated=True)
        if len(rows) != 1 or rows[0]["date_start"] != rows[0]["date_end"]:
            return None
        event_day = date.fromisoformat(rows[0]["date_start"])
        if event_day > question_date:
            return None
        count = (question_date - event_day).days
        expected = (float(count), "day")
        for prediction, source in ((candidate_prediction, "candidate"), (parent_prediction, "parent")):
            parsed = _duration(prediction)
            if parsed and _same_duration(parsed, expected):
                selected = prediction, source  # type: ignore[assignment]
                break
        if selected is None:
            selected = f"{count} day" + ("" if count == 1 else "s"), "computed"
        proof_evidence = rows
        computed = {"duration_days": count, "target_terms": list(target_terms)}
    elif _HOW_LONG_RE.search(question) and not _COMPOSITE_DURATION_RE.search(question):
        operation = "direct_duration"
        target_terms = _terms(question)
        rows = evidence(target_terms, dated=False)
        duration_rows: list[tuple[dict[str, Any], tuple[float, str], int]] = []
        for row in rows:
            item = row["_item"]
            parsed = None
            if type(item.get("numeric_value")) in {int, float} and type(item.get("numeric_unit")) is str:
                parsed = (float(item["numeric_value"]), str(item["numeric_unit"]).casefold().rstrip("s"))
            parsed = parsed or _duration(item.get("summary"))
            if parsed and parsed[1] in {"day", "week", "month", "year"}:
                event_interval = _item_interval(item)
                distance = abs((question_date - event_interval[1]).days) if event_interval else 10**9
                duration_rows.append((row, parsed, distance))
        if not duration_rows:
            return None
        duration_rows.sort(key=lambda value: (value[2], allowed.index(value[0]["handle_id"])))
        best = duration_rows[0]
        if len(duration_rows) > 1 and duration_rows[1][2] == best[2] and not _same_duration(duration_rows[1][1], best[1]):
            return None
        for prediction, source in ((candidate_prediction, "candidate"), (parent_prediction, "parent")):
            parsed = _duration(prediction)
            if parsed and _same_duration(parsed, best[1]):
                selected = prediction, source  # type: ignore[assignment]
                break
        if selected is None:
            number = int(best[1][0]) if best[1][0].is_integer() else best[1][0]
            unit = best[1][1] + ("" if best[1][0] == 1 else "s")
            selected = f"{number} {unit}", "computed"
        proof_evidence = [best[0]]
        computed = {"duration_unit": best[1][1], "duration_value": best[1][0], "target_terms": list(target_terms)}
    else:
        return None
    if selected is None:
        return None
    for row in proof_evidence:
        row.pop("_item", None)
    proof = {
        "allowed_handle_ids_sha256": identity_sha256(list(allowed)),
        "candidate_prediction_sha256": quote_sha256(candidate_prediction),
        "computed": computed,
        "evidence": proof_evidence,
        "format": FORMAT,
        "operation": operation,
        "parent_prediction_sha256": quote_sha256(parent_prediction),
        "provider_calls": 0,
        "provider_input_sha256": identity_sha256(provider_input),
        "question_sha256": quote_sha256(dated_question),
        "retained_transformer_token_state_bytes": 0,
        "selected_prediction_sha256": quote_sha256(selected[0]),
        "selected_prediction_source": selected[1],
        "source_receipt_sha256": source_receipt_sha256,
        "validation_contract_sha256": identity_sha256(validation_contract),
    }
    assert_gold_blind(proof, path="temporal_event_reconciler_proof")
    return TemporalEventResolution(
        prediction=selected[0], prediction_source=selected[1], operation=operation,
        proof_handle_ids=tuple(row["handle_id"] for row in proof_evidence),
        proof_json=_canonical(proof), proof_receipt_sha256=identity_sha256(proof),
        source_receipt_sha256=source_receipt_sha256,
    )


__all__ = ["FORMAT", "RESOLUTION_FORMAT", "TemporalEventReconcilerError",
           "TemporalEventResolution", "reconcile_temporal_events"]
