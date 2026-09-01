"""Sealed prompt and fail-closed validation for specialist operator proofs.

This adapter keeps parent-union evidence available as explanatory context while
making the narrower ``specialist_advisories`` plane authoritative for numeric
reduction, temporal selection/order, and scoped absence.  It is deliberately
question-identity-agnostic and provider-free.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .typed_numeric_semantics import (
    NumericDimension,
    NumericQualifier,
    numeric_mentions,
)
from .typed_operator_adapter import conservative_numeric_value
from .typed_operator_spec import normalized_terms


FORMAT = "memory-condense-specialist-scoped-completion-v1"
PROMPT_FORMAT = f"{FORMAT}-prompt-v1"
SCOPE_FORMAT = f"{FORMAT}-scope-v1"
DECISION_FORMAT = f"{FORMAT}-decision-v1"
VALIDATION_CONTRACT_FORMAT = (
    "memory-condense-typed-memory-final-arm-v1-"
    "completion-validation-contract-v3"
)

HARD_COMPLETE_CHAT_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768
MAX_CHAT_PROMPT_TOKENS = HARD_COMPLETE_CHAT_TOKEN_CAP - OUTPUT_TOKEN_RESERVE

SPECIALIST_SYSTEM_PROMPT = (
    "Answer one dated long-memory question from the supplied typed evidence. "
    "The protected parent prediction is fallback-not-evidence. Return exactly "
    "one JSON object with exactly decision, prediction, used_handle_ids; no "
    "markdown. decision is keep_parent or replace. keep_parent requires the "
    "exact protected parent prediction and an empty handle list. replace "
    "requires concise grounded text and one or more supplied H handles. "
    "specialist_advisories are sealed operator-proof scopes. For a numeric "
    "operand_groups advisory, reduce each group exactly once, treat repeated "
    "candidate handles inside one group as the same event, and cite at least "
    "one advisory H handle per group. In a temporal_order bundle, enumerate "
    "ordered_handle_ids in exactly that order. In a temporal_relative bundle, "
    "answer from winner_handle_id only; predecessor_handle_id is only a "
    "comparator, so do not enumerate the bundle. In a temporal_interval, "
    "compute from the winner date to query_time with whole calendar months "
    "(adjusting for the day), otherwise days. For a profile "
    "preference advisory, cite only its coherent specialist cluster and ground "
    "the prediction in meaningful user-memory terms. Never expand an "
    "aggregate or timeline with parent-union handles outside that advisory. "
    "For an applicable absence_certificate with "
    "may_conclude_operator_insufficient true, report concise insufficiency for "
    "the missing slot and never copy a value from another entity; cite only "
    "its selected supporting H handles. Parent evidence outside a specialist "
    "scope may explain context but cannot become an operand. Preserve "
    "approximate or bounded numeric wording. If no advisory proof safely "
    "supports replacement, keep the protected parent exactly."
)

_FORBIDDEN_PROVIDER_KEYS = frozenset(
    {
        "namespace_id",
        "partition_id",
        "question_id",
        "source_id",
        "source_prefix",
        "store_path",
    }
)
_HANDLE_RE = re.compile(r"^H[0-9]+$")
_GROUP_RE = re.compile(r"^G[0-9]+$")
_INSUFFICIENCY_RE = re.compile(
    r"\b(?:insufficient|unknown|not\s+(?:provided|recorded|specified|known)|"
    r"cannot\s+determine|can't\s+determine|do\s+not\s+(?:have|know)|"
    r"don't\s+(?:have|know)|no\s+(?:explicit\s+)?(?:count|number|memory\s+evidence)|"
    r"not\s+enough\s+(?:information|evidence))\b",
    re.IGNORECASE,
)
_META_ANCHOR_TERMS = frozenset(
    {
        "author",
        "before",
        "bundle",
        "by",
        "corroborat",
        "day",
        "event",
        "exact",
        "handle",
        "history",
        "link",
        "near",
        "operand",
        "order",
        "predecessor",
        "relation",
        "role",
        "scope",
        "slot",
        "support",
        "target",
        "temporal",
        "user",
        "window",
        "winner",
        "within",
    }
)
_NUMERIC_ANSWER_QUESTION_TERMS = frozenset(
    {
        "age",
        "amount",
        "budget",
        "cost",
        "count",
        "discount",
        "distance",
        "duration",
        "far",
        "long",
        "many",
        "measure",
        "money",
        "much",
        "number",
        "paid",
        "pay",
        "percent",
        "percentage",
        "price",
        "quantity",
        "rate",
        "rating",
        "salary",
        "score",
        "spend",
        "spent",
        "temperature",
        "total",
        "weigh",
        "weight",
    }
)
_COUNT_QUESTION_TERMS = frozenset({"count", "many", "number", "quantity"})
_DURATION_UNITS = frozenset(
    {"second", "minute", "hour", "day", "week", "month", "year"}
)
_MEASURE_UNIT_ALIASES = {
    "feet": "feet",
    "foot": "foot",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "inch": "inch",
    "inches": "inch",
    "kg": "kg",
    "kgs": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "kilometer": "km",
    "kilometers": "km",
    "km": "km",
    "lb": "lb",
    "lbs": "lb",
    "meter": "meter",
    "meters": "meter",
    "metre": "metre",
    "metres": "metre",
    "mile": "mile",
    "miles": "mile",
    "ounce": "oz",
    "ounces": "oz",
    "oz": "oz",
    "pound": "lb",
    "pounds": "lb",
}


class SpecialistScopedCompletionError(MatchedEvalContractError):
    """Raised when a specialist prompt, proof, or provenance seal changes."""


class SpecialistProofKind(str, Enum):
    NUMERIC_OPERAND_GROUPS = "numeric_operand_groups"
    TEMPORAL_ORDER = "temporal_order"
    TEMPORAL_RELATIVE = "temporal_relative"
    TEMPORAL_INTERVAL = "temporal_interval"
    ABSENCE_CERTIFICATE = "absence_certificate"
    PROFILE_PREFERENCE = "profile_preference"


def _require(ok: object, message: str) -> None:
    if not ok:
        raise SpecialistScopedCompletionError(message)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _ordered_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    _require(
        all(type(value) is str and bool(value) for value in result)
        and len(set(result)) == len(result),
        f"{label} must be ordered unique text",
    )
    return result


def _handle(value: object, label: str) -> str:
    _require(type(value) is str and _HANDLE_RE.fullmatch(value) is not None, label)
    assert isinstance(value, str)
    return value


def _group(value: object, label: str) -> str:
    _require(type(value) is str and _GROUP_RE.fullmatch(value) is not None, label)
    assert isinstance(value, str)
    return value


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    assert type(value) is dict
    return dict(value)


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact list")
    assert type(value) is list
    return list(value)


def _reject_provider_locator_keys(value: object, *, path: str = "provider") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            _require(
                normalized not in _FORBIDDEN_PROVIDER_KEYS,
                f"raw locator key escaped into {path}.{key}",
            )
            _reject_provider_locator_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_provider_locator_keys(child, path=f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class SpecialistPromptEnvelope:
    messages_json: str
    provider_input_sha256: str
    specialist_advisories_sha256: str
    prompt_token_proxy: int
    output_token_reserve: Literal[768] = OUTPUT_TOKEN_RESERVE
    hard_complete_chat_token_cap: Literal[8000] = HARD_COMPLETE_CHAT_TOKEN_CAP
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        try:
            raw_messages = json.loads(self.messages_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SpecialistScopedCompletionError(
                "specialist prompt messages changed encoding"
            ) from exc
        _require(
            type(raw_messages) is list
            and len(raw_messages) == 2
            and all(
                type(row) is dict
                and set(row) == {"content", "role"}
                and row.get("role") in {"system", "user"}
                and type(row.get("content")) is str
                for row in raw_messages
            )
            and raw_messages[0]["content"] == SPECIALIST_SYSTEM_PROMPT
            and self.messages_json == _canonical_json(raw_messages),
            "specialist prompt messages changed schema",
        )
        require_sha256(self.provider_input_sha256, "specialist provider input")
        require_sha256(self.specialist_advisories_sha256, "specialist advisories")
        _require(
            type(self.prompt_token_proxy) is int
            and self.prompt_token_proxy
            == count_chat_prompt_token_proxy(raw_messages)
            and self.output_token_reserve == OUTPUT_TOKEN_RESERVE
            and self.hard_complete_chat_token_cap == HARD_COMPLETE_CHAT_TOKEN_CAP
            and self.prompt_token_proxy + self.output_token_reserve
            <= self.hard_complete_chat_token_cap
            and self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0,
            "specialist prompt escaped the exact hard-budget contract",
        )
        computed = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == computed, "specialist prompt receipt changed")
        object.__setattr__(self, "receipt_sha256", computed)
        assert_gold_blind(self.projection(), path="specialist_scoped_prompt")

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        rows = json.loads(self.messages_json)
        return tuple(dict(row) for row in rows)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": PROMPT_FORMAT,
            "hard_complete_chat_token_cap": self.hard_complete_chat_token_cap,
            "messages": list(self.messages),
            "messages_sha256": identity_sha256(list(self.messages)),
            "output_token_reserve": self.output_token_reserve,
            "prompt_token_proxy": self.prompt_token_proxy,
            "provider_input_sha256": self.provider_input_sha256,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "specialist_advisories_sha256": self.specialist_advisories_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def render_specialist_scoped_prompt(
    provider_input: Mapping[str, Any],
    *,
    output_token_reserve: int = OUTPUT_TOKEN_RESERVE,
    hard_complete_chat_token_cap: int = HARD_COMPLETE_CHAT_TOKEN_CAP,
) -> SpecialistPromptEnvelope:
    """Render and seal the actual specialist-aware provider prompt."""

    _require(type(provider_input) is dict, "specialist provider input must be exact")
    value = dict(provider_input)
    advisories = _exact_list(
        value.get("specialist_advisories"), "specialist advisories"
    )
    _require(bool(advisories), "specialist prompt requires at least one advisory")
    _reject_provider_locator_keys(value)
    assert_gold_blind(value, path="specialist_scoped_provider_input")
    _require(
        output_token_reserve == OUTPUT_TOKEN_RESERVE
        and hard_complete_chat_token_cap == HARD_COMPLETE_CHAT_TOKEN_CAP,
        "specialist renderer budget constants changed",
    )
    messages = [
        {"role": "system", "content": SPECIALIST_SYSTEM_PROMPT},
        {"role": "user", "content": _canonical_json(value)},
    ]
    return SpecialistPromptEnvelope(
        messages_json=_canonical_json(messages),
        provider_input_sha256=identity_sha256(value),
        specialist_advisories_sha256=identity_sha256(advisories),
        prompt_token_proxy=count_chat_prompt_token_proxy(messages),
    )


@dataclass(frozen=True, slots=True)
class SpecialistHandleEvidence:
    handle_id: str
    group_handle: str
    semantic_rows_json: str
    usable_item_receipt_sha256s: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _handle(self.handle_id, "specialist evidence handle changed")
        _group(self.group_handle, "specialist evidence group changed")
        _ordered_unique(
            self.usable_item_receipt_sha256s, "specialist usable item receipts"
        )
        _require(
            bool(self.usable_item_receipt_sha256s),
            "specialist handle requires usable evidence",
        )
        for value in self.usable_item_receipt_sha256s:
            require_sha256(value, "specialist usable item")
        rows = self.semantic_rows
        _require(bool(rows), "specialist handle requires semantic rows")
        computed = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == computed, "specialist evidence receipt changed")
        object.__setattr__(self, "receipt_sha256", computed)

    @property
    def semantic_rows(self) -> tuple[dict[str, Any], ...]:
        try:
            rows = json.loads(self.semantic_rows_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SpecialistScopedCompletionError(
                "specialist semantic rows changed encoding"
            ) from exc
        _require(
            type(rows) is list
            and self.semantic_rows_json == _canonical_json(rows)
            and all(type(row) is dict for row in rows),
            "specialist semantic rows changed schema",
        )
        return tuple(dict(row) for row in rows)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "group_handle": self.group_handle,
            "handle_id": self.handle_id,
            "semantic_rows": list(self.semantic_rows),
            "usable_item_receipt_sha256s": list(
                self.usable_item_receipt_sha256s
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SpecialistProof:
    kind: SpecialistProofKind
    mechanism_id: str
    handle_ids: tuple[str, ...]
    advisory_receipt_sha256: str
    payload_json: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.kind) is SpecialistProofKind, "specialist proof kind changed")
        require_text(self.mechanism_id, "specialist proof mechanism")
        _ordered_unique(self.handle_ids, "specialist proof handles")
        for handle_id in self.handle_ids:
            _handle(handle_id, "specialist proof handle changed")
        require_sha256(self.advisory_receipt_sha256, "specialist advisory receipt")
        payload = self.payload
        _require(
            self.payload_json == _canonical_json(payload),
            "specialist proof payload changed encoding",
        )
        computed = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == computed, "specialist proof receipt changed")
        object.__setattr__(self, "receipt_sha256", computed)

    @property
    def payload(self) -> dict[str, Any]:
        try:
            value = json.loads(self.payload_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SpecialistScopedCompletionError(
                "specialist proof payload changed encoding"
            ) from exc
        return _exact_dict(value, "specialist proof payload")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "advisory_receipt_sha256": self.advisory_receipt_sha256,
            "handle_ids": list(self.handle_ids),
            "kind": self.kind.value,
            "mechanism_id": self.mechanism_id,
            "payload": self.payload,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SpecialistValidationScope:
    terminal_allowed_handle_ids: tuple[str, ...]
    specialist_declared_handle_ids: tuple[str, ...]
    handle_evidence: tuple[SpecialistHandleEvidence, ...]
    proofs: tuple[SpecialistProof, ...]
    specialist_advisories_sha256: str
    validation_contract_sha256: str
    sealed_source_receipt_sha256: str
    prompt_envelope_receipt_sha256: str
    prompt_token_proxy: int
    output_token_reserve: Literal[768] = OUTPUT_TOKEN_RESERVE
    hard_complete_chat_token_cap: Literal[8000] = HARD_COMPLETE_CHAT_TOKEN_CAP
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        terminal = _ordered_unique(
            self.terminal_allowed_handle_ids, "terminal allowed handles"
        )
        declared = _ordered_unique(
            self.specialist_declared_handle_ids, "specialist declared handles"
        )
        for value in (*terminal, *declared):
            _handle(value, "specialist scope handle changed")
        _require(
            set(declared) <= set(terminal),
            "specialist handles escaped terminal allowed handles",
        )
        _require(
            type(self.handle_evidence) is tuple
            and all(type(row) is SpecialistHandleEvidence for row in self.handle_evidence)
            and tuple(row.handle_id for row in self.handle_evidence) == declared,
            "specialist evidence does not exactly cover declared handles",
        )
        _require(
            type(self.proofs) is tuple
            and all(type(row) is SpecialistProof for row in self.proofs)
            and all(set(row.handle_ids) <= set(declared) for row in self.proofs),
            "specialist proofs escaped declared handles",
        )
        for value, label in (
            (self.specialist_advisories_sha256, "specialist advisories"),
            (self.validation_contract_sha256, "specialist validation contract"),
            (self.sealed_source_receipt_sha256, "specialist sealed source"),
            (self.prompt_envelope_receipt_sha256, "specialist prompt envelope"),
        ):
            require_sha256(value, label)
        _require(
            type(self.prompt_token_proxy) is int
            and self.output_token_reserve == OUTPUT_TOKEN_RESERVE
            and self.hard_complete_chat_token_cap == HARD_COMPLETE_CHAT_TOKEN_CAP
            and self.prompt_token_proxy + self.output_token_reserve
            <= self.hard_complete_chat_token_cap
            and self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0,
            "specialist scope escaped the exact hard-budget contract",
        )
        computed = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == computed, "specialist scope receipt changed")
        object.__setattr__(self, "receipt_sha256", computed)
        assert_gold_blind(self.projection(), path="specialist_validation_scope")

    @property
    def evidence_by_handle(self) -> dict[str, SpecialistHandleEvidence]:
        return {row.handle_id: row for row in self.handle_evidence}

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": SCOPE_FORMAT,
            "handle_evidence": [row.projection() for row in self.handle_evidence],
            "hard_complete_chat_token_cap": self.hard_complete_chat_token_cap,
            "output_token_reserve": self.output_token_reserve,
            "proofs": [row.projection() for row in self.proofs],
            "prompt_envelope_receipt_sha256": self.prompt_envelope_receipt_sha256,
            "prompt_token_proxy": self.prompt_token_proxy,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "sealed_source_receipt_sha256": self.sealed_source_receipt_sha256,
            "specialist_advisories_sha256": self.specialist_advisories_sha256,
            "specialist_declared_handle_ids": list(
                self.specialist_declared_handle_ids
            ),
            "terminal_allowed_handle_ids": list(self.terminal_allowed_handle_ids),
            "validation_contract_sha256": self.validation_contract_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _validated_semantic_row(
    raw: object,
    *,
    usable_item_receipts: set[str],
    include_proposed: bool,
) -> dict[str, Any]:
    row = _exact_dict(raw, "specialist semantic row")
    item_receipt = require_sha256(
        row.get("item_receipt_sha256"), "specialist semantic item"
    )
    require_sha256(row.get("semantic_unit_sha256"), "specialist semantic unit")
    numeric = row.get("numeric_value")
    date = row.get("date")
    unit = row.get("unit")
    _require(
        item_receipt in usable_item_receipts
        and row.get("status") not in {"cancelled", None}
        and (include_proposed or row.get("status") != "proposed")
        and (numeric is None or type(numeric) in {int, float} and math.isfinite(numeric))
        and (date is None or type(date) is str and bool(date))
        and (unit is None or type(unit) is str and bool(unit)),
        "specialist semantic scalar/status changed",
    )
    for key in (
        "action_concepts",
        "completed_action_concepts",
        "entity_terms",
        "group_terms",
        "relation_terms",
        "summary_terms",
        "supported_slot_ids",
    ):
        values = row.get(key)
        _require(
            type(values) is list
            and all(type(value) is str and bool(value) for value in values),
            f"specialist semantic {key} changed",
        )
    return row


def _compile_handle_evidence(
    handle_id: str,
    group_handle: str,
    contract: Mapping[str, Any],
    *,
    include_proposed: bool,
) -> SpecialistHandleEvidence:
    item_receipts = _ordered_unique(
        _exact_list(
            contract.get("usable_item_receipt_sha256s"),
            "specialist usable item receipts",
        ),
        "specialist usable item receipts",
    )
    _require(bool(item_receipts), "specialist handle has no usable items")
    for value in item_receipts:
        require_sha256(value, "specialist usable item")
    statuses = _exact_list(contract.get("status_values"), "specialist statuses")
    _require(
        bool(statuses)
        and all(type(value) is str and bool(value) for value in statuses)
        and "cancelled" not in statuses
        and (include_proposed or "proposed" not in statuses),
        "specialist handle is not usable",
    )
    rows = tuple(
        _validated_semantic_row(
            raw,
            usable_item_receipts=set(item_receipts),
            include_proposed=include_proposed,
        )
        for raw in _exact_list(
            contract.get("semantic_rows"), "specialist semantic rows"
        )
    )
    _require(bool(rows), "specialist handle has no semantic rows")
    return SpecialistHandleEvidence(
        handle_id,
        group_handle,
        _canonical_json(list(rows)),
        item_receipts,
    )


def _numeric_proof(
    advisory: Mapping[str, Any],
    candidate_map: Mapping[str, str],
    evidence_by_handle: Mapping[str, SpecialistHandleEvidence],
    handle_group_by_id: Mapping[str, str],
    advisory_receipt_sha256: str,
) -> SpecialistProof:
    raw_groups = _exact_list(advisory.get("operand_groups"), "numeric operand groups")
    _require(bool(raw_groups), "numeric advisory requires operand groups")
    normalized_groups: list[dict[str, Any]] = []
    proof_handles: list[str] = []
    modes: list[str] = []
    units: set[str] = set()
    computed_scalar = 0.0
    candidate_owners: set[str] = set()
    for raw in raw_groups:
        group = _exact_dict(raw, "numeric operand group")
        _require(
            set(group)
            == {
                "action_class",
                "candidate_ids",
                "entity_key",
                "operand_values",
                "operation_mode",
                "source_group_handles",
                "value_basis",
            },
            "numeric operand group schema changed",
        )
        candidates = _ordered_unique(
            _exact_list(group["candidate_ids"], "numeric group candidates"),
            "numeric group candidates",
        )
        _require(
            bool(candidates)
            and set(candidates) <= set(candidate_map)
            and not set(candidates) & candidate_owners,
            "numeric group candidates escaped or overlap",
        )
        candidate_owners.update(candidates)
        handles = tuple(candidate_map[value] for value in candidates)
        source_groups = _ordered_unique(
            _exact_list(group["source_group_handles"], "numeric source groups"),
            "numeric source groups",
        )
        for value in source_groups:
            _group(value, "numeric source group changed")
        _require(
            source_groups
            == tuple(dict.fromkeys(handle_group_by_id[value] for value in handles)),
            "numeric source groups disagree with handle provenance",
        )
        values = _exact_list(group["operand_values"], "numeric operand values")
        _require(
            len(values) == 1
            and type(values[0]) in {int, float}
            and math.isfinite(values[0]),
            "numeric operand group must have one finite scalar",
        )
        scalar = float(values[0])
        mode = group.get("operation_mode")
        _require(mode in {"count", "sum"}, "numeric operation mode changed")
        if mode == "count":
            _require(scalar >= 0, "numeric count operand became negative")
        for handle_id in handles:
            matches = tuple(
                row
                for row in evidence_by_handle[handle_id].semantic_rows
                if row.get("numeric_value") is not None
                and abs(float(row["numeric_value"]) - scalar) <= 1e-9
            )
            _require(
                bool(matches),
                "numeric advisory scalar is not bound to every candidate handle",
            )
            units.update(
                str(row["unit"]) for row in matches if row.get("unit") is not None
            )
        modes.append(str(mode))
        computed_scalar += scalar
        proof_handles.extend(handles)
        normalized_groups.append(
            {
                "action_class": require_text(
                    group.get("action_class"), "numeric action class"
                ),
                "candidate_ids": list(candidates),
                "entity_key": require_text(group.get("entity_key"), "numeric entity"),
                "handle_ids": list(handles),
                "operand_value": scalar,
                "source_group_handles": list(source_groups),
                "value_basis": require_text(
                    group.get("value_basis"), "numeric value basis"
                ),
            }
        )
    _require(len(set(modes)) == 1, "numeric advisory mixed operation modes")
    _require(len(units) <= 1, "numeric advisory mixed units")
    payload = {
        "computed_scalar": computed_scalar,
        "groups": normalized_groups,
        "operation_mode": modes[0],
        "unit": next(iter(units), None),
    }
    return SpecialistProof(
        SpecialistProofKind.NUMERIC_OPERAND_GROUPS,
        require_text(advisory.get("mechanism_id"), "numeric mechanism"),
        tuple(dict.fromkeys(proof_handles)),
        advisory_receipt_sha256,
        _canonical_json(payload),
    )


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SpecialistScopedCompletionError(
            "temporal advisory date changed format"
        ) from exc


def _temporal_proof(
    advisory: Mapping[str, Any],
    candidate_map: Mapping[str, str],
    evidence_by_handle: Mapping[str, SpecialistHandleEvidence],
    advisory_receipt_sha256: str,
) -> SpecialistProof:
    bundle = _exact_dict(advisory.get("temporal_bundle"), "temporal bundle")
    _require(
        set(bundle)
        == {
            "ordered_candidate_ids",
            "ordered_handle_ids",
            "original_population_count",
            "predecessor_candidate_id",
            "predecessor_handle_id",
            "query_time",
            "requested_cardinality",
            "route",
            "target_date",
            "terminal_selection_truncated",
            "winner_candidate_id",
            "winner_handle_id",
        },
        "temporal bundle schema changed",
    )
    candidates = _ordered_unique(
        _exact_list(bundle["ordered_candidate_ids"], "temporal candidates"),
        "temporal candidates",
    )
    handles = _ordered_unique(
        _exact_list(bundle["ordered_handle_ids"], "temporal handles"),
        "temporal handles",
    )
    _require(
        bool(candidates)
        and len(candidates) == len(handles)
        and set(candidates) <= set(candidate_map)
        and tuple(candidate_map[value] for value in candidates) == handles,
        "temporal candidate/handle order changed",
    )
    moments: list[datetime] = []
    dates: list[str] = []
    for handle_id in handles:
        handle_dates = tuple(
            dict.fromkeys(
                str(row["date"])
                for row in evidence_by_handle[handle_id].semantic_rows
                if row.get("date") is not None
            )
        )
        _require(
            len(handle_dates) == 1,
            "temporal handle does not bind one exact date",
        )
        dates.append(handle_dates[0])
        moments.append(_parse_datetime(handle_dates[0]))
    _require(
        all(left <= right for left, right in zip(moments, moments[1:])),
        "temporal advisory order disagrees with evidence dates",
    )
    original_count = bundle.get("original_population_count")
    requested = bundle.get("requested_cardinality")
    _require(
        type(original_count) is int
        and original_count >= len(handles)
        and (requested is None or type(requested) is int and requested >= 1)
        and type(bundle.get("terminal_selection_truncated")) is bool,
        "temporal bundle counts changed",
    )
    route = require_text(bundle.get("route"), "temporal route")
    _require(route.startswith("temporal_"), "temporal route changed")
    if route == "temporal_order" and requested is not None:
        _require(
            requested == len(handles),
            "temporal order cardinality differs from sealed handles",
        )

    def role_pair(candidate_key: str, handle_key: str) -> tuple[str | None, str | None]:
        candidate = bundle.get(candidate_key)
        handle_id = bundle.get(handle_key)
        _require(
            (candidate is None and handle_id is None)
            or (
                type(candidate) is str
                and candidate in candidate_map
                and type(handle_id) is str
                and candidate_map[candidate] == handle_id
                and handle_id in handles
            ),
            f"temporal {candidate_key} binding changed",
        )
        return candidate, handle_id

    winner_candidate, winner_handle = role_pair(
        "winner_candidate_id", "winner_handle_id"
    )
    predecessor_candidate, predecessor_handle = role_pair(
        "predecessor_candidate_id", "predecessor_handle_id"
    )
    _require(winner_handle is not None, "temporal bundle lost its winner")
    if predecessor_handle is not None:
        _require(
            handles.index(predecessor_handle) < handles.index(winner_handle),
            "temporal predecessor no longer precedes winner",
        )
    query_time = require_text(bundle.get("query_time"), "temporal query time")
    query_moment = _parse_datetime(query_time)
    target_date = bundle.get("target_date")
    _require(
        target_date is None or type(target_date) is str and bool(target_date),
        "temporal target date changed",
    )
    if target_date is not None:
        _parse_datetime(target_date)
    if route == "temporal_order":
        kind = SpecialistProofKind.TEMPORAL_ORDER
    elif route == "temporal_interval":
        kind = SpecialistProofKind.TEMPORAL_INTERVAL
    else:
        kind = SpecialistProofKind.TEMPORAL_RELATIVE
    interval_projection: dict[str, Any] | None = None
    if kind is SpecialistProofKind.TEMPORAL_INTERVAL:
        start = moments[handles.index(winner_handle)]
        end = query_moment
        _require(end >= start, "temporal interval boundaries reversed")
        days = (end - start).days
        months = (end.year - start.year) * 12 + end.month - start.month
        if end.day < start.day:
            months -= 1
        interval_projection = {
            "computed_scalar": float(months if months >= 1 else days),
            "end": query_time,
            "start": dates[handles.index(winner_handle)],
            "start_handle_id": winner_handle,
            "unit": "month" if months >= 1 else "day",
        }
    payload = {
        "dates": dates,
        "ordered_candidate_ids": list(candidates),
        "ordered_handle_ids": list(handles),
        "original_population_count": original_count,
        "interval": interval_projection,
        "predecessor_candidate_id": predecessor_candidate,
        "predecessor_handle_id": predecessor_handle,
        "query_time": query_time,
        "question_terms": list(
            _ordered_unique(
                _exact_list(
                    advisory["_validation_question_terms"],
                    "temporal validation question terms",
                ),
                "temporal validation question terms",
            )
        ),
        "requested_cardinality": requested,
        "route": route,
        "target_date": target_date,
        "terminal_selection_truncated": bundle["terminal_selection_truncated"],
        "winner_candidate_id": winner_candidate,
        "winner_handle_id": winner_handle,
    }
    return SpecialistProof(
        kind,
        require_text(advisory.get("mechanism_id"), "temporal mechanism"),
        handles,
        advisory_receipt_sha256,
        _canonical_json(payload),
    )


def _absence_proof(
    advisory: Mapping[str, Any],
    candidate_map: Mapping[str, str],
    evidence_by_handle: Mapping[str, SpecialistHandleEvidence],
    advisory_receipt_sha256: str,
) -> SpecialistProof:
    certificate = _exact_dict(
        advisory.get("absence_certificate"), "absence certificate"
    )
    _require(
        set(certificate)
        == {
            "applicable",
            "every_exact_entity_posting_scanned",
            "every_scoped_source_row_scanned",
            "may_conclude_operator_insufficient",
            "physical_content_rows_scanned",
            "physical_sentence_windows_scanned",
            "provider_instruction",
            "scope_definition",
            "scoped_content_row_count",
            "scoped_source_count",
            "semantic_absence_may_be_inferred",
            "slot_coverage",
        },
        "absence certificate schema changed",
    )
    _require(
        certificate["applicable"] is True
        and certificate["every_exact_entity_posting_scanned"] is True
        and certificate["every_scoped_source_row_scanned"] is True
        and certificate["may_conclude_operator_insufficient"] is True
        and certificate["semantic_absence_may_be_inferred"] is False,
        "absence certificate is not a complete scoped proof",
    )
    for key in (
        "physical_content_rows_scanned",
        "physical_sentence_windows_scanned",
        "scoped_content_row_count",
        "scoped_source_count",
    ):
        _require(
            type(certificate[key]) is int and certificate[key] >= 0,
            f"absence certificate {key} changed",
        )
    instruction = require_text(
        certificate.get("provider_instruction"), "absence provider instruction"
    )
    require_text(certificate.get("scope_definition"), "absence scope definition")
    _require(
        _INSUFFICIENCY_RE.search(instruction) is not None,
        "absence provider instruction no longer requires insufficiency",
    )
    slots: list[dict[str, Any]] = []
    selected_handles: list[str] = []
    missing_count = 0
    for raw in _exact_list(certificate["slot_coverage"], "absence slot coverage"):
        slot = _exact_dict(raw, "absence slot")
        _require(
            set(slot)
            == {
                "entity_assertion_source_count",
                "entity_assertion_window_count",
                "exact_entity_terms",
                "explicit_numeric_assertion_source_count",
                "explicit_numeric_assertion_window_count",
                "explicit_numeric_operand_missing",
                "scope_has_grounded_predicate_assertion",
                "selected_supporting_handle_ids",
                "slot_id",
                "slot_label",
            },
            "absence slot schema changed",
        )
        require_sha256(slot.get("slot_id"), "absence slot")
        label = require_text(slot.get("slot_label"), "absence slot label")
        entity_terms = _ordered_unique(
            _exact_list(slot["exact_entity_terms"], "absence exact entity terms"),
            "absence exact entity terms",
        )
        _require(bool(entity_terms), "absence slot lost its entity terms")
        supports = _ordered_unique(
            _exact_list(
                slot["selected_supporting_handle_ids"],
                "absence supporting handles",
            ),
            "absence supporting handles",
        )
        _require(
            set(supports) <= set(candidate_map.values()),
            "absence supporting handle escaped advisory map",
        )
        for key in (
            "entity_assertion_source_count",
            "entity_assertion_window_count",
            "explicit_numeric_assertion_source_count",
            "explicit_numeric_assertion_window_count",
        ):
            _require(
                type(slot[key]) is int and slot[key] >= 0,
                f"absence slot {key} changed",
            )
        missing = slot.get("explicit_numeric_operand_missing")
        grounded = slot.get("scope_has_grounded_predicate_assertion")
        _require(
            type(missing) is bool and type(grounded) is bool and grounded,
            "absence slot flags changed",
        )
        if missing:
            missing_count += 1
            _require(
                slot["explicit_numeric_assertion_source_count"] == 0
                and slot["explicit_numeric_assertion_window_count"] == 0
                and not supports,
                "absence missing slot retained a numeric support",
            )
        else:
            _require(bool(supports), "absence covered slot lost supporting handles")
            evidence_terms = set().union(
                *(
                    _semantic_terms(evidence_by_handle[handle_id])
                    for handle_id in supports
                )
            )
            _require(
                set(entity_terms) & evidence_terms,
                "absence supporting handle is not entity-grounded",
            )
        selected_handles.extend(supports)
        slots.append(
            {
                "exact_entity_terms": list(entity_terms),
                "explicit_numeric_operand_missing": missing,
                "selected_supporting_handle_ids": list(supports),
                "slot_id": slot["slot_id"],
                "slot_label": label,
            }
        )
    _require(missing_count > 0, "absence certificate has no missing slot")
    proof_handles = tuple(dict.fromkeys(selected_handles))
    allowed_numeric_values = sorted(
        {
            float(row["numeric_value"])
            for handle_id in proof_handles
            for row in evidence_by_handle[handle_id].semantic_rows
            if row.get("numeric_value") is not None
        }
    )
    payload = {
        "allowed_numeric_values": allowed_numeric_values,
        "provider_instruction": instruction,
        "scope_definition_sha256": quote_sha256(certificate["scope_definition"]),
        "slots": slots,
    }
    return SpecialistProof(
        SpecialistProofKind.ABSENCE_CERTIFICATE,
        require_text(advisory.get("mechanism_id"), "absence mechanism"),
        proof_handles,
        advisory_receipt_sha256,
        _canonical_json(payload),
    )


def _semantic_terms(evidence: SpecialistHandleEvidence) -> set[str]:
    result: set[str] = set()
    for row in evidence.semantic_rows:
        for key in (
            "action_concepts",
            "completed_action_concepts",
            "entity_terms",
            "group_terms",
            "relation_terms",
            "summary_terms",
        ):
            result.update(row[key])
    return result


def _profile_proof(
    advisory: Mapping[str, Any],
    candidate_map: Mapping[str, str],
    evidence_by_handle: Mapping[str, SpecialistHandleEvidence],
    advisory_receipt_sha256: str,
    question_terms: Sequence[str],
) -> SpecialistProof:
    handles = tuple(candidate_map.values())
    grounded = tuple(
        handle_id
        for handle_id in handles
        if "user" in _semantic_terms(evidence_by_handle[handle_id])
    )
    _require(
        bool(grounded),
        "profile advisory has no user-grounded specialist evidence",
    )
    payload = {
        "question_terms": list(_ordered_unique(question_terms, "profile question terms")),
        "user_grounded_handle_ids": list(grounded),
    }
    return SpecialistProof(
        SpecialistProofKind.PROFILE_PREFERENCE,
        require_text(advisory.get("mechanism_id"), "profile mechanism"),
        handles,
        advisory_receipt_sha256,
        _canonical_json(payload),
    )


def compile_specialist_validation_scope(
    *,
    specialist_advisories: Sequence[Mapping[str, Any]],
    declared_specialist_advisories_sha256: str,
    sealed_source_receipt_sha256: str,
    terminal_allowed_handle_ids: Sequence[str],
    handle_group_by_id: Mapping[str, str],
    validation_contract: Mapping[str, Any],
    prompt_envelope: SpecialistPromptEnvelope,
) -> SpecialistValidationScope:
    """Compile advisory-only proof universes from one sealed terminal prompt."""

    _require(
        type(prompt_envelope) is SpecialistPromptEnvelope,
        "specialist scope requires an exact prompt envelope",
    )
    advisories = tuple(_exact_dict(row, "specialist advisory") for row in specialist_advisories)
    _require(bool(advisories), "specialist scope requires advisories")
    advisory_sha = identity_sha256(list(advisories))
    _require(
        advisory_sha
        == require_sha256(
            declared_specialist_advisories_sha256,
            "declared specialist advisories",
        )
        == prompt_envelope.specialist_advisories_sha256,
        "specialist advisory seal differs from rendered prompt",
    )
    source_receipt = require_sha256(
        sealed_source_receipt_sha256, "specialist sealed source"
    )
    terminal = _ordered_unique(
        terminal_allowed_handle_ids, "terminal allowed handles"
    )
    _require(bool(terminal), "terminal handle population is empty")
    groups = dict(handle_group_by_id)
    _require(
        set(groups) == set(terminal)
        and all(
            _HANDLE_RE.fullmatch(key) is not None
            and type(value) is str
            and _GROUP_RE.fullmatch(value) is not None
            for key, value in groups.items()
        ),
        "terminal handle/group provenance changed",
    )
    contract = dict(validation_contract)
    by_handle = contract.get("by_handle")
    include_proposed = contract.get("include_proposed")
    question_terms = contract.get("question_terms")
    _require(
        contract.get("format") == VALIDATION_CONTRACT_FORMAT
        and type(by_handle) is dict
        and set(by_handle) == set(terminal)
        and type(include_proposed) is bool
        and type(question_terms) is list
        and all(type(value) is str and bool(value) for value in question_terms),
        "specialist validation contract changed",
    )
    assert_gold_blind(advisories, path="specialist_advisories")

    parsed_advisories: list[tuple[dict[str, Any], dict[str, str], str]] = []
    declared_handles: list[str] = []
    mechanism_ids: list[str] = []
    for advisory in advisories:
        mechanism_id = require_text(
            advisory.get("mechanism_id"), "specialist advisory mechanism"
        )
        _require(mechanism_id not in mechanism_ids, "specialist mechanisms repeat")
        mechanism_ids.append(mechanism_id)
        raw_map = _exact_dict(
            advisory.get("candidate_handle_map"), "specialist candidate handle map"
        )
        _require(bool(raw_map), "specialist candidate handle map is empty")
        candidate_map: dict[str, str] = {}
        for candidate_id, raw_handle in raw_map.items():
            require_sha256(candidate_id, "specialist candidate")
            handle_id = _handle(raw_handle, "specialist candidate handle changed")
            _require(
                handle_id in terminal and handle_id not in candidate_map.values(),
                "specialist candidate handle escaped or repeats",
            )
            candidate_map[candidate_id] = handle_id
            if handle_id not in declared_handles:
                declared_handles.append(handle_id)
        parsed_advisories.append(
            (advisory, candidate_map, identity_sha256(advisory))
        )

    evidence = tuple(
        _compile_handle_evidence(
            handle_id,
            groups[handle_id],
            _exact_dict(by_handle[handle_id], "specialist handle contract"),
            include_proposed=include_proposed,
        )
        for handle_id in declared_handles
    )
    evidence_by_handle = {row.handle_id: row for row in evidence}
    proofs: list[SpecialistProof] = []
    for source, candidate_map, receipt in parsed_advisories:
        enriched = {**source, "_validation_question_terms": list(question_terms)}
        structured = False
        if source.get("operand_groups") is not None:
            structured = True
            proofs.append(
                _numeric_proof(
                    source,
                    candidate_map,
                    evidence_by_handle,
                    groups,
                    receipt,
                )
            )
        if source.get("temporal_bundle") is not None:
            structured = True
            proofs.append(
                _temporal_proof(
                    enriched,
                    candidate_map,
                    evidence_by_handle,
                    receipt,
                )
            )
        if source.get("absence_certificate") is not None:
            structured = True
            proofs.append(
                _absence_proof(
                    source,
                    candidate_map,
                    evidence_by_handle,
                    receipt,
                )
            )
        if not structured and str(source.get("mechanism_id", "")).startswith(
            "profile_preference"
        ):
            proofs.append(
                _profile_proof(
                    source,
                    candidate_map,
                    evidence_by_handle,
                    receipt,
                    question_terms,
                )
            )
    return SpecialistValidationScope(
        terminal,
        tuple(declared_handles),
        evidence,
        tuple(proofs),
        advisory_sha,
        identity_sha256(contract),
        source_receipt,
        prompt_envelope.receipt_sha256,
        prompt_envelope.prompt_token_proxy,
    )


@dataclass(frozen=True, slots=True)
class ParsedSpecialistDecision:
    valid: bool
    decision: Literal["keep_parent", "replace", "invalid"]
    prediction: str
    used_handle_ids: tuple[str, ...]
    proof_kind: str
    proof_receipt_sha256: str | None
    validation_basis: str
    error_code: str
    scope_receipt_sha256: str
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.valid) is bool, "specialist decision validity changed")
        _require(
            self.decision in {"keep_parent", "replace", "invalid"}
            and type(self.prediction) is str,
            "specialist decision schema changed",
        )
        _ordered_unique(self.used_handle_ids, "specialist decision handles")
        require_text(self.proof_kind, "specialist decision proof kind")
        if self.proof_receipt_sha256 is not None:
            require_sha256(self.proof_receipt_sha256, "specialist decision proof")
        require_text(self.validation_basis, "specialist decision basis")
        require_text(self.error_code, "specialist decision error")
        require_sha256(self.scope_receipt_sha256, "specialist decision scope")
        _require(
            self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0,
            "specialist parser must remain provider-free and zero-state",
        )
        computed = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == computed, "specialist decision receipt changed")
        object.__setattr__(self, "receipt_sha256", computed)
        assert_gold_blind(self.projection(), path="specialist_scoped_decision")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "decision": self.decision,
            "error_code": self.error_code,
            "format": DECISION_FORMAT,
            "prediction": self.prediction,
            "prediction_sha256": quote_sha256(self.prediction),
            "proof_kind": self.proof_kind,
            "proof_receipt_sha256": self.proof_receipt_sha256,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "scope_receipt_sha256": self.scope_receipt_sha256,
            "used_handle_ids": list(self.used_handle_ids),
            "valid": self.valid,
            "validation_basis": self.validation_basis,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _invalid_decision(
    scope: SpecialistValidationScope,
    code: str,
) -> ParsedSpecialistDecision:
    return ParsedSpecialistDecision(
        False,
        "invalid",
        "",
        (),
        "none",
        None,
        "invalid",
        require_text(code, "specialist parse error"),
        scope.receipt_sha256,
    )


def _valid_decision(
    scope: SpecialistValidationScope,
    *,
    decision: Literal["keep_parent", "replace"],
    prediction: str,
    used: tuple[str, ...],
    proof: SpecialistProof | None,
    basis: str,
) -> ParsedSpecialistDecision:
    return ParsedSpecialistDecision(
        True,
        decision,
        prediction,
        used,
        "none" if proof is None else proof.kind.value,
        None if proof is None else proof.receipt_sha256,
        basis,
        "none",
        scope.receipt_sha256,
    )


def _parent_equivalent_replace_basis(
    prediction: str,
    parent_prediction: str,
) -> str | None:
    """Return the narrow basis for a replacement that preserves the parent.

    Provider completions occasionally select ``replace`` and attach advisory
    handles even though their prediction is the protected parent verbatim.
    Keeping that text does not consume those handles as evidence.  A second,
    deliberately tiny equivalence handles the typography-only difference
    between ASCII apostrophe and U+2019 RIGHT SINGLE QUOTATION MARK.  No case,
    whitespace, dash, quote, or general Unicode normalization is performed.
    """

    if prediction == parent_prediction:
        return "normalized_identical_replace"
    ascii_apostrophe_prediction = prediction.replace(
        "\N{RIGHT SINGLE QUOTATION MARK}", "'"
    )
    ascii_apostrophe_parent = parent_prediction.replace(
        "\N{RIGHT SINGLE QUOTATION MARK}", "'"
    )
    if ascii_apostrophe_prediction == ascii_apostrophe_parent:
        return "right_single_quote_equivalent_replace"
    return None


def _numeric_validation_error(
    prediction: str,
    used: tuple[str, ...],
    proof: SpecialistProof,
) -> str | None:
    payload = proof.payload
    for group in payload["groups"]:
        if not set(group["handle_ids"]) & set(used):
            return "specialist_numeric_group_incomplete"
    scalar = conservative_numeric_value(prediction)
    if scalar is None or abs(scalar - float(payload["computed_scalar"])) > 1e-9:
        return "specialist_numeric_reduction_disagreement"
    mentions = numeric_mentions(prediction)
    if len(mentions) != 1 or mentions[0].qualifier is not NumericQualifier.EXACT:
        return "specialist_numeric_prediction_unsafe"
    unit = payload.get("unit")
    if unit is not None and mentions[0].unit != unit:
        return "specialist_numeric_unit_loss"
    return None


def _anchor_positions(
    prediction: str,
    ordered_handles: Sequence[str],
    *,
    evidence_by_handle: Mapping[str, SpecialistHandleEvidence],
    question_terms: Sequence[str],
) -> tuple[int, ...] | None:
    prediction_terms = tuple(normalized_terms(prediction))
    question = set(question_terms)
    terms_by_handle = {
        handle_id: _semantic_terms(evidence_by_handle[handle_id])
        for handle_id in ordered_handles
    }
    positions: list[int] = []
    for handle_id in ordered_handles:
        others = set().union(
            *(
                terms
                for other, terms in terms_by_handle.items()
                if other != handle_id
            )
        ) if len(terms_by_handle) > 1 else set()
        own = terms_by_handle[handle_id] - question - _META_ANCHOR_TERMS
        discriminative = own - others or own
        matched = [
            index
            for index, term in enumerate(prediction_terms)
            if term in discriminative
        ]
        if not matched:
            return None
        positions.append(min(matched))
    return tuple(positions)


def _temporal_numeric_question_terms(question_terms: Sequence[str]) -> set[str]:
    """Return the sealed terms that explicitly request a numeric answer.

    ``question_terms`` has already passed through ``normalized_terms`` and
    therefore intentionally omits interrogative stop words such as ``how``.
    Requiring one of these answer-shape terms keeps incidental numerals in a
    temporal winner (model names, venue numbers, and selector dates) from
    silently turning a nonnumeric lookup into a numeric proof.
    """

    terms = {str(term).casefold() for term in question_terms}
    return terms & _NUMERIC_ANSWER_QUESTION_TERMS


def _temporal_evidence_numeric_contract(
    unit: str | None,
    *,
    question_terms: Sequence[str],
) -> tuple[NumericDimension, str | None] | None:
    """Map one sealed evidence unit to the canonical answer-value contract."""

    numeric_question_terms = _temporal_numeric_question_terms(question_terms)
    if not numeric_question_terms:
        return None
    terms = {str(term).casefold() for term in question_terms}
    if unit is None:
        dimension = (
            NumericDimension.COUNT
            if terms & _COUNT_QUESTION_TERMS
            else NumericDimension.GENERIC
        )
        return dimension, None
    folded = unit.strip().casefold()
    if folded in {"$", "dollar", "dollars", "usd", "us dollar", "us dollars"}:
        return NumericDimension.CURRENCY, "$"
    if folded in {"%", "percent", "percentage"}:
        return NumericDimension.PERCENTAGE, "%"
    singular = folded.rstrip("s")
    if singular in _DURATION_UNITS:
        return NumericDimension.DURATION, singular
    measure = _MEASURE_UNIT_ALIASES.get(folded)
    if measure is not None:
        return NumericDimension.MEASURE, measure
    # Unknown units cannot be checked for compatibility by numeric_mentions;
    # retaining lexical validation is safer than pretending they are unitless.
    return None


def _temporal_winner_numeric_validation_error(
    prediction: str,
    *,
    winner_evidence: SpecialistHandleEvidence,
    question_terms: Sequence[str],
) -> str | None:
    """Validate an exact, question-bound numeric value on a temporal winner.

    Relevance is deliberately bounded to semantic rows that (a) bind at least
    one required question slot, (b) expose a finite numeric value, (c) are not
    explicitly approximate/bounded, and (d) have a unit the shared numeric
    lexer can check.  The function never searches parent-union evidence and
    never infers a value from the predecessor.
    """

    expectations: set[tuple[float, NumericDimension, str | None]] = set()
    for row in winner_evidence.semantic_rows:
        value = row.get("numeric_value")
        supported_slots = row.get("supported_slot_ids")
        qualifier = row.get("numeric_qualifier", NumericQualifier.EXACT.value)
        role = row.get("numeric_role")
        if (
            type(value) not in {int, float}
            or not math.isfinite(value)
            or type(supported_slots) is not list
            or not supported_slots
            or qualifier != NumericQualifier.EXACT.value
            or role == "none"
        ):
            continue
        contract = _temporal_evidence_numeric_contract(
            row.get("unit"),
            question_terms=question_terms,
        )
        if contract is None:
            continue
        dimension, canonical_unit = contract
        expectations.add((float(value), dimension, canonical_unit))
    if not expectations:
        return None
    if len(expectations) != 1:
        return "specialist_temporal_winner_numeric_evidence_conflict"

    expected_value, expected_dimension, expected_unit = next(iter(expectations))
    mentions = numeric_mentions(
        prediction,
        expected_dimension=expected_dimension,
    )
    if not mentions:
        return "specialist_temporal_winner_numeric_entailment"
    # Every answer-dimension mention must agree.  This rejects a completion
    # that states the sealed winner and also leaks an older/conflicting state.
    if any(abs(mention.value - expected_value) > 1e-9 for mention in mentions):
        return "specialist_temporal_winner_numeric_disagreement"
    if any(mention.qualifier is not NumericQualifier.EXACT for mention in mentions):
        return "specialist_temporal_winner_numeric_prediction_unsafe"
    if any(mention.unit != expected_unit for mention in mentions):
        return "specialist_temporal_winner_numeric_unit_loss"
    return None


def _temporal_validation_error(
    prediction: str,
    used: tuple[str, ...],
    proof: SpecialistProof,
    evidence_by_handle: Mapping[str, SpecialistHandleEvidence],
) -> str | None:
    payload = proof.payload
    ordered = tuple(payload["ordered_handle_ids"])
    question_terms = tuple(payload["question_terms"])
    if proof.kind is SpecialistProofKind.TEMPORAL_ORDER:
        if used != ordered:
            return "specialist_temporal_order_scope"
        positions = _anchor_positions(
            prediction,
            ordered,
            evidence_by_handle=evidence_by_handle,
            question_terms=question_terms,
        )
        if positions is None or any(
            left >= right for left, right in zip(positions, positions[1:])
        ):
            return "specialist_temporal_order_entailment"
        return None
    winner = payload["winner_handle_id"]
    # The predecessor is an internal deterministic comparator, never answer
    # support.  Requiring the singleton also prevents a model from rebuilding
    # a broader timeline through citations that the prompt forbids.
    if used != (winner,):
        return "specialist_temporal_role_scope"
    positions = _anchor_positions(
        prediction,
        (winner,),
        evidence_by_handle=evidence_by_handle,
        question_terms=question_terms,
    )
    if positions is None:
        return "specialist_temporal_winner_entailment"
    return _temporal_winner_numeric_validation_error(
        prediction,
        winner_evidence=evidence_by_handle[winner],
        question_terms=question_terms,
    )


def _interval_validation_error(
    prediction: str,
    used: tuple[str, ...],
    proof: SpecialistProof,
) -> str | None:
    payload = proof.payload
    interval = _exact_dict(payload.get("interval"), "temporal interval proof")
    winner = payload["winner_handle_id"]
    # Interval arithmetic is bound to winner -> query_time.  A predecessor may
    # have helped the specialist select the winner, but cannot be cited as a
    # second answer operand.
    if used != (winner,):
        return "specialist_temporal_interval_scope"
    mentions = numeric_mentions(
        prediction,
        expected_dimension=NumericDimension.DURATION,
    )
    if (
        len(mentions) != 1
        or abs(mentions[0].value - float(interval["computed_scalar"])) > 1e-9
    ):
        return "specialist_temporal_interval_disagreement"
    if (
        len(mentions) != 1
        or mentions[0].qualifier is not NumericQualifier.EXACT
        or mentions[0].unit != interval["unit"]
    ):
        return "specialist_temporal_interval_unit"
    return None


def _profile_validation_error(
    prediction: str,
    used: tuple[str, ...],
    proof: SpecialistProof,
    evidence_by_handle: Mapping[str, SpecialistHandleEvidence],
) -> str | None:
    payload = proof.payload
    grounded = set(payload["user_grounded_handle_ids"])
    if not set(used) & grounded:
        return "specialist_profile_user_grounding"
    question_terms = set(payload["question_terms"])
    meaningful = set().union(
        *(_semantic_terms(evidence_by_handle[handle_id]) for handle_id in used)
    ) - question_terms - _META_ANCHOR_TERMS
    prediction_terms = set(normalized_terms(prediction)) - question_terms
    if not meaningful or not meaningful & prediction_terms:
        return "specialist_profile_text_entailment"
    return None


def _absence_validation_error(
    prediction: str,
    used: tuple[str, ...],
    proof: SpecialistProof,
) -> str | None:
    payload = proof.payload
    if set(used) != set(proof.handle_ids):
        return "specialist_absence_support_scope"
    if _INSUFFICIENCY_RE.search(prediction) is None:
        return "specialist_absence_not_expressed"
    missing_terms = set().union(
        *(
            set((*slot["exact_entity_terms"], *normalized_terms(slot["slot_label"])))
            for slot in payload["slots"]
            if slot["explicit_numeric_operand_missing"]
        )
    )
    prediction_terms = set(normalized_terms(prediction))
    missing_slot_named = bool(missing_terms & prediction_terms)
    if not missing_slot_named:
        # ``normalized_terms`` intentionally retains an internal ASCII hyphen
        # (for example ``chili-pepper``) as one lexical token.  Treat that token
        # as the corresponding spaced missing-slot phrase only when *every*
        # normalized component is already sealed in the missing-slot terms.
        # This stays fail-closed for lookalikes such as ``chili-pepperoni``.
        missing_slot_named = any(
            len(parts) >= 2 and set(parts) <= missing_terms
            for term in prediction_terms
            if "-" in term
            for parts in (normalized_terms(term.replace("-", " ")),)
        )
    if not missing_slot_named:
        return "specialist_absence_missing_slot_not_named"
    allowed_values = tuple(float(value) for value in payload["allowed_numeric_values"])
    for mention in numeric_mentions(prediction):
        if not any(abs(mention.value - value) <= 1e-9 for value in allowed_values):
            return "specialist_absence_unsupported_numeric_value"
    return None


def parse_specialist_scoped_completion(
    completion: str,
    *,
    parent_prediction: str,
    scope: SpecialistValidationScope,
) -> ParsedSpecialistDecision:
    """Parse strict three-field JSON against advisory-local proof scopes only."""

    if type(completion) is not str:
        raise TypeError("specialist completion must be exact text")
    require_text(parent_prediction, "specialist parent prediction")
    _require(
        type(scope) is SpecialistValidationScope,
        "specialist parser requires an exact compiled scope",
    )
    try:
        raw = json.loads(
            completion,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (json.JSONDecodeError, ValueError):
        return _invalid_decision(scope, "invalid_json")
    if type(raw) is not dict or set(raw) != {
        "decision",
        "prediction",
        "used_handle_ids",
    }:
        return _invalid_decision(scope, "root_schema")
    decision = raw["decision"]
    prediction = raw["prediction"]
    raw_used = raw["used_handle_ids"]
    if (
        type(decision) is not str
        or type(prediction) is not str
        or type(raw_used) is not list
        or any(type(value) is not str for value in raw_used)
        or len(raw_used) != len(set(raw_used))
    ):
        return _invalid_decision(scope, "value_schema")
    used = tuple(raw_used)
    if decision == "keep_parent":
        if prediction != parent_prediction or used:
            return _invalid_decision(scope, "keep_parent_contract")
        return _valid_decision(
            scope,
            decision="keep_parent",
            prediction=parent_prediction,
            used=(),
            proof=None,
            basis="keep_parent_contract",
        )
    if decision != "replace":
        return _invalid_decision(scope, "decision")
    if not prediction or prediction.strip() != prediction:
        return _invalid_decision(scope, "replace_contract")
    if not set(used) <= set(scope.terminal_allowed_handle_ids):
        return _invalid_decision(scope, "unknown_handle")
    if not set(used) <= set(scope.specialist_declared_handle_ids):
        return _invalid_decision(scope, "specialist_scope_escape")
    parent_equivalent_basis = _parent_equivalent_replace_basis(
        prediction,
        parent_prediction,
    )
    if parent_equivalent_basis is not None and not used:
        return _valid_decision(
            scope,
            decision="keep_parent",
            prediction=parent_prediction,
            used=(),
            proof=None,
            basis=parent_equivalent_basis,
        )
    if not used:
        return _invalid_decision(scope, "replace_contract")

    evidence = scope.evidence_by_handle
    applicable = tuple(
        proof for proof in scope.proofs if set(used) <= set(proof.handle_ids)
    )
    if not applicable:
        return _invalid_decision(scope, "cross_specialist_proof_scope")
    accepted: list[tuple[SpecialistProof, str]] = []
    errors: list[str] = []
    for proof in applicable:
        if proof.kind is SpecialistProofKind.NUMERIC_OPERAND_GROUPS:
            error = _numeric_validation_error(prediction, used, proof)
            basis = "specialist_numeric_operand_group_reduction"
        elif proof.kind is SpecialistProofKind.TEMPORAL_INTERVAL:
            error = _interval_validation_error(prediction, used, proof)
            basis = "specialist_temporal_interval_agreement"
        elif proof.kind in {
            SpecialistProofKind.TEMPORAL_ORDER,
            SpecialistProofKind.TEMPORAL_RELATIVE,
        }:
            error = _temporal_validation_error(prediction, used, proof, evidence)
            basis = "specialist_temporal_bundle_agreement"
        elif proof.kind is SpecialistProofKind.PROFILE_PREFERENCE:
            error = _profile_validation_error(prediction, used, proof, evidence)
            basis = "specialist_profile_cluster_grounding"
        else:
            error = _absence_validation_error(prediction, used, proof)
            basis = "specialist_scoped_absence_certificate"
        if error is None:
            accepted.append((proof, basis))
        else:
            errors.append(error)
    if (
        not accepted
        and parent_equivalent_basis is not None
        and applicable
        and all(
            proof.kind
            in {
                SpecialistProofKind.TEMPORAL_ORDER,
                SpecialistProofKind.TEMPORAL_RELATIVE,
            }
            for proof in applicable
        )
    ):
        return _valid_decision(
            scope,
            decision="keep_parent",
            prediction=parent_prediction,
            used=(),
            proof=None,
            basis=parent_equivalent_basis,
        )
    if len(accepted) != 1:
        return _invalid_decision(
            scope,
            errors[0] if not accepted and errors else "ambiguous_specialist_proof",
        )
    proof, basis = accepted[0]
    return _valid_decision(
        scope,
        decision="replace",
        prediction=prediction,
        used=used,
        proof=proof,
        basis=basis,
    )


__all__ = [
    "DECISION_FORMAT",
    "FORMAT",
    "HARD_COMPLETE_CHAT_TOKEN_CAP",
    "MAX_CHAT_PROMPT_TOKENS",
    "OUTPUT_TOKEN_RESERVE",
    "PROMPT_FORMAT",
    "ParsedSpecialistDecision",
    "SCOPE_FORMAT",
    "SPECIALIST_SYSTEM_PROMPT",
    "SpecialistHandleEvidence",
    "SpecialistProof",
    "SpecialistProofKind",
    "SpecialistPromptEnvelope",
    "SpecialistScopedCompletionError",
    "SpecialistValidationScope",
    "compile_specialist_validation_scope",
    "parse_specialist_scoped_completion",
    "render_specialist_scoped_prompt",
]
