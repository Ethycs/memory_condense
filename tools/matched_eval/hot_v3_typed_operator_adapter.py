"""Provider-free bridge from a sealed hot-v3 arm to typed operators.

The hot path already owns retrieval, source-preserving composition, ranked
prefix packing, and the responder prompt.  This module does not repeat any of
that work.  It authenticates one materialized A3 arm, assigns opaque H/G
identities to the *packed* rows only, and projects those citations into the
existing typed-operator contracts.  The resulting frontier is deliberately
bounded: a ranked retrieval packet is useful evidence, not proof that the
unseen memory contains no other operands or members.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval._retrieval_qa_prompt import (
    RESPONDER_OUTPUT_TOKEN_RESERVE,
    build_qa_prompt,
)

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .typed_numeric_semantics import single_numeric_mention
from .typed_operator_adapter import (
    DEFAULT_OUTPUT_TOKEN_RESERVE,
    ConflictPolicy,
    EvidenceFrontierReceipt,
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    NumericRole,
    OpaqueEvidenceHandle,
    ProviderPayloadMode,
    ProvenanceGrade,
    RejectedTypedItem,
    TypedEvidenceItem,
    TypedEvidencePacket,
    build_frontier_receipt,
    build_typed_evidence_packet,
    compact_evidence_content_projection,
    parse_typed_items,
)
from .typed_operator_spec import TypedOperatorSpec, compile_typed_operator_spec


FORMAT = "memory-condense-hot-v3-typed-operator-adapter-v1"
LOCAL_INVENTORY_FORMAT = "memory-condense-hot-v3-local-typed-inventory-v1"
LOCAL_OPERATOR_EVIDENCE_FORMAT = (
    "memory-condense-hot-v3-local-operator-evidence-v1"
)
PROVIDER_PACKET_FORMAT = "memory-condense-source-seed-hybrid-provider-packet-v1"
MAX_CONTEXT_TOKENS = 7_000
MAX_PROMPT_TOKENS = 8_000

_ARM_KEYS = frozenset(
    {
        "selected_evidence",
        "packed_evidence",
        "selected_chunk_ids",
        "packed_chunk_ids",
        "dropped_chunk_ids",
        "context_token_proxy",
        "prompt_token_proxy",
        "prompt_workspace_token_proxy",
        "provider_messages",
        "provider_payload_sha256",
        "provider_payload_utf8_bytes",
        "raw_evidence_only",
    }
)
_RAW_EVIDENCE_KEYS = frozenset(
    {
        "evidence_id",
        "chunk_id",
        "turn_id",
        "source_id",
        "role",
        "created_at",
        "route",
        "score",
        "raw_text",
        "raw_text_sha256",
        "rendered_text",
        "rendered_text_sha256",
    }
)
_PROJECTION_EVIDENCE_KEYS = (_RAW_EVIDENCE_KEYS - {"turn_id"}) | frozenset(
    {
        "assertion_fact_receipt_sha256",
        "source_handle",
        "created_at_semantics",
    }
)
_PROJECTION_KEYS = frozenset(
    {
        "assertion_fact_receipt_sha256",
        "source_handle",
        "created_at_semantics",
    }
)
_PROJECTION_ROUTE = "activated_assertion_projection"
_PROJECTION_DATE_SEMANTICS = "source_metadata_only_not_event_time"
_SOURCE_HANDLE_RE = re.compile(r"^G[0-9]{6}$")
_PROVIDER_PACKET_KEYS = frozenset(
    {
        "format",
        "route",
        "selected_count",
        "selected_chunk_ids_sha256",
        "packed_count",
        "packed_chunk_ids_sha256",
        "dropped_count",
        "dropped_chunk_ids_sha256",
        "context_token_proxy",
        "prompt_token_proxy",
        "prompt_workspace_token_proxy",
        "provider_payload_sha256",
        "provider_payload_utf8_bytes",
        "raw_evidence_only",
        "parent_receipts",
        "receipt_sha256",
    }
)
_PROVIDER_PARENT_KEYS = frozenset(
    {
        "v2_selection_sha256",
        "v2_question_sha256",
        "v2_projection_packet_receipt_sha256",
        "v7_fallback_reference_receipt_sha256",
        "projection_prefix_receipt_sha256",
        "hybrid_receipt_sha256",
    }
)


def _require(ok: object, message: str) -> None:
    if not ok:
        raise MatchedEvalContractError(message)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _ordered_ids(value: object, label: str) -> list[str]:
    _require(type(value) is list, f"{label} must be an exact list")
    result = list(value)
    _require(
        all(type(row) is str and row and row.strip() == row for row in result),
        f"{label} must contain exact non-empty text",
    )
    _require(len(result) == len(set(result)), f"{label} must be ordered unique")
    return result


def _validate_evidence_row(value: object, *, index: int) -> dict[str, Any]:
    label = f"hot-v3 evidence {index}"
    _require(type(value) is dict, f"{label} must be an exact object")
    row = dict(value)
    keys = frozenset(row)
    projected = keys in {
        _PROJECTION_EVIDENCE_KEYS,
        _RAW_EVIDENCE_KEYS | _PROJECTION_KEYS,
    }
    _require(
        keys == _RAW_EVIDENCE_KEYS or projected,
        f"{label} changed schema",
    )
    for key in ("evidence_id", "chunk_id", "source_id"):
        require_text(row[key], f"{label} {key}")
    if "turn_id" in row:
        require_text(row["turn_id"], f"{label} turn_id")
    _require(
        row["evidence_id"] == row["chunk_id"],
        f"{label} evidence/chunk identity changed",
    )
    _require(
        type(row["role"]) is str
        and row["role"] in {"user", "assistant", "system"},
        f"{label} role changed",
    )
    _require(
        type(row["created_at"]) is str,
        f"{label} created_at must be exact text",
    )
    _require(type(row["route"]) is str, f"{label} route must be exact text")
    score = row["score"]
    if not projected and type(score) is str:
        try:
            numeric_score = float(score)
        except ValueError as exc:
            raise MatchedEvalContractError(
                f"{label} score must be a canonical finite decimal"
            ) from exc
        if numeric_score == 0.0:
            numeric_score = 0.0
        score_valid = (
            math.isfinite(numeric_score)
            and score == format(numeric_score, ".9g")
        )
    elif projected:
        # Activated-assertion projection scores are exact JSON floats; raw
        # retrieval routes use the canonical nine-significant-digit string.
        score_valid = type(score) is float and math.isfinite(score)
    else:
        score_valid = False
    _require(score_valid, f"{label} score must preserve its route schema")
    raw_text = row["raw_text"]
    rendered_text = row["rendered_text"]
    require_text(raw_text, f"{label} raw text")
    require_text(rendered_text, f"{label} rendered text")
    _require(
        row["raw_text_sha256"] == quote_sha256(raw_text),
        f"{label} raw text digest changed",
    )
    _require(
        row["rendered_text_sha256"] == quote_sha256(rendered_text),
        f"{label} rendered text digest changed",
    )
    _require(rendered_text.endswith(raw_text), f"{label} rendered/raw binding changed")
    if projected:
        _require(row["route"] == _PROJECTION_ROUTE, f"{label} projection route changed")
        _require(
            row["created_at_semantics"] == _PROJECTION_DATE_SEMANTICS,
            f"{label} projection date semantics changed",
        )
        require_sha256(
            row["assertion_fact_receipt_sha256"],
            f"{label} assertion receipt",
        )
        source_handle = row["source_handle"]
        _require(
            type(source_handle) is str
            and _SOURCE_HANDLE_RE.fullmatch(source_handle) is not None,
            f"{label} projection source handle changed",
        )
    else:
        _require(
            row["route"] != _PROJECTION_ROUTE,
            f"{label} projection metadata is missing",
        )
    return row


def _validate_arm(
    dated_question: str,
    arm: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], int, int, str]:
    require_text(dated_question, "hot-v3 dated question")
    _require(isinstance(arm, Mapping), "hot-v3 arm must be a mapping")
    _require(set(arm) == _ARM_KEYS, "hot-v3 arm changed schema")
    assert_gold_blind(arm, path="hot_v3_arm")
    _require(arm["raw_evidence_only"] is True, "hot-v3 arm is not raw-evidence-only")

    selected_raw = arm["selected_evidence"]
    packed_raw = arm["packed_evidence"]
    _require(
        type(selected_raw) is list and type(packed_raw) is list,
        "hot-v3 evidence populations must be exact lists",
    )
    selected = tuple(
        _validate_evidence_row(row, index=index)
        for index, row in enumerate(selected_raw)
    )
    packed = tuple(
        _validate_evidence_row(row, index=index)
        for index, row in enumerate(packed_raw)
    )
    selected_ids = _ordered_ids(arm["selected_chunk_ids"], "selected chunk IDs")
    packed_ids = _ordered_ids(arm["packed_chunk_ids"], "packed chunk IDs")
    dropped_ids = _ordered_ids(arm["dropped_chunk_ids"], "dropped chunk IDs")
    _require(
        selected_ids == [row["chunk_id"] for row in selected],
        "hot-v3 selected evidence lost its ID binding",
    )
    _require(
        packed_ids == [row["chunk_id"] for row in packed],
        "hot-v3 packed evidence lost its ID binding",
    )
    _require(
        packed == selected[: len(packed)],
        "hot-v3 packed evidence is not the selected ranked prefix",
    )
    _require(
        dropped_ids == selected_ids[len(packed) :],
        "hot-v3 dropped IDs are not the selected tail",
    )

    # Projection-local source handles are authenticated as a consistent
    # relation, but never reused as the new typed packet's opaque G namespace.
    source_by_projection_handle: dict[str, str] = {}
    projection_handle_by_source: dict[str, str] = {}
    for row in selected:
        if "source_handle" not in row:
            continue
        source_id = row["source_id"]
        source_handle = row["source_handle"]
        previous_source = source_by_projection_handle.setdefault(source_handle, source_id)
        previous_handle = projection_handle_by_source.setdefault(source_id, source_handle)
        _require(
            previous_source == source_id and previous_handle == source_handle,
            "hot-v3 projection source-handle relation changed",
        )

    rendered = [row["rendered_text"] for row in packed]
    expected_messages = build_qa_prompt(dated_question, rendered)
    _require(
        arm["provider_messages"] == expected_messages,
        "hot-v3 provider messages changed question or packed evidence",
    )
    context_tokens = (
        0
        if not rendered
        else count_tokens(
            "\n".join(f"[{index}] {text}" for index, text in enumerate(rendered, 1))
        )
    )
    prompt_tokens = count_chat_prompt_token_proxy(expected_messages)
    workspace_tokens = prompt_tokens + RESPONDER_OUTPUT_TOKEN_RESERVE
    for key in (
        "context_token_proxy",
        "prompt_token_proxy",
        "prompt_workspace_token_proxy",
        "provider_payload_utf8_bytes",
    ):
        _require(
            type(arm[key]) is int and arm[key] >= 0,
            f"hot-v3 {key} must be a non-negative exact integer",
        )
    _require(
        arm["context_token_proxy"] == context_tokens
        and arm["prompt_token_proxy"] == prompt_tokens
        and arm["prompt_workspace_token_proxy"] == workspace_tokens,
        "hot-v3 token accounting changed",
    )
    _require(
        context_tokens <= MAX_CONTEXT_TOKENS
        and workspace_tokens <= MAX_PROMPT_TOKENS,
        "hot-v3 arm exceeds its provider envelope",
    )
    serialized = _canonical_json_bytes({"messages": expected_messages})
    provider_payload_sha256 = arm["provider_payload_sha256"]
    require_sha256(provider_payload_sha256, "hot-v3 provider payload")
    _require(
        provider_payload_sha256 == hashlib.sha256(serialized).hexdigest()
        and arm["provider_payload_utf8_bytes"] == len(serialized),
        "hot-v3 provider payload bytes changed",
    )
    return packed, len(selected), len(dropped_ids), provider_payload_sha256


def _validate_provider_packet(
    arm: Mapping[str, Any],
    provider_packet: Mapping[str, Any],
) -> str:
    _require(
        type(provider_packet) is dict,
        "hot-v3 provider-packet receipt must be an exact object",
    )
    packet = dict(provider_packet)
    _require(
        frozenset(packet) == _PROVIDER_PACKET_KEYS,
        "hot-v3 provider-packet receipt changed schema",
    )
    assert_gold_blind(packet, path="hot_v3_provider_packet_receipt")
    receipt_sha256 = packet.pop("receipt_sha256")
    require_sha256(receipt_sha256, "hot-v3 provider packet")
    _require(
        receipt_sha256 == identity_sha256(packet),
        "hot-v3 provider-packet receipt changed",
    )
    _require(
        packet["format"] == PROVIDER_PACKET_FORMAT
        and packet["route"] in {"hybrid", "raw_fallback"}
        and packet["raw_evidence_only"] is True,
        "hot-v3 provider-packet policy changed",
    )
    for key in (
        "selected_count",
        "packed_count",
        "dropped_count",
        "context_token_proxy",
        "prompt_token_proxy",
        "prompt_workspace_token_proxy",
        "provider_payload_utf8_bytes",
    ):
        _require(
            type(packet[key]) is int and packet[key] >= 0,
            f"hot-v3 provider-packet {key} changed type",
        )
    parent_receipts = packet["parent_receipts"]
    _require(
        type(parent_receipts) is dict
        and frozenset(parent_receipts) == _PROVIDER_PARENT_KEYS,
        "hot-v3 provider-packet parents changed schema",
    )
    for key, value in parent_receipts.items():
        require_sha256(value, f"hot-v3 provider-packet parent {key}")

    selected_ids = arm["selected_chunk_ids"]
    packed_ids = arm["packed_chunk_ids"]
    dropped_ids = arm["dropped_chunk_ids"]
    expected = {
        "selected_count": len(selected_ids),
        "selected_chunk_ids_sha256": identity_sha256(selected_ids),
        "packed_count": len(packed_ids),
        "packed_chunk_ids_sha256": identity_sha256(packed_ids),
        "dropped_count": len(dropped_ids),
        "dropped_chunk_ids_sha256": identity_sha256(dropped_ids),
        "context_token_proxy": arm["context_token_proxy"],
        "prompt_token_proxy": arm["prompt_token_proxy"],
        "prompt_workspace_token_proxy": arm["prompt_workspace_token_proxy"],
        "provider_payload_sha256": arm["provider_payload_sha256"],
        "provider_payload_utf8_bytes": arm["provider_payload_utf8_bytes"],
    }
    _require(
        all(packet[key] == value for key, value in expected.items()),
        "hot-v3 provider-packet receipt does not bind the arm",
    )
    return receipt_sha256


def _numeric_role(text: str, numeric: object) -> str:
    if numeric is None:
        return NumericRole.NONE.value
    if re.search(r"\b(?:baseline|started|initial(?:ly)?)\b", text, re.I):
        return NumericRole.BASELINE.value
    if re.search(r"\b(?:ended|ending|current|now|reached|grew to)\b", text, re.I):
        return NumericRole.END.value
    if re.search(r"\b(?:increase|gain|grew by|decrease|loss|delta)\b", text, re.I):
        return NumericRole.DELTA.value
    return NumericRole.OPERAND.value


def _raw_typed_item(
    row: Mapping[str, Any],
    *,
    handle_id: str,
    operator_spec: TypedOperatorSpec,
    dated_question: str,
) -> dict[str, Any]:
    text = str(row["raw_text"])
    mention = single_numeric_mention(
        text,
        operator_spec=operator_spec,
        question=dated_question,
    )
    created_at = str(row["created_at"])
    projected = row.get("created_at_semantics") == _PROJECTION_DATE_SEMANTICS
    date_basis = (
        _PROJECTION_DATE_SEMANTICS
        if projected
        else "source_created_at"
        if created_at
        else "none"
    )
    raw: dict[str, Any] = {
        "handle_ids": [handle_id],
        # Role-incompatible rows remain locally inspectable but cannot satisfy
        # proof slots or operators.  System rows are excluded before this
        # function because their text must not enter either execution plane.
        "included": (
            operator_spec.required_evidence_role is None
            or row["role"] == operator_spec.required_evidence_role
        ),
        "kind": "operand" if mention is not None else "direct",
        "numeric_role": _numeric_role(text, mention),
        "relation": f"authored_by_{row['role']};date_basis={date_basis}",
        "summary": text,
        # The cited value is explicit even when its separate timestamp anchor
        # is metadata.  Timestamp provenance is carried only by date_basis.
        "value_authority": "explicit",
    }
    # ``created_at`` is source metadata for both raw and projected rows.  It is
    # authenticated by the local binding/row receipt, but it is never promoted
    # to TypedEvidenceItem.date (which means executable event time).
    if mention is not None:
        raw["numeric_qualifier"] = mention.qualifier.value
        raw["numeric_value"] = mention.value
        if mention.unit is not None:
            raw["unit"] = mention.unit
    return raw


@dataclass(frozen=True, slots=True)
class HotV3LocalTypedEvidenceInventory:
    """Uncapped, local-only typed view of every packed non-system row.

    This object deliberately has no ``provider_projection`` method.  Its
    operator input can exceed 8k and is valid only for deterministic local CPU
    operators.  The separately constructed :class:`TypedEvidencePacket`
    remains the sole provider-eligible, hard-capped projection.
    """

    dated_question: str
    operator_spec: TypedOperatorSpec
    handles: tuple[OpaqueEvidenceHandle, ...]
    local_bindings: tuple[EvidenceHandleBinding, ...]
    items: tuple[TypedEvidenceItem, ...]
    rejected_items: tuple[RejectedTypedItem, ...]
    frontier: EvidenceFrontierReceipt
    conflict_policy: ConflictPolicy
    selection_sha256: str
    provider_packet_receipt_sha256: str
    packed_chunk_ids_sha256: str
    packed_row_sha256s: tuple[str, ...]
    handle_roles: tuple[str, ...]
    temporal_anchors_by_handle: tuple[str | None, ...]
    system_excluded_handle_ids: tuple[str, ...]
    role_filtered_handle_ids: tuple[str, ...]
    local_operator_token_proxy: int = 0
    provider_use_forbidden: Literal[True] = True
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.dated_question, "hot-v3 local dated question")
        _require(
            type(self.operator_spec) is TypedOperatorSpec,
            "hot-v3 local operator spec changed type",
        )
        _require(
            type(self.handles) is tuple
            and all(type(row) is OpaqueEvidenceHandle for row in self.handles),
            "hot-v3 local handles changed type",
        )
        _require(
            type(self.local_bindings) is tuple
            and all(
                type(row) is EvidenceHandleBinding for row in self.local_bindings
            ),
            "hot-v3 local bindings changed type",
        )
        _require(
            type(self.items) is tuple
            and all(type(row) is TypedEvidenceItem for row in self.items),
            "hot-v3 local items changed type",
        )
        _require(
            type(self.rejected_items) is tuple
            and all(type(row) is RejectedTypedItem for row in self.rejected_items),
            "hot-v3 local rejections changed type",
        )
        _require(
            type(self.frontier) is EvidenceFrontierReceipt
            and self.frontier.mode is FrontierMode.BOUNDED
            and self.frontier.closed is False,
            "hot-v3 local inventory must remain a bounded open frontier",
        )
        _require(
            type(self.conflict_policy) is ConflictPolicy,
            "hot-v3 local conflict policy changed type",
        )
        for value, label in (
            (self.selection_sha256, "hot-v3 local selection"),
            (self.provider_packet_receipt_sha256, "hot-v3 local provider packet"),
            (self.packed_chunk_ids_sha256, "hot-v3 local packed IDs"),
        ):
            require_sha256(value, label)
        _require(
            type(self.packed_row_sha256s) is tuple
            and len(self.packed_row_sha256s) == len(self.handles),
            "hot-v3 local packed-row inventory changed",
        )
        for value in self.packed_row_sha256s:
            require_sha256(value, "hot-v3 local packed row")
        _require(
            type(self.handle_roles) is tuple
            and len(self.handle_roles) == len(self.handles)
            and all(row in {"user", "assistant", "system"} for row in self.handle_roles),
            "hot-v3 local role inventory changed",
        )
        _require(
            type(self.temporal_anchors_by_handle) is tuple
            and len(self.temporal_anchors_by_handle) == len(self.handles)
            and all(
                row is None
                or (
                    type(row) is str
                    and bool(row)
                    and row.strip() == row
                )
                for row in self.temporal_anchors_by_handle
            ),
            "hot-v3 local temporal-anchor inventory changed",
        )
        handle_ids = tuple(row.handle_id for row in self.handles)
        binding_ids = tuple(row.handle_id for row in self.local_bindings)
        _require(
            handle_ids == binding_ids == self.frontier.available_handle_ids,
            "hot-v3 local handles lost their ordered bindings",
        )
        _require(
            all(
                handle.binding_receipt_sha256 == binding.receipt_sha256
                and binding.sealed_artifact_sha256 == self.selection_sha256
                and binding.parent_receipt_sha256
                == self.provider_packet_receipt_sha256
                for handle, binding in zip(
                    self.handles, self.local_bindings, strict=True
                )
            ),
            "hot-v3 local binding lineage changed",
        )
        system_handles = tuple(
            handle
            for handle, role in zip(handle_ids, self.handle_roles, strict=True)
            if role == "system"
        )
        _require(
            self.system_excluded_handle_ids == system_handles,
            "hot-v3 local system exclusion changed",
        )
        _require(
            all(
                anchor is None
                for anchor, role in zip(
                    self.temporal_anchors_by_handle,
                    self.handle_roles,
                    strict=True,
                )
                if role == "system"
            ),
            "hot-v3 system metadata entered the local temporal anchors",
        )
        required_role = self.operator_spec.required_evidence_role
        expected_role_filtered = tuple(
            handle
            for handle, role in zip(handle_ids, self.handle_roles, strict=True)
            if role != "system" and required_role is not None and role != required_role
        )
        _require(
            self.role_filtered_handle_ids == expected_role_filtered,
            "hot-v3 local required-role filter changed",
        )
        represented = {
            handle for item in self.items for handle in item.handle_ids
        }
        expected_represented = set(handle_ids) - set(system_handles)
        _require(
            represented == expected_represented
            and set(self.frontier.represented_handle_ids) == expected_represented
            and set(self.frontier.omitted_handle_ids) == set(system_handles)
            and not self.rejected_items,
            "hot-v3 local inventory did not examine every packed non-system row",
        )
        role_by_handle = dict(zip(handle_ids, self.handle_roles, strict=True))
        role_filtered = set(self.role_filtered_handle_ids)
        for item in self.items:
            _require(
                len(item.handle_ids) == 1,
                "hot-v3 local item must retain one packed-row handle",
            )
            handle = item.handle_ids[0]
            role = role_by_handle[handle]
            _require(
                item.relation is not None
                and item.relation.startswith(f"authored_by_{role};")
                and item.included == (handle not in role_filtered),
                "hot-v3 local item role semantics changed",
            )
            _require(
                item.date is None,
                "hot-v3 source metadata was promoted to executable event time",
            )
        _require(
            self.provider_use_forbidden is True
            and self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "hot-v3 local inventory crossed its execution boundary",
        )
        operator_tokens = count_tokens(
            json.dumps(
                self.operator_input(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if self.local_operator_token_proxy:
            _require(
                self.local_operator_token_proxy == operator_tokens,
                "hot-v3 local operator token proxy changed",
            )
        object.__setattr__(self, "local_operator_token_proxy", operator_tokens)
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "hot-v3 local inventory changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_v3_local_typed_inventory")

    def operator_projection(self) -> dict[str, Any]:
        """Return full local operator evidence; never a provider payload."""

        system_handles = set(self.system_excluded_handle_ids)
        execution_bindings = tuple(
            row
            for row in self.local_bindings
            if row.handle_id not in system_handles
        )
        execution_frontier = build_frontier_receipt(
            self.operator_spec,
            execution_bindings,
            self.items,
            self.rejected_items,
            mode=FrontierMode.BOUNDED,
            truncated=self.frontier.truncated,
            conflict_policy=self.conflict_policy,
        )
        content = compact_evidence_content_projection(
            self.items,
            execution_bindings,
        )
        anchor_by_handle = dict(
            zip(
                (row.handle_id for row in self.handles),
                self.temporal_anchors_by_handle,
                strict=True,
            )
        )
        for item in content["items"]:
            handles = item["handle_ids"]
            _require(
                type(handles) is list and len(handles) == 1,
                "hot-v3 local temporal anchor lost its row handle",
            )
            anchor = anchor_by_handle[str(handles[0])]
            if anchor is not None:
                # This is an authenticated reference clock for explicit
                # relative language, never an executable event timestamp.
                item["temporal_anchor"] = anchor
        # System handles remain bound by ``projection()`` but are absent from
        # this execution view, along with their text.  Every non-system handle
        # is represented by exactly one item and therefore survives here.
        value = {
            "conflict_policy": self.conflict_policy.value,
            "format": LOCAL_OPERATOR_EVIDENCE_FORMAT,
            "frontier": execution_frontier.projection(),
            "gold_loaded": False,
            "local_only": True,
            "operator_spec": self.operator_spec.projection(),
            "provider_prompt_count": 0,
            "provider_use_forbidden": True,
            "retained_transformer_token_state_bytes": 0,
            **content,
        }
        assert_gold_blind(value, path="hot_v3_local_operator_evidence")
        return value

    def operator_input(self) -> dict[str, Any]:
        """Return an operator-first-compatible, explicitly local-only input."""

        value = {
            "dated_question": self.dated_question,
            "typed_evidence": self.operator_projection(),
        }
        assert_gold_blind(value, path="hot_v3_local_operator_input")
        return value

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "binding_receipt_sha256s": [
                row.receipt_sha256 for row in self.local_bindings
            ],
            "dated_question_sha256": quote_sha256(self.dated_question),
            "format": LOCAL_INVENTORY_FORMAT,
            "frontier": self.frontier.projection(),
            "gold_loaded": False,
            "handle_roles": list(self.handle_roles),
            "item_receipt_sha256s": [row.receipt_sha256 for row in self.items],
            "local_only": True,
            "local_operator_input_sha256": identity_sha256(self.operator_input()),
            "local_operator_token_proxy": self.local_operator_token_proxy,
            "operator_spec_receipt_sha256": self.operator_spec.receipt_sha256,
            "packed_chunk_ids_sha256": self.packed_chunk_ids_sha256,
            "packed_row_sha256s": list(self.packed_row_sha256s),
            "provider_packet_receipt_sha256": (
                self.provider_packet_receipt_sha256
            ),
            "provider_prompt_count": 0,
            "provider_use_forbidden": True,
            "rejected_item_receipt_sha256s": [
                row.rejection_sha256 for row in self.rejected_items
            ],
            "retained_transformer_token_state_bytes": 0,
            "role_filtered_handle_ids": list(self.role_filtered_handle_ids),
            "selection_sha256": self.selection_sha256,
            "system_excluded_handle_ids": list(
                self.system_excluded_handle_ids
            ),
            "temporal_anchors_by_handle": list(
                self.temporal_anchors_by_handle
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class HotV3TypedOperatorAudit:
    selection_sha256: str
    provider_packet_receipt_sha256: str
    provider_payload_sha256: str
    dated_question_sha256: str
    operator_spec_receipt_sha256: str
    evidence_packet_receipt_sha256: str
    provider_input_sha256: str
    local_inventory_receipt_sha256: str
    local_operator_input_sha256: str
    selected_evidence_count: int
    packed_evidence_count: int
    dropped_evidence_count: int
    packed_raw_evidence_count: int
    packed_projection_evidence_count: int
    source_group_count: int
    citation_binding_count: int
    scalar_item_count: int
    local_item_count: int
    system_excluded_handle_count: int
    role_filtered_handle_count: int
    provider_capacity_omitted_handle_count: int
    local_operator_token_proxy: int
    represented_handle_count: int
    omitted_handle_count: int
    unresolved_slot_count: int
    frontier_mode: FrontierMode
    frontier_closed: bool
    frontier_truncated: bool
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.selection_sha256, "hot-v3 selection"),
            (self.provider_packet_receipt_sha256, "hot-v3 provider packet"),
            (self.provider_payload_sha256, "hot-v3 provider payload"),
            (self.dated_question_sha256, "hot-v3 dated question"),
            (self.operator_spec_receipt_sha256, "hot-v3 operator spec"),
            (self.evidence_packet_receipt_sha256, "hot-v3 evidence packet"),
            (self.provider_input_sha256, "hot-v3 provider input"),
            (self.local_inventory_receipt_sha256, "hot-v3 local inventory"),
            (self.local_operator_input_sha256, "hot-v3 local operator input"),
        ):
            require_sha256(value, label)
        counts = (
            self.selected_evidence_count,
            self.packed_evidence_count,
            self.dropped_evidence_count,
            self.packed_raw_evidence_count,
            self.packed_projection_evidence_count,
            self.source_group_count,
            self.citation_binding_count,
            self.scalar_item_count,
            self.local_item_count,
            self.system_excluded_handle_count,
            self.role_filtered_handle_count,
            self.provider_capacity_omitted_handle_count,
            self.local_operator_token_proxy,
            self.represented_handle_count,
            self.omitted_handle_count,
            self.unresolved_slot_count,
        )
        _require(
            all(type(value) is int and value >= 0 for value in counts),
            "hot-v3 typed audit counts must be non-negative exact integers",
        )
        _require(
            self.selected_evidence_count
            == self.packed_evidence_count + self.dropped_evidence_count,
            "hot-v3 typed audit selected partition changed",
        )
        _require(
            self.packed_evidence_count
            == self.packed_raw_evidence_count
            + self.packed_projection_evidence_count
            == self.citation_binding_count,
            "hot-v3 typed audit packed population changed",
        )
        _require(
            self.represented_handle_count
            + self.omitted_handle_count
            + self.system_excluded_handle_count
            == self.citation_binding_count,
            "hot-v3 typed audit frontier partition changed",
        )
        _require(
            self.local_item_count + self.system_excluded_handle_count
            == self.citation_binding_count
            and self.provider_capacity_omitted_handle_count
            == self.omitted_handle_count
            and self.role_filtered_handle_count <= self.local_item_count,
            "hot-v3 local/provider item partition changed",
        )
        _require(
            type(self.frontier_mode) is FrontierMode
            and type(self.frontier_closed) is bool
            and type(self.frontier_truncated) is bool,
            "hot-v3 typed audit frontier changed",
        )
        _require(
            self.frontier_mode is FrontierMode.BOUNDED
            and self.frontier_closed is False,
            "hot-v3 ranked retrieval must remain a bounded open frontier",
        )
        _require(
            self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "hot-v3 typed adapter must be provider-free/gold-blind/zero-state",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "hot-v3 typed audit changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_v3_typed_audit")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "citation_binding_count": self.citation_binding_count,
            "dated_question_sha256": self.dated_question_sha256,
            "dropped_evidence_count": self.dropped_evidence_count,
            "evidence_packet_receipt_sha256": self.evidence_packet_receipt_sha256,
            "format": FORMAT,
            "frontier_closed": self.frontier_closed,
            "frontier_mode": self.frontier_mode.value,
            "frontier_truncated": self.frontier_truncated,
            "gold_loaded": False,
            "local_inventory_receipt_sha256": self.local_inventory_receipt_sha256,
            "local_item_count": self.local_item_count,
            "local_operator_input_sha256": self.local_operator_input_sha256,
            "local_operator_token_proxy": self.local_operator_token_proxy,
            "omitted_handle_count": self.omitted_handle_count,
            "operator_spec_receipt_sha256": self.operator_spec_receipt_sha256,
            "packed_evidence_count": self.packed_evidence_count,
            "packed_projection_evidence_count": self.packed_projection_evidence_count,
            "packed_raw_evidence_count": self.packed_raw_evidence_count,
            "provider_input_sha256": self.provider_input_sha256,
            "provider_capacity_omitted_handle_count": (
                self.provider_capacity_omitted_handle_count
            ),
            "provider_packet_receipt_sha256": self.provider_packet_receipt_sha256,
            "provider_payload_sha256": self.provider_payload_sha256,
            "provider_prompt_count": 0,
            "represented_handle_count": self.represented_handle_count,
            "retained_transformer_token_state_bytes": 0,
            "role_filtered_handle_count": self.role_filtered_handle_count,
            "scalar_item_count": self.scalar_item_count,
            "selected_evidence_count": self.selected_evidence_count,
            "selection_sha256": self.selection_sha256,
            "source_group_count": self.source_group_count,
            "system_excluded_handle_count": self.system_excluded_handle_count,
            "unresolved_slot_count": self.unresolved_slot_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class HotV3TypedOperatorBundle:
    operator_spec: TypedOperatorSpec
    evidence_packet: TypedEvidencePacket
    local_inventory: HotV3LocalTypedEvidenceInventory
    provider_input: Mapping[str, Any]
    audit: HotV3TypedOperatorAudit
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.operator_spec) is TypedOperatorSpec, "hot-v3 operator spec changed")
        _require(type(self.evidence_packet) is TypedEvidencePacket, "hot-v3 evidence packet changed")
        _require(
            type(self.local_inventory) is HotV3LocalTypedEvidenceInventory,
            "hot-v3 local inventory changed type",
        )
        _require(type(self.audit) is HotV3TypedOperatorAudit, "hot-v3 typed audit changed type")
        _require(
            type(self.provider_input) is MappingProxyType,
            "hot-v3 provider input must be immutable",
        )
        provider_input = dict(self.provider_input)
        _require(
            set(provider_input) == {"dated_question", "typed_evidence"},
            "hot-v3 provider input changed schema",
        )
        dated_question = provider_input["dated_question"]
        require_text(dated_question, "hot-v3 provider dated question")
        _require(
            provider_input["typed_evidence"] == self.evidence_packet.provider_projection(),
            "hot-v3 provider input escaped its typed packet",
        )
        provider_input_sha256 = identity_sha256(provider_input)
        _require(
            self.evidence_packet.operator_spec.receipt_sha256
            == self.operator_spec.receipt_sha256
            == self.local_inventory.operator_spec.receipt_sha256
            == self.audit.operator_spec_receipt_sha256
            and self.evidence_packet.receipt_sha256
            == self.audit.evidence_packet_receipt_sha256
            and quote_sha256(dated_question) == self.audit.dated_question_sha256
            and provider_input_sha256 == self.audit.provider_input_sha256,
            "hot-v3 bundle receipt graph changed",
        )
        local_operator_input = self.local_inventory.operator_input()
        _require(
            self.local_inventory.receipt_sha256
            == self.audit.local_inventory_receipt_sha256
            and identity_sha256(local_operator_input)
            == self.audit.local_operator_input_sha256
            and self.local_inventory.local_operator_token_proxy
            == self.audit.local_operator_token_proxy
            and self.local_inventory.selection_sha256 == self.audit.selection_sha256
            and self.local_inventory.provider_packet_receipt_sha256
            == self.audit.provider_packet_receipt_sha256,
            "hot-v3 local inventory receipt graph changed",
        )
        system_handles = set(self.local_inventory.system_excluded_handle_ids)
        _require(
            not any(
                system_handles & set(item.handle_ids)
                for item in self.evidence_packet.items
            )
            and not system_handles
            & set(self.evidence_packet.frontier.available_handle_ids)
            and not system_handles
            & set(self.evidence_packet.frontier.represented_handle_ids)
            and not system_handles
            & set(self.evidence_packet.frontier.omitted_handle_ids),
            "hot-v3 system evidence entered the capped execution plane",
        )
        _require(
            self.audit.frontier_mode is self.evidence_packet.frontier.mode
            and self.audit.frontier_closed is self.evidence_packet.frontier.closed
            and self.audit.frontier_truncated is self.evidence_packet.frontier.truncated,
            "hot-v3 bundle frontier audit changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "hot-v3 typed bundle changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="hot_v3_typed_bundle")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "audit_receipt_sha256": self.audit.receipt_sha256,
            "citation_binding_count": self.audit.citation_binding_count,
            "evidence_packet_receipt_sha256": self.evidence_packet.receipt_sha256,
            "format": FORMAT,
            "frontier_closed": self.audit.frontier_closed,
            "frontier_mode": self.audit.frontier_mode.value,
            "frontier_truncated": self.audit.frontier_truncated,
            "gold_loaded": False,
            "local_inventory_receipt_sha256": self.local_inventory.receipt_sha256,
            "local_item_count": self.audit.local_item_count,
            "local_operator_input_sha256": self.audit.local_operator_input_sha256,
            "local_operator_token_proxy": self.audit.local_operator_token_proxy,
            "new_provider_calls": 0,
            "omitted_handle_count": self.audit.omitted_handle_count,
            "operator_spec_receipt_sha256": self.operator_spec.receipt_sha256,
            "provider_input_sha256": self.audit.provider_input_sha256,
            "provider_capacity_omitted_handle_count": (
                self.audit.provider_capacity_omitted_handle_count
            ),
            "represented_handle_count": self.audit.represented_handle_count,
            "retained_transformer_token_state_bytes": 0,
            "source_group_count": self.audit.source_group_count,
            "system_excluded_handle_count": (
                self.audit.system_excluded_handle_count
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def adapt_hot_v3_arm(
    dated_question: str,
    arm: Mapping[str, Any],
    *,
    selection_sha256: str,
    provider_packet: Mapping[str, Any],
    output_token_reserve: int = DEFAULT_OUTPUT_TOKEN_RESERVE,
) -> HotV3TypedOperatorBundle:
    """Authenticate and adapt one materialized A3 packet without a provider.

    Only ``packed_evidence`` crosses the adapter.  ``selected_evidence`` and
    ``dropped_chunk_ids`` authenticate ranked-prefix packing but never become
    typed handles, items, citations, or model-visible content.
    """

    require_sha256(selection_sha256, "hot-v3 selection")
    _require(
        type(output_token_reserve) is int
        and 1 <= output_token_reserve < MAX_PROMPT_TOKENS,
        "hot-v3 typed output reserve is invalid",
    )
    packed, selected_count, dropped_count, provider_payload_sha256 = _validate_arm(
        dated_question,
        arm,
    )
    provider_packet_receipt_sha256 = _validate_provider_packet(
        arm,
        provider_packet,
    )
    operator_spec = compile_typed_operator_spec(dated_question)

    group_by_source: dict[str, str] = {}
    bindings: list[EvidenceHandleBinding] = []
    raw_items: list[dict[str, Any]] = []
    handle_roles: list[str] = []
    temporal_anchors: list[str | None] = []
    system_excluded_handle_ids: list[str] = []
    packed_projection_count = 0
    for index, row in enumerate(packed, 1):
        source_id = str(row["source_id"])
        group_handle = group_by_source.setdefault(
            source_id,
            f"G{len(group_by_source) + 1:03d}",
        )
        handle_id = f"H{index:03d}"
        projected = "assertion_fact_receipt_sha256" in row
        if projected:
            packed_projection_count += 1
        evidence_receipt = (
            str(row["assertion_fact_receipt_sha256"])
            if projected
            else identity_sha256(dict(row))
        )
        locator = {
            "chunk_id": row["chunk_id"],
            "created_at": row["created_at"],
            "evidence_id": row["evidence_id"],
            "role": row["role"],
            "route": row["route"],
            "source_id": source_id,
        }
        if "turn_id" in row:
            locator["turn_id"] = row["turn_id"]
        bindings.append(
            EvidenceHandleBinding(
                handle_id,
                EvidenceOrigin.DIRECT_POINTER,
                ProvenanceGrade.DIRECT_POINTER,
                group_handle,
                selection_sha256,
                provider_packet_receipt_sha256,
                evidence_receipt,
                str(row["raw_text_sha256"]),
                str(row["raw_text_sha256"]),
                len(str(row["raw_text"])),
                identity_sha256(locator),
            )
        )
        role = str(row["role"])
        handle_roles.append(role)
        temporal_anchors.append(
            None if role == "system" else str(row["created_at"]) or None
        )
        if role == "system":
            system_excluded_handle_ids.append(handle_id)
        else:
            raw_items.append(
                _raw_typed_item(
                    row,
                    handle_id=handle_id,
                    operator_spec=operator_spec,
                    dated_question=dated_question,
                )
            )

    exact_bindings = tuple(bindings)
    parsed = parse_typed_items(
        raw_items,
        operator_spec=operator_spec,
        bindings=exact_bindings,
    )
    local_frontier = build_frontier_receipt(
        operator_spec,
        exact_bindings,
        parsed.accepted_items,
        parsed.rejected_items,
        mode=FrontierMode.BOUNDED,
        truncated=bool(dropped_count),
        conflict_policy=ConflictPolicy.QUARANTINE,
    )
    exact_roles = tuple(handle_roles)
    handle_ids = tuple(row.handle_id for row in exact_bindings)
    role_filtered_handle_ids = tuple(
        handle
        for handle, role in zip(handle_ids, exact_roles, strict=True)
        if role != "system"
        and operator_spec.required_evidence_role is not None
        and role != operator_spec.required_evidence_role
    )
    local_inventory = HotV3LocalTypedEvidenceInventory(
        dated_question=dated_question,
        operator_spec=operator_spec,
        handles=tuple(row.opaque() for row in exact_bindings),
        local_bindings=exact_bindings,
        items=parsed.accepted_items,
        rejected_items=parsed.rejected_items,
        frontier=local_frontier,
        conflict_policy=ConflictPolicy.QUARANTINE,
        selection_sha256=selection_sha256,
        provider_packet_receipt_sha256=provider_packet_receipt_sha256,
        packed_chunk_ids_sha256=identity_sha256(arm["packed_chunk_ids"]),
        packed_row_sha256s=tuple(identity_sha256(dict(row)) for row in packed),
        handle_roles=exact_roles,
        temporal_anchors_by_handle=tuple(temporal_anchors),
        system_excluded_handle_ids=tuple(system_excluded_handle_ids),
        role_filtered_handle_ids=role_filtered_handle_ids,
    )
    system_handle_set = set(system_excluded_handle_ids)
    provider_bindings = tuple(
        row for row in exact_bindings if row.handle_id not in system_handle_set
    )
    packet = build_typed_evidence_packet(
        operator_spec,
        provider_bindings,
        parsed,
        sealed_input_artifact_sha256s=(selection_sha256,),
        frontier_mode=FrontierMode.BOUNDED,
        conflict_policy=ConflictPolicy.QUARANTINE,
        output_token_reserve=output_token_reserve,
        truncated=bool(dropped_count),
        provider_payload_mode=ProviderPayloadMode.COMPACT_FINAL_V2,
    )
    provider_input_dict = {
        "dated_question": dated_question,
        "typed_evidence": packet.provider_projection(),
    }
    assert_gold_blind(provider_input_dict, path="hot_v3_typed_provider_input")
    provider_input = MappingProxyType(provider_input_dict)
    frontier = packet.frontier
    local_omitted = set(local_frontier.omitted_handle_ids)
    provider_omitted = set(frontier.omitted_handle_ids)
    audit = HotV3TypedOperatorAudit(
        selection_sha256=selection_sha256,
        provider_packet_receipt_sha256=provider_packet_receipt_sha256,
        provider_payload_sha256=provider_payload_sha256,
        dated_question_sha256=quote_sha256(dated_question),
        operator_spec_receipt_sha256=operator_spec.receipt_sha256,
        evidence_packet_receipt_sha256=packet.receipt_sha256,
        provider_input_sha256=identity_sha256(provider_input_dict),
        local_inventory_receipt_sha256=local_inventory.receipt_sha256,
        local_operator_input_sha256=identity_sha256(
            local_inventory.operator_input()
        ),
        selected_evidence_count=selected_count,
        packed_evidence_count=len(packed),
        dropped_evidence_count=dropped_count,
        packed_raw_evidence_count=len(packed) - packed_projection_count,
        packed_projection_evidence_count=packed_projection_count,
        source_group_count=len(group_by_source),
        citation_binding_count=len(exact_bindings),
        scalar_item_count=sum(
            item.numeric_value is not None for item in packet.items
        ),
        local_item_count=len(local_inventory.items),
        system_excluded_handle_count=len(system_excluded_handle_ids),
        role_filtered_handle_count=len(role_filtered_handle_ids),
        provider_capacity_omitted_handle_count=len(
            provider_omitted - local_omitted
        ),
        local_operator_token_proxy=local_inventory.local_operator_token_proxy,
        represented_handle_count=len(frontier.represented_handle_ids),
        omitted_handle_count=len(frontier.omitted_handle_ids),
        unresolved_slot_count=len(frontier.unresolved_slot_ids),
        frontier_mode=frontier.mode,
        frontier_closed=frontier.closed,
        frontier_truncated=frontier.truncated,
    )
    return HotV3TypedOperatorBundle(
        operator_spec=operator_spec,
        evidence_packet=packet,
        local_inventory=local_inventory,
        provider_input=provider_input,
        audit=audit,
    )


__all__ = [
    "FORMAT",
    "LOCAL_INVENTORY_FORMAT",
    "LOCAL_OPERATOR_EVIDENCE_FORMAT",
    "PROVIDER_PACKET_FORMAT",
    "MAX_CONTEXT_TOKENS",
    "MAX_PROMPT_TOKENS",
    "HotV3TypedOperatorAudit",
    "HotV3TypedOperatorBundle",
    "HotV3LocalTypedEvidenceInventory",
    "adapt_hot_v3_arm",
]
