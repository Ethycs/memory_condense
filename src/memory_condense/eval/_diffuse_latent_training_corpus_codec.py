"""Closed, content-bearing codec for latent-router corpus payload shards.

Only payload shards use this codec.  Corpus manifests and receipts keep text
out of their schemas.  The payload stores one exact retrieval query, one
``ClosurePlan``, and one selected ``EvidencePacket``.  Packet atoms and
bundles are references into the plan tables so authoritative atom text is
stored exactly once.

The module is deliberately provider- and tensor-free.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from memory_condense.domain._discourse_identity import canonical_json
from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
    ClosureReceipt,
    ClosureScopeWitness,
    DiscourseSnapshot,
    EpisodeSeed,
    EvidenceAtom,
    EvidenceBundle,
    EvidenceObligation,
    EvidencePacket,
    EvidenceSpan,
    ObligationResult,
    QueryProgram,
)


LATENT_TRAINING_PAYLOAD_FORMAT = (
    "memory-condense-diffuse-latent-training-payload-v1"
)

_FORBIDDEN_OPEN_CHANNELS = frozenset(
    {
        "gold", "gold_answer", "answer", "category", "prediction", "judge",
        "evaluator_score", "annotated_source_label", "tensor", "tensors",
        "embedding", "embeddings", "embedding_vector", "embedding_values",
        "activation", "activations", "attention_tensor", "hidden_states",
        "logits",
    }
)


def _open_mapping_firebreak(value: Any, label: str) -> None:
    """Close evaluator/model-output channels inside otherwise open JSON."""

    value_type = type(value)
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError(f"{label} keys must be exact strings")
            if key.strip().casefold() in _FORBIDDEN_OPEN_CHANNELS:
                raise ValueError(f"{label} contains a forbidden output channel")
            _open_mapping_firebreak(child, f"{label}.{key}")
        return
    if value_type in {tuple, list}:
        for index, child in enumerate(value):
            _open_mapping_firebreak(child, f"{label}[{index}]")
        return
    if value_type is str and value.strip().casefold() in _FORBIDDEN_OPEN_CHANNELS:
        raise ValueError(f"{label} contains a forbidden output channel")


@dataclass(frozen=True, slots=True)
class DecodedLatentTrainingPayload:
    """Immutable decoded content handed to a future training consumer."""

    question_id: str
    retrieval_query: str
    prompt_question: str
    plan: ClosurePlan
    packet: EvidencePacket

    def __post_init__(self) -> None:
        for name in ("question_id", "retrieval_query", "prompt_question"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"payload {name} must be a non-empty exact string")
        if type(self.plan) is not ClosurePlan:
            raise TypeError("payload plan must be an exact ClosurePlan")
        if type(self.packet) is not EvidencePacket:
            raise TypeError("payload packet must be an exact EvidencePacket")
        if self.retrieval_query != self.plan.query_program.query:
            raise ValueError("payload query differs from its closure plan")

    @property
    def query(self) -> str:
        """Compatibility name for the exact retrieval query."""

        return self.retrieval_query


def _json_tree(value: Any, label: str) -> Any:
    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("payload contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{label} mapping keys must be exact strings")
        return {
            key: _json_tree(child, f"{label}.{key}")
            for key, child in value.items()
        }
    if value_type is tuple:
        return [
            _json_tree(child, f"{label}[{index}]")
            for index, child in enumerate(value)
        ]
    raise TypeError(f"{label} has unsupported type {type(value).__name__}")


def _encode_span(value: EvidenceSpan, label: str) -> dict[str, Any]:
    if type(value) is not EvidenceSpan:
        raise TypeError(f"{label} must be an exact EvidenceSpan")
    return {
        "chunk_id": _string(value.chunk_id, f"{label}.chunk_id"),
        "start_char": _integer(value.start_char, f"{label}.start_char"),
        "end_char": _integer(value.end_char, f"{label}.end_char"),
        "quote_sha256": _string(value.quote_sha256, f"{label}.quote_sha256"),
        "ordinal": _integer(value.ordinal, f"{label}.ordinal"),
        "source_id": (
            None
            if value.source_id is None
            else _string(value.source_id, f"{label}.source_id")
        ),
        "turn_start_char": _integer(
            value.turn_start_char, f"{label}.turn_start_char"
        ),
        "turn_id": (
            None
            if value.turn_id is None
            else _string(value.turn_id, f"{label}.turn_id")
        ),
        "role": (
            None if value.role is None else _string(value.role, f"{label}.role")
        ),
        "created_at": (
            None
            if value.created_at is None
            else _string(value.created_at, f"{label}.created_at")
        ),
    }


def _encode_atom(value: EvidenceAtom, label: str) -> dict[str, Any]:
    if type(value) is not EvidenceAtom:
        raise TypeError(f"{label} must be an exact EvidenceAtom")
    return {
        "atom_id": _string(value.atom_id, f"{label}.atom_id"),
        "span": _encode_span(value.span, f"{label}.span"),
        "text": _string(value.text, f"{label}.text", empty=True),
        "label": _string(value.label, f"{label}.label"),
        "role": (
            None if value.role is None else _string(value.role, f"{label}.role")
        ),
        "created_at": (
            None
            if value.created_at is None
            else _string(value.created_at, f"{label}.created_at")
        ),
    }


def _encode_bundle(value: EvidenceBundle, label: str) -> dict[str, Any]:
    if type(value) is not EvidenceBundle:
        raise TypeError(f"{label} must be an exact EvidenceBundle")
    return {
        "bundle_id": _string(value.bundle_id, f"{label}.bundle_id"),
        "atom_ids": list(_exact_string_tuple(value.atom_ids, f"{label}.atom_ids")),
        "obligation_ids": list(
            _exact_string_tuple(value.obligation_ids, f"{label}.obligation_ids")
        ),
        "unit_ids": list(_exact_string_tuple(value.unit_ids, f"{label}.unit_ids")),
        "relation_ids": list(
            _exact_string_tuple(value.relation_ids, f"{label}.relation_ids")
        ),
        "required": _boolean(value.required, f"{label}.required"),
        "utility": _float(value.utility, f"{label}.utility"),
    }


def _encode_obligation(value: EvidenceObligation, label: str) -> dict[str, Any]:
    if type(value) is not EvidenceObligation:
        raise TypeError(f"{label} must be an exact EvidenceObligation")
    return {
        "obligation_id": _string(value.obligation_id, f"{label}.obligation_id"),
        "kind": _string(value.kind, f"{label}.kind"),
        "required": _boolean(value.required, f"{label}.required"),
        "weight": _float(value.weight, f"{label}.weight"),
        "unit_kinds": list(
            _exact_string_tuple(value.unit_kinds, f"{label}.unit_kinds")
        ),
        "relation_types": list(
            _exact_string_tuple(value.relation_types, f"{label}.relation_types")
        ),
        "subject_terms": list(
            _exact_string_tuple(value.subject_terms, f"{label}.subject_terms")
        ),
        "dependencies": list(
            _exact_string_tuple(value.dependencies, f"{label}.dependencies")
        ),
        "min_count": _integer(value.min_count, f"{label}.min_count"),
        "max_count": (
            None
            if value.max_count is None
            else _integer(value.max_count, f"{label}.max_count")
        ),
        "temporal_stance": _string(
            value.temporal_stance, f"{label}.temporal_stance"
        ),
    }


def _encode_query_program(value: QueryProgram, label: str) -> dict[str, Any]:
    if type(value) is not QueryProgram:
        raise TypeError(f"{label} must be an exact QueryProgram")
    if type(value.obligations) is not tuple:
        raise TypeError(f"{label}.obligations must be an exact tuple")
    return {
        "query": _string(value.query, f"{label}.query"),
        "intent": _string(value.intent, f"{label}.intent"),
        "subject_terms": list(
            _exact_string_tuple(value.subject_terms, f"{label}.subject_terms")
        ),
        "obligations": [
            _encode_obligation(item, f"{label}.obligations[{index}]")
            for index, item in enumerate(value.obligations)
        ],
        "as_of_ordinal": (
            None
            if value.as_of_ordinal is None
            else _integer(value.as_of_ordinal, f"{label}.as_of_ordinal")
        ),
        "ordering": _string(value.ordering, f"{label}.ordering"),
        "cardinality": (
            None
            if value.cardinality is None
            else _integer(value.cardinality, f"{label}.cardinality")
        ),
        "program_sha256": _string(
            value.program_sha256, f"{label}.program_sha256"
        ),
    }


def _encode_policy(value: ClosurePolicy, label: str) -> dict[str, Any]:
    if type(value) is not ClosurePolicy:
        raise TypeError(f"{label} must be an exact ClosurePolicy")
    return {
        "max_hops": _integer(value.max_hops, f"{label}.max_hops"),
        "max_units": _integer(value.max_units, f"{label}.max_units"),
        "max_relations": _integer(value.max_relations, f"{label}.max_relations"),
        "max_degree": _integer(value.max_degree, f"{label}.max_degree"),
        "max_episode_neighbors": _integer(
            value.max_episode_neighbors, f"{label}.max_episode_neighbors"
        ),
        "max_frontier": _integer(value.max_frontier, f"{label}.max_frontier"),
        "max_bundles": _integer(value.max_bundles, f"{label}.max_bundles"),
        "beam_width": _integer(value.beam_width, f"{label}.beam_width"),
        "min_relation_confidence": _float(
            value.min_relation_confidence, f"{label}.min_relation_confidence"
        ),
    }


def _encode_snapshot(value: DiscourseSnapshot, label: str) -> dict[str, Any]:
    if type(value) is not DiscourseSnapshot:
        raise TypeError(f"{label} must be an exact DiscourseSnapshot")
    return {
        "max_turn_ordinal": _integer(
            value.max_turn_ordinal, f"{label}.max_turn_ordinal"
        ),
        "chunk_count": _integer(value.chunk_count, f"{label}.chunk_count"),
        "graph_revision": _integer(
            value.graph_revision, f"{label}.graph_revision"
        ),
        "schema_version": _integer(
            value.schema_version, f"{label}.schema_version"
        ),
        "artifact_ids": list(
            _exact_string_tuple(value.artifact_ids, f"{label}.artifact_ids")
        ),
        "source_revision": _integer(
            value.source_revision, f"{label}.source_revision"
        ),
        "graph_content_revision": _integer(
            value.graph_content_revision, f"{label}.graph_content_revision"
        ),
        "source_content_sha256": _string(
            value.source_content_sha256, f"{label}.source_content_sha256"
        ),
        "graph_content_sha256": _string(
            value.graph_content_sha256, f"{label}.graph_content_sha256"
        ),
        "snapshot_sha256": _string(
            value.snapshot_sha256, f"{label}.snapshot_sha256"
        ),
    }


def _encode_seed(value: EpisodeSeed, label: str) -> dict[str, Any]:
    if type(value) is not EpisodeSeed:
        raise TypeError(f"{label} must be an exact EpisodeSeed")
    return {
        "episode_id": _string(value.episode_id, f"{label}.episode_id"),
        "anchor_chunk_id": _string(
            value.anchor_chunk_id, f"{label}.anchor_chunk_id"
        ),
        "score": _float(value.score, f"{label}.score"),
        "route": _string(value.route, f"{label}.route"),
        "path": list(_exact_string_tuple(value.path, f"{label}.path")),
    }


def _encode_result(value: ObligationResult, label: str) -> dict[str, Any]:
    if type(value) is not ObligationResult:
        raise TypeError(f"{label} must be an exact ObligationResult")
    return {
        "obligation_id": _string(value.obligation_id, f"{label}.obligation_id"),
        "status": _string(value.status, f"{label}.status"),
        "unit_ids": list(_exact_string_tuple(value.unit_ids, f"{label}.unit_ids")),
        "relation_ids": list(
            _exact_string_tuple(value.relation_ids, f"{label}.relation_ids")
        ),
        "bundle_ids": list(
            _exact_string_tuple(value.bundle_ids, f"{label}.bundle_ids")
        ),
        "reason": (
            None
            if value.reason is None
            else _string(value.reason, f"{label}.reason")
        ),
    }


def _encode_witness(value: ClosureScopeWitness, label: str) -> dict[str, Any]:
    if type(value) is not ClosureScopeWitness:
        raise TypeError(f"{label} must be an exact ClosureScopeWitness")
    _open_mapping_firebreak(value.detail, f"{label}.detail")
    return {
        "kind": _string(value.kind, f"{label}.kind"),
        "subject_id": _string(value.subject_id, f"{label}.subject_id"),
        "requested_limit": (
            None
            if value.requested_limit is None
            else _integer(value.requested_limit, f"{label}.requested_limit")
        ),
        "returned_count": _integer(
            value.returned_count, f"{label}.returned_count"
        ),
        "exhaustive": _boolean(value.exhaustive, f"{label}.exhaustive"),
        "detail": _json_tree(value.detail, f"{label}.detail"),
        "witness_sha256": _string(
            value.witness_sha256, f"{label}.witness_sha256"
        ),
    }


def _encode_plan(value: ClosurePlan, label: str) -> dict[str, Any]:
    if type(value) is not ClosurePlan:
        raise TypeError(f"{label} must be an exact ClosurePlan")
    for name in (
        "seeds",
        "atoms",
        "bundles",
        "obligation_results",
        "scope_witnesses",
    ):
        if type(getattr(value, name)) is not tuple:
            raise TypeError(f"{label}.{name} must be an exact tuple")
    return {
        "query_program": _encode_query_program(
            value.query_program, f"{label}.query_program"
        ),
        "policy": _encode_policy(value.policy, f"{label}.policy"),
        "snapshot": _encode_snapshot(value.snapshot, f"{label}.snapshot"),
        "seeds": [
            _encode_seed(item, f"{label}.seeds[{index}]")
            for index, item in enumerate(value.seeds)
        ],
        "atoms": [
            _encode_atom(item, f"{label}.atoms[{index}]")
            for index, item in enumerate(value.atoms)
        ],
        "bundles": [
            _encode_bundle(item, f"{label}.bundles[{index}]")
            for index, item in enumerate(value.bundles)
        ],
        "obligation_results": [
            _encode_result(item, f"{label}.obligation_results[{index}]")
            for index, item in enumerate(value.obligation_results)
        ],
        "visited_episode_ids": list(
            _exact_string_tuple(
                value.visited_episode_ids, f"{label}.visited_episode_ids"
            )
        ),
        "visited_unit_ids": list(
            _exact_string_tuple(value.visited_unit_ids, f"{label}.visited_unit_ids")
        ),
        "visited_relation_ids": list(
            _exact_string_tuple(
                value.visited_relation_ids, f"{label}.visited_relation_ids"
            )
        ),
        "stopping_reason": _string(
            value.stopping_reason, f"{label}.stopping_reason"
        ),
        "complete_claimed": _boolean(
            value.complete_claimed, f"{label}.complete_claimed"
        ),
        "scope_witnesses": [
            _encode_witness(item, f"{label}.scope_witnesses[{index}]")
            for index, item in enumerate(value.scope_witnesses)
        ],
        "direct_chunk_ids": list(
            _exact_string_tuple(value.direct_chunk_ids, f"{label}.direct_chunk_ids")
        ),
        "expansion_receipt_sha256": (
            None
            if value.expansion_receipt_sha256 is None
            else _string(
                value.expansion_receipt_sha256,
                f"{label}.expansion_receipt_sha256",
            )
        ),
        "artifact_id": (
            None
            if value.artifact_id is None
            else _string(value.artifact_id, f"{label}.artifact_id")
        ),
        "plan_sha256": _string(value.plan_sha256, f"{label}.plan_sha256"),
    }


def _encode_receipt(value: ClosureReceipt, label: str) -> dict[str, Any]:
    if type(value) is not ClosureReceipt:
        raise TypeError(f"{label} must be an exact ClosureReceipt")
    _open_mapping_firebreak(
        value.dropped_bundle_reasons, f"{label}.dropped_bundle_reasons"
    )
    dropped = _json_tree(value.dropped_bundle_reasons, f"{label}.dropped")
    if type(dropped) is not dict or any(type(item) is not str for item in dropped.values()):
        raise TypeError(f"{label}.dropped_bundle_reasons must map strings to strings")
    return {
        "plan_sha256": _string(value.plan_sha256, f"{label}.plan_sha256"),
        "context_sha256": _string(
            value.context_sha256, f"{label}.context_sha256"
        ),
        "selected_bundle_ids": list(
            _exact_string_tuple(
                value.selected_bundle_ids, f"{label}.selected_bundle_ids"
            )
        ),
        "selected_atom_ids": list(
            _exact_string_tuple(
                value.selected_atom_ids, f"{label}.selected_atom_ids"
            )
        ),
        "dropped_bundle_reasons": dropped,
        "context_token_proxy": _integer(
            value.context_token_proxy, f"{label}.context_token_proxy"
        ),
        "max_context_token_proxy": _integer(
            value.max_context_token_proxy, f"{label}.max_context_token_proxy"
        ),
        "tokenizer_identity": _string(
            value.tokenizer_identity, f"{label}.tokenizer_identity"
        ),
        "stopping_reason": _string(
            value.stopping_reason, f"{label}.stopping_reason"
        ),
        "complete_claimed": _boolean(
            value.complete_claimed, f"{label}.complete_claimed"
        ),
        "retained_request_token_state_bytes": _integer(
            value.retained_request_token_state_bytes,
            f"{label}.retained_request_token_state_bytes",
        ),
        "prompt_token_proxy": _optional_exact_integer(
            value.prompt_token_proxy, f"{label}.prompt_token_proxy"
        ),
        "max_prompt_token_proxy": _optional_exact_integer(
            value.max_prompt_token_proxy, f"{label}.max_prompt_token_proxy"
        ),
        "responder_output_token_reserve": _integer(
            value.responder_output_token_reserve,
            f"{label}.responder_output_token_reserve",
        ),
        "prompt_workspace_token_proxy": _optional_exact_integer(
            value.prompt_workspace_token_proxy,
            f"{label}.prompt_workspace_token_proxy",
        ),
        "base_messages_sha256": _optional_exact_string(
            value.base_messages_sha256, f"{label}.base_messages_sha256"
        ),
        "evidence_message_role": _optional_exact_string(
            value.evidence_message_role, f"{label}.evidence_message_role"
        ),
        "evidence_prefix_sha256": _optional_exact_string(
            value.evidence_prefix_sha256, f"{label}.evidence_prefix_sha256"
        ),
        "evidence_suffix_sha256": _optional_exact_string(
            value.evidence_suffix_sha256, f"{label}.evidence_suffix_sha256"
        ),
        "prompt_messages_sha256": _optional_exact_string(
            value.prompt_messages_sha256, f"{label}.prompt_messages_sha256"
        ),
        "receipt_sha256": _string(
            value.receipt_sha256, f"{label}.receipt_sha256"
        ),
    }


def _same_identity(left: Any, right: Any, label: str) -> None:
    left_payload = left.identity_payload()
    right_payload = right.identity_payload()
    if canonical_json(left_payload) != canonical_json(right_payload):
        raise ValueError(f"packet {label} differs from its closure-plan table")


def encode_latent_training_payload(
    retrieval_query: str,
    plan: ClosurePlan,
    packet: EvidencePacket,
    *,
    question_id: str,
    prompt_question: str,
) -> bytes:
    """Encode one exact row using canonical UTF-8 JSON without a newline."""

    for name, value in (
        ("question_id", question_id),
        ("retrieval_query", retrieval_query),
        ("prompt_question", prompt_question),
    ):
        if type(value) is not str or not value.strip():
            raise ValueError(f"payload {name} must be a non-empty exact string")
    if type(plan) is not ClosurePlan:
        raise TypeError("payload plan must be an exact ClosurePlan")
    if type(packet) is not EvidencePacket:
        raise TypeError("payload packet must be an exact EvidencePacket")
    if retrieval_query != plan.query_program.query:
        raise ValueError("payload query differs from its closure plan")
    plan_atoms = {atom.atom_id: atom for atom in plan.atoms}
    plan_bundles = {bundle.bundle_id: bundle for bundle in plan.bundles}
    for atom in packet.atoms:
        expected = plan_atoms.get(atom.atom_id)
        if expected is None:
            raise ValueError("packet atom is absent from its closure plan")
        _same_identity(atom, expected, "atom")
    for bundle in packet.bundles:
        expected = plan_bundles.get(bundle.bundle_id)
        if expected is None:
            raise ValueError("packet bundle is absent from its closure plan")
        _same_identity(bundle, expected, "bundle")
    value = {
        "format": LATENT_TRAINING_PAYLOAD_FORMAT,
        "query": {
            "question_id": question_id,
            "retrieval_query": retrieval_query,
            "prompt_question": prompt_question,
        },
        "plan": _encode_plan(plan, "payload.plan"),
        "packet": {
            "context": packet.context,
            "atom_ids": [atom.atom_id for atom in packet.atoms],
            "bundle_ids": [bundle.bundle_id for bundle in packet.bundles],
            "receipt": _encode_receipt(packet.receipt, "payload.packet.receipt"),
        },
    }
    return canonical_json(value).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ValueError(f"unsupported JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("payload contains a duplicate JSON key")
        result[key] = value
    return result


def _object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be an exact JSON object")
    if set(value) != keys:
        raise ValueError(f"{label} has a non-closed schema")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{label} must be an exact JSON array")
    return value


def _string(value: Any, label: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value.strip()):
        raise TypeError(f"{label} must be an exact non-empty string")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    return None if value is None else _string(value, label)


def _optional_exact_string(value: Any, label: str) -> str | None:
    return None if value is None else _string(value, label)


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact boolean")
    return value


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    return value


def _optional_integer(value: Any, label: str) -> int | None:
    return None if value is None else _integer(value, label)


def _optional_exact_integer(value: Any, label: str) -> int | None:
    return None if value is None else _integer(value, label)


def _float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{label} must be an exact finite float")
    return value


def _optional_float(value: Any, label: str) -> float | None:
    return None if value is None else _float(value, label)


def _strings(value: Any, label: str) -> tuple[str, ...]:
    rows = _list(value, label)
    return tuple(_string(item, f"{label} item") for item in rows)


def _exact_string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not tuple or any(type(item) is not str for item in value):
        raise TypeError(f"{label} must be an exact tuple of exact strings")
    return value


def _frozen_json(value: Any, label: str) -> Any:
    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return value
    if value_type is list:
        return tuple(
            _frozen_json(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    if value_type is dict:
        if any(type(key) is not str for key in value):
            raise TypeError(f"{label} has a non-string key")
        return {key: _frozen_json(item, f"{label}.{key}") for key, item in value.items()}
    raise TypeError(f"{label} contains a non-JSON value")


def _decode_span(value: Any, label: str) -> EvidenceSpan:
    row = _object(
        value,
        {
            "chunk_id",
            "start_char",
            "end_char",
            "quote_sha256",
            "ordinal",
            "source_id",
            "turn_start_char",
            "turn_id",
            "role",
            "created_at",
        },
        label,
    )
    return EvidenceSpan(
        chunk_id=_string(row["chunk_id"], f"{label}.chunk_id"),
        start_char=_integer(row["start_char"], f"{label}.start_char"),
        end_char=_integer(row["end_char"], f"{label}.end_char"),
        quote_sha256=_string(row["quote_sha256"], f"{label}.quote_sha256"),
        ordinal=_integer(row["ordinal"], f"{label}.ordinal"),
        source_id=_optional_string(row["source_id"], f"{label}.source_id"),
        turn_start_char=_integer(
            row["turn_start_char"], f"{label}.turn_start_char"
        ),
        turn_id=_optional_string(row["turn_id"], f"{label}.turn_id"),
        role=_optional_string(row["role"], f"{label}.role"),
        created_at=_optional_string(row["created_at"], f"{label}.created_at"),
    )


def _decode_atom(value: Any, label: str) -> EvidenceAtom:
    row = _object(
        value,
        {"atom_id", "span", "text", "label", "role", "created_at"},
        label,
    )
    return EvidenceAtom(
        atom_id=_string(row["atom_id"], f"{label}.atom_id"),
        span=_decode_span(row["span"], f"{label}.span"),
        text=_string(row["text"], f"{label}.text", empty=True),
        label=_string(row["label"], f"{label}.label"),
        role=_optional_string(row["role"], f"{label}.role"),
        created_at=_optional_string(row["created_at"], f"{label}.created_at"),
    )


def _decode_bundle(value: Any, label: str) -> EvidenceBundle:
    row = _object(
        value,
        {
            "bundle_id",
            "atom_ids",
            "obligation_ids",
            "unit_ids",
            "relation_ids",
            "required",
            "utility",
        },
        label,
    )
    return EvidenceBundle(
        bundle_id=_string(row["bundle_id"], f"{label}.bundle_id"),
        atom_ids=_strings(row["atom_ids"], f"{label}.atom_ids"),
        obligation_ids=_strings(
            row["obligation_ids"], f"{label}.obligation_ids"
        ),
        unit_ids=_strings(row["unit_ids"], f"{label}.unit_ids"),
        relation_ids=_strings(row["relation_ids"], f"{label}.relation_ids"),
        required=_boolean(row["required"], f"{label}.required"),
        utility=_float(row["utility"], f"{label}.utility"),
    )


def _decode_obligation(value: Any, label: str) -> EvidenceObligation:
    row = _object(
        value,
        {
            "obligation_id",
            "kind",
            "required",
            "weight",
            "unit_kinds",
            "relation_types",
            "subject_terms",
            "dependencies",
            "min_count",
            "max_count",
            "temporal_stance",
        },
        label,
    )
    return EvidenceObligation(
        obligation_id=_string(
            row["obligation_id"], f"{label}.obligation_id"
        ),
        kind=_string(row["kind"], f"{label}.kind"),
        required=_boolean(row["required"], f"{label}.required"),
        weight=_float(row["weight"], f"{label}.weight"),
        unit_kinds=_strings(row["unit_kinds"], f"{label}.unit_kinds"),
        relation_types=_strings(
            row["relation_types"], f"{label}.relation_types"
        ),
        subject_terms=_strings(row["subject_terms"], f"{label}.subject_terms"),
        dependencies=_strings(row["dependencies"], f"{label}.dependencies"),
        min_count=_integer(row["min_count"], f"{label}.min_count"),
        max_count=_optional_integer(row["max_count"], f"{label}.max_count"),
        temporal_stance=_string(
            row["temporal_stance"], f"{label}.temporal_stance"
        ),
    )


def _decode_query_program(value: Any, label: str) -> QueryProgram:
    row = _object(
        value,
        {
            "query",
            "intent",
            "subject_terms",
            "obligations",
            "as_of_ordinal",
            "ordering",
            "cardinality",
            "program_sha256",
        },
        label,
    )
    obligations = tuple(
        _decode_obligation(item, f"{label}.obligations[{index}]")
        for index, item in enumerate(_list(row["obligations"], f"{label}.obligations"))
    )
    return QueryProgram(
        query=_string(row["query"], f"{label}.query"),
        intent=_string(row["intent"], f"{label}.intent"),
        subject_terms=_strings(row["subject_terms"], f"{label}.subject_terms"),
        obligations=obligations,
        as_of_ordinal=_optional_integer(
            row["as_of_ordinal"], f"{label}.as_of_ordinal"
        ),
        ordering=_string(row["ordering"], f"{label}.ordering"),
        cardinality=_optional_integer(row["cardinality"], f"{label}.cardinality"),
        program_sha256=_string(
            row["program_sha256"], f"{label}.program_sha256"
        ),
    )


def _decode_policy(value: Any, label: str) -> ClosurePolicy:
    row = _object(
        value,
        {
            "max_hops",
            "max_units",
            "max_relations",
            "max_degree",
            "max_episode_neighbors",
            "max_frontier",
            "max_bundles",
            "beam_width",
            "min_relation_confidence",
        },
        label,
    )
    return ClosurePolicy(
        max_hops=_integer(row["max_hops"], f"{label}.max_hops"),
        max_units=_integer(row["max_units"], f"{label}.max_units"),
        max_relations=_integer(row["max_relations"], f"{label}.max_relations"),
        max_degree=_integer(row["max_degree"], f"{label}.max_degree"),
        max_episode_neighbors=_integer(
            row["max_episode_neighbors"], f"{label}.max_episode_neighbors"
        ),
        max_frontier=_integer(row["max_frontier"], f"{label}.max_frontier"),
        max_bundles=_integer(row["max_bundles"], f"{label}.max_bundles"),
        beam_width=_integer(row["beam_width"], f"{label}.beam_width"),
        min_relation_confidence=_float(
            row["min_relation_confidence"], f"{label}.min_relation_confidence"
        ),
    )


def _decode_snapshot(value: Any, label: str) -> DiscourseSnapshot:
    row = _object(
        value,
        {
            "max_turn_ordinal",
            "chunk_count",
            "graph_revision",
            "schema_version",
            "artifact_ids",
            "source_revision",
            "graph_content_revision",
            "source_content_sha256",
            "graph_content_sha256",
            "snapshot_sha256",
        },
        label,
    )
    return DiscourseSnapshot(
        max_turn_ordinal=_integer(
            row["max_turn_ordinal"], f"{label}.max_turn_ordinal"
        ),
        chunk_count=_integer(row["chunk_count"], f"{label}.chunk_count"),
        graph_revision=_integer(
            row["graph_revision"], f"{label}.graph_revision"
        ),
        schema_version=_integer(row["schema_version"], f"{label}.schema_version"),
        artifact_ids=_strings(row["artifact_ids"], f"{label}.artifact_ids"),
        source_revision=_integer(
            row["source_revision"], f"{label}.source_revision"
        ),
        graph_content_revision=_integer(
            row["graph_content_revision"], f"{label}.graph_content_revision"
        ),
        source_content_sha256=_string(
            row["source_content_sha256"], f"{label}.source_content_sha256"
        ),
        graph_content_sha256=_string(
            row["graph_content_sha256"], f"{label}.graph_content_sha256"
        ),
        snapshot_sha256=_string(
            row["snapshot_sha256"], f"{label}.snapshot_sha256"
        ),
    )


def _decode_seed(value: Any, label: str) -> EpisodeSeed:
    row = _object(
        value,
        {"episode_id", "anchor_chunk_id", "score", "route", "path"},
        label,
    )
    return EpisodeSeed(
        episode_id=_string(row["episode_id"], f"{label}.episode_id"),
        anchor_chunk_id=_string(
            row["anchor_chunk_id"], f"{label}.anchor_chunk_id"
        ),
        score=_float(row["score"], f"{label}.score"),
        route=_string(row["route"], f"{label}.route"),
        path=_strings(row["path"], f"{label}.path"),
    )


def _decode_result(value: Any, label: str) -> ObligationResult:
    row = _object(
        value,
        {
            "obligation_id",
            "status",
            "unit_ids",
            "relation_ids",
            "bundle_ids",
            "reason",
        },
        label,
    )
    return ObligationResult(
        obligation_id=_string(
            row["obligation_id"], f"{label}.obligation_id"
        ),
        status=_string(row["status"], f"{label}.status"),
        unit_ids=_strings(row["unit_ids"], f"{label}.unit_ids"),
        relation_ids=_strings(row["relation_ids"], f"{label}.relation_ids"),
        bundle_ids=_strings(row["bundle_ids"], f"{label}.bundle_ids"),
        reason=_optional_string(row["reason"], f"{label}.reason"),
    )


def _decode_witness(value: Any, label: str) -> ClosureScopeWitness:
    row = _object(
        value,
        {
            "kind",
            "subject_id",
            "requested_limit",
            "returned_count",
            "exhaustive",
            "detail",
            "witness_sha256",
        },
        label,
    )
    _open_mapping_firebreak(row["detail"], f"{label}.detail")
    return ClosureScopeWitness(
        kind=_string(row["kind"], f"{label}.kind"),
        subject_id=_string(row["subject_id"], f"{label}.subject_id"),
        requested_limit=_optional_integer(
            row["requested_limit"], f"{label}.requested_limit"
        ),
        returned_count=_integer(
            row["returned_count"], f"{label}.returned_count"
        ),
        exhaustive=_boolean(row["exhaustive"], f"{label}.exhaustive"),
        detail=_frozen_json(row["detail"], f"{label}.detail"),
        witness_sha256=_string(
            row["witness_sha256"], f"{label}.witness_sha256"
        ),
    )


def _decode_plan(value: Any, label: str) -> ClosurePlan:
    row = _object(
        value,
        {
            "query_program",
            "policy",
            "snapshot",
            "seeds",
            "atoms",
            "bundles",
            "obligation_results",
            "visited_episode_ids",
            "visited_unit_ids",
            "visited_relation_ids",
            "stopping_reason",
            "complete_claimed",
            "scope_witnesses",
            "direct_chunk_ids",
            "expansion_receipt_sha256",
            "artifact_id",
            "plan_sha256",
        },
        label,
    )
    atoms = tuple(
        _decode_atom(item, f"{label}.atoms[{index}]")
        for index, item in enumerate(_list(row["atoms"], f"{label}.atoms"))
    )
    bundles = tuple(
        _decode_bundle(item, f"{label}.bundles[{index}]")
        for index, item in enumerate(_list(row["bundles"], f"{label}.bundles"))
    )
    return ClosurePlan(
        query_program=_decode_query_program(
            row["query_program"], f"{label}.query_program"
        ),
        policy=_decode_policy(row["policy"], f"{label}.policy"),
        snapshot=_decode_snapshot(row["snapshot"], f"{label}.snapshot"),
        seeds=tuple(
            _decode_seed(item, f"{label}.seeds[{index}]")
            for index, item in enumerate(_list(row["seeds"], f"{label}.seeds"))
        ),
        atoms=atoms,
        bundles=bundles,
        obligation_results=tuple(
            _decode_result(item, f"{label}.obligation_results[{index}]")
            for index, item in enumerate(
                _list(row["obligation_results"], f"{label}.obligation_results")
            )
        ),
        visited_episode_ids=_strings(
            row["visited_episode_ids"], f"{label}.visited_episode_ids"
        ),
        visited_unit_ids=_strings(
            row["visited_unit_ids"], f"{label}.visited_unit_ids"
        ),
        visited_relation_ids=_strings(
            row["visited_relation_ids"], f"{label}.visited_relation_ids"
        ),
        stopping_reason=_string(
            row["stopping_reason"], f"{label}.stopping_reason"
        ),
        complete_claimed=_boolean(
            row["complete_claimed"], f"{label}.complete_claimed"
        ),
        scope_witnesses=tuple(
            _decode_witness(item, f"{label}.scope_witnesses[{index}]")
            for index, item in enumerate(
                _list(row["scope_witnesses"], f"{label}.scope_witnesses")
            )
        ),
        direct_chunk_ids=_strings(
            row["direct_chunk_ids"], f"{label}.direct_chunk_ids"
        ),
        expansion_receipt_sha256=_optional_string(
            row["expansion_receipt_sha256"],
            f"{label}.expansion_receipt_sha256",
        ),
        artifact_id=_optional_string(row["artifact_id"], f"{label}.artifact_id"),
        plan_sha256=_string(row["plan_sha256"], f"{label}.plan_sha256"),
    )


def _decode_receipt(value: Any, label: str) -> ClosureReceipt:
    row = _object(
        value,
        {
            "plan_sha256",
            "context_sha256",
            "selected_bundle_ids",
            "selected_atom_ids",
            "dropped_bundle_reasons",
            "context_token_proxy",
            "max_context_token_proxy",
            "tokenizer_identity",
            "stopping_reason",
            "complete_claimed",
            "retained_request_token_state_bytes",
            "prompt_token_proxy",
            "max_prompt_token_proxy",
            "responder_output_token_reserve",
            "prompt_workspace_token_proxy",
            "base_messages_sha256",
            "evidence_message_role",
            "evidence_prefix_sha256",
            "evidence_suffix_sha256",
            "prompt_messages_sha256",
            "receipt_sha256",
        },
        label,
    )
    dropped = _object(
        row["dropped_bundle_reasons"],
        set(row["dropped_bundle_reasons"])
        if type(row["dropped_bundle_reasons"]) is dict
        else set(),
        f"{label}.dropped_bundle_reasons",
    )
    if any(type(key) is not str or type(item) is not str for key, item in dropped.items()):
        raise TypeError("dropped bundle reasons must contain exact strings")
    _open_mapping_firebreak(dropped, f"{label}.dropped_bundle_reasons")
    return ClosureReceipt(
        plan_sha256=_string(row["plan_sha256"], f"{label}.plan_sha256"),
        context_sha256=_string(row["context_sha256"], f"{label}.context_sha256"),
        selected_bundle_ids=_strings(
            row["selected_bundle_ids"], f"{label}.selected_bundle_ids"
        ),
        selected_atom_ids=_strings(
            row["selected_atom_ids"], f"{label}.selected_atom_ids"
        ),
        dropped_bundle_reasons=dropped,
        context_token_proxy=_integer(
            row["context_token_proxy"], f"{label}.context_token_proxy"
        ),
        max_context_token_proxy=_integer(
            row["max_context_token_proxy"], f"{label}.max_context_token_proxy"
        ),
        tokenizer_identity=_string(
            row["tokenizer_identity"], f"{label}.tokenizer_identity"
        ),
        stopping_reason=_string(
            row["stopping_reason"], f"{label}.stopping_reason"
        ),
        complete_claimed=_boolean(
            row["complete_claimed"], f"{label}.complete_claimed"
        ),
        retained_request_token_state_bytes=_integer(
            row["retained_request_token_state_bytes"],
            f"{label}.retained_request_token_state_bytes",
        ),
        prompt_token_proxy=_optional_integer(
            row["prompt_token_proxy"], f"{label}.prompt_token_proxy"
        ),
        max_prompt_token_proxy=_optional_integer(
            row["max_prompt_token_proxy"], f"{label}.max_prompt_token_proxy"
        ),
        responder_output_token_reserve=_integer(
            row["responder_output_token_reserve"],
            f"{label}.responder_output_token_reserve",
        ),
        prompt_workspace_token_proxy=_optional_integer(
            row["prompt_workspace_token_proxy"],
            f"{label}.prompt_workspace_token_proxy",
        ),
        base_messages_sha256=_optional_string(
            row["base_messages_sha256"], f"{label}.base_messages_sha256"
        ),
        evidence_message_role=_optional_string(
            row["evidence_message_role"], f"{label}.evidence_message_role"
        ),
        evidence_prefix_sha256=_optional_string(
            row["evidence_prefix_sha256"], f"{label}.evidence_prefix_sha256"
        ),
        evidence_suffix_sha256=_optional_string(
            row["evidence_suffix_sha256"], f"{label}.evidence_suffix_sha256"
        ),
        prompt_messages_sha256=_optional_string(
            row["prompt_messages_sha256"], f"{label}.prompt_messages_sha256"
        ),
        receipt_sha256=_string(
            row["receipt_sha256"], f"{label}.receipt_sha256"
        ),
    )


def decode_latent_training_payload(payload: bytes) -> DecodedLatentTrainingPayload:
    """Strictly decode canonical bytes into exact domain object types."""

    if type(payload) is not bytes:
        raise TypeError("payload must be exact bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("cannot decode latent-training payload") from exc
    row = _object(value, {"format", "query", "plan", "packet"}, "payload")
    if row["format"] != LATENT_TRAINING_PAYLOAD_FORMAT:
        raise ValueError("unsupported latent-training payload format")
    query_row = _object(
        row["query"],
        {"question_id", "retrieval_query", "prompt_question"},
        "payload.query",
    )
    question_id = _string(query_row["question_id"], "payload.query.question_id")
    retrieval_query = _string(
        query_row["retrieval_query"], "payload.query.retrieval_query"
    )
    prompt_question = _string(
        query_row["prompt_question"], "payload.query.prompt_question"
    )
    plan = _decode_plan(row["plan"], "payload.plan")
    packet_row = _object(
        row["packet"],
        {"context", "atom_ids", "bundle_ids", "receipt"},
        "payload.packet",
    )
    atom_ids = _strings(packet_row["atom_ids"], "payload.packet.atom_ids")
    bundle_ids = _strings(packet_row["bundle_ids"], "payload.packet.bundle_ids")
    atom_by_id = {atom.atom_id: atom for atom in plan.atoms}
    bundle_by_id = {bundle.bundle_id: bundle for bundle in plan.bundles}
    try:
        atoms = tuple(atom_by_id[atom_id] for atom_id in atom_ids)
        bundles = tuple(bundle_by_id[bundle_id] for bundle_id in bundle_ids)
    except KeyError as exc:
        raise ValueError("packet references an unknown plan row") from exc
    packet = EvidencePacket(
        context=_string(
            packet_row["context"], "payload.packet.context", empty=True
        ),
        atoms=atoms,
        bundles=bundles,
        receipt=_decode_receipt(packet_row["receipt"], "payload.packet.receipt"),
    )
    result = DecodedLatentTrainingPayload(
        question_id=question_id,
        retrieval_query=retrieval_query,
        prompt_question=prompt_question,
        plan=plan,
        packet=packet,
    )
    if encode_latent_training_payload(
        retrieval_query,
        plan,
        packet,
        question_id=question_id,
        prompt_question=prompt_question,
    ) != payload:
        raise ValueError("latent-training payload is not canonical JSON")
    return result


__all__ = [
    "DecodedLatentTrainingPayload",
    "LATENT_TRAINING_PAYLOAD_FORMAT",
    "decode_latent_training_payload",
    "encode_latent_training_payload",
]
