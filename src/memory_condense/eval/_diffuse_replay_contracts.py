"""Canonical receipt projections for a shared-base diffuse replay package."""

from __future__ import annotations

import json
import hashlib
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_condense.domain.discourse import (
    ClosureReceipt,
    ClosureScopeWitness,
    EvidenceSpan,
    identity_sha256,
    make_atom_id,
    make_bundle_id,
)
from memory_condense.eval._diffuse_base_contracts import (
    DiffuseBaseStoreManifest,
    DiffuseDerivedFinalization,
    DiffuseDerivedOrigin,
    DiffuseQueryInputManifest,
)
from memory_condense.eval.diffuse_longmemeval import (
    DIFFUSE_QUERY_RECEIPT_FORMAT,
    LongMemEvalDiffuseQueryReceipt,
)
from memory_condense.eval.schemas import EvalConfig


REPLAY_FORMAT = "memory-condense-longmemeval-shared-base-replay-v1"
REPLAY_MANIFEST_NAME = "replay-manifest.json"
_DIGEST = r"^[0-9a-f]{64}$"
_COMMIT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_MODES = ("fixed_interval", "lexical_embedding", "qwen_head")
_FORBIDDEN_IDENTITY_KEYS = frozenset(
    {
        "content",
        "context",
        "messages",
        "prompt_question",
        "query",
        "question",
        "subject_terms",
        "text",
    }
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_file_sha256(value: object) -> tuple[str, int]:
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _inspect_text_free(value: object, *, path: str = "identity") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _inspect_text_free(child, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            if key.casefold() in _FORBIDDEN_IDENTITY_KEYS:
                raise ValueError(f"{path} admits forbidden raw field {key!r}")
            _inspect_text_free(child, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} contains a non-JSON value")


def _require_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} has an unsupported identity schema")


class CanonicalIdentityBody(_FrozenModel):
    """Immutable exact JSON body with a recomputed identity digest."""

    canonical_identity_json: str = Field(min_length=2)
    identity_sha256: str = Field(pattern=_DIGEST)
    self_hash_field: Literal[
        "receipt_sha256",
        "plan_sha256",
        "snapshot_sha256",
    ] | None = None

    @model_validator(mode="after")
    def _validate_identity(self) -> "CanonicalIdentityBody":
        try:
            payload = json.loads(self.canonical_identity_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("identity body is not JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("identity body must be a JSON object")
        if self.canonical_identity_json != _canonical_json(payload):
            raise ValueError("identity body must use canonical JSON")
        _inspect_text_free(payload)
        unsigned = dict(payload)
        if self.self_hash_field is not None:
            if unsigned.pop(self.self_hash_field, None) != self.identity_sha256:
                raise ValueError("identity body self hash differs from its digest")
        if identity_sha256(unsigned) != self.identity_sha256:
            raise ValueError("identity body digest does not match")
        return self

    @classmethod
    def seal(
        cls,
        payload: dict[str, object],
        *,
        identity_sha256_value: str,
        self_hash_field: Literal[
            "receipt_sha256", "plan_sha256", "snapshot_sha256"
        ] | None = None,
    ) -> "CanonicalIdentityBody":
        return cls(
            canonical_identity_json=_canonical_json(payload),
            identity_sha256=identity_sha256_value,
            self_hash_field=self_hash_field,
        )


class ReplayExecutionIdentity(_FrozenModel):
    certification: Literal["tracked_launcher_v1"] = "tracked_launcher_v1"
    launcher_sha256: str = Field(pattern=_DIGEST)
    source_commit: str = Field(min_length=40, max_length=64)
    tracked_worktree_clean: Literal[True] = True

    @model_validator(mode="after")
    def _commit_is_hex(self) -> "ReplayExecutionIdentity":
        if _COMMIT.fullmatch(self.source_commit) is None:
            raise ValueError("source_commit must be a 40- or 64-digit hex ID")
        return self


class ReplayMessageDescriptor(_FrozenModel):
    ordinal: int = Field(ge=0)
    role: str = Field(min_length=1)
    content_sha256: str = Field(pattern=_DIGEST)
    content_bytes: int = Field(ge=0)
    content_characters: int = Field(ge=0)
    content_token_proxy: int = Field(ge=0)
    chat_framing_token_proxy: Literal[8] = 8


class ReplayEvidenceCoordinate(_FrozenModel):
    atom_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=1)
    quote_sha256: str = Field(pattern=_DIGEST)
    ordinal: int = Field(ge=0)
    source_id: str | None = None
    turn_start_char: int = Field(ge=0)
    turn_id: str | None = None
    role: Literal["user", "assistant", "system"] | None = None
    created_at: str | None = None
    label: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered_span(self) -> "ReplayEvidenceCoordinate":
        if self.end_char <= self.start_char:
            raise ValueError("evidence coordinate span is empty")
        span = EvidenceSpan(**self.identity_payload())
        if make_atom_id(span) != self.atom_id:
            raise ValueError("evidence atom ID does not match its source span")
        return self

    def identity_payload(self) -> dict[str, object]:
        return self.model_dump(mode="python", exclude={"atom_id", "label"})


class ReplayClosureAtom(_FrozenModel):
    coordinate: ReplayEvidenceCoordinate
    text_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def _same_quote(self) -> "ReplayClosureAtom":
        if self.text_sha256 != self.coordinate.quote_sha256:
            raise ValueError("closure atom text and span digests disagree")
        return self


class ReplayEvidenceBundle(_FrozenModel):
    bundle_id: str = Field(min_length=1)
    atom_ids: tuple[str, ...]
    obligation_ids: tuple[str, ...]
    unit_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    required: bool
    utility: float

    @model_validator(mode="after")
    def _bundle_is_closed(self) -> "ReplayEvidenceBundle":
        if not self.atom_ids or not math.isfinite(self.utility):
            raise ValueError("bundle needs atoms and finite utility")
        for name in ("atom_ids", "obligation_ids", "unit_ids", "relation_ids"):
            values = getattr(self, name)
            if any(not value for value in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique non-empty IDs")
        if make_bundle_id(
            atom_ids=self.atom_ids,
            obligation_ids=self.obligation_ids,
            unit_ids=self.unit_ids,
            relation_ids=self.relation_ids,
        ) != self.bundle_id:
            raise ValueError("bundle ID does not match its projections")
        return self

    def identity_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class ReplayEpisodeSeed(_FrozenModel):
    episode_id: str = Field(min_length=1)
    anchor_chunk_id: str = Field(min_length=1)
    score: float
    route: str = Field(min_length=1)
    path: tuple[str, ...]


class ReplayObligationResult(_FrozenModel):
    obligation_id: str = Field(min_length=1)
    status: Literal["satisfied", "not_found", "conflicted", "budget_impossible"]
    unit_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    bundle_ids: tuple[str, ...]
    reason: str | None = None


class ReplayScopeWitness(_FrozenModel):
    kind: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    requested_limit: int | None = Field(default=None, ge=0)
    returned_count: int = Field(ge=0)
    exhaustive: bool
    canonical_detail_json: str
    witness_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def _validate_witness(self) -> "ReplayScopeWitness":
        try:
            detail = json.loads(self.canonical_detail_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("scope witness detail is not JSON") from exc
        if not isinstance(detail, dict) or _canonical_json(detail) != (
            self.canonical_detail_json
        ):
            raise ValueError("scope witness detail must be a canonical object")
        _inspect_text_free(detail, path="scope_witness.detail")
        authoritative = ClosureScopeWitness(
            kind=self.kind,
            subject_id=self.subject_id,
            requested_limit=self.requested_limit,
            returned_count=self.returned_count,
            exhaustive=self.exhaustive,
            detail=detail,
            witness_sha256=self.witness_sha256,
        )
        if authoritative.witness_sha256 != self.witness_sha256:
            raise ValueError("scope witness digest changed")
        return self

    def identity_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "subject_id": self.subject_id,
            "requested_limit": self.requested_limit,
            "returned_count": self.returned_count,
            "exhaustive": self.exhaustive,
            "detail": json.loads(self.canonical_detail_json),
            "witness_sha256": self.witness_sha256,
        }


class ReplayClosurePlanProjection(_FrozenModel):
    query_program_sha256: str = Field(pattern=_DIGEST)
    policy_sha256: str = Field(pattern=_DIGEST)
    snapshot_sha256: str = Field(pattern=_DIGEST)
    seeds: tuple[ReplayEpisodeSeed, ...]
    atoms: tuple[ReplayClosureAtom, ...]
    bundles: tuple[ReplayEvidenceBundle, ...]
    obligation_results: tuple[ReplayObligationResult, ...]
    visited_episode_ids: tuple[str, ...]
    visited_unit_ids: tuple[str, ...]
    visited_relation_ids: tuple[str, ...]
    scope_witnesses: tuple[ReplayScopeWitness, ...]
    direct_chunk_ids: tuple[str, ...]
    expansion_receipt_sha256: str | None = Field(default=None, pattern=_DIGEST)
    artifact_id: str | None = None
    stopping_reason: Literal[
        "complete",
        "frontier_exhausted",
        "budget_exhausted",
        "budget_impossible",
        "workspace_cap",
        "conflicted",
        "not_found",
    ]
    complete_claimed: bool
    plan_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def _validate_plan_projection(self) -> "ReplayClosurePlanProjection":
        if identity_sha256(self.identity_payload(include_sha=False)) != self.plan_sha256:
            raise ValueError("closure plan projection digest does not match")
        return self

    def identity_payload(self, *, include_sha: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "query_program_sha256": self.query_program_sha256,
            "policy_sha256": self.policy_sha256,
            "snapshot_sha256": self.snapshot_sha256,
            "seeds": [item.model_dump(mode="json") for item in self.seeds],
            "atoms": [
                {
                    "atom_id": item.coordinate.atom_id,
                    "span": item.coordinate.identity_payload(),
                    "text_sha256": item.text_sha256,
                    "label": item.coordinate.label,
                    "role": item.coordinate.role,
                    "created_at": item.coordinate.created_at,
                }
                for item in self.atoms
            ],
            "bundles": [item.identity_payload() for item in self.bundles],
            "obligation_results": [
                item.model_dump(mode="json") for item in self.obligation_results
            ],
            "visited_episode_ids": list(self.visited_episode_ids),
            "visited_unit_ids": list(self.visited_unit_ids),
            "visited_relation_ids": list(self.visited_relation_ids),
            "scope_witnesses": [
                item.identity_payload() for item in self.scope_witnesses
            ],
            "direct_chunk_ids": list(self.direct_chunk_ids),
            "expansion_receipt_sha256": self.expansion_receipt_sha256,
            "artifact_id": self.artifact_id,
            "stopping_reason": self.stopping_reason,
            "complete_claimed": self.complete_claimed,
        }
        if include_sha:
            payload["plan_sha256"] = self.plan_sha256
        return payload


class ReplayDroppedBundleReason(_FrozenModel):
    bundle_id: str = Field(min_length=1)
    reason: str


class ReplayClosureReceipt(_FrozenModel):
    plan_sha256: str = Field(pattern=_DIGEST)
    context_sha256: str = Field(pattern=_DIGEST)
    selected_bundle_ids: tuple[str, ...]
    selected_atom_ids: tuple[str, ...]
    dropped_bundle_reasons: tuple[ReplayDroppedBundleReason, ...]
    context_token_proxy: int = Field(ge=0)
    max_context_token_proxy: int = Field(ge=0)
    tokenizer_identity: str = Field(min_length=1)
    stopping_reason: Literal[
        "complete",
        "frontier_exhausted",
        "budget_exhausted",
        "budget_impossible",
        "workspace_cap",
        "conflicted",
        "not_found",
    ]
    complete_claimed: bool
    retained_request_token_state_bytes: Literal[0] = 0
    prompt_token_proxy: int | None = Field(default=None, ge=0)
    max_prompt_token_proxy: int | None = Field(default=None, ge=0)
    responder_output_token_reserve: int = Field(ge=0)
    prompt_workspace_token_proxy: int | None = Field(default=None, ge=0)
    base_messages_sha256: str | None = Field(default=None, pattern=_DIGEST)
    evidence_message_role: str | None = None
    evidence_prefix_sha256: str | None = Field(default=None, pattern=_DIGEST)
    evidence_suffix_sha256: str | None = Field(default=None, pattern=_DIGEST)
    prompt_messages_sha256: str | None = Field(default=None, pattern=_DIGEST)
    receipt_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def _validate_closure_receipt(self) -> "ReplayClosureReceipt":
        reasons = {item.bundle_id: item.reason for item in self.dropped_bundle_reasons}
        if len(reasons) != len(self.dropped_bundle_reasons):
            raise ValueError("dropped bundle reasons must be unique")
        authoritative = ClosureReceipt(
            plan_sha256=self.plan_sha256,
            context_sha256=self.context_sha256,
            selected_bundle_ids=self.selected_bundle_ids,
            selected_atom_ids=self.selected_atom_ids,
            dropped_bundle_reasons=reasons,
            context_token_proxy=self.context_token_proxy,
            max_context_token_proxy=self.max_context_token_proxy,
            tokenizer_identity=self.tokenizer_identity,
            stopping_reason=self.stopping_reason,
            complete_claimed=self.complete_claimed,
            retained_request_token_state_bytes=0,
            prompt_token_proxy=self.prompt_token_proxy,
            max_prompt_token_proxy=self.max_prompt_token_proxy,
            responder_output_token_reserve=self.responder_output_token_reserve,
            prompt_workspace_token_proxy=self.prompt_workspace_token_proxy,
            base_messages_sha256=self.base_messages_sha256,
            evidence_message_role=self.evidence_message_role,
            evidence_prefix_sha256=self.evidence_prefix_sha256,
            evidence_suffix_sha256=self.evidence_suffix_sha256,
            prompt_messages_sha256=self.prompt_messages_sha256,
            receipt_sha256=self.receipt_sha256,
        )
        if authoritative.receipt_sha256 != self.receipt_sha256:
            raise ValueError("closure receipt digest changed")
        return self


class ReplayLongMemEvalQueryReceipt(_FrozenModel):
    artifact_id: str = Field(min_length=1)
    snapshot_sha256: str = Field(pattern=_DIGEST)
    anchor_sequence_sha256: str = Field(pattern=_DIGEST)
    input_anchor_chunk_ids: tuple[str, ...]
    episode_policy_sha256: str = Field(pattern=_DIGEST)
    expansion_receipt_sha256: str = Field(pattern=_DIGEST)
    representative_receipt_sha256: str | None = Field(default=None, pattern=_DIGEST)
    representative_scope_exhaustive: bool | None = None
    representative_runtime_binding_certified: bool | None = None
    representative_returned_plan_transformer_state_bytes: int | None = Field(
        default=None, ge=0
    )
    combined_expansion_sha256: str = Field(pattern=_DIGEST)
    representative_seed_episode_ids: tuple[str, ...]
    truncated_episode_ids: tuple[str, ...]
    truncated_direct_chunk_ids: tuple[str, ...]
    expansion_exhaustive: bool
    query_program_sha256: str = Field(pattern=_DIGEST)
    retrieval_query_sha256: str = Field(pattern=_DIGEST)
    prompt_question_sha256: str = Field(pattern=_DIGEST)
    closure_policy_sha256: str = Field(pattern=_DIGEST)
    closure_plan_sha256: str = Field(pattern=_DIGEST)
    closure_stopping_reason: str = Field(min_length=1)
    closure_complete_claimed: bool
    scope_witness_sha256s: tuple[str, ...]
    closure_scope_exhaustive: bool
    packet_receipt_sha256: str = Field(pattern=_DIGEST)
    context_sha256: str = Field(pattern=_DIGEST)
    evidence_coordinates_sha256: str = Field(pattern=_DIGEST)
    prompt_messages_sha256: str = Field(pattern=_DIGEST)
    prompt_token_proxy: int = Field(ge=0)
    max_input_prompt_token_proxy: int = Field(ge=0)
    responder_output_token_reserve: int = Field(ge=0)
    prompt_workspace_token_proxy: int = Field(ge=0)
    max_prompt_workspace_token_proxy: int = Field(ge=0)
    packet_retained_request_token_state_bytes: Literal[0] = 0
    store_retained_request_token_state_bytes: Literal[0] | None = None
    format: Literal[DIFFUSE_QUERY_RECEIPT_FORMAT] = DIFFUSE_QUERY_RECEIPT_FORMAT
    receipt_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def _validate_query_receipt(self) -> "ReplayLongMemEvalQueryReceipt":
        authoritative = LongMemEvalDiffuseQueryReceipt(
            **self.model_dump(mode="python")
        )
        if authoritative.receipt_sha256 != self.receipt_sha256:
            raise ValueError("LongMemEval query receipt digest changed")
        return self


class DiffuseReplayQueryRecord(_FrozenModel):
    question_ordinal: int = Field(ge=0)
    question_id_sha256: str = Field(pattern=_DIGEST)
    question_probe_sha256: str = Field(pattern=_DIGEST)
    frozen_input: CanonicalIdentityBody
    frozen_anchor_projection_sha256: str = Field(pattern=_DIGEST)
    legacy_input: CanonicalIdentityBody
    analysis_query: CanonicalIdentityBody
    source_scope: CanonicalIdentityBody
    direct_expansion: CanonicalIdentityBody
    representative_expansion: CanonicalIdentityBody
    query_receipt: ReplayLongMemEvalQueryReceipt
    closure_plan: ReplayClosurePlanProjection
    closure_receipt: ReplayClosureReceipt
    evidence_coordinates: tuple[ReplayEvidenceCoordinate, ...]
    selected_bundles: tuple[ReplayEvidenceBundle, ...]
    messages: tuple[ReplayMessageDescriptor, ...]
    fixed_chat_framing_token_proxy: Literal[8] = 8
    matched_source_scope_identity_sha256: str = Field(pattern=_DIGEST)
    packet_identity_sha256: str = Field(pattern=_DIGEST)
    record_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def _validate_record(self) -> "DiffuseReplayQueryRecord":
        if self.query_receipt.packet_receipt_sha256 != self.closure_receipt.receipt_sha256:
            raise ValueError("query and packet receipts disagree")
        if self.query_receipt.closure_plan_sha256 != self.closure_plan.plan_sha256:
            raise ValueError("query and closure plan disagree")
        if self.closure_receipt.plan_sha256 != self.closure_plan.plan_sha256:
            raise ValueError("packet and closure plan disagree")
        if (
            self.query_receipt.query_program_sha256
            != self.closure_plan.query_program_sha256
            or self.query_receipt.closure_policy_sha256
            != self.closure_plan.policy_sha256
            or self.query_receipt.snapshot_sha256
            != self.closure_plan.snapshot_sha256
            or self.query_receipt.closure_stopping_reason
            != self.closure_plan.stopping_reason
            or self.query_receipt.closure_complete_claimed
            != self.closure_plan.complete_claimed
            or self.query_receipt.scope_witness_sha256s
            != tuple(
                item.witness_sha256 for item in self.closure_plan.scope_witnesses
            )
        ):
            raise ValueError("query receipt and closure projection disagree")
        if self.query_receipt.context_sha256 != self.closure_receipt.context_sha256:
            raise ValueError("query and packet context commitments disagree")
        coordinate_payload = [
            {
                "atom_id": item.atom_id,
                **item.identity_payload(),
                "label": item.label,
            }
            for item in self.evidence_coordinates
        ]
        if identity_sha256(coordinate_payload) != (
            self.query_receipt.evidence_coordinates_sha256
        ):
            raise ValueError("evidence coordinate commitment changed")
        if tuple(item.atom_id for item in self.evidence_coordinates) != (
            self.closure_receipt.selected_atom_ids
        ):
            raise ValueError("selected atom coordinates changed")
        if tuple(item.bundle_id for item in self.selected_bundles) != (
            self.closure_receipt.selected_bundle_ids
        ):
            raise ValueError("selected bundle descriptors changed")
        plan_atoms = {
            item.coordinate.atom_id: item for item in self.closure_plan.atoms
        }
        if any(
            plan_atoms.get(item.atom_id) is None
            or plan_atoms[item.atom_id].coordinate != item
            for item in self.evidence_coordinates
        ):
            raise ValueError("selected evidence coordinates differ from the plan")
        plan_bundles = {
            item.bundle_id: item for item in self.closure_plan.bundles
        }
        if any(
            plan_bundles.get(item.bundle_id) != item
            or any(atom_id not in plan_atoms for atom_id in item.atom_ids)
            for item in self.selected_bundles
        ):
            raise ValueError("selected bundle projections differ from the plan")
        if self.direct_expansion.identity_sha256 != (
            self.query_receipt.expansion_receipt_sha256
        ) or self.representative_expansion.identity_sha256 != (
            self.query_receipt.representative_receipt_sha256
        ):
            raise ValueError("query receipt changed an expansion plan")
        direct = json.loads(self.direct_expansion.canonical_identity_json)
        representative = json.loads(
            self.representative_expansion.canonical_identity_json
        )
        _require_keys(
            direct,
            {
                "policy_sha256", "seeds", "direct_fallbacks",
                "truncated_episode_ids", "truncated_direct_chunk_ids",
                "receipt_sha256",
            },
            "direct expansion",
        )
        _require_keys(
            representative,
            {
                "artifact_id", "policy_sha256", "query_sha256",
                "query_input_sha256", "linker_identity_sha256",
                "runtime_binding_certified", "source_scope_receipt_sha256",
                "source_universe_exhaustive", "source_scans",
                "candidate_witnesses", "seeds", "truncated_source_ids",
                "truncated_episode_ids", "unavailable_episode_ids", "passes",
                "max_workspace_candidates", "max_workspace_tokens",
                "total_candidate_inspections",
                "returned_plan_transformer_state_bytes", "receipt_sha256",
            },
            "representative expansion",
        )
        direct_seeds = tuple(
            ReplayEpisodeSeed.model_validate(item)
            for item in direct.get("seeds", ())
        )
        representative_seeds = tuple(
            ReplayEpisodeSeed.model_validate(item)
            for item in representative.get("seeds", ())
        )
        selected: dict[str, ReplayEpisodeSeed] = {}
        for seed in (*direct_seeds, *representative_seeds):
            prior = selected.get(seed.episode_id)
            if prior is None or (
                -seed.score, seed.anchor_chunk_id, seed.route, seed.path
            ) < (-prior.score, prior.anchor_chunk_id, prior.route, prior.path):
                selected[seed.episode_id] = seed
        combined = tuple(
            sorted(
                selected.values(),
                key=lambda seed: (
                    -seed.score, seed.episode_id, seed.anchor_chunk_id,
                    seed.route, seed.path,
                ),
            )
        )
        direct_fallbacks = tuple(direct.get("direct_fallbacks", ()))
        if any(
            not isinstance(item, dict)
            or set(item) != {"chunk_id", "score", "route", "failure_code", "path"}
            for item in direct_fallbacks
        ):
            raise ValueError("direct expansion fallback schema changed")
        direct_chunk_ids = tuple(item["chunk_id"] for item in direct_fallbacks)
        if len(set(direct_chunk_ids)) != len(direct_chunk_ids):
            raise ValueError("direct expansion fallback chunks must be unique")
        canonical_direct_chunk_ids = tuple(sorted(set(direct_chunk_ids)))
        combined_sha256 = identity_sha256(
            {
                "direct_expansion_receipt_sha256": (
                    self.direct_expansion.identity_sha256
                ),
                "representative_expansion_receipt_sha256": (
                    self.representative_expansion.identity_sha256
                ),
                "seeds": [item.model_dump(mode="json") for item in combined],
                "direct_chunk_ids": list(direct_chunk_ids),
            }
        )
        if (
            direct.get("policy_sha256")
            != self.query_receipt.episode_policy_sha256
            or combined != self.closure_plan.seeds
            or canonical_direct_chunk_ids != self.closure_plan.direct_chunk_ids
            or tuple(direct.get("truncated_episode_ids", ()))
            != self.query_receipt.truncated_episode_ids
            or tuple(direct.get("truncated_direct_chunk_ids", ()))
            != self.query_receipt.truncated_direct_chunk_ids
            or tuple(seed.episode_id for seed in representative_seeds)
            != self.query_receipt.representative_seed_episode_ids
            or representative.get("artifact_id")
            != self.query_receipt.artifact_id
            or representative.get("query_sha256")
            != self.query_receipt.retrieval_query_sha256
            or representative.get("source_scope_receipt_sha256")
            != self.source_scope.identity_sha256
            or representative.get("runtime_binding_certified") is not True
            or representative.get("returned_plan_transformer_state_bytes") != 0
            or combined_sha256 != self.query_receipt.combined_expansion_sha256
        ):
            raise ValueError("expansion plans do not produce the closure seeds")
        source_scope = json.loads(self.source_scope.canonical_identity_json)
        matched_scope = {
            name: source_scope[name]
            for name in (
                "source_revision",
                "source_content_sha256",
                "query_sha256",
                "router_policy_sha256",
                "universe_source_ids",
                "candidates",
                "truncated_source_ids",
                "universe_enumerated",
            )
        }
        if identity_sha256(matched_scope) != (
            self.matched_source_scope_identity_sha256
        ):
            raise ValueError("matched source-scope identity changed")
        analysis = json.loads(self.analysis_query.canonical_identity_json)
        if (
            analysis.get("legacy_input_receipt_sha256")
            != self.legacy_input.identity_sha256
            or analysis.get("diffuse_query_receipt_sha256")
            != self.query_receipt.receipt_sha256
        ):
            raise ValueError("analysis query does not bind its nested receipts")
        if tuple(item.ordinal for item in self.messages) != tuple(range(len(self.messages))):
            raise ValueError("message descriptors must be contiguous")
        prompt_proxy = (
            sum(
                item.content_token_proxy + item.chat_framing_token_proxy
                for item in self.messages
            )
            + self.fixed_chat_framing_token_proxy
        )
        if prompt_proxy != self.query_receipt.prompt_token_proxy:
            raise ValueError("message token descriptors changed")
        if tuple(item.role for item in self.messages) != ("system", "user"):
            raise ValueError("replay requires the authoritative two-message prompt")
        packet_identity = identity_sha256(
            {
                "receipt_sha256": self.closure_receipt.receipt_sha256,
                "context_sha256": self.closure_receipt.context_sha256,
                "atom_ids": [item.atom_id for item in self.evidence_coordinates],
                "bundle_ids": [item.bundle_id for item in self.selected_bundles],
                "tokenizer": self.closure_receipt.tokenizer_identity,
            }
        )
        if packet_identity != self.packet_identity_sha256:
            raise ValueError("packet identity commitment changed")
        if (
            self.closure_plan.artifact_id != self.query_receipt.artifact_id
            or self.closure_plan.expansion_receipt_sha256
            != self.query_receipt.combined_expansion_sha256
            or self.query_receipt.closure_scope_exhaustive
            != bool(
                self.closure_plan.scope_witnesses
                and all(item.exhaustive for item in self.closure_plan.scope_witnesses)
            )
            or self.closure_receipt.prompt_messages_sha256
            != self.query_receipt.prompt_messages_sha256
            or self.closure_receipt.prompt_token_proxy
            != self.query_receipt.prompt_token_proxy
            or self.closure_receipt.responder_output_token_reserve
            != self.query_receipt.responder_output_token_reserve
            or self.closure_receipt.prompt_workspace_token_proxy
            != self.query_receipt.prompt_workspace_token_proxy
            or self.closure_receipt.max_prompt_token_proxy
            != self.query_receipt.max_prompt_workspace_token_proxy
        ):
            raise ValueError("packet, plan, and query receipt fields disagree")
        legacy = json.loads(self.legacy_input.canonical_identity_json)
        frozen = json.loads(self.frozen_input.canonical_identity_json)
        _require_keys(
            frozen,
            {
                "format", "query_sha256", "retrieval_policy_sha256",
                "anchor_sequence_sha256", "lexical_sources",
                "universe_source_ids", "source_streams_sha256",
                "receipt_sha256",
            },
            "frozen input",
        )
        _require_keys(
            legacy,
            {
                "format", "artifact_id", "query_sha256",
                "retrieval_policy_sha256", "anchor_sequence_sha256",
                "source_candidate_sequence_sha256",
                "source_candidate_scope_receipt_sha256", "anchor_chunk_ids",
                "source_candidate_ids", "receipt_sha256",
            },
            "legacy input",
        )
        if (
            legacy.get("source_candidate_scope_receipt_sha256")
            != self.source_scope.identity_sha256
            or legacy.get("artifact_id") != self.query_receipt.artifact_id
            or legacy.get("query_sha256")
            != self.query_receipt.retrieval_query_sha256
            or tuple(legacy.get("anchor_chunk_ids", ()))
            != self.query_receipt.input_anchor_chunk_ids
            or tuple(legacy.get("source_candidate_ids", ()))
            != tuple(item["source_id"] for item in source_scope["candidates"])
            or legacy.get("source_candidate_sequence_sha256")
            != identity_sha256(source_scope["candidates"])
        ):
            raise ValueError("legacy input and query receipt disagree")
        if (
            frozen.get("query_sha256")
            != self.query_receipt.retrieval_query_sha256
            or frozen.get("retrieval_policy_sha256")
            != legacy.get("retrieval_policy_sha256")
        ):
            raise ValueError("frozen and artifact-bound query policies disagree")
        if self.frozen_anchor_projection_sha256 != (
            legacy.get("anchor_sequence_sha256")
        ):
            raise ValueError("text-free frozen anchor projection changed")
        if (
            source_scope.get("artifact_id") != self.query_receipt.artifact_id
            or source_scope.get("snapshot_sha256")
            != self.query_receipt.snapshot_sha256
            or source_scope.get("query_sha256")
            != self.query_receipt.retrieval_query_sha256
        ):
            raise ValueError("source scope and query receipt disagree")
        if (
            tuple(frozen.get("universe_source_ids", ()))
            != tuple(source_scope.get("universe_source_ids", ()))
        ):
            raise ValueError("frozen input and source scope changed their universe")
        if representative.get("source_scope_receipt_sha256") != (
            self.source_scope.identity_sha256
        ):
            raise ValueError("representative expansion changed its source scope")
        if (
            analysis.get("question_probe_sha256")
            != self.question_probe_sha256
            or analysis.get("artifact_id") != self.query_receipt.artifact_id
            or analysis.get("snapshot_sha256")
            != self.query_receipt.snapshot_sha256
        ):
            raise ValueError("analysis query changed probe or graph coordinates")
        unsigned = self.model_dump(mode="json", exclude={"record_sha256"})
        if identity_sha256(unsigned) != self.record_sha256:
            raise ValueError("query replay record digest changed")
        return self


class DiffuseReplayArmRecord(_FrozenModel):
    boundary_mode: Literal["fixed_interval", "lexical_embedding", "qwen_head"]
    arm_identity: CanonicalIdentityBody
    episode_policy: CanonicalIdentityBody
    closure_policy: CanonicalIdentityBody
    derived_origin_receipt_sha256: str = Field(pattern=_DIGEST)
    derived_origin: DiffuseDerivedOrigin
    finalization: DiffuseDerivedFinalization
    final_snapshot: CanonicalIdentityBody
    compilation: CanonicalIdentityBody
    retrieval_phase: CanonicalIdentityBody
    runtime_result: CanonicalIdentityBody
    queries: tuple[DiffuseReplayQueryRecord, ...]
    record_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def _validate_arm(self) -> "DiffuseReplayArmRecord":
        if self.arm_identity.identity_sha256 != self.finalization.arm_sha256:
            raise ValueError("arm identity and finalization disagree")
        if (
            self.derived_origin_receipt_sha256
            != self.finalization.origin_receipt_sha256
            or self.derived_origin.receipt_sha256
            != self.derived_origin_receipt_sha256
            or self.boundary_mode != self.finalization.arm_id
            or self.derived_origin.arm_id != self.boundary_mode
            or self.derived_origin.arm_sha256 != self.arm_identity.identity_sha256
        ):
            raise ValueError("arm origin, mode, and finalization disagree")
        if identity_sha256(
            self.derived_origin.model_dump(
                mode="json", exclude={"receipt_sha256"}
            )
        ) != self.derived_origin.receipt_sha256 or identity_sha256(
            self.finalization.model_dump(
                mode="json", exclude={"receipt_sha256"}
            )
        ) != self.finalization.receipt_sha256:
            raise ValueError("derived origin or finalization self hash changed")
        arm_payload = json.loads(self.arm_identity.canonical_identity_json)
        if (
            arm_payload.get("arm_id") != self.boundary_mode
            or arm_payload.get("episode_policy_sha256")
            != self.episode_policy.identity_sha256
            or arm_payload.get("closure_policy_sha256")
            != self.closure_policy.identity_sha256
            or any(
                item.closure_plan.policy_sha256
                != self.closure_policy.identity_sha256
                for item in self.queries
            )
        ):
            raise ValueError("arm policy bodies differ from their declared hashes")
        if any(
            (
                json.loads(item.analysis_query.canonical_identity_json).get(
                    "analysis_arm_sha256"
                )
                != self.arm_identity.identity_sha256
                or json.loads(item.analysis_query.canonical_identity_json).get(
                    "compilation_receipt_sha256"
                )
                != self.compilation.identity_sha256
            )
            for item in self.queries
        ):
            raise ValueError("analysis queries belong to another arm")
        if self.compilation.identity_sha256 != (
            self.finalization.compilation_receipt_sha256
        ) or self.retrieval_phase.identity_sha256 != (
            self.finalization.retrieval_phase_receipt_sha256
        ):
            raise ValueError("arm finalization does not bind its phases")
        if self.final_snapshot.identity_sha256 != self.finalization.final_snapshot_sha256:
            raise ValueError("arm final snapshot changed")
        unsigned = self.model_dump(mode="json", exclude={"record_sha256"})
        if identity_sha256(unsigned) != self.record_sha256:
            raise ValueError("arm replay record digest changed")
        return self


class DiffuseLongMemEvalReplayReceipt(_FrozenModel):
    format: Literal[REPLAY_FORMAT] = REPLAY_FORMAT
    sample_id_sha256: str = Field(pattern=_DIGEST)
    treatment_identity_sha256: str = Field(pattern=_DIGEST)
    base_manifest_file_sha256: str = Field(pattern=_DIGEST)
    base_manifest: DiffuseBaseStoreManifest
    query_manifest_file_sha256: str = Field(pattern=_DIGEST)
    query_manifest: DiffuseQueryInputManifest
    verified_base_provider_identity: CanonicalIdentityBody
    eval_config: CanonicalIdentityBody
    retrieval_policy: CanonicalIdentityBody
    evaluation_policy: CanonicalIdentityBody
    runtime_binding: CanonicalIdentityBody
    matched_phase_suite: CanonicalIdentityBody
    matched_runtime_suite: CanonicalIdentityBody
    execution_identity: ReplayExecutionIdentity | None = None
    launcher_binding_certified: bool
    arms: tuple[DiffuseReplayArmRecord, ...]
    files: tuple["ReplayFileIdentity", ...]
    qa_responder_or_judge_calls: Literal[0] = 0
    retrieval_input_schema_contains_gold_fields: Literal[False] = False
    treatment_population_membership_certified: Literal[False] = False
    provider_transports_invoked_by_runner: Literal[0] = 0
    receipt_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def _validate_replay(self) -> "DiffuseLongMemEvalReplayReceipt":
        if tuple(item.boundary_mode for item in self.arms) != _MODES:
            raise ValueError("replay requires the three canonical arms in order")
        if self.launcher_binding_certified != (self.execution_identity is not None):
            raise ValueError("launcher certification and identity disagree")
        if self.query_manifest.base_store_key != self.base_manifest.base_store_key:
            raise ValueError("query manifest belongs to another base")
        if (
            self.sample_id_sha256 != self.base_manifest.sample_id_sha256
            or self.treatment_identity_sha256
            != self.query_manifest.treatment_identity_sha256
            or self.query_manifest.base_artifact_sha256
            != self.base_manifest.artifact_sha256
            or self.query_manifest.base_manifest_sha256
            != self.base_manifest_file_sha256
            or self.query_manifest.config_identity.retrieval_policy_sha256
            != self.retrieval_policy.identity_sha256
        ):
            raise ValueError("top-level base/query identities disagree")
        if (
            self.query_manifest.database_sha256
            != self.base_manifest.database_sha256
            or self.query_manifest.index_sha256 != self.base_manifest.index_sha256
            or self.query_manifest.source_streams_sha256
            != self.base_manifest.source_streams_sha256
            or self.query_manifest.turn_sequence_sha256
            != self.base_manifest.turn_sequence_sha256
            or self.query_manifest.chunk_sequence_sha256
            != self.base_manifest.chunk_sequence_sha256
            or self.query_manifest.embedding_identity_sha256
            != self.base_manifest.embedding_identity_sha256
            or identity_sha256(
                self.base_manifest.chunker_identity.model_dump(mode="json")
            )
            != self.base_manifest.chunker_identity_sha256
            or identity_sha256(
                self.base_manifest.embedding_identity.model_dump(mode="json")
            )
            != self.base_manifest.embedding_identity_sha256
            or identity_sha256(
                self.base_manifest.build_runtime_identity.model_dump(mode="json")
            )
            != self.base_manifest.build_runtime_identity_sha256
            or identity_sha256(
                self.query_manifest.treatment_identity.model_dump(mode="json")
            )
            != self.query_manifest.treatment_identity_sha256
            or identity_sha256(
                self.query_manifest.config_identity.model_dump(mode="json")
            )
            != self.query_manifest.config_identity_sha256
        ):
            raise ValueError("base/query nested lineage identities disagree")
        base_file_sha, _base_file_bytes = _canonical_file_sha256(
            self.base_manifest.model_dump(mode="json")
        )
        query_file_sha, _query_file_bytes = _canonical_file_sha256(
            self.query_manifest.model_dump(mode="json")
        )
        if (
            base_file_sha != self.base_manifest_file_sha256
            or query_file_sha != self.query_manifest_file_sha256
            or identity_sha256(
                self.base_manifest.model_dump(
                    mode="json", exclude={"artifact_sha256"}
                )
            )
            != self.base_manifest.artifact_sha256
            or identity_sha256(
                self.query_manifest.model_dump(
                    mode="json", exclude={"artifact_sha256"}
                )
            )
            != self.query_manifest.artifact_sha256
        ):
            raise ValueError("base or query manifest self identity changed")
        eval_payload = json.loads(self.eval_config.canonical_identity_json)
        retrieval_payload = json.loads(
            self.retrieval_policy.canonical_identity_json
        )
        try:
            eval_config = EvalConfig.model_validate(eval_payload)
        except Exception as exc:
            raise ValueError("evaluation config body is not authoritative") from exc
        exact_evaluation = {
            "chunker": eval_config.chunker.model_dump(mode="json"),
            "retrieval": eval_config.retrieval.model_dump(mode="json"),
            "max_prompt_tokens": eval_config.max_prompt_tokens,
        }
        if (
            eval_config.model_dump(mode="json") != eval_payload
            or
            eval_payload.get("retrieval") != retrieval_payload
            or eval_payload.get("chunker")
            != self.base_manifest.chunker_identity.model_dump(mode="json")
            or json.loads(self.evaluation_policy.canonical_identity_json)
            != exact_evaluation
        ):
            raise ValueError("full evaluation config differs from base policies")
        if any(
            item.finalization.index_sha256 != self.base_manifest.index_sha256
            or item.finalization.index_bytes != self.base_manifest.index_bytes
            for item in self.arms
        ):
            raise ValueError("derived HNSW identities differ from the shared base")
        if len({item.derived_origin_receipt_sha256 for item in self.arms}) != 3:
            raise ValueError("derived clone origins must be distinct")
        question_counts = {len(item.queries) for item in self.arms}
        if question_counts != {self.query_manifest.query_count}:
            raise ValueError("replay arms do not cover the frozen query set")
        for arm in self.arms:
            if tuple(item.question_ordinal for item in arm.queries) != tuple(
                range(self.query_manifest.query_count)
            ):
                raise ValueError("arm query ordinals are not exact")
            if len({item.question_id_sha256 for item in arm.queries}) != len(
                arm.queries
            ) or len({item.question_probe_sha256 for item in arm.queries}) != len(
                arm.queries
            ):
                raise ValueError("arm query identities are not unique")
        coordinates = tuple(
            (
                item.question_ordinal,
                item.question_id_sha256,
                item.question_probe_sha256,
                item.frozen_input.identity_sha256,
            )
            for item in self.arms[0].queries
        )
        if any(
            tuple(
                (
                    item.question_ordinal,
                    item.question_id_sha256,
                    item.question_probe_sha256,
                    item.frozen_input.identity_sha256,
                )
                for item in arm.queries
            )
            != coordinates
            for arm in self.arms[1:]
        ):
            raise ValueError("matched arms changed frozen query coordinates")
        provider_sha256 = self.verified_base_provider_identity.identity_sha256
        provider_body = json.loads(
            self.verified_base_provider_identity.canonical_identity_json
        )
        _require_keys(
            provider_body,
            {
                "implementation_type", "implementation", "python_code_sha256",
                "declared_identity",
            },
            "verified-base provider callable",
        )
        declared = provider_body.get("declared_identity")
        if not isinstance(declared, dict):
            raise ValueError("verified-base provider lacks a declared identity")
        _require_keys(
            declared,
            {
                "provider", "acquisition", "base_store_key",
                "base_artifact_sha256", "base_manifest_sha256",
                "query_input_key", "query_artifact_sha256",
                "query_manifest_sha256", "frozen_inputs_sha256",
                "query_set_sha256", "ordered_frozen_receipts_sha256",
                "max_sources", "rrf_constant",
            },
            "verified-base provider declaration",
        )
        if (
            declared.get("provider") != "verified-shared-base-pointer-v1"
            or declared.get("acquisition") != "verified_shared_base_pointer_v1"
            or declared.get("base_store_key") != self.base_manifest.base_store_key
            or declared.get("base_artifact_sha256")
            != self.base_manifest.artifact_sha256
            or declared.get("base_manifest_sha256")
            != self.base_manifest_file_sha256
            or declared.get("query_input_key") != self.query_manifest.query_input_key
            or declared.get("query_artifact_sha256")
            != self.query_manifest.artifact_sha256
            or declared.get("query_manifest_sha256")
            != self.query_manifest_file_sha256
            or declared.get("frozen_inputs_sha256")
            != self.query_manifest.frozen_inputs_sha256
            or declared.get("query_set_sha256") != self.query_manifest.query_set_sha256
        ):
            raise ValueError("verified-base provider declaration is not authoritative")
        if any(
            json.loads(query.analysis_query.canonical_identity_json).get(
                "legacy_input_provider_identity_sha256"
            )
            != provider_sha256
            for arm in self.arms
            for query in arm.queries
        ):
            raise ValueError("analysis queries do not bind the verified-base provider")
        runtime_payload = json.loads(self.runtime_binding.canonical_identity_json)
        matched_runtime = json.loads(
            self.matched_runtime_suite.canonical_identity_json
        )
        matched_phase = json.loads(
            self.matched_phase_suite.canonical_identity_json
        )
        if matched_runtime.get("runtime_binding_sha256") != (
            self.runtime_binding.identity_sha256
        ):
            raise ValueError("matched runtime changed the binding identity")
        if runtime_payload.get("runtime_binding_certified") is not True:
            raise ValueError("replay runtime binding is not certified")
        if tuple(matched_runtime.get("runtime_result_receipt_sha256s", ())) != tuple(
            arm.runtime_result.identity_sha256 for arm in self.arms
        ) or tuple(matched_phase.get("retrieval_phase_receipt_sha256s", ())) != tuple(
            arm.retrieval_phase.identity_sha256 for arm in self.arms
        ):
            raise ValueError("matched suite bodies differ from arm results")
        if any(
            json.loads(arm.retrieval_phase.canonical_identity_json).get(
                "evaluation_policy_sha256"
            )
            != self.evaluation_policy.identity_sha256
            for arm in self.arms
        ):
            raise ValueError("arm phases changed the evaluation policy")
        expected_files = tuple(
            sorted(
                f"{mode}/{name}"
                for mode in _MODES
                for name in (
                    "derived-final.json",
                    "derived-open.claim",
                    "derived-origin.json",
                    "hnsw_index.bin",
                    "memory.db",
                )
            )
        )
        if tuple(item.relative_path for item in self.files) != expected_files:
            raise ValueError("replay file inventory is incomplete or unordered")
        by_path = {item.relative_path: item for item in self.files}
        if len(by_path) != len(self.files):
            raise ValueError("replay file inventory contains duplicates")
        for arm in self.arms:
            if (
                arm.derived_origin.base_store_key
                != self.base_manifest.base_store_key
                or arm.derived_origin.base_artifact_sha256
                != self.base_manifest.artifact_sha256
                or arm.derived_origin.query_input_key
                != self.query_manifest.query_input_key
                or arm.derived_origin.query_artifact_sha256
                != self.query_manifest.artifact_sha256
            ):
                raise ValueError("derived origin belongs to another base/query pair")
            database = by_path[f"{arm.boundary_mode}/memory.db"]
            index = by_path[f"{arm.boundary_mode}/hnsw_index.bin"]
            if (
                database.sha256 != arm.finalization.database_sha256
                or database.bytes != arm.finalization.database_bytes
                or index.sha256 != arm.finalization.index_sha256
                or index.bytes != arm.finalization.index_bytes
            ):
                raise ValueError("file inventory differs from arm finalization")
            origin = by_path[f"{arm.boundary_mode}/derived-origin.json"]
            final = by_path[f"{arm.boundary_mode}/derived-final.json"]
            lease = by_path[f"{arm.boundary_mode}/derived-open.claim"]
            origin_sha, origin_bytes = _canonical_file_sha256(
                arm.derived_origin.model_dump(mode="json")
            )
            final_sha, final_bytes = _canonical_file_sha256(
                arm.finalization.model_dump(mode="json")
            )
            lease_sha, lease_bytes = _canonical_file_sha256(
                {
                    "format": "memory-condense-longmemeval-derived-open-claim-v1",
                    "origin_receipt_sha256": arm.derived_origin_receipt_sha256,
                }
            )
            if (
                (origin.sha256, origin.bytes) != (origin_sha, origin_bytes)
                or (final.sha256, final.bytes) != (final_sha, final_bytes)
                or (lease.sha256, lease.bytes) != (lease_sha, lease_bytes)
            ):
                raise ValueError("semantic sidecar inventory identity changed")
        from memory_condense.eval._diffuse_replay_validation import (
            validate_replay_crosslinks,
        )
        validate_replay_crosslinks(self)
        unsigned = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if identity_sha256(unsigned) != self.receipt_sha256:
            raise ValueError("replay receipt digest changed")
        return self


class ReplayFileIdentity(_FrozenModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=_DIGEST)
    bytes: int = Field(ge=1)

    @model_validator(mode="after")
    def _safe_relative_path(self) -> "ReplayFileIdentity":
        if (
            "\\" in self.relative_path
            or self.relative_path.startswith("/")
            or any(part in {"", ".", ".."} for part in self.relative_path.split("/"))
        ):
            raise ValueError("inventory path must be a canonical relative path")
        return self


DiffuseLongMemEvalReplayReceipt.model_rebuild()


__all__ = [
    "REPLAY_FORMAT",
    "REPLAY_MANIFEST_NAME",
    "CanonicalIdentityBody",
    "DiffuseLongMemEvalReplayReceipt",
    "DiffuseReplayArmRecord",
    "DiffuseReplayQueryRecord",
    "ReplayClosureAtom",
    "ReplayClosurePlanProjection",
    "ReplayClosureReceipt",
    "ReplayDroppedBundleReason",
    "ReplayEpisodeSeed",
    "ReplayEvidenceBundle",
    "ReplayEvidenceCoordinate",
    "ReplayExecutionIdentity",
    "ReplayFileIdentity",
    "ReplayLongMemEvalQueryReceipt",
    "ReplayMessageDescriptor",
    "ReplayObligationResult",
    "ReplayScopeWitness",
]
