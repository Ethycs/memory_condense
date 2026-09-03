#!/usr/bin/env python3
"""Provider-free confirmation R7/V6.1/V7 and terminal-v5 construction.

The module has two deliberately separate lifecycles.

``prepare_confirmation_semantic_facet_vectors`` runs while the already-pinned
BGE instance is resident in staged phase A.  It freezes every question-only
semantic residual facet into supplemental per-namespace artifacts without
changing the S0--S3 query-vector descriptors.  The release object is chained
to the exact BGE release receipt later embedded in the Qwen barrier.

``materialize_confirmation_semantic_planes`` runs after the protected
specialist V3 parent exists.  Eligibility is derived independently for every
question.  Each used namespace is opened once, one immutable full-store index
and its stored chunk-vector set are built, then the existing residual, local,
global, and linked/backfilled terminal-v5 cores run without a provider.

No API accepts a validation ordinal, allowlist, label, reference answer, or
target population.  The only positional value emitted is the downstream
terminal assay's legacy ``ordinal`` identity field; it is added after all
routing and retrieval have completed because the frozen plan consumer still
authenticates that field.
"""

from __future__ import annotations

import gc
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain.discourse import identity_sha256 as runtime_identity_sha256
from memory_condense.domain.discourse import quote_sha256
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import DiscourseStore
from memory_condense.search.episodes.retrieval import EpisodeRetrievalPolicy
from tools import confirmation_cumulative_retrieval as cumulative
from tools import confirmation_staged_cumulative_coordinator as staged
from tools import confirmation_terminal_policy_boundary as terminal_boundary
from tools.confirmation_contracts import SealedJson, publish_sealed_json, read_sealed_json
from tools.confirmation_namespace_store_adapter import SealedPayload, read_sealed_payload
from tools.materialize_confirmation_numeric_v5_overlay import (
    VerifiedNamespaceStore,
    VerifiedNamespaceStoreSet,
    _verify_store_bytes,
)
from tools.matched_eval import semantic_residual_search as residual
from tools.matched_eval.confirmation_semantic_helpers import (
    TERMINAL_COMPILATION_MODE_V5,
    build_separate_terminal_prompt,
    compile_answer_plan_core,
    ordered_protected_union,
    protected_evidence,
    question_inputs,
    selected_handle_bindings,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.full_store_slot_closure import (
    LocalCitationBinding,
    build_full_store_window_index,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.semantic_global_completion import (
    SemanticGlobalCompletionPolicy,
    SemanticGlobalCompletionResult,
    compile_semantic_global_completion_request,
    replay_semantic_global_completion,
    search_semantic_global_completion,
)
from tools.matched_eval.semantic_global_terminal_adapter import (
    SemanticGlobalTerminalCompilation,
    SemanticGlobalTerminalPolicy,
    TerminalSealedSources,
    compile_semantic_global_terminal,
    load_selected_protected_owner_evidence,
    replay_semantic_global_terminal,
)
from tools.matched_eval.semantic_residual_eligibility import (
    SemanticResidualEligibilityDecision,
    SemanticResidualEligibilityPolicy,
    evaluate_semantic_residual_eligibility,
    replay_semantic_residual_eligibility,
)
from tools.matched_eval.source_group_reinjection import (
    SourceGroupReinjectionPolicy,
    SourceGroupReinjectionResult,
    authenticate_source_group_selection,
    replay_source_group_reinjection,
    search_source_group_reinjection,
)
from tools.confirmation_canonical import canonical_sha256


FACET_VECTOR_FORMAT = "memory-condense-confirmation-semantic-facet-vectors-v1"
FACET_CHECKPOINT_FORMAT = f"{FACET_VECTOR_FORMAT}-namespace-checkpoint-v1"
FACET_PREPARATION_FORMAT = f"{FACET_VECTOR_FORMAT}-preparation-v1"
FACET_RELEASE_FORMAT = f"{FACET_VECTOR_FORMAT}-release-v1"
CHECKPOINT_FORMAT = "memory-condense-confirmation-semantic-planes-checkpoint-v1"
MATERIALIZATION_FORMAT = "memory-condense-confirmation-semantic-planes-v1"
REPLAY_FORMAT = MATERIALIZATION_FORMAT

FACET_PREPARATION_NAME = "confirmation-semantic-facet-preparation-v1.json"
FACET_RELEASE_NAME = "confirmation-semantic-facet-release-v1.json"
MATERIALIZATION_NAME = "confirmation-semantic-planes-v1.json"
REPLAY_NAME = "confirmation-semantic-planes-replay-v1.json"
TERMINAL_PLAN_EXPORT_NAME = "confirmation-terminal-v5-plan-export-v1.json"

FIXED_INTERVAL_EPISODE_KINDS = (
    "fixed_interval",
    "longmemeval-diffuse-fixed_interval",
)

ELIGIBILITY_POLICY = SemanticResidualEligibilityPolicy()
RESIDUAL_POLICY = residual.SemanticResidualPolicy(
    max_cell_tokens=2_048,
    payload_token_cap=2_400,
    cosine_upper_bound_floor=0.05,
    specificity_upper_bound_ratio=0.75,
    dual_gate_enabled=True,
    classifier_mode=residual.EVIDENCE_CONSERVING_RESIDUAL_CLASSIFIER_MODE,
)
LOCAL_POLICY = SourceGroupReinjectionPolicy(
    local_payload_token_cap=1_200,
    max_selected_segments=64,
    base_segments_per_group=3,
    max_query_term_obligations=6,
    source_neighbor_radius=1,
    max_source_neighbors_per_anchor=2,
    max_episode_segments_per_seed=4,
)
GLOBAL_POLICY = SemanticGlobalCompletionPolicy()
TERMINAL_POLICY = SemanticGlobalTerminalPolicy()

EXPECTED_POLICY_RECEIPTS = MappingProxyType(
    {
        "eligibility_policy": "ebaa6acade2c12f2dcf5f5e52e8e45661870ba916e85fb0f4a79bab5c2ccc955",
        "residual_search_policy": "288c9a08051f547626836a3b08fcc85a0844fe1397b27a3f6a978b8801ee6e88",
        "local_policy": "c15f430054445dc96c246a3ba156710a6990b807c07dea571f042104a52795df",
        "global_policy": "504cab6a3d145442e7ebc9d1efa71ac9673249c01092e40cec5f00837157bb61",
        "terminal_policy": "e2b3b5a5eb9dabf4841b56e30ab60998fd12bbfb552a2cff76a594e90f196d3b",
    }
)

_FORBIDDEN_ROUTING_KEYS = frozenset(
    {
        "allowlist",
        "eligible_ordinals",
        "eligible_question_ids",
        "gold",
        "label",
        "labels",
        "miss_ordinals",
        "ordinals",
        "reference",
        "reference_answer",
        "target_ordinals",
        "target_question_ids",
        "validation_ordinals",
        "validation_question_ids",
        "whitelist",
    }
)


class ConfirmationSemanticPlanesError(MatchedEvalContractError):
    """A facet, store, protected parent, semantic result, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationSemanticPlanesError(message)


def _sha(value: object, label: str) -> str:
    try:
        return require_sha256(value, label)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ConfirmationSemanticPlanesError(str(exc)) from exc


def _text(value: object, label: str) -> str:
    try:
        return require_text(value, label)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ConfirmationSemanticPlanesError(str(exc)) from exc


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value  # type: ignore[return-value]


def _list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an array")
    return value  # type: ignore[return-value]


def _self_seal(value: Mapping[str, Any], key: str, label: str) -> str:
    declared = _sha(value.get(key), f"{label} receipt")
    body = dict(value)
    body.pop(key, None)
    _require(identity_sha256(body) == declared, f"{label} self-seal changed")
    return declared


def _assert_routing_neutral(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            _require(
                key not in _FORBIDDEN_ROUTING_KEYS,
                f"population/label routing field is forbidden: {path}.{raw_key}",
            )
            if key == "gold_loaded":
                _require(child is False, f"gold sentinel changed: {path}")
            _assert_routing_neutral(child, f"{path}.{raw_key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _assert_routing_neutral(child, f"{path}[{index}]")


def _sealed_payload(
    path: Path, payload: Mapping[str, Any], *, label: str
) -> tuple[SealedPayload, bool]:
    try:
        return cumulative._publish_sealed(  # noqa: SLF001
            path,
            cumulative._plain_json(payload),  # noqa: SLF001
            label=label,
        )
    except ValueError as exc:
        raise ConfirmationSemanticPlanesError(str(exc)) from exc


def _policy_projection(inputs: terminal_boundary.ConfirmationTerminalInputs) -> Mapping[str, Any]:
    treatment = _mapping(inputs.policy.payload.get("treatment_policy"), "treatment policy")
    bindings = _mapping(treatment.get("full100_policy_bindings"), "full100 policy bindings")
    expected = {
        "eligibility_policy": ELIGIBILITY_POLICY.projection(),
        "residual_search_policy": RESIDUAL_POLICY.projection(),
        "local_policy": LOCAL_POLICY.projection(),
        "global_policy": GLOBAL_POLICY.projection(),
        "terminal_policy": TERMINAL_POLICY.projection(),
    }
    _require(
        treatment.get("policy_id") == "policy-v5-r3"
        and bindings.get("terminal_compilation_format")
        == terminal_boundary.V5_COMPILATION_FORMAT
        and all(bindings.get(key) == value for key, value in expected.items()),
        "semantic plane policies differ from the frozen policy-v5-r3 binding",
    )
    _require(
        {key: value.receipt_sha256 for key, value in (
            ("eligibility_policy", ELIGIBILITY_POLICY),
            ("residual_search_policy", RESIDUAL_POLICY),
            ("local_policy", LOCAL_POLICY),
            ("global_policy", GLOBAL_POLICY),
            ("terminal_policy", TERMINAL_POLICY),
        )} == dict(EXPECTED_POLICY_RECEIPTS),
        "compiled semantic policy receipts changed",
    )
    return bindings


def semantic_facet_query_batch(
    dated_questions: Sequence[str], /
) -> tuple[str, ...]:
    """Return the complete ordered, deduplicated BGE facet batch."""

    questions = tuple(dated_questions)
    _require(
        bool(questions)
        and all(type(row) is str and bool(row.strip()) for row in questions),
        "semantic facet questions must be nonempty text",
    )
    return tuple(
        dict.fromkeys(
            facet
            for question in questions
            for facet in residual.semantic_residual_query_facets(question)
        )
    )


@dataclass(frozen=True, slots=True)
class SemanticFacetVectorDescriptor:
    namespace_id: str
    namespace_store_id: str
    preparation_checkpoint_sha256: str
    retrieval_vector_artifact_sha256: str
    artifact_path: Path
    artifact_sha256: str
    artifact_receipt_sha256: str
    query_batch: tuple[str, ...]
    query_batch_sha256: str
    vector_values_sha256: str
    dimension: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.namespace_id, "facet namespace"),
            (self.namespace_store_id, "facet namespace store"),
            (self.preparation_checkpoint_sha256, "facet preparation checkpoint"),
            (self.retrieval_vector_artifact_sha256, "retrieval vector artifact"),
            (self.artifact_sha256, "facet vector artifact"),
            (self.artifact_receipt_sha256, "facet vector artifact receipt"),
            (self.query_batch_sha256, "facet query batch"),
            (self.vector_values_sha256, "facet vector values"),
        ):
            _sha(value, label)
        _require(
            self.artifact_path.is_absolute()
            and type(self.query_batch) is tuple
            and bool(self.query_batch)
            and len(set(self.query_batch)) == len(self.query_batch)
            and type(self.dimension) is int
            and self.dimension > 0,
            "semantic facet descriptor changed",
        )


@dataclass(frozen=True, slots=True)
class ConfirmationSemanticFacetPreparation:
    artifact: SealedPayload
    descriptors: tuple[SemanticFacetVectorDescriptor, ...]
    created_count: int
    reused_count: int
    local_embedding_batch_calls: int

    def __post_init__(self) -> None:
        _require(
            type(self.artifact) is SealedPayload
            and bool(self.descriptors)
            and self.created_count + self.reused_count == len(self.descriptors)
            and self.local_embedding_batch_calls == self.created_count,
            "semantic facet preparation accounting changed",
        )


@dataclass(frozen=True, slots=True)
class ConfirmationSemanticVectorRelease:
    preparation: SealedPayload
    release: SealedPayload
    barrier: SealedPayload
    descriptors_by_namespace: Mapping[str, SemanticFacetVectorDescriptor]

    def __post_init__(self) -> None:
        _require(
            type(self.preparation) is SealedPayload
            and type(self.release) is SealedPayload
            and type(self.barrier) is SealedPayload
            and bool(self.descriptors_by_namespace)
            and all(
                key == row.namespace_id
                for key, row in self.descriptors_by_namespace.items()
            ),
            "semantic vector release changed",
        )


def _facet_paths(root: Path, store_id: str) -> tuple[Path, Path]:
    base = root / "staged-preparation" / "semantic-facets"
    return (
        base / "vectors" / f"{store_id}.json",
        base / "checkpoints" / f"{store_id}.json",
    )


def _normalize_facet_vectors(
    *,
    request: cumulative.CumulativeNamespaceRequest,
    descriptor: staged.FrozenQueryDescriptor,
    preparation_checkpoint_sha256: str,
    backend: staged.StagedPreparationBackend,
    query_batch: tuple[str, ...],
    vectors: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    _require(tuple(vectors) == query_batch, "semantic facet vectors changed order")
    rows: list[dict[str, Any]] = []
    dimensions: set[int] = set()
    for query in query_batch:
        raw = vectors[query]
        _require(
            isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)),
            "semantic facet vector changed type",
        )
        vector = [float(value) for value in raw]
        _require(vector and all(math.isfinite(value) for value in vector), "semantic facet vector is nonfinite")
        dimensions.add(len(vector))
        rows.append(
            {
                "query": query,
                "query_sha256": quote_sha256(query),
                "vector": vector,
                "vector_sha256": canonical_sha256(vector),
            }
        )
    _require(len(dimensions) == 1, "semantic facet vector dimensions differ")
    dimension = dimensions.pop()
    embedding_identity = cumulative._plain_json(backend.embedding_identity)  # noqa: SLF001
    declared = embedding_identity.get("dimension")
    _require(
        declared is None or declared == dimension,
        "semantic facet dimension differs from the staged BGE identity",
    )
    identities = [
        {"query_sha256": row["query_sha256"], "vector_sha256": row["vector_sha256"]}
        for row in rows
    ]
    body = {
        "dimension": dimension,
        "embedding_identity_sha256": runtime_identity_sha256(embedding_identity),
        "format": FACET_VECTOR_FORMAT,
        "namespace_id": request.work.namespace_id,
        "namespace_store_id": request.work.namespace_store_id,
        "preparation_checkpoint_sha256": preparation_checkpoint_sha256,
        "query_batch_sha256": runtime_identity_sha256(
            [{"query_sha256": quote_sha256(query)} for query in query_batch]
        ),
        "retrieval_query_vector_artifact_sha256": descriptor.artifact_sha256,
        "rows": rows,
        "vector_values_sha256": canonical_sha256(identities),
        "work_receipt_sha256": request.work.work_receipt_sha256,
    }
    _assert_routing_neutral(body, "semantic_facet_vectors")
    assert_gold_blind(body, path="confirmation_semantic_facet_vectors")
    return {**body, "artifact_receipt_sha256": identity_sha256(body)}


def _validate_facet_vector_artifact(
    sealed: SealedPayload,
    *,
    request: cumulative.CumulativeNamespaceRequest,
    retrieval_descriptor: staged.FrozenQueryDescriptor,
    preparation_checkpoint_sha256: str,
    embedding_identity_sha256: str,
    expected_batch: tuple[str, ...],
) -> tuple[SemanticFacetVectorDescriptor, Mapping[str, tuple[float, ...]]]:
    value = dict(sealed.payload)
    _assert_routing_neutral(value, "semantic_facet_vectors")
    _require(
        set(value)
        == {
            "artifact_receipt_sha256",
            "dimension",
            "embedding_identity_sha256",
            "format",
            "namespace_id",
            "namespace_store_id",
            "preparation_checkpoint_sha256",
            "query_batch_sha256",
            "retrieval_query_vector_artifact_sha256",
            "rows",
            "vector_values_sha256",
            "work_receipt_sha256",
        },
        "semantic facet artifact has a non-closed schema",
    )
    receipt = _self_seal(value, "artifact_receipt_sha256", "semantic facet artifact")
    rows = _list(value.get("rows"), "semantic facet rows")
    _require(
        value.get("format") == FACET_VECTOR_FORMAT
        and value.get("namespace_id") == request.work.namespace_id
        and value.get("namespace_store_id") == request.work.namespace_store_id
        and value.get("preparation_checkpoint_sha256")
        == preparation_checkpoint_sha256
        and value.get("retrieval_query_vector_artifact_sha256")
        == retrieval_descriptor.artifact_sha256
        and value.get("embedding_identity_sha256") == embedding_identity_sha256
        and value.get("work_receipt_sha256") == request.work.work_receipt_sha256,
        "semantic facet artifact escaped its staged namespace",
    )
    dimension = value.get("dimension")
    _require(type(dimension) is int and dimension > 0, "semantic facet dimension changed")
    vectors: dict[str, tuple[float, ...]] = {}
    identities: list[dict[str, str]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"semantic facet row {index}")
        _require(
            set(row) == {"query", "query_sha256", "vector", "vector_sha256"},
            "semantic facet row has a non-closed schema",
        )
        query = _text(row.get("query"), "semantic facet query")
        vector_raw = _list(row.get("vector"), "semantic facet vector")
        vector = tuple(float(item) for item in vector_raw)
        _require(
            len(vector) == dimension
            and all(math.isfinite(item) for item in vector)
            and row.get("query_sha256") == quote_sha256(query)
            and row.get("vector_sha256") == canonical_sha256(vector_raw)
            and query not in vectors,
            "semantic facet row changed",
        )
        vectors[query] = vector
        identities.append(
            {
                "query_sha256": str(row["query_sha256"]),
                "vector_sha256": str(row["vector_sha256"]),
            }
        )
    batch = tuple(vectors)
    expected_batch_sha = runtime_identity_sha256(
        [{"query_sha256": quote_sha256(query)} for query in expected_batch]
    )
    _require(
        batch == expected_batch
        and value.get("query_batch_sha256") == expected_batch_sha
        and value.get("vector_values_sha256") == canonical_sha256(identities),
        "semantic facet vector population changed",
    )
    return (
        SemanticFacetVectorDescriptor(
            namespace_id=request.work.namespace_id,
            namespace_store_id=request.work.namespace_store_id,
            preparation_checkpoint_sha256=preparation_checkpoint_sha256,
            retrieval_vector_artifact_sha256=retrieval_descriptor.artifact_sha256,
            artifact_path=sealed.path.resolve(),
            artifact_sha256=sealed.sha256,
            artifact_receipt_sha256=receipt,
            query_batch=batch,
            query_batch_sha256=expected_batch_sha,
            vector_values_sha256=str(value["vector_values_sha256"]),
            dimension=dimension,
        ),
        MappingProxyType(vectors),
    )


def _staged_preparation_inventory(
    preparation: staged.StagedPreparationExecution,
) -> tuple[Mapping[str, staged.FrozenQueryDescriptor], Mapping[str, str]]:
    _require(
        type(preparation) is staged.StagedPreparationExecution
        and bool(preparation.descriptors)
        and len(preparation.descriptors)
        == len(preparation.checkpoint_paths)
        == len(preparation.checkpoint_sha256s),
        "staged preparation changed",
    )
    descriptors = {row.namespace_store_id: row for row in preparation.descriptors}
    _require(
        len(descriptors) == len(preparation.descriptors),
        "staged preparation repeated a namespace store",
    )
    checkpoints: dict[str, str] = {}
    for path, expected_sha in zip(
        preparation.checkpoint_paths,
        preparation.checkpoint_sha256s,
        strict=True,
    ):
        sealed = read_sealed_payload(path, label="staged preparation checkpoint")
        _require(sealed.sha256 == expected_sha, "staged preparation checkpoint changed")
        store_id = _text(
            sealed.payload.get("namespace_store_id"),
            "staged preparation namespace store",
        )
        _require(store_id not in checkpoints, "staged preparation repeated a checkpoint")
        checkpoints[store_id] = expected_sha
    _require(
        set(checkpoints) == set(descriptors),
        "staged preparation descriptor/checkpoint population differs",
    )
    return MappingProxyType(descriptors), MappingProxyType(checkpoints)


def prepare_confirmation_semantic_facet_vectors(
    inputs: cumulative.ConfirmationCumulativeInput,
    preparation: staged.StagedPreparationExecution,
    *,
    backend: staged.StagedPreparationBackend,
    output_root: str | Path,
    token_counter: Any | None = None,
) -> ConfirmationSemanticFacetPreparation:
    """Freeze every residual facet through the still-resident staged BGE.

    This is the executable pre-release hook.  It publishes a supplemental
    artifact and never mutates the already-frozen retrieval descriptor.
    """

    root = Path(output_root).resolve()
    requests = cumulative.confirmation_cumulative_requests(
        inputs, output_root=root, token_counter=token_counter
    )
    by_store = {request.work.namespace_store_id: request for request in requests}
    _require(len(by_store) == len(requests), "semantic facet requests repeat a store")
    retrieval_descriptors, preparation_checkpoints = _staged_preparation_inventory(
        preparation
    )
    _require(
        set(by_store) == set(retrieval_descriptors),
        "semantic facet request population differs from staged preparation",
    )
    backend_identity = _sha(backend.identity_sha256, "semantic facet backend")
    _require(
        _sha(backend.policy_freeze_sha256, "semantic facet policy freeze")
        == inputs.source_policy_sha256,
        "semantic facet backend binds another policy freeze",
    )
    embedding_identity = cumulative._plain_json(backend.embedding_identity)  # noqa: SLF001
    embedding_sha = runtime_identity_sha256(embedding_identity)
    descriptors: list[SemanticFacetVectorDescriptor] = []
    created = reused = 0
    checkpoint_refs: list[dict[str, Any]] = []
    for store_id in sorted(by_store):
        request = by_store[store_id]
        retrieval_descriptor = retrieval_descriptors[store_id]
        checkpoint_sha = preparation_checkpoints[store_id]
        batch = semantic_facet_query_batch(
            tuple(query.dated_question for query in request.queries)
        )
        vector_path, facet_checkpoint_path = _facet_paths(root, store_id)
        vector_exists = vector_path.exists() or vector_path.is_symlink()
        checkpoint_exists = (
            facet_checkpoint_path.exists() or facet_checkpoint_path.is_symlink()
        )
        _require(
            vector_exists == checkpoint_exists,
            "partial semantic facet checkpoint is unsafe",
        )
        if checkpoint_exists:
            facet_checkpoint = read_sealed_payload(
                facet_checkpoint_path, label="semantic facet checkpoint"
            )
            checkpoint = dict(facet_checkpoint.payload)
            _require(
                set(checkpoint)
                == {
                    "backend_identity_sha256",
                    "checkpoint_receipt_sha256",
                    "embedding_identity_sha256",
                    "format",
                    "gold_loaded",
                    "namespace_id",
                    "namespace_store_id",
                    "physical_provider_calls",
                    "preparation_checkpoint_sha256",
                    "query_vector_artifact_receipt_sha256",
                    "query_vector_artifact_relative_path",
                    "query_vector_artifact_sha256",
                    "retrieval_query_vector_artifact_sha256",
                    "work_receipt_sha256",
                },
                "semantic facet checkpoint has a non-closed schema",
            )
            _self_seal(
                checkpoint,
                "checkpoint_receipt_sha256",
                "semantic facet checkpoint",
            )
            relative = Path(str(checkpoint["query_vector_artifact_relative_path"]))
            _require(not relative.is_absolute(), "semantic facet path is absolute")
            resolved = (root / relative).resolve()
            _require(
                resolved == vector_path.resolve() and resolved.is_relative_to(root),
                "semantic facet artifact escaped its address",
            )
            sealed_vector = read_sealed_payload(
                resolved, label="semantic facet vector artifact"
            )
            _require(
                checkpoint.get("format") == FACET_CHECKPOINT_FORMAT
                and checkpoint.get("gold_loaded") is False
                and checkpoint.get("physical_provider_calls") == 0
                and checkpoint.get("backend_identity_sha256") == backend_identity
                and checkpoint.get("embedding_identity_sha256") == embedding_sha
                and checkpoint.get("namespace_id") == request.work.namespace_id
                and checkpoint.get("namespace_store_id") == store_id
                and checkpoint.get("preparation_checkpoint_sha256") == checkpoint_sha
                and checkpoint.get("retrieval_query_vector_artifact_sha256")
                == retrieval_descriptor.artifact_sha256
                and checkpoint.get("query_vector_artifact_sha256")
                == sealed_vector.sha256
                and checkpoint.get("work_receipt_sha256")
                == request.work.work_receipt_sha256,
                "semantic facet checkpoint binding changed",
            )
            descriptor, _vectors = _validate_facet_vector_artifact(
                sealed_vector,
                request=request,
                retrieval_descriptor=retrieval_descriptor,
                preparation_checkpoint_sha256=checkpoint_sha,
                embedding_identity_sha256=embedding_sha,
                expected_batch=batch,
            )
            _require(
                checkpoint.get("query_vector_artifact_receipt_sha256")
                == descriptor.artifact_receipt_sha256,
                "semantic facet checkpoint lost its vector receipt",
            )
            reused += 1
        else:
            try:
                frozen = backend.freeze_query_batch(batch)
            except (AttributeError, ValueError) as exc:
                raise ConfirmationSemanticPlanesError(
                    "semantic facets must be frozen before BGE release"
                ) from exc
            payload = _normalize_facet_vectors(
                request=request,
                descriptor=retrieval_descriptor,
                preparation_checkpoint_sha256=checkpoint_sha,
                backend=backend,
                query_batch=batch,
                vectors=frozen,
            )
            sealed_vector, vector_created = _sealed_payload(
                vector_path, payload, label="semantic facet vector artifact"
            )
            _require(vector_created, "semantic facet vector unexpectedly existed")
            descriptor, _vectors = _validate_facet_vector_artifact(
                sealed_vector,
                request=request,
                retrieval_descriptor=retrieval_descriptor,
                preparation_checkpoint_sha256=checkpoint_sha,
                embedding_identity_sha256=embedding_sha,
                expected_batch=batch,
            )
            body = {
                "backend_identity_sha256": backend_identity,
                "embedding_identity_sha256": embedding_sha,
                "format": FACET_CHECKPOINT_FORMAT,
                "gold_loaded": False,
                "namespace_id": request.work.namespace_id,
                "namespace_store_id": store_id,
                "physical_provider_calls": 0,
                "preparation_checkpoint_sha256": checkpoint_sha,
                "query_vector_artifact_receipt_sha256": (
                    descriptor.artifact_receipt_sha256
                ),
                "query_vector_artifact_relative_path": str(
                    vector_path.relative_to(root)
                ).replace("\\", "/"),
                "query_vector_artifact_sha256": sealed_vector.sha256,
                "retrieval_query_vector_artifact_sha256": (
                    retrieval_descriptor.artifact_sha256
                ),
                "work_receipt_sha256": request.work.work_receipt_sha256,
            }
            facet_checkpoint, checkpoint_created = _sealed_payload(
                facet_checkpoint_path,
                {**body, "checkpoint_receipt_sha256": identity_sha256(body)},
                label="semantic facet checkpoint",
            )
            _require(checkpoint_created, "semantic facet checkpoint unexpectedly existed")
            created += 1
        descriptors.append(descriptor)
        checkpoint_refs.append(
            {
                "checkpoint_sha256": facet_checkpoint.sha256,
                "namespace_id": request.work.namespace_id,
                "namespace_store_id": store_id,
                "query_batch_sha256": descriptor.query_batch_sha256,
                "query_vector_artifact_sha256": descriptor.artifact_sha256,
                "retrieval_query_vector_artifact_sha256": (
                    descriptor.retrieval_vector_artifact_sha256
                ),
            }
        )
    body = {
        "backend_identity_sha256": backend_identity,
        "embedding_identity_sha256": embedding_sha,
        "format": FACET_PREPARATION_FORMAT,
        "freeze_sha256": inputs.source_policy_sha256,
        "gold_loaded": False,
        "namespace_count": len(descriptors),
        "physical_provider_calls": 0,
        "preflight_sha256": inputs.preflight.sha256,
        "question_count": sum(len(request.queries) for request in requests),
        "semantic_facet_count": sum(len(row.query_batch) for row in descriptors),
        "semantic_facets": checkpoint_refs,
        "staged_preparation_checkpoint_population_sha256": identity_sha256(
            sorted(preparation_checkpoints.values())
        ),
        "workset_identity_sha256": inputs.workset.workset_identity_sha256,
    }
    _assert_routing_neutral(body, "semantic_facet_preparation")
    artifact, _aggregate_created = _sealed_payload(
        root / "staged-preparation" / FACET_PREPARATION_NAME,
        {**body, "preparation_receipt_sha256": identity_sha256(body)},
        label="semantic facet preparation",
    )
    return ConfirmationSemanticFacetPreparation(
        artifact=artifact,
        descriptors=tuple(descriptors),
        created_count=created,
        reused_count=reused,
        local_embedding_batch_calls=created,
    )


confirmation_semantic_facet_pre_release_hook = (
    prepare_confirmation_semantic_facet_vectors
)


def publish_confirmation_semantic_facet_release(
    preparation: ConfirmationSemanticFacetPreparation,
    bge_release_receipt: Mapping[str, Any],
    *,
    output_root: str | Path,
) -> SealedPayload:
    """Bind the supplemental freeze to the exact staged BGE close receipt."""

    _require(
        type(preparation) is ConfirmationSemanticFacetPreparation,
        "semantic facet release requires an exact preparation",
    )
    release = dict(_mapping(bge_release_receipt, "BGE release receipt"))
    release_receipt = _self_seal(
        release, "release_receipt_sha256", "BGE release receipt"
    )
    source = preparation.artifact.payload
    _require(
        release.get("format") == staged.BGE_RELEASE_FORMAT
        and release.get("embedding_released_before_qwen_load") is True
        and release.get("physical_provider_calls") == 0
        and release.get("preparation_backend_identity_sha256")
        == source.get("backend_identity_sha256")
        and release.get("embedding_identity_sha256")
        == source.get("embedding_identity_sha256"),
        "semantic facet preparation escaped the BGE release",
    )
    root = Path(output_root).resolve()
    preparation_path = preparation.artifact.path.resolve()
    _require(
        preparation_path.is_relative_to(root),
        "semantic facet preparation escaped its output root",
    )
    body = {
        "bge_release_receipt": release,
        "bge_release_receipt_sha256": release_receipt,
        "embedding_identity_sha256": source["embedding_identity_sha256"],
        "format": FACET_RELEASE_FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "preparation_artifact_relative_path": str(
            preparation_path.relative_to(root)
        ).replace("\\", "/"),
        "preparation_artifact_sha256": preparation.artifact.sha256,
        "preparation_receipt_sha256": source["preparation_receipt_sha256"],
    }
    artifact, _created = _sealed_payload(
        root / "staged-preparation" / FACET_RELEASE_NAME,
        {**body, "release_binding_receipt_sha256": identity_sha256(body)},
        label="semantic facet release",
    )
    return artifact


def _validate_loaded_facet_descriptor(
    *,
    sealed: SealedPayload,
    namespace_id: str,
    store: VerifiedNamespaceStore,
    preparation_ref: Mapping[str, Any],
    retrieval_vector_artifact_sha256: str,
    embedding_identity_sha256: str,
    expected_batch: tuple[str, ...],
) -> SemanticFacetVectorDescriptor:
    value = dict(sealed.payload)
    _assert_routing_neutral(value, "loaded_semantic_facet_vectors")
    _require(
        set(value)
        == {
            "artifact_receipt_sha256",
            "dimension",
            "embedding_identity_sha256",
            "format",
            "namespace_id",
            "namespace_store_id",
            "preparation_checkpoint_sha256",
            "query_batch_sha256",
            "retrieval_query_vector_artifact_sha256",
            "rows",
            "vector_values_sha256",
            "work_receipt_sha256",
        },
        "loaded semantic facet artifact has a non-closed schema",
    )
    receipt = _self_seal(value, "artifact_receipt_sha256", "semantic facet artifact")
    rows = _list(value.get("rows"), "semantic facet rows")
    dimension = value.get("dimension")
    _require(type(dimension) is int and dimension > 0, "semantic facet dimension changed")
    identities: list[dict[str, str]] = []
    batch: list[str] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"semantic facet row {index}")
        _require(
            set(row) == {"query", "query_sha256", "vector", "vector_sha256"},
            "semantic facet row has a non-closed schema",
        )
        query = _text(row.get("query"), "semantic facet query")
        vector = _list(row.get("vector"), "semantic facet vector")
        _require(
            len(vector) == dimension
            and all(type(item) in {int, float} and math.isfinite(float(item)) for item in vector)
            and row.get("query_sha256") == quote_sha256(query)
            and row.get("vector_sha256") == canonical_sha256(vector),
            "semantic facet row changed",
        )
        batch.append(query)
        identities.append(
            {
                "query_sha256": str(row["query_sha256"]),
                "vector_sha256": str(row["vector_sha256"]),
            }
        )
    batch_tuple = tuple(batch)
    batch_sha = runtime_identity_sha256(
        [{"query_sha256": quote_sha256(query)} for query in batch_tuple]
    )
    _require(
        batch_tuple == expected_batch
        and len(set(batch_tuple)) == len(batch_tuple)
        and value.get("format") == FACET_VECTOR_FORMAT
        and value.get("namespace_id") == namespace_id
        and value.get("namespace_store_id") == store.namespace_store_id
        and value.get("preparation_checkpoint_sha256")
        == store.preparation_checkpoint_sha256
        and value.get("retrieval_query_vector_artifact_sha256")
        == retrieval_vector_artifact_sha256
        and value.get("embedding_identity_sha256") == embedding_identity_sha256
        and value.get("query_batch_sha256") == batch_sha
        and value.get("vector_values_sha256") == canonical_sha256(identities)
        and preparation_ref.get("query_batch_sha256") == batch_sha
        and preparation_ref.get("query_vector_artifact_sha256") == sealed.sha256
        and preparation_ref.get("retrieval_query_vector_artifact_sha256")
        == retrieval_vector_artifact_sha256,
        "loaded semantic facet artifact binding changed",
    )
    return SemanticFacetVectorDescriptor(
        namespace_id=namespace_id,
        namespace_store_id=store.namespace_store_id,
        preparation_checkpoint_sha256=store.preparation_checkpoint_sha256,
        retrieval_vector_artifact_sha256=retrieval_vector_artifact_sha256,
        artifact_path=sealed.path.resolve(),
        artifact_sha256=sealed.sha256,
        artifact_receipt_sha256=receipt,
        query_batch=batch_tuple,
        query_batch_sha256=batch_sha,
        vector_values_sha256=str(value["vector_values_sha256"]),
        dimension=dimension,
    )


def load_confirmation_semantic_vector_release(
    inputs: terminal_boundary.ConfirmationTerminalInputs,
    stores: VerifiedNamespaceStoreSet,
    *,
    staged_output_root: str | Path,
    facet_release_path: str | Path,
    expected_facet_release_sha256: str,
    barrier_path: str | Path,
    expected_barrier_sha256: str,
) -> ConfirmationSemanticVectorRelease:
    """Authenticate the post-release facet inventory without bulk-loading it."""

    _require(
        type(inputs) is terminal_boundary.ConfirmationTerminalInputs
        and type(stores) is VerifiedNamespaceStoreSet,
        "semantic vector release inputs changed type",
    )
    root = Path(staged_output_root).resolve()
    release = read_sealed_payload(
        Path(facet_release_path), label="semantic facet release"
    )
    _require(
        release.sha256 == _sha(expected_facet_release_sha256, "semantic facet release"),
        "semantic facet release differs from its external seal",
    )
    release_payload = dict(release.payload)
    _require(
        set(release_payload)
        == {
            "bge_release_receipt",
            "bge_release_receipt_sha256",
            "embedding_identity_sha256",
            "format",
            "gold_loaded",
            "physical_provider_calls",
            "preparation_artifact_relative_path",
            "preparation_artifact_sha256",
            "preparation_receipt_sha256",
            "release_binding_receipt_sha256",
        },
        "semantic facet release has a non-closed schema",
    )
    _self_seal(
        release_payload,
        "release_binding_receipt_sha256",
        "semantic facet release",
    )
    bge_release = _mapping(
        release_payload.get("bge_release_receipt"), "BGE release receipt"
    )
    _require(
        release_payload.get("format") == FACET_RELEASE_FORMAT
        and release_payload.get("gold_loaded") is False
        and release_payload.get("physical_provider_calls") == 0
        and bge_release.get("release_receipt_sha256")
        == release_payload.get("bge_release_receipt_sha256")
        and bge_release.get("embedding_identity_sha256")
        == release_payload.get("embedding_identity_sha256"),
        "semantic facet release binding changed",
    )
    _self_seal(bge_release, "release_receipt_sha256", "BGE release receipt")
    preparation_relative = Path(
        _text(
            release_payload.get("preparation_artifact_relative_path"),
            "semantic facet preparation path",
        )
    )
    _require(not preparation_relative.is_absolute(), "semantic preparation path is absolute")
    preparation_path = (root / preparation_relative).resolve()
    _require(preparation_path.is_relative_to(root), "semantic preparation escaped root")
    preparation = read_sealed_payload(
        preparation_path, label="semantic facet preparation"
    )
    _require(
        preparation.sha256 == release_payload.get("preparation_artifact_sha256"),
        "semantic facet preparation differs from release",
    )
    prep = dict(preparation.payload)
    _require(
        set(prep)
        == {
            "backend_identity_sha256",
            "embedding_identity_sha256",
            "format",
            "freeze_sha256",
            "gold_loaded",
            "namespace_count",
            "physical_provider_calls",
            "preflight_sha256",
            "preparation_receipt_sha256",
            "question_count",
            "semantic_facet_count",
            "semantic_facets",
            "staged_preparation_checkpoint_population_sha256",
            "workset_identity_sha256",
        }
        and _self_seal(prep, "preparation_receipt_sha256", "semantic facet preparation")
        == release_payload.get("preparation_receipt_sha256"),
        "semantic facet preparation has a non-closed schema",
    )
    _require(
        prep.get("format") == FACET_PREPARATION_FORMAT
        and prep.get("gold_loaded") is False
        and prep.get("physical_provider_calls") == 0
        and prep.get("embedding_identity_sha256")
        == release_payload.get("embedding_identity_sha256")
        and prep.get("freeze_sha256") == inputs.policy.sha256
        and prep.get("preflight_sha256") == inputs.treatment_preflight.sha256
        and stores.policy_manifest_sha256 == inputs.policy.sha256
        and stores.treatment_preflight_sha256 == inputs.treatment_preflight.sha256,
        "semantic facet preparation escaped terminal/store ancestry",
    )

    raw_barrier = read_sealed_payload(Path(barrier_path), label="staged BGE barrier")
    _require(
        raw_barrier.sha256
        == _sha(expected_barrier_sha256, "expected staged BGE barrier")
        == stores.barrier_sha256,
        "semantic stage received another BGE barrier",
    )
    qwen_identity = _sha(
        raw_barrier.payload.get("qwen_factory_identity_sha256"),
        "barrier Qwen factory",
    )
    barrier = staged._verified_qwen_barrier(  # noqa: SLF001
        raw_barrier, qwen_factory_identity_sha256=qwen_identity
    )
    _require(
        barrier.payload.get("barrier_receipt_sha256") == stores.barrier_receipt_sha256
        and barrier.payload.get("release_receipt") == bge_release,
        "semantic facet release differs from the Qwen barrier",
    )
    barrier_by_store = {
        str(row["namespace_store_id"]): _mapping(row, "barrier preparation")
        for row in _list(barrier.payload.get("preparations"), "barrier preparations")
    }
    prep_refs = _list(prep.get("semantic_facets"), "semantic facet preparation refs")
    ref_by_namespace = {
        _text(row.get("namespace_id"), "semantic facet namespace"): _mapping(
            row, "semantic facet preparation ref"
        )
        for row in prep_refs
    }
    _require(
        len(ref_by_namespace) == len(prep_refs)
        and set(ref_by_namespace) == set(stores.stores_by_namespace)
        and prep.get("namespace_count") == len(ref_by_namespace)
        and prep.get("question_count") == len(inputs.rows),
        "semantic facet namespace population changed",
    )
    input_namespace_receipts = {
        namespace_id: (namespace_receipt, tuple(question_ids))
        for namespace_id, namespace_receipt, question_ids in inputs.namespaces
    }
    _require(
        set(input_namespace_receipts) == set(ref_by_namespace),
        "semantic facets differ from terminal namespaces",
    )
    descriptors: dict[str, SemanticFacetVectorDescriptor] = {}
    for namespace_id in sorted(ref_by_namespace):
        store = stores.stores_by_namespace[namespace_id]
        ref = ref_by_namespace[namespace_id]
        namespace_receipt, question_ids = input_namespace_receipts[namespace_id]
        parent_rows = tuple(row for row in inputs.rows if row.namespace_id == namespace_id)
        _require(
            store.namespace_receipt_sha256 == namespace_receipt
            and tuple(row.question_id for row in parent_rows) == question_ids,
            "semantic facet terminal namespace membership changed",
        )
        barrier_ref = barrier_by_store.get(store.namespace_store_id)
        _require(
            barrier_ref is not None
            and barrier_ref.get("preparation_checkpoint_sha256")
            == store.preparation_checkpoint_sha256
            and ref.get("namespace_store_id") == store.namespace_store_id,
            "semantic facet store escaped the staged barrier",
        )
        facet_path, checkpoint_path = _facet_paths(root, store.namespace_store_id)
        checkpoint = read_sealed_payload(
            checkpoint_path, label="semantic facet checkpoint"
        )
        checkpoint_payload = dict(checkpoint.payload)
        _self_seal(
            checkpoint_payload,
            "checkpoint_receipt_sha256",
            "semantic facet checkpoint",
        )
        _require(
            checkpoint.sha256 == ref.get("checkpoint_sha256")
            and checkpoint_payload.get("namespace_id") == namespace_id
            and checkpoint_payload.get("namespace_store_id") == store.namespace_store_id
            and checkpoint_payload.get("preparation_checkpoint_sha256")
            == store.preparation_checkpoint_sha256
            and checkpoint_payload.get("query_vector_artifact_sha256")
            == ref.get("query_vector_artifact_sha256")
            and checkpoint_payload.get("retrieval_query_vector_artifact_sha256")
            == barrier_ref.get("query_vector_artifact_sha256"),
            "semantic facet checkpoint escaped its preparation",
        )
        sealed_vector = read_sealed_payload(
            facet_path, label="semantic facet vector artifact"
        )
        expected_batch = semantic_facet_query_batch(
            tuple(row.dated_question for row in parent_rows)
        )
        descriptors[namespace_id] = _validate_loaded_facet_descriptor(
            sealed=sealed_vector,
            namespace_id=namespace_id,
            store=store,
            preparation_ref=ref,
            retrieval_vector_artifact_sha256=str(
                barrier_ref["query_vector_artifact_sha256"]
            ),
            embedding_identity_sha256=str(prep["embedding_identity_sha256"]),
            expected_batch=expected_batch,
        )
    _require(
        sum(len(row.query_batch) for row in descriptors.values())
        == prep.get("semantic_facet_count"),
        "semantic facet count changed",
    )
    return ConfirmationSemanticVectorRelease(
        preparation=preparation,
        release=release,
        barrier=barrier,
        descriptors_by_namespace=MappingProxyType(descriptors),
    )


def load_confirmation_semantic_facet_vectors(
    release: ConfirmationSemanticVectorRelease,
    namespace_id: str,
    /,
) -> Mapping[str, tuple[float, ...]]:
    """Load and authenticate exactly one namespace's supplemental vectors."""

    _require(
        type(release) is ConfirmationSemanticVectorRelease,
        "semantic vector release changed type",
    )
    descriptor = release.descriptors_by_namespace.get(namespace_id)
    _require(descriptor is not None, "semantic vector namespace is absent")
    assert descriptor is not None
    sealed = read_sealed_payload(
        descriptor.artifact_path, label="semantic facet vector artifact"
    )
    _require(sealed.sha256 == descriptor.artifact_sha256, "semantic facet bytes changed")
    rows = _list(sealed.payload.get("rows"), "semantic facet rows")
    vectors: dict[str, tuple[float, ...]] = {}
    for raw in rows:
        row = _mapping(raw, "semantic facet row")
        query = _text(row.get("query"), "semantic facet query")
        vector_raw = _list(row.get("vector"), "semantic facet vector")
        _require(
            row.get("query_sha256") == quote_sha256(query)
            and row.get("vector_sha256") == canonical_sha256(vector_raw)
            and len(vector_raw) == descriptor.dimension,
            "semantic facet vector changed during bounded load",
        )
        vectors[query] = tuple(float(item) for item in vector_raw)
    _require(
        tuple(vectors) == descriptor.query_batch,
        "semantic facet bounded replay changed order",
    )
    return MappingProxyType(vectors)


class ProtectedV3EvidenceAdapter(Protocol):
    """Narrow typed ancestry seam for exact provider-visible P owners."""

    @property
    def identity_sha256(self) -> str: ...

    @property
    def protected_owner_artifact_sha256(self) -> str: ...

    def protected_evidence(
        self, parent: terminal_boundary.TerminalParentRow, /
    ) -> tuple[LocalCitationBinding, ...]: ...


@dataclass(frozen=True, slots=True)
class SpecialistV3ProtectedEvidenceAdapter:
    """Rehydrate P from exact specialist-V3 and typed-final ancestors."""

    v3_plane: Any
    typed_plane: Any
    identity_sha256: str = field(init=False)
    protected_owner_artifact_sha256: str = field(init=False)
    _rows_by_source_receipt: Mapping[
        str, tuple[Mapping[str, Any], Mapping[str, Any], str]
    ] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from tools.confirmation_specialist_v3 import (  # noqa: PLC0415
            VerifiedConfirmationSpecialistV3Plane,
        )
        from tools.confirmation_typed_final import (  # noqa: PLC0415
            VerifiedConfirmationTypedFinalPlane,
        )
        _require(
            type(self.v3_plane) is VerifiedConfirmationSpecialistV3Plane
            and type(self.typed_plane) is VerifiedConfirmationTypedFinalPlane,
            "protected adapter requires exact V3 and typed-final planes",
        )
        constructions = tuple(
            _mapping(row, "specialist V2 construction row")
            for row in _list(
                self.v3_plane.v2_plane.construction_artifact.payload.get("questions"),
                "specialist V2 construction questions",
            )
        )
        compositions = tuple(
            _mapping(row, "typed composition row")
            for row in _list(
                self.typed_plane.composition_artifact.payload.get("questions"),
                "typed composition questions",
            )
        )
        v3_rows = tuple(self.v3_plane.result_rows)
        _require(
            bool(v3_rows)
            and len(v3_rows) == len(constructions) == len(compositions),
            "protected adapter ancestor populations differ",
        )
        construction_by_question: dict[str, Mapping[str, Any]] = {}
        composition_by_question: dict[str, Mapping[str, Any]] = {}
        for construction in constructions:
            question_id = _text(
                construction.get("question_id"), "specialist construction question"
            )
            _require(
                question_id not in construction_by_question,
                "specialist construction repeats a question",
            )
            construction_by_question[question_id] = construction
        for composition in compositions:
            _dated, _prediction, question_id = question_inputs(composition)
            _require(
                question_id not in composition_by_question,
                "typed composition repeats a question",
            )
            composition_by_question[question_id] = composition
        rows: dict[str, tuple[Mapping[str, Any], Mapping[str, Any], str]] = {}
        composition_sha = self.typed_plane.composition_artifact.sha256
        for answer in v3_rows:
            question_id = _text(answer.get("question_id"), "specialist V3 question")
            source_receipt = _sha(
                answer.get("source_row_sha256"), "specialist V3 source row"
            )
            _require(
                question_id in construction_by_question
                and question_id in composition_by_question
                and source_receipt not in rows,
                "protected adapter cannot join an exact ancestor row",
            )
            rows[source_receipt] = (
                construction_by_question[question_id],
                composition_by_question[question_id],
                composition_sha,
            )
        owner_sha = self.v3_plane.run_artifact.sha256
        identity = identity_sha256(
            {
                "format": "memory-condense-confirmation-specialist-v3-protected-evidence-adapter-v1",
                "specialist_v2_construction_artifact_sha256": (
                    self.v3_plane.v2_plane.construction_artifact.sha256
                ),
                "specialist_v3_replay_artifact_sha256": self.v3_plane.replay_artifact.sha256,
                "specialist_v3_run_artifact_sha256": owner_sha,
                "typed_composition_artifact_sha256": composition_sha,
                "typed_replay_artifact_sha256": self.typed_plane.replay_artifact.sha256,
                "typed_run_artifact_sha256": self.typed_plane.run_artifact.sha256,
            }
        )
        object.__setattr__(self, "protected_owner_artifact_sha256", owner_sha)
        object.__setattr__(self, "identity_sha256", identity)
        object.__setattr__(self, "_rows_by_source_receipt", MappingProxyType(rows))

    def protected_evidence(
        self, parent: terminal_boundary.TerminalParentRow, /
    ) -> tuple[LocalCitationBinding, ...]:
        _require(
            type(parent) is terminal_boundary.TerminalParentRow,
            "protected adapter parent changed type",
        )
        joined = self._rows_by_source_receipt.get(parent.source_row_receipt_sha256)
        _require(joined is not None, "protected adapter parent has no exact V3 ancestor")
        assert joined is not None
        construction, composition, composition_sha = joined
        _require(
            construction.get("question_id") == parent.question_id,
            "protected adapter source receipt escaped its question",
        )
        result = protected_evidence(
            construction_row=construction,
            composition_row=composition,
            composition_sha256=composition_sha,
            namespace_id=parent.namespace_id,
        )
        _require(
            type(result) is tuple
            and all(
                type(row) is LocalCitationBinding
                and row.namespace_id == parent.namespace_id
                for row in result
            ),
            "protected V3 evidence escaped typed ancestry",
        )
        return result


@dataclass(frozen=True, slots=True)
class SemanticNamespaceResources:
    residual_index: residual.SemanticResidualIndex
    episode_lookup: DiscourseStore
    episode_policy: EpisodeRetrievalPolicy
    episode_artifact_binding_receipt_sha256: str

    def __post_init__(self) -> None:
        _require(
            type(self.residual_index) is residual.SemanticResidualIndex
            and type(self.episode_lookup) is DiscourseStore
            and type(self.episode_policy) is EpisodeRetrievalPolicy,
            "semantic namespace resources changed type",
        )
        _sha(
            self.episode_artifact_binding_receipt_sha256,
            "semantic namespace episode binding",
        )


class SemanticNamespaceBackend(Protocol):
    @property
    def identity_sha256(self) -> str: ...

    def open_namespace(
        self, store: VerifiedNamespaceStore, /
    ) -> Any: ...


class ProductionSemanticNamespaceBackend:
    """Build one exact full-store semantic index in one read-only DB pass."""

    def __init__(self) -> None:
        self._identity_sha256 = identity_sha256(
            {
                "database_open_mode": "read_only_once_per_used_namespace",
                "episode_artifact_kinds": list(FIXED_INTERVAL_EPISODE_KINDS),
                "episode_policy": {
                    "max_anchor_episodes": 8,
                    "max_direct_fallbacks": 16,
                    "max_episode_seeds": 24,
                    "next_episodes": 1,
                    "previous_episodes": 1,
                },
                "format": "memory-condense-confirmation-semantic-namespace-backend-v1",
                "provider_calls": 0,
                "semantic_residual_policy_receipt_sha256": RESIDUAL_POLICY.receipt_sha256,
                "stored_vector_granularity": "exact_stored_chunk_vectors",
                "window_population": "exact_full_store",
            }
        )

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    @contextmanager
    def open_namespace(
        self, store: VerifiedNamespaceStore, /
    ) -> Iterator[SemanticNamespaceResources]:
        _require(type(store) is VerifiedNamespaceStore, "semantic store changed type")
        _verify_store_bytes(store)
        database_path = store.store_dir / "memory.db"
        database = Database(database_path, read_only=True)
        try:
            streams = scan_discourse_source_chunks(database)
            namespace = FrozenSourceNamespace.from_source_streams(
                snapshot_id=store.combined_store_receipt.snapshot_sha256,
                combined_store_receipt_sha256=(
                    store.combined_store_receipt.receipt_sha256
                ),
                source_streams=streams,
            )
            cache = cache_namespace_partitions(
                database,
                namespace,
                source_database_sha256=(
                    store.combined_store_receipt.target_database_sha256
                ),
                source_store_receipt_sha256=(
                    store.combined_store_receipt.receipt_sha256
                ),
            )
            window_index = build_full_store_window_index(cache)
            stored_vectors = residual.load_stored_chunk_vectors(database, window_index)
            semantic_index = residual.build_semantic_residual_index(
                window_index, stored_vectors, policy=RESIDUAL_POLICY
            )
            placeholders = ", ".join("?" for _ in FIXED_INTERVAL_EPISODE_KINDS)
            rows = database.execute(
                "SELECT DISTINCT a.artifact_id "
                "FROM discourse_artifacts AS a "
                "JOIN episodes AS e ON e.artifact_id = a.artifact_id "
                f"WHERE a.kind IN ({placeholders}) ORDER BY a.artifact_id",
                tuple(FIXED_INTERVAL_EPISODE_KINDS),
            ).fetchall()
            _require(
                len(rows) == 1,
                "semantic namespace requires exactly one populated fixed-interval episode artifact",
            )
            episode_store = DiscourseStore(database)
            artifact_id = str(rows[0][0])
            artifact = episode_store.get_artifact(artifact_id)
            _require(artifact is not None, "semantic namespace episode artifact is absent")
            assert artifact is not None
            episode_count = int(
                database.execute(
                    "SELECT COUNT(*) FROM episodes WHERE artifact_id = ?",
                    (artifact_id,),
                ).fetchone()[0]
            )
            _require(episode_count > 0, "semantic namespace episode artifact is empty")
            binding_body = {
                "artifact_id": artifact.artifact_id,
                "artifact_kind": artifact.kind,
                "artifact_receipt_sha256": identity_sha256(artifact.identity_payload()),
                "episode_count": episode_count,
                "resolution_mode": "authenticated_namespace_fixed_interval_auto",
            }
            binding = {
                **binding_body,
                "episode_artifact_binding_receipt_sha256": identity_sha256(
                    binding_body
                ),
            }
            policy = EpisodeRetrievalPolicy(
                artifact_id=artifact.artifact_id,
                max_anchor_episodes=8,
                previous_episodes=1,
                next_episodes=1,
                max_episode_seeds=24,
                max_direct_fallbacks=16,
            )
            yield SemanticNamespaceResources(
                residual_index=semantic_index,
                episode_lookup=episode_store,
                episode_policy=policy,
                episode_artifact_binding_receipt_sha256=binding[
                    "episode_artifact_binding_receipt_sha256"
                ],
            )
        finally:
            database.close()
            gc.collect()


@dataclass(frozen=True, slots=True)
class ConfirmationSemanticQuestionRow:
    parent: terminal_boundary.TerminalParentRow
    eligibility: SemanticResidualEligibilityDecision
    protected_evidence: tuple[LocalCitationBinding, ...]
    query: residual.SemanticResidualQuery | None
    residual_result: residual.SemanticResidualSearchResult | None
    local_result: SourceGroupReinjectionResult | None
    global_result: SemanticGlobalCompletionResult | None
    terminal_compilation: SemanticGlobalTerminalCompilation | None
    question_assay: Mapping[str, Any] | None
    receipt_sha256: str

    def __post_init__(self) -> None:
        _require(
            type(self.parent) is terminal_boundary.TerminalParentRow
            and type(self.eligibility) is SemanticResidualEligibilityDecision,
            "semantic question ancestry changed type",
        )
        _sha(self.receipt_sha256, "semantic question row")
        typed = (
            type(self.query) is residual.SemanticResidualQuery
            and type(self.residual_result) is residual.SemanticResidualSearchResult
            and type(self.local_result) is SourceGroupReinjectionResult
            and type(self.global_result) is SemanticGlobalCompletionResult
            and type(self.terminal_compilation) is SemanticGlobalTerminalCompilation
            and isinstance(self.question_assay, Mapping)
        )
        empty = all(
            value is None
            for value in (
                self.query,
                self.residual_result,
                self.local_result,
                self.global_result,
                self.terminal_compilation,
                self.question_assay,
            )
        ) and not self.protected_evidence
        _require(
            (self.eligibility.eligible and typed)
            or (not self.eligibility.eligible and empty),
            "semantic question eligibility/result relation changed",
        )


@dataclass(frozen=True, slots=True)
class ConfirmationSemanticPlaneMaterialization:
    artifact: SealedPayload
    terminal_plan_export: terminal_boundary.ConfirmationTerminalV5PlanExport
    rows: tuple[ConfirmationSemanticQuestionRow, ...]
    checkpoint_paths: tuple[Path, ...]
    checkpoint_sha256_by_namespace_receipt: Mapping[str, str]
    created_checkpoint_count: int
    reused_checkpoint_count: int
    physical_provider_calls: int = 0

    def __post_init__(self) -> None:
        _require(
            type(self.artifact) is SealedPayload
            and type(self.terminal_plan_export)
            is terminal_boundary.ConfirmationTerminalV5PlanExport
            and bool(self.rows)
            and len(self.checkpoint_paths)
            == len(self.checkpoint_sha256_by_namespace_receipt)
            == self.created_checkpoint_count + self.reused_checkpoint_count
            and self.physical_provider_calls == 0,
            "semantic materialization changed",
        )


def _eligibility(
    parent: terminal_boundary.TerminalParentRow,
) -> SemanticResidualEligibilityDecision:
    for value, label in (
        (parent.answer_row, "answer"),
        (parent.construction_row, "construction"),
        (parent.prior_answer_row, "prior answer"),
        (parent.reconciliation_row, "reconciliation"),
    ):
        if value is not None:
            _assert_routing_neutral(value, f"eligibility_{label.replace(' ', '_')}")
    decision = evaluate_semantic_residual_eligibility(
        parent.answer_row,
        parent.construction_row,
        prior_answer_row=parent.prior_answer_row,
        reconciliation_row=parent.reconciliation_row,
        policy=ELIGIBILITY_POLICY,
    )
    replayed = replay_semantic_residual_eligibility(
        parent.answer_row,
        parent.construction_row,
        decision,
        prior_answer_row=parent.prior_answer_row,
        reconciliation_row=parent.reconciliation_row,
        policy=ELIGIBILITY_POLICY,
    )
    _require(
        replayed.projection() == decision.projection(),
        "semantic eligibility changed during replay",
    )
    return decision


def _semantic_row_projection(
    parent: terminal_boundary.TerminalParentRow,
    eligibility: SemanticResidualEligibilityDecision,
    *,
    protected_evidence: Sequence[LocalCitationBinding] = (),
    query: residual.SemanticResidualQuery | None = None,
    residual_result: residual.SemanticResidualSearchResult | None = None,
    local_result: SourceGroupReinjectionResult | None = None,
    global_result: SemanticGlobalCompletionResult | None = None,
    terminal_compilation: SemanticGlobalTerminalCompilation | None = None,
    question_assay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "eligibility": eligibility.projection(),
        "format": f"{MATERIALIZATION_FORMAT}-question-v1",
        "global_result": None if global_result is None else global_result.projection(),
        "local_result": None if local_result is None else local_result.projection(),
        "namespace_id": parent.namespace_id,
        "new_provider_calls": 0,
        "parent_row_receipt_sha256": parent.row_receipt_sha256,
        "protected_evidence": [row.projection() for row in protected_evidence],
        "question_assay": None if question_assay is None else dict(question_assay),
        "query": None if query is None else query.projection(),
        "residual_result": (
            None if residual_result is None else residual_result.projection()
        ),
        "retained_transformer_token_state_bytes": 0,
        "terminal_compilation": (
            None
            if terminal_compilation is None
            else terminal_compilation.projection(include_local=True)
        ),
    }
    return {**body, "row_receipt_sha256": identity_sha256(body)}


def _question_vectors(
    parent: terminal_boundary.TerminalParentRow,
    vectors: Mapping[str, tuple[float, ...]],
) -> tuple[tuple[float, ...], ...]:
    facets = residual.semantic_residual_query_facets(parent.dated_question)
    _require(
        all(facet in vectors for facet in facets),
        "semantic facet artifact omitted a required question facet",
    )
    return tuple(vectors[facet] for facet in facets)


def _build_answer_plan_and_assay(
    *,
    source_index: int,
    parent: terminal_boundary.TerminalParentRow,
    semantic_index: residual.SemanticResidualIndex,
    query: residual.SemanticResidualQuery,
    protected: tuple[LocalCitationBinding, ...],
    residual_result: residual.SemanticResidualSearchResult,
    local_result: SourceGroupReinjectionResult,
    global_result: SemanticGlobalCompletionResult,
    terminal_prompt: Mapping[str, Any],
    protected_union: tuple[LocalCitationBinding, ...],
    sealed_sources: TerminalSealedSources,
    episode_artifact_binding_receipt_sha256: str,
) -> tuple[SemanticGlobalTerminalCompilation, dict[str, Any]]:
    provider_input = _mapping(
        terminal_prompt.get("provider_input"), "R7 terminal provider input"
    )
    selected_owner_rows = tuple(
        _mapping(row, "selected protected-owner evidence")
        for row in _list(
            provider_input.get("protected_owner_evidence"),
            "selected protected-owner evidence",
        )
    )
    selected_owners = load_selected_protected_owner_evidence(selected_owner_rows)
    compilation = compile_semantic_global_terminal(
        dated_question=parent.dated_question,
        parent_prediction=parent.parent_prediction,
        residual_index=semantic_index,
        query=query,
        protected_owner_universe_bindings=protected,
        selected_protected_owner_evidence=selected_owners,
        residual_result=residual_result,
        local_result=local_result,
        global_result=global_result,
        sealed_sources=sealed_sources,
        policy=TERMINAL_POLICY,
        enable_selected_evidence_discourse_links=True,
        enable_post_dedup_backfill=True,
    )
    replayed = replay_semantic_global_terminal(
        dated_question=parent.dated_question,
        parent_prediction=parent.parent_prediction,
        residual_index=semantic_index,
        query=query,
        protected_owner_universe_bindings=protected,
        selected_protected_owner_evidence=selected_owners,
        residual_result=residual_result,
        local_result=local_result,
        global_result=global_result,
        sealed_sources=sealed_sources,
        sealed_compilation=compilation,
        policy=TERMINAL_POLICY,
    )
    _require(
        replayed.projection(include_local=True)
        == compilation.projection(include_local=True),
        "terminal-v5 compilation changed during replay",
    )
    core = compile_answer_plan_core(
        dated_question=parent.dated_question,
        parent_prediction=parent.parent_prediction,
        residual_index=semantic_index,
        query=query,
        protected_owner_universe_bindings=protected,
        selected_protected_owner_evidence_rows=selected_owner_rows,
        residual_result=residual_result,
        local_result=local_result,
        global_result=global_result,
        sealed_sources=sealed_sources,
        policy=TERMINAL_POLICY,
        terminal_mode=TERMINAL_COMPILATION_MODE_V5,
    )
    _require(
        core.get("terminal_compilation")
        == compilation.projection(include_local=True),
        "terminal-v5 core differs from its typed compilation",
    )
    plan_body = {
        **core,
        "dated_question_sha256": quote_sha256(parent.dated_question),
        "ordinal": source_index,
        "question_id": parent.question_id,
        "question_sha256": quote_sha256(parent.question),
    }
    plan = {
        **plan_body,
        "answer_plan_receipt_sha256": identity_sha256(plan_body),
    }
    assay_body = {
        "dated_question_sha256": quote_sha256(parent.dated_question),
        "episode_artifact_binding_receipt_sha256": (
            episode_artifact_binding_receipt_sha256
        ),
        "global_completion": global_result.projection(),
        "namespace_id": parent.namespace_id,
        "new_provider_calls": 0,
        "ordinal": source_index,
        "protected_union_binding_receipt_sha256s": [
            row.receipt_sha256 for row in protected_union
        ],
        "protected_union_count": len(protected_union),
        "question_id": parent.question_id,
        "question_sha256": quote_sha256(parent.question),
        "r7_exact_question_rebuilt": True,
        "r7_question_receipt_sha256": residual_result.receipt_sha256,
        "retained_transformer_token_state_bytes": 0,
        "terminal_answer_plan": plan,
        "v6_exact_replay_identical": True,
        "v6_frontier": local_result.frontier.projection(),
        "v6_local_binding_receipt_sha256s": [
            row.receipt_sha256 for row in local_result.local_bindings
        ],
        "v6_result_receipt_sha256": local_result.receipt_sha256,
        "v7_exact_replay_identical": True,
    }
    assay = {
        **assay_body,
        "question_assay_receipt_sha256": identity_sha256(assay_body),
    }
    # This is the compatibility-only positional boundary.  Routing and every
    # mechanism call above were completed without an ordinal.
    terminal_boundary._validate_frozen_v5_question_plan(  # noqa: SLF001
        parent, source_index, assay
    )
    return compilation, assay


def _checkpoint_path(root: Path, store: VerifiedNamespaceStore) -> Path:
    return root / "semantic-planes" / "checkpoints" / f"{store.namespace_store_id}.json"


def materialize_confirmation_semantic_planes(
    inputs: terminal_boundary.ConfirmationTerminalInputs,
    stores: VerifiedNamespaceStoreSet,
    vector_release: ConfirmationSemanticVectorRelease,
    protected_adapter: ProtectedV3EvidenceAdapter,
    *,
    output_root: str | Path,
    backend: SemanticNamespaceBackend | None = None,
    expected_checkpoint_sha256_by_namespace_receipt: Mapping[str, str] | None = None,
) -> ConfirmationSemanticPlaneMaterialization:
    """Materialize exact eligible R/L/G/v5 rows over arbitrary namespaces."""

    _require(
        type(inputs) is terminal_boundary.ConfirmationTerminalInputs
        and type(stores) is VerifiedNamespaceStoreSet
        and type(vector_release) is ConfirmationSemanticVectorRelease,
        "semantic materialization inputs changed type",
    )
    policy_bindings = _policy_projection(inputs)
    _require(
        stores.policy_manifest_sha256 == inputs.policy.sha256
        and stores.treatment_preflight_sha256 == inputs.treatment_preflight.sha256
        and stores.barrier_sha256 == vector_release.barrier.sha256
        and set(stores.stores_by_namespace)
        == set(vector_release.descriptors_by_namespace),
        "semantic store/vector/input ancestry changed",
    )
    adapter_identity = _sha(
        protected_adapter.identity_sha256, "protected evidence adapter"
    )
    protected_owner_sha = _sha(
        protected_adapter.protected_owner_artifact_sha256,
        "protected owner artifact",
    )
    active_backend = (
        ProductionSemanticNamespaceBackend() if backend is None else backend
    )
    backend_identity = _sha(active_backend.identity_sha256, "semantic backend")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    decisions = tuple(_eligibility(parent) for parent in inputs.rows)
    grouped: dict[str, list[tuple[int, terminal_boundary.TerminalParentRow]]] = (
        defaultdict(list)
    )
    for source_index, (parent, decision) in enumerate(
        zip(inputs.rows, decisions, strict=True)
    ):
        if decision.eligible:
            grouped[parent.namespace_id].append((source_index, parent))

    rows_by_parent: dict[str, ConfirmationSemanticQuestionRow] = {}
    for namespace_id in sorted(grouped):
        store = stores.stores_by_namespace.get(namespace_id)
        descriptor = vector_release.descriptors_by_namespace.get(namespace_id)
        _require(
            store is not None and descriptor is not None,
            "eligible semantic namespace is absent",
        )
        assert store is not None and descriptor is not None
        facet_vectors = load_confirmation_semantic_facet_vectors(
            vector_release, namespace_id
        )
        with active_backend.open_namespace(store) as resources:
            _require(
                type(resources) is SemanticNamespaceResources
                and resources.residual_index.namespace_id == namespace_id
                and resources.residual_index.policy.receipt_sha256
                == RESIDUAL_POLICY.receipt_sha256,
                "semantic backend returned another namespace/policy",
            )
            semantic_index = resources.residual_index
            for source_index, parent in grouped[namespace_id]:
                decision = decisions[source_index]
                question_vectors = _question_vectors(parent, facet_vectors)
                query = residual.compile_semantic_residual_query(
                    semantic_index,
                    parent.dated_question,
                    query_vectors=question_vectors,
                    query_vector_artifact_sha256=descriptor.artifact_sha256,
                )
                protected = tuple(protected_adapter.protected_evidence(parent))
                _require(
                    len({row.receipt_sha256 for row in protected}) == len(protected)
                    and all(row.namespace_id == namespace_id for row in protected),
                    "protected evidence population changed",
                )
                residual_result = residual.search_semantic_residual(
                    semantic_index, query, protected_evidence=protected
                )
                residual_replay = residual.replay_semantic_residual_search(
                    semantic_index,
                    query,
                    residual_result,
                    protected_evidence=protected,
                )
                _require(
                    residual_replay.projection() == residual_result.projection(),
                    "semantic residual search changed during replay",
                )
                _require(
                    not residual_result.fallback_required
                    and bool(residual_result.evidence),
                    "eligible semantic row has no packable novel residual evidence",
                )
                terminal_prompt, terminal_reason = (
                    build_separate_terminal_prompt(
                        dated_question=parent.dated_question,
                        current_prediction=parent.parent_prediction,
                        result=residual_result,
                        residual_index=semantic_index,
                        protected_evidence=protected,
                        policy=ELIGIBILITY_POLICY,
                        protected_owner_token_cap=2_400,
                    )
                )
                _require(
                    terminal_reason == "none" and terminal_prompt is not None,
                    "eligible semantic row cannot construct its R7 terminal",
                )
                assert terminal_prompt is not None
                handle_bindings, handle_groups, retained_sources = (
                    selected_handle_bindings(
                        semantic_index,
                        residual_result,
                        protected,
                        terminal_prompt,
                    )
                )
                selection = authenticate_source_group_selection(
                    semantic_index,
                    handle_bindings,
                    group_universe_source_ids=retained_sources,
                    selected_handle_groups=handle_groups,
                )
                local_result = search_source_group_reinjection(
                    semantic_index,
                    query,
                    selection,
                    protected_handle_bindings=handle_bindings,
                    policy=LOCAL_POLICY,
                    episode_lookup=resources.episode_lookup,
                    episode_policy=resources.episode_policy,
                )
                local_replay = replay_source_group_reinjection(
                    semantic_index,
                    query,
                    selection,
                    local_result,
                    protected_handle_bindings=handle_bindings,
                    episode_lookup=resources.episode_lookup,
                    episode_policy=resources.episode_policy,
                )
                _require(
                    local_replay.projection() == local_result.projection(),
                    "local reinjection changed during replay",
                )
                protected_union = ordered_protected_union(
                    protected,
                    residual_result.local_bindings,
                    local_result.local_bindings,
                )
                global_request = compile_semantic_global_completion_request(
                    query,
                    prior_needs_global_search=True,
                    operand_closure_missing=(
                        query.operator_spec.requires_complete_frontier
                    ),
                    local_frontier_unresolved=local_result.frontier.needs_global_search,
                )
                global_result = search_semantic_global_completion(
                    semantic_index,
                    query,
                    global_request,
                    policy=GLOBAL_POLICY,
                    protected_evidence=protected_union,
                )
                global_replay = replay_semantic_global_completion(
                    semantic_index,
                    query,
                    global_request,
                    global_result,
                    protected_evidence=protected_union,
                )
                _require(
                    global_replay.projection() == global_result.projection(),
                    "global completion changed during replay",
                )
                sealed_sources = TerminalSealedSources(
                    protected_owner_artifact_sha256=protected_owner_sha,
                    residual_artifact_sha256=descriptor.artifact_sha256,
                    parent_artifact_sha256=inputs.parent_population.sha256,
                )
                terminal_compilation, question_assay = _build_answer_plan_and_assay(
                    source_index=source_index,
                    parent=parent,
                    semantic_index=semantic_index,
                    query=query,
                    protected=protected,
                    residual_result=residual_result,
                    local_result=local_result,
                    global_result=global_result,
                    terminal_prompt=terminal_prompt,
                    protected_union=protected_union,
                    sealed_sources=sealed_sources,
                    episode_artifact_binding_receipt_sha256=(
                        resources.episode_artifact_binding_receipt_sha256
                    ),
                )
                projection = _semantic_row_projection(
                    parent,
                    decision,
                    protected_evidence=protected,
                    query=query,
                    residual_result=residual_result,
                    local_result=local_result,
                    global_result=global_result,
                    terminal_compilation=terminal_compilation,
                    question_assay=question_assay,
                )
                row = ConfirmationSemanticQuestionRow(
                    parent=parent,
                    eligibility=decision,
                    protected_evidence=protected,
                    query=query,
                    residual_result=residual_result,
                    local_result=local_result,
                    global_result=global_result,
                    terminal_compilation=terminal_compilation,
                    question_assay=MappingProxyType(question_assay),
                    receipt_sha256=str(projection["row_receipt_sha256"]),
                )
                _require(
                    parent.row_receipt_sha256 not in rows_by_parent,
                    "semantic parent row repeated",
                )
                rows_by_parent[parent.row_receipt_sha256] = row

    for parent, decision in zip(inputs.rows, decisions, strict=True):
        if decision.eligible:
            continue
        projection = _semantic_row_projection(parent, decision)
        rows_by_parent[parent.row_receipt_sha256] = ConfirmationSemanticQuestionRow(
            parent=parent,
            eligibility=decision,
            protected_evidence=(),
            query=None,
            residual_result=None,
            local_result=None,
            global_result=None,
            terminal_compilation=None,
            question_assay=None,
            receipt_sha256=str(projection["row_receipt_sha256"]),
        )
    ordered_rows = tuple(rows_by_parent[parent.row_receipt_sha256] for parent in inputs.rows)
    _require(len(ordered_rows) == len(inputs.rows), "semantic row population changed")

    expected_checkpoints = (
        None
        if expected_checkpoint_sha256_by_namespace_receipt is None
        else dict(expected_checkpoint_sha256_by_namespace_receipt)
    )
    if expected_checkpoints is not None:
        _require(
            set(expected_checkpoints)
            == {store.namespace_receipt_sha256 for store in stores.stores_by_namespace.values()},
            "semantic checkpoint expectation population changed",
        )
    checkpoint_paths: list[Path] = []
    checkpoint_shas: dict[str, str] = {}
    created = reused = 0
    for namespace_id, namespace_receipt, question_ids in inputs.namespaces:
        store = stores.stores_by_namespace[namespace_id]
        namespace_rows = tuple(
            row for row in ordered_rows if row.parent.namespace_id == namespace_id
        )
        _require(
            tuple(row.parent.question_id for row in namespace_rows) == question_ids,
            "semantic checkpoint namespace membership changed",
        )
        row_projections = [
            _semantic_row_projection(
                row.parent,
                row.eligibility,
                protected_evidence=row.protected_evidence,
                query=row.query,
                residual_result=row.residual_result,
                local_result=row.local_result,
                global_result=row.global_result,
                terminal_compilation=row.terminal_compilation,
                question_assay=row.question_assay,
            )
            for row in namespace_rows
        ]
        body = {
            "backend_identity_sha256": backend_identity,
            "format": CHECKPOINT_FORMAT,
            "gold_loaded": False,
            "namespace_id": namespace_id,
            "namespace_receipt_sha256": namespace_receipt,
            "namespace_store_identity_sha256": store.store_identity_sha256,
            "new_provider_calls": 0,
            "ordered_parent_row_receipts_sha256": identity_sha256(
                [row.parent.row_receipt_sha256 for row in namespace_rows]
            ),
            "policy_receipts": dict(EXPECTED_POLICY_RECEIPTS),
            "protected_adapter_identity_sha256": adapter_identity,
            "question_count": len(namespace_rows),
            "rows": row_projections,
            "semantic_facet_vector_artifact_sha256": (
                vector_release.descriptors_by_namespace[namespace_id].artifact_sha256
            ),
            "store_set_identity_sha256": stores.identity_sha256,
        }
        path = _checkpoint_path(root, store)
        checkpoint, was_created = _sealed_payload(
            path,
            {**body, "checkpoint_receipt_sha256": identity_sha256(body)},
            label="semantic plane namespace checkpoint",
        )
        if expected_checkpoints is not None:
            _require(
                checkpoint.sha256 == expected_checkpoints[namespace_receipt],
                "semantic namespace checkpoint differs from external seal",
            )
        checkpoint_paths.append(path)
        checkpoint_shas[namespace_receipt] = checkpoint.sha256
        created += int(was_created)
        reused += int(not was_created)

    assays = tuple(
        dict(row.question_assay)
        for row in ordered_rows
        if row.eligibility.eligible and row.question_assay is not None
    )
    _require(
        len(assays) == sum(row.eligibility.eligible for row in ordered_rows),
        "eligible assay is absent",
    )
    plan_artifact, _plan_created = (
        terminal_boundary.publish_confirmation_terminal_v5_plan_export(
            inputs,
            frozen_question_assays=assays,
            output_path=root / TERMINAL_PLAN_EXPORT_NAME,
        )
    )
    plan_export = terminal_boundary.load_confirmation_terminal_v5_plan_export(
        inputs,
        path=plan_artifact.path,
        expected_sha256=plan_artifact.sha256,
    )
    materialization_body = {
        "backend_identity_sha256": backend_identity,
        "checkpoint_sha256_by_namespace_receipt": checkpoint_shas,
        "eligible_question_count": sum(row.eligibility.eligible for row in ordered_rows),
        "format": MATERIALIZATION_FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "ordered_parent_row_receipts_sha256": identity_sha256(
            [row.parent.row_receipt_sha256 for row in ordered_rows]
        ),
        "ordered_semantic_row_receipts_sha256": identity_sha256(
            [row.receipt_sha256 for row in ordered_rows]
        ),
        "parent_population_sha256": inputs.parent_population.sha256,
        "policy_bindings": dict(policy_bindings),
        "policy_receipts": dict(EXPECTED_POLICY_RECEIPTS),
        "protected_adapter_identity_sha256": adapter_identity,
        "question_count": len(ordered_rows),
        "retained_transformer_token_state_bytes": 0,
        "semantic_vector_release_sha256": vector_release.release.sha256,
        "status": "complete",
        "store_set_identity_sha256": stores.identity_sha256,
        "terminal_v5_plan_export_sha256": plan_export.artifact.sha256,
    }
    _assert_routing_neutral(materialization_body, "semantic_materialization")
    artifact, _materialization_created = _sealed_payload(
        root / MATERIALIZATION_NAME,
        {
            **materialization_body,
            "artifact_identity_sha256": identity_sha256(materialization_body),
        },
        label="semantic plane materialization",
    )
    return ConfirmationSemanticPlaneMaterialization(
        artifact=artifact,
        terminal_plan_export=plan_export,
        rows=ordered_rows,
        checkpoint_paths=tuple(checkpoint_paths),
        checkpoint_sha256_by_namespace_receipt=MappingProxyType(checkpoint_shas),
        created_checkpoint_count=created,
        reused_checkpoint_count=reused,
    )


def replay_confirmation_semantic_planes(
    inputs: terminal_boundary.ConfirmationTerminalInputs,
    stores: VerifiedNamespaceStoreSet,
    vector_release: ConfirmationSemanticVectorRelease,
    protected_adapter: ProtectedV3EvidenceAdapter,
    *,
    output_root: str | Path,
    expected_materialization_sha256: str,
    expected_checkpoint_sha256_by_namespace_receipt: Mapping[str, str],
    backend: SemanticNamespaceBackend | None = None,
) -> ConfirmationSemanticPlaneMaterialization:
    """Rebuild all typed results and require identical durable receipts."""

    result = materialize_confirmation_semantic_planes(
        inputs,
        stores,
        vector_release,
        protected_adapter,
        output_root=output_root,
        backend=backend,
        expected_checkpoint_sha256_by_namespace_receipt=(
            expected_checkpoint_sha256_by_namespace_receipt
        ),
    )
    _require(
        result.artifact.sha256
        == _sha(expected_materialization_sha256, "semantic materialization"),
        "semantic materialization differs from external seal",
    )
    root = Path(output_root).resolve()
    replay, _created = _sealed_payload(
        root / REPLAY_NAME,
        result.artifact.payload,
        label="semantic plane replay",
    )
    _require(
        replay.sha256 == result.artifact.sha256,
        "semantic materialization replay changed bytes",
    )
    return result


__all__ = [
    "CHECKPOINT_FORMAT",
    "ConfirmationSemanticFacetPreparation",
    "ConfirmationSemanticPlaneMaterialization",
    "ConfirmationSemanticPlanesError",
    "ConfirmationSemanticQuestionRow",
    "ConfirmationSemanticVectorRelease",
    "ELIGIBILITY_POLICY",
    "EXPECTED_POLICY_RECEIPTS",
    "FACET_CHECKPOINT_FORMAT",
    "FACET_PREPARATION_FORMAT",
    "FACET_PREPARATION_NAME",
    "FACET_RELEASE_FORMAT",
    "FACET_RELEASE_NAME",
    "FACET_VECTOR_FORMAT",
    "GLOBAL_POLICY",
    "LOCAL_POLICY",
    "MATERIALIZATION_FORMAT",
    "MATERIALIZATION_NAME",
    "ProductionSemanticNamespaceBackend",
    "ProtectedV3EvidenceAdapter",
    "REPLAY_NAME",
    "RESIDUAL_POLICY",
    "SemanticFacetVectorDescriptor",
    "SemanticNamespaceBackend",
    "SemanticNamespaceResources",
    "SpecialistV3ProtectedEvidenceAdapter",
    "TERMINAL_PLAN_EXPORT_NAME",
    "TERMINAL_POLICY",
    "confirmation_semantic_facet_pre_release_hook",
    "load_confirmation_semantic_facet_vectors",
    "load_confirmation_semantic_vector_release",
    "materialize_confirmation_semantic_planes",
    "prepare_confirmation_semantic_facet_vectors",
    "publish_confirmation_semantic_facet_release",
    "replay_confirmation_semantic_planes",
    "semantic_facet_query_batch",
]
