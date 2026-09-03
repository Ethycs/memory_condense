#!/usr/bin/env python3
"""Construct the frozen local retrieval runtime without validation inputs.

The confirmation treatment may consume only the policy-v5-r3 treatment
projection and the sanitized confirmation population.  In particular, it may
not reopen the historical validation policy merely to recover S0--S3 runtime
settings.  This module therefore carries the already-frozen, label-free
retrieval controls as ordinary code constants and verifies their complete
resolved projections before constructing any model runtime.

Construction opens the owned BGE execution binding, but does not load Qwen,
open a store, or contact a provider.  The top-level executor therefore creates
this runtime lazily only for the ingest/staged phases.  BGE remains owned by
that binding until the staged coordinator seals its release barrier; Qwen can
be loaded only after that barrier.  A resumed post-staged process must not
construct this initial runtime.  Query expansion has a separate, explicit
fresh-BGE session below and closes it before returning.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain.discourse import ClosurePolicy, identity_sha256
from memory_condense.eval.diffuse_compilation import DiffuseCompilationPolicy
from memory_condense.eval.diffuse_longmemeval_runtime import (
    DiffuseLongMemEvalExecutionBinding,
    DiffuseLongMemEvalRuntimeConfig,
    build_diffuse_longmemeval_execution_binding,
)
from memory_condense.eval.schemas import ChunkerConfig, EvalConfig, RetrievalConfig
from memory_condense.search.episodes import EpisodeRetrievalPolicy
from memory_condense.search.packing.context_packer import ContextBudget
from tools.confirmation_cumulative_retrieval import STAGED_PRODUCTION_MODE
from tools.confirmation_namespace_store_adapter import (
    ProductionBaseStoreBackend,
    build_production_source_treatment_contract,
)
from tools.confirmation_staged_cumulative_coordinator import (
    ProductionStagedPreparationBackend,
    ProductionStagedQwenRuntimeFactory,
    ProductionStagedRetrievalBackendFactory,
)
from tools.matched_eval.query_expansion import ExistingPartitionHybridSearch
from tools.confirmation_canonical import canonical_sha256


FORMAT = "memory-condense-confirmation-production-runtime-v1"
RUNTIME_POLICY_FORMAT = f"{FORMAT}-policy-v1"

# These digests were frozen from the fully resolved validation-v3 retrieval
# projection before confirmation was opened.  They bind controls, not a
# validation population or artifact path.
FROZEN_RETRIEVAL_POLICY_SHA256 = (
    "0abb42db8fd35b566029be135630af34fa42ee9a464ef8c4a889948729cbab13"
)
FROZEN_FULL_CONFIG_SHA256 = (
    "815a20173e62f907d8332b10f720ed47c2588b7b0e71ada4cfb7bfdbbdbaf5d4"
)
FROZEN_SOURCE_CONFIG_SHA256 = (
    "efa25bba2c08e490049a44f368d96fcfe8a21e827d6202221ccad69c2b93512f"
)
FROZEN_SOURCE_RETRIEVAL_POLICY_SHA256 = (
    "573a78a8af398a23275355221051f93b3c7ebea3c03ce477a57352801bc5fb5e"
)

MAX_CONTEXT_TOKENS = 7_000
MAX_PROMPT_TOKENS = 8_000
RESPONDER_OUTPUT_TOKEN_RESERVE = 256
SOURCE_ROUTER_MAX_SOURCES = 64
SOURCE_ROUTER_RRF_CONSTANT = 60

_DIRECT_EPISODE_CONTROLS: Mapping[str, Any] = MappingProxyType(
    {
        "max_anchor_episodes": 96,
        "previous_episodes": 1,
        "next_episodes": 1,
        "max_episode_seeds": 256,
        "max_direct_fallbacks": 96,
    }
)
_CLOSURE_CONTROLS: Mapping[str, Any] = MappingProxyType(
    {
        "max_hops": 3,
        "max_units": 1024,
        "max_relations": 2048,
        "max_degree": 32,
        "max_episode_neighbors": 2,
        "max_frontier": 1024,
        "max_bundles": 256,
        "beam_width": 128,
        "min_relation_confidence": 0.5,
    }
)


class ConfirmationProductionRuntimeError(ValueError):
    """Frozen local runtime controls or ownership changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationProductionRuntimeError(message)


def _causal_graph_context_budget(retrieval: RetrievalConfig) -> ContextBudget:
    """Prediction-safe copy of the frozen causal-graph packing projection."""

    return ContextBudget(
        recent_window_tokens=0,
        memory_header_tokens=0,
        expansion_tokens=retrieval.consolidation_expansion_tokens,
        max_expansions=retrieval.k + retrieval.neighbor_slots + retrieval.source_slots,
        max_consolidation_expansions=retrieval.consolidation_chunk_slots,
        budget_aware_expansions=retrieval.consolidation_budget_aware_packing,
        source_diverse_expansions=retrieval.consolidation_source_diverse_packing,
        query_aware_sentence_expansions=(
            retrieval.consolidation_query_aware_sentence_packing
        ),
        max_sentences_per_expansion=(
            retrieval.consolidation_max_sentences_per_expansion
        ),
        information_gain_expansions=(
            retrieval.consolidation_information_gain_packing
        ),
        min_information_gain_per_token=(
            retrieval.consolidation_min_information_gain_per_token
        ),
        source_metadata_expansions=(
            retrieval.consolidation_source_metadata_packing
        ),
    )


class _OwnedNamespaceQueryRetriever:
    """Open one authenticated namespace index for one search call at a time."""

    def __init__(self, session: "ConfirmationQueryRetrieverSession", snapshot: Any) -> None:
        self._session = session
        self._snapshot = snapshot
        self.namespace = snapshot.namespace

    def search_many(self, queries: Any, *, budget: Any) -> Any:
        return self._session.search_namespace(
            self._snapshot,
            queries=queries,
            budget=budget,
        )


class ConfirmationQueryRetrieverSession:
    """Fresh-BGE, post-Qwen query-expansion retrieval ownership boundary.

    The query-expansion API requires a complete namespace-to-retriever map.
    The map returned here is lazy: each value opens, searches, and closes its
    own immutable namespace store, so at most one HNSW/SQLite index is resident
    at a time while all values share one BGE embedder.  Closing the session
    releases BGE before any later provider/terminal phase.
    """

    def __init__(self, *, runtime: "ConfirmationProductionRuntime", context: Any) -> None:
        from tools.confirmation_query_expansion_adapter import (  # noqa: PLC0415
            ConfirmationQueryExpansionContext,
        )

        _require(
            type(context) is ConfirmationQueryExpansionContext,
            "query retriever session requires the exact confirmation context",
        )
        context.revalidate_store_bytes()
        self._runtime = runtime
        self._context = context
        self._lock = threading.Lock()
        self._closed = False
        self._active_indexes = 0
        self._maximum_active_indexes = 0
        self._namespace_open_count = 0
        self._namespace_close_count = 0
        self._search_call_count = 0
        self.retrievers = MappingProxyType(
            {
                snapshot.namespace.namespace_id: _OwnedNamespaceQueryRetriever(
                    self, snapshot
                )
                for snapshot in context.namespace_snapshots
            }
        )
        _require(
            set(self.retrievers) == set(context.store_dirs_by_namespace),
            "query retriever namespace population differs",
        )
        identity_body = {
            "format": f"{FORMAT}-query-retriever-session-v1",
            "runtime_identity_sha256": runtime.identity_sha256,
            "protected_plane_sha256": context.protected_artifact.sha256,
            "cumulative_artifact_sha256": context.cumulative_artifact.sha256,
            "namespace_store_bindings": [
                {
                    "database_sha256": context.database_sha256_by_namespace[
                        snapshot.namespace.namespace_id
                    ],
                    "index_sha256": context.index_sha256_by_namespace[
                        snapshot.namespace.namespace_id
                    ],
                    "namespace_id": snapshot.namespace.namespace_id,
                    "namespace_store_id": snapshot.namespace_store_id,
                }
                for snapshot in context.namespace_snapshots
            ],
        }
        self.identity_sha256 = canonical_sha256(identity_body)

    def __enter__(self) -> "ConfirmationQueryRetrieverSession":
        _require(not self._closed, "query retriever session is closed")
        return self

    def search_namespace(self, snapshot: Any, *, queries: Any, budget: Any) -> Any:
        _require(not self._closed, "query retriever session is closed")
        with self._lock:
            _require(self._active_indexes == 0, "namespace query indexes overlapped")
            self._context.revalidate_store_bytes()
            condenser = MemoryCondenser(
                data_dir=snapshot.store_dir,
                chunker_min_tokens=self._runtime.config.chunker.min_tokens,
                chunker_max_tokens=self._runtime.config.chunker.max_tokens,
                auto_extract=False,
                budget=_causal_graph_context_budget(self._runtime.config.retrieval),
                embedder=self._runtime.binding.embedder,
                persist_index_on_close=False,
                retriever_max_elements=max(1, len(snapshot.namespace.chunk_to_source)),
                read_only=True,
            )
            self._active_indexes = 1
            self._maximum_active_indexes = max(
                self._maximum_active_indexes, self._active_indexes
            )
            self._namespace_open_count += 1
            try:
                result = ExistingPartitionHybridSearch(
                    condenser, snapshot.namespace
                ).search_many(queries, budget=budget)
                self._search_call_count += 1
                return result
            finally:
                condenser.close()
                self._namespace_close_count += 1
                self._active_indexes = 0
                self._context.revalidate_store_bytes()

    def close(self) -> None:
        if self._closed:
            return
        _require(self._active_indexes == 0, "cannot close an active query index")
        close = getattr(self._runtime.binding.embedder, "close", None)
        _require(callable(close), "query BGE embedder has no owned close boundary")
        close()
        self._closed = True
        self._context.revalidate_store_bytes()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def audit_projection(self) -> Mapping[str, Any]:
        body = {
            "format": f"{FORMAT}-query-retriever-session-audit-v1",
            "session_identity_sha256": self.identity_sha256,
            "namespace_count": len(self.retrievers),
            "namespace_open_count": self._namespace_open_count,
            "namespace_close_count": self._namespace_close_count,
            "search_call_count": self._search_call_count,
            "maximum_simultaneous_namespace_indexes": self._maximum_active_indexes,
            "bge_model_load_count": 1,
            "bge_released": self._closed,
            "physical_provider_calls": 0,
        }
        return MappingProxyType({**body, "receipt_sha256": canonical_sha256(body)})


def build_confirmation_query_retriever_session(
    *,
    context: Any,
    policy_manifest_sha256: str,
    qwen_prefix_model_dir: str | Path,
    qwen_choice_model_dir: str | Path,
    runtime_builder: Any = None,
) -> ConfirmationQueryRetrieverSession:
    """Construct an owned fresh-BGE session after staged Qwen has closed."""

    builder = runtime_builder or build_confirmation_production_runtime
    runtime = builder(
        policy_manifest_sha256=policy_manifest_sha256,
        qwen_prefix_model_dir=qwen_prefix_model_dir,
        qwen_choice_model_dir=qwen_choice_model_dir,
    )
    _require(
        type(runtime) is ConfirmationProductionRuntime,
        "query retriever runtime factory returned another runtime type",
    )
    try:
        return ConfirmationQueryRetrieverSession(runtime=runtime, context=context)
    except BaseException:
        close = getattr(runtime.binding.embedder, "close", None)
        if callable(close):
            close()
        raise


def confirmation_retrieval_config(*, device: str = "cuda") -> EvalConfig:
    """Return and authenticate the exact frozen S0--S3 retrieval config."""

    _require(device == "cuda", "confirmation runtime device must remain cuda")
    retrieval = RetrievalConfig(
        mode="causal_graph",
        k=10,
        ef_search=50,
        alpha=0.65,
        candidates=100,
        neighbor_radius=5,
        neighbor_slots=24,
        neighbor_replacement_slots=0,
        neighbor_direction="next",
        source_slots=48,
        source_activation_k=65,
        source_candidate_pool=750,
        source_local_search=True,
        source_tfisf_activation=True,
        source_tfisf_slots=8,
        source_hsc_activation=True,
        source_hsc_slots=8,
        source_hsc_hops=2,
        source_hsc_chunk_slots=4,
        source_partition_routing=True,
        source_partition_slots=4,
        source_partition_separator="::",
        role_aware_retrieval=True,
        role_user_weight=1.25,
        role_assistant_weight=0.75,
        role_system_weight=0.5,
        consolidation_chunk_slots=24,
        consolidation_hops=2,
        consolidation_candidates=128,
        consolidation_diffusion_width=32,
        consolidation_min_count=2,
        consolidation_expansion_tokens=2250,
        consolidation_training_expansion_tokens=1600,
        consolidation_budget_aware_packing=True,
        consolidation_training_k=10,
        consolidation_max_event_nodes=9,
        consolidation_new_event_nodes=5,
        consolidation_max_training_prompt_tokens=128,
        consolidation_query_aware_sentence_packing=True,
        consolidation_max_sentences_per_expansion=2,
        consolidation_information_gain_packing=True,
        consolidation_min_information_gain_per_token=0.008,
        consolidation_source_metadata_packing=True,
        coverage_selection=True,
        coverage_selector_backend="qwen_prefix_choice",
        coverage_selector_model="Qwen3-8B+Qwen3-0.6B",
        coverage_selector_dtype="float16",
        coverage_selector_candidate_pool=64,
        coverage_selector_candidate_tokens=64,
        coverage_selector_query_tokens=96,
        coverage_selector_max_workspace_tokens=8192,
        coverage_selector_max_new_tokens=4096,
        coverage_selector_null_threshold=0.85,
        coverage_selector_uncertainty_entropy=0.95,
        coverage_selector_prefix_layers=2,
        coverage_selector_attention_layer=1,
        coverage_selector_merge_similarity=0.985,
        coverage_selector_same_source_merge_similarity=0.9,
        coverage_selector_strict=True,
        allow_selected_scope_fixed_k_closure=True,
        coverage_selector_prefix_model_id="Qwen/Qwen3-8B",
        coverage_selector_prefix_revision=(
            "b968826d9c46dd6066d109eabc6255188de91218"
        ),
        coverage_selector_prefix_checkpoint_sha256=(
            "76273516aa6924b12344d5e83daa485b66459b663c745cb3b9ef51cc17c7440d"
        ),
        coverage_selector_prefix_device="cuda",
        coverage_selector_prefix_dtype="float16",
        coverage_selector_choice_model_id="Qwen/Qwen3-0.6B",
        coverage_selector_choice_revision=(
            "c1899de289a04d12100db370d81485cdf75e47ca"
        ),
        coverage_selector_choice_checkpoint_sha256=(
            "a940db06d5d9a3b298412376966b492f09ad7f088495fb75c05aa45db943d86e"
        ),
        coverage_selector_choice_device="cuda",
        coverage_selector_choice_dtype="float16",
        coverage_selector_choice_batch_size=8,
        coverage_selector_choice_max_candidates=128,
        coverage_selector_choice_query_tokens=192,
        coverage_selector_choice_candidate_tokens=128,
        coverage_selector_choice_max_prompt_tokens=512,
        coverage_selector_choice_max_workspace_tokens=8192,
    )
    config = EvalConfig(
        chunker=ChunkerConfig(min_tokens=120, max_tokens=250),
        retrieval=retrieval,
        embedding_device=device,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
    _require(
        identity_sha256(retrieval.model_dump(mode="json"))
        == FROZEN_RETRIEVAL_POLICY_SHA256
        and identity_sha256(config.model_dump(mode="json"))
        == FROZEN_FULL_CONFIG_SHA256,
        "resolved confirmation retrieval config drifted",
    )
    return config


def confirmation_source_config(config: EvalConfig) -> EvalConfig:
    """Derive and authenticate the direct, query-free source-ingest config."""

    # This is the complete frozen source-acquisition projection.  Keeping it
    # here avoids importing the historical 1M source CLI, whose dependency
    # closure owns the raw benchmark loader.
    source = config.model_copy(
        update={"retrieval": RetrievalConfig(mode="dense", k=10)}
    )
    _require(
        identity_sha256(source.model_dump(mode="json"))
        == FROZEN_SOURCE_CONFIG_SHA256
        and identity_sha256(source.retrieval.model_dump(mode="json"))
        == FROZEN_SOURCE_RETRIEVAL_POLICY_SHA256,
        "resolved confirmation source config drifted",
    )
    return source


def confirmation_compilation_policy() -> DiffuseCompilationPolicy:
    return DiffuseCompilationPolicy(boundary_mode="fixed_interval")


def confirmation_episode_policy(artifact_id: str) -> EpisodeRetrievalPolicy:
    _require(type(artifact_id) is str and bool(artifact_id), "episode artifact ID is empty")
    return EpisodeRetrievalPolicy(artifact_id=artifact_id, **_DIRECT_EPISODE_CONTROLS)


def confirmation_closure_policy() -> ClosurePolicy:
    return ClosurePolicy(**_CLOSURE_CONTROLS)


def _runtime_policy_binding(
    *,
    policy_manifest_sha256: str,
    binding: DiffuseLongMemEvalExecutionBinding,
) -> Mapping[str, Any]:
    body = {
        "format": RUNTIME_POLICY_FORMAT,
        "model_residency_mode": STAGED_PRODUCTION_MODE,
        "policy_manifest_sha256": policy_manifest_sha256,
        "retrieval_policy_sha256": FROZEN_RETRIEVAL_POLICY_SHA256,
        "runtime_binding_sha256": binding.binding_sha256,
        "budgets": {
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "responder_output_token_reserve": RESPONDER_OUTPUT_TOKEN_RESERVE,
            "source_router_max_sources": SOURCE_ROUTER_MAX_SOURCES,
            "source_router_rrf_constant": SOURCE_ROUTER_RRF_CONSTANT,
        },
    }
    return MappingProxyType({**body, "receipt_sha256": canonical_sha256(body)})


@dataclass(frozen=True, slots=True)
class ConfirmationProductionRuntime:
    """Owned initial-BGE factory set for ingest and staged cumulative recall."""

    policy_manifest_sha256: str
    config: EvalConfig
    source_config: EvalConfig
    binding: DiffuseLongMemEvalExecutionBinding
    source_treatment_contract: Mapping[str, Any]
    runtime_policy_binding: Mapping[str, Any]
    base_backend: ProductionBaseStoreBackend
    preparation_backend: ProductionStagedPreparationBackend
    qwen_factory: ProductionStagedQwenRuntimeFactory
    retrieval_factory: ProductionStagedRetrievalBackendFactory
    identity_sha256: str

    def __post_init__(self) -> None:
        _require(
            self.base_backend.identity_sha256
            == self.preparation_backend._source_backend_identity,  # noqa: SLF001
            "preparation backend escaped the base source",
        )
        _require(
            self.runtime_policy_binding.get("policy_manifest_sha256")
            == self.policy_manifest_sha256,
            "runtime policy binds another freeze",
        )


def build_confirmation_production_runtime(
    *,
    policy_manifest_sha256: str,
    qwen_prefix_model_dir: str | Path,
    qwen_choice_model_dir: str | Path,
) -> ConfirmationProductionRuntime:
    """Construct initial BGE-owned adapters; Qwen remains barrier-gated."""

    _require(
        type(policy_manifest_sha256) is str
        and len(policy_manifest_sha256) == 64
        and set(policy_manifest_sha256) <= set("0123456789abcdef"),
        "policy manifest SHA-256 is invalid",
    )
    config = confirmation_retrieval_config()
    source_config = confirmation_source_config(config)
    runtime = DiffuseLongMemEvalRuntimeConfig(
        qwen_model_dir=Path(qwen_prefix_model_dir),
        residency_mode=STAGED_PRODUCTION_MODE,
        embedding_batch_size=32,
        qwen_device="cuda",
        qwen_dtype="float16",
        qwen_max_candidates=8,
        qwen_max_workspace_tokens=2048,
        resident_min_free_mib=3072,
        source_router_max_sources=SOURCE_ROUTER_MAX_SOURCES,
        source_router_rrf_constant=SOURCE_ROUTER_RRF_CONSTANT,
    )
    # The owned binding supplies BGE and the condenser factory.  Its config
    # must describe the direct source-acquisition route; packed causal_graph
    # is deliberately constructed later by the cumulative backend.
    binding = build_diffuse_longmemeval_execution_binding(
        config=source_config,
        runtime=runtime,
    )
    _require(binding.runtime_binding_certified, "local runtime factories are uncertified")
    source_contract = build_production_source_treatment_contract(
        source_config,
        binding.embedding_identity,
    )
    base = ProductionBaseStoreBackend(
        config=source_config,
        embedder=binding.embedder,
        embedding_identity=binding.embedding_identity,
        condenser_factory=binding.new_condenser,
    )
    compilation = confirmation_compilation_policy()
    policy_binding = _runtime_policy_binding(
        policy_manifest_sha256=policy_manifest_sha256,
        binding=binding,
    )
    preparation = ProductionStagedPreparationBackend(
        policy_freeze_sha256=policy_manifest_sha256,
        source_backend_identity_sha256=base.identity_sha256,
        source_treatment_contract_sha256=source_contract["contract_sha256"],
        config=config,
        embedder=binding.embedder,
        embedding_identity=binding.embedding_identity,
        compilation_policy=compilation,
    )
    qwen = ProductionStagedQwenRuntimeFactory(
        config=config,
        qwen_prefix_model_dir=qwen_prefix_model_dir,
        qwen_choice_model_dir=qwen_choice_model_dir,
    )
    retrieval = ProductionStagedRetrievalBackendFactory(
        policy_freeze_sha256=policy_manifest_sha256,
        runtime_policy_binding=policy_binding,
        source_backend_identity_sha256=base.identity_sha256,
        source_treatment_contract_sha256=source_contract["contract_sha256"],
        config=config,
        compilation_policy=compilation,
        episode_policy_factory=confirmation_episode_policy,
        representative_policy_factory=binding.representative_policy_factory,
        closure_policy=confirmation_closure_policy(),
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        responder_output_token_reserve=RESPONDER_OUTPUT_TOKEN_RESERVE,
        source_router_max_sources=SOURCE_ROUTER_MAX_SOURCES,
        source_router_rrf_constant=SOURCE_ROUTER_RRF_CONSTANT,
        embedding_identity=binding.embedding_identity,
    )
    identity_body = {
        "format": FORMAT,
        "policy_manifest_sha256": policy_manifest_sha256,
        "retrieval_policy_sha256": FROZEN_RETRIEVAL_POLICY_SHA256,
        "source_config_sha256": FROZEN_SOURCE_CONFIG_SHA256,
        "source_treatment_contract_sha256": source_contract["contract_sha256"],
        "runtime_binding_sha256": binding.binding_sha256,
        "runtime_policy_binding_sha256": policy_binding["receipt_sha256"],
        "base_backend_identity_sha256": base.identity_sha256,
        "preparation_backend_identity_sha256": preparation.identity_sha256,
        "qwen_factory_identity_sha256": qwen.identity_sha256,
        "retrieval_factory_identity_sha256": retrieval.identity_sha256,
    }
    return ConfirmationProductionRuntime(
        policy_manifest_sha256=policy_manifest_sha256,
        config=config,
        source_config=source_config,
        binding=binding,
        source_treatment_contract=source_contract,
        runtime_policy_binding=policy_binding,
        base_backend=base,
        preparation_backend=preparation,
        qwen_factory=qwen,
        retrieval_factory=retrieval,
        identity_sha256=canonical_sha256(identity_body),
    )


__all__ = [
    "FORMAT",
    "FROZEN_FULL_CONFIG_SHA256",
    "FROZEN_RETRIEVAL_POLICY_SHA256",
    "FROZEN_SOURCE_CONFIG_SHA256",
    "FROZEN_SOURCE_RETRIEVAL_POLICY_SHA256",
    "MAX_CONTEXT_TOKENS",
    "MAX_PROMPT_TOKENS",
    "RESPONDER_OUTPUT_TOKEN_RESERVE",
    "SOURCE_ROUTER_MAX_SOURCES",
    "SOURCE_ROUTER_RRF_CONSTANT",
    "ConfirmationProductionRuntime",
    "ConfirmationProductionRuntimeError",
    "ConfirmationQueryRetrieverSession",
    "build_confirmation_production_runtime",
    "build_confirmation_query_retriever_session",
    "confirmation_closure_policy",
    "confirmation_compilation_policy",
    "confirmation_episode_policy",
    "confirmation_retrieval_config",
    "confirmation_source_config",
]
