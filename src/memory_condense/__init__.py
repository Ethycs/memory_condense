"""Long-term memory condensation for LLM conversations.

The package root is the stable object-level API. Implementations live in
responsibility-focused subpackages and are imported lazily so importing
``memory_condense`` does not initialize retrieval or executable modules.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any as _Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "MemoryCondenser": (
        "memory_condense.application.condenser",
        "MemoryCondenser",
    ),
    "ConsolidationNodeKind": (
        "memory_condense.associations.consolidation",
        "ConsolidationNodeKind",
    ),
    "ConsolidationNode": (
        "memory_condense.associations.consolidation",
        "ConsolidationNode",
    ),
    "ConsolidationUpdate": (
        "memory_condense.associations.consolidation",
        "ConsolidationUpdate",
    ),
    "ConsolidationNeighbor": (
        "memory_condense.associations.consolidation",
        "ConsolidationNeighbor",
    ),
    "LiveConsolidationStore": (
        "memory_condense.associations.consolidation",
        "LiveConsolidationStore",
    ),
    "context_activations": (
        "memory_condense.associations.consolidation",
        "context_activations",
    ),
    "expand_context_associations": (
        "memory_condense.associations.consolidation",
        "expand_context_associations",
    ),
    "inspect_qwen_context_hyperplane": (
        "memory_condense.associations.consolidation",
        "inspect_qwen_context_hyperplane",
    ),
    "qwen_head_activations": (
        "memory_condense.associations.consolidation",
        "qwen_head_activations",
    ),
    "AssociationArtifact": (
        "memory_condense.associations.association_store",
        "AssociationArtifact",
    ),
    "AssociationStore": (
        "memory_condense.associations.association_store",
        "AssociationStore",
    ),
    "HebbianUpdate": (
        "memory_condense.associations.association_store",
        "HebbianUpdate",
    ),
    "StoredHebbianNeighbor": (
        "memory_condense.associations.association_store",
        "StoredHebbianNeighbor",
    ),
    "DiffusedHeatNode": (
        "memory_condense.associations.heat_diffusion",
        "DiffusedHeatNode",
    ),
    "HeatDiffusion": (
        "memory_condense.associations.heat_diffusion",
        "HeatDiffusion",
    ),
    "diffuse_association_heat": (
        "memory_condense.associations.heat_diffusion",
        "diffuse_association_heat",
    ),
    "expand_heat_diffusion_results": (
        "memory_condense.associations.heat_diffusion",
        "expand_heat_diffusion_results",
    ),
    "expand_hebbian_results": (
        "memory_condense.associations.hebbian_retrieval",
        "expand_hebbian_results",
    ),
    "retrieval_concept_activations": (
        "memory_condense.associations.hebbian_retrieval",
        "retrieval_concept_activations",
    ),
    "CausalTransitionPolicy": (
        "memory_condense.associations.transition_policy",
        "CausalTransitionPolicy",
    ),
    "TransitionCandidate": (
        "memory_condense.associations.transition_policy",
        "TransitionCandidate",
    ),
    "TransitionDecision": (
        "memory_condense.associations.transition_policy",
        "TransitionDecision",
    ),
    "TransitionFeedback": (
        "memory_condense.associations.transition_policy",
        "TransitionFeedback",
    ),
    "ScoredTransition": (
        "memory_condense.associations.transition_policy",
        "ScoredTransition",
    ),
    "Turn": ("memory_condense.domain.schemas", "Turn"),
    "Chunk": ("memory_condense.domain.schemas", "Chunk"),
    "RetrievalResult": ("memory_condense.domain.schemas", "RetrievalResult"),
    "MemoryItem": ("memory_condense.domain.schemas", "MemoryItem"),
    "MemoryType": ("memory_condense.domain.schemas", "MemoryType"),
    "MemoryStatus": ("memory_condense.domain.schemas", "MemoryStatus"),
    "PinState": ("memory_condense.domain.schemas", "PinState"),
    "Heat": ("memory_condense.domain.schemas", "Heat"),
    "Provenance": ("memory_condense.domain.schemas", "Provenance"),
    "CreateOp": ("memory_condense.domain.schemas", "CreateOp"),
    "MemoryOps": ("memory_condense.domain.schemas", "MemoryOps"),
    "MemoryResult": ("memory_condense.domain.schemas", "MemoryResult"),
    "effective_energy": ("memory_condense.domain.decay", "effective_energy"),
    "heat_for": ("memory_condense.domain.decay", "heat_for"),
    "item_energy": ("memory_condense.domain.decay", "item_energy"),
    "item_heat": ("memory_condense.domain.decay", "item_heat"),
    "rank_score": ("memory_condense.domain.ranking", "rank_score"),
    "RankWeights": ("memory_condense.domain.ranking", "RankWeights"),
    "DEFAULT_WEIGHTS": ("memory_condense.domain.ranking", "DEFAULT_WEIGHTS"),
    "ContextPacker": (
        "memory_condense.search.packing.context_packer",
        "ContextPacker",
    ),
    "ContextBudget": (
        "memory_condense.search.packing.context_packer",
        "ContextBudget",
    ),
    "PackedContext": ("memory_condense.domain.schemas", "PackedContext"),
    "SetOperator": (
        "memory_condense.search.selectors.coverage_selector",
        "SetOperator",
    ),
    "SetProgram": (
        "memory_condense.search.selectors.coverage_selector",
        "SetProgram",
    ),
    "CandidateAssignment": (
        "memory_condense.search.selectors.coverage_selector",
        "CandidateAssignment",
    ),
    "CoverageSelectionReport": (
        "memory_condense.search.selectors.coverage_selector",
        "CoverageSelectionReport",
    ),
    "QueryConditionedCoverageSelector": (
        "memory_condense.search.selectors.coverage_selector",
        "QueryConditionedCoverageSelector",
    ),
    "QwenPrefixCoverageSelector": (
        "memory_condense.search.selectors.coverage_selector",
        "QwenPrefixCoverageSelector",
    ),
    "compile_set_program": (
        "memory_condense.search.selectors.coverage_selector",
        "compile_set_program",
    ),
    "load_conversation": ("memory_condense.ingest.loader", "load_conversation"),
    "load_directory": ("memory_condense.ingest.loader", "load_directory"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> _Any:
    """Load and cache a supported facade object on first access."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(_import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy facade objects to interactive tooling."""
    return sorted(set(globals()) | set(__all__))
