"""memory_condense — Long-term memory condensation for LLM conversations."""

from memory_condense.association_store import (
    AssociationArtifact,
    AssociationStore,
    HebbianUpdate,
    StoredHebbianNeighbor,
)
from memory_condense.condenser import MemoryCondenser
from memory_condense.context_packer import ContextBudget, ContextPacker
from memory_condense.decay import effective_energy, heat_for, item_energy, item_heat
from memory_condense.heat_diffusion import (
    DiffusedHeatNode,
    HeatDiffusion,
    diffuse_association_heat,
    expand_heat_diffusion_results,
)
from memory_condense.hebbian_retrieval import (
    expand_hebbian_results,
    retrieval_concept_activations,
)
from memory_condense.loader import load_conversation, load_directory
from memory_condense.ranking import DEFAULT_WEIGHTS, RankWeights, rank_score
from memory_condense.schemas import (
    Chunk,
    CreateOp,
    Heat,
    MemoryItem,
    MemoryOps,
    MemoryResult,
    MemoryStatus,
    MemoryType,
    PackedContext,
    PinState,
    Provenance,
    RetrievalResult,
    Turn,
)
from memory_condense.transition_policy import (
    CausalTransitionPolicy,
    ScoredTransition,
    TransitionCandidate,
    TransitionDecision,
    TransitionFeedback,
)

__all__ = [
    # facade
    "MemoryCondenser",
    "AssociationArtifact",
    "AssociationStore",
    "HebbianUpdate",
    "StoredHebbianNeighbor",
    "DiffusedHeatNode",
    "HeatDiffusion",
    "diffuse_association_heat",
    "expand_heat_diffusion_results",
    "expand_hebbian_results",
    "retrieval_concept_activations",
    "CausalTransitionPolicy",
    "TransitionCandidate",
    "TransitionDecision",
    "TransitionFeedback",
    "ScoredTransition",
    # transcript / chunk layer
    "Turn",
    "Chunk",
    "RetrievalResult",
    # memory layer
    "MemoryItem",
    "MemoryType",
    "MemoryStatus",
    "PinState",
    "Heat",
    "Provenance",
    "CreateOp",
    "MemoryOps",
    "MemoryResult",
    # decay + ranking
    "effective_energy",
    "heat_for",
    "item_energy",
    "item_heat",
    "rank_score",
    "RankWeights",
    "DEFAULT_WEIGHTS",
    # context packing
    "ContextPacker",
    "ContextBudget",
    "PackedContext",
    # loading
    "load_conversation",
    "load_directory",
]
