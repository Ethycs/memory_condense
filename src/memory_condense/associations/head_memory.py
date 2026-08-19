"""Compatibility facade for experimental per-head memory.

The concrete models, indexes, graph, bounded workspace, Qwen workflows, and
CLI now live in focused sibling modules. Existing imports and ``python -m``
execution remain stable through this facade.
"""

from __future__ import annotations

from memory_condense.associations.associative_composition import (
    compose_associative_candidates,
)
from memory_condense.associations.cav_memory import CAVBank, CAVLinkIndex
from memory_condense.associations.head_association_graph import (
    HeadAssociationGraph,
    _rank_association_walk,
)
from memory_condense.associations.head_kv_store import HeadKVStore
from memory_condense.associations.head_memory_cli import (
    build_parser,
    main,
    run_smoke_benchmark,
)
from memory_condense.associations.head_memory_models import (
    AssociativeComposition,
    AssociativeMemoryCandidate,
    CAVNeighbor,
    HeadAddress,
    HeadAssociationEdge,
    HeadMemoryItem,
    LiveMemoryHit,
    LiveMemoryResult,
    MemoryLinkHit,
    MemoryLinkResult,
    NestedMemoryInspection,
)
from memory_condense.associations.qwen_live_memory import QwenLiveHeadMemory
from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder


__all__ = [
    "AssociativeComposition",
    "AssociativeMemoryCandidate",
    "CAVBank",
    "CAVLinkIndex",
    "CAVNeighbor",
    "HeadAddress",
    "HeadAssociationEdge",
    "HeadAssociationGraph",
    "HeadKVStore",
    "HeadMemoryItem",
    "LiveMemoryHit",
    "LiveMemoryResult",
    "MemoryLinkHit",
    "MemoryLinkResult",
    "NestedMemoryInspection",
    "Qwen3PrefixEncoder",
    "QwenLiveHeadMemory",
    "QwenMemoryLinker",
    "build_parser",
    "compose_associative_candidates",
    "main",
    "run_smoke_benchmark",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
