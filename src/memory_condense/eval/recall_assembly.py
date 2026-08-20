"""Stateful context assembly and forward heat-survival measurement."""

from __future__ import annotations

from memory_condense.domain import decay
from memory_condense.eval.answer_value_coverage import contains_answer
from memory_condense.eval.schemas import EvalConfig
from memory_condense.eval.search_kwargs import (
    graph_search_kwargs,
    source_search_kwargs,
)
from memory_condense.search.packing.context_packer import is_source_metadata_text

def _assemble(
    mc, question: str, config: EvalConfig
) -> tuple[list[str], list[str], list[str | None], list[bool]]:
    """Return header, body, and the body items' durable source IDs.

    ``reheat`` is off throughout: this is a measurement, and an item must not
    become hotter merely because a measurement looked at it.
    """
    mc.last_raw_graph_source_ids = []
    if config.retrieval.mode in {
        "memory",
        "causal_consolidation",
        "causal_graph",
    }:
        causal = config.retrieval.mode in {
            "causal_consolidation",
            "causal_graph",
        }
        graph_results = (
            mc.search_hybrid_graph(
                question,
                **graph_search_kwargs(config.retrieval, routing=True),
            )
            if config.retrieval.mode == "causal_graph"
            else None
        )
        packed = mc.build_context(
            question,
            recent_turns=0,
            k_memories=0 if causal else config.retrieval.k_memories,
            k_expansions=(0 if graph_results is not None else config.retrieval.k),
            # Hybrid is the production facade's default expansion retriever
            # and B0's strongest in-regime arm.  Memory mode should not
            # silently override it back to dense.
            hybrid=True,
            reheat_memories=False,
            use_consolidation=causal,
            learn_consolidation=False,
            consolidation_memory_slots=0 if causal else 1,
            consolidation_chunk_slots=(
                config.retrieval.consolidation_chunk_slots if causal else 1
            ),
            consolidation_min_count=config.retrieval.consolidation_min_count,
            consolidation_hops=config.retrieval.consolidation_hops,
            consolidation_candidates=config.retrieval.consolidation_candidates,
            consolidation_diffusion_width=(
                config.retrieval.consolidation_diffusion_width
            ),
            expansion_results=graph_results,
        )
        if graph_results is not None:
            mc.last_raw_graph_source_ids = list(
                dict.fromkeys(
                    result.durable_source_id
                    for result in graph_results
                    if not is_source_metadata_text(result.chunk.text)
                )
            )
        header = [packed.memory_header] if packed.memory_header else []
        sources: list[str | None] = []
        if causal:
            for chunk_id in packed.expansion_chunk_ids:
                hydrated = mc.retriever.hydrate_chunk(
                    chunk_id,
                    score=0.0,
                    route="source_diagnostic",
                )
                sources.append(
                    getattr(getattr(hydrated, "turn", None), "source_id", None)
                )
        direct = set(packed.direct_expansion_chunk_ids)
        return (
            header,
            list(packed.expansions),
            sources,
            [chunk_id not in direct for chunk_id in packed.expansion_chunk_ids],
        )

    if config.retrieval.mode == "span":
        results = mc.search_spans(
            question,
            levels=config.retrieval.span_levels,
            k_per_level=config.retrieval.k_per_level,
        )
    elif config.retrieval.mode == "source":
        results = mc.search_sources(
            question,
            k_sources=config.retrieval.k_sources,
        )
    elif config.retrieval.mode == "anchored_source":
        results = mc.search_anchored_sources(
            question,
            k=config.retrieval.k,
            ef_search=config.retrieval.ef_search,
            candidates=config.retrieval.candidates,
            alpha=config.retrieval.alpha,
        )
    elif config.retrieval.mode == "hybrid_source":
        results = mc.search_hybrid_sources(
            question,
            **source_search_kwargs(config.retrieval),
        )
    elif config.retrieval.mode == "hybrid_graph":
        results = mc.search_hybrid_graph(
            question,
            **graph_search_kwargs(config.retrieval),
        )
    elif config.retrieval.mode == "hybrid_neighbor":
        results = mc.search_hybrid_neighbors(
            question,
            k=config.retrieval.k,
            radius=config.retrieval.neighbor_radius,
            max_neighbors=config.retrieval.neighbor_slots,
            replacement_slots=config.retrieval.neighbor_replacement_slots,
            ef_search=config.retrieval.ef_search,
            candidates=config.retrieval.candidates,
            alpha=config.retrieval.alpha,
        )
    elif config.retrieval.effective_hybrid:
        results = mc.search_hybrid(
            question,
            k=config.retrieval.k,
            ef_search=config.retrieval.ef_search,
            candidates=config.retrieval.candidates,
            alpha=config.retrieval.alpha,
        )
    else:
        results = mc.search(
            question, k=config.retrieval.k, ef_search=config.retrieval.ef_search
        )
    return (
        [],
        [r.chunk.text for r in results],
        [getattr(getattr(r, "turn", None), "source_id", None) for r in results],
        [False] * len(results),
    )


def _survival(mc, gold: str, horizons_turns) -> dict[int, bool]:
    """Would the answer still sit in a non-COLD memory item N turns from now?

    Projects :func:`decay.effective_energy` forward over the stored items from
    the transcript's current position. Horizon 0 is not a projection at all —
    it is the store as it stands, which is now a real reading because turns
    have actually elapsed during the run.

    An empty memory store (the chunk arms, where nothing is extracted) yields
    ``False`` at every horizon — correctly: there is no memory item holding
    the answer.
    """
    items = mc.memory.list_items()
    now_turn = mc.transcript.current_turn()
    out: dict[int, bool] = {}
    for turns in horizons_turns:
        alive = [
            f"{i.content} {i.details or ''}"
            for i in items
            if decay.item_heat(i, now_turn=now_turn + turns) is not decay.Heat.COLD
        ]
        out[turns] = contains_answer(alive, gold)
    return out
