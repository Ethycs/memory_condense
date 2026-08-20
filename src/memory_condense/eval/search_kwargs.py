"""Shared search-call kwarg builders for the benchmark and recall arms.

Recall is the cheap predictor of the paid benchmark comparison, so both arms
must issue byte-identical retrieval calls.  Building the kwargs here means a
flag added for one arm necessarily reaches the other; a flag added to an
inlined copy would silently decouple them and no test could catch it.
"""

from __future__ import annotations

from memory_condense.eval.schemas import RetrievalConfig


def _routing_search_kwargs(retrieval: RetrievalConfig) -> dict:
    """Facet/role/diversity/partition kwargs shared by the source arm and the
    causal-graph arm's full ``search_hybrid_graph`` call."""
    return {
        "query_facet_retrieval": retrieval.query_facet_retrieval,
        "query_facet_slots": retrieval.query_facet_slots,
        "query_facet_max": retrieval.query_facet_max,
        "role_aware_retrieval": retrieval.role_aware_retrieval,
        "role_user_weight": retrieval.role_user_weight,
        "role_assistant_weight": retrieval.role_assistant_weight,
        "role_system_weight": retrieval.role_system_weight,
        "multi_fact_source_diversity": retrieval.multi_fact_source_diversity,
        "source_partition_routing": retrieval.source_partition_routing,
        "source_partition_slots": retrieval.source_partition_slots,
        "source_partition_separator": retrieval.source_partition_separator,
    }


def graph_search_kwargs(
    retrieval: RetrievalConfig, *, routing: bool = False
) -> dict:
    """Kwargs for ``search_hybrid_graph`` derived from ``retrieval``.

    ``routing=True`` adds the facet/role/partition kwargs that only the
    causal-graph arm passes; the plain ``hybrid_graph`` arm omits them.
    """
    kwargs = {
        "k": retrieval.k,
        "neighbor_radius": retrieval.neighbor_radius,
        "neighbor_slots": retrieval.neighbor_slots,
        "neighbor_direction": retrieval.neighbor_direction,
        "source_slots": retrieval.source_slots,
        "source_candidate_pool": retrieval.source_candidate_pool,
        "source_activation_k": retrieval.source_activation_k,
        "source_tfisf_activation": retrieval.source_tfisf_activation,
        "source_tfisf_slots": retrieval.source_tfisf_slots,
        "source_hsc_activation": retrieval.source_hsc_activation,
        "source_hsc_slots": retrieval.source_hsc_slots,
        "source_hsc_hops": retrieval.source_hsc_hops,
        "source_hsc_chunk_slots": retrieval.source_hsc_chunk_slots,
        "source_local_search": retrieval.source_local_search,
        "use_source_reranker": retrieval.qwen_rerank,
        "use_attention_feedback": retrieval.qwen_feedback,
        "feedback_slots": retrieval.qwen_feedback_slots,
        "feedback_seed_slots": retrieval.qwen_feedback_seed_slots,
        "feedback_evidence_tokens": retrieval.qwen_feedback_evidence_tokens,
        "feedback_query_tokens": retrieval.qwen_feedback_query_tokens,
        "ef_search": retrieval.ef_search,
        "candidates": retrieval.candidates,
        "alpha": retrieval.alpha,
    }
    if routing:
        kwargs.update(_routing_search_kwargs(retrieval))
    return kwargs


def source_search_kwargs(retrieval: RetrievalConfig) -> dict:
    """Kwargs for ``search_hybrid_sources`` derived from ``retrieval``."""
    return {
        "k": retrieval.k,
        "source_slots": retrieval.source_slots,
        "source_candidate_pool": retrieval.source_candidate_pool,
        "source_activation_k": retrieval.source_activation_k,
        **_routing_search_kwargs(retrieval),
        "source_local_search": retrieval.source_local_search,
        "use_source_reranker": retrieval.qwen_rerank,
        "ef_search": retrieval.ef_search,
        "candidates": retrieval.candidates,
        "alpha": retrieval.alpha,
    }
