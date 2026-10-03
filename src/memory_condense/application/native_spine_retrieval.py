"""One live query embedding followed by exact native evidence hydration."""
from dataclasses import dataclass

from memory_condense.application.section_retrieval import SectionRetrievalResult, hydrate_section_plan
from memory_condense.search.native_spine_routing import NativeSpineRoute, NativeSpineRouter
from memory_condense.search.summary_semantic_index import summary_embedding_identity, compatible_summary_embedding
from memory_condense.search.summary_time_prior_v2 import question_day


@dataclass(frozen=True)
class NativeSpineRetrieval:
    routing: NativeSpineRoute
    hydration: SectionRetrievalResult
    live_query_embedding: bool = True


class ResidentNativeSpineMemory:
    """The caller owns the resident encoder and the cold-admitted namespace."""

    def __init__(self, atomic_semantic, hierarchy, *, encoder, load_turn):
        self.router = NativeSpineRouter(atomic_semantic, hierarchy)
        if not compatible_summary_embedding(encoder, atomic_semantic.embedding_identity):
            raise ValueError("native query encoder differs from the stored summary vectors")
        self.encoder, self.load_turn = encoder, load_turn
        self._query_embedding_identity = summary_embedding_identity(encoder)

    def retrieve(self, query, dated_question, *, augment=True, max_context_tokens=3072,
                 max_raw_spans=128, **routing_limits):
        question_day(query, dated_question)
        identity = summary_embedding_identity(self.encoder)
        if identity != self._query_embedding_identity:
            raise ValueError("native query encoder changed")
        vector = self.encoder.embed_query(query)
        if summary_embedding_identity(self.encoder) != identity:
            raise ValueError("native query encoder changed during live encoding")
        # Admission above authenticates the opt-in device transition. The
        # router's namespace retains the identity of its stored vectors.
        routing = self.router.route_vector(query, dated_question, vector,
                                            embedding_identity=self.router.semantic.embedding_identity,
                                            **routing_limits)
        hydration = hydrate_section_plan(routing.expanded if augment else routing.baseline,
            load_turn=self.load_turn, max_context_tokens=max_context_tokens, max_raw_spans=max_raw_spans)
        return NativeSpineRetrieval(routing, hydration)
