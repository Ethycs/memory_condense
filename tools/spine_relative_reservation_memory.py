"""Add bounded relative-day user evidence to the frozen as-of memory path."""
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.relative_spine_reservation import RelativeSpineReservation
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools.spine_as_of_memory import ResidentMemory as AsOfMemory
from tools.spine_facet_memory import supplemental_plan


class ResidentMemory(AsOfMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.relative_reservation = RelativeSpineReservation(self.as_of_router, self.source_spine)

    def relative_plan(self, query, dated_question):
        identity = summary_embedding_identity(self.encoder)
        view = ordered_content_query(query)
        vectors = self.encoder.embed_queries([query, view]) if view != query else [self.encoder.embed_query(query)]
        if summary_embedding_identity(self.encoder) != identity:
            raise ValueError("live query encoder changed")
        plans, route_audit = self.as_of_router.route_vectors(query, dated_question, vectors[0],
            embedding_identity=identity, content_vector=vectors[1] if len(vectors) == 2 else None)
        prior, expansion = self.as_of_expansion.expand(query, dated_question, plans,
            supplemental_plan(plans["users"], plans["facets"]))
        final, reservation = self.relative_reservation.reserve(query, dated_question, prior,
            self.as_of_router.users._spine.score_all(vectors[0]))
        return final, {"routing": route_audit, "expansion": expansion, "relative_reservation": reservation}

    def retrieve(self, query, arm, dated_question):
        if arm != "source_spine_relative_reservation":
            return super().retrieve(query, arm, dated_question)
        plan, _ = self.relative_plan(query, dated_question)
        return hydrate_section_plan(plan, load_turn=self.turns.get, max_context_tokens=3072, max_raw_spans=128)
