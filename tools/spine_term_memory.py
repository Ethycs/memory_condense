"""Live source-scoped term coverage over the unchanged summary hierarchy."""
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.spine_term_coverage_v2 import ScopedSpineTermCoverage
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools.spine_facet_memory import ResidentMemory as FacetMemory, supplemental_plan


class ResidentMemory(FacetMemory):
    def __init__(self, *args):
        super().__init__(*args)
        try:
            self.term_coverage = ScopedSpineTermCoverage(self.semantic.hierarchy, self.source_spine)
        except Exception:
            self.encoder.close()
            raise

    def retrieve(self, query, arm, dated_question):
        if arm != "source_spine_term_coverage":
            return super().retrieve(query, arm, dated_question)
        identity = summary_embedding_identity(self.encoder)
        view = ordered_content_query(query)
        vectors = self.encoder.embed_queries([query, view]) if view != query else [self.encoder.embed_query(query)]
        if summary_embedding_identity(self.encoder) != identity:
            raise ValueError("live query encoder changed")
        selected, _ = self.router.route_vectors(query, dated_question, vectors[0], embedding_identity=identity,
            content_vector=vectors[1] if len(vectors) == 2 else None)
        wider = self.router.user_addresses.route_vector(query, vectors[0], embedding_identity=identity,
            user_weight=1, max_sections=8, lexical_reserve=0)
        facets, _ = self.facets.route_vector(query, vectors[0], embedding_identity=identity, max_sections=8)
        prior, _ = self.supplement.expand(selected, supplemental_plan(wider, facets))
        plan, _ = self.term_coverage.expand(query, prior)
        # Production performs one hydration after all summary-only selection.
        return hydrate_section_plan(plan, load_turn=self.turns.get, max_context_tokens=3072, max_raw_spans=128)
