"""Resident source-diverse summary routing followed by scoped term coverage."""
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.spine_source_coverage import SpineSourceCoverage
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools.spine_facet_memory import supplemental_plan
from tools.spine_term_memory import ResidentMemory as TermMemory


class ResidentMemory(TermMemory):
    def __init__(self, *args):
        super().__init__(*args)
        try:
            self.source_coverage = SpineSourceCoverage(self.source_spine)
        except Exception:
            self.encoder.close()
            raise

    def source_plan(self, query, dated_question):
        identity = summary_embedding_identity(self.encoder)
        view = ordered_content_query(query)
        vectors = self.encoder.embed_queries([query, view]) if view != query else [self.encoder.embed_query(query)]
        if summary_embedding_identity(self.encoder) != identity:
            raise ValueError("live query encoder changed")
        selected, _ = self.router.route_vectors(query, dated_question, vectors[0], embedding_identity=identity,
            content_vector=vectors[1] if len(vectors) == 2 else None)
        users = self.router.user_addresses.route_vector(query, vectors[0], embedding_identity=identity,
            user_weight=1, max_sections=8, lexical_reserve=0)
        facets, _ = self.facets.route_vector(query, vectors[0], embedding_identity=identity, max_sections=8)
        prior, _ = self.supplement.expand(selected, supplemental_plan(users, facets))
        user_frontier = self.router.user_addresses.route_vector(query, vectors[0], embedding_identity=identity,
            user_weight=1, max_sections=32, lexical_reserve=0)
        facet_frontier, _ = self.facets.route_vector(query, vectors[0], embedding_identity=identity, max_sections=32)
        diverse, source_audit = self.source_coverage.expand(prior, user_frontier, facet_frontier)
        final, term_audit = self.term_coverage.expand(query, diverse)
        return final, {"source_coverage": source_audit, "term_coverage": term_audit}

    def retrieve(self, query, arm, dated_question):
        if arm != "source_spine_diverse":
            return super().retrieve(query, arm, dated_question)
        plan, _ = self.source_plan(query, dated_question)
        return hydrate_section_plan(plan, load_turn=self.turns.get, max_context_tokens=3072, max_raw_spans=128)
