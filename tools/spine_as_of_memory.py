"""Experimental as-of summary routing with the existing reader and raw budgets."""
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.as_of_spine_routing import AsOfSpineRouter, AsOfSpineExpansion
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools.spine_facet_memory import supplemental_plan
from tools.spine_semantic_seed_memory import ResidentMemory as SeedMemory


class ResidentMemory(SeedMemory):
    def __init__(self, *args):
        super().__init__(*args)
        try:
            self.as_of_router = AsOfSpineRouter(self.semantic, self.router.user_addresses, self.facets)
            self.as_of_expansion = AsOfSpineExpansion(self.source_spine, self.supplement,
                                                    self.source_coverage, self.term_coverage)
        except Exception:
            self.encoder.close()
            raise

    def as_of_plan(self, query, dated_question, *, extended_relative_prior=False):
        identity = summary_embedding_identity(self.encoder)
        view = ordered_content_query(query)
        vectors = self.encoder.embed_queries([query, view]) if view != query else [self.encoder.embed_query(query)]
        if summary_embedding_identity(self.encoder) != identity:
            raise ValueError("live query encoder changed")
        plans, audit = self.as_of_router.route_vectors(query, dated_question, vectors[0],
            embedding_identity=identity, content_vector=vectors[1] if len(vectors) == 2 else None,
            extended_relative_prior=extended_relative_prior)
        plan, expansion = self.as_of_expansion.expand(query, dated_question, plans,
            supplemental_plan(plans["users"], plans["facets"]))
        return plan, {"routing": audit, "expansion": expansion}

    def retrieve(self, query, arm, dated_question):
        if arm not in ("source_spine_as_of", "source_spine_as_of_relative"):
            return super().retrieve(query, arm, dated_question)
        plan, _ = self.as_of_plan(query, dated_question,
                                 extended_relative_prior=arm == "source_spine_as_of_relative")
        return hydrate_section_plan(plan, load_turn=self.turns.get, max_context_tokens=3072, max_raw_spans=128)
