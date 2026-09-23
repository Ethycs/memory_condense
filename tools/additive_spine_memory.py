"""Retrieve the original packet, then add bounded summary-ranked user evidence."""
from memory_condense.application.additive_threaded_context import supplement_threaded_context
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.application.threaded_section_context import TranscriptOrder, render_threaded_sections
from memory_condense.search.fine_spine_supplement import FineSpineSupplement
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from memory_condense.search.summary_time_prior_v2 import question_day
from tools.fine_spine_memory import ResidentMemory as FineMemory
from tools.spine_facet_memory import supplemental_plan


class ResidentMemory(FineMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.order = TranscriptOrder(self.turns.values())
        self.fine_supplement = FineSpineSupplement(self.fine_router)

    def build(self, query, dated_question, *, include_supplement):
        identity = summary_embedding_identity(self.encoder)
        view = ordered_content_query(query)
        vectors = self.encoder.embed_queries([query, view]) if view != query else [self.encoder.embed_query(query)]
        if summary_embedding_identity(self.encoder) != identity:
            raise ValueError('live query encoder changed')
        plans, route_audit = self.as_of_router.route_vectors(query, dated_question, vectors[0],
            embedding_identity=identity, content_vector=vectors[1] if len(vectors) == 2 else None)
        prior, expansion = self.as_of_expansion.expand(query, dated_question, plans,
            supplemental_plan(plans['users'], plans['facets']))
        reserved, reservation = self.relative_reservation.reserve(query, dated_question, prior,
            self.as_of_router.users._spine.score_all(vectors[0]))
        base = hydrate_section_plan(reserved, load_turn=self.turns.get, max_context_tokens=3072, max_raw_spans=128)
        grouped = render_threaded_sections(base, self.order)
        addition = None
        audit = {'routing': route_audit, 'expansion': expansion, 'relative_reservation': reservation}
        if include_supplement:
            selected_turn_ids = {r.span.turn_id for s in base.sections for r in s.evidence}
            plan, fine_audit = self.fine_supplement.route_vector(query, dated_question, vectors[0],
                embedding_identity=identity, selected_turn_ids=selected_turn_ids)
            addition = supplement_threaded_context(base, self.order, plan, load_turn=self.turns.get,
                                                  asked_day=question_day(query, dated_question))
            audit['fine_supplement'] = fine_audit
        return base, grouped, addition, audit
