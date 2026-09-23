"""Live fine user-summary routing on the unchanged complete memory population."""
import hashlib
from pathlib import Path

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.fine_spine_routing import FineSpineRouter
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools.compile_spine_semantic_index import load_index
from tools.matched_eval.artifacts import read_sealed_json
from tools.spine_facet_memory import supplemental_plan
from tools.spine_relative_reservation_memory import ResidentMemory as RelativeMemory


class ResidentMemory(RelativeMemory):
    def __init__(self, *args, fine_root):
        super().__init__(*args)
        try:
            preflight = read_sealed_json(fine_root.parent / 'preflight.json')
            complete = read_sealed_json(fine_root.parent / 'complete.json')
            manifest, fine = load_index(fine_root)
            p = manifest.payload
            if (p['index_sha256'] != args[1] or p['atoms_sha256'] != self.atoms_sha256 or
                    p['source_partition_sha256'] != self.source_spine.index.receipt_sha256 or
                    p['preflight_sha256'] != preflight.sha256 or
                    complete.payload['preflight_sha256'] != preflight.sha256 or
                    {'root': str(fine_root.resolve()), 'sha256': manifest.sha256} not in complete.payload['indexes'] or
                    fine.embedding_identity != self.semantic.embedding_identity):
                raise ValueError('fine addresses changed their source or encoder binding')
            for name, digest in preflight.payload['implementation'].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                    raise ValueError('fine compiler changed')
            self.fine_index_sha256 = manifest.sha256
            self.fine_router = FineSpineRouter(fine, self.source_spine)
        except Exception:
            self.encoder.close()
            raise

    def fine_plan(self, query, dated_question):
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
        final, fine_audit = self.fine_router.route_vector(query, dated_question, vectors[0],
            embedding_identity=identity, prior=reserved, reserved_ids=reservation.get('reserved_section_ids', ()))
        return final, {'routing': route_audit, 'expansion': expansion,
                       'relative_reservation': reservation, 'fine': fine_audit}

    def retrieve(self, query, arm, dated_question):
        limits = {'fine_spine': 3072, 'fine_spine_compact': 2048}
        if arm not in limits:
            return super().retrieve(query, arm, dated_question)
        plan, _ = self.fine_plan(query, dated_question)
        return hydrate_section_plan(plan, load_turn=self.turns.get, max_context_tokens=limits[arm], max_raw_spans=128)
