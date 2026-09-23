"""Resident summary-passage supplement with exact user-turn hydration."""
import hashlib
from pathlib import Path

import numpy as np

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.source_spine_supplement import SourceSpineSupplement
from memory_condense.search.spine_summary_facets import SpineFacetAddressIndex
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools.evaluate_source_spine_overflow import ResidentMemory as OverflowMemory
from tools.matched_eval.artifacts import read_sealed_json


def supplemental_plan(user_plan, facet_plan):
    if (user_plan.index_sha256, user_plan.query_sha256) != (
            facet_plan.index_sha256, facet_plan.query_sha256):
        raise ValueError("supplemental address plans differ in hierarchy or query")
    if user_plan.eligible_source_ids is not None or facet_plan.eligible_source_ids is not None:
        raise ValueError("this supplement requires unscoped summary routes")
    selected = {}
    for i in range(max(len(user_plan.routes), len(facet_plan.routes))):
        for plan in (user_plan, facet_plan):
            if i < len(plan.routes):
                section = plan.routes[i].section
                if section.section_id in selected and selected[section.section_id] != section:
                    raise ValueError("supplemental descriptors disagree")
                selected.setdefault(section.section_id, section)
    return SectionRoutePlan(user_plan.index_sha256, user_plan.query_sha256,
        tuple(SectionRoute(s, 1 / (i + 1), ()) for i, s in enumerate(selected.values())),
        max(user_plan.matched_section_count, facet_plan.matched_section_count), None,
        max(1, len(selected)), routing_backend="summary_dense")


class ResidentMemory(OverflowMemory):
    def __init__(self, index_root, index_sha256, addresses_root, atoms_path, facets_root,
                 addresses_sha256=None, atoms_sha256=None, facets_sha256=None):
        manifest = read_sealed_json(facets_root / "addresses.json")
        preflight = read_sealed_json(facets_root / "preflight.json")
        path = facets_root / "facet-vectors.npy"
        p = manifest.payload
        if (p["base_index_sha256"] != index_sha256 or p["preflight_sha256"] != preflight.sha256 or
                p["matrix_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest() or
                (facets_sha256 is not None and manifest.sha256 != facets_sha256)):
            raise ValueError("facet address inputs differ from the compiled index")
        for name, digest in preflight.payload["implementation"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError("facet compiler implementation changed")
        matrix = np.load(path, allow_pickle=False)
        super().__init__(index_root, index_sha256, addresses_root, atoms_path, addresses_sha256, atoms_sha256)
        try:
            self.facets = SpineFacetAddressIndex(self.semantic.hierarchy, matrix,
                embedding_identity=p["embedding_identity"])
            if self.facets.receipt_sha256 != p["address_index_sha256"]:
                raise ValueError("facet address identity changed")
            if self.facets.embedding_identity != self.semantic.embedding_identity:
                raise ValueError("facet and whole-summary embeddings differ")
            self.facets_sha256 = manifest.sha256
            self.supplement = SourceSpineSupplement(self.source_spine)
        except Exception:
            self.encoder.close()
            raise

    def retrieve(self, query, arm, dated_question):
        if arm != "source_spine_facets":
            return super().retrieve(query, arm, dated_question)
        identity = summary_embedding_identity(self.encoder)
        view = ordered_content_query(query)
        vectors = self.encoder.embed_queries([query, view]) if view != query else [self.encoder.embed_query(query)]
        if summary_embedding_identity(self.encoder) != identity:
            raise ValueError("live query encoder changed")
        selected, _ = self.router.route_vectors(query, dated_question, vectors[0], embedding_identity=identity,
            content_vector=vectors[1] if len(vectors) == 2 else None)
        wider_users = self.router.user_addresses.route_vector(query, vectors[0], embedding_identity=identity,
            user_weight=1, max_sections=8, lexical_reserve=0)
        facets, _ = self.facets.route_vector(query, vectors[0], embedding_identity=identity, max_sections=8)
        plan, _ = self.supplement.expand(selected, supplemental_plan(wider_users, facets))
        return hydrate_section_plan(plan, load_turn=self.turns.get, max_context_tokens=3072, max_raw_spans=128)
