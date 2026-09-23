"""Resident Qwen summary-tree traversal beside the unchanged flat control."""
from pathlib import Path

import numpy as np

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.bounded_spine_hierarchy import BoundedSpineHierarchyRouter, project_hierarchy_leaves
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.summary_semantic_index import SemanticSectionIndex, summary_embedding_identity
from memory_condense.search.summary_time_prior_v2 import question_day
from tools.matched_eval.artifacts import read_sealed_json
from tools.spine_relative_reservation_memory import ResidentMemory as RelativeMemory


class ResidentMemory(RelativeMemory):
    def __init__(self, *args, parent_root, parent_sha256, linker):
        parent_root = Path(parent_root)
        artifact = read_sealed_json(parent_root / "hierarchy.json")
        preflight = read_sealed_json(parent_root / "preflight.json")
        topology = read_sealed_json(parent_root / "topology.json")
        p = artifact.payload
        if (artifact.sha256 != parent_sha256 or p["preflight_sha256"] != preflight.sha256
            or p["topology_sha256"] != topology.sha256 or topology.payload["preflight_sha256"] != preflight.sha256
            or p["parent_summary_compilation_complete"] is not True or p["complete_namespace"] is not True
            or p["raw_inputs_to_qwen"] is not False or p["parent_count"] <= 0):
            raise ValueError("hierarchical memory requires a completed bound summary hierarchy")
        if preflight.payload["serving_index_sha256"] != args[1]:
            raise ValueError("parent hierarchy belongs to another serving memory")
        hierarchy = SectionSummaryIndex.from_json(p["index_json"])
        super().__init__(*args)
        try:
            if (tuple(s for s in hierarchy.sections if not s.child_section_ids) != self.semantic.sections
                or p["leaf_index_sha256"] != self.semantic.hierarchy.receipt_sha256):
                raise ValueError("restored hierarchy changed the evaluated leaf population")
            semantic = SemanticSectionIndex(hierarchy,
                np.load(Path(args[0]) / "summary-vectors.npy", allow_pickle=False),
                embedding_identity=self.semantic.embedding_identity)
            self.hierarchy_router = BoundedSpineHierarchyRouter(semantic)
            self.linker = linker  # Owned by the caller, resident across memories.
            self.parent_sha256 = artifact.sha256
        except Exception:
            self.encoder.close()
            raise

    def hierarchical_plan(self, query, dated_question):
        identity = summary_embedding_identity(self.encoder)
        vector = self.encoder.embed_query(query)
        if summary_embedding_identity(self.encoder) != identity:
            raise ValueError("live query encoder changed")
        asked = question_day(query, dated_question)
        attention = self.hierarchy_router.route_vector(query, vector, embedding_identity=identity,
            linker=self.linker, asked_day=asked, root_shortlist=8, beam=4, max_depth=16)
        projected, projection = project_hierarchy_leaves(attention, self.hierarchy_router.index, asked)
        return projected, {"parent_hierarchy_sha256": self.parent_sha256,
            "attention_plan": attention.identity_payload(), "projection": projection,
            "live_query_embedding": True, "raw_inputs_to_qwen": False}

    def retrieve(self, query, arm, dated_question):
        if arm != "source_spine_qwen_hierarchy":
            return super().retrieve(query, arm, dated_question)
        plan, _ = self.hierarchical_plan(query, dated_question)
        return hydrate_section_plan(plan, load_turn=self.turns.get,
                                    max_context_tokens=3072, max_raw_spans=128)
