from types import SimpleNamespace

import numpy as np

from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.summary_semantic_index import SemanticSectionIndex, summary_embedding_identity
from tests.test_bounded_spine_hierarchy import fixture
from tests.test_summary_shortlist_attention import Linker
from tools.hierarchical_spine_memory import ResidentMemory, RelativeMemory
from tools.matched_eval.artifacts import publish_sealed_json


def test_resident_adapter_embeds_live_traverses_parents_and_hydrates_with_original_renderer(tmp_path, monkeypatch):
    turns, router = fixture()
    calls = []
    encoder = SimpleNamespace(model_name="test-bge", model_revision="test", checkpoint_sha256="a" * 64,
        execution_identity={"output_dtype":"float32"}, close=lambda: None)
    def embed(query):
        calls.append(query)
        return np.array([1., 0.], dtype=np.float32)
    encoder.embed_query = embed
    matrix = np.array([[1., 0.], [.9, .4358899], [0., 1.], [0., 1.]], dtype=np.float32)
    np.save(tmp_path / "summary-vectors.npy", matrix, allow_pickle=False)
    leaf_index = SectionSummaryIndex(router.semantic.sections)
    original = SemanticSectionIndex(leaf_index, matrix, embedding_identity=summary_embedding_identity(encoder))
    def initialize(self, *args):
        self.semantic, self.encoder = original, encoder
        self.turns = {t.turn_id:t for t in turns}
    monkeypatch.setattr(RelativeMemory, "__init__", initialize)
    monkeypatch.setattr(RelativeMemory, "retrieve", lambda self, *args: "unchanged control")
    base_sha = "b" * 64
    root = tmp_path / "parents"
    preflight, _ = publish_sealed_json(root / "preflight.json", {"serving_index_sha256":base_sha})
    topology, _ = publish_sealed_json(root / "topology.json", {"preflight_sha256":preflight.sha256})
    artifact, _ = publish_sealed_json(root / "hierarchy.json", {
        "preflight_sha256":preflight.sha256, "topology_sha256":topology.sha256,
        "parent_summary_compilation_complete":True, "complete_namespace":True,
        "raw_inputs_to_qwen":False, "parent_count":2, "index_json":router.index.to_json(),
        "leaf_index_sha256":leaf_index.receipt_sha256})
    linker = Linker()
    linker.max_candidates = 8
    memory = ResidentMemory(tmp_path, base_sha, parent_root=root, parent_sha256=artifact.sha256, linker=linker)
    result = memory.retrieve("travel completed", "source_spine_qwen_hierarchy",
                             "[Question asked at 2026/09/09 (Wed) 12:00]\ntravel completed")
    assert calls == ["travel completed"] and len(linker.inputs) == 2
    assert "RAW_CANARY" not in repr(linker.inputs)
    assert all(e.text == memory.turns[e.span.turn_id].text for s in result.sections for e in s.evidence)
    assert sum(len(s.evidence) for s in result.sections) == 4
    assert memory.retrieve("query", "source_spine_relative_reservation", "dated") == "unchanged control"
