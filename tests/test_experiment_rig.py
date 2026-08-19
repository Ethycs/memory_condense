from __future__ import annotations

import json
import re
import zlib

import numpy as np
import pytest

from memory_condense.associations.association_store import AssociationArtifact
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.tooling.experiment_rig import (
    AssociativeSweepRig,
    SweepArm,
    SweepQuestion,
    load_anchor_pack,
    save_anchor_pack,
)
from memory_condense.domain.schemas import Chunk


class CountingEmbedder:
    def __init__(self, dim: int = 32) -> None:
        self._dim = dim
        self.query_calls = 0
        self.batch_calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        values = np.zeros(self._dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            values[zlib.crc32(token.encode()) % self._dim] += 1
        if not values.any():
            values[0] = 1
        return values

    def embed_query(self, query: str) -> np.ndarray:
        self.query_calls += 1
        return self._vec(query)

    def embed_queries(self, queries) -> np.ndarray:
        self.batch_calls += 1
        return np.stack([self._vec(query) for query in queries])

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk.model_copy(update={"embedding": self._vec(chunk.text).tolist()})
            for chunk in chunks
        ]


def _artifact() -> AssociationArtifact:
    return AssociationArtifact.create(
        model_id="Qwen/Qwen3-8B",
        checkpoint_id="test",
        prefix_layers=7,
        head_layer=1,
        cav_layer=5,
        concept_names=("context", "binding"),
        head_count=4,
    )


@pytest.fixture
def rig_fixture(tmp_path):
    embedder = CountingEmbedder()
    mc = MemoryCondenser(data_dir=tmp_path / "store", embedder=embedder)
    for text in (
        "SQLite storage decision and project database",
        "Python implementation preference and tooling",
        "The index must remain below one gigabyte",
    ):
        mc.ingest("user", text)
    anchors = mc.search_hybrid_many(["SQLite project storage"], k=3)[0]
    artifact = mc.associations.register_artifact(_artifact())
    mc.associations.upsert_edge(
        anchors[0].chunk.chunk_id,
        anchors[2].chunk.chunk_id,
        artifact.artifact_id,
        [0.9, 0.2, 0.1, 0.1],
        qk_score=0.9,
    )
    question = SweepQuestion(
        question_id="q0",
        gold_chunk_ids=(anchors[2].chunk.chunk_id,),
        anchors=tuple(anchors),
        question="SQLite project storage",
    )
    yield mc, embedder, artifact, question
    mc.close()


def test_anchor_pack_is_hashed_lean_and_round_trips(tmp_path, rig_fixture):
    _, _, _, question = rig_fixture
    path = tmp_path / "anchors.json"
    payload = save_anchor_pack(path, [question], metadata={"split": "test"})
    loaded, loaded_payload = load_anchor_pack(path)

    assert loaded_payload["sha256"] == payload["sha256"]
    assert loaded[0].gold_chunk_ids == question.gold_chunk_ids
    assert all(anchor.chunk.embedding is None for anchor in loaded[0].anchors)

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["questions"][0]["question_id"] = "changed"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_anchor_pack(path)


def test_parallel_sweep_reuses_anchors_without_embedding(rig_fixture):
    mc, embedder, artifact, question = rig_fixture
    embedder.query_calls = 0
    embedder.batch_calls = 0
    report = AssociativeSweepRig(mc, artifact.artifact_id, workers=2).run(
        [question],
        [
            SweepArm("hybrid_k2", k=2, association_slots=0, repeats=2),
            SweepArm("linked_k2", k=2, association_slots=1, repeats=2),
        ],
    )

    assert report["execution"]["workers"] == 2
    assert report["execution"]["qwen_workers"] == 0
    assert embedder.query_calls == 0
    assert embedder.batch_calls == 0
    assert report["arms"]["hybrid_k2"]["linked_recall"] == 0.0
    assert report["arms"]["linked_k2"]["linked_recall"] == 1.0
    assert report["arms"]["linked_k2"]["qk_links"] == 1
    assert report["arms"]["linked_k2"]["recall_changes"] == {
        "recovered": 1,
        "lost": 0,
    }
    assert report["arms"]["linked_k2"]["qk_hop_counts"] == {"1": 1}
    assert len(report["arms"]["linked_k2"]["elapsed_samples_s"]) == 2


def test_sweep_can_run_heat_diffusion_and_measure_source_exposure(rig_fixture):
    mc, embedder, artifact, question = rig_fixture
    embedder.query_calls = 0
    embedder.batch_calls = 0

    report = AssociativeSweepRig(mc, artifact.artifact_id, workers=1).run(
        [question],
        [
            SweepArm(
                "heat_k2",
                k=2,
                association_slots=1,
                qk_reserve=1,
                neighbors_per_anchor=3,
                association_hops=1,
                max_association_candidates=8,
                cav_candidates=0,
                retrieval_strategy="heat",
                heat_weighted_packing=True,
                repeats=2,
            )
        ],
    )

    heat = report["arms"]["heat_k2"]
    assert heat["linked_recall"] == 1.0
    assert heat["heat_links"] == 1
    assert heat["heat_hop_counts"] == {"1": 1}
    assert heat["mean_sources_exposed"] >= 1.0
    assert heat["mean_linked_packed_tokens"] > 0
    assert embedder.query_calls == 0
    assert embedder.batch_calls == 0


def test_pruned_arm_uses_an_isolated_sqlite_snapshot(rig_fixture):
    mc, _, artifact, question = rig_fixture
    source = question.anchors[0].chunk.chunk_id
    lower_ranked = question.anchors[1].chunk.chunk_id
    mc.associations.upsert_edge(
        source,
        lower_ranked,
        artifact.artifact_id,
        [0.1, 0.1, 0.1, 0.1],
        qk_score=0.1,
    )
    assert mc.associations.stats(artifact.artifact_id)["edges"] == 2

    report = AssociativeSweepRig(mc, artifact.artifact_id, workers=1).run(
        [question],
        [
            SweepArm(
                "pruned",
                k=1,
                association_slots=1,
                neighbors_per_anchor=3,
                cav_candidates=0,
                prune_max_neighbors=1,
            )
        ],
    )

    pruning = report["arms"]["pruned"]["pruning"]
    assert pruning == {
        "max_neighbors": 1,
        "edges_before": 2,
        "edges_removed": 1,
        "edges_after": 1,
        "head_payload_bytes_before": 32,
        "head_payload_bytes_after": 16,
        "retained_request_token_state_bytes": 0,
        "retained_token_state_bytes": 0,
    }
    assert report["arms"]["pruned"]["linked_recall"] == 1.0
    assert mc.associations.stats(artifact.artifact_id)["edges"] == 2
