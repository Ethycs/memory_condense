from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from memory_condense.eval.compiled_cache import compiled_store_ingest_fn
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.eval.transition_trace import (
    TransitionArm,
    build_transition_trace,
    load_transition_trace,
    save_transition_trace,
    score_transition_arm,
)
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample


class TraceEmbedder:
    model_name = "test/trace-v1"
    dim = 4

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        vector = np.zeros(4, dtype=np.float32)
        vector[0 if "anchor" in text.lower() else 1] = 1.0
        return vector

    def embed_chunks(self, chunks):
        return [
            chunk.model_copy(update={"embedding": self._vector(chunk.text).tolist()})
            for chunk in chunks
        ]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def embed_queries(self, texts) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])


SAMPLE = BenchmarkSample(
    sample_id="transition-sample",
    turns=[
        ("user", "unrelated before"),
        ("assistant", "distinctive anchor topic"),
        ("user", "the secret answer is cobalt"),
        ("assistant", "unrelated after"),
    ],
    turn_source_ids=["session-a"] * 4,
    questions=[
        BenchmarkQuestion(
            question_id="transition-q",
            question="What followed the anchor?",
            answer="cobalt",
            evidence_sources=["session-a"],
        )
    ],
)


def _pack(tmp_path: Path):
    embedder = TraceEmbedder()
    config = EvalConfig(
        retrieval=RetrievalConfig(mode="hybrid_neighbor", k=1, alpha=0.65)
    )
    ingest = compiled_store_ingest_fn(
        tmp_path / "cache",
        embedder=embedder,
    )
    return build_transition_trace(
        [SAMPLE],
        config,
        ingest_fn=ingest,
        embedder=embedder,
        dataset_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        split="development",
        max_radius=2,
    )


def test_transition_trace_preserves_routes_provenance_and_hash(tmp_path: Path):
    pack = _pack(tmp_path)

    assert pack.verified()
    assert len(pack.questions) == 1
    candidates = pack.questions[0].candidates
    assert candidates[0].route == "hybrid_anchor"
    target = next(candidate for candidate in candidates if "cobalt" in candidate.text)
    assert target.route == "source_neighbor"
    assert target.source_id == "session-a"
    assert target.transition_distance == 1
    assert target.transition_direction == "next"
    assert target.anchor_chunk_id == candidates[0].chunk_id

    path = save_transition_trace(pack, tmp_path / "trace.json")
    assert load_transition_trace(path) == pack


def test_transition_trace_arm_scoring_measures_stay_and_next(tmp_path: Path):
    pack = _pack(tmp_path)
    stay = TransitionArm(name="stay", retain_anchors=1, neighbor_slots=0)
    walk = TransitionArm(
        name="next-1",
        retain_anchors=0,
        neighbor_slots=1,
        max_distance=1,
        direction="next",
    )

    stay_score, stay_hits = score_transition_arm(pack, stay)
    walk_score, _walk_hits = score_transition_arm(
        pack,
        walk,
        stay_hits=stay_hits,
    )

    assert stay_score.literal_recall == 0.0
    assert walk_score.literal_recall == 1.0
    assert walk_score.gained_vs_stay == 1
    assert walk_score.lost_vs_stay == 0
    assert walk_score.evidence_all_source_recall == 1.0


def test_transition_trace_loader_rejects_tampering(tmp_path: Path):
    path = save_transition_trace(_pack(tmp_path), tmp_path / "trace.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["questions"][0]["answer"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_transition_trace(path)
