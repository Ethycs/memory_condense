from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tools import compile_native_spine_vectors as vectors
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tests.test_native_spine_routing import fixture


class Encoder:
    model_name = "deterministic vector test double"
    model_revision = "test"
    checkpoint_sha256 = "test"
    execution_identity = {"test_double": True}

    def __init__(self):
        self.calls = []

    def embed_queries(self, texts):
        self.calls.append(tuple(texts))
        return np.array([[len(text), 1] for text in texts], dtype=np.float32)


def store(monkeypatch, texts):
    class Store:
        def __init__(self, _):
            self.manifest = SimpleNamespace(sha256="synthetic-summary-store", payload={
                "sources_sha256": "synthetic-sources", "complete_source_compilation": False})
            self.connection = SimpleNamespace(execute=lambda _: [("synthetic-body",)])

        def load(self, _):
            return [{"summary": text} for text in texts]

        def close(self):
            pass
    monkeypatch.setattr(vectors, "AdmittedSummaryBodies", Store)


def test_distinct_occurrences_reuse_identical_summary_vectors_and_completed_replay_is_zero_call(tmp_path, monkeypatch):
    history, semantic, _, _ = fixture()
    texts = [a.summary for a in history.atoms]
    store(monkeypatch, [*texts, *texts])
    encoder = Encoder()
    plan = vectors.prepare("unused", tmp_path, encoder)
    assert plan.payload["unique_summary_count"] == 3
    result = vectors.execute(tmp_path, encoder)
    assert sum(len(call) for call in encoder.calls) == 3
    cache = vectors.NativeSummaryVectors(tmp_path)
    index = cache.semantic_index(semantic.hierarchy)
    assert index.sections == semantic.sections
    assert not result.payload["complete_source_compilation"]
    encoder.calls.clear()
    assert vectors.execute(tmp_path, encoder).sha256 == result.sha256
    assert encoder.calls == []


def test_successor_snapshot_embeds_only_new_summary_strings(tmp_path, monkeypatch):
    store(monkeypatch, ["Existing summary."])
    encoder = Encoder()
    old, new = tmp_path/"old", tmp_path/"new"
    vectors.prepare("unused", old, encoder)
    vectors.execute(old, encoder)
    previous = vectors.NativeSummaryVectors(old).values["Existing summary."].copy()
    store(monkeypatch, ["Existing summary.", "A new summary."])
    encoder.calls.clear()
    vectors.prepare("unused", new, encoder, reuse_roots=(old,))
    vectors.execute(new, encoder)
    assert encoder.calls == [("A new summary.",)]
    assert np.array_equal(vectors.NativeSummaryVectors(new).values["Existing summary."], previous)


def test_changed_matrix_and_missing_summary_are_rejected(tmp_path, monkeypatch):
    store(monkeypatch, ["Only this summary."])
    encoder = Encoder()
    vectors.prepare("unused", tmp_path, encoder)
    vectors.execute(tmp_path, encoder)
    _, semantic, _, _ = fixture()
    with pytest.raises(ValueError, match="no authenticated embedding"):
        vectors.NativeSummaryVectors(tmp_path).semantic_index(semantic.hierarchy)
    path = tmp_path/"batches"/"000000.npy"
    path.write_bytes(b"changed vector file")
    with pytest.raises(ValueError, match="vector file changed"):
        vectors.NativeSummaryVectors(tmp_path)


def test_failed_embedding_publishes_no_completion_and_successful_checkpoints_resume(tmp_path, monkeypatch):
    store(monkeypatch, [f"Summary {i}" for i in range(130)])
    class Failing(Encoder):
        def embed_queries(self, texts):
            if self.calls:
                raise RuntimeError("local embedding failure")
            return super().embed_queries(texts)
    encoder = Failing()
    vectors.prepare("unused", tmp_path, encoder)
    with pytest.raises(RuntimeError, match="local embedding failure"):
        vectors.execute(tmp_path, encoder)
    assert not (tmp_path/"result.json").exists()
    first = read_sealed_json(tmp_path/"batches"/"000000.json").sha256
    successor = Encoder()
    vectors.execute(tmp_path, successor)
    assert sum(len(call) for call in successor.calls) == 2
    assert read_sealed_json(tmp_path/"batches"/"000000.json").sha256 == first


def test_zero_vectors_and_encoder_identity_changes_are_rejected(tmp_path, monkeypatch):
    store(monkeypatch, ["A summary."])
    class Zero(Encoder):
        def embed_queries(self, texts):
            return np.zeros((len(texts), 2), dtype=np.float32)
    vectors.prepare("unused", tmp_path, Zero())
    with pytest.raises(ValueError, match="zero or invalid norms"):
        vectors.execute(tmp_path, Zero())
    changed = Encoder()
    changed.model_revision = "other"
    with pytest.raises(ValueError, match="encoder changed"):
        vectors.execute(tmp_path, changed)
    assert not (tmp_path/"result.json").exists()
