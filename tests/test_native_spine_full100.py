import json
from types import SimpleNamespace

import pytest

from memory_condense.application.native_spine_retrieval import ResidentNativeSpineMemory
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from tests.test_native_joint_population import case, manifests
from tests.test_native_spine_routing import fixture, QUERY, DATED
from tools import evaluate_native_spine_full100 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def test_native_build_uses_fresh_embeddings_and_exact_date_eligible_raw_evidence():
    history, semantic, hierarchy, encoder = fixture(future=True)
    memory = ResidentNativeSpineMemory(semantic, hierarchy, encoder=encoder, load_turn=history.get_turn)
    q = {"retrieval_query": QUERY, "prompt_question": DATED}
    baseline = evaluation.build(memory, q, "flat")
    candidate = evaluation.build(memory, q, "hierarchy")
    assert encoder.calls == [QUERY, QUERY]
    assert "botanical garden" in baseline[0][1]["content"]
    assert candidate[1]["sections"][:len(baseline[1]["sections"])] == baseline[1]["sections"]
    assert all(r[2]["query_qwen_passes"] == r[2]["raw_reads_during_routing"] == 0 for r in (baseline, candidate))
    assert "2026-09-13" not in json.dumps(candidate[1])


@pytest.mark.parametrize("producer_format", [evaluation.parent_compiler.FORMAT,
    evaluation.expanding_parent_compiler.FORMAT, evaluation.bounded_parent_compiler.FORMAT])
def test_partial_preparation_stops_before_encoder_corpus_or_provider_construction(tmp_path, monkeypatch, producer_format):
    source, store, hierarchy = manifests()
    store.payload["complete_source_compilation"] = False
    hierarchy.payload["producer_format"] = producer_format
    settings = {key: tmp_path/key for key in ("sources", "store", "vectors", "m_dataset")}
    settings["hierarchies"] = tmp_path/"hierarchies.json"
    publish_sealed_json(settings["sources"]/"sources.json", source.payload)
    actual_source = read_sealed_json(settings["sources"]/"sources.json")
    store.payload["sources_sha256"] = actual_source.sha256
    publish_sealed_json(settings["store"]/"summary-bodies.json", store.payload)
    publish_sealed_json(settings["hierarchies"], hierarchy.payload)
    for name in ("EmbeddingService", "RecoveredParentNativeSpineCorpus", "ExpandingParentNativeSpineCorpus",
                 "BoundedParentNativeSpineCorpus", "_completion_client"):
        monkeypatch.setattr(evaluation, name, lambda *a, **k: pytest.fail("incomplete preparation constructed model runtime"))
    with pytest.raises(ValueError, match="every source body"):
        evaluation.prepare(tmp_path/"evaluation", settings)
    assert not (tmp_path/"evaluation"/"preflight.json").exists()


def synthetic_population(root):
    calls = []
    for ordinal in range(100):
        c = dict(case(ordinal), question=f"Question {ordinal}")
        q = evaluation.question(c)
        prompts = {a: evaluation.protocol.messages(q) for a in evaluation.ARMS}
        hyd = {a: {"fixture": a, "ordinal": ordinal} for a in evaluation.MEMORY_ARMS}
        routing = {a: {"fixture_route": a, "ordinal": ordinal} for a in evaluation.MEMORY_ARMS}
        evidence, _ = publish_sealed_json(root/"evidence"/f"{ordinal:03d}.json", {
            "question": q, "case": c, "messages": prompts, "hydration": hyd, "routing": routing})
        for arm in evaluation.protocol.call_order(ordinal):
            calls.append({"call_index": len(calls), "question": q, "arm": arm,
                "messages": prompts[arm], "messages_sha256": identity_sha256(prompts[arm]),
                "evidence_sha256": evidence.sha256})
    return calls


def test_incomplete_native_answers_cannot_open_references(tmp_path, monkeypatch):
    calls = synthetic_population(tmp_path)
    plan = SimpleNamespace(payload={"calls": calls}, sha256="fixture")
    monkeypatch.setattr(evaluation, "load_preflight", lambda _: plan)
    monkeypatch.setattr(evaluation, "load_references", lambda _: pytest.fail("gold opened before all400 answers"))
    with pytest.raises(ValueError, match="all400"):
        evaluation.judge(tmp_path)


def test_synthetic400_fresh_streams_200_logical_judgments_and_no_call_replay(tmp_path, monkeypatch):
    calls = synthetic_population(tmp_path)
    plan, _ = publish_sealed_json(tmp_path/"preflight.json", {"calls": calls, "settings": {"vectors": "fixture-vectors"},
        "embedding_identity": "fixture", "population_admission_sha256": "synthetic-fixture"})
    monkeypatch.setattr(evaluation, "load_preflight", lambda _: plan)
    monkeypatch.setattr(evaluation, "require_idle", lambda: None)
    corpus = SimpleNamespace(close=lambda: None, load_namespace=lambda *a, **k: object())
    monkeypatch.setattr(evaluation, "open_corpus", lambda _: corpus)
    monkeypatch.setattr(evaluation.vector_compiler, "NativeSummaryVectors", lambda _: object())
    monkeypatch.setattr(evaluation.population, "namespace_receipt", lambda *a: None)
    monkeypatch.setattr(evaluation, "resident", lambda *a: object())
    monkeypatch.setattr(evaluation, "EmbeddingService", lambda **k: SimpleNamespace(close=lambda: None, embed_query=lambda q: None))
    monkeypatch.setattr(evaluation, "summary_embedding_identity", lambda _: "fixture")
    builds, sent = [], []
    def build(memory, q, arm):
        builds.append((q["ordinal"], arm))
        return evaluation.protocol.messages(q), {"fixture": arm, "ordinal": q["ordinal"]}, {"fixture_route": arm, "ordinal": q["ordinal"]}
    monkeypatch.setattr(evaluation, "build", build)
    class Stream:
        def __iter__(self):
            yield {"model": evaluation.MODEL, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
            yield {"model": evaluation.MODEL, "choices": [{"index": 0, "delta": {"content": "fact"}, "finish_reason": "stop"}]}
        def close(self):
            pass
    class Client:
        max_retries = 0
        chat = property(lambda self: SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        def with_options(self, **kwargs):
            return self
        def close(self):
            pass
        def create(self, **kwargs):
            sent.append(kwargs)
            if kwargs.get("stream"):
                return Stream()
            assert (tmp_path/"answers.json").exists()
            return SimpleNamespace(id=f"judge-{len(sent)}", model="codex_sdk/gpt-5.6-sol", usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content="CORRECT"), finish_reason="stop")])
    monkeypatch.setattr(evaluation, "_completion_client", lambda *a: Client())
    def references(settings):
        assert len(list((tmp_path/"journal").glob("*.response.json"))) == 400
        assert (tmp_path/"answers.json").exists()
        return {f"q{i}": "fact" for i in range(100)}
    monkeypatch.setattr(evaluation, "load_references", references)
    report = evaluation.run(tmp_path, True)
    assert len(builds) == 200
    assert sum(bool(c.get("stream")) for c in sent) == 400
    assert sum(not c.get("stream") for c in sent) == 100
    assert report.payload["accuracy"] == {"flat": 100, "hierarchy": 100}
    assert report.payload["same_streamed_answers_scored"] is True
    monkeypatch.setattr(evaluation, "_completion_client", lambda *a: pytest.fail("replay made a provider call"))
    assert evaluation.judge(tmp_path, False).sha256 == report.sha256
    with pytest.raises(ValueError, match="another answer release"):
        evaluation.run(tmp_path, True)


@pytest.mark.parametrize("changed", [False, True])
def test_references_stream_the_bound_native_m_dataset_and_check_the_original_answer(tmp_path, changed):
    c = dict(case(), reference_sha256=quote_sha256("fact"))
    data = tmp_path/"m.json"
    data.write_text(json.dumps([{**c, "answer": "altered" if changed else "fact"}]), encoding="utf-8")
    publish_sealed_json(tmp_path/"sources.json", {"source_artifacts": {"m_dataset_sha256": evaluation.digest(data)}})
    publish_sealed_json(tmp_path/"evaluation-cases.json", {"cases": [c]})
    settings = {"sources": tmp_path, "m_dataset": data}
    if changed:
        with pytest.raises(ValueError, match="locked M question"):
            evaluation.load_references(settings)
    else:
        assert evaluation.load_references(settings) == {"q0": "fact"}
