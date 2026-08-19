from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace

import pytest

from memory_condense.eval import diffuse_longmemeval_runtime as runtime_module
from memory_condense.application.discourse_sources import SourceChunkStream
from memory_condense.domain.discourse import DiscourseSnapshot
from memory_condense.domain.schemas import Chunk, RetrievalResult, Turn
from memory_condense.eval.diffuse_longmemeval_runtime import (
    DiffuseLongMemEvalRuntimeConfig,
    DiffuseLongMemEvalRuntimeFactories,
    FrozenLegacyDiffuseInputProvider,
    ResidencyPreflightObservation,
    build_diffuse_longmemeval_execution_binding,
    freeze_legacy_query_inputs,
    gold_blind_from_treatment_sample,
    retrieve_exact_legacy_anchors,
)
from memory_condense.eval.schemas import (
    ChunkerConfig,
    EvalConfig,
    RetrievalConfig,
)
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)
from memory_condense.modeling.qwen_prefix import (
    DEFAULT_MODEL_ID as QWEN_MODEL_ID,
    DEFAULT_MODEL_REVISION as QWEN_MODEL_REVISION,
    expected_prefix_checkpoint_sha256,
)


@dataclass(frozen=True)
class _TreatmentQuestion:
    question_id: str
    question: str
    question_date: str | None


@dataclass(frozen=True)
class _TreatmentSample:
    sample_id: str
    turns: tuple[tuple[str, str], ...]
    turn_source_ids: tuple[str | None, ...]
    turn_created_at: tuple[datetime | None, ...]
    questions: tuple[_TreatmentQuestion, ...]


def _eval_config(retrieval: RetrievalConfig | None = None) -> EvalConfig:
    return EvalConfig(
        chunker=ChunkerConfig(min_tokens=3, max_tokens=47),
        retrieval=retrieval or RetrievalConfig(mode="dense", k=5),
        embedding_device="cuda",
        max_prompt_tokens=4096,
    )


def test_treatment_adapter_keeps_date_out_of_the_retrieval_query() -> None:
    timestamp = datetime(2025, 1, 2, tzinfo=timezone.utc)
    sample = _TreatmentSample(
        sample_id="safe-sample",
        turns=(("user", "The launch badge was amber."),),
        turn_source_ids=("source-a",),
        turn_created_at=(timestamp,),
        questions=(
            _TreatmentQuestion(
                question_id="q-1",
                question="What color was the launch badge?",
                question_date="2025/02/03",
            ),
        ),
    )

    blind = gold_blind_from_treatment_sample(sample)

    assert blind.questions[0].retrieval_query == (
        "What color was the launch badge?"
    )
    assert blind.questions[0].prompt_question == (
        "[Question asked at 2025/02/03]\nWhat color was the launch badge?"
    )
    assert not hasattr(blind.questions[0], "answer")
    assert blind.turn_source_ids == ("source-a",)


class _RecordingRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def search_hybrid_graph(self, query: str, **kwargs):
        self.calls.append(("hybrid_graph", query, kwargs))
        return []

    def search_hybrid(self, query: str, **kwargs):
        self.calls.append(("hybrid", query, kwargs))
        return []


def test_exact_anchor_router_maps_every_live_hybrid_graph_flag() -> None:
    retrieval = RetrievalConfig(
        mode="hybrid_graph",
        k=7,
        ef_search=73,
        candidates=211,
        alpha=0.42,
        neighbor_radius=3,
        neighbor_slots=9,
        neighbor_direction="previous",
        source_slots=17,
        source_candidate_pool=211,
        source_activation_k=11,
        query_facet_retrieval=True,
        query_facet_slots=3,
        query_facet_max=3,
        role_aware_retrieval=True,
        role_user_weight=1.4,
        role_assistant_weight=0.6,
        role_system_weight=0.2,
        multi_fact_source_diversity=True,
        source_tfisf_activation=True,
        source_tfisf_slots=5,
        source_hsc_activation=True,
        source_hsc_slots=6,
        source_hsc_hops=3,
        source_hsc_chunk_slots=4,
        source_partition_routing=True,
        source_partition_slots=2,
        source_partition_separator="//",
        source_local_search=True,
        qwen_feedback_slots=7,
        qwen_feedback_seed_slots=4,
        qwen_feedback_evidence_tokens=41,
        qwen_feedback_query_tokens=181,
    )
    recorder = _RecordingRetriever()

    assert retrieve_exact_legacy_anchors(recorder, "  exact query  ", retrieval) == ()

    assert recorder.calls == [
        (
            "hybrid_graph",
            "exact query",
            {
                "k": 7,
                "neighbor_radius": 3,
                "neighbor_slots": 9,
                "neighbor_direction": "previous",
                "source_slots": 17,
                "source_candidate_pool": 211,
                "source_activation_k": 11,
                "query_facet_retrieval": True,
                "query_facet_slots": 3,
                "query_facet_max": 3,
                "role_aware_retrieval": True,
                "role_user_weight": 1.4,
                "role_assistant_weight": 0.6,
                "role_system_weight": 0.2,
                "multi_fact_source_diversity": True,
                "source_tfisf_activation": True,
                "source_tfisf_slots": 5,
                "source_hsc_activation": True,
                "source_hsc_slots": 6,
                "source_hsc_hops": 3,
                "source_hsc_chunk_slots": 4,
                "source_partition_routing": True,
                "source_partition_slots": 2,
                "source_partition_separator": "//",
                "source_local_search": True,
                "use_source_reranker": False,
                "use_attention_feedback": False,
                "feedback_slots": 7,
                "feedback_seed_slots": 4,
                "feedback_evidence_tokens": 41,
                "feedback_query_tokens": 181,
                "ef_search": 73,
                "candidates": 211,
                "alpha": 0.42,
            },
        )
    ]


def test_effective_hybrid_and_packed_modes_do_not_change_meaning() -> None:
    recorder = _RecordingRetriever()
    hybrid = RetrievalConfig(
        mode="dense",
        hybrid=True,
        k=4,
        ef_search=61,
        candidates=99,
        alpha=0.31,
    )

    retrieve_exact_legacy_anchors(recorder, "query", hybrid)

    assert recorder.calls == [
        (
            "hybrid",
            "query",
            {"k": 4, "ef_search": 61, "candidates": 99, "alpha": 0.31},
        )
    ]
    with pytest.raises(ValueError, match="packed context"):
        retrieve_exact_legacy_anchors(
            recorder,
            "query",
            RetrievalConfig(mode="memory"),
        )


class _FakeEmbedding:
    def __init__(self, calls, **kwargs) -> None:
        calls.append(("embedding", kwargs))
        self.closed = False

    @property
    def dim(self) -> int:
        return DEFAULT_MODEL_DIM

    def close(self) -> None:
        self.closed = True


def _fake_factories(calls: list[tuple]) -> DiffuseLongMemEvalRuntimeFactories:
    def embedding(**kwargs):
        return _FakeEmbedding(calls, **kwargs)

    def condenser(**kwargs):
        calls.append(("condenser", kwargs))
        return SimpleNamespace(marker="empty-condenser")

    def encoder(model_dir, **kwargs):
        value = SimpleNamespace(model_dir=model_dir, **kwargs)
        calls.append(("encoder", model_dir, kwargs))
        return value

    def linker(encoder_value, **kwargs):
        value = SimpleNamespace(encoder=encoder_value, **kwargs)
        calls.append(("linker", encoder_value, kwargs))
        return value

    def scorer(linker_value, **kwargs):
        value = SimpleNamespace(linker=linker_value, **kwargs)
        calls.append(("scorer", linker_value, kwargs))
        return value

    def reranker(linker_value, **kwargs):
        value = SimpleNamespace(linker=linker_value, **kwargs)
        calls.append(("reranker", linker_value, kwargs))
        return value

    def preflight(device, required):
        calls.append(("preflight", device, required))
        return ResidencyPreflightObservation(
            policy="fake-resident-preflight-v1",
            device=device,
            required_free_bytes=required,
            observed_free_bytes=required + 1024,
            observed_total_bytes=8 * 1024**3,
            embedding_released_before_qwen_load=False,
        )

    return DiffuseLongMemEvalRuntimeFactories(
        embedding=embedding,
        condenser=condenser,
        qwen_encoder=encoder,
        qwen_linker=linker,
        qwen_scorer=scorer,
        qwen_reranker=reranker,
        resident_preflight=preflight,
    )


def test_binding_pins_bge_and_reuses_one_qwen_linker_for_all_controls(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    retrieval = RetrievalConfig(
        mode="hybrid_graph",
        source_local_search=True,
        qwen_rerank=True,
        qwen_rerank_candidate_pool=37,
        qwen_rerank_slots=5,
        qwen_rerank_group_size=6,
        qwen_rerank_beam_per_group=2,
        qwen_rerank_candidate_tokens=45,
        qwen_rerank_query_tokens=67,
        qwen_rerank_score_weight=0.27,
        qwen_rerank_prefix_layers=2,
        qwen_rerank_attention_layer=1,
        qwen_rerank_max_workspace_tokens=2048,
    )
    runtime = DiffuseLongMemEvalRuntimeConfig(
        qwen_model_dir=tmp_path / "qwen",
    )
    binding = build_diffuse_longmemeval_execution_binding(
        config=_eval_config(retrieval),
        runtime=runtime,
        factories=_fake_factories(calls),
    )

    assert calls[0] == (
        "embedding",
        {
            "model_name": DEFAULT_MODEL_NAME,
            "model_revision": DEFAULT_MODEL_REVISION,
            "device": "cuda",
            "batch_size": 32,
            "verify_checkpoint": True,
        },
    )
    assert binding.embedding_identity == {
        "backend": "sentence-transformers.encode-v1",
        "model_id": DEFAULT_MODEL_NAME,
        "model_revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "dimension": DEFAULT_MODEL_DIM,
        "device": "cuda",
        "batch_size": 32,
        "normalize_embeddings": False,
        "output_dtype": "float32",
    }
    assert not binding.runtime_binding_certified

    empty = binding.new_condenser(tmp_path / "store")
    assert empty.marker == "empty-condenser"
    condenser_kwargs = next(item[1] for item in calls if item[0] == "condenser")
    assert condenser_kwargs == {
        "data_dir": tmp_path / "store",
        "model_name": DEFAULT_MODEL_NAME,
        "chunker_min_tokens": 3,
        "chunker_max_tokens": 47,
        "device": "cuda",
        "auto_extract": False,
        "embedder": binding.embedder,
        "persist_index_on_close": True,
    }

    observation = binding._resident_preflight()
    owned = binding.ensure_qwen_runtime()

    assert observation.required_free_bytes == 3072 * 1024 * 1024
    encoder_call = next(item for item in calls if item[0] == "encoder")
    assert encoder_call[1] == tmp_path / "qwen"
    assert encoder_call[2] == {
        "layers": 2,
        "device": "cuda",
        "dtype": "float16",
        "model_id": QWEN_MODEL_ID,
        "model_revision": QWEN_MODEL_REVISION,
        "expected_checkpoint_sha256": expected_prefix_checkpoint_sha256(2),
    }
    linker_kwargs = next(item[2] for item in calls if item[0] == "linker")
    assert linker_kwargs == {
        "layer": 1,
        "cav_bank": None,
        "max_candidates": 8,
        "max_workspace_tokens": 2048,
    }
    scorer_call = next(item for item in calls if item[0] == "scorer")
    reranker_call = next(item for item in calls if item[0] == "reranker")
    assert scorer_call[1] is owned.linker
    assert reranker_call[1] is owned.linker
    assert reranker_call[2] == {
        "candidate_pool": 37,
        "qwen_slots": 5,
        "group_size": 6,
        "beam_per_group": 2,
        "candidate_tokens": 45,
        "query_tokens": 67,
        "score_weight": 0.27,
        "association_artifact": None,
    }
    representative = binding.representative_policy_factory("artifact-1")
    assert representative.max_input_sources == 64
    assert representative.max_source_groups == 64
    assert representative.group_size == 8
    assert representative.representative_tokens == 96
    assert representative.query_tokens == 96


def test_binding_identity_is_cross_root_and_factory_sensitive(tmp_path: Path) -> None:
    config = _eval_config()
    first = build_diffuse_longmemeval_execution_binding(
        config=config,
        runtime=DiffuseLongMemEvalRuntimeConfig(
            qwen_model_dir=tmp_path / "checkout-a" / "qwen"
        ),
    )
    second = build_diffuse_longmemeval_execution_binding(
        config=config,
        runtime=DiffuseLongMemEvalRuntimeConfig(
            qwen_model_dir=tmp_path / "checkout-b" / "qwen"
        ),
    )

    assert first.runtime_binding_certified
    assert second.runtime_binding_certified
    assert first.binding_sha256 == second.binding_sha256

    class EmbeddingA(_FakeEmbedding):
        pass

    class EmbeddingB(_FakeEmbedding):
        pass

    calls_a: list[tuple] = []
    calls_b: list[tuple] = []
    factories_a = replace(
        _fake_factories(calls_a),
        embedding=lambda **kwargs: EmbeddingA(calls_a, **kwargs),
    )
    factories_b = replace(
        _fake_factories(calls_b),
        embedding=lambda **kwargs: EmbeddingB(calls_b, **kwargs),
    )
    injected_a = build_diffuse_longmemeval_execution_binding(
        config=config,
        runtime=DiffuseLongMemEvalRuntimeConfig(qwen_model_dir=tmp_path / "qwen"),
        factories=factories_a,
    )
    injected_b = build_diffuse_longmemeval_execution_binding(
        config=config,
        runtime=DiffuseLongMemEvalRuntimeConfig(qwen_model_dir=tmp_path / "qwen"),
        factories=factories_b,
    )

    assert not injected_a.runtime_binding_certified
    assert not injected_b.runtime_binding_certified
    assert injected_a.binding_sha256 != injected_b.binding_sha256


def test_callable_identity_ignores_checkout_filename_recursively() -> None:
    def fixture(value: int) -> tuple[int, ...]:
        return tuple(item + 1 for item in range(value))

    def relocate(code: CodeType) -> CodeType:
        return code.replace(
            co_filename="Z:/other-checkout/runtime.py",
            co_consts=tuple(
                relocate(item) if isinstance(item, CodeType) else item
                for item in code.co_consts
            ),
        )

    cloned_code = relocate(fixture.__code__)
    cloned = FunctionType(cloned_code, fixture.__globals__, fixture.__name__)
    cloned.__qualname__ = fixture.__qualname__
    cloned.__module__ = fixture.__module__

    assert runtime_module._callable_identity(fixture) == (
        runtime_module._callable_identity(cloned)
    )


def _anchor() -> RetrievalResult:
    turn = Turn(
        turn_id="turn-a",
        role="user",
        text="Direct alpha evidence.",
        source_id="source-a",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    chunk = Chunk(
        chunk_id="chunk-a",
        turn_id=turn.turn_id,
        text=turn.text,
        start_char=0,
        end_char=len(turn.text),
        token_count=4,
        embedding=[1.0, 0.0],
        lexical_weights={"direct": 1.0},
    )
    return RetrievalResult(
        chunk=chunk,
        turn=turn,
        score=0.9,
        route="dense",
    )


class _StagedSourceFake:
    def __init__(self, anchor: RetrievalResult, snapshot: DiscourseSnapshot) -> None:
        self.anchor = anchor
        self.search_calls = 0
        self.lexical_calls = 0
        self._streams = (
            SourceChunkStream(
                source_id="source-a",
                content_chunk_ids=("chunk-a",),
                metadata_chunk_ids=(),
                first_ordinal=0,
                last_ordinal=0,
                stream_sha256="a" * 64,
            ),
            SourceChunkStream(
                source_id="source-b",
                content_chunk_ids=("chunk-b",),
                metadata_chunk_ids=(),
                first_ordinal=1,
                last_ordinal=1,
                stream_sha256="b" * 64,
            ),
        )
        self.discourse = SimpleNamespace(snapshot=lambda: snapshot)
        self.retriever = SimpleNamespace(source_tfisf_query=self._source_tfisf)

    def search(self, _query, **_kwargs):
        self.search_calls += 1
        return [self.anchor]

    def _source_tfisf(self, _query, *, k_sources):
        self.lexical_calls += 1
        assert k_sources == 2
        return [("source-b", 9.0)]

    def discourse_source_streams(self):
        return self._streams


def _staged_case():
    artifact_id = "artifact-v4"
    snapshot = DiscourseSnapshot(
        max_turn_ordinal=1,
        chunk_count=2,
        graph_revision=1,
        schema_version=11,
        artifact_ids=(artifact_id,),
        source_revision=2,
        graph_content_revision=1,
        source_content_sha256="c" * 64,
        graph_content_sha256="d" * 64,
    )
    fake = _StagedSourceFake(_anchor(), snapshot)
    retrieval = RetrievalConfig(mode="dense", k=1)
    query = "Where is the beta evidence?"
    return artifact_id, fake, retrieval, query


def test_staged_source_scope_can_recover_a_source_missing_from_anchors() -> None:
    artifact_id, fake, retrieval, query = _staged_case()
    frozen = freeze_legacy_query_inputs(fake, (query,), retrieval)
    provider = FrozenLegacyDiffuseInputProvider(frozen, max_sources=64)

    candidates = provider(
        fake,
        query=query,
        retrieval=retrieval,
        artifact_id=artifact_id,
    )

    assert fake.search_calls == 1
    assert fake.lexical_calls == 1
    assert [item.turn.source_id for item in candidates.anchors] == ["source-a"]
    assert candidates.source_candidate_scope is not None
    assert candidates.source_candidate_scope.universe_source_ids == (
        "source-a",
        "source-b",
    )
    assert {item.source_id for item in candidates.source_candidates} == {
        "source-a",
        "source-b",
    }
    beta = next(
        item for item in candidates.source_candidates if item.source_id == "source-b"
    )
    assert "source_tfisf" in beta.route
    assert candidates.source_candidate_scope.selected_scope_exhaustive


def test_frozen_provider_rejects_mutation_before_construction() -> None:
    _, fake, retrieval, query = _staged_case()
    frozen = freeze_legacy_query_inputs(fake, (query,), retrieval)
    frozen[0].anchors[0].chunk.lexical_weights["direct"] = 0.25  # type: ignore[index]

    with pytest.raises(ValueError, match="receipt does not match"):
        FrozenLegacyDiffuseInputProvider(frozen)


def test_frozen_provider_detaches_original_anchor_graph() -> None:
    artifact_id, fake, retrieval, query = _staged_case()
    frozen = freeze_legacy_query_inputs(fake, (query,), retrieval)
    provider = FrozenLegacyDiffuseInputProvider(frozen)
    identity = provider.analysis_identity_payload()
    original = frozen[0].anchors[0]
    original.score = 0.1
    original.chunk.embedding[0] = -1.0  # type: ignore[index]
    original.chunk.lexical_weights["direct"] = 0.25  # type: ignore[index]

    candidates = provider(
        fake,
        query=query,
        retrieval=retrieval,
        artifact_id=artifact_id,
    )

    assert candidates.anchors[0].score == 0.9
    assert candidates.anchors[0].chunk.embedding == [1.0, 0.0]
    assert candidates.anchors[0].chunk.lexical_weights == {"direct": 1.0}
    assert provider.analysis_identity_payload() == identity


def test_frozen_provider_materializes_a_fresh_anchor_graph_per_call() -> None:
    artifact_id, fake, retrieval, query = _staged_case()
    frozen = freeze_legacy_query_inputs(fake, (query,), retrieval)
    provider = FrozenLegacyDiffuseInputProvider(frozen)

    first = provider(
        fake,
        query=query,
        retrieval=retrieval,
        artifact_id=artifact_id,
    )
    first_anchor = first.anchors[0]
    first_anchor.score = 0.1
    first_anchor.chunk.embedding.append(7.0)  # type: ignore[union-attr]
    first_anchor.chunk.lexical_weights["injected"] = 2.0  # type: ignore[index]
    second = provider(
        fake,
        query=query,
        retrieval=retrieval,
        artifact_id=artifact_id,
    )
    second_anchor = second.anchors[0]

    assert second_anchor.score == 0.9
    assert second_anchor.chunk.embedding == [1.0, 0.0]
    assert second_anchor.chunk.lexical_weights == {"direct": 1.0}
    assert second_anchor is not first_anchor
    assert second_anchor.chunk is not first_anchor.chunk
    assert second_anchor.turn is not first_anchor.turn
