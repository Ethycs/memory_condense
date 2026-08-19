from __future__ import annotations

import gc
import subprocess
import sys
import weakref
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest

import memory_condense.search.episodes.builder as builder_module
import memory_condense.search.episodes.qwen_episode_signal as qwen_signal_module
import memory_condense.search.episodes.surprise as surprise_module
import memory_condense.application.discourse_workflow as discourse_workflow_module
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.domain.discourse import (
    DiscourseArtifact,
    EvidenceSpan,
    identity_sha256,
    quote_sha256,
)
from memory_condense.search.episodes import (
    BoundaryProposal,
    CohesionBoundaryRefiner,
    EpisodeBuilder,
    LexicalEmbeddingChangeScorer,
    QwenAttentionHeadSurpriseScorer,
    ScoredSurpriseSequence,
)
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder


class _FakeEmbedder:
    @property
    def dim(self) -> int:
        return 4

    def embed_query(self, _query: str) -> np.ndarray:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)

    def embed_chunks(self, chunks: list[Any]) -> list[Any]:
        vector = self.embed_query("").tolist()
        return [
            chunk.model_copy(update={"embedding": vector})
            for chunk in chunks
        ]


class _LifetimeMarker:
    __slots__ = ("label", "__weakref__")

    def __init__(self, label: str) -> None:
        self.label = label


class _TransportTensor:
    __slots__ = ("array", "__weakref__")

    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self) -> _TransportTensor:
        return self

    def float(self) -> _TransportTensor:
        return self

    def cpu(self) -> _TransportTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class _FakePrefixLinker:
    def __init__(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        max_candidates: int = 3,
        max_workspace_tokens: int = 64,
        consume: int | None = None,
        mode: str = "valid",
        reported_workspace_tokens: int | None = None,
        track_lifetimes: bool = False,
        return_kv_cache: bool = False,
        retained_state: Any = 0,
    ) -> None:
        self.vectors = tuple(tuple(row) for row in vectors)
        self.max_candidates = max_candidates
        self.max_workspace_tokens = max_workspace_tokens
        self.consume = consume
        self.mode = mode
        self.reported_workspace_tokens = reported_workspace_tokens
        self.track_lifetimes = track_lifetimes
        self.return_kv_cache = return_kv_cache
        self.retained_state = retained_state
        self.layer = 1
        self.head_vote_k = 2
        self.cav_bank = None
        self.encoder = SimpleNamespace(
            checkpoint_identity=SimpleNamespace(
                model_id="Qwen/Qwen3-8B",
                model_revision="revision-1",
                checkpoint_sha256="a" * 64,
            ),
            device="cuda:0",
            dtype_name="float16",
            layers=2,
        )
        self.submitted_batch_sizes: list[int] = []
        self.accepted_indices: list[int] = []
        self.probes: list[str] = []
        self.lifetime_refs: list[weakref.ReferenceType[Any]] = []

    @staticmethod
    def _candidate_index(candidate: Any) -> int:
        return int(str(candidate.episode_id).split("-", 2)[1])

    def _transport(self, index: int) -> Any:
        values = self.vectors[index]
        if self.mode == "nonfinite_signature" and index == 1:
            values = (float("nan"), *values[1:])
        if self.mode == "wrong_width" and index == 1:
            values = (*values, 0.5)
        array = np.asarray(values, dtype=np.float32).copy()
        if self.mode == "matrix_signature" and index == 1:
            array = array.reshape(1, -1)
        if not self.track_lifetimes:
            return array
        tensor = _TransportTensor(array)
        self.lifetime_refs.extend((weakref.ref(array), weakref.ref(tensor)))
        return tensor

    def inspect_coverage(self, probe: str, candidates: Sequence[Any]) -> Any:
        self.probes.append(probe)
        self.submitted_batch_sizes.append(len(candidates))
        consumed = min(
            len(candidates),
            self.consume if self.consume is not None else len(candidates),
        )
        accepted = list(candidates[:consumed])
        self.accepted_indices.extend(self._candidate_index(row) for row in accepted)

        hits = []
        for candidate in accepted:
            index = self._candidate_index(candidate)
            activation = _LifetimeMarker(f"activation-{index}")
            if self.track_lifetimes:
                self.lifetime_refs.append(weakref.ref(activation))
            hits.append(
                SimpleNamespace(
                    episode_id=candidate.episode_id,
                    qk_score=0.5,
                    ov_transport=1.0,
                    transport_signature=self._transport(index),
                    _transient_activation=activation,
                )
            )

        if self.mode == "missing" and hits:
            hits.pop()
        elif self.mode == "duplicate" and len(hits) >= 2:
            duplicate = hits[0]
            hits[1] = SimpleNamespace(
                episode_id=duplicate.episode_id,
                qk_score=duplicate.qk_score,
                ov_transport=duplicate.ov_transport,
                transport_signature=duplicate.transport_signature,
            )

        kv_cache = _LifetimeMarker("kv-cache")
        if self.track_lifetimes:
            self.lifetime_refs.append(weakref.ref(kv_cache))
        workspace_tokens = (
            self.reported_workspace_tokens
            if self.reported_workspace_tokens is not None
            else 10 + consumed
        )
        return SimpleNamespace(
            hits=tuple(hits),
            workspace_candidates=consumed,
            workspace_tokens=workspace_tokens,
            passes=1,
            total_candidate_inspections=consumed,
            retained_transformer_state_bytes=self.retained_state,
            past_key_values=kv_cache if self.return_kv_cache else None,
            _transient_kv_cache=kv_cache,
        )


class _BoundaryAtTwo:
    requires_surprise_scores = True
    method = "test_injected_boundary"

    def __init__(self) -> None:
        self.observed_scores: tuple[float, ...] | None = None

    def detect(self, surprises: Sequence[float]) -> tuple[BoundaryProposal, ...]:
        self.observed_scores = tuple(float(value) for value in surprises)
        return (
            BoundaryProposal(
                position=2,
                score=self.observed_scores[2],
                threshold=-1.0,
            ),
        )


class _RecordingSequenceScorer:
    def __init__(self, delegate: QwenAttentionHeadSurpriseScorer) -> None:
        self.delegate = delegate
        self.last_signal: ScoredSurpriseSequence | None = None
        self.request_token_ids = (1, 2, 3)

    def score_sequence(
        self,
        texts: Sequence[str],
        *,
        embeddings: Sequence[Sequence[float] | None] | None = None,
    ) -> ScoredSurpriseSequence:
        self.last_signal = self.delegate.score_sequence(
            texts,
            embeddings=embeddings,
        )
        return self.last_signal


def _texts(count: int) -> tuple[str, ...]:
    return tuple(f"source span {index}" for index in range(count))


def _spans(texts: Sequence[str]) -> tuple[EvidenceSpan, ...]:
    return tuple(
        EvidenceSpan(
            chunk_id=f"chunk-{index}",
            start_char=0,
            end_char=len(text),
            quote_sha256=quote_sha256(text),
            ordinal=index,
            source_id="source-a",
            turn_id=f"turn-{index}",
            role="user" if index % 2 == 0 else "assistant",
        )
        for index, text in enumerate(texts)
    )


def test_cold_episode_import_stays_out_of_eval_and_provider_modules() -> None:
    code = (
        "import sys; import memory_condense.search.episodes; "
        "print(sorted({'litellm','anthropic','openai','google','cohere','mistralai'} "
        "& set(sys.modules))); "
        "print(any(name.startswith('memory_condense.eval') for name in sys.modules)); "
        "print('memory_condense.associations.qwen_memory_linker' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.splitlines() == ["[]", "False", "False"]


def test_attention_head_scorer_drains_every_bounded_partial_batch() -> None:
    vectors = (
        (1.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (-1.0, 0.0),
    )
    linker = _FakePrefixLinker(vectors, max_candidates=3, consume=1)
    signal = QwenAttentionHeadSurpriseScorer(
        linker,
        max_spans=5,
    ).score_sequence(_texts(5))

    assert linker.submitted_batch_sizes == [3, 3, 3, 2, 1]
    assert linker.accepted_indices == [0, 1, 2, 3, 4]
    assert len(set(linker.probes)) == 1
    assert signal.scores == (0.0, 0.0, 0.5, 0.0, 0.5)
    assert signal.receipt.workspace_batches == 5
    assert signal.receipt.forward_passes == 5
    assert signal.receipt.inspected_spans == 5
    assert signal.receipt.max_workspace_candidates == 1
    assert signal.receipt.max_workspace_tokens == 11
    assert signal.receipt.total_workspace_tokens == 55
    assert signal.receipt.similarity_scalar_pairs == 10
    assert signal.receipt.retained_signal_transformer_state_bytes == 0


def test_attention_head_receipt_binds_exact_inputs_scores_and_matrix() -> None:
    texts = _texts(4)
    linker = _FakePrefixLinker(
        ((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
    )
    scorer = QwenAttentionHeadSurpriseScorer(linker, max_spans=4)
    signal = scorer.score_sequence(texts)
    replay = scorer.score_sequence(texts)
    receipt = signal.receipt

    assert signal == replay
    assert receipt.input_sequence_sha256 == identity_sha256(
        {"quote_sha256": [quote_sha256(text) for text in texts]}
    )
    assert receipt.score_sequence_sha256 == identity_sha256(
        {"scores": [float(value) for value in signal.scores]}
    )
    assert receipt.similarity_matrix_sha256 == identity_sha256(
        {
            "similarities": [
                [float(value) for value in row]
                for row in signal.similarities
            ]
        }
    )
    assert receipt.receipt_sha256 == identity_sha256(
        receipt.identity_payload(include_receipt=False)
    )
    assert receipt.checkpoint_sha256 == "a" * 64
    assert receipt.prefix_layers == 2
    assert receipt.attention_layer == 1
    assert receipt.owned_runtime_binding is False
    assert len(receipt.implementation_sha256) == 64

    signal.validate_inputs(texts)
    with pytest.raises(ValueError, match="ordered input texts"):
        signal.validate_inputs(tuple(reversed(texts)))
    with pytest.raises(ValueError, match="receipt"):
        replace(receipt, model_revision="tampered-revision")
    with pytest.raises(ValueError, match="receipt"):
        replace(receipt, owned_runtime_binding=True)

    tampered_scores = list(signal.scores)
    tampered_scores[1] = 0.25
    with pytest.raises(ValueError, match="scores do not match their receipt"):
        ScoredSurpriseSequence(
            tuple(tampered_scores),
            signal.similarities,
            receipt,
        )

    tampered_matrix = [list(row) for row in signal.similarities]
    tampered_matrix[0][2] = tampered_matrix[2][0] = 0.25
    with pytest.raises(ValueError, match="similarities do not match their receipt"):
        ScoredSurpriseSequence(
            signal.scores,
            tuple(tuple(row) for row in tampered_matrix),
            receipt,
        )


def test_owned_runtime_binding_requires_exact_unshadowed_owned_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = object.__new__(Qwen3PrefixEncoder)
    encoder.__dict__.update(
        {
            "model_dir": None,
            "layers": 2,
            "model_id": "Qwen/Qwen3-8B",
            "model_revision": "revision-1",
            "checkpoint_identity": None,
            "checkpoint_sha256": "a" * 64,
            "_torch": None,
            "_apply_rotary_pos_emb": None,
            "device": "cpu",
            "dtype": None,
            "dtype_name": "float32",
            "config": None,
            "model": None,
            "tokenizer": None,
            "loaded_parameter_names": frozenset(),
        }
    )
    linker = object.__new__(QwenMemoryLinker)
    linker.__dict__.update(
        {
            "encoder": encoder,
            "layer": 1,
            "cav_bank": None,
            "max_candidates": 2,
            "max_workspace_tokens": 64,
            "max_neighbors_per_episode": 4,
            "head_vote_k": 2,
        }
    )

    assert surprise_module._owned_qwen_runtime_binding(linker) is True
    owned_implementation = surprise_module._attention_head_implementation_sha256(
        linker
    )

    linker.inspect_coverage = lambda *_args, **_kwargs: None
    assert surprise_module._owned_qwen_runtime_binding(linker) is False

    attached_state = object.__new__(QwenMemoryLinker)
    attached_state.__dict__.update(
        {
            key: value
            for key, value in linker.__dict__.items()
            if key != "inspect_coverage"
        }
    )
    attached_state.request_token_ids = (1, 2, 3)
    assert surprise_module._owned_qwen_runtime_binding(attached_state) is False
    assert discourse_workflow_module._episode_retention_attestation(
        surprise_scores=None,
        surprise_scorer=object(),
        builder=EpisodeBuilder(min_size=1, max_size=1),
        build=SimpleNamespace(
            surprise_signal_receipt=SimpleNamespace(
                owned_runtime_binding=(
                    surprise_module._owned_qwen_runtime_binding(attached_state)
                )
            )
        ),
    ) is None

    class _InjectedLinker(QwenMemoryLinker):
        pass

    injected = object.__new__(_InjectedLinker)
    injected.encoder = encoder
    assert surprise_module._owned_qwen_runtime_binding(injected) is False

    class_level = object.__new__(QwenMemoryLinker)
    class_level.__dict__.update(attached_state.__dict__)
    del class_level.request_token_ids
    monkeypatch.setattr(
        QwenMemoryLinker,
        "inspect_coverage",
        lambda _self, *_args, **_kwargs: None,
    )
    assert surprise_module._owned_qwen_runtime_binding(class_level) is False
    assert (
        surprise_module._attention_head_implementation_sha256(class_level)
        != owned_implementation
    )


def test_receipt_snapshots_mutable_tokenizer_proxy_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_identity = {"schema": "proxy-v1", "vocabulary_sha256": "b" * 64}
    monkeypatch.setattr(
        surprise_module,
        "tokenizer_proxy_identity",
        lambda: tokenizer_identity,
    )
    signal = QwenAttentionHeadSurpriseScorer(
        _FakePrefixLinker(((1.0, 0.0),)),
        max_spans=1,
    ).score_sequence(_texts(1))
    expected = identity_sha256(dict(tokenizer_identity))

    tokenizer_identity["schema"] = "mutated-after-receipt"

    assert signal.receipt.tokenizer_proxy_sha256 == expected
    assert signal.receipt.receipt_sha256 == identity_sha256(
        signal.receipt.identity_payload(include_receipt=False)
    )


def test_builder_refines_with_the_same_head_matrix_not_lexical_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = _texts(6)
    vectors = (
        (1.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    )
    linker = _FakePrefixLinker(vectors)
    scorer = QwenAttentionHeadSurpriseScorer(linker, max_spans=6)
    detector = _BoundaryAtTwo()

    def forbid_lexical_fallback(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("head signal unexpectedly used lexical similarity")

    monkeypatch.setattr(
        builder_module,
        "_similarity_lookup",
        forbid_lexical_fallback,
    )
    result = EpisodeBuilder(
        min_size=1,
        max_size=6,
        detector=detector,
        refiner=CohesionBoundaryRefiner(
            window=2,
            max_nodes=6,
            max_degree=5,
        ),
    ).build(
        source_id="source-a",
        artifact_id="artifact-head-signal",
        spans=_spans(texts),
        texts=texts,
        surprise_scorer=scorer,
    )

    assert detector.observed_scores == (0.0, 0.0, 0.0, 0.5, 0.0, 0.0)
    assert result.initial_boundaries[0].position == 2
    assert result.refined_boundaries[0].position == 3
    assert result.refined_boundaries[0].cohesion is not None
    assert [len(episode.evidence) for episode in result.episodes] == [3, 3]
    assert result.surprise_signal_receipt is not None
    assert result.surprise_signal_receipt.input_spans == 6
    with pytest.raises(ValueError, match="ordered episode evidence"):
        replace(result, episodes=tuple(reversed(result.episodes)))


def test_discourse_publication_carries_the_exact_qwen_receipt(tmp_path: Any) -> None:
    vectors = ((1.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    recording_scorer = _RecordingSequenceScorer(
        QwenAttentionHeadSurpriseScorer(
            _FakePrefixLinker(vectors),
            max_spans=3,
        )
    )
    artifact = DiscourseArtifact.create(
        kind="qwen-head-episode-test",
        implementation_sha256="b" * 64,
        policy={"episode_boundary": "qwen-head-signal"},
    )

    with MemoryCondenser(
        data_dir=tmp_path / "qwen-receipt-workflow",
        embedder=_FakeEmbedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
        persist_index_on_close=False,
    ) as condenser:
        chunks = [
            condenser.ingest(
                "user" if index % 2 == 0 else "assistant",
                text,
                source_id="qwen-receipt-thread",
            )[1][0]
            for index, text in enumerate(_texts(3))
        ]
        publication = condenser.build_and_publish_discourse_episodes(
            artifact,
            tuple(chunk.chunk_id for chunk in reversed(chunks)),
            builder=EpisodeBuilder(min_size=1, max_size=3),
            surprise_scorer=recording_scorer,
            representative_limit=0,
        )

    assert recording_scorer.last_signal is not None
    assert (
        publication.build.surprise_signal_receipt
        is not recording_scorer.last_signal.receipt
    )
    assert publication.build.surprise_signal_receipt is not None
    assert (
        publication.build.surprise_signal_receipt.evidence_sequence_sha256
        is not None
    )
    assert recording_scorer.request_token_ids == (1, 2, 3)
    assert publication.returned_signal_transformer_state_bytes is None


def test_qwen_attestation_failure_precedes_durable_publication(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = QwenAttentionHeadSurpriseScorer(
        _FakePrefixLinker(((1.0, 0.0), (0.0, 1.0))),
        max_spans=2,
    )
    artifact = DiscourseArtifact.create(
        kind="qwen-attestation-atomicity-test",
        implementation_sha256="c" * 64,
        policy={"episode_boundary": "qwen-head-signal"},
    )
    publish_calls: list[object] = []

    with MemoryCondenser(
        data_dir=tmp_path / "qwen-attestation-atomicity",
        embedder=_FakeEmbedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
        persist_index_on_close=False,
    ) as condenser:
        chunks = [
            condenser.ingest(
                "user",
                text,
                source_id="qwen-attestation-thread",
            )[1][0]
            for text in _texts(2)
        ]

        def forbidden_publish(*_args: Any, **_kwargs: Any) -> Any:
            publish_calls.append(object())
            raise AssertionError("publication ran before signal attestation")

        def bomb_matcher(*_args: Any, **_kwargs: Any) -> bool:
            raise RuntimeError("attestation bomb")

        monkeypatch.setattr(condenser._discourse, "publish", forbidden_publish)
        monkeypatch.setattr(
            discourse_workflow_module,
            "_owned_qwen_receipt_matches",
            bomb_matcher,
        )
        with pytest.raises(RuntimeError, match="attestation bomb"):
            condenser.build_and_publish_discourse_episodes(
                artifact,
                tuple(chunk.chunk_id for chunk in chunks),
                builder=EpisodeBuilder(min_size=1, max_size=2),
                surprise_scorer=scorer,
                representative_limit=0,
            )

    assert publish_calls == []


def test_episode_publication_retention_attestation_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_receipt = SimpleNamespace(surprise_signal_receipt=None)
    exact_builder = EpisodeBuilder(min_size=1, max_size=1)
    assert discourse_workflow_module._episode_retention_attestation(
        surprise_scores=None,
        surprise_scorer=None,
        builder=exact_builder,
        build=no_receipt,
    ) == 0
    assert discourse_workflow_module._episode_retention_attestation(
        surprise_scores=(0.0,),
        surprise_scorer=None,
        builder=exact_builder,
        build=no_receipt,
    ) == 0
    assert discourse_workflow_module._episode_retention_attestation(
        surprise_scores=None,
        surprise_scorer=LexicalEmbeddingChangeScorer(),
        builder=exact_builder,
        build=no_receipt,
    ) == 0
    assert discourse_workflow_module._episode_retention_attestation(
        surprise_scores=None,
        surprise_scorer=object(),
        builder=exact_builder,
        build=no_receipt,
    ) is None

    owned_receipt = SimpleNamespace(owned_runtime_binding=True)
    exact_qwen_scorer = object.__new__(QwenAttentionHeadSurpriseScorer)
    monkeypatch.setattr(
        discourse_workflow_module,
        "_owned_qwen_receipt_matches",
        lambda scorer, receipt: (
            scorer is exact_qwen_scorer and receipt is owned_receipt
        ),
    )
    assert discourse_workflow_module._episode_retention_attestation(
        surprise_scores=None,
        surprise_scorer=exact_qwen_scorer,
        builder=exact_builder,
        build=SimpleNamespace(surprise_signal_receipt=owned_receipt),
    ) == 0

    assert discourse_workflow_module._episode_retention_attestation(
        surprise_scores=None,
        surprise_scorer=object(),
        builder=exact_builder,
        build=SimpleNamespace(surprise_signal_receipt=owned_receipt),
    ) is None

    monkeypatch.setattr(
        discourse_workflow_module,
        "_owned_qwen_receipt_matches",
        lambda _scorer, _receipt: False,
    )
    with pytest.raises(RuntimeError, match="no longer matches"):
        discourse_workflow_module._episode_retention_attestation(
            surprise_scores=None,
            surprise_scorer=exact_qwen_scorer,
            builder=exact_builder,
            build=SimpleNamespace(surprise_signal_receipt=owned_receipt),
        )

    class _CustomBuilder(EpisodeBuilder):
        pass

    custom_builder = _CustomBuilder(min_size=1, max_size=1)
    assert discourse_workflow_module._episode_retention_attestation(
        surprise_scores=None,
        surprise_scorer=None,
        builder=custom_builder,
        build=no_receipt,
    ) is None
    assert discourse_workflow_module._episode_retention_attestation(
        surprise_scores=(0.0,),
        surprise_scorer=None,
        builder=custom_builder,
        build=no_receipt,
    ) is None


def test_callable_digest_is_checkout_path_independent_but_code_sensitive() -> None:
    def sample(value: int) -> int:
        def nested(offset: int) -> int:
            return value + offset

        return nested(1)

    code = sample.__code__
    relocated = code.replace(co_filename="OTHER_ROOT/package/module.py")
    stable_name = "tests.test_qwen_episode_signals.sample"

    assert qwen_signal_module._canonical_callable_code(
        code,
        stable_filename=stable_name,
    ) == qwen_signal_module._canonical_callable_code(
        relocated,
        stable_filename=stable_name,
    )

    changed = code.replace(co_consts=(*code.co_consts, "changed-constant"))
    assert qwen_signal_module._canonical_callable_code(
        code,
        stable_filename=stable_name,
    ) != qwen_signal_module._canonical_callable_code(
        changed,
        stable_filename=stable_name,
    )


def test_transport_normalization_is_overflow_safe() -> None:
    raw = np.asarray((3e38, 3e38), dtype=np.float32)

    normalized = surprise_module._normalized_transport_signature(
        raw,
        max_dimension=2,
    )

    assert normalized is not None
    assert np.isfinite(normalized).all()
    assert np.linalg.norm(normalized.astype(np.float64)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing", "incomplete transport hits"),
        ("duplicate", "incomplete transport hits"),
        ("nonfinite_signature", "omitted a finite OV signature"),
        ("wrong_width", "inconsistent width"),
        ("matrix_signature", "one-dimensional"),
    ],
)
def test_attention_head_scorer_rejects_invalid_transport_signatures(
    mode: str,
    message: str,
) -> None:
    scorer = QwenAttentionHeadSurpriseScorer(
        _FakePrefixLinker(
            ((1.0, 0.0), (0.0, 1.0)),
            max_candidates=2,
            mode=mode,
        ),
        max_spans=2,
    )

    with pytest.raises((RuntimeError, ValueError), match=message):
        scorer.score_sequence(_texts(2))


def test_attention_head_scorer_enforces_sequence_and_workspace_caps() -> None:
    span_limited_linker = _FakePrefixLinker(
        ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
    )
    span_limited = QwenAttentionHeadSurpriseScorer(
        span_limited_linker,
        max_spans=2,
    )
    with pytest.raises(MemoryError, match="above the hard cap"):
        span_limited.score_sequence(_texts(3))
    assert span_limited_linker.submitted_batch_sizes == []

    workspace_limited = QwenAttentionHeadSurpriseScorer(
        _FakePrefixLinker(
            ((1.0, 0.0),),
            max_workspace_tokens=16,
            reported_workspace_tokens=17,
        ),
        max_spans=1,
    )
    with pytest.raises(RuntimeError, match="workspace token cap"):
        workspace_limited.score_sequence(_texts(1))

    dimension_limited = QwenAttentionHeadSurpriseScorer(
        _FakePrefixLinker(((1.0, 0.0, 0.0),)),
        max_spans=1,
        max_transport_dimension=2,
    )
    with pytest.raises(MemoryError, match="transport-dimension cap"):
        dimension_limited.score_sequence(_texts(1))

    fractional_retention = QwenAttentionHeadSurpriseScorer(
        _FakePrefixLinker(((1.0, 0.0),), retained_state=0.5),
        max_spans=1,
    )
    with pytest.raises(ValueError, match="integer"):
        fractional_retention.score_sequence(_texts(1))


def test_qwen_signal_uses_only_lossless_source_prefixes() -> None:
    assert surprise_module._lossless_proxy_prefix("A😀", 2) == "A"
    with pytest.raises(ValueError, match="complete source character"):
        surprise_module._lossless_proxy_prefix("😀", 1)

    linker = _FakePrefixLinker(((1.0, 0.0),))
    scorer = QwenAttentionHeadSurpriseScorer(
        linker,
        max_spans=1,
        span_token_cap=1,
    )
    with pytest.raises(ValueError, match="complete source character"):
        scorer.score_sequence(("😀",))
    assert linker.submitted_batch_sizes == []


def test_precomputed_and_scorer_inputs_are_mutually_exclusive() -> None:
    texts = _texts(1)
    linker = _FakePrefixLinker(((1.0, 0.0),))
    scorer = QwenAttentionHeadSurpriseScorer(linker, max_spans=1)

    with pytest.raises(ValueError, match="mutually exclusive"):
        EpisodeBuilder(min_size=1, max_size=1).build(
            source_id="source-a",
            artifact_id="artifact-exclusive-signal",
            spans=_spans(texts),
            texts=texts,
            surprise_scores=(0.0,),
            surprise_scorer=scorer,
        )
    assert linker.submitted_batch_sizes == []


def test_attention_head_scorer_releases_transient_state_and_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized_vector_refs: list[weakref.ReferenceType[np.ndarray]] = []
    original_normalize = surprise_module._normalized_transport_signature

    def track_normalized(
        value: Any,
        *,
        max_dimension: int,
    ) -> np.ndarray | None:
        normalized = original_normalize(
            value,
            max_dimension=max_dimension,
        )
        if normalized is not None:
            normalized_vector_refs.append(weakref.ref(normalized))
        return normalized

    monkeypatch.setattr(
        surprise_module,
        "_normalized_transport_signature",
        track_normalized,
    )
    linker = _FakePrefixLinker(
        ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
        max_candidates=2,
        track_lifetimes=True,
    )
    signal = QwenAttentionHeadSurpriseScorer(
        linker,
        max_spans=3,
    ).score_sequence(_texts(3))

    gc.collect()
    assert signal.receipt.retained_signal_transformer_state_bytes == 0
    assert normalized_vector_refs
    assert linker.lifetime_refs
    assert all(reference() is None for reference in normalized_vector_refs)
    assert all(reference() is None for reference in linker.lifetime_refs)

    kv_linker = _FakePrefixLinker(
        ((1.0, 0.0),),
        track_lifetimes=True,
        return_kv_cache=True,
    )
    with pytest.raises(RuntimeError, match="K/V cache"):
        QwenAttentionHeadSurpriseScorer(
            kv_linker,
            max_spans=1,
        ).score_sequence(_texts(1))
    gc.collect()
    assert kv_linker.lifetime_refs
    assert all(reference() is None for reference in kv_linker.lifetime_refs)
