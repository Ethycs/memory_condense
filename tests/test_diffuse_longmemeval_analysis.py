from __future__ import annotations

import re
import zlib
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

import memory_condense.eval.diffuse_longmemeval_analysis as analysis_module
import memory_condense.search.episodes.representative_retrieval as rep_module
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.associations.head_memory_models import (
    MemoryLinkHit,
    NestedMemoryInspection,
)
from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval.diffuse_compilation import (
    DiffuseCompilationPolicy,
    DiffuseCompilationReceipt,
)
from memory_condense.eval.diffuse_longmemeval_analysis import (
    DiffuseLongMemEvalArm,
    LegacyDiffuseCandidates,
    capture_legacy_diffuse_inputs,
    gold_blind_longmemeval_sample,
    ingest_gold_blind_sample_deterministically,
    matched_diffuse_boundary_arms,
    measure_diffuse_longmemeval_sample,
    run_diffuse_longmemeval_analysis,
    validate_matched_diffuse_retrieval_phases,
)
from memory_condense.eval.schemas import (
    ChunkerConfig,
    EvalConfig,
    RetrievalConfig,
)
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample
from memory_condense.modeling.qwen_prefix import (
    Qwen3PrefixEncoder,
    QwenPrefixCheckpointIdentity,
)
from memory_condense.search.episodes import (
    EpisodeRepresentativeRetrievalPolicy,
)


class _DeterministicEmbedder:
    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def dim(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            vector[zlib.crc32(token.encode("utf-8")) % self._dimension] += 1.0
        if not vector.any():
            vector[0] = 1.0
        return vector

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)

    def embed_chunks(self, chunks):
        return [
            chunk.model_copy(update={"embedding": self._vector(chunk.text).tolist()})
            for chunk in chunks
        ]


class _SelectEveryEpisodeLinker:
    """Provider-free nested-linker fixture with no model or retained state."""

    max_candidates = 8
    max_workspace_tokens = 1024
    layer = None
    head_vote_k = None
    encoder = None

    def inspect_nested(
        self,
        _source_text,
        candidate_groups,
        *,
        beam_per_group=2,
        top_k=4,
        score_mode="qk_ov",
    ):
        candidates = [item for group in candidate_groups for item in group]
        hits = tuple(
            MemoryLinkHit(
                episode_id=item.episode_id,
                qk_score=1.0 - index * 0.01,
                ov_transport=0.5 if score_mode == "qk_ov" else 0.0,
                head_weights=(1.0,),
            )
            for index, item in enumerate(candidates[:top_k])
        )
        return NestedMemoryInspection(
            hits=hits,
            passes=len(candidate_groups),
            max_workspace_candidates=max(
                (len(group) for group in candidate_groups),
                default=0,
            ),
            max_workspace_tokens=sum(len(item.text) for item in candidates),
            total_candidate_inspections=len(candidates),
        )


def test_owned_linker_torch_device_has_strict_json_analysis_identity() -> None:
    """Exercise the production identity shape without loading a model."""

    torch = pytest.importorskip("torch")
    encoder = object.__new__(Qwen3PrefixEncoder)
    encoder.__dict__.update(
        {
            "model_dir": None,
            "layers": 2,
            "model_id": "Qwen/Qwen3-8B",
            "model_revision": "revision-1",
            "checkpoint_identity": QwenPrefixCheckpointIdentity(
                model_id="Qwen/Qwen3-8B",
                model_revision="revision-1",
                checkpoint_sha256="a" * 64,
                verified_files=(),
            ),
            "checkpoint_sha256": "a" * 64,
            "_torch": torch,
            "_apply_rotary_pos_emb": None,
            "device": torch.device("cuda:0"),
            "dtype": torch.float32,
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
            "max_candidates": 8,
            "max_workspace_tokens": 2048,
            "max_neighbors_per_episode": 16,
            "head_vote_k": 4,
        }
    )

    payload = rep_module._linker_identity(linker)

    assert payload["owned_runtime_binding"] is True
    assert payload["device"] == "cuda:0"
    cuda_zero_sha256 = analysis_module._representative_linker_identity_sha256(
        linker
    )
    assert cuda_zero_sha256 == identity_sha256(payload)

    encoder.device = torch.device("cuda:1")
    assert (
        analysis_module._representative_linker_identity_sha256(linker)
        != cuda_zero_sha256
    )


def _sample(*, answer: str = "cobalt blue") -> BenchmarkSample:
    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    turns = [
        ("user", "Atlas architecture review opened with an alpha baseline."),
        ("assistant", "The alpha baseline kept only one direct dense anchor."),
        ("user", "Alpha follow-up recorded a prompt budget constraint."),
        ("user", "A separate beta session compared the diffuse dependencies."),
        ("assistant", "The final Atlas recommendation was cobalt blue."),
        ("user", "Beta closed by requiring exact source-span provenance."),
    ]
    return BenchmarkSample(
        sample_id="longmemeval-diffuse-analysis-fixture",
        turns=turns,
        turn_source_ids=["alpha"] * 3 + ["beta"] * 3,
        turn_created_at=[timestamp] * len(turns),
        questions=[
            BenchmarkQuestion(
                question_id="diffuse-q1",
                question="What was the final Atlas recommendation?",
                answer=answer,
                evidence_sources=["beta"],
                question_date="2025/02/01",
            )
        ],
    )


def _config() -> EvalConfig:
    return EvalConfig(
        chunker=ChunkerConfig(min_tokens=1, max_tokens=80),
        retrieval=RetrievalConfig(mode="dense", k=10, ef_search=50),
        max_prompt_tokens=4000,
    )


def _fixed_arm() -> DiffuseLongMemEvalArm:
    return DiffuseLongMemEvalArm(
        arm_id="fixed_interval",
        compilation=DiffuseCompilationPolicy(
            boundary_mode="fixed_interval",
            min_episode_size=1,
            max_episode_size=8,
            fixed_interval=8,
            representative_limit=1,
        ),
        max_context_tokens=2048,
        responder_output_token_reserve=128,
    )


def _representative_policy(artifact_id: str):
    return EpisodeRepresentativeRetrievalPolicy(
        artifact_id=artifact_id,
        max_input_sources=8,
        max_source_groups=2,
        max_episodes_per_source=8,
        max_total_episodes=16,
        max_representatives_per_episode=1,
        group_size=8,
        beam_per_group=1,
        top_k=4,
        representative_tokens=64,
        query_tokens=64,
    )


def _factory(embedder, calls=None):
    def create(data_dir, config):
        if calls is not None:
            calls.append((data_dir, config.model_dump(mode="json")))
        return MemoryCondenser(
            data_dir=data_dir,
            embedder=embedder,
            auto_extract=False,
            chunker_min_tokens=config.chunker.min_tokens,
            chunker_max_tokens=config.chunker.max_tokens,
            persist_index_on_close=False,
        )

    return create


def test_fresh_runner_is_gold_blind_until_frozen_packet_measurement(tmp_path):
    sample = _sample()
    config = _config()
    embedder = _DeterministicEmbedder()
    factory_calls = []
    provider_calls = []

    def legacy_inputs(condenser, *, query, retrieval, artifact_id):
        provider_calls.append(
            (query, retrieval.model_dump(mode="json"), artifact_id)
        )
        ranked = condenser.search(query, k=retrieval.k, ef_search=retrieval.ef_search)
        # Freeze a genuine legacy result from alpha. Independently route beta;
        # beta is intentionally absent from the direct anchors.
        alpha = tuple(
            row
            for row in ranked
            if row.turn is not None and row.turn.source_id == "alpha"
        )[:1]
        assert alpha
        source_scope = condenser.route_discourse_episode_sources(
            query,
            alpha,
            artifact_id=artifact_id,
            max_sources=8,
        )
        assert "beta" in {
            candidate.source_id for candidate in source_scope.candidates
        }
        return LegacyDiffuseCandidates(
            anchors=alpha,
            source_candidate_scope=source_scope,
        )

    analysis = run_diffuse_longmemeval_analysis(
        sample,
        config=config,
        arm=_fixed_arm(),
        data_dir=tmp_path / "fresh-fixed",
        condenser_factory=_factory(embedder, factory_calls),
        legacy_input_provider=legacy_inputs,
        representative_linker=_SelectEveryEpisodeLinker(),
        representative_policy_factory=_representative_policy,
    )

    assert len(factory_calls) == 1
    assert len(provider_calls) == 1
    measured = analysis.questions[0]
    frozen = measured.gold_blind
    phase = analysis.retrieval_phase
    blind = gold_blind_longmemeval_sample(sample)
    assert phase.deterministic_turn_ids == blind.deterministic_turn_ids
    assert provider_calls[0][0] == sample.questions[0].question
    assert frozen.probe.prompt_question == sample.questions[0].dated_question
    assert frozen.retrieval.receipt.retrieval_query_sha256 != (
        frozen.retrieval.receipt.prompt_question_sha256
    )
    assert phase.compilation.artifact.kind.endswith("fixed_interval")
    assert frozen.legacy_inputs.receipt.anchor_chunk_ids == (
        frozen.retrieval.receipt.input_anchor_chunk_ids
    )
    assert {
        row.turn.source_id
        for row in frozen.legacy_inputs.candidates.anchors
        if row.turn is not None
    } == {"alpha"}
    assert "beta" in frozen.legacy_inputs.receipt.source_candidate_ids
    assert (
        frozen.legacy_inputs.receipt.source_candidate_scope_receipt_sha256
        == frozen.legacy_inputs.candidates.source_candidate_scope.receipt_sha256
    )
    assert frozen.retrieval.representative_expansion is not None
    assert (
        frozen.retrieval.representative_expansion.source_scope_receipt_sha256
        == frozen.legacy_inputs.receipt.source_candidate_scope_receipt_sha256
    )
    assert frozen.retrieval.representative_expansion.source_universe_exhaustive
    assert {
        scan.source_id
        for scan in frozen.retrieval.representative_expansion.source_scans
    } >= {"beta"}
    assert frozen.receipt.compilation_receipt_sha256 == (
        phase.compilation.receipt_sha256
    )
    assert frozen.receipt.legacy_input_receipt_sha256 == (
        frozen.legacy_inputs.receipt.receipt_sha256
    )
    assert frozen.receipt.diffuse_query_receipt_sha256 == (
        frozen.retrieval.receipt.receipt_sha256
    )
    assert len(frozen.receipt.legacy_input_provider_identity_sha256) == 64
    assert frozen.receipt.representative_linker_identity_sha256 == (
        frozen.retrieval.representative_expansion.linker_identity_sha256
    )
    assert frozen.receipt.representative_policy_sha256 == (
        frozen.retrieval.representative_expansion.policy_sha256
    )
    assert len(
        frozen.receipt.representative_policy_factory_identity_sha256
    ) == 64
    assert len(frozen.receipt.representative_policy_controls_sha256) == 64
    assert (
        frozen.retrieval.receipt.packet_retained_request_token_state_bytes
        == 0
    )
    assert measured.metrics.answer_present is True
    assert measured.metrics.evidence_source_recall == 1.0
    assert measured.metrics.source_span_hash_valid is True

    # Change only the answer after retrieval. The frozen phase stays identical,
    # while the measurement receipt and answer-reachability result change.
    altered = _sample(answer="magenta triangle")
    atom_text = {
        atom.span: atom.text for atom in frozen.retrieval.packet.atoms
    }
    rescored = measure_diffuse_longmemeval_sample(
        phase,
        altered,
        hydrate_span=atom_text.__getitem__,
    )
    assert rescored.retrieval_phase.receipt_sha256 == phase.receipt_sha256
    assert rescored.questions[0].metrics.answer_present is False
    assert rescored.questions[0].receipt_sha256 != measured.receipt_sha256


def test_lexical_embedding_arm_compiles_through_same_pure_runner(tmp_path):
    sample = _sample()
    config = _config()
    embedder = _DeterministicEmbedder()

    def anchors_only(condenser, *, query, retrieval, artifact_id):
        assert artifact_id
        return LegacyDiffuseCandidates(
            anchors=tuple(
                condenser.search(
                    query,
                    k=retrieval.k,
                    ef_search=retrieval.ef_search,
                )
            ),
        )

    arm = replace(
        _fixed_arm(),
        arm_id="lexical_embedding",
        compilation=replace(
            _fixed_arm().compilation,
            boundary_mode="lexical_embedding",
            surprise_window=4,
            surprise_min_history=1,
        ),
    )
    analysis = run_diffuse_longmemeval_analysis(
        sample,
        config=config,
        arm=arm,
        data_dir=tmp_path / "fresh-lexical",
        condenser_factory=_factory(embedder),
        legacy_input_provider=anchors_only,
        embedding_identity={
            "model_id": "deterministic-test-embedder",
            "revision": "1",
            "dimension": embedder.dim,
        },
    )

    compilation = analysis.retrieval_phase.compilation
    assert compilation.artifact.kind.endswith("lexical_embedding")
    assert compilation.artifact.metadata["boundary_policy_id"] == (
        "lexical_embedding"
    )
    assert all(
        receipt.episode_build_sha256 is not None
        for receipt in compilation.source_receipts
        if receipt.content_chunks
    )
    assert analysis.questions[0].metrics.answer_present is True
    assert analysis.questions[0].metrics.hard_budget_compliant is True


def test_matched_arm_factory_and_fresh_store_firebreak(tmp_path):
    arms = matched_diffuse_boundary_arms(_fixed_arm())
    assert tuple(item.compilation.boundary_mode for item in arms) == (
        "fixed_interval",
        "lexical_embedding",
        "qwen_head",
    )
    assert len({item.matched_controls_sha256 for item in arms}) == 1
    assert len({item.arm_sha256 for item in arms}) == 3
    assert all(item.require_owned_representative_runtime for item in arms)

    with pytest.raises(ValueError, match="arm_id must equal"):
        replace(_fixed_arm(), arm_id="qwen_head")
    with pytest.raises(ValueError, match="owned representative"):
        DiffuseLongMemEvalArm(
            arm_id="qwen_head",
            compilation=replace(
                _fixed_arm().compilation,
                boundary_mode="qwen_head",
            ),
        )

    existing = tmp_path / "not-fresh"
    existing.mkdir()
    called = False

    def forbidden_factory(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("freshness check must run before ingestion")

    with pytest.raises(FileExistsError, match="fresh store"):
        run_diffuse_longmemeval_analysis(
            _sample(),
            config=_config(),
            arm=arms[0],
            data_dir=existing,
            condenser_factory=forbidden_factory,
            legacy_input_provider=lambda *_args, **_kwargs: (
                LegacyDiffuseCandidates(anchors=())
            ),
        )
    assert called is False

    blind = gold_blind_longmemeval_sample(_sample())
    assert not hasattr(blind.questions[0], "answer")
    assert not hasattr(blind.questions[0], "evidence_sources")


def test_deterministic_ingest_reuses_turn_chunk_and_stream_identities(tmp_path):
    # Missing authoritative timestamps get one declared deterministic sentinel,
    # rather than MemoryCondenser's wall-clock default varying across arms.
    sample = _sample().model_copy(update={"turn_created_at": []})
    blind = gold_blind_longmemeval_sample(sample)
    config = _config()
    first = _factory(_DeterministicEmbedder())(tmp_path / "id-a", config)
    second = _factory(_DeterministicEmbedder())(tmp_path / "id-b", config)
    try:
        first_ids = ingest_gold_blind_sample_deterministically(first, blind)
        second_ids = ingest_gold_blind_sample_deterministically(second, blind)
        first_streams = first.discourse_source_streams()
        second_streams = second.discourse_source_streams()

        assert first_ids == second_ids == blind.deterministic_turn_ids
        assert first_streams == second_streams
        assert [
            first.transcript.get_turn(turn_id).created_at
            for turn_id in first_ids
        ] == [
            second.transcript.get_turn(turn_id).created_at
            for turn_id in second_ids
        ]
    finally:
        first.close()
        second.close()


def test_runner_rejects_a_provider_identity_that_changes_during_call(tmp_path):
    class ChangingProvider:
        def __init__(self) -> None:
            self.revision = 0

        def analysis_identity_payload(self):
            return {"kind": "changing-provider", "revision": self.revision}

        def __call__(self, condenser, *, query, retrieval, artifact_id):
            del condenser, query, retrieval, artifact_id
            self.revision += 1
            return LegacyDiffuseCandidates(anchors=())

    with pytest.raises(RuntimeError, match="provider identity changed"):
        run_diffuse_longmemeval_analysis(
            _sample(),
            config=_config(),
            arm=_fixed_arm(),
            data_dir=tmp_path / "changing-provider",
            condenser_factory=_factory(_DeterministicEmbedder()),
            legacy_input_provider=ChangingProvider(),
        )


def test_callable_identity_is_independent_of_checkout_filename():
    source = (
        "def provider(value):\n"
        "    def nested(item):\n"
        "        return item + 1\n"
        "    return nested(value)\n"
    )

    def compiled_provider(filename, text=source):
        namespace = {"__name__": "stable_provider_fixture"}
        exec(compile(text, filename, "exec"), namespace)
        return namespace["provider"]

    first = compiled_provider("C:/checkout-a/provider.py")
    second = compiled_provider("D:/other/checkout-b/provider.py")
    assert first.__code__.co_filename != second.__code__.co_filename
    assert analysis_module._callable_implementation_sha256(
        first,
        "provider",
    ) == analysis_module._callable_implementation_sha256(second, "provider")

    changed = compiled_provider(
        "C:/checkout-a/provider.py",
        source.replace("item + 1", "item + 2"),
    )
    assert analysis_module._callable_implementation_sha256(
        first,
        "provider",
    ) != analysis_module._callable_implementation_sha256(changed, "provider")


def test_suite_receipt_proves_matched_inputs_and_declared_pipeline_only(
    tmp_path,
    monkeypatch,
):
    """Exercise the suite contract without constructing or loading a model.

    The Qwen compilation receipt is a deliberately synthetic unit-test
    witness. Production cannot obtain the same certification by injection:
    ``compile_diffuse_artifact`` requires the exact owned Qwen scorer.
    """

    import memory_condense.search.episodes.representative_retrieval as rep

    original_linker_identity = rep._linker_identity

    def test_owned_linker_identity(linker):
        payload = dict(original_linker_identity(linker))
        payload["owned_runtime_binding"] = True
        payload["implementation_sha256"] = identity_sha256(
            {"unit_test_owned_linker": True}
        )
        return payload

    monkeypatch.setattr(rep, "_linker_identity", test_owned_linker_identity)
    original_compile = analysis_module.compile_diffuse_artifact

    def provider_free_compilation_stub(
        condenser,
        *,
        policy,
        qwen_scorer=None,
        embedding_identity=None,
    ):
        if policy.boundary_mode != "qwen_head":
            return original_compile(
                condenser,
                policy=policy,
                qwen_scorer=qwen_scorer,
                embedding_identity=embedding_identity,
            )
        base = original_compile(
            condenser,
            policy=replace(policy, boundary_mode="fixed_interval"),
            qwen_scorer=None,
            embedding_identity=embedding_identity,
        )
        policy_sha256 = identity_sha256(
            {
                "synthetic_test_qwen_pipeline": policy.identity_payload(),
                "base_policy_sha256": base.policy_sha256,
            }
        )
        artifact = replace(
            base.artifact,
            kind="longmemeval-diffuse-qwen-head-test-witness",
            policy_sha256=policy_sha256,
            metadata={
                **base.artifact.metadata,
                "boundary_policy_id": "qwen_head",
                "scorer_id": identity_sha256(
                    {"synthetic_test_qwen_scorer": True}
                ),
            },
        )
        sources = tuple(
            replace(
                source,
                surprise_signal_receipt_sha256=(
                    None
                    if source.content_chunks == 0
                    else identity_sha256(
                        {
                            "source_id": source.source_id,
                            "synthetic_test_qwen_signal": True,
                        }
                    )
                ),
                returned_signal_transformer_state_bytes=0,
                receipt_sha256="",
            )
            for source in base.source_receipts
        )
        return DiffuseCompilationReceipt(
            artifact=artifact,
            compilation_policy_sha256=policy.policy_sha256,
            policy_sha256=policy_sha256,
            source_receipts=sources,
            episode_coverage_receipt_sha256=(
                base.episode_coverage_receipt_sha256
            ),
            discourse_coverage_receipt_sha256=(
                base.discourse_coverage_receipt_sha256
            ),
            final_snapshot=base.final_snapshot,
            persisted_request_token_state_bytes=0,
        )

    monkeypatch.setattr(
        analysis_module,
        "compile_diffuse_artifact",
        provider_free_compilation_stub,
    )
    sample = _sample()
    config = _config()
    embedder = _DeterministicEmbedder()
    linker = _SelectEveryEpisodeLinker()
    arms = matched_diffuse_boundary_arms(_fixed_arm())

    def legacy_inputs(condenser, *, query, retrieval, artifact_id):
        anchors = tuple(
            condenser.search(
                query,
                k=retrieval.k,
                ef_search=retrieval.ef_search,
            )
        )[:2]
        return LegacyDiffuseCandidates(
            anchors=anchors,
            source_candidate_scope=condenser.route_discourse_episode_sources(
                query,
                anchors,
                artifact_id=artifact_id,
                max_sources=8,
            ),
        )

    phases = []
    for arm in arms:
        analysis = run_diffuse_longmemeval_analysis(
            sample,
            config=config,
            arm=arm,
            data_dir=tmp_path / f"matched-{arm.arm_id}",
            condenser_factory=_factory(embedder),
            legacy_input_provider=legacy_inputs,
            embedding_identity={
                "model_id": "deterministic-test-embedder",
                "revision": "1",
                "dimension": embedder.dim,
            },
            representative_linker=linker,
            representative_policy_factory=_representative_policy,
        )
        phases.append(analysis.retrieval_phase)

    receipt = validate_matched_diffuse_retrieval_phases(phases[::-1])
    assert receipt.pipeline_modes == (
        "fixed_interval",
        "lexical_embedding",
        "qwen_head",
    )
    assert len(set(receipt.pipeline_arm_sha256s)) == 3
    assert len(receipt.probes) == 1
    assert receipt.qwen_owned_representative_runtime is True
    assert receipt.zero_returned_transformer_state is True
    assert receipt.zero_persisted_transformer_state is True
    assert receipt.qwen_source_signal_receipt_sha256s

    wrong_compilation_policy = replace(
        phases[0].compilation,
        compilation_policy_sha256=(
            phases[1].arm.compilation.policy_sha256
        ),
        receipt_sha256="",
    )
    wrong_policy_query_receipt = replace(
        phases[0].questions[0].receipt,
        compilation_receipt_sha256=(
            wrong_compilation_policy.receipt_sha256
        ),
        receipt_sha256="",
    )
    wrong_policy_query = replace(
        phases[0].questions[0],
        receipt=wrong_policy_query_receipt,
    )
    wrong_policy_phase = replace(
        phases[0],
        compilation=wrong_compilation_policy,
        questions=(wrong_policy_query,),
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="does not match its declared arm"):
        validate_matched_diffuse_retrieval_phases(
            (wrong_policy_phase, phases[1], phases[2])
        )

    def consistently_partial_phase(phase):
        question = phase.questions[0]
        scope = question.legacy_inputs.candidates.source_candidate_scope
        assert scope is not None
        retained_source = scope.universe_source_ids[0]
        retained_candidates = tuple(
            candidate
            for candidate in scope.candidates
            if candidate.source_id == retained_source
        )
        assert retained_candidates
        partial_scope = replace(
            scope,
            universe_source_ids=(retained_source,),
            candidates=retained_candidates,
            truncated_source_ids=(),
            universe_enumerated=True,
            receipt_sha256="",
        )
        partial_inputs = capture_legacy_diffuse_inputs(
            query=sample.questions[0].question,
            retrieval=config.retrieval,
            artifact_id=phase.compilation.artifact.artifact_id,
            candidates=LegacyDiffuseCandidates(
                anchors=question.legacy_inputs.candidates.anchors,
                source_candidate_scope=partial_scope,
            ),
        )
        representative = replace(
            question.retrieval.representative_expansion,
            source_scope_receipt_sha256=partial_scope.receipt_sha256,
            source_universe_exhaustive=True,
            receipt_sha256="",
        )
        diffuse_receipt = replace(
            question.retrieval.receipt,
            representative_receipt_sha256=representative.receipt_sha256,
            representative_scope_exhaustive=(
                representative.candidate_scope_exhaustive
            ),
            receipt_sha256="",
        )
        retrieval = replace(
            question.retrieval,
            representative_expansion=representative,
            receipt=diffuse_receipt,
        )
        analysis_receipt = replace(
            question.receipt,
            legacy_input_receipt_sha256=(
                partial_inputs.receipt.receipt_sha256
            ),
            diffuse_query_receipt_sha256=diffuse_receipt.receipt_sha256,
            receipt_sha256="",
        )
        partial_question = replace(
            question,
            legacy_inputs=partial_inputs,
            retrieval=retrieval,
            receipt=analysis_receipt,
        )
        return replace(
            phase,
            questions=(partial_question,),
            receipt_sha256="",
        )

    consistently_partial = tuple(
        consistently_partial_phase(phase) for phase in phases
    )
    with pytest.raises(ValueError, match="compiled source universe"):
        validate_matched_diffuse_retrieval_phases(consistently_partial)

    qwen_phase = phases[2]
    first_qwen_source = qwen_phase.compilation.source_receipts[0]
    unattested_source = replace(
        first_qwen_source,
        returned_signal_transformer_state_bytes=None,
        receipt_sha256="",
    )
    unattested_compilation = replace(
        qwen_phase.compilation,
        source_receipts=(
            unattested_source,
            *qwen_phase.compilation.source_receipts[1:],
        ),
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="zero-state attestation"):
        replace(
            qwen_phase,
            compilation=unattested_compilation,
            receipt_sha256="",
        )

    missing_signal_source = replace(
        first_qwen_source,
        surprise_signal_receipt_sha256=None,
        receipt_sha256="",
    )
    missing_signal_compilation = replace(
        qwen_phase.compilation,
        source_receipts=(
            missing_signal_source,
            *qwen_phase.compilation.source_receipts[1:],
        ),
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="lacks a Qwen signal receipt"):
        replace(
            qwen_phase,
            compilation=missing_signal_compilation,
            receipt_sha256="",
        )

    changed_provider = replace(
        phases[1].questions[0].receipt,
        legacy_input_provider_identity_sha256=identity_sha256(
            {"different_provider": True}
        ),
        receipt_sha256="",
    )
    changed_query = replace(
        phases[1].questions[0],
        receipt=changed_provider,
    )
    changed_phase = replace(
        phases[1],
        questions=(changed_query,),
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="legacy input provider"):
        validate_matched_diffuse_retrieval_phases(
            (phases[0], changed_phase, phases[2])
        )

    original_inputs = phases[1].questions[0].legacy_inputs
    changed_anchors = capture_legacy_diffuse_inputs(
        query=sample.questions[0].question,
        retrieval=config.retrieval,
        artifact_id=phases[1].compilation.artifact.artifact_id,
        candidates=LegacyDiffuseCandidates(
            anchors=tuple(reversed(original_inputs.candidates.anchors)),
            source_candidate_scope=(
                original_inputs.candidates.source_candidate_scope
            ),
        ),
    )
    changed_anchor_receipt = replace(
        phases[1].questions[0].receipt,
        legacy_input_receipt_sha256=changed_anchors.receipt.receipt_sha256,
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="changed the exact legacy anchors"):
        replace(
            phases[1].questions[0],
            legacy_inputs=changed_anchors,
            receipt=changed_anchor_receipt,
        )

    no_scope_candidates = replace(
        phases[1].questions[0].legacy_inputs.candidates,
        source_candidate_scope=None,
    )
    no_scope_receipt = replace(
        phases[1].questions[0].legacy_inputs.receipt,
        source_candidate_scope_receipt_sha256=None,
        receipt_sha256="",
    )
    no_scope_inputs = replace(
        phases[1].questions[0].legacy_inputs,
        candidates=no_scope_candidates,
        receipt=no_scope_receipt,
    )
    no_scope_analysis_receipt = replace(
        phases[1].questions[0].receipt,
        legacy_input_receipt_sha256=no_scope_receipt.receipt_sha256,
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="changed the exact source scope"):
        replace(
            phases[1].questions[0],
            legacy_inputs=no_scope_inputs,
            receipt=no_scope_analysis_receipt,
        )
