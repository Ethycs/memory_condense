from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import memory_condense.eval.diffuse_longmemeval_replay as replay_module
import memory_condense.search.episodes.representative_retrieval as rep_module
import memory_condense.search.episodes.qwen_episode_signal as signal_module
import tests.test_diffuse_longmemeval_analysis as analysis_fixture
import tests.test_diffuse_longmemeval_base as base_fixture
from tests.test_qwen_episode_signals import _FakePrefixLinker
from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval._diffuse_base_contracts import canonical_json_bytes
from memory_condense.eval._diffuse_replay_contracts import CanonicalIdentityBody
from memory_condense.eval.diffuse_longmemeval_replay import (
    run_diffuse_longmemeval_shared_base_replay,
    verify_diffuse_longmemeval_replay_package,
)
from memory_condense.eval.diffuse_longmemeval_runtime import (
    ResidencyPreflightObservation,
)
from memory_condense.search.episodes import QwenAttentionHeadSurpriseScorer


def _resign_canonical(value, mutate) -> CanonicalIdentityBody:
    body = json.loads(value.canonical_identity_json)
    mutate(body)
    unsigned = dict(body)
    if value.self_hash_field is not None:
        unsigned.pop(value.self_hash_field)
    digest = identity_sha256(unsigned)
    if value.self_hash_field is not None:
        body[value.self_hash_field] = digest
    return CanonicalIdentityBody.seal(
        body,
        identity_sha256_value=digest,
        self_hash_field=value.self_hash_field,
    )


def _resign_record(value, field: str):
    digest = identity_sha256(value.model_dump(mode="json", exclude={field}))
    return value.model_copy(update={field: digest})


def _publish_tampered_manifest(path, receipt) -> None:
    path.write_bytes(canonical_json_bytes(receipt.model_dump(mode="json")))


def _install_provider_free_qwen_witness(monkeypatch) -> None:
    original_identity = rep_module._linker_identity

    def test_owned_linker_identity(linker):
        payload = dict(original_identity(linker))
        payload["owned_runtime_binding"] = True
        payload["implementation_sha256"] = identity_sha256(
            {"unit_test_owned_linker": True}
        )
        return payload

    monkeypatch.setattr(rep_module, "_linker_identity", test_owned_linker_identity)
    monkeypatch.setattr(
        signal_module,
        "_owned_qwen_runtime_binding",
        lambda _linker: True,
    )


def _fake_resident_binding(*, base, embedder, factory):
    linker = analysis_fixture._SelectEveryEpisodeLinker()
    signal_linker = _FakePrefixLinker(
        tuple((1.0, 0.0) if index % 2 == 0 else (0.0, 1.0) for index in range(32)),
        max_candidates=8,
        max_workspace_tokens=2048,
    )
    scorer = QwenAttentionHeadSurpriseScorer(signal_linker)
    required = 3 * 1024 * 1024 * 1024
    observation = ResidencyPreflightObservation(
        policy="cuda-mem-get-info-min-free-v1",
        device="cuda:0",
        required_free_bytes=required,
        observed_free_bytes=required + 1024,
        observed_total_bytes=required * 2,
        embedding_released_before_qwen_load=False,
    )
    embedding = base.store_manifest.embedding_identity.model_dump(mode="json")
    retrieval = base._config.retrieval
    factory_names = (
        "embedding", "condenser", "qwen_encoder", "qwen_linker",
        "qwen_scorer", "qwen_reranker", "resident_preflight",
    )
    payload = {
        "format": "memory-condense-longmemeval-diffuse-runtime-v1",
        "runtime_binding_certified": True,
        "residency_mode": "resident_bge_qwen",
        "resident_preflight": {
            "policy": observation.policy,
            "required_free_bytes": required,
        },
        "embedding": embedding,
        "qwen": {
            "model_locator": "local-verified-checkpoint",
            "model_id": "test/qwen-prefix",
            "model_revision": "fixture-v1",
            "checkpoint_sha256": identity_sha256({"fake_qwen": True}),
            "prefix_layers": retrieval.qwen_rerank_prefix_layers,
            "attention_layer": retrieval.qwen_rerank_attention_layer,
            "device": "cuda:0",
            "dtype": "float16",
            "max_candidates": 8,
            "max_workspace_tokens": 2048,
            "surprise": {
                "max_spans": 256,
                "span_token_cap": 64,
                "probe_token_cap": 96,
                "max_transport_dimension": 8192,
            },
        },
        "source_router": {"max_sources": 8, "rrf_constant": 60},
        "representative": {
            "max_input_sources": 8,
            "max_source_groups": 2,
            "max_episodes_per_source": 8,
            "max_total_episodes": 16,
            "max_representatives_per_episode": 1,
            "group_size": 8,
            "beam_per_group": 1,
            "top_k": 4,
            "representative_tokens": 64,
            "query_tokens": 64,
            "score_mode": "qk_ov",
        },
        "retrieval_policy_sha256": identity_sha256(
            base._config.retrieval.model_dump(mode="json")
        ),
        "factories": {
            name: {
                "callable": f"tests.fake.{name}",
                "python_code_sha256": identity_sha256({"factory": name}),
            }
            for name in factory_names
        },
    }

    class Binding:
        config = base._config
        embedding_identity = embedding
        runtime_binding_certified = True
        runtime = SimpleNamespace(
            residency_mode="resident_bge_qwen",
            source_router_max_sources=8,
            source_router_rrf_constant=60,
        )
        representative_policy_factory = staticmethod(
            analysis_fixture._representative_policy
        )

        def __init__(self):
            self.embedder = embedder
            self.new_condenser = factory

        @property
        def binding_sha256(self):
            return identity_sha256(payload)

        def analysis_identity_payload(self):
            return payload

        def prepare_resident_replay_runtime(self):
            return observation, SimpleNamespace(
                linker=linker,
                scorer=scorer,
                reranker=None,
            )

    return Binding()


@pytest.mark.parametrize(
    ("embedding_device", "qwen_device"),
    (("cpu", "cuda"), ("cuda:0", "cuda:1"), ("cuda:garbage", "cuda:0")),
)
def test_resident_replay_requires_one_canonical_cuda_device(
    embedding_device,
    qwen_device,
):
    binding = SimpleNamespace(
        embedding_identity={"device": embedding_device},
        runtime=SimpleNamespace(qwen_device=qwen_device),
    )
    with pytest.raises(ValueError):
        replay_module._require_resident_cuda_pair(binding)


def test_resident_replay_normalizes_default_cuda_device():
    binding = SimpleNamespace(
        embedding_identity={"device": "cuda"},
        runtime=SimpleNamespace(qwen_device="cuda:0"),
    )
    replay_module._require_resident_cuda_pair(binding)


def test_provider_free_shared_base_replay_is_closed_and_reconstructable(
    tmp_path,
    monkeypatch,
):
    _install_provider_free_qwen_witness(monkeypatch)
    execution_getter = base_fixture._DeterministicEmbedder.execution_identity.fget
    embedding_factory = base_fixture._embedding_identity
    monkeypatch.setattr(
        base_fixture._DeterministicEmbedder,
        "execution_identity",
        property(
            lambda self: {
                **execution_getter(self),
                "device": "cuda:0",
            }
        ),
    )
    monkeypatch.setattr(
        base_fixture,
        "_embedding_identity",
        lambda: embedding_factory().model_copy(update={"device": "cuda:0"}),
    )
    config = analysis_fixture._config().model_copy(
        update={"embedding_device": "cuda:0"}
    )
    with base_fixture._published(
        tmp_path / "published",
        config=config,
    ) as published:
        (
            base,
            sample,
            _config,
            treatment,
            _embedding,
            build_runtime,
            embedder,
            _calls,
            factory,
        ) = published
        binding = _fake_resident_binding(
            base=base,
            embedder=embedder,
            factory=factory,
        )
        monkeypatch.setattr(
            replay_module,
            "_require_owned_binding",
            lambda supplied: supplied,
        )
        monkeypatch.setattr(
            replay_module,
            "owned_build_runtime_identity",
            lambda _factory: build_runtime,
        )
        monkeypatch.setattr(
            replay_module,
            "publish_diffuse_longmemeval_base",
            lambda *_args, **_kwargs: base,
        )
        target = tmp_path / "replay"
        receipt = run_diffuse_longmemeval_shared_base_replay(
            sample,
            treatment_identity=treatment,
            binding=binding,
            reference_arm=analysis_fixture._fixed_arm(),
            cache_root=tmp_path / "cache-coordinate",
            replay_root=target,
        )

        assert verify_diffuse_longmemeval_replay_package(
            target,
            base=base,
            expected_runtime_binding_sha256=binding.binding_sha256,
        ) == receipt
        assert receipt.launcher_binding_certified is False
        assert receipt.treatment_population_membership_certified is False
        assert receipt.retrieval_input_schema_contains_gold_fields is False
        assert receipt.qa_responder_or_judge_calls == 0
        assert tuple(arm.boundary_mode for arm in receipt.arms) == (
            "fixed_interval",
            "lexical_embedding",
            "qwen_head",
        )
        manifest = (target / "replay-manifest.json").read_text(encoding="utf-8")
        assert all(text not in manifest for _role, text in sample.turns)
        assert sample.questions[0].retrieval_query not in manifest
        assert sample.questions[0].prompt_question not in manifest

        with pytest.raises(RuntimeError, match="another runtime binding"):
            verify_diffuse_longmemeval_replay_package(
                target,
                base=base,
                expected_runtime_binding_sha256="0" * 64,
            )

        unexpected = target / "unexpected.txt"
        unexpected.write_text("not admitted", encoding="utf-8")
        try:
            with pytest.raises(Exception, match="unexpected"):
                verify_diffuse_longmemeval_replay_package(
                    target,
                    base=base,
                    expected_runtime_binding_sha256=binding.binding_sha256,
                )
        finally:
            unexpected.unlink()

        manifest_path = target / "replay-manifest.json"
        original_manifest = manifest_path.read_bytes()
        first_arm = receipt.arms[0]
        first_query = first_arm.queries[0]
        altered_legacy = _resign_canonical(
            first_query.legacy_input,
            lambda body: body.update(
                source_candidate_ids=["invented-source-coordinate"]
            ),
        )
        altered_query = _resign_record(
            first_query.model_copy(update={"legacy_input": altered_legacy}),
            "record_sha256",
        )
        altered_arm = _resign_record(
            first_arm.model_copy(
                update={"queries": (altered_query, *first_arm.queries[1:])}
            ),
            "record_sha256",
        )
        altered_receipt = _resign_record(
            receipt.model_copy(
                update={"arms": (altered_arm, *receipt.arms[1:])}
            ),
            "receipt_sha256",
        )
        _publish_tampered_manifest(manifest_path, altered_receipt)
        try:
            with pytest.raises(RuntimeError, match="invalid replay manifest"):
                verify_diffuse_longmemeval_replay_package(
                    target,
                    base=base,
                    expected_runtime_binding_sha256=binding.binding_sha256,
                )
        finally:
            manifest_path.write_bytes(original_manifest)

        for flag in (
            "qwen_owned_representative_runtime",
            "zero_returned_transformer_state",
        ):
            altered_phase = _resign_canonical(
                receipt.matched_phase_suite,
                lambda body, flag=flag: body.update({flag: False}),
            )
            altered_runtime = _resign_canonical(
                receipt.matched_runtime_suite,
                lambda body: body.update(
                    matched_suite_receipt_sha256=(
                        altered_phase.identity_sha256
                    )
                ),
            )
            altered_receipt = _resign_record(
                receipt.model_copy(
                    update={
                        "matched_phase_suite": altered_phase,
                        "matched_runtime_suite": altered_runtime,
                    }
                ),
                "receipt_sha256",
            )
            _publish_tampered_manifest(manifest_path, altered_receipt)
            try:
                with pytest.raises(RuntimeError, match="invalid replay manifest"):
                    verify_diffuse_longmemeval_replay_package(
                        target,
                        base=base,
                        expected_runtime_binding_sha256=binding.binding_sha256,
                    )
            finally:
                manifest_path.write_bytes(original_manifest)

        altered_result = _resign_canonical(
            first_arm.runtime_result,
            lambda body: body.update(
                format="invented-runtime-result-v999"
            ),
        )
        altered_arm = _resign_record(
            first_arm.model_copy(update={"runtime_result": altered_result}),
            "record_sha256",
        )
        altered_runtime = _resign_canonical(
            receipt.matched_runtime_suite,
            lambda body: body.update(
                runtime_result_receipt_sha256s=[
                    altered_result.identity_sha256,
                    *body["runtime_result_receipt_sha256s"][1:],
                ]
            ),
        )
        altered_receipt = _resign_record(
            receipt.model_copy(
                update={
                    "arms": (altered_arm, *receipt.arms[1:]),
                    "matched_runtime_suite": altered_runtime,
                }
            ),
            "receipt_sha256",
        )
        _publish_tampered_manifest(manifest_path, altered_receipt)
        try:
            with pytest.raises(RuntimeError, match="invalid replay manifest"):
                verify_diffuse_longmemeval_replay_package(
                    target,
                    base=base,
                    expected_runtime_binding_sha256=binding.binding_sha256,
                )
        finally:
            manifest_path.write_bytes(original_manifest)

        with pytest.raises(FileExistsError):
            run_diffuse_longmemeval_shared_base_replay(
                sample,
                treatment_identity=treatment,
                binding=binding,
                reference_arm=analysis_fixture._fixed_arm(),
                cache_root=tmp_path / "cache-coordinate",
                replay_root=target,
            )
