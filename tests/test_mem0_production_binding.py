from __future__ import annotations

import copy
import hashlib
import inspect
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)
from tools.mem0_eval.preflight import tool_implementation_sha256
from tools.mem0_eval.production_binding import (
    FrozenMem0RetrievalLauncher,
    FrozenMem0ScoringLauncher,
    HardTransportAttemptCap,
    InjectedHardCappedExtractionTransport,
    InjectedHardCappedJudgeTransport,
    InjectedHardCappedResponderTransport,
    ProductionBindingBlocked,
    ProductionBindingError,
    TransportAttemptLimitExceeded,
    probe_local_bge_m3_runtime,
    production_binding_readiness,
    validate_local_bge_m3_contract,
    validate_production_mem0_config,
    verify_frozen_artifact_binding,
)
from tools.mem0_eval.run_shard import (
    ProviderCallResult,
    RetrievalStageAuthorization,
    ScoringStageAuthorization,
    TrustedRuntimeBinding,
    canonical_json_sha256,
)
from memory_condense.eval.schemas import UsageStats


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def _embedder_identity(*, device: str = "cuda") -> dict[str, Any]:
    body = {
        "provider": "huggingface",
        "model": DEFAULT_MODEL_NAME,
        "revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "dimension": DEFAULT_MODEL_DIM,
        "device": device,
        "dtype": "float32",
        "execution": "local_offline",
        "network_calls_authorized": 0,
        "runtime_probe_required": True,
    }
    return {**body, "model_identity_sha256": canonical_json_sha256(body)}


def _extraction_identity() -> dict[str, Any]:
    body = {
        "provider": "unresolved-provider",
        "model": "unresolved-model",
        "revision": "unresolved-revision",
        "provider_retries": 0,
        "logical_call_boundary": "Memory.llm.generate_response",
        "logical_calls_per_add": 1,
        "http_attempts_certified": False,
    }
    return {**body, "model_identity_sha256": canonical_json_sha256(body)}


def _stable_payload(*, device: str = "cuda") -> dict[str, Any]:
    return {
        "protocol": "mem0-oss-2.0.18-certified-local-v1",
        "config": {
            "version": "v1.1",
            "custom_instructions": None,
            "reranker": None,
            "llm": {
                "provider": "unresolved-provider",
                "config": {"model": "unresolved-model"},
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": DEFAULT_MODEL_NAME,
                    "embedding_dims": DEFAULT_MODEL_DIM,
                    "huggingface_base_url": None,
                    "model_kwargs": {
                        "revision": DEFAULT_MODEL_REVISION,
                        "local_files_only": True,
                        "trust_remote_code": False,
                        "device": device,
                    },
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "embedding_model_dims": DEFAULT_MODEL_DIM,
                    "collection_name": "longmemeval",
                    "path": "<owned_state>/qdrant",
                    "on_disk": True,
                },
            },
            "history_db_path": "<owned_state>/history.sqlite",
        },
        "stack": {
            "dependency_versions": {
                "mem0ai": "2.0.18",
                "qdrant-client": "1.15.1",
                "fastembed": "0.7.3",
                "spacy": "3.8.7",
                "en-core-web-sm": "3.8.0",
            },
            "bm25_model": "Qdrant/bm25",
            "spacy_model": "en_core_web_sm",
            "bm25_operational": True,
            "entity_extraction_operational": True,
        },
    }


def _retrieval_authorization(
    *,
    policy_sha256: str = SHA_A,
    lock_sha256: str = SHA_B,
    tool_sha256: str = SHA_C,
    device: str = "cuda",
) -> RetrievalStageAuthorization:
    stable = _stable_payload(device=device)
    extraction = _extraction_identity()
    embedder = _embedder_identity(device=device)
    return RetrievalStageAuthorization(
        sample_offset=0,
        sample_sha256=SHA_D,
        raw_history_bundle_sha256=SHA_E,
        question_ids=tuple(f"q-{index}" for index in range(10)),
        authorized_add_operations=2,
        authorized_extraction_calls=2,
        authorized_search_operations=10,
        source_validation_policy_sha256=SHA_F,
        source_implementation_sha256=SHA_D,
        source_environment_lock_sha256=SHA_E,
        mem0_policy_sha256=policy_sha256,
        mem0_tool_implementation_sha256=tool_sha256,
        mem0_environment_lock_sha256=lock_sha256,
        mem0_stable_config_sha256=canonical_json_sha256(stable),
        source_evaluation_identity={},
        mem0_stable_payload=stable,
        extraction_model_identity=extraction,
        extraction_model_identity_sha256=canonical_json_sha256(extraction),
        embedder_model_identity=embedder,
        embedder_model_identity_sha256=canonical_json_sha256(embedder),
    )


def _scoring_authorization(
    retrieval: RetrievalStageAuthorization,
    *,
    source_lock_sha256: str,
) -> ScoringStageAuthorization:
    return ScoringStageAuthorization(
        sample_offset=retrieval.sample_offset,
        sample_sha256=retrieval.sample_sha256,
        raw_history_bundle_sha256=retrieval.raw_history_bundle_sha256,
        question_ids=retrieval.question_ids,
        retrieval_artifact_sha256=SHA_F,
        source_validation_policy_sha256=retrieval.source_validation_policy_sha256,
        source_implementation_sha256=retrieval.source_implementation_sha256,
        source_environment_lock_sha256=source_lock_sha256,
        mem0_policy_sha256=retrieval.mem0_policy_sha256,
        mem0_tool_implementation_sha256=retrieval.mem0_tool_implementation_sha256,
        mem0_environment_lock_sha256=retrieval.mem0_environment_lock_sha256,
        mem0_stable_config_sha256=retrieval.mem0_stable_config_sha256,
        source_evaluation_identity=retrieval.source_evaluation_identity,
        mem0_stable_payload=retrieval.mem0_stable_payload,
        scoring_policy_sha256=retrieval.mem0_policy_sha256,
        responder_model="openai/codex_sdk/gpt-5.6-terra",
        judge_model="openai/codex_sdk/gpt-5.6-sol",
        responder_model_identity_sha256=SHA_A,
        judge_model_identity_sha256=SHA_B,
        extraction_model_identity=retrieval.extraction_model_identity,
        extraction_model_identity_sha256=(
            retrieval.extraction_model_identity_sha256
        ),
        embedder_model_identity=retrieval.embedder_model_identity,
        embedder_model_identity_sha256=retrieval.embedder_model_identity_sha256,
    )


def _offline_environment() -> dict[str, str]:
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "true",
        "MEM0_TELEMETRY": "false",
    }


def test_readiness_is_explicitly_blocked_without_concrete_transports() -> None:
    status = dict(production_binding_readiness())

    assert status["status"] == "blocked"
    assert status["production_binding_issuance_permitted"] is False
    assert status["external_provider_persistence_certified"] is False
    assert {
        row["code"] for row in status["blockers"]
    } >= {
        "extraction_provider_model_and_transport_unresolved",
        "production_mem0_adapter_factory_unresolved",
        "responder_send_transport_unresolved",
        "judge_send_transport_unresolved",
    }


def test_exact_launcher_types_are_final_and_cannot_currently_issue() -> None:
    with pytest.raises(ProductionBindingBlocked, match="extraction provider/model"):
        FrozenMem0RetrievalLauncher()
    with pytest.raises(ProductionBindingBlocked, match="Terra and Sol"):
        FrozenMem0ScoringLauncher()

    with pytest.raises(TypeError, match="cannot be subclassed"):

        class ForgedRetrievalLauncher(FrozenMem0RetrievalLauncher):
            pass


@pytest.mark.parametrize(
    "launcher",
    [object(), object.__new__(FrozenMem0RetrievalLauncher)],
    ids=["arbitrary", "constructor-bypass"],
)
def test_runner_issuer_rejects_arbitrary_or_uninitialized_launcher(
    launcher: object,
) -> None:
    from tools.mem0_eval import run_shard

    authorization = _retrieval_authorization()
    with pytest.raises((ProductionBindingError, ProductionBindingBlocked)):
        run_shard._issue_trusted_runtime_binding(
            launcher=launcher,
            stage="retrieval",
            authorization=authorization,
            bound_callables=(lambda _state: None,),
        )


def test_frozen_artifacts_are_hashed_directly_and_bound_to_authorization(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "policy.json"
    lock = tmp_path / "pixi.lock"
    tools = tmp_path / "tool"
    tools.mkdir()
    policy.write_bytes(b'{"status":"frozen"}\n')
    lock.write_bytes(b"locked-environment\n")
    (tools / "runner.py").write_text("VALUE = 1\n", encoding="utf-8")
    authorization = _retrieval_authorization(
        policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        tool_sha256=tool_implementation_sha256(tools),
    )

    receipt = verify_frozen_artifact_binding(
        authorization,
        stage="retrieval",
        policy_path=policy,
        mem0_environment_lock_path=lock,
        tool_root=tools,
    )

    assert receipt.policy_sha256 == authorization.mem0_policy_sha256
    assert (
        receipt.mem0_environment_lock_sha256
        == authorization.mem0_environment_lock_sha256
    )
    assert (
        receipt.tool_implementation_sha256
        == authorization.mem0_tool_implementation_sha256
    )
    assert receipt.source_environment_lock_path is None


@pytest.mark.parametrize("target", ["policy", "lock", "tool"])
def test_frozen_artifact_binding_rejects_each_hash_mismatch(
    tmp_path: Path,
    target: str,
) -> None:
    policy = tmp_path / "policy.json"
    lock = tmp_path / "pixi.lock"
    tools = tmp_path / "tool"
    tools.mkdir()
    policy.write_text("policy-v1", encoding="utf-8")
    lock.write_text("lock-v1", encoding="utf-8")
    (tools / "runner.py").write_text("VALUE = 1\n", encoding="utf-8")
    authorization = _retrieval_authorization(
        policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        tool_sha256=tool_implementation_sha256(tools),
    )
    if target == "policy":
        policy.write_text("policy-v2", encoding="utf-8")
    elif target == "lock":
        lock.write_text("lock-v2", encoding="utf-8")
    else:
        (tools / "runner.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ProductionBindingError, match="SHA-256"):
        verify_frozen_artifact_binding(
            authorization,
            stage="retrieval",
            policy_path=policy,
            mem0_environment_lock_path=lock,
            tool_root=tools,
        )


def test_scoring_artifact_binding_also_hashes_the_source_lock(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "policy.json"
    mem0_lock = tmp_path / "mem0.lock"
    source_lock = tmp_path / "root.lock"
    tools = tmp_path / "tool"
    tools.mkdir()
    policy.write_text("policy", encoding="utf-8")
    mem0_lock.write_text("mem0-lock", encoding="utf-8")
    source_lock.write_text("source-lock", encoding="utf-8")
    (tools / "runner.py").write_text("PASS = True\n", encoding="utf-8")
    retrieval = _retrieval_authorization(
        policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        lock_sha256=hashlib.sha256(mem0_lock.read_bytes()).hexdigest(),
        tool_sha256=tool_implementation_sha256(tools),
    )
    source_sha = hashlib.sha256(source_lock.read_bytes()).hexdigest()
    scoring = _scoring_authorization(retrieval, source_lock_sha256=source_sha)

    receipt = verify_frozen_artifact_binding(
        scoring,
        stage="scoring",
        policy_path=policy,
        mem0_environment_lock_path=mem0_lock,
        source_environment_lock_path=source_lock,
        tool_root=tools,
    )

    assert receipt.source_environment_lock_sha256 == source_sha


def test_local_bge_contract_binds_checkpoint_revision_dim_device_dtype_and_offline() -> None:
    receipt = validate_local_bge_m3_contract(
        _retrieval_authorization(), environment=_offline_environment()
    )

    assert receipt == {
        "format": "memory-condense-local-bge-m3-contract-v1",
        "model": DEFAULT_MODEL_NAME,
        "revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "dimension": DEFAULT_MODEL_DIM,
        "device": "cuda",
        "dtype": "float32",
        "local_files_only": True,
        "trust_remote_code": False,
        "huggingface_base_url": None,
        "network_calls_authorized": 0,
        "offline_environment_sha256": canonical_json_sha256(
            dict(sorted(_offline_environment().items()))
        ),
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model", "other/model"),
        ("revision", "main"),
        ("checkpoint_sha256", SHA_A),
        ("dimension", 768),
        ("device", "auto"),
        ("dtype", "float16"),
        ("execution", "remote"),
        ("network_calls_authorized", 1),
        ("runtime_probe_required", False),
    ],
)
def test_local_bge_contract_rejects_identity_drift(
    field: str,
    bad_value: Any,
) -> None:
    authorization = _retrieval_authorization()
    identity = dict(authorization.embedder_model_identity or {})
    identity[field] = bad_value
    body = dict(identity)
    body.pop("model_identity_sha256", None)
    identity["model_identity_sha256"] = canonical_json_sha256(body)
    authorization = replace(
        authorization,
        embedder_model_identity=identity,
        embedder_model_identity_sha256=canonical_json_sha256(identity),
    )

    with pytest.raises(ProductionBindingError):
        validate_local_bge_m3_contract(
            authorization, environment=_offline_environment()
        )


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("huggingface_base_url",), "https://example.invalid"),
        (("model_kwargs", "local_files_only"), False),
        (("model_kwargs", "trust_remote_code"), True),
        (("model_kwargs", "revision"), "main"),
        (("model_kwargs", "device"), "cpu"),
    ],
)
def test_local_bge_contract_rejects_online_or_mismatched_mem0_config(
    path: tuple[str, ...],
    bad_value: Any,
) -> None:
    authorization = _retrieval_authorization()
    stable = copy.deepcopy(authorization.mem0_stable_payload)
    config = stable["config"]["embedder"]["config"]
    target = config
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = bad_value
    authorization = replace(
        authorization,
        mem0_stable_payload=stable,
        mem0_stable_config_sha256=canonical_json_sha256(stable),
    )

    with pytest.raises(ProductionBindingError, match="BGE-M3 config"):
        validate_local_bge_m3_contract(
            authorization, environment=_offline_environment()
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_retries", 99),
        ("temperature", 2),
        ("api_base", "https://untrusted.invalid"),
        ("proxy", "https://untrusted.invalid"),
    ],
)
def test_production_config_rejects_arbitrary_extraction_config_fields(
    field: str,
    value: Any,
) -> None:
    authorization = _retrieval_authorization()
    stable = copy.deepcopy(authorization.mem0_stable_payload)
    stable["config"]["llm"]["config"][field] = value
    authorization = replace(
        authorization,
        mem0_stable_payload=stable,
        mem0_stable_config_sha256=canonical_json_sha256(stable),
    )

    with pytest.raises(ProductionBindingError, match="LLM config.config fields"):
        validate_production_mem0_config(authorization)


def test_production_config_rejects_secret_suffixes_and_credential_values() -> None:
    authorization = _retrieval_authorization()
    stable = copy.deepcopy(authorization.mem0_stable_payload)
    stable["config"]["llm"]["config"]["azure_session_token"] = "TOPSECRET"
    with pytest.raises(ProductionBindingError, match="secret field"):
        validate_production_mem0_config(
            replace(
                authorization,
                mem0_stable_payload=stable,
                mem0_stable_config_sha256=canonical_json_sha256(stable),
            )
        )

    extraction = dict(authorization.extraction_model_identity or {})
    extraction["model"] = "sk-live_credential_material_123456789"
    body = dict(extraction)
    body.pop("model_identity_sha256", None)
    extraction["model_identity_sha256"] = canonical_json_sha256(body)
    with pytest.raises(ProductionBindingError, match="credential-shaped"):
        validate_production_mem0_config(
            replace(
                authorization,
                extraction_model_identity=extraction,
                extraction_model_identity_sha256=canonical_json_sha256(extraction),
            )
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qdrant", "<owned_state>/../outside"),
        ("qdrant", "<owned_state>/qdrant/../../outside"),
        ("qdrant", "<owned_state>/qdrant\\..\\outside"),
        ("history", "<owned_state>/../history.sqlite"),
        ("history", "<owned_state>/state/../../history.sqlite"),
        ("history", "<owned_state>/state\\..\\history.sqlite"),
    ],
)
def test_production_config_rejects_owned_state_path_traversal(
    field: str,
    value: str,
) -> None:
    authorization = _retrieval_authorization()
    stable = copy.deepcopy(authorization.mem0_stable_payload)
    if field == "qdrant":
        stable["config"]["vector_store"]["config"]["path"] = value
    else:
        stable["config"]["history_db_path"] = value
    authorization = replace(
        authorization,
        mem0_stable_payload=stable,
        mem0_stable_config_sha256=canonical_json_sha256(stable),
    )

    with pytest.raises(ProductionBindingError, match="path traversal"):
        validate_production_mem0_config(authorization)


@pytest.mark.parametrize("name", sorted(_offline_environment()))
def test_local_bge_contract_requires_every_offline_environment_flag(name: str) -> None:
    environment = _offline_environment()
    del environment[name]

    with pytest.raises(ProductionBindingError, match=name):
        validate_local_bge_m3_contract(
            _retrieval_authorization(), environment=environment
        )


@pytest.mark.parametrize("role", ["extraction", "responder", "judge"])
def test_send_boundary_cap_rejects_an_extra_before_dispatch(role: str) -> None:
    dispatched: list[int] = []
    cap = HardTransportAttemptCap(role=role, authorized=2)

    assert cap.call(lambda value: dispatched.append(value) or value, 1) == 1
    assert cap.call(lambda value: dispatched.append(value) or value, 2) == 2
    cap.assert_closed()
    with pytest.raises(TransportAttemptLimitExceeded):
        cap.call(lambda value: dispatched.append(value) or value, 3)

    assert dispatched == [1, 2]
    assert cap.receipt() == {
        "kind": "local_transport_send_cap",
        "role": role,
        "authorized": 2,
        "attempted": 2,
        "completed": 2,
        "failed": 0,
        "rejected": 1,
        "retries_authorized": 0,
    }


def test_injected_transport_caps_remain_explicitly_nonproduction() -> None:
    extraction = InjectedHardCappedExtractionTransport(
        lambda value: value.upper(), authorized=1
    )
    assert extraction("memory") == "MEMORY"
    extraction_receipt = extraction.transport_receipt()
    assert extraction_receipt["production_eligible"] is False
    assert extraction_receipt["external_http_attempts_certified"] is False
    assert (
        extraction_receipt["external_provider_persistence_certified"] is False
    )

    def scoring_delegate(*_args: Any, **_kwargs: Any) -> ProviderCallResult:
        return ProviderCallResult(
            text="answer",
            usage=UsageStats(input_tokens=10, output_tokens=1, calls=1),
        )

    scoring_delegate.request_token_state_receipt = lambda: {  # type: ignore[attr-defined]
        "external_provider_persistence_certified": False
    }
    for wrapper_type, role in (
        (InjectedHardCappedResponderTransport, "responder"),
        (InjectedHardCappedJudgeTransport, "judge"),
    ):
        wrapper = wrapper_type(
            scoring_delegate,
            authorized=1,
            expected_model=f"model-{role}",
        )
        result = wrapper(
            [{"role": "user", "content": "question"}],
            model=f"model-{role}",
            max_output_tokens=16,
        )
        assert result.text == "answer"
        assert wrapper.transport_receipt()["production_eligible"] is False
        assert wrapper.request_token_state_receipt() == {
            "external_provider_persistence_certified": False
        }


def test_runtime_probe_has_no_injectable_loader_or_verifier_seam() -> None:
    assert set(inspect.signature(probe_local_bge_m3_runtime).parameters) == {
        "authorization",
        "model_dir",
    }


def test_direct_trusted_binding_construction_remains_impossible() -> None:
    with pytest.raises(TypeError, match="direct construction"):
        TrustedRuntimeBinding()
