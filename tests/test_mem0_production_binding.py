from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)
from tools.mem0_eval.preflight import tool_implementation_sha256
from tools.mem0_eval.policy import (
    MEM0_EXTRACTION_MODEL,
    MEM0_EXTRACTION_PROVIDER,
    MEM0_EXTRACTION_REVISION,
    MEM0_EXTRACTION_ROUTE_IDENTITY,
    MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
)
from tools.mem0_eval.production_binding import (
    ExactMem0AdapterFactory,
    FrozenMem0RetrievalLauncher,
    FrozenMem0ScoringLauncher,
    HardTransportAttemptCap,
    InjectedHardCappedExtractionTransport,
    InjectedHardCappedJudgeTransport,
    InjectedHardCappedResponderTransport,
    LiteLLMTerraExtractionTransport,
    ProductionBindingBlocked,
    ProductionBindingError,
    TransportAttemptLimitExceeded,
    probe_local_bge_m3_runtime,
    production_binding_readiness,
    run_mem0_factory_canary,
    run_terra_extraction_canary,
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


def test_isolated_mem0_manifest_pins_v3_eager_litellm_dependency() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1] / "tools" / "mem0_eval" / "pixi.toml"
    )
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["pypi-dependencies"]["litellm"] == "==1.96.2"


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
        "provider": MEM0_EXTRACTION_PROVIDER,
        "model": MEM0_EXTRACTION_MODEL,
        "revision": MEM0_EXTRACTION_REVISION,
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
                "provider": MEM0_EXTRACTION_PROVIDER,
                "config": {"model": MEM0_EXTRACTION_MODEL},
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


def test_readiness_has_exact_extraction_binding_but_scoring_stays_blocked() -> None:
    status = dict(production_binding_readiness())

    assert status["format"] == "memory-condense-mem0-production-readiness-v2"
    assert status["status"] == "blocked"
    assert status["production_binding_issuance_permitted"] is False
    assert status["external_provider_persistence_certified"] is False
    assert {
        row["code"] for row in status["blockers"]
    } >= {
        "responder_send_transport_unresolved",
        "judge_send_transport_unresolved",
    }
    assert {
        "extraction_provider_model_and_transport_unresolved",
        "extraction_send_transport_unresolved",
        "production_mem0_adapter_factory_unresolved",
        "actual_mem0_embedder_instance_probe_unimplemented",
    }.isdisjoint({
        row["code"] for row in status["blockers"]
    })
    assert status["extraction_route"] == {
        **dict(MEM0_EXTRACTION_ROUTE_IDENTITY),
        "route_identity_sha256": MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
        "revision": MEM0_EXTRACTION_REVISION,
    }


def test_exact_launcher_types_are_final_and_cannot_currently_issue() -> None:
    with pytest.raises(ProductionBindingBlocked, match="trusted issuance"):
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


def test_tool_implementation_hash_excludes_isolated_runtime_environments(
    tmp_path: Path,
) -> None:
    tools = tmp_path / "tool"
    tools.mkdir()
    (tools / "runner.py").write_text("VALUE = 1\n", encoding="utf-8")
    before = tool_implementation_sha256(tools)

    runtime = tools / ".pixi" / "envs" / "default" / "site-packages"
    runtime.mkdir(parents=True)
    (runtime / "provider.py").write_text("SECRET = 'runtime'\n", encoding="utf-8")
    cache = tools / "__pycache__"
    cache.mkdir()
    (cache / "generated.py").write_text("VALUE = 99\n", encoding="utf-8")

    assert tool_implementation_sha256(tools) == before
    (tools / "runner.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert tool_implementation_sha256(tools) != before


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


@pytest.mark.parametrize(
    ("field", "substitute"),
    [
        ("provider", "attacker-provider"),
        ("model", "attacker/model"),
        ("revision", "observable-route-sha256:" + "f" * 64),
    ],
)
def test_production_config_rejects_fully_rehashed_extraction_substitution(
    field: str,
    substitute: str,
) -> None:
    authorization = _retrieval_authorization()
    extraction = dict(authorization.extraction_model_identity or {})
    extraction[field] = substitute
    body = dict(extraction)
    body.pop("model_identity_sha256")
    extraction["model_identity_sha256"] = canonical_json_sha256(body)

    stable = copy.deepcopy(authorization.mem0_stable_payload)
    if field == "provider":
        stable["config"]["llm"]["provider"] = substitute
    elif field == "model":
        stable["config"]["llm"]["config"]["model"] = substitute

    authorization = replace(
        authorization,
        extraction_model_identity=extraction,
        extraction_model_identity_sha256=canonical_json_sha256(extraction),
        mem0_stable_payload=stable,
        mem0_stable_config_sha256=canonical_json_sha256(stable),
    )

    with pytest.raises(
        ProductionBindingError,
        match=rf"frozen extraction identity {field}",
    ):
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
    extraction["model"] = "sk-" + "live_credential_material_123456789"
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


def _terra_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Extract durable memory facts."},
        {"role": "user", "content": "Alice's favorite color is blue."},
    ]


def _completion_response(content: str, *, model: str = MEM0_EXTRACTION_MODEL) -> dict:
    return {
        "id": "chatcmpl-mem0-test",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _patch_http_transport(monkeypatch, handler):
    import httpx

    construction: list[dict[str, Any]] = []

    def build(**kwargs):
        construction.append(dict(kwargs))
        return httpx.MockTransport(handler)

    monkeypatch.setattr(httpx, "HTTPTransport", build)
    monkeypatch.setenv("LITELLM_KEY", "unit-test-key")
    return construction


def test_exact_terra_transport_caps_the_real_http_boundary_and_locks_request(
    monkeypatch,
) -> None:
    import httpx

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json=_completion_response(
                '{"memory":[{"text":"Alice likes blue."}]}'
            ),
        )

    construction = _patch_http_transport(monkeypatch, handler)
    transport = LiteLLMTerraExtractionTransport(authorized=1)
    try:
        result = transport.generate_response(
            _terra_messages(),
            response_format={"type": "json_object"},
        )
        assert json.loads(result)["memory"][0]["text"] == "Alice likes blue."
        assert len(requests) == 1
        assert requests[0].url == (
            "https://central-dev.zt:4000/v1/chat/completions"
        )
        assert json.loads(requests[0].content) == {
            "model": MEM0_EXTRACTION_MODEL,
            "messages": _terra_messages(),
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 2_000,
        }
        assert construction[0]["retries"] == 0
        transport.assert_call_budget_closed()

        with pytest.raises(Exception):
            transport.generate_response(
                _terra_messages(),
                response_format={"type": "json_object"},
            )
        assert len(requests) == 1
        receipt = transport.transport_receipt()
        request_identity_sha256 = receipt.pop("request_identity_sha256")
        provider_latency_s = receipt.pop("provider_latency_s")
        assert isinstance(request_identity_sha256, str)
        assert len(request_identity_sha256) == 64
        assert isinstance(provider_latency_s, float) and provider_latency_s >= 0.0
        assert receipt == {
            "kind": "local_transport_send_cap",
            "role": "extraction",
            "authorized": 1,
            "attempted": 1,
            "completed": 1,
            "failed": 0,
            "rejected": 1,
            "retries_authorized": 0,
            "provider_usage_status": "provider_reported_exact",
            "provider_usage_records": 1,
            "provider_input_tokens": 10,
            "provider_output_tokens": 5,
            "provider_total_tokens": 15,
            "production_eligible": True,
            "provider": MEM0_EXTRACTION_PROVIDER,
            "model": MEM0_EXTRACTION_MODEL,
            "revision": MEM0_EXTRACTION_REVISION,
            "route_identity_sha256": MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
            "gateway_url": "https://central-dev.zt:4000/v1",
            "max_completion_tokens": 2_000,
            "sampling_parameters_omitted": True,
            "sdk_retries": 0,
            "http_transport_retries": 0,
            "follow_redirects": False,
            "trust_env": False,
            "cap_boundary": "httpx.BaseTransport.handle_request",
            "external_http_attempts_certified": True,
            "external_provider_persistence_certified": False,
        }
    finally:
        transport.close()


def test_exact_terra_transport_rejects_prompt_or_request_drift_before_send(
    monkeypatch,
) -> None:
    def handler(_request):
        pytest.fail("invalid extraction requests must not reach HTTP")

    _patch_http_transport(monkeypatch, handler)
    transport = LiteLLMTerraExtractionTransport(authorized=1)
    try:
        with pytest.raises(ProductionBindingError, match="system/user"):
            transport.generate_response(
                [{"role": "user", "content": "memory"}],
                response_format={"type": "json_object"},
            )
        with pytest.raises(ProductionBindingError, match="tools"):
            transport.generate_response(
                _terra_messages(),
                tools=[{"type": "function"}],
                response_format={"type": "json_object"},
            )
        with pytest.raises(ProductionBindingError, match="response format"):
            transport.generate_response(
                _terra_messages(),
                response_format={"type": "text"},
            )
        assert transport.transport_receipt()["attempted"] == 0
    finally:
        transport.close()


def test_exact_terra_transport_rejects_response_model_substitution(
    monkeypatch,
) -> None:
    import httpx

    def handler(_request):
        return httpx.Response(
            200,
            json=_completion_response(
                '{"memory":[{"text":"fact"}]}',
                model="codex_sdk/gpt-5.6-luna",
            ),
        )

    _patch_http_transport(monkeypatch, handler)
    transport = LiteLLMTerraExtractionTransport(authorized=1)
    try:
        with pytest.raises(ProductionBindingError, match="response model"):
            transport.generate_response(
                _terra_messages(),
                response_format={"type": "json_object"},
            )
        receipt = transport.transport_receipt()
        assert receipt["attempted"] == receipt["completed"] == 1
        assert receipt["failed"] == receipt["rejected"] == 0
    finally:
        transport.close()


@pytest.mark.parametrize(
    "usage,match",
    [
        (
            {"completion_tokens": 5, "total_tokens": 15},
            "omitted exact non-negative provider usage",
        ),
        (
            {"prompt_tokens": True, "completion_tokens": 5, "total_tokens": 6},
            "omitted exact non-negative provider usage",
        ),
        (
            {"prompt_tokens": 10, "completion_tokens": -1, "total_tokens": 9},
            "omitted exact non-negative provider usage",
        ),
        (
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 16},
            "usage total does not close",
        ),
    ],
)
def test_exact_terra_transport_rejects_partial_or_mismatched_provider_usage(
    monkeypatch, usage: dict[str, object], match: str
) -> None:
    import httpx

    response = _completion_response('{"memory":[{"text":"fact"}]}')
    response["usage"] = usage
    _patch_http_transport(
        monkeypatch, lambda _request: httpx.Response(200, json=response)
    )
    transport = LiteLLMTerraExtractionTransport(authorized=1)
    try:
        with pytest.raises(ProductionBindingError, match=match):
            transport.generate_response(
                _terra_messages(), response_format={"type": "json_object"}
            )
        receipt = transport.transport_receipt()
        assert receipt["attempted"] == receipt["completed"] == 1
        assert receipt["provider_usage_records"] == 0
        with pytest.raises(ProductionBindingError, match="usage records"):
            transport.assert_call_budget_closed()
    finally:
        transport.close()


def test_bound_memory_cleanup_rejects_an_incomplete_http_budget(monkeypatch) -> None:
    import httpx
    from tools.mem0_eval import production_binding as binding

    _patch_http_transport(
        monkeypatch,
        lambda _request: httpx.Response(
            200,
            json=_completion_response('{"memory":[{"text":"fact"}]}'),
        ),
    )

    class OldClient:
        def close(self) -> None:
            return None

    OpenAILLM = type("OpenAILLM", (), {})
    OpenAILLM.__module__ = "mem0.llms.openai"
    old_llm = OpenAILLM()
    old_llm.client = OldClient()
    memory = SimpleNamespace(llm=old_llm)
    transport = LiteLLMTerraExtractionTransport(authorized=1)
    binding._bind_memory_transport(memory, transport)

    with pytest.raises(ProductionBindingError, match="attempt accounting"):
        memory.close()
    assert transport._client is None


def test_terra_extraction_canary_returns_only_a_hashed_payload_receipt(
    monkeypatch,
) -> None:
    import httpx

    def handler(_request):
        return httpx.Response(
            200,
            json=_completion_response(
                '{"memory":[{"text":"Alice likes blue."}]}'
            ),
        )

    _patch_http_transport(monkeypatch, handler)
    receipt = dict(run_terra_extraction_canary())

    assert receipt["memory_count"] == 1
    assert receipt["transport"]["attempted"] == 1
    assert receipt["transport"]["completed"] == 1
    assert "Alice" not in repr(receipt)
    body = dict(receipt)
    supplied = body.pop("receipt_sha256")
    assert supplied == canonical_json_sha256(body)


def test_exact_mem0_factory_materializes_owned_paths_and_replaces_default_llm(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import httpx
    from memory_condense.eval import mem0_adapter
    from tools.mem0_eval import production_binding as binding

    _patch_http_transport(
        monkeypatch,
        lambda _request: httpx.Response(
            200,
            json=_completion_response(
                '{"memory":[{"text":"factory fact"}]}'
            ),
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "original-openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "original-openrouter-key")
    events: list[str] = []
    captured: dict[str, Any] = {}

    class RuntimeModel:
        device = "cuda:0"

        @staticmethod
        def get_sentence_embedding_dimension() -> int:
            return DEFAULT_MODEL_DIM

    HuggingFaceEmbedding = type("HuggingFaceEmbedding", (), {})
    HuggingFaceEmbedding.__module__ = "mem0.embeddings.huggingface"
    embedder = HuggingFaceEmbedding()
    embedder.config = SimpleNamespace(
        model=DEFAULT_MODEL_NAME,
        embedding_dims=DEFAULT_MODEL_DIM,
        huggingface_base_url=None,
        model_kwargs={
            "revision": DEFAULT_MODEL_REVISION,
            "local_files_only": True,
            "trust_remote_code": False,
            "device": "cuda",
        },
    )
    embedder.model = RuntimeModel()

    class OldClient:
        def close(self) -> None:
            events.append("old_llm_client_closed")

    OpenAILLM = type("OpenAILLM", (), {})
    OpenAILLM.__module__ = "mem0.llms.openai"
    old_llm = OpenAILLM()
    old_llm.client = OldClient()
    memory = SimpleNamespace(llm=old_llm, embedding_model=embedder)

    class Owned:
        backend = memory

        def close(self) -> None:
            events.append("owned_close")
            close = getattr(self.backend, "close", None)
            if callable(close):
                close()

    owned = Owned()

    monkeypatch.setattr(
        binding,
        "_harden_owned_qdrant_cleanup",
        lambda value: {
            "format": "test-qdrant-cleanup",
            "owned_identity_preserved": value is owned,
        },
    )
    monkeypatch.setattr(
        binding,
        "_materialize_exact_qdrant_stores",
        lambda value: (object(), object())
        if value is memory
        else pytest.fail("unexpected Memory object"),
    )
    monkeypatch.setattr(
        binding,
        "_bind_exact_bm25_encoders",
        lambda value: {
            "format": "memory-condense-bound-mem0-bm25-v1",
            "memory_identity_preserved": value is memory,
            "receipt_sha256": SHA_A,
        },
    )

    class FakeBackendFactory:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def __call__(self):
            assert os.environ["OPENAI_API_KEY"] == (
                "mem0-constructor-only-no-network"
            )
            assert "OPENROUTER_API_KEY" not in os.environ
            return owned

    monkeypatch.setattr(mem0_adapter, "Mem0OSSBackendFactory", FakeBackendFactory)
    authorization = _retrieval_authorization(device="cuda")
    factory = ExactMem0AdapterFactory(authorization)
    state = tmp_path / "owned-state"

    adapter = factory(state)

    assert captured["owned_state_dir"] == state.resolve()
    assert captured["llm_model_id"] == MEM0_EXTRACTION_MODEL
    assert captured["embedder_model_id"] == DEFAULT_MODEL_NAME
    assert captured["_stack_preflight"] is binding._exact_mem0_stack_preflight
    assert captured["config"]["history_db_path"] == str(
        state.resolve() / "history.sqlite"
    )
    assert captured["config"]["vector_store"]["config"]["path"] == str(
        state.resolve() / "qdrant"
    )
    assert type(memory.llm) is LiteLLMTerraExtractionTransport
    assert events == ["old_llm_client_closed"]
    assert os.environ["OPENAI_API_KEY"] == "original-openai-key"
    assert os.environ["OPENROUTER_API_KEY"] == "original-openrouter-key"
    receipt = dict(factory.binding_receipt())
    assert receipt["bound_embedder"]["model"] == DEFAULT_MODEL_NAME
    assert receipt["bound_cleanup"]["owned_identity_preserved"] is True
    assert receipt["bound_bm25"]["memory_identity_preserved"] is True
    assert receipt["transport"]["authorized"] == 2
    with pytest.raises(ProductionBindingError, match="single-use"):
        factory(tmp_path / "other-state")

    for _index in range(2):
        assert json.loads(
            memory.llm.generate_response(
                _terra_messages(),
                response_format={"type": "json_object"},
            )
        )["memory"]
    adapter.cleanup()
    assert events == ["old_llm_client_closed", "owned_close"]
    assert memory.llm._client is None
    assert factory.binding_receipt()["transport"]["completed"] == 2


def test_exact_mem0_stack_preflight_executes_pinned_local_only_probes(
    monkeypatch,
) -> None:
    from tools.mem0_eval import production_binding as binding

    expected_versions = {
        "mem0ai": "2.0.18",
        "qdrant-client": "1.15.1",
        "fastembed": "0.7.3",
        "spacy": "3.8.7",
        "en-core-web-sm": "3.8.0",
        "click": "8.4.2",
    }
    class ProbeDoc(list):
        ents = [object()]

    class NLP:
        pipe_names = ("ner", "lemmatizer")

        def __call__(self, text: str) -> ProbeDoc:
            assert "Seattle" in text
            return ProbeDoc([SimpleNamespace(lemma_="visit")])

    class SpacyModel:
        @staticmethod
        def load() -> NLP:
            return NLP()

    def import_module(name: str) -> Any:
        if name == "en_core_web_sm":
            return SpacyModel
        pytest.fail(f"unexpected module import {name!r}")

    monkeypatch.setattr(
        binding.importlib.metadata,
        "version",
        lambda name: expected_versions[name],
    )
    monkeypatch.setattr(binding.importlib, "import_module", import_module)
    monkeypatch.setattr(
        binding,
        "_new_exact_bm25_encoder",
        lambda: (SimpleNamespace(), {"model": "Qdrant/bm25"}),
    )

    identity = binding._exact_mem0_stack_preflight()

    assert dict(identity.dependency_versions) == {
        key: expected_versions[key]
        for key in (
            "mem0ai",
            "qdrant-client",
            "fastembed",
            "spacy",
            "en-core-web-sm",
        )
    }
    assert identity.certified is True


def test_exact_mem0_stack_preflight_rejects_dependency_drift_before_probe(
    monkeypatch,
) -> None:
    from tools.mem0_eval import production_binding as binding

    monkeypatch.setattr(
        binding.importlib.metadata,
        "version",
        lambda name: "0.7.4" if name == "fastembed" else {
            "mem0ai": "2.0.18",
            "qdrant-client": "1.15.1",
        }.get(name, "unreached"),
    )
    monkeypatch.setattr(
        binding.importlib,
        "import_module",
        lambda _name: pytest.fail("version drift must fail before model import"),
    )

    with pytest.raises(ProductionBindingError, match="fastembed"):
        binding._exact_mem0_stack_preflight()


def test_exact_bm25_encoder_uses_verified_cache_and_local_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from tools.mem0_eval import production_binding as binding

    cache_root = tmp_path / "fastembed-cache"
    snapshot = cache_root / "snapshot"
    snapshot.mkdir(parents=True)
    seen: dict[str, Any] = {}

    class SparseTextEmbedding:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)
            self.model = SimpleNamespace(_model_dir=str(snapshot))

        @staticmethod
        def embed(rows: list[str]) -> list[SimpleNamespace]:
            assert rows == ["memory retrieval operational probe"]
            return [SimpleNamespace(indices=[1], values=[1.0])]

    cache_receipt = {
        "model": "Qdrant/bm25",
        "revision": "revision",
        "asset_tree_sha256": SHA_A,
    }
    model_management = SimpleNamespace(time=__import__("time"))
    monkeypatch.setattr(
        binding,
        "_verified_fastembed_bm25_cache",
        lambda: (cache_root, snapshot, cache_receipt),
    )
    def import_module(name: str) -> Any:
        modules = {
            "fastembed": SimpleNamespace(
                SparseTextEmbedding=SparseTextEmbedding
            ),
            "fastembed.common.model_management": model_management,
        }
        if name not in modules:
            pytest.fail(f"unexpected module import {name!r}")
        return modules[name]

    monkeypatch.setattr(binding.importlib, "import_module", import_module)

    encoder, receipt = binding._new_exact_bm25_encoder()

    assert isinstance(encoder, SparseTextEmbedding)
    assert receipt == {
        **cache_receipt,
        "specific_model_path": snapshot.as_posix(),
        "retry_sleep_attempts": 0,
    }
    assert seen == {
        "model_name": "Qdrant/bm25",
        "cache_dir": str(cache_root),
        "local_files_only": True,
        "specific_model_path": str(snapshot),
    }


def test_exact_bm25_encoder_rejects_internal_retry_sleep(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from tools.mem0_eval import production_binding as binding

    cache_root = tmp_path / "fastembed-cache"
    snapshot = cache_root / "snapshot"
    snapshot.mkdir(parents=True)
    real_time = __import__("time")
    model_management = SimpleNamespace(time=real_time)

    class SparseTextEmbedding:
        def __init__(self, **_kwargs: Any) -> None:
            model_management.time.sleep(3)

    monkeypatch.setattr(
        binding,
        "_verified_fastembed_bm25_cache",
        lambda: (cache_root, snapshot, {"model": "Qdrant/bm25"}),
    )
    def import_module(name: str) -> Any:
        modules = {
            "fastembed": SimpleNamespace(
                SparseTextEmbedding=SparseTextEmbedding
            ),
            "fastembed.common.model_management": model_management,
        }
        if name not in modules:
            pytest.fail(f"unexpected module import {name!r}")
        return modules[name]

    monkeypatch.setattr(binding.importlib, "import_module", import_module)

    with pytest.raises(ProductionBindingError, match="retry sleep"):
        binding._new_exact_bm25_encoder()
    assert model_management.time is real_time


def test_blocked_network_probe_nests_and_restores_only_after_outer_exit() -> None:
    import socket

    from tools.mem0_eval import production_binding as binding

    original_create = socket.create_connection
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    with binding._blocked_network_probe() as outer_attempts:
        outer_create = socket.create_connection
        assert outer_create is not original_create
        with binding._blocked_network_probe() as inner_attempts:
            assert socket.create_connection is not outer_create
            with pytest.raises(ProductionBindingError, match="forbidden network"):
                socket.create_connection(("nested.invalid", 443))
        assert len(inner_attempts) == 1
        assert not outer_attempts
        assert socket.create_connection is outer_create
        with pytest.raises(ProductionBindingError, match="forbidden network"):
            socket.socket().connect(("outer.invalid", 443))
        assert len(outer_attempts) == 1

    assert socket.create_connection is original_create
    assert socket.socket.connect is original_connect
    assert socket.socket.connect_ex is original_connect_ex


def test_exact_bm25_binding_uses_two_distinct_verified_instances(
    monkeypatch,
) -> None:
    from tools.mem0_eval import production_binding as binding

    Qdrant = type("Qdrant", (), {})
    Qdrant.__module__ = "mem0.vector_stores.qdrant"
    memory_store = Qdrant()
    entity_store = Qdrant()
    for store in (memory_store, entity_store):
        store._has_bm25_slot = True
        store._bm25_encoder = None
    memory = SimpleNamespace(
        vector_store=memory_store,
        entity_store=entity_store,
    )
    created: list[Any] = []
    cache_receipt = {
        "model": "Qdrant/bm25",
        "revision": "revision",
        "asset_tree_sha256": SHA_A,
        "cache_root": "cache",
        "file_count": 18,
        "local_files_only": True,
        "network_calls_authorized": 0,
    }

    def new_encoder() -> tuple[Any, dict[str, Any]]:
        encoder = object()
        created.append(encoder)
        return encoder, dict(cache_receipt)

    monkeypatch.setattr(binding, "_new_exact_bm25_encoder", new_encoder)

    receipt = binding._bind_exact_bm25_encoders(memory)

    assert len(created) == 2
    assert created[0] is not created[1]
    assert memory_store._bm25_encoder is created[0]
    assert entity_store._bm25_encoder is created[1]
    assert receipt["encoder_instances"] == 2
    assert receipt["bound_store_roles"] == ["memory", "entity"]
    assert receipt["internal_lazy_download_path_reachable"] is False


def test_factory_canary_is_synthetic_text_free_and_closes_exact_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from tools.mem0_eval import production_binding as binding

    captured: dict[str, Any] = {}

    class Adapter:
        cleaned = False

        @staticmethod
        def ingest_longmemeval_record(record: Mapping[str, Any]) -> Any:
            captured["record"] = record
            return SimpleNamespace(
                returned_memory_ids=("memory-id",),
                comparison_certified=True,
            )

        @staticmethod
        def search(query: str, **kwargs: Any) -> Any:
            captured["search_query"] = query
            captured["search_kwargs"] = kwargs
            return SimpleNamespace(
                raw_pool=(object(),),
                packed=(object(),),
                context="Maya's preferred tea is oolong.",
            )

        def cleanup(self) -> None:
            self.cleaned = True

    adapter = Adapter()

    class Factory:
        def __init__(self, authorization: Any) -> None:
            captured["authorization"] = authorization

        def __call__(self, state: Path) -> Adapter:
            captured["state"] = state
            return adapter

        @staticmethod
        def binding_receipt() -> Mapping[str, Any]:
            return {
                "kind": "exact_mem0_adapter_factory_v1",
                "transport": {
                    "authorized": 1,
                    "attempted": 1,
                    "completed": 1,
                    "failed": 0,
                    "rejected": 0,
                },
            }

    monkeypatch.setattr(binding, "ExactMem0AdapterFactory", Factory)
    receipt = dict(
        run_mem0_factory_canary(owned_state_dir=tmp_path / "canary-state")
    )

    authorization = captured["authorization"]
    assert authorization.authorized_add_operations == 1
    assert authorization.authorized_extraction_calls == 1
    assert authorization.authorized_search_operations == 1
    assert captured["record"]["question_id"] == "mem0-factory-canary-v2"
    assert captured["record"]["haystack_sessions"][0][0]["content"] == (
        "Maya's preferred tea is oolong."
    )
    assert captured["record"]["haystack_dates"] == ["2026-08-29 00:00"]
    assert captured["search_query"] == "Which tea does Maya prefer?"
    assert receipt["campaign_authority"] is False
    assert receipt["returned_memory_count"] == 1
    assert receipt["raw_search_pool_count"] == 1
    assert receipt["owned_state_removed"] is True
    assert adapter.cleaned is True
    assert "Maya" not in repr(receipt)
    assert "oolong" not in repr(receipt)
    body = dict(receipt)
    supplied = body.pop("receipt_sha256")
    assert supplied == canonical_json_sha256(body)


def test_factory_canary_rejects_preexisting_state_before_factory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from tools.mem0_eval import production_binding as binding

    state = tmp_path / "existing"
    state.mkdir()
    monkeypatch.setattr(
        binding,
        "ExactMem0AdapterFactory",
        lambda _authorization: pytest.fail("factory must not be constructed"),
    )

    with pytest.raises(ProductionBindingError, match="already exists"):
        run_mem0_factory_canary(owned_state_dir=state)


def _fake_qdrant_cleanup_store(local: Any) -> Any:
    QdrantClient = type("QdrantClient", (), {"_client": local})
    QdrantClient.__module__ = "qdrant_client.qdrant_client"
    Qdrant = type("Qdrant", (), {"client": QdrantClient()})
    Qdrant.__module__ = "mem0.vector_stores.qdrant"
    return Qdrant()


def _fake_qdrant_local(collections: Mapping[str, Any]) -> Any:
    QdrantLocal = type("QdrantLocal", (), {"collections": dict(collections)})
    QdrantLocal.__module__ = "qdrant_client.local.qdrant_local"
    return QdrantLocal()


def _fake_local_collection(events: list[str], label: str) -> Any:
    LocalCollection = type(
        "LocalCollection",
        (),
        {"close": lambda _self: events.append(f"{label}_close")},
    )
    LocalCollection.__module__ = "qdrant_client.local.local_collection"
    return LocalCollection()


def test_exact_qdrant_cleanup_closes_shared_registry_before_delete() -> None:
    from tools.mem0_eval import production_binding as binding

    events: list[str] = []
    local = _fake_qdrant_local(
        {
            "memory": _fake_local_collection(events, "memory"),
            "entity": _fake_local_collection(events, "entity"),
        }
    )
    memory = SimpleNamespace(
        vector_store=_fake_qdrant_cleanup_store(local),
        _entity_store=_fake_qdrant_cleanup_store(local),
    )

    class Owned:
        backend = memory

        def close(self) -> None:
            events.append("owned_close")

    owned = Owned()
    receipt = dict(binding._harden_owned_qdrant_cleanup(owned))
    owned.close()

    assert events == ["memory_close", "entity_close", "owned_close"]
    assert receipt == {
        "format": "memory-condense-owned-qdrant-cleanup-v1",
        "qdrant_client_version": "1.15.1",
        "initial_local_clients_bound": 1,
        "initial_collection_handles_bound": 2,
        "dynamic_store_and_collection_registries_bound": True,
        "collection_handles_preclosed_before_delete": True,
    }


def test_exact_qdrant_cleanup_closes_distinct_registries_before_delete() -> None:
    from tools.mem0_eval import production_binding as binding

    events: list[str] = []
    memory_local = _fake_qdrant_local(
        {"memory": _fake_local_collection(events, "memory")}
    )
    entity_local = _fake_qdrant_local(
        {"entity": _fake_local_collection(events, "entity")}
    )
    memory = SimpleNamespace(
        vector_store=_fake_qdrant_cleanup_store(memory_local),
        _entity_store=_fake_qdrant_cleanup_store(entity_local),
    )

    class Owned:
        backend = memory

        def close(self) -> None:
            events.append("owned_close")

    owned = Owned()
    receipt = dict(binding._harden_owned_qdrant_cleanup(owned))
    owned.close()

    assert events == ["memory_close", "entity_close", "owned_close"]
    assert receipt["initial_local_clients_bound"] == 2
    assert receipt["initial_collection_handles_bound"] == 2


def test_factory_presend_cleanup_canary_proves_all_local_handles_closed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from tools.mem0_eval import production_binding as binding

    history = SimpleNamespace(connection=object())
    memory = SimpleNamespace(db=history)
    local = SimpleNamespace(closed=False)
    owned = SimpleNamespace(backend=memory)

    class Adapter:
        _backend = owned

        @staticmethod
        def cleanup() -> None:
            history.connection = None
            memory.db = None
            local.closed = True
            raise ProductionBindingError(
                "extraction HTTP attempt accounting did not close"
            )

    class Factory:
        def __init__(self, _authorization: Any) -> None:
            return None

        @staticmethod
        def __call__(_state: Path) -> Adapter:
            return Adapter()

        @staticmethod
        def binding_receipt() -> Mapping[str, Any]:
            return {
                "kind": "exact_mem0_adapter_factory_v1",
                "transport": {
                    "authorized": 1,
                    "attempted": 0,
                    "completed": 0,
                    "failed": 0,
                    "rejected": 0,
                },
            }

    topology = {
        "format": "memory-condense-mem0-cleanup-topology-v1",
        "qdrant_store_count": 2,
        "distinct_local_qdrant_client_count": 1,
        "entity_store_materialized": True,
        "history_connection_live_before_cleanup": True,
        "graph_store_absent": True,
        "telemetry_store_absent": True,
    }
    monkeypatch.setattr(binding, "ExactMem0AdapterFactory", Factory)
    monkeypatch.setattr(
        binding,
        "_capture_exact_factory_cleanup_topology",
        lambda adapter: (memory, history, (local,), topology)
        if isinstance(adapter, Adapter)
        else pytest.fail("unexpected adapter"),
    )

    receipt = dict(
        binding.run_mem0_factory_presend_cleanup_canary(
            owned_state_dir=tmp_path / "provider-free-state"
        )
    )

    assert receipt["network_attempts"] == 0
    assert receipt["unused_transport_budget_rejected"] is True
    assert receipt["owned_state_removed"] is True
    assert receipt["history_connection_closed"] is True
    assert receipt["all_local_qdrant_clients_closed"] is True
    assert receipt["topology"] == topology
    body = dict(receipt)
    supplied = body.pop("receipt_sha256")
    assert supplied == canonical_json_sha256(body)


def test_factory_canary_preserves_operation_and_incomplete_budget_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from tools.mem0_eval import production_binding as binding

    class Adapter:
        @staticmethod
        def ingest_longmemeval_record(_record: Mapping[str, Any]) -> Any:
            raise ValueError("synthetic pre-send failure")

        @staticmethod
        def cleanup() -> None:
            raise ProductionBindingError(
                "extraction HTTP attempt accounting did not close"
            )

    class Factory:
        def __init__(self, _authorization: Any) -> None:
            return None

        @staticmethod
        def __call__(_state: Path) -> Adapter:
            return Adapter()

    monkeypatch.setattr(binding, "ExactMem0AdapterFactory", Factory)

    with pytest.raises(BaseExceptionGroup) as raised:
        run_mem0_factory_canary(
            owned_state_dir=tmp_path / "pre-send-failure-state"
        )

    assert [type(exc) for exc in raised.value.exceptions] == [
        ValueError,
        ProductionBindingError,
    ]
    assert "pre-send failure" in str(raised.value.exceptions[0])
    assert "attempt accounting" in str(raised.value.exceptions[1])


def test_runtime_probe_has_no_injectable_loader_or_verifier_seam() -> None:
    assert set(inspect.signature(probe_local_bge_m3_runtime).parameters) == {
        "authorization",
        "model_dir",
    }


def test_direct_trusted_binding_construction_remains_impossible() -> None:
    with pytest.raises(TypeError, match="direct construction"):
        TrustedRuntimeBinding()
