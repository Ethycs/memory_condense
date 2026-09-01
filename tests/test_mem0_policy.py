from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from memory_condense.domain._tokenizer import tokenizer_proxy_identity
from memory_condense.eval.sample_identity import sample_sha256
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample
from tools.mem0_eval.policy import (
    MEM0_EMBEDDER_CHECKPOINT_SHA256,
    MEM0_EMBEDDER_REVISION,
    MEM0_EXTRACTION_MODEL,
    MEM0_EXTRACTION_PROVIDER,
    MEM0_EXTRACTION_REVISION,
    MEM0_EXTRACTION_ROUTE_IDENTITY,
    MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
    MEM0_EXTRACTION_USAGE_POLICY,
    MEM0_EXTRACTION_BOUNDARY,
    MEM0_INGESTION_POLICY,
    MEM0_LOCKED_POPULATION_EXPECTATIONS,
    MEM0_NEUTRAL_FALLBACK,
    MEM0_POLICY_FORMAT,
    MEM0_POLICY_STATUS,
    MEM0_PRODUCTION_CANDIDATE_FORMAT,
    MEM0_PRODUCTION_INELIGIBLE,
    MEM0_STANDALONE_ANSWER_POLICY,
    Mem0PolicyError,
    canonical_json_sha256,
    expected_shard_policy_rows,
    inspect_mem0_comparison_policy,
    load_mem0_comparison_policy,
    observed_campaign_population,
)
from tools.mem0_eval.preflight import (
    SourceValidationPlan,
    tool_implementation_sha256,
)
from tools.mem0_eval.protocol import (
    RawStressShard,
    CompositeAddBatch,
    build_composite_add_batches,
    compose_raw_stress_record,
    count_official_add_requests,
)
from memory_condense.eval.sample_identity import canonical_sha256


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _source_identity() -> dict[str, Any]:
    return {
        "responder_model": "openai/codex_sdk/gpt-5.6-terra",
        "judge_model": "openai/codex_sdk/gpt-5.6-sol",
        "use_judge": True,
        "provider_retries": 0,
        "max_provider_calls_per_shard": 20,
        "max_prompt_tokens": 8_000,
        "prompt_cap_semantics": (
            "local_prompt_token_proxy_with_provider_usage_postcheck_v1"
        ),
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "responder_output_token_reserve": 256,
        "recent_window": 4,
        "accuracy_target": 0.95,
        "min_target_questions": 100,
        "stress_context_tokens": 1_000_000,
        "stress_questions": 10,
        "stress_question_offset": 0,
        "max_samples": 1,
        "sample_offsets": list(range(0, 100, 10)),
    }


def _source_plan() -> SourceValidationPlan:
    return SourceValidationPlan(
        dataset_sha256=_digest("dataset"),
        split_manifest_sha256=_digest("split"),
        policy_manifest_sha256=_digest("source-policy"),
        implementation_sha256=_digest("source-code"),
        environment_lock_sha256=_digest("source-lock"),
        sample_offsets=tuple(range(0, 100, 10)),
        target_tokens=1_000_000,
        questions_per_shard=10,
        evaluation_identity=_source_identity(),
    )


def _shards() -> tuple[RawStressShard, ...]:
    rows: list[RawStressShard] = []
    for offset in range(0, 100, 10):
        questions = [
            BenchmarkQuestion(
                question_id=f"q-{offset:03d}-{index}",
                question=f"Question {index}?",
                answer=f"Answer {index}",
            )
            for index in range(10)
        ]
        sample = BenchmarkSample(
            sample_id=f"stress-{offset:03d}",
            turns=[("user", f"history at offset {offset}")],
            turn_source_ids=[f"source-{offset}"],
            questions=questions,
        )
        history_id = f"history-{offset}"
        raw_record = {
            "question_id": history_id,
            "haystack_sessions": [
                [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                    {"role": "user", "content": "third"},
                    {"role": "assistant", "content": "fourth"},
                    {"role": "user", "content": "fifth"},
                    {"role": "assistant", "content": "sixth"},
                ]
            ],
            "haystack_session_ids": ["session-1"],
            "haystack_dates": ["2025-01-01"],
        }
        raw_bundle = compose_raw_stress_record(
            [raw_record], sample_id=f"mem0-stress-{offset:03d}"
        )
        rows.append(
            RawStressShard(
                sample_offset=offset,
                parsed_sample=sample,
                sample_sha256=sample_sha256(sample),
                history_sample_ids=(history_id,),
                raw_history_bundle=raw_bundle,
                raw_history_bundle_sha256=canonical_sha256(raw_bundle),
                add_batches=build_composite_add_batches([raw_record]),
                add_counts=count_official_add_requests([raw_record]),
            )
        )
    return tuple(rows)


def _policy_payload(
    *,
    plan: SourceValidationPlan,
    shards: tuple[RawStressShard, ...],
    lock_digest: str,
) -> dict[str, Any]:
    extraction_without_hash = {
        "provider": MEM0_EXTRACTION_PROVIDER,
        "model": MEM0_EXTRACTION_MODEL,
        "revision": MEM0_EXTRACTION_REVISION,
        "provider_retries": 0,
        "logical_call_boundary": MEM0_EXTRACTION_BOUNDARY,
        "logical_calls_per_add": 1,
        "http_attempts_certified": False,
    }
    extraction = {
        **extraction_without_hash,
        "model_identity_sha256": canonical_json_sha256(extraction_without_hash),
    }
    embedder_without_hash = {
        "provider": "huggingface",
        "model": "BAAI/bge-m3",
        "revision": MEM0_EMBEDDER_REVISION,
        "checkpoint_sha256": MEM0_EMBEDDER_CHECKPOINT_SHA256,
        "dimension": 1024,
        "device": "cuda",
        "dtype": "float32",
        "execution": "local_offline",
        "network_calls_authorized": 0,
        "runtime_probe_required": True,
    }
    embedder = {
        **embedder_without_hash,
        "model_identity_sha256": canonical_json_sha256(embedder_without_hash),
    }
    stack = {
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
    }
    stable_payload = {
        "protocol": "mem0-oss-2.0.18-certified-local-v1",
        "config": {
            "version": "v1.1",
            "llm": {
                "provider": extraction["provider"],
                "config": {"model": extraction["model"]},
            },
            "embedder": {
                "provider": embedder["provider"],
                "config": {
                    "model": embedder["model"],
                    "embedding_dims": embedder["dimension"],
                    "huggingface_base_url": None,
                    "model_kwargs": {
                        "revision": embedder["revision"],
                        "local_files_only": True,
                        "trust_remote_code": False,
                        "device": embedder["device"],
                    },
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "longmemeval",
                    "embedding_model_dims": 1024,
                    "on_disk": True,
                    "path": "<owned_state>/qdrant",
                },
            },
            "history_db_path": "<owned_state>/history.sqlite",
            "custom_instructions": None,
            "reranker": None,
        },
        "stack": stack,
    }
    responder = {
        "provider": "central-dev",
        "model": plan.evaluation_identity["responder_model"],
    }
    judge = {
        "provider": "central-dev",
        "model": plan.evaluation_identity["judge_model"],
    }
    return {
        "format": MEM0_POLICY_FORMAT,
        "status": MEM0_POLICY_STATUS,
        "arm_id": "mem0_oss_2_0_18_direct_1m_v1",
        "source": {
            "validation_policy_sha256": plan.policy_manifest_sha256,
            "dataset_sha256": plan.dataset_sha256,
            "split_manifest_sha256": plan.split_manifest_sha256,
            "implementation_sha256": plan.implementation_sha256,
            "environment_lock_sha256": plan.environment_lock_sha256,
            "evaluation_identity": copy.deepcopy(plan.evaluation_identity),
            "evaluation_identity_sha256": canonical_json_sha256(
                plan.evaluation_identity
            ),
        },
        "tool": {
            "implementation_sha256": tool_implementation_sha256(),
            "environment_lock_sha256": lock_digest,
        },
        "mem0": {
            "runtime_protocol": "mem0-oss-2.0.18-certified-local-v1",
            "mem0ai_version": "2.0.18",
            "api_version": "v1.1",
            "input_order_protocol": (
                "locked-record-order+official-within-record-date-sort+"
                "consecutive-1-or-2-turn-slices-v1"
            ),
            "extraction_identity": extraction,
            "embedder_identity": embedder,
            "search": {
                "top_k": 200,
                "threshold": 0.1,
                "rerank": False,
                "explain": False,
            },
            "rendering_mode": "official-memory-text-created-at",
            "storage": {
                "provider": "qdrant",
                "local_owned_state": True,
                "on_disk": True,
                "fresh_process_per_shard": True,
                "resumable_across_fresh_processes": True,
                "resume_requires_closed_owned_state": True,
                "immutable_prefix_checkpoint_adds": 256,
                "terminal_output_before_checkpoint_gc": True,
                "cleanup_required": True,
            },
            "provenance": {
                "attribution_kind": "request_window_non_evidence",
                "supports_exact_source_provenance": False,
                "source_session_date_exposure": (
                    "diagnostics_only_not_model_input"
                ),
                "retrieved_created_at_exposure": (
                    "answer_prompt_date_headings"
                ),
            },
            "stable_payload": stable_payload,
            "stable_config_sha256": canonical_json_sha256(stable_payload),
        },
        "production_candidate": {
            "format": MEM0_PRODUCTION_CANDIDATE_FORMAT,
            "status": MEM0_PRODUCTION_INELIGIBLE,
            "unresolved_required_fields": [],
            "blockers": ["locked_population_mismatch"],
            "standalone_answer": copy.deepcopy(
                MEM0_STANDALONE_ANSWER_POLICY
            ),
            "ingestion": copy.deepcopy(MEM0_INGESTION_POLICY),
            "population_expectations": copy.deepcopy(
                MEM0_LOCKED_POPULATION_EXPECTATIONS
            ),
            "extraction_usage": copy.deepcopy(
                MEM0_EXTRACTION_USAGE_POLICY
            ),
        },
        "scoring": {
            "responder_identity": responder,
            "responder_identity_sha256": canonical_json_sha256(responder),
            "judge_identity": judge,
            "judge_identity_sha256": canonical_json_sha256(judge),
            "responder_calls_per_shard": 10,
            "judge_calls_per_shard": 10,
            "provider_retries": 0,
            "max_prompt_tokens": 8_000,
            "responder_max_output_tokens": 256,
            "judge_max_output_tokens": 1_024,
        },
        "shards": list(expected_shard_policy_rows(shards)),
    }


@pytest.fixture
def policy_fixture(tmp_path: Path) -> tuple[Path, Path, SourceValidationPlan, tuple[RawStressShard, ...], dict[str, Any]]:
    lock = tmp_path / "pixi.lock"
    lock.write_text("isolated-lock\n", encoding="utf-8")
    plan = _source_plan()
    shards = _shards()
    payload = _policy_payload(
        plan=plan,
        shards=shards,
        lock_digest=hashlib.sha256(lock.read_bytes()).hexdigest(),
    )
    path = tmp_path / "mem0-policy.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, lock, plan, shards, payload


def _inspect(
    fixture: tuple[
        Path,
        Path,
        SourceValidationPlan,
        tuple[RawStressShard, ...],
        dict[str, Any],
    ],
):
    path, lock, plan, shards, _payload = fixture
    return inspect_mem0_comparison_policy(
        path,
        source_plan=plan,
        mem0_environment_lock=lock,
        expected_shards=shards,
    )


def test_policy_freezes_exact_route_but_preserves_population_gate(policy_fixture):
    policy = _inspect(policy_fixture)
    shard = policy_fixture[3][0]

    assert policy.production_eligible is False
    assert policy.production_candidate.unresolved_required_fields == ()
    assert policy.production_candidate.blockers == ("locked_population_mismatch",)
    assert policy.extraction_identity["provider"] == MEM0_EXTRACTION_PROVIDER
    assert policy.extraction_identity["model"] == MEM0_EXTRACTION_MODEL
    assert policy.extraction_identity["revision"] == MEM0_EXTRACTION_REVISION
    assert dict(policy.production_candidate.population_expectations) == (
        MEM0_LOCKED_POPULATION_EXPECTATIONS
    )
    assert dict(policy.production_candidate.observed_population) == {
        "namespace_count": 10,
        "add_operations": 30,
        "extraction_calls": 30,
        "search_operations": 100,
        "questions": 100,
    }
    assert policy.production_candidate.standalone_answer[
        "fallback_prediction"
    ] == MEM0_NEUTRAL_FALLBACK
    with pytest.raises(Mem0PolicyError, match="ineligible"):
        policy.retrieval_authorization(shard)
    with pytest.raises(Mem0PolicyError, match="ineligible"):
        policy.scoring_authorization(
            shard, retrieval_artifact_sha256=_digest("retrieval")
        )
    with pytest.raises(Mem0PolicyError, match="ineligible"):
        load_mem0_comparison_policy(
            policy_fixture[0],
            source_plan=policy_fixture[2],
            mem0_environment_lock=policy_fixture[1],
            expected_shards=policy_fixture[3],
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (
            ("standalone_answer", "prediction_inputs"),
            ["question", "mem0_retrieved_memories", "parent_prediction"],
        ),
        (("standalone_answer", "external_predictions_authorized"), 1),
        (("standalone_answer", "fallback_prediction"), "copied answer"),
        (("ingestion", "infer"), False),
        (("population_expectations", "add_operations"), 30),
        (("extraction_usage", "silent_zero_fill_authorized"), True),
        (
            ("extraction_usage", "missing_provider_usage_semantics"),
            "zero_tokens_zero_cost",
        ),
    ],
)
def test_production_candidate_rejects_hybrid_or_cost_contract_drift(
    policy_fixture, path: tuple[str, ...], value: Any
) -> None:
    policy_path, lock, plan, shards, payload = policy_fixture
    target = payload["production_candidate"]
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    policy_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Mem0PolicyError, match="frozen contract"):
        inspect_mem0_comparison_policy(
            policy_path,
            source_plan=plan,
            mem0_environment_lock=lock,
            expected_shards=shards,
        )


def test_production_candidate_rejects_embedded_parent_prediction(
    policy_fixture,
) -> None:
    policy_path, lock, plan, shards, payload = policy_fixture
    payload["production_candidate"]["standalone_answer"][
        "parent_prediction"
    ] = "prediction copied from our treatment"
    policy_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Mem0PolicyError, match="hybridize"):
        inspect_mem0_comparison_policy(
            policy_path,
            source_plan=plan,
            mem0_environment_lock=lock,
            expected_shards=shards,
        )


@pytest.mark.parametrize(
    ("field", "substitute"),
    [
        ("provider", "attacker-provider"),
        ("model", "attacker/model"),
        ("revision", "observable-route-sha256:" + "f" * 64),
    ],
)
def test_policy_rejects_fully_rehashed_extraction_substitution(
    policy_fixture, field: str, substitute: str
) -> None:
    policy_path, lock, plan, shards, payload = policy_fixture
    identity = payload["mem0"]["extraction_identity"]
    identity[field] = substitute
    body = dict(identity)
    body.pop("model_identity_sha256")
    identity["model_identity_sha256"] = canonical_json_sha256(body)
    stable = payload["mem0"]["stable_payload"]
    if field == "provider":
        stable["config"]["llm"]["provider"] = substitute
    elif field == "model":
        stable["config"]["llm"]["config"]["model"] = substitute
    payload["mem0"]["stable_config_sha256"] = canonical_json_sha256(stable)
    policy_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        Mem0PolicyError,
        match=rf"{field} does not match the frozen Terra extraction route",
    ):
        inspect_mem0_comparison_policy(
            policy_path,
            source_plan=plan,
            mem0_environment_lock=lock,
            expected_shards=shards,
        )


def test_production_candidate_cannot_claim_eligibility_with_population_mismatch(
    policy_fixture,
) -> None:
    policy_path, lock, plan, shards, payload = policy_fixture
    payload["production_candidate"]["status"] = (
        "eligible_for_production_execution"
    )
    policy_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Mem0PolicyError, match="contradicts derived eligibility"):
        inspect_mem0_comparison_policy(
            policy_path,
            source_plan=plan,
            mem0_environment_lock=lock,
            expected_shards=shards,
        )


def test_observable_route_revision_is_the_canonical_code_owned_route() -> None:
    assert canonical_json_sha256(dict(MEM0_EXTRACTION_ROUTE_IDENTITY)) == (
        MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256
    )
    assert MEM0_EXTRACTION_REVISION == (
        f"observable-route-sha256:{MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256}"
    )


def test_locked_population_remains_exactly_24923_infer_true_adds() -> None:
    add_counts = [2548, 2405, 2457, 2542, 2521, 2483, 2390, 2483, 2516, 2578]
    rows = [
        {
            "authorized_add_operations": count,
            "authorized_extraction_calls": count,
            "authorized_search_operations": 10,
            "questions": 10,
        }
        for count in add_counts
    ]

    assert observed_campaign_population(rows) == MEM0_LOCKED_POPULATION_EXPECTATIONS
    assert sum(add_counts) == 24_923

    rows[0]["authorized_extraction_calls"] += 1
    assert observed_campaign_population(rows) != MEM0_LOCKED_POPULATION_EXPECTATIONS


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("source", "dataset_sha256"), _digest("other-dataset")),
        (("tool", "implementation_sha256"), _digest("other-tool")),
        (("mem0", "search", "top_k"), 10),
        (("mem0", "extraction_identity", "provider_retries"), 1),
        (("scoring", "provider_retries"), 1),
        (("shards", 0, "authorized_add_operations"), 2),
    ],
)
def test_policy_rejects_identity_or_authorization_drift(
    policy_fixture, path, value
):
    policy_path, lock, plan, shards, payload = policy_fixture
    mutated = copy.deepcopy(payload)
    target: Any = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    policy_path.write_text(json.dumps(mutated), encoding="utf-8")

    with pytest.raises(Mem0PolicyError):
        inspect_mem0_comparison_policy(
            policy_path,
            source_plan=plan,
            mem0_environment_lock=lock,
            expected_shards=shards,
        )


@pytest.mark.parametrize(
    ("secret_key", "secret_value"),
    [
        ("api_key", "secret-value"),
        ("Authorization", "Bearer TOPSECRET"),
        ("azure_openai_api_key", "TOPSECRET"),
    ],
)
def test_policy_rejects_secret_material(
    policy_fixture, secret_key: str, secret_value: str
):
    policy_path, lock, plan, shards, payload = policy_fixture
    payload["mem0"]["stable_payload"]["config"]["llm"]["config"][
        secret_key
    ] = secret_value
    payload["mem0"]["stable_config_sha256"] = canonical_json_sha256(
        payload["mem0"]["stable_payload"]
    )
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Mem0PolicyError, match="secret material"):
        inspect_mem0_comparison_policy(
            policy_path,
            source_plan=plan,
            mem0_environment_lock=lock,
            expected_shards=shards,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution", "remote"),
        ("network_calls_authorized", 1),
        ("runtime_probe_required", False),
    ],
)
def test_policy_requires_local_offline_embedder_contract(
    policy_fixture, field: str, value: Any
) -> None:
    policy_path, lock, plan, shards, payload = policy_fixture
    identity = payload["mem0"]["embedder_identity"]
    identity[field] = value
    body = dict(identity)
    body.pop("model_identity_sha256")
    identity["model_identity_sha256"] = canonical_json_sha256(body)
    policy_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Mem0PolicyError):
        inspect_mem0_comparison_policy(
            policy_path,
            source_plan=plan,
            mem0_environment_lock=lock,
            expected_shards=shards,
        )


def test_policy_rejects_nonfinite_json(policy_fixture):
    policy_path, lock, plan, shards, payload = policy_fixture
    payload["mem0"]["search"]["threshold"] = float("nan")
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Mem0PolicyError, match="non-finite"):
        inspect_mem0_comparison_policy(
            policy_path,
            source_plan=plan,
            mem0_environment_lock=lock,
            expected_shards=shards,
        )


def test_policy_rejects_stable_config_model_mismatch(policy_fixture):
    policy_path, lock, plan, shards, payload = policy_fixture
    payload["mem0"]["stable_payload"]["config"]["llm"]["config"][
        "model"
    ] = "different/model"
    payload["mem0"]["stable_config_sha256"] = canonical_json_sha256(
        payload["mem0"]["stable_payload"]
    )
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Mem0PolicyError, match="LLM model"):
        inspect_mem0_comparison_policy(
            policy_path,
            source_plan=plan,
            mem0_environment_lock=lock,
            expected_shards=shards,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_retries", 99),
        ("temperature", 2.0),
        ("api_base", "https://untrusted.invalid"),
    ],
)
def test_policy_rejects_unfrozen_llm_behavior_config(
    policy_fixture, field: str, value: Any
) -> None:
    policy_path, lock, plan, shards, payload = policy_fixture
    payload["mem0"]["stable_payload"]["config"]["llm"]["config"][field] = value
    payload["mem0"]["stable_config_sha256"] = canonical_json_sha256(
        payload["mem0"]["stable_payload"]
    )
    policy_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Mem0PolicyError, match="stable config"):
        inspect_mem0_comparison_policy(
            policy_path,
            source_plan=plan,
            mem0_environment_lock=lock,
            expected_shards=shards,
        )


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("vector_store", "config", "path"), "<owned_state>/../escape"),
        (("history_db_path",), "<owned_state>/../escape.sqlite"),
    ],
)
def test_policy_rejects_owned_state_path_traversal(
    policy_fixture, field_path: tuple[str, ...], value: str
) -> None:
    policy_path, lock, plan, shards, payload = policy_fixture
    target = payload["mem0"]["stable_payload"]["config"]
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = value
    payload["mem0"]["stable_config_sha256"] = canonical_json_sha256(
        payload["mem0"]["stable_payload"]
    )
    policy_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Mem0PolicyError, match="stable config"):
        inspect_mem0_comparison_policy(
            policy_path,
            source_plan=plan,
            mem0_environment_lock=lock,
            expected_shards=shards,
        )


def test_policy_recheck_detects_lock_replacement(policy_fixture):
    policy = _inspect(policy_fixture)
    policy_fixture[1].write_text("changed-lock\n", encoding="utf-8")
    with pytest.raises(Mem0PolicyError, match="environment lock changed"):
        policy.recheck()


def test_policy_recomputes_mutable_sample_content(policy_fixture):
    _path, _lock, _plan, shards, _payload = policy_fixture
    shard = copy.deepcopy(shards[0])
    shard.parsed_sample.questions[0].question = "mutated after hashing"
    with pytest.raises(Mem0PolicyError, match="sample content SHA"):
        expected_shard_policy_rows((shard,))


def test_policy_recomputes_mutable_raw_bundle_content(policy_fixture):
    _path, _lock, _plan, shards, _payload = policy_fixture
    shard = copy.deepcopy(shards[0])
    shard.raw_history_bundle["records"][0]["haystack_sessions"][0][0][
        "content"
    ] = "mutated after hashing"
    with pytest.raises(Mem0PolicyError, match="raw history bundle content SHA"):
        expected_shard_policy_rows((shard,))


def test_policy_independently_rebuilds_add_sequence(policy_fixture):
    _path, _lock, _plan, shards, _payload = policy_fixture
    shard = shards[0]
    first = shard.add_batches[0]
    forged = replace(
        shard,
        add_batches=(
            CompositeAddBatch(
                source_sample_id=first.source_sample_id,
                source=first.source,
                date=first.date,
                session_index=first.session_index,
                original_session_index=first.original_session_index,
                batch_index=first.batch_index,
                turn_start=first.turn_start,
                messages=(("user", "forged message"),) + first.messages[1:],
            ),
            *shard.add_batches[1:],
        ),
    )
    with pytest.raises(Mem0PolicyError, match="not independently derived"):
        expected_shard_policy_rows((forged,))
