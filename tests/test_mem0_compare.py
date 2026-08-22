from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
    tokenizer_proxy_identity,
)
from memory_condense.eval.benchmark import build_qa_prompt, exact_match, f1_score
from memory_condense.eval.validation_profile import LONGMEMEVAL_1M_95_PROFILE
from tools.mem0_eval.compare import (
    COMPARISON_SCHEMA_VERSION,
    COMPARISON_REPORT_TYPE,
    FROZEN_DATASET_SHA256,
    FROZEN_SAMPLE_SHA256_BY_OFFSET,
    FROZEN_SOURCE_ENVIRONMENT_SHA256,
    FROZEN_SOURCE_IMPLEMENTATION_SHA256,
    FROZEN_SOURCE_POLICY_SHA256,
    FROZEN_SPLIT_MANIFEST_SHA256,
    PairedComparisonError,
    canonical_sha256,
    compare_campaign_reports,
)
from tools.mem0_eval.policy import (
    MEM0_EMBEDDER_CHECKPOINT_SHA256,
    MEM0_EMBEDDER_DIMENSION,
    MEM0_EMBEDDER_DTYPE,
    MEM0_EMBEDDER_MODEL,
    MEM0_EMBEDDER_PROVIDER,
    MEM0_EMBEDDER_REVISION,
)
from tools.mem0_eval.prompt_pack import (
    MEM0_EFFECTIVE_RECENT_WINDOW,
    MEM0_PROMPT_PACK_PROTOCOL,
    MEM0_RECENT_WINDOW_SEMANTICS,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _frozen_retrieval_config() -> dict:
    policy_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "10 - Research Log"
        / "data"
        / "longmemeval-qwen-choice-coverage-operational-validation-v3.json"
    )
    retrieval = json.loads(policy_path.read_text(encoding="utf-8"))["retrieval"]
    return {
        key: value
        for key, value in retrieval.items()
        if key
        not in {"chunker_min_tokens", "chunker_max_tokens", "max_prompt_tokens"}
    }


def test_direct_tool_import_forces_offline_environment() -> None:
    keys = (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "LITELLM_LOCAL_MODEL_COST_MAP",
        "MEM0_TELEMETRY",
    )
    expected = ("1", "1", "1", "true", "false")
    environment = os.environ.copy()
    environment.update(dict.fromkeys(keys, "hostile-preexisting-value"))
    script = (
        "import os,socket;"
        "socket.socket.connect=lambda *a,**k: "
        "(_ for _ in ()).throw(AssertionError('network attempted'));"
        "import tools.mem0_eval.compare;"
        f"keys={keys!r};expected={expected!r};"
        "assert tuple(os.environ.get(k) for k in keys)==expected"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _nearest(values: list[int], quantile: float) -> int:
    return values[max(0, math.ceil(quantile * len(values)) - 1)]


def _distribution(values) -> dict:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": math.fsum(float(value) for value in ordered) / len(ordered),
        "p50": _nearest(ordered, 0.50),
        "p90": _nearest(ordered, 0.90),
        "p95": _nearest(ordered, 0.95),
        "p99": _nearest(ordered, 0.99),
        "max": ordered[-1],
        "values": ordered,
    }


def _usage(input_tokens: int, *, elapsed: float) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": 8,
        "cache_read_input_tokens": 0,
        "elapsed_s": elapsed,
        "calls": 1,
    }


def _sum_usage(rows: list[dict]) -> dict:
    return {
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "cache_read_input_tokens": sum(
            row["cache_read_input_tokens"] for row in rows
        ),
        "elapsed_s": math.fsum(row["elapsed_s"] for row in rows),
        "calls": sum(row["calls"] for row in rows),
    }


def _core_metrics(rows: list[dict], reserve: int) -> dict:
    contexts = [row["context_tokens"] for row in rows]
    prompts = [row["prompt_token_proxy"] for row in rows]
    requests = [value + reserve for value in prompts]
    return {
        "mean_f1": math.fsum(row["f1"] for row in rows) / len(rows),
        "exact_match_rate": math.fsum(
            1.0 if row["exact_match"] else 0.0 for row in rows
        )
        / len(rows),
        "judge_accuracy": math.fsum(
            1.0 if row["judge_correct"] else 0.0 for row in rows
        )
        / len(rows),
        "mean_context_tokens": math.fsum(contexts) / len(rows),
        "mean_prompt_token_proxy": math.fsum(prompts) / len(rows),
        "mean_request_token_proxy": math.fsum(requests) / len(rows),
        "context_token_distribution": _distribution(contexts),
        "prompt_token_proxy_distribution": _distribution(prompts),
        "request_token_proxy_distribution": _distribution(requests),
    }


def _protocols() -> tuple[dict, dict]:
    shared = {
        "responder_model": "openai/codex_sdk/gpt-5.6-terra",
        "judge_model": "openai/codex_sdk/gpt-5.6-sol",
        "use_judge": True,
        "provider_retries": 0,
        "max_prompt_tokens": 8000,
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
    }
    treatment = {
        **shared,
        "embedding_device": "cuda",
        "benchmark_format": "longmemeval",
        "max_provider_calls": 20,
    }
    mem0 = {
        **shared,
        "max_provider_calls_per_shard": 20,
        "sample_offsets": list(range(0, 100, 10)),
    }
    return treatment, mem0


def _treatment_campaign() -> dict:
    treatment_protocol, _ = _protocols()
    rows: list[dict] = []
    for index in range(100):
        correct = index < 95
        gold = f"gold-{index}"
        prediction = gold if correct else "nope"
        chunks = [f"fact {index}"]
        context_tokens = sum(count_tokens(chunk) for chunk in chunks)
        dated_question = f"Question {index}?\nReference date: 2026-01-01"
        prompt = count_chat_prompt_token_proxy(
            build_qa_prompt(dated_question, chunks)
        )
        rows.append(
            {
                "question_id": f"q-{index:03d}",
                "question": f"Question {index}?",
                "gold_answer": gold,
                "predicted_answer": prediction,
                "category": "single-session-user",
                "retrieved_chunks": chunks,
                "f1": f1_score(prediction, gold),
                "exact_match": exact_match(prediction, gold),
                "judge_correct": correct,
                "judge_reasoning": (
                    "CORRECT: same answer" if correct else "INCORRECT: wrong answer"
                ),
                "context_tokens": context_tokens,
                "prompt_token_proxy": prompt,
                "prompt_tokens": prompt,
                "responder_output_token_reserve": 256,
                "request_token_proxy": prompt + 256,
                "provider_prompt_budget_compliant": True,
                "transcript_tokens": 1_000_000,
                "context_fraction": context_tokens / 1_000_000,
                "transcript_token_savings": 1.0
                - context_tokens / 1_000_000,
                "responder_usage": _usage(prompt, elapsed=0.1),
                "judge_usage": _usage(90, elapsed=0.2),
            }
        )
    inputs = [
        {
            "name": f"treatment-{index}.json",
            "sha256": _digest(f"treatment-input-{index}"),
            "num_samples": 1,
            "num_questions": 10,
            "target_status": "insufficient_questions",
        }
        for index in range(10)
    ]
    sources = {}
    sample_hashes = []
    for index, row in enumerate(rows):
        shard = index // 10
        sample_hash = FROZEN_SAMPLE_SHA256_BY_OFFSET[str(shard * 10)]
        sample_hashes.append(sample_hash)
        sources[row["question_id"]] = {
            "report_name": inputs[shard]["name"],
            "report_sha256": inputs[shard]["sha256"],
            "sample_id": f"sample-{shard}",
            "sample_sha256": sample_hash,
        }
    metrics = _core_metrics(rows, 256)
    responders = [row["responder_usage"] for row in rows]
    judges = [row["judge_usage"] for row in rows]
    category = {
        "category": "single-session-user",
        "num_questions": 100,
        "mean_f1": metrics["mean_f1"],
        "exact_match_rate": metrics["exact_match_rate"],
        "judge_accuracy": metrics["judge_accuracy"],
    }
    provider_distribution = _distribution(
        row["responder_usage"]["input_tokens"] for row in rows
    )
    return {
        "schema_version": 1,
        "report_type": "benchmark_campaign",
        "inputs": inputs,
        "input_count": 10,
        "input_set_sha256": canonical_sha256(
            sorted(row["sha256"] for row in inputs)
        ),
        "benchmark": "longmemeval_s_cleaned",
        "dataset_sha256": FROZEN_DATASET_SHA256,
        "split_manifest_sha256": FROZEN_SPLIT_MANIFEST_SHA256,
        "benchmark_split": "validation",
        "implementation_sha256": FROZEN_SOURCE_IMPLEMENTATION_SHA256,
        "environment_lock_sha256": FROZEN_SOURCE_ENVIRONMENT_SHA256,
        "policy_manifest_sha256": FROZEN_SOURCE_POLICY_SHA256,
        "chunker_config": {"min_tokens": 120, "max_tokens": 250},
        "retrieval_config": _frozen_retrieval_config(),
        "responder_model": treatment_protocol["responder_model"],
        "judge_model": treatment_protocol["judge_model"],
        "embedding_device": "cuda",
        "recent_window": 4,
        "max_prompt_tokens": 8000,
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "responder_output_token_reserve": 256,
        "evaluation_protocol": treatment_protocol,
        "claim_profile": LONGMEMEVAL_1M_95_PROFILE,
        "claim_profile_verified": True,
        "cache_receipts_by_sample": {
            digest: {} for digest in sorted(set(sample_hashes))
        },
        "num_samples": 10,
        "num_questions": 100,
        "question_results": rows,
        "question_sources": sources,
        "mean_f1": metrics["mean_f1"],
        "exact_match_rate": metrics["exact_match_rate"],
        "judge_accuracy": metrics["judge_accuracy"],
        "mean_context_tokens": metrics["mean_context_tokens"],
        "mean_prompt_token_proxy": metrics["mean_prompt_token_proxy"],
        "p95_prompt_token_proxy": metrics["prompt_token_proxy_distribution"][
            "p95"
        ],
        "max_prompt_token_proxy_observed": metrics[
            "prompt_token_proxy_distribution"
        ]["max"],
        "mean_request_token_proxy": metrics["mean_request_token_proxy"],
        "mean_prompt_tokens": metrics["mean_prompt_token_proxy"],
        "p95_prompt_tokens": metrics["prompt_token_proxy_distribution"]["p95"],
        "max_prompt_tokens_observed": metrics["prompt_token_proxy_distribution"][
            "max"
        ],
        "context_token_distribution": metrics["context_token_distribution"],
        "prompt_token_proxy_distribution": metrics[
            "prompt_token_proxy_distribution"
        ],
        "request_token_proxy_distribution": metrics[
            "request_token_proxy_distribution"
        ],
        "provider_input_token_distribution": provider_distribution,
        "prompt_token_distribution": metrics["prompt_token_proxy_distribution"],
        "transcript_token_distribution": _distribution(
            row["transcript_tokens"] for row in rows
        ),
        "prompt_token_proxy_budget_compliance": True,
        "provider_prompt_budget_compliance": True,
        "provider_input_usage_status": "complete",
        "prompt_budget_compliance": True,
        "responder_usage": _sum_usage(responders),
        "judge_usage": _sum_usage(judges),
        "by_category": {"single-session-user": category},
        "accuracy_target": 0.95,
        "min_target_questions": 100,
        "accuracy_target_met": True,
        "metric_accuracy_target_met": True,
        "locked_population_verified": True,
        "target_status": "passed",
    }


def _mem0_campaign(treatment: dict) -> dict:
    _, protocol = _protocols()
    proxy = tokenizer_proxy_identity()
    rows: list[dict] = []
    for index in range(100):
        correct = index < 88 or index in {95, 96}
        gold = f"gold-{index}"
        prediction = gold if correct else "nope"
        dated_question = f"Question {index}?\nReference date: 2026-01-01"
        context = f"Mem0 fact {index}"
        messages = build_qa_prompt(dated_question, [context])
        prompt = count_chat_prompt_token_proxy(messages)
        rows.append(
            {
                "question_index": index % 10 + 1,
                "question_id": f"q-{index:03d}",
                "question": f"Question {index}?",
                "dated_question": dated_question,
                "gold_answer": gold,
                "prediction": prediction,
                "category": "single-session-user",
                "retrieval_row_sha256": _digest(f"retrieval-row-{index}"),
                "query_sha256": hashlib.sha256(dated_question.encode()).hexdigest(),
                "context": context,
                "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                "context_tokens": count_tokens(context),
                "messages": messages,
                "messages_sha256": canonical_sha256(messages),
                "prompt_token_proxy": prompt,
                "max_prompt_tokens": 8000,
                "residual_prompt_tokens": 8000 - prompt,
                "prompt_token_proxy_identity": proxy,
                "raw_pool_count": 5,
                "raw_pool_sha256": _digest(f"raw-pool-{index}"),
                "raw_memory_tokens": 100,
                "packed_count": 2,
                "packed_memory_tokens": 40,
                "packed_pool_sha256": _digest(f"packed-pool-{index}"),
                "search_latency_s": 0.01,
                "attribution_kind": "request_window_non_evidence",
                "supports_exact_source_provenance": False,
                "exact_match": exact_match(prediction, gold),
                "f1": f1_score(prediction, gold),
                "judge_correct": correct,
                "judge_reasoning": (
                    "CORRECT: same answer" if correct else "INCORRECT: wrong answer"
                ),
                "prompt_pack_protocol": MEM0_PROMPT_PACK_PROTOCOL,
                "provider_prompt_budget_compliant": True,
                "configured_recent_window": 4,
                "effective_recent_window": MEM0_EFFECTIVE_RECENT_WINDOW,
                "recent_window_semantics": MEM0_RECENT_WINDOW_SEMANTICS,
                "responder_usage": _usage(prompt, elapsed=0.12),
                "judge_usage": _usage(90, elapsed=0.22),
            }
        )
    inputs = [
        {
            "sample_offset": index * 10,
            "sample_sha256": FROZEN_SAMPLE_SHA256_BY_OFFSET[
                str(index * 10)
            ],
            "name": f"mem0-{index}.json",
            "sha256": _digest(f"mem0-input-{index}"),
            "retrieval_artifact_sha256": _digest(f"artifact-{index}"),
            "retrieval_trace_sha256": _digest(f"retrieval-trace-{index}"),
            "scoring_trace_sha256": _digest(f"scoring-trace-{index}"),
        }
        for index in range(10)
    ]
    sources = {}
    for index, row in enumerate(rows):
        shard = index // 10
        sources[row["question_id"]] = {
            "sample_offset": shard * 10,
            "report_name": inputs[shard]["name"],
            "report_sha256": inputs[shard]["sha256"],
            "retrieval_artifact_sha256": inputs[shard][
                "retrieval_artifact_sha256"
            ],
        }
    metrics = _core_metrics(rows, 256)
    responders = [row["responder_usage"] for row in rows]
    judges = [row["judge_usage"] for row in rows]
    common = [
        {
            "question_id": row["question_id"],
            "predicted_answer": row["prediction"],
            "judge_correct": row["judge_correct"],
            "f1": row["f1"],
            "exact_match": row["exact_match"],
            "context_tokens": row["context_tokens"],
            "prompt_token_proxy": row["prompt_token_proxy"],
            "responder_usage": row["responder_usage"],
            "judge_usage": row["judge_usage"],
        }
        for row in rows
    ]
    extraction_without_hash = {
        "provider": "openai",
        "model": "extractor-model",
        "revision": "r1",
        "provider_retries": 0,
        "logical_call_boundary": "Memory.llm.generate_response",
        "logical_calls_per_add": 1,
        "http_attempts_certified": False,
    }
    extraction = {
        **extraction_without_hash,
        "model_identity_sha256": canonical_sha256(extraction_without_hash),
    }
    embedder_without_hash = {
        "provider": MEM0_EMBEDDER_PROVIDER,
        "model": MEM0_EMBEDDER_MODEL,
        "revision": MEM0_EMBEDDER_REVISION,
        "checkpoint_sha256": MEM0_EMBEDDER_CHECKPOINT_SHA256,
        "dimension": MEM0_EMBEDDER_DIMENSION,
        "device": "cuda",
        "dtype": MEM0_EMBEDDER_DTYPE,
        "execution": "local_offline",
        "network_calls_authorized": 0,
        "runtime_probe_required": True,
    }
    embedder = {
        **embedder_without_hash,
        "model_identity_sha256": canonical_sha256(embedder_without_hash),
    }
    identity = {
        "source_validation_policy_sha256": treatment["policy_manifest_sha256"],
        "source_implementation_sha256": treatment["implementation_sha256"],
        "source_environment_lock_sha256": treatment["environment_lock_sha256"],
        "mem0_policy_sha256": _digest("mem0-policy"),
        "mem0_environment_lock_sha256": _digest("mem0-environment"),
        "mem0_tool_implementation_sha256": _digest("mem0-tool"),
        "mem0_stable_config_sha256": _digest("stable-config"),
        "extraction_model_identity": extraction,
        "extraction_model_identity_sha256": canonical_sha256(extraction),
        "embedder_model_identity": embedder,
        "embedder_model_identity_sha256": canonical_sha256(embedder),
        "scoring_policy_sha256": _digest("scoring-policy"),
        "source_evaluation_identity_sha256": canonical_sha256(protocol),
    }
    return {
        "schema_version": 2,
        "report_type": "mem0_longmemeval_campaign",
        "arm_id": "mem0_oss_2_0_18_direct_1m_v1",
        "run_status": "complete",
        "inputs": inputs,
        "input_count": 10,
        "input_set_sha256": canonical_sha256(
            [row["sha256"] for row in inputs]
        ),
        "identity": identity,
        "model_identity": {
            "responder_model": protocol["responder_model"],
            "responder_model_identity_sha256": _digest("responder-model"),
            "judge_model": protocol["judge_model"],
            "judge_model_identity_sha256": _digest("judge-model"),
        },
        "runtime_model_identity_probe": {
            "kind": "unavailable_injected_nonproduction",
            "extraction_model_identity_sha256": identity[
                "extraction_model_identity_sha256"
            ],
            "embedder_model_identity_sha256": identity[
                "embedder_model_identity_sha256"
            ],
            "before_match": False,
            "after_match": False,
            "comparison_certified": False,
        },
        "config": {
            "max_prompt_tokens": 8000,
            "responder_max_output_tokens": 256,
            "judge_max_output_tokens": 64,
            "authorized_local_wrapper_retries": 0,
            "external_retry_attempts_certified": False,
            "mem0_top_k": 200,
            "mem0_threshold": 0.1,
            "rendering_mode": "official-memory-text-created-at",
        },
        "benchmark": "longmemeval",
        "dataset_sha256": treatment["dataset_sha256"],
        "split_manifest_sha256": treatment["split_manifest_sha256"],
        "benchmark_split": "validation",
        "implementation_sha256": treatment["implementation_sha256"],
        "environment_lock_sha256": treatment["environment_lock_sha256"],
        "policy_manifest_sha256": treatment["policy_manifest_sha256"],
        "responder_model": protocol["responder_model"],
        "judge_model": protocol["judge_model"],
        "recent_window": 4,
        "max_prompt_tokens": 8000,
        "prompt_token_proxy_identity": proxy,
        "responder_output_token_reserve": 256,
        "evaluation_protocol": protocol,
        "population_identity": {
            "question_ids_sha256": canonical_sha256(
                [row["question_id"] for row in rows]
            ),
            "sample_offsets": list(range(0, 100, 10)),
            "sample_sha256_by_offset": {
                str(offset): FROZEN_SAMPLE_SHA256_BY_OFFSET[str(offset)]
                for offset in range(0, 100, 10)
            },
        },
        "prompt_identity": {
            "prompt_pack_protocol": MEM0_PROMPT_PACK_PROTOCOL,
            "max_prompt_tokens": 8000,
            "prompt_cap_semantics": protocol["prompt_cap_semantics"],
            "prompt_token_proxy_identity": proxy,
            "responder_output_token_reserve": 256,
            "configured_recent_window": 4,
            "effective_recent_window": MEM0_EFFECTIVE_RECENT_WINDOW,
            "recent_window_semantics": MEM0_RECENT_WINDOW_SEMANTICS,
        },
        "sample_offsets": list(range(0, 100, 10)),
        "num_samples": 10,
        "num_questions": 100,
        "question_results": rows,
        "common_question_result_schema": "memory-condense-common-qa-result-v1",
        "common_question_results": common,
        "question_sources": sources,
        "raw_input_totals": {"raw_pairs": 24_928, "skipped_empty_pairs": 5},
        "operation_totals": {
            "mem0_adds": 24_923,
            "mem0_searches": 100,
            "responder_logical_wrapper_calls": 100,
            "judge_logical_wrapper_calls": 100,
            "answer_judge_logical_wrapper_calls": 200,
            "mem0_local_logical_wrapper_calls": 24_923,
            "mem0_logical_extraction_call_boundary": "Memory.llm.generate_response",
            "external_http_attempts_certified": False,
            "underlying_mem0_provider_calls": None,
            "underlying_mem0_provider_usage_status": (
                "unavailable_from_mem0_oss_public_api"
            ),
        },
        "mean_f1": metrics["mean_f1"],
        "exact_match_rate": metrics["exact_match_rate"],
        "judge_accuracy": metrics["judge_accuracy"],
        "mean_context_tokens": metrics["mean_context_tokens"],
        "mean_prompt_token_proxy": metrics["mean_prompt_token_proxy"],
        "p95_prompt_token_proxy": metrics["prompt_token_proxy_distribution"][
            "p95"
        ],
        "context_token_distribution": metrics["context_token_distribution"],
        "prompt_token_proxy_distribution": metrics[
            "prompt_token_proxy_distribution"
        ],
        "request_token_proxy_distribution": metrics[
            "request_token_proxy_distribution"
        ],
        "max_prompt_token_proxy_observed": metrics[
            "prompt_token_proxy_distribution"
        ]["max"],
        "prompt_token_proxy_budget_compliance": True,
        "provider_prompt_budget_compliance": True,
        "provider_input_usage_status": "local_injected_receipts_complete",
        "external_provider_usage_certified": False,
        "responder_usage": _sum_usage(responders),
        "judge_usage": _sum_usage(judges),
        "provenance": {
            "attribution_kind": "request_window_non_evidence",
            "supports_exact_source_provenance": False,
            "source_session_date_exposure": "diagnostics_only_not_model_input",
            "retrieved_created_at_exposure": "answer_prompt_date_headings",
            "source_coverage_status": "unavailable_exact_source_provenance",
            "source_coverage": None,
            "request_window_diagnostic_only": True,
            "source_coverage_reason": (
                "mem0_request_window_attribution_is_not_exact_evidence_provenance"
            ),
        },
        "source_coverage_status": "unavailable_exact_source_provenance",
        "source_coverage": None,
        "exact_provenance_requirement_met": False,
        "local_request_token_state_contract_satisfied": True,
        "zero_persisted_transformer_token_state_verified": False,
        "external_provider_persistence_certified": False,
        "production_binding_certified": False,
        "certification_status": "injected_core_nonproduction",
        "locked_population_verified": True,
        "local_comparison_protocol_verified": True,
        "accuracy_target": 0.95,
        "min_target_questions": 100,
        "metric_accuracy_target_met": False,
        "accuracy_target_met": False,
        "target_status": "metric_failed_noncertified",
    }


@pytest.fixture
def campaign_pair() -> tuple[dict, dict]:
    treatment = _treatment_campaign()
    return treatment, _mem0_campaign(treatment)


def test_recomputes_strict_paired_metrics_and_preserves_noncertified_status(
    campaign_pair,
):
    treatment, mem0 = campaign_pair

    result = compare_campaign_reports(treatment, mem0)

    assert COMPARISON_SCHEMA_VERSION == 2
    assert result["schema_version"] == COMPARISON_SCHEMA_VERSION
    assert result["report_type"] == COMPARISON_REPORT_TYPE
    assert result["metric_comparison"] == {
        "valid": True,
        "status": "paired_metrics_recomputed_from_primitive_rows",
        "num_questions": 100,
    }
    assert result["arm_metrics"]["treatment"]["judge_accuracy"] == 0.95
    assert result["arm_metrics"]["mem0"]["judge_accuracy"] == 0.9
    assert result["treatment_minus_mem0"]["judge_accuracy"] == pytest.approx(
        0.05
    )
    assert result["paired_judge_outcomes"] == {
        "treatment_wins": 7,
        "ties": 91,
        "treatment_losses": 2,
    }
    assert result["certification"]["certified"] is False
    assert result["certification"]["status"] == "metric_only_noncertified"
    assert result["certification"]["blocking_reasons"] == [
        "mem0_production_binding_certified_false"
    ]
    assert result["provenance_comparison"]["mem0"] == {
        "status": "not_applicable",
        "value": None,
        "supports_exact_source_provenance": False,
        "attribution_kind": "request_window_non_evidence",
        "reason": "mem0_request_window_attribution_is_not_exact_evidence_provenance",
    }
    assert len(result["question_results"]) == 100
    assert result["question_results"][88]["outcome"] == "treatment_win"
    assert result["question_results"][95]["outcome"] == "treatment_loss"
    assert result["input_hashes"]["treatment_campaign_canonical_sha256"] == (
        canonical_sha256(treatment)
    )
    assert result["input_hashes"]["mem0_campaign_canonical_sha256"] == (
        canonical_sha256(mem0)
    )
    assert result["paired_population_identity"][
        "sample_sha256_by_offset"
    ] == dict(FROZEN_SAMPLE_SHA256_BY_OFFSET)
    assert result["paired_population_identity"]["sample_set_sha256"] == (
        canonical_sha256(dict(FROZEN_SAMPLE_SHA256_BY_OFFSET))
    )
    assert result["shared_identity"]["configured_recent_window"] == 4
    assert result["shared_identity"]["effective_recent_window"] == 0
    assert result["shared_identity"]["recent_window_semantics"] == (
        MEM0_RECENT_WINDOW_SEMANTICS
    )


def test_mem0_zero_provider_input_usage_is_unavailable_not_an_empty_prompt(
    campaign_pair,
):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    for row, common in zip(
        mem0["question_results"],
        mem0["common_question_results"],
        strict=True,
    ):
        row["responder_usage"]["input_tokens"] = 0
        row["judge_usage"]["input_tokens"] = 0
        row["provider_prompt_budget_compliant"] = None
        common["responder_usage"] = row["responder_usage"]
        common["judge_usage"] = row["judge_usage"]
        assert row["messages"]
        assert row["prompt_token_proxy"] > 0
    mem0["responder_usage"] = _sum_usage(
        [row["responder_usage"] for row in mem0["question_results"]]
    )
    mem0["judge_usage"] = _sum_usage(
        [row["judge_usage"] for row in mem0["question_results"]]
    )
    mem0["provider_prompt_budget_compliance"] = None
    mem0["provider_input_usage_status"] = (
        "local_injected_receipts_unavailable"
    )

    result = compare_campaign_reports(treatment, mem0)

    assert result["metric_comparison"]["valid"] is True
    assert mem0["responder_usage"]["input_tokens"] == 0
    assert all(
        row["mem0"]["prompt_token_proxy"] > 0
        for row in result["question_results"]
    )


def test_rejects_mem0_effective_recent_window_drift(campaign_pair):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    mem0["question_results"][0]["effective_recent_window"] = 4

    with pytest.raises(PairedComparisonError, match="effective_recent_window"):
        compare_campaign_reports(treatment, mem0)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda treatment, _mem0: treatment.update({"smuggled": True}),
            "fields mismatch",
        ),
        (
            lambda treatment, _mem0: treatment["question_results"][0].update(
                {"smuggled": True}
            ),
            "fields mismatch",
        ),
        (
            lambda _treatment, mem0: mem0["common_question_results"][0].update(
                {"smuggled": True}
            ),
            "fields mismatch",
        ),
        (
            lambda treatment, _mem0: treatment["question_results"][0].update(
                {"f1": 0.5}
            ),
            "f1 drift",
        ),
        (
            lambda _treatment, mem0: mem0.update({"judge_accuracy": 0.99}),
            "judge_accuracy drift",
        ),
        (
            lambda _treatment, mem0: mem0["question_results"][0].update(
                {"prompt_token_proxy": float("nan")}
            ),
            "non-finite",
        ),
        (
            lambda treatment, _mem0: treatment["retrieval_config"].update(
                {"api_key": "sk-live-secret"}
            ),
            "secret",
        ),
    ],
)
def test_rejects_schema_metric_nonfinite_and_secret_smuggling(
    campaign_pair, mutation, match
):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    mutation(treatment, mem0)

    with pytest.raises(PairedComparisonError, match=match):
        compare_campaign_reports(treatment, mem0)


@pytest.mark.parametrize("field", ["azure_openai_api_key", "azure_session_token"])
def test_rejects_namespaced_credential_smuggling(campaign_pair, field):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    treatment["retrieval_config"][field] = "TOPSECRET"

    with pytest.raises(PairedComparisonError, match="secret"):
        compare_campaign_reports(treatment, mem0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", "A different question?"),
        ("gold_answer", "unrelated-gold"),
        ("category", "different-category"),
    ],
)
def test_rejects_cross_arm_question_semantic_drift(
    campaign_pair, field, value
):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    # Row 90 is already an incorrect, zero-F1 answer, so changing its gold to
    # another disjoint string leaves the Mem0 arm's local aggregates valid.
    mem0["question_results"][90][field] = value

    with pytest.raises(PairedComparisonError, match=field):
        compare_campaign_reports(treatment, mem0)


def test_rejects_treatment_history_population_drift_with_internal_bindings_intact(
    campaign_pair,
):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    replacement = _digest("different-treatment-composed-history")
    old = treatment["question_sources"]["q-030"]["sample_sha256"]
    for index in range(30, 40):
        treatment["question_sources"][f"q-{index:03d}"][
            "sample_sha256"
        ] = replacement
    treatment["cache_receipts_by_sample"][replacement] = treatment[
        "cache_receipts_by_sample"
    ].pop(old)

    with pytest.raises(
        PairedComparisonError,
        match=r"paired question q-03\d composed sample_sha256",
    ):
        compare_campaign_reports(treatment, mem0)


def test_rejects_jointly_consistent_nonfrozen_history_population(campaign_pair):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    replacement = _digest("jointly-different-composed-history")
    old = treatment["question_sources"]["q-030"]["sample_sha256"]
    for index in range(30, 40):
        treatment["question_sources"][f"q-{index:03d}"][
            "sample_sha256"
        ] = replacement
    treatment["cache_receipts_by_sample"][replacement] = treatment[
        "cache_receipts_by_sample"
    ].pop(old)
    mem0["inputs"][3]["sample_sha256"] = replacement
    mem0["population_identity"]["sample_sha256_by_offset"]["30"] = replacement

    with pytest.raises(
        PairedComparisonError,
        match="mem0 frozen composed-sample population",
    ):
        compare_campaign_reports(treatment, mem0)


def test_rejects_collapsing_all_questions_onto_one_mem0_input(campaign_pair):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    first = mem0["inputs"][0]
    for source in mem0["question_sources"].values():
        source.update(
            {
                "sample_offset": first["sample_offset"],
                "report_name": first["name"],
                "report_sha256": first["sha256"],
                "retrieval_artifact_sha256": first[
                    "retrieval_artifact_sha256"
                ],
            }
        )

    with pytest.raises(
        PairedComparisonError,
        match="repeats question_index|ten questions per input",
    ):
        compare_campaign_reports(treatment, mem0)


def test_rejects_collapsing_treatment_questions_onto_one_input(campaign_pair):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    first = treatment["inputs"][0]
    first_source = treatment["question_sources"]["q-000"]
    for source in treatment["question_sources"].values():
        source.update(
            {
                "report_name": first["name"],
                "report_sha256": first["sha256"],
                "sample_id": first_source["sample_id"],
                "sample_sha256": first_source["sample_sha256"],
            }
        )

    with pytest.raises(PairedComparisonError, match="ten questions per input"):
        compare_campaign_reports(treatment, mem0)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda treatment, mem0: mem0.update(
                {"dataset_sha256": _digest("different-dataset")}
            ),
            "dataset_sha256",
        ),
        (
            lambda treatment, mem0: mem0.update(
                {"responder_model": "different-responder"}
            ),
            "responder_model",
        ),
        (
            lambda treatment, mem0: mem0["evaluation_protocol"].update(
                {"recent_window": 5}
            ),
            "recent_window",
        ),
        (
            lambda treatment, mem0: mem0["question_results"].__setitem__(
                0,
                {
                    **mem0["question_results"][0],
                    "question_id": "q-not-locked",
                },
            ),
            "question",
        ),
    ],
)
def test_rejects_identity_and_population_drift(campaign_pair, mutation, match):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    mutation(treatment, mem0)

    with pytest.raises(PairedComparisonError, match=match):
        compare_campaign_reports(treatment, mem0)


def test_rejects_treatment_retrieval_policy_drift(campaign_pair):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    treatment["retrieval_config"]["k"] = 11

    with pytest.raises(PairedComparisonError, match="frozen retrieval identity"):
        compare_campaign_reports(treatment, mem0)


def test_rejects_duplicate_and_missing_question_ids(campaign_pair):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    treatment["question_results"][1]["question_id"] = treatment[
        "question_results"
    ][0]["question_id"]

    with pytest.raises(PairedComparisonError, match="duplicate question IDs"):
        compare_campaign_reports(treatment, mem0)


def test_never_certifies_when_mem0_production_binding_is_false(campaign_pair):
    treatment, mem0 = campaign_pair
    result = compare_campaign_reports(treatment, mem0)

    assert mem0["production_binding_certified"] is False
    assert result["certification"]["certified"] is False
    assert "certified" not in result["certification"]["status"].removeprefix(
        "non"
    ) or result["certification"]["status"] == "metric_only_noncertified"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution", "remote"),
        ("network_calls_authorized", 1),
        ("runtime_probe_required", False),
    ],
)
def test_rejects_nonlocal_or_unprobed_embedder_binding(
    campaign_pair, field, value
):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    mem0["identity"]["embedder_model_identity"][field] = value

    with pytest.raises(PairedComparisonError, match="mem0 embedder"):
        compare_campaign_reports(treatment, mem0)


def test_rejects_self_consistent_but_unfrozen_embedder_identity(campaign_pair):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    embedder = mem0["identity"]["embedder_model_identity"]
    embedder["revision"] = "different-revision"
    without_hash = {
        key: value
        for key, value in embedder.items()
        if key != "model_identity_sha256"
    }
    embedder["model_identity_sha256"] = canonical_sha256(without_hash)
    mem0["identity"]["embedder_model_identity_sha256"] = canonical_sha256(
        embedder
    )
    mem0["runtime_model_identity_probe"][
        "embedder_model_identity_sha256"
    ] = mem0["identity"]["embedder_model_identity_sha256"]

    with pytest.raises(PairedComparisonError, match="frozen revision"):
        compare_campaign_reports(treatment, mem0)


@pytest.mark.parametrize(
    "field",
    ["source_session_date_exposure", "retrieved_created_at_exposure"],
)
def test_requires_explicit_mem0_date_provenance_fields(campaign_pair, field):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    del mem0["provenance"][field]

    with pytest.raises(PairedComparisonError, match="mem0.provenance fields mismatch"):
        compare_campaign_reports(treatment, mem0)


def test_rejects_forged_runtime_model_probe_certification(campaign_pair):
    treatment, mem0 = copy.deepcopy(campaign_pair)
    mem0["runtime_model_identity_probe"]["comparison_certified"] = True

    with pytest.raises(PairedComparisonError, match="runtime_model_identity_probe"):
        compare_campaign_reports(treatment, mem0)
