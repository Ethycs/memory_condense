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
from memory_condense.eval.benchmark import (
    build_judge_prompt,
    build_qa_prompt,
    exact_match,
    f1_score,
)
from memory_condense.eval.recall_guarded_cumulative_final_answer import (
    build_responder_prompt_policy_identity,
)
from memory_condense.eval.validation_profile import LONGMEMEVAL_1M_95_PROFILE
from tools.mem0_eval.compare import (
    COMPARISON_SCHEMA_VERSION,
    COMPARISON_REPORT_TYPE,
    FIXED_STAGE_COMPARISON_SCHEMA_VERSION,
    FIXED_STAGE_ID,
    FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS,
    FIXED_STAGE_PROMPT_CAP,
    FIXED_STAGE_RESPONDER_MAX_OUTPUT_TOKENS,
    FIXED_STAGE_SOL_MODEL,
    FIXED_STAGE_TERRA_MODEL,
    FIXED_STAGE_TREATMENT_FORMAT,
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
from tools.mem0_eval.compare_fixed_stage_derivation import (
    verify_treatment_prompt_derivation,
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


def _canonical_artifact_digest(value: object) -> str:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _fixed_accuracy_summary(correct: int, questions: int = 100) -> dict:
    accuracy = correct / questions
    passed = questions >= 100 and accuracy >= 0.95
    return {
        "questions": questions,
        "correct": correct,
        "incorrect": questions - correct,
        "binary_accuracy": accuracy,
        "target_accuracy": 0.95,
        "minimum_questions": 100,
        "minimum_correct_at_observed_population": math.ceil(0.95 * questions),
        "accuracy_threshold_met": accuracy >= 0.95,
        "minimum_population_met": questions >= 100,
        "gate_passed": passed,
        "status": "pass" if passed else "below_accuracy_target",
    }


def _fixed_stage_semantic_score(
    treatment: dict,
    mem0: dict,
    *,
    answer_artifact_sha256: str | None = None,
    retrieval_sha256: str | None = None,
) -> dict:
    gateway_url = "https://central-dev.zt:4000/v1"
    responder_runtime_sha = _digest("fixed-responder-runtime")
    responder_prompt_policy = build_responder_prompt_policy_identity(
        [row["messages"] for row in mem0["question_results"]]
    )
    responder_prompt_policy_sha = canonical_sha256(responder_prompt_policy)
    answer_artifact_sha = (
        answer_artifact_sha256 or _digest("fixed-answer-artifact")
    )
    retrieval_sha = retrieval_sha256 or _digest("fixed-retrieval")
    population_sha = _digest("fixed-source-population")
    implementation_sha = _digest("fixed-judge-implementation")
    policy = {
        "format": "memory-condense-fixed-stage-semantic-judge-policy-v1",
        "answer_artifact_validator": (
            "memory_condense.eval.recall_guarded_cumulative_final_answer."
            "validate_final_answer_artifact"
        ),
        "judge_prompt_builder": "memory_condense.eval.benchmark.build_judge_prompt",
        "verdict_parser": (
            "memory_condense.eval._binary_judge_protocol."
            "parse_binary_judge_verdict"
        ),
        "question_form": "undated benchmark question",
        "fixed_stage_id": FIXED_STAGE_ID,
        "gate_unit": "one preregistered fixed retrieval stage",
        "target_accuracy": 0.95,
        "minimum_questions": 100,
        "responder_model": FIXED_STAGE_TERRA_MODEL,
        "responder_gateway_model": "codex_sdk/gpt-5.6-terra",
        "responder_runtime_format": (
            "memory-condense-recall-guarded-fixed-stage-final-answer-runtime-v1"
        ),
        "responder_prompt_policy_binding": (
            "derived from sealed system messages and verified QA framing"
        ),
        "responder_max_prompt_token_proxy": FIXED_STAGE_PROMPT_CAP,
        "responder_max_new_tokens": FIXED_STAGE_RESPONDER_MAX_OUTPUT_TOKENS,
        "judge_model": FIXED_STAGE_SOL_MODEL,
        "judge_gateway_model": "codex_sdk/gpt-5.6-sol",
        "judge_runtime_format": (
            "memory-condense-recall-guarded-semantic-judge-runtime-v1"
        ),
        "judge_max_new_tokens": FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS,
        "gateway_url": gateway_url,
        "provider_retries": 0,
        "responder_temperature": None,
        "judge_temperature": None,
        "deduplication": (
            "identical canonical question+gold+prediction messages"
        ),
        "gold_access": "after complete final-answer artifact validation",
    }
    policy_sha = canonical_sha256(policy)
    planned = []
    for ordinal, (source, mem0_row) in enumerate(
        zip(treatment["question_results"], mem0["question_results"], strict=True)
    ):
        judge_messages = build_judge_prompt(
            mem0_row["question"],
            mem0_row["gold_answer"],
            source["predicted_answer"],
        )
        planned.append(
            {
                "ordinal": ordinal,
                "question_id": source["question_id"],
                "category": source["category"],
                "question_sha256": hashlib.sha256(
                    mem0_row["question"].encode()
                ).hexdigest(),
                "dated_question_sha256": hashlib.sha256(
                    mem0_row["dated_question"].encode()
                ).hexdigest(),
                "gold_answer_sha256": hashlib.sha256(
                    mem0_row["gold_answer"].encode()
                ).hexdigest(),
                "prediction_sha256": hashlib.sha256(
                    source["predicted_answer"].encode()
                ).hexdigest(),
                "answer_call_key_sha256": _digest(f"answer-call-{ordinal}"),
                "answer_response_journal_sha256": _digest(
                    f"answer-response-{ordinal}"
                ),
                "judge_messages_sha256": canonical_sha256(judge_messages),
                "judge_prompt_token_proxy": count_chat_prompt_token_proxy(
                    judge_messages
                ),
                "judge_output": source["judge_reasoning"],
            }
        )
    gold_population = [
        {
            "ordinal": row["ordinal"],
            "question_id": row["question_id"],
            "question_sha256": row["question_sha256"],
            "dated_question_sha256": row["dated_question_sha256"],
            "gold_answer_sha256": row["gold_answer_sha256"],
        }
        for row in planned
    ]
    gold_population_sha = canonical_sha256(gold_population)
    ordered_judgments = [
        {
            "ordinal": row["ordinal"],
            "question_id": row["question_id"],
            "question_sha256": row["question_sha256"],
            "gold_answer_sha256": row["gold_answer_sha256"],
            "prediction_sha256": row["prediction_sha256"],
            "judge_messages_sha256": row["judge_messages_sha256"],
            "answer_call_key_sha256": row["answer_call_key_sha256"],
            "answer_response_journal_sha256": row[
                "answer_response_journal_sha256"
            ],
        }
        for row in planned
    ]
    prompt_population = [
        {
            "messages_sha256": row["judge_messages_sha256"],
            "logical_references": 1,
        }
        for row in planned
    ]
    campaign = {
        "format": (
            "memory-condense-fixed-stage-final-answer-semantic-judge-campaign-v1"
        ),
        "final_answer_artifact_sha256": answer_artifact_sha,
        "responder_runtime_identity_sha256": responder_runtime_sha,
        "responder_prompt_policy": responder_prompt_policy,
        "responder_prompt_policy_sha256": responder_prompt_policy_sha,
        "retrieval_sha256": retrieval_sha,
        "population_identity_sha256": population_sha,
        "gold_scoring_population_sha256": gold_population_sha,
        "question_count": 100,
        "fixed_stage_id": FIXED_STAGE_ID,
        "ordered_judgment_population_sha256": canonical_sha256(
            ordered_judgments
        ),
        "logical_judgment_count": 100,
        "unique_judge_prompt_count": 100,
        "judge_prompt_population_sha256": canonical_sha256(prompt_population),
        "maximum_judge_prompt_token_proxy": max(
            row["judge_prompt_token_proxy"] for row in planned
        ),
        "authorized_unique_judge_calls": 100,
        "semantic_judge_policy_sha256": policy_sha,
        "semantic_judge_implementation_sha256": implementation_sha,
        "target_accuracy": 0.95,
        "minimum_questions": 100,
        "responder_model": FIXED_STAGE_TERRA_MODEL,
        "responder_max_new_tokens": FIXED_STAGE_RESPONDER_MAX_OUTPUT_TOKENS,
        "responder_prompt_cap": FIXED_STAGE_PROMPT_CAP,
        "judge_model": FIXED_STAGE_SOL_MODEL,
        "judge_max_new_tokens": FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS,
        "provider_retries": 0,
    }
    campaign_sha = canonical_sha256(campaign)
    runtime = {
        "format": "memory-condense-recall-guarded-semantic-judge-runtime-v1",
        "gateway_url": gateway_url,
        "caller_model": FIXED_STAGE_SOL_MODEL,
        "gateway_model": "codex_sdk/gpt-5.6-sol",
        "default_max_new_tokens": FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS,
        "retries": 0,
        "temperature": None,
        "authorized_unique_calls": 100,
        "campaign_binding": campaign,
        "campaign_binding_sha256": campaign_sha,
        "request_journal_format": (
            "memory-condense-semantic-judge-call-request-v1"
        ),
        "response_journal_format": (
            "memory-condense-semantic-judge-call-response-v1"
        ),
    }
    runtime_sha = canonical_sha256(runtime)
    rows = []
    for row in planned:
        ordinal = row["ordinal"]
        output_sha = hashlib.sha256(row["judge_output"].encode()).hexdigest()
        output_token_proxy = count_tokens(row["judge_output"])
        call_key = canonical_sha256(
            {
                "messages_sha256": row["judge_messages_sha256"],
                "runtime_identity_sha256": runtime_sha,
                "max_new_tokens": FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS,
                "campaign_binding_sha256": campaign_sha,
            }
        )
        request_sha = _digest(f"judge-request-{ordinal}")
        response_sha = _digest(f"judge-response-{ordinal}")
        completion = {
            "gateway_url": gateway_url,
            "caller_model": FIXED_STAGE_SOL_MODEL,
            "gateway_model": "codex_sdk/gpt-5.6-sol",
            "call_key_sha256": call_key,
            "runtime_identity_sha256": runtime_sha,
            "campaign_binding_sha256": campaign_sha,
            "request_journal_sha256": request_sha,
            "messages_sha256": row["judge_messages_sha256"],
            "completion_sha256": output_sha,
            "response_id": f"response-{ordinal}",
            "response_model": "codex_sdk/gpt-5.6-sol",
            "finish_reason": "stop",
            "max_new_tokens": FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS,
            "reported_usage_available": True,
            "reported_input_tokens": 90,
            "reported_output_tokens": 4,
            "reported_total_tokens": 94,
            "reported_input_tokens_available": True,
            "reported_output_tokens_available": True,
            "reported_total_tokens_available": True,
            "input_token_proxy": row["judge_prompt_token_proxy"],
            "output_token_proxy": output_token_proxy,
            "elapsed_s": 0.2,
            "retries": 0,
            "cache_hit": False,
            "physical_call": True,
            "cumulative_logical_calls": ordinal + 1,
            "cumulative_unique_calls": ordinal + 1,
            "cumulative_physical_calls": ordinal + 1,
            "cumulative_checkpoint_hits": 0,
        }
        correct = row["judge_output"].startswith("CORRECT")
        rows.append(
            {
                "ordinal": ordinal,
                "question_id": row["question_id"],
                "category": row["category"],
                "question_sha256": row["question_sha256"],
                "dated_question_sha256": row["dated_question_sha256"],
                "gold_answer_sha256": row["gold_answer_sha256"],
                "prediction_sha256": row["prediction_sha256"],
                "fixed_stage_id": FIXED_STAGE_ID,
                "answer_call_key_sha256": row["answer_call_key_sha256"],
                "answer_response_journal_sha256": row[
                    "answer_response_journal_sha256"
                ],
                "judge_messages_sha256": row["judge_messages_sha256"],
                "judge_prompt_token_proxy": row["judge_prompt_token_proxy"],
                "correct": correct,
                "judge_output": row["judge_output"],
                "judge_output_sha256": output_sha,
                "call_key_sha256": call_key,
                "request_journal_sha256": request_sha,
                "response_journal_sha256": response_sha,
                "completion_report": completion,
            }
        )
    aggregate = _fixed_accuracy_summary(95)
    return {
        "format": FIXED_STAGE_TREATMENT_FORMAT,
        "final_answer_artifact_sha256": answer_artifact_sha,
        "responder_runtime_identity_sha256": responder_runtime_sha,
        "responder_prompt_policy": responder_prompt_policy,
        "responder_prompt_policy_sha256": responder_prompt_policy_sha,
        "retrieval_sha256": retrieval_sha,
        "population_identity_sha256": population_sha,
        "gold_scoring_population_sha256": gold_population_sha,
        "question_count": 100,
        "gold_loaded_posthoc": True,
        "independent_llm_judge": True,
        "fixed_stage_id": FIXED_STAGE_ID,
        "responder_model": FIXED_STAGE_TERRA_MODEL,
        "judge_model": FIXED_STAGE_SOL_MODEL,
        "judge_runtime_identity": runtime,
        "judge_runtime_identity_sha256": runtime_sha,
        "semantic_judge_policy": policy,
        "semantic_judge_policy_sha256": policy_sha,
        "semantic_judge_implementation_sha256": implementation_sha,
        "campaign_binding": campaign,
        "campaign_binding_sha256": campaign_sha,
        "logical_judgment_count": 100,
        "unique_judge_prompt_count": 100,
        "deduplicated_logical_judgment_count": 0,
        "judge_prompt_preflight": {
            "completed_before_provider_calls": True,
            "logical_prompt_count": 100,
            "unique_prompt_count": 100,
            "maximum_prompt_token_proxy": max(
                row["judge_prompt_token_proxy"] for row in planned
            ),
        },
        "judge_usage": {
            "unique_journaled_calls": 100,
            "reported_input_tokens_available_calls": 100,
            "reported_input_tokens": 9_000,
            "reported_output_tokens_available_calls": 100,
            "reported_output_tokens": 400,
            "reported_total_tokens_available_calls": 100,
            "reported_total_tokens": 9_400,
            "input_token_proxy": sum(
                row["completion_report"]["input_token_proxy"] for row in rows
            ),
            "output_token_proxy": sum(
                row["completion_report"]["output_token_proxy"] for row in rows
            ),
            "elapsed_s": 20.0,
            "retries": 0,
        },
        "questions": rows,
        "category_counts": {"single-session-user": 100},
        "category_aggregates": [
            {"category": "single-session-user", **aggregate}
        ],
        "aggregate": aggregate,
        "target_gate": {
            **aggregate,
            "gate_unit": "one preregistered fixed retrieval stage",
            "fixed_stage_id": FIXED_STAGE_ID,
        },
    }


@pytest.fixture
def campaign_pair() -> tuple[dict, dict]:
    treatment = _treatment_campaign()
    return treatment, _mem0_campaign(treatment)


@pytest.fixture
def fixed_stage_campaign_pair(campaign_pair) -> tuple[dict, dict]:
    treatment, mem0 = copy.deepcopy(campaign_pair)
    mem0["config"]["judge_max_output_tokens"] = (
        FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS
    )
    return _fixed_stage_semantic_score(treatment, mem0), mem0


def test_accepts_hash_only_fixed_stage_semantic_score_as_schema_v3(
    fixed_stage_campaign_pair,
):
    treatment, mem0 = fixed_stage_campaign_pair

    result = compare_campaign_reports(treatment, mem0)

    assert result["schema_version"] == FIXED_STAGE_COMPARISON_SCHEMA_VERSION
    assert result["metric_comparison"] == {
        "valid": True,
        "status": (
            "paired_binary_judge_metrics_recomputed_from_hash_bound_rows"
        ),
        "num_questions": 100,
        "supported_metrics": ["binary_judge_accuracy"],
        "unsupported_metrics": [
            "f1",
            "exact_match",
            "context_tokens",
            "prompt_tokens",
        ],
    }
    assert result["arm_metrics"]["treatment"] == {
        "num_questions": 100,
        "correct": 95,
        "judge_accuracy": 0.95,
    }
    assert result["arm_metrics"]["mem0"]["judge_accuracy"] == 0.9
    assert result["paired_judge_outcomes"] == {
        "treatment_wins": 7,
        "ties": 91,
        "treatment_losses": 2,
    }
    assert result["certification"]["blocking_reasons"] == [
        "paired_source_population_identity_unverified",
        "shared_sampling_policy_identity_unverified",
        "shared_zero_retry_policy_identity_unverified",
        "shared_model_deployment_identity_unverified",
        "treatment_final_answer_artifact_derivation_unverified",
        "mem0_production_binding_certified_false",
    ]
    required = result["certification"]["required_schema_updates"]
    assert required["mem0_campaign"]["minimum_schema_version"] == 3
    assert "population_identity_sha256" in required["mem0_campaign"][
        "required_fields"
    ]
    assert result["shared_identity"][
        "responder_prompt_policy_identity_verified"
    ] is False
    assert result["shared_identity"][
        "responder_prompt_policy_identity_object_equal"
    ] is True
    assert result["shared_identity"][
        "responder_prompt_policy_derivation_verified"
    ] is False
    assert result["shared_identity"]["judge_prompt_derivation_verified"] is False
    assert result["certification"][
        "treatment_semantic_score_internal_structure_verified"
    ] is True
    assert result["certification"][
        "treatment_semantic_score_internal_contract_verified"
    ] is False
    assert result["certification"][
        "treatment_semantic_score_prompt_accounting_verified"
    ] is False
    assert result["paired_population_identity"][
        "same_source_population_certified"
    ] is False
    bindings = result["shared_identity"]["identity_hash_bindings"]
    assert bindings["treatment_responder_runtime_identity_sha256"] == (
        treatment["responder_runtime_identity_sha256"]
    )
    assert bindings["treatment_responder_prompt_policy_sha256"] == (
        treatment["responder_prompt_policy_sha256"]
    )
    assert bindings["treatment_judge_runtime_identity_sha256"] == (
        treatment["judge_runtime_identity_sha256"]
    )
    assert bindings["mem0_responder_model_identity_sha256"] == mem0[
        "model_identity"
    ]["responder_model_identity_sha256"]
    assert "predicted_answer" not in result["question_results"][0][
        "treatment"
    ]


def test_verifies_fixed_stage_prompt_derivation_from_bound_inputs(
    campaign_pair,
    monkeypatch: pytest.MonkeyPatch,
):
    legacy_treatment, mem0 = copy.deepcopy(campaign_pair)
    mem0["config"]["judge_max_output_tokens"] = (
        FIXED_STAGE_JUDGE_MAX_OUTPUT_TOKENS
    )
    responder_prompt_policy = build_responder_prompt_policy_identity(
        [row["messages"] for row in mem0["question_results"]]
    )
    retrieval = {
        "format": "test-fixed-stage-retrieval",
        "population_identity_sha256": _digest("fixed-source-population"),
    }
    retrieval_sha = _canonical_artifact_digest(retrieval)
    artifact = {
        "runtime_identity_sha256": _digest("fixed-responder-runtime"),
        "responder_prompt_policy_sha256": canonical_sha256(
            responder_prompt_policy
        ),
        "population_identity_sha256": _digest("fixed-source-population"),
        "retrieval_sha256": retrieval_sha,
        "campaign_binding": {
            "responder_prompt_policy": responder_prompt_policy,
        },
        "questions": [
            {
                "ordinal": ordinal,
                "question_id": source["question_id"],
                "question_sha256": hashlib.sha256(
                    mem0_row["question"].encode()
                ).hexdigest(),
                "dated_question_sha256": hashlib.sha256(
                    mem0_row["dated_question"].encode()
                ).hexdigest(),
                "answer": {
                    "text": source["predicted_answer"],
                    "sha256": hashlib.sha256(
                        source["predicted_answer"].encode()
                    ).hexdigest()
                },
                "call_key_sha256": _digest(f"answer-call-{ordinal}"),
                "response_journal_sha256": _digest(
                    f"answer-response-{ordinal}"
                ),
            }
            for ordinal, (source, mem0_row) in enumerate(
                zip(
                    legacy_treatment["question_results"],
                    mem0["question_results"],
                    strict=True,
                )
            )
        ],
    }
    artifact_sha = _canonical_artifact_digest(artifact)
    treatment = _fixed_stage_semantic_score(
        legacy_treatment,
        mem0,
        answer_artifact_sha256=artifact_sha,
        retrieval_sha256=retrieval_sha,
    )
    calls = []

    def validate(value, **kwargs):
        calls.append((value, kwargs))

    monkeypatch.setattr(
        "tools.mem0_eval.compare_fixed_stage_derivation."
        "validate_final_answer_artifact",
        validate,
    )

    result = compare_campaign_reports(
        treatment,
        mem0,
        fixed_stage_final_answer_artifact=artifact,
        fixed_stage_retrieval=retrieval,
    )

    assert len(calls) == 1
    assert result["shared_identity"][
        "responder_prompt_policy_identity_verified"
    ] is True
    assert result["shared_identity"][
        "responder_prompt_policy_derivation_verified"
    ] is True
    assert result["shared_identity"]["judge_prompt_derivation_verified"] is True
    assert result["certification"][
        "treatment_semantic_score_internal_contract_verified"
    ] is True
    assert result["certification"][
        "treatment_semantic_score_prompt_accounting_verified"
    ] is True
    assert (
        "treatment_final_answer_artifact_derivation_unverified"
        not in result["certification"]["blocking_reasons"]
    )

    drifted = copy.deepcopy(treatment)
    drifted_row = drifted["questions"][0]
    drifted_row["judge_prompt_token_proxy"] -= 1
    drifted_row["completion_report"]["input_token_proxy"] -= 1
    drifted["judge_usage"]["input_token_proxy"] -= 1
    with pytest.raises(PairedComparisonError, match="judge_prompt_token_proxy"):
        compare_campaign_reports(
            drifted,
            mem0,
            fixed_stage_final_answer_artifact=artifact,
            fixed_stage_retrieval=retrieval,
        )


def test_fixed_stage_prompt_derivation_requires_both_bound_inputs(
    fixed_stage_campaign_pair,
):
    treatment, mem0 = fixed_stage_campaign_pair

    with pytest.raises(PairedComparisonError, match="requires both"):
        compare_campaign_reports(
            treatment,
            mem0,
            fixed_stage_final_answer_artifact={},
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda treatment, _mem0: treatment.update(
                {"fixed_stage_id": "synthesized_episode_answer"}
            ),
            "fixed_stage_id",
        ),
        (
            lambda treatment, _mem0: treatment[
                "semantic_judge_policy"
            ].update({"responder_max_new_tokens": 257}),
            "responder_max_new_tokens",
        ),
        (
            lambda treatment, _mem0: treatment[
                "judge_runtime_identity"
            ].update({"temperature": 0}),
            "temperature",
        ),
        (
            lambda treatment, _mem0: treatment["questions"][0].update(
                {"gold_answer_sha256": _digest("different-gold")}
            ),
            "gold_scoring_population_sha256",
        ),
        (
            lambda treatment, _mem0: treatment.update(
                {"population_identity_sha256": _digest("other-population")}
            ),
            "population_identity_sha256",
        ),
        (
            lambda treatment, _mem0: treatment.update(
                {
                    "responder_prompt_policy_sha256": _digest(
                        "other-responder-prompt-policy"
                    )
                }
            ),
            "responder_prompt_policy_sha256",
        ),
        (
            lambda _treatment, mem0: mem0["config"].update(
                {"judge_max_output_tokens": 64}
            ),
            "judge_max_output_tokens",
        ),
    ],
)
def test_rejects_fixed_stage_contract_and_fairness_drift(
    fixed_stage_campaign_pair, mutation, match
):
    treatment, mem0 = copy.deepcopy(fixed_stage_campaign_pair)
    mutation(treatment, mem0)

    with pytest.raises(PairedComparisonError, match=match):
        compare_campaign_reports(treatment, mem0)


def test_rejects_fixed_stage_judge_output_tamper(fixed_stage_campaign_pair):
    treatment, mem0 = copy.deepcopy(fixed_stage_campaign_pair)
    treatment["questions"][0]["judge_output"] = "INCORRECT: tampered"

    with pytest.raises(PairedComparisonError, match="judge_output_sha256"):
        compare_campaign_reports(treatment, mem0)


def test_rejects_fixed_stage_replay_projection_as_immutable_call_report(
    fixed_stage_campaign_pair,
):
    treatment, mem0 = copy.deepcopy(fixed_stage_campaign_pair)
    for row in treatment["questions"]:
        row["completion_report"]["cache_hit"] = True
        row["completion_report"]["physical_call"] = False

    with pytest.raises(PairedComparisonError, match="cache_hit"):
        compare_campaign_reports(treatment, mem0)


def test_rejects_fixed_stage_journal_receipt_aliases(
    fixed_stage_campaign_pair,
):
    treatment, mem0 = copy.deepcopy(fixed_stage_campaign_pair)
    first, second = treatment["questions"][:2]
    second["request_journal_sha256"] = first["request_journal_sha256"]
    second["response_journal_sha256"] = first["response_journal_sha256"]
    second["completion_report"]["request_journal_sha256"] = first[
        "request_journal_sha256"
    ]

    with pytest.raises(PairedComparisonError, match="request_journal_sha256"):
        compare_campaign_reports(treatment, mem0)


def test_rejects_fixed_stage_cross_namespace_receipt_alias(
    fixed_stage_campaign_pair,
):
    treatment, mem0 = copy.deepcopy(fixed_stage_campaign_pair)
    row = treatment["questions"][0]
    row["response_journal_sha256"] = row["request_journal_sha256"]

    with pytest.raises(PairedComparisonError, match="namespaces"):
        compare_campaign_reports(treatment, mem0)


def test_rejects_fixed_stage_output_token_proxy_drift(
    fixed_stage_campaign_pair,
):
    treatment, mem0 = copy.deepcopy(fixed_stage_campaign_pair)
    treatment["questions"][0]["completion_report"]["output_token_proxy"] += 1
    treatment["judge_usage"]["output_token_proxy"] += 1

    with pytest.raises(PairedComparisonError, match="output_token_proxy"):
        compare_campaign_reports(treatment, mem0)


def test_rejects_fixed_stage_impossible_cumulative_counters(
    fixed_stage_campaign_pair,
):
    treatment, mem0 = copy.deepcopy(fixed_stage_campaign_pair)
    treatment["questions"][0]["completion_report"][
        "cumulative_unique_calls"
    ] = 1_000_000

    with pytest.raises(PairedComparisonError, match="cumulative call counters"):
        compare_campaign_reports(treatment, mem0)


@pytest.mark.parametrize(
    "field",
    (
        "prediction_sha256",
        "answer_call_key_sha256",
        "answer_response_journal_sha256",
    ),
)
def test_bound_inputs_reject_score_answer_provenance_drift(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
):
    prompt_policy = {"format": "test-responder-prompt-policy"}
    retrieval = {"format": "test-retrieval"}
    retrieval_sha = _canonical_artifact_digest(retrieval)
    artifact_row = {
        "ordinal": 0,
        "question_id": "q000",
        "question_sha256": _digest("question"),
        "dated_question_sha256": _digest("dated-question"),
        "answer": {"sha256": _digest("prediction")},
        "call_key_sha256": _digest("answer-call"),
        "response_journal_sha256": _digest("answer-response"),
    }
    artifact = {
        "runtime_identity_sha256": _digest("runtime"),
        "responder_prompt_policy_sha256": canonical_sha256(prompt_policy),
        "population_identity_sha256": _digest("population"),
        "retrieval_sha256": retrieval_sha,
        "campaign_binding": {"responder_prompt_policy": prompt_policy},
        "questions": [artifact_row],
    }
    score_row = {
        "ordinal": 0,
        "question_id": "q000",
        "question_sha256": _digest("question"),
        "dated_question_sha256": _digest("dated-question"),
        "prediction_sha256": _digest("prediction"),
        "answer_call_key_sha256": _digest("answer-call"),
        "answer_response_journal_sha256": _digest("answer-response"),
    }
    score_row[field] = _digest(f"drift-{field}")
    report = {
        "final_answer_artifact_sha256": _canonical_artifact_digest(artifact),
        "responder_runtime_identity_sha256": _digest("runtime"),
        "responder_prompt_policy": prompt_policy,
        "responder_prompt_policy_sha256": canonical_sha256(prompt_policy),
        "population_identity_sha256": _digest("population"),
        "retrieval_sha256": retrieval_sha,
        "questions": [score_row],
    }
    monkeypatch.setattr(
        "tools.mem0_eval.compare_fixed_stage_derivation."
        "validate_final_answer_artifact",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(PairedComparisonError, match=field):
        verify_treatment_prompt_derivation(
            report,
            final_answer_artifact=artifact,
            retrieval=retrieval,
            scoring_rows=[
                {
                    "question_id": "q000",
                    "question": "Question?",
                    "gold_answer": "Gold",
                }
            ],
        )


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
