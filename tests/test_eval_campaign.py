from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import tokenizer_proxy_identity
from memory_condense.eval.campaign import (
    CampaignMergeError,
    build_locked_validation_plan,
    main,
    merge_benchmark_reports,
    save_campaign_report,
)
from memory_condense.eval.cache_receipts import cache_receipts_sha256
from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    file_sha256,
    implementation_sha256,
)
from memory_condense.eval.validation_profile import (
    LONGMEMEVAL_1M_95_PROFILE,
    ValidationClaimProfileError,
    validate_longmemeval_claim_profile,
)


_DATASET_HASH = "1" * 64
_SPLIT_HASH = "2" * 64
_IMPLEMENTATION_HASH = "3" * 64
_ENVIRONMENT_HASH = "4" * 64
_POLICY_HASH = "5" * 64


def _usage(*, calls: int = 1, elapsed_s: float = 0.25) -> dict[str, int | float]:
    return {
        "input_tokens": 100,
        "output_tokens": 5,
        "cache_read_input_tokens": 10,
        "elapsed_s": elapsed_s,
        "calls": calls,
    }


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cache_receipts(
    sample_digest: str,
    shard: int,
    *,
    implementation_digest: str,
    environment_digest: str,
    turn_count: int = 1,
) -> dict[str, list[dict[str, object]]]:
    compiled_key = _digest(f"compiled-key-{shard}")
    execution_digest = _digest("embedding-execution")
    return {
        "compiled": [
            {
                "manifest_sha256": _digest(f"compiled-manifest-{shard}"),
                "cache_key": compiled_key,
                "sample_sha256": sample_digest,
                "database_sha256": _digest(f"compiled-database-{shard}"),
                "index_sha256": _digest(f"compiled-index-{shard}"),
                "embedding_execution_sha256": execution_digest,
                "implementation_sha256": implementation_digest,
                "environment_lock_sha256": environment_digest,
                "turn_count": turn_count,
                "chunk_count": 1,
            }
        ],
        "causal": [
            {
                "manifest_sha256": _digest(f"causal-manifest-{shard}"),
                "cache_key": _digest(f"causal-key-{shard}"),
                "sample_sha256": sample_digest,
                "compiled_cache_key": compiled_key,
                "database_sha256": _digest(f"causal-database-{shard}"),
                "index_sha256": _digest(f"causal-index-{shard}"),
                "build_protocol_sha256": _digest("causal-build-protocol"),
                "embedding_execution_sha256": execution_digest,
                "implementation_sha256": implementation_digest,
                "environment_lock_sha256": environment_digest,
            }
        ],
    }


def _question(
    question_id: str,
    *,
    correct: bool = True,
    f1: float = 1.0,
    context_tokens: int = 100,
    prompt_tokens: int = 200,
    category: str = "single-session-user",
) -> dict:
    return {
        "question_id": question_id,
        "question": f"Question {question_id}?",
        "gold_answer": "gold",
        "predicted_answer": "gold" if correct else "wrong",
        "category": category,
        "retrieved_chunks": ["evidence"],
        "f1": f1,
        "exact_match": correct,
        "judge_correct": correct,
        "judge_reasoning": "graded",
        "context_tokens": context_tokens,
        "prompt_token_proxy": prompt_tokens,
        "prompt_tokens": prompt_tokens,
        "responder_output_token_reserve": 256,
        "request_token_proxy": prompt_tokens + 256,
        "provider_prompt_budget_compliant": True,
        "transcript_tokens": 1_000_000,
        "context_fraction": context_tokens / 1_000_000,
        "transcript_token_savings": 1.0 - context_tokens / 1_000_000,
        "responder_usage": _usage(elapsed_s=0.25),
        "judge_usage": _usage(elapsed_s=0.5),
    }


def _report(shard: int, *, count: int = 10) -> dict:
    questions = [
        _question(
            f"q-{shard:02d}-{index:02d}",
            context_tokens=100 + shard * 10 + index,
            prompt_tokens=200 + shard * 10 + index,
            category="temporal-reasoning" if index % 2 else "multi-session",
        )
        for index in range(count)
    ]
    return {
        "config": {
            "chunker": {"min_tokens": 120, "max_tokens": 250},
            "retrieval": {
                "mode": "causal_graph",
                "k": 10,
                "coverage_selection": True,
            },
            "responder_model": "openai/codex_sdk/gpt-5.6-terra",
            "judge_model": "openai/codex_sdk/gpt-5.6-sol",
            "embedding_device": "cuda",
            "recent_window": 4,
            "accuracy_target": 0.95,
            "min_target_questions": 100,
            "max_prompt_tokens": 8000,
        },
        "benchmark": "longmemeval_s_cleaned",
        "samples": [
            {
                "sample_id": f"sample-{shard:02d}",
                "num_questions": count,
                "question_results": questions,
            }
        ],
        "num_samples": 1,
        "num_questions": count,
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "responder_output_token_reserve": 256,
        "max_prompt_token_proxy_observed": max(
            question["prompt_token_proxy"] for question in questions
        ),
        "max_prompt_tokens_observed": max(
            question["prompt_tokens"] for question in questions
        ),
        "prompt_token_proxy_budget_compliance": True,
        "provider_prompt_budget_compliance": True,
        "provider_input_usage_status": "complete",
        "prompt_budget_compliance": True,
        "accuracy_target": 0.95,
        "min_target_questions": 100,
        "dataset_sha256": _DATASET_HASH,
        "split_manifest_sha256": _SPLIT_HASH,
        "benchmark_split": "validation",
        "implementation_sha256": _IMPLEMENTATION_HASH,
        "environment_lock_sha256": _ENVIRONMENT_HASH,
        "policy_manifest_sha256": _POLICY_HASH,
        # Every ten-question validation shard is expected to be too small to
        # certify the campaign target by itself.
        "target_status": "insufficient_questions",
    }


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _ten_shards(tmp_path: Path) -> list[Path]:
    return [
        _write(tmp_path / f"shard-{shard:02d}.json", _report(shard))
        for shard in range(10)
    ]


def test_ten_by_ten_campaign_recomputes_question_weighted_result(tmp_path: Path):
    paths = _ten_shards(tmp_path)
    # Four incorrect answers still clear a 95% target. Shard-level statuses
    # remain insufficient_questions and must not poison the aggregate.
    for index, path in enumerate(paths[:4]):
        payload = json.loads(path.read_text(encoding="utf-8"))
        question = payload["samples"][0]["question_results"][index]
        question.update(
            {
                "f1": 0.0,
                "exact_match": False,
                "judge_correct": False,
                "predicted_answer": "wrong",
            }
        )
        _write(path, payload)

    result = merge_benchmark_reports(paths, min_questions=100, accuracy_target=0.95)

    assert result["num_questions"] == 100
    assert result["judge_accuracy"] == pytest.approx(0.96)
    assert result["exact_match_rate"] == pytest.approx(0.96)
    assert result["mean_f1"] == pytest.approx(0.96)
    assert result["target_status"] == "unverified_population"
    assert result["accuracy_target_met"] is False
    assert result["metric_accuracy_target_met"] is True
    assert result["locked_population_verified"] is False
    assert result["prompt_budget_compliance"] is True
    assert result["prompt_token_proxy_budget_compliance"] is True
    assert result["provider_prompt_budget_compliance"] is True
    assert result["provider_input_usage_status"] == "complete"
    assert (
        result["prompt_token_proxy_distribution"]
        == result["prompt_token_distribution"]
    )
    assert result["mean_prompt_token_proxy"] == result["mean_prompt_tokens"]
    assert result["responder_output_token_reserve"] == 256
    assert result["responder_usage"] == {
        "input_tokens": 10_000,
        "output_tokens": 500,
        "cache_read_input_tokens": 1_000,
        "elapsed_s": 25.0,
        "calls": 100,
    }
    assert result["judge_usage"]["calls"] == 100
    assert result["context_token_distribution"] == {
        "count": 100,
        "min": 100,
        "mean": 149.5,
        "p50": 149,
        "p90": 189,
        "p95": 194,
        "p99": 198,
        "max": 199,
        "values": list(range(100, 200)),
    }
    assert result["prompt_token_distribution"]["p95"] == 294
    assert result["by_category"]["multi-session"]["num_questions"] == 50
    assert result["by_category"]["temporal-reasoning"]["num_questions"] == 50
    assert len(result["question_results"]) == 100
    assert len(result["question_sources"]) == 100
    assert all(row["target_status"] == "insufficient_questions" for row in result["inputs"])
    inputs_by_name = {row["name"]: row for row in result["inputs"]}
    for path in paths:
        assert inputs_by_name[path.name]["sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("dataset_sha256",), "a" * 64),
        (("split_manifest_sha256",), "a" * 64),
        (("implementation_sha256",), "a" * 64),
        (("environment_lock_sha256",), "a" * 64),
        (("policy_manifest_sha256",), "a" * 64),
        (("config", "chunker", "max_tokens"), 300),
        (("config", "retrieval", "k"), 11),
        (("config", "responder_model"), "different/responder"),
        (("config", "judge_model"), "different/judge"),
        (("config", "embedding_device"), "cpu"),
        (("config", "recent_window"), 8),
        (("config", "max_prompt_tokens"), 9000),
    ],
)
def test_rejects_locked_identity_drift(
    tmp_path: Path, field_path: tuple[str, ...], replacement: object
):
    first = _report(0)
    drifted = copy.deepcopy(_report(1))
    target = drifted
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = replacement
    paths = [_write(tmp_path / "a.json", first), _write(tmp_path / "b.json", drifted)]

    with pytest.raises(CampaignMergeError, match="identity drift"):
        merge_benchmark_reports(paths, min_questions=100)


@pytest.mark.parametrize(
    ("field_path", "replacement", "message"),
    [
        (("config", "accuracy_target"), 0.90, "accuracy_target drift"),
        (("accuracy_target",), 0.90, "accuracy_target drift"),
        (("config", "min_target_questions"), 99, "min_target_questions drift"),
        (("min_target_questions",), 99, "min_target_questions drift"),
    ],
)
def test_rejects_campaign_threshold_drift(
    tmp_path: Path,
    field_path: tuple[str, ...],
    replacement: object,
    message: str,
):
    report = _report(0)
    target = report
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = replacement
    path = _write(tmp_path / "threshold-drift.json", report)

    with pytest.raises(CampaignMergeError, match=message):
        merge_benchmark_reports([path], min_questions=100, accuracy_target=0.95)


def test_rejects_non_validation_shard(tmp_path: Path):
    report = _report(0)
    report["benchmark_split"] = "development"
    path = _write(tmp_path / "development.json", report)

    with pytest.raises(CampaignMergeError, match="must be 'validation'"):
        merge_benchmark_reports([path], min_questions=100)


def test_rejects_missing_locked_identity(tmp_path: Path):
    report = _report(0)
    report.pop("policy_manifest_sha256")
    path = _write(tmp_path / "missing-policy.json", report)

    with pytest.raises(CampaignMergeError, match="policy_manifest_sha256"):
        merge_benchmark_reports([path], min_questions=100)


def test_rejects_duplicate_question_ids_across_shards(tmp_path: Path):
    first = _report(0)
    duplicate = _report(1)
    duplicate["samples"][0]["question_results"][0]["question_id"] = "q-00-00"
    paths = [
        _write(tmp_path / "a.json", first),
        _write(tmp_path / "b.json", duplicate),
    ]

    with pytest.raises(CampaignMergeError, match="duplicate question_id 'q-00-00'"):
        merge_benchmark_reports(paths, min_questions=100)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report.update(prompt_budget_compliance=False), "must be true"),
        (
            lambda report: report["samples"][0]["question_results"][0].update(
                error="provider timeout"
            ),
            "per-question error",
        ),
        (
            lambda report: report["samples"][0]["question_results"][0].pop(
                "judge_correct"
            ),
            "judge_correct must be a boolean",
        ),
        (
            lambda report: report["samples"][0]["question_results"][0].update(
                prompt_token_proxy=8001,
                prompt_tokens=8001,
                request_token_proxy=8257,
            ),
            "exceeds the locked prompt cap",
        ),
    ],
)
def test_rejects_incomplete_or_budget_violating_shard(
    tmp_path: Path, mutation, message: str
):
    report = _report(0)
    mutation(report)
    path = _write(tmp_path / "bad.json", report)

    with pytest.raises(CampaignMergeError, match=message):
        merge_benchmark_reports([path], min_questions=100)


def test_rejects_nonzero_provider_input_above_cap(tmp_path: Path):
    report = _report(0)
    question = report["samples"][0]["question_results"][0]
    question["responder_usage"]["input_tokens"] = 8001
    question["provider_prompt_budget_compliant"] = False
    report["provider_prompt_budget_compliance"] = False

    with pytest.raises(
        CampaignMergeError,
        match="provider input usage exceeds the locked prompt cap",
    ):
        merge_benchmark_reports(
            [_write(tmp_path / "provider-over-cap.json", report)],
            min_questions=100,
        )


def test_zero_provider_input_is_explicitly_unavailable(tmp_path: Path):
    report = _report(0)
    for question in report["samples"][0]["question_results"]:
        question["responder_usage"]["input_tokens"] = 0
        question["provider_prompt_budget_compliant"] = None
    report["provider_prompt_budget_compliance"] = None
    report["provider_input_usage_status"] = "unavailable"
    report["config"]["min_target_questions"] = 10
    report["min_target_questions"] = 10

    result = merge_benchmark_reports(
        [_write(tmp_path / "usage-unavailable.json", report)],
        min_questions=10,
    )

    assert result["provider_prompt_budget_compliance"] is None
    assert result["provider_input_usage_status"] == "unavailable"
    assert result["provider_input_token_distribution"]["count"] == 0


def test_requires_aggregate_question_floor(tmp_path: Path):
    paths = [
        _write(tmp_path / f"shard-{index}.json", _report(index))
        for index in range(9)
    ]

    with pytest.raises(CampaignMergeError, match="at least 100 are required"):
        merge_benchmark_reports(paths, min_questions=100)


def test_rejects_incorrect_reported_prompt_maximum(tmp_path: Path):
    report = _report(0)
    report["max_prompt_tokens_observed"] -= 1
    path = _write(tmp_path / "bad-maximum.json", report)

    with pytest.raises(CampaignMergeError, match="max_prompt_tokens_observed"):
        merge_benchmark_reports([path], min_questions=100)


def test_output_is_deterministic_and_cli_writes_it(tmp_path: Path):
    paths = _ten_shards(tmp_path)
    first = merge_benchmark_reports(paths, min_questions=100)
    second = merge_benchmark_reports(reversed(paths), min_questions=100)
    assert first == second

    direct_a = save_campaign_report(first, tmp_path / "direct-a.json")
    direct_b = save_campaign_report(second, tmp_path / "direct-b.json")
    assert direct_a.read_bytes() == direct_b.read_bytes()

    output = tmp_path / "cli.json"
    assert (
        main(
            [
                "--reports",
                *(str(path) for path in reversed(paths)),
                "--output",
                str(output),
                "--min-questions",
                "100",
                    "--accuracy-target",
                    "0.95",
                    "--allow-unverified-summary",
                ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == first


def test_campaign_identity_is_portable_across_artifact_directories(tmp_path: Path):
    first_paths = [
        _write(tmp_path / "first" / f"shard-{index:02d}.json", _report(index))
        for index in range(10)
    ]
    second_paths = [
        _write(tmp_path / "second" / path.name, json.loads(path.read_text()))
        for path in first_paths
    ]

    first = merge_benchmark_reports(first_paths)
    second = merge_benchmark_reports(second_paths)

    assert first == second
    expected_digest = hashlib.sha256(
        json.dumps(
            sorted(hashlib.sha256(path.read_bytes()).hexdigest() for path in first_paths),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert first["input_set_sha256"] == expected_digest
    assert all("/" not in row["name"] for row in first["inputs"])


def _locked_campaign_fixture(tmp_path: Path, *, certified: bool = False):
    population = 100 if certified else 4
    questions_per_shard = 10 if certified else 2
    record_count = population + 2
    records = [
        {
            "question_id": f"locked-q-{index}",
            "question_type": "single-session-user",
            "question": f"What was marker {index}?",
            "answer": f"gold-{index}",
            "haystack_session_ids": [f"session-{index}"],
            "haystack_sessions": [[{"role": "user", "content": "x"}]],
            "answer_session_ids": [f"session-{index}"],
        }
        for index in range(record_count)
    ]
    dataset = tmp_path / "locked-longmemeval.json"
    dataset.write_text(json.dumps(records), encoding="utf-8")
    split = tmp_path / "locked-split.json"
    split.write_text(
        json.dumps(
            {
                "format": "memory-condense-locked-benchmark-split-v1",
                "dataset_sha256": file_sha256(dataset),
                "salt": "campaign-test",
                "splits": {
                    "development": 1,
                    "validation": population,
                    "confirmation": 1,
                },
                "algorithm": "stratified-largest-remainder-v1",
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.csv"
    selection.write_text("locked selection\n", encoding="utf-8")
    evaluation = {
        "responder_model": "openai/codex_sdk/gpt-5.6-terra",
        "judge_model": "openai/codex_sdk/gpt-5.6-sol",
        "embedding_device": "cuda",
        "benchmark_format": "longmemeval",
        "use_judge": True,
        "provider_retries": 0,
        "max_provider_calls": 2 * questions_per_shard,
        "max_prompt_tokens": 8000,
        "prompt_cap_semantics": (
            "local_prompt_token_proxy_with_provider_usage_postcheck_v1"
        ),
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "responder_output_token_reserve": 256,
        "recent_window": 4,
        "accuracy_target": 0.95,
        "min_target_questions": population,
        "stress_context_tokens": 1_000_000 if certified else 2,
        "stress_questions": questions_per_shard,
        "stress_question_offset": 0,
        "max_samples": 1,
        "sample_offsets": list(range(0, population, questions_per_shard)),
    }
    retrieval = {
        "mode": "causal_graph",
        "k": 10,
        "max_prompt_tokens": 8000,
        "chunker_min_tokens": 120,
        "chunker_max_tokens": 250,
    }
    policy = tmp_path / "validation-policy.json"
    policy_payload = {
        "format": "memory-condense-retrieval-policy-v1",
        "status": "validation_frozen",
        "dataset_sha256": file_sha256(dataset),
        "split_manifest": split.name,
        "split_manifest_sha256": file_sha256(split),
        "split": "validation",
        "selection_artifact": selection.name,
        "selection_artifact_sha256": file_sha256(selection),
        "selection_artifact_required": True,
        "implementation_sha256": implementation_sha256(),
        "environment_lock_sha256": environment_lock_sha256(),
        "retrieval": retrieval,
        "evaluation": evaluation,
    }
    if certified:
        policy_payload["claim_profile"] = LONGMEMEVAL_1M_95_PROFILE
    policy.write_text(
        json.dumps(policy_payload),
        encoding="utf-8",
    )
    plan = build_locked_validation_plan(
        benchmark_file=dataset,
        benchmark_format="longmemeval",
        split_manifest=split,
        policy_manifest=policy,
        repository_root=tmp_path,
    )
    report_paths: list[Path] = []
    for shard_index, offset in enumerate(plan.sample_offsets):
        expected = plan.shards[offset]
        report = _report(shard_index, count=len(expected.questions))
        report.update(
            {
                "benchmark": dataset.stem,
                "dataset_sha256": plan.dataset_sha256,
                "split_manifest_sha256": plan.split_manifest_sha256,
                "implementation_sha256": plan.implementation_sha256,
                "environment_lock_sha256": plan.environment_lock_sha256,
                "policy_manifest_sha256": plan.policy_manifest_sha256,
                "accuracy_target": 0.95,
                "min_target_questions": population,
                "evaluation_protocol": {
                    **plan.evaluation,
                    "sample_offset": offset,
                },
            }
        )
        report["config"].update(
            {
                "retrieval": {"mode": "causal_graph", "k": 10},
                "accuracy_target": 0.95,
                "min_target_questions": population,
            }
        )
        sample = report["samples"][0]
        sample.update(
            {
                "sample_id": expected.sample_id,
                "sample_sha256": expected.sample_sha256,
                "num_turns": expected.num_turns,
            }
        )
        receipts = _cache_receipts(
            expected.sample_sha256,
            shard_index,
            implementation_digest=plan.implementation_sha256,
            environment_digest=plan.environment_lock_sha256,
            turn_count=expected.num_turns,
        )
        sample["cache_receipts"] = receipts
        sample["cache_receipts_sha256"] = cache_receipts_sha256(receipts)
        for question, expected_question in zip(
            sample["question_results"], expected.questions, strict=True
        ):
            question.update(expected_question)
            question["transcript_tokens"] = expected.transcript_tokens
            question["judge_reasoning"] = "CORRECT: equivalent answer"
        report_paths.append(
            _write(tmp_path / f"locked-shard-{offset}.json", report)
        )
    return plan, report_paths, selection, policy


def test_locked_campaign_reconstructs_and_certifies_exact_population(tmp_path: Path):
    plan, paths, _selection, _policy = _locked_campaign_fixture(tmp_path)

    result = merge_benchmark_reports(
        paths,
        min_questions=4,
        accuracy_target=0.95,
        locked_plan=plan,
    )

    assert result["num_questions"] == 4
    assert result["judge_accuracy"] == 1.0
    assert result["target_status"] == "unverified_claim_profile"
    assert result["accuracy_target_met"] is False
    assert result["locked_population_verified"] is True
    assert result["claim_profile_verified"] is False
    assert result["evaluation_protocol"] == plan.evaluation
    assert set(result["question_sources"]) == set(plan.question_ids)
    assert set(result["cache_receipts_by_sample"]) == {
        shard.sample_sha256 for shard in plan.shards.values()
    }


def test_locked_campaign_rejects_forged_profile_verification(tmp_path: Path):
    plan, paths, _selection, _policy = _locked_campaign_fixture(tmp_path)
    forged_plan = replace(
        plan,
        claim_profile=LONGMEMEVAL_1M_95_PROFILE,
        claim_profile_verified=True,
    )

    with pytest.raises(CampaignMergeError, match="claim profile disagrees"):
        merge_benchmark_reports(
            paths,
            min_questions=4,
            accuracy_target=0.95,
            locked_plan=forged_plan,
        )


def test_locked_campaign_passes_only_exact_named_100_question_profile(
    tmp_path: Path,
    monkeypatch,
):
    import memory_condense.eval.context_stress as context_stress_module

    def bounded_fixture_tokens(sample):
        if sample.sample_id.startswith("context-stress-"):
            return 1_000_000
        return 100_000

    # Keep this exact-population integration test CPU-small while exercising
    # the real locked split and stress-shard reconstruction.  Production uses
    # the unpatched tokenizer-backed counter.
    monkeypatch.setattr(
        context_stress_module,
        "transcript_tokens",
        bounded_fixture_tokens,
    )
    monkeypatch.setattr(
        "memory_condense.eval.campaign.transcript_tokens",
        bounded_fixture_tokens,
    )
    plan, paths, _selection, _policy = _locked_campaign_fixture(
        tmp_path,
        certified=True,
    )

    result = merge_benchmark_reports(
        paths,
        min_questions=100,
        accuracy_target=0.95,
        locked_plan=plan,
    )

    assert result["num_questions"] == 100
    assert result["target_status"] == "passed"
    assert result["accuracy_target_met"] is True
    assert result["claim_profile"] == LONGMEMEVAL_1M_95_PROFILE
    assert result["claim_profile_verified"] is True


def test_locked_campaign_cli_requires_and_uses_source_manifests(tmp_path: Path):
    _plan, paths, _selection, policy = _locked_campaign_fixture(tmp_path)
    output = tmp_path / "locked-campaign.json"

    assert (
        main(
            [
                "--reports",
                *(str(path) for path in reversed(paths)),
                "--output",
                str(output),
                "--min-questions",
                "4",
                "--accuracy-target",
                "0.95",
                "--benchmark-file",
                str(tmp_path / "locked-longmemeval.json"),
                "--benchmark-format",
                "longmemeval",
                "--split-manifest",
                str(tmp_path / "locked-split.json"),
                "--policy-manifest",
                str(policy),
                "--repository-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["target_status"] == "unverified_claim_profile"
    assert result["locked_population_verified"] is True
    assert (tmp_path / "locked-campaign.json.sha256").is_file()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda report: report["samples"][0].update(
                sample_sha256="f" * 64
            ),
            "sample_sha256",
        ),
        (
            lambda report: report["samples"][0].update(num_turns=999),
            "num_turns",
        ),
        (
            lambda report: report["samples"][0]["question_results"][0].update(
                gold_answer="tampered"
            ),
            "gold_answer",
        ),
        (
            lambda report: report["samples"][0]["question_results"][0].update(
                transcript_tokens=999
            ),
            "transcript_tokens",
        ),
        (
            lambda report: report["samples"][0]["question_results"][0][
                "judge_usage"
            ].update(calls=0),
            "exactly one completed",
        ),
        (
            lambda report: report["samples"][0]["question_results"][0].update(
                judge_reasoning="CORRECTNESS"
            ),
            "exact binary judge verdict",
        ),
        (
            lambda report: report["samples"][0]["question_results"][0].update(
                judge_correct=False
            ),
            "disagrees with the provider verdict",
        ),
        (
            lambda report: report["samples"][0]["question_results"][0].update(
                predicted_answer=""
            ),
            "predicted_answer",
        ),
    ),
)
def test_locked_campaign_rejects_sample_or_provider_tampering(
    tmp_path: Path, mutation, message: str
):
    plan, paths, _selection, _policy = _locked_campaign_fixture(tmp_path)
    report = json.loads(paths[0].read_text(encoding="utf-8"))
    mutation(report)
    _write(paths[0], report)

    with pytest.raises(CampaignMergeError, match=message):
        merge_benchmark_reports(
            paths,
            min_questions=4,
            accuracy_target=0.95,
            locked_plan=plan,
        )


def test_locked_campaign_rejects_missing_or_duplicate_shard_offsets(tmp_path: Path):
    plan, paths, _selection, _policy = _locked_campaign_fixture(tmp_path)
    with pytest.raises(CampaignMergeError, match="missing frozen validation shards"):
        merge_benchmark_reports(
            paths[:1],
            min_questions=4,
            accuracy_target=0.95,
            locked_plan=plan,
        )

    second = json.loads(paths[0].read_text(encoding="utf-8"))
    _write(paths[1], second)
    with pytest.raises(CampaignMergeError, match="duplicate validation sample_offset"):
        merge_benchmark_reports(
            paths,
            min_questions=4,
            accuracy_target=0.95,
            locked_plan=plan,
        )


def test_locked_campaign_rejects_retrieval_drift(tmp_path: Path):
    plan, paths, _selection, _policy = _locked_campaign_fixture(tmp_path)
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        report["config"]["retrieval"]["k"] = 9
        _write(path, report)

    with pytest.raises(CampaignMergeError, match="frozen retrieval policy"):
        merge_benchmark_reports(
            paths,
            min_questions=4,
            accuracy_target=0.95,
            locked_plan=plan,
        )


def test_locked_campaign_rejects_cache_receipt_tampering(tmp_path: Path):
    plan, paths, _selection, _policy = _locked_campaign_fixture(tmp_path)
    report = json.loads(paths[0].read_text(encoding="utf-8"))
    report["samples"][0]["cache_receipts"]["causal"][0][
        "compiled_cache_key"
    ] = "f" * 64
    _write(paths[0], report)

    with pytest.raises(CampaignMergeError, match="compiled cache key"):
        merge_benchmark_reports(
            paths,
            min_questions=4,
            accuracy_target=0.95,
            locked_plan=plan,
        )


def test_locked_campaign_rejects_self_consistent_wrong_cache_turn_count(
    tmp_path: Path,
):
    plan, paths, _selection, _policy = _locked_campaign_fixture(tmp_path)
    report = json.loads(paths[0].read_text(encoding="utf-8"))
    sample = report["samples"][0]
    sample["cache_receipts"]["compiled"][0]["turn_count"] += 1
    sample["cache_receipts_sha256"] = cache_receipts_sha256(
        sample["cache_receipts"]
    )
    _write(paths[0], report)

    with pytest.raises(CampaignMergeError, match="turn_count"):
        merge_benchmark_reports(
            paths,
            min_questions=4,
            accuracy_target=0.95,
            locked_plan=plan,
        )


def test_longmemeval_claim_profile_rejects_weak_or_cherry_picked_values():
    evaluation = {
        "accuracy_target": 0.95,
        "min_target_questions": 100,
        "stress_context_tokens": 1_000_000,
        "stress_questions": 10,
        "stress_question_offset": 0,
        "max_samples": 1,
        "recent_window": 4,
        "sample_offsets": list(range(0, 100, 10)),
    }
    policy = {"claim_profile": LONGMEMEVAL_1M_95_PROFILE}
    assert (
        validate_longmemeval_claim_profile(
            policy,
            evaluation,
            population_size=100,
        )
        == LONGMEMEVAL_1M_95_PROFILE
    )

    for field, value in (
        ("accuracy_target", 0.94),
        ("min_target_questions", 90),
        ("stress_context_tokens", 999_999),
        ("stress_questions", 9),
        ("recent_window", 3),
        ("sample_offsets", list(range(0, 90, 10))),
    ):
        weakened = {**evaluation, field: value}
        with pytest.raises(ValidationClaimProfileError):
            validate_longmemeval_claim_profile(
                policy,
                weakened,
                population_size=100,
            )


def test_locked_campaign_rechecks_sources_before_certification(tmp_path: Path):
    plan, paths, selection, _policy = _locked_campaign_fixture(tmp_path)
    selection.write_text("changed after plan construction\n", encoding="utf-8")

    with pytest.raises(CampaignMergeError, match="selection artifact changed"):
        merge_benchmark_reports(
            paths,
            min_questions=4,
            accuracy_target=0.95,
            locked_plan=plan,
        )


def test_locked_plan_rejects_cherry_picked_population(tmp_path: Path):
    _plan, _paths, _selection, policy_path = _locked_campaign_fixture(tmp_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["evaluation"]["sample_offsets"] = [0]
    policy["evaluation"]["min_target_questions"] = 2
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(CampaignMergeError, match="exact locked population"):
        build_locked_validation_plan(
            benchmark_file=tmp_path / "locked-longmemeval.json",
            benchmark_format="longmemeval",
            split_manifest=tmp_path / "locked-split.json",
            policy_manifest=policy_path,
            repository_root=tmp_path,
        )
