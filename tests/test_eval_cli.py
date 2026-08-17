from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest

from memory_condense.eval.__main__ import (
    _apply_sample_offset,
    _make_answer_fn,
    _make_judge_fn,
    _planned_provider_calls,
    _verified_policy_sha256,
    build_parser,
    config_from_args,
)
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.loader import BenchmarkQuestion, BenchmarkSample


def _sample(question_count: int) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"sample-{question_count}",
        turns=[("user", "history")],
        questions=[
            BenchmarkQuestion(
                question_id=f"q-{index}",
                question="question",
                answer="answer",
            )
            for index in range(question_count)
        ],
    )


def test_remote_benchmark_defaults_refuse_calls_and_retries():
    args = build_parser().parse_args(["--benchmark-file", "sample.json"])

    assert args.max_provider_calls == 0
    assert args.provider_retries == 0


def test_sufficiency_audit_is_a_first_class_cli_mode():
    args = build_parser().parse_args(
        ["--sufficiency-audit", "sample.json", "--max-samples", "2"]
    )

    assert args.sufficiency_audit == "sample.json"
    assert args.max_samples == 2


def test_source_local_search_is_explicit_and_defaults_off():
    parser = build_parser()

    assert parser.parse_args(["--benchmark-file", "sample.json"]).source_local_search is False
    assert (
        parser.parse_args(
            ["--benchmark-file", "sample.json", "--source-local-search"]
        ).source_local_search
        is True
    )


def test_source_tfisf_activation_is_explicit_and_bounded():
    parser = build_parser()
    baseline = config_from_args(
        parser.parse_args(["--benchmark-file", "sample.json"])
    )
    treatment = config_from_args(
        parser.parse_args(
            [
                "--benchmark-file",
                "sample.json",
                "--source-tfisf-activation",
                "--source-tfisf-slots",
                "5",
            ]
        )
    )

    assert baseline.retrieval.source_tfisf_activation is False
    assert treatment.retrieval.source_tfisf_activation is True
    assert treatment.retrieval.source_tfisf_slots == 5


def test_source_hsc_activation_is_explicit_and_bounded():
    parser = build_parser()
    treatment = config_from_args(
        parser.parse_args(
            [
                "--benchmark-file",
                "sample.json",
                "--mode",
                "causal_graph",
                "--source-hsc-activation",
                "--source-hsc-slots",
                "6",
                "--source-hsc-hops",
                "3",
                "--source-hsc-chunk-slots",
                "4",
            ]
        )
    )

    assert treatment.retrieval.source_hsc_activation is True
    assert treatment.retrieval.source_hsc_slots == 6
    assert treatment.retrieval.source_hsc_hops == 3
    assert treatment.retrieval.source_hsc_chunk_slots == 4


def test_information_gain_packing_is_explicit_and_thresholded():
    parser = build_parser()
    treatment = config_from_args(
        parser.parse_args(
            [
                "--benchmark-file",
                "sample.json",
                "--consolidation-information-gain-packing",
                "--consolidation-min-information-gain-per-token",
                "0.005",
            ]
        )
    )

    assert treatment.retrieval.consolidation_information_gain_packing is True
    assert (
        treatment.retrieval.consolidation_min_information_gain_per_token
        == 0.005
    )


def test_source_partition_routing_is_explicit_and_bounded():
    parser = build_parser()
    baseline = config_from_args(
        parser.parse_args(["--benchmark-file", "sample.json"])
    )
    routed = config_from_args(
        parser.parse_args(
            [
                "--benchmark-file",
                "sample.json",
                "--mode",
                "causal_graph",
                "--source-partition-routing",
                "--source-partition-slots",
                "2",
            ]
        )
    )

    assert baseline.retrieval.source_partition_routing is False
    assert routed.retrieval.source_partition_routing is True
    assert routed.retrieval.source_partition_slots == 2


def test_qwen_reranker_is_enabled_only_by_an_explicit_checkpoint():
    parser = build_parser()
    baseline = config_from_args(
        parser.parse_args(["--benchmark-file", "sample.json"])
    )
    treatment = config_from_args(
        parser.parse_args(
            [
                "--benchmark-file",
                "sample.json",
                "--mode",
                "causal_graph",
                "--source-local-search",
                "--qwen-rerank-model-dir",
                ".cache/models/Qwen3-8B",
                "--qwen-rerank-slots",
                "6",
            ]
        )
    )

    assert baseline.retrieval.qwen_rerank is False
    assert treatment.retrieval.qwen_rerank is True
    assert treatment.retrieval.qwen_rerank_slots == 6


def test_qwen_feedback_selects_second_hop_instead_of_direct_reranking():
    parser = build_parser()
    config = config_from_args(
        parser.parse_args(
            [
                "--benchmark-file",
                "sample.json",
                "--mode",
                "causal_graph",
                "--source-local-search",
                "--qwen-rerank-model-dir",
                ".cache/models/Qwen3-8B",
                "--qwen-feedback",
                "--qwen-feedback-slots",
                "10",
            ]
        )
    )

    assert config.retrieval.qwen_feedback is True
    assert config.retrieval.qwen_rerank is False
    assert config.retrieval.qwen_feedback_slots == 10
    with pytest.raises(ValueError, match="requires --qwen-rerank-model-dir"):
        config_from_args(
            parser.parse_args(
                [
                    "--benchmark-file",
                    "sample.json",
                    "--mode",
                    "causal_graph",
                    "--source-local-search",
                    "--qwen-feedback",
                ]
            )
        )


def test_planned_provider_calls_respects_sample_limit_and_local_answerer():
    samples = [_sample(2), _sample(3)]

    assert _planned_provider_calls(
        samples,
        max_samples=1,
        local_answerer=False,
        use_judge=True,
    ) == 4
    assert _planned_provider_calls(
        samples,
        max_samples=None,
        local_answerer=True,
        use_judge=True,
    ) == 5


def test_sample_offset_supports_non_overlapping_benchmark_shards():
    samples = [_sample(1), _sample(2), _sample(3)]
    args = SimpleNamespace(sample_offset=1)

    assert _apply_sample_offset(args, samples) == samples[1:]
    with pytest.raises(ValueError, match="outside"):
        _apply_sample_offset(SimpleNamespace(sample_offset=3), samples)
    with pytest.raises(ValueError, match="non-negative"):
        _apply_sample_offset(SimpleNamespace(sample_offset=-1), samples)


def test_metered_provider_wrappers_disable_retries(monkeypatch):
    calls = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="CORRECT"))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
    )

    def completion(**kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(litellm, "completion", completion)
    answer, answer_usage = _make_answer_fn("anthropic/test")([
        {"role": "user", "content": "question"}
    ])
    correct, reasoning, judge_usage = _make_judge_fn("anthropic/judge")(
        "question", "gold", "prediction"
    )

    assert answer == "CORRECT"
    assert correct is True
    assert reasoning == "CORRECT"
    assert answer_usage.calls == 1
    assert judge_usage.calls == 1
    assert [call["num_retries"] for call in calls] == [0, 0]


def test_policy_manifest_must_match_active_retrieval_config(tmp_path: Path):
    config = EvalConfig(
        retrieval=RetrievalConfig(
            mode="hybrid_neighbor",
            k=10,
            neighbor_radius=6,
            neighbor_slots=23,
        ),
        max_prompt_tokens=8000,
    )
    retrieval = {
        "mode": "hybrid_neighbor",
        "k": 10,
        "ef_search": 50,
        "alpha": 0.65,
        "candidates": 100,
        "neighbor_radius": 6,
        "neighbor_slots": 23,
        "neighbor_replacement_slots": 0,
        "max_prompt_tokens": 8000,
        "chunker_min_tokens": 120,
        "chunker_max_tokens": 250,
    }
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "status": "development_candidate_not_validated",
                "dataset_sha256": "a" * 64,
                "split_manifest": "split.json",
                "retrieval": retrieval,
            }
        ),
        encoding="utf-8",
    )

    assert len(
        _verified_policy_sha256(
            path,
            config=config,
            dataset_sha256="a" * 64,
            split_manifest=str(tmp_path / "split.json"),
        )
    ) == 64

    path.write_text(
        json.dumps(
            {
                "status": "superseded_after_fix",
                "dataset_sha256": "a" * 64,
                "split_manifest": "split.json",
                "retrieval": retrieval,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not active"):
        _verified_policy_sha256(
            path,
            config=config,
            dataset_sha256="a" * 64,
            split_manifest=str(tmp_path / "split.json"),
        )
