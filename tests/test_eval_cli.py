from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest

from memory_condense.eval.__main__ import (
    _LazyCrossEncoderCoverageSelector,
    _LazyQwenPrefixCoverageSelector,
    _apply_sample_offset,
    _assert_implementation_unchanged,
    _attach_coverage_selector,
    _benchmark_evaluation_identity,
    _benchmark_ingest_fn,
    _load_coverage_selector,
    _make_answer_fn,
    _make_central_dev_client,
    _make_judge_fn,
    _make_sufficiency_fn,
    _planned_provider_calls,
    _parse_binary_judge_verdict,
    _coverage_prefix_policy_identity,
    _policy_retrieval_identity,
    _reserve_embedding_device_for_transient_models,
    _validated_blind_cache_receipts,
    _validate_prepare_cache_args,
    _verified_policy_sha256,
    build_parser,
    config_from_args,
    run_benchmark_mode,
    run_prepare_cache_only,
)
from memory_condense.eval.compiled_cache import sample_sha256
from memory_condense.eval.reproducibility import file_sha256
from memory_condense.eval.schemas import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_RESPONDER_MODEL,
    EvalConfig,
    RetrievalConfig,
)
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample


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


def test_prepare_cache_tracks_explicit_responder_and_judge_flags():
    parser = build_parser()

    omitted = parser.parse_args(["--benchmark-file", "sample.json"])
    explicit_responder = parser.parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--responder-model",
            DEFAULT_RESPONDER_MODEL,
        ]
    )
    explicit_judge = parser.parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--judge-model",
            DEFAULT_JUDGE_MODEL,
        ]
    )

    assert not getattr(omitted, "_responder_model_explicit", False)
    assert not getattr(omitted, "_judge_model_explicit", False)
    assert explicit_responder._responder_model_explicit is True
    assert explicit_judge._judge_model_explicit is True


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


def test_query_facet_retrieval_is_explicit_and_bounded():
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
                "--source-slots",
                "12",
                "--query-facet-retrieval",
                "--query-facet-slots",
                "5",
                "--query-facet-max",
                "3",
            ]
        )
    )

    assert baseline.retrieval.query_facet_retrieval is False
    assert treatment.retrieval.query_facet_retrieval is True
    assert treatment.retrieval.query_facet_slots == 5
    assert treatment.retrieval.query_facet_max == 3


def test_role_aware_retrieval_is_explicit_and_weighted():
    parser = build_parser()
    treatment = config_from_args(
        parser.parse_args(
            [
                "--benchmark-file",
                "sample.json",
                "--mode",
                "causal_graph",
                "--role-aware-retrieval",
                "--role-user-weight",
                "1.4",
                "--role-assistant-weight",
                "0.6",
            ]
        )
    )

    assert treatment.retrieval.role_aware_retrieval is True
    assert treatment.retrieval.role_user_weight == 1.4
    assert treatment.retrieval.role_assistant_weight == 0.6


def test_multi_fact_source_diversity_is_explicit():
    parser = build_parser()
    treatment = config_from_args(
        parser.parse_args(
            [
                "--benchmark-file",
                "sample.json",
                "--mode",
                "causal_graph",
                "--multi-fact-source-diversity",
            ]
        )
    )

    assert treatment.retrieval.multi_fact_source_diversity is True


def test_local_coverage_selector_is_explicit_and_bounded():
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
                "--coverage-selector-local-model-dir",
                "models/Qwen3-0.6B",
                "--coverage-selector-candidate-pool",
                "32",
                "--coverage-selector-candidate-tokens",
                "72",
                "--coverage-selector-max-workspace-tokens",
                "4096",
                "--coverage-selector-dtype",
                "float16",
            ]
        )
    )

    assert baseline.retrieval.coverage_selection is False
    assert treatment.retrieval.coverage_selection is True
    assert treatment.retrieval.coverage_selector_backend == "local_ini"
    assert treatment.retrieval.coverage_selector_model == "Qwen3-0.6B"
    assert treatment.retrieval.coverage_selector_candidate_pool == 32
    assert treatment.retrieval.coverage_selector_candidate_tokens == 72
    assert treatment.retrieval.coverage_selector_max_workspace_tokens == 4096
    assert treatment.retrieval.coverage_selector_dtype == "float16"


def test_selected_scope_fixed_k_closure_is_explicit_and_policy_frozen():
    parser = build_parser()
    baseline = config_from_args(
        parser.parse_args(
            [
                "--benchmark-file",
                "sample.json",
                "--mode",
                "causal_graph",
                "--source-partition-routing",
                "--coverage-selector-qwen-prefix-model-dir",
                "models/Qwen3-8B",
            ]
        )
    )
    treatment = config_from_args(
        parser.parse_args(
            [
                "--benchmark-file",
                "sample.json",
                "--mode",
                "causal_graph",
                "--source-partition-routing",
                "--coverage-selector-qwen-prefix-model-dir",
                "models/Qwen3-8B",
                "--allow-selected-scope-fixed-k-closure",
            ]
        )
    )

    assert baseline.retrieval.allow_selected_scope_fixed_k_closure is False
    assert treatment.retrieval.allow_selected_scope_fixed_k_closure is True
    assert treatment.retrieval.label.endswith("-selected-scope-closure")
    assert "allow_selected_scope_fixed_k_closure" not in (
        _policy_retrieval_identity(baseline)
    )
    assert _policy_retrieval_identity(treatment)[
        "allow_selected_scope_fixed_k_closure"
    ] is True


def test_qwen_prefix_coverage_selector_is_distinct_and_six_layers():
    parser = build_parser()
    treatment = config_from_args(
        parser.parse_args(
            [
                "--benchmark-file",
                "sample.json",
                "--mode",
                "causal_graph",
                "--coverage-selector-qwen-prefix-model-dir",
                "models/Qwen3-8B",
                "--coverage-selector-candidate-pool",
                "24",
                "--coverage-selector-prefix-layers",
                "6",
                "--coverage-selector-attention-layer",
                "5",
                "--coverage-selector-dtype",
                "float16",
            ]
        )
    )

    assert treatment.retrieval.coverage_selection is True
    assert treatment.retrieval.coverage_selector_backend == "qwen_prefix"
    assert treatment.retrieval.coverage_selector_model == "Qwen3-8B"
    assert treatment.retrieval.coverage_selector_candidate_pool == 24
    assert treatment.retrieval.coverage_selector_prefix_layers == 6
    assert treatment.retrieval.coverage_selector_attention_layer == 5
    assert (
        treatment.retrieval.coverage_selector_prefix_model_id
        == "Qwen/Qwen3-8B"
    )
    assert (
        treatment.retrieval.coverage_selector_prefix_revision
        == "b968826d9c46dd6066d109eabc6255188de91218"
    )
    assert len(
        treatment.retrieval.coverage_selector_prefix_checkpoint_sha256
    ) == 64
    assert treatment.retrieval.coverage_selector_prefix_device == "cuda"
    assert treatment.retrieval.coverage_selector_prefix_dtype == "float16"
    assert treatment.retrieval.label.endswith("-coverage-qwen-prefix")

    policy_identity = _coverage_prefix_policy_identity(treatment)
    assert policy_identity == {
        "coverage_selector_prefix_model_id": "Qwen/Qwen3-8B",
        "coverage_selector_prefix_revision": (
            "b968826d9c46dd6066d109eabc6255188de91218"
        ),
        "coverage_selector_prefix_checkpoint_sha256": (
            treatment.retrieval.coverage_selector_prefix_checkpoint_sha256
        ),
        "coverage_selector_prefix_device": "cuda",
        "coverage_selector_prefix_dtype": "float16",
    }


@pytest.mark.parametrize(
    (
        "checkpoint_name",
        "expected_model_id",
        "expected_revision",
        "expected_sha256",
    ),
    [
        (
            "Qwen3-0.6B",
            "Qwen/Qwen3-0.6B",
            "c1899de289a04d12100db370d81485cdf75e47ca",
            "a940db06d5d9a3b298412376966b492f09ad7f088495fb75c05aa45db943d86e",
        ),
        (
            "SmolLM2-360M-Instruct",
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            "a10cc1512eabd3dde888204e902eca88bddb4951",
            "0a3f2e4e51ebb669743372be34afd46f4a004bc8ad06e4c8f5de45c1c74fa629",
        ),
    ],
)
def test_choice_coverage_backend_infers_only_pinned_checkpoint_identities(
    checkpoint_name,
    expected_model_id,
    expected_revision,
    expected_sha256,
):
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--mode",
            "causal_graph",
            "--coverage-selector-qwen-prefix-model-dir",
            "models/Qwen3-8B",
            "--coverage-selector-choice-model-dir",
            f"models/{checkpoint_name}",
            "--coverage-selector-choice-batch-size",
            "4",
            "--coverage-selector-choice-max-candidates",
            "96",
        ]
    )

    treatment = config_from_args(args)

    assert treatment.retrieval.coverage_selector_backend == "qwen_prefix_choice"
    assert treatment.retrieval.coverage_selector_choice_model_id == expected_model_id
    assert treatment.retrieval.coverage_selector_choice_revision == expected_revision
    assert (
        treatment.retrieval.coverage_selector_choice_checkpoint_sha256
        == expected_sha256
    )
    assert treatment.retrieval.coverage_selector_choice_batch_size == 4
    assert treatment.retrieval.coverage_selector_choice_max_candidates == 96
    assert checkpoint_name.casefold().replace(".", "-") in (
        treatment.retrieval.label
    )


def test_choice_coverage_requires_prefix_and_exact_unknown_identity():
    parser = build_parser()
    without_prefix = parser.parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--mode",
            "causal_graph",
            "--coverage-selector-choice-model-dir",
            "models/Qwen3-0.6B",
        ]
    )
    with pytest.raises(ValueError, match="requires.*qwen-prefix-model-dir"):
        config_from_args(without_prefix)

    unknown = parser.parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--mode",
            "causal_graph",
            "--coverage-selector-qwen-prefix-model-dir",
            "models/Qwen3-8B",
            "--coverage-selector-choice-model-dir",
            "models/custom-causal",
        ]
    )
    with pytest.raises(ValueError, match="unknown choice checkpoint"):
        config_from_args(unknown)

    explicit = parser.parse_args(
        [
            *[
                "--benchmark-file",
                "sample.json",
                "--mode",
                "causal_graph",
                "--coverage-selector-qwen-prefix-model-dir",
                "models/Qwen3-8B",
                "--coverage-selector-choice-model-dir",
                "models/custom-causal",
            ],
            "--coverage-selector-choice-model-id",
            "example/custom-causal",
            "--coverage-selector-choice-model-revision",
            "exact-revision",
            "--coverage-selector-choice-checkpoint-sha256",
            "a" * 64,
        ]
    )
    config = config_from_args(explicit)
    assert config.retrieval.coverage_selector_choice_model_id == (
        "example/custom-causal"
    )


def test_choice_coverage_is_mutually_exclusive_with_other_score_backends():
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--mode",
            "causal_graph",
            "--coverage-selector-qwen-prefix-model-dir",
            "models/Qwen3-8B",
            "--coverage-selector-choice-model-dir",
            "models/Qwen3-0.6B",
            "--coverage-selector-cross-encoder-model-dir",
            "models/ms-marco-MiniLM-L6-v2",
        ]
    )
    with pytest.raises(ValueError, match="separate coverage arms"):
        config_from_args(args)


def test_ms_marco_cross_encoder_is_exact_opt_in_backend():
    from memory_condense.search.selectors.cross_encoder_selector import (
        MS_MARCO_MODEL_ID,
        MS_MARCO_MODEL_REVISION,
        MS_MARCO_WEIGHTS_SHA256,
    )

    treatment = config_from_args(
        build_parser().parse_args(
            [
                "--benchmark-file",
                "sample.json",
                "--mode",
                "causal_graph",
                "--coverage-selector-cross-encoder-model-dir",
                "models/ms-marco-MiniLM-L6-v2",
                "--coverage-selector-cross-encoder-device",
                "cuda:0",
                "--coverage-selector-cross-encoder-candidate-pool",
                "192",
                "--coverage-selector-cross-encoder-batch-size",
                "16",
                "--coverage-selector-cross-encoder-max-length",
                "384",
            ]
        )
    )

    assert treatment.retrieval.coverage_selection is True
    assert treatment.retrieval.coverage_selector_backend == "cross_encoder"
    assert treatment.retrieval.coverage_selector_model == "ms-marco-MiniLM-L6-v2"
    assert treatment.retrieval.coverage_selector_cross_encoder_model_id == (
        MS_MARCO_MODEL_ID
    )
    assert treatment.retrieval.coverage_selector_cross_encoder_revision == (
        MS_MARCO_MODEL_REVISION
    )
    assert (
        treatment.retrieval.coverage_selector_cross_encoder_checkpoint_sha256
        == MS_MARCO_WEIGHTS_SHA256
    )
    assert treatment.retrieval.coverage_selector_cross_encoder_device == "cuda:0"
    assert (
        treatment.retrieval.coverage_selector_cross_encoder_candidate_pool == 192
    )
    assert treatment.retrieval.coverage_selector_cross_encoder_batch_size == 16
    assert treatment.retrieval.coverage_selector_cross_encoder_max_length == 384
    assert treatment.retrieval.label.endswith("-coverage-cross-encoder")


def test_ms_marco_can_feed_two_layer_qwen_duplicate_grouper():
    treatment = config_from_args(
        build_parser().parse_args(
            [
                "--benchmark-file",
                "sample.json",
                "--mode",
                "causal_graph",
                "--coverage-selector-cross-encoder-model-dir",
                "models/ms-marco-MiniLM-L6-v2",
                "--coverage-selector-qwen-prefix-model-dir",
                "models/Qwen3-8B",
                "--coverage-selector-prefix-layers",
                "2",
                "--coverage-selector-attention-layer",
                "1",
                "--coverage-selector-candidate-pool",
                "64",
                "--neighbor-slots",
                "24",
                "--source-slots",
                "48",
                "--consolidation-chunk-slots",
                "24",
            ]
        )
    )

    assert treatment.retrieval.coverage_selector_backend == (
        "cross_encoder_qwen_prefix"
    )
    assert treatment.retrieval.coverage_selector_model == (
        "ms-marco-MiniLM-L6-v2+Qwen3-8B"
    )
    assert treatment.retrieval.coverage_selector_prefix_layers == 2
    assert treatment.retrieval.coverage_selector_attention_layer == 1
    assert treatment.retrieval.coverage_selector_candidate_pool == 64
    assert (
        treatment.retrieval.coverage_selector_cross_encoder_candidate_pool == 128
    )
    assert treatment.retrieval.label.endswith(
        "-coverage-cross-encoder-qwen-prefix"
    )


def test_ms_marco_companion_only_mode_is_explicit_and_keeps_baseline_ranking():
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--mode",
            "causal_graph",
            "--coverage-selector-cross-encoder-model-dir",
            "models/ms-marco-MiniLM-L6-v2",
            "--no-coverage-selector-cross-encoder-semantic-rerank",
            "--coverage-selector-strict",
        ]
    )
    treatment = config_from_args(args)
    lazy = _load_coverage_selector(args, treatment)

    assert (
        treatment.retrieval.coverage_selector_cross_encoder_semantic_rerank
        is False
    )
    assert treatment.retrieval.label.endswith(
        "-coverage-cross-encoder-companion-only"
    )
    assert isinstance(lazy, _LazyCrossEncoderCoverageSelector)
    assert lazy.requires_baseline_ranking is True
    assert lazy.strict is True
    assert lazy.loaded is False


def test_ms_marco_score_only_mode_preserves_order_and_exposes_logits():
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--mode",
            "causal_graph",
            "--coverage-selector-cross-encoder-model-dir",
            "models/ms-marco-MiniLM-L6-v2",
            "--coverage-selector-qwen-prefix-model-dir",
            "models/Qwen3-8B",
            "--coverage-selector-cross-encoder-score-only",
        ]
    )
    treatment = config_from_args(args)
    lazy = _load_coverage_selector(args, treatment)

    assert (
        treatment.retrieval.coverage_selector_cross_encoder_semantic_rerank
        is False
    )
    assert treatment.retrieval.coverage_selector_cross_encoder_score_only is True
    assert treatment.retrieval.label.endswith(
        "-coverage-cross-encoder-qwen-prefix-score-only"
    )
    assert isinstance(lazy, _LazyCrossEncoderCoverageSelector)
    assert lazy.requires_baseline_ranking is True
    assert lazy.semantic_score_only is True
    assert lazy.loaded is False


def test_coverage_selector_backends_are_mutually_exclusive():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--mode",
            "causal_graph",
            "--coverage-selector-local-model-dir",
            "models/SmolLM2-360M-Instruct",
            "--coverage-selector-qwen-prefix-model-dir",
            "models/Qwen3-8B",
        ]
    )

    with pytest.raises(ValueError, match="choose either the Qwen prefix"):
        config_from_args(args)


def test_qwen_prefix_selector_stages_after_embedder_release():
    events = []

    class InnerSelector:
        last_report = "report"
        last_candidate_trace = [
            {"chunk_id": "candidate", "group_id": "event-1"}
        ]

        def select(self, query, candidates, **_kwargs):
            events.append(("select", query))
            return list(candidates)

        def close(self):
            events.append("close-prefix")

    class Condenser:
        def set_context_candidate_selector(self, selector):
            self.selector = selector

    def ingest(_sample, _config, _data_dir):
        events.append("ingest")
        return Condenser()

    ingest.release_embedder = lambda: events.append("release-bge")
    lazy = _LazyQwenPrefixCoverageSelector(
        lambda: events.append("load-prefix") or InnerSelector()
    )

    assert lazy.requires_complete_frontier_for(
        "List all museums I visited"
    ) is True
    assert lazy.requires_complete_frontier_for("Where did I go?") is False
    assert lazy.loaded is False

    condenser = _attach_coverage_selector(ingest, lazy)(None, None, None)

    assert events == ["ingest", "release-bge"]
    assert condenser.selector.select("query", ["candidate"]) == ["candidate"]
    assert events[-2:] == ["load-prefix", ("select", "query")]
    assert lazy.last_report == "report"
    assert lazy.last_candidate_trace == [
        {"chunk_id": "candidate", "group_id": "event-1"}
    ]


def test_lazy_cross_encoder_forwards_source_companion_selection():
    events = []
    expected = {"source-a": "winner"}

    class InnerSelector:
        last_source_companion_report = {"inspected_candidates": 2}

        def select_source_companions(self, query, candidates_by_source):
            events.append(("companions", query, candidates_by_source))
            return expected

        def close(self):
            events.append("close-cross")

    lazy = _LazyCrossEncoderCoverageSelector(
        lambda: events.append("load-cross") or InnerSelector()
    )
    candidates = {"source-a": ["first", "winner"]}

    assert lazy.select_source_companions("query", candidates) is expected
    assert events == ["load-cross", ("companions", "query", candidates)]
    assert lazy.last_source_companion_report == {"inspected_candidates": 2}


def test_choice_selector_loads_staged_and_injects_forward_only_provider(
    monkeypatch,
):
    import torch

    import memory_condense.search.selectors.causal_choice_scorer as choice_module
    import memory_condense.eval.local_qwen as local_qwen_module
    import memory_condense.associations.head_memory as head_memory_module
    import memory_condense.modeling.qwen_prefix as qwen_prefix_module

    events = []

    class FakeChoiceScorer:
        strict = True
        last_source_companion_report = {"inspected_candidates": 2}

        @classmethod
        def from_local_checkpoint(cls, model_dir, **kwargs):
            events.append(("load-choice", model_dir, kwargs))
            return cls()

        def select_source_companions(self, query, candidates_by_source, **kwargs):
            events.append(("choice-companions", query, kwargs))
            return {
                source_id: candidates[-1]
                for source_id, candidates in candidates_by_source.items()
            }

        def close(self):
            events.append("close-choice")

    class FakeEncoder:
        def __init__(self, model_dir, **kwargs):
            events.append(("load-prefix", model_dir, kwargs))

    class FakeLinker:
        def __init__(self, encoder, **kwargs):
            self.encoder = encoder
            events.append(("link-prefix", kwargs))

    monkeypatch.setattr(choice_module, "CausalChoiceScorer", FakeChoiceScorer)
    monkeypatch.setattr(qwen_prefix_module, "Qwen3PrefixEncoder", FakeEncoder)
    monkeypatch.setattr(head_memory_module, "QwenMemoryLinker", FakeLinker)
    monkeypatch.setattr(
        local_qwen_module,
        "resolve_local_qwen_dtype",
        lambda *_args, **_kwargs: (torch.float16, "float16"),
    )
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--mode",
            "causal_graph",
            "--coverage-selector-qwen-prefix-model-dir",
            "models/Qwen3-8B",
            "--coverage-selector-choice-model-dir",
            "models/Qwen3-0.6B",
            "--coverage-selector-choice-device",
            "cpu",
            "--coverage-selector-choice-batch-size",
            "4",
            "--coverage-selector-null-threshold",
            "0.77",
            "--coverage-selector-uncertainty-entropy",
            "0.88",
            "--coverage-selector-strict",
        ]
    )
    config = config_from_args(args)

    lazy = _load_coverage_selector(args, config)

    assert isinstance(lazy, _LazyQwenPrefixCoverageSelector)
    assert lazy.loaded is False
    assert lazy.requires_complete_frontier is True
    inner = lazy._ensure_loaded()
    assert events[0][0] == "load-choice"
    choice_kwargs = events[0][2]
    assert choice_kwargs["require_single_token_labels"] is True
    assert choice_kwargs["batch_size"] == 4
    assert choice_kwargs["strict"] is True
    assert events[1][0] == "load-prefix"
    prefix_kwargs = events[1][2]
    assert prefix_kwargs["model_id"] == "Qwen/Qwen3-8B"
    assert prefix_kwargs["model_revision"] == (
        "b968826d9c46dd6066d109eabc6255188de91218"
    )
    assert len(prefix_kwargs["expected_checkpoint_sha256"]) == 64
    assert prefix_kwargs["dtype"] == "float16"
    assert inner.score_provider.__class__ is FakeChoiceScorer
    assert inner.null_threshold == 0.77
    assert inner.uncertainty_entropy == 0.88
    lazy.close()
    assert "close-choice" in events


def test_lazy_qwen_prefix_forwards_companions_to_score_provider():
    observed = []

    class Provider:
        last_source_companion_report = {"inspected_candidates": 2}

        def select_source_companions(
            self,
            query,
            candidates_by_source,
            *,
            source_timestamps=None,
        ):
            observed.append((query, candidates_by_source, source_timestamps))
            return {"source-a": "answer"}

    class Inner:
        score_provider = Provider()

        def close(self):
            pass

    lazy = _LazyQwenPrefixCoverageSelector(lambda: Inner(), strict=True)
    candidates = {"source-a": ["first", "answer"]}
    timestamps = {"source-a": "2024-03-02"}

    selected = lazy.select_source_companions(
        "question",
        candidates,
        source_timestamps=timestamps,
    )

    assert selected == {"source-a": "answer"}
    assert observed == [("question", candidates, timestamps)]
    assert lazy.last_source_companion_report == {"inspected_candidates": 2}


def test_local_ini_selector_loads_only_after_embedder_release(monkeypatch):
    import memory_condense.search.selectors.coverage_selector as coverage_selector_module
    import memory_condense.eval.local_qwen as local_qwen_module

    events = []

    class FakeAnswerer:
        dtype_name = "float16"

        def __init__(self, model_dir, **kwargs):
            events.append(("load-local", model_dir, kwargs))

    class FakeSelector:
        last_report = "local-report"

        def __init__(self, answerer, **kwargs):
            events.append(("wrap-local", answerer, kwargs))

        def select(self, query, candidates, **_kwargs):
            events.append(("select", query))
            return list(candidates)

        def close(self):
            events.append("close-local")

    class Condenser:
        def set_context_candidate_selector(self, selector):
            self.selector = selector

    def ingest(_sample, _config, _data_dir):
        events.append("ingest")
        return Condenser()

    ingest.release_embedder = lambda: events.append("release-bge")
    monkeypatch.setattr(local_qwen_module, "LocalQwenAnswerer", FakeAnswerer)
    monkeypatch.setattr(
        coverage_selector_module,
        "QueryConditionedCoverageSelector",
        FakeSelector,
    )
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--mode",
            "causal_graph",
            "--coverage-selector-local-model-dir",
            "models/SmolLM2-360M-Instruct",
            "--coverage-selector-dtype",
            "float16",
        ]
    )
    config = config_from_args(args)
    lazy = _load_coverage_selector(args, config)

    assert type(lazy) is _LazyQwenPrefixCoverageSelector
    assert events == []
    condenser = _attach_coverage_selector(ingest, lazy)(None, config, None)
    assert events == ["ingest", "release-bge"]

    assert condenser.selector.select("query", ["candidate"]) == ["candidate"]
    assert events[2][0] == "load-local"
    assert events[2][2]["dtype"] == "float16"
    assert events[-1] == ("select", "query")
    assert lazy.last_report == "local-report"
    lazy.close()
    assert events[-1] == "close-local"


def test_cross_encoder_selector_loads_only_after_embedder_release(monkeypatch):
    import memory_condense.search.selectors.cross_encoder_selector as cross_encoder_module
    import sentence_transformers

    events = []

    class FakeCrossEncoder:
        def __init__(self, model_dir, **kwargs):
            events.append(("load-cross", model_dir, kwargs))

        def predict(self, _pairs, **_kwargs):
            raise AssertionError("empty candidate set must not run prediction")

    class Condenser:
        def set_context_candidate_selector(self, selector):
            self.selector = selector

    def ingest(_sample, _config, _data_dir):
        events.append("ingest")
        return Condenser()

    def verify(model_dir):
        events.append(("verify-cross", model_dir))
        return cross_encoder_module.MS_MARCO_WEIGHTS_SHA256

    ingest.release_embedder = lambda: events.append("release-bge")
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(
        cross_encoder_module,
        "verify_ms_marco_checkpoint",
        verify,
    )
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--mode",
            "causal_graph",
            "--coverage-selector-cross-encoder-model-dir",
            "models/ms-marco-MiniLM-L6-v2",
            "--coverage-selector-cross-encoder-batch-size",
            "16",
        ]
    )
    config = config_from_args(args)
    lazy = _load_coverage_selector(args, config)

    assert isinstance(lazy, _LazyCrossEncoderCoverageSelector)
    assert lazy.requires_baseline_ranking is False
    assert events == []
    condenser = _attach_coverage_selector(ingest, lazy)(None, config, None)
    assert events == ["ingest", "release-bge"]

    assert condenser.selector.select("query", []) == []
    assert events[2][0] == "verify-cross"
    assert events[3][0] == "load-cross"
    assert events[3][2]["local_files_only"] is True
    assert events[3][2]["trust_remote_code"] is False
    assert events[3][2]["model_kwargs"] == {"use_safetensors": True}
    assert lazy.last_report is not None
    assert lazy.last_report.retained_transformer_state_bytes == 0
    assert lazy._selector.candidate_pool == 128
    lazy.close()
    assert lazy.loaded is False


@pytest.mark.parametrize(
    ("mode", "expected_device"),
    [
        ("causal_consolidation", None),
        ("causal_graph", None),
        ("memory", "cpu"),
    ],
)
def test_local_ini_selector_keeps_bge_on_gpu_only_when_staged(
    mode,
    expected_device,
):
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--mode",
            mode,
            "--coverage-selector-local-model-dir",
            "models/SmolLM2-360M-Instruct",
        ]
    )

    _reserve_embedding_device_for_transient_models(args)

    assert args.embedding_device == expected_device


@pytest.mark.parametrize(
    ("mode", "expected_device"),
    [("causal_graph", None), ("memory", "cpu")],
)
def test_cross_encoder_keeps_bge_on_gpu_only_when_staged(
    mode,
    expected_device,
):
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--mode",
            mode,
            "--coverage-selector-cross-encoder-model-dir",
            "models/ms-marco-MiniLM-L6-v2",
        ]
    )

    _reserve_embedding_device_for_transient_models(args)

    assert args.embedding_device == expected_device


@pytest.mark.parametrize("mode", ("causal_graph", "memory"))
def test_qwen_reranker_resolves_embedding_device_identically_for_all_modes(mode):
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            "sample.json",
            "--mode",
            mode,
            "--qwen-rerank-model-dir",
            "models/Qwen3-8B",
        ]
    )

    _reserve_embedding_device_for_transient_models(args)

    assert args.embedding_device == "cpu"


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


def test_source_metadata_packing_is_explicit_and_defaults_off():
    parser = build_parser()
    baseline = config_from_args(
        parser.parse_args(["--benchmark-file", "sample.json"])
    )
    treatment = config_from_args(
        parser.parse_args(
            [
                "--benchmark-file",
                "sample.json",
                "--consolidation-source-metadata-packing",
            ]
        )
    )

    assert baseline.retrieval.consolidation_source_metadata_packing is False
    assert treatment.retrieval.consolidation_source_metadata_packing is True


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
    assert _planned_provider_calls(
        samples,
        max_samples=1,
        local_answerer=False,
        use_judge=True,
        provider_retries=2,
    ) == 12
    with pytest.raises(ValueError, match="non-negative"):
        _planned_provider_calls(
            samples,
            max_samples=1,
            local_answerer=False,
            use_judge=True,
            provider_retries=-1,
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("CORRECT", True),
        ("correct: equivalent answer", True),
        ("INCORRECT - different fact", False),
    ),
)
def test_binary_judge_parser_accepts_only_an_exact_leading_label(
    text: str, expected: bool
):
    assert _parse_binary_judge_verdict(text) is expected


@pytest.mark.parametrize(
    "text",
    ("", "CORRECTNESS", "CORRECT or INCORRECT", "CORRECT/INCORRECT", "maybe"),
)
def test_binary_judge_parser_rejects_missing_or_ambiguous_labels(text: str):
    with pytest.raises(RuntimeError, match="malformed|ambiguous"):
        _parse_binary_judge_verdict(text)


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


def test_codex_sdk_answerer_omits_temperature(monkeypatch):
    calls = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
    )

    def completion(**kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(litellm, "completion", completion)

    answer_fn = _make_answer_fn("openai/codex_sdk/gpt-5.6-luna")
    answer, _usage = answer_fn([{"role": "user", "content": "question"}])

    assert answer == "answer"
    assert "temperature" not in calls[0]


def test_central_dev_transport_is_shared_by_all_remote_wrappers(monkeypatch):
    calls = []
    client = object()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="CORRECT"))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
    )

    def completion(**kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(
        "memory_condense.eval.__main__._make_central_dev_client",
        lambda _model: client,
    )

    _make_answer_fn("openai/codex_sdk/gpt-5.6-luna")(
        [{"role": "user", "content": "question"}]
    )
    _make_judge_fn("openai/codex_sdk/gpt-5.6-sol")(
        "question", "gold", "prediction"
    )
    _make_sufficiency_fn("openai/codex_sdk/gpt-5.6-sol")(
        "question", "gold", ["evidence"]
    )

    assert [call["client"] for call in calls] == [client, client, client]


def test_codex_sdk_uses_gateway_native_litellm_key(monkeypatch):
    calls = []
    client = object()

    class FakeOpenAI:
        def __new__(cls, **kwargs):
            calls.append(kwargs)
            return client

    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LITELLM_KEY", "gateway-key")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    monkeypatch.setattr("httpx.Client", lambda **_kwargs: object())

    result = _make_central_dev_client("openai/codex_sdk/gpt-5.6-terra")

    assert result is client
    assert calls[0]["api_key"] == "gateway-key"
    assert str(calls[0]["base_url"]) == "https://central-dev.zt:4000/v1"
    assert calls[0]["max_retries"] == 0


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


def test_policy_manifest_verifies_optional_provenance_bindings(tmp_path: Path):
    repository = tmp_path / "repo"
    repository.mkdir()
    split = repository / "split.json"
    split.write_text("locked split", encoding="utf-8")
    selection = repository / "selection.csv"
    selection.write_text("selected evidence", encoding="utf-8")
    config = EvalConfig(
        retrieval=RetrievalConfig(mode="hybrid_neighbor", k=10),
        max_prompt_tokens=8000,
    )
    payload = {
        "status": "development_candidate_not_validated",
        "dataset_sha256": "a" * 64,
        "split_manifest": split.name,
        "split_manifest_sha256": file_sha256(split),
        "split": "development",
        "selection_artifact": selection.name,
        "selection_artifact_sha256": file_sha256(selection),
        "selection_artifact_required": True,
        "implementation_sha256": "b" * 64,
        "environment_lock_sha256": "c" * 64,
        "retrieval": _policy_retrieval_identity(config),
    }
    policy = repository / "policy.json"
    policy.write_text(json.dumps(payload), encoding="utf-8")

    assert len(
        _verified_policy_sha256(
            policy,
            config=config,
            dataset_sha256="a" * 64,
            split_manifest=str(split),
            active_split="development",
            active_implementation_sha256="b" * 64,
            active_environment_lock_sha256="c" * 64,
            repository_root=repository,
        )
    ) == 64

    with pytest.raises(ValueError, match="active split mismatch"):
        _verified_policy_sha256(
            policy,
            config=config,
            dataset_sha256="a" * 64,
            split_manifest=str(split),
            active_split="validation",
            active_implementation_sha256="b" * 64,
            active_environment_lock_sha256="c" * 64,
            repository_root=repository,
        )
    with pytest.raises(ValueError, match="implementation SHA-256 mismatch"):
        _verified_policy_sha256(
            policy,
            config=config,
            dataset_sha256="a" * 64,
            split_manifest=str(split),
            active_split="development",
            active_implementation_sha256="d" * 64,
            active_environment_lock_sha256="c" * 64,
            repository_root=repository,
        )
    with pytest.raises(ValueError, match="environment-lock SHA-256 mismatch"):
        _verified_policy_sha256(
            policy,
            config=config,
            dataset_sha256="a" * 64,
            split_manifest=str(split),
            active_split="development",
            active_implementation_sha256="b" * 64,
            active_environment_lock_sha256="d" * 64,
            repository_root=repository,
        )

    split.write_text("drifted split", encoding="utf-8")
    with pytest.raises(ValueError, match="locked-split SHA-256 mismatch"):
        _verified_policy_sha256(
            policy,
            config=config,
            dataset_sha256="a" * 64,
            split_manifest=str(split),
            active_split="development",
            active_implementation_sha256="b" * 64,
            active_environment_lock_sha256="c" * 64,
            repository_root=repository,
        )
    split.write_text("locked split", encoding="utf-8")

    selection.write_text("drifted evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="selection artifact SHA-256 mismatch"):
        _verified_policy_sha256(
            policy,
            config=config,
            dataset_sha256="a" * 64,
            split_manifest=str(split),
            active_split="development",
            active_implementation_sha256="b" * 64,
            active_environment_lock_sha256="c" * 64,
            repository_root=repository,
        )


def _frozen_validation_policy(tmp_path: Path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("locked dataset", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text("locked split", encoding="utf-8")
    selection = tmp_path / "selection.csv"
    selection.write_text("locked selection", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            str(dataset),
            "--benchmark-format",
            "longmemeval",
            "--benchmark-split-manifest",
            str(split),
            "--benchmark-split",
            "validation",
            "--embedding-device",
            "cuda",
            "--use-judge",
            "--max-provider-calls",
            "4",
            "--stress-context-tokens",
            "2",
            "--stress-questions",
            "2",
            "--max-samples",
            "1",
        ]
    )
    config = config_from_args(args)
    active_evaluation = _benchmark_evaluation_identity(args, config)
    frozen_evaluation = dict(active_evaluation)
    frozen_evaluation.pop("sample_offset")
    frozen_evaluation["sample_offsets"] = [0, 2]
    policy = tmp_path / "validation-policy.json"
    policy.write_text(
        json.dumps(
            {
                "format": "memory-condense-retrieval-policy-v1",
                "status": "validation_frozen",
                "dataset_sha256": file_sha256(dataset),
                "split_manifest": split.name,
                "split_manifest_sha256": file_sha256(split),
                "split": "validation",
                "selection_artifact": selection.name,
                "selection_artifact_sha256": file_sha256(selection),
                "selection_artifact_required": True,
                "implementation_sha256": "b" * 64,
                "environment_lock_sha256": "c" * 64,
                "retrieval": _policy_retrieval_identity(config),
                "evaluation": frozen_evaluation,
            }
        ),
        encoding="utf-8",
    )
    return policy, split, config, active_evaluation


def test_frozen_validation_policy_binds_full_evaluation_and_allowed_offsets(
    tmp_path: Path,
):
    policy, split, config, active = _frozen_validation_policy(tmp_path)
    for offset in (0, 2):
        identity = {**active, "sample_offset": offset}
        assert len(
            _verified_policy_sha256(
                policy,
                config=config,
                dataset_sha256=file_sha256(tmp_path / "dataset.json"),
                split_manifest=str(split),
                active_split="validation",
                active_implementation_sha256="b" * 64,
                active_environment_lock_sha256="c" * 64,
                repository_root=tmp_path,
                evaluation_identity=identity,
            )
        ) == 64

    changed = {**active, "judge_model": "different/judge"}
    with pytest.raises(ValueError, match="evaluation config mismatch"):
        _verified_policy_sha256(
            policy,
            config=config,
            dataset_sha256=file_sha256(tmp_path / "dataset.json"),
            split_manifest=str(split),
            active_split="validation",
            active_implementation_sha256="b" * 64,
            active_environment_lock_sha256="c" * 64,
            repository_root=tmp_path,
            evaluation_identity=changed,
        )

    changed = {**active, "recent_window": active["recent_window"] + 1}
    with pytest.raises(ValueError, match="evaluation config mismatch"):
        _verified_policy_sha256(
            policy,
            config=config,
            dataset_sha256=file_sha256(tmp_path / "dataset.json"),
            split_manifest=str(split),
            active_split="validation",
            active_implementation_sha256="b" * 64,
            active_environment_lock_sha256="c" * 64,
            repository_root=tmp_path,
            evaluation_identity=changed,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("embedding_device", "cpu", "cache-preparation config mismatch"),
        ("benchmark_format", "auto", "cache-preparation config mismatch"),
        ("stress_context_tokens", 3, "cache-preparation config mismatch"),
        ("stress_questions", 1, "cache-preparation config mismatch"),
        ("stress_question_offset", 1, "cache-preparation config mismatch"),
        ("max_samples", 2, "cache-preparation config mismatch"),
        ("sample_offset", 1, "sample_offset is not in the policy"),
    ),
)
def test_validation_prepare_binds_only_authorized_cache_shape(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
):
    policy, split, config, active = _frozen_validation_policy(tmp_path)
    identity = dict(active)
    identity[field] = value
    # Provider/answer controls are intentionally irrelevant during the blind
    # build; only fields capable of changing cache bytes are compared.
    identity.update(
        responder_model="unused/responder",
        judge_model="unused/judge",
        use_judge=False,
        provider_retries=99,
        max_provider_calls=0,
        accuracy_target=0.0,
        min_target_questions=1,
        max_prompt_tokens=1,
    )

    with pytest.raises(ValueError, match=message):
        _verified_policy_sha256(
            policy,
            config=config,
            dataset_sha256=file_sha256(tmp_path / "dataset.json"),
            split_manifest=str(split),
            active_split="validation",
            active_implementation_sha256="b" * 64,
            active_environment_lock_sha256="c" * 64,
            repository_root=tmp_path,
            evaluation_identity=identity,
            prepare_only=True,
        )


def test_validation_prepare_accepts_authorized_offsets_and_ignores_qa_controls(
    tmp_path: Path,
):
    policy, split, config, active = _frozen_validation_policy(tmp_path)
    for offset in (0, 2):
        identity = {
            **active,
            "sample_offset": offset,
            "responder_model": "unused/responder",
            "judge_model": "unused/judge",
            "use_judge": False,
            "provider_retries": 99,
            "max_provider_calls": 0,
            "accuracy_target": 0.0,
            "min_target_questions": 1,
            "max_prompt_tokens": 1,
        }
        assert len(
            _verified_policy_sha256(
                policy,
                config=config,
                dataset_sha256=file_sha256(tmp_path / "dataset.json"),
                split_manifest=str(split),
                active_split="validation",
                active_implementation_sha256="b" * 64,
                active_environment_lock_sha256="c" * 64,
                repository_root=tmp_path,
                evaluation_identity=identity,
                prepare_only=True,
            )
        ) == 64


def test_policy_manifest_rejects_unpaired_or_unsafe_selection_artifact(
    tmp_path: Path,
):
    repository = tmp_path / "repo"
    repository.mkdir()
    split = repository / "split.json"
    split.write_text("locked split", encoding="utf-8")
    outside = tmp_path / "outside.csv"
    outside.write_text("outside", encoding="utf-8")
    config = EvalConfig(
        retrieval=RetrievalConfig(mode="hybrid_neighbor", k=10),
        max_prompt_tokens=8000,
    )
    payload = {
        "status": "development_candidate_not_validated",
        "dataset_sha256": "a" * 64,
        "split_manifest": split.name,
        "retrieval": _policy_retrieval_identity(config),
        "selection_artifact": "selection.csv",
        "selection_artifact_sha256": "d" * 64,
    }
    policy = repository / "policy.json"
    policy.write_text(json.dumps(payload), encoding="utf-8")

    assert len(
        _verified_policy_sha256(
            policy,
            config=config,
            dataset_sha256="a" * 64,
            split_manifest=str(split),
            repository_root=repository,
        )
    ) == 64

    payload["selection_artifact_required"] = True
    payload.pop("selection_artifact_sha256")
    policy.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be provided together"):
        _verified_policy_sha256(
            policy,
            config=config,
            dataset_sha256="a" * 64,
            split_manifest=str(split),
            repository_root=repository,
        )

    payload["selection_artifact"] = "../outside.csv"
    payload["selection_artifact_sha256"] = file_sha256(outside)
    policy.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="safe repository-relative path"):
        _verified_policy_sha256(
            policy,
            config=config,
            dataset_sha256="a" * 64,
            split_manifest=str(split),
            repository_root=repository,
        )


def test_completed_benchmark_rejects_implementation_drift(monkeypatch):
    monkeypatch.setattr(
        "memory_condense.eval.__main__.implementation_sha256",
        lambda: "a" * 64,
    )
    _assert_implementation_unchanged("a" * 64)
    with pytest.raises(RuntimeError, match="changed during benchmark run"):
        _assert_implementation_unchanged("b" * 64)


def test_benchmark_report_reuses_start_provenance_and_rechecks_code(
    monkeypatch,
    tmp_path: Path,
):
    dataset = tmp_path / "benchmark.json"
    dataset.write_text("[]", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            str(dataset),
            "--max-samples",
            "1",
            "--max-provider-calls",
            "1",
        ]
    )
    implementation_calls = []
    environment_calls = []
    captured = {}

    def implementation_fingerprint():
        implementation_calls.append(True)
        return "b" * 64

    def environment_fingerprint():
        environment_calls.append(True)
        return "c" * 64

    def fake_run_benchmark(*_args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "memory_condense.eval.__main__.load_benchmark",
        lambda *_args: [_sample(1)],
    )
    monkeypatch.setattr(
        "memory_condense.eval.__main__._make_answer_fn",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "memory_condense.eval.__main__.implementation_sha256",
        implementation_fingerprint,
    )
    monkeypatch.setattr(
        "memory_condense.eval.__main__.environment_lock_sha256",
        environment_fingerprint,
    )
    monkeypatch.setattr(
        "memory_condense.eval.__main__.run_benchmark",
        fake_run_benchmark,
    )
    monkeypatch.setattr(
        "memory_condense.eval.__main__.print_benchmark_summary",
        lambda _result: None,
    )
    monkeypatch.setattr(
        "memory_condense.eval.__main__.save_benchmark_report",
        lambda _result, _directory: tmp_path / "report.json",
    )

    run_benchmark_mode(args)

    assert captured["implementation_sha256"] == "b" * 64
    assert captured["environment_lock_sha256"] == "c" * 64
    assert len(implementation_calls) == 2
    assert len(environment_calls) == 1


@pytest.mark.parametrize(
    "forbidden",
    (
        ["--responder-model", DEFAULT_RESPONDER_MODEL],
        ["--judge-model", DEFAULT_JUDGE_MODEL],
        ["--use-judge"],
        ["--local-qwen-model-dir", "models/Qwen3-8B"],
        ["--max-provider-calls", "1"],
        ["--provider-retries", "1"],
    ),
)
def test_prepare_cache_rejects_all_responder_and_judge_controls(
    forbidden,
    tmp_path: Path,
):
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            str(tmp_path / "dataset.json"),
            "--prepare-cache-only",
            "--mode",
            "causal_graph",
            "--compiled-store-cache",
            str(tmp_path / "compiled"),
            "--causal-store-cache",
            str(tmp_path / "causal"),
            "--policy-manifest",
            str(tmp_path / "policy.json"),
            *forbidden,
        ]
    )

    with pytest.raises(ValueError, match="rejects responder, judge"):
        _validate_prepare_cache_args(args, config_from_args(args))


def test_prepare_cache_requires_exact_builder_receipts():
    with pytest.raises(RuntimeError, match="exactly one compiled and one causal"):
        _validated_blind_cache_receipts(SimpleNamespace())

    with pytest.raises(RuntimeError, match="exactly one compiled and one causal"):
        _validated_blind_cache_receipts(
            SimpleNamespace(
                blind_cache_receipts={"compiled": [], "causal": []}
            )
        )


def test_prepare_cache_is_blind_and_emits_only_hashes_counts_and_timings(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("dataset bytes", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text("split bytes", encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text("policy bytes", encoding="utf-8")
    compiled = tmp_path / "compiled"
    causal = tmp_path / "causal"
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            str(dataset),
            "--prepare-cache-only",
            "--benchmark-split-manifest",
            str(split),
            "--benchmark-split",
            "validation",
            "--mode",
            "causal_graph",
            "--compiled-store-cache",
            str(compiled),
            "--causal-store-cache",
            str(causal),
            "--policy-manifest",
            str(policy),
            "--coverage-selector-local-model-dir",
            "models/selector-must-not-load",
            "--max-samples",
            "1",
        ]
    )
    sample = BenchmarkSample(
        sample_id="secret-sample-id",
        turns=[("user", "secret history text")],
        turn_source_ids=["secret-source-id"],
        questions=[
            BenchmarkQuestion(
                question_id="secret-question-id",
                question="secret held-out question",
                answer="secret gold answer",
                evidence=["secret gold evidence"],
            )
        ],
    )
    events: list[object] = []

    class FakeStore:
        causal_consolidation_stats = {
            "staging": {"source_turns": 1, "events": 2, "elapsed_s": 0.25},
            "learning": {"events_offered": 2, "events_applied": 2, "elapsed_s": 0.5},
        }

        def __init__(self, path: Path) -> None:
            self.database_path = path / "memory.db"
            path.mkdir(parents=True)
            self.database_path.write_bytes(b"database")
            index_path = path / "hnsw_index.bin"
            index_path.write_bytes(b"index")
            self.blind_cache_receipts = {
                "compiled": [
                    {
                        "manifest_sha256": "1" * 64,
                        "cache_key": "2" * 64,
                        "sample_sha256": sample_sha256(sample),
                        "database_sha256": "3" * 64,
                        "index_sha256": "4" * 64,
                        "embedding_execution_sha256": "8" * 64,
                        "implementation_sha256": "e" * 64,
                        "environment_lock_sha256": "f" * 64,
                        "turn_count": 1,
                        "chunk_count": 1,
                    }
                ],
                "causal": [
                    {
                        "manifest_sha256": "5" * 64,
                        "cache_key": "6" * 64,
                        "sample_sha256": sample_sha256(sample),
                        "compiled_cache_key": "2" * 64,
                        "database_sha256": file_sha256(self.database_path),
                        "index_sha256": file_sha256(index_path),
                        "build_protocol_sha256": "7" * 64,
                        "embedding_execution_sha256": "8" * 64,
                        "implementation_sha256": "e" * 64,
                        "environment_lock_sha256": "f" * 64,
                    }
                ],
            }
            self.closed = False

        def close(self) -> None:
            events.append("close")
            self.closed = True

    fake_store = FakeStore(tmp_path / "artifact")

    def fake_verify(*_args, **_kwargs):
        events.append("verify")
        return "d" * 64

    def fake_load(*_args, **_kwargs):
        events.append("load")
        return [sample]

    def fake_ingest_factory(_args, _config, *, prepare_only=False):
        events.append(("factory", prepare_only))

        def ingest(received, _active_config, _scratch):
            events.append("ingest")
            assert received is sample
            digest = sample_sha256(received)
            for root, name in (
                (compiled, "compiled-store.json"),
                (causal, "causal-store.json"),
            ):
                entry = root / "opaque"
                entry.mkdir(parents=True, exist_ok=True)
                (entry / name).write_text(
                    json.dumps(
                        {
                            "sample_id": "secret-sample-id",
                            "sample_sha256": digest,
                            "cache_key": "9" * 64,
                            "database_sha256": "b" * 64,
                            "index_sha256": "c" * 64,
                            "turn_count": 1,
                            "chunk_count": 1,
                        }
                    ),
                    encoding="utf-8",
                )
            return fake_store

        ingest.release_embedder = lambda: events.append("release")
        return ingest

    def forbidden(*_args, **_kwargs):
        raise AssertionError("evaluation/provider/selector path must stay dark")

    monkeypatch.setattr(
        "memory_condense.eval.__main__._verified_policy_sha256", fake_verify
    )
    monkeypatch.setattr(
        "memory_condense.eval.__main__.implementation_sha256", lambda: "e" * 64
    )
    monkeypatch.setattr(
        "memory_condense.eval.__main__.environment_lock_sha256", lambda: "f" * 64
    )
    monkeypatch.setattr("memory_condense.eval.__main__.load_benchmark", fake_load)
    monkeypatch.setattr(
        "memory_condense.eval.__main__._apply_locked_split",
        lambda _args, samples, *, verbose: samples,
    )
    monkeypatch.setattr(
        "memory_condense.eval.__main__._benchmark_ingest_fn",
        fake_ingest_factory,
    )
    for name in (
        "run_benchmark",
        "run_recall",
        "_make_answer_fn",
        "_make_judge_fn",
        "_load_candidate_reranker",
        "_load_coverage_selector",
    ):
        monkeypatch.setattr(f"memory_condense.eval.__main__.{name}", forbidden)

    report = run_prepare_cache_only(args)
    output = capsys.readouterr().out

    assert events == [
        "verify",
        "load",
        ("factory", True),
        "ingest",
        "close",
        "release",
    ]
    assert fake_store.closed is True
    assert report["sample_count"] == 1
    assert report["turn_count"] == 1
    assert report["source_count"] == 1
    assert report["samples"][0]["staging_events"] == 2
    assert report["samples"][0]["learning_events_applied"] == 2
    assert len(report["samples"][0]["compiled_cache_entries"]) == 1
    assert len(report["samples"][0]["causal_cache_entries"]) == 1
    assert report["samples"][0]["compiled_cache_entries"][0]["cache_key"] == (
        "2" * 64
    )
    assert report["samples"][0]["causal_cache_entries"][0]["cache_key"] == (
        "6" * 64
    )
    for secret in (
        "secret-sample-id",
        "secret-source-id",
        "secret history text",
        "secret held-out question",
        "secret gold answer",
        "secret gold evidence",
        str(dataset),
        str(compiled),
        str(causal),
    ):
        assert secret not in output


def test_prepare_cache_policy_hash_drift_blocks_before_parse_or_ingest(
    monkeypatch,
    tmp_path: Path,
):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("locked dataset", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text("locked split", encoding="utf-8")
    policy = tmp_path / "policy.json"
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            str(dataset),
            "--prepare-cache-only",
            "--benchmark-split-manifest",
            str(split),
            "--benchmark-split",
            "validation",
            "--mode",
            "causal_graph",
            "--compiled-store-cache",
            str(tmp_path / "compiled"),
            "--causal-store-cache",
            str(tmp_path / "causal"),
            "--policy-manifest",
            str(policy),
        ]
    )
    config = config_from_args(args)
    policy.write_text(
        json.dumps(
            {
                "status": "active",
                "dataset_sha256": "0" * 64,
                "split_manifest": split.name,
                "split": "validation",
                "retrieval": _policy_retrieval_identity(config),
            }
        ),
        encoding="utf-8",
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("drift must fail before dataset parse or ingest")

    monkeypatch.setattr("memory_condense.eval.__main__.load_benchmark", forbidden)
    monkeypatch.setattr(
        "memory_condense.eval.__main__._benchmark_ingest_fn", forbidden
    )

    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        run_prepare_cache_only(args)
    assert not (tmp_path / "compiled").exists()
    assert not (tmp_path / "causal").exists()


@pytest.mark.parametrize(
    ("policy_filename", "allow_selected_scope_closure"),
    (
        (
            "longmemeval-qwen-choice-coverage-operational-development-v1.json",
            False,
        ),
        (
            "longmemeval-qwen-choice-coverage-operational-development-v2.json",
            True,
        ),
    ),
)
def test_selected_operational_policy_matches_cli_and_rejects_coverage_drift(
    policy_filename: str,
    allow_selected_scope_closure: bool,
):
    root = Path(__file__).parents[1]
    policy_path = (
        root
        / "docs"
        / "10 - Research Log"
        / "data"
        / policy_filename
    )
    split_path = (
        root
        / "docs"
        / "10 - Research Log"
        / "data"
        / "longmemeval-95-target-split-v2.json"
    )
    selected_args = [
        "--mode", "causal_graph",
        "--k", "10",
        "--ef-search", "50",
        "--source-slots", "48",
        "--source-candidate-pool", "750",
        "--source-activation-k", "65",
        "--role-aware-retrieval",
        "--source-tfisf-activation",
        "--source-tfisf-slots", "8",
        "--source-hsc-activation",
        "--source-hsc-slots", "8",
        "--source-hsc-hops", "2",
        "--source-hsc-chunk-slots", "4",
        "--source-local-search",
        "--source-partition-routing",
        "--source-partition-slots", "4",
        "--neighbor-radius", "5",
        "--neighbor-slots", "24",
        "--neighbor-direction", "next",
        "--consolidation-chunk-slots", "24",
        "--consolidation-hops", "2",
        "--consolidation-candidates", "128",
        "--consolidation-expansion-tokens", "2250",
        "--consolidation-query-aware-sentence-packing",
        "--consolidation-max-sentences-per-expansion", "2",
        "--consolidation-information-gain-packing",
        "--consolidation-min-information-gain-per-token", "0.008",
        "--consolidation-source-metadata-packing",
        "--coverage-selector-qwen-prefix-model-dir", ".cache/models/Qwen3-8B",
        "--coverage-selector-choice-model-dir", ".cache/models/Qwen3-0.6B",
        "--coverage-selector-prefix-layers", "2",
        "--coverage-selector-attention-layer", "1",
        "--coverage-selector-prefix-device", "cuda",
        "--coverage-selector-dtype", "float16",
        "--coverage-selector-candidate-pool", "64",
        "--coverage-selector-candidate-tokens", "64",
        "--coverage-selector-query-tokens", "96",
        "--coverage-selector-max-workspace-tokens", "8192",
        "--coverage-selector-choice-device", "cuda",
        "--coverage-selector-choice-dtype", "float16",
        "--coverage-selector-choice-batch-size", "8",
        "--coverage-selector-choice-max-candidates", "128",
        "--coverage-selector-choice-query-tokens", "192",
        "--coverage-selector-choice-candidate-tokens", "128",
        "--coverage-selector-choice-max-prompt-tokens", "512",
        "--coverage-selector-choice-max-workspace-tokens", "8192",
        "--coverage-selector-strict",
    ]
    if allow_selected_scope_closure:
        selected_args.append("--allow-selected-scope-fixed-k-closure")
    config = config_from_args(build_parser().parse_args(selected_args))
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))

    assert payload["status"] == "development_candidate_not_validated"
    assert payload["split"] == "development"
    assert payload["split_manifest"] == split_path.name
    assert payload["split_manifest_sha256"] == file_sha256(split_path)
    assert payload["dataset_sha256"] == split_payload["dataset_sha256"]
    historical_retrieval = payload["retrieval"]
    current_retrieval = _policy_retrieval_identity(config)
    checkpoint_field = "coverage_selector_choice_checkpoint_sha256"
    assert historical_retrieval[checkpoint_field] == (
        "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
    )
    assert current_retrieval[checkpoint_field] == (
        "a940db06d5d9a3b298412376966b492f09ad7f088495fb75c05aa45db943d86e"
    )
    assert {
        key: value
        for key, value in historical_retrieval.items()
        if key != checkpoint_field
    } == {
        key: value
        for key, value in current_retrieval.items()
        if key != checkpoint_field
    }
    # These v1/v2 bytes are immutable development artifacts. The earlier
    # weights-only digest is intentionally rejected by the current full
    # config/tokenizer/index/weights manifest identity.
    with pytest.raises(ValueError, match="retrieval config mismatch"):
        _verified_policy_sha256(
            policy_path,
            config=config,
            dataset_sha256=payload["dataset_sha256"],
            split_manifest=str(split_path),
            active_split=payload["split"],
        )

    changed_config = config_from_args(
        build_parser().parse_args(
            selected_args + ["--coverage-selector-candidate-pool", "63"]
        )
    )
    with pytest.raises(ValueError, match="retrieval config mismatch"):
        _verified_policy_sha256(
            policy_path,
            config=changed_config,
            dataset_sha256=payload["dataset_sha256"],
            split_manifest=str(split_path),
            active_split=payload["split"],
        )


def test_v3_selected_policy_and_compact_artifact_match_current_cli():
    root = Path(__file__).parents[1]
    data_dir = root / "docs" / "10 - Research Log" / "data"
    policy_path = data_dir / (
        "longmemeval-qwen-choice-coverage-operational-development-v3.json"
    )
    selection_path = data_dir / (
        "longmemeval-qwen-choice-coverage-selection-development-v3.json"
    )
    split_path = data_dir / "longmemeval-95-target-split-v2.json"
    selected_args = [
        "--mode", "causal_graph",
        "--k", "10",
        "--ef-search", "50",
        "--source-slots", "48",
        "--source-candidate-pool", "750",
        "--source-activation-k", "65",
        "--role-aware-retrieval",
        "--source-tfisf-activation",
        "--source-tfisf-slots", "8",
        "--source-hsc-activation",
        "--source-hsc-slots", "8",
        "--source-hsc-hops", "2",
        "--source-hsc-chunk-slots", "4",
        "--source-local-search",
        "--source-partition-routing",
        "--source-partition-slots", "4",
        "--neighbor-radius", "5",
        "--neighbor-slots", "24",
        "--neighbor-direction", "next",
        "--consolidation-chunk-slots", "24",
        "--consolidation-hops", "2",
        "--consolidation-candidates", "128",
        "--consolidation-expansion-tokens", "2250",
        "--consolidation-query-aware-sentence-packing",
        "--consolidation-max-sentences-per-expansion", "2",
        "--consolidation-information-gain-packing",
        "--consolidation-min-information-gain-per-token", "0.008",
        "--consolidation-source-metadata-packing",
        "--coverage-selector-qwen-prefix-model-dir", ".cache/models/Qwen3-8B",
        "--coverage-selector-choice-model-dir", ".cache/models/Qwen3-0.6B",
        "--coverage-selector-prefix-layers", "2",
        "--coverage-selector-attention-layer", "1",
        "--coverage-selector-prefix-device", "cuda",
        "--coverage-selector-dtype", "float16",
        "--coverage-selector-candidate-pool", "64",
        "--coverage-selector-candidate-tokens", "64",
        "--coverage-selector-query-tokens", "96",
        "--coverage-selector-max-workspace-tokens", "8192",
        "--coverage-selector-choice-device", "cuda",
        "--coverage-selector-choice-dtype", "float16",
        "--coverage-selector-choice-batch-size", "8",
        "--coverage-selector-choice-max-candidates", "128",
        "--coverage-selector-choice-query-tokens", "192",
        "--coverage-selector-choice-candidate-tokens", "128",
        "--coverage-selector-choice-max-prompt-tokens", "512",
        "--coverage-selector-choice-max-workspace-tokens", "8192",
        "--coverage-selector-strict",
        "--allow-selected-scope-fixed-k-closure",
    ]
    config = config_from_args(build_parser().parse_args(selected_args))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    assert policy["retrieval"] == _policy_retrieval_identity(config)
    assert policy["selection_artifact_sha256"] == file_sha256(selection_path)
    assert selection["source_artifact"]["sha256"] == (
        "df12c9d5cfebe591d7780808046acc601a61b471a207df7733f08dfc73c907f9"
    )
    assert selection["metrics"]["packed_evidence_source_coverage"] == 1.0
    assert selection["metrics"]["answer_value_components_found"] == 11
    assert selection["claims"]["held_out_validation_run"] is False
    assert _verified_policy_sha256(
        policy_path,
        config=config,
        dataset_sha256=policy["dataset_sha256"],
        split_manifest=str(split_path),
        active_split="development",
        active_implementation_sha256=policy["implementation_sha256"],
        active_environment_lock_sha256=policy["environment_lock_sha256"],
        repository_root=root,
    ) == file_sha256(policy_path)


def test_validation_ingest_is_cache_hit_only(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_factory(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "memory_condense.eval.causal_benchmark.causal_consolidation_ingest_fn",
        fake_factory,
    )
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            "dataset.json",
            "--benchmark-split-manifest",
            "split.json",
            "--benchmark-split",
            "validation",
            "--mode",
            "causal_graph",
            "--compiled-store-cache",
            str(tmp_path / "compiled"),
            "--causal-store-cache",
            str(tmp_path / "causal"),
        ]
    )
    config = config_from_args(args)

    _benchmark_ingest_fn(args, config)

    assert captured["require_cache_hit"] is True
    assert captured["prepare_only"] is False


def test_validation_rejects_noncausal_cache_path(tmp_path: Path):
    args = build_parser().parse_args(
        [
            "--benchmark-file",
            "dataset.json",
            "--benchmark-split-manifest",
            "split.json",
            "--benchmark-split",
            "validation",
            "--mode",
            "hybrid",
            "--compiled-store-cache",
            str(tmp_path / "compiled"),
        ]
    )

    with pytest.raises(ValueError, match=r"compiled\+learned cache receipts"):
        _benchmark_ingest_fn(args, config_from_args(args))
