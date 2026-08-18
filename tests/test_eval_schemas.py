from memory_condense.eval.schemas import (
    ChunkerConfig,
    ConversationResult,
    EvalConfig,
    EvalRunResult,
    RetrievalConfig,
    SweepReport,
    TurnResult,
    UsageStats,
)
import pytest


def test_chunker_config_defaults():
    c = ChunkerConfig()
    assert c.min_tokens == 120
    assert c.max_tokens == 250


def test_retrieval_config_defaults():
    r = RetrievalConfig()
    assert r.k == 10
    assert r.ef_search == 50
    assert r.allow_selected_scope_fixed_k_closure is False


def test_selected_scope_closure_requires_partitioned_coverage():
    with pytest.raises(ValueError, match="coverage_selection"):
        RetrievalConfig(allow_selected_scope_fixed_k_closure=True)
    with pytest.raises(ValueError, match="source_partition_routing"):
        RetrievalConfig(
            mode="causal_graph",
            coverage_selection=True,
            coverage_selector_backend="qwen_prefix",
            allow_selected_scope_fixed_k_closure=True,
        )

    configured = RetrievalConfig(
        mode="causal_graph",
        coverage_selection=True,
        coverage_selector_backend="qwen_prefix",
        source_partition_routing=True,
        allow_selected_scope_fixed_k_closure=True,
    )
    assert configured.allow_selected_scope_fixed_k_closure is True

    with pytest.raises(ValueError, match="Qwen prefix"):
        RetrievalConfig(
            mode="causal_graph",
            coverage_selection=True,
            coverage_selector_backend="local_ini",
            source_partition_routing=True,
            allow_selected_scope_fixed_k_closure=True,
        )


def test_hybrid_neighbor_label_captures_transition_budget():
    config = RetrievalConfig(
        mode="hybrid_neighbor",
        k=10,
        neighbor_radius=6,
        neighbor_slots=22,
        neighbor_replacement_slots=0,
    )

    assert config.label == "hybrid-neighbor-k10-r6-s22"


def test_hybrid_source_label_captures_second_stage_bounds():
    config = RetrievalConfig(
        mode="hybrid_source", k=10, source_slots=24, source_candidate_pool=200
    )

    assert config.label == "hybrid-source-k10-s24-a10-p200"


def test_hybrid_graph_label_captures_both_link_budgets():
    config = RetrievalConfig(
        mode="hybrid_graph",
        k=10,
        neighbor_radius=5,
        neighbor_slots=24,
        neighbor_direction="next",
        source_slots=24,
        source_candidate_pool=200,
        source_activation_k=20,
    )

    assert config.label == "hybrid-graph-k10-r5-n24-next-s24-a20-p200"


def test_source_tfisf_defaults_off_and_has_a_positive_bound():
    config = RetrievalConfig()

    assert config.source_tfisf_activation is False
    assert config.source_tfisf_slots == 8
    with pytest.raises(ValueError):
        RetrievalConfig(source_tfisf_slots=0)


def test_source_hsc_requires_graph_mode_and_fits_source_budget():
    with pytest.raises(ValueError, match="requires a graph mode"):
        RetrievalConfig(mode="dense", source_hsc_activation=True)
    with pytest.raises(ValueError, match="cannot exceed source_slots"):
        RetrievalConfig(
            mode="causal_graph",
            source_slots=3,
            source_hsc_activation=True,
            source_hsc_chunk_slots=4,
        )


def test_query_facet_retrieval_requires_graph_mode_and_fits_reserves():
    with pytest.raises(ValueError, match="requires a graph mode"):
        RetrievalConfig(mode="dense", query_facet_retrieval=True)

    with pytest.raises(ValueError, match="cannot exceed source_slots"):
        RetrievalConfig(
            mode="causal_graph",
            source_slots=3,
            query_facet_retrieval=True,
            query_facet_slots=4,
        )

    with pytest.raises(ValueError, match="facet and HSC reserves"):
        RetrievalConfig(
            mode="causal_graph",
            source_slots=8,
            query_facet_retrieval=True,
            query_facet_slots=5,
            source_hsc_activation=True,
            source_hsc_chunk_slots=4,
        )


def test_role_aware_retrieval_requires_graph_mode():
    with pytest.raises(ValueError, match="requires a graph mode"):
        RetrievalConfig(mode="dense", role_aware_retrieval=True)

    config = RetrievalConfig(mode="causal_graph", role_aware_retrieval=True)
    assert config.label.endswith("-role")


def test_multi_fact_source_diversity_requires_graph_mode():
    with pytest.raises(ValueError, match="requires a graph mode"):
        RetrievalConfig(mode="dense", multi_fact_source_diversity=True)

    config = RetrievalConfig(
        mode="causal_graph", multi_fact_source_diversity=True
    )
    assert config.label.endswith("-diverse")


def test_coverage_selection_requires_packed_mode_and_has_distinct_label():
    with pytest.raises(ValueError, match="requires a packed memory or causal mode"):
        RetrievalConfig(mode="hybrid_graph", coverage_selection=True)

    config = RetrievalConfig(mode="causal_graph", coverage_selection=True)

    assert config.label.endswith("-coverage-local-ini")
    assert config.coverage_selector_candidate_pool == 64
    assert config.coverage_selector_max_workspace_tokens == 8192
    assert config.coverage_selector_dtype == "auto"


def test_qwen_prefix_coverage_selector_bounds_are_validated():
    config = RetrievalConfig(
        mode="causal_graph",
        coverage_selection=True,
        coverage_selector_backend="qwen_prefix",
    )

    assert config.label.endswith("-coverage-qwen-prefix")
    assert config.coverage_selector_prefix_layers == 6
    assert config.coverage_selector_attention_layer == 5

    with pytest.raises(ValueError, match="attention layer must be inside"):
        RetrievalConfig(
            mode="causal_graph",
            coverage_selection=True,
            coverage_selector_backend="qwen_prefix",
            coverage_selector_prefix_layers=6,
            coverage_selector_attention_layer=6,
        )

    with pytest.raises(ValueError, match="attention layer must be at least 1"):
        RetrievalConfig(
            mode="causal_graph",
            coverage_selection=True,
            coverage_selector_backend="qwen_prefix",
            coverage_selector_attention_layer=0,
        )

    with pytest.raises(ValueError, match="same-source merge threshold"):
        RetrievalConfig(
            mode="causal_graph",
            coverage_selection=True,
            coverage_selector_backend="qwen_prefix",
            coverage_selector_same_source_merge_similarity=0.99,
            coverage_selector_merge_similarity=0.98,
        )

    with pytest.raises(ValueError, match="prefix checkpoint SHA-256"):
        RetrievalConfig(
            mode="causal_graph",
            coverage_selection=True,
            coverage_selector_backend="qwen_prefix",
            coverage_selector_prefix_checkpoint_sha256="not-a-digest",
        )


def test_cross_encoder_backend_is_bounded_and_composes_with_qwen_prefix():
    semantic = RetrievalConfig(
        mode="causal_graph",
        coverage_selection=True,
        coverage_selector_backend="cross_encoder",
        coverage_selector_cross_encoder_model_id=(
            "cross-encoder/ms-marco-MiniLM-L6-v2"
        ),
    )
    composite = semantic.model_copy(
        update={
            "coverage_selector_backend": "cross_encoder_qwen_prefix",
            "coverage_selector_prefix_layers": 2,
            "coverage_selector_attention_layer": 1,
        }
    )

    assert semantic.label.endswith("-coverage-cross-encoder")
    assert composite.label.endswith("-coverage-cross-encoder-qwen-prefix")
    assert semantic.coverage_selector_cross_encoder_batch_size == 32
    assert semantic.coverage_selector_cross_encoder_max_length == 256
    assert semantic.coverage_selector_cross_encoder_candidate_pool == 128
    assert semantic.model_copy(
        update={"coverage_selector_cross_encoder_semantic_rerank": False}
    ).label.endswith("-coverage-cross-encoder-companion-only")
    assert semantic.model_copy(
        update={
            "coverage_selector_cross_encoder_semantic_rerank": False,
            "coverage_selector_cross_encoder_score_only": True,
        }
    ).label.endswith("-coverage-cross-encoder-score-only")

    with pytest.raises(ValueError, match="max length cannot exceed"):
        RetrievalConfig(
            mode="causal_graph",
            coverage_selection=True,
            coverage_selector_backend="cross_encoder",
            coverage_selector_max_workspace_tokens=128,
            coverage_selector_cross_encoder_max_length=256,
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        RetrievalConfig(
            mode="causal_graph",
            coverage_selection=True,
            coverage_selector_backend="cross_encoder",
            coverage_selector_cross_encoder_semantic_rerank=True,
            coverage_selector_cross_encoder_score_only=True,
        )

    with pytest.raises(ValueError, match="attention layer must be inside"):
        RetrievalConfig(
            mode="causal_graph",
            coverage_selection=True,
            coverage_selector_backend="cross_encoder_qwen_prefix",
            coverage_selector_prefix_layers=2,
            coverage_selector_attention_layer=2,
        )


def test_qwen_choice_backend_requires_exact_identity_and_bounded_workspace():
    identity = {
        "coverage_selector_choice_model_id": "Qwen/Qwen3-0.6B",
        "coverage_selector_choice_revision": "exact-revision",
        "coverage_selector_choice_checkpoint_sha256": "a" * 64,
    }
    config = RetrievalConfig(
        mode="causal_graph",
        coverage_selection=True,
        coverage_selector_backend="qwen_prefix_choice",
        **identity,
    )

    assert config.label.endswith("-coverage-qwen-prefix-choice-qwen3-0-6b")
    assert config.coverage_selector_choice_batch_size == 8
    assert config.coverage_selector_choice_max_candidates == 128

    with pytest.raises(ValueError, match="exact model identity"):
        RetrievalConfig(
            mode="causal_graph",
            coverage_selection=True,
            coverage_selector_backend="qwen_prefix_choice",
        )
    with pytest.raises(ValueError, match="64 hex digits"):
        RetrievalConfig(
            mode="causal_graph",
            coverage_selection=True,
            coverage_selector_backend="qwen_prefix_choice",
            **{
                **identity,
                "coverage_selector_choice_checkpoint_sha256": "not-a-digest",
            },
        )
    with pytest.raises(ValueError, match="cannot hold one candidate prompt"):
        RetrievalConfig(
            mode="causal_graph",
            coverage_selection=True,
            coverage_selector_backend="qwen_prefix_choice",
            coverage_selector_choice_max_prompt_tokens=512,
            coverage_selector_choice_max_workspace_tokens=511,
            **identity,
        )


def test_coverage_selector_uncertainty_entropy_is_a_probability_bound():
    with pytest.raises(ValueError):
        RetrievalConfig(coverage_selector_uncertainty_entropy=1.01)


def test_partition_local_label_is_distinct_from_historical_pool_arm():
    config = RetrievalConfig(
        mode="hybrid_graph",
        k=10,
        source_local_search=True,
    )

    assert config.label.endswith("-local")


def test_hierarchical_partition_routing_is_explicit_and_graph_only():
    config = RetrievalConfig(
        mode="causal_graph",
        source_partition_routing=True,
        source_partition_slots=2,
    )

    assert "-part2" in config.label
    with pytest.raises(ValueError, match="requires hybrid_graph or causal_graph"):
        RetrievalConfig(mode="dense", source_partition_routing=True)


def test_qwen_rerank_label_and_bounds_are_explicit():
    config = RetrievalConfig(
        mode="causal_graph",
        source_local_search=True,
        qwen_rerank=True,
        qwen_rerank_slots=6,
    )

    assert config.label.endswith("-local-qwen6")
    with pytest.raises(ValueError, match="source_local_search"):
        RetrievalConfig(mode="causal_graph", qwen_rerank=True)
    with pytest.raises(ValueError, match="cannot exceed source_slots"):
        RetrievalConfig(
            mode="causal_graph",
            source_local_search=True,
            source_slots=2,
            qwen_rerank=True,
            qwen_rerank_slots=3,
        )


def test_qwen_feedback_is_a_distinct_bounded_graph_arm():
    config = RetrievalConfig(
        mode="causal_graph",
        source_local_search=True,
        qwen_feedback=True,
        qwen_feedback_slots=12,
    )

    assert config.label.endswith("-local-qwenfb12")
    with pytest.raises(ValueError, match="separate arms"):
        RetrievalConfig(
            mode="causal_graph",
            source_local_search=True,
            qwen_rerank=True,
            qwen_feedback=True,
        )
    with pytest.raises(ValueError, match="cannot exceed source_slots"):
        RetrievalConfig(
            mode="causal_graph",
            source_local_search=True,
            source_slots=4,
            qwen_feedback=True,
            qwen_feedback_slots=5,
        )


def test_eval_config_defaults():
    ec = EvalConfig()
    # The 3.5-Haiku defaults were retired 2026-02-19 and now 404.
    assert ec.judge_model == "anthropic/claude-sonnet-5"
    assert ec.responder_model == "anthropic/claude-haiku-4-5"
    assert ec.recent_window == 4


def test_eval_config_judge_and_responder_differ():
    """Judge and responder must not be the same model (validity)."""
    ec = EvalConfig()
    assert ec.judge_model != ec.responder_model


def test_usage_stats_defaults():
    u = UsageStats()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.cache_read_input_tokens == 0
    assert u.elapsed_s == 0.0
    assert u.calls == 0
    assert u.total_tokens == 0


def test_usage_stats_add():
    a = UsageStats(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=2,
        elapsed_s=1.5,
        calls=1,
    )
    b = UsageStats(
        input_tokens=7,
        output_tokens=3,
        cache_read_input_tokens=1,
        elapsed_s=0.5,
        calls=1,
    )
    c = a + b
    assert c.input_tokens == 17
    assert c.output_tokens == 8
    assert c.cache_read_input_tokens == 3
    assert c.elapsed_s == 2.0
    assert c.calls == 2
    assert c.total_tokens == 25
    # operands are unchanged (frozen)
    assert a.input_tokens == 10
    assert b.input_tokens == 7


def test_usage_stats_sum():
    stats = [UsageStats(input_tokens=i, calls=1) for i in range(4)]
    total = sum(stats, UsageStats())
    assert total.input_tokens == 6
    assert total.calls == 4
    # sum() without an explicit start works via __radd__
    assert sum(stats).input_tokens == 6


def test_usage_stats_from_litellm_defensive():
    class _NoUsage:
        pass

    u = UsageStats.from_litellm(_NoUsage(), elapsed_s=0.25)
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.elapsed_s == 0.25
    assert u.calls == 1


def test_usage_stats_from_litellm_reads_fields():
    class _Usage:
        prompt_tokens = 120
        completion_tokens = 34
        cache_read_input_tokens = 12

    class _Resp:
        usage = _Usage()

    u = UsageStats.from_litellm(_Resp(), elapsed_s=1.0)
    assert u.input_tokens == 120
    assert u.output_tokens == 34
    assert u.cache_read_input_tokens == 12
    assert u.total_tokens == 154


def test_turn_result():
    tr = TurnResult(
        turn_index=0,
        user_text="hi",
        actual_response="hello",
        generated_response="hey there",
        retrieved_chunks=["chunk1"],
        score=4,
        judge_reasoning="Good match",
    )
    assert tr.score == 4
    # New instrumentation fields default to empty
    assert tr.responder_usage == UsageStats()
    assert tr.judge_usage == UsageStats()
    assert tr.retrieval_s == 0.0
    assert tr.context_tokens == 0


def test_turn_result_with_usage():
    tr = TurnResult(
        turn_index=0,
        user_text="hi",
        actual_response="hello",
        generated_response="hey",
        retrieved_chunks=[],
        score=3,
        judge_reasoning="ok",
        responder_usage=UsageStats(input_tokens=50, output_tokens=10, calls=1),
        judge_usage=UsageStats(input_tokens=30, output_tokens=5, calls=1),
        retrieval_s=0.02,
        context_tokens=412,
    )
    combined = tr.responder_usage + tr.judge_usage
    assert combined.total_tokens == 95
    assert tr.context_tokens == 412


def test_conversation_result():
    cr = ConversationResult(
        filename="test.txt",
        num_turns=10,
        turn_results=[],
        mean_score=3.5,
    )
    assert cr.mean_score == 3.5
    assert cr.usage == UsageStats()


def test_eval_run_result_cost_fields():
    run = EvalRunResult(
        config=EvalConfig(),
        conversations=[],
        aggregate_mean_score=0.0,
        aggregate_recall_at_4=0.0,
        run_timestamp="2026-01-01T00:00:00Z",
    )
    assert run.usage == UsageStats()
    assert run.total_elapsed_s == 0.0
    assert run.mean_context_tokens == 0.0
    assert run.tokens_per_scored_turn == 0.0


def test_sweep_report():
    sr = SweepReport(runs=[], generated_at="2025-01-01T00:00:00Z")
    assert sr.best_config is None
