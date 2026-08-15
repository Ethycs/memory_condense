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


def test_chunker_config_defaults():
    c = ChunkerConfig()
    assert c.min_tokens == 120
    assert c.max_tokens == 250


def test_retrieval_config_defaults():
    r = RetrievalConfig()
    assert r.k == 10
    assert r.ef_search == 50


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
