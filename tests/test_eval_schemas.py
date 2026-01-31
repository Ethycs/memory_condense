from memory_condense.eval.schemas import (
    ChunkerConfig,
    ConversationResult,
    EvalConfig,
    EvalRunResult,
    RetrievalConfig,
    SweepReport,
    TurnResult,
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
    assert ec.judge_model == "anthropic/claude-3-5-haiku-20241022"
    assert ec.responder_model == "anthropic/claude-3-5-haiku-20241022"
    assert ec.recent_window == 4


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


def test_conversation_result():
    cr = ConversationResult(
        filename="test.txt",
        num_turns=10,
        turn_results=[],
        mean_score=3.5,
    )
    assert cr.mean_score == 3.5


def test_sweep_report():
    sr = SweepReport(runs=[], generated_at="2025-01-01T00:00:00Z")
    assert sr.best_config is None
