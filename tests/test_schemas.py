from memory_condense.schemas import Chunk, RetrievalResult, Turn


def test_turn_defaults():
    t = Turn(role="user", text="hello")
    assert t.turn_id  # non-empty
    assert t.role == "user"
    assert t.text == "hello"
    assert t.created_at is not None


def test_turn_frozen():
    t = Turn(role="user", text="hello")
    try:
        t.text = "changed"
        assert False, "Should have raised"
    except Exception:
        pass


def test_chunk_defaults():
    c = Chunk(turn_id="abc", text="hello", start_char=0, end_char=5, token_count=1)
    assert c.chunk_id
    assert c.embedding is None
    assert c.lexical_weights is None


def test_retrieval_result():
    c = Chunk(turn_id="abc", text="hello", start_char=0, end_char=5, token_count=1)
    r = RetrievalResult(chunk=c, score=0.95)
    assert r.score == 0.95
    assert r.turn is None
