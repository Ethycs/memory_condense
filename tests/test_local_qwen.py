from memory_condense.eval.local_qwen import strip_qwen_thinking


def test_strip_qwen_thinking_keeps_only_visible_answer() -> None:
    assert strip_qwen_thinking("<think>private work</think>\nBoston") == "Boston"


def test_strip_qwen_thinking_leaves_plain_answer_unchanged() -> None:
    assert strip_qwen_thinking("  SQLite  ") == "SQLite"
