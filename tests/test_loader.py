import json
from pathlib import Path

import pytest

from memory_condense.loader import (
    detect_benchmark_format,
    load_benchmark,
    load_conversation,
    load_directory,
    parse_locomo,
    parse_longmemeval,
    parse_md,
    parse_txt,
)


def test_parse_txt_basic():
    text = (
        "User:\n"
        "Hello world\n"
        "\n"
        "Claude:\n"
        " Hi there! How can I help?\n"
        "\n"
        "User:\n"
        "Tell me about Python\n"
    )
    turns = parse_txt(text)
    assert len(turns) == 3
    assert turns[0] == ("user", "Hello world")
    assert turns[1] == ("assistant", "Hi there! How can I help?")
    assert turns[2] == ("user", "Tell me about Python")


def test_parse_txt_multiline():
    text = (
        "User:\n"
        "First line\n"
        "Second line\n"
        "\n"
        "Claude:\n"
        " Response line one\n"
        " Response line two\n"
    )
    turns = parse_txt(text)
    assert len(turns) == 2
    assert "First line" in turns[0][1]
    assert "Second line" in turns[0][1]
    assert "Response line one" in turns[1][1]


def test_parse_txt_skips_empty_bodies():
    text = "User:\n\nClaude:\n \n\nUser:\nActual content\n"
    turns = parse_txt(text)
    assert len(turns) == 1
    assert turns[0][1] == "Actual content"


def test_parse_md_basic():
    text = (
        "**User:**\n"
        "\n"
        "Hello world\n"
        "\n"
        "**Assistant:**\n"
        "\n"
        "Hi there! How can I help?\n"
    )
    turns = parse_md(text)
    assert len(turns) == 2
    assert turns[0] == ("user", "Hello world")
    assert turns[1] == ("assistant", "Hi there! How can I help?")


def test_parse_md_skips_empty_assistant():
    text = (
        "**Assistant:**\n"
        "\n"
        "\n"
        "\n"
        "**Assistant:**\n"
        "\n"
        "Actual response here\n"
    )
    turns = parse_md(text)
    assert len(turns) == 1
    assert turns[0] == ("assistant", "Actual response here")


def test_load_conversation_txt(tmp_path: Path):
    f = tmp_path / "chat.txt"
    f.write_text("User:\nHello\n\nClaude:\n Reply\n", encoding="utf-8")
    turns = load_conversation(f)
    assert len(turns) == 2
    assert turns[0][0] == "user"
    assert turns[1][0] == "assistant"


def test_load_conversation_md(tmp_path: Path):
    f = tmp_path / "chat.md"
    f.write_text("**User:**\nHello\n\n**Assistant:**\nReply\n", encoding="utf-8")
    turns = load_conversation(f)
    assert len(turns) == 2


def test_load_directory(tmp_path: Path):
    (tmp_path / "a.txt").write_text("User:\nHello\n\nClaude:\n Hi\n")
    (tmp_path / "b.md").write_text("**User:**\nHey\n\n**Assistant:**\nHi\n")
    (tmp_path / "c.json").write_text("{}")  # should be skipped

    result = load_directory(tmp_path)
    assert "a.txt" in result
    assert "b.md" in result
    assert "c.json" not in result
    assert len(result["a.txt"]) == 2
    assert len(result["b.md"]) == 2


def test_load_real_txt_format():
    """Test against the actual format seen in the user's files."""
    text = (
        "User:\n"
        "Can I use a transformer to calculate shannon entropy\n"
        "\n"
        "Claude:\n"
        " You can use a transformer to calculate Shannon entropy, "
        "though it's not the most straightforward application.\n"
        "\n"
        "User:\n"
        "basically I want to calculate conditional probability\n"
    )
    turns = parse_txt(text)
    assert len(turns) == 3
    assert turns[0][0] == "user"
    assert "shannon entropy" in turns[0][1].lower()
    assert turns[1][0] == "assistant"
    assert turns[2][0] == "user"


def test_load_real_md_format():
    """Test against the actual format seen in the user's .md files."""
    text = (
        "**Assistant:**\n"
        "\n"
        "\n"
        "\n"
        "**Assistant:**\n"
        "\n"
        "The phrase **genericity** is about singularity theory.\n"
        "\n"
        "**User:**\n"
        "\n"
        "Tell me more about Whitney stratification\n"
    )
    turns = parse_md(text)
    assert len(turns) == 2
    assert turns[0][0] == "assistant"
    assert "genericity" in turns[0][1]
    assert turns[1][0] == "user"


# ---------------------------------------------------------------------------
# Benchmark fixtures (inline — nothing is downloaded)
# ---------------------------------------------------------------------------


LONGMEMEVAL_RECORD = {
    "question_id": "lme_001",
    "question_type": "multi-session",
    "question": "Which city did I move to?",
    "answer": "Boston",
    "question_date": "2023/06/01 (Thu) 10:00",
    "haystack_dates": ["2023/05/01 (Mon) 09:00", "2023/05/20 (Sat) 02:29"],
    "haystack_session_ids": ["s1", "s2"],
    "haystack_sessions": [
        [
            {"role": "user", "content": "I am thinking about relocating."},
            {"role": "assistant", "content": "Where are you considering?"},
        ],
        [
            {"role": "user", "content": "I moved to Boston last week."},
            {"role": "assistant", "content": "Congratulations on the move!"},
        ],
    ],
    "answer_session_ids": ["s2"],
}

LOCOMO_RECORD = {
    "sample_id": "conv-26",
    "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel!"},
            {"speaker": "Melanie", "dia_id": "D1:2", "text": "Hi Caroline!"},
        ],
        "session_2_date_time": "4:00 pm on 20 May, 2023",
        "session_2": [
            {"speaker": "Melanie", "dia_id": "D2:1", "text": "I adopted a dog."},
            {"speaker": "Caroline", "dia_id": "D2:2", "text": "What is its name?"},
        ],
    },
    "qa": [
        {
            "question": "What did Melanie adopt?",
            "answer": "A dog",
            "category": 2,
            "evidence": ["D2:1"],
        },
        {
            "question": "Who greeted first?",
            "answer": "Caroline",
            "category": 1,
        },
    ],
}


def test_parse_longmemeval_flattens_sessions_in_order():
    samples = parse_longmemeval([LONGMEMEVAL_RECORD])
    assert len(samples) == 1
    s = samples[0]
    assert s.sample_id == "lme_001"
    assert [t[0] for t in s.turns] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert s.turns[0][1] == "I am thinking about relocating."
    assert s.turns[2][1] == "I moved to Boston last week."

    assert len(s.questions) == 1
    q = s.questions[0]
    assert q.question == "Which city did I move to?"
    assert q.answer == "Boston"
    assert q.category == "multi-session"
    assert q.evidence == ["s2"]


def test_parse_longmemeval_tolerates_missing_optional_keys():
    minimal = {"question": "Q?", "answer": "A"}
    samples = parse_longmemeval([minimal])
    assert len(samples) == 1
    assert samples[0].turns == []
    assert samples[0].questions[0].category is None
    assert samples[0].questions[0].evidence == []


def test_parse_longmemeval_skips_malformed_records():
    data = [
        "not a dict",
        {"no_question_key": True},
        {"question": "", "answer": "x"},
        {"question": "Good?", "answer": "yes", "haystack_sessions": "bad type"},
        LONGMEMEVAL_RECORD,
    ]
    samples = parse_longmemeval(data)
    assert len(samples) == 2
    assert samples[0].turns == []  # bad haystack type -> no turns, no raise
    assert samples[1].sample_id == "lme_001"


def test_parse_locomo_orders_sessions_and_maps_speakers():
    samples = parse_locomo([LOCOMO_RECORD])
    assert len(samples) == 1
    s = samples[0]
    assert s.sample_id == "conv-26"
    # First-seen speaker (Caroline) -> user, the other -> assistant.
    # Each session is preceded by its timestamp — see
    # test_parse_locomo_ingests_session_timestamps for why.
    assert s.turns == [
        ("system", "[session_1 took place at 1:56 pm on 8 May, 2023]"),
        ("user", "Hey Mel!"),
        ("assistant", "Hi Caroline!"),
        ("system", "[session_2 took place at 4:00 pm on 20 May, 2023]"),
        ("assistant", "I adopted a dog."),
        ("user", "What is its name?"),
    ]
    assert [q.question for q in s.questions] == [
        "What did Melanie adopt?",
        "Who greeted first?",
    ]
    assert s.questions[0].category == "2"
    assert s.questions[0].evidence == ["D2:1"]
    assert s.questions[1].evidence == []
    assert s.questions[0].question_id == "conv-26_q0"


def test_parse_locomo_ingests_session_timestamps():
    """LoCoMo's largest question category asks *when*, and the answers are
    these dates — which live in `session_N_date_time`, not in any turn's text.

    Dropping them made that category unanswerable from the ingested data. On
    the real conv-26, only 2.7% of category-2 gold answers appeared anywhere
    in the haystack before this, against 43-45% for the non-temporal
    categories; a benchmark run would have scored retrieval for information it
    was never given.
    """
    samples = parse_locomo([LOCOMO_RECORD])
    text = " ".join(t for _, t in samples[0].turns)

    assert "8 May, 2023" in text
    assert "20 May, 2023" in text


def test_parse_locomo_tolerates_a_session_with_no_timestamp():
    record = {
        "sample_id": "conv-undated",
        "conversation": {"session_1": [{"speaker": "A", "text": "hello"}]},
        "qa": [],
    }
    assert parse_locomo([record])[0].turns == [("user", "hello")]


def test_parse_locomo_sorts_sessions_numerically():
    record = {
        "sample_id": "conv-num",
        "conversation": {
            "session_10": [{"speaker": "A", "text": "tenth"}],
            "session_2": [{"speaker": "A", "text": "second"}],
            "session_1": [{"speaker": "A", "text": "first"}],
        },
        "qa": [],
    }
    samples = parse_locomo([record])
    assert [t[1] for t in samples[0].turns] == ["first", "second", "tenth"]


def test_parse_locomo_skips_malformed_records():
    data = [
        42,
        {"sample_id": "no-conversation"},
        {"sample_id": "bad-conv", "conversation": "not a dict"},
        {
            "sample_id": "partial",
            "conversation": {"session_1": ["junk", {"speaker": "A", "text": "ok"}]},
            "qa": ["junk", {"no_question": 1}, {"question": "Real?", "answer": "y"}],
        },
    ]
    samples = parse_locomo(data)
    assert len(samples) == 1
    assert samples[0].sample_id == "partial"
    assert samples[0].turns == [("user", "ok")]
    assert len(samples[0].questions) == 1
    assert samples[0].questions[0].question == "Real?"


def test_detect_benchmark_format():
    assert detect_benchmark_format([LONGMEMEVAL_RECORD]) == "longmemeval"
    assert detect_benchmark_format([LOCOMO_RECORD]) == "locomo"
    with pytest.raises(ValueError):
        detect_benchmark_format([{"unrelated": 1}])


def test_load_benchmark_auto_detects_longmemeval(tmp_path: Path):
    f = tmp_path / "lme.json"
    f.write_text(json.dumps([LONGMEMEVAL_RECORD]), encoding="utf-8")
    samples = load_benchmark(f)
    assert len(samples) == 1
    assert len(samples[0].turns) == 4


def test_load_benchmark_auto_detects_locomo(tmp_path: Path):
    f = tmp_path / "locomo.json"
    f.write_text(json.dumps([LOCOMO_RECORD]), encoding="utf-8")
    samples = load_benchmark(f)
    assert len(samples) == 1
    assert len(samples[0].questions) == 2


def test_load_benchmark_jsonl_and_skips_bad_lines(tmp_path: Path):
    f = tmp_path / "lme.jsonl"
    f.write_text(
        json.dumps(LONGMEMEVAL_RECORD) + "\n" + "{not json}\n" + "\n",
        encoding="utf-8",
    )
    samples = load_benchmark(f, format="longmemeval")
    assert len(samples) == 1


def test_load_benchmark_explicit_format_and_bad_format(tmp_path: Path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps([LOCOMO_RECORD]), encoding="utf-8")
    assert load_benchmark(f, format="locomo")[0].sample_id == "conv-26"
    with pytest.raises(ValueError):
        load_benchmark(f, format="nonsense")


def test_load_benchmark_accepts_wrapper_dict(tmp_path: Path):
    f = tmp_path / "wrapped.json"
    f.write_text(json.dumps({"data": [LONGMEMEVAL_RECORD]}), encoding="utf-8")
    samples = load_benchmark(f)
    assert len(samples) == 1
