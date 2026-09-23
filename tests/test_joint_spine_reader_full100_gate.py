import pytest
import sys

from tools.report_joint_spine_reader_full100 import full100_gate
from tools.report_joint_spine_reader_v2_full100 import full100_gate as reader_v2_gate


@pytest.fixture(autouse=True, params=(full100_gate, reader_v2_gate))
def gate_version(request, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "full100_gate", request.param)


def population():
    questions = [{"ordinal": i, "question_id": str(i)} for i in range(100)]
    rows = [{"ordinal": i, "question_id": str(i), "arm": arm, "correct": i < correct,
             "raw_tokens": 1_041_276, "memory": {"ttft_s": 5.2, "total_s": 5.5},
             "matched_api": {"ttft_s": 5, "total_s": 5.2}, "short_api": {"ttft_s": 5, "total_s": 5.2}}
            for arm, correct in (("base", 94), ("reader", 95)) for i in range(100)]
    return questions, rows


def test_accuracy_and_latency_must_pass_on_one_complete_method():
    questions, rows = population()
    result = full100_gate(rows, questions)
    assert not result["base"]["joint_gate_passed"]
    assert result["reader"]["joint_gate_passed"]
    for row in rows:
        if row["arm"] == "reader" and row["ordinal"] >= 90:
            row["memory"]["ttft_s"] = 9
    result = full100_gate(rows, questions)
    assert result["reader"]["accuracy_passed"]
    assert not result["reader"]["joint_gate_passed"]
    assert result["reader"]["latency_ratios"]["matched_api"]["ttft_s"]["p95_s"] == 1.8


def test_partial_duplicate_foreign_or_small_memory_cannot_pass():
    questions, rows = population()
    for changed in (rows[:-1], rows + [rows[0]], [{**rows[0], "question_id": "foreign"}, *rows[1:]],
                    [{**rows[0], "raw_tokens": 800_000}, *rows[1:]]):
        with pytest.raises(ValueError):
            full100_gate(changed, questions)
    with pytest.raises(ValueError, match="complete locked"):
        full100_gate(rows[:20], questions[:10])


def test_fast_matched_prompt_does_not_hide_slow_short_chat_comparison():
    questions, rows = population()
    for row in rows:
        row["short_api"] = {"ttft_s": 2, "total_s": 2.1}
    result = full100_gate(rows, questions)["reader"]
    assert result["latency_passed"]["matched_api"]
    assert not result["latency_passed"]["short_api"]
    assert not result["joint_gate_passed"]
