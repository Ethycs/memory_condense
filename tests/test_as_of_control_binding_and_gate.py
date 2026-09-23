from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from tools import evaluate_spine_as_of as evaluation
from tools.report_joint_spine_as_of_full100 import full100_gate


def control_and_candidate():
    original, calls = [], []
    for i in range(10):
        question = {"ordinal": i, "question_id": f"opaque-{i}", "retrieval_query": "Which adapter?",
            "prompt_question": "[Question asked at 2026/09/10 (Thu) 12:00]\nWhich adapter?"}
        control_messages = evaluation.answer_messages(question,
            SimpleNamespace(render_context=lambda: "Exact original evidence."))
        original.append({"arm": "semantic_seeds", "question": question, "messages": control_messages})
        for arm in evaluation.call_arm_order(i):
            if arm == "short_api":
                messages = evaluation.answer_messages(question)
            elif arm.startswith("semantic_seeds"):
                messages = deepcopy(control_messages)
            else:
                messages = evaluation.answer_messages(question,
                    SimpleNamespace(render_context=lambda: "Exact eligible evidence."))
            calls.append({"call_index": len(calls), "arm": arm, "question": dict(question),
                          "messages": messages, "messages_sha256": identity_sha256(messages)})
    return SimpleNamespace(payload={"calls": original, "shard_offset": 0}), calls


def test_control_messages_and_question_order_reproduce_the_independent_frozen_control():
    original, calls = control_and_candidate()
    evaluation.validate_matched_calls(calls)
    evaluation.validate_control_calls(original, calls)


@pytest.mark.parametrize("change", ["question", "control", "order", "missing", "index"])
def test_resealing_a_changed_control_population_or_prompt_does_not_make_it_valid(change):
    original, calls = control_and_candidate()
    if change == "question":
        calls[0]["question"]["question_id"] = "foreign"
    elif change == "control":
        for call in calls:
            if call["arm"].startswith("semantic_seeds"):
                call["messages"][1]["content"] += " Added evidence."
                call["messages_sha256"] = identity_sha256(call["messages"])
        # The pair still matches each other; only the old control binding detects it.
        evaluation.validate_matched_calls(calls)
    elif change == "order":
        calls[1], calls[2] = calls[2], calls[1]
        for index, call in enumerate(calls):
            call["call_index"] = index
    elif change == "missing":
        calls.pop()
    else:
        calls[0]["call_index"] = 10
    with pytest.raises(ValueError):
        evaluation.validate_control_calls(original, calls)


def population():
    questions = [{"ordinal": i, "question_id": f"opaque-{i}"} for i in range(100)]
    rows = [{"ordinal": i, "question_id": f"opaque-{i}", "arm": arm, "correct": i < correct,
        "raw_tokens": 1_041_276, "memory": {"ttft_s": 5.2, "total_s": 5.5},
        "matched_api": {"ttft_s": 5, "total_s": 5.2}, "short_api": {"ttft_s": 5, "total_s": 5.2}}
        for arm, correct in (("semantic_seeds", 94), ("as_of", 95)) for i in range(100)]
    return questions, rows


def test_accuracy_and_both_latency_baselines_must_pass_on_the_same_complete_arm():
    questions, rows = population()
    result = full100_gate(rows, questions)
    assert not result["semantic_seeds"]["joint_gate_passed"]
    assert result["as_of"]["joint_gate_passed"]
    for row in rows:
        if row["arm"] == "as_of":
            row["short_api"] = {"ttft_s": 2, "total_s": 2.1}
    result = full100_gate(rows, questions)["as_of"]
    assert result["accuracy_passed"] and result["latency_passed"]["matched_api"]
    assert not result["latency_passed"]["short_api"] and not result["joint_gate_passed"]


def test_slow_p95_fails_despite_fast_median_and_95_correct_answers():
    questions, rows = population()
    for row in rows:
        if row["arm"] == "as_of" and row["ordinal"] >= 90:
            row["memory"] = {"ttft_s": 9, "total_s": 9.1}
    result = full100_gate(rows, questions)["as_of"]
    assert result["accuracy_passed"] and not result["joint_gate_passed"]
    assert result["latency_ratios"]["matched_api"]["ttft_s"]["p95_s"] == 1.8


@pytest.mark.parametrize("change", ["missing", "duplicate", "small", "nonfinite", "chronology", "unjudged", "relaxed"])
def test_incomplete_or_invalid_measurements_cannot_pass(change):
    questions, rows = population()
    allowance = 1.10
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif change == "small":
        rows[0]["raw_tokens"] = 999_999
    elif change == "nonfinite":
        rows[0]["memory"]["total_s"] = float("nan")
    elif change == "chronology":
        rows[0]["memory"]["ttft_s"] = 10
    elif change == "unjudged":
        rows[0]["correct"] = 1
    else:
        allowance = 1.2
    with pytest.raises(ValueError):
        full100_gate(rows, questions, latency_ratio_limit=allowance)
