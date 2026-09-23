import pytest
import sys

from tools.report_joint_source_spine_full100 import full100_gate
from tools.report_joint_source_spine_overflow_full100 import full100_gate as overflow_gate
from tools.report_joint_source_spine_overflow_full100_v2 import full100_gate as overflow_gate_v2
from tools.report_joint_source_spine_overflow_full100_v3 import full100_gate as overflow_gate_v3
from tools.report_joint_source_spine_overflow_full100_v4 import full100_gate as overflow_gate_v4
from tools.report_joint_source_spine_facets_full100 import full100_gate as facet_gate
from tools.report_joint_source_spine_facets_full100_v2 import full100_gate as facet_gate_v2
from tools.report_joint_source_spine_facets_full100_v3 import full100_gate as facet_gate_v3
from tools.report_joint_spine_reader_v3_full100 import full100_gate as reader_gate_v3
from tools.report_joint_source_spine_facets_full100_v4 import full100_gate as facet_gate_v4
from tools.report_joint_spine_reader_v3_full100_v2 import full100_gate as reader_gate_v3_v2
from tools.report_joint_spine_term_coverage_full100 import full100_gate as term_coverage_gate
from tools.report_joint_spine_source_coverage_full100 import full100_gate as source_coverage_gate
from tools.report_joint_spine_source_coverage_full100_v2 import full100_gate as source_coverage_gate_v2
from tools.report_joint_spine_combined_reader_full100 import full100_gate as combined_reader_gate
from tools.report_joint_spine_semantic_seeds_full100 import full100_gate as semantic_seed_gate
from tools.report_joint_spine_semantic_seeds_full100_v2 import full100_gate as semantic_seed_gate_v2
from tools.report_joint_spine_semantic_seeds_full100_v3 import full100_gate as semantic_seed_gate_v3


BASE_ARM, CANDIDATE_ARM = "spine_union", "source_spine"
EXTRA_ARMS = ()


@pytest.fixture(autouse=True, params=(
    (full100_gate, "spine_union", "source_spine"),
    (overflow_gate, "source_spine", "source_spine_overflow"),
    (overflow_gate_v2, "source_spine", "source_spine_overflow"),
    (overflow_gate_v3, "source_spine", "source_spine_overflow"),
    (overflow_gate_v4, "source_spine", "source_spine_overflow"),
    (facet_gate, "source_spine_overflow", "source_spine_facets"),
    (facet_gate_v2, "source_spine_overflow", "source_spine_facets"),
    (facet_gate_v3, "source_spine_overflow", "source_spine_facets"),
    (reader_gate_v3, "base", "reader"),
    (facet_gate_v4, "source_spine_overflow", "source_spine_facets"),
    (reader_gate_v3_v2, "base", "reader"),
    (term_coverage_gate, "source_spine_facets", "source_spine_term_coverage"),
    (source_coverage_gate, "source_spine_facets", "source_spine_diverse"),
    (source_coverage_gate_v2, "source_spine_facets", "source_spine_diverse"),
    (combined_reader_gate, "base", "combined_reader"),
    (semantic_seed_gate, "base", "semantic_seeds"),
    (semantic_seed_gate_v2, "base", "semantic_seeds"),
    (semantic_seed_gate_v3, "base", "semantic_seeds"),
))
def gate_version(request, monkeypatch):
    for name, value in zip(("full100_gate", "BASE_ARM", "CANDIDATE_ARM"), request.param):
        monkeypatch.setattr(sys.modules[__name__], name, value)
    arms = sys.modules[request.param[0].__module__].evaluation.MEMORY_ARMS
    monkeypatch.setattr(sys.modules[__name__], "EXTRA_ARMS", tuple(a for a in arms if a not in request.param[1:]))


def population():
    questions = [{"ordinal": i, "question_id": str(i)} for i in range(100)]
    rows = [{"ordinal": i, "question_id": str(i), "arm": arm, "correct": i < correct,
             "raw_tokens": 1_041_276, "memory": {"ttft_s": 5.2, "total_s": 5.5},
             "matched_api": {"ttft_s": 5, "total_s": 5.2}, "short_api": {"ttft_s": 5, "total_s": 5.2}}
            for arm, correct in ((BASE_ARM, 94), (CANDIDATE_ARM, 95), *((a, 93) for a in EXTRA_ARMS)) for i in range(100)]
    return questions, rows


def test_accuracy_and_latency_must_pass_on_one_complete_method():
    questions, rows = population()
    result = full100_gate(rows, questions)
    assert not result[BASE_ARM]["joint_gate_passed"]
    assert result[CANDIDATE_ARM]["joint_gate_passed"]
    for row in rows:
        if row["arm"] == CANDIDATE_ARM and row["ordinal"] >= 90:
            row["memory"]["ttft_s"] = 9
            row["memory"]["total_s"] = 9.5
    result = full100_gate(rows, questions)
    assert result[CANDIDATE_ARM]["accuracy_passed"]
    assert not result[CANDIDATE_ARM]["joint_gate_passed"]
    assert result[CANDIDATE_ARM]["latency_ratios"]["matched_api"]["ttft_s"]["p95_s"] == 1.8


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
    result = full100_gate(rows, questions)[CANDIDATE_ARM]
    assert result["latency_passed"]["matched_api"]
    assert not result["latency_passed"]["short_api"]
    assert not result["joint_gate_passed"]
