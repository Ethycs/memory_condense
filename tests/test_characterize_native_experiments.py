"""Guard the experimental assay against coverage and scoring artifacts."""
import pytest

from tools.characterize_native_experiments import covered_quote, job_id, validate_grades


def span(start, end, turn="source"):
    return {"span": {"turn_id": turn, "start_char": start, "end_char": end}}


def test_quote_coverage_requires_contiguous_original_source():
    support = dict(turn_id="source", start_char=10, end_char=30)
    assert covered_quote(support, [span(0, 20), span(20, 35)])
    assert covered_quote(support, [span(15, 40), span(0, 25)])
    assert not covered_quote(support, [span(0, 19), span(20, 35)])
    assert not covered_quote(support, [span(0, 40, "copy")])


def test_deduplication_uses_complete_reader_input():
    first = [{"role": "user", "content": "question and evidence"}]
    same = [{"content": "question and evidence", "role": "user"}]
    changed = [{"role": "user", "content": "question and other evidence"}]
    assert job_id("Q1", first) == job_id("Q1", same)
    assert job_id("Q1", first) != job_id("Q1", changed)


def result(label="A", score=2):
    return dict(label=label, checkpoints={"F1": score}, scope_honesty=True, contradictions=[])


def test_grades_require_exact_answers_and_facts():
    valid = validate_grades({"answers": [result()]}, ["A"], ["F1"])
    assert valid["A"]["points"] == valid["A"]["possible"] == 2
    for answers in ([], [result("B")], [result(), result()], [result(score=True)],
                    [dict(result(), checkpoints={"F2": 2})]):
        with pytest.raises(ValueError):
            validate_grades({"answers": answers}, ["A"], ["F1"])
