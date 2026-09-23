from copy import deepcopy

import pytest

from tools.assess_native_occurrences import occurrences
from tools.diagnose_native_spine_retrieval import annotation_targets, coverage
from tools.prepare_native_spine_sources import source_session


def fixture():
    turns = [{"role": "user", "content": "", "has_answer": False},
             {"role": "user", "content": "I bought a Philips bulb.", "has_answer": True}]
    record = {"haystack_session_ids": ["repeated", "repeated"],
              "haystack_dates": ["2023/05/01 (Mon) 10:00", "2023/05/02 (Tue) 11:00"],
              "haystack_sessions": [turns, deepcopy(turns)]}
    return record, [source_session(o, "M") for o in occurrences(record)]


def test_annotations_preserve_repeated_session_occurrences_and_normalized_turn_order():
    record, sources = fixture()
    targets = annotation_targets(record, sources)
    assert len(targets) == 2
    assert len({t["source_id"] for t in targets.values()}) == 2
    assert len({t["created_at"] for t in targets.values()}) == 2
    assert len({t["turn_text_sha256"] for t in targets.values()}) == 1


def test_annotations_reject_changed_actual_source_date_even_for_identical_text():
    record, sources = fixture()
    sources[0]["created_at"] = sources[1]["created_at"]
    with pytest.raises(ValueError, match="different native occurrence"):
        annotation_targets(record, sources)


def test_coverage_distinguishes_missing_partial_and_complete_exact_turns():
    record, sources = fixture()
    targets = annotation_targets(record, sources)
    first, second = targets.values()
    partial = dict(first, start_char=0, end_char=5)
    assert not coverage([], targets)["any_overlap"]
    assert coverage([partial], targets)["overlapped_turns"] == 1
    assert coverage([partial], targets)["fully_covered_turns"] == 0
    spans = [partial, dict(first, start_char=5, end_char=first["length"]),
             dict(second, start_char=0, end_char=second["length"])]
    assert coverage(spans, targets)["all_fully_covered"]
    assert not coverage([], {})["all_fully_covered"]


@pytest.mark.parametrize("change", ["foreign_source", "invalid_interval"])
def test_coverage_rejects_false_identity_or_out_of_bounds_evidence(change):
    record, sources = fixture()
    targets = annotation_targets(record, sources)
    target = next(iter(targets.values()))
    span = dict(target, start_char=0, end_char=target["length"])
    if change == "foreign_source":
        span["source_id"] = "elsewhere"
    else:
        span["end_char"] += 1
    with pytest.raises(ValueError):
        coverage([span], targets)
