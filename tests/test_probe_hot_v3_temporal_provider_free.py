from __future__ import annotations

import json

from tools.probe_hot_v3_temporal_provider_free import (
    _computed_temporal_is_admissible,
    _evidence_posthoc,
    _iter_json_array,
    _without_posthoc,
)


def test_computed_terminal_admissibility_is_shape_and_operation_bounded() -> None:
    duration = {"answer_shape": "duration"}
    assert _computed_temporal_is_admissible(
        prediction_source="computed",
        operation="direct_duration",
        validation_contract=duration,
        provider_input={},
    ) == (True, "duration")
    assert _computed_temporal_is_admissible(
        prediction_source="computed",
        operation="event_interval",
        validation_contract={"answer_shape": "direct"},
        provider_input={},
    ) == (False, "direct")
    assert _computed_temporal_is_admissible(
        prediction_source="parent",
        operation="event_order",
        validation_contract={"answer_shape": "ordered_list"},
        provider_input={},
    ) == (False, "ordered_list")


def test_computed_terminal_admissibility_reads_legacy_operator_fallback() -> None:
    assert _computed_temporal_is_admissible(
        prediction_source="computed",
        operation="event_order",
        validation_contract={},
        provider_input={"typed_evidence": {"operator_spec": {"answer_shape": "ordered_list"}}},
    ) == (True, "ordered_list")


def test_large_dataset_reader_crosses_small_chunk_boundaries(tmp_path) -> None:
    rows = [
        {"question_id": "one", "answer": "café"},
        {"question_id": "two", "answer": ["alpha", "beta"]},
    ]
    path = tmp_path / "population.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    assert list(_iter_json_array(path, chunk_chars=7)) == rows


def test_source_reach_is_distinct_from_answer_bearing_span() -> None:
    target = {"answer": "a smoker", "answer_session_ids": ["answer-session"]}
    binding = {"candidate_id": "candidate", "source_id": "qid::answer-session"}

    wrong_span = _evidence_posthoc(
        [{"candidate_id": "candidate", "quote": "Remember to turn off appliances."}],
        [binding],
        target,
    )
    right_span = _evidence_posthoc(
        [{"candidate_id": "candidate", "quote": "I bought a smoker yesterday."}],
        [binding],
        target,
    )

    assert wrong_span["all_target_sources_reached"] is True
    assert wrong_span["answer_literal_in_target_source_quote"] is False
    assert right_span["answer_literal_in_target_source_quote"] is True


def test_gold_free_projection_strips_only_posthoc_fields() -> None:
    value = {
        "ordinal": 54,
        "full_store": {
            "artifact_sha256": "a" * 64,
            "posthoc_source_coverage": {"all_target_sources_reached": True},
            "rebound": True,
        },
        "posthoc_reference_answer": "a smoker",
        "v7_correct": False,
    }

    assert _without_posthoc(value) == {
        "ordinal": 54,
        "full_store": {"artifact_sha256": "a" * 64, "rebound": True},
    }
