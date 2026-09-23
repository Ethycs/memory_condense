from __future__ import annotations

from dataclasses import replace

import pytest

from memory_condense.associations.head_memory_models import (
    AssociativeMemoryCandidate,
    MemoryLinkHit,
    NestedMemoryInspection,
)
from memory_condense.search.selectors.qkov_distilled_selector import (
    QKOVDistilledPolicy,
    QKOVDistilledSelector,
    QKOVStudentIdentity,
)


def _digest(character: str) -> str:
    return character * 64


def _identity() -> QKOVStudentIdentity:
    return QKOVStudentIdentity(
        model_id="local/minilm-qkov",
        model_revision="revision-1",
        checkpoint_sha256=_digest("a"),
        tokenizer_sha256=_digest("b"),
        contract_sha256=_digest("c"),
    )


class _Student:
    def __init__(self, scores=None, error: Exception | None = None) -> None:
        self.scores = [0.1, 0.9, 0.4] if scores is None else scores
        self.error = error
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((tuple(pairs), kwargs))
        if self.error is not None:
            raise self.error
        return self.scores


class _Teacher:
    max_candidates = 8

    def __init__(self, winner: str = "c", error: Exception | None = None) -> None:
        self.winner = winner
        self.error = error
        self.calls = []

    def inspect_nested(self, query, groups, **kwargs):
        self.calls.append((query, groups, kwargs))
        if self.error is not None:
            raise self.error
        return NestedMemoryInspection(
            hits=(
                MemoryLinkHit(
                    episode_id=self.winner,
                    qk_score=0.75,
                    ov_transport=0.25,
                    head_weights=(0.5,),
                    metadata={"selection_backend": "teacher"},
                ),
            ),
            passes=2,
            max_workspace_candidates=3,
            max_workspace_tokens=30,
            total_candidate_inspections=4,
        )


class _StaticTeacher:
    max_candidates = 8

    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def inspect_nested(self, query, groups, **kwargs):
        self.calls.append((query, groups, kwargs))
        return self.result


def _groups():
    return (
        (
            AssociativeMemoryCandidate("a", "alpha", metadata={"source": "one"}),
            AssociativeMemoryCandidate("b", "bravo"),
        ),
        (AssociativeMemoryCandidate("c", "charlie"),),
    )


def _selector(
    student,
    *,
    mode="apply",
    margin=0.1,
    teacher=None,
    identity=None,
    expected_identity=None,
    scope_complete=True,
    requires_completeness=False,
    candidate_count_truncated=False,
    max_student_candidates=64,
):
    actual = identity or _identity()
    policy = QKOVDistilledPolicy(
        expected_student_identity_sha256=(
            expected_identity or actual.identity_sha256
        ),
        mode=mode,
        min_margin=margin,
        max_student_candidates=max_student_candidates,
    )
    return QKOVDistilledSelector(
        student,
        identity=actual,
        policy=policy,
        scope_complete=scope_complete,
        requires_completeness=requires_completeness,
        candidate_count_truncated=candidate_count_truncated,
        teacher=teacher,
    )


def test_apply_accepts_only_the_stable_high_margin_student_order() -> None:
    student = _Student()
    teacher = _Teacher()
    selector = _selector(student, teacher=teacher)

    result = selector.inspect_nested(
        "question", _groups(), beam_per_group=2, top_k=2, score_mode="qk_ov"
    )

    assert [hit.episode_id for hit in result.hits] == ["b", "c"]
    assert teacher.calls == []
    assert student.calls[0][0] == (
        ("question", "alpha"),
        ("question", "bravo"),
        ("question", "charlie"),
    )
    assert result.hits[0].qk_score == 0.0
    assert result.hits[0].ov_transport == 0.0
    assert not result.hits[0].metadata["qk_score_available"]
    assert not result.hits[0].metadata["ov_transport_available"]
    assert result.hits[0].metadata["compatibility_qk_score_is_placeholder"]
    assert result.hits[0].metadata["score_kind"] == "distilled_ranking_scalar"
    assert result.hits[0].metadata["score_semantics"].endswith("not_qk_or_ov")
    assert selector.last_distilled_selection.hits[0].score == pytest.approx(0.9)
    assert not hasattr(selector.last_distilled_selection.hits[0], "qk_score")
    assert selector.last_report.decision == "student"
    assert selector.last_report.effective_top_k == 2
    assert not selector.last_report.selection_exhaustive
    assert selector.last_report.observed_margin == pytest.approx(0.3)
    assert not selector.requires_raw_fallback


def test_margin_is_measured_at_the_returned_set_cutoff() -> None:
    teacher = _Teacher()
    selector = _selector(
        _Student([1.0, 0.8, 0.79]),
        margin=0.05,
        teacher=teacher,
    )

    result = selector.inspect_nested("question", _groups(), top_k=2)

    assert result.hits[0].episode_id == "c"
    assert selector.last_report.decision == "teacher"
    assert selector.last_report.observed_margin == pytest.approx(0.01)
    assert selector.last_report.student_candidate_ids == ("a", "b")
    assert teacher.calls[0][2]["top_k"] == 2


def test_effective_top_k_and_exhaustive_selection_are_explicit() -> None:
    exhaustive = _selector(_Student())

    exhaustive_result = exhaustive.inspect_nested(
        "question",
        _groups(),
        beam_per_group=2,
        top_k=99,
    )

    assert [hit.episode_id for hit in exhaustive_result.hits] == ["b", "c", "a"]
    assert exhaustive.last_report.effective_top_k == 3
    assert exhaustive.last_report.selection_exhaustive
    assert exhaustive.last_report.observed_margin is None
    assert exhaustive.last_report.reason == "student_exhaustive_accepted"

    structurally_capped = _selector(_Student())
    capped_result = structurally_capped.inspect_nested(
        "question",
        (_groups()[0] + _groups()[1],),
        beam_per_group=1,
        top_k=99,
    )

    assert [hit.episode_id for hit in capped_result.hits] == ["b"]
    assert structurally_capped.last_report.effective_top_k == 1
    assert not structurally_capped.last_report.selection_exhaustive
    assert structurally_capped.last_report.observed_margin == pytest.approx(0.5)


def test_nested_linker_bounds_are_enforced_before_scoring() -> None:
    student = _Student()
    selector = _selector(student)

    with pytest.raises(ValueError, match="smaller than max_candidates"):
        selector.inspect_nested(
            "question",
            _groups(),
            beam_per_group=selector.max_candidates,
        )
    oversized = tuple(
        AssociativeMemoryCandidate(str(index), f"candidate {index}")
        for index in range(selector.max_candidates + 1)
    )
    with pytest.raises(ValueError, match="group exceeds max_candidates"):
        selector.inspect_nested("question", (oversized,))

    assert student.calls == []


@pytest.mark.parametrize(
    ("scores", "reason"),
    (
        ([0.4, 0.5, 0.4], "student_margin_not_strictly_above_threshold"),
        ([0.5, 0.5, 0.1], "student_top_score_tie"),
        ([0.1, float("nan"), 0.2], "nonfinite_student_score"),
        ([0.1, 0.2], "score_count_mismatch"),
    ),
)
def test_uncertain_student_delegates_full_unchanged_groups(scores, reason) -> None:
    teacher = _Teacher()
    selector = _selector(_Student(scores), margin=0.1, teacher=teacher)
    groups = _groups()

    result = selector.inspect_nested("question", groups, top_k=1)

    assert result.hits[0].episode_id == "c"
    assert teacher.calls[0][1] == groups
    passed_ids = [
        candidate.episode_id
        for group in teacher.calls[0][1]
        for candidate in group
    ]
    assert passed_ids == [
        "a",
        "b",
        "c",
    ]
    assert selector.last_report.decision == "teacher"
    assert selector.last_report.reason == reason


@pytest.mark.parametrize(
    ("scope_complete", "requires_completeness", "reason"),
    (
        (False, False, "scope_incomplete"),
        (True, True, "query_requires_completeness"),
    ),
)
def test_scope_and_completeness_gates_run_teacher_without_student(
    scope_complete, requires_completeness, reason
) -> None:
    student = _Student()
    teacher = _Teacher()
    selector = _selector(
        student,
        teacher=teacher,
        scope_complete=scope_complete,
        requires_completeness=requires_completeness,
    )

    selector.inspect_nested("question", _groups())

    assert student.calls == []
    assert len(teacher.calls) == 1
    assert selector.last_report.reason == reason
    assert selector.last_report.requires_raw_fallback


def test_explicit_truncated_candidate_count_never_calls_student() -> None:
    student = _Student()
    teacher = _Teacher()
    selector = _selector(
        student,
        teacher=teacher,
        candidate_count_truncated=True,
    )

    selector.inspect_nested("question", _groups())

    assert student.calls == []
    assert len(teacher.calls) == 1
    assert selector.last_report.reason == "candidate_count_truncated"
    assert selector.last_report.candidate_count_truncated


def test_identity_mismatch_and_candidate_overflow_never_call_student() -> None:
    student = _Student()
    teacher = _Teacher()
    mismatch = _selector(
        student,
        teacher=teacher,
        expected_identity=_digest("d"),
    )
    mismatch.inspect_nested("question", _groups())
    assert mismatch.last_report.reason == "student_identity_mismatch"

    overflow = _selector(
        student,
        teacher=teacher,
        max_student_candidates=2,
    )
    overflow.inspect_nested("question", _groups())
    assert overflow.last_report.reason == "student_candidate_limit_exceeded"
    assert student.calls == []
    assert len(teacher.calls) == 2


def test_shadow_records_proposal_but_returns_full_frontier_teacher_result() -> None:
    teacher = _Teacher(winner="a")
    selector = _selector(_Student(), mode="shadow", teacher=teacher)

    result = selector.inspect_nested("question", _groups(), top_k=1)

    assert [hit.episode_id for hit in result.hits] == ["a"]
    assert selector.last_report.decision == "teacher"
    assert selector.last_report.reason == "shadow_mode"
    assert selector.last_report.student_accepted
    assert selector.last_report.student_candidate_ids == ("b",)
    assert len(teacher.calls) == 1


@pytest.mark.parametrize(
    "teacher",
    (None, _Teacher(error=RuntimeError("offline"))),
)
def test_unavailable_teacher_fails_open_and_marks_raw(teacher) -> None:
    selector = _selector(
        _Student(error=RuntimeError("student failed")), teacher=teacher
    )

    result = selector.inspect_nested("question", _groups(), top_k=1)

    assert [hit.episode_id for hit in result.hits] == ["a", "b", "c"]
    assert selector.last_report.decision == "fail_open"
    assert selector.last_report.requires_raw_fallback
    assert selector.requires_raw_fallback
    assert all(hit.metadata["requires_raw_fallback"] for hit in result.hits)
    assert all(not hit.metadata["qk_score_available"] for hit in result.hits)
    assert all(not hit.metadata["ov_transport_available"] for hit in result.hits)


def test_fail_open_deduplicates_by_first_occurrence_in_original_order() -> None:
    duplicate_groups = (
        (
            AssociativeMemoryCandidate("a", "first", metadata={"copy": "first"}),
            AssociativeMemoryCandidate("a", "second", metadata={"copy": "second"}),
            AssociativeMemoryCandidate("b", "third"),
        ),
    )
    selector = _selector(_Student(), teacher=None)

    result = selector.inspect_nested("question", duplicate_groups)

    assert [hit.episode_id for hit in result.hits] == ["a", "b"]
    assert result.hits[0].metadata["copy"] == "first"
    assert selector.last_report.reason == (
        "teacher_unavailable:duplicate_or_empty_candidate_id"
    )


def test_student_exception_uses_teacher_and_retains_teacher_qkov() -> None:
    teacher = _Teacher(winner="b")
    selector = _selector(
        _Student(error=RuntimeError("student failed")), teacher=teacher
    )

    result = selector.inspect_nested("question", _groups())

    assert result.hits[0].episode_id == "b"
    assert result.hits[0].qk_score == pytest.approx(0.75)
    assert result.hits[0].ov_transport == pytest.approx(0.25)
    assert selector.last_report.reason == "student_exception"


@pytest.mark.parametrize("score_field", ("qk_score", "ov_transport"))
def test_nonfinite_teacher_scores_fail_open(score_field) -> None:
    hit = MemoryLinkHit("a", 0.5, 0.25, ())
    result = NestedMemoryInspection(
        (replace(hit, **{score_field: float("nan")}),),
        2,
        3,
        30,
        4,
    )
    teacher = _StaticTeacher(result)
    selector = _selector(
        _Student(error=RuntimeError("student failed")),
        teacher=teacher,
    )

    inspected = selector.inspect_nested("question", _groups(), top_k=1)

    assert [item.episode_id for item in inspected.hits] == ["a", "b", "c"]
    assert selector.last_report.decision == "fail_open"
    assert selector.last_report.requires_raw_fallback


@pytest.mark.parametrize(
    ("counter_name", "counter_value"),
    (
        ("passes", 0),
        ("max_workspace_candidates", 4),
        ("max_workspace_tokens", -1),
        ("total_candidate_inspections", 2),
        ("total_candidate_inspections", 7),
    ),
)
def test_insane_teacher_counters_fail_open(counter_name, counter_value) -> None:
    valid = NestedMemoryInspection(
        (MemoryLinkHit("a", 0.5, 0.25, ()),),
        2,
        3,
        30,
        4,
    )
    teacher = _StaticTeacher(replace(valid, **{counter_name: counter_value}))
    selector = _selector(
        _Student(error=RuntimeError("student failed")),
        teacher=teacher,
    )

    inspected = selector.inspect_nested("question", _groups(), top_k=1)

    assert [item.episode_id for item in inspected.hits] == ["a", "b", "c"]
    assert selector.last_report.reason == "teacher_failed:ValueError"


def test_teacher_cannot_return_more_than_effective_top_k() -> None:
    result = NestedMemoryInspection(
        (
            MemoryLinkHit("a", 0.5, 0.25, ()),
            MemoryLinkHit("b", 0.4, 0.20, ()),
        ),
        2,
        3,
        30,
        4,
    )
    teacher = _StaticTeacher(result)
    selector = _selector(
        _Student(error=RuntimeError("student failed")),
        teacher=teacher,
    )

    inspected = selector.inspect_nested("question", _groups(), top_k=1)

    assert [item.episode_id for item in inspected.hits] == ["a", "b", "c"]
    assert teacher.calls[0][1] == _groups()
    assert teacher.calls[0][2]["top_k"] == 1
    assert selector.last_report.decision == "fail_open"
