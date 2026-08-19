"""Closed-schema validation for label-free treatment inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import (
    FirebreakError,
    FileSnapshot,
    assert_snapshot_unchanged,
    canonical_sha256,
    exact_keys,
    parse_json_bytes,
    read_snapshot,
    require_list,
    require_mapping,
    require_sha256,
    require_text,
)
from .population import Population, PopulationSample


TREATMENT_INPUT_FORMAT = "memory-condense-evaluator-firebreak-treatment-input-v2"
_TOP_LEVEL_KEYS = {
    "format",
    "role",
    "dataset_sha256",
    "split_manifest_sha256",
    "ordered_question_ids_sha256",
    "samples",
}
_SAMPLE_KEYS = {
    "sample_id",
    "turns",
    "turn_source_ids",
    "turn_created_at",
    "questions",
}
_QUESTION_KEYS = {"question_id", "question", "question_date"}


@dataclass(frozen=True, slots=True)
class TreatmentInputReceipt:
    role: str
    file_sha256: str
    file_bytes: int
    sample_count: int
    ordered_question_ids_sha256: str
    sanitized_projection_sha256: str

    def json_value(self) -> dict[str, Any]:
        return {
            "file_sha256": self.file_sha256,
            "file_bytes": self.file_bytes,
            "sample_count": self.sample_count,
            "ordered_question_ids_sha256": self.ordered_question_ids_sha256,
            "sanitized_projection_sha256": self.sanitized_projection_sha256,
            "scorer_labels_present": False,
        }


@dataclass(frozen=True, slots=True)
class TreatmentQuestion:
    """One label-free query delivered to retrieval."""

    question_id: str
    question: str
    question_date: str | None

    @property
    def dated_question(self) -> str:
        if self.question_date is None:
            return self.question
        return f"[Question asked at {self.question_date}]\n{self.question}"


@dataclass(frozen=True, slots=True)
class TreatmentSample:
    """Immutable retrieval input with no scorer or oracle fields."""

    sample_id: str
    turns: tuple[tuple[str, str], ...]
    turn_source_ids: tuple[str | None, ...]
    turn_created_at: tuple[datetime | None, ...]
    questions: tuple[TreatmentQuestion, ...]


@dataclass(frozen=True, slots=True)
class AnalysisTreatmentInput:
    """Content-addressed analysis input safe to hand to retrieval."""

    file_sha256: str
    sanitized_projection_sha256: str
    dataset_sha256: str
    split_manifest_sha256: str
    ordered_question_ids_sha256: str
    samples: tuple[TreatmentSample, ...]


def validate_treatment_input(
    snapshot: FileSnapshot,
    population: Population,
) -> TreatmentInputReceipt:
    value = require_mapping(
        parse_json_bytes(snapshot.payload, "treatment input"),
        "treatment input",
    )
    exact_keys(value, _TOP_LEVEL_KEYS, "treatment input")
    if require_text(value["format"], "treatment input format") != TREATMENT_INPUT_FORMAT:
        raise FirebreakError("unsupported treatment-input format")
    role = require_text(value["role"], "treatment input role")
    expected_samples = population.role_samples(role)
    if (
        require_sha256(value["dataset_sha256"], "treatment dataset SHA-256")
        != population.dataset_sha256
    ):
        raise FirebreakError("treatment input does not bind the dataset")
    if (
        require_sha256(value["split_manifest_sha256"], "treatment split SHA-256")
        != population.split_manifest_sha256
    ):
        raise FirebreakError("treatment input does not bind the split manifest")
    expected_ids_sha = population.role_ids_sha256(role)
    if require_sha256(
        value["ordered_question_ids_sha256"],
        "treatment ordered question IDs SHA-256",
    ) != expected_ids_sha:
        raise FirebreakError("treatment input does not bind the ordered population")
    actual_samples = require_list(value["samples"], "treatment samples")
    if len(actual_samples) != len(expected_samples):
        raise FirebreakError("treatment input has the wrong population size")
    for index, (actual, expected) in enumerate(zip(actual_samples, expected_samples, strict=True)):
        _validate_sample(actual, expected, index)
    projection_sha = canonical_sha256(actual_samples)
    expected_projection_sha = canonical_sha256(
        [sample.treatment_projection for sample in expected_samples]
    )
    if projection_sha != expected_projection_sha:
        raise FirebreakError("treatment projection differs from the locked source")
    return TreatmentInputReceipt(
        role=role,
        file_sha256=snapshot.sha256,
        file_bytes=snapshot.size,
        sample_count=len(actual_samples),
        ordered_question_ids_sha256=expected_ids_sha,
        sanitized_projection_sha256=projection_sha,
    )


def _validate_sample(value: Any, expected: PopulationSample, index: int) -> None:
    sample = require_mapping(value, f"treatment sample {index}")
    exact_keys(sample, _SAMPLE_KEYS, f"treatment sample {index}")
    require_text(sample["sample_id"], f"treatment sample {index} ID")
    if sample["sample_id"] != expected.treatment_projection["sample_id"]:
        raise FirebreakError(f"treatment sample {index} is reordered or overlaps another partition")
    turns = require_list(sample["turns"], f"treatment sample {index} turns")
    for turn_index, turn in enumerate(turns):
        pair = require_list(turn, f"treatment sample {index} turn {turn_index}")
        if len(pair) != 2 or not all(isinstance(item, str) for item in pair):
            raise FirebreakError(f"treatment sample {index} has an invalid turn")
    sources = require_list(
        sample["turn_source_ids"], f"treatment sample {index} source IDs"
    )
    if len(sources) != len(turns) or any(
        source is not None and not isinstance(source, str) for source in sources
    ):
        raise FirebreakError(f"treatment sample {index} has invalid source coordinates")
    timestamps = require_list(
        sample["turn_created_at"], f"treatment sample {index} turn timestamps"
    )
    if len(timestamps) != len(turns):
        raise FirebreakError(f"treatment sample {index} has misaligned timestamps")
    for timestamp in timestamps:
        _parse_timestamp(timestamp, f"treatment sample {index} timestamp")
    questions = require_list(
        sample["questions"], f"treatment sample {index} questions"
    )
    if len(questions) != 1:
        raise FirebreakError(f"treatment sample {index} must contain one query")
    question = require_mapping(
        questions[0], f"treatment sample {index} query"
    )
    # This exact allowlist is the label firebreak. Gold answers, categories,
    # evidence labels, judge outputs, correctness, F1, and aliases cannot enter.
    exact_keys(question, _QUESTION_KEYS, f"treatment sample {index} query")
    require_text(question["question_id"], f"treatment sample {index} query ID")
    require_text(question["question"], f"treatment sample {index} query")
    if question["question_date"] is not None and not isinstance(
        question["question_date"], str
    ):
        raise FirebreakError(f"treatment sample {index} has an invalid query date")
    if sample != expected.treatment_projection:
        raise FirebreakError(f"treatment sample {index} was modified")


def load_analysis_treatment_input(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_sanitized_projection_sha256: str,
    expected_dataset_sha256: str,
    expected_split_manifest_sha256: str,
    expected_ordered_question_ids_sha256: str,
    expected_sample_count: int,
) -> AnalysisTreatmentInput:
    """Load one verified analysis artifact without access to benchmark gold.

    The caller pins the evaluator receipt fields.  The file is snapshotted
    once, its closed schema is rechecked, and only that exact payload is
    decoded into immutable retrieval objects.  No dataset, exposure audit, or
    scorer label is accepted by this consumer.
    """

    expected_file = require_sha256(expected_file_sha256, "expected file SHA-256")
    expected_projection = require_sha256(
        expected_sanitized_projection_sha256,
        "expected projection SHA-256",
    )
    expected_dataset = require_sha256(
        expected_dataset_sha256,
        "expected dataset SHA-256",
    )
    expected_split = require_sha256(
        expected_split_manifest_sha256,
        "expected split SHA-256",
    )
    expected_ids = require_sha256(
        expected_ordered_question_ids_sha256,
        "expected ordered ID SHA-256",
    )
    if (
        isinstance(expected_sample_count, bool)
        or not isinstance(expected_sample_count, int)
        or expected_sample_count < 1
    ):
        raise FirebreakError("expected sample count must be a positive integer")

    snapshot = read_snapshot(path, "analysis treatment input")
    if snapshot.sha256 != expected_file:
        raise FirebreakError("analysis treatment input differs from its receipt")
    value = require_mapping(
        parse_json_bytes(snapshot.payload, "analysis treatment input"),
        "analysis treatment input",
    )
    exact_keys(value, _TOP_LEVEL_KEYS, "analysis treatment input")
    if require_text(value["format"], "analysis treatment format") != (
        TREATMENT_INPUT_FORMAT
    ):
        raise FirebreakError("unsupported treatment-input format")
    if require_text(value["role"], "analysis treatment role") != "analysis":
        raise FirebreakError("retrieval consumer accepts analysis role only")
    if require_sha256(value["dataset_sha256"], "analysis dataset SHA-256") != (
        expected_dataset
    ):
        raise FirebreakError("analysis treatment input binds another dataset")
    if require_sha256(
        value["split_manifest_sha256"], "analysis split SHA-256"
    ) != expected_split:
        raise FirebreakError("analysis treatment input binds another split")
    if require_sha256(
        value["ordered_question_ids_sha256"], "analysis ordered ID SHA-256"
    ) != expected_ids:
        raise FirebreakError("analysis treatment input binds another population")

    raw_samples = require_list(value["samples"], "analysis treatment samples")
    if len(raw_samples) != expected_sample_count:
        raise FirebreakError("analysis treatment input has the wrong population size")
    if canonical_sha256(raw_samples) != expected_projection:
        raise FirebreakError("analysis treatment projection differs from its receipt")
    samples = tuple(
        _decode_treatment_sample(raw_sample, index)
        for index, raw_sample in enumerate(raw_samples)
    )
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise FirebreakError("analysis treatment input repeats a sample ID")
    if canonical_sha256(sample_ids) != expected_ids:
        raise FirebreakError("analysis treatment samples are reordered or overlap")
    assert_snapshot_unchanged(snapshot, "analysis treatment input")
    return AnalysisTreatmentInput(
        file_sha256=snapshot.sha256,
        sanitized_projection_sha256=expected_projection,
        dataset_sha256=expected_dataset,
        split_manifest_sha256=expected_split,
        ordered_question_ids_sha256=expected_ids,
        samples=samples,
    )


def _decode_treatment_sample(value: Any, index: int) -> TreatmentSample:
    sample = require_mapping(value, f"analysis treatment sample {index}")
    exact_keys(sample, _SAMPLE_KEYS, f"analysis treatment sample {index}")
    sample_id = require_text(sample["sample_id"], f"analysis sample {index} ID")
    raw_turns = require_list(sample["turns"], f"analysis sample {index} turns")
    turns: list[tuple[str, str]] = []
    for turn_index, raw_turn in enumerate(raw_turns):
        pair = require_list(
            raw_turn,
            f"analysis sample {index} turn {turn_index}",
        )
        if len(pair) != 2 or not all(isinstance(item, str) for item in pair):
            raise FirebreakError(f"analysis sample {index} has an invalid turn")
        role, text = pair
        if role not in {"user", "assistant", "system"} or not text:
            raise FirebreakError(f"analysis sample {index} has an invalid turn")
        turns.append((role, text))
    raw_sources = require_list(
        sample["turn_source_ids"], f"analysis sample {index} source IDs"
    )
    if len(raw_sources) != len(turns) or any(
        source is not None and (not isinstance(source, str) or not source)
        for source in raw_sources
    ):
        raise FirebreakError(f"analysis sample {index} has invalid source coordinates")
    raw_timestamps = require_list(
        sample["turn_created_at"], f"analysis sample {index} timestamps"
    )
    if len(raw_timestamps) != len(turns):
        raise FirebreakError(f"analysis sample {index} has misaligned timestamps")
    timestamps = tuple(
        _parse_timestamp(value, f"analysis sample {index} timestamp")
        for value in raw_timestamps
    )
    raw_questions = require_list(
        sample["questions"], f"analysis sample {index} questions"
    )
    if len(raw_questions) != 1:
        raise FirebreakError(f"analysis sample {index} must contain one query")
    raw_question = require_mapping(
        raw_questions[0], f"analysis sample {index} query"
    )
    exact_keys(raw_question, _QUESTION_KEYS, f"analysis sample {index} query")
    question_id = require_text(
        raw_question["question_id"], f"analysis sample {index} query ID"
    )
    if question_id != sample_id:
        raise FirebreakError(f"analysis sample {index} query ID differs from sample ID")
    question = require_text(
        raw_question["question"], f"analysis sample {index} query"
    )
    question_date = raw_question["question_date"]
    if question_date is not None and (
        not isinstance(question_date, str) or not question_date
    ):
        raise FirebreakError(f"analysis sample {index} has an invalid query date")
    return TreatmentSample(
        sample_id=sample_id,
        turns=tuple(turns),
        turn_source_ids=tuple(raw_sources),
        turn_created_at=timestamps,
        questions=(
            TreatmentQuestion(
                question_id=question_id,
                question=question,
                question_date=question_date,
            ),
        ),
    )


def _parse_timestamp(value: Any, label: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise FirebreakError(f"{label} must be an ISO UTC timestamp or null")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FirebreakError(f"{label} must be an ISO UTC timestamp or null") from exc
    if parsed.tzinfo is None:
        raise FirebreakError(f"{label} must include a UTC offset")
    normalized = parsed.astimezone(timezone.utc)
    canonical = normalized.isoformat().replace("+00:00", "Z")
    if canonical != value:
        raise FirebreakError(f"{label} is not canonical UTC")
    return normalized
