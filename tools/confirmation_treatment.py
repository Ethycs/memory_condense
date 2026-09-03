"""Label-free confirmation treatment loader with no evaluator capability.

This module accepts only the already-sanitized, content-addressed confirmation
artifact.  It deliberately contains no population reconstruction, split
selection, reference answer, scorer, or judge imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.confirmation_canonical import (
    FirebreakError,
    assert_snapshot_unchanged,
    canonical_sha256,
    exact_keys,
    parse_json_bytes,
    read_snapshot,
    require_int,
    require_list,
    require_mapping,
    require_sha256,
    require_text,
)

CONFIRMATION_TREATMENT_INPUT_FORMAT = (
    "memory-condense-v4-confirmation-treatment-input-v1"
)
_SAMPLE_KEYS = {
    "sample_id",
    "turns",
    "turn_source_ids",
    "turn_created_at",
    "questions",
}
_QUESTION_KEYS = {"question_id", "question", "question_date"}
_CONFIRMATION_TOP_LEVEL_KEYS = {
    "format",
    "role",
    "dataset_sha256",
    "split_manifest_sha256",
    "sample_count",
    "ordered_question_ids_sha256",
    "ordered_normalized_sample_bindings_sha256",
    "ordered_raw_record_bindings_sha256",
    "sanitized_projection_sha256",
    "samples",
}


@dataclass(frozen=True, slots=True)
class ConfirmationTreatmentStaticLock:
    dataset_sha256: str
    split_manifest_sha256: str
    sample_count: int
    ordered_question_ids_sha256: str
    ordered_normalized_sample_bindings_sha256: str
    ordered_raw_record_bindings_sha256: str


PRODUCTION_CONFIRMATION_TREATMENT_LOCK = ConfirmationTreatmentStaticLock(
    dataset_sha256=(
        "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
    ),
    split_manifest_sha256=(
        "8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4"
    ),
    sample_count=200,
    ordered_question_ids_sha256=(
        "6270b044792dbda79cd79a104ab6a519b2f81980c47522c19a196583d8c0d102"
    ),
    ordered_normalized_sample_bindings_sha256=(
        "cbabcc97cad2f945c397fd980ef3bb3fb65ba8403dbeadf38b1b8224bc4a066d"
    ),
    ordered_raw_record_bindings_sha256=(
        "cf86373d06725b26117e9ce96ce906a16d545d346a1d2888f200d425f7a27fd9"
    ),
)


@dataclass(frozen=True, slots=True)
class TreatmentQuestion:
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
    sample_id: str
    turns: tuple[tuple[str, str], ...]
    turn_source_ids: tuple[str | None, ...]
    turn_created_at: tuple[datetime | None, ...]
    questions: tuple[TreatmentQuestion, ...]


@dataclass(frozen=True, slots=True)
class ConfirmationTreatmentInput:
    file_sha256: str
    sanitized_projection_sha256: str
    dataset_sha256: str
    split_manifest_sha256: str
    ordered_question_ids_sha256: str
    ordered_normalized_sample_bindings_sha256: str
    ordered_raw_record_bindings_sha256: str
    samples: tuple[TreatmentSample, ...]

def load_confirmation_treatment_input(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_sanitized_projection_sha256: str,
) -> ConfirmationTreatmentInput:
    """Load the one production confirmation role without benchmark labels.

    The caller supplies only the content-addressed export receipt.  Dataset,
    split, count, order, and source-binding roots are compiled into this
    boundary, so the caller cannot select a role or population.  The file
    digest is checked before the payload reaches the JSON decoder.
    """

    return _load_confirmation_treatment_input(
        path,
        expected_file_sha256=expected_file_sha256,
        expected_sanitized_projection_sha256=(
            expected_sanitized_projection_sha256
        ),
        static_lock=PRODUCTION_CONFIRMATION_TREATMENT_LOCK,
    )


def _load_confirmation_treatment_input(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_sanitized_projection_sha256: str,
    static_lock: ConfirmationTreatmentStaticLock,
) -> ConfirmationTreatmentInput:
    """Testable implementation; production callers use the fixed wrapper."""

    expected_file = require_sha256(expected_file_sha256, "expected file SHA-256")
    expected_projection = require_sha256(
        expected_sanitized_projection_sha256,
        "expected projection SHA-256",
    )
    _validate_confirmation_static_lock(static_lock)

    snapshot = read_snapshot(path, "confirmation treatment input")
    if snapshot.sha256 != expected_file:
        raise FirebreakError("confirmation treatment input differs from its receipt")

    # The untrusted payload is decoded only after its exact byte identity is
    # known.  This ordering is a security invariant, not just an optimization.
    value = require_mapping(
        parse_json_bytes(snapshot.payload, "confirmation treatment input"),
        "confirmation treatment input",
    )
    exact_keys(value, _CONFIRMATION_TOP_LEVEL_KEYS, "confirmation treatment input")
    if require_text(value["format"], "confirmation treatment format") != (
        CONFIRMATION_TREATMENT_INPUT_FORMAT
    ):
        raise FirebreakError("unsupported confirmation treatment-input format")
    if require_text(value["role"], "confirmation treatment role") != "confirmation":
        raise FirebreakError("confirmation treatment role is fixed")
    if require_sha256(
        value["dataset_sha256"], "confirmation dataset SHA-256"
    ) != static_lock.dataset_sha256:
        raise FirebreakError("confirmation treatment binds another dataset")
    if require_sha256(
        value["split_manifest_sha256"], "confirmation split SHA-256"
    ) != static_lock.split_manifest_sha256:
        raise FirebreakError("confirmation treatment binds another split")
    if require_int(
        value["sample_count"], "confirmation sample count", minimum=1
    ) != static_lock.sample_count:
        raise FirebreakError("confirmation treatment has the wrong population size")
    if require_sha256(
        value["ordered_question_ids_sha256"],
        "confirmation ordered ID SHA-256",
    ) != static_lock.ordered_question_ids_sha256:
        raise FirebreakError("confirmation treatment binds another population order")
    if require_sha256(
        value["ordered_normalized_sample_bindings_sha256"],
        "confirmation normalized-binding SHA-256",
    ) != static_lock.ordered_normalized_sample_bindings_sha256:
        raise FirebreakError("confirmation normalized source bindings differ from the lock")
    if require_sha256(
        value["ordered_raw_record_bindings_sha256"],
        "confirmation raw-binding SHA-256",
    ) != static_lock.ordered_raw_record_bindings_sha256:
        raise FirebreakError("confirmation raw source bindings differ from the lock")
    if require_sha256(
        value["sanitized_projection_sha256"],
        "confirmation projection SHA-256",
    ) != expected_projection:
        raise FirebreakError("confirmation projection differs from its receipt")

    raw_samples = require_list(value["samples"], "confirmation treatment samples")
    if len(raw_samples) != static_lock.sample_count:
        raise FirebreakError("confirmation treatment has the wrong population size")
    if canonical_sha256(raw_samples) != expected_projection:
        raise FirebreakError("confirmation treatment projection differs from its receipt")
    samples = tuple(
        _decode_treatment_sample(raw_sample, index, role_label="confirmation")
        for index, raw_sample in enumerate(raw_samples)
    )
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise FirebreakError("confirmation treatment repeats a sample ID")
    if canonical_sha256(sample_ids) != static_lock.ordered_question_ids_sha256:
        raise FirebreakError("confirmation treatment samples are reordered or overlap")
    assert_snapshot_unchanged(snapshot, "confirmation treatment input")
    return ConfirmationTreatmentInput(
        file_sha256=snapshot.sha256,
        sanitized_projection_sha256=expected_projection,
        dataset_sha256=static_lock.dataset_sha256,
        split_manifest_sha256=static_lock.split_manifest_sha256,
        ordered_question_ids_sha256=static_lock.ordered_question_ids_sha256,
        ordered_normalized_sample_bindings_sha256=(
            static_lock.ordered_normalized_sample_bindings_sha256
        ),
        ordered_raw_record_bindings_sha256=(
            static_lock.ordered_raw_record_bindings_sha256
        ),
        samples=samples,
    )


def _validate_confirmation_static_lock(
    static_lock: ConfirmationTreatmentStaticLock,
) -> None:
    require_sha256(static_lock.dataset_sha256, "locked confirmation dataset SHA-256")
    require_sha256(
        static_lock.split_manifest_sha256,
        "locked confirmation split SHA-256",
    )
    require_sha256(
        static_lock.ordered_question_ids_sha256,
        "locked confirmation ordered ID SHA-256",
    )
    require_sha256(
        static_lock.ordered_normalized_sample_bindings_sha256,
        "locked confirmation normalized-binding SHA-256",
    )
    require_sha256(
        static_lock.ordered_raw_record_bindings_sha256,
        "locked confirmation raw-binding SHA-256",
    )
    if (
        isinstance(static_lock.sample_count, bool)
        or not isinstance(static_lock.sample_count, int)
        or static_lock.sample_count < 1
    ):
        raise FirebreakError("locked confirmation sample count must be positive")


def _decode_treatment_sample(
    value: Any,
    index: int,
    *,
    role_label: str,
) -> TreatmentSample:
    sample = require_mapping(value, f"{role_label} treatment sample {index}")
    exact_keys(sample, _SAMPLE_KEYS, f"{role_label} treatment sample {index}")
    sample_id = require_text(sample["sample_id"], f"{role_label} sample {index} ID")
    raw_turns = require_list(sample["turns"], f"{role_label} sample {index} turns")
    turns: list[tuple[str, str]] = []
    for turn_index, raw_turn in enumerate(raw_turns):
        pair = require_list(
            raw_turn,
            f"{role_label} sample {index} turn {turn_index}",
        )
        if len(pair) != 2 or not all(isinstance(item, str) for item in pair):
            raise FirebreakError(f"{role_label} sample {index} has an invalid turn")
        role, text = pair
        if role not in {"user", "assistant", "system"} or not text:
            raise FirebreakError(f"{role_label} sample {index} has an invalid turn")
        turns.append((role, text))
    raw_sources = require_list(
        sample["turn_source_ids"], f"{role_label} sample {index} source IDs"
    )
    if len(raw_sources) != len(turns) or any(
        source is not None and (not isinstance(source, str) or not source)
        for source in raw_sources
    ):
        raise FirebreakError(f"{role_label} sample {index} has invalid source coordinates")
    raw_timestamps = require_list(
        sample["turn_created_at"], f"{role_label} sample {index} timestamps"
    )
    if len(raw_timestamps) != len(turns):
        raise FirebreakError(f"{role_label} sample {index} has misaligned timestamps")
    timestamps = tuple(
        _parse_timestamp(value, f"{role_label} sample {index} timestamp")
        for value in raw_timestamps
    )
    raw_questions = require_list(
        sample["questions"], f"{role_label} sample {index} questions"
    )
    if len(raw_questions) != 1:
        raise FirebreakError(f"{role_label} sample {index} must contain one query")
    raw_question = require_mapping(
        raw_questions[0], f"{role_label} sample {index} query"
    )
    exact_keys(raw_question, _QUESTION_KEYS, f"{role_label} sample {index} query")
    question_id = require_text(
        raw_question["question_id"], f"{role_label} sample {index} query ID"
    )
    if question_id != sample_id:
        raise FirebreakError(f"{role_label} sample {index} query ID differs from sample ID")
    question = require_text(
        raw_question["question"], f"{role_label} sample {index} query"
    )
    question_date = raw_question["question_date"]
    if question_date is not None and (
        not isinstance(question_date, str) or not question_date
    ):
        raise FirebreakError(f"{role_label} sample {index} has an invalid query date")
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


__all__ = [
    "CONFIRMATION_TREATMENT_INPUT_FORMAT",
    "ConfirmationTreatmentInput",
    "ConfirmationTreatmentStaticLock",
    "PRODUCTION_CONFIRMATION_TREATMENT_LOCK",
    "TreatmentQuestion",
    "TreatmentSample",
    "_decode_treatment_sample",
    "load_confirmation_treatment_input",
]
