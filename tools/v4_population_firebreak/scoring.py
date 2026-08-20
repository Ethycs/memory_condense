"""Export and load one analysis-only scorer label behind the v4 firebreak.

The existing analysis reconstruction may decode designated development and
validation records to verify their locked projections.  This module emits and
returns labels for exactly one selected analysis ordinal.  Confirmation record
histories, question text, answers, and evidence labels never enter the JSON
decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analysis import (
    AnalysisPopulation,
    _RecordMetadata,
    _load_analysis_population,
    _partition_metadata,
    _scan_dataset_metadata,
)
from .canonical import (
    FirebreakError,
    FileSnapshot,
    assert_snapshot_unchanged,
    bytes_sha256,
    canonical_json_bytes,
    canonical_sha256,
    exact_keys,
    package_sha256,
    parse_json_bytes,
    publish_no_clobber,
    read_snapshot,
    require_int,
    require_list,
    require_mapping,
    require_sha256,
    require_text,
)
from .population import _answer_text, _parse_record, _string_list
from .verifier import ExpectedPopulationLock, PRODUCTION_LOCK


SCORING_LABEL_FORMAT = "memory-condense-v4-analysis-scoring-label-v1"
SCORING_LABEL_EXPORT_RECEIPT_FORMAT = (
    "memory-condense-v4-analysis-scoring-label-export-receipt-v1"
)

_TOP_LEVEL_KEYS = {
    "format",
    "dataset_sha256",
    "split_manifest_sha256",
    "analysis_ordered_question_ids_sha256",
    "analysis_sample_count",
    "sample_ordinal",
    "sample_id_sha256",
    "raw_record_sha256",
    "raw_record_span_sha256",
    "label_record_sha256",
    "label",
}
_LABEL_KEYS = {
    "question_id",
    "question_id_sha256",
    "question_text_sha256",
    "question_probe_sha256",
    "gold_answer",
    "gold_answer_sha256",
    "evidence_source_ids",
    "evidence_source_ids_sha256",
}


@dataclass(frozen=True, slots=True)
class AnalysisScoringLabel:
    """Immutable gold for exactly one locked analysis ordinal."""

    file_sha256: str
    label_record_sha256: str
    dataset_sha256: str
    split_manifest_sha256: str
    analysis_ordered_question_ids_sha256: str
    analysis_sample_count: int
    sample_ordinal: int
    sample_id_sha256: str
    raw_record_sha256: str
    raw_record_span_sha256: str
    question_id: str
    question_id_sha256: str
    question_text_sha256: str
    question_probe_sha256: str
    gold_answer: str
    gold_answer_sha256: str
    evidence_source_ids: tuple[str, ...]
    evidence_source_ids_sha256: str


def export_analysis_scoring_label(
    *,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    output_path: str | Path,
    sample_ordinal: int,
    expected_question_probe_sha256: str,
    expected: ExpectedPopulationLock = PRODUCTION_LOCK,
) -> dict[str, Any]:
    """Publish canonical scorer gold for one locked analysis ordinal.

    ``expected_question_probe_sha256`` must come from the already-frozen,
    gold-blind retrieval input.  This prevents an evaluator from selecting a
    different record or question after observing labels.
    """

    ordinal = require_int(sample_ordinal, "analysis sample ordinal")
    expected_probe = require_sha256(
        expected_question_probe_sha256,
        "expected question probe SHA-256",
    )
    dataset, split, population = _load_analysis_population(
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        expected=expected,
    )
    metadata = _analysis_metadata(dataset, population)
    samples = population.role_samples("analysis")
    if ordinal >= len(samples):
        raise FirebreakError("analysis sample ordinal is outside the locked pool")
    selected_metadata = metadata[ordinal]
    selected_sample = samples[ordinal]
    if selected_metadata.sample_id != selected_sample.sample_id:
        raise FirebreakError("analysis byte span differs from the locked order")

    raw_span = dataset.payload[selected_metadata.start : selected_metadata.end]
    record = require_mapping(
        parse_json_bytes(raw_span, "selected analysis scorer record"),
        "selected analysis scorer record",
    )
    projected = _parse_record(record, selected_metadata.index)
    if projected is None:
        raise FirebreakError("selected analysis scorer record is invalid")
    if (
        projected.sample_id != selected_sample.sample_id
        or projected.normalized_sha256 != selected_sample.normalized_sha256
        or projected.raw_record_sha256 != selected_sample.raw_record_sha256
        or projected.treatment_projection != selected_sample.treatment_projection
    ):
        raise FirebreakError("selected scorer record differs from the locked sample")

    question_projection = require_mapping(
        require_list(
            projected.treatment_projection["questions"],
            "selected analysis questions",
        )[0],
        "selected analysis question",
    )
    question_id = require_text(
        question_projection["question_id"],
        "selected analysis question ID",
    ).strip()
    sample_id = selected_sample.sample_id.strip()
    if not question_id or question_id != sample_id:
        raise FirebreakError("selected scoring coordinates are not canonical")
    question_text = require_text(
        question_projection["question"],
        "selected analysis question text",
    )
    question_date = question_projection["question_date"]
    prompt_question = (
        question_text
        if question_date is None
        else f"[Question asked at {question_date}]\n{question_text}"
    )
    probe_sha256 = canonical_sha256(
        {
            "question_id": question_id,
            "retrieval_query": question_text,
            "prompt_question": prompt_question,
        }
    )
    if probe_sha256 != expected_probe:
        raise FirebreakError("selected scorer record belongs to another frozen probe")

    gold_answer = _answer_text(record.get("answer"))
    if not gold_answer:
        raise FirebreakError("selected analysis scorer record has no gold answer")
    evidence_source_ids = list(
        dict.fromkeys(
            source.strip()
            for source in _string_list(record.get("answer_session_ids"))
            if source.strip()
        )
    )
    label = {
        "question_id": question_id,
        "question_id_sha256": canonical_sha256({"question_id": question_id}),
        "question_text_sha256": bytes_sha256(question_text.encode("utf-8")),
        "question_probe_sha256": probe_sha256,
        "gold_answer": gold_answer,
        "gold_answer_sha256": bytes_sha256(gold_answer.encode("utf-8")),
        "evidence_source_ids": evidence_source_ids,
        "evidence_source_ids_sha256": canonical_sha256(evidence_source_ids),
    }
    value = {
        "format": SCORING_LABEL_FORMAT,
        "dataset_sha256": population.dataset_sha256,
        "split_manifest_sha256": population.split_manifest_sha256,
        "analysis_ordered_question_ids_sha256": population.role_ids_sha256(
            "analysis"
        ),
        "analysis_sample_count": len(samples),
        "sample_ordinal": ordinal,
        "sample_id_sha256": canonical_sha256({"sample_id": sample_id}),
        "raw_record_sha256": projected.raw_record_sha256,
        "raw_record_span_sha256": bytes_sha256(raw_span),
        "label_record_sha256": canonical_sha256(label),
        "label": label,
    }
    payload = canonical_json_bytes(value) + b"\n"
    publish_no_clobber(output_path, payload)
    artifact = read_snapshot(output_path, "analysis scoring label")
    loaded = _decode_analysis_scoring_label(
        artifact,
        expected_file_sha256=artifact.sha256,
        expected_label_record_sha256=value["label_record_sha256"],
        expected_dataset_sha256=value["dataset_sha256"],
        expected_split_manifest_sha256=value["split_manifest_sha256"],
        expected_analysis_ordered_question_ids_sha256=value[
            "analysis_ordered_question_ids_sha256"
        ],
        expected_analysis_sample_count=value["analysis_sample_count"],
        expected_sample_ordinal=value["sample_ordinal"],
        expected_sample_id_sha256=value["sample_id_sha256"],
        expected_question_id_sha256=label["question_id_sha256"],
        expected_question_text_sha256=label["question_text_sha256"],
        expected_question_probe_sha256=label["question_probe_sha256"],
        expected_raw_record_sha256=value["raw_record_sha256"],
        expected_raw_record_span_sha256=value["raw_record_span_sha256"],
    )
    assert_snapshot_unchanged(dataset, "dataset")
    assert_snapshot_unchanged(split, "split manifest")
    assert_snapshot_unchanged(artifact, "analysis scoring label")
    return _export_receipt(population, loaded, artifact.size)


def load_analysis_scoring_label(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_label_record_sha256: str,
    expected_dataset_sha256: str,
    expected_split_manifest_sha256: str,
    expected_analysis_ordered_question_ids_sha256: str,
    expected_analysis_sample_count: int,
    expected_sample_ordinal: int,
    expected_sample_id_sha256: str,
    expected_question_id_sha256: str,
    expected_question_text_sha256: str,
    expected_question_probe_sha256: str,
    expected_raw_record_sha256: str,
    expected_raw_record_span_sha256: str,
) -> AnalysisScoringLabel:
    """Load one externally pinned label artifact for scoring only."""

    snapshot = read_snapshot(path, "analysis scoring label")
    result = _decode_analysis_scoring_label(
        snapshot,
        expected_file_sha256=expected_file_sha256,
        expected_label_record_sha256=expected_label_record_sha256,
        expected_dataset_sha256=expected_dataset_sha256,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_analysis_ordered_question_ids_sha256=(
            expected_analysis_ordered_question_ids_sha256
        ),
        expected_analysis_sample_count=expected_analysis_sample_count,
        expected_sample_ordinal=expected_sample_ordinal,
        expected_sample_id_sha256=expected_sample_id_sha256,
        expected_question_id_sha256=expected_question_id_sha256,
        expected_question_text_sha256=expected_question_text_sha256,
        expected_question_probe_sha256=expected_question_probe_sha256,
        expected_raw_record_sha256=expected_raw_record_sha256,
        expected_raw_record_span_sha256=expected_raw_record_span_sha256,
    )
    assert_snapshot_unchanged(snapshot, "analysis scoring label")
    return result


def _analysis_metadata(
    dataset: FileSnapshot,
    population: AnalysisPopulation,
) -> tuple[_RecordMetadata, ...]:
    records = _scan_dataset_metadata(dataset.payload)
    selected = _partition_metadata(
        records,
        {
            "development": len(population.partitions["development"].samples),
            "validation": len(population.partitions["validation"].samples),
            "confirmation": population.confirmation_count,
        },
        population.split_salt,
    )
    analysis = tuple(selected["development"] + selected["validation"])
    samples = population.role_samples("analysis")
    if len(analysis) != len(samples) or tuple(
        item.sample_id for item in analysis
    ) != tuple(sample.sample_id for sample in samples):
        raise FirebreakError("analysis byte spans differ from the locked order")
    if canonical_sha256([item.sample_id for item in analysis]) != (
        population.role_ids_sha256("analysis")
    ):
        raise FirebreakError("analysis byte spans do not bind the ordered pool")
    return analysis


def _decode_analysis_scoring_label(
    snapshot: FileSnapshot,
    *,
    expected_file_sha256: str,
    expected_label_record_sha256: str,
    expected_dataset_sha256: str,
    expected_split_manifest_sha256: str,
    expected_analysis_ordered_question_ids_sha256: str,
    expected_analysis_sample_count: int,
    expected_sample_ordinal: int,
    expected_sample_id_sha256: str,
    expected_question_id_sha256: str,
    expected_question_text_sha256: str,
    expected_question_probe_sha256: str,
    expected_raw_record_sha256: str,
    expected_raw_record_span_sha256: str,
) -> AnalysisScoringLabel:
    expected_file = require_sha256(expected_file_sha256, "expected file SHA-256")
    expected_label = require_sha256(
        expected_label_record_sha256,
        "expected label record SHA-256",
    )
    expected_dataset = require_sha256(
        expected_dataset_sha256,
        "expected dataset SHA-256",
    )
    expected_split = require_sha256(
        expected_split_manifest_sha256,
        "expected split SHA-256",
    )
    expected_order = require_sha256(
        expected_analysis_ordered_question_ids_sha256,
        "expected analysis order SHA-256",
    )
    expected_count = require_int(
        expected_analysis_sample_count,
        "expected analysis sample count",
        minimum=1,
    )
    expected_ordinal = require_int(
        expected_sample_ordinal,
        "expected analysis sample ordinal",
    )
    if expected_ordinal >= expected_count:
        raise FirebreakError("expected sample ordinal is outside the analysis pool")
    expected_sample = require_sha256(
        expected_sample_id_sha256,
        "expected sample ID SHA-256",
    )
    expected_question_id = require_sha256(
        expected_question_id_sha256,
        "expected question ID SHA-256",
    )
    expected_question_text = require_sha256(
        expected_question_text_sha256,
        "expected question text SHA-256",
    )
    expected_probe = require_sha256(
        expected_question_probe_sha256,
        "expected question probe SHA-256",
    )
    expected_raw = require_sha256(
        expected_raw_record_sha256,
        "expected raw record SHA-256",
    )
    expected_raw_span = require_sha256(
        expected_raw_record_span_sha256,
        "expected raw record span SHA-256",
    )
    if snapshot.sha256 != expected_file:
        raise FirebreakError("analysis scoring label differs from its receipt")

    value = require_mapping(
        parse_json_bytes(snapshot.payload, "analysis scoring label"),
        "analysis scoring label",
    )
    if snapshot.payload != canonical_json_bytes(value) + b"\n":
        raise FirebreakError("analysis scoring label is not canonical JSON")
    exact_keys(value, _TOP_LEVEL_KEYS, "analysis scoring label")
    if require_text(value["format"], "analysis scoring label format") != (
        SCORING_LABEL_FORMAT
    ):
        raise FirebreakError("unsupported analysis scoring-label format")
    bindings = (
        (
            require_sha256(value["dataset_sha256"], "label dataset SHA-256"),
            expected_dataset,
            "another dataset",
        ),
        (
            require_sha256(value["split_manifest_sha256"], "label split SHA-256"),
            expected_split,
            "another split",
        ),
        (
            require_sha256(
                value["analysis_ordered_question_ids_sha256"],
                "label analysis order SHA-256",
            ),
            expected_order,
            "another analysis order",
        ),
        (
            require_sha256(value["sample_id_sha256"], "label sample ID SHA-256"),
            expected_sample,
            "another sample",
        ),
        (
            require_sha256(value["raw_record_sha256"], "label raw record SHA-256"),
            expected_raw,
            "another raw record",
        ),
        (
            require_sha256(
                value["raw_record_span_sha256"],
                "label raw record span SHA-256",
            ),
            expected_raw_span,
            "another raw record span",
        ),
    )
    for actual, expected_value, mismatch in bindings:
        if actual != expected_value:
            raise FirebreakError(f"analysis scoring label binds {mismatch}")
    if require_int(
        value["analysis_sample_count"],
        "label analysis sample count",
        minimum=1,
    ) != expected_count:
        raise FirebreakError("analysis scoring label binds another population size")
    if require_int(value["sample_ordinal"], "label sample ordinal") != (
        expected_ordinal
    ):
        raise FirebreakError("analysis scoring label binds another sample ordinal")

    label = require_mapping(value["label"], "analysis scoring label record")
    exact_keys(label, _LABEL_KEYS, "analysis scoring label record")
    label_record_sha256 = require_sha256(
        value["label_record_sha256"],
        "label record SHA-256",
    )
    if label_record_sha256 != expected_label or label_record_sha256 != (
        canonical_sha256(label)
    ):
        raise FirebreakError("analysis scoring label record identity differs")
    question_id = require_text(label["question_id"], "scoring question ID")
    if question_id != question_id.strip():
        raise FirebreakError("scoring question ID is not canonical")
    question_id_sha256 = require_sha256(
        label["question_id_sha256"],
        "scoring question ID SHA-256",
    )
    if question_id_sha256 != expected_question_id or question_id_sha256 != (
        canonical_sha256({"question_id": question_id})
    ):
        raise FirebreakError("analysis scoring label binds another question ID")
    if canonical_sha256({"sample_id": question_id}) != expected_sample:
        raise FirebreakError("analysis scoring question belongs to another sample")
    question_text_sha256 = require_sha256(
        label["question_text_sha256"],
        "scoring question text SHA-256",
    )
    if question_text_sha256 != expected_question_text:
        raise FirebreakError("analysis scoring label binds another question text")
    question_probe_sha256 = require_sha256(
        label["question_probe_sha256"],
        "scoring question probe SHA-256",
    )
    if question_probe_sha256 != expected_probe:
        raise FirebreakError("analysis scoring label binds another frozen probe")
    gold_answer = require_text(label["gold_answer"], "scoring gold answer")
    gold_answer_sha256 = require_sha256(
        label["gold_answer_sha256"],
        "scoring gold answer SHA-256",
    )
    if gold_answer_sha256 != bytes_sha256(gold_answer.encode("utf-8")):
        raise FirebreakError("analysis scoring gold-answer identity differs")
    raw_sources = require_list(
        label["evidence_source_ids"],
        "scoring evidence source IDs",
    )
    if any(
        not isinstance(source, str)
        or not source
        or source != source.strip()
        for source in raw_sources
    ) or len(raw_sources) != len(set(raw_sources)):
        raise FirebreakError("scoring evidence source IDs are not canonical")
    evidence_source_ids_sha256 = require_sha256(
        label["evidence_source_ids_sha256"],
        "scoring evidence source IDs SHA-256",
    )
    if evidence_source_ids_sha256 != canonical_sha256(raw_sources):
        raise FirebreakError("analysis scoring evidence-source identity differs")
    return AnalysisScoringLabel(
        file_sha256=snapshot.sha256,
        label_record_sha256=label_record_sha256,
        dataset_sha256=expected_dataset,
        split_manifest_sha256=expected_split,
        analysis_ordered_question_ids_sha256=expected_order,
        analysis_sample_count=expected_count,
        sample_ordinal=expected_ordinal,
        sample_id_sha256=expected_sample,
        raw_record_sha256=expected_raw,
        raw_record_span_sha256=expected_raw_span,
        question_id=question_id,
        question_id_sha256=question_id_sha256,
        question_text_sha256=question_text_sha256,
        question_probe_sha256=question_probe_sha256,
        gold_answer=gold_answer,
        gold_answer_sha256=gold_answer_sha256,
        evidence_source_ids=tuple(raw_sources),
        evidence_source_ids_sha256=evidence_source_ids_sha256,
    )


def _export_receipt(
    population: AnalysisPopulation,
    label: AnalysisScoringLabel,
    file_bytes: int,
) -> dict[str, Any]:
    return {
        "format": SCORING_LABEL_EXPORT_RECEIPT_FORMAT,
        "status": "verified",
        "verifier_implementation_sha256": package_sha256(Path(__file__).parent),
        "artifact": {
            "file_sha256": label.file_sha256,
            "file_bytes": file_bytes,
            "label_record_sha256": label.label_record_sha256,
            "record_count": 1,
        },
        "population": {
            "dataset_sha256": label.dataset_sha256,
            "split_manifest_sha256": label.split_manifest_sha256,
            "analysis_ordered_question_ids_sha256": (
                label.analysis_ordered_question_ids_sha256
            ),
            "analysis_sample_count": label.analysis_sample_count,
        },
        "selection": {
            "sample_ordinal": label.sample_ordinal,
            "sample_id_sha256": label.sample_id_sha256,
            "question_id_sha256": label.question_id_sha256,
            "question_text_sha256": label.question_text_sha256,
            "question_probe_sha256": label.question_probe_sha256,
            "raw_record_sha256": label.raw_record_sha256,
            "raw_record_span_sha256": label.raw_record_span_sha256,
            "gold_answer_sha256": label.gold_answer_sha256,
            "evidence_source_ids_sha256": label.evidence_source_ids_sha256,
        },
        "confirmation_membership": {
            "count": population.confirmation_count,
            "history_decoded": False,
            "question_text_decoded": False,
            "gold_decoded": False,
            "content_emitted": False,
        },
        "firebreak": {
            "closed_single_record_schema": True,
            "selected_partition": "analysis",
            "selected_record_decoded": True,
            "confirmation_history_decoded": False,
            "confirmation_question_text_decoded": False,
            "confirmation_gold_decoded": False,
            "question_or_history_text_in_receipt": False,
            "gold_or_evidence_values_in_receipt": False,
            "verifier_dependency_class": "python_standard_library_only",
        },
    }


__all__ = [
    "AnalysisScoringLabel",
    "SCORING_LABEL_EXPORT_RECEIPT_FORMAT",
    "SCORING_LABEL_FORMAT",
    "export_analysis_scoring_label",
    "load_analysis_scoring_label",
]
