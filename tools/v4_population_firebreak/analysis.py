"""Analysis-only sanitizer/export path that never decodes confirmation labels."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import (
    FirebreakError,
    FileSnapshot,
    assert_snapshot_unchanged,
    canonical_json_bytes,
    canonical_sha256,
    exact_keys,
    package_sha256,
    parse_json_bytes,
    publish_no_clobber,
    read_snapshot,
    require_int,
    require_mapping,
    require_text,
)
from .population import Partition, PopulationSample, _parse_record
from .treatment import TREATMENT_INPUT_FORMAT, validate_treatment_input
from .verifier import ExpectedPopulationLock, PRODUCTION_LOCK


ANALYSIS_EXPORT_RECEIPT_FORMAT = (
    "memory-condense-v4-analysis-treatment-export-receipt-v1"
)
_NUMBER_RE = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_HEX = frozenset(b"0123456789abcdefABCDEF")


@dataclass(frozen=True, slots=True)
class _RecordMetadata:
    index: int
    sample_id: str
    category: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class AnalysisPopulation:
    """Development/validation content plus confirmation membership only."""

    dataset_sha256: str
    dataset_bytes: int
    split_manifest_sha256: str
    split_format: str
    split_algorithm: str
    split_salt: str
    partitions: dict[str, Partition]
    confirmation_count: int
    confirmation_ordered_ids_sha256: str
    confirmation_category_counts: dict[str, int]

    def role_samples(self, role: str) -> tuple[PopulationSample, ...]:
        if role != "analysis":
            raise FirebreakError("analysis-only population cannot expose confirmation")
        return (
            self.partitions["development"].samples
            + self.partitions["validation"].samples
        )

    def role_ids_sha256(self, role: str) -> str:
        return canonical_sha256(
            [sample.sample_id for sample in self.role_samples(role)]
        )


def export_analysis_treatment_input(
    *,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    output_path: str | Path,
    expected: ExpectedPopulationLock = PRODUCTION_LOCK,
) -> dict[str, Any]:
    """Publish a canonical, label-free analysis artifact without clobbering.

    Confirmation records are scanned only for the ID/category metadata needed
    to reproduce locked split membership. Their questions, histories, answer
    labels, and evidence labels are never decoded into Python values.
    """

    dataset, split, population = _load_analysis_population(
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        expected=expected,
    )
    value = _treatment_value(population)
    payload = canonical_json_bytes(value) + b"\n"
    publish_no_clobber(output_path, payload)
    treatment = read_snapshot(output_path, "exported analysis treatment input")
    treatment_receipt = validate_treatment_input(
        treatment,
        population,  # type: ignore[arg-type] - closed role protocol
    )
    assert_snapshot_unchanged(dataset, "dataset")
    assert_snapshot_unchanged(split, "split manifest")
    assert_snapshot_unchanged(treatment, "exported analysis treatment input")
    return _receipt(population, treatment_receipt.json_value())


def verify_analysis_treatment_input(
    *,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    treatment_input_path: str | Path,
    expected: ExpectedPopulationLock = PRODUCTION_LOCK,
) -> dict[str, Any]:
    """Verify an analysis artifact while leaving confirmation content closed."""

    dataset, split, population = _load_analysis_population(
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        expected=expected,
    )
    treatment = read_snapshot(treatment_input_path, "analysis treatment input")
    treatment_receipt = validate_treatment_input(
        treatment,
        population,  # type: ignore[arg-type] - closed role protocol
    )
    assert_snapshot_unchanged(dataset, "dataset")
    assert_snapshot_unchanged(split, "split manifest")
    assert_snapshot_unchanged(treatment, "analysis treatment input")
    return _receipt(population, treatment_receipt.json_value())


def _load_analysis_population(
    *,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    expected: ExpectedPopulationLock,
) -> tuple[FileSnapshot, FileSnapshot, AnalysisPopulation]:
    dataset = read_snapshot(dataset_path, "dataset")
    split = read_snapshot(split_manifest_path, "split manifest")
    if dataset.sha256 != expected.dataset_sha256 or dataset.size != expected.dataset_bytes:
        raise FirebreakError("dataset identity differs from the locked population")
    if split.sha256 != expected.split_manifest_sha256:
        raise FirebreakError("split-manifest identity differs from the lock")
    population = _reconstruct_analysis_population(dataset, split)
    _verify_analysis_population(population, expected)
    return dataset, split, population


def _reconstruct_analysis_population(
    dataset: FileSnapshot,
    split_manifest: FileSnapshot,
) -> AnalysisPopulation:
    manifest = require_mapping(
        parse_json_bytes(split_manifest.payload, "split manifest"),
        "split manifest",
    )
    exact_keys(
        manifest,
        {"format", "dataset_sha256", "salt", "algorithm", "splits"},
        "split manifest",
    )
    if require_text(manifest["dataset_sha256"], "split dataset SHA-256") != (
        dataset.sha256
    ):
        raise FirebreakError("split manifest does not bind the dataset snapshot")
    split_format = require_text(manifest["format"], "split format")
    split_algorithm = require_text(manifest["algorithm"], "split algorithm")
    split_salt = require_text(manifest["salt"], "split salt")
    if split_format != "memory-condense-locked-benchmark-split-v1":
        raise FirebreakError("unsupported split manifest format")
    if split_algorithm != "stratified-largest-remainder-v1":
        raise FirebreakError("unsupported split algorithm")
    raw_counts = require_mapping(manifest["splits"], "split counts")
    split_names = list(raw_counts)
    if split_names != ["development", "validation", "confirmation"]:
        raise FirebreakError("split order or names differ from the locked protocol")
    counts = {
        name: require_int(value, f"split count {name}", minimum=1)
        for name, value in raw_counts.items()
    }

    metadata = _scan_dataset_metadata(dataset.payload)
    if len(metadata) != sum(counts.values()):
        raise FirebreakError("split counts do not cover the dataset record population")
    ids = [item.sample_id for item in metadata]
    if len(ids) != len(set(ids)):
        raise FirebreakError("benchmark sample IDs are not unique")
    selected = _partition_metadata(metadata, counts, split_salt)

    content_partitions: dict[str, Partition] = {}
    for name in ("development", "validation"):
        samples: list[PopulationSample] = []
        for item in selected[name]:
            # Only selected analysis spans enter the strict JSON decoder.
            record = parse_json_bytes(
                dataset.payload[item.start : item.end],
                f"{name} record",
            )
            sample = _parse_record(record, item.index)
            if sample is None:
                raise FirebreakError(f"{name} contains an invalid benchmark record")
            if sample.sample_id != item.sample_id or sample.category != item.category:
                raise FirebreakError(f"{name} metadata changed during projection")
            samples.append(sample)
        content_partitions[name] = Partition(name=name, samples=tuple(samples))

    confirmation = selected["confirmation"]
    return AnalysisPopulation(
        dataset_sha256=dataset.sha256,
        dataset_bytes=dataset.size,
        split_manifest_sha256=split_manifest.sha256,
        split_format=split_format,
        split_algorithm=split_algorithm,
        split_salt=split_salt,
        partitions=content_partitions,
        confirmation_count=len(confirmation),
        confirmation_ordered_ids_sha256=canonical_sha256(
            [item.sample_id for item in confirmation]
        ),
        confirmation_category_counts=dict(
            sorted(Counter(item.category for item in confirmation).items())
        ),
    )


def _partition_metadata(
    records: list[_RecordMetadata],
    counts: dict[str, int],
    split_salt: str,
) -> dict[str, list[_RecordMetadata]]:
    split_names = list(counts)
    strata: dict[str, list[_RecordMetadata]] = {}
    for record in records:
        strata.setdefault(record.category, []).append(record)
    quotas: dict[str, dict[str, int]] = {}
    remainders: dict[str, dict[str, float]] = {}
    assigned = {name: 0 for name in split_names}
    leftovers: dict[str, int] = {}
    for stratum, members in strata.items():
        quotas[stratum] = {}
        remainders[stratum] = {}
        for name in split_names:
            ideal = len(members) * counts[name] / len(records)
            base = int(ideal)
            quotas[stratum][name] = base
            remainders[stratum][name] = ideal - base
            assigned[name] += base
        leftovers[stratum] = len(members) - sum(quotas[stratum].values())
    deficits = {name: counts[name] - assigned[name] for name in split_names}
    for stratum in sorted(strata):
        used: set[str] = set()
        for _ in range(leftovers[stratum]):
            choices = [name for name in split_names if deficits[name] > 0]
            pool = [name for name in choices if name not in used] or choices
            if not pool:
                raise FirebreakError("split apportionment exhausted capacity")
            name = max(
                pool,
                key=lambda candidate: (
                    remainders[stratum][candidate],
                    deficits[candidate],
                    -split_names.index(candidate),
                ),
            )
            quotas[stratum][name] += 1
            deficits[name] -= 1
            used.add(name)
    if any(deficits.values()):
        raise FirebreakError("split apportionment did not fill every partition")

    selected: dict[str, list[_RecordMetadata]] = {
        name: [] for name in split_names
    }
    for stratum in sorted(strata):
        ordered = sorted(
            strata[stratum],
            key=lambda record: hashlib.sha256(
                f"{split_salt}\0{stratum}\0{record.sample_id}".encode("utf-8")
            ).digest(),
        )
        offset = 0
        for name in split_names:
            count = quotas[stratum][name]
            selected[name].extend(ordered[offset : offset + count])
            offset += count
    for name in split_names:
        selected[name].sort(
            key=lambda record: hashlib.sha256(
                f"{split_salt}\0order\0{record.sample_id}".encode("utf-8")
            ).digest()
        )
    return selected


def _verify_analysis_population(
    population: AnalysisPopulation,
    expected: ExpectedPopulationLock,
) -> None:
    if (
        population.dataset_sha256 != expected.dataset_sha256
        or population.dataset_bytes != expected.dataset_bytes
        or population.split_manifest_sha256 != expected.split_manifest_sha256
        or population.split_format != expected.split_format
        or population.split_algorithm != expected.split_algorithm
        or population.split_salt != expected.split_salt
    ):
        raise FirebreakError("analysis population protocol differs from the lock")
    for name in ("development", "validation"):
        actual = population.partitions[name]
        locked = expected.partitions[name]
        if not all(
            (
                len(actual.samples) == locked.count,
                actual.ordered_ids_sha256 == locked.ordered_question_ids_sha256,
                actual.ordered_normalized_bindings_sha256
                == locked.ordered_normalized_sample_bindings_sha256,
                actual.ordered_raw_bindings_sha256
                == locked.ordered_raw_record_bindings_sha256,
                actual.category_counts == locked.category_counts,
            )
        ):
            raise FirebreakError(f"{name} population differs from the lock")
    confirmation = expected.partitions["confirmation"]
    if not all(
        (
            population.confirmation_count == confirmation.count,
            population.confirmation_ordered_ids_sha256
            == confirmation.ordered_question_ids_sha256,
            population.confirmation_category_counts == confirmation.category_counts,
        )
    ):
        raise FirebreakError("confirmation membership metadata differs from the lock")
    if population.role_ids_sha256("analysis") != (
        expected.analysis_ordered_question_ids_sha256
    ):
        raise FirebreakError("analysis-pool order differs from the lock")


def _treatment_value(population: AnalysisPopulation) -> dict[str, Any]:
    samples = population.role_samples("analysis")
    return {
        "format": TREATMENT_INPUT_FORMAT,
        "role": "analysis",
        "dataset_sha256": population.dataset_sha256,
        "split_manifest_sha256": population.split_manifest_sha256,
        "ordered_question_ids_sha256": population.role_ids_sha256("analysis"),
        "samples": [sample.treatment_projection for sample in samples],
    }


def _receipt(
    population: AnalysisPopulation,
    treatment_receipt: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": ANALYSIS_EXPORT_RECEIPT_FORMAT,
        "status": "verified",
        "verifier_implementation_sha256": package_sha256(Path(__file__).parent),
        "dataset": {
            "sha256": population.dataset_sha256,
            "bytes": population.dataset_bytes,
            "contents_emitted": False,
        },
        "split_manifest": {
            "sha256": population.split_manifest_sha256,
            "format": population.split_format,
            "algorithm": population.split_algorithm,
            "salt_sha256": canonical_sha256(population.split_salt),
        },
        "partitions": {
            name: population.partitions[name].receipt()
            for name in ("development", "validation")
        },
        "analysis_pool": {
            "source_partitions": ["development", "validation"],
            "count": len(population.role_samples("analysis")),
            "ordered_question_ids_sha256": population.role_ids_sha256("analysis"),
            "status": "designated_analysis_used_provider_free_tuning_only",
        },
        "confirmation_membership": {
            "count": population.confirmation_count,
            "ordered_question_ids_sha256": (
                population.confirmation_ordered_ids_sha256
            ),
            "history_decoded": False,
            "gold_decoded": False,
            "content_emitted": False,
        },
        "treatment_input": treatment_receipt,
        "firebreak": {
            "closed_treatment_schema": True,
            "scorer_labels_in_treatment_input": False,
            "confirmation_history_decoded": False,
            "confirmation_gold_decoded": False,
            "confirmation_ids_emitted": False,
            "question_or_history_text_in_receipt": False,
            "gold_or_evidence_labels_emitted": False,
            "verifier_dependency_class": "python_standard_library_only",
        },
    }


def _scan_dataset_metadata(payload: bytes) -> list[_RecordMetadata]:
    """Extract only top-level ID/category fields from a JSON record array."""

    index = _whitespace(payload, 0)
    if index >= len(payload) or payload[index] != ord("["):
        raise FirebreakError("analysis-only dataset must be a top-level JSON array")
    index = _whitespace(payload, index + 1)
    records: list[_RecordMetadata] = []
    if index < len(payload) and payload[index] == ord("]"):
        end = _whitespace(payload, index + 1)
        if end != len(payload):
            raise FirebreakError("dataset has trailing bytes")
        return records
    while True:
        start = index
        end = _skip_value(payload, start)
        records.append(_record_metadata(payload, start, end, len(records)))
        index = _whitespace(payload, end)
        if index >= len(payload):
            raise FirebreakError("unterminated dataset array")
        token = payload[index]
        if token == ord("]"):
            index = _whitespace(payload, index + 1)
            if index != len(payload):
                raise FirebreakError("dataset has trailing bytes")
            return records
        if token != ord(","):
            raise FirebreakError("dataset array separator is invalid")
        index = _whitespace(payload, index + 1)
        if index >= len(payload) or payload[index] == ord("]"):
            raise FirebreakError("dataset array has a trailing comma")


def _record_metadata(
    payload: bytes,
    start: int,
    end: int,
    record_index: int,
) -> _RecordMetadata:
    index = _whitespace(payload, start)
    if index >= end or payload[index] != ord("{"):
        raise FirebreakError("dataset records must be JSON objects")
    index = _whitespace(payload, index + 1)
    values: dict[str, Any] = {}
    keys: set[str] = set()
    if index < end and payload[index] == ord("}"):
        index += 1
    else:
        while True:
            key_start = index
            key_end = _skip_string(payload, key_start)
            key = _decode_string(payload[key_start:key_end], "dataset record key")
            if key in keys:
                raise FirebreakError("dataset record repeats a JSON key")
            keys.add(key)
            index = _whitespace(payload, key_end)
            if index >= end or payload[index] != ord(":"):
                raise FirebreakError("dataset record key lacks a value")
            value_start = _whitespace(payload, index + 1)
            value_end = _skip_value(payload, value_start)
            if key in {"question_id", "question_type"}:
                values[key] = _decode_scalar(
                    payload[value_start:value_end],
                    f"dataset {key}",
                )
            index = _whitespace(payload, value_end)
            if index >= end:
                raise FirebreakError("unterminated dataset record")
            token = payload[index]
            if token == ord("}"):
                index += 1
                break
            if token != ord(","):
                raise FirebreakError("dataset record separator is invalid")
            index = _whitespace(payload, index + 1)
            if index >= end or payload[index] == ord("}"):
                raise FirebreakError("dataset record has a trailing comma")
    if index != end:
        raise FirebreakError("dataset record span is inconsistent")
    sample_id = str(values.get("question_id") or f"longmemeval_{record_index}")
    category_value = values.get("question_type")
    category = (
        str(category_value) if category_value is not None else "uncategorized"
    )
    return _RecordMetadata(
        index=record_index,
        sample_id=sample_id,
        category=category,
        start=start,
        end=end,
    )


def _skip_value(payload: bytes, index: int) -> int:
    index = _whitespace(payload, index)
    if index >= len(payload):
        raise FirebreakError("truncated JSON value")
    token = payload[index]
    if token == ord('"'):
        return _skip_string(payload, index)
    if token == ord("{"):
        return _skip_object(payload, index)
    if token == ord("["):
        return _skip_array(payload, index)
    for literal in (b"true", b"false", b"null"):
        if payload.startswith(literal, index):
            return index + len(literal)
    match = _NUMBER_RE.match(payload, index)
    if match is not None:
        return match.end()
    raise FirebreakError("invalid JSON value")


def _skip_object(payload: bytes, index: int) -> int:
    index = _whitespace(payload, index + 1)
    if index < len(payload) and payload[index] == ord("}"):
        return index + 1
    while True:
        if index >= len(payload) or payload[index] != ord('"'):
            raise FirebreakError("JSON object key must be text")
        index = _whitespace(payload, _skip_string(payload, index))
        if index >= len(payload) or payload[index] != ord(":"):
            raise FirebreakError("JSON object key lacks a value")
        index = _whitespace(payload, _skip_value(payload, index + 1))
        if index >= len(payload):
            raise FirebreakError("unterminated JSON object")
        if payload[index] == ord("}"):
            return index + 1
        if payload[index] != ord(","):
            raise FirebreakError("JSON object separator is invalid")
        index = _whitespace(payload, index + 1)
        if index >= len(payload) or payload[index] == ord("}"):
            raise FirebreakError("JSON object has a trailing comma")


def _skip_array(payload: bytes, index: int) -> int:
    index = _whitespace(payload, index + 1)
    if index < len(payload) and payload[index] == ord("]"):
        return index + 1
    while True:
        index = _whitespace(payload, _skip_value(payload, index))
        if index >= len(payload):
            raise FirebreakError("unterminated JSON array")
        if payload[index] == ord("]"):
            return index + 1
        if payload[index] != ord(","):
            raise FirebreakError("JSON array separator is invalid")
        index = _whitespace(payload, index + 1)
        if index >= len(payload) or payload[index] == ord("]"):
            raise FirebreakError("JSON array has a trailing comma")


def _skip_string(payload: bytes, index: int) -> int:
    if index >= len(payload) or payload[index] != ord('"'):
        raise FirebreakError("JSON string is invalid")
    index += 1
    while index < len(payload):
        token = payload[index]
        if token == ord('"'):
            return index + 1
        if token < 0x20:
            raise FirebreakError("JSON string contains a control byte")
        if token == ord("\\"):
            index += 1
            if index >= len(payload):
                raise FirebreakError("JSON string escape is truncated")
            escape = payload[index]
            if escape == ord("u"):
                digits = payload[index + 1 : index + 5]
                if len(digits) != 4 or any(value not in _HEX for value in digits):
                    raise FirebreakError("JSON unicode escape is invalid")
                index += 5
                continue
            if escape not in b'"\\/bfnrt':
                raise FirebreakError("JSON string escape is invalid")
        index += 1
    raise FirebreakError("unterminated JSON string")


def _whitespace(payload: bytes, index: int) -> int:
    while index < len(payload) and payload[index] in b" \t\r\n":
        index += 1
    return index


def _decode_string(payload: bytes, label: str) -> str:
    value = _decode_scalar(payload, label)
    if not isinstance(value, str):
        raise FirebreakError(f"{label} must be text")
    return value


def _decode_scalar(payload: bytes, label: str) -> Any:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FirebreakError(f"cannot decode {label}") from exc
    if isinstance(value, (dict, list)):
        raise FirebreakError(f"{label} must be scalar")
    return value


__all__ = [
    "ANALYSIS_EXPORT_RECEIPT_FORMAT",
    "AnalysisPopulation",
    "export_analysis_treatment_input",
    "verify_analysis_treatment_input",
]
