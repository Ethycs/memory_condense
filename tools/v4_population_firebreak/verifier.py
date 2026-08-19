"""Orchestrate the population lock and evaluator-only label firebreak."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import (
    FirebreakError,
    assert_snapshot_unchanged,
    canonical_sha256,
    package_sha256,
    parse_json_bytes,
    read_snapshot,
    require_list,
    require_mapping,
    require_text,
)
from .population import Population, reconstruct_population
from .treatment import validate_treatment_input


@dataclass(frozen=True, slots=True)
class PartitionExpectation:
    count: int
    ordered_question_ids_sha256: str
    ordered_normalized_sample_bindings_sha256: str
    ordered_raw_record_bindings_sha256: str
    category_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class ExpectedPopulationLock:
    dataset_sha256: str
    dataset_bytes: int
    split_manifest_sha256: str
    split_format: str
    split_algorithm: str
    split_salt: str
    partitions: dict[str, PartitionExpectation]
    analysis_ordered_question_ids_sha256: str
    exposure_audit_sha256: str
    exposed_confirmation_count: int
    exposed_confirmation_ordered_ids_sha256: str


PRODUCTION_LOCK = ExpectedPopulationLock(
    dataset_sha256="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    dataset_bytes=277_383_467,
    split_manifest_sha256="8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4",
    split_format="memory-condense-locked-benchmark-split-v1",
    split_algorithm="stratified-largest-remainder-v1",
    split_salt="memory-condense-longmemeval-95-v1-2026-08-16",
    partitions={
        "development": PartitionExpectation(
            count=200,
            ordered_question_ids_sha256=(
                "533aa545efb8032f7b181f39264c6d10a49471bd460414f420e37dc840a19c55"
            ),
            ordered_normalized_sample_bindings_sha256=(
                "fabb9bd4527201294184598e0655964717325cba88c3d9da7c39d92cdd1459ea"
            ),
            ordered_raw_record_bindings_sha256=(
                "d28196b5933b3ddd6c8ea2870d048c9d3f8853d3d69963ba25d2266648114cdb"
            ),
            category_counts={
                "knowledge-update": 31,
                "multi-session": 53,
                "single-session-assistant": 23,
                "single-session-preference": 12,
                "single-session-user": 28,
                "temporal-reasoning": 53,
            },
        ),
        "validation": PartitionExpectation(
            count=100,
            ordered_question_ids_sha256=(
                "7a67aa6f43ffb94d487fb9184f871735bd9edac1974a3154898846d1140c83a1"
            ),
            ordered_normalized_sample_bindings_sha256=(
                "718c6cdf238baa868d270bb7ae63f74472fe73cc2fb1d1217b736f87fb3ae679"
            ),
            ordered_raw_record_bindings_sha256=(
                "babcb9f497d742ccccb4c5e1e4d01d6d0ef55cc2c7b8942eadaafed6f593824f"
            ),
            category_counts={
                "knowledge-update": 16,
                "multi-session": 27,
                "single-session-assistant": 11,
                "single-session-preference": 6,
                "single-session-user": 14,
                "temporal-reasoning": 26,
            },
        ),
        "confirmation": PartitionExpectation(
            count=200,
            ordered_question_ids_sha256=(
                "6270b044792dbda79cd79a104ab6a519b2f81980c47522c19a196583d8c0d102"
            ),
            ordered_normalized_sample_bindings_sha256=(
                "cbabcc97cad2f945c397fd980ef3bb3fb65ba8403dbeadf38b1b8224bc4a066d"
            ),
            ordered_raw_record_bindings_sha256=(
                "cf86373d06725b26117e9ce96ce906a16d545d346a1d2888f200d425f7a27fd9"
            ),
            category_counts={
                "knowledge-update": 31,
                "multi-session": 53,
                "single-session-assistant": 22,
                "single-session-preference": 12,
                "single-session-user": 28,
                "temporal-reasoning": 54,
            },
        ),
    },
    analysis_ordered_question_ids_sha256=(
        "cf5e8648b71634e4e22be872881766e37e0dc24a2931d0c63365e075b2742046"
    ),
    exposure_audit_sha256="0d2e83bcc8d6f9f84c752b579fdbc2d580fd454efa95101c93aa36606eef280b",
    exposed_confirmation_count=15,
    exposed_confirmation_ordered_ids_sha256=(
        "8b6fd7053b8139834baedd5623e8b3c55fc14b5908c32c7dbe97811a3f9c8a76"
    ),
)


def verify_evaluator_firebreak(
    *,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    exposure_audit_path: str | Path | None,
    treatment_input_paths: Iterable[str | Path],
    expected: ExpectedPopulationLock = PRODUCTION_LOCK,
    required_roles: tuple[str, ...] = ("analysis", "confirmation"),
) -> dict[str, Any]:
    """Verify exact populations and closed, label-free treatment inputs.

    The returned receipt contains only counts, hashes, fixed protocol labels,
    and booleans. It intentionally contains no sample IDs, question/history
    text, gold values, evidence labels, predictions, or judge fields.
    """

    if required_roles == ("analysis",):
        if exposure_audit_path is not None:
            raise FirebreakError(
                "analysis-only verification cannot accept an exposure audit"
            )
        paths = tuple(treatment_input_paths)
        if len(paths) != 1:
            raise FirebreakError(
                "analysis-only verification requires exactly one treatment input"
            )
        # The selective implementation is imported only for this explicit
        # mode; confirmation lock mode below retains its full audit.
        from .analysis import verify_analysis_treatment_input

        return verify_analysis_treatment_input(
            dataset_path=dataset_path,
            split_manifest_path=split_manifest_path,
            treatment_input_path=paths[0],
            expected=expected,
        )
    if exposure_audit_path is None:
        raise FirebreakError("confirmation lock mode requires an exposure audit")

    dataset = read_snapshot(dataset_path, "dataset")
    split = read_snapshot(split_manifest_path, "split manifest")
    exposure = read_snapshot(exposure_audit_path, "exposure audit")
    if dataset.sha256 != expected.dataset_sha256 or dataset.size != expected.dataset_bytes:
        raise FirebreakError("dataset identity differs from the locked population")
    if split.sha256 != expected.split_manifest_sha256:
        raise FirebreakError("split-manifest identity differs from the lock")
    if exposure.sha256 != expected.exposure_audit_sha256:
        raise FirebreakError("exposure-audit identity differs from the lock")
    population = reconstruct_population(dataset, split)
    _verify_population(population, expected)
    exposure_receipt = _verify_exposure(exposure.payload, population, expected)

    receipts: dict[str, dict[str, Any]] = {}
    for index, path in enumerate(treatment_input_paths):
        snapshot = read_snapshot(path, f"treatment input {index}")
        result = validate_treatment_input(snapshot, population)
        if result.role in receipts:
            raise FirebreakError("treatment-input role is repeated")
        receipts[result.role] = result.json_value()
        assert_snapshot_unchanged(snapshot, f"treatment input {index}")
    if set(receipts) != set(required_roles) or len(receipts) != len(required_roles):
        raise FirebreakError("required treatment-input roles were not verified exactly once")

    assert_snapshot_unchanged(dataset, "dataset")
    assert_snapshot_unchanged(split, "split manifest")
    assert_snapshot_unchanged(exposure, "exposure audit")

    partition_receipts = {
        name: population.partitions[name].receipt()
        for name in ("development", "validation", "confirmation")
    }
    return {
        "format": "memory-condense-v4-evaluator-firebreak-receipt-v1",
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
        "partitions": partition_receipts,
        "analysis_pool": {
            "source_partitions": ["development", "validation"],
            "count": len(population.role_samples("analysis")),
            "ordered_question_ids_sha256": population.role_ids_sha256("analysis"),
            "status": "designated_analysis_used_provider_free_tuning_only",
        },
        "confirmation": {
            "source_partition": "confirmation",
            "count": len(population.role_samples("confirmation")),
            "ordered_question_ids_sha256": population.role_ids_sha256("confirmation"),
            "status": "designated_evaluator_held_final_only",
        },
        "potentially_exposed_confirmation": exposure_receipt,
        "treatment_inputs": receipts,
        "firebreak": {
            "closed_treatment_schema": True,
            "scorer_labels_in_treatment_inputs": False,
            "sample_ids_emitted": False,
            "question_or_history_text_emitted": False,
            "gold_or_evidence_labels_emitted": False,
            "verifier_dependency_class": "python_standard_library_only",
        },
    }


def _verify_population(
    population: Population,
    expected: ExpectedPopulationLock,
) -> None:
    if (
        population.split_format != expected.split_format
        or population.split_algorithm != expected.split_algorithm
        or population.split_salt != expected.split_salt
    ):
        raise FirebreakError("split protocol differs from the population lock")
    if set(population.partitions) != set(expected.partitions):
        raise FirebreakError("partition names differ from the population lock")
    for name, expectation in expected.partitions.items():
        actual = population.partitions[name]
        checks = (
            len(actual.samples) == expectation.count,
            actual.ordered_ids_sha256 == expectation.ordered_question_ids_sha256,
            actual.ordered_normalized_bindings_sha256
            == expectation.ordered_normalized_sample_bindings_sha256,
            actual.ordered_raw_bindings_sha256
            == expectation.ordered_raw_record_bindings_sha256,
            actual.category_counts == expectation.category_counts,
        )
        if not all(checks):
            raise FirebreakError(f"{name} population differs from the lock")
    if (
        population.role_ids_sha256("analysis")
        != expected.analysis_ordered_question_ids_sha256
    ):
        raise FirebreakError("analysis-pool order differs from the lock")
    # The production expectation pins this to 200. Keeping the verifier
    # parameterized permits small adversarial fixtures without weakening that
    # production lock.
    if (
        len(population.partitions["confirmation"].samples)
        != expected.partitions["confirmation"].count
    ):
        raise FirebreakError("confirmation membership differs from the lock")


def _verify_exposure(
    payload: bytes,
    population: Population,
    expected: ExpectedPopulationLock,
) -> dict[str, Any]:
    value = require_mapping(parse_json_bytes(payload, "exposure audit"), "exposure audit")
    rows = require_list(value.get("numeric_answers"), "exposure audit numeric metadata")
    ids: list[str] = []
    for row in rows:
        item = require_mapping(row, "exposure audit numeric metadata entry")
        # Deliberately read only identity. Never return or render the answer value.
        ids.append(require_text(item.get("question_id"), "exposure metadata ID"))
    if len(ids) != len(set(ids)):
        raise FirebreakError("exposure audit repeats an identity")
    known = {
        sample.sample_id
        for partition in population.partitions.values()
        for sample in partition.samples
    }
    if not set(ids) <= known:
        raise FirebreakError("exposure audit names an unknown identity")
    exposed = set(ids)
    ordered_confirmation = [
        sample.sample_id
        for sample in population.partitions["confirmation"].samples
        if sample.sample_id in exposed
    ]
    digest = canonical_sha256(ordered_confirmation)
    if (
        len(ordered_confirmation) != expected.exposed_confirmation_count
        or digest != expected.exposed_confirmation_ordered_ids_sha256
    ):
        raise FirebreakError("confirmation exposure ledger differs from the lock")
    return {
        "count": len(ordered_confirmation),
        "ordered_ids_sha256": digest,
        "audit_sha256": expected.exposure_audit_sha256,
        "ids_emitted": False,
        "values_emitted": False,
        "claim": "answer_only_metadata_exposure_recorded",
    }
