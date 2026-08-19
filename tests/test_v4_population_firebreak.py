from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from memory_condense.ingest.loader import parse_longmemeval
from tools.v4_population_firebreak import (
    FirebreakError,
    export_analysis_treatment_input,
    load_analysis_treatment_input,
    verify_analysis_treatment_input,
)
from tools.v4_population_firebreak import analysis as analysis_module
from tools.v4_population_firebreak import treatment as treatment_module
from tools.v4_population_firebreak.canonical import (
    canonical_sha256,
    read_snapshot,
)
from tools.v4_population_firebreak.population import (
    Population,
    _parse_record,
    reconstruct_population,
)
from tools.v4_population_firebreak.treatment import TREATMENT_INPUT_FORMAT
from tools.v4_population_firebreak.verifier import (
    PRODUCTION_LOCK,
    ExpectedPopulationLock,
    PartitionExpectation,
    verify_evaluator_firebreak,
)


@dataclass(frozen=True)
class _Fixture:
    dataset: Path
    split: Path
    exposure: Path
    analysis: Path
    confirmation: Path
    population: Population
    expected: ExpectedPopulationLock
    answer_values: tuple[str, ...]


def _record(index: int) -> dict[str, object]:
    sample_id = f"locked-sample-{index}"
    return {
        "question_id": sample_id,
        "question_type": "kind-a" if index % 2 == 0 else "kind-b",
        "question": f"private query {index}",
        "answer": f"private gold {index}",
        "question_date": f"2026/08/{index + 1:02d}",
        "haystack_session_ids": [f"source-{index}"],
        "haystack_dates": [f"2026/07/{index + 1:02d}"],
        "haystack_sessions": [
            [
                {"role": "user", "content": f"private history user {index}"},
                {
                    "role": "assistant",
                    "content": f"private history assistant {index}",
                },
            ]
        ],
        "answer_session_ids": [f"source-{index}"],
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _partition_expectation(population: Population, name: str) -> PartitionExpectation:
    partition = population.partitions[name]
    return PartitionExpectation(
        count=len(partition.samples),
        ordered_question_ids_sha256=partition.ordered_ids_sha256,
        ordered_normalized_sample_bindings_sha256=(
            partition.ordered_normalized_bindings_sha256
        ),
        ordered_raw_record_bindings_sha256=partition.ordered_raw_bindings_sha256,
        category_counts=partition.category_counts,
    )


def _treatment_value(population: Population, role: str) -> dict[str, object]:
    samples = population.role_samples(role)
    return {
        "format": TREATMENT_INPUT_FORMAT,
        "role": role,
        "dataset_sha256": population.dataset_sha256,
        "split_manifest_sha256": population.split_manifest_sha256,
        "ordered_question_ids_sha256": population.role_ids_sha256(role),
        "samples": [copy.deepcopy(sample.treatment_projection) for sample in samples],
    }


def _fixture(tmp_path: Path) -> _Fixture:
    dataset = tmp_path / "dataset.json"
    split = tmp_path / "split.json"
    exposure = tmp_path / "exposure.json"
    analysis = tmp_path / "analysis.json"
    confirmation = tmp_path / "confirmation.json"
    records = [_record(index) for index in range(6)]
    _write_json(dataset, records)
    dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    split_value = {
        "format": "memory-condense-locked-benchmark-split-v1",
        "dataset_sha256": dataset_sha,
        "salt": "test-firebreak-salt",
        "algorithm": "stratified-largest-remainder-v1",
        "splits": {"development": 2, "validation": 2, "confirmation": 2},
    }
    _write_json(split, split_value)
    population = reconstruct_population(
        read_snapshot(dataset, "test dataset"),
        read_snapshot(split, "test split"),
    )
    exposed_id = population.partitions["confirmation"].samples[0].sample_id
    exposure_value = {
        "format": "test-exposure-ledger-v1",
        "numeric_answers": [
            {"question_id": exposed_id, "answer": "never emit this value"}
        ],
    }
    _write_json(exposure, exposure_value)
    expected = ExpectedPopulationLock(
        dataset_sha256=population.dataset_sha256,
        dataset_bytes=dataset.stat().st_size,
        split_manifest_sha256=population.split_manifest_sha256,
        split_format=population.split_format,
        split_algorithm=population.split_algorithm,
        split_salt=population.split_salt,
        partitions={
            name: _partition_expectation(population, name)
            for name in ("development", "validation", "confirmation")
        },
        analysis_ordered_question_ids_sha256=population.role_ids_sha256("analysis"),
        exposure_audit_sha256=hashlib.sha256(exposure.read_bytes()).hexdigest(),
        exposed_confirmation_count=1,
        exposed_confirmation_ordered_ids_sha256=canonical_sha256([exposed_id]),
    )
    _write_json(analysis, _treatment_value(population, "analysis"))
    _write_json(confirmation, _treatment_value(population, "confirmation"))
    return _Fixture(
        dataset=dataset,
        split=split,
        exposure=exposure,
        analysis=analysis,
        confirmation=confirmation,
        population=population,
        expected=expected,
        answer_values=tuple(str(record["answer"]) for record in records)
        + ("never emit this value",),
    )


def _verify(fixture: _Fixture) -> dict[str, object]:
    return verify_evaluator_firebreak(
        dataset_path=fixture.dataset,
        split_manifest_path=fixture.split,
        exposure_audit_path=fixture.exposure,
        treatment_input_paths=[fixture.analysis, fixture.confirmation],
        expected=fixture.expected,
    )


def _all_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _all_strings(item)]
    if isinstance(value, dict):
        return [
            text
            for key, item in value.items()
            for text in [*_all_strings(key), *_all_strings(item)]
        ]
    return []


def test_verified_receipt_emits_only_counts_hashes_and_protocol_metadata(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    receipt = _verify(fixture)

    assert receipt["status"] == "verified"
    assert receipt["analysis_pool"]["count"] == 4
    assert receipt["confirmation"]["count"] == 2
    assert receipt["potentially_exposed_confirmation"]["count"] == 1
    assert receipt["potentially_exposed_confirmation"]["ids_emitted"] is False
    assert receipt["firebreak"]["scorer_labels_in_treatment_inputs"] is False
    emitted = _all_strings(receipt)
    private_values = {
        sample.sample_id
        for partition in fixture.population.partitions.values()
        for sample in partition.samples
    }
    private_values.update(fixture.answer_values)
    private_values.update(
        f"private {kind} {index}"
        for index in range(6)
        for kind in ("query", "history user", "history assistant")
    )
    assert private_values.isdisjoint(emitted)
    assert "kind-a" not in emitted
    assert "kind-b" not in emitted


@pytest.mark.parametrize(
    "field,value",
    [
        ("answer", "gold"),
        ("gold_answer", "gold"),
        ("category", "kind-a"),
        ("evidence_sources", ["source"]),
        ("judge_correct", True),
        ("judge_reasoning", "classified"),
        ("f1", 1.0),
    ],
)
def test_any_scorer_or_oracle_field_in_treatment_query_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fixture = _fixture(tmp_path)
    treatment = json.loads(fixture.confirmation.read_text(encoding="utf-8"))
    treatment["samples"][0]["questions"][0][field] = value
    _write_json(fixture.confirmation, treatment)

    with pytest.raises(FirebreakError, match="non-closed schema"):
        _verify(fixture)


def test_confirmation_reordering_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    treatment = json.loads(fixture.confirmation.read_text(encoding="utf-8"))
    treatment["samples"].reverse()
    _write_json(fixture.confirmation, treatment)

    with pytest.raises(FirebreakError, match="reordered or overlaps"):
        _verify(fixture)


def test_cross_partition_overlap_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    confirmation = json.loads(fixture.confirmation.read_text(encoding="utf-8"))
    analysis = json.loads(fixture.analysis.read_text(encoding="utf-8"))
    confirmation["samples"][0] = analysis["samples"][0]
    _write_json(fixture.confirmation, confirmation)

    with pytest.raises(FirebreakError, match="reordered or overlaps"):
        _verify(fixture)


def test_treatment_text_tampering_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    treatment = json.loads(fixture.confirmation.read_text(encoding="utf-8"))
    treatment["samples"][0]["turns"][0][1] += " tampered"
    _write_json(fixture.confirmation, treatment)

    with pytest.raises(FirebreakError, match="was modified"):
        _verify(fixture)


def test_duplicate_or_missing_treatment_role_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(FirebreakError, match="role is repeated"):
        verify_evaluator_firebreak(
            dataset_path=fixture.dataset,
            split_manifest_path=fixture.split,
            exposure_audit_path=fixture.exposure,
            treatment_input_paths=[fixture.analysis, fixture.analysis],
            expected=fixture.expected,
        )
    with pytest.raises(FirebreakError, match="required treatment-input roles"):
        verify_evaluator_firebreak(
            dataset_path=fixture.dataset,
            split_manifest_path=fixture.split,
            exposure_audit_path=fixture.exposure,
            treatment_input_paths=[fixture.analysis],
            expected=fixture.expected,
        )


@pytest.mark.parametrize("target", ["dataset", "split", "exposure"])
def test_locked_source_tampering_fails_closed(tmp_path: Path, target: str) -> None:
    fixture = _fixture(tmp_path)
    path = getattr(fixture, target)
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(FirebreakError, match="identity differs"):
        _verify(fixture)


def test_exposure_hash_is_disclosed_without_ids_or_values(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    receipt = _verify(fixture)
    exposure = receipt["potentially_exposed_confirmation"]
    assert set(exposure) == {
        "count",
        "ordered_ids_sha256",
        "audit_sha256",
        "ids_emitted",
        "values_emitted",
        "claim",
    }
    assert exposure["ordered_ids_sha256"] == (
        fixture.expected.exposed_confirmation_ordered_ids_sha256
    )
    rendered = json.dumps(exposure, sort_keys=True)
    assert all(value not in rendered for value in fixture.answer_values)
    assert all(
        sample.sample_id not in rendered
        for sample in fixture.population.partitions["confirmation"].samples
    )


def test_duplicate_json_keys_fail_closed_before_schema_validation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = fixture.confirmation.read_text(encoding="utf-8")
    duplicate = payload[:-1] + ',"role":"confirmation"}'
    fixture.confirmation.write_text(duplicate, encoding="utf-8")

    with pytest.raises(FirebreakError, match="strict JSON"):
        _verify(fixture)


def test_projection_matches_authoritative_loader_chronology_and_timestamps() -> None:
    record = {
        "question_id": "chronology-probe",
        "question_type": "multi-session",
        "question": "What happened in order?",
        "answer": "early then late",
        "question_date": "2024/06/01 (Sat) 12:00",
        "haystack_session_ids": ["late", "same-a", "bad", "early", "same-b"],
        "haystack_dates": [
            "2024/05/04 (Sat) 12:00",
            "2024/05/02 (Thu) 12:00",
            "not-a-date",
            "2024/05/01 (Wed) 12:00 (UTC)",
            "2024/05/02 (Thu) 12:00",
        ],
        "haystack_sessions": [
            [{"role": "user", "content": "late event"}],
            [{"role": "user", "content": "same time a"}],
            [{"role": "user", "content": "unparseable event"}],
            [{"role": "user", "content": "early event"}],
            [{"role": "user", "content": "same time b"}],
            [{"role": "user", "content": "missing date event"}],
        ],
        "answer_session_ids": ["early", "late"],
    }

    projected = _parse_record(record, 0)
    assert projected is not None
    authoritative = parse_longmemeval([record])[0].model_dump(mode="json")
    treatment = projected.treatment_projection
    assert treatment["turns"] == authoritative["turns"]
    assert treatment["turn_source_ids"] == authoritative["turn_source_ids"]
    assert treatment["turn_created_at"] == authoritative["turn_created_at"]
    assert treatment["turn_source_ids"] == [
        "early",
        "early",
        "same-a",
        "same-a",
        "same-b",
        "same-b",
        "late",
        "late",
        "bad",
        "bad",
        "session_6",
    ]
    assert treatment["turn_created_at"][:2] == [
        "2024-05-01T12:00:00Z",
        "2024-05-01T12:00:00Z",
    ]
    assert treatment["turn_created_at"][-3:] == [None, None, None]

    # Compatibility is intentional: production normalized roots bind the v1
    # file-order projection, while the independently bound treatment v2 is
    # chronology-aware.
    legacy_turns: list[list[str]] = []
    legacy_sources: list[str] = []
    for index, session in enumerate(record["haystack_sessions"]):
        source = (
            record["haystack_session_ids"][index]
            if index < len(record["haystack_session_ids"])
            else f"session_{index + 1}"
        )
        if index < len(record["haystack_dates"]):
            date = record["haystack_dates"][index]
            legacy_turns.append(["system", f"[{source} took place at {date}]"])
            legacy_sources.append(source)
        legacy_turns.append(["user", session[0]["content"]])
        legacy_sources.append(source)
    legacy_normalized = {
        "sample_id": "chronology-probe",
        "turns": legacy_turns,
        "turn_source_ids": legacy_sources,
        "questions": [
            {
                "question_id": "chronology-probe",
                "question": "What happened in order?",
                "answer": "early then late",
                "category": "multi-session",
                "evidence": ["early", "late"],
                "evidence_sources": ["early", "late"],
                "question_date": "2024/06/01 (Sat) 12:00",
            }
        ],
    }
    assert projected.normalized_sha256 == canonical_sha256(legacy_normalized)
    assert treatment["turns"] != legacy_turns


def test_analysis_only_export_never_decodes_confirmation_content_or_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    confirmation_ids = {
        sample.sample_id
        for sample in fixture.population.partitions["confirmation"].samples
    }
    confirmation_indexes = {
        int(sample_id.rsplit("-", 1)[1]) for sample_id in confirmation_ids
    }
    forbidden = {
        text.encode("utf-8")
        for index in confirmation_indexes
        for text in (
            f"private query {index}",
            f"private gold {index}",
            f"private history user {index}",
            f"private history assistant {index}",
        )
    }
    original_parse = analysis_module.parse_json_bytes
    decoded_labels: list[str] = []

    def guarded_parse(payload: bytes, label: str):
        assert all(value not in payload for value in forbidden)
        decoded_labels.append(label)
        return original_parse(payload, label)

    monkeypatch.setattr(analysis_module, "parse_json_bytes", guarded_parse)
    output = tmp_path / "analysis-v2.json"
    receipt = export_analysis_treatment_input(
        dataset_path=fixture.dataset,
        split_manifest_path=fixture.split,
        output_path=output,
        expected=fixture.expected,
    )

    assert receipt["confirmation_membership"]["count"] == 2
    assert receipt["confirmation_membership"]["history_decoded"] is False
    assert receipt["confirmation_membership"]["gold_decoded"] is False
    assert all("confirmation record" not in label for label in decoded_labels)
    rendered_artifact = output.read_text(encoding="utf-8")
    assert all(answer not in rendered_artifact for answer in fixture.answer_values)
    assert all(sample_id not in rendered_artifact for sample_id in confirmation_ids)

    binding = receipt["treatment_input"]
    loaded = load_analysis_treatment_input(
        output,
        expected_file_sha256=binding["file_sha256"],
        expected_sanitized_projection_sha256=(
            binding["sanitized_projection_sha256"]
        ),
        expected_dataset_sha256=receipt["dataset"]["sha256"],
        expected_split_manifest_sha256=receipt["split_manifest"]["sha256"],
        expected_ordered_question_ids_sha256=(
            receipt["analysis_pool"]["ordered_question_ids_sha256"]
        ),
        expected_sample_count=receipt["analysis_pool"]["count"],
    )
    analysis_ids = {
        sample.sample_id
        for name in ("development", "validation")
        for sample in fixture.population.partitions[name].samples
    }
    assert {sample.sample_id for sample in loaded.samples} == analysis_ids
    assert all(sample.turn_created_at for sample in loaded.samples)
    with pytest.raises(FrozenInstanceError):
        loaded.samples[0].sample_id = "mutated"  # type: ignore[misc]

    verified = verify_analysis_treatment_input(
        dataset_path=fixture.dataset,
        split_manifest_path=fixture.split,
        treatment_input_path=output,
        expected=fixture.expected,
    )
    assert verified["treatment_input"] == binding
    delegated = verify_evaluator_firebreak(
        dataset_path=fixture.dataset,
        split_manifest_path=fixture.split,
        exposure_audit_path=None,
        treatment_input_paths=[output],
        expected=fixture.expected,
        required_roles=("analysis",),
    )
    assert delegated["treatment_input"] == binding
    with pytest.raises(FirebreakError, match="cannot accept an exposure audit"):
        verify_evaluator_firebreak(
            dataset_path=fixture.dataset,
            split_manifest_path=fixture.split,
            exposure_audit_path=fixture.exposure,
            treatment_input_paths=[output],
            expected=fixture.expected,
            required_roles=("analysis",),
        )


def _load_rebased_analysis(path: Path, value: dict[str, object]):
    _write_json(path, value)
    samples = value["samples"]
    assert isinstance(samples, list)
    return load_analysis_treatment_input(
        path,
        expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_sanitized_projection_sha256=canonical_sha256(samples),
        expected_dataset_sha256=str(value["dataset_sha256"]),
        expected_split_manifest_sha256=str(value["split_manifest_sha256"]),
        expected_ordered_question_ids_sha256=canonical_sha256(
            [sample["sample_id"] for sample in samples]
        ),
        expected_sample_count=len(samples),
    )


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-07-01T00:00:00+00:00",
        "2026-06-30T17:00:00-07:00",
        "2026-07-01T00:00:00",
        "not-a-timestamp",
    ],
)
def test_analysis_consumer_rejects_noncanonical_timestamp_even_if_rebased(
    tmp_path: Path,
    timestamp: str,
) -> None:
    fixture = _fixture(tmp_path)
    value = json.loads(fixture.analysis.read_text(encoding="utf-8"))
    value["samples"][0]["turn_created_at"][0] = timestamp
    with pytest.raises(FirebreakError, match="timestamp"):
        _load_rebased_analysis(tmp_path / "rebased.json", value)


def test_analysis_consumer_rejects_timestamp_misalignment_and_duplicate_ids(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    misaligned = json.loads(fixture.analysis.read_text(encoding="utf-8"))
    misaligned["samples"][0]["turn_created_at"].pop()
    with pytest.raises(FirebreakError, match="misaligned timestamps"):
        _load_rebased_analysis(tmp_path / "misaligned.json", misaligned)

    repeated = json.loads(fixture.analysis.read_text(encoding="utf-8"))
    repeated["samples"][1]["sample_id"] = repeated["samples"][0]["sample_id"]
    repeated["samples"][1]["questions"][0]["question_id"] = (
        repeated["samples"][0]["sample_id"]
    )
    repeated["ordered_question_ids_sha256"] = canonical_sha256(
        [sample["sample_id"] for sample in repeated["samples"]]
    )
    with pytest.raises(FirebreakError, match="repeats a sample ID"):
        _load_rebased_analysis(tmp_path / "repeated.json", repeated)


def test_analysis_consumer_hash_gate_precedes_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)

    def forbidden_decode(*_args, **_kwargs):
        raise AssertionError("wrong-hash input must not be decoded")

    monkeypatch.setattr(treatment_module, "parse_json_bytes", forbidden_decode)
    with pytest.raises(FirebreakError, match="differs from its receipt"):
        load_analysis_treatment_input(
            fixture.analysis,
            expected_file_sha256="0" * 64,
            expected_sanitized_projection_sha256="1" * 64,
            expected_dataset_sha256=fixture.population.dataset_sha256,
            expected_split_manifest_sha256=fixture.population.split_manifest_sha256,
            expected_ordered_question_ids_sha256=(
                fixture.population.role_ids_sha256("analysis")
            ),
            expected_sample_count=4,
        )


def test_production_lock_preserves_exact_v2_roots_and_200_confirmation() -> None:
    assert PRODUCTION_LOCK.dataset_sha256 == (
        "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
    )
    assert PRODUCTION_LOCK.split_manifest_sha256 == (
        "8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4"
    )
    assert PRODUCTION_LOCK.partitions["development"].ordered_question_ids_sha256 == (
        "533aa545efb8032f7b181f39264c6d10a49471bd460414f420e37dc840a19c55"
    )
    assert PRODUCTION_LOCK.partitions["validation"].ordered_question_ids_sha256 == (
        "7a67aa6f43ffb94d487fb9184f871735bd9edac1974a3154898846d1140c83a1"
    )
    confirmation = PRODUCTION_LOCK.partitions["confirmation"]
    assert confirmation.count == 200
    assert confirmation.ordered_question_ids_sha256 == (
        "6270b044792dbda79cd79a104ab6a519b2f81980c47522c19a196583d8c0d102"
    )
    assert PRODUCTION_LOCK.exposed_confirmation_count == 15
    assert PRODUCTION_LOCK.exposed_confirmation_ordered_ids_sha256 == (
        "8b6fd7053b8139834baedd5623e8b3c55fc14b5908c32c7dbe97811a3f9c8a76"
    )


def test_cold_import_loads_no_provider_or_model_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        "import sys;"
        f"sys.path.insert(0,{str(root)!r});"
        "import tools.v4_population_firebreak;"
        "bad=sorted(name for name in sys.modules if name.split('.')[0] in "
        "{'torch','transformers','openai','anthropic','requests','httpx'});"
        "print(','.join(bad))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == ""
