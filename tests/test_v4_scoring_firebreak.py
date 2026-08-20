from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from tools.v4_population_firebreak import (
    FirebreakError,
    export_analysis_scoring_label,
    load_analysis_scoring_label,
)
from tools.v4_population_firebreak import analysis as analysis_module
from tools.v4_population_firebreak import scoring as scoring_module
from tools.v4_population_firebreak.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    read_snapshot,
)
from tools.v4_population_firebreak.population import Population, reconstruct_population
from tools.v4_population_firebreak.verifier import (
    ExpectedPopulationLock,
    PartitionExpectation,
)


@dataclass(frozen=True)
class _Fixture:
    dataset: Path
    split: Path
    population: Population
    expected: ExpectedPopulationLock
    sample_ordinal: int
    question_probe_sha256: str


def _record(index: int) -> dict[str, object]:
    return {
        "question_id": f"synthetic-sample-{index}",
        "question_type": "kind-a" if index % 2 == 0 else "kind-b",
        "question": f"synthetic private query {index}",
        "answer": f"synthetic private gold {index}",
        "question_date": f"2026/08/{index + 1:02d}",
        "haystack_session_ids": [f"source-{index}-a", f"source-{index}-b"],
        "haystack_dates": [
            f"2026/07/{index + 2:02d}",
            f"2026/07/{index + 1:02d}",
        ],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": f"synthetic private late history {index}",
                }
            ],
            [
                {
                    "role": "assistant",
                    "content": f"synthetic private early history {index}",
                }
            ],
        ],
        "answer_session_ids": [f"source-{index}-a", f"source-{index}-b"],
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _partition_expectation(
    population: Population,
    name: str,
) -> PartitionExpectation:
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


def _probe_sha256(sample_projection: dict[str, object]) -> str:
    questions = sample_projection["questions"]
    assert isinstance(questions, list) and len(questions) == 1
    question = questions[0]
    assert isinstance(question, dict)
    question_id = question["question_id"]
    question_text = question["question"]
    question_date = question["question_date"]
    assert isinstance(question_id, str)
    assert isinstance(question_text, str)
    assert question_date is None or isinstance(question_date, str)
    prompt = (
        question_text
        if question_date is None
        else f"[Question asked at {question_date}]\n{question_text}"
    )
    return canonical_sha256(
        {
            "question_id": question_id,
            "retrieval_query": question_text,
            "prompt_question": prompt,
        }
    )


def _fixture(tmp_path: Path) -> _Fixture:
    dataset = tmp_path / "dataset.json"
    split = tmp_path / "split.json"
    _write_json(dataset, [_record(index) for index in range(9)])
    dataset_sha256 = hashlib.sha256(dataset.read_bytes()).hexdigest()
    _write_json(
        split,
        {
            "format": "memory-condense-locked-benchmark-split-v1",
            "dataset_sha256": dataset_sha256,
            "salt": "synthetic-scoring-firebreak-salt",
            "algorithm": "stratified-largest-remainder-v1",
            "splits": {"development": 3, "validation": 3, "confirmation": 3},
        },
    )
    population = reconstruct_population(
        read_snapshot(dataset, "synthetic dataset"),
        read_snapshot(split, "synthetic split"),
    )
    expected = ExpectedPopulationLock(
        dataset_sha256=population.dataset_sha256,
        dataset_bytes=population.dataset_bytes,
        split_manifest_sha256=population.split_manifest_sha256,
        split_format=population.split_format,
        split_algorithm=population.split_algorithm,
        split_salt=population.split_salt,
        partitions={
            name: _partition_expectation(population, name)
            for name in ("development", "validation", "confirmation")
        },
        analysis_ordered_question_ids_sha256=population.role_ids_sha256("analysis"),
        exposure_audit_sha256="0" * 64,
        exposed_confirmation_count=0,
        exposed_confirmation_ordered_ids_sha256=canonical_sha256([]),
    )
    ordinal = 2
    selected = population.role_samples("analysis")[ordinal]
    return _Fixture(
        dataset=dataset,
        split=split,
        population=population,
        expected=expected,
        sample_ordinal=ordinal,
        question_probe_sha256=_probe_sha256(selected.treatment_projection),
    )


def _export(fixture: _Fixture, output: Path) -> dict[str, object]:
    return export_analysis_scoring_label(
        dataset_path=fixture.dataset,
        split_manifest_path=fixture.split,
        output_path=output,
        sample_ordinal=fixture.sample_ordinal,
        expected_question_probe_sha256=fixture.question_probe_sha256,
        expected=fixture.expected,
    )


def _load(output: Path, receipt: dict[str, object], **overrides: object):
    artifact = receipt["artifact"]
    population = receipt["population"]
    selection = receipt["selection"]
    assert isinstance(artifact, dict)
    assert isinstance(population, dict)
    assert isinstance(selection, dict)
    arguments = {
        "expected_file_sha256": artifact["file_sha256"],
        "expected_label_record_sha256": artifact["label_record_sha256"],
        "expected_dataset_sha256": population["dataset_sha256"],
        "expected_split_manifest_sha256": population["split_manifest_sha256"],
        "expected_analysis_ordered_question_ids_sha256": population[
            "analysis_ordered_question_ids_sha256"
        ],
        "expected_analysis_sample_count": population["analysis_sample_count"],
        "expected_sample_ordinal": selection["sample_ordinal"],
        "expected_sample_id_sha256": selection["sample_id_sha256"],
        "expected_question_id_sha256": selection["question_id_sha256"],
        "expected_question_text_sha256": selection["question_text_sha256"],
        "expected_question_probe_sha256": selection["question_probe_sha256"],
        "expected_raw_record_sha256": selection["raw_record_sha256"],
        "expected_raw_record_span_sha256": selection["raw_record_span_sha256"],
    }
    arguments.update(overrides)
    return load_analysis_scoring_label(output, **arguments)


def _record_index(sample_id: str) -> int:
    return int(sample_id.rsplit("-", 1)[1])


def test_export_and_load_one_canonical_immutable_analysis_label(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "scoring-label.json"
    receipt = _export(fixture, output)
    value = json.loads(output.read_text(encoding="utf-8"))

    assert output.read_bytes() == canonical_json_bytes(value) + b"\n"
    assert receipt["artifact"]["record_count"] == 1
    assert receipt["selection"]["sample_ordinal"] == fixture.sample_ordinal
    assert receipt["confirmation_membership"] == {
        "count": 3,
        "history_decoded": False,
        "question_text_decoded": False,
        "gold_decoded": False,
        "content_emitted": False,
    }
    loaded = _load(output, receipt)
    selected_id = fixture.population.role_samples("analysis")[
        fixture.sample_ordinal
    ].sample_id
    index = _record_index(selected_id)
    assert loaded.question_id == selected_id
    assert loaded.gold_answer == f"synthetic private gold {index}"
    assert loaded.evidence_source_ids == (
        f"source-{index}-a",
        f"source-{index}-b",
    )
    assert loaded.question_probe_sha256 == fixture.question_probe_sha256
    with pytest.raises(FrozenInstanceError):
        loaded.gold_answer = "changed"  # type: ignore[misc]

    rendered_receipt = json.dumps(receipt, sort_keys=True)
    assert loaded.question_id not in rendered_receipt
    assert loaded.gold_answer not in rendered_receipt
    assert all(source not in rendered_receipt for source in loaded.evidence_source_ids)


def test_export_never_decodes_confirmation_content_or_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    confirmation_ids = {
        sample.sample_id
        for sample in fixture.population.partitions["confirmation"].samples
    }
    confirmation_indexes = {_record_index(sample_id) for sample_id in confirmation_ids}
    forbidden = {
        text.encode("utf-8")
        for index in confirmation_indexes
        for text in (
            f"synthetic private query {index}",
            f"synthetic private gold {index}",
            f"synthetic private late history {index}",
            f"synthetic private early history {index}",
        )
    }
    analysis_parse = analysis_module.parse_json_bytes
    scoring_parse = scoring_module.parse_json_bytes

    def guard_analysis(payload: bytes, label: str):
        assert all(secret not in payload for secret in forbidden)
        return analysis_parse(payload, label)

    def guard_scoring(payload: bytes, label: str):
        assert all(secret not in payload for secret in forbidden)
        return scoring_parse(payload, label)

    monkeypatch.setattr(analysis_module, "parse_json_bytes", guard_analysis)
    monkeypatch.setattr(scoring_module, "parse_json_bytes", guard_scoring)
    output = tmp_path / "single-analysis-label.json"
    receipt = _export(fixture, output)

    rendered_artifact = output.read_bytes()
    assert all(secret not in rendered_artifact for secret in forbidden)
    assert receipt["firebreak"]["confirmation_gold_decoded"] is False
    assert receipt["firebreak"]["confirmation_question_text_decoded"] is False


def test_confirmation_cannot_be_selected_by_ordinal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    confirmation_indexes = {
        _record_index(sample.sample_id)
        for sample in fixture.population.partitions["confirmation"].samples
    }
    forbidden = {
        f"synthetic private gold {index}".encode("utf-8")
        for index in confirmation_indexes
    }
    original = analysis_module.parse_json_bytes

    def guard(payload: bytes, label: str):
        assert all(secret not in payload for secret in forbidden)
        return original(payload, label)

    monkeypatch.setattr(analysis_module, "parse_json_bytes", guard)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(FirebreakError, match="outside the locked pool"):
        export_analysis_scoring_label(
            dataset_path=fixture.dataset,
            split_manifest_path=fixture.split,
            output_path=output,
            sample_ordinal=len(fixture.population.role_samples("analysis")),
            expected_question_probe_sha256=fixture.question_probe_sha256,
            expected=fixture.expected,
        )
    assert not output.exists()


def test_export_rejects_probe_mismatch_and_refuses_to_clobber(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    mismatch = tmp_path / "mismatch.json"
    with pytest.raises(FirebreakError, match="another frozen probe"):
        export_analysis_scoring_label(
            dataset_path=fixture.dataset,
            split_manifest_path=fixture.split,
            output_path=mismatch,
            sample_ordinal=fixture.sample_ordinal,
            expected_question_probe_sha256="f" * 64,
            expected=fixture.expected,
        )
    assert not mismatch.exists()

    output = tmp_path / "existing.json"
    output.write_bytes(b"do not replace")
    with pytest.raises(FirebreakError, match="refusing to overwrite"):
        _export(fixture, output)
    assert output.read_bytes() == b"do not replace"


def test_loader_rejects_extra_fields_even_with_rebased_file_hash(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "label.json"
    receipt = _export(fixture, output)
    value = json.loads(output.read_text(encoding="utf-8"))
    value["label"]["second_record"] = {
        "gold_answer": "must never be accepted"
    }
    output.write_bytes(canonical_json_bytes(value) + b"\n")

    with pytest.raises(FirebreakError, match="non-closed schema"):
        _load(
            output,
            receipt,
            expected_file_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        )


def test_loader_rejects_reordered_ordinal_even_with_rebased_file_hash(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "label.json"
    receipt = _export(fixture, output)
    value = json.loads(output.read_text(encoding="utf-8"))
    value["sample_ordinal"] += 1
    output.write_bytes(canonical_json_bytes(value) + b"\n")

    with pytest.raises(FirebreakError, match="another sample ordinal"):
        _load(
            output,
            receipt,
            expected_file_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ("expected_label_record_sha256", "record identity differs"),
        ("expected_dataset_sha256", "another dataset"),
        ("expected_split_manifest_sha256", "another split"),
        (
            "expected_analysis_ordered_question_ids_sha256",
            "another analysis order",
        ),
        ("expected_sample_id_sha256", "another sample"),
        ("expected_question_id_sha256", "another question ID"),
        ("expected_question_text_sha256", "another question text"),
        ("expected_question_probe_sha256", "another frozen probe"),
        ("expected_raw_record_sha256", "another raw record"),
        ("expected_raw_record_span_sha256", "another raw record span"),
    ],
)
def test_loader_rejects_every_external_identity_mismatch(
    tmp_path: Path,
    argument: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "label.json"
    receipt = _export(fixture, output)
    with pytest.raises(FirebreakError, match=message):
        _load(output, receipt, **{argument: "f" * 64})


def test_loader_requires_canonical_json_even_when_file_hash_is_rebased(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "label.json"
    receipt = _export(fixture, output)
    value = json.loads(output.read_text(encoding="utf-8"))
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")

    with pytest.raises(FirebreakError, match="not canonical JSON"):
        _load(
            output,
            receipt,
            expected_file_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        )


def test_loader_file_hash_gate_precedes_label_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "label.json"
    receipt = _export(fixture, output)

    def forbidden_decode(*_args, **_kwargs):
        raise AssertionError("wrong-hash label must not be decoded")

    monkeypatch.setattr(scoring_module, "parse_json_bytes", forbidden_decode)
    with pytest.raises(FirebreakError, match="differs from its receipt"):
        _load(output, receipt, expected_file_sha256="f" * 64)
