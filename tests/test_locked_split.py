from __future__ import annotations

from pathlib import Path

import pytest

from memory_condense.eval.locked_split import (
    LockedSplitManifest,
    file_sha256,
    select_locked_split,
)
from memory_condense.loader import BenchmarkQuestion, BenchmarkSample


def _samples(n: int) -> list[BenchmarkSample]:
    return [
        BenchmarkSample(
            sample_id=f"sample-{i}",
            turns=[("user", f"turn {i}")],
            questions=[
                BenchmarkQuestion(
                    question_id=f"q-{i}", question="question", answer="answer"
                )
            ],
        )
        for i in range(n)
    ]


def test_locked_splits_are_deterministic_disjoint_and_complete(tmp_path: Path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("fixture", encoding="utf-8")
    manifest = LockedSplitManifest(
        dataset_sha256=file_sha256(dataset),
        salt="fixed",
        splits={"development": 4, "validation": 2, "confirmation": 4},
    )
    samples = _samples(10)

    partitions = {
        name: select_locked_split(
            list(reversed(samples)),
            dataset_path=dataset,
            manifest=manifest,
            split=name,
        )
        for name in manifest.splits
    }
    ids = [{sample.sample_id for sample in part} for part in partitions.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set().union(*ids) == {sample.sample_id for sample in samples}

    repeated = select_locked_split(
        samples,
        dataset_path=dataset,
        manifest=manifest,
        split="development",
    )
    assert [sample.sample_id for sample in repeated] == [
        sample.sample_id for sample in partitions["development"]
    ]


def test_locked_split_rejects_dataset_drift(tmp_path: Path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("changed", encoding="utf-8")
    manifest = LockedSplitManifest(
        dataset_sha256="0" * 64,
        salt="fixed",
        splits={"development": 1},
    )
    with pytest.raises(ValueError, match="SHA-256"):
        select_locked_split(
            _samples(1),
            dataset_path=dataset,
            manifest=manifest,
            split="development",
        )


def test_locked_split_requires_full_population(tmp_path: Path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("fixture", encoding="utf-8")
    manifest = LockedSplitManifest(
        dataset_sha256=file_sha256(dataset),
        salt="fixed",
        splits={"development": 1, "confirmation": 1},
    )
    with pytest.raises(ValueError, match="population"):
        select_locked_split(
            _samples(3),
            dataset_path=dataset,
            manifest=manifest,
            split="development",
        )
