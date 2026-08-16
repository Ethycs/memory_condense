"""Deterministic, dataset-hash-verified benchmark splits.

The manifest locks the population and allocation algorithm before retrieval
tuning. It deliberately contains no questions or answers: samples are ordered
by a salted hash of their public ID, then sliced into named partitions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from memory_condense.loader import BenchmarkSample


class LockedSplitManifest(BaseModel):
    format: str = "memory-condense-locked-benchmark-split-v1"
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    salt: str = Field(min_length=1)
    splits: dict[str, int]
    algorithm: str = "stratified-largest-remainder-v1"

    model_config = {"frozen": True}

    def model_post_init(self, __context) -> None:
        if not self.splits or any(count < 1 for count in self.splits.values()):
            raise ValueError("locked split counts must all be positive")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split_manifest(path: str | Path) -> LockedSplitManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return LockedSplitManifest.model_validate(payload)


def select_locked_split(
    samples: list[BenchmarkSample],
    *,
    dataset_path: str | Path,
    manifest: LockedSplitManifest,
    split: str,
) -> list[BenchmarkSample]:
    """Validate the dataset and return one non-overlapping locked partition."""

    actual_hash = file_sha256(dataset_path)
    if actual_hash != manifest.dataset_sha256:
        raise ValueError(
            "benchmark dataset SHA-256 does not match the locked split manifest: "
            f"expected {manifest.dataset_sha256}, got {actual_hash}"
        )
    if split not in manifest.splits:
        choices = ", ".join(manifest.splits)
        raise ValueError(f"unknown locked split {split!r}; choose one of {choices}")
    if sum(manifest.splits.values()) != len(samples):
        raise ValueError(
            "locked split counts do not cover the parsed benchmark population"
        )

    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark sample IDs must be unique for locked splitting")
    if manifest.algorithm != "stratified-largest-remainder-v1":
        raise ValueError(f"unsupported locked split algorithm: {manifest.algorithm}")

    # LongMemEval has one question per sample. For a general multi-question
    # sample, the sorted set of categories is its stable stratum label.
    strata: dict[str, list[BenchmarkSample]] = {}
    for sample in samples:
        categories = sorted(
            {question.category or "uncategorized" for question in sample.questions}
        )
        stratum = "|".join(categories) or "uncategorized"
        strata.setdefault(stratum, []).append(sample)

    split_names = list(manifest.splits)
    population = len(samples)
    quotas: dict[str, dict[str, int]] = {}
    remainders: dict[str, dict[str, float]] = {}
    column_assigned = {name: 0 for name in split_names}
    row_leftovers: dict[str, int] = {}
    for stratum, members in strata.items():
        quotas[stratum] = {}
        remainders[stratum] = {}
        for name in split_names:
            ideal = len(members) * manifest.splits[name] / population
            base = int(ideal)
            quotas[stratum][name] = base
            remainders[stratum][name] = ideal - base
            column_assigned[name] += base
        row_leftovers[stratum] = len(members) - sum(quotas[stratum].values())

    column_deficit = {
        name: manifest.splits[name] - column_assigned[name] for name in split_names
    }
    for stratum in sorted(strata):
        used_for_remainder: set[str] = set()
        for _ in range(row_leftovers[stratum]):
            choices = [name for name in split_names if column_deficit[name] > 0]
            if not choices:
                raise AssertionError("no split capacity remains during apportionment")
            unused = [name for name in choices if name not in used_for_remainder]
            pool = unused or choices
            name = max(
                pool,
                key=lambda candidate: (
                    remainders[stratum][candidate],
                    column_deficit[candidate],
                    -split_names.index(candidate),
                ),
            )
            quotas[stratum][name] += 1
            column_deficit[name] -= 1
            used_for_remainder.add(name)
    if any(column_deficit.values()):
        raise AssertionError("stratified apportionment did not fill every split")

    partitions: dict[str, list[BenchmarkSample]] = {name: [] for name in split_names}
    for stratum in sorted(strata):
        ordered = sorted(
            strata[stratum],
            key=lambda sample: hashlib.sha256(
                f"{manifest.salt}\0{stratum}\0{sample.sample_id}".encode("utf-8")
            ).digest(),
        )
        offset = 0
        for name in split_names:
            count = quotas[stratum][name]
            partitions[name].extend(ordered[offset : offset + count])
            offset += count
    # Membership is stratified; presentation order is independently salted so
    # ``--max-samples`` smokes do not consume one category block at a time.
    return sorted(
        partitions[split],
        key=lambda sample: hashlib.sha256(
            f"{manifest.salt}\0order\0{sample.sample_id}".encode("utf-8")
        ).digest(),
    )
