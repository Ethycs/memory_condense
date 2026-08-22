"""Gold-blind population contracts for locked 1M cumulative campaigns.

The original cumulative launcher is intentionally tied to one development
concatenation.  This module provides the population layer needed by a later
sharded launcher without weakening that historical contract: one function
reconstructs any hash-locked split/offset shard, while the merger replays the
source construction and accepts only the exact ordered ten-shard population.

Exported identities contain corpus and question-probe digests, never answers,
answer-source labels, evidence labels, or question categories.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.context_stress import (
    compose_context_stress_sample,
    transcript_tokens,
)
from memory_condense.eval.locked_split import load_split_manifest, select_locked_split
from memory_condense.ingest.loader import BenchmarkSample, load_benchmark


LOCKED_LONGMEMEVAL_DATASET_SHA256 = (
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
)
LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256 = (
    "8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4"
)
LOCKED_CONTEXT_TARGET_TOKENS = 1_000_000
LOCKED_QUESTIONS_PER_SHARD = 10
LOCKED_100Q_OFFSETS = tuple(range(0, 100, LOCKED_QUESTIONS_PER_SHARD))

SHARD_IDENTITY_FORMAT = (
    "memory-condense-locked-cumulative-1m-shard-population-v1"
)
POPULATION_IDENTITY_FORMAT = (
    "memory-condense-locked-cumulative-1m-100q-population-v1"
)
QUESTION_PROBE_FORMAT = "memory-condense-gold-blind-question-probe-v1"

_SHA256_ALPHABET = frozenset("0123456789abcdef")
_SHARD_FIELDS = frozenset(
    {
        "format",
        "benchmark_format",
        "dataset_sha256",
        "split_manifest_sha256",
        "split",
        "construction",
        "sample_id_sha256",
        "gold_blind_corpus_sha256",
        "transcript_tokens",
        "turn_count",
        "source_count",
        "question_count",
        "ordered_question_probes",
        "gold_fields_present",
        "shard_identity_sha256",
    }
)
_CONSTRUCTION_FIELDS = frozenset(
    {"target_tokens", "questions_per_shard", "sample_offset"}
)
_PROBE_FIELDS = frozenset(
    {
        "format",
        "ordinal",
        "question_id_sha256",
        "retrieval_query_sha256",
        "prompt_question_sha256",
        "probe_identity_sha256",
    }
)
_POPULATION_FIELDS = frozenset(
    {
        "format",
        "benchmark_format",
        "dataset_sha256",
        "split_manifest_sha256",
        "split",
        "construction",
        "shard_count",
        "question_count",
        "total_transcript_tokens",
        "total_turn_count",
        "ordered_shard_identity_sha256s",
        "ordered_question_id_sha256s",
        "ordered_question_probe_sha256s",
        "gold_fields_present",
        "population_identity_sha256",
    }
)
_POPULATION_CONSTRUCTION_FIELDS = frozenset(
    {"target_tokens", "questions_per_shard", "shard_offsets"}
)


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_ALPHABET for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_exact_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} has an unexpected schema")


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    body = {key: child for key, child in value.items() if key != field}
    return identity_sha256(body)


@dataclass(frozen=True, slots=True)
class LockedCumulativePopulationPlan:
    """Exact source and construction controls for one ordered 100Q campaign."""

    dataset_sha256: str
    split_manifest_sha256: str
    split: str
    target_tokens: int = LOCKED_CONTEXT_TARGET_TOKENS
    questions_per_shard: int = LOCKED_QUESTIONS_PER_SHARD
    shard_offsets: tuple[int, ...] = LOCKED_100Q_OFFSETS
    benchmark_format: str = "longmemeval"

    def __post_init__(self) -> None:
        _require_sha256(self.dataset_sha256, "plan.dataset_sha256")
        _require_sha256(
            self.split_manifest_sha256,
            "plan.split_manifest_sha256",
        )
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("plan.split must be a non-empty string")
        if self.benchmark_format != "longmemeval":
            raise ValueError("cumulative population requires LongMemEval")
        _require_exact_int(self.target_tokens, "plan.target_tokens", minimum=1)
        questions = _require_exact_int(
            self.questions_per_shard,
            "plan.questions_per_shard",
            minimum=1,
        )
        if questions != LOCKED_QUESTIONS_PER_SHARD:
            raise ValueError("locked cumulative shards must contain ten questions")
        offsets = self.shard_offsets
        if not isinstance(offsets, tuple) or offsets != LOCKED_100Q_OFFSETS:
            raise ValueError(
                "locked 100Q population requires exact offsets 0,10,...,90"
            )
        if len(offsets) * questions != 100:
            raise ValueError("locked population must contain exactly 100 questions")


LOCKED_LONGMEMEVAL_VALIDATION_PLAN = LockedCumulativePopulationPlan(
    dataset_sha256=LOCKED_LONGMEMEVAL_DATASET_SHA256,
    split_manifest_sha256=LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
    split="validation",
)


def _require_locked_sources(
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    *,
    dataset_sha256: str,
    split_manifest_sha256: str,
) -> tuple[Path, Path]:
    dataset = Path(dataset_path).resolve()
    split = Path(split_manifest_path).resolve()
    expected_dataset = _require_sha256(dataset_sha256, "dataset_sha256")
    expected_split = _require_sha256(
        split_manifest_sha256,
        "split_manifest_sha256",
    )
    if not dataset.is_file():
        raise FileNotFoundError(f"locked LongMemEval dataset is missing: {dataset}")
    if not split.is_file():
        raise FileNotFoundError(f"locked split manifest is missing: {split}")
    observed_dataset = file_sha256(dataset)
    if observed_dataset != expected_dataset:
        raise ValueError(
            "LongMemEval dataset SHA-256 mismatch: "
            f"{observed_dataset} != {expected_dataset}"
        )
    observed_split = file_sha256(split)
    if observed_split != expected_split:
        raise ValueError(
            "locked split-manifest SHA-256 mismatch: "
            f"{observed_split} != {expected_split}"
        )
    return dataset, split


def _load_locked_partition(
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    *,
    dataset_sha256: str,
    split_manifest_sha256: str,
    split_name: str,
) -> tuple[list[BenchmarkSample], Path, Path]:
    dataset, split = _require_locked_sources(
        dataset_path,
        split_manifest_path,
        dataset_sha256=dataset_sha256,
        split_manifest_sha256=split_manifest_sha256,
    )
    samples = load_benchmark(dataset, "longmemeval")
    selected = select_locked_split(
        samples,
        dataset_path=dataset,
        manifest=load_split_manifest(split),
        split=split_name,
    )
    return selected, dataset, split


def _question_probe(question: Any, ordinal: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "format": QUESTION_PROBE_FORMAT,
        "ordinal": ordinal,
        "question_id_sha256": identity_sha256(
            {"question_id": str(question.question_id)}
        ),
        "retrieval_query_sha256": quote_sha256(str(question.question)),
        "prompt_question_sha256": quote_sha256(str(question.dated_question)),
    }
    body["probe_identity_sha256"] = identity_sha256(body)
    return body


def _gold_blind_corpus_sha256(sample: BenchmarkSample) -> str:
    source_ids = sample.turn_source_ids or [None] * len(sample.turns)
    timestamps = sample.turn_created_at or [None] * len(sample.turns)
    if len(source_ids) != len(sample.turns) or len(timestamps) != len(sample.turns):
        raise ValueError("population turn provenance is misaligned")
    return identity_sha256(
        {
            "sample_id_sha256": identity_sha256(
                {"sample_id": sample.sample_id}
            ),
            "turns": [
                {
                    "ordinal": ordinal,
                    "role": role,
                    "text_sha256": quote_sha256(text),
                    "source_id_sha256": (
                        None
                        if source_id is None
                        else identity_sha256({"source_id": source_id})
                    ),
                    "created_at": (
                        None
                        if created_at is None
                        else created_at.isoformat()
                    ),
                }
                for ordinal, ((role, text), source_id, created_at) in enumerate(
                    zip(sample.turns, source_ids, timestamps, strict=True)
                )
            ],
        }
    )


def _shard_identity(
    sample: BenchmarkSample,
    *,
    dataset_sha256: str,
    split_manifest_sha256: str,
    split_name: str,
    sample_offset: int,
    target_tokens: int,
    questions_per_shard: int,
) -> dict[str, Any]:
    if len(sample.questions) != questions_per_shard:
        raise ValueError(
            f"locked shard has {len(sample.questions)} questions; "
            f"expected {questions_per_shard}"
        )
    sources = sample.turn_source_ids or [None] * len(sample.turns)
    body: dict[str, Any] = {
        "format": SHARD_IDENTITY_FORMAT,
        "benchmark_format": "longmemeval",
        "dataset_sha256": dataset_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "split": split_name,
        "construction": {
            "target_tokens": target_tokens,
            "questions_per_shard": questions_per_shard,
            "sample_offset": sample_offset,
        },
        "sample_id_sha256": identity_sha256({"sample_id": sample.sample_id}),
        "gold_blind_corpus_sha256": _gold_blind_corpus_sha256(sample),
        "transcript_tokens": transcript_tokens(sample),
        "turn_count": len(sample.turns),
        "source_count": len(
            {source_id for source_id in sources if source_id is not None}
        ),
        "question_count": len(sample.questions),
        "ordered_question_probes": [
            _question_probe(question, ordinal)
            for ordinal, question in enumerate(sample.questions)
        ],
        "gold_fields_present": False,
    }
    body["shard_identity_sha256"] = identity_sha256(body)
    validate_locked_cumulative_shard_identity(body)
    return body


def _reconstruct_from_partition(
    selected: list[BenchmarkSample],
    *,
    dataset_sha256: str,
    split_manifest_sha256: str,
    split_name: str,
    sample_offset: int,
    target_tokens: int,
    questions_per_shard: int,
) -> tuple[BenchmarkSample, dict[str, Any]]:
    offset = _require_exact_int(sample_offset, "sample_offset")
    if offset >= len(selected):
        raise ValueError(
            f"sample offset {offset} is outside locked split of {len(selected)}"
        )
    sample = compose_context_stress_sample(
        selected[offset:],
        target_tokens=target_tokens,
        max_questions=questions_per_shard,
    )
    identity = _shard_identity(
        sample,
        dataset_sha256=dataset_sha256,
        split_manifest_sha256=split_manifest_sha256,
        split_name=split_name,
        sample_offset=offset,
        target_tokens=target_tokens,
        questions_per_shard=questions_per_shard,
    )
    return sample, identity


def reconstruct_locked_cumulative_shard(
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    *,
    dataset_sha256: str,
    split_manifest_sha256: str,
    split_name: str,
    sample_offset: int,
    target_tokens: int = LOCKED_CONTEXT_TARGET_TOKENS,
    questions_per_shard: int = LOCKED_QUESTIONS_PER_SHARD,
) -> tuple[BenchmarkSample, dict[str, Any]]:
    """Reconstruct one exact split/offset shard and its gold-blind identity."""

    _require_exact_int(target_tokens, "target_tokens", minimum=1)
    questions = _require_exact_int(
        questions_per_shard,
        "questions_per_shard",
        minimum=1,
    )
    if questions != LOCKED_QUESTIONS_PER_SHARD:
        raise ValueError("locked cumulative shards must contain ten questions")
    selected, _dataset, _split = _load_locked_partition(
        dataset_path,
        split_manifest_path,
        dataset_sha256=dataset_sha256,
        split_manifest_sha256=split_manifest_sha256,
        split_name=split_name,
    )
    return _reconstruct_from_partition(
        selected,
        dataset_sha256=dataset_sha256,
        split_manifest_sha256=split_manifest_sha256,
        split_name=split_name,
        sample_offset=sample_offset,
        target_tokens=target_tokens,
        questions_per_shard=questions,
    )


def validate_locked_cumulative_shard_identity(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one hostile shard identity without consulting gold fields."""

    if not isinstance(value, Mapping):
        raise TypeError("shard identity must be a mapping")
    shard = dict(value)
    _require_exact_keys(shard, _SHARD_FIELDS, "shard identity")
    if shard["format"] != SHARD_IDENTITY_FORMAT:
        raise ValueError("unexpected shard identity format")
    if shard["benchmark_format"] != "longmemeval":
        raise ValueError("shard identity changed benchmark format")
    _require_sha256(shard["dataset_sha256"], "shard.dataset_sha256")
    _require_sha256(
        shard["split_manifest_sha256"],
        "shard.split_manifest_sha256",
    )
    if not isinstance(shard["split"], str) or not shard["split"]:
        raise ValueError("shard split must be a non-empty string")
    construction = shard["construction"]
    if not isinstance(construction, Mapping):
        raise ValueError("shard construction must be an object")
    _require_exact_keys(
        construction,
        _CONSTRUCTION_FIELDS,
        "shard construction",
    )
    _require_exact_int(construction["target_tokens"], "target_tokens", minimum=1)
    if (
        _require_exact_int(
            construction["questions_per_shard"],
            "questions_per_shard",
            minimum=1,
        )
        != LOCKED_QUESTIONS_PER_SHARD
    ):
        raise ValueError("shard does not bind ten questions")
    _require_exact_int(construction["sample_offset"], "sample_offset")
    for name in ("sample_id_sha256", "gold_blind_corpus_sha256"):
        _require_sha256(shard[name], f"shard.{name}")
    _require_exact_int(shard["transcript_tokens"], "transcript_tokens", minimum=1)
    _require_exact_int(shard["turn_count"], "turn_count", minimum=1)
    _require_exact_int(shard["source_count"], "source_count", minimum=1)
    if shard["question_count"] != LOCKED_QUESTIONS_PER_SHARD:
        raise ValueError("shard question count changed")
    probes = shard["ordered_question_probes"]
    if not isinstance(probes, list) or len(probes) != LOCKED_QUESTIONS_PER_SHARD:
        raise ValueError("shard question-probe population changed")
    seen: set[str] = set()
    for ordinal, raw_probe in enumerate(probes):
        if not isinstance(raw_probe, Mapping):
            raise ValueError("question probe must be an object")
        probe = dict(raw_probe)
        _require_exact_keys(probe, _PROBE_FIELDS, "question probe")
        if probe["format"] != QUESTION_PROBE_FORMAT or probe["ordinal"] != ordinal:
            raise ValueError("question probe order or format changed")
        for name in (
            "question_id_sha256",
            "retrieval_query_sha256",
            "prompt_question_sha256",
            "probe_identity_sha256",
        ):
            _require_sha256(probe[name], f"question_probe.{name}")
        if probe["probe_identity_sha256"] != _self_hash(
            probe,
            "probe_identity_sha256",
        ):
            raise ValueError("question probe self-hash changed")
        question_id = str(probe["question_id_sha256"])
        if question_id in seen:
            raise ValueError("shard repeats a question identity")
        seen.add(question_id)
    if shard["gold_fields_present"] is not False:
        raise ValueError("shard identity must explicitly exclude gold fields")
    _require_sha256(
        shard["shard_identity_sha256"],
        "shard.shard_identity_sha256",
    )
    if shard["shard_identity_sha256"] != _self_hash(
        shard,
        "shard_identity_sha256",
    ):
        raise ValueError("shard identity self-hash changed")
    return shard


def _merge_verified_shards(
    shards: Sequence[Mapping[str, Any]],
    *,
    plan: LockedCumulativePopulationPlan,
) -> dict[str, Any]:
    if len(shards) != len(plan.shard_offsets):
        raise ValueError("locked population requires exactly ten shard identities")
    normalized = [validate_locked_cumulative_shard_identity(row) for row in shards]
    question_ids: set[str] = set()
    ordered_question_ids: list[str] = []
    probe_hashes: list[str] = []
    for expected_offset, shard in zip(plan.shard_offsets, normalized, strict=True):
        if (
            shard["dataset_sha256"] != plan.dataset_sha256
            or shard["split_manifest_sha256"] != plan.split_manifest_sha256
            or shard["split"] != plan.split
            or shard["benchmark_format"] != plan.benchmark_format
            or shard["construction"]
            != {
                "target_tokens": plan.target_tokens,
                "questions_per_shard": plan.questions_per_shard,
                "sample_offset": expected_offset,
            }
        ):
            raise ValueError("shard identity differs from the ordered population plan")
        for probe in shard["ordered_question_probes"]:
            question_id = str(probe["question_id_sha256"])
            if question_id in question_ids:
                raise ValueError("locked population repeats a question across shards")
            question_ids.add(question_id)
            ordered_question_ids.append(question_id)
            probe_hashes.append(str(probe["probe_identity_sha256"]))
    if len(question_ids) != 100 or len(probe_hashes) != 100:
        raise ValueError("locked merged population must contain 100 unique questions")
    body: dict[str, Any] = {
        "format": POPULATION_IDENTITY_FORMAT,
        "benchmark_format": plan.benchmark_format,
        "dataset_sha256": plan.dataset_sha256,
        "split_manifest_sha256": plan.split_manifest_sha256,
        "split": plan.split,
        "construction": {
            "target_tokens": plan.target_tokens,
            "questions_per_shard": plan.questions_per_shard,
            "shard_offsets": list(plan.shard_offsets),
        },
        "shard_count": len(normalized),
        "question_count": len(probe_hashes),
        "total_transcript_tokens": sum(
            int(shard["transcript_tokens"]) for shard in normalized
        ),
        "total_turn_count": sum(int(shard["turn_count"]) for shard in normalized),
        "ordered_shard_identity_sha256s": [
            str(shard["shard_identity_sha256"]) for shard in normalized
        ],
        "ordered_question_id_sha256s": ordered_question_ids,
        "ordered_question_probe_sha256s": probe_hashes,
        "gold_fields_present": False,
    }
    body["population_identity_sha256"] = identity_sha256(body)
    validate_locked_cumulative_population_identity(body, plan=plan)
    return body


def merge_locked_cumulative_shard_identities(
    shard_identities: Sequence[Mapping[str, Any]],
    *,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    plan: LockedCumulativePopulationPlan = LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
) -> dict[str, Any]:
    """Reconstruct and merge only the exact ordered shards in ``plan``.

    Self-hashes are necessary but not sufficient: every supplied identity is
    compared byte-for-byte as a value with a fresh reconstruction from the
    hash-locked dataset and split manifest.
    """

    selected, _dataset, _split = _load_locked_partition(
        dataset_path,
        split_manifest_path,
        dataset_sha256=plan.dataset_sha256,
        split_manifest_sha256=plan.split_manifest_sha256,
        split_name=plan.split,
    )
    if len(selected) != 100:
        raise ValueError(
            "locked 100Q population plan requires a split of exactly 100 samples"
        )
    expected = [
        _reconstruct_from_partition(
            selected,
            dataset_sha256=plan.dataset_sha256,
            split_manifest_sha256=plan.split_manifest_sha256,
            split_name=plan.split,
            sample_offset=offset,
            target_tokens=plan.target_tokens,
            questions_per_shard=plan.questions_per_shard,
        )[1]
        for offset in plan.shard_offsets
    ]
    if len(shard_identities) != len(expected):
        raise ValueError("locked population requires exactly ten shard identities")
    observed = [
        validate_locked_cumulative_shard_identity(value)
        for value in shard_identities
    ]
    for ordinal, (actual, reconstructed) in enumerate(
        zip(observed, expected, strict=True)
    ):
        if actual != reconstructed:
            raise ValueError(
                f"shard identity {ordinal} differs from locked reconstruction"
            )
    return _merge_verified_shards(observed, plan=plan)


def build_locked_cumulative_population_identity(
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    *,
    plan: LockedCumulativePopulationPlan = LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
) -> tuple[tuple[BenchmarkSample, ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    """Reconstruct all ten shards once and return their strict 100Q identity."""

    selected, _dataset, _split = _load_locked_partition(
        dataset_path,
        split_manifest_path,
        dataset_sha256=plan.dataset_sha256,
        split_manifest_sha256=plan.split_manifest_sha256,
        split_name=plan.split,
    )
    if len(selected) != 100:
        raise ValueError(
            "locked 100Q population plan requires a split of exactly 100 samples"
        )
    reconstructed = [
        _reconstruct_from_partition(
            selected,
            dataset_sha256=plan.dataset_sha256,
            split_manifest_sha256=plan.split_manifest_sha256,
            split_name=plan.split,
            sample_offset=offset,
            target_tokens=plan.target_tokens,
            questions_per_shard=plan.questions_per_shard,
        )
        for offset in plan.shard_offsets
    ]
    samples = tuple(sample for sample, _identity in reconstructed)
    identities = tuple(identity for _sample, identity in reconstructed)
    population = _merge_verified_shards(identities, plan=plan)
    return samples, identities, population


def validate_locked_cumulative_population_identity(
    value: Mapping[str, Any],
    *,
    plan: LockedCumulativePopulationPlan,
) -> dict[str, Any]:
    """Validate the closed-schema aggregate identity against its exact plan."""

    if not isinstance(value, Mapping):
        raise TypeError("population identity must be a mapping")
    population = dict(value)
    _require_exact_keys(population, _POPULATION_FIELDS, "population identity")
    if population["format"] != POPULATION_IDENTITY_FORMAT:
        raise ValueError("unexpected population identity format")
    expected_head = {
        "benchmark_format": plan.benchmark_format,
        "dataset_sha256": plan.dataset_sha256,
        "split_manifest_sha256": plan.split_manifest_sha256,
        "split": plan.split,
    }
    if any(population[name] != expected for name, expected in expected_head.items()):
        raise ValueError("population identity differs from its source plan")
    construction = population["construction"]
    if not isinstance(construction, Mapping):
        raise ValueError("population construction must be an object")
    _require_exact_keys(
        construction,
        _POPULATION_CONSTRUCTION_FIELDS,
        "population construction",
    )
    if dict(construction) != {
        "target_tokens": plan.target_tokens,
        "questions_per_shard": plan.questions_per_shard,
        "shard_offsets": list(plan.shard_offsets),
    }:
        raise ValueError("population construction differs from its exact plan")
    if population["shard_count"] != 10 or population["question_count"] != 100:
        raise ValueError("population must bind ten shards and 100 questions")
    _require_exact_int(
        population["total_transcript_tokens"],
        "total_transcript_tokens",
        minimum=1,
    )
    _require_exact_int(
        population["total_turn_count"],
        "total_turn_count",
        minimum=1,
    )
    shard_hashes = population["ordered_shard_identity_sha256s"]
    question_id_hashes = population["ordered_question_id_sha256s"]
    probe_hashes = population["ordered_question_probe_sha256s"]
    if not isinstance(shard_hashes, list) or len(shard_hashes) != 10:
        raise ValueError("population changed its ordered shard identities")
    if not isinstance(probe_hashes, list) or len(probe_hashes) != 100:
        raise ValueError("population changed its ordered question probes")
    if not isinstance(question_id_hashes, list) or len(question_id_hashes) != 100:
        raise ValueError("population changed its ordered question identities")
    for digest in [*shard_hashes, *question_id_hashes, *probe_hashes]:
        _require_sha256(digest, "population member identity")
    if (
        len(set(shard_hashes)) != 10
        or len(set(question_id_hashes)) != 100
        or len(set(probe_hashes)) != 100
    ):
        raise ValueError("population member identities must be unique")
    if population["gold_fields_present"] is not False:
        raise ValueError("population identity must explicitly exclude gold fields")
    _require_sha256(
        population["population_identity_sha256"],
        "population.population_identity_sha256",
    )
    if population["population_identity_sha256"] != _self_hash(
        population,
        "population_identity_sha256",
    ):
        raise ValueError("population identity self-hash changed")
    return population


__all__ = [
    "LOCKED_100Q_OFFSETS",
    "LOCKED_CONTEXT_TARGET_TOKENS",
    "LOCKED_LONGMEMEVAL_DATASET_SHA256",
    "LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256",
    "LOCKED_LONGMEMEVAL_VALIDATION_PLAN",
    "LOCKED_QUESTIONS_PER_SHARD",
    "LockedCumulativePopulationPlan",
    "POPULATION_IDENTITY_FORMAT",
    "QUESTION_PROBE_FORMAT",
    "SHARD_IDENTITY_FORMAT",
    "build_locked_cumulative_population_identity",
    "merge_locked_cumulative_shard_identities",
    "reconstruct_locked_cumulative_shard",
    "validate_locked_cumulative_population_identity",
    "validate_locked_cumulative_shard_identity",
]
