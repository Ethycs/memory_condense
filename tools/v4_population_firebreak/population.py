"""Pure reconstruction of the locked LongMemEval-S population."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .canonical import (
    FirebreakError,
    FileSnapshot,
    canonical_sha256,
    exact_keys,
    parse_json_bytes,
    require_int,
    require_mapping,
    require_text,
)


_ROLE_MAP = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "bot": "assistant",
    "system": "assistant",
}
_LONGMEMEVAL_WEEKDAY_RE = re.compile(r"\([^)]*\)")


@dataclass(frozen=True, slots=True)
class PopulationSample:
    sample_id: str
    category: str
    normalized_sha256: str
    raw_record_sha256: str
    treatment_projection: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Partition:
    name: str
    samples: tuple[PopulationSample, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(sample.sample_id for sample in self.samples)

    @property
    def ordered_ids_sha256(self) -> str:
        return canonical_sha256(list(self.ids))

    @property
    def ordered_normalized_bindings_sha256(self) -> str:
        return canonical_sha256(
            [[sample.sample_id, sample.normalized_sha256] for sample in self.samples]
        )

    @property
    def ordered_raw_bindings_sha256(self) -> str:
        return canonical_sha256(
            [[sample.sample_id, sample.raw_record_sha256] for sample in self.samples]
        )

    @property
    def category_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(sample.category for sample in self.samples).items()))

    def receipt(self) -> dict[str, Any]:
        return {
            "count": len(self.samples),
            "ordered_question_ids_sha256": self.ordered_ids_sha256,
            "ordered_normalized_sample_bindings_sha256": (
                self.ordered_normalized_bindings_sha256
            ),
            "ordered_raw_record_bindings_sha256": self.ordered_raw_bindings_sha256,
            "category_counts_sha256": canonical_sha256(self.category_counts),
            "category_count": len(self.category_counts),
        }


@dataclass(frozen=True, slots=True)
class Population:
    dataset_sha256: str
    dataset_bytes: int
    split_manifest_sha256: str
    split_format: str
    split_algorithm: str
    split_salt: str
    partitions: dict[str, Partition]

    def role_samples(self, role: str) -> tuple[PopulationSample, ...]:
        if role == "analysis":
            return (
                self.partitions["development"].samples
                + self.partitions["validation"].samples
            )
        if role == "confirmation":
            return self.partitions["confirmation"].samples
        raise FirebreakError("unknown treatment-input role")

    def role_ids_sha256(self, role: str) -> str:
        return canonical_sha256(
            [sample.sample_id for sample in self.role_samples(role)]
        )


def reconstruct_population(
    dataset: FileSnapshot,
    split_manifest: FileSnapshot,
) -> Population:
    raw_manifest = require_mapping(
        parse_json_bytes(split_manifest.payload, "split manifest"),
        "split manifest",
    )
    exact_keys(
        raw_manifest,
        {"format", "dataset_sha256", "salt", "algorithm", "splits"},
        "split manifest",
    )
    if require_text(raw_manifest["dataset_sha256"], "split dataset SHA-256") != dataset.sha256:
        raise FirebreakError("split manifest does not bind the dataset snapshot")
    split_format = require_text(raw_manifest["format"], "split format")
    split_algorithm = require_text(raw_manifest["algorithm"], "split algorithm")
    split_salt = require_text(raw_manifest["salt"], "split salt")
    if split_format != "memory-condense-locked-benchmark-split-v1":
        raise FirebreakError("unsupported split manifest format")
    if split_algorithm != "stratified-largest-remainder-v1":
        raise FirebreakError("unsupported split algorithm")
    raw_counts = require_mapping(raw_manifest["splits"], "split counts")
    split_names = list(raw_counts)
    if split_names != ["development", "validation", "confirmation"]:
        raise FirebreakError("split order or names differ from the locked protocol")
    counts = {
        name: require_int(value, f"split count {name}", minimum=1)
        for name, value in raw_counts.items()
    }
    samples = _parse_dataset(dataset.payload)
    if sum(counts.values()) != len(samples):
        raise FirebreakError("split counts do not cover the parsed population")
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise FirebreakError("benchmark sample IDs are not unique")

    strata: dict[str, list[PopulationSample]] = {}
    for sample in samples:
        strata.setdefault(sample.category, []).append(sample)
    quotas: dict[str, dict[str, int]] = {}
    remainders: dict[str, dict[str, float]] = {}
    assigned = {name: 0 for name in split_names}
    leftovers: dict[str, int] = {}
    for stratum, members in strata.items():
        quotas[stratum] = {}
        remainders[stratum] = {}
        for name in split_names:
            ideal = len(members) * counts[name] / len(samples)
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

    selected: dict[str, list[PopulationSample]] = {name: [] for name in split_names}
    for stratum in sorted(strata):
        ordered = sorted(
            strata[stratum],
            key=lambda sample: hashlib.sha256(
                f"{split_salt}\0{stratum}\0{sample.sample_id}".encode("utf-8")
            ).digest(),
        )
        offset = 0
        for name in split_names:
            count = quotas[stratum][name]
            selected[name].extend(ordered[offset : offset + count])
            offset += count
    partitions: dict[str, Partition] = {}
    for name in split_names:
        ordered = sorted(
            selected[name],
            key=lambda sample: hashlib.sha256(
                f"{split_salt}\0order\0{sample.sample_id}".encode("utf-8")
            ).digest(),
        )
        partitions[name] = Partition(name=name, samples=tuple(ordered))

    partition_sets = [set(partition.ids) for partition in partitions.values()]
    if any(
        partition_sets[left] & partition_sets[right]
        for left in range(len(partition_sets))
        for right in range(left + 1, len(partition_sets))
    ):
        raise FirebreakError("locked partitions overlap")
    if set().union(*partition_sets) != set(ids):
        raise FirebreakError("locked partitions do not cover the population")
    return Population(
        dataset_sha256=dataset.sha256,
        dataset_bytes=dataset.size,
        split_manifest_sha256=split_manifest.sha256,
        split_format=split_format,
        split_algorithm=split_algorithm,
        split_salt=split_salt,
        partitions=partitions,
    )


def _parse_dataset(payload: bytes) -> list[PopulationSample]:
    raw = parse_json_bytes(payload, "LongMemEval dataset")
    records = _records(raw)
    samples: list[PopulationSample] = []
    for index, record in enumerate(records):
        sample = _parse_record(record, index)
        if sample is not None:
            samples.append(sample)
    return samples


def _parse_record(record: Any, index: int) -> PopulationSample | None:
    """Project one record while keeping scorer labels out of treatment data.

    ``normalized_sha256`` deliberately retains the v1 normalization used by
    the production population lock.  The treatment projection follows the
    current authoritative runtime contract: sessions are ordered by parsed
    date and every flattened turn carries the exact UTC ``created_at`` value.
    """

    if not isinstance(record, dict):
        return None
    question_text = record.get("question")
    if not isinstance(question_text, str) or not question_text.strip():
        return None
    answer = _answer_text(record.get("answer"))
    if not answer:
        return None
    sample_id = str(record.get("question_id") or f"longmemeval_{index}")
    category = (
        str(record["question_type"])
        if record.get("question_type") is not None
        else "uncategorized"
    )
    legacy_turns, legacy_sources, _legacy_created_at = _flatten_sessions(
        record,
        chronological=False,
    )
    turns, sources, created_at = _flatten_sessions(record, chronological=True)
    evidence = _string_list(record.get("answer_session_ids"))
    question_date = _as_text(record.get("question_date")) or None
    normalized_question = {
        "question_id": sample_id,
        "question": question_text.strip(),
        "answer": answer,
        "category": None if record.get("question_type") is None else category,
        "evidence": evidence,
        "evidence_sources": evidence,
        "question_date": question_date,
    }
    # Keep this legacy identity stable so the existing production lock remains
    # a check of the exact historical v1 parser.  The corrected treatment
    # projection below is independently compared byte-for-byte to its export.
    normalized = {
        "sample_id": sample_id,
        "turns": legacy_turns,
        "turn_source_ids": legacy_sources,
        "questions": [normalized_question],
    }
    treatment_projection = {
        "sample_id": sample_id,
        "turns": turns,
        "turn_source_ids": sources,
        "turn_created_at": created_at,
        "questions": [
            {
                "question_id": sample_id,
                "question": question_text.strip(),
                "question_date": question_date,
            }
        ],
    }
    return PopulationSample(
        sample_id=sample_id,
        category=category,
        normalized_sha256=canonical_sha256(normalized),
        raw_record_sha256=canonical_sha256(record),
        treatment_projection=treatment_projection,
    )


def _flatten_sessions(
    record: dict[str, Any],
    *,
    chronological: bool,
) -> tuple[list[list[str]], list[str | None], list[str | None]]:
    turns: list[list[str]] = []
    sources: list[str | None] = []
    created_at: list[str | None] = []
    sessions = record.get("haystack_sessions") or []
    if not isinstance(sessions, list):
        sessions = []
    session_ids = _string_list(record.get("haystack_session_ids"))
    session_dates = _string_list(record.get("haystack_dates"))
    rows: list[tuple[int, list[Any], str, str, datetime | None]] = []
    for session_index, session in enumerate(sessions):
        if not isinstance(session, list):
            continue
        source_id = (
            session_ids[session_index]
            if session_index < len(session_ids)
            else f"session_{session_index + 1}"
        )
        session_date = (
            session_dates[session_index]
            if session_index < len(session_dates)
            else ""
        )
        rows.append(
            (
                session_index,
                session,
                source_id,
                session_date,
                _parse_longmemeval_date(session_date),
            )
        )
    if chronological:
        distant_future = datetime.max.replace(tzinfo=timezone.utc)
        rows.sort(
            key=lambda row: (
                row[4] is None,
                row[4] or distant_future,
                row[0],
            )
        )
    for _index, session, source_id, session_date, parsed_date in rows:
        timestamp = _timestamp_json(parsed_date)
        if session_date:
            turns.append(["system", f"[{source_id} took place at {session_date}]"])
            sources.append(source_id)
            created_at.append(timestamp)
        for turn_index, turn in enumerate(session):
            if not isinstance(turn, dict):
                continue
            text = _as_text(turn.get("content", turn.get("text")))
            if not text:
                continue
            turns.append([_role(turn.get("role"), turn_index), text])
            sources.append(source_id)
            created_at.append(timestamp)
    return turns, sources, created_at


def _records(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("data", "samples", "records", "questions", "instances"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return candidate
        return [value]
    return []


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                inner = part.get("text") or part.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
        return "\n".join(parts).strip()
    return ""


def _answer_text(value: Any) -> str:
    text = _as_text(value)
    if text:
        return text
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _role(value: Any, index: int) -> str:
    if isinstance(value, str):
        mapped = _ROLE_MAP.get(value.strip().lower())
        if mapped:
            return mapped
    return "user" if index % 2 == 0 else "assistant"


def _parse_longmemeval_date(value: str) -> datetime | None:
    cleaned = _LONGMEMEVAL_WEEKDAY_RE.sub(" ", value.strip())
    cleaned = " ".join(cleaned.split())
    for format_string in (
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(cleaned, format_string).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
    return None


def _timestamp_json(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
