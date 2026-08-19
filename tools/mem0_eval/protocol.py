"""Pure, provider-free construction of the locked Mem0 comparison shards.

The memory-condense validation arm evaluates ten independently composed
one-million-token samples.  A fair Mem0 arm must ingest the exact same source
histories and answer the exact same questions, while retaining Mem0's official
LongMemEval chronology and consecutive one-or-two-message add protocol.

Nothing in this module imports Mem0, starts a model, or calls a provider.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.eval.context_stress import (
    compose_context_stress_sample,
    transcript_tokens,
)
from memory_condense.eval.locked_split import (
    LockedSplitManifest,
    load_split_manifest,
)
from memory_condense.eval.sample_identity import canonical_sha256, sample_sha256
from .source_compat import BenchmarkSample, parse_longmemeval


class Mem0ComparisonProtocolError(ValueError):
    """The frozen dataset cannot be represented by the comparison protocol."""


@dataclass(frozen=True, slots=True)
class RawAddCounts:
    """Exact public Mem0 add-request accounting for one raw corpus."""

    raw_pairs: int
    skipped_empty_pairs: int
    add_requests: int
    whitespace_only_pairs: int


@dataclass(frozen=True, slots=True)
class RawStressShard:
    """One provider-free, content-addressed Mem0 comparison shard."""

    sample_offset: int
    parsed_sample: BenchmarkSample
    sample_sha256: str
    history_sample_ids: tuple[str, ...]
    raw_history_bundle: Mapping[str, Any]
    raw_history_bundle_sha256: str
    add_batches: tuple["CompositeAddBatch", ...]
    add_counts: RawAddCounts

    @property
    def question_ids(self) -> tuple[str, ...]:
        return tuple(
            question.question_id for question in self.parsed_sample.questions
        )


@dataclass(frozen=True, slots=True)
class CompositeAddBatch:
    """One exact Mem0 add request in the direct 1M comparison arm.

    Records retain their locked validation order. Sessions are sorted by date
    only within their original record, matching the official per-question
    LongMemEval runner without inventing a chronology between ten unrelated
    histories.
    """

    source_sample_id: str
    source: str
    date: str
    session_index: int
    original_session_index: int
    batch_index: int
    turn_start: int
    messages: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class LockedRawPopulation:
    """One validated parse of the locked split and its lossless raw records."""

    validation: tuple[BenchmarkSample, ...]
    raw_by_id: Mapping[str, Mapping[str, Any]]


def _read_dataset_snapshot(path: str | Path) -> bytes:
    """Read the dataset path exactly once into an immutable byte snapshot.

    Every dataset identity check and representation used by this module is
    derived from the returned ``bytes``. In particular, callers must not
    reopen ``path`` after this boundary: doing so would allow a replacement
    between normalized LongMemEval parsing and lossless raw reconstruction.
    """

    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise Mem0ComparisonProtocolError(
            f"cannot read LongMemEval dataset: {exc}"
        ) from exc


def _decode_dataset_snapshot(snapshot: bytes) -> Any:
    """Decode one already-captured dataset snapshot without filesystem I/O."""

    try:
        return json.loads(snapshot.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Mem0ComparisonProtocolError(
            f"cannot decode LongMemEval dataset snapshot: {exc}"
        ) from exc


def _raw_records_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Validate the lossless raw-record view of a decoded snapshot."""

    if not isinstance(payload, list) or not payload:
        raise Mem0ComparisonProtocolError(
            "LongMemEval dataset must be a non-empty JSON array"
        )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(payload):
        if not isinstance(value, dict):
            raise Mem0ComparisonProtocolError(
                f"LongMemEval record {index} is not an object"
            )
        question_id = value.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise Mem0ComparisonProtocolError(
                f"LongMemEval record {index} has no question_id"
            )
        question_id = question_id.strip()
        if question_id in seen:
            raise Mem0ComparisonProtocolError(
                f"duplicate LongMemEval question_id {question_id!r}"
            )
        seen.add(question_id)
        records.append(value)
    return records


def _select_locked_split_from_snapshot(
    samples: list[BenchmarkSample],
    *,
    snapshot_sha256: str,
    manifest: LockedSplitManifest,
    split: str,
) -> list[BenchmarkSample]:
    """Select a locked partition after validating the captured byte digest.

    This is the in-memory counterpart of
    :func:`memory_condense.eval.locked_split.select_locked_split`. That public
    helper hashes a path internally, so calling it here would reintroduce the
    parse/hash time-of-check-to-time-of-use race this comparison boundary must
    exclude. Keep the allocation logic equivalent to the frozen
    ``stratified-largest-remainder-v1`` implementation.
    """

    if snapshot_sha256 != manifest.dataset_sha256:
        raise Mem0ComparisonProtocolError(
            "benchmark dataset snapshot SHA-256 does not match the locked "
            "split manifest: "
            f"expected {manifest.dataset_sha256}, got {snapshot_sha256}"
        )
    if split not in manifest.splits:
        choices = ", ".join(manifest.splits)
        raise Mem0ComparisonProtocolError(
            f"unknown locked split {split!r}; choose one of {choices}"
        )
    if sum(manifest.splits.values()) != len(samples):
        raise Mem0ComparisonProtocolError(
            "locked split counts do not cover the parsed benchmark population"
        )

    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise Mem0ComparisonProtocolError(
            "benchmark sample IDs must be unique for locked splitting"
        )
    if manifest.algorithm != "stratified-largest-remainder-v1":
        raise Mem0ComparisonProtocolError(
            f"unsupported locked split algorithm: {manifest.algorithm}"
        )

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
        name: manifest.splits[name] - column_assigned[name]
        for name in split_names
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

    partitions: dict[str, list[BenchmarkSample]] = {
        name: [] for name in split_names
    }
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
    return sorted(
        partitions[split],
        key=lambda sample: hashlib.sha256(
            f"{manifest.salt}\0order\0{sample.sample_id}".encode("utf-8")
        ).digest(),
    )


def _parallel_raw_sessions(
    record: Mapping[str, Any],
) -> tuple[list[Any], list[Any], list[Any]]:
    sessions = record.get("haystack_sessions")
    session_ids = record.get("haystack_session_ids")
    dates = record.get("haystack_dates")
    if not all(isinstance(value, list) for value in (sessions, session_ids, dates)):
        raise Mem0ComparisonProtocolError(
            "every raw LongMemEval record needs list-valued sessions, IDs, and dates"
        )
    assert isinstance(sessions, list)
    assert isinstance(session_ids, list)
    assert isinstance(dates, list)
    if not (len(sessions) == len(session_ids) == len(dates)):
        raise Mem0ComparisonProtocolError(
            "raw LongMemEval sessions, IDs, and dates are not parallel"
        )
    return sessions, session_ids, dates


def compose_raw_stress_record(
    records: Sequence[Mapping[str, Any]],
    *,
    sample_id: str,
) -> dict[str, Any]:
    """Build a content-addressed raw-history bundle, not an ingest sequence.

    Keeping the original records separate is essential: Mem0's official
    runner sorts sessions *within one LongMemEval record*. Flattening first
    would globally interleave ten unrelated users by date and change the
    extraction model's rolling request window.
    """

    if not isinstance(sample_id, str) or not sample_id.strip():
        raise Mem0ComparisonProtocolError("sample_id must be non-empty")
    if not records:
        raise Mem0ComparisonProtocolError("a stress shard needs source records")

    records_out: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for record in records:
        question_id_value = record.get("question_id")
        if not isinstance(question_id_value, str) or not question_id_value.strip():
            raise Mem0ComparisonProtocolError("source record has no question_id")
        question_id = question_id_value.strip()
        if question_id in seen_questions:
            raise Mem0ComparisonProtocolError(
                f"duplicate source record {question_id!r} in one stress shard"
            )
        seen_questions.add(question_id)
        sessions, session_ids, dates = _parallel_raw_sessions(record)
        sanitized_sessions: list[Any] = []
        sanitized_ids: list[str] = []
        sanitized_dates: list[str] = []
        for session, source_value, date_value in zip(sessions, session_ids, dates):
            if not isinstance(session, list):
                raise Mem0ComparisonProtocolError(
                    f"record {question_id!r} contains a non-list session"
                )
            if not isinstance(source_value, str) or not source_value.strip():
                raise Mem0ComparisonProtocolError(
                    f"record {question_id!r} contains an empty session ID"
                )
            if not isinstance(date_value, str) or not date_value.strip():
                raise Mem0ComparisonProtocolError(
                    f"record {question_id!r} contains an empty session date"
                )
            sanitized_sessions.append(session)
            sanitized_ids.append(f"{question_id}::{source_value.strip()}")
            sanitized_dates.append(date_value)
        records_out.append(
            {
                "source_sample_id": question_id,
                "haystack_sessions": sanitized_sessions,
                "haystack_session_ids": sanitized_ids,
                "haystack_dates": sanitized_dates,
            }
        )

    return {
        "format": "memory-condense-mem0-raw-history-bundle-v1",
        "question_id": sample_id.strip(),
        "records": records_out,
    }


_WEEKDAY_RE = re.compile(r"\s+\([^)]*\)\s+")


def _parse_session_date(value: str) -> datetime:
    cleaned = _WEEKDAY_RE.sub(" ", value.strip())
    for format_string in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, format_string)
        except ValueError:
            continue
    raise Mem0ComparisonProtocolError(
        f"unsupported LongMemEval session date {value!r}"
    )


def build_composite_add_batches(
    records: Sequence[Mapping[str, Any]],
) -> tuple[CompositeAddBatch, ...]:
    """Render the exact ordered add sequence for a ten-record 1M shard."""

    batches: list[CompositeAddBatch] = []
    seen_questions: set[str] = set()
    for record in records:
        question_id_value = record.get("question_id")
        if not isinstance(question_id_value, str) or not question_id_value.strip():
            raise Mem0ComparisonProtocolError("source record has no question_id")
        question_id = question_id_value.strip()
        if question_id in seen_questions:
            raise Mem0ComparisonProtocolError(
                f"duplicate source record {question_id!r} in one stress shard"
            )
        seen_questions.add(question_id)
        sessions, session_ids, dates = _parallel_raw_sessions(record)
        ordered: list[tuple[datetime, int, str, str, list[Any]]] = []
        for original_index, (session, source_value, date_value) in enumerate(
            zip(sessions, session_ids, dates), start=1
        ):
            if not isinstance(session, list):
                raise Mem0ComparisonProtocolError("raw session must be a list")
            if not isinstance(source_value, str) or not source_value.strip():
                raise Mem0ComparisonProtocolError("raw session ID must be non-empty")
            if not isinstance(date_value, str) or not date_value.strip():
                raise Mem0ComparisonProtocolError("raw session date must be non-empty")
            ordered.append(
                (
                    _parse_session_date(date_value),
                    original_index,
                    f"{question_id}::{source_value.strip()}",
                    date_value,
                    session,
                )
            )
        ordered.sort(key=lambda item: (item[0], item[1]))

        for chronological_index, (
            _parsed,
            original_index,
            source,
            date,
            session,
        ) in enumerate(ordered, start=1):
            for turn_start in range(0, len(session), 2):
                raw_pair = session[turn_start : turn_start + 2]
                messages: list[tuple[str, str]] = []
                for turn in raw_pair:
                    if not isinstance(turn, Mapping):
                        raise Mem0ComparisonProtocolError("raw turn must be an object")
                    role = turn.get("role")
                    content = turn.get("content")
                    if not isinstance(role, str) or role.strip().lower() not in {
                        "user",
                        "assistant",
                    }:
                        raise Mem0ComparisonProtocolError(
                            f"unsupported raw LongMemEval role {role!r}"
                        )
                    if not isinstance(content, str):
                        raise Mem0ComparisonProtocolError(
                            "raw LongMemEval content must be a string"
                        )
                    messages.append((role.strip().lower(), content))
                if any(not content.strip() for _role, content in messages):
                    continue
                batches.append(
                    CompositeAddBatch(
                        source_sample_id=question_id,
                        source=source,
                        date=date,
                        session_index=chronological_index,
                        original_session_index=original_index,
                        batch_index=(turn_start // 2) + 1,
                        turn_start=turn_start,
                        messages=tuple(messages),
                    )
                )
    return tuple(batches)


def count_official_add_requests(
    record: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> RawAddCounts:
    """Count official consecutive slices and reject protocol ambiguity.

    The current official runner skips a whole one-or-two-message slice when a
    message is empty after ``strip()``.  The frozen adapter's public contract
    treats only the empty string as empty.  A whitespace-only message would
    therefore make the two protocols diverge, so certification fails before
    any Mem0 call rather than silently choosing one interpretation.
    """

    records = [record] if isinstance(record, Mapping) else list(record)
    raw_pairs = 0
    skipped_empty_pairs = 0
    whitespace_only_pairs = 0
    for source_record in records:
        sessions, _session_ids, _dates = _parallel_raw_sessions(source_record)
        for session in sessions:
            if not isinstance(session, list):
                raise Mem0ComparisonProtocolError("raw session must be a list")
            for start in range(0, len(session), 2):
                raw_pairs += 1
                pair = session[start : start + 2]
                contents: list[str] = []
                for turn in pair:
                    if not isinstance(turn, Mapping):
                        raise Mem0ComparisonProtocolError("raw turn must be an object")
                    role = turn.get("role")
                    content = turn.get("content")
                    if not isinstance(role, str) or role.strip().lower() not in {
                        "user",
                        "assistant",
                    }:
                        raise Mem0ComparisonProtocolError(
                            f"unsupported raw LongMemEval role {role!r}"
                        )
                    if not isinstance(content, str):
                        raise Mem0ComparisonProtocolError(
                            "raw LongMemEval content must be a string"
                        )
                    contents.append(content)
                official_empty = any(not content.strip() for content in contents)
                adapter_empty = any(not content for content in contents)
                if official_empty:
                    skipped_empty_pairs += 1
                if official_empty != adapter_empty:
                    whitespace_only_pairs += 1

    if whitespace_only_pairs:
        raise Mem0ComparisonProtocolError(
            "whitespace-only messages make the official runner and frozen "
            "adapter disagree; comparison certification is unavailable"
        )
    return RawAddCounts(
        raw_pairs=raw_pairs,
        skipped_empty_pairs=skipped_empty_pairs,
        add_requests=raw_pairs - skipped_empty_pairs,
        whitespace_only_pairs=0,
    )


def _records_from_raw_history_bundle(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Invert :func:`compose_raw_stress_record` without trusting add batches.

    The bundle stores source IDs as ``question_id::session_id``.  Rebuilding
    the original record-shaped inputs lets the execution boundary derive the
    official chronology, pairing, empty-pair handling, and messages again
    instead of trusting the shallow-frozen ``RawStressShard.add_batches``.
    """

    if value.get("format") != "memory-condense-mem0-raw-history-bundle-v1":
        raise Mem0ComparisonProtocolError("raw history bundle format mismatch")
    records_value = value.get("records")
    if not isinstance(records_value, list) or not records_value:
        raise Mem0ComparisonProtocolError("raw history bundle records are invalid")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(records_value):
        if not isinstance(raw, Mapping):
            raise Mem0ComparisonProtocolError(
                f"raw history bundle record {index} is not an object"
            )
        sample_id_value = raw.get("source_sample_id")
        if not isinstance(sample_id_value, str) or not sample_id_value.strip():
            raise Mem0ComparisonProtocolError(
                f"raw history bundle record {index} has no source_sample_id"
            )
        sample_id = sample_id_value.strip()
        if sample_id in seen:
            raise Mem0ComparisonProtocolError(
                f"raw history bundle repeats source sample {sample_id!r}"
            )
        seen.add(sample_id)
        sessions, source_ids, dates = _parallel_raw_sessions(raw)
        prefix = f"{sample_id}::"
        original_ids: list[str] = []
        for source_id in source_ids:
            if (
                not isinstance(source_id, str)
                or not source_id.startswith(prefix)
                or len(source_id) == len(prefix)
            ):
                raise Mem0ComparisonProtocolError(
                    "raw history bundle source namespace mismatch"
                )
            original_ids.append(source_id[len(prefix) :])
        records.append(
            {
                "question_id": sample_id,
                "haystack_sessions": sessions,
                "haystack_session_ids": original_ids,
                "haystack_dates": dates,
            }
        )
    return tuple(records)


def add_batches_sha256(batches: Sequence[CompositeAddBatch]) -> str:
    """Content-address an exact ordered Mem0 add sequence without persisting it."""

    return canonical_sha256(
        [
            {
                "source_sample_id": batch.source_sample_id,
                "source": batch.source,
                "date": batch.date,
                "session_index": batch.session_index,
                "original_session_index": batch.original_session_index,
                "batch_index": batch.batch_index,
                "turn_start": batch.turn_start,
                "messages": [list(message) for message in batch.messages],
            }
            for batch in batches
        ]
    )


def validate_raw_stress_shard(shard: RawStressShard) -> dict[str, Any]:
    """Recompute every mutable/content-bearing identity at an execution edge.

    ``RawStressShard`` is a frozen dataclass, but its Pydantic sample, mappings,
    and nested message tuples can still be replaced or mutated by a caller.
    This verifier derives the add sequence independently from the bound raw
    bundle and refuses stored digests or counts that merely *claim* to match.
    """

    observed_sample_sha256 = sample_sha256(shard.parsed_sample)
    if observed_sample_sha256 != shard.sample_sha256:
        raise Mem0ComparisonProtocolError("stress sample content SHA mismatch")
    observed_raw_sha256 = canonical_sha256(shard.raw_history_bundle)
    if observed_raw_sha256 != shard.raw_history_bundle_sha256:
        raise Mem0ComparisonProtocolError("raw history bundle content SHA mismatch")
    records = _records_from_raw_history_bundle(shard.raw_history_bundle)
    record_ids = tuple(str(record["question_id"]) for record in records)
    if record_ids != tuple(shard.history_sample_ids):
        raise Mem0ComparisonProtocolError(
            "raw history contributors differ from admitted history IDs"
        )
    derived_counts = count_official_add_requests(records)
    if derived_counts != shard.add_counts:
        raise Mem0ComparisonProtocolError("raw add-request counts are not derived")
    derived_batches = build_composite_add_batches(records)
    if derived_batches != tuple(shard.add_batches):
        raise Mem0ComparisonProtocolError("raw add sequence is not independently derived")
    question_ids = tuple(shard.question_ids)
    if len(question_ids) != len(set(question_ids)):
        raise Mem0ComparisonProtocolError("stress sample repeats question IDs")
    return {
        "sample_sha256": observed_sample_sha256,
        "raw_history_bundle_sha256": observed_raw_sha256,
        "history_sample_ids_sha256": canonical_sha256(list(record_ids)),
        "question_ids_sha256": canonical_sha256(list(question_ids)),
        "add_batches_sha256": add_batches_sha256(derived_batches),
        "raw_pairs": derived_counts.raw_pairs,
        "skipped_empty_pairs": derived_counts.skipped_empty_pairs,
        "add_requests": derived_counts.add_requests,
    }


def _admitted_histories(
    validation: Sequence[BenchmarkSample],
    *,
    sample_offset: int,
    target_tokens: int,
) -> list[BenchmarkSample]:
    if sample_offset < 0 or sample_offset >= len(validation):
        raise Mem0ComparisonProtocolError(
            f"sample offset {sample_offset} is outside the validation split"
        )
    admitted: list[BenchmarkSample] = []
    total = 0
    for sample in validation[sample_offset:]:
        admitted.append(sample)
        total += transcript_tokens(sample)
        if total >= target_tokens:
            break
    if total < target_tokens:
        raise Mem0ComparisonProtocolError(
            "validation tail cannot satisfy the stress-token target"
        )
    return admitted


def load_locked_raw_population(
    *,
    benchmark_file: str | Path,
    split_manifest: str | Path,
) -> LockedRawPopulation:
    """Load/hash/parse/split one immutable dataset snapshot.

    The source path is opened exactly once. The normalized benchmark samples,
    raw Mem0 records, and locked-manifest identity all derive from those exact
    bytes, so replacing the path during either parse phase cannot mix dataset
    versions.
    """

    dataset_path = Path(benchmark_file).resolve()
    manifest = load_split_manifest(split_manifest)
    snapshot = _read_dataset_snapshot(dataset_path)
    snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
    if snapshot_sha256 != manifest.dataset_sha256:
        raise Mem0ComparisonProtocolError(
            "benchmark dataset snapshot SHA-256 does not match the locked "
            "split manifest: "
            f"expected {manifest.dataset_sha256}, got {snapshot_sha256}"
        )
    normalized_payload = _decode_dataset_snapshot(snapshot)
    parsed = parse_longmemeval(normalized_payload)
    # Decode the raw view independently from the same immutable bytes. This
    # also prevents accidental mutation by either parser from contaminating
    # the other representation without performing another filesystem read.
    raw_payload = _decode_dataset_snapshot(snapshot)
    raw_records = _raw_records_from_payload(raw_payload)
    validation = _select_locked_split_from_snapshot(
        parsed,
        snapshot_sha256=snapshot_sha256,
        manifest=manifest,
        split="validation",
    )
    raw_by_id = {
        str(record["question_id"]).strip(): record for record in raw_records
    }
    return LockedRawPopulation(
        validation=tuple(validation),
        raw_by_id=raw_by_id,
    )


def _build_raw_stress_shard_from_population(
    population: LockedRawPopulation,
    *,
    sample_offset: int,
    target_tokens: int,
    max_questions: int,
) -> RawStressShard:
    validation = population.validation
    admitted = _admitted_histories(
        validation,
        sample_offset=sample_offset,
        target_tokens=target_tokens,
    )
    stress_sample = compose_context_stress_sample(
        list(validation[sample_offset:]),
        target_tokens=target_tokens,
        max_questions=max_questions,
        question_offset=0,
    )
    history_ids = tuple(sample.sample_id for sample in admitted)
    missing = [
        sample_id for sample_id in history_ids if sample_id not in population.raw_by_id
    ]
    if missing:
        raise Mem0ComparisonProtocolError(
            "parsed validation histories are missing from the raw dataset: "
            + ", ".join(missing)
        )
    source_records = [population.raw_by_id[sample_id] for sample_id in history_ids]
    raw_history_bundle = compose_raw_stress_record(
        source_records,
        sample_id=(
            f"mem0-context-stress-{target_tokens}-offset-{sample_offset:03d}"
        ),
    )
    add_counts = count_official_add_requests(source_records)
    add_batches = build_composite_add_batches(source_records)
    if len(add_batches) != add_counts.add_requests:
        raise Mem0ComparisonProtocolError(
            "prepared add sequence disagrees with the raw add-request count"
        )
    shard = RawStressShard(
        sample_offset=sample_offset,
        parsed_sample=stress_sample,
        sample_sha256=sample_sha256(stress_sample),
        history_sample_ids=history_ids,
        raw_history_bundle=raw_history_bundle,
        raw_history_bundle_sha256=canonical_sha256(raw_history_bundle),
        add_batches=add_batches,
        add_counts=add_counts,
    )
    validate_raw_stress_shard(shard)
    return shard


def build_raw_stress_shard(
    *,
    benchmark_file: str | Path,
    split_manifest: str | Path,
    sample_offset: int,
    target_tokens: int = 1_000_000,
    max_questions: int = 10,
) -> RawStressShard:
    """Reconstruct one exact locked validation shard and its raw Mem0 input."""

    population = load_locked_raw_population(
        benchmark_file=benchmark_file,
        split_manifest=split_manifest,
    )
    return _build_raw_stress_shard_from_population(
        population,
        sample_offset=sample_offset,
        target_tokens=target_tokens,
        max_questions=max_questions,
    )


def build_raw_stress_shards(
    *,
    benchmark_file: str | Path,
    split_manifest: str | Path,
    sample_offsets: Sequence[int],
    target_tokens: int = 1_000_000,
    max_questions: int = 10,
) -> tuple[RawStressShard, ...]:
    """Reconstruct several shards with one dataset parse and split check."""

    if len(set(sample_offsets)) != len(sample_offsets):
        raise Mem0ComparisonProtocolError("sample offsets must be unique")
    population = load_locked_raw_population(
        benchmark_file=benchmark_file,
        split_manifest=split_manifest,
    )
    return tuple(
        _build_raw_stress_shard_from_population(
            population,
            sample_offset=int(offset),
            target_tokens=target_tokens,
            max_questions=max_questions,
        )
        for offset in sample_offsets
    )


def shard_receipt(shard: RawStressShard) -> dict[str, Any]:
    """Return a text-free preflight receipt suitable for policy freezing."""

    return {
        "format": "memory-condense-mem0-raw-shard-v1",
        "sample_offset": shard.sample_offset,
        "sample_id": shard.parsed_sample.sample_id,
        "sample_sha256": shard.sample_sha256,
        "raw_history_bundle_sha256": shard.raw_history_bundle_sha256,
        "history_samples": len(shard.history_sample_ids),
        "questions": len(shard.question_ids),
        "question_ids": list(shard.question_ids),
        "turns": len(shard.parsed_sample.turns),
        "transcript_tokens": transcript_tokens(shard.parsed_sample),
        "raw_pairs": shard.add_counts.raw_pairs,
        "skipped_empty_pairs": shard.add_counts.skipped_empty_pairs,
        "mem0_add_requests": shard.add_counts.add_requests,
        "add_batches_sha256": add_batches_sha256(shard.add_batches),
        "whitespace_only_pairs": shard.add_counts.whitespace_only_pairs,
    }
