"""Gold-blind Qwen-to-MiniLM ranking-distillation assay.

This is an isolated research tool.  It does not alter the production retrieval
path.  Its four subcommands form explicit process boundaries:

``project``
    Reconstruct the locked LongMemEval *development* membership using only
    question IDs and question types, split those records 140/30/30, and emit a
    gold-free plane.  Memories are pooled across every record in their
    train/calibration/test partition before query-only candidate selection.

``teacher``
    Score two independent groups of eight proxy cards with the resident Qwen
    coverage scorer.  Only exact hexadecimal scalar scores cross the boundary.

``train``
    Measure the pinned zero-shot MS MARCO MiniLM first.  Fine-tune with a
    manual pairwise RankNet objective only when the frozen zero-shot
    calibration gate misses, then calibrate the student-margin fallback
    threshold on that development partition.  Training logic does not score
    or index the test rows, although the full sealed plane is process-visible.

``assay``
    Measure held-out teacher agreement and warm scorer latency.  A live Qwen
    fallback is optional; without it the report explicitly limits latency to
    projected candidate scoring.

The source dataset is an oracle projection, not the million-token parent
corpus.  Proxy cards are deterministic 48-token renderings, not LFM2
Transcript cards.  Consequently this assay can establish mechanism and
latency feasibility, but it cannot promote a production retrieval policy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT), str(_ROOT / "tools")]

from memory_condense.domain._tokenizer import truncate_to_tokens  # noqa: E402


PLANE_FORMAT = "memory-condense-qwen-minilm-gold-blind-plane-v1"
TEACHER_FORMAT = "memory-condense-qwen-minilm-teacher-v1"
TRAINING_FORMAT = "memory-condense-qwen-minilm-training-v1"
ASSAY_FORMAT = "memory-condense-qwen-minilm-assay-v1"
LOCKED_SPLIT_FORMAT = "memory-condense-locked-benchmark-split-v1"
LOCKED_SPLIT_ALGORITHM = "stratified-largest-remainder-v1"
SOURCE_KIND = "longmemeval-oracle-development-projection"
PROXY_KIND = "deterministic-raw-turn-proxy-card-not-lfm2-transcript"
UTILITY = "max(0,qk)+log1p(max(0,ov_transport))"
PARTITION_COUNTS = {"train": 140, "calibration": 30, "test": 30}
PARTITION_SALT = "memory-condense-qwen-minilm-distillation-v1-2026-09-05"
DEFAULT_CANDIDATE_TOKENS = 48
DEFAULT_QUERY_TOKENS = 48
DEFAULT_CANDIDATES = 16
DEFAULT_GROUP_SIZE = 8
DEFAULT_LEXICAL = 12
DEFAULT_DISTRACTORS = 4

_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answers",
        "answer_session_ids",
        "answer_source_ids",
        "evidence",
        "evidence_sources",
        "gold",
        "gold_answer",
        "has_answer",
        "is_answer",
        "label",
        "labels",
        "prediction",
        "predictions",
        "reference",
        "references",
    }
)
_TERM = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)
_PLANE_ROOT_KEYS = frozenset(
    {
        "format",
        "status",
        "source_kind",
        "promotion_eligible",
        "limitations",
        "projection_dataset_sha256",
        "split_manifest_sha256",
        "locked_split",
        "partition_policy",
        "candidate_policy",
        "counts",
        "questions",
        "memories",
    }
)
_QUESTION_KEYS = frozenset(
    {
        "question_id",
        "question",
        "question_type",
        "question_date",
        "partition",
        "candidate_memory_ids",
    }
)
_MEMORY_KEYS = frozenset(
    {
        "memory_id",
        "source_id",
        "source_date",
        "role",
        "text",
        "partition",
    }
)


class AssayError(RuntimeError):
    """The assay input or execution violates its sealed contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        buffer = bytearray(1024 * 1024)
        view = memoryview(buffer)
        while count := stream.readinto(buffer):
            digest.update(view[:count])
    return digest.hexdigest()


def _identity(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AssayError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AssayError(f"{label} must be an array")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssayError(f"{label} must be non-empty text")
    return value.strip()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AssayError(f"{label} keys differ; missing={missing}, extra={extra}")


def _assert_gold_free(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if normalized in _FORBIDDEN_KEYS:
                raise AssayError(f"forbidden gold-bearing key at {path}.{key}")
            _assert_gold_free(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_gold_free(item, path=f"{path}[{index}]")


def _publish_json(path: Path, payload: Mapping[str, Any]) -> tuple[str, Path]:
    _assert_gold_free(payload)
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    digest = _sha256_bytes(encoded)
    path = path.resolve()
    sidecar = path.with_name(path.name + ".sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    created_path = False
    created_sidecar = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        created_path = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        created_sidecar = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(f"{digest}  {path.name}\n".encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created_sidecar and sidecar.exists():
            sidecar.unlink()
        if created_path and path.exists():
            path.unlink()
        raise
    return digest, sidecar


def _runtime_output_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    runtime_root = (_ROOT / ".tmp").resolve()
    if not resolved.is_relative_to(runtime_root):
        raise AssayError(f"{label} must be under the worktree .tmp directory")
    return resolved


def _read_sealed(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise AssayError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssayError(f"cannot decode {label}: {exc}") from exc
    payload = _require_mapping(payload, label)
    if raw != (_canonical_json(payload) + "\n").encode("utf-8"):
        raise AssayError(f"{label} is not canonical JSON")
    digest = _sha256_bytes(raw)
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar.is_symlink() or not sidecar.is_file() or sidecar.read_bytes() != expected:
        raise AssayError(f"{label} SHA-256 sidecar is missing or invalid")
    _assert_gold_free(payload)
    return payload, digest


def _opaque_id(kind: str, *parts: str) -> str:
    digest = hashlib.sha256()
    digest.update(kind.encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return f"{kind}-{digest.hexdigest()[:24]}"


def _terms(text: str) -> tuple[str, ...]:
    terms: list[str] = []
    for raw in _TERM.findall(str(text).casefold()):
        if len(raw) < 2 or raw in _STOP:
            continue
        term = raw
        for suffix in ("ing", "ed", "es", "s"):
            if term.endswith(suffix) and len(term) > len(suffix) + 3:
                term = term[: -len(suffix)]
                break
        terms.append(term)
    return tuple(terms)


def _stratified_partitions(
    rows: Sequence[Mapping[str, Any]],
    *,
    counts: Mapping[str, int],
    salt: str,
    id_key: str,
    stratum_key: str,
) -> dict[str, list[Mapping[str, Any]]]:
    """Replay the repository's stratified largest-remainder assignment."""

    names = list(counts)
    if not names or any(int(counts[name]) < 1 for name in names):
        raise AssayError("partition counts must be positive")
    if sum(int(counts[name]) for name in names) != len(rows):
        raise AssayError("partition counts do not cover the population")
    strata: dict[str, list[Mapping[str, Any]]] = {}
    seen_ids: set[str] = set()
    for row in rows:
        row_id = _require_text(row.get(id_key), f"{id_key}")
        if row_id in seen_ids:
            raise AssayError(f"duplicate population ID: {row_id}")
        seen_ids.add(row_id)
        stratum = str(row.get(stratum_key) or "uncategorized").strip() or "uncategorized"
        strata.setdefault(stratum, []).append(row)

    quotas: dict[str, dict[str, int]] = {}
    remainders: dict[str, dict[str, float]] = {}
    assigned = {name: 0 for name in names}
    leftovers: dict[str, int] = {}
    population = len(rows)
    for stratum, members in strata.items():
        quotas[stratum] = {}
        remainders[stratum] = {}
        for name in names:
            ideal = len(members) * int(counts[name]) / population
            base = int(ideal)
            quotas[stratum][name] = base
            remainders[stratum][name] = ideal - base
            assigned[name] += base
        leftovers[stratum] = len(members) - sum(quotas[stratum].values())
    deficits = {name: int(counts[name]) - assigned[name] for name in names}
    for stratum in sorted(strata):
        used: set[str] = set()
        for _ in range(leftovers[stratum]):
            choices = [name for name in names if deficits[name] > 0]
            unused = [name for name in choices if name not in used]
            pool = unused or choices
            if not pool:
                raise AssayError("stratified apportionment exhausted capacity")
            selected = max(
                pool,
                key=lambda name: (
                    remainders[stratum][name],
                    deficits[name],
                    -names.index(name),
                ),
            )
            quotas[stratum][selected] += 1
            deficits[selected] -= 1
            used.add(selected)
    if any(deficits.values()):
        raise AssayError("stratified apportionment left a partition deficit")

    partitions: dict[str, list[Mapping[str, Any]]] = {name: [] for name in names}
    for stratum in sorted(strata):
        ordered = sorted(
            strata[stratum],
            key=lambda row: hashlib.sha256(
                f"{salt}\0{stratum}\0{row[id_key]}".encode("utf-8")
            ).digest(),
        )
        offset = 0
        for name in names:
            count = quotas[stratum][name]
            partitions[name].extend(ordered[offset : offset + count])
            offset += count
    for name in names:
        partitions[name].sort(
            key=lambda row: hashlib.sha256(
                f"{salt}\0order\0{row[id_key]}".encode("utf-8")
            ).digest()
        )
    return partitions


def _raw_role(value: Any, index: int) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized in {"user", "human"}:
        return "user"
    if normalized in {"assistant", "ai", "bot", "system"}:
        return "assistant"
    return "user" if index % 2 == 0 else "assistant"


def _raw_text(turn: Mapping[str, Any]) -> str:
    value = turn.get("content", turn.get("text"))
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text", item.get("content"))
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _candidate_ids(
    question: Mapping[str, Any],
    memories: Sequence[Mapping[str, Any]],
    *,
    term_frequencies: Mapping[str, Counter[str]],
    document_lengths: Mapping[str, int],
    document_frequency: Mapping[str, int],
    average_length: float,
    lexical_count: int,
    distractor_count: int,
    salt: str,
) -> list[str]:
    """Select from a whole partition pool using the query and salt only."""

    query_terms = tuple(dict.fromkeys(_terms(str(question["question"]))))
    population = len(memories)
    scored: list[tuple[float, str]] = []
    k1 = 1.2
    b = 0.75
    for memory in memories:
        memory_id = str(memory["memory_id"])
        frequencies = term_frequencies[memory_id]
        document_length = document_lengths[memory_id]
        length_normalizer = k1 * (
            1.0 - b + b * document_length / max(1.0, average_length)
        )
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            df = int(document_frequency.get(term, 0))
            inverse_frequency = math.log(
                1.0 + (population - df + 0.5) / (df + 0.5)
            )
            score += inverse_frequency * (
                frequency * (k1 + 1.0) / (frequency + length_normalizer)
            )
        scored.append((score, memory_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    lexical = [memory_id for _score, memory_id in scored[:lexical_count]]
    lexical_set = set(lexical)
    remaining = [memory_id for _score, memory_id in scored if memory_id not in lexical_set]
    question_id = str(question["question_id"])
    remaining.sort(
        key=lambda memory_id: hashlib.sha256(
            f"{salt}\0distractor\0{question_id}\0{memory_id}".encode("utf-8")
        ).digest()
    )
    selected = lexical + remaining[:distractor_count]
    if len(selected) != lexical_count + distractor_count:
        raise AssayError("partition pool cannot fill the fixed candidate set")
    if len(selected) != len(set(selected)):
        raise AssayError("candidate selection produced duplicate memory IDs")
    return selected


def _validate_plane(value: Mapping[str, Any]) -> None:
    _exact_keys(value, _PLANE_ROOT_KEYS, "projection")
    if value.get("format") != PLANE_FORMAT:
        raise AssayError("unsupported projection format")
    if value.get("source_kind") != SOURCE_KIND:
        raise AssayError("projection source kind differs")
    if value.get("promotion_eligible") is not False:
        raise AssayError("projection must be marked ineligible for promotion")
    questions = _require_list(value.get("questions"), "projection questions")
    memories = _require_list(value.get("memories"), "projection memories")
    memory_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(memories):
        memory = _require_mapping(raw, f"memory {index}")
        _exact_keys(memory, _MEMORY_KEYS, f"memory {index}")
        memory_id = _require_text(memory.get("memory_id"), f"memory {index} ID")
        if memory_id in memory_by_id:
            raise AssayError(f"duplicate memory ID: {memory_id}")
        if memory.get("partition") not in PARTITION_COUNTS:
            raise AssayError("memory partition is invalid")
        _require_text(memory.get("source_id"), "memory source ID")
        _require_text(memory.get("role"), "memory role")
        _require_text(memory.get("text"), "memory text")
        memory_by_id[memory_id] = memory
    seen_questions: set[str] = set()
    observed_counts = {name: 0 for name in PARTITION_COUNTS}
    candidate_policy = _require_mapping(value.get("candidate_policy"), "candidate policy")
    candidate_count = int(candidate_policy.get("candidate_count", 0))
    for index, raw in enumerate(questions):
        question = _require_mapping(raw, f"question {index}")
        _exact_keys(question, _QUESTION_KEYS, f"question {index}")
        question_id = _require_text(question.get("question_id"), "question ID")
        if question_id in seen_questions:
            raise AssayError(f"duplicate question ID: {question_id}")
        seen_questions.add(question_id)
        _require_text(question.get("question"), "question text")
        partition = question.get("partition")
        if partition not in PARTITION_COUNTS:
            raise AssayError("question partition is invalid")
        observed_counts[str(partition)] += 1
        candidates = _require_list(
            question.get("candidate_memory_ids"), "candidate memory IDs"
        )
        if len(candidates) != candidate_count or len(candidates) != len(set(candidates)):
            raise AssayError("question candidate cardinality differs")
        for memory_id in candidates:
            memory = memory_by_id.get(str(memory_id))
            if memory is None:
                raise AssayError(f"question references unknown memory: {memory_id}")
            if memory.get("partition") != partition:
                raise AssayError("candidate crossed a source-disjoint partition")
    if observed_counts != PARTITION_COUNTS:
        raise AssayError(f"question partition counts differ: {observed_counts}")
    _assert_gold_free(value)


def _project(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = args.dataset.resolve()
    manifest_path = args.split_manifest.resolve()
    projection_dataset_sha256 = _file_sha256(dataset_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssayError(f"cannot read locked split manifest: {exc}") from exc
    manifest = _require_mapping(manifest, "locked split manifest")
    if manifest.get("format") != LOCKED_SPLIT_FORMAT:
        raise AssayError("locked split manifest format differs")
    if manifest.get("algorithm") != LOCKED_SPLIT_ALGORITHM:
        raise AssayError("locked split algorithm differs")
    parent_dataset_sha256 = _require_text(
        manifest.get("dataset_sha256"), "locked parent dataset SHA-256"
    )
    if not re.fullmatch(r"[0-9a-f]{64}", parent_dataset_sha256):
        raise AssayError("locked parent dataset SHA-256 is invalid")
    locked_counts = _require_mapping(manifest.get("splits"), "locked split counts")
    if list(locked_counts) != ["development", "validation", "confirmation"]:
        raise AssayError("locked split names or order differ")
    try:
        raw_records = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssayError(f"cannot read LongMemEval source: {exc}") from exc
    raw_records = _require_list(raw_records, "LongMemEval population")

    # This is the only pass over non-development records.  It deliberately
    # projects only the two fields required by the locked membership logic.
    metadata: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_records):
        record = _require_mapping(raw, f"dataset record {index}")
        question_id = _require_text(record.get("question_id"), "question_id")
        question_type = str(record.get("question_type") or "uncategorized")
        metadata.append(
            {
                "question_id": question_id,
                "question_type": question_type,
                "record_index": index,
            }
        )
    locked = _stratified_partitions(
        metadata,
        counts={name: int(value) for name, value in locked_counts.items()},
        salt=_require_text(manifest.get("salt"), "locked split salt"),
        id_key="question_id",
        stratum_key="question_type",
    )
    development_metadata = locked["development"]
    if len(development_metadata) != 200:
        raise AssayError("the distillation policy requires 200 development records")
    # Only after membership has been reconstructed do we dereference content.
    # Confirmation and validation records remain metadata-only throughout.
    raw_by_id = {
        str(meta["question_id"]): _require_mapping(
            raw_records[int(meta["record_index"])], "development record"
        )
        for meta in development_metadata
    }
    distill = _stratified_partitions(
        development_metadata,
        counts=PARTITION_COUNTS,
        salt=PARTITION_SALT,
        id_key="question_id",
        stratum_key="question_type",
    )

    questions: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    partition_memories: dict[str, list[dict[str, Any]]] = {
        name: [] for name in PARTITION_COUNTS
    }
    for partition in PARTITION_COUNTS:
        for meta in distill[partition]:
            question_id = str(meta["question_id"])
            record = raw_by_id[question_id]
            question = {
                "question_id": question_id,
                "question": _require_text(record.get("question"), "development question"),
                "question_type": str(record.get("question_type") or "uncategorized"),
                "question_date": (
                    str(record["question_date"]).strip()
                    if record.get("question_date") is not None
                    else None
                ),
                "partition": partition,
                "candidate_memory_ids": [],
            }
            questions.append(question)
            sessions = _require_list(
                record.get("haystack_sessions"), "development haystack sessions"
            )
            session_ids = record.get("haystack_session_ids")
            session_dates = record.get("haystack_dates")
            session_ids = session_ids if isinstance(session_ids, list) else []
            session_dates = session_dates if isinstance(session_dates, list) else []
            for session_index, raw_session in enumerate(sessions):
                session = _require_list(raw_session, "development session")
                raw_source_id = (
                    str(session_ids[session_index])
                    if session_index < len(session_ids)
                    else f"session-{session_index}"
                )
                source_date = (
                    str(session_dates[session_index]).strip()
                    if session_index < len(session_dates)
                    and session_dates[session_index] is not None
                    else None
                )
                opaque_source = _opaque_id(
                    "source", question_id, str(session_index), raw_source_id
                )
                for turn_index, raw_turn in enumerate(session):
                    turn = _require_mapping(raw_turn, "development turn")
                    text = _raw_text(turn)
                    if not text:
                        continue
                    memory_id = _opaque_id(
                        "memory",
                        question_id,
                        str(session_index),
                        str(turn_index),
                        _sha256_bytes(text.encode("utf-8")),
                    )
                    memory = {
                        "memory_id": memory_id,
                        "source_id": opaque_source,
                        "source_date": source_date,
                        "role": _raw_role(turn.get("role"), turn_index),
                        "text": text,
                        "partition": partition,
                    }
                    memories.append(memory)
                    partition_memories[partition].append(memory)

    # Candidate construction is partition-global.  No query's originating
    # record or source is consulted here, and no answer source is injected.
    for partition, pool in partition_memories.items():
        if len(pool) < args.candidates:
            raise AssayError(f"{partition} memory pool is too small")
        doc_terms = {
            str(memory["memory_id"]): _terms(str(memory["text"])) for memory in pool
        }
        term_frequencies = {
            memory_id: Counter(terms) for memory_id, terms in doc_terms.items()
        }
        document_lengths = {
            memory_id: len(terms) for memory_id, terms in doc_terms.items()
        }
        df: Counter[str] = Counter()
        for terms in doc_terms.values():
            df.update(set(terms))
        average_length = statistics.fmean(max(1, len(terms)) for terms in doc_terms.values())
        for question in questions:
            if question["partition"] != partition:
                continue
            question["candidate_memory_ids"] = _candidate_ids(
                question,
                pool,
                term_frequencies=term_frequencies,
                document_lengths=document_lengths,
                document_frequency=df,
                average_length=average_length,
                lexical_count=args.lexical,
                distractor_count=args.distractors,
                salt=PARTITION_SALT,
            )

    questions.sort(key=lambda row: (list(PARTITION_COUNTS).index(row["partition"]), row["question_id"]))
    memories.sort(key=lambda row: (list(PARTITION_COUNTS).index(row["partition"]), row["memory_id"]))
    counts = {
        "source_population_records": len(raw_records),
        "development_questions": len(questions),
        "memories": len(memories),
        "sources": len({memory["source_id"] for memory in memories}),
        "questions_by_partition": dict(PARTITION_COUNTS),
        "memories_by_partition": {
            name: len(partition_memories[name]) for name in PARTITION_COUNTS
        },
        "sources_by_partition": {
            name: len(
                {
                    memory["source_id"]
                    for memory in partition_memories[name]
                }
            )
            for name in PARTITION_COUNTS
        },
    }
    payload: dict[str, Any] = {
        "format": PLANE_FORMAT,
        "status": "projected_gold_blind_development_only",
        "source_kind": SOURCE_KIND,
        "promotion_eligible": False,
        "limitations": [
            "source is the LongMemEval oracle development projection, not the 1M parent",
            "candidate text is a deterministic proxy card, not LFM2 Transcript output",
            "this plane can test ranking distillation but cannot promote retrieval policy",
        ],
        "projection_dataset_sha256": projection_dataset_sha256,
        "split_manifest_sha256": _file_sha256(manifest_path),
        "locked_split": {
            "format": LOCKED_SPLIT_FORMAT,
            "algorithm": LOCKED_SPLIT_ALGORITHM,
            "selected": "development",
            "selected_count": len(development_metadata),
            "membership_fields_read": ["question_id", "question_type"],
            "nondevelopment_content_projected": False,
            "parent_dataset_sha256": parent_dataset_sha256,
            "parent_dataset_bytes_unavailable": True,
            "projection_bytes_are_parent_bytes": False,
        },
        "partition_policy": {
            "algorithm": LOCKED_SPLIT_ALGORITHM,
            "salt_sha256": _sha256_bytes(PARTITION_SALT.encode("utf-8")),
            "counts": dict(PARTITION_COUNTS),
            "candidate_scope": "partition_global_pool_never_record_local",
            "question_partitions_disjoint": True,
            "opaque_source_partitions_disjoint": True,
        },
        "candidate_policy": {
            "kind": "query_only_bm25_plus_salted_distractors",
            "candidate_count": args.candidates,
            "lexical_count": args.lexical,
            "distractor_count": args.distractors,
            "candidate_tokens": args.candidate_tokens,
            "query_tokens": args.query_tokens,
            "proxy_kind": PROXY_KIND,
            "record_or_answer_source_injection": False,
        },
        "counts": counts,
        "questions": questions,
        "memories": memories,
    }
    _validate_plane(payload)
    return payload


def _float_hex(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise AssayError("a scorer emitted a non-finite scalar")
    return number.hex()


def _parse_float_hex(value: Any, label: str) -> float:
    text_value = _require_text(value, label)
    try:
        number = float.fromhex(text_value)
    except ValueError as exc:
        raise AssayError(f"{label} is not a hexadecimal float") from exc
    if not math.isfinite(number) or number.hex() != text_value:
        raise AssayError(f"{label} is not a canonical finite hexadecimal float")
    return number


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _query_proxy(question: Mapping[str, Any], policy: Mapping[str, Any]) -> str:
    date = question.get("question_date")
    prefix = f"Date: {date}\n" if date else ""
    return truncate_to_tokens(
        prefix + str(question["question"]), int(policy["query_tokens"])
    )


def _memory_proxy(memory: Mapping[str, Any], policy: Mapping[str, Any]) -> str:
    date = memory.get("source_date") or "unknown"
    rendered = (
        f"Date: {date}\nRole: {memory['role']}\nFact: {memory['text']}"
    )
    return truncate_to_tokens(rendered, int(policy["candidate_tokens"]))


def _plane_indexes(
    plane: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    questions = {
        str(row["question_id"]): row for row in _require_list(plane["questions"], "questions")
    }
    memories = {
        str(row["memory_id"]): row for row in _require_list(plane["memories"], "memories")
    }
    return questions, memories


def _teacher_utility(qk: float, ov_transport: float) -> float:
    return max(0.0, float(qk)) + math.log1p(max(0.0, float(ov_transport)))


def _validate_teacher(value: Mapping[str, Any], plane: Mapping[str, Any], plane_sha: str) -> None:
    expected_root = frozenset(
        {
            "format",
            "status",
            "promotion_eligible",
            "limitations",
            "plane_sha256",
            "teacher_identity",
            "score_policy",
            "counts",
            "rows",
        }
    )
    _exact_keys(value, expected_root, "teacher artifact")
    if value.get("format") != TEACHER_FORMAT:
        raise AssayError("unsupported teacher artifact format")
    if value.get("plane_sha256") != plane_sha:
        raise AssayError("teacher artifact is bound to a different projection")
    if value.get("promotion_eligible") is not False:
        raise AssayError("teacher artifact must remain ineligible for promotion")
    score_policy = _require_mapping(value.get("score_policy"), "teacher score policy")
    if score_policy.get("utility") != UTILITY:
        raise AssayError("teacher utility differs from the frozen policy")
    if int(score_policy.get("group_size", 0)) != DEFAULT_GROUP_SIZE:
        raise AssayError("teacher group size differs from the frozen policy")
    questions, _memories = _plane_indexes(plane)
    rows = _require_list(value.get("rows"), "teacher rows")
    if len(rows) != len(questions):
        raise AssayError("teacher artifact is incomplete")
    seen: set[str] = set()
    for row_index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"teacher row {row_index}")
        _exact_keys(
            row,
            frozenset(
                {
                    "question_id",
                    "partition",
                    "query_sha256",
                    "groups",
                    "teacher_top_memory_id",
                    "teacher_margin_hex",
                }
            ),
            f"teacher row {row_index}",
        )
        question_id = _require_text(row.get("question_id"), "teacher question ID")
        if question_id in seen or question_id not in questions:
            raise AssayError("teacher question ID is duplicate or unknown")
        seen.add(question_id)
        question = questions[question_id]
        if row.get("partition") != question.get("partition"):
            raise AssayError("teacher partition differs from projection")
        policy = _require_mapping(plane["candidate_policy"], "candidate policy")
        query = _query_proxy(question, policy)
        if row.get("query_sha256") != _sha256_bytes(query.encode("utf-8")):
            raise AssayError("teacher query binding differs")
        expected_ids = list(question["candidate_memory_ids"])
        observed_ids: list[str] = []
        scored: list[tuple[float, str]] = []
        groups = _require_list(row.get("groups"), "teacher groups")
        if len(groups) != len(expected_ids) // DEFAULT_GROUP_SIZE:
            raise AssayError("teacher group count differs")
        for group_index, raw_group in enumerate(groups):
            group = _require_mapping(raw_group, "teacher group")
            _exact_keys(
                group,
                frozenset({"group_index", "workspace_tokens", "scores"}),
                "teacher group",
            )
            if group.get("group_index") != group_index:
                raise AssayError("teacher group order differs")
            scores = _require_list(group.get("scores"), "teacher scores")
            if len(scores) != DEFAULT_GROUP_SIZE:
                raise AssayError("teacher group is not an independent group of eight")
            for raw_score in scores:
                score = _require_mapping(raw_score, "teacher score")
                _exact_keys(
                    score,
                    frozenset(
                        {
                            "memory_id",
                            "qk_hex",
                            "ov_transport_hex",
                            "utility_hex",
                        }
                    ),
                    "teacher score",
                )
                memory_id = _require_text(score.get("memory_id"), "scored memory ID")
                qk = _parse_float_hex(score.get("qk_hex"), "QK score")
                ov = _parse_float_hex(score.get("ov_transport_hex"), "OV transport")
                utility = _parse_float_hex(score.get("utility_hex"), "teacher utility")
                if _float_hex(_teacher_utility(qk, ov)) != _float_hex(utility):
                    raise AssayError("teacher utility cannot be replayed exactly")
                observed_ids.append(memory_id)
                scored.append((utility, memory_id))
        if observed_ids != expected_ids:
            raise AssayError("teacher candidate order differs from projection")
        ordered = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)
        if row.get("teacher_top_memory_id") != ordered[0][1]:
            raise AssayError("teacher top memory does not replay")
        margin = ordered[0][0] - ordered[1][0]
        if _float_hex(margin) != row.get("teacher_margin_hex"):
            raise AssayError("teacher margin does not replay")
    _assert_gold_free(value)


def _teacher(args: argparse.Namespace) -> dict[str, Any]:
    from memory_condense.associations.head_memory_models import (
        AssociativeMemoryCandidate,
    )
    from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
    from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder

    plane, plane_sha = _read_sealed(args.plane, label="projection")
    _validate_plane(plane)
    policy = _require_mapping(plane["candidate_policy"], "candidate policy")
    if (
        int(policy.get("candidate_count", 0)) != DEFAULT_CANDIDATES
        or int(policy.get("lexical_count", 0)) != DEFAULT_LEXICAL
        or int(policy.get("distractor_count", 0)) != DEFAULT_DISTRACTORS
        or int(policy.get("candidate_tokens", 0)) != DEFAULT_CANDIDATE_TOKENS
    ):
        raise AssayError("projection differs from the fixed 12+4, 48-token assay")
    questions, memories = _plane_indexes(plane)

    loaded_started = time.perf_counter()
    encoder = Qwen3PrefixEncoder(
        args.qwen_model_dir.resolve(),
        layers=args.layers,
        device=args.device,
        dtype=args.dtype,
    )
    linker = QwenMemoryLinker(
        encoder,
        layer=args.attention_layer,
        max_candidates=DEFAULT_GROUP_SIZE,
        max_workspace_tokens=args.workspace_tokens,
    )
    load_seconds = time.perf_counter() - loaded_started
    rows: list[dict[str, Any]] = []
    pass_seconds: list[float] = []
    workspace_tokens: list[int] = []
    try:
        for question in plane["questions"]:
            query = _query_proxy(question, policy)
            candidate_ids = list(question["candidate_memory_ids"])
            groups: list[dict[str, Any]] = []
            all_scores: list[tuple[float, str]] = []
            for group_index, start in enumerate(range(0, len(candidate_ids), DEFAULT_GROUP_SIZE)):
                group_ids = candidate_ids[start : start + DEFAULT_GROUP_SIZE]
                candidates = [
                    AssociativeMemoryCandidate(
                        episode_id=memory_id,
                        text=_memory_proxy(memories[memory_id], policy),
                    )
                    for memory_id in group_ids
                ]
                started = time.perf_counter()
                result = linker.inspect_coverage(query, candidates)
                pass_seconds.append(time.perf_counter() - started)
                if result.workspace_candidates != len(group_ids):
                    raise AssayError(
                        "Qwen workspace cap silently omitted a fixed candidate"
                    )
                hit_by_id = {hit.episode_id: hit for hit in result.hits}
                scores: list[dict[str, str]] = []
                for memory_id in group_ids:
                    if memory_id not in hit_by_id:
                        raise AssayError("Qwen omitted a candidate score")
                    hit = hit_by_id[memory_id]
                    utility = _teacher_utility(hit.qk_score, hit.ov_transport)
                    scores.append(
                        {
                            "memory_id": memory_id,
                            "qk_hex": _float_hex(hit.qk_score),
                            "ov_transport_hex": _float_hex(hit.ov_transport),
                            "utility_hex": _float_hex(utility),
                        }
                    )
                    all_scores.append((utility, memory_id))
                groups.append(
                    {
                        "group_index": group_index,
                        "workspace_tokens": result.workspace_tokens,
                        "scores": scores,
                    }
                )
                workspace_tokens.append(int(result.workspace_tokens))
            ranked = sorted(
                all_scores, key=lambda item: (item[0], item[1]), reverse=True
            )
            rows.append(
                {
                    "question_id": question["question_id"],
                    "partition": question["partition"],
                    "query_sha256": _sha256_bytes(query.encode("utf-8")),
                    "groups": groups,
                    "teacher_top_memory_id": ranked[0][1],
                    "teacher_margin_hex": _float_hex(ranked[0][0] - ranked[1][0]),
                }
            )
    finally:
        checkpoint_sha = encoder.checkpoint_sha256
        model_id = encoder.model_id
        model_revision = encoder.model_revision
        del linker, encoder
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    payload: dict[str, Any] = {
        "format": TEACHER_FORMAT,
        "status": "qwen_teacher_scores_complete",
        "promotion_eligible": False,
        "limitations": list(plane["limitations"]),
        "plane_sha256": plane_sha,
        "teacher_identity": {
            "model_id": model_id,
            "model_revision": model_revision,
            "checkpoint_sha256": checkpoint_sha,
            "layers": args.layers,
            "attention_layer": args.attention_layer,
            "device": args.device,
            "dtype": args.dtype,
            "load_seconds_hex": _float_hex(load_seconds),
        },
        "score_policy": {
            "operator": "QwenMemoryLinker.inspect_coverage",
            "groups_are_independent": True,
            "group_size": DEFAULT_GROUP_SIZE,
            "passes_per_question": DEFAULT_CANDIDATES // DEFAULT_GROUP_SIZE,
            "utility": UTILITY,
            "scalar_encoding": "python-float.hex",
            "direct_benchmark_truth_fields_available": False,
            "candidate_universe_oracle_derived": True,
        },
        "counts": {
            "questions": len(rows),
            "passes": len(pass_seconds),
            "candidate_inspections": len(rows) * DEFAULT_CANDIDATES,
            "max_workspace_tokens": max(workspace_tokens, default=0),
            "pass_latency_ms_p50_hex": _float_hex(
                1000.0 * (_percentile(pass_seconds, 0.5) or 0.0)
            ),
            "pass_latency_ms_p95_hex": _float_hex(
                1000.0 * (_percentile(pass_seconds, 0.95) or 0.0)
            ),
        },
        "rows": rows,
    }
    _validate_teacher(payload, plane, plane_sha)
    return payload


def _require_student_stack() -> tuple[Any, Any, Any]:
    # Must be set before torch initializes cuBLAS. Eager attention below avoids
    # the nondeterministic memory-efficient SDPA backward kernel.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise AssayError("torch and transformers are required for student scoring") from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer


def _synchronize(torch: Any, device: Any) -> None:
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.synchronize(device)


def _load_student(
    model_dir: Path, *, device_name: str, verify_base: bool
) -> tuple[Any, Any, Any, Any, str | None]:
    torch, AutoModelForSequenceClassification, AutoTokenizer = _require_student_stack()
    root = model_dir.resolve()
    checkpoint_sha: str | None = None
    if verify_base:
        from memory_condense.search.selectors.cross_encoder_selector import (
            verify_ms_marco_checkpoint,
        )

        checkpoint_sha = verify_ms_marco_checkpoint(root)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        root,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation="eager",
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise AssayError("CUDA was requested for MiniLM but is unavailable")
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device, checkpoint_sha


def _student_scores(
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    query: str,
    candidates: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
) -> list[float]:
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            batch = list(candidates[start : start + batch_size])
            encoded = tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.float().reshape(-1)
            scores.extend(float(value) for value in logits.detach().cpu().tolist())
    if len(scores) != len(candidates) or any(not math.isfinite(value) for value in scores):
        raise AssayError("student scorer returned an invalid score vector")
    return scores


def _teacher_score_maps(
    teacher: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, list[list[str]]]]:
    score_maps: dict[str, dict[str, float]] = {}
    groups_by_question: dict[str, list[list[str]]] = {}
    for row in teacher["rows"]:
        question_id = str(row["question_id"])
        scores: dict[str, float] = {}
        groups: list[list[str]] = []
        for group in row["groups"]:
            ids: list[str] = []
            for score in group["scores"]:
                memory_id = str(score["memory_id"])
                scores[memory_id] = _parse_float_hex(
                    score["utility_hex"], "teacher utility"
                )
                ids.append(memory_id)
            groups.append(ids)
        score_maps[question_id] = scores
        groups_by_question[question_id] = groups
    return score_maps, groups_by_question


def _ranked_ids(scores: Mapping[str, float]) -> list[str]:
    return sorted(scores, key=lambda key: (float(scores[key]), key), reverse=True)


def _agreement_metrics(
    student_scores: Mapping[str, Mapping[str, float]],
    teacher_scores: Mapping[str, Mapping[str, float]],
    groups_by_question: Mapping[str, Sequence[Sequence[str]]],
) -> dict[str, Any]:
    top1_correct = 0
    pair_correct = 0
    pair_total = 0
    margins: list[float] = []
    for question_id in sorted(student_scores):
        student = student_scores[question_id]
        teacher = teacher_scores[question_id]
        student_order = _ranked_ids(student)
        teacher_order = _ranked_ids(teacher)
        top1_correct += int(student_order[0] == teacher_order[0])
        margins.append(float(student[student_order[0]]) - float(student[student_order[1]]))
        candidate_ids = [
            str(memory_id)
            for group in groups_by_question[question_id]
            for memory_id in group
        ]
        for left in range(len(candidate_ids)):
            for right in range(left + 1, len(candidate_ids)):
                left_id = candidate_ids[left]
                right_id = candidate_ids[right]
                teacher_prefers_left = (teacher[left_id], left_id) > (
                    teacher[right_id],
                    right_id,
                )
                student_prefers_left = (student[left_id], left_id) > (
                    student[right_id],
                    right_id,
                )
                pair_correct += int(teacher_prefers_left == student_prefers_left)
                pair_total += 1
    question_count = len(student_scores)
    return {
        "questions": question_count,
        "top1_agree": top1_correct,
        "top1_agreement_hex": _float_hex(top1_correct / max(1, question_count)),
        "pairs": pair_total,
        "pairwise_agree": pair_correct,
        "pairwise_agreement_hex": _float_hex(pair_correct / max(1, pair_total)),
        "student_margin_p50_hex": _float_hex(_percentile(margins, 0.5) or 0.0),
        "student_margin_p95_hex": _float_hex(_percentile(margins, 0.95) or 0.0),
    }


def _score_partition(
    plane: Mapping[str, Any],
    partition: str,
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    max_length: int,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    policy = _require_mapping(plane["candidate_policy"], "candidate policy")
    _questions, memories = _plane_indexes(plane)
    output: dict[str, dict[str, float]] = {}
    for question in plane["questions"]:
        if question["partition"] != partition:
            continue
        query = _query_proxy(question, policy)
        candidate_ids = list(question["candidate_memory_ids"])
        texts = [_memory_proxy(memories[memory_id], policy) for memory_id in candidate_ids]
        values = _student_scores(
            torch,
            tokenizer,
            model,
            device,
            query,
            texts,
            max_length=max_length,
            batch_size=batch_size,
        )
        output[str(question["question_id"])] = dict(zip(candidate_ids, values, strict=True))
    return output


def _ranknet_pairs(
    plane: Mapping[str, Any],
    teacher_scores: Mapping[str, Mapping[str, float]],
    groups_by_question: Mapping[str, Sequence[Sequence[str]]],
    *,
    temperature: float,
) -> list[tuple[str, str, str, float]]:
    policy = _require_mapping(plane["candidate_policy"], "candidate policy")
    _questions, memories = _plane_indexes(plane)
    pairs: list[tuple[str, str, str, float]] = []
    for question in plane["questions"]:
        if question["partition"] != "train":
            continue
        question_id = str(question["question_id"])
        query = _query_proxy(question, policy)
        teacher = teacher_scores[question_id]
        candidate_ids = [
            str(memory_id)
            for group in groups_by_question[question_id]
            for memory_id in group
        ]
        for left in range(len(candidate_ids)):
            for right in range(left + 1, len(candidate_ids)):
                left_id = candidate_ids[left]
                right_id = candidate_ids[right]
                delta = (teacher[left_id] - teacher[right_id]) / temperature
                if delta >= 40.0:
                    target = 1.0
                elif delta <= -40.0:
                    target = 0.0
                else:
                    target = 1.0 / (1.0 + math.exp(-delta))
                pairs.append(
                    (
                        query,
                        _memory_proxy(memories[left_id], policy),
                        _memory_proxy(memories[right_id], policy),
                        target,
                    )
                )
    return pairs


def _fine_tune_ranknet(
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    pairs: Sequence[tuple[str, str, str, float]],
    *,
    epochs: int,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[str], float]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.BCEWithLogitsLoss()
    indices = list(range(len(pairs)))
    losses: list[str] = []
    started = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(indices)
        epoch_total = 0.0
        epoch_examples = 0
        for offset in range(0, len(indices), batch_size):
            selected = [pairs[index] for index in indices[offset : offset + batch_size]]
            queries = [row[0] for row in selected]
            left_texts = [row[1] for row in selected]
            right_texts = [row[2] for row in selected]
            encoded = tokenizer(
                queries + queries,
                left_texts + right_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            target = torch.tensor(
                [row[3] for row in selected], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(**encoded).logits.float().reshape(-1)
            count = len(selected)
            loss = loss_function(logits[:count] - logits[count:], target)
            loss.backward()
            optimizer.step()
            epoch_total += float(loss.detach().cpu()) * count
            epoch_examples += count
        losses.append(_float_hex(epoch_total / max(1, epoch_examples)))
    _synchronize(torch, device)
    elapsed = time.perf_counter() - started
    model.eval()
    return losses, elapsed


def _calibrate_margin(
    student_scores: Mapping[str, Mapping[str, float]],
    teacher_scores: Mapping[str, Mapping[str, float]],
    *,
    precision_target: float,
    minimum_fast: int,
) -> dict[str, Any]:
    observations: list[tuple[float, bool]] = []
    for question_id, student in student_scores.items():
        order = _ranked_ids(student)
        margin = float(student[order[0]]) - float(student[order[1]])
        teacher_top = _ranked_ids(teacher_scores[question_id])[0]
        observations.append((margin, order[0] == teacher_top))
    selected: tuple[float, int, int] | None = None
    # Match QKOVDistilledSelector exactly: only a strict positive margin above
    # its non-negative min_margin enters the distilled fast path.
    thresholds = {0.0, *(margin for margin, _correct in observations)}
    for threshold in sorted(thresholds):
        fast = [correct for margin, correct in observations if margin > threshold]
        correct = sum(fast)
        if len(fast) >= minimum_fast and correct / len(fast) >= precision_target:
            selected = (threshold, len(fast), correct)
            break
    if selected is None:
        return {
            "status": "no_threshold_met_precision_and_minimum_fast_count",
            "threshold_hex": None,
            "precision_target_hex": _float_hex(precision_target),
            "minimum_fast_count": minimum_fast,
            "fast_count": 0,
            "fallback_count": len(observations),
            "fast_precision_hex": None,
            "fast_coverage_hex": _float_hex(0.0),
        }
    threshold, fast_count, correct = selected
    return {
        "status": "calibrated_on_calibration_partition_only",
        "threshold_hex": _float_hex(threshold),
        "precision_target_hex": _float_hex(precision_target),
        "minimum_fast_count": minimum_fast,
        "fast_count": fast_count,
        "fallback_count": len(observations) - fast_count,
        "fast_precision_hex": _float_hex(correct / fast_count),
        "fast_coverage_hex": _float_hex(fast_count / len(observations)),
    }


def _uses_fast_path(margin: float, threshold: float | None) -> bool:
    """Replay the production selector's strict, non-negative margin gate."""

    return threshold is not None and float(margin) > float(threshold)


def _model_inventory(root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AssayError("student model directory contains a symlink")
        if not path.is_file():
            continue
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not inventory:
        raise AssayError("student model directory is empty")
    return inventory


def _save_student_no_clobber(
    model: Any, tokenizer: Any, output_dir: Path
) -> tuple[list[dict[str, Any]], str]:
    output_dir = _runtime_output_path(output_dir, "student model output")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"student model output already exists: {output_dir}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    moved = False
    try:
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        inventory = _model_inventory(staging)
        identity = _identity(inventory)
        os.replace(staging, output_dir)
        moved = True
        return inventory, identity
    finally:
        if not moved and staging.exists():
            shutil.rmtree(staging)


def _train(args: argparse.Namespace) -> dict[str, Any]:
    from memory_condense.search.selectors.cross_encoder_selector import (
        MS_MARCO_MODEL_ID,
        MS_MARCO_MODEL_REVISION,
    )

    plane, plane_sha = _read_sealed(args.plane, label="projection")
    _validate_plane(plane)
    teacher, teacher_sha = _read_sealed(args.teacher, label="teacher artifact")
    _validate_teacher(teacher, plane, plane_sha)
    teacher_scores, groups_by_question = _teacher_score_maps(teacher)
    torch, tokenizer, model, device, checkpoint_sha = _load_student(
        args.minilm_model_dir,
        device_name=args.device,
        verify_base=True,
    )
    if checkpoint_sha is None:  # pragma: no cover - defensive
        raise AssayError("base MiniLM verification did not return an identity")

    # The zero-shot development score is deliberately computed before any
    # update. Test remains unseen until the student and threshold are fixed.
    zero_calibration_scores = _score_partition(
        plane,
        "calibration",
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
        batch_size=args.score_batch_size,
    )
    zero_calibration_metrics = _agreement_metrics(
        zero_calibration_scores, teacher_scores, groups_by_question
    )
    top1_rate = _parse_float_hex(
        zero_calibration_metrics["top1_agreement_hex"], "zero-shot top-1 agreement"
    )
    pair_rate = _parse_float_hex(
        zero_calibration_metrics["pairwise_agreement_hex"], "zero-shot pairwise agreement"
    )
    gate_passed = top1_rate >= args.top1_gate and pair_rate >= args.pairwise_gate

    training_pairs = _ranknet_pairs(
        plane,
        teacher_scores,
        groups_by_question,
        temperature=args.temperature,
    )
    epoch_losses: list[str] = []
    training_seconds = 0.0
    if not gate_passed:
        epoch_losses, training_seconds = _fine_tune_ranknet(
            torch,
            tokenizer,
            model,
            device,
            training_pairs,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )

    calibration_scores = _score_partition(
        plane,
        "calibration",
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
        batch_size=args.score_batch_size,
    )
    calibration_metrics = _agreement_metrics(
        calibration_scores, teacher_scores, groups_by_question
    )
    calibration = _calibrate_margin(
        calibration_scores,
        teacher_scores,
        precision_target=args.fast_precision,
        minimum_fast=args.minimum_fast,
    )

    model_output = _runtime_output_path(args.model_output, "student model output")
    inventory, model_identity = _save_student_no_clobber(
        model, tokenizer, model_output
    )
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    payload: dict[str, Any] = {
        "format": TRAINING_FORMAT,
        "status": "zero_shot_retained" if gate_passed else "ranknet_fine_tuned",
        "promotion_eligible": False,
        "limitations": [
            *plane["limitations"],
            "teacher agreement is not answer recall and this pilot cannot promote",
        ],
        "plane_sha256": plane_sha,
        "teacher_sha256": teacher_sha,
        "base_model_identity": {
            "model_id": MS_MARCO_MODEL_ID,
            "model_revision": MS_MARCO_MODEL_REVISION,
            "checkpoint_sha256": checkpoint_sha,
        },
        "training_policy": {
            "objective": "manual-pytorch-ranknet-soft-pairwise-bce",
            "teacher_pair_scope": "all_pairs_across_16_order_invariant_scalar_scores",
            "temperature_hex": _float_hex(args.temperature),
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "score_batch_size": args.score_batch_size,
            "max_length": args.max_length,
            "candidate_tokens": DEFAULT_CANDIDATE_TOKENS,
            "learning_rate_hex": _float_hex(args.learning_rate),
            "seed": args.seed,
            "train_partition_only": True,
        },
        "zero_shot_gate": {
            "evaluated_before_training": True,
            "partition": "calibration",
            "top1_gate_hex": _float_hex(args.top1_gate),
            "pairwise_gate_hex": _float_hex(args.pairwise_gate),
            "passed": gate_passed,
            "metrics": zero_calibration_metrics,
            "test_not_scored_or_indexed_by_training_logic": True,
            "test_rows_process_visible_in_sealed_inputs": True,
        },
        "fine_tuning": {
            "performed": not gate_passed,
            "reason": (
                "zero_shot_met_both_gates"
                if gate_passed
                else "zero_shot_missed_at_least_one_gate"
            ),
            "pair_count": len(training_pairs),
            "epoch_mean_losses_hex": epoch_losses,
            "elapsed_seconds_hex": _float_hex(training_seconds),
        },
        "calibration": {
            **calibration,
            "partition": "calibration",
            "agreement_metrics": calibration_metrics,
        },
        "student_model": {
            "directory": str(model_output),
            "inventory": inventory,
            "identity_sha256": model_identity,
            "safe_serialization": True,
        },
        "counts": {
            "train_questions": PARTITION_COUNTS["train"],
            "calibration_questions": PARTITION_COUNTS["calibration"],
            "test_questions": PARTITION_COUNTS["test"],
            "pairs_per_train_question": math.comb(DEFAULT_CANDIDATES, 2),
            "train_pairs": len(training_pairs),
        },
    }
    _assert_gold_free(payload)
    return payload


def _verify_student_manifest(
    training: Mapping[str, Any],
    *,
    plane_sha: str,
    teacher_sha: str,
    model_dir: Path,
) -> str:
    if training.get("format") != TRAINING_FORMAT:
        raise AssayError("unsupported training artifact format")
    if training.get("promotion_eligible") is not False:
        raise AssayError("training artifact must remain ineligible for promotion")
    if training.get("plane_sha256") != plane_sha:
        raise AssayError("training artifact projection binding differs")
    if training.get("teacher_sha256") != teacher_sha:
        raise AssayError("training artifact teacher binding differs")
    student = _require_mapping(training.get("student_model"), "student model manifest")
    inventory = _require_list(student.get("inventory"), "student model inventory")
    if _identity(inventory) != student.get("identity_sha256"):
        raise AssayError("student model inventory identity differs")
    observed = _model_inventory(model_dir)
    if observed != inventory:
        raise AssayError("student model files differ from sealed inventory")
    _assert_gold_free(training)
    return _require_text(student.get("identity_sha256"), "student identity")


def _live_teacher_scores(
    linker: Any,
    query: str,
    candidate_ids: Sequence[str],
    memories: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, float]:
    from memory_condense.associations.head_memory_models import (
        AssociativeMemoryCandidate,
    )

    scores: dict[str, float] = {}
    for start in range(0, len(candidate_ids), DEFAULT_GROUP_SIZE):
        group_ids = list(candidate_ids[start : start + DEFAULT_GROUP_SIZE])
        result = linker.inspect_coverage(
            query,
            [
                AssociativeMemoryCandidate(
                    episode_id=memory_id,
                    text=_memory_proxy(memories[memory_id], policy),
                )
                for memory_id in group_ids
            ],
        )
        if result.workspace_candidates != len(group_ids):
            raise AssayError("live teacher omitted a fallback candidate")
        for hit in result.hits:
            scores[hit.episode_id] = _teacher_utility(
                hit.qk_score, hit.ov_transport
            )
    if set(scores) != set(candidate_ids):
        raise AssayError("live teacher fallback score set differs")
    return scores


def _latency_summary(values_ms: Sequence[float]) -> dict[str, Any]:
    if not values_ms:
        return {
            "samples": 0,
            "mean_ms_hex": None,
            "p50_ms_hex": None,
            "p95_ms_hex": None,
            "minimum_ms_hex": None,
            "maximum_ms_hex": None,
        }
    return {
        "samples": len(values_ms),
        "mean_ms_hex": _float_hex(statistics.fmean(values_ms)),
        "p50_ms_hex": _float_hex(_percentile(values_ms, 0.5) or 0.0),
        "p95_ms_hex": _float_hex(_percentile(values_ms, 0.95) or 0.0),
        "minimum_ms_hex": _float_hex(min(values_ms)),
        "maximum_ms_hex": _float_hex(max(values_ms)),
    }


def _assay(args: argparse.Namespace) -> dict[str, Any]:
    plane, plane_sha = _read_sealed(args.plane, label="projection")
    _validate_plane(plane)
    teacher, teacher_sha = _read_sealed(args.teacher, label="teacher artifact")
    _validate_teacher(teacher, plane, plane_sha)
    training, training_sha = _read_sealed(args.training, label="training artifact")
    training_policy = _require_mapping(
        training.get("training_policy"), "training scoring policy"
    )
    if int(training_policy.get("max_length", 0)) != args.max_length:
        raise AssayError("assay max length differs from the sealed training contract")
    if int(training_policy.get("score_batch_size", 0)) != args.score_batch_size:
        raise AssayError("assay score batch size differs from the sealed training contract")
    if int(training_policy.get("candidate_tokens", 0)) != DEFAULT_CANDIDATE_TOKENS:
        raise AssayError("sealed training candidate-token contract differs")
    student_manifest = _require_mapping(
        training.get("student_model"), "student model manifest"
    )
    model_dir = (
        args.model_dir.resolve()
        if args.model_dir is not None
        else Path(
            _require_text(student_manifest.get("directory"), "student directory")
        ).resolve()
    )
    student_identity = _verify_student_manifest(
        training,
        plane_sha=plane_sha,
        teacher_sha=teacher_sha,
        model_dir=model_dir,
    )
    teacher_scores, groups_by_question = _teacher_score_maps(teacher)
    policy = _require_mapping(plane["candidate_policy"], "candidate policy")
    _questions, memories = _plane_indexes(plane)
    threshold_raw = training["calibration"].get("threshold_hex")
    threshold = (
        None
        if threshold_raw is None
        else _parse_float_hex(threshold_raw, "calibrated margin threshold")
    )

    # Measure the held-out zero-shot baseline only after the student and its
    # threshold have been fixed; this measurement cannot influence training.
    torch, base_tokenizer, base_model, base_device, base_checkpoint_sha = _load_student(
        args.base_model_dir,
        device_name=args.device,
        verify_base=True,
    )
    heldout_zero_shot_scores = _score_partition(
        plane,
        "test",
        torch=torch,
        tokenizer=base_tokenizer,
        model=base_model,
        device=base_device,
        max_length=args.max_length,
        batch_size=args.score_batch_size,
    )
    heldout_zero_shot_metrics = _agreement_metrics(
        heldout_zero_shot_scores, teacher_scores, groups_by_question
    )
    del base_model, base_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch, tokenizer, model, device, _checkpoint_sha = _load_student(
        model_dir,
        device_name=args.device,
        verify_base=False,
    )
    # Exclude model load and allocator initialization. Both a tokenizer/model
    # warmup and a few complete proxy-render/scoring warmups precede timing.
    test_questions = [
        question for question in plane["questions"] if question["partition"] == "test"
    ]
    warm_question = test_questions[0]
    warm_query = _query_proxy(warm_question, policy)
    warm_ids = list(warm_question["candidate_memory_ids"])
    warm_texts = [_memory_proxy(memories[memory_id], policy) for memory_id in warm_ids]
    for _ in range(args.warmup):
        _student_scores(
            torch,
            tokenizer,
            model,
            device,
            warm_query,
            warm_texts,
            max_length=args.max_length,
            batch_size=args.score_batch_size,
        )
    _synchronize(torch, device)

    test_scores = _score_partition(
        plane,
        "test",
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
        batch_size=args.score_batch_size,
    )
    student_metrics = _agreement_metrics(
        test_scores, teacher_scores, groups_by_question
    )
    decision_rows: list[dict[str, Any]] = []
    projected_correct = 0
    fallback_required_count = 0
    for question in test_questions:
        question_id = str(question["question_id"])
        scores = test_scores[question_id]
        order = _ranked_ids(scores)
        margin = scores[order[0]] - scores[order[1]]
        fallback_required = not _uses_fast_path(margin, threshold)
        teacher_top = _ranked_ids(teacher_scores[question_id])[0]
        selected = teacher_top if fallback_required else order[0]
        correct = selected == teacher_top
        projected_correct += int(correct)
        fallback_required_count += int(fallback_required)
        decision_rows.append(
            {
                "question_id": question_id,
                "student_top_memory_id": order[0],
                "teacher_top_memory_id": teacher_top,
                "student_margin_hex": _float_hex(margin),
                "fallback_required": fallback_required,
                "upper_bound_selected_memory_id": selected,
                "oracle_fallback_upper_bound_teacher_agreement": correct,
            }
        )

    live_encoder: Any | None = None
    live_linker: Any | None = None
    live_checks = 0
    live_matches = 0
    if args.qwen_model_dir is not None:
        from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
        from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder

        identity = _require_mapping(teacher["teacher_identity"], "teacher identity")
        live_encoder = Qwen3PrefixEncoder(
            args.qwen_model_dir.resolve(),
            layers=int(identity["layers"]),
            device=args.device,
            dtype=str(identity["dtype"]),
        )
        if live_encoder.checkpoint_sha256 != identity["checkpoint_sha256"]:
            raise AssayError("live Qwen checkpoint differs from sealed teacher")
        live_linker = QwenMemoryLinker(
            live_encoder,
            layer=int(identity["attention_layer"]),
            max_candidates=DEFAULT_GROUP_SIZE,
            max_workspace_tokens=args.workspace_tokens,
        )
        # Warm the exact fallback path separately from scored timings.
        _live_teacher_scores(live_linker, warm_query, warm_ids, memories, policy)

    student_latency_ms: list[float] = []
    hybrid_latency_ms: list[float] = []
    render_latency_ms: list[float] = []
    live_fallback_available = live_linker is not None
    try:
        for _repeat in range(args.repeats):
            for question in test_questions:
                materialize_started = time.perf_counter()
                query = _query_proxy(question, policy)
                candidate_ids = list(question["candidate_memory_ids"])
                candidate_texts = [
                    _memory_proxy(memories[memory_id], policy)
                    for memory_id in candidate_ids
                ]
                render_elapsed = time.perf_counter() - materialize_started
                _synchronize(torch, device)
                score_started = time.perf_counter()
                values = _student_scores(
                    torch,
                    tokenizer,
                    model,
                    device,
                    query,
                    candidate_texts,
                    max_length=args.max_length,
                    batch_size=args.score_batch_size,
                )
                _synchronize(torch, device)
                student_elapsed = time.perf_counter() - score_started
                student_map = dict(zip(candidate_ids, values, strict=True))
                order = _ranked_ids(student_map)
                margin = student_map[order[0]] - student_map[order[1]]
                fallback_required = not _uses_fast_path(margin, threshold)
                hybrid_started = materialize_started
                if fallback_required and live_linker is not None:
                    live = _live_teacher_scores(
                        live_linker, query, candidate_ids, memories, policy
                    )
                    live_checks += 1
                    question_id = str(question["question_id"])
                    live_matches += int(
                        _ranked_ids(live)[0]
                        == _ranked_ids(teacher_scores[question_id])[0]
                    )
                hybrid_elapsed = time.perf_counter() - hybrid_started
                render_latency_ms.append(render_elapsed * 1000.0)
                student_latency_ms.append(student_elapsed * 1000.0)
                hybrid_latency_ms.append(hybrid_elapsed * 1000.0)
    finally:
        del model, tokenizer
        if live_linker is not None:
            del live_linker
        if live_encoder is not None:
            del live_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload: dict[str, Any] = {
        "format": ASSAY_FORMAT,
        "status": "proxy_teacher_student_assay_complete",
        "promotion_eligible": False,
        "limitations": [
            *plane["limitations"],
            "test cards are deterministic oracle-turn proxies, not production Transcript cards",
            "calibration controls both the train-or-skip gate and margin threshold",
            "the held-out test was not scored until student and threshold were fixed",
            (
                "latency is warm proxy rendering plus student scoring only; live Qwen fallback was not supplied"
                if not live_fallback_available
                else "latency includes warm proxy rendering, student scoring, and required live Qwen fallback"
            ),
            "lexical candidate construction, corpus lookup, and final answer generation are excluded",
            "the student cross-encoder is an assay artifact and is not wired into the production selector",
        ],
        "plane_sha256": plane_sha,
        "teacher_sha256": teacher_sha,
        "training_sha256": training_sha,
        "student_identity_sha256": student_identity,
        "base_checkpoint_sha256": base_checkpoint_sha,
        "heldout_zero_shot_agreement": heldout_zero_shot_metrics,
        "test_agreement": student_metrics,
        "scoring_contract": {
            "max_length": args.max_length,
            "score_batch_size": args.score_batch_size,
            "candidate_tokens": DEFAULT_CANDIDATE_TOKENS,
            "query_tokens": DEFAULT_QUERY_TOKENS,
            "candidate_count": DEFAULT_CANDIDATES,
        },
        "margin_policy": {
            "threshold_hex": threshold_raw,
            "threshold_source": "calibration_partition_only",
            "fast_path_when": "student_margin_strictly_greater_than_threshold",
            "fallback_required": fallback_required_count,
            "fast_path": len(test_questions) - fallback_required_count,
            "oracle_fallback_upper_bound_teacher_agree": projected_correct,
            "oracle_fallback_upper_bound_teacher_agreement_hex": _float_hex(
                projected_correct / len(test_questions)
            ),
        },
        "latency": {
            "scope": (
                "warm_proxy_render_student_and_live_qwen_fallback"
                if live_fallback_available
                else "warm_projected_proxy_render_and_student_scorer_only"
            ),
            "model_load_excluded": True,
            "candidate_count": DEFAULT_CANDIDATES,
            "candidate_tokens": DEFAULT_CANDIDATE_TOKENS,
            "repeats": args.repeats,
            "proxy_render": _latency_summary(render_latency_ms),
            "student_scorer": _latency_summary(student_latency_ms),
            "observed_pipeline": _latency_summary(hybrid_latency_ms),
            "live_fallback_available": live_fallback_available,
            "live_fallback_checks": live_checks,
            "live_fallback_top1_matches": live_matches,
            "live_fallback_match_hex": (
                None if live_checks == 0 else _float_hex(live_matches / live_checks)
            ),
        },
        "rows": decision_rows,
    }
    _assert_gold_free(payload)
    return payload


def _preflight_output(path: Path, label: str) -> Path:
    output = _runtime_output_path(path, label)
    sidecar = output.with_name(output.name + ".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"{label} or its sidecar already exists: {output}")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    project = subparsers.add_parser("project", help="build the sealed gold-blind plane")
    project.add_argument("--dataset", type=Path, required=True)
    project.add_argument("--split-manifest", type=Path, required=True)
    project.add_argument("--output", type=Path, required=True)
    project.set_defaults(
        handler=_project,
        candidates=DEFAULT_CANDIDATES,
        lexical=DEFAULT_LEXICAL,
        distractors=DEFAULT_DISTRACTORS,
        candidate_tokens=DEFAULT_CANDIDATE_TOKENS,
        query_tokens=DEFAULT_QUERY_TOKENS,
    )

    teacher = subparsers.add_parser("teacher", help="run the sealed Qwen teacher")
    teacher.add_argument("--plane", type=Path, required=True)
    teacher.add_argument("--qwen-model-dir", type=Path, required=True)
    teacher.add_argument("--output", type=Path, required=True)
    teacher.add_argument("--layers", type=int, default=6)
    teacher.add_argument("--attention-layer", type=int, default=5)
    teacher.add_argument("--dtype", default="float16")
    teacher.add_argument("--device", default="cuda")
    teacher.add_argument("--workspace-tokens", type=int, default=2048)
    teacher.set_defaults(handler=_teacher)

    train = subparsers.add_parser("train", help="zero-shot gate and optional RankNet tune")
    train.add_argument("--plane", type=Path, required=True)
    train.add_argument("--teacher", type=Path, required=True)
    train.add_argument("--minilm-model-dir", type=Path, required=True)
    train.add_argument("--model-output", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--train-batch-size", type=int, default=16)
    train.add_argument("--score-batch-size", type=int, default=16)
    train.add_argument("--max-length", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--temperature", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=20260905)
    train.add_argument("--top1-gate", type=float, default=0.95)
    train.add_argument("--pairwise-gate", type=float, default=0.97)
    train.add_argument("--fast-precision", type=float, default=0.95)
    train.add_argument("--minimum-fast", type=int, default=5)
    train.set_defaults(handler=_train)

    assay = subparsers.add_parser("assay", help="measure test agreement and warm latency")
    assay.add_argument("--plane", type=Path, required=True)
    assay.add_argument("--teacher", type=Path, required=True)
    assay.add_argument("--training", type=Path, required=True)
    assay.add_argument("--base-model-dir", type=Path, required=True)
    assay.add_argument("--model-dir", type=Path)
    assay.add_argument("--qwen-model-dir", type=Path)
    assay.add_argument("--output", type=Path, required=True)
    assay.add_argument("--device", default="cuda")
    assay.add_argument("--score-batch-size", type=int, default=16)
    assay.add_argument("--max-length", type=int, default=128)
    assay.add_argument("--workspace-tokens", type=int, default=2048)
    assay.add_argument("--warmup", type=int, default=5)
    assay.add_argument("--repeats", type=int, default=5)
    assay.set_defaults(handler=_assay)
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.command == "teacher":
        if args.layers < 1 or not 0 <= args.attention_layer < args.layers:
            raise AssayError("attention layer must lie inside the loaded prefix")
        if args.workspace_tokens < 1:
            raise AssayError("workspace tokens must be positive")
    if args.command == "train":
        for name in ("epochs", "train_batch_size", "score_batch_size", "max_length"):
            if int(getattr(args, name)) < 1:
                raise AssayError(f"{name} must be positive")
        if args.learning_rate <= 0.0 or args.temperature <= 0.0:
            raise AssayError("learning rate and temperature must be positive")
        for name in ("top1_gate", "pairwise_gate", "fast_precision"):
            if not 0.0 <= float(getattr(args, name)) <= 1.0:
                raise AssayError(f"{name} must be in [0, 1]")
        if args.minimum_fast < 1 or args.minimum_fast > PARTITION_COUNTS["calibration"]:
            raise AssayError("minimum fast count is outside the calibration population")
    if args.command == "assay":
        if args.score_batch_size < 1 or args.max_length < 1:
            raise AssayError("assay scorer sizes must be positive")
        if args.warmup < 1 or args.repeats < 1 or args.workspace_tokens < 1:
            raise AssayError("assay warmup, repeats, and workspace must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_arguments(args)
        output = _preflight_output(args.output, f"{args.command} output")
        if args.command == "train":
            _preflight_output(args.model_output, "student model output")
        payload = args.handler(args)
        digest, sidecar = _publish_json(output, payload)
        print(
            _canonical_json(
                {
                    "command": args.command,
                    "output": str(output),
                    "sha256": digest,
                    "sha256_file": str(sidecar),
                    "status": payload["status"],
                    "promotion_eligible": False,
                }
            )
        )
        return 0
    except (AssayError, FileExistsError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
