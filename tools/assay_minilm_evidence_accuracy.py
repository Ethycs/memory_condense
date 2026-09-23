"""Development-only MiniLM evidence-ranking accuracy assay.

This command deliberately uses LongMemEval development truth to answer one
narrow question: can the already-fast MiniLM scorer learn to put evidence in
the first eight positions of a query-only lexical frontier?  It is not a
gold-blind retrieval run and can never promote a production policy.

The input plane is the canonical, gold-free artifact produced by
``assay_qwen_minilm_distillation.py``.  This assay verifies that plane against
the original oracle projection, rebuilds a pure lexical-96 frontier inside
each of the plane's three source-disjoint partitions, and evaluates five
freshly initialized models out of fold.  Gold labels never enter model text:
they are used only to construct the loss and post-hoc development metrics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT), str(_ROOT / "tools")]
else:  # pragma: no cover - module execution
    _ROOT = Path(__file__).resolve().parents[1]


FORMAT = "memory-condense-minilm-evidence-accuracy-oof-v1"
STATUS = "analysis_used_development_oof_complete_nonpromotable"
FOLD_SALT = "memory-condense-gold-listwise-v1-2026-09-05"
FOLD_COUNTS = {f"fold-{index}": 40 for index in range(5)}
BASE_SEED = 20260905

CANDIDATES = 96
TOP_K = 8
QUERY_TOKENS = 48
CANDIDATE_TOKENS = 48
MAX_LENGTH = 128
SCORE_SUB_BATCH = 32
EFFECTIVE_QUESTION_BATCH = 4
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.10
GRADIENT_CLIP = 1.0
SMOOTHMAX_TAU = 0.25
RANK_MARGIN = 0.2
LOSS_WEIGHTS = {
    "any_at_8": 1.0,
    "source_balanced_at_8": 0.5,
    "exact_at_8": 0.5,
    "any_at_1": 0.25,
}
LATENCY_WARMUP_QUESTIONS = 3


class AssayError(RuntimeError):
    """An input or runtime invariant of the sealed assay was violated."""


@dataclass(frozen=True, slots=True)
class GoldCoordinates:
    required_source_ids: tuple[str, ...]
    exact_memory_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedQuestion:
    question_id: str
    question_type: str
    original_partition: str
    fold: str
    query: str
    candidate_ids: tuple[str, ...]
    candidate_texts: tuple[str, ...]
    candidate_source_ids: tuple[str, ...]
    required_source_ids: tuple[str, ...]
    exact_memory_ids: tuple[str, ...]


def _pilot() -> Any:
    # Keep torch/transformers and even the existing assay helpers off --help.
    from tools import assay_qwen_minilm_distillation

    return assay_qwen_minilm_distillation


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


def _float_hex(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise AssayError("a metric is not finite")
    return number.hex()


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


def _runtime_output(path: Path) -> Path:
    resolved = path.resolve()
    runtime_root = (_ROOT / ".tmp").resolve()
    if not resolved.is_relative_to(runtime_root):
        raise AssayError("output must be under the worktree .tmp directory")
    sidecar = resolved.with_name(resolved.name + ".sha256")
    if resolved.exists() or sidecar.exists():
        raise FileExistsError(f"output or sidecar already exists: {resolved}")
    return resolved


def _publish(path: Path, payload: Mapping[str, Any]) -> tuple[str, Path]:
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    digest = _sha256_bytes(encoded)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--minilm-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def _load_plane_and_dataset(
    plane_path: Path,
    dataset_path: Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], str]:
    pilot = _pilot()
    plane, plane_sha256 = pilot._read_sealed(plane_path, label="gold-free plane")
    pilot._validate_plane(plane)
    dataset_path = dataset_path.resolve()
    if dataset_path.is_symlink() or not dataset_path.is_file():
        raise AssayError("dataset must be a regular non-symlink file")
    dataset_sha256 = _file_sha256(dataset_path)
    if dataset_sha256 != plane.get("projection_dataset_sha256"):
        raise AssayError("oracle projection bytes differ from the plane binding")
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssayError(f"cannot decode oracle projection: {exc}") from exc
    records = _require_list(raw, "oracle projection")
    typed: list[dict[str, Any]] = []
    for index, value in enumerate(records):
        typed.append(_require_mapping(value, f"oracle record {index}"))
    return plane, plane_sha256, typed, dataset_sha256


def _reconstruct_gold_coordinates(
    plane: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, GoldCoordinates]:
    """Rebuild opaque development coordinates and verify the full plane.

    Answer strings are never read.  Only ``answer_session_ids`` and boolean
    ``has_answer`` annotations are projected into in-memory training labels.
    """

    pilot = _pilot()
    questions = {
        str(row["question_id"]): row
        for row in _require_list(plane.get("questions"), "plane questions")
    }
    if len(questions) != 200:
        raise AssayError("the accuracy assay requires all 200 development questions")
    raw_by_id: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        question_id = _require_text(
            record.get("question_id"), f"oracle record {index} question_id"
        )
        if question_id in raw_by_id:
            raise AssayError(f"duplicate oracle question_id: {question_id}")
        raw_by_id[question_id] = record
    if any(question_id not in raw_by_id for question_id in questions):
        raise AssayError("the oracle projection lacks a plane question")

    expected_memories: dict[str, dict[str, Any]] = {}
    coordinates: dict[str, GoldCoordinates] = {}
    for question_id, question in questions.items():
        record = raw_by_id[question_id]
        if _require_text(record.get("question"), "oracle question") != question["question"]:
            raise AssayError(f"question text differs for {question_id}")
        expected_type = str(record.get("question_type") or "uncategorized")
        if expected_type != question["question_type"]:
            raise AssayError(f"question type differs for {question_id}")
        raw_date = record.get("question_date")
        expected_date = None if raw_date is None else str(raw_date).strip()
        if expected_date != question.get("question_date"):
            raise AssayError(f"question date differs for {question_id}")

        sessions = _require_list(
            record.get("haystack_sessions"), f"{question_id} haystack_sessions"
        )
        session_ids_value = record.get("haystack_session_ids")
        session_dates_value = record.get("haystack_dates")
        session_ids = session_ids_value if isinstance(session_ids_value, list) else []
        session_dates = (
            session_dates_value if isinstance(session_dates_value, list) else []
        )
        answer_ids_value = record.get("answer_session_ids")
        answer_ids = _require_list(
            answer_ids_value, f"{question_id} answer_session_ids"
        )
        if not answer_ids or any(
            not isinstance(value, str) or not value.strip() for value in answer_ids
        ):
            raise AssayError(f"{question_id} has invalid answer-session coordinates")
        normalized_answers = {str(value) for value in answer_ids}
        required_sources: list[str] = []
        exact_memories: list[str] = []
        matched_answer_ids: set[str] = set()
        partition = str(question["partition"])

        for session_index, raw_session in enumerate(sessions):
            session = _require_list(
                raw_session, f"{question_id} session {session_index}"
            )
            raw_source_id = (
                str(session_ids[session_index])
                if session_index < len(session_ids)
                else f"session-{session_index}"
            )
            opaque_source = pilot._opaque_id(
                "source", question_id, str(session_index), raw_source_id
            )
            is_required = raw_source_id in normalized_answers
            if is_required:
                matched_answer_ids.add(raw_source_id)
                required_sources.append(opaque_source)
            source_date = (
                str(session_dates[session_index]).strip()
                if session_index < len(session_dates)
                and session_dates[session_index] is not None
                else None
            )
            for turn_index, raw_turn_value in enumerate(session):
                raw_turn = _require_mapping(
                    raw_turn_value,
                    f"{question_id} session {session_index} turn {turn_index}",
                )
                has_answer = raw_turn.get("has_answer")
                if has_answer is not None and type(has_answer) is not bool:
                    raise AssayError("has_answer annotations must be exact booleans")
                text = pilot._raw_text(raw_turn)
                if not text:
                    continue
                memory_id = pilot._opaque_id(
                    "memory",
                    question_id,
                    str(session_index),
                    str(turn_index),
                    _sha256_bytes(text.encode("utf-8")),
                )
                if memory_id in expected_memories:
                    raise AssayError(f"duplicate reconstructed memory: {memory_id}")
                expected_memories[memory_id] = {
                    "memory_id": memory_id,
                    "source_id": opaque_source,
                    "source_date": source_date,
                    "role": pilot._raw_role(raw_turn.get("role"), turn_index),
                    "text": text,
                    "partition": partition,
                }
                if is_required and has_answer is True:
                    exact_memories.append(memory_id)
        if matched_answer_ids != normalized_answers:
            raise AssayError(f"{question_id} answer session is absent from its history")
        coordinates[question_id] = GoldCoordinates(
            required_source_ids=tuple(dict.fromkeys(required_sources)),
            exact_memory_ids=tuple(dict.fromkeys(exact_memories)),
        )

    observed_memories = {
        str(row["memory_id"]): dict(row)
        for row in _require_list(plane.get("memories"), "plane memories")
    }
    if observed_memories != expected_memories:
        missing = sorted(set(expected_memories) - set(observed_memories))[:3]
        extra = sorted(set(observed_memories) - set(expected_memories))[:3]
        changed = sorted(
            memory_id
            for memory_id in set(observed_memories) & set(expected_memories)
            if observed_memories[memory_id] != expected_memories[memory_id]
        )[:3]
        raise AssayError(
            "oracle reconstruction differs from the sealed plane; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return coordinates


def _fold_assignments(plane: Mapping[str, Any]) -> dict[str, str]:
    pilot = _pilot()
    questions = [
        _require_mapping(row, "plane question")
        for row in _require_list(plane.get("questions"), "plane questions")
    ]
    folds = pilot._stratified_partitions(
        questions,
        counts=FOLD_COUNTS,
        salt=FOLD_SALT,
        id_key="question_id",
        stratum_key="question_type",
    )
    assignments = {
        str(question["question_id"]): fold
        for fold, members in folds.items()
        for question in members
    }
    if len(assignments) != 200 or Counter(assignments.values()) != Counter(FOLD_COUNTS):
        raise AssayError("five-fold assignment did not produce five groups of forty")
    return assignments


def _lexical_frontiers(
    plane: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Rebuild a pure BM25-style top-96 inside each original partition."""

    pilot = _pilot()
    questions = [
        _require_mapping(row, "plane question")
        for row in _require_list(plane.get("questions"), "plane questions")
    ]
    memories = [
        _require_mapping(row, "plane memory")
        for row in _require_list(plane.get("memories"), "plane memories")
    ]
    output: dict[str, tuple[str, ...]] = {}
    for partition in ("train", "calibration", "test"):
        pool = [row for row in memories if row["partition"] == partition]
        if len(pool) < CANDIDATES:
            raise AssayError(f"{partition} has fewer than {CANDIDATES} memories")
        doc_terms = {
            str(memory["memory_id"]): pilot._terms(str(memory["text"]))
            for memory in pool
        }
        term_frequencies = {
            memory_id: Counter(terms) for memory_id, terms in doc_terms.items()
        }
        document_lengths = {
            memory_id: len(terms) for memory_id, terms in doc_terms.items()
        }
        document_frequency: Counter[str] = Counter()
        for terms in doc_terms.values():
            document_frequency.update(set(terms))
        average_length = statistics.fmean(
            max(1, len(terms)) for terms in doc_terms.values()
        )
        for question in questions:
            if question["partition"] != partition:
                continue
            selected = pilot._candidate_ids(
                question,
                pool,
                term_frequencies=term_frequencies,
                document_lengths=document_lengths,
                document_frequency=document_frequency,
                average_length=average_length,
                lexical_count=CANDIDATES,
                distractor_count=0,
                salt=FOLD_SALT,
            )
            if len(selected) != CANDIDATES or len(set(selected)) != CANDIDATES:
                raise AssayError("pure lexical frontier has invalid cardinality")
            output[str(question["question_id"])] = tuple(selected)
    if len(output) != 200:
        raise AssayError("pure lexical frontiers do not cover development")
    return output


def _prepare_questions(
    plane: Mapping[str, Any],
    coordinates: Mapping[str, GoldCoordinates],
    folds: Mapping[str, str],
    frontiers: Mapping[str, Sequence[str]],
) -> list[PreparedQuestion]:
    pilot = _pilot()
    memory_by_id = {
        str(row["memory_id"]): row
        for row in _require_list(plane.get("memories"), "plane memories")
    }
    render_policy = {
        "query_tokens": QUERY_TOKENS,
        "candidate_tokens": CANDIDATE_TOKENS,
    }
    prepared: list[PreparedQuestion] = []
    for raw in _require_list(plane.get("questions"), "plane questions"):
        question = _require_mapping(raw, "plane question")
        question_id = str(question["question_id"])
        candidate_ids = tuple(str(value) for value in frontiers[question_id])
        candidates = [memory_by_id[memory_id] for memory_id in candidate_ids]
        gold = coordinates[question_id]
        prepared.append(
            PreparedQuestion(
                question_id=question_id,
                question_type=str(question["question_type"]),
                original_partition=str(question["partition"]),
                fold=folds[question_id],
                query=pilot._query_proxy(question, render_policy),
                candidate_ids=candidate_ids,
                candidate_texts=tuple(
                    pilot._memory_proxy(memory, render_policy) for memory in candidates
                ),
                candidate_source_ids=tuple(
                    str(memory["source_id"]) for memory in candidates
                ),
                required_source_ids=gold.required_source_ids,
                exact_memory_ids=gold.exact_memory_ids,
            )
        )
    prepared.sort(key=lambda row: row.question_id)
    return prepared


def _label_indices(
    question: PreparedQuestion,
) -> tuple[tuple[int, ...], dict[str, tuple[int, ...]], tuple[int, ...]]:
    required = set(question.required_source_ids)
    any_positive = tuple(
        index
        for index, source_id in enumerate(question.candidate_source_ids)
        if source_id in required
    )
    by_source = {
        source_id: tuple(
            index
            for index, observed in enumerate(question.candidate_source_ids)
            if observed == source_id
        )
        for source_id in question.required_source_ids
    }
    exact_set = set(question.exact_memory_ids)
    exact = tuple(
        index
        for index, memory_id in enumerate(question.candidate_ids)
        if memory_id in exact_set
    )
    return any_positive, by_source, exact


def _smoothmax(torch: Any, values: Any, *, tau: float) -> Any:
    if not math.isfinite(float(tau)) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("smoothmax requires a non-empty rank-one tensor")
    return float(tau) * (
        torch.logsumexp(values / float(tau), dim=0)
        - math.log(int(values.numel()))
    )


def _hit_at_k_loss(
    torch: Any,
    scores: Any,
    positive_indices: Sequence[int],
    negative_indices: Sequence[int],
    *,
    k: int,
    tau: float,
    margin: float,
) -> Any:
    """Smooth hinge for putting one positive above the kth negative."""

    if isinstance(k, bool) or int(k) < 1:
        raise ValueError("k must be positive")
    if not math.isfinite(float(margin)) or margin < 0.0:
        raise ValueError("margin must be finite and non-negative")
    if scores.ndim != 1:
        raise ValueError("scores must be rank one")
    positives = tuple(int(index) for index in positive_indices)
    negatives = tuple(int(index) for index in negative_indices)
    if not positives:
        raise ValueError("positive_indices must not be empty")
    if not negatives:
        return scores.sum() * 0.0
    size = int(scores.numel())
    if any(index < 0 or index >= size for index in (*positives, *negatives)):
        raise ValueError("loss index is outside the score vector")
    if set(positives) & set(negatives):
        raise ValueError("positive and negative indices must be disjoint")
    positive_tensor = scores[list(positives)]
    negative_tensor = scores[list(negatives)]
    cutoff_count = min(int(k), int(negative_tensor.numel()))
    negative_cutoff = torch.topk(
        negative_tensor,
        k=cutoff_count,
        largest=True,
        sorted=True,
    ).values[-1]
    positive_score = _smoothmax(torch, positive_tensor, tau=tau)
    return torch.nn.functional.softplus(
        float(margin) + negative_cutoff - positive_score
    )


def _evidence_loss(
    torch: Any,
    scores: Any,
    *,
    any_positive_indices: Sequence[int],
    positives_by_source: Mapping[str, Sequence[int]],
    exact_positive_indices: Sequence[int],
) -> tuple[Any, dict[str, float]]:
    """Fixed evidence-aligned listwise/top-k objective for one question."""

    any_positive = tuple(int(value) for value in any_positive_indices)
    if not any_positive:
        raise ValueError("evidence loss requires a reachable positive candidate")
    size = int(scores.numel())
    any_set = set(any_positive)
    ordinary_negatives = tuple(index for index in range(size) if index not in any_set)
    any_at_8 = _hit_at_k_loss(
        torch,
        scores,
        any_positive,
        ordinary_negatives,
        k=TOP_K,
        tau=SMOOTHMAX_TAU,
        margin=RANK_MARGIN,
    )
    any_at_1 = _hit_at_k_loss(
        torch,
        scores,
        any_positive,
        ordinary_negatives,
        k=1,
        tau=SMOOTHMAX_TAU,
        margin=RANK_MARGIN,
    )
    source_terms: list[Any] = []
    for source_id in sorted(positives_by_source):
        indices = tuple(int(value) for value in positives_by_source[source_id])
        if not indices:
            # A required source outside the candidate frontier cannot supply a
            # gradient. The ceiling and evaluation retain the miss.
            continue
        source_index_set = set(indices)
        source_competitors = tuple(
            index for index in range(size) if index not in source_index_set
        )
        source_terms.append(
            _hit_at_k_loss(
                torch,
                scores,
                indices,
                source_competitors,
                k=TOP_K,
                tau=SMOOTHMAX_TAU,
                margin=RANK_MARGIN,
            )
        )
    source_balanced = (
        torch.stack(source_terms).mean()
        if source_terms
        else scores.sum() * 0.0
    )
    exact = tuple(int(value) for value in exact_positive_indices)
    if exact:
        exact_set = set(exact)
        exact_negatives = tuple(index for index in range(size) if index not in exact_set)
        exact_at_8 = _hit_at_k_loss(
            torch,
            scores,
            exact,
            exact_negatives,
            k=TOP_K,
            tau=SMOOTHMAX_TAU,
            margin=RANK_MARGIN,
        )
        exact_present = 1.0
    else:
        exact_at_8 = scores.sum() * 0.0
        exact_present = 0.0
    total = (
        LOSS_WEIGHTS["any_at_8"] * any_at_8
        + LOSS_WEIGHTS["source_balanced_at_8"] * source_balanced
        + LOSS_WEIGHTS["exact_at_8"] * exact_at_8
        + LOSS_WEIGHTS["any_at_1"] * any_at_1
    )
    components = {
        "total": float(total.detach().cpu()),
        "any_at_8": float(any_at_8.detach().cpu()),
        "source_balanced_at_8": float(source_balanced.detach().cpu()),
        "exact_at_8": float(exact_at_8.detach().cpu()),
        "any_at_1": float(any_at_1.detach().cpu()),
        "exact_present": exact_present,
        "represented_required_sources": float(len(source_terms)),
    }
    if any(not math.isfinite(value) for value in components.values()):
        raise AssayError("training loss became non-finite")
    return total, components


def _ordered_indices(scores: Sequence[float]) -> tuple[int, ...]:
    values = tuple(float(value) for value in scores)
    if any(not math.isfinite(value) for value in values):
        raise AssayError("a scorer returned a non-finite value")
    return tuple(sorted(range(len(values)), key=lambda index: (-values[index], index)))


def _forward_scores(
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    question: PreparedQuestion,
    *,
    training: bool,
) -> Any:
    parts: list[Any] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for start in range(0, len(question.candidate_texts), SCORE_SUB_BATCH):
            texts = list(
                question.candidate_texts[start : start + SCORE_SUB_BATCH]
            )
            encoded = tokenizer(
                [question.query] * len(texts),
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.float().reshape(-1)
            parts.append(logits if training else logits.detach())
    scores = torch.cat(parts)
    if int(scores.numel()) != CANDIDATES or not bool(torch.isfinite(scores).all()):
        raise AssayError("MiniLM returned an invalid lexical-96 score vector")
    return scores


def _score_question(
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    question: PreparedQuestion,
) -> tuple[tuple[float, ...], float]:
    pilot = _pilot()
    pilot._synchronize(torch, device)
    started = time.perf_counter()
    scores = _forward_scores(
        torch, tokenizer, model, device, question, training=False
    )
    values = tuple(float(value) for value in scores.cpu().tolist())
    pilot._synchronize(torch, device)
    return values, (time.perf_counter() - started) * 1000.0


def _state_sha256(model: Any) -> str:
    """Content identity for an ephemeral trained state without saving it."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        header = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        encoded = _canonical_json(header).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _learning_rate_for_step(step: int, total_steps: int) -> float:
    warmup_steps = max(1, math.ceil(total_steps * WARMUP_FRACTION))
    completed = step + 1
    if completed <= warmup_steps:
        scale = completed / warmup_steps
    else:
        scale = max(0.0, (total_steps - completed) / (total_steps - warmup_steps))
    return LEARNING_RATE * scale


def _train_fold(
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    questions: Sequence[PreparedQuestion],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], float, int, int]:
    eligible = [question for question in questions if _label_indices(question)[0]]
    skipped = len(questions) - len(eligible)
    if not eligible:
        raise AssayError("a fold has no reachable training questions")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    steps_per_epoch = math.ceil(len(eligible) / EFFECTIVE_QUESTION_BATCH)
    total_steps = EPOCHS * steps_per_epoch
    step = 0
    epoch_reports: list[dict[str, Any]] = []
    pilot = _pilot()
    pilot._synchronize(torch, device)
    started = time.perf_counter()
    model.train()
    for epoch in range(EPOCHS):
        ordered = list(eligible)
        random.Random(seed + epoch).shuffle(ordered)
        totals: defaultdict[str, float] = defaultdict(float)
        observations: defaultdict[str, int] = defaultdict(int)
        for offset in range(0, len(ordered), EFFECTIVE_QUESTION_BATCH):
            group = ordered[offset : offset + EFFECTIVE_QUESTION_BATCH]
            optimizer.zero_grad(set_to_none=True)
            for question in group:
                scores = _forward_scores(
                    torch,
                    tokenizer,
                    model,
                    device,
                    question,
                    training=True,
                )
                any_positive, by_source, exact = _label_indices(question)
                loss, components = _evidence_loss(
                    torch,
                    scores,
                    any_positive_indices=any_positive,
                    positives_by_source=by_source,
                    exact_positive_indices=exact,
                )
                (loss / len(group)).backward()
                for name, value in components.items():
                    totals[name] += value
                    observations[name] += 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP
            )
            if not math.isfinite(float(gradient_norm.detach().cpu())):
                raise AssayError("gradient norm became non-finite")
            learning_rate = _learning_rate_for_step(step, total_steps)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            optimizer.step()
            step += 1
        epoch_reports.append(
            {
                "epoch": epoch + 1,
                "mean_components_hex": {
                    name: _float_hex(totals[name] / observations[name])
                    for name in sorted(totals)
                },
                "optimizer_steps_completed": step,
                "last_learning_rate_hex": _float_hex(
                    _learning_rate_for_step(step - 1, total_steps)
                ),
            }
        )
    pilot._synchronize(torch, device)
    elapsed = time.perf_counter() - started
    model.eval()
    if step != total_steps:
        raise AssayError("optimizer step accounting differs from the fixed schedule")
    return epoch_reports, elapsed, len(eligible), skipped


def _row_metrics(
    question: PreparedQuestion,
    order: Sequence[int],
) -> dict[str, Any]:
    ordered = tuple(int(value) for value in order)
    if len(ordered) != CANDIDATES or set(ordered) != set(range(CANDIDATES)):
        raise AssayError("ranking order is not a permutation of the frontier")
    required = set(question.required_source_ids)
    exact = set(question.exact_memory_ids)

    def selected(k: int) -> tuple[int, ...]:
        return ordered[: min(k, len(ordered))]

    def any_source(k: int) -> bool:
        return any(question.candidate_source_ids[index] in required for index in selected(k))

    def exact_turn(k: int) -> bool:
        return any(question.candidate_ids[index] in exact for index in selected(k))

    selected_sources = {
        question.candidate_source_ids[index] for index in selected(TOP_K)
    }
    source_hits = required & selected_sources
    reciprocal_rank = 0.0
    for rank, index in enumerate(ordered, start=1):
        if question.candidate_source_ids[index] in required:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "any_source_at_1": any_source(1),
        "any_source_at_4": any_source(4),
        "any_source_at_8": any_source(8),
        "exact_turn_at_1": exact_turn(1),
        "exact_turn_at_4": exact_turn(4),
        "exact_turn_at_8": exact_turn(8),
        "source_recall_at_8_hex": _float_hex(
            len(source_hits) / len(required)
        ),
        "all_sources_at_8": source_hits == required,
        "reciprocal_rank_hex": _float_hex(reciprocal_rank),
        "selected_memory_ids_at_8": [
            question.candidate_ids[index] for index in selected(TOP_K)
        ],
    }


def _candidate_ceiling(
    question: PreparedQuestion,
) -> dict[str, Any]:
    required = set(question.required_source_ids)
    observed_sources = set(question.candidate_source_ids)
    exact = set(question.exact_memory_ids)
    observed_memories = set(question.candidate_ids)
    hits = required & observed_sources
    return {
        "any_source": bool(hits),
        "exact_turn": bool(exact & observed_memories),
        "exact_turn_labeled": bool(exact),
        "source_recall_hex": _float_hex(len(hits) / len(required)),
        "all_sources": hits == required,
    }


def _rate(count: int, denominator: int) -> dict[str, Any]:
    return {
        "count": int(count),
        "denominator": int(denominator),
        "rate_hex": _float_hex(count / denominator if denominator else 0.0),
    }


def _aggregate_metrics(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
) -> dict[str, Any]:
    if not rows:
        raise AssayError("cannot aggregate an empty question population")
    metrics = [
        _require_mapping(
            _require_mapping(row.get("arms"), "row arms").get(arm),
            f"row arm {arm}",
        )
        for row in rows
    ]
    count = len(metrics)
    exact_labeled = sum(
        bool(_require_mapping(row.get("candidate_ceiling"), "ceiling")["exact_turn_labeled"])
        for row in rows
    )
    exact_eligible = sum(
        bool(_require_mapping(row.get("candidate_ceiling"), "ceiling")["exact_turn"])
        for row in rows
    )
    exact_hits_at_8 = sum(bool(row["exact_turn_at_8"]) for row in metrics)
    source_recalls = [
        float.fromhex(str(row["source_recall_at_8_hex"])) for row in metrics
    ]
    reciprocal_ranks = [
        float.fromhex(str(row["reciprocal_rank_hex"])) for row in metrics
    ]
    return {
        "questions": count,
        "any_source": {
            f"hit_at_{k}": _rate(
                sum(bool(row[f"any_source_at_{k}"]) for row in metrics), count
            )
            for k in (1, 4, 8)
        },
        "exact_turn": {
            "labeled_questions": exact_labeled,
            "frontier_eligible_questions": exact_eligible,
            **{
                f"hit_at_{k}_over_all": _rate(
                    sum(bool(row[f"exact_turn_at_{k}"]) for row in metrics), count
                )
                for k in (1, 4, 8)
            },
            "hit_at_8_conditional_on_frontier_hex": _float_hex(
                exact_hits_at_8 / exact_eligible if exact_eligible else 0.0
            ),
        },
        "mean_source_recall_at_8_hex": _float_hex(statistics.fmean(source_recalls)),
        "all_sources_at_8": _rate(
            sum(bool(row["all_sources_at_8"]) for row in metrics), count
        ),
        "mean_reciprocal_rank_hex": _float_hex(statistics.fmean(reciprocal_ranks)),
    }


def _aggregate_ceilings(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise AssayError("cannot aggregate empty candidate ceilings")
    ceilings = [
        _require_mapping(row.get("candidate_ceiling"), "candidate ceiling")
        for row in rows
    ]
    count = len(ceilings)
    exact_labeled = sum(bool(row["exact_turn_labeled"]) for row in ceilings)
    exact_eligible = sum(bool(row["exact_turn"]) for row in ceilings)
    source_recalls = [float.fromhex(str(row["source_recall_hex"])) for row in ceilings]
    return {
        "questions": count,
        "any_source": _rate(sum(bool(row["any_source"]) for row in ceilings), count),
        "exact_turn": {
            "labeled_questions": exact_labeled,
            "frontier_eligible": _rate(exact_eligible, exact_labeled),
            "rate_over_all": _rate(exact_eligible, count),
        },
        "mean_source_recall_hex": _float_hex(statistics.fmean(source_recalls)),
        "all_sources": _rate(sum(bool(row["all_sources"]) for row in ceilings), count),
    }


def _comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    candidate: str,
    metric: str = "any_source_at_8",
) -> dict[str, Any]:
    rescued: list[str] = []
    regressed: list[str] = []
    unchanged_success = 0
    unchanged_miss = 0
    for row in rows:
        arms = _require_mapping(row.get("arms"), "row arms")
        before = bool(_require_mapping(arms.get(baseline), baseline)[metric])
        after = bool(_require_mapping(arms.get(candidate), candidate)[metric])
        if after and not before:
            rescued.append(str(row["question_id"]))
        elif before and not after:
            regressed.append(str(row["question_id"]))
        elif before:
            unchanged_success += 1
        else:
            unchanged_miss += 1
    return {
        "baseline": baseline,
        "candidate": candidate,
        "rescued": len(rescued),
        "regressed": len(regressed),
        "net_marginal": len(rescued) - len(regressed),
        "unchanged_success": unchanged_success,
        "unchanged_miss": unchanged_miss,
        "rescued_question_ids": rescued,
        "regressed_question_ids": regressed,
    }


def _latency_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise AssayError("latency population is empty")
    return {
        "samples": len(values),
        "mean_ms_hex": _float_hex(statistics.fmean(values)),
        "p50_ms_hex": _float_hex(_percentile(values, 0.50) or 0.0),
        "p95_ms_hex": _float_hex(_percentile(values, 0.95) or 0.0),
        "max_ms_hex": _float_hex(max(values)),
    }


def _base_inventory(model_root: Path) -> tuple[list[dict[str, Any]], str]:
    if model_root.is_symlink() or not model_root.is_dir():
        raise AssayError("MiniLM directory must be a regular non-symlink directory")
    inventory: list[dict[str, Any]] = []
    for path in sorted(model_root.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            raise AssayError("MiniLM directory contains a symlink")
        if not path.is_file():
            continue
        inventory.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not inventory:
        raise AssayError("MiniLM directory is empty")
    return inventory, _identity(inventory)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    plane, plane_sha256, records, dataset_sha256 = _load_plane_and_dataset(
        args.plane.resolve(), args.dataset.resolve()
    )
    coordinates = _reconstruct_gold_coordinates(plane, records)
    folds = _fold_assignments(plane)
    frontiers = _lexical_frontiers(plane)
    questions = _prepare_questions(plane, coordinates, folds, frontiers)
    model_root = args.minilm_model_dir.resolve()
    inventory, inventory_sha256 = _base_inventory(model_root)

    contract = {
        "format": "memory-condense-minilm-evidence-accuracy-contract-v1",
        "development_population": 200,
        "candidate_policy": {
            "operator": "query_only_bm25_same_as_qwen_minilm_pilot",
            "scope": "original_plane_partition_global_pool",
            "candidate_count": CANDIDATES,
            "lexical_count": CANDIDATES,
            "distractor_count": 0,
            "rerank_top_k": TOP_K,
            "query_tokens": QUERY_TOKENS,
            "candidate_tokens": CANDIDATE_TOKENS,
            "max_length": MAX_LENGTH,
            "score_sub_batch": SCORE_SUB_BATCH,
        },
        "fold_policy": {
            "algorithm": "stratified-largest-remainder-v1",
            "stratum": "question_type",
            "salt": FOLD_SALT,
            "fold_counts": FOLD_COUNTS,
            "question_labels_out_of_fold": True,
            "document_pool_shared_within_original_partition": True,
        },
        "label_taxonomy": {
            "grade_2_exact_turn": 2,
            "grade_1_required_answer_session_other_turn": 1,
            "grade_0_other": 0,
            "usage": (
                "overlapping any-source, source-balanced, and exact-turn objectives; "
                "not a scalar graded-ranking loss"
            ),
        },
        "loss": {
            "operator": "smoothmax_kth_negative_topk_hinge_v1",
            "smoothmax_tau_hex": _float_hex(SMOOTHMAX_TAU),
            "margin_hex": _float_hex(RANK_MARGIN),
            "weights_hex": {
                name: _float_hex(value) for name, value in LOSS_WEIGHTS.items()
            },
            "source_competitors": "all_candidates_outside_that_required_source",
            "any_competitors": "grade_zero_candidates",
            "unreachable_training_questions": "excluded_no_gradient_retained_in_eval",
        },
        "optimization": {
            "optimizer": "AdamW",
            "epochs": EPOCHS,
            "learning_rate_hex": _float_hex(LEARNING_RATE),
            "weight_decay_hex": _float_hex(WEIGHT_DECAY),
            "effective_question_batch": EFFECTIVE_QUESTION_BATCH,
            "gradient_accumulation": (
                "one_complete_96_candidate_list_at_a_time_then_step_after_four"
            ),
            "warmup_fraction_hex": _float_hex(WARMUP_FRACTION),
            "schedule": "linear_warmup_then_linear_decay_to_zero",
            "gradient_clip_hex": _float_hex(GRADIENT_CLIP),
            "base_seed": BASE_SEED,
            "fold_seed": "base_seed_plus_zero_based_fold_index",
            "fresh_verified_base_per_fold": True,
        },
        "latency": {
            "warmup_questions_per_fold": LATENCY_WARMUP_QUESTIONS,
            "scope": "tokenization_plus_96_candidate_student_forward",
            "model_load_excluded": True,
            "proxy_render_excluded_precomputed": True,
        },
    }
    contract_sha256 = _identity(contract)
    fold_assignment = {
        fold: sorted(question_id for question_id, value in folds.items() if value == fold)
        for fold in FOLD_COUNTS
    }
    fold_assignment_sha256 = _identity(fold_assignment)
    frontier_sha256 = _identity(
        [
            {"question_id": question_id, "candidate_memory_ids": list(frontiers[question_id])}
            for question_id in sorted(frontiers)
        ]
    )
    gold_coordinate_sha256 = _identity(
        [
            {
                "question_id": question_id,
                "required_source_ids": list(coordinates[question_id].required_source_ids),
                "exact_memory_ids": list(coordinates[question_id].exact_memory_ids),
            }
            for question_id in sorted(coordinates)
        ]
    )

    base_scores: dict[str, tuple[float, ...]] = {}
    trained_scores: dict[str, tuple[float, ...]] = {}
    base_latency_ms: list[float] = []
    trained_latency_ms: list[float] = []
    fold_reports: list[dict[str, Any]] = []
    expected_base_sha256: str | None = None

    for fold_index, fold in enumerate(FOLD_COUNTS):
        held_out = [question for question in questions if question.fold == fold]
        training = [question for question in questions if question.fold != fold]
        if len(held_out) != 40 or len(training) != 160:
            raise AssayError("fold population differs from the frozen 160/40 contract")
        seed = BASE_SEED + fold_index
        load_started = time.perf_counter()
        pilot = _pilot()
        torch, tokenizer, model, device, checkpoint_sha256 = pilot._load_student(
            model_root,
            device_name=str(args.device),
            verify_base=True,
        )
        load_seconds = time.perf_counter() - load_started
        if checkpoint_sha256 is None:
            raise AssayError("verified MiniLM load omitted its checkpoint identity")
        if expected_base_sha256 is None:
            expected_base_sha256 = checkpoint_sha256
        elif checkpoint_sha256 != expected_base_sha256:
            raise AssayError("fresh fold loads disagree on the base checkpoint")

        model.eval()
        for question in held_out[:LATENCY_WARMUP_QUESTIONS]:
            _forward_scores(
                torch, tokenizer, model, device, question, training=False
            )
        for question in held_out:
            values, elapsed_ms = _score_question(
                torch, tokenizer, model, device, question
            )
            base_scores[question.question_id] = values
            base_latency_ms.append(elapsed_ms)

        epoch_reports, training_seconds, eligible_count, skipped_count = _train_fold(
            torch,
            tokenizer,
            model,
            device,
            training,
            seed=seed,
        )
        for question in held_out[:LATENCY_WARMUP_QUESTIONS]:
            _forward_scores(
                torch, tokenizer, model, device, question, training=False
            )
        for question in held_out:
            values, elapsed_ms = _score_question(
                torch, tokenizer, model, device, question
            )
            trained_scores[question.question_id] = values
            trained_latency_ms.append(elapsed_ms)
        trained_state_sha256 = _state_sha256(model)
        fold_reports.append(
            {
                "fold": fold,
                "fold_index": fold_index,
                "seed": seed,
                "train_question_ids_sha256": _identity(
                    sorted(question.question_id for question in training)
                ),
                "heldout_question_ids_sha256": _identity(
                    sorted(question.question_id for question in held_out)
                ),
                "train_questions": len(training),
                "train_reachable_questions": eligible_count,
                "train_frontier_miss_questions": skipped_count,
                "heldout_questions": len(held_out),
                "heldout_original_partition_counts": dict(
                    sorted(Counter(question.original_partition for question in held_out).items())
                ),
                "heldout_question_type_counts": dict(
                    sorted(Counter(question.question_type for question in held_out).items())
                ),
                "base_checkpoint_sha256": checkpoint_sha256,
                "trained_state_sha256": trained_state_sha256,
                "ephemeral_model_not_saved": True,
                "load_seconds_hex": _float_hex(load_seconds),
                "training_seconds_hex": _float_hex(training_seconds),
                "epoch_losses": epoch_reports,
            }
        )
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(base_scores) != 200 or len(trained_scores) != 200:
        raise AssayError("out-of-fold scoring did not cover every development question")
    if expected_base_sha256 is None:
        raise AssayError("no MiniLM fold was executed")

    rows: list[dict[str, Any]] = []
    for question in questions:
        lexical_order = tuple(range(CANDIDATES))
        base_order = _ordered_indices(base_scores[question.question_id])
        trained_order = _ordered_indices(trained_scores[question.question_id])
        rows.append(
            {
                "question_id": question.question_id,
                "question_type": question.question_type,
                "original_partition": question.original_partition,
                "fold": question.fold,
                "candidate_memory_ids_sha256": _identity(list(question.candidate_ids)),
                "candidate_ceiling": _candidate_ceiling(question),
                "arms": {
                    "lexical": _row_metrics(question, lexical_order),
                    "base_minilm": _row_metrics(question, base_order),
                    "oof_trained_minilm": _row_metrics(question, trained_order),
                },
            }
        )
    rows.sort(key=lambda row: str(row["question_id"]))

    for report in fold_reports:
        fold_rows = [row for row in rows if row["fold"] == report["fold"]]
        report["candidate_ceiling"] = _aggregate_ceilings(fold_rows)
        report["metrics"] = {
            arm: _aggregate_metrics(fold_rows, arm)
            for arm in ("lexical", "base_minilm", "oof_trained_minilm")
        }
    by_type = {
        question_type: {
            "candidate_ceiling": _aggregate_ceilings(
                [row for row in rows if row["question_type"] == question_type]
            ),
            "metrics": {
                arm: _aggregate_metrics(
                    [row for row in rows if row["question_type"] == question_type],
                    arm,
                )
                for arm in ("lexical", "base_minilm", "oof_trained_minilm")
            },
        }
        for question_type in sorted({str(row["question_type"]) for row in rows})
    }
    aggregate = {
        "candidate_ceiling": _aggregate_ceilings(rows),
        "metrics": {
            arm: _aggregate_metrics(rows, arm)
            for arm in ("lexical", "base_minilm", "oof_trained_minilm")
        },
    }
    lexical_comparison = _comparison(
        rows, baseline="lexical", candidate="oof_trained_minilm"
    )
    lexical_exact_comparison = _comparison(
        rows,
        baseline="lexical",
        candidate="oof_trained_minilm",
        metric="exact_turn_at_8",
    )
    base_comparison = _comparison(
        rows, baseline="base_minilm", candidate="oof_trained_minilm"
    )
    trained_latency = _latency_summary(trained_latency_ms)
    base_latency = _latency_summary(base_latency_ms)

    trained_any = int(
        aggregate["metrics"]["oof_trained_minilm"]["any_source"]["hit_at_8"][
            "count"
        ]
    )
    trained_source_recall = float.fromhex(
        aggregate["metrics"]["oof_trained_minilm"]["mean_source_recall_at_8_hex"]
    )
    trained_exact_conditional = float.fromhex(
        aggregate["metrics"]["oof_trained_minilm"]["exact_turn"][
            "hit_at_8_conditional_on_frontier_hex"
        ]
    )
    trained_exact_count = int(
        aggregate["metrics"]["oof_trained_minilm"]["exact_turn"][
            "hit_at_8_over_all"
        ]["count"]
    )
    trained_p95_ms = float.fromhex(trained_latency["p95_ms_hex"])
    fold_hit_counts = {
        report["fold"]: int(
            report["metrics"]["oof_trained_minilm"]["any_source"]["hit_at_8"][
                "count"
            ]
        )
        for report in fold_reports
    }
    gate_checks = {
        "any_source_hit_at_8_at_least_190_of_200": trained_any >= 190,
        "mean_required_source_recall_at_8_at_least_0_95": (
            trained_source_recall >= 0.95
        ),
        "exact_turn_hit_at_8_conditional_on_frontier_at_least_0_95": (
            trained_exact_conditional >= 0.95
        ),
        "regressions_against_lexical_at_most_2": (
            int(lexical_comparison["regressed"]) <= 2
        ),
        "exact_turn_regressions_against_lexical_at_most_2": (
            int(lexical_exact_comparison["regressed"]) <= 2
        ),
        "every_fold_any_source_hit_at_8_at_least_37_of_40": all(
            value >= 37 for value in fold_hit_counts.values()
        ),
        "warm_p95_below_50_ms": trained_p95_ms < 50.0,
    }

    from memory_condense.search.selectors.cross_encoder_selector import (
        MS_MARCO_MODEL_ID,
        MS_MARCO_MODEL_REVISION,
    )

    payload: dict[str, Any] = {
        "format": FORMAT,
        "status": STATUS,
        "promotion_eligible": False,
        "development_truth_used": True,
        "limitations": [
            "all 200 questions are analysis-used development and no row is a fresh holdout",
            "five-fold OOF prevents direct question-label fitting but is not validation",
            "all development labels are process-visible; fold isolation is enforced by code paths",
            "candidate documents are shared across question folds inside original partitions",
            "source/session and exact annotated-turn metrics are distinct; neither proves answer correctness",
            "the source is an oracle projection, not a million-token ingest",
            "proxy turns are not production Transcript cards or episode representatives",
            "no responder or semantic judge was run, so this is retrieval accuracy only",
            "fold checkpoints are ephemeral and this report cannot deploy a model",
        ],
        "inputs": {
            "plane": str(args.plane.resolve()),
            "plane_sha256": plane_sha256,
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": dataset_sha256,
            "oracle_records_loaded": len(records),
            "development_records_dereferenced": 200,
            "answer_field_not_accessed_after_decode": True,
            "answer_text_process_visible_in_source_json": True,
            "label_fields_read": ["answer_session_ids", "has_answer"],
        },
        "contract": contract,
        "contract_sha256": contract_sha256,
        "identities": {
            "assay_implementation_sha256": _file_sha256(Path(__file__).resolve()),
            "fold_assignment_sha256": fold_assignment_sha256,
            "lexical_frontiers_sha256": frontier_sha256,
            "gold_coordinates_sha256": gold_coordinate_sha256,
            "base_oof_scores_sha256": _identity(
                {
                    question_id: [float(value).hex() for value in values]
                    for question_id, values in sorted(base_scores.items())
                }
            ),
            "trained_oof_scores_sha256": _identity(
                {
                    question_id: [float(value).hex() for value in values]
                    for question_id, values in sorted(trained_scores.items())
                }
            ),
            "base_model": {
                "model_id": MS_MARCO_MODEL_ID,
                "model_revision": MS_MARCO_MODEL_REVISION,
                "weights_sha256": expected_base_sha256,
                "directory_inventory_sha256": inventory_sha256,
                "directory_inventory": inventory,
            },
            "trained_fold_state_sha256": {
                report["fold"]: report["trained_state_sha256"]
                for report in fold_reports
            },
        },
        "fold_assignment": {
            "counts": FOLD_COUNTS,
            "question_ids_sha256_by_fold": {
                fold: _identity(question_ids)
                for fold, question_ids in fold_assignment.items()
            },
            "explicit_fold_member_lists_not_emitted": True,
            "row_fold_membership_emitted": True,
        },
        "aggregate": aggregate,
        "comparisons": {
            "lexical_to_oof_trained": lexical_comparison,
            "lexical_exact_turn_to_oof_trained": lexical_exact_comparison,
            "base_to_oof_trained": base_comparison,
        },
        "latency": {
            "base_minilm": base_latency,
            "oof_trained_minilm": trained_latency,
        },
        "advance_gate": {
            "purpose": "development-only decision to justify a later sealed validation shadow",
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
            "promotion_authorized_even_if_passed": False,
            "fold_hit_at_8_counts": fold_hit_counts,
            "exact_turn_hit_at_8_count": trained_exact_count,
            "exact_turn_frontier_eligible_questions": int(
                aggregate["metrics"]["oof_trained_minilm"]["exact_turn"][
                    "frontier_eligible_questions"
                ]
            ),
        },
        "folds": fold_reports,
        "by_question_type": by_type,
        "rows": rows,
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not str(args.device).strip():
            raise AssayError("device must be non-empty")
        output = _runtime_output(args.output)
        payload = _run(args)
        digest, sidecar = _publish(output, payload)
        print(
            _canonical_json(
                {
                    "output": str(output),
                    "sha256": digest,
                    "sha256_file": str(sidecar),
                    "status": payload["status"],
                    "promotion_eligible": False,
                    "advance_gate_passed": payload["advance_gate"]["passed"],
                }
            )
        )
        return 0
    except (AssayError, FileExistsError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
