"""Run the sealed post-benchmark INT8 Qwen landmark experiment.

The runner publishes resumable, source-bound landmark batches and reuses the
existing development and holdout questions, BGE addresses, exact support
labels, and production packets. Qwen receives summaries and questions only.
Raw turns are opened by the existing validators solely to score exact support.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.associations.qwen_int8_landmarks import (
    LANDMARK_FORMAT,
    Int8KVLandmarks,
    compile_int8_kv_landmarks,
    score_symmetric_int8_kv_landmark_batch,
)
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.modeling.qwen_prefix import expected_prefix_checkpoint_sha256
from memory_condense.persistence import native_spine_store
from memory_condense.runtime.artifacts import read, read_sealed_json, save
from memory_condense.runtime.config import RuntimeAssets
from memory_condense.runtime.local import LocalRuntime
from memory_condense.search.summary_semantic_index import (
    compatible_summary_embedding,
    summary_embedding_identity,
)
from tools.assay_attention_feature_retrieval import (
    ORIGINAL,
    REFERENCE,
    _score_arm,
)
from tools.assay_qwen_int8_landmarks import _candidate_prompt, _project_selected_kv, _query_prompt
from tools.characterize_native_experiments import cases
from tools.evaluate_dialogue_neighborhood import digest, source_turns
from tools.validate_attention_feature_holdout import HOLDOUT, _measure, _supports


R27 = Path("eval_results/longmemeval-s-native-release-sol-20261014-r27")
DEVELOPMENT_RANKINGS = Path(
    "eval_results/attention-feature-retrieval-20261009-r1/rankings.json"
)
FORMAT = "memory-condense-qwen-int8-landmark-post-benchmark-v1"
LAYER = 5
LANDMARKS = 8
KV_HEADS = 8
HEAD_DIM = 128
_PRE_PUBLICATION_FIX_SCRIPT_SHA256 = (
    "2552e9d37ab93e72731bf69845632f412746ac019a2c0d0a14c6b04692f7da43"
)


@dataclass(frozen=True, slots=True)
class Population:
    name: str
    directory: Path
    turns: tuple[Any, ...]
    snapshot: Any
    sections: tuple[Any, ...]
    questions: tuple[dict[str, Any], ...]
    question_ids: tuple[str, ...]
    question_texts: tuple[str, ...]
    reference: Any
    source_identity: dict[str, Any]


def _file_sha256(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest_value.update(block)
    return digest_value.hexdigest()


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("latency sample must not be empty")

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "samples": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "maximum_ms": ordered[-1],
    }


def _r27_receipt() -> dict[str, Any]:
    report_path = R27 / "report.json"
    report = read(report_path)
    if report != {
        "answered": 500,
        "complete": True,
        "correct": 299,
        "format": "longmemeval-s-native-release-sol-report-v1",
        "judge_accuracy": 0.598,
        "judged": 500,
        "mean_answer_s": report["mean_answer_s"],
        "mean_ingest_s": report["mean_ingest_s"],
        "mean_recall_s": report["mean_recall_s"],
    }:
        raise ValueError("r27 is not the completed 500-question benchmark")
    state_path = R27 / "background-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("stage") != "complete" or state.get("status") != "complete":
        raise ValueError("r27 background state is not terminal")
    return {
        "path": str(R27),
        "report_sha256": _file_sha256(report_path),
        "background_state_sha256": _file_sha256(state_path),
        "score": report["judge_accuracy"],
        "answered": report["answered"],
        "judged": report["judged"],
    }


def _load_population(name: str) -> Population:
    if name == "development":
        source = read(ORIGINAL / "source.json")
        directory = Path(source["directory"])
        if any(digest(directory / filename) != expected for filename, expected in source["files"].items()):
            raise ValueError("development source archive changed")
        turns = tuple(source_turns(directory))
        snapshot = native_spine_store.load(directory / "native-spine.sqlite", turns=turns)
        questions = tuple(cases())
        return Population(
            name=name,
            directory=directory,
            turns=turns,
            snapshot=snapshot,
            sections=tuple(snapshot.semantic.sections),
            questions=questions,
            question_ids=tuple(question["id"] for question in questions),
            question_texts=tuple(question["question"] for question in questions),
            reference=read(REFERENCE),
            source_identity={"source_receipt": source},
        )
    if name == "holdout":
        directory = HOLDOUT / "application"
        turns = tuple(source_turns(directory))
        snapshot = native_spine_store.load(directory / "native-spine.sqlite", turns=turns)
        questions_artifact = read(HOLDOUT / "questions" / "questions.json")
        references = read(HOLDOUT / "questions" / "references.json")
        questions = tuple(questions_artifact["questions"])
        return Population(
            name=name,
            directory=directory,
            turns=turns,
            snapshot=snapshot,
            sections=tuple(snapshot.semantic.sections),
            questions=questions,
            question_ids=tuple(question["question_id"] for question in questions),
            question_texts=tuple(question["question"] for question in questions),
            reference=_supports(turns, references),
            source_identity={
                "ingest_complete_sha256": _file_sha256(HOLDOUT / "ingest-complete.json"),
                "questions_sha256": _file_sha256(HOLDOUT / "questions" / "questions.json"),
                "references_sha256": _file_sha256(HOLDOUT / "questions" / "references.json"),
            },
        )
    raise ValueError("population must be development or holdout")


def _section_item(section: Any) -> dict[str, Any]:
    return {
        "section_id": section.section_id,
        "section_receipt_sha256": section.receipt_sha256,
        "source_id": section.source_id,
        "summary_sha256": quote_sha256(section.summary),
        "span_receipt_sha256s": [span.receipt_sha256 for span in section.spans],
    }


def _plan(population: Population, *, batch_size: int) -> dict[str, Any]:
    script = Path(__file__)
    module = Path("src/memory_condense/associations/qwen_int8_landmarks.py")
    return {
        "format": FORMAT,
        "population": population.name,
        "archive": str(population.directory),
        "source_identity": population.source_identity,
        "sections": len(population.sections),
        "section_identity_sha256": identity_sha256(
            [_section_item(section) for section in population.sections]
        ),
        "questions": len(population.questions),
        "question_identity_sha256": identity_sha256(
            list(zip(population.question_ids, population.question_texts, strict=True))
        ),
        "semantic_index_sha256": population.snapshot.semantic.receipt_sha256,
        "qwen_checkpoint_sha256": expected_prefix_checkpoint_sha256(6),
        "qwen_layers": 6,
        "layer": LAYER,
        "landmarks": LANDMARKS,
        "landmark_format": LANDMARK_FORMAT,
        "batch_size": batch_size,
        "raw_inputs_to_qwen": False,
        "linker_origin_commit": "0bc2d89",
        "implementation_sha256": {
            str(script): _file_sha256(script),
            str(module): _file_sha256(module),
        },
        "r27": _r27_receipt(),
    }


def _load_or_publish_plan(
    root: Path,
    population: Population,
    *,
    batch_size: int,
):
    """Reuse the one pre-fix plan only across the memmap publication repair."""

    proposed = _plan(population, batch_size=batch_size)
    path = root / "plan.json"
    if not path.exists() and not path.with_name(path.name + ".sha256").exists():
        return save(path, proposed)

    existing = read_sealed_json(path)
    if existing.payload == proposed:
        return existing

    script = str(Path(__file__))
    compatible = json.loads(json.dumps(proposed))
    compatible["implementation_sha256"][script] = _PRE_PUBLICATION_FIX_SCRIPT_SHA256
    if existing.payload != compatible:
        raise ValueError("existing experiment plan is not compatible with this runner")
    save(
        root / "publication-repair.json",
        {
            "format": "memory-condense-qwen-int8-landmark-publication-repair-v1",
            "plan_sha256": existing.sha256,
            "previous_script_sha256": _PRE_PUBLICATION_FIX_SCRIPT_SHA256,
            "repaired_script_sha256": proposed["implementation_sha256"][script],
            "scope": "Close every NumPy memmap before atomic replacement on Windows.",
            "compiled_landmarks_changed": False,
        },
    )
    return existing


def _atomic_npz(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _file_sha256(path)


def _batch_paths(root: Path, start: int, stop: int) -> tuple[Path, Path]:
    stem = f"{start:06d}-{stop - 1:06d}"
    return root / "batches" / f"{stem}.npz", root / "batches" / f"{stem}.json"


def _validate_batch(
    data_path: Path,
    receipt_path: Path,
    *,
    plan_sha256: str,
    items: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    receipt = read(receipt_path)
    if (
        receipt.get("format") != "memory-condense-qwen-int8-landmark-batch-v1"
        or receipt.get("plan_sha256") != plan_sha256
        or receipt.get("items") != list(items)
        or receipt.get("artifact_sha256") != _file_sha256(data_path)
    ):
        raise ValueError(f"sealed landmark batch changed: {receipt_path}")
    arrays = np.load(data_path, allow_pickle=False)
    count = len(items)
    expected = {
        "keys": ((count, KV_HEADS, LANDMARKS, HEAD_DIM), np.dtype(np.int8)),
        "values": ((count, KV_HEADS, LANDMARKS, HEAD_DIM), np.dtype(np.int8)),
        "key_scales": ((count, KV_HEADS), np.dtype(np.float16)),
        "value_scales": ((count, KV_HEADS), np.dtype(np.float16)),
        "token_counts": ((count, LANDMARKS), np.dtype(np.uint16)),
        "source_tokens": ((count,), np.dtype(np.uint16)),
    }
    for name, (shape, dtype) in expected.items():
        if name not in arrays or arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"invalid {name} in sealed landmark batch")
    return receipt


def _compile_batches(
    root: Path,
    population: Population,
    *,
    plan_sha256: str,
    encoder: Any,
    batch_size: int,
) -> dict[str, Any]:
    reused = compiled = 0
    elapsed = 0.0
    for start in range(0, len(population.sections), batch_size):
        stop = min(start + batch_size, len(population.sections))
        data_path, receipt_path = _batch_paths(root, start, stop)
        items = [_section_item(section) for section in population.sections[start:stop]]
        if data_path.is_file() and receipt_path.is_file() and receipt_path.with_name(receipt_path.name + ".sha256").is_file():
            _validate_batch(
                data_path,
                receipt_path,
                plan_sha256=plan_sha256,
                items=items,
            )
            reused += len(items)
            continue

        prompts_and_spans = [
            _candidate_prompt(section.summary)
            for section in population.sections[start:stop]
        ]
        started = time.perf_counter()
        projected = _project_selected_kv(
            encoder,
            [row[0] for row in prompts_and_spans],
            [row[1] for row in prompts_and_spans],
            layer=LAYER,
        )
        codes: list[Int8KVLandmarks] = []
        for keys, values in projected:
            code, _, _ = compile_int8_kv_landmarks(keys, values, landmarks=LANDMARKS)
            codes.append(code)
        batch_elapsed = time.perf_counter() - started
        elapsed += batch_elapsed
        artifact_sha256 = _atomic_npz(
            data_path,
            keys=np.stack([code.keys for code in codes]),
            values=np.stack([code.values for code in codes]),
            key_scales=np.stack([code.key_scales for code in codes]),
            value_scales=np.stack([code.value_scales for code in codes]),
            token_counts=np.stack([code.token_counts for code in codes]),
            source_tokens=np.asarray([code.source_tokens for code in codes], dtype=np.uint16),
        )
        save(
            receipt_path,
            {
                "format": "memory-condense-qwen-int8-landmark-batch-v1",
                "plan_sha256": plan_sha256,
                "start": start,
                "stop": stop,
                "items": items,
                "artifact_sha256": artifact_sha256,
                "elapsed_s": batch_elapsed,
            },
        )
        _validate_batch(
            data_path,
            receipt_path,
            plan_sha256=plan_sha256,
            items=items,
        )
        compiled += len(items)
        print(f"compiled {stop}/{len(population.sections)} summaries", flush=True)
    return {"compiled": compiled, "reused": reused, "new_compile_s": elapsed}


def _publish_index(
    root: Path,
    population: Population,
    *,
    plan_sha256: str,
    batch_size: int,
) -> dict[str, Any]:
    index_root = root / "index"
    manifest_path = index_root / "artifact.json"
    names = ("keys", "values", "key_scales", "value_scales", "token_counts", "source_tokens")
    if manifest_path.is_file() and manifest_path.with_name(manifest_path.name + ".sha256").is_file():
        manifest = read(manifest_path)
        if manifest.get("plan_sha256") != plan_sha256:
            raise ValueError("published landmark index belongs to another plan")
        for name in names:
            path = index_root / f"{name}.npy"
            if manifest["arrays"][name]["sha256"] != _file_sha256(path):
                raise ValueError(f"published landmark index array changed: {name}")
        return manifest

    index_root.mkdir(parents=True, exist_ok=True)
    count = len(population.sections)
    shapes = {
        "keys": ((count, KV_HEADS, LANDMARKS, HEAD_DIM), np.int8),
        "values": ((count, KV_HEADS, LANDMARKS, HEAD_DIM), np.int8),
        "key_scales": ((count, KV_HEADS), np.float16),
        "value_scales": ((count, KV_HEADS), np.float16),
        "token_counts": ((count, LANDMARKS), np.uint16),
        "source_tokens": ((count,), np.uint16),
    }
    temporary_paths = {name: index_root / f".{name}.tmp.npy" for name in names}
    writers: dict[str, np.memmap] = {}
    try:
        writers = {
            name: np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            for name, path in temporary_paths.items()
            for shape, dtype in (shapes[name],)
        }
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            data_path, receipt_path = _batch_paths(root, start, stop)
            items = [_section_item(section) for section in population.sections[start:stop]]
            _validate_batch(
                data_path,
                receipt_path,
                plan_sha256=plan_sha256,
                items=items,
            )
            with np.load(data_path, allow_pickle=False) as arrays:
                for name in names:
                    writers[name][start:stop] = arrays[name]
        for writer in writers.values():
            writer.flush()
            writer._mmap.close()
        writers.clear()
        for name in names:
            os.replace(temporary_paths[name], index_root / f"{name}.npy")
    finally:
        for writer in writers.values():
            writer.flush()
            writer._mmap.close()
        writers.clear()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

    arrays = {}
    for name in names:
        path = index_root / f"{name}.npy"
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        arrays[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return save(
        manifest_path,
        {
            "format": "memory-condense-qwen-int8-landmark-memmap-v1",
            "plan_sha256": plan_sha256,
            "sections": count,
            "payload_bytes_per_summary": 16_432,
            "arrays": arrays,
        },
    ).payload


def _open_index(root: Path) -> dict[str, np.ndarray]:
    return {
        name: np.load(root / "index" / f"{name}.npy", allow_pickle=False, mmap_mode="r")
        for name in ("keys", "values", "key_scales", "value_scales", "token_counts", "source_tokens")
    }


def _development_shortlists(population: Population) -> dict[str, np.ndarray]:
    frozen = read(DEVELOPMENT_RANKINGS)["bge"]["64"]
    positions = {section.section_id: index for index, section in enumerate(population.sections)}
    result = {}
    for row in frozen:
        result[row["id"]] = np.asarray(
            [positions[section_id] for section_id in row["selected_section_ids"]],
            dtype=np.int64,
        )
    if set(result) != set(population.question_ids) or any(len(value) != 64 for value in result.values()):
        raise ValueError("frozen development BGE top-64 rankings are incomplete")
    return result


def _live_bge_shortlists(
    population: Population,
    *,
    runtime: LocalRuntime,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    encoder = runtime.embedding()
    stored_identity = population.snapshot.semantic.embedding_identity
    if not compatible_summary_embedding(encoder, stored_identity):
        raise ValueError("resident BGE cannot query the stored summary address space")
    started = time.perf_counter()
    queries = np.asarray(encoder.embed_queries(population.question_texts), dtype=np.float32)
    elapsed = time.perf_counter() - started
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    documents = population.snapshot.semantic.address_matrix
    section_ids = tuple(section.section_id for section in population.sections)
    result = {}
    lookup_times = []
    for row, question_id in enumerate(population.question_ids):
        lookup_started = time.perf_counter()
        scores = documents @ queries[row]
        order = np.asarray(
            sorted(range(len(section_ids)), key=lambda index: (-float(scores[index]), section_ids[index]))[:64],
            dtype=np.int64,
        )
        lookup_times.append((time.perf_counter() - lookup_started) * 1000.0)
        result[question_id] = order
    return result, {
        "encoder_identity": summary_embedding_identity(encoder),
        "stored_embedding_identity": stored_identity,
        "query_encode_s": elapsed,
        "mean_query_encode_ms": elapsed * 1000.0 / len(population.questions),
        "top64_lookup": _percentiles(lookup_times),
    }


def _rank_landmarks(
    population: Population,
    arrays: dict[str, np.ndarray],
    shortlists: dict[str, np.ndarray],
    query_kv: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    arms = {name: {} for name in ("bge_top8", "bge_top64", "landmark_top8", "landmark_top64", "landmark_all")}
    gather_times: list[float] = []
    shortlist_score_times: list[float] = []
    corpus_score_times: list[float] = []
    section_ids = tuple(section.section_id for section in population.sections)
    all_positions = np.arange(len(section_ids), dtype=np.int64)

    def rerank(
        positions: np.ndarray,
        query_keys: np.ndarray,
        query_values: np.ndarray,
        *,
        record: bool,
    ) -> np.ndarray:
        gather_started = time.perf_counter()
        gathered = {
            name: np.ascontiguousarray(arrays[name][positions])
            for name in ("keys", "values", "key_scales", "value_scales", "token_counts")
        }
        gather_ms = (time.perf_counter() - gather_started) * 1000.0
        score_started = time.perf_counter()
        scores, _ = score_symmetric_int8_kv_landmark_batch(
            query_keys,
            query_values,
            keys=gathered["keys"],
            values=gathered["values"],
            key_scales=gathered["key_scales"],
            value_scales=gathered["value_scales"],
            token_counts=gathered["token_counts"],
            head_vote_k=4,
        )
        score_ms = (time.perf_counter() - score_started) * 1000.0
        if record:
            gather_times.append(gather_ms)
            shortlist_score_times.append(score_ms)
        else:
            corpus_score_times.append(score_ms)
        return np.asarray(
            sorted(
                range(len(positions)),
                key=lambda index: (-float(scores[index]), section_ids[int(positions[index])]),
            ),
            dtype=np.int64,
        )

    for question_id, (query_keys, query_values) in zip(
        population.question_ids, query_kv, strict=True
    ):
        top64 = shortlists[question_id]
        first8 = top64[:8]
        top8_order = rerank(first8, query_keys, query_values, record=True)
        top64_order = rerank(top64, query_keys, query_values, record=True)
        all_order = rerank(all_positions, query_keys, query_values, record=False)
        arms["bge_top8"][question_id] = first8
        arms["bge_top64"][question_id] = top64
        arms["landmark_top8"][question_id] = first8[top8_order]
        arms["landmark_top64"][question_id] = top64[top64_order]
        arms["landmark_all"][question_id] = all_positions[all_order]
    return arms, {
        "memmap_gather": _percentiles(gather_times),
        "cpu_symmetric_score": _percentiles(shortlist_score_times),
        "cpu_corpus_score": _percentiles(corpus_score_times),
    }


def _score_population(
    population: Population,
    arms: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if population.name == "development":
        for name, rankings in arms.items():
            limits = (4, 8) if name in {"bge_top8", "landmark_top8"} else (4, 8, 16, 32, 64)
            result[name] = {
                str(limit): _score_arm(
                    rankings=rankings,
                    sections=population.sections,
                    questions=population.questions,
                    reference=population.reference,
                    limit=limit,
                )[0]
                for limit in limits
            }
        return result

    section_positions = {
        section.section_id: index for index, section in enumerate(population.sections)
    }
    for name, rankings in arms.items():
        limits = (4, 8) if name in {"bge_top8", "landmark_top8"} else (4, 8, 16, 32, 64)
        result[name] = {}
        for limit in limits:
            metrics, novel = _measure(
                rankings=rankings,
                sections=population.sections,
                section_position=section_positions,
                questions=population.questions,
                supports=population.reference,
                limit=limit,
            )
            metrics["novel_questions"] = novel
            result[name][str(limit)] = metrics
    return result


def _gpu_score_benchmark(
    arrays: dict[str, np.ndarray],
    positions: np.ndarray,
    query: tuple[np.ndarray, np.ndarray],
    *,
    repeats: int,
) -> dict[str, Any]:
    import torch

    gathered = {
        name: np.ascontiguousarray(arrays[name][positions])
        for name in ("keys", "values", "key_scales", "value_scales", "token_counts")
    }
    host = {name: torch.from_numpy(value).pin_memory() for name, value in gathered.items()}
    device = {name: torch.empty_like(value, device="cuda") for name, value in host.items()}
    query_keys = torch.from_numpy(np.ascontiguousarray(query[0])).to("cuda")
    query_values = torch.from_numpy(np.ascontiguousarray(query[1])).to("cuda")

    def normalized_query(value):
        pooled = value.float().mean(dim=1)
        return pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def execute() -> None:
        for name in device:
            device[name].copy_(host[name], non_blocking=True)
        weights = device["token_counts"].float()
        weights = weights / weights.sum(dim=1, keepdim=True)
        scores = []
        for query_value, quantized, scales in (
            (query_keys, device["keys"], device["key_scales"]),
            (query_values, device["values"], device["value_scales"]),
        ):
            values = quantized.float() * scales.float()[:, :, None, None]
            values = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            pooled = torch.einsum("chld,cl->chd", values, weights)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scores.append(torch.einsum("hd,chd->ch", normalized_query(query_value), pooled))
        ((scores[0] + scores[1]) * 0.5).topk(4, dim=1).values.mean(dim=1)

    for _ in range(50):
        execute()
    torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        execute()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "candidates": len(positions),
        "transfer_bytes": int(sum(value.nbytes for value in gathered.values())),
        "end_to_end_transfer_dequantize_score": _percentiles(timings),
    }


def _query_latency(
    encoder: Any,
    question: str,
    *,
    repeats: int,
) -> dict[str, Any]:
    import torch

    prompt, span = _query_prompt(question)
    for _ in range(50):
        _project_selected_kv(encoder, [prompt], [span], layer=LAYER)
    if encoder.device.type == "cuda":
        torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        _project_selected_kv(encoder, [prompt], [span], layer=LAYER)
        if encoder.device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000.0)
    return _percentiles(timings)


def run(root: Path, *, population_name: str, batch_size: int, latency_repeats: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    population = _load_population(population_name)
    plan_artifact = _load_or_publish_plan(
        root,
        population,
        batch_size=batch_size,
    )
    plan = plan_artifact.payload
    runtime = LocalRuntime(root / "runtime", assets=RuntimeAssets.resolve(".cache"))
    try:
        attention = runtime.attention(root / "attention-cache")
        prepare_started = time.perf_counter()
        attention.prepare()
        prepare_s = time.perf_counter() - prepare_started
        encoder = attention.scorer.linker.encoder
        if encoder.checkpoint_sha256 != plan["qwen_checkpoint_sha256"]:
            raise ValueError("loaded Qwen prefix differs from the experiment plan")
        import torch

        with attention.gate, torch.cuda.stream(attention.stream), torch.inference_mode():
            compilation = _compile_batches(
                root,
                population,
                plan_sha256=plan_artifact.sha256,
                encoder=encoder,
                batch_size=batch_size,
            )
            attention.stream.synchronize()
            index_manifest = _publish_index(
                root,
                population,
                plan_sha256=plan_artifact.sha256,
                batch_size=batch_size,
            )
            query_started = time.perf_counter()
            query_kv: list[tuple[np.ndarray, np.ndarray]] = []
            for start in range(0, len(population.questions), batch_size):
                prompts_and_spans = [
                    _query_prompt(value)
                    for value in population.question_texts[start:start + batch_size]
                ]
                query_kv.extend(
                    _project_selected_kv(
                        encoder,
                        [row[0] for row in prompts_and_spans],
                        [row[1] for row in prompts_and_spans],
                        layer=LAYER,
                    )
                )
            attention.stream.synchronize()
            query_s = time.perf_counter() - query_started
            query_latency = _query_latency(
                encoder,
                population.question_texts[0],
                repeats=latency_repeats,
            )

        if population.name == "development":
            shortlists = _development_shortlists(population)
            bge_receipt = {"source": str(DEVELOPMENT_RANKINGS), "new_embedding_calls": 0}
        else:
            shortlists, bge_receipt = _live_bge_shortlists(population, runtime=runtime)
        arrays = _open_index(root)
        arms, route_timing = _rank_landmarks(population, arrays, shortlists, query_kv)
        metrics = _score_population(population, arms)
        gpu_timing = _gpu_score_benchmark(
            arrays,
            shortlists[population.question_ids[0]],
            query_kv[0],
            repeats=latency_repeats,
        )
    finally:
        runtime.close()

    rankings_payload = {
        arm: {
            question_id: [population.sections[int(position)].section_id for position in ranking]
            for question_id, ranking in values.items()
        }
        for arm, values in arms.items()
    }
    save(root / "rankings.json", rankings_payload)
    report = {
        "format": FORMAT,
        "population": population.name,
        "provider_calls": 0,
        "raw_inputs_to_qwen": False,
        "plan_sha256": plan_artifact.sha256,
        "r27": plan["r27"],
        "sections": len(population.sections),
        "questions": len(population.questions),
        "compilation": compilation,
        "index": index_manifest,
        "bge": bge_receipt,
        "metrics": metrics,
        "timing": {
            "qwen_prepare_s": prepare_s,
            "qwen_query_encode_s": query_s,
            "qwen_mean_batched_query_encode_ms": query_s * 1000.0 / len(population.questions),
            "qwen_warm_single_query": query_latency,
            "route": route_timing,
            "gpu_top64": gpu_timing,
            "full_qwen_reference_ms": 78.0,
        },
        "stop_rules": {
            "reader_calls_admitted": False,
            "reason": "Reader admission is decided after retrieval, grounding and latency audit.",
        },
    }
    save(root / "report.json", report)
    print(report, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", choices=("development", "holdout"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-repeats", type=int, default=500)
    args = parser.parse_args()
    if args.batch_size < 1 or args.latency_repeats < 1:
        parser.error("batch size and latency repeats must be positive")
    run(
        args.output,
        population_name=args.population,
        batch_size=args.batch_size,
        latency_repeats=args.latency_repeats,
    )


if __name__ == "__main__":
    main()
