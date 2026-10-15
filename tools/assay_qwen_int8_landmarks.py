"""Assay eight query-independent INT8 Qwen K/V landmarks on real memory.

This provider-free experiment reuses the 783-summary engineering archive and
the twelve source-bound questions from the attention-feature assay.  Qwen sees
summaries and questions only.  It never receives raw transcript bodies.

The matched FP32 arm isolates eight-landmark pooling loss from INT8 loss.  The
existing BGE and pooled-Qwen A@V results are copied into the report as frozen
comparators rather than recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from memory_condense.associations.qwen_int8_landmarks import (
    LANDMARK_FORMAT,
    Int8KVLandmarks,
    compile_int8_kv_landmarks,
    grouped_query_head_indices,
    score_int8_landmarks,
    score_landmark_keys,
)
from memory_condense.associations.qwen_memory_linker import (
    _capture_coverage_attention_input,
    token_span_for_characters,
)
from memory_condense.persistence import native_spine_store
from memory_condense.runtime.artifacts import read, save
from memory_condense.runtime.config import RuntimeAssets
from memory_condense.runtime.local import LocalRuntime
from tools.assay_attention_feature_retrieval import (
    ORIGINAL,
    PRIOR,
    REFERENCE,
    _score_arm,
)
from tools.characterize_native_experiments import cases
from tools.evaluate_dialogue_neighborhood import digest, source_turns


FROZEN_COMPARATOR = Path(
    "eval_results/attention-feature-retrieval-20261009-r1/report.json"
)


def _candidate_prompt(summary: str) -> tuple[str, tuple[int, int]]:
    prefix = "[Memory]\n"
    body = f"{summary}\n"
    suffix = "[Readout]\n"
    return prefix + body + suffix, (len(prefix), len(prefix) + len(body))


def _query_prompt(question: str) -> tuple[str, tuple[int, int]]:
    prefix = f"[Question] {question}\n"
    readout = "[Readout]\n"
    return prefix + readout, (len(prefix), len(prefix) + len(readout))


def _project_selected_layer(
    encoder: Any,
    texts: Sequence[str],
    spans: Sequence[tuple[int, int]],
    *,
    layer: int,
    projection: str,
) -> list[np.ndarray]:
    """Return unrotated Q or K/V projections for exact character spans."""

    torch = encoder._torch
    attention = encoder.model.layers[layer].self_attn
    original_padding_side = encoder.tokenizer.padding_side
    encoder.tokenizer.padding_side = "right"
    try:
        tokenized = encoder.tokenizer(
            list(texts),
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = tokenized.pop("offset_mapping")
        model_inputs = {key: value.to(encoder.device) for key, value in tokenized.items()}
        with torch.inference_mode():
            hidden = _capture_coverage_attention_input(
                encoder,
                model_inputs,
                layer=layer,
            )
            shape = (*hidden.shape[:-1], -1, attention.head_dim)
            if projection == "query":
                projected = attention.q_norm(attention.q_proj(hidden).view(shape))
            elif projection == "key":
                projected = attention.k_norm(attention.k_proj(hidden).view(shape))
            elif projection == "value":
                projected = attention.v_proj(hidden).view(shape)
            else:
                raise ValueError("projection must be query, key, or value")
            projected = projected.transpose(1, 2)
            result: list[np.ndarray] = []
            for row, character_span in enumerate(spans):
                positions = token_span_for_characters(
                    offsets[row].tolist(),
                    character_span,
                )
                if not positions:
                    raise ValueError("tokenizer produced an empty landmark span")
                result.append(
                    projected[row, :, positions, :].float().cpu().numpy()
                )
            return result
    finally:
        encoder.tokenizer.padding_side = original_padding_side


def _project_selected_kv(
    encoder: Any,
    texts: Sequence[str],
    spans: Sequence[tuple[int, int]],
    *,
    layer: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return unrotated K and V span projections from one shared prefix pass."""

    torch = encoder._torch
    attention = encoder.model.layers[layer].self_attn
    original_padding_side = encoder.tokenizer.padding_side
    encoder.tokenizer.padding_side = "right"
    try:
        tokenized = encoder.tokenizer(
            list(texts),
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = tokenized.pop("offset_mapping")
        model_inputs = {key: value.to(encoder.device) for key, value in tokenized.items()}
        with torch.inference_mode():
            hidden = _capture_coverage_attention_input(
                encoder,
                model_inputs,
                layer=layer,
            )
            shape = (*hidden.shape[:-1], -1, attention.head_dim)
            keys = attention.k_norm(attention.k_proj(hidden).view(shape)).transpose(1, 2)
            values = attention.v_proj(hidden).view(shape).transpose(1, 2)
            result: list[tuple[np.ndarray, np.ndarray]] = []
            for row, character_span in enumerate(spans):
                positions = token_span_for_characters(
                    offsets[row].tolist(),
                    character_span,
                )
                if not positions:
                    raise ValueError("tokenizer produced an empty landmark span")
                result.append(
                    (
                        keys[row, :, positions, :].float().cpu().numpy(),
                        values[row, :, positions, :].float().cpu().numpy(),
                    )
                )
            return result
    finally:
        encoder.tokenizer.padding_side = original_padding_side


def _compile_documents(
    encoder: Any,
    summaries: Sequence[str],
    *,
    layer: int,
    batch_size: int,
    landmark_count: int,
) -> tuple[list[Int8KVLandmarks], list[np.ndarray], dict[str, float]]:
    codes: list[Int8KVLandmarks] = []
    fp32_keys: list[np.ndarray] = []
    key_squared_error = key_squared_total = 0.0
    value_squared_error = value_squared_total = 0.0
    started = time.perf_counter()
    for start in range(0, len(summaries), batch_size):
        prompts_and_spans = [_candidate_prompt(value) for value in summaries[start:start+batch_size]]
        prompts = [value[0] for value in prompts_and_spans]
        spans = [value[1] for value in prompts_and_spans]
        projected = _project_selected_kv(encoder, prompts, spans, layer=layer)
        for key_matrix, value_matrix in projected:
            code, pooled_keys, pooled_values = compile_int8_kv_landmarks(
                key_matrix,
                value_matrix,
                landmarks=landmark_count,
            )
            dequantized_keys = code.dequantized_keys()
            dequantized_values = code.dequantized_values()
            key_squared_error += float(
                np.square(dequantized_keys - pooled_keys, dtype=np.float32).sum()
            )
            key_squared_total += float(np.square(pooled_keys, dtype=np.float32).sum())
            value_squared_error += float(
                np.square(dequantized_values - pooled_values, dtype=np.float32).sum()
            )
            value_squared_total += float(np.square(pooled_values, dtype=np.float32).sum())
            codes.append(code)
            fp32_keys.append(pooled_keys)
    elapsed = time.perf_counter() - started
    return codes, fp32_keys, {
        "compile_s": elapsed,
        "compile_s_per_summary": elapsed / len(summaries),
        "key_relative_rmse": math.sqrt(key_squared_error / max(key_squared_total, 1e-30)),
        "value_relative_rmse": math.sqrt(
            value_squared_error / max(value_squared_total, 1e-30)
        ),
    }


def _encode_queries(
    encoder: Any,
    questions: Sequence[str],
    *,
    layer: int,
    batch_size: int,
) -> tuple[list[np.ndarray], float]:
    states: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(questions), batch_size):
        prompts_and_spans = [_query_prompt(value) for value in questions[start:start+batch_size]]
        states.extend(
            _project_selected_layer(
                encoder,
                [value[0] for value in prompts_and_spans],
                [value[1] for value in prompts_and_spans],
                layer=layer,
                projection="query",
            )
        )
    return states, time.perf_counter() - started


def _ordered(scores: np.ndarray, section_ids: Sequence[str]) -> np.ndarray:
    return np.asarray(
        sorted(
            range(len(section_ids)),
            key=lambda index: (-float(scores[index]), section_ids[index]),
        ),
        dtype=np.int64,
    )


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("rankings must have equal shape")
    left_rank = np.empty(len(left), dtype=np.float64)
    right_rank = np.empty(len(right), dtype=np.float64)
    left_rank[left] = np.arange(len(left), dtype=np.float64)
    right_rank[right] = np.arange(len(right), dtype=np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _write_codes(path: Path, codes: Sequence[Int8KVLandmarks]) -> dict[str, Any]:
    keys = np.stack([code.keys for code in codes])
    values = np.stack([code.values for code in codes])
    key_scales = np.stack([code.key_scales for code in codes])
    value_scales = np.stack([code.value_scales for code in codes])
    token_counts = np.stack([code.token_counts for code in codes])
    source_tokens = np.asarray([code.source_tokens for code in codes], dtype=np.uint16)
    with path.open("wb") as handle:
        np.savez(
            handle,
            keys=keys,
            values=values,
            key_scales=key_scales,
            value_scales=value_scales,
            token_counts=token_counts,
            source_tokens=source_tokens,
        )
    payload_bytes = sum(code.retained_bytes for code in codes) + source_tokens.nbytes
    return {
        "payload_bytes": int(payload_bytes),
        "file_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _transfer_benchmark(
    torch: Any,
    *,
    landmark_bytes: int,
    full_layer_bytes: int,
    repeats: int = 200,
) -> dict[str, Any]:
    """Measure pinned host-to-device copies, including synchronization."""

    def measure(size: int, count: int) -> dict[str, float]:
        source = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        target = torch.empty(size, dtype=torch.uint8, device="cuda")
        target.copy_(source, non_blocking=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(count):
            target.copy_(source, non_blocking=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return {
            "bytes": size,
            "repeats": count,
            "mean_ms": elapsed * 1000.0 / count,
            "effective_gib_s": (size * count / elapsed) / (1024.0**3),
        }

    return {
        "eight_int8_landmarks": measure(landmark_bytes, repeats),
        "eight_full_layer5_fp16": measure(full_layer_bytes, max(25, repeats // 4)),
    }


def run(root: Path, *, batch_size: int = 8, landmark_count: int = 8) -> None:
    root.mkdir(parents=True, exist_ok=False)
    source = read(ORIGINAL / "source.json")
    directory = Path(source["directory"])
    if any(digest(directory / name) != sha for name, sha in source["files"].items()):
        raise ValueError("archived source changed")
    turns = source_turns(directory)
    snapshot = native_spine_store.load(directory / "native-spine.sqlite", turns=turns)
    sections = tuple(snapshot.semantic.sections)
    section_ids = tuple(section.section_id for section in sections)
    questions = tuple(cases())
    reference = read(REFERENCE)
    summaries = tuple(section.summary for section in sections)
    question_texts = tuple(question["question"] for question in questions)

    runtime = LocalRuntime(root / "runtime", assets=RuntimeAssets.resolve(".cache"))
    layer = 5
    try:
        attention = runtime.attention(root / "attention-cache")
        prepare_started = time.perf_counter()
        attention.prepare()
        prepare_s = time.perf_counter() - prepare_started
        import torch

        encoder = attention.scorer.linker.encoder
        module = encoder.model.layers[layer].self_attn
        with attention.gate, torch.cuda.stream(attention.stream), torch.inference_mode():
            codes, fp32_keys, compile_metrics = _compile_documents(
                encoder,
                summaries,
                layer=layer,
                batch_size=batch_size,
                landmark_count=landmark_count,
            )
            query_states, query_encode_s = _encode_queries(
                encoder,
                question_texts,
                layer=layer,
                batch_size=batch_size,
            )
            attention.stream.synchronize()

        groups = grouped_query_head_indices(
            int(encoder.config.num_attention_heads),
            int(encoder.config.num_key_value_heads),
        )
        scaling = float(module.scaling)
        fp32_rankings: dict[str, np.ndarray] = {}
        int8_rankings: dict[str, np.ndarray] = {}
        score_errors: list[float] = []
        score_started = time.perf_counter()
        for question, query in zip(questions, query_states, strict=True):
            fp32_scores = np.empty(len(codes), dtype=np.float32)
            int8_scores = np.empty(len(codes), dtype=np.float32)
            for index, (code, keys) in enumerate(zip(codes, fp32_keys, strict=True)):
                fp32_scores[index] = score_landmark_keys(
                    query,
                    keys,
                    code.token_counts,
                    key_group_indices=groups,
                    scaling=scaling,
                    head_vote_k=4,
                )[0]
                int8_scores[index] = score_int8_landmarks(
                    query,
                    code,
                    key_group_indices=groups,
                    scaling=scaling,
                    head_vote_k=4,
                )[0]
            score_errors.extend(np.abs(fp32_scores - int8_scores).tolist())
            fp32_rankings[question["id"]] = _ordered(fp32_scores, section_ids)
            int8_rankings[question["id"]] = _ordered(int8_scores, section_ids)
        score_s = time.perf_counter() - score_started

        overlaps: dict[str, float] = {}
        for limit in (8, 16, 32, 64):
            overlaps[str(limit)] = float(
                np.mean(
                    [
                        len(
                            set(fp32_rankings[q["id"]][:limit].tolist())
                            & set(int8_rankings[q["id"]][:limit].tolist())
                        )
                        / limit
                        for q in questions
                    ]
                )
            )
        correlations = [
            _rank_correlation(fp32_rankings[q["id"]], int8_rankings[q["id"]])
            for q in questions
        ]
        top1_agreement = float(
            np.mean(
                [
                    fp32_rankings[q["id"]][0] == int8_rankings[q["id"]][0]
                    for q in questions
                ]
            )
        )

        metrics: dict[str, dict[str, Any]] = {"fp32_landmarks": {}, "int8_landmarks": {}}
        details: dict[str, dict[str, Any]] = {"fp32_landmarks": {}, "int8_landmarks": {}}
        for arm, rankings in (
            ("fp32_landmarks", fp32_rankings),
            ("int8_landmarks", int8_rankings),
        ):
            for limit in (8, 16, 32, 64):
                metric, records = _score_arm(
                    rankings=rankings,
                    sections=sections,
                    questions=questions,
                    reference=reference,
                    limit=limit,
                )
                metrics[arm][str(limit)] = metric
                details[arm][str(limit)] = records

        artifact = _write_codes(root / "int8-kv-landmarks.npz", codes)
        payload_per_summary = int(sum(code.retained_bytes for code in codes) / len(codes))
        eight_candidate_bytes = payload_per_summary * 8
        full_layer5_bytes = int(
            8
            * 2
            * int(encoder.config.num_key_value_heads)
            * 128
            * int(module.head_dim)
            * 2
        )
        transfer = _transfer_benchmark(
            torch,
            landmark_bytes=eight_candidate_bytes,
            full_layer_bytes=full_layer5_bytes,
        )
    finally:
        runtime.close()

    comparator = read(FROZEN_COMPARATOR)
    report = {
        "format": "memory-condense-qwen-int8-landmark-assay-v1",
        "landmark_format": LANDMARK_FORMAT,
        "provider_calls": 0,
        "raw_inputs_to_qwen": False,
        "archive": str(directory),
        "sections": len(sections),
        "questions": len(questions),
        "qwen": {
            "layer": layer,
            "landmarks_per_summary": landmark_count,
            "kv_heads": int(codes[0].kv_heads),
            "head_dim": int(codes[0].head_dim),
            "query_heads": len(groups),
            "rotary_applied": False,
            "candidate_query_joint_encoding": False,
        },
        "metrics": metrics,
        "frozen_comparator_metrics": {
            "bge": comparator["metrics"]["bge"],
            "pooled_qwen_attention_query": comparator["metrics"]["attention_query"],
        },
        "quantization": {
            **compile_metrics,
            "mean_absolute_score_error": float(np.mean(score_errors)),
            "max_absolute_score_error": float(np.max(score_errors)),
            "mean_spearman_rank_correlation": float(np.mean(correlations)),
            "top1_agreement": top1_agreement,
            "top_k_overlap": overlaps,
        },
        "timing": {
            "qwen_prepare_s": prepare_s,
            "qwen_document_compile_s": compile_metrics["compile_s"],
            "qwen_query_encode_s": query_encode_s,
            "qwen_mean_query_encode_s": query_encode_s / len(questions),
            "cpu_score_both_arms_s": score_s,
            "cpu_score_both_arms_mean_per_question_s": score_s / len(questions),
            "pinned_transfer": transfer,
        },
        "storage": {
            "payload_bytes_per_summary": payload_per_summary,
            "payload_bytes_100k_summaries": payload_per_summary * 100_000,
            "eight_candidate_transfer_bytes": eight_candidate_bytes,
            "eight_candidate_full_layer5_fp16_bytes": full_layer5_bytes,
            "transfer_reduction": 1.0 - eight_candidate_bytes / full_layer5_bytes,
            "artifact": artifact,
        },
        "limitations": [
            "One development engineering archive and twelve source-bound questions.",
            "Unrotated separately encoded Q/K states form a semantic index; they are not an exact causal KV continuation.",
            "Contiguous mean-pooled landmarks are query-independent and therefore reusable.",
            "The matched FP32 arm isolates INT8 error, while the frozen A@V comparator uses a different attention representation.",
            "This is a pre-reader evidence-reachability assay and does not measure final generated answer quality.",
        ],
    }
    save(root / "rankings.json", details)
    save(root / "report.json", report)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--landmarks", type=int, default=8)
    args = parser.parse_args()
    run(args.output, batch_size=args.batch_size, landmark_count=args.landmarks)


if __name__ == "__main__":
    main()
