"""Rerank frozen BGE top-eight candidates with reusable INT8 Qwen landmarks.

This bounded companion to ``assay_qwen_int8_landmarks`` is intended for a busy
GPU host.  It compiles only the 79 unique summaries already selected by the
twelve-question BGE top-eight control and can run the six-layer Qwen prefix on
CPU without evicting resident production models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from memory_condense.associations.qwen_int8_landmarks import (
    grouped_query_head_indices,
    score_int8_landmarks,
    score_landmark_keys,
)
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
from memory_condense.persistence import native_spine_store
from memory_condense.runtime.artifacts import read, save
from tools.assay_attention_feature_retrieval import (
    ORIGINAL,
    REFERENCE,
    _score_arm,
)
from tools.assay_qwen_int8_landmarks import (
    _compile_documents,
    _encode_queries,
    _rank_correlation,
    _write_codes,
)
from tools.characterize_native_experiments import cases
from tools.evaluate_dialogue_neighborhood import digest, source_turns


FROZEN_RANKINGS = Path(
    "eval_results/attention-feature-retrieval-20261009-r1/rankings.json"
)


def run(root: Path, *, threads: int = 2) -> None:
    root.mkdir(parents=True, exist_ok=False)
    source = read(ORIGINAL / "source.json")
    directory = Path(source["directory"])
    if any(digest(directory / name) != sha for name, sha in source["files"].items()):
        raise ValueError("archived source changed")
    turns = source_turns(directory)
    snapshot = native_spine_store.load(directory / "native-spine.sqlite", turns=turns)
    sections = tuple(snapshot.semantic.sections)
    section_by_id = {section.section_id: section for section in sections}
    position_by_id = {section.section_id: index for index, section in enumerate(sections)}
    questions = tuple(cases())
    reference = read(REFERENCE)

    frozen = read(FROZEN_RANKINGS)["bge"]["8"]
    frozen_by_question = {
        row["id"]: tuple(row["selected_section_ids"])
        for row in frozen
    }
    candidate_ids = tuple(
        sorted(
            {section_id for ids in frozen_by_question.values() for section_id in ids},
            key=position_by_id.__getitem__,
        )
    )
    candidate_sections = tuple(section_by_id[value] for value in candidate_ids)

    import torch

    torch.set_num_threads(threads)
    layer = 5
    prepare_started = time.perf_counter()
    encoder = Qwen3PrefixEncoder(
        ".cache/models/Qwen3-8B",
        layers=6,
        device="cpu",
        dtype="float32",
        verify_checkpoint_files=False,
    )
    prepare_s = time.perf_counter() - prepare_started
    module = encoder.model.layers[layer].self_attn
    codes, fp32_keys, compile_metrics = _compile_documents(
        encoder,
        [section.summary for section in candidate_sections],
        layer=layer,
        batch_size=8,
        landmark_count=8,
    )
    query_states, query_encode_s = _encode_queries(
        encoder,
        [question["question"] for question in questions],
        layer=layer,
        batch_size=8,
    )
    groups = grouped_query_head_indices(
        int(encoder.config.num_attention_heads),
        int(encoder.config.num_key_value_heads),
    )
    scaling = float(module.scaling)
    code_position = {section_id: index for index, section_id in enumerate(candidate_ids)}

    bge_rankings: dict[str, np.ndarray] = {}
    fp32_rankings: dict[str, np.ndarray] = {}
    int8_rankings: dict[str, np.ndarray] = {}
    score_errors: list[float] = []
    correlations: list[float] = []
    top1: list[bool] = []
    score_started = time.perf_counter()
    for question, query in zip(questions, query_states, strict=True):
        qid = question["id"]
        shortlist = frozen_by_question[qid]
        fp32_scores = np.empty(len(shortlist), dtype=np.float32)
        int8_scores = np.empty(len(shortlist), dtype=np.float32)
        for row, section_id in enumerate(shortlist):
            index = code_position[section_id]
            code = codes[index]
            fp32_scores[row] = score_landmark_keys(
                query,
                fp32_keys[index],
                code.token_counts,
                key_group_indices=groups,
                scaling=scaling,
                head_vote_k=4,
            )[0]
            int8_scores[row] = score_int8_landmarks(
                query,
                code,
                key_group_indices=groups,
                scaling=scaling,
                head_vote_k=4,
            )[0]
        fp32_order = np.asarray(
            sorted(range(len(shortlist)), key=lambda i: (-float(fp32_scores[i]), shortlist[i])),
            dtype=np.int64,
        )
        int8_order = np.asarray(
            sorted(range(len(shortlist)), key=lambda i: (-float(int8_scores[i]), shortlist[i])),
            dtype=np.int64,
        )
        score_errors.extend(np.abs(fp32_scores - int8_scores).tolist())
        correlations.append(_rank_correlation(fp32_order, int8_order))
        top1.append(bool(fp32_order[0] == int8_order[0]))
        bge_rankings[qid] = np.asarray(
            [position_by_id[value] for value in shortlist], dtype=np.int64
        )
        fp32_rankings[qid] = np.asarray(
            [position_by_id[shortlist[index]] for index in fp32_order], dtype=np.int64
        )
        int8_rankings[qid] = np.asarray(
            [position_by_id[shortlist[index]] for index in int8_order], dtype=np.int64
        )
    score_s = time.perf_counter() - score_started

    metrics = {}
    details = {}
    for arm, rankings in (
        ("bge_frozen", bge_rankings),
        ("fp32_landmark_rerank", fp32_rankings),
        ("int8_landmark_rerank", int8_rankings),
    ):
        metrics[arm] = {}
        details[arm] = {}
        for limit in (4, 8):
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
    report = {
        "format": "memory-condense-qwen-int8-landmark-shortlist-assay-v1",
        "provider_calls": 0,
        "raw_inputs_to_qwen": False,
        "execution": {
            "device": "cpu",
            "dtype": "float32",
            "threads": threads,
            "reason": "preserve the concurrently resident GPU benchmark models",
        },
        "population": {
            "archive_sections": len(sections),
            "compiled_unique_summaries": len(candidate_sections),
            "questions": len(questions),
            "candidates_per_question": 8,
            "candidate_source": "frozen BGE top-eight ranking",
        },
        "metrics": metrics,
        "quantization": {
            **compile_metrics,
            "mean_absolute_score_error": float(np.mean(score_errors)),
            "max_absolute_score_error": float(np.max(score_errors)),
            "mean_spearman_rank_correlation": float(np.mean(correlations)),
            "top1_agreement": float(np.mean(top1)),
            "full_top8_set_agreement": 1.0,
        },
        "timing": {
            "qwen_prepare_s": prepare_s,
            "qwen_document_compile_s": compile_metrics["compile_s"],
            "qwen_query_encode_s": query_encode_s,
            "cpu_score_both_landmark_arms_s": score_s,
        },
        "storage": {
            "payload_bytes_per_summary": codes[0].retained_bytes,
            "payload_bytes_100k_summaries": codes[0].retained_bytes * 100_000,
            "eight_candidate_transfer_bytes": codes[0].retained_bytes * 8,
            "artifact": artifact,
        },
        "limitations": [
            "Landmarks rerank a frozen BGE top-eight shortlist; this run does not test corpus-wide landmark retrieval.",
            "Top-eight evidence recall cannot change because reranking preserves the candidate set; top-four measures prioritization.",
            "The matched FP32 arm isolates quantization error from landmark pooling and separate-encoding error.",
            "This is a pre-reader support assay, not a generated-answer evaluation.",
        ],
    }
    save(root / "compiled-section-ids.json", {"section_ids": list(candidate_ids)})
    save(root / "rankings.json", details)
    save(root / "report.json", report)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.output, threads=args.threads)


if __name__ == "__main__":
    main()
