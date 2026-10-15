"""Reuse compiled INT8 landmarks and test symmetric K/V query readout."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np

from memory_condense.associations.qwen_int8_landmarks import (
    Int8KVLandmarks,
    score_symmetric_int8_kv_landmarks,
)
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
from memory_condense.persistence import native_spine_store
from memory_condense.runtime.artifacts import read, save
from tools.assay_attention_feature_retrieval import ORIGINAL, REFERENCE, _score_arm
from tools.assay_qwen_int8_landmarks import _project_selected_kv, _query_prompt
from tools.characterize_native_experiments import cases
from tools.evaluate_dialogue_neighborhood import digest, source_turns


FROZEN_RANKINGS = Path(
    "eval_results/attention-feature-retrieval-20261009-r1/rankings.json"
)


def _load_codes(source: Path) -> tuple[tuple[Int8KVLandmarks, ...], tuple[str, ...]]:
    report = read(source / "report.json")
    artifact = source / "int8-kv-landmarks.npz"
    expected = report["storage"]["artifact"]["sha256"]
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
        raise ValueError("compiled landmark artifact changed")
    arrays = np.load(artifact, allow_pickle=False)
    codes = tuple(
        Int8KVLandmarks(
            keys=arrays["keys"][index],
            values=arrays["values"][index],
            key_scales=arrays["key_scales"][index],
            value_scales=arrays["value_scales"][index],
            token_counts=arrays["token_counts"][index],
            source_tokens=int(arrays["source_tokens"][index]),
        )
        for index in range(len(arrays["keys"]))
    )
    section_ids = tuple(read(source / "compiled-section-ids.json")["section_ids"])
    if len(codes) != len(section_ids):
        raise ValueError("landmark codes and section IDs have different lengths")
    return codes, section_ids


def run(root: Path, *, source: Path, threads: int = 2) -> None:
    root.mkdir(parents=True, exist_ok=False)
    codes, compiled_ids = _load_codes(source)
    code_by_id = dict(zip(compiled_ids, codes, strict=True))

    source_receipt = read(ORIGINAL / "source.json")
    directory = Path(source_receipt["directory"])
    if any(
        digest(directory / name) != sha
        for name, sha in source_receipt["files"].items()
    ):
        raise ValueError("archived source changed")
    turns = source_turns(directory)
    snapshot = native_spine_store.load(directory / "native-spine.sqlite", turns=turns)
    sections = tuple(snapshot.semantic.sections)
    position_by_id = {section.section_id: index for index, section in enumerate(sections)}
    questions = tuple(cases())
    reference = read(REFERENCE)
    frozen = read(FROZEN_RANKINGS)["bge"]["8"]
    frozen_by_question = {
        row["id"]: tuple(row["selected_section_ids"])
        for row in frozen
    }

    import torch

    torch.set_num_threads(threads)
    prepare_started = time.perf_counter()
    encoder = Qwen3PrefixEncoder(
        ".cache/models/Qwen3-8B",
        layers=6,
        device="cpu",
        dtype="float32",
        verify_checkpoint_files=False,
    )
    prepare_s = time.perf_counter() - prepare_started

    query_started = time.perf_counter()
    query_kv: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, len(questions), 8):
        prompts_and_spans = [
            _query_prompt(question["question"])
            for question in questions[start:start + 8]
        ]
        query_kv.extend(
            _project_selected_kv(
                encoder,
                [value[0] for value in prompts_and_spans],
                [value[1] for value in prompts_and_spans],
                layer=5,
            )
        )
    query_s = time.perf_counter() - query_started

    bge_rankings: dict[str, np.ndarray] = {}
    symmetric_rankings: dict[str, np.ndarray] = {}
    score_started = time.perf_counter()
    for question, (query_keys, query_values) in zip(questions, query_kv, strict=True):
        qid = question["id"]
        shortlist = frozen_by_question[qid]
        scores = np.asarray(
            [
                score_symmetric_int8_kv_landmarks(
                    query_keys,
                    query_values,
                    code_by_id[section_id],
                    head_vote_k=4,
                )[0]
                for section_id in shortlist
            ],
            dtype=np.float32,
        )
        order = sorted(
            range(len(shortlist)),
            key=lambda index: (-float(scores[index]), shortlist[index]),
        )
        bge_rankings[qid] = np.asarray(
            [position_by_id[value] for value in shortlist], dtype=np.int64
        )
        symmetric_rankings[qid] = np.asarray(
            [position_by_id[shortlist[index]] for index in order], dtype=np.int64
        )
    score_s = time.perf_counter() - score_started

    metrics = {}
    details = {}
    for arm, rankings in (
        ("bge_frozen", bge_rankings),
        ("int8_symmetric_kv", symmetric_rankings),
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

    report = {
        "format": "memory-condense-qwen-int8-landmark-readout-assay-v1",
        "provider_calls": 0,
        "raw_inputs_to_qwen": False,
        "source_landmark_artifact": {
            "path": str(source),
            "sha256": read(source / "report.json")["storage"]["artifact"]["sha256"],
            "compiled_summaries_reused": len(codes),
            "candidate_reencoding_passes": 0,
        },
        "readout": "mean-pooled symmetric K/K plus V/V cosine; top four of eight KV heads",
        "metrics": metrics,
        "timing": {
            "qwen_prepare_s": prepare_s,
            "qwen_query_kv_encode_s": query_s,
            "qwen_mean_query_kv_encode_s": query_s / len(questions),
            "cpu_score_s": score_s,
        },
        "limitations": [
            "The same twelve development questions characterize the readout; no tuning or threshold was fitted.",
            "The readout reranks a frozen BGE top-eight shortlist and cannot recover evidence outside it.",
            "This is a provider-free pre-reader support assay.",
        ],
    }
    save(root / "rankings.json", details)
    save(root / "report.json", report)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.output, source=args.source, threads=args.threads)


if __name__ == "__main__":
    main()
