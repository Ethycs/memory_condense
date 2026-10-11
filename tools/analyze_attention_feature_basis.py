"""Compress a measured Qwen A@V retrieval space and recheck exact support."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from memory_condense.persistence import native_spine_store
from memory_condense.runtime.artifacts import read, save
from tools.assay_attention_feature_retrieval import (
    ORIGINAL,
    REFERENCE,
    _normalize,
    _ordered,
    _score_arm,
)
from tools.characterize_native_experiments import cases
from tools.evaluate_dialogue_neighborhood import source_turns


def _basis(matrix: np.ndarray, *, centered: bool, maximum: int):
    mean = matrix.mean(axis=0, keepdims=True) if centered else np.zeros((1, matrix.shape[1]), np.float32)
    values = matrix - mean
    gram = values @ values.T
    eigenvalues, left = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    left = left[:, order]
    positive = eigenvalues > max(float(eigenvalues[0]) * 1e-7, 1e-9)
    width = min(maximum, int(positive.sum()))
    scales = np.sqrt(eigenvalues[:width])
    components = (left[:, :width].T @ values) / scales[:, None]
    return mean, components.astype(np.float32), eigenvalues


def _novel_question_summary(records: list[dict]) -> list[dict]:
    """Keep a compact audit trail for gains over the production packet."""
    result = []
    for record in records:
        novel_quotes = sum(fact["novel_quotes"] for fact in record["facts"].values())
        completed_facts = [
            fact_id
            for fact_id, fact in record["facts"].items()
            if fact["union_quotes"] == fact["total_quotes"]
            and fact["baseline_quotes"] < fact["total_quotes"]
        ]
        if novel_quotes or completed_facts:
            result.append(
                {
                    "id": record["id"],
                    "novel_quotes_over_baseline": novel_quotes,
                    "newly_completed_facts": completed_facts,
                }
            )
    return result


def run(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    archive = read(ORIGINAL / "source.json")
    directory = Path(archive["directory"])
    turns = source_turns(directory)
    snapshot = native_spine_store.load(directory / "native-spine.sqlite", turns=turns)
    sections = tuple(snapshot.semantic.sections)
    section_ids = tuple(section.section_id for section in sections)
    questions = tuple(cases())
    reference = read(REFERENCE)
    documents = np.load(source / "attention-values-fp16.npy", allow_pickle=False).astype(np.float32)
    queries = np.load(source / "attention-query-values-fp16.npy", allow_pickle=False).astype(np.float32)
    if documents.shape != (len(sections), 4096) or queries.shape != (len(questions), 4096):
        raise ValueError("attention feature population changed")

    results = {}
    started = time.perf_counter()
    for name, centered in (("uncentered_svd", False), ("centered_pca", True)):
        mean, components, eigenvalues = _basis(documents, centered=centered, maximum=256)
        total = float(eigenvalues.sum())
        cumulative = np.cumsum(eigenvalues) / max(total, 1e-12)
        arm = {
            "effective_rank_participation": float(total * total / np.square(eigenvalues).sum()),
            "stable_rank": float(total / max(float(eigenvalues[0]), 1e-12)),
            "dimensions_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
            "dimensions_for_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
            "dimensions_for_99pct": int(np.searchsorted(cumulative, 0.99) + 1),
            "ranks": {},
        }
        for width in (8, 16, 32, 64, 128, 256):
            if width > len(components):
                continue
            document_codes = _normalize((documents - mean) @ components[:width].T)
            query_codes = _normalize((queries - mean) @ components[:width].T)
            rankings = {
                question["id"]: _ordered(document_codes @ query_codes[row], section_ids)
                for row, question in enumerate(questions)
            }
            metrics = {}
            for limit in (8, 16, 32, 64):
                score, records = _score_arm(
                    rankings=rankings,
                    sections=sections,
                    questions=questions,
                    reference=reference,
                    limit=limit,
                )
                score["novel_questions"] = _novel_question_summary(records)
                metrics[str(limit)] = score
            document_code_bytes = int(len(sections) * width * 2)
            basis_bytes = int(width * documents.shape[1] * 2)
            arm["ranks"][str(width)] = {
                "explained_energy": float(cumulative[width - 1]),
                "fp16_document_code_bytes": document_code_bytes,
                "fp16_basis_bytes": basis_bytes,
                "fp16_total_bytes": document_code_bytes + basis_bytes,
                "metrics": metrics,
            }
        results[name] = arm
    report = {
        "format": "memory-condense-attention-feature-basis-analysis-v1",
        "source": str(source),
        "sections": len(sections),
        "questions": len(questions),
        "original_width": int(documents.shape[1]),
        "elapsed_s": time.perf_counter() - started,
        "results": results,
        "limitations": [
            "The basis is unsupervised and fitted on one archive's stored summaries.",
            "Questions and support labels do not participate in basis fitting.",
            "Support measurements remain provider-free pre-reader retrieval gates.",
        ],
    }
    save(output / "report.json", report)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()
