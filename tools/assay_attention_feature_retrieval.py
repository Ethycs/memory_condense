"""Measure whether retained Qwen A@V features add authenticated retrieval support.

This is a provider-free structural assay.  It reuses one persisted engineering
archive, its twelve source-bound questions, and the existing BGE address matrix.
Qwen sees stored summaries and the live questions, never raw transcript bodies.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np

from memory_condense.persistence import native_spine_store
from memory_condense.runtime.artifacts import read, save
from memory_condense.runtime.config import RuntimeAssets
from memory_condense.runtime.local import LocalRuntime
from tools.characterize_native_experiments import cases, covered_quote
from tools.evaluate_dialogue_neighborhood import digest, source_turns


ORIGINAL = Path("eval_results/dialogue-neighborhood-20261006-r1")
PRIOR = Path("eval_results/native-experiment-quality-20261008-r1")
REFERENCE = Path(
    "eval_results/cav-reinjection-five-concept-20261009-r1/private-reference.json"
)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if value.ndim != 2 or np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("feature matrix must contain finite nonzero rows")
    return value / norms


def _ordered(scores: np.ndarray, section_ids: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        sorted(range(len(section_ids)), key=lambda i: (-float(scores[i]), section_ids[i])),
        dtype=np.int64,
    )


def _reciprocal_rank_scores(scores: np.ndarray, section_ids: tuple[str, ...]) -> np.ndarray:
    order = _ordered(scores, section_ids)
    ranks = np.empty(len(order), dtype=np.float32)
    ranks[order] = np.arange(len(order), dtype=np.float32)
    return 1.0 / (60.0 + ranks)


def _section_covers(support: dict, sections: tuple[object, ...], positions: np.ndarray) -> bool:
    intervals = []
    for position in positions:
        for span in sections[int(position)].spans:
            if span.turn_id == support["turn_id"]:
                intervals.append((int(span.start_char), int(span.end_char)))
    cursor = int(support["start_char"])
    target = int(support["end_char"])
    for start, end in sorted(intervals):
        if end <= cursor:
            continue
        if start > cursor:
            break
        cursor = max(cursor, end)
        if cursor >= target:
            return True
    return False


def _score_arm(
    *,
    rankings: dict[str, np.ndarray],
    sections: tuple[object, ...],
    questions: tuple[dict, ...],
    reference: dict,
    limit: int,
) -> tuple[dict, list[dict]]:
    quote_total = quote_hits = union_hits = novel_hits = 0
    fact_total = fact_hits = union_fact_hits = novel_fact_hits = 0
    records = []
    for question in questions:
        qid = question["id"]
        selected = rankings[qid][:limit]
        packet = read(PRIOR / "packets" / qid / "baseline.json")
        row_facts = {}
        for fact_id in question["facts"]:
            supports = tuple(reference["facts"][fact_id]["supports"])
            selected_values = tuple(_section_covers(s, sections, selected) for s in supports)
            baseline_values = tuple(covered_quote(s, packet["references"]) for s in supports)
            union_values = tuple(a or b for a, b in zip(selected_values, baseline_values))
            new_values = tuple(a and not b for a, b in zip(selected_values, baseline_values))
            quote_total += len(supports)
            quote_hits += sum(selected_values)
            union_hits += sum(union_values)
            novel_hits += sum(new_values)
            fact_total += 1
            fact_hits += all(selected_values)
            union_fact_hits += all(union_values)
            novel_fact_hits += all(union_values) and not all(baseline_values)
            row_facts[fact_id] = {
                "selected_quotes": sum(selected_values),
                "baseline_quotes": sum(baseline_values),
                "union_quotes": sum(union_values),
                "total_quotes": len(supports),
                "novel_quotes": sum(new_values),
            }
        records.append(
            {
                "id": qid,
                "selected_section_ids": [sections[int(i)].section_id for i in selected],
                "facts": row_facts,
            }
        )
    return (
        {
            "limit": limit,
            "quotes": quote_hits,
            "quote_total": quote_total,
            "quote_recall": quote_hits / quote_total,
            "complete_facts": fact_hits,
            "fact_total": fact_total,
            "complete_fact_recall": fact_hits / fact_total,
            "baseline_union_quotes": union_hits,
            "novel_quotes_over_baseline": novel_hits,
            "baseline_union_complete_facts": union_fact_hits,
            "novel_complete_facts_over_baseline": novel_fact_hits,
        },
        records,
    )


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run(root: Path) -> None:
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
    summaries = [section.summary for section in sections]
    prompts = [question["question"] for question in questions]

    runtime = LocalRuntime(root / "runtime", assets=RuntimeAssets.resolve(".cache"))
    try:
        attention = runtime.attention(root / "attention-cache")
        prepare_started = time.perf_counter()
        attention.prepare()
        prepare_s = time.perf_counter() - prepare_started
        import torch

        encoder = attention.scorer.linker.encoder
        with attention.gate, torch.cuda.stream(attention.stream), torch.inference_mode():
            started = time.perf_counter()
            document_features = encoder.encode_attention_values(
                summaries, layers=(4,), batch_size=8
            )[4].numpy().astype(np.float32)
            attention.stream.synchronize()
            document_encode_s = time.perf_counter() - started
            started = time.perf_counter()
            query_features = encoder.encode_attention_values(
                prompts, layers=(4,), batch_size=8
            )[4].numpy().astype(np.float32)
            attention.stream.synchronize()
            query_encode_s = time.perf_counter() - started

        bge = runtime.embedding()
        started = time.perf_counter()
        query_bge = _normalize(np.asarray(bge.embed_queries(prompts), dtype=np.float32))
        bge_query_encode_s = time.perf_counter() - started
        document_bge = _normalize(snapshot.semantic.address_matrix)
    finally:
        runtime.close()

    np.save(root / "attention-values-fp16.npy", document_features.astype(np.float16))
    np.save(root / "attention-query-values-fp16.npy", query_features.astype(np.float16))
    attention_documents = _normalize(document_features)
    attention_queries = _normalize(query_features)
    attention_graph = attention_documents @ attention_documents.T
    np.fill_diagonal(attention_graph, -1.0)

    rankings: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in (
            "bge",
            "attention_query",
            "bge_attention_rrf",
            "bge_seed_attention_neighbors",
        )
    }
    for row, question in enumerate(questions):
        qid = question["id"]
        bge_scores = document_bge @ query_bge[row]
        attention_scores = attention_documents @ attention_queries[row]
        rrf = _reciprocal_rank_scores(bge_scores, section_ids)
        rrf += _reciprocal_rank_scores(attention_scores, section_ids)
        seed_order = _ordered(bge_scores, section_ids)[:8]
        seed_logits = bge_scores[seed_order].astype(np.float64)
        seed_weights = np.exp(seed_logits - seed_logits.max())
        seed_weights /= seed_weights.sum()
        neighbor_scores = attention_graph[:, seed_order] @ seed_weights.astype(np.float32)
        neighbor_rrf = _reciprocal_rank_scores(bge_scores, section_ids)
        neighbor_rrf += _reciprocal_rank_scores(neighbor_scores, section_ids)
        rankings["bge"][qid] = _ordered(bge_scores, section_ids)
        rankings["attention_query"][qid] = _ordered(attention_scores, section_ids)
        rankings["bge_attention_rrf"][qid] = _ordered(rrf, section_ids)
        rankings["bge_seed_attention_neighbors"][qid] = _ordered(
            neighbor_rrf, section_ids
        )

    metrics = {}
    details = {}
    for arm, arm_rankings in rankings.items():
        metrics[arm] = {}
        details[arm] = {}
        for limit in (8, 16, 32, 64):
            result, records = _score_arm(
                rankings=arm_rankings,
                sections=sections,
                questions=questions,
                reference=reference,
                limit=limit,
            )
            metrics[arm][str(limit)] = result
            details[arm][str(limit)] = records

    baseline_quotes = baseline_facts = total_quotes = total_facts = 0
    for question in questions:
        packet = read(PRIOR / "packets" / question["id"] / "baseline.json")
        for fact_id in question["facts"]:
            values = [
                covered_quote(support, packet["references"])
                for support in reference["facts"][fact_id]["supports"]
            ]
            baseline_quotes += sum(values)
            total_quotes += len(values)
            baseline_facts += all(values)
            total_facts += 1

    attention_cache = Path(
        "eval_results/native-spine-engineering-live-session-20260922-r1/attention-cache"
    )
    matrix_fp16_bytes = int(len(sections) * len(sections) * 2)
    report = {
        "format": "memory-condense-attention-feature-retrieval-assay-v1",
        "provider_calls": 0,
        "raw_inputs_to_qwen": False,
        "attention_representation": "layer-4 mean-pooled concatenated per-head A@V before W_O",
        "archive": str(directory),
        "sections": len(sections),
        "questions": len(questions),
        "feature_width": int(document_features.shape[1]),
        "baseline": {
            "quotes": baseline_quotes,
            "quote_total": total_quotes,
            "complete_facts": baseline_facts,
            "fact_total": total_facts,
        },
        "metrics": metrics,
        "timing": {
            "qwen_prepare_s": prepare_s,
            "qwen_document_encode_s": document_encode_s,
            "qwen_query_encode_s": query_encode_s,
            "qwen_mean_query_encode_s": query_encode_s / len(questions),
            "bge_query_encode_s": bge_query_encode_s,
        },
        "storage": {
            "existing_scalar_attention_cache_bytes": _directory_bytes(attention_cache),
            "persisted_fp16_attention_codes_bytes": (root / "attention-values-fp16.npy").stat().st_size,
            "dense_document_similarity_fp16_bytes_if_retained": matrix_fp16_bytes,
            "native_memory_directory_bytes": _directory_bytes(directory),
        },
        "artifacts": {
            "attention_values_fp16_sha256": hashlib.sha256(
                (root / "attention-values-fp16.npy").read_bytes()
            ).hexdigest(),
            "attention_query_values_fp16_sha256": hashlib.sha256(
                (root / "attention-query-values-fp16.npy").read_bytes()
            ).hexdigest(),
        },
        "limitations": [
            "One development engineering archive and twelve questions.",
            "Direct attention-query retrieval pays a Qwen query pass; the BGE-seed neighbor arm does not.",
            "This measures authenticated support reachability before packet budgeting or reader generation.",
            "The representation is pooled A@V, not raw token-by-token attention maps.",
        ],
    }
    save(root / "report.json", report)
    save(root / "rankings.json", details)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
