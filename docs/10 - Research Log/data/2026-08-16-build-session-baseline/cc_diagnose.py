"""Cheap follow-up diagnostics over a persisted B0 chunk store.

This script never ingests or mutates the corpus.  It repeats the established
hybrid query sequence, measures stability, sweeps only retrieval-time settings,
and reports where the gold chunks sit in the exact dense and BM25 rankings.
The probe is a development set; sweep results must not be presented as blind.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from memory_condense._tokenizer import count_tokens
from memory_condense.condenser import MemoryCondenser
from memory_condense.eval.recall import contains_answer


ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--store",
    type=Path,
    default=ROOT / "data" / "build-session-8f7f7561.store",
)
parser.add_argument("--probe", type=Path, default=BASE / "cc_probe.json")
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "eval_results" / "build_session_b0_diagnostics.json",
)
args = parser.parse_args()

probe = json.loads(args.probe.read_text(encoding="utf-8"))
questions = [
    {"question_id": f"q{i}", "question": q, "answer": a, "category": c}
    for i, (q, a, c, _) in enumerate(probe)
]


def evaluate(
    mc: MemoryCondenser,
    query_embeddings: dict[str, np.ndarray],
    *,
    alpha: float,
    candidates: int = 100,
) -> dict:
    rows = []
    for question in questions:
        results = mc._retriever.hybrid_query(
            query_text=question["question"],
            query_embedding=query_embeddings[question["question_id"]],
            k=10,
            ef_search=max(50, candidates),
            candidates=candidates,
            alpha=alpha,
        )
        texts = [result.chunk.text for result in results]
        rows.append(
            {
                **question,
                "hit": contains_answer(texts, question["answer"]),
                "answer_rank": next(
                    (
                        rank
                        for rank, text in enumerate(texts, start=1)
                        if contains_answer([text], question["answer"])
                    ),
                    None,
                ),
                "context_tokens": sum(count_tokens(text) for text in texts),
                "chunk_ids": [result.chunk.chunk_id for result in results],
                "results": [
                    {
                        "chunk_id": result.chunk.chunk_id,
                        "score": result.score,
                        "dense_score": result.dense_score,
                        "lexical_score": result.lexical_score,
                        "token_count": result.chunk.token_count,
                        "text_excerpt": result.chunk.text[:600],
                    }
                    for result in results
                ],
            }
        )
    return {
        "alpha": alpha,
        "candidates": candidates,
        "recall": sum(row["hit"] for row in rows) / len(rows),
        "mean_context_tokens": sum(row["context_tokens"] for row in rows)
        / len(rows),
        "rows": rows,
    }


mc = MemoryCondenser(data_dir=args.store, auto_extract=False, device="cpu")
try:
    query_embeddings = {
        question["question_id"]: np.asarray(
            mc._embedder.embed_query(question["question"]), dtype=np.float32
        )
        for question in questions
    }
    first = evaluate(mc, query_embeddings, alpha=0.65)
    second = evaluate(mc, query_embeddings, alpha=0.65)
    changed = [
        {
            "question_id": left["question_id"],
            "first_hit": left["hit"],
            "second_hit": right["hit"],
            "first_ids": left["chunk_ids"],
            "second_ids": right["chunk_ids"],
        }
        for left, right in zip(first["rows"], second["rows"])
        if left["chunk_ids"] != right["chunk_ids"]
    ]

    sweeps = [
        evaluate(mc, query_embeddings, alpha=alpha)
        for alpha in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
    ]
    candidate_sweeps = [
        evaluate(mc, query_embeddings, alpha=0.65, candidates=candidates)
        for candidates in (25, 50, 100, 200, 400)
    ]

    stored = mc._db.execute(
        "SELECT chunk_id, turn_id, rowid, text, token_count, embedding FROM chunks "
        "WHERE embedding IS NOT NULL ORDER BY rowid"
    ).fetchall()
    chunk_ids = [row[0] for row in stored]
    texts = [row[3] for row in stored]
    row_by_chunk_id = {row[0]: int(row[2]) for row in stored}
    matrix = np.stack([np.frombuffer(row[5], dtype=np.float32) for row in stored])
    matrix_norms = np.linalg.norm(matrix, axis=1)

    misses = []
    for baseline_row in first["rows"]:
        if baseline_row["hit"]:
            continue
        query = baseline_row["question"]
        answer = baseline_row["answer"]
        gold_indices = [
            i for i, text in enumerate(texts) if contains_answer([text], answer)
        ]
        query_vector = query_embeddings[baseline_row["question_id"]]
        denominator = matrix_norms * max(float(np.linalg.norm(query_vector)), 1e-12)
        dense_scores = (matrix @ query_vector) / np.maximum(denominator, 1e-12)
        dense_order = np.argsort(-dense_scores)
        dense_ranks = {int(index): rank for rank, index in enumerate(dense_order, 1)}

        lexical = mc._retriever.lexical.search(query, limit=len(stored))
        lexical_ranks = {
            chunk_id: rank for rank, (chunk_id, _) in enumerate(lexical, 1)
        }
        misses.append(
            {
                "question_id": baseline_row["question_id"],
                "question": query,
                "answer": answer,
                "gold_chunks": [
                    {
                        "chunk_id": chunk_ids[index],
                        "turn_id": stored[index][1],
                        "rowid": int(stored[index][2]),
                        "dense_rank_exact": dense_ranks[index],
                        "dense_cosine": float(dense_scores[index]),
                        "lexical_rank": lexical_ranks.get(chunk_ids[index]),
                        "token_count": int(stored[index][4]),
                        "text_excerpt": texts[index][:600],
                    }
                    for index in gold_indices
                ],
                "baseline_top_10": [
                    {
                        **result,
                        "rowid": row_by_chunk_id[result["chunk_id"]],
                        "nearest_gold_row_distance": min(
                            (
                                abs(
                                    row_by_chunk_id[result["chunk_id"]]
                                    - int(stored[index][2])
                                )
                                for index in gold_indices
                            ),
                            default=None,
                        ),
                    }
                    for result in baseline_row["results"]
                ],
            }
        )
finally:
    mc.close()

report = {
    "protocol": "B0 development diagnostics; no new blind claim",
    "store": str(args.store),
    "repeat": {
        "first_recall": first["recall"],
        "second_recall": second["recall"],
        "changed_queries": changed,
    },
    "alpha_sweep": [
        {
            "alpha": result["alpha"],
            "recall": result["recall"],
            "mean_context_tokens": result["mean_context_tokens"],
            "misses": [row["question_id"] for row in result["rows"] if not row["hit"]],
        }
        for result in sweeps
    ],
    "candidate_sweep": [
        {
            "candidates": result["candidates"],
            "recall": result["recall"],
            "mean_context_tokens": result["mean_context_tokens"],
            "misses": [row["question_id"] for row in result["rows"] if not row["hit"]],
        }
        for result in candidate_sweeps
    ],
    "baseline_rows": first["rows"],
    "miss_diagnostics": misses,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(
    f"repeat: {first['recall']:.1%} -> {second['recall']:.1%}; "
    f"rankings changed for {len(changed)} queries"
)
for result in report["alpha_sweep"]:
    print(
        f"alpha={result['alpha']:.2f}: {result['recall']:.1%}; "
        f"misses={','.join(result['misses']) or '-'}"
    )
for result in report["candidate_sweep"]:
    print(
        f"candidates={result['candidates']}: {result['recall']:.1%}; "
        f"misses={','.join(result['misses']) or '-'}"
    )
print(f"diagnostics: {args.output}")
