"""Gate Qwen prefix residuals as a reranker on B0's existing misses.

This is explicitly a development diagnostic.  Dense/BM25 candidate generation
is unchanged; Qwen only sees their union.  The purpose is to decide whether a
more expensive live-QK reranker is justified on this real corpus.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np

from memory_condense import ranking
from memory_condense._tokenizer import count_tokens
from memory_condense.condenser import MemoryCondenser
from memory_condense.eval.recall import contains_answer
from memory_condense.qwen_prefix import Qwen3PrefixEncoder


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
    "--model-dir", type=Path, default=ROOT / ".cache" / "models" / "Qwen3-8B"
)
parser.add_argument("--question-ids", nargs="*", default=["q0", "q13", "q38"])
parser.add_argument(
    "--all-questions",
    action="store_true",
    help="Run the development reranker over all 39 B0 questions.",
)
parser.add_argument("--candidates", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "eval_results" / "build_session_b0_qwen_rerank_gate.json",
)
args = parser.parse_args()

probe = json.loads(args.probe.read_text(encoding="utf-8"))
selected = [
    {
        "question_id": f"q{i}",
        "question": question,
        "answer": answer,
        "category": category,
    }
    for i, (question, answer, category, _) in enumerate(probe)
    if args.all_questions or f"q{i}" in set(args.question_ids)
]

mc = MemoryCondenser(data_dir=args.store, auto_extract=False, device="cpu")
try:
    stored_rows = mc._db.execute(
        "SELECT chunk_id, rowid, text, token_count FROM chunks "
        "WHERE embedding IS NOT NULL AND hnsw_label IS NOT NULL ORDER BY rowid"
    ).fetchall()
    text_by_id = {row[0]: row[2] for row in stored_rows}
    tokens_by_id = {row[0]: int(row[3]) for row in stored_rows}
    rowid_by_id = {row[0]: int(row[1]) for row in stored_rows}
    id_by_rowid = {int(row[1]): row[0] for row in stored_rows}
    candidate_sets: list[dict] = []
    all_candidate_ids: set[str] = set()
    for question in selected:
        query_embedding = mc._embedder.embed_query(question["question"])
        dense = mc._retriever._dense_candidates(
            query_embedding, args.candidates, args.candidates
        )
        lexical = mc._retriever.lexical.search(
            question["question"], limit=args.candidates
        )
        dense_normalized = ranking.min_max_normalize([score for _, score in dense])
        lexical_normalized = ranking.min_max_normalize(
            [score for _, score in lexical]
        )
        dense_scores = {
            chunk_id: score
            for (chunk_id, _), score in zip(dense, dense_normalized)
        }
        lexical_scores = {
            chunk_id: score
            for (chunk_id, _), score in zip(lexical, lexical_normalized)
        }
        ordered_ids = [chunk_id for chunk_id, _ in dense]
        seen = set(ordered_ids)
        for chunk_id, _ in lexical:
            if chunk_id not in seen:
                seen.add(chunk_id)
                ordered_ids.append(chunk_id)
        current_scores = {
            chunk_id: ranking.blend_hybrid(
                dense_scores.get(chunk_id, 0.0),
                lexical_scores.get(chunk_id, 0.0),
                0.65,
            )
            for chunk_id in ordered_ids
        }
        current_order = sorted(
            ordered_ids, key=lambda chunk_id: -current_scores[chunk_id]
        )
        association_ids = list(ordered_ids)
        association_seen = set(association_ids)
        neighbor_ids: list[str] = []
        for seed_id in current_order[:10]:
            seed_rowid = rowid_by_id[seed_id]
            for delta in (-2, -1, 1, 2):
                neighbor_id = id_by_rowid.get(seed_rowid + delta)
                if neighbor_id is not None and neighbor_id not in association_seen:
                    association_seen.add(neighbor_id)
                    association_ids.append(neighbor_id)
                    neighbor_ids.append(neighbor_id)
        candidate_sets.append(
            {
                **question,
                "ids": ordered_ids,
                "association_ids": association_ids,
                "neighbor_ids": neighbor_ids,
                "current_scores": current_scores,
            }
        )
        all_candidate_ids.update(association_ids)
finally:
    mc.close()
    del mc
    gc.collect()

candidate_ids = sorted(all_candidate_ids)
texts = [question["question"] for question in selected] + [
    text_by_id[chunk_id] for chunk_id in candidate_ids
]
encoder = Qwen3PrefixEncoder(
    args.model_dir,
    layers=7,
    device="cuda",
    dtype="bfloat16",
)
vectors = encoder.encode_layers(texts, layers=(1, 5), batch_size=args.batch_size)


def normalized_cosines(query_vector, candidate_vectors) -> np.ndarray:
    query = query_vector.numpy()
    candidates = candidate_vectors.numpy()
    numerator = candidates @ query
    denominator = np.linalg.norm(candidates, axis=1) * max(
        float(np.linalg.norm(query)), 1e-12
    )
    cosine = numerator / np.maximum(denominator, 1e-12)
    normalized = ranking.min_max_normalize(cosine.tolist())
    return np.asarray(normalized, dtype=np.float32)


candidate_offset = len(selected)
candidate_position = {chunk_id: i for i, chunk_id in enumerate(candidate_ids)}
methods = (
    "current_hybrid",
    "dedupe_raw_top_10",
    "dedupe_then_qwen_5",
    "qwen_layer_1",
    "qwen_layer_5",
    "hybrid_50_qwen_5_50",
    "hybrid_35_qwen_5_65",
)
report_rows = []
for query_index, candidate_set in enumerate(candidate_sets):
    ids = candidate_set["ids"]
    association_ids = candidate_set["association_ids"]
    positions = [candidate_position[chunk_id] for chunk_id in association_ids]
    qwen_scores: dict[int, dict[str, float]] = {}
    for layer in (1, 5):
        candidate_vectors = vectors[layer][
            [candidate_offset + position for position in positions]
        ]
        scores = normalized_cosines(vectors[layer][query_index], candidate_vectors)
        qwen_scores[layer] = {
            chunk_id: float(score)
            for chunk_id, score in zip(association_ids, scores)
        }

    score_maps = {
        "current_hybrid": candidate_set["current_scores"],
        "qwen_layer_1": qwen_scores[1],
        "qwen_layer_5": qwen_scores[5],
        "hybrid_50_qwen_5_50": {
            chunk_id: 0.5 * candidate_set["current_scores"][chunk_id]
            + 0.5 * qwen_scores[5][chunk_id]
            for chunk_id in ids
        },
        "hybrid_35_qwen_5_65": {
            chunk_id: 0.35 * candidate_set["current_scores"][chunk_id]
            + 0.65 * qwen_scores[5][chunk_id]
            for chunk_id in ids
        },
    }
    current_order = sorted(
        ids, key=lambda chunk_id: -candidate_set["current_scores"][chunk_id]
    )

    def content_key(chunk_id: str) -> str:
        return re.sub(r"\s+", " ", text_by_id[chunk_id]).strip().casefold()

    deduped: list[str] = []
    seen_content: set[str] = set()
    for chunk_id in current_order[:10]:
        key = content_key(chunk_id)
        if key not in seen_content:
            seen_content.add(key)
            deduped.append(chunk_id)

    qwen_order = sorted(
        association_ids, key=lambda chunk_id: -qwen_scores[5][chunk_id]
    )
    recycled = list(deduped)
    neighbor_order = sorted(
        candidate_set["neighbor_ids"],
        key=lambda chunk_id: -qwen_scores[5][chunk_id],
    )
    if len(recycled) < 10:
        for chunk_id in neighbor_order:
            key = content_key(chunk_id)
            if key not in seen_content:
                seen_content.add(key)
                recycled.append(chunk_id)
                break
    for chunk_id in qwen_order:
        if len(recycled) >= 10:
            break
        key = content_key(chunk_id)
        if key not in seen_content:
            seen_content.add(key)
            recycled.append(chunk_id)

    orders = {
        "current_hybrid": current_order,
        "dedupe_raw_top_10": deduped,
        "dedupe_then_qwen_5": recycled,
        "qwen_layer_1": sorted(
            association_ids, key=lambda chunk_id: -qwen_scores[1][chunk_id]
        ),
        "qwen_layer_5": qwen_order,
        "hybrid_50_qwen_5_50": sorted(
            ids, key=lambda chunk_id: -score_maps["hybrid_50_qwen_5_50"][chunk_id]
        ),
        "hybrid_35_qwen_5_65": sorted(
            ids, key=lambda chunk_id: -score_maps["hybrid_35_qwen_5_65"][chunk_id]
        ),
    }
    method_rows = {}
    for method in methods:
        ordered = orders[method]
        top_10 = ordered[:10]
        gold_ranks = [
            rank
            for rank, chunk_id in enumerate(ordered, 1)
            if contains_answer([text_by_id[chunk_id]], candidate_set["answer"])
        ]
        method_rows[method] = {
            "hit": any(rank <= 10 for rank in gold_ranks),
            "best_gold_rank": min(gold_ranks, default=None),
            "context_tokens": sum(tokens_by_id[chunk_id] for chunk_id in top_10),
            "item_count": len(top_10),
            "top_10": [
                {
                    "chunk_id": chunk_id,
                    "score": (
                        score_maps[method][chunk_id]
                        if method in score_maps and chunk_id in score_maps[method]
                        else qwen_scores[5].get(chunk_id)
                    ),
                    "answer_hit": contains_answer(
                        [text_by_id[chunk_id]], candidate_set["answer"]
                    ),
                    "text_excerpt": text_by_id[chunk_id][:300],
                }
                for chunk_id in top_10
            ],
        }
    report_rows.append(
        {
            "question_id": candidate_set["question_id"],
            "question": candidate_set["question"],
            "answer": candidate_set["answer"],
            "candidate_count": len(ids),
            "methods": method_rows,
        }
    )

summary = {
    method: {
        "recall": sum(row["methods"][method]["hit"] for row in report_rows)
        / len(report_rows),
        "mean_context_tokens": sum(
            row["methods"][method]["context_tokens"] for row in report_rows
        )
        / len(report_rows),
        "mean_item_count": sum(
            row["methods"][method]["item_count"] for row in report_rows
        )
        / len(report_rows),
    }
    for method in methods
}
report = {
    "protocol": (
        "B0 development rerank; all 39 questions"
        if args.all_questions
        else "B0 development rerank gate; three known misses only"
    ),
    "model": "Qwen/Qwen3-8B BF16 prefix layers 0-6",
    "candidate_generator": f"union(dense top-{args.candidates}, BM25 top-{args.candidates})",
    "summary": summary,
    "questions": report_rows,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

for method, metrics in summary.items():
    ranks = [row["methods"][method]["best_gold_rank"] for row in report_rows]
    print(
        f"{method:<26} recall={metrics['recall']:.1%} "
        f"mean_tokens={metrics['mean_context_tokens']:.0f} "
        f"mean_items={metrics['mean_item_count']:.1f} ranks={ranks}"
    )
print(f"diagnostics: {args.output}")
