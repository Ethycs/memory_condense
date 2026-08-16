"""Compose B0 hybrid anchors, live-QK neighbors, and layer-5 associations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_condense.condenser import MemoryCondenser
from memory_condense.eval.recall import contains_answer
from memory_condense.head_memory import (
    AssociativeMemoryCandidate,
    compose_associative_candidates,
)


ROOT = Path(__file__).resolve().parents[4]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--store",
    type=Path,
    default=ROOT / "data" / "build-session-8f7f7561.store",
)
parser.add_argument(
    "--rerank-report",
    type=Path,
    default=ROOT / "eval_results" / "build_session_b0_qwen_rerank_all.json",
)
parser.add_argument(
    "--qk-report",
    type=Path,
    default=ROOT / "eval_results" / "build_session_b0_qk_neighbor_all.json",
)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "eval_results" / "build_session_b0_live_composed.json",
)
args = parser.parse_args()

rerank = json.loads(args.rerank_report.read_text(encoding="utf-8"))
qk = json.loads(args.qk_report.read_text(encoding="utf-8"))
qk_by_id = {row["question_id"]: row for row in qk["questions"]}

mc = MemoryCondenser(data_dir=args.store, auto_extract=False, device="cpu")
try:
    rows = mc._db.execute(
        "SELECT chunk_id, text, token_count FROM chunks WHERE embedding IS NOT NULL"
    ).fetchall()
finally:
    mc.close()
text_by_id = {row[0]: row[1] for row in rows}
tokens_by_id = {row[0]: int(row[2]) for row in rows}


def candidates(results: list[dict], route: str) -> list[AssociativeMemoryCandidate]:
    return [
        AssociativeMemoryCandidate(
            episode_id=result["chunk_id"],
            text=text_by_id[result["chunk_id"]],
            score=float(result.get("score", 0.0)),
            route=route,
            metadata=dict(result),
        )
        for result in results
    ]


composed_rows = []
for row in rerank["questions"]:
    baseline = row["methods"]["current_hybrid"]["top_10"]
    qwen = row["methods"]["qwen_layer_5"]["top_10"]
    qk_neighbors = qk_by_id[row["question_id"]]["layers"]["1"][
        "ranked_neighbors"
    ]

    composition = compose_associative_candidates(
        candidates(baseline, "hybrid"),
        qk_neighbors=candidates(qk_neighbors, "qk_neighbor"),
        residual_candidates=candidates(qwen, "layer5_residual"),
        top_k=10,
        qk_reserve=1,
    )
    selected = [
        {**candidate.metadata, "route": candidate.route}
        for candidate in composition.candidates
    ]

    hit_ranks = [
        rank
        for rank, result in enumerate(selected, 1)
        if contains_answer([text_by_id[result["chunk_id"]]], row["answer"])
    ]
    composed_rows.append(
        {
            "question_id": row["question_id"],
            "question": row["question"],
            "answer": row["answer"],
            "hit": bool(hit_ranks),
            "answer_rank": min(hit_ranks, default=None),
            "context_tokens": sum(
                tokens_by_id[result["chunk_id"]] for result in selected
            ),
            "item_count": len(selected),
            "duplicates_removed": composition.duplicates_removed,
            "qk_neighbors_added": composition.qk_added,
            "layer5_candidates_added": composition.residual_added,
            "selected": selected,
        }
    )

baseline_metrics = rerank["summary"]["current_hybrid"]
metrics = {
    "recall": sum(row["hit"] for row in composed_rows) / len(composed_rows),
    "mean_context_tokens": sum(row["context_tokens"] for row in composed_rows)
    / len(composed_rows),
    "mean_item_count": sum(row["item_count"] for row in composed_rows)
    / len(composed_rows),
    "duplicates_removed": sum(row["duplicates_removed"] for row in composed_rows),
    "qk_neighbors_added": sum(row["qk_neighbors_added"] for row in composed_rows),
    "layer5_candidates_added": sum(
        row["layer5_candidates_added"] for row in composed_rows
    ),
}
report = {
    "protocol": "B0 development composition; hard cap 10; no unique hybrid anchor removed",
    "baseline": baseline_metrics,
    "composed": metrics,
    "rows": composed_rows,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"baseline recall: {baseline_metrics['recall']:.1%}")
print(f"baseline mean tokens: {baseline_metrics['mean_context_tokens']:.0f}")
print(f"composed recall: {metrics['recall']:.1%}")
print(f"composed mean tokens: {metrics['mean_context_tokens']:.0f}")
print(f"composed mean items: {metrics['mean_item_count']:.1f}")
print(f"duplicate slots recycled: {metrics['duplicates_removed']}")
print(f"QK neighbors added: {metrics['qk_neighbors_added']}")
print(f"layer-5 candidates added: {metrics['layer5_candidates_added']}")
print(
    "misses: "
    + ",".join(row["question_id"] for row in composed_rows if not row["hit"])
)
print(f"diagnostics: {args.output}")
