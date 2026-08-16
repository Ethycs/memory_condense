"""Test actual QK attention on the temporal neighbor subgraph of B0 misses."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from memory_condense.condenser import MemoryCondenser
from memory_condense.eval.recall import contains_answer
from memory_condense.head_memory import QwenLiveHeadMemory
from memory_condense.qwen_prefix import Qwen3PrefixEncoder


ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--store",
    type=Path,
    default=ROOT / "data" / "build-session-8f7f7561.store",
)
parser.add_argument(
    "--rerank-report",
    type=Path,
    default=ROOT
    / "eval_results"
    / "build_session_b0_qwen_recycled_neighbor_gate.json",
)
parser.add_argument("--probe", type=Path, default=BASE / "cc_probe.json")
parser.add_argument(
    "--model-dir", type=Path, default=ROOT / ".cache" / "models" / "Qwen3-8B"
)
parser.add_argument("--question-ids", nargs="*", default=["q0", "q38"])
parser.add_argument("--all-questions", action="store_true")
parser.add_argument("--layers", nargs="+", type=int, default=[1, 5])
parser.add_argument("--radius", type=int, default=2)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "eval_results" / "build_session_b0_qk_neighbor_gate.json",
)
args = parser.parse_args()

probe = json.loads(args.probe.read_text(encoding="utf-8"))
questions = {
    f"q{i}": {"question": question, "answer": answer}
    for i, (question, answer, _, _) in enumerate(probe)
}
rerank = json.loads(args.rerank_report.read_text(encoding="utf-8"))
report_by_id = {row["question_id"]: row for row in rerank["questions"]}

mc = MemoryCondenser(data_dir=args.store, auto_extract=False, device="cpu")
try:
    rows = mc._db.execute(
        "SELECT chunk_id, rowid, text, token_count FROM chunks "
        "WHERE embedding IS NOT NULL AND hnsw_label IS NOT NULL ORDER BY rowid"
    ).fetchall()
finally:
    mc.close()

row_by_id = {row[0]: int(row[1]) for row in rows}
by_row = {int(row[1]): row for row in rows}
candidate_sets = []
selected_ids = list(report_by_id) if args.all_questions else args.question_ids
for question_id in selected_ids:
    source = report_by_id[question_id]
    seed_ids = [
        result["chunk_id"]
        for result in source["methods"]["current_hybrid"]["top_10"]
    ]
    neighbor_rows = []
    seen_content: set[str] = set()
    for seed_id in seed_ids:
        seed_row = row_by_id[seed_id]
        for distance in range(1, args.radius + 1):
            for neighbor_rowid in (seed_row - distance, seed_row + distance):
                row = by_row.get(neighbor_rowid)
                if row is None:
                    continue
                key = re.sub(r"\s+", " ", row[2]).strip().casefold()
                if key in seen_content:
                    continue
                seen_content.add(key)
                neighbor_rows.append(row)
    candidate_sets.append(
        {
            "question_id": question_id,
            **questions[question_id],
            "neighbors": neighbor_rows,
        }
    )

encoder = Qwen3PrefixEncoder(
    args.model_dir,
    layers=7,
    device="cuda",
    dtype="bfloat16",
)
results = []
for candidate_set in candidate_sets:
    layer_results = {}
    for layer in args.layers:
        memory = QwenLiveHeadMemory(
            encoder,
            layer=layer,
            association_layer=layer,
            association_candidates=0,
        )
        for chunk_id, rowid, text, token_count in candidate_set["neighbors"]:
            memory.write(
                chunk_id,
                text,
                metadata={"rowid": int(rowid), "token_count": int(token_count)},
            )
        retrieved = memory.retrieve(
            candidate_set["question"],
            top_k=len(candidate_set["neighbors"]),
            hops=1,
        )
        ranked = []
        for rank, hit in enumerate(retrieved.hits, 1):
            item = next(
                item for item in memory.store.items if item.episode_id == hit.episode_id
            )
            ranked.append(
                {
                    "rank": rank,
                    "chunk_id": hit.episode_id,
                    "score": hit.score,
                    "answer_hit": contains_answer(
                        [hit.text], candidate_set["answer"]
                    ),
                    "qk_attention_mass": item.qk_attention_mass,
                    "ov_transport": item.ov_transport,
                    "rowid": hit.metadata["rowid"],
                    "token_count": hit.metadata["token_count"],
                    "text_excerpt": hit.text[:300],
                }
            )
        layer_results[str(layer)] = {
            "best_gold_rank": min(
                (row["rank"] for row in ranked if row["answer_hit"]), default=None
            ),
            "ranked_neighbors": ranked,
        }
    results.append(
        {
            "question_id": candidate_set["question_id"],
            "question": candidate_set["question"],
            "answer": candidate_set["answer"],
            "neighbor_count": len(candidate_set["neighbors"]),
            "layers": layer_results,
        }
    )

report = {
    "protocol": "B0 development gate; actual QK over +/-2 row neighbor subgraphs",
    "model": "Qwen/Qwen3-8B BF16 prefix layers 0-6",
    "questions": results,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
for result in results:
    ranks = " ".join(
        f"layer{layer}_gold={result['layers'][str(layer)]['best_gold_rank']}"
        for layer in args.layers
    )
    print(
        f"{result['question_id']}: neighbors={result['neighbor_count']} "
        f"{ranks}"
    )
print(f"diagnostics: {args.output}")
