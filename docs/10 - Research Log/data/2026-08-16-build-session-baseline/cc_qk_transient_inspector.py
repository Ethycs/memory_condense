"""Check B0 hard misses with a capped, no-storage Qwen head inspector."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from memory_condense.condenser import MemoryCondenser
from memory_condense.eval.recall import contains_answer
from memory_condense.head_memory import (
    AssociativeMemoryCandidate,
    QwenMemoryLinker,
)
from memory_condense.qwen_prefix import Qwen3PrefixEncoder


ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--store", type=Path, default=ROOT / "data" / "build-session-8f7f7561.store"
)
parser.add_argument(
    "--rerank-report",
    type=Path,
    default=ROOT / "eval_results" / "build_session_b0_qwen_rerank_all.json",
)
parser.add_argument("--probe", type=Path, default=BASE / "cc_probe.json")
parser.add_argument(
    "--model-dir", type=Path, default=ROOT / ".cache" / "models" / "Qwen3-8B"
)
parser.add_argument("--question-ids", nargs="*", default=["q0", "q38"])
parser.add_argument("--radius", type=int, default=2)
parser.add_argument("--anchor-groups", type=int, default=4)
parser.add_argument("--max-candidates", type=int, default=4)
parser.add_argument("--workspace-tokens", type=int, default=1152)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "eval_results" / "build_session_b0_transient_inspector.json",
)
args = parser.parse_args()

probe = json.loads(args.probe.read_text(encoding="utf-8"))
questions = {
    f"q{index}": {"question": row[0], "answer": row[1]}
    for index, row in enumerate(probe)
}
rerank = json.loads(args.rerank_report.read_text(encoding="utf-8"))
rerank_by_id = {row["question_id"]: row for row in rerank["questions"]}

condenser = MemoryCondenser(data_dir=args.store, auto_extract=False, device="cpu")
try:
    rows = condenser._db.execute(
        "SELECT chunk_id, rowid, text, token_count FROM chunks "
        "WHERE embedding IS NOT NULL AND hnsw_label IS NOT NULL ORDER BY rowid"
    ).fetchall()
finally:
    condenser.close()
row_by_id = {row[0]: int(row[1]) for row in rows}
by_row = {int(row[1]): row for row in rows}

candidate_sets = []
for question_id in args.question_ids:
    source = rerank_by_id[question_id]
    seed_ids = [
        result["chunk_id"] for result in source["methods"]["current_hybrid"]["top_10"]
    ]
    neighbor_groups = []
    for seed_id in seed_ids[: args.anchor_groups]:
        neighbors = []
        seen_content: set[str] = set()
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
                neighbors.append(row)
        neighbor_groups.append(neighbors)
    candidate_sets.append((question_id, neighbor_groups))

encoder = Qwen3PrefixEncoder(
    args.model_dir,
    layers=7,
    device="cuda",
    dtype="bfloat16",
)
inspector = QwenMemoryLinker(
    encoder,
    layer=1,
    max_candidates=args.max_candidates,
    max_workspace_tokens=args.workspace_tokens,
)
results = []
for question_id, neighbor_groups in candidate_sets:
    question = questions[question_id]
    candidate_groups = [
        [
            AssociativeMemoryCandidate(
                episode_id=row[0],
                text=row[2],
                route="local_neighbor",
                metadata={"rowid": int(row[1]), "token_count": int(row[3])},
            )
            for row in neighbors[: args.max_candidates]
        ]
        for neighbors in neighbor_groups
    ]
    text_by_candidate = {
        candidate.episode_id: candidate.text
        for group in candidate_groups
        for candidate in group
    }
    inspected = inspector.inspect_nested(
        question["question"],
        candidate_groups,
        beam_per_group=2,
        top_k=args.max_candidates,
    )
    ranked = []
    for rank, hit in enumerate(inspected.hits, start=1):
        ranked.append(
            {
                "rank": rank,
                "chunk_id": hit.episode_id,
                "rowid": hit.metadata["rowid"],
                "token_count": hit.metadata["token_count"],
                "qk_score": hit.qk_score,
                "ov_transport": hit.ov_transport,
                "answer_hit": contains_answer(
                    [text_by_candidate[hit.episode_id]], question["answer"]
                ),
            }
        )
    results.append(
        {
            "question_id": question_id,
            "passes": inspected.passes,
            "total_candidate_inspections": inspected.total_candidate_inspections,
            "max_candidates_per_pass": inspected.max_workspace_candidates,
            "max_workspace_tokens": inspected.max_workspace_tokens,
            "gold_present_in_workspace": any(row["answer_hit"] for row in ranked),
            "best_gold_rank": min(
                (row["rank"] for row in ranked if row["answer_hit"]), default=None
            ),
            "ranked": ranked,
        }
    )

report = {
    "protocol": (
        "B0 development hard-miss check; one fresh bounded local-neighbor "
        "inspection per direct anchor; no context accumulation or retained token K/V"
    ),
    "anchor_groups": args.anchor_groups,
    "max_candidates": args.max_candidates,
    "max_workspace_tokens": args.workspace_tokens,
    "retained_token_kv_bytes": 0,
    "questions": results,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
for result in results:
    print(
        f"{result['question_id']}: passes={result['passes']} "
        f"max_candidates={result['max_candidates_per_pass']} "
        f"max_tokens={result['max_workspace_tokens']} "
        f"gold_present={result['gold_present_in_workspace']} "
        f"gold_rank={result['best_gold_rank']}"
    )
print(f"report: {args.output}")
