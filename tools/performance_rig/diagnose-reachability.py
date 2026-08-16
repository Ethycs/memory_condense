"""Diagnose whether gold chunks are reachable through persisted QK/CAV links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_condense.association_store import AssociationStore
from memory_condense.db import Database
from memory_condense.experiment_rig import load_anchor_pack


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--store", type=Path, required=True)
parser.add_argument("--artifact-id", required=True)
parser.add_argument("--anchor-pack", type=Path, required=True)
parser.add_argument("--max-hops", type=int, default=3)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

questions, pack = load_anchor_pack(args.anchor_pack)
db_path = args.store / "memory.db" if args.store.is_dir() else args.store
rows = []
with Database(db_path) as db:
    store = AssociationStore(db, cache_neighbors=True)
    signature_count = store.stats(args.artifact_id)["signatures"]
    for question in questions:
        gold = set(question.gold_chunk_ids)
        by_budget = {}
        for k in (3, 5, 10):
            anchors = list(question.anchors[:k])
            anchor_ids = [anchor.chunk.chunk_id for anchor in anchors]
            direct_ranks = [
                rank
                for rank, chunk_id in enumerate(anchor_ids, start=1)
                if chunk_id in gold
            ]
            seen = set(anchor_ids)
            frontier = list(anchor_ids)
            qk_gold_hop = None
            qk_reached = 0
            for hop in range(1, args.max_hops + 1):
                groups = store.neighbors_many(
                    frontier,
                    args.artifact_id,
                    top_k_per_source=100,
                    exclude=tuple(seen),
                    now_turn=db.current_turn(),
                )
                next_frontier = []
                for source_id in frontier:
                    for edge in groups[source_id]:
                        destination = edge.destination_chunk_id
                        if destination in seen:
                            continue
                        seen.add(destination)
                        next_frontier.append(destination)
                        if destination in gold and qk_gold_hop is None:
                            qk_gold_hop = hop
                qk_reached += len(next_frontier)
                frontier = next_frontier
                if not frontier:
                    break
            cav = store.cav_neighbors(
                anchor_ids,
                args.artifact_id,
                top_k=signature_count,
                exclude=anchor_ids,
            )
            cav_gold_ranks = [
                rank
                for rank, hit in enumerate(cav, start=1)
                if hit.chunk_id in gold
            ]
            by_budget[f"k{k}"] = {
                "direct_rank": min(direct_ranks, default=None),
                "qk_gold_hop": qk_gold_hop,
                "qk_chunks_reached": qk_reached,
                "cav_gold_rank": min(cav_gold_ranks, default=None),
                "cav_candidates": len(cav),
            }
        rows.append(
            {
                "question_id": question.question_id,
                "source_family": question.source_family,
                "gold_chunk_ids": list(question.gold_chunk_ids),
                "budgets": by_budget,
            }
        )

report = {
    "format": "memory-condense-reachability-v1",
    "anchor_pack_sha256": pack["sha256"],
    "artifact_id": args.artifact_id,
    "max_hops": args.max_hops,
    "rows": rows,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
for row in rows:
    summary = ", ".join(
        f"{budget}:direct={values['direct_rank']} "
        f"qk-hop={values['qk_gold_hop']} cav-rank={values['cav_gold_rank']}"
        for budget, values in row["budgets"].items()
    )
    print(f"{row['question_id']}: {summary}")
print(f"report: {args.output}")
