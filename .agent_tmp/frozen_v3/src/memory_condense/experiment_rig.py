"""Compile-once, fan-out-many performance rig for associative retrieval.

The expensive embedding and Qwen stages do not belong in parameter workers.
This module consumes a frozen pack of hybrid anchors and evaluates independent
association-budget arms concurrently against one read-mostly artifact store.
Workers never load either model and call ``expand_associative(..., touch=False)``
so sweep order cannot reinforce or prune the graph under test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.association_store import AssociationStore
from memory_condense.associative_retrieval import expand_associative_results
from memory_condense.condenser import MemoryCondenser
from memory_condense.context_packer import ContextBudget, ContextPacker
from memory_condense.db import Database
from memory_condense.heat_diffusion import expand_heat_diffusion_results
from memory_condense.retrieval import hydrate_chunk_result
from memory_condense.schemas import RetrievalResult


@dataclass(frozen=True, slots=True)
class SweepArm:
    name: str
    k: int
    association_slots: int
    qk_reserve: int = 1
    ranked_qk_reserve: int = 0
    neighbors_per_anchor: int = 4
    association_hops: int = 1
    max_association_candidates: int = 64
    cav_candidates: int = 8
    lexical_protection_threshold: float | None = None
    max_prompt_token_increase: int | None = None
    prune_max_neighbors: int | None = None
    retrieval_strategy: str = "ranked"
    diffusion_restart_probability: float = 0.35
    seed_temperature: float = 1.0
    edge_temperature: float = 1.0
    max_source_token_fraction: float = 1.0
    heat_weighted_packing: bool = False
    packing_token_budget: int | None = None
    repeats: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("arm name must be non-empty")
        if self.k < 1:
            raise ValueError("k must be positive")
        if not 0 <= self.association_slots <= self.k:
            raise ValueError("association_slots must lie in [0, k]")
        if self.qk_reserve < 0:
            raise ValueError("qk_reserve must be non-negative")
        if not 0 <= self.ranked_qk_reserve <= self.qk_reserve:
            raise ValueError("ranked_qk_reserve must lie in [0, qk_reserve]")
        if self.neighbors_per_anchor < 0 or self.cav_candidates < 0:
            raise ValueError("candidate limits must be non-negative")
        if self.association_hops < 1 or self.max_association_candidates < 1:
            raise ValueError("hop count and association candidate cap must be positive")
        if self.prune_max_neighbors is not None and self.prune_max_neighbors < 0:
            raise ValueError("prune_max_neighbors must be non-negative")
        if self.lexical_protection_threshold is not None and not (
            0.0 <= self.lexical_protection_threshold <= 1.0
        ):
            raise ValueError("lexical_protection_threshold must lie in [0, 1]")
        if self.max_prompt_token_increase is not None and (
            self.max_prompt_token_increase < 0
        ):
            raise ValueError("max_prompt_token_increase must be non-negative")
        if self.repeats < 1:
            raise ValueError("repeats must be positive")
        if self.retrieval_strategy not in {"ranked", "heat"}:
            raise ValueError("retrieval_strategy must be 'ranked' or 'heat'")
        if not 0.0 <= self.diffusion_restart_probability <= 1.0:
            raise ValueError("diffusion_restart_probability must lie in [0, 1]")
        if self.seed_temperature <= 0.0 or self.edge_temperature <= 0.0:
            raise ValueError("diffusion temperatures must be positive")
        if not 0.0 < self.max_source_token_fraction <= 1.0:
            raise ValueError("max_source_token_fraction must lie in (0, 1]")
        if self.packing_token_budget is not None and self.packing_token_budget < 1:
            raise ValueError("packing_token_budget must be positive")


@dataclass(frozen=True, slots=True)
class SweepQuestion:
    question_id: str
    gold_chunk_ids: tuple[str, ...]
    anchors: tuple[RetrievalResult, ...]
    source_family: str | None = None
    question: str | None = None


def _lean_result(result: RetrievalResult) -> RetrievalResult:
    return result.model_copy(
        update={
            "chunk": result.chunk.model_copy(
                update={"embedding": None, "lexical_weights": None}
            ),
            "turn": None,
        }
    )


def save_anchor_pack(
    path: str | Path,
    questions: Sequence[SweepQuestion],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the immutable handoff between model work and CPU-only sweeps."""
    payload: dict[str, Any] = {
        "format": "memory-condense-anchor-pack-v1",
        "metadata": dict(metadata or {}),
        "questions": [
            {
                "question_id": question.question_id,
                "source_family": question.source_family,
                "question": question.question,
                "gold_chunk_ids": list(question.gold_chunk_ids),
                "anchors": [
                    _lean_result(result).model_dump(mode="json")
                    for result in question.anchors
                ],
            }
            for question in questions
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_anchor_pack(path: str | Path) -> tuple[list[SweepQuestion], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "memory-condense-anchor-pack-v1":
        raise ValueError("unsupported anchor-pack format")
    claimed_hash = payload.get("sha256")
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    actual_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if claimed_hash != actual_hash:
        raise ValueError("anchor-pack hash mismatch")
    questions = [
        SweepQuestion(
            question_id=row["question_id"],
            source_family=row.get("source_family"),
            question=row.get("question"),
            gold_chunk_ids=tuple(row["gold_chunk_ids"]),
            anchors=tuple(
                RetrievalResult.model_validate(result) for result in row["anchors"]
            ),
        )
        for row in payload["questions"]
    ]
    return questions, payload


class AssociativeSweepRig:
    """Run independent CPU-only association arms over cached hybrid anchors."""

    def __init__(
        self,
        store: MemoryCondenser | str | Path,
        artifact_id: str,
        *,
        workers: int | None = None,
    ) -> None:
        if isinstance(store, MemoryCondenser):
            self.db_path = store.database_path
        else:
            candidate = Path(store)
            self.db_path = candidate / "memory.db" if candidate.is_dir() else candidate
        with Database(self.db_path) as db:
            if AssociationStore(db).get_artifact(artifact_id) is None:
                raise KeyError(f"unknown association artifact: {artifact_id}")
        self.artifact_id = artifact_id
        default_workers = min(8, max(1, os.cpu_count() or 1))
        self.workers = default_workers if workers is None else int(workers)
        if self.workers < 1:
            raise ValueError("workers must be positive")

    def _backup_database(self, destination: Path) -> None:
        """Snapshot the prepared SQLite store, including committed WAL pages."""
        source_uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
        # sqlite3.Connection.__exit__ commits or rolls back but deliberately
        # does not close. ``closing`` is required here or Windows keeps the
        # temporary snapshot locked when the arm tries to remove it.
        with closing(sqlite3.connect(source_uri, uri=True)) as source:
            with closing(sqlite3.connect(destination)) as target:
                source.backup(target)

    def _run_arm(
        self,
        arm: SweepArm,
        questions: Sequence[SweepQuestion],
    ) -> dict[str, Any]:
        timings: list[float] = []
        final_rows: list[dict[str, Any]] = []
        temporary: tempfile.TemporaryDirectory[str] | None = None
        arm_db_path = self.db_path
        if arm.prune_max_neighbors is not None:
            temporary = tempfile.TemporaryDirectory(
                prefix="memory-condense-prune-arm-"
            )
            arm_db_path = Path(temporary.name) / "memory.db"
            self._backup_database(arm_db_path)
        try:
            result = self._run_arm_on_database(
                arm,
                questions,
                arm_db_path,
                timings,
                final_rows,
            )
        finally:
            if temporary is not None:
                temporary.cleanup()
        return result

    def _run_arm_on_database(
        self,
        arm: SweepArm,
        questions: Sequence[SweepQuestion],
        database_path: Path,
        timings: list[float],
        final_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        with Database(database_path) as db:
            # The prepared bundle is immutable for the duration of a sweep,
            # so bounded per-worker neighbor caching is safe. Live facade
            # stores leave this cache disabled to observe concurrent updates.
            associations = AssociationStore(db, cache_neighbors=True)
            now_turn = db.current_turn()
            stats_before = associations.stats(self.artifact_id)
            edges_removed = 0
            if arm.prune_max_neighbors is not None:
                edges_removed = associations.prune_edges(
                    self.artifact_id,
                    arm.prune_max_neighbors,
                    now_turn=now_turn,
                )
            stats_after = associations.stats(self.artifact_id)

            def hydrate(chunk_id: str, **kwargs):
                return hydrate_chunk_result(db, chunk_id, **kwargs)

            for _ in range(arm.repeats):
                started = time.perf_counter()
                rows: list[dict[str, Any]] = []
                for question in questions:
                    anchors = list(question.anchors[: arm.k])
                    if arm.retrieval_strategy == "heat":
                        results = expand_heat_diffusion_results(
                            anchors,
                            self.artifact_id,
                            store=associations,
                            hydrate=hydrate,
                            now_turn=now_turn,
                            k=arm.k,
                            association_slots=arm.association_slots,
                            qk_reserve=arm.qk_reserve,
                            ranked_qk_reserve=arm.ranked_qk_reserve,
                            neighbors_per_node=arm.neighbors_per_anchor,
                            diffusion_hops=arm.association_hops,
                            max_diffusion_nodes=arm.max_association_candidates,
                            restart_probability=(
                                arm.diffusion_restart_probability
                            ),
                            seed_temperature=arm.seed_temperature,
                            edge_temperature=arm.edge_temperature,
                            lexical_protection_threshold=(
                                arm.lexical_protection_threshold
                            ),
                            max_prompt_token_increase=(
                                arm.max_prompt_token_increase
                            ),
                            max_source_token_fraction=(
                                arm.max_source_token_fraction
                            ),
                            touch=False,
                        )
                    else:
                        results = expand_associative_results(
                            anchors,
                            self.artifact_id,
                            store=associations,
                            hydrate=hydrate,
                            now_turn=now_turn,
                            k=arm.k,
                            association_slots=arm.association_slots,
                            qk_reserve=arm.qk_reserve,
                            neighbors_per_anchor=arm.neighbors_per_anchor,
                            association_hops=arm.association_hops,
                            max_association_candidates=arm.max_association_candidates,
                            cav_candidates=arm.cav_candidates,
                            lexical_protection_threshold=(
                                arm.lexical_protection_threshold
                            ),
                            max_prompt_token_increase=arm.max_prompt_token_increase,
                            touch=False,
                        )
                    gold = set(question.gold_chunk_ids)
                    baseline_ids = [result.chunk.chunk_id for result in anchors]
                    result_ids = [result.chunk.chunk_id for result in results]
                    routes = [result.route for result in results]
                    packing_budget = arm.packing_token_budget or max(
                        1,
                        sum(result.chunk.token_count for result in anchors),
                    )
                    packer_kwargs = {
                        "expansion_tokens": packing_budget,
                        "max_expansions": arm.k,
                        "max_expansion_tokens": packing_budget,
                    }
                    baseline_packed = ContextPacker(
                        ContextBudget(**packer_kwargs)
                    ).pack(expansions=anchors)
                    linked_packed = ContextPacker(
                        ContextBudget(
                            **packer_kwargs,
                            heat_weighted_expansions=arm.heat_weighted_packing,
                            max_source_expansion_fraction=(
                                arm.max_source_token_fraction
                            ),
                        )
                    ).pack(expansions=results)
                    exposed = linked_packed.expansion_source_token_counts
                    exposed_total = sum(exposed.values())
                    rows.append(
                        {
                            "question_id": question.question_id,
                            "source_family": question.source_family,
                            "baseline_hit": bool(gold.intersection(baseline_ids)),
                            "linked_hit": bool(gold.intersection(result_ids)),
                            "baseline_tokens": sum(
                                result.chunk.token_count for result in anchors
                            ),
                            "linked_tokens": sum(
                                result.chunk.token_count for result in results
                            ),
                            "item_count": len(results),
                            "qk_links": routes.count("qk"),
                            "cav_links": routes.count("cav"),
                            "heat_links": routes.count("heat"),
                            "anchors_displaced": len(
                                set(baseline_ids) - set(result_ids)
                            ),
                            "routes": routes,
                            "association_hops": [
                                result.association_hop
                                for result in results
                                if result.route in {"qk", "heat"}
                            ],
                            "qk_hops": [
                                result.association_hop
                                for result in results
                                if result.route == "qk"
                            ],
                            "heat_hops": [
                                result.association_hop
                                for result in results
                                if result.route == "heat"
                            ],
                            "association_paths": [
                                list(result.association_path or ())
                                for result in results
                                if result.route in {"qk", "heat"}
                            ],
                            "diffusion_heat": [
                                result.diffusion_heat
                                for result in results
                                if result.diffusion_heat is not None
                            ],
                            "baseline_packed_tokens": baseline_packed.token_counts[
                                "expansions"
                            ],
                            "linked_packed_tokens": linked_packed.token_counts[
                                "expansions"
                            ],
                            "source_tokens_exposed": exposed,
                            "sources_exposed": len(exposed),
                            "source_concentration": (
                                max(exposed.values()) / exposed_total
                                if exposed_total
                                else 0.0
                            ),
                        }
                    )
                timings.append(time.perf_counter() - started)
                if final_rows and rows != final_rows:
                    raise RuntimeError(
                        f"arm {arm.name!r} changed across immutable repeats"
                    )
                final_rows = rows
        count = max(1, len(final_rows))
        mean_baseline_tokens = statistics.mean(
            row["baseline_tokens"] for row in final_rows
        )
        mean_linked_tokens = statistics.mean(
            row["linked_tokens"] for row in final_rows
        )
        hop_counts: dict[str, int] = {}
        heat_hop_counts: dict[str, int] = {}
        for row in final_rows:
            for hop in row["qk_hops"]:
                key = str(hop)
                hop_counts[key] = hop_counts.get(key, 0) + 1
            for hop in row["heat_hops"]:
                key = str(hop)
                heat_hop_counts[key] = heat_hop_counts.get(key, 0) + 1
        return {
            "config": asdict(arm),
            "baseline_recall": sum(row["baseline_hit"] for row in final_rows)
            / count,
            "linked_recall": sum(row["linked_hit"] for row in final_rows) / count,
            "mean_baseline_tokens": mean_baseline_tokens,
            "mean_linked_tokens": mean_linked_tokens,
            "mean_token_delta": mean_linked_tokens - mean_baseline_tokens,
            "prompt_token_reduction_fraction": (
                (mean_baseline_tokens - mean_linked_tokens)
                / mean_baseline_tokens
                if mean_baseline_tokens
                else 0.0
            ),
            "recall_changes": {
                "recovered": sum(
                    not row["baseline_hit"] and row["linked_hit"]
                    for row in final_rows
                ),
                "lost": sum(
                    row["baseline_hit"] and not row["linked_hit"]
                    for row in final_rows
                ),
            },
            "qk_links": sum(row["qk_links"] for row in final_rows),
            "cav_links": sum(row["cav_links"] for row in final_rows),
            "heat_links": sum(row["heat_links"] for row in final_rows),
            "qk_hop_counts": hop_counts,
            "heat_hop_counts": heat_hop_counts,
            "mean_baseline_packed_tokens": statistics.mean(
                row["baseline_packed_tokens"] for row in final_rows
            ),
            "mean_linked_packed_tokens": statistics.mean(
                row["linked_packed_tokens"] for row in final_rows
            ),
            "mean_sources_exposed": statistics.mean(
                row["sources_exposed"] for row in final_rows
            ),
            "mean_source_concentration": statistics.mean(
                row["source_concentration"] for row in final_rows
            ),
            "anchors_displaced": sum(
                row["anchors_displaced"] for row in final_rows
            ),
            "pruning": {
                "max_neighbors": arm.prune_max_neighbors,
                "edges_before": stats_before["edges"],
                "edges_removed": edges_removed,
                "edges_after": stats_after["edges"],
                "head_payload_bytes_before": stats_before["head_payload_bytes"],
                "head_payload_bytes_after": stats_after["head_payload_bytes"],
                "retained_request_token_state_bytes": stats_after[
                    "retained_request_token_state_bytes"
                ],
                # Compatibility alias for historical sweep artifacts.
                "retained_token_state_bytes": stats_after[
                    "retained_token_state_bytes"
                ],
            },
            "elapsed_samples_s": timings,
            "median_elapsed_s": statistics.median(timings),
            "rows": final_rows,
        }

    def run(
        self,
        questions: Sequence[SweepQuestion],
        arms: Sequence[SweepArm],
    ) -> dict[str, Any]:
        if not questions:
            raise ValueError("at least one sweep question is required")
        if not arms:
            raise ValueError("at least one sweep arm is required")
        names = [arm.name for arm in arms]
        if len(set(names)) != len(names):
            raise ValueError("sweep arm names must be unique")
        started = time.perf_counter()
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(self.workers, len(arms))) as executor:
            futures = {
                executor.submit(self._run_arm, arm, questions): arm.name
                for arm in arms
            }
            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()
        with Database(self.db_path) as db:
            artifact_stats = AssociationStore(db).stats(self.artifact_id)
        return {
            "format": "memory-condense-association-sweep-v1",
            "artifact_id": self.artifact_id,
            "execution": {
                "workers": min(self.workers, len(arms)),
                "cpu_count": os.cpu_count(),
                "qwen_workers": 0,
                "embedding_workers": 0,
                "touch": False,
                "parallel_wall_seconds": time.perf_counter() - started,
            },
            "artifact_stats": artifact_stats,
            "question_count": len(questions),
            "arms": {name: results[name] for name in names},
        }


def _load_arms(path: str | Path) -> list[SweepArm]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["arms"] if isinstance(payload, dict) else payload
    return [SweepArm(**row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--anchor-pack", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    questions, anchor_payload = load_anchor_pack(args.anchor_pack)
    arms = _load_arms(args.arms)
    report = AssociativeSweepRig(
        args.store,
        args.artifact_id,
        workers=args.workers,
    ).run(questions, arms)
    report["anchor_pack_sha256"] = anchor_payload["sha256"]
    arms_bytes = args.arms.read_bytes()
    arms_payload = json.loads(arms_bytes)
    report["arms_sha256"] = hashlib.sha256(arms_bytes).hexdigest()
    report["arms_protocol"] = (
        arms_payload.get("protocol") if isinstance(arms_payload, dict) else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"parallel wall: {report['execution']['parallel_wall_seconds']:.3f}s")
    for name, arm in report["arms"].items():
        print(
            f"{name}: recall={arm['linked_recall']:.1%} "
            f"tokens={arm['mean_linked_tokens']:.1f} "
            f"median={arm['median_elapsed_s']:.3f}s"
        )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
