"""Leakage-safe next-turn replay over a compiled QK/CAV memory graph.

For turn ``t`` the replay exposes only edges from that turn to chunks already
written before it.  The target is the strongest history edge produced when
turn ``t + 1`` was later inspected.  The policy ranks first, the next-turn
target is revealed second, and only then may scalar statistics update.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field

from memory_condense.transition_policy import (
    CausalTransitionPolicy,
    TransitionCandidate,
)


@dataclass(frozen=True, slots=True)
class TransitionReplayExample:
    source_group: str
    source_turn_id: str
    from_role: str
    next_role: str
    source_cav: tuple[float, ...]
    cav_velocity: tuple[float, ...]
    next_cav: tuple[float, ...]
    candidates: tuple[TransitionCandidate, ...]
    actual_destination_id: str


class TransitionReplayRow(BaseModel):
    source_group: str
    source_turn_id: str
    target_in_candidates: bool
    baseline_rank: int | None = None
    learned_rank: int | None = None
    baseline_delta_cosine: float = 0.0
    learned_delta_cosine: float = 0.0


class TransitionReplayReport(BaseModel):
    artifact_id: str = ""
    train_source_groups: list[str] = Field(default_factory=list)
    evaluation_source_groups: list[str] = Field(default_factory=list)
    training_transitions: int = 0
    evaluation_transitions: int = 0
    target_candidate_coverage: float = 0.0
    baseline_recall_at_1: float = 0.0
    learned_recall_at_1: float = 0.0
    baseline_mrr: float = 0.0
    learned_mrr: float = 0.0
    baseline_mean_delta_cosine: float = 0.0
    learned_mean_delta_cosine: float = 0.0
    improved: bool = False
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    rows: list[TransitionReplayRow] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _TurnRecord:
    turn_id: str
    role: str
    ordinal: int
    source_id: str | None
    chunk_ids: tuple[str, ...]
    cav: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _EdgeRecord:
    destination_id: str
    head_weights: tuple[float, ...]
    qk_score: float


def _unpack_f32(blob: bytes | None) -> tuple[float, ...]:
    if not blob:
        return ()
    if len(blob) % 4:
        raise ValueError("float32 association blob has an invalid byte length")
    return tuple(struct.unpack(f"<{len(blob) // 4}f", blob))


def _mean_vectors(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        return ()
    width = len(vectors[0])
    if width == 0 or any(len(vector) != width for vector in vectors):
        raise ValueError("CAV signatures must be non-empty and fixed-width")
    return tuple(
        sum(float(vector[index]) for vector in vectors) / len(vectors)
        for index in range(width)
    )


def _delta(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    if len(left) != len(right):
        raise ValueError("CAV vectors need equal dimensions")
    return tuple(float(b) - float(a) for a, b in zip(left, right, strict=True))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("CAV deltas need equal dimensions")
    numerator = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def _read_only_connection(path: str | Path) -> sqlite3.Connection:
    resolved = Path(path).resolve()
    return sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)


def _source_groups(
    turns: Sequence[tuple[str, str, int, str | None]],
    legacy_source_blocks: Sequence[tuple[str, int]],
) -> dict[str, list[str]]:
    if any(source_id for _, _, _, source_id in turns):
        groups: dict[str, list[str]] = {}
        for turn_id, _role, _ordinal, source_id in turns:
            key = source_id or turn_id
            groups.setdefault(key, []).append(turn_id)
        return groups

    blocks = list(legacy_source_blocks)
    if not blocks:
        raise ValueError(
            "legacy compiled stores need source blocks to prevent replay "
            "across conversation boundaries"
        )
    if sum(count for _, count in blocks) != len(turns):
        raise ValueError("legacy source-block counts do not match stored turns")
    groups: dict[str, list[str]] = {}
    cursor = 0
    for source_id, count in blocks:
        if not source_id or count < 1:
            raise ValueError("source blocks need non-empty IDs and positive counts")
        groups[source_id] = [turn[0] for turn in turns[cursor : cursor + count]]
        cursor += count
    return groups


def load_compiled_transition_examples(
    database_path: str | Path,
    *,
    artifact_id: str | None = None,
    legacy_source_blocks: Sequence[tuple[str, int]] = (),
    max_candidates: int = 8,
) -> tuple[str, list[TransitionReplayExample]]:
    """Build turn-boundary examples from an immutable compiled SQLite store."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    with _read_only_connection(database_path) as connection:
        if artifact_id is None:
            row = connection.execute(
                "SELECT artifact_id FROM association_artifacts "
                "ORDER BY created_at DESC, artifact_id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise ValueError("compiled store has no association artifact")
            artifact_id = str(row[0])

        turn_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(turns)").fetchall()
        }
        source_expression = "source_id" if "source_id" in turn_columns else "NULL"
        turns = [
            (str(row[0]), str(row[1]), int(row[2]), row[3])
            for row in connection.execute(
                f"SELECT turn_id, role, ordinal, {source_expression} FROM turns "
                "ORDER BY ordinal, rowid"
            ).fetchall()
        ]
        groups = _source_groups(turns, legacy_source_blocks)
        turn_meta = {
            turn_id: (role, ordinal, source_id)
            for turn_id, role, ordinal, source_id in turns
        }

        chunk_rows = connection.execute(
            "SELECT c.chunk_id, c.turn_id, c.start_char, s.signature "
            "FROM chunks AS c "
            "LEFT JOIN chunk_cav_signatures AS s "
            "ON s.chunk_id = c.chunk_id AND s.artifact_id = ? "
            "ORDER BY c.rowid, c.start_char, c.chunk_id",
            (artifact_id,),
        ).fetchall()
        chunks_by_turn: dict[str, list[str]] = defaultdict(list)
        signatures_by_turn: dict[str, list[tuple[float, ...]]] = defaultdict(list)
        signature_by_chunk: dict[str, tuple[float, ...]] = {}
        chunk_ordinal: dict[str, int] = {}
        for chunk_id, turn_id, _start_char, signature_blob in chunk_rows:
            chunk_id = str(chunk_id)
            turn_id = str(turn_id)
            chunks_by_turn[turn_id].append(chunk_id)
            if turn_id in turn_meta:
                chunk_ordinal[chunk_id] = turn_meta[turn_id][1]
            signature = _unpack_f32(signature_blob)
            if signature:
                signatures_by_turn[turn_id].append(signature)
                signature_by_chunk[chunk_id] = signature

        outgoing: dict[str, list[_EdgeRecord]] = defaultdict(list)
        edge_rows = connection.execute(
            "SELECT source_chunk_id, destination_chunk_id, head_weights, qk_score "
            "FROM chunk_head_edges WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchall()
        for source_id, destination_id, weights_blob, qk_score in edge_rows:
            outgoing[str(source_id)].append(
                _EdgeRecord(
                    destination_id=str(destination_id),
                    head_weights=_unpack_f32(weights_blob),
                    qk_score=float(qk_score),
                )
            )

    records: dict[str, _TurnRecord] = {}
    for turn_id, (role, ordinal, source_id) in turn_meta.items():
        signatures = signatures_by_turn.get(turn_id, [])
        records[turn_id] = _TurnRecord(
            turn_id=turn_id,
            role=role,
            ordinal=ordinal,
            source_id=None if source_id is None else str(source_id),
            chunk_ids=tuple(chunks_by_turn.get(turn_id, [])),
            cav=_mean_vectors(signatures) if signatures else (),
        )

    def aggregate_edges(
        turn: _TurnRecord,
        *,
        maximum_destination_ordinal: int,
    ) -> list[_EdgeRecord]:
        best: dict[str, _EdgeRecord] = {}
        for chunk_id in turn.chunk_ids:
            for edge in outgoing.get(chunk_id, []):
                destination_ordinal = chunk_ordinal.get(edge.destination_id)
                if (
                    destination_ordinal is None
                    or destination_ordinal > maximum_destination_ordinal
                ):
                    continue
                current = best.get(edge.destination_id)
                if current is None or edge.qk_score > current.qk_score:
                    best[edge.destination_id] = edge
        return sorted(
            best.values(),
            key=lambda edge: (edge.qk_score, edge.destination_id),
            reverse=True,
        )

    examples: list[TransitionReplayExample] = []
    for source_group, turn_ids in groups.items():
        ordered = sorted(
            (records[turn_id] for turn_id in turn_ids), key=lambda turn: turn.ordinal
        )
        for previous, current, following in zip(
            ordered, ordered[1:], ordered[2:]
        ):
            if not previous.cav or not current.cav or not following.cav:
                continue
            candidates = aggregate_edges(
                current,
                maximum_destination_ordinal=current.ordinal - 1,
            )[:max_candidates]
            # The target is revealed by the next turn's teacher inspection,
            # but it may address anything that existed through the current turn.
            next_edges = aggregate_edges(
                following,
                maximum_destination_ordinal=current.ordinal,
            )
            if not candidates or not next_edges:
                continue
            transition_candidates = tuple(
                TransitionCandidate(
                    destination_id=edge.destination_id,
                    base_score=edge.qk_score,
                    head_attention=edge.head_weights,
                    head_cav_deltas=tuple(
                        _delta(
                            current.cav,
                            signature_by_chunk[edge.destination_id],
                        )
                        for _ in edge.head_weights
                    ),
                )
                for edge in candidates
                if edge.destination_id in signature_by_chunk
            )
            if not transition_candidates:
                continue
            examples.append(
                TransitionReplayExample(
                    source_group=source_group,
                    source_turn_id=current.turn_id,
                    from_role=current.role,
                    next_role=following.role,
                    source_cav=current.cav,
                    cav_velocity=_delta(previous.cav, current.cav),
                    next_cav=following.cav,
                    candidates=transition_candidates,
                    actual_destination_id=next_edges[0].destination_id,
                )
            )
    return artifact_id, examples


def _rank_of(destination_id: str, ranked_ids: Sequence[str]) -> int | None:
    try:
        return ranked_ids.index(destination_id) + 1
    except ValueError:
        return None


def run_transition_replay(
    examples: Sequence[TransitionReplayExample],
    *,
    train_source_groups: Sequence[str],
    policy: CausalTransitionPolicy | None = None,
    artifact_id: str = "",
    feedback_mode: str = "exact",
) -> TransitionReplayReport:
    """Fit on named groups, then score later groups prequentially and causally."""
    learner = policy or CausalTransitionPolicy()
    if feedback_mode not in {"exact", "cav"}:
        raise ValueError("feedback_mode must be 'exact' or 'cav'")
    train_groups = list(dict.fromkeys(train_source_groups))
    train_set = set(train_groups)
    available_groups = list(dict.fromkeys(example.source_group for example in examples))
    unknown = train_set - set(available_groups)
    if unknown:
        raise ValueError(f"unknown training source groups: {sorted(unknown)}")
    evaluation_groups = [group for group in available_groups if group not in train_set]
    if not evaluation_groups:
        raise ValueError("at least one source group must remain for evaluation")

    ordered = [
        example for group in train_groups for example in examples
        if example.source_group == group
    ] + [
        example for group in evaluation_groups for example in examples
        if example.source_group == group
    ]
    rows: list[TransitionReplayRow] = []
    training_count = 0
    evaluation_count = 0
    initial_snapshot = learner.snapshot()
    latest_prior_turn = max(
        (
            int(row.get("last_turn", 0))
            for section in ("heads", "edges")
            for row in initial_snapshot.get(section, [])
        ),
        default=-1,
    )
    turn_offset = latest_prior_turn + 1

    for turn, example in enumerate(ordered):
        baseline_ids = [
            candidate.destination_id
            for candidate in sorted(
                example.candidates,
                key=lambda candidate: (candidate.base_score, candidate.destination_id),
                reverse=True,
            )
        ]
        decision = learner.propose(
            source_id=example.source_turn_id,
            from_role=example.from_role,
            expected_next_role=example.next_role,
            source_cav=example.source_cav,
            cav_velocity=example.cav_velocity,
            candidates=example.candidates,
            turn=turn_offset + turn * 2,
            top_k=len(example.candidates),
        )
        learned_ids = [
            item.candidate.destination_id for item in decision.selected
        ]
        observed_delta = _delta(example.source_cav, example.next_cav)
        baseline_top = next(
            candidate
            for candidate in example.candidates
            if candidate.destination_id == baseline_ids[0]
        )
        learned_top = decision.selected[0].candidate
        baseline_delta_cosine = _cosine(
            baseline_top.head_cav_deltas[0], observed_delta
        )
        learned_delta_cosine = _cosine(
            learned_top.head_cav_deltas[0], observed_delta
        )
        learner.observe(
            decision,
            actual_destination_id=(
                example.actual_destination_id if feedback_mode == "exact" else None
            ),
            actual_next_role=example.next_role,
            next_cav=example.next_cav,
            turn=turn_offset + turn * 2 + 1,
        )

        if example.source_group in train_set:
            training_count += 1
            continue
        evaluation_count += 1
        baseline_rank = _rank_of(example.actual_destination_id, baseline_ids)
        learned_rank = _rank_of(example.actual_destination_id, learned_ids)
        rows.append(
            TransitionReplayRow(
                source_group=example.source_group,
                source_turn_id=example.source_turn_id,
                target_in_candidates=baseline_rank is not None,
                baseline_rank=baseline_rank,
                learned_rank=learned_rank,
                baseline_delta_cosine=baseline_delta_cosine,
                learned_delta_cosine=learned_delta_cosine,
            )
        )

    if evaluation_count == 0:
        raise ValueError("the evaluation source groups contain no transitions")
    baseline_rr = [0.0 if row.baseline_rank is None else 1.0 / row.baseline_rank for row in rows]
    learned_rr = [0.0 if row.learned_rank is None else 1.0 / row.learned_rank for row in rows]
    baseline_r1 = sum(rank == 1 for rank in (row.baseline_rank for row in rows)) / evaluation_count
    learned_r1 = sum(rank == 1 for rank in (row.learned_rank for row in rows)) / evaluation_count
    baseline_mrr = sum(baseline_rr) / evaluation_count
    learned_mrr = sum(learned_rr) / evaluation_count
    baseline_delta_cosine = (
        sum(row.baseline_delta_cosine for row in rows) / evaluation_count
    )
    learned_delta_cosine = (
        sum(row.learned_delta_cosine for row in rows) / evaluation_count
    )
    return TransitionReplayReport(
        artifact_id=artifact_id,
        train_source_groups=train_groups,
        evaluation_source_groups=evaluation_groups,
        training_transitions=training_count,
        evaluation_transitions=evaluation_count,
        target_candidate_coverage=(
            sum(row.target_in_candidates for row in rows) / evaluation_count
        ),
        baseline_recall_at_1=baseline_r1,
        learned_recall_at_1=learned_r1,
        baseline_mrr=baseline_mrr,
        learned_mrr=learned_mrr,
        baseline_mean_delta_cosine=baseline_delta_cosine,
        learned_mean_delta_cosine=learned_delta_cosine,
        improved=(
            learned_delta_cosine > baseline_delta_cosine
            if feedback_mode == "cav"
            else learned_r1 > baseline_r1 and learned_mrr >= baseline_mrr
        ),
        policy_snapshot=learner.snapshot(),
        rows=rows,
    )


def source_blocks_from_compile_report(
    path: str | Path,
) -> list[tuple[str, int]]:
    """Recover equal-sized legacy source blocks from an old compile report."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    source_ids = [
        str(source.get("source_family") or f"source_{index + 1}")
        for index, source in enumerate(report.get("sources", []))
    ]
    total = int(report.get("corpus", {}).get("assistant_episodes", 0))
    if not source_ids or total < 1 or total % len(source_ids):
        raise ValueError("compile report cannot prove equal legacy source blocks")
    size = total // len(source_ids)
    return [(source_id, size) for source_id in source_ids]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--compile-report", type=Path)
    parser.add_argument("--artifact-id")
    parser.add_argument("--train-sources", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--transition-weight", type=float, default=0.25)
    parser.add_argument("--velocity-weight", type=float, default=0.0)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--feedback-mode", choices=("exact", "cav"), default="cav")
    parser.add_argument(
        "--policy-snapshot",
        type=Path,
        help="Prior replay report or direct policy snapshot to continue causally",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    blocks = (
        source_blocks_from_compile_report(args.compile_report)
        if args.compile_report
        else []
    )
    artifact_id, examples = load_compiled_transition_examples(
        args.database,
        artifact_id=args.artifact_id,
        legacy_source_blocks=blocks,
        max_candidates=args.max_candidates,
    )
    groups = list(dict.fromkeys(example.source_group for example in examples))
    if not 0 <= args.train_sources < len(groups):
        raise ValueError("train-sources must leave at least one evaluation group")
    if args.policy_snapshot:
        snapshot_payload = json.loads(
            args.policy_snapshot.read_text(encoding="utf-8")
        )
        snapshot = snapshot_payload.get("policy_snapshot", snapshot_payload)
        policy = CausalTransitionPolicy.from_snapshot(snapshot)
    else:
        policy = CausalTransitionPolicy(
            transition_weight=args.transition_weight,
            velocity_weight=args.velocity_weight,
            gate_temperature=args.gate_temperature,
        )
    report = run_transition_replay(
        examples,
        train_source_groups=groups[: args.train_sources],
        policy=policy,
        artifact_id=artifact_id,
        feedback_mode=args.feedback_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"transitions: train={report.training_transitions} eval={report.evaluation_transitions}")
    print(f"target candidate coverage: {report.target_candidate_coverage:.1%}")
    print(
        f"R@1 baseline={report.baseline_recall_at_1:.1%} "
        f"learned={report.learned_recall_at_1:.1%}"
    )
    print(
        f"MRR baseline={report.baseline_mrr:.3f} "
        f"learned={report.learned_mrr:.3f}"
    )
    print(
        f"delta cosine baseline={report.baseline_mean_delta_cosine:.3f} "
        f"learned={report.learned_mean_delta_cosine:.3f} "
        f"improved={report.improved}"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
