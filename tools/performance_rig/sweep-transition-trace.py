"""Recompose a frozen transition trace across stay/walk action budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_condense.eval.transition_trace import (
    TransitionArm,
    load_transition_trace,
    score_transition_arm,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max-prompt-tokens", type=int, default=8000)
parser.add_argument("--max-add-slots", type=int, default=20)
parser.add_argument("--max-replace-slots", type=int, default=9)
parser.add_argument(
    "--distances",
    help="Comma-separated radii to sweep (default: every radius in the trace)",
)
args = parser.parse_args()

pack = load_transition_trace(args.trace)
stay = TransitionArm(
    name="stay",
    retain_anchors=pack.hybrid_k,
    neighbor_slots=0,
)
stay_score, stay_hits = score_transition_arm(
    pack,
    stay,
    max_prompt_tokens=args.max_prompt_tokens,
)
scores = [stay_score]
hit_union = dict(stay_hits)

arms: list[TransitionArm] = []
directions = ("both", "previous", "next")
distances = (
    [int(value) for value in args.distances.split(",") if value.strip()]
    if args.distances
    else list(range(1, pack.max_radius + 1))
)
if not distances or any(
    distance < 1 or distance > pack.max_radius for distance in distances
):
    raise ValueError(f"distances must be within 1..{pack.max_radius}")
for distance in distances:
    for direction in directions:
        for slots in range(1, args.max_add_slots + 1):
            arms.append(
                TransitionArm(
                    name=f"add-{direction}-d{distance}-s{slots}",
                    retain_anchors=pack.hybrid_k,
                    neighbor_slots=slots,
                    max_distance=distance,
                    direction=direction,
                )
            )
        for slots in range(
            1,
            min(args.max_replace_slots, pack.hybrid_k - 1) + 1,
        ):
            arms.append(
                TransitionArm(
                    name=f"replace-{direction}-d{distance}-s{slots}",
                    retain_anchors=pack.hybrid_k - slots,
                    neighbor_slots=slots,
                    max_distance=distance,
                    direction=direction,
                )
            )

for arm in arms:
    score, hits = score_transition_arm(
        pack,
        arm,
        max_prompt_tokens=args.max_prompt_tokens,
        stay_hits=stay_hits,
    )
    scores.append(score)
    for question_id, hit in hits.items():
        hit_union[question_id] = hit_union.get(question_id, False) or hit

pareto = []
for candidate in scores:
    dominated = any(
        other.literal_recall >= candidate.literal_recall
        and other.mean_context_tokens <= candidate.mean_context_tokens
        and (
            other.literal_recall > candidate.literal_recall
            or other.mean_context_tokens < candidate.mean_context_tokens
        )
        for other in scores
    )
    if not dominated:
        pareto.append(candidate)
pareto.sort(key=lambda score: (-score.literal_recall, score.mean_context_tokens))

report = {
    "format": "memory-condense-transition-sweep-v1",
    "trace_sha256": pack.trace_sha256,
    "split": pack.split,
    "max_prompt_tokens": args.max_prompt_tokens,
    "stay": stay_score.model_dump(mode="json"),
    "oracle_action_union_recall": (
        sum(hit_union.values()) / len(hit_union) if hit_union else 0.0
    ),
    "pareto": [score.model_dump(mode="json") for score in pareto],
    "scores": [score.model_dump(mode="json") for score in scores],
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(
    f"stay: recall={stay_score.literal_recall:.1%} "
    f"tokens={stay_score.mean_context_tokens:.1f}\n"
    f"oracle action union: {report['oracle_action_union_recall']:.1%}\n"
    f"arms: {len(scores)}  pareto: {len(pareto)}"
)
for score in pareto[:20]:
    print(
        f"{score.arm.name:32} recall={score.literal_recall:.1%} "
        f"tokens={score.mean_context_tokens:.1f} "
        f"+{score.gained_vs_stay}/-{score.lost_vs_stay}"
    )
print(f"report: {args.output}")
