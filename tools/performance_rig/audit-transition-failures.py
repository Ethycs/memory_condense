"""Classify trace misses without conflating retrieval and answer reasoning."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from memory_condense.eval.recall import contains_answer
from memory_condense.eval.transition_trace import (
    TransitionArm,
    compose_transition_context,
    load_transition_trace,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retain-anchors", type=int, default=10)
    parser.add_argument("--neighbor-slots", type=int, default=0)
    parser.add_argument("--max-distance", type=int, default=1)
    parser.add_argument(
        "--direction", choices=("both", "previous", "next"), default="both"
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=8000)
    args = parser.parse_args()

    pack = load_transition_trace(args.trace)
    arm = TransitionArm(
        name="audit",
        retain_anchors=args.retain_anchors,
        neighbor_slots=args.neighbor_slots,
        max_distance=args.max_distance,
        direction=args.direction,
    )
    rows = []
    for question in pack.questions:
        texts, sources = compose_transition_context(
            question,
            arm,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        hit = contains_answer(texts, question.answer)
        union_candidates = [candidate.text for candidate in question.candidates]
        union_hit = contains_answer(union_candidates, question.answer)
        expected_sources = set(question.evidence_sources)
        retrieved_sources = {source for source in sources if source}
        evidence_coverage = (
            len(expected_sources & retrieved_sources) / len(expected_sources)
            if expected_sources
            else None
        )
        answer_candidates = [
            candidate
            for candidate in question.candidates
            if contains_answer([candidate.text], question.answer)
        ]
        if hit:
            failure_class = "hit"
        elif not question.answer_in_haystack:
            failure_class = "requires_reasoning_or_semantic_generation"
        elif union_hit:
            failure_class = "policy_or_prompt_admission_miss"
        else:
            failure_class = "candidate_boundary_miss"
        first_answer = answer_candidates[0] if answer_candidates else None
        rows.append(
            {
                "sample_id": question.sample_id,
                "category": question.category,
                "failure_class": failure_class,
                "answer_in_haystack": question.answer_in_haystack,
                "selected_literal_hit": hit,
                "candidate_union_hit": union_hit,
                "evidence_coverage": evidence_coverage,
                "question": question.question,
                "answer": question.answer,
                "first_answer_route": first_answer.route if first_answer else "",
                "first_answer_anchor_rank": (
                    first_answer.anchor_rank if first_answer else ""
                ),
                "first_answer_distance": (
                    first_answer.transition_distance if first_answer else ""
                ),
                "first_answer_direction": (
                    first_answer.transition_direction if first_answer else ""
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["failure_class"])
        counts[label] = counts.get(label, 0) + 1
    print(f"questions: {len(rows)}")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
