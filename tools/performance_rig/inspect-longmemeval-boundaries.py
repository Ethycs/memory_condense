"""Inspect literal-answer locations relative to activated LongMemEval sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval.locked_split import load_split_manifest, select_locked_split
from memory_condense.eval.recall import contains_answer
from memory_condense.eval.transition_trace import load_transition_trace
from memory_condense.ingest.loader import load_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    args = parser.parse_args()

    samples = select_locked_split(
        load_benchmark(args.benchmark_file),
        dataset_path=args.benchmark_file,
        manifest=load_split_manifest(args.split_manifest),
        split=args.split,
    )
    by_id = {sample.sample_id: sample for sample in samples}
    trace = load_transition_trace(args.trace)
    trace_by_id = {row.sample_id: row for row in trace.questions}

    requested = args.sample_id or [row.sample_id for row in trace.questions]
    for sample_id in requested:
        sample = by_id[sample_id]
        question = sample.questions[0]
        row = trace_by_id[sample_id]
        anchor_sources = list(
            dict.fromkeys(
                candidate.source_id
                for candidate in row.candidates
                if candidate.route == "hybrid_anchor" and candidate.source_id
            )
        )
        sources: dict[str, list[str]] = {}
        for (_role, text), source_id in zip(
            sample.turns, sample.turn_source_ids, strict=True
        ):
            if source_id is not None:
                sources.setdefault(source_id, []).append(text)

        answer_locations = []
        for source_id, texts in sources.items():
            for turn_index, text in enumerate(texts):
                if contains_answer([text], question.answer):
                    answer_locations.append(
                        {
                            "source_id": source_id,
                            "source_activated": source_id in anchor_sources,
                            "is_evidence_source": source_id in question.evidence_sources,
                            "turn_index": turn_index,
                            "source_turns": len(texts),
                            "source_tokens": sum(count_tokens(item) for item in texts),
                            "text": text,
                        }
                    )
        print(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "category": question.category,
                    "question_date": question.question_date,
                    "question": question.question,
                    "answer": question.answer,
                    "evidence_sources": question.evidence_sources,
                    "anchor_sources": anchor_sources,
                    "answer_locations": answer_locations,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
