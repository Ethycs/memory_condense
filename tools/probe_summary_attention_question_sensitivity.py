"""Synthetic, local-only check that question changes move attention rankings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from memory_condense.associations.head_memory_models import AssociativeMemoryCandidate
from memory_condense.search.episodes.qwen_episode_signal import qwen_linker_identity
from tools.evaluate_hierarchical_spine_full100 import implementation, new_linker
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


FACTS = (
    ("commuter bicycle brand", "Trek"),
    ("houseplant variety", "jade"),
    ("weekend hobby", "pottery"),
    ("preferred tea", "jasmine"),
    ("running shoe brand", "Asics"),
    ("telescope brand", "Celestron"),
    ("wallet color", "burgundy"),
    ("breakfast cereal", "granola"),
)
NEUTRAL = "What information did I share?"


def fixtures(structured):
    result = []
    for i, (topic, value) in enumerate(FACTS):
        text = f"The user's {topic} is {value}."
        if structured:
            text = json.dumps({"user_spine": text,
                "attached_context_not_user_assertions": "No additional context.",
                "transcript_date_range": ["2026-01-01T00:00:00+00:00"]}, sort_keys=True)
        result.append(AssociativeMemoryCandidate(episode_id=f"synthetic-{i}",
                      text=text, route="section_summary"))
    return tuple(result)


def rankings(rows):
    """Score diagnostics only; no change to the production ranking rule."""
    output = {}
    for representation in ("plain", "structured"):
        group = [r for r in rows if r["representation"] == representation]
        neutral = {h["id"]: h for h in group[0]["hits"]}
        cases = []
        for row in group[1:]:
            ranked = sorted(row["hits"], key=lambda h: (-h["qk"], -h["ov"], h["id"]))
            centered = sorted(row["hits"], key=lambda h: (-(h["qk"] - neutral[h["id"]]["qk"]), h["id"]))
            expected = row["expected_id"]
            cases.append({"expected_id": expected, "raw_top_id": ranked[0]["id"],
                "raw_expected_rank": next(i+1 for i,h in enumerate(ranked) if h["id"] == expected),
                "neutral_centered_top_id": centered[0]["id"],
                "neutral_centered_expected_rank": next(i+1 for i,h in enumerate(centered) if h["id"] == expected)})
        output[representation] = {"questions": len(cases),
            "raw_top1_correct": sum(c["raw_expected_rank"] == 1 for c in cases),
            "raw_top4_retained": sum(c["raw_expected_rank"] <= 4 for c in cases),
            "neutral_centered_top1_correct": sum(c["neutral_centered_expected_rank"] == 1 for c in cases),
            "neutral_centered_top4_retained": sum(c["neutral_centered_expected_rank"] <= 4 for c in cases),
            "distinct_raw_winners": len({c["raw_top_id"] for c in cases}), "cases": cases}
    return output


def run(root):
    bound = {**implementation(), str(Path(__file__).relative_to(Path.cwd())):
             hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    preflight, _ = publish_sealed_json(root / "preflight.json", {
        "format": "synthetic-summary-attention-question-sensitivity-v1",
        "facts": FACTS, "neutral_query": NEUTRAL, "implementation_sha256": bound,
        "local_model_batches": 18, "raw_corpus_inputs": False,
        "benchmark_inputs": False, "provider_calls": 0})
    if (root / "result.json").exists():
        result = read_sealed_json(root / "result.json")
        if result.payload["preflight_sha256"] != preflight.sha256:
            raise ValueError("probe preflight changed")
        if rankings(result.payload["rows"]) != result.payload["rankings"]:
            raise ValueError("saved score reduction changed")
        print(json.dumps({"result_sha256": result.sha256, "replay": True,
                          "new_model_batches": 0, "rankings": result.payload["rankings"]}, indent=2))
        return
    linker = new_linker()
    identity = qwen_linker_identity(linker, strict=True)
    rows = []
    try:
        for structured in (False, True):
            candidates = fixtures(structured)
            for n, query in enumerate((NEUTRAL, *(f"What is my {topic}?" for topic, _ in FACTS))):
                started = time.perf_counter()
                inspection = linker.inspect_coverage(query, candidates)
                elapsed = time.perf_counter() - started
                if (inspection.passes != 1 or inspection.total_candidate_inspections != 8
                        or len(inspection.hits) != 8):
                    raise ValueError("probe requires complete eight-candidate inspection")
                rows.append({"representation": "structured" if structured else "plain",
                    "query": query, "expected_id": None if n == 0 else f"synthetic-{n-1}",
                    "seconds": elapsed, "workspace_tokens": inspection.workspace_tokens,
                    "hits": [{"id": h.episode_id, "qk": h.qk_score, "ov": h.ov_transport,
                              "head_weights": h.head_weights} for h in inspection.hits]})
                print(f"Synthetic batch {len(rows)}/18 saved in memory", flush=True)
        if qwen_linker_identity(linker, strict=True) != identity:
            raise ValueError("local model identity changed")
        result, _ = publish_sealed_json(root / "result.json", {
            "preflight_sha256": preflight.sha256, "linker_identity": identity,
            "rows": rows, "rankings": rankings(rows), "local_model_batches": len(rows),
            "provider_calls": 0, "benchmark_accuracy_claim": False,
            "raw_corpus_inputs": False, "retained_transformer_token_state_bytes": 0})
        print(json.dumps({"result_sha256": result.sha256, "rankings": result.payload["rankings"]}, indent=2), flush=True)
    finally:
        # Qwen3PrefixEncoder has no close protocol; this CLI owns its process.
        del linker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args().output_root)
