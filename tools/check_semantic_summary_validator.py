"""Small production-encoder admission check for proxy semantic validation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from memory_condense.runtime.artifacts import save
from memory_condense.runtime.embedding import FastEmbedBGE
from memory_condense.runtime.semantic_summary import (
    SEMANTIC_VALIDATION_POLICY,
    SEMANTIC_VALIDATION_SHA256,
    _cosines,
    _critical_values,
    _polarity_conflict,
    _support_candidates,
)


CASES = (
    {
        "name": "status_paraphrase",
        "source": "Release r18 is planned, not deployed.",
        "summary": "Release r18 remains scheduled and has not shipped.",
        "expected": True,
    },
    {
        "name": "home_price_paraphrase",
        "source": "The accepted purchase price for the home was $425,000 after inspection.",
        "summary": "The home cost $425,000 after the inspection.",
        "expected": True,
    },
    {
        "name": "changed_identifier",
        "source": "Release r18 is planned, not deployed.",
        "summary": "Release r19 remains scheduled and has not shipped.",
        "expected": False,
    },
    {
        "name": "reversed_status",
        "source": "Release r18 is planned, not deployed.",
        "summary": "Release r18 deployed.",
        "expected": False,
    },
    {
        "name": "unrelated_claim",
        "source": "Release r18 is planned, not deployed.",
        "summary": "The family attended a theater performance downtown.",
        "expected": False,
    },
)


def main(output: Path) -> None:
    encoder = FastEmbedBGE()
    rows = []
    try:
        for case in CASES:
            candidates = _support_candidates(case["source"])
            scores = _cosines(encoder, case["summary"], candidates)
            best = int(np.argmax(scores))
            support, cosine = candidates[best], float(scores[best])
            critical = _critical_values(case["summary"]).issubset(_critical_values(case["source"]))
            polarity = _polarity_conflict(case["summary"], support)
            accepted = bool(cosine >= SEMANTIC_VALIDATION_POLICY["minimum_support_cosine"]
                            and critical and not polarity)
            rows.append({
                **case,
                "support": support,
                "cosine": cosine,
                "critical_facts_preserved": critical,
                "polarity_conflict": polarity,
                "accepted": accepted,
                "passed": accepted is case["expected"],
            })
    finally:
        encoder.close()
    receipt = save(output, {
        "format": "semantic-summary-validator-production-encoder-check-v1",
        "policy": SEMANTIC_VALIDATION_POLICY,
        "policy_sha256": SEMANTIC_VALIDATION_SHA256,
        "cases": rows,
        "passed": all(row["passed"] for row in rows),
    })
    if not receipt.payload["passed"]:
        raise SystemExit("Semantic validation production-encoder check failed")
    print({"output": str(receipt.path), "sha256": receipt.sha256, "cases": len(rows)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("eval_results/semantic-summary-validator-check.json"))
    main(parser.parse_args().output)
