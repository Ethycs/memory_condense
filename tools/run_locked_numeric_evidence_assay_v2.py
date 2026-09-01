#!/usr/bin/env python3
"""Freeze the V2 actual-schema numeric reconciliation over all 72 prompts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.matched_eval.artifacts import publish_sealed_json
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256, require_sha256
from tools.matched_eval.numeric_evidence_reconciler_v2 import (
    FORMAT as RECONCILIATION_FORMAT,
    POLICY_ID,
    reconcile_sealed_numeric_evidence_v2,
)
from tools.run_locked_numeric_evidence_assay import (
    DEFAULT_SOURCE,
    _load_exact_json,
    _provider_input_from_row,
    _validated_source_rows,
)


FORMAT = "memory-condense-locked-numeric-evidence-assay-v2"
ROW_FORMAT = f"{FORMAT}-row-v1"
POPULATION_FORMAT = f"{FORMAT}-population-v1"
DEFAULT_OUTPUT = Path(
    "eval_results/matched_eval_100/numeric-evidence-reconciler-v2/"
    "locked-specialist-final-answer-v2-numeric-population-v2.json"
)


def build_frozen_population_v2(
    source: Mapping[str, Any],
    *,
    source_artifact_sha256: str,
) -> dict[str, Any]:
    rows = _validated_source_rows(source, source_artifact_sha256)
    frozen_rows: list[dict[str, Any]] = []
    for row in rows:
        provider_input, provider_sha = _provider_input_from_row(row)
        reconciliation = reconcile_sealed_numeric_evidence_v2(
            provider_input,
            sealed_provider_input_sha256=provider_sha,
        )
        frozen_row: dict[str, Any] = {
            "answer_plan_receipt_sha256": require_sha256(
                row.get("answer_plan_receipt_sha256"), "V2 numeric answer plan"
            ),
            "format": ROW_FORMAT,
            "messages_sha256": require_sha256(
                row.get("messages_sha256"), "V2 numeric messages"
            ),
            "ordinal": row["ordinal"],
            "provider_input_sha256": provider_sha,
            "reconciliation": reconciliation.projection(),
        }
        transform = row.get("adapter_prompt_transform")
        if transform is not None:
            frozen_row["adapter_prompt_transform_receipt_sha256"] = require_sha256(
                transform.get("receipt_sha256"),
                "V2 numeric adapter prompt transform receipt",
            )
        frozen_row["row_receipt_sha256"] = identity_sha256(frozen_row)
        assert_gold_blind(frozen_row, path="locked_numeric_v2_row")
        frozen_rows.append(frozen_row)

    population_projection = {
        "format": POPULATION_FORMAT,
        "ordered_rows": frozen_rows,
        "source_answer_plan_population_sha256": require_sha256(
            source.get("answer_plan_population_sha256"),
            "V2 numeric answer-plan population",
        ),
        "source_preflight_artifact_sha256": source_artifact_sha256,
        "source_prompt_population_sha256": require_sha256(
            source.get("prompt_population_sha256"),
            "V2 numeric prompt population",
        ),
    }
    statuses = Counter(row["reconciliation"]["status"] for row in frozen_rows)
    artifact: dict[str, Any] = {
        "format": FORMAT,
        "gold_loaded": False,
        "ordered_rows": frozen_rows,
        "policy_id": POLICY_ID,
        "population_sha256": identity_sha256(population_projection),
        "provider_prompt_count": 0,
        "reconciliation_format": RECONCILIATION_FORMAT,
        "retained_transformer_token_state_bytes": 0,
        "row_count": len(frozen_rows),
        "source_answer_plan_population_sha256": population_projection[
            "source_answer_plan_population_sha256"
        ],
        "source_preflight_artifact_sha256": source_artifact_sha256,
        "source_prompt_population_sha256": population_projection[
            "source_prompt_population_sha256"
        ],
        "status_counts": {
            "conflicted": statuses.get("conflicted", 0),
            "insufficient": statuses.get("insufficient", 0),
            "supported": statuses.get("supported", 0),
        },
    }
    artifact["receipt_sha256"] = identity_sha256(artifact)
    assert_gold_blind(artifact, path="locked_numeric_v2_population")
    return artifact


def freeze_locked_numeric_population_v2(
    source_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    source, source_sha = _load_exact_json(source_path)
    population = build_frozen_population_v2(
        source,
        source_artifact_sha256=source_sha,
    )
    artifact, _created = publish_sealed_json(output_path, population)
    return artifact.payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = freeze_locked_numeric_population_v2(args.source, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "population_sha256": result["population_sha256"],
                "receipt_sha256": result["receipt_sha256"],
                "row_count": result["row_count"],
                "status_counts": result["status_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
