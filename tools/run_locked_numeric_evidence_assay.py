#!/usr/bin/env python3
"""Freeze a provider-free numeric assay over the locked 72 prompt rows.

The runner reads only the locked preflight.  It never opens predictions,
references, judge output, or score artifacts and never calls a provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
)
from tools.matched_eval.artifacts import publish_sealed_json
from tools.matched_eval.numeric_evidence_reconciler import (
    FORMAT as RECONCILIATION_FORMAT,
    POLICY_ID,
    reconcile_sealed_numeric_evidence,
)


FORMAT = "memory-condense-locked-numeric-evidence-assay-v1"
ROW_FORMAT = f"{FORMAT}-row-v1"
POPULATION_FORMAT = f"{FORMAT}-population-v1"
SOURCE_FORMAT = "memory-condense-locked-specialist-final-terra-answer-v2-preflight"
EXPECTED_SOURCE_SHA256 = (
    "61371cd58b239a07f493ea4c116908a7f72e252cb503c0a5210f30c7f66ad413"
)
EXPECTED_PHYSICAL_ROW_COUNT = 72
ORDINARY_TYPED_TRANSFORM_FORMAT = (
    "memory-condense-locked-specialist-final-terra-answer-v2-"
    "ordinary-typed-prompt-transform-v1"
)
DEFAULT_SOURCE = Path(
    "eval_results/matched_eval_100/locked-specialist-final-answer-v2/"
    "locked-specialist-final-answer-preflight-v2.json"
)
DEFAULT_OUTPUT = Path(
    "eval_results/matched_eval_100/numeric-evidence-reconciler-v1/"
    "locked-specialist-final-answer-v2-numeric-population-v1.json"
)


class LockedNumericEvidenceAssayError(MatchedEvalContractError):
    """The locked preflight or frozen population failed authentication."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LockedNumericEvidenceAssayError(message)


def _load_exact_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LockedNumericEvidenceAssayError(
            "locked numeric source is not exact UTF-8 JSON"
        ) from exc
    _require(type(value) is dict, "locked numeric source must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def _provider_input_from_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    messages = row.get("messages")
    _require(
        type(messages) is list and len(messages) == 2,
        "locked numeric row must have exactly two messages",
    )
    messages_sha = require_sha256(row.get("messages_sha256"), "numeric messages")
    _require(
        identity_sha256(messages) == messages_sha,
        "locked numeric messages SHA-256 mismatch",
    )
    terminal = messages[1]
    _require(
        type(terminal) is dict
        and set(terminal) == {"content", "role"}
        and terminal.get("role") == "user"
        and type(terminal.get("content")) is str,
        "locked numeric terminal message changed schema",
    )
    try:
        provider_input = json.loads(terminal["content"])
    except json.JSONDecodeError as exc:
        raise LockedNumericEvidenceAssayError(
            "locked numeric provider input is not strict JSON"
        ) from exc
    _require(
        type(provider_input) is dict,
        "locked numeric provider input must be an object",
    )
    provider_sha = identity_sha256(provider_input)
    transform = row.get("adapter_prompt_transform")
    if transform is not None:
        _require(
            type(transform) is dict,
            "numeric adapter prompt transform changed schema",
        )
        require_sha256(
            transform.get("receipt_sha256"), "numeric adapter transform receipt"
        )
        bound_provider_sha = transform.get("provider_input_sha256")
        if transform.get("format") == ORDINARY_TYPED_TRANSFORM_FORMAT:
            _require(
                bound_provider_sha == provider_sha,
                "ordinary typed transform provider-input SHA-256 mismatch",
            )
        elif bound_provider_sha is not None:
            _require(
                bound_provider_sha == provider_sha,
                "adapter prompt transform provider-input SHA-256 mismatch",
            )
        _require(
            transform.get("target_messages_sha256") == messages_sha,
            "adapter prompt transform messages SHA-256 mismatch",
        )
    return provider_input, provider_sha


def _validated_source_rows(
    source: Mapping[str, Any],
    source_artifact_sha256: str,
) -> list[dict[str, Any]]:
    require_sha256(source_artifact_sha256, "numeric source artifact")
    _require(
        source_artifact_sha256 == EXPECTED_SOURCE_SHA256,
        "numeric assay source is not the locked preflight revision",
    )
    _require(source.get("format") == SOURCE_FORMAT, "numeric source format changed")
    _require(source.get("gold_loaded") is False, "numeric source loaded gold")
    rows = source.get("physical_prompt_rows")
    _require(
        type(rows) is list and len(rows) == EXPECTED_PHYSICAL_ROW_COUNT,
        "numeric physical-row population changed",
    )
    _require(
        all(type(row) is dict for row in rows),
        "numeric physical row must be an object",
    )
    ordinals = [row.get("ordinal") for row in rows]
    _require(
        all(type(value) is int and value >= 0 for value in ordinals)
        and ordinals == sorted(ordinals)
        and len(set(ordinals)) == len(ordinals),
        "numeric physical-row ordinals changed",
    )
    return rows


def build_frozen_population(
    source: Mapping[str, Any],
    *,
    source_artifact_sha256: str,
) -> dict[str, Any]:
    """Reexecute all sealed rows and return one canonical frozen population."""

    rows = _validated_source_rows(source, source_artifact_sha256)

    frozen_rows: list[dict[str, Any]] = []
    for row in rows:
        provider_input, provider_sha = _provider_input_from_row(row)
        reconciliation = reconcile_sealed_numeric_evidence(
            provider_input,
            sealed_provider_input_sha256=provider_sha,
        )
        frozen_row: dict[str, Any] = {
            "answer_plan_receipt_sha256": require_sha256(
                row.get("answer_plan_receipt_sha256"), "numeric answer plan"
            ),
            "format": ROW_FORMAT,
            "messages_sha256": require_sha256(
                row.get("messages_sha256"), "numeric messages"
            ),
            "ordinal": row["ordinal"],
            "provider_input_sha256": provider_sha,
            "reconciliation": reconciliation.projection(),
        }
        transform = row.get("adapter_prompt_transform")
        if transform is not None:
            frozen_row["adapter_prompt_transform_receipt_sha256"] = require_sha256(
                transform.get("receipt_sha256"),
                "numeric adapter prompt transform receipt",
            )
        frozen_row["row_receipt_sha256"] = identity_sha256(frozen_row)
        assert_gold_blind(frozen_row, path="locked_numeric_row")
        frozen_rows.append(frozen_row)

    population_projection = {
        "format": POPULATION_FORMAT,
        "ordered_rows": frozen_rows,
        "source_answer_plan_population_sha256": require_sha256(
            source.get("answer_plan_population_sha256"),
            "numeric answer-plan population",
        ),
        "source_preflight_artifact_sha256": source_artifact_sha256,
        "source_prompt_population_sha256": require_sha256(
            source.get("prompt_population_sha256"),
            "numeric prompt population",
        ),
    }
    population_sha = identity_sha256(population_projection)
    statuses = Counter(
        row["reconciliation"]["status"] for row in frozen_rows
    )
    artifact: dict[str, Any] = {
        "format": FORMAT,
        "gold_loaded": False,
        "ordered_rows": frozen_rows,
        "policy_id": POLICY_ID,
        "population_sha256": population_sha,
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
    assert_gold_blind(artifact, path="locked_numeric_population")
    return artifact


def freeze_locked_numeric_population(source_path: Path, output_path: Path) -> dict[str, Any]:
    source, source_sha = _load_exact_json(source_path)
    population = build_frozen_population(
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
    result = freeze_locked_numeric_population(args.source, args.output)
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
