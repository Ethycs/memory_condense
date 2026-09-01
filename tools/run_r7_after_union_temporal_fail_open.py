"""Seal and replay the provider-free temporal fail-open disposition successor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json  # noqa: E402
from tools.matched_eval.r7_after_union_temporal_fail_open import (  # noqa: E402
    build_temporal_fail_open_artifacts,
)


DEFAULT_A1_ROOT = Path(
    "eval_results/matched_eval_100/locked-r7-after-union-a1-preflight-v2"
)
DEFAULT_CLASSIFIER_ROOT = DEFAULT_A1_ROOT / "terra-classifier-v1-network-recovery1"
DEFAULT_OUTPUT_ROOT = DEFAULT_CLASSIFIER_ROOT / "temporal-fail-open-effective-v1"
A1_NAME = "r7-after-union-a1-preflight-v2.json"
A1_REPLAY_NAME = "r7-after-union-a1-preflight-replay-v2.json"
DISPOSITIONS_NAME = "r7-after-union-a1-classifier-dispositions-v1.json"
DISPOSITIONS_REPLAY_NAME = (
    "r7-after-union-a1-classifier-dispositions-replay-v1.json"
)
EFFECTIVE_NAME = "r7-after-union-a1-temporal-fail-open-effective-v1.json"
EFFECTIVE_REPLAY_NAME = (
    "r7-after-union-a1-temporal-fail-open-effective-replay-v1.json"
)
REPORT_NAME = "r7-after-union-a1-temporal-fail-open-report-v2.json"
REPORT_REPLAY_NAME = "r7-after-union-a1-temporal-fail-open-report-replay-v2.json"


def run(args: argparse.Namespace) -> dict[str, object]:
    a1_artifact = read_sealed_json(Path(args.a1_artifact))
    a1_replay_artifact = read_sealed_json(Path(args.a1_replay))
    base_artifact = read_sealed_json(Path(args.base_dispositions))
    base_replay_artifact = read_sealed_json(Path(args.base_dispositions_replay))
    effective_payload, report_payload = build_temporal_fail_open_artifacts(
        a1_artifact.payload,
        a1_artifact.sha256,
        a1_replay_artifact.payload,
        a1_replay_artifact.sha256,
        base_artifact.payload,
        base_artifact.sha256,
        base_replay_artifact.payload,
        base_replay_artifact.sha256,
    )
    output_root = Path(args.output_root)
    effective, effective_created = publish_sealed_json(
        output_root / EFFECTIVE_NAME, effective_payload
    )
    report, report_created = publish_sealed_json(
        output_root / REPORT_NAME, report_payload
    )

    replayed_effective, replayed_report = build_temporal_fail_open_artifacts(
        a1_artifact.payload,
        a1_artifact.sha256,
        a1_replay_artifact.payload,
        a1_replay_artifact.sha256,
        base_artifact.payload,
        base_artifact.sha256,
        base_replay_artifact.payload,
        base_replay_artifact.sha256,
    )
    effective_replay, effective_replay_created = publish_sealed_json(
        output_root / EFFECTIVE_REPLAY_NAME, replayed_effective
    )
    report_replay, report_replay_created = publish_sealed_json(
        output_root / REPORT_REPLAY_NAME, replayed_report
    )
    if effective.sha256 != effective_replay.sha256:
        raise ValueError("temporal fail-open effective-disposition replay differs")
    if report.sha256 != report_replay.sha256:
        raise ValueError("temporal fail-open report replay differs")
    return {
        "base_disposition_artifact_sha256": base_artifact.sha256,
        "effective_artifact_created": effective_created,
        "effective_artifact_sha256": effective.sha256,
        "effective_replay_byte_identical": True,
        "effective_replay_created": effective_replay_created,
        "override_count": effective.payload["temporal_fail_open_override_count"],
        "physical_provider_calls": 0,
        "protected_selected_leaf_count": report.payload[
            "protected_selected_leaf_count"
        ],
        "question_count": report.payload["question_count"],
        "report_artifact_created": report_created,
        "report_artifact_sha256": report.sha256,
        "report_replay_byte_identical": True,
        "report_replay_created": report_replay_created,
        "retained_transformer_token_state_bytes": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a1-artifact", type=Path, default=DEFAULT_A1_ROOT / A1_NAME
    )
    parser.add_argument(
        "--a1-replay", type=Path, default=DEFAULT_A1_ROOT / A1_REPLAY_NAME
    )
    parser.add_argument(
        "--base-dispositions",
        type=Path,
        default=DEFAULT_CLASSIFIER_ROOT / DISPOSITIONS_NAME,
    )
    parser.add_argument(
        "--base-dispositions-replay",
        type=Path,
        default=DEFAULT_CLASSIFIER_ROOT / DISPOSITIONS_REPLAY_NAME,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "A1_NAME",
    "A1_REPLAY_NAME",
    "DEFAULT_A1_ROOT",
    "DEFAULT_CLASSIFIER_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DISPOSITIONS_NAME",
    "DISPOSITIONS_REPLAY_NAME",
    "EFFECTIVE_NAME",
    "EFFECTIVE_REPLAY_NAME",
    "REPORT_NAME",
    "REPORT_REPLAY_NAME",
    "build_parser",
    "main",
    "run",
]
