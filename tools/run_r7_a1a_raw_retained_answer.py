"""Seal/replay A1a raw-retained terminal prompt requests without provider IO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tools.matched_eval.artifacts import (  # noqa: E402
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.r7_a1a_raw_retained_answer import (  # noqa: E402
    build_r7_a1a_raw_retained_payload,
    replay_r7_a1a_raw_retained_payload,
)
from tools.matched_eval.r7_after_union_temporal_fail_open import (  # noqa: E402
    EFFECTIVE_DISPOSITIONS_FORMAT,
)


DEFAULT_A1_ROOT = Path(
    "eval_results/matched_eval_100/locked-r7-after-union-a1-preflight-v2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/matched_eval_100/locked-r7-a1a-raw-retained-terminal-preflight-v1"
)
A1_CONSTRUCTION_NAME = "r7-after-union-a1-preflight-v2.json"
A1_REPLAY_NAME = "r7-after-union-a1-preflight-replay-v2.json"
CONSTRUCTION_NAME = "r7-a1a-raw-retained-terminal-preflight-v1.json"
REPLAY_NAME = "r7-a1a-raw-retained-terminal-preflight-replay-v1.json"


def run(args: argparse.Namespace) -> dict[str, object]:
    source = read_sealed_json(args.a1_construction)
    source_replay = read_sealed_json(args.a1_replay)
    if source.sha256 != source_replay.sha256 or source.payload != source_replay.payload:
        raise ValueError("A1 v2 construction and replay are not byte-identical")
    dispositions = read_sealed_json(args.dispositions)
    disposition_replay = None
    disposition_replay_path = getattr(args, "dispositions_replay", None)
    if disposition_replay_path is not None:
        disposition_replay = read_sealed_json(disposition_replay_path)
        if (
            dispositions.sha256 != disposition_replay.sha256
            or dispositions.payload != disposition_replay.payload
        ):
            raise ValueError(
                "A1 classifier disposition construction and replay are not "
                "byte-identical"
            )
    base_dispositions = None
    base_dispositions_replay = None
    effective_overlay = (
        dispositions.payload.get("format") == EFFECTIVE_DISPOSITIONS_FORMAT
    )
    if effective_overlay:
        if disposition_replay is None:
            raise ValueError("effective dispositions require a byte-identical replay")
        base_path = getattr(args, "base_dispositions", None)
        base_replay_path = getattr(args, "base_dispositions_replay", None)
        if base_path is None or base_replay_path is None:
            raise ValueError(
                "effective dispositions require the base disposition pair"
            )
        base_dispositions = read_sealed_json(base_path)
        base_dispositions_replay = read_sealed_json(base_replay_path)
        if (
            base_dispositions.sha256 != base_dispositions_replay.sha256
            or base_dispositions.payload != base_dispositions_replay.payload
        ):
            raise ValueError(
                "base disposition construction and replay are not byte-identical"
            )
    elif getattr(args, "base_dispositions", None) is not None or getattr(
        args, "base_dispositions_replay", None
    ) is not None:
        raise ValueError("base temporal dispositions require an effective overlay")
    expected_question_count = getattr(args, "expected_question_count", 11)
    payload = build_r7_a1a_raw_retained_payload(
        source.payload,
        source.sha256,
        source_replay.sha256,
        dispositions.payload,
        dispositions.sha256,
        a1_preflight_replay_payload=source_replay.payload,
        disposition_replay_payload=(
            disposition_replay.payload if disposition_replay is not None else None
        ),
        disposition_replay_artifact_sha256=(
            disposition_replay.sha256 if disposition_replay is not None else None
        ),
        base_disposition_payload=(
            base_dispositions.payload if base_dispositions is not None else None
        ),
        base_disposition_artifact_sha256=(
            base_dispositions.sha256 if base_dispositions is not None else None
        ),
        base_disposition_replay_payload=(
            base_dispositions_replay.payload
            if base_dispositions_replay is not None
            else None
        ),
        base_disposition_replay_artifact_sha256=(
            base_dispositions_replay.sha256
            if base_dispositions_replay is not None
            else None
        ),
        expected_question_count=expected_question_count,
    )
    output_root = Path(args.output_root)
    construction, created = publish_sealed_json(
        output_root / CONSTRUCTION_NAME, payload
    )
    replayed = replay_r7_a1a_raw_retained_payload(
        construction.payload,
        source.payload,
        source.sha256,
        source_replay.sha256,
        dispositions.payload,
        dispositions.sha256,
        a1_preflight_replay_payload=source_replay.payload,
        disposition_replay_payload=(
            disposition_replay.payload if disposition_replay is not None else None
        ),
        disposition_replay_artifact_sha256=(
            disposition_replay.sha256 if disposition_replay is not None else None
        ),
        base_disposition_payload=(
            base_dispositions.payload if base_dispositions is not None else None
        ),
        base_disposition_artifact_sha256=(
            base_dispositions.sha256 if base_dispositions is not None else None
        ),
        base_disposition_replay_payload=(
            base_dispositions_replay.payload
            if base_dispositions_replay is not None
            else None
        ),
        base_disposition_replay_artifact_sha256=(
            base_dispositions_replay.sha256
            if base_dispositions_replay is not None
            else None
        ),
    )
    replay, replay_created = publish_sealed_json(
        output_root / REPLAY_NAME, replayed
    )
    if replay.sha256 != construction.sha256:
        raise ValueError("A1a replay bytes differ from construction")
    return {
        "construction_created": created,
        "construction_sha256": construction.sha256,
        "construction_status": payload["construction_status"],
        "control_prompt_request_count": payload["control_prompt_request_count"],
        "max_fixed_union_control_prompt_token_proxy": payload[
            "max_fixed_union_control_prompt_token_proxy"
        ],
        "max_terminal_prompt_token_proxy": payload[
            "max_terminal_prompt_token_proxy"
        ],
        "prompt_request_count": payload["prompt_request_count"],
        "provider_calls_performed_by_core": 0,
        "question_count": payload["question_count"],
        "replay_byte_identical": True,
        "replay_created": replay_created,
        "replay_sha256": replay.sha256,
        "retained_leaf_count": payload["density_totals"]["retained_leaf_count"],
        "fixed_union_leaf_count": payload["density_totals"][
            "fixed_union_leaf_count"
        ],
        "pruned_leaf_count": payload["density_totals"]["pruned_leaf_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a1-construction",
        type=Path,
        default=DEFAULT_A1_ROOT / A1_CONSTRUCTION_NAME,
    )
    parser.add_argument(
        "--a1-replay",
        type=Path,
        default=DEFAULT_A1_ROOT / A1_REPLAY_NAME,
    )
    parser.add_argument("--dispositions", type=Path, required=True)
    parser.add_argument(
        "--dispositions-replay",
        type=Path,
        help="Optional byte-identical sealed disposition replay to bind.",
    )
    parser.add_argument("--base-dispositions", type=Path)
    parser.add_argument("--base-dispositions-replay", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-question-count", type=int, default=11)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "A1_CONSTRUCTION_NAME",
    "A1_REPLAY_NAME",
    "CONSTRUCTION_NAME",
    "DEFAULT_A1_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "REPLAY_NAME",
    "build_parser",
    "main",
    "run",
]
