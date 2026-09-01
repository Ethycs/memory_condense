"""Seal and replay the provider-free R7 A1a target-retention audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tools.matched_eval.a1a_postseal_target_retention import (  # noqa: E402
    A1aPostsealTargetRetentionError,
    build_a1a_postseal_target_retention_audit,
    replay_a1a_postseal_target_retention_audit,
)
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)


DEFAULT_TARGET_AUDIT = Path(
    "eval_results/matched_eval_100/locked-semantic-global-terminal-v2-r7/"
    "semantic-global-terminal-postseal-fact-audit-v2.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/matched_eval_100/locked-r7-a1a-postseal-target-retention-audit-v1"
)
AUDIT_NAME = "r7-a1a-postseal-target-retention-audit-v1.json"
REPLAY_NAME = "r7-a1a-postseal-target-retention-audit-replay-v1.json"


def _inputs(
    args: argparse.Namespace,
) -> tuple[SealedArtifact, SealedArtifact, SealedArtifact]:
    # Authenticate the runtime seal before opening the gold-informed audit.
    construction = read_sealed_json(args.runtime_construction)
    replay = read_sealed_json(args.runtime_replay)
    if construction.sha256 != replay.sha256 or construction.payload != replay.payload:
        raise A1aPostsealTargetRetentionError(
            "A1a runtime construction and replay are not byte-identical"
        )
    target_audit = read_sealed_json(args.target_audit)
    return construction, replay, target_audit


def run(args: argparse.Namespace) -> dict[str, object]:
    """Build, seal, and byte-replay one post-seal retention audit."""

    construction, replay, target_audit = _inputs(args)
    payload = build_a1a_postseal_target_retention_audit(
        construction.payload,
        construction.sha256,
        replay.sha256,
        target_audit.payload,
        target_audit.sha256,
    )
    output_root = Path(args.output_root)
    sealed, created = publish_sealed_json(output_root / AUDIT_NAME, payload)
    replayed = replay_a1a_postseal_target_retention_audit(
        sealed.payload,
        construction.payload,
        construction.sha256,
        replay.sha256,
        target_audit.payload,
        target_audit.sha256,
    )
    sealed_replay, replay_created = publish_sealed_json(
        output_root / REPLAY_NAME,
        replayed,
    )
    if sealed.sha256 != sealed_replay.sha256:
        raise A1aPostsealTargetRetentionError(
            "A1a post-seal audit replay bytes differ"
        )
    return {
        "all_control_prompts_within_hard_cap": payload[
            "all_control_prompts_within_hard_cap"
        ],
        "all_prompts_within_hard_cap": payload["all_prompts_within_hard_cap"],
        "audit_created": created,
        "audit_sha256": sealed.sha256,
        "decision": payload["decision"],
        "new_provider_calls": 0,
        "question_count": payload["question_count"],
        "replay_byte_identical": True,
        "replay_created": replay_created,
        "replay_sha256": sealed_replay.sha256,
        "retained_transformer_token_state_bytes": 0,
        "semantic_atom_retained_count": payload["semantic_atom_retained_count"],
        "strict_go": payload["strict_go"],
        "target_bearing_leaf_pruned_count": payload[
            "target_bearing_leaf_pruned_count"
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-construction", type=Path, required=True)
    parser.add_argument("--runtime-replay", type=Path, required=True)
    parser.add_argument(
        "--target-audit",
        type=Path,
        default=DEFAULT_TARGET_AUDIT,
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
    "AUDIT_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_TARGET_AUDIT",
    "REPLAY_NAME",
    "build_parser",
    "main",
    "run",
]
