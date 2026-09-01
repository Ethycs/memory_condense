"""Build or replay the provider-free R7 after-union A1 preflight.

This command reads only the sealed R7 runtime construction/replay pair.  It
does not read the post-seal target audit and it never calls a provider.  With
no external disposition or compiler-output artifacts, every selected leaf is
fail-open ``uncertain`` and the output is an exact-cover compilation worklist.
"""

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
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.r7_after_union_a1 import (  # noqa: E402
    DEFAULT_MAX_LEAVES_PER_CLASSIFIER_SHARD,
    DEFAULT_MAX_LEAVES_PER_SHARD,
    build_r7_after_union_a1_payload,
    replay_r7_after_union_a1_payload,
)
from tools.matched_eval.r7_after_union_temporal_fail_open import (  # noqa: E402
    EFFECTIVE_DISPOSITIONS_FORMAT,
)


DEFAULT_SOURCE_ROOT = Path(
    "eval_results/matched_eval_100/locked-semantic-global-terminal-v2-r7"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/matched_eval_100/locked-r7-after-union-a1-preflight-v2"
)
SOURCE_NAME = "reduced-semantic-global-terminal-assay-v2.json"
SOURCE_REPLAY_NAME = "reduced-semantic-global-terminal-assay-replay-v2.json"
CONSTRUCTION_NAME = "r7-after-union-a1-preflight-v2.json"
REPLAY_NAME = "r7-after-union-a1-preflight-replay-v2.json"


def _read_optional(path: Path | None) -> SealedArtifact | None:
    return read_sealed_json(path) if path is not None else None


def _inputs(args: argparse.Namespace) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact | None,
    SealedArtifact | None,
    SealedArtifact | None,
    SealedArtifact | None,
    SealedArtifact | None,
    SealedArtifact | None,
    SealedArtifact | None,
]:
    source = read_sealed_json(args.source_construction)
    replay = read_sealed_json(args.source_replay)
    if source.sha256 != replay.sha256 or source.payload != replay.payload:
        raise ValueError("R7 construction and replay are not byte-identical")
    dispositions = _read_optional(args.dispositions)
    dispositions_replay = _read_optional(
        getattr(args, "dispositions_replay", None)
    )
    compiler_outputs = _read_optional(args.compiler_outputs)
    temporal_a1 = _read_optional(getattr(args, "temporal_a1_construction", None))
    temporal_a1_replay = _read_optional(
        getattr(args, "temporal_a1_replay", None)
    )
    base_dispositions = _read_optional(getattr(args, "base_dispositions", None))
    base_dispositions_replay = _read_optional(
        getattr(args, "base_dispositions_replay", None)
    )
    effective = bool(
        dispositions is not None
        and dispositions.payload.get("format") == EFFECTIVE_DISPOSITIONS_FORMAT
    )
    temporal_parents = (
        dispositions_replay,
        temporal_a1,
        temporal_a1_replay,
        base_dispositions,
        base_dispositions_replay,
    )
    if effective and any(value is None for value in temporal_parents):
        raise ValueError(
            "effective dispositions require replay, temporal A1, and base-disposition pairs"
        )
    if not effective and any(
        value is not None
        for value in (
            temporal_a1,
            temporal_a1_replay,
            base_dispositions,
            base_dispositions_replay,
        )
    ):
        raise ValueError("temporal parent artifacts require effective dispositions")
    return (
        source,
        replay,
        dispositions,
        dispositions_replay,
        compiler_outputs,
        temporal_a1,
        temporal_a1_replay,
        base_dispositions,
        base_dispositions_replay,
    )


def _build(
    args: argparse.Namespace,
    source: SealedArtifact,
    replay: SealedArtifact,
    dispositions: SealedArtifact | None,
    dispositions_replay: SealedArtifact | None,
    compiler_outputs: SealedArtifact | None,
    temporal_a1: SealedArtifact | None,
    temporal_a1_replay: SealedArtifact | None,
    base_dispositions: SealedArtifact | None,
    base_dispositions_replay: SealedArtifact | None,
) -> dict[str, object]:
    return build_r7_after_union_a1_payload(
        source.payload,
        source.sha256,
        replay.sha256,
        disposition_payload=(
            dispositions.payload if dispositions is not None else None
        ),
        disposition_artifact_sha256=(
            dispositions.sha256 if dispositions is not None else None
        ),
        disposition_replay_payload=(
            dispositions_replay.payload if dispositions_replay is not None else None
        ),
        disposition_replay_artifact_sha256=(
            dispositions_replay.sha256 if dispositions_replay is not None else None
        ),
        temporal_a1_payload=(temporal_a1.payload if temporal_a1 is not None else None),
        temporal_a1_artifact_sha256=(
            temporal_a1.sha256 if temporal_a1 is not None else None
        ),
        temporal_a1_replay_payload=(
            temporal_a1_replay.payload if temporal_a1_replay is not None else None
        ),
        temporal_a1_replay_artifact_sha256=(
            temporal_a1_replay.sha256 if temporal_a1_replay is not None else None
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
        compiler_output_payload=(
            compiler_outputs.payload if compiler_outputs is not None else None
        ),
        compiler_output_artifact_sha256=(
            compiler_outputs.sha256 if compiler_outputs is not None else None
        ),
        max_leaves_per_shard=args.max_leaves_per_shard,
        max_leaves_per_classifier_shard=args.max_leaves_per_classifier_shard,
        expected_question_count=getattr(args, "expected_question_count", 11),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    """Build, seal, and byte-replay one provider-free A1 artifact."""

    (
        source,
        source_replay,
        dispositions,
        dispositions_replay,
        compiler_outputs,
        temporal_a1,
        temporal_a1_replay,
        base_dispositions,
        base_dispositions_replay,
    ) = _inputs(args)
    payload = _build(
        args,
        source,
        source_replay,
        dispositions,
        dispositions_replay,
        compiler_outputs,
        temporal_a1,
        temporal_a1_replay,
        base_dispositions,
        base_dispositions_replay,
    )
    output_root = Path(args.output_root)
    construction, created = publish_sealed_json(
        output_root / CONSTRUCTION_NAME, payload
    )
    replayed = replay_r7_after_union_a1_payload(
        construction.payload,
        source.payload,
        source.sha256,
        source_replay.sha256,
        disposition_payload=(
            dispositions.payload if dispositions is not None else None
        ),
        disposition_artifact_sha256=(
            dispositions.sha256 if dispositions is not None else None
        ),
        disposition_replay_payload=(
            dispositions_replay.payload if dispositions_replay is not None else None
        ),
        disposition_replay_artifact_sha256=(
            dispositions_replay.sha256 if dispositions_replay is not None else None
        ),
        temporal_a1_payload=(temporal_a1.payload if temporal_a1 is not None else None),
        temporal_a1_artifact_sha256=(
            temporal_a1.sha256 if temporal_a1 is not None else None
        ),
        temporal_a1_replay_payload=(
            temporal_a1_replay.payload if temporal_a1_replay is not None else None
        ),
        temporal_a1_replay_artifact_sha256=(
            temporal_a1_replay.sha256 if temporal_a1_replay is not None else None
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
        compiler_output_payload=(
            compiler_outputs.payload if compiler_outputs is not None else None
        ),
        compiler_output_artifact_sha256=(
            compiler_outputs.sha256 if compiler_outputs is not None else None
        ),
    )
    replay_artifact, replay_created = publish_sealed_json(
        output_root / REPLAY_NAME, replayed
    )
    if replay_artifact.sha256 != construction.sha256:
        raise ValueError("A1 replay bytes differ from construction")
    return {
        "construction_created": created,
        "construction_sha256": construction.sha256,
        "construction_status": payload["construction_status"],
        "classifier_payload_class": payload["classifier_payload_class"],
        "classifier_request_count": payload["classifier_request_count"],
        "compiler_payload_class": payload["compiler_payload_class"],
        "compiler_request_count": payload["compiler_request_count"],
        "missing_external_call_count": payload["missing_external_call_count"],
        "missing_classifier_call_count": payload["missing_classifier_call_count"],
        "missing_compiler_call_count": payload["missing_compiler_call_count"],
        "provider_calls_performed_by_core": 0,
        "question_count": payload["question_count"],
        "replay_byte_identical": True,
        "replay_created": replay_created,
        "replay_sha256": replay_artifact.sha256,
        "retained_transformer_token_state_bytes": 0,
        "selected_leaf_count": payload["selected_leaf_count"],
        "selected_population_sha256": payload["selected_population_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-construction",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / SOURCE_NAME,
    )
    parser.add_argument(
        "--source-replay",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / SOURCE_REPLAY_NAME,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dispositions", type=Path)
    parser.add_argument("--dispositions-replay", type=Path)
    parser.add_argument("--temporal-a1-construction", type=Path)
    parser.add_argument("--temporal-a1-replay", type=Path)
    parser.add_argument("--base-dispositions", type=Path)
    parser.add_argument("--base-dispositions-replay", type=Path)
    parser.add_argument("--compiler-outputs", type=Path)
    parser.add_argument(
        "--max-leaves-per-shard",
        type=int,
        default=DEFAULT_MAX_LEAVES_PER_SHARD,
    )
    parser.add_argument(
        "--max-leaves-per-classifier-shard",
        type=int,
        default=DEFAULT_MAX_LEAVES_PER_CLASSIFIER_SHARD,
    )
    parser.add_argument("--expected-question-count", type=int, default=11)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONSTRUCTION_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SOURCE_ROOT",
    "REPLAY_NAME",
    "SOURCE_NAME",
    "SOURCE_REPLAY_NAME",
    "build_parser",
    "main",
    "run",
]
