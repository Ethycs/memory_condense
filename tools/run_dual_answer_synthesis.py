#!/usr/bin/env python3
"""Run the sealed legacy-S0/v2 dual-answer synthesis experiment.

The answer side is gold-blind.  It verifies the locked legacy S0 observation,
the matched S0-v2 answer plane, and the S0-v4 packet preflight before Terra is
allowed to synthesize one answer.  The judge commands first reconstruct that
sealed synthesis plane and only then load the benchmark dataset for Sol.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import identity_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN_ROOT = (
    REPOSITORY_ROOT
    / "eval_results"
    / "longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
)
DEFAULT_RETRIEVAL = (
    REPOSITORY_ROOT
    / "eval_results"
    / "longmemeval-1m-recall-guarded-cumulative-validation-20260822"
    / "retrieval.json"
)
DEFAULT_SPLIT = (
    REPOSITORY_ROOT
    / "docs"
    / "10 - Research Log"
    / "data"
    / "longmemeval-95-target-split-v2.json"
)
DEFAULT_LEGACY_ARTIFACT_ROOT = DEFAULT_CAMPAIGN_ROOT
DEFAULT_V2_ROOT = DEFAULT_CAMPAIGN_ROOT / "matched-eval-spine-v2" / "s0-control-v2"
DEFAULT_V4_ROOT = DEFAULT_CAMPAIGN_ROOT / "matched-eval-spine-v4" / "s0-control-v4"
DEFAULT_V4_PREFLIGHT = DEFAULT_V4_ROOT / "s0-v4-preflight.json"
DEFAULT_SYNTHESIS_ROOT = DEFAULT_CAMPAIGN_ROOT / "dual-answer-synthesis-v1"
DEFAULT_FULL_ROOT = DEFAULT_SYNTHESIS_ROOT / "full100"
DEFAULT_FLIP10_ROOT = DEFAULT_SYNTHESIS_ROOT / "flip10"

DEFAULT_V2_ANSWER_RUN_SHA256 = (
    "1a2545655d4a5e2061dc1b80efae39c7f8c70f5dc394f36c97d1312f70f39d8a"
)
DEFAULT_V4_PREFLIGHT_SHA256 = (
    "fb26557cac1a9290e0b7b7173ac70d7e3e2b94aae5419c2308e011f33dca79e5"
)

SYNTHESIS_PREFLIGHT_NAME = "synthesis-preflight.json"
SYNTHESIS_ANSWER_RUN_NAME = "synthesis-answer-run.json"
SYNTHESIS_ANSWER_REPLAY_NAME = "synthesis-answer-run-replay.json"
DIAGNOSTIC_GATE_NAME = "diagnostic-gate.json"

FLIP10_ORDINALS = (5, 16, 29, 34, 50, 52, 65, 79, 83, 97)
FLIP10_RESCUE_ORDINALS = (29, 34, 50)
FLIP10_REGRESSION_ORDINALS = (5, 16, 52, 65, 79, 83, 97)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_argument(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return dict(value)


def _scope(args: argparse.Namespace) -> tuple[Path, tuple[int, ...] | None]:
    requested = getattr(args, "output_root", None)
    if bool(args.flip10):
        root = DEFAULT_FLIP10_ROOT if requested is None else Path(requested)
        if root.resolve() != DEFAULT_FLIP10_ROOT.resolve():
            raise ValueError(
                "--flip10 has a fixed output root; omit --output-root or use "
                f"{DEFAULT_FLIP10_ROOT}"
            )
        return root, FLIP10_ORDINALS
    return (DEFAULT_FULL_ROOT if requested is None else Path(requested)), None


def _validate_provider_authorization(args: argparse.Namespace) -> None:
    enabled = bool(args.enable_provider)
    authorized = int(args.authorized_provider_calls)
    if enabled and authorized < 1:
        raise ValueError(
            "--enable-provider requires a positive exact "
            "--authorized-provider-calls budget"
        )
    if not enabled and authorized != 0:
        raise ValueError(
            "--authorized-provider-calls must be zero unless --enable-provider "
            "is set"
        )


def _load_v2_answer_plane(args: argparse.Namespace) -> Any:
    """Reconstruct the full locked v2 plane without loading benchmark gold."""

    from tools.matched_eval.live import (
        ANSWER_REPLAY_NAME,
        ANSWER_RUN_NAME,
        load_verified_s0_v2_answer_plane,
    )

    root = Path(args.v2_root)
    return load_verified_s0_v2_answer_plane(
        root / ANSWER_RUN_NAME,
        root / ANSWER_REPLAY_NAME,
        expected_run_sha256=str(args.expected_v2_answer_run_sha256),
        retrieval_path=Path(args.retrieval),
        max_concurrency=int(args.max_concurrency),
    )


def _source_arguments(
    args: argparse.Namespace,
    *,
    output_root: Path,
    source_ordinals: tuple[int, ...] | None,
) -> dict[str, Any]:
    return {
        "legacy_artifact_root": Path(args.legacy_artifact_root),
        "v2_answer_plane": _load_v2_answer_plane(args),
        "v4_preflight_path": Path(args.v4_preflight),
        "expected_v4_preflight_sha256": str(
            args.expected_v4_preflight_sha256
        ),
        "retrieval_path": Path(args.retrieval),
        "output_root": output_root,
        "source_ordinals": source_ordinals,
    }


def _ordered_rows(payload: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    raw = payload.get("ordered_rows")
    if type(raw) is not list:
        raise RuntimeError(f"{label} is missing ordered_rows")
    return [_mapping(row, f"{label} ordered row") for row in raw]


def _source_ordinal(row: Mapping[str, Any]) -> int:
    value = row.get("source_ordinal", row.get("ordinal"))
    if type(value) is not int or value < 0:
        raise RuntimeError("synthesis preflight row is missing source_ordinal")
    return value


def _sha256_leaves(value: object, *, path: str = "$") -> dict[str, str]:
    leaves: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = value[key]
            child_path = f"{path}.{key}"
            if isinstance(child, str) and _SHA256.fullmatch(child):
                leaves[child_path] = child
            else:
                leaves.update(_sha256_leaves(child, path=child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            leaves.update(_sha256_leaves(child, path=f"{path}[{index}]"))
    return leaves


def _source_bindings(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the explicit shared-source binding from a synthesis preflight."""

    raw = payload.get("source_bindings")
    if not isinstance(raw, Mapping):
        raise RuntimeError("synthesis preflight is missing source_bindings")
    bindings = dict(raw)
    leaves = _sha256_leaves(bindings)
    lowered_paths = tuple(path.casefold() for path in leaves)
    for family in ("legacy", "v2", "v4"):
        if not any(family in path for path in lowered_paths):
            raise RuntimeError(
                f"synthesis source_bindings omit the {family} hash family"
            )
    return bindings


def _row_hash_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_ordinal": _source_ordinal(row),
            "sha256_fields": _sha256_leaves(row),
        }
        for row in rows
    ]


def _require_declared_ordinals(
    payload: Mapping[str, Any],
    *,
    expected: Sequence[int],
    label: str,
) -> None:
    for key in ("source_ordinals", "selected_source_ordinals"):
        if key in payload and payload[key] != list(expected):
            raise RuntimeError(f"{label} {key} changed")


def _seal_flip10_gate(
    *,
    output_root: Path,
    diagnostic_preflight: SealedArtifact,
    full_root: Path = DEFAULT_FULL_ROOT,
) -> SealedArtifact:
    """Seal the preregistered flip-case and full-population promotion gates."""

    if output_root.resolve() == full_root.resolve():
        raise RuntimeError("the flip10 diagnostic requires a dedicated root")
    full = read_sealed_json(full_root / SYNTHESIS_PREFLIGHT_NAME)
    full_rows = _ordered_rows(full.payload, "full synthesis preflight")
    diagnostic_rows = _ordered_rows(
        diagnostic_preflight.payload, "flip10 synthesis preflight"
    )
    if len(full_rows) != 100:
        raise RuntimeError("full synthesis preflight must contain 100 rows")
    _require_declared_ordinals(
        full.payload,
        expected=range(100),
        label="full synthesis preflight",
    )
    _require_declared_ordinals(
        diagnostic_preflight.payload,
        expected=FLIP10_ORDINALS,
        label="flip10 synthesis preflight",
    )
    by_ordinal = {_source_ordinal(row): row for row in full_rows}
    if len(by_ordinal) != 100 or set(by_ordinal) != set(range(100)):
        raise RuntimeError(
            "full synthesis preflight source ordinals must be exactly 0..99"
        )
    expected_rows = [by_ordinal[ordinal] for ordinal in FLIP10_ORDINALS]
    if diagnostic_rows != expected_rows:
        raise RuntimeError(
            "flip10 prompts are not an exact view of the sealed full preflight"
        )

    full_sources = _source_bindings(full.payload)
    diagnostic_sources = _source_bindings(diagnostic_preflight.payload)
    if diagnostic_sources != full_sources:
        raise RuntimeError(
            "flip10 and full preflights do not share exact legacy/v2/v4 sources"
        )
    full_row_hashes = _row_hash_projection(full_rows)
    diagnostic_row_hashes = _row_hash_projection(diagnostic_rows)
    gate = {
        "format": "memory-condense-dual-answer-synthesis-diagnostic-gate-v1",
        "diagnostic_preflight_sha256": diagnostic_preflight.sha256,
        "full_preflight_sha256": full.sha256,
        "gold_loaded_into_answer_prompts": False,
        "selection_is_posthoc_outcome_conditioned": True,
        "selected_ordinals": list(FLIP10_ORDINALS),
        "rescue_ordinals": list(FLIP10_RESCUE_ORDINALS),
        "regression_ordinals": list(FLIP10_REGRESSION_ORDINALS),
        "minimum_proceed_gate": {
            "minimum_correct": 8,
            "minimum_regressions_recovered": 5,
            "minimum_rescues_retained": 3,
            "question_count": 10,
            "retain_all_rescues": True,
        },
        "strong_recovery_goal": {
            "minimum_correct": 10,
            "recover_all_regressions": True,
            "retain_all_rescues": True,
        },
        "full_100_promotion_gate": {
            "baseline_arm": "S0_CONTROL_V2",
            "baseline_correct": 53,
            "minimum_correct": 60,
            "minimum_paired_net_improvement_vs_v2": 7,
            "question_count": 100,
        },
        "source_bindings": full_sources,
        "source_bindings_sha256": identity_sha256(full_sources),
        "source_prompt_hash_bindings": {
            "diagnostic_ordered_rows_sha256": identity_sha256(
                diagnostic_row_hashes
            ),
            "diagnostic_rows_are_exact_full_view": True,
            "full_ordered_rows_sha256": identity_sha256(full_row_hashes),
        },
    }
    artifact, _created = publish_sealed_json(
        output_root / DIAGNOSTIC_GATE_NAME, gate
    )
    return artifact


def _require_flip10_gate(
    *,
    output_root: Path,
    source_ordinals: tuple[int, ...] | None,
) -> None:
    if source_ordinals != FLIP10_ORDINALS:
        return
    preflight = read_sealed_json(output_root / SYNTHESIS_PREFLIGHT_NAME)
    gate = read_sealed_json(output_root / DIAGNOSTIC_GATE_NAME)
    full = read_sealed_json(DEFAULT_FULL_ROOT / SYNTHESIS_PREFLIGHT_NAME)
    if (
        gate.payload.get("format")
        != "memory-condense-dual-answer-synthesis-diagnostic-gate-v1"
        or gate.payload.get("diagnostic_preflight_sha256") != preflight.sha256
        or gate.payload.get("full_preflight_sha256") != full.sha256
        or gate.payload.get("selected_ordinals") != list(FLIP10_ORDINALS)
    ):
        raise RuntimeError("flip10 gate is not bound to these preflights")


def _load_synthesis_plane(
    args: argparse.Namespace,
    *,
    output_root: Path,
    source_ordinals: tuple[int, ...] | None,
) -> Any:
    """Verify synthesis run, replay, journals, and ledger before gold loads."""

    from tools.matched_eval.synthesis import (
        load_verified_dual_answer_synthesis_plane,
    )

    expected = str(args.expected_answer_run_sha256)
    return load_verified_dual_answer_synthesis_plane(
        output_root / SYNTHESIS_ANSWER_RUN_NAME,
        output_root / SYNTHESIS_ANSWER_REPLAY_NAME,
        expected_run_sha256=expected,
        **_source_arguments(
            args,
            output_root=output_root,
            source_ordinals=source_ordinals,
        ),
        max_concurrency=int(args.max_concurrency),
    )


def _add_source_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    command.add_argument(
        "--legacy-artifact-root",
        type=Path,
        default=DEFAULT_LEGACY_ARTIFACT_ROOT,
    )
    command.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    command.add_argument(
        "--expected-v2-answer-run-sha256",
        type=_sha256_argument,
        default=DEFAULT_V2_ANSWER_RUN_SHA256,
    )
    command.add_argument(
        "--v4-preflight", type=Path, default=DEFAULT_V4_PREFLIGHT
    )
    command.add_argument(
        "--expected-v4-preflight-sha256",
        type=_sha256_argument,
        default=DEFAULT_V4_PREFLIGHT_SHA256,
    )
    command.add_argument("--output-root", type=Path)
    command.add_argument(
        "--flip10",
        action="store_true",
        help="use the fixed ten-case diagnostic selection and output root",
    )
    command.add_argument("--max-concurrency", type=int, default=4)


def _add_expected_answer_run(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--expected-answer-run-sha256",
        type=_sha256_argument,
        required=True,
    )


def _add_provider_authorization(command: argparse.ArgumentParser) -> None:
    command.add_argument("--api-key-env", default="LITELLM_KEY")
    command.add_argument("--enable-provider", action="store_true")
    command.add_argument(
        "--authorized-provider-calls",
        type=int,
        default=0,
        help="exact number of distinct calls authorized for this sealed batch",
    )


def _add_judge_arguments(command: argparse.ArgumentParser) -> None:
    _add_expected_answer_run(command)
    command.add_argument("--dataset", type=Path, required=True)
    command.add_argument("--split", type=Path, default=DEFAULT_SPLIT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="seal the gold-blind dual-answer prompt population"
    )
    _add_source_arguments(preflight)

    answer = subparsers.add_parser(
        "answer", help="execute the exactly authorized Terra synthesis batch"
    )
    _add_source_arguments(answer)
    _add_provider_authorization(answer)

    answer_replay = subparsers.add_parser(
        "answer-replay", help="replay Terra synthesis journals with zero calls"
    )
    _add_source_arguments(answer_replay)
    _add_expected_answer_run(answer_replay)

    judge_preflight = subparsers.add_parser(
        "judge-preflight",
        help="verify synthesis before loading gold and sealing Sol prompts",
    )
    _add_source_arguments(judge_preflight)
    _add_judge_arguments(judge_preflight)

    judge = subparsers.add_parser(
        "judge", help="execute the exactly authorized post-hoc Sol batch"
    )
    _add_source_arguments(judge)
    _add_judge_arguments(judge)
    _add_provider_authorization(judge)

    judge_replay = subparsers.add_parser(
        "judge-replay", help="replay Sol journals and score ledger with zero calls"
    )
    _add_source_arguments(judge_replay)
    _add_judge_arguments(judge_replay)
    judge_replay.add_argument(
        "--expected-judge-sha256", type=_sha256_argument, required=True
    )
    return parser


def _result_projection(value: Any) -> dict[str, Any]:
    if hasattr(value, "answer_artifact"):
        return {
            "answer_run_sha256": value.answer_artifact.sha256,
            "checkpoint_hits": value.checkpoint_hits,
            "physical_provider_calls": value.physical_provider_calls,
            "runtime_ledger_sha256": value.runtime_ledger_artifact.sha256,
        }
    return {
        "answer_run_sha256": value.run_sha256,
        "answer_replay_sha256": value.replay_sha256,
        "physical_provider_calls": 0,
        "runtime_ledger_sha256": value.runtime_ledger_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if int(args.max_concurrency) < 1:
        raise ValueError("--max-concurrency must be positive")
    output_root, source_ordinals = _scope(args)

    if args.command in {"answer", "judge"}:
        _validate_provider_authorization(args)
    if args.command != "preflight":
        _require_flip10_gate(
            output_root=output_root,
            source_ordinals=source_ordinals,
        )

    if args.command == "preflight":
        from tools.matched_eval.synthesis import preflight_dual_answer_synthesis

        artifact = preflight_dual_answer_synthesis(
            **_source_arguments(
                args,
                output_root=output_root,
                source_ordinals=source_ordinals,
            )
        )
        result: dict[str, Any] = {
            "preflight_sha256": artifact.sha256,
            "physical_provider_calls": 0,
        }
        for key in (
            "logical_prompt_count",
            "required_authorized_provider_calls",
            "unique_prompt_count",
        ):
            if key in artifact.payload:
                result[key] = artifact.payload[key]
        if source_ordinals == FLIP10_ORDINALS:
            gate = _seal_flip10_gate(
                output_root=output_root,
                diagnostic_preflight=artifact,
            )
            result.update(
                {
                    "diagnostic_gate": str(gate.path),
                    "diagnostic_gate_sha256": gate.sha256,
                }
            )
    elif args.command == "answer":
        from tools.matched_eval.synthesis import run_dual_answer_synthesis

        run = run_dual_answer_synthesis(
            **_source_arguments(
                args,
                output_root=output_root,
                source_ordinals=source_ordinals,
            ),
            enable_provider=bool(args.enable_provider),
            authorized_provider_calls=int(args.authorized_provider_calls),
            api_key_env=str(args.api_key_env),
            max_concurrency=int(args.max_concurrency),
        )
        result = _result_projection(run)
    elif args.command == "answer-replay":
        from tools.matched_eval.synthesis import replay_dual_answer_synthesis

        plane = replay_dual_answer_synthesis(
            **_source_arguments(
                args,
                output_root=output_root,
                source_ordinals=source_ordinals,
            ),
            expected_run_sha256=str(args.expected_answer_run_sha256),
            max_concurrency=int(args.max_concurrency),
        )
        result = _result_projection(plane)
    else:
        # This call verifies the answer/replay pair and Terra journals before
        # the dataset path is supplied to any gold-bearing judge function.
        answer_plane = _load_synthesis_plane(
            args,
            output_root=output_root,
            source_ordinals=source_ordinals,
        )
        if args.command == "judge-preflight":
            from tools.matched_eval.judging import (
                preflight_verified_answer_plane_judge,
            )

            artifact = preflight_verified_answer_plane_judge(
                answer_plane=answer_plane,
                dataset_path=Path(args.dataset),
                split_path=Path(args.split),
                output_root=output_root,
            )
            result = {
                "judge_preflight_sha256": artifact.sha256,
                "physical_provider_calls": 0,
                "required_authorized_provider_calls": artifact.payload[
                    "required_authorized_provider_calls"
                ],
            }
        elif args.command == "judge":
            from tools.matched_eval.judging import (
                run_verified_answer_plane_judge,
            )

            judged = run_verified_answer_plane_judge(
                answer_plane=answer_plane,
                dataset_path=Path(args.dataset),
                split_path=Path(args.split),
                output_root=output_root,
                enable_provider=bool(args.enable_provider),
                authorized_provider_calls=int(args.authorized_provider_calls),
                api_key_env=str(args.api_key_env),
                max_concurrency=int(args.max_concurrency),
            )
            result = {
                "correct": judged.correct,
                "judge_sha256": judged.judge_artifact.sha256,
                "checkpoint_hits": judged.checkpoint_hits,
                "physical_provider_calls": judged.physical_provider_calls,
                "score_ledger_sha256": judged.score_ledger_artifact.sha256,
            }
        else:
            from tools.matched_eval.judging import (
                replay_verified_answer_plane_judge,
            )

            replay = replay_verified_answer_plane_judge(
                answer_plane=answer_plane,
                expected_judge_sha256=str(args.expected_judge_sha256),
                dataset_path=Path(args.dataset),
                split_path=Path(args.split),
                output_root=output_root,
                max_concurrency=int(args.max_concurrency),
            )
            result = {
                "correct": replay.correct,
                "judge_replay_sha256": replay.judge_artifact.sha256,
                "physical_provider_calls": 0,
                "score_ledger_replay_sha256": (
                    replay.score_ledger_artifact.sha256
                ),
            }

    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
