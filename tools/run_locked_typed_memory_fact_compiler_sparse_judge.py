#!/usr/bin/env python3
"""Seal changed-only Sol judge prompts for sparse v3 fact answers.

The complete 24-row answer run is verified before any score-plane artifact is
opened.  Twenty predictions are byte-identical to parents already judged
incorrect in the sealed miss-27 authority, so only the four changed predictions
are sent to Sol.  Provider, materialization, and replay remain the existing
common judge runner's responsibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from tools import (  # noqa: E402
    run_locked_typed_memory_fact_compiler_sparse as sparse_cli,
)
from tools.matched_eval import judging, live  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    require_sha256,
)
from tools.matched_eval.typed_memory_final_judging import (  # noqa: E402
    PREFLIGHT_NAME,
    load_locked_typed_final_gold,
    preflight_projection,
)
from tools.run_locked_query_answer_judge import DEFAULT_DATASET  # noqa: E402
from tools.run_matched_eval_spine import DEFAULT_SPLIT  # noqa: E402


FORMAT = "memory-condense-locked-typed-fact-sparse-changed4-sol-judge-adapter-v1"
CHANGED_ORDINALS = (36, 43, 54, 81)
EXPECTED_ANSWER_PREFLIGHT_SHA256 = (
    "6e6fd12b86d11be3b1d0a948ead2ff0e51afa70ea36e2ca8420ecd53b08574d9"
)
EXPECTED_ANSWER_RUN_SHA256 = (
    "7ac80bfad4fbeabe43300fa706f6b0b10379140dabfbd643fbfbec4522a27765"
)
EXPECTED_ANSWER_REPLAY_SHA256 = (
    "6e25ac40c17c0bac90014b2378b5fa03d3a0c858e4d4a2010a0f09a5344e96f0"
)

DEFAULT_SPARSE_ROOT = sparse_cli.DEFAULT_OUTPUT
DEFAULT_JUDGE_OUTPUT = DEFAULT_SPARSE_ROOT / "sol-judge-sparse-v3-changed4"
DEFAULT_PARENT_JUDGE = (
    Path(__file__).resolve().parents[1]
    / "eval_results"
    / "matched_eval_100"
    / "typed-memory-final-v3-posthoc-miss27-recovery1"
    / "sol-judge-v1"
    / "typed-final-semantic-judge-sol-v1.json"
)
EXPECTED_PARENT_JUDGE_SHA256 = (
    "56afe080c630bf2575fa44207b50bfcdaab9e66a4647fd93a8ff5499e144ac62"
)


class LockedTypedFactSparseJudgeAdapterError(MatchedEvalContractError):
    """The sparse answer seam or selected-subset judge projection changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedTypedFactSparseJudgeAdapterError(message)


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    # This complete answer/replay verification is deliberately the first
    # operation.  The benchmark dataset is not opened until it succeeds.
    run, replay, source_rows = sparse_cli.read_verified_sparse_answer_run(
        args.sparse_root,
        expected_preflight_sha256=require_sha256(
            args.expected_answer_preflight_sha256,
            "expected sparse answer preflight",
        ),
        expected_run_sha256=require_sha256(
            args.expected_answer_run_sha256,
            "expected sparse answer run",
        ),
        expected_replay_sha256=require_sha256(
            args.expected_answer_replay_sha256,
            "expected sparse answer replay",
        ),
    )
    _require(
        len(source_rows) == sparse_cli.SUBSET_QUESTION_COUNT
        and tuple(row.get("ordinal") for row in source_rows)
        == sparse_cli.REMAINING_ORDINALS,
        "sparse judge source population changed",
    )

    # This is the score-plane authority that made the remaining-24 selection
    # possible.  It is opened only after the new answer bytes are immutable.
    parent_judge = read_sealed_json(args.parent_judge)
    _require(
        parent_judge.sha256
        == require_sha256(
            args.expected_parent_judge_sha256,
            "expected miss-27 parent judge",
        ),
        "miss-27 parent judge SHA-256 changed",
    )
    parent_questions = parent_judge.payload.get("questions")
    _require(
        type(parent_questions) is list and len(parent_questions) == 27,
        "miss-27 parent judge population changed",
    )
    authority_by_ordinal = {
        row.get("ordinal"): row
        for row in parent_questions
        if type(row) is dict
    }
    for source in source_rows:
        authority = authority_by_ordinal.get(source.get("ordinal"))
        _require(
            type(authority) is dict
            and authority.get("correct") is False
            and authority.get("prediction_sha256")
            == source.get("parent_prediction_sha256"),
            "sparse parent does not match a sealed incorrect miss prediction",
        )
    changed_rows = tuple(
        row for row in source_rows if row.get("changed_from_parent") is True
    )
    _require(
        tuple(row.get("ordinal") for row in changed_rows) == CHANGED_ORDINALS,
        "sparse changed-only judge population changed",
    )

    # Gold opens only after the answer plane is fixed and replay-verified.
    gold_rows, gold_sha256 = load_locked_typed_final_gold(
        dataset_path=args.dataset,
        split_path=args.split,
        source_rows=changed_rows,
        allow_subset=True,
    )
    payload, _prompts = preflight_projection(
        run_artifact=run,
        replay_artifact_sha256=replay.sha256,
        source_rows=changed_rows,
        gold_rows=gold_rows,
        gold_population_sha256=gold_sha256,
        mode="selected_subset",
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    prompt_rows = payload.get("prompt_rows")
    _require(
        payload.get("judge_mode") == "selected_subset"
        and payload.get("selected_question_count") == len(CHANGED_ORDINALS)
        and payload.get("required_authorized_provider_calls")
        == len(CHANGED_ORDINALS)
        and type(prompt_rows) is list
        and tuple(row.get("ordinal") for row in prompt_rows)
        == CHANGED_ORDINALS,
        "sparse common Sol judge projection changed",
    )
    payload.update(
        {
            "changed_ordinals": list(CHANGED_ORDINALS),
            "inherited_incorrect_count": (
                sparse_cli.SUBSET_QUESTION_COUNT - len(CHANGED_ORDINALS)
            ),
            "parent_judge_artifact_sha256": parent_judge.sha256,
            "source_population_question_count": sparse_cli.SUBSET_QUESTION_COUNT,
        }
    )
    artifact, created = publish_sealed_json(
        Path(args.judge_output_root) / PREFLIGHT_NAME,
        payload,
    )
    return {
        "answer_replay_sha256": replay.sha256,
        "answer_run_sha256": run.sha256,
        "created": created,
        "format": FORMAT,
        "gold_loaded": True,
        "judge_mode": "selected_subset",
        "judge_preflight_sha256": artifact.sha256,
        "physical_provider_calls": 0,
        "inherited_incorrect_count": (
            sparse_cli.SUBSET_QUESTION_COUNT - len(CHANGED_ORDINALS)
        ),
        "parent_judge_sha256": parent_judge.sha256,
        "required_authorized_provider_calls": len(CHANGED_ORDINALS),
        "selected_ordinals": list(CHANGED_ORDINALS),
        "selected_question_count": len(CHANGED_ORDINALS),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument(
        "--sparse-root", type=Path, default=DEFAULT_SPARSE_ROOT
    )
    preflight.add_argument(
        "--judge-output-root", type=Path, default=DEFAULT_JUDGE_OUTPUT
    )
    preflight.add_argument(
        "--expected-answer-preflight-sha256",
        default=EXPECTED_ANSWER_PREFLIGHT_SHA256,
    )
    preflight.add_argument(
        "--expected-answer-run-sha256",
        default=EXPECTED_ANSWER_RUN_SHA256,
    )
    preflight.add_argument(
        "--expected-answer-replay-sha256",
        default=EXPECTED_ANSWER_REPLAY_SHA256,
    )
    preflight.add_argument(
        "--parent-judge", type=Path, default=DEFAULT_PARENT_JUDGE
    )
    preflight.add_argument(
        "--expected-parent-judge-sha256",
        default=EXPECTED_PARENT_JUDGE_SHA256,
    )
    preflight.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    preflight.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    preflight.add_argument("--model", default=judging.DEFAULT_SOL_GATEWAY_MODEL)
    preflight.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)
    preflight.add_argument("--max-concurrency", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _preflight(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_JUDGE_OUTPUT",
    "DEFAULT_PARENT_JUDGE",
    "DEFAULT_SPARSE_ROOT",
    "CHANGED_ORDINALS",
    "EXPECTED_ANSWER_PREFLIGHT_SHA256",
    "EXPECTED_ANSWER_REPLAY_SHA256",
    "EXPECTED_ANSWER_RUN_SHA256",
    "EXPECTED_PARENT_JUDGE_SHA256",
    "FORMAT",
    "LockedTypedFactSparseJudgeAdapterError",
    "main",
]
