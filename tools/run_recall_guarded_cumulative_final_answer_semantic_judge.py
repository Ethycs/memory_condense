#!/usr/bin/env python3
"""Run or replay independent Sol judging of fixed-stage Terra answers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval.recall_guarded_cumulative_1m import (
    DEFAULT_SPLIT,
    STAGE_IDS,
    _atomic_write_json,
    _read_canonical_json,
    load_original_population,
)
from memory_condense.eval.recall_guarded_cumulative_final_answer import (
    FIXED_STAGE_ID,
    validate_final_answer_artifact,
)
from memory_condense.eval.recall_guarded_cumulative_final_answer_semantic_judge import (
    LOCKED_JUDGE_MAX_NEW_TOKENS,
    LOCKED_JUDGE_MODEL,
    build_final_answer_semantic_judge_campaign_binding,
    judge_recall_guarded_cumulative_final_answers,
)
from memory_condense.eval.recall_guarded_cumulative_population import (
    LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    build_locked_cumulative_population_identity,
    reconstruct_locked_cumulative_shard,
)
from memory_condense.eval.recall_guarded_cumulative_semantic_judge_runtime import (
    RecallGuardedCumulativeSemanticJudgeRuntime,
)
from memory_condense.ingest.loader import BenchmarkSample


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--expected-answers-sha256")
    parser.add_argument("--expected-retrieval-sha256")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--population",
        choices=("development-10q", "validation-shard-10q", "validation-100q"),
        required=True,
    )
    parser.add_argument(
        "--fixed-stage-id",
        choices=STAGE_IDS,
        default=FIXED_STAGE_ID,
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "run", "replay"),
        default="preflight",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--authorized-unique-calls",
        type=int,
        required=True,
        help="Must exactly equal the provider-free unique judge-prompt count.",
    )
    parser.add_argument("--api-key-env", default="LITELLM_KEY")
    return parser


def _load_scoring_population(
    args: argparse.Namespace,
    retrieval: dict[str, object],
) -> BenchmarkSample:
    if args.population == "development-10q":
        return load_original_population(args.dataset, args.split_manifest)
    if args.population == "validation-shard-10q":
        offset = retrieval.get("shard_offset")
        if type(offset) is not int:
            raise ValueError("validation shard retrieval omitted its offset")
        sample, shard = reconstruct_locked_cumulative_shard(
            args.dataset,
            args.split_manifest,
            dataset_sha256=LOCKED_LONGMEMEVAL_VALIDATION_PLAN.dataset_sha256,
            split_manifest_sha256=(
                LOCKED_LONGMEMEVAL_VALIDATION_PLAN.split_manifest_sha256
            ),
            split_name=LOCKED_LONGMEMEVAL_VALIDATION_PLAN.split,
            sample_offset=offset,
            target_tokens=LOCKED_LONGMEMEVAL_VALIDATION_PLAN.target_tokens,
            questions_per_shard=(
                LOCKED_LONGMEMEVAL_VALIDATION_PLAN.questions_per_shard
            ),
        )
        if shard.get("shard_identity_sha256") != retrieval.get(
            "shard_identity_sha256"
        ):
            raise ValueError("scoring source differs from the sealed retrieval shard")
        return sample
    samples, _shards, population = build_locked_cumulative_population_identity(
        args.dataset,
        args.split_manifest,
        plan=LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    )
    questions = [
        question
        for sample in samples
        for question in sample.questions
    ]
    if len(questions) != 100:
        raise RuntimeError("locked validation scoring population is not 100Q")
    return BenchmarkSample(
        sample_id=(
            "locked-validation-100q-"
            + str(population["population_identity_sha256"])
        ),
        questions=questions,
    )


def _default_output_path(
    answers_path: Path,
    *,
    retention_assay: bool,
    fixed_stage_id: str,
) -> Path:
    if retention_assay:
        return answers_path.with_name(
            f"validation10-retention-assay-{fixed_stage_id}-sol.json"
        )
    return answers_path.with_name("final-answer-semantic-judge-sol.json")


def run(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    retention_assay = args.population == "validation-shard-10q"
    if retention_assay and not args.expected_retrieval_sha256:
        raise ValueError(
            "validation-shard-10q requires --expected-retrieval-sha256"
        )
    answers_path = Path(args.answers).resolve()
    retrieval_path = Path(args.retrieval).resolve()
    answers, answers_sha = _read_canonical_json(answers_path)
    retrieval, retrieval_sha = _read_canonical_json(retrieval_path)
    if (
        args.expected_answers_sha256 is not None
        and args.expected_answers_sha256 != answers_sha
    ):
        raise ValueError("answers do not match --expected-answers-sha256")
    if (
        args.expected_retrieval_sha256 is not None
        and args.expected_retrieval_sha256 != retrieval_sha
    ):
        raise ValueError("retrieval does not match --expected-retrieval-sha256")
    # Seal and provenance validation precede opening the gold-bearing dataset.
    validate_final_answer_artifact(
        answers,
        retrieval=retrieval,
        artifact_sha256=answers_sha,
        retrieval_sha256=retrieval_sha,
        fixed_stage_id=args.fixed_stage_id,
    )
    sample = _load_scoring_population(args, retrieval)
    binding = build_final_answer_semantic_judge_campaign_binding(
        answers,
        retrieval=retrieval,
        sample=sample,
        artifact_sha256=answers_sha,
        retrieval_sha256=retrieval_sha,
        authorized_unique_calls=args.authorized_unique_calls,
        fixed_stage_id=args.fixed_stage_id,
        retention_assay=retention_assay,
    )
    if args.mode == "preflight":
        return binding, identity_sha256(binding)

    output = (
        Path(args.output).resolve()
        if args.output is not None
        else _default_output_path(
            answers_path,
            retention_assay=retention_assay,
            fixed_stage_id=args.fixed_stage_id,
        )
    )
    checkpoint = (
        Path(args.checkpoint_dir).resolve()
        if args.checkpoint_dir is not None
        else answers_path.with_name(
            "validation10-retention-assay-sol-calls"
            if retention_assay
            else "final-answer-semantic-judge-sol-calls"
        )
    )
    if retention_assay:
        checkpoint = checkpoint / identity_sha256(binding)
    secret = None
    if args.mode == "run":
        secret = os.environ.get(str(args.api_key_env), "").strip()
        if not secret:
            raise RuntimeError(
                f"provider API key environment variable is empty: "
                f"{args.api_key_env}"
            )
    with RecallGuardedCumulativeSemanticJudgeRuntime(
        checkpoint_dir=checkpoint,
        campaign_binding=binding,
        authorized_unique_calls=args.authorized_unique_calls,
        api_key=secret,
        caller_model=LOCKED_JUDGE_MODEL,
        max_new_tokens=LOCKED_JUDGE_MAX_NEW_TOKENS,
        replay_only=args.mode == "replay",
    ) as runtime:
        score = judge_recall_guarded_cumulative_final_answers(
            answers,
            retrieval=retrieval,
            sample=sample,
            artifact_sha256=answers_sha,
            retrieval_sha256=retrieval_sha,
            runtime=runtime,
            fixed_stage_id=args.fixed_stage_id,
            retention_assay=retention_assay,
        )
        session_physical_calls = runtime.usage["physical_calls"]
    digest = _atomic_write_json(output, score)
    _answers_after, answers_sha_after = _read_canonical_json(answers_path)
    _retrieval_after, retrieval_sha_after = _read_canonical_json(retrieval_path)
    if answers_sha_after != answers_sha or retrieval_sha_after != retrieval_sha:
        raise RuntimeError("judge input changed during scoring")
    print(
        f"Published {output} ({digest}); "
        f"session_physical_calls={session_physical_calls}",
        flush=True,
    )
    return score, digest


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = _parser().parse_args(argv)
    if type(args.authorized_unique_calls) is not int or (
        args.authorized_unique_calls < 1
    ):
        raise ValueError("--authorized-unique-calls must be positive")
    result, digest = run(args)
    if args.mode == "preflight":
        print(
            "Fixed-stage semantic-judge preflight passed: "
            f"stage={args.fixed_stage_id}; "
            f"questions={result['question_count']}; "
            f"unique_judge_calls={result['unique_judge_prompt_count']}; "
            f"binding={digest}",
            flush=True,
        )
        return 0
    gate = result["target_gate"]
    label = "retention10" if args.population == "validation-shard-10q" else "95%/min100"
    print(
        f"Semantic accuracy={gate['correct']}/{gate['questions']} "
        f"({gate['binary_accuracy']:.6f}); {label}={gate['status']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
