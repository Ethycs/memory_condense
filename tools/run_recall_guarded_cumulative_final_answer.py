#!/usr/bin/env python3
"""Run or replay the locked fixed-S1 Terra final-answer campaign."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval.recall_guarded_cumulative_1m import (
    _atomic_write_json,
    _read_canonical_json,
)
from memory_condense.eval.recall_guarded_cumulative_final_answer import (
    answer_recall_guarded_cumulative_stage,
    build_final_answer_campaign_binding,
    final_answer_prompt_population,
)
from memory_condense.eval.recall_guarded_cumulative_final_answer_runtime import (
    RecallGuardedCumulativeFinalAnswerRuntime,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Answer exactly the preregistered direct-episode cumulative stage "
            "with the zero-retry, 256-token Terra responder"
        )
    )
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--authorized-provider-calls",
        type=int,
        required=True,
        help="Must exactly equal the provider-free unique prompt preflight.",
    )
    parser.add_argument("--api-key-env", default="LITELLM_KEY")
    parser.add_argument(
        "--mode",
        choices=("preflight", "run", "replay"),
        default="preflight",
        help=(
            "preflight performs no writes/calls, run may contact Terra, and "
            "replay refuses every missing journal"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    retrieval_path = Path(args.retrieval).resolve()
    retrieval, retrieval_sha = _read_canonical_json(retrieval_path)

    # This full-population validation happens before output-root or checkpoint
    # creation, so a bad late prompt cannot leave an earlier paid call behind.
    prompts = final_answer_prompt_population(
        retrieval,
        retrieval_sha256=retrieval_sha,
    )
    unique_calls = len({identity_sha256(list(prompt)) for prompt in prompts})
    if (
        type(args.authorized_provider_calls) is not int
        or args.authorized_provider_calls != unique_calls
    ):
        raise ValueError(
            "--authorized-provider-calls must exactly equal the preflight "
            f"unique-prompt count ({args.authorized_provider_calls} != "
            f"{unique_calls})"
        )
    campaign = build_final_answer_campaign_binding(
        retrieval,
        retrieval_sha256=retrieval_sha,
        authorized_unique_calls=unique_calls,
    )
    if args.mode == "preflight":
        return campaign, identity_sha256(campaign)
    secret = None
    if args.mode == "run":
        secret = os.environ.get(str(args.api_key_env), "").strip()
        if not secret:
            raise RuntimeError(
                f"provider API key environment variable is empty: "
                f"{args.api_key_env}"
            )

    output_root = Path(args.output_root).resolve()
    checkpoint_dir = output_root / "final-answer-calls"
    with RecallGuardedCumulativeFinalAnswerRuntime(
        checkpoint_dir=checkpoint_dir,
        campaign_binding=campaign,
        prompt_population=prompts,
        authorized_unique_calls=unique_calls,
        api_key=secret,
        replay_only=args.mode == "replay",
    ) as runtime:
        artifact = answer_recall_guarded_cumulative_stage(
            retrieval,
            retrieval_sha256=retrieval_sha,
            runtime=runtime,
        )
    artifact_path = output_root / "final-answers.json"
    digest = _atomic_write_json(artifact_path, artifact)
    _retrieval_after, retrieval_sha_after = _read_canonical_json(retrieval_path)
    if retrieval_sha_after != retrieval_sha:
        raise RuntimeError("retrieval changed during final-answer generation")
    return artifact, digest


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = _parser().parse_args(argv)
    artifact, digest = run(args)
    if args.mode == "preflight":
        print(
            "Fixed-stage final-answer preflight passed: "
            f"questions={artifact['question_count']}; "
            f"unique_provider_calls={artifact['unique_provider_prompt_count']}; "
            f"binding={digest}",
            flush=True,
        )
        return 0
    print(
        "Fixed-stage final answers published: "
        f"{Path(args.output_root).resolve() / 'final-answers.json'} "
        f"({digest}); questions={artifact['question_count']}; "
        f"unique_provider_calls={artifact['unique_provider_prompt_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
