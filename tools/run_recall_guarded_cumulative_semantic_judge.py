"""Run or replay independent Sol judging over a sealed S1--S3 synthesis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval.recall_guarded_cumulative_1m import (
    DEFAULT_SPLIT,
    _atomic_write_json,
    _read_canonical_json,
    load_original_population,
)
from memory_condense.eval.recall_guarded_cumulative_semantic_judge import (
    DEFAULT_RESPONDER_PROMPT_CAP,
    LOCKED_JUDGE_MAX_NEW_TOKENS,
    build_semantic_judge_campaign_binding,
    judge_recall_guarded_cumulative_synthesis,
)
from memory_condense.eval.recall_guarded_cumulative_semantic_judge_runtime import (
    DEFAULT_JUDGE_MAX_NEW_TOKENS,
    DEFAULT_JUDGE_MODEL,
    RecallGuardedCumulativeSemanticJudgeRuntime,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--mode",
        choices=("preflight", "run", "replay"),
        default="replay",
        help=(
            "preflight validates without creating journals, run may contact "
            "the gateway, and replay refuses every cache miss"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--authorized-unique-calls",
        type=int,
        required=True,
        help=(
            "exact hard cap on distinct judge prompts; it must equal the "
            "fully deduplicated population"
        ),
    )
    parser.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--responder-prompt-cap",
        type=int,
        default=DEFAULT_RESPONDER_PROMPT_CAP,
    )
    return parser


def run(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    synthesis, synthesis_sha256 = _read_canonical_json(args.synthesis)
    retrieval, retrieval_sha256 = _read_canonical_json(args.retrieval)
    sample = load_original_population(args.dataset, args.split)
    binding = build_semantic_judge_campaign_binding(
        synthesis,
        retrieval=retrieval,
        sample=sample,
        synthesis_sha256=synthesis_sha256,
        retrieval_sha256=retrieval_sha256,
        responder_prompt_cap=args.responder_prompt_cap,
        authorized_unique_calls=args.authorized_unique_calls,
    )
    if args.mode == "preflight":
        digest = identity_sha256(binding)
        formal_gate = (
            "eligible"
            if binding["responder_output_reserve_protocol_eligible"]
            else "protocol_ineligible"
        )
        print(
            "Artifact preflight passed; "
            f"logical_judgments={binding['logical_judgment_count']}; "
            f"unique_prompts={binding['unique_judge_prompt_count']}; "
            f"max_responder_prompt={binding['max_responder_prompt_token_proxy']}; "
            f"formal_gate={formal_gate}; "
            f"binding={digest}",
            flush=True,
        )
        return binding, digest
    output = args.output or args.synthesis.with_name("semantic-judge-sol.json")
    checkpoint_dir = args.checkpoint_dir or args.synthesis.with_name(
        "semantic-judge-sol-calls"
    )
    api_key = None
    if args.mode == "run":
        api_key = os.environ.get("LITELLM_KEY", "").strip()
        if not api_key:
            raise ValueError("run mode requires LITELLM_KEY in the environment")
    with RecallGuardedCumulativeSemanticJudgeRuntime(
        checkpoint_dir=checkpoint_dir,
        campaign_binding=binding,
        authorized_unique_calls=args.authorized_unique_calls,
        api_key=api_key,
        caller_model=args.judge_model,
        max_new_tokens=args.judge_max_new_tokens,
        replay_only=args.mode == "replay",
    ) as runtime:
        score = judge_recall_guarded_cumulative_synthesis(
            synthesis,
            retrieval=retrieval,
            sample=sample,
            synthesis_sha256=synthesis_sha256,
            retrieval_sha256=retrieval_sha256,
            runtime=runtime,
            responder_prompt_cap=args.responder_prompt_cap,
        )
        session_usage = dict(runtime.usage)
    digest = _atomic_write_json(output, score)
    print(
        f"Published {output} {digest}; "
        f"unique_prompts={score['unique_judge_prompt_count']}; "
        f"session_physical_calls={session_usage['physical_calls']}",
        flush=True,
    )
    for row in score["stage_aggregates"]:
        print(
            f"  {row['stage_id']}: semantic={row['correct']}/"
            f"{row['questions']} ({row['binary_accuracy']:.6f}); "
            f"95%/min100={row['status']}",
            flush=True,
        )
    print(
        f"Target gate: {score['target_gate']['status']}; "
        f"responder local prompt cap: "
        f"{score['responder_prompt_cap_diagnostics']['local_prompt_cap_status']}",
        flush=True,
    )
    return score, digest


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parser().parse_args(argv)
    if args.authorized_unique_calls < 1:
        raise ValueError("--authorized-unique-calls must be positive")
    if args.judge_max_new_tokens != LOCKED_JUDGE_MAX_NEW_TOKENS:
        raise ValueError(
            "--judge-max-new-tokens must equal the locked official judge "
            f"allowance ({LOCKED_JUDGE_MAX_NEW_TOKENS})"
        )
    if args.responder_prompt_cap != DEFAULT_RESPONDER_PROMPT_CAP:
        raise ValueError(
            "--responder-prompt-cap must equal the locked protocol cap "
            f"({DEFAULT_RESPONDER_PROMPT_CAP})"
        )
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
