"""Run local or provider synthesis over the sealed 1M retrieval ladder."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from memory_condense.eval.recall_guarded_cumulative_1m import (
    DEFAULT_SPLIT,
    _atomic_write_json,
    _read_canonical_json,
    load_original_population,
)
from memory_condense.eval.recall_guarded_cumulative_provider_synthesis_runtime import (
    DEFAULT_CALLER_MODEL,
    RecallGuardedCumulativeProviderSynthesisRuntime,
)
from memory_condense.eval.recall_guarded_cumulative_synthesis import (
    SYNTHESIS_PROMPT_POLICY_SHA256,
    assemble_synthesis_artifact,
    normalize_fallback_abstentions,
    score_recall_guarded_synthesis,
    synthesize_question,
    validate_published_retrieval,
)
from memory_condense.eval.recall_guarded_cumulative_synthesis_runtime import (
    RecallGuardedCumulativeSynthesisRuntime,
)
from memory_condense.eval.reproducibility import implementation_sha256


DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-"
    "development-20260821/retrieval.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-"
    "llm-synthesis-development-20260821"
)
DEFAULT_MODEL = Path(".cache/models/Qwen3-0.6B")
PROVIDER_CAMPAIGN_FORMAT = (
    "memory-condense-recall-guarded-provider-synthesis-campaign-v1"
)


def _provider_campaign_binding(
    args: argparse.Namespace,
    *,
    retrieval_sha256: str,
) -> dict[str, Any]:
    """Bind every provider call to this exact gold-blind campaign."""

    return {
        "format": PROVIDER_CAMPAIGN_FORMAT,
        "retrieval_sha256": retrieval_sha256,
        "synthesis_implementation_sha256": implementation_sha256(),
        "synthesis_prompt_policy_sha256": SYNTHESIS_PROMPT_POLICY_SHA256,
        "request_policy": {
            "attempt_structured": bool(args.attempt_structured),
            "allow_attribution_fallback": _allow_attribution_fallback(args),
            "max_new_tokens": int(args.max_new_tokens),
        },
        "authorized_completion_calls": int(
            args.authorized_provider_calls
        ),
    }


def _synthesis_runtime(
    args: argparse.Namespace,
    *,
    retrieval_sha256: str | None = None,
) -> Any:
    """Construct the selected completion runtime without retaining secrets."""

    provider_model = getattr(args, "provider_model", None)
    if provider_model is not None:
        if retrieval_sha256 is None:
            raise ValueError("provider runtime requires the retrieval SHA-256")
        authorized_calls = int(
            getattr(args, "authorized_provider_calls", 0)
        )
        if authorized_calls < 1:
            raise ValueError(
                "--provider-model requires a positive "
                "--authorized-provider-calls budget"
            )
        api_key = os.environ.get("LITELLM_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "--provider-model requires LITELLM_KEY in the environment"
            )
        checkpoint_dir = getattr(args, "provider_checkpoint_dir", None)
        if checkpoint_dir is None:
            checkpoint_dir = args.output_root / "provider-calls"
        return RecallGuardedCumulativeProviderSynthesisRuntime(
            args.model_dir,
            api_key=api_key,
            caller_model=provider_model,
            max_new_tokens=args.max_new_tokens,
            gpu_memory=args.gpu_memory,
            checkpoint_dir=checkpoint_dir,
            campaign_binding=_provider_campaign_binding(
                args,
                retrieval_sha256=retrieval_sha256,
            ),
            authorized_completion_calls=authorized_calls,
        )
    return RecallGuardedCumulativeSynthesisRuntime(
        args.model_dir,
        max_new_tokens=args.max_new_tokens,
        gpu_memory=args.gpu_memory,
    )


def _allow_attribution_fallback(args: argparse.Namespace) -> bool:
    """Keep legacy local short-answer mode valid, but structured mode strict."""

    return bool(
        getattr(args, "allow_attribution_fallback", False)
        or not args.attempt_structured
    )


def _existing_parts(
    root: Path,
    *,
    question_count: int,
) -> tuple[list[dict[str, Any] | None], list[int]]:
    parts: list[dict[str, Any] | None] = []
    missing: list[int] = []
    for ordinal in range(question_count):
        path = root / "synthesis-parts" / f"q{ordinal:03d}.json"
        if path.is_file():
            part, _digest = _read_canonical_json(path)
            parts.append(part)
        else:
            parts.append(None)
            missing.append(ordinal)
    return parts, missing


def run_synthesis(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if (
        getattr(args, "provider_model", None) is not None
        and args.output_root.resolve() == DEFAULT_OUTPUT_ROOT.resolve()
    ):
        raise ValueError(
            "--provider-model requires a separate caller-specified --output-root"
        )
    retrieval, retrieval_sha256 = _read_canonical_json(args.retrieval)
    validate_published_retrieval(retrieval)
    questions = retrieval["questions"]
    parts, missing = _existing_parts(
        args.output_root, question_count=len(questions)
    )
    if missing:
        provider_model = getattr(args, "provider_model", None)
        print(
            (
                f"Loading {provider_model} with pinned local Qwen scoring"
                if provider_model is not None
                else "Loading pinned local Qwen"
            )
            + f" for {len(missing)} missing question part(s)",
            flush=True,
        )
        with _synthesis_runtime(
            args,
            retrieval_sha256=retrieval_sha256,
        ) as runtime:
            for ordinal in missing:
                source = questions[ordinal]
                question_id = source["question_id"]
                print(
                    f"Question {ordinal + 1}/{len(questions)} {question_id}",
                    flush=True,
                )
                part = synthesize_question(
                    source,
                    retrieval_sha256=retrieval_sha256,
                    runtime=runtime,
                    max_new_tokens=args.max_new_tokens,
                    allow_attribution_fallback=(
                        _allow_attribution_fallback(args)
                    ),
                    attempt_structured=args.attempt_structured,
                    progress=lambda message: print(f"  {message}", flush=True),
                )
                path = (
                    args.output_root
                    / "synthesis-parts"
                    / f"q{ordinal:03d}.json"
                )
                digest = _atomic_write_json(path, part)
                parts[ordinal] = part
                print(f"  published {digest}", flush=True)
    else:
        print("All synthesis question checkpoints already exist", flush=True)
    if any(part is None for part in parts):
        raise RuntimeError("synthesis part population is incomplete")
    artifact = assemble_synthesis_artifact(
        retrieval,
        retrieval_sha256=retrieval_sha256,
        question_parts=[part for part in parts if part is not None],
    )
    digest = _atomic_write_json(args.output_root / "synthesis.json", artifact)
    print(
        f"Published synthesis.json {digest}; "
        f"unique_calls={artifact['unique_synthesis_calls']}; "
        f"episodic_items={artifact['episodic_evidence_count']}",
        flush=True,
    )
    return artifact, digest


def run_score(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.dataset is None:
        raise ValueError("--dataset is required for score and all phases")
    synthesis_path = args.synthesis or (
        args.output_root / "synthesis-normalized.json"
    )
    if args.synthesis is None and not synthesis_path.is_file():
        synthesis_path = args.output_root / "synthesis.json"
    synthesis, synthesis_sha256 = _read_canonical_json(synthesis_path)
    sample = load_original_population(args.dataset, args.split)
    score = score_recall_guarded_synthesis(
        synthesis,
        sample=sample,
        synthesis_sha256=synthesis_sha256,
    )
    score_name = args.scores_name or (
        "scores-normalized.json"
        if synthesis_path.name == "synthesis-normalized.json"
        else "scores.json"
    )
    digest = _atomic_write_json(args.output_root / score_name, score)
    print(f"Published {score_name} {digest}", flush=True)
    for row in score["stage_aggregates"]:
        print(
            f"  {row['stage_id']}: EM={row['exact_matches']}/"
            f"{row['questions']}, mean_F1={row['mean_f1']:.6f}, "
            f"episodic_p(A)={row['mean_causal_answerability']:.6f}, "
            f"claim_p(A)={row['mean_claim_answerability']:.6f}",
            flush=True,
        )
    return score, digest


def run_normalize(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    source, source_sha256 = _read_canonical_json(
        args.output_root / "synthesis.json"
    )
    normalized = normalize_fallback_abstentions(
        source,
        source_synthesis_sha256=source_sha256,
    )
    digest = _atomic_write_json(
        args.output_root / "synthesis-normalized.json", normalized
    )
    print(
        f"Published synthesis-normalized.json {digest}; "
        f"normalized_stage_rows="
        f"{normalized['normalization']['normalized_stage_rows']}",
        flush=True,
    )
    return normalized, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "synthesize", "normalize", "score"),
        default="all",
    )
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--synthesis", type=Path)
    parser.add_argument("--scores-name")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory", default="6GiB")
    parser.add_argument(
        "--provider-model",
        nargs="?",
        const=DEFAULT_CALLER_MODEL,
        help=(
            "use the composite provider/local-scoring runtime; when MODEL is "
            f"omitted, use {DEFAULT_CALLER_MODEL}. Supply a separate "
            "--output-root for provider artifacts"
        ),
    )
    parser.add_argument(
        "--authorized-provider-calls",
        type=int,
        default=0,
        help=(
            "hard cap on distinct provider responses; provider synthesis "
            "refuses to start unless this is positive"
        ),
    )
    parser.add_argument(
        "--provider-checkpoint-dir",
        type=Path,
        help=(
            "immutable per-call journal directory (default: "
            "OUTPUT_ROOT/provider-calls)"
        ),
    )
    parser.add_argument(
        "--attempt-structured",
        action="store_true",
        help=(
            "try the strict nested JSON synthesis before the declared "
            "short-answer/forced-choice attribution path"
        ),
    )
    parser.add_argument(
        "--allow-attribution-fallback",
        action="store_true",
        help=(
            "permit a failed structured response to make a second, declared "
            "short-answer provider call; structured generation is fail-closed "
            "without this flag"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parser().parse_args(argv)
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.phase in {"all", "score"} and args.dataset is None:
        raise ValueError("--dataset is required for score and all phases")
    if args.phase in {"all", "synthesize"}:
        run_synthesis(args)
    if args.phase in {"all", "normalize"}:
        run_normalize(args)
    if args.phase in {"all", "score"}:
        run_score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
