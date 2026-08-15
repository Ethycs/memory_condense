"""CLI entry point for the evaluation pipeline.

Four modes, selected by flags:

    # 1. Self-replay on your own exported conversations (the default)
    pixi run python -m memory_condense.eval --conversation-dir <path>

    # 2. Parameter sweep over chunker/retrieval settings
    pixi run python -m memory_condense.eval --conversation-dir <path> --sweep

    # 3. Public benchmark (LongMemEval / LoCoMo) QA probes
    pixi run python -m memory_condense.eval --benchmark-file longmemeval_oracle.json

    # 4. Offline analysis of two saved runs (no API calls, no cost)
    pixi run python -m memory_condense.eval --compare baseline.json treatment.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from memory_condense.eval.analysis import (
    compare_runs,
    load_run,
    print_comparison,
    to_csv,
)
from memory_condense.eval.benchmark import (
    build_judge_prompt,
    print_benchmark_summary,
    run_benchmark,
    save_benchmark_report,
)
from memory_condense.eval.judge import JUDGE_MAX_TOKENS
from memory_condense.eval.report import (
    print_run_summary,
    print_sweep_table,
    save_run_result,
    save_sweep_report,
)
from memory_condense.eval.runner import run_eval
from memory_condense.eval.schemas import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_RESPONDER_MODEL,
    ChunkerConfig,
    EvalConfig,
    RetrievalConfig,
)
from memory_condense.eval.sweep import run_sweep
from memory_condense.loader import load_benchmark, load_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m memory_condense.eval",
        description="Evaluate memory_condense retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    parser.add_argument(
        "--conversation-dir",
        help="Directory of .txt/.md conversation exports (self-replay mode)",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Run the full parameter sweep"
    )
    parser.add_argument(
        "--benchmark-file",
        help="LongMemEval/LoCoMo JSON or JSONL file (benchmark QA mode)",
    )
    parser.add_argument(
        "--benchmark-format",
        default="auto",
        choices=["auto", "longmemeval", "locomo"],
        help="Benchmark format (default: auto-detect)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "TREATMENT"),
        help="Compare two saved eval_results JSON files offline (no API calls)",
    )

    # Models — import the defaults so the CLI can never drift from the schema.
    parser.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL, help="LLM model for judging"
    )
    parser.add_argument(
        "--responder-model",
        default=DEFAULT_RESPONDER_MODEL,
        help="LLM model for response generation",
    )

    parser.add_argument(
        "--results-dir", default="./eval_results", help="Output directory"
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Limit number of conversations evaluated (self-replay mode)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of benchmark samples evaluated",
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=4,
        help="Number of recent turns to include in context",
    )
    parser.add_argument(
        "--use-judge",
        action="store_true",
        help="Also grade benchmark answers with an LLM judge (doubles API cost)",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="Write per-turn results as CSV (with --compare, writes the treatment run)",
    )

    # Retrieval / chunker params
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument("--k", type=int, default=10, help="Chunks retrieved (0 = no-memory baseline)")
    parser.add_argument("--ef-search", type=int, default=50)
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Blend BM25 lexical retrieval with dense (default: dense only)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Dense weight for --hybrid (1.0 = pure dense)",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        chunker=ChunkerConfig(min_tokens=args.min_tokens, max_tokens=args.max_tokens),
        retrieval=RetrievalConfig(
            k=args.k,
            ef_search=args.ef_search,
            hybrid=args.hybrid,
            alpha=args.alpha,
        ),
        judge_model=args.judge_model,
        responder_model=args.responder_model,
        conversation_dir=args.conversation_dir or args.benchmark_file or "",
        results_dir=args.results_dir,
        max_conversations=args.max_conversations,
        recent_window=args.recent_window,
    )


def _content(response) -> str:
    """The assistant text, or "" if the provider returned none.

    A refusal, a content filter, or a `max_tokens` stop before any visible text
    all yield ``content=None``. Reaching ``.strip()`` on that raises
    ``AttributeError`` deep in a paid run, after every preceding call has
    already been billed.
    """
    try:
        return (response.choices[0].message.content or "").strip()
    except (AttributeError, IndexError, TypeError):
        return ""


def _make_answer_fn(model: str):
    """Answer a benchmark question. Short, deterministic answers — F1/EM depend on it."""
    import litellm

    def answer_fn(messages: list[dict[str, str]]) -> str:
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
            num_retries=5,
        )
        return _content(response)

    return answer_fn


def _make_judge_fn(model: str):
    """Semantic-equivalence grading, for answers that F1 scores unfairly.

    ``max_tokens`` is JUDGE_MAX_TOKENS for the reason spelled out in
    ``judge.py``: the default judge is Sonnet 5, which runs adaptive thinking,
    and ``max_tokens`` caps thinking + visible text together. A tight 256 spends
    the whole budget on thinking and returns an empty verdict — which this path
    then scored as INCORRECT for every answer. The replay judge got this fix;
    this one did not, so it is deliberately expressed as the same constant.
    """
    import litellm

    def judge_fn(question: str, gold: str, prediction: str) -> tuple[bool, str]:
        response = litellm.completion(
            model=model,
            messages=build_judge_prompt(question, gold, prediction),
            max_tokens=JUDGE_MAX_TOKENS,
            num_retries=5,
        )
        text = _content(response)
        return text.upper().startswith("CORRECT"), text

    return judge_fn


def run_compare(args: argparse.Namespace) -> None:
    baseline_path, treatment_path = args.compare
    baseline = load_run(baseline_path)
    treatment = load_run(treatment_path)
    report = compare_runs(baseline, treatment)
    print_comparison(report)

    if args.csv:
        Path(args.csv).write_text(to_csv(treatment), encoding="utf-8")
        print(f"\nPer-turn CSV written to {args.csv}")


def run_benchmark_mode(args: argparse.Namespace) -> None:
    print(f"Loading benchmark from {args.benchmark_file}...")
    samples = load_benchmark(args.benchmark_file, args.benchmark_format)
    if not samples:
        print("No benchmark samples found.")
        sys.exit(1)

    questions = sum(len(s.questions) for s in samples)
    print(f"Loaded {len(samples)} samples / {questions} questions")

    if args.max_samples is None:
        print(
            "\nWARNING: no --max-samples set. A full run ingests every haystack "
            "through bge-m3 and makes one LLM call per question"
            f"{' (doubled by --use-judge)' if args.use_judge else ''}. "
            "Start with --max-samples 10.\n"
        )

    config = config_from_args(args)
    result = run_benchmark(
        samples,
        config,
        answer_fn=_make_answer_fn(args.responder_model),
        judge_fn=_make_judge_fn(args.judge_model) if args.use_judge else None,
        max_samples=args.max_samples,
        # Label the run with the dataset, not the --benchmark-format flag, which
        # defaults to "auto" and would name every report benchmark_auto_*.json.
        benchmark=Path(args.benchmark_file).stem,
        verbose=True,
    )
    print_benchmark_summary(result)
    path = save_benchmark_report(result, args.results_dir)
    print(f"\nBenchmark report saved to {path}")


def run_replay_mode(args: argparse.Namespace) -> None:
    print(f"Loading conversations from {args.conversation_dir}...")
    conversations = load_directory(args.conversation_dir)
    if not conversations:
        print("No conversations found.")
        sys.exit(1)
    print(f"Found {len(conversations)} conversations")

    config = config_from_args(args)

    if args.sweep:
        report = run_sweep(config, conversations)
        print_sweep_table(report)
        path = save_sweep_report(report, args.results_dir)
        print(f"\nSweep report saved to {path}")
        return

    print("\nRunning single eval...")
    result = run_eval(config, conversations)
    print_run_summary(result)
    path = save_run_result(result, args.results_dir)
    print(f"\nResult saved to {path}")

    if args.csv:
        Path(args.csv).write_text(to_csv(result), encoding="utf-8")
        print(f"Per-turn CSV written to {args.csv}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    modes = [bool(args.compare), bool(args.benchmark_file), bool(args.conversation_dir)]
    if sum(modes) > 1:
        parser.error(
            "--compare, --benchmark-file, and --conversation-dir are mutually exclusive"
        )

    if args.compare:
        run_compare(args)
    elif args.benchmark_file:
        run_benchmark_mode(args)
    elif args.conversation_dir:
        run_replay_mode(args)
    else:
        parser.error(
            "one of --conversation-dir, --benchmark-file, or --compare is required"
        )


if __name__ == "__main__":
    main()
