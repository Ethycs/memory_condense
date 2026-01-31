"""CLI entry point for the evaluation pipeline.

Usage:
    pixi run python -m memory_condense.eval --conversation-dir <path>
    pixi run python -m memory_condense.eval --conversation-dir <path> --sweep
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from memory_condense.eval.report import (
    print_run_summary,
    print_sweep_table,
    save_run_result,
    save_sweep_report,
)
from memory_condense.eval.runner import run_eval
from memory_condense.eval.schemas import ChunkerConfig, EvalConfig, RetrievalConfig
from memory_condense.eval.sweep import run_sweep
from memory_condense.loader import load_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate memory_condense retrieval via conversation replay"
    )
    parser.add_argument(
        "--conversation-dir",
        required=True,
        help="Path to directory containing .txt/.md conversation files",
    )
    parser.add_argument(
        "--judge-model",
        default="anthropic/claude-3-5-haiku-20241022",
        help="LLM model for judging",
    )
    parser.add_argument(
        "--responder-model",
        default="anthropic/claude-3-5-haiku-20241022",
        help="LLM model for response generation",
    )
    parser.add_argument(
        "--results-dir", default="./eval_results", help="Output directory"
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Limit number of conversations to evaluate",
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=4,
        help="Number of recent turns to include in context",
    )

    # Single run params
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=50)

    # Sweep mode
    parser.add_argument(
        "--sweep", action="store_true", help="Run full parameter sweep"
    )

    args = parser.parse_args()

    # Load conversations
    print(f"Loading conversations from {args.conversation_dir}...")
    conversations = load_directory(args.conversation_dir)
    if not conversations:
        print("No conversations found.")
        sys.exit(1)
    print(f"Found {len(conversations)} conversations")

    config = EvalConfig(
        chunker=ChunkerConfig(min_tokens=args.min_tokens, max_tokens=args.max_tokens),
        retrieval=RetrievalConfig(k=args.k, ef_search=args.ef_search),
        judge_model=args.judge_model,
        responder_model=args.responder_model,
        conversation_dir=args.conversation_dir,
        results_dir=args.results_dir,
        max_conversations=args.max_conversations,
        recent_window=args.recent_window,
    )

    if args.sweep:
        report = run_sweep(config, conversations)
        print_sweep_table(report)
        path = save_sweep_report(report, args.results_dir)
        print(f"\nSweep report saved to {path}")
    else:
        print(f"\nRunning single eval...")
        result = run_eval(config, conversations)
        print_run_summary(result)
        path = save_run_result(result, args.results_dir)
        print(f"\nResult saved to {path}")


if __name__ == "__main__":
    main()
