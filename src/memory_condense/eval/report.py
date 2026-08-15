"""Output formatting for eval results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from memory_condense.eval.schemas import EvalRunResult, SweepReport


def save_run_result(result: EvalRunResult, output_dir: str | Path) -> Path:
    """Save a single run result as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    c = result.config.chunker
    r = result.config.retrieval
    mode = f"_hybrid{r.alpha:g}" if r.hybrid else ""
    filename = (
        f"eval_{c.min_tokens}-{c.max_tokens}_k{r.k}_ef{r.ef_search}{mode}_{timestamp}.json"
    )
    path = output_dir / filename

    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def save_sweep_report(report: SweepReport, output_dir: str | Path) -> Path:
    """Save full sweep report as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"sweep_{timestamp}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def print_run_summary(result: EvalRunResult) -> None:
    """Print a summary of a single eval run, including cost/latency."""
    c = result.config.chunker
    r = result.config.retrieval
    u = result.usage
    n_turns = sum(len(cr.turn_results) for cr in result.conversations)

    print(f"\n{'=' * 72}")
    print(f"Config: chunk({c.min_tokens}-{c.max_tokens}) k={r.k} ef={r.ef_search}")
    print(f"Models: responder={result.config.responder_model} judge={result.config.judge_model}")
    print(f"Mean Score:  {result.aggregate_mean_score:.2f}")
    print(f"Recall@4:    {result.aggregate_recall_at_4:.1%}")
    print(f"Scored turns:{n_turns:>6}")
    print(
        f"Tokens:      in={u.input_tokens} out={u.output_tokens} "
        f"cached={u.cache_read_input_tokens} total={u.total_tokens}"
    )
    print(f"Tokens/turn: {result.tokens_per_scored_turn:.1f}")
    print(f"Ctx tokens:  {result.mean_context_tokens:.1f} mean/turn")
    print(
        f"Latency:     {u.elapsed_s:.1f}s in {u.calls} LLM calls "
        f"| {result.total_elapsed_s:.1f}s wall"
    )
    print(f"{'=' * 72}")

    header = (
        f"  {'conversation':<32}  {'score':>5}  {'turns':>5}  "
        f"{'tokens':>8}  {'tok/turn':>8}  {'llm_s':>7}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for cr in result.conversations:
        n = len(cr.turn_results)
        per_turn = cr.usage.total_tokens / n if n else 0.0
        print(
            f"  {cr.filename[:32]:<32}  {cr.mean_score:>5.2f}  {n:>5}  "
            f"{cr.usage.total_tokens:>8}  {per_turn:>8.1f}  {cr.usage.elapsed_s:>7.1f}"
        )


def print_sweep_table(report: SweepReport) -> None:
    """Print a comparison table of all configs."""
    if not report.runs:
        print("No results to display.")
        return

    # Sort by score descending
    sorted_runs = sorted(
        report.runs, key=lambda r: r.aggregate_mean_score, reverse=True
    )

    # Header
    header = (
        f"{'#':>3}  {'min':>4}  {'max':>4}  {'k':>3}  {'ef':>4}  {'Score':>6}  "
        f"{'Recall@4':>9}  {'Convos':>6}  {'Tokens':>9}  {'Tok/Turn':>9}  "
        f"{'Ctx':>7}  {'LLM_s':>8}"
    )
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")

    for i, run in enumerate(sorted_runs):
        c = run.config.chunker
        r = run.config.retrieval
        best_marker = " *" if run.config == report.best_config else ""
        print(
            f"{i + 1:>3}  {c.min_tokens:>4}  {c.max_tokens:>4}  "
            f"{r.k:>3}  {r.ef_search:>4}  "
            f"{run.aggregate_mean_score:>6.2f}  "
            f"{run.aggregate_recall_at_4:>8.1%}  "
            f"{len(run.conversations):>6}  "
            f"{run.usage.total_tokens:>9}  "
            f"{run.tokens_per_scored_turn:>9.1f}  "
            f"{run.mean_context_tokens:>7.0f}  "
            f"{run.usage.elapsed_s:>8.1f}{best_marker}"
        )

    print(f"{'=' * len(header)}")

    if report.best_config:
        c = report.best_config.chunker
        r = report.best_config.retrieval
        print(
            f"\nBest: chunk({c.min_tokens}-{c.max_tokens}) "
            f"k={r.k} ef={r.ef_search}"
        )
