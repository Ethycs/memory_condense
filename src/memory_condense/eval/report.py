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
    filename = f"eval_{c.min_tokens}-{c.max_tokens}_k{r.k}_ef{r.ef_search}_{timestamp}.json"
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
    """Print a summary of a single eval run."""
    c = result.config.chunker
    r = result.config.retrieval
    print(f"\n{'=' * 60}")
    print(f"Config: chunk({c.min_tokens}-{c.max_tokens}) k={r.k} ef={r.ef_search}")
    print(f"Mean Score: {result.aggregate_mean_score:.2f}")
    print(f"Recall@4:   {result.aggregate_recall_at_4:.1%}")
    print(f"{'=' * 60}")

    for cr in result.conversations:
        print(f"  {cr.filename}: {cr.mean_score:.2f} ({len(cr.turn_results)} turns scored)")


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
    header = f"{'#':>3}  {'min':>4}  {'max':>4}  {'k':>3}  {'ef':>4}  {'Score':>6}  {'Recall@4':>9}  {'Convos':>6}"
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
            f"{len(run.conversations):>6}{best_marker}"
        )

    print(f"{'=' * len(header)}")

    if report.best_config:
        c = report.best_config.chunker
        r = report.best_config.retrieval
        print(
            f"\nBest: chunk({c.min_tokens}-{c.max_tokens}) "
            f"k={r.k} ef={r.ef_search}"
        )
