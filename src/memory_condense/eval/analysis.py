"""Analysis helpers over saved eval results.

`ConversationResult.scores_by_position` has always been captured but nothing
read it. It is the data needed to show that memory value grows with
conversation depth (hypothesis H2 in docs/00 - Theory).

Everything here is a pure function over `EvalRunResult` — no plotting
dependency (matplotlib is not a project dependency), no network, no I/O beyond
`load_run`.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from pydantic import BaseModel, Field

from memory_condense.eval.schemas import (
    ConversationResult,
    EvalConfig,
    EvalRunResult,
    UsageStats,
)


class BinStats(BaseModel):
    """Aggregated scores for one turn-position bin."""

    bin_index: int
    n: int
    mean_score: float

    model_config = {"frozen": True}


class BinDelta(BaseModel):
    """Per-bin difference between two runs."""

    bin_index: int
    baseline_mean: float
    treatment_mean: float
    delta: float
    baseline_n: int
    treatment_n: int

    model_config = {"frozen": True}


class ConversationDelta(BaseModel):
    """Per-conversation difference between two runs."""

    filename: str
    baseline_mean: float
    treatment_mean: float
    delta: float
    baseline_n: int
    treatment_n: int

    model_config = {"frozen": True}


class ComparisonReport(BaseModel):
    """The k=0 vs k=N ablation, rendered as data."""

    baseline_label: str
    treatment_label: str
    bins: int

    baseline_mean_score: float
    treatment_mean_score: float
    delta_mean_score: float

    baseline_recall_at_4: float
    treatment_recall_at_4: float
    delta_recall_at_4: float

    baseline_usage: UsageStats = Field(default_factory=UsageStats)
    treatment_usage: UsageStats = Field(default_factory=UsageStats)
    delta_total_tokens: int = 0
    delta_elapsed_s: float = 0.0

    baseline_mean_context_tokens: float = 0.0
    treatment_mean_context_tokens: float = 0.0
    delta_mean_context_tokens: float = 0.0

    baseline_tokens_per_scored_turn: float = 0.0
    treatment_tokens_per_scored_turn: float = 0.0
    delta_tokens_per_scored_turn: float = 0.0

    baseline_bins: list[BinStats] = Field(default_factory=list)
    treatment_bins: list[BinStats] = Field(default_factory=list)
    bin_deltas: list[BinDelta] = Field(default_factory=list)
    by_conversation: list[ConversationDelta] = Field(default_factory=list)


def config_label(config: EvalConfig) -> str:
    """Short human-readable label for a run config."""
    c = config.chunker
    r = config.retrieval
    return f"chunk({c.min_tokens}-{c.max_tokens}) k={r.k} ef={r.ef_search}"


def load_run(path: str | Path) -> EvalRunResult:
    """Parse a saved eval_results JSON file into an EvalRunResult."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalRunResult.model_validate(data)


def _scores(cr: ConversationResult) -> list[float]:
    """Per-position scores for a conversation."""
    if cr.scores_by_position:
        return [float(s) for s in cr.scores_by_position]
    return [float(tr.score) for tr in cr.turn_results]


def _bin_index(position: int, n: int, bins: int) -> int:
    """Map a turn position within a conversation of `n` turns to a bin."""
    if n <= 0 or bins <= 0:
        return 0
    return min((position * bins) // n, bins - 1)


def binned_scores(result: EvalRunResult, bins: int = 5) -> list[BinStats]:
    """Split each conversation's scores into equal position bins.

    Bins are relative to each conversation's own length, so a short and a long
    conversation both contribute to every bin. Returns one BinStats per bin
    (including empty ones, with n=0 and mean_score=0.0).
    """
    if bins <= 0:
        return []

    totals = [0.0] * bins
    counts = [0] * bins

    for cr in result.conversations:
        scores = _scores(cr)
        n = len(scores)
        for position, score in enumerate(scores):
            b = _bin_index(position, n, bins)
            totals[b] += score
            counts[b] += 1

    return [
        BinStats(
            bin_index=b,
            n=counts[b],
            mean_score=(totals[b] / counts[b]) if counts[b] else 0.0,
        )
        for b in range(bins)
    ]


def compare_runs(
    baseline: EvalRunResult,
    treatment: EvalRunResult,
    bins: int = 5,
) -> ComparisonReport:
    """Compare two runs (the k=0 vs k=N ablation).

    `baseline` is the no-memory / weaker-retrieval run; `treatment` is the one
    under test. All deltas are treatment - baseline.
    """
    b_bins = binned_scores(baseline, bins=bins)
    t_bins = binned_scores(treatment, bins=bins)

    bin_deltas = [
        BinDelta(
            bin_index=i,
            baseline_mean=b.mean_score,
            treatment_mean=t.mean_score,
            delta=t.mean_score - b.mean_score,
            baseline_n=b.n,
            treatment_n=t.n,
        )
        for i, (b, t) in enumerate(zip(b_bins, t_bins))
    ]

    b_convos = {cr.filename: cr for cr in baseline.conversations}
    t_convos = {cr.filename: cr for cr in treatment.conversations}
    by_conversation: list[ConversationDelta] = []
    for filename in sorted(set(b_convos) | set(t_convos)):
        b_cr = b_convos.get(filename)
        t_cr = t_convos.get(filename)
        b_mean = b_cr.mean_score if b_cr else 0.0
        t_mean = t_cr.mean_score if t_cr else 0.0
        by_conversation.append(
            ConversationDelta(
                filename=filename,
                baseline_mean=b_mean,
                treatment_mean=t_mean,
                delta=t_mean - b_mean,
                baseline_n=len(b_cr.turn_results) if b_cr else 0,
                treatment_n=len(t_cr.turn_results) if t_cr else 0,
            )
        )

    return ComparisonReport(
        baseline_label=config_label(baseline.config),
        treatment_label=config_label(treatment.config),
        bins=bins,
        baseline_mean_score=baseline.aggregate_mean_score,
        treatment_mean_score=treatment.aggregate_mean_score,
        delta_mean_score=treatment.aggregate_mean_score
        - baseline.aggregate_mean_score,
        baseline_recall_at_4=baseline.aggregate_recall_at_4,
        treatment_recall_at_4=treatment.aggregate_recall_at_4,
        delta_recall_at_4=treatment.aggregate_recall_at_4
        - baseline.aggregate_recall_at_4,
        baseline_usage=baseline.usage,
        treatment_usage=treatment.usage,
        delta_total_tokens=treatment.usage.total_tokens - baseline.usage.total_tokens,
        delta_elapsed_s=treatment.usage.elapsed_s - baseline.usage.elapsed_s,
        baseline_mean_context_tokens=baseline.mean_context_tokens,
        treatment_mean_context_tokens=treatment.mean_context_tokens,
        delta_mean_context_tokens=treatment.mean_context_tokens
        - baseline.mean_context_tokens,
        baseline_tokens_per_scored_turn=baseline.tokens_per_scored_turn,
        treatment_tokens_per_scored_turn=treatment.tokens_per_scored_turn,
        delta_tokens_per_scored_turn=treatment.tokens_per_scored_turn
        - baseline.tokens_per_scored_turn,
        baseline_bins=b_bins,
        treatment_bins=t_bins,
        bin_deltas=bin_deltas,
        by_conversation=by_conversation,
    )


def ascii_curve(values: list[float], width: int = 60, height: int = 12) -> str:
    """Dependency-free text plot of a value series.

    Returns exactly `height` lines, each exactly `width` characters, joined by
    newlines. Values are stretched across the full width (nearest-neighbour) so
    a 5-point series still fills the plot.
    """
    if width <= 0 or height <= 0:
        return ""

    blank = "\n".join(" " * width for _ in range(height))
    if not values:
        return blank

    grid = [[" "] * width for _ in range(height)]

    lo = min(values)
    hi = max(values)
    span = hi - lo

    n = len(values)
    for x in range(width):
        # Nearest-neighbour resample onto the plot width.
        idx = 0 if width == 1 else min(n - 1, (x * n) // width)
        value = values[idx]
        if span == 0:
            row = (height - 1) // 2
        else:
            norm = (value - lo) / span
            row = height - 1 - int(round(norm * (height - 1)))
            row = max(0, min(height - 1, row))
        grid[row][x] = "*"

    return "\n".join("".join(row) for row in grid)


def to_csv(result: EvalRunResult) -> str:
    """One CSV row per scored turn, so the data can leave the tool."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        [
            "conversation",
            "position",
            "turn_index",
            "score",
            "context_tokens",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "retrieval_s",
            "elapsed_s",
        ]
    )
    for cr in result.conversations:
        for position, tr in enumerate(cr.turn_results):
            usage = tr.responder_usage + tr.judge_usage
            writer.writerow(
                [
                    cr.filename,
                    position,
                    tr.turn_index,
                    tr.score,
                    tr.context_tokens,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    f"{tr.retrieval_s:.4f}",
                    f"{usage.elapsed_s:.4f}",
                ]
            )
    return buf.getvalue()


def print_comparison(report: ComparisonReport) -> None:
    """Render the ablation as a readable table plus the ASCII curve."""
    print(f"\n{'=' * 72}")
    print("ABLATION")
    print(f"  baseline : {report.baseline_label}")
    print(f"  treatment: {report.treatment_label}")
    print(f"{'=' * 72}")

    rows = [
        ("Mean score", report.baseline_mean_score, report.treatment_mean_score, "{:.2f}"),
        ("Recall@4", report.baseline_recall_at_4, report.treatment_recall_at_4, "{:.1%}"),
        (
            "Total tokens",
            float(report.baseline_usage.total_tokens),
            float(report.treatment_usage.total_tokens),
            "{:.0f}",
        ),
        (
            "Tokens/turn",
            report.baseline_tokens_per_scored_turn,
            report.treatment_tokens_per_scored_turn,
            "{:.1f}",
        ),
        (
            "Ctx tokens",
            report.baseline_mean_context_tokens,
            report.treatment_mean_context_tokens,
            "{:.1f}",
        ),
        (
            "LLM seconds",
            report.baseline_usage.elapsed_s,
            report.treatment_usage.elapsed_s,
            "{:.1f}",
        ),
    ]
    print(f"{'metric':<14}  {'baseline':>12}  {'treatment':>12}  {'delta':>12}")
    print(f"{'-' * 56}")
    for name, base, treat, fmt in rows:
        delta = treat - base
        sign = "+" if delta >= 0 else ""
        print(
            f"{name:<14}  {fmt.format(base):>12}  {fmt.format(treat):>12}  "
            f"{sign + fmt.format(delta):>12}"
        )

    print(f"\nBy position bin (0 = start of conversation, {report.bins - 1} = end)")
    print(
        f"{'bin':>4}  {'baseline':>9}  {'treatment':>9}  {'delta':>8}  "
        f"{'n_base':>7}  {'n_treat':>7}"
    )
    print(f"{'-' * 52}")
    for bd in report.bin_deltas:
        sign = "+" if bd.delta >= 0 else ""
        print(
            f"{bd.bin_index:>4}  {bd.baseline_mean:>9.2f}  {bd.treatment_mean:>9.2f}  "
            f"{sign + f'{bd.delta:.2f}':>8}  {bd.baseline_n:>7}  {bd.treatment_n:>7}"
        )

    treatment_values = [b.mean_score for b in report.treatment_bins]
    baseline_values = [b.mean_score for b in report.baseline_bins]
    delta_values = [bd.delta for bd in report.bin_deltas]

    if treatment_values:
        print("\nTreatment score vs position bin")
        print(ascii_curve(treatment_values))
    if baseline_values:
        print("\nBaseline score vs position bin")
        print(ascii_curve(baseline_values))
    if delta_values:
        print("\nDelta (treatment - baseline) vs position bin")
        print(ascii_curve(delta_values))

    print("\nBy conversation")
    print(f"{'conversation':<34}  {'baseline':>9}  {'treatment':>9}  {'delta':>8}")
    print(f"{'-' * 66}")
    for cd in report.by_conversation:
        sign = "+" if cd.delta >= 0 else ""
        print(
            f"{cd.filename[:34]:<34}  {cd.baseline_mean:>9.2f}  "
            f"{cd.treatment_mean:>9.2f}  {sign + f'{cd.delta:.2f}':>8}"
        )
