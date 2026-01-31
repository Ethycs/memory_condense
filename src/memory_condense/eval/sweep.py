"""Parameter sweep over chunker and retrieval configs."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import product

from memory_condense.eval.runner import run_eval
from memory_condense.eval.schemas import (
    ChunkerConfig,
    EvalConfig,
    EvalRunResult,
    RetrievalConfig,
    SweepReport,
)

DEFAULT_CHUNKER_GRID = {
    "min_tokens": [80, 120, 180],
    "max_tokens": [200, 300, 400],
}

DEFAULT_RETRIEVAL_GRID = {
    "k": [5, 10, 15],
    "ef_search": [50, 100],
}


def generate_configs(
    base_config: EvalConfig,
    chunker_grid: dict | None = None,
    retrieval_grid: dict | None = None,
) -> list[EvalConfig]:
    """Generate all valid config combinations from parameter grids."""
    cg = chunker_grid or DEFAULT_CHUNKER_GRID
    rg = retrieval_grid or DEFAULT_RETRIEVAL_GRID

    configs: list[EvalConfig] = []

    chunker_combos = list(
        product(cg.get("min_tokens", [120]), cg.get("max_tokens", [250]))
    )
    retrieval_combos = list(
        product(rg.get("k", [10]), rg.get("ef_search", [50]))
    )

    for min_tok, max_tok in chunker_combos:
        if min_tok >= max_tok:
            continue  # invalid combo

        for k, ef in retrieval_combos:
            configs.append(
                EvalConfig(
                    chunker=ChunkerConfig(min_tokens=min_tok, max_tokens=max_tok),
                    retrieval=RetrievalConfig(k=k, ef_search=ef),
                    judge_model=base_config.judge_model,
                    responder_model=base_config.responder_model,
                    conversation_dir=base_config.conversation_dir,
                    results_dir=base_config.results_dir,
                    max_conversations=base_config.max_conversations,
                    recent_window=base_config.recent_window,
                )
            )

    return configs


def run_sweep(
    base_config: EvalConfig,
    conversations: dict[str, list[tuple[str, str]]],
    chunker_grid: dict | None = None,
    retrieval_grid: dict | None = None,
) -> SweepReport:
    """Run the full parameter sweep.

    Outer loop: chunker configs (expensive — requires re-embedding).
    Inner loop: retrieval configs (cheap — just re-query).

    Note: current implementation re-ingests per config for simplicity.
    Future optimization: share embedding across retrieval configs
    with the same chunker config.
    """
    configs = generate_configs(base_config, chunker_grid, retrieval_grid)
    print(f"Running sweep with {len(configs)} configurations...")

    runs: list[EvalRunResult] = []
    for i, config in enumerate(configs):
        c = config.chunker
        r = config.retrieval
        print(
            f"\n=== Config {i + 1}/{len(configs)}: "
            f"chunk({c.min_tokens}-{c.max_tokens}) "
            f"retrieval(k={r.k}, ef={r.ef_search}) ==="
        )
        result = run_eval(config, conversations)
        runs.append(result)
        print(
            f"    Score: {result.aggregate_mean_score:.2f} "
            f"| Recall@4: {result.aggregate_recall_at_4:.1%}"
        )

    # Find best config
    best = max(runs, key=lambda r: r.aggregate_mean_score) if runs else None

    return SweepReport(
        runs=runs,
        best_config=best.config if best else None,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
