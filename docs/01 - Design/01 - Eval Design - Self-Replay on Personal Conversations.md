# Eval Design — Self-Replay on Personal Conversations

**Status**: CURRENT — realized in `src/memory_condense/eval/`
**Date**: 2026-08-14
**Supersedes**: the revision that listed "judge = responder model" and "no token-cost or latency instrumentation" as open validity issues — **both are resolved**
**Depends on**: `00 - Theory/00 - Retrieval-Weighted Context and Self-Replay Evaluation.md`

## Why self-replay *and* (now) public benchmarks

The original argument for self-replay over LoCoMo/LongMemEval still holds, but the framing has changed: a benchmark adapter now exists, so this is a choice of *primary* eval, not the only one available.

1. **Annotation cost is zero here.** Public benchmarks need labeled QA probes; self-replay uses the user's own exported Claude conversations as ground truth — every assistant turn is a free label.
2. **Personal relevance.** The system's target workload *is* these conversations. A benchmark drawn from the deployment distribution beats a synthetic one for tuning decisions.
3. **Benchmark skepticism.** LoCoMo numbers are widely cited and increasingly criticized; MemDelta showed headline gains on LongMemEval can flip sign under embedding changes alone (see `04 - Reference`). A private eval avoids overfitting to a contested leaderboard.
4. **The trade-off, previously stated as permanent, is now closable**: self-replay gives no external comparability. `loader.load_benchmark` + `eval/benchmark.py` implement the LongMemEval/LoCoMo QA-probe protocol (SQuAD-normalized token F1, exact match, optional semantic judge, per-category breakdown). **known rough edge / honest status: the harness has never been run.** There are still zero common-benchmark numbers, so no competitiveness claim is possible yet — see the Decision Point in `06 - Roadmaps/00`.

The intended division of labour: **self-replay tunes** (cheap, personal, matched-pair deltas), **benchmarks compare** (expensive, external, absolute numbers).

## Key design decisions and their rationale

1. **Teacher forcing** (`runner.py:replay_conversation`) — after scoring the generated response, ingest the *actual* recorded turns, not the generated ones. Prevents error compounding; every turn is scored on the same trajectory. This is the right call and must be preserved in any rewrite.
2. **k=0 ablation reuses the normal code path** (`--k 0`) rather than a separate baseline harness — baseline and treatment share identical prompt construction, so the measured delta is attributable to retrieval alone. This is also why `retrieval.query()` was left untouched when `hybrid_query` was added. (Side effect: baseline files are tagged `k0_ef50` even though ef is unused.)
3. **LLM judge with strict JSON** (`judge.py`) — `{"score": 1-5, "reasoning": str}`; parse failure scores 1 (conservative). `num_retries=5` on both judge and responder calls.
4. **The judge passes no sampling parameters at all.** `temperature`, `top_p` and `top_k` are deliberately absent: Claude Sonnet 5 rejects non-default sampling parameters with a 400. Steer the judge with the prompt, not the knobs. `max_tokens` is `1024`, not a tight `256`, because Sonnet 5 runs adaptive thinking by default and `max_tokens` caps thinking **and** visible text together — `256` truncates the JSON verdict. The responder (Haiku 4.5) does accept sampling parameters and still passes `temperature=0.3`.
5. **`scores_by_position` captured per conversation** (`eval/schemas.py`) — enables the score-vs-turn-depth curve that would directly demonstrate hypothesis H2 (memory value grows with depth). The consumer now exists (`eval/analysis.py`: `binned_scores`, `compare_runs`, `ascii_curve`); **known rough edge**: it has not been run against the archived 2026-01-31 data yet, and doing so needs no API calls (`--compare`).
6. **Prompt shape** (`responder.py:build_prompt`): system prompt → `"Relevant memory context:"` block with `[Memory i]` entries → last `recent_window` (default 4) turns → current user message. **placeholder**: still no token budgeting on this path — the `ContextPacker` (4500/900/800) exists but the replay runner does not use it. Acceptable while chunks×k is small (10 × ≤250 ≈ 2,500 tokens max), and `context_tokens` is now measured per turn so the assumption is checkable instead of assumed.

## Model defaults — a correction that mattered

**BUG (found 2026-08-14, fixed in the working tree):** both roles defaulted to `anthropic/claude-3-5-haiku-20241022`, which was **retired on 2026-02-19 and now returns 404**. Every eval run attempted since February would have failed. This is the most likely explanation for `eval_results/` stopping at 2026-01-31.

| Role | Old default (retired) | New default | Notes |
| --- | --- | --- | --- |
| Responder | `anthropic/claude-3-5-haiku-20241022` | `anthropic/claude-haiku-4-5` | documented replacement for 3.5 Haiku; accepts `temperature` |
| Judge | `anthropic/claude-3-5-haiku-20241022` | `anthropic/claude-sonnet-5` | stronger and a different tier; rejects non-default sampling params |

Defaults live in `eval/schemas.py` (`DEFAULT_RESPONDER_MODEL` / `DEFAULT_JUDGE_MODEL`) and are imported by `eval/__main__.py`, so the CLI help text can never drift from the schema.

**Consequence for the archived numbers**: everything in `08 - Analysis` was produced by the retired Haiku as both responder *and* judge. Those deltas remain directionally meaningful but are not comparable to anything produced from here on. Re-running the ablation pair is a Tier-0 item.

## Validity issues — status

| Issue | Status | Detail |
| --- | --- | --- |
| ~~Judge = responder model~~ | ✅ **Resolved** | Judge is Sonnet 5, responder is Haiku 4.5 — different models, different tiers. Absolute scores are still not comparable to the pre-2026-08-14 runs, but they no longer conflate memory quality with a single model's capability. |
| ~~No token-cost or latency instrumentation~~ | ✅ **Resolved** | `UsageStats` (input / output / cache-read tokens, `elapsed_s`, `calls`) flows through `TurnResult.responder_usage` + `.judge_usage` + `.retrieval_s` + `.context_tokens` → `ConversationResult.usage` → `EvalRunResult.usage` / `.total_elapsed_s` / `.mean_context_tokens` / `.tokens_per_scored_turn`. Efficiency — the axis SimpleMem competes on — is now measurable. |
| **Single-user, small-n** | 🔲 **Open** | The most recent run is one 283-turn conversation. Directional, not publishable. Public-benchmark runs are the fix, not more personal conversations. |
| **Ground truth is a stronger model than the responder** | 🔲 **Open** | Recorded assistant turns came from a stronger Claude than Haiku 4.5. Absolute scores stay deflated; only within-pair `k=0` vs `k=N` deltas are load-bearing. Unfixable without regenerating ground truth, which would destroy the "annotation cost is zero" property. |
| **The replay eval does not exercise the memory layer** | 🟡 **Partly addressed** | `search_hybrid` is now reachable via `RetrievalConfig.hybrid` / `--hybrid` (dense remains the default), and the wasted extraction is gone — both eval paths pass `auto_extract=False`. Still open: the prompt is built from raw `[Memory i]` chunks, so `recall_memories` and `build_context` — the memory header, typed bullets, expansions — are exercised by no measurement at all. A/B-ing `build_context` against raw-chunk prompting is the highest-value open eval task. |

---

**Verification block**: run

```powershell
pixi run python -m memory_condense.eval --compare `
  eval_results/eval_120-250_k0_ef50_20260131_041148.json `
  eval_results/eval_120-250_k10_ef50_20260131_041341.json
```

No API calls, no cost — it reads two archived runs and prints the ablation table, the per-position-bin deltas, and the ASCII score-vs-depth curves. Token/latency rows will read zero (those runs predate `UsageStats`). Then decide: re-run the pair with the fixed models, or spend the first API budget on `--benchmark-file` instead.
