# Gap Analysis and Roadmap

**Status:** Living Document
**Date:** 2026-08-14
**Supersedes:** the earlier 2026-08-14 revision, which listed Phases 1–4 as unbuilt and the Decision Point as blocking Tier 2
**Applies to:** commit `cd9f423` + the **UNCOMMITTED** memory-layer working tree

## Current status

Everything marked ✅ below is **built and unit-tested but not committed**. `git status` is the authority.

| Design component (from `01 - Design/00`) | Verdict |
| --- | --- |
| TranscriptStore / Chunker / bge-m3 / ANN index / similarity demo | ✅ Complete |
| ~~Hybrid sparse retrieval (lexical weights)~~ | ✅ Complete — BM25 (`lexical.py`) + `hybrid_query(alpha=0.65)`; `chunks.lexical_weights` and `chunk_terms` are populated by `add_chunks` |
| ~~Rerank formula (importance/recency/pins)~~ | ✅ Complete — `ranking.py`, one implementation shared by memory + chunk paths |
| ~~ContextPacker (token budgets)~~ | ✅ Complete — `context_packer.py`, 4500/900/800, ≤3 expansions × ≤250 tok, drops counted |
| ~~MemoryItem / MemoryOps / Validator / provenance~~ | ✅ Complete — `schemas.py`, `memory_store.py`, `validator.py`, `extractor.py` |
| ~~Decay / HOT-WARM-COLD tiering~~ | ✅ Complete — `decay.py`, lazy exponential decay, reheat on access, pins exempt |
| ~~Token + latency instrumentation in eval~~ | ✅ Complete — `UsageStats` through `TurnResult` → `ConversationResult` → `EvalRunResult` |
| ~~Judge decoupled from responder~~ | ✅ Complete — judge `anthropic/claude-sonnet-5`, responder `anthropic/claude-haiku-4-5` |
| ~~Benchmark loader (LongMemEval / LoCoMo)~~ | ✅ Complete — `loader.load_benchmark` + `eval/benchmark.py` QA-probe harness |
| ~~`scores_by_position` analysis code~~ | ✅ Complete — `eval/analysis.py`; 🔲 **never run against real data** |
| ~~Hardcoded `dim = 1024`~~ | ✅ Fixed — `embedding.py` reads the true dimension from the loaded model |
| Eval harness (self-replay + judge + ablation) | ✅ ~95% — instrumented and model-fixed; sweep still never run |
| **Benchmark numbers on LongMemEval / LoCoMo** | 🔲 **Open — harness exists, has never been run. No competitiveness claim is possible yet.** |
| **The parameter sweep** | 🔲 Open — `sweep.py` exists, 54 configs, never executed |
| **Cold-tier era summaries** (design Phase 4) | 🔲 Open — unimplemented |
| ~~Hybrid retrieval reachable from the self-replay eval~~ | ✅ Complete — `RetrievalConfig.hybrid/alpha/candidates`, `--hybrid`/`--alpha`, distinct result filename. 🔲 **the delta vs dense is still unmeasured** |

## Decision Point (now UNBLOCKED, still OPEN)

MemDelta (see `04 - Reference/00`) showed memory-ops architectures can lose to a well-embedded RAG baseline once confounds are controlled. The original gate was: *before building Phase 2+, run the current pipeline on a common benchmark.*

**What changed:** the blocker was "no benchmark adapter exists." That adapter now exists (`load_benchmark` + `eval/benchmark.py`, F1/EM/per-category, injected `answer_fn`/`judge_fn`). The memory layer was also built ahead of the gate — so the gate's *purpose* shifts from "should we build Phase 2?" to **"does any of what we built earn its place?"**

**The decision is still open, and no benchmark run has happened.** Until one does:

- We cannot claim competitiveness against SimpleMem / Mem0 / Zep.
- We cannot claim the memory layer beats the dense-retrieval baseline it sits on top of.
- The honest statement of record is: *"a full local memory manager exists and is unit-tested; its value over a strong dense baseline is unmeasured."*

Deciding requires exactly two runs: a `--k 10` dense baseline and a memory/hybrid treatment on the same benchmark file. Start with `longmemeval_oracle.json` and `--max-samples 10`.

## Tier 0 — Quick wins (<1 day each)

| Gap | Depends on | Blocks | Status |
| --- | --- | --- | --- |
| ~~Commit `num_retries=5` on judge/responder~~ | — | — | ✅ in the tree (still UNCOMMITTED, with everything else) |
| ~~Token + latency instrumentation~~ | — | — | ✅ done |
| ~~Decouple judge from responder~~ | — | — | ✅ done |
| **Commit the working tree** | — | *every* claim below being reproducible by anyone else | 🔲 **do this first** |
| Run the first benchmark: `--benchmark-file longmemeval_oracle.json --max-samples 10` | commit; LongMemEval data downloaded | Decision Point | 🔲 |
| Run the `scores_by_position` analysis against the 2026-01-31 pair (`--compare`) | — | hypothesis H2 becoming a figure instead of a claim | 🔲 |
| Re-run the 2026-01-31 ablation pair with the fixed models | commit | all four archived numbers were produced by a now-retired responder+judge | 🔲 |

## Tier 1 — Measurement (days)

| Gap | Depends on | Blocks | Status |
| --- | --- | --- | --- |
| Full LongMemEval + LoCoMo runs with per-category breakdown | Tier 0 first run | Decision Point; any external comparison | 🔲 |
| Run the parameter sweep (54 configs — 9 chunker × 3 k × 2 ef) | commit; benchmark run first, so the sweep optimizes something worth optimizing | defaults (120–250, k=10, ef 50) are still guesses | 🔲 |
| Measure hybrid vs dense: run twice (plain, then `--hybrid`) and `--compare` | — | knowing whether BM25 helps *this* workload | 🔲 wiring done, run missing |
| A/B the memory layer itself: `build_context` (memory header + expansions) vs the current raw-chunk prompt | — | knowing whether condensation beats raw retrieval | 🔲 |
| Tune `alpha` (currently `0.65`) and `RankWeights` (1.0 / 0.3 / 0.5 / 0.2 / 1.0) — both are unmeasured defaults | a working benchmark loop | — | 🔲 |

## Tier 2 — Build (weeks, gated on measurement)

| Gap | Depends on | Blocks | Status |
| --- | --- | --- | --- |
| Cold-tier era summaries (design Phase 4: cluster summaries + centroid index) | evidence that COLD items are worth keeping at all | — | 🔲 gated |
| `LLMExtractor` in the default path (currently `RuleBasedExtractor`) | evidence that rule-based extraction is the bottleneck | — | 🔲 gated |
| MC-STD-MEMOPS: freeze the `MemoryOps` wire contract | an external consumer existing | — | 🔲 gated |
| Second ANN index for memory items | memory item counts reaching thousands (brute-force cosine is deliberate below that) | — | 🔲 gated, probably never |

## Known rough edges (tracked, not blocking)

1. ~~`runner.replay_conversation` pays for extraction it never reads.~~ **Fixed** — `runner.py` and `benchmark.ingest_sample` both now pass `auto_extract=False`, with a comment saying to turn it back on only alongside a responder that consumes `build_context`. Worth noting how it was found: every test passed with the waste in place, because no test asserted anything about cost.
2. ~~Benchmark reports were named after the `--benchmark-format` flag (`benchmark_auto_*.json`).~~ **Fixed** — the run is now labelled with the dataset file's stem.
3. The sweep re-ingests (re-chunks, re-embeds) the whole corpus per config — self-documented at `sweep.py:78`. 54 configs × corpus embedding time is the dominant cost.
4. `eval_results/` is gitignored and still contains only the four 2026-01-31 files. Any number worth keeping must be copied into `08 - Analysis` by hand.

## Explicitly not planned

- Competing on architecture feature-count with SimpleMem/Letta — the differentiator is the self-replay eval + strong-baseline discipline, not feature parity.
- A separate baseline harness — axiom 5 in `03 - Architecture/00` (shared code path) stands.
- Re-adding `temperature` to the judge call — Claude Sonnet 5 rejects non-default sampling parameters with a 400. Steer the judge with the prompt.

---

**Verification block**: run

```powershell
git status --short                                     # expect ~40 modified/untracked paths — nothing committed yet
pixi run -e dev pytest -q -m "not slow"                # expect 366 passed, 13 deselected
ls eval_results                                        # expect exactly the four 2026-01-31 files
```

Then decide: **commit the tree**, or run the first `--max-samples 10` benchmark against the uncommitted tree to find out whether it is worth committing as-is. Committing first is the recommended order — every number produced from an uncommitted tree is unreproducible.
