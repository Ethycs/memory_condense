# Session Handoff — memory layer, hybrid retrieval, eval instrumentation, benchmark path

**Date**: 2026-08-14 **Status**: ✅ LANDED (code complete, all tests green) — **everything is UNCOMMITTED**
**Scope**: whole repo — design Phases 1–3 and 5 built out; Phase 4 deliberately not
**Supersedes**: nothing; follows `2026-08-14_docs-tree-creation-and-repo-snapshot.md` the same day

## 1. What this session did (chronological)

1. Built the shared spine by hand: memory schemas, SQLite schema v2 + migration, decay/tiering math, the rerank scalar, the context packer.
2. Fanned out four parallel agents on disjoint file sets: memory store + validator + extractor; hybrid BM25 retrieval; eval instrumentation + analysis; benchmark loader + QA-probe harness.
3. Wired the `MemoryCondenser` facade over all of it and rebuilt the eval CLI into four modes.
4. Reconciled the `docs/` tree against the as-built system (separate pass).

## 2. Changes in the working tree

**All uncommitted.** 11 new source modules, 14 modified, 12 new test files.

| File | Change | Test state |
| --- | --- | --- |
| `schemas.py` | + MemoryItem, Provenance, 5 MemoryOps types, ValidationReport, MemoryResult, PackedContext | covered |
| `db.py` | schema **v1 → v2** + migration path; memory_items, memory_provenance, chunk_terms, `chunks.term_count` | `test_db.py`, incl. real v1→v2 upgrade |
| `decay.py` *(new)* | exponential energy decay, HOT/WARM/COLD, reheat, pins exempt | `test_decay.py` |
| `ranking.py` *(new)* | the design's rerank scalar in one place + hybrid blend helpers | `test_ranking.py` |
| `context_packer.py` *(new)* | hard per-section budgets (4500/900/800), drops counted | `test_context_packer.py` |
| `memory_store.py` *(new)* | CRUD, supersede, soft delete, pin, touch/reheat, ranked retrieve | `test_memory_store.py` |
| `validator.py` *(new)* | provenance enforcement — quotes must exist verbatim in the turn | `test_validator.py` |
| `extractor.py` *(new)* | RuleBasedExtractor (offline) + LLMExtractor (injected callable) | `test_extractor.py` |
| `lexical.py` *(new)* | BM25 over a SQLite inverted index | `test_lexical.py` |
| `retrieval.py` | + `hybrid_query`, `delete_chunk`; `lexical_weights` now populated | `test_retrieval.py` |
| `embedding.py` | **BUG FIXED**: `dim` no longer hardcoded to 1024 | `test_embedding.py` |
| `condenser.py` | facade wires memory + packer; `build_context`, `recall_memories`, injectable embedder | `test_condenser.py` |
| `eval/*` | retired models replaced, token/latency instrumentation, `analysis.py`, `benchmark.py` | 48 eval tests |
| `eval/__main__.py` | four CLI modes: replay, sweep, benchmark, offline compare | manual (`--help` verified) |
| `loader.py` | + LongMemEval / LoCoMo JSON parsing | `test_loader.py` |

**Test baseline moved from 48 → 366 passing** (13 slow deselected).

## 3. Findings (the valuable part)

1. **BUG, now fixed — the eval harness has been dead since February.** Both judge and responder defaulted to `anthropic/claude-3-5-haiku-20241022`, which was **retired 2026-02-19** and returns 404. Every eval run attempted since then would have failed. This is the actual explanation for why `eval_results/` stops at 2026-01-31 — the results aren't merely stale, the harness was broken. Defaults are now `anthropic/claude-haiku-4-5` (responder) and `anthropic/claude-sonnet-5` (judge).
2. **BUG, now fixed — argparse was shadowing the config defaults.** `eval/__main__.py` re-literalled the model IDs as argparse defaults, so fixing `EvalConfig` alone would *not* have fixed CLI runs; argparse always supplies a value and would have kept forcing the retired model. The CLI now imports `DEFAULT_JUDGE_MODEL` / `DEFAULT_RESPONDER_MODEL` so the two can never drift again.
3. **validated — the judge/responder conflation is resolved.** The old harness used the same model for both, so absolute scores mixed "memory worked" with "responder can't match ground truth". Judge is now a stronger, different-tier model. Note the judge deliberately passes **no `temperature`**: Claude Sonnet 5 rejects non-default sampling parameters with a 400. There is a comment in `judge.py` and a regression test so nobody helpfully re-adds it.
4. **BUG, now fixed — `EmbeddingService.dim` was hardcoded to 1024**, which silently corrupted the index on any model swap rather than failing loudly.
5. **validated — the "vestigial lexical weights" gap is closed.** `chunks.lexical_weights` had existed since Phase 0 but nothing ever wrote it. It now carries real term-frequency data backing BM25, and `hybrid_query` blends dense + lexical.
6. **validated — hybrid retrieval is measurable from the eval.** `RetrievalConfig` gained `hybrid` / `alpha` / `candidates`, the runner dispatches to `search_hybrid` when enabled, and `save_run_result` tags hybrid runs in the filename so a hybrid run cannot silently overwrite the dense run it is being compared against. Dense stays the default so the existing k=0/k=N ablation measures the same baseline as before. **The effect is still unmeasured** — the capability exists, the number does not.
7. **validated — core stayed provider-agnostic.** Phase 2 needed an LLM call in the core package, which would have violated the "litellm only in eval/" axiom. Instead `LLMExtractor` and `benchmark.py` take injected callables; the litellm binding lives in `eval/__main__.py`. `grep -r litellm src/memory_condense/` outside `eval/` returns nothing.
8. **BUG, now fixed — wiring memory into the facade silently taxed both evals.** `MemoryCondenser` defaults to `auto_extract=True`, so once the eval harness picked up the new facade it ran rule-based extraction on every single ingest — while the responder prompt reads retrieved *chunks*, not memory items. Pure cost, zero effect on any score, and worst on benchmark haystacks. `runner.py` and `benchmark.py` now pass `auto_extract=False` with a comment explaining when to turn it back on. Caught during doc reconciliation, not by a test — a reminder that "all tests pass" and "the code does something sensible" are different claims.
9. **One real measurement came out of this session, and it argues against us.** Running the new `--compare` over the two archived 2026-01-31 files (offline, zero API cost) produced the score-vs-position curve for the first time. Retrieval's gain by position bin is **+0.214, +0.296, +0.464, +0.333, +0.185** — it *peaks mid-conversation and is smallest in the final fifth*. Hypothesis H2 in `00 - Theory` predicts the opposite. The likeliest explanation is deflating: bin 3 is also where the no-memory baseline scores worst (3.357), so retrieval may just have the most headroom there rather than the most value. Written up in `08 - Analysis/00`. **Do not cite H2 as confirmed.**
10. **Otherwise, no new evidence.** Everything else here is capability, not measurement. The sweep still hasn't run and no benchmark has been executed — see open work.

## 4. Open work, in priority order

1. **Commit this.** It is a large uncommitted tree; nothing is version-controlled yet. Suggest splitting: spine + memory layer, retrieval, eval, docs.
2. **Re-run the ablation pair** now that the harness works again — this is the first real evidence in ~7 months, and the numbers in `08 - Analysis` were produced by a now-different judge model so they are not directly comparable. Blocking: nothing.
3. **Run the benchmark.** `--benchmark-file longmemeval_oracle.json --max-samples 10` first. This is the Decision Point in `06 - Roadmaps` — it is now *unblocked*, and until it runs there is still no basis for any competitiveness claim against SimpleMem/Mem0/Zep.
4. **Measure dense vs hybrid** on the same corpus: run once plain and once with `--hybrid`, then `--compare` the two result files. The wiring is done; only the run is missing.
5. **Run the 54-config sweep** (`--sweep`) to replace the guessed defaults (120–250, k=10, ef 50) with measured ones. Now affordable to interpret because token/latency are instrumented.
6. **Repeat the position-bin analysis across several conversations** (finding 9). One transcript cannot separate "retrieval helps most mid-conversation" from "retrieval helps most where the baseline is weakest", and that distinction decides whether H2 survives.
7. **A/B `build_context` against the raw-chunk prompt.** The memory layer — header, typed bullets, expansions — is currently exercised by *no* measurement at all; the replay eval still prompts with raw `[Memory i]` chunks. This is the biggest untested claim in the repo.
8. Design Phase 4 (cold-tier era summaries) remains unimplemented and ungated.

## 5. Artifacts

| Path | What |
| --- | --- |
| `src/memory_condense/` | 11 new modules; the memory layer, hybrid retrieval, packing |
| `src/memory_condense/eval/` | instrumented harness + `analysis.py` + `benchmark.py` |
| `examples/memory_demo.py` | runnable demo: extraction → provenance rejection → pinning → recall → packing |
| `docs/` | reconciled against the as-built system in the same session |
| `eval_results/` | **unchanged** — still the four 2026-01-31 files, still gitignored |

---

**First action for the next session**: run `pixi run -e dev pytest -q -m "not slow"` (expect **366 passed, 13 deselected**) and `git status --short` (expect ~40 uncommitted paths). Then decide: commit the tree first, or run the ablation pair (`--k 0` then `--k 10`) to get the first working numbers since February.
