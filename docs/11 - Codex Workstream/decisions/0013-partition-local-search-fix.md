# 0013. Fix step 4 with partition-local search

- **Status:** Accepted
- **Date:** 2026-08-17
- **Tag:** LOCK-IN

## Context

The locked n=40 result (DR-0012) showed 99.5% mean evidence-source coverage but only
23/24 recoverable literal answers: source discovery was essentially solved, so the
remaining retrieval bottleneck had to be chunk selection inside already-activated
sources. Walking the retrieval chain step by step named the flaw. The pool feeding
within-partition selection was step 4 — "a global top-200 pool, filtered to activated
partitions" — and, as diagnosed: "That means an answer-bearing chunk outside the global
top 200 is unavailable, even when we correctly identified its partition. A Qwen reranker
cannot recover a chunk it never receives."

The failure was structural, not a tuning problem: the pipeline could report 100% source
coverage while the answer-bearing chunk never entered the candidate pool at all. When
the user confirmed the diagnosis ("You said the weak spot was 4?") and asked what the
fix looked like, the design was `activate partition → search/enumerate all chunks within
it → select a bounded local candidate set → optionally Qwen-rerank → pack`, and the
order was given: "ok do that."

## Decision

Replace the global-pool filter with true partition-local candidate generation: scan all
chunks inside activated sources with bounded dense/BM25 candidate buffers (streamed
embeddings, top-k heaps), hydrate text only after ranking, calibrate scores globally so
weak partitions cannot crowd out evidence, and allocate candidate quotas across
partitions. Implement the local BGE/BM25 search first, without Qwen, to isolate whether
step 4 was responsible; ship it behind `--source-local-search` while preserving the
frozen historical policy as the default.

## Consequences

- **Positive:** The architectural hole is closed — an answer-bearing chunk in a
  correctly activated partition can no longer be silenced by the global ranking — and a
  regression test pins the failure mode. Second-stage rerankers now receive candidates
  they previously never saw. The diagnosis itself (fixed global pools crowd out
  partitions) proved the durable asset: at 1M scale the same bottleneck reappeared as
  pool crowding and was addressed by widening.
- **Negative / cost:** On the locked n=40 development set the fix was measurably
  neutral — literal recall unchanged at 23/40, coverage 99.5%, mean context 6,666.9 vs
  6,637.8 tokens, no row-level hit changes — so it costs slightly more without improving
  recall at 100K scale. It therefore ships as an available ablation, not the selected
  policy.
- **Follow-ups:** With step 4 fixed and still neutral, the investigation moved up a
  level: the gold-source sufficiency audit showed literal search itself was saturated,
  leading to evidence-chain assembly (DR-0014, DR-0015) and ultimately the metric
  reframe (DR-0016). Key changes landed in `src/memory_condense/retrieval.py` and
  `src/memory_condense/lexical.py`; 751 tests passed.

## Alternatives considered

- **Keep tuning the global top-200 pool (larger k, better weights)** — rejected because
  the failure is structural: no global ranking filtered by partition can guarantee a
  correctly activated partition contributes its answer-bearing chunk, and no downstream
  reranker can recover a chunk it never receives.
- **Swap in Qwen embeddings to improve the global pool** — considered and explicitly
  demoted: "Qwen embeddings might improve the global top-200 pool, but that only
  mitigates the flaw. The stronger architectural correction is true partition-local
  candidate generation."
- **Lead with a Qwen QK/CAV reranker over the existing pool** — rejected as sequencing:
  the local BGE/BM25 search runs first without Qwen, so any improvement can be
  attributed to local enumeration rather than the reranker; Qwen remains an optional
  matched second-stage arm over the final 32-64 candidates.

## Source

- **Source merged turns:** 123, 127
- **Raw sub-turns:**
  [turn-657-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-657-assistant.md),
  [turn-659-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-659-assistant.md),
  [turn-660-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-660-user.md),
  [turn-673-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-673-assistant.md)
- **Dev guide:** [chapter 04](../dev-guide/04-longmemeval-debugging-and-1m-baseline.md)
