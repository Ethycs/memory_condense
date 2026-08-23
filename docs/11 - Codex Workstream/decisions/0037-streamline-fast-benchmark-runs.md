# 0037. Drop exact validation rebuilds; streamline for speed

- **Status:** Accepted
- **Date:** 2026-08-23
- **Tag:** SCOPE-CUT

## Context

The exact-validation shard campaign — rebuilding stores per shard to reproduce
scores exactly — cost 50-75 minutes of store construction per cycle. Six
shards had completed and the offset-60 build was partially done, but the
remaining shards answered no open design question, while live architecture
questions (does CAV ordering help? does the Hebbian arm help?) could not be
iterated on at that cadence at all.

The user cut the scope directly: "We don't need exact validation, we just need
the memory retrieval task + summarization against the benchmarks and we need
to streamline operation to make it speedy." Iteration speed had become a
validity constraint, not a convenience — at 90+ minutes per exact rebuild the
locked benchmark gate could not be reached affordably.

## Decision

Stop the exact-validation shard campaign immediately and perform no further
exact rebuilds. Preserve the six completed shards and the partial offset-60
build as the exactness evidence. Replace the campaign with a lean loop: build
the corpus once, run each retrieval method incrementally against the sealed
artifacts, synthesize the retrieved evidence with the LLM, and score directly
against the benchmark with latency and cost reported alongside accuracy.

## Consequences

- **Positive:** A design-question cycle drops from 50-75 minutes of rebuild to
  seconds: sealed-artifact preflight 1.76 s, cached replay 2.01 s with zero
  provider calls, scoring with journal validation 19.47 s (the fast runtime in
  `src/memory_condense/eval/run_fast_1m_cav.py` and supporting `fast_cav_*`
  modules, commit `d1c8808`). Latency and cost become first-class reported
  metrics. The fast path is what makes reaching the locked 100-question gate
  cheap enough to attempt.
- **Negative / cost:** Gives up ongoing bit-exact reproduction of scores from
  fresh rebuilds; exactness evidence is frozen at the six completed shards.
  Fast-path results are ten-question development diagnostics, explicitly
  distinct from the locked 100-question gate. The one-time Qwen load
  (216.84 s) now dominates the fast path's feature phase.
- **Follow-ups:** Provenance hashes and sealed artifacts must carry the
  correctness burden that rebuilds carried. The fast loop becomes the vehicle
  for the ladder-restoration experiments (DR-0038, DR-0039) and the per-case
  cumulative progression (DR-0040). Running the locked 100-question gate
  through the fast path, and keeping Qwen resident between feature
  experiments, remain open work.

## Alternatives considered

- **Continue the exact-validation shard campaign to completion** — the status
  quo; proved correctness but at 50-75 minutes of store construction per
  cycle, and the remaining shards answered no open design question. Rejected
  as pure cost.
- No other alternatives were live at decision time; this was a clean pivot,
  with the completed shard evidence preserved rather than discarded.

## Source

- **Source merged turns:** 449
- **Raw sub-turns:**
  [turn-3307-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3307-user.md),
  [turn-3308-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3308-assistant.md)
- **Dev guide:** [chapter 09](../dev-guide/09-acceleration-scoring-and-ladder-restoration.md)
