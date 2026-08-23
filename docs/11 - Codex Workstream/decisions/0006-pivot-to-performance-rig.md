# 0006. Pivot to performance optimization with a parallel-run rig

- **Status:** Accepted
- **Date:** 2026-08-16
- **Tag:** PIVOT

## Context

The easy notes benchmark had saturated — 8/8 recall at hybrid k=3 — so further recall tuning on
that slice was noise. The user redirected: "If we saturated benchmarks, let's do some performance
optimizations, make a rig that can run multiple runs in parallel and do that stuff in the
downloads folder" (turn 218).

One hardware constraint shaped the rig's design from the first response: "launching several
4.6 GB Qwen-prefix workers on the same GPU would multiply VRAM and undermine the bounded-memory
design" (turn 219). Parallelism had to come from the model-free side. A first test run added a
second constraint: a single SQLite connection shared across concurrent sweep threads corrupts, so
each arm needs its own lightweight read connection (turn 228).

## Decision

Build a dedicated parallel-run rig in the Downloads workspace, outside the repo: parallel
independent retrieval/evaluation arms on CPU, each opening its own lightweight DB/association
reader (no ANN index, embedding model, or Qwen per arm), with a single serialized Qwen
compilation worker per GPU, explicit CPU/GPU concurrency limits, and persisted artifacts reused
across arms. Sweeps isolate one variable at a time (degree, QK-only vs QK+CAV, reserved-slot
count) under identical budgets, and the in-flight fresh untouched-split run finishes first as the
evidence gating the pivot.

## Consequences

- **Positive:** Real parallelism without violating the bounded-memory design or duplicating the
  4.6 GB prefix. The rig's first honest measurement was the pivotal one: the untouched split did
  not saturate — linked k=3 with one reserved slot regressed to 75.0% against hybrid k=3 at 83.3%
  (turn 237) — exposing that compiled links were not yet earning displacement. The subsequent
  sweep found the first real improvement on the harder split: two-hop QK with one reserved slot
  at 91.7% recall, matching hybrid k=10 with about 54% fewer tokens (turn 243).
- **Negative / cost:** The rig lives outside the repo, so its artifacts are not versioned with
  the code. Any split the sweep touches becomes tuned development data — the two-hop result was
  explicitly hardened as an experimental preset, not a new default, for exactly that reason. GPU
  compilation throughput stays serialized at one worker.
- **Follow-ups:** The 75.0% regression and the three never-linked misses it surfaced motivate the
  write-time coverage work (diversified candidate pool, coverage funnel) and the read-stage
  redesign in [DR-0007](0007-heat-diffusion-framing.md). The untouched-split discipline continues
  the evaluation honesty established in [DR-0004](0004-halt-infrastructure-drift.md).

## Alternatives considered

- **Keep tuning recall on the saturated notes slice** — rejected: at 8/8 further gains there are
  indistinguishable from noise; effort moves to performance and to a harder untouched split.
- **Parallel Qwen workers, one per arm** — the naive way to parallelize end-to-end runs.
  Rejected: several 4.6 GB prefix workers on one GPU multiply VRAM and undermine the
  bounded-memory design; instead one serialized compilation worker per GPU with model-free CPU
  arms.
- **Shared DB connection across sweep threads** — the initial implementation. Rejected after the
  test run showed a single SQLite connection cannot be shared across concurrent threads; one
  lightweight read connection per arm is required.

## Source

- **Source merged turns:** 049
- **Raw sub-turns:**
  - [turn-218-user.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-218-user.md)
  - [turn-219-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-219-assistant.md)
  - [turn-228-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-228-assistant.md)
  - [turn-237-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-237-assistant.md)
- **Dev guide:** [chapter 02](../dev-guide/02-retrieval-grounding-and-heat-diffusion.md)
