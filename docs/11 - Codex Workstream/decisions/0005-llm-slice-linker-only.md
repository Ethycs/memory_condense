# 0005. Restrict the LLM slice to linker/inspector role

- **Status:** Accepted
- **Date:** 2026-08-16
- **Tag:** LOCK-IN

## Context

The implementation had crossed the intended role boundary: token-level Qwen K/V was being cached
for every chunk, which turns the model slice itself into the memory store. The user restated the
intent: "My intent was to use the LLM slice as linker for memory, not as an active retrieval
system subject to memory constraint... if we are to use the transformer head not to classify but
to remember, we need to take care that memory doesn't become an issue" (turn 170). A benchmark
already underway was cancelled mid-run "because its K/V-cache growth would measure the wrong
architecture" (turn 171).

A second correction (merged turn 043) closed the remaining loophole: heads may inspect nested
memory layers, "but we cannot store anything here because that repeats the problem of transformer
context" (turn 175). Nesting must mean repeated bounded inspections, never layers of stored
transformer state — otherwise the system recreates corpus-scale context accumulation, the exact
problem it exists to solve.

## Decision

Restrict the Qwen slice to a transient linker/inspector. Fixed-size model weights (the Qwen
prefix) plus a fixed-size transient workspace under an explicit token/candidate ceiling; every
activation, attention map, and K/V tensor is discarded after each inspection. The only durable
state lives outside the model: external source pointers, compact CAV coordinates, sparse QK/OV
link edges, and usage/decay counters. Linking happens at write time so reads follow stored links
without keeping Qwen activations alive; nested inspection passes only candidate IDs and scalar
scores between hops; the legacy K/V laboratory gets a hard 64-item ceiling; and the API is named
"inspector" / "link compiler" so the role cannot be mistaken for a head-resident memory.

## Consequences

- **Positive:** Memory cost is proportional to the link graph, not the corpus. Retained
  transformer K/V is zero bytes, and runs assert it. The corrected architecture immediately
  produced measurable results: 8/8 notes recall at hybrid k=3 with 636 mean tokens (70.3% fewer
  than k=10), bounded passes of at most four candidates / 1,056 tokens, and edge pruning
  337 → 225 with recall preserved (turn 196).
- **Negative / cost:** No K/V reuse — every hop recomputes activations over a fresh bounded
  workspace, and the head only ever sees a small candidate set at a time. Retrieval quality
  becomes limited by what the write-time compiler linked: compiled links did not help the easy
  notes slice, and coverage of the link graph becomes the system's bottleneck.
- **Follow-ups:** Persist the compact artifacts (SQLite, then the backend-neutral storage plane);
  the write-time coverage problem this creates drives the diversified candidate pool and coverage
  funnel work surfaced by [DR-0006](0006-pivot-to-performance-rig.md), and the bounded read stage
  this role permits is designed in [DR-0007](0007-heat-diffusion-framing.md).

## Alternatives considered

- **Token-level K/V caching per chunk** — cache Qwen K/V for every stored chunk so the model can
  attend over memory directly. Rejected: it makes the model slice the memory store, its growth is
  corpus-scale, and the running benchmark measuring it was cancelled as measuring the wrong
  architecture.
- **Stored transformer state across nested hops** — carry context from one inspection into the
  next to build "nested memory layers" inside the model. Rejected by the user in turn 175: it
  repeats the transformer-context problem; nesting was redefined as repeated bounded inspections
  with only IDs/scores crossing hops.
- **Head-as-classifier over the collection** — use the attention head to classify or hold the
  memory collection rather than link it. Rejected as the same boundary violation in different
  clothes; the head inspects memory, it never contains it.

## Source

- **Source merged turns:** 041, 043
- **Raw sub-turns:**
  - [turn-170-user.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-170-user.md)
  - [turn-171-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-171-assistant.md)
  - [turn-175-user.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-175-user.md)
  - [turn-176-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-176-assistant.md)
- **Dev guide:** [chapter 02](../dev-guide/02-retrieval-grounding-and-heat-diffusion.md)
