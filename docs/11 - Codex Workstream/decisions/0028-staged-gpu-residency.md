# 0028. Adopt staged GPU residency

- **Status:** Accepted
- **Date:** 2026-08-18
- **Tag:** LOCK-IN

## Context

Restoring the six-layer Qwen3-8B prefix (DR-0027) put ~3.5 GiB of selector
weights on the 8 GiB RTX 2070, and locked questions were suddenly taking
about ten minutes each. The user's probe — "why are we on cpu?" (turn
1143) — surfaced the cause: "I had a defensive rule that moved BGE to CPU
so its ~2.3 GB weights would not collide with the ~3.5 GB Qwen prefix on
an 8 GB card. That rule caused the 10-minute slowdown" (turn 1144).

The collision the rule defended against is real but marginal, not
absolute: measured GPU baseline ~1.75 GiB, plus ~3.5 GiB Qwen prefix, plus
~2.3 GiB BGE-M3 puts static co-residency near 7.5 GiB, leaving "~0.6 GiB
[which] is not a safe operating margin" before attention activations,
temporary CUDA buffers, and allocator fragmentation (turn 1147). Neither
standing policy — BGE permanently on CPU, or both models permanently on
GPU — is acceptable. The key observation is that the retrieval and
selection stages never need both models at once.

## Decision

Remove the BGE-on-CPU defensive rule and adopt staged GPU residency:
run BGE-M3 retrieval on the GPU, freeze the query vectors, unload BGE,
then load the Qwen prefix on the GPU for the bounded candidate read —
`BGE on GPU -> freeze query vectors -> unload BGE -> Qwen prefix on GPU`.
The two models are never simultaneously resident (turns 1144, 1147).

## Consequences

- **Positive:** Both failure modes disappear at once — the ~10-minute
  CPU-BGE regression and the ~0.6 GiB co-residency fragmentation/OOM risk.
  Staged correctly, the prefix read is fast: a 10.47 s one-time load, then
  0.39 s for all eight candidates in a single forward pass over 240 active
  token positions, confirming "CPU BGE — not Qwen attention — was the
  regression" (turn 1145). The 96 focused tests passed and the locked
  questions were rerun with the staged all-GPU flow.
- **Negative / cost:** The pipeline becomes order-dependent: query vectors
  must be frozen before BGE is released, and interleaving retrieval with
  selection is no longer possible within a question. Each stage transition
  pays a model load/unload, so the design only works because the stages
  are cleanly sequential.
- **Follow-ups:** Simultaneous FP16 BGE residency was noted as a possible
  later optimization "but it is unnecessary for this pipeline" (turn
  1147). The staged flow becomes part of the end-of-phase pipeline shape
  documented in chapter 06.

## Alternatives considered

- **Keep BGE permanently on CPU** — the prior defensive rule. Rejected on
  measurement: it cost roughly ten minutes per question, dominating
  end-to-end latency (turn 1144).
- **Full static GPU co-residency** — leave both models loaded. Rejected on
  measured GiB arithmetic: ~0.6 GiB of headroom before activations, CUDA
  buffers, and allocator fragmentation is a real OOM risk, not a safe
  operating margin (turn 1147).
- **Simultaneous FP16 BGE residency** — shrink BGE until both fit.
  Deferred as a separate optimization, not adopted: staging already removes
  the need for simultaneous residency (turn 1147).

## Source

- **Source merged turns:** 306
- **Raw sub-turns:**
  [turn-1143-user.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1143-user.md),
  [turn-1144-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1144-assistant.md),
  [turn-1145-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1145-assistant.md),
  [turn-1147-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1147-assistant.md)
- **Dev guide:** [chapter 06](../dev-guide/06-set-completion-selector.md)
