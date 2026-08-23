# 0019. Choose four-slot HSC channel and reversible pruning

- **Status:** Accepted
- **Date:** 2026-08-17
- **Tag:** LOCK-IN

## Context

The first HSC run with an eight-slot reserve produced a clean diagnostic: "HSC
did recover the previously missing `..._2` source, but its eight-slot reserve
displaced `..._5`, leaving aggregate coverage unchanged. So hierarchy
construction and descent worked; allocation is the problem" (turn 800).
Coverage stayed at 5/6 not because evidence was missing or unreachable, but
because too much information competed for a fixed-width channel — a
routing-and-capacity failure, not a recall failure.

Shrinking the reserve to four slots preserved the baseline's five sources and
admitted the recovered sibling, yielding the phase's first 100% mean
evidence-source coverage — all required sources on 10/10 questions — at 2,179
mean tokens on the 1,039,203-token memory, with no provider calls (turns 802,
804, 806). When the follow-up question asked whether *more aggressive* channel
pruning would help further, the same displacement evidence set the answer's
shape: the four-slot channel beat the eight-slot channel precisely because the
larger channel evicted useful baseline evidence, so pruning "should be
query-conditioned and reversible, not aggressive permanent deletion"
(turn 808).

## Decision

Fix the HSC channel into the final packet at four slots, and make all channel
pruning query-conditioned and reversible rather than permanent deletion.
Prune only low-marginal-utility channels (redundant chunks from covered
sources, high-frequency low-ISF concepts, unreinforced Hebbian edges,
near-duplicate paths); protect rare TF-ISF signals, temporal and contradiction
edges, low-degree bridge nodes, sources not yet represented in the packet, and
provenance paths for multi-premise questions.

## Consequences

- **Positive:** First 100% source-coverage result on the 1M development set;
  isolates the load-bearing insight that the bottleneck is flow allocation
  through bounded cuts, not recall — which later justifies rejecting
  sum-to-one (softmax-style) channel competition; reversibility means no
  pruning decision can permanently lose evidence for a future query.
- **Negative / cost:** Four is a hard-coded constant tuned on the locked
  10-question development set, not a learned or adaptive allocation; reserves
  of 1–3 were identified as worth testing but the sweep was not run;
  reversible pruning alone does not lower prompt tokens unless coupled to an
  early-stop rule or a lower dynamic token budget.
- **Follow-ups:** The proposed adaptive-flow scheme — open slots
  incrementally and admit only while marginal new-source/new-concept gain
  clears a threshold — became the information-bottleneck greedy packer
  ([DR-0020](0020-ib-greedy-channel-packer.md)). A learned multi-label channel
  gate to replace the hard-coded four-slot allocation is designed but
  unimplemented at phase end. Builds on the HSC layer adopted in
  [DR-0017](0017-tf-isf-hsc-adoption.md).

## Alternatives considered

- **Eight-slot HSC reserve** — the initial configuration; rejected on direct
  evidence: it admitted the recovered source only by displacing another
  required source, leaving coverage stuck at 5/6.
- **Aggressive permanent channel pruning** — proposed in the follow-up
  question; rejected because it would delete channels (rare signals, bridge
  nodes, provenance paths) that a future query needs, when the displacement
  evidence showed the real problem is allocation, not excess retained
  structure.
- **Smaller reserves (1–3 slots)** — flagged as the natural next sweep
  ("the current result indicates four is enough for full source flow; the
  next question is whether two or three can retain it") but not chosen,
  since four was the measured sufficient value.

## Source

- **Source merged turns:** 196, 198
- **Raw sub-turns:**
  - [turn-800-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-800-assistant.md)
  - [turn-802-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-802-assistant.md)
  - [turn-806-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-806-assistant.md)
  - [turn-808-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-808-assistant.md)
- **Dev guide:** [chapter 05](../dev-guide/05-packet-compression-and-operational-replacement.md)
