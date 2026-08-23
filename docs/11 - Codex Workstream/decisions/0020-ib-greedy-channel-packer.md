# 0020. Adopt the information-bottleneck greedy channel packer

- **Status:** Accepted
- **Date:** 2026-08-17
- **Tag:** LOCK-IN

## Context

After the four-slot HSC channel reached 100% evidence-source coverage at 2,179
tokens ([DR-0019](0019-four-slot-hsc-reversible-pruning.md)), further packing
needed a principle rather than another arbitrary token cap. The formulation
chosen was a conditional information bottleneck / rate–distortion problem:
maximize I(C;Y|Q) subject to a token budget, with channels admitted by marginal
information gain per token rather than independent relevance scores — "a
channel that looks relevant can have nearly zero marginal information because
another channel already carries the same fact" (turn 814).

The same framing retroactively explained the HSC reserve result: "the fifth
through eighth reserved HSC channels had lower marginal information than the
baseline channels they displaced" (turn 814). Attention could estimate the
marginal gain, but the governing principle would be information gain, not
attention itself. One failure class surfaced during implementation: on a
multi-fact temporal query ("order of the concerts…"), "repeated related
excerpts look redundant even though each event is required" (turn 830), which
motivated a query-cardinality guard following rate–distortion logic — dropping
one fact has higher distortion when the requested answer is a set or sequence.

## Decision

Adopt an opt-in greedy channel packer as a post-retrieval stage: a monotone
information-per-token filter that estimates query relevance plus marginal
concept/source/temporal novelty per token, recomputes after every admitted
channel, and stops early when the next channel's gain rate falls below a
threshold (best value 0.008), with a query-cardinality guard that lowers
pruning pressure for enumeration, ordering, comparison, and "all/each"
questions. The packer runs after retrieval, so raw memory, HSC structure, and
retrieval scores remain unchanged.

## Consequences

- **Positive:** Returned context dropped 2,179 → 1,986 tokens (~523:1
  compression) with literal recall, best token-F1, and 100% evidence-source
  coverage all held on the 1M development set, at zero provider calls (turn
  838); the threshold boundary is empirically located (0.00825 drops literal
  recall); early stop prevents filling the budget merely because space remains.
- **Negative / cost:** The information-gain estimate is heuristic (no access
  to the true answer variable Y at retrieval time); the coverage metric is
  source-granular, so one surviving chunk marks a session covered even when
  another required fact from that session was pruned (turn 830) — a gap that
  the operational test later exposed; the result is a ten-question offline
  development measurement, not an answer-accuracy claim.
- **Follow-ups:** The operational transcript-replacement trial on this packed
  context ([DR-0021](0021-operational-replacement-via-gateway.md)) is the real
  test of whether within-source fact coverage survives. A learned multi-label
  channel gate (counterfactual-ablation training) is noted as the correct
  future learned form of the same admission rule but is not implemented; the
  monotone filter remains deterministic.

## Alternatives considered

- **Independent per-channel relevance scoring** — score every candidate on its
  own and cut at a cap; rejected because marginally redundant channels look
  relevant while adding near-zero new information, which is exactly the
  displacement failure the HSC reserve experiment exhibited.
- **Softmax-style attention as the channel selector** — sum-to-one weights
  recreate the competition/displacement problem between necessary premises;
  attention is admissible only as an estimator of marginal gain, never as the
  governing selection principle (turn 814).
- **A lower fixed token cap** — the phase had already located a reliability
  knee where a hard cap silently lost a literal answer; rate–distortion tuning
  of the gain threshold replaces the arbitrary cap with a measured boundary.

## Source

- **Source merged turns:** 206
- **Raw sub-turns:**
  - [turn-814-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-814-assistant.md)
  - [turn-816-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-816-assistant.md)
  - [turn-830-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-830-assistant.md)
  - [turn-838-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-838-assistant.md)
- **Dev guide:** [chapter 05](../dev-guide/05-packet-compression-and-operational-replacement.md)
