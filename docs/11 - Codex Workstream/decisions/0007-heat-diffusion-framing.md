# 0007. Adopt heat-diffusion framing for the read stage

- **Status:** Accepted
- **Date:** 2026-08-16
- **Tag:** LOCK-IN

## Context

The read stage needed a framing that was both bounded and better than what existed. Putting a
fixed candidate triple into one attention context suffers cross-candidate softmax interference,
candidate-order bias, and constant re-normalization; and the existing best-path traversal had
already shown three hops regress recall (83.3%/75%) because it keeps only the strongest single
path to each candidate. The user supplied the reframe: "the idea is heat diffusion, attention
only needs the next item iteratively or like a reranker" (turn 294).

The framing resolves both problems at once. The attention head becomes a local transition
operator — it "does not need to hold or classify the memory collection. It only answers: given
this active memory, which candidate should receive heat next?" (turn 295) — while the external
graph carries the heat and sums evidence over corroborating paths, so multi-anchor agreement
becomes the admission signal instead of the strongest single edge. It also stays inside the
linker/inspector boundary of [DR-0005](0005-llm-slice-linker-only.md): activations are bounded
and discarded, and only IDs, scalar heat, paths, and compact CAV values cross iterations. The
user approved implementation directly: "ok let's do this and experiment with it" (turn 298).

## Decision

Adopt attention as a local transition operator for the read stage and implement it as a bounded,
model-free diffusion over the persisted attention graph. Transition score = calibrated QK(i→j) +
λ·OV alignment + μ·CAV compatibility − cycle/popularity penalties; heat is maintained externally
and multiplied by calibrated transition probabilities — raw attention weight is never used as
global heat, because attention is locally normalized within one workspace. Source heat converts
into prompt-token allocation (chunk priority = source_heat × query_relevance × novelty ×
relation_confidence / token_cost), with a scalar ID frontier, a tiny external beam, fixed
hydration at k=5, and the existing safe admission path preserved as the control arm.

## Consequences

- **Positive:** Constant transformer memory and a streamable candidate pool — 20 candidates can
  be inspected sequentially retaining only IDs and scores. Multi-anchor accumulation makes deep
  reads safe: three-hop diffusion with a 32-ID frontier and two heat slots preserved recall while
  cutting raw memory tokens 12.4% (973.9 → 852.9) on the development split (turn 305) and 9–19%
  on replayed earlier stores (turn 306) — where three-hop best-path traversal had regressed.
  Rejected candidates stay external, so exploring more memory costs no final prompt tokens.
- **Negative / cost:** Pairwise pass energies need calibration against a null/control candidate
  to stay comparable. Pure diffusion dropped the development split's one ranked-QK recovery, so
  the selector had to become dual-channel — one calibrated max-path attention choice protected as
  an exploitation slot (turn 306). Diffusion recovered none of the three unreachable answers:
  read-stage quality is now capped by write-time link coverage.
- **Follow-ups:** Guards against known failure modes (degree normalization, restart/leak,
  per-source caps, novelty penalties, minimum diversity allocation, turn 297). Whether one
  protected exploitation slot suffices on larger stores is untested. The unreachable misses hand
  the baton back to the write-time coverage work surfaced by
  [DR-0006](0006-pivot-to-performance-rig.md).

## Alternatives considered

- **One-shot attention classification over a fixed candidate triple** — score three candidates
  jointly in one context. Rejected: cross-candidate softmax interference, candidate-order and
  length bias, and re-normalization make the joint context fragile; pairwise/streamed scoring
  gives constant transformer memory.
- **Deeper best-path traversal** — extend the existing ranked-QK walk to three hops. Rejected by
  measurement: it regressed recall to 83.3%/75% because it retains one best path instead of
  summing corroborating paths.
- **Greedy walk (beam width one)** — each step selects exactly one next item. Rejected for
  recall: an early wrong transition permanently loses the correct route; a tiny external beam
  (about four IDs) is kept while each head invocation still picks one next item per active node.
- **Raw attention weight as global heat** — rejected: attention is locally normalized within one
  workspace, so workspace composition would change the total; heat lives externally, moved by
  calibrated transition probabilities.

## Source

- **Source merged turns:** 065, 069
- **Raw sub-turns:**
  - [turn-294-user.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-294-user.md)
  - [turn-295-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-295-assistant.md)
  - [turn-298-user.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-298-user.md)
  - [turn-306-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-306-assistant.md)
- **Dev guide:** [chapter 02](../dev-guide/02-retrieval-grounding-and-heat-diffusion.md)
