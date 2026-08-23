# 0001. Pivot architecture toward CAV pullback over attention heads

- **Status:** Accepted (operator ambition later narrowed by DR-0025)
- **Date:** 2026-08-16
- **Tag:** PIVOT

## Context

Before this turn, memory_condense was a text-chunk retrieval pipeline: a
BGE-M3 + BM25 hybrid over a SQLite store, which had just come through a
performance pass (batched reheating, N+1 elimination — merged turn 006). That
design treats every memory as an opaque text vector. It cannot represent
*which relationship* links two memories, and its recall is limited to
similarity between whole chunks.

The user redirected the design in one message (turn 018): "what about Concept
activation vector pullback followed by chunking using conceptual chunks QK
matrix OV or headoutput. the idea is build memories around CAV and link CAVs
together" — and sharpened it two turns later (turn 021): "take the
atten[tion] heads from a larger model, do recursive association with the QK
maps to related memory data as the memory layers."

The responses to both messages established the formal core that survives the
phase: the OV pullback `c_source = (W_OV)^T c` decomposes head output into
"did this head write the concept" (`c^T o_i`), "which source positions were
routed" (`A_ij`), and "how much concept-bearing content each source supplied"
(`((W_OV)^T c)^T r_j`) — QK selects *where* information comes from, OV
determines *what* gets written. A CAV is never the memory itself; it indexes
grounded episodes that retain source text and provenance.

## Decision

Redirect the project toward a live memory system built around CAVs as memory
addresses, with a larger model's attention heads as the relational machinery:
pull concept vectors back through per-head OV matrices to attribute
concept-bearing sources, use QK structure to link conceptual chunks, and
store the results as a typed graph of Concept nodes (versioned CAVs) and
Episode nodes (grounded source spans), with edges such as SUPPORTS,
COACTIVATES, ROUTES_TO, and WRITES_TO.

## Consequences

- **Positive:** typed, inspectable memory structure instead of opaque text
  vectors; a path to paraphrase, cross-turn, and compositional recall; token
  savings, because the persistent index stores sparse concept IDs and graph
  statistics rather than activation tensors or whole associative chains.
- **Negative / cost:** requires access to a large teacher model's
  activations; raw QK attention is insufficient evidence for a memory
  relationship (edges must start provisional and earn confidence); CAVs are
  tied to a model fingerprint + layer + probe version and are not portable
  across model versions.
- **Follow-ups:** decide how much of the teacher model must be retained
  (DR-0002); obtain a real, non-mock substrate (DR-0003); benchmark against
  the existing hybrid retriever at a matched context-token budget. The
  recursive QK/OV-operator ambition introduced here is later abandoned in
  favor of a marginal set-completion selector (DR-0025, phase 06).

## Alternatives considered

- **Continue the BGE-M3 + BM25 hybrid text pipeline** — the incumbent design,
  freshly optimized. Rejected because it treats memories as opaque vectors
  and cannot represent which relationship links two memories.
- **CAV chunking without head operators** — the first formulation in the
  pivot turn (chunk around CAV activations only). Superseded within the same
  session by the user's recursive-head proposal, which promised associative
  recall through QK maps rather than similarity alone.
- **Sparse dictionary-learning features as concept primitives** — noted as
  possibly better than supervised CAVs due to polysemanticity, but deferred:
  CAVs were judged the faster, more controllable prototype.

## Source

- **Source merged turns:** 007, 009
- **Raw sub-turns:**
  - [turn-018-user.md](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-018-user.md) — the CAV pullback proposal (merged turn 007)
  - [turn-020-assistant.md](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-020-assistant.md) — pullback formulation, conceptual chunking, memory graph
  - [turn-021-user.md](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-021-user.md) — heads of a larger model as recursive memory layers (merged turn 009)
  - [turn-023-assistant.md](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-023-assistant.md) — recursive memory attention; three implementation levels
- **Dev guide:** [chapter 01](../dev-guide/01-cav-attention-head-ideation.md)
