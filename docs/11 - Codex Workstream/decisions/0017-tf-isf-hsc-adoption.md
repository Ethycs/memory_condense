# 0017. Adopt TF-ISF activation with minimal HSC layer

- **Status:** Accepted
- **Date:** 2026-08-17
- **Tag:** LOCK-IN

## Context

Phase 05 opened with a 6,203-token retrieval packet that had to shrink without
losing the 1M-token retrieval result, and with a candidate-competition problem:
source activation was still inferred from a ranked chunk prefix, so terms that
distinguish one conversation from all others carried no dedicated signal. A
review of the four `layer_context_seg` compression algorithms (Forest Atlas,
DHS, HSC, SOM) framed the choice: "Forest Atlas and an adapted HSC are highly
relevant; SOM is more useful as an auxiliary partitioner than the primary
retriever" (turn 786).

The key observation motivating TF-ISF was that BM25's chunk-level IDF answers
the wrong question for source routing: a term like `cerulean` occurring in 40
chunks of a single conversation looks common to chunk-IDF but is maximally
distinctive to inverse *source* frequency. "TF-IDF/BM25: which chunk contains
the relevant wording? TF-ISF: which conversation or memory partition owns this
concept?" (turn 791). Separately, the system lacked a consolidation hierarchy;
HSC's multi-level contraction was "almost exactly the missing consolidation
hierarchy," but its published form is batch-oriented and LLM-synthesizing,
while the project requires no-model, provenance-preserving structure. A code
audit confirmed the fit: the existing consolidation graph "stores only durable
IDs and scalar edges, which is exactly the provenance discipline HSC needs"
(turn 789).

## Decision

Adopt TF-ISF source activation as a separate, bounded, opt-in activation
channel — never modifying BM25 chunk scores or widening the prompt — and add
the smallest provenance-preserving incremental HSC layer that can be updated
live: raw chunks as immutable leaves, parent nodes holding only retrieval
cards, TF-ISF weights, child IDs, time ranges, and Hebbian strength, with
retrieval descending hot branches and always hydrating original evidence.
Benchmark against the locked `65/750/2250` configuration.

## Consequences

- **Positive:** Gives source routing its own no-model signal (which
  conversation owns a concept) orthogonal to chunk-level BM25; provides the
  missing consolidation hierarchy without a model call at index or query time;
  parent nodes never hold the only copy of a fact, so hydration always ends at
  raw evidence. This pipeline stage carried the packet toward the phase's
  first 100% source-coverage result.
- **Negative / cost:** Two new bounded channels to tune and audit; the
  published batch HSC contraction had to be reworked into an incremental form
  (reconsider only the new node, its nearest concepts, and affected
  ancestors), with periodic background rebuilds to correct local mistakes.
- **Follow-ups:** The HSC channel width immediately became the next question —
  resolved by the four-slot allocation and reversible pruning in
  [DR-0019](0019-four-slot-hsc-reversible-pruning.md). SOM, the fourth
  algorithm in the same triage, was deferred in
  [DR-0018](0018-defer-som-ablation.md). Small-model labeling of HSC
  parents/retrieval cards is permitted only as an additional index with
  provenance, never as a replacement for raw evidence.

## Alternatives considered

- **Small-model summarization as the compressor** — replacing retrieval with
  small-model summaries; subordinated because summary cards are acceptable
  only as an additional provenance-carrying index, and the no-model pipeline
  captured the compression win first.
- **DHS-style global PageRank clustering** — PageRank anchors plus LLM
  clustering; rejected because the existing query-seeded heat diffusion plus
  Hebbian co-access graph is already the retrieval-oriented version, and
  global PageRank favors frequently connected memories over query-relevant
  ones. Only the personalized-PageRank framing was kept.
- **Prim's MST replacing the Hebbian hypergraph** — rejected as a
  replacement; at most a sparse navigational backbone inside a partition,
  with the richer graph retaining temporal, co-access, contradiction, and
  provenance edges.
- **SOM as primary retrieval index** — rejected for retrieval and deferred as
  an ablation; see [DR-0018](0018-defer-som-ablation.md).

## Source

- **Source merged turns:** 186
- **Raw sub-turns:**
  - [turn-784-user.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-784-user.md)
  - [turn-786-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-786-assistant.md)
  - [turn-788-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-788-assistant.md)
  - [turn-789-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-789-assistant.md)
- **Dev guide:** [chapter 05](../dev-guide/05-packet-compression-and-operational-replacement.md)
