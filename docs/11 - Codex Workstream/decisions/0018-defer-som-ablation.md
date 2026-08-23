# 0018. Defer SOM to a later ablation

- **Status:** Accepted
- **Date:** 2026-08-17
- **Tag:** SCOPE-CUT

## Context

The same four-algorithm triage that adopted TF-ISF and HSC
([DR-0017](0017-tf-isf-hsc-adoption.md)) had to dispose of the fourth
candidate: the `claude-som` Self-Organizing Map plus pathfinding approach,
which organizes embeddings into a 2D conceptual grid and constructs narratives
from paths through the map. It had plausible uses — detecting broad topical
regions, finding isolated memories, diversifying retrieved candidates across
conceptual neighborhoods, choosing coarse partitions before exact retrieval.

But the assessment against the project's retrieval requirement was direct
(turn 786): "Mapping high-dimensional semantics into 2D loses distinctions.
Incremental training can move previously assigned regions. Nearby map cells do
not guarantee evidence relevance. Pathfinding optimizes conceptual traversal,
not necessarily recall." With TF-ISF and HSC identified as the moves most
likely to improve the 1M retrieval result, building a fourth arm now would
spend effort on the least promising candidate before the first two had been
benchmarked.

## Decision

Do not build the SOM arm now. Keep SOM as a later partition/diversity
ablation, to be picked up only if the TF-ISF and HSC work exposes a clear need
for coarse partitioning or candidate diversification that those layers cannot
supply.

## Consequences

- **Positive:** Phase 05 effort concentrates on the two arms with the
  strongest fit; no incremental-training instability or 2D-projection loss
  enters the retrieval path; the phase's compression results stay attributable
  to a smaller set of mechanisms.
- **Negative / cost:** No empirical number for SOM on this workload — the
  deferral is an assessment, not a measured rejection; any partition-level
  diversity benefit SOM might offer remains untested.
- **Follow-ups:** This is the deferred half of the triage whose accepted half
  is [DR-0017](0017-tf-isf-hsc-adoption.md). Later in the phase, coarse
  partition routing was tested by other means and rejected
  ([DR-0022](0022-reject-two-partition-routing.md)), which weakens — but does
  not formally close — the case for a SOM partitioning ablation.

## Alternatives considered

- **SOM as the primary retrieval index** — rejected outright: 2D projection
  loses semantic distinctions, incremental training moves previously assigned
  regions, and map adjacency does not imply evidence relevance.
- **Build the SOM partition/diversity arm immediately alongside TF-ISF and
  HSC** — rejected as sequencing: the priority list (turn 786) put TF-ISF
  benchmarking and provenance-preserving HSC first, and turn 788 committed to
  keeping SOM "as a later ablation unless the first two expose a clear need
  for it."

## Source

- **Source merged turns:** 186
- **Raw sub-turns:**
  - [turn-784-user.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-784-user.md)
  - [turn-786-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-786-assistant.md)
  - [turn-788-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-788-assistant.md)
- **Dev guide:** [chapter 05](../dev-guide/05-packet-compression-and-operational-replacement.md)
