# 0036. Add LLM synthesis/rescoring of S1-S3 episodic evidence

- **Status:** Accepted
- **Date:** 2026-08-22
- **Tag:** LOCK-IN

## Context

The completed real 1M run showed 100% source recall at every stage but flat
evidence F1 from S1 through S3: "S1 added 171 evidence items and 44,110 tokens
across the questions, but improved F1 on only 2/10. S2 added 1,714 more tokens
with no measured gain." The audit found the scoring path breaks before final
packing — direct episodes inherit anchor rank and temporal decay,
representatives get Qwen QK/OV relevance, closure replaces that with obligation
weight plus relation confidence, and "packing has no semantic
evidence-per-token score; token cost is only a late tie-break." Episode
relevance never propagates into final evidence density, so the episodic layers
pay tokens without buying evidence.

The user asked directly whether an LLM could "synthesize the large context,
categorize each by the density of evidence it provides," then locked it in:
"Can you use an llm to synthesize and rescore S1 through S3?" — and directed
the overlay at the LiteLLM gateway ("maybe you should use the litellm
endpoint") rather than the local model.

## Decision

Add an LLM overlay above the cumulative retrieval ladder that synthesizes and
rescores S1-S3 episodic evidence, running against the LiteLLM (Terra)
endpoint. Each episodic addition receives two labels — an evidence role
(decisive, supporting/temporal bridge, qualifier/conflict, context, redundant,
irrelevant) and a density band (critical, high, medium, low, none, unknown) —
followed by citation-bound extractive synthesis (atom IDs and quote hashes;
unverifiable claims discarded) and answering from the compressed, labeled
evidence. The overlay operates only on bounded, already-retrieved evidence,
never the million-token corpus, and it is not S4: it changes answer
construction, not retrieval.

## Consequences

- **Positive:** Closes the broken scoring path — a semantic evidence-per-token
  signal finally reaches selection and answer construction. On the same
  1,039,203-token test the ladder went from 0/10 EM / 0.0102 F1 (local Qwen
  synthesis) to 5/10 EM / 0.7184 F1 at S1, showing answer synthesis, not
  retrieval, had been hiding the ladder's value. Role/density labels also
  diagnose the layers themselves (all five S2-only additions scored
  irrelevant/none).
- **Negative / cost:** Adds a remote LLM dependency (gateway availability,
  latency, per-call cost) to the evaluation loop; mitigated by durable
  request/response checkpoints and byte-identical normalized scoring replay.
- **Follow-ups:** Fast benchmark runs to exercise this loop cheaply
  (DR-0037); the overlay's position at the top of the ladder is later fixed as
  the layer above CAV reinjection (DR-0038) in the cumulative ladder
  (DR-0040). Whether density-aware packing rescues S2/S3 remains open.

## Alternatives considered

- **Local forced-choice scorer only (no generation for v1)** — score whether an
  exact sentence proves an answer value using the existing
  `causal_choice_scorer`, run between novelty projection and packing. Proposed
  as the generation-free first version; superseded when the user asked for
  full LLM synthesis and rescoring of S1-S3.
- **Separate density branch beside a frozen control** — keep raw S1→S2→S3
  frozen and add a parallel D1-D4 density ladder for a matched comparison.
  Not taken; the LLM overlay rescored the existing ladder directly instead.
- **Local Qwen answer synthesis** — the incumbent path; scored 0/10 EM with
  0.0102 F1 on evidence the Terra endpoint answered at 5/10 EM / 0.7184 F1.
  Qwen remains where it is strong (feature extraction, routing, forced-choice
  scoring); synthesis and rescoring move to the gateway.

## Source

- **Source merged turns:** 435, 441
- **Raw sub-turns:**
  [turn-2437-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-2437-user.md),
  [turn-2440-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-2440-assistant.md),
  [turn-2441-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-2441-user.md),
  [turn-2460-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-2460-user.md)
- **Dev guide:** [chapter 09](../dev-guide/09-acceleration-scoring-and-ladder-restoration.md)
