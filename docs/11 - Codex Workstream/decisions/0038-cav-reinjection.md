# 0038. Reinject CAV instead of recomputing it

- **Status:** Accepted
- **Date:** 2026-08-23
- **Tag:** LOCK-IN

## Context

The working design had drifted away from a fourth retrieval/representation
layer that existed in the original architecture. When the user asked "There
was a fourth layer on top of this, do you remember what it was?", the first
answer misidentified it as the LLM synthesis/rescoring overlay. The user
corrected the record: "Ah the the idea was reinjecting CAV instead of
computing it."

The subsequent audit found why the layer had been forgotten: in the canonical
pair of rectangular passes — `C0 → C1` extraction (concepts extract from
evidence) followed by `X → X1` reinjection (evidence nodes receive from
concepts) — the implementation computed `X1` and then discarded it before any
downstream use. With the real layer silently inert, interim interpretations
("fixed concept bank", "answer-model injection") had crept into the logs in
its place. Recomputing the question/evidence direction from the full context
each time would repay exactly the cost the ladder exists to avoid.

## Decision

Restore cached CAV reinjection as the fourth layer: reuse the previously
computed question/evidence direction in the model's hidden state instead of
recomputing that representation from the full context. Fix the load-bearing
bug so the `X → X1` reinjection pass is actually consumed downstream, record
both passes without persisting hidden states, and correct the theory note
(`docs/00 - Theory/graph_transformer_cav_summary.md`) so the canonical
`C0 → C1` / `X → X1` architecture — not the interim interpretations — is the
documented contract.

## Consequences

- **Positive:** The cached direction is reused at near-zero marginal cost,
  where recomputation from the full context repays the cost the ladder avoids.
  The measured system and the described architecture agree again, and the
  corrected theory note plus recorded passes guard against re-drift.
- **Negative / cost:** True reinjection at answer time still lacks a consumer:
  the remote responder consumes text only, so the earlier text-ordering "CAV
  treatment" survives as an explicit proxy for the real layer until synthesis
  can accept latent links.
- **Follow-ups:** DR-0040 clarifies what this layer *is* — CAV as the
  linking/fusion layer over S0-S3 evidence (link, enrich, retrieve no new
  text, preserve membership and provenance), positioned between S3 and LLM
  synthesis/rescoring (DR-0036) in the cumulative ladder. v2 CAV artifacts
  must carry real link receipts (`fast_cav_link_synthesis.py`,
  `fast_cav_links.py`); v1 artifacts still replay exactly.

## Alternatives considered

- **Recompute CAV from the full context** — recompute the question/evidence
  direction on demand instead of caching it. Rejected: it repays the
  full-context cost the ladder exists to avoid, and is precisely what the
  forgotten layer was designed against.
- **LLM synthesis overlay as the fourth layer** — the assistant's first
  recollection when asked. Rejected by the user's correction; the overlay is a
  distinct downstream layer above the ladder (DR-0036), not the fourth
  retrieval/representation layer.
- **Fixed concept bank** — an interim interpretation that filled the gap while
  the layer was forgotten. Explicitly retired in the corrected theory note.
- **Answer-model injection** — the other drift interpretation (inject into the
  answer model rather than reinject into evidence nodes). Likewise retired as
  never the intended fourth layer.

## Source

- **Source merged turns:** 455, 457
- **Raw sub-turns:**
  [turn-3320-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3320-user.md),
  [turn-3321-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3321-assistant.md),
  [turn-3323-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3323-user.md),
  [turn-3324-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3324-assistant.md)
- **Dev guide:** [chapter 09](../dev-guide/09-acceleration-scoring-and-ladder-restoration.md)
