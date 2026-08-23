# 0040. Clarify CAV as the linking/fusion layer over S0-S3

- **Status:** Accepted
- **Date:** 2026-08-23
- **Tag:** LOCK-IN

## Context

After the Hebbian restoration (DR-0039) ran H1 and CAV as separate aggregate
ablations, the user pushed back on the framing twice. First: "Do you
understand that the general design has been to add additional layers of
greater complexity after each layer?" — prompting the concession "I broke
this design by testing CAV and Hebbian as separate aggregate ablations."
Then: "Wasn't CAV a linking technique?" The answer resolved what CAV *is*:
"CAV was the linking/fusion technique — not another retrieval or
answer-selection method," and "the experiment we called 'CAV treatment' only
converted those latent scores into text ordering because the remote responder
could not consume hidden states. That was a proxy ablation, not the actual
CAV linking layer."

This is the final architectural correction of the conversation: it refines
the recovered CAV reinjection layer of DR-0038 and completes the cumulative
per-case ladder contract re-locked in DR-0035, which the aggregate-arm
experiments had drifted away from.

## Decision

Fix the architecture as a cumulative complexity ladder with CAV as the
linking/fusion layer over already-gathered S0-S3 evidence, not a retrieval
method or a competing answer-rescue stage:
`S0 → S1 episodes → S2 representatives → S3 global closure → CAV links/fuses
evidence representations → reinject fused CAV information into evidence
nodes → LLM synthesis/rescoring → answer`. The CAV layer creates
query-conditioned latent links among retrieved evidence, propagates
information across otherwise separated episodes, retrieves no new text,
preserves evidence membership and provenance, and passes enriched node
representations to synthesis. Each ladder layer inherits the prior layer's
evidence and result, acts only on unresolved or weakly supported cases, and
must preserve the prior result unless it demonstrates stronger source
support — a monotonic gate that governs the synthesized answer, not CAV
itself. Relabel the text-ordering "CAV treatment" as a proxy; require real
link receipts on new v2 CAV artifacts (`fast_cav_link_synthesis.py`,
`fast_cav_links.py`) while v1 artifacts still replay exactly.

## Consequences

- **Positive:** Places every component correctly in one stack: retrieval
  layers build the evidence set, CAV links and reinjects latent information
  without changing membership, and only the downstream synthesizer proposes
  answer changes under the monotonic gate. Both rectangular CAV passes
  (concepts extracting from evidence, evidence receiving from concepts) are
  now recorded without persisting hidden states. Hebbian retrieval is
  positioned as an auxiliary expansion signal inside the ladder, with the
  H1-vs-base run kept as a negative ablation.
- **Negative / cost:** The +0.0505 F1 "CAV treatment" result is demoted to a
  proxy diagnostic, not evidence for the CAV layer itself. True reinjection
  still lacks a consumer — the remote responder accepts text only — so the
  proxy-versus-real gap persists until synthesis can accept latent links
  (v2 link receipts are the prepared interface).
- **Follow-ups:** Convert the existing S0-S3/CAV/synthesis/Hebbian outputs
  into a per-question layer progression (the `linear_case_ledger` machinery)
  — the work in flight when the conversation ends. The locked 100-question
  gate has still not been run through the fast path. This is the last locked
  decision of the conversation.

## Alternatives considered

- **CAV as a competing aggregate answer arm** — the framing the phase's own
  experiments had drifted into (CAV treatment versus base as sibling
  ablations). Rejected: "I should not model CAV as another competing
  answer-rescue stage in the ultimate stack"; it misplaces CAV and violates
  the cumulative, per-case contract of DR-0035.
- **Text reordering as the CAV layer itself** — treating the X/X1
  latent-score-to-ordering conversion as the fourth layer. Retained only as
  a labeled proxy forced by a responder that cannot consume hidden states,
  never as the layer's definition.
- **Applying the monotonic gate to CAV directly** — rejected as a
  misclassification: CAV changes representations, not answers, so the gate
  governs the synthesized answer downstream of it.

## Source

- **Source merged turns:** 467, 470
- **Raw sub-turns:**
  [turn-3463-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3463-user.md),
  [turn-3468-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3468-user.md),
  [turn-3469-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3469-assistant.md),
  [turn-3474-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3474-assistant.md)
- **Dev guide:** [chapter 09](../dev-guide/09-acceleration-scoring-and-ladder-restoration.md)
