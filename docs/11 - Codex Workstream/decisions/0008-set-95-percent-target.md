# 0008. Set 95% accuracy on long chats as the target

- **Status:** Accepted
- **Date:** 2026-08-16
- **Tag:** PIVOT

## Context

Up to this point the work had been steering by token reduction and small local
containment replays. Those replays showed 83-100% containment, but the assistant's own
assessment at the decision moment was that they were "too small and too consumed to
establish that target" — they could not certify performance on genuinely long chats, and
optimizing token savings without an accuracy gate left the campaign without a falsifiable
success criterion.

The user cut through this directly: "I want you to target 95% accuracy on long chats."
This is a single quantitative target that subordinates everything else — retrieval
mechanisms, token budgets, and the QK/heat policy all become candidates to be measured
against it rather than goals in themselves.

## Decision

Adopt 95% accuracy on long chats as the explicit optimization target for the campaign.
Treat accuracy as the hard gate and measure token savings only among configurations that
clear it; treat the current heat/QK policy as one candidate, not a commitment; and stop
using the small local containment replays as evidence of progress toward the target.

## Consequences

- **Positive:** Every subsequent architecture addition in the phase can be judged by a
  measured delta toward one number. This target is what later exposes the Hebbian
  overlay's zero evidence gain (DR-0009) and forces the correction of the benchmark
  substrate to the real LongMemEval-S corpus.
- **Negative / cost:** Token reduction is demoted from objective to tiebreaker.
  Long-chat evaluation is expensive — the corrected corpus runs at roughly 115k tokens
  per question — so progress checks become slower and heavier than the local replays.
- **Follow-ups:** Requires a valid benchmark on which "95%" is meaningful (the corpus
  correction lands under DR-0009's retrenchment) and an operational definition of what is
  being scored (settled by DR-0010). The target is deliberately not declared reached at
  end of phase: 97.44% literal evidence recall on 39 questions is not answer-stage
  accuracy on ≥100 judged questions.

## Alternatives considered

- **Token reduction as the primary objective** — the implicit prior stance: maximize
  compression on local probes. Rejected because compression without an accuracy gate has
  no success criterion; accuracy becomes the gate and token savings the secondary metric.
- **Keep the small containment replays as the yardstick** — the existing 83-100%
  containment runs. Rejected at decision time as too small and too consumed to establish
  a 95% claim on genuinely long chats.

## Source

- **Source merged turns:** 073
- **Raw sub-turns:**
  [turn-316-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-316-user.md),
  [turn-317-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-317-assistant.md)
- **Dev guide:** [chapter 03](../dev-guide/03-95-percent-associative-memory-campaign.md)
