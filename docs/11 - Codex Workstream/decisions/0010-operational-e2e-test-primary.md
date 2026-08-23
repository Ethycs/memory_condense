# 0010. Make the operational end-to-end test primary

- **Status:** Accepted
- **Date:** 2026-08-16
- **Tag:** LOCK-IN

## Context

The corrected-corpus preflight had just posted 100% "evidence retrieval," and the user
asked the obvious question: "what's evidence retrieval 100% isn't that what we wanted?"
The metric explanation exposed the gap. Coverage numbers (any-source 100%, all-sources
97.5%, mean coverage 99%) only measure whether an annotated evidence *session* was
represented in context. Literal answer recall was 57.5% against a hard 60% ceiling —
only 24/40 gold answers appear verbatim anywhere in the haystack, because many answers
are derived (dates, counts, durations) rather than copied. Semantic answer accuracy —
the actual 95% target — was not measured at all: "Retrieval/source selection is nearly
solved on this 40-question preflight. Answer reasoning is now the main unmeasured
bottleneck."

The user then set the priority explicitly: "The test I want to prioritize is the
operational one, basically given a finished set of turns, can we produce the outcome
without sending the whole transcript."

## Decision

Make the operational end-to-end test the primary test: ingest the completed conversation
once, ask a later question, send only the bounded memory context to the responder, and
grade the resulting answer. The benchmark headline measures answer correctness together
with transcript-to-prompt compression; each question records completed-transcript size,
retrieved-context size, fraction sent, tokens saved, the actual answer, and semantic
correctness. The 95% gate additionally fails if any prompt exceeds its configured
ceiling. Evidence-source coverage is demoted to a failure diagnostic.

## Consequences

- **Positive:** The scored quantity now matches what the system exists to do — answer
  from memory at a fraction of the transcript. Removes the false comfort of saturated
  coverage numbers; makes compression and correctness jointly visible per question.
- **Negative / cost:** The primary metric requires a responder (and judge) in the loop,
  which was not yet wired to the workspace at end of phase — so the phase closes with the
  headline metric defined but unmeasured, and the project deliberately holds no
  answer-accuracy claim. Coverage, the one metric that was saturated, no longer counts as
  success.
- **Follow-ups:** Benchmark report reworked to the operational definition (focused tests
  31/31 green at decision time). The ≥100-question judged responder run becomes the next
  honest gate for the DR-0008 target. The four-arm consolidation ablation (DR-0011) is
  evaluated on the operational evidence test that follows from this definition.

## Alternatives considered

- **Evidence-source coverage as the headline** — the status quo. Rejected because
  "at least one annotated source represented" says nothing about whether the
  answer-bearing sentence was present or whether a responder could derive the answer;
  the assistant noted the label "sounds stronger than what it technically measures."
- **Literal answer recall as the headline** — rejected as structurally capped: only 60%
  of gold answers exist verbatim in the haystack, since many are derived values the
  responder must calculate (e.g. a "two weeks" gap between two dates).

## Source

- **Source merged turns:** 085, 086
- **Raw sub-turns:**
  [turn-484-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-484-assistant.md),
  [turn-485-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-485-user.md),
  [turn-486-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-486-assistant.md),
  [turn-488-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-488-assistant.md)
- **Dev guide:** [chapter 03](../dev-guide/03-95-percent-associative-memory-campaign.md)
