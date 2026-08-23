# 0022. Reject the two-partition routing arm

- **Status:** Accepted
- **Date:** 2026-08-18
- **Tag:** SCOPE-CUT

## Context

The role-aware operational baseline stood at 93% gold-source coverage on the
retrieval gate, with the remaining failures on multi-event set questions. The
routing hypothesis was plausible: the 1M stress memory is ten independently
namespaced chat histories, each question's gold evidence lives in exactly one,
and the set failures looked like cross-history competition for a fixed
evidence budget — so coarse partition routing should let the budget search the
right history deeply.

Every routing configuration lost to the unrestricted baseline. One-history
routing scored 58%, traced to an ordering bug (the coarse partition vote ran
before role weighting); with the bug fixed it reached only 78% — "well below
the unrestricted 93%. I'm not promoting it" (turn 943). The two-partition
safety variant reached 86.3% at roughly twice the local scan cost: "I'm also
treating the latest two-partition arm as rejected for now: 86.3% coverage is
still below the simpler role-aware baseline's 93%" (turn 946). The
route-decision audit explained why hard routing can only destroy information:
"the correct history is rank 1 for 7/10 questions" and within the top 4 for
all 10, so "a sparse four-history beam has 100% routing recall on this set"
(turn 948) — routing recall was never the bottleneck. The missing multi-event
sessions are lost while "multiple same-concept event sources compete for
slots" (turn 950), i.e. before final packing, inside the correctly-selected
history.

## Decision

Reject the two-partition routing arm — and hard coarse partition routing
generally — and keep the unrestricted role-aware baseline as the operational
configuration. Pursue the remaining multi-event failures with a sharper fine
cue inside the routed locality (a deterministic event cue derived from
list/order questions, with reserved source-local candidates), not with coarser
routing.

## Consequences

- **Positive:** Avoids promoting an arm that is worse on both promotion
  criteria (accuracy and cost); converts the routing experiments into a
  load-bearing negative result — the bottleneck is within-locality set
  completion, not history selection — which cleanly scopes phase 06.
- **Negative / cost:** The multi-event set failures remain open at phase end
  (concerts 4/5 gold sessions, museums 4/6); the routing implementation work
  is shelved rather than promoted, retaining only the diagnostic
  instrumentation and the soft cue-beam finding.
- **Follow-ups:** The cue-sharpening experiment specified in turn 953
  (deterministic event cue such as "concerts musical events attended", with
  reserved source-local candidates, keeping the original question for ranking
  and answering) becomes phase 06's set-completion work. A learned sparse
  cue-to-partition router with its own feedback (the Neural Storage lesson,
  turn 947) is noted as the useful future form of routing but not built.

## Alternatives considered

- **One-history hard routing** — route the whole evidence budget into the
  top-voted history; 58% with the ordering bug, 78% fixed — far below the 93%
  baseline, not promoted (turn 943).
- **Two-partition (two-history safety) routing** — the rejected arm itself:
  86.3% coverage at roughly twice the local scan cost, below baseline on
  accuracy and worse on cost.
- **Soft four-history cue beam** — 100% routing recall (turn 948) and a small
  coverage nudge, but it completed neither remaining multi-event set; retained
  only as a diagnostic locality, not a promoted configuration.
- **Widening the fine-cue activation frontier (65 → 250 sessions)** — regressed
  coverage to 91.3%, "confirming that more activated sessions create
  competition rather than recovering the missing events. I'm rejecting it"
  (turn 953).

## Source

- **Source merged turns:** 232
- **Raw sub-turns:**
  - [turn-946-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-946-assistant.md)
  - [turn-948-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-948-assistant.md)
  - [turn-950-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-950-assistant.md)
  - [turn-953-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-953-assistant.md)
- **Dev guide:** [chapter 05](../dev-guide/05-packet-compression-and-operational-replacement.md)
