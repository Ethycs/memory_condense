# 0004. Halt infrastructure drift, refocus on measured retrieval

- **Status:** Accepted
- **Date:** 2026-08-16
- **Tag:** SCOPE-CUT

## Context

Work had drifted from improving the measured retrieval result into building infrastructure for a
future J-Space compiler — an install was actively in progress. The user halted it directly:
"hold on can we check in on what we're doing, I want to focus on improving existing results"
(turn 092). The agent's own assessment agreed: "We drifted from improving the measured retrieval
result into infrastructure for a future J-Space compiler. I'm stopping that install now."

The result worth improving was already identified and concrete: CAV-gated layer-5 entry stood at
0.750/0.875 (R@1/R@3), while association evidence helped only R@3. The infrastructure work had no
bearing on either number.

A second, subtler drift surfaced immediately after the refocus: the fusion fix that produced
1.000/1.000 was informed by the four inspected evaluation errors, so it was "a development
result — not blind evidence" (turn 108). The scope cut therefore came paired with an evaluation
discipline: fresh association sets and fixed-token comparisons before any claim.

## Decision

Stop the J-Space compiler install and pause its integration, leaving no environment or dependency
behind (the downloaded weights stay inert in the ignored cache). Refocus all effort on the
existing measured retrieval path — specifically improving R@1 by fusing layer-5 entry scores with
layer-1 selected-head association evidence, tuned on calibration links and evaluated on held-out
links.

## Consequences

- **Positive:** Effort lands on a measurable target with a known baseline (0.750/0.875). The
  immediate refocus produced the calibrated direction + score fusion (1.000/1.000 on the
  development set) and pruning/utility machinery in `head_memory.py`, verified by the full suite
  (565 passed).
- **Negative / cost:** J-Space compiler capability is deferred indefinitely; the downloaded
  weights sit unused. The 1.000/1.000 headline number is explicitly demoted to development data,
  so the honest claimable result is weaker than the best observed one.
- **Follow-ups:** Fresh (untouched) association sets and fixed-token comparisons against the
  existing hybrid retriever become mandatory before claims — the discipline that later drives the
  parallel benchmark rig ([DR-0006](0006-pivot-to-performance-rig.md)) and the untouched-split
  rule throughout the phase.

## Alternatives considered

- **Continue the J-Space compiler integration** — finish the install and build toward the future
  compiler. Rejected as drift: it did not touch the measured retrieval numbers, and the user
  explicitly redirected to improving existing results.
- **Treat the 1.000/1.000 fusion result as done and move on** — the reranking fix was informed by
  the four evaluation errors it fixed, so the number is tuned development data, not blind
  evidence; the honest next step is re-evaluation on fresh links, not a victory lap.

## Source

- **Source merged turns:** 033
- **Raw sub-turns:**
  - [turn-092-user.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-092-user.md)
  - [turn-093-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-093-assistant.md)
  - [turn-108-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-108-assistant.md)
- **Dev guide:** [chapter 02](../dev-guide/02-retrieval-grounding-and-heat-diffusion.md)
