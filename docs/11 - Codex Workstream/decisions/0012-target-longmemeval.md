# 0012. Shift the target to the locked LongMemEval benchmark

- **Status:** Accepted
- **Date:** 2026-08-17
- **Tag:** PIVOT

## Context

Phase 03 closed with a development win on the local 39-probe rig: live per-turn
consolidation (DR-0011) validated against an apparatus the project itself had built.
That apparatus proved the mechanisms but could not certify them — a self-authored probe
set has no external standing, and the whole campaign's 95% gate (DR-0008) was always
meant to be judged on a benchmark the project does not control.

The user closed the loop in one line: "Ok now let's target the LongMemEval target."
The stated plan for the move preserved the phase-03 discipline while changing the
optimization target: "I'm moving from the local 39-probe development win to the actual
locked LongMemEval gate. I'll first verify the dataset/manifests and benchmark
integration, then wire schema-v9 consolidation into per-sample LongMemEval stores, run
the free reachability screen, and only consider answer/judge calls once an arm is
Pareto-worthy under the 8k cap."

## Decision

Make the locked LongMemEval benchmark — official corpus, hash-verified manifests, a
locked 40-question development set — the optimization target. Wire schema-v9
consolidation into per-sample LongMemEval stores, run the free (no-provider)
reachability screen first, and hold answer/judge calls until an arm is Pareto-worthy
under the 8,000-token prompt cap. Local probe results remain development scaffolding;
claims are made against the locked gate.

## Consequences

- **Positive:** Results become externally comparable and hash-verifiable instead of
  self-graded. The first locked n=40 confirmation landed immediately: 23/24 recoverable
  literal answers (57.5% overall), 99.5% mean evidence-source coverage, all evidence for
  39/40 questions, at 6,638 mean context tokens — matching the prior best row-for-row
  while saving 664 tokens/question, after which the development policy was frozen.
- **Negative / cost:** The locked set imports LongMemEval's metric assumptions — 16 of
  40 gold answers are not literal spans, so literal-containment has a hard ceiling that
  later forced the metric reframe (DR-0016). Transcripts average only ~104K tokens, so
  the benchmark alone cannot express a million-token claim.
- **Follow-ups:** Every subsequent phase-04 mechanism (DR-0013, DR-0014, DR-0015) is
  admitted or rejected against this locked set with matched arms at identical token
  budgets. The 100K ceiling motivates the merged 1M stress store and the reframed
  success criterion (DR-0016).

## Alternatives considered

- **Continue optimizing the local 39-probe rig** — the standing development target.
  Rejected implicitly by the pivot: it had served its purpose as a free, fast
  development gate, and further wins there certify nothing externally.
- **Go straight to answer/judge evaluation on LongMemEval** — rejected in the same
  breath as the pivot: provider calls are deferred until a retrieval arm is
  Pareto-worthy under the 8k cap, so the free reachability screen comes first.

## Source

- **Source merged turns:** 115
- **Raw sub-turns:**
  [turn-591-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-591-user.md),
  [turn-592-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-592-assistant.md),
  [turn-642-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-642-assistant.md)
- **Dev guide:** [chapter 04](../dev-guide/04-longmemeval-debugging-and-1m-baseline.md)
