# 0035. Re-lock the linear-cumulative "ultimate" design

- **Status:** Accepted
- **Date:** 2026-08-21
- **Tag:** LOCK-IN

## Context

The fresh 1M run that DR-0034 unblocked completed mechanically but regressed
badly: 3/10 literal answers against frozen v3's 6/10 on the identical shard
and questions, at 2.37x the context spend. Diagnosis showed the run was not
the requested recreation — the retrieval treatment had been silently swapped
from frozen v3 `causal_graph` (direct chunks + causal expansion + coverage
selection) to an `episode_primary` replacement stack with no direct-chunk
fallback. The agent had drifted from the agreed development process.

The user halted the drift explicitly: "Let's stop, do you have all the tools
in place to do this you were supposed to linearly improve the retrival cases
for each new method but you ended up doing something else." The agent's
answer confirmed the diagnosis — "I treated `episode_primary` as a
replacement architecture. Your intended process was cumulative" — and
enumerated what was still missing: a fixed reproducible 1M baseline, a
cumulative retrieval ladder, a monotonic fallback so a new method cannot
discard the previous winner's evidence, per-stage traces, and same-budget
A/B scoring with regression gates. The user then issued the lock-in: "Ok
build it and document it if you haven't already."

## Decision

Re-lock the linear-cumulative development contract and build its enforcement
machinery: baseline, then add one retrieval method at a time, preserving all
prior evidence, tested on identical corpus/questions/budget, kept only if it
improves. Implement the "ultimate" version as a recall-preserving composite
route — frozen v3 `causal_graph` evidence authoritative and byte-frozen,
representative episodes and artifact-wide closure strictly additive, packet
assembly carrying per-stage provenance — with monotonicity enforced at
return time so every new arm must include the previous arm's admitted
evidence before adding its own. Keep existing routes unchanged as controls,
and document what is implemented versus what still requires a full 1M
measurement.

## Consequences

- **Positive:** The regression class that caused the 3/10 result becomes
  unrepresentable rather than merely discouraged — an additive stage that
  cannot fit in budget becomes a no-op instead of evicting predecessor
  evidence. Per-stage traces make any future all-zero score localizable to
  its first failing gate. Episode retrieval survives, demoted to an additive
  breadth stage rather than sole authority.
- **Negative / cost:** The composite route can never beat the baseline by
  reallocating the baseline's budget — v3's prefix is fixed cost on every
  arm. Substantial harness work (cumulative experiment layer, monotonicity
  receipts, phased 1M checkpoints) precedes any new measured result, and the
  measured cumulative 1M number did not yet exist at phase close.
- **Follow-ups:** Dispatch inconsistencies between ordinary and diffuse
  retrieval, flagged in the re-lock's gap list, remain open. This re-lock is
  the governing contract for the phase-09 ladder-restoration work (DR-0039,
  DR-0040), which restores and scores the cumulative retrieval ladder under
  the same monotonicity rules.

## Alternatives considered

- **Continue with `episode_primary` as the replacement architecture** — the
  path the agent had drifted onto. Rejected by measurement: on the identical
  shard and ten questions it scored 3/10 literal against frozen v3's 6/10
  (93.3% source recall) at over twice the packet tokens, and it changed the
  population while removing prior safety nets, invalidating the progression.
- **Winner-take-all stack selection per run** — compare stacks and pick the
  best whole architecture each time. Rejected in the design response: the
  composite is explicitly "a recall-preserving composite instead of choosing
  one stack over another," because stack choice is exactly the mechanism
  that let admitted evidence be silently discarded.
- **Attempt the ultimate route immediately without the harness** — the agent
  declined this itself ("No — not yet... the experimental harness needed for
  trustworthy linear improvement does not [exist]"), stopping implementation
  until the cumulative experiment layer could prove each rung.

## Source

- **Source merged turns:** 425, 427
- **Raw sub-turns:**
  [turn-2289-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2289-user.md),
  [turn-2290-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2290-assistant.md),
  [turn-2291-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2291-user.md),
  [turn-2292-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2292-assistant.md)
- **Dev guide:** [chapter 08](../dev-guide/08-1m-test-execution-and-regression.md)
