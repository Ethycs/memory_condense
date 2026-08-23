# 0009. Retrench after Hebbian work shows zero evidence gain

- **Status:** Accepted (Hebbian arm later restored by DR-0039)
- **Date:** 2026-08-16
- **Tag:** SCOPE-CUT

## Context

The first half of the phase built a live Hebbian co-retrieval graph at the user's
direction — a time-decayed co-visitation graph with rank-weighted strengthening, hub
normalization, bounded degree, and one-hop recall. It was correctly bounded, persisted
only IDs and scalars, and passed its tests. But the user's reprimand was blunt: "I don't
see you making good progress to 95%."

The assistant conceded immediately: "The Hebbian work added a bounded mechanism but
produced **zero evidence toward 95%**; it should not have displaced benchmark-driven
work." Investigation then showed the problem was worse than lost time — the "locked"
benchmark the mechanism would have been judged against was itself invalid. The run used a
stale 500-record file with the wrong SHA, the development trace contained six
non-abstention questions with blank gold answers, and the 200-question "development" work
had been built from the **oracle corpus** — traces that essentially contained the answer
sessions, explaining the artificial 100% evidence-source coverage. In the assistant's
words: "we were optimizing an oracle diagnostic, not making credible progress toward 95%
on long chats."

## Decision

Stop architecture additions and retrench to benchmark-driven work. Establish the exact
current score, separate retrieval misses from answer-generation misses, and change only
the bottleneck accounting for the largest recoverable error set — no more architecture
additions without a measured delta. As part of the retrenchment, correct the benchmark
substrate first: download the current official LongMemEval-S artifact, verify its
published SHA, audit schema and population, and create a new lock before touching
retrieval, replacing the oracle-corpus workflow with a staged real-corpus run.

## Consequences

- **Positive:** Produces the valid benchmark substrate the whole phase depends on (real
  277 MB LongMemEval-S, SHA-verified, locked development preflight with reusable stores).
  Establishes the working rule that gates the rest of the campaign: measured delta or no
  change.
- **Negative / cost:** The Hebbian overlay's engineering effort yields no immediate
  benchmark benefit; the mechanism sits unwired. Correcting the substrate invalidates the
  prior 47% figure and all oracle-corpus results, restarting the measurement baseline.
- **Follow-ups:** The retrenchment was a sequencing correction, not a verdict on the
  idea: the Hebbian graph returns within the same phase as the seed of the consolidation
  layer (DR-0011), and a dedicated Hebbian retrieval arm is restored to the evaluation
  ladder much later under full measurement discipline (DR-0039, phase 09).

## Alternatives considered

- **Continue extending the Hebbian overlay** — the immediately preceding discussion had
  sketched feeding the learned Hebbian edges into the existing personalized-PageRank-like
  heat diffusion. Rejected because the mechanism had zero measured evidence toward 95%
  and the benchmark it would be scored on was invalid; building further would compound
  unmeasured work.
- **Keep optimizing against the existing "locked" run** — rejected once inspection showed
  it was a saturated literal-string diagnostic on a stale, partly malformed file, and the
  broader development set was the oracle corpus rather than a real long-chat retrieval
  test.
- **Discard the Hebbian graph entirely** — not chosen. The code remained, bounded and
  tested; only its priority was cut. This preserved it as the seed for DR-0011.

## Source

- **Source merged turns:** 081, 082
- **Raw sub-turns:**
  [turn-447-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-447-user.md),
  [turn-448-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-448-assistant.md),
  [turn-449-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-449-assistant.md),
  [turn-464-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-464-assistant.md)
- **Dev guide:** [chapter 03](../dev-guide/03-95-percent-associative-memory-campaign.md)
