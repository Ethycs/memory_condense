# 0039. Restore the Hebbian arm to the evaluation ladder

- **Status:** Accepted (restoration in progress at conversation end)
- **Date:** 2026-08-23
- **Tag:** LOCK-IN

## Context

The Hebbian co-retrieval graph from the August 16 theory work — retrenched in
DR-0009 as unmeasured architecture, then preserved as tested code — had an
update rule, pruning, lookup, and unit coverage, but had silently fallen out
of the executable evaluation ladder. The user asked point-blank: "what
happened to the hebbian arm?" The audit confirmed the suspicion that "the
fast CAV experiment bypassed the Hebbian path instead of nesting it," and
verification against the sealed 1M store was categorical:
"`hebbian_access_events = 0`, `hebbian_chunk_edges = 0`, and
`hebbian_chunk_nodes = 0`. The code was exported and test-covered, but
outside tests there was no benchmark caller producing/consuming that state."

This was the second time in the phase that a designed layer existed in theory
and tests but not in the measured system (after the discarded `X1` CAV
reinjection of DR-0038). The distinction had to be recorded explicitly so
"implemented" is no longer confused with "actually exercised."

## Decision

Restore the Hebbian arm as a complete, replayable, measured component of the
1M development experiment rather than dormant code. Reconstruct co-access
history causally from the sealed 5,400-turn combined transcript — each
simulated retrieval sees only state that existed before the current turn,
with no test-question gold present — derive the graph from the sealed
2,379-event history (5,978 nodes, 51,072 edges), and evaluate an H1 arm that
allows at most one budget-neutral Hebbian tail replacement in the sealed S0
evidence packet against a matched control
(`src/memory_condense/eval/run_fast_1m_hebbian.py`,
`hebbian_derived_store.py`, `hebbian_history.py`; Research Log 37).

## Consequences

- **Positive:** The arm exists as a measured, fail-closed component (152
  focused tests; implementation SHA matched to the sealed history receipt)
  instead of dormant code, and the causal-reconstruction design keeps the
  measurement honest. The matched run produced a real, recorded result — a
  negative one: base 6/10 normalized EM at 0.836 F1 versus H1 5/10 at 0.736
  (Research Log 37, 2026-08-22). H1 made three replacements; two were
  answer-neutral and one removed decisive evidence.
- **Negative / cost:** The one-tail-replacement policy loses evidence and
  does not earn promotion; the restoration effort yields no immediate score
  gain. The sibling H1-vs-base framing itself violated the cumulative
  ladder's rule that later layers act only on unresolved cases, which forced
  the follow-on correction in DR-0040.
- **Follow-ups:** Reposition Hebbian retrieval as an auxiliary expansion
  signal inside the cumulative ladder, keeping the sibling run only as a
  negative ablation (DR-0040). A promotable, guarded replacement policy
  remains open work. Restoration was still in flight when the conversation
  ended (the per-case ledger conversion and the linked end-to-end test had
  not run).

## Alternatives considered

- **Leave the arm dormant as tested-but-unwired code** — the status quo since
  DR-0009. Rejected because zero rows in every Hebbian table showed the
  mechanism had never been exercised by any benchmark caller; keeping it
  would perpetuate the confusion of "implemented" with "measured."
- **Nest Hebbian expansion inside the existing fast CAV experiment** — the
  path the fast CAV run had bypassed. Not taken for the restoration itself;
  instead a dedicated matched H1-vs-base development experiment was built so
  the arm's effect could be isolated against a control on sealed evidence.

## Source

- **Source merged turns:** 463, 464
- **Raw sub-turns:**
  [turn-3365-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3365-user.md),
  [turn-3366-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3366-assistant.md),
  [turn-3399-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3399-assistant.md),
  [turn-3459-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3459-assistant.md)
- **Dev guide:** [chapter 09](../dev-guide/09-acceleration-scoring-and-ladder-restoration.md)
