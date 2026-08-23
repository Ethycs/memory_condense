# 0014. Adopt two-hop attention-guided retrieval

- **Status:** Accepted
- **Date:** 2026-08-17
- **Tag:** LOCK-IN

## Context

After the partition-local fix (DR-0013) proved neutral, the Qwen recursive reranking
tournament came under scrutiny and was diagnosed as structurally wrong: "it recursively
applies the same unsupervised score. If QK/OV does not represent 'needed to answer this
question,' each pruning round can compound the mistake. Once an important premise loses
an early group, later rounds cannot recover it."

At the same time, long-chat retrieval was recognized as genuinely recursive in a
different sense: "How long had I been a member?" requires both "joined three weeks ago"
and "meetup last week," and neither excerpt independently resembles "two weeks" —
"The first retrieved fact should alter the search state for the second." Recursion
therefore needs a changing objective (what evidence is still missing), not a deeper
reranker over a fixed pool. The user proposed the shape directly: "should we do
attention against the recalled evidence and another round of retrieval?"

## Decision

Adopt two-hop attention-guided retrieval: run high-recall first-round retrieval; inspect
the recalled evidence with Qwen QK/OV in a bounded workspace (eight candidates, ~1,024
tokens, no retained transformer state); diffuse heat through memory associations from
the attended items; run a second retrieval round with a fixed candidate and prompt
budget; union both rounds, dedup, and finally rerank. Attention is a bounded feedback
step that adds candidates — strong first-round evidence is protected, never erased —
limited initially to two rounds, persisting only scalar heat, QK/OV-derived weights,
and access edges.

## Consequences

- **Positive:** Turns Qwen into a live memory-navigation controller — "the role
  originally intended" — rather than an expensive replacement for the existing ranker,
  while keeping every persistence invariant (IDs and scalars only). The mechanism
  demonstrably works: on the matched five-row test, 60 actual second-hop candidates were
  admitted while mean context rose only 2 tokens and the peak workspace stayed at eight
  candidates/613 tokens.
- **Negative / cost:** Measurably neutral on the matched test — literal hits stayed 3/5
  and source coverage stayed 100% — so it is recorded as an opt-in treatment, not the
  selected policy. Literal containment cannot see multi-premise value; judging the arm
  fairly requires a semantic sufficiency metric that does not yet exist. A second
  retrieval round adds latency and Qwen inference cost per query.
- **Follow-ups:** The evaluation plan requires three matched arms — one-shot hybrid
  retrieval, a standard second round via query decomposition/relevance feedback (the
  conventional multi-hop RAG control), and the Qwen-attention feedback arm. The
  mechanism was refined the same day into recurrent CAV activation (DR-0015). The
  decisive metric shifts toward semantic evidence sufficiency and answer accuracy,
  anticipating DR-0016.

## Alternatives considered

- **Deeper/wider recursive attention tournament over one fixed pool** — the incumbent
  Qwen mechanism. Rejected: it reranks the original candidates against the unchanged
  original question with the same uncalibrated score; early pruning mistakes compound
  and eliminated premises are unrecoverable — the likely reason Qwen made substitutions
  without improving recall.
- **Standard multi-hop RAG stack as the mechanism (query decomposition + trained
  cross-encoder + relevance feedback)** — acknowledged as the conventional solution, but
  kept as the named matched control arm rather than the adopted mechanism, so live
  attention can be measured against it instead of against plain cosine ordering.
- **Persisting attention state across rounds** — rejected by standing invariant: round
  two is anchored to the original question plus compact selected-premise IDs/CAV
  coordinates, following graph/source links rather than storing Qwen activations.

## Source

- **Source merged turns:** 139, 141
- **Raw sub-turns:**
  [turn-694-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-694-assistant.md),
  [turn-697-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-697-user.md),
  [turn-698-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-698-assistant.md),
  [turn-705-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-705-assistant.md)
- **Dev guide:** [chapter 04](../dev-guide/04-longmemeval-debugging-and-1m-baseline.md)
