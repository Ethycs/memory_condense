# 0025. Abandon QK/OV operator construction for marginal set selection

- **Status:** Accepted
- **Date:** 2026-08-18
- **Tag:** PIVOT

## Context

This decision reverses the operator ambition of phase 01. DR-0001 committed
the project to CAV pullback over attention heads as the relational machinery,
and DR-0002 locked attention heads in as the only substrate; the phase-06
mechanism exploration had just fully specified the descendant of that
ambition — a native query-conditioned QK/OV operator in which the old memory
activation M attends to the new prompt activation N (yielding a
recontextualized M'), the prompt then reads the recontextualized memory
(yielding an enriched search state N'), implemented as a causal sandwich
`[current prompt][memory span][readout probe]` over the retained prefix
(turn 1090).

Before any of that was built, the user interrupted with a simpler
formulation: "Hold there might be a simpler way to do this, if we can check
each answer in the set to see how it partially satisfies the query, then we
can reject others like it" (turn 1091). The failing questions needed a
set-cover decision — "does this candidate add a new required answer" — not a
recontextualization. The missing chunks already existed in the candidate
pool; they were being crowded out by repeated high-scoring material, and a
selector fixes crowding directly.

## Decision

Abandon QK/OV activation-operator construction as the retrieval mechanism and
adopt query-conditioned marginal set selection instead: for each candidate
`c` given the selected set `S`, estimate
`gain(c|S) = support(q,c) * [1 - max_{s in S} P(same answer/event | q,c,s)] - lambda*tokens(c)`,
and run an online greedy loop that rejects non-supporting candidates, keeps
only the better or cheaper evidence when the same answer/event is already
selected, and otherwise accepts. Judge novelty on answer identity, not text
distance. QK/OV survive only as the scorer — QK estimates `supports_query`
and same-answer/event, OV produces the candidate's partial-answer
representation — while the deterministic selector makes every keep/reject
decision.

## Consequences

- **Positive:** Considerably simpler than inventing a new attention
  mechanism, and aimed at the proven bottleneck: if raw candidate coverage
  is 10/10, this "could plausibly take us from 8/10 to 10/10 without adding
  another learned memory representation" (turn 1092). Deduplication
  granularity follows the query (museums vs. visits vs. earliest concert),
  and the greedy loop is inspectable and cannot hallucinate evidence.
- **Negative / cost:** The selector cannot recover a chunk that never
  reaches the candidate pool, so the raw-candidate-versus-packed-context
  diagnostic remains decisive. It requires a coverage-aware conditional
  scorer `g_i = f(q, c_i, S)` rather than a standard cross-encoder's
  `f(q, c_i)`. The M'/N' recontextualization operator — the fully specified
  investment of DR-0001/DR-0002's lineage — goes unbuilt.
- **Follow-ups:** The selector was implemented as a bounded transient
  listwise judge with the INI protocol (DR-0026) on the restored six-layer
  Qwen prefix (DR-0027) under staged GPU residency (DR-0028). The
  recontextualization operator remains a documented future ablation, not a
  deleted idea, as a fallback if greedy selection hits a ceiling.

## Alternatives considered

- **QK/OV recontextualization operator (M'/N')** — the causal-sandwich
  operator of turn 1090, with per-head QK capture, query-specific clustering
  of M', and a feedback retrieval round from N'. Set aside because it
  answers "how do these representations relate" while the failures required
  "does this candidate add a new required answer"; the simpler formulation
  subsumed the need. Retained as a specified future ablation.
- **SAE-based operator construction** — filter polysemantic noise through a
  sparse autoencoder before composing the operator. Already deprioritized
  during exploration: the first experiment should test the pretrained Qwen
  circuit directly (turn 1090); made moot by this pivot.
- **Similarity-based deduplication (text distance / MMR-style)** — reject
  candidates that look like selected ones. Rejected within the pivot itself:
  two museum memories naturally sound similar but may provide different
  required answers, so novelty must be judged on answer identity (turn 1092).

## Source

- **Source merged turns:** 285, 286
- **Raw sub-turns:**
  [turn-1090-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1090-assistant.md)
  (the operator specification being abandoned, merged turn 284),
  [turn-1091-user.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1091-user.md),
  [turn-1092-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1092-assistant.md)
- **Dev guide:** [chapter 06](../dev-guide/06-set-completion-selector.md)
