# 0015. Refine two-hop into recurrent CAV activation

- **Status:** Accepted
- **Date:** 2026-08-17
- **Tag:** LOCK-IN

## Context

The two-hop feedback implementation (DR-0014) had just tested neutral, and its seam was
visible: "Currently Qwen selects six seeds once, concatenates their text, and performs
another retrieval." The selection was a one-shot text concatenation, not a conceptual
state that evolves as evidence accumulates.

The user proposed closing that gap with the existing prompt-CAV machinery: "Can we use
the initial activations + the new candidates as activations similar to prompt cav
window +1?" — and then pinned the exact dataflow: "I want the initial search activations
to be given the added evidence and then the combined state used to search." That
reframes retrieval as a bounded activation trajectory: a `window + 1` controller where
each candidate is scored by how it changes the current conceptual state (QK routing, OV
transport, alignment of the activation delta with unresolved concepts, novelty, minus
redundancy), rather than by static resemblance to the original question.

## Decision

Refine two-hop retrieval into recurrent CAV activation. The original-question activation
selects six items from recalled evidence; `question + selected evidence` is re-encoded
into one transient combined activation window — recomputed, never assembled by adding
raw residuals from incompatible token sequences; BGE/BM25 supplies a fresh lower-ranked
candidate pool for the unchanged query; the combined Qwen QK/OV state searches that new
pool. Six activation-selected candidates plus six scalar fallbacks occupy a fixed
12-slot reserve while 36 first-round source candidates remain protected. Preserve the
original question activation so recursion cannot drift toward an early distractor, keep
the evidence window bounded and transient, and persist only compact CAV state, scalar
deltas, selected IDs, and reinforced graph edges.

## Consequences

- **Positive:** Gives the two-hop loop the missing state: the second search is driven by
  a combined conceptual activation instead of concatenated seed text, which is the
  originally intended controller role for the Qwen slice. The mechanism ran as designed
  on the matched five-row test — 30 new candidates selected by combined activation, 60
  second-hop candidates admitted, peak workspace 936/1,024 tokens, no transformer state
  retained, no provider calls. Full suite: 764 passed.
- **Negative / cost:** Neutral on the matched test (literal hits 3/5 both arms, source
  coverage 100% both, mean context 6,669.8 → 6,672.2 tokens), so it stays an opt-in
  treatment. The existing CAV bank lives at layer 5 and requires the seven-layer prefix,
  while the fast path loads only layers 0-1 — the recurrence currently carries the
  expensive runtime.
- **Follow-ups:** Refines DR-0014; both arms await a semantic sufficiency metric before
  they can be judged fairly (the concern that drives DR-0016). The proposed comparison
  ladder — QK/OV feedback with the two-layer prefix, layer-5 CAV `window + 1`, and a
  cheaper newly trained layer-1 CAV `window + 1` (the practical target) — remains open
  work. Implementation landed in `src/memory_condense/condenser.py` and
  `src/memory_condense/qwen_rerank.py`.

## Alternatives considered

- **Keep the one-shot seed-concatenation feedback** — the just-tested DR-0014
  implementation. Superseded: selecting seeds once and concatenating their text never
  updates the conceptual state; a true `window + 1` controller evaluates candidates by
  how each changes that state, selects a small beam, updates, and retrieves again.
- **Algebraically adding raw residuals from different token sequences** — the naive
  reading of "initial activations + new candidates." Rejected explicitly: combine CAV
  coordinates or their deltas, or re-encode the combined window; raw residuals from
  incompatible sequences do not compose.
- **Full multi-hop recurrence immediately** — the general design supports iterating
  until activation change is small or a hop limit is hit; the shipped version is bounded
  to the single combined-state second hop with fixed slot reserves, keeping first-round
  evidence protected and cost capped.

## Source

- **Source merged turns:** 145, 146
- **Raw sub-turns:**
  [turn-706-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-706-user.md),
  [turn-707-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-707-assistant.md),
  [turn-708-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-708-user.md),
  [turn-712-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-712-assistant.md)
- **Dev guide:** [chapter 04](../dev-guide/04-longmemeval-debugging-and-1m-baseline.md)
