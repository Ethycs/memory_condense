# 0024. Build a minimal CAV event/concept-link reachability layer

- **Status:** Accepted
- **Date:** 2026-08-18
- **Tag:** LOCK-IN

## Context

With reachability locked in as the objective (DR-0023), the open question was
whether the event-semantics idea — typed links between events and concepts,
usable at query time to collect complete sets — was an open research problem
or something buildable now. The user probed exactly that: "I was trying to
get a sense as to if the problem was open or not, then let's push to do it"
(turn 989). The answer: "the open part is the integration, and we can test it
without pretending the whole research problem is solved" (turn 990).

The risk to manage was complexity without evidence. A neural CAV/QK/OV
compiler for links might add machinery without adding recall, so the build
had to be the smallest version that could be measured against a deterministic
baseline, and it had to avoid persisting heavy transformer state.

## Decision

Build the smallest reachability-focused version of the event-semantics idea:
event-sized spans, typed event/concept links with provenance, and a
query-time set collector. Let CAV/QK/OV populate or validate links when
available, and keep deterministic extraction as the control arm so the neural
compiler must demonstrate added recall rather than merely added complexity.
Persist only compact state — one float32 CAV coordinate per chunk, no
activations or K/V — and cap the live Qwen workspace at 8 candidates / 1,024
tokens.

## Consequences

- **Positive:** The layer shipped and was measurable: an event CAV probe at
  93.8% held-out balanced accuracy; 2,478 user chunks indexed from 6,450
  transient conceptual spans; conceptual-span pooling recovered positive
  membership for two of three hard evidence sources — candidates below
  ordinary search cutoffs became reachable (turn 1052). The
  deterministic control arm makes the neural component falsifiable.
- **Negative / cost:** Headline metrics did not move — the locked 1M-token
  variants stayed at 94.7% evidence coverage and 80% complete-source
  questions. The layer added indexing and compilation cost (transient span
  extraction, probe training) for a discovery-side gain the pipeline could
  not yet convert into packed coverage.
- **Follow-ups:** The null headline result narrowed the bottleneck to
  selection and packing, prompting the raw-graph-versus-packed-prompt
  coverage diagnostic added in the same build (turn 1052). That diagnostic
  proved discovery was already at 100% and led directly to the marginal set
  selection pivot (DR-0025). Structured event records remain a query-time
  working representation over authoritative raw chunks, not the memory
  architecture.

## Alternatives considered

- **A permanent structured event schema as the memory architecture** — make
  extracted event records the thing retrieval operates on. Rejected:
  extraction will always miss implicit, ambiguous, compound, or
  previously-irrelevant meanings, so a structured entry can never be the
  condition for recall; links are kept as confidence-weighted hypotheses over
  raw chunks instead.
- **Neural-only link compilation, no control arm** — trust CAV/QK/OV to
  build the links outright. Rejected: without deterministic extraction as
  the control, there is no way to show the neural compiler adds recall
  rather than complexity (turn 990).
- **Persisting activations or K/V state per chunk** — cache transformer
  state to make links cheap to recompute. Rejected: the build deliberately
  persisted exactly one float32 CAV coordinate per chunk, keeping the store
  compact and checkpoint-agnostic (turn 1052).

## Source

- **Source merged turns:** 251, 252
- **Raw sub-turns:**
  [turn-989-user.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-989-user.md),
  [turn-990-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-990-assistant.md),
  [turn-1052-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1052-assistant.md)
- **Dev guide:** [chapter 06](../dev-guide/06-set-completion-selector.md)
