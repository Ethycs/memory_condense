# 0011. Define consolidation as live per-turn association/strengthening

- **Status:** Accepted
- **Date:** 2026-08-16
- **Tag:** LOCK-IN

## Context

The architecture review against the state of the art had identified consolidation — the
step that turns multiple episodic fragments into a compact current conclusion with
provenance links — as the diagnosed missing layer, and the user directed sequencing:
"You've identified it as a limiting factor and it's a part we haven't built yet, we
should build it before we run any more assessments."

The first implementation sketch was conventional: provenance-preserving semantic
consolidation into cold/era memories, i.e. periodically writing summaries. The user
redefined the mechanism in one sentence: "I think consolidation should take place using
subsequent chats/prompts. in other words, just like we have a decay system, we should
have an association/strengthening system." The system already had a principled decay
path with no symmetric opposite; consolidation as a scheduled batch job would have
broken the text-free/scalar persistence discipline and required storing and regenerating
text.

## Decision

Define consolidation as online consolidation by use — live, per-turn
association/strengthening driven by subsequent prompts, mirroring the decay system.
Every later prompt strengthens the bounded links among the memory items and evidence
actually exposed together; unused links decay; repeated assemblies become easy to
reactivate from any member. Persist this as a schema-v8, text-free, model-independent
consolidation graph over typed memories and source chunks: strong edges fill bounded
reserved slots, packing stays under unchanged budgets, and only items that actually
reached the model are reinforced. Edges require repeated independent co-activation,
decay in turn-space, and are degree-pruned. Frozen Qwen activations (CAV coordinates,
QK association strength, OV transport alignment) act as a transient per-turn association
teacher gating edge updates, with rank-only operation as the fallback; no prompt text,
K/V, or hidden state is ever stored.

## Consequences

- **Positive:** Consolidation becomes an ongoing property of use rather than a scheduled
  job; requires no stored text or model state; reuses the retrenched Hebbian graph
  (DR-0009) as its seed, this time wired into normal context assembly. Validated in the
  four-arm ablation: Qwen consolidation reached 38/39 evidence recall versus 35/39
  original, with no regressions.
- **Negative / cost:** Learning is not outcome-conditioned — the graph strengthens what
  was retrieved together, not what produced correct answers, a self-reinforcement risk
  requiring the edge-hygiene rules (independent co-activation; graph-recalled items
  cannot reinforce their own links). Delayed per-turn consolidation needs the Qwen model
  resident (~0.67 s per event versus a 13 s per-process load). Mature assemblies get no
  compact semantic representative yet.
- **Follow-ups:** Schema v8 (and v9 causal prompt→response bindings), the four-arm
  replay rig, and the `bae4bca` commit implement this decision. Answer-stage validation
  still runs through the operational test (DR-0010). Outcome-conditioned learning
  signals remain future work.

## Alternatives considered

- **Periodic LLM-summary consolidation** — the initial implementation target
  ("periodically write an LLM summary" into cold/era memories). Rejected: it requires
  storing and regenerating text, breaks the text-free/scalar persistence discipline, and
  continuous textual rewriting risks corrupting useful memories; authoritative evidence
  is never rewritten.
- **Persisting per-turn residual-space objects** — storing the Qwen "hyperplane per
  turn" itself. Rejected when the Qwen teacher was designed in: the projection must be
  transient, persisting only the strengthened sparse relationships, not an ever-growing
  residual-space object per turn.
- **Qwen as a hard dependency** — rejected; the Qwen pass is a weighted-input seam with
  rank-only fallback, so consolidation works with no checkpoint loaded.

## Source

- **Source merged turns:** 097, 100
- **Raw sub-turns:**
  [turn-509-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-509-user.md),
  [turn-511-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-511-user.md),
  [turn-512-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-512-assistant.md),
  [turn-513-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-513-assistant.md)
- **Dev guide:** [chapter 03](../dev-guide/03-95-percent-associative-memory-campaign.md)
