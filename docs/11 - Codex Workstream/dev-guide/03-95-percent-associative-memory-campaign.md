# 03 — 95% Accuracy Campaign: Hebbian Overlay and Consolidation Layer

**Phase:** 03 | **Merged turns:** 073-114 | **Dates:** 2026-08-16 to 2026-08-17

## Purpose

Hold the memory system to a single quantitative target: 95% accuracy on long chats
([DR-0008](../decisions/0008-set-95-percent-target.md)). Everything in this phase exists to
serve that number. The phase produces three durable outcomes:

1. A corrected benchmark substrate — the real LongMemEval-S corpus instead of the oracle
   corpus that earlier work had silently been optimizing against.
2. An operational end-to-end test as the primary gate
   ([DR-0010](../decisions/0010-operational-e2e-test-primary.md)): given a finished set of
   turns, produce the outcome without sending the whole transcript.
3. The consolidation layer — live, per-turn association/strengthening that mirrors the
   existing decay system ([DR-0011](../decisions/0011-live-per-turn-consolidation.md)) —
   implemented, ablated in four arms, and committed (`bae4bca`).

The phase also contains a load-bearing reversal: a Hebbian co-retrieval overlay was built
early in the phase, produced zero evidence toward 95%, and was retrenched
([DR-0009](../decisions/0009-hebbian-retrench.md)) before its core idea was rehabilitated —
under measurement discipline — as the seed of the consolidation layer.

## Design

### The operational test is the primary gate

The benchmark's headline is answer correctness together with transcript-to-prompt
compression, not evidence coverage. The operational definition:

> Ingest the completed conversation once, ask a later question, send only the bounded
> memory context to the responder, and grade the resulting answer.

Each question in the report records completed-transcript size, retrieved-context size,
fraction sent, tokens saved, the actual answer, and semantic correctness. The 95% gate
additionally fails if any prompt exceeds its configured ceiling (the CLI supplies 8,000
tokens; a `None` ceiling in older test configs means "uncapped" and skips the gate).
Evidence-source coverage remains available, but only as a diagnostic to explain failures —
never as the outcome. The distinction matters because retrieving an evidence session does
not guarantee retrieving the answer-bearing sentence, and many gold answers are derived
(dates, counts, durations) rather than copied, so "100% evidence retrieval" can coexist
with an unmeasured answer-stage accuracy.

The benchmark substrate itself is locked and verified. The earlier "development" runs used
a stale 500-record file and, worse, the oracle corpus — traces that essentially contained
the answer sessions, which explains an artificial 100% coverage. The corrected pipeline
downloads the current official LongMemEval-S artifact (277 MB, roughly 115k tokens per
question), verifies its published SHA, audits schema and population, and creates a new lock
before any retrieval change. On the resulting 40-question development preflight, literal
recall has a hard 60% ceiling (only 24/40 answers appear verbatim anywhere in the
haystack); the graph-retrieval arm reaches 57.5% literal reachability — 23 of those 24 —
at 7,302 tokens, with 99% mean evidence-source coverage.

### The consolidation layer

Consolidation is defined as **online consolidation by use**, not periodic summarization:
just as the system has a decay path, it has a symmetric association/strengthening path
driven by subsequent prompts. Every later prompt strengthens the bounded links among the
memory items and evidence actually exposed together; unused links decay; repeated
assemblies become easy to reactivate from any member.

The persistent structure is a schema-v8, text-free, model-independent consolidation graph
over both typed memories and source chunks. A normal context build:

1. uses existing strong edges to fill bounded reserved slots;
2. packs under the unchanged token budgets;
3. only then reinforces the direct items that actually reached the model.

Edge hygiene rules:

- An edge requires repeated **independent** co-activation before it affects recall.
- Edges decay in turn-space and are degree-pruned.
- Graph-recalled items cannot reinforce their own links unless later retrieved
  independently — otherwise the system would manufacture familiarity from its own guesses.
- No prompt text, activations, residuals, or K/V state is ever persisted; only IDs and
  scalars.

### Qwen activations as the association teacher

A frozen seven-layer Qwen3-8B BF16 prefix acts as a transient consolidation observer — a
hippocampus-like role in the computational analogy. Per turn, it projects the bounded
memory assembly into CAV space: CAV coordinates describe the active conceptual assembly,
QK supplies directed association strength (which memories bind), and OV alignment
indicates what concept-bearing information was actually transported between them. Those
values gate the learning rate on the durable external graph; all activations are then
discarded. Rank-only operation is the fallback when no checkpoint is loaded, so the Qwen
pass is a weighted input seam, not a dependency.

Consolidation runs delayed — after response generation — with the model resident (a fresh
process per turn is too slow; measured cost is about 0.67 seconds per event against a
13-second one-time load).

The committed implementation additionally includes: schema-v9 causal prompt→response
bindings; complete streaming coverage of response/tool chunks; fixed nine-node events and
three-candidate Qwen workspaces; additive graph candidates that cannot blindly evict
direct evidence; two-hop scalar heat diffusion with hop-balanced selection; live-query
cosine reranking; score-per-√token packing; and a reproducible four-arm replay rig.

### Ablation result

On the locked 39-question operational evidence test, chronological replay:

| Arm | Recall | Mean tokens |
| --- | ---: | ---: |
| Original | 35/39 — 89.74% | 1,418.5 |
| Packing only | 36/39 — 92.31% | 1,349.0 |
| Rank consolidation | 37/39 — 94.87% | 1,431.9 |
| Qwen consolidation | 38/39 — 97.44% | 1,423.8 |

No regressions; Qwen uniquely recovered a two-hop question at slightly fewer tokens than
rank consolidation. Retrieval overhead versus the original arm is roughly 11 ms per
question. This crosses 95% for local literal **evidence retrieval**; it does not satisfy
the primary 95% **answer-stage** LongMemEval target, which requires at least 100 judged
questions with a responder receiving only the produced context. The tree was committed as
`bae4bca` with 743 tests passing.

## Why this shape

**Consolidation was the diagnosed gap, not more retrieval expansion.** An architecture
review against the state of the art (Mem0, SimpleMem, A-MEM, Zep, AgeMem, Eywa) concluded
that the project's provenance discipline — append-only transcript, span-level provenance,
verbatim evidence requirements, bounded prompts, attention treated as routing evidence
rather than truth — is competitive or leading, while semantic memory formation and
consolidation lag. Source discovery was nearly saturated on the preflight (99% coverage);
selecting and presenting answer-bearing material inside those sources was the bottleneck.
A follow-up correction sharpened this: the original architecture already had the
partition-and-link design; what was missing was the consolidation *implementation* — the
step that turns multiple episodic fragments into a compact current conclusion with
provenance links back to its episodes. The user directed that this layer be built before
any further assessments.

**Live per-turn strengthening mirrors decay.** The system already had a principled decay
path; defining consolidation as its symmetric opposite ("just like we have a decay system,
we should have an association/strengthening system") makes consolidation an ongoing
property of use rather than a scheduled batch job, requires no stored text or model state,
and reuses the retrenched Hebbian graph as its seed — that graph covered chunk co-access
only and was not wired into normal context assembly.

**Qwen as observer, not authority.** The QK/OV/CAV machinery from phases 01-02 is
contained to exactly the role the review said it had earned: an optional association
compiler whose weights gate graph updates, with a rank-only fallback and mandatory
discard of activations. The closest published relative is AGMR (retrieval-head attention
guiding memory refinement), so the honest claim is "mechanistic-attention-guided online
systems consolidation into a bounded external associative memory" — a defensible
direction, not established novelty.

**Delayed supervision at the turn boundary is causally safe.** The turn transition (the
question this phase opened on) provides learning signal only after the next turn occurs,
for later retrieval — no leakage into the turn being answered.

## Why not X

### Why not the Hebbian overlay (yet)

The first half of the phase built two learning mechanisms in quick succession: a
turn-transition policy (delayed bandit-style supervision over `stay`/`previous`/`next`/
`switch source` actions) and, at the user's direction, a live Hebbian co-retrieval graph —
rank-weighted activation of concepts retrieved in the same turn, hub normalization,
turn-based decay, bounded to 12 concepts/event and degree 32, recalled through reserved
slots. It was correctly bounded, persisted only IDs and scalars, and passed its tests.

It was still retrenched ([DR-0009](../decisions/0009-hebbian-retrench.md)). The mechanism
produced **zero measured evidence toward 95%** and had displaced benchmark-driven work; on
inspection, the benchmark it would have been judged against was itself invalid (stale
file, oracle corpus). The retrenchment rule that came out of it: establish the exact
current score, separate retrieval misses from answer-generation misses, and change only
the bottleneck accounting for the largest recoverable error set — no more architecture
additions without a measured delta.

The retrenchment was a sequencing correction, not a verdict on the idea. Within the same
phase the Hebbian graph returned as the seed of the consolidation layer, this time wired
into context assembly and validated by the four-arm ablation. A dedicated Hebbian
retrieval arm returns again, much later and under full measurement discipline, in
[chapter 09](09-acceleration-scoring-and-ladder-restoration.md) (DR-0039).

### Why not evidence coverage as the headline metric

"100% evidence retrieval" measures only that at least one annotated source session was
represented in context. It says nothing about whether the answer-bearing sentence was
present, or whether a responder could derive the answer.
[DR-0010](../decisions/0010-operational-e2e-test-primary.md) demotes coverage to a failure
diagnostic and makes the operational answer-plus-compression test primary.

### Why not periodic LLM-summary consolidation

The initial implementation target was "periodically write an LLM summary." It was replaced
by online consolidation by use ([DR-0011](../decisions/0011-live-per-turn-consolidation.md)):
summary rewriting requires storing and regenerating text, breaks the text-free/scalar
persistence discipline, and recent research indicates continuous textual consolidation can
corrupt useful memories. Mature assemblies may later earn a compact semantic
representative, but they do not need one to become associated. Authoritative evidence is
never rewritten.

### Why not declare 95% reached

The 97.44% figure is literal evidence recall on a 39-question local test. The target
([DR-0008](../decisions/0008-set-95-percent-target.md)) is answer accuracy on at least 100
judged questions with a responder and judge. No remote responder/judge calls had been made
by end of phase, so the project deliberately holds no LongMemEval answer-accuracy claim.

## Open questions

- **Answer-stage accuracy is unmeasured.** The central-dev responder/judge endpoint
  configuration was not exposed to the workspace; the ≥100-question judged run is the next
  honest gate.
- **q13 remains disconnected.** Its answer is stored but unreachable from its final query
  seeds; the fix is a bounded rolling episode bridge or compact episode node, not greater
  read depth.
- **Learning is not outcome-conditioned.** The graph strengthens what was retrieved
  together, not what produced correct answers — a self-reinforcement risk the architecture
  review flagged. Answer success/failure/contradiction signals are future work.
- **Novelty is unproven.** The publishable claim requires showing separate QK/OV signals
  beat embeddings, rank co-access, and plain attention weights at equal tokens, latency,
  and storage; AGMR narrows the claim.
- **Scale is unproven.** Million-token behavior (BEAM, LongMemEval-V2) is untested; the
  cold cluster/era-summary partition from the original plan remains unbuilt.
- **Temporal/entity semantics lag SOTA.** No canonical model of entities, events, fact
  validity intervals, or current-versus-historical state exists yet.

## Source turns

Raw transcript for this phase:
[phase-03 overview](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/00-overview.md)

Key moments:

- Target set (DR-0008):
  [turn-316-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-316-user.md)
- Turn-transition learning proposed and built:
  [turn-336-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-336-user.md),
  [turn-337-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-337-assistant.md),
  [turn-401-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-401-assistant.md)
- Hebbian co-retrieval graph requested and implemented:
  [turn-438-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-438-user.md),
  [turn-444-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-444-assistant.md),
  [turn-446-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-446-assistant.md)
- Retrenchment (DR-0009):
  [turn-447-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-447-user.md),
  [turn-448-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-448-assistant.md),
  [turn-449-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-449-assistant.md),
  [turn-464-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-464-assistant.md)
- Corrected-corpus results and metric explanation:
  [turn-482-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-482-assistant.md),
  [turn-484-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-484-assistant.md)
- Operational test made primary (DR-0010):
  [turn-485-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-485-user.md),
  [turn-486-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-486-assistant.md),
  [turn-488-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-488-assistant.md)
- Architecture review and consolidation diagnosis:
  [turn-501-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-501-user.md),
  [turn-505-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-505-assistant.md),
  [turn-508-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-508-assistant.md)
- Consolidation defined as live per-turn strengthening (DR-0011):
  [turn-509-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-509-user.md),
  [turn-511-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-511-user.md),
  [turn-512-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-512-assistant.md),
  [turn-513-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-513-assistant.md)
- Qwen as association teacher and human-memory analogy:
  [turn-514-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-514-user.md),
  [turn-515-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-515-assistant.md),
  [turn-517-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-517-assistant.md),
  [turn-526-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-526-assistant.md)
- Real Qwen path and four-arm ablation:
  [turn-541-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-541-assistant.md),
  [turn-586-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-586-assistant.md)
- Commit:
  [turn-587-user.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-587-user.md),
  [turn-590-assistant.md](../../../_ingest/codex-2026-08/raw/phase-03-95-percent-associative-memory-campaign/turn-590-assistant.md)
