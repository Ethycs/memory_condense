# Query-conditioned coverage loop

**Status:** frozen v3 development treatment implemented and measured; held-out
validation remains pending
**Date:** 2026-08-18
**Applies to:** exhaustive, count, fixed-cardinality, and temporal set queries

## Purpose

Similarity retrieval answers a pointwise question: *which chunk resembles the
query?* Complete-set memory needs a different decision: *which distinct
requested items are represented, which candidates repeat the same information,
and which unresolved candidates must remain visible?*

```text
query
  -> query-shape hint
  -> coarse partition and broad candidate reachability
  -> bounded query-conditioned interpretation
  -> conservative coverage groups
  -> one representative per group
  -> every unresolved candidate
  -> remaining support candidates
  -> bounded raw-evidence context
```

The controller prevents several strong chunks about one item from consuming the
packet before a different valid item gets one opportunity.

## Measured boundary

The original locked 1,039,203-token development stress exposed the boundary:

- 100% raw graph evidence-source coverage;
- 100% raw complete evidence sets;
- 94.7% mean packed evidence-source coverage; and
- 8/10 complete packed evidence sets.

The final v3 treatment reaches 100% raw and packed evidence-source coverage,
10/10 complete packed evidence sets, and 11/11 scored answer-value components
at a mean 1,985.6 returned tokens. This is development selection evidence; it
does not replace the frozen 100-question held-out gate. The selector still
cannot manufacture evidence missing from its routed or explicitly scanned
frontier.

## Query-shape hint

The implementation recognizes explicit `all`/`each`/`every`, count, fixed
cardinality, ordering, earliest, and latest forms. It also suppresses set
coverage for narrow derived-scalar/current-value forms and carries conservative
role and relative-time constraints. Typed identity hints distinguish museum
venues and completed performance occurrences from topic-level mentions.

This is an inspectable regex-derived control hint, not a semantic query
compiler and not an answer executor. It orders raw evidence for the responder;
it does not itself count or answer earliest/latest questions.

## Selected composite and secondary backend

| Backend | Group signal | Rejection/NULL | Calibration | Generation |
|---|---|---|---|---|
| `qwen_prefix_choice` (selected) | layer-1 OV transport direction, QK/OV utility, retrieval features, typed transient identities, and a bounded Qwen3-0.6B forced-choice component | temporal contradictions only; uncertain rows remain fail-open | none; energy/posterior controls are explicitly uncalibrated | none |
| `local_ini` (secondary ablation) | generated event key plus normalized EXISTING/NEW/NULL assignment scores | high-confidence threshold | none | compact INI |

The selected treatment deliberately combines a prefix measurement with a
generation-free causal-choice score. “Bayesian” still describes the shape of
the controller, not calibrated probabilities: QK/OV affinities and A/B token
likelihoods are features, not proof of same-event identity or irrelevance.

## Full-width Qwen3-8B prefix (primary)

The frozen v3 selector loads the Qwen3-8B token embedding and complete decoder
blocks 0 and 1 directly from the checkpoint, reading QK/OV features at layer 1.
The implementation remains configurable for deeper prefix ablations. It does
not instantiate later blocks or an LM head and cannot generate tokens.
Complete retained blocks still include their pretrained attention and MLP
sublayers: deleting the MLPs would put later retained attention off its trained
residual stream.

Each bounded candidate is interpreted in an independent causal row:

```text
[Memory] <candidate> [Question] <current query> [Readout]
```

All rows execute layers 0–1. Only the layer-1 readout-to-memory QK block and its
candidate-specific OV update are materialized. The constant `[Memory]` marker
is excluded from the transported span. The active padded batch is charged to a
hard workspace cap.

The selector moves one normalized 4,096-dimensional OV update direction per
inspected candidate to CPU, forms conservative complete-link transport-affinity
groups, orders one original raw chunk per group, and then discards every
vector. Independent rows prevent candidates from attending one another and
make extracted per-ID signals invariant to frontier order.

This is a heuristic ablation. Similar OV update directions do not yet establish
semantic event identity, and QK/OV utility does not establish irrelevance.
Accordingly this version performs no NULL pruning: uninspected, malformed, or
unusable rows remain unresolved and receive a first-pass position before
support candidates. CAV projection and a trained pair/set head remain separate
future ablations.

## Generation-free forced-choice scorer

The selected secondary component loads the pinned full Qwen3-0.6B checkpoint
only after BGE has been released. For each bounded candidate it compares two
single-token continuations at the final prompt position. It does not call
`generate`, sets `use_cache=False`, microbatches under an explicit workspace
cap, and transfers only scalar probabilities to the controller.

The prompt includes candidate role and source timestamp. The score is useful
as one query-conditioned feature, but local probes found it weakly calibrated;
it is therefore not an absolute relevance gate. Typed structure, role,
surface-value evidence, retrieval order, and fail-open uncertainty remain
authoritative safeguards.

## Compact local INI classifier (secondary ablation)

The secondary arm loads a complete small causal model, with SmolLM2-360M-
Instruct as the intended first comparison. One bounded listwise call emits:

```ini
[items]
0=event_key|answer_value|timestamp|p_existing|p_new|p_null|answerability
[end]
```

Only validated known candidate IDs map back to original raw chunks. Generated
evidence text is never trusted. A missing, malformed, out-of-workspace, or
high-entropy row remains uncertain; a whole-call error or invalid INI fails
open. Legacy JSON parsing exists only for old fixtures/artifacts.

These scores are posterior-shaped controls, not calibrated Bayesian
likelihoods. SmolLM/INI remains a secondary classifier ablation, not part of
the frozen v3 treatment.

## Selected-partition structural scan

For typed complete-set queries, the condenser can enumerate every real content
chunk in the selected partitions before final packing. It keeps the scan in
bounded raw-row form and streams model inspection in batches; it persists no
activations. Conservative venue and completed-performance parsers produce
transient identities, merge equal nonempty identities, keep distinct
identities even within one source, and abstain on ambiguity.

The scan proves completeness only inside its exact selected partition/source
snapshot. The report separately records routed-frontier, active-partition,
selected-scope, and global completeness. A fixed-K tail may close inside an
approximately selected scope only under the explicit frozen policy flag, and
that closure must report `closure_global_recall_guaranteed=false`.

## Coverage-first packing

The selected backend reorders candidates as:

```text
one representative from every credible group
  + every unresolved candidate
  + remaining support candidates
```

`ContextPacker` remains authoritative for exact item and token ceilings and
returns only original raw excerpts. Typed/fixed representatives receive a
fair preallocation with a minimum useful body floor when feasible; infeasible
sets degrade deterministically and remain visible in the trace. Ordinary
queries keep the original information-gain path. The responder remains
responsible for the actual answer.

The selector runs after source metadata/companion hydration and over the final
route union. Complete-set queries may add a fixed-count structural scan before
selection; all additions retain exact original `RetrievalResult` provenance.
Unknown, replaced, or model-invented result objects are rejected and omitted
inputs are appended as a fail-open tail.

## Storage and memory boundary

```text
SQLite/HNSW raw chunks
  -> bounded candidates
  -> transient two-layer QK/OV read plus forced-choice scalars
  -> selected durable chunk IDs
  -> discard hidden states, attention, OV vectors, prompt, output, and K/V
```

No affinity group, generated event key, activation, OV vector, or K/V cache is
persisted. Durable memory remains authoritative raw chunks plus compact graph,
source, lifecycle, and optional CAV scalars.

## Implementation map

- `coverage_selector.py`: compositional query hints, typed identities,
  prefix-affinity grouping, uncertainty, fail-open ordering, and text-free
  diagnostics.
- `head_memory.py`: independent-row bounded QK/OV readout from the configured
  Qwen3-8B prefix.
- `causal_choice_scorer.py`: generation-free single-token A/B likelihoods,
  hard workspaces, and role-aware source-companion choice.
- `performance_events.py`: conservative transient completed-performance
  identity shared by active scanning and selection.
- `ContextPacker._build_expansions`: invokes the optional selector after source
  timestamp binding and before exact budget enforcement.
- `MemoryCondenser.set_context_candidate_selector`: attaches one shared
  transient selector without changing durable storage.
- eval CLI: `--coverage-selector-qwen-prefix-model-dir` plus
  `--coverage-selector-choice-model-dir` selects the frozen composite;
  `--coverage-selector-local-model-dir` remains the INI ablation.
- recall CSV: raw and packed coverage plus selector inspections, groups,
  unresolved rows, workspace tokens, latency, and fallback reason.

## Evidence gate

Compare on the identical locked frontier:

1. current scalar/graph packet;
2. bounded Qwen-prefix coverage ordering;
3. Qwen-prefix plus generation-free forced-choice scoring;
4. SmolLM compact-INI classification; and
5. the older grouped Qwen tournament only as a separate control.

Report raw source coverage, packed source coverage, complete evidence sets,
literal reachability, returned tokens, p50/p95 selector latency, peak memory,
and fallback rate. V3 recovered both missing packed evidence sets without
inflating the 2,250-token packet. The next gate is the frozen 100-question
held-out campaign, not further development tuning.

## Safety invariants

1. Raw chunks remain the only final evidence.
2. Unknown model-produced IDs never enter the packet.
3. Uninspected or malformed candidates are not silently rejected.
4. Invalid model output or feature shape fails open.
5. No request-derived transformer token state crosses a selection call;
   reusable static weights and tokenizer assets are explicitly excluded from
   this metric.
6. Exact token and item limits remain deterministic.
7. Gold answers, gold sources, and benchmark categories never enter selection.
