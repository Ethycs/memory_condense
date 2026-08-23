# 06 — Set Completion: Diagnosis, Mechanism, and Selector Build

**Phase:** 06 (merged turns 233-322, 2026-08-18)
**Previous:** [05 — Packet compression and operational replacement](05-packet-compression-and-operational-replacement.md)
**Next:** [07 — Diffuse retrieval buildout](07-diffuse-retrieval-buildout.md)

## Purpose

This chapter documents the diagnosis of the system's number-one failure —
complete-set reachability — and the design that fixed it: a query-conditioned
marginal set selector running on a transient Qwen3-8B prefix, speaking a
compact INI protocol, under staged GPU residency. It is the largest phase in
the workstream and its hinge decision (DR-0025) reverses the operator ambition
of [chapter 01](01-cav-attention-head-ideation.md): QK/OV attention is demoted
from retrieval mechanism to scorer, and a deterministic coverage loop makes
the keep/reject decisions.

By the end of the phase the locked 1M-token development replay reaches 100%
packed evidence coverage (10/10 complete sets, 11/11 checkable answer values)
with an average returned context of 1,986 tokens from a ~1.04M-token history —
a 99.81% reduction — and ten blind held-out shards are prepared but
deliberately unmeasured.

## The set-completion problem

Entering this phase, retrieval routed queries to the correct conversation
reliably, but the packed context missed members of enumerable sets: concerts
at 4/5 required sessions, museums at 4/6, overall evidence-source coverage at
94.7%, and only 8/10 questions with every required source present.

The diagnosis: retrieval optimizes "highest-scoring individual chunks" while
enumeration questions require "recover every distinct event the question
needs." Repeated high-scoring material crowds out one or two legitimate but
weaker-scoring events. This is an objective mismatch between top-k relevance
and exhaustive set retrieval — not a search-scale problem, and not something
more semantic ranking alone can fix, since a better ranker still produces
another top-k list.

Two structural facts sharpen the problem:

- A chunk that says "I visited the Science Museum" occupies the same
  embedding/CAV neighborhood as "the assistant recommended a museum" or "we
  discussed opening hours." Similarity does not establish membership in
  "museums the user actually visited." Membership requires event-level
  binding: actor, predicate, object, status, time, provenance.
- Deduplication granularity is query-relative. "Which museums?" deduplicates
  by museum identity; "which museum visits?" by event/date; "earliest
  concert?" collapses descriptions of the same concert but preserves distinct
  dates. No permanent classification can anticipate all of these — the
  comparison must be made against the current query at retrieval time.

A raw-graph-versus-packed-prompt diagnostic later proved decisive: raw
candidate coverage was already 100% on the locked run — every required source
was retrieved — and the old information-gain packetizer was what dropped
coverage to 94.7%. The loss lived in selection and packing, not discovery.

## Design

### Reachability as the objective (DR-0023)

Answer generation is frozen. The pipeline is judged on three invariants:

1. the correct history enters the local beam;
2. every required evidence session survives candidate selection;
3. the final packet preserves each selected event's minimal supporting
   sentence with exact provenance.

The first implementation move under this objective is a protected union:
narrow scalar-retrieval winners form a protected prefix (42 sources on the
locked run), and bounded attention may spend a small number of slots (six)
only on previously unseen sources from the broader frontier. No
attention-selected item may evict a scalar evidence source.

### The CAV event/concept-link reachability layer (DR-0024)

The minimal reachability-focused version of the event-semantics idea:
event-sized spans, typed event/concept links with provenance, and a
query-time set collector. CAV/QK/OV populate or validate links when
available; deterministic extraction remains the control arm so the neural
compiler must demonstrate added recall, not just added complexity.

As built: an event CAV probe at 93.8% held-out balanced accuracy; 2,478 user
chunks indexed from 6,450 transient conceptual spans; exactly one float32 CAV
coordinate persisted per chunk — no activations or K/V state; the live Qwen
workspace capped at 8 candidates / 1,024 tokens. The layer recovered
candidates below ordinary search cutoffs but did not move headline coverage
(still 94.7%), which is what motivated the raw-vs-packed diagnostic and
shifted attention downstream to selection.

Structured event records are a query-time working representation and optional
cached view — not the memory architecture. Raw chunks remain the source of
truth behind multiple lossy indexes (lexical, BGE, CAV, entity, temporal,
access graph), unioned rather than intersected, with typed attributes treated
as hypotheses and never as hard retrieval gates.

### Query-conditioned marginal set selection (DR-0025)

The core mechanism. For each candidate `c` given the already-selected set
`S`, estimate its marginal gain:

    gain(c | S) = support(q, c)
                  * [1 - max over s in S of P(same answer/event | q, c, s)]
                  - lambda * tokens(c)

An online greedy loop interprets each candidate against the query, rejects
non-supporting candidates, keeps only the better/cheaper evidence when the
selected set already contains the same answer or event, and otherwise
accepts. The crucial property is that novelty is judged on **answer
identity**, not text distance: two museum memories sound alike but may
contribute different required answers. This is greedy set cover plus MMR,
with the redundancy term conditioned on the question.

Formally this needs `g_i = f(q, c_i, S)` — a coverage-aware conditional
cross-encoder — rather than a standard cross-encoder's `r_i = f(q, c_i)`.
QK/OV still participate, but only as the scorer: QK estimates
`supports_query` and `same answer/event`, OV produces the candidate's
partial-answer representation, and the deterministic selector makes every
keep/reject decision.

### The small-model selector and the INI protocol (DR-0026)

The selector runs as a bounded, transient listwise judge (the useful part of
the earlier `QwenLiveHeadMemory` pattern): query plus the whole bounded
candidate set in one workspace, per-candidate decisions of
existing / new / null relative to query-conditioned event groups, coverage-
first packing keeping one representative per event, and fail-open handling
of malformed or uncertain classifications so a parser error can never lose
recall.

The exchange format is INI, not JSON: a `[request]` / `[candidates]` /
`[items]` layout with one pipe-delimited row per candidate —

    id=event|answer|time|existing|new|null|answerability

— and `[end]` as the generation stop. JSON is retained only as a
backward-compatible parse fallback. Small models emit the compact rows with
far fewer protocol failures and fewer tokens than nested JSON.

Model sizing was settled empirically: the full 8B model as an online
generator exceeded five minutes per question and was stopped; Qwen3-0.6B and
SmolLM2-360M run the identical INI contract as classifier ablations, with an
FP16-on-pre-Ampere loader fix (Turing GPUs emulate BF16) recovering GPU
utilization.

### The six-layer Qwen3-8B prefix (DR-0027)

The primary architecture is not a small generator at all: it is the full
Qwen3-8B **representation**, truncated to its first transformer blocks and
used transiently for QK/OV readout — no LM head, no token generation, no
later layers, no KV cache, no activation database. The prefix loader reads
only shard 1 (embeddings plus layers 0-5, ~3.5 GiB). The linker exposes one
transient normalized OV transport vector per candidate; the coverage
controller clusters those query-conditioned vectors, keeps one
representative per event, then discards every vector and returns ordinary
chunk IDs.

The layer ablation bounded the depth requirement:

- Hard minimum is two blocks: a layer-0 readout query is still just the
  token embedding; only from block 1 onward is the readout genuinely
  query-conditioned.
- The two-block/layer-1 arm matched the six-block/layer-5 arm exactly on the
  known failing questions (80% on q3, 66.7% on q8) and on the full locked
  run (94.7% mean coverage, 8/10 complete) at roughly half the latency
  (0.48 s mean selector time).
- The ablation also exposed why depth was irrelevant to the residual
  failures: the missing conversational chunks never reached Qwen — only
  their timestamp rows did. No layer can classify evidence it never
  receives. That upstream hydration defect (metadata masquerading as
  evidence, then hydrated chunks losing during ordering/prefiltering) was
  the actual remaining bug, fixed by the planned MS MARCO cross-encoder
  relevance stage plus proper source hydration.

Six layers remain the reference configuration; two blocks are the practical
operating point.

### Staged GPU residency (DR-0028)

On the 8 GiB RTX 2070, static co-residency is unsafe: ~1.75 GiB baseline
+ ~3.5 GiB Qwen prefix + ~2.3 GiB BGE-M3 leaves ~0.6 GiB before activations,
CUDA buffers, and allocator fragmentation. The prior defensive rule — BGE
permanently on CPU — caused a 10-minute-per-question slowdown. The replacement
is staging:

    BGE on GPU -> freeze query vectors -> unload BGE -> Qwen prefix on GPU

The two models never need simultaneous residency. Staged correctly, the
prefix costs a 10.47 s one-time load and 0.39 s for all eight candidates in
a single forward pass over 240 active token positions.

### The end-of-phase pipeline

1. Route to a bounded union of candidate memories (all retrieval routes,
   unioned; protected scalar prefix).
2. Hydrate every activated source with real evidence, never timestamp
   metadata alone.
3. MS MARCO cross-encoder for query-memory semantic relevance.
4. Transient Qwen prefix QK/OV grouping for duplicate/coverage control via
   the marginal-gain selector.
5. Pack under the hard token cap, one representative per distinct event,
   exact original provenance retained.
6. Send only that packet (~2K tokens) to the answering model; persist no
   transformer token state.

Validated at 100% packed coverage and 10/10 judged development answers on
the locked replay; 100 held-out questions prepared as ten blind read-only
1M-token shards (10.44M tokens, 79,915 chunks total) with zero responder or
judge calls made.

## Why this shape

- **Selection was the bottleneck, provably.** The raw-vs-packed diagnostic
  showed discovery at 100% and packing at 94.7%. Building more retrieval
  machinery would have optimized the solved half of the problem.
- **Set membership is query-relative.** Because deduplication granularity
  and operator semantics (`all`, `earliest`, `count`) depend on the
  question, classification must happen at retrieval time against the query
  — late binding — with raw chunks authoritative and all ingestion-time
  structure treated as hints.
- **Deterministic control, neural scoring.** The greedy marginal-gain loop
  is inspectable and cannot hallucinate evidence; the neural components
  only score support, identity, and coverage. Fail-open parsing means
  selector degradation costs precision, never recall.
- **Transient compute, compact state.** The prefix produces per-candidate
  vectors that are used and discarded; only source IDs, scalar strengths,
  one CAV coordinate per chunk, and provenance persist. This keeps the
  memory store small and the architecture checkpoint-agnostic.
- **Hardware honesty.** Every arm was sized against the actual 8 GiB card:
  the 8B generator was rejected on measured latency, BF16 on measured
  Turing emulation, co-residency on measured GiB arithmetic.

## Why not X

### Why not QK/OV operator construction ([DR-0025](../decisions/0025-marginal-set-selection-over-qkov.md))

This is the phase's hinge, and it reverses the ambition of
[chapter 01](01-cav-attention-head-ideation.md). The fully-specified
alternative existed: a native query-conditioned operator where the old
memory activation M attends to the new prompt activation N (yielding a
recontextualized M'), and the prompt then reads the recontextualized memory
(yielding an enriched search state N'), implemented as a causal sandwich
`[current prompt][memory span][readout probe]` over the retained prefix,
with activation-composition variants ("lighthouse" + "lighthouse keepers")
explored alongside it.

It was set aside because a simpler formulation subsumed the need: if each
candidate can be checked for how it **partially satisfies the query**, then
others like it can be rejected — query-conditioned marginal set selection.
The operator machinery answers "how do these representations relate," but
the failing questions needed "does this candidate add a new required
answer," which is a set-cover decision, not a recontextualization. The
missing chunks already existed in the candidate pool; they were being
crowded out, and a selector fixes crowding directly. QK/OV survives only as
the scoring substrate inside the selector. The recontextualization operator
remains a documented future ablation, not a deleted idea.

### Why not fixing answer generation first ([DR-0023](../decisions/0023-freeze-generation-reachability-objective.md))

Answer quality cannot exceed evidence completeness, and evidence
completeness was exactly measurable (4/5, 4/6). Freezing generation made
the failing variable observable and prevented prompt-side tuning from
masking retrieval loss.

### Why not a permanent event schema instead of the link layer ([DR-0024](../decisions/0024-cav-reachability-layer.md))

Extraction will always miss implicit, ambiguous, compound, or
previously-irrelevant meanings, so a structured entry can never be the
condition for recall. The reachability layer keeps typed event/concept
links as confidence-weighted hypotheses over authoritative raw chunks, with
a deterministic-extraction control arm to prove the neural compiler earns
its complexity.

### Why not a BGE duplicate pass, a plain MS MARCO cross-encoder, MMR, or bigger top-k

All were examined during mechanism exploration. A BGE "is this a duplicate"
pass and a standard cross-encoder both score `f(q, c)` — individual
relevance or pairwise similarity — and cannot see the selected set, so they
reproduce the top-k objective mismatch. MMR diversifies by topic distance
and can discard repeated museums that are distinct answers. Heat-diffusion
and PageRank find associated memories but guarantee nothing about set
completeness. Raising top-k was already shown (250-activation regression)
to increase crowding. The MS MARCO cross-encoder was kept — but as the
relevance stage in front of the selector, not as the selector.

### Why not JSON for the selector protocol ([DR-0026](../decisions/0026-ini-selector-protocol.md))

Nested JSON is verbose and brittle for sub-1B models; the pipe-delimited
INI rows cut tokens and strict-parse failures while remaining trivially
machine-checkable. JSON parsing is retained as a fallback only.

### Why not a small generator model as the primary path ([DR-0027](../decisions/0027-restore-six-layer-qwen-prefix.md))

Running Qwen3-0.6B or SmolLM as full generators was an accidental drift
from the specified architecture, caught and corrected mid-phase. The
intended design uses the 8B model's representation quality without its
generation cost: six prefix layers, transient QK/OV readout, no LM head.
The generator arms survive only as protocol ablations. The subsequent layer
ablation (two blocks matching six) validated the prefix approach at even
lower cost.

### Why not permanent BGE-on-CPU or full co-residency ([DR-0028](../decisions/0028-staged-gpu-residency.md))

CPU BGE cost ~10 minutes per question; full GPU co-residency leaves ~0.6
GiB of margin on the 8 GiB card, which is a real fragmentation/OOM risk.
Staged residency avoids both, because the retrieval and selection stages
never need the two models simultaneously.

## Open questions

- **Held-out accuracy.** The 100% figures are development-replay numbers on
  ten locked questions. The ten blind 1M-token shards (100 questions) were
  prepared with zero responder/judge calls; the decisive 200-call
  measurement had not been run at phase end (it is picked up in
  [chapter 08](08-1m-test-execution-and-regression.md)).
- **Selector generalization.** The marginal-gain loop was tuned against
  enumeration/ordering failures; behavior on `changed-over-time` and
  contradiction-preserving queries (where duplicates must be kept and
  ordered) is designed but not stress-tested.
- **Two versus six layers at scale.** The two-block arm matched six blocks
  on the development set after the hydration fix; whether deeper readouts
  matter on harder paraphrase binding remains an open ablation, as does the
  intermediate 3-5 block range that was skipped when the endpoints tied.
- **The recontextualization operator.** M'/N' activation recontextualization
  and the learned coverage-aware set selector (separate relevance/duplicate/
  coverage projections, trained STOP head) remain specified but unbuilt
  fallbacks if greedy selection hits a ceiling.
- **CAV compilation operationalization.** Incremental per-turn compilation
  and a versioned artifact cache were listed as remaining work; the
  two-minute retrospective backfill per experimental clone is still the
  cost model.

## Source turns

Raw transcript for this phase:
[phase-06-set-completion-selector](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/00-overview.md)

Key moments:

- Diagnosis — zoom-out and #1-problem statement:
  [turn-954-user.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-954-user.md),
  [turn-976-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-976-assistant.md),
  [turn-978-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-978-assistant.md),
  [turn-983-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-983-assistant.md)
- DR-0023 reachability freeze and protected union:
  [turn-972-user.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-972-user.md),
  [turn-973-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-973-assistant.md)
- DR-0024 concept-link layer, build and results:
  [turn-989-user.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-989-user.md),
  [turn-990-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-990-assistant.md),
  [turn-1052-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1052-assistant.md)
- Mechanism exploration — RAG framing, late binding, QK/OV operator:
  [turn-1069-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1069-assistant.md),
  [turn-1071-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1071-assistant.md),
  [turn-1073-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1073-assistant.md),
  [turn-1090-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1090-assistant.md)
- DR-0025 pivot to marginal set selection:
  [turn-1091-user.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1091-user.md),
  [turn-1092-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1092-assistant.md),
  [turn-1097-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1097-assistant.md)
- Raw-vs-packed diagnostic result:
  [turn-1115-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1115-assistant.md)
- DR-0026 INI protocol:
  [turn-1126-user.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1126-user.md),
  [turn-1127-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1127-assistant.md),
  [turn-1128-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1128-assistant.md)
- DR-0027 six-layer prefix restoration and layer ablation:
  [turn-1132-user.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1132-user.md),
  [turn-1133-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1133-assistant.md),
  [turn-1151-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1151-assistant.md),
  [turn-1176-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1176-assistant.md)
- DR-0028 staged GPU residency:
  [turn-1144-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1144-assistant.md),
  [turn-1147-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1147-assistant.md)
- Accuracy wrap-up and held-out preparation:
  [turn-1181-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1181-assistant.md),
  [turn-1452-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1452-assistant.md),
  [turn-1483-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1483-assistant.md)
