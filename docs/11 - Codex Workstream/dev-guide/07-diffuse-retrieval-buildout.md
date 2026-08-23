# 07 — Diffuse Retrieval: Closure-Aware RAG Design and Buildout

**Phase:** 07 (merged turns 323-392, 2026-08-18 to 2026-08-21)
**Previous:** [06 — Set completion selector](06-set-completion-selector.md)
**Next:** [08 — 1M test execution and regression](08-1m-test-execution-and-regression.md)

## Purpose

This chapter documents the diffuse-information retrieval architecture: how the
system answers questions like "how should we improve this design?" over a long
engineering conversation, where the evidence is not a single fact or an
enumerable set but a connected web of goals, constraints, experiments,
failures, revisions, and open questions. The locked design is an
evidence-closure RAG pipeline (DR-0029) built from the codebase's own
surprise/attention-head machinery (DR-0032) rather than EM-LLM (DR-0031),
implemented on a repository reorganized into objects, transformations, and
workflows (DR-0030), and hardened by a deliberately bounded refactor of the
replay/eval plumbing only (DR-0033).

## The diffuse-information problem

The set-completion work of [chapter 06](06-set-completion-selector.md) fixed
enumerable-set questions: "which museums did I visit?" has a checkable,
finite answer. The diffuse frontier is different in kind. An improvement
question over an engineering log requires the original problem and its
constraints, the experiments and their outcomes, rejected alternatives and
why, later corrections that supersede earlier decisions, and unresolved
dependencies — evidence scattered across hundreds of turns with no single
high-similarity chunk that answers anything.

Top-k similarity retrieval fails here structurally: it can surface a
conclusion while dropping its premise or counterevidence, and no per-chunk
relevance score expresses "this decision was later revised." The success
metric therefore changes: not whether individually relevant passages
appeared, but whether a complete minimal evidence set was retrieved — or the
system explicitly reported what is missing.

## Design

### Evidence-closure RAG (DR-0029)

The retrieval pipeline for diffuse questions is structured, iterative RAG
with an explicit closure criterion:

1. Hybrid dense + lexical search retrieves broad raw-text seeds.
2. A domain-neutral discourse graph links claims, constraints, decisions,
   evidence, revisions, contradictions, dependencies, and open questions.
3. A query compiler turns the question into evidence obligations — for an
   improvement question: goals, current design, observed failures,
   constraints, attempted fixes, tradeoffs, unresolved risks.
4. Retrieval iterates until it assembles a connected sufficient evidence
   set covering those obligations, or explicitly reports what is missing.
5. The packer selects whole evidence bundles — never isolated chunks —
   under the hard token cap, so a conclusion cannot enter the prompt
   without its premise or its counterevidence.
6. The answer LLM receives a compact relation index plus verbatim, cited
   excerpts.

The graph is an index and planning device only. The raw conversation remains
factual authority, which keeps the design general across engineering,
research, planning, and incident-response conversations instead of encoding
engineering-specific rules or GraphRAG-style summaries.

### Objects, transformations, workflows (DR-0030)

The codebase is organized in three layers plus thin facades:

```text
immutable objects/contracts -> stateless transformations -> stateful workflows/adapters -> thin facades
```

- **Objects** hold durable state, identities, and resource ownership: keys,
  vectors and index handles, typed relations, source-span coordinates,
  hashes, revisions. Service objects are reserved for actual stateful
  boundaries — SQLite, indexes, model runtimes, orchestration.
- **Transformations** are pure: parsing, segmentation, retrieval, ranking,
  attention routing, closure, hydration, packing, metrics.
- **Workflows/facades** hold orchestration and stay small.

The reorganization decomposed the monoliths — coverage selector 3,464 to a
64-line facade, eval CLI 3,895 to 245, condenser 4,420 to 680, retrieval
1,500 to 59 — and added a 1,300-line source ceiling with facade-size
regression checks. It is retrieval-neutral by construction: the frozen v3
evaluation runs from an isolated exact source snapshot while the reorganized
tree carries a new v4 implementation identity.

The "Attention As Graphs" theory note sharpened one boundary: query-time
attention output is a transient `AttentionWitness` (query-bound virtual
edges, scores, model identity, caps, zero-state receipt), never a durable
object. Attention scores are hypotheses, not factual graph edges.

### Episodes from existing machinery (DR-0031, DR-0032)

The episodic front-end is EM-inspired, not EM-LLM. EM-LLM contributes
exactly two useful techniques — surprise-based episode boundaries and
temporal-neighbor recall — and both are implemented as interchangeable
boundary strategies (fixed/source boundaries, embedding-change boundaries,
or an injected surprise scorer) so the closure system works without any of
them and matched ablations decide which earns a production role. EM-LLM's
persistent K/V memory is rejected outright: it violates the zero
persisted-transformer-token-state requirement.

The surprise signal itself is not new. The Qwen prefix/head machinery from
[chapter 06](06-set-completion-selector.md) already produces
existing/new/null posteriors, and
`semantic_surprisal = -log(1 - p_new)` is the boundary signal. The same
QK/OV attention heads provide bounded semantic-change detection and cohesion
refinement. The buildout was plumbing that signal into episode formation and
recording its identity — not inventing another algorithm.

As built, the episodic path is:

- attention-head/OV transport signals for episode boundaries
  (`search/episodes/qwen_episode_signal.py`);
- persisted episodes with representatives, and direct query-to-episode
  representative retrieval (`search/episodes/representative_retrieval.py`);
- bounded QK/OV routing (`associations/qwen_memory_linker.py`);
- deterministic evidence closure (`search/closure/engine.py`);
- exact final packets with provenance (`domain/discourse.py`).

By end of phase, episode-primary retrieval is genuine: selected episodes
drive closure with no artifact-global bypass
(`eval/diffuse_longmemeval.py`).

### Post-retrieval latent fusion

A Perceiver-style latent resampler sits after episodic retrieval, never in
place of it. Placement is fixed:

```text
episodic retrieval -> bounded exact evidence and relation bundles
  -> K query-conditioned latent slots (extract -> relate -> decode)
  -> ranked evidence-backed conclusions -> exact-span evidence packet -> answer
```

The router does true K×N extraction followed by N×K reinjection — never N×N
attention over the corpus (`search/fusion/latent_router.py`). Slots are
query- and obligation-conditioned (fact, temporal order, contradiction,
revision, causal support) rather than anonymous learned slots. Output is
bundle IDs, relation hypotheses, scores, and an auditable attention witness;
the latent vectors and KV state disappear after the query. The planner emits
bounded grouping/order plans over exact evidence — no abstractive prose
(`search/fusion/planner.py`). Node features are produced by a causal
readout placed after `[Evidence] ... [Question] ...`, with every bounded
atom processed (no silent workspace-prefix truncation), returned as a GPU
tensor plus a hash-only provenance receipt.

### GPU replay execution

Evaluation runs as content-addressed stages with a hard firebreak:

```text
sanitized sample -> base BGE store + frozen anchors
  -> per-arm derived episode stores -> frozen packets
  -> gold-only measurement
```

- The corpus is embedded once; all arms derive from frozen identical
  inputs, so the three-arm ablation (fixed intervals vs embedding-change vs
  Qwen-head episodes) shares the same anchors, prompt budget, and
  final-packet scoring.
- BGE and Qwen are both GPU-resident for latency; a real-model one-sample
  canary validates each path before the refactor that extracts it.
- Gold labels cannot enter until every packet is frozen.
- Tranches are contract-first: the implementation contract is frozen in
  `docs/02 - Implementation` before code, and each tranche proves one
  boundary (tranche A: the GPU row/lifecycle boundary) before matched
  routing or renderer behavior is added.
- The writable derived-store lifecycle is capability-held: atomic
  claim/finalization for SQLite/HNSW, race-safe abort/cleanup, single-
  snapshot finalized replay verification, legacy bytes preserved exactly.
- Production activation stays fail-closed: audits found rename-after-path-
  exposure windows in workspace handoffs and a checkpoint load/rehash ABA
  window, so the runner records the exact remaining blocker rather than
  activating.

### Bounded cleanup, not a rewrite (DR-0033)

Complexity cost is real but concentrated in replay and scientific-
attestation plumbing — the replay module at the 1,300-line ceiling,
identities spread across near-cap modules, and a callable hash that embedded
a source line number so a four-line import shift invalidated an unchanged
artifact. The locked response is surgical:

1. split replay reconstruction/verification from orchestration;
2. replace line-sensitive callable hashes with versioned semantic
   identities;
3. keep one explicit compatibility path for the frozen v1 artifact;
4. freeze the evaluation API afterward;
5. return immediately to the retrieval/consolidation algorithm.

The EM/episode/closure core and domain objects are explicitly out of scope
for rewriting.

## Measured state at end of phase

- Frozen v3 on the 100-question held-out audit: 87.6% mean labeled-source
  recall, 92% any-source, 82% all-source recovery, 48% literal-answer
  presence in final context. The earlier 10/10 was a selected development
  shard; the audit showed a generalization gap, not a code regression —
  every frozen byte re-verified.
- Held-out QA accuracy is 0/100 provider-scored; the 95% goal is
  undemonstrated.
- The full suite stands at 1,953 passed with independent audits finding no
  P0-P2; the canonical v1 campaign still uses legacy retrieval, and the
  latent router is untrained infrastructure — architecture without a
  measured performance improvement yet.

## Why this shape

- **Raw text stays authoritative because summaries cannot be audited.** The
  graph guides retrieval and the resampler proposes relations, but only
  exact source spans with provenance enter the final prompt. This is the
  same fail-closed stance as the packet layer in
  [chapter 05](05-packet-compression-and-operational-replacement.md), and it
  is what makes the closure criterion checkable.
- **The zero persisted transformer-token-state constraint prunes the design
  space.** It rejects EM-LLM's K/V memory, makes attention output a
  transient witness, and forces surprise to be computed transiently with
  only scalar boundary evidence persisted.
- **Iteration cost dictates where refactoring pays.** A 30-minute model run
  plus a ~9-minute verifier makes every plumbing mistake expensive, so
  identity and lifecycle seams get hardened — but a whole-codebase rewrite
  would erase the frozen baseline and add provenance drift for no
  algorithmic gain.

## Why not X

### Why not vanilla top-k RAG ([DR-0029](../decisions/0029-closure-aware-rag.md))

"Embed the question, fetch top-k chunks, ask an LLM" cannot express evidence
obligations, cannot follow revisions or contradictions, and can pack a
conclusion without its premise. A better ranker still yields another top-k
list; the failure is the absence of a closure criterion, not ranking
quality. Pure GraphRAG summaries fail the other way: they replace the
factual authority with generated text.

### Why not EM-LLM as a dependency ([DR-0031](../decisions/0031-reject-em-llm-dependency.md))

EM-LLM's genuinely useful ideas are two front-end techniques, not a
foundation, and its model-integrated K/V memory violates the zero-state
requirement. Adopting it wholesale would couple the closure system to a
segmentation strategy that matched ablations had not yet justified. The
system is instead built so EM-style segmentation is one injectable strategy
among three.

### Why not a new surprise model ([DR-0032](../decisions/0032-reuse-surprise-attention-machinery.md))

The Qwen prefix/head machinery already produces the needed posteriors and a
usable episode-boundary surprisal, and the QK/OV heads already do bounded
routing. Building a separate autoregressive token-NLL scorer would duplicate
capability, add a second model identity to attest, and delay the actual gap:
wiring the existing signal into episode formation.

### Why not vector/key-only domain objects ([DR-0030](../decisions/0030-objects-transformations-workflows-reorg.md))

Reducing objects to vectors and keys was considered for efficiency and
rejected: it discards exact provenance. Objects keep text out but retain
source-span pointers and hashes so selected evidence can be verified and
hydrated late. The efficiency win comes from the layer split itself — pure
transformations over immutable contracts — not from thinning the objects.

### Why not a heavy whole-codebase refactor ([DR-0033](../decisions/0033-targeted-refactor-only.md))

The recurring cleanup-vs-progress debate resolved the same way each time it
was raised: the slowdown lives in replay/attestation plumbing, the
EM/episode/closure core is reasonably modular, and a broad rewrite of domain
objects or the database would add risk and delay the algorithm. Cleanup is
bounded to the seams a real canary proved painful, then the evaluation API
freezes.

## Open questions

- **Held-out accuracy.** 0/100 validation questions have used the responder
  or judge; the 95% target is unproven, and the next meaningful result must
  come from an untouched population, not tuning against the exposed set.
- **Query-to-episode discovery.** Episode expansion mostly reorganizes
  evidence already found upstream; segmentation alone cannot lift 87.6%
  source recall to 95%. The v2 campaign receipt carrying `episode_primary`
  and the matched topology-only vs latent-fusion comparison are the next
  measurements.
- **Untrained latent router.** The fusion adapter needs training (analysis
  split only) and freezing before it can claim anything; extractive
  summaries are deferred until the comparison passes.
- **`answer_fact` representation.** The EM arm's required answer-fact
  representation still fails on the canary.
- **Activation authority.** Production activation remains NO-GO until the
  rename-after-exposure and checkpoint ABA windows are closed.
- **HNSW nondeterminism.** Rebuilding the same base produces different index
  bytes with identical observed anchors; frozen shared bases are the
  mitigation, not a fix.
- **Mem0 comparison.** The isolated Mem0 adapter (benchmark boundary only)
  still cannot produce a certified score; the comparison arm stays blocked.

## Source turns

Raw transcript for this phase:
[phase-07-diffuse-retrieval-buildout](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/00-overview.md)

Key moments:

- The diffuse-information frontier is opened:
  [turn-1490-user.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1490-user.md)
- DR-0029 closure-aware RAG over vanilla RAG:
  [turn-1497-user.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1497-user.md),
  [turn-1498-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1498-assistant.md),
  [turn-1503-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1503-assistant.md),
  [turn-1513-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1513-assistant.md)
- DR-0030 objects/transformations/workflows reorganization:
  [turn-1563-user.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1563-user.md),
  [turn-1564-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1564-assistant.md),
  [turn-1570-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1570-assistant.md),
  [turn-1574-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1574-assistant.md),
  [turn-1549-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1549-assistant.md)
- DR-0031 EM-LLM rejected as a dependency:
  [turn-1586-user.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1586-user.md),
  [turn-1587-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1587-assistant.md),
  [turn-1604-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1604-assistant.md)
- DR-0032 reuse of existing surprise/attention machinery:
  [turn-1631-user.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1631-user.md),
  [turn-1632-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1632-assistant.md),
  [turn-1633-user.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1633-user.md),
  [turn-1634-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1634-assistant.md)
- Generalization-gap audit and honest status:
  [turn-1627-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1627-assistant.md),
  [turn-1629-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1629-assistant.md),
  [turn-1646-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1646-assistant.md),
  [turn-1656-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1656-assistant.md),
  [turn-1661-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1661-assistant.md),
  [turn-1675-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1675-assistant.md)
- Mem0 adapter as a benchmark boundary:
  [turn-1666-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1666-assistant.md),
  [turn-1667-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1667-assistant.md)
- GPU residency, content-addressed stages, attention-as-graphs, latent fusion:
  [turn-1689-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1689-assistant.md),
  [turn-1690-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1690-assistant.md),
  [turn-1704-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1704-assistant.md),
  [turn-1711-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1711-assistant.md),
  [turn-1717-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1717-assistant.md),
  [turn-1724-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1724-assistant.md)
- DR-0033 cleanup-vs-progress resolution and episode-primary landing:
  [turn-1823-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1823-assistant.md),
  [turn-1827-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1827-assistant.md),
  [turn-1828-user.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1828-user.md),
  [turn-1829-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1829-assistant.md),
  [turn-1860-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1860-assistant.md)
- Contract-first tranches, node-vector readout, lifecycle hardening:
  [turn-1862-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1862-assistant.md),
  [turn-1863-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1863-assistant.md),
  [turn-1865-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1865-assistant.md),
  [turn-1871-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1871-assistant.md),
  [turn-2159-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-2159-assistant.md),
  [turn-2165-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-2165-assistant.md),
  [turn-2170-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-2170-assistant.md)
