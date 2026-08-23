# 02 — Retrieval grounding, benchmarks, and heat diffusion

**Phase window:** merged turns 033–072 (2026-08-16) · **Previous:** [CAV attention-head ideation](01-cav-attention-head-ideation.md) · **Next:** [95% associative memory campaign](03-95-percent-associative-memory-campaign.md)

## Purpose

This chapter fixes the design as it stood at the end of phase 02: retrieval work re-grounded in real corpora and untouched evaluation splits, the transformer slice locked to a transient linker/inspector role with zero durable transformer state, a backend-pluggable storage plane, a parallel benchmark rig, and a heat-diffusion read stage over the persisted attention graph.

## Design

### The core invariant: zero durable transformer state

Everything in this phase hangs off one rule, locked in by [DR-0005](../decisions/0005-llm-slice-linker-only.md): the Qwen slice inspects memory; it never contains it.

- The model contributes fixed-size weights (the Qwen prefix) and a fixed-size transient workspace holding at most a small candidate set under an explicit token/candidate ceiling.
- Every activation, attention map, and K/V tensor is discarded after each inspection. Retained transformer K/V is zero bytes, and evaluation runs assert it.
- The only durable state lives outside the model: external source pointers (text/chunks in the existing store), compact CAV coordinates, sparse QK/OV link edges with per-head statistics, and usage/decay/reinforcement counters.
- "Nested memory layers" means repeated bounded inspections, never layers of stored transformer state:

  ```text
  external IDs/text → fetch small candidate set → heads inspect QK/OV
      → emit scores/next IDs → discard every activation/K/V tensor
      → fetch the next small set
  ```

  Nothing from one hop is appended to the next transformer context; only candidate IDs and scalar scores cross hops.
- The legacy K/V laboratory carries a hard 64-item ceiling so it cannot silently become a corpus-scale store again.

The API terminology reflects the role: the slice is an **inspector** / **link compiler**, not a head-resident memory.

### Write path: link compilation

When a memory is written, the compiler places bounded candidate groups before the new memory in one workspace, measures attention from the new-memory tokens to each candidate span, produces a 32-value per-head QK vector per candidate, ranks candidates, and persists only compact edges (QK evidence, OV transport, head coordinates, provenance) before discarding all activations. Linking happens at write time; reads can then follow stored links without keeping Qwen activations alive.

The measured bottleneck at phase end is write-time graph coverage, not read depth: the compiler feeds Qwen only three candidates drawn from prior hybrid results. The agreed extension is a diversified candidate pool — a union of dense and BM25 neighbors, adjacent conversation turns, same document/section, shared entities, reply/provenance relations, and CAV buckets — inspected in several bounded groups, with a contextual bandit (discounted Thompson sampling or LinUCB) later allocating the fixed inspection budget across those sources. The bandit is staged behind a fixed diversified policy plus a logged coverage funnel (`gold in candidate pool → retained as edge → two-hop reachable → survives ranking → admitted at k=5`), because an unevidenced bandit would reinforce whichever source already produces the most links.

The division of labor is fixed:

```text
candidate generators
      ↓
contextual bandit chooses what deserves inspection
      ↓
attention heads choose the useful links
      ↓
safe admission decides whether links may affect retrieval
```

### Storage plane

The association-store contract is backend-neutral so backends can be benchmarked rather than baked into the retrieval algorithm:

- **SQLite** — deterministic tests and portable local operation. One lightweight read connection per concurrent worker; a shared connection across sweep threads corrupts.
- **Redis** — the live association layer: packed CAV blobs keyed by memory ID, bounded sorted sets per node for QK links, packed hashes for edge/head statistics, atomic reinforce/decay/utility updates with top-`max_degree` trimming, strict `maxmemory`, AOF+RDB or rebuild-from-event-log durability.
- **Chroma** (or the existing chunk store) — the document/embedding layer only. It is a poor fit for constantly reinforced-and-pruned adjacency, and its approximate HNSW is unnecessary while the CAV bank is tiny.

No backend ever stores token K/V or hidden-state sequences.

Layered on top is the **native hypergraph memory plane** (documented in `docs/03 - Architecture/01 - Native Hypergraph Memory Plane.md`): each Qwen inspection is naturally higher-order — one source, several candidates, activated CAVs, selected heads, one episode — so hyperedges (implemented as ordinary `hyperedges`/`hyperedge_members` tables) become the canonical compiled-event layer, storing member IDs and roles, aggregate QK evidence, OV transport, reinforcement, and provenance. Pairwise QK edges remain the fast serving projection. A relation among A+B+C costs O(3) incidences instead of O(3²) pairwise links, and pruning can cool one inspection event atomically. Adoption is gated behind migration and measurement gates; the pairwise graph is not replaced yet.

### Read stage: heat diffusion

[DR-0007](../decisions/0007-heat-diffusion-framing.md) reframes the read stage. The attention head is a local transition operator; the external memory graph carries the heat. The head never holds or classifies the collection — it only answers "given this active memory, which candidate should receive heat next?", one candidate (or a tiny pairwise-scored stream) at a time.

```text
query → seed memories with initial heat
    → choose hottest node
    → stream neighboring candidates one at a time
    → attention head scores current → candidate
    → move heat to the best next item or tiny beam
    → discard activations and repeat
```

Key properties:

- **Transition score** = calibrated QK(i→j) + λ·OV alignment + μ·CAV compatibility − cycle/popularity penalties. Raw attention weight is never used as global heat directly, because attention is locally normalized within one workspace; heat is maintained externally and multiplied by calibrated transition probabilities so workspace composition cannot change the total.
- **Multi-anchor accumulation**: heat sums over corroborating paths, so a memory reached independently from several anchors accumulates evidence — a better admission signal than the strongest single edge, and the reason three-hop diffusion succeeds where three-hop best-path traversal regressed.
- **Heat as budget**: source heat converts directly into prompt-token allocation — `attention evidence → source heat → token allocation → memories shown to the LLM`. Chunk priority is `source_heat × query_relevance × novelty × relation_confidence / token_cost`, so heat purchases information rather than rewarding verbose chunks. Only selected chunks reach the final LLM; rejected candidates stay external, so exploring more memory costs no final prompt tokens.
- **Bounded and model-free at read time**: the implemented stage runs over the persisted attention graph with a scalar ID frontier (32 IDs), a tiny external beam, and fixed hydration at k=5. Only IDs, scalar heat, paths, and compact CAV values cross iterations.
- **Dual-channel selector**: replay showed pure diffusion can miss a rare high-value edge that ranked-QK traversal recovers, so one calibrated max-path attention choice is protected as an exploitation slot while diffusion allocates the remaining exposure budget.
- Guards against known failure modes: degree normalization, restart/leak, per-source caps, novelty penalties, and a minimum diversity allocation prevent large-source heat capture, cycle circulation, and single-source prompt monopoly.

Measured at phase end: three-hop diffusion with the 32-ID frontier and two heat slots preserved recall while cutting raw memory tokens 12.4% (973.9 → 852.9) on the development split and 9–19% on replayed earlier stores.

### Benchmark rig and evaluation discipline

[DR-0006](../decisions/0006-pivot-to-performance-rig.md) moved the work onto a dedicated parallel-run rig (housed under the Downloads workspace, outside the repo):

- Parallel independent retrieval/evaluation arms on CPU, each opening its own lightweight DB/association reader — no ANN index, embedding model, or Qwen per arm.
- A single serialized Qwen compilation worker per GPU, because several 4.6 GB prefix workers would multiply VRAM and violate the bounded-memory design. Explicit CPU/GPU concurrency limits; persisted artifacts are reused across arms.
- Sweeps isolate one variable at a time (degree, QK-only vs QK+CAV, reserved-slot count) under identical budgets.

Evaluation is grounded in real corpora: the 13 MB session snapshot (`data/build-session-8f7f7561.snapshot.jsonl`) and a 59 MB personal notes repository inventoried by a read-only loader that classifies, hashes, deduplicates (21 redundant files removed exactly), and preserves provenance in a manifest without copying content. Splits that informed any fix are demoted to development data; claims are made only on untouched splits, and cold-start ingestion cost is measured separately from recall-time cost. Representative honest numbers at phase end: hybrid k=3 at 83.3% on the fresh locked split, two-hop QK with one reserved slot reaching 91.7% at ~54% fewer tokens than hybrid k=10 (development data), degree-two pruning from 1,204 → 812 edges with no recall loss, and three fresh misses that remain unreachable because their gold was never linked at write time.

## Why this shape

1. **Storing transformer state recreates the problem the system exists to solve.** Caching token-level K/V per chunk turns the model slice into the memory store and reintroduces transformer-context accumulation with corpus-scale growth. Bounding the workspace and persisting only compact scalars keeps memory cost proportional to the link graph, not the corpus ([DR-0005](../decisions/0005-llm-slice-linker-only.md)).
2. **Development results were quietly becoming tuned results.** The 1.000/1.000 fusion score was informed by inspected errors; the untouched-split discipline, funnel instrumentation, and parallel rig exist to keep every claim attached to data the fix never saw ([DR-0004](../decisions/0004-halt-infrastructure-drift.md), [DR-0006](../decisions/0006-pivot-to-performance-rig.md)).
3. **Single-GPU VRAM is the binding hardware constraint.** One serialized Qwen worker plus embarrassingly parallel CPU arms delivers throughput without violating the bounded-memory design, and pushes all cheap experimentation into the model-free read stage.

## Why not X

- **Why not keep building J-Space compiler infrastructure.** It was drift away from the measurable retrieval result (0.750/0.875 CAV-gated entry); the install was stopped and integration paused with no dependency left behind. See [DR-0004](../decisions/0004-halt-infrastructure-drift.md).
- **Why not token-level K/V caching as the memory store.** A benchmark already underway was cancelled mid-run because its K/V growth "would measure the wrong architecture." Persistent memory is compact external records plus CAV/edge metadata only. See [DR-0005](../decisions/0005-llm-slice-linker-only.md).
- **Why not keep tuning recall on saturated splits.** Once the easy notes slice hit 8/8, further tuning there was noise; the pivot moved effort to performance and to a harder untouched split, which promptly showed compiled links regressing (75.0% at linked k=3) — the signal that made the coverage work necessary. See [DR-0006](../decisions/0006-pivot-to-performance-rig.md).
- **Why not one-shot attention classification over a fixed candidate triple.** Cross-candidate softmax interference, candidate-order and length bias, and constant re-normalization make joint contexts fragile; pairwise/streamed transition scoring gives constant transformer memory and a streamable pool. See [DR-0007](../decisions/0007-heat-diffusion-framing.md).
- **Why not deeper best-path traversal instead of diffusion.** Three-hop best-path reads regressed recall to 83.3%/75%; diffusion at three hops did not, because it sums corroborating paths instead of keeping one. The two results are consistent, not contradictory.
- **Why not Chroma (or any vector DB) for the live link graph.** Bounded, constantly reinforced adjacency needs atomic counters and sorted-set trimming; read–rank–rewrite cycles over a vector store are the wrong primitive. Chroma stays at the document/embedding layer.

## Open questions at phase end

- **Write-time coverage.** Three fresh misses have no QK path within three hops and poor CAV rank; the diversified candidate pool and coverage funnel are designed but not yet run. Does the fixed eight-candidate diverse pool raise oracle coverage — the precondition for the bandit mattering at all?
- **Head selection.** The "strongest four heads per candidate" rule rewards arbitrary attention spikes. A fixed sparse head/layer scorer learned on training families, validated on untouched families, is specified but unbuilt.
- **OV semantics.** OV evidence is still essentially update magnitude; alignment against the new memory's residual direction, a relevant CAV, or the relation being learned is unimplemented, and direction-asymmetric edges are not yet handled.
- **CAV vocabulary.** Two float32 coordinates per chunk are too coarse for document-scale navigation; the 16–64 validated-concept vocabulary (entities, decisions, constraints, temporal continuation, cause/effect) is future work.
- **Coverage-aware pruning.** Degree-two pruning is safe so far, but utility-only pruning can sever the sole bridge to a rare concept; bridge/unique-coverage protection is specified, not proven.
- **Dual-channel generalization.** The protected exploitation slot fixed the one replay regression; whether one slot suffices on larger stores is untested.
- **Hypergraph plane adoption.** The measurement gates for promoting hyperedges from documented design to serving path have not been exercised.
- **Positioning.** Against Mem0, the heat layer is a stronger context-control primitive (allocation of text budget rather than a fused retrieval score), but the maturity and public-evaluation gap is large.

## Source turns

Key raw turns for this phase (full index in [00-overview.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/00-overview.md)):

- DR-0004 scope cut: [turn-092-user.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-092-user.md), [turn-093-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-093-assistant.md), [turn-108-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-108-assistant.md)
- Real corpora grounding: [turn-115-user.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-115-user.md), [turn-137-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-137-assistant.md), [turn-153-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-153-assistant.md)
- DR-0005 linker/inspector boundary: [turn-170-user.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-170-user.md), [turn-171-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-171-assistant.md), [turn-172-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-172-assistant.md), [turn-175-user.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-175-user.md), [turn-176-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-176-assistant.md), [turn-196-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-196-assistant.md)
- Storage backends: [turn-201-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-201-assistant.md) (Redis), [turn-207-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-207-assistant.md) (Chroma)
- DR-0006 performance rig: [turn-218-user.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-218-user.md), [turn-219-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-219-assistant.md), [turn-228-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-228-assistant.md), [turn-237-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-237-assistant.md), [turn-243-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-243-assistant.md)
- Hypergraph plane: [turn-245-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-245-assistant.md), [turn-251-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-251-assistant.md)
- Recall diagnosis, bandit, selector: [turn-289-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-289-assistant.md), [turn-291-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-291-assistant.md), [turn-293-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-293-assistant.md)
- DR-0007 heat diffusion: [turn-294-user.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-294-user.md), [turn-295-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-295-assistant.md), [turn-297-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-297-assistant.md), [turn-299-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-299-assistant.md), [turn-305-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-305-assistant.md), [turn-306-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-306-assistant.md)
- Mem0 comparison: [turn-308-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-308-assistant.md), [turn-315-assistant.md](../../../_ingest/codex-2026-08/raw/phase-02-retrieval-grounding-and-heat-diffusion/turn-315-assistant.md)
