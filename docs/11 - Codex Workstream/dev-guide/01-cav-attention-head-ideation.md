# 01 — CAV Attention-Head Architecture Ideation

**Phase:** 01 | **Merged turns:** 001–032 | **Dates:** 2026-08-15 to 2026-08-16

> **Status note.** Much of the QK/OV-operator ambition described here — recursive
> head-based traversal, OV state updates, attention heads as general relational
> operators — is later abandoned in favor of a much simpler marginal set-completion
> selector (DR-0025). Read this chapter as the origin of the substrate and the
> concept-vector vocabulary, not as the surviving retrieval mechanism. See
> [chapter 06](06-set-completion-selector.md) for what replaced it.

## Purpose

This chapter answers: what is the live memory system supposed to be built from,
and why is the substrate a truncated prefix of a large model's attention layers
rather than a text-embedding index? It records the architecture as it stood at
the end of the ideation phase, immediately before the drift-halt that opens
[chapter 02](02-retrieval-grounding-and-heat-diffusion.md).

## Design

### Core thesis

Concept Activation Vectors (CAVs) define memory addresses; a large model's
attention heads supply the relational machinery. A CAV is never the memory
itself — it indexes grounded episodes that retain source text and provenance.
The formal decomposition, for head *h* and concept vector *c*, is the OV
pullback `c_source = (W_OV)^T c`: `c^T o_i` says whether the head wrote the
concept, the attention pattern `A_ij` says which source positions were routed,
and `((W_OV)^T c)^T r_j` says how much concept-bearing content each source
supplied. QK selects *where* information comes from; OV determines *what* gets
written.

### Substrate: a resident seven-layer Qwen3-8B prefix

The system is **live, not a static compiler**. The truncated model stays
resident so that every new memory write and every recall query pass through the
same prefix and therefore inhabit the same activation space — which is what
eliminates the query-adapter problem that a compile-then-discard design would
face.

Concretely, at end of phase:

- Only `model-00001-of-00005.safetensors` from the official BF16
  [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) checkpoint is required for
  the live path: 3.996 GB, containing token embeddings plus complete
  transformer layers 0–6. Three incomplete layer-7 tensors in the shard are
  ignored by the loader.
- The prefix is 1.973 B parameters, runs in native BF16 on CUDA (PyTorch 2.7.1
  / CUDA 12.6, via Pixi), and peaks at 3.96 GB VRAM on the 8 GiB RTX 2070
  SUPER.
- A streaming safetensors loader (`src/memory_condense/qwen_prefix.py`)
  materializes the coherent prefix and exposes layer residuals, RoPE-adjusted
  Q/K, V, full QK attention maps, aggregate attention output, and isolated
  per-head `W_O` contributions. Runnable via `pixi run -e dev
  qwen-download-prefix` and `pixi run -e dev qwen-smoke`.
- Everything after the prefix — later blocks, final norm, LM head, generation
  machinery — is discarded. The memory system does not generate language; it
  returns grounded source spans to whichever model answers the user.

### Write path

Incoming text runs through the prefix once. At selected layers the system
produces conceptual chunks and stores, per episode: token-level per-head keys
and values, a CAV signature, the source text, and provenance. Chunks may
overlap — one span can belong simultaneously to several concepts. Writes also
update coactivation and temporal associations, and the episode is retrievable
on the next turn.

### Read path: the empirically measured split

The initial hypothesis — that raw head QK scores could serve as the semantic
entry index across independently encoded texts — **failed measurably** (12.5%
Recall@1 on the lookup smoke; token-level cross-sequence QK also failed; the
faithful token-level QK→softmax→V→O circuit scored 0). The heads learned
routing among tokens co-present in one causal context, not nearest-neighbor
search between contexts. Probing produced a layer split that defines the
end-of-phase architecture:

- **Entry: layer 5, CAV-gated residuals.** Raw layer-5 residuals reach 50%
  Recall@3; using the CAV as a bounded classifier / type prior (not as a
  similarity feature) raises the entry arm to 75% Recall@1 and 87.5% Recall@3
  on the calibration set. Layer-5 CAVs cleared explicit quality gates:
  held-out balanced accuracy up to 1.000, bootstrap direction stability 0.894,
  random-label controls at chance.
- **Association: selected layer-1 heads.** QK edges are only written inside a
  bounded *shared write context* (where the model actually learned them), with
  all 32 per-head edge weights preserved and the best four heads selected by
  calibration on known episode links. On a held-out split (heads selected on
  four anchor→fact pairs, evaluated on four unseen pairs), layer-1 graph
  expansion raises associative Recall@3 from 75% to 100%. This is expansion
  value, not entry value — Recall@1 stays at 0%.
- **Failed arms, kept explicit:** raw external QK as an entry index, and
  bounded OV recursion (multi-hop state updates), which never improved recall
  on either the lookup or the two-step associative tests.

### Lifecycle: live-head pruning

Pruning signals come from the live circuit rather than access counts. Each
retrieved memory accumulates (a) exponentially decayed QK attention mass —
"this memory was consulted" — and (b) its own RMS residual contribution after
V→O — "this memory actually moved the state." Pruning combines these with
recency and importance; `access_count` is diagnostic only. Guardrails against
attention forming self-reinforcing hubs: pins, source evidence, and a small
novelty/recency reserve held outside the learned utility, so a memory cannot be
erased solely because the current heads stopped looking at it.

### Offline teacher for J-space (in flight at phase end)

A true J-lens (average downstream Jacobian from an intermediate residual to the
final residual, composed with the unembedding) cannot be derived from the
seven-layer prefix — the missing layers are part of the measurement. So the
full five-shard BF16 teacher (16.38 GB, 399/399 tensors verified) was
downloaded strictly as an **offline compiler dependency**, never a runtime one.
Because the official jlens package requires Transformers 5.5+ while the working
`dev` environment is pinned at 4.57.6, J-lens work is isolated in a separate
`jspace` Pixi environment; the stable prefix environment is never upgraded in
place.

## Why this shape

- **Shared activation space beats adapters.** Keeping the prefix resident
  means writes and queries are encoded by the same machinery, so no projection
  from a separate retrieval encoder into teacher Q-space is ever trained. This
  is the decisive argument for a live substrate over an offline compile-and-
  discard pipeline.
- **8 GiB VRAM forces the prefix cut.** Full Qwen3-8B BF16 is ~16 GB of
  weights; the RTX 2070 SUPER cannot host it. Shard 1 happens to contain a
  coherent embeddings-plus-layers-0–6 prefix that runs in under 4 GB, so the
  cut is both principled (a contiguous prefix is the only slice whose input
  distribution is valid) and hardware-mandated.
- **Only the memory-addressing interface must survive.** The system never
  needs the teacher's language-generation ability — only `text →
  teacher-compatible query` and the head-level K/V/OV machinery. That is what
  licenses discarding everything after layer 6 and the entire LM head.
- **Measurements over hypotheses.** Every arm (entry, graph, recursion) was
  gated on a measured recall number before being kept, which is why the
  end-of-phase design is a layer-5/layer-1 split with two explicitly failed
  arms instead of the original uniform "recursive head traversal" picture.

## Why not X

- **Why not embed completed text chunks (the pre-pivot pipeline)?** The
  existing BGE-M3 + BM25 hybrid treats memories as opaque text vectors and
  cannot represent *which relationship* links two memories. CAV pullback plus
  head evidence gives typed, inspectable structure (concept nodes, episode
  nodes, typed edges) and stores sparse concept IDs instead of activation
  tensors. See [DR-0001](../decisions/0001-pivot-to-cav-attention-heads.md).
- **Why not keep the whole model?** Heads consume layer-specific residuals,
  not raw text, so the prefix up to the deepest used layer must be retained —
  but nothing after it. Everything downstream of layer 6 (and the LM head) is
  dead weight for memory addressing. See
  [DR-0002](../decisions/0002-attention-heads-only-substrate.md).
- **Why not download the full checkpoint from the start?** One shard supplies
  a complete, coherent prefix that fits the GPU; the other four shards were
  fetched later only when the J-lens made the full teacher an offline
  dependency. See [DR-0003](../decisions/0003-qwen-8b-head-safetensors.md).
- **Why not offline graph compilation as the primary mode?** It was the first
  proposal (compile QK edges with the big model, discard it, traverse at
  runtime), but it produces static associations and reintroduces the
  query-adapter problem. It survives only as an optional compression mode; the
  primary design is the continuously running partial-model engine.
- **Why not raw QK as the retrieval index?** Measured failure, twice: pooled
  episode-level mean keys collapsed into universal hubs, and even token-level
  cross-sequence QK (including the faithful softmax→V→O circuit) scored at or
  near zero. QK/OV is transport and association *after* a conceptual entry
  point, never the cross-context address itself.

## Open questions

- **Does QK/OV transport ever earn its keep?** At phase end it is retained as
  an expansion/transport layer on top of CAV entry, but no arm has shown it
  beating the entry retriever alone outside the held-out layer-1 association
  smoke. This question is eventually answered negatively — see
  [chapter 06](06-set-completion-selector.md) and DR-0025.
- **J-lens compilation is unresolved.** The `jspace` environment was still
  resolving at phase end; whether a sparse J-space dictionary can supply the
  third pruning term ("the moved information was concept-bearing") is untested.
- **OV recursion remains a failed arm.** Bounded multi-hop state updates never
  improved recall; the safeguards (gates, normalization, hop caps, cycle
  detection) exist but the mechanism has no demonstrated value.
- **Tiny calibration sets.** The layer split rests on eight-pair smokes with a
  four/four held-out split; nothing has been validated against LoCoMo-scale
  data or the existing hybrid retriever at a matched token budget.
- **CAV lifecycle.** Stable-versus-adaptive CAV components, drift detection,
  and cheap online updates (running centroids with counterexample correction)
  are specified but unbuilt; CAVs remain tied to a model fingerprint + layer +
  probe version and are not portable across model versions.

## Source turns

Key raw turns for this phase (sub-turn files; decision refs DR-0001/2/3 map to
merged turns 007/009, 011/013, and 019/021 respectively):

- [turn 001](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-001-user.md) — project goal: token saving, recall, pruning
- [turn 018](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-018-user.md) — the CAV pullback / QK–OV pivot (DR-0001)
- [turn 020](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-020-assistant.md) — pullback formulation, conceptual chunking, memory graph
- [turn 021](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-021-user.md) — heads of a larger model as recursive memory layers
- [turn 023](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-023-assistant.md) — recursive memory attention; three implementation levels
- [turn 024](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-024-user.md) — "why can't we just use the attention heads?" (DR-0002)
- [turn 025](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-025-assistant.md) — what can be discarded; memory-addressing interface
- [turn 030](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-030-assistant.md) — theory note landed in docs
- [turn 031](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-031-user.md) — partial-layer download question
- [turn 034](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-034-user.md) — "live memory system, not a static one"
- [turn 035](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-035-assistant.md) — live write/read paths; persistent KV cortex
- [turn 036](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-036-user.md) — download Qwen 8B head layers (DR-0003)
- [turn 041](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-041-user.md) — "just get the right safetensor files"
- [turn 050](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-050-assistant.md) — verified shard-1 prefix substrate
- [turn 056](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-056-assistant.md) — layer-5 CAV quality gates pass
- [turn 059](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-059-assistant.md) — first retrieval failure of raw QK
- [turn 064](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-064-assistant.md) — CAV-as-classifier fixes the hub problem
- [turn 070](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-070-assistant.md) — held-out layer-1 association result; layer split
- [turn 074](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-074-user.md) — QK routes, OV moves information
- [turn 076](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-076-user.md) — J-Space question
- [turn 078](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-078-assistant.md) — J-lens requires the full teacher, offline
- [turn 080](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-080-user.md) — live heads as the pruning signal
- [turn 082](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-082-assistant.md) — live-head pruning implemented
- [turn 088](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-088-assistant.md) — full teacher verified offline; jlens compatibility
