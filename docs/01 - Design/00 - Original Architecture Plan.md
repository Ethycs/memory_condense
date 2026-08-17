# Original Architecture Plan — LLM API + local bge-m3 memory manager

**Status**: 🟡 LARGELY REALIZED — Phases 0, 1, 2, 3 and 5 are built. Phase 4A, use-dependent relational consolidation across memory partitions, is implemented in schema v9; Phase 4B, materialized cold era summaries, remains unbuilt. See `03 - Architecture/00` for the as-built map and `06 - Roadmaps/00` for what is open.
**Date**: 2026-08-14
**Date filed**: 2026-08-14 (content predates repo's first commit)
**Supersedes**: the "Phases 1–4 unbuilt and under re-evaluation" status of this document; also the canonical copy of the root-level `arch_instructions.md` (original retained in place for now; see `09 - Archived/ARCHIVE-INDEX.md`)

The plan assumes: no attention/hidden-state access, local memory manager owns all state, bge-m3 provides embeddings + optional lexical weights, LLM API answers the user and proposes memory updates in a strict schema.

## 1. Architecture

1. **TranscriptStore** (append-only) — every turn: raw text, role, timestamp, turn_id.
2. **Chunker** — sentence→paragraph merge; chunk_id, text, span pointers into transcript.
3. **Embedding + Salience (bge-m3)** — dense embedding per chunk; optionally lexical weights for sparse-ish match.
4. **MemoryStore** — typed, compact "memory items" + provenance pointers; chunk embeddings + lightweight index.
5. **Retrieval & Rerank** — hybrid candidates (sparse-ish + dense); rerank by relevance × importance × recency/pins.
6. **ContextPacker** — deterministic token budget allocation (recent window + memory header + expansions).
7. **LLM API Client** — one main call per user message (answer + memory_ops); optional second cheap call for memory_ops.
8. **Validator** — enforces provenance and conflict/supersede rules; rejects hallucinated memories.

## 2. Data schemas

**Transcript turn**: `turn_id`, `role`, `text`, `created_at`.

**Chunk**: `chunk_id`, `turn_id`, `text`, `start_char`, `end_char`, `embedding`, `lexical_weights?`.

**MemoryItem** (the long-term unit):
- `mem_id`
- `type`: `Decision | Preference | Constraint | Entity | Definition | Task | Correction`
- `content`: 1–2 lines canonical form; `details`: optional (short)
- `provenance`: `{turn_ids, chunk_ids, quote_spans}` — **required**
- `status`: `active | superseded | deleted`; `supersedes`: mem_id?
- `pins`: `user_pinned | system_pinned | none`
- `heat`: `HOT | WARM | COLD`
- `energy`: float 0..1 + decay stats (`created_at`, `last_access_at`, `half_life_s`)

**MemoryOps** (LLM output schema): `create[]`, `update[]`, `supersede[]`, `delete[]`, `pin[]` — every op must include provenance (turn/chunk refs + quote). **This one rule is what keeps a pure-LLM-API approach from drifting.**

## 3. Retrieval + ranking

Two internal scores: **Relevance**(query, item) — useful right now; **Importance**(item) — worth keeping hot.

Candidate generation (fast): dense ANN (cosine) and/or sparse-ish (BM25 or bge-m3 lexical weights); top N=50–200.

Rerank (cheap deterministic scalar):

```
score = wR*relevance + wI*importance + wP*pin_boost + wT*recency - wS*superseded_penalty
```

Importance: rule/features baseline (decisions, constraints, corrections, IDs/numbers, named entities), optionally a tiny LLM classify call at ingestion. Relevance: `cos_sim(query_emb, item_emb)`, optionally blended with lexical overlap.

## 4. Context packing (deterministic, budgeted)

- Recent window: 4,500 tokens · Memory header: 600–1,200 tokens (typed bullets) · Expansions: 0–800 tokens (verbatim quotes only when needed).
- Order: system/policies → memory header (active + pinned + top-ranked only) → recent turns → expanded snippets.

## 5. Ingestion loop (every turn)

1. Store transcript → 2. Chunk (80–300 token merge) → 3. Embed (bge-m3) → 4. Extract candidate memory items (V1: rules; V2: short LLM MemoryOps call) → 5. Validate & apply (must quote actual turns; corrections **supersede**) → 6. Update energy/heat (new important items HOT, others WARM, lazy decay on access).

## 6. Storage & indexing

SQLite for transcript + memory items; FAISS or hnswlib for ANN; embeddings as SQLite blobs or mmap + ID mapping. Cold-tier later: cluster summaries + centroid index.

## 7. Build phases

| Phase | Scope | Status (2026-08-14) | Realized in |
| --- | --- | --- | --- |
| 0 | TranscriptStore + Chunker + bge-m3 + similarity demo | ✅ Built | `transcript_store.py`, `chunker.py`, `embedding.py`, `retrieval.py` |
| 1 | MemoryItem + pins + ContextPacker + HOT/WARM retrieval | ✅ Built | `schemas.py`, `memory_store.py`, `context_packer.py`, `ranking.py` |
| 2 | LLM memory_ops (strict JSON) + Validator + supersede | ✅ Built — **rule-based extractor is the default**; the LLM path exists and is exercised by tests, but is opt-in via `extractor=LLMExtractor(complete=…)` | `extractor.py`, `validator.py`, `memory_store.supersede` |
| 3 | Decay + tiering + access reheating | ✅ Built | `decay.py`, `memory_store.touch` / `heat_counts` |
| 4A | Live relational consolidation driven by later packed prompts | ✅ Built — causal prompt/response binding, decayed repeated co-activation, bounded iterative reads, optional CAV/QK/OV weighting | `consolidation.py`, schema v9, `MemoryCondenser.build_context` |
| 4B | Materialized cold summaries (cluster "era summaries") | 🔲 **Unbuilt** — now gated on stable assemblies learned by 4A | — |
| 5 | Eval harness (scripted conversations + QA probes; token cost, recall, correction robustness) | ✅ Built — self-replay **and** QA-probe benchmark modes; token cost + latency now instrumented via `UsageStats` | `eval/runner.py`, `eval/benchmark.py`, `eval/analysis.py` |

All of the above is committed and merged to `main` as of 2026-08-14 (merge `f3edc91`). "Built" here means "exists and passes tests" (407 passing), not "measured" — see the Decision Point in `06 - Roadmaps/00`. Merging changed the first half of that sentence and nothing about the second.

### Where the as-built system departs from this plan

| Plan said | As built | Why |
| --- | --- | --- |
| §1.5 "hybrid candidates (sparse-ish + dense)" via bge-m3 lexical weights | Classic Okapi BM25 (`k1=1.5`, `b=0.75`) over a SQLite inverted index; `chunks.lexical_weights` stores plain term frequencies | BM25 needs no model call and no extra dependency, and is deterministic across machines. bge-m3 learned weights remain a possible swap-in behind the same interface. |
| §6 "FAISS or hnswlib for ANN" for memory items too | hnswlib for **chunks**; memory items use brute-force exact cosine (numpy) | Memory items number in the tens-to-low-hundreds. An exact scan is faster than maintaining a second graph and never returns a stale neighbour after a supersede. |
| §5.4 "V1: rules; V2: short LLM MemoryOps call" | Both exist; V1 is the default and V2 takes an **injected** `complete` callable rather than importing an SDK | Keeps the core provider-agnostic (axiom 1 in `03 - Architecture/00`). |
| §4 memory header "600–1,200 tokens" | Fixed at 900 (`ContextBudget.memory_header_tokens`) | A single default inside the stated range; the range is still the tuning envelope. |

## 8. Local vs API split

**Local**: chunking, embedding + indexing (bge-m3), retrieval + scoring, memory state machine, context packing. **LLM API**: user-facing generation, structured memory_ops extraction.

## 9. Default operating parameters

Chunk 120–250 tokens · candidate top-100 · memory header ~900 tokens · expansions max 3 × ≤250 tokens · HOT cap ~20 · heat thresholds HOT ≥ 0.75, WARM ≥ 0.25 · pins override decay.

Realized defaults, for reference: `ContextBudget` 4500 / 900 / 800 with up to 10 ranked expansions of ≤250 tokens (the 800-token aggregate ceiling decides how many fit); `RankWeights` relevance 1.0, importance 0.3, pin 0.5, **energy** 0.2, superseded penalty 1.0; hybrid `alpha` 0.65 (dense weight) over 100 candidates per side; half-life 30 conversation turns; reheat closes 25% of remaining headroom at most once per turn. The design's **HOT cap ~20 is now enforced** (`decay.heat_map`) — pool-relative and applied at tier derivation, so heat stays derived-never-stored.

Two departures from the plan's wording, both deliberate and both corrections of as-built behaviour:

- The scalar's fourth term is **energy**, not recency. The plan wrote `wT*recency`; the implementation computed that from a second copy of the decay exponential, which had drifted from `decay.py` and — because `touch` restamps `last_access_at` — evaluated to a constant 1.0 for every item ever recalled. Energy is the same time signal times a stored amplitude that access frequency moves. See `08 - Analysis/01`.
- Reheat is **multiplicative**, not the plan's flat addition. A flat `+0.25` has a fixed point that clamps at 1.0 for anything touched more often than ~every three days, which reintroduces the constant-term problem one level down.

---

**Reconciliation note (2026-08-14)**: the earlier note here said Phases 1–4 should not be built without consulting `04 - Reference/00 - Competitive Landscape 2026.md`, because MemDelta suggests memory-ops machinery may not beat a well-tuned retrieval baseline. **They were built anyway, before that gate was cleared.** That does not invalidate the caution — it relocates it: the open question is no longer "should we build it?" but "does it earn its place?", and it is answered by the same benchmark run. The Decision Point in `06 - Roadmaps/00` is now *unblocked* (the benchmark harness exists) and still *open* (no benchmark has been run).

---

**Verification block**: run

```powershell
pixi run python examples/memory_demo.py      # downloads bge-m3 on first run (~2.3 GB)
```

It exercises Phases 1–3 end to end: rule-based extraction with provenance quotes, a fabricated memory being rejected by the Validator, pinning (decay exemption), ranked recall with the score breakdown, and a packed context with per-section `token_counts` and `dropped` counts. Then decide whether Phase 4 (cold era summaries) is worth building, or whether the benchmark run should come first.
