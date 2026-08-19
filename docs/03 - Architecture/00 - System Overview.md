# memory_condense System Overview (as-built)

**Status**: CURRENT
**Date**: 2026-08-18
**Supersedes**: the earlier 2026-08-14 "dense-retrieval-only / no condensation yet" version of this document, which described the tree at commit `cd9f423`
**Applies to**: the current working tree — core memory, compiled Qwen
association artifacts, bounded associative reads, eval harness, and MCP server.

## Executive summary

memory_condense is a local memory manager: transcript → chunker → bge-m3
dense index plus BM25, a typed `MemoryItem` store with provenance-enforced
extraction, exponential energy decay, deterministic reranking, and a
hard-budgeted `ContextPacker`. An optional staged Qwen3 prefix compiler emits
compact CAV signatures and sparse QK/OV association edges into SQLite schema
v11, whose turns also retain document/session source identity and whose live
Hebbian projection learns bounded same-turn chunk co-access. Ordinary
associative reads do not load Qwen; they traverse only external
IDs and scalars, then hydrate the selected chunks. The core remains
provider-agnostic, while public common-benchmark validation remains open.

## 1. Architecture

```
                          ┌──────────────────────────────────┐
                          │   MemoryCondenser (facade)       │  condenser.py
                          │  ingest · search · search_hybrid │
                          │  recall_memories · build_context │
                          └───┬──────────┬─────────┬─────────┘
        ingest(role,text)     │          │         │  build_context(user_text)
   ┌──────────────────────────┘          │         └──────────────────────┐
   ▼                                     ▼                                ▼
┌───────────────┐   ┌──────────────┐  ┌────────────────────┐   ┌────────────────────┐
│TranscriptStore│──▶│   Chunker    │  │ Extractor          │   │   ContextPacker    │
│ (append-only) │   │ pySBD+greedy │  │ RuleBased (default)│   │ 4500 / 900 / 800   │
│  turns table  │   │  120–250 tok │  │ LLM (injected fn)  │   │ ≤3 exp × ≤250 tok  │
└───────┬───────┘   └──────┬───────┘  └─────────┬──────────┘   └─────────┬──────────┘
        │                  ▼                    ▼                        ▲
        │        ┌──────────────────┐   ┌────────────────┐               │
        │        │ EmbeddingService │   │   Validator    │  quote MUST    │
        │        │ BAAI/bge-m3      │   │ provenance gate│  appear in the │
        │        │ dim from model   │   └───────┬────────┘  cited turn    │
        │        │ unnormalized f32 │           │                         │
        │        └────────┬─────────┘           ▼                         │
        │                 │            ┌────────────────┐                 │
        │                 │            │  MemoryStore   │─── MemoryResult ┤
        │                 │            │ CRUD·supersede │                 │
        │                 │            │ soft-delete·pin│                 │
        │                 │            │ touch(reheat)  │                 │
        │                 │            │ brute-force cos│                 │
        │                 │            └───────┬────────┘                 │
        │                 │                    │ decay.py (½-life 30 turn)│
        │                 │                    │  ranking.py (rerank)     │
        │                 ▼                    │                          │
        │   ┌───────────────────────────┐      │                          │
        │   │    SimilarityRetriever    │──────┼─── RetrievalResult ──────┘
        │   │ query()      dense only   │      │
        │   │ hybrid_query() dense∪BM25 │      │
        │   │ hnswlib cosine M=16 efc200│      │
        │   │ + LexicalIndex (BM25)     │      │
        │   └─────────────┬─────────────┘      │
        ▼                 ▼                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ SQLite — db.py: WAL, foreign_keys=ON, schema_version 11 (v1 migrates up)  │
│   turns(source_id) · chunks(text, embedding, lexical weights, hnsw label) │
│   chunk_terms(term, chunk_id, tf)          ← BM25 inverted index          │
│   memory_items(...) · memory_provenance(...) ← mandatory provenance rows  │
│   CAV/QK/OV + Hebbian + episodic discourse artifacts/coverage/receipts    │
│   meta(schema_version)                                                    │
│   + hnsw_index.bin  (cache only — rebuildable via rebuild_index())        │
└───────────────────────────────────────────────────────────────────────────┘

  eval/ (litellm → Anthropic API; the ONLY LLM callers in the repo)
    loader ─┬─ .txt/.md exports ─▶ runner (teacher-forced replay) ─▶ responder ─▶ judge ─▶ report
            └─ LongMemEval/LoCoMo ─▶ benchmark (QA probes, F1/EM) ─▶ report
    analysis (offline: binned_scores · compare_runs · ascii_curve · to_csv)
```

## 2. Per-subsystem

### TranscriptStore — `transcript_store.py`
- **Domain**: durable raw history. **Runs at**: every `ingest`. **Inputs**: `(role, text, source_id?)`. **Outputs**: source-identified `Turn` rows.
- **Structure**: append-only `turns` table; `append` / `get_turn` / `get_recent` / `get_all` / `count`.
- **Hard constraint**: never mutated, never deleted. Every other table is derivable from it.

### Chunker — `chunker.py`
- **Domain**: span segmentation. **Inputs**: one turn. **Outputs**: `Chunk`s with char-span provenance into the turn.
- **Structure**: pySBD sentence split (`clean=False`) → greedy merge into `[120, 250]` cl100k tokens; oversized sentences sub-split on `"; "` then `", "` then word-level `_hard_split`; trailing runt merged back if it fits.

### EmbeddingService — `embedding.py`
- **Domain**: dense representation. **Runs at**: lazy model load, batch 32.
- **Structure**: sentence-transformers `BAAI/bge-m3`, `normalize_embeddings=False`.
- **`dim` is no longer hardcoded** (`validated`): it returns a cached value if resolved, the constant `1024` **only** when the model is unloaded *and* `model_name == "BAAI/bge-m3"` (so `MemoryCondenser.__init__` can size hnswlib without a 2.3 GB download), otherwise it loads the model and asks `get_sentence_embedding_dimension()`. A non-default model therefore reports its true dimension instead of silently corrupting a 1024-d index.

### LexicalIndex — `lexical.py`
- **Domain**: sparse candidate generation. **Inputs**: chunk text. **Outputs**: `(chunk_id, bm25_score)` pairs.
- **Structure**: Okapi BM25, `k1=1.5`, `b=0.75`, over the `chunk_terms` inverted index with document length in `chunks.term_count`. Tokenizer = lowercase, maximal alphanumeric runs (`[^\W_]+`), drop tokens shorter than 2 chars, drop a deliberately small stopword list (`no`/`not` are **not** stopwords — they carry preference polarity).
- **Hard constraints**: no in-memory state — `N`, `avgdl` and document frequencies are recomputed from SQLite on every read, so two processes on one database always agree. Scores are raw BM25; callers must normalize before blending. Ties break on `chunk_id`, so ordering is deterministic.

### SimilarityRetriever — `retrieval.py`
- **Domain**: chunk candidate generation + ranking. **Outputs**: `RetrievalResult`.
- **`query(query_embedding, k, ef_search)`** — pure dense, `score = 1 − cosine_distance`. **Deliberately unchanged**: the `k=0` / `k=N` ablation and every historical number depend on it.
- **`hybrid_query(query_text, query_embedding, k, ef_search, candidates=100, alpha=0.65)`** — pulls `candidates` from each side, min-max normalizes each side independently, unions them (dense candidates in rank order, then lexical-only ones), blends with `ranking.blend_hybrid(dense, lexical, alpha)` where `alpha` is the **dense** weight. `alpha=1.0` reproduces the dense ordering; `alpha=0.0` is pure BM25. Results carry the blend in `score` and the normalized components in `dense_score` / `lexical_score`, so `score == blend_hybrid(dense_score, lexical_score, alpha)` exactly.
- **`source_query(query_embedding, k_sources)`** — mean-pools normalized chunk embeddings per durable source, ranks sources, then returns complete winning sources in conversation order. It is a diagnostic/whole-document arm and remains subject to the same final prompt cap.
- **`hydrate_source_neighbors(anchors, radius, max_neighbors)`** — keeps ranked
  anchors first, then exposes bounded earlier/later chunk shells inside only
  their activated sources. It carries IDs and scalar scores until final chunk
  hydration; it neither stores token state nor loads a transformer.
- **Source-conditioned second stage** — direct hybrid anchors remain stable
  while a separately bounded ranked prefix activates source IDs. Lower-ranked
  chunks may enter only through those source IDs. This separates evidence
  discovery from the number of chunks immediately returned.
- **Partition-local source search** — the opt-in `source_local_search` strategy
  streams every durable embedding in each activated source instead of filtering
  the global ANN pool. Dense and BM25 candidate buffers are bounded per source;
  text is hydrated only for the final rank. Scores are normalized once across
  the activated-source union. Per-source normalization is intentionally
  forbidden here because it makes every weak partition manufacture a
  top-scoring chunk and was measured to displace temporal evidence.
- **TF-ISF source activation** — an opt-in lexical route treats each durable
  source/session as an aggregate document. BM25-style term frequency and
  inverse source frequency select a bounded set of source IDs; only candidates
  already inside the bounded retrieval pool can be admitted. The live index
  caches aggregate source lengths, not text or postings outside the existing
  chunk index.
- **Lazy source contraction** — `SourceContractionIndex` builds a transient,
  partition-local HSC-style hierarchy from source-centroid embeddings. Internal
  nodes contain child pointers, a derived centroid, level, and member count—no
  transcript text or transformer state. Query-seeded reads ascend from active
  source leaves and inspect bounded siblings; writes invalidate the hierarchy
  and the next read reconstructs it from authoritative chunk embeddings.
- **`add_chunks`** writes the dense index, persists the embedding, **populates `chunks.lexical_weights`** with the chunk's term-frequency map, and feeds the same chunks to the BM25 index. The column that was always NULL is now real.
- **`delete_chunk(chunk_id)`** — clears embedding + `hnsw_label` in SQLite (authoritative), marks the label deleted in the live hnswlib graph best-effort, and drops the BM25 postings. The chunk **row** survives so memory provenance pointing at it cannot dangle.
- **Structure**: hnswlib `space="cosine"`, `M=16`, `ef_construction=200`, `max_elements=100_000`; chunk↔label mapping lives in `chunks.hnsw_label` (single source of truth); `rebuild_index()` reconstructs the `.bin` from SQLite blobs.

### Compiled and live association plane — `association_store.py`, `associative_retrieval.py`, `heat_diffusion.py`, `hebbian_retrieval.py`, `consolidation.py`

- **Write-time domain**: a bounded Qwen3 prefix inspects small candidate sets
  and compiles fixed-width CAV signatures plus per-head QK/OV edge evidence.
  The model slice is a linker, not the durable memory store.
- **Durable state**: versioned association artifacts, float32 CAV signatures,
  and degree-bounded sparse edges. No token K/V or residual sequence is
  persisted; source chunks and exact transcript provenance remain authoritative.
- **Ranked read**: bounded max-path traversal reserves association slots inside
  the existing `k`, protects strong lexical anchors, and can reject a result
  set that adds prompt tokens.
- **Heat read**: a finite restart walk row-normalizes stored edge utility,
  accumulates support from multiple parents, and caps the live frontier to
  chunk IDs, scalar heat, and one compact path. Text is hydrated only for the
  final candidates.
- **Selected policy**: one ranked-QK exploitation slot plus one heat exploration
  slot, two hops, at most 16 frontier entries, and degree-two physical pruning.
  This is a development policy pending a new locked or public evaluation.
- **Live Hebbian projection**: the final chunks retrieved together in one turn
  may be observed as an idempotent access event. Rank-discounted co-access mass
  forms symmetric chunk edges; node-mass normalization suppresses hubs and an
  independent turn-decay term cools stale links. Reads use reserved tail slots
  inside the existing `k` and default to zero prompt-token increase. Events are
  capped at 12 nodes, node degree at 32, and retry receipts at 4,096. Receipts
  retain only an event ID plus membership hash. Query text and transformer token
  state are never persisted. See
  [`02 - Live Hebbian Co-Retrieval Memory.md`](../00%20-%20Theory/02%20-%20Live%20Hebbian%20Co-Retrieval%20Memory.md).
- **Prompt-driven systems consolidation**: schema v9 adds a separate,
  model-independent graph whose typed nodes point into both `memory_items` and
  `chunks`. Completed interactions bind the stored prompt/prior anchors to every
  new response/tool chunk through fixed-size slices; `causal_count` distinguishes
  that evidence from ordinary repeated co-access. Reads add candidates without
  evicting direct evidence, may diffuse through two bounded scalar hops, and
  rerank the frontier against the live query before the unchanged hard token
  cap. Associations decay in turn-space and are degree-pruned. Graph-admitted
  results cannot reinforce themselves.
- **Qwen-weighted consolidation**: `observe_context_access` accepts transient
  CAV-derived node activity plus bounded QK/OV pair affinities. The prefix
  workspace is discarded; schema v9 persists only IDs, scalar masses/counts,
  turn coordinates, and idempotency hashes. Rank-discounted activity remains
  the provider-free fallback. See
  [`03 - Prompt-Driven Systems Consolidation.md`](../00%20-%20Theory/03%20-%20Prompt-Driven%20Systems%20Consolidation.md).
- **Episodic discourse closure**: schema v11 adds immutable annotation
  artifacts, source-local episodes and representatives, typed discourse units,
  evidenced n-ary relations, finalized whole-corpus coverage, and content-bound
  source/graph snapshots. Exact raw spans are revalidated against chunks and
  turns, and coverage cannot finalize if any non-whitespace turn content is
  absent from the chunk layer. An opt-in read path maps caller-supplied
  hybrid hits to episodes, expands bounded temporal neighbors, closes explicit
  query obligations over strong source-grounded relations, and atomically
  packs evidence under context and full-prompt proxy caps. EM-style surprise
  is optional; fixed and lexical/embedding-change boundaries work without a
  model. Built-in paths retain zero request-token state; injected strategies
  require their own certification. See
  [`05 - EM-LLM Episodic Discourse Closure for Diffuse Retrieval.md`](../00%20-%20Theory/05%20-%20EM-LLM%20Episodic%20Discourse%20Closure%20for%20Diffuse%20Retrieval.md).

### Causal transition policy — `transition_policy.py`

- **Prediction order**: rank a bounded QK/OV candidate set using only state
  through turn `t`; reveal turn `t+1` only afterward.
- **Learning signal**: next-source correctness plus optional alignment between
  projected per-head OV CAV deltas and the observed CAV change.
- **Evolving attention**: separate user→assistant and assistant→user decayed
  head utilities become multiplicative gates for the following QK decision;
  sparse recurrent edge utility can provide a second bounded score.
- **Durable state**: only scalar reward sums, decayed mass, counts, role/head
  IDs, and source/destination IDs. The one-turn decision may carry compact CAV
  deltas in memory, but snapshots exclude them and all request-derived
  transformer token state. Reusable static weights/tokenizers are linker
  assets, not request state, and are outside that metric.
- **Admission rule**: this policy is implemented but not yet admitted to QA
  retrieval or pruning. Exact-target and 2-D CAV/velocity chronological replays
  failed to transfer to a separate compiled store, so they remain diagnostics.
- **Local transition arm**: `search_hybrid_neighbors` tests bounded source-local
  previous/next moves. Radius and extra slots are hard-capped; transition
  candidates may either be appended or compete with weak anchors. Development
  measurements reject both unconditional append and five-way replacement as
  production defaults, so `stay` remains mandatory in any learned controller.
- **Graph-union arm**: `search_hybrid_graph` preserves the direct anchors,
  admits a bounded directional transition shell, then fills remaining
  candidates from a bounded source-conditioned rerank. Only chunk/source IDs
  and scalar scores exist during the walk; the final evaluator prompt cap is
  still authoritative.
- **Partition-local read**: `source_local_search=True` uses the global hybrid
  prefix only to activate source IDs, then streams dense and lexical candidates
  inside that eligible union. Scores are calibrated once across the union;
  independent per-source normalization is forbidden because it manufactures a
  score-1 candidate in every partition.
- **Transient Qwen candidate arm**: an optional `QwenCandidateReranker` protects
  the strong scalar prefix and reserves a fixed number of source-local slots
  for a recursive QK+OV tournament. Each forward sees at most eight candidates
  and a hard token cap. Candidate text and Q/K/V/residual tensors die after the
  pass; only normal `RetrievalResult` rows and scalar workspace diagnostics
  cross the boundary. This is a measured experimental read arm, not a durable
  transformer cache or a production default.
- **Recursive combined-activation arm**: `qwen_feedback` first lets the
  original-question activation select a stratified subset of recalled
  evidence. It then re-encodes `question + selected evidence` as one bounded
  activation window and uses that combined QK/OV state to search a fresh pool
  of lower-ranked candidates from the selected source partitions. BGE supplies
  only the broad second-stage pool; it never receives a fabricated projection
  of the Qwen activation. The original anchors and most source candidates are
  protected, while the recursive hop competes for a fixed reserve.

### MemoryStore — `memory_store.py`
- **Domain**: the typed long-term memory state machine. **Inputs**: validated `MemoryOps`. **Outputs**: `MemoryItem` / `MemoryResult`.
- **Structure**: `create · get · list_items · count · update · supersede · delete · pin · apply · touch · heat_counts · retrieve`, over `memory_items` + `memory_provenance`.
- **Hard constraints**:
  1. **Nothing is ever hard-deleted.** `delete` flips status to `deleted`; `supersede` creates the replacement (with `supersedes` pointing back) and flips the old row to `superseded`. Both rows survive, so the correction chain stays walkable.
  2. **Decay is lazy** — no timer, no background job. Energy is decayed forward from `last_access_turn` to the transcript's current ordinal on read; `last_access_at` is audit-only.
  3. **Retrieval is brute-force exact cosine with numpy**, deliberately **not** a second ANN index: memory items number in the tens-to-low-hundreds, so an exact scan is faster and simpler than maintaining a second hnswlib graph, and it can never return a stale neighbour after a supersede. Cosine is mapped `(cos + 1) / 2` into `[0, 1]` so it composes with the other rank components.
  4. Every item returned by `retrieve` is reheated, so returned items reflect post-reheat energy while their `score` reflects query-time state. The rows are updated with one batched SQLite transaction, and provenance is hydrated for the final top-k with one query rather than once per candidate.

### Decay — `decay.py`
- **Domain**: how hot an item is *now*. Pure functions, no I/O.
- **Structure**: `effective_energy = energy × 0.5^(turns_elapsed / half_life_turns)`, clamped to `[0, 1]`; default half-life 30 turns. `heat_for`: HOT ≥ `0.75`, WARM ≥ `0.25`, else COLD. `reheat` closes `0.25` of remaining headroom and can boost only once per turn. `seed_energy`: `0.8` when `importance ≥ 0.7`, else `0.5` — important items enter HOT, everything else WARM.
- **Hard constraint**: **pins override decay entirely.** A pinned item returns its stored energy regardless of elapsed turns (`MemoryStore.touch` still refreshes the wall-clock audit stamp).

### Ranking — `ranking.py`
- **Domain**: the one place the design's rerank scalar lives. Pure functions.
- **Structure**: `score = wR·relevance + wI·importance + wP·pin_boost + wE·energy − wS·superseded_penalty`. `RankWeights` defaults: relevance `1.0`, importance `0.3`, pin `0.5`, energy `0.2`, superseded penalty `1.0`. Pin boost: user-pinned `1.0`, system-pinned `0.6`, none `0.0`. Also `blend_hybrid`, `min_max_normalize` (flat input → all-`1.0`, i.e. "no signal"), `top_k`.
- **The `wE·energy` term was `wT·recency` until 2026-08-14**, computed here from a second copy of the exponential in `decay.py`. They were never independent — `effective_energy ≡ energy × recency_score` — and had drifted to opposite semantics for a non-positive half-life. Worse, since `touch` restamps `last_access_at` on every retrieve, `recency` was `1.0` for every item ever recalled, so the term discriminated nothing and decay influenced nothing. `ranking` now holds no decay arithmetic at all; `decay.decay_factor` is the one kernel.
- **Hard constraint**: both the memory store and the hybrid retriever score through this module, so weighting is never forked.

### Extractor — `extractor.py`
- **Domain**: proposing candidate memory. **Outputs**: `MemoryOps` — *proposed*, never trusted.
- **`RuleBasedExtractor`** (the default in `MemoryCondenser`) — ordered regex cue table over sentence splits; first match wins, checked Correction → Decision → Constraint → Preference → Definition → Task. Corrections/decisions/constraints get importance `0.8`, the rest `0.5`. Zero LLM calls, fully offline, fully deterministic. The provenance quote **is** the matched sentence, verbatim.
- **`LLMExtractor`** — strict-JSON `memory_ops` through an **injected** `complete(system_prompt, user_prompt) -> str` callable. This module imports no LLM SDK. Failure policy: never raise, never invent — transport error, unparsable JSON, or schema mismatch all yield an empty `MemoryOps`. A dropped memory is recoverable next turn; a fabricated one is not.

### Validator — `validator.py`
- **Domain**: the provenance gate between any extractor and the store. **Inputs**: `MemoryOps`. **Outputs**: `ValidationReport` (`accepted` ops + explained `rejected` list).
- **Hard constraints**: never raises, never mutates state. A quote is accepted only when the whitespace-normalized quote is a substring of the whitespace-normalized turn text. Nothing else is relaxed — **no case folding, no punctuation stripping, no fuzzy match. An LLM that paraphrases gets rejected; that is the point.** Rejection reasons are stable slugs: `missing_provenance`, `unknown_turn`, `quote_not_found`, `unknown_mem_id`, `empty_content`.

### ContextPacker — `context_packer.py`
- **Domain**: making context cost *predictable*. **Inputs**: system prompt, memories, recent turns, expansions, user text. **Outputs**: `PackedContext`.
- **Structure**: `ContextBudget` = recent window `4500` tok · memory header `900` tok · expansions `800` tok, at most `3` expansions of `≤250` tok each. Section order: system → memory header (typed bullets, active only, pinned marked `*`) → recent turns (chronological, most-recent-first fitting) → expansions (verbatim excerpts) → current user message.
- **Heat-aware option**: expansion candidates can be scheduled by source heat
  per token with a per-source expansion fraction. The packer reports actual
  expansion tokens by source in `expansion_source_token_counts`, making "heat
  equals amount of source text seen" observable rather than metaphorical.
- **Information-rate option**: after query-aware sentence extraction, a
  deterministic conditional information-bottleneck proxy estimates candidate
  IDF surprise, lexical/dense relevance, concept/source novelty, and temporal
  novelty per rendered token. It is a monotone filter over retrieval order,
  never a global softmax or reranker, so weak candidates cannot displace strong
  anchors. Enumeration, ordering, comparison, and all/each queries lower the
  rejection threshold because superficially repetitive excerpts may encode
  distinct required members. The filter stores no learned state.
- **Hard constraint**: every section has an independent ceiling, and **anything that does not fit is dropped and counted** in `PackedContext.dropped` (`memories` / `recent_turns` / `expansions`) — never silently truncated without a record. Token accounting per section lands in `token_counts`.

### MemoryCondenser — `condenser.py`
- **Domain**: the facade that wires all of the above. **Hard constraint**: makes no LLM call itself.
- `ingest(role, text, source_id=None)` → store source-identified turn → chunk → embed → dense+lexical index → (when `auto_extract=True`, the default) extract → validate → apply.
- `search` (dense) · `search_hybrid` (dense+BM25) ·
  `search_hybrid_neighbors` (bounded source-local transitions) ·
  `search_hybrid_sources` (source-conditioned second stage) ·
  `search_hybrid_graph` (directional transition/source union) · `search_associative`
  (ranked compiled links) · `expand_hebbian` / `observe_retrieval_access`
  (bounded live co-access over pre-ranked anchors; the one-call
  `search_heat_associative` / `search_hebbian` wrappers were deleted 2026-08-19
  as caller-less — heat diffusion stays reachable via
  `expand_heat_diffusion_results`)
  · `recall_memories` (ranked `MemoryResult`, reheats) · `build_context`
  (packed prompt; `hybrid=True` by default; live consolidation read/write) ·
  `observe_context_access` (optional CAV/QK/OV-weighted update) · `heat_counts`
  · properties `transcript` / `memory` / `retriever` / `associations` /
  `consolidation` / `validator`.
- Constructor params of note: `extractor`, `budget`, `auto_extract`, and **`embedder`** — injectable so tests substitute a fake and never download bge-m3.

### Loader — `loader.py`
- **Domain**: corpus ingestion, two families.
- Claude exports: `.txt` (`User:` / `Claude:`) and `.md` (`**User:**` / `**Assistant:**`) — unchanged.
- Public benchmarks: `BenchmarkQuestion`, `BenchmarkSample`, `parse_longmemeval`, `parse_locomo`, `detect_benchmark_format`, `load_benchmark(path, format="auto")`. Accepts `.json` and `.jsonl`; malformed records are skipped rather than failing the file. Session/document IDs remain parallel to flattened turns, source timestamps are ingested as system turns, and LongMemEval's question date prefixes its query. LoCoMo has no intrinsic user/assistant split, so the **first speaker seen in the earliest session maps to `user`** and every other speaker to `assistant`; sessions are ordered by numeric suffix (`session_10` follows `session_9`).

### Eval harness — `eval/`
- **Domain**: measurement. Two protocols plus an offline analysis mode.
- **Self-replay** (`runner.py`) — teacher-forced: after scoring, ingest the *actual* recorded turns, never the generated ones. Metrics: mean judge score, Recall@4.
- **Benchmark QA probes** (`benchmark.py`) — ingest a sample's whole haystack, enforce a hard full-prompt ceiling, answer each question from dense/hybrid/span/source/memory context, and grade with SQuAD-normalized token F1 + exact match plus an optional LLM `judge_fn` for semantic equivalence. `answer_fn` / `judge_fn` are **injected**; the module imports no litellm.
- **Free retrieval diagnostics** (`recall.py`) — report gold-string reachability and tokens together, plus any/all/fractional gold evidence-source coverage when the benchmark supplies source labels.
- **Evidence sufficiency** (`sufficiency.py`) — compares the retrieved prompt
  with a same-budget gold-source/session oracle. LongMemEval does not provide
  exact evidence-turn labels, so the report states its granularity explicitly.
  Literal, inference-required, and source-coverage diagnostics are local; an
  injected semantic judge can separately label oracle and retrieved
  sufficiency when calls are explicitly authorized.
- **Compiled benchmark cache** (`compiled_cache.py`) — content-addressed,
  per-sample SQLite/HNSW artifacts keyed by all write-time inputs. Manifests
  hold exact file hashes and are published last; verified reads cannot rewrite
  the ANN file.
- **Transition trace** (`transition_trace.py`) — batch-encodes queries once and
  exports a self-hashed, provenance-complete direct/previous/next candidate
  plane for model-free policy recomposition. Gold-bearing traces are tuning
  artifacts and must respect the locked split protocol.
- **Analysis** (`analysis.py`) — `binned_scores`, `compare_runs`, `ascii_curve`, `to_csv`, `print_comparison`. Pure functions over saved JSON; no API calls, no cost.
- **Instrumentation** (`validated`): `UsageStats` (input / output / cache-read tokens, `elapsed_s`, `calls`) flows through `TurnResult.responder_usage` + `.judge_usage` + `.retrieval_s` + `.context_tokens` → `ConversationResult.usage` → `EvalRunResult.usage` / `.total_elapsed_s` / `.mean_context_tokens` / `.tokens_per_scored_turn`.
- **Retrieval mode is configurable**: `RetrievalConfig.hybrid` (with `alpha`, `candidates`) switches `runner.replay_conversation` between `mc.search` and `mc.search_hybrid`; dense stays the default so the k=0/k=N ablation keeps measuring the same baseline. Hybrid runs get a distinct result filename so they cannot overwrite the dense run being compared against.
- Both eval paths construct `MemoryCondenser` with `auto_extract=False`: their prompts are built from retrieved chunks, not memory items, so extraction would be cost without effect.
- **known rough edge**: the replay eval still prompts with raw `[Memory i]` chunks rather than `build_context`, so the memory layer — header, typed bullets, expansions — is not exercised by any measurement. Deciding whether replay should use `build_context` is an open task (`06 - Roadmaps`).

## Design axioms (non-negotiable)

1. **Local/API split.** Chunking, embedding, both indexes, the memory state machine, and context packing live locally; the API is generation-only. **No core module imports an LLM SDK at module scope.** `llm_provider.py` is the single seam that binds one, and it does `import litellm` *inside* the function that needs it, so `import memory_condense` still costs nothing and needs no credentials. Everything else takes **injected callables** — `extractor.LLMExtractor(complete=…)`, `benchmark.run_benchmark(answer_fn=…, judge_fn=…)` — so the LLM paths exist without a provider dependency reaching the core.

   This was previously stated as "validated by grep", which is validated exactly once, on the day someone runs it. It is now `tests/test_architecture.py`, checked over the AST (so a docstring mentioning litellm is not an offence) plus a subprocess assertion that importing the package pulls no SDK into `sys.modules`.
2. **Transcript is append-only**; all derived state (chunks, terms, embeddings, memory) must be reconstructible from it + config.
3. **Provenance over trust.** Any LLM-written memory must quote real turn spans. This is now enforced code, not intent: `Validator` is the only path into `MemoryStore.apply` used by `MemoryCondenser.extract_memory`.
4. **One tokenizer proxy**: cl100k_base (`_tokenizer.py`) for all budgeting, regardless of runtime LLM.
5. **Baseline and treatment share one code path** in eval (`--k 0` vs `--k N`), never a forked harness. This is why `retrieval.query()` was left byte-for-byte alone when `hybrid_query` was added.
6. **Nothing is destroyed.** Memory rows are soft-deleted or superseded, never removed; `delete_chunk` keeps the chunk row so provenance cannot dangle.

---

**Verification block**: run

```powershell
pixi run --frozen -e dev pytest -q -m "not slow"
pixi run python -c "from memory_condense.persistence.db import Database; import tempfile, pathlib; d=Database(pathlib.Path(tempfile.mkdtemp())/'v.db'); print('schema_version', d.schema_version)"
git log --oneline -1                            # expect merge f3edc91 on main
```

Expect `schema_version 11`. Canonical package ownership and import paths are
listed in [`03 - Code Package Layout.md`](03%20-%20Code%20Package%20Layout.md).
