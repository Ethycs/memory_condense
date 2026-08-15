# memory_condense System Overview (as-built)

**Status**: CURRENT
**Date**: 2026-08-14
**Supersedes**: the earlier 2026-08-14 "dense-retrieval-only / no condensation yet" version of this document, which described the tree at commit `cd9f423`
**Applies to**: branch `feat/memory-layer` — `80262ea` (memory layer), `abdcc9b` (eval harness), `24c7323` (docs), on top of `cd9f423`. Committed, **not merged to `main`**.

## Executive summary

memory_condense is now the full local memory manager the design called for: transcript → chunker → bge-m3 dense index **and** a BM25 inverted index, a typed `MemoryItem` store with provenance-enforced extraction, exponential energy decay with HOT/WARM/COLD tiering, a deterministic rerank scalar, and a hard-budgeted `ContextPacker`. Everything stateful is local (SQLite schema v2 + bge-m3 + hnswlib + BM25); the only LLM callers in the repo are still under `eval/`, and the core stays provider-agnostic by taking an injected completion callable instead of importing an SDK. What is **not** built: cold-tier "era summaries" (design Phase 4), and — separately — nothing here has yet been *measured* on a public benchmark.

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
        │                 │                    │  decay.py (½-life 7d)    │
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
│ SQLite — db.py: WAL, foreign_keys=ON, schema_version 2 (v1 migrates up)   │
│   turns · chunks(text, embedding, lexical_weights, hnsw_label, term_count)│
│   chunk_terms(term, chunk_id, tf)          ← BM25 inverted index          │
│   memory_items(...) · memory_provenance(...) ← mandatory provenance rows  │
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
- **Domain**: durable raw history. **Runs at**: every `ingest`. **Inputs**: `(role, text)`. **Outputs**: `Turn` rows.
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
- **`add_chunks`** writes the dense index, persists the embedding, **populates `chunks.lexical_weights`** with the chunk's term-frequency map, and feeds the same chunks to the BM25 index. The column that was always NULL is now real.
- **`delete_chunk(chunk_id)`** — clears embedding + `hnsw_label` in SQLite (authoritative), marks the label deleted in the live hnswlib graph best-effort, and drops the BM25 postings. The chunk **row** survives so memory provenance pointing at it cannot dangle.
- **Structure**: hnswlib `space="cosine"`, `M=16`, `ef_construction=200`, `max_elements=100_000`; chunk↔label mapping lives in `chunks.hnsw_label` (single source of truth); `rebuild_index()` reconstructs the `.bin` from SQLite blobs.

### MemoryStore — `memory_store.py`
- **Domain**: the typed long-term memory state machine. **Inputs**: validated `MemoryOps`. **Outputs**: `MemoryItem` / `MemoryResult`.
- **Structure**: `create · get · list_items · count · update · supersede · delete · pin · apply · touch · heat_counts · retrieve`, over `memory_items` + `memory_provenance`.
- **Hard constraints**:
  1. **Nothing is ever hard-deleted.** `delete` flips status to `deleted`; `supersede` creates the replacement (with `supersedes` pointing back) and flips the old row to `superseded`. Both rows survive, so the correction chain stays walkable.
  2. **Decay is lazy** — no timer, no background job. Energy is decayed forward from `last_access_at` on read.
  3. **Retrieval is brute-force exact cosine with numpy**, deliberately **not** a second ANN index: memory items number in the tens-to-low-hundreds, so an exact scan is faster and simpler than maintaining a second hnswlib graph, and it can never return a stale neighbour after a supersede. Cosine is mapped `(cos + 1) / 2` into `[0, 1]` so it composes with the other rank components.
  4. Every item returned by `retrieve` is `touch`ed (access reheating), so returned items reflect post-reheat energy while their `score` reflects query-time state.

### Decay — `decay.py`
- **Domain**: how hot an item is *now*. Pure functions, no I/O.
- **Structure**: `effective_energy = energy × 0.5^(elapsed / half_life_s)`, clamped to `[0, 1]`; default half-life 7 days (`604800 s`). `heat_for`: HOT ≥ `0.75`, WARM ≥ `0.25`, else COLD. `reheat` adds `+0.25` capped at 1.0. `seed_energy`: `0.8` when `importance ≥ 0.7`, else `0.5` — important items enter HOT, everything else WARM.
- **Hard constraint**: **pins override decay entirely.** A pinned item returns its stored energy regardless of elapsed time (`MemoryStore.touch` still refreshes `last_access_at` so recency stays honest).

### Ranking — `ranking.py`
- **Domain**: the one place the design's rerank scalar lives. Pure functions.
- **Structure**: `score = wR·relevance + wI·importance + wP·pin_boost + wT·recency − wS·superseded_penalty`. `RankWeights` defaults: relevance `1.0`, importance `0.3`, pin `0.5`, recency `0.2`, superseded penalty `1.0`. Pin boost: user-pinned `1.0`, system-pinned `0.6`, none `0.0`. Also `recency_score` (half-life decay to `[0,1]`), `blend_hybrid`, `min_max_normalize` (flat input → all-`1.0`, i.e. "no signal"), `top_k`.
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
- **Hard constraint**: every section has an independent ceiling, and **anything that does not fit is dropped and counted** in `PackedContext.dropped` (`memories` / `recent_turns` / `expansions`) — never silently truncated without a record. Token accounting per section lands in `token_counts`.

### MemoryCondenser — `condenser.py`
- **Domain**: the facade that wires all of the above. **Hard constraint**: makes no LLM call itself.
- `ingest(role, text)` → store turn → chunk → embed → dense+lexical index → (when `auto_extract=True`, the default) extract → validate → apply.
- `search` (dense) · `search_hybrid` (dense+BM25) · `recall_memories` (ranked `MemoryResult`, reheats) · `build_context` (packed prompt; `hybrid=True` by default) · `heat_counts` · properties `transcript` / `memory` / `retriever` / `validator`.
- Constructor params of note: `extractor`, `budget`, `auto_extract`, and **`embedder`** — injectable so tests substitute a fake and never download bge-m3.

### Loader — `loader.py`
- **Domain**: corpus ingestion, two families.
- Claude exports: `.txt` (`User:` / `Claude:`) and `.md` (`**User:**` / `**Assistant:**`) — unchanged.
- Public benchmarks: `BenchmarkQuestion`, `BenchmarkSample`, `parse_longmemeval`, `parse_locomo`, `detect_benchmark_format`, `load_benchmark(path, format="auto")`. Accepts `.json` and `.jsonl`; malformed records are skipped rather than failing the file. LoCoMo has no intrinsic user/assistant split, so the **first speaker seen in the earliest session maps to `user`** and every other speaker to `assistant`; sessions are ordered by numeric suffix (`session_10` follows `session_9`).

### Eval harness — `eval/`
- **Domain**: measurement. Two protocols plus an offline analysis mode.
- **Self-replay** (`runner.py`) — teacher-forced: after scoring, ingest the *actual* recorded turns, never the generated ones. Metrics: mean judge score, Recall@4.
- **Benchmark QA probes** (`benchmark.py`) — ingest a sample's whole haystack, answer each question from top-k retrieved chunks only, grade with SQuAD-normalized token F1 + exact match, optional LLM `judge_fn` for semantic equivalence, per-category breakdown. `answer_fn` / `judge_fn` are **injected**; the module imports no litellm.
- **Analysis** (`analysis.py`) — `binned_scores`, `compare_runs`, `ascii_curve`, `to_csv`, `print_comparison`. Pure functions over saved JSON; no API calls, no cost.
- **Instrumentation** (`validated`): `UsageStats` (input / output / cache-read tokens, `elapsed_s`, `calls`) flows through `TurnResult.responder_usage` + `.judge_usage` + `.retrieval_s` + `.context_tokens` → `ConversationResult.usage` → `EvalRunResult.usage` / `.total_elapsed_s` / `.mean_context_tokens` / `.tokens_per_scored_turn`.
- **Retrieval mode is configurable**: `RetrievalConfig.hybrid` (with `alpha`, `candidates`) switches `runner.replay_conversation` between `mc.search` and `mc.search_hybrid`; dense stays the default so the k=0/k=N ablation keeps measuring the same baseline. Hybrid runs get a distinct result filename so they cannot overwrite the dense run being compared against.
- Both eval paths construct `MemoryCondenser` with `auto_extract=False`: their prompts are built from retrieved chunks, not memory items, so extraction would be cost without effect.
- **known rough edge**: the replay eval still prompts with raw `[Memory i]` chunks rather than `build_context`, so the memory layer — header, typed bullets, expansions — is not exercised by any measurement. Deciding whether replay should use `build_context` is an open task (`06 - Roadmaps`).

## Design axioms (non-negotiable)

1. **Local/API split.** Chunking, embedding, both indexes, the memory state machine, and context packing live locally; the API is generation-only. `import litellm` appears in exactly three files, all under `eval/` (`judge.py`, `responder.py`, `__main__.py`) — `validated` by grep. The core honours this even where an LLM is *useful*: `extractor.LLMExtractor` and `benchmark.run_benchmark` take **injected callables** (`complete`, `answer_fn`, `judge_fn`) instead of importing a provider, so the Phase-2 LLM path exists without a provider dependency reaching the core package.
2. **Transcript is append-only**; all derived state (chunks, terms, embeddings, memory) must be reconstructible from it + config.
3. **Provenance over trust.** Any LLM-written memory must quote real turn spans. This is now enforced code, not intent: `Validator` is the only path into `MemoryStore.apply` used by `MemoryCondenser.extract_memory`.
4. **One tokenizer proxy**: cl100k_base (`_tokenizer.py`) for all budgeting, regardless of runtime LLM.
5. **Baseline and treatment share one code path** in eval (`--k 0` vs `--k N`), never a forked harness. This is why `retrieval.query()` was left byte-for-byte alone when `hybrid_query` was added.
6. **Nothing is destroyed.** Memory rows are soft-deleted or superseded, never removed; `delete_chunk` keeps the chunk row so provenance cannot dangle.

---

**Verification block**: run

```powershell
pixi run -e dev pytest -q -m "not slow"        # expect 366 passed, 13 deselected
pixi run python -c "from memory_condense.db import Database; import tempfile, pathlib; d=Database(pathlib.Path(tempfile.mkdtemp())/'v.db'); print('schema_version', d.schema_version)"
git log --oneline main..HEAD                    # expect 3 commits, unmerged
```

Expect `schema_version 2`. Then decide: commit the memory layer as one change, or split the eval-model fix (`eval/schemas.py`, `judge.py`, `responder.py`) into its own commit first since it is the only part that unblocks *running* anything.
