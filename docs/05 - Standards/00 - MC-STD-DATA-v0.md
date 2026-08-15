**Standard ID:** MC-STD-DATA v0
**Title:** Data contracts — storage, embedding, chunking, memory, loader, eval output
**Status:** DRAFT (not yet frozen — freeze at first release or first external consumer, whichever comes first)
**Date:** 2026-08-14
**Supersedes:** the v1-schema-only revision of this standard (documented `turns` + `chunks` and nothing else)
**Applies to:** `src/memory_condense/` and `eval_results/` JSON
**Depends on:** • `03 - Architecture/00 - System Overview.md` (as-built map)

> **Branch-only**: the schema this standard describes (`schema_version` 2) landed in `80262ea` on `feat/memory-layer`. A reader on `main` will still find version 1.

## 0. Scope

Covers: the SQLite schema (v2, plus the v1→v2 migration), the embedding contract, chunk sizing, the lexical/BM25 index, the memory-item and provenance contract, loader input formats (Claude exports + LongMemEval/LoCoMo), and eval result JSON.

Does NOT cover: prompt formats (implementation detail), rank weights or `alpha` (tuning parameters, not contract), or the `MemoryOps` **wire** contract for external producers — that gets its own standard (MC-STD-MEMOPS) if and when a non-local producer exists.

## 1. Normative Goals

1. Derived state MUST be reconstructible from the `turns` table plus configuration. The hnswlib `.bin` file is a cache, never a source of truth. The BM25 inverted index is likewise derived (`LexicalIndex.rebuild()`).
2. Every chunk MUST carry provenance: `turn_id` + `start_char`/`end_char` spans into the turn text.
3. Writers MUST NOT mutate or delete `turns` rows (append-only).
4. Embeddings MUST be `float32`, produced by the configured sentence-transformers model, stored **unnormalized**. A consumer MUST NOT dot-product raw blobs without normalizing; cosine comparisons go through the hnswlib cosine space or explicit normalization.
5. Any change to embedding model or dimension MUST bump `schema_version` and rebuild the index. `EmbeddingService.dim` now reports the model's **true** dimension (falling back to the constant `1024` only for the unloaded default `BAAI/bge-m3`), so a model swap is no longer silently corrupting — but it is still a schema change and MUST be treated as one.
6. Chunk token counts MUST be measured with tiktoken `cl100k_base` and lie in `[min_tokens, max_tokens]` except single indivisible words.
7. Eval baseline and treatment runs MUST share the same code path (`k=0` vs `k>0`), never a separate harness. `retrieval.query()` (pure dense) MUST remain behaviourally unchanged for this reason; hybrid retrieval is an additive method (`hybrid_query`), not a modification.
8. **Every memory item MUST carry at least one provenance entry**, and every provenance `quote` MUST appear verbatim in the referenced turn's text, compared after whitespace normalization only (runs of whitespace collapsed to one space, ends stripped). No case folding, no punctuation stripping, no fuzzy matching. An op that cannot satisfy this MUST be rejected, not stored. *(Enforced by `validator.Validator`; the sole exception is `UpdateOp`, which MAY carry an empty provenance list because it amends an item that already has provenance — any entry it **does** carry is checked in full.)*
9. **Memory rows MUST NOT be destroyed.** `delete` MUST set `status = 'deleted'`; a correction MUST be expressed as a supersede — a new row whose `supersedes` names the old one, with the old row set to `status = 'superseded'`. Hard `DELETE` on `memory_items` is forbidden: the audit trail from a memory back to the transcript that justifies it MUST stay walkable. Removing a chunk from the indexes (`retrieval.delete_chunk`) MUST likewise keep the `chunks` row so provenance cannot dangle.
10. **A database file MUST be migrated in place, never recreated.** A file at any `schema_version` < `CURRENT_SCHEMA_VERSION` MUST be upgraded by applying each intervening migration in order; a fresh file is created directly at the current version. Migrations MUST be additive (new tables/columns) so that clause 3 holds across upgrades. Every version bump MUST ship its migration in `db._MIGRATIONS` in the same change.

## 2. Core concept

One SQLite database (WAL, `foreign_keys=ON`, `schema_version` 2) holds everything durable: the transcript, the chunks, the BM25 inverted index, and the memory items with their provenance. Two derived caches sit beside it — the hnswlib index file (`hnsw_index.bin`, cosine, `M=16`, `ef_construction=200`, `max_elements=100 000`, rebuildable via `rebuild_index()`) and the `chunk_terms` postings (rebuildable via `LexicalIndex.rebuild()`).

## 3. Storage schema (v2)

| Table | Columns (contract-relevant) | Notes |
| --- | --- | --- |
| `turns` | `turn_id` PK, `role` CHECK ∈ {user, assistant, system}, `text`, `created_at` | append-only |
| `chunks` | `chunk_id` PK, `turn_id` FK, `text`, `start_char`, `end_char`, `token_count`, `embedding` BLOB (dim×4 bytes f32), `lexical_weights` TEXT (JSON term→tf, **now populated**), `hnsw_label` INTEGER UNIQUE, `term_count` INTEGER | `hnsw_label` is the sole chunk↔dense-index mapping; `term_count` is the BM25 document length (NULL ⇒ not lexically indexed) |
| `chunk_terms` | `term`, `chunk_id` FK, `tf`, PK `(term, chunk_id)` | BM25 inverted index; postings are replaced wholesale per chunk, never appended to |
| `memory_items` | `mem_id` PK, `type`, `content`, `details`, `status` CHECK ∈ {active, superseded, deleted}, `supersedes`, `pin` CHECK ∈ {user_pinned, system_pinned, none}, `energy` REAL, `half_life_s` REAL (default 604800), `importance` REAL, `created_at`, `last_access_at`, `embedding` BLOB | see clause 9 — status transitions only, no deletes |
| `memory_provenance` | `mem_id` FK ON DELETE CASCADE, `turn_id`, `chunk_id` (nullable), `quote`, UNIQUE `(mem_id, turn_id, quote)` | see clause 8 — at least one row per item |
| `meta` | `key` PK, `value` | holds `schema_version` |

### 3.1 Migration path

| From | To | Applied changes |
| --- | --- | --- |
| (no file / no `meta` table) | 2 | full schema created directly at v2 |
| 1 | 2 | `ALTER TABLE chunks ADD COLUMN term_count`; create `chunk_terms`, `memory_items`, `memory_provenance` and their indexes; `UPDATE meta SET value = '2'` |

`Database.schema_version` reports the on-disk version (`0` when unreadable). Migrations run inside `Database.__init__`, so opening a v1 file upgrades it — no separate migration command exists, and none should be added without also making the upgrade opt-in.

### 3.2 Value contracts

| Quantity | Contract |
| --- | --- |
| `energy` | `[0, 1]`; decayed lazily on read as `energy × 0.5^(elapsed / half_life_s)`; pinned items exempt |
| heat tier | derived, never stored: HOT ≥ `0.75`, WARM ≥ `0.25`, else COLD |
| `importance` | `[0, 1]`; seeds energy at `0.8` when `≥ 0.7`, else `0.5` |
| memory `embedding` | same dtype/dimension contract as chunk embeddings (clause 4); items without one score `relevance = 0` rather than erroring |
| BM25 | Okapi with `k1 = 1.5`, `b = 0.75`; scores are **raw** and MUST be normalized before blending with dense scores |

## 4. Loader input formats

### 4.1 Claude conversation exports

| Format | Turn delimiter (regex, line-anchored) | Roles |
| --- | --- | --- |
| `.txt` | `^(User\|Claude):$` | User→user, Claude→assistant |
| `.md` | `^\*\*(User\|Assistant):\*\*$` | as named |

### 4.2 Public benchmarks

| Format | Detection signature | Sample shape |
| --- | --- | --- |
| `longmemeval` | `haystack_sessions` key (fallbacks: `haystack_dates`, `answer_session_ids`) | one record = one sample carrying exactly one question; `haystack_sessions` concatenated in the order given |
| `locomo` | `conversation` + `qa` keys (fallback: `conversation` containing `session_N` keys) | one record = one sample with many questions; sessions ordered by numeric suffix |

Normative loader rules:

1. `load_benchmark(path, format="auto")` MUST accept `.json` (one document) and `.jsonl`/`.ndjson` (one record per line).
2. Malformed records and malformed JSONL lines MUST be skipped, never raised — a partially readable benchmark file still yields usable samples.
3. Roles MUST normalize to `"user"` / `"assistant"`. LoCoMo dialogues have no intrinsic split, so the **first speaker seen in the earliest session** maps to `user` and every other speaker to `assistant`. Unknown/missing roles fall back to alternating by turn index (user first).
4. `detect_benchmark_format` MUST raise `ValueError` rather than guess when neither signature is present.

## 5. Eval result JSON (informative)

### 5.1 Self-replay run (`eval_*.json`)

Top level: `config` (chunker, retrieval, models, dirs, `recent_window`), `conversations[]` (per file: `mean_score`, `scores_by_position[]`, `usage`, per-turn records), and aggregates `aggregate_mean_score`, `aggregate_recall_at_4`, `usage`, `total_elapsed_s`, `mean_context_tokens`, `tokens_per_scored_turn`.

Per turn: `turn_index`, truncated `user_text` / `actual_response` / `generated_response` (500 chars), `retrieved_chunks[]` (top 5, 200 chars each), `score` 1–5, `judge_reasoning`, `responder_usage`, `judge_usage`, `retrieval_s`, `context_tokens`.

`UsageStats` fields: `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `elapsed_s`, `calls`. They add associatively (`__add__` / `__radd__`), so conversation and run totals are exact sums of turn-level values.

Filename: `eval_{min}-{max}_k{k}_ef{ef}_{YYYYMMDD_HHMMSS}.json`.

### 5.2 Benchmark run (`benchmark_*.json`)

Top level: `config`, `benchmark` (free-form label), `samples[]`, `num_samples`, `num_questions`, `mean_f1`, `exact_match_rate`, `judge_accuracy` (null unless `--use-judge`), `by_category{}`, `run_timestamp`.

Grading is SQuAD-style: lowercase, strip punctuation, remove articles, collapse whitespace; then token-level F1 and normalized exact match. Both-empty scores `1.0`; exactly-one-empty scores `0.0`.

Filename: `benchmark_{label}_{min}-{max}_k{k}_ef{ef}_{YYYYMMDD_HHMMSS}.json`. *(known rough edge: the CLI currently passes `--benchmark-format` as the label, so the default is literally `benchmark_auto_...`.)*

---

**Verification block**: run

```powershell
pixi run -e dev pytest -q -m "not slow" tests/test_db.py tests/test_memory_store.py tests/test_validator.py tests/test_lexical.py
pixi run python -c "import sqlite3, tempfile, pathlib; from memory_condense.db import Database; p=pathlib.Path(tempfile.mkdtemp())/'v.db'; d=Database(p); print(d.schema_version); print(sorted(r[0] for r in d.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")))"
```

Expect `2` and the table list `['chunk_terms', 'chunks', 'memory_items', 'memory_provenance', 'meta', 'turns']`. If a table is missing, the migration in `db._MIGRATIONS[2]` and `_SCHEMA_SQL` have drifted apart — fix that before writing any data.
