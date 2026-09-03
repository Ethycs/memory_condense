**Standard ID:** MC-STD-DATA v0
**Title:** Data contracts — storage, embedding, chunking, memory, loader, eval output
**Status:** DRAFT (not yet frozen — freeze at first release or first external consumer, whichever comes first)
**Date:** 2026-09-01
**Supersedes:** the v1-schema-only revision of this standard (documented `turns` + `chunks` and nothing else)
**Applies to:** `src/memory_condense/` and `eval_results/` JSON
**Depends on:** • `03 - Architecture/00 - System Overview.md` (as-built map)

> **Current worktree**: `schema_version` 13. v4 moves lifecycle decay to conversation-turn coordinates; v5 adds compact CAV/QK/OV artifacts; v6 adds source/session identity; v7 adds chunk Hebbian co-access; v8/v9 add live consolidation and causal counts; v10/v11 add source-grounded discourse plus content-bound revision receipts; v12 adds many-to-one memory-successor redirects; and v13 adds durable pending-to-indexed ingest manifests plus normalized globally unique chunk reservations. Stores at any earlier version migrate in place on open — see clause 10.

## 0. Scope

Covers: the SQLite schema and migrations through v13, the embedding contract, chunk sizing and crash-replay receipts, dense and lexical indexes, memory-item and provenance contracts, compact association/consolidation/discourse artifacts, loader input formats (Claude exports + LongMemEval/LoCoMo), and eval result JSON.

Does NOT cover: prompt formats (implementation detail), rank weights or `alpha` (tuning parameters, not contract), or the `MemoryOps` **wire** contract for external producers — that gets its own standard (MC-STD-MEMOPS) if and when a non-local producer exists.

## 1. Normative Goals

1. Derived state MUST be reconstructible from the `turns` table plus sealed configuration/receipts. For v13 ingestion, `pending_ingests` and `ingest_chunk_reservations` preserve the exact chunk topology and globally unique ownership when ambient chunker settings later change. The hnswlib `.bin` file is a cache, never a source of truth; a corrupt image MUST rebuild from SQLite. A partial ANN add or synchronization and an ambiguous or interrupted native retirement MUST discard the process-local graph for reconstruction from SQLite. The BM25 inverted index is likewise derived (`LexicalIndex.rebuild()`).
2. Every chunk MUST carry provenance: `turn_id` + `start_char`/`end_char` spans into the turn text.
3. Writers MUST NOT mutate or delete `turns` rows (append-only).
4. Embeddings MUST be `float32`, produced by the configured sentence-transformers model, stored **unnormalized**. A consumer MUST NOT dot-product raw blobs without normalizing; cosine comparisons go through the hnswlib cosine space or explicit normalization.
5. Any change to embedding model or dimension MUST bump `schema_version` and rebuild the index. `EmbeddingService.dim` now reports the model's **true** dimension (falling back to the constant `1024` only for the unloaded default `BAAI/bge-m3`), so a model swap is no longer silently corrupting — but it is still a schema change and MUST be treated as one.
6. Chunk token counts MUST be measured with tiktoken `cl100k_base` and lie in `[min_tokens, max_tokens]` except single indivisible words.
7. Eval baseline and treatment runs MUST share the same code path (`k=0` vs `k>0`), never a separate harness. `retrieval.query()` (pure dense) MUST remain behaviourally unchanged for this reason; hybrid retrieval is an additive method (`hybrid_query`), not a modification.
8. **Every memory item MUST carry at least one provenance entry**, and every provenance `quote` MUST appear verbatim in the referenced turn's text, compared after whitespace normalization only (runs of whitespace collapsed to one space, ends stripped). No case folding, no punctuation stripping, no fuzzy matching. An op that cannot satisfy this MUST be rejected, not stored. *(Enforced by `validator.Validator`; the sole exception is `UpdateOp`, which MAY carry an empty provenance list because it amends an item that already has provenance — any entry it **does** carry is checked in full.)*
9. **Memory rows MUST NOT be destroyed.** `delete` MUST set `status = 'deleted'`; a correction MUST be expressed as a supersede — a new row whose `supersedes` names the old one, with the old row set to `status = 'superseded'`. Hard `DELETE` on `memory_items` is forbidden: the audit trail from a memory back to the transcript that justifies it MUST stay walkable. Removing a chunk from the indexes (`retrieval.delete_chunk`) MUST likewise keep the `chunks` row so provenance cannot dangle.
10. **A database file MUST be migrated in place, never recreated.** A file at any `schema_version` < `CURRENT_SCHEMA_VERSION` MUST be upgraded by applying each intervening migration in order; a fresh file is created directly at the current version. Migrations MUST be additive (new tables/columns) so that clause 3 holds across upgrades. Every version bump MUST ship its migration in `db._MIGRATIONS` in the same change.
11. **Active memory items MUST be unique on content identity.** Identity is `(type, content)` after collapsing whitespace runs and case folding — `schemas.content_key`, stored as `memory_items.content_hash`. A create whose identity matches an existing **active** item MUST merge into that item — adding its provenance and refreshing its energy — rather than inserting a second row. Two consequences are deliberate: identity is scoped to `active`, so forgetting a fact and stating it again recreates it; and the type is part of the key, so the same sentence recorded as a `Constraint` and as a `Decision` remains two claims. **Near-duplicate collapsing by embedding similarity is forbidden** — "the beta ships on Friday" and "the beta ships on Monday" are highly similar and contradictory, and merging them would destroy the distinction clause 9 exists to preserve. Semantic conflict is expressed by supersede, never by dedup.

    Note the normalization here differs from clause 8's on purpose. Clause 8 decides whether a quote is genuine *evidence*, where a change of case changes the evidence, so it MUST NOT case-fold. This clause decides whether two memories are the *same memory*. The two MUST NOT be unified.
12. **Request-derived transformer token state MUST NOT be durable memory.** A head-inspection pass MAY materialize token IDs, Q/K/V, attention maps, head outputs, residual streams, or generation K/V inside a hard-bounded workspace. None may cross a pass boundary or be written to the durable store. The invariant and `retained_request_token_state_bytes` metric concern this request-derived state; reusable static checkpoint weights and tokenizer assets are explicitly outside the metric and are not memories. The only durable head-derived records are fixed-width `float32` CAV coordinates, fixed-width per-head edge weights, scalar QK/OV evidence, chunk IDs, artifact identity, and lifecycle counters. Every retrieval call MUST separately cap hydrated chunks; graph traversal is not permission to grow model context.
13. **Source identity MUST survive chunking.** When an ingest caller supplies a session/document `source_id`, every chunk derived from that turn MUST remain traceable to it through `chunks.turn_id -> turns.source_id`. Source-aware retrieval and packing MUST fall back to `turn_id` only for legacy turns whose source is NULL. A source ID groups provenance; it does not authorize source-wide prompt expansion beyond the hard token budget.
14. **Live consolidation MUST learn only from bounded durable references that
    actually reached a model context.** Its graph MAY connect active
    `memory_items` and retrievable `chunks`, but MUST store no source text,
    prompt, request-derived token state, embedding, attention matrix, residual,
    or K/V cache.
    Graph-selected candidates MUST NOT reinforce the edge that selected them;
    they must later recur through independent retrieval. Reads MUST have a
    recurrence threshold and fixed result slots. Node/edge statistics MUST
    decay in conversation-turn space and graph degree plus receipt history MUST
    be bounded. CAV/QK/OV signals MAY weight a scalar update but MUST NOT bypass
    these constraints or become factual authority.
15. **Turn-to-index publication MUST be replayable and topology sealed.** A
    newly ingested turn and its canonical chunk manifest MUST publish in one
    transaction. The manifest MUST name the complete ordered set of chunk IDs,
    source spans, token counts, and text hashes without copying source text or
    embeddings. Each manifest member MUST also have one normalized reservation
    whose `chunk_id` is globally unique and whose owner, span, token count, and
    text hash exactly equal that manifest member. Its status may advance only
    `pending -> indexed`, and only in the transaction that proves every expected
    chunk has an embedding, HNSW
    label, and lexical document length and that no unexpected chunk belongs to
    the turn. A failed or interrupted index step MUST leave the append-only turn
    plus a replayable pending receipt; it MUST NOT compensate by deleting the
    turn. An embedder result MUST be a one-to-one derivative of a pre-call
    deep snapshot of the staged chunks; a provider MUST receive separate deep
    copies, so no nested mutation can alter the validation baseline. No missing,
    extra, duplicate, unembedded, or source-field replacement is admissible.
    Completed receipts and reservations MUST remain durable so a
    retry under different chunker settings fails closed rather than replacing
    history. Exact manifest-member insertion, receipt-identity and reservation
    immutability, receipt/reservation durability, and complete monotonic
    transition MUST be enforced by SQLite triggers with
    `recursive_triggers=ON` on every connection.
    Supported direct dense or lexical completion MUST be restricted to pending
    members. Once a receipt is indexed, a missing or incomplete member is
    terminal retired state and MUST NOT be reactivated. The sole lexical repair
    exception is no-argument `LexicalIndex.rebuild()`, which snapshots its live
    batch from authoritative SQLite before clearing postings; a caller-supplied
    iterable remains a direct write and MUST reject terminal retired members.

## 2. Core concept

One SQLite database (WAL, `foreign_keys=ON`, `recursive_triggers=ON`, `schema_version` 13) holds everything durable: source-identified turns, exact chunk-ingest receipts and globally unique reservations, chunks/BM25 postings, memories and successor/provenance edges, compact association state, live consolidation, and immutable source-grounded discourse receipts. The hnswlib file beside it is a derived cache (`hnsw_index.bin`, cosine, `M=16`, `ef_construction=200`, `max_elements=100 000`, rebuildable via `rebuild_index()`); it publishes by flushed private file plus atomic replacement. `chunk_terms` is also derived and rebuildable through `LexicalIndex.rebuild()`. Qwen is never part of durable state; it is a bounded compiler/inspector that emits compact records and unloads.

## 3. Storage schema (v13)

| Table | Columns (contract-relevant) | Notes |
| --- | --- | --- |
| `turns` | `turn_id` PK, `role` CHECK ∈ {user, assistant, system}, `text`, `source_id` nullable, `created_at`, `ordinal` | append-only; `(source_id, ordinal)` indexed for session/document hydration |
| `chunks` | `chunk_id` PK, `turn_id` FK, `text`, `start_char`, `end_char`, `token_count`, `embedding` BLOB (dim×4 bytes f32), `lexical_weights` TEXT (JSON term→tf, **now populated**), `hnsw_label` INTEGER UNIQUE, `term_count` INTEGER | `hnsw_label` is the sole chunk↔dense-index mapping; `term_count` is the BM25 document length (NULL ⇒ not lexically indexed) |
| `chunk_terms` | `term`, `chunk_id` FK, `tf`, PK `(term, chunk_id)` | BM25 inverted index; postings are replaced wholesale per chunk, never appended to |
| `memory_items` | `mem_id` PK, `type`, `content`, `details`, `status` CHECK ∈ {active, superseded, deleted}, `supersedes`, `pin` CHECK ∈ {user_pinned, system_pinned, none}, `energy` REAL, `half_life_turns` REAL (default 30), `last_access_turn` INTEGER, `importance` REAL, `created_at`, `last_access_at` (audit only), `embedding` BLOB, `content_hash` (v3, indexed, **not** UNIQUE); legacy `half_life_s` remains inert for additive migration compatibility | see clause 9 — status transitions only, no deletes |
| `memory_provenance` | `mem_id` FK ON DELETE CASCADE, `turn_id`, `chunk_id` (nullable), `quote`, UNIQUE `(mem_id, turn_id, quote)` | see clause 8 — at least one row per item |
| `association_artifacts` | `artifact_id` PK, model/checkpoint identity, prefix/head/CAV layers, JSON concept names, head count, creation time, JSON metadata | defines how every compact vector must be interpreted; reusing an ID with a different interpretation is rejected |
| `chunk_cav_signatures` | `(chunk_id, artifact_id)` PK/FKs, fixed-width f32 `signature` BLOB, created/access turns, access count | concept coordinates only; width is exactly the artifact's concept count |
| `chunk_head_edges` | `(source_chunk_id, destination_chunk_id, artifact_id)` PK/FKs, fixed-width f32 `head_weights`, scalar `qk_score`, scalar `ov_transport`, evidence/traversal counters, last-access turn, optional temporal direction | sparse directed graph; self-edges forbidden; width is exactly the artifact's query-head count |
| `hebbian_access_events`, `hebbian_chunk_nodes`, `hebbian_chunk_edges` | artifact-scoped event fingerprints, chunk IDs, scalar node/edge masses, counts, and turn coordinates | v7 bounded live co-access projection over conceptual chunks; no request-derived prompt/token state |
| `consolidation_access_events` | `event_id` PK, observed turn, SHA-256 membership fingerprint, member count | v8 bounded idempotency receipts; no rendered context |
| `consolidation_nodes` | typed `node_key` pointing to exactly one active `memory_item` or retrievable `chunk`, scalar access mass/count and last-access turn | cross-partition address only; retired source state removes this derived node through triggers |
| `consolidation_edges` | ordered node pair PK/FKs, scalar co-activation mass/count, causal count, last-reinforced turn | model-independent live assembly; distinguishes completed-interaction binding from incidental co-access; hard degree pruning and turn decay |
| `discourse_artifacts`, `episodes`, `episode_evidence`, `episode_representatives` | immutable artifact identity, source/ordinal episode boundaries, exact chunk/span/hash evidence coordinates, representative chunk/vector receipts | v10 episodic projection; source text remains in turns/chunks |
| `discourse_units`, `discourse_unit_evidence`, `discourse_relations`, `discourse_relation_members`, `discourse_relation_evidence` | immutable typed unit/relation identities, scalar confidence/weights, exact evidence coordinates | v10 source-grounded discourse graph; no generated evidence text or token state |
| `discourse_graph_revisions`, `discourse_revision_state`, `discourse_artifact_coverage`, `discourse_artifact_coverage_receipts` | source and graph revision counters/hashes, artifact coverage including `no_output`, immutable snapshot receipts | v10/v11 closes content/snapshot and zero-output ambiguity |
| `memory_successor_redirects` | predecessor PK, successor FK, reason, creation time | v12 additive forward edge when one replacement absorbs more predecessors than scalar `supersedes` can name |
| `pending_ingests` | turn PK/FK, canonical manifest hash/JSON, status CHECK ∈ {pending, indexed}, creation/index times | v13 exact chunk-topology receipt; indexed rows remain durable and pending rows are replayable |
| `ingest_chunk_reservations` | `chunk_id` PK, turn FK, span, token count, text hash | v13 normalized global ownership; insert must exactly match one member of the owning manifest; rows are immutable and durable |
| `meta` | `key` PK, `value` | holds `schema_version`, ANN-label allocation, and cross-process `chunk_index_revision` coordinates |

### 3.1 Migration path

| From | To | Applied changes |
| --- | --- | --- |
| (no file / no `meta` table) | 13 | full schema created directly at v13 |
| 1 | 2 | `ALTER TABLE chunks ADD COLUMN term_count`; create `chunk_terms`, `memory_items`, `memory_provenance` and their indexes; `UPDATE meta SET value = '2'` |
| 2 | 3 | `ALTER TABLE memory_items ADD COLUMN content_hash`; `idx_memory_content_hash`; **post-migration backfill** of `content_hash` for existing rows; `UPDATE meta SET value = '3'` |
| 3 | 4 | add `turns.ordinal`, `memory_items.half_life_turns`, and `memory_items.last_access_turn`; backfill turn ordinals and enter existing memories at the latest turn |
| 4 | 5 | create `association_artifacts`, `chunk_cav_signatures`, and `chunk_head_edges` plus artifact/destination indexes; no transcript or memory row is rewritten |
| 5 | 6 | add nullable `turns.source_id` plus `(source_id, ordinal)` index; legacy turns remain valid and use `turn_id` as the source fallback |
| 6 | 7 | add artifact-scoped bounded Hebbian access-event, chunk-node, and chunk-edge tables |
| 7 | 8 | add model-independent cross-partition consolidation events, typed nodes, scalar edges, indexes, and retirement triggers |
| 8 | 9 | add `consolidation_edges.causal_count` so completed-interaction binding is distinct from co-access |
| 9 | 10 | add immutable source-grounded episode/discourse artifacts, exact evidence coordinates, coverage members, and graph revision receipts |
| 10 | 11 | add authoritative role/time/turn evidence fields, monotonic source/graph content revisions and hashes, exact per-artifact chunk coverage including `no_output`, and immutable publication triggers |
| 11 | 12 | add `memory_successor_redirects` for many-to-one exact-duplicate successor history |
| 12 | 13 | add canonical pending-to-indexed ingest receipts, normalized global chunk reservations, and trigger-enforced exact membership/durability/monotonicity; migration seals every legacy chunked turn as historical `indexed` state and leaves zero-chunk legacy turns unclaimed because interruption cannot be inferred safely |

`Database.schema_version` reports the on-disk version (`0` when unreadable). Migrations run inside `Database.__init__`, so opening a v1 file upgrades it — no separate migration command exists, and none should be added without also making the upgrade opt-in.

Some migrations need work SQL cannot express — v3 hashes normalized memory content, v4 backfills the turn clock, v11 seals a content-bound discourse baseline, and v13 builds canonical manifests and reservations for legacy chunked turns. Those live in `db._POST_MIGRATIONS`, keyed by target version, and execute inside the same migration transaction as their schema/version publication.

The trigger contract protects supported writers and migrations; it is not a
security boundary against a caller with arbitrary raw-SQL authority. Such a
caller can manufacture an initially `indexed` receipt because the completion
proof trigger governs updates. Public stores, ingest/index APIs, and migrations
MUST fail closed and MUST NOT expose that privileged construction path.

**`content_hash` is indexed but deliberately not UNIQUE.** Stores written before v3 almost certainly already contain duplicates — that is the bug the column exists to stop — and `CREATE UNIQUE INDEX` would raise inside `Database.__init__`, making an existing store permanently unopenable. Uniqueness is enforced in `MemoryStore.create`; `MemoryStore.dedupe_existing()` cleans a legacy store on request (never from a migration: opening a database must not silently rewrite it). Promoting the constraint into the schema is a later version, after that cleanup has run.

### 3.2 Value contracts

| Quantity | Contract |
| --- | --- |
| `energy` | `[0, 1]`; decayed lazily on read as `energy × 0.5^((now_turn − last_access_turn) / half_life_turns)`; pinned items exempt. An access raises it by `0.25 × (1 − energy)` — closing a fraction of the *remaining* headroom, so it approaches 1.0 without reaching it — and can boost at most once per conversation turn. Implementations MUST compute decay through `decay.decay_factor`; a `half_life_turns ≤ 0` MUST mean "does not decay" |
| heat tier | derived, never stored: HOT ≥ `0.75`, WARM ≥ `0.25`, else COLD — **and** at most `HOT_CAP` (20) *unpinned* items may hold HOT at once, the excess derived down one tier to WARM, lowest energy first. Pins neither occupy a slot nor get demoted. A cap is still derivation; it simply derives from the pool rather than from one row |
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
5. LongMemEval `haystack_session_ids` and LoCoMo `session_N` names MUST populate `turns.source_id`; their session timestamps MUST be ingested as source-tagged system turns. LongMemEval `question_date` MUST be included in the retrieval/answer query so temporal questions are not evaluated after silently discarding time.

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
pixi run python -c "import sqlite3, tempfile, pathlib; from memory_condense.persistence.db import Database; p=pathlib.Path(tempfile.mkdtemp())/'v.db'; d=Database(p); print(d.schema_version); print(sorted(r[0] for r in d.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")))"
```

Expect `13` and a table list containing transcript, chunk/BM25,
memory/provenance/successor redirects, CAV/QK/OV, Hebbian and consolidation,
source-grounded discourse/coverage/revision receipts, `pending_ingests`, and
`ingest_chunk_reservations`.
Schema v12 adds explicit many-to-one successor redirects; schema v13 seals the
turn-to-index topology and recovery state described by clause 15.

Drift between `_SCHEMA_SQL` and `_MIGRATIONS` is no longer something to catch by hand: `tests/test_db.py::TestSchemaParity` builds a fresh database and a migrated one from both v1 and v2, then asserts they converge on the same tables, columns, and indexes. It compares shape rather than DDL text, because `ALTER TABLE ADD COLUMN` and `CREATE TABLE` render the same logical column differently and a text comparison would fail on every additive migration until everyone learned to ignore it.
