**Standard ID:** MC-STD-DATA v0
**Title:** Data contracts — storage, embedding, chunking, memory, loader, eval output
**Status:** DRAFT (not yet frozen — freeze at first release or first external consumer, whichever comes first)
**Date:** 2026-09-07
**Supersedes:** the v1-schema-only revision of this standard (documented `turns` + `chunks` and nothing else)
**Applies to:** `src/memory_condense/` and `eval_results/` JSON
**Depends on:** • `03 - Architecture/00 - System Overview.md` (as-built map)

> **Current worktree**: `schema_version` 17. v4 moves lifecycle decay to conversation-turn coordinates; v5 adds compact CAV/QK/OV artifacts; v6 adds source/session identity; v7 adds chunk Hebbian co-access; v8/v9 add live consolidation and causal counts; v10/v11 add source-grounded discourse plus content-bound revision receipts; v12 adds many-to-one memory-successor redirects; v13 adds durable T0-to-T1 ingest manifests plus normalized globally unique chunk reservations; v14 adds manifest-bound T2 obligations; v15 adds deterministic staged T2 publication, bounded/fair retry state, source-order retirement history, exact legacy-enrichment quarantine, per-operation correction receipts, and stale-writer fences; v16 adds the manifest-bound incremental conversation-graph journal, immutable deltas, and checkpoint state; and v17 adds policy-scoped append-only user-led envelope assignments and events. Stores at any earlier version migrate in place on open — see clauses 10 and 17–19.

## 0. Scope

Covers: the SQLite schema and migrations through v17, the T0/T1/T1g/T2 publication contracts, append-only conversation-envelope publication, embedding and chunking, crash-replay and retry receipts, dense and lexical indexes, memory-item and provenance contracts, compact association/consolidation/discourse artifacts, loader input formats (Claude exports + LongMemEval/LoCoMo), and eval result JSON.

Does NOT cover: prompt formats (implementation detail), rank weights or `alpha` (tuning parameters, not contract), or the `MemoryOps` **wire** contract for external producers — that gets its own standard (MC-STD-MEMOPS) if and when a non-local producer exists.

## 1. Normative Goals

1. Derived state MUST be reconstructible from the `turns` table plus sealed configuration/receipts. `pending_ingests` and `ingest_chunk_reservations` preserve the exact T0 chunk topology and globally unique ownership when ambient chunker settings later change. `pending_enrichments` records a T2 obligation independently of T1 indexing, while `pending_enrichment_state` seals the canonical operation payload and digest needed for deterministic replay until atomic publication. `pending_graph_compilations` plus the immutable graph delta/checkpoint chain seal the v16 incremental graph, and policy-scoped envelope assignments/events seal the v17 user-led exchange projection. Neither projection may duplicate raw turn or chunk text. The hnswlib `.bin` file is a cache, never a source of truth; a corrupt image MUST rebuild from SQLite. A partial ANN add or synchronization and an ambiguous or interrupted native retirement MUST discard the process-local graph for reconstruction from SQLite. The BM25 inverted index is likewise derived (`LexicalIndex.rebuild()`).
2. Every chunk MUST carry provenance: `turn_id` + `start_char`/`end_char` spans into the turn text.
3. Writers MUST NOT mutate or delete `turns` rows (append-only).
4. Embeddings MUST be `float32`, produced by the configured sentence-transformers model, stored **unnormalized**. A consumer MUST NOT dot-product raw blobs without normalizing; cosine comparisons go through the hnswlib cosine space or explicit normalization.
5. Any change to embedding model or dimension MUST bump `schema_version` and rebuild the index. `EmbeddingService.dim` now reports the model's **true** dimension (falling back to the constant `1024` only for the unloaded default `BAAI/bge-m3`), so a model swap is no longer silently corrupting — but it is still a schema change and MUST be treated as one.
6. Chunk token counts MUST be measured with tiktoken `cl100k_base` and lie in `[min_tokens, max_tokens]` except single indivisible words.
7. Eval baseline and treatment runs MUST share the same code path (`k=0` vs `k>0`), never a separate harness. `retrieval.query()` (pure dense) MUST remain behaviourally unchanged for this reason; hybrid retrieval is an additive method (`hybrid_query`), not a modification.
8. **Every memory item MUST carry at least one provenance entry**, and every provenance `quote` MUST appear verbatim in the referenced turn's text, compared after whitespace normalization only (runs of whitespace collapsed to one space, ends stripped). No case folding, no punctuation stripping, no fuzzy matching. An op that cannot satisfy this MUST be rejected, not stored. *(Enforced by `validator.Validator`; the sole exception is `UpdateOp`, which MAY carry an empty provenance list because it amends an item that already has provenance — any entry it **does** carry is checked in full.)*
9. **Memory rows MUST NOT be destroyed, and terminal chronology MUST be source ordered.** `delete` MUST set `status = 'deleted'`; a correction MUST be expressed as a supersede — a new row whose `supersedes` names the old one, with the old row set to `status = 'superseded'`. Every v15 active-to-terminal transition MUST set `retired_at_turn` to the current transcript ordinal and append the retired identity to `memory_identity_retirements` in the same transaction. An active in-place identity change MUST append the prior identity with reason `updated` before changing it. A delayed T2 create from source ordinal *s* MUST be suppressed when the same identity has a retirement at ordinal ≥ *s*. Terminal `status`, `content_hash`, ledger entries, and a non-NULL `retired_at_turn` are immutable. A pre-v15 terminal row MAY receive exactly one explicit `NULL -> retired_at_turn` binding within the recorded migration boundary; the binding and its ledger entry MUST be atomic. Hard `DELETE` on `memory_items` is forbidden: the audit trail from a memory back to the transcript that justifies it MUST stay walkable. Removing a chunk from the indexes (`retrieval.delete_chunk`) MUST likewise keep the `chunks` row so provenance cannot dangle.
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
15. **T0 capture and T1 search publication MUST be replayable and topology
    sealed.** T0 is the committed transaction containing the append-only turn,
    its canonical `pending_ingests` manifest, every
    `ingest_chunk_reservations` row, its manifest-bound T1g graph obligation,
    its policy/turn-keyed envelope assignment, and (when requested) its
    manifest-bound T2 obligation. T0 MUST complete before any embedding,
    extraction, envelope-boundary assignment, ANN, lexical-index, or graph-
    compilation work begins. The manifest MUST name the complete ordered set
    of chunk IDs, source spans, token counts, and text hashes without copying
    source text or embeddings. Each manifest member MUST have one globally
    unique normalized reservation whose owner, span, token count, and text hash
    exactly equal that member.

    T1 is the one legal `pending -> indexed` receipt transition. It MUST occur
    only in the transaction that proves every expected chunk has an embedding,
    HNSW label, and lexical document length and that no unexpected chunk belongs
    to the turn. A failed or interrupted T1 step MUST leave T0 and its pending
    receipt intact; it MUST NOT compensate by deleting the turn. An embedder
    result MUST be a one-to-one derivative of a pre-call deep snapshot of the
    reconstructed manifest chunks, and the provider MUST receive separate deep
    copies so nested mutation cannot alter the validation baseline. Provider
    work MUST finish before a SQLite writer lock is acquired. No missing,
    extra, duplicate, unembedded, or source-field replacement is admissible.
    Completed receipts and reservations MUST remain durable. A retry MUST
    reconstruct the sealed topology instead of rerunning the ambient chunker or
    generating new IDs, MUST fail closed on mismatched source content, and MAY
    reuse complete durable vectors without re-embedding.

    Exact manifest membership, receipt/reservation immutability and durability,
    and the monotonic complete transition MUST be enforced by SQLite triggers
    with `recursive_triggers=ON` on every connection. Supported direct dense or
    lexical completion MUST be restricted to pending members. Once a receipt is
    indexed, a missing or incomplete member is terminal retired state and MUST
    NOT be reactivated. The sole lexical repair exception is no-argument
    `LexicalIndex.rebuild()`, which snapshots its live batch from authoritative
    SQLite before clearing postings; a caller-supplied iterable remains a
    direct write and MUST reject terminal retired members.
16. **T2 enrichment MUST stage deterministically and publish atomically.** When
    automatic extraction is requested, T0 MUST create one
    `pending_enrichments` receipt in the capture transaction, bound to the exact
    ingest-manifest hash. A worker adopting an older pending T1 receipt under an
    explicit `auto_extract=True` policy MUST create that obligation before any
    provider or T1 work. Migration MUST NOT infer extraction intent for turns
    that have no receipt. The receipt may advance only `pending -> enriched` and
    only after its T1 receipt is `indexed`; a valid no-op result still completes
    the obligation. Recovery of an indexed turn MUST NOT re-embed its chunks.

    Extraction and memory-embedding providers MUST run without a SQLite writer
    lock. The first validated result committed to `pending_enrichment_state`
    wins: its canonical `MemoryOps` JSON, sorted live chunk-ID set, and SHA-256
    over both form one immutable staged identity. Concurrent or restarted
    helpers MUST replay that exact winner and MUST NOT substitute later provider
    prose. After revalidating live evidence and prepared memory state under the
    writer lock, T2 MUST atomically apply safe creates, enqueue every still-
    grounded `Correction` as an independent `pending_corrections` operation,
    advance the T2 receipt to `enriched`, and clear the staged JSON/chunk list
    while retaining its digest. Any failure before that commit MUST leave the
    T2 receipt pending and the staged payload recoverable. Thus provider calls
    are at-least-once, but durable memory effects, correction publication,
    receipt finalization, and staged-payload clearing share one commit boundary.

    Deferred T2 observes drain-time turn/heat state rather than promising
    synchronous causal visibility. It MUST exclude retired chunks from both
    extractor-visible text and provenance; if no live chunk remains, it MUST
    stage and publish a no-op without invoking the extractor. A correction MUST
    NOT guess its target during automatic T2. Each operation remains pending
    independently after safe creates publish. Resolution MUST revalidate live
    grounding, the immutable operation digest, the reviewed active target's
    revision, semantic non-no-op, and absence of an active replacement
    collision both before provider work and under the writer lock; superseding
    the target and marking the correction `resolved` MUST be atomic. Dismissal
    MUST be an explicit digest-checked `pending -> dismissed` transaction.
    Correction rows, operation indexes/payloads/digests, and terminal decisions
    MUST remain durable and immutable.
17. **Recovery scheduling and schema upgrades MUST fail safely under load.** T1
    and T2 failures MUST retain bounded state: attempt count, last/next attempt
    timestamps, and a bounded exception-class label, never provider payloads.
    Both stages MUST retry once immediately and then use exponential backoff
    capped at 60 seconds, except that a twice-failed multi-turn T1 cohort MUST
    receive its one immediate singleton-isolation pass before cooldown. When
    fresh and retry work are both eligible, the durable stage scheduler MUST
    alternate classes and fall through to the other class if a concurrent
    worker empties its first choice. One failed T1
    provider cohort MAY be retried once as that same cohort; a second cohort
    failure MUST dissolve it into singleton retries so one poison manifest
    cannot exclude unrelated work. Every automatic T1/T2 maintenance tick MUST
    process one caller-bounded batch; a manifest is indivisible and the first
    eligible manifest MAY exceed a chunk/token bound solely to guarantee
    forward progress.

    v15 records the maximum pre-migration turn ordinal and permanently
    quarantines each exact pending T2 receipt present at migration, plus any T2
    receipt later adopted from a source at or before that boundary. Such a
    receipt MUST be excluded from automatic replay because pre-v15 identity
    retirement order cannot be reconstructed. It MAY receive an explicit
    `discarded_legacy` disposition only after its T1 receipt is `indexed`; that
    disposition, T2 terminal transition, and staged-payload clear MUST be one
    transaction, and the indexed T1 evidence MUST remain searchable. A
    quarantine marker or disposition MUST NOT be updated or deleted.

    Migration requires a stop-the-world writer boundary: all writer processes
    MUST close before upgrade and restart on v17 code afterward. v17 SQLite
    triggers call `memory_condense_writer_schema_version()` to fail closed when
    an already-open stale supported writer attempts its first turn, ingest,
    chunk, lexical, enrichment, memory, graph, or envelope mutation, but those
    fences are not a rolling-upgrade protocol or a security boundary against
    arbitrary raw SQL.
    They also cannot coordinate an already-loaded process-local HNSW graph, so
    process restart (and cache reconstruction when integrity is uncertain) is
    mandatory. Migration itself MUST serialize on a writer transaction and
    re-read `schema_version` after acquiring the lock so concurrent openers do
    not apply the same migration twice.
18. **T1g conversation-graph publication MUST be incremental, policy-bound,
    and fail open from T1.** T0 MUST claim one `pending_graph_compilations` row
    for the active `graph_artifacts` identity and exact ingest-manifest digest;
    it MUST NOT extract phrases or publish graph evidence in that transaction.
    A worker MAY compile the turn only after the matching T1 receipt is
    `indexed`, and MUST commit all graph chunks, phrase occurrences,
    source-local story-term memberships, aggregate state, and the terminal
    `ready` or `no_output` receipt for that turn atomically. Failure MUST retain
    a retryable pending receipt and MUST NOT roll back or hide T1 evidence.

    Graph artifacts MUST seal extraction, story-index, and persistence-policy
    identities. Graph chunks, phrase occurrences, and story memberships MUST be
    immutable and durable; `conversation_graph_state` MUST advance
    monotonically and MUST NOT be deleted. Every append MUST bind its delta to
    the prior SHA-256 checkpoint and the resulting state digest. These rows MAY
    retain IDs, ordinals, source spans, canonical terms, counts, and hashes, but
    MUST hydrate factual text from authoritative turns/chunks. A resident read
    MUST pin a target revision/state/checkpoint before hydration, apply only
    deltas through that target, and fail closed on a missing source row,
    identity mismatch, noncontiguous revision, or invalid hash chain. A
    concurrent writer MUST NOT make one read mix revisions or chase a moving
    head without bound.

    `bootstrap_conversation_graph(max_turns=32)` MUST claim at most one finite
    current-artifact page (hard maximum 512) transactionally and then compile
    only those selected turn IDs. It MUST validate canonical indexed ingest
    receipts, remain resumable and idempotent under the same policy, create no
    historical work implicitly on database open, and report otherwise-complete
    legacy turns without manifests as unsupported instead of reconstructing
    them heuristically. The explicit graph drain MAY retain its opt-in
    `max_turns=None` all-pending behavior; automatic idle maintenance MUST use a
    finite batch.
19. **Conversation envelopes MUST be append-only, user-led, source-local, and
    fail open from searchable state.** T0 MUST claim exactly one
    `pending_conversation_envelope_assignments` row per
    `(policy_sha256, turn_id)`, retaining only turn/source coordinates, current
    role, actor/authority labels, optional `parent_turn_id`, and canonical
    input/receipt digests—not raw text. Normal ingest completion MUST attempt
    boundary assignment only after T1; the explicit bootstrap MAY derive it
    from a valid T0 ingest manifest because it publishes no search state. A user turn
    with a stable non-NULL `source_id` MUST open a deterministic envelope;
    assistant/system turns MUST attach to the latest earlier user opener in the
    same source, unless a valid explicit parent routes them to that parent's
    opener. A parent MUST already have a same-policy event in the same source
    at a strictly lower ordinal. Current capture accepts only
    user/assistant/system roles; tool actor/authority values are a persistence
    seam and MUST NOT be interpreted as broader capture support.

    Each later user opener MUST name the prior same-source envelope through
    `predecessor_envelope_id`; it MUST NOT update the predecessor. A worker MUST
    not pass a lower-ordinal pending assignment in the same source. Missing
    source, absence of a prior user, and invalid parentage MUST produce distinct
    terminal `no_anchor` receipts. Policies, terminal assignments, and event
    rows MUST be durable; policy/event rows MUST be immutable and no envelope
    row may be deleted. Replay under one policy MUST reproduce the exact event
    and receipt identities. These live envelopes MUST remain distinct from the
    existing immutable discourse `episodes` artifacts.

    Envelope drains and bootstrap pages MUST default to 32 turns and reject a
    page above 512. Bootstrap MUST transactionally claim only canonical
    pending-ingest-manifest-backed turns for the active policy, perform no
    boundary work while claiming, then drain only the exact selected IDs. It
    MUST be resumable/idempotent, report legacy turns without manifests as
    unsupported, and never rebuild implicitly on open. Because predecessor
    links cannot be retrofitted, bootstrap MUST fail closed when selected
    missing history precedes that policy's maximum published event ordinal and
    require a fresh policy identity. When one post-T1 maintenance path services
    both projections, it MUST attempt envelopes before T1g; either worker's
    failure MUST be recorded independently without rolling back T1 or
    suppressing the other attempt.

    Retrieval expansion over this journal MUST remain explicit and opt-in; the
    unexpanded raw retrieval result MUST remain the default. A plan MUST bind
    the current envelope policy, its exact ordered original `(chunk_id,
    token_count)` footprint, its formats, and all envelope/turn/companion
    limits without storing transcript text. Hydration MUST authenticate every
    selected chunk's turn, source, and token coordinates against that plan,
    MUST emit each admitted group in source order with the user opener first,
    and MUST fail open atomically for a missing, stale, foreign, mismatched, or
    over-budget group. Exact-ID collisions MUST retain the caller's original
    retrieval object, and every original row MUST survive expansion.

    Chunk and token limits on this read path MUST apply only to newly hydrated
    companions; they MUST NOT purchase space by evicting raw hits. Companion
    rows MUST carry a distinct nonrecursive route and MUST NOT be learned as
    independent direct retrieval hits. Expansion MUST occur after selection
    and exact-ID deduplication MUST occur only while composing the two selected
    lanes. The final context packer MUST independently enforce its prompt
    ceiling and MAY omit expansion rows that do not fit.

## 2. Core concept

One SQLite database (WAL, `foreign_keys=ON`, `recursive_triggers=ON`, `schema_version` 17) holds everything durable: source-identified turns; exact T0/T1 manifests and globally unique reservations; manifest-bound T1g graph jobs, immutable graph deltas/checkpoints, and policy-scoped envelope assignments/events; manifest-bound T2, staged-result, retry, quarantine, disposition, and correction receipts; chunks/BM25 postings; memory retirement/provenance/successor history; compact association state; live consolidation; and immutable source-grounded discourse receipts. The hnswlib file beside it is a derived cache (`hnsw_index.bin`, cosine, `M=16`, `ef_construction=200`, `max_elements=100 000`, rebuildable via `rebuild_index()`); it publishes by flushed private file plus atomic replacement. `chunk_terms` is also derived and rebuildable through `LexicalIndex.rebuild()`. Qwen is never part of durable state; it is a bounded compiler/inspector that emits compact records and unloads.

## 3. Storage schema (v17)

| Table | Columns (contract-relevant) | Notes |
| --- | --- | --- |
| `turns` | `turn_id` PK, `role` CHECK ∈ {user, assistant, system}, `text`, `source_id` nullable, `created_at`, `ordinal` | append-only; `(source_id, ordinal)` indexed for session/document hydration |
| `chunks` | `chunk_id` PK, `turn_id` FK, `text`, `start_char`, `end_char`, `token_count`, `embedding` BLOB (dim×4 bytes f32), `lexical_weights` TEXT (JSON term→tf, **now populated**), `hnsw_label` INTEGER UNIQUE, `term_count` INTEGER | `hnsw_label` is the sole chunk↔dense-index mapping; `term_count` is the BM25 document length (NULL ⇒ not lexically indexed) |
| `chunk_terms` | `term`, `chunk_id` FK, `tf`, PK `(term, chunk_id)` | BM25 inverted index; postings are replaced wholesale per chunk, never appended to |
| `memory_items` | `mem_id` PK, `type`, `content`, `details`, `status` CHECK ∈ {active, superseded, deleted}, `supersedes`, `pin` CHECK ∈ {user_pinned, system_pinned, none}, `energy` REAL, `half_life_turns` REAL (default 30), `last_access_turn` INTEGER, `importance` REAL, `created_at`, `last_access_at` (audit only), `embedding` BLOB, `content_hash` (v3, indexed, **not** UNIQUE), `retired_at_turn` nullable nonnegative; legacy `half_life_s` remains inert for additive migration compatibility | see clause 9 — source-ordered terminal transitions only, no deletes; pre-v15 terminal rows retain NULL until explicitly bound once |
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
| `pending_enrichments` | turn PK/FK, ingest-manifest hash, status CHECK ∈ {pending, enriched}, creation/enrichment times | v14 automatic-extraction obligation; bound by composite FK to the exact ingest receipt, durable and monotonic; no historical rows are inferred |
| `pending_ingest_attempts` | turn PK/FK, positive `attempt_count`, `last_attempt_at`, `next_attempt_at`, bounded `last_error_kind` | v15 monotonic T1 retry history; no exception/provider payload is retained |
| `pending_enrichment_state` | turn PK/FK, retry fields, canonical `staged_ops_json`, sorted `staged_chunk_ids_json`, retained `staged_result_sha256` | v15 T2 retry and deterministic-replay state; staged payload may be published once and cleared only after T2 completion |
| `pending_work_schedule` | stage PK CHECK ∈ {ingest, enrichment}, `prefer_retry` Boolean | v15 durable fair alternation between eligible fresh and retry classes |
| `memory_identity_retirements` | memory FK, retired `content_hash`, nonnegative `retired_at_turn`, reason CHECK ∈ {updated, deleted, superseded, deduplicated}, composite PK | v15 immutable source-order identity tombstones used to suppress stale T2 creates |
| `pending_enrichment_legacy_quarantine` | turn PK/FK, nonnegative migration boundary | immutable exact marker for ambiguous pre-v15 T2 work; automatic replay excludes it |
| `pending_enrichment_dispositions` | turn PK/FK, `discarded_legacy`, decision time/reason | immutable explicit terminal outcome distinct from successful extraction; allowed only after T1 is indexed |
| `pending_corrections` | digest-derived correction PK, turn FK, immutable operation index/JSON/SHA-256, status CHECK ∈ {pending, resolved, dismissed}, decision time, reviewed target/successor IDs, reason | v15 per-operation correction queue; exactly one immutable terminal decision |
| `graph_artifacts` | SHA-256 `artifact_id` PK, format, extraction/story-index/persistence policy digests, creation time | v16 immutable interpretation identity; a policy change produces a separate artifact |
| `pending_graph_compilations` | `(artifact_id, turn_id)` PK/FKs, exact ingest-manifest digest, status CHECK ∈ {pending, ready, no_output}, revision/count/checkpoint/receipt fields, bounded failure metadata | v16 T1g obligation; claimed at T0, terminalized only after T1 is indexed, durable and monotone |
| `conversation_graph_chunks`, `conversation_phrase_occurrences`, `conversation_story_term_memberships` | artifact-scoped chunk/source/span/ordinal coordinates, canonical phrase/story keys, identity/quote/runtime/delta/checkpoint digests, append revisions and bounded counts | v16 immutable incremental graph deltas; no raw text duplication; factual payload hydrates from turns/chunks |
| `conversation_graph_state` | artifact PK/FK, monotone revision/counts, checkpoint and state digests, update time | v16 aggregate head; updates advance one complete turn, deletion is forbidden, and resident reads pin one target state |
| `conversation_envelope_policies` | SHA-256 policy PK, fixed format, creation time | v17 immutable policy identity; policy changes do not reinterpret existing events |
| `pending_conversation_envelope_assignments` | `(policy_sha256, turn_id)` PK/FKs, source/ordinal/role, actor/authority, optional parent, input digest, status CHECK ∈ {pending, ready, no_anchor}, terminal receipt/diagnostic and bounded failure metadata | v17 text-free T0 claim and monotone assignment receipt; rows are durable and source-ordered by the worker |
| `conversation_envelope_events` | `(policy_sha256, turn_id)` PK/FK, deterministic envelope/opener IDs, optional predecessor envelope, event kind, source/ordinal, actor/authority/parent, unique receipt | v17 immutable/no-delete open/member journal; a new opener links its predecessor without mutating it |
| `meta` | `key` PK, `value` | holds `schema_version`, ANN-label allocation, cross-process `chunk_index_revision`, and `v15_legacy_retirement_boundary` |

### 3.1 Migration path

| From | To | Applied changes |
| --- | --- | --- |
| (no file / no `meta` table) | 17 | full schema created directly at v17 |
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
| 13 | 14 | add the composite ingest-receipt identity index required by the manifest-bound FK, pending-to-enriched receipts, and trigger-enforced durability/monotonicity; migration deliberately creates no historical enrichment receipts because prior extraction intent/completion cannot be inferred |
| 14 | 15 | add source-order retirement coordinates and immutable identity-retirement ledger; T1/T2 attempt state and durable fair scheduler; canonical staged T2 payload/digest state; exact legacy T2 quarantine and explicit disposition receipts; per-operation correction receipts; terminal chronology guards; and stale supported-writer fences. Record the maximum pre-migration turn ordinal, leave legacy terminal chronology NULL rather than inventing it, and quarantine exact pending T2 receipts rather than applying them |
| 15 | 16 | add versioned graph artifacts, manifest-bound T1g jobs, immutable graph chunk/phrase/story deltas, monotone aggregate checkpoint state, and graph writer fences; do not infer historical graph jobs or replay the corpus during migration |
| 16 | 17 | add immutable envelope policies, durable policy/turn assignment claims, append-only envelope events and their source/order/parent guards; replace v16 writer fences with v17 fences covering graph and envelope publication; do not infer historical envelope claims during migration |

`Database.schema_version` reports the on-disk version (`0` when unreadable). Migrations run inside `Database.__init__`, so opening a v1 file upgrades it — no separate migration command exists, and none should be added without also making the upgrade opt-in.

Some migrations need work SQL cannot express — v3 hashes normalized memory content, v4 backfills the turn clock, v11 seals a content-bound discourse baseline, and v13 builds canonical manifests and reservations for legacy chunked turns. Those live in `db._POST_MIGRATIONS`, keyed by target version, and execute inside the same migration transaction as their schema/version publication. v15 uses a pre-migration hook only to add `memory_items.retired_at_turn` when absent; its tables, boundary capture, quarantine snapshot, and guards publish in that same serialized migration transaction. v16 and v17 require no automatic data backfill: their policy-scoped journals begin empty for historical turns, and only the explicit bounded, manifest-validating bootstrap APIs may adopt that history.

The trigger contract protects supported writers and migrations; it is not a
security boundary against a caller with arbitrary raw-SQL authority. Such a
caller can manufacture an initially `indexed` receipt because the completion
proof trigger governs updates. Public stores, ingest/index APIs, and migrations
MUST fail closed and MUST NOT expose that privileged construction path. The v17
writer-version function and triggers are an additional stale-process fuse, not
permission to run mixed schema versions; clause 17's stop-the-world restart
boundary remains mandatory.

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

Expect `17` and a table list containing transcript, chunk/BM25,
memory/provenance/successor/retirement history, CAV/QK/OV, Hebbian and
consolidation, source-grounded discourse/coverage/revision receipts,
`pending_ingests`, `ingest_chunk_reservations`, `pending_enrichments`, both
attempt/state tables, `pending_work_schedule`, both legacy quarantine/disposition
tables, `pending_corrections`, the v16 graph artifact/job/delta/state tables, and
the v17 envelope policy/assignment/event tables.
Schema v12 adds explicit many-to-one successor redirects; schema v13 seals the
turn-to-index topology and recovery state described by clause 15; schema v14
seals the automatic-extraction obligation; schema v15 makes T2 staging and
publication deterministic, records source-order retirements, isolates legacy
ambiguity and corrections, schedules retries fairly, and fences stale writers
as described by clauses 16 and 17. Schema v16 adds the incremental graph
contract in clause 18; schema v17 adds the append-only user-led envelope
contract in clause 19 and advances the supported-writer fence.

Drift between `_SCHEMA_SQL` and `_MIGRATIONS` is no longer something to catch by hand: `tests/test_db.py::TestSchemaParity` builds fresh and migrated databases from multiple historical versions, then asserts they converge on the same tables, columns, indexes, and triggers. It compares shape rather than DDL text, because `ALTER TABLE ADD COLUMN` and `CREATE TABLE` render the same logical column differently and a text comparison would fail on every additive migration until everyone learned to ignore it.
