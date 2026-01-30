Perfect: **LLM API for generation + BAAI bge-m3 locally for retrieval/salience** is a clean, deployable split. Here’s an implementation plan that assumes:

* **No attention / no hidden-states**
* **Local memory manager owns all state**
* **bge-m3 provides embeddings + optional lexical weights**
* **LLM API answers the user + proposes memory updates in a strict schema**

---

## 1) Architecture you’ll build

### Services / modules

1. **TranscriptStore (append-only)**

* Stores every turn (raw text, role, timestamp, turn_id)

2. **Chunker**

* Splits turns into **chunks** (sentence→paragraph merge)
* Output: chunk_id, text, span pointers into transcript

3. **Embedding + Salience (bge-m3)**

* Compute dense embedding for each chunk
* Optionally compute lexical weights (for sparse/keyword-ish match)

4. **MemoryStore**

* Stores “memory items” (typed, compact) + provenance pointers
* Also stores chunk embeddings and a lightweight index

5. **Retrieval & Rerank**

* Hybrid retrieve candidates (sparse-ish + dense)
* Rerank by relevance × importance × recency/pins

6. **ContextPacker**

* Deterministic token budget allocation (recent window + memory header + expansions)

7. **LLM API Client**

* One main call per user message (answer + memory_ops)
* Optional second cheap call for memory_ops if needed

8. **Validator**

* Enforces provenance and conflict/supersede rules
* Rejects hallucinated memories

---

## 2) Data schemas (lock these first)

### Transcript turn

* `turn_id`, `role`, `text`, `created_at`

### Chunk

* `chunk_id`, `turn_id`, `text`, `start_char`, `end_char`, `embedding`, `lexical_weights?`

### MemoryItem (the long-term unit)

* `mem_id`
* `type`: `Decision | Preference | Constraint | Entity | Definition | Task | Correction`
* `content`: 1–2 lines canonical form
* `details`: optional (short)
* `provenance`: `{turn_ids: [...], chunk_ids: [...], quote_spans: [...]}`  ✅ required
* `status`: `active | superseded | deleted`
* `supersedes`: mem_id?
* `pins`: `user_pinned | system_pinned | none`
* `heat`: `HOT | WARM | COLD`
* `energy`: float 0..1 + decay stats (`created_at`, `last_access_at`, `half_life_s`)

### MemoryOps (LLM output schema)

* `create[]`, `update[]`, `supersede[]`, `delete[]`, `pin[]`
* Every op must include **provenance** (turn/chunk refs + quote)

This one rule is what keeps a pure-LLM API approach from drifting.

---

## 3) Retrieval + ranking (bge-m3 does the heavy lifting)

### Two internal scores

* **Relevance(query, item)**: “useful right now”
* **Importance(item)**: “worth keeping hot / worth surfacing”

### Candidate generation (fast)

* Use one (or both):

  * **Dense ANN**: cosine(query_emb, chunk_emb)
  * **Sparse-ish**: BM25 over chunk text *or* bge-m3 lexical weights

Pick top **N=50–200** candidates.

### Rerank (cheap deterministic scalar)

For each candidate memory/chunk:

```
score = wR*relevance + wI*importance + wP*pin_boost + wT*recency - wS*superseded_penalty
```

**Importance** (no big model needed):

* baseline: rule/features (decisions, constraints, corrections, IDs/numbers, named entities)
* plus optional: a tiny LLM API “classify importance + type” during ingestion (very short prompt, low tokens)

**Relevance**:

* `cos_sim(query_emb, item_emb)`
* optionally blend lexical overlap score

---

## 4) Context packing (deterministic, budgeted)

Set hard budgets (example defaults):

* **Recent window:** 4,500 tokens
* **Memory header:** 600–1,200 tokens (typed bullets)
* **Expansions:** 0–800 tokens (verbatim quotes only when needed)

Order:

1. System / policies
2. “Memory header” (compact, typed, *only active + pinned + top-ranked*)
3. Recent turns
4. Expanded snippets (only for top few items when precision matters)

This makes cost stable and avoids surprise token spikes.

---

## 5) Ingestion loop (every turn)

1. **Store transcript**
2. **Chunk** (sentence→merge into 80–300 token chunks)
3. **Embed chunks** (bge-m3)
4. **Extract candidate memory items**

   * V1: rules only (fast to ship)
   * V2: one short LLM API call returning MemoryOps candidates
5. **Validate & apply**

   * Must reference actual turn/chunk quotes
   * Corrections must **supersede** old items
6. **Update energy/heat**

   * Newly created important items enter HOT
   * Others start WARM
   * Decay runs lazily on access

---

## 6) Storage & indexing (practical choices)

* **SQLite** for transcript + memory items (easy, robust)
* **ANN index**: FAISS or hnswlib (desktop/server-friendly)
* Store embeddings in:

  * SQLite blob table or separate mmap file + ID mapping

Cold-tier compression later:

* cluster summaries in SQLite + separate centroid index

---

## 7) Build phases (ship in order)

### Phase 0 — scaffolding (day 1–2)

* TranscriptStore + Chunker + bge-m3 embedding service
* Simple “retrieve chunks by similarity” demo

### Phase 1 — minimal memory manager (day 3–5)

* MemoryItem schema + pinned items
* ContextPacker (stable budgets)
* Retrieval from HOT + WARM

### Phase 2 — automatic memory extraction (week 2)

* Add LLM API “memory_ops” output (strict JSON)
* Validator + supersede logic
* Start tracking types (Decision/Constraint/Preference)

### Phase 3 — decay + tiering (week 2–3)

* Energy/half-life + HOT/WARM/COLD promotion/demotion
* Access reheating

### Phase 4 — cold summaries (week 3+)

* Cluster older items
* “Era summaries” (5–10 bullets per cluster)

### Phase 5 — eval harness (parallel)

* Scripted conversations + QA probes
* Metrics: token cost, recall accuracy, correction robustness

---

## 8) What runs locally vs via API

### Local

* chunking
* embedding + indexing (bge-m3)
* retrieval + scoring
* memory state machine (tiers/decay)
* context packing + budgets

### LLM API

* user-facing response generation
* structured memory_ops extraction (optional second call)

---

## 9) Default operating parameters (good starting values)

* Chunk size: **120–250 tokens** (merge adjacent sentences)
* Candidate retrieval: **top 100 chunks**
* Memory header cap: **~900 tokens**
* Expansions: **max 3** items, **≤250 tokens each**
* HOT cap: **~20 items**
* Heat thresholds: HOT ≥ 0.75, WARM ≥ 0.25, else COLD
* Pins override decay

---

If you want, next I can give you:

* the exact **prompt templates** for (A) main answer call and (B) memory_ops call,
* the **JSON schema** for MemoryOps (Pydantic-ready),
* and a minimal folder layout + component interfaces so you can start coding immediately.
