# memory_condense Vocabulary Reference

**Status**: Living Document
**Date**: 2026-08-14
**Supersedes**: the revision in which the entire Lifecycle section and most Retrieval terms were marked *(planned)*

One-line scope: canonical terms as used in this repo's code, docs, and eval results. Terms marked *(planned)* exist in design docs but not in code — as of this revision there is exactly one left.

## Core objects

| Canonical | Also seen as | Meaning |
| --- | --- | --- |
| turn | message | One (role, text) unit in a conversation; append-only row in `turns` |
| chunk | span, segment | Contiguous 120–250-token (cl100k) slice of a turn, with char-span provenance |
| memory item | fact, memory unit | Typed, compact long-term unit with mandatory provenance — `MemoryItem` in `schemas.py`, stored in `memory_items` |
| memory type | — | One of `Decision · Preference · Constraint · Entity · Definition · Task · Correction` |
| provenance | — | `(turn_id, quote, chunk_id?)` pointer from a memory item back to the transcript that justifies it |
| provenance quote | quote | The exact substring of the cited turn that justifies the memory. Matched **whitespace-normalized only** — no case folding, no punctuation stripping, no fuzzy match. A paraphrase is rejected; that is the point |
| recent window | — | Last N turns always included in the prompt (eval default 4; `ContextPacker` caps the section at 4,500 tokens) |
| memory header | memory context | The typed-bullet block of ranked memory items placed above the recent turns (`ContextPacker`, 900-token cap). In the *eval replay* prompt the equivalent block is `[Memory i]` raw retrieved chunks, not memory items |
| expansion | verbatim excerpt | A verbatim chunk quote appended when precision matters; at most 3, each ≤250 tokens, 800 tokens total |
| packed context | — | `PackedContext`: messages + per-section `token_counts` + `dropped` counts. Nothing is truncated without being counted |

## Retrieval

| Canonical | Also seen as | Meaning |
| --- | --- | --- |
| relevance | similarity | Dense: `1 − cosine_distance(query_emb, chunk_emb)`. For memory items, cosine mapped `(cos + 1) / 2` into `[0, 1]` |
| importance | salience | Query-independent keep-priority in `[0, 1]`; the rule-based extractor assigns `0.8` to decisions/constraints/corrections, `0.5` otherwise |
| dense retrieval | ANN, hnswlib search | `SimilarityRetriever.query` — the untouched baseline path the ablations compare against |
| BM25 | lexical search, sparse retrieval | Okapi BM25 (`k1=1.5`, `b=0.75`) over the `chunk_terms` inverted index; `LexicalIndex` in `lexical.py` |
| hybrid retrieval | dense ∪ lexical | `SimilarityRetriever.hybrid_query` — `candidates` pulled from each side, each side min-max normalized, then blended |
| alpha (`α`) | dense weight | Blend weight in `blend_hybrid(dense, lexical, α) = α·dense + (1−α)·lexical`. Default `0.65`. `α=1.0` reproduces dense ordering; `α=0.0` is pure BM25 |
| min-max normalize | — | Scaling each side's raw scores into `[0, 1]` before blending. A flat input maps to all-`1.0` ("no signal"), not all-`0.0` |
| rerank scalar | rank score | `wR·relevance + wI·importance + wP·pin_boost + wT·recency − wS·superseded_penalty` (`ranking.rank_score`); defaults 1.0 / 0.3 / 0.5 / 0.2 / 1.0 |
| lexical weights | sparse weights | `chunks.lexical_weights` — now **populated** with the chunk's term→frequency map by `add_chunks` (formerly always NULL) |
| term_count | document length | `chunks.term_count`: number of BM25 tokens in the chunk, the `\|d\|` in the BM25 denominator. NULL ⇒ not lexically indexed |
| ef_search | ef | hnswlib query-time beam width (default 50) |
| k | top-k | Chunks retrieved per query; `k=0` disables retrieval (baseline) |
| candidates | candidate pool | How many chunks each side of a hybrid query contributes before blending (default 100) |

## Lifecycle

| Canonical | Meaning |
| --- | --- |
| energy | `[0, 1]` decaying keep-score; `energy × 0.5^(elapsed / half_life_s)`, computed lazily on read — no timer, no background job |
| half-life | Per-item decay constant; default 7 days (`604800 s`) |
| HOT / WARM / COLD | Heat tiers by energy thresholds `0.75` / `0.25`. Derived on read, never stored |
| reheat | The `+0.25` energy boost applied when an item is retrieved (`decay.reheat`, applied by `MemoryStore.touch`) |
| seed energy | Starting energy for a new item: `0.8` when `importance ≥ 0.7` (enters HOT), else `0.5` (WARM) |
| pin | User/system flag exempting an item from decay entirely. `user_pinned` boosts rank by 1.0, `system_pinned` by 0.6 |
| supersede | A correction replacing an older item: new row with `supersedes` pointing back, old row set to `superseded`. **Never a deletion** |
| soft delete | `status = 'deleted'`. The row survives so the audit trail to the transcript stays walkable |
| memory ops | `MemoryOps` — the `create / update / supersede / delete / pin` batch an extractor proposes and the Validator gates |
| era summary *(planned)* | Cold-tier cluster summary + centroid index (design Phase 4). No code |

## Eval

| Canonical | Also seen as | Meaning |
| --- | --- | --- |
| self-replay | conversation replay | Regenerating each assistant turn from memory + recent window, scored vs the recorded one |
| teacher forcing | — | Ingesting the *actual* recorded turns after scoring, never the generated ones |
| judge score | score | 1–5 LLM-judge rating of generated vs actual response; parse failure ⇒ 1 |
| Recall@4 | recall_at_4 | Fraction of scored turns with judge score ≥ 4 |
| k0 baseline | no-memory baseline | Run with `--k 0`: recent window only, identical code path |
| ablation pair | — | Matched k=0 / k=N runs on the same corpus; the delta isolates retrieval |
| UsageStats | usage | Token + latency accounting for one or more LLM calls: `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `elapsed_s`, `calls`. Adds associatively, so run totals are exact sums of turn totals |
| context tokens | `context_tokens` | cl100k count of the assembled responder prompt for one turn — measures what the prompt actually costs, independent of what the provider reports |
| benchmark sample | `BenchmarkSample` | One public-benchmark record: a haystack of turns plus the questions asked about it |
| QA probe | benchmark question | One `(question, gold answer, category, evidence)` asked against an ingested haystack |
| F1 | token F1 | SQuAD-style token-level F1 between predicted and gold answer after normalization (lowercase, strip punctuation, drop articles, collapse whitespace). The standard LongMemEval/LoCoMo string metric |
| exact match | EM | Normalized string equality between predicted and gold answer |
| judge accuracy | — | Fraction of answers a semantic-equivalence judge marks CORRECT (`--use-judge` only; `null` otherwise) |
| position bin | `scores_by_position` bin | Turn positions bucketed relative to each conversation's own length, so short and long conversations both populate every bin (`analysis.binned_scores`, default 5) |

### Example (real, from `eval_results/eval_120-250_k10_ef50_20260131_041341.json`)

```json
{"aggregate_mean_score": 4.0438, "aggregate_recall_at_4": 0.7810}
```

means: over 137 scored turns, mean judge score 4.04 and 78.1% of turns scored ≥ 4. Note what is *absent* — no `usage` block: that run predates `UsageStats`, so it loads with zeros in every token and latency column.

---

**Verification block**: run

```powershell
pixi run python -c "import memory_condense as m; print(len(m.__all__)); print(sorted(m.__all__))"
```

Expect **25** exported names. Every *concept* behind them should be findable above (`MemoryOps` → "memory ops", `heat_for` → "HOT / WARM / COLD", `ContextBudget` → "packed context"). If an export names a concept with no row here, the vocabulary has drifted — add the row in the same change that added the symbol, and decide whether the term belongs in Core objects, Retrieval, Lifecycle, or Eval before writing it.
