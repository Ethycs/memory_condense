# Session handoff — wire-ups, measurement, and the chunking finding

**Date**: 2026-08-15
**Branch**: `main`, clean, 9 commits from `f2be62d` to `f77781b`
**Tests**: 523 passing, 13 slow deselected
**Cost incurred**: $0. Every measurement below is offline and keyless.

## What changed

| Commit | Substance |
| --- | --- |
| `e11802e` | Benchmark judge capped Sonnet 5 at `max_tokens=256`, which `judge.py` documents as wrong (adaptive thinking counts against the cap) then read `.content` unguarded. Would have failed a paid run mid-flight. Plus three free measurements |
| `0d86038` | **Decay was decorative.** `rank_score` had no energy term; energy was read only for display. Collapsed two decay kernels into one, saturating reheat, HOT cap enforced |
| `615b78f` | Exact-duplicate memories merge instead of inserting (schema v3). 12.6% of extracted items on the local corpus were duplicates |
| `b62166a` | `LLMExtractor` bound to a provider; the LLM-boundary axiom is now a test, not a doc claim |
| `f73f131` | `--mode memory` routes the responder through `build_context`. **Until this, `ContextPacker`/`MemoryStore.retrieve`/`rank_score`/`decay` were exercised by no run** |
| `88af21e` | `--answer-recall`: free, keyless retrieval measurement |
| `139e6cd` | LoCoMo loader was discarding the session timestamps its largest question category asks about |
| `f77781b` | Pooled-span retrieval + cost instrumentation |

## The headline finding

`Chunker.chunk_turn` chunks **one turn at a time and never merges across turns**, so `min_tokens=120` is unreachable for short turns. Every number this project had ever produced came from long-form monologue, where that never showed.

| corpus | median chunk | context at k=10 |
| --- | --- | --- |
| Long-form (all prior results) | 227 tok | ~2,270 tok |
| LoCoMo (real dialogue) | **27 tok** | **~222 tok** |

Same code, same settings, **8.4× less context**. On this repo's own build transcript, 60% of turns fall under 120 tokens while tool output supplies 71% of chunks — the index is mostly tool noise and the conversation carrying decisions is the part shredded.

**Nothing regressed.** `retrieval.py`, `chunker.py`, `embedding.py`, `lexical.py` are byte-identical to the pre-session merge; verified by diff. The evidence base was simply narrower than it looked.

## Pooled-span retrieval — replicated

`span_query` pools contiguous chunks up to a token target, matches the pooled vector, returns **member chunks** (so provenance and `ContextPacker` are untouched, and no schema change was needed).

Replicated across 4 LoCoMo samples, n=757:

| arm | pooled recall | mean ctx | % of ceiling |
| --- | --- | --- | --- |
| dense k=10 | 10.3% | ~209 | 19–30% |
| stratified 110+220 ×2 | 23.4% | ~667 | 38–67% |
| stratified ×3 | 26.4% | ~1,005 | 46–73% |

Better on **every sample individually**. At matched token budget the gap is ~2.2×.

Three things that are load-bearing and easy to break:

1. **Retrieval must stratify per level.** Cosine is not length-invariant — per-turn chunks average 0.678 top-10 cosine vs 0.602 for spans, because short text has fewer competing topical directions. One mixed pool lets small chunks crowd out every span: measured, recall collapsed 21.6% → 6.0%. `test_shorter_text_scores_higher_cosine_than_a_span_containing_it` pins the inequality.
2. **Levels are token targets, not chunk counts.** Counted in chunks, `span=4` is ~110 tokens on dialogue and ~900 on long-form prose — one setting helping one corpus and wrecking the other.
3. **Bigger is not better.** ~440-token spans score *below* ~220 at 2.2× the cost; a mean vector washes out once a span straddles topics.

## Open, in priority order

1. **Span vs the long-form corpus.** Never measured there. Token targets should make it safe, but that is an assumption and this session is an argument against shipping those. Free.
2. **Re-run memory vs dense on top of span retrieval.** The current verdict — memory costs +75% tokens for +0.8pp recall, 26.4 vs 42.6 per 1k — was measured over the chunking defect and deserves a rematch. Free.
3. **conv-30 is an outlier** for span (37.5% of ceiling vs 63–73%). Unexplained.
4. **The paid QA run** (~$1.70) — the only thing producing a number comparable to published work. Worth doing *after* 1 and 2.
5. **Phase 4 remains unbuilt and the signal is against it**: 4.8% of answers are held by a memory item, 0.0% survive to day 14.

## Traps worth carrying forward

- **`recall_per_1k_tokens` is gameable by retrieving nothing.** A failed experiment scored 102.2/1k — the best of the session — while finding 6% of answers. Never read it without absolute recall beside it.
- **Never compare per-token efficiency across different operating points.** `k=10` looks efficient purely because it sits where marginal returns are highest.
- **LoCoMo's verbatim-containment ceiling is 33–47%, not 100%** — many gold answers are derived, not quoted. A recall number without its ceiling is misleading.
- **We are not comparable to published results.** Ours is retrieval-stage containment; theirs is answer-stage judge accuracy. `docs/README.md` still correctly says competitiveness is unanswerable.
- **GPG signing times out** when the passphrase cache expires. Retry usually works; the work stays staged.

---

**Verification block**:

```powershell
git log --oneline -1                      # expect f77781b
pixi run -e dev pytest -q -m "not slow"   # expect 523 passed, 13 deselected
pixi run python -m memory_condense.eval --answer-recall data/locomo10.json `
    --benchmark-format locomo --max-samples 1 --mode span    # free, no key
```

`data/locomo10.json` is gitignored; re-fetch from `snap-research/locomo` if absent.
