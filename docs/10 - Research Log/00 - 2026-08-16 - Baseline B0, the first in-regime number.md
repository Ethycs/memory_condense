# B0 — the first in-regime baseline

**Status**: Baseline of record (operator-declared, 2026-08-16)
**Corpus**: `build-session-8f7f7561` — this project's own Claude Code build session, snapshotted 2026-08-16
**Probe**: `bs-probe-v1`, 39 questions ([data/2026-08-16-build-session-baseline/](data/2026-08-16-build-session-baseline/))
**Cost**: $0. Local, keyless. One-time ingest 937 s (bge-m3 on CPU, ~318 tok/s).

## The baseline

> **B0: hybrid k=10 — 92.3% recall at 1,533 context tokens (0.50% of a 305k-token transcript), 60.2 recall-pts per 1k tokens, per-turn cost flat.**

36 of 39 verified session facts recovered. Chunk-level ceiling 100% (chunking split no probe answer). Any future retrieval change is measured against this row on this probe.

| arm | recall | ctx tok | recall/1k |
| --- | --- | --- | --- |
| ceiling (turns / chunks) | 100% / 100% | 305,103 / 297,767 | 0.33 |
| random k=40 | 2.6% | 4,054 | 0.6 |
| recent k=40 | 35.9% | 6,545 | 5.5 |
| dense k=10 | 87.2% | 1,573 | 55.4 |
| **hybrid k=10 (B0)** | **92.3%** | **1,533** | **60.2** |
| hybrid k=50 | 94.9% | 7,540 | 12.6 |
| span x2 | 66.7% | 477 | 139.8 |
| span x4 | 79.5% | 958 | 83.0 |
| memory k=10 (assembled) | 64.1% | 481 | 133.3 |

Marginal cost along the frontier: span x2 → x4 ≈ 27 pp/1k; → hybrid k=10 ≈ 22 pp/1k; → hybrid k=50 ≈ **0.4 pp/1k**. The knee is B0. The per-1k column alone would rank span x2 first — the standing trap: never read the ratio without absolute recall beside it.

**Session economics** (R5): 2,420 turns, t=126, crossover N\*=2B/t≈98 — the session ran 25× past it. Bounded ≈ 3.7M cumulative tokens vs ≈ 369M for resend-history: **99% reduction for 92% of the answers**, per-turn cost constant.

## Method (reproducibility)

1. **Snapshot first** — the session JSONL was copied *before* any probe question was authored, so authoring turns cannot appear in the corpus (contamination guard). The 13 MB snapshot itself is deliberately not committed; scripts + probe are.
2. **Parse** (`cc_parse.py`): user text and assistant text as turns; `tool_use` (input truncated 300 ch) and `tool_result` (truncated 2,000 ch) as system turns; thinking blocks, meta/caveat records, command wrappers, sidechains excluded. Yield: 2,420 turns, 305,103 tokens — user 7%, assistant 14%, tool_use 26%, tool_result 53% (tool traffic 79%, matching the earlier 71%-of-chunks census).
3. **Probe** (`cc_questions.py` → `cc_verify.py`): 48 hand-authored Q/A pairs across five session phases; kept only answers verbatim-present (SQuAD-normalized containment) in ≥1 turn and ≤25 turns (ubiquity filter). 39 survive; most answers live in 1–5 turns of 2,420.
4. **Measure** (`cc_bench.py`): ingest via `ingest_sample`, all arms via `eval.recall._assemble`, `reheat=False` throughout.

## Findings beyond the headline

**The regime flip — last session's prediction confirmed.** The falsifiable prediction from the crossover model ("span's gain should vanish where chunks are already evidence-sized") held, and overshot: the arm ranking *inverted* between corpora.

| | LoCoMo conv-26 (t=31, 27-tok chunks) | build session (t=126, 81-tok chunks) |
| --- | --- | --- |
| span vs hybrid | span wins every matched budget | hybrid wins by 13–26 pp |
| dense | redundant (+0.0 pp to union) | 87.2%, near-frontier |
| recency | *worse than random* | 14× better than random |

No retriever dominates across regimes. `t` and chunk size are observable at runtime and predict the right arm: short-turn chat → span; agentic/long-form → hybrid. (Confound, stated: the probe is needle-type by construction, which favours hybrid/dense; corpus and question-type effects are not fully separated.)

**The assembled system trails its own retriever by 28 pp.** memory k=10 = 64.1% where raw hybrid k=10 = 92.3%: `ContextPacker.max_expansions=3` discards 7 of the 10 hits while using only 481 of its ~1,700-token budget. The C3 confound from the original plan, now quantified. Fix is R3's top work-list item.

**What the corpus cannot contain.** Probe answers that existed only inside Edit-tool payloads (`REHEAT_ONCE_PER_TURN`, `COALESCE(MAX(ordinal), 0)`, `idx_turns_ordinal`…) are **absent from the ingested corpus** — bounded tool records never carried them. The memory can only recall what the session surface said. Deployment property, now measured, dropped 6 of 48 authored questions.

**Misses.** The 2 questions missed by span x4 ∪ hybrid k=50 (94.9%) both have answers living in exactly one turn (`unrelated chatter`, `expensive operations`) — residual failures are true single-needle cases.

## Against published long-context recall (web data, 2026)

Corrected mapping after an operator catch — the probe is **one fact per query**, so 8-needle MRCR was the wrong anchor:

| probe slice | share | native model holds 92.3% to ≈ |
| --- | --- | --- |
| unique string + lexical overlap (NIAH-like) | ~2/3 | ~1M — no recall edge for us, **cost edge only (~200×)** |
| numeric needle in a distractor field (MRCR-like) | ~1/3 | ~32k–128k |
| semantic, no lexical overlap (NoLiMa-like) | ~0 sampled | ~2–8k (predicted edge, unmeasured) |

Reference points: MRCR v2 8-needle @128k — Gemini 3 Pro 77.0, GPT-5.1 61.6, Sonnet 5 81.5, GPT-5.6 Terra 93.5, Gemini 3.7 Flash 97.0; @1M best published ≈78 (Opus 4.6). NoLiMa: 10/12 models <50% of base by 32k. The honest headline is therefore the **cost claim**, not recall superiority — except on the distractor/semantic slices. Definitive test: ask a model the same 39 questions over the raw transcript (~$12 Haiku-class; ~$4 for the 13 distractor-numeric questions only, where divergence is expected).

## Caveats of record

1. Probe authored by the assistant that lived the session → selection bias toward memorable, distinctive strings. **B0 is an upper estimate**; organic questions will be vaguer.
2. n=39 → ±8 pp. Good for "does it work"; not for fine arm ranking.
3. Recency baseline flattered by probe composition (24/39 questions from the session's second half).
4. Tool results truncated at 2,000 chars — untruncated dumps would grow the corpus and likely *lower* every arm's recall share of it.
5. Needle-type questions; the semantic slice — where retrieval should beat native context hardest — is unsampled because verification demanded verbatim answers.

## What this session changed (context for the numbers)

Decay re-coordinated from wall-clock to turns (schema v4, `9aea4cd`) after establishing the spec was wrong from commit one (`249e4bb`); operating requirements R1–R6 written down (`12ca61b`). Full arc: [`06 - Roadmaps/01`](../06%20-%20Roadmaps/01%20-%20Delivering%20the%20Specified%20System.md), [`01 - Design/02`](../01%20-%20Design/02%20-%20Operating%20Requirements.md).

## Open, in priority order

1. **R3**: lift `max_expansions` to use the budget; re-measure the assembled system against B0 (expected to close most of the 28 pp).
2. **R5**: span-cache invalidation schedule (`add_chunks` clears per append — O(N)/turn in the span path).
3. Adaptive arm choice from observable regime (`t`, chunk size) — hybrid vs span per corpus (R4).
4. The $4 distractor-slice native comparison; the $12 full one.
5. A semantic probe (paraphrased questions, judge-scored) — the slice B0 cannot see.
