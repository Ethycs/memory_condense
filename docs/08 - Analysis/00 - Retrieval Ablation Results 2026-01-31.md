# Retrieval Ablation — dense k=10 vs no-memory baseline

**Status:** Living Document — data from 2026-01-31 runs; **stale by ~6.5 months** as of 2026-08-14, and predates everything merged in `f3edc91`
**Data:** 4 JSON files in `eval_results/` (gitignored — local-only; numbers preserved here deliberately)

## Executive summary (the verdict)

**Dense retrieval (k=10, bge-m3, chunk 120–250) adds +0.30 mean judge score and +13.1pp Recall@4 on a 283-turn conversation, and nothing on a 27-turn one.** Retrieval's value is real and concentrated exactly where theory (H2, `00 - Theory`) predicts: turns whose dependencies fall outside the 4-turn recent window.

## Run pair A — long conversation (most recent, 2026-01-31 ~04:1x UTC)

Corpus: `Genericity_of_Singularities_6892a425_2025-09-09….md`, 283 turns, 137 scored. Config: chunk 120–250, ef_search 50, Haiku responder+judge.

| Run | Mean score | Recall@4 | Score distribution 1/2/3/4/5 |
| --- | --- | --- | --- |
| `eval_120-250_k0_ef50_20260131_041148.json` (baseline) | 3.7445 | 64.96% | 1 / 15 / 32 / 59 / 30 |
| `eval_120-250_k10_ef50_20260131_041341.json` | **4.0438** | **78.10%** | 0 / 7 / 23 / 64 / 43 |
| **Δ** | **+0.299** | **+13.1pp** | worst tail (1s+2s) 16 → 7 |

Note the shape: retrieval doesn't just lift the top — it halves the failure tail.

## Run pair B — two shorter conversations (2026-01-31 ~02:40/03:11 UTC)

Corpus: `99__Observation__Mathematical_Convergence….txt` (27 turns, 13 scored) + `AI_Citation_Relevance_Check….md` (146 turns, 72 scored), `max_conversations=2`.

| Run | Mean | Recall@4 | Per-conversation means (.txt / .md) |
| --- | --- | --- | --- |
| `…k0…031126.json` (baseline) | 4.1176 | 84.71% | 4.31 / 4.08 |
| `…k10…024030.json` | **4.3294** | **90.59%** | 4.31 / **4.33** |
| **Δ** | **+0.212** | **+5.9pp** | 0.00 / +0.25 |

The gain lives entirely in the 146-turn conversation; the 27-turn one is identical both ways — it never outgrows the recent window.

## Position-bin analysis (added 2026-08-14 — H2 needs qualifying)

Run pair A re-analysed with `eval/analysis.py` (`--compare`, offline, zero API cost). Turn positions are bucketed into five bins relative to the conversation's own length:

| Position bin | Baseline mean | k=10 mean | Δ | n |
| --- | --- | --- | --- | --- |
| 1 (earliest 20%) | 3.750 | 3.964 | **+0.214** | 28 |
| 2 | 4.074 | 4.370 | **+0.296** | 27 |
| 3 (middle) | 3.357 | 3.821 | **+0.464** | 28 |
| 4 | 3.815 | 4.148 | **+0.333** | 27 |
| 5 (latest 20%) | 3.741 | 3.926 | **+0.185** | 27 |

**This contradicts the naive reading of H2.** The theory doc predicts retrieval's contribution grows with depth, so bin 5 should show the largest gain. It shows the *smallest* — smaller even than bin 1. The gain peaks in the middle of the conversation and falls away at the end.

Two candidate explanations, neither yet tested:

1. **Bin 3 is the hardest baseline, not the best treatment.** Note the baseline column: bin 3 is where the no-memory run scores worst (3.357). Retrieval may simply have the most headroom there rather than the most value. The treatment means are much flatter (3.82–4.37) than the baseline means (3.36–4.07), which is consistent with retrieval *levelling* difficulty rather than *increasing* in usefulness.
2. **Late turns may be locally coherent.** The end of a long conversation is often a wrap-up that depends on the immediately preceding turns — exactly what the 4-turn recent window already supplies — so there is little for retrieval to add.

Explanation 1 is the more likely and the more deflating: it would mean the depth-vs-gain story is partly an artifact of where the baseline happens to struggle. **Do not cite H2 as confirmed without re-running this on more than one conversation** — n≈27 per bin from a single transcript is not enough to separate these.

## Interpretation

1. **H2 is directionally supported across conversations but not within them**: the gain scales with total conversation length (0 at 27 turns, +0.25 at 146, +0.30 at 283), yet within the 283-turn conversation it does *not* rise with position — see the bin table above.
2. **Baseline inflation**: even k=0 scores 3.74–4.12 because the judge rewards generic-but-plausible responses and both judge and responder are Haiku while ground truth is a stronger Claude. Absolute values are not comparable across models; **only within-pair deltas are load-bearing.**
3. **What we cannot claim from this data**: any external competitiveness (no common benchmark), any efficiency numbers (no token/latency instrumentation), any parameter optimality (sweep never run).

## Next measurements (in value order)

1. ~~Score-vs-position curve from the already-captured `scores_by_position`.~~ **Done 2026-08-14** — see the bin table above. It did not confirm what it was expected to confirm, which makes it the most useful thing in this document.
2. **Re-run pair A** with the fixed harness. This is now both possible and necessary: the models these numbers were produced with were retired on 2026-02-19, and the judge is no longer the same model as the responder, so every absolute score above is from a configuration that can no longer be reproduced. Treat the table as historical.
3. **Repeat the bin analysis across several conversations** to separate "retrieval helps most where the baseline is weakest" from "retrieval helps most mid-conversation". One transcript cannot distinguish them.
4. **Dense vs hybrid** on the same corpus (`--hybrid`) — BM25 blending is wired but its effect is unmeasured.
5. The **54-config sweep** (3 min × 3 max × 3 k × 2 ef; no combinations are skipped since 180 < 200) — replaces guessed defaults (120–250, k=10, ef 50) with measured ones.
