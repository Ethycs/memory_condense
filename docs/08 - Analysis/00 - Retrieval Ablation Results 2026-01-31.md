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

### RETRACTION (2026-08-14, later the same day): the bin table does not replicate

Pair B had never been binned. It is free, so it was. Result:

| Position bin | Baseline | k=10 | Δ | n |
| --- | --- | --- | --- | --- |
| 1 (earliest 20%) | 3.78 | 3.94 | +0.17 | 18 |
| 2 | 4.47 | 4.53 | **+0.06** | 17 |
| 3 (middle) | 4.12 | 4.29 | +0.18 | 17 |
| 4 | 4.12 | 4.53 | **+0.41** | 17 |
| 5 (latest 20%) | 4.12 | 4.38 | +0.25 | 16 |

**Neither the H2 story nor its counter-explanation survives.** Pair A peaks in the middle with the final fifth *smallest*; pair B peaks in the **fourth** fifth with the final fifth **second-largest**. And explanation 1 fails on pair B too: its weakest bin (+0.06) has the *highest* baseline (4.47), while three bins tied at baseline 4.12 span +0.18 to +0.41 — identical baselines, deltas differing by 2×.

The arithmetic that should have been done before publishing the first table: per-turn scores are integers on 1–5 with SD ≈ 0.9, so at n ≈ 17–28 per bin the standard error of a bin mean is ≈ 0.2. **Every bin-to-bin difference in both tables is inside noise.** The aggregate deltas are not — SE ≈ 0.08 at n=137 and ≈ 0.10 at n=85, making +0.30 and +0.21 a 2–4 SE effect.

So: **the aggregate ablation result stands; the position-bin decomposition is withdrawn.** Item 1 under "Next measurements" called this table "the most useful thing in this document." It was the least — a five-way split of a sample that only supports one number. H2 remains untested, in either direction.

*Reproduce:* `pixi run python -m memory_condense.eval --compare eval_results/eval_120-250_k0_ef50_20260131_031126.json eval_results/eval_120-250_k10_ef50_20260131_024030.json` — offline, no API key.

## Interpretation

1. **H2 is weakly supported across conversations and untested within them**: the gain scales with total conversation length (0 at 27 turns, +0.25 at 146, +0.30 at 283) — three points, so suggestive at best. The within-conversation claim is withdrawn; see the retraction above. Three conversations cannot establish a trend either, and two of the three points come from the same run pair.
2. **Baseline inflation**: even k=0 scores 3.74–4.12 because the judge rewards generic-but-plausible responses and both judge and responder are Haiku while ground truth is a stronger Claude. Absolute values are not comparable across models; **only within-pair deltas are load-bearing.**
3. **What we cannot claim from this data**: any external competitiveness (no common benchmark), any efficiency numbers (no token/latency instrumentation), any parameter optimality (sweep never run).

## Next measurements (in value order)

1. ~~Score-vs-position curve from the already-captured `scores_by_position`.~~ ~~**Done 2026-08-14** — the most useful thing in this document.~~ **Retracted the same day**: it does not replicate on pair B, and every bin-to-bin difference is inside noise. See the retraction above.
2. **Re-run pair A** with the fixed harness. This is now both possible and necessary: the models these numbers were produced with were retired on 2026-02-19, and the judge is no longer the same model as the responder, so every absolute score above is from a configuration that can no longer be reproduced. Treat the table as historical.
3. **Test H2 with a design that can actually detect it** — enough scored turns that a per-bin SE is well under the effect you are looking for. At SD ≈ 0.9 and a hypothesised per-bin Δ of 0.1, that is n ≈ 300+ *per bin*, i.e. thousands of scored turns, not one 283-turn transcript. The cheaper route is the benchmark path (`02 - Implementation/01`), where LoCoMo yields ~200 graded questions per haystack ingest.
4. **Dense vs hybrid** on the same corpus (`--hybrid`) — BM25 blending is wired but its effect is unmeasured.
5. The **54-config sweep** (3 min × 3 max × 3 k × 2 ef; no combinations are skipped since 180 < 200) — replaces guessed defaults (120–250, k=10, ef 50) with measured ones.
