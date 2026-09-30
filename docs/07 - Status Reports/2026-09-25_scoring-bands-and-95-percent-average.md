# Scoring bands and the 95% average

**Date:** 2026-09-25.  
**Status:** Saved interpretation of completed results; no new model run.  
**Evidence:** [All 87 misses source review](../10%20-%20Research%20Log/256%20-%202026-09-25%20-%20Raw%20context%20control%20for%20all%2087%20campaign%20misses.md).

**Our evaluated memory-plus-reader system has a 95.3% source-review-adjusted
average across ten million-token histories: approximately 95%.** Each history
has 100 questions, so the mean of the ten history percentages equals the pooled
953/1,000 result. This describes the evaluated system using the
`codex_sdk/gpt-5.6-sol` reader; it is not an average across different model families.

The scoring bands keep the original strict result and the subsequent review
visible:

| Scoring basis | Correct / population | Score |
|---|---:|---:|
| Original automated grading | 913 / 1,000 | 91.3% |
| Original passes plus 40 flagged answers accepted against source | 953 / 1,000 | **95.3%** |
| Conditional ceiling if all nine unresolved reviews were accepted | 962 / 1,000 | 96.2% |

The last row is a sensitivity bound, not an achieved score or confidence interval.
The adjusted row leaves all nine unresolved answers outside the numerator.
The review labels 38 original memory answers incorrect, seven ambiguous and two
invalid. It does not count answers newly recovered by the raw-context control as
memory successes.

| History | Original correct | Source-review-adjusted correct |
|---|---:|---:|
| 01 | 94 | 97 |
| 02 | 85 | 91 |
| 03 | 88 | 94 |
| 04 | 95 | 96 |
| 05 | 93 | 99 |
| 06 | 93 | 97 |
| 07 | 88 | 92 |
| 08 | 93 | 96 |
| 09 | 89 | 95 |
| 10 | 95 | 96 |
| **Average percentage** | **91.3%** | **95.3%** |

The practical conclusion is that the original 91.3% score understated the
quality recognized by the subsequent source review. The approximately 95%
average is a useful, evidence-backed description when labeled
**source-review-adjusted**. This adjustment assumes the original 913 passes
remain valid: only the 87 flagged misses received the new review, whose judgments
also contain documented ambiguities. It does not replace the sealed original
benchmark or establish the score of the later cap-8 policy or the separate
engineering/research artifact battery.

Rechecked directly from the [sealed review report](../../eval_results/native-spine-ten100-misses-raw-control-20260925-r1/report.json),
SHA-256 `cbcf04e191cce9fc63ac2c0ebd735fa8c109965fe113ef77a79347c13f62e203`.
The ten history totals above reproduce 913 original passes and 40 accepted
corrections without any new provider calls.
