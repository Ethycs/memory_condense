# Validation-v3 provider-free retrieval audit

**Status:** completed provider-free retrieval audit of the exact frozen v3
treatment on all 100 validation questions. This is not an answer-accuracy run:
no responder, judge, Mem0, or other provider call was made. The result exposes
a development-to-validation retrieval gap and consumes this population as a
retrieval-analysis set; it must not now be tuned on and relabeled untouched.

**Depends on:** the
[`v3 retrieval freeze`](16%20-%202026-08-18%20-%20V3%20retrieval%20freeze%20and%20validation%20campaign.md),
the
[`locked treatment handoff`](17%20-%202026-08-18%20-%20Locked%20treatment%20handoff%20and%20discourse%20closure%20frontier.md),
and the tracked merger at
[`tools/merge_locked_v3_recall.py`](../../tools/merge_locked_v3_recall.py),
with the independent structural auditor at
[`tools/frozen_treatment_audit`](../../tools/frozen_treatment_audit).

## Result in one sentence

The earlier development campaign artifacts consistently report 10/10 positive
judge verdicts on a narrow selected slice, but they do not authenticate the
provider or judge executions or independently establish factual correctness.
Exact frozen v3 produced much weaker retrieval admission on the 100-question
validation population, while answer accuracy there remains unknown because
0/100 questions were sent to the responder or judge.

## Execution boundary

The audit ran offsets `0, 10, ..., 90` from the exact frozen source and the ten
previously prepared read-only cache pairs. Each shard contained ten questions
and approximately one million transcript-token proxies. Network access and
provider calls were disabled. Local BGE-M3 and Qwen prefix/choice execution
were allowed under the frozen policy; cache misses, writes, identity drift, or
receipt drift were fatal.

The ten shard reports are local ignored artifacts named:

```text
eval_results/validation-v3-offline-recall-offset-{000,010,...,090}.csv
```

The provider-free merger validated their common schema, locked population,
offsets, identities, question uniqueness, cache receipts, evidence metrics,
selector diagnostics, and the consistency of reported retained-state scalar
fields. It published:

```text
eval_results/validation-v3-offline-recall-campaign-v1.json
eval_results/validation-v3-offline-recall-campaign-v1.json.sha256
```

The campaign JSON intentionally contains aggregates and content-free
identities, not question text, answers, retrieved excerpts, or candidate
traces.

## Frozen identity chain

| Identity | SHA-256 |
| --- | --- |
| Frozen source commit | `bfa5b6daf6a5e61881ac10f0555e5d9972f9e1c2` |
| `src/memory_condense` implementation | `452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83` |
| Environment lock | `058083871240979257ada7ca4c71dd816fee64792b275ef11e4857c9f5ebba33` |
| Cleaned dataset | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Split manifest | `8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4` |
| Validation policy | `5263d5afd15298ec4088db9d6381ae243ddb685e9a3cf4d9892fc84e14fb9883` |
| Development selection artifact | `a82a3ffb2880121e3952f0e581c2affe199e48e2a3d0cdddf2fe09492b6e4a3e` |
| Retrieval identity | `08ffd89a8b30803a0d8121445c1d54171120b1f1e51c866d4015f2d36b87cbaf` |
| Merged campaign file | `806aa697d838c3602cd4d3a614ca03f6e6d7402ca88bb72a6070030d613ced36` |
| Merged campaign receipt | `924c0471eebcdeaaf9d0ba65905697b078b6972e1e417e4ddadfb1acc1256020` |
| Ten-shard input set | `87de6f54134e0fce2ec000cf1a63bfa83b286f2437ddcbf84f5643b271324c9f` |
| Population identity | `7a67aa6f43ffb94d487fb9184f871735bd9edac1974a3154898846d1140c83a1` |

The current organized v4 source tree has a different path-sensitive
implementation identity. This audit did not run current v4 code and does not
transfer v3 evidence to it.

## Aggregate retrieval observations

| Measure | Result |
| --- | ---: |
| Questions / unique IDs | 100 / 100 |
| Provider calls | 0 |
| Mean context tokens | 2,139.01 |
| Literal gold answer present in source history | 61% |
| Literal gold answer present in packed context | 48% |
| Mean best token F1 against gold answer | 0.118897 |
| Mean labeled evidence-source recall | 87.5834% |
| At least one labeled evidence source retrieved | 92% |
| Every labeled evidence source retrieved | 82% |
| Mean raw-graph evidence recall | 87.6667% |
| Every raw-graph evidence item retrieved | 83% |
| Scored answer-value components | 2/6 across two applicable questions |
| Selector calls / bypasses | 100 / 61 |
| Candidates inspected and classified | 3,459 |
| Local score-model forward passes | 451 |
| Routed frontiers audited/exhaustive | 39 / 39 |
| Selected scopes structurally complete | 1 |
| Post-coverage closure calls | **0** |
| Fallback/degraded executions | 0 |
| Reported retained selector/score-state fields | 100/100 recorded 0 bytes; not an absolute heap/state certificate |

`literal gold answer present` is deliberately not called accuracy. LongMemEval
answers can be paraphrased, normalized, assembled from several facts, or
derived temporally. In the development answer pilot, literal context recall
was only 50% while the campaign reported 10/10 positive judge verdicts.
Conversely,
source-level evidence recall cannot prove that the final packet contains the
right details in a usable form. Both metrics are diagnostic lower-level gates.

## Category profile

| Category | n | Mean evidence-source recall | All sources | Literal answer in context |
| --- | ---: | ---: | ---: | ---: |
| Knowledge update | 16 | 100.00% | 100.00% | 81.25% |
| Multi-session | 27 | 88.58% | 81.48% | 66.67% |
| Single-session assistant | 11 | 100.00% | 100.00% | 27.27% |
| Single-session preference | 6 | 66.67% | 66.67% | 0.00% |
| Single-session user | 14 | 92.86% | 92.86% | 78.57% |
| Temporal reasoning | 26 | 75.64% | 61.54% | 11.54% |

The largest weaknesses are preference and temporal questions. The diagnostic
profile is consistent with a routing/assembly problem, not evidence that the
answer model would necessarily fail every missed literal string.

## What changed relative to development

Nothing in the frozen treatment changed. That is the important finding.
Package reorganization, diffuse closure work, and Mem0 tooling are separate v4
or tool-tree changes and did not participate in this audit.

The final development replay had 100% labeled raw and packed source coverage
and 11/11 scored answer-value components on ten selected questions. The prior
answer-pilot artifact on the same small development slice reports 10/10
positive judge decisions. On the 100 validation questions, source coverage fell to
87.6% on average and complete-source recovery to 82%. The validation query
mix also never activated v3's narrow fixed-set post-coverage closure. This is
a generalization failure of the frozen admission/closure policy, not a newly
introduced code regression.

The accurate performance statement is therefore:

> v3 performed strongly on one selected ten-question development slice. Its
> exact frozen retrieval policy did not preserve that evidence-admission
> profile on the 100-question validation population. Answer accuracy on that
> population remains unmeasured.

## Cache and state post-audit

After execution, the audit rehashed every cache payload rather than trusting
the run reports:

- ten compiled and ten causal entries;
- forty SQLite/HNSW payloads totaling 3,034,010,240 bytes;
- every manifest and causal-to-compiled parent link;
- exactly three expected files per cache directory;
- no build, staging, WAL, SHM, journal, temporary, or partial remnants; and
- `retained_prompt_state_bytes=0` in every causal receipt.

All hashes matched the pre-run receipts. The result was
`CACHE_POST_AUDIT_OK`.

## Independent audit claim boundary

The follow-up `frozen_treatment_audit` tool reconstructs the locked
population and prompt identities, validates every shard/input/sample and
aggregate binding, checks exact provenance coordinates, and scans the cache
artifacts using a closed schema and content-addressed pre/post snapshots. Its
CLI also requires an externally supplied expected audit-tool digest and
publishes atomically without clobbering an existing result.

That evidence is deliberately narrower than an accuracy certificate. A
regression constructs an internally consistent synthetic report claiming
100% while using fabricated predictions and judge rows; the auditor accepts
the report's structural consistency but does not authenticate those provider
events or call the answers factually correct. Its receipt therefore states
that provider execution, judge execution, independent factual accuracy, and
retrieval replay are not authenticated. It also declines an absolute
zero-transformer-state certificate because permitted vector, ANN, and scalar
payloads cannot be proven semantically unrelated to hidden state by file
inspection alone.

## Merger regression found during the audit

The evaluation algorithm did not regress, but the first complete ten-shard merge
found an operational bug in the new audit tool: Python's default CSV field
limit rejected multi-megabyte candidate-trace fields. The merger now raises
the parser limit to a bounded 16 MiB only while reading, restores the process
default afterward, and has a 200,000-character regression test. This bug
blocked report assembly; it did not affect retrieval outputs or performance.

## Consequences for the next experiment

1. Do not spend provider calls on v3 merely to turn these retrieval
   diagnostics into an answer score. The retrieval admission profile makes a
   95% result risky and provider calls remain unauthorized.
2. Do not tune v4 against these 100 validation labels and then call this
   population untouched. Preserve the v3 reports as a failed-confirmation
   artifact.
3. Use the existing Qwen prefix-head QK/OV machinery to derive a bounded
   OV-transport semantic-change signal, form and refine episodes, and combine
   them with the new discourse-obligation closure and atomic packet path. Do
   not relabel that proxy as EM-LLM's autoregressive token-NLL surprise.
4. Freeze a new implementation epoch and evaluate deterministic/fixed,
   embedding-change, attention-surprise, and head-refinement ablations on a
   new development population.
5. Reserve a genuinely untouched confirmation population before any tuning,
   then run answerer/judge calls only after the provider-free retrieval gate
   is admitted.
6. Keep the controlled Mem0 comparison independent. Its production runtime is
   still deliberately NO-GO and no Mem0 score exists.

This audit lowers confidence in v3 reaching the primary goal, but it improves
the truthfulness of the project: the development artifact remains internally
consistent, its trust boundary is now explicit, and the next architecture has
a concrete failure mode to solve rather than an assumed accuracy claim.
