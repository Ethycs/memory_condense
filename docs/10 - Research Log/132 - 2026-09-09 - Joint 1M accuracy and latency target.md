# Joint 1M accuracy and latency target

**Date:** 2026-09-09  
**Status:** active; joint target unverified  
**Predecessor:** [131 - Real data user spine answer and judge results](131%20-%202026-09-09%20-%20Real%20data%20user%20spine%20answer%20and%20judge%20results.md)

The user clarified the actual objective: at least 95% answer accuracy on
1M-token memory with latency close to chatting directly with an API. The
three-source development pilot does not satisfy this objective. The active goal
requires accuracy and end-to-end latency on the same implementation and
population. The provisional latency allowance is 10%, pending an optional user
preference; measure median and p95 for visible time to first token and complete
response time. Neither offline construction nor cached answers count as live
query performance. Report cold ingest/setup separately.

## Authenticated starting point

`tools/assess_joint_1m_target.py` verifies all 100 question IDs, question hashes,
dated-question hashes, reference hashes, and prediction-to-judgment bindings
between the historical slow policy and the strongest recorded fast user-spine
full100 run. Their outer population hashes use different contracts; the row
identities match.

| Outcome | Questions |
| --- | ---: |
| Both correct | 72 |
| Slow policy correct, fast policy incorrect | 23 |
| Fast policy correct, slow policy incorrect | 1 |
| Both incorrect | 4 |

The slow policy is 95/100; fast user-spine is 73/100. Of the 23 lost answers,
10 are multi-session, 8 temporal reasoning, 3 knowledge updates, 1 user lookup,
and 1 preference. This is an analysis of different pipelines, not a causal
ablation. These outcome labels must never select production routes or supply
cached benchmark answers. Most slow rows preserve prior answers from an
expensive cumulative lineage; their already-correct answers cannot simply be
moved into ingest as though they were question-independent memory.

Report: `eval_results/joint-1m-target-assessment-20260909-r1/accuracy-gap.json`,
SHA `9719e970c5d927ccd85b43c45e74a670ce50e27794c13a57597fac3ebd2d5bbf`.
No new provider calls. This is already examined validation, not untouched
confirmation evidence. Confirmation200 remains unopened by this continuation.

## Corpus scale

Every source database was authenticated against the locked namespace manifest.
The ten namespaces contain 54,246 turns, including 24,518 user turns, across
4,805 source conversations. Each namespace is approximately 1.04M token
proxies; the total is 10,441,617. Whole-transcript role/date/text hashes found
4,805 distinct transcripts, so this inventory found no cross-namespace reuse.

Inventory: `eval_results/joint-1m-target-assessment-20260909-r1/corpus-inventory.json`,
SHA `12d3d8def648136fec03623a05da6b81837dbb0c3d3755cb8de0883e01f66a6f`.
The inventory reads every turn. No question, answer, score, or source relevance
filter selects its source population.

## Paired streaming diagnostic

`tools/benchmark_hot_api_latency.py` freezes all 200 requests before sending
any: one short API prompt and one previously sealed raw evidence packet for
each full100 question. Model, system policy, dated question, and 256-token output
cap are matched. Execution is serial, pair order alternates, and retries are
zero. Output lengths and semantic work can differ: the short no-memory arm is
a latency baseline, not an answer-accuracy control or a causal prefill ablation.

The new streaming measurement includes blocking stream creation, ignores
role-only and hidden-reasoning deltas for visible TTFT, requires a finish event,
and closes the stream. A live prompt-builder callback can include retrieval and
hydration in end-to-end timing. This diagnostic instead uses old packets, so
retrieval and hydration are explicitly excluded and it cannot pass the target.
It measures the API cost of the current evidence prompt. A future live path must
also compare against directly sending its identical hydrated prompt to isolate
retrieval overhead.

Root: `eval_results/full100-api-latency-20260909-r1`. Preflight SHA:
`f22ea11e1a863e90442e5957c03d11ed3edc28871483c98f8618a4aa47850f19`.
All 200 requests completed with a normal stop and authenticated usage. Every
answer arrived in one visible content event, so visible TTFT was essentially
complete-response time. The full report replays without calls, byte-identical at
SHA `b84412d949dc3930d0812baf31ab76b6c654a3f3450f993f59cb1e939a8e793b`
(`report-200.json`). Keep the pinned implementation for replay.

| Arm | Visible TTFT median | Visible TTFT p95 | Total median | Total p95 |
| --- | ---: | ---: | ---: | ---: |
| Short direct API | 5.012 s | 6.731 s | 5.013 s | 6.732 s |
| Existing raw evidence packet | 6.164 s | 12.822 s | 6.165 s | 12.823 s |
| Packet / short ratio | 1.230 | 1.905 | 1.230 | 1.905 |

The current packet misses the provisional 1.10 ratio even before retrieval.
Short prompts averaged 594.51 token proxies versus 7,605.19 for the packet.
The immutable report's `prompt_token_proxy` distribution inherits generic
`median_s`/`p95_s`/`mean_s` keys; those three values are **token counts, not
seconds**. Only `api_ttft` and `api_total` contain seconds. A successor should
give token distributions unit-appropriate names.

The short no-memory responses and packet responses need not have the same
length or reasoning work, so these ratios do not isolate prefill cost. They
measure observable chat latency for this frozen setup. This new streaming run
was not independently judged for answer accuracy; 73/100 remains the prior
non-streaming judgment result. During the serial API run, local source
preparation and a local GPU smoke also ran; there were no other provider calls
from this continuation until all 200 latency requests completed. Account for
that local client workload when designing a dedicated serving measurement.

Journals reserve before outbound I/O and cannot retry an unacknowledged request.
Reports replay recorded live measurements without presenting replay speed as
fresh latency.

## Ingest and bounded attention changes

`search/spine_batch_summary.py` batches exact raw fragments for Terra at ingest.
Every fragment requires its own attributed summary and exact local support.
The prompt preserves stated event times, mention times, status, corrections,
quantities, and unresolved contradictions. Exact quote checks prove membership,
not semantic entailment or completeness. Qwen receives the compiled summary
text afterward; raw support remains local provenance.

`tools/prepare_spine_corpus.py` prepares all ten namespaces without question-based
selection. It preserves source order and every raw byte through authenticated
2,048-token fragments, packs at most ten fragments and 7,000 prompt tokens per
Terra request, and allows 3,072 output tokens. It does not execute providers or
construct the hierarchy. Preparation completed under
`eval_results/full100-spine-corpus-20260909-r1`: 8,305 exact requests cover
54,294 fragments and all 54,246 turns. Preflight SHA:
`08c1be4651165f26136ff6513f21cbeb3a84d7a1b49d2c69cf9b7b6600644999`.
Summed fragment token proxies are 10,441,616, one fewer than the whole-turn
inventory because tokenization is not additive across a split boundary; raw
byte coverage is exact.

`tools/execute_spine_corpus.py` stages ordered prefixes without relevance
selection and can resume each request from its own authenticated zero-retry
checkpoint. Invalid batches remain diagnostics and cannot supply hierarchy
atoms. The first four batches of namespace 0 are prepared for schema validation
at execution-preflight SHA
`1918d6a1cd00b157292f9d4ececfc171068fbd1295d0ba7e5b21bfb2221789a0`.
They contain 24 atoms / 3,852 raw token proxies. All four Terra requests ran and
replayed with four authenticated hits and no new calls. Three batches passed;
one batch contained one paraphrased support quote and was correctly rejected.
Initial accepted output: 14 atoms, SHA
`7106faede5aa04545aae83c944383bbff3c0ca674dba849be937fc14debae691`.

`tools/repair_spine_batch_support.py` replaces invalid support generation with
Terra selection of numbered exact raw spans. It preserves every original
summary string and keeps the earlier completion and failure receipts. One
support-selection call repaired this batch; all **24 atoms** now pass the
existing strict parser. Repair preflight SHA:
`95fd9f2dedb77e10959f678a922e63aba63aa69656a5e5a354ed54f34b89fcfd`.
Repaired atoms SHA:
`f52f463eda1654ab77e2ae29a27593296c0b0bfe3a397e9f1b693bf15223f4d3`
at `offset-000/support-prefix-0004/atoms.json`. The repair replays with one
authenticated hit and no new calls. These are five total provider calls,
including the separate repair; they are not four perfectly valid first-pass
summarizations.

Namespace 0 as a whole requires 850 batches over 1,041,276 raw token proxies.
Its full execution is **running**, under execution-preflight SHA
`60029328ffdb802cf139b03c99831fffdf9e699f1eb1845d42c5dc3401d899c9`:
four existing hits, at most 846 new Terra requests, concurrency four, retries
zero. The local-gateway execution was approved for this concrete scope. Do not
ask again for these same requests. Preserve this running process and its
per-request journals. Original invalid batches are still reported as invalid
by the original compiler; apply the separately sealed support-selector repair
after execution before admitting any invalid atoms to the hierarchy.

`search/summary_shortlist_attention.py` adds an opt-in local Qwen reranker over
a conventional summary shortlist. It makes exactly one `inspect_coverage`
batch, never a recursive query-time tournament, and rejects partial workspace
coverage, foreign candidates, and nonfinite scores. Qwen sees query and stored
summaries only. Signed QK scores and OV strengths are retained as scalar audit
data; raw pointers are hydrated afterward. No transient head vectors persist.
This bounds model work, but does not yet prove latency or accuracy at scale.

A real local smoke over the existing three-source hierarchy completed 40 warm
measurements (eight questions, five repeats each): **184.25 ms median,
199.24 ms p95**, including the BM25 summary shortlist, one Qwen pass, and exact
hydration from preloaded raw turns. Cold Qwen loading took 149.24 seconds and is
separate. The shortlist had eight sections for seven questions, but only two
for the benchmark chronology question: the lexical admission failure remains.
This is a timing mechanism result, not answer accuracy or full-memory routing.
All per-question route receipts were stable across repeats. Reproduce with
`tools/assay_bounded_summary_attention.py` and a fresh output root.

Smoke root: `eval_results/bounded-summary-attention-smoke-20260909-r1`.
Runtime SHA:
`54e405bce5a1a3a58d1503cdb790ce47a851edb4af870cb99b0ce3719e54c981`.

The existing model precision remains FP16 prefix weights/forward with FP32
softmax and pooled readout calculations. It runs the preceding five complete
transformer blocks and the sixth block's attention readout, skipping its MLP;
it is not an FP32-only or attention-head-only forward pass.

New streaming, lifecycle, raw batch, and bounded-attention contracts plus the
existing integration suites pass **140 tests in 12.30 seconds**. Existing frozen
pilots and prior source files remain unchanged.
No new router or summary compiler is promoted. Remaining work includes complete
raw summary execution and audit, ingest hierarchy construction, full-memory
resident routing, exact hydration, and joint judged accuracy/latency evaluation.

## Active execution handoff

The full namespace-0 Terra ingest job is running in unified exec session
`78281` (PowerShell, worktree root). Poll it with `write_stdin`; do not launch a
second copy. At the 48-completion observation, 20 batches passed directly and
28 were rejected solely for exact-support-quote membership/budget violations.
The summaries in rejected batches have not been admitted. This is a recurring
format/grounding issue, not proof of semantic correctness. The label-selection
repair above resolves exact membership without regenerating summaries; inspect
its entailment limits and any new failure classes before scaling further.

The source/compiler/executor files pinned by existing preflights must stay
unchanged during this job. A failed or interrupted provider request may retain
an unacknowledged reservation; preserve it and account for uncertainty rather
than silently retrying. No other provider jobs from this continuation are
running. The 200-call latency run and local Qwen timing smoke both finished.

After namespace execution, authenticate `atoms-prefix-0850.json`, prepare the
separate support-selector repair over all completed raw outputs, and audit raw
span partition/role/time preservation before building the user-spine hierarchy.
Existing support-selector results can be reused only through exact immutable
request/summary/source bindings, with reuse explicitly recorded. Evaluate
conventional semantic summary admission alongside the bounded Qwen reranker;
the current literal BM25 shortlist still misses singular/plural and semantic
matches. Do not infer a 95% result from summaries, source coverage, or timing.

The two added support-repair tests also pass, verifying unchanged summary text,
exact source-label hydration, and rejection of foreign support labels. Total
new/existing checks in this continuation: 142 passing tests across the recorded
140-test integration run and the focused two-test repair run.
