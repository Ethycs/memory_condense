# Ten million-token session evaluation

**Status:** Complete — all ten sessions and 1,000 questions; workers closed.  
**Started:** 2026-09-22. **Completed:** 2026-09-23.  
**Applies to:** The user-authorized ten-session, 1,000-question battery.  
**Depends on:** [Research Log 238](238%20-%202026-09-16%20-%20Five%20new%20100%20question%20evaluations%20and%20build%20replay.md).

The user requested a large battery of ten approximately million-token sessions.
This campaign uses ten additional source histories (source ordinals 6–15), with
100 newly authored questions per history. Each has at least one million raw
tokens eligible through its question date. Question sources exclude those used
for the original 100 questions, the subsequent 500 questions, and every other
question in this campaign. Histories come from the existing real-transcript
corpus; they are not ten newly collected engineering transcripts.

The established direct8/2048 user-spine routing, exact user-section hydration,
v7 reader, Sol answer model and semantic grading remain unchanged. Each history
is freshly ingested using `MemoryCondenser.ingest_many`, receives its compiled
native snapshot and parent-user vectors, closes, and reopens in a separate
read-only answer process. Existing source summaries and Qwen attention artifacts
are reused. This measures application ingestion/persistence/retrieval/answering
with cached compilation, not fresh end-to-end Qwen compilation.

Questions and references are locked before that history's candidate answers.
They do not enter ingestion. Grading begins only after all 100 answers for the
history are sealed. Source-only question authoring can overlap ingestion; timed
answers run serially without another campaign GPU worker. Every served evidence
packet receives an independent reconstruction check against the original raw
source. No full-context control calls are scheduled.

The campaign is separate from all prior sealed runs:

- [Campaign manifest](../../eval_results/native-spine-ten100-20260922-r1/campaign.json)
- [Live controller status](../../eval_results/native-spine-ten100-20260922-r1/controller-status.json)
- [Runner](../../tools/native_spine_ten100.py)
- [Controller](../../tools/run_native_spine_ten100.py)
- [Aggregate reporter](../../tools/report_native_spine_ten100.py)

Campaign SHA-256: `fe4eb589ac698d0878f915db6a10fb5d93c978e7fe697325bbe542bcb18c41cf`.

## Verification before launch

Existing quote-validation, application lifecycle and user-evidence tests:
**27 passed**. The initial test invocation encountered a sandbox permission
error in the shared system temporary directory; rerunning with a fresh temporary
directory inside this workspace passed. All ten scopes were sealed and verified
above one million eligible raw tokens. The controller launched as PID 70248.

Final accuracy, latency, token usage and failures are recorded below. Warm latency excludes ingestion and cold application reopen,
which are measured separately. These generated questions and their semantic
grades are not an official LongMemEval score.

## Completed result — 2026-09-23

**913/1,000 (91.3%)**, with every requested session and question completed.

| Session | Eligible raw tokens | Correct | Warm median | p95 | Mean input tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,115,343 | 94/100 | 4.744 s | 7.476 s | 1484.18 |
| 2 | 1,111,235 | 85/100 | 4.523 s | 7.370 s | 1480.03 |
| 3 | 1,151,461 | 88/100 | 4.615 s | 6.444 s | 1532.26 |
| 4 | 1,144,363 | 95/100 | 4.235 s | 6.252 s | 1509.36 |
| 5 | 1,138,817 | 93/100 | 4.900 s | 6.524 s | 1487.66 |
| 6 | 1,102,411 | 93/100 | 6.353 s | 11.520 s | 1465.96 |
| 7 | 1,151,568 | 88/100 | 4.826 s | 9.348 s | 1571.86 |
| 8 | 1,158,491 | 93/100 | 4.880 s | 7.152 s | 1438.66 |
| 9 | 1,118,241 | 89/100 | 4.914 s | 8.113 s | 1539.01 |
| 10 | 1,064,589 | 95/100 | 4.799 s | 8.232 s | 1494.83 |

Combined warm median: **4.783 s**; mean: **5.238 s**; p95: **8.935 s**. **574/1,000** answers finished under five seconds, and **9/10** session medians were under five seconds. Retrieval and prompt preparation averaged **0.282 s** (p95 **0.380 s**). Session 6 experienced a model/gateway slowdown: sampled preparation times remained about 0.3 seconds while total response times rose.

Mean answer input: **1,500.381 tokens**; mean output: **31.242 tokens**. Each session had **1.06–1.16M eligible raw tokens**, totaling **11,256,519**. There are 1,000 distinct question texts and 1,000 distinct question-source bodies, with no overlap with the prior 600 question-source bodies.

All **1,000** answers stopped normally. All **1,000** evidence packets passed independent exact raw reconstruction. Every application was ingested, closed and reopened in a separate process. Ingestion plus cached-source setup took **10.7–17.2 minutes** per session; cold reopen took **24.3–31.5 seconds**. These costs are excluded from the warm answer timings. Existing summary/attention compilation was reused, with zero new Qwen calls and zero full-context control calls.

The aggregate 95% target is not met under the unchanged grader; two sessions scored 95/100. Of 87 marked misses, 61 contained all recorded support quotations and 26 did not. Quote presence does not establish answer correctness. The saved misses include actual evidence-selection or answer errors and reference/grader mismatches. For example, session 1 question 35 correctly answered the requested format and word count, but the grader demanded additional reference details. Session 2 question 28 correctly answered necklace type and style, but was penalized for omitting an unasked price. No scores were adjusted.

### Source-citation recovery

Three authoring-validation stops occurred before the affected history answered any questions. Session 5 question 58 supplied four quotations; two from the same user turn were joined into one exact contiguous span. Session 7 question 87 and session 10 question 39 paraphrased a quotation; each was replaced by the exact wording from the same attributed source turn. Session 9 question 62 also received the already-configured unique case-only correction. The original author responses and all correction receipts are retained. All 1,000 authored question and reference-answer strings are unchanged. These repairs made no new model calls and did not repeat ingestion or completed answers.

The three citation-merging checks passed in addition to the 27 launch checks. The final source/answer audit passed with zero model calls. Its first timing-summary attempt used a helper restricted to 100 observations; the corrected aggregate calculation handles all 1,000 observations without changing any measurements.

All campaign controllers and generation workers are closed. The preserved controller failures are historical authoring-validation events; the final controller status is complete.

- [Complete report and all marked misses](../../eval_results/native-spine-ten100-20260922-r1/report.md)
- [Sealed aggregate](../../eval_results/native-spine-ten100-20260922-r1/aggregate-report.json)
- [Final source/answer verification](../../eval_results/native-spine-ten100-20260922-r1/completion-verification.json)

Aggregate SHA-256: `bdd4542fc53deff6506b319fb8757d39afeaa42694222c2df20f91b22e30ced5`.  
Completion verification SHA-256: `e166040bfed63ab6922928e393c81d3b58e832c8575b811bcae399a5b545b0f7`.
