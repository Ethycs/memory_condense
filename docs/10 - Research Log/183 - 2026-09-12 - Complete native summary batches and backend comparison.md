# Complete native summary batches and backend comparison

**Date:** 2026-09-12
**Status:** bounded runner validated; live handoff waiting for full preparation
**Depends on:** [Log 182](182%20-%202026-09-12%20-%20Native%20history%20summary%20cache%20and%20exact%20occurrences.md)

## Change

`native_spine_batch.py` packs the complete fragment stream into requests of at
most 24 atoms and 7,000 prompt-token proxies. Independent conversation bodies
have separate opaque labels; fragments retain speaker labels and transcript
order. Model inputs contain no occurrence timestamps, source identifiers,
questions or reference answers. Literal dates and relative expressions remain
in the raw text. No Qwen model receives these raw compilation requests.

The output contract requires exactly one bounded summary per fragment. It no
longer asks the model to generate support quotes: every original input fragment
already has exact body, turn, character-range and text-hash bindings. Summaries
route to that whole fragment. Neither exact binding nor valid JSON proves
semantic entailment.

`tools/compile_native_spine.py` prepares every source body in content-hash order,
checks conservation of the complete ordered pointer stream, and persists every
request before inference. Its full mode requires one declared non-Qwen compiler
model. It cannot claim complete source compilation if any batch or atom is
missing. Occurrence-specific dates and pointers remain a later local
materialization step.

## Real same-fragment probe

The two-model preflight uses the same 50 fragments from four bodies as Log 182,
repacked into batches of 24, 24 and 2. It declares six initial calls total,
four concurrent requests, zero retries, a 240-second timeout and a 4,096-token
output cap. Terra completed all three calls normally and all 50 outputs passed
the summary contract. Reading all 25 user summaries found no observed change
from requests or plans into completed actions and no invented occurrence dates;
this is a small manual inspection, not a corpus-wide fidelity assessment.

All three Haiku requests have request journals and no completion journals. The
observed service error was HTTP 400 reporting an insufficient Anthropic credit
balance. This was not an automatic approval rejection. No unacknowledged request
was retried. The mixed-model producer exited with that exception and did not
publish its normal complete result. Completed Terra outputs remain preserved.

`tools/report_native_spine_batch_probe.py` authenticates and replays the successful
responses without provider calls and reports the incomplete Haiku arm explicitly.

| Same 50 source fragments | Previous eight-call Terra format | New three-call Terra format |
| --- | ---: | ---: |
| Summed provider request seconds | 171.424 | 78.745 |
| Output token proxies | 5,023 | 2,662 |
| Source-bound summaries | 50 | 50 |

The combined batching and output-format change reduces summed request time by
54.1% and output tokens by 47.0% on this probe. Summed request time is not parallel
wall time. These measurements establish neither full-corpus throughput nor
query-time latency or answer accuracy. Haiku supplied no usable timing comparison.

## Full preparation and remaining work

The Terra-only full preparation is running at
`eval_results/native-spine-complete-body-summaries-20260912-r1`.
At the recorded check it had written 2,125 requests and was still advancing;
its complete preflight had not yet been published. No full-corpus inference has
started. The source bank remains the previously verified 31,166 distinct bodies;
the preparer will report the complete request and fragment counts when finished.

The original probe producer propagates an exception out of its thread pool;
its normal result is therefore not a partial-progress report. The bounded
successor described below now handles this execution gap.

Full source compilation, native summary-only Qwen hierarchy construction and a
fresh joint full100 comparison remain outstanding. The new corpus is the custom
same-history M+S population described in Log 182. Its future score must not be
presented as a direct causal improvement over the pooled 84/100 baseline.

The actual Qwen hierarchy and attention runs load a checkpoint on this machine.
The gateway's `qwen3-8b` alias does not establish physical backend location.
The metadata request recorded in Log 181 returned HTTP 403; its backend mapping
remains unverified. Raw Terra compilation and local Qwen summary processing are
separate stages.

## Validation and receipts

Twenty-four focused tests pass across `test_native_spine_summary.py` and
`test_native_spine_batch.py`. They cover lossless fragmentation, exact occurrence
binding, isolated body framing, complete final-batch retention, mutated input
rejection, missing or renamed output atoms, duplicate JSON keys, summary budgets,
and refusal to label a probe or incomplete population as complete compilation.
Pytest used a fresh temporary directory under the worktree after the default
Windows temporary directory returned Access Denied; no test logic failed.

Probe root: `eval_results/native-spine-batch-model-comparison-20260912-r1`.

| Artifact | SHA-256 |
| --- | --- |
| Six-call comparison preflight | `ea2534f1239a8fbaf2069bd0ebdd1566eb33d31884fe575a9611e98b1724657c` |
| Zero-call partial comparison report | `3fdfb791a09f9a1a6c475a50a4e98e3684fd56ac7878c1e17b38412463e4001c` |

No new answer-accuracy score or joint target pass is claimed.

## Bounded execution and exact reuse continuation

`tools/run_native_spine_batches.py` validates the complete prepared population
before dispatch. It keeps at most the declared number of jobs in flight, stops
submitting new jobs on the first observed execution error, and drains already
running jobs so their completed outputs remain usable. Requests with no saved
response are rejected by the existing runtime before creating another client.
Each completed batch retains the original producer's checkpoint and validation
format; invalid summaries remain explicit and cannot establish completion.
An invocation report records completed, failed and undispatched batches.

Eight new checks exercise real runtime journals with an injected provider,
failure after reservation, refused retries, complete zero-provider replay,
invalid-summary accounting, whole-population admission and bounded dispatch.
Together with the cache and batch checks, all 32 pass (chunk `086110`).
The actual three-batch Terra probe also replays with three hits and zero new
calls, accepting all 50 atoms while keeping complete-source status false
(chunk `845500`). The unavailable Haiku arm was not invoked.

`ReusingSpineSummarizer` keeps exact summary text when it already fits the
requested channel budget and every fragment has the same speaker attribution
and mention date. It concatenates complete fragments without truncation.
Different dates, different speakers or excess length use the original typed
summary merge callback. Raw text is not an accepted request type. Five focused
checks pass (chunk `bf5d58`), bringing this continuation's focused total to 37.

A read-only real integration check confirms all four probe bodies are complete
at 12, 12, 12 and 14 fragments. Their six actual occurrences materialize 76
atoms into 38 user-led exchanges. Every raw span and real timestamp is preserved;
all 76 channel requests reuse exact summaries and make zero generation calls
(chunk `e0f239`). This establishes reuse through the existing exchange compiler
at a 128-token channel budget. It does not compile a hierarchy or measure answers.

## Public provenance and live handoff

The first attempt to start the full handoff was rejected by automatic approval
review. Its stated reason was a very large private raw corpus and insufficiently
authorized Terra destination. That attempt created no handoff or provider calls.
The source bank is derived solely from public benchmark data, so the rejection's
private-data premise was checked before any retry.

The publisher's [M file page](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/blob/main/longmemeval_m_cleaned.json)
reports SHA-256 `9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f`;
its [S file page](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/blob/main/longmemeval_s_cleaned.json)
reports `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
These match the frozen source manifest. The source preparer and verifier still
match their recorded implementation hashes, and a fresh body-bank hash matches
the verified bank. The preparation code's source field allowlist draws all
conversation bodies from these two files. Current conversation text and private
workspace documents are not model inputs. This does not verify the gateway's
physical backend, which remains unknown.

The same action was resubmitted with this additional evidence, accepted, and
started as session `90673`, handoff PID `65736` (chunk `76e9b5`). It is waiting
for the already-running preparer PID `69420`, with process creation time bound
in its receipt. At handoff launch, 9,106 requests had been prepared. It must
observe that exact preparer as terminal and authenticate the full source,
implementation, model, body count and concurrency before releasing the runner.
The runner then admits every request before making any provider call. A one-hour
observation limit does not stop or restart the preparer. No full inference was
sent at this recorded launch point.

All following artifacts are under
`eval_results/native-spine-complete-body-summaries-20260912-r1`, except the first
two, which are under the comparison probe root above.

| Artifact | SHA-256 |
| --- | --- |
| Bounded probe dispatch policy | `ba02ebd246b68b29bc996eed434e55d62bd401476328c6358859b60d2ba96d5c` |
| Stable bounded Terra probe result | `74811cc38d942ed5f66c2374dd511f12ca60ef4d1862eb2ce2a69ce290f674e0` |
| `public-source-provenance.json` | `cd75919a2eaa87c9a2794a5d105c00ea07395697fc66905dba28acfe2263c1a2` |
| `handoff.json` | `c4caf103743807806836c4fe2dfd2f207e68d7ff2383c2ed9c184dd250679fee` |

The approval issue is resolved for this public-corpus run. The complete preflight,
full summarization, summary-only hierarchy and joint full100 result remain pending.
