# Native full100 runner and completed expanded attention

**Date**: 2026-09-12  
**Status**: Expanded exchanges and attention complete; native joint runner tested; full corpus still incomplete  
**Depends on**: [Research Log 195](195%20-%202026-09-12%20-%20Bounded%20summary%20recovery%20and%20expanded%20source%20repairs.md)

## Results and target status

All **7,121 prepared body exchange sets** are complete, containing **37,226
exchanges**. The matching attention stage also completed: **7,217 cached summary
windows** using the unchanged local Qwen method. Parent compilation is released
against those completed artifacts, reusing the 143 accepted original parent
merges.

The expanded summary store now admits **11,438 bodies and 119,144 sections**,
preserving the previous 7,121-body store exactly. Its vector preparation contains
118,693 unique summaries: 74,059 existing vectors can be reused and 44,634 need
encoding. Encoding is queued after the owned parent process exits.

The native full100 runner is implemented and passes **43 focused tests**, including
a synthetic 400-stream execution and 200 logical judgments with zero-call replay.
The real corpus correctly fails its admission check before any model or provider
construction. This is infrastructure validation, not a new accuracy score. Full
source compilation and a fresh joint full100 run remain necessary to establish
95% accuracy and the eight required latency ratios at most 1.10.

The preceding goal turn recovered rejected summaries and source batches. This
turn completed their expanded exchange/attention stages, admitted a larger body
snapshot, and added the native evaluation runner. Both made concrete progress.

## Complete expanded exchanges and attention

The first recovered exchange allowance completed 7,112 bodies with 128 new jobs
in 35 batches, then ended normally with nine pending dependencies. Its dependent
attention and parent waiters stopped without starting model work because that
result was partial. Their old one-shot controls remain terminal.

The next explicit continuation permitted at most 512 jobs in 128-job invocations.
It needed only **35 jobs in 15 batches**, completing the remaining dependent
merges in one invocation. Prior requests, responses and accepted summaries remain
unchanged. This completion is still the prepared subset of the full source bank.

Root: `eval_results/native-spine-exchanges-20260912-r3`.

- Complete result: `614e928f05a13a3cbdc3881b4eca3162fb317f1905a435b37fe657dd97112758`.
- First continuation result: `f4e1225a08e5f7f6bcf3a1d36b924d317b22ba9b531220143f03f937423dace3`.
- First completion receipt: `65d816a55010859fb3ae4a97a025040aa04642f84a1b40a6e346c9a4552207ac`.
- Final continuation policy: `b47d543589bc9ff82a22b293800e76db4ed25d6ad7dba8163917bfae2cfa7a9c`.
- Final completion receipt: `7a2a19e2cf575f070ec50b7a7eaa6f3e62327c4866a69c450215b78809bd9737`.
- Sessions 91733 and 99223 exited zero (`854809`, `a3c4d1`).

The recovered attention admission authenticated the completed exchange producer,
then ran preparation and attention in separate processes. It reused the existing
cache and kept the same summary-only scoring method. The Qwen forward weights
remain FP16, with FP32 softmax/readout; this is not an all-FP32 model or a set of
standalone attention heads.

Root: `eval_results/native-spine-attention-20260912-r4`.

- Preflight: `cf7fcb6381c6bc712bfa0b3d882e067578cb765930e73c3a9fc4a500ac54794a`.
- Producer admission: `8f886e5bb02eb5b44d9d66a5bd007edfbd1857d838e99f27019610016d6ea578`.
- Complete result: `c54569076255c13f42c84c348785b2222b598a84cfba697606737cdbbe45f761`.
- Handoff completion: `2ebdf895193a9c5676dc5ac5d8b5ab700bdc34e78d641e58e614b135a6449fa0`.
- Session 10691 exited zero (`68eff8`). Its former driver PID was 29616,
  creation time 1789227740.9019449.

## Larger admitted source snapshot

Root: `eval_results/native-spine-admitted-body-store-20260912-r5`.

- Manifest: `1c41af85418a6a7d0c4f7f82610ae3ff915fd2531fc0fc69b37800e4b33bf14a`.
- Admission snapshot: `47123f9bfa25bcba04b195d06904673417149386c107b022ed9b53d10f5514b3`.
- Database: `1fb73ee8be2bc9ba9050af8efa4e93389547d2be5afc73ba9c58913aa0da1680`.
- Previous-store preservation: `e1ed3812497f939e8bee28fd3c801871bee95ea56cc4a69e06a34f6fd8a3220d`.
- Handoff completion: `87af36a30dc47d4c98733e5b94ea6ec58e754b39d51c5f3a471cdec1054362d4`.
- Session 33896 exited zero (`38e17d`).

The snapshot includes 5,079 admitted batches, 197 repaired batches and six
transport recoveries. It covers 118,665 original fragments plus 479 additional
subdivision sections. At snapshot time, 8,698 batches were pending and 35 remained
unrepaired. Those counts are frozen admission state, not the latest live ingestion
counters. Entire incomplete bodies remain excluded.

Vector root: `eval_results/native-spine-summary-vectors-20260912-r4`.
Its prepared manifest is
`026a8326b6e1b60f458f23d7e6c2ef44d1a4105caae54ea26f51964ebd0bc111`.
The 118,693-summary population and its 74,059/44,634 reuse split were checked
directly (`cfbbff`). Preparation loaded no embedding model.

## Native full100 evaluation contract

`tools/native_spine_joint_population.py` admits the complete native population
before model setup. It requires every source body, hierarchy and summary vector,
all 100 distinct locked cases and their own source namespaces, and at least one
million **actual body-text tokens through each question's date**. It recounts
materialized raw turns, excludes future turns from the minimum, and does not count
generated source boundaries. Missing vectors or explicit partial materialization
are rejected.

`tools/evaluate_native_spine_full100.py` uses the unchanged answer renderer and
matched-stream protocol with the native resident retriever. Its canonical arm
names retain `flat`, `hierarchy`, `hierarchy_api` and `short_api`; here `flat` means
direct original-atom retrieval and `hierarchy` adds context from offline
attention-defined chunks. The latter is the scored candidate.

Each memory-arm request performs a fresh query embedding, summary routing and
exact raw hydration inside the end-to-end timer. It performs no query-time Qwen
passes. Cold corpus, index and encoder setup is excluded from the explicitly warm
latency measurement. API controls send fresh requests; the identical-evidence
control uses exactly the candidate's prepared prompt, and live candidate retrieval
must reproduce that evidence. Adjacent candidate/control calls reverse order for
half the questions.

All 400 answers must be sealed before reference text is loaded from the pinned M
dataset. Reference question IDs, text, dates and answer hashes must match the
separate case plane. Sol judges 200 logical memory-arm answers; identical judge
prompts may share physical requests. The quality and latency gate binds judgments
to those same streamed responses and requires 95 correct candidate answers, all
eight median/p95 TTFT/total ratios at most 1.10, and normal completion of every
stream. Unacknowledged requests remain terminal until explicit recovery; no
automatic retry or cached-answer latency measurement is introduced.

The CLI provides `prepare`, `run --enable-provider`, and `replay`. Preparation
takes the source root, complete summary store, complete recovered-parent report,
complete vector root and pinned M dataset path. None of the current partial
stores is eligible for that release.

## Tests and real readiness check

`115e3b`: fourteen admission tests pass, including actual tokenization of a
million-token fixture, future-only content rejection, incorrect totals, missing
vectors, duplicate namespaces and partial compilation.

`34b362`: all **43 focused checks** pass in 75.11 seconds across the new native
runner and population tests, the reused joint gate tests and streaming latency
tests. The synthetic orchestration test performs 400 fake fresh streams, 200
logical judgments and 100 deduplicated fake judge calls, then reproduces the
report with zero provider calls. A separate test invokes actual native retrieval
and hydration with deterministic test vectors. These tests do not run a benchmark
model or demonstrate benchmark accuracy.

The real readiness check binds the new 11,438-body store, the currently completed
1,669 parent templates and the full 31,166-body source bank. Preparation rejects
the incomplete population before model runtime or provider construction, without
publishing an evaluation preflight.

Root: `eval_results/native-spine-joint-readiness-20260912-r1`.
Receipt: `d66ea67da1041295d852502da2a49385c15259d4cb9c300e267210359f23783b`
(`a9079c`). This receipt binds the evaluation implementation; preserve those
files while using that provenance.
All 102 bound implementation files still match, and eight added Python files
pass parsing and whitespace checks (`3b53c0`).

## Active handoff

Main Terra ingestion remains at PID 56400 / creation time 1789216671.9337437.
It owns the remaining never-dispatched source requests; do not start another
runner against that population.
The final live check finds 5,302 accepted original validations and 252 invalid,
5,554 total (`19823b`). Repair overlays do not change those original states;
the 197 admitted repairs leave additional rejected batches for the next selection.

The recovered parent continuation is under
`native-spine-recovered-parent-hierarchies-20260912-r1/continuations/2048-20260912-r2`:
session **64318**, PID **48976**, creation time **1789228197.1589692**, policy
`945152f161fed81a2484777d09df99ab0eaed0a980e8865f4dc9637185df7909`.
It starts with a zero-generation publication and then allows at most 2,048 local
jobs in 128-job invocations. The earlier r1 control remains terminal.

The vector handoff is under `native-spine-summary-vectors-20260912-r4/handoff`:
session **64473**, PID **22840**, creation time **1789228400.1653416**, policy
`7a77dcc39bc23c9a367d3c1e09ce2bfe6c0c40c44739523d51466d3cb0806ce3`.
It requires that exact parent process to exit with a completion receipt before
loading BGE on CUDA, batch size eight. It sends no provider calls.
Main ingestion, the parent process and the vector waiter were all confirmed live
at the final check (`19823b`). Parent loading also reproduced the complete
7,121-body exchange result with zero model calls (`2a6110`).

Revalidate process handles and artifact state before continuing. Finish these
stages, continue source ingestion and rejected-fragment repair, and expand
admission toward the complete source bank. Only then prepare and execute the
native joint full100 run. The active goal remains unachieved.
