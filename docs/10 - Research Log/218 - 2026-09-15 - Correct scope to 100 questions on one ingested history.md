# Correct scope to 100 questions on one ingested history

**Status:** RUN COMPLETE — 71/100 candidate accuracy; warm median 3.700 s; 95% target not met.
**Date:** 2026-09-15.
**Scope:** Exactly 100 questions against one already ingested 1,098,417-token history.
**Supersedes:** The 100-history execution in [Research Log 217](217%20-%202026-09-15%20-%20Frozen%20corpus%20ready%20and%20full100%20preparation%20started.md).

The user explicitly corrected the scope: **100 questions, not 100 histories**.
Treat this as a persistent execution constraint. Freezing the design does not
authorize a 100-history run. Do not resume the r1/r2/r3 frozen-corpus controllers
or `evaluate_frozen_native_spine_full100.py` for this request.

The previous counter meant histories passing data admission, not question tests.
It reached 22 histories before stopping. No benchmark answer release occurred.
The r3 controller PID 19760 and preparation child PID 47904 were checked against
their creation times and terminated at 21:55:45 UTC. Both were confirmed stopped.
All completed caches were preserved. The stop receipt is
`eval_results/native-spine-scope-correction-20260915-r1/stopped.json`, SHA-256
`4fc17eab007ee6f07344b76245d4558e4d68a13effb246ed2561fb0b8b048882`.
Old exec session 82986 is terminal. Its completed cache work is not an accuracy result.

## Corrected lifecycle

1. Reuse the previously ingested, frozen history at
   `eval_results/native-spine-design-pilot-20260914-r1`. Its 520 unique bodies
   and existing exchange, attention, parent, and vector caches remain unchanged.
2. Lock 100 source-grounded questions and separate reference answers before
   executing retrieval or candidate answers. The question author sees real user
   turns from this one history, never candidate outputs or retrieval results.
3. Load and admit that one memory namespace once. Retain the resident memory
   across all questions. Recompute query embeddings and retrieval inside each
   timed memory answer; hydrate exact source sections through the existing code.
4. Save all measured answers before opening reference answers for judging.
   Report accuracy and latency together, then independently reconstruct served
   raw packets with the existing audit function.

This is a generated, source-grounded evaluation on real transcripts. It is not
an official LongMemEval score or a claim of generalization. The question set is
frozen before results, and questions are not selected or removed based on scores.
Question and reference text never enters ingestion or the retrieval index.

## Locked question set and existing method

`tools/prepare_native_spine_single_history100.py` selected 100 distinct source
bodies from 446 eligible bodies in the same cached history using a fixed salted
hash order. It requested one question/reference pair per body from the authorized
Sol gateway and checked every support quote against its actual user turn. These
source bodies are conversation segments within one memory namespace, not 100
new memory histories. It completed 100 authoring calls and exited 0.

The locked question artifact is
`eval_results/native-spine-single-history100-20260915-r1/questions.json`, SHA-256
`76492c4fbc3b142d803eff6bdcb86ac456bb68119431f7da5c9b20d335751bc1`.
It binds the same history scope
`18478c4f3a4e625c9c068f5c2a8e29d1eb29a1b76721eeeaa8295f58b9ab4836`.

`tools/evaluate_native_spine_single_history100.py` adapts the existing cached
history loader and frozen retrieval, streaming, journal, judge, and raw-packet
audit functions. It enforces one namespace and 100 unique questions. It uses the
unchanged v5 reader, 1,024 context tokens, exact raw hydration, and summary-only
Qwen processing from the frozen candidate. No retrieval algorithm or memory
artifact was changed for the new questions.

There are 100 questions, each measured with the parent-context candidate,
user-first comparison, and an identical-prompt API control: 300 answer streams
and 200 logical judgments. These are three measurements per question, not
additional question sets or histories. The target remains at least 95/100
candidate answers with warm median total latency below five seconds. Report
p95, fraction below five seconds, and the matched API comparison as well.

Three focused checks passed in 3.36 seconds in
`tests/test_single_history100_scope.py`: reject a second history, reject duplicate
questions, and reject reference quotes that differ from the real user text.
These checks do not establish answer accuracy.

## Completed execution

| Item | Binding |
| --- | --- |
| Run root | `eval_results/native-spine-single-history100-20260915-r1` |
| Terminal exec session | `44016`, exit 0 |
| Answer worker | PID `18732`, creation time `1789509976.4833322` |
| Log | `eval_results/native-spine-single-history100-20260915-r1/answer-run.log` |
| Namespace | `native-spine-0b341ca78960ea878446ed15e5f7ae57273e6fae870e8669a5d7a7856d02a666` |

The worker completed all 100 questions across three arms, then judged and audited
the saved evidence. It loaded one resident history with 1,098,417 actual eligible
tokens, zero new histories, and zero compilation jobs. Its preflight SHA-256 is
`eaeb31a2d58ae713c7c23e5fdf432744e0e3a031b6cface0b5a65cd9a982e3be`.
Independent inspection of that plan confirmed one unique namespace, 100 unique
question IDs, one namespace load, zero compilations, and no references loaded
before answers. All 300 responses ended with a normal stop. The 200 logical
judgments required 154 distinct provider prompts because identical predictions
share identical judge inputs. Replaying the judge reproduced the same report with
zero new calls and 154 authenticated cache hits.

| Method | Accuracy | Warm median total | p95 total | Answers below 5 s |
| --- | ---: | ---: | ---: | ---: |
| Parent-context candidate | 71/100 | 3.700 s | 6.478 s | 83/100 |
| User-first comparison | 64/100 | 3.950 s | 6.545 s | 87/100 |
| Matched candidate API control | Not independently scored | 3.677 s | 5.788 s | 91/100 |

Candidate packet preparation had a median of 0.189 s. The candidate/API ratio
of median totals was 1.0062. The 39.020-second resident cold load was excluded;
this is a warm serving evaluation, not an ingestion-latency measurement or a
test of a deployed service endpoint. It exercises the resident memory retrieval
and exact hydration application code over completed ingestion artifacts.

The existing raw audit reconstructed 200 memory packets and verified 2,859 served
raw spans against their original bodies, turn identities, roles, dates and exact
character slices. A separate lifecycle audit recomputed scores and timing
distributions directly from saved responses and checked reference support against
the original source body, rather than accepting an identical quote from any body.

| Artifact in the run root | SHA-256 |
| --- | --- |
| `answers.json` | `04aa1a406807a446dda07ddf01848b6f3ff982e79cd68080836987087b4f143e` |
| `joint-report.json` | `999e6a75d162e1d61072f9a25a62f071b6e6a2880d1713c2268e5cb41cf58a33` |
| `raw-audit.json` | `cc1443cb49649e5525081ebfb5d017975200873721e9edcf719901f9c19f7fa0` |
| `lifecycle-audit.json` | `35395170274e5c08d72d91619b072b8a45ac7a82c5a922272f473ee8832b02ea` |

Reproduce the judge and lifecycle checks without provider calls:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.evaluate_native_spine_single_history100 judge --root eval_results/native-spine-single-history100-20260915-r1
.\.pixi\envs\dev\python.exe -X utf8 -m tools.audit_native_spine_single_history100 --root eval_results/native-spine-single-history100-20260915-r1
```

## Failure diagnosis and next boundary

Of 29 candidate misses, 19 contain every recorded reference support quote in the
served packet, six contain some, and four contain none. This quote check is a
diagnostic, not a proof that the entire reference answer is supported by the
packet. Of 71 judged correct answers, 70 contain all original-source support
quotes and one contains only some.

Many answers omit details despite available evidence. The frozen reader requests
the shortest possible response, while the generated references often enumerate
several details. This is a concrete reader-completeness issue to investigate.
There is also a confirmed grading anomaly: question 000 returned `B` against
`B, not A`, and the judge rejected it for omitting `not A`. Some generated
questions also identify their source conversation too vaguely among repeated
topics in this large history. These limitations remain visible; the locked
71/100 result has not been revised, and no question has been removed.

The goal remains active. Improve reader completeness and trace the ten missing
support packets on this same cached history. Any subsequent result is development
on an exposed question set and must be labeled accordingly. Preserve the current
question/reference manifest and baseline journals. Do not silently adjust grading
to claim 95%, rebuild histories, or restart the stopped 100-history controllers.
