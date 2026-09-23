# Native benchmark support availability and ingestion continuation

**Date**: 2026-09-12  
**Status**: Expanded retrieval diagnostic complete; full ingestion and hierarchy work remain  
**Depends on**: [Research Log 190](190%20-%202026-09-12%20-%20Native%20resident%20retrieval%20probe%20and%20direct%20source%20repairs.md)

## Finding and next action

The first native diagnostic using all 100 locked benchmark questions finds that
source availability is the main current bottleneck. Of 97 questions with
eligible M annotations, **76 have no annotated turn in the admitted snapshot**.
Direct summary retrieval and exact hydration retain some annotated evidence for
**20 of the 21 questions with support available**. All baseline evidence survives
the additive attention-chunk arm; its eight added sections do not change this
support count. This is evidence about routing, not a 95% answer score.

All 100 histories remain partial, containing **166,319–259,798 raw body tokens**.
Only eight questions have all eligible annotated turns available, and six have
all those turns selected and hydrated. There were no reader calls, judgments,
or matched API controls. The joint 95%/1M/latency goal remains unproven; the latest
full answer comparison is still the earlier pooled flat 84 versus hierarchy 8.

The previous goal continuation made progress by completing the resident probe,
repairing source batches and releasing expanded vectors. The intervening alias
question only clarified existing evidence. This continuation revalidated live
handles, completed the vector stage and native question diagnostic, and advanced
the next ingestion/hierarchy work. It is not a blocked or no-progress turn.

## Completed expanded serving path

`tools/expanded_native_spine_namespace.py` adds an explicit adapter over the old
corpus reader. It authenticates the original corpus/template store, loads the
new direct-repair producer, and checks every old summary/pointer record before
admitting the larger population. Both store identities remain visible; it does
not attribute the new body store to the old producer. Every one of the 1,669
old bodies and 17,190 old atoms is unchanged in the 5,481-body store. Actual
occurrence rebinding still validates each cached hierarchy against exact atoms.

The expanded vector stage completed **57,020 distinct summary strings** in 446
checkpoints. It reused all 17,146 earlier vectors and embedded 39,874 new strings.
Vectors are normalized FP32, 1,024-dimensional BGE outputs. The local model saw
summaries only. The process exited successfully; session 23061 is terminal.

The preceding Qwen continuation stopped cleanly after 512 jobs in four bounded
invocations. Its immutable final report contains **269 trees, 578 leaves, 309
parents and 1,369 original atoms**. Those templates are the fixed hierarchy input
to this diagnostic, although later compilation may produce more trees.

| Artifact | SHA-256 |
| --- | --- |
| Active body store, r3 | `5c127b17048f22a0afed4d8b59b35e740ee6d0e945234433e195a9fba0365fcc` |
| Expanded vector result, r2 | `761030d30d5e66a234812ecb68121162ae1f6ecb0ab15768edb89c7a040e6a1e` |
| Expanded vector handoff completion | `432f3a3b26eb7580b4f0f655b8198eaa01848a9a3f24c07b98ba588d82cf0b18` |
| Fixed 269-tree report | `89cecf3158a591a4138d8a7c1fae5671b4801599114405a6efe5178b358ffc06` |
| Stopped Qwen continuation completion | `c8d0bbb6497778283bf1ba8a2b4dde33a31544f19fd10486d76fba434feac055` |

## Benchmark-question diagnostic

`tools/diagnose_native_spine_retrieval.py` seals all 100 retrieval records before
loading annotation data. Each request uses its own namespace and actual M
question date, one fresh BGE query embedding, 32 direct candidates, two lexical
reserve slots and at most eight added atoms from attention chunks. Hydration
shares the existing 3,072-token/128-read limits. No Qwen passes or raw content
reads occur during routing; raw text is loaded during cold materialization and
exact selected-section hydration.

The run performs 100 request embeddings plus one cold warmup. Both arms preserve
all baseline evidence, with **4,844 exact hydrated spans** in total. The cached
hierarchies bind to 472 occurrences across the 100 partial histories. Component
timers include one embedding and both arms' hydration; they are not independently
timed answer arms and must not be used for a matched latency pass.

The subsequent audit authenticates the pinned public M file and binds annotation
flags to the exact occurrence, normalized turn ordinal, role, timestamp and raw
text hash. Repeated session IDs cannot merge evidence. The M/S source plane and
summary ingestion never receive these flags or gold answers. Confirmation200
records are not analyzed. Annotation overlap is not semantic sufficiency.

| Stage | Questions with any eligible annotated overlap | All eligible annotated turns covered |
| --- | ---: | ---: |
| Admitted source availability | 21 | 8 |
| Direct candidate selection | 20 | 6 |
| Direct exact hydration | 20 | 6 |
| Candidates with attention context | 20 | 6 |
| Exact hydration with attention context | 20 | 6 |

Question 93 (a business milestone mentioned four weeks ago) is the one complete
loss among available-support cases. Question 65 (two hobbies leading to online
communities) retains only one of its two available annotated turns. These are
routing losses before hydration. They warrant general temporal/multi-fact
retrieval investigation; neither question text nor gold was changed. Completing
the source population remains necessary before the real joint evaluation.

Artifacts are under
`eval_results/native-spine-benchmark-routing-diagnostic-20260912-r1`:

| Artifact | SHA-256 |
| --- | --- |
| Execution policy | `60b9309b9061274a61df574c7c2d854e6815fdfda8b52d05b6bc23660ae0e34e` |
| Retrieval preflight | `261ce4bdf045b6556415ffe10e1702bb17a305385a5071e548eb0310033ff7a9` |
| Sealed retrieval result | `771a58437ebcbc16efb4f63d8132e863607b44dcdcc5b274ddb803d05f72a3a1` |
| Post-retrieval support audit | `f82819197a147141bbf738146bf3b3e8978977dc221e85d36acfb260997eb6e0` |
| Finished receipt | `e5610f5d3dea73fff4b61b9a72f83d37bd2364dbfa4ba049e845b9f8fa00c4be` |

Session 81227 exited successfully; PID 13804 is gone. Its driver
`.tmp/run_native_benchmark_diagnostic_20260912_r1.py` must not be rerun or
relabelled as an accuracy evaluation.

## Raw compiler interruption and continuation

The original raw compiler stopped and drained after **2,735 completed batches**:
2,614 accepted, 121 invalid summaries and 61,250 accepted atoms. It encountered
three timeouts (2733, 2734, 2737) and three HTTP 500 responses (2738–2740), leaving
11,071 requests never dispatched. Existing repair overlays do not relabel these
original validation counts. PID 65736 is gone and session 90673 returned exit 1.

The original execution report is
`native-spine-complete-body-summaries-20260912-r1/executions/229ae64927614f79a5355c4cbb15f3ae.json`,
SHA `33f7e3d7e6abd9355ba826f7c7f19886bd92fe4dfd005d07008658a31802d6c5`.
Its handoff completion confirms six failed jobs and an incomplete source corpus.

A read-only model-list check failed inside the network sandbox. The same check
outside it succeeded and listed Terra (chunk `f90484`). This proves gateway
reachability, not that the provider has recovered. Automatic approval review
approved the subsequent public-data continuation; no permission is outstanding.

`.tmp/resume_unsent_native_batches_20260912_r1.py` authenticates the full original
request population, binds the terminal report and public-source proof, and
selects only ordinals 2741–13811. Every selected checkpoint/validation path must
be absent. The six uncertain requests and all completed outputs remain intact.
It tries one new batch first, then uses the unchanged eight-worker dispatcher,
which stops new submissions and drains active calls after any execution error.
Automatic retries remain zero. Existing assemblers can read its normal original
request/response and validation receipts.

The prepared continuation policy SHA is
`5ce039d54749349fcd5f43c52a6936c89b881e96f55a96ced10ba0cfebd2f508`, under
`native-spine-complete-body-summaries-20260912-r1/continuations/unsent-20260912-r1`.
Session 98453 completed authentication and its first new batch **2741 was
accepted with all 24 summaries**, using one new call and zero replay hits. The
first-response receipt SHA is
`8fbfbac83ae2064d3311cfa3346c0a98f69d0f474ea30ef2872a2f6809c0150f`.
The continuation has released the remaining unsent requests to eight workers.
PID 56400, creation time 1789216671.9337437 was verified live (chunk `30d4ea`).
Consult `started.json`, `first-response.json` and `finished.json` for subsequent state.
Completing these unsent requests alone cannot complete ingestion: invalid
summaries and the six uncertain requests still require explicit repair/recovery.

## Ongoing local hierarchy work and checks

After verifying the diagnostic and previous Qwen processes had exited,
`.tmp/continue_native_hierarchy_4096_20260912_r3.py` started a bounded continuation
of up to 4,096 new local merge jobs, at most 128 per invocation. It reuses the
same authenticated summary-only backend and prior journals. Its control root is
`native-spine-hierarchies-20260912-r1/continuations/4096-20260912-r3`; policy SHA
`8a6a9f7b83c8ae9bf41f688e214c19e3f2edfb99fdeb29d656fc52b16c13dce2`.
Session 95218, PID 67944, creation time 1789216584.6193802 was verified live.
The actual checkpoint loaded and generated its first 44 new merge jobs, with
normal journaled outputs (chunk `e70018`).
Do not launch another GPU model while this exact process owns the stage.

Six extension tests pass, including changed pointer/text/source/compiler and
removed-body rejection. Five support-diagnostic tests pass, including repeated
session identities, normalized turn positions, partial interval coverage and
foreign/out-of-bounds evidence rejection. These add 11 focused checks, bringing
the native cumulative count to **147**; they do not establish answer accuracy.
All 17 diagnostic, 20 vector, 32 hierarchy and six raw-compiler preflight-bound
implementation files were verified unchanged after the real run. Seven new
Python files parse successfully. Existing staged work is preserved.
README diff whitespace and all five new tool/test/document files passed the
focused whitespace check (chunk `7f12eb`).
