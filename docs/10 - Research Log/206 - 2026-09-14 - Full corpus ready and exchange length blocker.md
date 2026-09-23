# Full corpus ready and exchange length blocker

**Status:** The complete 31,166-body source store is ready. The full-corpus
pipeline stopped in exchange compilation on one exhausted 129-token summary
against its 128-token limit. A separately recorded single-row retry reproduced
the same length failure. No native joint full100 accuracy or speed result exists.

## Full source admission completed

Source completion `native-spine-source-completion-20260913-r1` finished
successfully after the four-call final repair. It admitted all 31,166 bodies and
verified preservation of the R6 store. Its process has exited.

| Artifact | SHA-256 |
| --- | --- |
| Complete source store | `721ef21c4e1d439cb30346c1abf0a6c25f1b19cd61ff52f8ce51f2969bd7e267` |
| Previous-store preservation | `e7125809d1ae062de045bf78b62ae6ae72be07d20c0b9f1f08b2802fd7dd2dc0` |
| Source completion | `5f72275e32fabc2c3abe642bdceaa7e1700a827f8cdb974029142f440cc1df9c` |

The full-corpus pipeline released exchange-input preparation successfully.
Its inputs SHA is
`ef1258514de71b56f9287af46249b38f20ee337d0496d15a175ecdde0fcf0c51`.
These are the complete source inputs, not the earlier 13,468-body snapshot.

## Failed exchange stage

Pipeline root: `eval_results/native-spine-full-corpus-pipeline-20260913-r1`.
Its failure SHA is
`26884fa8ccc25568f9177d44c0d5d778cde2bce67690cf8fa90cf11fd97ce704`.
The original controller and exchange-stage processes are gone, and the stage
exit record is one. `stages/exchanges.log` identifies the failure as exhausting
the local summary refinements. The subsequent attention, parent, vector and
evaluation stages were not released.

The last compilation scan completed 31,063 body exchange sets and identified
103 pending merges. Generation then accepted additional results before raising
the refinement error. A read-only journal replay, without loading Qwen, found
832 accepted own merge keys and exactly one exhausted own key:
`b6a34579a998dce948393adb8f65f22eafc2346c665deb328259d0591526ac4c`.
The failing attached-context summary reached EOS but contained 129 tokens;
its limit is 128. Variants 0, 1 and 2 had all been attempted. The final original
response came from a batch of four.

The exchange preflight SHA is
`067bedde9bd4e7994595605b16d610eede9c04865a9821ce2938c7d623eaab74`.
Accepted journals and compiled artifacts remain intact. The precise remaining
body population requires another compiler replay after an accepted resolution;
the final pre-error scan is not a count of all subsequently accepted merges.

## One explicit additional attempt

Driver `.tmp/retry_native_exchange_final_label_20260914_r1.py` ran one explicit
single-row retry with exactly the original summary inputs, message construction,
backend identity, generation options and output limit. It made no remote calls
and supplied no raw source text to Qwen.

The separate policy records that this is an additional attempt after the
original refinements were exhausted. A new request and actual response were
appended with an explicit recovery-policy reference; every prior request and
response file remained byte-identical. No rejected response was rewritten or
accepted by relaxing its budget.

Recovery root: `eval_results/native-spine-exchange-explicit-retry-20260914-r1`.
Policy SHA: `7083708bfb8670daf2c93160ce3e36ae1cfeef281fd7dff5978b39808c6c806b`.
Result SHA: `e75a37fe15a04347c437ea98806ab3bc0c1fddd3177e5e7aa3a4db4e1c18479d`.
The retry again reached EOS with 129 tokens and failed the unchanged validator.
Session 32212 exited one; process 26232 is gone. No further retry was launched.

The draft `.tmp/resume_native_full_pipeline_20260914_r1.py` was not launched.
It requires an accepted recovery, which does not exist. Its intended continuation
retains prior exchange journals, deducts already-executed generation jobs from
the original allowance, and runs later GPU stages in separate processes. It is
not an active handoff or evidence of completed work.

## Speed-test status

No native full100 answer streams or joint latency report have been produced.
The required test still measures retrieval, exact hydration, prompt construction
and the API answer together. Both time to first token and total latency must
meet median and p95 ratios of at most 1.10 against identical-evidence and
short-chat API controls, with at least 95% answer accuracy on the same fresh
100 separate histories of at least 1M actual body tokens each. Offline ingest
and compilation timings do not establish this serving-latency target.
