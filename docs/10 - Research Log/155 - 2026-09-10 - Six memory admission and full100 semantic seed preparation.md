# Six memory admission and full100 semantic seed preparation

**Date:** 2026-09-10  
**Status:** six complete memories verified; 300 requests prepared; no new answer score

The six complete memories now use one replayed source-admission method. The
semantic-seed comparison has 300 prepared requests covering their sixty
questions. The full100 execution protocol still requires ten complete memories,
500 fresh answer requests, and accuracy and latency measured on those same
answers. No candidate answer or judge request has been sent.

The latest scored result remains Log 154: the existing v2 reader scored 41/50,
against 39/50 for both v3 successors. None passes the joint target. The earlier
95/100 remains the cumulative retrieval and answer-repair validation result;
it did not demonstrate the required end-to-end latency.

## Common source admission

`tools/verify_spine_admission_method_v9.py` extends the frozen v8 verifier with
`tools/spine_compaction_transport.py`. It checks a separately declared
compaction transport stage, preserves every original successful response and
the unresolved original request, and requires the reissued request to use the
identical protocol. It permits at most one additional transport attempt and
rejects omitted or changed stage bindings. Raw source text is not sent to Qwen.

The sixth memory therefore accounts for eight compaction attempts: the blocked
original, its completed reissue, and six bounded slot-recovery calls. Its old
v8 receipt counts seven and remains preserved as the predecessor.

Session 40129 completed successfully. All six admissions replayed with zero
provider calls and unchanged admitted atoms:

`eval_results/full1m-spine-admission-six-memories-v9-20260910-r1/verification.json`

SHA `f9e87f17aaa1e2a235230dfdf989441d18077824fba86a7ab928eaae3a145312`.
Common method: `f972a565e4f54f4495593bac1053c683900cc429dafd271eaa2883a593e0c7bc`.

| Offset | Raw token proxies | Compaction attempts | v9 admission verification |
| --- | ---: | ---: | --- |
| 000 | 1,041,276 | 1 | `12397494d15bf893f62c0bd4b5bf2dd95952675fa7475ae8cb6391d33c469e19` |
| 010 | 1,044,341 | 2 | `662374c63a701c8ce3010055a77ea6eae1def60d98b43bdcafac85d57afef747` |
| 020 | 1,045,527 | 3 | `1bd2674803fc60a0e63d47e6b9d25790a48428d28e646df7dcb3669521911ce3` |
| 030 | 1,043,571 | 1 | `b137ddcd0abc46d5e822bd0777b9c870a8887b72e71067d1ca74143a2a380b03` |
| 040 | 1,046,567 | 4 | `6806b6fc187989f7472137fa6d371bddb91dcf2ae3d0c2fa5ce5c4391007e649` |
| 050 | 1,051,365 | 8 | `1013e97a36561b5e5a6f0ed07a101d67dd711c09d56ba9deaee58344528d680a` |

Session 47592 also completed all three sixth-memory indexes. Under `eval_results`:

- `full1m-spine-semantic-offset050-20260910-r1/index.json`:
  `6812d92c05f7047cfd1ef90ed305d58bc44e47e94b074ed7629a6b3d014aa966`.
- `full1m-spine-user-addresses-offset050-20260910-r1/addresses.json`:
  `202f54ad8b4f547a68295780d06a330a4ca97f94a285a09a89ff543b0b984d87`.
- `full1m-spine-facet-addresses-offset050-20260910-r1/addresses.json`:
  `87fe7af24ad3479947e63e817f79df53d315aeb9a835d9542f15e4769e1ce605`.

The sixth memory has 2,712 attention-partitioned leaves and 5,876 summary
passage addresses. All six retain the existing common leaf compilation policy.
Parent summaries remain deferred. Query-time routing uses BGE summary vectors;
this candidate adds no query-time Qwen call or embedding pass.

## Fixed full100 comparison

New evaluation files:

- `tools/evaluate_spine_semantic_seeds.py`
- `tools/report_joint_spine_semantic_seeds_full100.py`
- `tools/run_spine_semantic_seed_full100.py`

Both memory arms use the existing v2 reader, the 3,072-token raw evidence budget,
and the 128-span limit. The candidate changes only the first six whole-summary
seeds from two reserved lexical matches plus dense matches to dense summary
matches. User and calendar supplements, source diversity, and scoped terms
stay fixed. The candidate can displace earlier user spans; there is no claim
that it preserves all earlier evidence.

Each question has five calls: a shared short API control, the existing memory
arm and its identical-evidence API control, and the semantic-seed memory arm
and its identical-evidence API control. Matched pairs are adjacent and alternate
which method runs first. All five calls use the same reader policy and question.
Live embedding, routing, and exact raw hydration remain inside the memory clock;
prepared evidence is used only for API controls and reproduction checks.

The full100 runner requires every namespace before releasing any answers. It
checks completion of the remaining bulk ingestion, local workload isolation,
and fresh gateway readiness. It authenticates all 500 answer journals and seals
the complete answer population before any new judging. A failed request stops
execution without an automatic retry or clearing its reservation. Final
reporting replays judgments and verifies the shared admission, leaf, and passage
methods across all ten memories.

Campaign root:
`eval_results/full1m-spine-semantic-seeds-full100-20260910-r1`.

- `protocol.json`: `607388b699a19fca34a544b98fb98c3be6dd36b0cb0fd05b7daa4481b259ccf3`
- `preparation-first60.json`: `94166fa0babe463a4db6bb095f1cfbf15d03eca65359280e835d9571f8a1847f`

Session 87468 completed the first-six preparation with zero provider calls.
All fifty prior control prompts and all fifty live-audited candidate prompts
match exactly. Every complete memory passes the common evaluation policy and
passage/leaf compilation checks.

Namespace roots follow
`eval_results/full1m-spine-semantic-seeds-joint-offsetNNN-20260910-r1`:

| Offset | Prepared calls | Preflight SHA |
| --- | ---: | --- |
| 000 | 50 | `da6403d7b12cf9d03995637c225f6013ff7d9003469c9af48e479d1e9da5fe7a` |
| 010 | 50 | `d262341ba24edbcd08d4014825b9428139df8de98ed946d3db4174aba38d44d9` |
| 020 | 50 | `d772f2cd21add2c491c2597a6810f396f2eb8f099b4f411b9838c8e22acfcf8f` |
| 030 | 50 | `2c00df1838794395ebf095736c6f7118e7b4f28b1a83f9b06560949375b2df76` |
| 040 | 50 | `3f3daa9fbac7c67c86a89c9e4e1ba38437ef943061d398a41a883213637306b6` |
| 050 | 50 | `a29a10aabd6eccc02120baa5e253cc6a8cd46c31b5a4324015990f6d4762be34` |

The real full100 runner validated these six preflights and stopped at the
missing offset-060 prepared binding. No runner plan, execution reservation,
answer, or judgment was published. The first-sixty preparation is not a
full100 launch or a measured result.

## Validation and continuation

Transport/admission checks passed: 14 tests in 119.55 seconds. The semantic-seed
evaluator, full100 gates, passage verification, and routing checks passed:
178 tests in 5.60 seconds. These are separate test runs.

The new full100 runner checks covered all-answers-before-judging, incomplete
memory populations, missing answers, a preserved timeout reservation, and
busy/unfinished/unready dependencies. Six passed initially; the seventh expected
`FileNotFoundError` instead of the artifact library's `SealedArtifactError`.
Correcting that test expectation passed in 1.62 seconds. The initial failure
did not require an implementation change. `git diff --check` passed.

Raw-ingest session 84398 subsequently exited with code 1. Tool chunk `882c43`
reported `openai.APITimeoutError` caused by `httpx.ReadTimeout`; the scheduler
published its failure receipt and preserved all journals. The final inventory
contains 403 completed responses, six unresolved requests, and 383 unstarted
requests. The six unresolved batch indices are 401, 402, 405, 406, 407, and 408.
The workers reserved two additional requests after the earlier four-pending
observation. Offsets 070/080/090 remain unstarted.

Terminal observation:
`eval_results/full100-spine-remaining-corpus-20260910-r1/raw-timeout-offset060.json`,
SHA `468a35e0b887d31d51f75b0d530985dfe32ad93e4b59d9874205615f0d31a52d`.
It records the terminal tool session and verifies absence of the original
`run_remaining.py` process. The original PID was not captured; no PID or process
creation time is invented to satisfy the older stage-v2 observation contract.

Two fresh synthetic readiness probes ran through the authorized gateway path.
Both Qwen and Terra returned HTTP 500. Session 20411 completed its observations;
it did not resend original requests. Report:
`eval_results/full100-spine-remaining-corpus-20260910-r1/gateway-readiness-after-timeout-r1/report.json`,
SHA `a94f7d9401705baefc3ffa795ea879417edebccf44fd2bcd9ca86580e95338f3`.

The separate recovery root is
`eval_results/spine-transport-recovery-offset060-20260910-r1`.
Its `stage_from_terminal.py` authenticates the terminal inventory, replays and
copies completed responses, preserves all original reservations, and prepares
at most 389 future calls: 383 first attempts and six explicit reissues. This
stage makes no provider calls. Session 73479 completed successfully, replaying
and copying all 403 successful responses unchanged:

- `stage-preflight.json`: `ffe2e228e75295bdbdab6a3efe3efa652021d8aace0502f0206da8241a46286f`
- `stage.json`: `f7e892006fc64b29dfdb2c392fd14d2de45302368e2e15e7eaeb4cbc0a964cc7`

The stage uses a distinct v3 format for the terminal-session observation.
Its transport lineage and execution must be implemented and tested before
provider release or full100 admission. The frozen v9 admission method and
original-scheduler completion dependency do not yet cover this new stage;
extend them through an explicit successor without changing the six prepared
memories' content. All timed evaluation remains unstarted.

Next: finish and verify the preserved recovery stage, restore successful gateway
readiness, and continue the four remaining complete admissions and indexes.
Prepare their 200 requests under the same answer/routing protocol and an updated
common transport verification, then run all 500 answers with isolated timing.
Judge those same answers and apply the full100 accuracy and latency gate.
Confirmation remains unopened by this continuation; retain Log 102's historical
exposure qualification when it is eventually evaluated.
