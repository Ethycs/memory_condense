# Qualified reader and complete memory development40 preflight

**Date:** 2026-09-10  
**Status:** four complete indexes; 240 requests prepared; scheduled execution stopped before calls  
**Predecessor:** [144 - Fresh passage routing development30 result](144%20-%202026-09-10%20-%20Fresh%20passage%20routing%20development30%20result.md)

**Current continuation:** [Log 146](146%20-%202026-09-10%20-%20Fifth%20memory%20gateway%20timeout%20and%20preserved%20recovery.md)
records the fifth ingest's confirmed timeout, both original processes' terminal
exits, failed HTTP-500 readiness probes, and a recovery stage preserving 99
completed responses. No reader answers were sent. The waiting-runner notes
below describe the earlier checkpoint and must not trigger a restart.

The preceding turn made concrete progress: the fresh passage comparison sealed
26/30 versus 25/30, the fourth namespace was admitted and replayed, and all four
memories acquired the same verified source-admission method. This continuation
finished the fourth memory's indexes and prepared an experiment aimed at the
remaining reader failures. No new answer-accuracy result is claimed yet.

## Fourth complete memory

All 5,516 exact raw fragments, 5,514 turns and 473 sources are represented by
2,751 attention-guided leaves. The namespace contains 1,043,571 token proxies.
Its semantic and whole-user summary vectors are complete, along with 5,849
exact-passage addresses derived only from stored user summaries.

| Artifact under `eval_results/` | SHA-256 |
| --- | --- |
| `full1m-spine-leaves-offset030-20260910-r1/hierarchy.json` | `bd5d46cc0d8f43208ec85c08a67f2575a40a6b17d78889d3570deac2c88c211e` |
| `full1m-spine-semantic-offset030-20260910-r1/index.json` | `babfb1228f83773f11b903a2f24296bfc16537658a5470e8c7f67eb746488171` |
| `full1m-spine-user-addresses-offset030-20260910-r1/addresses.json` | `d3238a78d4b5ed2a1eeaa5ba5d9107bc115ec5abbaf6ca5cfd6240d04c9450d4` |
| `full1m-spine-facet-addresses-offset030-20260910-r1/addresses.json` | `aae8c2c34aed4c48b6ea2106b22ee0816357b82b11b91ac39825cc6d8bc60ccd` |

The aggregate `fourth-memory-indexes.json` in
`full1m-source-spine-facets-development30-20260910-r1` has SHA
`21481554a26c1499238643e077ba56d254398895a3f5a7d2696cbb0eff79aff3`.
Compiler session **72839** is complete. All embedding inputs were summaries;
no question, answer or gold selected the compiled population.

The existing overflow-versus-passage comparison for offset 30 also has a
50-request preflight, SHA
`fb1b16d2b41416d4b369d876fbf53bc8de2f68f9206e6cc6f15d9fdbed0d77fd`,
under `full1m-source-spine-facets-joint-offset030-20260910-r1`. Those requests
have not executed and are not part of the scheduled reader experiment.

## Reader treatment

`src/memory_condense/eval/spine_reader_policy_v3.py` replaces the v2 reader's
mandatory two-sentence recommendation structure with a shorter policy:

- Match the precise entity, activity, qualifications, time window and event
  status. Do not assume a question's premise or transfer facts from a similar
  activity.
- Count distinct qualifying items; separate completed actions from intentions
  and avoid counting duplicate mentions or broader overlapping labels as
  additional items.
- Attach specific preferences, established interests and compatibility
  constraints directly to recommendations. Avoid synthesizing a current
  inventory from disconnected or conflicting excerpts.

The policy retains explicit temporal boundaries, relative-date resolution,
newest relevant updates, approximate current quantities, supported arithmetic,
description-only entity identification and abstention for unsupported answers.
It includes no benchmark identities, reference answers or domain-specific
corrections. Its effects must be judged on fresh answers.

System-prompt token proxies decrease from 360 to 259. This is a 101-token
reduction in system instructions, not a 28% reduction in the whole evidence
prompt. Policy hashes:

- v2 control: `385d3e864bd95b700cbdd8a30a34883cdc3e3020e219168352fc3bf37a2bd81b`;
- v3 candidate: `99b3d2103aa6f4938a464d6c45bd40ed2762ffc3275d6ad09fe4383f2c6877ae`.

## Matched evaluation

`tools/evaluate_spine_reader_v3.py` holds passage routing, exact raw hydration,
the 3,072-token evidence allowance, Terra answer model and 256-token output
cap fixed. Both methods retrieve inside the end-to-end clock. Each reader has
its own short API baseline and its own byte-identical-evidence API control;
six counterbalanced requests per question give 240 answer calls for 40
questions. Only the memory-arm predictions are scored. Sol judging remains
zero-retry with the same answer/prediction bindings.

The four preflights cover every question in offsets 0, 10, 20 and 30, without
selecting only prior misses. Every control prompt is byte-identical to the
existing passage-routing prompt. Candidate and control user/evidence messages
are byte-identical for all 40 questions. No predictions or reference answers
were inputs to preparation.

Development root:
`eval_results/full1m-spine-reader-v3-development40-20260910-r1`.
`preparation.json` SHA:
`a60a834e4dad8a92040bf75b583ba7c6b483bb853e96bdb6998186f896856404`.
The four `full1m-spine-reader-v3-joint-offsetNNN-20260910-r1` preflights are:

- offset 0: `86ef8e9be7cd3ce03580295e3fb6b5a3d8e35936ebce8a7da6caa8051242969c`;
- offset 10: `0e4448dfc21e18005879c128b53035ff66a52c9cd302a0f3d8f9c77a03a63f74`;
- offset 20: `fc3d91af63a2950e499611102f331233e106b57de4053682eb778b0bb314d7b6`;
- offset 30: `dbbe1f19efeb3150b7a222a32f1d2d94fc0b79beaca2420788395812b7d210fd`.

The new reader policy and evaluator are now bound by these preflights. Do not
edit them and silently reuse this population. Preparation session **95939**
finished with zero provider calls.

The full100 successor `tools/report_joint_spine_reader_v3_full100.py` retains
the complete-source and passage-population verifiers, identical method checks
across all ten memories, all 100 question identities for each reader, and the
joint 95/100 plus provisional 10% median/p95 TTFT and total-latency gate against
both API baselines. Forty questions remain development evidence even if they
pass; the full100 objective and unopened confirmation set are unchanged.

Eighty-two focused tests pass across live retrieval and matched policy controls,
interrupted-call handling, full100 gates, complete passage verification and
streaming latency. `git diff --check` passes for the changed test files.

## Scheduled execution and continuation

Raw ingest session **38496** is still running the complete 837-request offset
40 population. Its actual Python process was verified as PID **2856**, creation
time **1789027208.5106108**. This is a live process binding, not a stale lock.
The existing raw execution preflight is
`70383d4318a15c185b7bffbca47bc8fa57a8cfc323d8a1f4a2905aa54f35c495`.
Offsets 50 through 90 remain prepared and unstarted.

Session **73512** runs the development root's `run_comparisons.py`. Its sealed
`runner-plan.json` has SHA
`7d98d7547e17dae71c4b7431bfe424edb1f8afa160b91811d97c057c311e4d9e`.
It waits for that exact ingest process and checks its creation time and command
hash. A failed or incomplete ingest stops the runner. After normal completion,
it authenticates all 837 raw responses and the full aggregate with zero calls,
then starts the 240 serial answers and at most 80 logical Sol judgments. It
replays every judge population without calls before writing the combined
`development-report.json`. It never retries an uncertain answer reservation.

**Do not start other provider work, GPU compilation or large replay jobs while
session 73512 is active, including its dependency wait.** The automatic
transition can begin as soon as ingest finishes. Light status and documentation
work can continue. Poll sessions 38496 and 73512; do not duplicate them or treat
an observation timeout as terminal. A missing handle requires inspecting the
actual process and artifacts before deciding whether any work stopped.

After this runner finishes, diagnose the full fresh 40-question comparison and
continue complete source admission for offset 40 and the remaining five
namespaces. Keep the current 26/30 result distinct from any future result.
