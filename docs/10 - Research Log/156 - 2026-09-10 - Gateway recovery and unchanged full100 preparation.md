# Gateway recovery and unchanged full100 preparation

**Date:** 2026-09-10  
**Status:** bounded raw recovery running; six complete memories and 300 prepared answer calls preserved

Both model routes recovered after the timeout recorded in Log 155. The tested
successor scheduler now executes the preserved offset-060 recovery, followed by
the three untouched namespaces. The first six complete memories replay under
one extended admission method, and their 300 prepared answer requests remain
unchanged. No semantic-seed answer or judgment has been sent.

The latest measured comparison remains 41/50 for the existing v2 reader and
39/50 for each v3 successor in Log 154. This continuation adds no accuracy or
latency result. The joint 95%-at-1M target remains open.

## Preserved transport and admission

New implementation files:

- `tools/spine_session_transport_stage.py`
- `tools/spine_transport_lineage_v3.py`
- `tools/execute_spine_transport_recovery_v3.py`
- `tools/verify_spine_admission_method_v10.py`

The stage verifier binds the recorded terminal exec observation to the failed
scheduler's preflight, release, and failure receipts. It checks that the original
scheduler command is absent, rechecks the complete original inventory, and
reconstructs the exact work allowance. The recorded exec-session exit is the
terminal evidence; no original PID or creation time is fabricated.

The transport verifier authenticates every retained response and unresolved
request against the runtime's original protocol. It rejects changed or missing
successes, foreign requests, incomplete populations, a still-live original
scheduler, and a second unacknowledged successor. Every original reservation
remains in place. Recovery releases now also record their actual PID and process
creation time for future terminal observations.

The new admission verifier preserves the old v9 path for earlier memories and
adds full source/compaction replay for the terminal-session raw recovery. The
raw recovery adds no compaction allowance; a separate compaction transport stage
must still be declared and verified. Summary compaction, bounded invalid-slot
recovery, and exact raw-fragment admission retain their existing methods.

Session 14742 verified the real staged corpus and prepared its execution:
403 preserved responses, 383 first attempts, six explicit reissues, and 389
maximum new calls. Execution preflight:

`eval_results/spine-transport-recovery-offset060-20260910-r1/execution-raw-v3.json`

SHA `1366cea05dd9efd3e628c3dec95e497be872839984aa6346b567fd5e33ee9a62`.

Session 27320 replayed all six complete admissions without provider calls or
changes to any admitted atom:

`eval_results/full1m-spine-admission-six-memories-v10-20260910-r1/verification.json`

SHA `32d17e2706806d35fbde961542a2645b82a51c32f07e7ba8638a0a87ee29d9f5`.
Common method: `8c1894fd1ce4722b1e468a32e5759350c761d6baf57f88dc6560a6b1660ee4f8`.
The sixth memory still accounts for eight compaction attempts. The seventh
memory's full admission remains pending until its raw recovery completes.

## Gateway observations and resumed execution

The second post-timeout readiness check again returned HTTP 500 from both
routes. Its report is
`eval_results/full100-spine-remaining-corpus-20260910-r1/gateway-readiness-after-timeout-r2/report.json`,
SHA `a34a89ffce734dd0c75c51b2a94381924cf2b8a204d6bf640b4cbfb6c6b4e775`.

At 14:58 UTC, a fresh pair completed: Qwen in 1.616 seconds and Terra in 5.339
seconds. These were synthetic readiness inputs, not benchmark answers. Report:

`eval_results/full100-spine-after-offset060-timeout-20260910-r1/gateway-readiness-r1/report.json`

SHA `b6523f571f6640811a7552cf9951402682725e3b41c1d9f825f8adf37df00b15`.

The successor scheduler is
`eval_results/full100-spine-after-offset060-timeout-20260910-r1/run_after_timeout.py`.
It preserves all 403 successes, completes the 389-request offset-060 recovery,
then executes the original 838/830/868 requests for offsets 070/080/090. The
maximum new raw-call count is 2,925, including the six explicit reissues.
Automatic retries remain zero. A new failure stops all following namespaces.

Session 78361 prepared the scheduler with zero provider calls. Its preflight is
`ed2c488757508298b4f88410713fa0f1d52fea3f334b5009bceb0387ac776033`.

Session **42892** is the live provider execution. It released at 14:59:24 UTC,
using PID **63016**, process creation time **1789052354.6007824**. Release SHA:
`073f4813e0cd9d4a3f5ef5a752af37f529c4659028606cc50144d454402a549a`.
The offset-060 transport release is
`16f45327efa0b0d4d4071b16a21d740a9cce15fcb243f3e8fbd2f06534e55b0a`.

The live output confirms responses for all six previously unresolved batch
indices. At the latest journal inspection, the successor held 468 responses
and 472 requests, including the 403 preserved successes. This is ongoing
ingestion, not a completed namespace. Do not independently launch any remaining
namespace or rerun the staging program over the active successor.

## Same answer experiment, updated verification

`tools/report_joint_spine_semantic_seeds_full100_v2.py` uses admission v10.
`tools/run_spine_semantic_seed_full100_v2.py` requires the successor scheduler's
complete result, replays its preserved transport accounting, and still requires
all ten complete memories and all 500 fresh answer calls before judging. The
accuracy threshold, both API latency baselines, 10% allowance, reader, routing,
evidence budgets, and counterbalanced call order remain unchanged.

Session 25047 transferred the existing preparation to
`eval_results/full1m-spine-semantic-seeds-full100-20260910-r2`:

- `protocol.json`: `c8b27afa6589d548c0225afef120ae1a61929822febf08f5fd03508e173da99e`
- `preparation-first60.json`: `d08721a9b6aeca06eec89417991bdbb9b773b48e93fa56fbfd869d25f6ab6765`

All 300 calls retain their original namespace preflights from Log 155. No query
embedding, model call, cached prediction, or benchmark reference was needed for
the transfer. The six new admission bindings and passage verification receipts
match those same complete memories. The real v2 runner checked all six and
stopped at the missing offset-060 preparation without publishing a runner plan
or execution reservation.

## Validation and next work

Separate checks completed:

- Nine transport replay/execution checks and seven terminal-observation checks.
  The latter initially failed in the fixture because Windows translated a SHA
  sidecar newline; writing its exact bytes corrected the fixture. All seven then
  passed in 4.23 seconds, without changing the implementation.
- Six admission-v10 content, recovery, and compatibility checks in 7.14 seconds.
- 183 shared full100 gate and passage-verification checks in 4.28 seconds.
- Six remaining-ingest scheduler checks in 1.50 seconds.
- Eleven full100-runner and preserved-bulk-completion checks in 72.77 seconds.

The content-admission tests exercise native raw/compaction replay with a stubbed
transport result; transport is exercised separately with native runtime journals
and on the real staged corpus. Full end-to-end verification of offset 060 still
requires its complete responses and admitted atoms. These checks are not answer
accuracy evidence. `git diff --check` passed.

Continue polling session 42892. After each remaining raw namespace completes,
audit and admit its entire source population, compile the existing attention
leaves and all summary indexes, then prepare its ten questions under the r2
campaign. No new memory-compilation worker has been launched. Finish all 100
questions before the isolated timed comparison and judge those same predictions.
Confirmation remains unopened by this continuation, with Log 102's historical
exposure qualification retained.
