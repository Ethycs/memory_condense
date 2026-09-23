# Ten complete memories and live full100 evaluation

**Date:** 2026-09-10  
**Status:** historical execution checkpoint; full100 completed in Research Log 166  
**Predecessor:** [164 - Ninth memory admission and final raw namespace](164%20-%202026-09-10%20-%20Ninth%20memory%20admission%20and%20final%20raw%20namespace.md)

**Successor:** [166 - Full100 as-of result and reader failure audit](166%20-%202026-09-10%20-%20Full100%20as-of%20result%20and%20reader%20failure%20audit.md).
All 500 responses and 200 logical judgments completed, followed by the full
source-admission and judgment replay. Session 13811 exited zero. The result is
80/100 for the cutoff and 74/100 for semantic seeds; both joint gates failed.
Live-session instructions below describe the earlier checkpoint and are
superseded. Do not restart or poll session 13811 as an active process.

All ten approximately 1M-token memories are fully admitted and indexed. The
frozen semantic-seed versus as-of comparison has all 500 requests prepared,
both fresh readiness probes passed, and serial answer execution has started.
At 19:18:57 UTC, twenty answer responses were saved. No new accuracy or joint
latency result is claimed; all 500 answers must be sealed before judging.

## Complete ingestion and final source admission

Raw scheduler session **42892** finished with exit code zero. Its sealed
completion accounts for offsets 060/070/080/090, with 792/838/830/868 logical
raw requests. The 2,925-call successor allowance retains the earlier 403
completed offset-060 responses and six additional original unknown attempts;
no failed reservation or earlier response was discarded. The raw scheduler's
completion covers ingestion, while the separate compiler completion below
covers admission and indexing.

The final namespace contains **1,046,567 raw token proxies and 5,624 admitted
fragments**. Its strict initial quote diagnostic marked 573 batches invalid;
the complete source audit found **zero unresolved schema failures and eight
oversized summaries across seven batches**. All eight compacted in one Qwen
batch, with no invalid-slot recovery calls. Native v11 verification reproduced
the admitted artifact without new calls and retained 2,217 quote-diagnostic
annotations. Summary entailment remains unverified.

All paths below are relative to `eval_results/`.

| Ingestion or admission artifact | SHA-256 |
| --- | --- |
| `full100-spine-after-offset060-timeout-20260910-r1/complete.json` | `85312e76c11dc72e719a95a4b7abab28fe5a94c57f98c46dc737365f776fad64` |
| `full100-spine-after-offset060-timeout-20260910-r1/completed-offset-090.json` | `1b4e953d425e1bb4dfec6044d0dc9dee911a79e495c8aec2da446f1b0b8e3119` |
| `full100-spine-corpus-20260909-r1/offset-090/atoms-prefix-0868.json` | `00c4b08520067e26e905cbeb2a113424f8abbd09212c0717bfa842c832dcaee9` |
| `full1m-spine-source-admission-offset090-20260910-r1/audit.json` | `5ece6b4fbdbf31c9b21995431537fb60abacdaa4c8895ebaf5e16f08776f2225` |
| `full1m-spine-budget-repair-offset090-20260910-r1/preflight.json` | `8c5384e5d7f5dcdd8fa6fdefb25476fd9cb852187c1b83a71713a7a557e161d1` |
| `full1m-spine-budget-repair-offset090-20260910-r1/repairs.json` | `5a04adfdf3f49931422fe89e54da04831637848c6c5402ae1bb85012009265df` |
| `full100-spine-corpus-20260909-r1/offset-090/source-bound-atoms-prefix-0868.json` | `3ca9b1be6a896b6d5e99fd07c4d94f0993fbf852f1a7d8b868e95c738b62b6db` |
| `full100-spine-corpus-20260909-r1/offset-090/conditional-method-v11-prefix-0868.json` | `41113298d5862984f53061e5380eadceae388d3ac83cafd9a4ea820ef5a2449f` |

All ten memories share admission method
`1b6ba1149c3962eb3a16e1fb55b6b2e16d8db811e274f9dc3ff21db3fe4d601a`.
The complete raw receipts were read and their sidecars checked in exec
`6d5746`. Raw ingestion was not restarted after its successful exit.

## Final attention leaves and indexes

The final memory has **2,794 attention-guided leaves across 496 sources**.
Leaf construction used fourteen summary-generation calls, including its
bounded exchange-summary repairs; local attention-forward computation is
separate from this call count. Parent summaries remain deferred under the
unchanged compilation policy. Qwen received summaries only.

The semantic compiler verified exact whole-turn reconstruction. Both fragment
and whole-turn token counts are 1,046,567, with no nonadditive turns and no
mismatch tolerance. User addresses cover all 2,794 leaves, and passage
indexing produced **6,007 addresses**. The three index stages made no provider
calls. The v3 compilation scheduler then prepared the final original control
namespace and validated all 500 original control requests.

| Final compilation artifact | SHA-256 |
| --- | --- |
| `full1m-spine-leaves-offset090-20260910-r1/hierarchy.json` | `7ec3488a63a45b1a58289b4a6b8b375b2196b5045934a30006c727f6f6c254a7` |
| `full1m-spine-semantic-offset090-20260910-r1/index.json` | `9e59be36790e276374e423473dec650430c906aa854b12ad7f6b336971ea9d0c` |
| `full1m-spine-user-addresses-offset090-20260910-r1/addresses.json` | `0a02a5bec30a0d2497008e2a797d4cc0760a420e30ed21a93f3cd9f0a3def55f` |
| `full1m-spine-facet-addresses-offset090-20260910-r1/addresses.json` | `d6610dcedf41f9f3ecfe2c98416954735af14538dd577744491f66fa592114fe` |
| `full1m-spine-semantic-seeds-joint-offset090-20260910-r1/preflight.json` | `1cca6eff7dff5fd0bfbfe2f006e7e359c0f052b426739e42b0b76a153a670b3d` |
| `full100-spine-memory-completion-20260910-r3/offset-090/complete.json` | `fdbc07d1d4d087ebf1e4b1b635685ecdf9ca20a7b93c94421af233c796130b4d` |
| `full100-spine-memory-completion-20260910-r3/complete.json` | `3a900668eb43c0b3c5b452391c04ad695e7e6d9c160b5d0697d978eade4ddb87` |

The original control runner plan SHA is
`df15a5abceb246aecaa710f8c498c3504523ea932474da414693b8595f103bf3`.
Compilation session **48873** exited successfully (exec `ccd1db`). Its complete
artifacts and exact token-accounting fields were read again in exec `97be23`.
That older answer campaign remains unexecuted; its controls supply the live
as-of comparison.

## Final preparation, readiness, and timed release

Handoff session **13811** verified both terminal dependencies and their sealed
completions, then prepared the remaining as-of namespaces in separate local
encoder processes. Every timed memory request still recomputes query embedding,
summary routing, date projection, and exact raw hydration inside its clock.
Resident setup time is recorded separately. No cached predictions or cached
query vectors serve the benchmark requests.

The live campaign root is
`eval_results/full1m-spine-as-of-full100-20260910-r1`.

| Live campaign binding | SHA-256 |
| --- | --- |
| Unchanged protocol | `09e2b5c8c69a509bc23b2f62e24e0cb53507e30389127db1e6e7d67797ba29ad` |
| Offset 080 preflight | `16c947a1568b26358f9bbc0cd5357153a760a8786d986f04479f492620c06e7d` |
| Offset 080 prepared binding | `58e3a670af4d087af357723309fdd15a3ef77c2975549ff00986b22707a32aaa` |
| Offset 090 preflight | `eeab08e882b5c3c00ebb772ca35dca86dbecfcb5e19238aca3065e45af200838` |
| Offset 090 prepared binding | `b63b46ff21b43c602228f49fbb71db5226f4dc7a8e9c0d58e3af665f05bf3c80` |
| Full100 runner plan | `092b2d83409882eedaec32e97d21f4b560bd9739241b51ce6ddb19c0608e8ed1` |
| Readiness report | `5704a6be4b35dcf8caeff9bf488b9b24affba3b326cf05e99264c8b1e4a6d91d` |
| Timed evaluation release | `d9bc4f9adc3e511f5cdcdcfc69b54c4693bf3bee37ca1174cd594d793817ad6c` |

The two synthetic readiness calls completed successfully, with zero retries.
The timed release began at **19:16:30 UTC**. The first five streaming requests
completed in exec `0e51bc`; twenty saved answer responses were observed at
19:18:57 UTC in exec `0fa94f`, which also checked the evaluation release seal.
Individual timings are not an aggregate latency result or an accuracy score.

The first namespace subsequently sealed all fifty answers, SHA
`f9fc7a541d4d56470ae8d54307e4ffab806862569227bc609cbaaffce896952d`
at `namespaces/offset-000/answers.json`. The runner advanced into the second
namespace without judging (exec `d72560`). This is an execution checkpoint;
all 500 answers and the subsequent judgments are still required.

Handoff execution root:
`eval_results/full1m-spine-as-of-after-compilation-20260910-r1`.
Its execution release SHA is
`7b25d01639ebde93a26483a14c6e55432c7c6364b8c8d40879e64f7f4240ddeb`.
The Python executor PID is **64500**, creation time **1789067636.2072325**,
under the existing PowerShell handoff session **13811**. Raw session 42892 and
compilation session 48873 are terminal and must not be restarted or polled as
active dependencies.

Continue observing session 13811. Keep other model calls, local Python jobs,
GPU work, and tests out of the timed comparison. All 500 answers must finish
and seal before the existing runner performs at most 200 logical Sol judgments.
It will then replay judgments and publish `joint-full100.json`. Inspect that
report and its bound predictions/timings before claiming any target pass.
The same arm must achieve at least 95/100 and the unchanged 1.10 median/p95
visible-TTFT and total-latency limits against both API controls. A failed
request preserves its reservation and stops; do not retry or replace it inside
this frozen execution.
