# Ninth memory admission and final raw namespace

**Date:** 2026-09-10  
**Status:** nine complete indexed memories; 450 control requests and 400 date-comparison requests prepared  
**Predecessor:** [163 - Automatic full100 handoff after complete ingestion](163%20-%202026-09-10%20-%20Automatic%20full100%20handoff%20after%20complete%20ingestion.md)

The ninth memory completed all 830 raw requests and admitted all 5,464 source
fragments. Its 1,041,987 raw token proxies exceed the unchanged million-token
threshold. Source admission verifies under the same v11 method as the first
eight memories. Its 2,708 attention leaves and all three summary indexes are
complete, with 50 additional control requests prepared. The existing compiler
now waits for the tenth namespace while its raw ingestion continues.

No answer or judge calls have started in the frozen semantic-seed versus as-of
full100 experiment. Its existing 400 prepared requests and all accuracy/latency
gates remain unchanged. The automatic handoff owns the last two date-comparison
preparations and the later full100 release.

## Complete raw input and source admission

The original strict summary validation recorded 541 invalid batches. The full
source-admission audit found zero unresolved schema failures and three oversized
summaries across three batches. All three compacted successfully in one Qwen
batch, with zero invalid-slot recovery calls. The original completion, quote
diagnostics, and exact raw sources remain preserved. Quote membership does not
certify summary entailment.

All paths below are relative to `eval_results/`.

| Artifact | SHA-256 |
| --- | --- |
| `full100-spine-after-offset060-timeout-20260910-r1/completed-offset-080.json` | `ad82a09b5d4be32aebb54e3904c244e8375194e3059e271865a42e64772aa250` |
| `full100-spine-corpus-20260909-r1/offset-080/atoms-prefix-0830.json` | `a28d2b3a946ba16011f29acabaa72b761e60fee7dd777ab61f92ba727920ea94` |
| `full1m-spine-source-admission-offset080-20260910-r1/audit.json` | `a799c7f9e7838e113c7617211cdccc1949f90220b1cc1cb91676ce56acc1e647` |
| `full1m-spine-budget-repair-offset080-20260910-r1/preflight.json` | `0c78d1679761408cd1495d2c8e77312acc11abcb08ced9c72f50367b40772e41` |
| `full1m-spine-original-compaction-finish-offset080-20260910-r1/complete.json` | `c3c6b5fd50b4aead7546c40f6a7093f6ddbd669cac81e8e84d831d09237fcb24` |
| `full1m-spine-budget-repair-offset080-20260910-r1/repairs.json` | `42697e7ff9100159fe5f276da5e592cacbf4089fec2e5972b7238cbe2af23a1a` |
| `full100-spine-corpus-20260909-r1/offset-080/source-bound-atoms-prefix-0830.json` | `d7d58fba870a1c772c47799a93ec9d9252d28f8f5cfdb274bb022d61183ccd1b` |
| `full100-spine-corpus-20260909-r1/offset-080/conditional-method-v11-prefix-0830.json` | `29c587d7ffcc0a3868a03440dab301d2e63a01b79face50575115cba950c1b1c` |

The verified common admission method remains
`1b6ba1149c3962eb3a16e1fb55b6b2e16d8db811e274f9dc3ff21db3fe4d601a`.
The receipt accounts for all 830 authenticated raw completions, all 5,464
fragments, one compaction attempt, and zero recovery attempts. Its replay
reproduced the original admitted artifact without new calls. The admitted
population retains 2,242 quote-diagnostic annotations; summary entailment is
explicitly unverified. The sealed admission and verification artifacts were
read again in exec `00e2b0`.

## Attention compilation and continuation

Compilation session **48873** completed offset 080 under its existing v3
release. The ninth memory used four exchange-summary merge jobs in one batch,
then loaded local Qwen for user-summary attention. The first leaf wave had
43 merge jobs in six batches; a second wave completed four jobs in one batch.
Leaf construction finished with **2,708 leaves across 488 sources and eight
new summary-generation calls**, with zero replay hits. This call count is
separate from local attention-forward computation. Parent summaries remain
deferred under the unchanged leaf policy. Qwen received summaries only.

The token-accounting successor compiler from Log 162 completed successfully.
Both fragment and exact whole-turn counts are 1,041,987 for this namespace;
there are no nonadditive turns and no mismatch tolerance. The user index covers
all 2,708 leaves, and the passage index contains **5,887 addresses**. These
index stages made no provider calls.

| Completed artifact | SHA-256 |
| --- | --- |
| `full1m-spine-leaves-offset080-20260910-r1/hierarchy.json` | `f71ab7d1fc0b99d6d5789b5f093a1a59d6f98d17600352d879fe9d2c8c7d0c20` |
| `full1m-spine-semantic-offset080-20260910-r1/index.json` | `aacd9cc49808b42d5e5299339b9c3b89a53d39dc3c9ac20c99cba571df1f8d13` |
| `full1m-spine-user-addresses-offset080-20260910-r1/addresses.json` | `a1416a418cea63e87554d72a4df3dbfa8f81fa21c17636303bb8b6f22642b2d5` |
| `full1m-spine-facet-addresses-offset080-20260910-r1/addresses.json` | `22ab8975e8c6a98f4840b812ebd234a1a60a5e3a69de4f312d22e879ed00d51f` |
| `full1m-spine-semantic-seeds-joint-offset080-20260910-r1/preflight.json` | `448782a1140b07fd57d6940d2a48fd9a57680bfffae3d00bb33d966f15a40d51` |
| `full100-spine-memory-completion-20260910-r3/offset-080/complete.json` | `7d4a7c3cf5b9b5503ad8e1d4b3743933c253d015e08047a0cd76996c52a18119` |

The original control campaign's `prepared/offset-080.json` SHA is
`d38d0d7e9ca8cd3bafa009b68c0a6830df1d131f1fccffbbb182d5fbf44e72bb`.
The actual control runner revalidated all nine namespaces and their 450
requests, then stopped at missing offset 090 before publishing a runner plan
or reserving execution (exec `640dc2`). This does not prepare the ninth as-of
namespace; the separate date-comparison preparation remains at 400 requests.

Raw session **42892** owns the tenth namespace's 868 prepared requests. It had
saved 103 responses at 17:58:51 UTC (exec `d6f0e3`). The
same scheduler completed the ninth namespace and advanced without restarting
any raw request. Handoff session **13811** remains live and waiting for both
existing dependencies to finish. It will prepare offsets 080/090 for the as-of
comparison only after all raw ingestion and compilation are complete.

Do not independently launch either remaining date-comparison preparation or
another answer runner while the handoff is active. Observe its existing
session and the original dependency handles. Failure requires diagnosis of
the preserved artifacts; an observation timeout alone does not justify a
restart. Full100 accuracy and latency remain unmeasured for both new arms.
Existing focused tests remain applicable; this continuation changed
documentation and verified new real artifacts without editing frozen runtime
code or rerunning unrelated tests.
