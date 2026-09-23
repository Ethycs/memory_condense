# Two complete memory overflow comparisons

**Date:** 2026-09-09 (local; execution observations are September 10 UTC)  
**Status:** two ten-question comparisons scored and replayed; full100 target open  
**Predecessor:** [138 - Explicit transport recovery and resumed joint evaluation](138%20-%202026-09-09%20-%20Explicit%20transport%20recovery%20and%20resumed%20joint%20evaluation.md)

The same overflow candidate scored **9/10 in each of two complete memories**,
versus **9/10 and 7/10** for its fresh source-spine controls. This is 18/20
versus 16/20 across the two development batches. It is not a full100 result,
an untouched confirmation result, or a demonstration of 95% accuracy.

Both arms use reader policy v2, a 3,072-token evidence budget, and a 128-span
cap. Overflow preserves additional whole user turns from already routed
sections. Live query embedding, routing, source expansion, and exact hydration
are included in memory response timing. Each memory arm has a byte-identical
evidence API control, alongside short API chat. The 50 requests per memory use
adjacent, counterbalanced pairs. Bulk provider work and GPU compilation were
deferred during each answer timing window. Qwen receives summaries only.

## Measured results

| Memory | Raw token proxies | Source-spine accuracy | Overflow accuracy |
| --- | ---: | ---: | ---: |
| Offset 0 | 1,041,276 | 9/10 | 9/10 |
| Offset 10 | 1,044,341 | 7/10 | 9/10 |

Total response latency, in seconds:

| Memory | Arm | Median | p95 |
| --- | --- | ---: | ---: |
| 0 | Short API | 3.022 | 4.129 |
| 0 | Source spine | 3.945 | 9.468 |
| 0 | Source-spine evidence API | 4.034 | 9.461 |
| 0 | Overflow | 4.149 | 7.505 |
| 0 | Overflow evidence API | 3.811 | 9.240 |
| 10 | Short API | 4.267 | 8.406 |
| 10 | Source spine | 4.668 | 9.593 |
| 10 | Source-spine evidence API | 4.149 | 5.581 |
| 10 | Overflow | 4.318 | 6.668 |
| 10 | Overflow evidence API | 4.349 | 8.609 |

Median live preparation cost was approximately 0.24–0.26 seconds. Visible
time to first token was nearly total response time through this gateway.
The overflow candidate fits the provisional 10% median/p95 allowance against
both API controls in offset 10, but fails against short API in offset 0.
Each ten-question p95 is the maximum observation and remains noisy. The joint
accuracy and latency target is unmet; neither control nor candidate is promoted.

Offset 0 has different misses: overflow fixes the photography recommendation
but loses descriptive identification of a bluegrass band. These must not be
combined into 10/10. Offset 10 has a shared table-tennis-versus-tennis error;
overflow additionally recovers cuisine counting and the apartment duration.
These are recorded judge outcomes, not production routing instructions.

## Sealed evidence and replay

Offset 0 root:
`eval_results/full1m-source-spine-overflow-joint-offset000-20260909-r1`.

- Preflight: `a8be6df727e24fcebb5e5db329276634f1230b56980bec3749fef8796e4d6921`.
- Answers: `31bf60a9ebe236eb7cac0ace360219e833d8a4892fb1023ee7ca1700803ab102`.
- Report: `c3a201620da6549a85ed5053e9b727355e244dbbb68dbd31755f62738fdef19d`.
- Fifty Terra calls; 20 logical Sol judgments from 12 physical requests.
  Judge replay: 12 authenticated hits, zero calls, identical report.

Offset 10 root:
`eval_results/full1m-source-spine-overflow-joint-offset010-20260910-r1`.

- Preflight: `83cdc611242b2983605634acb80f83310714d3962da540bdcba2c0801c6bec04`.
- Answers: `b0ff1d9e24e0502dada2a9118ca7bb8eca3439e658b3bfaf14de4b2178ab7647`.
- Report: `1063671922701b4363b28d2ad22e750b8e95832e6dbb16ad0ddb59a75dd9324f`.
- Fifty Terra calls; 20 logical Sol judgments from 14 physical requests.
  Judge replay: 14 authenticated hits, zero calls, identical report.

Answer sessions 67654 and 63865, and the subsequent judge and replay sessions,
are complete. The original failed transport reservations remain preserved.
Offset 10 now has 5,245 admitted atoms, 2,625 compiled leaves, 464 sources,
semantic vectors, and user-summary addresses. Its version-2 admission method
matches offset 0 after replay; all non-namespace answer policy fields match.

## What the earlier 95 measured

The earlier **95/100 already covered approximately 1M-token memories**. It
used the cumulative retrieval and answer-repair pipeline, retaining previously
authenticated answers where the successor did not replace them. That is a
recorded validation accuracy result; it did not demonstrate that the complete
method delivers fresh answers at API-like end-to-end latency.

The historical fast method scored 73/100 on the same 100 question identities.
It lost 23 previously correct answers and gained one: ten losses involved
multi-session questions and eight involved temporal reasoning. This comparison
does not isolate a single causal mechanism. The current goal requires one
method to meet accuracy and latency together; prior correct benchmark answers
cannot be copied into the new memory. See [132 - Joint 1M accuracy and latency
target](132%20-%202026-09-09%20-%20Joint%201M%20accuracy%20and%20latency%20target.md).

## Diagnostics and remaining work

Later continuation: the third memory is now fully admitted after deterministic
batching and one bounded recovery call. All three memories share a replayed
version-4 admission method. See [140 - Multi-batch admission and third complete
memory](140%20-%202026-09-09%20-%20Multi-batch%20admission%20and%20third%20complete%20memory.md).
The following notes preserve the state at completion of the second comparison.

The offset-0 postscore evidence-scope audit is sealed at
`postscore-evidence-scope-audit.json`, SHA
`61cea4d8e1a7f635a25c07303192bbecf511361259012616f86be9143c3099da`.
Both packets contain the same exact user statement about the bluegrass band.
The photography packet contains Nikon and Canon mentions in actual user turns
from other conversations; the judge's wording that they were invented should
not be repeated as a literal raw-evidence finding. The renderer exposes role
and date but hides the stable conversation identifiers retained in hydration
audits. Conversation coherence and reader interpretation remain hypotheses
for a future general improvement, not established fixes.

The complete ingest prompt-reuse audit found 8,305 distinct exact inputs among
8,305 prepared requests, with no outstanding exact-cache hits. Artifact:
`eval_results/full100-spine-corpus-20260909-r1/exact-ingest-prompt-reuse-audit-20260910-r1.json`,
SHA `0748c25de3d8362afdc0434803213fc6e6f6a7b3012b4d90b0236516c99f19bc`.

Offset 20 raw ingest finished all 817 responses: 584 retained successes and
233 explicitly authorized new requests, including five transport reissues.
Its complete admission audit found nine oversized summaries and zero schema
failures. Audit:
`eval_results/spine-transport-recovery-20260910-r1/corpus/offset-020/source-admission-audit-prefix-0817.json`,
SHA `03e744a8313e9f75e92567e9edef8ff59d3b04ab07a564cf65bc720b5faff199`.
All nine summaries must be compacted in two bounded Qwen batches. The draft
`tools/repair_spine_summary_budget_v2.py` exists and parses, but its behavioral
tests, multi-batch admission, and version-3 full100 verification are unfinished.
The old one-successful-batch certificate must not silently admit two batches.
No offset-20 compaction request has been sent yet. Seven raw namespaces remain
prepared but unstarted. Confirmation200 remains unopened.
