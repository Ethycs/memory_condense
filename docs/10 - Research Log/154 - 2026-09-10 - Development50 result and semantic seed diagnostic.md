# Development50 result and semantic seed diagnostic

**Date:** 2026-09-10  
**Status:** comparison complete; neither successor promoted; full100 goal active

The frozen comparison completed 400 fresh Terra answer requests over five
complete memories, each approximately 1.04M token proxies. Sol judged the same
streamed memory predictions: 150 logical judgments, 89 physical requests
(16, 18, 17, 19, 19 by namespace), and zero calls when all judgments replayed.
Session 36114 ended with exit code 0. No frozen implementation changed.

## Measured result

All values below come from the completed campaign, including live query
embedding, routing and exact raw hydration in the memory clocks. Each memory
arm has its own identical-evidence API control and a policy-matched short API
control. The two v3 arms share their short control.

| Arm | Correct | Memory median / p95 | Identical-evidence API median / p95 | Short API median / p95 |
| --- | ---: | ---: | ---: | ---: |
| Source diversity and scoped terms, v2 reader | 41/50 | 6.524 / 9.714 s | 6.492 / 10.874 s | 5.409 / 7.690 s |
| Same evidence, v3 reader | 39/50 | 6.936 / 9.782 s | 6.360 / 9.964 s | 5.399 / 6.218 s |
| Whole-summary source supplement, v3 reader | 39/50 | 6.982 / 13.711 s | 6.343 / 11.821 s | 5.399 / 6.218 s |

The existing path's total-latency ratios are 1.005 median and 0.893 p95 against
identical evidence, but 1.206 and 1.263 against short chat. The v2 and v3 paths
without extra whole-summary coverage meet the provisional matched-evidence
allowance; all three fail the short-chat allowance. All four required ratio
checks include both visible TTFT and total time. Every answer arrived in one
visible event and finished with `stop`, so observed visible TTFT is essentially
complete-response time. No token-streaming advantage is established.

The existing path has a median 0.310 s local preparation time and 3,401.5
prompt token proxies. Its median answer is two token proxies; the short API
control has a median 435 prompt tokens. These are workload observations, not
a causal decomposition of the gateway's internal latency.

| Namespace offset | Existing v2 | Same evidence v3 | Expanded evidence v3 |
| --- | ---: | ---: | ---: |
| 000 | 9/10 | 8/10 | 9/10 |
| 010 | 8/10 | 8/10 | 8/10 |
| 020 | 10/10 | 10/10 | 9/10 |
| 030 | 8/10 | 6/10 | 6/10 |
| 040 | 6/10 | 7/10 | 7/10 |

The v3 reader gains ordinals 13 and 42 but loses 6, 16, 31 and 32 against v2.
Whole-summary coverage gains 6 and 32 but loses 27 and 38 against the same v3
reader. It still fails the comedy recommendation despite hydrating the user
preference. Preserving all prior user evidence did not preserve all needed
assistant evidence or ensure correct synthesis.

The unchanged v2 control on the first forty questions again scores 35/40.
Its fresh answer gains 27 and loses 14 relative to the earlier 35/40 result.
Historical answers are used only for this post-score comparison. They never
supply current predictions, routing decisions or latency measurements.

Campaign root:
`eval_results/full1m-spine-combined-reader-development50-20260910-r1`.

- `development-report.json`: `5fc4b402ef3f253911ff9f4e061e15913fd60c12aadc95207f7016931b9534c1`
- `complete.json`: `955ffc3523798acf4197eb6bf655333cec20956b35a824beb4c64a61a974f446`
- `comparison-diagnostic.json`: `19c49c54db447f6d3cf85fe3716268910f7259ef2b60b68ffd37730ef2bd82ef`

`assess_results.py` completed successfully. It authenticates all 400 saved
responses, all 150 prediction/judgment bindings, and reproduces every reported
accuracy and latency distribution. Neither successor is promoted. This is
development50, not full100 or confirmation, and the joint target is unmet.

## Semantic seed priority

The current hybrid router inserts two lexical matches before dense matches
within its first six routes. Source expansion then prioritizes the first six
distinct conversations. In the camera question, a compatible Sony flash
summary is already dense rank one, while the hybrid seeds put broader lexical
matches ahead of it. Calendar routing is inactive for that question.

The new experimental path changes only these six initial whole-summary seeds
to dense summary matches. User and calendar supplements, source diversity,
scoped terms, the existing v2 reader, and the 3,072-token/128-span hydration
budgets stay fixed. Source identifiers remain provenance, not question-ID
selectors. No benchmark references or predictions enter this route.

Implementation:

- `src/memory_condense/search/semantic_spine_seeds.py`
- `tools/spine_semantic_seed_memory.py`
- `tools/audit_spine_semantic_seeds.py`
- `tests/test_semantic_spine_seeds.py`

The new path is separate from the frozen comparison. Twelve focused checks
pass in 1.52 seconds, covering lexical competition, exact raw hydration,
unchanged controls, calendar behavior, and query/date/encoder bindings. One
initial test incorrectly assumed a two-word ordering view activated the
existing three-word-minimum rule; the fixture was corrected, without changing
that rule or any frozen implementation.

The saved-rank diagnostic was selected after the current first40 outcomes. It
uses no query model or answer calls and reproduces all forty current v2 control
packets before evaluating the seed change. It changes 29 packets, adding 65
raw spans and removing 61, including 51 user spans. Its aggregate is:

`eval_results/full1m-spine-dense-seed-diagnostic40-20260910-r1/audit.json`

SHA `361cbe9da9d67f13b80a1414b14872c3c99ab8f291796f20982127890f554ce8`.

The separate live diagnostic recomputes query embeddings for all fifty
questions, reproduces all fifty control packets, and exactly matches every
saved-rank candidate for the first forty. Its aggregate is:

`eval_results/full1m-spine-semantic-seed-live50-20260910-r1/audit.json`

SHA `bb6b396f01274ce82a5c18304867f0bc6ac737c897e6cd3ebebc07970d117fac`.

Across fifty questions, 37 packets change, with 73 added spans and 70 removed,
including 57 user spans. The Spanish-Catalan singer-songwriter passage absent
from the old packet is now hydrated. The gardening witness is still missing.
The Denver encounter was already present in the old packet, showing a reader
binding failure. These are post-score evidence observations; none establishes
candidate answer accuracy or live latency. No new answer/judge calls were made.

## Remaining corpus and sixth-memory recovery

The remaining raw-ingest scheduler passed both gateway readiness probes and
released its 3,328 prepared requests only after timed work ended. Session
84398 is now executing raw ingestion, not waiting. Offset 060 is active; do
not launch any of offsets 060/070/080/090 independently.

Root: `eval_results/full100-spine-remaining-corpus-20260910-r1`.
Readiness report SHA: `e4d1fb7b768fb091bc36b57d500ca6f787df5a9f2701e4de46bfcd78b09b6b87`.
Release SHA: `aeecf8303dbc13cfab771c63bc451d3f0ebb18c576bb3ec88875a13b6900fcf1`.

The sixth-memory audit replayed all 834 completed raw responses. It found five
oversized summaries across four batches and zero unresolved schema failures.
Audit SHA: `34386b39017f7fc16fa91f81d7b1458bfb6edd8017d1828c6f64321546314ebe` at
`eval_results/full1m-spine-source-admission-offset050-20260910-r1/audit.json`.

The first compaction connection failed locally with `WinError 10013`; its
original request remains reserved without a response. One separately recorded
reissue through the authorized network path completed. Its five summaries
were over budget, so the existing bounded slot-recovery procedure recovered
them in six summary-only Qwen calls. Automatic retries remain zero.

The task-specific stage and transport verification live at
`eval_results/full1m-spine-compaction-transport-offset050-20260910-r1`:

- Original preflight: `2a88918726d2e5cd57d59fea8df86a414eba9c6d841765a276124c26f5bf05cc`
- `stage.json`: `aa9cdd7971212552560a7a0c91dcc210b2290c28aa66a0274df68d776f87f075`
- `execution/complete.json`: `7768d745e47f05dee40ad85b8407ec569997dac02cff9c90a1e3a3c6ef923346`
- `transport-verification.json`: `73fc05561a96294077d8a493815a15877f61ecb16c5824ff89109ba4876e7cd9`

The transport verifier reauthenticates the identical old/new request protocol,
preserved original reservation, completed reissue, all six slot recoveries,
and the admitted atoms' repair binding with zero calls. It counts eight
compaction attempts conservatively: one blocked original, one completed batch,
and six slot recoveries. **This supplementary accounting is not yet integrated
into the full100 admission-method verifier.** The old v8 receipt alone would
omit the blocked original; do not claim a shared six-memory method yet.

Recovery root: `eval_results/full1m-spine-budget-recovery-offset050-20260910-r1`.
Its preflight is `e38a2da8043344cb5f0b3cda8a52a94b6bedcd145ac1a22c24b479f0278ada4f`;
`repairs.json` is `834c05534d4f8514a9c6d133b969afaa28c49965be5437f4dedbba2f1cfd608f`.

Source admission finished with all 5,410 fragments, exactly five budget
compactions, and 834 authenticated raw-response hits. The admitted atoms are:

`eval_results/full100-spine-corpus-20260909-r1/offset-050/source-bound-atoms-prefix-0834.json`

SHA `c23871e1239545f14537ff4ecc719a55710a55176d50f9248b26728563a83afd`.

Session **97674** completed with exit code 0: 2,712 attention-guided leaves over
480 sources, using 12 summary-only Qwen calls under the existing leaf policy.
All raw fragments remain in the exact leaf partition; parent summaries are
deferred as in the five earlier memories. Root:
`eval_results/full1m-spine-leaves-offset050-20260910-r1`.
Preflight: `f141833613df4d0c1e491d66708abd55bcc66dc3b6eada7c18d9121f0996bd84`.
Hierarchy: `733c3aea5781275c66c11dc8381eb992c90d3debc23227c236cdc5d856605bd6`.

Session **47592** is compiling the semantic, user-summary, and facet indexes
sequentially, stopping if any stage fails. Its output roots are respectively
`full1m-spine-semantic-offset050-20260910-r1`,
`full1m-spine-user-addresses-offset050-20260910-r1`, and
`full1m-spine-facet-addresses-offset050-20260910-r1` under `eval_results`.
No completion or index hashes are claimed yet.

## Next work

1. Poll sessions 47592 and 84398; neither has been stopped or restarted.
2. Integrate the new compaction-only transport accounting into a versioned
   common admission verifier and replay existing memories without changing atoms.
3. Finish the sixth memory's indexes and continue all remaining full namespaces.
4. Evaluate the semantic-seed candidate with fresh answers and matched latency
   after model workloads are isolated. The full100 gate still requires at least
   95 correct on the same implementation that passes the latency allowance.
5. Confirmation remains unopened by this continuation. Preserve the historical
   exposure qualification from Log 102 when eventually reporting confirmation.
