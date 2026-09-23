# Development30 result and omitted user evidence

**Date:** 2026-09-09 (local; execution observations are September 10 UTC)  
**Status:** 30 questions scored across three complete memories; target unmet  
**Predecessor:** [140 - Multi-batch admission and third complete memory](140%20-%202026-09-09%20-%20Multi-batch%20admission%20and%20third%20complete%20memory.md)

The third matched comparison scored **7/10 for both source spine and overflow**.
Across the three completed memories, overflow is **25/30 (83.3%)**, versus
**23/30 (76.7%)** for its fresh controls. Each memory contains over 1M raw token
proxies. The same frozen reader, evidence budget, routing and hydration policies
ran on all 30 questions. Neither accuracy nor the complete joint target passes.
These are examined development questions; confirmation200 remains unopened.

## Third comparison and replay

Root: `eval_results/full1m-source-spine-overflow-joint-offset020-20260910-r1`.

- Preflight: `9297b13623172f2f2543e9896be8b8de71a2555aea561656aed4538a54a9f5ef`.
- Answers: `665cf31eb463338ae5fb454f224401223f2bcea56fa1a4395167c0fc20208183`.
- Report: `ed53452321e78439189f30948534f236b08883312118d694afd1680532dc25df`.
- Fifty Terra calls completed. Sol scored 20 logical memory answers through
  15 distinct physical requests. Replay used 15 authenticated hits, zero calls,
  and reproduced the report byte-for-byte.

All answer timing finished before the next bulk ingest began. Live retrieval
and exact hydration are included; resident startup is recorded separately.
Qwen remains summary-only, and there is no query-time Qwen reranking.

| Third memory arm | Accuracy | Median total | p95 total |
| --- | ---: | ---: | ---: |
| Short API | Not scored | 4.643 s | 7.583 s |
| Source spine | 7/10 | 4.822 s | 12.263 s |
| Source-spine evidence API | Not scored | 5.266 s | 7.305 s |
| Overflow | 7/10 | 5.702 s | 7.883 s |
| Overflow evidence API | Not scored | 5.499 s | 10.352 s |

## Combined development result

The three frozen result populations have disjoint question identities and
identical non-namespace answer policies. The development report aggregates
their original measured request rows rather than averaging shard percentiles.

| Thirty-question arm | Accuracy | Median total | p95 total |
| --- | ---: | ---: | ---: |
| Short API | Not scored | 4.018 s | 7.583 s |
| Source spine | 23/30 | 4.535 s | 9.593 s |
| Source-spine evidence API | Not scored | 4.256 s | 7.305 s |
| Overflow | 25/30 | 4.490 s | 7.505 s |
| Overflow evidence API | Not scored | 4.433 s | 10.243 s |

Overflow fits the provisional median/p95 allowance against its identical-
evidence API control. Its median is about **11.8% above short API chat**, exceeding
the provisional 10% allowance; p95 is slightly below that control. Accuracy
also remains below 95%. Five errors are already present in the first 30 rows,
so this unchanged candidate could reach 95/100 only by answering all remaining
70 questions correctly. The target must not be claimed from a smaller prefix.

Report root: `eval_results/full1m-source-spine-overflow-development30-20260910-r1`.
`development-report.json` SHA:
`a473437e49222ea521024616ead43bec3bf583a9ac214b71348dbfb18151d30c`.
The local `build_report.py` reproduces this analysis from sealed artifacts with
zero provider calls. It explicitly marks the result ineligible for full100.

## Postscore evidence diagnosis

Both third-memory arms miss the same three questions. Inspection found the
following raw user statements in the complete memory but outside both final
packets:

| Ordinal | Required evidence found in raw memory | Recorded omission |
| --- | --- | --- |
| 21, newest streaming service | User started a Disney+ free trial last month. | Neither the witness nor its source appears in the expanded plan. |
| 27, painting inspiration | User has been looking at online tutorials for a painting studio. | The source is selected, but this user turn is absent from the expanded plan. |
| 28, bikes serviced or planned in March | User plans to replace the commuter bike's front tire this month, before April. | Neither the witness nor its source appears in the expanded plan; the serviced road bike is present. |

The painting packet already contains the user's recently started 30-day
challenge, but the answers also mishandle that existing experience. Recovering
the omitted evidence therefore does not by itself prove a complete reader fix.
The streaming and bike omissions still require tracing the original route
frontier versus source-expansion limits; an expanded-plan absence alone does
not identify which earlier stage lost the evidence.

`postscore-witness-audit.json` binds exact turn IDs, raw text hashes, response
hashes, and the saved hydration plans. SHA:
`8067abdb7972c66f05001f6017257874d23ee2600af4d35f282922b91a7b212a`.
Witnesses were selected after judging and are explicitly diagnostic. They must
not select production routes, become benchmark-specific ingest facts, or supply
cached correct answers. No production routing or reader policy changed here.

## Continuation

Offset 30 raw ingest is running in session **76404**, through the previously
prepared 838-request execution preflight:
`583cf02779d46eda972f0e82f81853dd91f9713336b42554ff86eb3c49da83d1`.
Root: `eval_results/full100-spine-corpus-20260909-r1`.
The old strict `invalid_summary` batch statuses include quote diagnostics and
are not the final source-admission verdicts. Preserve every response and audit
the complete namespace before applying the conditional admission policy.

Continue the remaining corpus work while tracing summary ranks and user-turn
selection across the scored development population. A general routing or
expansion change must receive a new frozen comparison, retain summary-only
model inputs and exact raw hydration, and ultimately meet accuracy and latency
together on the full benchmark. Do not start timed answer evaluation while
bulk provider work or GPU compilation is active.
