# Preserve selected user evidence before attached context

**Date:** 2026-09-09  
**Status:** exact-evidence diagnostic complete; matched answer evaluation prepared  
**Predecessor:** [135 - Source spine reader comparison and full100 continuation](135%20-%202026-09-09%20-%20Source%20spine%20reader%20comparison%20and%20full100%20continuation.md)

The source-spine expansion can drop user turns that its summary router already
selected. On the complete 1,041,276-token development memory, five of ten
questions lost routed user turns, including baseline selections on two. A
new optional expansion restores all those routed users while preserving every
previously hydrated user excerpt and the 3,072-token context limit. It changes
which attached assistant sections fit. Answer accuracy and end-to-end latency
have not yet been measured for this option; the latest sealed reader result
remains 9/10 versus 9/10.

## Evidence loss after routing

`tools/audit_source_spine_route_retention.py` reconstructs the frozen reader
comparison's live query vectors, summary route, source expansion, and exact
hydration. Every baseline prompt must reproduce its frozen bytes. It reads
neither predictions nor reference answers and makes no provider calls.

Audit: `eval_results/full1m-spine-reader-joint-offset000-20260909-r1/routed-user-retention-audit.json`,
SHA `e3f41ae25be2569c80929ecd4ab3fae7939f554045a0e000750a899a953667fd`.

Five questions lose routed user turns at the six-source expansion cap. Two
questions also lose some of the original hybrid baseline's user turns before
the reader. All user sections that reach the final hydration plan survive its
final budget. Thus the source and metadata selection stage, rather than final
hydration, accounts for these omissions. The relevance of omitted turns is
unmeasured; this audit does not establish that they caused an incorrect answer.

The historical 95% numeric proof path checks a complete store census under a
restricted action/entity/state grammar before trusting a computed answer.
Its full-store scan and previously compiled typed evidence cannot be equated
with the current bounded summary shortlist. The transferable requirement is
to preserve needed operands and distinguish complete candidate sets from
truncated retrieval. No old benchmark answer or question-specific numeric
result has been copied into this memory. See
`tools/matched_eval/numeric_policy_frontier_bridge.py` and the
[campaign playbook](../02%20-%20Implementation/13%20-%2095%20Percent%20Full100%20Campaign%20Playbook.md).

## Optional expansion

`search/source_spine_overflow.py` wraps the existing source-spine expansion.
It keeps its admitted user sections in their original order, then offers any
omitted user turns from the existing summary route, then offers the previous
attached context. The added turns must already have a selected summary leaf;
the wrapper reads only authenticated descriptors and coordinates. The exact
hydrator still enforces 3,072 context tokens and 128 raw spans. Oversized
additions can be rejected without displacing protected user turns. Summary
routing continues to declare `frontier_closed=False`.

`tools/audit_source_spine_overflow.py` compares both hydrations on all ten
questions. The corrected membership audit is
`routed-user-overflow-audit-r2.json` under the reader comparison root, SHA
`dab13d6a4384671f0d15410951e882e1f10bcdd0e28bc3e94bdff8c6a0691630`.
It adds 13 user-turn excerpts across five prompts, leaves no routed user turn
omitted, and preserves all prior user text, roles, dates, and ordering. One
previous attached section is removed on each of three questions; replacement
attachments can leave section counts unchanged. The earlier r1 audit's
`attached_sections_displaced` field measured only a net count difference;
use r2's explicit removed-section identities for membership claims.

All contexts remain within budget. The wrapper's complete metadata expansion
measured 8.299 ms median and 12.144 ms p95 in this local diagnostic. Those
figures exclude embedding, hydration, and the answer API and are not an
incremental latency comparison or a joint-target result.

## Next matched comparison

`tools/evaluate_source_spine_overflow.py` uses the prepared reader-v2 instruction
for both memory arms. The baseline is the existing source-spine hydration; the
candidate adds routed user turns as above. Each gets an adjacent, byte-identical
evidence API control, and both share a short API control with the same system
policy, dated question, model, and output cap. Pair order is counterbalanced.
All embedding, routing, expansion, and hydration remain inside the live memory
timer. No query vectors or predictions are cached.

Root: `eval_results/full1m-source-spine-overflow-joint-offset000-20260909-r1`.
The 50-call preflight is sealed at SHA
`a8be6df727e24fcebb5e5db329276634f1230b56980bec3749fef8796e4d6921`.
`prompt-comparison.json`, SHA
`4275f9017d9f3195cce2793945e0db5cb7f2044935dd7641d8c8943944ef5a47`,
confirms all ten baseline prompts exactly match the prior prepared reader-v2
candidate and that overflow changes evidence on five questions. Neither
preparation has sent answer calls. This comparison takes priority over the
unexecuted six-arm reader-v2-only comparison: it holds the reader fixed and
directly measures the newly observed evidence-loss tradeoff.

After bulk provider work is quiescent:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.evaluate_source_spine_overflow run --output-root eval_results/full1m-source-spine-overflow-joint-offset000-20260909-r1 --max-calls 50 --enable-provider
```

Then execute `judge` with `--enable-provider` and replay `judge` without it.
`tools/report_joint_source_spine_overflow_full100.py` provides the corresponding
100-question gate. It requires all ten complete memories, one frozen method,
authenticated conditional admission replay, at least 95 accepted answers, and
the same accuracy/latency requirements against both API controls. The ten-question
comparison alone cannot satisfy it.

## Ingest handoff

**Later status:** both outstanding jobs exited with inference timeouts, and
both subsequent small readiness probes failed. No local provider job remains
live. [Research Log 137](137%20-%202026-09-09%20-%20Gateway%20timeouts%20and%20complete-memory%20recovery%20handoff.md)
supersedes the in-flight observations below and records exact recovery state.

Raw-ingest session **85650** finished offset 10 with all **801** responses and
exit code zero. This memory covers **1,044,341 raw tokens**, 464 sources, 5,241
turns, and 5,245 fragments. Its older strict quote-check artifact remains an
immutable diagnostic; source-bound admission is a separate complete replay.
Session **91664** continues offset 20's 817 requests, with a confirmed contiguous
prefix of 584 saved responses. It permits four concurrent requests and zero retries.
Do not restart an unacknowledged request. The other seven namespaces remain
prepared without provider execution. Latency measurements are still deferred
until bulk provider work is quiescent.

The first 600 offset-10 responses contain five over-budget summaries, while the
first 360 offset-20 responses contain four. Neither audit found schema or
attribution failures. Audit SHAs are
`1d6a0c51ccded1371b692f132682e4609b127df25d54581bfdd42cb854171521`
and `c1d4ec5e1825bda73365e6ab508f1fbf539bd7b54d41657caef0dace2c3060b7`
at `offset-010/source-admission-audit-prefix-0600.json` and
`offset-020/source-admission-audit-prefix-0360.json` in the corpus root.
The completed offset-10 audit, `offset-010/source-admission-audit-prefix-0801.json`,
is sealed at SHA
`5850c40483dc683b926cb91d37cfdf762080835d3443a26b7e4af221dd579bd3`.
It found five over-budget summaries (129, 148, 131, 132, and 188 tokens), no
schema or attribution failures, and made zero provider calls. Its one-call
summary-only Qwen compaction preflight is sealed at SHA
`d130c629a127c9e1fcfdc15f997fb510b721751e7ccf2e84326d17a07a9999c0`
under `eval_results/full1m-spine-budget-repair-offset010-20260909-r1`.
That call is pending in session **43793**; the process is confirmed live.
No compaction call has run for offset 20. Audit its complete response population
before preparing repairs. The existing tool handles at most eight summaries
in one batch; if the complete population exceeds that, extend the versioned
repair/verification path rather than dropping a failure.

The later offset-20 prefix-584 audit is sealed at SHA
`8fc427b378a91306f359be9aa4afeba83458193572c79d4ef473ca09b3afbf1c`
in `offset-020/source-admission-audit-prefix-0584.json`. It found five
over-budget summaries and no schema or attribution failures, with zero calls.
Both outstanding process handles remain live despite a gap in new responses.
A read-only gateway health check returned HTTP 200 using the repository's
Windows trust-store TLS configuration; no inference request was retried.

After admission, run conditional-method verification, compile the complete
attention-partitioned leaf index, semantic vectors, and user-summary addresses.
Use the same frozen overflow harness for subsequent namespace comparisons.
Unneeded full-parent summaries can remain deferred as in the existing compiler.

## Verification

The expansion, original hydration, both versioned evaluation lifecycles, and
both full100 gates pass 23 focused tests. They cover preservation of exact
user evidence, recovery beyond the source cap, safe rejection of oversized
additions, foreign-route rejection, live retrieval, changed-prompt rejection,
interrupted-request and gold guards, counterbalanced controls, and joint gates
that cannot pass from partial or mixed populations. The two real-data evidence
audits and the 50-call preparation made zero provider calls. No new accuracy
result or router promotion is claimed.
