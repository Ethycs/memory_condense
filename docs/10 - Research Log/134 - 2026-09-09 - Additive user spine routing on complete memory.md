# Additive user spine routing on complete memory

**Date:** 2026-09-09  
**Status:** both development comparisons complete; latest source-spine arm scored 8/10; joint target unverified  
**Predecessor:** [133 - Semantic summary admission and complete-memory evaluation path](133%20-%202026-09-09%20-%20Semantic%20summary%20admission%20and%20complete-memory%20evaluation%20path.md)

The first complete-memory joint evaluation scored 7/10 for summary hybrid and
4/10 for hybrid plus local Qwen reranking. Both used the same 1,041,276-token
memory and exact section hydration. This does not supersede the older
validation100 score of 95/100: that score belongs to the cumulative reasoning
and evidence-repair pipeline, whose direct-API-like latency was not established.
The current goal requires accuracy and latency together on one implementation.

## Completed full hierarchy

Session 83890 finished normally with 191 new summary-only Qwen gateway calls
against its allowance of 198, plus 58 authenticated historical call hits.
There are 3,003 exchanges, 516 attention windows, 4,989 sections, 2,744 leaves
and 499 source roots. No hierarchy provider process remains active.

Root: `eval_results/full1m-spine-hierarchy-offset000-20260909-r2`.
Hierarchy SHA: `bc44dbbe26c769fdbae241b9fef80b9b2c855c4f0cab005c18d13f27c3a07528`.

Its raw leaf partition exactly matches the evaluated leaf projection. Twelve
leaf summaries differ because the independently completed merge requests
produced different summaries; do not substitute this hierarchy into the frozen
evaluated index. The partition audit is
`leaf-projection-partition-audit.json`, SHA
`20563018cb529d800427695e1304e472e5a0e9a79dac41811ac138ed008826d0`.
The earlier combined summary/span equality check correctly returned false
(`leaf-projection-comparison.json`, SHA
`f01711676cbbcb826cc1b1e00d1de805d6b50a5f3d9bc42e5f105a8fc51d8f9a`).

## Routing diagnostics

These probes operate on summaries and query text without raw reader calls or
reference inputs. Evidence-source labels are opened only after routing seals.
Source coverage can match the wrong leaf within a conversation and is not
answer accuracy. These are already examined development questions.

The wide MiniLM probe finished. Its 176–203 summary candidates per query
contained every required source conversation for all ten questions, but its
final ranking still missed the sports chronology. Rewriting only the ordering
query to a content view also failed to recover all three events near the top.
Neither method was promoted.

`summary_time_prior.py` introduces a soft mention-date priority for one explicit
relative window, such as “last Friday” or “past month.” It retains global
candidates because a transcript date need not be an event date. A strict
timestamp cutoff would discard required retrospective evidence in this dataset.
The first four-section time prior recovered two of three sports sources.

A follow-up compared original/content query views and combined/user-only
summary addresses, with at most one preferred leaf per source. Only the
user-only content view put all three sports sources in its first four distinct
sources. This motivates an additive development experiment; it does not prove
generalization or justify replacing the existing retrieval channel.

| Artifact root under `eval_results/` | Result artifact SHA |
| --- | --- |
| `full1m-spine-cross-encoder-offset000-20260909-r1` | `22f2414d840294944ead40ca9b721046d49f9cf7b0a62046afd6d68062c35813` |
| `full1m-spine-order-view-offset000-20260909-r1` | `ebc51da6e6ac78746ec864db14b8585c249c9c980181c6e27c6efb150b49d9ca` |
| `full1m-spine-time-prior-offset000-20260909-r1` | `5c6dd6502ce1d28119ae5a22c520033b70be676d1200e45fb3e8ee16f7497345` |
| `full1m-spine-time-frontier-offset000-20260909-r1` | `110bc0d91853cd1e6509a8ab31a8c7da8667ffe1f08a34f152b94c53c965fcee` |

The final probe's post-seal source audit SHA is
`e2d3d30f2d56df8219147fe468d2c8344c1e424b4d7776c4c8d69ddafb31251d`.
No source identifier, answer string or question ordinal appears in the router.

## Additive route and matched execution

`search/spine_union_routing.py` preserves the six original summary-hybrid
routes, adds the top four user-summary routes, and deduplicates exact section
IDs. Where an explicit time window exists, it first adds up to four distinct
sources from a 32-leaf user-summary frontier within that mention window. It uses
a separate content-query vector for explicit ordering questions. Global
baseline routes remain in the plan. Maximum selected leaves: 14.

Hydration retains the existing limits: 4,096 context tokens and 128 raw spans.
Every admitted section is exact and complete; an over-budget section is
rejected with a diagnostic. Retaining baseline routes in the plan does not
guarantee all will fit in the final prompt. These outcomes must be inspected
alongside accuracy. Qwen receives summaries only during compilation; neither
measured query arm in this comparison uses Qwen reranking.

`tools/evaluate_spine_union.py` is a separate versioned harness. It keeps the
original reader policy, complete memory, questions, 256-token answer cap,
counterbalanced adjacent pairs, journal protection and post-answer Sol scoring.
Each live memory request recomputes embedding, routing and hydration within
its clock and must reproduce its frozen identical-evidence API control.
No query vectors or answers are reused between calls. Short API chat is also
measured. Cold model/store loading remains separately excluded from warm latency.

Root: `eval_results/full1m-spine-union-joint-offset000-20260909-r1`.
Preflight SHA: `e56037e2b146d238913ca75a555f708fd17db5b6034816a38e22e8a03852c360`.
All ten baseline prompts match the previous evaluation byte-for-byte. The
additive route changes four prompts; the other six remain identical after
bounded hydration. Prompt comparison SHA:
`0151da03bbf24a05c1136b71ec11cc809b9b0747cc6c3e524af9888915e8f9f5`.

Session 75830 completed all 50 serial Terra streaming calls. Answers SHA:
`d0db4be58ad784f8831be025a36c986a16a9081c90516a2b4bf0bead01654f04`.
The judge's 32-token local proxy cap rejected a response at 33 tokens before
journaling it. Session 69263 exited with 13 reservations, ten saved responses,
and three reservations without saved responses. The failed attempt remains
untouched; do not resume that judge checkpoint directory.

`tools/judge_spine_union_successor.py` separately sealed the complete 20 logical
judge prompts at the standard 1,024-token judge allowance. It rejudged every
prediction uniformly: 13 unique physical calls, without selecting examples by
an earlier verdict. Root: `judge1024-r1` beneath the answer root. Preflight SHA:
`544c5bd8bfb4938ab8b17ad7d9f20d9c112e47b74561fec9fa9a187c24a0900f`.
Session 55232 finished successfully. Replay made zero calls, authenticated all
13 responses, and reproduced the joint report byte-for-byte at SHA
`79de09e51d450e18cc02ad7dd0fbc0262d2f9b95612490fbe3ea240cb674b134`.

| Arm | Accuracy | Preparation median | Total median | Total p95 |
| --- | ---: | ---: | ---: | ---: |
| Short API | Not scored | <0.001 s | 5.537 s | 7.221 s |
| Summary hybrid | 7/10 | 0.153 s | 6.892 s | 9.890 s |
| Its identical-evidence API control | Not scored | <0.001 s | 6.135 s | 9.177 s |
| Additive spine union | 8/10 | 0.139 s | 6.080 s | 11.489 s |
| Its identical-evidence API control | Not scored | <0.001 s | 6.598 s | 12.434 s |

The union recovered the birthday date difference and lost no previously correct
answer in this sample. Music identification and sports chronology still failed.
The median was approximately 9.8% above short API chat, while p95 was about
59.1% above it. Ten observations and substantial API variation cannot establish
a causal speed improvement. Accuracy and tail latency both miss the joint goal.

The post-score hydration audit is
`post-score-hydration-audit.json`, SHA
`d7b1f8c3e783e8d07a8ba25cadc25125dc56fe5071762449f16e218f9f86ad52`.
The union had 14 explicit budget rejections across ten queries. Its preferred
music leaf contained a later jazz-recommendation request; the baseline leaf
with the bluegrass statement was pushed out of the final budget. Sports source
coverage was complete, but two selected leaves contained later training or
cycling requests rather than the event statements. Conversation-level coverage
therefore overstated answer-bearing leaf coverage.

## Source-spine hydration successor

The query-free leaf inventory found 127,500 user tokens and 913,776 attached
context tokens: about 88% of raw tokens are outside the user role. Leaf sizes
have median 377, p95 724 and maximum 2,986 token proxies. There are 864 whole
exchanges above the nominal 512-token leaf target; the constructor correctly
keeps exchanges atomic. Inventory SHA:
`ae3f7c6331951d09cf89841edc4ea3c7d550dc5750e9db209a8050f57857167b`.

`search/source_spine_hydration.py` builds a different exact raw partition from
the existing authenticated atoms. Whole user turns become individual sections;
the non-user spans of each original leaf become its attached-context section.
The complete fragment multiset must match the original memory exactly once.
Compilation reads summary text and coordinates, with no raw text, query or
reference access and no provider calls.

The successor expands the first six selected sources in source rounds. User
turns already selected by the summary router come first within each source,
followed by earlier transcript user turns to recover facts from the same
conversation. A conservative 2,048-token metadata reservation admits at most
24 whole user turns. Selected attached context follows those turns. The final
hydrator enforces a smaller 3,072-token context budget and 128 raw spans;
everything sent to the reader remains exact raw text with its original role
and timestamp. The original reader policy remains unchanged.

`tools/evaluate_source_spine.py` compares the existing union route with this
source-spine hydration method, retaining the five-arm streaming protocol and
using the standard judge token allowance from the outset. The source corpus,
retrieval method and reader model stay fixed; the evidence partition, source
expansion and smaller context budget are the experimental change.

Preparation finished at `eval_results/full1m-source-spine-joint-offset000-20260909-r1`,
preflight SHA `d0d4452eeaa22dd9a0984fdf6a8b5604ecf3d8ec9abf79c2c98506ac2d6440e1`.
The ten union control prompts match the previous run exactly. The new source
packets range from 2,906 to 3,365 total prompt token proxies. Prompt comparison
SHA: `1a8c0e35acef38e2d74766e1f4c89817680577a747c7210e4c984d6df55c83ba`.
Session 15650 completed all 50 streaming answer calls. Answers SHA:
`63972f91157ac39cb76d9740588f0c2ea162d5c6220e84536df6ca4c7fd9203c`.
Session 59867 completed the 13 unique Sol calls for 20 logical judgments.
The reader policy stayed unchanged. Session 31050 replayed all 13 judge records
with zero calls and reproduced the report at SHA
`e64dacb8244945c9446605e9b7f99949da747deac76e5e822a4b8791767695ba`.
All provider processes from this continuation have finished.

| Arm | Accuracy | Preparation median | Total median | Total p95 |
| --- | ---: | ---: | ---: | ---: |
| Short API | Not scored | <0.001 s | 5.543 s | 6.342 s |
| Spine union | 7/10 | 0.295 s | 6.378 s | 10.963 s |
| Its identical-evidence API control | Not scored | <0.001 s | 6.410 s | 8.736 s |
| Source-spine hydration | 8/10 | 0.268 s | 6.147 s | 7.750 s |
| Its identical-evidence API control | Not scored | <0.001 s | 6.512 s | 8.665 s |

The source-spine packet recovered the three-event chronology. Both arms failed
the photography recommendation and descriptive music identification. The fresh
union answer omitted the Sony compatibility/quality constraints despite its
unchanged prompt; its prior answer had named a Sony-compatible flash and passed.
That change from 8/10 to 7/10 is reader variation, not a routing change.
The source-spine arm is about 10.9% above short-chat median and 22.2% above p95.
It still fails the joint target, and no ten-question result can satisfy full100.

Post-score hydration audit SHA:
`759f677e40068ea7c1bf1e03c2f598898fa6ddd205662b7bc2e7c92b539b4b96`.
Every user section admitted by metadata reservation survived final hydration;
all final source-spine budget rejections were attached-context sections. This
does not imply all source user turns passed the earlier metadata reservation.
The evidence-presence check confirms that exact hydrated user statements
include Sony and quality in the recommendation case, and the bluegrass band
and banjo in the identification case. That post-score diagnostic is sealed at
SHA `c88af479885a2aad9e5b9b0dc4b4b9a6040d1f15ca93541d61407180f1a3ba16`;
its terms never enter routing.

The next bounded experiment should address reader behavior on this unchanged
source-spine evidence: preserve relevant compatibility/preferences explicitly
when recommending, and return a supported identifying description when a
proper name is absent. Use general instructions without benchmark names,
answers or source IDs. Evaluate all ten questions with fresh matched answers;
do not combine the best answers from previous runs. At this log's original
seal, no such reader-policy change had been implemented or evaluated. The
subsequent matched comparison and full100 continuation are recorded in
[Research Log 135](135%20-%202026-09-09%20-%20Source%20spine%20reader%20comparison%20and%20full100%20continuation.md).
A passing development result
would still require the other nine complete namespaces and the full100 gate.

`tools/report_joint_source_spine_full100.py` supplies the full100 gate for this
versioned harness. It requires all ten complete memories, uniform retrieval,
compilation and atom-admission policies, authenticated judge replay, at least
95 correct answers, and the joint latency checks against both API controls.
Population-specific file paths and hashes are verified separately and excluded
from method equality. The full100 gate has not run because the other nine
complete summary memories are not yet built.

## Verification

The union evaluation lifecycle, calendar hints, query views and user-summary
address tests passed 30 tests. Three additional router tests passed, covering
additive evidence preservation, original hydration descriptors, source-diverse
calendar priority with outside-window evidence, and independent content-query
vectors. The existing frozen evaluator and its inputs remain unchanged.
The source-spine expansion and its versioned evaluation lifecycle passed eight
tests, including recovery of earlier user evidence after a later topic match,
priority for directly selected user turns, exact text/role provenance, raw
partition validation, live retrieval, and safe handling of incomplete journals.
The successor full100 gate passed three tests covering accuracy/latency on one
method, rejection of partial or mixed populations, and separate short-chat and
identical-evidence timing requirements. These tests are not an accuracy result.
