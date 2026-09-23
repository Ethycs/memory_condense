# User spine attention hierarchy and summary compilation

**Status**: IMPLEMENTED, EXPERIMENTAL — historical smoke result; raw compilation subsequently approved and completed in Research Log 130
**Date**: 2026-09-09
**Applies to**: `perf/durable-ingest-pipeline`, `.worktrees/ingest-speed`
**Depends on**: [Research Log 122](122%20-%202026-09-08%20-%20Qwen%20summary%20hierarchy%20and%20exact%20section%20hydration.md), [Research Log 128](128%20-%202026-09-08%20-%20Compact%20conventional%20fast%20packet%20admission.md)

**Successor**: The user subsequently said "Approve sending raw" and requested a
real-data evaluation. [Research Log 130](130%20-%202026-09-09%20-%20Real%20data%20user%20spine%20routing%20evaluation.md)
records the completed 39-turn compilation, revised 128-token channel policy,
real routing comparisons, and the separate automatic-review rejection of raw
answer generation. Approval-pending statements below describe the earlier stage.

## Decision and implementation

The user resumed attention-guided hierarchical chunking, with summarization
driven by the user spine. The six conventional packet variants remain rejected
comparisons over a fixed candidate pool; they do not prove that conventional
routing has reached its limit.

The new builder changes the unit of hierarchy construction from independent
token fragments to complete user-led exchanges. Each user turn owns all following
machine turns until the next user turn, within its exact source. A pre-user
prelude remains explicitly unowned. Input transcript order is preserved;
timestamps do not reorder turns or establish event time.

The pipeline is:

1. A separate **non-Qwen raw summarizer** compiles bounded atomic summaries with
   exact source/turn/character/hash bindings. The first local LFM attempt failed
   validation. The prepared successor uses Terra with at most 2,048 raw tokens
   per atom, 128 summary tokens, and exact supporting quotes kept only in audit.
2. **Qwen summarizes the user spine** from user-summary fragments only. A second
   summary channel describes attached responses in the context of that spine.
   Roles and transcript dates accompany fragments; descriptor source IDs and
   raw locators are not serialized into model prompts. Every merge consumes at
   most two summaries. This is a structural boundary, not a guarantee that a
   generated summary never copies an identifier mentioned inside source text.
3. **Qwen attention guides contiguous exchange boundaries.** The local pinned
   Qwen3-8B six-layer prefix supplies OV-transport change between user-spine
   summaries. Attached-response summaries never enter this signal. Overlapping
   windows retain the boundary measurement at their seam. Within a balanced
   range, the strongest change chooses the next cut.
4. Parent summaries separately combine child user spines and attached context.
   The full Qwen gateway model sees these summary requests, never raw turns.
5. The resulting `SectionSummaryIndex` supports the existing BM25 and full-Qwen
   routing APIs. Both use the **same hierarchy and exact hydration**. Only the
   selected raw sections enter the answer context; routing summaries do not.

The new files are
[`spine_summary.py`](../../src/memory_condense/search/spine_summary.py) and
[`user_spine_hierarchy.py`](../../src/memory_condense/search/episodes/user_spine_hierarchy.py).
The existing opt-in `search_summary_sections` and
`search_reasoned_summary_sections` APIs accept `hierarchy.summary_index()`;
no default retrieval behavior changes.

## Bounds and authority

User and attached summary channels are each bounded to 64 tokens. Summary
requests are bounded to 2,048 prompt tokens and reject excessive output instead
of silently truncating it. Model-generated text can still be incomplete or
wrong: channel separation and prompting are not semantic factuality proofs.

The pilot uses a 512-token raw leaf target and at most two exchanges per leaf.
An individual exchange exceeding that target remains a whole leaf and is listed
in `oversized_exchange_ids`. The target is therefore not an unconditional hard
cap for indivisible exchanges. Hydration still enforces its actual 4,096-token
context and 128-span caps atomically, with an explicit fallback diagnostic.
This preserves user ownership instead of dropping the lead to fit a response.

Source boundaries are hard. This version does not infer common user identity
from source-ID prefixes, construct cross-source event identity, prove complete
set coverage, or fix final-answer temporal reasoning. Summary routing retains
`frontier_closed=false`. A narrow beam may still omit useful branches, and an
oversized selection may require the existing raw fallback.

## Real-source pilot

Tool: [`assay_user_spine_hierarchy.py`](../../tools/assay_user_spine_hierarchy.py).
Prepared successor root:
`eval_results/user-spine-hierarchy-real-source-20260909-r2`.
Preflight SHA:
`9c1de80899915bcf1136a308facdd5aea84309d1bc8da0628c1ff1d4e822a58b`.
Raw request manifest SHA:
`8da2fea4c9f5dd256a1e77a302f3b9480bc504db0fd2ca5b9a7bf75d7d1bb6ee`.

Before summary generation, the pilot freezes the first three distinct sources
in the existing r9 packet's global citation order for ordinal 86. It loads the
**complete source transcripts**: 39 turns, 18 user turns and 8,855 raw-token
proxies. Source selection is inherited from the already analyzed development
packet. This is not full-corpus discovery or untouched validation. Benchmark
answers and judge references are not loaded.

The immutable database and whole-source turn population are authenticated.
Raw summaries are checkpointed separately from Qwen summary requests. Each
adaptive Qwen request is sealed before its one possible completion, uses zero
SDK retries and is resumable from an authenticated completion checkpoint. The
total ceiling is 160 Qwen requests, with at most 32 routing calls. Replays must
reproduce the hierarchy and result artifacts without new gateway calls.

The local `r1` attempt used LFM2-2.6B-Transcript. It produced 43 saved raw-summary
checkpoints and stopped on the next completion's output-budget failure. Inspection
also found a semantic failure: a user request for Eastern Sierra campsite advice
became recommendations naming new campsites. A system timestamp header became a
meeting. These summaries were rejected before any Qwen call. Rejection artifact:
`d0be6597b89563851d851bb27fbcb9c6ac36a88d79386b9839b876b34fc56d7e`.

The prepared Terra successor requires exactly **39 non-Qwen completions**, with
zero retries and concurrency eight. Automatic approval review rejected launching
that raw-payload action: it did not consider the 39 private raw turns clearly
covered by the earlier gateway authorization. The rejection occurred before
process launch; the later provider-free manifest preparation made zero calls.
Approval was requested specifically for this payload. Block record:
`2e18f5a2ca4b6f2c231adc6898958d908dcab6a3c9ce01889c1f61a08ef25544`.
This is an approval boundary, not a claim that the user had forbidden local
gateways. Do not retry or route this raw payload indirectly without resolving it.

A local LFM2-350M-Extract alternative was tested on three turns with two prompt
forms, including a JSON-schema form. All six outputs failed schema/support
validation, with invented details and truncation among the failures. Both probes
made zero provider calls. Result SHAs:
`366844a3b510b15f3c22238f4cab4624da6ca77bfb1125c3ff46e3430f9a0ec2` and
`1842969d949a5ca9131aee464c9afd63a9990ddc164d7298bb3ef37dc3991d85`.

No real-source hierarchy, answer score or compiler-throughput result is claimed.
The original `r1` preflight remains preserved; its failed compiler implementation
was superseded by the current prepared Terra tool, so its old command is not a
current-code replay path.

## Independent live Qwen smoke

[`assay_user_spine_hierarchy_smoke.py`](../../tools/assay_user_spine_hierarchy_smoke.py)
uses eight authored summary fragments, four user-led exchanges and raw canaries.
It compiles the user and attached channels with full gateway Qwen, constructs
seven nodes using two local Qwen attention windows, then compares BM25 with
full-Qwen summary routing and hydrates exact raw exchanges. Every gateway request
is checked before sending for the raw canaries and private fixture source ID.

The run completed **14 Qwen summary calls and two routing calls**, with zero
retries. Its 56.203-second elapsed time includes hierarchy compilation and model
loading, not just query routing. A replay used **16 authenticated checkpoint hits,
zero new gateway calls**, and reproduced result SHA
`4798f0c4647e77d22df62789c43a77ed1ce5e3bc0f3177790cb5b90a959152d4`.
Root: `eval_results/user-spine-qwen-canary-smoke-20260909-r1`.

Exact hydration and the summary-only boundary passed. Semantic routing did not:
BM25 selected the telescope-reservation exchange; Qwen selected the neighboring
observatory-hours/stargazing exchange. The attention tree placed the telescope
exchange under a mixed orchard/telescope parent, while the neighboring exchange
was a more focused leaf. Qwen's width-one decision pruned the mixed parent even
though its summary explicitly mentioned telescope reservations. This localizes a
failure of narrow hierarchical routing, without proving that attention caused it.

This is a synthetic mechanism smoke with authored atomic summaries, not a raw
compiler result or benchmark accuracy comparison. It does not earn promotion.
The next routing repair should examine branch coverage and beam width separately
from the final raw-section count. The shared-tree BM25 control isolates routing;
it does not isolate attention-based chunking from other tree construction methods.

## Verification

The final integration suite passes **93 tests in 5.76 seconds**. It checks complete user-led
ownership, separate sources, bounded summary merges, unchanged boundaries when
assistant topics change, attention-dependent cuts, Unicode coverage, raw-input
rejection at the Qwen boundary, output overflow, exact hydration, stale-response
rejection of the entire exchange, oversized-exchange fallback and exact support
validation without passing raw support quotes into Qwen requests.

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.assay_user_spine_hierarchy_smoke --output-root eval_results/user-spine-qwen-canary-smoke-20260909-r1
# Only after resolving the explicit raw-payload approval block:
.\.pixi\envs\dev\python.exe -X utf8 -m tools.assay_user_spine_hierarchy atoms --output-root eval_results/user-spine-hierarchy-real-source-20260909-r2 --enable-provider
.\.pixi\envs\dev\python.exe -X utf8 -m tools.assay_user_spine_hierarchy run --output-root eval_results/user-spine-hierarchy-real-source-20260909-r2 --enable-provider
.\.pixi\envs\dev\python.exe -X utf8 -m tools.assay_user_spine_hierarchy run --output-root eval_results/user-spine-hierarchy-real-source-20260909-r2
.\.pixi\envs\dev\python.exe -X utf8 -m pytest tests/test_user_spine_hierarchy.py tests/test_attention_summary_sections.py tests/test_summary_reasoning.py tests/test_summary_section_routing.py tests/test_user_led_episodes.py tests/test_lfm_completion.py tests/test_conversation_envelope_retrieval.py -q --basetemp .tmp-pytest-spine-hierarchy-NEW
```
