# Summary routing controls before fast packet promotion

**Status**: Routing diagnostics complete; conventional routing has not been ruled out; matched fast-packet comparison still required
**Date**: 2026-09-08
**Applies to**: `perf/durable-ingest-pipeline`, `.worktrees/ingest-speed`
**Depends on**: [Research Log 122](122%20-%202026-09-08%20-%20Qwen%20summary%20hierarchy%20and%20exact%20section%20hydration.md), [Research Log 123](123%20-%202026-09-08%20-%20Authorized%20temporal%20reference%20chain%20reduced30%20result.md)

## Decision

The user asked whether conventional routing with the fast packet had been ruled
out before advancing the Qwen router. **It had not.** Historical fast-path
results and synthetic summary-selection diagnostics do not replace a matched
conventional-routing versus Qwen-routing answer test using the same packet.

The direct conventional control now shows no reason to prefer Qwen for these
simple summary-routing cases: BGE-M3 cosine search succeeds on all 96 candidate
presentations, while the full Qwen decoder returns 90 contract-valid selections.
All six invalid Qwen responses overrun the requested one-label cap; their first
label would also select the correct summary. The decoder therefore offers no
observed semantic advantage on this diagnostic, even if that output-cap issue
were repaired. Keep both Qwen routing implementations experimental and establish
the conventional fast-packet baseline first.

The original constraint still holds: Qwen receives summaries and questions only.
Attention-guided hierarchy construction remains available independently of the
choice of query-time router. Raw text is hydrated only after section selection.

## Frozen routing diagnostics

`tests/fixtures/summary_routing_diagnostic_v1.json` contains 32 distinct synthetic
queries in eight four-summary groups. Each query is presented in forward and
reverse candidate order: 64 presentations. The original prefix comparison froze
the fixture and arm definitions before model execution. There are separate
development and validation labels, but both partitions have now been examined.

Before measuring the full decoder, a further 16 queries in four new topic groups
were frozen in `summary_routing_confirmation_v1.json`. Their two orderings add 32
presentations. The full decoder and conventional controls therefore share exactly
48 queries, each with four summary candidates and two candidate orders. These
are deliberately small synthetic routing tasks, not LongMemEval, a corpus-scale
retrieval measurement, or a fast-packet answer-quality result.

| Router | Development /32 | Original validation /32 | Fresh confirmation /32 |
| --- | ---: | ---: | ---: |
| Six-layer joint QK/OV prefix | 8 | 8 | not run |
| Six-layer independent QK/OV prefix | 8 | 8 | not run |
| Six-layer attention-transport cosine | 10 | 14 | not run |
| Full gateway Qwen3-8B, strict output contract | 30 | 28 | 32 |
| BGE-M3 cosine over summaries | 32 | 32 | 32 |
| Production BM25 summary index | 8 | 6 | 14 |

The initial prefix diagnostic also included a simple term-overlap control that
scored 12/32 on each of its two partitions. That is a different algorithm from
the production BM25 index and should not be called BM25.

The joint prefix router changed its selected section under reversal for 24 of
32 distinct queries. Independent QK/OV and transport-cosine scoring removed this
position effect but remained weak at semantic selection. The full decoder's
six strict failures all returned too many integer labels, so the implemented
parser correctly rejected them before raw hydration. No prompt or label parser
was tuned against these cases after the results were observed.

The conventional comparator was added after the user raised the missing control.
It uses the repository's pinned BGE-M3 checkpoint and existing BM25 summary
index with no learned weights, threshold fitting or fixture-specific overrides.
This is a post-hoc comparator on the same fixed inputs, not an untouched external
validation set. All conventional inference ran locally without completion calls.

## Full-decoder prototype and verification

The additional prototype in
`src/memory_condense/search/summary_reasoning.py` builds requests from a question
and summary strings. It asks the full `qwen3-8b` gateway model to return only
`selected_labels`, uses zero SDK retries and a 256-token output allowance, and
rejects malformed, duplicate, foreign or over-cap labels. An empty choice never
certifies factual absence.

`reason_over_summary_hierarchy` reduces bounded candidate groups at each tree
level, then descends selected branches. Defaults are four selected sections,
eight summaries per group, 32 hierarchy rounds, 128 model calls maximum and a
2,048-token prompt proxy cap. `max_sections` must be smaller than `group_size`
so the tournament converges. Call or prompt exhaustion fails before hydration.
At the depth limit, selected internal sections retain their whole raw membership.

`MemoryCondenser.search_reasoned_summary_sections` connects this optional router
to the existing exact hydrator. It is a separate API from the prefix-based
`search_attention_summary_sections`; neither was promoted to replace ordinary
retrieval. `SectionRoutePlan` distinguishes decoder reasoning receipts from
measured attention receipts. Decoder route scores are reciprocal ranks, not
calibrated probabilities or measured QK/OV values. Gateway checkpoint weights
are not independently pinned by the model alias; the locally loaded attention
prefix and BGE checkpoint have their existing file verification.

The relevant integration suite passed **282 tests in 41.69 seconds**:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m pytest tests/test_summary_reasoning.py tests/test_attention_summary_sections.py tests/test_summary_section_routing.py tests/test_qwen_episode_signals.py tests/test_qwen_memory_linker_early_exit.py tests/test_condenser.py tests/test_retrieval.py tests/test_conversation_envelope_retrieval.py -q --basetemp .tmp-pytest-summary-reasoning-integration-r1
```

Tests cover summary-only requests, exact selected-leaf hydration, multi-group
traversal, source scopes, depth/call/prompt bounds, no raw reads after invalid
responses, and the local gateway's scalar parameter format. They do not certify
semantic retrieval quality.

## Artifacts and call accounting

| Diagnostic | Root under `eval_results/` | Sealed result SHA-256 |
| --- | --- | --- |
| Prefix controls | `summary-routing-diagnostic-20260908-r1` | `901130812cadb7991991609d55110e833206d339dfa4d189e71959144ddb22d7` |
| Full decoder | `qwen-summary-reasoning-diagnostic-20260908-r1` | `b9a6a8dfa7df2be38963a31201e3e3e5f14d3d178a5146e91f90b65cdf6f6ac8` |
| Conventional controls | `conventional-summary-routing-diagnostic-20260908-r2` | `e33478bba7b4c5092b6624eb7336ec70d2178351ebe848309ab8d9897aaca07a` |

Prefix preflight SHA:
`a971bd3e7857241372b30c3b38f4e63b2cf61ed0a1dae4c418fa1d801c691a25`.
Full-decoder preflight SHA:
`3bd172b73b3599fb6a49b9c689bf7500f76d54bf06478bbed0d39c6467e83c02`.

The full-decoder diagnostic made exactly **96 authorized Qwen completion calls**
through `https://central-dev.zt:4000/v1`, with concurrency four and zero retries.
The batch took 38.407 seconds. Its replay used all 96 authenticated checkpoints,
made zero new calls and reproduced the same result SHA. Before that run there
were two isolated compatibility requests: one rejected nested
`chat_template_kwargs` at the gateway, and one successful request using scalar
`enable_thinking: false`. Thus compatibility calls must not be hidden inside the
96-call diagnostic accounting. No raw transcript content was submitted to Qwen.

The conventional r1 process was stopped during unwanted optional Hugging Face
metadata probes and produced no result. R2 forces offline loading, verifies the
pinned checkpoint and completes using only the installed cache. Its timing is
batch embedding plus model loading/closing, not a measured fast-packet serving
latency. No conventional control made completion-provider calls.

Reproduce or replay with:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.assay_summary_routing_diagnostic --qwen-model-dir F:\Keytone\Documents\GitHub\memory_condense\.cache\models\Qwen3-8B --output-dir eval_results/NEW_PREFIX_ROOT
.\.pixi\envs\dev\python.exe -X utf8 -m tools.assay_qwen_summary_reasoning run --output-dir eval_results/qwen-summary-reasoning-diagnostic-20260908-r1 --authorized-provider-calls 0
.\.pixi\envs\dev\python.exe -X utf8 -m tools.assay_conventional_summary_routing --output-dir eval_results/NEW_CONVENTIONAL_ROOT
```

## Required next comparison

Use conventional routing as the primary control before further Qwen promotion.
For the actual fast-packet answer comparison, hold the questions, corpus/source
scope, raw hydration rules, packet renderer, context and output budgets,
responder and independent judge fixed. Change only routing. Report evidence
retention, judged answers, routing/model calls and latency separately.

The user was asked whether “conventional routing” means the existing BGE-M3/BM25
hot retrieval feeding the fast packet, conventional search over summaries before
raw hydration, or both. The matched fast-packet control has not yet been run;
the diagnostic above must not be represented as that missing experiment. Prior
69–73/100 fast-path results use different packet/policy variants, and the 18/19
r9 supporting-evidence audit further cautions against assuming routing is the
dominant remaining problem.
