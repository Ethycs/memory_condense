# Conventional fast packet source grouping comparison

**Status**: REJECTED — grouped packet scored 8/30 versus the fresh control's 12/30
**Date**: 2026-09-08
**Applies to**: `perf/durable-ingest-pipeline`, `.worktrees/ingest-speed`; changes uncommitted
**Depends on**: [Research Log 121](121%20-%202026-09-08%20-%20r9%20reduced30%20execution%20handoff.md), [Research Log 124](124%20-%202026-09-08%20-%20Summary%20routing%20controls%20before%20fast%20packet%20promotion.md)

## Scope and fixed comparison

The user prioritized conventional routing plus the fast packet before further
hierarchical-attention work. This experiment holds the existing conventional
r9 retrieval output fixed and changes only packet layout. It is not a comparison
between BGE and Qwen, and it does not replace the complete retrieval stack with
a pure dense-search baseline.

The control is a fresh execution of the sealed r9 reduced30 prompts. The
candidate uses exactly the same global raw excerpts, episode exchanges, fact
quotes, typed advisories, numeric-completion hints, dated questions, system
policy, responder, independent judge, output allowance and hard context limits.
Only the explanatory user-message prefix and raw evidence layout change.

Hard constraints:

1. Group by the complete exact source ID. Never infer personal identity from an
   opaque ID, including benchmark-specific prefixes.
2. Preserve all raw excerpt strings and whole episode blocks, including existing
   assistant-owner labels, reference markers and representation collisions.
3. Preserve all fact/advisory/completion text and original citation labels.
4. A shared timestamp describes excerpt mention time. It does not prove event
   chronology, a correction, cross-session identity, or exhaustive coverage.
5. Keep the existing 10,000-token context and 11,000-token workspace caps. If the
   complete grouped packet exceeds either cap, reuse the original prompt intact.
6. Freeze predictions before joining references; zero retries and exact call
   accounting apply to both arms. Qwen receives no raw input and makes no calls.

## Implementation and construction

`src/memory_condense/search/packing/source_grouped_packet.py` is a small reusable
renderer over already selected evidence. It groups sources in first-encounter
order, combines matching timestamp headers within each source, keeps upstream
order within each timestamp, and returns raw-text character offsets for exact
conservation checks. It performs no retrieval or model inference.

`tools/matched_eval/hot_source_grouped_packet.py` authenticates the sealed
legacy packet, reproduces its complete original rendering from its manifests,
and applies the new renderer. Full audit copying and legacy reconstruction stay
in this experimental adapter, not in the small renderer. The existing retrieval
entry points and default packet behavior are not promoted by this experiment.

All 30 packets fit without fallback. Every raw excerpt, episode block, tail and
system message is preserved. Across the cohort the context decreases by only
183 token proxies, or 6.1 per question. This is a source-attribution and layout
test, not a material context-compression result.

The exact parent selection remains:
`41d47ba7c0b42edbad0f408d1856ca563a49e11bdaf365a099ee283b9c8bfe28`.
The historical parent dependency identity remains unchanged:
`524f14c4dec54b7f03fca8671d385ffca31e765a570937d8cff99a7cf50f8d77`.

Candidate root:
`eval_results/longmemeval-fast-source-grouped-reduced30-20260908-r1`.
Its selection SHA is
`dccf679b02df741067b89d9127e7b8026adb514f99c14214a016d1d39ba713d7`.
An independent construction replay reproduced that complete artifact exactly.

## Paired execution

The grouped candidate scored **8/30**, versus **12/30** for the fresh original
packet. It gained ordinal 81 and lost 51, 82, 83, 87 and 94. Do not promote this
renderer or interpret its structurally valid source grouping as an accuracy fix.
This rejects the layout change, not conventional retrieval. The next bounded
experiment returns to the original format and tests BGE-M3 ordering of the same
selected raw global excerpts, preserving the original episode/fact tail.

The fresh control differs from the historical r9 score of 11/30. Twenty-four of
its 30 prediction strings changed despite identical prompts. That variation is
another reason to compare with a fresh control and avoid attributing individual
rescues to a packet transformation without further evidence.

Pair root:
`eval_results/longmemeval-fast-packet-rendering-pair-20260908-r1`.
The frozen pair preflight SHA is
`815f3b0c0a75ae15a22aa89229cd8a5a0951728f0f5ccf3be1a90bddf4f66507`.

| Artifact | Control | Grouped |
| --- | --- | --- |
| Answer preflight | `907085d1c923f12358d4c942652e30b8ed6e969c2cde42d59d1b1d232623e5d7` | `ccb31db4d68fd2143c71f0b063f053c0718c0332d99c669b90c3de76c9109485` |
| Sealed answers | `cc573cbebd6b2047fff7ef517afdaa88da17242d2f090e2466b2a09d68a66fd6` | `e1171bf458f8a4c74168824460b6de7aedf3514fd699a70931352e5c7f16feb3` |
| Judge preflight | `a99516396d76001f4960dc7216caa4a8cd684a1c9584677a3444714373aba0d6` | `f5865eeeee2d79dad897eb4aeb857e257ddde32b853b535f33d20fc9398681b4` |
| Sealed judgments | `f9b54ead8904a2ef3a97417c013de08207b05e98203a79e414c42de691890646` | `fd26fbaf7879df622d800b06070e24602eeae9e86cbc12371c46141490217ccf` |

Offline paired comparison SHA:
`e1f66de8da569fcde0ded9c817f7cd03bec18274ed893155455849d30a5da03a`.
Both answer and both judge artifacts replayed from 30 authenticated checkpoints each with
zero new provider calls and identical sealed hashes.

Each answer arm made exactly 30 Terra calls through the authorized local
`https://central-dev.zt:4000/v1` gateway, with concurrency 10 and zero retries.
The control answer batch took 51.407 seconds; the grouped batch took 50.270
seconds. These are consecutive concurrent batches, not randomized single-query
latency measurements. Their small timing difference does not establish speedup.
Each judge arm made exactly 30 Sol calls, with zero retries. The control and
candidate judge batches took 48.456 and 48.709 seconds respectively. Total new
calls for this comparison: 60 answers and 60 judgments, all explicitly recorded
in sealed phase run receipts.

## Renderer runtime and tests

The renderer was measured on all 30 actual packets, 30 repetitions each:

| Boundary | Measurement |
| --- | ---: |
| Raw renderer mean | 0.450 ms |
| Raw renderer p95 | 0.593 ms |
| Separate context-token count mean | 12.239 ms |

This excludes corpus loading, retrieval, hydration, legacy adapter/audit copying,
and provider time. It is neither full fast-path latency nor an ingest benchmark.
Runtime receipt SHA:
`5a8576c12268ab1444eef86e15c8031f2d9380cdc848dd6918ccc080540d51cc`.

The relevant integration suite passed **144 tests in 25.13 seconds**. It covers
opaque source isolation, raw offsets including Unicode and apparent embedded
markers, timestamp/role validation, duplicate citations, source and content
tampering, exact fallback, gold rejection, successor replay, protected evidence,
typed reductions and the answer/judge lifecycle.

## Verification and decision

Run from the ingest-speed worktree:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m pytest tests/test_source_grouped_packet.py tests/test_hot_temporal_reference_chain.py tests/test_assay_hot_v6_spine_episode_fact_ledger_full100.py tests/test_assay_hot_reduced30_construction.py tests/test_hot_v6_typed_reducer.py tests/test_hot_v6_typed_reducer_advisory.py tests/test_run_hot_reduced30_answer_judge.py tests/test_evaluate_hot_retrieval_full100_profiles.py -q --basetemp .tmp-pytest-fast-packet-NEW
.\.pixi\envs\dev\python.exe -X utf8 -m tools.assay_hot_source_grouped_reduced30 verify --output-root eval_results/longmemeval-fast-source-grouped-reduced30-20260908-r1
.\.pixi\envs\dev\python.exe -X utf8 -m tools.compare_fast_packet_pair --root eval_results/longmemeval-fast-packet-rendering-pair-20260908-r1
```

The 30 questions are the previously analyzed v6/r3 failures, not an untouched
confirmation population. A positive result here would still require a complete
100-question non-regression measurement before promotion. A negative result
would reject this rendering candidate, not conventional retrieval as a whole.
