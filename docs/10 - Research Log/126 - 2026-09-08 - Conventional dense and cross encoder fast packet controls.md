# Conventional dense and cross encoder fast packet controls

**Status**: COMPLETE — BGE ordering rejected; MiniLM tied the control with no accuracy gain; neither promoted
**Date**: 2026-09-08
**Applies to**: `perf/durable-ingest-pipeline`, `.worktrees/ingest-speed`; uncommitted experimental controls
**Depends on**: [Research Log 125](125%20-%202026-09-08%20-%20Conventional%20fast%20packet%20source%20grouping%20comparison.md)

## Result and scope

Conventional routing remains the active baseline. Following the rejected source
grouping experiment, two standard relevance controls reorder the existing global
raw excerpts. They retain the original packet format, complete raw membership,
episode exchanges, fact quotes, numeric hints, typed advisories and answer policy.
They neither add Qwen inference nor search a different corpus.

| Packet | Correct on the locked reduced30 | Decision |
| --- | ---: | --- |
| Fresh original r9 control | 12/30 | Current control |
| Grouped by exact source | 8/30 | Rejected, Log 125 |
| Original format, BGE-M3 order | 10/30 | Rejected |
| Original format, MiniLM cross-encoder order | 12/30 | No accuracy gain; not promoted |

BGE ordering gained global ordinals 6, 40, 49 and 59, and lost 48, 51, 66, 82,
86 and 87 against the fresh control. It changed all 30 prompts but left their
token counts exactly unchanged. This supplies no measured accuracy benefit from
putting dense matches first.

MiniLM gained 5, 40, 43, 59 and 81, and lost 42, 48, 82, 83 and 87. Its 12/30
tie and additional scoring cost supply no demonstrated advantage over the
original conventional packet. Retain the original order. The next packet test
removes the derived fact/advisory tail while keeping all original raw evidence,
to isolate whether those hints help final answer construction.

These are presentation-order ablations over an already selected conventional
candidate set. They are not pure BGE-versus-BM25 full-corpus retrieval tests,
not a matched conventional-versus-hierarchical-routing comparison, and not a
full100 answer result. The post-hoc failure30 cohort has already informed
development. The same fresh control is reused for both ordering arms; it must
not be counted as a new set of provider calls each time it appears in a table.

## Frozen implementations

`tools/assay_hot_dense_order_reduced30.py` compiles the 1,905 unique selected raw
global excerpt strings with the repository's verified BGE-M3 checkpoint. It
normalizes their vectors and separately encodes the 30 plain questions. Existing
`ExactDenseAddressIndex` cosine search orders every selected global excerpt;
exact evidence IDs break ties. Excerpts and questions are bound to their text
hashes, and vector files are sealed by SHA-256 before provider execution.

`tools/assay_hot_cross_encoder_order_reduced30.py` scores the same selected
global excerpts with the installed, verified
`cross-encoder/ms-marco-MiniLM-L6-v2` checkpoint, revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8`, weights SHA
`821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`.
It binds the configuration and tokenizer files as well. Query and passage
scoring windows are limited to 128 and 320 token proxies, with the actual model
pair capped at 512 tokens and batches of 16. These windows affect scoring only:
the provider still receives each entire original excerpt. Relevance logits are
used for descending order, with stable upstream-order ties, not as probabilities
or evidence of completeness.

Both controls preserve the original G citation labels even when their display
order changes. The legacy manifests retain the parent order; the new
`dense_packet_order.ordered_rows` or
`cross_encoder_packet_order.ordered_citations` audit explicitly records the
actual candidate display order. Original E references and F backing citations
therefore continue to resolve. The exact E/F/advisory tail remains byte-identical.
No factual closure or personal identity is inferred from a relevance score.

Both implementations are isolated assays, not promoted changes to default
retrieval. The source-grouped renderer is also retained only for reproducibility.
The historical r9 code and sealed artifacts are unchanged.

## Timing boundaries

| Measurement | Observed |
| --- | ---: |
| BGE address preparation, including load, selected excerpts only | 40.060 s |
| Warm BGE query encoding, mean / p95 | 32.198 / 47.070 ms |
| MiniLM scoring all selected globals, mean / p95 | 87.831 / 91.982 ms |
| Control Terra answer batch, concurrency 10 | 51.407 s |
| BGE-order Terra answer batch, concurrency 10 | 52.779 s |
| MiniLM-order Terra answer batch, concurrency 10 | 58.027 s |

These are separate boundaries. Model scoring excludes upstream retrieval,
hydration, packet audit work and provider time. BGE address preparation is not a
full-corpus ingestion benchmark. Provider batches ran consecutively, not in a
randomized latency experiment. No serving speedup is established here. The fresh
control's mean per-call provider elapsed time was 16.686 seconds; BGE ordering
measured 16.185 seconds. The small difference is not a reliable speed claim.

## Artifact identities

All roots below are under `eval_results/`.

| Artifact | Root / value |
| --- | --- |
| BGE construction root | `longmemeval-fast-dense-order-reduced30-20260908-r1` |
| BGE addresses | `52b3e02b6c0a3cb28dc44db0d6b23fe154e70b0c80cfce01b96d68ef0963f1af` |
| BGE selection | `23139f263077158a9a9801feab98aa2dcc807b436d2fba0532ce42f2b9ccdba2` |
| BGE evaluation root | `longmemeval-fast-dense-order-pair-20260908-r1/dense` |
| BGE answer preflight | `40045dc060f92e035dd3b2a165b6ffeb03d8053c1c569566ab31288330476899` |
| BGE answers | `efd3b01783ea23ce9dcf393852d62b475890713730192efac86cbb8a1e855684` |
| BGE judge preflight | `66f61bd16b98a2088f221f1e8c80f38fe357fc890080d2665c94264ab7b77e33` |
| BGE judgments | `ff91d2a852c62124266a19e541b83cac188c0ec4e33c352532be05aa1c505d6f` |
| BGE comparison | `d033e0ebc4e6ef28b0fae15160312a0f5ec7649f124a0e66a7efcc83bbdab5c7` |
| MiniLM construction root | `longmemeval-fast-cross-encoder-order-reduced30-20260908-r1` |
| MiniLM scores | `2c6e51f0a0159b894f3933da511a7b0f836bbfc75924335bfd3b737b6e38033c` |
| MiniLM selection | `521238984fb1a6e446098a40e15cf5c622b27de27ee3ae9080062e195dc17aa7` |
| MiniLM evaluation root | `longmemeval-fast-cross-encoder-order-pair-20260908-r1/cross_encoder` |
| MiniLM answer preflight | `0b1f26d3e40465dbdfbd8ebd5f1d77c2ae8b872ff94d3c2916d6700cd30f0ed9` |
| MiniLM answers | `7d275405a781c030c20b7dc5b50d22839dfd6a198117dfecbc12ae0335d1f05b` |
| MiniLM judge preflight | `bec84882762f04b3d0fc8d85804bdba30714386fb2a91d3daee95b03048f325c` |
| MiniLM judgments | `adeb8af16668a5f5c57f70efd7e70f4299a670f574c469c5665f73f7a94d6719` |
| MiniLM comparison | `a99cbee46bc2d00e269fd9f9b4284661cb33083ee4ea25503184931387c62a35` |

The BGE construction replay and both completed provider-phase zero-call replays
reproduced the same sealed hashes. The MiniLM construction and both provider
phases also replayed exactly, with zero new provider calls.
Each new ordering arm authorizes exactly 30 Terra answers and 30 independent Sol
judgments through the user's authorized local gateway, with zero retries and
concurrency 10. Gold is joined only after predictions are sealed. Phase receipts
record actual new calls and batch wall time. All model inference used for
ordering ran locally, without completion calls.

Across the fresh control, source grouping, BGE ordering and MiniLM ordering,
exactly 120 Terra answers and 120 Sol judgments were made: 240 new provider calls
in total. Reusing the same control in the three comparisons adds zero calls.

## Verification and next decision

The packet-conservation and lifecycle integration suite passed 144 tests. The
additional dense ordering, raw retrieval and lifecycle suite passed 65 tests;
the cross-encoder ordering and selector suite passed 18 tests. These runs overlap
and must not be summed into a unique-test total. New tests verify stable exact
replay, complete raw preservation, unchanged answer policy, vector tampering and
score-to-text binding failures.

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.assay_hot_dense_order_reduced30 verify --output-root eval_results/longmemeval-fast-dense-order-reduced30-20260908-r1
.\.pixi\envs\dev\python.exe -X utf8 -m tools.assay_hot_cross_encoder_order_reduced30 verify --output-root eval_results/longmemeval-fast-cross-encoder-order-reduced30-20260908-r1
.\.pixi\envs\dev\python.exe -X utf8 -m tools.compare_fast_packet_pair --root eval_results/longmemeval-fast-dense-order-pair-20260908-r1
.\.pixi\envs\dev\python.exe -X utf8 -m tools.compare_fast_packet_pair --root eval_results/longmemeval-fast-cross-encoder-order-pair-20260908-r1
.\.pixi\envs\dev\python.exe -X utf8 -m pytest tests/test_cross_encoder_packet_order.py tests/test_dense_packet_order.py tests/test_cross_encoder_selector.py -q --basetemp .tmp-pytest-ordering-NEW
```

Choose the next conventional fast-packet change from actual answer failures,
not from synthetic summary routing scores. Keep hierarchical attention on hold.
No failed ordering candidate justifies replacing conventional routing, and no
reduced30 gain alone establishes full100 accuracy or safe default promotion.
