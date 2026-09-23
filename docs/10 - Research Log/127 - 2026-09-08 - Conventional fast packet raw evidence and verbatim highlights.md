# Conventional fast packet raw evidence and verbatim highlights

**Status**: COMPLETE — raw-only tied at 12/30; highlights scored 11/30; neither promoted
**Date**: 2026-09-08
**Applies to**: `perf/durable-ingest-pipeline`, `.worktrees/ingest-speed`; experimental changes uncommitted
**Depends on**: [Research Log 125](125%20-%202026-09-08%20-%20Conventional%20fast%20packet%20source%20grouping%20comparison.md), [Research Log 126](126%20-%202026-09-08%20-%20Conventional%20dense%20and%20cross%20encoder%20fast%20packet%20controls.md)

## Purpose and current result

After grouping and relevance ordering failed to improve the conventional
fast-packet control, these two ablations test its derived tail. Every original
raw global excerpt and whole episode remains visible in its original order.
The question, answer system policy, responder, judge and hard budgets stay fixed.

| Packet | Correct on the locked reduced30 | Mean prompt-token proxy |
| --- | ---: | ---: |
| Fresh original r9 control | 12/30 | 9,478.5 |
| Raw G/E evidence, derived hints removed | 12/30 | 9,006.9 |
| Raw G/E evidence plus verbatim MiniLM highlights | 11/30 | 9,409.4 |

Removing the old hints reduces the prompt by about 5%, but gives no aggregate
accuracy improvement. It gains ordinals 40, 43, 59, 79 and 81, and loses 42, 48,
51, 87 and 94. Its answer batch took 55.556 seconds, compared with the control's
51.407 seconds; it therefore demonstrates neither a serving speedup nor accuracy
non-inferiority beyond this single observed tie.

The highlights gained 5, 40, 43, 59 and 81, and lost 17, 51, 82, 83, 87 and 94
against the same control. Their answer batch took 54.813 seconds. No tested
presentation or tail variant improves on the original 12/30. The next
conventional control tests bounded admission of complete evidence units, with
explicit omissions, instead of retaining the full original raw candidate pool.

This remains a post-hoc development assay over the 30 historical v6/r3 failures.
It is not full100 accuracy, untouched confirmation, or a routing-model
comparison. The fresh control is reused from Log 125, not regenerated for each
candidate. The verbatim-highlight mechanism was frozen before the raw-only
judgment result was inspected.

## Mechanisms and boundaries

`tools/assay_hot_raw_packet_reduced30.py` authenticates the original raw context
and removes only the subsequent derived F/advisory/completion hints. Numeric
completion anchors and operands must still resolve to retained raw citations;
in the real ordinal-75 packet the Maui operand remains in `E4.U1`. Removed hint
manifests remain in a clearly named parent-audit field. Provider-facing fact
counts and provenance are updated to show no F facts. No raw evidence is removed.

This does **not** demonstrate that query-time fact compilation can be deleted.
The frozen r9 raw selection already includes evidence admitted by that process.
This assay only removes its provider-visible hints. A replacement for the
selection work requires its own reconstruction and latency measurement.

`src/memory_condense/search/packing/evidence_highlights.py` supplies a reusable
bounded renderer for verbatim navigation snippets. It accepts ranked raw
excerpts, copies at most six 48-token prefixes, and caps the complete highlight
block at 450 token proxies. Each snippet retains its original G citation,
speaker and timestamp. Non-verbatim Unicode-boundary decodes are skipped, and
references are never partly clipped to make them fit. The header explicitly
states that the snippets are partial and that the full raw evidence must still
be checked. These snippets certify no answer, completeness, identity or absence.

`tools/assay_hot_highlight_packet_reduced30.py` reuses the frozen MiniLM relevance
scores from Log 126 to choose these highlights. It replaces the old derived
tail with the highlights while retaining every full raw G/E excerpt in place.
No new local model scoring is needed for construction or replay. The highlight
block averages 401.9 token proxies, and the largest resulting workspace is
10,744 tokens, within the unchanged 11,000-token cap.

The snippets are opening prefixes rather than query-centered extraction spans;
an answer appearing later in a cited excerpt remains available only in the full
raw block. This is a limitation of the frozen candidate, not a claim that its
snippet contains every relevant fact.

## Sealed artifacts

Roots below are under `eval_results/`.

| Artifact | Root / SHA-256 |
| --- | --- |
| Raw-only construction root | `longmemeval-fast-raw-packet-reduced30-20260908-r1` |
| Raw-only selection | `248a3d521b0b7c1d79bf2fd6ce37a8777795a624558b4fbf1e44e4000fc3519e` |
| Raw-only evaluation root | `longmemeval-fast-raw-packet-pair-20260908-r1/raw_only` |
| Raw-only pair preflight | `e1f1674a04cd5ad4b85e5818b75854e93989e87a7a097e668d3b847a3811c9fa` |
| Raw-only answer preflight | `625112cc494d7675463841c18b81e010b58a7035e908f921d6f74bfca84208e6` |
| Raw-only answers | `707a079a28e6c892cd8d4f26238f45b5b43a455fca119248125867ab444f60e4` |
| Raw-only judge preflight | `6889447cfd0736a9a9599d12a71c3e2343264d106aa22f103f0eb4d5739f2c4a` |
| Raw-only judgments | `d03dfd862a1e304a172a639a17a0004e09b65556ab5821d867df1eefcec8fa8f` |
| Raw-only comparison | `c7c553c79936c95c2b19598a60e22b5acfbf2990b0d9639550c4b805663e7f3e` |
| Highlight construction root | `longmemeval-fast-highlight-packet-reduced30-20260908-r1` |
| Highlight selection | `25babfa13d0409407a33ad4e5faad3eb94179a80276678bac877c4bb3ed62bff` |
| Highlight evaluation root | `longmemeval-fast-highlight-packet-pair-20260908-r1/highlights` |
| Highlight pair preflight | `d13324599f5b1b406feffa618973f15a2cd6fa6f11f8ca9746ae4d21a37ca3be` |
| Highlight answer preflight | `a286e54e607e95ad14bb4b20c1bbf81a827b9332e8dd2cf557d98f08708ce0b2` |
| Highlight answers | `5af9b7ebaae72b4482e10f6dfce33d2369020bf0b423bafb46637819eec59be8` |
| Highlight judge preflight | `af7d90fc73c98aa0fb28593bfdd68957b9c5c5f924ca03364b9dc0617aafe7f3` |
| Highlight judgments | `69f96a4605085dad086de5e774d2238f23ecfad80a7cc0fed69c6f833227ccca` |
| Highlight comparison | `d4506cb8376d2a67cf8173a0d61a4c3559cf0f4ace674967d8d7f811f6d56e59` |

Each new arm runs exactly 30 Terra answers and 30 independent Sol judgments
through the authorized local gateway, with concurrency 10 and zero retries.
Predictions are sealed before reference-answer joins. Construction is provider
free; exact replays reuse the frozen parent, vectors/scores and checkpoints.
Run receipts distinguish new calls from authenticated checkpoint hits.
All four provider phases in this log replayed from 30 authenticated checkpoints
each with zero new calls and identical sealed hashes. The six distinct arms
across Logs 125–127 used exactly 180 Terra answers and 180 Sol judgments: 360
new provider calls total. Reused controls add no calls.

## Verification and decision

The raw-packet construction and lifecycle suite passed 18 tests. The highlight,
ordering and lifecycle suite passed 21 tests. These overlap with earlier runs
and must not be summed as a unique-test total. Both complete real candidate
selections replayed to their original sealed hashes before answer execution.
The final combined packet, retrieval, selector, reducer and lifecycle suite
passed **183 tests in 28.71 seconds**; `git diff --check` was clean.

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.assay_hot_raw_packet_reduced30 verify --output-root eval_results/longmemeval-fast-raw-packet-reduced30-20260908-r1
.\.pixi\envs\dev\python.exe -X utf8 -m tools.assay_hot_highlight_packet_reduced30 verify --output-root eval_results/longmemeval-fast-highlight-packet-reduced30-20260908-r1
.\.pixi\envs\dev\python.exe -X utf8 -m tools.compare_fast_packet_pair --root eval_results/longmemeval-fast-raw-packet-pair-20260908-r1
.\.pixi\envs\dev\python.exe -X utf8 -m tools.compare_fast_packet_pair --root eval_results/longmemeval-fast-highlight-packet-pair-20260908-r1
```

Keep conventional retrieval as the baseline and hierarchical attention on hold.
Do not promote a candidate based solely on fewer prompt tokens, a single tied
score, or the correctness of its conservation tests. Packet answer reliability
must be measured separately from routing diagnostics and renderer speed.
