# Compact conventional fast packet admission

**Status**: REJECTED FOR ADVANCE — compact packet scored 10/30 against the control's 12/30; no full100 promotion
**Date**: 2026-09-08
**Applies to**: `perf/durable-ingest-pipeline`, `.worktrees/ingest-speed`; uncommitted experiment
**Depends on**: [Research Log 127](127%20-%202026-09-08%20-%20Conventional%20fast%20packet%20raw%20evidence%20and%20verbatim%20highlights.md)

## Experiment

The compact packet scored **10/30**, below the predeclared 12/30 advance
criterion. It gained ordinals 59, 75 and 81, and lost 51, 82, 86, 87 and 94.
The answer batch took 60.447 seconds versus the control's 51.407 seconds; mean
per-call provider time was 18.121 seconds versus 16.686 seconds. The substantial
token reduction therefore produced neither an observed accuracy gain nor a
serving-latency improvement in this run. Do not advance this candidate to full100.

The seven completed conventional packet controls are:

| Arm | Correct /30 | Mean prompt-token proxy |
| --- | ---: | ---: |
| Fresh original r9 control | 12 | 9,478.5 |
| Source grouping | 8 | 9,485.4 |
| BGE ordering | 10 | 9,478.5 |
| MiniLM ordering | 12 | 9,478.5 |
| Raw-only tail ablation | 12 | 9,006.9 |
| Verbatim highlights | 11 | 9,409.4 |
| Compact whole-unit admission | 10 | 3,413.4 |

No candidate earned promotion. The original conventional path remains the
control, and **fast-packet answer reliability is still unresolved**. These
negative packet experiments do not rule out conventional routing or justify a
switch to hierarchical attention. They also do not supply a full-corpus routing
comparison: the candidate pool was held to the existing r9 selection.

The earlier five presentation/tail changes did not beat the fresh conventional
r9 control's 12/30. This experiment changes actual admission: it scores complete
global excerpts and complete episode exchanges with the existing conventional
MiniLM model, then admits at most 24 primary units under a 4,000-token context
cap. Each admitted exchange also brings every global excerpt it references.

This is deliberately a smaller raw selection, not an additive r9 repair. It
records every omitted unit and explicitly sets `parent_rows_protected=false`.
Every admitted block is copied whole from the authenticated parent; no raw
block, assistant owner label, or reference is truncated. References cannot be
left dangling. Omission and relevance scores do not imply exhaustive coverage
or factual absence. The original question, system policy and raw G/E framing
remain fixed; the derived F/advisory tail is omitted.

The candidate pool is still the frozen conventional r9 pool. This is not yet a
new full-corpus BGE/BM25 retrieval run or a replacement for the upstream r9
candidate-generation cost. It tests whether conventional relevance plus a
compact packet can improve this operating point before hierarchical attention.

## Frozen mechanism and measurements

Implementation: `tools/assay_hot_compact_packet_reduced30.py`.

The global relevance scores are reused from the frozen MiniLM control. Complete
episode blocks are separately scored with the same pinned checkpoint and the
same 128/320 proxy-token query/passage windows, 512-token model cap and batch
size 16. For scoring, episode references are resolved to their raw text. For
presentation, original reference labels and whole exchange blocks are retained.

Admission ranks whole units by score, with stable upstream-order ties. Each
trial reserves the entire exchange plus missing G dependencies. It skips a
unit if that complete set does not fit, then tries lower-ranked units. A
conservative sum of independently counted block costs bounds admission, followed
by an exact final context count. Selected units retain original display order.

| Measure | Compact candidate |
| --- | ---: |
| Mean / maximum context tokens | 2,731.6 / 3,925 |
| Mean / maximum prompt tokens | 3,413.4 / 4,608 |
| Mean admitted units, including dependencies | 25.7 |
| Mean omitted units | 53.7 |
| Every admitted exchange dependency present | 30/30 |
| Admission mean / p95 | 9.780 / 11.476 ms |

The original control averaged 9,478.5 prompt-token proxies. The compact arm
reduces that by about 64%. Admission timing includes its token accounting and
was measured over five repetitions of all 30 real candidate sets. It excludes
model scoring, hydration, legacy candidate generation and provider time; it is
not an end-to-end retrieval latency claim.

The focused admission/lifecycle suite passed 15 tests. Tests cover whole-unit
admission, atomic dependency admission, foreign/cyclic references, primary-unit
caps, stable ties, score population mismatch and nonfinite scores. The preceding
combined packet/retrieval/reducer/lifecycle suite passed 183 tests.

## Artifacts and advance criterion

Construction root:
`eval_results/longmemeval-fast-compact-packet-reduced30-20260908-r1`.
Evaluation root:
`eval_results/longmemeval-fast-compact-packet-pair-20260908-r1/compact`.

| Artifact | SHA-256 |
| --- | --- |
| Scores | `a9a3a76c3ffdfc0ad9038baa2e0637d0f7e6daf633e66e54bcd08454b64e5a0c` |
| Selection | `57eed639a2a55c9c002032fdd7667999cfd64d79ac2a8aba5600c0afce4d5fd2` |
| Admission runtime | `610d32cce59777dc2091a3852bd055d239de4ff199fb5bf12ba01f08cd00d73e` |
| Pair preflight | `5f65d4007c0955213027e7887018d35bf2a54ed6ffd6641508f41a128574a06a` |
| Answer preflight | `df2d0a681d17af124249cfdeed2b4705737860be95706d8568ec7a4093ecbd15` |
| Answers | `03f69495a13b9472e3a8eecd8058fa8360ed5c778bc8636ee194aa1a7911fea2` |
| Judge preflight | `dff4318d92515ef3d791dc1b390033f53af4717620d0a544465b42a4233a399d` |
| Judgments | `953dc059e1f51f2a83957aa508a552ae8cd7dabb8bfe4bba95f84d0494eb0741` |
| Comparison | `902cd7437b8c4c62d4ba17232acd86aa217a4996b15e65b624c84251d0655b46` |
| Advance gate, sealed before judging | `b8ca9cd41175550ca297c34cb346c6503febecf582927fa81fd19b70a68d6dab` |

The fresh 12/30 control is reused without new provider calls. The compact arm
completed exactly 30 Terra answers followed by 30 independent Sol judgments through
the authorized local gateway, with zero retries. References are joined only
after predictions are sealed. This remains the already analyzed failure30
development cohort, not untouched validation.

Before inspecting compact judgments, the advance criterion is: at least 12/30
on this cohort, with the compact context/workspace caps and exact admitted-unit
checks passing. Given the much smaller packet, passing that screen warrants a
full100 non-regression measurement. It does not itself prove non-inferiority or
justify default promotion. A score below 12 rejects this candidate for advance.
The measured 10/30 fails that gate. Both provider phases replayed from 30
authenticated checkpoints with zero new calls and identical sealed hashes.

## Failure localization and handoff

Post-hoc inspection of two losses separates the remaining problems:

- Ordinal 51 retains both the 50 mm and 70–200 mm lens anchors, but the compact
  answer selects the older 50 mm prime. This remains a temporal interpretation
  failure with relevant raw evidence visible.
- Ordinal 86 drops the Muir Woods, Big Sur and Yosemite anchors entirely. The
  compact answer instead identifies only a Yellowstone trip. Conventional
  relevance-based admission can discard the source/story witnesses recovered by
  the existing specialized retrieval path. More aggressive compaction is not a
  justified next fix for that case.

The next mechanism needs to address these distinct failures: preserve justified
source/story witnesses during admission, and make temporal/entity interpretation
reliable when the evidence is already present. A new full-corpus conventional
retrieval control remains a separate experiment; do not relabel these fixed-pool
packet ablations as that comparison. Do not keep tuning display order against
this already analyzed cohort or use an oracle mixture of the best answers.

The consolidated report is
`eval_results/conventional-fast-packet-controls-20260908-r1/summary.json`, SHA
`701d94890bd770a82b46861e6ddc1b68167f49a8f54f5916c5fef45e0a57064d`.
It binds every pair comparison and all phase/replay receipts. Across seven
distinct arms, exactly **210 Terra answers and 210 Sol judgments** completed:
420 new provider calls. All fourteen provider phases replayed with zero new
calls and 30 authenticated hits each. The control was executed only once and
reused in six comparisons. The original r9 implementation identity remains
`524f14c4dec54b7f03fca8671d385ffca31e765a570937d8cff99a7cf50f8d77`.

## Verification

The final combined packet, admission, retrieval, selector, reducer and lifecycle
suite passed **189 tests in 27.62 seconds**. `git diff --check` passed. All new
controls are experimental and uncommitted; no default retrieval behavior changed.

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.assay_hot_compact_packet_reduced30 verify --output-root eval_results/longmemeval-fast-compact-packet-reduced30-20260908-r1
.\.pixi\envs\dev\python.exe -X utf8 -m pytest tests/test_compact_packet_admission.py tests/test_run_hot_reduced30_answer_judge.py -q --basetemp .tmp-pytest-compact-NEW
.\.pixi\envs\dev\python.exe -X utf8 -m tools.compare_fast_packet_pair --root eval_results/longmemeval-fast-compact-packet-pair-20260908-r1
```

Keep hierarchical attention on hold. Apply the advance criterion to the actual
judged result, and keep the original r9 artifacts immutable regardless of outcome.
