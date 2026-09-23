# Whole assistant-section ablation on the user spine

**Completed: 96/100 under the original grader, with a 4.947-second warm median
versus 4.356 seconds for matched direct API calls.** The full fixed-benchmark
target is verified on one 1,098,417-token history. This is an exposed development
set result, not held-out generalization or human-adjudicated accuracy.

The full source review in Log 236 found answer attribution errors despite
successful retrieval. In particular, an answer imported assistant-suggested
penalty details into what the user wanted, and another promoted an uncertain
furniture date to a definite one. The v7 reader already tells the model to
preserve speaker and certainty. Test the effect of the supplied context while
keeping that reader and the Sol answer model unchanged.

## Fixed comparison

Reuse the same normally ingested, persisted and reopened 1,098,417-token history,
the same user-spine summary hierarchy, the same summary-only Qwen artifacts,
the same BGE routing and parent expansion, and all 100 locked questions and
original references/grades. There is no new ingestion or Qwen call. The current
94/100 Sol run is the baseline; no predictions will be combined across runs.

`application/user_evidence_projection.py` authenticates the entire hydrated
packet, then omits only whole assistant-only sections when user-bearing sections
are present. Every user section, mixed-role section and other-role section stays
intact. An assistant-only packet without user evidence is preserved. Omitted
section IDs, original hydration identity and projected hydration identity remain
in the rendering receipt. No text is truncated, rewritten or attention-pruned.
The projection receives neither questions nor references and uses only role
metadata to select sections.

This is an ablation, not a general claim that assistant context is unnecessary.
An adopted assistant recommendation or an unresolved user reference may require
that context. The candidate is not promoted merely because its packets are
shorter or because a selected failure improves.

## Verification and execution

Twenty-nine projection, evaluation, model-journal and timestamp checks pass in
4.41 seconds. They cover exact Unicode/span preservation, whole mixed-role
retention, fallback without user evidence, rejection of changed omitted sources,
unchanged routing/hydration, no removed user evidence, and matching alternating
memory/API prompts with the unchanged reader.

`tools/assess_native_spine_user_evidence.py` reconstructs all 100 candidate packets
from saved hydration and the original body bank, without a provider or GPU call.
It does not open references and does not claim fresh retrieval or accuracy.
The answer runner separately reopens the normal application once, rebuilds every
packet and requires equality to those independently audited packets. Fresh
retrieval and projection occur again inside each memory-answer timer.

- Admission root: `eval_results/native-spine-user-evidence-admission-20260915-r1`
- Admission worker `83667` completed with terminal exit 0. All 100 packets pass
  independent source reconstruction: 1,459 spans hydrated, 1,145 user spans
  retained and 314 whole assistant-only sections omitted. Median rendered context
  falls from 1,554 to 610.5 tokens. This is not an accuracy result.
- Admission report: `5575be06fb95558af8dda6c5a00aa1e04ef5e8988522827d4302d1e26a78850c`
- Candidate root: `eval_results/native-spine-app-user-evidence100-20260915-r1`
- Answer worker `56140` completed with terminal exit 0. Do not restart it.
- Log: `eval_results/native-spine-app-user-evidence100-20260915-r1.log`
- Runner: `tools/evaluate_native_spine_user_evidence100.py`
- Policies: existing `dense-parent-2048-direct8-v1.json` and `user-coverage-v7.json`.
- Answer schedule: 100 memory answers and 100 matched direct API controls,
  serial alternating order, Sol, 256 output tokens, no automatic retries.
- Grading: the original Sol semantic grader, opened only after all 200 answer
  journals are complete. Original grading remains the acceptance gate.
- Raw audit reports both all hydrated spans and actually served spans; omitted
  assistant-only sections must not be counted as text served to the answer model.

## Full result

| Measure | Memory | Matched direct API |
|---|---:|---:|
| Original-grader accuracy | 96/100 | Not regraded in this comparison |
| Warm median total | 4.947 s | 4.356 s |
| Warm p95 total | 7.638 s | 6.938 s |
| Warm mean total | 5.286 s | 4.723 s |
| Answers below five seconds | 52/100 | 72/100 |

Median memory preparation is 0.308 seconds. Cold application setup is 30.308
seconds and is excluded from the warm numbers. The memory/API median ratio is
1.136: memory adds 0.591 seconds at the median. Every one of the 200 answer calls
returns `stop` and reports `codex_sdk/gpt-5.6-sol`. The measured median meets the
accepted under-five-second threshold; the p95 and nearly half of individual
memory answers exceed five seconds.

All 100 memory packets pass independent raw reconstruction. The audit counts
1,459 hydrated spans and exactly 1,145 served user spans, with 314 assistant-only
sections explicitly omitted. All original user evidence and routing remain
unchanged. This is a single complete candidate population, with no retries,
selected answer replacements, reference edits or combined cross-run score.

Against the preceding Sol 94/100 run, gains are ordinals **93 and 95**, with no
graded losses. Ordinal 93 now selects the reference's vinyl/camera/novel set,
though the source question remains ambiguous across multiple real collections.
Ordinal 95 still includes source-supported French-film interests, which the
grader now accepts; this is evidence of grading variation, not proof that the
earlier answer was false. The two-point score increase does not establish a
statistically reliable improvement from the context change.

The unmodified failures are **53, 58, 82 and 96**. The first has ambiguous concert
scope. The open-mic and drama grades retain the source/reference defects recorded
in Log 236. The new cashback answer no longer separates household use from other
online-shopping apps, so it has a more substantial scope concern than the prior
answer even though both received failed grades.

Separate source inspection finds useful behavioral changes in answers that
already passed: ordinal 31 drops assistant-suggested penalty details, 38 retains
the user's educational-game description without the assistant's implementation
details, 59 drops the inferred commute-specific preference reason, 79 avoids
merging other routine conversations, and 81 preserves the complete opening while
dropping assistant-written tactical specifics. Ordinal 42 still overstates the
uncertain Victorian dating. The original grader's false passes and false failures
remain a limitation; no human-adjudicated 96% claim is made.

## Replay and requirement audit

Provider-free judge replay `28992` completed with terminal exit 0, 100 cache hits,
zero new calls and an identical joint report. Comparison worker `59701` and
requirement-audit worker `41546` also completed with terminal exit 0. All workers
are terminal; no model or benchmark process is left running by this experiment.

| Artifact | SHA-256 |
|---|---|
| Preflight | `fdf767dc1d884424737843ef65c00cb94b2c5145b932800d391214065ac338d9` |
| Joint report | `8b3a55c714da53a9dc93de61feafc34322c9f90ee7440b757905a84374726cf7` |
| Raw audit | `8ef7dc14a9f6e4b2cdf8b01e74bfece2cb828d20ed42c78ffd396542c0d0217c` |
| Completion | `400a03f97aadfacdb3b167a4895ddfdd99356cae3b78c7016d18a82cd11125b0` |
| Comparison | `fef6ed4a812016e4ad39726360d13c0075e9c90df29051fa4e70769f5f4ac965` |
| Goal requirement audit | `117123a72693b4c0e3c6dd7202d6980f758b963cbd77c73081ebc12ff31f7083` |

`tools/verify_native_spine_user_evidence_goal.py` revalidates the full application
and answer bindings, recomputes reported latency from all response measurements,
rechecks persisted application and parent-vector file hashes, and binds the
520-body/525-summary-window Qwen provenance audit to the same history. Its eleven
requirements pass: one history/100 questions, eligible raw-token scale, normal
ingestion/restart, no evaluation inputs in ingestion, user-spine summary-only
Qwen hierarchy, timed fresh retrieval, all matched answers, exact served evidence,
the unchanged 95% grading gate, the accepted warm-median threshold, and separately
reported cold/matched-API latency.

The existing compiled summary/attention caches were reused during normal
application ingestion; this result does not claim a new end-to-end Qwen build.
Assistant omission remains a tested user-recall configuration, not a blanket
default for questions requiring assistant context. The original fixed-benchmark
goal is met without changing the acceptance policy; a broader production or
held-out accuracy claim requires separate evidence.
