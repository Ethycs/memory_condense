# Conversation ordered answers through persisted application memory

**Status:** COMPLETE — 85/100; warm median 3.964 s; 95% target unmet.
**Date:** 2026-09-15.
**Scope:** One persisted 1,098,417-token application memory and 100 unchanged questions.
**Depends on:** [Research Log 223](223%20-%202026-09-15%20-%20Application%20native%20memory%20ingest%20and%20reopen.md) and [Research Log 221](221%20-%202026-09-15%20-%20Dense%20seeded%20parent%20context%20on%20one%20history.md).

## Comparison

Thirteen of the 84/100 candidate's sixteen misses already contain all recorded
support. This comparison groups selected excerpts by conversation and restores
original turn order using the existing `threaded_section_context` renderer.
Every selected raw span must appear exactly once and byte-for-byte unchanged.
The retrieval policy, 2,048-token hydration budget, v6 reader, Terra answer model,
100 questions, reference answers and Sol grading policy remain fixed.

The earlier grouping experiment also changed the reader. This experiment
isolates presentation at the answer prompt: all 100 hydration and routing
receipts must equal the 84/100 baseline. No improvement is assumed in advance.
This remains development on exposed generated questions over real transcripts,
not an official LongMemEval score or evidence of generalization.

## Application lifecycle

The prerequisite integration ingested all 5,357 raw turns normally, installed
validated cached summaries/hierarchy/vectors, closed the application, exited,
and reopened it in an independent process. All 100 baseline packets matched.
This runner requires that completed receipt and verifies the saved application
files before proceeding. It loads the persisted native snapshot once and calls
`MemoryCondenser.retrieve_native_spine` for fresh timed query embedding, routing
and application-database hydration. It does not reconstruct a source namespace
or load the external summary-vector cache during answering.

One hundred candidate streams alternate with 100 identical-prompt direct API
controls. Every stream must finish and its prompt/response binding must validate
before the references are opened for 100 logical judgments. After grading,
independent raw reconstruction also checks conversation order and exact rendered
placements against the original source bank. Cold ingestion and application load
are reported separately from warm answer latency.

## Released execution

| Item | Binding |
| --- | --- |
| Root | `eval_results/native-spine-threaded100-app-20260915-r1` |
| Log | `eval_results/native-spine-threaded100-app-20260915-r1.log` |
| Exec session | `15164`, terminal exit 0 |
| Worker | PID `36524`, creation time `1789517524.351143` |
| Preflight SHA-256 | `cad30573da8434b67210948ebab1fcca68825a25cc4ead5e0bd88dd34c1b1582` |
| Reopen prerequisite SHA-256 | `ca721112762da52024df005868c3b092b8f684c5d37d2e9f99af62497c88d2e0` |
| Questions SHA-256 | `76492c4fbc3b142d803eff6bdcb86ac456bb68119431f7da5c9b20d335751bc1` |
| Retrieval policy SHA-256 | `b25c34b8ca304c1b799618722c2d1e000f33566ada6914b0c918db115a797d97` |
| Report SHA-256 | `797e7af7f942c870042794de09e22624d8a34d4536eaf9408acdf022f5ee0176` |
| Raw audit SHA-256 | `2603ee1ac376880d9f9955ca48adeffb0c3c7a5942f513c07d5b076c9ca1525c` |
| Independent comparison SHA-256 | `6fff0a906ba8be16ed7b69688ec216e408211fa830d30162198b2a9676a9c7fc` |

The completed command, for identification rather than duplicate execution:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -u -m tools.evaluate_native_spine_threaded100 run --root eval_results/native-spine-threaded100-app-20260915-r1 --policy eval_results/native-spine-context-policies/dense-parent-2048-v1.json --enable-provider
```

## Completed result

All 200 answer streams stopped normally, and all 100 judgments completed. A
provider-disabled judge replay reproduced the same report with 100 authenticated
cache hits and zero new calls. The independent comparison audit recomputed all
latency distributions and completion counts directly from the saved responses.
All 100 packets and all 2,656 exact raw spans passed source reconstruction,
including conversation order and every rendered placement.

| Measurement | Application memory | Matched direct API |
| --- | ---: | ---: |
| Accuracy under unchanged grading | 85/100 | Not separately scored |
| Warm median total | 3.964 s | 3.688 s |
| p95 total | 6.331 s | 5.539 s |
| Answers under five seconds | 79/100 | 85/100 |
| Median packet preparation | 0.299 s | Less than 0.001 s |

The 25.187-second application/model load is excluded from warm timing. The
normal raw ingestion was already completed separately in Research Log 223.
Median memory/API latency ratio is approximately 1.075. The median meets the
user's accepted five-second range; the 95% accuracy target remains unmet.

Seventeen focused admission/presentation checks passed before execution. Normal
application ingestion, independent process reopening and all 100 live answer
retrievals have now been exercised together. This establishes the tested opt-in
native API lifecycle using cached compiler outputs. It is not a fresh summary
generation timing claim or a claim about the default chunk-retrieval API.

## Gains, regressions and next action

Relative to the flat presentation's 84/100, gains are ordinals **9, 16, 62, 69,
87**; regressions are **1, 5, 81, 89**. Five gains and four losses do not establish
a reliable grouping advantage from one pair of stochastic runs. Keep both runs
separate; do not combine their correct answers.

The misses are **0, 1, 5, 6, 14, 19, 35, 53, 65, 81, 82, 89, 93, 95, 96**.
Manual checks of the served packets identify concrete reader failures:

- Ordinal 1 adds one bag of chips and four drinks from an assistant suggestion
  to the user's historical Subway combo. The user supplied neither quantity.
- Ordinals 6, 14, 19 and 65 omit served qualifications or follow-up interests:
  practical cave-art purposes, modern reproduction of the Colosseum, personal
  experience adding depth to acting, and considering a specialist inspection.
- Ordinal 81 abbreviates the supplied opening line and loses its nightmare
  contrast even though the full Arabic and English wording is served.
- Ordinal 82 omits served preferences for real-world issues and current events;
  ordinal 95 omits the served Tribeca interest. Some other reference details for
  ordinal 82 were separately flagged as beyond its question's wording.

Known evaluation issues remain separate. Ordinal 0 answers `B.` and is marked
wrong solely for not also saying `not A`. The concert/collection ambiguities
remain. The extra French/international film interest and cashback apps named
in ordinals 95 and 96 are explicit user statements elsewhere in the memory,
despite the judge calling them unsupported relative to its reference. Ordinal
35 still lacks the recorded Telegram-group support. No score or reference has
been changed to compensate for these issues.

Next work should target complete reading of relevant user statements, preserving
qualifications and keeping assistant suggestions separate from user history.
Further layout changes have little supporting evidence. Reuse the now-persisted
application history for the next isolated comparison; no new raw ingestion or
100-history preparation is needed.
