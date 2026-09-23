# Single history design pilot

**Status:** The first reduced design check is complete. Flat scored 1/1 and
the candidate 0/1, but their prompts were identical because the baseline filled
the packet budget. Identical-prompt controls returned the opposite answers.
This demonstrates packet saturation and answer variability, not a hierarchy
accuracy difference. The 100-history pipeline remains stopped as directed in
[Research Log 208](208%20-%202026-09-14%20-%20Defer%20full100%20until%20design%20is%20finalized.md).

## Selected scope

Root: `eval_results/native-spine-design-pilot-20260914-r1`.
Selection is the existing case at ordinal zero, question ID `8a137a7f`, made
before generating answers or reading its reference. Its source namespace is
`native-spine-0b341ca78960ea878446ed15e5f7ae57273e6fae870e8669a5d7a7856d02a666`.
The source namespace SHA is
`41d4696d753ec10fcc43959bf2ffa12923b425c061a8c9ec5260576947da1248`.

The selected history has **1,098,417 actual body tokens** through the question
day, excluding generated source boundaries and metadata. It contains 522 source
occurrences of 520 distinct bodies. The earlier 1,112,070-token namespace proxy
includes source metadata and is not the actual-body figure used for admission.

`tools/prepare_native_spine_design_slice.py` reads and verifies only these selected
body artifacts. It reuses all 520 completed exchange records from the stopped
run's partial result, retains their exact section and raw-span values, and exports
the selected population for the existing attention compiler. It neither invokes
the full-corpus compiler nor regenerates exchanges or raw summaries.

Scope SHA:
`18478c4f3a4e625c9c068f5c2a8e29d1eb29a1b76721eeeaa8295f58b9ab4836`.
Selected exchange-result SHA:
`9ed2d079988412c0b427cef9262e9743de74ab69bf94132d56d6a1dd8f6448c6`.
Preparation made zero model calls.

## Compilation limited to this history

All **525 attention windows are complete**: 226 cache hits and 299 newly computed
local windows. The attention method remains user-summary-only, using the existing
Qwen prefix and head scoring policy. Its process exited successfully before the
parent generator was started.

Attention preflight SHA:
`bc7900a7c2be6202b8fc026fa7da2ffc81daadd128f4e33eb4cf2c4c9750ce32`.
Attention result SHA:
`60823fc93fc93abd0701e9586662f1797798f0f21b318b9e103668f89a47ef85`.

`tools/compile_native_spine_design_slice.py` reuses 220 existing parent templates
and compiles only the remaining 300 bodies. Its first pass completed 266 of those
without generation, leaving 34 initial summary jobs. The local generation cap
is 128 jobs for this selected history. Accepted parent merges are loaded once;
the full exchange compiler and its ancestor body scans are never replayed.

Parent compilation completed all 520 bodies using **35 new local jobs in ten
batches**, including one ordinary refinement. Parent result SHA:
`c7631e1da83de9c37d91b1dd15ed1d66eb7ea633cc84fa2681609a3cda981684`.

The vector phase selects only this history's atomic summary texts and reuses
the completed R5 vectors. It does not embed the full source bank. Attention,
parent generation, embedding and answering execute in separate sequential
processes so that GPU models do not overlap.

All **5,340 selected atomic summary vectors** are complete: 2,188 reused and
3,152 newly embedded. Vector result SHA:
`15f6ba779cc4abecd1c9918fee3879abe7c2693d56652a1ca96b9f839e34dfca`.

## Initial design measurement

`tools/evaluate_native_spine_design_slice.py` is separate from the full100 runner.
It uses the same resident retrieval, exact hydration, prompt construction and
streaming measurement functions. It requires complete coverage of the selected
history and at least 1M actual eligible tokens.

The initial experiment has one locked original question:

- Flat atomic-summary retrieval and the attention-context candidate each produce
  a fresh answer. Each memory call embeds the query, routes summaries, hydrates
  exact raw evidence and constructs its prompt inside the measured interval.
- Each memory arm has its own identical-prompt direct API control; a fifth call
  measures the short-chat control. Resident construction and model warmup are
  recorded separately from warm query latency.
- All five answer responses are saved before the selected reference is read.
  Two Sol judgments score the memory answers, with zero automatic retries.
- Results report the numerator and denominator as zero or one correct out of
  one question, and individual latency observations. No 95% claim or meaningful
  p95 estimate is possible from this initial sample. More design questions can
  be evaluated against the same cached history without constructing 99 others.

Twenty-one focused checks pass, including isolated history selection, refusal
of missing selected exchanges, exact raw hydration and baseline preservation,
five fresh streams before any gold access, two judgments, and report replay
without additional calls.

## Actual answers and timings

Five fresh Terra answer calls and two Sol judgments completed, all with a stop
finish reason. The report replayed identically without additional calls, and the
evaluation process exited zero. Cold resident setup was 60.783 seconds and is
recorded separately from the following warm-query timings.

Question: "What type of bulb did I replace in my bedside lamp?"
Reference: "Philips LED bulb". The reference was loaded after all five answers
were sealed.

| Call | Answer | TTFT | Total | Live preparation |
| --- | --- | ---: | ---: | ---: |
| Flat memory | Philips LED bulb | 10.877 s | 10.882 s | 0.459 s |
| Flat identical-prompt API | I don't know | 3.906 s | 3.907 s | <0.001 s |
| Attention-context memory | I don't know | 5.835 s | 5.836 s | 0.383 s |
| Attention identical-prompt API | Philips LED bulb | 8.414 s | 8.415 s | <0.001 s |
| Short-chat API | I don't know | 3.440 s | 3.441 s | <0.001 s |

The memory-arm judgments are therefore flat **1 correct out of 1** and candidate
**0 correct out of 1**. These are individual outcomes, not reliable accuracy
rates. Do not interpret the candidate's shorter single total time as evidence
that it is faster in general.

Evaluation report: `evaluation/report.json`, SHA
`3c9ec0efde35b06d6b1be63e71b77045adccdef9a99248bc3473b17da742d542`.

## What the small test exposed

The flat packet and the candidate packet are **identical**. Each contains 17
hydrated sections and 3,065 context tokens under the 3,072-token cap. Both read
32 raw turns and include the gold phrase. The candidate added zero atomic
sections, while preserving the full baseline packet. It had only seven tokens
of remaining context budget.

Subsequent routing inspection separates two issues: zero additions occurred
**before hydration**, because the eight consulted exchange leaves each contained
two atoms already present in the 32 direct candidates. The router never visited
their attention-built parents. Packet saturation would also starve candidates
appended afterward, but it did not cause the empty candidate expansion in this
run. Merely increasing the raw token cap would not fix that routing behavior.

All four evidence-bearing requests have message SHA
`b77725c394a40ace6b0568fea079a466a563704721711dc98b3fdc2f23e2d7e7`.
Two answered correctly and two abstained despite identical prompts. The score
difference therefore cannot be attributed to attention or different retrieved
evidence. The read-only packet audit is `evaluation/packet-comparison.json`, SHA
`cc930b705e61de11e6961f004e76fb6b49d44dfa556cd155119cccd68672a3ec`.

The next design work is to give hierarchical context a meaningful role within
the finite packet budget and investigate answer consistency when the relevant
raw evidence is present. Use this already compiled history for those iterations;
add a small fixed query set and keep the original result unchanged. No further
100-history preparation is needed while resolving these design issues.
