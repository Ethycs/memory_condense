# Additive user evidence with unchanged reader joint evaluation

Log 171 rejected the combined conversation-grouping/v4-reader change: 78/100
versus a fresh 80/100 control, with the short-chat latency allowance also
failing. That comparison did not isolate grouping from the new reader. This
successor restores the v2 reader throughout and tests grouping separately from
a bounded addition of user evidence. The prior retrieval path stays the control.

## Candidate and evidence boundary

The existing grouped packets leave a median of 618 tokens under the original
3,072-token context limit. Their original hydration attempts a median of 32
raw spans, at most 56. The supplement uses this remaining capacity while
preserving every original selected raw span.

`FineSpineSupplement` scores the existing generated user-fragment summaries
with the same live BGE query vector used by the original route. It ranks at
most 32 eligible whole user turns, excludes turns already in the selected
packet, and visits one candidate from each source before another from that
source. Summary scores determine rank. Raw text, benchmark references and
question IDs do not rank or select evidence. Only original source identity
groups candidates; its textual value is not a relevance signal.

`supplement_threaded_context` admits at most eight additional whole user turns.
It enforces the inclusive question-day cutoff, exact whole-turn identity,
original-span preservation and the final 3,072-token reader limit. A large or
invalid candidate cannot evict prior evidence or prevent a later fitting
candidate from being considered. Total attempted raw spans, including the
original hydration, remain at most 128. Missing/stale evidence is recorded.
The original evidence is retained even when nothing can be added.

Legacy section framing may temporarily require up to 8,192 tokens internally
while validating a combined packet. That staging representation is not sent
to the reader. The actual grouped reader text remains at most 3,072 tokens.
Original source text is never rewritten or truncated to make an addition fit.

The fine addresses reuse already compiled summaries from Log 170. There are
no new summaries or Qwen calls. Qwen's existing summary-only ingest boundary
remains unchanged. The separate native corpus stays parked.

## Full100 joint comparison

The same ten approximately 1.04M-token memories provide the same 100 questions.
The six arms are:

- Original relative-reservation evidence with the unchanged v2 reader.
- Conversation-grouped evidence with that same v2 reader.
- Grouped evidence plus bounded user supplements with that same v2 reader.
- An identical-evidence API control for grouping.
- An identical-evidence API control for supplementation.
- A common short API control using the same question and v2 reader.

Each candidate/control pair is adjacent and counterbalanced, with each member
first for 50 questions. Group order rotates. All600 fresh Terra responses are
serial streamed requests with a 256-token output cap and temperature omitted.
Memory clocks include live query embedding, routing, exact hydration and
rendering. Resident setup is recorded separately. No query vectors or previous
answers are reused during timing.

All600 responses must seal before 300 logical Sol judgments. The judge uses
the tested independent verified TLS client per worker, eight workers and zero
retries. Each candidate has an independent joint gate: at least 95/100 and
median/p95 visible TTFT and total time within 1.10 times both its own exact
API control and short chat. Accuracy and timings bind to the same responses.

## Preparation

Twenty-four focused checks pass, including exact evidence preservation,
whole-turn and date constraints, token/read limits, both counterbalanced API
pairs, independent joint gates, and the complete 300-judgment path plus replay
through the actual completion runtime with synthetic clients.

Preparation session 42536 completed successfully. Every original hydration,
grouped rendering and flat control reproduces the previous frozen preparation.
All 2,680 original raw spans are preserved. The packets add 780 whole user
turns: 88 packets add eight, five add seven, six add six, and one adds five.
Median supplemented context is 2,971 tokens, maximum 3,072. Maximum total
attempted raw spans is 64. The grouped-only median remains 2,454 tokens.

All600 outbound prompts are frozen under preflight SHA
`8ddd7042c7eb97e39da0323a57a5293014f19dd1ad4b0ee1af25a296fbcd196f`.
The actual runtime preflight passed in session 10479, validating all600 calls
and ten complete memories; the smallest memory contains 1,039,791 token
proxies. The completed execution is described below. Do not release another
execution into this root.

Source root: `eval_results/full1m-threaded-spine-joint-full100-20260910-r2`.
Fine index root: `eval_results/full1m-fine-spine-addresses-20260910-r1`.
Output root: `eval_results/full1m-additive-spine-joint-full100-20260910-r1`.

## Completed result: reject both candidates

| Arm | Correct /100 | Median total, seconds | p95 total, seconds |
| --- | ---: | ---: | ---: |
| Original relative-reservation control | 81 | 5.016 | 10.181 |
| Grouped, unchanged v2 reader | 77 | 5.055 | 9.814 |
| Grouped plus user supplements, v2 reader | 76 | 5.346 | 9.350 |
| Grouped identical-evidence API | Not scored | 4.826 | 8.724 |
| Supplemented identical-evidence API | Not scored | 4.761 | 11.117 |
| Common short API | Not scored | 3.890 | 5.633 |

Grouping gains 65 and 88 but loses 5, 16, 17, 40, 52 and 77 relative to the
fresh control: 75 both correct and 17 both incorrect. Supplementation gains
53 and 65 but loses 5, 27, 40, 50, 52, 77 and 94: 74 both correct and 17 both
incorrect. Relative to grouping alone, supplementation gains 16, 17 and 53
and loses 27, 50, 88 and 94. Retain the original flat packet and v2 reader.

This isolates a regression from grouping while keeping the reader unchanged.
It cannot isolate conversation grouping, ordering and timestamp framing from
one another. The addition restores the plant-count answer, but adding 780
user turns does not produce a net quality improvement. No candidate is
promoted, and no existing score is replaced with selected successful answers.

Grouped median/p95 total ratios are 1.048/1.125 against its exact API control
and 1.300/1.742 against short chat. Supplemented ratios are 1.123/0.841 against
its exact API control and 1.374/1.660 against short chat. Visible-TTFT ratios
are effectively identical. Both candidates fail accuracy and latency. Grouping
fails the matched p95 allowance; supplementation fails the matched median
allowance. Both also fail short-chat latency. Both joint gates are false.

Live preparation median/p95 is 0.319/0.387 seconds for the control,
0.319/0.362 for grouping and 0.509/0.678 for supplementation. These are warm
query measurements; resident setup remains separately recorded.

Execution session 55478 completed successfully. All600 fresh streamed answers
sealed and their live evidence matched preparation before 300 logical Sol
judgments using 151 physical calls. Replay session 6556 reproduced the same
report with 151 judge checkpoint hits and zero new calls. Both sessions exited
zero. The independent TLS clients completed judging without the prior shared
client failure. The 24 focused implementation/evaluation checks passed before
execution; those source files remain frozen.

## Post-answer evidence coverage and hierarchy check

The existing diagnostic annotations cover 97 questions. Complete annotated
turn coverage rises from 84 for the original packet to 88 for supplementation.
Coverage increases at 19, 40, 53, 69, 77 and 87, becoming complete at 19, 53,
69 and 77. Despite that increase, supplementation has lower answer accuracy.
The diagnostic verifies exact raw slices against annotated source text. It
does not establish semantic sufficiency, alter judgments or select production
routes. In particular, newly complete coverage at 77 accompanies a regression.

Inspecting all ten actual serving indexes finds 27,062 leaf sections and zero
parent sections. Every leaf is a root in the serving index. Qwen shaped the
ingest partitions, but these complete-memory runs route BGE summary addresses
over leaves; they do not traverse a populated summary hierarchy with Qwen at
query time. The existing `SectionSummary` parent contract is also source-local:
all spans share one source and children partition those spans. A higher-level
catalog spanning conversations needs explicit references to source sections,
without weakening their exact raw-source boundary.

Stop the current packet-format and global fine-supplement experiments. The
next implementation should construct the missing parent summaries from the
existing summaries, preserve the exact leaf/source bindings, and evaluate
actual summary-hierarchy routing with Qwen while retaining the flat raw reader
format. Any broader catalog must preserve distinct conversations. This is an
implementation gap, not proof that adding parents will reach 95%. The separate
native corpus remains parked and the full goal remains unmet.

| Artifact | SHA-256 |
| --- | --- |
| answers.json | `7f769496991a29eb1afcf60d158e59deb11a099fda2c8acef85e0dad2054b6fc` |
| joint-report.json | `0949a276ec0650e9adbb15e6fb4ad25f12e0bb529ba51aa8803703315ce01f02` |
| complete.json | `02a22442b5e8aad793b9f158302fcc55a11591cc6ae35773cb6ee83410d62c7b` |
| paired-outcomes.json | `a7fe872e89c210eab86802a1fa387899c0c40e0ea5c66b830053a57ec3ecc14b` |
| coverage-diagnostic.json | `cdacc1a32c4ff7b30e5146b72faad7ae5eb06ede457545b3277c3afdfdc42e2b` |
| hierarchy-shape-diagnostic.json | `153beb4599890b111cd857f21ae9a5b52cde0ac62c85fbcaedf5871b04d8909b` |
