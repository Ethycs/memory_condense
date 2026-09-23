# Conversation ordered evidence joint full100 evaluation

The global fine-summary replacement lost useful context in Log 170. This
experiment preserves the relative-reservation route and every selected raw
span. It groups excerpts by conversation, restores original transcript order,
and adds an event-qualified v4 reader instruction. The two changes are tested
together; a result cannot isolate the renderer from the reader policy.

The renderer verifies original text, role, source and timestamp identities.
It preserves gaps and timestamp changes and records exact output placements
for every hydrated span. Readers see opaque conversation/turn labels, roles,
timestamps and raw excerpts. Source IDs and generated summaries are not added
to the reader context. Qwen's summary-only boundary remains unchanged; this
experiment adds no Qwen calls. Query-time routing still uses BGE summaries.

All100 control prompts reproduce the previous relative-reservation preparation
exactly. Across 100 packets, all 2,680 selected raw spans are preserved. Median
context size falls from 2,959 to 2,454 token proxies; removing repeated framing
saves 51,060 tokens in total. Every candidate remains within 3,072 context
tokens. Smaller packets do not establish either better accuracy or latency.

## Joint comparison

The existing ten approximately 1.04M-token memories provide all100 validation
questions. Four arms produce 400 fresh serial streamed Terra answers:

- Flat evidence with the current relative-reservation route and v2 reader.
- The same evidence grouped in transcript order with the v4 reader.
- An API control with exactly the candidate's prepared messages.
- A short API control with the same question and v4 reader, without evidence.

The candidate and its identical-evidence control are adjacent, with each arm
first for 50 questions. Group order rotates. Live query embedding, retrieval,
hydration and rendering are inside each memory response clock; resident setup
is reported separately. No query-vector reuse or response reuse is permitted.
All400 responses must seal before 200 logical Sol judgments. There are no
automatic retries. The output cap is 256 tokens; reader temperature is omitted.

The candidate must score at least 95/100 and keep median and p95 visible TTFT
and total latency within 1.10 times both API controls. Accuracy and latency
must refer to the same streamed responses, all finishing normally. The flat
arm is a paired accuracy control, without an independent joint gate here.

## Preparation

Preparation completed. Twenty-three focused checks pass, including exact-span
preservation, ordering, invalid-population rejection, the joint gate, and the
complete synthetic judging path after sealing all answers.

Review found a missing judge-prompt import in the first evaluator before any
provider execution. The original module and unexecuted preparation remain
preserved. The v2 successor imports the existing judge builder and reuses all
authenticated prepared prompts unchanged. It does not rerun retrieval to
repair this evaluator-only issue.

- Original unexecuted root:
  `eval_results/full1m-threaded-spine-joint-full100-20260910-r1`.
- Original preflight:
  `955fe4a64c4ceafd6d82d581a58594d46d15ea8dac5397ae349045aa19560f7e`.
- Executable successor root:
  `eval_results/full1m-threaded-spine-joint-full100-20260910-r2`.
- Successor preflight:
  `0570d9b977a17c44f1eb13b8a214293ce7045bfb9fe3afd2f43a8921444e5d2a`.

## Completed result: reject the combined change

| Arm | Correct /100 | Median total, seconds | p95 total, seconds |
| --- | ---: | ---: | ---: |
| Current relative-reservation control, v2 reader | 80 | 4.569 | 8.269 |
| Grouped evidence and v4 reader | 78 | 4.729 | 8.247 |
| Identical-evidence API control | Not scored | 4.379 | 11.583 |
| Short API control | Not scored | 3.221 | 4.556 |

The candidate gains five answers (13, 27, 42, 58 and 83), loses seven (0, 6,
16, 31, 74, 77 and 97), and shares 73 correct and 15 incorrect answers with
the control. The new event qualification helps the reading-duration answer,
but other duration, identification and comparison answers regress. Neither
the renderer nor the reader is isolated by this combined experiment. Do not
promote this candidate or attribute its changes solely to evidence ordering.

Candidate median/p95 total latency is 1.080/0.712 times the identical-evidence
API control and 1.468/1.810 times short API chat. Visible-TTFT ratios are
effectively the same. The matched-evidence allowance passes, while accuracy
and the short-chat allowance fail. Smaller context did not produce a faster
candidate median than the fresh flat control in this comparison. The joint
target gate is false.

All400 answers came from the requested Terra route and finished with `stop`.
The median/p95 live candidate preparation cost is 0.220/0.358 seconds. Every
timed hydration and grouped rendering matches its frozen preparation, including
all raw-span identities. Cold resident setup is excluded from these warm
query measurements. The separate native corpus remains parked.

## Judging recovery and replay

Execution session 27491 completed all400 streamed answers, then terminated
with exit code 1 during judging: seven judge responses were recorded and one
request failed in TLS certificate verification. The answer population sealed
before this failure. Do not restart answer generation or remove the failed
judge reservation.

The installed Windows trust-store transport temporarily changes state on its
SSL context while establishing connections. Shared concurrent context use is
a plausible cause of the intermittent error, not a conclusively reproduced
root cause. A successor gives each judge worker an independent client and TLS
context, retaining certificate and hostname verification and zero retries.
Its initial wrapper omitted the runtime's explicit `max_retries=0` interface;
that launch stopped before any provider request. The v2 wrapper both exposes
and enforces the zero-retry contract. Integration tests exercise the real
completion runtime, concurrent client isolation, closure and offline replay.

The complete 200-row logical judge population was rerun uniformly in a separate
root. Prior verdicts did not select replacements. All145 distinct Sol judge
calls completed successfully in session 24226. Replay session 84442 reproduced
the same report with 145 checkpoint hits and zero new calls. Both sessions
exited zero. The final combined run passes all27 focused checks, covering the
renderer, evaluation gates, complete judging path and client isolation.
`git diff --check` passes.

Final judge/report root:
`eval_results/full1m-threaded-spine-joint-full100-judge-20260910-r2`.

| Artifact | SHA-256 |
| --- | --- |
| Original answers.json | `a3f473181e33948e32b9ac433f1116fd5cdc9ba5203c40e304efecffaf417deb` |
| Judge recovery preflight.json | `df28d4f2242276eb0187390ba8216fac2d932d75529f369f0705ad4c4cbaa011` |
| Joint report | `9174afbe09948ca4095eab62598e2874ad378963ea3e363a9026cbfb56726efe` |
| Complete record | `f50b33c58b42f847e3f70fe5e3801023e5bf488e3ba20aa4c09cc1c8bc871663` |
| Paired outcomes | `1bb8786089f36cc6abc1fd87be210fa1d716a16917aa98b6093b4b5f397ed48e` |

## Next implementation boundary

Keep the prior reader and retrieval path as the control. The global fine
replacement and this broad reader replacement have both regressed. The next
bounded evidence experiment should retain all current raw evidence, restore
the v2 reader, and test whether a small amount of additional user evidence
can use the space saved by grouping. It must isolate grouping without the
v4 reader, preserve exact original spans, and avoid another global replacement
of the source context. This is a proposed next experiment, not an implemented
improvement or a new accuracy claim. The full 95% and latency goal stays open.
