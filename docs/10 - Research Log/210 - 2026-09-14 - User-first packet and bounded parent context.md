# User-first packet and bounded parent context

The follow-up uses the same cached history as Log 209: 1,098,417 actual eligible
body tokens, 520 unique bodies, and the existing bulb question. No history,
summary, attention window, parent or document vector was rebuilt. This question
and its reference were already exposed during development; repeated answers
must not be reported as independent benchmark questions.

## Diagnosis and implementation

The previous context router only consulted exchange leaves. Each of the eight
selected leaves had two atoms already in the direct shortlist. Its zero
additions therefore happened before hydration. The nearly full raw packet was
a separate problem: appending candidates after all 32 direct routes would
usually leave them little rendered-token budget.

The new opt-in context router preserves the first four direct routes and every
original candidate address, but schedules remaining user evidence before long
assistant replies. Its parent-context option visits one stored ancestor for
each of four direct seeds, takes at most eight context atoms, and prioritizes
user facts across those neighborhoods. Existing exact hydration enforces the
same 3,072 rendered-token and 128 raw-span budgets. Reordering can change which
direct candidates are hydrated; this variant does not claim preservation of
the entire old hydrated packet.

Qwen still processes user-spine summaries during ingestion only. This variant
uses the resulting attention-built topology, with zero query-time Qwen passes
and no raw-text relevance scoring. The query is freshly embedded on every
timed memory call. Raw bytes are read only after selection and pass the original
hash, source, timestamp, role, span and token checks.

Code:

- `src/memory_condense/search/native_spine_context_routing.py`
- `src/memory_condense/application/native_spine_context_retrieval.py`
- `tools/evaluate_native_spine_context_pilot.py`
- `tests/test_native_spine_context_routing.py`

Fourteen focused routing checks passed. They include a parent-only user sibling
that the old leaf lookup misses, promotion of a user fact already in the direct
tail, bounded raw reads and tokens, future exclusion, and rejection of changed
raw text. The existing router and ingest compilers are unchanged.

## Real data comparison

Question: "What type of bulb did I replace in my bedside lamp?"
Reference: "Philips LED bulb".

Eight fresh Terra streams and six Sol judgments completed and stopped normally.
There were two repetitions per memory method, plus one identical-prompt API
control for parent context and one short-chat control. The second repetition
reversed the memory-method order. No automatic retries occurred. Model, answer
instructions, token limits, streaming measurement and gateway stayed the same.

| Method | Correct responses | Unique questions | Median live preparation | Median TTFT | Median total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original flat packet | 1/2 | 1 | 0.279 s | 4.910 s | 4.912 s |
| User-first ordering | 2/2 | 1 | 0.238 s | 4.409 s | 4.410 s |
| User-first with parent context | 2/2 | 1 | 0.352 s | 5.298 s | 5.299 s |

The parent identical-prompt API control answered correctly in 5.767 seconds
total, with TTFT 5.766 seconds. Short chat abstained in 2.970 seconds. These are
single control observations; they do not establish latency distributions or
the target's joint latency gate. Cold resident setup took 41.974 seconds and is
excluded from warm timings.

The old packet retained 13 user sections and four assistant sections, totaling
3,065 tokens. Both revised packets retained 20 user sections and three assistant
sections, totaling 2,998 tokens. Relative to the old packet, seven user sections
entered and one assistant section left. The two revised packets contained the
same exact evidence set in different orders. Parent context added zero novel
atomic candidates on this question, because its bounded neighborhood also lay
inside the direct shortlist.

Thus the result supports further testing of user-first packet ordering. It
does not establish an attention advantage, general accuracy, or a stable speed
improvement. Keep parent context opt-in until a small, fixed set of context-
dependent questions shows a benefit over the user-first control. Do not expand
to 100 histories during this design investigation.

## Saved result

Root: `eval_results/native-spine-parent-context-pilot-20260914-r1`.

Report SHA:
`ce30a9be73466956b04b5f21cd04f41452b41f99d9d238abfea65042d72d34ef`.

The independent journal audit verified all eight request/prompt/response
bindings, prediction hashes, six answer-to-judge bindings and verdicts, and
normal finish events without making new provider calls. The process exited
zero. Original pilot artifacts remain intact. The 95% target remains unmet.
