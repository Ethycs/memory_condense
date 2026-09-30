# Downstream compensation check for routing ablation

**Status:** Complete — stage audit and two additional 100-answer cells.  
**Date:** 2026-09-23.  
**Applies to:** The same 1,115,343-token history and 100 questions used in Log 246.  
**Depends on:** [Research Log 246](246%20-%202026-09-23%20-%20Single%20million-token%20routing%20heuristic%20ablation.md), [Log 237](237%20-%202026-09-15%20-%20Whole%20assistant%20section%20ablation%20on%20the%20user%20spine.md), and [Log 245](245%20-%202026-09-23%20-%20Earliest%20loss%20trace%20and%20user%20completion%20routing.md).

Downstream compensations exist, but removing the presentation compensations
from both routing variants did not recover the ablated system's accuracy.
The routing advantage remains **20 percentage points**, compared with the
previous 21-point conditional result. The one-point difference is within
observed grading noise and should not be interpreted as an established effect.

## What remained downstream

- `user_evidence_projection.py` discards assistant-only sections after hydration
  has charged them against the shared raw-context budget. Log 237 introduced
  this filter to reduce attribution errors from assistant suggestions.
- `user_spine_section_context_v2.py` separately puts user statements before
  other turns. Removing the router's user-priority ordering in Log 246 did not
  remove this later role-based presentation.
- The v7 reader combines earlier completeness, attribution, correction, and
  certainty rules. Those instructions remain identical in all four cells here;
  this experiment does not ablate the reader.
- The eight-direct-match limit was selected with parent expansion enabled
  (Logs 226–227). This is another configuration dependency, upstream of the
  downstream transforms tested here. This experiment does not retune dense-only
  retrieval or establish that eight matches are its best configuration.

The legacy extractor and energy/importance ranking were already disabled in
every cell. The new user-completion repair from Log 245 is not enabled here.

## Corrected comparison

Reuse the previously authenticated hydrated packets. In each new cell, replace
the projection and user-first layout with `render_threaded_sections`: retain
every hydrated user and assistant span and interleave turns chronologically
within each conversation. Source ordering, routing, exact evidence, timestamps,
2,048-token raw cap, questions, reader text, model alias, output cap and grader
remain fixed. All 200 transformed packets preserve every hydrated span exactly;
only the context portion of each answer prompt changes.

| Routing additions | Original projection and user-first layout | All hydrated roles in conversation order |
| --- | ---: | ---: |
| On | 94/100 | **93/100** |
| Off | 73/100 | **73/100** |
| Observed routing advantage | 21 points | **20 points** |

With routing on, changing presentation yields one graded improvement and two
regressions. With routing off, it yields two improvements and two regressions.
The latter gains recover the H-1B payment detail (Q6) and the Fashionoscope
value-proposition/Turkish request (Q80), so downstream omission can matter for
individual answers. In the two new cells, routing-on answers pass 20 questions
that routing-off answers fail, with no reverse cases.

One apparent routing-off regression, Q47, has a **byte-identical prediction**
under both presentations, but receives opposite grader verdicts. Across both
arms, 66 predictions are unchanged from their projected counterparts. Original
scores are retained; no answer or grade was replaced. The small presentation
score differences do not establish a reliable gain or loss.

## Where evidence was lost

The provider-free trace binds each support quote to its original source
occurrence and user turn, rather than accepting a matching string elsewhere.
There are 191 recorded quotes over the 100 questions.

| Stage diagnostic | Routing on | Routing off |
| --- | ---: | ---: |
| Recorded quotes served | 187 | 153 |
| Recorded quotes never routed | 0 | 38 |
| Recorded quotes lost at hydration | 4 | 0 |
| Recorded quotes lost at final projection | 0 | 0 |
| Questions with all quotes before/after projection | 97 / 97 | 69 / 69 |
| Packets discarding assistant sections after hydration | 96 | 83 |
| Mean raw text tokens discarded after hydration | 874.06 | 507.61 |
| Packets with some routed user section rejected for budget | 12 | 2 |

The budget/filter mismatch is real, including user turns displaced by text that
is later discarded. It does not explain the ablated run's recorded-quote losses:
those quotes never entered its selected routes. Retaining assistant text can
provide useful context, as the two recovered answers show, but does not replace
the missing routing coverage overall.

All-role prompts are larger. With one common local chat-token estimator, mean
input rises from 1,497.18 to 2,351.01 tokens for routing on, and from 1,280.87 to
1,763.69 for routing off. These are **local estimates**, not provider-reported
usage: this gateway returned no usable usage for the new non-streaming calls.
The answers ran four at a time on cached packets, so no fresh interactive
retrieval or end-to-end latency claim is made.

## Evidence and interpretation

There was one history, no new ingestion, no new embeddings, and no Qwen calls.
The two new 100-answer cells required 196 unique answer calls; four identical
prompt pairs were deduplicated. Unchanged grading required 169 unique calls.
References did not enter packet construction or answer requests. The original
projection cells are historical, and all questions are exposed development
data. This is a joint test of omission and layout, not an individual component
attribution or a fully retuned alternative architecture.

The initial report assumed provider usage was present. Its summary step was
repaired to retain missing usage as null and report estimates separately. All
answer and grader journals were reused; no provider request was retried. The
exact original answer runner is archived and hash-checked separately from the
report-only repair. Two focused tests pass, covering full-role exact rendering,
unchanged reader/question text, and honest handling of missing usage.

- Preparation/audit: `tools/assess_native_spine_downstream_compensation.py`.
- Answer/grading runner: `tools/evaluate_native_spine_downstream_compensation.py`.
- Results: `eval_results/native-spine-downstream-compensation-20260923-r2/`.
- Report SHA-256: `1c0beeb6b9932c2c9da83c8437d904b0d370034d44ebeea41692057f4670dabf`.
- Stage audit SHA-256: `7f82b46daa8d75c9acbbea177c62a4333590cfc08cade0ba5db39c45c7b1fdb2`.
- `diagnostics.json` records the identical-answer grading flip and paired outcomes.

Provider-free report replay:

```powershell
$env:PYTHONPATH = 'src;.'
$env:PYTHONUTF8 = '1'
& .pixi/envs/dev/python.exe -m tools.evaluate_native_spine_downstream_compensation report
```

Retain the combined active routing path on this evidence, while treating the
budget/filter mismatch as a separate repair. Neither this result nor Log 246
proves every heuristic is necessary. Production defaults and original sealed
campaign artifacts remain unchanged.
