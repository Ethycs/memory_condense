# Cap-eight integration and reader scope comparison

**Status:** Complete — cap-8 integrated; reader v9 evaluated and not promoted.  
**Date:** 2026-09-23.  
**Applies to:** Public native-summary retrieval and the same 100-question, 1,115,343-token history used in Log 248.  
**Depends on:** [Log 248](248%20-%202026-09-23%20-%20Cap-eight%20repair%20on%20one%20complete%20million-token%20history.md) and [Log 233](233%20-%202026-09-15%20-%20Extractive%20reader%20on%20unchanged%20user%20spine%20evidence.md).

Cap-8 is now the default of `MemoryCondenser.retrieve_native_spine` when the
persisted native snapshot includes its parent-user summary index. The reader
comparison scored **93/100**, against **94/100** for v7 on identical evidence,
while increasing input tokens. Keep **v7** as the reader.

## Integrated serving behavior

`application/native_spine_policy.py` supplies immutable defaults exactly matching
the evaluated `user-completion8-2048-direct8-v1.json` policy, SHA-256
`bc523c267708dea21110013f541f1d964265af851d07262f2b0d4881eaedc31f`.
The public native retrieval entrypoint now loads and validates the persisted
parent index, selects the completion router, and supplies the measured limits:
eight direct routes, at most eight additional user atoms, and a 2,048-token raw
hydration budget. Explicit caller limits override the defaults.

```python
from memory_condense import MemoryCondenser

with MemoryCondenser(application_path, embedder=encoder,
                     auto_extract=False, read_only=True) as memory:
    result = memory.retrieve_native_spine(query, dated_question)
```

No special evaluation subclass or explicit cap is needed for a complete native
snapshot. Parent-index corruption fails admission, including repeated attempts;
it cannot silently select the older router. Older native snapshots without a
parent index retain their original context router. `native_parent_user_receipt()`
exposes the validated parent binding, or rejects an absent parent index. The
explicit `ParentUserMemoryCondenser` facade retains the historical parent-only
behavior for controlled comparisons. Ordinary chunk-based `build_context` remains
a separate API; this change integrates the tested native-summary path.

Integration tests reopen persisted application memory and compare the public
default's complete routing and hydration receipts against the previously explicit
cap-8 path. They also exercise overrides, historical parent-only routing, and
corrupt-index rejection. Together with existing lifecycle, completion, routing,
projection and evaluation checks, **74 tests pass**. The native router still
performs zero raw reads and zero query-time Qwen passes; exact raw reads occur
only during hydration.

## Reader experiment

The earlier standalone extractive v8 prompt reduced accuracy on its own
unchanged-packet comparison, so it was not reused. The new experimental
`eval/spine_reader_policy_v9.py` retains v7 and adds a question-slot check:
collect compatible details for every requested aspect, preserve distinctions
between situations, and remove answer clauses outside the question's scope.
It contains no benchmark names, IDs, reference answers or question-specific
exceptions. It is nevertheless informed by exposed development failures.

`tools/evaluate_native_spine_reader9.py` seals all 100 original cap-8 answer
packets and changes only the system message. The entire evidence/question message
is byte-identical. Source bindings lead back to Log 248's independent audit of
100 packets and 1,851 exact raw spans. No history is rebuilt, no retrieval is
rerun, and no summary or Qwen call is made. The answer worker opens no references;
grading begins only after all 100 new answers are sealed.

The model remains `codex_sdk/gpt-5.6-sol`, with the same streaming transport,
256-token output cap and binary grading prompt. Four concurrent answer requests
make this an accuracy comparison, **not an interactive latency measurement**.
There are 100 fresh answers and 100 fresh grader calls, with no automatic retries.

| Measurement | Existing v7 + cap-8 | Candidate v9 + cap-8 |
| --- | ---: | ---: |
| Graded correct answers | 94/100 | 93/100 |
| Mean provider-reported input tokens | 1,761.60 | 1,951.60 |
| Evidence/question packets | Same 100 | Same 100 |
| Answers ending with `stop` | 100 | 100 |

The prompt adds exactly **190 input tokens per answer**, about **10.79%**.
V9 gains Q50 and Q68 and loses Q19, Q63 and Q74. No byte-identical prediction
receives opposite grades, but inspecting the changed answers exposes grading
limitations:

- **Q50, breakfast:** Both answers include the reference breakfast and a later
  oatmeal routine. V9 dates the latter explicitly; the grader accepts this as a
  non-conflicting update. V7 had also mentioned the beginning-of-March timing.
- **Q68, books:** V9 omits *The Chronicles of Narnia*, matching the narrower
  reference. The source does say the user loved its personified animals, so
  this is a scope distinction rather than removal of an invented title.
- **Q19, stand-up writing:** Both answers mention notebook ideas and applying
  lessons from an open mic. The new grade calls these unsupported, although
  the user explicitly describes them in the served evidence. Both answers
  strengthen the user's “some” terrible ideas to “many.” The grade difference
  does not establish a new factual regression.
- **Q63, burnout:** V9 adds trying painting as a hobby. The user explicitly
  proposes painting in the same self-care conversation, then expresses excitement
  about trying it. The grade rejects the addition because it is absent from the
  reference; this is not a fabricated fact.
- **Q74, botanical garden:** The two answers are close paraphrases. Both omit
  exploring plant species; only the new answer is rejected for that omission.

All original grades and the 100-question denominator are retained. No manually
corrected accuracy is claimed. The clearest existing completeness miss, **Q66**,
remains: both readers name the Garmin and separate heart-rate monitor but omit
Strava, which is explicitly present in the same conversation. More general
coverage instructions did not reliably recover that detail.

The comparison provides no reason to pay the extra prompt cost or replace v7.
The next reader design should make coverage observable through claims linked to
source turns, and assess unsupported claims against the actual evidence. Merely
adding stronger completeness instructions has now failed to establish a gain.
Any future source-aware assessment must remain separate from these frozen grades.

## Artifacts and replay

- Root: `eval_results/native-spine-reader9-cap8-20260923-r1`.
- Preflight SHA-256: `29708561b33e9282ef38559a1c0b7e3f4e22048e0f4675f015473b1b9c5eea10`.
- Report SHA-256: `dfebd61a737a8b3dce17216eceff95be1cfbda5f31b0c34009e804391a58741d`.
- Provider-free report replay completed with zero new grader calls and the same hash.

```powershell
$env:PYTHONPATH = 'src;.'
$env:PYTHONUTF8 = '1'
& .pixi/envs/dev/python.exe -m tools.evaluate_native_spine_reader9 report
```

This exposed single-history reader comparison does not replace the ten-session
913/1,000 score or establish a latency change. The original campaign, cap-8
answers, references and grader remain unchanged.
