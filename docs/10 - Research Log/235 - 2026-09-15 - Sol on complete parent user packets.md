# Sol on complete parent user packets

The earlier Sol comparison in Log 228 tied 93/100 on the older direct-eight
packets, with 97/100 recorded-support coverage and conversation-order layout.
The current parent-user packets have 99/100 support coverage and group exact
user statements before assistant context. This run tests Sol on that completed
pipeline, keeping the v7 reader and original grading fixed.

The paired-answer diagnostic in Log 234 confirmed reader omissions and grading
defects on identical prompts. It did not establish an accuracy improvement or
authorize combining predictions. The shorter v8 reader scored 90/100 and is
not used here.

## Fixed comparison

`tools/evaluate_native_spine_parent_sol100.py` preserves the current parent-user
application, persisted summary index, retrieval policy, v7 instructions, exact
hydration and v2 presentation. Every evidence artifact, including the complete
messages, must equal the Terra baseline. Only the answer-model alias changes
to `codex_sdk/gpt-5.6-sol`, authenticated against the existing gateway inventory.

The original model-comparison journal validator binds the selected model to
every request/response, preserves the 256-token limit and rejects gaps or
unacknowledged requests. The grader remains the original Sol protocol. All 200
answers must seal before references open for grading. The application is
reopened once; each timed memory answer retrieves afresh. Direct API controls
use identical prompts and alternate order with memory answers.

Twenty-eight current-packet, model-journal and parent-evaluator tests pass in
3.92 s. They reject changed evidence or reader instructions, model relabeling,
changed outputs/caps, altered request bindings, incomplete populations and
unauthorized raw-answer model aliases.

## Completed evaluation: 94/100

- Root: `eval_results/native-spine-app-parent-sol100-20260915-r1`
- Log: `eval_results/native-spine-app-parent-sol100-20260915-r1.log`
- Exec session `62006`, terminal exit 0.
- Provider-free judge replay: session `77152`, terminal exit 0, 100 cache hits,
  zero new calls and the identical report hash.
- One existing history: 1,098,417 eligible raw tokens, 5,357 raw turns.
- Baseline: `native-spine-app-parent-users100-20260915-r1`, 93/100.
- Policy: `dense-parent-2048-direct8-v1.json`.
- Reader: `user-coverage-v7.json`.

The full run scores **94/100**, gaining ordinals 19 and 81 and losing ordinal 58
under unchanged grading. The gains preserve the acting-depth qualification and
the complete story-opening contrast. Every prompt and evidence artifact matches
the Terra baseline, and all 100 packets / 1,459 raw spans pass independent audit.

| Measurement | Memory | Matched direct Sol |
| --- | ---: | ---: |
| Warm median total | 3.854 s | 3.532 s |
| Warm p95 total | 5.021 s | 5.776 s |
| Answers below five seconds | 93/100 | 92/100 |

Fresh retrieval preparation has a 0.288 s median. Cold resident setup took
25.733 s and is excluded from warm latency. All 200 responses end with `stop`
and report the Sol alias. The memory/API median ratio is 1.091. No new ingestion
or Qwen pass occurred. Gateway buffering still prevents a claim about true
incremental visible decoding.

## Remaining grades and the new evaluation finding

Failed ordinals are **53, 58, 82, 93, 95 and 96**. The new ordinal-58 failure is
a confirmed grading error, not a reader regression. The original user turn says:

> I live in the city, and I'm open to exploring different genres. I've been
> attending local music events like open mic nights at coffee shops, and I'd
> love to discover new artists.

Sol accurately reports prior attendance. The grader nevertheless claims that
the user only expressed interest, following an incomplete reference. Both the
served raw packet and the reference author's saved source turns contain the
explicit attendance statement.

53 and 93 retain the documented ambiguous-question issues; 53 additionally has
partial coverage of its designated reference support. 82 now states the requested
drama requirements but the grade demands the unrequested show/character choice.
95 includes source-supported French/international preferences absent from the
reference. 96 distinguishes household apps from additional online-shopping apps,
but the grade rejects the extra app names. Preserve these original grades.

The best unchanged automated score is **94/100**, not 95. A source-grounded review
of all 100 predictions, including passing predictions, is the next diagnostic.
It must remain separately reported; do not repair only one failed grade to claim
the threshold. The user has been asked whether source-grounded adjudication may
become the acceptance criterion while retaining the original score. Until that
decision, the original gate remains unmet.

- Preflight: `e3e54c02977f214f43bc373792c92f43fea66e80bf979cbffebe7594d9928e2d`
- Joint report: `d5c1aaaeb460934eb55d010ee949b1b0685ab9d9dc480cd1e4513cb5f1c7a964`
- Raw audit: `3eae13e9ef821a55f961cef0f7f9007f0494601dbb36c12fc2a00d24d3d27c69`
- Comparison: `80b90c9ed97aeac35178d1c26a6d09d328a59c68f8a098712b59e983741d5578`

This is an exposed development set over real transcripts, not an official
LongMemEval or generalization result. The 95% goal remains active.
