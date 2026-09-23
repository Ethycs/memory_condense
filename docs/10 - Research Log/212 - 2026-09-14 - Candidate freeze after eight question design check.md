# Candidate freeze after eight-question design check

The parent-context candidate is frozen for broad evaluation. Its latest development
check answered 8/8 correctly on the existing 1,098,417-token history. This is one
exposed official question plus seven manually authored, source-grounded probes;
it does not establish 95% benchmark accuracy or superiority over other methods.
No additional histories were constructed during these checks.

The remaining work is corpus preparation and integration of the frozen candidate
into the broad runner. The original 100-history controller remains stopped because
it binds the previous router and repeatedly replays old compilation ancestry.
Do not resume that controller. Design iteration no longer requires additional
pilot questions before the broad evaluation.

## Completed comparisons

Both runs used a 1,024-token rendered evidence cap, zero protected direct prefix,
exact raw-section hydration and fresh query embeddings inside the serving timer.
Each question received two memory answers, their two identical-prompt API controls
and one short-chat control. Answers streamed sequentially without automatic retry;
Sol judgments followed after all answers were sealed. Cold setup was excluded.

| Set and reader | Method | Correct | Median total | Matched API median total |
| --- | --- | ---: | ---: | ---: |
| Six questions, v2 | User-first | 5/6 | 4.648 s | 3.470 s |
| Six questions, v2 | Parent context | 6/6 | 4.265 s | 3.492 s |
| Eight questions, v5 | User-first | 7/8 | 4.999 s | 4.426 s |
| Eight questions, v5 | Parent context | 8/8 | 4.783 s | 4.117 s |

The eight-question set retains five earlier questions, drops the simple charity
amount and adds an explicit denial of auction wins, an actual tea preference
change versus contemplated alternatives, and report ordering across two dated
conversations. Both methods answered all three new probes correctly. Their
reference quotations were frozen before retrieval and stored separately.

The only user-first miss in each run was the official bulb question. The source
explicitly identifies the Philips LED bulb owned and used in the bedside lamp;
it does not explicitly say the bulb was replaced. Identical-prompt controls have
also alternated between answering and abstaining. The generic v5 reader clarifies
attribute questions versus evidence that an action occurred, while respecting
denials and corrections. It did not resolve the bulb instability. Neither the
reader change nor attention context has a demonstrated causal accuracy advantage.
Freeze the tested configuration and let the broad comparison assess this.

The latest parent-context median is approximately 16.2% slower than its matched
API control and 21.8% slower than short chat (3.926 s). Its total p95 is 6.156 s
versus 5.616 s for its matched control. These eight observations cannot establish
stable population latency; the joint accuracy and latency gates remain unpassed.

Reports under `eval_results/`:

- `native-spine-six-question-zero-prefix-20260914-r1/report.json`, SHA
  `75afd0fd6cfa5c6a56c507b36e574323db39942e4553c453597b641a03d647d3`.
- `native-spine-eight-question-reader-v5-20260914-r1/report.json`, SHA
  `33af4ea04fa71e917a38bd3fc3342ff0f2481708f726dd24f554b0013ea6efed`.
- Eight-question inputs:
  `native-spine-design-challenges-20260914-r1/questions.json`, SHA
  `3f520fd3b05d4e835693077913ca4200a9016323e3cc048ea8a6c4047e356924`.

## Bounded serving profile

One offline profile loaded the same resident history and rebuilt the 16 evaluated
memory packets, each with a fresh query embedding. Every message, hydration object
and routing object matched its evaluated counterpart exactly. An additional eight
parent builds collected cProfile diagnostics separately from stage timings. There
were no provider calls, Qwen calls or corpus compilation.

| Parent-context stage | Median |
| --- | ---: |
| Fresh BGE query embedding | 0.039 s |
| Summary routing and parent context | 0.080 s |
| Exact raw hydration | 0.041 s |
| Reader message construction | 0.002 s |
| Other validation and serialization | 0.008 s |
| Complete packet preparation | 0.172 s |

Component medians do not sum exactly to the total median. Cold setup took 38.723 s
and remains outside warm serving. During the preceding API evaluation, parent
preparation median was 0.325 s; the offline timing does not replace that measured
end-to-end result. The profile shows CPU routing, receipt construction and hydration
work alongside embedding, without a single large defect that warrants another
design iteration before broader evaluation. No serving implementation was changed
after the eight-question run.

Profile result: `native-spine-design-hot-profile-20260914-r1/result.json`, SHA
`6a1310ad8fd100d58d86d38d23234782a40de1350e83bcacacf9ce43c5464d7e`.
The profiling script and cProfile data are preserved with their hashes.

## Frozen candidate and outstanding preparation

The frozen method uses the v5 reader; parent context; 32 direct candidates with
two lexical reservations; zero protected direct prefix; four context seeds; one
ancestor hop; at most eight context atoms; 1,024 rendered context tokens; and 128
raw-span attempts. Terra answers have a 256-token output cap. Qwen processes only
summaries during ingestion, and its cached attention topology supplies parent
neighborhoods. Serving performs one fresh BGE query embedding and exact hydration,
with no live Qwen pass. User-first routing remains the comparison arm.

Freeze artifact: `native-spine-design-freeze-20260914-r1/candidate.json`, SHA
`53b9b9f40ba2f0ff34c802a0af814d04e127721042a20fab0ce2f072fed6fb3f`.
It binds the tested preflight and exact implementation hashes. This freezes the
candidate for evaluation; it does not label the objective complete.

After this freeze, the user said: "Under 5 second is fine, we don't need
milliseconds, and accuracy looks promising." The accepted current figure is the
observed 4.783-second median. Stop further serving-latency tuning and prioritize
the broad accuracy evaluation. Continue reporting median and tail latency with
the matched API controls; this acceptance does not imply that every answer was
under five seconds. The previous 1.10-ratio optimization requirement is no longer
a reason to delay this candidate's broad accuracy run.

The complete source store already contains summaries for all 31,166 unique bodies.
The previous parent result contains 13,468 bodies, and this selected history adds
300 distinct bodies: 13,768 parent-body bindings are available, leaving 17,398.
The count comes from existing sealed manifests; all body files were not revalidated
for a new full admission. Attention, parent merges and remaining vectors still need
completion. These shared body caches should be reused across the 100 separate
histories, with completed work loaded once rather than copied and replayed every
small batch. The full100 runner also needs to bind this candidate; the old runner
still fixes the previous 3,072-token routing policy.

Cache inventory: `native-spine-design-freeze-20260914-r1/cache-inventory.json`, SHA
`794702d47f6d08e8387fd43af383c82dc3e87f1894704cd5f28330ef1f7ad371`.
No reliable completion time is available until the remaining compilation is
scheduled without the previous replay overhead. No broad answer run has started.

## Verification

Both answer processes and the profile exited zero. A separate read-only audit
verified all 70 answer journals and 28 judgments from these two comparisons,
matched prompts, normal stop events, prediction hashes, raw-span hashes and token
counts, accuracy totals, and recomputed latency distributions. It checked exact
current implementation hashes or the preserved prior runner. The reader and
routing checks passed 16 tests during implementation.

Audit: `native-spine-design-freeze-20260914-r1/audit.json`, SHA
`db8b30d6c3df817b28733755887676641efbae9b51ae7eda4e7f05700ed17a2c`.
The v2 runner is preserved under `native-spine-reader-v5-code-snapshots-20260914-r1`.
Earlier results and the stopped pipeline remain intact. This continuation made
design progress; the 95% joint benchmark goal remains outstanding.
