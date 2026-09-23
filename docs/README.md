# memory_condense — documentation tree

**Status**: Living Document
**Date**: 2026-09-23 (includes the completed ten-session battery and engineering replay)
**Applies to**: the whole repository
**Depends on**: [`Agentic Technique Master.md`](../Agentic%20Technique%20Master.md) — the style guide governing this tree

This tree follows the folder system in the style guide: each numbered folder is a prerequisite for the folders after it. A change is only "real" when backed by at least one of the three lanes — tests, documentation, code.

**Ten-session battery complete:** Ten additional histories of **1.06–1.16M eligible
raw tokens**, with 100 new questions each, scored **913/1,000 (91.3%)** under the
unchanged grader. Warm median was **4.783 s**, mean **5.238 s**, p95 **8.935 s**;
mean answer input was **1,500 tokens**, and retrieval/prompt preparation averaged
**0.282 s**. All 1,000 exact raw-evidence checks passed. Every history used fresh
application ingestion and a separate-process reopen with cached summary/attention
compilation. Two sessions reached 95%; the aggregate target remains unmet.
All workers are closed. See [Research Log 244](10%20-%20Research%20Log/244%20-%202026-09-22%20-%20Ten%20million-token%20session%20evaluation.md).

The [status report](07%20-%20Status%20Reports/2026-09-23_ten-session-battery-and-failure-review.md)
records the handoff; [Analysis 35](08%20-%20Analysis/35%20-%20Ten-session%20failure%20patterns%20and%20repair%20priorities%202026-09-23.md)
holds the failure findings, source examples, and repair priorities.

**Engineering artifact comparison:** The original longer session's implementation
and the preserved memory-generated code were checked with the same behavioral
assertions: **8/8 original, 7/8 memory**, with only the MCP energy display failing
in the memory artifact. It plausibly substitutes for the core implementation and
fixes an evaluation-side reheating issue. It has not reproduced the original
session's real-data measurements, which the replay's tool surface did not expose.
Latency is assumed equal at the earlier seconds-level rate. No new model calls
were needed. See [Research Log 243](10%20-%20Research%20Log/243%20-%202026-09-22%20-%20Engineering%20artifact%20substitution%20assessment.md).

The separately attempted fresh full-context control failed at the gateway before
any new engineering answer and is not used to judge code quality. That attempt is
closed; the user's subsequent request was to compare the saved artifacts. See
[Research Log 242](10%20-%20Research%20Log/242%20-%202026-09-22%20-%20Matched%20engineering%20control%20gateway%20failure.md).

**Real engineering-session result:** All **8 original prompts** were replayed on
the historical checkout, producing **25 changed files**. Final validation found
**237 regression checks passed**, **73 additional checks passed**, and **6/8 frozen
independent checks passed**. Earlier completed work was recovered through memory,
with a bounded current-user working conversation. The final memory save **failed**:
a separate process reopened **481 of 524 journaled turns**, leaving 43 final events
outside application memory. Live retrieval averaged **2.460 s**, but successful
ingestion batches took **176–554 s**. This demonstrates useful engineering work;
reliable, interactive context replacement remains unproven. All workers are closed.
See [Research Log 240](10%20-%20Research%20Log/240%20-%202026-09-22%20-%20Real%20engineering%20session%20replay%20outcome.md).

**Engineering token savings:** Recorded coding and summary generation used
**4.616M input tokens**, versus an estimated **49.821M** for resending the same
observed history at every response, excluding internal memory receipts: **90.73%
fewer input tokens**. This includes recorded summary overhead and retries; it is
a local-token estimate, with no full-context control. See
[Research Log 241](10%20-%20Research%20Log/241%20-%202026-09-22%20-%20Engineering%20session%20token%20savings.md).

**Previous engineering result:** The bounded replay stopped with **3/8 prompts
completed, zero code changes and zero test runs**. A repaired retrieval policy was
checked for five more coding actions; all remained reads/searches. Both workers
are closed. This setup has not demonstrated replacement of context during real
engineering work. The five-action check required 88 generation calls, with mean
per-action ingestion of 152.88 s, retrieval of 1.84 s and coding generation of
19.70 s. That failure motivated the bounded working conversation evaluated above.
See [Research Log 239](10%20-%20Research%20Log/239%20-%202026-09-17%20-%20Bounded%20engineering%20session%20memory%20evaluation.md).

**Completed question-answer evaluation:** Five additional 100-question tests scored **93, 92,
90, 96 and 91**, totaling **462/500 (92.4%)**. Each history contained over 1.1M
eligible raw tokens and followed normal ingestion plus a separate-process reopen.
Combined warm median was **4.469 s**, p95 **6.893 s**, and mean input **1,537 tokens**;
all 500 exact raw evidence checks passed. The subsequent eight-prompt raw-transcript
build replay changed 16 isolated files: **240 regression checks passed**, while
**four of eight independent behavioral checks failed**. It exposed a user-requirement
continuity gap, missing reinforcement during rapid turns, and an MCP energy-display
bug. Those runs are complete and their workers closed. See
[Research Log 238](10%20-%20Research%20Log/238%20-%202026-09-16%20-%20Five%20new%20100%20question%20evaluations%20and%20build%20replay.md).

**Previous completed result:** **96/100** on the original 100 questions over one
normally ingested and reopened **1,098,417-token** history. User-spine routing and
summary-only Qwen artifacts are unchanged; the answer context retains exact user
sections and omits whole assistant-only sections. Warm median is **4.947 s**
versus **4.356 s** for matched direct API calls; p95 is 7.638 s and 52/100 memory
answers finish under five seconds. Cold setup is 30.308 s. All 100 packets pass
raw reconstruction, and provider-free grading replay is identical. The original
fixed-benchmark threshold is met. This exposed development result has known
semantic-grader limitations and does not establish held-out accuracy. See
[Research Log 237](10%20-%20Research%20Log/237%20-%202026-09-15%20-%20Whole%20assistant%20section%20ablation%20on%20the%20user%20spine.md).
The chronological record below retains earlier, superseded results.

> **Prior completed scope: exactly 100 questions on one already ingested ~1M-token history.**
> The user explicitly rejected 100 histories. The r3 100-history controller and
> admission worker are stopped; no benchmark answers were released by that run.
> Do not resume them. The corrected run reuses the existing 1,098,417-token memory,
> with 100 source-grounded questions locked before answers. It loads one namespace
> once and uses the frozen retrieval method. Questions and reference answers do
> not enter ingestion. Follow
> [Research Log 218](10%20-%20Research%20Log/218%20-%202026-09-15%20-%20Correct%20scope%20to%20100%20questions%20on%20one%20ingested%20history.md)
> for the completed run and audit. The candidate scored **71/100**, versus 64/100
> for user-first retrieval. Warm median total was **3.700 s**, versus 3.677 s for
> the matched API control; p95 was 6.478 s and 83/100 answers finished below 5 s.
> All 200 memory packets passed exact raw reconstruction. The 95% target remains
> unmet. These are generated questions over real transcripts, not an official
> LongMemEval score. Cold load and ingestion are excluded from warm latency.
> The completed reader-completeness comparison scored **81/100**, a ten-point gain
> on the same questions and identical retrieved packets under unchanged grading.
> Its warm median total was **3.695 s**, versus 3.780 s for fresh matched API controls;
> all 100 memory packets passed raw reconstruction. The goal remains unmet.
> Follow [Research Log 219](10%20-%20Research%20Log/219%20-%202026-09-15%20-%20Complete%20reader%20on%20the%20locked%20single%20history%20100%20questions.md).
> A separate prediction-blind review of all 100 evaluation items is complete.
> It flags question/reference defects without changing either score; two questions
> have confirmed ambiguity among real source statements. Follow
> [Research Log 220](10%20-%20Research%20Log/220%20-%202026-09-15%20-%20Prediction%20blind%20quality%20audit%20of%20the%20100%20question%20set.md)
> for those limitations and the next summary-routing work on this same history.
> The completed semantic-seed/two-level-parent comparison scored **84/100**, using
> a 2,048-token raw packet with the v6 reader and grading unchanged. Warm median
> total was **3.985 s**, versus 3.869 s for the matched API. All 100 packets passed
> raw reconstruction; 97 contain every recorded support quote. That coverage is
> not an accuracy score, and the 95% goal remains unmet. No new ingestion occurred.
> Follow
> [Research Log 221](10%20-%20Research%20Log/221%20-%202026-09-15%20-%20Dense%20seeded%20parent%20context%20on%20one%20history.md)
> for the result, sealed policy, failures and next work on the same history.
> **Lifecycle boundary of the earlier scores:** Those runs exercised cached ingestion artifacts loaded
> into the native memory components, fresh question retrieval, exact hydration,
> and answer grading. They do not exercise `MemoryCondenser.ingest` or the normal
> application's persistence/restart path. They establish cached-memory answer
> accuracy, not full application lifecycle validation. The latest audit reproduced
> all 100 packets and 2,656 original raw spans with zero model calls. See
> [Research Log 222](10%20-%20Research%20Log/222%20-%202026-09-15%20-%20Memory%20evaluation%20lifecycle%20boundary.md).
> Application integration is now verified: normal ingestion persisted all 5,357
> raw turns, and a new process reopened the application and reproduced all 100
> baseline packets and 2,656 exact spans. Eleven lifecycle tests, 138 existing
> condenser tests and 17 admission/presentation checks pass.
> Follow [Research Log 223](10%20-%20Research%20Log/223%20-%202026-09-15%20-%20Application%20native%20memory%20ingest%20and%20reopen.md).
> The completed conversation-order comparison scored **85/100** through that
> reopened application, with unchanged evidence and reader. Warm median was
> **3.964 s** versus 3.688 s for matched API controls; p95 was 6.331 s and 79/100
> finished below five seconds. All 100 packets and 2,656 raw spans passed audit.
> Five gains and four losses do not establish a reliable layout improvement.
> The 95% target remains unmet. See [Research Log 224](10%20-%20Research%20Log/224%20-%202026-09-15%20-%20Conversation%20ordered%20answers%20through%20persisted%20application%20memory.md).
> The completed v7 reader comparison scored **87/100** on that same application
> memory, changing only system instructions. Warm median was **4.465 s** versus
> 3.962 s for matched API controls; p95 was 7.015 s and 71/100 finished below five
> seconds. All 100 packets and 2,656 raw spans passed audit. The 95% goal remains
> unmet. Follow [Research Log 225](10%20-%20Research%20Log/225%20-%202026-09-15%20-%20Complete%20user%20statement%20reader%20through%20application%20memory.md).
> A provider-free packet-width diagnostic retained all recorded support on the
> same 97 questions while reducing median conversations from 14 to 4. All 200
> candidate packets passed raw reconstruction. This is coverage, not accuracy;
> the next answer comparison tests 8 direct matches with the reader unchanged.
> See [Research Log 226](10%20-%20Research%20Log/226%20-%202026-09-15%20-%20Narrower%20retrieval%20packet%20coverage.md).
> The completed width-8 comparison scored **93/100** through the reopened application,
> with the same reader/model and unchanged grading. Warm median was **4.298 s**
> versus 4.251 s for matched API controls; p95 was 6.741 s and 73/100 finished below
> five seconds. All 100 packets and 1,430 exact spans passed audit. The seven misses
> include two clear reader omissions, one routing miss and evaluation defects.
> The 95% target remains unmet. Follow [Research Log 227](10%20-%20Research%20Log/227%20-%202026-09-15%20-%20Eight%20direct%20matches%20through%20application%20memory.md).
> The completed Sol comparison also scored **93/100** on identical user-spine
> packets, with three gains and three losses. Warm median was **4.615 s**, versus
> 4.383 s for matched direct Sol; p95 was 8.500 s. All 100 packets and 1,430 spans
> passed raw audit. The model switch did not improve the total score.
> Follow [Research Log 228](10%20-%20Research%20Log/228%20-%202026-09-15%20-%20Answer%20model%20comparison%20on%20identical%20user%20spine%20packets.md).
> Two complete lexical-summary diagnostics recovered the remaining empty-support
> question but exposed competition in parent expansion. Neither is promoted.
> Next work appends one lexical address after the unchanged semantic/parent routes,
> preserving the existing packet before using spare capacity. No reingestion is needed.
> Follow [Research Log 229](10%20-%20Research%20Log/229%20-%202026-09-15%20-%20Restore%20lexical%20recall%20without%20displacing%20semantic%20seeds.md).
> The append-only implementation now preserves every prior route and raw section
> across all 100 questions. Recorded-support coverage rises to **98/100**, with no
> losses; all 100 packets and 1,439 spans pass independent raw audit. Coverage is
> not accuracy. The completed 100-answer Terra comparison scored **87/100**, with
> median **4.945 s** versus 4.383 s for matched API controls; p95 was 7.347 s.
> Seven of eight regressions occurred on byte-identical prompts to the prior run,
> exposing answer/grading variability. All 100 packets passed raw audit. The
> change is not promoted on accuracy; the best completed result remains 93/100.
> Follow [Research Log 230](10%20-%20Research%20Log/230%20-%202026-09-15%20-%20Append%20lexical%20summaries%20after%20preserved%20parent%20context.md).
> The next comparison groups exact user statements before assistant context within
> each conversation, preserving all evidence, retrieval and reader instructions.
> The completed comparison scored **93/100**, with median **4.651 s** versus
> 4.581 s for matched API controls, and p95 7.084 s. All 100 packets and 1,439
> spans passed independent raw verification. It ties the best score; 95% remains
> unmet. Two remaining support gaps trace to parent-seed selection and a parent
> group that omits an earlier user statement. A Sonnet readiness request was rejected by the
> upstream account's insufficient credit, so this comparison uses the available
> Terra model. Follow [Research Log 231](10%20-%20Research%20Log/231%20-%202026-09-15%20-%20User%20statements%20before%20assistant%20context.md).
> A new persisted index of 522 complete parent user summaries now supplies up to
> two additional exact user atoms after the unchanged route. All original evidence
> survives; recorded-support coverage rises to **99/100**, with all 100 packets
> and 1,459 spans passing raw audit. The completed answer comparison scores
> **93/100**, with warm median **4.246 s** versus 3.872 s for matched API controls,
> and p95 6.518 s. The saved-grade replay confirms the result with zero new calls.
> Coverage improved but answer accuracy still ties the best result; 95% remains
> unmet. Remaining misses include reader omissions despite complete user-spine
> evidence, conversation-scope errors and documented question/reference defects.
> No history was reingested and no new Qwen pass was needed. Follow
> [Research Log 232](10%20-%20Research%20Log/232%20-%202026-09-15%20-%20Supplemental%20routing%20through%20complete%20parent%20user%20summaries.md).
> The shorter extractive reader completed at **90/100**, versus 93/100 on identical
> v7 packets, and is not promoted. Its warm median is **4.025 s** versus 3.624 s
> for matched API controls, with p95 7.406 s. All 100 packets and 1,459 spans pass
> raw audit; the saved-grade replay is identical. Twenty-seven checks pass.
> A separate audit verifies all 520 source-body hierarchies against 525 stored
> user-summary attention windows and exact atomic addresses. The best accuracy
> remains 93/100; 95% is unmet. Follow
> [Research Log 233](10%20-%20Research%20Log/233%20-%202026-09-15%20-%20Extractive%20reader%20on%20unchanged%20user%20spine%20evidence.md).
> The completed paired-answer diagnostic scores the saved API controls at
> **93/100**, matching memory, with three different passes and three different
> failures. All 35 identical predictions receive matching grades; inspection of
> the six disagreements finds both reader omissions and semantic grading defects.
> No score is revised or combined. Follow
> [Research Log 234](10%20-%20Research%20Log/234%20-%202026-09-15%20-%20Paired%20answer%20variation%20on%20identical%20user%20spine%20prompts.md).
> Sol on the current user-first packets and v7 reader scores **94/100**, a new
> best under unchanged grading. Warm median is **3.854 s** versus 3.532 s for
> matched direct Sol, with p95 5.021 s; 93/100 finish under five seconds. All 100
> packets and 1,459 spans pass raw audit; the saved-grade replay is identical.
> The sole new failed grade contradicts an explicit user statement about prior
> open-mic attendance. Its grade remains unchanged after the separate source review;
> the original 95% acceptance gate is still unmet. Twenty-eight checks pass.
> Follow [Research Log 235](10%20-%20Research%20Log/235%20-%202026-09-15%20-%20Sol%20on%20complete%20parent%20user%20packets.md).
> The separate source-grounded review of **all 100** saved Sol answers is complete
> and reproduced with 100 cache hits and zero new calls. It returned 87 correct,
> seven incorrect, two ambiguous and four invalid reviews. These are diagnostic
> labels, not a replacement score. Source inspection finds both grading defects
> and answer attribution/qualification errors that the original grader passed;
> the new reviewer also makes mistakes. All 100 items and original grades remain
> intact. The acceptance gate remains unchanged pending the user's preference. Follow
> [Research Log 236](10%20-%20Research%20Log/236%20-%202026-09-15%20-%20Source%20grounded%20review%20of%20all%20100%20Sol%20answers.md).
> The completed assistant-context ablation scores **96/100** with the same Sol
> reader, routing, hydration, questions and original grader. All user sections
> remain exact; 314 whole assistant-only sections are omitted. Warm median is
> **4.947 s** versus 4.356 s for matched API calls; p95 is 7.638 s and 52/100 finish
> under five seconds. Twenty-nine checks and the full raw/requirement audit pass.
> The original benchmark gate is met; grader limitations and lack of held-out
> generalization remain explicit.
> Follow [Research Log 237](10%20-%20Research%20Log/237%20-%202026-09-15%20-%20Whole%20assistant%20section%20ablation%20on%20the%20user%20spine.md).
>
> **Superseded 100-history preparation — historical record only:** The user
> rejected full100-scale preparation during design iteration. That iteration is
> now complete on one cached 1,098,417-token history: the latest parent-context
> candidate scored 8/8 on the small development set, including denial, preference
> update and cross-conversation ordering. This establishes no population accuracy.
> Its median total was 4.783 s versus 4.117 s for the matched API control. A separate
> offline profile measured 0.172 s median packet preparation with identical packets.
> The candidate fixes the reader, 1,024-token cap, zero protected prefix and bounded
> parent context. No further pilot expansion is required before broad evaluation.
> The user accepted the observed sub-five-second response time and asked to focus
> on accuracy. Stop serving-latency tuning; retain accuracy and timing measurement
> together in the broad run. The reported 4.783 s is a median, not a tail guarantee.
> The original controller remains stopped: it binds the old router and repeatedly
> replays compilation ancestry. The active continuation completes the remaining
> corpus from existing caches and already binds the broad runner to the frozen
> candidate. Source summaries
> cover all 31,166 bodies; existing manifests contain parent trees for 13,768, leaving
> 17,398. This is a cache inventory, not a new full admission check. See
> [Research Log 212](10%20-%20Research%20Log/212%20-%202026-09-14%20-%20Candidate%20freeze%20after%20eight%20question%20design%20check.md).
> The 51 missing exchange bodies are complete, with all 707 raw spans independently
> verified. All 17,642 remaining attention windows are now cached. The parent
> compiler has completed and validated the combined 31,166-body cache, including
> all 17,398 missing parent hierarchies. Vector encoding also finished successfully:
> 323,124 rows, including 142,846 reused and 180,278 newly encoded summaries.
> The controller has started full100 population admission and packet preparation.
> Follow [Research Log 217](10%20-%20Research%20Log/217%20-%202026-09-15%20-%20Frozen%20corpus%20ready%20and%20full100%20preparation%20started.md)
> for the current process and completed cache bindings.
> The first continuation stopped after 15,616 of 17,398 missing parent bodies.
> Its saved checkpoints and journals were authenticated, and the replacement
> controller found one missing checksum in the unfinished batch. That body was
> reproduced byte-for-byte from cached summaries and attention; only its missing
> checksum was restored, with zero new model calls. The r3 controller now resumes
> the remaining work with the same frozen evaluation.
> [Research Log 216](10%20-%20Research%20Log/216%20-%202026-09-15%20-%20Resume%20interrupted%20frozen%20corpus%20evaluation.md)
> records that recovery; the original interruption cause is unknown.
> The new broad runner binds the frozen candidate: 100 questions, two scored
> methods and matched candidate API controls. Eight focused runner checks pass,
> in addition to ten remaining-parent compiler and reader checks. The real
> answer run still awaits population admission and packet preparation.
> Follow [Research Log 214](10%20-%20Research%20Log/214%20-%202026-09-15%20-%20Complete%20exchanges%20and%20bind%20frozen%20full100%20evaluation.md).
> A separate result audit is prepared to recompute final scores and timings and
> verify every served raw excerpt against its original occurrence. Its nine checks
> pass; run it after the real report exists. See
> [Research Log 215](10%20-%20Research%20Log/215%20-%202026-09-15%20-%20Independent%20full100%20result%20audit%20prepared.md).
> The 129-token label repair remains verified at 22 tokens; its 74 software checks
> establish no answer-accuracy score. No native full100 result exists. See
> [Research Log 208](10%20-%20Research%20Log/208%20-%202026-09-14%20-%20Defer%20full100%20until%20design%20is%20finalized.md).
>
> **Historical first single-history design check:** One history contains
> 1,098,417 actual eligible text tokens across 520 unique bodies. Its exchange,
> attention, parent and vector data are now cached for further iteration. Five
> fresh answers and two judgments completed. Flat scored 1/1 and the candidate
> 0/1, but all four evidence-bearing prompts were identical. The router consulted
> exchange leaves whose atoms were already in the direct shortlist, so it added
> no candidates before hydration. The packet also used 3,065 of 3,072 context
> tokens. The identical-prompt controls returned the opposite answers. This
> exposes ineffective context expansion and answer variability; it demonstrates
> no hierarchy advantage or population accuracy. Those findings led to the later
> cached-history checks and candidate freeze above. The original 100-history
> controller remains stopped; the current continuation is described above. See
> [Research Log 209](10%20-%20Research%20Log/209%20-%202026-09-14%20-%20Single%20history%20design%20pilot.md).

> **Follow-up on that same cached history:** User-first packet ordering and
> bounded parent context each answered correctly on both repetitions; the old
> packet answered correctly once out of two. Median total times were 4.41 s,
> 5.30 s and 4.91 s respectively. These are repetitions of one exposed question,
> not population accuracy. Both revised packets retained 20 user sections versus
> 13 previously; parent context added no new evidence beyond user-first ordering
> on this question. Keep that hierarchy option experimental. Fourteen routing
> checks pass, and all eight streams and six judgments completed. See
> [Research Log 210](10%20-%20Research%20Log/210%20-%202026-09-14%20-%20User-first%20packet%20and%20bounded%20parent%20context.md).

> **Prior packet-budget finding:** On six frozen questions in that same history,
> user-first ordering scored 6/6 at 3,072 context tokens, while parent context
> scored 5/6. Reducing the cap to 1,024 lost a field-guide fact because a protected
> 561-token assistant reply crowded it out. The router now permits a zero-length
> protected prefix so all user evidence can be scheduled first. That repair
> passed a focused two-question check in both methods; the field-guide answer
> took 3.41 s versus 3.54 s for its matched API control. Twenty-three checks pass.
> These are one exposed official question plus manually authored design probes;
> neither 95% benchmark accuracy nor the joint latency target is established.
> The fresh complete design-set run and temporal/update coverage were subsequently
> completed in Research Log 212. See
> [Research Log 211](10%20-%20Research%20Log/211%20-%202026-09-14%20-%20Six%20question%20packet%20budget%20comparison%20and%20user%20priority%20repair.md).

> **Active target: at least 95% answer accuracy over 1M-token memories, with
> end-to-end latency close to a matched direct API conversation.** Accuracy and
> latency must pass together on the same implementation. The historical slow
> 95/100 and fast 73/100 results cannot be combined into a target pass. The
> earlier 95/100 already covered approximately 1M-token memories; it used the
> cumulative retrieval and answer-repair pipeline, without demonstrating
> API-like end-to-end query latency. See
> [Research Log 132](10%20-%20Research%20Log/132%20-%202026-09-09%20-%20Joint%201M%20accuracy%20and%20latency%20target.md).
>
> **Latest full100: reject hierarchical routing, 8/100 versus flat 84/100.**
> All 400 fresh answer streams completed; 200 logical judgments and the joint
> report replay with zero new calls. Hierarchy median total latency is 6.46 s
> versus 4.49 s for its identical-evidence API control and 3.94 s for short chat.
> Both target gates fail. All ten compiled hierarchies replay independently;
> the compiler and evaluation handoff have exited successfully.
> Saved-route tracing finds annotated support in 94/97 root shortlists,
> 62/97 root selections and only 9/97 final leaf selections. Exact hydration
> preserves those nine; the major loss occurs during attention pruning.
> Retain the flat packet while investigating the Qwen scoring policy. See
> [Research Log 181](10%20-%20Research%20Log/181%20-%202026-09-12%20-%20Full100%20hierarchy%20failure%20assessment.md).
>
> **Historical source preparation (broad execution now deferred).** Further inspection
> confirms the previously documented pooled-history conflicts in flat misses.
> The existing M+S source bank keeps the 100 histories separate. All exceed 1M
> body tokens even after excluding generated source boundaries; the minimum
> through the question day is 1,007,016. The date-independent cache, batch runner
> and exact raw hydration integration pass 181 focused checks. Larger Terra batches produced all 50 summaries
> for the same probe fragments in three calls, with 54% less summed request
> time than the eight-call format. This is ingest work, not serving latency.
> Haiku returned no completions because its provider credit balance was too low.
> Full preparation is complete: 13,812 requests cover 323,143 fragments from
> all 31,166 bodies. The live handoff has released bounded Terra execution.
> The first Terra execution stopped after 2,735 completed batches and six
> transport failures. Its continuation accepted the first fresh batch and has
> resumed the remaining never-sent batches. Six interrupted batches now have
> verified recovery outputs, with their original uncertain requests preserved.
> The new summary-body store replays
> identically; real integration preserves all 76 tested occurrence-bound spans
> through the existing hydrator. Seven failed batches now have exact source
> subdivision repairs, preserving all 153 originally valid summaries. The
> initial admission snapshot incorporates those repairs into 1,669 complete bodies and
> 17,190 sections; incomplete bodies remain excluded. All 261 tested raw spans
> across the 23 bodies touched by the repairs hydrate exactly, with zero new
> model calls. Those 1,669 bodies now have 8,650 user-led exchanges and 1,695
> cached local-Qwen attention windows. Both stages replay without model calls;
> the 14 bodies needing Qwen merges preserve all 104 tested raw spans and actual
> dates through hydration. Separate parent budgets published 1,526 body trees
> without new generation; another 147 local jobs completed all 1,669. The new
> serving adapter preserves all original attention partitions and raw addresses.
> Real occurrence rebinding preserves 940 tested raw spans, and all six
> oversized-exchange probes recover their selected original atom within budget.
> The native serving candidate retains direct atomic matches and adds context
> from attention-defined chunks. All 100 separate namespace checks pass with
> exact hydration of 400 selected spans; every incomplete namespace correctly
> rejects full-evaluation admission. All 17,146 unique summary embeddings are now
> complete. A real BGE resident probe preserves all baseline evidence and exactly
> hydrates 5,584 spans across both arms. Retrieval median is 0.105 s direct and
> 0.107 s with added chunk context, on partial histories of only 34k–91k tokens;
> this is not a 1M or API comparison. All 79 selected failed batches are now
> repaired, retaining 1,696 valid summaries. The prior body snapshot admits 5,481
> complete bodies and 57,217 sections, preserving the earlier cache exactly.
> A further 40 batches are now repaired, retaining 875 valid original summaries.
> The successor store includes those repairs and six recovered batches, admitting
> 7,121 complete bodies and 74,327 sections; 10,641 batches remain pending in that
> snapshot. All 142 recovered sections hydrate exactly through actual namespaces.
> The 74,059-summary vector population is complete, reusing 57,020 vectors and
> computing 17,039 new ones. Expanded user-spine inputs are prepared; main
> ingestion continues.
> The expanded 57,020-summary embedding stage is complete. A diagnostic using all
> 100 benchmark questions finds annotated support available for only 21 of 97
> annotated cases; direct retrieval and exact hydration retain it for 20.
> Attention context preserves that evidence but does not improve this count.
> These are partial 166k–260k-token histories, not an accuracy or latency pass.
> The old Qwen compiler stopped cleanly. The successor completed all initial
> trees; real checks across all 100 partial namespaces preserve baseline evidence
> and exactly hydrate 3,556 spans. Expanded exchange compilation stopped on two
> over-length summaries; separately recorded local recovery now admits both under
> the unchanged 128-token limit. Subsequent local compilation completed all 7,121
> prepared body exchange sets, containing 37,226 exchanges. Attention is also
> complete for those bodies, with 7,217 cached summary windows; parent compilation
> is released. Another 71 failed source batches are fully repaired, retaining
> 1,541 valid original summaries. The next admitted store contains 11,438 bodies
> and 119,144 sections, preserving the previous store. Its 118,693-summary vector
> stage is prepared and queued. The native full100 joint runner passes 43 focused
> checks, including synthetic stream/judge orchestration. Actual admission rejects
> the incomplete corpus before any model or provider construction; no native
> full100 accuracy or API-latency result is available yet.
> Another 54 failed batches are fully repaired, preserving 1,161 valid original
> summaries. A malformed JSON batch separately recovered all 21 fragments in
> three calls. The successor assembler preserves that original failure record;
> 15 recovery/admission checks pass. Real serving verification of 6,725 expanded
> body hierarchies passes across all 100 partial histories, preserving baseline
> evidence and exactly hydrating 3,450 raw spans. The fresh corpus snapshot is
> complete with 13,468 bodies and 140,256 sections, preserving the earlier store.
> Its expanded exchange inputs are complete and reuse 317 authenticated Qwen
> summaries without new generation. A zero-generation pass completed 13,316 of
> its body exchange sets; 152 bodies still need merges. Matching attention,
> parent and vector stages are queued. Parent compilation on the earlier snapshot
> completed all 7,121 body hierarchies; 663 parent summaries are verified reusable.
> Another 34 rejected batches are now repaired, preserving 718 valid summaries;
> those repairs await a later body snapshot. The full100 loader supports the new
> producers and passes 44 focused checks. Real admission still rejects the
> incomplete corpus before model construction; the joint target remains unverified.
> The 118,693-summary vector build has completed and released expanded local-Qwen
> generation. The next vector population contains 139,694 summaries. A new live
> coordinator automatically repairs newly rejected public source batches and will
> assemble the full corpus after all original requests and repairs complete.
> It authenticated 286 existing repairs and six transport recoveries; its first
> cohort contains 25 further failures. Fourteen new orchestration checks pass,
> including both repair types followed by full-store assembly. The actual final
> corpus and native full100 evaluation remain incomplete.
> That first repair cohort has now completed all 25 batches in 19 calls. The
> complete-corpus handoff is live: it waits for the six existing producers, reuses
> their finished caches, compiles the full 31,166-body store in separate model
> processes, then runs and replays the unchanged joint full100 comparison.
> Fourteen additional pipeline checks pass. No full-corpus model stage or native
> full100 answer has been released yet.
> Exchanges are now complete for all 13,468 R6 bodies: 70,218 exchanges, with 437
> new local jobs and exact atomic coverage. That process exited successfully and
> released attention preparation. The stage prepared 13,643 summary windows and
> completed all of them with summary-only local Qwen. The attention worker exited
> successfully and released expanded parent compilation.
> The first parent pass completed 12,794 of 13,468 body trees without generation;
> the remaining 674 bodies require summary merges in the bounded continuation.
> Further automatic cohorts repaired 12 batches
> in five calls, 15 batches in eight calls, and ten batches in six calls. Full source ingestion and
> the joint native evaluation remain incomplete.
> A read-only verification of the new expanded reader and exact hydration is
> queued after the R6 parent/vector jobs. It uses existing summaries and cached
> vectors across all 100 partial namespaces, with no model or answer calls.
> An early check now loads all 12,794 initial parent templates through the new
> reader and constructs one real partial namespace, exactly hydrating all four
> raw spans of a selected leaf without model calls. Full admission still rejects it.
> Further source compilation,
> repairs, complete hierarchy construction and a fresh joint full100 comparison
> remain pending. See
> [Research Log 182](10%20-%20Research%20Log/182%20-%202026-09-12%20-%20Native%20history%20summary%20cache%20and%20exact%20occurrences.md) and
> [Research Log 183](10%20-%20Research%20Log/183%20-%202026-09-12%20-%20Complete%20native%20summary%20batches%20and%20backend%20comparison.md) and
> [Research Log 184](10%20-%20Research%20Log/184%20-%202026-09-12%20-%20Native%20summary%20body%20storage%20and%20exact%20hydration.md) and
> [Research Log 185](10%20-%20Research%20Log/185%20-%202026-09-12%20-%20Dense-list%20summary%20repair%20with%20exact%20source%20subdivision.md) and
> [Research Log 186](10%20-%20Research%20Log/186%20-%202026-09-12%20-%20Repair-aware%20native%20body%20admission.md) and
> [Research Log 187](10%20-%20Research%20Log/187%20-%202026-09-12%20-%20Native%20user-spine%20exchanges%20and%20reusable%20Qwen%20attention.md) and
> [Research Log 188](10%20-%20Research%20Log/188%20-%202026-09-12%20-%20Native%20hierarchy%20compilation%20and%20atomic%20evidence%20recovery.md) and
> [Research Log 189](10%20-%20Research%20Log/189%20-%202026-09-12%20-%20Native%20namespace%20retrieval%20and%20reusable%20summary%20vectors.md) and
> [Research Log 190](10%20-%20Research%20Log/190%20-%202026-09-12%20-%20Native%20resident%20retrieval%20probe%20and%20direct%20source%20repairs.md) and
> [Research Log 191](10%20-%20Research%20Log/191%20-%202026-09-12%20-%20Native%20benchmark%20support%20availability%20and%20ingestion%20continuation.md) and
> [Research Log 192](10%20-%20Research%20Log/192%20-%202026-09-12%20-%20Explicit%20transport%20recovery%20and%20complete%20repaired%20body%20admission.md) and
> [Research Log 193](10%20-%20Research%20Log/193%20-%202026-09-12%20-%20Separate%20parent%20budgets%20reduce%20native%20Qwen%20compilation.md) and
> [Research Log 194](10%20-%20Research%20Log/194%20-%202026-09-12%20-%20Complete%20parent%20budgets%20and%20reusable%20native%20exchange%20expansion.md) and
> [Research Log 195](10%20-%20Research%20Log/195%20-%202026-09-12%20-%20Bounded%20summary%20recovery%20and%20expanded%20source%20repairs.md) and
> [Research Log 196](10%20-%20Research%20Log/196%20-%202026-09-12%20-%20Native%20full100%20runner%20and%20completed%20expanded%20attention.md) and
> [Research Log 197](10%20-%20Research%20Log/197%20-%202026-09-12%20-%20Malformed%20JSON%20recovery%20and%20expanded%20native%20admission.md) and
> [Research Log 198](10%20-%20Research%20Log/198%20-%202026-09-12%20-%20Reusable%20expansion%20of%20repaired%20native%20summaries.md) and
> [Research Log 199](10%20-%20Research%20Log/199%20-%202026-09-12%20-%20Automatic%20source%20repair%20and%20full%20corpus%20assembly.md) and
> [Research Log 200](10%20-%20Research%20Log/200%20-%202026-09-12%20-%20Full%20corpus%20compilation%20and%20joint%20evaluation%20handoff.md) and
> [Research Log 201](10%20-%20Research%20Log/201%20-%202026-09-12%20-%20Expanded%20exchanges%20complete%20and%20attention%20released.md) and
> [Research Log 202](10%20-%20Research%20Log/202%20-%202026-09-12%20-%20Real%20expanded%20reader%20verification%20handoff.md).
>
> **Hierarchy restoration and bounded Qwen traversal implemented and evaluated.**
> All ten memories have validated plans for 22,257 parents, preserving all
> 27,062 existing leaves. The resident routing/hydration integration passes
> 32 focused tests. The first real parent-summary run failed because the local
> gateway's `qwen3-8b` alias points to an unavailable `qwen3-8b-gguf` backend.
> The local fallback below has since completed all ten memories. The full100
> result above rejects the current traversal policy.
> See [Research Log 173](10%20-%20Research%20Log/173%20-%202026-09-10%20-%20Restored%20attention%20topology%20and%20bounded%20hierarchy%20routing.md).
>
> **Earlier evaluator admission checks.** Sixteen additional checks include a
> complete synthetic 400-stream execution, 200 logical judgments and zero-call
> replay. Real admission correctly rejects the unfinished parents. A fresh
> readiness request confirms that the Qwen backend is still unavailable.
> Existing cached Qwen merges can recover some parent trees without new calls;
> provenance-bound cache reuse is the next compilation step. No real quality
> or latency gain is claimed. See
> [Research Log 174](10%20-%20Research%20Log/174%20-%202026-09-10%20-%20Full100%20hierarchy%20evaluator%20and%20Qwen%20readiness.md).
>
> **Earlier cached parent recovery completed and replayed.** It preserves 450 parents
> across 489 complete source trees with zero new calls. The remaining 4,316
> sources need summary merges; no complete hierarchy is ready yet. Five new
> cache checks pass. The full local-Qwen fit test completed: the real summary
> passed validation in 12.18 seconds with a 4.70 GiB peak allocation. This is
> ingest generation, not query latency or an accuracy result. Query-time
> attention remains unchanged. See
> [Research Log 175](10%20-%20Research%20Log/175%20-%202026-09-10%20-%20Cached%20parent%20recovery%20and%20local%20Qwen%20fallback.md).
>
> **Local parent compiler added.** Nine focused checks cover completion,
> replay, interrupted execution, attribution and bounded recovery. The eight-job
> real run completed: seven outputs passed immediately, one exceeded its summary
> budget, and 12 additional parents were published. Replay preserves the same
> progress with zero model calls. Its bounded repair passed, and the subsequent
> run supplied 90 reusable local summaries before a deliberate batch-policy
> switch. See
> [Research Log 176](10%20-%20Research%20Log/176%20-%202026-09-10%20-%20Local%20Qwen%20parent%20compiler.md).
>
> **Four-summary ingest batch measured 1.82x faster on the same four jobs.**
> All outputs passed validation at 5.07 GiB peak allocation. That successor
> reused 90 local summaries and started offset 000 from 159 completed source
> trees and 345 parents. Session 65191 later stopped on the JSON error above.
> All 4,805 source trees fit the query depth limit. No new answer-accuracy or
> query-latency result is claimed. See
> [Research Log 177](10%20-%20Research%20Log/177%20-%202026-09-10%20-%20Four-summary%20local%20Qwen%20ingest%20batches.md).
> **All ten hierarchies are complete and independently replayed.** Offset 000
> contains all 499 source trees, 2,245 parents and 2,744 unchanged original leaves;
> offset 010 contains all 464 source trees, 2,161 parents and 2,625 unchanged
> original leaves; offset 020 contains all 479 source trees, 2,214 parents and
> 2,693 unchanged original leaves; offset 030 contains all 473 source trees,
> 2,278 parents and 2,751 unchanged original leaves; offset 040 contains all 481
> source trees, 2,256 parents and 2,737 unchanged original leaves; offset 050
> contains all 480 source trees, 2,232 parents and 2,712 unchanged original leaves;
> offset 060 contains all 457 source trees, 2,155 parents and 2,612 unchanged
> original leaves; offset 070 contains all 488 source trees, 2,198 parents and
> 2,686 unchanged original leaves; offset 080 contains all 488 source trees,
> 2,220 parents and 2,708 unchanged original leaves; offset 090 contains all 496
> source trees, 2,298 parents and 2,794 unchanged original leaves.
> All ten complete hierarchies reconstruct identically from saved outputs with
> zero model calls, using the same compilation method. The tenth replay ran
> after the timed evaluation exited. Both full100 target gates failed.
> A small manual check found a parent that flattened conflicting event status
> and another that dropped named entities. Exact leaf evidence is preserved;
> the impact of these summary losses on routing accuracy is still unmeasured.
> A separate recount verified every memory exceeds one million `cl100k_base`
> BPE tokens in complete raw turns, excluding chat framing; the minimum is
> 1,039,792 tokens. This check made no model calls.
>
> **Original full100 handoff stopped safely.** Session 61096 stopped when its
> compiler terminated without a complete ten-memory population. It sent no
> evaluation calls. The successor handoff above retains the same complete-parent
> admission, idle-workspace check, 400 fresh answer streams, 200 logical judgments
> and zero-provider replay. Nine handoff checks pass. See
> [Research Log 178](10%20-%20Research%20Log/178%20-%202026-09-10%20-%20Full100%20handoff%20after%20local%20parent%20compilation.md).
>
> **Four-summary Terra generation comparison completed.** All four outputs
> passed the same summary contract, but the concurrent batch took 15.57 s versus
> 13.64 s for the saved Qwen batch. This small sample does not demonstrate an
> ingest speed gain; Qwen remained the compiler backend. Replay verified the
> four responses with zero new calls. See
> [Research Log 179](10%20-%20Research%20Log/179%20-%202026-09-10%20-%20Four-parent%20Terra%20generation%20comparison.md).
>
> **Previous timed full100: control 81/100, grouped 77/100, supplemented 76/100.**
> Both candidates are rejected with the same v2 reader throughout. All600
> fresh streams completed and 151 Sol judge calls replay with zero new calls.
> Grouped median/p95 total time is 5.055/9.814 s; supplemented is 5.346/9.350 s.
> Both fail their matched API allowance and short-chat allowance. Retain the
> flat packet and v2 reader. Twenty-four focused checks pass.
>
> **Earlier implementation gap, now addressed: the serving indexes were leaf-only.**
> They contained 27,062 leaves and zero parent summaries. Qwen shaped ingest
> partitions, while those query paths used BGE leaf retrieval. Parent compilation
> and the full100 hierarchical evaluation above now address that missing work;
> completing the architecture did not establish useful routing quality.
> See [Research Log 172](10%20-%20Research%20Log/172%20-%202026-09-10%20-%20Additive%20user%20evidence%20with%20unchanged%20reader%20joint%20evaluation.md).
>
> **Previous timed full100: current control 80/100; grouped evidence/v4 reader
> 78/100. Reject the combined change.** It preserves all 2,680 selected raw
> spans and cuts median context from 2,959 to 2,454 tokens, but loses more
> answers than it gains. Candidate median/p95 total latency is 4.729/8.247 s,
> versus 4.379/11.583 s for identical-evidence API and 3.221/4.556 s for short
> chat. Accuracy and short-chat latency fail. All400 streamed responses are
> verified; uniform judging recovered from a TLS failure and replays with
> 145 hits and zero calls. Retain the prior reader and retrieval path.
> See [Research Log 171](10%20-%20Research%20Log/171%20-%202026-09-10%20-%20Conversation%20ordered%20evidence%20joint%20full100%20evaluation.md).
>
> **Earlier timed full100 comparison: as-of date cutoff 80/100, semantic
> seeds 74/100.** All ten memories contain approximately 1.04M token proxies.
> All 500 fresh Terra responses sealed before 200 logical Sol judgments using
> 136 physical calls. Source admission and judgments replayed without new calls.
> The cutoff's median/p95 total latency was 6.580/9.366 s, versus 6.039/9.734 s
> for its identical-evidence API control and 5.083/6.353 s for short API chat.
> Both methods pass the provisional matched-evidence latency allowance, and
> both fail accuracy and short-chat latency. Neither joint target gate passes.
> Qwen still receives summaries only; query-time routing uses BGE summary
> addresses over Qwen attention-partitioned leaves, followed by exact raw hydration.
> See [Research Log 166](10%20-%20Research%20Log/166%20-%202026-09-10%20-%20Full100%20as-of%20result%20and%20reader%20failure%20audit.md).
>
> **New bounded routing fix, fresh accuracy screen: 78/100 versus 76/100.**
> Relative-day user evidence reservation changes three packets, recovering the
> smoker purchase and first-client contract answers with no paired regressions.
> All100 questions were answered afresh; 97 identical packets share fresh
> responses across arms. The 103 Terra answers and 103 Sol judgments replay
> with no calls. Thirty-six focused checks pass. This batch does not measure
> serving latency or replace the earlier independently timed 80/100 result.
> The prepared native corpus remains parked. See
> [Research Log 169](10%20-%20Research%20Log/169%20-%202026-09-10%20-%20Relative%20day%20user%20evidence%20reservation.md).
>
> **Global fine user-summary retrieval failed its full100 comparison.** The
> fresh control scores 81/100, versus 72/100 at 3,072 context tokens and 71/100
> at 2,048. The candidates recover some facts but lose more correct answers;
> neither is promoted. All 300 Terra responses and 158 Sol judgments replay
> with zero calls, and sixteen focused checks pass. Retain the prior retrieval
> path. These are batched accuracy results; the latest independently timed
> full100 at that point remained 80/100 and the joint target is still unmet.
> See [Research Log 170](10%20-%20Research%20Log/170%20-%202026-09-10%20-%20Fine%20user%20summary%20retrieval%20and%20compact%20full100%20comparison.md).
>
> **Historical95 versus current80 is now paired:** 78 both correct, 17 old-only,
> 2 current-only and 3 both wrong. Twelve of the seventeen losses are multi-session
> or temporal questions. The old cumulative answer-repair result did not prove
> matched API-like latency; the current independently timed pipeline has not
> regained its accuracy. See
> [Research Log 167](10%20-%20Research%20Log/167%20-%202026-09-10%20-%20Historical95%20comparison%20and%20native%20history%20conflict%20audit.md).
>
> **Reader and corpus audit complete.** Sol on the same twenty miss packets gets
> four judge accepts, including one inconsistent judgment of essentially the
> same university answer. This is not an 84/100 result or a latency measurement.
> All100 native-history checks find every original turn in the pooled memory;
> twelve wrong answers already have every annotated support turn in their packet.
> Ninety-six packets also contain user statements from outside the question's
> original history. One verified case adds an incompatible Sophia meeting to
> the packet. Foreign membership alone does not prove a contradiction. The audit
> replays identically with no calls and does not change scores or production routes.
> The native-data continuation below prepares a separate complete corpus;
> the old scores and their source population remain preserved.
>
> **Complete same-history M+S corpus prepared and verified.** The 100 separate
> memories contain 113.27M tokens in total, at least 1.076M each and at least
> 1.019M through each question day. The construction preserves native M and
> adds absent sessions from the same record's S history. It removes the known
> cross-history Sophia conflict without pooling different question histories.
> Repeated session IDs are preserved as separate occurrences. The verifier
> checks all 52,207 occurrences, 31,166 shared text bodies, raw-source identities
> and the separate evaluation plane. Earlier grouped-ID counts are superseded.
> Twenty-four focused checks pass. No model was called and no new score is
> claimed. This corpus is prepared and parked. Following the user's request
> for concrete progress, the immediate work returns to a bounded retrieval or
> reader improvement on the existing 80/100 population before further ingestion.
> See [Research Log 168](10%20-%20Research%20Log/168%20-%202026-09-10%20-%20Complete%20native%20history%20corpus%20and%20occurrence%20verification.md).
>
> **Full100 preparation: ten complete memories, 500 control requests prepared.**
> All ten admissions verify under the common v11 method. The final memory has
> 5,624 admitted fragments, 2,794 attention leaves across 496 sources, and all
> three summary indexes, including 6,007 passage addresses. Both raw ingestion
> and compilation schedulers finished successfully. The token-accounting repair
> preserves every raw byte and distinguishes fragment counts from whole-turn
> counts. Earlier failed executions and their reservations remain preserved.
> The old r3 controls supplied the completed comparison below; their original
> answer campaign remains unexecuted.
>
> **Completed timed full100 comparison:** the frozen experiment compares
> semantic seeds with the as-of date cutoff, keeping the same reader and raw
> budgets. It requires 500 fresh requests, both API latency controls, and all
> answers sealed before judging. All ten memories supply 500 prepared requests.
> The first namespace and offsets 050–090 used live query encoding;
> offsets 010–040 reuse authenticated live-diagnostic prompts for preflight only.
> Every timed memory request still recomputes embedding, routing, and hydration.
> Thirty-one evaluation checks pass. The actual runner validated all 500 requests,
> both fresh synthetic readiness probes passed, and all 500 responses completed.
> The final report verifies the same predictions and timings across all ten
> complete memories. See [Research Log 166](10%20-%20Research%20Log/166%20-%202026-09-10%20-%20Full100%20as-of%20result%20and%20reader%20failure%20audit.md).
>
> **Automatic evaluation handoff completed the comparison.** It verified both
> completed dependencies, prepared offsets 080/090, and released the unchanged
> experiment after readiness passed. Twenty focused checks pass. Failures
> preserve reservations and stop without automatic retries. All 500 answers
> sealed before any judging. Session 13811 exited successfully; its completion
> explicitly records that the target gate failed. Preserve the frozen runtime.
> The handoff protocol is documented in
> [Research Log 163](10%20-%20Research%20Log/163%20-%202026-09-10%20-%20Automatic%20full100%20handoff%20after%20complete%20ingestion.md).
> Ninth-memory admission is recorded in
> [Research Log 164](10%20-%20Research%20Log/164%20-%202026-09-10%20-%20Ninth%20memory%20admission%20and%20final%20raw%20namespace.md).
>
> **Earlier date-aware routing diagnostic:** the separate as-of candidate reproduces
> all fifty semantic-seed controls, removes all 240 future raw spans, and adds
> 273 eligible spans. The missing April 21 tomato-planting statement now reaches
> the gardening packet. All fourteen packets without future spans are unchanged;
> the other 36 change, displacing eleven previously included eligible user spans.
> A separate relative-date hint changes only one packet and is not needed for
> the planting-statement recovery. Mixed-date sources retain older evidence;
> exact hydration still enforces the same raw budgets. Twenty-two focused checks
> pass. No answer accuracy or API-relative latency gain is claimed, and the
> frozen full100 comparison is unchanged. See
> [Research Log 159](10%20-%20Research%20Log/159%20-%202026-09-10%20-%20As-of%20summary%20routing%20and%20exact%20dated%20hydration.md).
>
> **Earlier passage-routing diagnostic:** additional addresses from exact passages within
> stored user summaries recover all three third-memory witness statements while
> retaining every previously hydrated user statement across all 30 development
> questions. All three memories have compiled passage indexes. The fresh answers
> above recover two of those questions but lose the painting recommendation,
> showing that recovered evidence alone does not ensure answer synthesis. See
> [Research Log 142](10%20-%20Research%20Log/142%20-%202026-09-09%20-%20Summary%20passage%20addresses%20and%20preserved%20user%20evidence.md).
> The full100 passage report and support-syntax recovery now pass 54 focused
> checks. The fourth raw namespace is complete; its new full audit has six
> oversized summaries and no unresolved schema failures. All six compacted in
> one summary-only Qwen call; complete source admission now reproduces all
> 5,516 fragments without calls. Its 2,751 attention-guided leaves are complete,
> along with semantic vectors and 5,849 passage addresses. All four memories now
> share the replayed version-6 admission method. See
> [Research Log 143](10%20-%20Research%20Log/143%20-%202026-09-10%20-%20Full100%20passage%20gate%20and%20support%20syntax%20recovery.md).
>
> **Completed reader experiment (preparation history):** 240 requests were prepared for all 40 questions in
> the four complete memories. The shorter policy adds precise entity/event
> qualification and direct preference-to-recommendation binding. All control
> prompts match the existing passage prompts byte for byte, with identical raw
> evidence for both readers. Eighty-two focused checks pass. The serial runner
> stopped before sending any answers when the fifth-memory ingest timed out.
> The prepared reader comparison remains unchanged. No new accuracy result is claimed.
> See [Research Log 145](10%20-%20Research%20Log/145%20-%202026-09-10%20-%20Qualified%20reader%20and%20complete%20memory%20development40%20preflight.md).
> Both local model readiness probes subsequently returned HTTP 500. A separate
> recovery stage preserves all 99 completed fifth-memory responses and lists
> 733 first attempts plus five bounded additional attempts, without clearing
> original reservations or making recovery calls. Both original processes are
> terminal. See [Research Log 146](10%20-%20Research%20Log/146%20-%202026-09-10%20-%20Fifth%20memory%20gateway%20timeout%20and%20preserved%20recovery.md).
> **Continuation:** the 240-request comparison completed with the results above.
> Fresh readiness passed again, and the offset-40 recovery executor is now
> running after timed evaluation finished. A source-scoped summary-term
> supplement recovers the named schedule while preserving prior user evidence
> in all 40 diagnostic packets; its answer accuracy remains unmeasured. All four complete
> memories replay under one version-7 admission method with unchanged atom
> bytes; 120 focused recovery, admission, gate and scheduling checks pass.
> See [Research Log 147](10%20-%20Research%20Log/147%20-%202026-09-10%20-%20Preserved%20recovery%20execution%20and%20resumed%20reader%20comparison.md).
>
> The first two memories have complete admitted atoms, attention-guided leaves,
> semantic vectors, and user-summary addresses. The third is now fully admitted
> at 5,357 atoms and 1.045M token proxies. Its 2,693 attention-guided leaves,
> semantic vectors, and user-summary addresses are complete. Nine oversized source summaries
> were compacted in two Qwen batches plus one bounded recovery call. All three
> memories now share the same replayed version-4 admission method; 28 focused
> tests pass. See
> [Research Log 140](10%20-%20Research%20Log/140%20-%202026-09-09%20-%20Multi-batch%20admission%20and%20third%20complete%20memory.md).
> The fourth raw namespace has finished all 838 requests and complete source
> admission. The fifth stopped with 99 completed responses and five unresolved
> requests; five more namespaces are prepared but unstarted. Gateway inference
> had recovered from the earlier outage before the renewed failure above.
> Original failed reservations and completed responses from that earlier outage
> remain preserved through explicit transport accounting. See
> [Research Log 138](10%20-%20Research%20Log/138%20-%202026-09-09%20-%20Explicit%20transport%20recovery%20and%20resumed%20joint%20evaluation.md).
> Full100 evaluation and untouched confirmation remain outstanding.

> **Completed real-data pilot: user-spine hierarchy and exact raw hydration.**
> Forty Terra answers across five routing variants have now been judged by Sol.
> Every variant scored **7/7 on source-derived diagnostic questions and 0/1 on
> the separate benchmark chronology question**. This is a small development
> comparison over 39 turns in three already selected conversations, not a
> full-corpus validation. User-only Qwen routing reduced input tokens by 35%
> and observed provider time by 24%, without improving accuracy. All answer and
> judge phases replay with 61 authenticated hits and zero new calls. The local
> integration suite passes 115 tests. Qwen receives summaries only. No router
> is promoted. The user's subsequent continuation instruction resolved the
> execution review block. See
> [Research Log 131](10%20-%20Research%20Log/131%20-%202026-09-09%20-%20Real%20data%20user%20spine%20answer%20and%20judge%20results.md)
> for measured results, failure analysis, and reproducible artifacts.

> **Completed conventional packet comparisons.** The fresh r9
> reduced30 control scored 12/30. Source grouping scored 8/30, BGE ordering
> 10/30, MiniLM ordering 12/30, raw-only 12/30 and verbatim highlights 11/30.
> A compact packet reduced prompt tokens by 64% but scored 10/30 and did not
> improve observed answer latency. None was promoted; answer reliability remains
> unresolved. All seven arms are sealed and replayed, with 189 tests passing. See
> [Research Log 128](10%20-%20Research%20Log/128%20-%202026-09-08%20-%20Compact%20conventional%20fast%20packet%20admission.md)
> for results, exact call accounting and the remaining failure mechanisms.
>
> **Conventional routing remains a control.** Conventional routing with the
> fast packet has not been ruled out. On the same 48 synthetic summary queries
> presented in two orders, BGE-M3 selected correctly in 96/96 presentations;
> full Qwen returned 90 valid selections and six over-cap label lists. This is
> routing-only evidence, not fast-packet answer accuracy. Keep the Qwen routers
> experimental pending broader matched evidence on the new shared hierarchy. See
> [Research Log 124](10%20-%20Research%20Log/124%20-%202026-09-08%20-%20Summary%20routing%20controls%20before%20fast%20packet%20promotion.md).

> **Earlier experimental Qwen summary hierarchy.** Attention-guided
> hierarchical chunking and summary routing are implemented as an opt-in path.
> Qwen sees only summaries and the query; exact raw sections are authenticated
> and hydrated afterward. The latest integration suite passes 282 tests. A real local
> Qwen smoke verified zero raw model inputs and exact hydration, but selected
> the wrong semantic branch. Corpus-scale compilation, timing and matched answer
> accuracy remain unmeasured. See
> [Research Log 122](10%20-%20Research%20Log/122%20-%202026-09-08%20-%20Qwen%20summary%20hierarchy%20and%20exact%20section%20hydration.md)
> for the API, model-input boundary, local Qwen smoke and separate r9 lineage.

> **Authorized temporal-reference evaluation.** The user confirmed authorization
> for the local gateways. The previously prepared successor completed 30 Terra
> answer calls and 30 Sol judge calls, scoring **12/30 versus r9's 11/30**.
> Both artifacts replayed with zero calls. The changed-prompt group improved
> from 0/3 to 1/3; unchanged prompts had two gains and two losses. This does not
> evaluate the Qwen summary hierarchy or establish a causal improvement. See
> [Research Log 123](10%20-%20Research%20Log/123%20-%202026-09-08%20-%20Authorized%20temporal%20reference%20chain%20reduced30%20result.md).

> **Fast-packet r9 handoff (separate lineage).** The additive v7/r9 packet
> construction preserves all protected parent evidence and passes its focused
> 107-test contract suite, but the sealed answer result is only **11/30** on the
> exact v6/r3 failure cohort. The corresponding 81/100 number is a conditional
> non-regression projection, not a measured full100 result; no r9 full100 run
> was promoted. Gold-open inspection found sufficient supporting material in
> 18/19 remaining incorrect packets, localizing most residuals to source/event
> binding, temporal/operator semantics, preference synthesis, and final answer
> policy rather than raw-window absence. The implementation/docs are currently
> untracked and the sealed artifacts live under ignored `eval_results/`, so the
> operational continuation and durability warning are in [Research Log
> 121](10%20-%20Research%20Log/121%20-%202026-09-08%20-%20r9%20reduced30%20execution%20handoff.md),
> with mechanism details in [Research Log
> 120](10%20-%20Research%20Log/120%20-%202026-09-08%20-%20Fact-reserved%20episodic%20packet%20repair.md).

> **The locked validation100 promotion gate now passes at 95/100; confirmation remains open.** The proof-carrying `policy-v5-r3` successor combines the frozen terminal P/R/L/G memory stack with an exhaustive numeric frontier and reducer-observable post-admission state equivalence. Its provider-free frontier closed Q28, Q53, Q67, and Q69 while correctly leaving Q14, Q40, and Q77 open. The final differential judge reused 97 authenticated Sol judgments, made exactly three new zero-retry calls, and sealed **95/100** with five remaining misses. This is still analysis-used validation evidence: the policy must be source-frozen before the disjoint confirmation200 treatment is opened, and confirmation must report both full200 and the predeclared non-exposed185 sensitivity slice. See [Research Log 101](10%20-%20Research%20Log/101%20-%202026-09-01%20-%20Terminal%20v5%2095%20percent%20campaign.md) and [Analysis 30](08%20-%20Analysis/30%20-%20Proof-carrying%20computable%20answer%20policy%202026-09-02.md). `git log --oneline` and the machine-readable artifacts remain the authority over prose.

> **Current 1M synthesis result.** On the original ten-question development concatenation, the repaired v3 Terra policy held retrieval fixed and reached 6/10 exact match, 0.901019 mean F1, and 10/10 independent Sol semantic accuracy at S1, S2, and S3. The exact replay made zero provider calls and reproduced identical artifact bytes. This is diagnostic development evidence, not a target-gate result: the population is only ten and the structured synthesis call allowed 4,096 output tokens rather than the frozen answer-stage allowance of 256. The >=95% gate remains unpassed; see Research Log 25.

> **Earlier fixed-stage development status.** The 8,000/256-token fixed-S1 path first ran on the original ten development questions. That lineage is protocol-ineligible because a sandbox-blocked root duplicated its first Terra reservation; its sealed Sol score remains a diagnostic 9/10. A later clean 100-question validation lineage completed at 56/100 and failed the target. See Research Log 27 for the development audit and Research Log 45 for the formal result.

> **Locked cumulative validation result.** All ten retrieval shards are sealed and merged at `e36b54ec...22007f`. The exact 100-call fixed-S1 Terra responder completed and replayed at `d7fc47b8...2a38cd`. Independent Sol then scored the sealed predictions **56/100 semantically**, versus 33/100 normalized exact match. The judge used exactly 100 live calls with zero retries; its zero-call replay reproduced 56/100 and byte-identical SHA `5dc56a24...ec77df`. The preregistered >=95%/100Q gate therefore failed. See Research Log 45.

> **Routed numeric EM repair.** A question-only `numeric_reduce` route held the sealed retrieval fixed and treated 32/100 questions at answer time. Thirty-two Terra compression calls yielded 19 valid fact packets and 13 baseline fallbacks; 19 answer calls changed 11 predictions. Independent Sol found three rescues and two regressions, moving 56/100 to **57/100**. Compression, answer, and judge artifacts replay byte-identically with zero calls. This is analysis-used development evidence, not retrieval recall or a new target pass; see Research Log 47.

> **Isolated routed mechanism matrix.** All six question-only mechanisms now have separate compression, answer, fallback, and changed-prediction judge budgets. Only `numeric_reduce` is positive (+1); direct extraction is -1, temporal timeline is -3, and set/state/synthesis are flat. The positive-only composer therefore admitted numeric for 32 routed questions and preserved baseline behavior for 68, sealing **57/100** with zero additional provider calls. All 18 compression/run/judge replays are byte-identical; see Research Log 48.

> **Matched retrieval matrix (closure v9 complete; broader matrix active).** The common-renderer S0-v2 control scored **53/100**. All 79 eligible representative-bridge/artifact-global retrieval artifacts and ten shards sealed; their 100-row matched answer and judge planes replayed byte-identically with zero calls. Representative bridge and artifact global each scored **52/100** (two rescues, three regressions, net -1) and are rejected from positive-only composition. Earlier 57/100 S0, 60/100 EM-fact, and 53/100 CAV observations used historical renderers and are not common-renderer marginals. The 84/100 number was an earlier `56 + 28` counterfactual ceiling, never an observed score. The >=95% gate remains unpassed; see Research Logs 49 and 54--59 and Analysis 13.

> **Matched evaluation spine (live control and gated repairs).** Decision 2 is implemented under `tools/matched_eval/`. The fresh common-renderer S0-v2 control made exactly 100 Terra answer and 100 independent Sol judge calls, replayed both planes byte-identically without calls, and scored **53/100 semantic** versus legacy S0's different-renderer 57/100. On the ten paired verdict flips, compact v3 scored 4/10, compact question-sandwich v4 scored 5/10, and gold-blind synthesis over v4 evidence plus both sealed answer hypotheses scored 3/10. The sealed candidates have a posthoc oracle union of 10/10 on that slice and 60/100 full, but no oracle-free resolver achieved it. All three promotion gates failed, so no full-100 repair campaign ran. See Analysis 13 and Research Logs 51–53.

> **Earlier CAV-ordering diagnostic.** The original 1,039,203-token retrieval artifact can be answered downstream without rebuilding a corpus or opening the store. The fixed learned-CAV extraction/reinjection router ran over its globally deduplicated question/evidence features: on the same S1 evidence membership, the CAV-steered text-order arm reached 6/10 normalized exact match and 0.843171 mean F1, versus 5/10 and 0.811906 for matched original order. One encoder API call comprised 67 internal Qwen forward batches; 22 unique feature packets produced tensor-free order receipts. This is a ten-question X/X1 ordering proxy, not CAV linking or responder-side direct activation injection. See Research Logs 35--36 for that historical diagnostic.

> **Earlier genuine CAV-link dev10 result.** CAV is the linking/fusion layer after cumulative S3 evidence, not another retrieval arm. A v2 artifact now seals the real rectangular extraction `[K,N]` and reinjection `[N,K]` links without an evidence-pair matrix or retained transformer token state. Matched S3 prompts held evidence membership and order fixed and changed only the link-guide slot. Terra exact scores were 7/10 unlinked versus 6/10 linked, but independent Sol judged both arms 10/10: the entire exact-score difference was `190` versus semantically equivalent `190 pages`. This is no causal semantic-accuracy gain on dev10; the later raw fixed-S1 locked gate failed at 56/100, and the fair Mem0 gate remains open. See Research Logs 38 and 45.

> **Current causal Hebbian result.** The old mechanism was implemented but absent from the benchmark lineage: all three Hebbian tables in the sealed combined source store had zero rows. A new causal replay learned 2,379 prior-access events into an isolated 5,978-node/51,072-edge graph, then compared sealed S0 against one budget-neutral tail-replacement arm. Only 3/10 memberships changed; `base` reached 6/10 EM and 0.836009 F1, while `h1` fell to 5/10 and 0.736009. All three admitted edges had support 1 and co-access count 1; one replaced the decisive sixth-museum excerpt with unrelated recent evidence. Exact 13-call Terra execution and 13-hit/zero-call replay passed. This is a negative ten-question development diagnostic with no independent judge, not the >=100Q gate. See Research Log 37.

> **Current EM fact-memory result.** The sealed S1 stage runs without rebuilding retrieval: S0 anchors episode selection, then S0-selected rows are excluded from the answer-time EM delta. The original facts arm tied raw at 6/10 exact match, improved F1 from 0.805372 to 0.827558, and cut prompt tokens by 39.45%. Independent Sol now judges both raw and facts 10/10; appending the complete raw tail scores 9/10 semantic because it contaminates one correct answer. Facts-only is therefore the measured operational default. See Research Logs 41 and 43.

> **EM v2 result.** The facts-only candidate completed exactly 20 Terra calls with no retries: 171 selected EM rows became 17 facts citing 15 unique rows, mean prompt length was 3,190.6 tokens, exact match rose to 7/10, and mean F1 rose to 0.914065. Independent Sol judged all ten answers correct, and the zero-call replay reproduced 10/10. Protected S0 still supplied substantial evidence, so this proves representation preservation rather than an isolated EM recall gain. See Research Log 44.

> **Code/evidence boundary.** The organized source tree is implementation epoch
> v4. Frozen validation-v3 evidence still certifies commit
> `bfa5b6daf6a5e61881ac10f0555e5d9972f9e1c2` and implementation SHA
> `452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83`.
> Because the implementation digest includes package-relative paths, v3 caches
> cannot be relabeled for v4; see `03 - Architecture/03 - Code Package Layout.md`.

## Reconciliation state (2026-08-16)

| Doc | Reconciled? | Substance of the change |
| --- | --- | --- |
| `01 - Design/00 - Original Architecture Plan.md` | ✅ | Phases 0,1,2,3,5 and live relational consolidation (4A) built; materialized cold summaries (4B) remain unbuilt |
| `01 - Design/01 - Eval Design…` | ✅ | Retired-model BUG documented; judge≠responder and token instrumentation both **resolved**, not open |
| `02 - Implementation/00 - Setup…` | ✅ | 48-test baseline → **366**; hardcoded `dim=1024` bug marked fixed; schema-v2 migration gotcha added |
| `02 - Implementation/01 - Running the Eval Harness.md` | ✅ | Rewritten for four CLI modes; benchmark data sources + cost warning; sweep is 54 configs, not 48 |
| `02 - Implementation/03 - Qwen3 Prefix Attention Lab.md` | **experimental / integrated** | Seven-layer Qwen3-8B BF16 prefix, compact persistent CAV/QK/OV artifacts, bounded dual QK/heat reads, source-aware packing, safe admission, and physical pruning; public benchmarking remains open |
| `02 - Implementation/04 - Episode-Primary Latent Evidence Fusion.md` | **experimental / resident A+B implemented** | Exact query-preserving Qwen atom rows and same-GPU atomic K-latent matched fusion now pass provider-free, CUDA, and pinned-checkpoint smoke gates; extractive rendering, router training, route-bearing v2 evaluation, and any quality claim remain open |
| `02 - Implementation/05 - As-Built Mathematical Reference.md` | **implemented / test-covered** | Exact working formulas and edge cases for BM25/TF-ISF/RRF, co-access serving, heat diffusion, causal transitions, episodic surprise/refinement, coverage energies, forced choice, and evaluation metrics |
| [02 - Implementation/08 - R7 A1 After-Union Classifier Lifecycle.md](02%20-%20Implementation/08%20-%20R7%20A1%20After-Union%20Classifier%20Lifecycle.md) | **implemented / temporal successor passes retention gate** | Documents the exact five-stage Terra R/I/U lifecycle over the 381-leaf R7 union. The 76-leaf base sieve was rejected at 25/26 target atoms; a separate gold-blind temporal fail-open voter now retains 123 leaves and passes at 26/26 atoms and 29/29 target-bearing leaves, with no answer-accuracy claim yet. |
| [02 - Implementation/09 - R7 A1b Typed-Fact Compiler Lifecycle.md](02%20-%20Implementation/09%20-%20R7%20A1b%20Typed-Fact%20Compiler%20Lifecycle.md) | **implemented / 21-call compiler materialized and replayed** | Pins independently approved classified pair `d9071196…c4da1` and effective disposition `40a584d6…a278`; 21/21 Terra checkpoints materialized 54 exact-cited facts over 45 leaves, with 78 leaves explicitly unresolved, and replayed at `9782c266…`. The compiled pair is `0da8ae97…`; the provenance-inconsistent eb84/0c1 intermediary remains rejected. |
| [02 - Implementation/10 - R7 A1 Terminal Answer Lifecycle.md](02%20-%20Implementation/10%20-%20R7%20A1%20Terminal%20Answer%20Lifecycle.md) | **implemented / provider-free 33-request v2 preflight sealed** | Three-arm factorial over identical 123-leaf membership; SHA `97596e12…`, maximum prompt/full envelope 4,232/5,000 tokens, no release or answer call. |
| [02 - Implementation/11 - R7 A1 Factorial Sol Judge Lifecycle.md](02%20-%20Implementation/11%20-%20R7%20A1%20Factorial%20Sol%20Judge%20Lifecycle.md) | **implemented / provider-free verified / awaiting sealed predictions** | Gives each A/B/C terminal-answer arm an independent exact-11 Sol preflight, release, zero-retry journal, judge replay, and score replay. Prompts contain only dated question, locked reference, and one sealed prediction; question-ID derivation replaces ordinal routing. Focused tests pass 8/8, but no production judge preflight or call exists before the answer v2 run/replay is sealed. |
| [02 - Implementation/12 - Linker-to-Terminal Repair.md](02%20-%20Implementation/12%20-%20Linker-to-Terminal%20Repair.md) | **implemented / provider-free exact-11 assay complete** | Restores exact-span typed discourse links and authenticated leaf metadata after selection/dedup, keeps Qwen/CAV affinity as routing rather than factual semantics, preserves all 123 retained handles, and fits 59 typed links plus 32 graph links inside a 6,765/8,000-token maximum envelope. A new answer/judge result is still required. |
| `03 - Architecture/00 - System Overview.md` | ✅ | Diagram and every subsystem rewritten; "there is no condensation yet" was false |
| `03 - Architecture/01 - Native Hypergraph Memory Plane.md` | **new / proposed** | Event-centric hypergraph for live QK/OV/CAV observations, with the measured pairwise graph retained as a bounded serving projection; no durable request-derived transformer token state (static model/tokenizer assets excluded) |
| `03 - Architecture/02 - Query-Conditioned Bayesian Coverage Loop.md` | **implemented / prefix measurement pending** | Primary full-width Qwen3-8B layers 0–5 with layer-5 QK/OV transport-affinity grouping; secondary compact-INI classifier; recall-safe coverage ordering and zero durable transformer state |
| `03 - Architecture/03 - Code Package Layout.md` | **new / current** | Maps responsibility packages and the objects → transformations → stateful-workflows rule, stable facades, canonical imports, size gates, and the path-sensitive validation-v3 → implementation-v4 evidence boundary |
| `04 - Reference/01 - Vocabulary.md` | ✅ | Lifecycle + retrieval terms moved out of *(planned)*; BM25/hybrid/α/`term_count`/`UsageStats`/F1/provenance added |
| `05 - Standards/00 - MC-STD-DATA-v0.md` | ✅ | Schema v2 + migration path; new normative clauses 8–10 (provenance, no destruction, migrate-in-place). Still **DRAFT** |
| `06 - Roadmaps/00 - Gap Analysis and Roadmap.md` | ✅ | Status table and tiers rewritten; Decision Point now *unblocked* but still *open*. **Partly superseded 2026-08-15** — see below |
| `06 - Roadmaps/01 - Delivering the Specified System.md` | **new** | Decay was specified in wall-clock seconds; the design intent is per-turn. The energy term therefore contributed a constant, and **every memory-arm number is void** — including the Phase 4 verdict. Carries the git evidence that the spec was wrong from commit one, and the five-stage delivery sequence |
| `00 - Theory/00 …` | — | Not touched; stable by policy (corrections only) |
| `00 - Theory/01 …` | **new draft** | Extracted-head associative memory with CAV/J-Space concepts, QK routing, OV transport, live-head pruning, and a falsification sequence; the prefix prototype now has a locked local token-saving result but no fresh recall gain |
| `00 - Theory/03 …` | **implemented / locally measured** | Schema-v9 prompt/response binding across typed memories and evidence; repeated activation, turn decay, bounded two-hop reads, and transient CAV/QK/OV weighting |
| `00 - Theory/04 - From Top-K Recall to Proof-Carrying Factual Retrieval.md` | **implemented / development-evidenced** | Separates reachability, event identity, packet sufficiency, role/time integrity, and proof scope; explains the structural scan, scalar bypass, event deduplication, scoped closure, and reproducibility repairs |
| `00 - Theory/05 - EM-LLM Episodic Discourse Closure for Diffuse Retrieval.md` | **provider-free prototype / unmeasured** | Implements source-grounded episodes, discourse obligations, bounded closure, atomic packing, and an evidence-bound Qwen prefix OV-transport change/refinement path; paper-exact EM token NLL and raw-key modularity remain unbuilt ablations |
| `04 - Reference/00 - Competitive Landscape 2026.md` | — | Not touched this pass |
| `08 - Analysis/00 - Retrieval Ablation…` | ✅ | Sweep corrected to 54 configs; a position-bin analysis was added and then **retracted the same day** — it does not replicate on the second run pair, and every bin-to-bin difference is inside noise. The aggregate ablation result stands |
| `08 - Analysis/01 - Extraction and Decay Audit` | **new** | 70.6% of memory items never reach the prompt; COLD is unreachable by construction; the default extractor is 65% spurious `Constraint`s. All free, all previously unmeasured |
| `08 - Analysis/13 - Evaluation Consolidation Decision` | **current / Decision 2 live-tested** | The thin spine is implemented. S0-v2 scored 53/100 versus legacy S0 57 on identical retrieval; compact v3/v4 and dual-answer synthesis then failed their gated flip-10 diagnostics. The remaining issue includes evidence-to-answer arbitration, not just packing. |
| [08 - Analysis/14 - Query answer joint failure taxonomy 2026-08-27.md](08%20-%20Analysis/14%20-%20Query%20answer%20joint%20failure%20taxonomy%202026-08-27.md) | **current posthoc failure analysis** | Verifies the direct-payload 71/100 and query-fact 64/100 planes before opening references, classifies their 28 joint failures, and separates deployable question-only routes from gold-informed causal labels. Log 63 records the later partition/guided outcomes and five-arm ceiling. |
| [08 - Analysis/15 - First-principles memory stack audit 2026-08-27.md](08%20-%20Analysis/15%20-%20First-principles%20memory%20stack%20audit%202026-08-27.md) | **current architecture diagnosis / composition repair active** | Reconstructs the intended prompt tick from docs, code, artifacts, and git history; shows that S1 starved S2/S3, the matched star replaced the cumulative line, decay/heat/Hebbian/slew are absent from the current tick, and CAV `X1` has no production answer-time representation consumer. Its second-pass audit identifies overcompiled obligations, repeated provider-facing receipts, lost role/time metadata, the dormant V2 direct repack and EM lane, and broken same-chunk atomic dedup. The exact D5/G2 upper preflight expands 520 sources to 1,106 calls, so compact/provider-free policy repair precedes live mapping. |
| [08 - Analysis/16 - Remaining miss memory ownership analysis 2026-08-27.md](08%20-%20Analysis/16%20-%20Remaining%20miss%20memory%20ownership%20analysis%202026-08-27.md) | **current 72/100 posthoc ownership analysis** | Classifies all 28 remaining misses by primary source-memory responsibility, CAV relation need, answer operator, and exact mapped-fact failure boundary. Ten belong to EM, seven to S0, five to artifact-global, four to Hebbian, and two to representative bridges; 23 also require CAV linking. The deeper audit finds 9 discovery/admission failures, 5 representation losses, 4 affinity/relation collisions, 8 clean operator cases, and 2 disputed benchmark rows, requiring a genuinely composed repair rather than one wider retrieval or solver knob. |
| [08 - Analysis/20 - R7 failure boundary and closure-aware semantic completion 2026-08-29.md](08%20-%20Analysis/20%20-%20R7%20failure%20boundary%20and%20closure-aware%20semantic%20completion%202026-08-29.md) | **current R7 post-result architecture decision / V5 active** | Separates candidate validation, source-local projection, global discovery, and benchmark inconsistency; preserves V3 as protected state; specifies closure-aware exact-candidate selection, a separately budgeted source-group/episode reinjection plane, and a semantic global-to-local fallback only for unresolved typed obligations. |
| [08 - Analysis/23 - Soft topical boundaries for long-chat memory retrieval - literature review 2026-08-30.md](08%20-%20Analysis/23%20-%20Soft%20topical%20boundaries%20for%20long-chat%20memory%20retrieval%20-%20literature%20review%202026-08-30.md) | **focused literature review / ablation decision** | Finds strong support for chronological event spans plus a soft multi-label topical cover, but not hard topic routing. Maps primary literature to the exact-11 26/26-visible failure, specifies union-before-exclusion, typed cross-boundary links, post-union fact compilation, and an A0--A8 matched ablation sequence. |
| [08 - Analysis/24 - Locked full100 Sol judge-score lifecycle 2026-08-30.md](08%20-%20Analysis/24%20-%20Locked%20full100%20Sol%20judge-score%20lifecycle%202026-08-30.md) | **implemented / provider-free verified** | Defines the seven-stage full100 judge lifecycle from sealed preflight through deterministic score replay. It binds exactly 100 question/reference/prediction prompts to an authenticated answer population, uses a resumable zero-retry provider journal, excludes evidence and handles, exposes no ordinal routing, and passed the 30-test combined lifecycle suite without network calls. |
| [08 - Analysis/25 - Resumable namespace construction successor 2026-08-30.md](08%20-%20Analysis/25%20-%20Resumable%20namespace%20construction%20successor%202026-08-30.md) | **production import/replay complete / compact v2 preferred** | The v1 importer proved byte equivalence but took about 50m40s, sampled 5--11 GB working set, and duplicated 2.288 GiB of sidecars in 2.288 GiB of checkpoint payloads. Compact v2 imported the same `7fe63e38…` construction in about 12m55s at 0.87--0.99 GB, reduced ten checkpoints to about 19 KiB, and replayed byte-identically in 20.281s under exact attestation pinning and fail-closed filesystem controls. This is an apparatus result, not QA accuracy. |
| [08 - Analysis/26 - Method eligibility failure attribution and apparatus cleanup 2026-09-01.md](08%20-%20Analysis/26%20-%20Method%20eligibility%20failure%20attribution%20and%20apparatus%20cleanup%202026-09-01.md) | **current evaluation-contract analysis / initial cleanup measured** | Separates method ineligibility, non-attempt, retrieval-stage loss, packing/admission loss, and downstream answer failure; defines a common outcome ledger and records the first behavior-preserving cleanup, whose fixed provider-free slice fell from 60.19 to 23.71 seconds without changing sealed behavior. |
| [08 - Analysis/27 - Provider boundary and semantic storage cleanup 2026-09-01.md](08%20-%20Analysis/27%20-%20Provider%20boundary%20and%20semantic%20storage%20cleanup%202026-09-01.md) | **implemented / provider-free / compatibility-preserving** | Finds no gold, raw locator, path, or model-state leak in the active provider route, then introduces an explicit compact-v2 provider schema, call-scoped trace caches, zero-reuse classifier-cache removal, and one-population semantic trees. The sealed-ten full chat falls 29.32% and 1,024-cell descendant tuple storage falls 95.2%; historical compact v1 remains exact. |
| [08 - Analysis/31 - Minimal-compute hot-memory retrieval architecture 2026-09-05.md](08%20-%20Analysis/31%20-%20Minimal-compute%20hot-memory%20retrieval%20architecture%202026-09-05.md) | **adaptive full100 measured / v8 binary packer sealed / 93 of 100 source-complete / 70 of 100 semantic** | Provider-free source-balanced admission raises complete-source packets from 72 to 93 and semantic accuracy from 66 to 70. Sealed v8 preserves all 100 provider-bound payloads byte-for-byte while cutting packing mean 4.295x and p95 5.113x against the non-contemporaneous same-machine v7 artifact; this is packing-only, not end-to-end provider latency. |
| [08 - Analysis/32 - Incremental conversational association graph overlay 2026-09-06.md](08%20-%20Analysis/32%20-%20Incremental%20conversational%20association%20graph%20overlay%202026-09-06.md) | **persistent provider-free T1g + v4 graph read measured / 99 of 100 source-complete** | Defines the HippoRAG/Graphiti-shaped additive address plane, then records its append-at-ingest journal, immutable phrase/story deltas, bounded bootstrap, restart-safe hydration, and authenticated resident story query. The v4 assay reaches 99/100 strict source coverage with no retrieval-time model calls; production `build_context` composition remains open. |
| [08 - Analysis/33 - User-led conversational envelopes and additive recall overlay 2026-09-07.md](08%20-%20Analysis/33%20-%20User-led%20conversational%20envelopes%20and%20additive%20recall%20overlay%202026-09-07.md) | **schema-v17 envelope lifecycle + provider-free architecture pilot / opt-in** | Makes each user turn the stable lead of an exact conversational envelope, attaches machine turns without promoting their authority, and keeps surprise/coherence structure as links over preserved microepisodes. The sealed eight-question pilot reaches complete target evidence with fewer tokens, but is not a full100, 1M, or judged answer result. |
| `10 - Research Log/02 - 2026-08-16 - Qwen3 prefix CAV gate.md` | **new measurement** | Layers 0–5 passed held-out accuracy, bootstrap stability, and random-label controls for two project-relevant CAVs; layer 5 selected for the first live-memory prototype |
| `10 - Research Log/03 - 2026-08-16 - Live Qwen head memory smokes.md` | **new measurement** | Layer-5 CAV entry reached 0.750/0.875; calibrated layer-1 head/direction association reached 1.000 R@1/R@3 on four development links; fresh blind replication is required |
| `10 - Research Log/04 - 2026-08-16 - Safe associative memory confirmation.md` | **new confirmation** | On a locked fresh six-family split, safe CAV/QK arms preserved 83.3% hybrid recall while reducing prompt tokens by 1.3–2.7%; degree-two pruning removed 392/1,204 edges without a recall loss; no fresh recall gain was observed |
| `10 - Research Log/05 - 2026-08-16 - Source heat diffusion development.md` | **new development replay** | Two-hop dual allocation reserves one ranked-QK slot and one heat slot; degree-two replay preserved local recall while reducing selected text by 5.1–16.3%; pure heat lost the one development recovery, and no fresh recall gain is claimed |
| `10 - Research Log/06 - 2026-08-16 - 95 percent long-chat target.md` | **active target** | Locks 500 cleaned LongMemEval questions into 200/100/200 partitions and defines ≥95% judge accuracy under an 8k prompt ceiling as the hard gate |
| `10 - Research Log/08 - 2026-08-16 - Real Qwen consolidation path.md` | **operational smoke** | The real seven-layer BF16 prefix updated six schema-v8 edges from four packed pointers in 0.75 s after a 12.92 s startup load, retaining zero prompt/activation bytes; recall effect remains unmeasured |
| `10 - Research Log/09 - 2026-08-16 - Causal binding reaches 97.4 percent evidence recall.md` | **new development replay** | Four-arm chronological replay: original 35/39, packing-only 36/39, rank graph 37/39, Qwen graph 38/39 with no losses and zero retained transformer-state bytes; answer-stage evaluation remains open |
| `10 - Research Log/15 - 2026-08-18 - Policy-locked 1M-context answer pilot.md` | **development pilot artifact** | Campaign artifacts report ten of ten positive judge verdicts; structural consistency is verified, but provider/judge execution and factual correctness are not independently authenticated; mean reported responder prompt was 2,342 tokens from 1,039,203 transcript tokens |
| `10 - Research Log/16 - 2026-08-18 - V3 retrieval freeze and validation campaign.md` | **frozen development treatment** | Final no-provider replay reached 100% source and scored answer-value coverage at a mean 1,986 returned tokens; exact cache receipts, prompt-proxy identity, a 100-question campaign plan, and the corrected Mem0 protocol are frozen, but no held-out provider calls have run |
| `10 - Research Log/17 - 2026-08-18 - Locked treatment handoff and discourse closure frontier.md` | **operational handoff / incomplete goal** | Consolidates the treatment, ten prepared cache shards, hard invariants, controlled Mem0 tooling, current test evidence, explicit NO-GO boundaries, and the proposed general-purpose Grounded Discourse Closure RAG design |
| `10 - Research Log/18 - 2026-08-18 - Validation v3 provider-free retrieval audit.md` | **100-question provider-free audit / retrieval gate failed** | Exact frozen-v3 replay across all ten validation shards: 87.6% mean evidence-source recall, 82% all-source recovery, zero post-coverage closures, zero provider calls, unchanged cache hashes, and an explicit development-to-validation generalization gap; answer accuracy remains unmeasured |
| `10 - Research Log/21 - 2026-08-21 - Retrieval nesting and fresh 1M episode-primary test.md` | **1M functional ablation / retrieval regression** | A fresh validation-offset-0 `episode_primary` route completed end to end but replaced the v3 authority and fell to 3/10 literal reachability; this is not the original concatenated-memory control |
| `10 - Research Log/22 - 2026-08-21 - Recall-guarded cumulative retrieval.md` | **measured 1M development ladder** | The original 1,039,203-token development concatenation ran through four strictly nested provider-free stages: S0 recovered every labeled source and 5/10 literal answers; S1 improved mean evidence F1 by 4.62%, while S2 and S3 added no further scored gain under the cap |
| `10 - Research Log/23 - 2026-08-21 - Episodic evidence scoring and synthesis.md` | **measured local synthesis / negative answer result** | A pinned Qwen3-0.6B inspected all 176 episodic additions and made 12 unique S1-S3 answer calls; its historical raw-p(A) answerability proxies found no useful S2 addition, while every stage scored 0/10 exact match and 0.010227 mean F1, with no independent judge or calibrated-density claim |
| `10 - Research Log/24 - 2026-08-21 - LiteLLM Terra episodic synthesis and rescoring.md` | **measured provider synthesis / improved development answer result** | A strict, 12-call checkpointed Terra arm held retrieval fixed, labeled all 176 additions, and raised S1 to 5/10 exact match and 0.718433 F1; S2/S3 reached 4/10 and 0.706806, all five S2-only additions were labeled irrelevant/none, and no independent judge or held-out claim is made |
| `10 - Research Log/25 - 2026-08-22 - Independent Sol judge and v3 synthesis repair.md` | **independently judged development diagnostic / formal gate still open** | A separate Sol path reconstructed and recounted every sealed Terra prompt, judged v2 at 9/10, and judged the runtime-gold-blind v3 synthesis repair at 10/10 for S1-S3; byte-identical zero-call replay passed, but the ten-question population and 4,096-token synthesis output allowance make this ineligible for the locked answer-stage gate, and Mem0 remains unrun |
| `10 - Research Log/26 - 2026-08-22 - Fixed-stage S1 and locked 100Q campaign.md` | **reproducible launch surface / formal gate still open** | Locks S1 under the original 8,000/256 answer budget, adds ten independently sealed cumulative retrieval shards, an independent Sol >=95%/100Q gate, and a fail-closed schema-v3 Mem0 comparison boundary; real preflights pass, but the GPU/provider campaigns and fair Mem0 arm remain unrun |
| `10 - Research Log/27 - 2026-08-22 - Fixed-stage S1 LiteLLM development diagnostic.md` | **operational development diagnostic / protocol-ineligible** | The completed root internally records 10 fixed-S1 Terra and 10 independent Sol physical calls with zero SDK retries and exact offline replay, but the sandbox-blocked root duplicates the first Terra reservation, yielding 11 reservations for 10 unique calls across the lineage; Sol's sealed 9/10 remains `insufficient_population`, its approximate-answer negative is likely an adjudication false negative without a preregistered appeal, and neither the formal 100Q result nor Mem0 exists yet |
| `10 - Research Log/28 - 2026-08-22 - First locked validation shard seal.md` | **first locked validation retrieval shard / formal gate still open** | Offset 0 sealed all ten nested S0--S3 questions under the 7,000-context/8,000-prompt caps with zero provider calls and zero retained request-token state; canonical replay and an independent receipt/store audit passed, offsets 10--90 passed preflight, offset 10 is running, and the 100Q merge, Terra/Sol score, and Mem0 arm remain incomplete |
| `10 - Research Log/29 - 2026-08-22 - Locked validation offset 10 seal.md` | **second locked validation retrieval shard / formal gate still open** | Offset 10 sealed and replayed all ten provider-free ladders; S1 admitted 177 evidence rows, while S2 and S3 admitted none because the frozen context budget was exhausted. Offset 20 is running, and the 100Q merge, Terra/Sol gate, and Mem0 comparison remain incomplete |
| `10 - Research Log/30 - 2026-08-22 - Locked validation offset 20 seal.md` | **third locked validation retrieval shard / formal gate still open** | Offset 20 sealed and replayed all ten provider-free ladders; S1 admitted 178 evidence rows while S2/S3 were budget-inert, and `q025` sealed at 574.6 seconds alongside live paging-consistent OS observations. Offset 30 is running; the 100Q merge, Terra/Sol/>=95% gate, and Mem0 comparison remain incomplete |
| `10 - Research Log/31 - 2026-08-22 - Locked validation offset 30 S2 seal.md` | **fourth locked validation retrieval shard / formal gate still open** | Offset 30 sealed and replayed all ten provider-free ladders; S1 admitted 163 rows, and after S2 was inert throughout offsets 10 and 20, `q035` added four S2 rows under the frozen cap without implying an answer-accuracy gain. Offset 40 is running; the 100Q merge, Terra/Sol/>=95% gate, and Mem0 comparison remain incomplete |
| `10 - Research Log/32 - 2026-08-22 - Locked validation offset 40 S2 seal.md` | **fifth locked validation retrieval shard / formal gate still open** | Offset 40 sealed and replayed all ten provider-free ladders; S1 admitted 172 rows, `q047` appended four S2 rows, and S3 admitted none. The append proves bounded retrieval admission, not relevance or accuracy. Offset 50 is running; the 100Q merge, Terra/Sol/>=95% gate, and Mem0 comparison remain incomplete |
| `10 - Research Log/33 - 2026-08-22 - Locked validation offset 50 seal.md` | **sixth complete validation retrieval shard / formal gate still open** | Offset 50 sealed and replayed all ten provider-free ladders; S1 admitted 167 rows while S2/S3 admitted none. `q058` is a sealed timing outlier only, not causal or quality evidence. The later offset-60 completion is recorded in Log 39; the 100Q merge, fixed-S1 Terra/Sol/>=95% gate, and Mem0 comparison remain incomplete |
| [10 - Research Log/34 - 2026-08-22 - Cumulative apparatus performance diagnosis.md](10%20-%20Research%20Log/34%20-%202026-08-22%20-%20Cumulative%20apparatus%20performance%20diagnosis.md) | **docs-only performance diagnosis / historical six-shard boundary** | Separates sealed evidence from candidate cost mechanisms and unproven q058 causality: 963 snapshots for 480 sources, 12,891,681 source-row hash visits, S2 additions on 4/60 questions, and S3 additions on 0/60. This was the evidence boundary when written; Log 39 records the later offset-60 completion under an exact frozen-source resume |
| [10 - Research Log/35 - 2026-08-22 - Fast 1M retrieval and synthesis path.md](10%20-%20Research%20Log/35%20-%202026-08-22%20-%20Fast%201M%20retrieval%20and%20synthesis%20path.md) | **active streamlined design / CAV execution superseded by Log 36** | Defines S0--S3, the remembered Pure Attention CAV extraction/reinjection layer, and the separate LLM synthesis layer; explains why certification builds stopped; records a fresh text-only S1 result at 5/10 EM, 0.786009 F1, and 9/10 Sol; and reduces matched S0--S3 evaluation to 22 unique prompts. Log 36 contains the later measured CAV-ordering execution |
| [10 - Research Log/36 - 2026-08-22 - Fast CAV reinjection ablation and runtime refactor.md](10%20-%20Research%20Log/36%20-%202026-08-22%20-%20Fast%20CAV%20reinjection%20ablation%20and%20runtime%20refactor.md) | **measured 10Q downstream CAV-ordering diagnostic / formal gate still open** | Replays the sealed original-1M retrieval without rebuilding stores; one Qwen feature API call (67 internal batches) and 22 router calls produced tensor-free same-evidence orders, and the S1 CAV treatment reached 6/10 EM and 0.843171 F1 versus 5/10 and 0.811906 for matched original order. This is text ordering, not responder activation injection; the first request-only root produced no result, run2 completed 30 calls plus zero-call replay, and the locked >=100Q independently judged goal remains unpassed |
| [10 - Research Log/37 - 2026-08-22 - Causal Hebbian H1 arm restoration.md](10%20-%20Research%20Log/37%20-%202026-08-22%20-%20Causal%20Hebbian%20H1%20arm%20restoration.md) | **measured 10Q causal Hebbian negative diagnostic / formal gate still open** | Restores the previously unwired arm with 2,379 causal prior-access events and a matched S0/H1 prompt path. Three budget-neutral replacements reduced mean prompt proxy by 12.4 but changed 6/10 EM and 0.836009 F1 to 5/10 and 0.736009; all selected edges were single-support/single-coaccess, and one discarded decisive evidence. Thirteen Terra calls and exact zero-call replay passed; there is no independent judge or >=100Q claim |
| [10 - Research Log/38 - 2026-08-23 - Genuine CAV links and matched semantic result.md](10%20-%20Research%20Log/38%20-%202026-08-23%20-%20Genuine%20CAV%20links%20and%20matched%20semantic%20result.md) | **measured 10Q genuine CAV-link semantic tie / formal gate still open** | Places CAV correctly after S3 as latent linking/fusion, seals the two rectangular link passes with zero retained token state, and compares identical canonical S3 evidence with and without the rank-only link guide. Exact EM was 7/10 versus 6/10 because of `190` versus `190 pages`; independent Sol judged both 10/10, so dev10 shows no causal semantic-accuracy gain. The locked >=100Q and fair Mem0 gates remain incomplete |
| [10 - Research Log/39 - 2026-08-23 - Locked validation offset 60 seal and frozen-source resume.md](10%20-%20Research%20Log/39%20-%202026-08-23%20-%20Locked%20validation%20offset%2060%20seal%20and%20frozen-source%20resume.md) | **seventh locked validation retrieval shard / formal gate still open** | Offset 60 sealed and replayed all ten provider-free ladders at root SHA `34f27da4...f3188`; S1 added 173 rows, S2 added four on `q066`, and S3 added none. An exact historical source snapshot reproduced the original implementation hash and reused `q060`/`q061` without weakening checkpoint checks. The campaign now covers 70/100 questions, offset 70 is running, and responder/judge/Mem0 gates remain open |
| [10 - Research Log/41 - 2026-08-25 - Post-selection EM fact memory.md](10%20-%20Research%20Log/41%20-%202026-08-25%20-%20Post-selection%20EM%20fact%20memory.md) | **measured 10Q post-selection EM representation diagnostic / formal gate still open** | Keeps S0 as the episode-selection anchor, excludes it only after sealed S1 selection, and converts the 171-row EM delta into 19 exact-quote-cited facts. Facts-only tied raw at 6/10 EM, improved F1 from 0.805372 to 0.827558, and reduced mean prompt tokens by 39.45%; facts plus the complete raw tail fell to 5/10 and 0.755521. Forty journals, replay, score, and 93 regression tests passed; no independent judge or locked-100 claim is made. |
| [10 - Research Log/42 - 2026-08-26 - Streamlined EM v2 and independent semantic scoring.md](10%20-%20Research%20Log/42%20-%202026-08-26%20-%20Streamlined%20EM%20v2%20and%20independent%20semantic%20scoring.md) | **implementation and preflight checkpoint / execution superseded by Log 43** | Adds a v1-compatible, replay-safe Sol semantic judge and an opt-in two-call-per-question v2 facts-only policy with post-selection cited fallback, bounded reinjection, and answer-shape guidance. It sealed the exact 15-call v1-judge and 20-call v2 populations before authorization; their completed results and the later 100Q retrieval merge are recorded in Log 43. |
| [10 - Research Log/43 - 2026-08-26 - EM v2 result and locked 100Q retrieval merge.md](10%20-%20Research%20Log/43%20-%202026-08-26%20-%20EM%20v2%20result%20and%20locked%20100Q%20retrieval%20merge.md) | **measured development and retrieval checkpoint / execution superseded by Log 44** | V1 raw and facts both reach 10/10 independent Sol while full raw reinjection falls to 9/10; v2 facts-only improves lexical scoring to 7/10 and 0.914065 F1 using 17 facts over 15/171 EM rows. All ten validation shards merge at `e36b54ec...22007f`; S1 moves literal reachability 48→50/100 and S2/S3 add no measured gain. Log 44 records the later v2 judge and 100Q Terra answers. |
| [10 - Research Log/44 - 2026-08-26 - V2 semantic confirmation and locked 100Q answers.md](10%20-%20Research%20Log/44%20-%202026-08-26%20-%20V2%20semantic%20confirmation%20and%20locked%20100Q%20answers.md) | **v2 semantic confirmation plus sealed 100Q answers / judge superseded by Log 45** | Independent Sol confirms v2 facts-only at 10/10 with a zero-call replay. The exact 100-call fixed-S1 responder completes and replays at `d7fc47b8...2a38cd`; local scoring is 33/100 exact and 0.447494 F1. Log 45 records the later formal Sol result. |
| [10 - Research Log/45 - 2026-08-26 - Locked 100Q semantic gate result.md](10%20-%20Research%20Log/45%20-%202026-08-26%20-%20Locked%20100Q%20semantic%20gate%20result.md) | **formal locked-100 semantic result / target failed** | Exactly 100 independent Sol calls scored the sealed fixed-S1 Terra predictions 56/100; zero-retry journals and a byte-identical zero-call replay passed. Multi-session, temporal, preference, and abstention failures dominate. The >=95% gate failed, the population is now analysis-used, and a tuned system requires untouched confirmation. |
| [10 - Research Log/47 - 2026-08-26 - Routed numeric EM repair result.md](10%20-%20Research%20Log/47%20-%202026-08-26%20-%20Routed%20numeric%20EM%20repair%20result.md) | **measured answer-time numeric repair / positive but small development marginal** | Holds sealed S1 retrieval fixed, derives EM after selection, and routes 32 questions from question text alone. Thirty-two compression plus 19 answer Terra calls produced 11 changed predictions; Sol measured three rescues and two regressions, moving 56/100 to 57/100. Thirteen empty/invalid fact packets preserved the baseline, exact replay passed, and per-method protected budgets remain required. |
| [10 - Research Log/48 - 2026-08-26 - Isolated routed mechanism matrix.md](10%20-%20Research%20Log/48%20-%202026-08-26%20-%20Isolated%20routed%20mechanism%20matrix.md) | **complete six-arm answer-time ablation / sealed positive-only composition** | Tests every route with an isolated budget over the same sealed baseline. Numeric is +1 and admitted; direct is -1, timeline is -3, and set/state/synthesis are zero and rejected. The six-row ledger carries exact calls, fallbacks, paired outcomes, hashes, and byte-identical replays; provider-free composition seals the accepted numeric route at 57/100. |
| [10 - Research Log/49 - 2026-08-26 - Matched retrieval mechanism matrix roadmap.md](10%20-%20Research%20Log/49%20-%202026-08-26%20-%20Matched%20retrieval%20mechanism%20matrix%20roadmap.md) | **active matrix / bridge and global cells complete** | The common-renderer S0-v2 control is 53/100; v9 representative and global descendants are each 52/100 and rejected. Historical-renderer EM/CAV observations, S0-seeded Hebbian, and any genuinely positive composition remain separate. The 263-target registry preserves primary responsibility, alternate reachability, and pre-dedup discovery credit. |
| [10 - Research Log/50 - 2026-08-26 - S0 EM and CAV isolated retrieval results.md](10%20-%20Research%20Log/50%20-%202026-08-26%20-%20S0%20EM%20and%20CAV%20isolated%20retrieval%20results.md) | **sealed historical renderer observations** | Reproduces S0 57, post-selection EM facts 60, and membership-invariant CAV links 53 with changed-only Sol judging. The prompt templates differ, so these are useful mechanism observations but not yet common-renderer causal marginals. |
| [10 - Research Log/51 - 2026-08-26 - Matched evaluation spine v2 implementation.md](10%20-%20Research%20Log/51%20-%202026-08-26%20-%20Matched%20evaluation%20spine%20v2%20implementation.md) | **Decision 2 implementation / superseded live checkpoint** | Adds the immutable snapshot, typed deltas, protected-budget runner, common renderer, verified runtime/score join, legacy quarantine, and narrow S0 adapter. Historical 57/60/53 replay provider-free; Log 52 records the later live common-renderer control. |
| [10 - Research Log/52 - 2026-08-26 - Matched S0-v2 live control result.md](10%20-%20Research%20Log/52%20-%202026-08-26%20-%20Matched%20S0-v2%20live%20control%20result.md) | **measured common-renderer control / renderer regression** | Exactly 100 Terra answers and 100 Sol judgments replay byte-identically. Common S0-v2 scores 53/100, exact 27, mean F1 0.410760. Identical retrieval receipts and changed prompts isolate the legacy 57→53 loss to rendering/answering; Log 53 records the later failed v3/v4 repair gates. |
| [10 - Research Log/53 - 2026-08-26 - Compact renderer and dual-answer synthesis diagnostics.md](10%20-%20Research%20Log/53%20-%202026-08-26%20-%20Compact%20renderer%20and%20dual-answer%20synthesis%20diagnostics.md) | **completed gated diagnostics / no full promotion** | On the ten legacy/v2 verdict flips, compact v3 scored 4/10, v4 scored 5/10, and gold-blind synthesis over v4 evidence plus both sealed predictions scored 3/10. The posthoc candidate union is 10/10 on the slice and 60/100 full but is not oracle-free; all gates failed, all replays were exact, and no full-100 calls ran. |
| [10 - Research Log/54 - 2026-08-27 - Closure eligibility repair and v4 preflight.md](10%20-%20Research%20Log/54%20-%202026-08-27%20-%20Closure%20eligibility%20repair%20and%20v4%20preflight.md) | **superseded provider-free preflight / eligibility repair retained** | Replaces the superseded 57-question v3 closure eligibility with a gold-blind 79-question predicate based on temporal-metadata or complete-frontier demand. The repair restores four relative-time questions and nine artifact-global-owned sources; v4 then failed closed on a timing-contaminated S0 receipt before publishing a question artifact and is superseded by Log 56. |
| [10 - Research Log/55 - 2026-08-27 - Single-process matched evaluation pipeline.md](10%20-%20Research%20Log/55%20-%202026-08-27%20-%20Single-process%20matched%20evaluation%20pipeline.md) | **implemented orchestration-speed refactor / no accuracy claim** | Loads the immutable population once, reuses one posthoc judge plan, preserves separate exact Terra/Sol authorizations and byte-identical artifacts, and reverifies retrieval after judge replay. The expanded focused suite passes 44/44; underlying retrieval and provider latency are unchanged. |
| [10 - Research Log/56 - 2026-08-27 - Stable S0 closure protocol v6.md](10%20-%20Research%20Log/56%20-%202026-08-27%20-%20Stable%20S0%20closure%20protocol%20v6.md) | **superseded provider-free protocol / no question artifacts** | Diagnoses the historical selector-report hash as timing-contaminated, records the full fresh report and exact two-path normalization, and enforces bilateral stable S0 plus receipt-linkage checks. V6 passed that gate, then failed before publication on frozen-receipt serialization; no accuracy claim follows. |
| [10 - Research Log/57 - 2026-08-27 - Closure v8 source identity and canary.md](10%20-%20Research%20Log/57%20-%202026-08-27%20-%20Closure%20v8%20source%20identity%20and%20canary.md) | **superseded v8 canary / two question artifacts only** | Records the v7 attribution repair and v8 label-free cross-method source identity. V8 published ordinals 3 and 4, then failed closed before ordinal 5 publication because an authoritative scalar bypass truthfully had no nested scorer timer. It produced no shard, merged generation, answer, judge, or accuracy result. |
| [10 - Research Log/58 - 2026-08-27 - Closure v9 scalar-bypass normalization.md](10%20-%20Research%20Log/58%20-%202026-08-27%20-%20Closure%20v9%20scalar-bypass%20normalization.md) | **completed v9 retrieval protocol** | Distinguishes invoked scoring from the exact identity-only scalar bypass. All ten shards and 79/79 eligible artifacts now seal and merge; Log 59 records the matched answer and judge outcome. |
| [10 - Research Log/59 - 2026-08-27 - Independent closure v9 matched outcome.md](10%20-%20Research%20Log/59%20-%202026-08-27%20-%20Independent%20closure%20v9%20matched%20outcome.md) | **complete matched closure result / both arms rejected** | Exact retrieval, target, answer, judge, runtime, and score seals replay. Representative bridge and artifact global each score 52/100 against S0-v2's 53/100, so neither enters positive-only composition. It also records the target funnel, question-72 judge noise, and why 84/100 was a counterfactual ceiling rather than an observed result. |
| [10 - Research Log/60 - 2026-08-27 - Provider-free partition scan construction diagnostic.md](10%20-%20Research%20Log/60%20-%202026-08-27%20-%20Provider-free%20partition%20scan%20construction%20diagnostic.md) | **sealed provider-free construction diagnostic / no answer-accuracy claim** | Corrects closure accounting to the eligible 162-source denominator, where S0 and raw global both hit 135 and closure adds 0/27 novel sources. A full-store, gold-blind top-four partition scan then reaches 19/27 missing sources and admits 14/27 under its own 2,048-token budget. Source-ID reach does not prove the chosen excerpt contains the answer fact; no answer or judge calls ran. |
| [10 - Research Log/61 - 2026-08-27 - Partition allocation and matched EM fact gate.md](10%20-%20Research%20Log/61%20-%202026-08-27%20-%20Partition%20allocation%20and%20matched%20EM%20fact%20gate.md) | **sealed construction repair plus matched answer-time result** | Partition-balanced v2 preserves 19/27 candidate reach and raises selection/admission from 14/27 to 19/27, but source reach is not answer accuracy. A separate parent-preserving EM fact gate makes 25 Terra and 14 changed-only Sol calls and scores 54/100 versus S0-v2's 53, with one rescue and zero regressions. Both planes replay exactly; >=95% remains unpassed. |
| [10 - Research Log/62 - 2026-08-27 - Query expansion construction and packing audit.md](10%20-%20Research%20Log/62%20-%202026-08-27%20-%20Query%20expansion%20construction%20and%20packing%20audit.md) | **sealed query-planning construction audit / no answer score** | One hundred gold-blind Terra query plans produce 5,510 candidates, 3,926 selected rows, and 2,671 admitted exact spans. Query expansion reaches 20/27 missing-source candidates but admits 14/27; its admitted union with partition v2 reaches 156/162 eligible and 180/188 total source targets. That exceeds 95% structural source coverage, not QA accuracy, and identifies source-balanced packing plus query-routed exhaustive scans as the next construction layers. |
| [10 - Research Log/63 - 2026-08-27 - Query-era matched answer campaign.md](10%20-%20Research%20Log/63%20-%202026-08-27%20-%20Query-era%20matched%20answer%20campaign.md) | **completed matched query-era campaign / target still failed** | Direct query payload scores 71/100, query facts 64, partition payload 59, and guided payload 58 against S0-v2 at 53. The five-arm oracle is only 74; a one-pass structured operator regressed to 67 and exposed an abstention-before-mapping/parser failure. Exact replays and calls/hashes are recorded; >=95% is unmet and Mem0 remains open. |
| [10 - Research Log/64 - 2026-08-27 - First-principles composition checkpoint.md](10%20-%20Research%20Log/64%20-%202026-08-27%20-%20First-principles%20composition%20checkpoint.md) | **isolated evidence-map solver rejected / source-history composition active** | The mandatory evidence map yielded 268 validated items, but its isolated solver changed eleven predictions for one rescue and five regressions, scoring 67/100 versus the direct parent's 71. The map is retained for routing; combined replacement now requires an admitted post-map source-history fact, and the 549-call upper construction is held behind a provider-free policy sweep. |
| [10 - Research Log/65 - 2026-08-27 - Adaptive map result and tail execution incident.md](10%20-%20Research%20Log/65%20-%202026-08-27%20-%20Adaptive%20map%20result%20and%20tail%20execution%20incident.md) | **72/100 adaptive result / first tail execution abandoned** | Direct and Partition source-map fact lanes each rescue q40 without regression, producing the replay-verified 72/100 best result. A provider-free 79-source/80-window tail preflight sealed under the 8k envelope, but its sandbox-denied live attempt left four terminal request reservations. An interrupted cleanup removed those request files, so wave 1 is permanently fenced as protocol-invalid; a fresh recovery campaign must exclude all four exact identities rather than retry them. |
| [10 - Research Log/66 - 2026-08-27 - Tail recovery result and typed composition pivot.md](10%20-%20Research%20Log/66%20-%202026-08-27%20-%20Tail%20recovery%20result%20and%20typed%20composition%20pivot.md) | **Recovery replayed / tail not promoted** | A fresh 80-call recovery excludes the four abandoned identities and replays byte-identically with zero wave-1 reuse. It yields 22 accepted facts, but posthoc inspection finds only one credible new structured gain among the 28 misses; the next arm therefore composes bounded full-store slot closure with typed operators and the protected 72/100 parent. |
| [10 - Research Log/76 - 2026-08-28 - Reduced exact-ten treatment matrix and streamed memory control.md](10%20-%20Research%20Log/76%20-%202026-08-28%20-%20Reduced%20exact-ten%20treatment%20matrix%20and%20streamed%20memory%20control.md) | **sealed provider-free structural matrix and streamed control / no answer score** | Seven independent approximately 1M-token namespaces contain 7,208,302 indexed tokens. Child-per-namespace replay reduces simultaneous index residency by 85.66% yet reproduces all 7 namespace, 10 question, and 70 method receipts plus canonical bytes exactly. Coverage and CAV provenance activate but regress when used as replacement signals; separate protected lanes, post-selection deduplication, and terminal fair repacking precede any provider promotion. |
| [10 - Research Log/77 - 2026-08-28 - Reduced callback-union answer diagnostic.md](10%20-%20Research%20Log/77%20-%202026-08-28%20-%20Reduced%20callback-union%20answer%20diagnostic.md) | **sealed post-hoc exact-ten Terra/Sol diagnostic / 2 of 10** | Reuses all four fact arms' independently callback-selected rows without reopening a 1M store, deduplicates only afterward, and retains every unique row under 8k. Ten Terra plus ten Sol calls recover q31 and q81, while the remaining errors localize to callback retrieval, answer-bearing span identification, insufficiency, and temporal linking/noise. This known-miss subset is diagnostic and cannot be added to the official locked-100 score. |
| [10 - Research Log/78 - 2026-08-28 - Reduced specialist workload isolates technique failures.md](10%20-%20Research%20Log/78%20-%202026-08-28%20-%20Reduced%20specialist%20workload%20isolates%20technique%20failures.md) | **sealed reduced-workload diagnostic / memory pressure rejected / 1 of 10 validated** | Reuses only the ten prior misses with 2,772--5,832-token complete prompts. Final packing loses none of 21 selected target sources, yet Sol accepts 1/10; four correct raw Terra answers are discarded by globally scoped validation. The remaining defects split into sports/cocktail admission, operator-scoped validation, insufficiency adaptation, and temporal interval/synthesis. This post-hoc result is not an official locked-100 score. |
| [10 - Research Log/81 - 2026-08-28 - Deterministic reconciliation V3 reaches 89 of 100.md](10%20-%20Research%20Log/81%20-%202026-08-28%20-%20Deterministic%20reconciliation%20V3%20reaches%2089%20of%20100.md) | **sealed development result / 89 of 100 / confirmation pending** | Gold-blind temporal, numeric, and authority reconciliation changes ten V2 answers; all ten independently flip from wrong to correct with no regressions, moving 79/100 to 89/100. The run, 100-call Sol judgment, and zero-call replays are sealed; the analyzed validation100 is development-contaminated, so semantic residual, untouched confirmation200, and fair Mem0 remain open. |
| [10 - Research Log/82 - 2026-08-29 - Semantic residual atomic-frontier diagnostic.md](10%20-%20Research%20Log/82%20-%202026-08-29%20-%20Semantic%20residual%20atomic-frontier%20diagnostic.md) | **R7 sealed / 88 of 100 / below gate** | The 2.29 GB atomic-frontier failure is replaced by a 5.47 MiB exact-replay construction with 68/68 bounded residual prompts, zero eligible fallbacks, a 5,544/8,000 maximum envelope, exact provenance, and zero retained transformer state. The full 68-Terra/100-Sol lifecycle replays byte-identically but scores 88/100: brittle lexical candidate validation suppresses computed/paraphrased repairs and admits two regressions, motivating a semantic candidate selector plus global-to-local fallback. |
| [10 - Research Log/83B - 2026-08-29 - Closure-aware selector result and evidence-layer pivot.md](10%20-%20Research%20Log/83B%20-%202026-08-29%20-%20Closure-aware%20selector%20result%20and%20evidence-layer%20pivot.md) | **V5 sealed / 88 of 100 / selector rejected as answer parent** | Exact R7/V3 reconstruction freezes 15 candidates and 13 no-call normalizations under separate R/P budgets. Fifteen Sol selector and 100 Sol judge calls replay byte-identically, but three selected recommendations yield zero rescues and one regression. V3 remains protected; the exact miss assay moves the active repair boundary to source-local reinjection and global typed answer-span retrieval. |
| [10 - Research Log/89 - 2026-08-30 - Full100 construction runtime and checkpoint audit.md](10%20-%20Research%20Log/89%20-%202026-08-30%20-%20Full100%20construction%20runtime%20and%20checkpoint%20audit.md) | **live runtime audit / 19 of 68 lower-bound progress / no accuracy result** | PID 54232 was healthy and compute-bound while processing hash-sorted namespace 4 of 10. Publication is deferred until the complete resident build returns, no namespace checkpoint can resume an interrupted run, and replay performs another full build. The log records the exact namespace order and ranks resumable sidecars, replay separation, verified-input reuse, bounded parallelism, and durable progress logging without claiming an answer or judge score. |
| [10 - Research Log/90 - 2026-08-30 - R7 after-union A1 closure preflight.md](10%20-%20Research%20Log/90%20-%202026-08-30%20-%20R7%20after-union%20A1%20closure%20preflight.md) | **sealed provider-free A1 v2 preflight / no accuracy result** | Converts the fixed R7 union into 381 exact H leaves and 11 one-question R/I/U requests under the 8K envelope. It makes fact compilation non-actionable until dispositions seal, preserves fail-open unresolved leaves and union-before-exclusion, records byte-identical construction/replay SHA `ad22a5b9…`, and passes the 53-test closure/adapter/operator suite with zero provider calls or retained model state. |
| [10 - Research Log/91 - 2026-08-30 - A1a raw-retained paired prompt boundary.md](10%20-%20Research%20Log/91%20-%202026-08-30%20-%20A1a%20raw-retained%20paired%20prompt%20boundary.md) | **sealed paired preflight / rejected 25 of 26 retention arm** | Builds 11 R+U treatments and 11 renderer-matched fixed-union controls. The aggressive 76/381-leaf sieve fit comfortably but pruned the dated smoker witness, proving a classifier temporal-schema failure rather than a packing failure; no answer calls ran. |
| [10 - Research Log/92 - 2026-08-30 - Temporal fail-open composition restores A1a target retention.md](10%20-%20Research%20Log/92%20-%202026-08-30%20-%20Temporal%20fail-open%20composition%20restores%20A1a%20target%20retention.md) | **sealed provider-free successor / 26 of 26 retention GO** | Composes generic R/I/U with a one-way question-derived temporal veto, retaining 123/381 leaves while safely pruning 258. The isolated post-seal audit reaches 26/26 atoms and 29/29 target-bearing leaves; treatment/control maxima are 3,181/4,223 under 8K, and the repaired A1b worklist has 21 unopened compiler calls. |
| [10 - Research Log/93 - 2026-08-30 - A1 hybrid terminal answer preflight.md](10%20-%20Research%20Log/93%20-%202026-08-30%20-%20A1%20hybrid%20terminal%20answer%20preflight.md) | **rejected/superseded before provider release / no accuracy result** | The v1 `82cd00c6…` preflight sealed 22 unique Terra prompts, but it was superseded before any release or provider call. Its exact-cover and budget observations remain historical apparatus evidence; this index records no replacement-v2 SHA. |
| [10 - Research Log/94 - 2026-08-30 - Compact full100 checkpoint import and 20-second replay.md](10%20-%20Research%20Log/94%20-%202026-08-30%20-%20Compact%20full100%20checkpoint%20import%20and%2020-second%20replay.md) | **production compact-v2 import complete / 20.281-second byte-identical replay** | Imports the 2,457,003,621-byte legacy sidecar population into a roughly 2.299 GiB root with ten approximately 19 KiB reference checkpoints. Initial deep authentication took about 12m55s at a sampled 0.87--0.99 GB working set; exact-attestation-pinned replay reproduced SHA `7fe63e38…` in 20.281s with zero provider calls or retained token state. Focused 20/20 and full-file 39/39 tests plus independent review returned GO; this is apparatus evidence, not QA accuracy. |
| [10 - Research Log/95 - 2026-08-30 - A1 three-arm factorial terminal preflight.md](10%20-%20Research%20Log/95%20-%202026-08-30%20-%20A1%20three-arm%20factorial%20terminal%20preflight.md) | **sealed provider-free v2 preflight / no accuracy result** | Seals 33 prompts at `97596e12…`; exact 123-leaf membership, maximum prompt/envelope 4,232/5,000 tokens, zero calls. |
| [10 - Research Log/97 - 2026-09-01 - Retrieval apparatus refactor and residual speedup.md](10%20-%20Research%20Log/97%20-%202026-09-01%20-%20Retrieval%20apparatus%20refactor%20and%20residual%20speedup.md) | **provider-free behavior-preserving refactor / 60.6% focused-slice speedup** | Reuses immutable manifest inventory, semantic-tree metadata, node bounds, and repeated packet-fit probes; source-gate projections share one body but deliberately reseal later accesses, and terminal cap checks remain uncached. The six-file slice falls from 73 tests in 60.19 seconds to 74 in 23.71; remaining global-test cost is predominantly per-row SQLite fixture construction, not semantic search. |
| [10 - Research Log/98 - 2026-09-01 - Provider slot and semantic storage cleanup.md](10%20-%20Research%20Log/98%20-%202026-09-01%20-%20Provider%20slot%20and%20semantic%20storage%20cleanup.md) | **implemented provider-free cleanup / no accuracy claim** | Keeps compact v1 byte-identical, adds strict compact v2 with H/S/K aliases and local-only stable IDs, removes process-global raw-text caches, and stores one shared semantic cell population. Sealed-ten full-chat tokens fall from 40,783 to 28,827 and the complete matched-eval suite passes 878 with one skip. |
| [10 - Research Log/100 - 2026-09-01 - Orphaning audit and lifecycle repair.md](10%20-%20Research%20Log/100%20-%202026-09-01%20-%20Orphaning%20audit%20and%20lifecycle%20repair.md) | **provider-free lifecycle repair / no accuracy claim** | Repairs pending-ingest ownership and index publication so committed memory cannot be orphaned between database and index state, with explicit recovery and compatibility tests. |
| [10 - Research Log/101 - 2026-09-01 - Terminal v5 95 percent campaign.md](10%20-%20Research%20Log/101%20-%202026-09-01%20-%20Terminal%20v5%2095%20percent%20campaign.md) | **sealed validation pass / 95 of 100 / confirmation pending** | Freezes the terminal-v5 lineage, adds proof-carrying numeric frontier closure and reducer-observable state equivalence, reuses 97 exact prior judgments, and accepts all three novel rows. The disjoint confirmation200 population remains unopened pending source freeze and its missing 20-shard execution lifecycle. |
| [10 - Research Log/102 - 2026-09-03 - Policy v5 r3 freeze and confirmation executor.md](10%20-%20Research%20Log/102%20-%202026-09-03%20-%20Policy%20v5%20r3%20freeze%20and%20confirmation%20executor.md) | **policy source freeze / confirmation unopened** | Binds the 95/100 validation lineage to an exact implementation tree and immutable manifest, records the five remaining validation misses and exposure boundary, and implements the small content-addressed confirmation executor without opening the disjoint confirmation200 answers. |
| [10 - Research Log/103 - 2026-09-03 - Durable capture and searchable ingest throughput rig.md](10%20-%20Research%20Log/103%20-%202026-09-03%20-%20Durable%20capture%20and%20searchable%20ingest%20throughput%20rig.md) | **capture-first pipeline / controlled local-disk fake-model measurement** | Defines T0 durable capture, T1 searchable publication, and deferred T2 enrichment. The 2026-09-04 v3/r15 finite burst passed all applicable gates at 19,387.1924 end-to-end chunk-token proxies/s and 1.4880827 s T1 p95 lag; synchronized-repository diagnostics explain the slower-path regression and argue against a full Rust rewrite. |
| [10 - Research Log/104 - 2026-09-04 - Ingest-derived episode descriptor shadow.md](10%20-%20Research%20Log/104%20-%202026-09-04%20-%20Ingest-derived%20episode%20descriptor%20shadow.md) | **query-independent descriptor compilation / live narrowing disabled** | Compiles and validates the 1M store's 2,238 episode representatives once into an 8.74 MiB resident matrix. Warm exact scoring measures 0.246 ms median, but Qwen winner containment and an apply treatment remain promotion gates. |
| [10 - Research Log/105 - 2026-09-05 - Source-local contextual cards with LFM2 Transcript.md](10%20-%20Research%20Log/105%20-%202026-09-05%20-%20Source-local%20contextual%20cards%20with%20LFM2%20Transcript.md) | **sidecar contextual-card prototype / synthetic 3 of 3 composed smoke** | On the same one-summary v4 wire, LFM2-2.6B-Transcript produced 3/3 structurally accepted cards while 350M Extract produced 0/3, but native Transcript compilation was 43.9% slower. A 57-query post-warm-up comparison found that shadow source gating cut mean/median/p95 card-search latency by about 35% and candidate inspections by 44.4%; narrowed routes remain additive and require an untimed raw fallback. This is not a 1M or benchmark accuracy claim. |
| [10 - Research Log/106 - 2026-09-05 - QKOV to MiniLM distillation cascade.md](10%20-%20Research%20Log/106%20-%202026-09-05%20-%20QKOV%20to%20MiniLM%20distillation%20cascade.md) | **negative proxy-teacher distillation pilot / no live narrowing** | A tuned MiniLM ranker reached 56.7% held-out top-one agreement with independent Qwen coverage rankings and exposed no subset meeting the 95% calibration-precision gate. MiniLM was fast, but the teacher target was neither production nested linking nor evidence accuracy, so all cases correctly fell back. |
| [10 - Research Log/107 - 2026-09-05 - Evidence-supervised MiniLM accuracy experiment.md](10%20-%20Research%20Log/107%20-%202026-09-05%20-%20Evidence-supervised%20MiniLM%20accuracy%20experiment.md) | **development OOF accuracy gate failed / generic reranker retired** | Five fresh out-of-fold MiniLM models moved oracle-proxy session hit@8 from lexical 178/200 to 181/200 and all-session coverage from 138 to 145, but exact annotated-turn hit@8 fell from 155 to 127 with 43 regressions. All seven advance checks failed, including 75.62 ms p95; validation100 and confirmation200 remained closed. |
| [10 - Research Log/108 - 2026-09-05 - Hot raw-chunk retrieval and source-local linking.md](10%20-%20Research%20Log/108%20-%202026-09-05%20-%20Hot%20raw-chunk%20retrieval%20and%20source-local%20linking.md) | **linked-windowed dev1M successor / 71.4195 ms p95 / 10 of 10 Terra-Sol semantic** | The protected four-lane raw-chunk union reaches all required sources, restores `Serenity Yoga`, and applies an explicit calendar lookback before every lane selects, removing the stale Killers decoy that held v5 to 9/10. Retrieval replay is byte-identical with zero provider calls, raw packets max at 4,460/5,041 context/workspace proxies, and the separate 10-Terra/10-Sol zero-retry plane scores 10/10 on development—not the locked 95% validation population. |
| [10 - Research Log/109 - 2026-09-05 - Hot raw-chunk locked100 comparative result.md](10%20-%20Research%20Log/109%20-%202026-09-05%20-%20Hot%20raw-chunk%20locked100%20comparative%20result.md) | **policy-frozen comparative full100 / 72.5515 ms p95 / 66 of 100 semantic** | Across ten independent approximately-1M-token memories, provider-free retrieval replays byte-identically at 51.93795/72.5515 ms p50/p95 and the sealed 100-Terra/100-Sol answer plane scores 66/100. The wide frontier covers all sources for 95/100 but fixed admission retains only 72/100; this analysis-used fixture is not untouched confirmation. |
| [10 - Research Log/110 - 2026-09-06 - Adaptive source-balanced hot full100 result.md](10%20-%20Research%20Log/110%20-%202026-09-06%20-%20Adaptive%20source-balanced%20hot%20full100%20result.md) | **adaptive comparative full100 / 93 of 100 source-complete / 70 of 100 semantic** | V7 protects the exact v6 packet and admits 32 source-balanced candidates from its sealed BM25/dense/temporal frontiers. Complete-source reach rises 72→93, literal containment 51→54, and Terra/Sol semantic accuracy 66→70 through eight wins and four losses. Retrieval and both provider journals replay exactly; its quadratic overflow-packing tail is repaired byte-identically by sealed v8. |
| [10 - Research Log/111 - 2026-09-06 - Binary ranked-prefix packing full100 result.md](10%20-%20Research%20Log/111%20-%202026-09-06%20-%20Binary%20ranked-prefix%20packing%20full100%20result.md) | **sealed provider-free v8 packer / 100 of 100 byte-equivalent** | Binary ranked-prefix packing preserves every v7 provider-bound payload with 56 full-fit fast paths and 44 binary fallbacks, zero provider calls, and exact replay. Packing falls from 129.835349 to 30.229117 ms mean and 346.8103 to 67.8294 ms p95: 4.295x and 5.113x speedups against a non-contemporaneous same-machine baseline. Retrieval/provider accuracy is unchanged by byte equivalence. |
| [10 - Research Log/112 - 2026-09-06 - Activated-source assertion projection v2 provider-free result.md](10%20-%20Research%20Log/112%20-%202026-09-06%20-%20Activated-source%20assertion%20projection%20v2%20provider-free%20result.md) | **sealed provider-free projection / source-preserving hybrid candidate** | Exact assertion spans use about 2.5k context tokens and improve mean evidence F1, but standalone projection loses whole activated sources (81/100 complete versus v7's 93/100). A post-hoc source-seed hybrid preserves 93/100 source reach, raises literal containment 54→56, and identifies a 1,400-token projection budget for a separately sealed v3 candidate; no new semantic-answer claim is made. |
| [10 - Research Log/113 - 2026-09-06 - Source-seed assertion hybrid v3 provider-free result.md](10%20-%20Research%20Log/113%20-%202026-09-06%20-%20Source-seed%20assertion%20hybrid%20v3%20provider-free%20result.md) | **sealed provider-free source-seed hybrid / semantic evaluation pending** | The computable hybrid protects one raw chunk per v7 source, adds a 1,400-token assertion prefix, then refills from raw. All 100 packets take the hybrid route with zero fallback, preserving 93/100 complete-source reach while moving literal containment 54→56 and mean best evidence F1 0.099386→0.115009 (+15.72% relative). These are analysis-used evidence diagnostics, not semantic accuracy or a 95% claim. |
| [10 - Research Log/114 - 2026-09-06 - Provider-free typed witness successor and monotonicity rejection.md](10%20-%20Research%20Log/114%20-%202026-09-06%20-%20Provider-free%20typed%20witness%20successor%20and%20monotonicity%20rejection.md) | **97 of 100 structural source coverage / composition rejected** | Profile, typed-witness, and activated-turn lanes add four complete-source cases without provider calls, but exact-ID excerpt collisions and packing displacement violate monotone parent protection. The source diagnostic remains valid; the composition is not promoted and is not a 97% answer score. |
| [10 - Research Log/115 - 2026-09-06 - Gold-blind ordered-story residual7 repair.md](10%20-%20Research%20Log/115%20-%202026-09-06%20-%20Gold-blind%20ordered-story%20residual7%20repair.md) | **sealed provider-free residual7 evidence gain** | A strict exact-cardinality story selector recovers q86's Muir Woods → Big Sur/Monterey → Yosemite source sequence, moving source recall 2/3→3/3 while packing fewer rows and leaving six controls unchanged. No answer-model judgment ran. |
| [10 - Research Log/116 - 2026-09-06 - Persistent conversational graph and full100 transfer.md](10%20-%20Research%20Log/116%20-%202026-09-06%20-%20Persistent%20conversational%20graph%20and%20full100%20transfer.md) | **durable T1g graph + 98 of 100 structural source coverage** | Persists bounded phrase, sequence, and source-story deltas as turns become searchable, with fail-open retry, restart hydration without re-extraction, and exact receipts. The ordered-story full100 transfer reaches 98/100 complete reference-source coverage; resident q86 graph lookup is 3.332 ms core and 7.086 ms p95 through the authenticated wrapper. These are evidence and latency results, not judged answer accuracy. |
| [10 - Research Log/117 - 2026-09-07 - Online graph fast-path full100 99 source coverage.md](10%20-%20Research%20Log/117%20-%202026-09-07%20-%20Online%20graph%20fast-path%20full100%2099%20source%20coverage.md) | **sealed v4 / 99 of 100 structural source coverage / exact replay** | Composes the authenticated resident ordered-story graph and bounded business-milestone frontier, reaching 99/100 strict source coverage versus the 93/100 parent with 100 exact replay arms and zero retrieval model/provider calls. The sole strict miss already contains the literal answer; judged fast-path answer accuracy remains a separate gate. |
| [10 - Research Log/118 - 2026-09-07 - User-led envelope pilot and additive segmentation assay.md](10%20-%20Research%20Log/118%20-%202026-09-07%20-%20User-led%20envelope%20pilot%20and%20additive%20segmentation%20assay.md) | **sealed provider-free 8Q pilot / 100% target evidence / opt-in** | Preserves raw anchors, packs exact user-to-machine microepisodes atomically, and traverses separately sealed macro links only for frontier questions. Mean packet size is 59.875 tokens versus 87.25/127.25 for the representative comparators; this small synthetic architecture pilot is not answer accuracy or a causal segmentation-only estimate. |
| `07 - Status Reports/…` | ✅ | Six dated handoffs through 2026-08-19; the 2026-08-15 report remains the retrieval-measurement handoff, while the 2026-08-19 reports cover the later simplification audit and implementation |

## The tree

```
docs/
├── 00 - Theory/           Retrieval, proof-carrying coverage, associative memory, and episodic closure
├── 01 - Design/           The original architecture plan + eval design rationale
├── 02 - Implementation/   Setup, eval modes, Qwen labs, frozen contracts, and as-built mathematics
├── 03 - Architecture/     The as-built system map, package layout, and proposed hypergraph memory plane
├── 04 - Reference/        External landscape (SimpleMem, Mem0, MemDelta…) + vocabulary
├── 05 - Standards/        Normative data contracts (SQLite v2, embedding, memory provenance, formats)
├── 06 - Roadmaps/         Gap analysis: designed vs. built vs. measured, tiered next steps
├── 07 - Status Reports/   Dated snapshots (session handoffs)
├── 08 - Analysis/         Measured results, corrections, and evidence-backed engineering decisions
├── 09 - Archived/         Superseded material (append-only)
├── 10 - Research Log/     Dated experiment entries with data/ artifacts; baselines of record
└── 11 - Codex Workstream/ The Codex transcript decomposed: 9-chapter dev guide + 40 ADRs
```

## Governance

| Folder | Purpose | Freeze policy |
| --- | --- | --- |
| 00 Theory | Foundations | Stable; corrections only |
| 01 Design | Rationale | Archive when superseded |
| 02 Implementation | Setup + realized specs | Versioned with code |
| 03 Architecture | System map | Keep current (single trusted map) |
| 04 Reference | External + project-level | Living |
| 05 Standards | Normative contract | Frozen after release; amend by version |
| 06 Roadmaps | Planning | Living |
| 07 Status Reports | Dated snapshots | Archive when complete |
| 08 Analysis | Measured deep-dives | Living |
| 09 Archived | History | Append-only; never edit |

## Where to start

- Resuming the user-spine hierarchy? → [Research Log 129](10%20-%20Research%20Log/129%20-%202026-09-09%20-%20User%20spine%20attention%20hierarchy%20and%20summary%20compilation.md). The builder and summary-only Qwen smoke are implemented; 93 tests pass. The smoke exposes narrow-branch routing loss, and the real-source compiler is prepared pending explicit raw-payload approval.
- Reviewing the conventional fast-packet work? → [Research Log 128](10%20-%20Research%20Log/128%20-%202026-09-08%20-%20Compact%20conventional%20fast%20packet%20admission.md). Seven fixed-pool packet controls are complete; none improved the fresh 12/30 baseline. Admission lost source/story witnesses, while temporal interpretation also failed with evidence present. The user has since resumed hierarchical attention.
- Resuming the requested summary-routing task? → [Research Log 122](10%20-%20Research%20Log/122%20-%202026-09-08%20-%20Qwen%20summary%20hierarchy%20and%20exact%20section%20hydration.md). Qwen attention operates on hierarchical summaries; raw hydration happens only after selection.
- Resuming cold? → **`06 - Roadmaps/01 - Delivering the Specified System.md` first** — it explains why every memory-arm number on record is void and what order the remaining work has to happen in. Then `07 - Status Reports/2026-08-15_retrieval-measurement-session.md` for the retrieval half, which still stands.
- "What does the system do?" → `03 - Architecture/00 - System Overview.md`.
- "How did factual retrieval improve, and what remains unproven?" → `00 - Theory/04 - From Top-K Recall to Proof-Carrying Factual Retrieval.md`.
- "How will diffuse evidence use EM-LLM?" → `00 - Theory/05 - EM-LLM Episodic Discourse Closure for Diffuse Retrieval.md` — the general closure mechanics and a bounded transient Qwen OV-transport episode signal are implemented provider-free. The latter is an attention-head semantic-change adaptation, not paper-exact token-NLL surprise, and still requires matched evaluation against fixed and embedding-change controls.
- "Where does the code live, and which imports are supported?" → `03 - Architecture/03 - Code Package Layout.md`.
- "How would a native hypergraph interact with live memory?" → `03 - Architecture/01 - Native Hypergraph Memory Plane.md` — canonical higher-order observations, pairwise serving projections, bounded traversal, and event-aware pruning.
- "How are complete sets deduplicated without losing distinct events?" → `03 - Architecture/02 - Query-Conditioned Bayesian Coverage Loop.md` — a primary six-layer Qwen3-8B QK/OV affinity arm plus a secondary compact-INI classifier, followed by recall-safe representative-first packing; the locked baseline isolates a 100%-raw versus 94.7%-packed gap and the prefix treatment is pending measurement.
- "What's left to build?" → `06 - Roadmaps/00 - Gap Analysis and Roadmap.md`.
- "How do we cut 1M-memory retrieval latency with minimal online compute?" → [Analysis 31](08%20-%20Analysis/31%20-%20Minimal-compute%20hot-memory%20retrieval%20architecture%202026-09-05.md), [Analysis 32](08%20-%20Analysis/32%20-%20Incremental%20conversational%20association%20graph%20overlay%202026-09-06.md), and [Research Log 117](10%20-%20Research%20Log/117%20-%202026-09-07%20-%20Online%20graph%20fast-path%20full100%2099%20source%20coverage.md) — source-balanced admission and binary packing provide the fast raw path; the online graph and typed temporal frontier raise strict source coverage to 99/100 with warm processing at 139.164 ms mean and 337.139 ms p95. This is evidence coverage, not judged answer accuracy.
- "How does user-led conversational ingest become episodic recall?" → [Analysis 33](08%20-%20Analysis/33%20-%20User-led%20conversational%20envelopes%20and%20additive%20recall%20overlay%202026-09-07.md) and [Research Log 118](10%20-%20Research%20Log/118%20-%202026-09-07%20-%20User-led%20envelope%20pilot%20and%20additive%20segmentation%20assay.md) — user turns own stable envelopes, assistant/system payloads remain exact evidence, and macro structure links rather than replaces microepisodes. The opt-in production read API now expands selected raw hits under a separate companion budget; the eight-question pilot is positive and provider-free, while the full100 shadow gate remains open.
- "How do I run it?" → `02 - Implementation/01 - Running the Eval Harness.md` (start with the free `--compare` mode).
- "How do I run the Qwen attention-prefix experiment?" → `02 - Implementation/03 - Qwen3 Prefix Attention Lab.md`.
- "How will episode retrieval feed the K-latent attention fusion stage?" → `02 - Implementation/04 - Episode-Primary Latent Evidence Fusion.md` — a design-frozen, query-conditioned GPU feature-to-router contract with no trained or measured fusion claim yet.
- "What exact math does the working implementation execute?" → `02 - Implementation/05 - As-Built Mathematical Reference.md` — the code-aligned equations, defaults, edge cases, tie-breaks, and focused test map for working retrieval paths that were previously implicit.
- "How do I actually *use* it day to day?" → `02 - Implementation/02 - MCP Integration.md` — the memory system is exposed to Claude Code as an MCP server.
- "Is this competitive?" → **no evidence of that yet**. The protected typed-final result is 73/100 on this already analysis-used locked population, still well below the preregistered 95% target. Reduced controls rule out batch memory pressure; the exact-ten treatment matrix shows that coverage-aware selection and CAV provenance activate but are non-monotonic when they compete inside one replacement lane. See [Research Log 68](10%20-%20Research%20Log/68%20-%202026-08-28%20-%20Compact%20provider%20budget%20repair%20and%20typed%20final%20answer%20result.md) and [Research Log 76](10%20-%20Research%20Log/76%20-%202026-08-28%20-%20Reduced%20exact-ten%20treatment%20matrix%20and%20streamed%20memory%20control.md). The same-budget Mem0 arm is also still open.
- "What is the large-model attention-head memory idea?" → `00 - Theory/01 - Extracted Attention Heads as Recursive Associative Memory.md` — a **DRAFT** whose first CAV/live-head prototype is implemented, including the full-teacher J-Space implication.
- "How do later prompts consolidate connected memory partitions?" → `00 - Theory/03 - Prompt-Driven Systems Consolidation.md` — schema-v9 causal binding plus repeated co-activation across semantic memories and evidence, including bounded iterative reads, the transient Qwen hyperplane seam, and anti-self-reinforcement rules.
- "Did the downloaded Qwen prefix produce usable CAVs?" → `10 - Research Log/02 - 2026-08-16 - Qwen3 prefix CAV gate.md` — yes on the first controlled local probe; this is not yet a retrieval result.
- "Did extracted heads improve live-memory retrieval?" → `10 - Research Log/03 - 2026-08-16 - Live Qwen head memory smokes.md` — calibrated layer-1 heads and temporal direction reached 1.000 R@1/R@3 on four development links; direct QK/OV failed and blind replication remains open.
- "Did persistent CAV/QK memory save tokens on unseen source families?" → `10 - Research Log/04 - 2026-08-16 - Safe associative memory confirmation.md` — yes locally, without lowering baseline recall; a fresh recall gain and public-benchmark result remain unconfirmed.
- "Can attention heat control how much memory each source contributes?" → `10 - Research Log/05 - 2026-08-16 - Source heat diffusion development.md` — implemented as a bounded external scalar walk plus source-aware packing; the selected dual QK/heat policy is posthoc development evidence awaiting a new locked split.
- "What exactly does 95% long-chat accuracy mean?" → `10 - Research Log/06 - 2026-08-16 - 95 percent long-chat target.md` — answer-stage judge accuracy, minimum sample size, 8k hard prompt cap, locked LongMemEval partitions, and the experiment ladder.
- "Has live schema-v8 consolidation run through the real Qwen checkpoint?" → `10 - Research Log/08 - 2026-08-16 - Real Qwen consolidation path.md` — yes on a temporary store copy; it validates execution and memory bounds, not a recall gain.
- "Did causal Qwen consolidation improve the operational long-chat probe?" → `10 - Research Log/09 - 2026-08-16 - Causal binding reaches 97.4 percent evidence recall.md` — yes on the locked local literal-evidence test (38/39, no regressions); answer-stage judged LongMemEval remains the primary gate.
- "Did it answer LongMemEval questions from a 1M-token chat?" → `10 - Research Log/15 - 2026-08-18 - Policy-locked 1M-context answer pilot.md` — its campaign artifacts report 10/10 positive judge verdicts with a mean 2,342-token legacy local prompt proxy; structural bindings are verified, but provider/judge execution and factual correctness are not independently authenticated.
- "What is the frozen v3 treatment and validation plan?" → `10 - Research Log/16 - 2026-08-18 - V3 retrieval freeze and validation campaign.md` — the final no-provider replay, exact artifact/cache identities, ten-shard held-out plan, prompt-proxy semantics, and corrected Mem0 comparison boundary.
- "What is operational now, what is still unmeasured, and what comes next?" → `10 - Research Log/17 - 2026-08-18 - Locked treatment handoff and discourse closure frontier.md` — the current readiness matrix, exact invariants and hashes, Mem0 production NO-GO, authorization gates, and the general-purpose diffuse-retrieval design.
- "Did the frozen v3 treatment generalize to all 100 validation questions?" → `10 - Research Log/18 - 2026-08-18 - Validation v3 provider-free retrieval audit.md` — not at the retrieval-admission level; it records the exact metrics, identities, post-run cache audit, and why no provider accuracy claim follows.
- "How are new retrieval methods added without replacing the strongest prior packet, and what happened at 1M?" → `10 - Research Log/22 - 2026-08-21 - Recall-guarded cumulative retrieval.md` — four provider-ready stages preserve a frozen-v3-compatible protected root, then add direct episodes, representative episodes, and artifact-global closure. On the exact original 1,039,203-token development concatenation, only direct episodes improved a scored retrieval metric; later stages preserved evidence but added no further gain under the cap.
- "What happened to the Hebbian arm?" → [Research Log 37](10%20-%20Research%20Log/37%20-%202026-08-22%20-%20Causal%20Hebbian%20H1%20arm%20restoration.md) — the mechanism existed but was never wired into the cumulative benchmark, whose sealed source had zero Hebbian events, nodes, and edges. The restored causal S0/H1 test works end to end but is negative: three single-support replacements changed 6/10 EM and 0.836009 F1 to 5/10 and 0.736009 because one discarded decisive evidence.
- "Was CAV supposed to be a linking layer, and did the genuine-link arm help?" → [Research Log 38](10%20-%20Research%20Log/38%20-%202026-08-23%20-%20Genuine%20CAV%20links%20and%20matched%20semantic%20result.md) — yes: CAV links/fuses canonical S3 evidence through rectangular concept extraction and reinjection before synthesis. The genuine matched arm stayed within the 8k/256 budget and both arms reached 10/10 under independent Sol, so this development test shows no causal semantic-accuracy gain; its one exact-score loss is only `190 pages` versus `190`.
- "Did a larger synthesizer answer the cumulative 1M contexts and score the episodic additions?" → `10 - Research Log/24 - 2026-08-21 - LiteLLM Terra episodic synthesis and rescoring.md` — yes on the ten-question development concatenation: strict Terra synthesis reached 5/10 exact match and 0.718433 F1 at S1 with exact-quote citations, while both semantic and local numeric scoring found no strong S2-only evidence. This is not a held-out or independently judged result.
- "Did an independent judge verify the 1M answers, and did the synthesis repair work?" → yes. Research Log 25 records the 10Q development diagnostic; [Research Log 45](10%20-%20Research%20Log/45%20-%202026-08-26%20-%20Locked%20100Q%20semantic%20gate%20result.md) records the clean fixed-S1 validation result. Sol accepted 56/100 sealed Terra answers, so the formal >=95% gate failed.
- "Did the fixed-stage 100Q memory test run?" → [Research Log 45](10%20-%20Research%20Log/45%20-%202026-08-26%20-%20Locked%20100Q%20semantic%20gate%20result.md) — yes. The responder and judge each made exactly 100 live calls with zero retries and then replayed offline. The semantic result was 56/100, not a pass; the fair Mem0 arm remains outstanding.
- "Which retrieval styles are failing inside a method versus between methods?" → [Research Log 46](10%20-%20Research%20Log/46%20-%202026-08-26%20-%20Retrieval-style%20intra%20and%20inter%20method%20diagnosis.md) — the sealed 100Q result is now split by evidence topology and answer operator. Sixteen errors have incomplete S1 source coverage, while 28 occur after nominal full-source acquisition; temporal ordering is retrieval-boundary-heavy and numeric aggregation is representation/reasoning-heavy.
- "How do we distinguish a failed method from one that was never applicable, and what did the first cleanup find?" → [Analysis 26](08%20-%20Analysis/26%20-%20Method%20eligibility%20failure%20attribution%20and%20apparatus%20cleanup%202026-09-01.md) and [Research Log 97](10%20-%20Research%20Log/97%20-%202026-09-01%20-%20Retrieval%20apparatus%20refactor%20and%20residual%20speedup.md) — define the joined per-method outcome contract, remove repeated immutable work, record the 60.6% focused-slice speedup, and separate query latency from SQLite-heavy fixture construction.
- "Are provider prompts leaking local data or wasting slots, and how much tree memory was duplicated?" → [Analysis 27](08%20-%20Analysis/27%20-%20Provider%20boundary%20and%20semantic%20storage%20cleanup%202026-09-01.md) and [Research Log 98](10%20-%20Research%20Log/98%20-%202026-09-01%20-%20Provider%20slot%20and%20semantic%20storage%20cleanup.md) — no gold/raw locator/model-state leak was found, but stable internal IDs and repeated provider metadata were removed behind compact v2; sealed-ten chat tokens fall 29.32% and retained descendant tuple storage falls 95.2% while compact v1 stays exact.
- "Were retrieval stages excluding evidence only when combined, and is that repaired?" → [Analysis 28](08%20-%20Analysis/28%20-%20Combinatorial%20evidence%20conservation%20repair%202026-09-01.md) and [Research Log 99](10%20-%20Research%20Log/99%20-%202026-09-01%20-%20Combinatorial%20evidence%20conservation%20repair.md) — reproduced and repaired fail-closed heuristic pruning, premature lane termination, incomplete specialist closure, false frontier closure, pre-admission dedup loss, and dedup-without-refill. The complete matched-eval suite passes 896 tests with one skip; no provider call or new accuracy claim was made, and witnesses outside the bounded first tree frontier remain the next explicit recall gap.
- "Did the new fixed-stage answer and judge paths work live before the 100Q run?" → `10 - Research Log/27 - 2026-08-22 - Fixed-stage S1 LiteLLM development diagnostic.md` — operationally yes, but that ten-question lineage is protocol-ineligible because of one duplicate reservation. The later clean validation lineage is separate and completed at 56/100; see Research Log 45.
- "Has the locked 100Q retrieval campaign completed?" → yes. All ten shards merged at `e36b54ec...22007f`, the fixed-S1 answers sealed at `d7fc47b8...2a38cd`, and the independent judge completed at 56/100. Research Logs 43--45 contain the merge, answer, and judge records.
- "Why is the cumulative apparatus slow, and what changes after the frozen campaign?" → [Research Log 34](10%20-%20Research%20Log/34%20-%202026-08-22%20-%20Cumulative%20apparatus%20performance%20diagnosis.md) — the completed offset-50 build made 963 snapshots for 480 sources and 12,891,681 source-row hash visits; across the then-sealed 60 questions S2 added evidence on 4/60 and S3 on 0/60. It separates code-backed cost candidates from the unproven cause of the q058 outlier and defines a six-step, receipt-safe optimization and benchmark sequence. Log 39 records the later exact frozen-source resume; no optimization is being relabeled as part of the historical campaign.
- "Can the intended H2 then CAV stack execute without another evaluation framework?" → [Research Log 40](10%20-%20Research%20Log/40%20-%202026-08-25%20-%20Structural%20golf%20and%20H2%20to%20CAV%20composition.md) — yes in a provider-free integration path: exact H2 evidence now feeds genuine CAV links and the actual final 8k synthesis preflight through one transient digest-bound adapter. This is not yet a locked-100 answer-quality result; the locked shard adapter, final-stage-only routing, provider answers/judge, and official Mem0 arm remain open.
- "Can EM be converted to facts without duplicating selected evidence?" → [Research Log 41](10%20-%20Research%20Log/41%20-%202026-08-25%20-%20Post-selection%20EM%20fact%20memory.md) — yes: S0 first selects S1, then the protected S0 rows are excluded from the EM delta. On dev10, cited facts retained raw's 6/10 exact score, improved F1 to 0.827558, and used 39.45% fewer final-prompt tokens. Reattaching every raw row was worse, so raw inclusion remains an optional fallback rather than the default.
- "Did the consolidated test/eval path run live, and did the renderer repair work?" → [Research Log 53](10%20-%20Research%20Log/53%20-%202026-08-26%20-%20Compact%20renderer%20and%20dual-answer%20synthesis%20diagnostics.md) — the S0-v2 control ran and replayed at 53/100, but the follow-up gated repairs did not qualify. V3 scored 4/10, v4 5/10, and dual-answer synthesis 3/10 on the selected verdict flips. Their candidate union is only a posthoc oracle ceiling; no full-100 repair run was made.
- "Did independent representative/global closure improve the matched control?" → [Research Log 59](10%20-%20Research%20Log/59%20-%202026-08-27%20-%20Independent%20closure%20v9%20matched%20outcome.md) — no. Both sealed descendants scored 52/100 against S0-v2's 53/100 and are excluded from positive-only composition. The 84/100 figure was a counterfactual ceiling, not an observed run, and 95/100 remains unpassed.
- "Can a provider-free full-store scan recover sources that S0 and both closure pools never constructed?" → [Research Log 60](10%20-%20Research%20Log/60%20-%202026-08-27%20-%20Provider-free%20partition%20scan%20construction%20diagnostic.md) — partly. Question/S0 lexical routing plus complete scans of four semantic partitions reaches 19/27 missing sources and admits 14/27 under 2,048 tokens, with no question-ID prefix filtering or provider calls. This is source-history reach, not proof of answer-bearing excerpt relevance or answer accuracy.
- "Did balanced allocation fix partition-scan packing, and do routed EM facts improve the matched control?" → [Research Log 61](10%20-%20Research%20Log/61%20-%202026-08-27%20-%20Partition%20allocation%20and%20matched%20EM%20fact%20gate.md) — partly. Allocation v2 admits all 19 sources already reachable inside the top four partitions, up from 14, while eight sources remain outside the router and answer value is unmeasured. Separately, a parent-preserving facts-only gate scores 54/100 versus 53/100 with one rescue and no regressions; the two results are not additive and 95/100 remains open.
- "Does LLM query expansion add sources beyond the balanced scan, and where are they lost?" → [Research Log 62](10%20-%20Research%20Log/62%20-%202026-08-27%20-%20Query%20expansion%20construction%20and%20packing%20audit.md) — yes, but mostly before packing. Query expansion reaches 20/27 missing candidates and admits 14; its admitted union with partition v2 reaches 180/188 total source targets (95.74%). This is structural source-ID coverage, not answer accuracy. The measured losses motivate source-balanced repacking and query-routed exhaustive partition scans before fact conversion.
- "Did the query-era answer arms improve matched accuracy, and can their union reach 95?" → [Research Log 63](10%20-%20Research%20Log/63%20-%202026-08-27%20-%20Query-era%20matched%20answer%20campaign.md) and [Analysis 14](08%20-%20Analysis/14%20-%20Query%20answer%20joint%20failure%20taxonomy%202026-08-27.md) — direct payload improves S0-v2 from 53/100 to 71/100, while facts, partition, and guided payload score 64, 59, and 58. Their posthoc five-arm oracle is only 74/100, so recombination of existing predictions cannot reach 95; operator, answer-shape, sufficiency, and remaining construction failures require new work. Mem0 is still open.
- "Did reducing the workload show a memory-management problem?" → [Research Logs 71--76](10%20-%20Research%20Log/76%20-%202026-08-28%20-%20Reduced%20exact-ten%20treatment%20matrix%20and%20streamed%20memory%20control.md) — no. The exact-ten v3 resident run held seven approximately 1M-token indexes (7,208,302 tokens total) and was then replayed one namespace per child process. Streaming cut simultaneous index residency by 85.66% and peak worker memory to about 0.822 GiB, but reproduced 7/7 namespaces, 10/10 questions, 70/70 method/question outputs, and the canonical construction bytes exactly. The four fact treatments show a technique problem instead: coverage and CAV provenance both activate, but replacement inside one shared lane is non-monotonic. The baseline fact reread remains strongest at a structural parent union of 14/23 targets and 3/10 complete source sets; no union is terminally repacked or provider-ready, and no new answer score is claimed. The next step is separate protected mechanism budgets followed by post-selection deduplication and one terminal fair repack.
- "What happens when every callback-selected fact fits and only the ten misses are answered?" → [Research Log 77](10%20-%20Research%20Log/77%20-%202026-08-28%20-%20Reduced%20callback-union%20answer%20diagnostic.md) — the sealed delta-only treatment retains all 75--109 unique callback rows per nonempty question, stays within 7,443/8,000 tokens, and scores 2/10 under ten Terra plus ten independent Sol calls. It proves later admission/composition losses on q31 and q81, but most failures remain upstream or relational: missing callback targets, source hits without decisive local spans, the q72 insufficiency case, and q86 distractor-sensitive temporal synthesis. The post-hoc known-miss result is not a locked-100 score.
- "Does the exact-ten failure persist when the answer workload is only a few thousand tokens?" → [Research Log 78](10%20-%20Research%20Log/78%20-%202026-08-28%20-%20Reduced%20specialist%20workload%20isolates%20technique%20failures.md) — yes. Ten 2,772--5,832-token prompts lose no selected target sources during final fitting but score 1/10 after validation. Terra's raw outputs already contain the correct q31, q43, q61, and q86 answers; the global validator rejects all four and restores wrong parents. q7/q81 are upstream domain-admission gaps. This isolates technique and scope errors rather than process or prompt-memory pressure.
- "Can scoped specialist repairs recover those same ten misses?" → [Research Log 79](10%20-%20Research%20Log/79%20-%202026-08-28%20-%20Scoped%20specialist%20repairs%20recover%20the%20reduced%20exact%20ten.md) — yes on the post-hoc development set. The canonical construction reaches 23/23 labelled sources with 4,213--5,849-token complete envelopes; ten Terra answers and ten independent Sol judgments replay byte-identically and score 10/10 semantic correct. This strongly rejects memory pressure for these misses and validates specialist-local proof scopes, but it is not a locked-100 score and cannot be added to 73/100. Full-population generalization and the same-budget Mem0 arm remain open.

## The one distinction this tree tries hardest to keep

**Built ≠ measured, and locally measured ≠ externally competitive.** Passing tests establish implementation behavior. The local analyses and research logs establish only their stated datasets, splits, and metrics. The QK/CAV result currently supports token saving with recall non-regression on one locked fresh split; it does not support a general recall-gain claim. Any broader claim without a public benchmark is a bug — report it.

Dated status reports, analyses, research logs, and frozen scripts may cite the
flat module paths that existed when their artifacts were produced. Active code
examples use the canonical v4 package paths; historical evidence is not
mechanically rewritten.

---

**Verification block**: run

```powershell
git log --oneline -1
git status --short
pixi run --frozen -e dev pytest -q
```

If the suite is green, this tree is accurate as far as it goes: the core and
association and source-heat paths are implemented and locally measured, while external
competitiveness remains unknown. The next evidence gates are improved
write-time association coverage and `--answer-recall` or another public/common
benchmark—not a larger transformer context or a third retrieval hop.
