# Supplemental routing through complete parent user summaries

The previous user-first layout run scored 93/100 with a 4.651 s warm median,
versus 4.581 s for matched API answers. Log 231 traced two incomplete-support
packets to parent selection: a relevant rank-eight source received no parent
expansion, and another selected parent omitted an earlier vinyl-collection
statement. The broader stored parent user summaries still contain those facts.

## Derived parent index, no new ingestion

`search/native_spine_parent_users.py` projects each stored hierarchy root onto
its exact user spans and its existing `user_spine` string. It excludes the
`attached_context_not_user_assertions` field from embedding inputs. It retains
singleton roots and omits roots with no user spans. Raw text, question text,
reference answers, source IDs and address coordinates do not enter the embedding
text. Provenance binds the projection to the original hierarchy and root receipt.

`tools/compile_native_spine_parent_users.py` reopens the existing application and
embeds 522 projected root summaries with the same pinned FP32 BGE-M3 encoder
used for atomic summaries and queries. The longest input is 575 model tokens,
so no summary is truncated. Configured batch size remains eight, with an 8,192
document-token batch cap. Compilation took 47.406 s. It generated no answers,
called no Qwen model, and performed no raw ingestion.

`persistence/native_spine_parent_store.py` publishes the normalized FP32 matrix
once to `application/native-spine-parent-users-v1.sqlite`. It binds the native
snapshot, projection, encoder and vector bytes; reopening validates every binding.
The existing `memory.db`, HNSW index and `native-spine.sqlite` hashes are unchanged.
The cache is additive; never overwrite it or reingest the history to reproduce it.

- Compilation root: `eval_results/native-spine-parent-users-20260915-r1`
- Exec session: `24371`, terminal exit 0.
- Preflight: `bd41b1e889861ccbe07913fc8f90ab7ecfd1138bb41089b8d45ed243c5f0a347`
- Report: `69ed670698fa94ed478e1b275c3ed7062610751f7a38586260f3527c6cb7178f`

## Bounded application routing

`NativeSpineParentUserRouter` first computes the unchanged semantic/parent-context
and additive lexical route. It ranks the complete parent user summaries of the
sources in the eight direct candidates using the same already-computed query
vector. From the best parent, it takes at most two unselected user atoms, ranked
by their atomic-summary similarity to that vector. They append after the entire
prior route. An already-covered parent changes nothing.

No parent addition enters or competes for the existing four parent seeds or 16
context-atom slots. All original route order and selected evidence are retained.
The exact hydrator still owns the 2,048-token and 128-span caps. The new typed
receipt embeds the baseline and parent plan, checks the dated source subset and
the two-user-atom bound, and rejects altered prefixes or source scope.

`ParentUserMemoryCondenser` reuses normal transcript/snapshot admission and loads
the persisted parent index once. It retains encoder, revision and close gates.
Serving embeds one fresh query, scores cached atomic and parent vectors, and
hydrates the original raw sections. No live Qwen pass or raw-text scoring occurs.
Presentation uses the timestamp-corrected v2 user-first renderer; its text was
already proved identical to v1 on the previous 100 packets.

Twenty-three routing, persistence and lifecycle checks passed in 4.97 s. The
evaluator/routing/application-gate set passed 23 checks in 3.37 s. They cover
missing siblings, prefix/evidence preservation under identical budgets, future
source exclusion, wrong encoders, changed receipts, store corruption, normal
ingest/close/reopen, and exactly one live query embedding.

## Complete provider-free coverage result: 99/100

`tools/assess_native_spine_parent_users.py` reopens the actual application once,
retrieves all 100 questions, seals every packet before references open, and
independently reconstructs all raw text and presentation against the source bank.
Every original route and hydrated section remains present. Eighteen questions
receive extra route candidates; 20 additional raw spans fit the unchanged budget.

Complete recorded-support coverage increases from **98/100 to 99/100**, with one
partial packet and no losses. Ordinal 93 now includes the missing vinyl statement.
Median conversation count remains four, and median rendered length is 1,554
tokens versus 1,552.5 previously. All 100 packets and 1,459 spans pass raw audit.

Ordinal 53 remains partial. The best parent summary describes the other real
Katy Perry/Billie Eilish/Summer Sounds conversation, which also matches the vague
concert question. No source-specific override was added to force the reference
conversation. This ambiguity stays documented; coverage is not answer accuracy.

- Assessment root: `eval_results/native-spine-parent-users-coverage-20260915-r1`
- Exec session: `26339`, terminal exit 0.
- Report: `f3abfe57c77e8ca7413537a59e963489145b5fae32c52453d02ee7377e0c37fa`
- An additional evaluator admission check accepted all 100 saved packets before launch.

## Completed answer and latency comparison: 93/100

`tools/evaluate_native_spine_parent_users100.py` binds the parent cache, its
question/gold-free compilation, the provider-free audit and the new application
implementation. It preserves the same v7 reader, Terra model, original 100
questions/references and Sol grader. Each memory answer performs fresh retrieval
inside the timer, followed by an identical-prompt direct API control in alternating
order. All 200 answers sealed before grading. All 100 packets and 1,459 raw spans
passed independent reconstruction. A provider-free judge replay returned 100
cache hits, zero new calls and the identical report hash.

- Root: `eval_results/native-spine-app-parent-users100-20260915-r1`
- Log: `eval_results/native-spine-app-parent-users100-20260915-r1.log`
- Original exec session: `5621`; `complete.json` is published and the session is
  no longer available. Do not duplicate the completed run.
- Judge replay: session `50998`, terminal exit 0.
- Same one history: 1,098,417 raw tokens and 5,357 raw turns. The sealed scope also
  reports 1,098,417 tokens eligible through the question day.
- Same numeric policy: `dense-parent-2048-direct8-v1.json`.
- Same reader: `user-coverage-v7.json`.

| Measurement | Memory | Matched direct API |
| --- | ---: | ---: |
| Warm median total | 4.246 s | 3.872 s |
| Warm p95 total | 6.518 s | 5.870 s |
| Answers below five seconds | 68/100 | 85/100 |

Fresh retrieval preparation has a 0.288 s median. Cold resident setup took
30.282 s and is excluded from warm measurements. All 200 answers ended with
`stop` and reported the Terra alias. The median memory/API ratio is 1.097.
The gateway buffers output; these timings do not establish streaming behavior.

The candidate scores **93/100**, tying the immediate user-first layout and earlier
width-eight best. Against the immediate baseline it gains ordinal 26 and loses
ordinal 19; both prompts are byte-identical between runs. Fifteen other prompts
changed because of additional evidence. The answer-score tie does not establish
an accuracy gain from the parent supplement, despite its verified coverage gain.

The remaining failed ordinals are **19, 53, 81, 82, 93, 95 and 96**:

- 19, 81 and 82 omit relevant details already present in the first user block:
  the acting-depth qualification, the dreams/nightmares contrast, and the drama's
  real-world/current-events requirements. These are reader omissions.
- 53 retains the previously documented partial-support and ambiguous-concert
  problem. No reference-specific source override was added.
- 93 now contains the complete vinyl statement, but the reader still selects
  stamps among the requested three collections. Several collections are real;
  both the question's scope and the reader's selection need attention.
- 95's grade rejects French-cinema interests previously verified in user text;
  do not describe them as invented or silently change the original grade.
- 96 combines genuine app names from different shopping conversations, instead
  of staying within the household-shopping situation.

This result supports investigating answer completeness and conversation scope.
It does not establish that raw-text attention pruning or a model replacement is
required. The previous Sol comparison also tied 93/100; no new model comparison
was performed here.

- Preflight: `abdef23f1f70db7ddb89eea9ca68b17bebb20cb0782cdf2221f2aac25d095203`
- Joint report: `796f777eaa1e2216c1c8d676441adccf6ca580780b890b17daa1f2747a54058c`
- Raw audit: `24af21be6812ba0e63bb21d56771e91a158bb1c83df8817036a8cf9c030c5146`
- Comparison: `c3a7310a51ba69686ab9cedb8f0d06aca5981aeff428e411a215cd581b01cdbd`
- Failure diagnosis: `b984bafafc45e84d15457de286c0267ec832fe3efdcaf547ddc82f2c7656cfce`

The best completed score remains 93/100 and the 95% target is unmet.
These are exposed generated questions over real source conversations, not an
official LongMemEval or generalization result. Original grades and known question
defects remain unchanged. Do not conflate the 99/100 support result with accuracy.
