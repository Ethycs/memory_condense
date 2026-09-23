# Extractive reader on unchanged user spine evidence

The completed parent-summary supplement in Log 232 retrieved every recorded
support quote on 99/100 questions, but answer accuracy stayed at 93/100. Three
failed answers omitted relevant content already present in the first user block.
This comparison changes only the reader instructions, retaining the same history,
100 questions, references, Terra answer model, 256-token limit and Sol grading.

## Fixed reader policy

`src/memory_condense/eval/spine_reader_policy_v8.py` replaces the accumulated v7
instructions with a standalone extractive policy. It asks the reader to select
the conversation matching the full situation, preserve each relevant user
statement and qualification, and use a short list for multiple points. It
preserves complete supplied passages, distinguishes assistant suggestions from
user facts, and retains time, negation, uncertainty and abstention rules.

The policy contains no benchmark entities, source IDs, question IDs, reference
answers or per-question exceptions. It is development informed by exposed
failures, not a held-out experiment. Its instructions use 472 tokenizer tokens,
versus 800 for v7. No new Qwen processing or raw-text attention is introduced.

- Sealed policy: `eval_results/native-spine-reader-policies/user-extraction-v8.json`
- Policy SHA-256: `81a7c49ce0764ced09874d78dfe247971b7298f596f23c9089453750a45e1c2f`
- Evaluator: `tools/evaluate_native_spine_extractive100.py`

The evaluator requires the exact v8 policy and compares each fresh packet against
the completed parent-summary baseline. Routing, hydration, rendering and the
entire question/context message must be identical. Candidate and direct API
controls receive identical prompts in alternating order. Timed memory answers
perform fresh retrieval through the reopened persisted application. All 200
answers must finish before references open for grading.

## Validation before provider execution

Twenty-six extractive/parent evaluator checks passed in 3.33 s; the application
admission gate passed in 1.71 s. They cover fresh-query embedding, exact evidence
preservation, changed questions and prompts, mismatched controls, policy changes,
and rejection of grading before all answers finish.

A provider-free admission check compared all 100 saved baseline packets under
the new instructions. Only the system message changes; all evidence, routing,
presentation and question text remain identical. This check reuses sealed
packets and makes no claim of fresh retrieval or answer accuracy.

- Admission root: `eval_results/native-spine-extractive-admission-20260915-r1`
- Admission report: `c9167db0e4c1f0749c2a3ea4e3a88f1dcda6f39459eb40e9efbf93899ff7d3f5`
- Admission worker: session `81740`, terminal exit 0.

## Summary-attention provenance verification

While the answer run executes, a provider-free check replays the stored attention
receipts for every selected source body. It reconstructs each window from the
exact `UserSpineExchange.user_spine` strings and compares the signal receipt
sequence against that body's hierarchy template. All **520 bodies and 525
summary windows** pass. No model is loaded or invoked by this check.

The 220 reused templates have older enclosing exchange/atom artifact identities.
They pass `bind_parent_budget_hierarchy` against the current source records and
atoms: summary text, roles, raw hashes, character boundaries, token counts and
complete root coverage are preserved. Comparing array positions directly would
be incorrect because the summary index sorts section IDs; the binder uses the
root's original span order. All 520 templates pass the same content checks.

The authenticated attention method is Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, FP16 prefix computation, six prefix
layers, attention layer five and four-head voting. This verifies the existing
summary-only attention inputs; no query-time Qwen pass is introduced.

- Root: `eval_results/native-spine-summary-provenance-20260915-r1`
- Report: `02fed5a33c02401bb283ba8f78817be3da5ab469963b9a3621e6906174887384`
- Worker: session `62317`, terminal exit 0.

## Completed full comparison: 90/100, not promoted

- Root: `eval_results/native-spine-app-extractive100-20260915-r1`
- Log: `eval_results/native-spine-app-extractive100-20260915-r1.log`
- Answer/judge/audit worker: session `65661`, terminal exit 0.
- Provider-free judge replay: session `61353`, terminal exit 0, 100 cache hits
  and zero new calls; report hash unchanged.
- One existing history: 1,098,417 eligible raw tokens, 5,357 turns.
- Retrieval policy: `dense-parent-2048-direct8-v1.json`.
- Same parent-summary cache, exact raw hydrator and user-first v2 presentation.

| Measurement | Memory | Matched direct API |
| --- | ---: | ---: |
| Warm median total | 4.025 s | 3.624 s |
| Warm p95 total | 7.406 s | 6.612 s |
| Answers below five seconds | 70/100 | 81/100 |

The reader scored **90/100**, versus the unchanged-packet v7 baseline's 93/100.
All 200 responses ended with `stop` and reported the Terra alias. Query
preparation has a 0.290 s median. Cold resident setup took 30.458 s, excluded
from warm latency. The memory/API median ratio is 1.111. The gateway buffers
responses, so this does not measure incremental visible decoding.

All 100 packets and 1,459 raw spans passed independent source reconstruction.
The paired comparison confirms that routing, hydrated text, rendering and
questions are identical across all 100 items. Only the system instructions
changed. The shorter reader is **not promoted**; retain v7 for further work.

It gains ordinals **81 and 93**: the complete story opening is preserved, and
the answer now selects vinyl, cameras and novels with the expected details.
It loses **0, 6, 15, 26 and 78**. The ten failed ordinals are
**0, 6, 15, 19, 26, 53, 78, 82, 95 and 96**.

- 6 and 78 drop supplied details about cave-painting questions and HTML output
  requirements. 19 and 82 retain the prior partial-answer behavior.
- 0 repeats a confirmed grading defect: the answer `B.` is rejected because it
  does not repeat `not A`. This is not a retrieval or factual-answer failure.
- 15 gives the bake sale and both promotion channels. The reference also requires
  friends/family helping bake; that is event preparation beyond the explicit
  promotion question and needs separate question/reference review.
- 26 includes source-supported Indigenous events/festivals absent from the
  reference; the original grade rejects the additional true information.
- 53 retains the partial-support/ambiguous-concert problem. 96 mixes genuine
  app names from different shopping situations.
- 95 includes source-supported French/international interests rejected by the
  grade, and also omits the named Tribeca example expected by the reference.

The simplified instructions do not reliably fix completeness. Preserve every
original grade and the 100-item denominator; no corrected or cross-run score is
claimed. Further reader work should distinguish answer-generation variability
from grading defects using already saved paired answers before another prompt
sweep. The matched controls have saved predictions but have not received a
separate accuracy score in this experiment.

- Preflight: `c836568a71f2ec9a2249e00a2ce167e4b7f4b75f63363a38f90bb08f80bc8c85`
- Joint report: `2e4bc3df9c7d2e998ab7de462f671347c22cfe2a49f8b38a4887cc7688a8e51c`
- Raw audit: `f13dfbc1a1153e859a9abdc1807187b520bcf971ffae1aa94f052acea69e312b`
- Comparison: `56c4304615e599d4ee6f4cfbd0839d8a74607efbf12d3b979638ec529cb3d1cb`
- Failure diagnosis: `0a465b5487fe5eb9deaf1b38281c52c2f52193e18de24c88f3928c252184e4b3`

The best completed accuracy remains **93/100**. This is an exposed development
set over real transcripts, not official LongMemEval or evidence of
generalization. The 95% goal remains active.
