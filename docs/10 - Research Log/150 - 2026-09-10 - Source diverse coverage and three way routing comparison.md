# Source diverse coverage and three way routing comparison

**Date:** 2026-09-10  
**Status:** complete development40 evidence audit; fresh 280-request comparison running  
**Predecessor:** [149 - Scoped term preflight and omitted source trace](149%20-%202026-09-10%20-%20Scoped%20term%20preflight%20and%20omitted%20source%20trace.md)

## Source concentration on the complete development population

The previous turn localized a missing purchase to source selection. The
summary preserved the fact, but its ninth-place user-summary match fell
outside the eight-leaf supplement. `tools/audit_spine_source_frontier.py`
then reproduced every baseline prompt and inspected the first 32 summary
matches for every question in all four complete memories. It read no
predictions or gold and used no witness ID to select routes.

The eight user-summary matches covered a mean **5.125 conversations**. All
40 questions had repeated sources in those eight slots. Passage matches
covered a mean **5.8 conversations**, with repetitions in 37/40 questions.
The feed-purchase witness is the fifth distinct source despite being the
ninth leaf. This motivates testing source diversity without selecting a
larger cutoff from one successful witness.

Frontier aggregate:
`eval_results/full1m-spine-source-frontier-development40-20260910-r1/audit.json`,
SHA `52e307bee80c768a1c03333c0b748c03da1ae45eade630d272df3f7a1ccf4beb`.
Four per-memory audits under
`eval_results/full1m-spine-source-frontier-offsetNNN-20260910-r1` bind the
complete question and summary populations. This is an evidence diagnostic,
not answer accuracy or query latency.

## Bounded source coverage

`search/spine_source_coverage.py` selects the highest-ranked user-containing
leaf from each of up to eight distinct sources in each channel, inspecting
at most 32 user-summary and 32 passage matches. The 32-candidate bound already
exists in the calendar route. The channels interleave their selected leaves;
duplicate user turns enter once. Previously selected user sections remain
first, followed by these additions and prior attached context.

`tools/spine_source_memory.py` applies source coverage before the existing
source-scoped term supplement. Both selection steps use stored summaries and
authenticated descriptors only. Query embedding is unchanged, no query-time
Qwen call is added, and production performs one exact hydration within the
unchanged 3,072-token/128-span allowance. Qwen remains summary-only, with its
attention-guided ingest partition unchanged.

The focused evaluator, full100 gate, facet verification and coverage suite
passes **164 tests**. Coverage tests exercise repeated-source crowding, exact
raw hydration, source/query/scope binding, empty frontiers and hard limits.
The new evaluator tests include all seven request arms.

## Complete evidence comparison

`tools/audit_spine_source_coverage.py` reconstructed both existing controls
before constructing a candidate for all 40 questions. All facet and scoped
term prompts reproduce exactly. The candidate changes 40 facet prompts and
39 scoped-term prompts. It adds **215 raw spans**, removes **59 non-user
spans**, and removes **zero user spans** from either earlier candidate.

Postscore inspection confirms that the omitted 20-pound purchase now appears
as exact raw evidence, and the named Sunday shift table remains present. These
witnesses diagnose the mechanism; no witness, expected answer or verdict
enters production selection. Additional sources may be irrelevant and lost
assistant context may reduce answer quality. Fresh judgments are required.

Evidence aggregate:
`eval_results/full1m-spine-source-coverage-development40-20260910-r1/audit.json`,
SHA `4087b725472698a659436ec3cf996bb7166f59e9ee9c81e6d9a1d0cb967c694b`.

| Offset | Evidence audit SHA-256 |
| --- | --- |
| 0 | `18daf09dc859c97baa33e4460746305ad1a7238e2bbf163b4dcdbb0510b3ed09` |
| 10 | `4f5b15dbd757c94e5d0767d77829ae72ec861528fb7864ad358d584ba872de76` |
| 20 | `1b93377af5bb1533fdcadcc1b56d6d612fba5a77063314f6ecd6642c934af57e` |
| 30 | `08043f533f66cefb9a31c3c82894f20be10f99318565e2458ee105534116a607` |

The roots are `eval_results/full1m-spine-source-coverage-offsetNNN-20260910-r1`.
Witness prompt hashes are
`6556f9f3d3d15341d07920238362a622a65aa7a4946313c0a704dad38675cf48`
(purchase) and
`0494fad126f14594817b665e6a774564eb6e4d0c6b99e1b9ec106c95ab2984d7`
(schedule).

## Prepared fresh answer comparison

`tools/evaluate_spine_source_coverage.py` compares facet routing, scoped term
coverage, and source diversity followed by scoped term coverage. Every method
has its own adjacent, counterbalanced API control with identical evidence;
short API chat remains included. Reader v2, model, dated questions, raw budget,
256-token output limit, serial streaming and zero retries are fixed.

The new preparation reproduces all **200** previously prepared control
requests exactly and adds 80 requests for the source-diverse method and its
API control. The earlier separate 200-request experiment remains unexecuted.
There are **280 prepared answers** and at most **120 logical Sol judgments**
across the new four-memory comparison. All candidate prompts match their
audited bytes. No reference answers or predictions were read in preparation.

Preparation root:
`eval_results/full1m-spine-source-coverage-development40-20260910-r1`,
`preparation.json` SHA
`915df74a33a93136b7e46284070302dfa11b6d7e4b581b52fa1e80a0c7c10ca3`.

| Offset | Answer preflight SHA-256 |
| --- | --- |
| 0 | `17721b427ac02e570b295154876d996667df7ce452996bd545d1e5bc2e855563` |
| 10 | `c3d3001ab64f084407bb3fb4b123361b05b465094492c75cd0e5b927c97734c3` |
| 20 | `93810b1e4ca1f35e9875e4aedc863df22c7468301d34bae5ccf61364f89c6444` |
| 30 | `6e1ded536854d3e1e1d384aed91a689809e65d5d59320f0dea1a0d1a43dc52e4` |

Individual roots are
`eval_results/full1m-spine-source-coverage-joint-offsetNNN-20260910-r1`.
The prepared serial runner is `run_comparisons.py` in the development40 root;
its `runner-plan.json` SHA is
`36331f9f466482d628fee8a8a3bcbbde65f40b0fba30bc7e0a7f1fee1b6bfacf`.
It requires completed offset-40 recovery, fresh successful gateway readiness,
idle local Python work, and all 280 requests unstarted. A single release marker
prevents a second release; failures preserve reservations. Judgments replay
before the development aggregate is published.

`tools/report_joint_spine_source_coverage_full100.py` retains the full
100-question accuracy and matched-latency gate, common compilation/admission
policies, complete memories, and no per-question method switching. Neither
this evidence audit nor its future development40 answer report can pass that
full100 gate. The overall objective remains active and unmet.

## Execution and fifth-memory continuation

Raw recovery session **43809** completed with exit code 0. All 837 responses
are accounted for: 99 retained responses and 738 new recovery-stage calls,
including the five explicitly bounded additional attempts for original unknown
requests. The original reservations remain preserved. Complete transport
receipt SHA:
`ab0e270be697c4f52d409928209db4db181f541f96a9d2b9bbcacf06062487b5`.
The strict initial validation aggregate is
`4dd274b37d367b8177f8b7610d4c46a6480c006345b9e7e6386e5194151c61c0`;
its quote-validation failures are not the final source-admission result.

The complete version-3 source audit replayed all responses without calls. It
found nine oversized summaries across seven batches and zero unresolved schema
failures. Audit SHA:
`3556d45a11f15c1c40367ea31e427d817c09e8d17812f503fc6fab6be5f91352`,
under the staged corpus's `offset-040/source-admission-audit-v3-prefix-0837.json`.

Nine summary-only compaction jobs were prepared in two Qwen batches under
`eval_results/full1m-spine-budget-repair-offset040-20260910-r1`, preflight SHA
`18264688777ab4812b1b4e7cc65c239640cc566e8c7238c54125de96dba7546b`.
Session **41396** terminated with exit code 1 after its first saved response
failed the output-budget check. The response remains preserved; the second
batch was not started and no automatic retry occurred. Inspect individual
slots and arrange bounded recovery after timed evaluation. The fifth memory
is not yet fully admitted or compiled.

Both synthetic local model probes passed at 10:30:21 and 10:30:23 UTC. Report:
`eval_results/spine-gateway-readiness-offset040-20260910-r5/report.json`,
SHA `d0210bc575f8051e0399624b666a1109d9491a890465d1849902545ffb8d41c2`.
After raw ingest, audit and compaction were terminal, the frozen comparison
runner released **280 fresh requests** and started in session **68509**. Its
first short-API and facet-memory requests completed. Poll this session; do not
start another runner, bulk inference, GPU compilation or large replay audit
while it is timing answers. The runner judges each completed namespace and
replays judgments before publishing the complete development40 report.

All diagnostic and preparation processes are terminal: frontier audit 94418,
source-coverage audit 60423, preparation 43571, and source-admission audit
69833 exited successfully. Source repair, fifth-memory hierarchy/index
construction and untouched raw namespaces 50 through 90 remain pending after
timed evaluation. The earlier separate 200-request term experiment is still
unstarted; its controls are already included in the active three-way run.

## Interim results and repair implementation

The first two complete memories have fresh answers and judgments. These are
interim results from the still-running frozen comparison, not a full40 or
full100 conclusion.

| Memory offset | Facet control | Scoped terms | Source diversity plus scoped terms |
| --- | ---: | ---: | ---: |
| 0 | 10/10 | 8/10 | 9/10 |
| 10 | 8/10 | 8/10 | 9/10 |
| Combined interim | 18/20 | 16/20 | 18/20 |

Offset 0 answers SHA:
`14dba98ee6921f98d34089a60a02a8dd9be4403035a986c5acd88789506fb357`;
report SHA:
`872593a66fd0ebe78f4c8b1fc7fcc0d05f1b9aae1ce3717cc4eaa4a9859a61ac`.
Offset 10 answers SHA:
`ea1f0ae072952ed805f4903dff60a3dc690da611d2fef83583d5d81b803f7d25`;
report SHA:
`815419e7dee78473f9ae30ebc2a8cbc80cfcaa8ec8006bf3f89fad9aa14d5c91`.
The 60 logical judgments used 15 and 12 physical Sol calls respectively.
The runner will replay all judgments when the full development report is built.
No method or prompt changes were made during this comparison.

Separately, the following recovery implementation has been written but **not
yet tested or run**, to avoid local test/replay work during measured answers:

- `tools/finish_spine_compaction_batches.py`: prepare a snapshot of completed
  and unstarted original v4 compaction batches, preserve completed responses,
  refuse unknown reservations, and execute only the prepared unstarted batch
  population. Invalid completed outputs are reported for separate recovery.
- `tools/recover_spine_summary_budget_v2.py`: reconstruct the v4 original
  batches, preserve every valid output, and recover only invalid slots using
  the existing maximum of two individual calls at 48 then 24 words.
- `tools/verify_spine_admission_method_v8.py`: replay that complete recovery
  before accepting source atoms, count original and recovery attempts, and
  retain one conditional method across older and recovered memories.
- `tools/report_joint_spine_source_coverage_full100_v2.py`: retain the existing
  full100 accuracy and latency gate using version-8 admission verification.
- `tests/test_spine_compaction_batch_completion.py`: exercise unstarted-batch
  completion, valid-output preservation, invalid-slot recovery, raw-input
  canaries, zero-call replay, bounded exhaustion, and unknown-request refusal.
  Existing full100 and facet-verification tests now include the new report.

These new files do not modify code bound by the running evaluation. After
session 68509 becomes terminal, run the new focused checks before using them.
Then finish the original unstarted compaction batch, inspect the complete
invalid-slot population, prepare and execute its bounded recovery, admit the
fifth memory and verify all five memories under one version-8 method. Resume
raw namespace 50 and fifth-memory hierarchy construction after measured
evaluation has finished. Do not restart the original failed compaction command.
