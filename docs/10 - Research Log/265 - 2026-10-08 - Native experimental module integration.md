# Native experimental module integration

**Status:** Completed integration assay; default promotion pending answer-quality evidence  
**Date:** 2026-10-08

Seven dormant backend mechanisms now have opt-in adapters for the native
summary-routing/exact-hydration architecture: learned association recall,
scalar heat diffusion, Qwen summary routing, Qwen association linking, CAV
linking, temporary head-memory retrieval, and Qwen-weighted IO consolidation.
The runtime contract and activation profiles are in
[Native Experimental Modules](../02%20-%20Implementation/07%20-%20Native%20Experimental%20Modules.md).

## Findings

The local mechanisms execute successfully on the resident compressed FP16 Qwen
prefix. The existing ingest scorer uses zero-based attention layer 5; the new
residual-based adapters use complete layer 4. Their old capture path would have
executed the deliberately removed final MLP/norm and expected an uncompressed
output-projection weight. Early exit at the selected layer and bounded lossless
weight restoration resolve that incompatibility without loading another model.

Native recall previously learned co-access edges but did not consume them.
The adapter projects those original chunk edges onto current atomic summary
addresses and publishes an immutable, bounded scalar snapshot. Learning refreshes
the published view; ongoing readers retain their earlier view. The optional
heat arm reuses the existing mass-conserving engine with these co-access
conductivities. It does not mislabel them as stored transformer head edges.

Baseline routing and hydration remain independent of experimental additions.
Additional evidence passes the same exact-span checks, under a separate budget.
Those additions retain original provenance but cannot earn their own learning
credit. Query model inputs contain the question and stored summaries only.

## Real archive and controls

Reuse the already-ingested engineering archive from
`eval_results/dialogue-neighborhood-20261006-r1/source.json`:
775,741 tokens, 481 messages, with authenticated native summary vectors and
parent projections. No histories were reingested, no summaries regenerated,
and no answer API was called. Original source-file hashes were checked before
and after each completed assay.

The first run used three existing engineering questions. The second used all
15 existing dialogue-wide questions with the current cap-8 control. Each query
ran six individual recall modules plus the combined configuration: **105
comparisons**. Every comparison preserved baseline routing and exact hydration.
Additional spans were checked against the original transcript bytes.

The archive's original consolidation tables were empty. The second run replayed
the three *observed baseline packet accesses* from the first run, once each,
into a disposable SQLite backup. This produced 115 qualifying edges across
21 summary nodes. It did not invent success labels, duplicate observations to
inflate support, or modify the original archive. This is a small development
co-access history, not evidence about mature graph quality.

| Recall arm | Cases | Mean warm recall | Maximum | Cases delivering additions | Mean added section tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current cap-8 control | 15 | 0.203 s | — | — | 0 |
| Learned associations | 15 | 0.203 s | 0.273 s | 0 | 0 |
| Heat diffusion | 15 | 0.204 s | 0.249 s | 0 | 0 |
| Qwen summary routing | 15 | 0.427 s | 0.529 s | 13 | 496 |
| Qwen association linking | 15 | 0.331 s | 0.449 s | 14 | 657 |
| Temporary live head memory | 15 | 0.811 s | 0.904 s | 13 | 555 |
| CAV association linking | 15 | 0.338 s | 0.436 s | 15 | 566 |
| Combined | 15 | 1.258 s | 1.310 s | 14 | 536 |

The learned graph proposed links in some cases, but none supplied additional
delivered evidence beyond the baseline's candidate set. Its unit tests exercise
and verify actual expansion outside that set. The real archive result is
redundancy, not an accuracy gain. Qwen-selected additions were sometimes rejected
by exact packing limits; receipts distinguish proposals from delivery.

A separate three-question run used the **actual automatic BGE tactic path**,
including its hierarchy context. All 21 comparisons preserved the classifier
decision, hierarchy context, baseline routes and hydration. Combined warm recall
averaged **2.378 seconds**, maximum **2.924 seconds**, against **1.148 seconds**
for the automatic control. The first version of this assay incorrectly compared
the classifier's per-call elapsed time; the corrected comparison excludes that
timing field while retaining all decision content. No product change was needed
for that harness correction.

These are retrieval timings. They exclude reader generation and model startup.
Cold Qwen preparation took 250.46 seconds on the first run and about 16–20 seconds
with the asset cache warm. The same resident instance handled every subsequent
query. A separate real Qwen weighting probe returned valid scalar activations
from stored summaries in **0.141 seconds warm**, one prefix pass. Its synthetic
probe node labels were not persisted as observations; production learning and
publication were exercised by integration tests.

## CAV compatibility

The old CAV report named the model but did not bind a training checkpoint hash.
Rather than claiming that artifact was admitted, refit the existing two-concept
dataset on current layer-4 FP16 residuals: 80 distinct authored examples,
1.84 seconds of fitting/encoding after model readiness. Save checkpoint, dataset
and artifact hashes in a new local bundle.

Held-out balanced accuracy was 93.75% for context dependency and 75% for binding
constraints. Bootstrap and shuffled-label controls are saved in the report.
These are small controlled concept-classification results, **not memory answer
accuracy**. The bank remains an explicit supplied artifact, not a release
default. The runtime checks checkpoint, precision, layer, hashes, dimensions and
finite vectors/thresholds before use.

## Validation and artifacts

- The configured release regression suite plus affected legacy model/heat tests
  passed **391 tests**. The final affected run passed **69 tests**, including
  **25 experiment tests**; these counts overlap.
- Tests cover invalid activation, dated source isolation, foreign proposals,
  exact hydration, corrupt extra spans, separate budgets, no routing raw reads,
  no recursive graph learning, immutable publication refresh, summary-only
  weighting, heat conservation, model-hook cleanup, CAV provenance and installed
  proxy CLI propagation. The experiment suite is included in CI.
- `eval_results/native-experiments-20261008-r1`: initial real-model assay and
  checkpoint-bound CAV refit.
- `eval_results/native-experiments-20261008-r2`: 15-question cap-8 comparison,
  sealed plan, disposable access replay, packets, scalar-learning probe, report
  and aggregate `summary.json`.
- `eval_results/native-experiments-20261008-auto-r2`: automatic-tactic comparison
  and the warm scalar-learning measurement.
- `eval_results/native-experiments-regression-r1.xml` and
  `eval_results/native-experiments-final-r1.xml`: test receipts.
- Final package-related checks passed **53 tests** after the CAV admission
  tightening. `pixi build --output-dir dist/native-experiments-validation`
  produced `memory_condense-0.2.2-pyh4616a5c_0.conda`, SHA-256
  `3b35b7105a5775d6e084f63eaeb4abd75ffd2448db6ec414f7b4668223e7f160`.
  Twelve affected runtime files matched source bytes in the extracted payload.
  An isolated import/CLI smoke passed while prohibiting research-tool/test
  imports. Receipt: `eval_results/native-experiments-package-smoke/report.json`.
  This reused development dependencies; it was not a fresh dependency install
  or a public release.

Reproduce the local comparison with:

```powershell
python tools/check_native_experiments.py --output eval_results/my-experiment-run --gpu --fit-cav --questions 15
```

Use `--retrieval-policy auto` for the complete automatic path and
`--replay-accesses PATH_TO_PRIOR_RUN` for an explicitly recorded development
co-access replay. The assay stops on a model error or baseline mismatch.

## Decision

Integrate the seven modules as opt-in native capabilities, retain the established
release defaults, and expose explicit receipts for unavailable inputs or failed
arms. The existing neighborhood and hierarchy settings remain available through
their prior controls. A matched reader comparison is required before claiming an
answer-quality gain or promoting a module. A 5M live graph/publication soak is
also still needed before claiming that optional graph refresh has the same
production scaling evidence as the existing baseline.
