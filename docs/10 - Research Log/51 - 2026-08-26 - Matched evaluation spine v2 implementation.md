# Matched evaluation spine v2 implementation

**Date:** 2026-08-26

**Status:** DECISION 2 IMPLEMENTED — provider-free migration, fresh S0-v2
preflight, and live answer/judge path complete; the measured result is in
Research Log 52

**Cost:** the implementation checkpoint itself used $0 and zero calls; the
subsequent live control used exactly 100 Terra and 100 Sol calls, with monetary
cost not reported

**Source retrieval:** locked 100-question artifact
`e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f`

## Outcome

Decision 2 is now executable. Historical S0, EM, and CAV results remain
explicitly quarantined as legacy-renderer observations, while all new matched
runs have one versioned typed-slot renderer, one isolated/cumulative runner,
one runtime ledger, and one posthoc score ledger. A fresh S0-v2 population was
projected directly from the sealed locked-100 retrieval without rebuilding the
million-token corpus. The later live execution completed at 53/100 and is
documented separately in Research Log 52.

This checkpoint never claimed that S0-v2 inherited the historical 57/100. The
values 57/60/53 were reproduced from sealed historical journals whose three
prompt templates differ. They remain migration controls, not common-renderer
causal comparisons. The subsequent fresh S0-v2 control scored 53/100; that
new plain-S0 result must not be confused with the legacy CAV arm's coincidentally
equal score.

## What was built

The implementation lives under `tools/matched_eval/` so importing the legacy
checkpoint does not alter the historical implementation digest over
`src/memory_condense/`.

| Boundary | Implementation | Enforced behavior |
| --- | --- | --- |
| artifact | `artifacts.py` | canonical JSON, exact SHA sidecar, publish-once/idempotent reuse |
| snapshot and deltas | `contracts.py` | immutable typed tuples, gold firewall, exact token counts, disabled eval learning |
| renderer | `renderer.py` | `matched_typed_slots_v2`, one stable system message and one final user message |
| runner | `runner.py` | isolated star or cumulative line, non-borrowable stage budgets, exact no-op packets, final 8k render check |
| runtime/score join | `ledger.py` | gold-blind runtime plane, posthoc score plane, verified ordered row join and exact provenance |
| legacy quarantine | `legacy.py` | pinned import of 12 run/replay/judge artifacts with zero calls |
| S0 adapter | `population.py` | narrow sealed-boundary projection of the existing retrieval into S0-v2 prompts |
| command | `tools/run_matched_eval_spine.py` | migration, S0-v2 preflight, authorized answer/judge runs, and zero-call replays |

The S0 adapter validates the sealed file and sidecar, population and question
order, embedded question-part hashes, receipt self-seals, cumulative parent
prefixes, selected-then-added suffixes, and the protected S0 context. It does
not reconstruct the source store, embeddings, corpus, candidate models, or the
historical retrieval interior.

## Typed mechanism semantics

The common envelope does not pretend that every mechanism retrieves another
row.

- Membership stages add ordered raw evidence and cannot evict or duplicate the
  protected parent.
- EM representation records raw neighborhood candidates, selection, the
  post-selection dedup basis, non-admission, and admitted raw IDs. Fact IDs are
  a separate namespace. Facts cite the admitted raw IDs even when those rows
  are not rendered as raw additions.
- CAV linking binds current evidence and adds exactly zero evidence membership.
- Answer operators add instructions but no facts.
- Observation stages advance logical lineage only; evaluation cannot reheat or
  learn consolidation state.

A failed, invalid, or overflowing stage preserves the exact material parent
packet. Its receipt still advances the logical plan lineage and retains
candidate/selection/non-admission information. Every accepted packet is
rendered immediately, so under-reported mechanism tokens cannot evade the
8,000-token final-prompt ceiling.

## Legacy migration result

The migration command reproduced the sealed checkpoint with no provider
access:

| Arm | Historical renderer | Result | Historical calls represented |
| --- | --- | ---: | ---: |
| `S0_CONTROL` | `legacy_renderer/s0_qa_v1` | 57/100 | 100 Terra + 100 Sol |
| `S0_PLUS_EM_FACTS` | `legacy_renderer/em_facts_v1` | 60/100 | 100 Terra compression + 62 Terra answers + 43 Sol |
| `S0_PLUS_CAV_LINKS` | `legacy_renderer/cav_links_v1` | 53/100 | 100 Terra answers + 31 Sol + four shared local Qwen batches |

Common-ledger totals are 300 runtime rows over 100 questions, 362 historical
Terra calls, 174 historical Sol judge calls, four historical local Qwen
batches, and zero new calls. Runtime provenance contains only run/run-replay
artifacts. Judge/judge-replay artifacts exist only in the score plane. Raw
source-row hashes, judge-row hashes, verdict hashes, and reused baseline-row
hashes remain distinct.

Sealed outputs:

```text
legacy-migration.json       c5575153341ed222a1e201533ccf9b06a91dff618c9369c816bea7114c419534
legacy-runtime-ledger.json  9366cfd093dd077c218122f142eca8981877ed5e47edde228dc12a6f99bcf920
legacy-score-ledger.json    539bc46004eec3cb2d75d84574cdfe452611ae98b1ecadc2c67e1777220ea7dc
runtime ledger identity     3140f5beadc8bb7d808a8145fc51c53eb03a9ab8908aef9a1a0381dbe9f8e9a5
score ledger identity       5e7addf8bb51ae99dc5208ef33324ff364e48550ec804e49f871dd7973642bb7
```

The score ledger binds runtime identity
`3140f5beadc8bb7d808a8145fc51c53eb03a9ab8908aef9a1a0381dbe9f8e9a5`
exactly. A second migration reused all three artifacts byte-for-byte. The
measured wall times were 1.519 seconds for creation and 1.367 seconds for the
idempotent replay.

## Fresh S0-v2 prompt population

The common renderer produced 100 logical and 100 unique prompts. The largest
prompt proxy is 5,525 tokens under the hard 8,000-token cap. No protected S0
row was evicted; this includes one exact empty excerpt and source excerpts with
meaningful boundary whitespace.

```text
matched population identity  886e14025a0aedf5a9ba673be8ffc9183acc080b97645adc2b6dd003019438bf
prompt population identity   412b54912511fde49de02395efd3a406dff6009db323cfb4e69de16bff0eea15
preflight behavior identity  d3b23d0b2431ef94800bb94d980227c4d3c6b74699c48b0d7ffd98b5d2ce7ba5
s0-v2-preflight.json         96c109c64fbf6232e4cfa3fbc252aa8a008624d1e1bffe29ddbf0222d8f6e315
```

Creation and byte-identical replay each took about 23.3 seconds. That time
reads and validates the existing 23,303,384-byte merged retrieval and renders
the complete population; it performs no corpus, index, embedding, local-model,
or provider work.

## Subsequent live-control update

The exact preflight was later executed through the new answer and score planes:

- 100 unique Terra calls produced the sealed common-renderer answers;
- zero-call answer replay reproduced the answer and runtime-ledger bytes;
- only after that verification, 100 unique Sol calls judged the dated
  question, reference, and sealed prediction;
- zero-call judge replay reproduced both the judge and score-ledger bytes; and
- the final result was **53/100 semantic**, 27/100 normalized exact match, and
  0.410760 mean F1.

The identical retrieval/source-stage receipts and changed 100/100 provider
prompts isolate the legacy 57→common-v2 53 loss to renderer/answering behavior.
The renderer moved the question away from the generation boundary, expanded
mean prompt proxy by 1,888.24 tokens through a metadata-heavy typed surface,
removed parts of the proven role/temporal/calculation policy, and dropped the
terminal short-answer cue. Full paired outcomes, hashes, and the renderer-v3
decision are in
[Research Log 52](52%20-%202026-08-26%20-%20Matched%20S0-v2%20live%20control%20result.md).

## Verification

The final focused suite passed 62 tests in 24.49 seconds. A second 49-test
regression slice covering the existing completion runtime and the historical
S0/EM/CAV run/judge loaders passed in 17.51 seconds. Both runs used a scoped
workspace `--basetemp` and made no provider calls. The live answer and judge
artifacts were then replayed again at 53/100 with zero calls and the same four
answer/runtime/judge/score hashes.

```powershell
.pixi/envs/dev/python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .agent_test_tmp/matched-core-final `
  tests/test_matched_eval_artifacts.py `
  tests/test_matched_eval_contracts.py `
  tests/test_matched_eval_renderer.py `
  tests/test_matched_eval_runner.py `
  tests/test_matched_eval_ledger.py `
  tests/test_matched_eval_legacy.py `
  tests/test_matched_eval_population.py `
  tests/test_matched_eval_live_execution.py `
  tests/test_run_matched_eval_spine.py

.pixi/envs/dev/python.exe tools/run_matched_eval_spine.py migrate-legacy
.pixi/envs/dev/python.exe tools/run_matched_eval_spine.py s0-v2-preflight
.pixi/envs/dev/python.exe tools/run_matched_eval_spine.py inspect
```

Generated artifacts are under
`eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/matched-eval-spine-v2/`
and remain ignored experiment outputs.

## Next measured step

The live control now exists and exposes a renderer regression. The immediate
next operation is a renderer-v3 ablation: retain the typed packets and common
ledger, but use compact provider aliases, put the dated question back at the
generation boundary, restore the proven role/update/approximation/ordering/
calculation rules, and restore the terminal short-answer cue. Re-establish S0
before EM, representative bridge, artifact-global, robust Hebbian, and CAV
adapters are run separately and then recombined through the same runner. No
mechanism should introduce another population loader, renderer, completion
journal, or score schema.

The 95% objective, an untouched confirmation population, true responder-side
CAV activation reinjection, and the fair Mem0 comparison all remain open.
