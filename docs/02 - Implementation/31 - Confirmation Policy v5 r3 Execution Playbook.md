# Confirmation policy-v5-r3 execution playbook

The production confirmation workflow is two firewalled processes followed by
a separate evaluator. The raw benchmark exporter is not importable from the
prediction entrypoint, and the prediction dependency closure must not contain
gold, reference-answer, judge, or scorer capability.

## Boundaries

1. `tools/attest_confirmation_executor_v2.py` authenticates a clean committed
   apparatus and a provider-free offline test receipt.
2. `tools/export_confirmation_treatment_v5_r3.py` verifies that exact v2
   artifact before touching the raw dataset or split. It emits only the sealed
   sanitized 200-question treatment, uniform 20-by-10 preflight, and a minimal
   runtime-policy projection extracted from the full freeze.
3. `tools/run_confirmation_policy_v5_r3.py` accepts only those sanitized
   artifacts. It has no full-freeze path, executes the 17 prediction phases,
   and seals predictions.
4. The existing Sol judge lifecycle runs in a different process only after the
   prediction handoff exists. It is never a prediction-executor subcommand.

Every command below must use the SHA-256 printed by the immediately preceding
publication. Output roots belong under ignored `eval_results/` or `data/`
locations; a readiness artifact is invalid if the committed worktree changes.

## Attest, then export sanitized treatment

From a clean committed tree:

```powershell
.pixi\envs\dev\python.exe -m tools.attest_confirmation_executor_v2 test-receipt `
  --output eval_results/confirmation-policy-v5-r3/offline-tests.json

.pixi\envs\dev\python.exe -m tools.attest_confirmation_executor_v2 attest `
  --offline-test-receipt eval_results/confirmation-policy-v5-r3/offline-tests.json `
  --output eval_results/confirmation-policy-v5-r3/executor-readiness-v2.json
```

The raw dataset and split paths occur only in this standalone export process:

```powershell
.pixi\envs\dev\python.exe -m tools.export_confirmation_treatment_v5_r3 `
  --repository-root . `
  --readiness eval_results/confirmation-policy-v5-r3/executor-readiness-v2.json `
  --expected-readiness-sha256 <READINESS_SHA256> `
  --expected-policy-manifest-sha256 <POLICY_SHA256> `
  --policy-manifest 'docs/10 - Research Log/data/policy-v5-r3-confirmation-freeze-v1.json' `
  --output-root eval_results/confirmation-policy-v5-r3/treatment `
  --dataset <RAW_DATASET_PATH> `
  --split-manifest <LOCKED_SPLIT_PATH>
```

If readiness verification fails, neither raw path is resolved, statted, nor
opened. The exporter seals a treatment artifact, its filename-bearing sidecar,
the uniform namespace preflight, the runtime policy and its sidecar, and an
export receipt with zero provider calls. The runtime policy contains the exact
frozen treatment policy and source-freeze SHA, but no validation lineage,
validation result, miss ordinals, artifact paths, gold, reference, or judge
material.

## Initialize the prediction run

The following arguments are common to `init`, `status`, both granular advance
commands, `run-authorized`, and `publish-prediction-handoff`:

```powershell
$common = @(
  '--repository-root', '.',
  '--readiness', 'eval_results/confirmation-policy-v5-r3/executor-readiness-v2.json',
  '--expected-readiness-sha256', '<READINESS_SHA256>',
  '--expected-policy-manifest-sha256', '<POLICY_SHA256>',
  '--output-root', 'eval_results/confirmation-policy-v5-r3/run',
  '--runtime-policy', 'eval_results/confirmation-policy-v5-r3/treatment/confirmation-runtime-policy-v1.json',
  '--expected-runtime-policy-sha256', '<RUNTIME_POLICY_SHA256>',
  '--treatment-input', 'eval_results/confirmation-policy-v5-r3/treatment/confirmation-treatment-input-v1.json',
  '--expected-treatment-input-sha256', '<TREATMENT_SHA256>',
  '--treatment-preflight', 'eval_results/confirmation-policy-v5-r3/treatment/confirmation-treatment-pipeline-preflight-v1.json',
  '--expected-treatment-preflight-sha256', '<PREFLIGHT_SHA256>',
  '--qwen-prefix-model-dir', '.cache/models/Qwen3-8B',
  '--qwen-choice-model-dir', '.cache/models/Qwen3-0.6B',
  '--api-key-env', 'LITELLM_KEY'
)

& .pixi\envs\dev\python.exe -m tools.run_confirmation_policy_v5_r3 init @common
```

Initialization verifies v2 before opening the sanitized treatment and runtime
policy, then seals an immutable run manifest. The manifest separately binds
the source policy-freeze SHA and the sanitized runtime-policy SHA, arbitrary
population size, the namespace schedule, 1,000,000 target memory tokens per
namespace, model paths, the cumulative phase DAG, the exact ordered
production-adapter identity SHA-256 for every phase, and retry limit zero.
Those identities are derived from the default production adapter population
for the sealed runtime policy and paths; injected or synthetic identities are
not interchangeable with them.

## Fast single-process execution

The preferred operational path retains reconstructed memory objects and model
ownership between phases:

```powershell
& .pixi\envs\dev\python.exe -m tools.run_confirmation_policy_v5_r3 `
  run-authorized @common --approve-all-exact-provider-releases
```

The master flag is an explicit provider opt-in, not a numerical allowance. At
each provider boundary the executor first computes the authenticated remaining
journal count, then authorizes exactly that count for only that phase. Each
phase still publishes its native release and executor accounting receipt. Any
orphan journal, incomplete request/response pair, changed artifact, provider
error, or accounting mismatch stops the run immediately. Provider client
construction lazily loads `.env` without printing the credential.

## Granular recovery

These commands remain available for inspection and recovery:

```powershell
& .pixi\envs\dev\python.exe -m tools.run_confirmation_policy_v5_r3 `
  advance-provider-free @common

& .pixi\envs\dev\python.exe -m tools.run_confirmation_policy_v5_r3 `
  advance-provider @common --authorized-provider-calls <EXACT_REMAINING_FROM_PRIOR_OUTPUT>

& .pixi\envs\dev\python.exe -m tools.run_confirmation_policy_v5_r3 status @common
```

`advance-provider-free` retains one process while advancing local phases and
stops at the next provider release. `advance-provider` accepts only the exact
current remainder; over-authorization and under-authorization both fail.
Completed immutable journals count toward the cumulative physical-call total
after a restart, while the checkpoint also records how many calls occurred in
the invocation that finalized that phase.

## Seal the evaluator handoff

```powershell
& .pixi\envs\dev\python.exe -m tools.run_confirmation_policy_v5_r3 `
  publish-prediction-handoff @common
```

This command fails unless every prediction phase is sealed. Its handoff binds
the exact prediction artifact, all phase checkpoints, the ordered question
population, cumulative Terra accounting, zero retries, and the claim that no
gold/reference or evaluator process was opened. Status, resume, and handoff
publication reject any checkpoint whose adapter identity differs from its
manifest phase binding. Only then may a separate Sol judge process receive
prediction plus reference.

The evaluator does not accept a raw prediction path or digest. It authenticates
the externally pinned handoff, fixed-name run manifest, all 17 checkpoint files
and dependency edges, every checkpoint's manifest-bound production-adapter
identity, per-phase provider receipts, final artifact binding, and aggregate
accounting before it opens the dataset:

```powershell
.pixi\envs\dev\python.exe -m tools.confirmation_gold_judge_scaffold compile-plan `
  --policy-manifest '<POLICY_MANIFEST_PATH>' `
  --expected-policy-manifest-sha256 '<POLICY_SHA256>' `
  --treatment-input 'eval_results/confirmation-policy-v5-r3/treatment/confirmation-treatment-input-v1.json' `
  --expected-treatment-input-sha256 '<TREATMENT_SHA256>' `
  --treatment-preflight 'eval_results/confirmation-policy-v5-r3/treatment/confirmation-treatment-pipeline-preflight-v1.json' `
  --expected-treatment-preflight-sha256 '<PREFLIGHT_SHA256>' `
  --prediction-handoff 'eval_results/confirmation-policy-v5-r3/run/confirmation-policy-v5-r3-prediction-handoff-v1.json' `
  --expected-prediction-handoff-sha256 '<PREDICTION_HANDOFF_SHA256>' `
  --dataset '<RAW_DATASET_PATH>' `
  --split-manifest '<LOCKED_SPLIT_PATH>' `
  --exposure-audit '<EXPOSURE_AUDIT_PATH>' `
  --expected-exposure-audit-sha256 '<EXPOSURE_AUDIT_SHA256>' `
  --expected-exposed-count 15 `
  --expected-ordered-exposed-ids-sha256 '<ORDERED_EXPOSED_IDS_SHA256>' `
  --output 'eval_results/confirmation-policy-v5-r3/judge/confirmation-sol-judge-plan-v1.json'
```

Only the resulting judge-plan SHA enters the separately authorized Sol
lifecycle. Its `preflight`, `approve-release`, `provider-run`, `materialize`,
and `replay` commands remain the sole provider path. The scaffold itself makes
zero provider calls. For confirmation200, the exact authorized count printed
by preflight is 200:

```powershell
$judge = @(
  '--judge-plan', 'eval_results/confirmation-policy-v5-r3/judge/confirmation-sol-judge-plan-v1.json',
  '--expected-judge-plan-sha256', '<JUDGE_PLAN_SHA256>',
  '--output-root', 'eval_results/confirmation-policy-v5-r3/judge/sol'
)

& .pixi\envs\dev\python.exe -m tools.confirmation_sol_judge_lifecycle preflight @judge

& .pixi\envs\dev\python.exe -m tools.confirmation_sol_judge_lifecycle `
  approve-release @judge `
  --expected-preflight-sha256 '<SOL_PREFLIGHT_SHA256>' `
  --approve-provider-release --authorized-provider-calls 200

& .pixi\envs\dev\python.exe -m tools.confirmation_sol_judge_lifecycle `
  provider-run @judge `
  --expected-preflight-sha256 '<SOL_PREFLIGHT_SHA256>' `
  --expected-release-sha256 '<SOL_RELEASE_SHA256>' `
  --enable-provider --authorized-provider-calls 200 --api-key-env LITELLM_KEY

& .pixi\envs\dev\python.exe -m tools.confirmation_sol_judge_lifecycle `
  materialize @judge `
  --expected-preflight-sha256 '<SOL_PREFLIGHT_SHA256>' `
  --expected-release-sha256 '<SOL_RELEASE_SHA256>'

& .pixi\envs\dev\python.exe -m tools.confirmation_sol_judge_lifecycle `
  replay @judge `
  --expected-preflight-sha256 '<SOL_PREFLIGHT_SHA256>' `
  --expected-release-sha256 '<SOL_RELEASE_SHA256>' `
  --expected-completion-sha256 '<SOL_COMPLETION_SHA256>' `
  --expected-results-sha256 '<SOL_RESULTS_SHA256>'

& .pixi\envs\dev\python.exe -m tools.confirmation_gold_judge_scaffold score `
  --judge-plan 'eval_results/confirmation-policy-v5-r3/judge/confirmation-sol-judge-plan-v1.json' `
  --expected-judge-plan-sha256 '<JUDGE_PLAN_SHA256>' `
  --judge-results 'eval_results/confirmation-policy-v5-r3/judge/sol/confirmation-sol-judge-results-v1.json' `
  --expected-judge-results-sha256 '<SOL_RESULTS_SHA256>' `
  --exposure-audit '<EXPOSURE_AUDIT_PATH>' `
  --expected-exposure-audit-sha256 '<EXPOSURE_AUDIT_SHA256>' `
  --output 'eval_results/confirmation-policy-v5-r3/judge/confirmation-score-report-v1.json'
```

## Offline verification status

The executor, first-half production adapters, final provider plumbing,
readiness-first export boundary, lazy runtime ownership, exact accounting, and
synthetic full-DAG paths have focused provider-free tests. No confirmation
treatment or provider was accessed while building this apparatus. A production
v2 readiness artifact must not be published until the full recursive
prediction dependency firebreak and interrupted-provider resume tests pass from
the final clean committed tree.
