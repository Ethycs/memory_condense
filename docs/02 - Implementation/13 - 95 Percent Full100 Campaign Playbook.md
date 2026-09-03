# 95% full100 campaign playbook

**Status:** evidence-conserving R7, terminal-v5 full100 construction/replay, and
the 26/26 post-seal promotion gate are sealed; historical v2/v3 constructions
and scores remain frozen. Terra answer and Sol judge lifecycles are sealed and
byte-replayed. The terminal-v5 base campaign scored 88/100. The subsequent
proof-carrying `policy-v5-r2` overlay and differential judge merge scored
92/100. The `operator-material-v3` frontier successor and receipt-bound
`policy-v5-r3` overlay scored **95/100** and pass the validation100 promotion
target. Confirmation200 remains required before a final generalization claim.

This is the authoritative end-to-end procedure for testing a retrieval-policy
successor against the locked LongMemEval-S validation population. It unifies
the previously separate construction, Terra-answer, Sol-judge, and score
lifecycles.

The campaign is not a fresh ingest. It reopens the ten authenticated resident
stores that together represent 100 locked questions over ten approximately
1M-token namespaces. The fixed exact-11 promotion population is projected from
authenticated full100 namespace sidecars; it no longer reopens those stores in
a separate construction.

## Claim boundary

The validation gate is exactly 100 unique questions from the sealed
`longmemeval-s-1m-100q-95-v1` profile. Passing requires at least 95 accepted Sol
judgments. The exact-11 atom audit is a promotion gate and can never establish
the 95% claim. A validation100 pass freezes the policy but still requires a later
untouched confirmation200 campaign before a final generalization claim.

A proof-carrying answer is exact only relative to the authenticated finite
store, the versioned fact/identity/state grammar, the declared action or
operator, and a valid closure certificate. It does not prove that the grammar
covers arbitrary natural-language questions, that an open frontier is
complete, or that the routing/retrieval policy is globally optimal.

The active successor described here is:

```text
terminal compilation mode: v5-linked-backfill
format: memory-condense-semantic-global-terminal-compilation-v5
answer model: codex_sdk/gpt-5.6-terra
judge model: codex_sdk/gpt-5.6-sol
gateway: https://central-dev.zt:4000/v1
retries: 0
max concurrency: 4
```

Never write v5 output into a historical v2/v3 root. Never replace a historical
construction, answer, judge, score, replay, journal, or SHA sidecar.

## Fixed roots

Use explicit roots so every lifecycle stays separate from frozen artifacts:

```powershell
$python = '.pixi\envs\dev\python.exe'
$r7Tool = 'tools\run_locked_semantic_residual_construction_v4.py'
$auditTool = 'tools\audit_locked_semantic_global_terminal_postseal.py'
$fullTool = 'tools\run_locked_semantic_global_terminal_full100_resumable.py'
$answerTool = 'tools\run_locked_semantic_global_terminal_full100_answer.py'
$judgeTool = 'tools\run_locked_semantic_global_terminal_full100_judge.py'

$r7Root = 'eval_results\matched_eval_100\locked-semantic-residual-v4-r7-evidence-conserving-r1'
$r7Artifact = Join-Path $r7Root 'locked-semantic-residual-construction-v4.json'
$fullRoot = 'eval_results\matched_eval_100\locked-semantic-global-terminal-full100-v5-resumable-r1'
$answerRoot = 'eval_results\matched_eval_100\locked-semantic-global-terminal-full100-terra-answer-v5-r1'
$judgeRoot = 'eval_results\matched_eval_100\locked-semantic-global-terminal-full100-terra-answer-v5-r1\sol-judge-v1'
$audit = Join-Path $fullRoot 'semantic-global-terminal-postseal-fact-audit-v2.json'
$dataset = 'C:\Users\Keytone\Downloads\memory-condense-rig\datasets\longmemeval_s_cleaned.json'
$split = 'docs\10 - Research Log\data\longmemeval-95-target-split-v2.json'
```

Every new root must be absent before its first construct or preflight. A
resumable construction or partially completed provider campaign resumes only
through its authenticated checkpoint scanner; do not delete checkpoints or
journals and restart under the same name.

## Sealed authority and comparison point

The campaign inherits these frozen parents; changing one starts a different
campaign:

| Authority | SHA-256 |
|---|---|
| locked dataset file | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| validation split | `8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4` |
| merged retrieval | `e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f` |
| R7 gate | `779c711e090ecb9faad92d9845158d939411dfa3a965669a26cfe8a8062fb912` |
| historical R7 construction (legacy pruning; control only) | `d0f226b1577a6bf40c54758d2fdc477ab98483613ca7c4fc77ef93383a651f6a` |
| active R7 evidence-conserving successor | `6cd26b55092d0a93aca1afc5209874a1bb7ebf7927a805e4f2d8b274fb48f8e3` |
| R7 vectors and replay | `ce9b10803146a70ec18d9c907aceb2fa469fa5491818bc72721e7f5cefbcc8e2` |
| V3 parent answer | `07c6f3125e65094880384c1c1c6f7d9be0600475f1fe58d050796fc0f48493d1` |
| terminal-v5 full100 preflight | `c8373ef198fc5b360f9da70c0c6b366fd93aef01280adfc4dd6243ca51ae8277` |
| terminal-v5 full100 construction | `57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00` |
| terminal-v5 26/26 promotion audit | `65285b9db760cb649e621465492ff0c323c9449c0b2735a34dd8bd70f23cf369` |
| terminal-v5 Terra answer preflight | `0c4464cf288b93f814991fd7abc2d74d76c5ce7396ae8829eeac43d1ec38f289` |
| terminal-v5 Terra provider release | `2b9bb5741afe18e4b9c631b0e6ec2bb4d4dd2ee5e1d6f3b5630b70cbb5b4a5d7` |
| terminal-v5 Terra answer run | `f1d774e98f48758b8ced70be05064e0af0aa538f9673f7744f0df8607ba54946` |
| terminal-v5 Terra answer replay | `2cbb053f31c2ba713a9fa16819b4bd8d007ff5b758da99e919b4c6d8795f1d41` |
| terminal-v5 Sol judge preflight | `34099ddb56fa2c2e2ba5d42c50cbe4cb142c16e94a95ed4183d91b507f80c8b9` |
| terminal-v5 Sol provider release | `aea418d667daec86899636aaebfec09406f67123df6db994547769aca0e83573` |
| terminal-v5 Sol judge and replay | `edccbd49a20bf92fcb52306fe28557eeccb8ebba69e9e12a26d5d6cc5d530239` |
| terminal-v5 score and replay (88/100) | `91ae36ebb7ef48fb914f7236ca03998adb0b22f58d98c29bfa8ecccd3739dce1` |

The strongest comparable development score is V3 at 89/100; V2 scored
79/100. This terminal-v5 campaign scored 88/100. The prior linked exact-11
diagnostic scored 6/11, versus the earlier 5/11 best. The unrelated historical
closure-selector label `V5` is not this terminal-v5 policy.

The post-seal `policy-v5-r2` follow-up scored 92/100. It is a new,
receipt-bound answer-policy and judging lineage over the frozen terminal-v5
inputs, not a relabeling or overwrite of the 88/100 base campaign.

The active `policy-v5-r3` successor scored 95/100. It preserves the same
terminal-v5 inputs and adds only a proof-carrying numeric-frontier profile;
the base, r2, and all historical roots remain frozen.

Commit `2124f98` changed residual branch rejection to evidence-conserving
fail-open behavior. That treatment cannot byte-replay the historical R7
construction for ordinal 94, which legitimately expands from 709 retained plus
44 pruned leaves to 753 retained plus zero pruned leaves. Gate 1 therefore seals
a new R7 construction under an authenticated successor classifier policy and
passes that exact root/SHA through full100, promotion, and answer commands.
Never select legacy pruning merely to satisfy the historical R7 hash for this
post-repair campaign.

## Gate 0: apparatus and connectivity

Before construction:

1. Run the relevant provider-free regression suites and `git diff --check`.
2. Record the current `git rev-parse HEAD`, dirty-file inventory, and intended
   terminal mode in the research log.
3. Confirm the v5 roots above are absent.
4. Confirm `LITELLM_KEY` is populated without printing it.
5. Perform a zero-completion `GET /v1/models` through the same OpenAI,
   HTTPX, truststore, gateway, and `max_retries=0` transport. Do not use a
   campaign `provider-run` as a connectivity probe because it owns sealed
   checkpoint state.

The model catalog may omit the private `codex_sdk` routes. A route-level smoke
therefore requires a separate one-call Terra and one-call Sol namespace if
needed; those calls are not campaign calls and must be logged separately.

## Gate 1: evidence-conserving R7

Seal the provider-free R7 successor once. Its policy explicitly identifies
`evidence-conserving-fail-open`; the historical fieldless policy remains the
legacy-pruning control.

```powershell
& $python $r7Tool construct `
  --output-root $r7Root `
  --expected-gate-sha256 779c711e090ecb9faad92d9845158d939411dfa3a965669a26cfe8a8062fb912 `
  --expected-vector-sha256 ce9b10803146a70ec18d9c907aceb2fa469fa5491818bc72721e7f5cefbcc8e2 `
  --residual-classifier-mode evidence-conserving-fail-open
```

The sealed campaign artifact is `6cd26b55092d0a93aca1afc5209874a1bb7ebf7927a805e4f2d8b274fb48f8e3`.
A separate deep R7 replay is omitted from the streamlined route: Gate 2
reexecutes and binds every one of the 68 eligible R7 question rows while
building the downstream V6/V7/terminal population.

## Gate 2: resumable full100 v5 construction

Use the resumable constructor, not the one-shot resident command:

```powershell
& $python $fullTool construct `
  --output-root $fullRoot `
  --terminal-compilation-mode v5-linked-backfill `
  --r7-construction $r7Artifact `
  --expected-r7-construction-sha256 6cd26b55092d0a93aca1afc5209874a1bb7ebf7927a805e4f2d8b274fb48f8e3
```

It authenticates the fixed gate and upstream R7/vector/V3 artifacts, processes
each resident namespace once, and seals one checkpoint per namespace. A resume
uses the same command and root; complete matching checkpoints are reused and
partial, foreign, or changed state fails closed.

The first measured build sealed R7 at 2026-09-02 00:35:54 PDT, full100
preflight at 01:48:24, and full100 construction at 04:53:24. Construction took
3 h 05 m. Final namespace artifacts plus sidecars occupy 2.342 GiB and the
resumable checkpoints plus sidecars occupy another 2.342 GiB (4.684 GiB
total). This is provider-free serial replay and duplicated audit-payload I/O;
it is not provider latency. A future implementation should store the large
audit payload once and make checkpoints/final manifests reference it.

After construction:

```powershell
& $python $fullTool replay `
  --output-root $fullRoot `
  --terminal-compilation-mode v5-linked-backfill `
  --r7-construction $r7Artifact `
  --expected-r7-construction-sha256 6cd26b55092d0a93aca1afc5209874a1bb7ebf7927a805e4f2d8b274fb48f8e3 `
  --expected-construction-output-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00
```

Require byte-identical construction/replay, 100 ordered rows, 68 terminal
plans, 32 V3 passthroughs, ten namespace checkpoints, zero provider calls, and
zero retained transformer-token state.

Then project the fixed ordinals `14 28 40 49 53 54 67 69 82 94 97` from the
authenticated full100 sidecars and prove all 26 declared semantic atoms survive
the final provider packet:

```powershell
& $python $auditTool `
  --terminal-root $fullRoot `
  --expected-construction-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00 `
  --expected-replay-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00 `
  --promotion-from-full100 `
  --r7-construction $r7Artifact `
  --expected-r7-construction-sha256 6cd26b55092d0a93aca1afc5209874a1bb7ebf7927a805e4f2d8b274fb48f8e3 `
  --witness-manifest 'docs\10 - Research Log\data\longmemeval-exact11-target-witness-manifest-v1.json' `
  --expected-witness-manifest-sha256 f6add6368971d9b0b827bc0042c5e2a2e409f26df4f2a30ef18224c34c64bd60 `
  --semantic-atom-manifest 'docs\10 - Research Log\data\longmemeval-exact11-semantic-atom-manifest-v1.json' `
  --expected-semantic-atom-manifest-sha256 c40bbfc78f07eccbd6b2e489b79f4ad1ba5221dea2aeb707c64ecf84ac514008 `
  --expected-semantic-atom-manifest-identity-sha256 f3e8ad4975d953eac16a98003626d7fb3ebc39b4a335e6fcea703e40f487c69c `
  --expected-semantic-atom-population-sha256 e2a13b57f44f4b863df22b7d7e906bb6cd74e15c9b895add37bface21907c73c `
  --output $audit `
  --promotion-gate
```

Do not release provider calls unless `promotion_gate_passed` is true,
`semantic_atom_final_usable_count` is 26, and the artifact has its matching SHA
sidecar. The eleven full plans must equal their compact provider projections in
the same full100 population. No separate exact-11 construction is needed.

## Gate 3: Terra answer lifecycle

Preflight binds the full100 v5 replay and its directly projected exact-11 atom
audit to the same canonical root and SHA pair:

```powershell
& $python $answerTool preflight `
  --output-root $answerRoot `
  --full100-terminal-root $fullRoot `
  --expected-full100-construction-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00 `
  --expected-full100-replay-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00 `
  --promotion-terminal-root $fullRoot `
  --expected-promotion-terminal-construction-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00 `
  --expected-promotion-terminal-replay-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00 `
  --promotion-from-full100 `
  --r7-construction $r7Artifact `
  --expected-r7-construction-sha256 6cd26b55092d0a93aca1afc5209874a1bb7ebf7927a805e4f2d8b274fb48f8e3 `
  --postseal-audit $audit `
  --expected-postseal-audit-sha256 65285b9db760cb649e621465492ff0c323c9449c0b2735a34dd8bd70f23cf369

& $python $answerTool approve-release `
  --output-root $answerRoot `
  --full100-terminal-root $fullRoot `
  --expected-full100-construction-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00 `
  --expected-full100-replay-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00 `
  --promotion-terminal-root $fullRoot `
  --expected-promotion-terminal-construction-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00 `
  --expected-promotion-terminal-replay-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00 `
  --promotion-from-full100 `
  --r7-construction $r7Artifact `
  --expected-r7-construction-sha256 6cd26b55092d0a93aca1afc5209874a1bb7ebf7927a805e4f2d8b274fb48f8e3 `
  --postseal-audit $audit `
  --expected-postseal-audit-sha256 65285b9db760cb649e621465492ff0c323c9449c0b2735a34dd8bd70f23cf369 `
  --expected-preflight-sha256 0c4464cf288b93f814991fd7abc2d74d76c5ce7396ae8829eeac43d1ec38f289 `
  --approve-provider-release
```

A fresh provider run authorizes exactly 68 calls:

```powershell
& $python $answerTool provider-run `
  --output-root $answerRoot `
  --postseal-audit $audit `
  --expected-postseal-audit-sha256 65285b9db760cb649e621465492ff0c323c9449c0b2735a34dd8bd70f23cf369 `
  --expected-preflight-sha256 0c4464cf288b93f814991fd7abc2d74d76c5ce7396ae8829eeac43d1ec38f289 `
  --expected-release-sha256 2b9bb5741afe18e4b9c631b0e6ec2bb4d4dd2ee5e1d6f3b5630b70cbb5b4a5d7 `
  --enable-provider `
  --authorized-provider-calls 68
```

On resume, authorize exactly `68 - authenticated checkpoint hits`; never reuse
68 after partial completion. Then run the two provider-free phases:

```powershell
& $python $answerTool materialize `
  --output-root $answerRoot `
  --postseal-audit $audit `
  --expected-postseal-audit-sha256 65285b9db760cb649e621465492ff0c323c9449c0b2735a34dd8bd70f23cf369 `
  --expected-preflight-sha256 0c4464cf288b93f814991fd7abc2d74d76c5ce7396ae8829eeac43d1ec38f289 `
  --expected-release-sha256 2b9bb5741afe18e4b9c631b0e6ec2bb4d4dd2ee5e1d6f3b5630b70cbb5b4a5d7

& $python $answerTool replay `
  --output-root $answerRoot `
  --full100-terminal-root $fullRoot `
  --expected-full100-construction-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00 `
  --expected-full100-replay-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00 `
  --promotion-terminal-root $fullRoot `
  --expected-promotion-terminal-construction-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00 `
  --expected-promotion-terminal-replay-sha256 57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00 `
  --promotion-from-full100 `
  --r7-construction $r7Artifact `
  --expected-r7-construction-sha256 6cd26b55092d0a93aca1afc5209874a1bb7ebf7927a805e4f2d8b274fb48f8e3 `
  --postseal-audit $audit `
  --expected-postseal-audit-sha256 65285b9db760cb649e621465492ff0c323c9449c0b2735a34dd8bd70f23cf369 `
  --expected-preflight-sha256 0c4464cf288b93f814991fd7abc2d74d76c5ce7396ae8829eeac43d1ec38f289 `
  --expected-release-sha256 2b9bb5741afe18e4b9c631b0e6ec2bb4d4dd2ee5e1d6f3b5630b70cbb5b4a5d7 `
  --expected-run-sha256 f1d774e98f48758b8ced70be05064e0af0aa538f9673f7744f0df8607ba54946
```

Require the run and replay to be byte-identical and to expose 100 ordered judge
rows. Invalid Terra completions fall back to their sealed parent prediction;
they are not silently accepted as terminal repairs.

## Gate 4: Sol judge and deterministic score

The judge preflight authenticates the answer replay before opening locked gold.
Each Sol message contains only question, reference answer, and sealed
prediction:

```powershell
& $python $judgeTool preflight `
  --judge-output-root $judgeRoot `
  --answer-root $answerRoot `
  --dataset $dataset `
  --split $split `
  --expected-answer-preflight-sha256 0c4464cf288b93f814991fd7abc2d74d76c5ce7396ae8829eeac43d1ec38f289 `
  --expected-answer-run-sha256 f1d774e98f48758b8ced70be05064e0af0aa538f9673f7744f0df8607ba54946 `
  --expected-answer-replay-sha256 2cbb053f31c2ba713a9fa16819b4bd8d007ff5b758da99e919b4c6d8795f1d41 `
  --postseal-audit $audit `
  --expected-postseal-audit-sha256 65285b9db760cb649e621465492ff0c323c9449c0b2735a34dd8bd70f23cf369

& $python $judgeTool approve-release `
  --judge-output-root $judgeRoot `
  --answer-root $answerRoot `
  --expected-answer-preflight-sha256 0c4464cf288b93f814991fd7abc2d74d76c5ce7396ae8829eeac43d1ec38f289 `
  --expected-answer-run-sha256 f1d774e98f48758b8ced70be05064e0af0aa538f9673f7744f0df8607ba54946 `
  --expected-answer-replay-sha256 2cbb053f31c2ba713a9fa16819b4bd8d007ff5b758da99e919b4c6d8795f1d41 `
  --postseal-audit $audit `
  --expected-postseal-audit-sha256 65285b9db760cb649e621465492ff0c323c9449c0b2735a34dd8bd70f23cf369 `
  --expected-judge-preflight-sha256 34099ddb56fa2c2e2ba5d42c50cbe4cb142c16e94a95ed4183d91b507f80c8b9 `
  --approve-provider-release

& $python $judgeTool provider-run `
  --judge-output-root $judgeRoot `
  --expected-judge-preflight-sha256 34099ddb56fa2c2e2ba5d42c50cbe4cb142c16e94a95ed4183d91b507f80c8b9 `
  --expected-release-sha256 aea418d667daec86899636aaebfec09406f67123df6db994547769aca0e83573 `
  --enable-provider `
  --authorized-provider-calls 100
```

On resume, authorize exactly `100 - authenticated checkpoint hits`. Complete
the provider-free materialize, replay, score, and score-replay phases:

```powershell
& $python $judgeTool materialize --judge-output-root $judgeRoot `
  --expected-judge-preflight-sha256 34099ddb56fa2c2e2ba5d42c50cbe4cb142c16e94a95ed4183d91b507f80c8b9 `
  --expected-release-sha256 aea418d667daec86899636aaebfec09406f67123df6db994547769aca0e83573

& $python $judgeTool replay --judge-output-root $judgeRoot `
  --expected-judge-preflight-sha256 34099ddb56fa2c2e2ba5d42c50cbe4cb142c16e94a95ed4183d91b507f80c8b9 `
  --expected-release-sha256 aea418d667daec86899636aaebfec09406f67123df6db994547769aca0e83573 `
  --expected-judge-sha256 edccbd49a20bf92fcb52306fe28557eeccb8ebba69e9e12a26d5d6cc5d530239

& $python $judgeTool score --judge-output-root $judgeRoot `
  --expected-judge-preflight-sha256 34099ddb56fa2c2e2ba5d42c50cbe4cb142c16e94a95ed4183d91b507f80c8b9 `
  --expected-release-sha256 aea418d667daec86899636aaebfec09406f67123df6db994547769aca0e83573 `
  --expected-judge-sha256 edccbd49a20bf92fcb52306fe28557eeccb8ebba69e9e12a26d5d6cc5d530239 `
  --expected-judge-replay-sha256 edccbd49a20bf92fcb52306fe28557eeccb8ebba69e9e12a26d5d6cc5d530239

& $python $judgeTool score-replay --judge-output-root $judgeRoot `
  --expected-judge-preflight-sha256 34099ddb56fa2c2e2ba5d42c50cbe4cb142c16e94a95ed4183d91b507f80c8b9 `
  --expected-release-sha256 aea418d667daec86899636aaebfec09406f67123df6db994547769aca0e83573 `
  --expected-judge-sha256 edccbd49a20bf92fcb52306fe28557eeccb8ebba69e9e12a26d5d6cc5d530239 `
  --expected-judge-replay-sha256 edccbd49a20bf92fcb52306fe28557eeccb8ebba69e9e12a26d5d6cc5d530239 `
  --expected-score-sha256 91ae36ebb7ef48fb914f7236ca03998adb0b22f58d98c29bfa8ecccd3739dce1
```

The score and score replay must be byte-identical. Report accepted count over
all 100 questions, changed predictions, parent-to-v5 rescues and regressions,
fallback/invalid Terra counts, call/checkpoint counts, and artifact hashes.

## Gate 5: policy-v5-r2 proof overlay and differential judge

This follow-up is the provider-safe route from the frozen 88/100 answer run to
the sealed 92/100 result. Keep every root below distinct from the base answer
and judge roots.

1. Run `tools/run_locked_full100_numeric_frontier.py materialize`, then
   `replay`, against the authenticated full100 construction, Terra answer
   lineage, post-seal audit, and resident stores. This step is gold-blind and
   provider-free. It scans each required namespace once and may mark a numeric
   frontier closed only when every policy-relevant store atom is represented
   and its component, identity, and state boundary is certified. Require the
   v2 materialization and replay to be byte-identical.
2. Run `tools/revalidate_locked_semantic_global_terminal_full100_policy_v5.py`
   with `materialize`, then `replay`, with both numeric-frontier hashes bound. The
   asymmetric rule gives priority to a supported deterministic numeric proof;
   permits an exact-day direct fact to fill a genuine parent abstention; and
   permits any other semantic rewrite only with a parent-defect certificate,
   support for every material claim, and complete touched conflict
   neighborhoods. Otherwise it emits the exact protected-parent prediction.
   This is structural non-regression, not a gold-based promise of benchmark
   correctness: the policy sees no reference answer and exposes no provider or
   caller-ordinal execution path.
3. Run `tools/plan_provider_free_differential_judge.py plan` against the sealed
   policy run/replay and one or more authenticated prior Sol judge triplets.
   Reuse a judgment only when question, reference, and prediction hashes plus
   the judge contract and model identity match exactly. Conflicting eligible
   prior judgments fail closed. Planning makes zero provider calls and emits no
   score; it seals only the unmatched prompts and cannot score until their
   judgments are supplied. The frozen r2 plan reused 98 rows and selected only
   Q28 and Q97.
4. Execute those two rows through
   `tools/run_locked_differential_novel_sol_judge.py` in this order:
   `preflight`, `approve-release`, `provider-run`, `materialize`, `replay`.
   Preflight and release must bind the plan SHA, exact Sol model, canonical
   output/checkpoint root, retries of zero, journal owner, and the ordered
   two-row population. Release is explicit; a fresh run must use
   `--enable-provider --authorized-provider-calls 2`, and a resume must
   authorize exactly `2 - authenticated journal pairs`. There is no ordinal
   CLI. Materialization and replay are checkpoint-only and must be
   byte-identical.
5. Run `tools/plan_provider_free_differential_judge.py merge` with the sealed
   plan and authenticated novel preflight/run/replay triplet. Merge is
   provider-free and must refuse an incomplete, duplicated, foreign, or
   differently hashed row population. Only a complete 100-row merge may emit
   the score.

The archived r2 lineage is:

| Stage | Root | SHA-256 |
|---|---|---|
| numeric frontier v2 materialization and replay | `eval_results\matched_eval_100\locked-full100-numeric-frontier-v2` | `15a7d9bbd90666f441ed93089ef331d86497e569b59200eb52248a82bc231566` |
| asymmetric policy run | `eval_results\matched_eval_100\locked-semantic-global-terminal-full100-terra-answer-v5-r1\policy-v5-r2` | `cb19ee0649ab50f55ca6db42d9333bf881f3434cfa754449a9ed4da3fd1b9e84` |
| policy replay receipt | same policy root | `ec63ff495f86c48548e1490fb24dd87b8136a22990263a584ae8896fdd4186bb` |
| differential plan, 98 reused plus 2 novel | `...\policy-v5-r2\differential-sol-judge-v1` | `025a14c8e3191019c5fd66399f847f8b4e901c88ac9722abda72dd50bcad51b4` |
| novel preflight | `...\differential-sol-judge-v1\novel-sol-execution-v1` | `83267b1e4623a84ae946927929989c55d7186626ca0991f9e6082e54058e7358` |
| novel provider release | same novel root | `1c4675011a0e1c1e3b703110a933dfa65b5ea576a73be9ce4a192c25f8c710f3` |
| novel judge and byte-identical replay | same novel root | `75c687a20a4a9fca4ec7f33add823d1bd428daebe595bc4818c79f108179dd9c` |
| final merge, **92/100** | `...\differential-sol-judge-v1\merge-v1-r1` | `e20286f2b8d9e81e4b69dd947b59d7e111c2b47842f3a54b15e95c668e001f3c` |

The v2 proof search correctly left Q14, Q53, and Q69 open. Q14 still needs a
certified cuisine/component identity boundary, Q53 needs certified plant
identity and state reduction, and Q69 needs a certified current-versus-unknown
obligation-state boundary. Treating those frontiers as closed would remove the
premise of the proof; the next repair must add and test those semantics rather
than force an answer through the numeric operator.

## Gate 6: operator-material-v3 frontier successor

The provider-free `operator-material-v3` successor retains v2 full-row,
bidirectional material closure and changes only reducer-observable state
comparison. Raw state still controls compiler admission and exclusion. Only
after a candidate is admitted is its status normalized to `operator_eligible`
for material-fact equality. Cancelled, excluded, or unrequested proposed rows
therefore cannot become operands through this normalization.

The profile also extends proof applicability to jewelry and museum/gallery
counts. Applicability is not closure: the sealed assay found seven applicable
rows and closed four. Q28 remained closed; Q53, Q67, and Q69 newly closed;
Q14, Q40, and Q77 remained open and retained their protected parents. In
particular, Q40's expanded domain did not override its census/provider
mismatch.

The resident loader now authenticates the common sealed retrieval/query
population once per process instead of reconstructing it for every namespace.
Each selected namespace database, HNSW index, store receipt, partition cache,
and full-store window index remains independently checked. The verified loader
also binds a unique and complete lifecycle partition, exact per-namespace row
receipts, matching window-index receipts, and v3's normalized census status.
This is a dataflow optimization and verifier hardening; neither widens the
frontier predicate.

The sealed r3 lineage is:

| Stage | Root | SHA-256 |
|---|---|---|
| numeric frontier v3 materialization and byte-identical replay | `eval_results\matched_eval_100\locked-full100-numeric-frontier-v3-r1` | `94092dcd879a3869f63177a08bd9366f7221bbed3d2fa33da7b268bb16ca6f59` |
| policy-v5-r3 run | `eval_results\matched_eval_100\locked-semantic-global-terminal-full100-terra-answer-v5-r1\policy-v5-r3` | `a145c8d6d5587293347621c5ca32d367e9aefe050c706e7232691a6c49aa34a9` |
| policy-v5-r3 replay | same policy root | `ec0672539d5a4d8df33673896a7c07bb8b0052a871cae7df7c66851e35f55052` |
| differential plan, 97 reused plus 3 novel | `...\policy-v5-r3\differential-sol-judge-v1` | `6df257b380cd6f4d19dac785cb85017766b1f8fdfe5561abd10b445b4a45f39d` |
| novel preflight | `...\differential-sol-judge-v1\novel-sol-execution-v1` | `640d2b324e425ac3d679aff5400162207c9b51adb213276548d8b9555f20f053` |
| novel provider release | same novel root | `9eed49a96a6167f180224adb6abe5bb41457c475e2b3a58dd19a7a9dc9aae264` |
| novel judge and byte-identical replay | same novel root | `dc5d145cb422203b08ba4ee14b2ee9dad54c6f3d71bde6dcedc5a9608a9355ef` |
| final merge, **95/100** | `...\differential-sol-judge-v1\merge-v1-r1` | `aa210a8bba87897d7fc8e3f4e2a7e71cbcc929fa4eeac6ce5cbf6ef56567c952` |

The differential planner reused 97 authenticated judgments and exposed only
Q53, Q67, and Q69. Exactly three Sol calls were authorized and made at zero
retries; all three were accepted. Planning, frontier construction, policy
overlay, materialization, replay, and merge made no provider calls. The five
remaining misses are Q14, Q40, Q49, Q82, and Q94.

## Stop and failure conditions

Stop before provider release if any of the following occurs:

- construction or replay differs;
- the promotion audit is below 26/26 usable atoms;
- the exact-11 plan projection is not byte-identical inside full100;
- an output root contains foreign, partial, or historical state;
- the required authorization differs from the authenticated missing count;
- a request/response journal pair is incomplete;
- provider retries are not zero;
- prompt, source, policy, dataset, model, root, or SHA binding changes.

After release, any incomplete authenticated journal/checkpoint set or
non-byte-identical replay fails the campaign. A final validation score below
95/100 is a completed negative result, not a pre-release gate failure; record
it and analyze misses without relabeling it as a pass.

If validation is at least 95/100, freeze the complete v5 policy and artifact
lineage before constructing confirmation200. Do not tune on confirmation rows.
