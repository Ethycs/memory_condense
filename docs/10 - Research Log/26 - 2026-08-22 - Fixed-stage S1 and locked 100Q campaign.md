# Fixed-stage S1 and locked 100Q campaign

**Status:** the fixed-stage Terra responder, independent Sol judge, exact
ten-shard validation retrieval runner, and schema-v3 Mem0 comparison boundary
are built and provider-free preflighted. No validation shard has yet entered
the GPU retrieval phases, and no Terra or Sol call has been made for this new
campaign. There is therefore no live 100-question score, the >=95% target has
not been achieved, and the Mem0 comparison is not certified.

This closes the implementation gap identified at the end of
[Research Log 25](25%20-%202026-08-22%20-%20Independent%20Sol%20judge%20and%20v3%20synthesis%20repair.md).
It does not promote the ten-question diagnostic in that log into a formal
result.

## Outcome

The evaluation now has one linear retrieval experiment and one fixed answer
gate:

```text
ten independent ~1M validation shards
  -> S0 causal/coverage predecessor
  -> S1 direct episode additions       <- preregistered answer stage
  -> S2 representative additions       <- retained retrieval ablation
  -> S3 artifact-global closure         <- retained retrieval ablation
  -> exact ten-shard / 100Q merge
  -> one <=8,000-token S1 prompt per question
  -> Terra answer, <=256 output tokens
  -> one independent Sol verdict per question
  -> >=95/100 fixed-stage gate
```

The full S0--S3 ladder from
[Research Log 22](22%20-%202026-08-21%20-%20Recall-guarded%20cumulative%20retrieval.md)
has not been removed or replaced. Each validation question still produces all
four nested retrieval receipts, and every child must preserve its parent as an
exact ordered prefix. Only the answer-scoring branch is fixed at S1
`direct_episode_additions`.

S1 was selected before validation because it was the sole measured positive
retrieval increment on development: it increased mean best-evidence F1 while
S2 changed no reported quality metric and S3 admitted no evidence under the
frozen budget. This makes the target one test of one preregistered method,
rather than choosing the best of S1--S3 after reading validation answers.
[Research Log 23](23%20-%202026-08-21%20-%20Episodic%20evidence%20scoring%20and%20synthesis.md)
and
[Research Log 24](24%20-%202026-08-21%20-%20LiteLLM%20Terra%20episodic%20synthesis%20and%20rescoring.md)
remain the development evidence for the stage decision.

## Locked validation population and real preflight

The new population module reconstructs the ten validation shards directly
from the exact dataset and split manifest. It excludes answers, categories,
and labeled evidence sources from the retrieval identity. The resulting
population consists of offsets `0, 10, ..., 90`, ten questions per shard, 100
unique questions, 10,441,617 transcript-token proxies, and 54,246 turns.

The real offset-0 preflight completed on 2026-08-22 against the paths and
model directories shown in the command below. It reconstructed and validated
the input and policy identities but did not initialize either Qwen model,
touch the output root, use the GPU, or contact a provider.

| Identity | SHA-256 |
| --- | --- |
| LongMemEval dataset | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Locked split manifest | `8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4` |
| Validation policy manifest | `5263d5afd15298ec4088db9d6381ae243ddb685e9a3cf4d9892fc84e14fb9883` |
| Validation policy attestation | `3c054b584fa4ca7dff7e1d97d4bc532bb3e98b140b15b3fa6c35cd67be971558` |
| Validation execution policy | `c6b3d6f3f511d0a52b271e0880cb93c3866e649b614cfa4b5ac0604b704fdc8d` |
| Effective retrieval policy | `0abb42db8fd35b566029be135630af34fa42ee9a464ef8c4a889948729cbab13` |
| Environment lock | `c99f82aa37a1df009c6100110d24aeaee0bde3376ff09537700ad9848b371a95` |
| Frozen retrieval implementation | `020e5ba816c2246ba021944d1e847aa9a96ce2f7d0caa2e808d66c11ba0c5c92` |
| Gold-blind 100Q population | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| Offset-0 shard | `e852e258eee3d6699cedef5e9f6a9b68f356c065c60ddb695ca965cf400d792a` |

The ordered shard identities sealed into the population are:

```text
offset 00  e852e258eee3d6699cedef5e9f6a9b68f356c065c60ddb695ca965cf400d792a
offset 10  6cfcdb0d0c7bf24c56eeb72ed90417080f5345c5e8c6b377974476425fc93be7
offset 20  5241842b1c70c518050c12a3e00085b5ae23e5cf321c2801bc9c862885c43473
offset 30  4dbe74c5c48e13229103b10c257dc447149e0300f826a7e5bb1473a18339f2da
offset 40  926068ec45534aa64316080d68b53d7a6a65183514faa1cfc7cd8240beddc4e1
offset 50  ec4ca25ad97277c846b75e1628c5765ecb5b8dcaffa26bf5eeddd867e40a8f2e
offset 60  d0ec00671068de6d43b390ecbc2af2ff65407aaf40bce1f4077bc5b756883754
offset 70  bbb02cadb883ce8b6c2fea22f2031ce179cdb25e7e70d4f2a5484330c36e7c5b
offset 80  90f915453487a02a60b2336f66e9179648ebad69eede87d7a32827f2bab40aa1
offset 90  cccaaa47c93686eb751679f98779d3b4e5b82426cc5d41105922a6e1f0e265ec
```

These are input/preflight identities, not retrieval-result identities. Shard
retrieval SHA-256 values, the merged `retrieval.json` SHA-256, its external
reconstruction receipt, and the 100Q answer/judge campaign bindings do not
exist until those phases finish. They must be reported from the produced
artifacts rather than predicted here.

## Ten-shard retrieval runner

[The validation retrieval runner](../../tools/run_recall_guarded_cumulative_validation_retrieval.py)
adds the missing campaign surface without weakening the historical
development runner. For each offset it can:

1. reconstruct and preflight the exact gold-blind shard;
2. prepare the current exact-span source;
3. build a separate causal-plus-discourse store;
4. run and checkpoint the complete S0--S3 cumulative ladder; and
5. publish a canonical shard `retrieval.json` only after strict validation.

The merge phase requires exactly the ten locked offsets and validates every
shard without entering model/GPU preflight. The merged artifact retains each
child retrieval SHA-256, original question-part SHA-256, shard identity,
source/store/compilation bindings, and per-question typed stage receipts. It
also publishes an external reconstruction receipt so a later answer preflight
can reopen all ten child artifacts and prove that the self-contained 100Q
merge was not assembled from substituted questions or stores.

There is deliberately no fictional global database receipt. A question
continues to name the physical combined store from its own shard.

## Fixed-stage Terra answer contract

[The fixed-stage answer runner](../../tools/run_recall_guarded_cumulative_final_answer.py)
consumes either the historical single-store retrieval or the new merged
validation format, but it selects exactly one stage: S1
`direct_episode_additions`.

Before the first possible call it validates the complete retrieval population,
all typed cumulative receipts, evidence coordinates, question-part hashes,
provider-message hashes, reconstructed token counts, the 8,000-token prompt
cap, and the 256-token output reserve. The operator-supplied authorization must
equal the exact number of unique preflighted prompts.

The runtime is locked to:

| Property | Contract |
| --- | --- |
| Caller route | `openai/codex_sdk/gpt-5.6-terra` |
| Gateway route | `codex_sdk/gpt-5.6-terra` |
| Prompt cap | 8,000 `cl100k_base` proxy tokens |
| Output allowance | 256 tokens |
| Sampling | temperature omitted / represented as `null` |
| Retries | 0 |
| Local persisted transformer request state | false; 0 bytes |
| External provider persistence | not certified |

Every request reservation is durably journaled before network I/O and every
completed response is separately journaled afterward. A request journal with
no response is terminal uncertainty and cannot be retried. Replay constructs
no provider client and fails on any missing journal. The final artifact binds
the immutable completion reports, request/response journal hashes, actual
prompt-policy identity, answer hashes, runtime identity, and campaign identity.

The responder prompt-policy identity is derived from the system-message bytes
actually present in the selected prompts plus the verified QA user framing,
tokenizer proxy, and output reserve. It is not a generic label that allows two
different prompt wordings to collide.

## Historical development preflight boundary

The existing ten-question retrieval remains the immutable historical artifact
from Research Logs 22--25:

| Property | Value |
| --- | --- |
| Retrieval SHA-256 | `aa22f7c18470d9a7c931fd16f8f58bf67d8566e2298a45371ee2815c11a9bd97` |
| Population identity | `fa9a06ebd103d87086943cfa94091bdf607fe07874bc871e465aad409b85ca18` |
| Selected stage | S1 `direct_episode_additions` |
| Questions / unique Terra prompts | 10 / 10 |
| Maximum selected-stage prompt | 7,283 / 8,000 tokens |
| Final-answer policy SHA-256 | `ea43d5b5a0ef5e95453d03917b37ea5385feea89b7546ea6841da1c9fe866418` |
| Actual responder prompt-policy SHA-256 | `ebe671a232197818db00adb14a8f9ca3e649fe9f230abca60de09633d70c3f55` |
| Provider-free preflight binding | `19647c754981e4b08bcad1a1f08fce665494eb0e6238e4d003e8e687552284dc` |

That preflight made zero Terra calls and wrote no output root. It proves that
the historical retrieval can enter the new 256-token fixed-stage runtime; it
is not a new answer result.

The approximate-current-value correction derived from the development miss
in Research Log 25 is now part of the QA prompt constructor: a newest statement
such as "close to 1300 now" supports returning the stated approximate number
rather than abstaining solely because it is approximate. This correction was
frozen before validation construction and will apply to newly built validation
prompts. It does **not** retroactively change the system-message bytes inside
the historical ten-question retrieval above, so its historical prompt-policy
identity remains distinct.

## Independent fixed-stage Sol judge

[The fixed-stage judge runner](../../tools/run_recall_guarded_cumulative_final_answer_semantic_judge.py)
validates the entire answer artifact against its retrieval before loading
gold. It then constructs and authorizes the complete one-stage judge
population before a provider call is possible. Each row binds the question,
gold-answer, prediction, Terra response journal, Sol prompt, and Sol response
journal by SHA-256.

The judge route is `openai/codex_sdk/gpt-5.6-sol`, with a 1,024-token verdict
allowance, omitted temperature, and zero retries. Its score contains one
binary verdict per question and applies one exact gate:

```text
question_count >= 100 AND correct / question_count >= 0.95
```

The 10Q development population can therefore report diagnostic verdicts but
can never pass this gate. No fixed-stage Sol artifact has been produced yet.

## Mem0 schema-v3 boundary

[The Mem0 comparator](../../tools/mem0_eval/compare.py) now accepts the
fixed-stage semantic score directly. It strictly validates the treatment
score's internal structure and recomputes 100 paired binary outcomes from
hash-bound rows. It also independently constructs the Mem0 responder
prompt-policy object and requires it to equal the object carried by the
treatment score. Legacy schema-v2 comparison behavior remains separate.

Object equality is not treated as proof that the treatment policy was derived
from the actual Terra prompts. In hash-only mode the comparator reports
`responder_prompt_policy_identity_object_equal=true`, but leaves responder
derivation, responder identity, Sol prompt derivation, prompt accounting, the
broad treatment contract, and provenance verification false. It adds the
explicit blocker
`treatment_final_answer_artifact_derivation_unverified`.

The caller can additionally supply the canonical fixed-stage final-answer
artifact and its retrieval to `compare_campaign_reports`. That bound path
revalidates the answer artifact against the retrieval, checks their canonical
file identities, joins every score row to its exact answer row, reconstructs
every Sol message from the validated prediction plus the independently
validated Mem0 question and gold answer, and verifies both message SHA-256 and
prompt-token proxy. Only this path enables the treatment prompt-derivation,
prompt-accounting, contract, and provenance flags. The bound path still does
not certify the comparison as a whole.

A schema-v3 result is presently **metric-valid but noncertified**. The
supported comparison is binary judge accuracy plus paired wins, ties, and
losses. F1, exact match, context-token, and prompt-token comparisons are not
silently inferred. Certification remains false with these explicit blockers:

- `paired_source_population_identity_unverified`;
- `shared_sampling_policy_identity_unverified`;
- `shared_zero_retry_policy_identity_unverified`;
- `shared_model_deployment_identity_unverified`;
- `mem0_production_binding_certified_false`; and
- `mem0_locked_comparison_protocol_not_verified` when the supplied Mem0
  report does not certify that local protocol.

Certification requires a future Mem0 schema to bind its source corpus to the
same treatment `population_identity_sha256`, prove omitted responder/judge
sampling parameters and provider-level zero retries, and provide independent
deployment identities shared across both arms. The comparator does not infer
any of those claims from model names, dataset filenames, question IDs, or
treatment-specific runtime hashes.

This means a future schema-v3 comparison can be numerically useful while still
being labeled `metric_only_noncertified`. Neither a metric win nor a 95% arm
score alone closes the fairness boundary.

## Verification evidence

The implementation has dedicated tests for:

- exact gold-blind population reconstruction and shard merging;
- provider/model-free preflight and merge isolation;
- typed S0--S3 stage receipt validation and tamper rejection;
- whole-population answer preflight before output-root or client creation;
- exact unique-call authorization, request-first journaling, terminal
  uncertainty, and no-client replay;
- live-to-fresh-replay byte identity using deterministic fake clients;
- answer-policy, prompt-policy, runtime, campaign, and journal cross-binding;
- independent judge gold timing, 95/100 pass, 94/100 failure, and minimum-100
  enforcement; and
- strict schema-v3 Mem0 paired-metric recomputation and noncertification.

On the frozen bytes, the final focused integration selection spanning these
paths plus the affected legacy diffuse-receipt golden passed 349 tests in
100.57 seconds. A separate final audit selection spanning
the fixed-stage judge and schema-v3 comparator passed 82 tests in 23.54
seconds, and the independent adversarial review found no remaining concrete
fail-open within the stated artifact-integrity trust model. The broad
2,503-test run was interrupted after isolating one legitimate golden drift:
the new QA system prompt left the derived SQLite bytes unchanged but changed
the bounded packet and its sealed phase/finalization identities. All three
goldens were refreshed and the exact test passed both alone and in the final
349-test selection; a completed full-suite pass is not claimed here. No test
substitutes for the unrun GPU retrieval, Terra answer, or Sol judge campaigns.

## Exact campaign commands

The direct development-environment Python executable is used below because
`pixi run` can fail on this host while determining the home directory. The
commands intentionally keep preflight, expensive retrieval, answering, and
judging as separate authorization boundaries.

Provider- and model-free shard preflight:

```powershell
$python = ".pixi/envs/dev/python.exe"
$dataset = "C:\Users\Keytone\Downloads\memory-condense-rig\datasets\longmemeval_s_cleaned.json"
$split = "docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"
$policy = "docs/10 - Research Log/data/longmemeval-qwen-choice-coverage-operational-validation-v3.json"
$retrievalRoot = "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"

& $python -u tools/run_recall_guarded_cumulative_validation_retrieval.py `
  --phase preflight --dataset $dataset --split-manifest $split `
  --policy-manifest $policy --output-root $retrievalRoot `
  --sample-offset 0 --qwen-prefix-model-dir .cache/models/Qwen3-8B `
  --qwen-choice-model-dir .cache/models/Qwen3-0.6B --device cuda

if ($LASTEXITCODE -ne 0) {
  throw "Validation shard preflight failed for offset 0."
}
```

After separately checking all ten preflights, the future GPU retrieval and
provider-free merge are shown below. Run only one shard process at a time and
wait for any existing process for an offset to exit before invoking that
offset again. PowerShell does not reliably stop a `foreach` loop merely
because a native executable returns nonzero, so every invocation has an
explicit exit-code check and the merge cannot follow a failed shard:

```powershell
foreach ($offset in 0,10,20,30,40,50,60,70,80,90) {
  & $python -u tools/run_recall_guarded_cumulative_validation_retrieval.py `
    --phase all --dataset $dataset --split-manifest $split `
    --policy-manifest $policy --output-root $retrievalRoot `
    --sample-offset $offset --qwen-prefix-model-dir .cache/models/Qwen3-8B `
    --qwen-choice-model-dir .cache/models/Qwen3-0.6B --device cuda

  if ($LASTEXITCODE -ne 0) {
    throw "Validation retrieval failed for offset $offset."
  }
}

& $python -u tools/run_recall_guarded_cumulative_validation_retrieval.py `
  --phase merge --dataset $dataset --split-manifest $split `
  --policy-manifest $policy --output-root $retrievalRoot --device cuda

if ($LASTEXITCODE -ne 0) {
  throw "Locked ten-shard validation merge failed."
}
```

The fixed-stage provider-free preflight and future Terra run are:

```powershell
$answerRoot = "eval_results/longmemeval-1m-fixed-s1-validation-20260822"

& $python -u tools/run_recall_guarded_cumulative_final_answer.py `
  --mode preflight --retrieval "$retrievalRoot/retrieval.json" `
  --output-root $answerRoot --authorized-provider-calls 100

& $python -u tools/run_recall_guarded_cumulative_final_answer.py `
  --mode run --retrieval "$retrievalRoot/retrieval.json" `
  --output-root $answerRoot --authorized-provider-calls 100
```

The value `100` is not a loose budget: the run refuses it unless the complete
merged-population preflight finds exactly 100 unique Terra prompts. Replay uses
the same command with `--mode replay` and refuses a cache miss.

After a sealed answer artifact exists, the independent judge preflight and
future Sol run are:

```powershell
$answers = "$answerRoot/final-answers.json"
$judge = "$answerRoot/final-answer-semantic-judge-sol.json"
$judgeCalls = "$answerRoot/final-answer-semantic-judge-sol-calls"

& $python -u tools/run_recall_guarded_cumulative_final_answer_semantic_judge.py `
  --population validation-100q --mode preflight --answers $answers `
  --retrieval "$retrievalRoot/retrieval.json" --dataset $dataset `
  --split-manifest $split --authorized-unique-calls 100 `
  --output $judge --checkpoint-dir $judgeCalls

& $python -u tools/run_recall_guarded_cumulative_final_answer_semantic_judge.py `
  --population validation-100q --mode run --answers $answers `
  --retrieval "$retrievalRoot/retrieval.json" --dataset $dataset `
  --split-manifest $split --authorized-unique-calls 100 `
  --output $judge --checkpoint-dir $judgeCalls
```

Again, `100` must equal the provider-free unique judge-prompt population; it
does not authorize retries or calls outside that sealed population. A replay
changes only `--mode` to `replay`.

## Current conclusion

The intended experiment is finally represented directly in code: retain the
linearly nested retrieval cases, choose S1 from development evidence, answer
one fixed stage under the original 8,000/256 budget, judge it independently,
and compare paired outcomes without overstating Mem0 fairness.

What exists today is a reproducible launch surface and a real provider-free
preflight. The next evidence-producing action is the ten-shard GPU retrieval
campaign. Only after its merge, 100 Terra answers, and 100 Sol verdicts can the
>=95% claim be evaluated. A separately bound Mem0 production arm is still
required for a fair certified comparison.
