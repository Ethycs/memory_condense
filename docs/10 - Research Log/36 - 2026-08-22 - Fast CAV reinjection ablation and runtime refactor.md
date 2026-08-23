# Fast CAV reinjection ablation and runtime refactor

**Status:** a complete, replayable ten-question development diagnostic now
runs the fourth-layer CAV ablation downstream of the sealed original 1M
retrieval artifact. On the selected S1 packet, the CAV-steered text-order arm
scored 6/10 normalized exact match and 0.843171 mean token F1, versus 5/10 and
0.811906 for the matched original-order arm. This is a small positive
development result for **ordering the same evidence set**. It is not live
end-to-end retrieval, direct activation injection into the responder, an
independently judged result, or the locked minimum-100-question accuracy gate.

This entry supersedes the “CAV arm still open” execution status in
[Research Log 35](35%20-%202026-08-22%20-%20Fast%201M%20retrieval%20and%20synthesis%20path.md).
It does not supersede that entry's retrieval-stage definitions or its
separation of S0--S3, CAV routing, and later synthesis.

## What actually ran

The experiment is a downstream replay over the already sealed retrieval
result:

| Property | Exact value |
| --- | --- |
| Retrieval artifact | `eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821/retrieval.json` |
| Retrieval SHA-256 | `aa22f7c18470d9a7c931fd16f8f58bf67d8566e2298a45371ee2815c11a9bd97` |
| Source transcript represented by that artifact | 1,039,203 token proxies; 5,400 turns |
| Questions | 10 original development questions |
| Retrieval work in this experiment | none; no corpus build, store open, or S0--S3 rerun |
| Answered stage | S1, `direct_episode_additions` |
| Arms per question | `original`, `base`, `treatment` |
| Gold availability | unavailable to feature and answer phases; loaded only by the post-hoc score phase |

“1M” therefore describes the source transcript whose retrieval result was
sealed earlier. The new Qwen pass did **not** ingest one million tokens. It
encoded the globally deduplicated question and retrieved-evidence strings from
the read-only artifact. This distinction is the source of most of the runtime
improvement and must remain explicit in any comparison with live retrieval.

The experiment also did not use the historical provider messages as the
`original` arm. All three arms were freshly rendered with one canonical
evidence-catalog prompt contract so that their only intended difference was
catalog row order. The `original` arm preserves the artifact's S1 evidence
order; it is not byte-identical to the older fixed-S1 prompt in Research Log
35.

## Refactored execution seam

The heavy apparatus was split into five narrow phases:

```text
sealed retrieval.json
└── read-only immutable adapter
    └── one Qwen feature API call
        ├── 536 globally unique plain-text rows
        └── 67 internal forward batches at batch_size=8
            └── fixed three-CAV routing over 22 unique packets
                └── tensor-free original/base/treatment order receipts
                    └── 30 matched S1 text prompts
                        ├── provider answer
                        ├── zero-call journal replay
                        └── post-hoc gold scoring
```

The implementation boundaries are:

- `recall_guarded_cumulative_fast_artifact.py`: verifies the raw retrieval
  SHA-256 and sidecar, reconstructs exact questions from the final user
  messages, proves ordered-prefix S0--S3 nesting, and exposes immutable
  evidence provenance without opening a corpus or store;
- `fast_cav_feature_session.py`: globally deduplicates plain text, calls
  `Qwen3PrefixEncoder.encode_layers(...)` exactly once, reuses identical
  question-plus-evidence packets, and releases every feature/router tensor
  before returning frozen receipts;
- `fixed_cav_router.py` and `steered_readout.py`: apply the bounded two-pass
  CAV update and compare unsteered $X$ with steered $X_1$ against the exact
  same question vector;
- `fast_cav_prompts.py`: converts only the resulting evidence orders into
  matched, tensor-free prompts while proving identical evidence membership;
  and
- `fast_completion_runtime.py` and `run_fast_1m_cav.py`: preflight, immutable
  request/response journals, concurrent provider execution, replay, and
  post-hoc scoring.

No token IDs, model hidden states, or request-derived transformer state cross
the feature-manifest boundary. The feature and answer artifacts both record
zero retained transformer-token-state bytes. The external provider's own
persistence behavior is not certified.

## Feature and router execution

The adapter projected all four cumulative stages before the answer phase chose
S1:

| Feature-session quantity | Value |
| --- | ---: |
| S0--S3 logical evidence placements | 1,939 |
| Per-question unique evidence feature rows | 530 |
| Globally unique evidence texts | 526 |
| Globally unique raw questions | 10 |
| Total globally unique encoder inputs | 536 |
| Frozen stage receipts | 40 |
| Unique question/evidence packets routed | 22 |
| Encoder API calls | 1 |
| Actual internal Qwen forward batches | 67 (`ceil(536 / 8)`) |
| Retained token IDs / tensor bytes / persisted token-state bytes | 0 / 0 / 0 |

The “one encoder call” count is an orchestration/API count, not a claim that
Qwen performed one model forward. `encode_layers` received all 536 rows once
and executed 67 batches internally. This distinction is important when
interpreting both speed and GPU work.

The measured Qwen path loaded a one-layer Qwen3-8B prefix, read layer 0 in BF16
on CUDA, mean-pooled the selected residual, and returned a compact CPU FP32
`[536,4096]` feature matrix. Its identities were:

| Identity | SHA-256 |
| --- | --- |
| Prefix checkpoint | `76273516aa6924b12344d5e83daa485b66459b663c745cb3b9ef51cc17c7440d` |
| Feature backend/runtime | `13d3010d9b1e8c2908c067a6ef3283ef48f37c93520db71ab7a8244d6d2dc1b9` |
| Feature-session receipt | `b20d3c9eff9e5b7cbc665c6dc9897ec3730fbd5960c0d7e42f3bce94d5a59053` |

The fixed router ran on CPU FP32 with three layer-0 directions:
`autobiographical_completed_event`, `context_dependency`, and
`binding_constraint`. Both attention temperatures were 0.05 and residual
`alpha` was 1.0. The CAV-bank identity was
`3bdd657f8e8a41ec353308152e85c7d2a74f84ae59739200de15749c2e9766e3`;
the complete runtime identity was
`9c5b93a3b90910c1e70cfccefeb733c61e6cc4cabe7a609dd71cbccbdf7c639d`.

### Measured feature timings

| Phase | Seconds |
| --- | ---: |
| Read and validate sealed retrieval artifact | 0.101809 |
| Load fixed CAV router | 2.185720 |
| Load one-layer Qwen prefix | 216.841399 |
| One feature-session API call, including 67 forwards and routing | 7.309795 |
| Total before feature-manifest publication | 226.842578 |

Startup dominates this fresh process. Once `features.json` exists, answer,
replay, and scoring phases do not load Qwen and do not open a retrieval store.
The useful optimization is therefore phase separation and feature reuse, not
the fiction that all transformer work took one forward.

## What “CAV treatment” means here

For every question/stage packet, the readout used the same question vector
$q$ and exact evidence matrix $X$:

$$
s_i^{base}=\cos(X_i,q),\qquad
s_i^{treatment}=\cos((X_1)_i,q).
$$

The fixed CAV router produced $X_1$ by extracting three query-conditioned
concept rows from $X$ and reinjecting those concepts into the evidence-node
features. The treatment then sorted the exact evidence IDs by
$s^{treatment}$. The base arm sorted the same IDs by $s^{base}$, and the
original arm retained retrieval order.

All 40 S0--S3 rows preserved the exact evidence set, and all 40 had distinct
original, base, and treatment orderings. For the answered S1 slice, all 10
questions likewise had three different orders over the same 525 total
evidence rows. No arm added, removed, summarized, or rewrote evidence.

The S1 treatment changed the top-ranked row relative to base on 5/10
questions. Its mean within-question absolute rank displacement was 5.501128
positions. Across the 525 aligned evidence rows, the mean absolute cosine-score
change was 0.010761 and the maximum was 0.041582. These values establish that
the treatment was non-null while also showing that its numeric steering effect
was fairly small.

This is **not direct responder activation injection**. $X_1$ never entered
Terra's hidden state, KV cache, or residual stream. It affected only the
ordering of exact text rows in a normal provider prompt and was then released.
A direct responder-injection experiment would require an open-weight responder
and a separately specified intervention/measurement protocol. The current
result supports only the narrower claim that CAV-induced evidence ordering was
executable and modestly favorable on this development slice.

## Provider execution: failed root versus successful run2

Provider-free preflight certified 30 logical and 30 unique S1 prompts, with a
maximum local prompt-token proxy of 6,084 under the hard 8,000-token cap. There
was no prompt deduplication because each of the three orders differed for every
question.

The first provider attempt in
`eval_results/longmemeval-1m-fast-cav-development-20260822/` is **not a result**.
The controlled gateway returned completion text but reported prompt-token
usage as `0`, its sentinel for unavailable usage. The then-current runtime
incorrectly rejected that value before it could publish a response journal.
Because work had been submitted eagerly, the root contains 30 immutable
request journals, zero response journals, and no `answers.json`. This was a
harness-validation failure, not a transport failure or model-completion
failure. It remains preserved rather than being relabeled or retried in place.

The runtime was repaired to treat provider-reported zero usage as unavailable,
while continuing to enforce the independent local prompt proxy. The complete
answer run then used a fresh root:
`eval_results/longmemeval-1m-fast-cav-development-20260822-run2/`.

| Successful run2 quantity | Value |
| --- | ---: |
| Model route | `codex_sdk/gpt-5.6-terra` through the controlled gateway |
| Logical / unique / physical completions | 30 / 30 / 30 |
| Request / response journals | 30 / 30 |
| SDK retries | 0 |
| Maximum concurrency | 4 |
| Local prompt-token proxy | 158,598 total; 5,286.6 mean; 6,084 maximum |
| Local completion-token proxy | 253 total; 8.433 mean |
| Recorded provider elapsed time | 274.777732 s summed over concurrent calls |
| Provider elapsed per call | 9.159258 s mean; 9.091847 s median; 4.487734--12.700237 s range |
| Reported provider token usage | unavailable; zero sentinels were not counted as real zero-token requests |

The 274.778-second number is the sum of per-call provider durations, not wall
time; calls ran with concurrency four. Run2's counts and timing do not erase
the first root. Across both roots, the earlier completion responses were not
sealed, so the combined external lineage is terminally uncertain and no exact
all-attempt physical-call total is claimed. This is another reason to treat
run2 as a development diagnostic rather than certification evidence.

## Answer result

Gold was loaded only after the answer artifact and all 30 response journals
were revalidated. The exact S1 aggregates were:

| Arm | Evidence operation | Normalized EM | Mean token F1 | F1 change vs original |
| --- | --- | ---: | ---: | ---: |
| `original` | canonical catalog in retrieval order | 5/10 | 0.811906 | baseline |
| `base` | same rows, unsteered Qwen cosine order | 6/10 | 0.792629 | -0.019277 |
| `treatment` | same rows, CAV-steered cosine order | 6/10 | 0.843171 | +0.031264 |

Treatment exceeded base by 0.050542 mean F1 while preserving its 6/10 exact
match. Relative to original, treatment improved F1 on two questions, reduced
it on one, and tied on seven; the one additional exact match was `bbf86515`.
Relative to base, treatment improved two questions and tied eight. This is a
small, concentrated result rather than evidence of a broad or statistically
stable gain.

No independent semantic judge was run. The three-arm provider run is also
stochastic and was not replicated. The prior fixed-S1 numbers in Research Log
35 used a different prompt contract, so the appropriate control is the
`original` arm inside this manifest, not a cross-run score comparison.

## Sealed artifacts and replay

| Artifact | SHA-256 |
| --- | --- |
| `features.json` | `f57aee2ddb654989d4c35117e1fb7cf8d0e7e13ab89259f199817d198a603f89` |
| run2 `answers.json` | `3eb7ef688a4283958272f9ebf0d14f31baa3d4fa986aaa520c653771c1c37c8a` |
| run2 `scores.json` | `d6e786f0f89828b4bcee4de50659d419880a1598c4a128ff5d9527b066105618` |
| run2 `replay.json` | `261c5d51268c1414e143a1ad3ce56e71aaa1b222afd00c094f227d2ed72ee458` |

The replay reopened the 30 immutable response journals, made zero physical
provider calls, recorded 30 checkpoint hits, and reproduced all 30 prediction
and response-journal identities. `replay.json` is not byte-identical to
`answers.json` because its declared mode and usage counters correctly describe
replay rather than live execution.

The final provider-free command timings were:

| Verification operation | Result | Elapsed |
| --- | --- | ---: |
| Actual-feature S1 preflight | 30 logical / 30 unique prompts; zero calls | 1.76 s |
| All-stage identity-order lower-bound preflight | 120 logical / 22 unique prompts; zero calls | 1.98 s |
| Actual-feature all-stage preflight | 120 logical / 66 unique prompts; zero calls | 2.12 s |
| Completed-journal replay | 30 checkpoint hits / 0 physical calls | 2.01 s |
| Score plus complete journal revalidation | reproduced `d6e786f0...` | 19.47 s |

The 22 count is both the unique question/evidence routing-packet count and the
provider-free control obtained before feature-derived orders exist. It is not
the cost of answering all three actual CAV arms at all four stages. Once the
three distinct orders are rendered, the actual all-stage population contains
66 unique prompts. The measured answer run deliberately selected S1 and its
30 unique prompts.

## Reproduction commands

Use fresh feature and answer roots for a new provider experiment. The measured
roots are immutable evidence and should not be overwritten.

```powershell
$retrieval = "eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821/retrieval.json"
$featureRoot = "eval_results/longmemeval-1m-fast-cav-reproduction"
$answerRoot = "eval_results/longmemeval-1m-fast-cav-reproduction-run2"
$features = "$featureRoot/features.json"
$answers = "$answerRoot/answers.json"
$dataset = "C:\path\to\memory-condense-rig\datasets\longmemeval_s_cleaned.json"
$split = "docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"

# Provider-free all-stage identity-order lower bound; 120 logical / 22 unique.
pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_cav `
  --phase preflight --retrieval $retrieval --output-root $featureRoot --stages all

# One Qwen load and one encode_layers API call (67 internal batches).
pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_cav `
  --phase features --retrieval $retrieval --output-root $featureRoot --stages S1 `
  --model-dir .cache/models/Qwen3-8B --device cuda --dtype bfloat16 `
  --batch-size 8 `
  --event-cav eval_results/qwen3_event_membership_cav_probe.safetensors `
  --prefix-cav eval_results/qwen3_prefix_cav_probe.safetensors `
  --extraction-temperature 0.05 --reinjection-temperature 0.05 --alpha 1.0

# Recheck all actual feature-derived orders: 120 logical / 66 unique.
pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_cav `
  --phase preflight --retrieval $retrieval --output-root $featureRoot `
  --features $features --stages all

# Select the measured S1 population: 30 logical / 30 unique.
pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_cav `
  --phase preflight --retrieval $retrieval --output-root $featureRoot `
  --features $features --stages S1

# Explicitly authorized live provider phase: exactly 30 unique prompts.
pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_cav `
  --phase answer --retrieval $retrieval --output-root $answerRoot `
  --features $features --stages S1 --enable-provider `
  --authorized-provider-calls 30 --max-concurrency 4 --max-new-tokens 256 `
  --gateway-url https://central-dev.zt:4000/v1 `
  --gateway-model codex_sdk/gpt-5.6-terra `
  --caller-model openai/codex_sdk/gpt-5.6-terra `
  --api-key-env LITELLM_KEY

# Provider-free replay of the completed response journals.
pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_cav `
  --phase replay --retrieval $retrieval --output-root $answerRoot `
  --features $features --stages S1

# Gold becomes reachable only in this final local scoring phase.
pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_cav `
  --phase score --retrieval $retrieval --output-root $answerRoot `
  --features $features --answers $answers --dataset $dataset --split $split
```

Final focused verification was green after the runtime fixes: the complete fast
stack passed 85 tests in 10.12 seconds. The related existing Qwen/cumulative
suites passed another 35 tests in 16.86 seconds. These are focused integration
counts, not a claim that the repository-wide suite was run in this measurement.

```powershell
pixi run --frozen -e dev pytest -q `
  tests/test_recall_guarded_cumulative_fast_artifact.py `
  tests/test_fixed_cav_router.py `
  tests/test_steered_readout.py `
  tests/test_fast_cav_feature_session.py `
  tests/test_fast_cav_prompts.py `
  tests/test_fast_completion_runtime.py `
  tests/test_run_fast_1m_cav.py

pixi run --frozen -e dev pytest -q `
  tests/test_qwen_prefix.py `
  tests/test_recall_guarded_cumulative_1m.py `
  tests/test_recall_guarded_cumulative_population.py `
  tests/test_recall_guarded_cumulative.py
```

## Claim boundary and next gate

The measured result is useful but deliberately narrow:

- it replays sealed development retrieval rather than measuring retrieval
  latency, source recall, or answer reachability from a live 1M-token corpus;
- it changes text order over an identical S1 evidence set rather than adding
  evidence or injecting activations into the answer model;
- it uses ten repeatedly analyzed development questions, one fixed CAV bank,
  one layer-0 prefix configuration, and one provider sample per arm;
- it has normalized EM/F1 but no independent semantic judge, confidence
  interval, fresh held-out replication, or Mem0 comparison; and
- the first request-only provider root remains separate terminally uncertain
  lineage rather than being silently merged into run2.

The locked target remains `longmemeval-s-1m-100q-95-v1`: at least 100 fixed-
stage questions, a 256-token responder, an independent judge, and at least 95%
accuracy under the preregistered protocol. This 10Q three-arm development
diagnostic neither attempts nor passes that gate. Its proper conclusion is:
the streamlined CAV ordering ablation now works end to end and produced a
small favorable within-run signal worth testing on a fresh, sufficiently large
population.
