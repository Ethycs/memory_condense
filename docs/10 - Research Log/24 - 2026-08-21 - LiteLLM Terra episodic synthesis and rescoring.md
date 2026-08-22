# LiteLLM Terra episodic synthesis and rescoring

**Status:** implemented, checkpointed, focused-test covered, and measured on
the exact 1,039,203-token development concatenation. A strict Terra synthesis
arm raised S1 answer quality from the earlier local-Qwen arm's 0/10 exact
match and 0.010227 mean F1 to 5/10 and 0.718433. S2 and S3 scored 4/10 and
0.706806. This is a ten-question development result, not a held-out or
independently judged accuracy claim.

## Decision

Research Log 23 established that the pinned Qwen3-0.6B runtime was useful for
generation-free evidence scoring but inadequate as the large-context answer
synthesizer. This experiment changes only that synthesis role:

```text
sealed recall-guarded retrieval.json
├── numeric evidence and claim scores: pinned local Qwen3-0.6B
├── cited answer, semantic role, categorical density: LiteLLM Terra
└── exact/F1/source/component scoring: separate post-hoc gold phase
```

The retrieval ladder, evidence order, stage budgets, and 176-item episodic
population are unchanged. This is a matched synthesis arm, not a new
retrieval stack and not a relabeling of the local-Qwen result.

## Frozen population and gold firewall

| Property | Frozen value |
| --- | --- |
| Parent retrieval | `eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821/retrieval.json` |
| Retrieval SHA-256 | `aa22f7c18470d9a7c931fd16f8f58bf67d8566e2298a45371ee2815c11a9bd97` |
| Population identity SHA-256 | `fa9a06ebd103d87086943cfa94091bdf607fe07874bc871e465aad409b85ca18` |
| Transcript-token proxy | 1,039,203 |
| Turns / questions | 5,400 / 10 |
| Episodic additions | S1: 171; S2-only: 5; S3-only: 0 |
| Cumulative answer rows | 30: ten each at S1, S2, and S3 |

Synthesis read only the sealed, explicitly gold-free retrieval artifact. The
benchmark population containing answers, answer components, and labeled
source IDs was loaded only after both raw and normalized synthesis artifacts
had been published. Post-hoc scoring recomputed and verified the population
identity before accepting the synthesis.

## Provider and local-scorer identity

The controlled environment's internal service catalog and LLM API runbook
describe the OpenAI-compatible LiteLLM gateway used by this run. Their
operator-only documentation addresses are intentionally not published here.
The API key was constructor-only input loaded from `LITELLM_KEY`; it is absent
from arguments, identities, journals, artifacts, and logs.

| Runtime property | Value |
| --- | --- |
| Caller model | `openai/codex_sdk/gpt-5.6-terra` |
| Gateway model / per-call response model | `codex_sdk/gpt-5.6-terra` |
| Gateway | `https://central-dev.zt:4000/v1` |
| Sampling field | omitted for the Codex SDK route |
| Maximum output | 4,096 tokens |
| Retries / fallback | 0 / disabled |
| Structured mode | required; invalid output fails closed |
| Local scoring model | `Qwen/Qwen3-0.6B`, FP16, CUDA |
| Local revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Local checkpoint manifest | `a940db06d5d9a3b298412376966b492f09ad7f088495fb75c05aa45db943d86e` |
| Runtime identity SHA-256 | `29853c5a1aabc3c3581ed94c513fa6c93127420b0155d8443065fdc541b4d999` |
| Campaign binding SHA-256 | `ba2d147053955635e6d36226442d1046de8d5398a777740265a995a48491050e` |
| Implementation SHA-256 | `9c2c453f7488f67e1ccf7eff5e2376b9626edd1735938ac73bf8b0f30cb2efc8` |
| Prompt-policy SHA-256 | `5a1f581d33df89dbc1520823327e34f3096b119fe0b4bd9bbd17bf73b4bdffe0` |
| Request-policy SHA-256 | `0fe7dc46e1549e467194f3ae035b5ef1bd9e8b2fd90385a4dfa8a1c438e970ab` |

Terra is bound to a gateway route, not a provider-side immutable checkpoint
or revision. Every call records the returned response ID and model, and all 12
responses named the expected route, but alias mutability remains a limitation.
The implementation digest above remains the exact measured identity. Later
pre-publication portability fixes made the dataset path explicit and correctly
propagated caller-selected model/device paths; they did not rewrite any
campaign artifact, so a current-source replay necessarily receives a new
implementation digest.

## Exact call budget and durable replay

The 30 stage rows collapse to exactly 12 structured prompts: one prompt for
each question plus a second prompt for the two questions whose S2 evidence
actually changed. Per-question unique-call counts were
`[1,1,1,2,1,1,1,2,1,1]`. S3 introduced no new prompt.

The launcher required an explicit authorization of 12 unique provider calls.
For each prompt it wrote an immutable, canonical `<key>.request.json`
reservation before the network request and an atomically published
`<key>.response.json` afterward. The call key binds the exact message hash,
runtime identity, output cap, and campaign binding. A verified response is
replayed without another request; a request-only crash state refuses an
unsafe retry. The completed inventory contains:

- 12 request journals and 12 response journals;
- 10 canonical question-part checkpoints;
- 12 physical, 12 unique, and 12 logical provider calls;
- zero checkpoint hits, retries, fallbacks, or temporary files; and
- 12 `stop` finish reasons with nonempty response IDs.

## Answer result

| Cumulative stage | Exact match | Mean F1 | Mean answer-component recall | All components hit | Mean claim-component recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 direct episodes | 5/10 | 0.718433 | 0.600000 over 2 applicable questions | 1/2 | 0.700000 |
| S2 representative episodes | 4/10 | 0.706806 | 0.600000 over 2 applicable questions | 1/2 | 0.200000 |
| S3 artifact-global closure | 4/10 | 0.706806 | 0.600000 over 2 applicable questions | 1/2 | 0.200000 |

The only S1-to-S2 exact-match loss was question `gpt4_7abb270c`. The answer
retained all six expected list components, but changed comma separators to
arrows; exact match fell while F1 remained 0.883721 and component recall
remained 1.0. That is a formatting sensitivity, not evidence of a lost answer
component. S3 is identical to S2 because it admitted no additional evidence.

The matched local-Qwen synthesis arm in Log 23 scored 0/10 exact match and
0.010227 F1 at every stage. Holding retrieval fixed while changing synthesis
therefore identifies the small local generator as the dominant answer-stage
bottleneck in that arm.

## Grounding and labeled-source diagnostics

| Stage | Claims | Exact citations | Quote grounding | Expected-source overlap | Micro precision | Micro recall | Macro recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 15 | 28 | 1.000000 | 23 / 25 expected source IDs | 1.000000 | 0.920000 | 0.900000 |
| S2 | 14 | 27 | 1.000000 | 23 / 25 expected source IDs | 1.000000 | 0.920000 | 0.900000 |
| S3 | 14 | 27 | 1.000000 | 23 / 25 expected source IDs | 1.000000 | 0.920000 | 0.900000 |

Nine questions cited every labeled expected source and no unexpected source.
The tenth abstained and cited none, accounting for the two missed expected
source IDs. Mechanical quote grounding proves that each quote is an exact
substring of its cited evidence; it is not an independent entailment verdict.

Terra assigned only three S1 additions a `critical` or `high` categorical
density. All three came from labeled expected sources: precision 3/3 and
recall 3/25. This is conservative high-density selection, not broad source
coverage. Coverage came from claims and citations that were also allowed to
use the protected S0 packet.

## Episodic role and density result

Terra supplied semantic role and categorical density labels. The local pinned
Qwen scorer independently supplied an uncalibrated A/B answerability signal
and a distinct answerability-per-100-token density transform. These measures
are intentionally not merged.

Terra categorical density over each addition exactly once:

| Delta | Count | Critical | High | Medium | Low | None |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 minus S0 | 171 | 2 | 1 | 1 | 28 | 139 |
| S2 minus S1 | 5 | 0 | 0 | 0 | 0 | 5 |
| S3 minus S2 | 0 | 0 | 0 | 0 | 0 | 0 |

Terra semantic roles:

| Delta | Decisive | Supporting | Temporal bridge | Qualifier/conflict | Context | Redundant | Irrelevant |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 minus S0 | 2 | 8 | 0 | 3 | 31 | 3 | 124 |
| S2 minus S1 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |

Local forced-choice answerability bands preserve the historical raw p(A)
view:

| Delta | Count | Critical | High | Medium | Low | None | Mean p(A) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 minus S0 | 171 | 3 | 14 | 13 | 28 | 113 | 0.288773 |
| S2 minus S1 | 5 | 0 | 0 | 0 | 0 | 5 | 0.103625 |

The separate per-token density transform produced:

| Delta | Critical | High | Medium | Low | None | Mean p(A) per 100 tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 minus S0 | 4 | 17 | 33 | 45 | 72 | 0.451601 |
| S2 minus S1 | 0 | 0 | 0 | 1 | 4 | 0.085692 |

Both models therefore agree on the actionable stage-level conclusion: the
five S2-only additions supplied no strong answer evidence. S2 did not improve
answer quality, and S3 added nothing under the cap. S1 remains the best
measured point on this retrieval ladder.

## Usage and latency

| Counter | Measured value |
| --- | ---: |
| Provider input-token proxy | 61,482 |
| Provider output-token proxy | 6,329 |
| Provider call time | 229.472 s total; 19.123 s mean |
| Provider call-time range | 13.934–23.819 s |
| Local scorer calls / forward passes | 39 / 56 |
| Local scorer elapsed time | 10.856 s |

The gateway returned zero-filled usage objects, so reported provider token
usage is explicitly marked unavailable. The deterministic legacy token proxy
is the only token accounting claimed here.

## Artifacts and replay

Artifact root:

```text
eval_results/longmemeval-1m-recall-guarded-cumulative-litellm-terra-synthesis-development-20260821/
```

This root is intentionally ignored by Git. The hashes below identify local
campaign evidence; the checked-in implementation, tests, and replay commands
are the public reproduction surface.

| Artifact | SHA-256 | Meaning |
| --- | --- | --- |
| `synthesis.json` | `4a5871199ce5568fedbf016a4ab66c79cafb1cf707aebda2e97c4ee2c8c5d09e` | Raw, gold-blind strict Terra synthesis plus local numeric scores and receipts. |
| `synthesis-normalized.json` | `501708f2ab3bc2a10788745eaaa9f6b9307f34e2e554f7cd15488603d5cde28e` | Gold-blind normalization derivative; zero stage rows changed. |
| `scores.json` | `18771d773872d103e602fc74fb7679a425c4e4501f966505c35bd7a0b0e6f8ab` | Post-hoc score of the raw synthesis. |
| `scores-normalized.json` | `5e1028059e9696ef3dfe188103e2eb285c914ec791e0c083e92e4ac35a09a3d4` | Headline post-hoc score of the normalized synthesis. |
| `scores-normalized-replay.json` | `5e1028059e9696ef3dfe188103e2eb285c914ec791e0c083e92e4ac35a09a3d4` | Independent score replay; byte-identical to the headline score. |
| `synthesis-parts/q000.json` … `q009.json` | individually sealed | Question checkpoints used to assemble the synthesis. |
| `provider-calls/*.request.json` / `*.response.json` | 12 self-sealed pairs | Durable provider-call reservations and exact responses. |

Commands used:

```powershell
$root = "eval_results/longmemeval-1m-recall-guarded-cumulative-litellm-terra-synthesis-development-20260821"
$dataset = "C:\path\to\memory-condense-rig\datasets\longmemeval_s_cleaned.json"

pixi run --frozen -e dev python -u tools/run_recall_guarded_cumulative_synthesis.py `
  --phase synthesize `
  --retrieval eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821/retrieval.json `
  --model-dir .cache/models/Qwen3-0.6B `
  --output-root $root `
  --provider-model openai/codex_sdk/gpt-5.6-terra `
  --attempt-structured `
  --authorized-provider-calls 12 `
  --max-new-tokens 4096 `
  --gpu-memory 6GiB

pixi run --frozen -e dev python -u tools/run_recall_guarded_cumulative_synthesis.py `
  --phase normalize `
  --output-root $root

pixi run --frozen -e dev python -u tools/run_recall_guarded_cumulative_synthesis.py `
  --phase score `
  --dataset $dataset `
  --synthesis "$root/synthesis-normalized.json" `
  --scores-name scores-normalized.json `
  --output-root $root
```

The synthesis/provider/cumulative/architecture regression set passed 253
tests immediately before provider execution. The architecture refactor keeps
every source module below the repository's 1,300-line reviewability gate.

## Interpretation and remaining limits

1. A capable large-context synthesizer was necessary. Retrieval was held
   fixed, so the large improvement isolates answer synthesis rather than a
   retrieval-stack change.
2. S1 is the best measured retrieval stage. The five S2-only additions were
   low-value under both Terra's semantic labels and local numeric scoring; S3
   had zero additions.
3. Exact match understates the S2/S3 answer on one six-item list because only
   delimiters changed. The component metric preserves that distinction.
4. Cited-source precision and quote grounding are strong, but no independent
   semantic judge ran. Terra generated the answers and its own role/density
   labels; those labels are not independent evaluation.
5. The ten questions are a development slice from the original concatenated
   memory test. This result does not establish held-out generalization,
   external competitiveness, or the 95% target.
6. The next retrieval experiment should stop adding unfiltered S2 evidence
   and instead use the now-measured role/density signals in a new matched
   selection arm. The present artifact must remain immutable evidence, not be
   retroactively repacked.

## Conclusion

The missing episodic scoring and large-context synthesis paths now exist and
have been exercised end to end. Terra converts the same frozen cumulative
contexts from a local-generator failure into a useful 5/10 exact, 0.718433 F1
S1 result with mechanically grounded citations. The linear retrieval result
is also clear: direct episodes help; the current representative additions do
not; artifact-global closure adds nothing under the frozen cap. Future gains
should come from a new density-aware S2 selection treatment and held-out
answer evaluation, not from quietly replacing the strongest prior packet.
