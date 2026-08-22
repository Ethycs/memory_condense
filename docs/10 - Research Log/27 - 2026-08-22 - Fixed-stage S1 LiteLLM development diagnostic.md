# Fixed-stage S1 LiteLLM development diagnostic

**Status:** a live ten-question development diagnostic completed through the
new fixed-stage answer and judge paths. The answer run made exactly 10
physical Terra calls and the independent judge made exactly 10 physical Sol
calls, all with zero retries. Sol scored 9/10 and correctly classified the
formal gate as `insufficient_population`. Byte-identical replay passed. This
does **not** establish a 100-question result, >=95% accuracy, provider
persistence, or a fair Mem0 comparison.

This is the first live exercise of the launch surface frozen in
[Research Log 26](26%20-%202026-08-22%20-%20Fixed-stage%20S1%20and%20locked%20100Q%20campaign.md).
It deliberately reuses the original ten-question development retrieval from
[Research Log 22](22%20-%202026-08-21%20-%20Recall-guarded%20cumulative%20retrieval.md)
and answers only the preregistered S1 `direct_episode_additions` stage. It is a
pipeline diagnostic before the separate locked 100-question validation
campaign, not a substitute for that campaign.

## Bottom line

| Property | Observed result |
| --- | --- |
| Population | original ten-question development concatenation; 1,039,203 transcript-token proxies and 5,400 turns |
| Retrieval | immutable provider-free cumulative retrieval artifact |
| Answered stage | S1 `direct_episode_additions` only |
| Terra execution | 10 logical prompts, 10 unique prompts, 10 noncached physical calls, 0 retries |
| Sol execution | 10 logical judgments, 10 unique prompts, 10 noncached physical calls, 0 retries |
| Independent semantic result | 9 correct, 1 incorrect, 0.900000 binary accuracy |
| Formal target | >=0.95 accuracy on at least 100 questions at one fixed stage |
| Gate result | `insufficient_population`; `gate_passed=false` |
| Replay | byte-identical answer and judge artifacts; no new provider calls |
| Mem0 | not run or scored |

The 9/10 score is useful development evidence that the new 8,000-input /
256-output fixed-stage plumbing works end to end. It is neither a passing
score nor an estimate on the locked validation population.

## Execution boundary and artifact roots

The first invocation ran inside the filesystem sandbox and reached the
request-first journal boundary, but local network access to the LiteLLM
gateway was denied before a response was recorded. That aborted campaign was
preserved at:

```text
eval_results/longmemeval-1m-recall-guarded-cumulative-fixed-stage-final-answer-v1-development-20260822/
```

It contains exactly one request journal and no response journal or
`final-answers.json`:

| Preserved request-only identity | SHA-256 |
| --- | --- |
| Call key | `b7df1f59a455ca70d278aec7e8e63f54a6c3f2b4cb4871119cc7e03b4ee145d8` |
| Request journal identity embedded in the canonical object | `eb03782bb3a43b9a4023d21bfb39010775dd5eb5e20293a07277ab07f9f331ef` |
| Physical request-file bytes | `7ea3b2fff76b369c3b18cf86329e1c6b1395eec2c75c8ace87a694387ad96729` |

The request bytes are identical to the corresponding first request in the
successful campaign. The fixed-stage runtime treats a request without a
response as terminal uncertainty, so that root was not retried or silently
completed. The network-authorized run used a separately named root:

```text
eval_results/longmemeval-1m-recall-guarded-cumulative-fixed-stage-final-answer-v1-development-network-authorized-20260822/
```

Keeping both roots makes the failed boundary visible and prevents the later
success from being misrepresented as a retry of an uncertain journal entry.
The failed root proves only that a request reservation was published and no
response was retained; it does not independently certify what an external
provider may or may not have observed.

## Locked live routes and budgets

Both live stages used the same LiteLLM-compatible gateway,
`https://central-dev.zt:4000/v1`, through separate exact model routes:

| Property | Terra responder | Independent Sol judge |
| --- | --- | --- |
| Caller model | `openai/codex_sdk/gpt-5.6-terra` | `openai/codex_sdk/gpt-5.6-sol` |
| Gateway model | `codex_sdk/gpt-5.6-terra` | `codex_sdk/gpt-5.6-sol` |
| Maximum prompt proxy | 8,000 | whole-population preflight; observed maximum 229 |
| Maximum output tokens | 256 | 1,024 |
| Temperature | omitted / `null` | omitted / `null` |
| Provider retries | 0 | 0 |

The Terra preflight reconstructed every selected provider message before the
first call. Its maximum prompt was 7,283/8,000 proxy tokens. The ten live
completion reports all carry `physical_call=true`, `cache_hit=false`, and
`retries=0`; the final cumulative counters are 10 logical, 10 unique, 10
physical, and 0 checkpoint hits. Gold fields are absent from the answer
artifact.

The answer stage reports 68,284 input-token proxies and 88 output-token
proxies across the population, with 48.9597 seconds of accumulated call
latency. These are locally reconstructed proxy counts. The gateway returned
no usable provider token-usage fields, so the artifact does not relabel the
proxies as provider billing counts.

The Sol runner validated the complete answer artifact against its retrieval
before loading gold. Its ten live completion reports likewise carry
`physical_call=true`, `cache_hit=false`, and `retries=0`; final cumulative
counters are 10 logical, 10 unique, 10 physical, and 0 checkpoint hits. The
judge reports 1,532 input-token proxies, 179 output-token proxies, and 58.7898
seconds of accumulated call latency.

The responder records zero persisted local transformer token state. External
provider persistence is explicitly `not_certified`; neither the local receipt
nor this live run makes a claim about state retained behind the gateway.

## Independent result

The canonical Sol score is:

```text
correct=9
incorrect=1
questions=10
binary_accuracy=0.900000
minimum_questions=100
target_accuracy=0.950000
minimum_population_met=false
accuracy_threshold_met=false
gate_passed=false
status=insufficient_population
```

Nine predictions were accepted as semantically correct. The sole negative
verdict was the development knowledge-update question `a2f3aa27`:

| Field | Value |
| --- | --- |
| Question | `How many followers do I have on Instagram now?` |
| Gold answer | `1300` |
| Terra prediction | `Close to 1300` |
| Sol verdict | `INCORRECT` |
| Sol reason | the prediction was approximate while the gold answer specified exactly 1,300 followers |

This is an approximate-answer rendering miss, not a new retrieval-admission
measurement. The historical packet contained the newest update in
approximate form, and Terra preserved that form. The binary judge applied a
strict distinction between `Close to 1300` and `1300`.

## Historical prompt bytes versus the validation correction

The historical retrieval was sealed before the approximate-current-value
instruction was added to the shared QA prompt constructor. Its selected
system-message quote SHA-256 is
`728d9538e1efd1119d79f81b932d147ade876250b22ba8f0011760073784f2ca`,
and the complete historical responder prompt-policy SHA-256 is
`ebe671a232197818db00adb14a8f9ca3e649fe9f230abca60de09633d70c3f55`.
The fixed-stage runner consumed those exact stored provider messages; it did
not rewrite them merely because newer source code existed.

New validation prompts already include the correction: when the newest user
update states an approximate current value such as “close to 1300 now,” the
responder should return the stated number and should not abstain merely because
the value is approximate. That instruction was frozen into the validation
prompt constructor before validation construction. It addresses the observed
development failure mode prospectively while preserving the historical
diagnostic's immutable prompt bytes and identity.

This separation matters. Retrofitting the old ten prompts and rerunning them
would be another development experiment with a new prompt policy; it would
not be a replay of this diagnostic.

## Replay result

The Terra replay reopened the existing journals under the replay-only,
no-provider-client contract and republished byte-identical
`final-answers.json` bytes. Its console summary reported the sealed population
as `unique_provider_calls=10`; that field is the artifact's unique prompt
count, not ten new session calls. The replay path cannot construct a provider
client, and no new request/response journals were added.

The Sol replay explicitly reported `session_physical_calls=0`, reproduced the
same 9/10 `insufficient_population` result, and republished byte-identical
score bytes.

| Artifact | Live SHA-256 | Replay SHA-256 | Replay provider calls |
| --- | --- | --- | ---: |
| `final-answers.json` | `d85a37dad4ce4ec9cf21d59e3600bf7abdafd4347486dfc87b3b3a95177de84a` | `d85a37dad4ce4ec9cf21d59e3600bf7abdafd4347486dfc87b3b3a95177de84a` | 0 by no-client replay contract |
| `final-answer-semantic-judge-sol.json` | `dbeb7a54b3bbf40958c32f649eab711a2ebb0fa4f9c64cb749a1198b6ae869f0` | `dbeb7a54b3bbf40958c32f649eab711a2ebb0fa4f9c64cb749a1198b6ae869f0` | 0 observed in the replay session |

The files and their `.sha256` sidecars agree. The retained completion reports
inside a replayed artifact still describe the original physical calls; their
`physical_call=true` fields are immutable provenance, not evidence that replay
called the provider again.

## Artifact identities

Successful root:

```text
eval_results/longmemeval-1m-recall-guarded-cumulative-fixed-stage-final-answer-v1-development-network-authorized-20260822/
```

| Identity | SHA-256 |
| --- | --- |
| Historical retrieval artifact | `aa22f7c18470d9a7c931fd16f8f58bf67d8566e2298a45371ee2815c11a9bd97` |
| Gold-blind development population | `fa9a06ebd103d87086943cfa94091bdf607fe07874bc871e465aad409b85ca18` |
| Selected S1 stage population | `e9330ab89c2d2475b38a6c7fdfce6329a3d7eee9167ce35e0c31e9341c43c9d4` |
| Ordered question population | `7220f1ea80c436ba26920bf6cc525c8366a40044e0bd19127a132a214d4c22fa` |
| Terra provider-prompt population | `182ada7ac1dce65ef7a55ef5f5cf7591d7cf1d4c30069d77d3d1b5e348a5f33c` |
| Terra runtime prompt population | `fe2f6c7fdf9c5f40fe404c59ddb8789b077e70a14e2fde6e927e129e97cedd5f` |
| Fixed-stage implementation | `020e5ba816c2246ba021944d1e847aa9a96ce2f7d0caa2e808d66c11ba0c5c92` |
| Fixed-answer policy | `ea43d5b5a0ef5e95453d03917b37ea5385feea89b7546ea6841da1c9fe866418` |
| Historical responder prompt policy | `ebe671a232197818db00adb14a8f9ca3e649fe9f230abca60de09633d70c3f55` |
| Terra runtime identity | `afe6fbd3ffc67ea06023abe1aca7dde6b82e1cd9cae8ca19548aa9cf3bc81119` |
| Terra campaign binding | `19647c754981e4b08bcad1a1f08fce665494eb0e6238e4d003e8e687552284dc` |
| `final-answers.json` file | `d85a37dad4ce4ec9cf21d59e3600bf7abdafd4347486dfc87b3b3a95177de84a` |
| Gold scoring population | `1f5155a0450281fd50ea87a7feaee8c3626be392504682133546ebae4e214a84` |
| Sol prompt population | `677454ef3a2f9fcba3da0ddd0bdc4982d52c668d86d59bef647660b88451728c` |
| Ordered judgment population | `076b0c7b70751e0d2841bb9126cd66282ac099511200e49969a6b15f905a9e2c` |
| Semantic-judge policy | `29a58b49db746d0c9a09bcd684cc8ee397a56df7d78c139adc567f89383cfa50` |
| Sol runtime identity | `0cbe9e3309c346f08f1f8b56869e3a2ea5140bd7b761b101f6c0c54955072574` |
| Sol campaign binding | `1a5f2a23fd3c8bf1b730727186e9bbda9a927ff4d96302534ef82eb58d102800` |
| `final-answer-semantic-judge-sol.json` file | `dbeb7a54b3bbf40958c32f649eab711a2ebb0fa4f9c64cb749a1198b6ae869f0` |

Canonical artifacts:

- [`final-answers.json`](../../eval_results/longmemeval-1m-recall-guarded-cumulative-fixed-stage-final-answer-v1-development-network-authorized-20260822/final-answers.json)
- [`final-answer-semantic-judge-sol.json`](../../eval_results/longmemeval-1m-recall-guarded-cumulative-fixed-stage-final-answer-v1-development-network-authorized-20260822/final-answer-semantic-judge-sol.json)
- [preserved sandbox-blocked request](../../eval_results/longmemeval-1m-recall-guarded-cumulative-fixed-stage-final-answer-v1-development-20260822/final-answer-calls/b7df1f59a455ca70d278aec7e8e63f54a6c3f2b4cb4871119cc7e03b4ee145d8.request.json)

## What this does and does not establish

This run establishes that the fixed-S1 answer pipeline can validate the old
1M development retrieval, stay under the registered prompt cap, make exactly
the authorized unique Terra calls without retries, publish a gold-blind
answer artifact, hand it to an independently routed Sol judge, and replay both
artifacts without provider access.

It does not meet the formal gate for two independent reasons: the measured
accuracy is 0.90 rather than >=0.95, and the population is 10 rather than at
least 100. The authoritative status is therefore `insufficient_population`,
not pass. The 100-question validation retrieval, 100 Terra answers, and 100
Sol verdicts remain separate work and must produce their own identities and
score.

No Mem0 production arm was run, so there is no paired Mem0 metric or fairness
claim. No external-provider persistence guarantee was obtained. The result
should be cited as a live, replayable development diagnostic of the fixed
answer-and-judge path—and nothing broader.
