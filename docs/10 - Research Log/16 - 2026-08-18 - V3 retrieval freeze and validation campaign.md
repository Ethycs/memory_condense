# V3 retrieval freeze and validation campaign

**Status:** the final no-provider LongMemEval-S development replay passed the
retrieval/context-sufficiency gate and is frozen as the v3 treatment. The
minimum-100-question held-out campaign is fully specified but has not made any
provider calls. This is still development selection evidence, not held-out
accuracy or a complete-context-replacement certification.

## Final development replay

The locked development stress sample contained 5,400 turns and 1,039,203
cl100k transcript-token proxies. The replay rebuilt revision-3 compiled and
causal caches, then staged BGE-M3, a generation-free Qwen3-0.6B choice scorer,
and layers 0–1 of Qwen3-8B without an LM head. It made zero API calls.

| Metric | Result |
| --- | ---: |
| Questions | 10 |
| Raw evidence-source coverage | **100%** |
| Packed evidence-source coverage | **100%** |
| Questions with all packed evidence sources | **10/10** |
| Scored answer-value components | **11/11 (100%)** |
| Mean / maximum returned context | **1,985.6 / 2,219 tokens** |
| Mean transcript-token saving | **99.81%** |
| Selector applied / bypassed | 3 / 7 |
| Selector / score-provider fallbacks | **0 / 0** |
| Maximum retained request-token state | **0 bytes** |

The legacy whole-answer literal metric was 50% in the returned context. That
metric is not the operational objective: derived answers and multi-item lists
need not appear as one verbatim string. The source-provenance metric reached
every required source, while the post-budget value-component metric found all
eleven individually checkable list values.

The targeted repairs behaved as intended:

- q3 contracted eight concert/performance mentions into five distinct events
  and reserved all five answer-bearing representatives;
- q6 reserved exactly three user-role facts ahead of assistant suggestions,
  with zero cardinality deficit;
- q8 retained six distinct museum identities and closed the tail to a
  580-token packet; the closure is explicitly
  `selected_scope_policy` with `closure_global_recall_guaranteed=false`; and
- the first-person current-value query bypassed set coverage with zero prefix
  or choice inspections.

## Frozen artifacts

The detailed local CSV remains ignored evaluation output because it contains
large traces. A compact, text-free, tracked selection artifact binds its hash
and records only aggregate and per-question status fields.

| Artifact | SHA-256 |
| --- | --- |
| `eval_results/qwen-choice-coverage-full10-scalar-role-fixed-fp16-v3.csv` | `df12c9d5cfebe591d7780808046acc601a61b471a207df7733f08dfc73c907f9` |
| `data/longmemeval-qwen-choice-coverage-selection-development-v3.json` | `a82a3ffb2880121e3952f0e581c2affe199e48e2a3d0cdddf2fe09492b6e4a3e` |
| `data/longmemeval-qwen-choice-coverage-operational-development-v3.json` | `5ea9352372414a34805d5dd5c406aaad7f457a56b8d978cc87cf7dbbc6b15c54` |
| `data/longmemeval-qwen-choice-coverage-operational-validation-v3.json` | `5263d5afd15298ec4088db9d6381ae243ddb685e9a3cf4d9892fc84e14fb9883` |

The frozen implementation SHA-256 is
`452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83`;
the Pixi environment-lock SHA-256 is
`058083871240979257ada7ca4c71dd816fee64792b275ef11e4857c9f5ebba33`.
The policies also bind the cleaned dataset, locked split, Qwen prefix and
choice manifests, the exact retrieval object, and the compact selection
artifact.

Prompt accounting now uses an explicitly named local proxy: the exact
cl100k vocabulary identity, eight reserved framing tokens per message, eight
fixed framing tokens, and a separate 256-token responder-output reserve.
Nonzero provider-reported input usage is checked after each call and overrides
the proxy for compliance reporting. Zero gateway usage remains “unavailable,”
not proof of a zero-token request.

## Cache and campaign proof

Compiled and causal cache revision 3 binds:

- the exact composed sample;
- BGE-M3 model revision/checkpoint and resolved execution controls;
- implementation and environment hashes;
- the SQLite and ANN bytes; and
- the compiled-to-causal cache-key link.

Validation scoring is cache-hit-only and opens both stores read-only. It cannot
build or update a missing cache while questions are live. Each sample report
carries the verified receipt pair, and the campaign merger independently
reconstructs and checks the frozen population.

The validation policy reconstructs exactly 100 unique held-out questions as
ten 1M-token shards at offsets `0, 10, …, 90`, ten questions per shard. Its
named claim profile is `longmemeval-s-1m-100q-95-v1`: at least 95% independent
judge accuracy, a hard 8,000-token local prompt-proxy cap plus provider-usage
postcheck, and `recent_window=4`. The plan was dry-verified end to end without
calling a model.

## Mem0 comparison boundary

The provider-free Mem0 adapter now matches the official raw LongMemEval
chronology and consecutive-pair protocol. On the locked validation population
it reconstructs 24,928 raw pairs, skips five pairs containing an empty message,
and therefore requires **24,923 Mem0 extraction calls**. Certified rendering
contains only Mem0 memory text and returned `created_at`; request-window source
attribution stays diagnostic-only. Certified state must use a fresh owned local
on-disk Qdrant instance, and failures poison the run until cleanup succeeds.

Mem0 OSS and its optional BM25/entity dependencies have not been installed or
invoked. A real comparison still needs a separate locked environment and
explicit authorization for its extraction-model cost. The same 1M-composite
workload is the direct comparison; the ordinary per-record official benchmark
is a related but differently shaped arm.

## Verification and next gate

Before the replay, normal project execution reported:

```text
pixi run -e dev python -m pytest -q
1247 passed, 1 unrelated pydantic-settings warning
```

The next safe step is blind preparation of the ten validation cache shards.
That step makes no provider calls. Scored validation would then require exactly
200 authorized logical calls: one responder and one independent judge per
question. Until those calls are explicitly authorized and the campaign merger
certifies the ten reports, the correct claim is: **the v3 treatment achieves
100% source and scored answer-value coverage on the locked 1M-token
development replay while returning an average 1,986-token context packet.**
