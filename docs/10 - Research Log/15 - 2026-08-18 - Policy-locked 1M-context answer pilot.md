# Policy-locked 1M-context answer pilot

**Status:** the ten-question LongMemEval-S development pilot passed 10/10 by
the independent judge under the hard prompt cap. This is a development
milestone, not the required held-out, minimum-100-question validation and not
a Mem0 comparison.

## Result

One synthetic long-chat sample contained 5,400 turns and 1,039,203 local
cl100k transcript-token proxies. The responder received only the retrieved
packet and dated question;
it received no recent transcript turns, gold answer, or judge material.

| Metric | Result |
| --- | ---: |
| Operational judge accuracy | **10/10 (100%)** |
| Mean token F1 | 0.775 |
| Exact match | 50% |
| Mean retrieved context | 1,980.8 tokens |
| Mean responder prompt (legacy local proxy) | 2,342.1 cl100k content tokens |
| P95 / maximum responder prompt (legacy local proxy) | 2,613 / 2,613 cl100k content tokens |
| Transcript-token saving | 99.81% |
| Raw evidence-source coverage | 100% |
| Packed evidence-source coverage | 100% |
| Scored multi-value coverage | 100% (2 questions) |
| Selector / score-provider fallbacks | 0 / 0 |

The responder answered the formerly failing derived questions correctly:
four days, three weeks, and 190 remaining pages. It also returned all five
concert events and all six museums in the correct order. The exact-match
score is lower because semantically correct answers included units or omitted
gold-answer prose; the independent judge accepted all ten.

The central-dev gateway completed ten responder and ten judge calls. It did
not return provider token-usage counters, so the report records zero provider
input/output tokens; the repository's own cl100k message-content proxy supplied
the prompt measurements above. That historical report did not count chat
framing and therefore is not an exact provider-token measurement. The current
harness labels this quantity as a proxy, binds the vocabulary identity, adds
an explicit framing reserve, and checks nonzero provider input usage.

## Retrieval treatment

The selected treatment used:

- BGE-M3/BM25 coarse routing with role, TF-ISF, HSC, partition, neighbor, and
  causal-consolidation arms;
- a frozen two-layer prefix of Qwen3-8B, inspecting layer 1 QK/OV features
  without an LM head;
- a generation-free Qwen3-0.6B forced-choice scorer with K/V cache disabled;
- an exhaustive typed scan of every content chunk in the selected partitions;
- transient event identities for venue visits and completed performances;
- recall-safe, representative-first packing under a 2,250-token expansion
  budget; and
- zero persisted request-derived token activations, attention maps, residuals,
  or K/V state. Static checkpoint weights and tokenizer assets are reusable
  machinery and are outside this retained-state metric.

The performance identity repair contracted eight q3 mentions to five events:
Billie Eilish, the outdoor concert series, the Brooklyn festival, jazz night,
and Queen with Adam Lambert. The five primary evidence chunks occupied packed
ranks 1–5. The museum query contracted to six structural identities and used
the explicit selected-scope fixed-K closure, producing a 580-token packet.

## Provenance and limits

The final no-provider selection artifact is
`eval_results/qwen-choice-coverage-full10-event-dedup-fp16-final.csv`, SHA-256
`4acd735102c4af386c78b12c00b90a18b49319cbe6e3d43d8b9eba0d088ec7d6`.
Its prompt traces were identical to the preceding auto-dtype run; explicit
FP16 was rerun so policy identity would not depend on hardware.

The frozen development policy is
`data/longmemeval-qwen-choice-coverage-operational-development-v2.json`,
SHA-256
`7f688c90bb4f49b7ca83ac72e27d65b1d707a6f7164c724e3ce06dd043188bbd`.
The answer report is
`eval_results/benchmark_longmemeval_s_cleaned_120-250_k10_ef50_causal-graph-k10-s24-h2-part4-local-role-coverage-qwen-prefix-choice-qwen3-0-6b-selected-scope-closure_20260818_154904.json`,
SHA-256
`215f7fe27162f672b7a25f102102647f07cdd32d43242aba37dddf1f9dbd6307`.
It records implementation SHA-256
`706b7a37424a83bcde5c522a7b6547634c6fc2986fcf103f57599721c3887b76`
and environment-lock SHA-256
`058083871240979257ada7ca4c71dd816fee64792b275ef11e4857c9f5ebba33`.

The selected partition scan is exhaustive only inside the four routed
partitions. The q8 closure is therefore labeled `selected_scope_policy` and
`closure_global_recall_guaranteed=false`; it is not a corpus-completeness
proof. Q3 needed no closure. The development questions helped select this
treatment, so 10/10 cannot be reported as held-out accuracy.

Two non-blocking weaknesses remain visible:

1. a first-person current-state question (Instagram followers now) is
   miscompiled as set `COUNT` and wastes 51 clusters; and
2. a fixed-three event query reserves two assistant recommendations before
   the three exact user facts, which survive only through the fail-open tail.

Both packets were sufficient in this pilot, but the rules should be corrected
before a tighter-token or broader campaign.

## Verification and next gate

After the performance-event repair, normal project execution reported:

```text
pixi run -e dev python -m pytest -q
1069 passed, 1 unrelated pydantic-settings warning
```

The next evidence gate is at least 100 held-out LongMemEval questions with the
same policy fixed in advance, followed by a same-split, same-budget Mem0 arm.
The provider-call count and cost for that campaign must be authorized before
launch. Until then the accurate claim is: **the system passed a policy-locked
ten-question development pilot while replacing a 1M-token transcript with an
average 2.3k-token responder prompt.**
