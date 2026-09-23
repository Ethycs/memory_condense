# Source-local contextual cards with LFM2 Transcript

**Date:** 2026-09-05

**Status:** sidecar-only prototype implemented and locally exercised. The
three-case synthetic assay passes the composed source-gate-plus-attention path;
no live retrieval path, database schema, 1M corpus, or benchmark score changed.

## Question

Can more query-independent work move to ingestion by summarizing each new
memory relative to its last N same-source memories, then searching those small
summaries with the existing attention linker and hydrating raw chunks only
after selection?

## Decision

Yes, as a derived routing plane. Use `LiquidAI/LFM2-2.6B-Transcript` as the
first synthesis candidate, not the original `LFM2-350M-Extract` checkpoint.

The choice is empirical as well as task-driven. Liquid describes Transcript as
a single-turn, on-device model for long-form transcript summarization with key
points, decisions, and action items. That is closer to contextual memory
synthesis than field extraction. The [official model
card](https://huggingface.co/LiquidAI/LFM2-2.6B-Transcript) describes a roughly
2.6B-parameter, English-only transcript specialist and recommends a
temperature of 0.3. This assay deliberately used greedy decoding so each raw
completion could be sealed and replayed.

The original [350M Extract
model](https://huggingface.co/LiquidAI/LFM2-350M-Extract) remains a useful small
baseline. It is not promoted: Liquid's [deprecation
table](https://docs.liquid.ai/lfm/help/deprecations) lists 350M Extract with
LFM2.5-350M as its replacement, and the local contextual assay showed that
schema extraction ability did not imply target-relative discourse reasoning.

## As-built pipeline

`raw append -> source-local last-N window -> Transcript card -> source gate -> Qwen card attention -> raw-window hydration -> post-selection deduplication`

The prototype stops at verified hydrated evidence. Terminal packing and the
primary-LLM handoff remain integration work.

The implementation is isolated in:

- `src/memory_condense/search/context_cards.py` for bounded windows, the
  provider-independent prompt, strict output validation, exact quote support,
  and deterministic receipts;
- `src/memory_condense/modeling/lfm_extract.py` for a lazy local LFM adapter
  that supports both single-file and sharded safetensors checkpoints;
- `src/memory_condense/search/context_card_retrieval.py` for the shadow lexical
  source gate, Qwen candidate conversion, target-first raw hydration, and
  after-selection exclusion/deduplication;
- `tools/assay_contextual_cards.py` for query-free compilation; and
- `tools/assay_contextual_card_attention.py` for replay-only card search.

No table, index, pending-work journal, authoritative memory row, or production
retrieval stage was modified. Runtime artifacts remain under `.tmp/` because
they contain the synthetic memory text and raw model completions; they are not
committed as research data.

## Card contract

“Last N” means up to N memories preceding the target in the same source, plus
the target itself. Chunks within one turn use their source character offset as
a second ordering coordinate. The current default is N=16 with a 768-token
proxy ceiling.
Oldest context is dropped first; the target is never dropped or truncated. A
target that cannot fit by itself produces no card.

The model sees local aliases such as `M1`, never durable memory IDs, queries,
answers, or benchmark gold. The target is rendered first to match the
Transcript model's summarization behavior. The v4 model-facing record contains
one short statement, one exact target quote, an optional exact prior alias and
quote, topics, and entities. Exact keys and hard list/text bounds are enforced.

Every accepted statement has an exact contiguous target quote. This is a
structural grounding check, not an entailment check on the generated
paraphrase. The card also seals the IDs, order, source, and exact text hash of
the whole bounded window. After a card wins, hydration verifies selected raw
support, then applies S0/EM exclusion and cross-card deduplication. The
generated card is not factual authority; the intended primary-LLM handoff is
the verified raw chunks.

This provides hard guarantees for source locality, bounded input, exact target
quotes, immutable-window binding at compilation, and deterministic receipts.
The retrieval result explicitly signals when raw fallback is required; the
production union that executes that fallback is not wired yet. It does
**not** prove that a generated paraphrase is entailed by its quotes or that a
bounded window is semantically complete.

## Local model comparison

The checkpoints were pinned to immutable Hugging Face revisions:

| Model | Revision | Weight identity |
| --- | --- | --- |
| LFM2-350M-Extract | `d99a6f06ea16a2f83998789389a64b66d40c4198` | `c8160a82ffc7d91dc9a7567bf399c3c461ebfcdec14f986139f748149d2d2991` |
| LFM2-2.6B-Transcript | `1b607be3f244de841a55c9fe426713dd950ab281` | shard 1 `33937415ba3ee5cd4402ddb0f417ca7264a566d57aba06bae7b2006583e11389`; shard 2 `d6f2ac1348cbf73893845d0f46edd4bd1edd24638ecdbe3f8753e9d900ae110e` |

Both ran through Transformers 4.57.6 on an RTX 2070 SUPER in float16 under the
same current one-summary v4 prompt, validation contract, decoding policy, and
three targets. The
fixture has two interleaved sources and three targets: a simple garden
instruction, a pronoun-bearing schedule correction, and a dependent action
item.

| Treatment | Structurally accepted cards | Measured compile interval | Observed behavior |
| --- | ---: | ---: | --- |
| 350M Extract | 0/3 | 28.209 s | copied a prior-memory quote into the required target-quote field in all three cases, so all failed closed |
| 2.6B Transcript | 3/3 | 40.594 s | produced usable summaries for all targets and resolved “move it” to the Cobalt launch; semantic correctness was not validator-proven |

The measured interval includes checkpoint receipt hashing, model load, three
generation calls, and model close, but not final artifact serialization and
publication. Transcript's cold load was 14.827 s. Its generation calls were
6.507, 7.687, and 6.713 s for 91, 107, and 101 output tokens. Peak allocated CUDA
memory rose from 5,205,994,496 to 5,222,802,944 bytes. The query-free compile
artifact is `.tmp/contextual-cards-transcript-full-v5.json`, file SHA-256
`0312a8fc986815ee3465f90d5461aea465f453902f13e08f2fdcc2076152d309`.

The matched Extract control artifact is
`.tmp/contextual-cards-extract-v4-control-v1.json`, SHA-256
`d3177c8587efb6dd1e5865b4244014a2d8c47123ecf882dc55312d8272b87005`.

## Retrieval result

Searching all three cards with Qwen attention alone reached 2/3 top-one on the
synthetic questions. It misrouted the Friday/watering question to an Atlas
card. Qwen loaded in 15.235 s and the three searches took 0.416, 0.103, and
0.092 s. The artifact is
`.tmp/contextual-cards-transcript-attention-v3.json`, SHA-256
`5dba9f131d19e61bd9085f60a385f542c7cbd1e34368b8b6573773eca1a03d42`.

The intended pipeline first used the cheap card-text source gate, then ran
Qwen attention only inside the routed source. It reached 3/3 expected top-one
on this fixture. Qwen loaded in 15.130 s and the hardened replay searches took
0.393, 0.069, and 0.058 s
with two attention passes each. They
inspected two candidate instances for the garden route and four for each Atlas
route, using at most 109 Qwen workspace tokens. Raw hydration returned 2, 3,
and 4 complete source-local chunks respectively. The artifact is
`.tmp/contextual-cards-transcript-composed-v4.json`, SHA-256
`b4deb50b86c452bb5f6290ecb326709b45e585d9e1355c2af7739cd80ad3c5b1`.

This is evidence for composition, not evidence that 3/3 generalizes. The
lexical gate in this assay is a small shadow stand-in; a promoted treatment
must consume the existing proof-carrying source route rather than create a
second production router. Accordingly, all three narrowed shadow-gate results
now carry `requires_raw_fallback=true`: the card hit is useful, but it cannot
erase the existing raw/source candidate union.

### Sustained warm timing

A 20-round replay repeated the same three queries 60 times with one resident
Qwen instance per treatment. After discarding the first three-query warm-up
round, both arms contain 57 timed queries over the same sealed Transcript card
artifact and Qwen checkpoint.

| Retrieval arm | Mean | Median | p95 | Queries/s | Candidate inspections | Passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All-card attention | 98.95 ms | 93.37 ms | 137.44 ms | 10.11 | 342 | 171 |
| Shadow source gate + card attention | 64.31 ms | 60.45 ms | 89.34 ms | 15.55 | 190 | 114 |

The source gate reduced mean, median, and p95 latency by approximately 35%,
raised measured query throughput by 53.9%, reduced candidate inspections by
44.4%, and reduced attention passes by 33.3%. The all-card arm repeated its
2/3 top-one result; the composed arm repeated 3/3.

This is a speed bonus from pruning the Qwen candidate workspace, not from
Transcript generation. It is not yet a safe end-to-end latency claim because
all 60 shadow-gated rows correctly request additive raw fallback and this assay
does not execute or time that production union. The artifacts are
`.tmp/contextual-cards-transcript-attention-speed-v2.json`, SHA-256
`9cea221598560fa6095eb58e60dd596d36adbda7eef59ca2351b22fd1a8abc72`, and
`.tmp/contextual-cards-transcript-composed-speed-v1.json`, SHA-256
`7611031017510f14a91ef7ee1610e910856dea81c20cedeaddf4ef3e263a95de`.

## Performance interpretation

Warm card retrieval is already subsecond in this tiny assay. Cold model loads
must not occur per prompt; both the compiler and Qwen prefix need resident
workers. Sequential Transcript compilation at roughly 7.5 s/card is not yet a
high-throughput ingestion result. The next speed work is batch generation,
bounded asynchronous T2 compilation, and a Q4_K_M/GGUF or another measured
quantized runtime. The card prompt itself was 391--462 model tokens while the
raw windows were only 52--114 proxy tokens, so prompt/schema amortization is a
real target.

The 2.6B native checkpoint narrowly fits this 8 GB GPU at the tested window.
Liquid's “under 3 GB” statement applies to optimized local deployments and
must not be attributed to this native float16 run. The measured native peak
was about 5.22 GB allocated by PyTorch.

On the matched three-target compile, Transcript was not a raw ingestion-speed
win: compared with Extract it was 43.9% slower over the measured interval and
28.6% slower in generation. Its benefit was yield—3/3 structurally accepted
cards versus 0/3—rather than lower latency. With the model resident, the
observed interval after load corresponds to about 419 valid cards/hour before
batching.

## Promotion gates

1. Compile a representative, query-free corpus sample and report valid-card,
   exact-target-quote, unsupported-entity, empty-card, and fail-open rates.
2. Compare Transcript against current LFM2.5 350M/1.2B and Transcript
   Q4_K_M under the same card contract; model size alone does not decide.
3. Add batched/resumable background compilation and demonstrate searchable
   ingest remains faster than foreground generation with bounded T2 lag.
4. Feed cards only from the existing source scope; a lexical miss or stale,
   missing, invalid, or over-budget card must preserve raw candidates.
5. On the analysis-used validation100, measure raw target/evidence containment,
   prompt tokens, retrieval latency, and matched answer score. No card arm may
   replace the protected 95/100 parent unless it has no regressions.
6. Keep the disjoint confirmation population unopened until the complete
   policy, model revision, quantization, budgets, and fallback behavior freeze.

## Verification

The focused and neighboring card/Qwen suite currently passes 50 tests. The
exact local commands were:

```powershell
.\.pixi\envs\dev\python.exe -m pytest tests/test_context_cards.py tests/test_context_card_retrieval.py tests/test_lfm_completion.py tests/test_qwen_memory_linker_early_exit.py tests/test_qwen_prefix.py tests/test_episode_representative_retrieval.py -q -m 'not slow' --basetemp .pytest-context-card-final-v4

.\.pixi\envs\dev\python.exe tools/assay_contextual_cards.py --smoke `
  --model-dir F:\Keytone\Documents\GitHub\memory_condense\.cache\models\LFM2-2.6B-Transcript `
  --model-id LiquidAI/LFM2-2.6B-Transcript `
  --model-revision 1b607be3f244de841a55c9fe426713dd950ab281 `
  --output .tmp\contextual-cards-transcript-full-v5.json `
  --previous-memories 4 --max-new-tokens 256

.\.pixi\envs\dev\python.exe tools/assay_contextual_cards.py --smoke `
  --model-dir F:\Keytone\Documents\GitHub\memory_condense\.cache\models\LFM2-350M-Extract `
  --model-id LiquidAI/LFM2-350M-Extract `
  --model-revision d99a6f06ea16a2f83998789389a64b66d40c4198 `
  --output .tmp\contextual-cards-extract-v4-control-v1.json `
  --previous-memories 4 --max-new-tokens 256

.\.pixi\envs\dev\python.exe tools/assay_contextual_card_attention.py --smoke `
  --cards .tmp\contextual-cards-transcript-full-v5.json `
  --qwen-model-dir F:\Keytone\Documents\GitHub\memory_condense\.cache\models\Qwen3-8B `
  --output .tmp\contextual-cards-transcript-attention-v3.json `
  --attention-only

.\.pixi\envs\dev\python.exe tools/assay_contextual_card_attention.py --smoke `
  --cards .tmp\contextual-cards-transcript-full-v5.json `
  --qwen-model-dir F:\Keytone\Documents\GitHub\memory_condense\.cache\models\Qwen3-8B `
  --output .tmp\contextual-cards-transcript-composed-v4.json
```

No remote provider, benchmark responder, or judge call was made. Model weights
were downloaded from the official public repositories. The 3/3 score is a
synthetic mechanism smoke, not 1M retrieval accuracy and not evidence toward
the frozen confirmation score.
