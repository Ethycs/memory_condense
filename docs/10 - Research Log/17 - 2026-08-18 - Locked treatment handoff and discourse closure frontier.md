# Locked treatment handoff and discourse closure frontier

**Status:** the frozen v3 treatment, its ten held-out cache shards, and the
provider-free comparison controls are implemented and locally verified. The
project goal is **not complete**: treatment held-out accuracy is still 0/100
questions scored, and the Mem0 production/extraction runtime is deliberately
NO-GO.

**Depends on:** the
[`v3 retrieval freeze and validation campaign`](16%20-%202026-08-18%20-%20V3%20retrieval%20freeze%20and%20validation%20campaign.md),
the
[`policy-locked development answer pilot`](15%20-%202026-08-18%20-%20Policy-locked%201M-context%20answer%20pilot.md),
and the
[`locked Mem0 comparison runbook`](../../tools/mem0_eval/README.md).

## Readiness at this handoff

| Gate | Current evidence | Status |
| --- | --- | --- |
| Frozen treatment | Exact dataset, split, implementation, environment, retrieval, prompt-proxy, and model identities are bound by the validation policy | Ready |
| Local model artifacts | The Qwen3-8B prefix manifest reverified across seven files; the Qwen3-0.6B choice manifest also reverified exactly | Ready |
| Held-out inputs | Ten distinct one-million-token shards at offsets `0, 10, ..., 90`, ten questions each | Ready |
| Compiled and causal caches | All ten receipt pairs independently rehashed, linked, and found free of build/WAL/SHM remnants; scoring is cache-hit-only and read-only | Ready |
| Treatment development evidence | Final v3 replay: 10/10 packed source closure, 11/11 scored value components, mean 1,985.6 returned tokens; preceding answer pilot: 10/10 independent-judge decisions | Development only |
| Treatment held-out answer accuracy | No responder or judge call has been authorized or made on validation | **0/100 scored; unknown** |
| Mem0 protocol and comparison tooling | Provider-free reconstruction, policy validation, call caps, prompt packing, reports, merger, and paired comparator are implemented and tested | Provider-free ready |
| Mem0 production arm | No frozen Mem0 lock/policy, concrete extraction/Terra/Sol transports, trusted factory, or closed HTTP-attempt receipts | **NO-GO** |
| At least 95% accuracy and fair Mem0 result | Requires both scored campaigns and paired certification | **Not established** |
| Grounded Discourse Closure RAG | General-purpose design below; no implementation or measurement yet | Proposed |

“Ready” in this table means the local, provider-free prerequisite has passed.
It does not mean that a scored campaign, external comparison, or production
service is operational.

## What the locked treatment does

The treatment reduces a million-token long chat in three bounded stages:

1. BGE-M3/BM25 coarse routing combines role-aware, TF-ISF, HSC, partition,
   neighbor, and causal-consolidation arms.
2. A two-layer Qwen3-8B prefix inspects layer-1 QK/OV features without loading
   an LM head. A Qwen3-0.6B forced-choice scorer ranks candidates without
   generation and with K/V caching disabled.
3. Typed set/query handling and recall-safe representative-first packing
   preserve answer-bearing raw excerpts under a fixed packet and prompt
   budget. Selected-scope closure is explicitly non-global; it is never
   reported as corpus completeness.

The final v3 no-provider replay reached 100% raw and packed source coverage and
11/11 answer-value components on the ten development questions, with mean and
maximum returned contexts of 1,985.6 and 2,219 tokens. The earlier v2 answer
pilot reached 10/10 judge accuracy, 0.775 mean token F1, and 50% exact match.
These are two different development results: the v3 retrieval replay did not
make answer-model calls, and neither result is held-out accuracy.

## The prepared 100-question population

The frozen validation plan composes exactly ten independent shards. Together
they contain 100 questions, 54,246 turns, and 10,441,617 transcript-token
proxies. Cache preparation produced 79,915 chunks and 23,917 learned episodes.
Every offset has a unique sample hash, compiled key, and causal key; every
causal receipt binds its compiled parent.

The caches live in the external evaluation rig rather than Git. Their durable
reproduction contract is the tracked validation policy and the independently
verified receipt procedure recorded in
[`Research Log 16`](16%20-%202026-08-18%20-%20V3%20retrieval%20freeze%20and%20validation%20campaign.md).
A held-out run must abort on a cache miss, identity mismatch, write attempt, or
receipt mismatch. It must not repair or rebuild a cache while questions are
live.

Offset 0 currently resolves to sample SHA-256
`41e52404d4f323c7add44a59a2faf8a58a95125d8e291cd9d118560833c5e14d`
with 5,551 turns and 1,041,276 transcript-token proxies. The tracked policy
reconstructs the other nine identities rather than trusting copied prose.

## Hard invariants

### Prompt budget

- Responder input has a hard 8,000-token local proxy cap.
- The proxy is `tiktoken==0.13.0`, `cl100k_base`, vocabulary SHA-256
  `8cd4fc3b76f9fdaf9df7d14f20a41eda79ce45b3e9c5ae8f68b0a41a59c3a9c9`,
  with eight framing tokens per message and eight fixed framing tokens.
- A separate 256-token responder-output reserve is reported in the request
  proxy; it is not silently subtracted from or confused with input usage.
- Nonzero provider-reported input usage is authoritative and is checked
  against the same cap. Zero provider usage means unavailable, never a
  zero-token request.
- Treatment and Mem0 prompts use the same dated question, responder model,
  judge model, cap semantics, and judge message. The comparator independently
  rebuilds both prompts before accepting a result.

### Exact provenance

Dataset bytes are read once, hashed, and used to decode both normalized and raw
views. The locked split and all stress shards are then derived in memory from
that same immutable snapshot. Cache receipts bind the composed sample, BGE
execution, implementation and environment, SQLite and ANN bytes, and the
compiled-to-causal link. Question reports retain the exact provider-visible
source excerpts, while retrieval diagnostics and cache receipts bind their
source identities. The campaign merger reconstructs the 100-question
population rather than trusting report summaries.

Mem0 OSS does not expose exact evidence grounding for inferred memories.
Consequently its request-window attribution remains diagnostic and the paired
report marks exact Mem0 source recall unavailable; it never substitutes that
proxy for treatment provenance.

### Zero persisted transformer token state

Request-derived tokens, hidden states, residuals, attention maps, and K/V
caches are transient and have a retained-state maximum of zero bytes. Static
checkpoint/tokenizer files, source text, typed graph records, scalar edge
statistics, and ordinary chunk embeddings are reusable data structures, not
persisted request-token activations. The choice scorer disables generation and
K/V caching, and the prefix path loads no LM head. Certification requires the
arm-specific zero-state contracts and receipts; absent or nonzero evidence
fails closed.

## Frozen identities

| Identity | SHA-256 |
| --- | --- |
| Cleaned LongMemEval-S dataset | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Locked split manifest | `8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4` |
| Validation policy | `5263d5afd15298ec4088db9d6381ae243ddb685e9a3cf4d9892fc84e14fb9883` |
| Development selection artifact | `a82a3ffb2880121e3952f0e581c2affe199e48e2a3d0cdddf2fe09492b6e4a3e` |
| Frozen `src/memory_condense` implementation | `452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83` |
| Root Pixi environment lock | `058083871240979257ada7ca4c71dd816fee64792b275ef11e4857c9f5ebba33` |
| Canonical 79-field retrieval identity | `08ffd89a8b30803a0d8121445c1d54171120b1f1e51c866d4015f2d36b87cbaf` |
| BGE-M3 checkpoint | `a3d5c49f064ab58d7cf5bba1c2085918f529778e88535aca7de674c9094af0b7` |
| BGE-M3 resolved execution identity | `330aaff04f917de64e7b21c7decf82556b8fb9b1163c00d8a1e672a93ce78f38` |
| Qwen3-8B two-layer prefix manifest | `76273516aa6924b12344d5e83daa485b66459b663c745cb3b9ef51cc17c7440d` |
| Qwen3-0.6B forced-choice manifest | `a940db06d5d9a3b298412376966b492f09ad7f088495fb75c05aa45db943d86e` |

The prefix revision is
`b968826d9c46dd6066d109eabc6255188de91218`; the choice revision is
`c1899de289a04d12100db370d81485cdf75e47ca`; and the BGE-M3 revision is
`5617a9f61b028005a4858fdac845db406aefb181`. These values are repeated here
for audit convenience, but the machine-readable policy remains authoritative.

## Controlled Mem0 comparison tooling

The isolated tooling now provides:

- one-byte-snapshot dataset loading with adversarial replacement checks;
- exact official within-record chronology and consecutive one/two-turn slices,
  without globally date-sorting unrelated histories;
- ten locked raw shards containing 24,928 pairs, five empty-pair skips, 24,923
  `Memory.add(infer=True)` operations, and 100 searches;
- a one-logical-extraction-call-per-add supervisor with zero SDK retries;
- exact allowlists for Mem0 configuration, owned Qdrant state, and the same
  pinned local BGE-M3 checkpoint used by the treatment;
- offline-first imports, disabled telemetry, socket-denied provider-free
  preflight, secret rejection, and sanitized receipts;
- no-clobber two-stage publication: both outputs are rendered and fsynced
  before atomic publication, with verified rollback and protected inputs;
- independent prompt reconstruction, per-shard validation, ten-shard merger,
  and paired question/gold/category/population comparison.

The provider-free production-binding seam is intentionally non-forgeable and
non-issuing. Production remains closed until one exact Mem0 environment lock
and policy exist together with non-injectable runtime construction, concrete
zero-retry extraction/Terra/Sol transports, proof of the actual local BGE
instance, post-run HTTP attempt closure, complete source-artifact attestation,
and positive production receipt schemas. Mem0 dependencies have not been
installed or invoked. The current `tools/mem0_eval` and its tests are also
uncommitted working-tree additions; they must be committed and rehashed before
any publication freeze.

## Test and artifact evidence

- Full repository suite: **1,417 passed**, with one pre-existing
  `pydantic-settings` `IncompleteFieldDefinitionWarning`, in 179.71 seconds.
- Focused Mem0 suite before the full run: **212 passed**.
- Current provider-free `tools/mem0_eval` implementation SHA-256 (not a
  production freeze):
  `5dc9c2675663807f706ff956eea8af73134e31d6d4fcd51d5eeb228eac5b8dcb`.
- Frozen implementation rehash:
  `452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83`.
- Local Qwen verification: prefix manifest exact across seven files; choice
  manifest exact.

Tests establish the implemented contracts and rejection paths. They do not
stand in for held-out model accuracy, provider accounting, or a real Mem0 run.

## Next frontier: Grounded Discourse Closure RAG

Ordinary top-k RAG is not sufficient for diffuse questions such as “given the
entire engineering conversation, how should we improve this system?” The
answer may depend on a constraint stated early, a rejected experiment in the
middle, and a later revision. The proposed general-purpose design is Grounded
Discourse Closure RAG:

```text
question -> query obligations -> hybrid seeds -> discourse closure
         -> minimal evidence bundles -> hard-budget packing -> cited answer
```

The design has five domain-neutral parts:

1. **Grounded discourse store.** Raw spans remain authoritative. Derived nodes
   represent claims, observations, constraints, decisions, actions, results,
   questions, entities, artifacts, and time. Typed edges include `supports`,
   `depends_on`, `answers`, `revises`, `supersedes`, `contradicts`,
   `result_of`, and `refers_to`. Every node and edge points back to exact raw
   spans and immutable source hashes.
2. **Query program.** Generalize set handling into explicit obligations: facts
   to find, comparison axes, causal steps, chronology, current-state
   resolution, contradictions, revisions, and unresolved constraints. Missing
   obligations remain visible rather than being filled by model confidence.
3. **Closure retrieval.** Start with lexical, dense, entity, temporal, and
   metadata seeds. Expand only along edges that discharge an obligation;
   include supporting premises, dependencies, superseding statements, and
   live contradictions. Stop when obligations close, the frontier is
   exhausted, or the hard token budget forces an explicit deficit.
4. **Atomic evidence bundles.** Pack a claim with the smallest source-backed
   support/revision/contradiction bundle needed to interpret it. The graph is a
   planner and index, never a replacement for raw evidence. Packing returns a
   closure receipt describing satisfied and missing obligations.
5. **Grounded answer stage.** The LLM proposes improvements only from packed
   evidence, cites the source spans supporting each material recommendation,
   distinguishes settled decisions from open questions, and reports
   uncertainty when closure is incomplete.

This can be added without weakening the zero-token-state rule: persist typed
discourse records, source spans, embeddings, and scalar edges, but never
request-derived token tensors, K/V state, residuals, or attention maps.

The evaluation should freeze long, multi-step conversations with oracle
minimal evidence sets and measure `MinimalSetHit@B`, `SoftClosure@B`,
obligation coverage, revision/contradiction/path recall, packet sufficiency,
citation correctness, judged answer utility, and prompt tokens. Required
ablations are dense top-k, lexical+dense hybrid, graph-only, no revision edges,
no contradiction closure, no iterative closure, and no atomic bundles. The
next-frontier design is successful only if it improves answer utility under
the same hard budget on an untouched split; a better-looking graph is not an
evaluation result.

## Authorized next gates

No scored call should be inferred from documentation work.

1. The treatment offset-0 canary requires explicit authorization for exactly
   20 central-dev calls: ten Terra responder calls and ten Sol judge calls,
   with zero retries. Use the frozen-policy, verified-cache procedure in
   [`Research Log 16`](16%20-%202026-08-18%20-%20V3%20retrieval%20freeze%20and%20validation%20campaign.md);
   do not replace it with a copied command that can drift.
2. If the canary is accepted, the other nine treatment shards require a
   separate 180-call authorization, or the full campaign may be authorized as
   200 calls in advance.
3. Mem0 needs a separate production freeze before authorization. Its locked
   workload would require 24,923 logical extraction calls plus 100 Terra and
   100 Sol scoring calls. Underlying HTTP attempts are not yet certifiable, so
   that campaign is not runnable as a controlled result. Follow the
   [`Mem0 runbook`](../../tools/mem0_eval/README.md).
4. Only after both campaign reports pass their independent mergers should the
   paired comparator produce the accuracy/cost result.
5. Grounded Discourse Closure RAG is the next implementation frontier after
   the locked treatment and Mem0 evidence gates, not a substitute for them.

Until those gates close, the accurate project claim is: **the frozen treatment
and its 100-question read-only evaluation inputs are ready; development
evidence is strong; held-out treatment accuracy, Mem0 production behavior, and
the fair paired result remain unmeasured.**
