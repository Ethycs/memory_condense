# 05 — Packet Compression and Operational Replacement

**Phase:** 05 | **Merged turns:** 173-232 | **Dates:** 2026-08-17 to 2026-08-18

## Purpose

Take the 1M-token retrieval winner from phase 04 — 98.3% evidence coverage at a
6,203-token returned packet — and answer two questions in sequence:

1. How small can the returned context packet get without losing the retrieval result?
2. Once it is small, can the system actually **replace** the transcript in an operational
   chat — real responder, real answers, judged accuracy — rather than merely scoring well
   on coverage metrics?

The phase delivers a layered compression pipeline (deterministic sentence pruning, TF-ISF
source activation, a minimal HSC hierarchy with a four-slot channel, and an
information-bottleneck greedy packer) that takes the packet from 6,203 to under 2,000
tokens at 100% development source coverage. It then runs the first operational
transcript-replacement test through the central-dev gateway and discovers the decisive
gap: **100% source coverage does not imply sufficient within-source fact coverage.**
Successive "keep pushing" cycles close much of that gap (20% → 70% judged operational
accuracy) through source-date binding and role-aware retrieval, and the phase closes with
a load-bearing negative result: coarse partition routing, in every configuration tried,
loses to the simpler unrestricted role-aware baseline and is rejected.

## Design

### Compression pipeline

The packet that leaves retrieval passes through four bounded, deterministic stages. No
stage requires a model call, and no stage ever holds the only copy of a fact — raw chunks
remain immutable and hydration always ends at original evidence.

**1. Deterministic sentence-level pruning.** Inside already-retrieved chunks, keep only
sentences matching the question, named entities, dates, or graph-linked concepts. This
uses the project's existing pySBD segmentation and lexical tokenizer (NLTK was considered
and found unnecessary — no corpus dependencies needed). This single stage took the packet
from 6,203 to 2,178 tokens (~477:1 transcript-to-context compression) with coverage
unchanged and best token-F1 improved from 0.145 to 0.182. A 2,125-token cap lost a
literal answer, locating the reliability knee.

**2. TF-ISF source activation.** Inverse *source* frequency is computed at the
conversation/session level, distinct from BM25's chunk-level IDF: a term appearing in 40
chunks of one conversation is common to chunk-IDF but maximally distinctive to ISF. The
two answer different questions — BM25 finds the chunk with the right wording; TF-ISF
decides which conversation or partition owns the concept. TF-ISF runs as a separate,
bounded, opt-in activation channel; it never modifies BM25 scores or widens the prompt.

**3. Minimal incremental HSC layer.** A provenance-preserving contraction hierarchy
(adapted from the `layer_context_seg` claude-hsc skill): raw chunks as immutable leaves,
parent nodes holding only retrieval cards, TF-ISF weights, child IDs, time ranges, and
Hebbian strength. Retrieval descends only hot branches, then hydrates originals. The HSC
channel into the final packet is fixed at **four slots**: an eight-slot reserve recovered
the one missing source but displaced another required source (coverage stuck at 5/6);
four slots preserved the baseline frontier and admitted the recovered sibling, producing
the first 100% mean evidence-source coverage result on the 1M development set — all
required sources on 10/10 questions at 2,179 mean tokens.

**4. Information-bottleneck greedy packer.** The formal framing is a conditional
information bottleneck: maximize I(C;Y|Q) subject to a token budget. The implementation
is a monotone information-per-token filter that estimates query relevance plus marginal
concept/source/temporal novelty per token, recomputes after every admitted channel, and
stops early when the next channel's gain rate falls below a threshold (best value 0.008;
0.00825 drops literal recall, establishing the pruning boundary). A **query-cardinality
guard** lowers pruning pressure for enumeration, ordering, comparison, and "all/each"
questions, because repeated related excerpts look redundant even when each event is a
required set member. Result: 2,179 → 1,986 tokens (~523:1) with all retrieval metrics
held. The packer runs after retrieval as an opt-in stage; raw memory, HSC structure, and
retrieval scores are untouched.

Channel pruning throughout is **query-conditioned and reversible**, never permanent
deletion. Protected channels: rare TF-ISF signals, temporal/contradiction edges,
low-degree bridge nodes, sources not yet represented in the packet, and provenance paths
for multi-premise questions.

### Operational replacement path

The replacement contract: the model receives system instructions, the current user turn,
a short working window, and roughly 1.4–2K tokens of retrieved memory. The full 1M-token
transcript stays in durable storage and is never sent.

The responder runs through the central-dev OpenAI-compatible LiteLLM gateway
(`https://central-dev.zt:4000/v1`, authenticated with a `LITELLM_KEY` virtual key, TLS
via `truststore` so Python uses the Windows certificate store). The `codex_sdk` route
serves the responder model (`codex_sdk/gpt-5.6-luna`). Runs are fail-closed and bounded:
ten gateway calls, zero judge calls unless explicitly budgeted, every run emitting a
SHA-256-verified machine-readable manifest.

A **fast retrieval-only gate** precedes any model spend: it checks whether the assembled
context contains the required evidence. Only arms that win this gate are rerun through
the full chat-prompt operational path.

The first operational run (commits `b118734`, `7659746`) scored 20% exact match / 0.210
mean F1 at 99.8% token savings — and the audit showed every failure was retrieval or
packing, not the model: "Serenity Yoga" never reached the prompt; the current Instagram
count `1300` was absent while stale `1250` survived. The fixes that followed were all on
the selection side:

- **Source-date binding** (promote a real chunk from each timestamp-routed source):
  20% → 30% judged accuracy for +122 tokens; a subsequent Pareto pass held 30% at 1,377
  tokens (~754:1) by adding timestamp prefiltering and speaker-role provenance.
- **Role-aware retrieval** (prioritize user-authored facts over assistant suggestions for
  autobiographical queries): 30% → **70%** judged operational accuracy, 50% exact match,
  0.634 mean F1, at 1,908 tokens (~545:1). This is the end-of-phase operational baseline,
  with 93% gold-source coverage on the retrieval gate.

End-of-phase state: 814 tests passing; remaining failures are multi-event set questions
(concerts, museums, sculpting chronology) — a content-acquisition problem inside
activated sources, which phase 06 takes up.

## Why this shape

**Deterministic before learned.** Every compression gain in this phase came from
deterministic allocation, not from a model: the bounded Qwen attention treatments had
been neutral, while the four-slot HSC channel produced the first 100% coverage result.
The working rule: deterministic retrieval guarantees coverage; learned components (a
future channel gate, transient attention) may only prune and order the uncertain
remainder.

**The bottleneck is flow allocation, not recall.** The HSC reserve experiment isolated
the mechanism cleanly: the missing evidence existed, the hierarchy could find it, and the
failure was too much information competing for a fixed-width channel. This reframes
retrieval as routing all necessary premises through several bounded cuts without one
relevant branch suppressing another — which is why softmax-style competition (weights
summing to one) is exactly the wrong primitive, and why admission is governed by marginal
information gain per token rather than independent relevance scores.

**Operational accuracy is the only real gate.** Coverage metrics said the system was
ready; the operational run said 20%. Every subsequent decision was driven by auditing
actual packed prompts against actual judged answers, and the phase's central lesson —
source coverage ≠ fact coverage — is only visible at that level.

**Cheap gates before model spend.** The retrieval-only fast gate lets partition and
routing arms be tested and rejected without any gateway calls; model runs are reserved
for arms that already improved evidence reachability.

## Why not X

### Why not two-partition routing

The rejected routing arm ([DR-0022](../decisions/0022-reject-two-partition-routing.md))
is a load-bearing negative result. The hypothesis was plausible: the 1M stress memory is
ten independently namespaced chat histories, each locked question's gold evidence lives
in exactly one of them, and the multi-event failures looked like cross-history
competition for a fixed evidence budget. Coarse routing should therefore let the budget
search the right history deeply.

Every configuration lost to the unrestricted role-aware baseline's **93%** gold-source
coverage:

- One-history routing: **58%** — traced to a real ordering bug (the coarse partition vote
  ran before role weighting, so assistant echoes selected the wrong history).
- One-history routing with the bug fixed: **78%** — still far below baseline.
- Two-partition (two-history safety) routing: **86.3%**, and roughly twice the local scan
  cost — below baseline on accuracy *and* worse on cost, failing both promotion criteria.

The route-decision audit explained why: the correct history is already rank 1 for 7/10
questions and within the top 4 for all 10, so hard routing can only destroy information.
A soft four-history cue beam (100% routing recall) nudged coverage to 94.7% for +123
tokens but completed neither remaining multi-event set, and widening the fine-cue
activation frontier from 65 to 250 sessions regressed coverage to 91.3% — more activated
sessions create competition rather than recovering missing events. Conclusion: the
missing sessions are absent before final packing, so the fix is a sharper fine cue inside
the routed locality (phase 06's set-completion work), not coarser routing.

### Other rejected or deferred alternatives

- **Small-model summarization as the compressor** — considered and subordinated: summary
  "retrieval cards" are acceptable only as an additional index with source-chunk
  provenance, never a replacement for raw evidence, and the no-model pipeline above
  captured the compression win first
  ([DR-0017](../decisions/0017-tf-isf-hsc-adoption.md)).
- **DHS-style global PageRank clustering** — the existing query-seeded heat diffusion
  plus Hebbian co-access graph is already the retrieval-oriented version; global PageRank
  favors frequently connected memories over query-relevant ones
  ([DR-0017](../decisions/0017-tf-isf-hsc-adoption.md)).
- **SOM as a retrieval index** — 2D projection loses distinctions, incremental training
  moves assigned regions, and map adjacency does not imply evidence relevance; retained
  only as a later partition/diversity ablation
  ([DR-0018](../decisions/0018-defer-som-ablation.md)).
- **Eight-slot HSC reserve / aggressive permanent pruning** — the wider channel displaced
  required baseline evidence; pruning is four-slot, query-conditioned, and reversible
  ([DR-0019](../decisions/0019-four-slot-hsc-reversible-pruning.md)).
- **Softmax attention or plain cross-entropy as the channel selector** — sum-to-one
  weights recreate the displacement problem; the adopted packer uses budgeted submodular
  information gain, with multi-label sigmoid gating noted as the correct future learned
  form ([DR-0020](../decisions/0020-ib-greedy-channel-packer.md)).
- **Local Qwen or direct Anthropic-key responder for the operational test** — the local
  full responder was impractical on this hardware and raw SSH to central-dev failed; the
  documented service catalog's LiteLLM gateway with the `codex_sdk` route is the bounded,
  reproducible path ([DR-0021](../decisions/0021-operational-replacement-via-gateway.md)).
- **Facet retrieval, pure BM25, wider candidate pools, unconditional source diversity** —
  all tested during the role-aware push and rejected on the same locked gate; role-aware
  selection alone accounted for the 30% → 70% jump.

## Open questions

- **Multi-event set completion.** The two remaining failure classes (concerts 4/5,
  museums 4/6) lose set members while same-concept event sources compete for slots. The
  proposed next mechanism — a deterministic event cue derived from list/order questions
  ("concerts musical events attended"), with reserved source-local candidates — is
  specified but unbuilt at phase end.
- **Learned channel gating.** The counterfactual-ablation training scheme for a tiny
  multi-label channel gate (labels = measured coverage loss when a channel is removed) is
  designed but not implemented; the four-slot allocation remains hard-coded.
- **Update-chain representation.** Stale-value suppression (1000 → 1250 → 1300) is
  handled by temporal/update semantics in packing, but facts are not yet stored as
  explicit revision chains with superseded flags.
- **Held-out validation.** Every number in this phase is from the locked 10-question
  development stress set; the 95% target and non-inferiority comparison against a native
  long-context arm on a held-out LongMemEval split remain unrun.
- **Adaptive second retrieval pass.** Letting a completeness check trigger retrieval of
  the one missing premise (the iterative-attention idea) is repeatedly identified as the
  right long-term shape but deferred.

## Source turns

- Phase opening and Pareto framing:
  [turn-757-user.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-757-user.md),
  [turn-758-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-758-assistant.md)
- Deterministic sentence pruning in lieu of NLTK/LLM:
  [turn-761-user.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-761-user.md),
  [turn-762-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-762-assistant.md),
  [turn-777-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-777-assistant.md)
- Four-algorithm comparison and adoption plan (DR-0017, DR-0018):
  [turn-784-user.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-784-user.md),
  [turn-786-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-786-assistant.md),
  [turn-788-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-788-assistant.md),
  [turn-791-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-791-assistant.md)
- HSC reserve experiment, information-flow diagnosis, four-slot result (DR-0019):
  [turn-800-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-800-assistant.md),
  [turn-802-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-802-assistant.md),
  [turn-804-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-804-assistant.md),
  [turn-808-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-808-assistant.md)
- Information-theoretic formulation and IB packer result (DR-0020):
  [turn-814-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-814-assistant.md),
  [turn-816-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-816-assistant.md),
  [turn-830-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-830-assistant.md),
  [turn-838-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-838-assistant.md)
- Replacement readiness and gateway wiring (DR-0021):
  [turn-839-user.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-839-user.md),
  [turn-840-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-840-assistant.md),
  [turn-846-user.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-846-user.md),
  [turn-851-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-851-assistant.md),
  [turn-859-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-859-assistant.md)
- Operational verdict and failure audit:
  [turn-873-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-873-assistant.md),
  [turn-878-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-878-assistant.md),
  [turn-880-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-880-assistant.md)
- Keep-pushing cycles (source-date binding, Pareto point, role-aware retrieval):
  [turn-899-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-899-assistant.md),
  [turn-919-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-919-assistant.md),
  [turn-935-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-935-assistant.md)
- Partition routing arms and rejection (DR-0022):
  [turn-937-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-937-assistant.md),
  [turn-941-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-941-assistant.md),
  [turn-942-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-942-assistant.md),
  [turn-943-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-943-assistant.md),
  [turn-946-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-946-assistant.md)
- Neural Storage lens, cue beam, and fine-cue frontier regression:
  [turn-947-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-947-assistant.md),
  [turn-948-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-948-assistant.md),
  [turn-950-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-950-assistant.md),
  [turn-953-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-953-assistant.md)
