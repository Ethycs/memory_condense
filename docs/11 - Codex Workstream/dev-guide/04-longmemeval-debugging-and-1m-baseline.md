# 04 — LongMemEval Debugging and the 1M-Token Baseline

**Phase:** 04 | **Merged turns:** 115-172 | **Dates:** 2026-08-17

## Purpose

Take the phase-03 apparatus to the actual locked LongMemEval gate
([DR-0012](../decisions/0012-target-longmemeval.md)) and debug the retrieval chain
against it, step by measured step: a partition-local search fix
([DR-0013](../decisions/0013-partition-local-search-fix.md)), two-hop attention-guided
retrieval ([DR-0014](../decisions/0014-two-hop-retrieval.md)), and its refinement into
recurrent CAV activation ([DR-0015](../decisions/0015-recurrent-cav-refinement.md)).

The phase ends on its most consequential move — a reframe, not a mechanism
([DR-0016](../decisions/0016-beat-1m-full-context-baseline.md)):

> The memory system was never meant to answer questions. Its contract is to provide the
> right context to a model that answers them. Success means beating 1M-token
> full-context retrieval, with the system as the context provider.

Under that framing a real 1M-token workload is required, so the phase closes by building
a deterministic 1M-token merged stress transcript and characterizing retrieval against
it — including the scale regression it exposes and the widened configuration that
recovers it.

## Design

### The contract: context packets, never answers

This is the phase's anchor. The system's entire responsibility is:

```text
question / current turn + stored memory  →  small, sufficient context packet
```

A separate, fixed LLM consumes the packet and answers. Answer accuracy is therefore an
**integration test of context quality**, not a capability the memory system implements.
The Qwen slice remains a linker and context selector; it must not become the answer
model.

The decisive benchmark is a controlled head-to-head with the same answer model:

| System | Input |
| --- | --- |
| 1M baseline | Complete transcript + question |
| Memory system | Retrieved/condensed memory + question |

The memory system wins when it shows equal or higher answer accuracy, far fewer input
tokens, lower latency and cost, stable accuracy as conversation length grows, and no
loss of corrections, chronology, or multi-turn dependencies. The full diagnostic grid
has four conditions — full transcript (upper bound), gold evidence only (sufficiency of
compact evidence), retrieved memory (end-to-end system), no memory (difficulty floor) —
so retrieval failures separate cleanly from reasoning failures. The primary retrieval
metric is **required-evidence recall under a token budget**; downstream answer parity
against full context is the final validation that the packet was genuinely sufficient.

### The retrieval chain and its debugged seams

The selected policy at phase end retrieves within a covered partition via:

1. BGE dense cosine search (HNSW) plus BM25 lexical search, min-max blended at
   `0.65 × dense + 0.35 × BM25`;
2. a global top-200 candidate pool filtered to activated partitions (up to 48 chunks
   survive), plus up to 24 forward-neighbor chunks;
3. two-hop Hebbian/causal graph diffusion;
4. token-aware packing by `score / √tokens` under a 6,750-token evidence budget inside
   an 8,000-token prompt ceiling.

Three mechanisms were layered onto this chain during the phase, each implemented,
bounded, and measured:

**Partition-local search** (`--source-local-search`). The diagnosed flaw: the pool in
step 2 is a *global* ranking filtered by partition, so an answer-bearing chunk outside
the global top 200 is unavailable even when its partition was correctly activated — and
no downstream reranker can recover a chunk it never receives. The fix scans all chunks
inside activated sources with bounded dense/BM25 candidate buffers, hydrates text only
after ranking, and calibrates scores globally so weak partitions cannot crowd out
evidence. On the locked n=40 development set it changed no row-level hits, so it ships
as an available ablation rather than the default; the frozen historical policy is
preserved.

**Two-hop attention-guided retrieval.** Attention is used as a bounded feedback step,
not a deeper reranker: high-recall first-round retrieval; Qwen QK/OV inspection of the
recalled evidence (workspace capped at eight candidates / 1,024 tokens, no retained
transformer state); heat diffusion through memory associations from the attended items;
a second retrieval round with a fixed candidate and prompt budget; union, dedup, and
final rerank. Strong first-round evidence is protected — attention adds candidates, it
never erases them. Only scalar heat, QK/OV-derived weights, and access edges persist.

**Recurrent CAV activation.** The refinement treats retrieval as a bounded activation
trajectory — a `window + 1` controller. The original-question activation selects six
items from recalled evidence; `question + selected evidence` is re-encoded into one
transient combined activation state (recomputed, never assembled by adding raw residuals
from incompatible token sequences); that combined state searches a fresh lower-ranked
candidate pool. Six activation-selected candidates plus six scalar fallbacks occupy a
fixed 12-slot reserve while 36 first-round source candidates remain protected.
Conceptually, each candidate is scored by how it *changes* the current conceptual state
— QK routing, OV transport, alignment of the activation delta with unresolved concepts,
novelty, minus redundancy — with the original question activation preserved so recursion
cannot drift toward an early distractor.

All three Qwen arms were neutral on their matched five-row tests (3/5 literal hits, 100%
source coverage, ±3 tokens of context, both directions). The mechanisms demonstrably
work — 60 second-hop candidates admitted, 30 selected by combined activation — but no
recall delta was measured, so all remain opt-in treatments, not the selected policy.

### Measured position on locked LongMemEval

On the locked 40-question development set (official corpus, hash-verified):

- Mean evidence-source coverage 99.5%; any-source 100%; all-source 97.5%.
- Literal answer present in context: 23/40 (57.5%) — 23 of the 24 answers that exist
  verbatim anywhere in the haystack.
- Mean context 6,638 tokens against ~104K-token transcripts: 15.7× smaller, a 93.6%
  token reduction.
- A gold-source sufficiency audit: every literal answer available in capped gold sources
  was retrieved from them, 20/20. Search is saturated for observable literal evidence;
  the remaining 16 questions require inference, aggregation, temporal arithmetic, or
  paraphrase — answer-stage work, not wider retrieval.

The audit is what forced the reframe: even a capped gold-source *oracle* reaches only
50% literal match, so the literal-containment metric has a hard ceiling that no
retrieval improvement can push toward 95%.

### The 1M-token stress store

LongMemEval-S transcripts average ~104K tokens, so the locked set demonstrates
compression against roughly 100K context, not a million. The phase therefore builds a
deterministic 1M stress sample by merging locked LongMemEval histories into **one**
memory — 1,039,203 tokens, 5,400 turns, 10 questions — behind a reusable
`--stress-context-tokens` benchmark mode
(`src/memory_condense/eval/context_stress.py`), rather than summing separate 100K
samples.

First run, unchanged policy: mean source coverage 81.3%, any-source 90%, all-source 70%,
at 5,924 returned tokens (0.57% of memory); cold build 609.3 s, warm cached run 35.6 s
for all ten queries. Three questions that scored 100% as independent 100K memories lost
evidence after merging: the fixed top-20 source gate and 200-candidate pool, tuned at
100K, face roughly ten times the competition at 1M and crowd out relevant partitions.

A matched widening sweep (source activation 20→40→80, pool 200→500→1000, budget fixed)
recovered it. The best flat widened arm at phase end:

- Mean source coverage **98.3%**, any-source **100%**, all-source **90%**.
- 6,203 returned tokens from a 1,039,203-token memory: **168× reduction, 99.4% fewer
  tokens**, under the unchanged 6,750/8,000 ceilings.
- Warm retrieval 3.56 s/query — a harness upper bound including dataset reload and
  report writing; break-even against full-context prefill at roughly 290K input
  tokens/second, with steady-state service latency still unmeasured.

A capped hierarchical partition router was also tried and **underperformed** the flat
widened arm, so it was not promoted.

### Operational deployment sketch

The endgame of the reframe is the system as a context provider for live agent sessions.
The repository already carries an MCP server (`src/memory_condense/mcp_server.py`:
`remember`, `recall`, `search`, `ingest`, `supersede`, `forget`, `memory_stats`). The
identified upgrade path: a `retrieve_context(query, token_budget)` tool backed by the
causal-graph context-packet retrieval rather than basic top-k search;
`observe_turn(...)` for live consolidation; shared source/session partition IDs; server
instructions requiring retrieval before context-dependent work; and a single long-lived
HTTP service once sessions run concurrently. This does not eliminate the model's context
window — it changes what enters it: the million-token history stays outside the model
and only the ~6K-token selected packet becomes tool-result context.

## Why this shape

**Debugging followed the measured bottleneck down the chain.** Coverage was saturated
(99.5%), so "is it search?" resolved to: source discovery solved, within-partition
chunk selection suspect. That named step 2's global-pool filter as the specific flaw,
which the partition-local fix addressed architecturally. When the fix proved neutral,
the gold-sufficiency audit showed literal search itself was saturated — moving the
investigation up a level, to evidence-chain assembly and finally to the metric itself.
Each mechanism was admitted only against a matched arm at an identical token budget.

**Feedback loops instead of deeper reranking.** A recursive tournament over one fixed
candidate pool reapplies the same uncalibrated score against the unchanged question;
early pruning mistakes compound and lost premises cannot be recovered. Long-chat
retrieval is genuinely recursive in a different sense — the first retrieved premise
should alter the search state for the second ("joined three weeks ago" plus "meetup last
week" must become "two weeks") — so recursion needs a *changing objective*: what
evidence is still missing. Two-hop feedback provides the loop; recurrent CAV provides
the state.

**Recurrent CAV is the originally intended role for Qwen.** Scoring candidates by how
each one changes a compact conceptual state turns the attention slice into a live
memory-navigation controller rather than an expensive substitute for the existing
ranker, while keeping every invariant from earlier phases: bounded transient windows,
no persisted activations or KV state, IDs and scalars only.

**The reframe corrects a responsibility error.** Optimizing literal answer containment
assigned the system a job it was never designed for. Once the contract is "context
packets, never answers," the 57.5% literal figure stops being a failure number and the
real question becomes operational equivalence: does a fixed model with the 6K packet
match or beat the same model with the full transcript, at a fraction of the tokens,
latency, and cost. That question is answerable and worth winning.

**One merged 1M store, not forty 100K stores.** 40 × 104K ≈ 4.17M aggregate tokens is
not a million-token memory; retrieval difficulty scales with competition inside a single
store. Merging exposed a real scaling failure (fixed activation pools crowding at 10×)
that per-sample runs could never show — and the cached store makes iteration cheap
enough (35.6 s warm) to sweep configurations against it.

## Why not X

**Why not keep tuning the global top-200 pool?** Because the failure was structural:
a global ranking filtered by partition can produce 100% source coverage while the
answer-bearing chunk never enters the candidate pool, and no second-stage reranker can
recover a chunk it never sees. The correction is partition-local candidate generation —
[DR-0013](../decisions/0013-partition-local-search-fix.md). (Honest ledger: on the 100K
set the fix changed nothing measurable, confirming the *diagnosis* mattered more than
this instance of it; at 1M scale the bottleneck reappeared as pool crowding and was
addressed by widening.)

**Why not replace BGE with Qwen embeddings?** Raw Qwen residuals are not trained for
cosine retrieval and would likely trail BGE-M3; a retrieval-trained Qwen checkpoint is a
clean swap but demands a full re-embed and reindex for an improvement that still lives
in the representation, not the mechanism. The chosen shape — cheap BGE/BM25 discovery,
bounded Qwen conditional scoring inside already-covered partitions — puts the expensive
model only where independent embeddings are weakest.

**Why not a deeper recursive attention tournament?** It recursively applies the same
unsupervised QK/OV score to a fixed pool; if that score does not encode "needed to
answer this question," each round compounds the error and early-eliminated premises are
gone. Attention over recalled evidence followed by a fresh retrieval round —
[DR-0014](../decisions/0014-two-hop-retrieval.md) — adds candidates instead of
re-sorting them.

**Why not the standard multi-hop RAG stack?** Query decomposition, a trained
cross-encoder reranker, MMR diversity, and a sufficiency verifier are acknowledged as
the standard solution and the conventional system to beat. Raw attention magnitude is
not calibrated evidence relevance. The project deliberately keeps the standard stack as
the named control arm and confines the Qwen heads to the "choose the next evidence
item" step, where live attention can be measured against that baseline rather than
against plain cosine ordering —
[DR-0014](../decisions/0014-two-hop-retrieval.md) /
[DR-0015](../decisions/0015-recurrent-cav-refinement.md).

**Why not answer accuracy as the system's own requirement?** The system provides
context; a separate model answers. Holding the retrieval layer to a literal-answer
metric with a 50% oracle ceiling misattributes answer-stage reasoning (temporal
arithmetic, correction resolution, paraphrase) to the memory layer —
[DR-0016](../decisions/0016-beat-1m-full-context-baseline.md). Also the reason the
target moved off the 100K-transcript benchmark alone: the claim worth making is against
1M full context, which the locked set ([DR-0012](../decisions/0012-target-longmemeval.md))
cannot express without the merged store.

**Why not hierarchical partition routing (yet)?** The capped hierarchy underperformed
the flat widened arm on the 1M store, which already met the 95% mean-coverage gate at
98.3%. Hierarchy remains the anticipated architectural fix if widening's per-query cost
proves unacceptable as memory grows further, but it is not promoted on a losing
measurement.

## Open questions

- **The head-to-head itself has not run.** No answer model has yet consumed either the
  packets or the full transcripts; the four-condition grid (full / gold / retrieved /
  none) and the actual 1M-baseline comparison are designed but unexecuted.
- **Semantic sufficiency metric.** Literal containment is saturated and ceilinged;
  required-evidence recall under a token budget needs a real implementation before the
  neutral Qwen arms (two-hop, recurrent CAV) can be judged fairly — they may add
  multi-premise value that literal matching cannot see.
- **The last 1M failure.** All-source coverage sits at 90%; the open choice is adaptive
  widening (expand only on ambiguity) versus paying a permanent 1,000-candidate cost on
  every query.
- **CAV layer economics.** The CAV bank lives at layer 5 and needs the seven-layer
  prefix; the fast path loads layers 0-1. A cheaper newly trained layer-1 CAV for the
  recurrent controller is proposed but unbuilt.
- **Real latency.** 3.56 s/query is a harness upper bound; steady-state service latency
  and a time-to-first-token comparison against cached 1M full context are unmeasured.
- **The MCP context-packet service.** `retrieve_context` / `observe_turn` over the
  causal graph, plus a single shared HTTP service for concurrent sessions, are specified
  but not implemented.

## Source turns

Raw transcript for this phase:
[phase-04 overview](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/00-overview.md)

Key moments:

- Pivot to the locked benchmark (DR-0012):
  [turn-591-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-591-user.md),
  [turn-592-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-592-assistant.md),
  [turn-642-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-642-assistant.md)
- Step-4 diagnosis of the global-pool filter:
  [turn-643-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-643-user.md),
  [turn-644-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-644-assistant.md),
  [turn-652-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-652-assistant.md),
  [turn-657-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-657-assistant.md)
- Partition-local search fix designed and implemented (DR-0013):
  [turn-659-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-659-assistant.md),
  [turn-660-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-660-user.md),
  [turn-673-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-673-assistant.md)
- Gold-source sufficiency audit and Qwen reranker result:
  [turn-692-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-692-assistant.md)
- Recursion diagnosis and the standard multi-hop baseline:
  [turn-693-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-693-user.md),
  [turn-694-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-694-assistant.md),
  [turn-696-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-696-assistant.md)
- Two-hop attention-guided retrieval adopted (DR-0014):
  [turn-697-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-697-user.md),
  [turn-698-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-698-assistant.md),
  [turn-705-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-705-assistant.md)
- Recurrent CAV refinement (DR-0015):
  [turn-706-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-706-user.md),
  [turn-707-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-707-assistant.md),
  [turn-708-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-708-user.md),
  [turn-712-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-712-assistant.md)
- The reframe: context provider, beat 1M full context (DR-0016):
  [turn-717-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-717-user.md),
  [turn-718-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-718-assistant.md),
  [turn-719-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-719-user.md),
  [turn-720-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-720-assistant.md),
  [turn-721-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-721-user.md),
  [turn-722-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-722-assistant.md)
- Token accounting of LongMemEval:
  [turn-726-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-726-assistant.md)
- 1M merged stress store built and characterized:
  [turn-729-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-729-user.md),
  [turn-740-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-740-assistant.md),
  [turn-742-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-742-assistant.md)
- Widening sweep, best flat arm, and latency accounting:
  [turn-748-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-748-assistant.md),
  [turn-750-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-750-assistant.md),
  [turn-752-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-752-assistant.md)
- Equipping live sessions via MCP:
  [turn-753-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-753-user.md),
  [turn-756-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-756-assistant.md)
