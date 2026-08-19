# EM-LLM episodic discourse closure for diffuse retrieval

**Status**: ADOPTED DESIGN — EM-LLM mechanisms selected; externalized,
source-grounded implementation remains unbuilt
**Date**: 2026-08-18
**Applies to**: long conversations whose answer depends on information spread
across several episodes, especially explanation, diagnosis, recommendation,
planning, comparison, and status synthesis

## Decision

Use the event-formation and temporal-retrieval ideas from
[EM-LLM](https://em-llm.github.io/) as the front half of the diffuse retrieval
architecture:

```text
raw conversation
  -> surprise-based event boundaries
  -> graph-refined, source-grounded episodes
  -> similarity seeds + temporal contiguity
  -> discourse-obligation closure
  -> atomic evidence bundles
  -> hard-budget, cited answer
```

This is an adaptation, not a claim that memory-condense implements EM-LLM's
KV-cache architecture verbatim. The
[ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/c05144b635df16ac9bbf8246bbbd55ca-Paper-Conference.pdf)
stores past key/value states as episodic events and retrieves them separately
per Transformer layer. Memory-condense instead keeps raw transcript spans as
the authority and preserves the stronger invariant that no request-derived
token sequence, K/V cache, attention map, residual stream, or activation is
durable state.

The combined design is called **Episodic Discourse Closure RAG**. EM-LLM gives
it non-arbitrary event boundaries and a principled temporal recall path.
Grounded discourse closure supplies what ordinary episodic similarity does not:
cross-episode dependencies, revisions, contradictions, unresolved questions,
and an explicit proof that the final packet covers the query's obligations.

## Why ordinary RAG is insufficient

A diffuse query does not usually have one answer-bearing chunk. Consider a
long engineering conversation:

1. the user defines a success criterion;
2. an early experiment fails for a subtle reason;
3. a constraint rules out the obvious repair;
4. a later measurement changes the diagnosis;
5. the team revises one decision but leaves another open; and
6. the user finally asks, “How should we improve the system?”

Dense or lexical top-k can retrieve several individually relevant passages
without retrieving a *sufficient set*. Fixed chunks can also cut through the
middle of a coherent experiment or join the end of one episode to the start of
another. Increasing `k` raises distractor load and does not establish whether
the objective, constraint, result, revision, and unresolved issue are all
present.

The required unit of recall is therefore not “the nearest chunk.” It is:

```text
coherent episode -> relevant episode neighborhood -> obligation-complete proof
```

## What EM-LLM contributes

The official [paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/c05144b635df16ac9bbf8246bbbd55ca-Paper-Conference.pdf)
and [repository](https://github.com/em-llm/EM-LLM-model) define three mechanisms
that matter here.

### 1. Surprise-based event formation

For an autoregressive model, token surprise is the negative log-likelihood of
the observed token given its prefix. EM-LLM proposes a boundary when surprise
exceeds an adaptive moving threshold:

```text
s_t = -log P(x_t | x_<t)
T_t = mean(s_[t-tau:t]) + gamma * std(s_[t-tau:t])
boundary at t when s_t > T_t
```

The moving baseline is important. “Surprising” is relative to the current
stream, so a sustained technical section does not require the same absolute
threshold as casual dialogue. The paper reports that surprise segmentation
groups useful K/V states better than uniform blocks and aligns more closely
with human-perceived event boundaries than fixed segmentation.

### 2. Graph-theoretic boundary refinement

Initial surprise boundaries are refined using a similarity graph over
attention keys. EM-LLM seeks high within-event cohesion and low between-event
similarity, using modularity or conductance as the objective. Its bounded
one-pass adjustment considers candidate positions between consecutive initial
boundaries; the paper gives overall complexity `O(nm)` for sequence length `n`
and processing chunk size `m`.

This matters because surprise marks a change point, but it need not place the
boundary at the best point for recalling the material together. Refinement
makes an episode a retrieval unit rather than merely a span between two local
probability spikes.

### 3. Similarity plus temporal contiguity

EM-LLM first retrieves `k_s` events by nearest-neighbor similarity between the
current query and representative event tokens. It then enqueues temporal
neighbors of those events into a separate contiguity buffer of size `k_c`.
The resulting context combines initial tokens, a contiguity buffer, a
similarity buffer, and recent local context.

This corrects a familiar failure of semantic retrieval: a query may match the
result of an experiment but not the setup immediately before it, or match a
decision but not the qualification immediately after it. Temporal adjacency
is not proof of relevance, but it is a high-value route to the missing local
episode context.

## What we adopt and what we change

| Concern | EM-LLM | Episodic Discourse Closure adaptation |
| --- | --- | --- |
| Authoritative memory | Past per-layer K/V states | Immutable raw turns, chunks, and exact source spans |
| Boundary signal | Autoregressive token surprise | Frozen local surprise scorer when available; deterministic/source and embedding-change controls as ablations |
| Refinement graph | Per-head key similarity | Transient key similarity or source-grounded embedding similarity; persist only boundary receipts and scalar identities |
| Retrieval unit | K/V event block | Episode containing ordered evidence references |
| Representatives | Influential tokens per event | Bounded representative chunk/span IDs and index vectors |
| First retrieval stage | Similarity buffer | Existing lexical/dense/source routes plus episode similarity |
| Second retrieval stage | Temporal contiguity queue | Source-bounded preceding/following episode expansion with explicit token cost |
| Long-range relation | Primarily semantic and temporal | Typed discourse relations across episodes |
| Completion condition | Fixed retrieved-memory budget | Required query obligations closed, or an explicit incomplete reason |
| Answer evidence | Retrieved hidden state | Verified raw spans only |
| Durable transformer state | K/V memory is central | Exactly zero request-derived token-state bytes |

This adaptation deliberately preserves EM-LLM's insight while changing its
storage substrate. A boundary scorer may inspect local model outputs, keys, or
embeddings transiently. Durable state is limited to source IDs, span hashes,
episode boundaries, scalar scores, index vectors, relation records, and
receipts. Generated summaries and hidden states never become factual evidence.

## Write path: grounded episodes

Every episode belongs to one source timeline and a frozen content snapshot.
An episode record contains:

- `episode_id`, source ID, first/last turn ordinal, and first/last chunk ID;
- ordered evidence references with exact span and quote hashes;
- boundary method, scorer artifact identity, threshold-window identity, and
  pre/post refinement positions;
- bounded representative evidence IDs and their index-vector identities;
- previous/next episode IDs inside the same source; and
- an immutable receipt hash over all of the above.

Formation is incremental and idempotent:

1. append authoritative turns and chunks;
2. score only the new bounded window plus enough overlap to reconsider the
   last open boundary;
3. propose surprise boundaries;
4. refine within the bounded source-local window;
5. validate minimum/maximum episode size and exact source-span hashes;
6. atomically publish episodes and adjacency; and
7. advance the graph revision only after chunks and episodes are complete.

No event may cross unrelated source histories merely because timestamps are
close. If the semantic scorer fails, deterministic source/session/role
boundaries remain reconstructible and retrieval falls open to raw chunks.

## Read path: from episodes to closure

### Stage A — compile obligations

The query compiler emits a domain-neutral `QueryProgram`. For a recommendation
or improvement query, conservative required obligations are:

- objective or success criterion;
- current state or baseline;
- binding constraints;
- accepted and current decisions;
- observations and measured outcomes;
- failures or counterevidence;
- dependencies;
- unresolved issues or live alternatives; and
- revisions or contradictions affecting any item above.

Other intents produce different programs: lookup, enumeration, comparison,
explanation, diagnosis, planning, and status. An LLM may propose a program, but
the validated program is a routing plan, never answer evidence.

### Stage B — retrieve episodic seeds

Run existing lexical, dense, source, association, and metadata routes. Map
direct chunk hits to their episodes and add an episode similarity route over
bounded representatives. Direct raw hits remain fail-open even if episode
metadata is missing.

### Stage C — add temporal contiguity

For each strong episode seed, inspect bounded previous/next episodes from the
same source. Admission is based on new obligation gain, relationship to the
seed, exact incremental token cost, and a fixed contiguity quota. Temporal
neighbors are labeled as hypotheses until their spans actually discharge an
obligation.

This is the external analogue of EM-LLM's contiguity buffer. It prevents a
matched result from arriving without its nearby setup, but it cannot silently
consume the entire prompt.

### Stage D — close across discourse relations

Temporal neighbors solve local continuity. Diffuse evidence may still be far
apart, so a second graph connects source-grounded discourse units with typed,
evidenced relations:

```text
supports        contradicts       qualifies
revises         supersedes        retracts
depends_on      requires          causes
tests           produces          implements
addresses       resolves          rejects / accepts
refers_to       sequence / reply_to
```

Expansion occurs only when a relation can satisfy or disambiguate a live
obligation. A selected decision pulls its revision chain; a selected result
pulls its tested action/configuration; a selected conflict pulls both sides
and any resolution; a proposed improvement pulls the constraints and prior
outcomes that determine whether it is viable.

### Stage E — pack atomic evidence bundles

The final unit of packing is an evidence bundle, not a row. A bundle contains
the smallest verified set of spans needed to interpret one material claim:
for example, experiment setup + result, old decision + revision, or
contradiction side A + side B + resolution.

Packing counts the exact rendered cost under the frozen prompt proxy. Required
atoms are never prefix-truncated. If a required bundle cannot fit, the receipt
says `budget_impossible`; if the graph cannot find an obligation, it says
`not_found`; unresolved contradiction yields `conflicted`. Partial evidence is
never relabeled as complete because the model sounds confident.

## Closure receipt

Every answer packet carries a text-free, canonical receipt:

- query-program, policy, tokenizer, snapshot, and graph identities;
- direct chunk and episode seed IDs;
- similarity and contiguity admissions with paths and scores;
- visited discourse units and relations;
- selected evidence bundles and exact source-span hashes;
- status of every required and desired obligation;
- unresolved conflicts, dropped bundles, and reasons;
- exact context and prompt-token proxy counts;
- stopping reason; and
- `complete_claimed`, which is true only when every required obligation and
  its evidence path is valid under the frozen scope.

The receipt proves what the retrieval procedure inspected and packed. It does
not prove that an unknown relation or unannotated source does not exist unless
the corresponding scope is independently exhaustive.

## General-purpose, not engineering-specific

Engineering conversations motivate the first fixture, but no benchmark noun
belongs in the storage or closure contract. Episode kinds and relation types
are open strings with a small core vocabulary. Query obligations derive from
intent and grammatical roles rather than words such as “museum,” “concert,”
or “deployment.” Unknown content remains retrievable as raw evidence even when
no semantic unit can be validated.

The same mechanics cover:

- a medical history: symptom episode, intervention, outcome, revision;
- legal or policy analysis: requirement, exception, precedent, amendment;
- project management: goal, constraint, decision, task, blocker, resolution;
- research synthesis: hypothesis, method, result, counterresult, limitation;
- personal memory: plan, completed event, correction, current preference; and
- software diagnosis: observed failure, configuration, attempted fix, metric,
  dependency, and unresolved alternative.

## Evaluation and falsification

The diffuse benchmark must contain long noisy conversations with annotated
obligation graphs and one or more minimal sufficient raw-span sets. Primary
retrieval metrics are:

- `MinimalSetHit@B` under token budget `B`;
- weighted `SoftClosure@B`;
- required-obligation completion;
- episode-boundary and episode-recall quality;
- temporal-neighbor gain and distractor cost;
- evidence-path, revision-terminal, contradiction-pair, and resolution recall;
- false-complete rate;
- exact source-span validity and citation entailment;
- packet sufficiency to an answerer that sees only the packet; and
- answer utility under the unchanged hard prompt cap.

Required matched ablations are:

1. fixed chunks + dense retrieval;
2. fixed chunks + lexical/dense hybrid;
3. surprise episodes without refinement;
4. surprise + modularity refinement;
5. similarity episodes without contiguity;
6. similarity + contiguity;
7. episodes + discourse graph without obligations;
8. obligations without iterative closure;
9. full episodic discourse closure;
10. ordinary row packing versus atomic bundles; and
11. transient-key boundaries versus embedding-change and deterministic
    boundary controls.

All arms must share the same raw corpus, questions, answerer, judge, final
prompt cap, and seed budget. EM-LLM's published LongBench and InfiniteBench
results justify testing the mechanism; they do not establish performance on
our conversation workload.

## Smallest implementation tranche

1. Add source-grounded episode tables, boundary receipts, and snapshot/high
   watermark validation. Start with deterministic boundaries plus an injected
   surprise-score interface.
2. Add an episode representative index and a source-bounded contiguity route.
   Return direct chunks, episode seeds, neighbors, paths, and exact costs.
3. Build a synthetic noisy conversation containing a multi-step experiment,
   revised decision, unresolved contradiction, and distant constraint. Compare
   fixed chunks with surprise episodes under the same budget.
4. Add the discourse-unit/relation store and manual `QueryProgram` fixtures.
5. Add iterative obligation closure and atomic evidence-bundle packing.
6. Only then evaluate an automatic semantic linker and LLM query compiler.

This order tests the EM-LLM hypothesis before adding a large extractor. It also
keeps the locked validation-v3 treatment untouched: the new path is default
off, starts in implementation epoch v4, and requires its own frozen artifacts
and evaluation split.

## Claim boundary

The adopted claim is narrow:

> Surprise-refined episodes and temporal contiguity are a better front end for
> diffuse retrieval than arbitrary chunks alone, and they should feed a
> source-grounded obligation-closure system.

That claim is a design hypothesis until the matched ablations above pass.
EM-LLM itself notes important limitations, including non-parametric storage,
lack of hierarchical events, and lack of long-term consolidation. Our
discourse graph and existing consolidation work are proposed complements, not
published EM-LLM results.
