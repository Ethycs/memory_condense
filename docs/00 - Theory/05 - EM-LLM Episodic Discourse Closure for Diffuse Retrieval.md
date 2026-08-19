# EM-LLM-inspired episodic discourse closure for diffuse retrieval

**Status**: IMPLEMENTED PROVIDER-FREE PROTOTYPE — the general closure path and
a bounded Qwen attention-head semantic-change path are built; paper-exact
autoregressive token surprise remains an unevaluated ablation
**Date**: 2026-08-18
**Applies to**: long conversations whose answer depends on information spread
across several episodes, especially explanation, diagnosis, recommendation,
planning, comparison, and status synthesis

## Decision

Use the event-formation and temporal-retrieval ideas from
[EM-LLM](https://em-llm.github.io/) as one interchangeable front end for the
diffuse retrieval architecture. The closure system does **not** depend on the
EM-LLM package, its model wrapper, or persistent K/V memory:

```text
raw conversation
  -> exact raw chunks
  -> boundary strategy
       fixed interval | lexical/embedding change | Qwen prefix OV change
       | injected precomputed surprise
  -> optionally cohesion-refined, source-grounded episodes
  -> existing hybrid seeds + source-local temporal contiguity
  -> discourse-obligation closure
  -> atomic evidence bundles
  -> exact full-prompt-proxy cap + evidence packet for a downstream cited answer
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
us a hypothesis for better-than-arbitrary event boundaries and a principled
temporal recall path; it is not a required runtime component.
Grounded discourse closure supplies what ordinary episodic similarity does not:
cross-episode dependencies, revisions, contradictions, unresolved questions,
and an explicit proof that the final packet covers the query's obligations.

## Dependency boundary

There are two separate hypotheses:

1. **General closure hypothesis**: source-grounded raw evidence or episodes,
   typed discourse relations, explicit query obligations, bounded iterative
   closure, and atomic packing retrieve a sufficient evidence set more
   reliably than ordinary top-k chunks.
2. **EM-inspired segmentation hypothesis**: attention-head semantic change or
   autoregressive token surprise plus bounded cohesion refinement creates
   better episode boundaries than deterministic, lexical-change, or
   ordinary-embedding-change controls.

The first hypothesis can be implemented and tested without the second. The
provider-free implementation includes a fixed-interval detector, a
lexical/ordinary-embedding change scorer, bounded cohesion refinement, two
scorer seams, and a precomputed-numeric control. The stateless pairwise
`SurpriseScorer` remains the local control seam. The first-class
`QwenAttentionHeadSurpriseScorer` instead reuses
`QwenMemoryLinker.inspect_coverage`: one frozen neutral probe produces a
normalized OV transport signature for every source span; adjacent cosine
change proposes boundaries, and the refiner consumes a nonnegative-clipped
version of the same transient scalar cosine matrix. Its canonical, text-free
receipt binds the prefix model,
revision, verified checkpoint, layer, dtype, versioned algorithm, exact signal
implementation-source hash, token and workspace caps, input/score/matrix
hashes, full ordered evidence-span identities after builder binding, bounded
transport width, and observed work. A separate `owned_runtime_binding` bit
records that the exact owned `QwenMemoryLinker` and `Qwen3PrefixEncoder` types,
expected state-key shapes, and unshadowed inspection method were observed. It
is provenance, not independently authenticated execution or a heap-erasure
proof; injected test or research linkers cannot inherit it.

This implemented signal is semantic change in Qwen prefix transport space,
conditioned on a fixed probe. It is not the EM-LLM paper's autoregressive
`-log P(x_t | x_<t)`. Separately, the existing prefix coverage selector reports
`semantic_surprisal = -log(1 - p_new)`, where `p_new` is a
**query-conditioned retrieval posterior** over EXISTING/NEW/NULL. That signal
remains useful at retrieval time, but it is neither ingestion-time token NLL
nor the episode-boundary signal described here. A paper-exact token-surprise
experiment must still precompute and freeze a full-causal sequence signal,
bind its model and reduction identity, and pass it through `surprise_scores`.
No EM-LLM code is imported, and no K/V cache, token IDs, prompt buffer,
attention map, residual stream, activation, or transport vector is retained by
the episode result.

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
similarity, using modularity or conductance as the objective. The official
implementation exposes this distinction through settings including
`surprisal_threshold_gamma`, `similarity_metric` (`modularity` or
`conductance`), and `refine_from_layer`, and retains per-layer K/V events. Its
bounded one-pass adjustment considers candidate positions between consecutive
initial boundaries; the paper gives overall complexity `O(nm)` for sequence
length `n` and processing chunk size `m`.

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
| Boundary signal | Autoregressive token surprise | Implemented adjacent change in normalized Qwen prefix OV-transport space; caller-supplied score sequence, lexical/embedding change, and deterministic controls remain available |
| Refinement graph | Per-head key similarity with modularity or conductance | Implemented Qwen prefix OV-transport cosine clipped to `[0,1]` by bounded local cohesion, or source-grounded lexical/embedding similarity; paper-exact raw-key modularity/conductance remains an ablation |
| Retrieval unit | K/V event block | Episode containing ordered evidence references |
| Representatives | Influential tokens per event | Bounded representative chunk/span IDs plus feature/vector identity hashes; no episode ANN yet |
| First retrieval stage | Similarity buffer | Caller-supplied lexical/dense/source hits mapped to episodes; no independent episode-similarity index yet |
| Second retrieval stage | Temporal contiguity queue | Source-bounded preceding/following episode expansion with quotas and distance decay; exact token cost is applied during packing |
| Long-range relation | Primarily semantic and temporal | Typed discourse relations across episodes |
| Completion condition | Fixed retrieved-memory budget | Required query obligations closed, or an explicit incomplete reason |
| Answer evidence | Retrieved hidden state | Verified raw spans only |
| Durable transformer state | K/V memory is central | Exactly zero request-derived token-state bytes |

This adaptation deliberately preserves EM-LLM's insight while changing its
storage substrate. A boundary scorer may inspect local model outputs, keys, or
embeddings transiently. Durable state is limited to source IDs, span hashes,
episode boundaries, scalar scores, index vectors, relation records, and
receipts. Generated summaries and hidden states never become factual evidence.

## Efficient representation: project, do not replace

The hot retrieval path does not need to carry every rich Python object. A
future scalable implementation should maintain two representations:

```text
authoritative object
  -> stable key + compact vector/scalar projection
  -> retrieve IDs and graph neighbors
  -> hydrate exact source objects for finalists
  -> verify, close, and pack
```

The index plane can contain stable IDs, vectors, compact scalar features,
source/ordinal coordinates, and adjacency IDs. These records are disposable
and reconstructible. The evidence plane retains exact raw spans, role/time
provenance, discourse membership, content roots, and coverage receipts.

Replacing *all* objects with only vectors and keys would use less memory per
candidate, but it would also discard the information needed to distinguish a
revision from a contradiction, prove artifact scope, recover exact citations,
and verify what reached the answerer. The efficient boundary is therefore a
vector/key **projection** for routing, not vectors as factual authority. Only
the small admitted frontier is hydrated into rich immutable objects.

## Write path: grounded episodes

Every episode belongs to one source timeline. Publication returns a separate
content-bound whole-store snapshot; the episode record itself contains:

- `episode_id`, source ID, first/last turn ordinal, and first/last chunk ID;
- ordered evidence references with exact span and quote hashes;
- boundary method, scalar threshold/score values, and pre/post refinement
  positions;
- source-local sequence numbers from which previous/next adjacency is derived;
- an immutable receipt hash over those episode fields and evidence.

Representatives are separately published with their feature/vector identity
hashes. The implemented Qwen episode pass also returns a canonical signal
receipt that derives the actual prefix-checkpoint, layer, runtime, versioned
algorithm, signal implementation source, input, output, evidence coordinates,
bounded workspace, and observed owned-runtime identities. That receipt is
returned in memory through `EpisodePublication.build`; it is not yet persisted
or linked to the SQLite artifact. Threshold-window and publication-policy
identities can be declared in the caller-supplied annotation artifact, but the
workflow neither requires those declarations to match the returned signal
receipt nor combines them into one durable publication receipt.

The current provider-free API is a deterministic, idempotent batch publisher:

1. append authoritative turns and chunks;
2. have the caller select one source-local batch;
3. propose boundaries with the configured fixed, local-change, Qwen
   prefix-transport-change, or injected-surprise strategy;
4. refine within the bounded source-local window when configured, using the
   Qwen scalar cosine matrix clipped to `[0,1]` for the attention-head arm;
5. validate minimum/maximum episode size and exact source-span hashes;
6. atomically publish episodes and representatives; and
7. advance the graph revision only after chunks and episodes are complete.

The builder partitions every supplied span exactly once, while persistence
rejects overlapping or out-of-order episodes. Adjacency is derived at read time
from source-local sequence numbers. Incremental overlap and reconsideration of
the last open episode are future orchestrator responsibilities; the prototype
does not claim to manage that streaming policy automatically.

No event may cross unrelated source histories merely because timestamps are
close. If a semantic scorer is unavailable, the implemented model-free control
is fixed-interval segmentation inside one authoritative source. Retrieval
still falls open to exact raw chunks when episode annotations are missing.

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

The caller runs its chosen lexical, dense, source, association, or metadata
routes and passes their `RetrievalResult` rows into the combined workflow,
which maps direct chunk hits to episodes. Representatives are stored and
verified,
but the prototype does **not** yet have a separately trained episode ANN or an
independent episode-similarity route. Every admitted direct hit remains a
bounded, prioritized raw route whether or not it maps to an episode. If a configured
admission cap omits any direct hit or episode expansion, the exact omitted IDs
are recorded and the plan cannot claim complete coverage.

### Stage C — add temporal contiguity

For each strong episode seed, inspect bounded previous/next episodes from the
same source. The implemented seed expansion uses fixed previous/next quotas,
deterministic source order, and distance decay. Obligation gain and exact token
cost are applied later by closure and atomic packing; they are not yet used to
rank temporal expansion itself. Temporal neighbors remain routing hypotheses
until their spans actually discharge an obligation.

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

The bounded best-first walk favors potential obligation gain, but recognized
core semantic relations may also be traversed without first proving that the
next edge will satisfy a new obligation. A selected decision pulls its
revision chain; a selected result
pulls its tested action/configuration; a selected conflict pulls both sides
and any resolution; a proposed improvement pulls the constraints and prior
outcomes that determine whether it is viable.

### Stage E — pack atomic evidence bundles

The final unit of packing is an evidence bundle, not a row. A bundle contains
a verified grouped set of spans needed to interpret one material claim:
for example, experiment setup + result, old decision + revision, or
contradiction side A + side B + resolution.

Packing counts the exact rendered cost under the frozen prompt proxy. Required
atoms are never prefix-truncated. If a required bundle cannot fit, the receipt
says `budget_impossible`; if the graph cannot find an obligation, it says
`not_found`; unresolved contradiction yields `conflicted`. Partial evidence is
never relabeled as complete because the model sounds confident.

## Closure plan and packet receipt

The immutable `ClosurePlan` records the query program, policy, content-bound
snapshot, selected artifact, direct and episode routes, optional expansion
receipt, visited graph objects, evidence atoms and bundles, every obligation
result, exhaustive-scope witnesses, stopping reason, and completion state. The
packet `ClosureReceipt` then commits to the plan hash, selected bundle and atom
IDs, exact rendered-context hash, drop reasons, tokenizer identity, context and
full-prompt proxy budgets, stopping reason, and zero built-in retained request
state.

Together these objects prove what the implemented procedure inspected and
packed, relative to the accepted query program, caller-declared annotation
artifact, relation semantics, and coverage marks. They do not prove that an
annotation is semantically correct or that the compiler captured every
real-world requirement. A
positive completion additionally requires one explicit annotation artifact,
a current content-bound source/graph snapshot, an exhaustive artifact-wide
unit scan, exhaustive bounded-query witnesses, and a finalized coverage
receipt that includes chunks producing zero semantic rows. Episodes remain an
optional routing layer: their absence does not block raw/discourse-only
closure, while a truncated episode expansion does block completion. Otherwise
evidence is still returned, but completion remains false.

Finalizing that receipt also proves that exact chunks cover every
non-whitespace character of every authoritative turn. A turn committed before
a failed chunking step, or a partial/gapped chunk set, therefore cannot be
silently certified as a fully inspected corpus.

## General-purpose, not engineering-specific

Engineering conversations motivate the first fixture, but no benchmark noun
belongs in the storage or closure contract. Discourse-unit kinds and relation
types are open strings with a small core vocabulary. Query obligations derive
from intent and grammatical roles rather than words such as “museum,”
“concert,” or “deployment.” The built-in compiler and rule linker are
conservative English bootstrap implementations; the examples below describe
the schema's capacity or an injected-linker use, not validated default-stack
accuracy in every domain. Unknown content remains a bounded raw-evidence
candidate when no semantic unit can be validated, though final packing may
drop it when it cannot satisfy an obligation or fit the budget.

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
3. Qwen prefix-transport-change episodes without refinement;
4. Qwen prefix-transport change + bounded transport-space cohesion;
5. similarity episodes without contiguity;
6. similarity + contiguity;
7. episodes + discourse graph without obligations;
8. obligations without iterative closure;
9. full episodic discourse closure;
10. ordinary row packing versus atomic bundles; and
11. Qwen prefix-transport boundaries versus embedding-change and deterministic
    boundary controls;
12. frozen full-causal token NLL versus prefix-transport change; and
13. paper-exact raw-key modularity/conductance versus bounded transport-space
    cohesion.

All arms must share the same raw corpus, questions, answerer, judge, final
prompt cap, and seed budget. EM-LLM's published LongBench and InfiniteBench
results justify testing the mechanism; they do not establish performance on
our conversation workload.

## Implemented provider-free tranche

The first mechanical tranche now exists behind opt-in workflow methods. The
repository is constructed with `MemoryCondenser`, but no episode, linking,
closure, or packet workflow runs automatically:

1. SQLite schema v11 stores immutable annotation artifacts, source-local
   episodes, representatives, typed units, n-ary relations, exact evidence
   coordinates, coverage rows (including `no_output`), source/graph revision
   counters, and content-bound source/graph snapshot roots. Historic receipts
   from schemas that did not bind content are retired rather than relabeled.
2. Every evidence span is verified against both its chunk and authoritative
   turn on write and read. Multiple chunks inside one turn carry an explicit
   turn-relative position, so random chunk IDs cannot reorder the source.
3. Episode construction supports fixed intervals, lexical/ordinary-embedding
   change, adaptive injected surprise, and bounded cohesion refinement. A
   first-class Qwen adapter streams every span through bounded
   `inspect_coverage` workspaces, converts normalized OV transport signatures
   into adjacent semantic-change scores, and passes the same scalar cosine
   matrix to refinement, which clips negative cosine edges to zero. Its
   self-hashed receipt binds the model, checkpoint, layer, algorithm, source
   implementation, caps, full ordered evidence identities, scalar-output
   hashes, work counters, zero transformer-state bytes in the returned signal,
   and whether the exact owned runtime shape supplied the signal. The self-hash
   is tamper-evident consistency, not authentication. Only evidence references,
   scalar boundary data, and identities survive the returned build.
4. Caller-supplied retrieval results map to bounded episode seeds and
   source-local previous/next episodes. Missing or invalid annotations fall back to bounded
   original raw chunk IDs; every cap or omission is explicit and prevents a
   false completion claim.
5. The deterministic query compiler emits eight generic intents and flat
   obligation sets; the public `QueryProgram` contract can also represent
   caller-supplied dependency DAGs. The closure engine traverses bounded
   evidenced relations,
   handles revision terminals and contradiction/resolution groups, and reports
   every missing or conflicted obligation instead of fabricating completion.
6. Atomic evidence bundles are packed with exact union cost. The optional
   chat-prompt path counts base messages, evidence framing, BPE boundary
   effects, fixed chat framing, and an output reserve at every beam admission.
   It never prefix-truncates part of a required bundle.
7. Offline metrics measure minimal-sufficient-set hit, soft closure,
   obligation completion, evidence-path recall, revision and contradiction
   recall, false completion, budget compliance, and authoritative source-span
   validity from the final packet.

The provider-free end-to-end regression uses a 36-turn noisy engineering
conversation whose nine necessary facts are spread across the objective,
current state, constraint, failed experiment, dependency, observation,
decision revision, and unresolved issue. Ordinary hybrid retrieval at `k=1`
cannot contain that set. The default rule linker, fixed-interval episodes,
automatic recommendation program, corpus-scope closure, and atomic packer do
recover and pack all nine spans deterministically within the exact prompt
workspace cap, without importing EM-LLM or retaining request-token state. This
is mechanical evidence that the pipeline composes correctly, not a measured
accuracy result.

Important prototype limitations remain:

- there is no separately trained episode-representative ANN index; current
  caller-supplied chunk hits seed episodes;
- the conservative English rule linker recognizes only explicit cues, so a
  stronger semantic linker must remain injected and source-validated;
- linking is a caller-managed batch/rebuild workflow and cannot yet relate a
  new unit to an existing old unit without supplying the required prior
  evidence again;
- transformation identity is declared by the artifact supplied to the
  workflow rather than derived and attested from every runtime strategy;
- positive corpus completion requires every artifact unit to fit an
  exhaustive `max_units + 1` probe, so larger artifacts honestly return
  incomplete rather than using a scalable proof index;
- content roots stream the changed corpus/graph once per revision and use only
  an in-process revision cache, rather than an incremental Merkle structure;
- an episode-neighbor lookup failure currently degrades to no temporal
  neighbors rather than carrying a distinct failure code, although corpus
  completion still depends on the separate exhaustive discourse scope;
- paper-exact autoregressive token NLL segmentation and EM-LLM raw-key
  modularity/conductance have not been wired or evaluated against the Qwen
  prefix-transport, deterministic, and embedding-change controls;
- the Qwen adapter validates a bounded result and retains no request vectors in
  its returned signal; `owned_runtime_binding` is an observed type/state/code
  binding, not proof that external hooks, mutated model internals, or the wider
  process heap retained nothing;
- the implementation digest binds the local signal, receipt, prefix, linker,
  and tokenizer source plus live callable bytecode, but not the full
  Torch/Transformers/CUDA environment or authenticated loaded-code identity;
- calls through one scorer instance are serialized, but direct or concurrent
  use of the same linker's mutable tokenizer remains unsupported and must be
  externally serialized;
- the in-memory signal receipt is not durably linked to SQLite, and the caller's
  annotation artifact is not yet required to match its model, checkpoint,
  implementation, or policy identities;
- arbitrary injected scorers and linkers receive an unattested (`None`)
  publication retention value unless the exact owned binding is observed; and
- the Qwen signal is conditioned on one frozen neutral probe and uses reduced
  OV transport vectors, so it is an adaptation rather than parity with
  EM-LLM's persistent per-layer K/V events.

This path keeps the locked validation-v3 treatment untouched. Its workflows
are opt-in, it belongs to implementation epoch v4, and it requires its own frozen artifacts,
annotated diffuse benchmark, and matched evaluation before any accuracy claim.

## Claim boundary

The implemented claim is narrow:

> A general source-grounded obligation-closure system can consume raw chunks
> or episodes without persisting transformer token state. A bounded Qwen
> attention-head front end now forms and refines episodes from transient
> prefix OV-transport similarity with a canonical, bounded signal receipt and
> no transformer state in the returned signal. This is not an authenticated
> whole-process heap-erasure claim. Its retrieval value, and the value of
> paper-exact token NLL or raw-key graph refinement, must be established by
> matched ablation.

The mechanics and invariants are implemented; the accuracy claim remains a
hypothesis until the matched ablations above pass.
EM-LLM itself notes important limitations, including non-parametric storage,
lack of hierarchical events, and lack of long-term consolidation. Our
discourse graph and existing consolidation work are proposed complements, not
published EM-LLM results.
