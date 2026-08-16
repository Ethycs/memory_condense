# Native hypergraph memory plane

**Status**: PROPOSED — architecture note; schema v7 persists pairwise QK/OV
edges, CAV signatures, and a separate bounded live Hebbian co-access projection
**Date**: 2026-08-15
**Applies to**: the optional live Qwen linker and bounded associative retrieval path
**Depends on**: [`00 - System Overview.md`](00%20-%20System%20Overview.md), [`../00 - Theory/01 - Extracted Attention Heads as Recursive Associative Memory.md`](../00%20-%20Theory/01%20-%20Extracted%20Attention%20Heads%20as%20Recursive%20Associative%20Memory.md), and [`../05 - Standards/00 - MC-STD-DATA-v0.md`](../05%20-%20Standards/00%20-%20MC-STD-DATA-v0.md)

## Decision

A native hypergraph is useful for the memory plane because one bounded Qwen
inspection is a higher-order event. It can simultaneously bind several source
episodes, a directed source and destination, a selected head, active CAVs,
temporal context, QK routing evidence, and OV transport. Flattening that event
immediately into unrelated pairwise edges loses which facts were observed
together and duplicates the event's lifecycle state.

The hypergraph SHOULD become the canonical representation of compiled
association events only after a dual-write experiment passes. The existing
`chunk_head_edges` table SHOULD remain as a materialized pairwise projection
for fast bounded traversal. It is already compact and has produced a useful
two-hop development result; replacing it on faith would add risk without a
measured benefit.

This is an event hypergraph, not transformer context made durable. Token IDs,
Q/K/V tensors, attention matrices, residual streams, and head outputs remain
transient and MUST be discarded at the end of each bounded inspection pass.

## Why a pairwise graph is not the whole memory

Suppose a head observes the following bounded workspace:

- episode A establishes a constraint;
- episode B records a later decision;
- concept CAVs `binding_constraint` and `context_dependency` are active;
- head `(layer=1, head=17)` routes from A toward B;
- its OV output carries concept-aligned information into B.

The existing pairwise record can say `A -> B` and retain per-head QK/OV
weights. It cannot say, without duplication, that both concepts, the head,
the workspace, and the two episode roles belong to the same observation.
That distinction matters when the system later:

1. explains why a link exists;
2. updates one live observation after use or contradiction;
3. prunes a weak observation without erasing stronger independent evidence;
4. distinguishes a genuine multi-episode bridge from a high-frequency hub;
5. projects the same event differently for QK, CAV, temporal, or provenance
   traversal.

The pairwise graph remains the right serving index. The hypergraph is the
right loss-minimizing record of how that serving index was compiled.

## System boundary

```text
durable source text and chunk IDs
        |
        v
cheap hybrid candidate generation
        |
        v
hard-capped transient Qwen workspace
  QK = where information was addressed
  OV = what information was transported
  CAV = which concept coordinates were active
        |
        | compress, then discard every token-shaped tensor
        v
canonical association hyperedge
        |
        +----> pairwise QK/OV projection ----> bounded two-hop read path
        |
        +----> lifecycle/pruning counters
        |
        +----> provenance and explanation
```

The large model is a staged linker/compiler. It is not the durable store and
does not need to be loaded for ordinary reads. An optional read-time
inspection may examine a small fetched workspace, but it obeys the same rule:
only IDs, fixed-width coordinates, scalar evidence, and links cross the pass
boundary.

## Logical model

### Nodes

The minimum useful node types are deliberately small:

| Node type | Identity | Durable meaning |
| --- | --- | --- |
| `Episode` | `chunk_id` | Source-grounded text span and provenance |
| `Concept` | `(artifact_id, concept_index)` | Versioned CAV coordinate, not a fact |
| `Head` | `(artifact_id, layer, head_index)` | Versioned circuit address, not an activation |

Artifact identity binds the model checkpoint, tokenizer, prefix depth, head
layer, CAV layer, concept ordering, and head count. A node from one artifact
MUST NOT silently compose with a node from another artifact.

### Hyperedges

The canonical `AssociationObservation` hyperedge records one compressed
inspection event. Direction is expressed by member roles; a hyperedge is not
assumed to be undirected.

| Field | Meaning |
| --- | --- |
| `observation_id` | Stable, idempotent identity |
| `artifact_id` | Interpretation/version boundary |
| `created_turn` / `last_used_turn` | Turn-clock lifecycle coordinates |
| `qk_evidence` | Aggregated directed addressing strength |
| `ov_transport` | Aggregated amount of information moved |
| `cav_alignment` | Concept-bearing component of the transported update |
| `evidence_count` | Independent observations merged into this event |
| `traversal_count` | Successful or attempted read use, recorded separately |
| `utility` | Cached pruning/ranking scalar; always reconstructible |
| `metadata` | Small versioned scalar/configuration payload only |

Each member has `(observation_id, member_type, member_id, role, ordinal,
weight)`. Initial roles are:

| Role | Purpose |
| --- | --- |
| `anchor` | Episode that admitted the workspace from hybrid retrieval |
| `source` | Episode position addressed from |
| `target` | Episode proposed for later retrieval |
| `context` | Episode that conditioned the observation but is not a target |
| `concept` | Active/gating CAV with signed or thresholded weight |
| `head` | Selected head that supplied the evidence |

An observation may have multiple sources, targets, concepts, and heads. In
the first implementation it SHOULD normally contain one source, one target,
one head, and zero or more concepts; accepting the general shape now avoids a
schema rewrite when repeated observations are consolidated later.

### Physical SQLite shape

The proposed future additive schema is:

```sql
association_hyperedges(
    observation_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    created_turn INTEGER NOT NULL,
    last_used_turn INTEGER NOT NULL,
    qk_evidence REAL NOT NULL,
    ov_transport REAL NOT NULL,
    cav_alignment REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    traversal_count INTEGER NOT NULL,
    utility REAL NOT NULL,
    metadata_json TEXT NOT NULL,
    FOREIGN KEY (artifact_id) REFERENCES association_artifacts(artifact_id)
        ON DELETE CASCADE
)

association_hyperedge_members(
    observation_id TEXT NOT NULL,
    member_type TEXT NOT NULL,
    member_id TEXT NOT NULL,
    role TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY (observation_id, member_type, member_id, role),
    FOREIGN KEY (observation_id)
        REFERENCES association_hyperedges(observation_id) ON DELETE CASCADE
)
```

Indexes SHOULD cover `(member_type, member_id, role, observation_id)` for
incident lookup and `(artifact_id, utility)` for pruning. Episode members need
an enforced source-integrity path to `chunks`; concept and head members are
validated against `association_artifacts` in the store API because their IDs
are compact coordinates rather than separate mutable rows.

The stable `observation_id` SHOULD hash the artifact identity, typed/ordered
members, direction, and compilation protocol. Replaying the same compilation
therefore updates evidence instead of multiplying duplicate links.

## Pairwise serving projection

`chunk_head_edges` remains a derived read model. For every hyperedge, project
each `source x target` pair and aggregate compatible head evidence:

```text
(association observation)
    -> source episode x target episode x selected head
    -> qk_score, ov_transport, CAV compatibility, evidence count
    -> chunk_head_edges
```

Projection aggregation MUST be explicit and versioned. An exponential moving
average preserves repeated evidence; a maximum is useful for rare strong
bridges; a sum favors frequent hubs. The initial control SHOULD retain the
current weighted merge so the hypergraph arm can be compared with an exactly
equivalent pairwise arm.

The projection is disposable. It may be rebuilt from hyperedges and should be
treated like a serving cache, even while it remains in SQLite. This creates a
clean path to put incident sets and counters in Redis later without making
Redis the source of truth. Chroma can continue to provide document/vector
candidates, but it is not a suitable canonical store for directed,
role-bearing hyperedges and lifecycle transactions.

## Live write path

1. Persist the turn and source-grounded chunks first.
2. Generate a small candidate workspace with the existing hybrid retriever.
3. Stage one Qwen prefix model and one bounded batch at a time.
4. Inspect selected layers/heads once; compute CAV coordinates, QK evidence,
   actual OV transport, and member roles.
5. Compress the observation to IDs, fixed-width coordinates, and scalars.
6. In one transaction, upsert the hyperedge, its members, and affected
   pairwise projections.
7. Release residuals, Q/K/V, attention maps, head outputs, input IDs, and the
   CUDA workspace. Unload the staged model at the compilation boundary.

Compilation SHOULD run asynchronously behind a bounded queue. Reads continue
against the last committed projection under SQLite WAL. Queue depth, batch
size, and compilation age need metrics and hard limits; backpressure should
delay linking, never grow an unbounded activation buffer or block ordinary
retrieval.

This is what makes the memory live: new evidence changes durable links and
their utility over time. It does not mean the transformer remains resident or
that its context is retained.

## Bounded read path

The default read does not invoke Qwen:

1. Retrieve hybrid anchors.
2. Fetch incident hyperedges or their pairwise projection for those IDs.
3. Score candidates with QK evidence, OV transport, CAV compatibility,
   lifecycle utility, and cycle penalties.
4. Carry only `(chunk_id, score, route, observation_id)` between hops.
5. Trim the beam after every hop.
6. Hydrate source text only for the final fixed number of prompt slots.

Association admission is conservative at the prompt boundary. Near-max
lexical anchors are protected from displacement, and the complete associative
result is rejected if final hydration would increase prompt tokens. Rejected
routes do not receive lifecycle touches. These guards constrain both the
current pairwise projection and any future native-hypergraph reader.

The current development operating point is two hops, one association slot,
and a global candidate cap of eight. On the 12-question source-held-out split,
hybrid `k=5` recalled `83.3%` at `989.8` mean tokens. The pairwise QK
projection recalled `91.7%` with the same five final slots; degree one used
`965.2` mean tokens and degree three used `932.7`. This split has now been
consumed by tuning, so these are architecture-shaping development results,
not confirmation. A third hop hurt in the preceding sweep and is not the
default.

The hypergraph read arm MUST use the same two-hop, eight-candidate, one-slot
budget when compared with the pairwise projection. It may not claim a gain by
expanding more memories or hydrating intermediate nodes.

The locked fresh confirmation is the comparison floor: safe pairwise QK kept
83.3% hybrid recall while reducing mean prompt tokens from 973.9 to 961.1.
Degree-two physical pruning kept recall at 967.9 tokens and reduced 1,204 edges
to 812. A native hypergraph must beat that safe pairwise result on recall,
tokens, storage, latency, or pruning safety under the same candidate budget.

## Scoring and pruning

Hyperedge utility separates compilation evidence from later popularity:

```text
utility = decay(turn_distance) * (
    w_qk * qk_evidence
  + w_ov * ov_transport
  + w_cav * cav_alignment
  + w_evidence * independent_evidence
  + w_bridge * bounded_bridge_value
  + w_use * successful_traversal_rate
)
```

The exact weights are measured parameters, not schema. `traversal_count`
MUST NOT overwrite QK/OV evidence: a frequently served mistake is still a
mistake. Likewise, raw degree MUST NOT be rewarded because it creates
self-reinforcing hubs.

Pruning proceeds from cheapest and most reconstructible state toward source
truth:

1. Drop cold pairwise projections; rebuild them if their hyperedge survives.
2. Remove weak head or concept memberships when ablation shows no marginal
   routing value.
3. Merge duplicate observations only when artifact, roles, direction, and
   provenance agree.
4. Remove a hyperedge only after decay, contribution, bridge coverage, and
   contradiction checks.
5. Never delete or rewrite the underlying transcript to satisfy a graph
   budget. Pins, supersession, and provenance remain authoritative.

Bridge protection is where a native hypergraph can outperform pairwise
pruning. An infrequently used observation may be the only event connecting a
constraint, a decision, and a concept. Pairwise degree statistics can make
that event look like several individually weak edges; event-level coverage
keeps or removes it coherently.

## Interaction contract

The store-facing API should expose concepts rather than a database-specific
query language:

| Operation | Contract |
| --- | --- |
| `record_observation` | Idempotently write one compressed event and update projections |
| `incident` | Return bounded hyperedge IDs and scalar summaries for member IDs |
| `expand` | Run a hop- and candidate-capped traversal without hydrating text |
| `explain` | Return roles, artifact identity, evidence, and source pointers for a result |
| `touch` | Update usage counters after selection; no model invocation |
| `prune` | Apply turn-based lifecycle policy while respecting pins and bridge reserves |
| `reproject` | Rebuild pairwise serving edges from canonical observations |

SQLite is the deterministic first backend. A later Redis backend may implement
hot `incident`, `touch`, and bounded frontier operations. Backend choice must
not alter traversal caps, artifact validation, or the no-token-state rule.

## Failure containment

- **Hub collapse**: cap candidates globally and per source/concept; penalize
  repeated generic members.
- **Recursive drift**: require CAV or provenance compatibility, track visited
  IDs, and stop at two hops unless a fresh benchmark justifies more.
- **Attention-as-truth**: retain QK/OV as routing evidence, never factual
  authority; terminal text and provenance answer the query.
- **Artifact mismatch**: reject cross-checkpoint/layer/tokenizer composition
  unless an explicit migration maps both spaces.
- **Memory inversion**: record payload bytes, projection bytes, queue depth,
  compilation time, read latency, and prompt tokens independently.
- **Partial writes**: hyperedge, members, and projection updates commit
  atomically; a failed compiler batch leaves source chunks retrievable through
  the baseline.
- **Cache staleness**: live multi-process readers default to uncached incident
  queries or use explicit invalidation; immutable benchmark readers may use a
  bounded local cache.

## Migration and acceptance gates

1. **Schema and dual write**: add the two hypergraph tables behind an opt-in
   feature flag. Preserve schema-v5 pairwise behavior exactly.
2. **Projection equivalence**: rebuild `chunk_head_edges` from hyperedges and
   prove byte-for-byte or score/order equivalence for the control aggregator.
3. **Read ablation**: compare pairwise traversal, hyperedge-aware traversal,
   shuffled memberships, and no-association hybrid retrieval at identical
   final-token and frontier budgets.
4. **Pruning ablation**: compare pairwise utility pruning with event-aware
   pruning at matched storage budgets. Report rare-bridge retention separately.
5. **Backend profiling**: add Redis only if SQLite incident lookup or
   cross-process counters miss a measured real-time target. Do not add Chroma
   as a second graph authority.

The native hypergraph earns adoption only if it provides at least one measured
benefit over its pairwise projection: better recall at the same prompt budget,
equal recall at lower storage/latency, or materially safer pruning and
explanation at the same recall. Until then, it remains a loss-minimizing event
model and the pairwise graph remains the production read path.

## Non-goals

- storing transformer K/V cache, attention maps, residuals, or token history;
- loading Qwen for every retrieval;
- replacing hybrid dense/lexical entry retrieval;
- treating concepts or attention weights as source-grounded facts;
- unbounded recursive walks or graph-to-prompt expansion;
- introducing Redis or Chroma before a measured backend bottleneck exists;
- making the hypergraph the sole authority for transcript deletion,
  supersession, pinning, or provenance.
