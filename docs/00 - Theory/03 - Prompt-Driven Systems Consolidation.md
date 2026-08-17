# Prompt-driven systems consolidation

**Status**: IMPLEMENTED — schema v9, bounded cross-partition read/write path

**Applies to**: ordinary assembled contexts, typed semantic memories, source-grounded chunks, and optional transient Qwen CAV/QK/OV inspection

## Claim

Memory consolidation is a use-dependent association process before it is a
summarization process. A later prompt reactivates a bounded assembly of durable
memories. Repeated independent reactivation strengthens the connections inside
that assembly; inactivity weakens them. A future query may then activate one
member and recover another without materializing the complete history.

The biological comparison is computational rather than literal:

| Human-memory analogy | Implemented mechanism |
| --- | --- |
| fast episodic binding | bounded slices of one completed prompt/response episode |
| replay/reactivation | a later prompt independently retrieves the same nodes |
| synaptic strengthening | decayed node and pair co-activation masses |
| systems consolidation | a recurring assembly becomes traversable from a member |
| forgetting | turn-space decay plus minimum-score pruning |
| limited connectivity | hard event-size and node-degree bounds |

No prompt, token sequence, attention matrix, residual stream, or K/V cache is
durable state.

## Connected partitions

Schema v9 uses two node kinds:

- `memory:<mem_id>` addresses a compact, typed `MemoryItem`;
- `chunk:<chunk_id>` addresses source-grounded evidence.

One undirected edge may therefore connect semantic-to-semantic,
evidence-to-evidence, or semantic-to-evidence memory. The edge is not a fact.
It means either that its endpoints repeatedly survived context packing together
or that at least one endpoint was newly produced in their completed causal
interaction. Schema v9 keeps those evidence types distinct with
`coactivation_count` and `causal_count`.
The authoritative content and provenance remain in `memory_items`,
`memory_provenance`, `chunks`, and `turns`.

## Per-turn causal order

The live path is read-before-write:

1. Retrieve semantic memories and evidence directly from the current prompt.
2. Use only associations learned on earlier turns to propose bounded additive
   candidates.
3. Pack the final context under the unchanged hard token budgets.
4. Reheat memories that actually reached the prompt.
5. Reinforce only independently retrieved nodes that actually reached the
   prompt.

For a completed interaction, the same causal rule permits **episodic binding**.
The stored initiating prompt and direct prior context are joined to every new
assistant/tool/system chunk through as many fixed-size slices as required. This
update happens only after the new chunks exist and before the next user prompt.
Slice count can grow linearly with a long tool episode, but the transformer
workspace does not grow. Graph-admitted context remains excluded, so the new
write cannot certify a link that the graph invented.

Step 5 deliberately excludes nodes admitted by the consolidation graph. A
graph-selected candidate cannot strengthen the edge that selected it. It must
later be found independently before it contributes another observation.
Ordinary old-to-old co-access still requires two observations by default. One
completed prompt-to-response binding may be read after one observation; this
makes a unique outcome retrievable without weakening the noise guard for
incidental co-access.

## Update rule

For node activity `a_i`, learning rate `eta`, elapsed-turn decay `D`, and an
optional head-derived pair gate `g_ij`:

```text
node_mass_i(t) = D(node_mass_i) + eta * a_i^2
edge_mass_ij(t) = D(edge_mass_ij) + eta * a_i * a_j * g_ij
```

The read score normalizes edge mass by the geometric mean of endpoint masses,
then applies a separate freshness decay. Normalization suppresses ubiquitous
hubs; the extra freshness term prevents an isolated pair from remaining at a
normalized score of one forever while all its masses decay in lockstep.

Evidence from several active anchors combines by bounded noisy-OR rather than
an unbounded sum.

## Qwen activation hyperplane

"A hyperplane for the turn" is operationally a transient CAV-space activation
or low-dimensional subspace over the bounded candidate assembly. It is not a
durable transformer context.

The provider-free fallback assigns rank-discounted node activity. A Qwen prefix
inspector may instead provide:

- CAV-derived activity `a_i` for each durable node;
- a bounded `g_ij` in `[0, 1]` from selected-head QK evidence, OV/CAV alignment,
  or their calibrated combination.

`qwen_head_activations(...)` converts the bounded `MemoryLinkHit` QK and OV
outputs into within-turn normalized node activity, and
`MemoryCondenser.observe_context_access(...)` accepts those activities plus any
pair gates produced by a deeper member-to-member inspection. The inspector's
complete workspace is discarded after the scalar update. The
consolidation tables retain only typed IDs, masses, counts, turns, and event
fingerprints. Existing versioned `chunk_head_edges` remain the audit record for
compiled per-head evidence; schema v9 does not duplicate their tensors.

The Qwen path is an optional teacher. Attention is not factual authority, and a
head-derived gate cannot bypass provenance, status, packing, recurrence, or
degree bounds.

### Operational delayed pass

Production code should keep one `QwenMemoryLinker` resident and call
`consolidate_packed_context(...)` after the response with the exact
`PackedContext` used for generation. This avoids both duplicate retrieval and
per-turn model loading. The operator command exposes the same path and accepts
already packed IDs when debugging:

```powershell
pixi run -e dev qwen-consolidate `
  --data-dir data/session.store `
  --prompt "the completed user turn" `
  --event-id "stable-turn-id" `
  --memory-id <packed-mem-id> `
  --chunk-id <packed-chunk-id>
```

If the explicit IDs are omitted, the command can reconstruct direct retrieval
on CPU. That mode is a convenience for offline operation, not the real-time
path: loading bge-m3 and Qwen in a fresh process for every turn is deliberately
not the deployment design.

## Bounds

Bounds are deliberately explicit:

- at most 16 nodes enter one event;
- at most 32 live consolidation neighbors survive per node;
- at most 4,096 idempotency receipts survive;
- the replay profile uses at most nine nodes per event and three candidates per
  transient Qwen pass;
- learned evidence is additive and cannot evict a direct result merely because
  a graph slot was reserved;
- the packer admits at most three learned evidence candidates under the same
  hard evidence-token ceiling;
- the measured profile diffuses for two scalar graph hops, keeps a width-32
  first frontier and at most 128 candidates, then reranks durable chunk IDs by
  live-query cosine;
- hop-balanced slots prevent near-duplicate one-hop nodes from consuming the
  complete iterative read budget;
- score divided by square-root token cost prevents long marginal candidates
  from hiding short precise evidence while retaining the hard 1,600-token cap.

Receipts contain an event ID and SHA-256 membership fingerprint, not the query
or context. Soft-deleting or superseding a memory, or removing a chunk from its
retrieval indexes, deletes only the corresponding reconstructible graph node
and incident edges. The authoritative rows survive.

## Relation to textual summaries

This implements relational consolidation. It does not yet materialize a cold
era summary. A stable assembly may later earn a compact semantic representative,
but that is a separate, provenance-validated write operation. Association must
first demonstrate that the members recur together; otherwise a summary merely
compresses an arbitrary retrieval accident.

## Falsification

The mechanism earns production influence only if a chronological replay shows
better answer accuracy or equal accuracy at fewer tokens than direct retrieval,
with identical budgets. Required controls are:

- consolidation reads disabled;
- learning disabled;
- shuffled event memberships;
- rank-only learning versus CAV/QK/OV-weighted learning;
- one-observation admission versus the repeated-activation threshold.

The first locked chronological development replay now passes this falsification
check: the Qwen-weighted arm reached 38/39 literal evidence probes (97.44%) with
no losses, versus 35/39 for the original operational pack. This is evidence
reachability, not answer-stage judged accuracy; the latter remains the primary
95% target.
