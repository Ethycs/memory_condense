# Live Hebbian co-retrieval memory

**Status**: IMPLEMENTED — schema v7 and bounded read/write API
**Date**: 2026-08-16
**Applies to**: live memory access learning, CAV-centered conceptual chunks,
and fixed-budget associative recall

## Decision

The system should learn an undirected Hebbian relation between conceptual
chunks that were actually retrieved together for one model turn. Static QK/OV
compilation says which memories a selected head connected in a bounded
inspection. The live co-retrieval graph adds a different signal: which
source-grounded concepts repeatedly proved useful in the same exposed context.

This is not a K/V cache and it is not a growing transformer context. A node is
the durable `chunk_id` of a conceptual episode. An edge contains only two IDs,
two scalar counters, and a turn coordinate. Text remains in the source chunk
table and is hydrated only after a bounded graph read selects a winner.

The graph is useful precisely because it changes while the conversation is
running. A static CAV or head edge can seed candidate discovery; repeated
successful access then makes the relevant local concept assembly easier to
recover without another Qwen pass.

## Learning rule

For one final retrieval event, let `a_i` be concept `i`'s activity. The current
implementation derives it from rank as `1/sqrt(rank)`, avoiding dependence on
incomparable raw score scales. At turn `t` it updates:

```text
node_mass_i(t) = decay(node_mass_i) + eta * a_i^2
edge_mass_ij(t) = decay(edge_mass_ij) + eta * a_i * a_j
```

Only concepts in the same event produce a pair update. Concepts seen in
separate turns do not become associated merely because both are popular.

Read strength is hub-normalized and explicitly cooled:

```text
cosine_ij = edge_mass_ij / sqrt(node_mass_i * node_mass_j)
score_ij(t) = clamp(cosine_ij, 0, 1) * turn_decay(last_reinforced_ij, t)
```

The cosine term prevents a universally retrieved hub from winning solely on
frequency. The final freshness factor is necessary because the three masses
otherwise decay in lockstep and an isolated normalized pair would never cool.
Evidence from several active anchors combines by bounded noisy-OR rather than
an unbounded sum.

This edge means **co-access**, not logical support, semantic equivalence, or
causation. QK, OV, CAV, temporal, and provenance relations keep their distinct
types.

## Live flow

```text
query
  -> hybrid anchors
  -> optional compiled QK/OV or heat candidates
  -> one reserved Hebbian replacement slot
  -> hard item/token admission check
  -> final chunks exposed to the model
  -> one idempotent co-access observation
  -> discard all query/token/head workspace
```

Learning must happen from the final exposed set, not every ANN candidate. A
read-only evaluation omits the event ID and therefore cannot train the graph.
An exact retry with the same event ID and membership is a no-op; reusing that
ID with a different set is rejected.

## Boundedness and feedback controls

The defaults are deliberately conservative:

- at most 12 concept nodes enter one access event;
- at most 32 live Hebbian neighbors survive per node;
- only 4,096 event fingerprints are retained for retry idempotency;
- an event receipt stores a SHA-256 membership fingerprint, not the query or
  retrieved payload;
- `search_hebbian` reserves one slot inside the existing `k` rather than
  appending context;
- a replacement that adds any prompt tokens is rolled back by default;
- strong lexical anchors are protected from learned replacement;
- all traversal is one-hop and bounded before chunk text is hydrated.

Degree pruning bounds edge storage by the number of durable conceptual chunks,
not conversation length squared. Receipt pruning bounds per-turn bookkeeping.
The schema has no columns for query text, token IDs, attention, K/V, residuals,
or hidden states, and reports `retained_request_token_state_bytes = 0`.
That metric covers state derived from a request (token IDs, Q/K/V, attention,
residuals, and generation K/V), not reusable static model weights or tokenizer
assets. `retained_token_state_bytes` remains only as a compatibility alias.

The explicit event boundary also limits positive-feedback loops. Merely
inspecting an edge does not reinforce it. The caller records only the set that
really reached the model, and a single event cannot reinforce itself twice.

## As-built API

- `AssociationStore.reinforce_retrieval_coaccess(...)` performs the durable,
  idempotent update and pruning.
- `AssociationStore.hebbian_neighbors(...)` returns hub-normalized, decayed
  chunk IDs and scalar evidence.
- `AssociationStore.prune_hebbian_edges(...)` enforces the physical budget.
- `MemoryCondenser.observe_retrieval_access(...)` turns ranked results into one
  live event.
- `MemoryCondenser.search_hebbian(...)` uses learned links inside reserved
  retrieval slots; passing `access_event_id` also observes its final result.

The graph is artifact-scoped. A Qwen checkpoint/CAV/head interpretation cannot
silently share learned access weights with another artifact.

## Evaluation requirement

The feature is implemented, not yet admitted as the default retrieval policy.
Its fair test is a chronological multi-question conversation where only past
access events are visible. Compare hybrid control against Hebbian replacement
at identical final item and prompt-token budgets. Report answer recall,
evidence-source coverage, mean context tokens, edge count, degree distribution,
and cold-start versus warmed performance separately.

LongMemEval samples with only one terminal question cannot demonstrate online
learning within that sample; they are mainly a cold-start safety check. The
stronger test is a long chat with recurring entities/tasks and later questions
that require an earlier concept assembly. Shuffled event memberships and
frequency-only edges are required negative controls. The Hebbian arm earns
promotion only if it raises recall without exceeding the locked token budget
or degrading cold-start retrieval.
