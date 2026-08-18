# Live Hebbian co-retrieval graph

**Date**: 2026-08-16
**Status**: IMPLEMENTED AND UNIT/INTEGRATION VALIDATED; benchmark admission pending

## Question

Can the live memory system learn that conceptual chunks retrieved together in
one turn form a useful assembly, then bring a missing member back later without
storing transformer context or increasing prompt size?

Before this change the answer was no. `touch_edges` only incremented traversal
counters on QK/OV edges compiled earlier by Qwen, and `touch_signatures` only
updated individual CAV access counters. Neither operation created a new
relation between chunks co-retrieved in a live turn.

## Implementation

Schema v7 adds three compact tables:

- `hebbian_chunk_nodes`: decayed per-chunk activation mass and access count;
- `hebbian_chunk_edges`: symmetric pair IDs, decayed co-access mass, evidence
  count, and last-reinforced turn;
- `hebbian_access_events`: a bounded idempotency receipt containing an event ID
  and SHA-256 membership fingerprint, not query or retrieval text.

`AssociationStore.reinforce_retrieval_coaccess` observes at most 12 ranked
conceptual chunks from a final result set. It updates node mass with `a^2` and
pair mass with `a_i*a_j`, where `a=1/sqrt(rank)` in the facade. Read score is a
hub-normalized cosine multiplied by explicit turn freshness. This separate
freshness factor fixes a subtle cancellation: if node and edge masses all
decay together, an isolated pair's normalized cosine otherwise remains one
forever.

Physical degree defaults to 32 and is enforced after each event. Event receipt
history defaults to 4,096. A repeated event ID with identical membership is a
no-op; conflicting membership is rejected. Different single-concept events do
not create an edge.

`MemoryCondenser.search_hebbian` and `expand_hebbian` reserve tail slots inside
the existing `k`. Strong lexical anchors are protected. The public default
rejects a learned replacement if it adds even one prompt token. Supplying no
event ID makes the read non-learning, which is the required mode for static
evaluation and ablations.

The graph is artifact-scoped and persists only source-grounded chunk IDs and
scalars. Chunk text is hydrated after selection; Qwen is not loaded for reads.
Schema inspection tests reject token, attention, residual, hidden-state, query,
or text columns in the live graph tables.

## Verification

Normal frozen Pixi environment:

```text
pixi run --frozen -e dev python -m pytest \
  tests/test_hebbian_retrieval.py tests/test_association_store.py \
  tests/test_condenser.py tests/test_db.py -q
82 passed

pixi run --frozen -e dev python -m pytest -q
695 passed, 1 unrelated pydantic-settings warning
```

The focused tests cover:

- exact-event idempotency and conflicting-event rejection;
- no false edge across separate turns;
- restart persistence;
- turn-based stale-link cooling;
- hard degree and event-history caps;
- facade-level learned recall;
- fixed result count and token-growth rollback;
- deletion cleanup and zero retained request-derived transformer token state;
  reusable static weights/tokenizers are outside this metric.

Ruff is not installed in the frozen dev environment, so no lint result is
claimed.

## Result and limit

The mechanism now exists and its storage/budget invariants are enforced. It is
not yet evidence of improved LongMemEval answer accuracy. Most LongMemEval
samples have a terminal question and therefore mainly test cold-start safety;
they do not supply repeated online access events from which this graph can
learn. The right next experiment is a chronological long-chat sequence with
multiple real retrieval/generation turns, plus shuffled-membership and
frequency-only negative controls, all at the selected v2 policy's exact prompt
budget.

See
[`02 - Live Hebbian Co-Retrieval Memory.md`](../00%20-%20Theory/02%20-%20Live%20Hebbian%20Co-Retrieval%20Memory.md)
for the learning rule, safety boundary, and acceptance criteria.
