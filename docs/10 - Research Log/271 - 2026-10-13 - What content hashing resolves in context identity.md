# What content hashing resolves in context identity

**Status:** Finding; scopes the detection layer in [Context Fork and Branch Detection](../02%20-%20Implementation/32%20-%20Context%20Fork%20and%20Branch%20Detection.md)

**Date:** 2026-10-13

**Applies to** the proxy's history diff, and to any scheme that identifies an
incoming context against a stored conversation.

**Depends on:** [Context Fork and Branch Detection](../02%20-%20Implementation/32%20-%20Context%20Fork%20and%20Branch%20Detection.md)

Hashing was the obvious mechanism for recognising a branched, resumed, or
handed-off context. It resolves three of the five relations exactly, needs a
different construction for a fourth, and cannot resolve the fifth at all. The
boundary is worth writing down, because the fifth relation is the one that
occurs most often in practice.

## Measured

Five relations against a stored six-message conversation, scored by a
cumulative prefix chain (`chain_i = H(chain_(i-1) || canonical_i)`) and by a
tail-anchored consecutive run over per-message digests:

| Relation | Prefix chain | Tail-anchored run | Correct resolution |
| --- | --- | --- | --- |
| Identical | depth 6 | run 6 at 0 | same namespace |
| Continuation | depth 6 | run 6 at 0 | same namespace |
| Fork at *k* | **depth 2, exact** | run 2 at 0, ambiguous | fork from *k* |
| Handoff window | **depth 0** | run 2 at 2 | fork from interior |
| Compaction | **depth 0** | run **1** at 5 | same namespace |

The current `MemoryProxy._new_messages` returns HTTP 409 for the last three,
all with the same message, which is what prompted this note.

## What it resolves

**Identical, continuation and fork** are prefix relations, and a prefix chain
answers all three with indexed lookups. The chain is also the only construction
that returns an *exact* fork point; tail matching cannot, because a tail run
starting at offset 0 is indistinguishable from a prefix.

**A handoff window** is not a prefix. A chain commits from position 0, so a
window beginning at turn 8 carries a value unrelated to its parent's. This is
still solvable by incremental hashing, but by a **rolling** construction over a
sliding window (Rabin-Karp, content-defined chunking) rather than a chain.

## What it cannot resolve

**Compaction.** After a client compacts, the incoming context carries a message
that never existed in the stored history: a freshly generated summary. Its
digest matches nothing, by construction. Hashing answers whether two byte
sequences are the same. It cannot answer whether novel text is a lossy
compression of the bytes it replaced.

The general statement: **hashing yields equivalence classes, not intent.** Every
relation above that hashing resolves is one where the bytes were preserved
somewhere. Compaction preserves meaning while discarding bytes, which puts it
outside what any digest can see.

## Consequences

**Invert the anchor.** Chaining the prefix is the instinct and it is backwards
for real clients. The prefix is what they mutate: compaction rewrites the head,
resume replays it, archiving truncates it. The tail is what stays stable from
turn to turn. Tail-anchored matching found a candidate for all five relations
where the chain found none for two. Keep both; they answer different questions.

**Weak matches must not bind a namespace.** Compaction scored a one-message run,
and a single-message coincidence is cheap: an acknowledgement, a repeated tool
result. Admission needs a minimum token mass, not a minimum message count.

**Fork on ambiguity.** The direction of the fail-safe matters more than the
detector. Mistaking compaction for a fork fragments one conversation's memory
across namespaces and degrades recall. Mistaking a fork for compaction merges
two divergent conversations into one namespace and contaminates it with facts
from an abandoned branch. The second is worse and effectively unrecoverable.
Fragmenting loses recall; merging poisons it.

**Prefer a signal to an inference.** Compaction is a client-side event. Augment
mode already requires `x-memory-conversation-id`, so the same shim can carry a
compaction marker, and the detector is then needed only for uncooperative
clients in observe mode.

## Where hashing is constitutive, and where it is only an accelerator

This distinction governs how much machinery is justified.

It is **constitutive** where an artifact's identity *is* its content hash, so
that the artifact can be verified against its inputs and detected as stale.
`identity_sha256` receipts, `quote_sha256` spans, `source_content_sha256` and
`chunk_id = sha256(body)` are all already this.

It is **merely an accelerator** where the mechanism is a policy decision and
hashing only avoids recomputation: a batch admission gate condenses because the
batch would not fit, and the hash only lets an identical batch reuse an earlier
condensation. Useful, but it does not need a chain, a tree, or a registry.

Conflating the two is how a detection layer grows past what the problem needs.
