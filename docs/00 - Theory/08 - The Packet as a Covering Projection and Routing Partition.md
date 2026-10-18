# The packet as a covering projection and routing partition

**Status**: THEORY. The coverage machinery, the hierarchy tree, and the
feedback sink exist and are sealed; the proxy packet does not carry them, and
no component consumes the citations the prompt already asks for
**Date**: 2026-10-13
**Applies to**: proxy memory mode, large context admission, client-side
compaction, and retrieval routing
**Depends on**: [Proof-carrying factual retrieval](04%20-%20From%20Top-K%20Recall%20to%20Proof-Carrying%20Factual%20Retrieval.md),
[Live Hebbian co-retrieval memory](02%20-%20Live%20Hebbian%20Co-Retrieval%20Memory.md),
[What content hashing resolves](../10%20-%20Research%20Log/271%20-%202026-10-13%20-%20What%20content%20hashing%20resolves%20in%20context%20identity.md)

## Claim

The retrieved packet is treated as a per-turn payload. It should be a
**persistent covering projection** of memory. Promoting it collapses three
problems that currently have three separate answers:

```text
coverage proof -> bounded projection -> partition of memory
              -> addressed by labels -> output citations route the next retrieval
```

1. **Compaction stops being a question.** Not detected, obviated.
2. **Proof scope stops being per-turn.** Obligation five of proof-carrying
   retrieval, "what corpus or partition did the system actually inspect",
   becomes a durable property of an accumulating layer rather than a quantity
   recomputed and discarded each turn.
3. **The packet's own labels become an address space** that the answer's
   citations route over, turning generation output into a precise access signal.

This is a continuation of proof-carrying retrieval, not a replacement. The
receipt stays the unit of honesty. What changes is that the receipt persists,
accumulates, and is addressable.

## Why compaction is the wrong frame

A client compaction replaces history with a summary that never existed in the
transcript. Its digest matches nothing, so no hashing scheme can recognise it;
hashing yields equivalence classes, not intent. Attempting to detect it leads
to a heuristic whose failure modes are both bad: mistaking compaction for a
branch fragments one conversation's memory across namespaces, and mistaking a
branch for compaction contaminates one namespace with an abandoned branch.

Memory mode already discards the client's history and substitutes recall
evidence plus a recent window. So client compaction is redundant work upstream
of a system that was going to replace the history regardless. The question
worth answering is therefore not *did the client compact* but *does the
emission cover the memory*. The first is undecidable. The second is already
computed.

## The covering projection exists

Three pieces are in place and sealed.

`discourse_artifact_coverage` maps artifact to chunk with `coverage_kind`,
`source_revision`, `chunk_identity_sha256`, and a status in
`('annotated', 'no_output')`. That status is what makes a cover **provable**
rather than merely asserted: it distinguishes a chunk that was inspected and
yielded nothing from a chunk that was never inspected. Without that
distinction, absence of output is indistinguishable from absence of effort.

`discourse_artifact_coverage_receipts` seals the set per revision with both
`coverage_sha256` and `turn_coverage_sha256`, plus `chunk_count`. Coverage is
therefore relative to a revision, which is what makes staleness detectable.

`HierarchyContext.receipt()` emits `complete_index = omitted_sections == 0`
beside `token_count` and `max_tokens`. That is the covering predicate itself,
computed and recorded on every render. And `PreparedHierarchyContext` carries
`child_section_ids` and parents, so the projection is a **tree**. This matters
more than it appears: bounded context plus total coverage is unsatisfiable by
a flat projection, so a hierarchy is not a convenience but a requirement.

## The packet as a partition

A coverage receipt partitions memory into three classes, and each class implies
a different retrieval action:

| Class | Evidence | Action on the next turn |
| --- | --- | --- |
| Projected | present in the emission | do not search; it is already in context |
| Covered, omitted | counted in `omitted_sections`, represented by a parent | search only if the parent is cited or scored |
| Uncovered | outside `source_revision` | the only region requiring fresh retrieval |

This is the routing substance. A search that knows which partition it already
projected does not re-enter it, and a projection that declares its own
omissions tells the router exactly where the unexplored region is. Precomputed
work is reused rather than recomputed, and the tail can be closed honestly
because the boundary is recorded rather than assumed.

## Labels as an address space

`PreparedHierarchyContext` assigns stable aliases over content-addressed
section identifiers, and `HierarchyContext.NOTICE` already instructs the
answerer to "cite H labels for summary-derived claims".

**Nothing consumes those citations.** The prompt requests an address and the
system discards the reply. This is the open half of the loop, and the closed
half already exists: `learn(packet, access_event_id)` is the sink,
`hebbian_access_events` the store, and co-access and heat diffusion the
consumers.

A citation is a stronger signal than inferred relevance. Scored relevance is
the system's guess about what mattered; a citation is the answerer naming the
partition element the claim was drawn from. It identifies an element, not a
neighbourhood, and it arrives after generation rather than before, so it
reports what was *used* instead of what looked promising.

A cited label absent from the emitted packet is also detectable, and that is
free integrity checking rather than a nuisance: a label that was never offered
cannot have been read, so its appearance is evidence of fabrication and
belongs in the receipt.

## Two channels in one payload

Coverage and relevance pull against each other. A cover must include material
irrelevant to the query; relevance selection must discard material to fit. One
ranked list cannot be both, so the packet carries two channels with distinct
trust levels: a **relevance head** that is deep, exact and quotable, and a
**covering spine** that is shallow, complete, lossy and labelled.

The trust gradient is already written into `NOTICE`, which marks the hierarchy
as "lossy historical reports, not verified raw evidence", directs exact
verification to the raw passages, and states that "coverage does not establish
that every claim is true". Coverage is a claim about inspection, never about
truth.

## The promotion rule, which must be specified first

Accumulating, covering, and bounded are jointly satisfiable only if the
hierarchy **deepens**. If sections accumulate at one level, `omitted_sections`
grows monotonically, `complete_index` latches false, and it never recovers.

The failure is quiet in the worst way: the receipt reports the degradation
honestly while nothing acts on it, so the system keeps emitting a projection
that is no longer covering and keeps labelling it with a predicate that says
so. A rule for when a level exceeds budget and rolls into a parent is
therefore a precondition of the whole guarantee, not a later refinement.

## Obligations this creates

**Pruning must re-prove coverage.** A prune invalidates a cover by removing
referents. `source_revision` and `coverage_sha256` detect it; the prune must
bump the revision and re-establish the proof rather than silently converting
audited memory into unaudited memory whose receipts still appear valid.

**Construction belongs off the critical path.** Extending the layer per turn
is summarisation per turn, which is exactly where local hardware saturates.
The spine must extend in the background against the committed snapshot, which
the existing `recall_published` design already supports.

**The trust gradient must cross the proxy boundary.** Evidence deliberately
enters as a user message to keep untrusted source text out of the instruction
channel. A lossy spine and exact quotations then share one channel, so labels
and notice must survive intact or the answerer will cite a summary as though
it were a quotation.

## The pin, and the second learning rule it would permit

There should be a durable pin between the packet, the query that produced it,
and the success of the continuation. Two of the three legs already exist.
`packets(packet_id, query, text, refs, input_event_id, recent_events)` binds
the query to the projection; `feedback(packet_id, successful, event_id,
applied)` binds the projection to an outcome, keyed on the packet and
journalled as an event.

The third leg is a **declared constant**. There is one writer, and it is
unconditional:

```python
def record_use():
    # Co-access is the existing Hebbian signal. A completed exchange
    # confirms delivery/use, not correctness or comparative importance.
    for packet in packets:
        if packet.references:
            self.session.feedback(packet.packet_id, successful=True)
```

`ChatSession.feedback` says the same in its own words: "a co-access signal, not
a quality grade". So what is pinned today is delivery, accurately and by
intent.

**This is not a defect, and filling it would not unlock the existing Hebbian
system.** Correlational plasticity consumes co-occurrence, not reward, so a
constant is the right input: the call asserts that a packet was genuinely used,
and `_apply_feedback` routes that to `backend.learn(packet, access_event_id)`.
Credit is not flat within a packet either, since `rank_discount` is applied at
`consolidation.py:134`, and `decayed_mass`, `coaccess_edge_score` and
`select_prune_victims` complete an unsupervised associative memory with
recency decay and pruning.

What an outcome signal would add is a **second, different rule**: two-factor
Hebbian plasticity, pre times post, gaining a third modulatory factor. That is
an addition, not an activation, and two structural facts currently prevent it.

**There is no signed update.** `_apply_feedback` reads
`if successful: backend.learn(...)`, so a false outcome means *skip
reinforcement*, never *weaken*. The association modules carry only
non-negativity validators, so negative mass has nowhere to go. Potentiation or
nothing.

**The pin is foreclosed at delivery.** The unconditional true fires when the
exchange completes, and `feedback()` raises when a later call contradicts a
recorded outcome. An outcome observed on the following turn therefore cannot be
recorded at all. Delivery and outcome must become two facts rather than one
overloaded bit before anything else here is reachable.

The pin must also span turn *n* to turn *n+1*, where today it terminates inside
turn *n*. `input_event_id` is the available hook. The signal that would fill it
is implicit: a next turn that repairs or re-asks is evidence against the packet
that served the previous turn, and a conversation that moves on is weak
evidence for it. That observation is available **only** at the proxy, because
only the proxy sees both turns.

The risk deserves equal billing. Reward modulation on a noisy teacher can be
worse than no teacher, and correlational learning is robust precisely because
it has none. A user who rephrases because they changed their mind is
indistinguishable from one who rephrases because the answer was wrong. Any such
rule must be additive, separately weighted, and measured against the
unsupervised baseline rather than replacing it.

## What would falsify this

**If `complete_index` is never true at realistic conversation sizes**, the
covering property is aspirational, and the promotion rule is a prerequisite
rather than a refinement. This is one measurement against existing code and
should be taken before any of the above is built.

**If citations are unreliable**, the routing loop is noise. The cheap test is
whether cited labels exist in the emitted packet; a low hit rate falsifies the
address-space claim, while a high one makes the loop worth closing.

**If the two channels cannot coexist in budget**, the packet cannot be both
covering and relevant, and the design reduces to proof-carrying retrieval with
a persistent scope record, which remains worth having on its own.

**If repair does not separate from a changed mind**, the third factor is noise
and must not be built. The test is prior to any schema change: label a sample
of consecutive turn pairs for whether the second repairs the first, and measure
whether any proxy-observable feature separates them. A rule that cannot beat
the unsupervised baseline on held-out recall is a regression however principled
its derivation, because it degrades a structure that currently works.
