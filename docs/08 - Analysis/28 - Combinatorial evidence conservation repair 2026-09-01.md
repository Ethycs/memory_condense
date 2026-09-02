# Combinatorial evidence conservation repair

**Date:** 2026-09-01

**Status:** implemented and regression-tested; provider-free; no new answer-score
or 95% claim

## Decision

Retrieval stages may rank, budget, and deduplicate evidence, but no stage may
silently turn a heuristic absence, a failed specialist attempt, a duplicate
representation, or a full budget into proof that the underlying evidence does
not exist. The repaired stack therefore applies four rules:

1. heuristic negatives rank or audit; only authenticated impossibility can
   prune;
2. every hard limit leaves an explicit omitted, truncated, exhausted, or
   unresolved receipt;
3. mechanisms receive their independent admission opportunity before exact
   cross-mechanism deduplication; and
4. capacity released by deduplication is reconsidered against the exact sealed
   skipped order instead of being abandoned.

This is an evidence-conservation result. It establishes that the implemented
combinations no longer discard evidence in the reproduced cases. It does not
establish a new LongMemEval score, a 1M-token accuracy result, or progress to
the preregistered 95% answer gate.

## Defects reproduced

The audit found several individually reasonable operations whose composition
was non-monotonic:

- a branch containing one operand could be pruned because it lacked another
  operand's literal or required role;
- a heuristic dense/sparse dual gate was treated as a definitive negative;
- hydration retained only user neighbors and could slice away the only
  available obligation witness;
- entity-obligation construction stopped after four targets without leaving a
  closure record for later targets;
- one no-progress source-gate batch suppressed the rest of that lane, while a
  progressing lane could monopolize later rounds;
- a full unique-source budget could block later candidates from already-known
  sources;
- an invalid or abstaining V3 attempt could be marked as an applied terminal
  repair;
- a partially fitted numeric operand population could be described as a
  complete deterministic advisory;
- a hard-8k rejection, a no-op packet subset, or an oversized first item could
  create false closure or phantom group coverage;
- exact deduplication could erase a specialist representation before that
  specialist received its lane, or free capacity after selection without
  admitting the next unique skipped item; and
- compact-v2 group aliases could collide with literal group keys.

The failures are combinatorial because each local operation looked valid in
isolation. The error appeared only after ordering two or more of pruning,
selection, budgeting, deduplication, closure, and terminal fitting.

## Implemented repairs

### Conservative residual and global search

Exact-literal absence, required-role absence, and the dual heuristic gate remain
sealed diagnostic reasons but classify a branch as `may_answer`. Global tree
search likewise no longer treats literal or role absence as a proof of no
support.

Source-local hydration now keeps assistant and system segments as well as user
segments. One available witness per compiled obligation is moved to the front
of the hydration population. `max_hydrated_segments` remains a hard cap: excess
witnesses stay in the authenticated omitted partition, make hydration
incomplete, and keep global routing open.

`max_entity_obligations` now controls the primary entity-priority prefix rather
than deleting later query targets. Overflow entities are appended after the
date, role, and numeric structural obligations, so every target has either a
covered or unresolved closure identity. Independent lane exhaustion also keeps
`needs_further_global_search` true.

### Source-gate progression

No-progress consumes only the attempted prefix, not the whole lane. Tail lanes
now rotate strictly in route order whether or not the prior lane made progress,
so a stream of partial gains cannot starve the other specialists.

At the unique-source cap, unaffordable new-source candidates are skipped rather
than terminating the scan. The controller continues through the current lane
and later lanes for candidates whose source is already known and therefore
costs no additional unique-source slot. It stops with `UNIQUE_SOURCE_CAP` only
when no such candidate remains.

### Applied-repair and specialist completeness

A combined V3 answer is terminal only when the combined lane is non-fallback,
the parse is clean, the solver has not declared it invalid, and the prediction
does not abstain. A failed V3 attempt remains visible even when a prior answer
is retained as the answer basis.

Numeric specialist advisories are now all-or-nothing over the sealed operand
groups. If fitting removes every witness for any group, or handle ownership is
not injective, the whole deterministic numeric advisory is omitted. A surviving
subset is never relabeled as the complete reduction universe.

### Typed frontier and lane integrity

Any `hard_8k_*` rejection makes the typed frontier truncated and non-closed.
An identity packet subset preserves the original bindings, truncation state,
closure state, and frontier receipt. Compact-v2 aliases every non-null group
key injectively, including literal values that already look like `K001`.

Lane diversity is updated only after an item is usable and actually fits. An
unusable or oversized first candidate therefore cannot claim source-group
coverage or prevent a later candidate from satisfying that mechanism's
minimum.

### Admission, exact dedup, authority transfer, and refill

Typed additive composition now executes:

```text
independent lane minima
  -> shared-surplus admission
  -> exact cross-mechanism dedup
  -> freed-capacity backfill
  -> fair final packet fit
```

Every original lane minimum is translated through the exact dedup receipt to a
retained representative. The fair merge verifies that the representative and
its owner bindings survive. It no longer requires the original mechanism to
own the representative, because an exact duplicate in another mechanism can
legitimately carry that authority.

The backfill pass reconsiders only the surplus allocator's sealed
budget-omitted order. It skips exact cross-mechanism duplicates, admits unique
items only while the original aggregate lane cap still fits, and seals every
admitted, duplicate, and capacity-unfit disposition. This closes the case where
a duplicate consumed the last shared slot, was removed, and left a unique next
item stranded.

The semantic-global terminal has the analogous repair behind explicit
successor formats:

- v2 and linked v3 remain byte-compatible historical formats;
- backfill v4 and linked-backfill v5 opt into the new behavior; and
- replay derives the feature set from the sealed format.

Each plane reuses its original item/token budget and authenticated skipped
consideration order. Only globally novel exact spans enter the backfill. A
`PostDedupBackfillReceipt` binds the initial dedup, all plane selections,
considered rows, admissions, and final retained population.

## Verification

No provider or judge calls were made and no sealed historical artifact was
rewritten.

- Initial integrated combinatorial slice: 164 passed.
- Typed/locked downstream slice: 65 passed.
- Terminal construction and replay slice: 26 passed.
- Combined repaired core slice: 145 passed.
- Complete `test_matched_eval_*.py` suite: **896 passed, one skipped**, in
  419.56 seconds.
- Post-suite receipt-hardening focus: 22 passed across terminal backfill and
  typed additive composition.
- `git diff --check`: clean.

Focused regressions include split-literal branches, assistant answer bridges,
non-user neighbors, low-ranked hydration witnesses, fifth-entity overflow,
lane exhaustion, invalid and abstaining V3 attempts, strict three-lane
rotation, same-lane and cross-lane known-source aliases at the source cap,
phantom group coverage, hard-8k false closure, duplicate-minimum authority
transfer, and freed-capacity admission of the next unique item.

## Remaining limits

1. Witness protection begins after the bounded tree frontier. A low-scoring
   witness outside `max_node_visits` or `max_retained_leaf_cells` is not
   recovered by hydration. This is no longer silent—the unresolved tree and
   closure remain open—but recall still needs a bounded obligation-directed
   second descent over unresolved nodes.
2. Semantic-global terminal backfill is an opt-in v4/v5 successor. Existing
   locked v2/v3 construction scripts deliberately remain historical. A new
   named full-population construction must select the successor before this can
   affect a measured answer score.
3. Typed backfill can restore an item under the aggregate lane envelope, but
   the final 8k fit may still omit it when protocol metadata and protected
   minima consume the provider envelope. That omission remains explicit.
4. Conserving all entity targets can increase the obligation population for a
   long enumerative question. The hydration and provider caps stay hard; the
   cost and recall effect need measurement rather than another silent cap.

## Next experiment

Create a newly named semantic-global terminal construction using linked
backfill v5, run it first on the fixed known-miss fixtures, and compare:

- unique exact spans before and after dedup;
- backfilled spans and final 8k survival;
- unresolved obligations inside versus outside the tree frontier; and
- answer accuracy under the same responder/judge protocol.

In parallel, implement the obligation-directed second descent as a bounded,
receipt-bearing successor. Do not raise the general tree cap or relabel an open
frontier as complete.
