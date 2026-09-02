# Combinatorial evidence conservation repair

**Date:** 2026-09-01

**Status:** implemented, provider-free, and regression-tested; no new benchmark
score

## Question

Were remaining misses being caused only by weak retrieval, or were combinations
of otherwise valid stages excluding evidence that one of the mechanisms had
already found?

## Result

The audit reproduced multiple composition defects. The common pattern was a
loss of monotonicity across stage boundaries: heuristic absence became a hard
negative, one attempted prefix became lane exhaustion, a duplicate consumed a
budget before being removed, or a partial operand population was presented as
complete.

The repaired invariant is:

> rank and budget freely, but every exclusion needs exact authority and every
> exhausted boundary must remain visible to closure.

## Repairs

- Residual literal, role, and dual-gate negatives now fail open.
- Global hydration retains the complete local discourse neighborhood and
  prioritizes obligation witnesses inside a hard hydration cap.
- Entity targets after the fourth remain required or unresolved instead of
  disappearing.
- Source-gate tails rotate strictly and can use known-source evidence even when
  the unique-source budget is full.
- Invalid, parse-failed, or abstaining V3 attempts are not terminal repairs.
- Numeric advisories require every sealed operand group after final fitting.
- Hard-8k rejection, no-op subsetting, and lane admission preserve honest
  frontier and diversity state.
- Typed mechanisms receive independent lane admission before exact dedup;
  minimum authority transfers to a surviving representative.
- Dedup-released capacity is backfilled from the sealed omitted order.
- Semantic-global terminal v4/v5 adds the same refill as an explicit successor
  while historical v2/v3 remains byte-compatible.

Two concrete counterexamples now pass:

1. a specialist's last shared slot is spent on the same exact span already
   owned by the parent; after dedup, the specialist's next unique item is
   admitted into the freed capacity;
2. the source budget is full, the next candidate names a new source, and a
   later candidate or later lane names an already-known source; the known
   candidate remains selectable at zero unique-source cost.

## Verification

- 164/164 initial integrated combinatorial tests passed.
- 65/65 downstream typed and locked tests passed.
- 26/26 terminal construction/replay tests passed.
- 145/145 combined repaired-core tests passed.
- Complete matched-eval suite: **896 passed, one skipped** in 419.56 seconds.
- Post-suite receipt-hardening focus: 22/22 passed.
- `git diff --check` passed.

No provider calls, judge calls, new sealed campaign artifacts, or score updates
were made. This log records apparatus correctness, not a gain toward 95%.

## Interpretation

The repair removes avoidable evidence loss after retrieval and makes remaining
budget/frontier failures explicit. It does not prove that the missing answer is
inside the searched frontier. The important remaining construction gap is a
bounded obligation-directed second descent for witnesses outside the first
tree-retention frontier.

The semantic-global backfill is intentionally versioned and opt-in. Its next
use should be a new linked-v5 construction on fixed misses, followed by the same
answer and judge path. Historical v2/v3 artifacts must remain frozen.
