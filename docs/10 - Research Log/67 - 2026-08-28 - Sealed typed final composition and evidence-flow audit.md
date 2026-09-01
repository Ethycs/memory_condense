# Sealed typed final composition and evidence-flow audit

**Date:** 2026-08-28

**Status:** provider-free 100-question composition sealed and independently
audited; evidence losses localized before live answer scoring

## Result

The common typed-memory final arm completed one provider-free composition over
the locked 100-question population. The population contains ten independently
ingested long-memory namespaces and ten questions per namespace. Every
namespace cache was read from SQLite once. The sealed cache receipts contain
74,989 content rows, 79,798 physical rows, 564,665 sentence windows, 100
partitions, and 10,310,777 content tokens.

The principal artifacts are:

| Artifact | SHA-256 |
| --- | --- |
| full-store closure input | `044e60f308287dda4d87106646e4cc56f0e96d513b2bfd03a7473da9994ef5c4` |
| typed composition | `53842b2940aa3aac7215939be20f4e6af82ba77d127bb0b03277bca37705d6e5` |
| Terra preflight | `7d4329e1459b3001ce4473ff189809779c4cd158624d0d50b894feb19fd92a9f` |
| target-flow assay v1 | `43a6e3bd9aa9821abcdbe7eed5245e90b459dbd02bb553c4423d4621fce7bb97` |
| target-flow assay v2 | `59282a8a47d0c112e468e11d11281041e863dd4dfe31312e04b5234d08c939c4` |

Composition and preflight replayed without changing bytes. They loaded no
gold or target registry, made zero provider calls, exposed no raw source,
namespace, partition, or store locator to the provider projection, and retained
zero transformer token-state bytes.

## Exact evidence flow

The independent audit reconstructed every item and binding transition:

```text
3,591 retrieved local items
    -> 3,591 after exact post-selection dedup
    -> 1,887 retained by non-borrowable method lanes
    -> 1,820 retained by fair merge
    -> 1,820 retained by final hard-cap fitting
```

There were no exact-provenance dedup exclusions and no final hard-fit drops.
The 67 fair-merge drops comprised 43 adaptive-map items and 24 full-store
items. All final item, handle, group, mechanism, frontier, validation, and
connectivity sets agreed exactly.

The final evidence population contained:

- 909 active-reconstruction chunks;
- 566 full-store chunks;
- 205 adaptive parent-map items;
- 107 parent direct pointers;
- 13 base Direct+Guided source facts;
- 16 tail direct pointers; and
- four tail source facts.

All 1,475 retained full-store and active-reconstruction chunks preserved exact
source bytes, character lengths, hashes, and prompt-external provenance.
Every one of the 1,820 provider-visible opaque handles mapped to one sealed
local binding.

## Prompt budget

The complete prompt was well below the declared boundary:

| Statistic | Input tokens | Input plus 768-token output reserve |
| --- | ---: | ---: |
| minimum | 2,262 | 3,030 |
| mean | 2,922.90 | 3,690.90 |
| median | 2,878.5 | 3,646.5 |
| maximum | 3,884 | 4,652 |

The 8,000-token limit therefore was not the cause of the final evidence loss.
The large unused budget led directly to the serialization-boundary diagnosis
recorded in Research Log 68.

## Post-hoc target-flow assay

Only after the composition and preflight were sealed did the analyzer open the
historical 72/100 judge result and the runtime-forbidden target-owner plan. It
analyzed exactly the 28 baseline misses and 84 declared target components.

The v2 lifecycle assay added explicit lane, fair-merge, and hard-fit stages.
For the 59 exact source targets, the funnel was:

```text
46 retrieved -> 46 post-dedup -> 41 lane-selected
             -> 41 fair-merged -> 41 hard-fit -> 41 globally bound
```

The five retrieved source targets lost before the final prompt were all lost
inside per-method lane selection:

- q28: `answer_cc021f81_2`;
- q53: `answer_c2204106_2` and `answer_c2204106_1`;
- q54: `answer_56521e66_1`; and
- q67: `answer_990c8992_3`.

No target source was lost by deduplication, fair merge, final fitting,
provenance binding, or story binding. The remaining source deficit was 13
targets never retrieved locally. Those misses are concentrated in dispersed
temporal joins, with smaller preference, numeric, representative, and
insufficient-evidence cases.

Across all 84 source, relation, and coverage targets, 52 reached the global
prompt boundary and five were named by a deterministic operator advisory.
Only 10 of 23 required CAV-style relations survived with all operands and an
explicit story or operator link. This is a coverage assay, not an answer score.

## Decision

- Preserve the typed, provenance-exact common-memory plane.
- Treat the 13 unretrieved source targets as discovery/local-to-global
  connectivity failures.
- Treat the five retrieved-but-lost source targets as lane ranking/allocation
  failures.
- Do not attribute either class to deduplication or the terminal 8,000-token
  fitter.
- Measure the sealed arm with the fixed Terra/Sol answer protocol, then repair
  retrieval and lane flow from the measured failure set.
