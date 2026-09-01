# First-principles composition checkpoint

**Date:** 2026-08-27

**Status:** isolated evidence-map solver rejected; source-history composition in progress

## Result

The two-pass V2 evidence-map arm was completed and judged on the same locked,
analysis-used 100-question population as the 71/100 direct query-payload arm.
It is not an improvement.

| Plane | Calls | Preflight SHA-256 | Run / replay SHA-256 | Runtime or score SHA-256 |
| --- | ---: | --- | --- | --- |
| evidence map | 91 Terra | `5bffdac293c064b13bbc8580a0453ac1413bb0b5f3b8465690a66e945e8b8afe` | `f658d41b02bb764f85443af055530b0177e715d4e426e77061c4b6e975fce7bd` | runtime `0967c0206a0e6a5ee02eaa5995b33b6d7d5dd258c67dffbb763d9e55f83a974c` |
| isolated map solver | 91 Terra | `228ed6f1f3438c30ce7d724f398f7cd971a06319616303775fa2c6ed5a1f82a4` | `603b8b841ab199c64c72d220fc3ed762833cdc0e61b55d99550bfe1197887d21` | runtime `73c54f2964b91b6a35c540ef669d6b98c3cab6673ecb1246a8dcbc43d8420a3b` |
| changed-only judge | 11 Sol | `280eceaf27fc45e92ab0a9959b6780d6668f5be757431298c552de915aa4ffff` | `2d707d4429d39cf6bae1b9c800b0c9190e34b92dbaf10e33af70f795bdc6dd54` | score `9da9f41f94e1fcfe2d9a03ffd6a2fcf8ea7d08f5dd3271dea533984c7adfda2e` |

All three planes replay byte-identically with zero new provider calls. The map
accepted 268 individually validated items and rejected five. Nine state-chain
rows intentionally made no map call. Every one of the 91 submitted rows
retained at least one validated item.

The solver changed 11 predictions. Against the sealed direct-parent verdicts,
those changes produced one rescue, five regressions, and five neutral changes:

| Route | Changed | Parent correct | Child correct |
| --- | ---: | ---: | ---: |
| direct extraction | 4 | 3 | 2 |
| numeric reduction | 3 | 2 | 0 |
| set join | 1 | 0 | 0 |
| synthesis | 3 | 2 | 1 |
| **total** | **11** | **7** | **3** |

The final score is **67/100**, four points below the direct arm's **71/100**.
The evidence map remains a useful representation and routing plane; its
standalone solver policy is rejected from cumulative composition.

## Failure mechanism

The V2 solver allowed a validated map item to authorize replacement. Exact
quote validation proves that an extracted statement occurs in the supplied
direct neighborhood. It does not prove that the statement is the complete
operand set, latest event, requested set, or correct answer shape. The solver
therefore sometimes replaced a correct direct answer with a locally supported
but incomplete answer.

This separates two guarantees that had been conflated:

```text
fact is grounded in evidence != evidence is sufficient to replace the answer
```

The negative result is consistent with the first-principles audit: mapping is
a representation step, while sufficiency and answer operation are later
steps. Adding an LLM after the map does not close missing retrieval or
completeness by itself.

## Composition repair

The source-history path now has provider-free contracts for the complete
child operation:

1. bind the terminal V2 map to unresolved question-only obligations;
2. select direct, partition, and guided sources independently;
3. deduplicate physical work only after those logical selections;
4. hydrate exact selected source histories;
5. map each question-bound history window to exact-quote facts;
6. validate and pack facts in non-borrowable method lanes; and
7. issue one conservative final answer call.

The packet seam retains two different identities instead of overloading one:
the immutable raw source packet and the terminal evidence-map packet. The
source adapter validates the former; the fact union and final solver inherit
the latter. The corrected terminal adapter population is
`229c86490a32f9654a6cb12646734c67aa0718e822e014ab0076b850f0b29ea0`,
with 97 question-only activations.

The combined solver defaults to `keep_parent`. A `replace` response is valid
only when its cited evidence IDs contain at least one admitted, post-map
source-history fact. Map-only replacement is rejected and the exact direct
prediction is reused. This is deliberately stricter than the failed V2
solver; the live source-map output is the remaining input before the combined
arm can be run.

## Call-economy gate

The default direct-5 plus guided-2 base round activates 97 questions and makes
678 logical method selections. The first source-only projection estimated 549
unique physical sources. The later exact prompt-safe preflight, after the dual
packet repair and immutable-store revalidation, contains 520 unique sources,
1,413 logical history windows, 1,137 unique physical windows, 31 deferred
windows, and **1,106 required Terra calls**. Its largest prompt plus 1,024-token
output reserve is 7,989/8,000. The sealed preflight is
`b21f8522edeb48b40e65b455949835205459a92b644a8f5b49c2413e75a78380`.
That population is a valid upper construction, not an authorized or economical
treatment; no mapper provider call was made from it.

The exact preflight also exposed two multiplicative overheads. The current
adapter compiles every query-plan entity into an independent mandatory SUPPORT
obligation, producing 399 obligations and leaving 97/100 questions activated.
Those entities are search hints, not necessarily separate answer requirements.
Second, every mapper prompt repeats long source/chunk IDs and provenance hashes
that local validation already owns. In one representative five-chunk prompt,
the source JSON contained 3,740 characters of source text and 2,912 characters
of metadata. The repair under test uses one any-anchor support obligation plus
at most one typed operation obligation, retains a separate parent/map
disagreement trigger, and sends short provider-facing aliases that rebind to
the full prompt-external receipts locally.

A provider-free policy sweep is therefore required before any source-map
calls. The decision criterion is the Pareto relation among:

- exact question-bound physical mapper calls;
- independent lane/source coverage;
- registered hard-target source coverage;
- route and per-question call tails; and
- the fixed 8,000-token final envelope.

Only the selected sealed population will be sent to the mapper. No gold,
reference answer, prior verdict, or benchmark target registry enters a mapper
or solver prompt.

## Decision

- Retain direct query payload as the protected parent.
- Retain the validated evidence map as a routing and representation layer.
- Reject the isolated V2 map solver as an answer replacement policy.
- Do not run the 549-call source-map upper construction.
- Choose a smaller source-gate policy provider-free, then map and judge the
  conservative source-supported child.
- Make no 95/100 claim on this analysis-used population; a promoted policy
  still requires fresh held-out confirmation and the same-budget Mem0 arm.

## Related records

- [First-principles memory stack audit](../08%20-%20Analysis/15%20-%20First-principles%20memory%20stack%20audit%202026-08-27.md)
- [Query-era matched answer campaign](63%20-%202026-08-27%20-%20Query-era%20matched%20answer%20campaign.md)
- [Matched retrieval mechanism matrix roadmap](49%20-%202026-08-26%20-%20Matched%20retrieval%20mechanism%20matrix%20roadmap.md)
