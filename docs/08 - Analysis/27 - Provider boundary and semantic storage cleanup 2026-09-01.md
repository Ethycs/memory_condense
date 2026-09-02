# Provider boundary and semantic storage cleanup

**Date:** 2026-09-01

**Status:** implemented and locally regression-tested; no provider calls and no
new answer-accuracy claim

## Decision

The active specialist terminal path now has an explicitly versioned compact-v2
provider plane. Stable internal candidate, group, and slot identities remain on
the local audit plane; the provider receives short H, G, K, and S aliases plus
the evidence needed to answer. Historical compact-v1 rendering remains
byte-identical and is still the default for old construction and replay paths.

At the same time, evaluation caches and semantic-tree storage were reduced
without changing semantic-tree projections, receipts, traversal order, or
classifier behavior. This is an apparatus and boundary-hardening result. It
does not change the protected retrieval score or establish progress toward the
95% answer gate.

## Threat and cost boundary

The audit treated four different problems separately:

1. **Gold or locator leakage:** a reference answer, source path, namespace,
   chunk ID, or other local locator reaching a provider call.
2. **Stable internal identity exposure:** a hash is not plaintext gold, but a
   stable candidate/group/slot hash can correlate calls and consumes tokens
   even when an opaque provider handle already exists.
3. **Process retention:** raw corpus strings or large derived structures kept
   alive beyond the query or scoring call that needs them.
4. **Bounded structural amplification:** sealed tree, visit, manifest, and
   audit structures that are correct but repeat an O(N), O(N log N), or O(ND)
   population unnecessarily.

No gold, reference answer, raw filesystem path, raw source ID, chunk ID, or
retained model/KV state was found in the active provider messages inspected.
The source-history mapper's local work record does contain real source and
chunk identities, but its provider renderer replaces them with S/C aliases.
The answer runtime reconstructs and hash-checks the sealed messages before
sending them.

The provider contract still had a narrower real defect. Specialist advisories
included stable 64-hex candidate IDs beside H handles, and typed evidence could
repeat stable slot/group IDs despite having provider aliases. The existing
`assert_gold_blind` guard rejects forbidden key names; it is not a semantic
classifier for values stored under otherwise allowed keys. The compact-v2
schema closes this specific surface rather than overstating what the generic
guard proves.

## Implemented changes

### 1. Explicit compact-v2 provider schema

`ProviderPayloadMode.COMPACT_FINAL` continues to select the historical v1
projection. A new `COMPACT_FINAL_V2` mode owns every provider-visible change.
The additive composer accepts an explicit provider mode and defaults to v1;
only the new specialist construction opts into v2 in this tranche.

Compact v2:

- removes duplicated `available_handle_ids` and `represented_handle_ids` when
  the retained bindings already define the represented set;
- aliases required slot IDs to `S###` and stable hash-shaped group keys to
  `K###`;
- emits repeated item defaults and provenance-by-origin once instead of on
  every row;
- omits default-valued item fields while retaining overrides;
- emits specialist numeric and temporal membership directly as H handles;
- omits provider-visible candidate maps, candidate IDs, and absence slot IDs;
  and
- preserves local item, binding, frontier, candidate, and receipt projections.

The specialist proof compiler accepts both the historical unversioned advisory
and the strict v2 advisory. It reconstructs synthetic candidate and slot IDs
locally from sealed H handles and the advisory receipt. Unknown v2 fields,
candidate fields, raw slot IDs, repeated handles, and handles outside the
terminal evidence set fail closed.

### 2. Query/scoring lifetime cleanup

The transition-trace evaluator previously had three module-global
100,000-entry LRU caches keyed by raw text or `(text, answer)` and a 10,000-entry
dated-question cache. They are now one call-scoped cache owned by
`score_transition_arm`. Reuse remains available within the scoring call, and
the cache is cleared before normal return; an exception also releases it during
stack unwinding because no global reference exists.

The conservative residual classifier no longer retains per-node frozenset
caches for manifest terms and action concepts. Full descent classifies each
node once, so those dictionaries had no reuse. The useful query-local manifest
map and document-frequency preprocessing remain.

`_SealedRecord` now declares `__slots__ = ()`, so its slotted dataclass
subclasses no longer inherit an instance dictionary.

### 3. Shared semantic-tree population

Every semantic node previously retained a tuple slice containing every
descendant cell. A balanced tree therefore retained N(log2 N + 1) cell
references across 2N-1 tuple objects.

Nodes now share one immutable cell tuple and retain a `[span_start, span_end)`
view. Internal residual/global consumers iterate the span without allocating a
slice. The public constructor and `.cells` tuple view, dataclass equality and
hash behavior, copy/deepcopy, pickle restoration, projections, receipts, and
leaf ordering remain compatible. Cached preorder nodes and token totals remain
outside projected identity.

## Measurements

### Provider slots

The same ten sealed specialist inputs were rendered with the historical sealed
system prompt and with the new provider projection. No provider was called.

| Plane | Compact v1 | Compact v2 | Saved |
| --- | ---: | ---: | ---: |
| Provider JSON tokens | 37,643 | 25,687 | 11,956 (31.76%) |
| Complete chat tokens | 40,783 | 28,827 | 11,956 (29.32%) |
| Canonical provider bytes | 140,924 | 104,876 | 36,048 (25.58%) |

Per-question savings range from 337 to 1,799 tokens. The current
resource-preserving prompt adds 190 tokens to both treatments, so the absolute
saving is unchanged.

### Semantic-tree storage

At 1,024 cells and 2,047 nodes:

| Retained structure | Before | After | Change |
| --- | ---: | ---: | ---: |
| Descendant cell-tuple storage | 171,992 B | 8,232 B | -95.2% |
| Nodes + population + preorder cache | 368,544 B | 221,160 B | -40.0% |
| Distinct node cell populations | 2,047 | 1 | -99.95% |

The node object itself grows from 88 to 96 bytes because it retains the shared
population reference and population origin. That small per-node increase is
included in the combined measurement.

### Verification

- Cache/lifecycle slice: 26 passed.
- Semantic storage slice: 45 passed.
- Compact provider/advisory slice: 45 passed.
- Combined integration slice: 101 passed in 29.16 seconds.
- Complete `test_matched_eval_*.py` suite: 878 passed, one skipped, in 443.90
  seconds.
- Downstream construction/replay/locked-reader slice: 83 passed in 56.60
  seconds.
- `git diff --check`: clean.

The complete-suite wall time is not a query-latency result. An isolated timing
attributes 235.11 seconds to the frozen q82 V6.1 resident terminal contract and
39.08 seconds to the exact-span terminal dedup fixture; the remaining thirteen
tests in those two files take roughly eight seconds combined. This agrees with
the earlier finding that authenticated fixture/store construction, not one
semantic search, dominates that test surface.

## Existing artifact deduplication

The current full-100 compact construction already keeps the full resident
audits once in content-addressed namespace sidecars and references them from a
small manifest/checkpoint plane. Historical construction/replay pairs remain
separate sealed byte-identical files because changing them would destroy their
certification contract. This tranche does not rewrite or delete historical
evidence.

## Deliberate deferrals

Two large bounded costs remain:

1. `BranchVisit` and `GlobalTreeVisit` still retain span-wide coverage-ID
   tuples. Replacing the public tuple with an `InitVar` view breaks
   `dataclasses.replace()` compatibility. A compact visit-v2 container should
   store spans against one sealed population while retaining a legacy reader.
2. Node manifests still retain one Python-float centroid per tree node.
   Float32 changes certified bounds and receipts. A float64 shared matrix or
   immutable binary-vector store may preserve values, but it needs an explicit
   manifest-v2 equality, copy, pickle, and projection contract.

The semantic-global full-100 locked path also stays on compact v1. It can opt
into the now-available v2 packet mode only through a new explicitly named
construction successor; silently changing the existing format would invalidate
the historical full-100 lineage.

Finally, the ten specialist questions are a known-miss-conditioned development
assay. Their prompts remain gold-free, but their selection is posthoc and
cannot be reported as an unbiased benchmark population.

## Next engineering step

Use compact v2 in a newly named semantic-global terminal successor, then spend
the recovered prompt budget on a matched evidence-retention ablation rather
than merely reporting smaller prompts. In parallel, prototype visit-v2 and a
float64 manifest-vector store behind golden projection/receipt tests. Neither
change should be merged into a historical sealed format in place.
