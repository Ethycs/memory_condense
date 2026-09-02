# Retrieval apparatus refactor and residual speedup

**Date:** 2026-09-01

**Status:** initial provider-free, behavior-preserving cleanup and sealed-prompt
compatibility repair implemented and measured; no retrieval-accuracy result

## Question

Can the failure-related retrieval code be simplified enough to expose real
performance and architecture problems without changing candidate order,
evidence, prompt bytes, receipts, or answer behavior?

## Decision

Refactor repeated deterministic work at immutable boundaries first. Keep the
specialist mechanisms, budgets, union-before-exclusion rule, terminal packing,
and replay contracts unchanged. Use exact regression tests and a fixed
provider-free benchmark to distinguish retrieval cost from evaluation-fixture
cost.

## Measured result

The fixed benchmark was the same six test files for source gating, semantic
binary search, typed operator adaptation, semantic residual search, semantic
global completion, and source-group reinjection.

| Run | Tests | Wall time |
| --- | ---: | ---: |
| Before refactor | 73 passed | 60.19 s |
| After final reviewed refactor | 74 passed | 23.71 s |

The after run includes one additional regression test. Wall time fell by 36.48
seconds, or 60.6%, on this local slice.

The dominant defect was residual-search manifest access. A property that looked
like a lookup actually rebuilt the complete node-receipt dictionary. The
classifier called it once per visited tree node. Materializing that inventory
once per classifier, and similarly reusing document-frequency and immutable
term-set calculations, reduced representative residual tests from 1.53--5.39
seconds to about 0.13--0.25 seconds.

This is an apparatus speed result, not a new memory score. Candidate ordering,
retained/pruned partitions, packed evidence, projections, and receipts remain
under exact regression coverage.

## Changes

1. **Residual classifier inventory.** The classifier now binds one manifest
   map, immutable query/manifest/cell term sets, and one document-frequency
   table for its lifetime.
2. **Global best-first search.** The search binds one manifest map and caches
   lane bounds by node receipt so heap insertion and visit recording share the
   exact same computed tuple.
3. **Semantic tree metadata.** Frozen nodes cache descendant token counts; the
   frozen tree caches preorder traversal and total tokens. These caches are not
   dataclass fields, constructor parameters, equality inputs, representations,
   or projected identity. Explicit immutable copy/deepcopy behavior and
   constructor-based pickle reconstruction keep the slots valid.
4. **Source-gate sealing.** Projection builds one body and seals that exact
   body instead of rebuilding it through the receipt property. Receipts are
   intentionally resealed on later access so forced mutation cannot pair a new
   body with a stale cached identity.
5. **Typed-packet probing.** The salvage loop memoizes byte-identical
   exploratory populations by sealed item/rejection receipts. The terminal
   hard-cap decision still performs a fresh full render and token count.
6. **Small population walks.** Retained and pruned membership sets are
   materialized once outside their ordered output comprehensions.
7. **Prompt-version compatibility.** A broad regression run exposed a loader
   that always used the legacy typed-answer system prompt even when a sealed v4
   terminal artifact bound the resource-preserving prompt. The loader now
   recognizes two explicitly named immutable prompt versions and accepts only
   the unique rendering whose message SHA and exact token count match the
   sealed terminal. Unknown versions fail closed. Historical artifacts are not
   rewritten.

This compatibility failure is not caused by the cache refactor. An isolated
archive of clean commit `26bbcb7` reproduced the same six cross-plane-authority
failures. All ten historical v2 rows bind only the legacy prompt; all four v4
rows bind only the resource-preserving v2 prompt, whose reconstruction is
exactly 19 tokens longer for each affected row. Both populations remain below
the sealed cap.

## What the profile revealed

The remaining approximately 2.3-second semantic-global tests initially looked
like a search problem. A focused profile separated the phases:

- the complete test call took 2.727 seconds;
- fixture/store construction took 2.571 seconds;
- the actual semantic-global search took about 0.137 seconds; and
- 423 individual SQLite commits accounted for about 1.791 seconds.

The precise timings are local and profiler-sensitive, but the phase attribution
is decisive enough for the next engineering choice. Global search is not the
current bottleneck in that fixture. Batched authenticated store construction is
a better target, and test setup time must be reported separately from query
latency.

## Invariants and verification

Focused tests cover:

- golden semantic-tree projection and receipt hashes;
- unchanged dataclass constructor, equality, hash, and representation surfaces;
- cached preorder/token access and validation of a deliberately corrupted
  internal cache;
- one node-manifest map materialization across multiple classifier calls;
- source-gate projection bytes, one-body sealing, and resealing after forced
  mutation;
- typed-packet payload bytes, order, token count, rejection reason and receipt;
  and
- an uncached authoritative final hard-cap verification after cached probes.

All work is provider-free. It retains no model state and makes no LiteLLM,
responder, selector, or judge calls.

A final 27-file downstream compatibility run passed 310 tests in 298.43 seconds.
It covered the reduced semantic-binary assay; residual construction, candidate,
answer, and judge paths; global terminal adapters, assays, full100 construction,
answer, judge, revalidation, and postseal audit; source-gate adapters and base
policy analysis; and after-union fact closure.

A final focused review passed 78 tests covering cache and retrieval invariants,
both historical and resource-preserving terminal prompts, the cross-plane
authority guard, and the typed final arm. Tampered message identity and prompt
token type are rejected before use.

The final complete matched-eval pass covered all 89
`tests/test_matched_eval_*.py` files: 876 passed, one skipped, and none failed
in 346.19 seconds.

## Next seams

The cleanup points to five distinct follow-ups rather than another retrieval
layer:

1. lift immutable cell, manifest, and normalized-feature inventories from one
   query to the lifetime of the common memory index used across prompt ticks;
2. batch fixture and ingest transactions while preserving store and receipt
   boundaries;
3. precompute source-gate candidate and normalized fact/obligation views; and
4. add the gold-blind question-by-method outcome ledger from
   [Analysis 26](../08%20-%20Analysis/26%20-%20Method%20eligibility%20failure%20attribution%20and%20apparatus%20cleanup%202026-09-01.md),
   then spend recovered time only on applicable unresolved frontiers; and
5. include an explicit renderer-version field in newly sealed terminal prompts,
   while retaining hash-bound reconstruction for historical artifacts.

Only after those measurements should exact packer consolidation, source-history
hydration changes, or wider semantic recursion be considered. JAX or Numba has
no measured target here; repeated mappings and transaction boundaries dominate
the observed costs.

## Claim boundary

This result establishes a provider-free implementation speedup on a focused
test slice and identifies a separate construction-I/O bottleneck. It does not
change the protected 89/100 development result, establish a 95% score, validate
confirmation200, or complete the same-budget Mem0 comparison.
