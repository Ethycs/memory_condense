# 08 — 1M Test Execution and Regression Accountability

**Phase:** 08 (merged turns 393-430, 2026-08-21 to 2026-08-22)
**Previous:** [07 — Diffuse retrieval buildout](07-diffuse-retrieval-buildout.md)
**Next:** [09 — Acceleration scoring and ladder restoration](09-acceleration-scoring-and-ladder-restoration.md)

## Purpose

This chapter documents how the 1M-token memory test actually gets run — what
evidence standard applies, which blockers are waived and which are not — and
the regression-accountability contract that came out of running it: a
recall-guarded cumulative route in which no new retrieval method may ever
discard a predecessor's admitted evidence, measured through a phased,
checkpointed 1M harness.

The phase's hinge is a real regression. The first fresh 1M run silently swapped
the frozen v3 `causal_graph` stack for the new `episode_primary` route and
dropped literal recall from 6/10 to 3/10 on the same shard while spending
2.37x the context. DR-0035 re-locks the original linear-cumulative process so
that architecture swaps of this kind are structurally impossible.

## The run and what it exposed

Entering this phase, episodic retrieval passed at the implementation/replay
level (206 tests passed), but two things had never happened: the production
corpus launcher had never executed (it fails closed on path-handoff identity
concerns), and no fresh end-to-end 1M retrieval measurement existed for the
new episodic stack.

The sequence of events that defines the phase:

1. A controlled episodic-memory canary (ingest, close/reopen, query, verify
   evidence) passes with checkpoint hashes recorded before and after.
2. The corpus and checkpoint attestation blockers are waived for this
   environment (DR-0034) and the 1M test runs.
3. The fresh run completes mechanically (functional pass) but fails on
   quality: 3/10 literal answers, 0/10 gold evidence sources, all 10 closures
   incomplete (8 `workspace_cap`, 2 `conflicted`), in 50m 33s.
4. Diagnosis shows the run was not the requested recreation: the retrieval
   treatment had been changed from frozen v3 `causal_graph` (direct chunks +
   causal expansion + coverage selection) to `hybrid_graph` → 8 representative
   episode seeds → bounded closure, with no direct-chunk fallback. A
   same-shard control confirms frozen v3 recovers 93.3% of labeled sources on
   the identical ten validation questions.
5. The linear-cumulative development contract is re-locked (DR-0035) and the
   recall-guarded cumulative route plus its 1M harness are built.

The store itself was never the problem: all 18 labeled gold sources survived
ingestion with intact chunk, episode, unit, and relation provenance. The
regression lived entirely in retrieval-layer selection authority.

## Design

### Evidence standard: controlled functional run (DR-0034)

The 1M test runs under an explicit two-tier evidence standard:

- **Controlled functional run** — the operative tier. Single-user machine,
  static local checkpoint directories, no concurrent writer. Checkpoint
  hashes for BGE-M3 and the Qwen prefix runtime are recorded before and after
  the run as declared coordinates, and results are labeled "controlled
  functional," never "hostile-environment attested."
- **Production attestation** — out of scope. The path-race concerns (a
  same-user process replacing checkpoint files between verification and load;
  the launcher's plain-pathname handoffs between stages; create/register and
  rename/register crash windows) are real only against a hostile local actor.
  They are scientific-attestation blockers, not evidence that retrieval is
  broken, and they do not gate functional measurement.

Corollary: cache-identity ceremony is also dropped from the functional path.
When a cached store fails an exact-identity check against today's composed
shard, the answer is to rebuild, not to spend the run reconciling the cache.

### Fresh-rebuild protocol for the 1M population

A 1M measurement that claims to recreate the original concatenated-memory
test must rebuild its store from the original locked validation histories —
no reused database, no cloned cache directory. The valid fresh population is
1,041,276 tokens / 5,551 turns / 8,122 chunks, persisted and reopened before
querying. Failed cache-shortcut attempts are disposable directories, removed
after the active run releases the database.

The completed run's receipt is retained at
`eval_results/longmemeval-1m-episode-primary-controlled-20260821/result.json`
as the `episode_primary` ablation arm — a valid experiment, but not the
recreation, and not a baseline.

### The cumulative development contract (DR-0035)

The re-locked process is:

```text
baseline
→ add one retrieval method
→ preserve all prior evidence
→ test on identical corpus/questions/budget
→ keep it only if it improves
```

Everything the harness enforces follows from what the regression showed was
missing:

- One fixed, reproducible 1M baseline shared by every arm.
- A cumulative retrieval ladder, not unrelated modes compared across
  different populations.
- A monotonic fallback: a new method structurally cannot discard the previous
  winner's evidence.
- Per-stage candidate, source, episode, closure, and packet traces, so an
  all-zero score can always be localized to its first failing gate.
- Same-budget automated A/B scoring and regression gates.

### The recall-guarded composite route

The "ultimate" route is a recall-preserving composite, not a winner-take-all
stack choice:

| Stage | Authority | Behavior |
| ----- | --------- | -------- |
| 1. Frozen v3 `causal_graph` | Authoritative | Rendered excerpts frozen byte-for-byte into the final context |
| 2. Representative episodes | Additive | Spend only remaining prompt budget on breadth |
| 3. Artifact-wide closure | Additive | Recover misses beyond the episode workspace |
| 4. Packet assembly | Audit | Per-stage provenance in the receipt |

The monotonicity contract is enforced at return time: the composite route
only returns successfully if the predecessor evidence is retained, and the
receipt makes the check auditable. If an additive stage cannot fit in budget,
it becomes a no-op rather than evicting earlier evidence — the exact
regression mechanism observed in the `episode_primary` run is prohibited by
construction. Adversarial review hardened the contract further: the receipt
rejects predecessor reordering (not just omission), and immutability of
protected messages is deep, not shallow. Existing routes (`causal_graph`,
`episode_primary`) remain unchanged as controls.

Supporting fix: the causal replay seam that regenerated turn IDs and
timestamps is repaired, and causal caches are bumped to revision 4 so a fresh
1M reconstruction cannot silently reuse incompatible v3 staging bytes.

### The 1M cumulative harness

The runnable harness is `tools/run_recall_guarded_cumulative_1m.py`, backed
by `src/memory_condense/eval/recall_guarded_cumulative_1m.py`. A small
integration test is never presented as "the 1M test"; only a retained
per-stage result artifact over the full population counts.

Key harness properties:

- **Phased checkpoints.** `--phase source` builds and publishes an exact-span
  deterministic source store as a verified checkpoint (1,039,203 tokens /
  5,400 turns, receipt `92c764d7…a5bb45`), so a failure in the later combined
  causal-plus-discourse build never repeats the million-token embedding work.
- **Invariant binding, not digest worship.** Because loader identity
  semantics changed, the runner binds the facts that define the experiment —
  exact dataset/split hashes, the ordered ten question IDs, token and turn
  counts, and presence of all 25 source labels — and keeps the historical
  compiled-store digest only as provenance for the copied store.
- **Exact-evidence validation.** Archived chunk offsets that fail the current
  exact-span validator disqualify a legacy store from the comparison; the
  atomic builder fails safely without publishing a combined store, and the
  population is rebuilt with exact spans instead of weakening the check.

At end of phase the source checkpoint is published and the combined
causal-plus-discourse build is running as a verified long-running CPU build.
The measured cumulative result does not yet exist; diagnosing that run's
speed opens [chapter 09](09-acceleration-scoring-and-ladder-restoration.md).

## Why this shape

- **Measurement beats attestation on a trusted machine.** The checkpoint and
  corpus blockers defended against a hostile same-user process that does not
  exist in this environment. Holding a functional retrieval measurement
  hostage to that threat model produced weeks of apparatus and zero 1M
  results; the waiver converts the concern into labeling ("controlled
  functional") rather than a gate.
- **Monotonic evidence retention is the only structural defense against
  silent architecture swaps.** The regression happened because a new route
  was allowed to remove direct chunks, causal expansion, and coverage
  selection while claiming the same test name. A byte-frozen predecessor
  prefix plus a return-time retention check makes the failure class
  unrepresentable instead of merely discouraged.
- **Per-stage traces are what make regressions attributable.** The first run's
  compact artifact discarded expected and retrieved source IDs, so the 0%
  source-recall figure could not be separated from a namespaced-label scoring
  mismatch. The harness therefore treats stage traces and receipts as part of
  the result, not optional debugging output.

## Why not X

### Why not waiting for attestation-grade identity safety ([DR-0034](../decisions/0034-waive-corpus-checkpoint-blockers.md))

The checkpoint concern (hash path → load path → rehash path leaves a
swap window) and the corpus launcher's fail-closed path handoffs are genuine
gaps for cryptographic execution-attestation. But in a single-user controlled
environment the practical risk is low, and the stronger standard answers a
question nobody was asking yet ("were these exact bytes executed?") while
blocking the question that mattered ("does the memory system work at 1M?").
The gate survives as labeling: Qwen execution-attestation fields remain
false, and results are marked controlled-functional.

### Why not reusing the cached 1M store ([DR-0034](../decisions/0034-waive-corpus-checkpoint-blockers.md))

Two cache-clone attempts were discarded. The cached store matched the
5,551-turn count but not the composed shard's identity, and — decisively — a
cached store is not a recreation of the original concatenated-memory test.
The waiver covers attestation ceremony, not treatment fidelity: the store is
always rebuilt fresh from the original locked histories.

### Why not the episode-only replacement stack ([DR-0035](../decisions/0035-relock-linear-cumulative-design.md))

`episode_primary` replaced direct chunks, causal expansion, and the coverage
selector with eight representative episode seeds and bounded closure. On the
identical shard and questions, frozen v3 scored 6/10 literal and 93.3% source
recall; the replacement scored 3/10 literal with over twice the packet
tokens. Episode representatives survive — but demoted to an additive breadth
stage behind the byte-frozen v3 prefix, never as the sole retrieval
authority.

## Open questions

- The 0% gold-source recall figure from the `episode_primary` run is marked
  needs-verification, not settled: the compact artifact discarded source IDs,
  so a namespaced-label scoring mismatch cannot be excluded (the 3/10 vs 6/10
  literal drop is confirmed real).
- The measured recall-guarded cumulative 1M result did not exist at phase
  close; the combined causal-plus-discourse build was still running, and its
  slowness is the opening problem of chapter 09.
- The archived base store's `created_at` column carries 2026 ingestion times
  rather than LongMemEval session dates; whether the fairest comparison base
  is the byte-identical original store or a timestamp-correct rebuild is
  labeled per-run rather than resolved.
- Dispatch inconsistencies between ordinary and diffuse retrieval (flagged in
  the DR-0035 gap list) remain open.
- Held-out semantic answer quality is still uncertified; the phase measures
  retrieval reachability only.

## Source turns

Raw transcript for this phase:
[phase-08-1m-test-execution-and-regression](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/00-overview.md)

Key moments:

- Readiness check and checkpoint-identity discussion:
  [turn-2171-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2171-user.md),
  [turn-2175-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2175-assistant.md),
  [turn-2177-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2177-assistant.md),
  [turn-2179-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2179-assistant.md),
  [turn-2181-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2181-assistant.md)
- DR-0034 waiver — corpus blocker dismissed, controlled environment:
  [turn-2189-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2189-user.md),
  [turn-2190-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2190-assistant.md),
  [turn-2191-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2191-user.md),
  [turn-2192-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2192-user.md),
  [turn-2193-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2193-assistant.md)
- DR-0034 waiver — cache ceremony dropped, test forced to run:
  [turn-2195-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2195-user.md),
  [turn-2207-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2207-assistant.md),
  [turn-2208-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2208-user.md),
  [turn-2209-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2209-assistant.md)
- Fresh-rebuild correction and completed 1M run:
  [turn-2212-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2212-user.md),
  [turn-2213-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2213-assistant.md),
  [turn-2268-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2268-assistant.md),
  [turn-2271-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2271-assistant.md)
- Regression diagnosis — stack swap identified and quantified:
  [turn-2277-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2277-user.md),
  [turn-2278-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2278-assistant.md),
  [turn-2281-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2281-user.md),
  [turn-2282-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2282-assistant.md),
  [turn-2283-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2283-assistant.md),
  [turn-2284-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2284-assistant.md),
  [turn-2286-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2286-assistant.md)
- DR-0035 re-lock — cumulative contract restated and accepted:
  [turn-2287-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2287-user.md),
  [turn-2288-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2288-assistant.md),
  [turn-2289-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2289-user.md),
  [turn-2290-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2290-assistant.md),
  [turn-2291-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2291-user.md),
  [turn-2292-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2292-assistant.md)
- Composite route build and hardening:
  [turn-2293-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2293-assistant.md),
  [turn-2295-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2295-assistant.md),
  [turn-2296-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2296-assistant.md),
  [turn-2300-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2300-assistant.md)
- 1M cumulative harness build and phased checkpoints:
  [turn-2312-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2312-user.md),
  [turn-2313-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2313-assistant.md),
  [turn-2318-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2318-assistant.md),
  [turn-2320-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2320-assistant.md),
  [turn-2333-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2333-assistant.md),
  [turn-2334-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2334-assistant.md),
  [turn-2339-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2339-assistant.md),
  [turn-2345-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2345-assistant.md),
  [turn-2390-assistant.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2390-assistant.md)
