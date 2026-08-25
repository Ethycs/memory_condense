# Verification relocation — charter and roadmap

**Status**: Charter + roadmap for a dedicated grind session
**Date**: 2026-08-24
**Goal**: Move verification out of the retrieval hot path — to boundaries and
tests — so the codebase's focus can return to modularity and optimization.

## The finding (measured 2026-08-24)

- `eval/` is **74,821 LOC across 137 files — roughly 2× the entire retrieval
  system it measures** (`search` + `associations` + `domain` ≈ 35,500 LOC).
- ~12,500 eval lines touch verification apparatus (receipts, sha256
  identities, seals, attestation, replay cross-checks) — larger than the
  whole `associations` package. Verification density: eval 17%, search 11%,
  vs 3–4% in packages that just do their job.
- The apparatus verifies **bytes, not behavior**. The one regression that
  mattered (the silently swapped retrieval stack, Research Log 21 /
  dev-guide chapter 08) passed every receipt and was caught by a human
  reading scores. A harmless four-line source shift, meanwhile, invalidated
  a 30-minute run. False negative on the real failure, false positive on the
  non-failure.
- The apparatus disqualifies its own results (Research Log 27: a valid
  10-call run ruled protocol-ineligible over reservation bookkeeping) and
  taxes every experiment (any admission-math change must first negotiate
  with identity payloads).
- The ≥95%/100Q gate — the only verification that verifies **answers** —
  has never run, partly because every run drags the full harness.
- Three prior decisions already pushed against this and were trimmed rather
  than resolved: DR-0033 (targeted refactor of line-sensitive hashes),
  DR-0037 (drop exact validation rebuilds), chapter 08's "invariant binding
  over digest worship".

## The rule

**Boundaries seal. Tests assert. The hot path computes.**

Classify every check in the retrieval/eval path into exactly one of:

| Class | Criterion | Destination |
| --- | --- | --- |
| **Boundary** | Seals an input or output of a whole run | Keep — but only two per run: input seal (corpus/store snapshot hash) and output seal (result artifact hash + run-level replay contract) |
| **Test** | Asserts a property of a pure transformation (monotonic nesting, no-duplicate evidence, zero transformer state, policy consistency, artifact/query ownership) | Move to pytest over the transformation; delete the runtime raise |
| **Behavioral invariant** | Would have caught a real regression by its *effect* (recall must not fall below the predecessor stage) | Keep in-path — this is what "recall-guarded" means |
| **Delete** | Per-stage receipt chains, parent-hash lineage, per-call identity cross-checks, callable-hash sensitivity | Remove outright |

Default when uncertain: **Test**, not Boundary. A check earns Boundary status
only by sealing the whole run's input or output.

## Behavior-preservation gate

The refactor's own regression harness is the existing sealed dev10 replay —
one last, correct use of the apparatus being dismantled:

1. Before tranche 1, record the sealed dev10 cumulative run output
   (byte-identical replay already passes).
2. After **every tranche**, re-run: output must be byte-identical and
   `pixi run -e dev pytest -q -m "not slow"` must be green.
3. If replay diverges and the cause is not understood, **stop and report** —
   do not force the hash.

The two boundary seals are simplified **last**, after all interior checks
have moved, so the gate stays trustworthy throughout.

## Sequencing

1. **Map** — walk the cumulative path (`_recall_guarded_cumulative_ops.py`,
   `_recall_guarded_cumulative_contracts.py`, `_recall_guarded_cumulative_result.py`,
   the runners) and every attestation helper they import; classify each check
   per the table. Deliverable: a table (file:line, check, class, destination).
   Read-only; no edits.
2. **Add tests first** — write the pytest assertions for every Test-class
   check *before* deleting anything. Pure gain; tree stays green.
3. **Delete in tranches** — remove runtime checks tranche by tranche, gate
   after each. Small commits, one concern each.
4. **Collapse receipts** — replace per-stage receipt chains with the two
   boundary seals. This changes output bytes by design; re-baseline the gate
   deliberately and document it (this is the one intentional
   behavior-visible change — never silent, per DR-0035's lesson).
5. **Re-measure and cash in** — re-run the LOC/density measurement; then run
   the 100Q gate, which is the entire point: it should now be cheap enough
   to actually execute.

## Collision fence

A concurrent workstream has uncommitted edits in this package. **Do not
touch:**

- `src/memory_condense/eval/consolidation_replay.py`
- `src/memory_condense/eval/fast_cav_feature_session.py`
- `src/memory_condense/eval/recall_guarded_cumulative_fast_artifact.py`
- `src/memory_condense/eval/run_fast_1m_cav.py`
- the 18 untracked `eval/` files (fast Hebbian/CAV-link modules and their tests)

plus any file `git status` shows dirty at session start. If a Map-classified
check lives in a fenced file, record it in the map as *deferred* and move on.
Re-check the fence each session start; land fenced items only after that
workstream commits.

## Standing constraints (unchanged by this charter)

- Never bypass GPG signing; retry on timeout.
- The sealed 200-question v4 confirmation population stays sealed.
- Session snapshots and the Codex transcript are never committed.
- No silent behavior changes: anything behavior-visible gets its own commit
  and a docs note.

## Success criteria

- Tree green; dev10 replay byte-identical through step 3.
- Verification-touching LOC in `eval/` at least halved; no interior
  receipt/identity checks remain in the cumulative hot path.
- Every relocated invariant exists as a named pytest.
- The 100Q gate has actually been run once, end to end.

## Kill criteria

- Replay divergence without an understood cause.
- A relocated check turns out to be load-bearing at runtime (a test cannot
  express it) — reclassify as Behavioral invariant, document why, continue.

---

# Roadmap

Two module families hold nearly all the apparatus (measured 2026-08-24;
percentages are verification-touching lines per module):

- **Cumulative family** (~8,000 LOC): `_recall_guarded_cumulative_ops.py`,
  `_recall_guarded_cumulative_contracts.py` / `_result.py`,
  `_recall_guarded_cumulative_validation_campaign.py` (32%),
  `_recall_guarded_cumulative_validation_shard.py` (22%),
  `_recall_guarded_cumulative_synthesis_artifacts.py` (19%),
  `recall_guarded_cumulative_final_answer.py` (24%) + its semantic judge,
  and the `recall_guarded_cumulative{,_1m,_runtime}.py` runners.
- **Diffuse-replay family** (~8,500 LOC): `_diffuse_replay_contracts.py`
  (32%), `_diffuse_replay_validation.py` (34%),
  `_diffuse_replay_reconstruction.py` (23%),
  `diffuse_longmemeval_route_v2.py` (29%), `diffuse_longmemeval_replay.py`
  (25%), `diffuse_longmemeval_analysis.py` (22%), and the
  `_diffuse_latent_training_corpus_*` trio (20–27%).

The `fast_*` / Hebbian modules are also apparatus-heavy but are **fenced**
(active concurrent workstream); they inherit the pattern in a later pass.

## V0 — Session setup and baseline (first hour)

- Re-check the collision fence (`git status`); update the fence list.
- Run `pixi run -e dev pytest -q -m "not slow"` and record the pass count.
- Run the sealed dev10 cumulative replay; record the output artifact hash —
  this is the byte-identity gate for V2–V3.
- Re-run the LOC/density measurement (script in this doc's history); record
  the numbers as the "before" row.

**Exit**: baseline artifact hash + test count + LOC table written down.

## V1 — The relocation map (read-only)

Walk both families and classify **every** check per the charter table
(Boundary / Test / Behavioral invariant / Delete). Deliverable:
`docs/08 - Analysis/` map doc with one row per check — `file:line`,
what it checks, class, destination (named test / boundary seal / delete),
and *deferred* for anything in a fenced file.

Rough expected split, to be validated by the map itself: a handful of
Boundary rows, ~30–50 Test rows, 1–2 Behavioral invariants (the
recall-not-below-predecessor guard), everything else Delete.

**Exit**: map committed; no source file touched.

## V2 — Tests first (pure gain)

Write the named pytest for every Test-class row **before deleting
anything**: monotonic nesting, no-duplicate evidence, zero transformer
state, policy/artifact/query ownership, packing-cap arithmetic. Property
tests over the pure transformations, not integration re-runs.

**Exit**: tree green with the new tests; replay hash unchanged (nothing
deleted yet).

## V3 — Delete in tranches

One tranche = one commit = one concern. Gate after every tranche:
tests green **and** dev10 replay byte-identical.

- **Tranche A** — cumulative family interior checks: the runtime raises in
  `_recall_guarded_cumulative_ops.py` whose invariants now live in V2 tests.
- **Tranche B** — diffuse-replay family: contracts/validation/reconstruction
  interior cross-checks.
- **Tranche C** — runners and final-answer path: per-call identity
  cross-checks, lineage bookkeeping (the Research-Log-27 class of check).

**Exit**: no interior receipt/identity check remains in either family's hot
path; replay still byte-identical.

## V4 — Receipt collapse (the one visible change)

Replace per-stage receipt chains with the two boundary seals (input
snapshot hash, output artifact hash + run-level replay contract). This
changes output bytes **by design**: re-baseline the replay gate in its own
commit with a docs note — deliberate and documented, never silent
(DR-0035's lesson).

**Exit**: a run emits exactly two seals; replay contract re-baselined and
passing.

## V5 — Re-measure and cash in

- Re-run the LOC/density measurement; write the before/after into the map
  doc. Target: verification-touching LOC in `eval/` at least halved.
- **Run the 100Q gate end to end** — the payoff and the real success
  criterion. Write the result (pass or fail) as a Research Log entry.
- Update this doc's status; fold outcomes into
  `00 - Gap Analysis and Roadmap.md`.

**Exit**: the gate has run once; numbers on paper.

## Order-of-magnitude effort

V0 ≈ an hour. V1 is the largest single sitting (two families, ~16.5k LOC to
walk) — one focused session. V2–V3 are mechanical against the map — one to
two sessions. V4–V5 are small but must not be rushed (V4 is the only
behavior-visible step; V5 spends provider budget on the 100Q gate).
