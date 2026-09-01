# Resumable namespace construction successor

Date: 2026-08-30

## Outcome

`tools/run_locked_semantic_global_terminal_full100_resumable.py` is an opt-in,
provider-free successor to the resident full100 construction runner. It does
not change the existing `construct` or `replay` APIs, and its final manifest,
namespace sidecars, and replay use the existing formats and filenames.

The successor was implemented and tested only under temporary output roots.
It did not inspect, signal, or interrupt PID 54232, did not run the expensive
real corpus, and cannot write to the legacy default output root.

## Safe V7 seam

The existing V7 builder already groups its internally supplied workset by
namespace and opens each namespace independently. The successor therefore does
not clone retrieval or terminal-compilation logic. It:

1. Authenticates the gate, R7 construction, vectors/replay, and V3 parent using
   the existing source reader.
2. Derives all 68 eligible ordinals from that gate, then groups them by the
   authenticated namespace IDs. There is no ordinal CLI.
3. Seals one preflight containing the exact source, policy, output-root, and
   namespace-population bindings.
4. Calls the existing V7 builder once for each namespace and immediately
   publishes one deterministic namespace checkpoint.
5. On resume, accepts only canonical checkpoint JSON with a valid digest
   sidecar, checkpoint receipt, matching preflight/source/policy/population,
   exact namespace questions, and zero-provider/zero-state attestations.
6. Fails closed on a partial pair, byte tamper, foreign checkpoint entry, or
   semantic/binding drift. It never rebuilds or silently replaces such a row.
7. Merges complete checkpoints in canonical namespace and ordinal order, then
   calls the existing `_compose_payload` and existing sidecar projection.

The resulting construction is therefore byte-identical to the resident build,
while a crash loses at most the currently executing namespace.

## Commands

An explicit nonlegacy output root is mandatory.

```powershell
.pixi\envs\dev\python.exe tools\run_locked_semantic_global_terminal_full100_resumable.py construct `
  --output-root <NONLEGACY_OUTPUT_ROOT> `
  <THE SAME AUTHENTICATED SOURCE AND STORE ARGUMENTS AS THE RESIDENT RUN>

.pixi\envs\dev\python.exe tools\run_locked_semantic_global_terminal_full100_resumable.py replay `
  --output-root <NONLEGACY_OUTPUT_ROOT> `
  --expected-construction-output-sha256 <CONSTRUCTION_SHA> `
  <THE SAME AUTHENTICATED SOURCE AND STORE ARGUMENTS>
```

`construct` may be invoked again after a crash. Complete checkpoints are
verified and skipped; missing namespaces are built. `replay` never invokes V7
and requires the entire checkpoint population plus the published construction
and sidecars.

## Importing a completed legacy construction

A completed resident/legacy v1 construction can seed the successor without a
new resident database scan:

```powershell
.pixi\envs\dev\python.exe tools\run_locked_semantic_global_terminal_full100_resumable.py import-legacy `
  --output-root <DISTINCT_NONLEGACY_SUCCESSOR_ROOT> `
  --legacy-root <COMPLETED_LEGACY_ROOT> `
  --expected-legacy-construction-sha256 <EXACT_LEGACY_CONSTRUCTION_SHA> `
  <THE SAME AUTHENTICATED GATE, R7, VECTOR/REPLAY, AND V3 SOURCE ARGUMENTS>
```

The import does not read, require, or fabricate a legacy replay. Before its
first successor write, it uses the resident construction validator to bind the
exact expected legacy construction and all manifest-referenced,
content-addressed namespace sidecars to the sealed gate, R7 construction,
vectors/replay, V3 parent, and terminal policies. Namespace populations are
derived internally from those authenticated sources; the CLI exposes no
ordinal selector.

For each gate-derived namespace, the importer reconstructs the exact V7 subset
envelope from the authenticated legacy sidecar and passes it through the normal
successor namespace-execution validator. It then wraps that validated payload
in the ordinary preflight-bound checkpoint format. The merged checkpoints must
reproduce the legacy manifest byte-for-byte before publication is allowed.
The importer publishes the preflight, checkpoints, namespace sidecars, and
construction write-once into the distinct successor root. A later ordinary
`replay` validates and assembles them without opening V7 or the resident store.

This is a deliberate version boundary: `construct`/`replay` consume successor
checkpoints, while `import-legacy` is the sole conversion path from a completed
legacy v1 construction. It is safe only after the legacy construction and all
referenced sidecars exist and the caller supplies its exact construction SHA.
Partial or tampered legacy input is rejected before the successor root is
created. Partial, foreign, symlinked, or conflicting pre-existing successor
state is also rejected; matching sealed state is verified and reused.

## Verification

```powershell
.pixi\envs\dev\python.exe -m pytest -q `
  tests\test_run_locked_semantic_global_terminal_full100_construction.py `
  -p no:cacheprovider
```

Observed: 20/20 passed. The added tests prove:

- a synthetic crash after two namespace seals resumes from those two exact
  checkpoints;
- the resumed manifest and sidecars are byte-identical to the resident fixture;
- replay assembles from checkpoints without calling V7;
- byte-tampered and partial checkpoints are refused rather than skipped;
- none of the commands exposes an ordinal selector; and
- an omitted output root or the legacy default root is rejected before source
  construction begins;
- a completed legacy manifest plus sidecars imports without a legacy replay or
  any V7/store invocation, and normal replay reproduces the exact construction;
- a second identical import reuses the authenticated write-once checkpoints;
- tampering any referenced legacy sidecar is detected before the successor
  root is created; and
- foreign pre-existing successor state is refused rather than overwritten.

## Production results: v1 import and compact-v2 replay

The completed resident construction was subsequently imported and replayed in
production. This updates the earlier temporary-root-only status above; it does
not change the construction semantics or establish an answer-accuracy result.

The authenticated legacy input was:

```text
legacy root:
  eval_results/matched_eval_100/locked-semantic-global-terminal-full100-v1
legacy construction SHA-256:
  7fe63e3890936feebf239dc4f16541a1336306d55570a6d78010aefc0e7b9278
```

The first production successor import used the v1 root
`eval_results/matched_eval_100/locked-semantic-global-terminal-full100-resumable-v1`.
Its preflight SHA-256 was
`02119bf1a4a635676287891db354fa0fe298f1d364a7478aa0bba90f6146df22`.
It completed in approximately 50 minutes 40 seconds, with sampled working-set
memory between 5 and 11 GB. The format proved byte equivalence, but it stored
2.288 GiB of namespace sidecars and embedded another 2.288 GiB of checkpoint
payloads. That duplication and the importer's memory profile make v1 an
authenticated compatibility result rather than the preferred operational
layout.

The compact-v2 successor then imported the same legacy construction into:

```text
eval_results/matched_eval_100/
locked-semantic-global-terminal-full100-compact-resumable-v2
```

Its externally pinned import-attestation SHA-256 is
`c7ce5b79862e46194b2fc1c7c20291ce8926d056f61d2f7eae0331fc0f85682e`.
The import preserved the exact legacy construction SHA, produced ten compact
namespace checkpoints, and accounted for exactly 2,457,003,621 sidecar bytes.
The initial deep-authentication import took approximately 12 minutes 55
seconds; sampled working-set memory stayed between 0.87 and 0.99 GB. The whole
root was approximately 2.299 GiB, while the checkpoint payloads were only
about 19 KiB. The pinned replay then reproduced construction and replay SHA
`7fe63e3890936feebf239dc4f16541a1336306d55570a6d78010aefc0e7b9278`
byte-for-byte in 20.281 seconds, with zero provider calls and zero retained
transformer token state.

Compact v2 adds an exclusive lifecycle lock and requires the exact external
attestation pin for reuse and replay. Publication stages use exclusive-create
and no-follow behavior; stranded staging is validated rather than silently
trusted, and hardlinked staging files are rejected. Small checkpoints are
validated before multi-gigabyte sidecar scans. The attestation binds the
explicit canonical output root, reserved output-root basenames are rejected,
and Windows reparse points, junctions, symlinks, and redirected ancestors are
refused.

Final verification passed 20/20 focused compact-v2 tests and 39/39 tests in the
complete construction test file. An independent review returned GO. The
remaining P2 limitations are operational rather than construction-integrity
failures:

- a crash before the first attestation publication repeats deep source
  authentication;
- peak memory still includes one decoded 170--280 MiB namespace sidecar;
- resume redundantly hashes both source and target sidecars; and
- the lifecycle does not claim directory-fsync durability across whole-machine
  power loss.

This is an apparatus result: it establishes authenticated import, compact
checkpointing, and fast provider-free replay of the existing construction. It
does not measure QA accuracy, pass the >=95% gate, or supply a judge result.
Research Log 94 records the production measurement.
