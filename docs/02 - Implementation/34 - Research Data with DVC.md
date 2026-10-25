# Research data with DVC

**Status:** Selected research archive published and restore-tested; Windows access gaps recorded

**Date:** 2026-10-24

**Scope:** Source text, summaries, provenance, and reusable retrieval artifacts

DVC stores the data bytes in its content-addressed cache. Git stores the small
`.dvc` pointers, `.dvc/config`, `.dvcignore`, and this guide. The application and
release packages do not require DVC. This setup uses the separately installed
DVC 3.67.1 CLI with `dvc-s3`; it does not add dependencies to the model runtime.

## Initial coverage

| DVC pointer | Contents |
| --- | --- |
| `data.dvc` | LoCoMo input, LongMemEval oracle and LongMemEval-S inputs, archived engineering transcript, build-session turn export and JSONL snapshot |
| `eval_results/longmemeval-s-native-release-sol-20261014-r27.dvc` | Completed 500-question baseline: plan, reports, saved samples, answers, judgments and integrity sidecars |
| `eval_results/chronos-time-packet-audit-20261016-r1.dvc` | Saved packet metadata audit |
| `eval_results/chronos-time-accuracy-20261016-r2.dvc` | Completed export-anchor diagnostic, including its control rechecks |
| `eval_results/chronos-source-time-accuracy-20261016-r1.dvc` | Corrected source-date packets, answers, judgments, control rechecks and completion audit |

The LongMemEval-S dataset is available at `data/longmemeval_s_cleaned.json`.
Its SHA-256 is
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
The existing source in the Downloads directory was preserved.

The initial five snapshots contain **25,867 files / 2,494,098,705 bytes**
(about 2.49 GB before cache deduplication). The baseline snapshot includes all
500 saved answers and 500 judgments. These counts exclude the caches below.

This is an explicit first snapshot, not a claim that the entire research archive
has been migrated. `.dvcignore` admits these four completed runs and excludes the
other evaluation directories from traversal. It also excludes downloaded model
weights, temporary tests, mutable stores directly under `data/`, and the baseline's
compiler and transport caches. The baseline snapshot does contain the completed
250 sample stores created by r27, including their `memory.db` source turns and
`native-spine-live-v1.sqlite` summary indexes. Its first 250 answers were adopted
from r26 and, transitively, earlier runs; their originating stores are outside
this initial snapshot. This does **not** yet
preserve the complete originating-store and summary-cache population.
Small task definitions in `evals/` and project
model heads already tracked by Git remain in Git.

Existing `.sha256` sidecars remain authoritative for the project's sealed-artifact
checks. DVC adds storage/version tracking; it does not replace those checks.
Sealed plans may contain original machine paths. They are preserved byte for
byte, so restoring the data is not a promise that every historical command is
portable without choosing the appropriate input path.

## Inspect and restore

Run these commands from the repository root:

```powershell
dvc status
dvc remote list

# After the first remote publication, restore just the benchmark inputs:
dvc pull data.dvc

# Restore all tracked snapshots:
dvc pull
```

The raw payloads stay Git-ignored. `eval_results/*.dvc` is the sole exception
under the evaluation directory so Git can version the pointers. The exclusion
uses `/eval_results/?*`, requiring a child character: on this Windows setup,
`/eval_results/*` also matched DVC's trailing-slash check of `eval_results/`,
which silently hid the four evaluation stages from default push/pull discovery.
`dvc dag --outs` must show the five initial snapshots plus `research_data`.

## Original text and generated summaries

The original LongMemEval histories are in `data/longmemeval_s_cleaned.json`.
Each built memory store keeps exact source turns in `memory.db` (`turns.text`)
and its atomic/hierarchical/parent summary sections in
`native-spine-live-v1.sqlite`. Reusable compiler summaries also exist under
`compiler/cache/atoms` and `compiler/cache/merges`; their entries include source
text hashes and support evidence. Attention and vector caches are separate.

Following the saved adoption receipts found **all 500 source/summary store
pairs across 13 originating runs**. The complete frozen store set is in
`research_data/longmemeval-s-500-stores-20261021/`, tracked by `research_data.dvc`:

- `catalog.json` maps each question through its adoption history to its source
  store and archive.
- Thirteen `.tar.zst` archives preserve the original relative paths of all
  **1,500 database/index files / 4,124,027,776 bytes**. They compress to
  **1,528,611,413 bytes**, excluding JSON receipts.
- Each archive has a JSON manifest with every member's original path, size,
  timestamp and SHA-256, plus the archive's SHA-256. Every member was read back
  from its archive and matched against its original hash.

The frozen stores include `memory.db`, `native-spine-live-v1.sqlite`, and
`hnsw_index.bin`. Ephemeral `-shm` and empty `-wal` files are omitted; the
archiver refuses a nonempty WAL. Original stores remain in place.

After pulling `research_data.dvc`, extract a bundle into an empty restore
directory using `tar -xf BUNDLE.tar.zst -C RESTORE_DIRECTORY`. Windows tar
3.8.9 successfully read this format. Member paths recreate the original
`eval_results/<run>/samples/<question>/store/memory/` layout.

This complete 500-store set still does not cover the entire larger research
archive. The requested remaining scope is source text, summaries, provenance,
and reusable retrieval artifacts; temporary runs and debug artifacts are
outside it. The broader publication and its remaining access gaps are documented below.

The 2026-10-21 inventory measured **3,242,246 readable files /
125,221,388,344 logical bytes** across `data/`, `_ingest/`, and 1,006 evaluation
directories. This is a pre-exclusion inventory, not an uploaded-byte count or
deduplicated disk usage. It includes installed environments and retired
worktrees. Access-denied paths are recorded separately; their unknown contents
are not included in that total. Reusable `.cache/experiments` artifacts are
included in the broader archive where accessible.

The complete 500-store publication is lakeFS commit
`82d974caec273e113dd864f9da352c003540e9225fbcce3766a81c103e005dab`, associated
with Git commit `fc25ea0d3cccc589346ddc13747218b8dc7a1221`. Together with the
initial snapshots it contains **25,102 unique remote objects /
4,026,009,800 bytes**, including DVC directory manifests. The branch was
checked for missing objects, size mismatches, and uncommitted changes.

A fresh empty-cache DVC consumer restored an individual three-question bundle
from that immutable commit. Its archive and all **nine source/summary/index
members / 26,067,232 uncompressed bytes** matched their SHA-256 receipts.
Individual bundles can be pulled without restoring the entire research output:

```powershell
dvc pull research_data/longmemeval-s-500-stores-20261021/longmemeval-s-native-release-sol-20261010-r10.tar.zst
dvc pull research_data/longmemeval-s-500-stores-20261021/longmemeval-s-native-release-sol-20261010-r10.tar.zst.json
```

## Reusable summary handoff

`research_data/summary-handoff-20261024/` shares the broader archive's
`research_data.dvc` pointer and publication. It contains a roughly **38.35 MB**
consumer package, so a recipient can inspect and reuse summaries without
downloading and extracting the entire archive.

- `HANDOFF.json` records restore instructions, source/summary joins, provenance
  limits, and the complete archive's dependencies and access gaps.
- `summary-artifacts.jsonl.zst` maps matching artifacts to their original paths,
  bundles, sizes and SHA-256 receipts. `inventory.json` gives counts by source
  root. It indexes **253,366 atomic JSON caches and 13,843 merge JSON caches**,
  excluding checksum sidecars, plus summary databases and candidate provenance.
  The respective distinct file-content counts are **229,673 and 13,326**;
  neither count measures independent source conversations or gold labels.
- `source-families.json` identifies all 158 additional source stores by path and
  bundle. Its family labels are hints; shared-session and text-overlap checks
  are still required before assigning training splits.
- `read_summary_pairs.py` reads current per-section and older snapshot schemas,
  checks exact source-span hashes and attribution, and preserves child-summary
  references. `schemas.json`, `verified-examples.json`, and
  `join-verification.json` document the checks and limitations.
- `engineering-candidates.jsonl.zst` is a **5.14 MB** convenience shard with
  **4,542 records from 25 engineering stores**: 3,315 raw-summary candidates,
  601 hierarchy-merge candidates, and 626 structural-summary candidates.
  Every exported source span passed its SHA-256 check, and source and summary
  database files matched the frozen archive. Its coverage receipt is
  `engineering-shard-verification.json`.

Repeated runs and shared sources remain identified. These are unreviewed
existing targets, with no new model calls, human labels, or assigned splits.
Children reconstructed from a hierarchy are not proof of the exact original
model prompt; use saved generation receipts to recover prompts and teacher
identity. Copied, inline, concatenated, generated, and rejected outputs must
remain distinguishable. The recipient reported that earlier NCM models had
already trained on a subset of the LongMemEval-derived export, so that corpus
is not an untouched benchmark for those models. New training should exclude
the 500-store collection and overlapping sessions when preserving that
evaluation boundary.

```powershell
dvc pull research_data/summary-handoff-20261024
# Or retrieve only the small engineering shard and its coverage receipt:
dvc pull research_data/summary-handoff-20261024/engineering-candidates.jsonl.zst research_data/summary-handoff-20261024/engineering-shard-verification.json
```

## Broader archive publication

`research_data/reusable-research-20261021/`, tracked by `research_data.dvc`,
preserves **2,357,622 additional files /
79,421,060,633 uncompressed bytes** in verified bundles totaling
**28,571,676,558 compressed bytes**, excluding their manifests.
These are additional to the existing snapshots and complete 500-store archive;
their contents are reused through the catalog's explicit dependencies.

The publication is lakeFS commit `ded03c25a12e56d620710d2bff334ba5beed923584783c460084eb5922a8270a`,
associated with Git commit `1fbb00c3602c2d7fb22baf363030481df23b2a7e`. No original data was
deleted or changed. Temporary/debug directories, runtime transport traces,
installed environments and retired worktrees were excluded.

`catalog.json` maps original roots to bundles and records exclusions and
dependencies. `coverage.json` summarizes the data categories and shared-cache
links. Each bundle contains `data.tar.zst`, `members.jsonl.zst`, and
`receipt.json`; the receipt carries archive/manifest hashes and the compressed
manifest records every member's original path, size, timestamp and SHA-256.
All included members were read back and verified before publication.

**Coverage limitation:** Windows denied access to
**804 paths across 80 roots**.
The exact paths are retained in `access-omissions.json` and per-bundle receipts.
Accessible files from those roots were preserved. An included-byte integrity
pass does not imply those missing paths were backed up; their permissions were
not changed. This publication therefore does not claim the entire local archive
is recoverable. Reparse links are recorded rather than followed or recreated;
consult `coverage.json` for the target bundle and any target access gaps.

A new empty-cache consumer pulled a representative bundle from the immutable
lakeFS commit and verified **4,142 members /
33,403,462 uncompressed bytes**. It includes original memory,
summary indexes, atomic/hierarchy summary caches, vectors and attention caches.
The same empty-cache consumer restored the complete summary handoff and verified
all **12 manifest-listed files / 38,349,085 bytes** against their SHA-256 receipts.

```powershell
dvc pull research_data/reusable-research-20261021/catalog.json
dvc pull research_data/reusable-research-20261021/coverage.json
dvc pull research_data/reusable-research-20261021/access-omissions.json
# Pull just one bundle after selecting its path in the catalog:
dvc pull research_data/reusable-research-20261021/bundles/eval_results__longmemeval-s-native-release-sol-20261010-r7
```

Extract `data.tar.zst` into a separate empty restore directory using `tar -xf`.
Its member paths reconstruct the original repository-relative layout. A bundle
is an archival snapshot; the application continues to use its original stores.

## Shared storage and machine setup

Follow the infrastructure [project-data guide](https://central-dev.zt:9000/#operations/project-data.md).
The credential-free remote is configured in `.dvc/config`:

| Setting | Value |
| --- | --- |
| DVC remote | `lakefs` (default) |
| DVC URL | `s3://memory-condense/main/` |
| Endpoint | `https://central-dev.zt:8000` |
| lakeFS repository | `memory-condense` |
| Storage namespace | `s3://lakefs-storage/memory-condense` |

The first URL component is a lakeFS repository, not the underlying MinIO
bucket. Check that the repository exists before the first push; if absent,
create it once using authenticated `lakectl`:

```sh
lakectl repo create lakefs://memory-condense s3://lakefs-storage/memory-condense
```

Install DVC separately from the application environment on each client:

```powershell
pixi global install 'dvc=3.67.1' --with dvc-s3
dvc remote modify --local lakefs ssl_verify 'PATH_TO_TRUSTED_CADDY_ROOT_PEM'
```

Obtain the Caddy root from the infrastructure checkout or export the already
trusted certificate on the machine. Do not disable certificate verification.
This Windows checkout uses `.dvc/tmp/caddy-root.crt`, exported from its trusted
Caddy root, with an absolute path in `.dvc/config.local`.

Retrieve `LAKEFS_MODELS_ACCESS_KEY_ID` and `LAKEFS_MODELS_SECRET_ACCESS_KEY`
from Infisical project **Infrastructure** (`b1951664-d0f2-4a26-9db3-dc9a0e430369`),
environment `prod`, path `/lakefs`, following the access instructions in the
infrastructure guide. Store them as remote settings `access_key_id` and
`secret_access_key` only in ignored `.dvc/config.local`; restrict that file's
permissions to the account running DVC. Fetch them from Infisical on each
machine instead of copying another clone's configuration. The historically
named `MODELS` key has instance-wide administrative access.

`--local` keeps credentials, certificate paths and machine-specific endpoint
overrides out of Git. Neither application installation nor release packaging
requires these research-storage credentials.

## Publish a recoverable version

Commit the `.dvc` pointers in Git first, then publish the cached bytes and
commit the lakeFS branch. A DVC push only stages objects; it does not create a
lakeFS snapshot.

```powershell
dvc push
if ($LASTEXITCODE -ne 0) { throw 'DVC push failed; do not publish a lakeFS snapshot.' }
lakectl diff lakefs://memory-condense/main
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the staged lakeFS changes.' }
# Review that the staged objects belong to this publication, then:
$dataGitCommit = git rev-parse HEAD
lakectl commit lakefs://memory-condense/main -m 'Research data snapshot' --meta "git_commit=$dataGitCommit"
if ($LASTEXITCODE -ne 0) { throw 'The upload is not yet committed in lakeFS.' }
lakectl log lakefs://memory-condense/main --amount 5
```

Use authenticated `lakectl` as described by the infrastructure guide; it can
run on central-dev when unavailable on a Windows client. Use the Git commit
from this project for the metadata, not the infrastructure checkout's HEAD.
After publishing, check cloud status per target, for example
`dvc status --cloud data.dvc`, and test a pull into an empty-cache checkout.

The local `.dvc/cache` alone is not an off-machine backup. On 2026-10-21 the
first five snapshots were published as lakeFS commit
`32c35586de25c4364af513a2cf3abb47c8a91b529cada5ceacdb1cdbf5dc80a7`, linked
to Git commit `71f93a585b2e3cebf6b9a8a6733fa05eb0c5d342`. All **25,074 unique
objects / 2,496,708,246 remote bytes**, including directory manifests, matched
the expected keys and sizes. The branch had no uncommitted changes afterward.
Publication used lakeFS's authenticated commit API, equivalent to `lakectl
commit`, with the same Git metadata; no direct MinIO writes were used.

Do not write directly to MinIO or enable lakeFS garbage collection: the
infrastructure's `artifacts` repository overlaps the other storage namespaces.
Do not run `dvc gc --cloud` without accounting for retained Git history and
lakeFS snapshots.

## Update or admit another completed run

For an already tracked input or run, update its pointer and cache without
replacing the source files:

```powershell
dvc add --no-relink data
dvc add --no-relink eval_results/chronos-source-time-accuracy-20261016-r1
dvc status
```

Commit the changed `.dvc` pointers with the code or research note that explains
the new data version, then follow **Publish a recoverable version** above,
including the lakeFS commit. Avoid changing old sealed evidence to satisfy a
new experiment; create a new run instead.

For another completed run, first add both its directory and descendants to
the allowlist in `.dvcignore`, before the final exclusions:

```gitignore
!/eval_results/NEW_COMPLETED_RUN/
!/eval_results/NEW_COMPLETED_RUN/**
```

Then run `dvc add --no-relink eval_results/NEW_COMPLETED_RUN`. Inspect the
generated pointer's `size` and `nfiles` and run `dvc status` before committing.
The descendant rule matters: admitting only the directory can produce an
empty snapshot when its contents still match the archive-wide ignore rule.

The initial add uses `--no-relink` and the default copy/reflink cache policy to
preserve the existing sealed files. Do not switch this archive to hardlinks
without accounting for DVC's read-only protection and the artifact writers.

## Validation

Validation covers the selected data, not a new model evaluation:

- Verify the copied LongMemEval-S dataset against its original SHA-256.
- Compare each DVC directory manifest with the intended inventory, including
  nonzero file counts and compiler/live-store exclusions.
- Run `dvc status` against the local snapshots.
- Exercise a temporary local remote with the packet-audit snapshot, pull it
  into an empty DVC project, and compare every restored file's SHA-256.

The temporary restore check below was followed by a real lakeFS restore.

On 2026-10-21 the temporary-remote round trip passed: all **8 files / 218,261
bytes** in the packet-audit snapshot were restored into an empty-cache project
with identical SHA-256 values, and its `dvc status --json` returned `{}`.
The primary repository's `dvc status --json` also returned `{}` for all five
snapshots. Their directory-manifest hashes and file counts were checked, along
with the 500 baseline answer/judgment pairs and the intended exclusions.

The lakeFS check restored `data.dvc` and the packet-audit snapshot into a new
project with an empty cache, using the immutable commit above. All **14 files /
311,536,010 bytes** had identical SHA-256 hashes to their originals, and
`dvc status --json` returned `{}`. The restore and byte comparison took 63.98
seconds. Credentials were retrieved anew from Infisical and passed only in the
restore process's environment. This checks source inputs and audit evidence;
it is not a claim that the larger research archive has been migrated.

References: [DVC add](https://dvc.org/doc/command-reference/add),
[DVC ignore rules](https://dvc.org/doc/user-guide/project-structure/dvcignore-files),
[cache configuration](https://dvc.org/doc/user-guide/project-structure/configuration).
