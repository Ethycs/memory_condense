# Research data with DVC

**Status:** Local data tracked; lakeFS remote configured; first upload awaits authentication

**Date:** 2026-10-21

**Scope:** Benchmark inputs and an initial set of completed LongMemEval/Chronos evidence

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
from r26, whose stores are outside this initial snapshot. This does **not** yet
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
`dvc dag --outs` must show all five snapshots.

## Shared storage and machine setup

Follow the infrastructure [project-data guide](https://central-dev.zt:9000/#operations/project-data.md).
The credential-free remote is configured in `.dvc/config`:

| Setting | Value |
| --- | --- |
| DVC remote | `lakefs` (default) |
| DVC URL | `s3://memory-condense/main/` |
| Endpoint | `https://central-dev.zt:8000` |
| Intended lakeFS repository | `memory-condense` |
| Intended storage namespace | `s3://lakefs-storage/memory-condense` |

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

The local `.dvc/cache` alone is not an off-machine backup. The configured
destination is not proof that the data has been uploaded. The first remote
publication is still pending: on 2026-10-21 lakeFS health returned HTTP 204,
but local Infisical needed login and SSH authentication was unavailable.

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

The temporary restore check is not a substitute for pushing to and checking
the eventual shared remote.

On 2026-10-21 the temporary-remote round trip passed: all **8 files / 218,261
bytes** in the packet-audit snapshot were restored into an empty-cache project
with identical SHA-256 values, and its `dvc status --json` returned `{}`.
The primary repository's `dvc status --json` also returned `{}` for all five
snapshots. Their directory-manifest hashes and file counts were checked, along
with the 500 baseline answer/judgment pairs and the intended exclusions.

References: [DVC add](https://dvc.org/doc/command-reference/add),
[DVC ignore rules](https://dvc.org/doc/user-guide/project-structure/dvcignore-files),
[cache configuration](https://dvc.org/doc/user-guide/project-structure/configuration).
