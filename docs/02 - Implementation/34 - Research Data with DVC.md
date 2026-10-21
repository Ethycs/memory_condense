# Research data with DVC

**Status:** Local data tracking; shared remote pending configuration

**Date:** 2026-10-21

**Scope:** Benchmark inputs and an initial set of completed LongMemEval/Chronos evidence

DVC stores the data bytes in its content-addressed cache. Git stores the small
`.dvc` pointers, `.dvc/config`, `.dvcignore`, and this guide. The application and
release packages do not require DVC. This setup uses the separately installed
DVC 3.67.1 CLI; it does not add dependencies to the model runtime.

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
weights, temporary tests, mutable `.store`/database files, and the baseline's
compiler and transport caches. Small task definitions in `evals/` and project
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

# Once a remote is configured, restore just the benchmark inputs:
dvc pull data.dvc

# Restore all tracked snapshots:
dvc pull
```

The raw payloads stay Git-ignored. `eval_results/*.dvc` is the sole exception
under the evaluation directory so Git can version the pointers.

## Configure the shared copy

No remote destination is assumed. The local `.dvc/cache` is not an off-machine
backup. Configure the intended folder or storage URI before pushing:

```powershell
# Replace the quoted placeholder with the chosen local/network path or URI.
dvc remote add --local -d archive 'APPROVED_FOLDER_OR_STORAGE_URI'
dvc push
```

`--local` writes machine-specific settings to ignored `.dvc/config.local`.
For a shared, credential-free remote URI, omit `--local` so teammates receive
the location through Git. Keep credentials in local configuration or the
storage provider's credential mechanism. Cloud/SSH destinations may require
the matching DVC storage extra; the current installation supports local files
and HTTP(S).

## Update or admit another completed run

For an already tracked input or run, update its pointer and cache without
replacing the source files:

```powershell
dvc add --no-relink data
dvc add --no-relink eval_results/chronos-source-time-accuracy-20261016-r1
dvc status
dvc push
```

Commit the changed `.dvc` pointers with the code or research note that explains
the new data version. Avoid changing old sealed evidence to satisfy a new
experiment; create a new run instead.

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
