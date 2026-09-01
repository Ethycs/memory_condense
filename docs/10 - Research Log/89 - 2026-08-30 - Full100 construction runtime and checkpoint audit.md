# Full100 construction runtime and checkpoint audit

Date: 2026-08-30

## Result

The live provider-free full100 construction was healthy and compute-bound at
the audited snapshot, but its output root was not yet visible because the
current implementation defers every publication until all ten resident
namespaces have finished. The open `offset-090` store corresponded to
namespace position 4 of 10 in hash-sorted execution order. Therefore 19 of the
68 eligible questions were definitely complete at that moment; progress
within the eight-question current namespace could not be recovered from a
durable marker.

There is no construction checkpoint from which an interrupted process can
resume. A fresh `construct` invocation starts the 68-question resident build
again. The `replay` command also rebuilds the entire resident construction
before it compares bytes, so it is deterministic but not a cheap verification
pass.

This is a runtime and lifecycle audit, not an accuracy result. At the audited
snapshot no full100 construction artifact, provider answer result, or judge
score existed. The live process was not modified, attached to, interrupted, or
restarted during this audit. No code was changed as part of the audit.

## Scope and authority

The audited entry point was:

```text
.pixi\envs\dev\python.exe
tools\run_locked_semantic_global_terminal_full100_construction.py construct
```

The relevant implementation surfaces were:

- `tools/run_locked_semantic_global_terminal_full100_construction.py`
- `tools/run_reduced_semantic_global_completion_assay.py`
- `tools/run_reduced_second_read_retrieval_assay.py`
- `tools/matched_eval/query_guided_scan.py`
- `tools/matched_eval/artifacts.py`

The namespace population and eligibility counts came from the sealed gate:

```text
eval_results/matched_eval_100/locked-semantic-residual-v4-r7/
locked-semantic-residual-gate-v4.json
```

The runtime observations below describe one transient process snapshot. The
code and sealed source artifacts, rather than a process observation, remain
the authority for lifecycle behavior and population identity.

## Observed facts

### Transient process snapshot

At `2026-08-30T19:44:10-07:00`, PID `54232` reported:

| Field | Observation |
| --- | ---: |
| Start time | `2026-08-30T18:47:33-07:00` |
| Elapsed wall time | `00:56:37` |
| Accumulated CPU time | `3381.4` seconds |
| Working set | `2.17 GiB` |
| Private memory | `2.16 GiB` |
| Threads | `1` |
| Process state | responding/running |

A separate approximately five-second sample advanced user CPU time by 4.94
seconds while file-read and file-write counters did not change. That is
consistent with an active compute phase, not an observed I/O wait or process
stall. It does not predict completion time.

The only open repository data file at that snapshot was:

```text
eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822/
shards/offset-090/combined-store/memory.db
```

The parent PowerShell command contained no output redirection. Namespace
start/complete events were therefore transient stderr progress, not a durable
checkpoint log.

### Exact execution order and lower-bound progress

`build_assay()` groups eligible ordinals by namespace and iterates
`sorted(by_namespace)` in
`tools/run_reduced_semantic_global_completion_assay.py:449-454`. Namespace
order is therefore lexicographic SHA-256 order, not shard-offset order.

The exact gate-derived order was:

| Position | Namespace ID | Store offset | Eligible count | Eligible ordinals |
| ---: | --- | ---: | ---: | --- |
| 1 | `022ed03179381a9ddf4e1b8efbe1ddf2733101c1b277a3f223c99dc031493e2d` | 010 | 5 | 11, 13, 14, 18, 19 |
| 2 | `195d61a8457619d6d8b00f78d4a07e7973e0dbff099f800ea4864042a699bf7f` | 080 | 7 | 81, 82, 83, 84, 85, 86, 88 |
| 3 | `2030ab2d410ba5cd587d9826271ab7372a1f4bafbba868cdd1b1b54babd47173` | 040 | 7 | 40, 41, 43, 46, 47, 48, 49 |
| 4 | `26d4845bae41c910e00e1a01a1f7f564e79811178d3742263f201c16f5e4ea33` | 090 | 8 | 91, 92, 93, 94, 95, 97, 98, 99 |
| 5 | `29f396280600c20edb3c9157a6620d2bf6a4e199350727545873cc86462c3be8` | 000 | 6 | 3, 4, 5, 6, 7, 9 |
| 6 | `524aabf06143df7014bbbca6780e53976687229d84304968b040717ed4e89615` | 050 | 8 | 50, 51, 53, 54, 56, 57, 58, 59 |
| 7 | `9baa77e7040d26344466cd70b6e1931b3e2c4bb50a39ca10107a9f88d3fc880b` | 060 | 8 | 61, 63, 64, 65, 66, 67, 68, 69 |
| 8 | `b68cecc77945db0110af15da9bfbd41472bd70c6030e5419ed89e778e24717d3` | 020 | 7 | 20, 21, 22, 24, 27, 28, 29 |
| 9 | `c9274e896ed9201eb961bf6e01a5358dc08733e091fa022fd6efd379552e81b9` | 030 | 6 | 30, 31, 32, 33, 35, 36 |
| 10 | `f6ff567930f1ed4c4d5b3f9da96fae29162ffc8e5330e5895505eeae203b18ba` | 070 | 6 | 70, 72, 73, 75, 77, 78 |

Because the process had the position-4 database open, positions 1--3 were
complete. Their eligible counts sum to `5 + 7 + 7 = 19`, establishing a
19/68 lower bound. Six full namespaces containing 41 eligible questions
followed the current namespace. With no durable per-question marker inside
position 4, the bounded remaining work was 41--49 eligible questions plus
final composition and publication.

That bound is an observed-state inference from the exact loop order and open
database. It is not an ETA. Namespace cost varies with store size, eligible
question count, and question-specific local/global search work.

### Deferred publication

The missing output root was expected under the implementation:

1. `build_construction_bundle()` loads sources, derives all 68 eligible
   ordinals, and calls `v7_cli.build_assay()` for the whole population
   (`tools/run_locked_semantic_global_terminal_full100_construction.py:775-800`).
2. `build_assay()` keeps question results in `question_by_ordinal` and namespace
   receipts in an in-memory list until every namespace finishes
   (`tools/run_reduced_semantic_global_completion_assay.py:449-568`).
3. Only after the bundle returns does `run_construct()` publish namespace
   sidecars and then the compact construction manifest
   (`tools/run_locked_semantic_global_terminal_full100_construction.py:1184-1203`).

At the snapshot the configured root did not exist:

```text
eval_results/matched_eval_100/
locked-semantic-global-terminal-full100-v1
```

After a successful complete build, the intended files are:

```text
locked-semantic-global-terminal-full100-v1/
├── semantic-global-terminal-full100-namespace-sidecars-v1/
│   ├── <canonical-sidecar-sha256>.json
│   └── <canonical-sidecar-sha256>.json.sha256
├── semantic-global-terminal-full100-construction-v1.json
└── semantic-global-terminal-full100-construction-v1.json.sha256
```

`publish_sealed_json()` creates the parent directory only when publication
begins. It writes short-lived `.tmp` files in that destination and atomically
replaces them (`tools/matched_eval/artifacts.py:53-95`). Those temporary files
are publication mechanics, not progress checkpoints.

## Checkpoint and replay behavior

### No resumable construction checkpoint

The namespace cache is constructed from one read-only SQLite scan and stored as
Python objects, including a `MappingProxyType`
(`tools/matched_eval/query_guided_scan.py:561-665`). Per-question outputs and
namespace receipts also remain in process memory. The current implementation
does not serialize a completed namespace before proceeding to the next one.

Consequences:

- Letting the healthy live PID continue was safe.
- Interrupting or losing that PID would lose all resident progress.
- A new `construct` invocation would redo all ten namespaces and 68 eligible
  questions; it would not resume at position 4.
- Write-once publication is idempotent for already completed, byte-identical
  artifacts, but publication starts too late to protect ordinary mid-build
  progress.

### Replay performs a full rebuild

`run_replay()` first calls `build_construction_bundle()` again and only then
loads and compares the published sidecars and manifest
(`tools/run_locked_semantic_global_terminal_full100_construction.py:1217-1248`).
It therefore reopens and recomputes every namespace. Replay is safe after a
successful construction and with the expected construction SHA-256, but it is
not a lightweight load-and-hash check.

Within each construction pass, every eligible question also runs a fresh V6
local reinjection replay and a fresh V7 global-completion replay immediately
after its first search
(`tools/run_reduced_semantic_global_completion_assay.py:195-246`). This is an
assurance cost, not additional retrieval coverage.

## Estimates, not observations

The audit did not produce a reliable completion-time estimate. A naive
namespace-linear extrapolation from position 4 suggested roughly another
1.5--2 hours at the snapshot, but this estimate is weak because the precise
position within namespace 4 was unknown and namespace workloads are unequal.
It must not be treated as a promised completion time or a performance result.

Likewise, 19/68 was a lower bound on completed construction rows, not a recall
or accuracy score. No provider answers or judge results were generated by this
runtime observation.

## Ranked no-semantic-change refactors

The following changes are ranked by expected operational value. They are
proposals only; none was applied to the live process.

### 1. Yield and seal each completed namespace

Refactor the resident builder to yield a canonical namespace result as soon as
that namespace completes. Publish its authenticated sidecar immediately, and
on restart load and verify sidecars keyed by the frozen source, policy, vector,
and namespace identities before skipping completed work. Assemble the final
manifest in the same canonical namespace order.

This preserves retrieval decisions and final ordering while converting a
multi-hour all-or-nothing run into a resumable ten-unit run. It also makes the
output root and progress visible without weakening immutable publication.

### 2. Separate construction from duplicate deterministic audit replay

The hot construction path currently performs local search plus local replay
and global search plus global replay for every question. The top-level replay
then repeats the full doubled build. A versioned execution contract should
allow the production construction to compute each result once, while an
explicit audit mode performs the independent deterministic rebuild when that
assurance is required.

Retrieval semantics and provider payloads need not change, but the attestation
must truthfully distinguish `constructed_once` from `independently_replayed`.
This proposal therefore preserves answer semantics, not the current assurance
metadata byte-for-byte.

### 3. Pass already verified immutable inputs through the resident stack

The full100 wrapper loads and verifies the gate, R7 construction, and vector
artifacts before `build_assay()` loads the same roots again. The scoped
namespace helper also reloads the full query-preflight/retrieval population for
each namespace even though the resident builder already loaded it once.

Pass authenticated source objects and a precomputed namespace-to-store binding
through the call graph. Keep the per-store database/index hash checks at the
trust boundary. This removes repeated JSON parsing, hashing, and population
reconstruction without changing candidate ranking, budgets, or output order.

### 4. Add bounded namespace parallelism behind a deterministic reducer

Namespaces are independent before the final ordered composition. A bounded
worker pool could process several namespaces concurrently and then sort their
results by namespace ID before sealing. However, the observed resident process
used about 2.2 GiB, so concurrency must be explicitly memory-capped and tested
for byte-identical output against the serial implementation. Two workers are a
safer first assay than unconstrained parallelism.

### 5. Persist a progress ledger separate from authoritative artifacts

Write a small append-only or atomic progress receipt when a namespace starts
and completes, including namespace ID, position, eligible count, and elapsed
time. This does not accelerate retrieval by itself, but it replaces transient
stderr with durable observability and makes remaining-work estimates auditable.
The ledger should not be accepted as a resumable checkpoint unless the sealed
namespace sidecar in proposal 1 also verifies.

## Status at handoff

- Live construction: running and compute-active at the audited snapshot.
- Durable full100 construction output: not yet present at that snapshot.
- Definitely completed eligible rows: at least 19 of 68.
- Resume after interruption: unavailable.
- Replay cost: full resident rebuild.
- Accuracy result from this run: none yet.
- Live process interventions during audit: none.
