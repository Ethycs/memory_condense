# Application native memory ingest and reopen

**Status:** Real application ingestion and independent reopen verification complete; all 100 packets match.
**Date:** 2026-09-15.
**Scope:** The existing 1,098,417-token history, then its unchanged 100 questions.
**Depends on:** [Research Log 222](222%20-%202026-09-15%20-%20Memory%20evaluation%20lifecycle%20boundary.md).

## Application change

`MemoryCondenser` now exposes `install_native_spine`, `native_spine_receipt`, and
`retrieve_native_spine` through a small workflow mixin. The normal `ingest_many`
path durably stores, chunks, embeds and indexes raw turns first. Native snapshot
installation refuses pending ingestion and verifies complete, gap-free coverage
of every stored turn, exact source identity and timestamps, a complete hierarchy,
and matching normalized FP32 summary vectors.

The snapshot is committed transactionally to `native-spine.sqlite` beside
`memory.db`. It contains summaries, raw addresses and vectors. Raw evidence is
loaded from the application's `TranscriptStore`, not an external source-bank
callback. Reopening validates the persisted bytes and raw transcript identity
before rebuilding the resident summary index. A changed encoder or an appended
raw turn prevents stale native retrieval. The existing default chunk retrieval
API remains available; the evaluation will explicitly call the native API.

Previously compiled atomic summaries, local-Qwen attention/parent summaries,
and summary vectors are reused. This integration does not regenerate them or
send raw content to Qwen. The normal application ingest path additionally builds
its standard raw-chunk BGE index. That cold work is measured separately from
warm native answering.

## Verification

Eleven new lifecycle tests pass. They execute normal ingestion, close and reopen
the application, compare the complete retrieved packet with the resident
baseline, observe application-database hydration, and reject partial ingestion,
missing hierarchy/atoms, changed raw sources, modified vectors/summaries, stale
snapshots and mismatched encoders. All 138 existing condenser tests also passed.
The first test run had two fixture errors: direct SQLite corruption lacked the
application's schema-writer function. Supplying that function in the deliberate
corruption fixtures resolved both; no production logic was relaxed.

Seventeen further checks passed for the answer-run admission gate and exact
conversation rendering. The gate rejects multiple histories, fewer than 100
verified questions, missing process-reopen proof, source-cache reconstruction,
or a different persisted snapshot.

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m pytest tests/test_native_spine_application_lifecycle.py -q --basetemp=.tmp/pytest-native-app-lifecycle-20260915-r2
```

## Real history execution

`tools/verify_native_spine_application.py` has separate `ingest` and `verify`
phases. The second requires the first process to have exited. It opens only the
saved application for routing and hydration, with no source namespace or vector
cache reconstruction. All 100 fresh question packets must match the completed
84/100 baseline exactly before a new answer run can proceed.

| Item | Binding |
| --- | --- |
| Root | `eval_results/native-spine-application-lifecycle-20260915-r1` |
| Ingest log | `eval_results/native-spine-application-lifecycle-20260915-r1-ingest.log` |
| Ingest exec session | `47655`, terminal exit 0 |
| Reopen verification exec session | `39712`, terminal exit 0 |
| Worker | PID `40152`, creation time `1789516825.724352` |
| Ingest plan SHA-256 | `0229d5bd93dbec7b16101f72ae11f5d9111cdc42422e2722fba76a915e5dc0b1` |
| Raw turns | 5,357, excluding source-boundary metadata |
| Actual raw text tokens | 1,098,417 |
| Ingest completion SHA-256 | `2bd4f37b0ae813d260a4c67151b54f6741004d994a5d1ba1f929f797c8cb6235` |
| Persisted native snapshot SHA-256 | `65de50179645f9ecb96821dff970002ae0e42f2d237585c8432001c5cf30f54d` |
| Reopen verification SHA-256 | `ca721112762da52024df005868c3b092b8f684c5d37d2e9f99af62497c88d2e0` |

The ingest command has already been released; do not duplicate it:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -u -m tools.verify_native_spine_application ingest --root eval_results/native-spine-application-lifecycle-20260915-r1
```

The completed separate-process reopen command:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -u -m tools.verify_native_spine_application verify --root eval_results/native-spine-application-lifecycle-20260915-r1
```

The raw application ingest took **571.463 seconds**; total preparation including
cache validation, model load, index installation and close took 599.089 seconds.
The ingestion process exited before reopening began. All **100** fresh application
retrievals then matched the 84/100 baseline's complete routing, hydration and
reader messages, including **2,656 exact raw spans**. Median packet preparation
was **0.155 seconds**, p95 0.218 seconds. These are retrieval measurements without
answer-model calls, not end-to-end answer latency or a new accuracy score.

The conversation-order comparison now requires this completed verification and
uses `MemoryCondenser.retrieve_native_spine` for timed retrieval. It holds selected
evidence, reader, model, questions and grading fixed and changes only presentation.
Its released execution is recorded in Research Log 224. Latest completed answer
accuracy at release was 84/100. Research Log 224 now records its completed
85/100 result; matching retrieval packets alone did not create that score.
