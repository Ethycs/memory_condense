# Progress — cold-start visibility: status no longer waits on the work it describes

**Status:** IN PROGRESS — continues the 2026-10-17 handoff, which remains frozen
**Date:** 2026-10-18
**Workspace:** `F:/Keytone/Documents/GitHub/memory_condense`
**Branch / HEAD:** `main` / `ff98c20` (unchanged; nothing committed this session)
**Working-tree state:** The prior uncommitted implementation was preserved. This
session added to it; no file was reverted, committed or published.
**Continues:** [controls/cold-start handoff][handoff], [System Controls][controls]

## What this session completed

Steps 1–3 of the handoff's implementation order, and the renderer half of step
5. Steps 4, 6, 7 and 8 are untouched and remain the next session's work.

### 1. The known test mismatch — and two more instances

The handoff identified
`test_proxy_package.py::test_proxy_cli_wires_dialogue_profile_to_each_native_session`
as failing because its local `serve()` mock rejects the `locator` keyword. The
full suite shows the same stub defect in two further places:

- `tests/test_automatic_tactics.py::test_proxy_default_and_forced_modes_reach_native_sessions`
- `tests/test_native_experiments.py::test_installed_proxy_cli_propagates_experiments_to_all_sessions`

All three now accept `locator` and assert it was passed, so the production
locator is still covered rather than removed to make the tests green.

### 2–3. Status answers during a cold start

**The defect.** Three status paths queued behind the very work they describe:

| Path | Lock | Held by | Worst case |
| --- | --- | --- | --- |
| `NativeProxySessions.summary_status()` | `model_start_lock` | `Backend._open` for the whole model construction | the full 23–25 min cold build |
| `MemoryProxy.status()` | `MemoryProxy.lock` | `entry()` across construction and journal replay | a large journal replay |
| `StatusPublisher.document()` | — | calls both of the above | both |

`ChatSession.status()` was already safe: its `_lock` is never held across
`backend.sync()`, `sync_prepared()` or `prepare()`. `LocalFirstSummaries._lock`
was also safe — it guards counter updates only, never the grounder build.

**The fix.** Registry access was separated from the serialisation of model
starts. `NativeProxySessions` gained a `registry_lock` held only for dictionary
access; `model_start_lock` now serialises model construction and nothing else.
`MemoryProxy` keeps `_live`/`_opening` mirrors under its own `_status_lock` and
`status()` no longer takes `MemoryProxy.lock` at all. The only lock nesting is
`model_start_lock` → `registry_lock` and `lock` → `_status_lock`; neither order
is ever inverted.

**Lifecycle stages** are recorded *before* the wait they describe, so a blocked
start reads as blocked rather than as missing: `constructing`, `admitting`,
`awaiting_model_start`, `opening_models`, `ready`, `failed`. They surface under
`memory.startup` per namespace with `stage`, `error`, `elapsed_s` and
`stage_elapsed_s`; sessions still inside `entry()` appear under
`memory.opening_sessions`. `memory.sessions` retains per-session detail so
aggregation cannot hide which session is blocked.

**The idle lie.** `queue.*` is an observe-mode structure and stays empty in
augment mode, so a cold start published a confident `idle`. Top-level `state`
now also consults the native service: `starting` while any session is
pre-ready, `ingesting` while native work is outstanding with all sessions
ready. This is the one deliberate addition to schema 1 — a renderer that does
not know `starting` falls through to printing the state name. An unavailable
`memory.status()` is reported as unavailable and never inferred as progress.

### 5 (renderers only)

`one_line`, the `ctl status` table and the `UserPromptSubmit` hook now render
startup. Because a cold load has no totals and no ETA, they show the stage name
and elapsed time, which is what distinguishes "alive" from "wedged". The hook
additionally tracks the stage, so a long load produces a note when it advances
rather than going silent after the first turn. The CLI table prints a row per
session; it no longer prints `idle` under `current` during a cold start.

## Files changed this session

| File | Change |
| --- | --- |
| `src/memory_condense/runtime/sessions.py` | `registry_lock`, `_observe`, `startup_status()`, `_summary_runtime()`; `summary_status()` off `model_start_lock` |
| `src/memory_condense/interfaces/proxy_memory.py` | `_status_lock`, `_live`/`_opening` mirrors; `status()` off `MemoryProxy.lock`; `startup`/`opening_sessions`/`sessions`/`busy` |
| `src/memory_condense/control/publisher.py` | `_native_progress()`; `starting` and native-busy state derivation |
| `src/memory_condense/control/status.py` | `starting` in `STATES`; `startup_records()`, `describe_startup()`; `one_line` startup rendering |
| `src/memory_condense/control/cli.py` | startup in the table, per-session rows, unavailable-memory row |
| `.claude/hooks/memory_state_note.py` | `startup_stage()`; `starting` transitions and stage-advance notes |
| `tests/test_control_cold_start.py` | new — 9 tests |
| `tests/test_memory_surfaces.py` | hook cold-start stage-advance test |
| `tests/test_proxy_package.py`, `tests/test_automatic_tactics.py`, `tests/test_native_experiments.py` | `locator` stubs |
| `docs/01 - Design/03 - System Controls.md` | new "Startup and readiness (augment mode)" section; `starting` in the schema |
| `.github/workflows/ci.yml` | controls tests added to the explicit list |

## Validation

`tests/test_control_cold_start.py` blocks deterministically: the worker is held
on an event, the status call runs on another thread with a 10 s timeout, and
the worker is asserted **still blocked** when that call returns. No timing
threshold decides correctness. It covers a blocked model start, a failed model
start, a blocked session open, `starting` vs `idle` vs `ingesting`, an
unavailable memory service, and the CLI table.

- Handoff's focused command plus the controls and newly fixed files:
  **336 passed, 1 skipped, 0 failed** (its baseline was 276 / 1 / 1).
- The exact release CI list, with the controls tests added:
  **620 passed, 1 skipped, 0 failed** (97 s).
- Full `tests/`: 8509 passed, 34 failed before this session's fixes. Three of
  those were the `locator` stubs and are fixed. The remaining 31 are
  pre-existing and unrelated — none of their files import `runtime.sessions`,
  `interfaces.proxy_memory` or `control.*`. They are concentrated in
  `test_matched_eval_*`, `test_diffuse_longmemeval_*` and
  `test_run_locked_*`. The three `test_architecture.py` failures concern
  `src/cli.py`, `application/condenser.py` (815 > 800 lines) and seven
  oversized persistence/search modules — all untouched here.

The full suite is still not green, and this session did not make it green.

## What remains, unchanged in priority

Handoff steps 4 and 6–8, in its order:

4. **Native controls semantics.** Snapshot/load/reset and transcript admission
   still assume the observe-mode store and its capture queue. Define them
   against the native session's durable journal, artifacts and learning state,
   address the intended session explicitly, and make an unsupported operation
   visible as unsupported rather than reported as success.
6. **Production source timestamps.** Still the release blocker named in the
   handoff. Nothing here touched `span.created_at` binding.
7. **Operational lifecycle validation**, including the live check against an
   existing saved archive. Only deterministic unit-level cold-start behaviour
   is covered so far; no live cold open was run this session.
8. **Release candidate.** `v0.2.3-beta.1` suggested, not created. Version
   metadata is still `0.2.2`. Packaging and fresh-install gates were not run.

One caveat for step 7: the stage names are now the observability contract the
live check should be read against. If the 23–25 minute anomaly turns out to sit
inside `attention.prepare()`, `opening_models` will hold for that entire window
and the stage set will need splitting to localise it further.

[handoff]: <2026-10-17_controls-cold-start-release-handoff.md>
[controls]: <../01 - Design/03 - System Controls.md>
