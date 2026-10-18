# Handoff — native controls, cold-start visibility and release readiness

**Status:** FROZEN — handoff snapshot; implementation remains in flight  
**Date:** 2026-10-17  
**Workspace:** `F:/Keytone/Documents/GitHub/memory_condense`  
**Branch / HEAD:** `main` / `ff98c20` (`Assess lossless structural V compression`)  
**Working-tree state:** Substantial existing uncommitted implementation and documentation. Preserve it; this handoff did not commit or publish anything.  
**Depends on:** [Viable Directions Index][directions], [System Controls][controls], [proxy consolidation][proxy-plan], [temporal integration findings][chronos]

## Assignment for the next session

Make the existing memory proxy operationally observable from cold start and
finish the native-session control integration needed for the next beta.
The user agreed that this is the current priority over new retrieval research.
The production source-timestamp correction is a release blocker alongside it.

The concrete milestone: open a large saved conversation, immediately see what
startup/initial ingestion is doing, distinguish usable published memory from
pending work, observe a failure without a hanging status request, and operate
the native session through the same controls used by the CLI and MCP.

Start by reading the current code and preserving the working tree. Reuse the
existing publisher, API and lifecycle instead of creating another service.

## Decisions to carry forward

- **One control owner:** the existing proxy publishes status and hosts the
  control API. No separate controls daemon is required by the design.
- **One source of status:** HTTP, saved `status.json`, CLI, MCP and client
  surfaces must agree. Statusline/hooks are passive readers, not additional
  ingestion owners.
- **Isolated sessions:** multiple conversations and inherited branches are
  supported design directions. This is distinct from automatically combining
  unrelated sessions when answering a question, which is outside current scope.
- **Branching is still design work:** do not confuse conversation-ID detection
  with a completed fork/reuse mechanism. Handle it after the immediate controls
  milestone; preserve a place for per-session identity in status now.
- **Continuous durable IO:** inputs, outputs and tools use the existing journal
  and ingestion path. Keep model instances resident, preparation asynchronous,
  and publication/learning ordered.
- **Evidence remains exact:** summaries and events route; original source spans
  remain authoritative. Preserve packet ancestry and idempotent learning.
- **Reuse evaluation artifacts:** do not restart the 500-question campaign or
  rebuild many million-token histories to test controls. Use focused fixtures
  and an existing saved session for an operational smoke test.

## What is already implemented, and the actual gaps

| Area | Present in checkout | Work remaining |
| --- | --- | --- |
| Controls | `/_control/status`, save/load/reset/load-transcript routes; status publisher; CLI/MCP clients; statusline/hooks | Native augment mode uses per-session stores. Existing snapshot/reset/load paths require the observe-mode store; bulk loading requires its capture queue. Finish native semantics and expose capabilities honestly. |
| Native lifecycle status | `ChatSession.status()` reports durable/indexed/pending event counts, pending feedback, preparation jobs and last error | Startup stages, progress within initial work, ready/usable-prefix state and per-session presentation are incomplete. |
| Proxy status | `MemoryProxy.status()` aggregates active sessions, pending events/feedback and failures | Preserve session-level detail, publication freshness and startup state. Aggregation must not hide which session is blocked. |
| Publisher | Bulk jobs fill `current`, rates and ETA; native aggregate status is attached under `memory` | Top-level state is principally derived from capture/bulk activity and can miss native pending work. Audit this rather than displaying a misleading idle state. |
| Status during slow model work | Existing status assembly calls `memory.status()` and summary metrics | Inspect lock dependencies: `NativeProxySessions.summary_status()` takes `model_start_lock`. Status must not wait behind the operation it is supposed to describe. |
| Session attribution | Header/client-ID/transcript fallback exists in the uncommitted `proxy_sessions.py` integration | Automatic branch detection and inherited-artifact reuse remain unimplemented; see the separate branch brief. |
| Temporal metadata | Explicit question time is passed into recall; dated evidence envelopes exist | Native `span.created_at` is often an export anchor. The source-authenticated replay repair is not yet the production implementation. |
| Packaging | Windows/Linux CI, Pixi package, installers and release verification exist | Integrate selected pending changes, update versions/notes and run the complete release gates. Current version metadata is `0.2.2`. |

## Implementation map

Paths below are relative to the workspace.

| Files | Responsibility |
| --- | --- |
| `src/memory_condense/control/status.py` | Shared status shape, file location and atomic publication; schema is currently v1. |
| `src/memory_condense/control/publisher.py` | Aggregate status and periodic publication. |
| `src/memory_condense/control/bulk.py` | Existing transcript jobs, progress and recent-window rate/ETA calculation. |
| `src/memory_condense/control/snapshots.py` | Existing snapshot/store-swap machinery; currently based on the observe-mode store. |
| `src/memory_condense/interfaces/proxy_server.py` | API routes, lifecycle, status wiring and client attribution. |
| `src/memory_condense/interfaces/proxy_memory.py` | Native namespace ownership, session registry, request replay and aggregate status. |
| `src/memory_condense/runtime/sessions.py` | Shared resident models, per-session backend construction and model-start lock. |
| `src/memory_condense/application/chat_session.py` | Durable capture, initial admission, preparation, published prefix, feedback and session status. |
| `src/memory_condense/runtime/resident.py` | Native open/sync/publication and recall against a published snapshot. |
| `src/memory_condense/control/cli.py`, `src/memory_condense/interfaces/mcp_server.py` | User-facing control clients; retain one mutation path. |
| `tools/memory_statusline.py`, `.claude/hooks/memory_state_note.py` | Passive client status renderers. |
| `src/memory_condense/application/chat_native.py`, `src/memory_condense/search/recall_summary_reuse.py` | Evidence date rendering and authentication of reused packet summaries. |
| `src/memory_condense/runtime/compiler.py`, `src/memory_condense/domain/question_time.py` | Compilation/storage time boundary and explicit question-time binding. |

## Suggested implementation order and acceptance checks

1. **Capture the existing state and fix the known test mismatch.** Inspect
   `git status` before edits. The failing test's `serve(config, memory)` stub
   needs to reflect the new `build_app(..., locator=...)` contract while still
   checking the intended wiring. Do not remove the production locator to make
   the test green.
2. **Define startup and readiness status on the existing schema.** Represent
   stages such as model initialization, store admission/reopen, transcript
   preparation and publication. Distinguish a service that can answer status
   requests from a session whose memory is ready. Represent current usable
   prefix and pending source work. Preserve old consumers or version the
   contract deliberately.
3. **Publish progress before and during initial work.** Wire lightweight
   lifecycle observations from the owning runtime. Keep status polling free
   of model loads, ingestion waits and long-held model locks. Progress totals
   and ETA may be unknown; elapsed time and the current stage are still useful.
   A stalled model load must not be reported as completed ingestion.
4. **Complete native controls.** Address the intended session explicitly;
   define snapshot/load/reset and transcript admission against its durable
   journal, native artifacts and learning state. Reuse the existing control
   boundary and protect active requests/publication during a state change.
   Preserve backup-before-destroy and request replay behavior. An unsupported
   operation must be visible as unsupported, not reported as success.
5. **Unify surfaces.** HTTP and persisted status must reflect the same state;
   CLI/MCP and renderers must handle startup, unknown totals, errors, stopped
   publishers and per-session identity. Label stale saved status as stale.
   Verify that native pending work cannot appear idle merely because the
   observe-mode capture queue is empty.
6. **Repair production timestamps.** Bind each hydrated passage's mention date
   to the committed original `ChatEvent`, verifying source, role and exact text.
   Keep the native storage/export anchor separate so existing sealed indexes
   do not need an unplanned rewrite. Authenticate copied-packet dates against
   the same trusted binding. Do not substitute today's date for unknown history.
7. **Validate the operational lifecycle.** Test cold open, slow initialization,
   partial publication, ingestion failure, restart/recovery and session-specific
   controls. Include polling while work is blocked, two isolated sessions,
   snapshot restore and evidence/learning preservation. Use an existing saved
   archive for one live operational check after deterministic tests pass.
8. **Prepare the release candidate.** Review the pending change boundary, add
   new tests to release CI where needed, update setup/capability documentation,
   and run the configured package/fresh-install gates. `v0.2.3-beta.1` is the
   suggested next release, not an already-created tag.

Completion means the user can see and control a real native session from
startup through initial publication and steady-state ingestion, with reliable
error/freshness reporting. A status endpoint that only becomes useful after
the cold-start wait does not meet this milestone.

## Known validation and measurement state

The October 17 focused run completed in 20.03 seconds:

- **166 passed, 1 skipped, 1 failed.**
- Failure:
  `tests/test_proxy_package.py::test_proxy_cli_wires_dialogue_profile_to_each_native_session`.
- Cause: its local `serve()` mock rejects the newly passed `locator` keyword.
- Local report: `eval_results/release-readiness-20261017.xml`.
- Population: `test_proxy_package.py`, `test_proxy_responses.py`,
  `test_summary_api_fallback.py`, `test_semantic_grounding.py`,
  `test_chronos_time.py`, `test_cli_print_env.py`.

That is a focused check, not a green full release suite. New controls tests
exist but were not all included in this run. The release workflow uses an
explicit test list, so adding files alone does not put them into CI.

The completed LongMemEval-S source-date replay is **331/500 (66.2%)**, versus
299/500 (59.8%) on the original fixed packets. Excluding the 133 multi-session
questions gives **266/367 (72.5%)**, versus 236/367 (64.3%). These are answer
scores under the saved reader/judge protocol, not raw retrieval recall or
production migration validation. No event index was included. See
[Research Log 278][chronos] for the sealed artifacts and limitations.

The 23–25 minute cold BGE construction anomaly is separate from warm query
latency. Compressed model/environment files are a leading hypothesis, not a
confirmed root cause. Add observability before treating an uninstrumented
startup wait as ordinary ingestion cost. See [Research Log 273][landmarks].

## Verification commands and first action

Use the existing checkout, not the retired `ingest-speed` worktree. Confirm
the branch/HEAD again because another session may have advanced it:

```powershell
Set-Location 'F:/Keytone/Documents/GitHub/memory_condense'
git log -1 --oneline
git status --short
Get-Content 'docs/01 - Design/03 - System Controls.md'
Get-Content 'docs/01 - Design/04 - Viable Directions Index.md'
```

After inspecting the relevant code, a focused provider-free test command is:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONUTF8 = '1'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$controlsTestTemp = Join-Path $env:TEMP ('mc-controls-' + [guid]::NewGuid().ToString('N'))
& '.\.pixi\envs\dev\python.exe' -m pytest -q `
  tests/test_control.py tests/test_control_cli.py tests/test_memory_surfaces.py `
  tests/test_proxy_package.py tests/test_proxy_memory.py tests/test_proxy_responses.py `
  tests/test_proxy_sessions.py tests/test_proxy_attribution.py `
  tests/test_proxy_headers.py tests/test_proxy_receipts.py tests/test_proxy_units.py `
  tests/test_chat_session.py tests/test_chat_streaming.py `
  tests/test_chronos_time.py tests/test_recall_summary_reuse.py `
  --basetemp $controlsTestTemp
```

Run the configured [release checks][ci] after the implementation changes; the
command above does not substitute for packaging or a live startup check.

**First action:** trace `MemoryProxy.status()` → `ChatSession.status()` and
`NativeProxySessions.summary_status()` alongside `StatusPublisher.document()`.
Determine how a status request can return immediately while initial native
admission or model loading is in progress, then implement that path with a
deterministic blocked-worker test.

## Follow-on work, kept separate

- [Session branching and artifact reuse][branches], including
  [hashing/compaction boundaries][identity].
- Recent-packet pruning and reuse: **D08** in the [directions index][directions].
- Event index and bounded retrieval repair: **D11–D14** in that index.
- [Length-aware scheduling and client compaction][compaction].

The INT8 landmark route failed its holdout; do not enable it as part of this
release. Do not reinterpret the single-session product policy as a request to
merge conversations while building controls.

[directions]: <../01 - Design/04 - Viable Directions Index.md>
[controls]: <../01 - Design/03 - System Controls.md>
[proxy-plan]: <../06 - Roadmaps/06 - Proxy Consolidation.md>
[chronos]: <../10 - Research Log/278 - 2026-10-16 - Chronos temporal event lane integration plan.md>
[landmarks]: <../10 - Research Log/273 - 2026-10-15 - Post-benchmark INT8 landmark experiment plan.md>
[branches]: <../02 - Implementation/32 - Context Fork and Branch Detection.md>
[identity]: <../10 - Research Log/271 - 2026-10-13 - What content hashing resolves in context identity.md>
[compaction]: <../10 - Research Log/277 - 2026-10-15 - Client compaction and length-aware ingest scheduling.md>
[ci]: <../../.github/workflows/ci.yml>
