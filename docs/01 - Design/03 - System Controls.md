# System controls: status, progress, and the control plane

**Status**: Design note, implementation batched 2026-10-13
**Applies to**: the provider proxy (`interfaces/proxy_server.py`), the MCP
server, the installed CLI, and the Claude Code / Codex user surfaces

## Purpose

The memory system runs out of sight: a proxy on the model path, a capture
queue, background ingestion, a store. The user needs to see what it is doing
(queue, whether a whole transcript is being loaded and how long is left,
ingest rate, tokens ingested) and to control it (save, load, reset) from
inside the client they are already using. This note fixes how.

## The rule

**One status document, one control API. Every surface is a renderer of the
first or a client of the second.** No surface computes its own numbers; two
surfaces must never disagree.

- The **status document** is written by the process that owns the queue and
  the store: the proxy (observe or augment mode). Nothing else writes it.
- The **control API** is a handful of local-only HTTP routes on that same
  process. Mutating operations have exactly one path and always back up
  before they destroy.

## Status document

Path: `<data-dir>/control/status.json`, written atomically (temp + rename) on
every state change and at least once per second while a job is running.
`<data-dir>` is the proxy's `--data-dir`. Readers resolve it in this order and
take the first that exists (helper: `memory_condense.control.status.locate_status_file`):

1. `MEMORY_CONDENSE_STATUS` (explicit file path)
2. `$MEMORY_CONDENSE_DATA_DIR/control/status.json` — this variable steers the
   **writer** as well as every reader, so setting it alone puts the proxy and
   the surfaces in the same place (`MEMORY_DATA_DIR` is a deprecated alias,
   honoured by the proxy only)
3. `<project root>/.memory_condense/control/status.json`, then
   `./.memory_condense/control/status.json` for a store predating the
   project-root rule
4. `user_directory()/data/control/status.json` — no project found

## Which store is this project's

**The deployed release and a development checkout answer this differently, on
purpose.** Read this section with that split in mind.

### Deployment: one absolute store, not per project

The installed `memory-condense proxy` injects `--data-dir
user_directory()/'data'` in `cli.py` and never consults the resolver below.
That is deliberate and load-bearing: the Docker image sets
`XDG_DATA_HOME=/data` so `user_directory()` lands on the mounted volume with
no argument juggling, and `packaging/RELEASE_NOTES.md` tells upgraders to
restart with the same absolute `--data-dir` and to back that directory up.
Changing it would break the data volume and the documented upgrade path.

This is right for the proxy's shape: it is a long-lived daemon on a port
serving whatever client connects, so it owns one store across projects.

The **MCP server** is the opposite shape — the client spawns one per project —
so it is per-project, and has been since it shipped. The two therefore
disagree in deployment, and that is not a defect to unify away: it is why the
control tools' descriptions name which store they act on (see the Surfaces
table), and why `memory_reset` says plainly that it does not clear what
`recall` returns.

### The resolver (MCP server, and any proxy run without `--data-dir`)

`memory_condense.runtime.project.project_data_dir` is the single answer for
the surfaces that do infer a directory: the MCP server (`_data_dir`), the
proxy's `default_data_dir` fallback, and the reader order above. Those three
used to decide separately and disagreed about the project even when they
agreed about the mode.

1. `$MEMORY_CONDENSE_DATA_DIR` — an explicit override always wins.
2. `<project root>/.memory_condense` — the default. The root is
   `$CLAUDE_PROJECT_DIR` when the client sets it (the client naming the project
   beats any inference), else the nearest ancestor of the working directory
   holding a project marker. Markers are tried in order over the whole
   ancestor chain — `.git` first, so a monorepo subpackage shares its
   repository's memory, then `pyproject.toml` for a project that is not a git
   checkout. `.git` is a file in a worktree, so each worktree keeps its own
   store.
3. An already-existing `./.memory_condense` when no project was found — the
   MCP server's old default created stores without a project check, and
   adopting one beats opening an empty store and looking like data loss. Never
   creates the directory.
4. `user_directory()/data` — the session's home, when there is no project at
   all.

**The root is found by walking up, not by trusting the working directory.**
The previous `Path.cwd()/".memory_condense"` gave a process launched from
`src/` a different store than one launched from the repository root, splitting
one project's memory across however many subdirectories it was started from.

`runtime/project.py` is stdlib-only and imports `runtime.config` lazily: the
statusline imports it on every refresh, where `runtime.config` costs a
measured ~23 ms extra (it pulls `platform`, and `_wmi` on Windows). The walk
itself is a handful of `exists()` calls, ~4 ms against the ~51 ms
process-creation floor a spawned statusline already pays.

Schema (all fields present; null when not applicable):

```json
{
  "schema": 1,
  "written_at": "2026-10-13T08:41:02Z",
  "pid": 23628,
  "mode": "augment",                       // observe | augment
  "state": "ingesting",                    // idle | starting | ingesting | condensing | swapping | error
  "health": {"status": "ok", "reasons": []},  // mirrors /health
  "queue": {
    "capacity": 256, "in_flight": 3, "retained_bytes": 1048576,
    "rejected_saturated": 0, "terminal_failures": 0,
    "jobs": [{"id": "bulk-01a08337", "kind": "transcript", "session": "01a08337",
              "position": 1, "bytes": 258895902}]
  },
  "current": {                             // null when idle
    "job": "bulk-01a08337", "kind": "transcript",
    "records_done": 2944, "records_total": 5677,
    "bytes_done": 134217728, "bytes_total": 258895902,
    "tokens_done": 612000, "tokens_ingested": 48200,
    "started_at": "...", "eta_s": 98
  },
  "rates": {"tokens_per_s_recent": 6200, "records_per_s_recent": 31.0,
            "avg_s_per_1k_tokens": 0.16, "window_s": 30},
  "totals": {"tokens_ingested": 1210000, "records_ingested": 18231,
             "sessions": 3, "store_bytes": 48100000},
  "serving": {"requests": 41, "rewritten": 37,   // served requests only
              "static_tokens_last": 1104000, "served_tokens_last": 48200,
              "saved_tokens_total": 38900000},  // null served = unknown
  "store": {"path": "...", "snapshot_dir": "...", "last_snapshot": "2026-10-13T08-40-00.sqlite",
            "snapshots": ["..."]}
}
```

In **augment** mode `store.path` is the `sessions/` directory rather than a
single SQLite file, `totals.store_bytes` is that tree's recursive total, and
`store.snapshots` / `store.last_snapshot` are directory names with no
`.sqlite` suffix.

Sourcing: `queue.*` and `totals.records_ingested` come from the existing
`CaptureQueue` counters (`in_flight`, `retained_bytes`, `rejected_saturated`,
`capture_terminal_failures`, `ingested`); `health` mirrors the existing
`/health` route; `serving` comes from `interfaces/proxy_receipts.py`, the sole
producer of those numbers in **both** modes — `StatusPublisher` only
accumulates receipts and recomputes nothing; `current`/`rates` come from the
bulk loader below. **ETA is a recent-window rate** (last `window_s` seconds),
never a lifetime mean — ingest speed varies with record size, and a lifetime
mean lies for the first minute and the last.

**Two units, kept apart.** `totals.tokens_ingested` and
`current.tokens_ingested` count tokens of the text actually handed to the
sink, identically for live traffic and for bulk jobs — *not* file bytes. File
progress is a separate unit and lives in `current.tokens_done`,
`current.bytes_*`, `rates.*` and `eta_s`, which measure the transcript being
read (tool calls, thinking and meta records included) so the ETA tracks the
file. Expect `tokens_ingested` to be well below `tokens_done` on a real
client transcript.

**Augment capacity.** In augment mode `memory` carries `max_sessions`,
`admitted_sessions` and `sessions_at_capacity` beside `active_sessions`.
`MemoryProxy.entry` refuses a *new* conversation with a 429 once the registry
reaches `max_sessions`, and it evicts nothing, so health must say so before that
happens rather than after: `sessions_at_capacity` raises
`memory_sessions_at_capacity` in `health.reasons` and the status becomes
`degraded`. Conversations already open keep working, which is why this is
`degraded` and not `failed`. `admitted_sessions` counts the registry that gates
admission, not the sessions whose status happened to be readable, so the signal
cannot report room that does not exist. See
`01 - Design/05 - Local Model Handoff and Concurrent Linking.md`, package A.

**`serving` units.** `static_tokens_last` and `served_tokens_last` are always
in the same unit. When the prompt was *not* rewritten the upstream's exact
`usage` count describes both sides, so observe mode and an augment passthrough
show equal figures and zero savings rather than a gap made of estimator
error. Only a real rewrite has two different prompts to compare, and there the
client's side is the shared char/4 estimate against the upstream's exact count
— a conservative pairing, since char/4 tends to undercount, so
`saved_tokens_total` understates rather than overstates. A null
`served_tokens_last` means unknown (an idempotent ledger replay sent nothing),
never zero, and `requests`/`rewritten` count only requests that were actually
served — a failed or refused request leaves `serving` untouched.

## Startup and readiness (augment mode)

A native session is not usable the moment the proxy can answer HTTP. It must
construct its models — the resident encoder and the attention cache, whose
cold build has been observed at 23–25 minutes — before any recall is real.
Two rules follow.

**Status must never wait on the work it describes.** Model construction is
serialised by `NativeProxySessions.model_start_lock`, and a session is
registered under `MemoryProxy.lock` only after its journal has been replayed.
Neither lock may appear on a status path. `summary_status()` and
`startup_status()` read a registry guarded by a separate `registry_lock` that
is only ever held for a dictionary access; `MemoryProxy.status()` reads
mirrors of the live and opening sessions under its own `_status_lock` and
takes `MemoryProxy.lock` not at all. Lifecycle stages are recorded *before*
the wait they describe, so a blocked start reads as blocked rather than
missing.

**An empty capture queue is not an idle service.** `queue.*` is an
observe-mode structure and stays empty in augment mode, so top-level `state`
additionally consults the native service. `starting` is published while any
session is pre-ready; `ingesting` while native work is outstanding with every
session ready. This is the one deliberate addition to schema 1: a renderer
that does not know `starting` falls through to printing the state name.

Stages, in order: `constructing`, `admitting`, `awaiting_model_start`,
`opening_models`, `ready`, and `failed`. They appear under `memory.startup`
keyed by namespace, each with `stage`, `error`, `elapsed_s` and
`stage_elapsed_s`; sessions still inside `MemoryProxy.entry()` appear under
`memory.opening_sessions` with their waited seconds. Totals and an ETA are
unknown during a cold load — the stage name and elapsed time are what tell a
waiting user the service is alive. `memory.busy` summarises outstanding
native work, and `memory.sessions` keeps the per-session detail so
aggregation never hides which session is blocked.

## Bulk transcript loading

"Loading an entire transcript" is a job, not a stream of captures. A
`BulkLoader` takes an on-disk transcript (located and memory-mapped through
`live_sessions`, which never writes the original), knows `bytes_total` up
front and `records_total` after one index pass, and feeds records through the
**same capture queue** as live traffic so queue counters stay coherent. It
publishes progress into `current`/`rates`. Jobs are queued FIFO; `queue.jobs`
lists them with positions.

## Control API

Local-only routes on the proxy (bound to the proxy's host, same port):

| Route | Effect |
| --- | --- |
| `GET /_control/status` | the status document (also readable from disk without the proxy) |
| `POST /_control/save {"name": optional}` | atomic snapshot; returns the record. **Observe**: one `VACUUM INTO <data-dir>/snapshots/<ts or name>.sqlite`. **Augment**: a directory `<data-dir>/snapshots/<ts or name>/` holding `manifest.json` (schema 1, `kind: "sessions"`, namespaces, per-file list) plus a `sessions/` mirror — each database copied with `VACUUM INTO`, WAL/SHM/journal and lock files excluded |
| `POST /_control/load {"name"}` | swap the store to a snapshot. State → `swapping`; captures are queued, not dropped. **Observe**: requests are served passthrough during the swap. **Augment**: every session store is closed and the tree replaced, then conversations are re-admitted lazily on their next request, so a request arriving mid-swap waits rather than seeing a half-swapped tree |
| `POST /_control/reset {"confirm": true}` | automatic snapshot first, then re-initialize an empty store (both modes). Refused without `confirm` |
| `POST /_control/load-transcript {"session": id-or-path}` | enqueue a bulk job. **Observe mode only** — `BulkLoader` feeds the observe `CaptureQueue`, which augment does not use; returns 501 in augment |

Mutating routes require `confirm` or an explicit name; they log to the
proxy's receipts. Nothing on a passive surface can reach them.

## Surfaces

| Surface | Client | Reads / calls | What it shows |
| --- | --- | --- | --- |
| **Statusline** (`statusLine` setting) | Claude Code | `status.json` | one line, refreshed by the client: `mem ▸ ingest 52% (ETA 1m38s) · 1.21M tok · q:1 · serving 48k of 1.10M` |
| **Hook `systemMessage`** (`UserPromptSubmit`) | Claude Code, Codex | `status.json` | user-visible note **only when state changed since last turn** (ingest started/finished, snapshot taken, reset). Never continuous; never enters model context |
| **`/memory-status`** command | Claude Code | CLI `ctl status` | full table on demand: queue, job progress, rates, totals, snapshots |
| **CLI** `memory-condense ctl status\|save\|load\|reset\|load-transcript` | both | `status.json` for `status`; control API for the rest | the control plane; `reset` requires `--yes`. The CLI keeps `load` while the MCP tool is `memory_snapshot_restore`: the asymmetry is deliberate — `ctl load <name>` is typed deliberately by an operator, whereas a model picking from a tool list needed the destructive one to be unmistakable |
| **MCP tools** `memory_snapshot_save`, `memory_snapshot_restore`, `memory_reset`, `memory_load_transcript`, `memory_stats` | both | control API; `memory_stats` reads `status.json` | conversational access ("reset the memory system"); tool descriptions instruct confirmation before `memory_reset`. These four act **only on the proxy's store**, while `recall`/`remember`/`forget`/`supersede` act only on the MCP server's own — the two resolve directories independently (`$MEMORY_CONDENSE_DATA_DIR` → `$CLAUDE_PROJECT_DIR/.memory_condense` → `./.memory_condense` vs. `user_directory()/data`) and `memory_stats` lists them separately |
| **`GET /_control/status`** | any | — | a browser tab when not at the terminal |

Codex has no statusline; it gets the hook note, the CLI, and the MCP tools.
Its `notify` setting can raise a desktop notification when a bulk job finishes.

## Why this shape

- **Single writer.** The proxy owns the queue and the store, so it is the only
  process that can report them truthfully; renderers that compute would
  drift.
- **Passive surfaces are read-only by construction.** The statusline and
  hooks read a file; they cannot mutate even by accident.
- **Backup before destroy.** `reset` snapshots first; `load` serves
  passthrough during the swap so no inference request sees a half-swapped
  store. A swap whose mutation fails **reopens the store** and the state
  returns to `idle` — only a store that did not come back publishes
  `error`, and the 500 body carries `store_live` so the operator can tell
  the two apart. In augment mode recovery is implicit: sessions are
  re-admitted lazily, so a failed mutation leaves the files intact and the
  next request simply opens them again.
- **Nothing enters model context** unless deliberately routed through
  `additionalContext`. Status is for the user.

## Open questions

- **Two proxies.** `context_proxy/` (stdlib, pair-safe unit trimmer, bridge to
  on-disk originals, built 2026-09-22) overlaps `interfaces/proxy_server.py`
  (Starlette, capture + augment, receipts, backpressure). The serving path is
  the existing proxy; the control plane publishes from it. What of
  `context_proxy` to fold in (the unit trimmer, the bridge, the ledger
  semantics) versus retire is a consolidation decision — see
  `06 - Roadmaps/06 - Proxy Consolidation.md`. Its two load-bearing
  findings: augment mode rejected clients that lack
  `x-memory-conversation-id` (Claude Code and Codex send none), and it
  buffers the upstream response instead of streaming it. The first is
  **resolved**: `interfaces/proxy_sessions.py` resolves a conversation id
  without a header (explicit header → the client's own session id → the
  on-disk transcript whose first user message matches → the request
  digest), and the proxy injects the resolved id for the memory service.
  Response buffering still stands and must land before the existing proxy
  can serve those clients.
- **Statusline/hook installation** edits the user's `settings.json`; the
  implementation ships the scripts and an install snippet and does not touch
  live settings.

## Work packages (batched to agents, file ownership is exclusive)

| Package | Owns | Builds against |
| --- | --- | --- |
| **A — control core** | `src/memory_condense/control/{status,snapshots,bulk,publisher}.py` (`snapshots.py` owns both `SnapshotStore` for observe and `SessionSnapshotStore` for augment), additive wiring in `interfaces/proxy_server.py`, the augment swap/quiesce mechanics in `interfaces/proxy_memory.py`, `tests/test_control.py` | this note's schema and routes |
| **B — CLI + MCP** | `src/memory_condense/control/cli.py`, the `ctl` subcommand in `cli.py`, new tools in `interfaces/mcp_server.py`, `tests/test_control_cli.py` | the schema and routes (not A's internals) |
| **C — client surfaces** | `tools/memory_statusline.py`, `.claude/commands/memory-status.md`, `.claude/hooks/memory_state_note.py`, `docs/02 - Implementation/NN - Installing memory controls.md` | `status.json` only |
| **D — proxy consolidation (read-only)** | `docs/06 - Roadmaps/06 - Proxy Consolidation.md` | both proxies' code |
