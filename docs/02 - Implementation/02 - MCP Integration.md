# MCP Integration — using memory_condense from Claude Code

**Status**: CURRENT
**Date**: 2026-08-14
**Applies to**: branch `feat/memory-layer`
**Depends on**: `00 - Setup and Environment.md` (pixi env), `03 - Architecture/00` (what the tools wrap)

The memory system is exposed to Claude Code — and any other MCP client — as a stdio MCP server. The integration has two halves, and they do deliberately different jobs:

| Half | Mechanism | What it does | Cost |
| --- | --- | --- | --- |
| **Active** | MCP server ([`mcp_server.py`](../../src/memory_condense/mcp_server.py)) | Seven tools the model calls when it decides memory is relevant. Full semantic + keyword retrieval. | One model load per server process (lazy) |
| **Passive** *(opt-in)* | `UserPromptSubmit` hook ([`memory_context_hook.py`](../../examples/claude_hooks/memory_context_hook.py)) | Prepends pinned and still-hot facts to every prompt. | Single-digit ms; **no embedding model at all** |

The split is forced by how each runs. The MCP server is one long-lived process, so it can afford to hold bge-m3 in memory. A hook is a **fresh process on every prompt**, so semantic search there would reload a 2.3 GB model per turn. The hook therefore ranks by pin state and decayed energy only — pure SQLite. Query-specific recall is the `recall` tool's job, not the hook's.

## 1. The tools

Registered under the server name `memory_condense`, so the model sees them as `mcp__memory_condense__<tool>`.

| Tool | Purpose |
| --- | --- |
| `remember(content, type, details, pin)` | Store a durable fact. `type` ∈ Decision, Preference, Constraint, Entity, Definition, Task, Correction |
| `recall(query, limit)` | Ranked retrieval over stored facts, with the score breakdown (relevance / importance / recency) |
| `search(query, limit, hybrid)` | Search the raw ingested transcript rather than curated facts; `hybrid` blends BM25 with semantic similarity |
| `ingest(text, role)` | Chunk, embed, index, and auto-extract facts from a block of text |
| `memory_stats()` | Store location, turn count, active/pinned counts, heat distribution |
| `pin_memory(mem_id, pinned)` | Exempt a fact from decay. Accepts the 8-character short id |
| `forget(mem_id)` | Soft delete — the row and its provenance survive so the audit trail stays walkable |

**Provenance still applies through MCP.** `remember` writes the content as a transcript turn and cites it, so the item passes the same validator as everything else. A fact with no traceable source cannot enter the store by any path — see clause 8 of `05 - Standards/00`.

## 2. Registration

The repo ships a **project-scoped** [`.mcp.json`](../../.mcp.json), so anyone who clones the repo and opens Claude Code here is prompted to approve the server — no per-machine setup:

```json
{
  "mcpServers": {
    "memory_condense": {
      "type": "stdio",
      "command": "pixi",
      "args": ["run", "python", "-m", "memory_condense.mcp_server"]
    }
  }
}
```

To register it for **every** project instead of just this one:

```powershell
claude mcp add --scope user memory_condense -- pixi run python -m memory_condense.mcp_server
```

The `--` is required: everything after it is passed to the server untouched. Manage it with `claude mcp list`, `claude mcp get memory_condense`, `claude mcp remove memory_condense`.

**No `env` block is needed, and that is deliberate.** The MCP stdio client does not forward the parent environment to the server, so the server resolves its store from the working directory instead — see below. Hardcoding an absolute path here would break the file for everyone else who clones the repo.

## 3. Where memory is stored

Resolved in this order:

1. `$MEMORY_CONDENSE_DATA_DIR`
2. `$CLAUDE_PROJECT_DIR/.memory_condense`
3. `./.memory_condense`

In practice Claude Code launches a project-scoped server with the project as its working directory, so **each project gets its own memory** at `<project>/.memory_condense/`. That directory is gitignored — memory is local, never committed.

To share one store across projects, set `MEMORY_CONDENSE_DATA_DIR` in the server's `env` block in a **user-scoped** registration (where an absolute path is appropriate).

## 4. Gotchas

1. **The first tool call is slow on a cold machine** — bge-m3 is ~2.3 GB and downloads on first embedding. `memory_stats` does *not* embed, so it is a safe first call to confirm the server is alive. The model is loaded lazily, so listing tools never triggers it.
2. **stdout is the JSON-RPC channel.** All server logging goes to stderr. Anything that prints to stdout inside a tool will corrupt the protocol — do not add `print()` to `mcp_server.py`.
3. **The hnswlib index is saved on clean shutdown** (`atexit`). If the server is hard-killed, the `.bin` is stale — but SQLite is the source of truth, so `retriever.rebuild_index()` recovers it. This is clause 1 of the data standard working as designed.
4. **`ingest` runs extraction; the eval harness does not.** The MCP server uses the default `auto_extract=True` because its whole purpose is building memory. `runner.py` and `benchmark.py` pass `auto_extract=False` because their prompts read chunks, not memory items.
5. **`recall` reheats what it returns** (access reheating). The hook deliberately does not, so passive injection cannot quietly defeat the decay model.

## 5. Optional: the passive hook

Not enabled by default — it injects context into every prompt, so turning it on is a decision for the user, not a default. To enable, add to `.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "pixi run python examples/claude_hooks/memory_context_hook.py"
          }
        ]
      }
    ]
  }
}
```

It emits `hookSpecificOutput.additionalContext` with up to six facts, pinned first. It **fails open**: any error exits 0 with no output, so a missing or corrupt store can never block a prompt.

There is deliberately **no auto-capture hook**. Recording every prompt would grow the store without discrimination and is a privacy decision that should be made explicitly, not inherited from a default. If you want it, a `Stop` hook calling `ingest` is the shape — but consider whether `remember` on the facts that matter is better than ingesting everything.

---

**Verification block**: with the server registered, run

```powershell
claude mcp list                              # expect memory_condense listed and connected
pixi run -e dev pytest tests/test_mcp_server.py -q   # expect 28 passed
```

Then, in a Claude Code session, ask it to store and retrieve a fact — the round trip through `remember` → `recall` is the real check. If `claude mcp list` shows the server as failed, run `pixi run python -m memory_condense.mcp_server` directly: it should start, log to stderr, and wait on stdin rather than exiting.
