# MCP Integration — using memory_condense from Claude Code

**Status**: CURRENT
**Date**: 2026-08-14
**Applies to**: `main` (merged in `f3edc91`)
**Depends on**: `00 - Setup and Environment.md` (pixi env), `03 - Architecture/00` (what the tools wrap)

The memory system is exposed to Claude Code — and any other MCP client — as a stdio MCP server. The integration has two halves, and they do deliberately different jobs:

| Half | Mechanism | What it does | Cost |
| --- | --- | --- | --- |
| **Active** | MCP server ([`interfaces/mcp_server.py`](../../src/memory_condense/interfaces/mcp_server.py)) | Eight tools the model calls when it decides memory is relevant. Full semantic + keyword retrieval. | One model load per server process (lazy) |
| **Passive** *(opt-in)* | `UserPromptSubmit` hook ([`memory_context_hook.py`](../../examples/claude_hooks/memory_context_hook.py)) | Prepends pinned and still-hot facts to every prompt. | Single-digit ms; **no embedding model at all** |

The split is forced by how each runs. The MCP server is one long-lived process, so it can afford to hold bge-m3 in memory. A hook is a **fresh process on every prompt**, so semantic search there would reload a 2.3 GB model per turn. The hook therefore ranks by pin state and decayed energy only — pure SQLite. Query-specific recall is the `recall` tool's job, not the hook's.

## 1. The tools

Registered under the server name `memory_condense`, so the model sees them as `mcp__memory_condense__<tool>`.

| Tool | Purpose |
| --- | --- |
| `remember(content, type, details, pin)` | Store a durable fact. `type` ∈ Decision, Preference, Constraint, Entity, Definition, Task, Correction |
| `recall(query, limit)` | Ranked retrieval over stored facts, with the score breakdown (relevance / importance / energy) |
| `search(query, limit, hybrid)` | Search the raw ingested transcript rather than curated facts; `hybrid` blends BM25 with semantic similarity |
| `ingest(text, role)` | Chunk, embed, index, and auto-extract facts from a block of text |
| `memory_stats()` | Store location, turn count, active/pinned counts, heat distribution |
| `pin_memory(mem_id, pinned)` | Exempt a fact from decay. Accepts the 8-character short id |
| `supersede(mem_id, content, type, details)` | Replace a revised fact, keeping the old row and the link back to it |
| `forget(mem_id)` | Soft delete — the row and its provenance survive so the audit trail stays walkable |

### Witnessed vs. asserted provenance

Every path into the store goes through the same validator, but on the MCP path **that guarantee is weaker than it looks, and the tools say so out loud.**

`remember` first looks for an existing turn containing the content verbatim. If one exists, the memory cites *that* turn — real evidence, traceable to something the user actually said — and the reply is tagged `witnessed in the transcript`. If none exists, the content is recorded as its own source turn and the reply is tagged `asserted`.

An asserted memory satisfies clause 8 (it has a turn and an exact quote) but proves nothing: the model wrote both the claim and the source. Calling that "provenance-enforced" would be circular. The distinction is surfaced in the tool output rather than hidden, so a reader can tell which memories rest on evidence. **Auto-extraction is the path where the validator can genuinely fail** — there the extractor quotes turns it did not write, and a paraphrase is rejected.

`supersede` is a first-class tool because `remember` + `forget` is *not* equivalent: it produces two unrelated rows and destroys the `supersedes` link the audit trail depends on. The `forget` docstring points at `supersede` for revisions so the model does not learn the lossy workflow.

## 2. Registration

The repo ships a **project-scoped** [`.mcp.json`](../../.mcp.json), so anyone who clones the repo and opens Claude Code here is prompted to approve the server — no per-machine setup:

```json
{
  "mcpServers": {
    "memory_condense": {
      "type": "stdio",
      "command": "pixi",
      "args": ["run", "python", "-m", "memory_condense.interfaces.mcp_server"]
    }
  }
}
```

To register it for **every** project instead of just this one:

```powershell
claude mcp add --scope user memory_condense -- pixi run python -m memory_condense.interfaces.mcp_server
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

## 6. Why there is no auto-capture hook (and what would actually be needed)

There is no hook that records the conversation automatically. An earlier revision of this file said the reason was that "deciding what is worth keeping is unsolved", then corrected that to say selection was fully built. **Both were wrong**, and `08 - Analysis/01` measured why.

Selection is *designed* — and the design is sound — but two of its four mechanisms are not connected to anything:

| Concern auto-capture raises | Designed answer | As built |
| --- | --- | --- |
| The store fills with junk | `importance` seeds energy; unused items decay to COLD | ✅ **Wired 2026-08-14.** Decayed energy is the scalar's fourth term (`wE = 0.2`), replacing a `recency` term that evaluated to a constant. Nothing is evicted — decay demotes, it does not delete |
| Junk survives by being re-read | Reheating fires only when a query actually matched | ✅ Wired, and made saturating: each access closes 25% of the *remaining* headroom, with a 300 s refractory window, so energy is a rate estimator rather than a ratchet. Only a pin holds the top of the range |
| Important facts get buried | Pins are exempt from decay; `w_P` boosts them | ✅ Built and wired |
| Context cost grows with the store | `ContextPacker` enforces a hard per-section budget | ✅ Built — and it is the *binding* constraint: 70.6% of items are dropped by the 900-token header before decay is ever consulted |
| A wrong fact persists | `supersede` keeps the chain; `forget` retires it | ✅ Built and wired |

So "ingest broadly and let decay sort it out" now describes retrieval as well as `decay.py` — but note what is still true: **the 900-token header remains the binding constraint**, and it is checked after ranking, not instead of it. Decay changes *which* items win the header; it does not change how many fit.

The extraction half is measured too: the default `RuleBasedExtractor` produces **65% `Constraint` and 93% assistant-sourced** items on a 4,554-turn corpus, because its `must|never|always|cannot` pattern fires on ordinary technical modality. `Decision` and `Preference` together account for 8 items out of 4,463.

### Turning on LLM extraction

`LLMExtractor` now has a provider binding. Set in the server's environment:

| Variable | Values | Default |
| --- | --- | --- |
| `MEMORY_CONDENSE_EXTRACTOR` | `rules` · `llm` · `auto` | `rules` |
| `MEMORY_CONDENSE_LLM_MODEL` | any litellm model string | `anthropic/claude-haiku-4-5` |

`auto` uses the LLM when a key resolves and rules otherwise. **The default stays `rules` on purpose**: auto-extraction fires on every `ingest`, so an LLM default would spend money on every tool call without being asked. The choice, including any fallback, is logged once per process to stderr — visible in the client's MCP logs — because a silent fallback looks like bad extraction rather than a missing key.

Nothing here can fail the server: a missing key, an unknown mode, or an SDK that will not import all return the rule-based extractor. That matters because, as §2 notes, the stdio client does not forward the parent environment, so **no key present is the normal case**. To pass one deliberately, add an `env` block to a user-scoped registration; `.env` in the working directory is also read.

The core package still imports no LLM SDK at module scope — `llm_provider` does `import litellm` inside the function that needs it, and `tests/test_architecture.py` enforces that over the AST rather than trusting a doc.

Consent remains genuinely a user decision: recording every turn of every session should not be inherited from a default.

If you want it now, a `Stop` hook calling `ingest` is the shape. Prefer `remember` on the facts that matter.

---

**Verification block**: with the server registered, run

```powershell
claude mcp list                              # expect memory_condense listed and connected
pixi run -e dev pytest tests/test_mcp_server.py -q   # expect 38 passed
```

Then, in a Claude Code session, ask it to store and retrieve a fact — the round trip through `remember` → `recall` is the real check. If `claude mcp list` shows the server as failed, run `pixi run python -m memory_condense.interfaces.mcp_server` directly: it should start, log to stderr, and wait on stdin rather than exiting.
