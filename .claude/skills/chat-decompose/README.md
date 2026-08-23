# chat-decompose

A reusable Claude Code skill that turns a long LLM-conversation export
(a single `.md` transcript, typically 100k–10M chars) into:

- a **per-phase folder tree** with one file per conversation turn (full
  byte-level provenance via YAML frontmatter and SHA-256 hashes),
- an **imperative dev guide** (one chapter per phase, written as the
  design endpoint),
- **architecture decision records** in MADR-lite format,
- **operational manifests** (`turns.json`, `phases.json`,
  `decisions.json`, `reconciliation.md`) that are the source-of-truth
  for the decomposition.

The original transcript is **never edited**. Every artifact is
regenerable from the manifests.

## What you get

| Artifact | Where | Produced by |
|----------|-------|-------------|
| `source.sha256`, `turns.json` | `<output-dir>/manifests/` | Stage 0 (script) |
| `phases.json`, `decisions.json`, `reconciliation.md` | `<output-dir>/manifests/` | Stage 1 (agents) |
| `phase-NN-<slug>/turn-NNN-<role>.md` | `<output-dir>/raw/` | Stage 2 (script) |
| `dev-guide/00-overview.md`, `dev-guide/NN-<slug>.md` | `<docs-dir>/` | Stage 3 (agents) |
| `decisions/NNNN-<slug>.md`, `decisions/README.md` | `<docs-dir>/` | Stage 4 (agents) |
| `<docs-dir>/README.md`, root `README.md` | repo root | Stage 5 (orchestrator) |

A `verify.py` script runs 9 end-to-end checks (hash integrity, line
coverage, cross-references, link integrity, idempotency).

## Pipeline at a glance

```
source.md  ──►  Stage 0 ──►  Stage 1 ──►  Stage 2 ──►  Stage 3 ──►  Stage 4 ──►  Stage 5
                (script)     (agents)     (script)     (agents)     (agents)     (manual)
```

Scripts are deterministic (re-running produces byte-identical output);
agent stages produce semantically equivalent but not byte-identical
output across runs.

## Copy-paste install

This skill ships as a self-contained tree under `.claude/`. Drop it
into any project root with `.claude/` already initialized. Three things
need to be merged in:

```
your-project/
└── .claude/
    ├── skills/
    │   └── chat-decompose/        ← copy this entire folder
    ├── agents/
    │   ├── chat-decompose-clusterer.md      ← copy these four files
    │   ├── chat-decompose-reconciler.md
    │   ├── chat-decompose-chapter.md
    │   └── chat-decompose-adr.md
    └── commands/
        └── decompose-chat.md       ← copy this file
```

If your project doesn't already have `.claude/`, just copy the whole
`.claude/` directory.

The skill needs **Python 3.11+** for the scripts. If your project
already uses pixi, uv, poetry, or plain `python3`, the scripts will
work — they have no dependencies beyond the standard library.

## Running it

### Inside Claude Code

```text
/decompose-chat path/to/your-export.md
```

The slash command invokes the skill, which sniffs the format, runs all
five stages (orchestrating subagent dispatches), and finishes with a
verify run. Confirms the output paths before overwriting.

### Or in natural language

> "Decompose this chat export at `transcripts/team-design-call.md`
> into a doc tree."

The skill auto-triggers on phrasings like "decompose", "break down",
"extract a doc tree from", "turn this transcript into ADRs".

### Or directly via the scripts

You can also run the deterministic stages by hand without invoking
agents:

```bash
# Stage 0
python .claude/skills/chat-decompose/scripts/build_turn_index.py \
    --source path/to/export.md \
    --output-dir _ingest

# (then use Claude or another LLM to produce phases.json + decisions.json)

# Stage 2
python .claude/skills/chat-decompose/scripts/split_into_phases.py \
    --source path/to/export.md \
    --output-dir _ingest

# Verify any time
python .claude/skills/chat-decompose/scripts/verify.py \
    --source path/to/export.md \
    --output-dir _ingest \
    --docs-dir docs
```

## Configuring for non-standard transcript formats

The Stage 0 script defaults match the most common Claude/ChatGPT
markdown export format:

- Turn delimiter: `^---\s*$` (a `---` line outside code fences)
- Role heading: `^##\s+(User|Assistant)\s*$`

Other formats are supported via CLI flags. Examples:

```bash
# ChatGPT export with **You:** / **ChatGPT:**
--turn-delim '^\s*$' \
--role-pattern '^\*\*(You|ChatGPT):\*\*'

# Slack-style with H3 headings
--role-pattern '^###\s+(User|Assistant):'
```

The skill orchestrator will sniff the first 200 lines of the source
and propose appropriate flags before running.

## What the agents do

Five agent dispatches across the pipeline:

1. **Three parallel `chat-decompose-clusterer` agents** in Stage 1a —
   each reads a third of the merged turns and proposes phase boundaries
   + decision moments as JSON.
2. **One `chat-decompose-reconciler` agent** in Stage 1b — merges the
   three proposals into canonical `phases.json`, `decisions.json`, and
   `reconciliation.md`.
3. **One `chat-decompose-chapter` agent per phase** in Stage 3 (up to 3
   in parallel per round) — writes one imperative dev-guide chapter
   per phase.
4. **`chat-decompose-adr` agents** in Stage 4 (up to 3 in parallel,
   each may write 1–4 ADRs) — write the MADR-lite ADRs.

If those subagent definitions aren't installed, the skill falls back
to `general-purpose`/`Explore`/`Plan` agents with templated prompts.

## Why agents AND scripts?

- **Scripts where bytes matter.** Turn segmentation, phase splitting,
  and verification are deterministic — any LLM disagreement with the
  regex would be silent data loss.
- **Agents where meaning matters.** Phase clustering, chapter writing,
  and ADR extraction need judgment about topical seams, design
  endpoints, and rejected alternatives — exactly what an LLM does
  well.
- **Manifests as the contract between layers.** Scripts and agents both
  read and write `manifests/*.json`. The manifests are the project's
  schema; everything else is generated from them. Validate them
  aggressively (cross-reference checks, contiguity checks) because
  every downstream artifact depends on them.

## Re-running

| Stage | Command | Notes |
|-------|---------|-------|
| 0 | `python build_turn_index.py ...` | Always safe; byte-stable. |
| 1 | re-dispatch the three proposal agents + reconciler | LLM output is non-deterministic; expect minor phase-boundary drift. Inspect `phases.json` diff before committing. |
| 2 | `python split_into_phases.py ...` | Always safe; wipes and rebuilds `raw/`. |
| 3 | re-dispatch a single chapter agent per phase | Each chapter is independent — re-run only the one you want to refresh. |
| 4 | re-dispatch a single ADR agent per ADR | Same. |
| 5 | hand-edit the top-level READMEs | Trivial; do once. |

## Common gotchas

- **Multiple same-role blocks per logical turn.** Many exporters split
  a single assistant reply into multiple `## Assistant` blocks (one
  per text segment between tool calls). The merged-turn collapse
  handles this — operate on `merged_turns` for phase reasoning, on
  sub-turns for byte-level addressing.
- **`---` inside code fences.** The script ignores delimiters inside
  ` ``` ` / `~~~` fences. If the export uses different fence markers,
  pass a custom regex.
- **Turn-counting math.** N delimiters with both preamble and postamble
  produce N-1 turns (not N/2). Don't assume.
- **Plan-type subagents are read-only.** They cannot `Write`; they
  return content as text. Use `general-purpose` (or the
  `chat-decompose-chapter`/`chat-decompose-adr` subagents in this
  skill) for any agent that must produce files.
- **Cross-reference validation is mandatory.** After Stage 1b,
  validate that every decision's turn refs fall inside the phase it
  claims. Three real cross-reference bugs were caught in the
  bootstrap run of this pipeline.

## Layout (after running)

```
your-project/
├── path/to/export.md                 ← source-of-truth, never edited
├── _ingest/
│   ├── manifests/
│   │   ├── source.sha256
│   │   ├── turns.json
│   │   ├── phases.json
│   │   ├── decisions.json
│   │   └── reconciliation.md
│   └── raw/
│       └── phase-NN-<slug>/
│           ├── 00-overview.md
│           └── turn-NNN-<role>.md
└── docs/
    ├── README.md
    ├── dev-guide/
    │   ├── 00-overview.md
    │   └── NN-<slug>.md
    └── decisions/
        ├── README.md
        └── NNNN-<slug>.md
```

The skill's pipeline architecture is documented in detail in
`SKILL.md`. The bootstrap project that produced this skill (an
`llmb_rts_notebook` design conversation) is available as a worked
example.
