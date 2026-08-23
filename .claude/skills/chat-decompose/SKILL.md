---
name: chat-decompose
description: Decompose a long LLM-conversation export (markdown transcript) into a navigable nested document tree, an imperative dev guide, and a set of architecture decision records. Use when the user asks to "decompose / break down / split / extract a doc tree from a chat export / conversation transcript / .md transcript / LLM conversation / interview / design conversation", or runs `/decompose-chat`.
---

# Skill: chat-decompose

## What this does

Turns a single large markdown transcript into:

- A **per-phase folder tree** under `<output-dir>/raw/` with one file per
  conversation turn, each carrying YAML frontmatter for byte-level
  provenance.
- An **imperative dev guide** under `<docs-dir>/dev-guide/` — one chapter
  per phase, written as the design endpoint (not as a Q&A recap).
- **Architecture decision records** under `<docs-dir>/decisions/` —
  MADR-lite ADRs with context, decision, consequences, and alternatives
  considered.
- **Manifests** under `<output-dir>/manifests/` (`turns.json`,
  `phases.json`, `decisions.json`, `reconciliation.md`, `source.sha256`)
  that are the operational source-of-truth for the decomposition.

The original transcript is **never edited**. Every output is regenerable
from the manifests.

## When to use

Trigger this skill when the user has a markdown transcript of a long
LLM conversation (or design discussion, interview, etc.) and wants it
decomposed into navigable docs. Common phrasings: "break this down",
"decompose this chat", "extract a doc tree from this conversation",
"give me ADRs from this transcript", or `/decompose-chat <path>`.

## Inputs to ask the user for (if not given)

1. **Source file path** — the markdown transcript.
2. **Output directories** — defaults: `_ingest/` and `docs/` at repo
   root. Confirm before overwriting.
3. **Turn delimiter & role-heading patterns** — sniff from the file
   first; ask only if the sniffer is uncertain.

If the user invokes via `/decompose-chat <path>`, the path is the only
required input; everything else uses defaults.

---

## Pipeline overview

The pipeline has 5 numbered stages plus verification. **Scripts are
deterministic and re-runnable; agents produce semantically equivalent
output across runs but are not byte-identical.**

```
source.md
   |
   |  Stage 0: build_turn_index.py        (script, deterministic)
   v
manifests/source.sha256, manifests/turns.json
   |
   |  Stage 1: parallel proposal agents + reconciliation agent
   v
manifests/phases.json, manifests/decisions.json, manifests/reconciliation.md
   |
   |  Stage 2: split_into_phases.py       (script, deterministic)
   v
raw/phase-NN-<slug>/turn-NNN-<role>.md (one per sub-turn)
   |
   |  Stage 3: parallel chapter-writer agents
   v
<docs-dir>/dev-guide/NN-<slug>.md (one per phase) + 00-overview.md
   |
   |  Stage 4: parallel ADR-writer agents
   v
<docs-dir>/decisions/NNNN-<slug>.md + README.md
   |
   |  Stage 5: top-level wiring (manual)
   v
<docs-dir>/README.md, root README.md
   |
   |  Verification: verify.py
   v
ALL CHECKS PASSED
```

---

## Stage 0: turn index (script)

```bash
python <SKILL_DIR>/scripts/build_turn_index.py \
    --source <source.md> \
    --output-dir <output-dir> \
    [--turn-delim '<regex>'] \
    [--role-pattern '<regex>']
```

Defaults:

- `--turn-delim '^---\s*$'` (matches `---` on its own line outside code fences)
- `--role-pattern '^##\s+(User|Assistant)\s*$'` (group 1 is the role name)

**Sniffer step before invoking:** read the first ~200 lines of the
source. If it has `## User` / `## Assistant`, defaults work. If it uses
a different format (`**You:**`, `### Human:`, etc.), pass appropriate
regexes. If unsure, ask the user.

**Sanity-check the output before proceeding:** open `turns.json` and
confirm `stats.merged_alternation_ok == true`. If it's false,
inspect — usually means the role pattern is off, or the export emits
multiple same-role blocks per logical turn (the merged-turn collapse
handles this; if alternation still fails the regex needs adjustment).

---

## Stage 1: phase clustering (parallel agents)

### 1a — three parallel proposal agents

Read `manifests/turns.json`. The merged turn count is `stats.merged_turn_count`. Split it into thirds and dispatch three agents in **parallel** (single message, three Agent calls):

| Agent | Range                | Use template |
|-------|----------------------|--------------|
| A     | turns 001–N/3        | `templates/phase_proposal.md` |
| B     | turns N/3+1–2N/3     | same template |
| C     | turns 2N/3+1–N       | same template |

**Subagent type to use:**
- If `chat-decompose-clusterer` is available, dispatch with `subagent_type="chat-decompose-clusterer"`.
- Otherwise dispatch as `general-purpose` (or `Explore` for read-only research).

Each agent returns a JSON proposal: `{ proposed_phases: [...], decision_moments: [...] }`. Cap responses at 600 words.

### 1b — reconciliation

Single sequential agent (after 1a returns). Use `templates/phase_reconcile.md`. Embed all three proposals in the prompt. Output:

- `manifests/phases.json` — 6–9 canonical phases, contiguous coverage.
- `manifests/decisions.json` — deduplicated decision list, tagged `PIVOT` / `LOCK-IN` / `SCOPE-CUT`.
- `manifests/reconciliation.md` — narrative of merge choices.

**Subagent type:**
- `chat-decompose-reconciler` if available, else `Plan` agent type.

**Validate before Stage 2:** run a quick consistency check (every decision's `phase_id` and `merged_turn_refs` fall inside the corresponding phase's `merged_turn_range`; phase ranges are contiguous and cover 001..N). Fix manifest errors before splitting.

---

## Stage 2: physical split (script)

```bash
python <SKILL_DIR>/scripts/split_into_phases.py \
    --source <source.md> \
    --output-dir <output-dir>
```

Wipes and rebuilds `<output-dir>/raw/`. Each phase folder gets a `00-overview.md` linking to the phase's sub-turns and listing the decisions made in that phase.

---

## Stage 3: dev-guide chapters (parallel agents)

For each phase in `phases.json`, dispatch one agent using `templates/chapter.md`. Run **up to 3 in parallel per round**. Each agent reads its phase's raw folder and writes one chapter to `<docs-dir>/dev-guide/NN-<slug>.md`.

**Subagent type:** `chat-decompose-chapter` if available, else `general-purpose` (Plan agents are read-only — they can't `Write`).

Voice: imperative description of the design endpoint. Reversals appear only as short "Why not X" subsections where the reversal is load-bearing.

After all chapters are written, hand-write `00-overview.md` (~100 lines) in `<docs-dir>/dev-guide/` linking the chapters in order.

---

## Stage 4: decision records (parallel agents)

For each decision in `decisions.json`, dispatch one agent using `templates/adr.md` to write `<docs-dir>/decisions/NNNN-<slug>.md` in MADR-lite format. Up to 3 in parallel per round.

**Subagent type:** `chat-decompose-adr` if available, else `general-purpose`.

After all ADRs are written, hand-write `<docs-dir>/decisions/README.md` as the index table (ID, title, tag, phase, source turns).

---

## Stage 5: top-level wiring (manual)

Write:

- `<docs-dir>/README.md` — project intro + links to dev guide + ADR index.
- Root `README.md` — one-paragraph "what is this" + link to docs.

These are short hand-written files; no agent dispatch needed.

---

## Verification

```bash
python <SKILL_DIR>/scripts/verify.py \
    --source <source.md> \
    --output-dir <output-dir> \
    --docs-dir <docs-dir>
```

Runs 9 checks: source hash unchanged, line coverage, raw body hashes match, phase↔decision cross-refs, phase contiguity, folders/chapters/ADRs all present, link integrity. Exit code 0 if all pass.

---

## Output schema reference

### `manifests/turns.json`

```json
{
  "source": { "path": "...", "sha256": "...", "byte_count": N, "line_count": N },
  "config": { "turn_delim": "...", "role_pattern": "..." },
  "stats":  { "turn_count": N, "merged_turn_count": N, "merged_alternation_ok": true, ... },
  "merged_turns": [ { "merged_id": "001", "role": "user", "sub_turn_ids": ["001"], "line_start": ..., "line_end": ..., ... } ],
  "turns":        [ { "turn_id": "001", "merged_id": "001", "role": "user", "hash": "...", ... } ],
  "extras":       [ { "kind": "preamble"|"postamble", ... } ]
}
```

### `manifests/phases.json`

```json
{
  "phases": [
    {
      "phase_id": "01",
      "slug": "kebab-case",
      "name": "Human-readable name",
      "merged_turn_range": ["001", "008"],
      "summary": "1-3 sentences",
      "decision_ids": ["DR-0001"],
      "source_proposals": ["A.phase-1"],
      "notes": "reconciliation note"
    }
  ]
}
```

### `manifests/decisions.json`

```json
{
  "decisions": [
    {
      "decision_id": "DR-0001",
      "slug": "kebab-case",
      "title": "Imperative title",
      "merged_turn_refs": ["006", "007"],
      "phase_id": "02",
      "tag": "PIVOT" | "LOCK-IN" | "SCOPE-CUT",
      "one_line": "Concise statement of the decision."
    }
  ]
}
```

---

## Defaults (override with explicit user request)

- ADR format: **MADR-lite** (status / context / decision / consequences / alternatives / source).
- Folder split: `<output-dir>/` (default `_ingest/`) for raw + manifests + procedure; `<docs-dir>/` (default `docs/`) for polished dev guide + ADRs. Source transcript stays at repo root, **never edited**.
- Dev-guide voice: imperative description of the *current* design; reversals captured in ADRs, not in chapter prose.
- Max 3 agents in parallel per round (matches harness guidance).
- Phase count: aim for 6–9 canonical phases; raw proposals usually over-fragment to 12–15.

## Common gotchas

- **Multiple same-role blocks per logical turn.** Many exporters split a single assistant reply into multiple `## Assistant` blocks (one per text segment between tool calls). The merged-turn collapse handles this — operate on `merged_turns` for phase reasoning, on sub-turns for byte-level addressing.
- **`---` inside code fences.** The script ignores delimiters inside `\`\`\`` / `~~~` fences. If the export uses different fence markers, adjust the script.
- **Turn-counting math.** N delimiters with both preamble and postamble produce N-1 turns (not N/2). Don't assume.
- **Plan-type subagents are read-only.** They cannot `Write`; they will return content as text. Use `general-purpose` for any agent that must produce files.
- **Cross-reference validation is mandatory.** After Stage 1b, validate that every decision's turn refs fall inside the phase it claims. Three real cross-reference bugs were caught and fixed in the original run.

## How to invoke

- Slash command: `/decompose-chat <path-to-source.md>`
- Or natural language: "decompose this chat export at <path>"
- The skill orchestrator will sniff the format, confirm the output paths, then run all five stages plus verification.
