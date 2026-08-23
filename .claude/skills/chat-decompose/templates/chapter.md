# Dev-guide chapter prompt template

Substitute placeholders before dispatching:

- `{{PHASE_ID}}` — `01`, `02`, etc.
- `{{PHASE_SLUG}}` — kebab-case (e.g. `vega-rendering-substrate`).
- `{{PHASE_NAME}}` — human-readable phase name.
- `{{PHASE_SUMMARY}}` — 1-3 sentence summary from `phases.json`.
- `{{MERGED_TURN_RANGE}}` — e.g. `001-005`.
- `{{DECISIONS_LIST}}` — bulleted list of `DR-NNNN [TAG] — Title (turns ...)` for decisions in this phase. Empty list ok.
- `{{RAW_FOLDER}}` — absolute path to the raw phase folder.
- `{{DOCS_DIR}}` — absolute path to docs root (so the chapter's relative links resolve).
- `{{CHAPTER_LIST}}` — markdown list of all chapters with file names, for cross-reference.
- `{{PHASE_CHAR_COUNT}}` — total chars in phase (so the agent knows how aggressively to sample).

---

## Prompt body

You are writing chapter `{{PHASE_ID}}-{{PHASE_SLUG}}.md` of a dev guide. **Use the Write tool to write the file directly to disk.**

## Context

The project is decomposing a long LLM design conversation into a navigable doc tree. You are writing **one chapter** out of the full set:

{{CHAPTER_LIST}}

## Your phase

- **phase_id:** {{PHASE_ID}}
- **slug:** {{PHASE_SLUG}}
- **name:** {{PHASE_NAME}}
- **merged_turn_range:** {{MERGED_TURN_RANGE}}
- **summary:** {{PHASE_SUMMARY}}
- **decisions in this phase:**

{{DECISIONS_LIST}}

## Source

Read the raw turn files at:
`{{RAW_FOLDER}}`

The folder contains `00-overview.md` and a number of `turn-NNN-<role>.md` files. Phase total is roughly {{PHASE_CHAR_COUNT}} chars — sample appropriately. For large phases, focus on the lock-in turns and the user-driven topic pivots; you don't need every paragraph verbatim.

## Write the chapter to

`{{DOCS_DIR}}/dev-guide/{{PHASE_ID}}-{{PHASE_SLUG}}.md`

## Voice and structure

Rewrite as an **imperative description of the design as it stood at end of this phase** — not a Q&A, not a who-said-what narrative. The chapter is a dev-guide endpoint: it captures the design as locked at the end of the phase. State the design as fact, not as history. Reversals and abandoned approaches appear only as short "Why not X" subsections where the reversal is load-bearing.

If most of this phase's design **gets cut later** (e.g. a pre-V1 high-water-mark architecture that gets reduced in a later scope-cut), flag that at the top with a status note and a forward link.

## Required sections

1. **Purpose** — 1–2 sentences answering "what question does this chapter answer?".
2. **Design** — the current shape: components, interfaces, data, lifecycle. Use lists, tables, or short paragraphs as fits.
3. **Why this shape** — 1–3 short rationales, each tied to a constraint.
4. **Why not X** — alternatives that were tried and dropped, with link to the relevant ADR for each.
5. **Open questions** — what's deferred, what's unresolved (omit if everything in this phase landed cleanly).
6. **Source turns** — footer with relative markdown links back to `{{RAW_FOLDER}}/turn-NNN-*.md`.

## Constraints

- Cap at ~600 lines.
- Use markdown links of form `[text](path)` for cross-refs to other chapters and ADRs.
- No emojis.
- Plain prose, no Q&A format, no "the user said / the assistant said".
- Do NOT write full ADRs in this chapter — the ADRs live in `<docs-dir>/decisions/`. Just reference them by ID and link forward.

After writing, use Read to verify the file was written and report total lines.
