# ADR-writer prompt template

Substitute placeholders before dispatching. One agent can be told to write 1–4 ADRs in a single dispatch (use the multi-ADR variant below).

- `{{DECISION_ID}}` — `DR-0001`, etc.
- `{{DECISION_NUMBER}}` — `0001`, `0002`, etc. (without the `DR-` prefix; used in filenames).
- `{{SLUG}}` — kebab-case.
- `{{TITLE}}` — imperative title.
- `{{TAG}}` — `PIVOT`, `LOCK-IN`, or `SCOPE-CUT`.
- `{{ONE_LINE}}` — concise one-line summary.
- `{{TURN_REFS}}` — list of merged turn IDs.
- `{{PHASE_ID}}` — phase this decision belongs to.
- `{{PHASE_SLUG}}` — phase slug for raw-folder lookup.
- `{{RAW_FOLDER}}` — absolute path to the raw phase folder.
- `{{DOCS_DIR}}` — absolute path to docs root.
- `{{DATE}}` — date of the source conversation (ISO format, e.g. `2026-04-26`).
- `{{EXTRA_CONTEXT}}` — any decision-specific notes (e.g. "this supersedes DR-NNNN", "this refines DR-NNNN", "this was later cut by DR-NNNN").

---

## Prompt body

Write an architecture decision record (ADR) in MADR-lite format. **Use the Write tool to write the file directly to disk.**

## ADR to write

- **ID:** {{DECISION_ID}}
- **Title:** {{TITLE}}
- **Tag:** {{TAG}}
- **One-line:** {{ONE_LINE}}
- **Source merged turns:** {{TURN_REFS}}
- **Phase:** {{PHASE_ID}} ({{PHASE_SLUG}})
- **Raw turn folder:** `{{RAW_FOLDER}}`
- **Dev guide chapter:** `{{DOCS_DIR}}/dev-guide/{{PHASE_ID}}-{{PHASE_SLUG}}.md`

{{EXTRA_CONTEXT}}

## How to find the source

Read the dev guide chapter first (it's the design endpoint that this ADR's decision contributed to). Then sample 2–4 raw sub-turns in the phase folder that articulate the decision moment itself.

## Output path

`{{DOCS_DIR}}/decisions/{{DECISION_NUMBER}}-{{SLUG}}.md`

## MADR-lite template

```markdown
# {{DECISION_NUMBER}}. {{TITLE}}

- **Status:** Accepted
- **Date:** {{DATE}}
- **Tag:** {{TAG}}

## Context

What forced the decision. The constraint, the failure mode, the inadequacy of the prior approach. Be specific — quote a concrete observation that motivated the change. 1–3 short paragraphs.

## Decision

The chosen path, stated as an imperative. One paragraph. Match the one-line summary but with enough detail that someone reading just this section understands what was committed.

## Consequences

- **Positive:** what this enables / simplifies.
- **Negative / cost:** what this gives up / makes harder.
- **Follow-ups:** what other decisions or work this implies. Reference other DR IDs if relevant.

## Alternatives considered

- **<Option A>** — what it was, why it was rejected.
- **<Option B>** — what it was, why it was rejected.

(Two to four alternatives. If the decision was a clean pivot with no live alternatives at the time, state "no live alternatives at decision time" rather than inventing.)

## Source

- **Source merged turns:** {{TURN_REFS}}
- **Raw sub-turns:** [list 2–4 specific sub-turn files]
- **Dev guide:** [chapter {{PHASE_ID}}](../dev-guide/{{PHASE_ID}}-{{PHASE_SLUG}}.md)
```

## Constraints

- ADR length: ~60–150 lines. Concise.
- No emojis.
- Reference *only* the decision itself and turns that produced it. Don't restate the dev guide chapter — link to it.
- Status is "Accepted" for the first version of every ADR. If the decision is later superseded or refined, the status note should reflect that (e.g. "Accepted (superseded by DR-NNNN)" or "Accepted (refined by DR-NNNN)").
- Date is the date of the source conversation.

After writing, use Read to verify and report total lines.

---

## Multi-ADR variant

If batching N ADRs into one dispatch, repeat the "ADR to write" + "Output path" sections for each, share the "How to find the source", "MADR-lite template", and "Constraints" sections, and ask the agent to write all N files before reporting.
