# Phase-reconciliation prompt template

Substitute placeholders before dispatching:

- `{{SOURCE_PATH}}`, `{{MANIFEST_PATH}}` — as in phase_proposal.md.
- `{{TOTAL_MERGED_TURNS}}` — total merged turn count.
- `{{PROPOSAL_A_JSON}}`, `{{PROPOSAL_B_JSON}}`, `{{PROPOSAL_C_JSON}}` — the three JSON proposals returned by the proposal agents.
- `{{OUTPUT_DIR}}` — `<output-dir>/manifests/` (where the three files will be written).

---

## Prompt body

You are reconciling three independent phase-clustering proposals for an LLM design conversation into a single canonical manifest. Your output decides the folder structure for the entire decomposition project, so be deliberate.

## Source

- File: `{{SOURCE_PATH}}` (read line ranges from manifest)
- Manifest: `{{MANIFEST_PATH}}`
- {{TOTAL_MERGED_TURNS}} merged turns total. Three agents covered the conversation in equal thirds.

## Proposals to reconcile

### Agent A

```json
{{PROPOSAL_A_JSON}}
```

### Agent B

```json
{{PROPOSAL_B_JSON}}
```

### Agent C

```json
{{PROPOSAL_C_JSON}}
```

## Your task

Produce three files:

### 1. `{{OUTPUT_DIR}}/phases.json`

Reconcile the proposed phases into **6–9 canonical phases**. Some proposals over-fragment (multiple sub-phases all narrowing toward one design conclusion); collapse those. Validate the agent-coverage seams (the boundaries between A/B and B/C) — they may be real topic boundaries or just artifacts of where the agent ranges happened to fall. Feel free to merge across them if appropriate.

Format:

```json
{
  "phases": [
    {
      "phase_id": "01",
      "slug": "kebab-case",
      "name": "Human-readable name",
      "merged_turn_range": ["001", "005"],
      "summary": "1-3 sentence summary",
      "decision_ids": ["DR-0001"],
      "source_proposals": ["A.phase-1"],
      "notes": "reconciliation note"
    }
  ]
}
```

Phase IDs must be `01`..`NN`, slugs kebab-case. Phase ranges must be contiguous — every merged turn from 001 to {{TOTAL_MERGED_TURNS}} must be covered by exactly one phase.

### 2. `{{OUTPUT_DIR}}/decisions.json`

Reconcile the decision-moment lists into a canonical ADR list. Deduplicate (some moments appear in multiple proposals). Number them `DR-0001`..`DR-NNNN`.

Format:

```json
{
  "decisions": [
    {
      "decision_id": "DR-0001",
      "slug": "kebab-case",
      "title": "Imperative title",
      "merged_turn_refs": ["006", "007"],
      "phase_id": "01",
      "tag": "PIVOT" | "LOCK-IN" | "SCOPE-CUT",
      "one_line": "Concise statement of the decision."
    }
  ]
}
```

Tags: `PIVOT` for direction changes, `LOCK-IN` for design commitments, `SCOPE-CUT` for V1 simplifications.

**Critical consistency requirements:**
- Every decision's `phase_id` must reference an existing phase.
- Every `merged_turn_refs[*]` must fall inside that phase's `merged_turn_range`.
- Every decision listed in a phase's `decision_ids` must exist in `decisions.json` (and vice versa).

### 3. `{{OUTPUT_DIR}}/reconciliation.md`

Brief markdown narrative (≤ 500 words) explaining:

- How many phases the canonical merge has and why (not the raw count, not over-merged — what's the right granularity?).
- Which seams between agents you kept vs. dissolved.
- Which over-fragmented phases you merged, and which you kept separate (and why).
- Which decisions you deduplicated.

## Constraints

- Phase ranges must cover turns 001–{{TOTAL_MERGED_TURNS}} contiguously with no gaps and no overlaps. Validate this before writing.
- Each phase's `merged_turn_range` end must be one less than the next phase's start.
- Write all three files using the Write tool. Verify them with Read after writing.
- Do **not** modify the source chat export.
- After writing, return a short summary (≤200 words) of the phase count, decision count, and any reconciliation calls the user might want to revisit.
