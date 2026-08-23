# Phase-proposal prompt template

Substitute the placeholders below with concrete values before dispatching:

- `{{AGENT_LETTER}}` — A, B, or C.
- `{{TURN_START}}` — first merged turn id this agent covers (e.g. `001`).
- `{{TURN_END}}` — last merged turn id this agent covers (e.g. `035`).
- `{{SOURCE_PATH}}` — absolute path to the chat export.
- `{{MANIFEST_PATH}}` — absolute path to `manifests/turns.json`.
- `{{TOTAL_MERGED_TURNS}}` — total merged-turn count from turns.json.
- `{{SPECIAL_INSTRUCTIONS}}` — extra brief for this slice (see below).

## Special instructions per agent

- **Agent A** (first third): "Identify topic shifts; propose phases; list decision moments."
- **Agent B** (middle third): same as A, plus "flag any back-references to A's range."
- **Agent C** (final third): same as A, plus "tag scope-cut / lock-in decisions explicitly; characterize the conversation's final state."

---

## Prompt body (substitute placeholders, then send)

You are proposing phase boundaries for a slice of an LLM design conversation.

**Source file:** `{{SOURCE_PATH}}`
**Turn manifest:** `{{MANIFEST_PATH}}`

The conversation has {{TOTAL_MERGED_TURNS}} merged conversational turns (alternating user/assistant). You cover **merged turns {{TURN_START}}–{{TURN_END}}**. A "merged turn" collapses consecutive same-role export blocks into one logical reply — use the `merged_turns` array in `turns.json` for ranges.

**Your task:**

1. Read `turns.json` first to get exact line ranges for merged turns {{TURN_START}}–{{TURN_END}}.
2. Use `Read` with offset/limit to sample the source — **do not read the whole file**. Read each merged turn's first ~30 lines to identify topic, plus spot-checks of larger turns.
3. Identify topical phases (2–5 in your range). A phase is a coherent stretch where the same design problem is being worked on; phases end at user-driven topic pivots.
4. Note **decision moments**: places where the conversation locks in a design choice, abandons an approach, or chooses between alternatives. Record `merged_turn_id` and a one-line description.

{{SPECIAL_INSTRUCTIONS}}

**Output format (literal JSON in your reply, parseable):**

```json
{
  "agent": "{{AGENT_LETTER}}",
  "covered_range": ["{{TURN_START}}", "{{TURN_END}}"],
  "proposed_phases": [
    {
      "merged_turn_start": "...",
      "merged_turn_end": "...",
      "topic": "kebab-case-slug",
      "topic_long": "one-sentence description",
      "why": "what makes this phase coherent / why it ends here"
    }
  ],
  "decision_moments": [
    {
      "merged_turn_ids": ["..."],
      "description": "one-line description (prefix [SCOPE-CUT] or [LOCK-IN] if applicable)"
    }
  ],
  "notes": "anything noteworthy that future reconciliation should know — boundary uncertainty, references back/forward, hard-to-classify moments"
}
```

Cap response at **600 words including the JSON**. Quality of phase boundaries matters more than coverage detail.
