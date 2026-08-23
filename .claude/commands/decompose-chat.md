---
description: Decompose a long LLM-conversation export into a navigable doc tree, dev guide, and ADRs. Usage: /decompose-chat <path-to-source.md> [--output-dir _ingest] [--docs-dir docs]
---

The user has invoked `/decompose-chat $ARGUMENTS`.

Invoke the **chat-decompose** skill (`.claude/skills/chat-decompose/SKILL.md`) and follow its pipeline end-to-end:

1. Parse `$ARGUMENTS` for the source path and optional `--output-dir` / `--docs-dir` flags.
   - If no source path is provided, ask the user.
   - Default `--output-dir` is `_ingest`. Default `--docs-dir` is `docs`.
2. Sniff the source file's first ~200 lines to detect turn-delimiter and role-heading patterns. If they look standard (`---` delimiters and `## User`/`## Assistant` headings), proceed with defaults. Otherwise propose explicit `--turn-delim` and `--role-pattern` regexes and ask the user to confirm.
3. Run **Stage 0** (`build_turn_index.py`) and verify the manifest output (especially `merged_alternation_ok`).
4. Run **Stage 1** — dispatch three `chat-decompose-clusterer` agents in parallel, then one `chat-decompose-reconciler` agent for canonical phases/decisions/reconciliation.
5. Validate manifest cross-references (every decision's phase_id and turn refs are consistent with phases.json) before proceeding.
6. Run **Stage 2** (`split_into_phases.py`) to produce the per-phase raw folder tree.
7. Run **Stage 3** — dispatch `chat-decompose-chapter` agents (up to 3 in parallel per round) to write each dev-guide chapter. Then hand-write the dev-guide overview.
8. Run **Stage 4** — dispatch `chat-decompose-adr` agents (up to 3 in parallel per round, batching ADRs by phase) to write each ADR. Then hand-write the ADRs index.
9. Run **Stage 5** — write `<docs-dir>/README.md` and root `README.md` (short, hand-written).
10. Run `verify.py` and report the result. All 9 checks should pass.

If any step fails (manifest validation, hash mismatch, link integrity), stop and report the issue rather than continuing.

If `chat-decompose-clusterer`, `chat-decompose-reconciler`, `chat-decompose-chapter`, or `chat-decompose-adr` subagents are not available, fall back to `Explore`/`Plan`/`general-purpose` per the SKILL.md guidance.
