# Decomposition status — codex transcript → doc tree

**Interrupted mid-pipeline for context compaction, 2026-08-23.** This file is
the resume contract; delete it when Stage 5 + verification pass.

## What this is

`data/codex-transcript-2026-08-15-to-2026-08-23.md` (1.35M chars, the Codex
agent's own log of building this repo) run through the `chat-decompose` skill
(`.claude/skills/chat-decompose/` — SKILL.md is the full pipeline contract).
The skill was installed mid-session, so agents ran as `general-purpose` with
the skill's templates — the sanctioned fallback.

## Pipeline state

| Stage | State |
| --- | --- |
| 0 — turn index | ✅ 3,475 sub-turns → 471 merged, alternation ok. **Format deviation**: the Codex export has `## <ISO-ts> — Role` headings and no `---` delimiters, so a deterministic normalizer derives a canonical-format copy first: `_ingest/codex-2026-08/procedure/normalize_codex_export.py` (original sha `c6f872ba…`, derived sha `56526d7c…`, both printed by the script; original never edited) |
| 1 — phases/decisions | ✅ 9 phases, 40 decisions (5 PIVOT / 27 LOCK-IN / 8 SCOPE-CUT), cross-ref validation PASS. Committed: `manifests/phases.json`, `decisions.json`, `reconciliation.md` — **agent output, not regenerable; do not delete** |
| 2 — physical split | ✅ regenerable — `raw/` (3,484 files) is untracked; rebuild with `pixi run -e dev python .claude/skills/chat-decompose/scripts/split_into_phases.py --source "_ingest/codex-2026-08/source-normalized.md" --output-dir "_ingest/codex-2026-08"` (run the normalizer first if `source-normalized.md` is absent; `turns.json` rebuilds via `build_turn_index.py` with default regexes) |
| 3 — chapters | 🚧 **6 of 9 written** (01–06 in `dev-guide/`, 133–377 lines each). Remaining: **07, 08, 09**, then hand-write `dev-guide/00-overview.md` (~100 lines linking chapters in order) |
| 4 — ADRs | ⬜ **0 of 40 written.** All six existing chapters already link `../decisions/NNNN-<slug>.md` — slugs/IDs must come from `manifests/decisions.json` exactly. Then hand-write `decisions/README.md` index (ID, title, tag, phase, source turns) |
| 5 — wiring | ⬜ this folder's `README.md`; a one-line entry for `docs/README.md`'s tree. **Deviation from skill**: do NOT touch the repo-root README (governed tree) |
| verify | ⬜ `pixi run -e dev python .claude/skills/chat-decompose/scripts/verify.py --source "_ingest/codex-2026-08/source-normalized.md" --output-dir "_ingest/codex-2026-08" --docs-dir "docs/11 - Codex Workstream"` — requires the raw/ tree to exist (regenerate first) |

## How to resume (exact)

1. Regenerate `_ingest/codex-2026-08/` stages 0+2 if absent (commands above).
2. Chapters 07–09: one `general-purpose` agent each (parallel, one round),
   prompt built from `.claude/skills/chat-decompose/templates/chapter.md` with
   substitutions from `manifests/phases.json` + `decisions.json`. Per-phase
   char counts: 07 ≈ 492k, 08 ≈ 138k, 09 ≈ 659k (07/09 need aggressive
   sampling). Docs dir: `docs/11 - Codex Workstream`. Source-turn links use
   prefix `../../../_ingest/codex-2026-08/raw/phase-NN-<slug>/`. Style: match
   chapters 01–06 (imperative endpoint voice, Why-not-X subsections, ADR
   links, source-turn footer).
3. ADRs: batch per phase, ≤4 per agent (template's multi-ADR variant; phases
   with 5–6 decisions get two agents). Each agent reads its chapter first,
   then 2–4 raw sub-turns at the decision moments. Note: raw sub-turn
   filenames are NOT merged-turn numbers — agents map via the
   `merged_turn_id` frontmatter in the raw files.
4. Hand-write `dev-guide/00-overview.md`, `decisions/README.md`, folder
   `README.md`; add the docs-tree line; run verify.py; fix any of the 9
   checks that fail; commit; delete this file.

## Chapter-quality notes carried from rounds 1–2

Agents reliably discovered the merged-vs-raw numbering mismatch themselves
and mapped via frontmatter — keep that instruction explicit. Chapter 06
flagged that `../decisions/` doesn't exist yet; expected, resolves at Stage 4.
