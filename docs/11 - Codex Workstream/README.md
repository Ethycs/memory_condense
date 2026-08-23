# 11 - Codex Workstream

A decomposition of the Codex agent's design conversation for the
memory_condense retrieval stack — `data/codex-transcript-2026-08-15-to-2026-08-23.md`,
1.35M characters, 471 merged turns over nine days — into a navigable doc tree.
The Codex workstream built the eval/retrieval side of this repo (the
`src/memory_condense/eval/` ladder, the selector, the 1M harness) in parallel
with the main-session work, and its design decisions lived only in that
transcript until now.

## What's here

- **[dev-guide/](dev-guide/00-overview.md)** — nine chapters, one per design
  phase, each written as the design endpoint of its phase (imperative voice,
  "Why not X" for load-bearing reversals). Start at the
  [overview](dev-guide/00-overview.md).
- **[decisions/](decisions/README.md)** — 40 MADR-lite ADRs
  (5 PIVOT / 27 LOCK-IN / 8 SCOPE-CUT), numbered `0001`–`0040`, each grounded
  in the raw turns that produced it. Index in
  [decisions/README.md](decisions/README.md).

## Provenance

The decomposition was produced by the `chat-decompose` pipeline
(`.claude/skills/chat-decompose/`):

- **Manifests** (committed): `_ingest/codex-2026-08/manifests/` —
  `phases.json` (9 phases), `decisions.json` (40 decisions),
  `reconciliation.md`, `source.sha256`. These are the operational
  source-of-truth for phase boundaries and decision identity.
- **Raw turn tree** (untracked, regenerable): `_ingest/codex-2026-08/raw/` —
  one file per sub-turn with byte-level provenance frontmatter. Chapter and
  ADR source links point here. Rebuild it with the normalizer
  (`_ingest/codex-2026-08/procedure/normalize_codex_export.py`) followed by
  the skill's `build_turn_index.py` and `split_into_phases.py` (default
  regexes) against the derived `source-normalized.md`.
- **Source transcript** (never committed, never edited):
  `data/codex-transcript-2026-08-15-to-2026-08-23.md`, pinned by
  `manifests/source.sha256`.

Turn numbering: raw filenames are **sub-turn** numbers (0001–3475); chapters
and ADRs reference **merged** turns (001–471). Each raw file's frontmatter
carries its `merged_turn_id`.
