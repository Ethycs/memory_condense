# Session Handoff — docs-tree creation + repo state snapshot

**Date**: 2026-08-14 **Status**: ✅ LANDED (docs) / 🟡 IN FLIGHT (code)
**Scope**: whole repo — docs newly created, two code files **UNCOMMITTED**

## 1. What this session did (chronological)

1. Assessed competitiveness vs SimpleMem/Mem0/Zep/Letta/Hindsight + the MemDelta confound study (full write-up: `04 - Reference/00`).
2. Audited the entire implementation and eval results (findings distributed into `03 - Architecture`, `06 - Roadmaps`, `08 - Analysis`).
3. Created and populated this `docs/` tree per `Agentic Technique Master.md`.

## 2. Changes in the working tree

| File | Change | Test state |
| --- | --- | --- |
| `src/memory_condense/eval/judge.py` | `num_retries=5` added to litellm call | **UNCOMMITTED**; suite green (48/48 at last baseline) |
| `src/memory_condense/eval/responder.py` | `num_retries=5` added to litellm call | **UNCOMMITTED**; same |
| `docs/**` (14 files) | entire tree created this session | n/a (docs lane) |

Repo had exactly 2 commits before this session: `0871e05` (Phase 0 structure), `cd9f423` ("Good compressor").

## 3. Findings (verify, then act)

1. **validated** — Only Phases 0 and 5 of the design exist. No MemoryItem, no extraction, no decay, no hybrid retrieval, no token budgets. `grep -rE "decay|heat|energy|supersede|provenance" src/` → zero real hits.
2. **validated** — Retrieval helps, and only at depth: +0.30 mean / +13.1pp Recall@4 on a 283-turn conversation; zero gain on a 27-turn one (numbers: `08 - Analysis/00`).
3. **BUG-adjacent** — `EmbeddingService.dim` hardcoded to 1024 (`embedding.py:82`): a model swap silently corrupts the index.
4. **known rough edge** — judge = responder model (Haiku both); absolute eval scores are conflated, only k-ablation deltas are trustworthy.
5. **validated** — eval results are stale (2026-01-31, ~6.5 months old) and `eval_results/` is gitignored — the numbers exist only on this machine and in `08 - Analysis/00`.
6. **validated** — the 54-config sweep in `sweep.py` has never been run.
7. Competitive verdict: not competitive as an architecture; plausibly competitive as a strong baseline (MemDelta logic); genuinely novel in eval methodology. Full reasoning: `04 - Reference/00`.

## 4. Open work, in priority order

1. Commit the `num_retries` change — blocks any long eval run. (`git add -p src/memory_condense/eval/`)
2. Tier 0 of `06 - Roadmaps/00`: token/latency instrumentation → stronger judge model → run the sweep → plot `scores_by_position`.
3. Tier 1: LongMemEval loader (the competitiveness Decision Point), lexical weights, ContextPacker, un-hardcode dim.
4. Decide fate of root-level `arch_instructions.md` (canonical copy now at `01 - Design/00`; root original is a deletion/archive candidate — user decision).

## 5. Artifacts

| Path | What |
| --- | --- |
| `docs/` (this tree) | 14 docs: theory, design, implementation, architecture, reference, standard, roadmap, this handoff, analysis, archive index |
| `eval_results/*.json` (local-only) | 4 single-run results, 2026-01-31 |
| `Agentic Technique Master.md` | the style guide governing this tree |

**First action for the next session**: run `pixi run -e dev pytest -q` (expect 48 passed) and `git status` (expect only the two eval files modified + `docs/` untracked); then decide — commit code+docs together, or docs first.
