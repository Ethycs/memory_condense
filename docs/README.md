# memory_condense — documentation tree

**Status**: Living Document
**Date**: 2026-08-14 (reconciled against the as-built code the same day)
**Applies to**: the whole repository
**Depends on**: [`Agentic Technique Master.md`](../Agentic%20Technique%20Master.md) — the style guide governing this tree

This tree follows the folder system in the style guide: each numbered folder is a prerequisite for the folders after it. A change is only "real" when backed by at least one of the three lanes — tests, documentation, code.

> **Merged to `main`, still unmeasured — read this first.** The system these docs describe landed on `main` in merge `f3edc91`, bringing seven commits off `feat/memory-layer`: `80262ea` (memory layer), `abdcc9b` (eval harness repair + benchmark path), `24c7323` (this tree), `a9193ca` (doc reconciliation), `0093346` (MCP server), `a1bc585` (cross-process label allocation), `7717c3a` (supersede + honest provenance). What has *not* changed is the part that matters most: **nothing has been measured.** `git log --oneline` is the authority, not any sentence in this tree.

## Reconciliation state (2026-08-14)

| Doc | Reconciled? | Substance of the change |
| --- | --- | --- |
| `01 - Design/00 - Original Architecture Plan.md` | ✅ | Phases 0,1,2,3,5 built; Phase 4 (cold summaries) is the only unbuilt phase |
| `01 - Design/01 - Eval Design…` | ✅ | Retired-model BUG documented; judge≠responder and token instrumentation both **resolved**, not open |
| `02 - Implementation/00 - Setup…` | ✅ | 48-test baseline → **366**; hardcoded `dim=1024` bug marked fixed; schema-v2 migration gotcha added |
| `02 - Implementation/01 - Running the Eval Harness.md` | ✅ | Rewritten for four CLI modes; benchmark data sources + cost warning; sweep is 54 configs, not 48 |
| `03 - Architecture/00 - System Overview.md` | ✅ | Diagram and every subsystem rewritten; "there is no condensation yet" was false |
| `04 - Reference/01 - Vocabulary.md` | ✅ | Lifecycle + retrieval terms moved out of *(planned)*; BM25/hybrid/α/`term_count`/`UsageStats`/F1/provenance added |
| `05 - Standards/00 - MC-STD-DATA-v0.md` | ✅ | Schema v2 + migration path; new normative clauses 8–10 (provenance, no destruction, migrate-in-place). Still **DRAFT** |
| `06 - Roadmaps/00 - Gap Analysis and Roadmap.md` | ✅ | Status table and tiers rewritten; Decision Point now *unblocked* but still *open*. **Partly superseded 2026-08-15** — see below |
| `06 - Roadmaps/01 - Delivering the Specified System.md` | **new** | Decay was specified in wall-clock seconds; the design intent is per-turn. The energy term therefore contributed a constant, and **every memory-arm number is void** — including the Phase 4 verdict. Carries the git evidence that the spec was wrong from commit one, and the five-stage delivery sequence |
| `00 - Theory/00 …` | — | Not touched; stable by policy (corrections only) |
| `04 - Reference/00 - Competitive Landscape 2026.md` | — | Not touched this pass |
| `08 - Analysis/00 - Retrieval Ablation…` | ✅ | Sweep corrected to 54 configs; a position-bin analysis was added and then **retracted the same day** — it does not replicate on the second run pair, and every bin-to-bin difference is inside noise. The aggregate ablation result stands |
| `08 - Analysis/01 - Extraction and Decay Audit` | **new** | 70.6% of memory items never reach the prompt; COLD is unreachable by construction; the default extractor is 65% spurious `Constraint`s. All free, all previously unmeasured |
| `07 - Status Reports/…` | ✅ | Three dated handoffs; **2026-08-15 is the current one** and supersedes the earlier test counts (48 → 366 → 523) |

## The tree

```
docs/
├── 00 - Theory/           Why retrieval-weighted context should work; the eval's formal claim
├── 01 - Design/           The original architecture plan + eval design rationale
├── 02 - Implementation/   Setup, the four eval modes, and MCP/Claude Code integration
├── 03 - Architecture/     The as-built system map (what actually exists)
├── 04 - Reference/        External landscape (SimpleMem, Mem0, MemDelta…) + vocabulary
├── 05 - Standards/        Normative data contracts (SQLite v2, embedding, memory provenance, formats)
├── 06 - Roadmaps/         Gap analysis: designed vs. built vs. measured, tiered next steps
├── 07 - Status Reports/   Dated snapshots (session handoffs)
├── 08 - Analysis/         Measured results — the ablation numbers, and one retraction
├── 09 - Archived/         Superseded material (append-only)
└── 10 - Research Log/     Dated experiment entries with data/ artifacts; baselines of record
```

## Governance

| Folder | Purpose | Freeze policy |
| --- | --- | --- |
| 00 Theory | Foundations | Stable; corrections only |
| 01 Design | Rationale | Archive when superseded |
| 02 Implementation | Setup + realized specs | Versioned with code |
| 03 Architecture | System map | Keep current (single trusted map) |
| 04 Reference | External + project-level | Living |
| 05 Standards | Normative contract | Frozen after release; amend by version |
| 06 Roadmaps | Planning | Living |
| 07 Status Reports | Dated snapshots | Archive when complete |
| 08 Analysis | Measured deep-dives | Living |
| 09 Archived | History | Append-only; never edit |

## Where to start

- Resuming cold? → **`06 - Roadmaps/01 - Delivering the Specified System.md` first** — it explains why every memory-arm number on record is void and what order the remaining work has to happen in. Then `07 - Status Reports/2026-08-15_retrieval-measurement-session.md` for the retrieval half, which still stands.
- "What does the system do?" → `03 - Architecture/00 - System Overview.md`.
- "What's left to build?" → `06 - Roadmaps/00 - Gap Analysis and Roadmap.md`.
- "How do I run it?" → `02 - Implementation/01 - Running the Eval Harness.md` (start with the free `--compare` mode).
- "How do I actually *use* it day to day?" → `02 - Implementation/02 - MCP Integration.md` — the memory system is exposed to Claude Code as an MCP server.
- "Is this competitive?" → **on common benchmarks, still unanswerable** (`04 - Reference/00` + `08 - Analysis/`; zero common-benchmark runs exist). **In-regime, a baseline of record now exists**: `10 - Research Log/00` — B0, 92.3% recall at 0.5% of transcript tokens on this project's own build session, with its caveats stated.

## The one distinction this tree tries hardest to keep

**Built ≠ measured.** 523 passing tests prove the memory layer does what it says. They prove nothing about whether it beats the dense-retrieval baseline it sits on. Merging to `main` did not change that, and neither did any of the polish since. Any doc here that claims a benefit without a number in `08 - Analysis` behind it is a bug — report it.

---

**Verification block**: run

```powershell
git log --oneline -1                     # expect 03105c4 on main
git status --short                       # NOT clean — the decay re-coordination is in flight
pixi run -e dev pytest -q -m "not slow"  # 523 at 03105c4; red mid-change, by design
```

> **In flight as of 2026-08-15**: `schemas.py` and `decay.py` have been re-coordinated onto turns; schema v4 and the `now_turn` threading have not landed yet, so the working tree does not run. `06 - Roadmaps/01` §3 Stage 1 is the checklist.

If all three hold, this tree is accurate as far as it goes — which is to say it accurately describes something whose *external* competitiveness has never been measured. Retrieval itself now has numbers: see `07 - Status Reports/2026-08-15`. The next thing worth doing is not another feature; it is `--answer-recall` against the long-form corpus, which is free and needs no API key.
