# Operating Requirements

**Status**: Normative — Living Document
**Date**: 2026-08-16
**Why this exists**: The decay coordinate was wrong for the project's entire life because the intent lived only in conversation ([`06 - Roadmaps/01`](../06%20-%20Roadmaps/01%20-%20Delivering%20the%20Specified%20System.md) §2). Four more constraints of the same kind surfaced in the 2026-08-15/16 session, each stated by the operator as a correction *after* work had been measured against the wrong frame. This document is where such constraints live from now on. **A requirement that exists only in a chat transcript does not exist.**

Requirements carry IDs so tests, docs, and commit messages can cite them (`R5` etc.). Each carries its enforcement state honestly: met, violated, or unmeasured.

---

## R1 — The decay coordinate is conversation turns

Energy decays in **turns since last access**, never wall-clock time. Each subsequent turn differentially assigns decay: the conversation itself decides what stays warm — retrieval reheats the items it returns, appending a turn cools everything else. Wall-clock timestamps are audit metadata only; nothing in selection, ranking, or tiering may read them.

- **State**: **Met** (schema v4, `9aea4cd`). Enforced by `test_the_coordinate_is_turns_not_wall_clock` and the turn-based kernel in `decay.py`.
- **Trap**: any future term of the form `0.5 ** (elapsed_seconds / …)` reintroduces the defect. `decay.py`'s module docstring is the guard.

## R2 — Cost is a first-class metric beside recall

The system's claim is **dual benefit**: equal recall for fewer tokens, or more recall at equal cost. A recall number without its token cost is not a result. Per-token efficiency may never be compared across operating points, and never read without absolute recall beside it (a failed arm once scored the session's best per-1k figure while finding 6% of answers).

- **State**: **Met in the harness** (`context_tokens` on every `QuestionRecall` / `TurnResult`), **unevenly honoured in reporting** — several 2026-08-15 comparisons quoted component recall without matched budgets until corrected.

## R3 — Memory is layered; only the assembled context is "the system"

The deliverable is the **assembled, budgeted context**: recent window + memory header + verbatim expansions (+ era summaries when built), packed by `ContextPacker` under hard per-section ceilings. Retrieval modes (dense / hybrid / span) are **components inside layers**, not alternatives to the system. Benchmarking a component and reporting it as the system is a category error — it happened twice (the original harness bypassed `build_context`; the 2026-08-15/16 arm comparisons bypassed it again).

- **State**: **Built, mis-wired, under-measured.** The expansion layer can only draw from dense or hybrid ([`condenser.py:220`](../../src/memory_condense/condenser.py#L220)) — span, the best-measured retriever, is unreachable from the assembly. No whole-system number exists with all layers live.

## R4 — Operating envelope: 1 to 1,000,000 tokens

The system must behave correctly across six orders of magnitude of transcript size.

- **Below the envelope's value regime** (transcript ≲ affordable context): *sending everything is the correct strategy*, and the system should know it. This is H2 as policy — retrieval contributes ≈0 while the conversation fits. On LoCoMo conv-26 (13.5k tokens) the full transcript beats every retrieval arm (33.2% vs 26.6% best) at comparable session cost.
- **In the value regime** (transcript ≫ per-turn budget): bounded retrieval is the only viable strategy and its recall-at-budget is the figure of merit.
- The crossover is `N* = 2B/t` turns (B = per-turn budget, t = tokens/turn). Corpus regime is therefore **observable at runtime** from N and cumulative T alone, and budget policy should adapt to it. A fixed `ContextBudget` of 6,200 tokens against a 1M envelope (0.6%) is not a tuned choice; it is a leftover.
- **State**: **Unmet.** Budgets are static; no adaptive policy exists; no in-envelope corpus has ever been benchmarked (see R6).

## R5 — No compaction; linear token cost per turn

Per-turn context is bounded by a constant, so session cost is O(N). There is **no compaction step, ever** — the index absorbs the past instead. The naive alternative (resend history) is O(N²) and is what forces compaction; avoiding that is the point of the system.

**Amendment (operator, 2026-08-16)**: occasional expensive operations against the memory system are permitted — index rebuilds, era summarization, dedup passes — as amortized maintenance, *off the per-turn path*. The constraint is the **schedule**, not the existence of expensive work: nothing O(N) may run on every turn.

- **State**: **Violated in one place.** `add_chunks` clears the span cache on every append ([`retrieval.py:265`](../../src/memory_condense/retrieval.py#L265)), so the next query rebuilds pooled vectors over all chunks: O(N) per turn, O(N²) per session, in the currently-winning retriever. Fix is incremental or threshold-triggered rebuild — the right operation on the wrong schedule.
- Under this requirement, Phase 4 era summaries are re-justified: they are the mechanism that **bounds pool growth** so per-turn cost stays constant — periodic compaction of the *index* is what permits never compacting the *context*.

## R6 — The benchmark is the deployment corpus

The deployed use case is **agentic coding** (Claude Code sessions: ~500 tokens/turn, tool output dominating chunk volume, crossover ≈ 25 turns). LoCoMo (31 tok/turn, crossover ≈ 400 turns, 33–47% containment ceiling) is nearly the worst-case regime for this design and is valid **only for retriever-vs-retriever comparisons**, never for verdicts on whether the system earns its place.

- **State**: **In progress** (2026-08-16): this project's own build session is being converted into a probe benchmark. No in-regime number exists yet.

---

## Established results the requirements rest on

| Finding | Where measured |
| --- | --- |
| IDF/hybrid beats dense at every operating point, 1.4–2.9× at matched cost | conv-26 sweep, 2026-08-16 |
| Span beats hybrid at every matched budget; dense adds **0.0pp** to the union | conv-26 sweep + oracle union |
| Retrieval beats random-at-matched-tokens 34× and recency 15×; recency is *worse than random* | conv-26 baselines |
| A router's oracle ceiling over span+hybrid is +2.0pp — not worth a predictor | conv-26 hit-set overlap |
| Evidence-unit size is ~110–250 tokens; span pooling reconstructs it on any corpus; counted in **tokens**, never chunks | 2026-08-15 replication, 4 samples |
| Stratify per level — one mixed-granularity pool collapses recall (cosine length bias, 0.678 vs 0.602) | 2026-08-15 |
| The energy term reorders 100% of top-5 sets once the coordinate is turns (was structurally 0%) | conv-26, 2026-08-16 |
| `half_life_turns = 30` leaves a one-pass-ingested store 83% COLD; informative band 120–240 — **but** the one-pass harness never reheats, so this measures creation age, not access rate | heat probe, 2026-08-16 |

## Known violations & wiring gaps (work list)

1. `add_chunks` span-cache clear — R5 violation, fix schedule (incremental/threshold).
2. `build_context` cannot draw expansions from span — R3 wiring gap, one-line class of fix.
3. Static `ContextBudget` — R4, needs regime-aware policy (`send-everything` below crossover).
4. No whole-system measurement with all layers live — R3.
5. No in-envelope corpus result — R6 (in progress).
6. `half_life_turns` default unvalidated against live (reheating) usage — R1/R6 joint gap.
