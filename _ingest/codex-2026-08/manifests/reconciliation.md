# Reconciliation notes — codex-2026-08 phase manifest

## Outcome

Nine canonical phases covering merged turns 001–471 contiguously, and 40 decision records. The three agents proposed 15 phases (5 each); six merges brought that to 9.

## Agent seams

Both agent-coverage seams were checked against the source and dissolved:

- **A/B seam (157/158)**: turn 157 ("How many tokens was the LongMemEval?") through 160 continue the same token-count discussion mid-thread, and the 153–156 success reframe directly governs the 1M-baseline work at 161–172. A.phase-5 + B.phase-1 merged into phase 04 (115–172). The real pivot (161, "I want to see retrieval with a 1M transcript") lives inside the phase and is folded into DR-0016.
- **B/C seam (314/315)**: turn 315 ("did we do the layer ablation?") validates the selector/layer-prefix work from B.phase-5. C.phase-1 (315–322) merged into phase 06. Turn 323 ("the next frontier is diffuse information") is a genuine boundary and starts phase 07.

## Merges of over-fragmented proposals

- **A.phase-3 + A.phase-4 → phase 03 (073–114)**: both are one accuracy campaign under the 95% target, narrowing from Hebbian overlay through the architecture review to the committed consolidation layer. The 093 review request is kept as a sub-boundary note.
- **B.phase-2 + B.phase-3 → phase 05 (173–232)**: compress the packet, then test whether it can replace context — one arc; 207 is the payoff question, not a topic change.
- **B.phase-4 + B.phase-5 (+C.phase-1) → phase 06 (233–322)**: diagnosis (set reachability) and solution (marginal set selector) of a single problem, ending in validation. Agent B's soft splits at 261/267 kept as section breaks.
- **C.phase-4 + C.phase-5 → phase 09 (431–471)**: both driven by the same "runs are too slow/heavy" motivation (431, 447), closing with ladder restoration.

Kept separate: phases 01, 02 (real user drift-halt at 033), 07, 08 (real pivots at 323 and the 430/431 gap). Agent A's optional split at 049 and Agent C's at 377 were rejected per the agents' own reasoning.

## Decision deduplication and edits

- The only cross-proposal duplicate: A's 153/156 reframe and B's 161 benchmark lock describe one decision arriving in two halves across the seam; merged into **DR-0016** with refs [153, 156, 161].
- B's dual-tagged moment at 186 was split into **DR-0017** (LOCK-IN: TF-ISF + HSC) and **DR-0018** (SCOPE-CUT: SOM deferred) so each record carries one tag.
- **DR-0025** (285/286) retagged from B's LOCK-IN to **PIVOT**: abandoning QK/OV operator construction in favor of marginal set selection is a direction change, not a commitment to an in-flight design.
- Untagged moments were assigned tags from their descriptions: PIVOT for 007/009, 049, 073, 115; LOCK-IN for 145/146, 196/198, 299/300, 306, 463/464.

## Revisit candidates

- Phase 06 is the largest (90 turns); splitting at 261 would exceed neither correctness nor the 9-phase cap if a 10-phase budget is later allowed.
- Turn 221 (book-recommendation aside) sits inside phase 05; turn 295's pasted design doc (unsampled by B) sits inside phase 06.
