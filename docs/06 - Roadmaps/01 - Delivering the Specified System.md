# Delivering the Specified System

**Status**: Living Document
**Date**: 2026-08-15
**Applies to**: `main` — the decay re-coordination is **in flight**, not landed
**Supersedes**: the Phase 4 verdict in `00 - Gap Analysis and Roadmap.md`, and the "signal is against Phase 4" line in `07 - Status Reports/2026-08-15`

## 0. Why this document exists

The question it answers is *"how do we deliver the system that was specified?"* — which turned out to require first establishing that **the specification itself was wrong**, in one line, from the first commit.

## 1. What "the specified system" is

Two sources define it, and they demand different kinds of deliverable.

| Source | Defines | Delivered by |
| --- | --- | --- |
| [`arch_instructions.md`](../../arch_instructions.md) | Six phases, the data model, the ingestion loop | Code + tests |
| [`00 - Theory/00`](../00%20-%20Theory/00%20-%20Retrieval-Weighted%20Context%20and%20Self-Replay%20Evaluation.md) | H1, H2, the ranking equation | **Measurement.** These are claims, not components |

Phase status against `arch_instructions.md`:

| Phase | State |
| --- | --- |
| 0 — project structure | ✅ |
| 1 — transcript / chunk / embed / retrieve | ✅ |
| 2 — memory items / provenance / validator | ✅ |
| 3 — decay + tiering + reheat | ⚠️ **Built on the wrong coordinate** (§2) |
| 4 — cold-tier era summaries | 🔲 Unbuilt, and its gate has never been answerable |
| 5 — MCP / Claude Code integration | ✅ |

"Built ≠ measured" still governs. A phase is delivered when code, tests, **and** a number agree.

## 2. The finding that reorders everything

**Decay was specified in wall-clock seconds. The design intent is per-turn.**

The intent is that *each subsequent turn differentially assigns decay*: the conversation itself decides what stays warm. That mechanism was already wired — `MemoryStore.retrieve` reheats exactly the top-k it returns, so every turn selects its winners for free, at O(k) writes. But it was **inert**, because the exponent counted seconds and an ingest runs in minutes. `elapsed` rounded to nothing, so every item — retrieved or not — held a decay factor of ~1.0. Selection carried no consequence.

This is the same defect as the old `ranking.recency_score`: a term that looks like a discriminator and evaluates to a constant. It was found and fixed at the ranking level in `0d86038`, and not generalised one level down to the kernel feeding it. Two instances of one defect.

### It was never dropped — it was never written

Established from the history, not from memory:

| Check | Result |
| --- | --- |
| `git log --all -- arch_instructions.md` | **One commit** (`0871e05`, Phase 0). Never edited. Line 75 already said `half_life_s` |
| `git log -S` for `half_life_turns` / `last_access_turn` / `turns_since` / `turn_ordinal` | **Zero hits across all 21 commits.** Never implemented, never reverted |
| Files deleted since `cd9f423` "Good compressor" | None |
| `chunker.py` vs `cd9f423` | Byte-identical, zero commits — so the per-turn chunking defect is original behaviour, not regression |
| Skipped / xfailed tests | None |

Every layer faithfully propagated one line written at Phase 0: `arch_instructions.md:75` said seconds → the code implemented seconds → `docs/` was written later to reconcile against *the code*, so it inherited seconds and restated it in four more places. There was never an independent statement of intent for any of it to contradict.

`arch_instructions.md` is left **byte-intact as the historical record**. Rewriting it would destroy the evidence that the spec was wrong from commit one. The normative correction lives here and in `05 - Standards`, where governance belongs.

### What this voids

- **The 4.8% / 0.0% "signal against Phase 4" is void.** It was measured on the broken coordinate: nothing survived to a horizon no item can reach. That number described the clock, not cold memory. Phase 4 is **unmeasured**, not signal-against.
- **`_survival`'s day-14 and day-30 horizons could only ever return 0.0%.** COLD begins below `0.25`, energy is clamped to `≤ 1.0`, and `0.25 = 1.0 × 0.5²` — so 14 days is exactly two half-lives and the theoretical ceiling for *any* unpinned item. Two of four horizons were incapable of returning anything else.
- **Every memory-arm number is void**, including "memory costs +75% tokens for +0.8pp recall." All were taken with the energy term contributing a constant.
- **"COLD is structurally unreachable in an eval run — needs 7–11.75 days"** (`08 - Analysis/01`) was misfiled as an eval limitation. It is a *proof that the coordinate is wrong*: a tier the system can never reach through its own operation is not a tier.

Dense-arm and span numbers are **unaffected** — they never consulted energy.

## 3. Delivery sequence

Order is forced: stage 3 was unanswerable before stage 1.

### Stage 1 — Phase 3, on the right coordinate *(in flight)*

- `decay_factor(last_access_turn, now_turn, half_life_turns)`; `last_access_at` demoted to an audit field nothing in `decay` reads
- Refractory becomes **once per turn** — the 300 s window was only ever approximating that
- Schema v4: `turns.ordinal`, `memory_items.last_access_turn`, `half_life_turns`; additive, backfilled to the current turn so existing stores enter fresh rather than instantly COLD
- `now_turn` threaded through `memory_store` → `ranking` → `condenser` / `mcp_server` / `context_packer`
- `_survival` horizons in turns

**Default `half_life_turns = 30`.** No doc guidance existed, because turns were never the coordinate. At 30, an ordinary item (seed 0.5) reaches COLD after 30 untouched turns and an important one (seed 0.8) after ~50 — both inside a 283-turn transcript and a 200–600-turn LoCoMo conversation. It is **sweepable for the first time**, because a run now advances the coordinate itself.

**Deliberately not bundled:** relevance-weighted reheat (scaling the boost by the retrieval score, so rank-1 and rank-10 differ). It is nearly free and probably right, but changing two things at once makes the measurement unattributable.

**Gate:** tests green.

### Stage 2 — Re-measure what energy touches *(free, keyless)*

1. **Does the tier system populate?** `heat_counts` has never shown a real HOT/WARM/COLD distribution mid-conversation. If it still doesn't, `half_life_turns` is wrong.
2. `_survival` becomes an observation instead of a counterfactual replay.
3. **Memory vs dense rematch** — on span retrieval *and* a live energy term.

**Risk, named before the run:** a real energy term can *lower* answer recall, because items the conversation moved past are now actively demoted and some gold answers live there. That is intended behaviour. It is bounded — `min_energy` defaults to 0.0, so energy reorders but never filters, and at weight 0.2 against relevance 1.0 a highly-relevant cold item still wins. **If recall drops materially, the weight is wrong, not the coordinate.**

### Stage 3 — Phase 4's gate, asked properly

The design gates Phase 4 on *"evidence that COLD items are worth keeping at all."* For the first time COLD is a populated tier mid-conversation, so the question is answerable: **do COLD items hold answers that nothing else holds?**

**Kill criterion:** if COLD items contribute no answer that HOT/WARM or the chunk index does not already reach, Phase 4 does not earn its place and this document ends there. That is a legitimate finish, not a failure.

### Stage 4 — Phase 4, if the gate passes

Design is complete and unchanged: era summary **is** a `MemoryItem` (`memory_items.type` has no CHECK constraint, so a new type is additive); derived provenance via a separate `memory_derivation` table; greedy agglomerative complete-linkage clustering at τ=0.6; the centroid **is** the summary item's embedding, so no second ANN index.

### Stage 5 — H1 and H2

The theory's own claims. H2 (locality of gain) rests on one conversation pair, and the position-bin analysis that supported it was **retracted** — it does not replicate and every bin difference is inside noise. Needs the paid run (~$1.70).

## 4. Doc corrections owed

| Doc | Correction |
| --- | --- |
| `00 - Theory/00` §1.2 as-built note | Says the ranking equation realises "only the first term, all other weights 0." False since `0d86038` — `rank_score` implements the full scalar. It conflates two paths: *chunk* retrieval is raw cosine, *memory-item* ranking is the whole equation |
| `03 - Architecture/00` L108 | `effective_energy = energy × 0.5^(elapsed / half_life_s)`, "½-life 7d" (L42), "decayed forward from `last_access_at`" (L102) — all superseded by turns |
| `03 - Architecture/00` L108 | Also already stale independently: says `reheat` adds `+0.25` capped at 1.0; it has been saturating since `0d86038` |
| `05 - Standards/00` | Schema v4 + the turn ordinal as a normative field |
| `08 - Analysis/01` | The COLD-unreachability finding needs its reinterpretation recorded, not deleted |

## Verification block

```powershell
git log --all --oneline -- arch_instructions.md   # expect exactly one commit: 0871e05
git log --all --oneline -S half_life_turns        # expect empty before Stage 1 lands
pixi run -e dev pytest -q -m "not slow"           # 523 before Stage 1; expect growth
```

The first two are the evidence for §2 and should be re-runnable by anyone who doubts it.
