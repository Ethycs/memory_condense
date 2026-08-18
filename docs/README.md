# memory_condense — documentation tree

**Status**: Living Document
**Date**: 2026-08-18 (reconciled against the frozen v3 LongMemEval development treatment)
**Applies to**: the whole repository
**Depends on**: [`Agentic Technique Master.md`](../Agentic%20Technique%20Master.md) — the style guide governing this tree

This tree follows the folder system in the style guide: each numbered folder is a prerequisite for the folders after it. A change is only "real" when backed by at least one of the three lanes — tests, documentation, code.

> **Built and locally measured; external competitiveness remains open.** A ten-question LongMemEval-S development pilot reached 10/10 judge accuracy. The final v3 no-provider replay then reached 100% raw/packed source coverage and 11/11 scored answer-value components with a mean 1,986-token context from a 1,039,203-token transcript proxy. Prompt accounting is now explicitly a frozen local proxy with provider-usage postchecks. This is development evidence, not the required held-out minimum-100-question validation or a completed Mem0 comparison. `git log --oneline` and the machine-readable artifacts remain the authority over prose.

## Reconciliation state (2026-08-16)

| Doc | Reconciled? | Substance of the change |
| --- | --- | --- |
| `01 - Design/00 - Original Architecture Plan.md` | ✅ | Phases 0,1,2,3,5 and live relational consolidation (4A) built; materialized cold summaries (4B) remain unbuilt |
| `01 - Design/01 - Eval Design…` | ✅ | Retired-model BUG documented; judge≠responder and token instrumentation both **resolved**, not open |
| `02 - Implementation/00 - Setup…` | ✅ | 48-test baseline → **366**; hardcoded `dim=1024` bug marked fixed; schema-v2 migration gotcha added |
| `02 - Implementation/01 - Running the Eval Harness.md` | ✅ | Rewritten for four CLI modes; benchmark data sources + cost warning; sweep is 54 configs, not 48 |
| `02 - Implementation/03 - Qwen3 Prefix Attention Lab.md` | **experimental / integrated** | Seven-layer Qwen3-8B BF16 prefix, compact persistent CAV/QK/OV artifacts, bounded dual QK/heat reads, source-aware packing, safe admission, and physical pruning; public benchmarking remains open |
| `03 - Architecture/00 - System Overview.md` | ✅ | Diagram and every subsystem rewritten; "there is no condensation yet" was false |
| `03 - Architecture/01 - Native Hypergraph Memory Plane.md` | **new / proposed** | Event-centric hypergraph for live QK/OV/CAV observations, with the measured pairwise graph retained as a bounded serving projection; no durable request-derived transformer token state (static model/tokenizer assets excluded) |
| `03 - Architecture/02 - Query-Conditioned Bayesian Coverage Loop.md` | **implemented / prefix measurement pending** | Primary full-width Qwen3-8B layers 0–5 with layer-5 QK/OV transport-affinity grouping; secondary compact-INI classifier; recall-safe coverage ordering and zero durable transformer state |
| `04 - Reference/01 - Vocabulary.md` | ✅ | Lifecycle + retrieval terms moved out of *(planned)*; BM25/hybrid/α/`term_count`/`UsageStats`/F1/provenance added |
| `05 - Standards/00 - MC-STD-DATA-v0.md` | ✅ | Schema v2 + migration path; new normative clauses 8–10 (provenance, no destruction, migrate-in-place). Still **DRAFT** |
| `06 - Roadmaps/00 - Gap Analysis and Roadmap.md` | ✅ | Status table and tiers rewritten; Decision Point now *unblocked* but still *open*. **Partly superseded 2026-08-15** — see below |
| `06 - Roadmaps/01 - Delivering the Specified System.md` | **new** | Decay was specified in wall-clock seconds; the design intent is per-turn. The energy term therefore contributed a constant, and **every memory-arm number is void** — including the Phase 4 verdict. Carries the git evidence that the spec was wrong from commit one, and the five-stage delivery sequence |
| `00 - Theory/00 …` | — | Not touched; stable by policy (corrections only) |
| `00 - Theory/01 …` | **new draft** | Extracted-head associative memory with CAV/J-Space concepts, QK routing, OV transport, live-head pruning, and a falsification sequence; the prefix prototype now has a locked local token-saving result but no fresh recall gain |
| `00 - Theory/03 …` | **implemented / locally measured** | Schema-v9 prompt/response binding across typed memories and evidence; repeated activation, turn decay, bounded two-hop reads, and transient CAV/QK/OV weighting |
| `04 - Reference/00 - Competitive Landscape 2026.md` | — | Not touched this pass |
| `08 - Analysis/00 - Retrieval Ablation…` | ✅ | Sweep corrected to 54 configs; a position-bin analysis was added and then **retracted the same day** — it does not replicate on the second run pair, and every bin-to-bin difference is inside noise. The aggregate ablation result stands |
| `08 - Analysis/01 - Extraction and Decay Audit` | **new** | 70.6% of memory items never reach the prompt; COLD is unreachable by construction; the default extractor is 65% spurious `Constraint`s. All free, all previously unmeasured |
| `10 - Research Log/02 - 2026-08-16 - Qwen3 prefix CAV gate.md` | **new measurement** | Layers 0–5 passed held-out accuracy, bootstrap stability, and random-label controls for two project-relevant CAVs; layer 5 selected for the first live-memory prototype |
| `10 - Research Log/03 - 2026-08-16 - Live Qwen head memory smokes.md` | **new measurement** | Layer-5 CAV entry reached 0.750/0.875; calibrated layer-1 head/direction association reached 1.000 R@1/R@3 on four development links; fresh blind replication is required |
| `10 - Research Log/04 - 2026-08-16 - Safe associative memory confirmation.md` | **new confirmation** | On a locked fresh six-family split, safe CAV/QK arms preserved 83.3% hybrid recall while reducing prompt tokens by 1.3–2.7%; degree-two pruning removed 392/1,204 edges without a recall loss; no fresh recall gain was observed |
| `10 - Research Log/05 - 2026-08-16 - Source heat diffusion development.md` | **new development replay** | Two-hop dual allocation reserves one ranked-QK slot and one heat slot; degree-two replay preserved local recall while reducing selected text by 5.1–16.3%; pure heat lost the one development recovery, and no fresh recall gain is claimed |
| `10 - Research Log/06 - 2026-08-16 - 95 percent long-chat target.md` | **active target** | Locks 500 cleaned LongMemEval questions into 200/100/200 partitions and defines ≥95% judge accuracy under an 8k prompt ceiling as the hard gate |
| `10 - Research Log/08 - 2026-08-16 - Real Qwen consolidation path.md` | **operational smoke** | The real seven-layer BF16 prefix updated six schema-v8 edges from four packed pointers in 0.75 s after a 12.92 s startup load, retaining zero prompt/activation bytes; recall effect remains unmeasured |
| `10 - Research Log/09 - 2026-08-16 - Causal binding reaches 97.4 percent evidence recall.md` | **new development replay** | Four-arm chronological replay: original 35/39, packing-only 36/39, rank graph 37/39, Qwen graph 38/39 with no losses and zero retained transformer-state bytes; answer-stage evaluation remains open |
| `10 - Research Log/15 - 2026-08-18 - Policy-locked 1M-context answer pilot.md` | **development pilot** | Ten of ten LongMemEval-S answers passed the independent judge; mean responder prompt 2,342 tokens from 1,039,203 transcript tokens; selected-scope closure is non-global and ≥100 held-out validation remains open |
| `10 - Research Log/16 - 2026-08-18 - V3 retrieval freeze and validation campaign.md` | **frozen development treatment** | Final no-provider replay reached 100% source and scored answer-value coverage at a mean 1,986 returned tokens; exact cache receipts, prompt-proxy identity, a 100-question campaign plan, and the corrected Mem0 protocol are frozen, but no held-out provider calls have run |
| `07 - Status Reports/…` | ✅ | Three dated handoffs; **2026-08-15 is the current one** and supersedes the earlier test counts (48 → 366 → 523) |

## The tree

```
docs/
├── 00 - Theory/           Retrieval-weighted context plus extracted-head associative memory
├── 01 - Design/           The original architecture plan + eval design rationale
├── 02 - Implementation/   Setup, the four eval modes, and MCP/Claude Code integration
├── 03 - Architecture/     The as-built system map plus the proposed hypergraph memory plane
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
- "How would a native hypergraph interact with live memory?" → `03 - Architecture/01 - Native Hypergraph Memory Plane.md` — canonical higher-order observations, pairwise serving projections, bounded traversal, and event-aware pruning.
- "How are complete sets deduplicated without losing distinct events?" → `03 - Architecture/02 - Query-Conditioned Bayesian Coverage Loop.md` — a primary six-layer Qwen3-8B QK/OV affinity arm plus a secondary compact-INI classifier, followed by recall-safe representative-first packing; the locked baseline isolates a 100%-raw versus 94.7%-packed gap and the prefix treatment is pending measurement.
- "What's left to build?" → `06 - Roadmaps/00 - Gap Analysis and Roadmap.md`.
- "How do I run it?" → `02 - Implementation/01 - Running the Eval Harness.md` (start with the free `--compare` mode).
- "How do I run the Qwen attention-prefix experiment?" → `02 - Implementation/03 - Qwen3 Prefix Attention Lab.md`.
- "How do I actually *use* it day to day?" → `02 - Implementation/02 - MCP Integration.md` — the memory system is exposed to Claude Code as an MCP server.
- "Is this competitive?" → **not established yet**. `10 - Research Log/16` records the frozen v3 development treatment at 100% source/value coverage and 99.81% transcript-token-proxy savings, but the required ≥100 held-out run and same-budget Mem0 arm remain open.
- "What is the large-model attention-head memory idea?" → `00 - Theory/01 - Extracted Attention Heads as Recursive Associative Memory.md` — a **DRAFT** whose first CAV/live-head prototype is implemented, including the full-teacher J-Space implication.
- "How do later prompts consolidate connected memory partitions?" → `00 - Theory/03 - Prompt-Driven Systems Consolidation.md` — schema-v9 causal binding plus repeated co-activation across semantic memories and evidence, including bounded iterative reads, the transient Qwen hyperplane seam, and anti-self-reinforcement rules.
- "Did the downloaded Qwen prefix produce usable CAVs?" → `10 - Research Log/02 - 2026-08-16 - Qwen3 prefix CAV gate.md` — yes on the first controlled local probe; this is not yet a retrieval result.
- "Did extracted heads improve live-memory retrieval?" → `10 - Research Log/03 - 2026-08-16 - Live Qwen head memory smokes.md` — calibrated layer-1 heads and temporal direction reached 1.000 R@1/R@3 on four development links; direct QK/OV failed and blind replication remains open.
- "Did persistent CAV/QK memory save tokens on unseen source families?" → `10 - Research Log/04 - 2026-08-16 - Safe associative memory confirmation.md` — yes locally, without lowering baseline recall; a fresh recall gain and public-benchmark result remain unconfirmed.
- "Can attention heat control how much memory each source contributes?" → `10 - Research Log/05 - 2026-08-16 - Source heat diffusion development.md` — implemented as a bounded external scalar walk plus source-aware packing; the selected dual QK/heat policy is posthoc development evidence awaiting a new locked split.
- "What exactly does 95% long-chat accuracy mean?" → `10 - Research Log/06 - 2026-08-16 - 95 percent long-chat target.md` — answer-stage judge accuracy, minimum sample size, 8k hard prompt cap, locked LongMemEval partitions, and the experiment ladder.
- "Has live schema-v8 consolidation run through the real Qwen checkpoint?" → `10 - Research Log/08 - 2026-08-16 - Real Qwen consolidation path.md` — yes on a temporary store copy; it validates execution and memory bounds, not a recall gain.
- "Did causal Qwen consolidation improve the operational long-chat probe?" → `10 - Research Log/09 - 2026-08-16 - Causal binding reaches 97.4 percent evidence recall.md` — yes on the locked local literal-evidence test (38/39, no regressions); answer-stage judged LongMemEval remains the primary gate.
- "Did it answer LongMemEval questions from a 1M-token chat?" → `10 - Research Log/15 - 2026-08-18 - Policy-locked 1M-context answer pilot.md` — yes, 10/10 on the development pilot with a mean 2,342-token legacy local prompt proxy; provider input usage was unavailable, and held-out scale plus the Mem0 comparison remain open.
- "What is the frozen v3 treatment and validation plan?" → `10 - Research Log/16 - 2026-08-18 - V3 retrieval freeze and validation campaign.md` — the final no-provider replay, exact artifact/cache identities, ten-shard held-out plan, prompt-proxy semantics, and corrected Mem0 comparison boundary.

## The one distinction this tree tries hardest to keep

**Built ≠ measured, and locally measured ≠ externally competitive.** Passing tests establish implementation behavior. The local analyses and research logs establish only their stated datasets, splits, and metrics. The QK/CAV result currently supports token saving with recall non-regression on one locked fresh split; it does not support a general recall-gain claim. Any broader claim without a public benchmark is a bug — report it.

---

**Verification block**: run

```powershell
git log --oneline -1
git status --short
pixi run --frozen -e dev pytest -q
```

If the suite is green, this tree is accurate as far as it goes: the core and
association and source-heat paths are implemented and locally measured, while external
competitiveness remains unknown. The next evidence gates are improved
write-time association coverage and `--answer-recall` or another public/common
benchmark—not a larger transformer context or a third retrieval hop.
