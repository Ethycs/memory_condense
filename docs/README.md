# memory_condense — documentation tree

**Status**: Living Document
**Date**: 2026-08-22 (adds the audited fixed-stage S1 development diagnostic)
**Applies to**: the whole repository
**Depends on**: [`Agentic Technique Master.md`](../Agentic%20Technique%20Master.md) — the style guide governing this tree

This tree follows the folder system in the style guide: each numbered folder is a prerequisite for the folders after it. A change is only "real" when backed by at least one of the three lanes — tests, documentation, code.

> **Built and locally measured; external competitiveness remains open.** Campaign artifacts report 10/10 judge verdicts on a selected ten-question LongMemEval-S development slice; a structural audit verifies their internal bindings, not provider/judge execution or factual correctness. The exact frozen v3 treatment has now also run provider-free retrieval over all 100 validation questions: mean labeled evidence-source recall was 87.6%, every labeled source was recovered for 82%, and literal gold text appeared in 48% of final contexts. Those are retrieval diagnostics, not answer accuracy; no validation responder or judge call was made. The earlier artifact result did not generalize at the evidence-admission layer. This validation population is now analysis-used, Mem0 production remains NO-GO, and a tuned v4 system needs a new untouched confirmation population. `git log --oneline` and the machine-readable artifacts remain the authority over prose.

> **Current 1M synthesis result.** On the original ten-question development concatenation, the repaired v3 Terra policy held retrieval fixed and reached 6/10 exact match, 0.901019 mean F1, and 10/10 independent Sol semantic accuracy at S1, S2, and S3. The exact replay made zero provider calls and reproduced identical artifact bytes. This is diagnostic development evidence, not a target-gate result: the population is only ten and the structured synthesis call allowed 4,096 output tokens rather than the frozen answer-stage allowance of 256. The >=95% gate remains unpassed; see Research Log 25.

> **Fixed-stage campaign status.** The 8,000/256-token fixed-S1 path has now run on the original ten development questions. The completed root internally records 10 physical Terra calls and 10 physical Sol calls with zero SDK retries and byte-identical no-call replay, but an earlier sandbox-blocked root contains the same first Terra call key and request bytes. The combined lineage therefore has 11 reservations for 10 unique answer calls and violates the terminal-uncertainty/no-retry rule, making this diagnostic protocol-ineligible. The sealed Sol score remains 9/10 and `insufficient_population`; its sole negative on approximate `Close to 1300` is likely an adjudication false negative, but no post-hoc appeal is allowed. The formal 100Q run must start with network escalation/authorization, >=95% remains unproved, and Mem0 remains unrun. See Research Log 27 for the audit and Research Log 26 for the 100Q protocol.

> **Code/evidence boundary.** The organized source tree is implementation epoch
> v4. Frozen validation-v3 evidence still certifies commit
> `bfa5b6daf6a5e61881ac10f0555e5d9972f9e1c2` and implementation SHA
> `452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83`.
> Because the implementation digest includes package-relative paths, v3 caches
> cannot be relabeled for v4; see `03 - Architecture/03 - Code Package Layout.md`.

## Reconciliation state (2026-08-16)

| Doc | Reconciled? | Substance of the change |
| --- | --- | --- |
| `01 - Design/00 - Original Architecture Plan.md` | ✅ | Phases 0,1,2,3,5 and live relational consolidation (4A) built; materialized cold summaries (4B) remain unbuilt |
| `01 - Design/01 - Eval Design…` | ✅ | Retired-model BUG documented; judge≠responder and token instrumentation both **resolved**, not open |
| `02 - Implementation/00 - Setup…` | ✅ | 48-test baseline → **366**; hardcoded `dim=1024` bug marked fixed; schema-v2 migration gotcha added |
| `02 - Implementation/01 - Running the Eval Harness.md` | ✅ | Rewritten for four CLI modes; benchmark data sources + cost warning; sweep is 54 configs, not 48 |
| `02 - Implementation/03 - Qwen3 Prefix Attention Lab.md` | **experimental / integrated** | Seven-layer Qwen3-8B BF16 prefix, compact persistent CAV/QK/OV artifacts, bounded dual QK/heat reads, source-aware packing, safe admission, and physical pruning; public benchmarking remains open |
| `02 - Implementation/04 - Episode-Primary Latent Evidence Fusion.md` | **experimental / resident A+B implemented** | Exact query-preserving Qwen atom rows and same-GPU atomic K-latent matched fusion now pass provider-free, CUDA, and pinned-checkpoint smoke gates; extractive rendering, router training, route-bearing v2 evaluation, and any quality claim remain open |
| `02 - Implementation/05 - As-Built Mathematical Reference.md` | **implemented / test-covered** | Exact working formulas and edge cases for BM25/TF-ISF/RRF, co-access serving, heat diffusion, causal transitions, episodic surprise/refinement, coverage energies, forced choice, and evaluation metrics |
| `03 - Architecture/00 - System Overview.md` | ✅ | Diagram and every subsystem rewritten; "there is no condensation yet" was false |
| `03 - Architecture/01 - Native Hypergraph Memory Plane.md` | **new / proposed** | Event-centric hypergraph for live QK/OV/CAV observations, with the measured pairwise graph retained as a bounded serving projection; no durable request-derived transformer token state (static model/tokenizer assets excluded) |
| `03 - Architecture/02 - Query-Conditioned Bayesian Coverage Loop.md` | **implemented / prefix measurement pending** | Primary full-width Qwen3-8B layers 0–5 with layer-5 QK/OV transport-affinity grouping; secondary compact-INI classifier; recall-safe coverage ordering and zero durable transformer state |
| `03 - Architecture/03 - Code Package Layout.md` | **new / current** | Maps responsibility packages and the objects → transformations → stateful-workflows rule, stable facades, canonical imports, size gates, and the path-sensitive validation-v3 → implementation-v4 evidence boundary |
| `04 - Reference/01 - Vocabulary.md` | ✅ | Lifecycle + retrieval terms moved out of *(planned)*; BM25/hybrid/α/`term_count`/`UsageStats`/F1/provenance added |
| `05 - Standards/00 - MC-STD-DATA-v0.md` | ✅ | Schema v2 + migration path; new normative clauses 8–10 (provenance, no destruction, migrate-in-place). Still **DRAFT** |
| `06 - Roadmaps/00 - Gap Analysis and Roadmap.md` | ✅ | Status table and tiers rewritten; Decision Point now *unblocked* but still *open*. **Partly superseded 2026-08-15** — see below |
| `06 - Roadmaps/01 - Delivering the Specified System.md` | **new** | Decay was specified in wall-clock seconds; the design intent is per-turn. The energy term therefore contributed a constant, and **every memory-arm number is void** — including the Phase 4 verdict. Carries the git evidence that the spec was wrong from commit one, and the five-stage delivery sequence |
| `00 - Theory/00 …` | — | Not touched; stable by policy (corrections only) |
| `00 - Theory/01 …` | **new draft** | Extracted-head associative memory with CAV/J-Space concepts, QK routing, OV transport, live-head pruning, and a falsification sequence; the prefix prototype now has a locked local token-saving result but no fresh recall gain |
| `00 - Theory/03 …` | **implemented / locally measured** | Schema-v9 prompt/response binding across typed memories and evidence; repeated activation, turn decay, bounded two-hop reads, and transient CAV/QK/OV weighting |
| `00 - Theory/04 - From Top-K Recall to Proof-Carrying Factual Retrieval.md` | **implemented / development-evidenced** | Separates reachability, event identity, packet sufficiency, role/time integrity, and proof scope; explains the structural scan, scalar bypass, event deduplication, scoped closure, and reproducibility repairs |
| `00 - Theory/05 - EM-LLM Episodic Discourse Closure for Diffuse Retrieval.md` | **provider-free prototype / unmeasured** | Implements source-grounded episodes, discourse obligations, bounded closure, atomic packing, and an evidence-bound Qwen prefix OV-transport change/refinement path; paper-exact EM token NLL and raw-key modularity remain unbuilt ablations |
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
| `10 - Research Log/15 - 2026-08-18 - Policy-locked 1M-context answer pilot.md` | **development pilot artifact** | Campaign artifacts report ten of ten positive judge verdicts; structural consistency is verified, but provider/judge execution and factual correctness are not independently authenticated; mean reported responder prompt was 2,342 tokens from 1,039,203 transcript tokens |
| `10 - Research Log/16 - 2026-08-18 - V3 retrieval freeze and validation campaign.md` | **frozen development treatment** | Final no-provider replay reached 100% source and scored answer-value coverage at a mean 1,986 returned tokens; exact cache receipts, prompt-proxy identity, a 100-question campaign plan, and the corrected Mem0 protocol are frozen, but no held-out provider calls have run |
| `10 - Research Log/17 - 2026-08-18 - Locked treatment handoff and discourse closure frontier.md` | **operational handoff / incomplete goal** | Consolidates the treatment, ten prepared cache shards, hard invariants, controlled Mem0 tooling, current test evidence, explicit NO-GO boundaries, and the proposed general-purpose Grounded Discourse Closure RAG design |
| `10 - Research Log/18 - 2026-08-18 - Validation v3 provider-free retrieval audit.md` | **100-question provider-free audit / retrieval gate failed** | Exact frozen-v3 replay across all ten validation shards: 87.6% mean evidence-source recall, 82% all-source recovery, zero post-coverage closures, zero provider calls, unchanged cache hashes, and an explicit development-to-validation generalization gap; answer accuracy remains unmeasured |
| `10 - Research Log/21 - 2026-08-21 - Retrieval nesting and fresh 1M episode-primary test.md` | **1M functional ablation / retrieval regression** | A fresh validation-offset-0 `episode_primary` route completed end to end but replaced the v3 authority and fell to 3/10 literal reachability; this is not the original concatenated-memory control |
| `10 - Research Log/22 - 2026-08-21 - Recall-guarded cumulative retrieval.md` | **measured 1M development ladder** | The original 1,039,203-token development concatenation ran through four strictly nested provider-free stages: S0 recovered every labeled source and 5/10 literal answers; S1 improved mean evidence F1 by 4.62%, while S2 and S3 added no further scored gain under the cap |
| `10 - Research Log/23 - 2026-08-21 - Episodic evidence scoring and synthesis.md` | **measured local synthesis / negative answer result** | A pinned Qwen3-0.6B inspected all 176 episodic additions and made 12 unique S1-S3 answer calls; its historical raw-p(A) answerability proxies found no useful S2 addition, while every stage scored 0/10 exact match and 0.010227 mean F1, with no independent judge or calibrated-density claim |
| `10 - Research Log/24 - 2026-08-21 - LiteLLM Terra episodic synthesis and rescoring.md` | **measured provider synthesis / improved development answer result** | A strict, 12-call checkpointed Terra arm held retrieval fixed, labeled all 176 additions, and raised S1 to 5/10 exact match and 0.718433 F1; S2/S3 reached 4/10 and 0.706806, all five S2-only additions were labeled irrelevant/none, and no independent judge or held-out claim is made |
| `10 - Research Log/25 - 2026-08-22 - Independent Sol judge and v3 synthesis repair.md` | **independently judged development diagnostic / formal gate still open** | A separate Sol path reconstructed and recounted every sealed Terra prompt, judged v2 at 9/10, and judged the runtime-gold-blind v3 synthesis repair at 10/10 for S1-S3; byte-identical zero-call replay passed, but the ten-question population and 4,096-token synthesis output allowance make this ineligible for the locked answer-stage gate, and Mem0 remains unrun |
| `10 - Research Log/26 - 2026-08-22 - Fixed-stage S1 and locked 100Q campaign.md` | **reproducible launch surface / formal gate still open** | Locks S1 under the original 8,000/256 answer budget, adds ten independently sealed cumulative retrieval shards, an independent Sol >=95%/100Q gate, and a fail-closed schema-v3 Mem0 comparison boundary; real preflights pass, but the GPU/provider campaigns and fair Mem0 arm remain unrun |
| `10 - Research Log/27 - 2026-08-22 - Fixed-stage S1 LiteLLM development diagnostic.md` | **operational development diagnostic / protocol-ineligible** | The completed root internally records 10 fixed-S1 Terra and 10 independent Sol physical calls with zero SDK retries and exact offline replay, but the sandbox-blocked root duplicates the first Terra reservation, yielding 11 reservations for 10 unique calls across the lineage; Sol's sealed 9/10 remains `insufficient_population`, its approximate-answer negative is likely an adjudication false negative without a preregistered appeal, and neither the formal 100Q result nor Mem0 exists yet |
| `10 - Research Log/28 - 2026-08-22 - First locked validation shard seal.md` | **first locked validation retrieval shard / formal gate still open** | Offset 0 sealed all ten nested S0--S3 questions under the 7,000-context/8,000-prompt caps with zero provider calls and zero retained request-token state; canonical replay and an independent receipt/store audit passed, offsets 10--90 passed preflight, offset 10 is running, and the 100Q merge, Terra/Sol score, and Mem0 arm remain incomplete |
| `10 - Research Log/29 - 2026-08-22 - Locked validation offset 10 seal.md` | **second locked validation retrieval shard / formal gate still open** | Offset 10 sealed and replayed all ten provider-free ladders; S1 admitted 177 evidence rows, while S2 and S3 admitted none because the frozen context budget was exhausted. Offset 20 is running, and the 100Q merge, Terra/Sol gate, and Mem0 comparison remain incomplete |
| `07 - Status Reports/…` | ✅ | Six dated handoffs through 2026-08-19; the 2026-08-15 report remains the retrieval-measurement handoff, while the 2026-08-19 reports cover the later simplification audit and implementation |

## The tree

```
docs/
├── 00 - Theory/           Retrieval, proof-carrying coverage, associative memory, and episodic closure
├── 01 - Design/           The original architecture plan + eval design rationale
├── 02 - Implementation/   Setup, eval modes, Qwen labs, frozen contracts, and as-built mathematics
├── 03 - Architecture/     The as-built system map, package layout, and proposed hypergraph memory plane
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
- "How did factual retrieval improve, and what remains unproven?" → `00 - Theory/04 - From Top-K Recall to Proof-Carrying Factual Retrieval.md`.
- "How will diffuse evidence use EM-LLM?" → `00 - Theory/05 - EM-LLM Episodic Discourse Closure for Diffuse Retrieval.md` — the general closure mechanics and a bounded transient Qwen OV-transport episode signal are implemented provider-free. The latter is an attention-head semantic-change adaptation, not paper-exact token-NLL surprise, and still requires matched evaluation against fixed and embedding-change controls.
- "Where does the code live, and which imports are supported?" → `03 - Architecture/03 - Code Package Layout.md`.
- "How would a native hypergraph interact with live memory?" → `03 - Architecture/01 - Native Hypergraph Memory Plane.md` — canonical higher-order observations, pairwise serving projections, bounded traversal, and event-aware pruning.
- "How are complete sets deduplicated without losing distinct events?" → `03 - Architecture/02 - Query-Conditioned Bayesian Coverage Loop.md` — a primary six-layer Qwen3-8B QK/OV affinity arm plus a secondary compact-INI classifier, followed by recall-safe representative-first packing; the locked baseline isolates a 100%-raw versus 94.7%-packed gap and the prefix treatment is pending measurement.
- "What's left to build?" → `06 - Roadmaps/00 - Gap Analysis and Roadmap.md`.
- "How do I run it?" → `02 - Implementation/01 - Running the Eval Harness.md` (start with the free `--compare` mode).
- "How do I run the Qwen attention-prefix experiment?" → `02 - Implementation/03 - Qwen3 Prefix Attention Lab.md`.
- "How will episode retrieval feed the K-latent attention fusion stage?" → `02 - Implementation/04 - Episode-Primary Latent Evidence Fusion.md` — a design-frozen, query-conditioned GPU feature-to-router contract with no trained or measured fusion claim yet.
- "What exact math does the working implementation execute?" → `02 - Implementation/05 - As-Built Mathematical Reference.md` — the code-aligned equations, defaults, edge cases, tie-breaks, and focused test map for working retrieval paths that were previously implicit.
- "How do I actually *use* it day to day?" → `02 - Implementation/02 - MCP Integration.md` — the memory system is exposed to Claude Code as an MCP server.
- "Is this competitive?" → **not established**. `10 - Research Log/18` records the exact 100-question provider-free v3 validation audit: mean evidence-source recall fell to 87.6%, all-source recovery to 82%, and answer accuracy remains unmeasured. The same-budget Mem0 arm is also still open.
- "What is the large-model attention-head memory idea?" → `00 - Theory/01 - Extracted Attention Heads as Recursive Associative Memory.md` — a **DRAFT** whose first CAV/live-head prototype is implemented, including the full-teacher J-Space implication.
- "How do later prompts consolidate connected memory partitions?" → `00 - Theory/03 - Prompt-Driven Systems Consolidation.md` — schema-v9 causal binding plus repeated co-activation across semantic memories and evidence, including bounded iterative reads, the transient Qwen hyperplane seam, and anti-self-reinforcement rules.
- "Did the downloaded Qwen prefix produce usable CAVs?" → `10 - Research Log/02 - 2026-08-16 - Qwen3 prefix CAV gate.md` — yes on the first controlled local probe; this is not yet a retrieval result.
- "Did extracted heads improve live-memory retrieval?" → `10 - Research Log/03 - 2026-08-16 - Live Qwen head memory smokes.md` — calibrated layer-1 heads and temporal direction reached 1.000 R@1/R@3 on four development links; direct QK/OV failed and blind replication remains open.
- "Did persistent CAV/QK memory save tokens on unseen source families?" → `10 - Research Log/04 - 2026-08-16 - Safe associative memory confirmation.md` — yes locally, without lowering baseline recall; a fresh recall gain and public-benchmark result remain unconfirmed.
- "Can attention heat control how much memory each source contributes?" → `10 - Research Log/05 - 2026-08-16 - Source heat diffusion development.md` — implemented as a bounded external scalar walk plus source-aware packing; the selected dual QK/heat policy is posthoc development evidence awaiting a new locked split.
- "What exactly does 95% long-chat accuracy mean?" → `10 - Research Log/06 - 2026-08-16 - 95 percent long-chat target.md` — answer-stage judge accuracy, minimum sample size, 8k hard prompt cap, locked LongMemEval partitions, and the experiment ladder.
- "Has live schema-v8 consolidation run through the real Qwen checkpoint?" → `10 - Research Log/08 - 2026-08-16 - Real Qwen consolidation path.md` — yes on a temporary store copy; it validates execution and memory bounds, not a recall gain.
- "Did causal Qwen consolidation improve the operational long-chat probe?" → `10 - Research Log/09 - 2026-08-16 - Causal binding reaches 97.4 percent evidence recall.md` — yes on the locked local literal-evidence test (38/39, no regressions); answer-stage judged LongMemEval remains the primary gate.
- "Did it answer LongMemEval questions from a 1M-token chat?" → `10 - Research Log/15 - 2026-08-18 - Policy-locked 1M-context answer pilot.md` — its campaign artifacts report 10/10 positive judge verdicts with a mean 2,342-token legacy local prompt proxy; structural bindings are verified, but provider/judge execution and factual correctness are not independently authenticated.
- "What is the frozen v3 treatment and validation plan?" → `10 - Research Log/16 - 2026-08-18 - V3 retrieval freeze and validation campaign.md` — the final no-provider replay, exact artifact/cache identities, ten-shard held-out plan, prompt-proxy semantics, and corrected Mem0 comparison boundary.
- "What is operational now, what is still unmeasured, and what comes next?" → `10 - Research Log/17 - 2026-08-18 - Locked treatment handoff and discourse closure frontier.md` — the current readiness matrix, exact invariants and hashes, Mem0 production NO-GO, authorization gates, and the general-purpose diffuse-retrieval design.
- "Did the frozen v3 treatment generalize to all 100 validation questions?" → `10 - Research Log/18 - 2026-08-18 - Validation v3 provider-free retrieval audit.md` — not at the retrieval-admission level; it records the exact metrics, identities, post-run cache audit, and why no provider accuracy claim follows.
- "How are new retrieval methods added without replacing the strongest prior packet, and what happened at 1M?" → `10 - Research Log/22 - 2026-08-21 - Recall-guarded cumulative retrieval.md` — four provider-ready stages preserve a frozen-v3-compatible protected root, then add direct episodes, representative episodes, and artifact-global closure. On the exact original 1,039,203-token development concatenation, only direct episodes improved a scored retrieval metric; later stages preserved evidence but added no further gain under the cap.
- "Did a larger synthesizer answer the cumulative 1M contexts and score the episodic additions?" → `10 - Research Log/24 - 2026-08-21 - LiteLLM Terra episodic synthesis and rescoring.md` — yes on the ten-question development concatenation: strict Terra synthesis reached 5/10 exact match and 0.718433 F1 at S1 with exact-quote citations, while both semantic and local numeric scoring found no strong S2-only evidence. This is not a held-out or independently judged result.
- "Did an independent judge verify the 1M answers, and did the synthesis repair work?" → `10 - Research Log/25 - 2026-08-22 - Independent Sol judge and v3 synthesis repair.md` — yes as a diagnostic on the same ten development questions: Sol scored v2 at 9/10 and the fixed-retrieval v3 policy at 10/10 for every stage, with exact zero-call replay. The formal >=95% result is still open because the locked gate requires at least 100 questions at one fixed stage and a 256-token final responder, while this structured synthesis call allowed 4,096 output tokens.
- "Is the fixed-stage 100Q memory test ready to launch, and what exactly is certified?" → `10 - Research Log/26 - 2026-08-22 - Fixed-stage S1 and locked 100Q campaign.md` — the sealed ten-shard retrieval, fixed-S1 Terra responder, independent Sol gate, replay contracts, exact identities, commands, and schema-v3 Mem0 boundary are implemented and preflighted; the expensive retrieval/provider campaigns and a fair Mem0 result are still outstanding.
- "Did the new fixed-stage answer and judge paths work live before the 100Q run?" → `10 - Research Log/27 - 2026-08-22 - Fixed-stage S1 LiteLLM development diagnostic.md` — operationally yes, but the diagnostic is protocol-ineligible: the completed root has 10 Terra plus 10 Sol physical calls with zero SDK retries and exact offline replay, while the retained blocked root duplicates the first Terra request reservation. The sealed score stays 9/10 despite a likely adjudication false negative; the formal validation must start with network escalation/authorization, >=95% remains unproved, and Mem0 remains unrun.
- "Has the locked 100Q retrieval campaign produced real shards yet?" → `10 - Research Log/28 - 2026-08-22 - First locked validation shard seal.md` and `10 - Research Log/29 - 2026-08-22 - Locked validation offset 10 seal.md` — yes for offsets 0 and 10. Both provider-free roots passed verify-only replay and independent receipt/store audits; offset 20 is running, while the remaining shards, merge, Terra/Sol gate, and Mem0 comparison are incomplete.

## The one distinction this tree tries hardest to keep

**Built ≠ measured, and locally measured ≠ externally competitive.** Passing tests establish implementation behavior. The local analyses and research logs establish only their stated datasets, splits, and metrics. The QK/CAV result currently supports token saving with recall non-regression on one locked fresh split; it does not support a general recall-gain claim. Any broader claim without a public benchmark is a bug — report it.

Dated status reports, analyses, research logs, and frozen scripts may cite the
flat module paths that existed when their artifacts were produced. Active code
examples use the canonical v4 package paths; historical evidence is not
mechanically rewritten.

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
