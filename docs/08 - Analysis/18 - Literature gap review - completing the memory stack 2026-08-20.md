# Literature gap review — what would complete the memory stack

**Date**: 2026-08-20
**Method**: five parallel web surveys (agent memory systems; graph/associative retrieval; temporal & episodic memory; adaptive retrieval & completeness; LongMemEval SOTA), every load-bearing claim verified at its source by the surveying agent, then crossed against this project's own sealed miss analyses ([14 - Query answer joint failure taxonomy](14%20-%20Query%20answer%20joint%20failure%20taxonomy%202026-08-27.md), [16 - Remaining miss memory ownership analysis](16%20-%20Remaining%20miss%20memory%20ownership%20analysis%202026-08-27.md)).
**Anchor facts from our own data**: best replay-verified arm 72/100; the 28 misses are 10 numeric-reduction, 9 temporal-ordering/interval, 4 synthesis, 4 direct/insufficiency, 1 set-join; **16 of 28 are operator failures despite full registered-source coverage**.

## TL;DR

The system is not missing retrieval machinery. Across ~40 verified papers and system reports, the convergent finding is that what separates this stack from "complete" is **three subsystems and two runtime gates**, all downstream or upstream of retrieval:

1. **Event-time understanding at ingest** (normalize relative dates to absolute intervals; bi-temporal fact validity) — the single highest-evidence lever in the field.
2. **Deterministic computation at answer time** (date arithmetic, counting, set operations over retrieved evidence) — exactly our 16/28 operator-failure class.
3. **A synthesis tier above extraction** (entity summaries, belief records, hub consolidation) — what multi-session aggregation questions need and top-k retrieval cannot provide.
4. **A runtime sufficiency/abstention gate** — we already compute the signal (closure unmet-obligations); nothing consumes it at answer time.
5. **Reasoning-based bridge retrieval** for multi-hop — our attention-feedback round cannot mint a hop-1 answer entity it has never seen.

Meanwhile three of our mechanisms have **no published counterpart**: the closure engine's obligation/witness receipts (closest neighbors are cryptographic [V3DB] or statistical [Sequential-EDFL] — ours is the only structural-completeness proof at the application level), the CAV write-time concept signatures, and attention-head surprise segmentation as implemented. The closure engine in particular is publishable positioning, not a gap.

---

## 1. Where this system already leads

- **Closure receipts / provable retrieval completeness**: nothing in IR/RAG literature compiles quantifier set-programs into obligations with budget receipts and witnesses. V3DB (arXiv:2603.03065) proves top-k exactness cryptographically against a committed snapshot; Sequential-EDFL (arXiv:2510.06478) certifies information sufficiency statistically. Neither has answer-set semantics. **Genuinely novel.**
- **Hebbian co-access arm**: ahead of published work; the closest publication is HeLa-Mem (ACL 2026, arXiv:2604.16839), which validates the design and adds one thing we lack (see §2.3).
- **Raw-chunk substrate + extraction (not extraction-only)**: validated by the MemPalace critique (arXiv:2604.21284) and by LongMemEval's own ablation showing fact-extraction-only *loses* accuracy.
- **Episode segmentation via surprise**: matches EM-LLM (ICLR 2025); our segmentation is competitive — its *retrieval-side* contiguity buffer is the missing half (§4).

## 2. The five completing additions (mapped to our misses)

### 2.1 Event-time normalization at ingest + bi-temporal validity — targets our 9 temporal misses

Three surveys independently converged here, and the SOTA survey confirms every system at ≥90% on LongMemEval temporal-reasoning does it.

- **Chronos** (arXiv:2603.16862): SVO event tuples with resolved ISO date *ranges* (granularity-aware: "recently" gets a wide window, "last Tuesday" one day), stored in an events calendar beside the turn calendar. Ablation: the events calendar alone = **+58.9%** over their baseline; 95.5% temporal-reasoning, 100% knowledge-update.
- **Zep/Graphiti** (arXiv:2501.13956): four timestamps per fact — event-timeline `t_valid`/`t_invalid` plus ingestion-timeline created/expired; contradiction *invalidates an interval*, never deletes. Temporal reasoning 45.1→62.4 (GPT-4o). Caveat: LLM-judged invalidation regressed knowledge-update on the weak model.
- **THEANINE** (NAACL 2025): supersede as a typed *chain* (the succession itself is retrievable), complementing interval validity.
- **Event-time recency** (arXiv:2601.07468): decay on |query_time − event_time| for datable memories, not turn distance — a future appointment mentioned 200 turns ago should not be cold.

**Integration here**: a normalization pass in `application/ingest_workflow.py` anchored to the session-date boundary turns we already inject, emitting `[event_start, event_end]` beside the provenance timestamp; `valid_from/valid_until` on distilled memories with supersede setting `valid_until` on the event timeline; `latest`/`terminal` stances in `search/closure` resolving over event time; time-window predicates in query-program compilation (the LongMemEval paper's own time-aware query expansion: ~+11.3% recall).

### 2.2 Deterministic answer-time operators — targets our 16 operator failures and 10 numeric misses

Our own taxonomy already concluded "the next answer-stage gain depends more on deterministic operators" — the literature agrees and quantifies it:

- **TReMu** (ACL 2025 Findings, arXiv:2502.01630): LLM generates and executes Python for date subtraction/interval comparison instead of in-token arithmetic — temporal QA **29.8 → 77.7** (GPT-4o).
- **TISER** (ACL 2025, arXiv:2504.05258): prompt-side variant — materialize an explicit timeline from retrieved evidence, reason over it, self-verify against the context. SOTA on temporal benchmarks; synthesis-stage-only change.
- **Chronos error analysis**: counting/arithmetic errors persist even at 90%+ retrieval quality and shrink only with more reasoning compute — independent confirmation that retrieval work cannot touch this class.

**Integration here**: when a query program carries an ordered/interval/count stance, emit the retrieved evidence's normalized values into a compute step (sandboxed code or a precomputed delta/count table injected into the prompt) before synthesis; closure receipts can bind the computed values to source order. This is the highest-priority item because our miss data says so, not just the literature.

### 2.3 A synthesis tier above extraction — targets multi-session aggregation (the field-wide floor)

The SOTA survey shows multi-session is the worst category for *every* top system (83–88.7 even at 95%+ overall) — and the systems that do best there all maintain synthesized state:

- **Hindsight** (ACL 2026 demo, arXiv:2512.12818): four networks — facts, experiences, *entity summaries*, *evolving beliefs* — with a reflect operation; 83.6% LongMemEval with a 20B open model; multi-session 21.1→79.7 in secondary reporting.
- **Generative Agents reflection** (UIST 2023) is the primitive: periodic insight statements *citing their evidence memories* — a format that maps directly onto our provenance validator (a reflection is a memory whose witnesses are its cited children).
- **HeLa-Mem** (ACL 2026): graph-analytic trigger — dense hubs in the Hebbian graph get distilled into semantic entries. We already have the hub detector's substrate (co-access graph + centrality); this is the episodic→semantic consolidation step our consolidation graph never takes.
- **Mem0** (arXiv:2504.19413) write-time adjudication: ADD/UPDATE/DELETE/NOOP against top-k similar memories — we only merge *exact* duplicates; near-duplicate revision ("moved from Austin to Seattle") coexists today. **Sleep-time compute** (Letta, arXiv:2504.13171; +13–18% on stateful benchmarks) is the natural budget for all of this without write-path latency.

**Integration here**: derived-memory tier whose provenance is a set of quoted source memories (preserves the cannot-write-unquoted invariant); trigger from consolidation-graph hubs; run in an inter-session pass.

### 2.4 Runtime sufficiency/abstention gate — unclaimed territory, and we hold a unique asset

- **Sufficient Context** (ICLR 2025, arXiv:2411.06037): adding RAG context *destroys* abstention (Claude abstains 84%→52% with context); a sufficiency autorater + self-confidence jointly gate answer-vs-abstain (+2–10% among answered).
- The adaptive-retrieval survey verified against this codebase that our sufficiency/abstention machinery lives **only in eval** (`eval/sufficiency.py`, `eval/policy_gate.py`) — nothing gates the runtime answer path.
- Our differentiator: the closure engine's **unmet-obligation-with-exhausted-budget signal is exactly the "memory does not contain this" evidence the literature estimates statistically**. Nobody else has it; we don't surface it to the answerer. The SOTA survey adds: virtually no post-2025 system even reports the `_abs` subset — abstention is unclaimed leaderboard territory.
- **CRAG** (arXiv:2401.15884) is the retry-side complement: a lightweight evaluator gating whether the second retrieval round fires and which strategy it uses.

### 2.5 Reasoning-based bridge retrieval — targets multi-hop / set-join

- **IRCoT** (ACL 2023) → **Search-R1** (arXiv:2503.09516, +41% over RAG baselines) → **"When Iterative RAG Beats Ideal Evidence"** (TMLR 2026, arXiv:2601.19827: staged reason→retrieve beats even oracle all-evidence-upfront by up to 25.6 points on multi-hop).
- The structural point: our attention-feedback second round re-weights semantically *adjacent* memory; it cannot form a query containing a hop-1 **answer entity** that never co-occurs with the question's terms. One LLM step emitting an intermediate answer + bridge query, fed through the existing facet-slot machinery, closes this.

## 3. SOTA context (verified numbers; reader model is a huge confounder)

Full-context GPT-4o baseline: 60.2–60.6%. Oracle: 82.4–87.0. Verified top tier on LongMemEval-S: Chronos 95.6 (Opus 4.6 reader) / 92.6 (GPT-4o); Mastra Observational Memory 94.87 (gpt-5-mini) / 84.23 (GPT-4o); ByteRover 92.8; Honcho 90.4 (and the only published LongMemEval-M number: 88.8%); Hindsight 91.4 (Gemini-3) / 83.6 (20B). Flagged: Supermemory's 98.6% is a self-declared parody; Mem0's 94.4 has an undisclosed reader and third parties measure them at 60–70%; several "SOTA" claims are vendor-self-graded. LongMemEval-S is near-saturated with strong readers (judge noise ~3–5%); the field is moving to LongMemEval-M and V2. Category patterns: temporal fixed by write-time timestamps/intervals everywhere, never by embeddings; knowledge-update fixed by versioning at write time; single-session-assistant punishes user-only extraction (Zep 80.4 vs full-context 94.6) — worth checking our extraction lanes cover assistant utterances.

## 4. Secondary refinements (cheap, evidenced)

- **Episode contiguity buffer** (EM-LLM): retrieve ±k sequence-neighbors of each episode hit — nearly free given our sequence numbers; part of what beat InfLLM by 4.3%.
- **Adaptive-k scan budgets** (EMNLP 2025, arXiv:2506.08479): similarity-gap statistics set the initial budget for `all`-quantifier programs; no LLM calls; receipts machinery unchanged.
- **LLM-free phrase layer** (HippoRAG 2 + LazyGraphRAG): noun-phrase co-occurrence nodes as a third edge type in the association graph — the shared symbolic intermediary chunks lack, and it fixes Hebbian cold-start; add embedding-threshold synonym/alias edges (HippoRAG 1) for cross-session renames. LazyGraphRAG proves this works with zero LLM index cost.
- **Compression-as-denoising** (SeCom, ICLR 2025): index a compressed representation of each episode, keep the raw span as provenance payload.
- **Typed lanes**: tag speaker role and coarse type (preference/event/update/assistant-info) at write time; route closure obligations to type-filtered sub-indexes.
- **Community structure without summaries**: Leiden over the existing co-access+attention graph as diffusion pools — the structural benefit of GraphRAG within the no-LLM-artifact constraint.
- **EpBench** (arXiv:2501.13121) as an eval addition: isolates whether our episodes support cue-addressable, order-correct multi-event recall — a property LongMemEval QA cannot see directly.

## 5. What NOT to build (negative evidence)

- **Full entity-KG migration**: Mem0's own A/B (graph vs non-graph: 68.44 vs 66.88 overall, graph *loses* single-hop, 3x slower, 2x tokens) says entity graphs pay only on temporal/relational slices — targeted additions beat migration.
- **RAPTOR-style summary trees as default**: loses to plain dense retrieval on multi-hop (MuSiQue 28.9 vs 45.7); justified only as a routed aggregation path — and LazyGraphRAG's query-time laziness dominates it under our constraints.
- **GraphRAG global community summaries**: underperforms standard RAG on ground-truth-scored benchmarks (WildGraphBench); the original wins were LLM-judge comprehensiveness metrics.
- **Chasing the last LongMemEval-S points**: Chronos documents several wrong ground-truth answers in the benchmark; judge noise ~3–5%; the frontier is M/V2.

## 6. Suggested experiment order (mapped to the locked protocol)

| # | Addition | Miss class targeted | Expected mechanism of gain |
| --- | --- | --- | --- |
| 1 | Answer-time compute scaffold (TReMu/TISER hybrid) | 10 numeric + part of 9 temporal (16 operator failures) | Deterministic arithmetic where coverage already exists |
| 2 | Event-date normalization + event-time keys | 9 temporal, artifact-global ordering | Retrieval and ordering by world time, query time-windowing |
| 3 | Bi-temporal supersede (`valid_from/until`, chains) | knowledge-update, "before X" questions | Interval-aware latest/terminal stances |
| 4 | Closure-signal abstention gate at runtime | 4 direct/insufficiency (incl. Q42-class) | Unmet-obligation ⇒ abstain/second-round |
| 5 | Bridge-query round (IRCoT-style) | multi-hop remnants of EM/set-join misses | Hop-1 answer entities enter the query |
| 6 | Hub distillation + entity summaries (sleep-time) | EM/episodic dispersed-join misses (10) | Aggregations exist as retrievable statements |

Items 1–4 are independent of each other and of the retrieval stack; each is testable under the existing matched-arm protocol without touching the locked baselines.
