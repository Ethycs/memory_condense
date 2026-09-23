# Five new 100-question evaluations and a raw-transcript build replay

**Status:** Complete — results include failures, not an unqualified success claim.  
**Started:** 2026-09-16. **Completed:** 2026-09-17.  
**Scope:** Five separate histories, 100 new questions each, then eight consecutive original build prompts. All workers are closed.

## Result

The five question tests scored **462/500 (92.4%)** under the unchanged Sol semantic grader. Combined warm median was **4.469 seconds**, mean **4.739 seconds**, p95 **6.893 seconds**, and average answer input **1,537 tokens**. There were **355/500** answers under five seconds. This does not establish the 95% target across histories.

The coding replay produced actual changes in **16 files**. Final selected regression and architecture checks: **240 passed**. Independent behavioral review: **four passed, four failed**. The model completed an immediate compatibility milestone, but the final implementation did not retain the intended reinforcement behavior across rapid turns. This is a useful implementation result with concrete defects, not a clean behavioral pass.

- [Five-test report and every recorded miss](../../eval_results/native-spine-five100-20260916-r1/report.md)
- [Prompt-by-prompt coding assessment and generated replies](../../eval_results/native-spine-build-replay-20260916-r1/build-report.md)
- [Independent behavioral failures](../../eval_results/native-spine-build-replay-20260916-r1/independent-behavior.log)
- [Final validation receipt](../../eval_results/native-spine-build-replay-20260916-r1/validation-report.json)
- [Actual candidate patch](../../eval_results/native-spine-build-replay-20260916-r1/candidate.patch)

## Five 100-question tests

| History | Eligible raw tokens | Correct | Warm median | p95 | Mean input tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,190,100 | 93/100 | 4.509 s | 7.824 s | 1,544.03 |
| 2 | 1,149,903 | 92/100 | 4.323 s | 6.538 s | 1,599.01 |
| 3 | 1,139,166 | 90/100 | 4.594 s | 6.323 s | 1,524.89 |
| 4 | 1,131,414 | 96/100 | 4.553 s | 8.041 s | 1,538.09 |
| 5 | 1,127,693 | 91/100 | 4.254 s | 6.532 s | 1,480.85 |

Exactly five histories were used, with 500 distinct question-source bodies; the previous 100 questions were excluded. An optional scope clarification received no answer, so the stated default was five different histories with 100 new questions each. The old 100-history controller was not resumed.

All questions and references were locked before candidate answering. Questions and references did not enter memory ingestion. Each history followed normal `MemoryCondenser.ingest_many`, native snapshot and parent-vector installation, close, and a separate-process read-only reopen. Answering then used fresh retrieval and serial `codex_sdk/gpt-5.6-sol` responses. The unchanged original semantic grader scored all 500 answers. All answers stopped normally, and all **500 exact raw evidence packets** passed independent reconstruction.

The routing/reader policy remained the prior v7 user-spine method: direct8/2048, exact user-section hydration, with whole assistant-only sections omitted. The histories reused existing Qwen summary and attention caches: **zero new Qwen calls** during this five-test campaign. There were no full-context answer calls or new matched API controls.

Warm timings exclude ingestion, cached-source loading and cold reopen. History setup/ingestion took approximately 1,250, 1,200, 894, 649 and 659 seconds; cold reopen took approximately 25.4, 26.0, 24.2, 23.9 and 25.0 seconds. All five medians were below five seconds; one of five histories reached 95%.

These are generated questions over an existing real-transcript corpus, not an official benchmark score. The grader has clear false negatives, but the recorded score was not adjusted. Of 38 misses, 24 contained all recorded support quotations and 14 did not; quotation presence alone does not establish answer correctness. Examples include a correct April 15 date rejected in history 1 and a correct $250 offer rejected because the reference also included the unasked original $400 price in history 3. Other misses are actual evidence-selection gaps, including the newer car project in history 2 and several procedural/details questions in history 5.

Source-exact quote repairs were recorded before answering: a case-only repair in history 1 and an exact wording repair in history 4. Questions and reference answers were unchanged. Existing scope/evidence tests and the quote-repair checks passed.

Campaign SHA-256: `d328b657a383030a3309748f049164b1dc72abffdfaea01fe2b3ab379b8044d9`.  
Aggregate report SHA-256: `e867198863c704604013ee3eaddaff42b839a7ae5356b2fac5b21cdefb774969`.

## Coding replay

Source: `F:/Keytone/Documents/GitHub/memory_condense/data/build-session-8f7f7561.snapshot.jsonl`, 13,098,320 bytes, SHA-256 `4947dce90ec8f19ebd6720428b8ff1e160bd7dba42fbc6b6b5a214e4d9048a69`. The parsed transcript has 2,420 records. The selected episode contains every substantive user prompt between parsed indices 2052 and 2194: eight prompts, with none skipped inside that segment.

The isolated checkout started at `03105c423bf19b7a2b03f29a6b55e527f17ab2db`. Its original decay/database/memory-store baseline passed 110 checks. The replay seeded 214 earlier substantive, deduplicated user/assistant messages, totaling **21,749 tokens**. Tool dumps and compaction summaries were excluded. This is an eight-prompt feature episode, not a full-transcript or 1M-token coding test.

Only newly generated replies carried forward. No original future assistant solution was supplied. Each completed exchange was incrementally ingested, with a fresh native snapshot, parent vectors, close/reopen and exact raw retrieval before the next prompt. Historical memory grew from 214 to 228 messages before the eighth prompt. The answer model received retrieved user evidence, the latest live exchange and current-turn file-tool results. File-tool results were not ingested as historical memory.

Initial raw summaries used Sol under the user's raw-input authorization. Qwen merge calls and local attention received typed summaries only. Local attention used a six-layer Qwen prefix with FP16 weights and the selected head's FP32 softmax; BGE vectors used FP32. All eight route receipts recorded zero raw reads during routing and zero query-time Qwen passes. Exact selected raw evidence was hydrated afterward for the answer model. Backend processing as a whole is therefore not summary-only; the Qwen routing/attention boundary is.

| Prompt | Outcome |
| --- | --- |
| What is day 14 | Explained the historical checkpoint. |
| Differential decay per subsequent turn | Stated that useful/repeated evidence should reinforce memories while unrelated material decays. |
| Goldilocks zone | Preserved the retention/token-saving tradeoff in its reply. |
| Real time should not be decay; use turns | Implemented turn state, migration, decay, retrieval and tiering; 216 selected checks passed at that stage. |
| Review commit history | Identified compatibility and documentation gaps, but recommended returning reinforcement to a 300-second wall-clock gate. |
| Deliver the specified system | Carried that reinterpretation into the plan. The retrieved packet omitted the recent explicit correction. |
| Put it in docs | Actually edited the roadmap. |
| Go | Implemented the immediate compatibility milestone; its final focused run passed 132 checks. Independent review found the failures below. |

The complete replay took **185 model actions** and **2,542.7 seconds (42.4 minutes)** of API generation, excluding tools, memory setup and orchestration. The first refactor used 109 actions. This custom one-action-per-response coding harness is not the short-answer latency workload. Provider build usage counters were unreliable; saved request token proxies must not be presented as billed token usage.

## Independent findings

The final regression run covered decay, database, memory store, transcript store, condenser, MCP, evaluation, ranking and architecture: **240 passed**, with one existing Pydantic warning.

Eight independent review checks, hidden outside the actor's checkout, produced **four passes and four failures**. They were designed during this exploratory run and are not a pre-registered statistical coding benchmark.

Passing behavior: elapsed wall time does not change decay; normal ingested turns advance decay; state survives close/reopen; pins remain stable; recalling one item does not reinforce an unrelated item. Some checks cover more than one of these properties.

Three failures share a reinforcement cause. The final code gates boosts on 300 elapsed seconds and restamps every access, so rapid successive turns or duplicate evidence receive no boost. After 16 rapid turns, the repeatedly used item had energy **0.5535639528**, while the unused item had **0.5656854249**. The additional once-per-turn and repeated-evidence checks also failed. This is inconsistent with the earlier stated requirement to retain memories that subsequent turns keep making useful.

The fourth failure is an MCP display defect: `memory_stats` supplies the current tier but omits the current turn when formatting the item. It prints **WARM e=0.80** when the current decayed energy is **0.40**.

A continuity gap is directly visible in prompt 6: its 173-token retrieved packet contains older generic requests to finish the architecture, but omits “Real time shouldn't be the decay but rather per turn.” The latest generated history-review reply remains in live context. The model's reinterpretation then carries into the plan and code. This points to active user requirements needing stronger protection during routing; the QA result alone does not establish that the coding workflow preserves them.

The separate historical later-implementation oracle passed **85/118** checks and failed 33. Several failures depend on different API names (`now_turn`), a particular persistent turn ordinal or default policy. Treat this as a compatibility diagnostic, not another behavioral accuracy rate. The unmodified starting checkout could not collect that later suite; its collection failure was not counted as 118 failed tests.

The generated candidate was preserved without fixing its audit failures and was not merged into the live application. No full-context control was run.

## Harness and provenance limits

Nine initial harness boundary checks passed for chronology, path confinement, raw-input rejection at the Qwen boundary and role-specific cache identities. Initial Sol summarization required five recorded support-quote corrections across 214 fragments. Only support quotations changed; summary text and original responses were retained. Two bounded, separately journaled Qwen retries used the existing shorter prompts when a merge exceeded its token limit.

The build harness was corrected during execution: plain final prose was accepted without inventing a tool action; current-turn tool history stopped dropping after 12 actions; action limits expanded from 36 to 72 to 128; file-target search was fixed; and tests moved away from an inaccessible shared temp folder. All earlier requests/results remain. Windows option parsing initially stripped backslashes from the new temp paths, producing 692 runtime files inside the isolated checkout. They remain as artifacts and are explicitly excluded from candidate code diffs. Later calls use forward slashes and create the temp parent. One intervening run failed because that parent did not yet exist. These harness issues limit coding latency/model-efficiency conclusions.

Raw transcript timestamps were absent from the parsed source, so the replay used the explicit synthetic timestamp `2026-08-16T00:00:00+00:00`, recorded in its plan. The optional notes source was actually named `C:/Users/Keytone/Downloads/Github repo for notes`; it was read only. The project's decay episode offered a more concrete implementation test than that folder's graph-pipeline design conversation.

Reproduction helpers: `tools/native_spine_five100.py`, `tools/report_native_spine_five100.py`, the separately bound build answer/memory/tool adapters, `tools/report_native_spine_build_replay.py`, and `tools/validate_native_spine_build_replay.py`. Frozen original implementations and sealed artifacts were preserved. All generation and memory workers have exited; do not resume the completed campaigns.
