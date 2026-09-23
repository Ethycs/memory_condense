# Malformed JSON recovery and expanded native admission

**Status:** Real source repairs and expanded serving verification complete; successor admission running. No native full100 accuracy or API latency result.

The target remains at least 95 correct answers across the complete 100 separate
1M-token memories, with all eight matched median/p95 TTFT/total latency ratios
at most 1.10 in the same fresh evaluation. Component checks and partial histories
cannot establish that result. The preceding alias clarification made no benchmark
progress; this continuation completed repairs and released their real admission.

## Completed source repairs

The R12 selection froze 55 rejected batches outside the previous 197 completed
repair lineages. Its preparation stopped before any provider call because
original batch 4541 contains malformed JSON. Original validation SHA:
`0acd1d3238a22cfb3923febe2b403d9e895671d17dec0a8393de3f76d01bdf79`.
The parser reported an expected comma at offset 295. The failed preparation and
original response remain recorded; neither was converted into a transport failure.

The other 54 batches completed the separate R13 → R14 → R15 section-repair
lineage. R13 accepted 196/204 prepared sections in 26 calls, R14 accepted 18/19
new sections in three calls, and R15 accepted the final three subdivisions in
one call. All 54 original batches now pass. Their 1,161 valid original summaries
are unchanged; 78 rejected original fragments became 217 exact subdivisions.
The final population covers all 1,239 original fragments with 1,378 sections.

- Final root: `eval_results/native-spine-direct-section-repairs-20260912-r15`
- Result SHA: `bc70bc9bea0717ac9ab6fa858013fc83b98134dbe533b9a6c2e260c594b238aa`
- Final preflight SHA: `1e216dcb0a45de93eb71397947638d3a41b7534ae6550cf6e418f66ca9187312`
- R13 and R14 are ancestors, not additional admitted populations.

`tools/repair_native_json_batches.py` handles malformed responses explicitly.
It authenticates the original invalid validation and completion, requires an
actual JSON decoding failure, and requests fresh summaries for the exact
original fragment population. It admits no strings from an unparseable response.
Later refinements retain accepted new summaries and subdivide only rejected
pieces. Exact raw coverage is checked directly without reparsing or rewriting
the malformed response. Provider reservations still prevent implicit retries
of uncertain calls.

The real batch 4541 recovered all 21 original fragments in three Terra calls,
without subdivision. Qwen received no raw text.

- Root: `eval_results/native-spine-malformed-json-repairs-20260912-r1`
- Preflight SHA: `47b5d9698ba246fb323fdb2b7a16d5808c4c9faa607d3bc2078a3bd06f9066c2`
- Public scope SHA: `61a744981ad897172cd0f473527ff4ba9b52192fef9a8dbb3cb9bb633ab95353`
- Result SHA: `abccc2e24e2253c969a834cecbf367d772e029b6fca983d52ec2474ef51bacdb`

This repair wave used 33 calls in total: 30 section-repair calls and three JSON
recovery calls. Seven were newly released in this continuation; the 26-call R13
run had already been started. Main ingestion continues independently.

## Admission and verification

`tools/assemble_native_spine_json_recovered.py` preserves the original rejected
validation separately from its admitted JSON repair. It gives JSON repair its
own status and producer identity; transport recoveries remain separate. Admission
replays the repair with provider access disabled, rejects duplicate or foreign
lineages, and validates complete body coverage against the original raw bank.
`JsonRecoveredSummaryBodies` authenticates this additional producer identity.
Existing stores and their producer implementations remain unchanged.

The first recovery test found that the old final collector still parsed the
malformed response. Replacing that collector with direct source-coverage
validation resolved the failure. Nine focused recovery/direct-repair tests pass.
The subsequent recovery/admission suite passes 15 tests, covering complete and
subdivided JSON repairs, preservation of original journals, zero-call replay,
partial-admission rejection, duplicate/foreign lineages, and database changes.
These are implementation checks, not answer-accuracy measurements.

The real successor snapshot is running:

- Root: `eval_results/native-spine-admitted-body-store-20260912-r6`
- Driver: `.tmp/assemble_json_recovered_native_snapshot_20260912_r6.py`
- Policy SHA: `776d22997c768417344d30e41b4ac3fac6eb0c2893a7a4d55c1a8b281cc18244`
- PID 42240, creation time `1789230452.555384`, session 45661.
- It includes all prior final repairs/recoveries, final R15, and JSON recovery R1.
- After assembly, it checks preservation of every R5 body summary and records
  which bodies touched by batch 4541 are admitted. No model calls are released.

The expanded parent producer has published 6,725 complete body hierarchies out
of its 7,121-body input snapshot. Its 128-job partial result SHA is
`6b4a1710ed2ac8ad6543cd2c65626dcf0828ba27bf6d7ebb9bd5368d034243e7`.
Parent compilation remains live on PID 48976. Vector R4 still waits for that
exact process to finish before loading BGE, preventing competing GPU workloads.

`.tmp/verify_recovered_parent_native_serving_20260912_r4.py` completed checks of the new
parent adapter against that real partial report, the R4 source-summary store,
and completed R3 vectors. It uses cached summary self-queries across all 100
namespaces to check baseline preservation, exact date-eligible hydration and
rejection of incomplete 1M evaluation admission. It reads no benchmark questions
or gold and runs no embedding, Qwen, answer or judge model. Session 78781 exited 0.

- Verification root: `eval_results/native-spine-recovered-parent-serving-verification-20260912-r4`
- Preflight SHA: `c40c21259727e2f9ced0337407ec3022b282922e7b06dc0078f1039dabb9b38a`
- Result SHA: `0baf1f18a31a6867b632325b81eda462e282d21fe3a4036a729b29d58c973545`
- All 100 namespaces passed, binding all 6,725 templates into 11,330 occurrences.
- All 3,450 selected raw spans across both arms hydrated exactly. The expanded
  arm retained every baseline section and added 68 sections.
- All histories remain partial, at 215,806–323,184 admitted body tokens, and
  all rejected full1M admission. Answer accuracy and API latency were not measured.
- Self-query selection changes as compiled templates become available. Compare
  baseline preservation within this run; the raw counts do not measure accuracy
  improvement over the earlier serving check.

During these checks, the live parent compiler published its next partial result:
6,848 body hierarchies after 256 total new local jobs in this continuation.
Result SHA: `9d42a901b890a7961f9cb6103e0dc5e3e46c4c6734c1d4454560db7c1ec377ec`.
The serving check deliberately retained the earlier 6,725-tree report.

R6 admission snapshot SHA is
`557458f0544cd8f0620865630b5d646806c9b157419e2faa93dbae235e9a845f`.
It freezes 5,717 accepted originals, 251 ordinary repairs, six transport
recoveries and one JSON repair: 5,975 admitted batches. Another 21 invalid and
7,816 pending batches remain excluded. SQLite body assembly is still running;
this snapshot alone does not establish a completed body store.

For the next expansion, use the JSON-aware reader to authenticate R6. The old
recovered-store adapter intentionally rejects the new producer format. The old
reused-exchange compiler also does not accept the completed recovered R3 exchange
producer as a reuse root; an explicit successor is needed to reuse its 317
accepted merge keys for new input populations. Do not relabel either producer
or alter the already bound implementations to bypass those checks.

At the most recent source-validation count, main ingestion had 5,665 accepted
and 270 invalid original batches, 5,935 total. Original invalid statuses remain
unchanged after repair. PID 56400 was confirmed live, as were the parent,
vector-wait and new assembly processes. A complete corpus and the actual fresh
joint full100 evaluation remain outstanding.
