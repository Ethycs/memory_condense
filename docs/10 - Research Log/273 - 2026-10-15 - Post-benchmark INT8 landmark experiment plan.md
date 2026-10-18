# Post-benchmark INT8 landmark experiment plan

**Date:** 2026-10-15

**Status:** Complete; primary holdout rejected the INT8 landmark route

## Execution result (2026-10-16)

The post-benchmark run completed after r27 reached its terminal 500-question
report. It compiled and sealed eight INT8 K/V landmarks for all 5,483 holdout
summaries, reused the deployed BGE address space, and evaluated 100 questions
against 191 exact supports. Qwen received stored summaries and questions only;
raw transcript spans were opened only by the exact-support scorer after
selection.

The development result did not generalize. On the 783-summary, twelve-question
development archive, reranking BGE's top 64 improved the eight-section packet
from 5 to 9 support quotes and from 1 to 4 complete facts. On the predeclared
holdout, the same readout severely damaged every useful cutoff:

| Packet limit | BGE supports | Landmark supports | BGE complete questions | Landmark complete questions |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 132 | 6 | 58 | 1 |
| 8 | 153 | 12 | 69 | 3 |
| 16 | 171 | 29 | 85 | 9 |
| 32 | 178 | 73 | 90 | 27 |
| 64 | 183 | 183 | 94 | 94 |

The equality at 64 is mechanical: both arms contain the same BGE top-64 set.
The landmark order added no novel support at any smaller packet size. The
standalone corpus-wide landmark arm was worse, finding only 2/191 supports and
completing 1/100 questions at limit 64.

A saved-ranking diagnosis ruled out a simple polarity mistake. Reversing the
landmark order reached only 35 supports and 13 complete questions at limit 8.
A post-hoc 5% landmark/95% BGE rank blend reached 151 supports and 69 complete
questions, versus BGE's 153 and 69; it did not improve any tested cutoff. This
diagnostic is not a new promotion attempt and must not be tuned further on the
holdout.

### Measured speed and storage

| Measurement | Holdout result |
| --- | ---: |
| Landmark compilation | 5,483 summaries in 66.382 s |
| Compilation throughput | 82.6 summaries/s |
| Durable landmark payload | 16,432 bytes/summary; 85.92 MiB total |
| Memory-map gather | 0.185 ms mean |
| Warm single-query Qwen prefix | 61.483 ms mean; 65.209 ms p95 |
| GPU top-64 transfer + dequantization + score | 1.573 ms mean; 2.213 ms p95 |
| Interactive landmark total | about 63.06 ms mean; 67.42 ms additive p95 |
| Full-Qwen reference | 78 ms |
| Interactive speedup | 1.24x; about 14.95 ms saved |
| Batched 100-query throughput | 9.918 ms query encoding + 1.573 ms score per question |

The interactive route misses both predeclared speed gates: it is not at least
3x faster than the 78 ms reference, and its additive p95 exceeds 50 ms. The
batched number is useful for offline throughput but does not represent one
interactive chat request.

The holdout's one-time publication rate is acceptable, but cold startup on
this machine was anomalous. A fully cold Qwen diagnostic took 762.164 s, while
subsequent warm-cache preparations took 15--17 s. The uninstrumented
FastEmbed constructor window for the pinned 2.27 GB FP32 BGE-M3 ONNX export
then consumed roughly 23--25 minutes; the measured 100-query embedding work
itself took 45.426 s. Process telemetry showed nearly idle CPU with resident
memory rising slowly. A post-run file-attribute audit confirmed that the
2,266,886,160-byte ONNX data file is NTFS-compressed, stored as 1,972,076,544
bytes (only a 1.1:1 ratio), and is not sparse. All five Qwen safetensor shards
and sampled Transformers, PyTorch and SciPy package files are also marked
compressed. The broad compression policy is the leading explanation for both
pathological cold starts; filesystem filtering or storage contention may add
to it. Treat these cold-load intervals as an environment fault, test model and
environment files on an uncompressed local path, and do not use them as the
expected model-load or query latency.

### Decision

Stop before reader evaluation. The primary arm loses 141 exact supports and
66 complete questions at the intended eight-section packet, recovers no novel
support, fails the compression-support condition, and misses both speed gates.
Keep the vectorized INT8 scorer and resumable assay as experimental tooling,
but do not add landmark state or routing to the deployed pipeline. Preserve
BGE as the production candidate generator and ordering signal.

Sealed results:

- `eval_results/qwen-int8-landmark-post-benchmark-development-20261016-r1`
- `eval_results/qwen-int8-landmark-post-benchmark-holdout-20261016-r1`

Implementation and focused validation:

- `tools/run_qwen_int8_landmark_post_benchmark.py`
- `src/memory_condense/associations/qwen_int8_landmarks.py`
- `tests/test_qwen_int8_landmarks.py`: 7 passed

## Objective

Determine whether reusable eight-landmark INT8 Qwen K/V codes improve the
current memory system in either of two commercially useful ways:

1. recover exact support that the current production packet misses; or
2. preserve answer quality with a smaller packet by ordering useful evidence
   earlier.

The experiment also measures whether the landmark route is materially faster
than the existing full Qwen reranker after storage reads and host-to-device
transfer are charged. It does not assume that a faster Qwen route is faster
than production BGE, which already performs zero query-time Qwen passes.

At the time this plan was written, r27 remained healthy with 403 sealed answers
and both local models resident. The landmark experiment must not stop, restart,
or modify that campaign.

## What has already been tried

The following table distinguishes deployed components, measured experiments,
negative controls, calculations and unimplemented proposals.

| Mechanism | State | Quality finding | Speed and storage finding | Next action |
| --- | --- | --- | --- | --- |
| Protected production BGE route with exact hydration | Deployed baseline | Development packet contains 35/92 support quotes and 11/42 complete facts. The separate 100-question holdout baseline contains 187/191 supports and completely supports 97/100 questions. | Zero query-time Qwen passes. Full recall was about 1.0 s in the matched experimental campaign. | Preserve as the protected control. |
| Scalar Qwen attention cache | Deployed during compilation | Useful for query-independent hierarchy construction; not an arbitrary query-conditioned K/V memory. | About 78 ms per measured attention call and 7.63 s across roughly 98 calls in a mean benchmark history. Retained state is compact scalar evidence. | Keep deployed. Do not replace it in this experiment. |
| Full query-time Qwen coverage/linking | Experimental reference | Supplies the teacher QK/OV signal but has no demonstrated advantage over the protected production packet. | About 78 ms for one bounded call of up to eight candidates. Candidate text is re-encoded. | Use only as latency and ranking reference. |
| Layer-4 pooled Qwen A@V FP16 feature | Measured on development and 100-question holdout | Development top eight reached 6/92 quotes and 2/42 complete facts versus BGE's 5/92 and 1/42, but added no novel support over the protected packet. Holdout top eight reached 5/191 supports and 2/100 complete questions, again with zero novel support. | 8,192 bytes per summary. Prior GPU query encoding was 9.18-11.45 ms per question. | Keep as a frozen attention-code comparator. |
| Precomputed CAV readout | Experimental, not promoted | No new support; audited answer macro was 69.65% versus the protected route's 75.42%. | About 0.387 ms selector time after publication; 167.93 s cold backfill in the measured archive. | Retain as a speed control, not a candidate release route. |
| Sparse spectral/PPR reinjection | Experimental, not promoted | Changed rankings but recovered no new support. | 0.919 ms selector and 0.924 s full recall in the development assay. | Retain as a diversity control only. |
| Six-layer, full-token FP16 K/V | Calculated, not run as a persistent index | No accuracy result. | About 3 MiB per 128-token summary and 293 GiB per 100,000 summaries. | Reject as a persistent design. |
| Layer-5, full-token FP16 K/V | Calculated, not persisted | Needed as the transient no-pooling accuracy control. | About 512 KiB per 128-token summary and 48.8 GiB per 100,000 summaries. | Run only on bounded candidate sets, then discard. |
| Eight FP32 K/V landmark centroids | Measured diagnostic | Produced the same support counts as their INT8 versions in the 12-question shortlist assay. | Approximately 32 KiB per summary if retained as FP16; FP32 remained transient. | Use to isolate quantization loss. |
| Eight INT8 landmarks with raw Q-to-K readout | Measured negative control | Reduced BGE top-four support from 2 quotes to 1; recovered no complete fact. | 16,432 bytes per summary. | Reject this readout; preserve it as a negative control. |
| Eight INT8 landmarks with symmetric K/K plus V/V readout | Measured candidate | Increased BGE-shortlist top-four support from 2 to 4 quotes and from 0 to 1 complete fact. The extra support was already elsewhere in the protected packet, so measured overall production gain remained zero. | Saved candidate codes were reused with zero candidate re-encoding. Scoring took 38 ms for 12 CPU questions. Estimated warm top-eight GPU route is about 15 ms versus 78 ms for full Qwen, or roughly 5x faster. | Primary candidate for the post-benchmark run. |
| BGE-to-Qwen-Q adapter | Proposed, not implemented | No accuracy evidence. It could remove the remaining Qwen query prefix, but approximation risk is higher. | Could remove roughly 1.34 GiB of resident Qwen allocation and its cold startup if accurate. | Defer until native landmark retrieval demonstrates value. |
| Sixteen INT8 landmarks | Proposed conditional arm | No result. | Approximately doubles the eight-landmark state and transfer. | Run only if full-token K/V materially beats eight landmarks. |

The bounded INT8 result itself was:

| INT8 parity measurement | Result |
| --- | ---: |
| K relative RMSE | 2.31% |
| V relative RMSE | 1.13% |
| Mean routing-score error | 0.0119 |
| Mean eight-item rank correlation versus FP32 | 0.9821 |
| Top-one agreement versus FP32 | 10/12 |
| Top-eight set agreement versus FP32 | 12/12 |

## Execution trigger and resource preflight

Start only after all of the following are true:

1. r27 reports a terminal successful state with 500 sealed answers and its
   required judgments and final report;
2. its process has exited normally and its artifact hashes verify;
3. the pinned Llama server and benchmark-owned Qwen process have released the
   GPU;
4. `nvidia-smi` shows enough free memory to load the existing compressed
   six-layer Qwen runtime without eviction; and
5. commit `0bc2d89` and this plan are present in the execution identity.

Record the final r27 status and score in the new experiment receipt. Do not
copy, mutate or rebuild any r27 histories for the landmark experiment.

## Required harness work before the GPU run

The current tools contain the reusable code and bounded assays, but the
full-corpus runner must be extended before execution:

1. add the symmetric K/K plus V/V readout to
   `tools/assay_qwen_int8_landmarks.py`;
2. make candidate compilation resumable by sealed batch, so an interrupted
   run restarts after the last verified summary rather than rebuilding the
   index;
3. support both the 783-summary development archive and the existing
   5,483-summary, 100-question holdout adapter;
4. memory-map the INT8 codes and fetch only the current shortlist into one
   contiguous pinned buffer;
5. vectorize scoring across candidates rather than invoking one NumPy scorer
   per summary;
6. bind every artifact to section ID, source occurrence, Qwen checkpoint,
   layer, landmark count, quantization format and summary hash; and
7. persist FP32 centroids only for the matched diagnostic sample. The durable
   candidate index remains INT8.

The implementation must continue to route only summaries through Qwen. Exact
raw sections are hydrated after selection through the existing source boundary.

## Frozen populations

Use two populations in order.

### Development diagnostic

- Archive: the existing 783-summary engineering session.
- Questions: the existing twelve source-bound engineering questions.
- Purpose: verify execution, attribution, timing and failure diagnosis.
- Labels may be inspected only after every predeclared arm has produced a
  sealed ranking.

This population already informed the symmetric readout choice. It cannot be
used as confirmation evidence or to tune thresholds further.

### Primary holdout

- Archive: `native-spine-ten100-20260922-r1/history-01`.
- Summaries: 5,483.
- Questions: 100.
- Exact supports: 191.
- Current protected baseline: 187/191 supports and 97/100 completely supported
  questions.
- Existing Qwen-feature validation: provider-free and summary-only.

Run the holdout once after the development artifacts and policy are sealed. If
the primary arm fails, record the failure. Any revised readout belongs in a new
experiment against a new holdout or after returning to development data.

## Predeclared arms

All arms receive the same question and historical source scope. BGE embeddings,
summary text, exact support references and existing source hydration artifacts
must be reused.

| Arm | Candidate generation | Final ordering | Purpose |
| --- | --- | --- | --- |
| A | Current production path | Current protected ordering | Primary baseline. |
| B | BGE top 8 | Original BGE order | Priority and packet-compression control. |
| C | BGE top 8 | Eight INT8 symmetric K/V | Reproduce the bounded priority result. Candidate recall cannot change. |
| D | BGE top 64 | Original BGE order, first 8 | Wider-candidate BGE control. |
| E | BGE top 64 | Eight INT8 symmetric K/V, first 8 | **Primary landmark arm.** It can promote support that BGE placed between positions 9 and 64. |
| F | All summaries | Eight INT8 symmetric K/V, first 8 | Standalone corpus-wide landmark retrieval. Secondary because it is more expensive and discards the strong BGE prior. |
| G | BGE top 64 | Frozen pooled A@V comparator, first 8 | Determine whether K/V landmarks add anything over the prior attention code. |
| H | BGE top 64 | Full-token FP32 layer-5 K/V on a sealed diagnostic sample | Isolate eight-landmark pooling loss. Transient only. |
| I | Same candidates as E | Eight FP32 landmark centroids | Isolate INT8 quantization loss. Diagnostic only. |
| J | Same candidates as E | Raw Q-to-K INT8 readout | Frozen negative control; stop before reader evaluation. |

Do not add sixteen landmarks during the primary run. Admit that follow-up only
if arm H beats arm I by at least two exact supports or one complete question on
the 100-question holdout while INT8 remains close to FP32. This prevents paying
twice the state for a problem not shown to exist.

## Accuracy measurements

### Retrieval and grounding

At limits 4, 8, 16, 32 and 64, report raw counts for:

- exact support quotes recovered;
- questions or facts with every required support recovered;
- exact support added over the protected baseline;
- newly complete questions or facts;
- protected-baseline support lost;
- selected summaries that fail exact section hydration; and
- support provenance errors, cross-source errors and cross-conversation errors.

The primary comparison is arm E versus arms A and D. Arm C is useful for packet
priority but cannot improve candidate recall because it preserves BGE's top
eight set.

### Quantization and landmark compression

For arms H, I and E, report:

- per-head K and V relative RMSE;
- Pearson and Spearman score correlation;
- top-one, top-four and top-eight agreement;
- exact support disagreements at each cutoff;
- full-token versus eight-landmark ranking loss; and
- FP32-landmark versus INT8 ranking loss.

INT8 passes its parity gate if it loses no complete question, loses at most one
exact support across 100 questions, and retains at least 0.98 mean rank
correlation against the identical FP32 landmarks. The full-token control, not
INT8, determines whether eight landmarks are sufficient.

### Packet compression

Build source-identical packets from arms A, B, C and E under fixed caps of four
and eight sections. Measure:

- packet input tokens;
- exact support retained;
- duplicate and irrelevant sections;
- source-hydration bytes; and
- current-turn material retained by the six-exchange light context.

The compression path succeeds if landmarks reduce median sent tokens by at
least 20% while retaining every exact support present in the matched production
packet and without reducing answer accuracy by more than one answer in 100.

### Reader accuracy

Do not spend reader calls until an arm either recovers novel exact support or
passes the packet-compression support gate. For admitted packets:

1. use the same pinned reader, prompt, temperature and output limit;
2. blind packet identity and ordering labels from the reader and judge;
3. grade exact/numeric answers with deterministic validators first;
4. apply BGE/fuzzy semantic validation only to deterministic failures;
5. send only the unresolved failure set to the same frontier judge; and
6. retain exact source quotes with every judgment.

Report paired correctness, changed-answer counts and a paired bootstrap
confidence interval. A landmark arm is answer-noninferior if it produces no
more than one additional error in the 100-question holdout. Any apparent gain
must also have either new source support or a defensible ordering explanation;
generation variance alone is not a retrieval win.

## Speed, bandwidth and resource measurements

Measure cold publication once and warm query execution after 50 warmups. Use at
least 500 timed queries for stable p50 and p95 values. Synchronize CUDA around
GPU timing and separate the following components:

1. BGE query embedding and top-64 lookup;
2. landmark record lookup;
3. database or memory-map page faults;
4. packing selected records into a contiguous pinned buffer;
5. host-to-device transfer;
6. query Qwen prefix encoding;
7. INT8 dequantization;
8. symmetric K/V scoring;
9. ranking and exact hydration; and
10. complete recall latency.

Benchmark hot RAM, cold file cache and deliberately fragmented record layouts.
Compare eight and 64 candidates:

| Transfer case | Payload |
| --- | ---: |
| Eight INT8 landmarks | 131,456 bytes |
| 64 INT8 landmarks | 1,051,648 bytes |
| Eight full-token layer-5 FP16 candidates | 4,194,304 bytes |

Also record Qwen cold load, retained GPU allocation, peak GPU memory, host RAM,
artifact file size and database framing overhead. The expected durable payload
is 16,432 bytes per summary: about 12.27 MiB for 783 summaries, 85.92 MiB for
5,483 summaries and 1.53 GiB for 100,000 summaries.

The speed gates are:

- at least 3x faster than the measured 78 ms full-Qwen bounded reranker for a
  matched candidate population;
- no more than 50 ms p95 added to the production BGE recall path for the
  primary top-64-to-top-8 arm;
- no model load or GPU/CPU weight transfer per chat turn;
- publication throughput fast enough that landmark compilation does not make
  ingestion fall behind replies; and
- exact query results invariant between batched and unbatched scoring.

The earlier approximately 5x estimate is a hypothesis, not the result of the
post-benchmark run. The report must replace it with matched end-to-end numbers.

## Correctness and isolation tests

Add automated tests for:

- exact 16,432-byte ndarray payload and schema validation;
- short summaries, padded landmarks and maximum 128-token summaries;
- deterministic token buckets and stable section ordering;
- per-head INT8 saturation, zero vectors and scale round trips;
- sealed artifact resume after an interrupted batch;
- rejection of a changed summary, checkpoint, layer, source occurrence or
  landmark policy;
- no raw transcript field in Qwen inputs or retained landmark artifacts;
- conversation and source-scope isolation;
- exact pointer preservation from selected summary to hydrated raw section;
- CPU/GPU scorer parity within the declared numerical tolerance;
- identical results for scalar and vectorized scoring;
- one contiguous transfer for a selected shortlist;
- cold-storage corruption and truncated-record failure; and
- coexistence with the resident BGE/Qwen scheduler without exceeding the
  global CPU/GPU admission limits.

## Stop rules

Stop before reader evaluation if any of the following occurs:

- INT8 loses more than one exact support or any complete question versus FP32
  landmarks;
- eight landmarks materially underperform transient full-token K/V and the
  predeclared sixteen-landmark condition is not met;
- arm E adds no novel support and fails the packet-compression support gate;
- exact hydration, provenance or conversation isolation fails once;
- p95 overhead exceeds 50 ms after vectorization and contiguous transfer; or
- publication cannot be resumed without rebuilding already sealed summaries.

A stopped arm remains documented as a negative result. Do not repair it against
holdout labels and rerun under the same experiment identity.

## Promotion decision

Promote symmetric INT8 K/V landmarks only through one of these paths:

### Retrieval path

- arm E recovers at least one of the four currently missing holdout supports;
- it loses no protected support or complete question;
- the answer comparison is noninferior; and
- it passes the speed and resource gates.

### Compression path

- it reduces median packet tokens by at least 20%;
- it retains all exact support in the matched production packet;
- it loses no more than one correct answer in 100; and
- it passes the speed and resource gates.

If neither path passes, keep the module experimental. The result would still
show that INT8 preserves landmark rankings, but it would not justify adding
state and query work to the commercial production route.

## Planned artifacts

The run should publish a new, immutable output root containing:

- execution and checkpoint identity;
- resumable landmark batches and their hashes;
- section-ID and source-occurrence mapping;
- FP32/full-token diagnostic sample;
- per-arm rankings;
- exact-support and complete-question tables;
- reader packets and judgments only for admitted arms;
- cold and warm latency traces;
- transfer and memory measurements;
- failure classifications; and
- one final payoff matrix covering accuracy, packet tokens, latency, storage,
  memory and promotion state.

The existing inputs and bounded evidence remain:

- `eval_results/qwen-int8-landmark-shortlist-20261015-r1`;
- `eval_results/qwen-int8-landmark-readout-20261015-r1`;
- `eval_results/attention-feature-retrieval-20261009-r1`;
- `eval_results/attention-feature-holdout-20261009-r1`; and
- `docs/10 - Research Log/272 - 2026-10-15 - INT8 Qwen landmark assay.md`.
