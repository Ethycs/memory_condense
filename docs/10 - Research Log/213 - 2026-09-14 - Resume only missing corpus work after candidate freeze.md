# Resume only missing corpus work after candidate freeze

The user accepted the observed sub-five-second response time and asked to focus
on accuracy. The candidate in Research Log 212 remains frozen. Serving-latency
tuning has stopped; the next accuracy milestone is the fresh 100-question run.
Its remaining prerequisite is completing shared corpus caches without replaying
finished body compilation. The original full-corpus controller remains stopped.

## Exchange continuation

The stopped September 14 checkpoint already contains exchanges for 31,115 of
31,166 unique bodies. Its 51 missing bodies can be compiled independently using
the existing source summaries and exact merge cache. The previous continuation
instead replayed ancestor body compilers and reconstructed the full population
at each bounded invocation.

`tools/native_spine_exchange_journal_cache.py` now authenticates each ancestor's
response journal once per process. It retains the original acceptance boundary,
source and model identities, exact recorded ancestry cache hashes, explicit
length-recovery projections and rejection of unacknowledged executions. It
does not call a body compiler, load a model or copy prepared body files.

Real cache authentication recovered 1,468 accepted merge keys and 2,017 attempted
key/variant records from four roots in 33.793 seconds. This is journal validation
time, not a complete ingestion speed measurement. The source files were unchanged.

`tools/compile_pending_native_spine_exchanges.py` prepares only the 51 missing
body bindings. It reads their atomic summaries, reuses accepted merges, and uses
the same local Qwen backend and bounded recovery rules for missing merges. It
checks exact raw-span coverage before publishing each body. Each invocation
permits at most 128 new local jobs. The original 31,115 completed bodies remain
referenced by their checkpoint, with no recompilation or copying. A later full
admission still must validate the combined population; completing this small
continuation alone does not claim full admission.

Current artifacts under `eval_results/native-spine-frozen-corpus-20260914-r1/`:

- `exchange-cache.json`, SHA
  `21ee4646c0d7fa0b267f307bab5b7d3924178801275262d5a6b392da764ee731`.
- `pending-exchanges/preflight.json`, SHA
  `5443366bd7f1292284148281b7fdd3b2d8bf33fce1c893b66fe592d2cec840bc`.

The first real local-Qwen invocation completed 39 of the 51 pending bodies in
127 new local jobs across 37 batches and exited zero. Its partial result is
`pending-exchanges/partial-a665336f66b904ae8414ac963176310bcb03bb6917e3d7a0d2e9c2a61a4e41fc.json`,
SHA `07357c19c8b082c5c2593dba8a54885f705d9b9950b71c4dffa785cc41d2672b`.
`worker-first-exit.json` records the successful bounded invocation. Completing
an invocation is distinct from completing all pending bodies.

A second `--budget 128` invocation completed the remaining 12 bodies and exited
zero. It authenticated the first invocation's journal and reused its model outputs. Within
the small 51-body continuation, cached graphs are reconstructed to verify their
identical publication; the original 31,115-body checkpoint is not reconstructed.
The zero completed-body-recompilation counters refer to that original checkpoint.
The final `result.json` covers all 51 pending bodies, SHA
`1b331e08c83485aff54911d015f2b09f58b20a72d3fce5f7d5b0194e53839fc7`.
The second invocation used 50 new local jobs across 19 batches; both invocations
totaled 177 jobs across 56 batches. The original 31,115 bodies were retained.

The completed first worker is recorded in `pending-exchanges/worker-started.json`: PID
67764, creation time 1789452319.0579507, executor session 96732. The receipt SHA is
`1cafaced886d015591959e157e601537adbf6fb391f761f1166ee9b982448002`.
Match PID and creation time when inspecting the worker; do not act on PID alone.
The completed second worker is `pending-exchanges/worker-continued-02.json`: PID
29276, creation time 1789453393.6590526, executor session 82206. Its receipt SHA is
`0e83b53457cd73cc9435cb65f399447818bcc42922c78dc9326df80e3e2c9d1f`.

## Remaining hierarchy preparation

`tools/prepare_native_spine_frozen_corpus.py` combines the original completed
exchange bindings with the pending continuation's completed bindings. It retains
the 13,468 earlier parent trees and 300 additional selected-history trees by
reference. The resulting scope is scheduled only after all 51 pending bodies
finish. It does not duplicate prepared sources or model outputs.

Attention preparation excludes every body with an existing parent tree. It loads
the remaining exact exchange and atomic inputs, checks their complete raw-span
coverage and derives the same summary-only user-spine attention windows. Groups
of 256 bodies receive immutable preparation checkpoints, so resumption can reuse
finished groups. The existing attention executor and cache method remain unchanged.
The subsequent full scope contains 31,166 bodies, retaining 13,768 existing parent
trees. Attention preparation for the remaining 17,398 bodies has completed;
the local attention executor is processing their 17,642 unique summary windows.
Continue with Research Log 214 for that stage and the frozen broad runner.

The remaining stages are attention for missing hierarchies, bounded parent
compilation, remaining summary vectors, full population admission, and a broad
runner bound to the frozen candidate and matched API controls. No native full100
answer accuracy has been measured yet.

## Verification

Twelve focused checks passed: five for authenticated journal reuse without body
compiler/model calls; four for pending-only compilation, exact raw hydration,
bounded execution and interruption handling; and three for skipping existing
parents, resumable attention preparation and rejection of changed artifacts.
The real journal cache also reproduced the recorded ancestor hashes without new
model calls. No serving implementation or previous result was changed.
