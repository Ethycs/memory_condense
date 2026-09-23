# Native namespace retrieval and reusable summary vectors

**Date**: 2026-09-12  
**Status**: Namespace and retrieval components verified; local vector execution queued  
**Depends on**: [Research Log 188](188%20-%202026-09-12%20-%20Native%20hierarchy%20compilation%20and%20atomic%20evidence%20recovery.md)

## Result and target status

The native serving path now retrieves original atomic summaries directly and
adds context from the attention-defined chunk containing a strong match. All
direct routes remain first and unchanged. Parent scores cannot remove a fine
match, addressing the branch-selection failure diagnosed in the earlier
hierarchy evaluation.

All 100 real source namespaces have been materialized independently from the
current partial admission snapshot. All correctly reject full-evaluation
admission. Four hundred summary-selected raw spans hydrate exactly. Sixteen new
focused checks pass, bringing the cumulative native count to 127.

The reusable vector stage is prepared for 17,146 unique stored summaries. Its
live handoff waits for the existing Qwen process to terminate before loading
BGE. No new native answer-accuracy or API-latency score exists. The joint target
remains unachieved; component checks and partial-corpus queries are not a pass.

The immediately preceding user-question turn clarified the gateway alias but
did not advance the goal. This continuation revalidated the exact live worker
processes, changed the serving implementation, completed real namespace checks,
and released a concrete dependent embedding job.

## Serving behavior

`tools/native_spine_namespace.py` authenticates the source bank, summary-body
snapshot, exchange inputs, hierarchy preflight, and selected immutable partial
report. It materializes one requested history at a time, preserving actual
occurrence IDs and dates. Missing summaries and trees are recorded explicitly;
partial use requires `allow_partial=True`. Even then, `require_complete()`
rejects an incomplete or sub-million-token namespace.

Every admitted atomic address remains available even when its parent tree is
still compiling. Cold namespace construction counts actual source-body tokens,
binds available trees to their actual occurrences, and retains exact raw turns
for subsequent hydration. It performs no model calls and reads no evaluation
questions or answers.

`src/memory_condense/search/native_spine_routing.py` uses the existing semantic
summary index for direct retrieval. Defaults are 32 direct candidates with two
lexical reservations. Source timestamps are filtered against the bound question
day. Up to eight leading matches identify attention-defined leaves; up to eight
additional original atoms are drawn from those leaves in round-robin order.
The additions follow all direct candidates, so the existing sequential hydrator
retains every piece of accepted baseline evidence under the same budgets.

`src/memory_condense/application/native_spine_retrieval.py` connects one live
query embedding, routing, and exact hydration. The default limits remain 3,072
rendered context tokens and 128 raw span inspections. The baseline and augmented
arms can each perform a fresh query embedding. Qwen attention supplies the
ingest-time chunk boundaries; this candidate serving policy performs zero Qwen
passes at query time. It does not reuse the failed repeated branch pruning.

This is an additive context candidate. Keeping the direct evidence may leave
little room for additions, and no accuracy improvement is claimed before a
complete matched evaluation.

## Real namespace verification

Root: `eval_results/native-spine-namespace-serving-20260912-r1`.

Verification SHA:
`b5893e6ae5b476103f746aadc48b56b69248fe6cbc34b1e1bccda3b17a528bde`.

Script: `.tmp/verify_native_namespaces_20260912_r1.py`.

The check binds the immutable 170-tree report
`2f8c31471dc95860d42268a508140351cb275fca26ee8f8953b76c7020e96085` and
summary store `a60d1fc4fc4fb618f7e2b9d4c9b50916e9a7aaa12e11dbf55796dfa1970a8456`.
Later hierarchy output does not alter this report.

Across 100 separate histories, the partial store materializes 2,818 actual source
occurrences, 28,800 atomic sections, and 318 available hierarchy occurrences.
Four selected atoms per history hydrate exactly, for 400 checked spans. Source
ownership is checked against each namespace's admitted occurrences. Every
namespace rejects both ordinary complete admission and the subsequent 1M gate.

The first history illustrates the remaining gap: 26 of its 514 occurrences are
currently admitted, yielding 284 atoms and 59,329 body tokens. Only three of
those occurrences have completed trees in the bound report. Its complete source
bank exists, but the current compiled memory is not a million-token memory.

These queries use stored summaries with BM25 to exercise namespace admission and
hydration. They are not benchmark questions, semantic accuracy tests, or serving
latency measurements. Test-double vectors cover semantic routing behavior in
the focused tests separately.

## Reusable embedding stage

`tools/compile_native_spine_vectors.py` embeds only literal admitted atomic
summary strings using the existing pinned BGE implementation. It deduplicates
identical strings independently of occurrence dates, normalizes vectors to
FP32, and checkpoints at most 128 summaries at a time. Future snapshots can
reuse completed vectors for unchanged strings with the same encoder identity.
The cache rejects changed matrices, missing summaries, wrong encoder identity,
and incomplete compilation; completed replay requires no model load.

Root: `eval_results/native-spine-summary-vectors-20260912-r1`.

Preflight SHA:
`7ada3a5aed4fe4e5c9e0da4a2166a8aee203b74691f759b78530075b611cc248`.

The prepared population is **17,146 unique summaries**, in **134 checkpoints**,
from all 17,190 atoms in the current 1,669-body snapshot. Preparation loaded no
model. This snapshot is explicitly not the complete source compilation.

Driver: `.tmp/run_native_vectors_after_qwen_20260912_r1.py`.

Handoff policy SHA:
`56e46db1ed242b51e586cea47596c7ed9c0888c7f3d762fa7aaaf259a65151da`.

The handoff is session **20807**, PID **37308**, creation time
**1789211855.2634516**. That exact process was verified live. Its control files
are under the vector root's `handoff/` directory. It requires the exact owned
Qwen process to be gone and a matching terminal Qwen receipt before releasing
BGE on CUDA with batch size eight. It waits at most two hours and never restarts
Qwen. At the recorded check it had no release, completion, or failure artifact.
Do not start another GPU job while either Qwen or the released vector stage is
live. Raw-summary Terra calls can continue independently.

## Validation and active work

Eleven focused routing and namespace tests passed in 2.07 seconds (tool chunk
`98fad7`). Five vector-cache tests passed in 2.00 seconds (`7c039e`). They cover
exact baseline preservation, shared budgets, future-source exclusion, missing
trees, foreign topology, stale evidence, fresh query encoding, source admission,
vector reuse, interrupted local computation, corruption, and encoder changes.

Frozen implementation hashes remain unchanged: 53 files in the earlier full100
preflight, six in the full raw compiler, 26 in the exchange compiler, 31 in the
attention compiler, 32 in the hierarchy compiler, and 20 in the vector compiler.
The real component verifications' three- and four-file bindings also match.
Check output: `54a219`. Existing staged work was preserved.

The main raw worker remains session **90673**, PID **65736**, creation time
**1789201959.128199**. At the latest exact-process check it had 1,975 completed
batches: 1,892 accepted and 83 invalid summaries, with 44,255 accepted atoms.
There was no terminal handoff artifact. The seven earlier repaired batches are
separate overlays; these counts do not reclassify original rejected receipts.
The full preparation still contains 13,812 requests. Do not restart a live run.

Next work is to complete and replay vector compilation, connect those vectors
to resident native serving on real sources, continue source-summary repair and
admission, and finish the complete corpus. Run the full100 accuracy and matched
API timing comparison only on complete, admitted 1M histories. Both target
gates must pass on the same fresh answer run; the confirmation set stays unused.
