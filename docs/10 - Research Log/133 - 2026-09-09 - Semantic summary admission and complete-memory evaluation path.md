# Semantic summary admission and complete-memory evaluation path

> **Continuation:** full-parent compilation and the MiniLM probe have finished.
> The additive user-spine/calendar comparison and current execution status are
> recorded in [Research Log 134](134%20-%202026-09-09%20-%20Additive%20user%20spine%20routing%20on%20complete%20memory.md).
> Process references below describe earlier stages and are superseded by that log.

**Status:** Active development; the joint 95% accuracy and direct-API latency target remains open.
**Date:** 2026-09-09
**Predecessor:** [Research Log 132](132%20-%202026-09-09%20-%20Joint%201M%20accuracy%20and%20latency%20target.md)

## What changed

The query path can now search BGE-M3 embeddings of every leaf summary, reserve
lexical matches, and rerank the bounded shortlist with one local Qwen attention
pass. A summary need not share exact query words to enter the shortlist. Parents
cannot hide matching leaves. Exact raw sections are hydrated after selection.
Neither embedding nor Qwen routing consumes raw transcript text.

The complete-memory pipeline now has tools for batched summary merges, persisted
summary vectors, exact raw-fragment reassembly, and a joint streaming answer and
latency evaluation. All 850 raw-summary requests for the first complete 1M-token
namespace have finished, and all 5,556 atoms are admitted. Full hierarchy
compilation is running in session **10417**; the answer evaluation is pending.

## Measured semantic-routing mechanism

Artifact: `eval_results/semantic-summary-routing-mechanism-20260909-r2/runtime.json`

SHA-256: `9e8269334428aebda291248eda96b0dd3e8cb66e80fd4e8282ee078db9a05cfd`

This reuses the earlier 39-turn, three-conversation hierarchy solely as a
mechanism fixture: eight questions, five arms, three measured warm repetitions.
Every semantic query includes a fresh BGE encoding. Both models remain resident.
No answer or judge calls were made.

| Route | Median | p95 |
| --- | ---: | ---: |
| Summary BM25 | 7.924 ms | 9.952 ms |
| Summary dense | 37.098 ms | 68.326 ms |
| Summary hybrid | 35.856 ms | 48.760 ms |
| Dense + one Qwen pass | 231.356 ms | 253.704 ms |
| Hybrid + one Qwen pass | 230.016 ms | 241.805 ms |

All 40 arm/question combinations hydrated without diagnostics. On the previously
missed chronology question, BM25 selected one source; dense and hybrid selected
two; the Qwen reranked variants selected all three. Inspection of those selected
summaries found the relevant Muir Woods hike, Big Sur/Monterey road trip, and
Yosemite camping trip. This is evidence of improved routing on one examined
question, not an answer-accuracy score or full-memory result.

The fixture has 21 leaf vectors (86,016 bytes). Peak allocated CUDA memory was
5.655 GiB with both models resident. BGE compilation/load took 15.674 seconds and
Qwen load 6.411 seconds with warm filesystem caches. The older cold Qwen load of
149.238 seconds remains relevant; startup is outside these warm query timings.

Local Qwen still uses FP16 weights/forward, FP32 softmax/readout, five complete
transformer blocks and the sixth block's QK/OV readout. It is not an FP32-only or
attention-head-only forward. No precision policy changed.

## Pinned embedding load

The first mechanism attempt failed when the installed SentenceTransformers
AutoProcessor path attempted to resolve model metadata at `main`. The loader now
resolves the exact cached BGE snapshot with `local_files_only=True`, passes that
local directory to SentenceTransformer, and verifies that same snapshot.

The successful r2 run validates this change. The failed r1 directory is retained.
Its failed metadata HEAD attempts contained no transcript payload. Model and
revision remain `BAAI/bge-m3` at
`5617a9f61b028005a4858fdac845db406aefb181`.

## Routing-summary admission supersedes quote blocking

**This supersedes Log 132's instruction to repair every support-quote failure
before hierarchy compilation.** A generated exact quote is not an entailment
proof. The new admission policy requires a nonempty bounded summary, exact atom
labels, and the complete authenticated raw input span. Role, timestamp, source,
coordinates and hashes come from that input span. Generated support-quote
failures remain explicit diagnostics; summaries have no factual authority and
are never rendered as answer evidence. Full answer evaluation must determine
whether this routing representation preserves the needed information.

The first 100 batches yielded 663 admitted atoms and 174 atoms with support
diagnostics. An audit found 121 of those 174 affected by lossless quote-format
normalization; the remaining diagnostics were not hidden or called verified.

The first 350 batches exposed two malformed JSON responses, batches 202 and 336.
Both lacked a closing quote at the end of a support list. Batch 365, found in the
first 500 batches, placed that terminator after the closing bracket. The narrow
repair restores only these support-list boundaries. It parses the repaired atom
fields, proves inserted quotes are inside support values, and preserves every
summary character. Other syntax, schema, attribution or summary-budget faults
still fail. Original authenticated provider responses remain unchanged.

| Admission | Atoms | Atoms with quote diagnostics | New calls | Complete namespace |
| --- | ---: | ---: | ---: | --- |
| Prefix 100, original policy | 663 | 174 | 0 | No |
| Prefix 350, support syntax policy v2 | 2,286 | 826 | 0 | No |
| Prefix 500, support syntax policy v3 | 3,258 | 1,194 | 0 | No |
| Prefix 650, support syntax policy v3 | 4,260 | 1,557 | 0 | No |
| Full 850, authenticated budget repairs and policy v4 | 5,556 | 2,067 | 0 | Yes |

Artifacts under `eval_results/full100-spine-corpus-20260909-r1/offset-000/`:

- `source-bound-atoms-prefix-0100.json`: `b9cad479ec8e97ff505946fdf84ce425ce1670b2a41e4c1b6a06e6bdcb4c4149`
- `source-bound-atoms-prefix-0350.json`: `1f5907c821bc026eca99082b43034e1832e5d159e26ab372d08980b64d508e61`
- `source-bound-atoms-prefix-0500.json`: `f14f903d104409c79a5ac08e03441ee60b116847da6ab8286e22c789d095ee90`
- `source-bound-atoms-prefix-0650.json`: `12523aa7c4c47a5e951aff05e7a8f5043fdf4df83c73b78d328041675ef67d2a`
- `source-bound-atoms-prefix-0850.json`: `6444cf274d60e025793c70e60f0d41de4766efa77b6b98950b8dabd0dfb889a5`

The partial prefixes are operational checks, never eligible full-memory
evaluations. The full850 artifact retains all 5,556 prepared fragment spans,
5,551 turns, 499 sources and 1,041,276 raw tokens, with raw-span population SHA
`8d744a026bbc9630d9eeef24cf2242edd4b09ae879a7abbf97c752c39b90574a`.

The final audit found eight support-JSON boundary faults and four over-budget
assistant-context summaries (194, 130, 136 and 138 tokens). The existing syntax
repair handled all eight boundary faults. A distinct, sealed Qwen batch compacted
the four generated summaries; its inputs contained no raw transcript text.
This cost one new gateway call and then replayed with zero calls. The admission
policy v4 permits only these explicitly authenticated budget compactions; it
retains original-summary and replacement-summary hashes, and every other
summary and raw span remains unchanged. This is a semantic compaction step,
not an unreported relaxation of the 128-token limit.

- Full audit: `eval_results/joint-1m-target-assessment-20260909-r1/source-binding-audit-full850.json`,
  SHA `2c4588afbe7fa53813cdae96d0e89803d9ae6f6e09bd27002994606fb18f4c36`.
- Compaction root: `eval_results/full1m-spine-budget-repair-offset000-20260909-r1`.
  Preflight SHA `eeedd20e7c8e1b005d1e19cde1cac72706d7154dc8b316cc18bfb6f22407be96`;
  repairs SHA `f58b7a578cdc34787b5aad8f9f37afd224ab005b041c3bacbe5c259203777d57`.
- Full admission policy SHA:
  `664f984d1594b856fb6a1c75f1db839e649c8d98ed0e9573598e4a7d8daf07e6`.

To replay full admission, include
`--summary-repair-root eval_results/full1m-spine-budget-repair-offset000-20260909-r1`
with `tools.admit_spine_corpus --request-limit 850` and the original corpus root.

## Complete-memory execution path

### First joint complete-memory result

The first run is complete: **summary hybrid scored 7/10; hybrid plus Qwen
reranking scored 4/10**. Both use the same complete 1,041,276-token memory and
attention-partitioned leaf index. Sol scored exactly the streamed predictions
whose retrieval and API times were measured. There were 50 streaming answer
calls and 15 unique judge calls for 20 logical judgments; identical judgments
deduplicated. Joint report SHA:
`821cc5dfd0e57ce0b35ee2ca795ddee0ccd9c54185666af4c9ce46b43aeedc65`.
Judge preflight SHA:
`8570fd25e1d4fe87b2e597fd883e62f6f68e47b2c4dc8503943252a941a48250`.
Answer session 92701 and judge session 65086 are finished. No target pass.

| Arm | Accuracy | Retrieval median / p95 | Total median / p95 |
| --- | --- | --- | --- |
| Short API | Not scored; timing control | — | 5.219 / 6.811 s |
| Summary hybrid | 7/10 | 0.253 / 0.321 s | 6.106 / 8.892 s |
| Identical-prompt hybrid API | Not scored; timing control | — | 5.450 / 9.300 s |
| Hybrid + Qwen | 4/10 | 0.611 / 1.354 s | 6.563 / 8.744 s |
| Identical-prompt Qwen API | Not scored; timing control | — | 5.659 / 11.577 s |

Qwen reranking lost three answers the conventional hybrid got right: the RAM
upgrade, total writing count and photography preference. It recovered no failed
hybrid answers. Both failed the music description, sports chronology and birthday
date difference. Raw inspection after scoring found that hybrid had retrieved the
bluegrass-band description but the reader abstained; the sports packet omitted
the three actual events; the birthday packet had the order date but omitted the
party date in another source. These are development diagnostics, not held-out
results. A strict timestamp cutoff would be unsafe for this fixture: one required
reference session is an hour after the question timestamp on the same day.
The next investigation is role-specific summary addressing and broader event
coverage while retaining exact raw hydration and attention-defined boundaries.

The joint report replayed with **15 authenticated judge hits and zero new calls**.
The post-score failure audit is `development-failure-audit.json`, SHA
`5aa2bf80fa416557b0b6225be00538d4f8b4e739705b5ea9a17683459c8194a8`.
A local cProfile diagnostic, `local-profile.json`, SHA
`29fd7a882dc72bd6bb95744237fe7a96d4c715e16b2e0068a69738077c7386f8`,
reproduced the two frozen ordinal-1 prompts without provider calls. Profiling
overhead is included, and these are not replacement target timings.

The separate user-spine address probe completed at
`eval_results/full1m-spine-user-addresses-offset000-20260909-r1`. All 2,744
additional vectors were embedded from stored user summaries only. Six focused
tests cover role projection, immutable vectors, original hydration descriptors,
source scope and invalid weights. Address manifest SHA:
`f3227cf4d021ace860aa717b70837c9100bf5ca465939aef517b028917737c31`;
routing probe SHA `0889e6b433a556806837e59a97baf557697c6092d8857764c83b3042385557f4`.
It compares combined hybrid, user-only dense/hybrid and an 80% user / 20%
combined dense/hybrid mix over the same ten queries. Query vectors are shared
across these diagnostic variants, so this is not a live-latency evaluation.
No answers were generated and no raw text was read by the routing probe.

The subsequent source-label audit was opened only after routing sealed, SHA
`d7a1f013655e42cdff9e6ecd42a6113195c6659a57481fc0a1dad679f7fe03b6`.
The new address variants restore both birthday-date sources in the top six,
but lose the photography source and cover only one of the three sports-event
sources. They are not promoted. Source coverage is not answer accuracy and can
count a source whose selected leaf lacks the actual fact.

`tools.probe_spine_cross_encoder` is now running in session **32665**, root
`eval_results/full1m-spine-cross-encoder-offset000-20260909-r1`. It builds a
summary-only union of up to 128 candidates from each dense channel plus 16 user
lexical candidates, then applies the already cached, pinned MiniLM relevance
model. Complete summary pairs must fit 512 model tokens without truncation.
Plain rank and a maximum-two-per-source ordering are separate diagnostic arms.
No raw reader, reference loader or gateway call exists in this probe. Parent
hierarchy session **83890** remains live under its original bounded allowance;
do not extend that old job merely to finish unused parent summaries.

The immediate evaluation path now compiles the complete leaf projection first.
Both measured query arms search and hydrate leaf sections; they do not consume
internal parent summaries. `tools.spine_leaf_projection` uses the same user-spine
attention windows, balanced cut rule, whole-exchange size limits and leaf channel
merges as the full builder. Six fixtures, including overlapping windows and
oversized exchanges, matched every full-builder leaf byte and attention cut.
The combined projection, recovery, hierarchy and joint-evaluation suite passed
45 tests. This change defers parent summary generation; it does not omit raw
memory or replace missing leaf summaries with placeholders.

The first snapshot already completed 487 of the 499 sources' leaves. The next
wave contains 12 leaf merges in two Qwen requests. Its root is
`eval_results/full1m-spine-leaves-offset000-20260909-r1`, preflight SHA
`4a8334748e197f3cfd59c13e1a8f1d0fb7b27ba30de61fa81deab63c5902a7ae`.
Only authenticated completed parent-cache outputs entered the frozen input
snapshot; live parent calls were skipped. Session **35247 finished** after two
successful calls and a post-compilation validation error: the search index sorts
section IDs, while the raw partition hash is in transcript order. All leaf
merges were complete. The corrected check validates the ordered leaves before
constructing the search index. A regression test covers reversed IDs, missing
leaves and reordered raw fragments; ten focused projection/persistence tests pass.
Original compiler bytes are preserved in
`eval_results/runtime-source-snapshots/pre-leaf-partition-check-20260909-r1`,
manifest SHA `e38f569432e4788b0bc052781e33b9d1030f082b8cfd1853f1357af4a57ea5f5`.

The corrected successor completed with **zero new model calls** at
`eval_results/full1m-spine-leaves-offset000-20260909-r2/hierarchy.json`, SHA
`c4a3bd957834d066831f702130d0c7d2f67c5f7fb425cc59a684da38b14769d9`.
It contains **2,744 leaves, all 499 sources and every original raw fragment**.
It inherits authenticated input-cache and completed-response snapshots, checking
preserved implementation bytes when importing the old compiler's cache.
Semantic matrix compilation **finished**, session 8035, at
`eval_results/full1m-spine-semantic-offset000-20260909-r1`. Index manifest SHA:
`7b42e343926a54099455df79eaf1335aa1af99e5875ab570218ffa594cc478b5`.
The 2,744 FP32 vectors occupy an 11,239,552-byte NPY file. Exact raw hydration
reassembly and complete leaf coverage passed; this phase made zero provider calls.

The joint answer phase **finished**, session 92701, root
`eval_results/full1m-spine-joint-offset000-20260909-r1`, preflight SHA
`811cd79cb889dde067ecc961189a294e1b813462c24a8b51afffe788909d4a79`.
Preparation completed with zero calls. The answer phase completed exactly 50
streaming Terra calls, using the frozen ten-question population and five arms.
Answers SHA `245af183329c1987c3a26fafa2e0badeb63f65d264dd695ff2e3c8c7a84b671d`.
Every frozen matched prompt reproduced, and all 20 live memory hydrations had
zero diagnostics. All 50 gateway responses exposed one visible chunk, so TTFT
is effectively total time. The medians are 5.219 s for short API chat, 6.106 s
for summary hybrid, and 6.563 s for hybrid plus Qwen. Their identical-prompt API
controls measured 5.450 s and 5.659 s respectively. These are timing results;
accuracy scoring is pending and the target has not passed.
Qwen sees summaries only; Terra receives authorized exact raw evidence. The
environment note records the concurrent bounded Qwen parent-ingest workload,
which can affect shared gateway latency across all arms. Its SHA is
`ec18b354897b1766408731b07ac7e50c6b5a529d2822d52bd39f485948aa8c6f`.

The original dataset file was missing when judging began, including outside the
sandbox. Its official public LFS pointer matched the locked SHA exactly. The
277,383,467-byte file was restored from the authors' release, verified locally
against `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`,
and copied to the original expected path without overwriting any file. Receipt:
`dataset-restoration.json`, SHA
`e2d081deaf329b795255c0dec13ef8b496451e9deee29617acfe78afa379719a`.
The frozen evaluator and locked validation population remain unchanged.
Parent hierarchy session **83890** remains active with its original cap; after
the first 48 regular successor batches it repaired two summaries and advanced
to another 43 batches. Do not restart that live process. This is not yet an
answer-accuracy or query-latency result.

1. Raw ingest session **78281 is finished**, exit code zero. Its complete run
   made 846 new Terra calls and reused four original completions, with zero
   retries. The legacy strict output admitted 2,531 atoms and marked 520 batches
   invalid; its SHA is `71d757142b0bbf27771a275437fbd3ddfb92ef4fbaa6d68deee40a1bd698faa1`.
   Preserve that output as diagnostics; the new source-bound policy handles
   quote failures separately. Do not restart the raw job.
2. Full source-bound admission completed with all 5,556 atoms, four declared
   compactions and zero new calls during admission. Both the complete raw span
   population and all summary slots were checked.
3. `tools.build_spine_corpus_hierarchy` finished in session **10417**, exit one, at
   `eval_results/full1m-spine-hierarchy-offset000-20260909-r1`, under preflight
   `7ffa2997364575eb33b186c16e3c68c2551617a5e681690a2a382e092437106a`.
   All 58 submitted gateway calls completed and are authenticated. Two batches
   failed output validation: one duplicated all eight job labels, and another
   had one 149-token summary. The remaining 451 job outputs are reusable;
   nine jobs require explicit recovery. No requests remain unacknowledged.
   The initial eight exchange merges fit one successful Qwen batch. Independent sources
   advance in dependency waves; up to eight typed summary jobs share a Qwen
   gateway request, with four requests concurrently. Singleton or concatenated
   summaries that already fit are reused exactly. Over-budget merges use Qwen.
   No summary omission fallback is enabled. Every request is sealed before I/O;
   completed batches replay and unacknowledged calls are not retried.
   The first hierarchy wave contains 452 merge jobs in 57 gateway batches.
   The completed-wave audit SHA is
   `b65bd89894932e2eef4648071e06b93e148a1e9d20576fad24db6a85634fb0e9`.
   The original builder and its pinned modules remain unchanged. The successor
   `tools.build_spine_corpus_hierarchy_resilient` preserves valid, unambiguous
   outputs, recovers ambiguous slots singly from the original summary inputs,
   and permits at most two completed recovery attempts per failed job. All
   queued regular calls consume the run allowance before recovery can schedule
   another call. Completed scalar attention scores also reuse exact-input caches.
   Successor root: `eval_results/full1m-spine-hierarchy-offset000-20260909-r2`;
   preflight SHA `152dd31f91a84e772c9958f762c5dfcd4dfdc77f40f5d3f15bc32128faa8edda`.
   Session **83890** is the live successor, with an allowance of 198 new calls.
   Preparation made zero provider calls. Poll this process before any restart.
4. Qwen's local neutral attention probe sees user-spine summaries only. Scalar
   signals are cached during ingest so a later merge wave does not repeat the
   same attention forward. The existing source-bound hierarchy constructor
   preserves whole exchanges and exact raw partitions.
5. `tools.compile_spine_semantic_index` requires the complete namespace. It
   persists FP32 leaf vectors, binds their matrix bytes and encoder identity,
   and reassembles an authenticated raw hydration store from the prepared
   fragments. It rejects fragment gaps, overlaps and changed bytes.
6. `tools.evaluate_spine_namespace` freezes all ten questions for that namespace.
   Its v2 protocol rotates group order across five arms: short API, live summary
   hybrid, its identical-prompt API control, live hybrid plus Qwen, and its
   identical-prompt API control. Preparation freezes the evidence prompts;
   every live memory answer must recompute embedding, routing and hydration
   inside its end-to-end clock and reproduce the corresponding control's bytes.
   It never reuses a query vector or prediction. Parameters are eight Qwen
   candidates, six final sections, two reserved lexical matches, 4,096 context
   tokens and 256 answer tokens. The complete namespace requires 50 answer
   calls; Sol judges the 20 live-memory predictions after all answers seal.
   Each memory/control pair is adjacent; each method runs first on five of ten
   questions, counterbalancing order and potential provider cache effects.
7. `tools.report_joint_spine_full100` requires ten distinct complete namespaces
   and the same evaluation and hierarchy-compilation policies. It authenticates
   the judge checkpoints with zero calls before aggregating. Each method must
   have all 100 question identities; no inherited answers or per-question method
   switching can enter the score. Accuracy and latency must pass on one method.

This first namespace is an intermediate development result. The eventual target
requires the full100 population across ten complete memories, using the same
implementation for accuracy and latency. The provisional latency tolerance is
still 10% at median and p95; the user has not selected a numerical tolerance.
The short API arm is a UX timing baseline; its missing evidence can change its
output length. The identical hydrated-prompt controls isolate retrieval
overhead. The conservative full100 gate currently requires the provisional
10% median/p95 allowance against both baselines, reporting the two comparisons
separately. A fast identical-prompt result cannot hide a slow short-chat result.

The complete raw hydration store is already staged and authenticated at
`eval_results/full1m-spine-semantic-offset000-20260909-r1/raw-turns.json`, SHA
`364c486f67c84543c1081a1538d0d2368ffa3b602220d42ebef2d2f63bb55197`.
All 5,551 reconstructed turns, 499 sources and 1,041,276 raw tokens match the
prepared namespace. This used zero provider calls. The hierarchy and semantic
matrix are still required before the complete-memory evaluation can start.
Index compilation additionally checks that leaf spans cover every prepared raw
fragment exactly once, rather than relying on a completeness flag alone.

The dependency planner was exercised on the admitted prefix500 as an explicitly
partial operational check. Its initial exchange wave needed only six merges,
packed into one request. That one Qwen gateway request completed successfully;
all six summaries validated, with a maximum of 46 tokens. Provider-free replay
returned one authenticated completion hit and zero new calls. No local attention
or full hierarchy was built for this partial check.

The root is `eval_results/spine-merge-planning-prefix500-20260909-r1`:

- Preflight: `b585c18b538f1b3dbb5e1dee06c3ee893dee75ac01b088a6158f22e99049ab13`
- Batch: `d98a7fd95d534497456fa69c5287cdc194ca898a4924207cc4ff2214e545e53a`
- `batch-smoke-report.json`: `289f0b43b9262139fc2318374f71e0812239b76d1cf0f38dc743c73997426e21`

Its single-call process (session 15075) is finished. The later budget-compaction
session 47407, raw ingest session 78281, and original hierarchy session 10417
are also finished; session 83890 is the live full-hierarchy successor.
Do not run this partial prefix through the complete
evaluation or mistake its six summary merges for a 1M-token hierarchy.

## Verification and source continuity

The combined relevant suite passed **84 tests, with three slow tests deselected**.
After the additional misplaced-support-terminator case, its three focused tests
passed. An earlier run had five setup errors caused by inaccessible default
pytest temporary storage; all five passed using a workspace-local temp directory.
The subsequent matched-prompt, streaming and persistence suite passed 14 tests;
the full100 gate, namespace lifecycle and persistence suite passed 10 tests.
These are overlapping focused runs, not additive test counts.
Counterbalanced pairing and the full100 gate passed nine focused tests. The
summary-budget repair, batching and admission suite passed 15 tests.
The resilient recovery, batching and persistence suite passed 16 tests, including
partial output retention, rejection of ambiguous attribution, two-attempt
recovery with authenticated zero-call replay, scheduling allowances, and exact
summary-input attention cache reuse. The cross-namespace compilation policy hash
excludes only bound population and cache provenance, and verifies its declared
method hash before index persistence.

Tests cover summary-only batching and attribution, dependency-wave ordering,
atomic admission, malformed support syntax, exact Unicode fragment reassembly,
persisted-vector tampering, live retrieval before streaming, answer/response
binding, and refusal to retry unacknowledged calls or open gold prematurely.

Historical source copies live under `eval_results/runtime-source-snapshots/`:

- `pre-semantic-routing-20260909-r1`: preserves the old routing and shortlist
  modules. All 16 old fixture BM25 route receipts remained unchanged after the
  backend extension. Old whole-file source hashes require these prior bytes.
- `pre-local-embedding-loader-20260909-r1`: preserves the old embedding loader.
- `pre-support-json-repair-20260909-r1`: preserves original prefix100 admission;
  manifest SHA `7f3b19ad324bdd521f5f8055be9ce74944227048ee036dcf46505a130891eb9a`.
- `pre-support-json-boundary-repair-20260909-r1`: preserves prefix350 admission
  and syntax repair v2; manifest SHA
  `e1b5d14a6eb6adc6a6db03eea0a60381e08ef3435ae307a02ee5ef0cfbd608a7`.
- `pre-summary-budget-admission-20260909-r1`: preserves policy v3 admission for
  prefix500/650 replay; manifest SHA
  `997a81b15c917d2dd8afa947bc752c2350a28f36741465d9f4245976db4a44b6`.

No router is promoted. No new answer-accuracy score has been claimed. The
historical slow 95/100 and fast 73/100 results remain separate from this work.
