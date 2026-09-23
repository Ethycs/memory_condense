# Hot raw-chunk retrieval, source-local linking, and temporal windowing

Date: 2026-09-05

Status: complete dev1M successor; retrieval replay is byte-identical and the
sealed Terra/Sol answer plane scores 10/10 semantically

## Result

The linked-and-windowed v6 hot path queried one 1,039,203-token concatenated
memory with ten development questions for 20 cyclic rotations, producing 200
measured question-to-serialized-packet samples. Its protected union reached
every annotated evidence source for 10/10 questions, packed every selected
chunk, and ended at 52.58465 ms p50 and 71.4195 ms p95. The largest raw packet
contained 39 chunks, 4,460 context-token proxies, and 5,041
prompt-workspace-token proxies, below the frozen 7,000/8,000 caps.

The retrieval plane itself made zero Qwen or provider calls and replayed its
gold-blind selection byte-for-byte. The already sealed raw packets were then
sent concurrently to Terra with no gold. Ten sealed predictions were judged
independently by ten Sol calls carrying question, reference, and prediction.
Sol accepted **10/10**. Both provider phases used zero retries.

This supersedes a provisional manual sufficiency judgment on linked v5. When
v5's exact packets were actually answered, Terra included a stale `The
Killers` festival in the concert list and Sol scored the result 9/10. V6 adds
one general query-derived calendar-lookback rule before every lane's selection;
the concert question's inclusive 2023-02-22 through 2023-04-22 window removes
that out-of-range decoy while retaining the five required events.

The experiment used one shared 1M-token memory store and ten probes. It did
not independently ingest ten different 1M-token prompts. The 200 latency
samples are repeated retrievals from the resident store, not 200 generations.

## Frozen identity

| Item | Identity |
|---|---|
| Dataset | `C:\Users\Keytone\Downloads\memory-condense-rig\datasets\longmemeval_s_cleaned.json` |
| Dataset SHA-256 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Split manifest | `docs/10 - Research Log/data/longmemeval-95-target-split-v2.json` |
| Split SHA-256 | `8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4` |
| Population identity SHA-256 | `fa9a06ebd103d87086943cfa94091bdf607fe07874bc871e465aad409b85ca18` |
| Current corpus SHA-256 | `1bcfb2b4b59bce08f240d36be02f1d323d7f5a2294e69b3ccf13b1c647fc7e59` |
| Corpus | 1,039,203 transcript-token proxies; 5,400 turns; 7,895 chunks; 10 questions |
| Source selection SHA-256 | `16756d07d7ada13fec52387f9be585bcc24a5454499f33759b12d71a5d980f5b` |
| Source receipt SHA-256 | `92c764d7fabfbeef9d068fc52210148eb44b4613530d987f2c5856baeda5bb45` |
| Source SQLite SHA-256 | `222f2b3ff39d9e0b9ed4a04a75b60ef1edeb1ada3a7b67b74cf2b354fea88dbc` |
| Source HNSW SHA-256 | `5999fdc048ca02c1936957cff985f53808c9dc49b21b7101d2a2db51c78d561c` |
| Query encoder | `BAAI/bge-m3`, revision `5617a9f61b028005a4858fdac845db406aefb181`, 1,024 dimensions |
| Encoder checkpoint SHA-256 | `a3d5c49f064ab58d7cf5bba1c2085918f529778e88535aca7de674c9094af0b7` |

The query encoder ran on CUDA and made one embedding forward per question.
Stored dense rows are float32 and L2-normalized. The tokenizer proxy was
`tiktoken` 0.13.0 with `cl100k_base`; therefore token counts are stable budget
proxies rather than provider billing counts.

## What moved to ingestion and what remained online

The hot path separates query-independent addresses from the live query. The
base artifact and resident setup together provide:

- a text-free 7,895-row chunk manifest;
- a resident, exact Okapi-BM25 address over the durable SQLite term index;
- a memory-mapped `7,895 x 1,024` exact dense matrix;
- source, turn, ordinal, and within-turn coordinates for local linking; and
- query-independent features used by the implicit temporal-event specialist.

At query time the path performs one BGE-M3 query encoding, runs the retrieval
lanes, unions their already-selected outputs, hydrates the winning raw chunks,
renders the dated responder prompt, counts the budget, and serializes the
provider-ready bytes. Qwen attention, CAV construction, a reranker, a summary
model, and a final answer model are not on this measured path.

The retrieval question deliberately remains the plain question. Dates are
added only to the responder-facing prompt. The earlier attempt to embed the
dated presentation header changed the search vector and lost ordinary source
matches.

V6 nevertheless uses the sealed question date for an orthogonal operation. If
the question asks for an explicit calendar lookback, the question-only route
derives inclusive source-local wall-clock bounds and applies them to BM25,
exact dense, temporal-event, and source-neighborhood candidates **before each
lane selects its protected budget**. Questions without a dated lookback remain
unfiltered. This is one general parser and admissibility rule; it contains no
question ID, answer text, or gold-derived cutoff.

## Separate specialist budgets

| Lane | Fixed budget | Specific job |
|---|---:|---|
| `a0_bm25` | 8 | Exact lexical anchors, including names and rare terms |
| `a1_exact_dense` | 8 | Semantic anchors over all resident chunk vectors |
| `a2_source_neighborhood` | 8 | Immediate same-source predecessor/successor turns around selected anchors |
| `a2_temporal_event` | 24 | Diffuse first/last/earliest/latest event enumeration |
| `a3_protected_union` | additive | Protected post-selection union of all four lanes |

Each primary search may examine up to 96 candidates. Selection happens inside
each lane before global chunk-ID deduplication. If a later lane collides with a
protected earlier chunk, it refills from its own tail instead of silently
losing budget. The temporal lane reserves 24 candidates for those possible
collisions.

The source-neighborhood index is compiled once and retains no raw text. It
orders sources by their first protected anchor, walks anchors in protected
order, visits adjacent turns in source-local ordinal order, orders chunks
within a turn by `(start_char, chunk_id)`, and round-robins across sources. The
anchor set is the protected BM25/dense/event union. This makes local closure an
additive specialist rather than a replacement for global retrieval.

For `gpt4_d6585ce8`, the phrase "in the last two months" and sealed question
time 2023-04-22 19:31 derive `[2023-02-22 19:31,
2023-04-22 19:31]`. The same bounds govern all four lanes before selection.
The v5 Killers evidence falls outside them, so it cannot occupy a selected or
refill slot. This temporal rule reduced that packet to 32 chunks and 3,903
context-token proxies.

## The Serenity Yoga orphan and repair

Question `6ade9755` asks, "Where do I take yoga classes?" The required answer
is `Serenity Yoga`. Before linking, BM25 and dense search selected the correct
source, `6ade9755::answer_9398da02`, but selected a generic assistant list of
yoga applications rather than the adjacent user statement containing the
answer. A source-level hit therefore overstated answer readiness.

The relevant local sequence was:

1. ordinal 1084: the user asks for yoga-app recommendations;
2. ordinal 1085: the assistant gives the generic app list selected by the base
   retrievers; and
3. ordinal 1086: the user says that Down Dog helps on days they cannot make it
   to `Serenity Yoga`.

The missing chunk was
`6a4b613467db4f171d6bb458c6f2149941d28afc249711bc0e84b79b1fb16f8f`.
The eight-slot source-neighborhood lane recovered that immediate successor,
and it survived post-selection deduplication/refill and final packing. This is
a local-to-global connectivity repair: classical search identifies the right
region, then cheap source-local adjacency restores the answer-bearing turn.

## Exact command sequence

The five retrieval phases are process-separated so that gold cannot enter
selection and clock noise cannot enter the semantic selection artifact. Two
subsequent processes answer the sealed packets and judge only the sealed
predictions:

```powershell
$py = '.pixi\envs\dev\python.exe'
$out = 'eval_results\longmemeval-1m-hot-retrieval-linked-windowed-development-20260905'
$dataset = 'C:\Users\Keytone\Downloads\memory-condense-rig\datasets\longmemeval_s_cleaned.json'
$split = 'docs\10 - Research Log\data\longmemeval-95-target-split-v2.json'
$source = 'F:\Keytone\Documents\GitHub\memory_condense\eval_results\longmemeval-1m-recall-guarded-cumulative-development-20260821\source-current-selection.json'
$selectionSha = 'cafe769331b36a2500b43d012360a775669e9ffdc4519bf6a115f83c767cba06'

& $py tools\assay_hot_retrieval_1m.py --output-root $out export-probes --dataset $dataset --split-manifest $split
& $py tools\assay_hot_retrieval_1m.py --output-root $out compile-base --source-selection $source
& $py tools\assay_hot_retrieval_1m.py --output-root $out run --source-selection $source --device cuda --lane-budget 8 --candidates-per-lane 96 --max-context-tokens 7000 --max-prompt-tokens 8000 --warmup-rounds 1 --repeats 20
& $py tools\assay_hot_retrieval_1m.py --output-root $out replay --source-selection $source --device cuda
& $py tools\assay_hot_retrieval_1m.py --output-root $out score --dataset $dataset --split-manifest $split
& $py tools\evaluate_hot_retrieval_answers.py --output-root $out --expected-selection-sha256 $selectionSha --max-concurrency 10 answer
& $py tools\evaluate_hot_retrieval_answers.py --output-root $out --expected-selection-sha256 $selectionSha --max-concurrency 10 judge --dataset $dataset --split-manifest $split
```

`export-probes` is the only pre-score phase that reads the development dataset,
and it emits gold-free question records. `compile-base` reads no questions.
`run` and `replay` read the sealed probes and source only. `score` joins the
reference metadata only after the byte-identical replay is sealed. `answer`
reads only the gold-free `selection.json`; `judge` verifies the sealed answer
artifact before it joins references. The gateway key comes from the configured
`LITELLM_KEY` environment variable and is never written to an artifact.

## Latency

One warm-up rotation preceded 20 measured rotations. The 200 measured samples
were cyclically rotated to avoid always assigning the same question the same
position. The primary boundary begins with the plain question and ends with
serialized bytes ready for the final LLM:

| Stage | p50 (ms) | p95 (ms) |
|---|---:|---:|
| Query route plan | 0.0510 | 0.1501 |
| BGE-M3 query encode | 38.5519 | 56.5735 |
| Resident exact BM25 | 2.6228 | 5.6457 |
| BM25 temporal admissibility | 0.0004 | 0.3188 |
| Exact dense scan | 3.3172 | 3.9652 |
| Dense temporal admissibility | 0.0002 | 0.2563 |
| Temporal-event search | 0.0012 | 4.6731 |
| Parallel retrieval wall | 42.3199 | 61.6032 |
| Neighbor-anchor union | 0.0714 | 0.0895 |
| Source-neighborhood lookup | 0.4785 | 1.0225 |
| Neighbor temporal admissibility | 0.0003 | 0.1804 |
| Protected lane union | 0.0589 | 0.0802 |
| Raw chunk hydration | 2.9852 | 4.8178 |
| Pack, render, and count | 6.4458 | 8.3279 |
| Serialize | 0.1994 | 0.2677 |
| **Question to serialized provider bytes** | **52.58465** | **71.4195** |

BM25, dense, and event work overlaps inside the parallel-retrieval wall, so
the rows must not be summed. Setup, provider transport, provider prefill, and
answer decoding are outside this boundary. Post-boundary audit diagnostics are
also reported separately in `runtime.json` and do not inflate the provider-ready
number.

Both latency gates passed: p95 was below the initial 200 ms target and the
100 ms stretch target.

## Retrieval and packet results

| Arm | All required sources | Literal-answer hits | Mean source recall | Mean context proxies | Maximum context proxies |
|---|---:|---:|---:|---:|---:|
| BM25-8 | 6/10 | 2/10 | 0.7433 | 1,287.6 | 1,896 |
| Exact dense-8 | 6/10 | 4/10 | 0.7467 | 968.3 | 1,386 |
| Source neighborhood-8 alone | 8/10 | 3/10 | 0.8333 | 1,143.8 | 1,786 |
| Temporal event-24 alone | 2/10 | 0/10 | 0.2000 | 291.7 | 1,804 |
| **Protected union** | **10/10** | **5/10** | **1.0000** | **3,475.7** | **4,460** |

The temporal arm applies only to its target question class. Its low aggregate
standalone result is expected: on the two parseable distributed-list questions
it recovered all 11/11 answer components, giving the protected union mean
component recall 1.0 over eligible questions. It should not be judged as a
general retriever.

The final packet averaged 26.3 raw chunks and 3,475.7 context-token proxies.
Its maxima were 39 chunks, 4,460 context proxies, and 5,041 workspace proxies.
All selected chunks fit; there were zero packing drops.

Literal containment is intentionally a weak diagnostic. Five packets contain
a normalized answer string, but one scalar `3` can match an unrelated numbered
list, while several correct packets require sorting dated events, collecting a
distributed list, or subtracting two retrieved operands and therefore cannot
contain the final reference verbatim. The manual audit instead checked whether
the raw packet carried the decisive facts and operators' inputs. Source-local
linking restored the only clear evidence orphan, `Serenity Yoga`; v6 Terra then
answered it exactly. The v5 9/10 answer result also shows why evidence presence
alone is insufficient: an admissible but stale decoy can still alter a list.

## Compile, setup, and resident memory

The query-independent v6 compile took 1.6724382 seconds for 4,720.65 chunks/s
and emitted 35,174,675 bytes (33.545 MiB) of derived index artifacts. The dense
matrix accounts for 32,338,048 bytes (30.840 MiB), and the text-free manifest
for 2,836,627 bytes (2.705 MiB).

Before measured samples, runtime setup took 32.2055907 seconds. The main
reported components were:

| Setup component | Time (s) |
|---|---:|
| Model load, checkpoint verification, first forward | 18.2244 |
| Resident BM25 compile | 11.5738 |
| Diagnostic-arm materialization | 0.6559 |
| Compiled integrity and dense mmap | 0.2968 |
| Database open and lexical statistics | 0.1634 |
| Source integrity verification | 0.1505 |
| Probe integrity | 0.0165 |
| Warm-up | 0.9070 |

The resident BM25 numeric payload was 2,801,532 bytes (2.672 MiB). The
source-neighborhood coordinates occupied approximately 1,153,664 bytes
(1.100 MiB) across 7,895 chunks, 5,400 turns, and 482 sources. Total additional
compiled artifacts were 35,174,675 bytes, comfortably below the 512 MiB gate.

Resident process RSS was 1,681,760,256 bytes (1,603.852 MiB). CUDA reported
2,279,670,784 allocated bytes, 2,282,588,160 peak allocated bytes, and
2,298,478,592 reserved bytes. These process figures include the loaded BGE-M3
query encoder; they are not the size of the retrieval indexes alone.

A separate exact-parity microbenchmark on the same 7,895 chunks measured the
durable SQLite BM25 path at 59.54 ms/query and the resident BM25 path at
0.531 ms/query, a 112.2x speedup. That isolated microbenchmark is not the same
timing boundary as the integrated v6 BM25 row above; v6 additionally performs
query planning and competes for CPU time with parallel lanes.

## Sealed answer and judge result

The answer process submitted exactly the A3 raw-chunk provider messages sealed
by `selection.json`. It did not rerun retrieval and did not receive reference
answers. The judge process then verified the answer SHA, joined the pinned
references, and submitted only question, reference, and sealed prediction to
an independent model.

| Phase | Model | Physical calls | Retries | Concurrency | Per-call p50 | Per-call maximum | Result |
|---|---|---:|---:|---:|---:|---:|---:|
| Answer | `codex_sdk/gpt-5.6-terra` | 10 | 0 | 10 | 8.0746 s | 11.5196 s | Ten sealed predictions |
| Judge | `codex_sdk/gpt-5.6-sol` | 10 | 0 | 10 | 9.3051 s | 10.1345 s | **10/10 correct** |

Terra used a 256-token completion cap; Sol used the repository's 1,024-token
binary-judge cap. The answer prompt population carried 38,077 token proxies
and emitted 84 completion-token proxies. The judge population carried 1,528
prompt-token proxies and emitted 171 completion-token proxies. Per-call times
are distributions over concurrent requests, not sequential batch wall time
and not part of the 71.4195 ms retrieval boundary.

The two provider artifacts are:

- `answers.json`, SHA-256
  `b423e297c9ac916a08d729ef3c542af8d02e5f204ddfa9bb78beb530f65fa435`;
- `answer-judgments.json`, SHA-256
  `cc5638c2a0c80263a4f1b198902e54c88d8b38232791caf5852519f474249b08`.

The corrected concert prediction lists Billie Eilish, the free outdoor park
concert, the Brooklyn music festival, the local-bar jazz night, and Queen with
Adam Lambert, with no Killers event. The source-local repair also produces the
exact `Serenity Yoga` answer.

## Successor comparison

| Run | Retrieval policy | p50 (ms) | p95 (ms) | All-source reach | Literal hits | Terra/Sol semantic result | Mean/max context proxies |
|---|---|---:|---:|---:|---:|---:|---:|
| Dated-query ablation | Embed dated presentation text | 38.8391 | 51.3222 | 7/10 | 3/10 | Not run | 1,683.5 / 3,017 |
| Plain-query v4 | Plain question | 50.3209 | 56.6840 | 10/10 | 4/10 | Not run | 2,456.7 / 3,703 |
| Linked v5 | Plain + source neighborhood | 52.5522 | 74.7484 | 10/10 | 5/10 | **9/10** | 3,565.1 / 4,797 |
| **Windowed v6** | **V5 + pre-selection calendar window** | **52.58465** | **71.4195** | **10/10** | **5/10** | **10/10** | **3,475.7 / 4,460** |

The dated-query run was numerically faster but semantically worse. Adding a
date header to the text sent through BGE-M3 diluted three ordinary searches;
the loss was query formulation, not packing. V4 restored the plain retrieval
question and all ten required source sets, but exact source reach hid the
Serenity orphan. V5 kept that corrected query form and added bounded local
closure. Its later answer assay scored 9/10: Terra appended the stale `The
Killers festival` to the otherwise correct five-event concert list, and Sol
rejected that one row. The answer and judgment SHAs are respectively
`977481db55adb59e1b6be2d82337a9b6ffdc38d7656f0962c61533ae38c5f743`
and `f098651c8791a01c5a9fd4987429a985d403d03252655378e77fe78a5827b77c`.

V6 turns the failed row into a general policy repair rather than a question
patch. Compared with v5, p50 changes by only +0.03245 ms, p95 falls by 3.3289
ms, mean context falls by 89.4 proxies, and the maximum packet falls by 337
proxies. Each calendar-admissibility check remains below 0.319 ms at p95. The
stable conclusion is that source-local linking and query-derived temporal
scope compose under the 100 ms retrieval target.

Immediate predecessor artifact identities:

- Dated-query ablation: selection
  `bf5e7e3b30e815168a0b920c36e5b66ca39edf285f416c1b66968b1c23e940ed`,
  runtime `e1091c7ea9bfdf0d023c446659861ed173ff84583ad3cc321cb997be64c61aab`,
  replay `e9690bf9889ab42b8ec2ed6ed037045f106f31ef36516fad19cdbba9367e64f6`,
  score `fdda5daabaeea6250023986d35e247bcf9129c01ba6f52b77cb6a2887469d955`.
- Plain-query v4: probes
  `4c224e98eaa8d715cce6e93b548da229b6c7813006fa9dc2d1877431f268d7ea`,
  compiled `fa598121e8b119f68edf8749f82569d2f97459a284beda64a19544a8dbe4272b`,
  selection `117d84908dccf896022582c3623ac73037e09db72b40eabc8c025232cf60fd61`,
  runtime `5b07840a445913526cdaec4c09acc8c412c23d355ec4b62bb4531ee9ac478d1f`,
  replay `0076f8283273f7b3c7fa69f2f5d8d38c517200f63eb7049a752103e129088436`,
  score `5c13cf8b5e57e2443b701fb04e91f6c51d362c828de55fcf12fde0e02c7a2095`.

## V5 diagnosed-ablation receipts

Canonical directory:
`eval_results/longmemeval-1m-hot-retrieval-linked-development-20260905`

| Artifact | SHA-256 |
|---|---|
| `probes.json` | `697239022ab1f8e1e46a4e346987900ab3e04f1ada572442c17a1aeefd5ae6d9` |
| `compiled.json` | `0098b763b23d58b5a8693ca20b8fc8f614d2ca8ce09f1278227a9866232d54c9` |
| `compile-runtime.json` | `fdb074796a7919c4001f7ce40d188bcc530ee482acdcb14ad92b1370a3e9bf73` |
| `selection.json` | `555b184cf75f485b184ba7246e4d39ba766b844c5f89fbba93b9edc83e9c763a` |
| `runtime.json` | `47c09486e0e476499cf8f375a7cdce7e3d8642494e75b1f7b50f44bbaef66a89` |
| `replay.json` | `f341a27116d4c642e677ccdee0430646c74f06d152cfca1fe48a69e2f47af334` |
| `scores.json` | `54e8540743de00fb521e2cd18657424bd257a63a068a7860a4fa0e6d60920486` |
| Dense matrix | `336595ef11d1bda0a9a95bea5cba2c8f0a9d148aae877ea96da6620e50d1b632` |
| Text-free chunk manifest | `90a67ad2893787bb638170483e87a6c5dc25b4a2a58e6cfdaa31348c606e2bea` |
| Implementation receipt | `713cbe935f62f6d6e84f9a5768e1216d1485b673343fd7be18cfc03a052b502d` |

V5's retrieval replay is byte-identical, but its subsequent Terra answer SHA
`977481db55adb59e1b6be2d82337a9b6ffdc38d7656f0962c61533ae38c5f743`
and Sol judgment SHA
`f098651c8791a01c5a9fd4987429a985d403d03252655378e77fe78a5827b77c`
seal the 9/10 Killers-decoy failure. The artifact remains a useful diagnosed
ablation; it must not be relabeled as the 10/10 successor.

## V6 successor receipts

Canonical directory:
`eval_results/longmemeval-1m-hot-retrieval-linked-windowed-development-20260905`

| Artifact | SHA-256 |
|---|---|
| `probes.json` | `58b46ac89044780b02f21a3ce4fa8896caa8ab68027fb42dc69935a12d8409f7` |
| `compiled.json` | `0ff8dc76830d8ba52ee5acc78121ffede8313c450a78891d39379b904abbdca1` |
| `compile-runtime.json` | `980752d6cd0c866023850f8b5ec78aff7ae0c242dbdd43df6ae0b6ff61216bf9` |
| `selection.json` | `cafe769331b36a2500b43d012360a775669e9ffdc4519bf6a115f83c767cba06` |
| `runtime.json` | `6e6f0ab33a4b0a0d2c58f01a39d125ded3c9240af80571f91d0f4685e99be441` |
| `replay.json` | `2f7679bf1f8bb3788d663db9636fba916a7c1a33157d8365e2a95f0f06b50b8e` |
| `scores.json` | `3e3329004d54d9961e04fbef2f170db695a972f0400b236c4a0bbffac4c9d1d5` |
| `answers.json` | `b423e297c9ac916a08d729ef3c542af8d02e5f204ddfa9bb78beb530f65fa435` |
| `answer-judgments.json` | `cc5638c2a0c80263a4f1b198902e54c88d8b38232791caf5852519f474249b08` |
| Dense matrix | `336595ef11d1bda0a9a95bea5cba2c8f0a9d148aae877ea96da6620e50d1b632` |
| Text-free chunk manifest | `cd9fc81644f47da2af7e1e9d7ca2e0c0a70fe4f261fb02aeb2b4352c23b15d0e` |
| Implementation receipt | `5863bbbde0f8ebc277e23bfd4188c3c4aec2d64a57cc03272b07708097809859` |

Replay reconstructed the gold-blind semantic selection byte-for-byte. All
available development evidence gates passed: source reach, eligible component
recall, literal diagnostic threshold, protected BM25 retention, both packing
caps, address-size cap, replay, latency, and zero Qwen/provider calls.

Formal promotion is still false because this pinned dev1M projection does not
provide an exact annotated-turn non-regression metric. It is also an
analysis-used ten-question development set, so 10/10 is not a 95% claim on a
fresh population. What is now established on this exact fixture is narrower:
the hot retrieval packet stays below 100 ms p95, survives byte-identical
gold-blind replay, and supports 10/10 independent semantic answers when its
bounded raw chunks replace the primary LLM's original 1M-token context.
