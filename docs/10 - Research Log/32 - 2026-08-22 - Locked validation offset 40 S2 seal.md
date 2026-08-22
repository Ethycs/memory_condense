# Locked validation offset 40 S2 seal

**Status:** validation offset 40 is the fifth of ten sealed provider-free
retrieval shards in the locked 100-question campaign. Its canonical root and
ten question parts passed verify-only replay and a separate read-only
artifact/store audit. Offset 50 is now running. The ten-shard merge, fixed-S1
Terra answer artifact, independent Sol score, >=95% gate, and production-bound
Mem0 comparison remain unresolved.

This shard records a fully sealed S2 admission on `q047`: four evidence rows
were appended after S1 while preserving the frozen caps and prefix. Admission
does not demonstrate relevance, answer use, or improved accuracy. No responder,
judge, or gold-bearing score ran here, and the preregistered answer-stage
treatment remains fixed at S1.

## Root and question-part identities

The canonical artifact is
[`shards/offset-040/retrieval.json`](../../eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822/shards/offset-040/retrieval.json).

| Property | Observed value |
| --- | --- |
| Root file SHA-256 | `7ffc09f2d444146fdb0a5ad1636b28d108d3aed34ad545fe238878c5260bee29` |
| Root file size | 2,385,386 bytes |
| Gold-blind 100Q population | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| Offset-40 shard | `926068ec45534aa64316080d68b53d7a6a65183514faa1cfc7cd8240beddc4e1` |
| Retrieval implementation | `cf2577f21a7a1af1b9c5f331c7eb1672c5ba13af84ccdc2f718259192bb36e09` |
| Effective retrieval policy | `0abb42db8fd35b566029be135630af34fa42ee9a464ef8c4a889948729cbab13` |
| Environment lock | `c99f82aa37a1df009c6100110d24aeaee0bde3376ff09537700ad9848b371a95` |
| Transcript-token proxies | 1,046,567 |
| Turns / questions | 5,489 / 10 |

Every part is canonical JSON with an exact sidecar and an exact ordered match
to the root's `question_part_sha256s` entry and embedded question object:

| Part | Question ID | SHA-256 | Sealed elapsed seconds |
| --- | --- | --- | ---: |
| `q040` | `9d25d4e0` | `06a832e7cbbcfdacf3bb7433ea0628aa53747bdeaba68660fc0c823faa4e9c0a` | 154.32529589999467 |
| `q041` | `94f70d80` | `cfe944886eaec494bcba021ba6c5acbb71ca5a21209426e973f6fc5e9e23526a` | 140.59820639999816 |
| `q042` | `a96c20ee_abs` | `ad6d5e1571829c8bd5bca91713c7c1259e664fe7d92d3e821bb4b0df2011f7b4` | 134.96860480000032 |
| `q043` | `gpt4_1e4a8aec` | `a662c2f3e18295f43d66f6ec802e9da94d92c0ac441bd459238898bec0c1267f` | 197.9119442000083 |
| `q044` | `89941a93` | `d15758688f6cb01549b0dbe636ea0a2d3ca52317dd358f1a845eb73d386ca721` | 213.28627470000356 |
| `q045` | `70b3e69b` | `e2feaf4b681dc7adf90aeb3418675b5cba6c2926f3a9342a7f88957abf18fe47` | 141.9378233000025 |
| `q046` | `18bc8abd` | `d3f1b854feee7e101662868ec581d8b7f1979e6d3d0e02425616ac79c0a9fbaf` | 144.8268584000034 |
| `q047` | `b6019101` | `d41259b8c3a56586c1ee2890eeb2f81cefec95256f4bddad278791c395e2e1f7` | 275.79288630001247 |
| `q048` | `gpt4_0a05b494` | `c7a07dc5f97d258a677441ea695a168cbd6e64d1b53d713ca0a46c41f3a1b32a` | 159.48111829999834 |
| `q049` | `a89d7624` | `90dd47f8bc92dd3b1ff9c7fe88e6dbce620d6ee07b1949939bf0abe1132f8d57` | 242.81788190000225 |

The existing-artifact path re-read the root and every part, reconstructed the
locked inputs, and returned the same root SHA-256 without loading Qwen,
rebuilding a store, or contacting a provider.

## Nested-stage measurements

| Stage | Selected evidence | Additions |
| --- | ---: | ---: |
| S0 causal/coverage predecessor | 382 total; mean 38.2; range 15--46 | 382 root rows |
| S1 direct episodes | 554 total; mean 55.4; range 24--65 | 172 total; mean 17.2; range 9--20 |
| S2 representative episodes | 558 total | 4, all on `q047` |
| S3 artifact-global closure | 558 total | 0 |

The exact aggregate status counts are `root=10`, `added=11`, and
`budget_exhausted=19`. All ten S1 stages were `added`; q047's S2 was also
`added`, while the other nine S2 stages and all ten S3 stages were
`budget_exhausted`.

| Final-stage budget measure | Sum | Mean | Range |
| --- | ---: | ---: | ---: |
| Prompt-token proxy | 71,437 | 7,143.7 | 5,955--7,319 |
| Context-token proxy | 68,154 | 6,815.4 | 5,630--6,994 |
| Prompt workspace | 73,997 | 7,399.7 | 6,211--7,575 |

All stages retained the exact 8,000 prompt, 7,000 context, and 256 responder
reserve. The independent audit recomputed the S0--S3 seals, immediate-parent
hashes, exact prefix nesting, evidence-coordinate suffixes, question and
prompt hashes and token counts, plus final, predecessor, and ladder bindings.

The sealed elapsed values total `1805.946894200024` seconds, with mean
`180.5946894200024`, median `156.9032070999965`, and range
`134.96860480000032`--`275.79288630001247` seconds. These are question-level
retrieval timings and do not include combined-store build time.

## The sealed q047 S2 transition

For q047 (`b6019101`), S1 selected 24 evidence rows and S2 appended four more.
The transition preserved exact prefix order and changed the sealed counts and
token proxies as follows:

| Measure | S1 | S2 | Change |
| --- | ---: | ---: | ---: |
| Evidence rows | 24 | 28 | +4 |
| Context-token proxy | 4,182 | 5,630 | +1,448 |
| Prompt-token proxy | 4,507 | 5,955 | +1,448 |
| Prompt workspace including reserve | 4,763 | 6,211 | +1,448 |

The S2 stage receipt is
`e03d4cffebc7d1a7555ec9e59054232dc86a64ae17721cf0afaac52d68743c4e`.
It binds directly to the S1 receipt
`f3bf496354c31ff876f76a4975d22b6c8f10110c84e4db051b5bccf128958f49`,
and the following S3 receipt binds back to S2 while admitting no new row.

| Added evidence ID | Sealed source coordinate |
| --- | --- |
| `1dd390dcc1cfb246607c81c4c7203972832a3d467ec94c2a97f5cc5c8c4f7d38` | `94f70d80::b8c5e928_1` |
| `aac81af986e8713f7f451ab7df05d4249dbcd9b22cbd700e17accfd94c9690bd` | `94f70d80::b8c5e928_1` |
| `74a6701a379dc140c33e4df7becb2cee7268dcf93248c27d66f02365e834bc70` | `b6019101::2ed7c45e_1` |
| `19f0699e58714e25477d925ac61c9e5896f5181f7cf04655d58167b1a63fb4a3` | `b6019101::2ed7c45e_1` |

This establishes that the representative-episode stage appended four rows
within the locked budget. It does not establish their relevance, whether a
responder would use them, whether the answer changes, or whether accuracy
improves. Those questions require the still-unrun answer and judge stages.

## Timestamp-derived combined-build lifecycle

Local filesystem timestamps give a non-sealed operational bound for the
offset-40 combined build. The combined-store directory appeared at
13:56:25.8559168 and its canonical manifest was published at 14:46:35.8645764,
an observable interval of 3,010.008660 seconds, or 50.166811 minutes. This is
filesystem metadata, not a receipt-bound duration, and it may include small
orchestration overhead around the build.

The combined HNSW file finished writing at 14:14:34.9520083. Manifest
publication followed 1,920.912568 seconds, or 32.015209 minutes, later; the
database's final write at 14:46:33.0163325 preceded the manifest by 2.848244
seconds. These file times are phase proxies, not proof of exact internal phase
boundaries.

The post-index interval remains consistent with the compiler's cumulative
snapshot behavior. Each of 481 source-stream receipts can publish episode and
discourse state separately, and each publication canonically snapshots the
source rows and graph accumulated so far inside its transaction. Aggregate
work is therefore approximately proportional to
`source streams x accumulated corpus/graph rows`; rule linking can add
quadratic work within unusually long streams. This operational timing makes no
retrieval-quality or failure claim.

## Source and combined-store audit

| Source identity | SHA-256 |
| --- | --- |
| Selection file | `865f3855c3f2c2605bce9f266f2dafe4b9e3569b2d46ffd8d8b3f19f0e634205` |
| Source receipt | `c527d21d2827e898b0e82af94f4e911f8ea8332ad5e8e38903e36d37cefea9c3` |
| Corpus | `8858f1cfcd62f64f3b8ed69b7d8df087dd209e2a001103a087fb42c376f36e77` |
| Database | `d356a1dcd2fec29292abb07dd6c6e8fc9dd4c32c0b8d7f228e3b8c4b311e3324` |
| Index | `d6c6935afc649f2ad34aa9351564dbf1f24e5ad6fc5e179586e5e1aa6afdaf18` |
| Store manifest | `b5460d627e34ccad985ce024127d222f829c5322838b7de889308c85758bf34b` |

| Combined identity | SHA-256 |
| --- | --- |
| Manifest file | `9b7efe72584af352fd19eb3f9f413cf3bd33b51e705510aab6255224e7303929` |
| Combined-store receipt | `f5882f14a1b0096614ab2143fdcbe85ed8cb8eeaa0ce0cf3190693a677a70257` |
| Compilation receipt | `f49dcb5afcc606612fd406ed1511a24a37f9edce45cdfb2fd55e55adb8966624` |
| Target database | `3e0556171cbee6955a92519139431b8aa9eb790189dcfcb45c6454403da394c2` |
| Target index | `3a5e8c22022738818329177d223b32b2fa3b964744cb3365499787316d0d558c` |
| Source/target store identity | `9dd5e14900c54a714aa77f2047b043e6cddc67ea3cf69381ed84dab02b5a9434` |
| Snapshot | `ca6685d75f8263a26070f08ae7f292644a30e9c4ffa0fbad082825624e7fe242` |

The combined store records 2,426 causal events, 57,202 edges, 7,391 graph
nodes, and 481 compilation source receipts.

The audit enumerated 12 provider-call fields and 506 request-token or
transformer-state fields; every value was zero. It found no gold-bearing field
and no provider request, response, or journal file.

## Current campaign state

Offsets 0, 10, 20, 30, and 40 are sealed, replay-verified, and independently
audited. Offset 50 has entered the sequential GPU workflow and is still in
progress; no offset-50 root or metrics are claimed here. Offsets 60--90 remain
preflighted but unrun.

The campaign still lacks its exact ten-shard/100-question merge. The
preregistered fixed-S1 Terra responder, independent Sol judge, >=95% gate, and
matched Mem0 production arm therefore remain unresolved and are not reported
as complete.
