# Locked validation offset 20 seal

**Status:** validation offset 20 is the third of ten sealed provider-free
retrieval shards in the locked 100-question campaign. Its canonical root and
ten question parts passed verify-only replay and a separate read-only
artifact/store audit. Offset 30 is now running. The ten-shard merge, Terra
answer artifact, Sol score, >=95% gate, and production-bound Mem0 comparison
remain unresolved.

This entry separates three kinds of evidence: immutable retrieval timings in
the shard, live operating-system observations made while it ran, and an
implementation-level diagnosis of the combined-store build tail. Only the
first category is sealed into `retrieval.json`.

## Root and question-part identities

The canonical artifact is
[`shards/offset-020/retrieval.json`](../../eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822/shards/offset-020/retrieval.json).

| Property | Observed value |
| --- | --- |
| Root file SHA-256 | `09fda12b968d1f8152a42220d560c96b3763294118149d093507a68f14ce2430` |
| Root file size | 2,368,394 bytes |
| Gold-blind 100Q population | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| Offset-20 shard | `5241842b1c70c518050c12a3e00085b5ae23e5cf321c2801bc9c862885c43473` |
| Retrieval implementation | `cf2577f21a7a1af1b9c5f331c7eb1672c5ba13af84ccdc2f718259192bb36e09` |
| Effective retrieval policy | `0abb42db8fd35b566029be135630af34fa42ee9a464ef8c4a889948729cbab13` |
| Environment lock | `c99f82aa37a1df009c6100110d24aeaee0bde3376ff09537700ad9848b371a95` |
| Transcript-token proxies | 1,045,527 |
| Turns / questions | 5,353 / 10 |

Every part is canonical JSON with an exact sidecar and an exact ordered match
to the root's `question_part_sha256s` entry and embedded question object:

| Part | Question ID | SHA-256 | Sealed elapsed seconds |
| --- | --- | --- | ---: |
| `q020` | `099778bb` | `b0fd0e05f03c3a7e7b2e4359c9782d06e2280a0b6b06aa58de5d9e3e2f9741dc` | 140.97262459999183 |
| `q021` | `gpt4_2f56ae70` | `0459a0b6a57c37ddeeb66aa4682c17ab8cb16c5c97ac513b8fef175960479bf8` | 134.06797869999718 |
| `q022` | `gpt4_7a0daae1` | `b4359da4b55b817394d463a6c611adc9af819286ff94b92aa3ceb7722df02b8a` | 125.61578089999966 |
| `q023` | `2bf43736` | `d939c1d8730db0ec8bfd74e6ca678f7f9e0392b2d63d61e2c1906cb45a84015a` | 289.06178000000364 |
| `q024` | `9bbe84a2` | `5991e9f2997ceff9c27f312bedbcd505312967c3d15eba797ae5a431fe8a7edf` | 150.15896339999745 |
| `q025` | `993da5e2` | `9f0a37ec5a2b6577c8ec45f3ca47270e33a9136a897e608e0d426163cd083181` | 574.6187499999942 |
| `q026` | `352ab8bd` | `cc7ef352ed2e3b52485ee53de9fc896eb6a3e499002160cac85c47fe812f8c01` | 197.65787809999892 |
| `q027` | `6b7dfb22` | `686c1c4f33d7c7a0e36ef442914d1ec6ede256d45b897d7d5918cf1e22dc48dc` | 259.3136383000092 |
| `q028` | `a9f6b44c` | `5ca8fd71530fcd3250b2d298bd8a0901f60a6b38c8d0c8485f32e47900487705` | 157.10650019999593 |
| `q029` | `gpt4_7ddcf75f` | `5cd4052c81198d77f5b53df2d502d90ab45bfaaf771a0994996dc2c5adb91099` | 175.90168129999074 |

The existing-artifact path re-read the root and every part, reconstructed the
locked inputs, and returned the same root SHA-256 without loading Qwen,
rebuilding a store, or contacting a provider.

## Nested-stage measurements

Every question followed the exact status sequence
`S0=root -> S1=added -> S2=budget_exhausted -> S3=budget_exhausted`.

| Stage | Selected evidence | Additions |
| --- | ---: | ---: |
| S0 causal/coverage predecessor | 328 total; mean 32.8; range 16--45 | 328 root rows |
| S1 direct episodes | 506 total; mean 50.6; range 32--65 | 178 total; mean 17.8; range 15--21 |
| S2 representative episodes | 506 | 0 |
| S3 artifact-global closure | 506 | 0 |

The exact aggregate status counts are `root=10`, `added=10`, and
`budget_exhausted=20`. S2 and S3 were inert on all ten questions because the
remaining context headroom after S1 could not admit their proposed additions.
This is a bounded outcome of the frozen cumulative ladder, not evidence that
the later methods are generally incapable of finding candidates.

| Final-stage budget measure | Sum | Mean | Range |
| --- | ---: | ---: | ---: |
| Prompt-token proxy | 72,970 | 7,297.0 | 7,177--7,353 |
| Context-token proxy | 69,636 | 6,963.6 | 6,853--6,998 |
| Prompt workspace | 75,530 | 7,553.0 | 7,433--7,609 |

All stages retained the exact 8,000 prompt, 7,000 context, and 256 responder
reserve. The independent audit recomputed the S0--S3 seals, immediate-parent
hashes, exact prefix nesting, evidence-coordinate suffixes, question and
prompt hashes, prompt-token counts, and final, predecessor, and ladder
receipts.

The sealed elapsed values total `2204.4755754999787` seconds, with mean
`220.44755754999787`, median `166.50409074999334`, and range
`125.61578089999966`--`574.6187499999942` seconds. These are question-level
retrieval timings; they do not include combined-store build time.

## Build-tail and q025 runtime diagnosis

The completed combined-store builds established a normal roughly 53--60
minute envelope: offset 0 finished in 60.28 minutes and offset 10 in 53.06
minutes. Offset 20's continuing database growth, corpus scale, and final
publication were consistent with that envelope; there is no separate sealed
build-duration field to quote as a retrieval metric.

Live lifecycle timestamps placed the offset-20 build phases as follows:

1. causal staging ended with HNSW persistence at 10:27:14;
2. rank learning likely ended near the compiler/WAL reopen at 10:32:33; and
3. episode/discourse compilation continued until combined-store publication.

The long final tail is structural cumulative snapshot hashing, not evidence
of a hung process. Roughly 479 source streams publish episode and discourse
state separately. Each publication creates a canonical snapshot inside its
transaction, rescanning source rows and the graph accumulated so far. The
aggregate work is therefore approximately proportional to
`source streams x accumulated corpus/graph rows`; rule linking can add
quadratic work within unusually long streams. Offset 20 remained between the
earlier shards in corpus scale, had the expected per-chunk HNSW footprint, and
showed continuing database growth.

The global-ordinal-25 part `q025`--printed as question 26/100 by the CLI--is a
separate paging-associated retrieval outlier. Its only sealed runtime fact is
`574.6187499999942` seconds, or 574.6 seconds rounded. Live operating-system
counters for PID 50820 showed:

| Local observation time | Process counters | Interpretation boundary |
| --- | --- | --- |
| 11:16:04 | working set 41.29 GiB; private 49.66 GiB; responsive | live observation during `q025`, not an artifact field |
| 11:25:39 | working set 0.29 GiB; private still 49.66 GiB; responsive | consistent with major working-set trimming/paging |
| 11:29:32 | during `q026`: working set 7.53 GiB; private/paged 27.32 GiB; virtual 116.35 GiB; responsive | post-outlier recovery observation |

These counters make paging a plausible contributor to the `q025` delay, but
they do not prove causality and are not protected by the retrieval receipt.
They must not be presented as sealed model-memory measurements. The immutable
evidence is the per-question elapsed value above; the paging account is
operational diagnosis.

## Source and combined-store audit

| Source identity | SHA-256 |
| --- | --- |
| Selection file | `49d00f44608d9529737218f59733dd77885383e085e9139b70bd54293bbab80b` |
| Source receipt | `90d497d93154e2798ab2f557863d5fbbdcd67cbfa52ca67563dfc397a82a7998` |
| Corpus | `b80d9be379e917eba2349a5101d63a77ebcb988db61736915534bc965b277d3c` |
| Database | `8484fbcd15169becedd3b2ddf43a4a13f62541a0bf8e4d7f8948b07573193c8f` |
| Index | `30419a058fc74b34a0a2046006b433eb47b7cfa740c749ba0f514668c3fe8fee` |
| Store manifest | `8d57089a5fa7452193baad0d075c00b5d765622c7a709e7712bc3ac58858a5f4` |

| Combined identity | SHA-256 |
| --- | --- |
| Manifest file | `20942f470c4cfeb782e6a44bfcbfaa1f71e3c33dbb40e1fbc4a8ef65a9ad840f` |
| Combined-store receipt | `7c7394ef99748225f9185b6322b466ff980d76fe3df81cc454c32ffcc53315ab` |
| Compilation receipt | `8d146613b12b782d394617d87e87d249c2060118587692969c96af50beb0f7dd` |
| Target database | `0f366c7f5846c05a550a17052b4b7aa1ab297cf69769076d44e66036edc789ff` |
| Target index | `288ef22ac5cff30525e419194a8402be622eff3b7035ee786c8a1c31ac358f03` |
| Source/target store identity | `e51d60c01f9a404d46d695a7bef484b61e04e68dc2f31f5c2cb304fc2623389d` |
| Snapshot | `0f1fef39766751dc0099c2f0e0d4c85861af23274a0787690a8acc015f1ec70a` |

The combined store records 2,358 causal events, 55,409 edges, 7,204 graph
nodes, and 479 compilation source receipts.

The audit enumerated 12 provider-call fields and 504 request-token or
transformer-state fields; every value was zero. It found no gold-bearing field
and no provider request, response, or journal file.

## Current campaign state

Offsets 0, 10, and 20 are sealed, replay-verified, and independently audited.
Offset 30 has entered the sequential GPU workflow and is still in progress;
no offset-30 root or metrics are claimed here. Offsets 40--90 remain
preflighted but unrun.

The campaign still lacks its exact ten-shard/100-question merge. The
preregistered fixed-S1 Terra responder, independent Sol judge, >=95% gate, and
matched Mem0 production arm therefore remain unresolved and are not reported
as complete.
