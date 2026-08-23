# Locked validation offset 50 seal

**Status:** validation offset 50 is the sixth of ten sealed provider-free
retrieval shards in the locked 100-question campaign. Its canonical root and
ten question parts passed verify-only replay and a separate read-only
artifact/store audit. S1 appended 167 evidence rows; S2 and S3 appended none
because the frozen budget was exhausted. Offset 60 is now running. The
ten-shard merge, fixed-S1 Terra answer artifact, independent Sol score, >=95%
gate, and production-bound Mem0 comparison remain unresolved.

The longest sealed question timing was `q058` at about 595.4 seconds. That is
an elapsed-time outlier only: it is not evidence about causality, retrieval
difficulty, relevance, answerability, or answer quality. No responder, judge,
or gold-bearing score ran in this shard, and the preregistered answer-stage
treatment remains fixed at S1.

## Root and question-part identities

The canonical artifact is
[`shards/offset-050/retrieval.json`](../../eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822/shards/offset-050/retrieval.json).

| Property | Observed value |
| --- | --- |
| Root file SHA-256 | `44eda01cdb80ecfc5604789c2a089631ed2fd1a6c7a395807aecf3d0d8ae4942` |
| Root file size | 2,392,762 bytes |
| Gold-blind 100Q population | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| Offset-50 shard | `ec4ca25ad97277c846b75e1628c5765ecb5b8dcaffa26bf5eeddd867e40a8f2e` |
| Retrieval implementation | `cf2577f21a7a1af1b9c5f331c7eb1672c5ba13af84ccdc2f718259192bb36e09` |
| Effective retrieval policy | `0abb42db8fd35b566029be135630af34fa42ee9a464ef8c4a889948729cbab13` |
| Environment lock | `c99f82aa37a1df009c6100110d24aeaee0bde3376ff09537700ad9848b371a95` |
| Validation policy manifest | `5263d5afd15298ec4088db9d6381ae243ddb685e9a3cf4d9892fc84e14fb9883` |
| Validation attestation | `3c054b584fa4ca7dff7e1d97d4bc532bb3e98b140b15b3fa6c35cd67be971558` |
| Validation execution policy | `c6b3d6f3f511d0a52b271e0880cb93c3866e649b614cfa4b5ac0604b704fdc8d` |
| Transcript-token proxies | 1,051,365 |
| Turns / questions | 5,403 / 10 |

Every part is canonical JSON with an exact sidecar and an exact ordered match
to the root's `question_part_sha256s` entry and embedded question object:

| Part | Question ID | SHA-256 | Sealed elapsed seconds |
| --- | --- | --- | ---: |
| `q050` | `gpt4_5501fe77` | `31c1031645a9fd65660b4218b617e19e6a23e729e00186acc2face021fd617dc` | 141.67988549999427 |
| `q051` | `41698283` | `e84e38fdf1abd347a5c4aa944e96c731b3a66eb5a99a9c4aaca90312f12b0a8b` | 109.05904570000712 |
| `q052` | `3d86fd0a` | `bcd212a8fdebf23a5d60ff634115bf0be6b4907bce45f3c44827c7c4b5557974` | 138.47674430000188 |
| `q053` | `3a704032` | `bc68f7a8bf9d6c79e6e090a04c74ec6cc856b0af65092689d46704dc86705b01` | 153.57761939999182 |
| `q054` | `gpt4_8279ba03` | `a3534dbacf420cf39547b62336c088acb7a6c0c2eb6a56a47119c230f442cac0` | 324.27095970000664 |
| `q055` | `d52b4f67` | `1401e0780402bbccf117c15bf38ed4ba0b6d8c56386834b3961cf09bf441f139` | 196.3215909999999 |
| `q056` | `gpt4_1e4a8aeb` | `42c897bddebd9e04d19d694a1ada6ccb10992a814d9e50bfdd7db804fdd8a944` | 91.68753410001227 |
| `q057` | `6071bd76` | `72933618a6a5077a8dab109df354e2f7944aa594afe9dc987abfd1ac5f2b2dba` | 147.48572630000126 |
| `q058` | `b9cfe692` | `e1dc5b1e0ee67b830818366eb12c915f536eff26df3f10669d9592edd3e55c14` | 595.4114251999999 |
| `q059` | `b29f3365` | `4cc8f55082c4df873b09a9192203210064fa0871248e8c18a0b053344acb16da` | 170.5188494000031 |

The existing-artifact path re-read the root and every part, reconstructed the
locked inputs, and returned the same root SHA-256 without loading Qwen,
rebuilding a store, or contacting a provider.

## Nested-stage measurements

| Stage | Selected evidence | Additions |
| --- | ---: | ---: |
| S0 causal/coverage predecessor | 367 total; mean 36.7; range 20--46 | 367 root rows |
| S1 direct episodes | 534 total; mean 53.4; range 35--64 | 167 total; mean 16.7; range 15--19 |
| S2 representative episodes | 534 total | 0 |
| S3 artifact-global closure | 534 total | 0 |

The exact aggregate status counts are `root=10`, `added=10`, and
`budget_exhausted=20`. All ten S1 stages were `added`; every S2 and S3 stage
was `budget_exhausted`. The S2 and S3 stages therefore preserve S1's
provider-visible evidence and prompt exactly and provide no additional
evidence admission to evaluate.

| Final-stage budget measure | Sum | Mean | Range |
| --- | ---: | ---: | ---: |
| Prompt-token proxy | 72,681 | 7,268.1 | 7,171--7,314 |
| Context-token proxy | 69,418 | 6,941.8 | 6,849--6,994 |
| Prompt workspace | 75,241 | 7,524.1 | 7,427--7,570 |

All stages retained the exact 8,000 prompt, 7,000 context, and 256 responder
reserve. The independent audit recomputed the S0--S3 seals, immediate-parent
hashes, exact evidence-row prefixes and addition suffixes, protected source
coordinates, question and prompt hashes and token counts, plus final,
predecessor, and ladder bindings.

The sealed elapsed values total `2068.489380600018` seconds, with mean
`206.8489380600018`, median `150.53167284999654`, and range
`91.68753410001227`--`595.4114251999999` seconds. These are question-level
retrieval timings and do not include source or combined-store build time.

## q058 is a timing observation only

The sealed `q058` elapsed value is `595.4114251999999` seconds, or
`28.784843%` of the ten-question retrieval total. The other nine questions
averaged `163.6753284` seconds. q058 selected 39 S0 rows, appended 16 S1 rows,
and finished with 55 rows, a 6,923-token context proxy, and a 7,260-token
prompt proxy; like every other question in this shard, its S2 and S3 stages
were budget-exhausted no-ops.

Those sealed fields establish the elapsed distribution and final packet shape
only. They contain no runtime trace that assigns the delay to paging, storage,
model execution, a particular retrieval method, or any other cause. They also
do not measure evidence relevance or answer accuracy. Treating q058 as a
quality failure or success would exceed the artifact.

## Timestamp-derived build lifecycle

Local filesystem timestamps provide non-sealed operational bounds. The
`source-current` directory appeared at `15:19:32.6053836`, and the canonical
selection appeared at `15:27:44.6801106`, an observed interval of
492.074727 seconds, or 8.201245 minutes.

The combined-store directory appeared at `15:28:01.9186653`; its canonical
manifest appeared at `16:18:27.7964331`. That interval was
3,025.8777678 seconds, or 50.431296 minutes. The HNSW file's timestamp was
`15:47:33.0240359`, 1,854.7723972 seconds, or 30.912873 minutes, before
manifest publication. The database's final timestamp at
`16:18:24.8885739` preceded the manifest by 2.9078592 seconds. The retrieval
root appeared at `16:58:31.6486409`.

These timestamps are filesystem metadata, not receipt-bound durations, and
may include orchestration overhead. The post-index interval is structurally
consistent with the compiler publishing cumulative episode/discourse state
and snapshots across 480 source-stream receipts, but it does not prove a
specific performance cause. Likewise, the q058 sealed timing cannot be
causally attributed from these shard-level file timestamps.

## Source and combined-store audit

| Source identity | SHA-256 |
| --- | --- |
| Selection file | `6d98ca7326dc0a475ce531d6c33566c384c347940d2eaa10f3282d06d8ce129a` |
| Source receipt | `32984050c6e34771293964152346de1f1079ebca4ba7bdc0a93049ce3d65a80f` |
| Corpus | `a3bc6f29687dd69c3c3f5cc23661a9a550ad44c7cc1f78765ae558bd2dcf9311` |
| Database | `76918c704af089a0374437fbeaeecbbf904499d5867d29a7cc23956906fde822` |
| Index | `28ab678fefcacf9d3fb33792b30c9019ed1f2a20d87dbb879ae0be43adf95355` |
| Store manifest | `ae20bfb8a3330b7ab7053ae3d3bfd9aaf241100c44301bd6e70637e21cb95b85` |
| Query manifest | `36b484e6f099573bb693075a67a9a7d69ae767570e67e984b0a0003f46adeaab` |

| Combined identity | SHA-256 |
| --- | --- |
| Manifest file | `b8cf1e370047b5e54d92239599b03988f51467562b9879260a23541b6062a674` |
| Combined-store receipt | `f91db80398c5e3e70a866087a00f7ebf72f277f61bb794305fb649c39f8e5a71` |
| Compilation receipt | `5da04fc260b0c2d71c821dfcf087274ba3584c2c75bd27d3fdff7fe3c192fd4e` |
| Target database | `16b571c65ff4e6bc60cb73da4f7a8180ff61fde64be1d0ba682cca783a8d2f81` |
| Target index | `73f4e276f7ff3a075ca8dc40abaf0fe3681f48dd8d26d15a71ca7c85dfabd319` |
| Source/target store identity | `b5de9602d8fa9035cb368b7518eeaf26ec0a06877108d07b0367040a1519eca9` |
| Snapshot | `1dd05816e8a834be05fbe0ae033d126a782d29761d1aeb8082c0a4e13aa1ddd4` |

Both source and combined databases contained 5,403 turns and 7,984 chunks
under read-only SQLite counts. The combined store records 2,379 causal events,
56,250 edges, 7,304 graph nodes, 480 compilation source receipts, and 4,774
bound outcome chunks. It records zero insufficient-candidate skips and 61
large-prompt skips.

The audit checked canonical root, selection, question-part, and combined
manifest bytes; every available sidecar; all receipt self-hashes and physical
manifest/database/index hashes; and every receipt chain back to the selected
source. It enumerated 12 provider-call fields and 505 request-token,
transformer-state, or retained-prompt-state fields; every value was zero. It
found no gold-bearing field and no provider request, response, or journal file.

## Current campaign state

Offsets 0, 10, 20, 30, 40, and 50 are sealed, verify-only replayed, and
independently audited. Offset 60 is in the sequential GPU workflow; no
offset-60 root or metrics are claimed here. Offsets 70--90 remain preflighted
but unrun.

The campaign still lacks its exact ten-shard/100-question merge. The
preregistered fixed-S1 Terra responder has not run on the validation
population, so the independent Sol judge, >=95% gate, and matched Mem0
production arm also remain unresolved.
