# Locked validation offset 30 S2 seal

**Status:** validation offset 30 is the fourth of ten sealed provider-free
retrieval shards in the locked 100-question campaign. Its canonical root and
ten question parts passed verify-only replay and a separate read-only
artifact/store audit. Offset 40 is now running. The ten-shard merge, fixed-S1
Terra answer artifact, independent Sol score, >=95% gate, and production-bound
Mem0 comparison remain unresolved.

After S2 was budget-inert throughout offsets 10 and 20, this shard records a
fully sealed S2 admission: four rows on `q035`. That is a verified retrieval
transition, not a measured answer-quality gain. No responder, judge, or
gold-bearing score ran here, and the preregistered answer-stage treatment
remains fixed at S1. Offset 0 had already admitted five S2 rows across two
questions, so q035 is not presented as the campaign's first S2 addition.

## Root and question-part identities

The canonical artifact is
[`shards/offset-030/retrieval.json`](../../eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822/shards/offset-030/retrieval.json).

| Property | Observed value |
| --- | --- |
| Root file SHA-256 | `14e884527a54eeb0b23f17b82236b8f5670e17ae234b387ee8cc2e7c8fbb3aed` |
| Root file size | 2,330,873 bytes |
| Gold-blind 100Q population | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| Offset-30 shard | `4dbe74c5c48e13229103b10c257dc447149e0300f826a7e5bb1473a18339f2da` |
| Retrieval implementation | `cf2577f21a7a1af1b9c5f331c7eb1672c5ba13af84ccdc2f718259192bb36e09` |
| Effective retrieval policy | `0abb42db8fd35b566029be135630af34fa42ee9a464ef8c4a889948729cbab13` |
| Environment lock | `c99f82aa37a1df009c6100110d24aeaee0bde3376ff09537700ad9848b371a95` |
| Transcript-token proxies | 1,043,571 |
| Turns / questions | 5,514 / 10 |

Every part is canonical JSON with an exact sidecar and an exact ordered match
to the root's `question_part_sha256s` entry and embedded question object:

| Part | Question ID | SHA-256 | Sealed elapsed seconds |
| --- | --- | --- | ---: |
| `q030` | `4dfccbf7` | `8c30ad725f5ff534c1d9f6408df4e27e2e0571eb4cd75fd1bd47a7581c60a3d7` | 123.96560250000039 |
| `q031` | `bc149d6b` | `1cd885bd66adeeaf6bebb6aae0b535c935763228b7d3892e749d069b326a5ae0` | 260.70633230000385 |
| `q032` | `61f8c8f8` | `93d3053f86b8fce74b792eecb7285001d8a7b67d380dbf5f7dea188915fd02b5` | 284.5570411999943 |
| `q033` | `4adc0475` | `30eb7c13cc68a1645fb5b81aabda4ad11ee6798556331972cd64109d2363c9a6` | 147.86957019999682 |
| `q034` | `d682f1a2` | `05878003c5a34e3bacf435af0155de0283e6046c907455a0f3555c92e0eed315` | 171.11204670000006 |
| `q035` | `157a136e` | `ed58f4ae50247584254c072651d7f7841c999ebd514c6b76f4a1333353cf1547` | 346.83375449999585 |
| `q036` | `32260d93` | `9e39dc46e33b05d9b11474e64e0305f199144a2632aa752617cbd954ebfddeaa` | 158.16461510000227 |
| `q037` | `bc8a6e93` | `31cea9554cc98aa4ae1c4f7d96dd76a7705df2c41ef61b5616815336d62f89f4` | 155.61142929999914 |
| `q038` | `7161e7e2` | `13ee57c7ac37c499a2015a574fdc9bbcde044cf0cbe6a603c585c7049aed75d9` | 167.9114247999969 |
| `q039` | `ce6d2d27` | `cbbb22ff977e0794beaa6c7a3f0b9e8f5e76300acc1fef0fa30fd088ab48f764` | 147.6950247000059 |

The existing-artifact path re-read the root and every part, reconstructed the
locked inputs, and returned the same root SHA-256 without loading Qwen,
rebuilding a store, or contacting a provider.

## Nested-stage measurements

| Stage | Selected evidence | Additions |
| --- | ---: | ---: |
| S0 causal/coverage predecessor | 354 total; mean 35.4; range 16--45 | 354 root rows |
| S1 direct episodes | 517 total; mean 51.7; range 25--63 | 163 total; mean 16.3; range 9--22 |
| S2 representative episodes | 521 total | 4, all on `q035` |
| S3 artifact-global closure | 521 total | 0 |

The exact aggregate status counts are `root=10`, `added=11`, and
`budget_exhausted=19`. All ten S1 stages were `added`; q035's S2 was also
`added`, while the other nine S2 stages and all ten S3 stages were
`budget_exhausted`.

| Final-stage budget measure | Sum | Mean | Range |
| --- | ---: | ---: | ---: |
| Prompt-token proxy | 71,825 | 7,182.5 | 6,218--7,341 |
| Context-token proxy | 68,541 | 6,854.1 | 5,897--6,998 |
| Prompt workspace | 74,385 | 7,438.5 | 6,474--7,597 |

All stages retained the exact 8,000 prompt, 7,000 context, and 256 responder
reserve. The independent audit recomputed the S0--S3 seals, immediate-parent
hashes, exact prefix nesting, evidence-coordinate suffixes, question and
prompt hashes and token counts, plus final, predecessor, and ladder bindings.

The sealed elapsed values total `1964.4268412999954` seconds, with mean
`196.44268412999955`, median `163.03801994999958`, and range
`123.96560250000039`--`346.83375449999585` seconds. These are question-level
retrieval timings and do not include combined-store build time.

## The sealed q035 S2 transition

For q035 (`157a136e`), S1 selected 25 evidence rows and S2 appended four more.
The transition preserved exact prefix order and changed the sealed counts and
token proxies as follows:

| Measure | S1 | S2 | Change |
| --- | ---: | ---: | ---: |
| Evidence rows | 25 | 29 | +4 |
| Context-token proxy | 4,422 | 5,897 | +1,475 |
| Prompt-token proxy | 4,743 | 6,218 | +1,475 |
| Prompt workspace including reserve | 4,999 | 6,474 | +1,475 |

The S2 stage receipt is
`2a07d2b34a68c78439ad0526a43f820ef37e11a9e7c1d1cd30af2e4e4a653757`.
It binds directly to the S1 receipt
`21f07e273bf0323f4f9df88c1973d8090237c13508d63fccc15c5eff00c327b1`,
and the following S3 receipt binds back to S2 while admitting no new row.

| Added evidence ID | Sealed source coordinate |
| --- | --- |
| `c807e3ef8c656c7255ced5540c4d1eb69b98b867a347bc8590bddd7ca1594cc0` | `7161e7e2::ultrachat_490889` |
| `906269f91e5e3f815a3008b5ae5f5998e36772d61483921749cbf3bec2ec419d` | `7161e7e2::sharegpt_kovlLyh_0` |
| `57c620028ddaaf1f44540ae2016322af8a59d1341a7f3f8d29ed5c273d733d03` | `7161e7e2::sharegpt_kovlLyh_0` |
| `9d3af9f8d7f9470e27ac29fd1760a7e5393ec4aec7270b91d42e65cff23507d6` | `ce6d2d27::ultrachat_342964` |

This establishes that the representative-episode stage can contribute after
S1 when budget remains. It does not establish that these rows are relevant,
that they change the answer, or that they improve answer accuracy. Those
questions require the still-unrun answer and judge stages.

## Combined-build lifecycle and structural tail

Filesystem timestamps provide a cautious operational wall-time bound for the
offset-30 combined build. The combined-store directory appeared at
11:53:40.1613879 and its canonical manifest was published at 13:08:04.4658343,
an observable interval of 4,464.304446 seconds, or 74.405074 minutes. This is
longer than the earlier roughly 53--60 minute builds, but it is not a sealed
duration field and may include small orchestration overhead around the build.

The combined HNSW file finished writing at 12:34:47.9062971. Manifest
publication followed 1,996.559537 seconds, or 33.275992 minutes, later; the
database's final write at 13:08:01.5751066 preceded the manifest by 2.890728
seconds. These file times are phase proxies, not proof of exact internal phase
boundaries.

The long post-index tail remains consistent with the compiler's cumulative
snapshot behavior. Each of 473 source-stream receipts can publish episode and
discourse state separately, and each publication canonically snapshots the
source rows and graph accumulated so far inside its transaction. Aggregate
work is therefore approximately proportional to
`source streams x accumulated corpus/graph rows`; rule linking can add
quadratic work within unusually long streams. The timestamps show a longer
run, not a new sealed retrieval-quality or failure claim.

## Source and combined-store audit

| Source identity | SHA-256 |
| --- | --- |
| Selection file | `5ec4d39dc020548d0176baba11584c1a640f7c053c607235c766f49294887eaa` |
| Source receipt | `f798177a6709de65d9b14166987b5effb300f2e0717af76500b1f989163a9faa` |
| Corpus | `ad0abbf1e1f591dbd43d827025d326d81c7369cbb9e03892fbfd34ab72448f5f` |
| Database | `868eb285da400bfb015f6e586c08a19a712b7b713a97bae1f2207a768eb17cb2` |
| Index | `1a019db62677c7bb9ff72393f71b96d18323654c02d65f76229618c23d8c8078` |
| Store manifest | `a90eda2a8369bfc65cd8382e9116677489a3212da801689cce3d2802b84e1165` |

| Combined identity | SHA-256 |
| --- | --- |
| Manifest file | `fac5669b40b0d172eddc92bf49772f9992af641bf733fe7867214527a859b7cf` |
| Combined-store receipt | `710bfb934a2899eef78785355f0b07d035e97334763e037b9d735f978831de52` |
| Compilation receipt | `eb080fc263a08c723945fe1cc014b8594f49e45a5885821e509a785a5a4436a7` |
| Target database | `750ad6ea33c9398627b9b00aff6b740db007dc7972ad23ad49a016dc0a8bd2d9` |
| Target index | `bfd5074acc7ade13ee2966ed667fdfabd7ab4704a6db313157656b626494fdbd` |
| Source/target store identity | `7cf1bcec2d374ecef7120ae96a5184c2621167cbe9618da9d069209a490bd1a9` |
| Snapshot | `6c84be70129407c496dcb75cdadcc0745c4f78295446fc3541bf254f6d6c7d3c` |

The combined store records 2,455 causal events, 57,123 edges, 7,445 graph
nodes, and 473 compilation source receipts.

The audit enumerated 12 provider-call fields and 498 request-token or
transformer-state fields; every value was zero. It found no gold-bearing field
and no provider request, response, or journal file.

## Current campaign state

Offsets 0, 10, 20, and 30 are sealed, replay-verified, and independently
audited. Offset 40 has entered the sequential GPU workflow and is still in
progress; no offset-40 root or metrics are claimed here. Offsets 50--90 remain
preflighted but unrun.

The campaign still lacks its exact ten-shard/100-question merge. The
preregistered fixed-S1 Terra responder, independent Sol judge, >=95% gate, and
matched Mem0 production arm therefore remain unresolved and are not reported
as complete.
