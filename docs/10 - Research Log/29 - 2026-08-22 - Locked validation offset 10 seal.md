# Locked validation offset 10 seal

**Status:** validation offset 10 is the second of ten sealed provider-free
retrieval shards in the locked 100-question campaign. Its canonical root and
ten question parts passed the built-in verify-only replay and a separate
read-only artifact/store audit. Offset 20 is now running. The 100-question
merge, Terra answer artifact, Sol score, >=95% gate, and production-bound Mem0
comparison remain unresolved.

This is a retrieval result only. It does not measure answer accuracy, and the
two completed shards must not be reported as a 20-question answer result.

## Root and question-part identities

The canonical artifact is
[`shards/offset-010/retrieval.json`](../../eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822/shards/offset-010/retrieval.json).

| Property | Observed value |
| --- | --- |
| Root file SHA-256 | `230d36bd65de5bea6f05d9f00203ed6ba1da6ab86df565752389325c0b6f7b6d` |
| Root file size | 2,415,329 bytes |
| Gold-blind 100Q population | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| Offset-10 shard | `6cfcdb0d0c7bf24c56eeb72ed90417080f5345c5e8c6b377974476425fc93be7` |
| Retrieval implementation | `cf2577f21a7a1af1b9c5f331c7eb1672c5ba13af84ccdc2f718259192bb36e09` |
| Effective retrieval policy | `0abb42db8fd35b566029be135630af34fa42ee9a464ef8c4a889948729cbab13` |
| Environment lock | `c99f82aa37a1df009c6100110d24aeaee0bde3376ff09537700ad9848b371a95` |
| Transcript-token proxies | 1,044,341 |
| Turns / questions | 5,241 / 10 |

Every part is canonical JSON with an exact sidecar and an exact ordered match
to both the root's `question_part_sha256s` entry and embedded question object:

| Part | Question ID | SHA-256 | Elapsed seconds |
| --- | --- | --- | ---: |
| `q010` | `4c36ccef` | `7ac124f3633d0a08a1319fb904f86e68d1179545390080c4e4e389e8cb410c81` | 465.1848605999985 |
| `q011` | `2698e78f_abs` | `550bc7f03ca06800f3466ae65424e968ac64df3e5fb37c9c13c543899012bc94` | 181.55748719999974 |
| `q012` | `0ddfec37` | `5421b06ad3f2c1679eb62cc75b49eab6830ab7a3f31e908bdaa557671a831222` | 308.38689370000793 |
| `q013` | `f685340e_abs` | `8fce61e23efcbb9a96745d006c2988c03afbd3356bd2758a1d416bb7ecff3133` | 144.2946423000103 |
| `q014` | `d23cf73b` | `62a9af451765d932e83ec4a141902218043f65b045d358b323ce52dbe9fec253` | 266.87714229999983 |
| `q015` | `15745da0` | `fb170deb88ff90b54d9cee9744e89f2d483a011930472dc371cc1746df78919d` | 149.931574000002 |
| `q016` | `2133c1b5` | `d4e62fe9c76e40effbb781889ec1735a44821bd751b8b2af7f6501f7d34c454e` | 144.66615410000668 |
| `q017` | `gpt4_65aabe59` | `816606a064b30c72d8639acd47a50a31d51be7cc2b124ea847aa4bf497e9c072` | 143.45197939999343 |
| `q018` | `cc06de0d` | `8b94a39c9b054e343bb6d99f18710388123b925f3be5030dd55bd2d585d81b71` | 123.14957649999997 |
| `q019` | `3ba21379` | `9169c4403d92d55cea285dda29c8e1b12d641b8a44d672acd5a99492be1cd9a6` | 184.56189849998918 |

The built-in existing-artifact path re-read the root and every part, rebuilt
the locked input identities, and returned the same root SHA-256 without
loading a Qwen model, rebuilding a store, or contacting a provider.

## Nested-stage measurements

All ten questions preserve the exact ordered S0 -> S1 -> S2 -> S3 prefix
ladder. The independent audit recomputed every stage seal, immediate-parent
binding, evidence coordinate, prompt hash and token count, predecessor,
ladder, and final-retrieval receipt.

| Stage | Admission statuses | Selected evidence | Additions |
| --- | --- | ---: | ---: |
| S0 causal/coverage predecessor | `root=10` | 411 total; mean 41.1; range 28--50 | 411 root rows |
| S1 direct episodes | `added=10` | 588 total; mean 58.8; range 44--68 | 177 total; mean 17.7; range 15--21 |
| S2 representative episodes | `budget_exhausted=10` | 588 | 0 |
| S3 artifact-global closure | `budget_exhausted=10` | 588 | 0 |

The exact aggregate status counts are `root=10`, `added=10`, and
`budget_exhausted=20`.

S2 and S3 were inert on every question because S1 consumed the available
context budget: neither later method admitted an evidence coordinate. This is
not a claim that the methods found no candidates or are generally useless; it
is the measured outcome of the frozen additive ladder under this cap. Offset
10 therefore demonstrates a structural S0 -> S1 evidence increase but no
incremental S1 -> S2 or S2 -> S3 contribution.

| Final-stage budget measure | Sum | Mean | Range |
| --- | ---: | ---: | ---: |
| Prompt-token proxy | 72,586 | 7,258.6 | 6,962--7,326 |
| Context-token proxy | 69,323 | 6,932.3 | 6,631--7,000 |
| Prompt workspace | 75,146 | 7,514.6 | 7,218--7,582 |

Every stage respected the exact 8,000 prompt, 7,000 context, and 256 responder
reserve. Question `q019` reached exactly 7,000 context-token proxies but did
not exceed the cap.

The sealed per-question elapsed values total `2112.0622086000076` seconds,
with mean `211.20622086000077`, median `165.74453060000087`, and range
`123.14957649999997`--`465.1848605999985` seconds. These are retrieval-call
measurements only; they exclude source and combined-store build wall time.

## Source and combined-store audit

The independent audit checked the physical source and combined files against
their sealed receipts:

| Source identity | SHA-256 |
| --- | --- |
| Selection file | `08080c1430aa8bc8fc4309ac825b1221ec02af4d5d7909ff0819a93e4e91e9e2` |
| Source receipt | `aafa113789cab0ee1969c679bcce616f1646c00d5a82e52a8e596ee6515eb92f` |
| Corpus | `f9551b993c6ab6029928898b65dc617b21cba4305d953ff140a19b0977b41adb` |
| Database | `6c5e0dde2da6356b5b6a2032f3d439cbfeedf9fdb4ff2a021abdc7b28d1fee72` |
| Index | `9f600fb23a3a31847094f0ff5a73292b85f047c4c78c1a9fe54e3f8f98e5a904` |
| Store manifest | `065c9adab1e72f074f801af8416230663078387ffee1e847c31b52e02dd8af25` |

| Combined identity | SHA-256 |
| --- | --- |
| Manifest file | `541f6190a5d5462d343d17e5a9c59af2d0cd49b2a4e07a21626431c61fc62cde` |
| Combined-store receipt | `99224b5505fc1f925a6d75667d19f1e25122334d7f701ef0b1da67ae2cc504df` |
| Compilation receipt | `c68c61a22fc6389173e4901d92b198826ba4a1b5bb6aafefb522bbb7920222c7` |
| Target database | `3537c4ca9acec7c1c93ff36ce948bb65bf3b2f84db96078255104ee98108c52c` |
| Target index | `533ec90fd953fc9fe1919ea6d3984bedfe48fb8210ef9c61419f43da0bc68b40` |
| Source/target store identity | `d9b3643fd10bdd94ed3753ad85dfdb479949ccb93f16711c4332fe2139a1b142` |
| Snapshot | `2e95bd5622f2187984041b09d203ae5f3e8779b052c760f13db90936466dd7be` |

The combined store records 2,306 causal events, 54,683 edges, 7,195 graph
nodes, and 464 compilation source receipts.

## Gold, provider, and retained-state boundary

The audit enumerated 12 provider-call fields and 489 request-token or
transformer-state fields; every value was zero. It found no gold-bearing
field and no provider request, response, or journal file. The shard is fully
provider-free and retains zero request-derived transformer-token state.

These checks establish artifact integrity, strict nesting, and budget
compliance for offset 10. They do not establish source-label recall, semantic
answer accuracy, provider persistence, or cross-system fairness.

## Current campaign state

Offsets 0 and 10 are now sealed and independently audited. Offset 20 has
entered the sequential GPU workflow and is still in progress; no offset-20
root or measurements are claimed here. Offsets 30--90 remain preflighted but
unrun.

The campaign still lacks the exact ten-shard/100-question merge. Consequently
the preregistered fixed-S1 Terra responder, independent Sol judge, >=95% gate,
and matched Mem0 production arm cannot yet produce formal results. None is
reported as complete.
