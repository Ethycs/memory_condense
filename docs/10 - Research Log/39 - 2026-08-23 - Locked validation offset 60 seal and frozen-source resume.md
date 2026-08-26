# Locked validation offset 60 seal and frozen-source resume

**Status:** validation offset 60 is the seventh of ten sealed provider-free
retrieval shards in the locked 100-question campaign. All ten question parts
and the canonical root passed existing-artifact replay against the exact
historical implementation and environment identities. S1 appended 173
evidence rows, S2 appended four rows on `q066`, and S3 appended none. Offset
70 passed provider-free preflight under the same identities and is running.
The ten-shard merge, fixed-S1 Terra answers, independent Sol score, `>=95%`
gate, and fair Mem0 comparison remain incomplete.

This resume also closes a checkpoint-operability problem. The checkpoint
receipts correctly rejected the changed live package-wide source hash. An
isolated source snapshot reconstructed from the exact historical Git tree
reproduced the sealed implementation hash, so the campaign resumed the two
existing parts and then continued from `q062`; it did not relabel, weaken, or
rebuild the prior checkpoint lineage.

## Canonical shard identity

The canonical artifact is
[`shards/offset-060/retrieval.json`](../../eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822/shards/offset-060/retrieval.json).

| Property | Observed value |
| --- | --- |
| Root file SHA-256 | `34f27da4e2b811e85d48184e510f712d950ea20b70e172d445bcdfc4121f3188` |
| Root file size | 2,249,403 bytes |
| Gold-blind 100Q population | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| Offset-60 shard | `d0ec00671068de6d43b390ecbc2af2ff65407aaf40bce1f4077bc5b756883754` |
| Retrieval implementation | `cf2577f21a7a1af1b9c5f331c7eb1672c5ba13af84ccdc2f718259192bb36e09` |
| Effective retrieval policy | `0abb42db8fd35b566029be135630af34fa42ee9a464ef8c4a889948729cbab13` |
| Environment lock | `c99f82aa37a1df009c6100110d24aeaee0bde3376ff09537700ad9848b371a95` |
| Validation policy manifest | `5263d5afd15298ec4088db9d6381ae243ddb685e9a3cf4d9892fc84e14fb9883` |
| Validation attestation | `3c054b584fa4ca7dff7e1d97d4bc532bb3e98b140b15b3fa6c35cd67be971558` |
| Validation execution policy | `c6b3d6f3f511d0a52b271e0880cb93c3866e649b614cfa4b5ac0604b704fdc8d` |
| Transcript-token proxy | 1,040,624 |
| Turns / questions | 5,204 / 10 |
| Provider calls / gold fields | 0 / absent |
| Retained request-token state | 0 bytes |

Every part is canonical JSON with a matching sidecar and an ordered match to
the root's `question_part_sha256s` and embedded question object:

| Part | Question ID | SHA-256 | Sealed elapsed seconds |
| --- | --- | --- | ---: |
| `q060` | `3b6f954b` | `c26dc0bca610bb7284f89741c7ca4952ae42483b0eadcc0314cb6be879d48a52` | 185.16557979999925 |
| `q061` | `gpt4_15e38248` | `06cffd1cb9204b352369c973cc8254b4d680be2608aa6e26cd34b05b97e20c8c` | 184.05922710000596 |
| `q062` | `1903aded` | `2cc9990a05ad4804641ad89e51cb70a77f7b59d8af0a5a33f509d95abbc8a352` | 322.4726754000003 |
| `q063` | `gpt4_4edbafa2` | `ef4578bf93eafd70f5045aacbcc718395f148f9894c6559cb235d88580cc3f2f` | 208.1444106999843 |
| `q064` | `184da446` | `f8a00c2b38fe72a10a3a85a897a1081c734cc7460b6fbb4cf23e21f0ac035142` | 212.8574197999842 |
| `q065` | `5025383b` | `bfcd8ce3d57afdd3b2a27071bba7f601f710a4830563cefaef957cbd426da1da` | 174.54850820000865 |
| `q066` | `00ca467f` | `49c6c934b36ec9cff0e8a948527c89627ee99e7a549fa829680f6a018ca2a592` | 427.95459909999045 |
| `q067` | `80ec1f4f` | `f9541972f0f77750decdf72aecb1e460a7ba3a6852c87c5a564d818eb850512e` | 177.04213170000003 |
| `q068` | `8e9d538c` | `99223c8a5eeb125d8b415fb9ad2f5e9501c8f610e48f206233327066522d51c1` | 185.03225960000418 |
| `q069` | `0a995998` | `183553e4c6073a763f8efee4116b7f5847cf1d896701c3cfb9780a6879f23a5f` | 142.9258697000041 |

The existing-artifact `retrieve` path reconstructed the locked population,
policy, implementation, and environment identities, re-read the root and all
parts, and returned the same root SHA-256 without loading Qwen, rebuilding a
store, contacting a provider, or opening gold labels.

## Nested-stage measurements

| Stage | Selected evidence | Additions |
| --- | ---: | ---: |
| S0 causal/coverage predecessor | 273 total; mean 27.3; range 13--40 | 273 root rows |
| S1 direct episodes | 446 total; mean 44.6; range 31--58 | 173 total; mean 17.3; range 9--20 |
| S2 representative episodes | 450 total; mean 45.0; range 31--58 | 4, all on `q066` |
| S3 artifact-global closure | 450 total; mean 45.0; range 31--58 | 0 |

The exact aggregate status counts are `root=10`, `added=11`, and
`budget_exhausted=19`. All ten S1 stages appended evidence. S2 appended four
rows only on `q066`; the other nine S2 stages and every S3 stage were
budget-exhausted. This establishes bounded append behavior, not relevance or
answer accuracy. The preregistered answer treatment remains S1, so the S2
rows are not silently promoted into the primary answer-stage test.

| Final-stage budget measure | Sum | Mean | Range |
| --- | ---: | ---: | ---: |
| Prompt-token proxy | 71,845 | 7,184.5 | 6,172--7,327 |
| Context-token proxy | 68,569 | 6,856.9 | 5,849--6,999 |
| Prompt workspace | 74,405 | 7,440.5 | 6,428--7,583 |

All stages retained the exact 8,000 prompt, 7,000 context, and 256 responder
reserve. The sealed question timings total `2220.2026810999814` seconds,
with mean `222.02026810999814` and range
`142.9258697000041`--`427.95459909999045` seconds. These are retrieval-only
question timings; they exclude source and combined-store construction.

## Why the live tree could not reopen the checkpoint

The retrieval preflight binds `implementation_sha256()` over the package
source closure. The first resume attempt from the changed live tree failed
closed with `retrieval implementation changed after preflight`; it published
no new question part. This was correct behavior. New CAV/Hebbian/evaluation
modules had changed the package hash even though the old offset-60 question
parts and retrieval algorithm had not been edited.

The exact implementation identity was reconstructed from Git commit
`a66ff05d17cd7d598be33de402256f6a498aba19`. Its tree
`f4baff24befff6510cc49fae69b809808d911540` reproduces the sealed
`cf2577...36e09` implementation digest exactly. An isolated, read-only launch
snapshot lives at
[`frozen-validation-source-cf257-audit-20260823`](../../eval_results/frozen-validation-source-cf257-audit-20260823/).

| Frozen launch coordinate | SHA-256 |
| --- | --- |
| Snapshot manifest | `1523a0d2db8b666b0d3b26ece442c83fc10eb398e13fd517c4d025a51cb8332e` |
| Archived `src/memory_condense` tar | `e2f38138a84d156b734d5367b82faf027f1d334f472f1b8ad6870082e37cff86` |
| Archived `pixi.lock` tar | `d754a32a26138f4f7bb2be18d68ff802ab16c2b32f8fa21652aa3f7a05764929` |

The lock-tar value above is intentionally checked against the manifest before
launch. The launcher places only the archived `src` root first on
`PYTHONPATH` and continues to use the existing frozen development environment.
The offset-60 preflight then matched all of the original population, shard,
policy, implementation, and environment identities. `q060` and `q061` were
checkpoint hits; `q062`--`q069` were newly published under that unchanged
identity.

## Store cross-binding

| Combined-store coordinate | SHA-256 or count |
| --- | --- |
| Manifest file | `b5217fc6839c73cbbc42f2676f4e11f4e3add156223f5b77ff3c152770d6db96` |
| Combined-store receipt | `b16d4614888b3886063888392a8e06fccc34e7d2de860fdd58600c89b5438859` |
| Compilation receipt | `d5e7695ecd3afd86adda8743cea9978e89b99fa2097290edfbd76c805fb1acc9` |
| Source database | `8816ce7aafe612907ff5b33e30764a8adbc643172cafac27f9a5889326976368` |
| Target database | `6de8883079487934ee4518de8304ccfe2855ad9f75b22adff9726f067b9a96e4` |
| Target index | `a69157b2bcc3ad8b9d5379e0eca6968cc954613ff3b022da7961cab8df837940` |
| Source/target store identity | `e472264d347d3ca3b855d2b085174b040da3f0bddab8883b25802534e764b043` |
| Final snapshot | `0f152517458f143c78bb6701d77f714e3d6ce2e03163ee4d9a83152828cbf97f` |
| Turns / chunks | 5,204 / 7,808 |
| Compilation source receipts | 457 |
| Causal events / graph nodes / edges | 2,289 / 7,134 / 54,279 |

The root cross-binds the selected source receipt, combined-store receipt,
compilation receipt, policy, implementation, environment, ten question-part
hashes, and zero retained request-token state.

## Current campaign state

Offsets 0 through 60, in steps of ten, are now complete: 70 of the locked 100
questions have sealed provider-free cumulative retrievals. Offset 70 passed
preflight with shard identity
`bbb02cadb883ce8b6c2fea22f2031ce179cdb25e7e70d4f2a5484330c36e7c5b`
and is running sequentially under the same frozen implementation and
environment. Offsets 80 and 90 remain preflighted but unrun.

No responder or judge call has run on this validation population. The exact
ten-shard merge, fixed-S1 Terra artifact, independent Sol score, `>=95%` gate,
and fair Mem0 comparison therefore remain open.
