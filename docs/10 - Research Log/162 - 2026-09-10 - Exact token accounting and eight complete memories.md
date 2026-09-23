# Exact token accounting and eight complete memories

**Date:** 2026-09-10  
**Status:** eight complete memories; 400 date-comparison requests prepared; no new scored result  
**Predecessor:** [161 - Eighth memory admission and reused diagnostic preflights](161%20-%202026-09-10%20-%20Eighth%20memory%20admission%20and%20reused%20diagnostic%20preflights.md)

Eight approximately 1M-token memories now have complete source admission,
attention leaves, semantic indexes, user addresses, and passage addresses.
The frozen semantic-seed versus as-of experiment has 400 of its required
500 requests prepared. No answers or judgments have been sent in that
experiment. The latest scored development result remains 41/50 for the existing
reader, and the historical cumulative policy remains 95/100 without matched
fresh end-to-end latency proof; see Logs 154 and 132 respectively.

## Exact reconstruction exposed nonadditive token counts

Scheduler 12213 completed offset 070's 2,686 attention leaves using fourteen
Qwen calls, then stopped during semantic compilation. Its original compiler
required the sum of whole-turn token counts to equal the manifest's sum of
fragment token counts. The two units differ at one fragment boundary:
1,039,791 fragment tokens versus 1,039,792 whole-turn tokens. All 5,420
fragments reconstruct all 5,413 turns in 488 sources exactly.

The sole nonadditive turn is `eval-turn-b2bb89bb037be225fda4ee7a15d12e92`:
its two fragments total 2,180 tokens and the restored turn totals 2,181.
Its exact text SHA is
`161216962baec1111a4d6f81048f127ce31ecac6924568446741b40cd1ed7110`.

`tools/compile_spine_semantic_index_v2.py` retains the original exact
reconstruction, source/turn populations, partition verification, and
summary-only embedding method. It checks fragment tokens against the manifest
and records whole-turn counts separately, including every nonadditive turn.
It introduces no arbitrary mismatch tolerance. The original compiler and
failed scheduler artifacts remain unchanged.

Eight focused token-accounting tests passed in 1.05 s, including exact-byte,
coordinate, duplicate, ordering, and population rejection cases. The successor
compiler completed successfully; the following local index driver made no raw,
Qwen, answer, or judge calls.

All paths in this table are relative to `eval_results/`.

| Artifact | SHA-256 |
| --- | --- |
| `full1m-spine-semantic-offset070-20260910-r2/index.json` | `dfd13d1e9b4b7fe389f823e499aebf85e779abe1aca599fb0ddf1d185192875f` |
| `full1m-spine-user-addresses-offset070-20260910-r2/addresses.json` | `a5e4c43ff88c900a20e0df64a4da17177b1920add9f5e75eaf2e5a25e22e49c2` |
| `full1m-spine-facet-addresses-offset070-20260910-r2/addresses.json` | `5dc0e3568e2067d652881e6ced420cb12d907382b896f8d726716d4291730230` |
| `full1m-spine-semantic-seeds-joint-offset070-20260910-r2/preflight.json` | `bcc69c6313ce6c7f751bc617bc5ef49084118f315c65121182db539c5a2b8add` |
| `full100-spine-index-completion-offset070-20260910-r1/complete.json` | `443a131fa28653756fb0d75de5e3111bd168d62989262e7882c5628033106bd4` |

The eighth index has 5,846 passage addresses. Its v11 source-admission receipt
and common admission method remain those in Log 161. No raw bytes or earlier
seven memory indexes changed.

## Four hundred prepared requests

The separate-process preparation driver completed offsets 050, 060, and 070
with live local query encoding. It then validated all eight preflights and
their empty answer journals. The earlier four transferred diagnostic preflights
retain their provenance; no fresh-encoding claim is made for their transfer.
Timed execution still requires fresh embedding, routing, date projection, and
exact hydration inside each memory request's clock.

Root: `eval_results/full1m-spine-as-of-full100-20260910-r1`.

- Unchanged protocol SHA:
  `09e2b5c8c69a509bc23b2f62e24e0cb53507e30389127db1e6e7d67797ba29ad`.
- `preparation-first80.json` SHA:
  `43869f83742364a4d108faeaf2a297d6fd47c2aa699f2a4d3ed7ce12a76455cc`.
- Offset 050 preflight SHA:
  `53b4a5953666bc5cc86d2b081e0c92d05a2a217891ca3825bef52ecd61dec336`.
- Offset 060 preflight SHA:
  `7c16ccb07dc0b5f9e97362cb98153d26226f5b078e3df89c69252c1bc2ac544d`.
- Offset 070 preflight SHA:
  `a9207629a2785d7815be29ce4e5f5f447300bc45d8f1bc03c19d5b88922d2603`.

The actual full100 runner revalidated all 400 requests and rejected missing
offset 080 (exec `33a69d`). It published no runner plan, reserved no execution,
and made no provider calls. Existing runtime tests remain applicable; no
evaluation policy changed.

## Process ownership and continuation

Raw scheduler **42892**, PID **63016**, creation time **1789052354.6007824**,
continues offset 080 and then 090 under its existing release. Do not duplicate
or restart its requests.

The failed compilation scheduler **12213** is terminal at offset 070, with its
failure and reservations preserved in
`full100-spine-memory-completion-20260910-r2`. The separate eighth-index
completion driver is also terminal and successful.

New compilation scheduler **48873**, PID **10900**, creation time
**1789059427.7395196**, owns offsets 080 and 090 only. Its v3 worker uses the
successor semantic compiler with unchanged model caps and source-admission
rules. Fifteen worker/scheduler tests passed in 3.27 s. The release is at
`full100-spine-memory-completion-20260910-r3/release.json`, SHA
`da12b01028ded4d0a12e4046b183fffa1fe18cb80bf476d8b453c057cec565eb`;
its schedule preflight SHA is
`93cb9f4e8c979b228686f70abab8c319185bd155643537d53a296c7882e4fb9b`.
Both live process identities were rechecked in exec `33a69d`.

After each remaining original control finishes, prepare that namespace using
`tools.prepare_spine_as_of_full100 namespace` in a separate process. Do not
rerun the first eight preparations. The as-of experiment is the next timed
comparison; the older semantic-seed campaign remains its control source.
Require all ten complete memories, all 500 frozen requests, completed bulk
ingestion, fresh readiness, and an idle worktree before timed execution. Seal
all 500 answers before judging and assess accuracy and both API latency
comparisons on those same predictions.
