# Safe QK/CAV admission saves prompt tokens on a locked fresh split, but the development recall gain does not replicate

**Status**: measured local source-family confirmation; not a public benchmark
**Cost**: $0; local BF16 CUDA compilation plus model-free SQLite reads
**Model**: Qwen3-8B embeddings and complete layers 0–6
**Machine-readable summary**: [`../../eval_results/qwen_associative_memory_confirmation_summary.json`](../../eval_results/qwen_associative_memory_confirmation_summary.json)

## Result

The bounded external-memory architecture now has one confirmed benefit and one
unconfirmed benefit:

- On a locked, source-family-fresh 18-question split, safe CAV, QK, and
  QK+CAV retrieval all preserved hybrid `k=5` recall at **83.3%** while using
  **0.6%–2.7% fewer prompt tokens**.
- The earlier development result improved recall from 83.3% to 91.7%, but the
  fresh split recovered none of its three hybrid misses. A general recall-gain
  claim is therefore **not established**.
- Physical degree-2 pruning removed 392 of 1,204 directed edges and 50,176 of
  154,112 per-head payload bytes while preserving fresh recall.
- Qwen was absent from every read arm. Compilation retained 3,248 bytes of CAV
  coordinates, 154,112 bytes of per-head edge payload, and **zero bytes of
  token K/V state**.

The safe policy was not invented before all evidence. A prior locked split
first falsified unconditional association replacement. That failure is part of
the result and explains why a second confirmation was required.

## Protocol chronology

### Development split

The first 12-question source-family split selected the two-hop operating point:
`k=5`, one QK slot, three edges per frontier node, and a global association
candidate cap of eight. It became development data as soon as those choices
were made.

| Arm | Recall | Mean prompt tokens | Recovered / lost |
| --- | ---: | ---: | ---: |
| Hybrid `k=5` | 83.3% | 989.8 | — |
| Safe two-hop QK | **91.7%** | **917.9** | 1 / 0 |
| Safe two-hop QK+CAV | **91.7%** | 929.8 | 1 / 0 |
| Safe QK after degree-2 pruning | **91.7%** | 941.8 | 1 / 0 |

These safe numbers are post-hoc replays. They show that the later admission
rule did not erase the useful development link; they do not validate the rule.

### Locked v2: the failure that changed the design

The v2 launcher fixed a new six-family selection seed and excluded all seven
families consumed by earlier work. Its anchor pack hash was
`62e1068704df04370f3ab216e2a98a14186e72ddc24299742adc600c38010286`;
the locked arm hash was
`631032d4cf3c51a7f6199954348ba54e06741c6952fdfae8c10666169b6db83f`.

| Arm | Recall | Mean prompt tokens | Recovered / lost |
| --- | ---: | ---: | ---: |
| Hybrid `k=5` | **100.0%** | 1,032.4 | — |
| Unconditional CAV slot | 94.4% | 1,054.1 | 0 / 1 |
| Unconditional two-hop QK slot | 94.4% | 1,043.3 | 0 / 1 |
| Unconditional QK+CAV slots | 94.4% | 1,043.8 | 0 / 1 |

All association arms displaced the same valid rank-5 answer. That answer had a
normalized lexical score of 0.991544: it was low in the fused ordering but was
nearly the strongest direct BM25 match. Adding a positive graph score to an
anchor score could not solve this because graph and hybrid scores are not
calibrated onto the same scale.

Two admission invariants were introduced after this failure:

1. A reserved tail anchor with normalized lexical score at least `0.90` blocks
   displacement; reservation is not moved upward to a stronger anchor.
2. After final hydration, the whole association composition rolls back if it
   would use more prompt tokens than the direct anchors. Rejected edges are not
   touched, so they gain no pruning utility from a result the user never saw.

The safe replay restored v2 recall to 100.0%. Safe QK used 1,012.0 tokens and
safe QK+CAV used 978.1, versus 1,032.4 for hybrid. This was post-hoc evidence,
so another split was locked before those numbers were treated as confirmation.

### Locked v3: confirmation of safe admission

V3 excluded all 13 source families consumed through v2, then selected six new
families by SHA-256 ordering over metadata only. The seed was
`mc-association-confirmation-v3-safe-locked-20260816`. The resulting corpus had
90 assistant episodes, 406 chunks, and 18 questions.

- Anchor pack SHA-256:
  `0039cbc69c938c44d0ce1b4764b944097ded7e1425eb34db51115e451e292b0c`
- Arm-file SHA-256:
  `b0fe0c32073a8d26666e23bf5a57dbf76a86911846aa63ec1de80a50b6ef3399`
- Final item budget: `k=5`
- QK depth: two hops
- Per-frontier degree: three
- Global association candidate cap: eight
- Lexical protection: `0.90`
- Maximum prompt-token increase: zero

| Locked v3 arm | Recall | Mean prompt tokens | Token change | Recovered / lost |
| --- | ---: | ---: | ---: | ---: |
| Hybrid `k=5` | 83.3% | 973.9 | — | — |
| Safe CAV, one slot | 83.3% | **947.7** | **−2.69%** | 0 / 0 |
| Safe two-hop QK, one slot | 83.3% | 961.1 | −1.31% | 0 / 0 |
| Safe two-hop QK+CAV, two slots | 83.3% | 950.1 | −2.44% | 0 / 0 |
| Safe QK, physical degree 2 | 83.3% | 967.9 | −0.61% | 0 / 0 |

This confirms the narrow claim that bounded association can save prompt tokens
without lowering recall on this fresh local split. It does not show that CAV or
QK found evidence the hybrid retriever could not find.

## Compilation and memory boundary

V3 compiled 406 chunks in one staged Qwen run:

| Measurement | Value |
| --- | ---: |
| Baseline ingest | 120.3 s |
| Batched baseline queries | 5.7 s |
| Qwen link compilation | 291.6 s (0.718 s/chunk) |
| Maximum transient workspace | 1,024 tokens |
| Mean workspace candidates | 2.79 |
| Peak CUDA allocation including prefix | 4,611,899,904 bytes |
| CUDA allocation after Qwen unload | 8,986,624 bytes |
| CAV payload | 3,248 bytes |
| Directed QK/OV edges | 1,204 |
| Per-head edge payload | 154,112 bytes |
| Retained token K/V | **0 bytes** |

The store was closed and reopened before anchors were frozen and reads were
scored. The parallel sweep loaded neither Qwen nor BGE, used `touch=False`,
and opened independent SQLite connections. Pruned arms used SQLite backup
copies, so physical deletion could not mutate the source artifact.

## Pruning result

Degree-2 pruning is the smallest policy that has retained the useful result on
all three consumed splits:

| Split | Edges | Per-head payload | Recall before → after |
| --- | ---: | ---: | ---: |
| Development | 738 → 492 | 94,464 → 62,976 bytes | 91.7% → 91.7% |
| V2 safe replay | 1,420 → 952 | 181,760 → 121,856 bytes | 100.0% → 100.0% |
| V3 locked | 1,204 → 812 | 154,112 → 103,936 bytes | 83.3% → 83.3% |

Degree 1 is not equivalent to requesting one neighbor from an unpruned graph.
A read may exclude an edge whose destination is already a direct anchor; an
unpruned graph can then use its next edge, whereas a physically degree-1 graph
has no alternate. On development, physical degree 1 erased the recall gain.

Pruning was also changed from one `SELECT` plus individual deletes per source
to batched loading and one `executemany` transaction. On the same consumed
development store, the six-arm wall time fell from 3.60 s to 2.40 s with exact
retrieval metrics preserved. This adjacent-run timing is a performance probe,
not a general latency benchmark.

## Why the three fresh misses remained misses

The full persisted graph/CAV reachability diagnostic found:

- two misses had no QK path within three hops; their best CAV ranks were 17
  and 18, outside the locked CAV cap of eight;
- one miss was QK-reachable only at hop 3;
- a post-hoc safe three-hop arm still recovered no v3 miss and erased the
  development recovery.

The current recall limit is therefore primarily write-time link coverage and
association quality, not insufficient prompt slots. Deeper recursion is not
promoted: it adds drift and failed the cross-split control.

## Decision

Promote the two admission guards to the public `MemoryCondenser` facade.
Retain two-hop QK and CAV routes as opt-in association slots under a fixed item
budget. Keep degree 2 as the measured pruning floor and degree 1 as a negative
control. Continue to store only fixed-width CAV coordinates, per-head weights,
scalar QK/OV evidence, IDs, provenance, and lifecycle counters.

Do not claim a general recall improvement. The next recall work should improve
write-time link coverage or learn an independently validated association
admission score; increasing hop count or CAV candidate count does not make a
low-ranked link correct. Public long-memory evaluation remains open.
