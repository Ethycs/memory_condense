# Adaptive source-balanced hot full100 result

Date: 2026-09-06

Status: complete comparative v7 retrieval, answer, and judge run on the
previously analysis-used locked validation100 fixture; 93/100 complete-source
packets and 70/100 semantic answers

## Result

The v7 successor repaired the largest demonstrated defect in the fast v6
path without rerunning search or adding a local model. It authenticated the
sealed v6 selection and its retained 96-deep BM25, exact-dense, and temporal
frontiers, preserved every packed v6 A3 evidence row as an immutable prefix,
and admitted 32 additional chunks with a source-balanced bounded-RRF policy.

That one provider-free admission change moved complete labeled-source reach
from **72/100 to 93/100** and literal-answer containment from **51/100 to
54/100**. Mean labeled-source recall rose from 0.835833 to 0.955000. The
separate Terra/Sol answer plane moved from **66/100 to 70/100 semantic**.

The semantic change comprises eight wins and four losses. This is important:
v7 never removes a parent evidence row, but adding raw evidence can still
change the answer model's attention, synthesis, and wording. Additive
retrieval is monotone in evidence membership; unconstrained LLM answering is
not monotone in correctness.

This is a strong repair of the fixed-budget admission failure, but it is not
the 95% terminal result. The population was already used for analysis, the
semantic score remains 25 points below the policy-v5-r3 frontier, and a fresh
confirmation population was not opened.

## Frozen population and parent

| Item | Value |
|---|---|
| Dataset SHA-256 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Split SHA-256 | `8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4` |
| Population SHA-256 | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| Stores | 10 independent approximately-1M-token namespaces |
| Questions | 10 per namespace, 100 total |
| Corpus | 10,441,617 transcript-token proxies; 54,246 turns; 79,798 chunks |
| Parent policy | `hot-raw-chunk-v6-frozen-dev10` |
| Parent selection SHA-256 | `7062a1b23b231b9870d3e92ca94ac44f12a8a6ad68366787affd37d16ba737bf` |
| V7 policy | `parent-protected-opaque-source-bounded-rrf-v1` |
| V7 implementation identity | `109896e482b3025bfb0de095c6829c8c97a036baa3e3f3c38b9c8d55c207d72f` |
| Hard caps | 7,000 context-token proxies; 8,000 prompt-workspace proxies |

V7's `run` command receives no dataset and invokes no provider. It reads the
immutable v6 selection, compiled text-free chunk metadata, and source SQLite
stores. Gold enters only through `score`, after v7 selection and replay are
sealed.

## Exact policy

The selected controls are:

- protect the complete v6 `a3_protected_union.packed_evidence` sequence as the
  exact prefix;
- consider the stored BM25, exact-dense, and nonempty temporal-event
  frontiers, excluding the old source-neighborhood tail;
- treat source IDs as opaque exact-equality keys, never parsing an answer-like
  suffix or comparing them with question IDs;
- fuse candidate and source support by reciprocal rank with constant 60;
- count at most four hits from one source in one lane toward source strength;
- admit one representative from every reachable source absent from the
  parent before any source receives a second surplus chunk;
- fill the remaining 32-item surplus by deterministic source round robin;
  and
- pack the parent prefix followed by source-balanced surplus under the exact
  7,000/8,000 caps.

Duplicate chunks across routes accumulate rank support, but the earliest
route occurrence owns the hydrated result and each exact chunk ID is emitted
once. Raw route scores are deliberately not compared across lexical, dense,
and temporal lanes because their scales are not commensurate.

A separate provider-free neighborhood-suffix check regenerated immediate
same-source neighbors from the new tail seeds. It added zero complete-source
hits and zero literal hits while consuming prompt capacity, so it was not
included in the sealed v7 policy. Local linking remains useful in the
protected v6 prefix and as a separately triggered specialist; it is not a
free improvement after broad direct-source coverage.

## Retrieval diagnostics

| Measure | Frozen v6 | Adaptive v7 | Change |
|---|---:|---:|---:|
| Complete labeled-source packets | 72/100 | **93/100** | **+21** |
| Mean labeled-source recall | 0.835833 | **0.955000** | +0.119167 |
| Literal-answer containment | 51/100 | **54/100** | **+3** |
| Mean best evidence F1 | 0.086107 | **0.099386** | +0.013279 |
| Mean packed chunks | 24.05 | **52.86** | +28.81 |
| Maximum context proxy | 4,420 | **7,000** | +2,580 |
| Maximum workspace proxy | 4,995 | **7,608** | +2,613 |
| Candidate suffix drops | 0 | **319 across 44 questions** | +319 |

All 21 complete-source changes are gains; no previously complete-source
packet becomes incomplete. The three literal gains are ordinals 22, 55, and
88, with no literal regression.

Seven packets remain source-incomplete:

| Ordinal | Question ID | Category | Labeled-source recall |
|---:|---|---|---:|
| 7 | `gpt4_e061b84f` | temporal reasoning | 0.666667 |
| 36 | `32260d93` | single-session preference | 0.000000 |
| 54 | `gpt4_8279ba03` | temporal reasoning | 0.000000 |
| 61 | `gpt4_15e38248` | multi-session | 0.500000 |
| 77 | `0bc8ad92` | temporal reasoning | 0.666667 |
| 86 | `gpt4_7f6b06db` | temporal reasoning | 0.666667 |
| 93 | `eac54add` | temporal reasoning | 0.000000 |

Ordinals 7 and 36 are the two residual admission misses from the 17
answer-wrong cases whose complete source set was already present in the v6
wide frontier; v7 closes the other 15. Ordinals 54, 61, 77, 86, and 93 are
the five previously identified frontier misses. They require broader address
discovery or a specialist, not another redistribution of the same candidates.

## Semantic answer result

Terra received only the sealed provider-ready messages and no gold. Sol then
received only question, reference answer, and sealed prediction. Both phases
used 100 physical calls, zero automatic retries, and 100 unique prompts.

| Category | Frozen v6 | Adaptive v7 | Change |
|---|---:|---:|---:|
| Knowledge update | 14/16 | 14/16 | 0 |
| Multi-session | 14/27 | **16/27** | +2 |
| Single-session assistant | 9/11 | 9/11 | 0 |
| Single-session preference | 1/6 | 1/6 | 0 |
| Single-session user | 12/14 | 12/14 | 0 |
| Temporal reasoning | 16/26 | **18/26** | +2 |
| **Total** | **66/100** | **70/100** | **+4** |

The eight wrong-to-right flips are ordinals 21, 27, 28, 31, 43, 53, 55,
and 65. The four right-to-wrong flips are ordinals 49, 60, 67, and 97.
Thus source-balanced widening substantially improves retrieval, but the
answer result loses four of its eight gross gains to prompt-sensitive
synthesis. The next accuracy repair should preserve this retrieval result and
work on evidence density, obligation-aware packing, or bounded answer
verification rather than reverting the source allocation.

## Latency and the packing defect

The v7 runtime measures only the adaptive increment over the already sealed
v6 retrieval; it excludes parent retrieval, model/provider time, and artifact
publication.

| Incremental stage | p50 | p95 |
|---|---:|---:|
| Source-balanced selection | 4.233 ms | 6.564 ms |
| Surplus hydration | 3.459 ms | 5.304 ms |
| Pack, render, and token count | 12.221 ms | **346.810 ms** |
| Complete adaptive increment | **22.858 ms** | **356.965 ms** |

The distribution is bimodal. Fifty-six packets fit the complete proposed
prefix and finish in 19.126 ms median, including 9.619 ms median packing. The
other 44 overflow the context cap. Their packer first counts the complete
packet and then renders and recounts prefixes 1 through the first rejection.
Because each successively longer prefix repeats almost all prior text, that
branch performs quadratic cumulative text work. Overflow packets take
282.764 ms median end to end, of which 270.057 ms is packing. This is a
packing-algorithm defect, not evidence that source balancing, database
hydration, or 1M-memory search is intrinsically slow.

The subsequent sealed v8 packing-only run reproduces all 100 provider-bound
payloads byte-for-byte. Its binary packer measures 30.229117 ms mean, 11.8461
ms p50, 67.8294 ms p95, and 72.499 ms maximum, versus the sealed v7 artifact's
129.835349/12.22115/346.8103/395.3041 ms. That is a 4.295x mean/total and
5.113x p95 speedup. The v7 baseline is a non-contemporaneous same-machine
artifact, and these are isolated packing measurements rather than full
retrieval or provider latency. Research Log 111 records the seal, replay, and
exact artifact hashes.

## Replay and provider-journal verification

The gold-blind v7 replay reproduced the complete 100-question semantic
population byte-for-byte. Both question-population digests are
`5711a5e733984b736b854fa980e412d0ce3a3c85ee52fd45e5194c46552710c2`.

The answer and judge checkpoint directories each contain exactly 100 request
journals and 100 response journals, plus their lock file. Read-only replay
authenticated all 100 Terra records and all 100 Sol records from checkpoints
with no replacement provider call. A direct receipt audit also matched every
artifact row's call key and request/response `journal_sha256` to its
corresponding journal pair: 100/100 for answers and 100/100 for judgments.

| Plane | Runtime identity SHA-256 | Prompt population SHA-256 |
|---|---|---|
| Terra answers | `29a5d0975ed03968671f1dc98ed80611db56763bdd91586647b33aba30533833` | `7339764ae31e457700eabf9f813800c0e513b0f0aed7d0994e22d4ccb6b03f82` |
| Sol judgments | `d6fe4f87757ba0f11484b82b6c8fdc33ede0e5581b12a6e55e5c3b284f51aad3` | `599028b9d51d10d97e21c52cc2cd94f2bd6197bf2f0308c3d04e0c18de47893a` |

## Artifact receipts

Canonical root:
`eval_results/longmemeval-1m-hot-retrieval-adaptive-full100-validation-20260905`

| Artifact | SHA-256 |
|---|---|
| Frozen v6 parent selection | `7062a1b23b231b9870d3e92ca94ac44f12a8a6ad68366787affd37d16ba737bf` |
| Adaptive semantic selection | `867a4439af1c369c3f702491045392b8c64c5e2b3b4216c93fd973ada1b6df20` |
| Adaptive runtime | `aec7621c70c68b00fb3c08f921a09ced495ab348a1aa9d80def2ca76a2c56bf8` |
| Gold-blind replay | `5c7141556459e424e333928718f567202fa51645bd4dd0d84c856248a4c3ad8b` |
| Retrieval diagnostics | `3d3d79ea10711209c312212619c8a7311ef7abc7778e36d3a82c07c9030a74a9` |
| Sealed Terra answers | `26285e679c3d1d4979198f79202fe6934d4adbcc440b79f5b4e433a353d28064` |
| Sealed Sol judgments | `5ca56b60817875546f19afad5c8ee9a017a261fc0b498bd4a580e46a074ad10c` |

The adaptive tool source sealed into the selection has SHA-256
`489a0d1d5f12c1bc9fd72686e085d50de085600fca9e673a3c2222d08fbdc85c`;
the source-admission module has SHA-256
`7d2284a79a75ee5aa2e6e93a264fe9d4ad84089e2dd8ef6e79196ec979ef0823`.

## Reproduction sequence

```powershell
$py = '.pixi\envs\dev\python.exe'
$parent = 'eval_results\longmemeval-1m-hot-retrieval-full100-validation-20260905'
$out = 'eval_results\longmemeval-1m-hot-retrieval-adaptive-full100-validation-20260905'
$source = 'F:\Keytone\Documents\GitHub\memory_condense\eval_results\longmemeval-1m-recall-guarded-cumulative-validation-20260822'
$dataset = 'C:\Users\Keytone\Downloads\memory-condense-rig\datasets\longmemeval_s_cleaned.json'
$split = 'docs\10 - Research Log\data\longmemeval-95-target-split-v2.json'
$selection = '867a4439af1c369c3f702491045392b8c64c5e2b3b4216c93fd973ada1b6df20'

& $py tools\assay_hot_retrieval_adaptive_full100.py --parent-root $parent --output-root $out --source-root $source run --surplus-budget 32 --rrf-constant 60 --max-hits-per-source-per-lane 4
& $py tools\assay_hot_retrieval_adaptive_full100.py --output-root $out --source-root $source replay
& $py tools\assay_hot_retrieval_adaptive_full100.py --output-root $out score --dataset $dataset --split-manifest $split
& $py tools\evaluate_hot_retrieval_full100.py --selection-profile adaptive-v7 --output-root $out --expected-selection-sha256 $selection --authorized-provider-calls 100 answer
& $py tools\evaluate_hot_retrieval_full100.py --selection-profile adaptive-v7 --output-root $out --expected-selection-sha256 $selection --authorized-provider-calls 100 judge --dataset $dataset --split-manifest $split
```

On replay of an already complete provider journal, authorization must be zero,
not 100. The evaluator authenticates the exact missing-call count before it
can open a provider client.

## Decision

Promote source-balanced surplus admission as the v6 hot path's additive
retrieval successor. It recovers the great majority of candidates that fixed
per-lane budgets had already found and establishes that admission—not 1M
address search—was the dominant source-recall defect.

Do not call v7 terminal. The binary-prefix v8 latency repair is now sealed
with byte-identical packets. Next isolate the five true frontier misses from
the two remaining admission misses. Separately address answer non-monotonicity:
the four semantic regressions show why more retrieved evidence must be paired
with evidence-density or obligation-aware answer policy rather than assuming
that a larger raw prompt always helps.
