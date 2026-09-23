# Binary ranked-prefix packing full100 result

Date: 2026-09-06

Status: sealed provider-free v8 packing result; 100/100 payloads byte-equivalent
to v7, with no retrieval or provider-accuracy change

## Question

Can the quadratic overflow-packing branch in adaptive v7 be replaced by a
binary ranked-prefix packer without changing any provider-bound payload?

## Method

V8 consumes the immutable v7 selected packets and applies the same policy:
emit the longest ranked prefix that satisfies the 7,000-token context cap and
8,000-token prompt-workspace cap. It first tests the complete prefix. When the
complete packet does not fit, it uses binary search over the append-only,
nondecreasing prefix costs and audits the maximal fitting boundary.

This is deliberately a packing-only experiment. It does not rerun search,
change candidate ranking, inspect gold, or call Qwen, Terra, Sol, or any other
provider. The equivalence oracle is the sealed v7 linear ranked-prefix
payload.

## Sealed result

All 100 v8 payloads are byte-for-byte identical to their sealed v7 payloads.
The population contained 5,605 selected evidence rows: 5,286 were packed and
319 were dropped at the same boundaries as v7. Fifty-six packets took the
complete-prefix fast path and 44 took the binary fallback. The implementation
made 400 context-count calls and 281 prompt renders/counts.

Provider calls were zero, as were Qwen calls. Because the exact bytes reaching
the answer model are unchanged, the previously sealed retrieval and answer
measurements are unchanged too: 93/100 complete-source reach, 54/100 literal
containment, and 70/100 Terra/Sol semantic accuracy. These are inherited by
equivalence, not newly scored accuracy results.

The provider-free replay reproduced the run payload exactly across all 100
questions. It reports no gold fields and zero provider or Qwen calls.

## Packing latency

The timing scope is the packer call only. Artifact loading, v7 validation,
serialization, hashing, provider RTT, prefill, and decode are excluded.

| Packer artifact | Mean | p50 | p95 | Max |
|---|---:|---:|---:|---:|
| Sealed v7 linear prefix recount | 129.835349 ms | 12.22115 ms | 346.8103 ms | 395.3041 ms |
| Sealed v8 binary ranked prefix | 30.229117 ms | 11.8461 ms | 67.8294 ms | 72.499 ms |

Across the same 100-question population, v8 is 4.295x faster by mean and total
packing time, 1.032x at p50, and 5.113x at p95. The median changes little
because the 56 full-fit packets already used a cheap fast path. The large gain
is in the 44-packet overflow tail that previously repeated progressively
larger renders.

The v7 baseline is a **non-contemporaneous same-machine sealed artifact**, not
an interleaved paired timing run in the same process. The byte-equivalence
claim is exact; the latency comparison remains subject to ordinary
cross-run system variance. This artifact also does not claim a measured
end-to-end retrieval or provider latency improvement beyond the isolated
packing stage.

## Verification

An independent focused run passed 17 tests after redirecting the test base
temporary directory to a sandbox-local path. A separate agent run reported 28
focused-plus-legacy tests passing. These are distinct receipts and are not
combined into a larger test count.

## Artifacts

Root:
`eval_results/longmemeval-1m-hot-retrieval-binary-packing-full100-validation-20260905`

| Artifact | SHA-256 |
|---|---|
| `run.json` | `933898618191a779bdaf202cfe4e4e49d05f1aaedcc5345949f0488d5b3231f4` |
| `runtime.json` | `8c2bb320998cd301c458d4c68af162b3e2f14a9542ea4fe750f215119014f384` |
| `replay.json` | `1d86c7598cf336e94a1ff8c47c63c97166724c6074239e7c3ec8bc5b9b17b206` |

The sealed implementation digest is
`e4bb4aa5b4a2ccd6c71807ea9d5889306dd060322deffc6564a4c2d92f1e81e1`.
The run binds v7 selection
`867a4439af1c369c3f702491045392b8c64c5e2b3b4216c93fd973ada1b6df20`
and v7 runtime
`aec7621c70c68b00fb3c08f921a09ced495ab348a1aa9d80def2ca76a2c56bf8`.

## Decision

Promote binary ranked-prefix packing as the behavior-preserving successor to
the v7 linear overflow recount. The next latency measurement should cover the
integrated resident retrieval path end to end; it must not reinterpret this
packing-only result as provider latency. Accuracy work remains separate:
resolve the two admission misses, five frontier misses, and answer-policy
regressions without changing this sealed payload-equivalence finding.
