# Source-seed assertion hybrid v3 provider-free result

Date: 2026-09-06

Status: sealed provider-free composition assay on the previously analysis-used
locked validation100 fixture; source-preserving evidence diagnostics improved,
but semantic answer accuracy and untouched confirmation remain unmeasured

## Result

Source-seed assertion hybrid v3 composes the complementary strengths measured
in adaptive v7 raw retrieval and assertion projection v2. It protects one raw
representative from every v7-activated source, admits a bounded prefix of dense
projected assertions, then spends the remaining packet budget on v7 raw
evidence. Across all 100 questions, the computable source gate selected the
hybrid route 100 times and the exact v7 fallback zero times.

| Evidence diagnostic | V7 raw | Hybrid v3 | Change |
|---|---:|---:|---:|
| Complete gold source-ID reach | 93/100 | 93/100 | 0 |
| Mean gold source-ID recall | 0.955000 | 0.955000 | 0 |
| Literal-answer containment | 54/100 | **56/100** | +2 |
| Mean best evidence F1 | 0.099386 | **0.115009** | +0.015623 |

Mean best evidence F1 improves by 15.72% relative to v7 while its measured
source reach is conserved. These are **evidence diagnostics, not semantic
answer accuracy**. No model answered the hybrid packets and no semantic judge
scored predictions. This run therefore does not establish a 95% result.

The sealed run and byte-identical replay made zero provider, Qwen, responder,
and judge calls. A new semantic experiment requires fresh authorization for
the Terra answer calls and independent Sol judge calls; no prior call approval
is reusable for that experiment.

## Construction and decision rule

Each arm selects independently before cross-arm exclusion:

1. take the first v7 raw chunk for every distinct exact opaque source ID as a
   protected source seed;
2. binary-pack the v2 assertion sequence to a separately bounded 1,400-token
   projection prefix;
3. append the remaining v7 raw sequence;
4. remove exact chunk-ID duplicates only after both arms have selected; and
5. apply the same binary ranked-prefix packer under the 7,000 context-token
   and 8,000 prompt-workspace-token caps, with a 256-token output reserve.

The route is computable without gold. The source-seed gate adopts the hybrid
only if every source seed required by the v7 packet survives; otherwise it
returns the authenticated v7 packet exactly. All 100 packets passed that gate.
The gate proves source presence only. It does **not** prove that every
answer-bearing raw remainder survived, that a projected assertion is
sufficient, or that a downstream model will perform the required comparison,
ordering, aggregation, or other reasoning.

Post-selection exact-ID deduplication also relies on one explicit data
contract: raw and projection chunks must use the same authoritative global
chunk-ID namespace for an occurrence. If independently generated IDs can
collide or the same occurrence can receive different IDs, the conservation
claim no longer follows and the route must fail closed or use a stronger
occurrence identity.

The 1,400-token projection allowance is not a validated optimum. It was chosen
after comparing several budgets on this same analysis-used validation
population. V3 seals that observed candidate so it can be evaluated, but it
must not be described as tuned on an untouched split.

## Packet shape

| Measure | Value |
|---|---:|
| Questions / hybrid routes / raw fallbacks | 100 / 100 / 0 |
| Packed chunks, total / mean / maximum | 5,905 / 59.05 / 74 |
| Protected raw source seeds | 3,722 |
| Projection input / admitted prefix / dropped tail | 2,366 / 1,556 / 810 |
| Exact cross-arm duplicates removed | 534 |
| Maximum context-token proxy | 6,997 / 7,000 |
| Maximum prompt-workspace-token proxy | 7,596 / 8,000 |

The selected packet artifact is 1,372,159 bytes. It reconstructs only the
provider-bound evidence packets and their receipts from the sealed parents;
it does not rerun the expensive original retrieval or assertion projection.

## Incremental latency

The timing scope is the v3 successor over already sealed v7 and v2 parents. It
excludes parent retrieval, parent assertion projection, store construction,
provider round-trip time, model prefill, and answer decoding.

| Incremental stage | Mean | p95 where recorded |
|---|---:|---:|
| Projection-prefix pack | 10.389 ms | 12.916 ms |
| Hybrid compose and pack | 62.453 ms | 85.628 ms |
| Provider-packet materialization | 17.126 ms | 20.094 ms |
| Complete per-question v3 path | **141.250 ms** | **172.521 ms** |

The timed 100-question collection region took 14.209 seconds. That figure
excludes parent loading, the second parent-integrity check, and artifact
publication performed by the full command. Stage means do not sum to the
per-question total because that total also includes receipt construction,
validation, and orchestration outside the three named spans. This is an
incremental composition measurement, not end-to-end prompt-to-LLM latency.

## Artifact receipts

Canonical root:
`eval_results/longmemeval-1m-hot-retrieval-source-seed-hybrid-v3-full100-validation-20260906`

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `selection.json` | 1,372,159 | `0e8027d3150bdf8335ad7a83d1a8bb4a5551312444fea2d5a4e820ece05eb3b7` |
| `runtime.json` | 20,363 | `c99a3683b4faa0ff0d309be3ede4af6319dbcdc6fc68793a684092e2612f8082` |
| `run_manifest.json` | 946 | `b01db952af11a44de008bd4c5ef4d6f9f23faa36e537cd6e7fb7888738e43147` |
| `replay.json` | 1,291 | `deb0446486c0990507d7a295f5bfdbd5d1d25ef80ce474fbafb1c282dd23085c` |
| `scores.json` | 58,465 | `c7a009eca11188b2123afc02bbc585d4766157c6fff106d102b155886355a9b8` |

Implementation SHA-256 is
`a1024b6d20010bad6427b48bbca58dd08be42bc62bbd55e04aedf12cdd6cc1c7`.
The replay reconstructed the 1,372,159-byte selection byte-for-byte. The
provider-selection loader returned all 100 materialized packets, and the
combined source-seed core, assay, packer, and loader suite
passed 61 tests.

## Interpretation and next gate

V3 validates the composition mechanism that v2 suggested: reserve a small raw
source-cover before inserting dense facts, rather than replacing raw evidence
with projection or placing all raw evidence ahead of projection. It is a
stronger provider-ready candidate than v7 under the measured structural and
surface diagnostics, but the gain can still disappear or regress at answer
time.

The next legitimate measurement is a separately authorized, sealed 100-Terra
answer plane over these exact dated-question/evidence packets, followed by a
100-Sol question/reference/sealed-prediction judge plane and zero-call replay.
Because the validation100 population and the 1,400-token choice are already
analysis-used, even a strong result there remains comparative development
evidence. A source-frozen run on a disjoint untouched population is still
required for confirmation.
