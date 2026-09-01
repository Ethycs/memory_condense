# Closure v8 separates source identity from plan-local labels

**Status:** superseded provider-free v8 canary; ordinals 3 and 4 published,
then ordinal 5 failed closed before publication. No accuracy result exists.

The v6 stable-S0 protocol passed its new provider-visible S0 gate, then failed
before its first question artifact because `dataclasses.asdict()` attempted to
deep-copy an intentionally frozen `MappingProxyType` in a closure receipt. V7
replaced that serializer with an explicit immutable-to-JSON projection and
sealed a new 79-question preflight:

| Seal | SHA-256 |
| --- | --- |
| v7 eligibility | `90a9be1ef6f5a9a600e9691a78f49ff5effdaeaecd9298aac1e000d66325fc83` |
| v7 preflight | `f9dd191d61bf1389c6f9bb61e120c203095166517842be4c069ecaad319e9f3c` |

The offline v7 canary completed the first expensive cumulative retrieval and
passed the S0 gate, but failed closed before publication when the same
`atom_id` appeared in the representative and artifact-global plans with
different full atom identity payloads. No v7 question, shard, answer, or judge
artifact exists.

## Root cause

`atom_id` is derived from the exact authoritative `EvidenceSpan`. The closure
`BundleBuilder`, however, is instantiated separately for each plan and assigns
the atom's `label` from that plan's first use. The two mechanisms may therefore
give one source span different labels while retaining identical source bytes,
coordinates, role, and time. Treating the complete route-local atom payload as
the cross-method source key was incorrect.

V8 now keeps two identities:

- each route retains and seals its exact complete `EvidenceAtom` identity;
- the common structural target uses that exact identity minus only top-level
  `label`.

The common projection requires the exact six-field atom schema, reconstructs a
canonical `EvidenceSpan`, verifies `atom_id == make_atom_id(span)`, and binds
the text digest, role, and creation time back to the span. Label-only drift may
merge one source target; every other field remains fail-closed. Target IDs are
derived from this label-free source identity, while primary-route identity and
per-route identity hashes remain explicit.

The target disposition also now records a selection-packet receipt only for
targets actually selected by that packet. Candidate-pool seals remain the
provenance for unselected targets.

## Sealed v8 preflight

| Field | Sealed value |
| --- | --- |
| question population | 100 |
| eligible retrieval population | 79 |
| retrieval/provider calls during preflight | 0 / 0 |
| corpus/store rebuilds | 0 / 0 |
| eligibility manifest SHA-256 | `d3189675dd8efed99f63cdee213e21eb7f7a10fdebe1efa8c2be5b50a518a124` |
| preflight SHA-256 | `28be26ae9d593fcba96980629905bb852be215e06bb190445a70eb1108570a37` |

The generator's focused suite passes 36/36. It includes frozen-mapping JSON
projection, realistic span-derived atom identities, label-only target-ID
invariance, non-label/linkage tampering, missing and extra fields, exact
route-local identity preservation, selected/unselected receipt semantics, and
policy sealing. An independent source audit gave the v8 generator GO.

## Real artifacts published before termination

V8 published exactly two offset-0 question artifacts:

| Ordinal | Question ID | Question artifact SHA-256 |
| ---: | --- | --- |
| 3 | `gpt4_2f91af09` | `5d63564a55c6445207c81f09d676341c2d8c7c7aa1500df535a7553ef89ff748` |
| 4 | `45dc21b6` | `9e9673fa4a204061a4b66973201e58e5f67ea2427319c5ef01603a64e334fc46` |

Both downstream matched adapters independently accepted the exact sealed
ordinal-3 artifact:

| Arm | candidates | selected before dedup | exact-S0 exclusions | admitted |
| --- | ---: | ---: | ---: | ---: |
| representative bridge | 35 | 4 | 0 | 4 |
| artifact global | 249 | 9 | 5 | 4 |

The adapter also now reconstructs the same per-question and merged common
attribution, verifies exact root packet and sealed-retrieval provenance, and
rejects independently resealed span, metadata, target, or aggregate drift.
Separately, the common matched-evaluation ledger has a verified runtime-plane
loader that reconstructs stage rows and binds snapshot, plan, renderer,
receipts, parent/output packets, answer run, and replay.

## Terminal scalar-bypass failure

The next eligible row, ordinal 5 (`06878be2`), is a scalar synthesis request.
The frozen selector correctly bypassed its score provider because the query
does not require complete-set coverage. Its fresh coverage report therefore
contained top-level `elapsed_s` plus the exact construction-time provider
identity, but no nested `score_provider_report.elapsed_s`: no scoring call had
occurred from which such a timer could be measured.

V8's attestation required the nested timer for every report, so it rejected
this authoritative bypass as missing telemetry. The failure happened before
publishing `q005.json`. V8 consequently has two question artifacts and no
completed shard index, merged generation, matched answers, or judge result.
The v9 successor admits only this exact identity-only scalar-bypass shape and
continues to reject missing nested timing for an invoked or malformed scorer
report.

This is an execution and provenance result, not an accuracy result. No v8
retrieval arm was answered or scored, and no promotion or completion claim
follows.
