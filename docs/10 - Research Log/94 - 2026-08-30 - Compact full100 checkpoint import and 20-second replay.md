# Compact full100 checkpoint import and 20-second replay

Date: 2026-08-30

## Result

The completed resident full100 construction has now been imported into the
compact resumable-v2 layout and replayed byte-identically. The exact production
construction and replay SHA-256 is:

```text
7fe63e3890936feebf239dc4f16541a1336306d55570a6d78010aefc0e7b9278
```

The initial compact import took approximately 12 minutes 55 seconds. Sampled
working-set memory stayed between 0.87 and 0.99 GB. A subsequent fully pinned
replay completed in 20.281 seconds, reproduced the construction bytes exactly,
and made zero provider calls while retaining zero transformer token state.

This is a production apparatus result, not a QA-accuracy result. It certifies
the import, namespace checkpoints, filesystem boundary, and fast replay of an
existing construction. It does not score answers, provide a judge result, or
pass the >=95% accuracy gate.

## Pinned production lineage

The source construction was the completed legacy root:

```text
eval_results/matched_eval_100/locked-semantic-global-terminal-full100-v1
```

Its exact construction SHA-256 was
`7fe63e3890936feebf239dc4f16541a1336306d55570a6d78010aefc0e7b9278`.
The import did not reopen V7 or the resident store and did not require a legacy
replay artifact.

The first resumable successor was exercised at:

```text
eval_results/matched_eval_100/
locked-semantic-global-terminal-full100-resumable-v1
```

Its preflight SHA-256 was
`02119bf1a4a635676287891db354fa0fe298f1d364a7478aa0bba90f6146df22`.
The import completed in approximately 50 minutes 40 seconds, with sampled
working-set memory between 5 and 11 GB. It published 2.288 GiB of namespace
sidecars, but its checkpoint envelopes duplicated another 2.288 GiB of those
payloads. The v1 run proved authenticated conversion and replay compatibility;
the duplication made it unsuitable as the final operational format.

The compact successor used the distinct root:

```text
eval_results/matched_eval_100/
locked-semantic-global-terminal-full100-compact-resumable-v2
```

Its exact externally pinned import-attestation SHA-256 is:

```text
c7ce5b79862e46194b2fc1c7c20291ce8926d056f61d2f7eae0331fc0f85682e
```

No v1 preflight is relabeled as this v2 attestation. The version boundary is
explicit: the legacy construction bytes remain unchanged, while the v2
attestation and compact-checkpoint formats describe a different storage and
verification lifecycle.

## Compact storage result

The compact root contains the resident construction, its ten content-addressed
namespace sidecars, ten small authenticated reference checkpoints, and the v2
import attestation. The measured accounting was:

| Field | Production observation |
| --- | ---: |
| Namespace sidecar bytes | 2,457,003,621 |
| Namespace checkpoints | 10 |
| Approximate checkpoint payload total | 19 KiB |
| Approximate complete root size | 2.299 GiB |
| Initial import time | 12m55s |
| Sampled import working set | 0.87--0.99 GB |
| Pinned replay time | 20.281s |
| Provider calls | 0 |
| Retained transformer token state | 0 bytes |

Unlike v1, a compact checkpoint authenticates and refers to its published
sidecar instead of embedding the multi-hundred-megabyte sidecar payload. Replay
validates all ten small checkpoints before scanning the large sidecars, hashes
the sealed sidecars in bounded chunks, and publishes the replay as a verified
copy of the construction. The resulting replay SHA is the same
`7fe63e3890936feebf239dc4f16541a1336306d55570a6d78010aefc0e7b9278`.

## Fail-closed security boundary

Compact v2 applies the following controls to import, resume, and replay:

- Reuse and replay require the exact external attestation SHA
  `c7ce5b79862e46194b2fc1c7c20291ce8926d056f61d2f7eae0331fc0f85682e`;
  a merely self-consistent or self-rehashed attestation is insufficient.
- One exclusive lifecycle lock serializes mutation and validation for the
  canonical output root.
- Staging uses exclusive-create (`O_EXCL`) and no-follow behavior. Stranded
  staging is validated against the expected content instead of trusted or
  silently replaced, and hardlinked staging files are rejected.
- The small checkpoint population is authenticated before any expensive
  sidecar scan.
- The attestation explicitly binds the canonical output root, so an otherwise
  valid tree cannot be copied to a different root and accepted there.
- Reserved basenames are rejected before lifecycle work begins.
- On Windows, reparse points, junctions, symlinks, and redirected ancestors are
  rejected for the output and staging boundary.

Together these controls preserve write-once behavior while making interrupted
or foreign state distinguishable from an authenticated resumable root.

## Verification and decision

Final verification reported:

- 20/20 focused compact-v2 tests passed;
- 39/39 tests in the complete full100 construction test file passed; and
- an independent review returned GO.

The tests cover byte equivalence, exact attestation pinning, checkpoint-first
validation, resume behavior, staging and hardlink attacks, output-root binding,
reserved basenames, Windows redirection surfaces, and the zero-provider,
zero-state replay contract. The production replay then exercised the real
2,457,003,621-byte sidecar population rather than only synthetic fixtures.

## Remaining P2 limitations

The production result leaves four lower-priority operational limitations:

1. A crash before the first attestation is published repeats the initial deep
   authentication pass.
2. Peak memory includes one decoded namespace sidecar, approximately
   170--280 MiB, even though only one is retained at a time.
3. Resume performs redundant hashing of both source and target sidecars.
4. Publication does not claim directory-fsync durability across a
   whole-machine power-loss event.

None of these limitations changes the byte-identical construction result or
authorizes provider execution. Compact v2 is the preferred resumable artifact
layout for this completed construction; answer and judge lifecycles remain
separate gates.
