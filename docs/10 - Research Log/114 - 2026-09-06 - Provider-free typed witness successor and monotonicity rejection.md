# Provider-free typed-witness successor and monotonicity rejection

Date: 2026-09-06

Status: sealed full100 provider-free construction and post-hoc evidence assay;
the measured source-coverage gain is real, but this exact composition policy
is **rejected for promotion** because derivative excerpts can displace their
authoritative parent chunks. This is not an answer-accuracy result.

## Outcome

The successor added three independently selected, zero-model lanes to the
sealed source-seed hybrid v3 packet: profile/preference evidence, typed
action/date/role witnesses, and immediate activated-turn links. On the
analysis-used locked validation100 fixture it moved complete gold source-ID
reach from 93/100 to **97/100**, with no source-reach regression:

| Evidence diagnostic | Sealed v3 parent | Successor | Change |
|---|---:|---:|---:|
| Complete gold source-ID reach | 93/100 | **97/100** | +4 |
| Mean gold source-ID recall | 0.955000 | **0.983333** | +0.028333 |
| Literal-answer containment | 56/100 | **57/100** | +1 |
| Mean best evidence F1 | 0.115009 | **0.133699** | +0.018690 |

The four newly source-complete ordinals are 7, 36, 54, and 61. The remaining
source-incomplete ordinals are 77 (2/3 sources), 86 (2/3), and 93 (0/2).
All 100 questions took the successor route and none used the exact-parent
fallback. Construction made zero provider or model calls, loaded no gold, and
retained no transformer token state. Gold was opened only by the separate
`score` command after construction bytes were sealed.

Those aggregate diagnostics are not sufficient for promotion. An adversarial
composition audit found that profile and typed-witness sentence/clause
excerpts were labeled with the physical backing chunk ID. Because earlier
specialist lanes owned exact-ID collisions, an excerpt could replace a longer
authoritative raw chunk bearing the same ID. Across the 100 rows:

- 380 parent collisions occurred across 87 packets;
- another 440 parent rows were truncated across 83 packets after reranking;
- five ordinals (20, 42, 50, 76, and 79) regressed best-evidence F1; and
- the activated-turn lane selected 1,209 rows but retained only 308 after
  global deduplication, because seed chunks consumed its 1,200-token budget
  before their guaranteed parent collisions were removed. Twenty-three rows
  retained no activated-turn evidence at all.

Complete source-ID reach therefore remains a trustworthy property of this
artifact, but monotonic evidence preservation and answer recall do not. The
97/100 number must be described as **structural source coverage**, not 97%
semantic accuracy and not a promoted retrieval result.

## What the assay established

The positive finding survives the policy rejection: ingest-compiled typed
addresses and local turn topology expose real evidence that the v3 packet
misses. The typed lane is especially effective for completed-vs-planned
actions, first-person role, date constraints, and enumerated events. The
profile lane recovers a durable preference source under a generic
recommendation query. Immediate source adjacency can hydrate raw context
around those hits.

The test also localized the next connectivity problem. Ordinal 77 needs a
cross-source `museum` bridge plus participant/date filtering; ordinal 86 needs
travel linkage with a strict completed-vs-planned distinction; ordinal 93 is
primarily a dated business-milestone/action lookup. A phrase/passage graph is
most likely to add unique value on 77, conditionally on 86, and should remain
secondary to typed temporal lookup on 93. See [Analysis 32](../08%20-%20Analysis/32%20-%20Incremental%20conversational%20association%20graph%20overlay%202026-09-06.md).

## Latency

The full command took 311.946 seconds. That includes loading the authenticated
parents and store context, building one resident full-store, typed, and link
index for each of ten approximately 1M-token namespaces, running 100 queries,
validating receipts, and publishing the canonical artifact.

| Scope | Time |
|---|---:|
| Parent artifact load | 11.911 s |
| Locked store-context load | 10.983 s |
| Ten full resident-index builds | 127.228 s total |
| Ten typed-index builds | 123.764 s total |
| Ten link-index builds | 3.176 s total |
| Warm per-question mean / p95 | 321.271 / 487.428 ms |
| Warm profile / typed / link means | 6.161 / 37.490 / 5.178 ms |
| Warm final composition mean | 262.762 ms |

The warm composition dominates because this successor accidentally reused the
old linear overflow-prefix counter instead of the already implemented binary
ranked-prefix packer. That is an implementation defect, not an inherent graph
or typed-retrieval cost. The typed cold build is also an assay reconstruction;
in the intended service it is compiled incrementally as chunks arrive.

The reported 321.271 ms is also not prompt-ready latency. The inherited packer
returns the provider-ready timestamp, but this assay discarded it and timed
subsequent validation and receipt hashing inside composition. The repair must
report prompt-ready and audit/publication time separately.

## Artifact receipts

Canonical root:
`eval_results/longmemeval-1m-hot-v3-provider-free-witness-full100-20260906`

| Artifact | SHA-256 |
|---|---|
| `construction.json` | `f249e20b2eb7498cddc8b236c424c88071de5edd68c73a7e8f74928efb1ff6d3` |
| `runtime.json` | `24c4d5befd2c88cbf1f62baf864c3d99469a21d3efbf5e5125104499de10c137` |
| `scores.json` | `78b12fec3356a0a9587026b0c9319ff6a0cb35067b9097eec1a5640efeaa1504` |

The separate seven-row construction used root
`eval_results/longmemeval-1m-hot-v3-provider-free-witness-residual7-20260906`
and sealed construction SHA-256
`28de55a9571de665b54ce55077586f08488ac517f4997fb4b6d7c2fc1f1adfb9`.
It recovered the same four complete sources before the full100 transfer.

Focused typed/action/turn/source-neighborhood tests passed before the run, and
the orchestrator suite passed after the parent-row receipt binding was fixed.
The later adversarial audit demonstrates why passing those tests was not
enough: one test explicitly blessed different excerpt text under the same
physical chunk ID. That fixture must be reversed into a fail-closed or
parent-authority invariant.

This construction also cannot be byte-replayed as designed. Its canonical
`aggregate` embeds warm mean and p95 timings, so an otherwise identical rerun
changes `construction.json`. There is no replay command, and the recorded
implementation identity omits several decisive dependencies, including the
packer, tokenizer, full-store/source-neighborhood code, contracts, and scorer.
The artifact sidecars and the five hashes it did record are internally valid;
the missing firebreak and dependency closure still prevent a complete
standalone replay claim.

## Repair gate

The successor may be rerun only after all of the following are true:

1. profile and typed results are treated as addresses and hydrate the exact
   full raw backing chunks, or carry distinct derived-span identities without
   masquerading as those chunks;
2. a physical chunk-ID collision with different raw bytes fails closed or
   preserves the authoritative parent raw row;
3. activated-turn seeds remain activation inputs and do not consume the
   neighborhood evidence budget;
4. final packing uses the deterministic binary ranked-prefix packer;
5. construction excludes timings and supports byte-identical zero-call replay;
6. implementation identity covers every result-affecting dependency;
7. receipts report parent rows retained, displaced, and byte-conflicting;
8. reduced residual scoring shows the source gains survive; and
9. full100 reports source reach, literal/F1 regressions, and latency before any
   semantic answer calls are considered.

Only the repaired, newly sealed successor can become the parent of the
incremental phrase/fact graph assay. No Terra responder or Sol judge calls
were made for this experiment.
