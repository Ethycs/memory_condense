# Semantic residual atomic-frontier diagnostic

**Date:** 2026-08-29

**Status:** the failed atomic-frontier construction is superseded by the
compact R7 construction; the complete R7 Terra/Sol lifecycle is sealed and
replayed. R7 scores **88/100**, one point below V3's **89/100**, and therefore
does not pass the 95% gate.

## Outcome

The first locked semantic-residual V4 construction completed all ten memory
namespaces, used the sealed 131-facet BGE-M3 query-vector population, made zero
provider calls, and retained zero transformer-token state. Its sealed
construction SHA-256 is
`7006ab83e16af7eee63ac655006757e9fe8fe8f6f43b05d1b6693f383b183001`.

The run did **not** produce a usable residual treatment:

| Observation | Result |
| --- | ---: |
| gate-eligible questions | 68/100 |
| eligible questions with a synthesis prompt | 0/68 |
| `retained_unknowns_exceed_payload_cap` commitments | 136, exactly two per eligible row |
| stored chunks with embeddings in a checked namespace | 8,122/8,122 |
| construction JSON size | 2,293,502,513 bytes |
| observed serializer private working set | about 30.9 GB |
| provider calls | 0 |

No Terra answer, Sol judge, or score was run. This is therefore neither a
regression from 89 nor evidence that semantic retrieval lacks the answer. It
is evidence that the residual adapter could not turn its candidate frontier
into a bounded provider payload.

## Two independent apparatus defects

The search was deliberately conservative: it pruned only a certified role or
literal contradiction, or a branch satisfying both a low vector upper bound
and low query-specificity upper bound. All other branches remained `MAY` and
were descended. That is a recall-safe classifier, but the next step treated
the complete novel `MAY` union atomically. If the union exceeded the
non-borrowable 2,400-token residual budget, it emitted no evidence at all.
Every eligible question hit that cliff.

Threshold tuning is not the repair. It only moves the cliff and can turn
uncertainty into false-negative pruning. A bounded retriever must be able to
send its strongest evidence while truthfully recording that other candidates
remain unresolved.

Separately, the construction inlined full leaf outcomes, retained/pruned leaf
arrays, visit subtree memberships, classified frontiers, local audits, and
selected segment bodies in multiple per-question locations. Visit coverage is
approximately proportional to leaves times tree depth for every question.
Those repetitions explain both the 2.29 GB artifact and the much larger
canonical-serialization memory spike; the shared namespace index was not the
dominant cost.

## Replacement contract

The next construction keeps the conservative search but replaces the atomic
adapter with this gold-blind sequence:

1. select the complete retained segment population by semantic search;
2. rank that immutable population using sealed query-local relevance and
   deterministic source/time diversity;
3. deduplicate exact protected spans **after** selection;
4. greedily pack novel survivors using exact provider serialization under the
   2,400-token residual cap, skipping an oversize row and continuing; and
5. receipt-bind every unpacked novel survivor as unresolved.

A nonempty partial result is usable while declaring
`packing_closed=false`, `support_closure_proven=false`, and
`fallback_required=false`. Fallback is reserved for a missing prerequisite or
zero packable evidence. Protected duplicates remain excluded from the R plane
only after selection and are reinjected through the separately budgeted,
provider-visible P-owner plane.

The compact artifact retains index, query, full-result, ranking-policy, and
ordered-population receipts/digests plus the exact provider-visible evidence
and provenance needed for citations. Exact replay rebuilds the full in-memory
search and verifies those commitments; it does not need to serialize every
leaf and visit trace into every question. An optional content-addressed trace
sidecar may preserve deep audits once per namespace, but it cannot be part of
the normal provider or scoring path.

This change is architectural rather than a validation-set parameter sweep. It
turns the final semantic layer into the bounded binary-search fallback the
memory pipeline intended, while keeping its incompleteness explicit.

## R7 provider-free construction result

R7 implements that replacement contract. The gate admits 68/100 questions;
the remaining 32 are exact V3 passthroughs. All 68 eligible rows produce a
nonempty residual prompt and none falls back for a packing failure. Exact
replay reconstructs the full conservative search from the compact
commitments and is byte-identical to the run.

| Observation | R7 result |
| --- | ---: |
| residual prompts | 68/68 eligible |
| protected V3 passthroughs | 32 |
| eligible fallbacks | 0 |
| packed exact rows | 765 |
| unpacked survivors committed by receipt | 507,966 |
| residual-plane range | 2,354--2,400 tokens |
| maximum complete answer envelope | 5,544/8,000 tokens |
| construction artifact size | 5,732,727 bytes |
| construction/replay provider calls | 0 |
| retained transformer-token state | 0 bytes |

The 5.47 MiB artifact is about 400 times smaller than the failed 2.29 GB R6
artifact. An independent gold-blind audit recomputed 13,362 invariants and
3,805 receipts with zero errors. It verified all R/P token counts and hashes,
owner closure, complete leaf partitions, lossless G mappings, question
identity, the gold firewall, and 68/68 unique provider prompts. The same audit
also warned that a fixed eight-question evidence sample contained three strong
packets, one weak packet, and four materially noisy packets. Structural
integrity therefore passed; semantic precision remained an open empirical
question.

## R7 answer and judge result

Terra received exactly 68 unique question plus sealed-memory prompts. The
other 32 rows remained byte-identical V3 passthroughs. All request and response
journals closed with zero retries. Checkpoint-only materialization and replay
made zero provider calls and produced the same answer artifact SHA-256.

Terra returned 40 validated `keep_current` decisions, four validated
replacements, and 24 decisions that failed closed to V3. Thirteen failures
were exact-current answers that merely included evidence handles. Eleven were
replacement candidates rejected by the lexical
`unsupported_prediction_anchor` rule. Only three final predictions changed
from V3; one otherwise valid replacement was byte-equivalent to its parent.

The independent full-100 Sol judge then made exactly 100 unique calls with
zero retries. Judge and score materialization were checkpoint-only, and both
replayed byte-identically. The certified result is **88/100**:

- ordinal 31 changed from correct to incorrect;
- ordinal 50 changed but remained correct;
- ordinal 51 changed from correct to incorrect; and
- ordinal 82 kept the exact V3 prediction but the fresh Sol judgment flipped
  from incorrect to correct.

R7 therefore produced zero content-changing rescues, two regressions, and one
judge-only offset, for a net movement of 89 to 88. The remaining judged misses
are `14, 28, 31, 40, 49, 51, 53, 54, 67, 69, 94, 97`.

This is not evidence that the semantic search found nothing. A subsequent
gold-blind audit of the 24 rejected completions found that all 13
`keep_current_contract` failures were mechanically harmless handle-bearing
keeps. Among the 11 rejected replacement candidates, five were elementary
numeric/date/count derivations whose result need not occur verbatim in a
source quote, two were evidence-linked personalized syntheses, one was a
current-equivalent paraphrase, and three were genuinely unsafe/noisy. The
lexical subset rule therefore both over-rejected useful computed or
paraphrased answers and happened to reject some bad answers for the wrong
reason.

The next layer must not globally relax grounding. It should freeze every raw
replacement candidate, compare candidate versus protected current answer with
the complete bounded R/P packet, and let a gold-free semantic verifier select
one of those exact strings. Typed derivations remain locally executable;
personal claims still require user-role evidence; malformed or unsupported
verifier output fails closed. Questions for which neither candidate is
supported then proceed to the separately budgeted global-to-local semantic
search lane.

## Sealed R7 artifacts

| Artifact | SHA-256 |
| --- | --- |
| eligibility gate | `779c711e090ecb9faad92d9845158d939411dfa3a965669a26cfe8a8062fb912` |
| query vectors | `ce9b10803146a70ec18d9c907aceb2fa469fa5491818bc72721e7f5cefbcc8e2` |
| construction and byte-identical replay | `d0f226b1577a6bf40c54758d2fdc477ab98483613ca7c4fc77ef93383a651f6a` |
| Terra preflight | `52df0b0a4388ab2297a4af41b577839ab8bc1447df69cb49aa14017de3593bcc` |
| Terra answer and byte-identical replay | `de717ce73acad9d634f4639bea786bcae94843933d2acd882917c8ed2a25c2e2` |
| Sol preflight | `691132987c3abda35a46bbd9abda599583ea34db5f8d9530cb4d0734ddaf3981` |
| Sol judge and byte-identical replay | `907912c926d7963f89bf48e631494a9c4eff1df2cfc7deaae79a73834ac727f8` |
| Sol score and byte-identical replay | `6ea8abf1746a3a2df03815b98c09f1be6a9f1623f6b47fb38595fe07ff7afabc` |
