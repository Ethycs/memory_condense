# Role-aware million-context retrieval

Date: 2026-08-17

Status: selected development policy; not validation and not a pass of the 95%
target.

## Result

The locked ten-question, 1,039,203-token LongMemEval stress run reached **70%
judged operational accuracy**, 50% exact match, and 0.634 mean token-F1 while
returning 1,908 mean evidence tokens. This is approximately 545x context
compression and 99.82% transcript-token reduction.

The previous best was 30% judged accuracy. Its most efficient arm returned
1,377 evidence tokens; an earlier 30% arm returned 2,108. The new treatment
therefore spends 531 tokens more than the smallest 30% arm, but remains below
the earlier equal-accuracy packet while adding four correct answers.

| Locked development arm, ten questions | Literal recall | Mean source coverage | Judged accuracy | Mean F1 | Returned tokens |
|---|---:|---:|---:|---:|---:|
| Metadata + role labels in prompt (v8) | 40% | 78.0% | 30% | 0.295 | 1,377 |
| **Role-aware retrieval (v12)** | **50%** | **93.0%** | **70%** | **0.634** | **1,908** |

The responder and judge remained `codex_sdk/gpt-5.6-terra` and
`codex_sdk/gpt-5.6-sol` through the central-dev v1 gateway. The operational
run was capped at ten answer calls and ten judge calls. Mean total prompt
content was 2,216 tokens, p95 2,554, below the 8,000-token hard cap.

## Mechanism

Long-chat autobiographical questions ask about what the user did, preferred,
or updated. Similarity retrieval previously treated assistant suggestions and
user statements as equally authoritative. At million-token scale, long and
topically broad assistant answers frequently outranked the short user fact.

The treatment activates only when the query contains a first-person pronoun.
It multiplies transient candidate scores by role:

- user: 1.25;
- assistant: 0.75; and
- system: 0.50.

The adjusted scalar ordering is applied before anchor/source activation and
again inside source-local and HSC candidate selection. Durable embeddings,
chunks, graph edges, and heat are not rewritten. No role-conditioned text,
attention state, or second transcript copy is persisted.

This is a retrieval prior, not an answer-stage classifier. It is also bounded:
the candidate and returned-context limits are unchanged.

## Questions recovered

The treatment produced seven judged-correct answers:

1. Miss Bee Providore;
2. Serenity Yoga;
3. four days between the racing events;
4. the nursery, baby-shower, phone-case ordering;
5. Hawaii;
6. 190 pages left in *The Nightingale*; and
7. the current Instagram count, close to 1,300.

The Hawaii result is the clearest causal example. The labeled user turn nearly
repeats the question: the user loved Hawaii after going there with family for
a week. Without the role prior, generic assistant travel recommendations won
candidate competition and the labeled source had 0% packed coverage. With the
prior, every question retrieved at least one labeled source.

## Rejected arms

The following local arms did not justify provider calls:

| Arm | Literal recall | Source coverage | Best token-F1 | Returned tokens | Decision |
|---|---:|---:|---:|---:|---|
| Explicit colon-list facets (v9) | 40% | 78.0% | 0.149 | 1,392 | Rejected; no recall gain |
| Pure lexical ranking (v10) | 40% | 75.7% | 0.125 | 1,411 | Rejected; harmed every soft-F1 row |
| Wide workspace without role prior (v11) | 40% | 78.0% | 0.149 | 1,278 | Rejected; no routing gain |
| Wide role-aware workspace (v13) | 50% | 91.3% | 0.155 | 1,934 | Rejected; lower coverage than v12 |
| Multi-fact source round-robin (v14) | 50% | 89.7% | 0.154 | 1,849 | Rejected; museum coverage fell from 3/6 to 1/6 |

The negative results matter. Retrieval breadth, BM25, and unconditional source
diversity all admit more distractors. The useful signal was semantic authority:
who stated the candidate fact.

## Remaining three failures

- Concert ordering retrieves four of five labeled sessions, substitutes The
  Killers for Queen + Adam Lambert, and misorders Billie Eilish.
- Sculpting retrieves both labeled sessions but answers nine weeks instead of
  three. This is a temporal derivation/annotation ambiguity rather than a
  source-routing miss.
- Museum ordering retrieves three of six labeled sessions and abstains.

The next improvement should target temporal set completion without globally
diversifying sources. A promising direction is a bounded event ledger built
from role-aware user candidates: normalize each candidate into
`(event/entity, source timestamp, chunk ID)`, then request missing set members
iteratively. The ledger must remain transient and store only durable IDs plus
scalars between retrieval rounds.

Machine-readable metrics and hashes are in
`data/longmemeval-million-context-role-aware-development-v1.json`.
