# Full100 hierarchy failure assessment

**Status**: Measured failure; current hierarchical query policy rejected
**Date**: 2026-09-12
**Applies to**: Summary-only Qwen tree routing with exact raw-section hydration
**Depends on**: [Research Log 173](173%20-%202026-09-10%20-%20Restored%20attention%20topology%20and%20bounded%20hierarchy%20routing.md), [Research Log 180](180%20-%202026-09-11%20-%20Local%20Qwen%20JSON%20encoding%20recovery.md)

## Measured outcome

The completed hierarchy scored **8/100**, compared with **84/100** for the
unchanged flat control. Both the accuracy and latency gates failed. All ten
memories exceed one million `cl100k_base` tokens in complete raw turns, excluding
chat framing. The minimum is 1,039,792 tokens. This is the public tokenizer
count, not a claim about a private provider tokenizer.

| Arm | Accuracy | Median total | p95 total | Median preparation |
| --- | --- | --- | --- | --- |
| Flat control | 84/100 | 5.476 s | 9.346 s | 0.352 s |
| Hierarchy | 8/100 | 6.463 s | 8.523 s | 1.656 s |
| Identical hierarchy evidence sent directly to API | Unjudged timing control | 4.490 s | 8.062 s | <0.001 s |
| Short API prompt | Unjudged timing control | 3.942 s | 4.788 s | <0.001 s |

Hierarchy median total latency is 1.439 times its identical-evidence API control
and 1.639 times short chat; the permitted ratio is 1.10. Its p95 ratios are
1.057 and 1.780 respectively. Visible first-token medians are 6.462, 4.489 and
3.942 seconds for these three arms and also fail the joint allowance.
Seven questions are correct in both memory arms, 77 only in the flat arm,
one only in the hierarchy arm (ordinal 42), and 15 in neither.

All 400 fresh serial Terra answer streams ended normally before references
were opened for judging. The 200 logical Sol judgments required 196 physical
calls after deduplication. The joint report replayed identically with zero new
judge calls and 196 replay hits. The timed responses are the scored responses.
Cold index/model setup is reported separately from warm query latency; live
embedding, routing, projection, hydration and rendering are inside that timer.

The compiler and handoff have exited 0. All ten completed hierarchies have also
been reconstructed independently from saved outputs with generation forbidden.
The tenth proof ran after timed evaluation, preserving an idle measurement
workspace. No further parent compilation is needed for this population.

## Where support is lost

The postseal audit authenticates the completed report, all 100 saved evidence
packets, their parent hierarchies and the selected native validation histories.
Annotated support is matched by exact session, timestamp, role and full-turn
text hash, then by the union of raw character spans. A different record's
prefix is not itself evidence of an irrelevant or false turn.

| Stage | Any annotated-turn overlap, out of 97 | All annotated turns fully covered, out of 97 |
| --- | --- | --- |
| Original leaf population | 97 | 97 |
| Eight dense-nominated root candidates | 94 | 83 |
| Four roots retained by Qwen | 62 | 23 |
| Final leaves retained after descent | 9 | 3 |
| Actual exact raw hydration | 9 | 3 |

Three questions have no annotated support turns and are excluded from these
coverage denominators. Thirty-two questions lose all overlap at root pruning,
and another 53 lose it during descent. Three already lack it in the dense root
shortlist. Hydration introduces no additional complete loss of overlap.
Coverage is a diagnostic, not semantic sufficiency or answer accuracy.

The hierarchy produces an exact "I don't know" abstention on 84 questions,
versus four for the flat control. Its median literal user-evidence character
share is 10.75%, versus 60.14% for the flat packet; these are character shares,
not token shares. Median context size is 1,523.5 versus 2,959 tokens. The Qwen
traversal needs five passes at the median and seven at most.

For ordinal 0, the first dense root is the correct source. Its stored parent
summary explicitly names the Philips LED bulb in the bedside lamp, and Qwen
keeps that root. The next pruning round loses its annotated support completely.
The final packet instead includes unrelated living-room lampshade and kitchen
lighting advice; the reader abstains. The flat reader answers "Philips LED bulb"
correctly. Rebuilding this parent summary cannot explain or repair that example:
the necessary fact was already present before branch pruning.

## Consequence for the next change

Retain the flat packet and reader as the serving baseline. The failed policy
uses dense scores only to nominate roots; repeated Qwen pruning can then discard
every strong dense match. Investigate the scoring signal before another full
answer evaluation. Compilation integrity and structural routing tests did not
establish that this attention score measures question relevance.

The local query path uses six Qwen blocks, with attention readout at layer 5,
FP16 weights/forward, and FP32 softmax and pooled readout. It reads summaries
only. The full local Qwen model used for ingest generation is a separate path.
The gateway alias's physical backend remains unverified; direct checkpoint
loading establishes locality for these runs independently of that alias.
A read-only request to the authorized gateway's `/model/info` endpoint on
September 12 returned HTTP 403 (chunk `f2a677`). It sent no inference prompt
and exposed no routing metadata. This service response does not identify the
alias's host; no attempt was made to bypass its access restriction.

## Attention-score diagnosis

The scorer reads attention from fixed `[Readout]` marker tokens after the
question. It uses the mean of the four largest per-head, length-normalized
memory logit scores. Those uncalibrated scores are compared across independent
query/summary rows; they do not directly measure answer relevance.

A synthetic probe used eight unrelated user facts and eight corresponding
questions, with both plain and structured summary representations. The raw
score ranked the matching fact first in 3/8 plain cases and 7/8 structured
cases. Subtracting each candidate's score under the fixed neutral question
"What information did I share?" improved these to 5/8 and 8/8 respectively.
Thus the readout responds to the question, but summary presentation also
strongly affects ranking. These toy cases are not benchmark accuracy.

The same subtraction was then tested on the first two saved frontiers of ten
evenly spaced validation questions (ordinals 0, 10, ..., 90). All 20 original
selections reproduced exactly. Forty local summary-only inference batches
were sealed before opening support annotations; no reader or judge was called.

| Diagnostic across the 20 fixed frontiers | Any support overlap | All annotated turns fully covered |
| --- | --- | --- |
| Available candidates | 16 | 13 |
| Original/raw-score top four | 8 | 4 |
| Neutral-subtracted top four | 9 | 6 |

This is insufficient evidence to promote neutral subtraction. The probe uses
the original frontiers, so it does not measure the downstream tree that changed
choices would produce. No successor full100 answer evaluation was launched.
Avoid another parent rebuild: the demonstrated loss is mainly selection, and
this score adjustment does not repair it reliably. A successor must preserve
strong summary-retrieval matches and useful user evidence while reducing query
passes; that behavior has not yet been implemented or validated.

Synthetic probe: `tools/probe_summary_attention_question_sensitivity.py`, result
root `eval_results/summary-attention-question-sensitivity-20260912-r2`, SHA-256
`3703c3bde0d8a08b64e8ff373f3babefcaa817c4f1548c100f6f2fc41ea0db4b`.
Its 18-batch execution exited 0 and the score reduction replayed with no model
load or calls. The first attempt sealed identical ranking findings before an
invalid cleanup-method call; its original implementation is retained in the
`r1` artifact root. The corrected run uses process-owned model cleanup.

Real-frontier probe: `tools/probe_hierarchy_neutral_attention.py`, result root
`eval_results/hierarchy-neutral-frontier-probe-20260912-r1`, diagnosis SHA-256
`406574e46cf9b4043cfc92607ce55544a3746b23be4a337f81d0dd171988ceaa`.
Its 40-batch execution exited 0, and the complete diagnosis replayed identically
with zero new model batches (chunk `83c63f`). Both probes preserve scalar scores
and exact summary references, with no retained transformer token state or raw
Qwen input. All 53 implementation files bound by the completed full100
evaluation remain unchanged; `git diff --check` passes (chunk `b68746`).

## Reproduction and artifact bindings

Evaluation root: `eval_results/full1m-hierarchical-spine-joint-full100-20260911-r1`.
Audit root: `eval_results/full1m-hierarchy-route-support-audit-20260912-r1`.

| Artifact | SHA-256 |
| --- | --- |
| Evaluation preflight | `d6b270641d9be16ae47d7e9e286d6945b3013b8da0eaad174b3e29bd830af60b` |
| All 400 answers | `7ddc80e8115469b85b7dcb24f051fff89c95e557104907cb57a62d7d975d017f` |
| Joint report | `30a01eafdf0ad267c131694e2ef57d9d76c157598675758d8242b50636929c52` |
| Evaluation complete | `69dc0f8a6dadd5fa8d61d63075c44d0e9d335c60acd986a4a651ad08a2749594` |
| Route-support audit | `dbd628f6244b6cd6826d953ceaa44c4a2f3ce0640f5384b5655da454aca676a9` |
| Tenth independent hierarchy proof | `fa8683dba410ccba3609350e958407672589a71e71b2743b172620ef27d43a74` |

`tools/audit_hierarchical_spine_routing_failure.py` makes no model calls. Its
sealed full100 audit replayed identically. Four explicit checks verify absent
evidence, duplicate partial spans, adjacent fragments and wrong-turn hashes;
all passed. Benchmark annotations are opened only after answers have sealed
and are never router inputs. The separate 200-question confirmation population
is not used in this assessment. No production or frozen evaluator file was
changed to generate these findings. The 95% joint target remains open.
