# Routed numeric EM repair reaches 57/100

**Status:** measured answer-time development treatment; the numeric route
passes its positive-marginal gate narrowly, while the preregistered 95/100
target remains failed.

The first routed full-source repair increased independent Sol accuracy from
**56/100 to 57/100**. It rescued three previously wrong answers and regressed
two previously correct answers, for a **net +1**. This is a real paired gain,
but it is small: it does not support replacing the cumulative retrieval stack,
claiming the 84-question oracle ceiling, or claiming generalization.

## What was tested

This was an answer-time EM representation treatment over the existing sealed
S1 evidence. It did **not** retrieve another row, rebuild the corpus, change
S0--S3, or repack the historical retrieval artifact.

The runtime route was inferred from each dated question alone. It classified
32 of 100 questions as `numeric_reduce`; the other 68 retained their sealed
baseline predictions. For each eligible question, the treatment:

1. verified the exact protected S0 prefix and unchanged sealed S1 projection;
2. formed EM only after selection as `S1 - S0`;
3. asked Terra to turn that delta into atomic, exact-quote-cited numeric facts;
4. submitted protected S0 plus the valid facts, without a raw EM tail, to a
   numeric answer prompt; and
5. preserved the sealed baseline prediction whenever fact compression was
   empty or invalid.

Gold answers, benchmark categories, labeled answer sources, baseline
predictions, and judge verdicts were absent from routing, compression, and
answer generation. Gold was loaded only after `run.json` was sealed, for local
scoring and the independent Sol judge.

This boundary matters: the result measures whether a different representation
helps the answer model use the neighborhood already selected by retrieval. It
is not a retrieval-recall improvement.

## Calls and budget

| Phase | Eligible or submitted | Result |
| --- | ---: | --- |
| question-only numeric route | 32 | 32 compression prompts sealed |
| Terra fact compression | 32 calls | 19 valid, 12 empty, 1 invalid |
| Terra numeric answer | 19 calls | 13 eligible questions fell back to baseline |
| changed sealed predictions | 11 | eight answer calls reproduced the baseline prediction |
| independent Sol judgment | 11 calls | the other 89 verdicts were reused from the sealed baseline judge |

All provider paths used zero retries. The run therefore made **51 sealed Terra
calls** (`32 + 19`) and **11 Sol calls**. Replay made zero provider calls.
Empty or invalid compression was an explicit preservation decision, not an
abstention: those 13 questions kept their previous answers and received no
answer-model call.

| Prompt population | Minimum | Mean | Maximum | Hard cap |
| --- | ---: | ---: | ---: | ---: |
| Terra compression, 32 prompts | 1,219 | 2,725.31 | 3,446 | 8,000 |
| Terra answer, 19 prompts | 3,229 | 3,474.95 | 3,843 | 8,000 |

The 19 accepted compressions contained 39 validated facts. No prompt attached
an unbounded raw EM tail, and no retained transformer request state was
reported.

## Paired result

| Metric | Sealed baseline | Routed candidate | Change |
| --- | ---: | ---: | ---: |
| independent Sol semantic accuracy | 56/100 | 57/100 | +1 |
| normalized exact match | 33/100 | 35/100 | +2 |
| mean token F1 | 0.447494 | 0.455494 | +0.008000 |

Within the 32 eligible questions, semantic correctness moved from 15 to 16.
Only changed predictions were re-judged, so the semantic marginal is paired to
the same baseline rather than confounded by re-judging all unchanged answers.

| Outcome | Ordinal | Question ID | Compression |
| --- | ---: | --- | --- |
| rescue | 34 | `d682f1a2` | valid |
| rescue | 35 | `157a136e` | valid |
| rescue | 75 | `2318644b` | valid |
| regression | 87 | `2788b940` | valid |
| regression | 92 | `078150f1` | valid |

All five changed verdicts are in the posthoc
`numeric_aggregate_compare` diagnostic cell. Two rescues had nominal full S1
source coverage, one rescue had only partial labeled-source coverage, and both
regressions had previously correct answers with nominal full coverage. That is
why the result supports operator-specific representation as one repair, but
also requires a preservation gate: valid cited facts do not by themselves
prove that the derived calculation is better than the baseline answer.

## Method-budget implication

Yes: each retrieval method should have a separate protected budget, but the
budgets should be **bounded and asymmetric**, not equal slices.

Each method needs its own candidate quota, packed-token ceiling, provider-call
cap, fallback rule, and paired marginal. S0 keeps a non-borrowable control
reserve. EM, episodic/Hebbian, and CAV/link evidence receive bounded protected
allocations appropriate to their intended failure cells. Only unused capacity
may enter a final bounded residual pool. A question-only route may shift
allocations inside declared limits; it may not silently consume the S0 reserve
or attach an unbounded tail.

Answer operators need the same isolation. In this run, the numeric method was
charged exactly 32 compression calls, 19 dependent answer calls, its own prompt
token ledger, and 11 changed-prediction judge calls. The fallback firewall kept
13 unusable compressions and all 68 nonnumeric routes from consuming answer or
judge budget. This made the +1 marginal attributable and prevented one noisy
method from crowding out later methods.

The budget alone does not solve quality. The two regressions show that the next
numeric refinement should improve operand sufficiency and candidate-acceptance
checks before composition. The positive R2 gate permits work on the remaining
routes as separate arms; it does not justify giving them a shared raw context
or accepting any route without its own positive marginal.

## Replay and immutable artifacts

| Artifact | SHA-256 |
| --- | --- |
| sealed retrieval input | `e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f` |
| sealed baseline Terra answers | `d7fc47b8d1f372f002230c6ffe489dac8cd11bd71b35b8d3008b1255da2a38cd` |
| sealed baseline Sol judge | `5dc56a240315c5577d1032d40429df7e39adad0f40a098abc371ee2ea2ec77df` |
| route plan | `11ff958c4a9a4c46fc67671775c20de86be04021e619a87157d0a5bb39a07972` |
| provider-free preflight | `e176e5e2d4b994ad91cdd640333ae730a27ecf828d90071a7f044a50d2fb6e2a` |
| compression and zero-call compression replay | `92860130792a9e450188a58b257652ea2363f06b3693b03aa46c007559b9cff2` |
| treatment run and zero-call run replay | `793a487b7a16b5ce3c6acc072abcee15001ca6a2c30d58fcb2a0cc159582ad8f` |
| local score | `c854ce879619c30f73b262fc4cb30ba4b81339f5de2912ad1889a7629c801d05` |
| Sol judge and zero-call judge replay | `84cc3d0ccd69fc6b690faa639217b0648451d16367a3ccdf9fea2b7f52a49962` |
| [tracked flattened 100-row result ledger](data/longmemeval-locked-100-routed-numeric-repair-v1.csv) | `fc032cdf388158fc93d0aa7553ea3dc5f02132ba5f224dfe7657f94c3047fb46` |

The compression, answer, and judge artifacts each replayed byte-identically
from their immutable request/response journals. The flattened ledger contains
route eligibility, per-question budgets, compression status, fallback state,
prediction-change state, paired local and semantic outcomes, and content
hashes; it does not duplicate raw questions, references, or predictions.

## Interpretation and next gate

R2 satisfies the roadmap's minimum rule of a positive net semantic marginal,
but only narrowly. The next step is not a larger shared packet. It is to keep
numeric repair behind its fallback and regression ledger, then test direct,
synthesis, timeline, set, and state routes one at a time with their own
budgets. A route enters bounded composition only after it shows a positive
paired marginal.

The locked 100-question population is now analysis-used. The 57/100 result is
development evidence, remains far below 95/100, and requires an untouched
confirmation population before any generalization or competitive claim.
