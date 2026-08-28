# Query answer joint failure taxonomy — 2026-08-27

## Result

The sealed query-payload arm scored **71/100** and the sealed query-fact arm
scored **64/100** against the same S0-v2 parent at **53/100**. They are jointly
wrong on 28 questions. This audit verified every answer run/replay, runtime
ledger/replay, judge run/replay, score run/replay, preflight, and exact S0-v2
parent binding before opening the locked references. It made **zero provider
calls**, reran no retrieval, and changed no answer.

The machine-readable result is
[`locked-query-answer-joint-failure-taxonomy-v1.json`](../10%20-%20Research%20Log/data/locked-query-answer-joint-failure-taxonomy-v1.json),
sealed at `1b977ce25616efc13b633ede476c041dbc9e79e0d2a562ee3f0a9851514a9003`
with internal analysis identity
`5a31513f533e0ebe51b24471377190987eaf84ae53a2c9a99786624872c074b7`.

| Dominant posthoc cause | Questions | Count |
|---|---:|---:|
| Operator failure despite registered-source coverage | 6, 14, 16, 27, 28, 40, 43, 52, 65, 67, 69, 79, 81, 82, 94, 97 | 16 |
| Partial multi-source coverage | 7, 31, 61, 77, 86 | 5 |
| Source missing | 36, 37 | 2 |
| Candidate reached but packing dropped it | 54, 93 | 2 |
| Answer shape or judge ambiguity | 53, 75 | 2 |
| Other: unsupported answer to a negative/evidence-insufficiency target | 42 | 1 |

The route distribution is 10 numeric, 9 temporal, 4 synthesis, 4 direct
extract, and 1 set join. The 16 nominal full-source failures contain 6 numeric,
4 temporal, 3 synthesis, 2 direct-extract, and 1 set-join question. The next
answer-stage gain therefore depends more on deterministic operators than on a
single additional generic synthesizer.

## Evidence boundary

The target registry uses short source IDs while answer packets use exact
question-namespaced IDs. The join accepts only `target_id` or the exact
`question_id::target_id`; suffix matching is forbidden. This correction is
material: treating the two forms as unequal falsely labels covered questions
as missing.

For the 28 failures, the actual query-payload packets have full registered
source-ID coverage on 18 questions, partial coverage on 6, and no coverage on
4. The query-fact treatment descends from the same admitted query neighborhood,
then changes its representation to facts; it does not establish an independent
retrieval denominator.

The guided target audit is incorporated at exact SHA-256
`329c8490ca2f090fa81c85cbc9999c07f539cc564c84bbaa590300d5f9c4ca34`.
The posthoc OR of protected S0, partition-v2, query-repack-v2, and guided
admissions is full on 24/28 questions, partial on q61, and empty on q36, q37,
and q54. This is a **prospective structural union only**: it was not submitted
to an answer model or judge.

Source-ID reach is not proof that the packed excerpt contains the decisive
answer span. Accordingly, “operator failure despite source coverage” means
failure after nominal registered-source acquisition, not proven model failure
over a perfect evidence packet. q53 is labeled answer-shape-dominant because
the direct prediction said “at least 3” where the judge required exactly 3;
its partial 2/3 source coverage is retained as a secondary cause. q75 similarly
used “over/more than $270” instead of exactly $270. q42 is kept out of the
operator bucket because the reference is an evidence-insufficiency answer and
all three arms hallucinated a university.

## Deployment policy, separate from the posthoc labels

The causal labels above use references and cannot be used as an online router.
Deployment instead uses the already sealed question-only route:

| Question-only route | Deployable sequence |
|---|---|
| Numeric reduction | exhaustive retrieval → source-balanced packing → numeric executor |
| Temporal timeline | exhaustive retrieval → source-balanced packing → timeline event table |
| Set join | exhaustive retrieval → source-balanced packing → set join |
| Recommendation/explanation synthesis | exhaustive retrieval → source-balanced packing → coverage-aware synthesis |
| Direct extraction | exhaustive retrieval → packing → constrained synthesis/direct extraction |

Every route should also enforce evidence sufficiency so a negative or
unanswerable target like q42 does not become a confident unsupported answer.
The class-level remediation map in the sealed artifact is diagnostic only: it
shows what would have fixed each observed failure class, while the route table
above is the deployable, gold-blind policy.

## Reproduction

```powershell
.pixi/envs/default/python.exe tools/analyze_locked_query_answer_joint_failures.py
.pixi/envs/dev/python.exe -m pytest -q tests/test_analyze_locked_query_answer_joint_failures.py
```

The analyzer refuses changed checkpoint hashes, verifies row and ledger
identity seals, and verifies answer/judge/score bindings before loading the
canonical locked LongMemEval validation references.
