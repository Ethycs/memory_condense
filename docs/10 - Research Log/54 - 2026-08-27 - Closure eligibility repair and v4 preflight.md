# Closure eligibility repair removes a pre-retrieval recall ceiling

**Status:** eligibility repair retained; v4 preflight superseded before any
question artifact by the stronger v6 protocol in Research Log 56.

The first independent representative-bridge and artifact-global closure
campaign was stopped before it wrote any question artifact. Its v3 eligibility
rule admitted only questions whose top-level route reason was temporal order or
whose route requested a complete frontier. That condition was too narrow: it
excluded relative-time questions that request temporal metadata but whose
top-level route reason is not `TEMPORAL_ORDER`.

This was a real mechanism-coverage defect, not an evaluation nicety. Four
excluded questions owned nine of the 23 artifact-global source targets in the
locked target registry. The v3 population therefore capped artifact-global
primary-target recall at 14/23, or 60.9%, before retrieval ran.

## Gold-blind correction

Eligibility remains derived from question text alone. Version 4 admits a
question when either routed modifier is true:

```text
requires_temporal_metadata OR requires_complete_frontier
```

This changes the eligible population from 57 to 79 questions and restores the
four omitted relative-time questions at ordinals 6, 21, 43, and 93. No gold,
reference answer, benchmark category, source label, answer prediction, or
judge verdict enters the runtime eligibility decision. The posthoc target
registry was used only to quantify the consequence of the already-observed
question-only defect.

## Sealed v4 preflight

The corrected preflight reopened and byte-verified all ten existing stores
read-only. It rebuilt neither corpus nor store and made no retrieval or answer
provider calls.

| Field | Sealed value |
| --- | --- |
| question population | 100 |
| eligible retrieval population | 79 |
| planned retrieval invocations | 79 |
| answer-provider calls | 0 |
| corpus/store rebuilds | 0 / 0 |
| eligibility manifest SHA-256 | `cc0ffc946ccce84e577c877c983146723e6310d4bd5f22e6eeda0d30ebd438fb` |
| preflight SHA-256 | `ab9e66a23180e861418e7975915bda055107b0d99ce01ea876877dc8a864c2e7` |

The first shard contains seven eligible questions. Each question performs one
fresh cumulative retrieval pass and independently projects both protected
arms against exact S0. Each arm selects under its own 2,048-token allowance,
records selection before deduplication, removes exact S0 overlap only after
selection, and cannot borrow from the other arm.

Version 3 is retained as a superseded preflight record only. Because it was
stopped before producing per-question retrieval artifacts, no v3 result is
mixed into v4.

## Regression coverage

Focused tests prove that the four omitted relative-time questions now route
through the question-only v4 predicate, a timeless point lookup remains
ineligible, and reprojecting the real v3 manifest produces exactly 79 eligible
questions including all four corrected ordinals. A separate posthoc-only audit
also proves that the question set covers all 51 representative/global-owned
source targets (28 representative and 23 global) without loading that registry
into runtime routing. The focused closure suite passes 21/21.

This repair does not claim an accuracy gain. It removes an eligibility ceiling
so the isolated bridge/global arms can be measured honestly; answer and judge
results remain pending. The first v4 retrieval attempt then exposed a separate
timing-contaminated S0 receipt check and failed closed before publication.
Research Log 56 records that diagnosis and the independently audited v6 gate.
