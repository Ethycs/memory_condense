# Browsing access follow-up on research failures

**Status:** Complete — three paired follow-ups finished; workers closed; baseline preserved.  
**Date:** 2026-09-25  
**Applies to:** Matched engineering/research artifact evaluation.  
**Depends on:** [Completed battery](258%20-%202026-09-25%20-%20Matched%20engineering%20and%20research%20battery.md), [methods](../../evals/engineering_research/README.md).

## Question and correction

The completed battery gave both actors identical file and test tools, plus a
memory recall action, but no browsing. Its adapted research task also explicitly
forbade claiming externally verified literature. This was a restriction in the
evaluation harness, not an architectural requirement of the memory system.

The user requested browsing for both arms and a check of whether its absence
explained failures. The new optional `web_search` and `web_open` actions expose
public search and pages through a journaled controller bridge. Candidate code
execution remains restricted to its workspace; the browsing tool is separate.
Both arms receive identical browsing instructions, limits and tool responses
for their own requests. Historical evidence remains arm-specific.

## Results

**All six actors finished, but none requested browsing or recall.** Each issued
one `write_many` action followed by `finish`. All three memory stores completed
final ingestion and a separate-process reopen. Initial retrieved packets were
byte-identical to the corresponding baseline packets. All six ingestion receipts
report summary-only Qwen input; all three routing receipts report zero raw reads.

The known memory omissions remain in the new artifacts:

| Case | Memory content criteria | Full-context content criteria | Observed difference |
|---|---|---|---|
| R04 | 2, 0, 2, 2 | 2, 2, 2, 2 | Memory omits the 57.23% versus frozen 60% endpoint; full context preserves it |
| R09 | 2, 1, 2, 2 | 2, 2, 2, 2 | Memory calls RES undefined; full context reconstructs Relation, Exception, Scope and the global-relation premise |
| R10 | 2, 1, 2, 2 | 2, 2, 2, 2 | Memory says GR, its space and measure were never defined; full context distinguishes the original construction from the later realizable-goal claim |

Scores are 0 unmet, 1 partial and 2 met. Both reviews agree on these scores;
one review per pair verifies after the existing quotation-format normalization.
R10 C3 receives 2 here versus 1 in the baseline, despite the continuing absence
claim and unchanged retrieved packet. That scoring variation is not a recovery
attributable to browsing. The full-context construction audit also does not
certify the archived mathematics; it identifies defects in that construction.

Content ratings are separate from output validity. All three memory outputs have
valid exact historical citations (23, 21 and 11 respectively). R04 and R10 full
context have invalid JSON escapes in `claims.json`; R09 full context has
non-exact source quotations. Consequently none of the six attempts passes the
entire strict task gate. The original strict follow-up report records memory as
two failures/one unresolved and full context as three failures; its reviewer
quotation errors remain unchanged. The offline assessment resolves the content
ratings separately without rewriting that strict report.

**Interpretation:** enabling browsing alone did not repair these three observed
memory omissions. This run cannot estimate the effect of actual web evidence,
because no actor used the web tool. The source audit traces the misses to facts
in the historical session and to the reader not requesting recall. No inspected
miss is demonstrated to be caused by missing public information. A later reader
change should retrieve the referenced definitions, manuscript, or experimental
criteria before concluding they are absent.

All three previous full-context timeouts recover on this follow-up without any
web use. This establishes completed matched comparisons for these selected
cases; it does not establish why the earlier requests timed out or a general
research performance rate.

Saved results: [strict report](../../eval_results/engineering-research-web-20260925-r1/report-03.json),
[offline assessment](../../eval_results/engineering-research-web-20260925-r1/assessment-03.json),
[final overview](../../eval_results/engineering-research-web-20260925-r1/final-overview.json).

## Baseline failure audit

No saved failure is yet demonstrated to depend on a missing public fact. This
is a diagnosis of the observed misses, not evidence that browsing is useless.

| Failure | What is needed | Can public browsing directly supply it? |
|---|---|---|
| R01–R03 memory ingestion | Exact support quotation accepted before the actor starts | No; actor tools have not run |
| R04 memory threshold omission | The private 57.23% versus frozen 60% endpoint at T0127 | Public statistics references cannot establish the archived measurement/threshold |
| R09 memory RES omission | User definitions at T0037/T0039 | Public literature can inform criticism, but not establish this user's definition |
| R10 memory construction omission | Supplied paper preamble and session definitions | Public measure theory can help audit a construction once retrieved |
| Research generation timeouts | A completed actor response | Cause unresolved; no evidence missing browsing caused the timeout |
| Exact citation failures | Literal historical quotation formatting | Public URLs cannot repair quotation membership |
| Engineering misses | Source requirements, implementation fixes, or rubric clarification | No demonstrated missing external API/library fact |

The source-bound audit is saved in
[baseline-failure-audit.json](../../eval_results/engineering-research-web-20260925-r1/baseline-failure-audit.json).

## Bounded follow-up

Run **R04, R09 and R10 in both arms**, all three completed memory research
outputs with substantive semantic misses. These are six new attempts, not a
replacement of the twenty-pair baseline. The earlier full-context attempts on
these cases timed out, so their new completion would not itself prove a
browsing-related quality improvement.

- Preserve source cutoffs, model routes, decoding, 24 actor actions, 240-second
  API timeout, requested 4,096 output tokens and historical criteria.
- Permit at most four web actions per arm within the existing action limit.
  Use original papers, author pages and official documentation. The actor
  chooses whether to browse; lack of a call is reported rather than hidden.
- Relax the adapted task's prohibition on externally verified literature.
  Public citations go in `analysis.md`; `claims.json` continues to bind
  historical claims to exact supplied source turns.
- Reuse the existing source compilation cache. Ingest the original prefixes
  into fresh stores, reopen them separately, and persist/reopen new work using
  the existing memory lifecycle. Qwen continues to receive summaries only.
- Preserve all requests, responses, tool results and errors. Do not retry
  failed attempts until passing or silently change historical scores.

The [run plan](../../eval_results/engineering-research-web-20260925-r1/run-plan.json)
freezes actor ceilings of 144 calls / 12M input tokens, raw-summary ceilings of
80 calls / 560k input tokens, merge ceilings of 160 calls / 327,680 input tokens,
and grading ceilings of six calls / 786,432 input tokens. These are ceilings,
not requested consumption. Reused baseline compilation is not billed again.

## Implementation checks

Twenty-six focused harness and assessment tests pass, covering matched actor instructions,
relaxed literature restrictions, unchanged offline defaults, public URL
validation, web budget enforcement, response/request binding, restricted
candidate execution, and existing generation accounting.

Implementation: [web bridge](../../tools/engineering_research_web.py),
[runner](../../tools/run_engineering_research_battery.py),
[tests](../../tests/test_engineering_research_web.py).

A separate live smoke test successfully round-tripped a primary-source search
through the same bridge. It is stored under `diagnostics/web-bridge-smoke` and
is excluded from actor call counts and outcome claims.

## Runtime correction

Both R04 actors completed, but the first grading request was rejected locally:
JSON escaping of the full history plus both new artifacts produced 133,210
prompt tokens against the unchanged 131,072 cap. No provider call was made for
that rejected request. Amendment 001 renders every historical turn directly
instead of JSON-escaping it. The corrected request has 111,040 tokens; a
preflight verified every original source text is present exactly. Actor prompts,
answers, criteria, output limits and source cutoffs are unchanged. The rejected
request remains in the journal and the completed actor answers are reused.

## Accounting and closure

The run makes **38 reserved model calls**: twelve actor, three raw-summary,
seventeen summary-merge and six grader calls. The one additional locally rejected
grader request is not a provider call. All frozen call and input-token budgets
pass. Four actor responses exceed the requested output-token cap, an existing
gateway behavior retained in the report.

| New-call operation | Accounted input tokens |
|---|---:|
| Memory actor | 26,384 |
| Full-context actor | 421,644 |
| Raw summaries for new work | 14,184 |
| Summary merges for new work | 12,748 |
| Grading | 462,198 |

Counts mix positive provider usage with saved-prompt estimates where usage is
missing. Previously compiled baseline summaries are reused, so this is not a
total ingestion-cost estimate. The unequal content ratings also preclude
treating these totals alone as an equal-quality cost comparison.

The original twenty-pair assessment hash is unchanged. The follow-up controller
and generation worker exited successfully, the stop marker is present, and no
actor/browser request remains pending. No baseline answer or score was replaced.
