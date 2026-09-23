# Append lexical summaries after preserved parent context

The best completed answer result entering this experiment is 93/100 (Terra and
Sol tie). Log 229 ruled out prepending a lexical reserve: it recovered one missing
statement while displacing other parent-context evidence. The following change
preserves the previous route before offering one additional atomic summary match.

## Implementation and checks

`search/native_spine_additive_lexical.py` calls the existing context router with
unchanged limits. It separately selects the top lexical match over atomic
summaries within the same dated source scope. An already-selected match changes
nothing. Otherwise its exact raw address appends after the complete original
route. It never enters the four parent seeds or their 16 context-atom slots.
The existing whole-section hydrator retains the 2,048-token / 128-span caps.

The typed additive receipt embeds the original context receipt and lexical plan,
validates the unchanged prefix, source scope and one-match limit, and binds all
fields. Receipt reconstruction rejects tampering. No raw text is scored and no
live Qwen pass occurs. The stored hierarchy remains driven by the user spine.

`application/native_spine_additive_lexical.py` provides an explicit opt-in
`AdditiveLexicalMemoryCondenser`. It reuses the normal snapshot admission and
public retrieval path, swapping only the resident router after admission. It
retains encoder, transcript-revision, close and persistence checks. The historical
implementation files remain unchanged so old sealed results remain verifiable.

Thirty routing/lifecycle checks passed in 5.47 s. Twenty-five evaluator,
additive-router and application-gate checks passed in 3.72 s. They include exact
prefix preservation, no duplicate addition, future-source exclusion, budget
exhaustion without truncation, malformed receipt rejection, ingest/close/reopen,
one fresh query embedding, and unchanged matched-control/evidence boundaries.

## All 100 packets, no answer calls

`tools/assess_native_spine_additive_lexical.py` reopened the existing application
once and freshly queried all 100 questions. Every packet sealed before references
opened. Every original route and hydrated section was preserved. Eleven queries
received an extra candidate; the unchanged budget admitted nine additional spans.

Recorded-support coverage increased from 97/100 complete to **98/100 complete**,
with two partial packets and no empty-support packet. Ordinal 35 gained its
missing original user statement; no question lost complete support. Median
conversation count remained four; median rendered tokens rose from 1,496.5 to
1,506.5. Independent reconstruction verified all 100 packets and 1,439 raw spans.
This is evidence coverage, not answer accuracy.

- Assessment root: `eval_results/native-spine-additive-lexical-20260915-r1`
- Exec session: `99222`, terminal exit 0.
- Report SHA-256: `f86881e23bcd36e4f2a74611ff89b9e2de77ca0eae6b3b9ac487b1f8755e32ca`
- Numeric policy: `dense-parent-2048-direct8-v1.json`, unchanged from the 93/100 Terra run.
- No new ingestion, Qwen compilation, question authoring or answer calls.

## Completed full answer comparison: 87/100

`tools/evaluate_native_spine_additive100.py` binds the new application/router and
the completed provider-free assessment explicitly. All 100 evidence packets must
match that assessment and preserve the prior 93/100 packet evidence before any
answer calls can start. It keeps the same v7 reader, Terra answer model, original
100 questions/references and Sol grading. It alternates 100 fresh memory answers
with 100 identical-prompt API controls. Live retrieval stays inside the memory
timer. Grading opens only after all 200 answers seal; independent raw audit follows.

- Run root: `eval_results/native-spine-app-additive100-20260915-r2`
- Log: `eval_results/native-spine-app-additive100-20260915-r2.log`
- Exec session: `5686`, terminal exit 0. Worker PID 38556 is finished.
- One already ingested history: 1,098,417 raw tokens, 5,357 raw turns.
- Reader policy: `user-coverage-v7.json`, unchanged.

The complete run scored **87/100**, down from 93/100. All 200 answers stopped
normally and reported the expected Terra alias. All 100 packets and 1,439 exact
raw spans passed independent reconstruction. A provider-free judge replay
(`93517`, terminal exit 0) reproduced the same report with 100 cache hits and
zero new judge calls. The routing addition is not promoted on answer accuracy.

| Measurement | Memory | Identical-prompt API |
| --- | ---: | ---: |
| Warm median total | 4.945 s | 4.383 s |
| Warm p95 total | 7.347 s | 8.652 s |
| Answers below five seconds | 52/100 | 67/100 |
| Mean total | 5.398 s | 5.020 s |

Median packet preparation was 0.269 s. Cold setup was 28.325 s, excluded from
warm serving. The median memory/API ratio was 1.128. The best completed accuracy
remains the previous 93/100, and the 95% target is unmet.

| Artifact | SHA-256 |
| --- | --- |
| r2 preflight | `f13e103cd03005b35d82f73ea69feee206cf4bb5f4efbe0489e7e430ef36a799` |
| Joint report | `59289b6d476944e8bb4c67356e1e4a1966e5cbfd259f6e43888f7ecf99c47de7` |
| Raw audit | `c5a7d11cbe4974fc30097834b7ac8cf6748d8c85dfc1f5e7b1bc177b3735202c` |
| Comparison | `93b1d008afd63c8a28cb9a7ee76c077ce67cfb59efbb063ed40743bebee5b2bf` |

Generated questions on real transcripts form an exposed development set, not an
official LongMemEval or generalization result. Known question/reference defects
remain documented and do not change the original grades.

The preserved r1 attempt (`98733`, terminal exit 1) completed all preflight
checks, then failed its first gateway connection with WinError 10013 (sandbox
socket permissions). It contains one reserved request and zero responses.
Its preflight SHA-256 is
`9d123c2baf1c5b1529f731eadb993c4ec2b5507da4616a33c9a6279849a3cc20`.
The r2 attempt uses the identical implementation and policies with the required
network sandbox escalation. It reuses the same persisted memory; no history was
rebuilt. The user's authorization for this local gateway and raw data persists.
Its sealed transport-failure receipt is
`fc28e36c768224efd502f77ce3498c61c00d7609f4de01ed5fe493a683951157`.

## What the regression does and does not show

Only nine complete answer prompts changed: 35, 36, 52, 69, 74, 75, 79, 82 and 93.
The other 91 prompts were byte-for-byte identical to the earlier 93/100 run.
Gains were 26 and 35. Losses were 6, 14, 65, 69, 81, 95, 96 and 98. **Seven of
the eight losses occurred on unchanged prompts.** A retrieval change cannot
explain those seven losses through the supplied context. Their generation or
grading changed; the identical model alias does not prove backend determinism.

The missing Telegram statement at 35 is now served and correctly answered.
The one changed-prompt regression, 69, omits the creation-story detail even
though the exact user statement remains in the packet. Extra context could
distract the reader there; this single comparison cannot establish causation.
The remaining misses are 6, 14, 19, 38, 53, 65, 69, 81, 82, 93, 95, 96 and 98.

The saved identical-prompt API controls illustrate answer variation without
another generation run. At 14, the control includes modern construction
technology that the memory answer omits. At 65 it preserves the tentative
specialist evaluation. At 81 it preserves the full opening's nightmare clause.
These are source-based comparisons, not a newly graded API accuracy score.

Several failures still concern the benchmark rather than absent raw evidence:

- At 38, “educational game” is explicit in the next original user turn, despite
  the judge calling it unsupported. This is the previously documented defect.
- At 82, the new answer includes complex characters/storylines and timely
  real-world commentary. The judge instead requires unasked show/character names.
- At 95, both the old passing prediction and new failing prediction mention
  French/international cinema, which the user really requested. The new verdict
  rejects it because the gold answer omits it.
- At 96, all four named apps are explicit user statements. The question's
  household-shopping situation points toward Ibotta/Fetch; choosing the other
  conversation as well is a scope-selection problem, not an invented app name.
- At 98, the requested title appears in the user text, but served assistant
  context calls it a song while the question calls it an album. The memory answer
  abstains; its API control supplies the title and flags that premise conflict.
- The multi-conversation ambiguities at 53 and 93 remain unresolved.

No result is regraded or combined across runs to manufacture 95%. This experiment
establishes an evidence-preserving recall repair, not improved answer accuracy.
Do not continue sweeping retrieval budgets to explain failures on identical
prompts. The next useful work targets reader completeness/scope consistency and
keeps benchmark defects separate. Raw-text attention pruning is not established
as necessary by this evidence. Preserve the existing 93/100 baseline, the new
opt-in implementation and all receipts; no reingestion is needed.
