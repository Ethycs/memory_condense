# Full100 passage gate and support syntax recovery

**Date:** 2026-09-10  
**Status:** checkpoint superseded by Log 144; fresh comparison complete  
**Predecessor:** [142 - Summary passage addresses and preserved user evidence](142%20-%202026-09-09%20-%20Summary%20passage%20addresses%20and%20preserved%20user%20evidence.md)

**Continuation:** [Log 144](144%20-%202026-09-10%20-%20Fresh%20passage%20routing%20development30%20result.md)
records the completed 26/30 versus 25/30 comparison, zero-call judge replays,
54 passing focused checks and the fourth memory's cleared schema audit. The
running-session and pending-test notes below describe the earlier checkpoint.

The preceding goal turn made concrete progress: all three complete memories
received passage addresses, all 30 development evidence packets retained their
previous user evidence, and 150 matched requests were prepared and replayed.
The latest judged score remains 25/30. The current work does not claim new
accuracy or a full100 pass.

## Full100 report

`tools/report_joint_source_spine_facets_full100.py` requires ten complete
namespaces, both full100 method populations, matching query policies, matching
hierarchy and admission methods, and a common passage compilation method.
It reconstructs every expected summary passage and authenticates the persisted
matrix without loading an encoder or making model calls. It rejects missing
passages even if someone regenerates their manifest hash.

Accuracy must reach at least 95/100 for one complete method. Median and p95
visible TTFT and total response latency must each fit the provisional 10%
allowance against both short API chat and the identical-evidence API control.
The report rejects duplicate question identities, nonfinite or nonpositive
measurements, time-to-first-token later than total time, and a relaxed latency
allowance supplied to the reporting function.

The first verification of all three prepared memories reproduced their
complete admission artifacts and the common method-v4 identity. Artifact:
`eval_results/full1m-source-spine-overflow-development30-20260910-r1/facet-full100-method-verification.json`,
SHA `b85db17d1f6629fc0d909885972ba995b5806350ddfbf26b3beb2fee2ba14549`.
Their common passage policy is
`bf856860fc173039fe26ef53dcf88090495fa20bfdb271ca19d5739ed3f48816`.
This verifies three memory implementations, not ten answer populations.

## Newly observed support syntax fault

Fourth-memory ingest completed in session **76404**. Inspection of completed
responses found a duplicated unescaped opening quote at the start of one
support list, in batch 175, request
`e3647fceec55597b1db5c9467d663bc7ff7e21e3dcc39440ae3af964baca1236`.
The original response is preserved. The existing support-terminator repair
correctly rejected this different syntax error.

The early diagnostic is
`eval_results/full100-spine-corpus-20260909-r1/offset-030/early-nonquote-diagnostics-20260910-r1.json`,
SHA `9ddbae973dd198bb4c6e5e899f3fa05f05967ee0c6540469b6db229cce683404`.
It covers the non-quote failures inspected at that moment and is explicitly
not a complete namespace audit or an admission artifact.

`spine_quote_json_repair_v2.py` first preserves the existing parser and its
receipts. Its additional branch removes only the duplicated quote before an
escaped opening quote in a support array. The result must parse with exact,
unique, ordered atom fields. Every deleted character must fall inside a parsed
support value, and reinserting those quotes must recover the original response
byte-for-byte. Summary characters cannot change. Unrelated or combined syntax
defects remain rejected.

The real batch now yields all three exact source atoms without a new model
call. Diagnostic artifact:
`offset-030/duplicate-support-opening-quote-diagnostic.json`, SHA
`150d7cae2ec1284433e883c7c0f3865248865117c6942df6d12a273265214a1c`.
This diagnostic does not admit a partial namespace or establish summary
entailment. Qwen still receives summaries only.

The new parser has separate versions of the complete source audit, bounded
summary compaction and source admission:

- `tools/audit_spine_source_admission_v2.py` reports unresolved syntax as an
  audit failure and preserves the full response population.
- `tools/repair_spine_summary_budget_v3.py` retains the deterministic eight-job
  summary-only batching and immutable zero-retry journals.
- `tools/admit_spine_corpus_v4.py` emits source-binding policy v7 and preserves
  complete raw span coverage and every uncompacted summary.
- `tools/verify_spine_admission_method_v5.py` replays older admission methods
  unchanged, verifies the new conditional syntax rule, independently reconstructs
  the entire oversized-summary population, and replays the resulting atoms.
- `tools/report_joint_source_spine_facets_full100_v2.py` uses the new admission
  verifier with the same accuracy and latency requirements.

Forty-four focused tests pass across full100 gates, complete passage population
verification, source-admission replay and syntax repair. The new admission test
combines the duplicated quote with nine oversized summaries, verifies two
summary-only compaction batches, checks complete exact raw coverage, and
reproduces all artifacts without provider calls during replay.

## Active continuation

The three old complete memories replayed successfully under method v5; session
**30059** completed. Their common method is
`bf1006cae1a597edfee5eef632c232433b685c6c7d00e772cc7dab1e0500b0d0`.
The previous method-v4 artifacts and prepared answer prompts remain unchanged.
The aggregate `facet-full100-method-verification-v2.json` SHA is
`ddbf18ba89be0703bdc9d6decc4bbb2991f71027337c81c0161d759f82ad3ec0`.
Method-v5 certificate hashes for offsets 0, 10 and 20 are, respectively:

- `59b10e09df88039ae2cfdf5183c31a1e5450f7a5a6f2275448318253bbd9d7df`;
- `3436b4524fe24ece1c53cd5341b841de58416ef9fdf3a640a4a307d8d01d29e3`;
- `b76c191c1a5bf43d04c09d70f0847bc1b8fe6dba45a518b1a3bbbc81d780851c`.

The v5 verifier, v7 admission, v3 compactor, v2 syntax repair and new full100
report are now bound by these verification artifacts. Preserve those versions
and use a successor if further behavior changes are needed.

Raw ingest session **76404** finished with 838 new calls and zero replay hits.
It covers 5,516 fragments, 5,514 turns and 473 sources, totaling 1,043,571 raw
token proxies. The original strict aggregate SHA is
`debc739635f5cb50367450dc45e52b86211b29d70a8d510456e918c7829d6904`;
its 558 invalid batches include quote diagnostics and are not final source
admission verdicts.

The complete v2 source audit finished in session **34599** at SHA
`1f3b9bac2d27f81766bb7f52dd9bdc941974243e275f4fe8393a170bab55d287`:
six oversized summaries and one unresolved schema failure. The remaining fault
is batch 726, request
`4301d3d3ceadef569a452d26d5058e4927f5044f0917f07e835d7aeef155322d`.
Its support strings duplicate opening and closing quotes without escaping the
inner quotes. All original responses remain intact; the complete namespace has
not yet been admitted.

Further support-delimiter recovery is drafted in
`spine_quote_json_repair_v3.py`, with v3 audit, v4 compaction, v5 admission
(policy v8), v6 admission verification and v3 full100 passage report modules.
Their new tests are pending until the timed comparison finishes. Do not treat
these drafts as verified or replace the existing sealed versions. The remaining
schema failure must be resolved before preparing summary compaction.

The 150-request fresh passage comparison is running in session **86559**, after
raw ingest, the full audit and large verification jobs ended. Runner:
`eval_results/full1m-source-spine-facets-development30-20260910-r1/run_comparisons.py`.
It processes the three roots listed in Log 142 serially, judges every complete
batch, then replays all judgments before aggregating the original measurements.
The first 50 answers are sealed at
`a671fcdf6da2cdaf394fe60fe524556e2efdb8c2789cf722b506d3f2d5a79cda`;
its judge was running at this checkpoint. No new accuracy claim is made yet.

Poll the live runner; do not duplicate it. Compaction, further raw ingest, GPU
compilation and large replay jobs must not overlap the measurements. Evaluate
both methods on the complete prepared population; retain the full100 target
and unopened confirmation set.
