# Cap-eight repair on one complete million-token history

**Status:** Complete — 100 fresh answers, 100 grader calls, 100 raw-packet audits.  
**Date:** 2026-09-23.  
**Applies to:** Commit `a4adc03f1f6ef78788788e22133847b9c74929ac`, history 01 of the ten-session campaign.  
**Depends on:** [Log 245](245%20-%202026-09-23%20-%20Earliest%20loss%20trace%20and%20user%20completion%20routing.md), [Log 246](246%20-%202026-09-23%20-%20Single%20million-token%20routing%20heuristic%20ablation.md), and [Log 247](247%20-%202026-09-23%20-%20Downstream%20compensation%20check%20for%20routing%20ablation.md).

The committed cap-8 fix scored **94/100**, matching the original **94/100**
baseline on the same **1,115,343-token** history. Recorded-support coverage
improved from **97/100 to 100/100**, and every previously served evidence span
was retained. Evidence delivery improved without a measured net accuracy gain.

## Fixed scope and implementation

Use `user-completion8-2048-direct8-v1.json`, SHA-256
`bc523c267708dea21110013f541f1d964265af851d07262f2b0d4881eaedc31f`.
The fix places user routes before assistant context and appends at most eight
remaining user atoms from conversations already selected by the existing route.
All base retrieval limits remain unchanged, including eight direct summary
matches and the 2,048-token raw budget. The v7 reader, answer-model alias
(`codex_sdk/gpt-5.6-sol`), 256-token output cap, questions, references and grader
also remain unchanged.

`tools/evaluate_native_spine_completion_single100.py` supplies the complete
100-question population to the committed `run_native_spine_user_completion_answers.py`
runner. It reuses that runner's application reopen, live query embedding,
streaming timing, grading and independent raw reconstruction. This is a fresh
complete answer population: none of the earlier bounded repair answers were
copied into this score. There was one read-only application reopen and no new
ingestion, summary compilation or Qwen call.

## Results

| Metric | Original baseline | Cap-8 fix |
| --- | ---: | ---: |
| Correct answers | 94/100 | **94/100** |
| Questions containing all recorded support quotes | 97/100 | **100/100** |
| Mean provider-reported answer input tokens | 1,484.18 | 1,761.60 |
| Warm median end-to-end | 4.744 s | 4.345 s |
| Warm mean end-to-end | 5.053 s | 4.835 s |
| Warm p95 end-to-end | 7.476 s | 7.793 s |
| Mean retrieval/prompt preparation | 0.283 s | 0.262 s |
| Answers below five seconds | 61/100 | 70/100 |

Input increased by **277.42 tokens per answer (18.69%)**. Warm median remains
under five seconds; the p95 is slightly higher. Timing is compared with a saved
historical baseline, not a contemporaneous API control, so the lower median
does not establish that the fix causes faster responses. Cold setup was
54.446 seconds, excluded from warm timings.

The six previous misses produced one passing grade; 93 of the 94 previous
passes remained correct. The graded gain is **Q74**, the Amsterdam botanical
garden question. Its served prompt is byte-identical to the original and it
adds no completion atoms. This gain cannot be credited to changed retrieval;
answer/grader variation remains a plausible explanation. Fourteen prompts are
unchanged overall. No identical-prediction grade flips occurred in this run.

The regression is **Q46**, the meals-and-snacks question. The candidate adds
Sunday-breakfast and triathlon-training details to the answer, exceeding the
reference scope. Triathlon material was already present in the original packet;
the repair adds further training, cycling, breakfast and other user turns from
the selected conversations. The observation is consistent with scope dilution,
but one changed answer does not isolate its cause.

**Q66 demonstrates the repaired delivery and remaining reader issue.** The
Garmin-on-the-way and separate-heart-rate-monitor decision now reaches the
reader and is stated correctly. Strava is also present in the served packet,
but the answer omits it, so the unchanged grader still rejects the answer.
All six remaining failed answers contain every recorded support quote. Quote
coverage is not exhaustive reference coverage, and several questions retain
known grading/scope limitations; this does not prove all residual misses are
reader failures. No grades were manually changed.

## Verification and artifacts

- The commit's nine focused user-completion tests passed.
- All 100 base route receipts exactly match the original campaign.
- All 100 previously served span populations are preserved.
- Maximum completion additions: 8; mean: 5.73.
- All 100 packets reconstruct exactly from the original raw bank: 1,851 spans.
- Every answer ended with `stop`; no answer or grading retry was used.
- Root: `eval_results/native-spine-completion-single100-20260923-r1`.
- Summary report SHA-256: `cff1f98106b3991b5d654b41bc6229a4ad9bd98fa49afcbca42262ce77e3ab00`.
- History report SHA-256: `1ef3abca2724c82c56e6245df44366d24d4e73d2faf797e3411bf960396d643d`.
- `preservation-audit.json` binds original/candidate packets and evidence checks.
- `outcome-audit.json` records unchanged prompts and the inspected cases.

Provider-free summary replay:

```powershell
$env:PYTHONPATH = 'src;.'
$env:PYTHONUTF8 = '1'
& .pixi/envs/dev/python.exe -m tools.evaluate_native_spine_completion_single100 report
```

The fix repairs the demonstrated retrieval losses on this history. It has not
demonstrated an overall accuracy improvement or the 95% target here. This is an
exposed single-history result and does not replace the sealed 913/1,000 campaign
score. Production defaults and original campaign artifacts were not changed.
