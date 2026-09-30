# Single million-token routing heuristic ablation

**Status:** Complete — 100 fresh answers, 100 unchanged-grader calls, 100 raw-evidence audits.  
**Date:** 2026-09-23.  
**Applies to:** History 01 of `native-spine-ten100-20260922-r1`.  
**Depends on:** [Research Log 244](244%20-%202026-09-22%20-%20Ten%20million-token%20session%20evaluation.md) and [Analysis 35](../08%20-%20Analysis/35%20-%20Ten-session%20failure%20patterns%20and%20repair%20priorities%202026-09-23.md).

Removing the active summary-routing additions reduced accuracy from **94/100
to 73/100** on the same 100 questions over the same **1,115,343-token** history.
There were **21 regressions and no improvements**. The modest token and latency
savings do not justify removing these additions together on this evidence.

This is a conditional ablation with downstream projection and reader rules
retained. [Log 247](247%20-%202026-09-23%20-%20Downstream%20compensation%20check%20for%20routing%20ablation.md)
tests the presentation interaction: removing projection and user-first layout
from both variants gives **93/100 versus 73/100**. The routing advantage remains,
but neither comparison proves each individual heuristic is necessary.

## Controlled change

The old regex extractor and importance/energy memory ranking were already
disabled in the baseline. This experiment therefore ablates the active native
retrieval additions, not those unused legacy components:

- Parent-neighborhood expansion through the stored attention hierarchy.
- User-role and neighborhood-proximity reordering.
- The additive BM25 summary match.
- The parent-user-summary supplement.

The candidate uses the ordinary persisted application's native context router
with `max_additions=0`, `protected_direct=8`, and `ancestor_hops=0`, rather than
the lexical/parent-supplement subclasses. Dense top-8 summary matching,
`lexical_reserve=0`, dated source eligibility, the 2,048-token and 128-span caps,
exact raw hydration, whole-user-section presentation, v7 reader, answer-model
alias (`codex_sdk/gpt-5.6-sol`), 256-token output cap, questions, references, and
grader stay unchanged. All 100 initial ordered section selections and eligible
source lists matched the saved baseline exactly.

The first campaign history was selected without searching for a favorable
score. Its completed normal ingestion was reopened read-only in a fresh
process. No histories, summaries, attention artifacts, or question sets were
rebuilt; no Qwen calls were made. Answers ran sequentially, with fresh query
embedding and retrieval inside each timer. References opened only after all
100 candidate answers were sealed. The later user-completion repair in Log 245
is not part of either arm.

## Results

| Metric | Saved baseline | Dense-only candidate |
| --- | ---: | ---: |
| Correct answers | 94/100 | 73/100 |
| All recorded support quotes present | 97/100 | 69/100 |
| Mean answer input tokens | 1,484.18 | 1,267.87 |
| Warm median end-to-end | 4.744 s | 4.361 s |
| Warm mean end-to-end | 5.053 s | 4.733 s |
| Warm p95 end-to-end | 7.476 s | 7.010 s |
| Mean retrieval/prompt preparation | 0.283 s | 0.177 s |
| Answers below five seconds | 61/100 | 70/100 |

Input fell by 216.31 tokens per answer (14.57%); observed median latency fell
by 0.382 seconds. This is a historical timing comparison, not a contemporaneous
API control. Candidate cold setup took 168.923 seconds, versus 24.845 seconds
in the saved baseline; that separate startup cost is excluded from warm timing
and its cause was not isolated. Existing editor/MCP services remained running.

Twenty of the 21 regressions lost at least one recorded support quote that the
baseline had served. Examples include the acquired Leica lens (Q2), the later
H-1B payment detail (Q6), and the Shibuya booking with seven nights and nightly
rate (Q11). This is substantial evidence loss even though every served excerpt
remained exact. All 100 candidate packets reconstructed successfully from the
authenticated original raw bank, covering 792 raw spans; all answers stopped
normally, and no answer or grading retries were used.

This joint ablation also removes attention-derived contextual expansion. It
does not identify the individual contribution of hierarchy, role ordering,
lexical supplementation, or parent supplementation, and it does not establish
that every heuristic is necessary. It supports retaining the combined active
path while investigating targeted repairs. One exposed history and the existing
semantic grader do not establish held-out generalization.

## Reproducible evidence

- Runner: `tools/evaluate_native_spine_heuristic_ablation.py`.
- Tests: `tests/test_native_spine_heuristic_ablation.py` — **2 passed**, checking
  unchanged seed order/budgets and rejection of altered seeds or retained additions.
- Result: `eval_results/native-spine-heuristic-ablation-20260923-r2/report.json`.
- Report SHA-256: `9d2e614975043000dda27d2715402ba62266cb09fd89d83c0ce61b9f9ccb8df1`.
- `r1` contains an unused preflight only; its launch was stopped before any
  provider calls by an overly broad idle guard that matched VS Code's formatter.

Provider-free report replay, using the sealed grading checkpoints:

```powershell
$env:PYTHONPATH = 'src;.'
$env:PYTHONUTF8 = '1'
& .pixi/envs/dev/python.exe -m tools.evaluate_native_spine_heuristic_ablation report
```

Production routing defaults and the original campaign artifacts were not changed.
