# Paired answer variation on identical user spine prompts

The parent-user run in Log 232 scored 93/100. Its saved API controls received
the exact same prompts, answer model and 256-token limit in alternating order,
but only memory predictions were originally graded. Before another reader
change, evaluate those 100 already saved controls with the original Sol grader.
This requires no new answers, memory queries, history ingestion or Qwen work.

## Fixed diagnostic population

`tools/audit_native_spine_paired_answers.py` accepts the complete saved parent-user
run. It authenticates all 200 response journals, the raw audit, completion record,
original questions/references and memory grades. For every memory prediction it
reconstructs and matches the exact original grader prompt before creating the
corresponding control-grading prompt. Labels, memory grades and comparison
metadata do not enter the control grader's prompt.

All 100 controls receive independent judgments. Thirty-five pairs have identical
prediction strings; the other 65 differ. Independent grades on the identical
strings can expose grader variation. Different strings can differ in meaning or
just wording, so every score disagreement still requires inspection.

The two scores remain separate. The tool does not report a best-of-two score,
overwrite any memory grade or permit a target-completion claim. These are
previously collected direct API controls, not new timed memory answers.

Twenty-four diagnostic and parent-evaluator tests pass in 2.70 s. Tests reject
partial populations, mismatched prompts/questions, changed references or original
grades, altered grading prompts and incomplete verdict populations. Separate
accuracy and identical-prediction disagreements are checked explicitly.

## Completed result: both arms score 93/100

- Source: `eval_results/native-spine-app-parent-users100-20260915-r1`
- Source report: `796f777eaa1e2216c1c8d676441adccf6ca580780b890b17daa1f2747a54058c`
- Diagnostic root: `eval_results/native-spine-paired-answer-audit-20260915-r1`
- Log: `eval_results/native-spine-paired-answer-audit-20260915-r1.log`
- Preflight: `41613540045eea5365372ad677864937aada9184a1ae3e147275421de76f782b`
- Exec session `94876`, terminal exit 0.
- Provider-free replay: session `68068`, terminal exit 0; 100 cache hits, zero
  new calls and the identical report hash.
- Report: `a4da5193d3dd6e7340a085cac5e660d5e24048f1f212bbf4b9b763eb36907eec`
- Disagreement review: `9a012774f0f0af7d927d84cecc4092d7907f8ee6b3ca44d54bf4324a3c79d399`

The API controls score **93/100**, matching the memory arm, with different
passing sets. API-only passes are **19, 81 and 95**; memory-only passes are
**16, 22 and 26**. All 35 identical prediction strings receive matching grades.
The audit makes 100 new judge calls and zero new answer calls.

Direct inspection separates the disagreements:

- 16 and 19 show actual answer variation. The API drops the catchy-creative-lines
  follow-up for the eucalyptus request, but preserves the acting-depth
  qualification omitted by the memory answer.
- 22's API grade rejects carefree summer-vibe detail explicitly present in user
  text. Its broader "soulful music" wording is an inference from admiration for
  a soulful voice and should be examined separately.
- 26's API answer adds source-supported Indigenous events/festivals; the grade
  rejects the addition because it is absent from the reference.
- 81 exposes a semantic grading inconsistency: both answers omit the central
  dreams/nightmares contrast, but the API answer passes. It is not a demonstrated
  reader repair.
- 95's answers both contain the reference core plus French/international
  interests present in user evidence. Different grades need source-aware review;
  they do not prove that the memory system invented these interests.

The unchanged paired input rules out retrieval differences as the cause of
these six score disagreements. It does not establish that every changed grade
is caused by answer generation: the inspected grader defects remain material.
No original grade is altered and no union of passing answers is scored.

Next, compare Sol on the current complete, user-first packets with the v7
reader. The earlier Sol result used the older 97/100-support packets and another
layout, so it does not measure this combination. The current comparison changes
only the answer model; it does not use the rejected v8 reader.

The original memory score remains 93/100. The 95% target is unmet. References
and the original denominator remain fixed, including known evaluation defects.
