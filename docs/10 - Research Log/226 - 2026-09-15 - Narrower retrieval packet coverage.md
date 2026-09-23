# Narrower retrieval packet coverage

**Status:** Diagnostic complete; no new answer-accuracy score.
**Date:** 2026-09-15.
**Scope:** One persisted 1,098,417-token application memory and the same 100 questions.
**Depends on:** [Research Log 225](225%20-%202026-09-15%20-%20Complete%20user%20statement%20reader%20through%20application%20memory.md).

The latest answer run scored 87/100. Its packets contained a median of 14
conversations, and ten of its thirteen misses contained all recorded support
quotes. This makes excess cross-conversation material a plausible contributor
to reader omissions and attribution errors, but does not establish causation.

`tools/assess_native_spine_packet_width.py` reopens the existing application once
and retrieves all 100 questions with direct-match limits of 16 and 8. Every other
retrieval parameter remains fixed. All 200 packets are sealed before references
are opened. There are no answer calls, Qwen calls, new ingestions or new histories.

| Direct matches | All recorded support | Partial | None | Median conversations | Median rendered tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32, existing baseline | 97 | 2 | 1 | 14 | 1,563.5 |
| 16 | 97 | 2 | 1 | 8 | 1,563.0 |
| 8 | 97 | 2 | 1 | 4 | 1,496.5 |

The same 97 questions retain all recorded quotes at both narrower widths. This
does not demonstrate complete gold-answer coverage or improved answer accuracy.
Freed capacity can admit more nearby assistant context, so reduced conversation
count does not imply a proportionate reduction in text or attribution risk.

Two focused tests passed in 1.37 seconds. They cover adjacent same-turn Unicode
quotes, gaps, foreign source occurrences, and missing or invented supports.
An independent audit reconstructs all 200 packets from the original raw bank:
1,915 exact spans at width 16 and 1,430 at width 8. All pass.

| Artifact | Binding |
| --- | --- |
| Root | `eval_results/native-spine-packet-width-20260915-r1` |
| Exec session | `14534`, terminal exit 0 |
| Report SHA-256 | `21c46796bd6d7ac4b7d251f9c48d23b1ce9553039db37f7be27007cb843b3193` |
| Raw audit SHA-256 | `d517b59787b9895b1f3bdf86702901f12767611e026c3886083db97bb0fa25d0` |
| Width-8 policy | `eval_results/native-spine-context-policies/dense-parent-2048-direct8-v1.json` |
| Policy SHA-256 | `114516284f503fb8ba489b757f8a5ef2283b9e2112b3a25ec41c3a2a5e480e19` |

The next answer comparison uses width 8, preserving the v7 reader, answer model,
questions, references and grading. It must perform fresh timed application
retrieval, all 100 answers and matched API controls, then grading and raw-source
audit. The measured accuracy remains 87/100 until that run finishes. The 95%
target remains unmet, and this is an exposed development set.
