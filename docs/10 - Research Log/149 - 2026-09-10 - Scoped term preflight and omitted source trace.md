# Scoped term preflight and omitted source trace

**Date:** 2026-09-10  
**Status:** 200 fresh requests prepared, zero answer calls; one source omission localized  
**Predecessor:** [148 - Reader development40 result and summary term coverage](148%20-%202026-09-10%20-%20Reader%20development40%20result%20and%20summary%20term%20coverage.md)

The historical 95/100 already covered approximately 1M-token memories on the
same locked 100 questions. It belongs to the cumulative retrieval and
answer-repair lineage, which retained authenticated earlier answers and did
not demonstrate fresh full-population API-like latency. The historical fast
73/100 lost 23 answers and gained one; 18 losses involved multiple sessions or
temporal reasoning. The current reader's 33/40 versus 32/40 control is a
development comparison, not a replacement full100 score. Accuracy and latency
must still pass together. See Logs 132 and 148 for the measured comparisons.

## Prepared matched routing experiment

`tools/spine_term_memory.py` adds the source-scoped summary-term supplement to
the existing resident facet memory. Query embedding and ordinary summary
selection remain unchanged. Term expansion reads stored summaries in already
selected conversations; production then performs one exact hydration with the
same 3,072-token and 128-span limits. Qwen receives no raw content and adds no
query-time call.

`tools/evaluate_spine_term_coverage.py` holds reader policy v2 fixed and compares
facet routing with scoped term coverage. Each memory arm has an adjacent,
counterbalanced API control with byte-identical evidence. A short API control
remains included. Serial streamed answers use Terra with 256 output tokens,
reserve before sending, and perform no automatic retries.

`tools/report_joint_spine_term_coverage_full100.py` retains the complete
100-question, common admission/compilation policy, accuracy and latency gates.
The focused evaluator, full100 verification and term-coverage suite passed
**139 tests**. No full100 gate has passed.

Preparation reproduced all 40 prior control prompts and all 40 previously
audited scoped candidates byte for byte. It read no reference answers or
predictions and prepared **200 unstarted requests**, with at most 80 logical
judgments. The frozen reader and evidence budgets must not change within this
experiment. No answer calls or latency measurements have been made for it.

Artifacts under `eval_results/`:

- Preparation: `full1m-spine-term-coverage-development40-20260910-r1/preparation.json`,
  SHA `8cd1ceabf21926e66c43656777fcdf2b735c017d2cdda6e1d6b6c7c5203f517f`.
- Offset 0 preflight: `f78416fa6e5212c57ca54835f64230543b17cd37f99cd7528ec7e16c6f8dd562`.
- Offset 10: `697b3cf6774494719eac9ec3220046a9585ce309723cdd918c0ee9dbc139d547`.
- Offset 20: `b307c30fe441d56d6501584a0d0982ac5062cf962449764447c026f171576031`.
- Offset 30: `d55d1a52ccf953eeade73bf59eb7bb97ec2011c89346b54252cf0fd888145c7f`.

The individual roots are
`full1m-spine-term-coverage-joint-offsetNNN-20260910-r1`. Preparation remains
usable, but the source omission below warrants investigation before executing
another comparison. These artifacts must remain unchanged if a different
router is prepared.

## Postscore source-omission trace

Ordinal 31 asks for the total weight of new feed purchased. The current packet
contains the 50-pound purchase but omits a separate 20-pound purchase. Local
inspection found the second statement in a conversation about permits for
selling homemade jam and honey. This is already examined development evidence;
the witness must never become a production source selector or cached answer.

A local diagnostic reproduced the frozen baseline prompt, ran ordinary
summary routing and exact hydration, and only afterward used the witness turn
ID to inspect its ranks. The stored leaf summary explicitly preserves the
20 pounds of organic scratch grains. Its ranks under the unchanged query are:

| Address channel | Leaf rank | Current selection limit |
| --- | ---: | ---: |
| Combined summary, dense score | 10 | 6 total baseline slots, including 2 lexical slots |
| User summary, dense score | 9 | 8 supplemental slots |
| Best user-summary passage | 26 | 8 supplemental slots |

The witness conversation is absent from the expanded prior plan. Neither the
baseline nor scoped-term candidate hydrates the turn. This localizes this
omission to source selection: the information exists in the summary, and the
hydrator never receives its descriptor. A supplement limited to already
selected sources cannot recover it. This does not establish a causal accuracy
gain from increasing any limit, nor justify selecting a limit from this one
witness. The next routing experiment should assess source coverage across the
complete development population under the unchanged evidence budget.

Diagnostic script and artifact:
`eval_results/spine-witness-trace-development40-20260910-r1/trace.py` and
`ordinal031.json`, SHA
`0bc39f2288aadb8be601a4da21ffa76b042cc484914ece60a4bb6e002731b705`.
The script made zero provider calls, sent no raw text to a model, loaded no
predictions or gold itself, and made no accuracy or latency claim. The witness
was selected through earlier postscore inspection, so this is not a blind test.

## Full-corpus continuation

Offset-40 recovery session **43809** remains live; completed batches through
588 were observed at approximately 10:09 UTC. It retains the original successful
responses and uses the previously frozen recovery budget. No timed evaluation
runs concurrently. The witness diagnostic session **56177** completed with
exit code 0. The remaining source admission and memory construction steps are
unchanged from Log 148. Do not start a second raw recovery executor.
