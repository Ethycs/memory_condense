# Reduced oracle-source control isolates retrieval from technique

**Date:** 2026-08-28

**Status:** sealed post-hoc diagnostic complete; 17/24 remaining misses correct;
not a benchmark score or promotable runtime arm

## Question tested

The preceding miss-only replay reduced aggregate provider work while holding
every million-token store and every retrieved prompt byte-exact. It ruled out
batch/orchestration pressure, but it did not test whether searching the large
store caused retrieval degradation.

This second control removes that ambiguity. For each of the 24 questions still
wrong after the replay, it selects only the raw LongMemEval sessions named by
`answer_session_ids`. Terra receives the exact dated question and raw dialogue
turns, but no reference answer, question ID, source ID, category, prior
prediction, judge output, or scorer feedback. Sol receives the sealed
prediction only afterward together with the matching question and reference.

This selection is deliberately a **gold-source oracle diagnostic**. It tests
what happens when million-token source discovery is replaced by the labelled
small source set. It cannot be counted as retrieval or benchmark accuracy.

## Hard-cap treatment

The complete raw labelled sources do not all fit the 8,000-token envelope.
Under the exact QA renderer, 12/24 fit completely. The other 12 are fitted by
a deterministic question-only whole-turn policy:

1. split the labelled sources into exact raw dialogue turns;
2. score turns from question terms and phrases, with a small user-role prior;
3. reserve the strongest user-bearing turn from every labelled source;
4. consider adjacent turns and then remaining turns in deterministic order;
5. restore original chronology after every admission; and
6. admit only complete turns whose fully rendered prompt plus the 256-token
   answer reserve remains at or below 8,000.

The fitter never reads the answer or reference. Every selected raw-content
hash, selected/dropped count, source count, full-source token count, and fitted
token count is sealed. Therefore the two result strata have different claim
strength:

- **12 full-source rows:** a clean answer-technique control after oracle source
  selection;
- **12 fitted-source rows:** a capped oracle-source control whose misses can
  still be caused by within-source packing.

The final 24 Terra prompts total 160,320 input-token proxies, with a range of
3,601--7,739. The unfitted labelled sources would total 186,603. For context,
the corresponding frozen typed prompts total 129,005; this control reduces the
searched corpus, not necessarily the final per-question prompt. It is still
68.8% below the 513,276-token full-100 answer batch.

## Sealed execution

| Artifact | SHA-256 |
| --- | --- |
| gold-free answer preflight | `f0142a579a231e98c141c407e16df985ef0ca1639d7ee368b06f1414ecb5e6c0` |
| post-hoc judge authority | `c8dd8ea65503e48a7184ebed6768eafa12de69fbc6e3184be70675861a25ce40` |
| sealed Terra answer run | `b6d33f9251088bbf8c9f012dd80f07d66a48347f28248cb1729c483dc10d7df0` |
| sealed Sol judge run | `e88bf17fa8e59e7c00389447828fee9e701fa67153311fbb390ded4e5efc5b0a` |

Terra made exactly 24 physical calls. Its replay made zero calls and hit all
24 immutable journals. Sol then made exactly 24 physical calls; its replay
also made zero calls and hit all 24 journals. Both planes retain zero
transformer token state.

The streamlined runner is
`tools/run_reduced_oracle_source_assay.py`. Three focused tests verify exact
miss population, full-source preservation, question-only whole-turn fitting,
source-ID exclusion from prompts, and the complete token envelope.

## Result

Sol accepted 17/24 predictions, or 70.83%:

| Stratum | Correct | Accuracy |
| --- | ---: | ---: |
| complete labelled sources retained | 9/12 | 75.00% |
| question-only fitted labelled sources | 8/12 | 66.67% |
| **all remaining misses** | **17/24** | **70.83%** |

Correct ordinals are:

`7, 14, 16, 28, 31, 36, 43, 53, 54, 61, 65, 67, 72, 77, 79, 81, 86`.

Relative to the semantic failure assay, the oracle-source control rescues:

- 10/12 retrieval/localization cases; and
- 7/11 ranking/operator/synthesis cases.

The latter is important. Many nominal operator failures are not intrinsic
reasoning limits: once decisive raw evidence replaces the diluted packet, the
same Terra route answers them correctly.

This does **not** produce an official 90/100 score. The arithmetic
outcome-conditioned ceiling would be 73 previously correct plus 17 oracle
rescues, assuming no regressions outside the selected set. That is a causal
diagnostic, not a runnable gold-blind policy.

## Seven persistent failures

| Ordinal | Source condition | Failure after source reduction |
| ---: | --- | --- |
| 6 | fitted, 23/24 turns retained; decisive bluegrass/banjo turn retained | relative-date/entity-shape operator abstains because the memory gives a descriptor rather than a band name |
| 42 | full source | joins a thesis-poster event with Harvard attendance instead of enforcing the undergrad-course predicate and returning insufficiency |
| 49 | full source | returns relevant Denver music venues but omits the required Brandon Flowers/prior-experience personalization |
| 69 | fitted; blazer and old/new boots facts retained | collapses return and replacement-pickup action roles and counts two instead of three physical items/actions |
| 93 | fitted; first-client contract turn retained | selects the older website launch instead of resolving the four-week target to the contract milestone |
| 94 | fitted | locked evidence/reference conflict: the labelled source dates the class as 2022-03-20 while the question is 2022-04-15; the separate birthday-cake event is 2022-04-10, and neither supports accepted 21/22 days |
| 97 | full source | correctly sees 40% HelloFresh and 20% UberEats, but abstains because the latter turn does not explicitly call that UberEats order the first order; the reference assumes it |

Ordinals 94 and 97 are reference/evidence-policy problems, not clean memory
mechanism failures. The five well-posed persistent cases are exactly the next
operator/fact-compiler target: relative typed selection, scoped absence,
personalized synthesis, role-preserving cardinality, and temporal milestone
selection.

## Attribution

The two reduced controls now separate three hypotheses:

1. **Batch or process memory pressure:** unsupported. Reducing the answer batch
   by 71.9% produced no evidence-driven rescue.
2. **Million-token source discovery and prompt dilution:** dominant. Replacing
   the large-store search result with labelled source memory recovers 17/24.
3. **Downstream fact/operator technique:** still material. Five well-posed
   questions remain wrong even when their decisive turns are exposed; two
   additional rows have benchmark-evidence ambiguity.

The practical conclusion is not to tune generic top-k variables. The next
gold-blind treatment should approximate the oracle-source condition through
the intended common-memory pipeline:

```text
selected local/source neighborhoods
  -> query-conditioned exact-cited typed facts
  -> fact-derived global cue/read when slots remain unresolved
  -> role-sensitive dedup and evidence-density ranking
  -> genuine CAV linking over the final admitted fact frontier
  -> deterministic temporal/count/set/absence operators
  -> compact supporting raw turns + final Terra answer
```

If a gold-blind treatment can reproduce the 17 oracle-source rescues and solve
the five well-posed persistent operator cases without regressing the protected
73, the arithmetic target is 95/100. That hypothesis remains unproven until a
full-100 no-regression run.
