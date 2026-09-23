# Source-grounded review of all 100 Sol answers

The completed Sol run in Log 235 scores 94/100 under the original grader. Its
only new failed grade contradicts the explicit user statement that they have
been attending coffee-shop open-mic nights. Before proposing any adjudicated
score, review the complete population, including all 94 passing predictions.
The original acceptance gate remains unchanged pending the user's preference.

## Frozen diagnostic protocol

`tools/audit_native_spine_source_answers.py` authenticates the completed run,
all 200 response journals, the 100 original questions/references and the raw
audit. It then reviews only the 100 saved memory answers. It reads the reference
conversation's user turns from the original source-body bank and checks them
against the author's recorded source turns. It also supplies every exact raw
span retrieved for the answer, with speaker, conversation and timestamp labels.

Reference-conversation turns are explicitly labeled as potentially unserved.
They can identify missing facts and defective references; they do not falsely
improve the answer model's retrieval. Original grades, scores and the answering
model's identity are withheld from the reviewer. The audit uses the Terra alias;
the saved answer run used Sol. This does not establish independence of the
underlying gateway implementations.

The fixed protocol assesses the question's actual situation and requirements.
The reference is fallible: extra source-supported detail is not automatically
false, and unrequested or logically redundant facts are not mandatory. Actual
contradictions, missing requested details, speaker errors and scope errors remain
failures. Concrete unresolved question ambiguity receives its own verdict.

Every reported issue needs exact source quotes. Correct reviews require source
support and no answer issues; incorrect reviews require concrete issues;
ambiguity requires two distinct quoted texts. Invalid reviews stay invalid.
All 100 items remain in the denominator, including ambiguous/invalid reviews.
The review never overwrites original scores or permits a target-completion claim.
Model findings still require source inspection and user review of any proposed
acceptance-policy change.

Twenty-five source-review and paired-answer checks pass in 2.11 s. They cover
invented or unknown quotes, altered prediction/reference anchors, empty support,
inconsistent verdicts, unsupported ambiguity, duplicate JSON fields, trailing
prose and incomplete paired populations.

## Completed execution and replay

- Root: `eval_results/native-spine-source-answer-review-20260915-r1`
- Log: `eval_results/native-spine-source-answer-review-20260915-r1.log`
- Preflight: `977e6d7e39d1c19c04738f9b3abac6c9229008ccce4dc0161f0ba81c8cbe967f`
- Provider-free preparation: session `23186`, terminal exit 0.
- Review session `84160` completed with terminal exit 0; all 100 reviews are saved.
- Provider-free replay session `36919` completed with terminal exit 0, 100 cache
  hits and zero new calls, reproducing the same report hash.
- Report: `74177f31d3361c3200b470a93836fb8f291eeaf1eb787c15ebc45d9c6866f871`
- All 100 source bundles prepared; largest prompt is 5,428 proxy tokens, below
  the 16,384-token bound. No evidence is truncated.
- Exactly 100 reviews, concurrency eight, 1,536 output tokens and no
  automatic retries. Zero new answer calls, ingestions or Qwen passes.

## Diagnostic findings

The reviewer returned **87 correct, seven incorrect, two ambiguous and four
invalid reviews**. These are diagnostic labels, not a new accuracy rate. The
four invalid responses contain nonexact purported source quotes; they remain
invalid without retry or repair. All 100 questions remain in the denominator.

| Original automated grade | Source review | Count |
|---|---|---:|
| Pass | Correct | 83 |
| Pass | Incorrect | 7 |
| Pass | Invalid review | 4 |
| Fail | Correct | 4 |
| Fail | Ambiguous | 2 |

The four rejected answers accepted by this reviewer are ordinals 58, 82, 95 and
96. The two ambiguous items are 53 (multiple real concert histories) and 93
(more than three real vintage collections). This supports the earlier grading
concern without establishing a revised score.

Source inspection also confirms errors that the original grader passed:

- **17:** the user says "I'm actually planning to do 3-mile jogs"; the answer
  says "You're doing" those jogs and presents intended strength work as ongoing.
- **31:** the user asks whether policy violations will have penalties. The answer
  attributes proportionality and monetary fines to the user, although those
  details came from the assistant's reply.
- **42:** "It looks like it's from the Victorian era" becomes a definite
  "Victorian-era armchair." The answer also imports an assistant-suggested
  furniture-style learning goal.

The relevant original user turns were present in the served packets for all
three. These are answer attribution/qualification problems after successful
retrieval, despite using the user spine and an existing reader policy that
explicitly instructs the model to preserve those distinctions.

The new reviewer is fallible too. Its ordinal 38 explanation incorrectly calls
the educational-game detail unsupported: a served user follow-up explicitly
names the educational game. Ordinal 59 overlooks the user's commute-reading
context; the precise causal link to e-book preference remains an inference.
Ordinal 79 treats one reference conversation as exclusive despite a broad
routine question and compatible user statements elsewhere. Ordinal 81 correctly
flags attributing assistant-written tactical specifics to a user request, but
its objection to the year 1970 overlooks the preceding story the user was
discussing. These distinctions prevent treating all seven flags as established
new failures.

The four invalid reviews are 9, 13, 15 and 61. Their saved answers are supported
by the supplied user text; three reviewers inserted ellipses into purported
exact quotes and the budgeting reviewer also altered source wording. This is
why quote validation must remain strict even when an assessment looks plausible.

## Reviewable comparison

`tools/report_native_spine_source_review.py` validates the complete original and
review populations, prediction/reference hashes, review quotes and invalid
review diagnoses before writing a separate comparison. It does not call models.

- Comparison: `eval_results/native-spine-source-answer-review-20260915-r1/comparison.json`
- Comparison hash: `69a4152b7bfd7182286d7a0b8fc1100b5b464286d04359f9d69492ed9845f1fe`
- All 100 questions, saved answers, locked references, original grades, review
  findings and agent inspection notes are in
  [inspection.md](../../eval_results/native-spine-source-answer-review-20260915-r1/inspection.md).
- Inspection text hash: `e33b19c46fe3bafc60c3c3011942c0e513e8295b2ea9243df5155fae24907e99`
- No human adjudication is claimed. No original score or acceptance gate changes.

The next design decision should address the demonstrated attribution and
qualification errors. The current evidence does not establish a need for
query-time Qwen attention on raw text, nor show that another blind prompt
rewrite or a larger answer model would fix them. Cached summary-only Qwen
processing, exact hydration and the normal application restart lifecycle remain
the basis of the reported run. The user's acceptance-policy preference is still
pending; the original grader remains the gate in the meantime.

The unchanged automated score remains 94/100. Accuracy and latency from the
actual answer run remain those reported in Log 235; review latency is a separate
evaluation cost. The 95% goal remains active.
