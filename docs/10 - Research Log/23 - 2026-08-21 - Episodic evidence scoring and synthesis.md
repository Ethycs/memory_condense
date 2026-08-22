# Episodic evidence scoring and grounded synthesis

**Status:** implemented, focused-test covered, and measured over the sealed
S1-through-S3 development prompts. All 176 episodic additions received a
generation-free forced-choice score, and 12 distinct cumulative prompts were
answered with the pinned local Qwen3-0.6B runtime. Answer quality was a clear
negative result: every stage scored 0/10 exact match and 0.010227 mean token
F1. No independent judge ran, the raw-p(A) answerability bands are
uncalibrated, and this entry makes no semantic-success or
held-out-generalization claim.

## Decision

The completed one-million-token cumulative retrieval ladder exposed a scoring
gap that its stage names did not make obvious. Direct and representative
episode *seeds* had scores, but the 176 evidence additions produced from those
seeds were never individually scored for whether they appeared answer-bearing.
The completed local-Qwen branch added an uncalibrated answerability diagnostic;
it did not yet measure useful evidence per token.

The completed correction is a post-hoc, gold-blind branch over the already
sealed S1, S2, and S3 prompts:

```text
sealed retrieval.json
├── evidence arm: classify and forced-choice score each incremental addition
└── synthesis arm: answer from each complete cumulative stage prompt
    └── post-hoc gold scoring after all model outputs are sealed
```

This branch did not modify `retrieval.json`, reorder the cumulative ladder,
or retroactively call the old seed utility an evidence score. A later
density-packed retrieval treatment, if built, must be a new matched arm rather
than a relabeling of this measurement.

## Correction: episode seed utility was not evidence quality

S1 inherited its seed utility from the protected-anchor order. The predecessor
adapter assigned rank-derived scores `(N - rank + 1) / N`; source-local episode
neighbors inherited a distance-decayed form of that anchor score. Those values
answer “which retrieved anchor led here?” They do not answer “does this added
text establish the requested fact?”

S2 used a Qwen prefix inspection utility: nonnegative QK affinity plus, in
`qk_ov` mode, `log1p` OV transport and an optional CAV contribution. That
utility chooses representative episodes. It is not calibrated answer
probability, entailment, evidence density, or marginal value after S1.

S3 used the resulting episode seeds to drive artifact-global closure. Its
closure priorities and relation confidence likewise did not score every
rendered addition against the question. In the completed campaign S3 admitted
no evidence under the budget, so there is no S3 addition whose seed score can
be reinterpreted.

The old values remain valid routing and construction provenance. This new
branch adds separate question-conditioned evidence labels and causal-choice
signals without rewriting them.

## Exact frozen inference population

The parent artifact is the exact original development concatenation recorded
in Research Log 22:

| Property | Frozen value |
| --- | --- |
| Retrieval artifact | `eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821/retrieval.json` |
| Retrieval SHA-256 | `aa22f7c18470d9a7c931fd16f8f58bf67d8566e2298a45371ee2815c11a9bd97` |
| Population identity SHA-256 | `fa9a06ebd103d87086943cfa94091bdf607fe07874bc871e465aad409b85ca18` |
| Questions | 10 |
| Transcript-token proxy | 1,039,203 |
| Cumulative stages in scope | S1, S2, S3 |

The evidence-classification population is the ordered, question-conditioned
stage delta, not every item repeated in every cumulative prompt:

| Delta | Question-stage evidence additions |
| --- | ---: |
| S1 minus S0 | 171 |
| S2 minus S1 | 5 |
| S3 minus S2 | 0 |
| **Total** | **176** |

“176 items” therefore means 176 `(question, stage, evidence_id, exact text)`
rows. It does not claim 176 globally unique chunks, episodes, facts, or source
documents. S0's 354 protected items are excluded because the question is
specifically whether the episodic additions supplied useful evidence.

Answer synthesis still evaluates all 30 S1-through-S3 question-stage rows.
There are only 12 distinct sealed prompt hashes: the ten S1 prompts, two S2
prompts that actually changed, and no new S3 prompt. Deterministic completion
may be memoized by the exact `prompt_messages_sha256`; every stage row must
still record which shared completion it used.

## Gold firewall

Inference may read only fields already visible to a responder:

- the question and its query timestamp;
- the exact cumulative provider messages for S1, S2, or S3;
- the exact incremental evidence IDs and rendered text;
- source timestamps and speaker/evidence provenance already rendered in that
  text; and
- sealed retrieval, stage, and prompt identities.

Inference must not read the gold answer, labeled evidence-source IDs, answer
components, prior score rows, or any judge verdict. The gold-bearing population
is loaded only by a separate post-hoc score phase after raw model outputs and
their hashes have been published. The inference artifact must continue to
declare that it contains no gold fields.

The completed post-hoc scorer computed normalized exact match, token F1, and
expected-source overlap diagnostics. A semantic judge would be a separate,
explicit experiment. It was not part of this local-Qwen run.

## Pinned one-load Qwen runtime

The implementation lives in
`memory_condense.eval.recall_guarded_cumulative_synthesis_runtime`. It verifies
the complete local causal-checkpoint manifest before loading any model and
then shares that one model and tokenizer between deterministic generation and
the generation-free forced-choice scorer.

| Runtime property | Value |
| --- | --- |
| Model | `Qwen/Qwen3-0.6B` |
| Revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Behavioral checkpoint manifest | `a940db06d5d9a3b298412376966b492f09ad7f088495fb75c05aa45db943d86e` |
| Local directory | `.cache/models/Qwen3-0.6B` |
| Precision / device | FP16 / CUDA; non-CUDA or non-FP16 loading fails closed |
| Checkpoint context limit | 40,960 tokens |
| Default generation reserve | 2,048 tokens |
| Generation | greedy, `do_sample=False`, Qwen thinking disabled, ephemeral K/V |
| Forced-choice scoring | A/B full conditional likelihood, K/V disabled |
| Candidate / query caps | 256 / 192 tokens |
| Choice prompt / workspace caps | 768 / 8,192 tokens |
| Choice batch / candidate caps | 8 / 128 |

The current one-million-token stage prompts peak at 7,283 legacy proxy tokens,
but the runtime does not trust that proxy as a Qwen context bound. It counts
the rendered prompt with the pinned Qwen tokenizer and requires actual input
tokens plus the output reserve to be at most 40,960 before generation.

The runtime API is deliberately small:

```python
from memory_condense.eval.recall_guarded_cumulative_synthesis_runtime import (
    RecallGuardedCumulativeSynthesisRuntime,
)

with RecallGuardedCumulativeSynthesisRuntime(
    ".cache/models/Qwen3-0.6B",
    max_new_tokens=2048,
) as runtime:
    completion = runtime.complete(provider_messages)
    evidence_scores = runtime.score_candidates(question, evidence_by_id)
    identity = runtime.identity.model_dump()
    completion_report = runtime.last_completion_report.model_dump()
    choice_report = runtime.last_score_report.model_dump()
    usage = runtime.usage.model_dump()
```

Calls are serialized because generation and classification share one model.
Per-call generation overrides are restored afterward. Completion reports bind
the canonical message hash, completion hash, actual input and output token
counts, generation cap, model identity, and elapsed time. Clean shutdown drops
both owners of the shared model and clears the CUDA allocator cache.

The full Qwen3-8B checkpoint is not selected for this campaign. It is 16.38 GB
and requires CPU offload on the available 8-GB GPU; the earlier local-answerer
probe did not finish checkpoint placement within ten minutes. The 0.6B model
keeps this first controlled inference run tractable.

## Structured density schema and effective answerability fallback

The strict structured-output schema asked the model to supply exactly one role
and one categorical `density` value for every delta row. IDs could not be
omitted, duplicated, or invented. That model-supplied `density` field was a
semantic category with the following intended vocabulary; it was not defined
as the forced-choice p(A) score or as p(A) divided by token count.

Evidence-role enum:

| Role | Meaning |
| --- | --- |
| `decisive` | Directly states a requested answer value or one required set member. |
| `supporting` | Materially corroborates an answer value. |
| `temporal_bridge` | Supplies a timestamp, ordering link, or temporal operand needed to connect evidence. |
| `qualifier_or_conflict` | Establishes scope, negation, uncertainty, revision, currentness, or a competing value. |
| `context` | Topically relevant background that is not needed for the answer proof. |
| `redundant` | Repeats evidence already available in the same cumulative stage without material new support. |
| `irrelevant` | Provides no question-relevant support. |

Structured categorical `density` enum:

| Band | Intended structured meaning |
| --- | --- |
| `critical` | Indispensable direct proof. |
| `high` | Strong direct or near-direct evidence. |
| `medium` | A useful bridge, operand, qualifier, or conflict-resolution item. |
| `low` | Weakly useful context. |
| `none` | No detected marginal answer evidence. |

The first structured response was invalid, so the effective campaign did not
obtain those model-supplied semantic density categories. Its method was
`short_answer_with_forced_choice_attribution`: it retained each inspected
item's raw A-versus-B answerability, value-evidence logit, label likelihoods,
and token count, then mechanically mapped raw p(A) thresholds into the legacy
band strings and roles:

| Effective answerability band | Raw p(A) rule | Stored legacy band string | Stored role |
| --- | ---: | --- | --- |
| `critical` | p(A) ≥ 0.80 | `critical` | `decisive` |
| `high` | 0.65 ≤ p(A) < 0.80 | `high` | `supporting` |
| `medium` | 0.50 ≤ p(A) < 0.65 | `medium` | `context` |
| `low` | 0.35 ≤ p(A) < 0.50 | `low` | `redundant` |
| `none` | p(A) < 0.35 | `none` | `irrelevant` |

Accordingly, `critical`, `high`, `medium`, `low`, and `none` in the completed
historical artifact are answerability-derived proxy labels, even where the
legacy schema stored them in a field named `density_band`. They must not be
reported as evidence density or density per token.

The fallback did not emit `temporal_bridge` or `qualifier_or_conflict`; those
remain valid only for a successful structured model response. No formula
turns the A/B softmax into a validated “evidence per token” probability.

The current v2 code makes this distinction explicit: it retains a separate
`answerability_band` for raw-p(A) thresholds and derives
`answerability_per_100_tokens` from p(A) and candidate length before assigning
a separately named `evidence_density_band` under a versioned policy. That
p(A)-per-token density measure is a later code path; it does not retroactively
change the historical artifact, measurements, or hashes reported here.

The strict structured schema allowed a short answer plus atomic cited claims.
The first structured call was invalid, so it was retained separately and did
not become an accepted structured record. Effective answers then used the
declared short-answer path with forced-choice attribution. Evidence annotations
and synthesis claims remain separate so a high forced-choice score cannot
silently become semantic proof.

## Citation validation and failure behavior

Generated citations are admissible only when all mechanical checks pass:

1. the cited evidence ID belongs to the exact question and cumulative stage;
2. every supplied quote is a byte-exact substring of that evidence text;
3. the quote SHA-256 recomputes from those exact UTF-8 bytes;
4. no evidence ID is used as a substitute for another source coordinate; and
5. every synthesized factual claim carries at least one valid citation.

Unknown IDs, duplicate classification rows, missing rows, invalid enums,
malformed JSON, truncated output, non-exact quotes, or uncited claims fail
closed. The raw completion and its hash remain available for diagnosis, but an
invalid parse is not silently repaired into a successful structured record.

This validation proves coordinate and quote integrity. It does not prove that
the cited quote semantically entails the model's claim; that remains part of
the result audit.

The campaign exercised that boundary. One initial structured call was invalid
and was retained with its own report: 6,880 input tokens, 209 output tokens,
and 60.32 seconds. The effective fallback then produced `I don't know.` for
nine questions and `[1]` for one. Reuse across identical prompts expanded the
nine abstentions to 27 stage rows. Those rows initially carried meaningless
automatic attribution despite asserting no answer. A separate gold-blind
normalization removed their claims and citations while preserving every raw
completion byte and hash. The normalized derivative, rather than a rewrite of
the raw artifact, is the answer-score authority below.

## Measured result

The run produced two separate result families. All 176 evidence additions were
inspected by the forced-choice model. In the table below, the five band columns
are the historical raw-p(A) **answerability-band** counts stored under the
legacy labels; they are not evidence-density counts.

| Evidence population | Count | Critical | High | Medium | Low | None | Mean p(A) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 additions | 171 | 3 | 14 | 13 | 28 | 113 | 0.288773 |
| S2 additions only | 5 | 0 | 0 | 0 | 0 | 5 | 0.103625 |
| S2 cumulative | 176 | 3 | 14 | 13 | 28 | 118 | — |
| S3 additions only | 0 | 0 | 0 | 0 | 0 | 0 | — |

Under the mechanical role mapping, S1 therefore contained 3 `decisive`, 14
`supporting`, 13 `context`, 28 `redundant`, and 113 `irrelevant` rows. S2
added five more `irrelevant` rows. Neither the fallback nor the S3 no-op
produced a `temporal_bridge` or `qualifier_or_conflict` row.

The largest S2-addition p(A) was only 0.162380. At the question level, the
macro mean causal answerability declined from 0.287523 for S1 to 0.285419 for
S2; S3 reused S2 unchanged. Because the scores are uncalibrated, these values
are descriptive ranking diagnostics, not posterior probabilities.

Post-hoc comparison with labeled sources provides a limited sanity check. The
critical/high answerability-band rows spanned ten selected sources, four of
which overlapped an expected source: micro precision 0.400 and macro
expected-source recall 0.250. The values are identical for S1, S2, and S3.
This is not strong semantic selection evidence; most labeled-source coverage
was outside the critical/high answerability subset.

The effective answer method was
`short_answer_with_forced_choice_attribution`:

| Stage | Exact match | Mean answer F1 |
| --- | ---: | ---: |
| S1 direct episodes | 0/10 | 0.010227 |
| S2 representatives | 0/10 | 0.010227 |
| S3 artifact-global | 0/10 | 0.010227 |

There were 12 unique effective answer calls, consuming 95,904 Qwen input
tokens and 58 output tokens over 1,276.72 seconds of generation. The invalid
structured attempt described above is retained separately and is not folded
into those effective-call totals.

Identical prompt hashes must reproduce one shared deterministic completion;
they are not extra independent observations. S2 and S3 cannot be described as
improvements merely because their rows repeat an S1 result.

## Interpretation and caveats

- The same Qwen3-0.6B checkpoint supplies the forced-choice feature and the
  synthesis. Their agreement is correlated evidence, not independent
  confirmation.
- The forced-choice A/B score is explicitly uncalibrated. Its thresholds and
  answerability-derived proxy bands are descriptive until calibrated on a
  separate population; they are not evidence-density estimates.
- The 0.6B model did not provide useful answers here: it abstained on nine
  questions, emitted a bare citation marker on one, and its first structured
  response was invalid. This run does not isolate whether model scale,
  prompting, evidence distraction, or their interaction caused that failure.
- The model is not an independent judge of its own answers. This campaign uses
  deterministic string and source-overlap metrics only unless a separately
  identified judge arm is later authorized.
- The 176 rows come from a development population already used for analysis.
  They cannot establish held-out generalization.
- S3 has zero additions under the frozen cap. It remains a synthesis row over
  the S2-equivalent context, not an evidence-addition population of its own.

## Published artifacts and replay commands

The output root is separate from the frozen retrieval campaign:

```text
eval_results/longmemeval-1m-recall-guarded-cumulative-llm-synthesis-development-20260821/
```

This root is intentionally ignored by Git. The recorded hashes identify the
local campaign outputs; the checked-in implementation, tests, and replay
commands are the public reproduction surface.

The published chain preserves both the raw run and the gold-blind correction:

| Artifact | SHA-256 | Meaning |
| --- | --- | --- |
| `synthesis.json` | `f7132561d861d686364b2c55e522c1bd0c638b5febcb1775a54cf7c5e38d0fe1` | Raw gold-blind completions, forced-choice rows, automatic attribution, runtime reports, and question checkpoints. |
| `synthesis-normalized.json` | `8dd0e3045b6f23d6ebe31386dd38525c2aea05be9ec76d5675854de6913860ea` | Gold-blind derivative that removes meaningless attribution from 27 abstention stage rows without changing raw completions. |
| `scores.json` | recorded beside the raw synthesis | Post-hoc scores for the unnormalized raw artifact; retained for provenance, not used for the headline result. |
| `scores-normalized.json` | `3dae3cc67baaefe6b78dcc632f11c687236b6409df869cc56baee3212fae686c` | Gold-bearing post-hoc score authority for the normalized derivative. |

Both synthesis artifacts bind parent retrieval
`aa22f7c18470d9a7c931fd16f8f58bf67d8566e2298a45371ee2815c11a9bd97`.
The normalization phase is gold-blind; only the score phase loads benchmark
answers and expected sources.

The executable entry point is
`tools/run_recall_guarded_cumulative_synthesis.py`. Its actual phases and flag
names are:

```powershell
$synthesisRoot = "eval_results/longmemeval-1m-recall-guarded-cumulative-llm-synthesis-development-20260821"
$retrieval = "eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821/retrieval.json"
$dataset = "C:\path\to\memory-condense-rig\datasets\longmemeval_s_cleaned.json"

pixi run -e dev python -u tools/run_recall_guarded_cumulative_synthesis.py `
  --phase synthesize `
  --retrieval $retrieval `
  --model-dir .cache/models/Qwen3-0.6B `
  --output-root $synthesisRoot

pixi run -e dev python -u tools/run_recall_guarded_cumulative_synthesis.py `
  --phase normalize `
  --output-root $synthesisRoot

pixi run -e dev python -u tools/run_recall_guarded_cumulative_synthesis.py `
  --phase score `
  --dataset $dataset `
  --synthesis "$synthesisRoot/synthesis-normalized.json" `
  --scores-name scores-normalized.json `
  --output-root $synthesisRoot
```

Question parts under `synthesis-parts/` make the expensive generation phase
restartable. Reopening an already complete root assembles from those sealed
parts rather than generating them again.

The final focused implementation suite was:

```powershell
pixi run -e dev pytest `
  tests/test_recall_guarded_cumulative_synthesis.py `
  tests/test_recall_guarded_cumulative_synthesis_runtime.py -q
```

It reported 21 passing tests. That result establishes contracts and failure
behavior, not answer quality; the measured 0/10 answer result remains the
relevant quality outcome.

## Conclusion

The experiment answered the immediate engineering question but not in the
positive direction. We can use one local LLM checkpoint to score every
episodic addition, synthesize all distinct S1-through-S3 contexts, preserve a
gold firewall, and publish a citation-checked, replayable artifact chain. The
0.6B model and current prompt did not turn that evidence into useful answers.

S2's five additions were all in the lowest forced-choice answerability band
and slightly reduced macro mean p(A); S3 added nothing. The measured evidence
supports keeping S1 as the only additive retrieval stage with any prior
retrieval-side gain, treating S2/S3 as negative diagnostics under this cap,
and changing the synthesis model or prompt only as a new matched experiment.
It does not support promoting these uncalibrated answerability-derived proxy
bands into production packing weights or claiming that the generated citations
establish semantic correctness.
