# Post-selection EM fact memory

**Status:** the streamlined S1 fact-memory experiment completed without
rebuilding the million-token corpus or changing retrieval. Across ten
development questions, facts-only tied raw EM at 6/10 normalized exact match,
raised mean F1 from 0.805372 to 0.827558, and reduced the mean final prompt by
39.45%. Adding the complete raw EM tail back behind the facts fell to 5/10 and
0.755521 while using slightly more tokens than raw alone. This is a measured
development diagnostic, not a locked accuracy result.

## The corrected boundary

The treatment operates after episodic selection:

```text
sealed 1,039,203-token retrieval artifact
  -> S0 selected evidence anchors S1 episode selection
  -> materialize the sealed cumulative S1 selection
  -> EM delta = S1 minus protected S0 evidence
  -> convert only that EM delta to exact-quote-cited facts
  -> answer from one of three matched memory turns
```

S0 is therefore not removed before EM routing. It remains the anchor that
selects S1, and it remains exact in every answer prompt. Only after S1 has been
selected does the answer-time projection remove evidence already present in
S0. Exclusion uses evidence identity and exact `(source_id, text)` equality so
a re-keyed copy of a selected row cannot be repeated as EM. Distinct EM events
remain intact.

The sealed development artifact already has no exact-text collision between
S0 and the S1 delta, and no exact-text duplicate inside that delta. The rule is
still explicit because it defines the representation correctly for future
artifacts rather than relying on this sample's cleanliness.

## Why S1, not final S3

This is a post-retrieval representation experiment, not another retrieval
stack. It consumes the exact `direct_episode_additions` stage:

| Stage | Measured development behavior |
| --- | --- |
| S0 | 10/10 expected-source reachability, 5/10 literal answers, mean context 2,127.4 tokens |
| S1 | Adds 171 episodic rows, keeps 10/10 source reachability, raises best-evidence F1 by 4.62% relative, mean context 6,538.4 tokens |
| S2 | Adds five rows, all previously labeled irrelevant/none, with no scored retrieval gain |
| S3 | Adds no rows under the cap |

Using S3-minus-S0 would silently add S2's five measured low-value rows to the
treatment. S1 is the only additive EM layer with a prior retrieval-side gain,
so it is the fixed source stage here.

## The representation

`fast_em_fact_memory.py` splits a typed sealed question into its protected S0
root and post-selection S1 delta. A compressor sees the dated question and
only the alias-addressed delta. It returns compact JSON facts with one or more
byte-exact quotes from those aliases. The parser rejects unknown aliases,
non-exact quotes, duplicate JSON keys, ungrounded facts, and more than 24
facts. Fact IDs are assigned deterministically after parsing; the model does
not spend tokens or introduce failure modes by inventing identifiers.

Every final prompt uses the same three-message shape:

1. system answer policy;
2. one assistant turn containing retrieved memory;
3. the dated user question requesting a short answer.

The three matched memory turns are:

| Arm | Assistant memory turn |
| --- | --- |
| `payload` | Exact S0 followed by the raw post-selection EM delta |
| `facts` | Exact S0 followed by compact facts and their exact supporting quotes |
| `facts_payload` | Exact S0, facts first, then a bounded raw EM verification tail |

The combined arm prioritizes raw rows cited by the facts when the cap forces a
choice, but preserves original evidence aliases and original order in the
rendered tail. Raw evidence is therefore included without making it the only
format the final model has to interpret.

All final answer prompts use an 8,000-token-proxy hard input cap and reserve
256 output tokens. The separate intermediate compressor also has an 8,000
input cap but allows up to 1,024 output tokens so a cited JSON list cannot be
truncated at the final responder's deliberately short answer allowance.

## Fast runner

`run_fast_1m_em_facts.py` has only three phases:

```powershell
pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_em_facts --phase preflight

pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_em_facts `
  --phase run --enable-provider --authorized-provider-calls 40

pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_em_facts `
  --phase score `
  --dataset C:\Users\Keytone\Downloads\memory-condense-rig\datasets\longmemeval_s_cleaned.json
```

The runner reuses the sealed retrieval reader and the existing concurrent,
checkpointed completion runtime. It performs ten compression calls, then 30
matched answer calls. Completion text lives only in the immutable call
journals; `run.json` stores their identities and hashes instead of duplicating
responses. Gold is not opened until scoring replays both journal populations.
Scoring reports normalized exact match and F1 per arm without adding a judge
campaign.

The real sealed preflight completed with:

```text
questions=10
S1 EM rows=171
mean compression prompt=2,341.5 token proxy
maximum compression prompt=2,803 token proxy
authorized calls=40
provider calls=0
writes=0
```

## Measured result

The explicitly approved gateway run used
`codex_sdk/gpt-5.6-terra` through
`https://central-dev.zt:4000/v1`. It produced 19 facts with 20 exact-quote
citations from the 171-row S1-minus-S0 population. Two questions produced no
EM facts; both remained answerable from the protected S0 root.

| Arm | Exact match | Mean F1 | Mean final prompt | Change from raw prompt |
| --- | ---: | ---: | ---: | ---: |
| `payload` | 6/10 | 0.805372 | 5,238.6 | baseline |
| `facts` | 6/10 | **0.827558** | **3,172.2** | **-39.45%** |
| `facts_payload` | 5/10 | 0.755521 | 5,326.4 | +1.68% |

Every raw row fit in both raw-bearing arms: `payload` and `facts_payload`
selected all 171 EM rows and dropped none. The combined arm therefore did not
act as a selective verification tail on this sample; it recreated all of the
raw clutter and added the facts on top. Its only exact-match loss relative to
raw was `Serenity Yoga and at home via Down Dog` instead of `Serenity Yoga`, a
factually related over-answer encouraged by the additional home-practice
evidence.

A paired answer audit also shows why the exact count is conservative. Four
nominal misses used the correct ordered concert sequence, `3 weeks` for gold
`3`, the correct three-event order in compact wording, and `190 pages` for
gold `190`. No independent judge was authorized for this run, so these are
documented wording observations rather than a replacement semantic score.

The operational choice from this experiment is therefore:

1. use `facts` as the default EM representation;
2. retain `payload` as the matched raw baseline and diagnostic fallback;
3. keep raw inclusion optional, preferably limited to cited rows rather than
   automatically appending the entire EM neighborhood.

Machine-readable artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `eval_results/longmemeval-1m-em-facts-development-20260825/run.json` | `f58ad197a60519c79d3c5d6644db241895f215ebe92bfe840d09f7d67c7ec7be` |
| `eval_results/longmemeval-1m-em-facts-development-20260825/scores.json` | `5c0e532e0c3674e9d5c51dd7f6ced7f49e1736ba25db66b832e2acd5e9c4dd44` |

## Rejected shortcut

The prior Terra v3 synthesis artifact cannot be reused as the EM compressor
checkpoint. An exact audit of its 26 S1 claim citations found that 22 cite S0
root rows and only four cite the S1-minus-S0 delta. Importing those claims
would compress the protected baseline and confound the intended EM-only
transformation. The runner therefore requires fresh delta-only compression.

## Structural cleanup and verification

The two live retrieval paths also stopped carrying private copies of episode
seed serialization and merge ordering. `episode_seed_payload()` and
`combine_episode_seeds()` now live once in the episode retrieval package;
frozen replay validators remain independent.

Focused verification after post-selection deduplication and runner golf:

```text
29 passed in 1.47s
93 passed in 12.16s across the broader EM/retrieval/runtime regression set
sealed preflight passed: 10 questions, 40 calls, zero provider calls/writes
```

The first approved live invocation completed all ten compression calls and
then stopped locally before any answer call: one valid fact cited two distinct
exact quotes from the same evidence row, which the initial parser rejected.
The validator was narrowed to reject only an exact duplicate citation. The
compression prompt and runtime identity did not change. The resumed invocation
replayed all ten compressor journals as checkpoint hits and made only the 30
answer calls. The final lineage therefore contains exactly 40 response
journals, with no repeated compressor request.

## Evidence boundary

This result establishes a representation-efficiency gain and a modest F1 gain,
not higher exact-match accuracy. The development population has ten questions,
has already been used extensively, and received no independent semantic
judgment in this experiment. The locked >=95%/100-question gate and fair Mem0
comparison remain open.
