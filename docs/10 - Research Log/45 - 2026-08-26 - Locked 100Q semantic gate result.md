# Locked 100Q semantic gate result

**Status:** the preregistered fixed-S1 validation gate completed at **56/100
independent Sol semantic accuracy**. The population satisfies the 100-question
minimum but misses the >=95% target by 39 correct answers. The live run used
exactly 100 unique Sol calls with zero retries. A provider-free replay made
zero physical calls, reproduced 56/100, and republished byte-identical artifact
SHA-256 `5dc56a240315c5577d1032d40429df7e39adad0f40a098abc371ee2ea2ec77df`.

## Exact execution and replay

Each judge call carried one locked validation question, its reference answer,
and its already sealed Terra prediction. Gold was available to the independent
judge only; it was not present in the fixed-S1 responder population.

| Phase | Logical | Unique | Physical | Checkpoint hits | Retries |
| --- | ---: | ---: | ---: | ---: | ---: |
| live Sol judge | 100 | 100 | 100 | 0 | 0 |
| provider-free replay | 100 | 100 | 0 | 100 | 0 |

The journal contains 100 request and 100 response files with no temporary
files. The largest judge prompt was 222 proxy tokens. The live run used 13,985
input and 1,970 output proxy tokens over 435.53 seconds. Provider-reported
token usage was unavailable.

## Formal gate

| Measure | Result |
| --- | ---: |
| Questions | 100 |
| Correct | **56** |
| Incorrect | 44 |
| Semantic accuracy | **56.0%** |
| Required correct | 95 |
| Target | >=95% |
| Population minimum met | yes |
| Gate passed | **no** |
| Status | `below_accuracy_target` |

Reaching 95% on this same population would require correcting 39 of the 44
misses while preserving every current success.

## Category diagnostics

| Category | Correct | Accuracy |
| --- | ---: | ---: |
| knowledge-update | 15/16 | 93.75% |
| multi-session | 9/27 | 33.33% |
| single-session-assistant | 6/11 | 54.55% |
| single-session-preference | 1/6 | 16.67% |
| single-session-user | 12/14 | 85.71% |
| temporal-reasoning | 13/26 | 50.00% |
| **overall** | **56/100** | **56.00%** |

These category rows are diagnostics, not separate gates; none independently
meets the preregistered 100-question minimum.

## What the gap means

The semantic judge accepted all 33 normalized-exact answers plus 23 additional
non-exact answers. Exact match therefore understated quality, but it did not
hide a near-95% system.

Of the 27 explicit short abstentions, 25 were incorrect and two were
semantically correct. Abstentions account for 25/44 failures. Multi-session
and temporal-reasoning questions account for 31/44 failures. Preference has
the lowest rate, although its six-question denominator is small.

The locked retrieval diagnostic from
[Research Log 43](43%20-%202026-08-26%20-%20EM%20v2%20result%20and%20locked%20100Q%20retrieval%20merge.md)
already showed that fixed S1 contained at least one labeled source for 91/100
questions and every labeled source for 81/100, while literal answer text was
present for only 50/100. The 44 answer failures therefore cannot all be labeled
retrieval misses without question-level analysis. The useful separation is:

1. required evidence never reached fixed S1;
2. evidence reached S1 but was obscured or unused;
3. temporal calculation or event ordering failed;
4. preference or multi-session aggregation failed;
5. the answer was present but answer selection or formatting lost it.

## Question-level reachability cross-tab

The sealed judge rows were joined posthoc to the fixed-S1 retrieval diagnostics.
This does not alter either artifact or make another provider call.

| Fixed-S1 diagnostic | Correct | Incorrect | Questions | Semantic accuracy |
| --- | ---: | ---: | ---: | ---: |
| literal answer hit | 35 | 15 | 50 | 70.00% |
| no literal answer hit | 21 | 29 | 50 | 42.00% |
| every labeled source present | 53 | 28 | 81 | 65.43% |
| some, but not all, labeled sources present | 3 | 7 | 10 | 30.00% |
| no labeled source present | 0 | 9 | 9 | 0.00% |

All nine zero-source questions failed, so retrieval remains a hard constraint.
However, 28/44 misses had every labeled source and 15/44 contained a literal
answer span. Thirteen of the 25 incorrect abstentions had every labeled source;
the other twelve had an incomplete source set. Those full-source and literal
misses are the clearest first targets for fact conversion, evidence-density
ranking, and answer reasoning before adding retrieval breadth.

A grounded path to 95% cannot be synthesis-only. Repairing all 35 misses that
already have at least one labeled source would reach 91/100; at least four of
the nine zero-source cases must also recover the missing source evidence.

Source coverage is only a session-level diagnostic, not proof that the decisive
turn or fact was selected. For example, a row can have 100% source coverage
while omitting the required fact. The cross-tab therefore locates the boundary
between likely retrieval and synthesis work; it is not a causal classification
of every individual miss.

For implementation routing, the 44 sealed misses partition mechanically as
follows. The first four rows use only source coverage and literal-answer
presence; the final row is the sole manually flagged judge-ambiguity candidate.

| Mutually exclusive miss bucket | Questions | Abstained | Answered |
| --- | ---: | ---: | ---: |
| no labeled source in S1 | 9 | 7 | 2 |
| some but not all labeled sources in S1 | 7 | 5 | 2 |
| all labeled sources plus a literal gold span | 13 | 5 | 8 |
| all labeled sources, no literal gold span | 14 | 8 | 6 |
| strongest judge-ambiguity candidate | 1 | 0 | 1 |
| **total** | **44** | **25** | **19** |

Representative cases show why the buckets need different treatments:

- `bc8a6e93` had no labeled source and abstained on the birthday-cake answer;
- `bc149d6b` recovered one of two sources and answered 50 rather than 70 pounds;
- `gpt4_fa19884d` had the literal bluegrass/banjo evidence but abstained;
- `gpt4_7a0daae1` had the March 10 and March 17 events but did not compute one week;
- `352ab8bd` had nominal full-source coverage while the required 20% HAMT fact
  was absent, illustrating the session-level metric limitation.

`06878be2` is the strongest semantic-review candidate because its compatible
accessory recommendations arguably satisfy a broad reference despite including
non-Sony brands. It remains incorrect in the sealed score: there is no posthoc
appeal path, and changing that one verdict would not affect the failed gate.

The 16 incomplete-source misses contain nine temporal, four multi-session, two
preference, and one single-session-user question. This gives retrieval work a
specific initial population rather than treating every judged miss as a reason
to widen the neighborhood.

## Artifact identities

| Artifact or population | SHA-256 |
| --- | --- |
| Sol semantic judge and replay | `5dc56a240315c5577d1032d40429df7e39adad0f40a098abc371ee2ea2ec77df` |
| judge campaign binding | `84c871adac4b73bf4a40103c49b624227e4a12acb104d9335c3f9492171068da` |
| judge prompt population | `4b22b8240723866bf3bcb72f039db733ac5d0573e41fdfd5e393a66370722011` |
| ordered judgment population | `ee4352543059fd2520ca67f6d3cecbcb6d20d94f7561da8e8f7f9266374e066d` |
| judge runtime identity | `d3b32750ac1f812acfa7da95495391152dc9e6ed6169a6a212e886aedfc936c4` |
| sealed Terra answers | `d7fc47b8d1f372f002230c6ffe489dac8cd11bd71b35b8d3008b1255da2a38cd` |
| locked retrieval | `e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f` |
| locked population | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| gold scoring population | `128b1d2e478a6f73a86ab69952c412d175365df1cf4baa5ac76d4cef6ac0bfb3` |

## Decision

The dev10 10/10 facts-only result did not establish generalization to the
locked fixed-S1 population. This run measured the sealed raw fixed-S1 baseline;
it did not run the v2 fact converter over all 100 questions. Global raw
reinjection remains unsupported.

The next improvement should classify the sealed misses, apply v2 fact
conversion or cited raw fallback only where S1 already contains the answer,
add explicit temporal and preference synthesis where appropriate, and change
retrieval only when the required evidence is absent. This preserves the
cumulative design: improve representation and reasoning over already selected
evidence before adding a more complex retrieval layer.

This validation population is now analysis-used. Any tuned system needs a new
untouched confirmation population before making a competitive claim. The fair
Mem0 comparison remains unrun.
