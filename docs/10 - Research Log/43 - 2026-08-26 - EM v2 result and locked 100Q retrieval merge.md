# EM v2 result and locked 100Q retrieval merge

**Status:** both authorized development populations completed exactly, the
facts-only v2 treatment improved normalized exact match from 6/10 to 7/10 and
mean F1 from 0.827558 to 0.914065, and the prior v1 facts arm received a 10/10
independent Sol semantic score. Separately, all ten locked provider-free
validation shards merged into a canonical 100-question retrieval artifact.
The v2 semantic judge and the 100-question Terra answer stage are preflighted
but not authorized or run.

**Follow-up:** the v2 judge later confirmed 10/10, and the locked 100-question
Terra answer population completed. See
[Research Log 44](44%20-%202026-08-26%20-%20V2%20semantic%20confirmation%20and%20locked%20100Q%20answers.md).

## Development execution result

The two populations explicitly authorized after
[Research Log 42](42%20-%202026-08-26%20-%20Streamlined%20EM%20v2%20and%20independent%20semantic%20scoring.md)
completed without retries:

| Population | Logical calls | Unique physical calls | Checkpoint hits | Retries |
| --- | ---: | ---: | ---: | ---: |
| v1 Sol semantic judge | 30 | 15 | 0 | 0 |
| v2 Terra compression plus facts-only answer | 20 | 20 | 0 | 0 |

The provider-free Sol replay then used zero physical calls and all 15 sealed
checkpoints. It reproduced the same campaign, prompt population, completions,
judgments, and aggregates.

### V1 semantic score

| Arm | Normalized exact match | Mean F1 | Independent Sol |
| --- | ---: | ---: | ---: |
| raw `payload` | 6/10 | 0.805372 | **10/10** |
| `facts` | 6/10 | 0.827558 | **10/10** |
| `facts_payload` | 5/10 | 0.755521 | **9/10** |

The sole semantic negative is the same contamination already identified in
Research Log 41: for the yoga-location question, the full raw tail caused the
combined arm to append Down Dog to the correct `Serenity Yoga`. Raw EM therefore
adds no semantic recall on this slice and creates one genuine precision loss.

The difference between 6/10 lexical exact match and 10/10 semantic accuracy
also confirms that exact string matching was understating answer quality. Its
misses included correct units and correct ordered events expressed in a more
compact form.

### V2 facts-only score

V2 made ten compression calls and ten answer calls. Its immutable run and
provider-free score replay report:

| Metric | V1 facts | V2 facts-only | Change |
| --- | ---: | ---: | ---: |
| Normalized exact match | 6/10 | **7/10** | +1 question |
| Mean F1 | 0.827558 | **0.914065** | +0.086507 absolute |
| Mean final prompt | 3,172.2 | 3,190.6 | +0.58% |
| Reduction from raw v1 prompt | 39.45% | 39.09% | essentially retained |

The v2 compressor reduced the 171-row post-selection EM population to 17
facts with 17 source-exact citations covering 15 unique rows. All 17 facts fit
the final prompts. Two questions produced no EM facts and were still answered
correctly from protected S0. Facts-only reinjected zero raw EM rows.

| Prompt population | Minimum | Mean | Maximum | Output reserve |
| --- | ---: | ---: | ---: | ---: |
| Terra compression | 1,467 | 2,507.5 | 2,969 | 1,024 |
| Terra facts-only answer | 1,218 | 3,190.6 | 3,519 | 256 |

The three remaining exact-string negatives are all locally identifiable as
form differences:

- the correct five-event concert sequence, abbreviated and comma separated;
- `4` versus reference `4 days.` after the gold-blind policy requested a bare
  number for a question that already named the unit;
- the correct three-event order without the reference's connective prose.

This makes 10/10 semantic accuracy a strong expectation, not a result. The
real v2 Sol preflight contains ten logical and ten unique prompts, peaks at 232
tokens, and made zero calls or writes. It needs a separate ten-call approval.

The causal attribution is also limited. Protected S0 supplied complete answers
for the two questions with zero EM facts and contributed material to several
list answers whose compressed EM facts were incomplete. Some v2 facts still
contained distractors such as alternate online-yoga options and a beach-house
plan. V2 therefore improves the final lexical score and preserves efficiency;
it does not yet establish that EM compression itself improved semantic recall.

## Locked 100-question retrieval merge

Offsets 70, 80, and 90 finished after the previously sealed offsets 0--60.
Every shard published, all canonical shard bytes match the ordered merge
receipts, and the self-contained merged validator passed under the exact
frozen implementation and environment identities.

| Identity | SHA-256 |
| --- | --- |
| merged retrieval | `e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f` |
| locked population | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| frozen retrieval implementation | `cf2577f21a7a1af1b9c5f331c7eb1672c5ba13af84ccdc2f718259192bb36e09` |
| frozen environment lock | `c99f82aa37a1df009c6100110d24aeaee0bde3376ff09537700ad9848b371a95` |

The post-merge audit validated the artifact without gold first, then
reconstructed the locked validation population and computed provider-free
retrieval diagnostics:

| Stage | Literal hits | Best-evidence F1 | Source recall | All-source hits | Added rows / questions | Mean context | Max context | Max prompt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | 48/100 | 0.115933 | 85.75% | 81/100 | root | 2,275.68 | 2,364 | 2,698 |
| S1 | **50/100** | **0.118263** | 85.75% | 81/100 | +1,727 / 100 | 6,820.86 | 7,000 | 7,353 |
| S2 | 50/100 | 0.118263 | 85.75% | 81/100 | +22 / 7 | 6,901.60 | 7,000 | 7,353 |
| S3 | 50/100 | 0.118263 | 85.75% | 81/100 | +2 / 1 | 6,904.70 | 7,000 | 7,353 |

At least one labeled source was present for 91/100 questions at every stage.
Answer-component recall was unchanged at 0.333333, but only two questions were
eligible for that metric and neither recovered every component. Every context
and prompt respected its cap; the largest prompt plus the 256-token responder
reserve was 7,609 tokens.

The important result is the shape of the marginal return. S1 spent 1,727 new
rows to gain two literal hits and 0.002330 mean best-evidence F1, with no
source-recall gain. S2 and S3 spent another 24 rows and gained nothing on the
available retrieval metrics. This does not prove those rows cannot improve an
LLM answer, but it rules out the idea that simply adding more retrieved text is
the path to 95%.

## Answer-stage readiness

The merged artifact passed the actual fixed-S1 final-answer preflight:

```text
questions=100
unique Terra prompts=100
campaign binding=864fdff50e404d4ea4b1e081d78beb8549f2a12a48b55d758f929fee4b388623
provider calls=0
```

That answer population would send each locked validation question and its
fixed-S1 retrieved evidence to Terra without gold. It has not been authorized.
The independent 100-question Sol judge cannot be preflighted until those
answers exist and are sealed.

## Artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| v1 Sol judge | `63b96dd3e52ecbb8756db3fb44e0c8cd051823165c9f4700080bd42083c2e1af` |
| v1 Sol no-call replay | `484e8c15adab1a864ef1fd700ea473f23fb8bd34a69cd7e49512ca02dfe13b60` |
| v2 Terra run | `0115d14c77607f98830df299438a2fee9651d2bd87d794d8d6f17953318bba16` |
| v2 local scores | `91e615d892a5f2d1d80a1eda6f845a1b4637043464599cad614bbcb2f182a8e6` |
| locked 100Q retrieval | `e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f` |

All listed artifacts have matching canonical sidecars. Independent audits
found no orphan journals, request/response binding error, retry, prompt-cap
violation, or population mismatch.

## Decision

V1 facts-only remains the correct EM representation on this evidence: it
preserves raw payload's 10/10 semantic result, while the full raw tail should
not be restored globally. V2 discards more than 90% of the selected EM rows and
improves lexical conformity without materially increasing prompt size, but its
semantic result and its causal gain over protected S0 remain open.

The next measurements are now narrow and explicit:

1. run the ten-call independent v2 semantic judge;
2. run the already-preflighted 100-call fixed-S1 validation responder to learn
   the actual answer bottleneck on the locked population;
3. judge that sealed 100-question answer population independently;
4. use judged failure categories to decide whether each miss needs better fact
   compression, a cited-row fallback, temporal calculation, or genuinely new
   retrieval evidence.

The 95% semantic gate remains unproved. The corpus and retrieval apparatus are
no longer blocking it: the remaining gate is answer generation plus independent
judgment on the sealed 100-question population.
