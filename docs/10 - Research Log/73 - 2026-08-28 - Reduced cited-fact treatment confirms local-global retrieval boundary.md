# Reduced cited-fact treatment confirms local-global retrieval boundary

**Date:** 2026-08-28

**Status:** sealed post-hoc remaining-24 diagnostic complete; 0/24 rescues;
no full-100 promotion claim

## Question tested

The reduced oracle-source control recovered 17/24 remaining misses when the
labelled raw source sessions replaced the million-token retrieval result. This
treatment tested the narrower alternative explanation: perhaps the existing
retrieval packet already contained enough evidence, but its representation was
too diffuse for the final model.

The treatment therefore held the shared-surplus v3 retrieval composition fixed
and added one LLM-assisted conversion:

```text
existing selected typed evidence
  -> Terra exact-cited fact compiler
  -> provider-free grounding, dedup, density ranking, and hard fit
  -> Terra facts-only answer candidate
  -> existing protected-parent validator
```

No benchmark reference, accepted answer, judge result, or prior verdict entered
the compiler or answer prompts. The compiler did not see the parent prediction.
The answer model saw the parent only as fallback-not-evidence. Sol opened the
reference only after the answer run and replay were sealed.

This remains a post-hoc, outcome-conditioned diagnostic over the 24 questions
still wrong after the miss-27 treatment. It is not an official replacement arm.

## Reduced and guarded execution

The compiler prompt requested at most 12 atomic facts. Every citation had to be
an exact substring of a currently admitted summary carrying an opaque `H`
handle. Unsupported structured fields were stripped; handle mismatches,
non-exact quotes, and incompatible story groups remained fatal to that sibling.
Answer-facing summaries were exact cited bytes, not unchecked paraphrases.

The first strict materialization exposed a useful validation problem: 42/53
rejections were harmless `status: "unknown"` values where the schema requested
`null`. The already sealed raw responses were reparsed offline with field-level
salvage and citation-derived slots. This made no additional compiler calls and
produced 21 valid packets containing 71 retained facts with zero rejected
siblings. Ordinals 69, 72, and 97 remained invalid because required evidence
slots were genuinely unresolved.

Only the 21 valid packets entered the answer provider population. The other
three rows kept their parent locally, so no duplicate answer calls were spent on
known-incomplete packets. Only four final predictions differed from their
parents. The other 20 parent hashes matched the sealed miss-27 judge authority,
where all 20 were already incorrect, so Sol judged only the four changed rows.

| Plane | Physical calls | Maximum complete envelope |
| --- | ---: | ---: |
| Terra fact compiler | 24 | 7,945 / 8,000 |
| Terra sparse fact answer | 21 | 4,619 / 8,000 |
| Sol changed-only judge | 4 | within the common judge cap |
| replay/materialization | 0 | n/a |

All planes retain zero transformer token state.

## Sealed artifacts

| Artifact | SHA-256 |
| --- | --- |
| corrected compiler preflight | `c020b625011e67a71112b952a60c49f627f817a5c68a4155ff6c780bd8b44fc2` |
| raw compiler run | `2de0f0d27c6b08510fdc4e799dcfa8914cf5cf53a02de9fce3c1974d202c85b2` |
| raw compiler replay | `a35e5c05e1e006bab943a85db4a1f4a89e6bab669354a9021118ebb4c7469720` |
| offline rematerialization | `0de64b078bf8fdb5977e2f4d0f8fe89bed1b0a122dad1febba03e0445fd9f729` |
| rematerialization replay | `d2433122b2afc472b4853486615a10dc4e9f9a13f5ce1e1a5defec740b61f72a` |
| sparse answer preflight | `6e6fd12b86d11be3b1d0a948ead2ff0e51afa70ea36e2ca8420ecd53b08574d9` |
| sparse answer run | `7ac80bfad4fbeabe43300fa706f6b0b10379140dabfbd643fbfbec4522a27765` |
| sparse answer replay | `6e25ac40c17c0bac90014b2378b5fa03d3a0c858e4d4a2010a0f09a5344e96f0` |
| changed-four Sol preflight | `20bf1beeb4c11a21a36e04ccd4105bd08cb7bc4337f9d9d8002937ffdf4e4692` |
| changed-four Sol judgment/replay | `496e005eeee790bb655febd98bf4eff06aedba563141076f69785789502a40cc` |
| changed-four score/replay | `8250696e91a19c5bc09fd3524e3e24e9a8159047142d6f4f43c6080351790521` |

The compiler, sparse lifecycle, and changed-only judge adapter have 23 focused
tests across their three test modules. The compiler and sparse answer runs also
replay byte-identically from immutable checkpoints.

## Result

The treatment rescued **0/24** remaining misses.

Terra changed only ordinals 36, 43, 54, and 81. Sol rejected all four:

| Ordinal | Replacement | Failure |
| ---: | --- | --- |
| 36 | `Watch The OA tonight.` | distractor rather than a storytelling-oriented Netflix stand-up special |
| 43 | peace lily and succulent purchase | wrong event; the target is planting 12 tomato saplings |
| 54 | new toaster | wrong object; the target is a smoker |
| 81 | restatement of summer-cocktail interest | omits the mixology/Pimm's-aware recommendation |

Three additional `replace` completions normalized to the existing parent and
one failed the existing personalization citation validator. Thirteen valid
packet completions explicitly kept the parent. The three invalid packets were
local fallbacks. Therefore no selected prediction became correct, and the
protected 73/100 full result remains the system of record. No new full-100 run
was warranted.

## Failure assay

A post-hoc comparison against the sealed oracle-source authority partitions all
24 rows:

| Dominant boundary | Count | Ordinals |
| --- | ---: | --- |
| decisive target absent from compiler input | 8 | 7, 28, 31, 49, 53, 61, 86, 93 |
| compiler selected a distractor despite target presence | 4 | 6, 69, 77, 79 |
| answer kept the parent despite sufficient compiled facts | 5 | 14, 16, 42, 65, 67 |
| accepted replacement used a distractor | 4 | 36, 43, 54, 81 |
| unresolved or benchmark-evidence conflict | 3 | 72, 94, 97 |

Concrete examples make the boundary clear:

- q6's compiler input contained the exact bluegrass/banjo statement, but the
  packet retained only the weaker statement that the band name was unspecified.
- q69's input contained the navy-blazer pickup, but compilation retained only
  two boot actions and left the cardinality frontier unresolved.
- q77 contained the October Science Museum visit but selected a recent art
  museum distractor.
- q79 contained the $800 handbag but selected the $2,000 Gucci distractor.
- q14, q16, q65, and q67 had enough compiled operands or state facts, yet the
  parent-visible answer prompt stayed conservative.

For q43 the failure occurs even earlier: the fixed compiler input contains the
peace-lily/succulent event but contains no tomato or sapling text. A fact
compiler cannot recover evidence that the retrieval packet never supplied.

## Attribution

Together the three reduced controls isolate the dominant cause:

1. Reducing aggregate answer workload by 71.9% while keeping prompts identical
   produced no evidence-driven rescue: process or batch memory pressure is not
   supported.
2. Replacing large-store discovery with the labelled source sessions recovered
   17/24: source localization and evidence dilution are dominant.
3. Compiling the existing retrieved packet into exact-cited facts recovered
   0/24: compression alone does not repair the wrong neighborhood.

The current failure is therefore principally **local-to-global connectivity and
ranking**, with a secondary answer-arbitration problem. It is not a need for a
larger context window, another 1M ingest, or more generic top-k tuning.

## Next architecture

The existing full-store index is sufficient infrastructure. All ten auditable
missing-at-selection questions already have at least one selected citation in
the correct enclosing history component. The current active scan loses the
target because eight hop slots compete against 2,592--38,427 candidates and
source affinity outranks evidence density.

The next treatment should therefore reuse the resident index:

```text
selected cited fact
  -> low-fanout entity/action/time cue
  -> one bounded global posting read
  -> exact enclosing turn/source hydration
  -> post-selection dedup
  -> cited fact compilation
  -> parent-free candidate answer
  -> external protected-parent arbitration
```

Required changes are narrow:

1. rank cues by unresolved operator support, specificity, and inverse posting
   fanout rather than stable receipt order;
2. reserve separate exact-source, enclosing-history, and global fact-cue
   subchannels inside the same active budget;
3. hydrate the exact enclosing turn after a sentence wins;
4. generate the answer candidate without exposing the parent, then apply the
   existing validator afterward; and
5. add deterministic incomplete-operand handling for cases such as q72 without
   weakening provenance or treating a bounded frontier as globally closed.

Fact compilation remains useful after source reconstruction, but this result
rules it out as a substitute for reconstruction.
