# Provider-free partition scan repairs missing-source construction

**Status:** provider-free locked-100 retrieval diagnostic complete and sealed;
no answer or judge run has been authorized. The new isolated arm reaches
19/27 eligible source targets that were absent from S0 and both raw closure
pools, selects 14/27 under its own 2,048-token budget, and admits the same
14/27 after post-selection S0 deduplication. This is a construction result,
not an answer-accuracy result.

## Why this arm exists

The v9 closure target report originally presented 135 raw global source hits
against all 188 desired source targets. That full-plan total is valid but
obscures two facts:

- closure ran on 79 question-only-eligible questions, whose denominator is
  162 desired source targets; the other 26 targets belong to ineligible rows;
- S0 already reaches the same 135/162 sources. Raw global closure therefore
  adds **0/27** novel sources over S0, while representative closure reaches
  0/162 registered sources.

The corrected posthoc report now makes that intersection explicit:

| Eligible-source stage | Hit / target | Novel over the 27 S0 misses |
| --- | ---: | ---: |
| protected S0 | 135/162 | -- |
| representative raw closure | 0/162 | 0/27 |
| global raw closure | 135/162 | 0/27 |
| closure union | 135/162 | 0/27 |

This localizes the primary defect to candidate construction, before ordinary
selection or prompt packing can help.

## Runtime design

The arm is deliberately provider-free and gold-blind. For each eligible
question it:

1. combines ordered protected-S0 source signals with global BM25 chunk hits;
2. applies the existing `source_partition_ranking` coarse router;
3. selects at most four top-level source partitions;
4. inspects every non-metadata content row in those selected partitions;
5. keeps one exact query-centred span per source, with full `EvidenceSpan`,
   `make_atom_id`, quote hash, text hash, role, turn, time, and ordinal
   provenance;
6. selects source-diverse candidates under an isolated 2,048-token cap; and
7. only then excludes exact protected-S0 overlaps, recording explicit
   selected-to-protected alias bindings.

The combined stores intentionally concatenate ten histories. A question ID
is evaluation provenance, not a legal routing oracle. Runtime construction
therefore searches the full combined store and never filters a source by the
question-ID prefix. Per-question execution and lifecycle state remain
isolated. The 21 ineligible questions are exact zero-scan no-ops.

Across the locked population, the 79 eligible scans inspected 237,539 content
rows, reduced them to 15,202 one-per-source exact candidates, selected 7,472,
and admitted 7,213 after 259 S0 exclusions. Mean admitted evidence use was
1,971.47 tokens; the observed maximum was exactly 2,048. Retrieval,
generation, loading, and posthoc analysis made zero provider calls.

## The 27-source construction result

The immutable target plan was opened only after the new generation, v9
generation, eligibility manifest, and S0 population had all passed their
runtime validations. The exact 27 eligible sources absent from S0 and both
raw closure pools split as follows:

| Posthoc class | Missing sources |
| --- | ---: |
| `temporal_timeline` | 16 |
| `numeric_reduce` | 10 |
| `state_chain` | 1 |

| Primary owner | Missing sources |
| --- | ---: |
| artifact global | 12 |
| EM | 10 |
| representative episode | 4 |
| S0 | 1 |

| Partition-scan lifecycle | Hit / 27 |
| --- | ---: |
| correct source present among reduced candidates | **19/27** |
| correct source selected before S0 dedup | **14/27** |
| correct source admitted after S0 dedup | **14/27** |

The correct history partition ranked first for 13 missing sources, second for
three, third for three, and outside the selected four for eight. Five sources
were reachable but still lost under the global 2,048-token selection budget.
This cleanly separates the remaining coarse-routing misses from within-scope
ranking/budget misses.

Every one of the 27 sources is also absent from fixed S1 cumulative evidence
and from the sealed selected EM-source delta. The historical artifacts did
not persist the full raw EM candidate universe, so the report does not claim
that a source was absent before EM selection.

## Seals and calls

| Artifact | SHA-256 | Provider calls |
| --- | --- | ---: |
| frozen base retrieval | `e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f` | 0 |
| v9 eligibility manifest | `748bd56a7efb8fd70d36bc96f099a53fc506469565577de9635908f6773bdee1` | 0 |
| v9 closure generation | `cf541c40f0749dcf9e436080c56dcf251232fd9ac7c844be49e2dfd8764a7ee5` | 0 |
| pinned target-owner plan | `b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff` | 0 |
| partition-scan generation | `48c9f0b5eb2eb8f49a47002ce0beed843bbb6b478b45bf311d5c8d6c6e34f3f4` | 0 |
| missing-source posthoc analysis | `01248bc78a1721951cc1131f36707516701bbbe5a50481f6a75f930e196670df` | 0 |
| corrected closure target analysis v2 | `ed5c189069d64f51fcae13063a8586b5c26052207022560e47bf2f6ddfc1e0dc` | 0 |

The generation and missing-source artifacts live under:

```text
eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/partition-scan-v1
```

The corrected closure audit is
`independent-closure-v9/target-analysis-v2.json`. Focused contract,
round-trip, adapter, gold-firewall, and eligible-denominator tests pass 10/10.

## Claim boundary and next step

Source-ID reach proves that the arm found some exact excerpt from the desired
source history. It **does not prove that the selected excerpt contains the
answer-bearing fact**, that the final LLM will use it correctly, or that
semantic answer accuracy increased. Exact-span provenance proves the emitted
text is authentic; it does not prove relevance.

Accordingly, no result here is added to the 53/100 matched control and there
is no new 95% claim. The next efficient experiment is provider-free: ablate
coarse query expansion for the eight partition misses and partition-balanced
selection for the five reachable-but-unselected sources. Only a sealed,
gold-blind variant that improves structural reach should receive an answer
and changed-only judge budget.
