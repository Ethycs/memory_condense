# Query expansion adds complementary construction but loses it during packing

> **Supersession note, 2026-08-27:** This log preserves the construction-only
> checkpoint as it stood before answer evaluation. The completed query-era
> campaign now records direct query payload at **71/100** as the strongest
> matched result on this analysis-used population; see
> [Research Log 63](63%20-%202026-08-27%20-%20Query-era%20matched%20answer%20campaign.md).

**Status:** the locked-100 query-expansion construction run, its zero-call
replay, and its posthoc source-target audit are complete and sealed. The arm
does not yet have an answer-accuracy score. It is a complementary planning and
routing layer, not a replacement for partition-scan v2.

The provider-free target audit finds **20/27** previously missing eligible
source targets among query candidates, **18/27** after top-40 selection, and
**14/27** after the separate 2,400-token admission budget. Partition-scan v2
remains stronger after packing at 19/27, but the methods are complementary:
their admitted union reaches 21/27 missing targets. Combined with protected S0,
that is **156/162 eligible source targets (96.30%)** and **180/188 targets
across all 100 questions (95.74%)**.

Those percentages are structural source-ID coverage, not semantic QA
accuracy. The current matched answer score remains 54/100 for the separately
measured routed EM fact gate. No >=95% answer result follows from this audit.

## What ran

Each of the 100 dated questions was sent to Terra without S0 evidence, a
reference answer, a benchmark category, a known source prefix, or a gold
source label. Terra returned strict bounded JSON containing query variants,
entities, dates, and evidence operators. The provider phase opened no memory
database or ANN index. A client-free materialization phase then applied those
sealed plans to the entire frozen ten-partition combined store.

The environment required this explicit split: the network-enabled process
could reach the local LiteLLM gateway but could not open the frozen SQLite
stores, while the workspace process could open the stores but could not open
the network socket. The split is a deployment constraint, not a change to the
experiment. Provider journals, materialization, and replay bind the same prompt
population and exact completion bytes.

The executed query operators were:

| Operator | Questions |
| --- | ---: |
| timeline | 43 |
| enumerate repeated events | 41 |
| count distinct | 36 |
| before/after | 27 |
| latest | 26 |
| exact identifier | 21 |
| earliest | 16 |
| state transition | 10 |

All 100 plans parsed. They produced 588 materialized queries, with four to six
queries per question. All 100 rows admitted at least one novel exact chunk;
there were no invalid-plan, missing-query, retrieval-failure, or no-op rows.

## Independent budgets and runtime accounting

| Boundary | Hard cap or observed value |
| --- | ---: |
| query-planning prompt | 2,500 tokens |
| observed maximum query prompt | 315 tokens |
| generated query variants | 4 |
| materialized queries | 6 |
| selected partitions per query | 4 |
| hits per query | 16 |
| candidate union | 96 |
| selected candidates | 40 |
| admitted query evidence | 2,400 tokens |
| observed maximum admitted evidence | 2,400 tokens |
| historical Terra planning calls | 100 |
| posthoc provider calls | 0 |
| retained transformer token state | 0 bytes |

The materialized population contains 5,510 unique candidate memberships,
3,926 selected memberships, and 2,671 admitted exact spans. At the
question/source-membership level the funnel is 3,038 candidate, 2,112
selected, and 1,431 admitted memberships. Exact protected-S0 dedup excluded
zero rows in this run. Admission, not deduplication, dropped 1,255 selected
candidates.

The scope audit is explicit. All preflight, run, row, and routing filter flags
are false. The 588 routing receipts accepted 2,336 cross-question-prefix
candidate source memberships, selected 1,600, and admitted 1,043. Every
question constructed a cross-prefix candidate and 99/100 admitted one. The
runtime therefore did not recover targets by borrowing the known question or
source prefix.

## Corrected missing-source funnel

The target registry was parsed only after the query preflight, frozen stores,
run, reconstructed runtime ledger, S0, eligibility, closure, and both
partition-scan generations had verified. The eligible denominator is the 27
desired sources absent from protected S0 and the raw closure union.

| Method | Candidate | Selected | Admitted |
| --- | ---: | ---: | ---: |
| partition scan v1 | 19/27 | 14/27 | 14/27 |
| partition scan v2-r96 | 19/27 | 19/27 | **19/27** |
| query expansion v1 | **20/27** | 18/27 | 14/27 |
| v2 admitted union query | **24/27** candidate | **23/27** selected | **21/27** admitted |

Across all 30 sources missed by S0, query expansion reaches 21 candidates,
selects 19, and admits 15. The corresponding partition-v2/query unions are
25/30 candidate, 24/30 selected, and 22/30 admitted.

Full source-target composition gives the broader context:

| Composition | Eligible 162 | All 188 |
| --- | ---: | ---: |
| protected S0 | 135 (83.33%) | 158 (84.04%) |
| S0 + partition v2 admitted | 154 (95.06%) | 177 (94.15%) |
| S0 + query admitted | 149 (91.98%) | 173 (92.02%) |
| S0 + v2 + query admitted | **156 (96.30%)** | **180 (95.74%)** |

The query arm should therefore be retained for its two admitted-only rescues
beyond v2 and for its broader pre-packing frontier. It should not replace the
balanced scan, whose admission policy is materially stronger.

## Where the remaining construction loss occurs

Six eligible desired sources remain outside the admitted union:

| Ordinal | Desired sources still missing | Observed boundary |
| ---: | ---: | --- |
| 54 | 1 | absent from both candidate pools |
| 61 | 3 | two absent from candidates; one selected but not admitted |
| 93 | 2 | both query candidates; one selected, neither admitted |

This separates two fixes. Source-balanced repacking can recover the ordinal-61
and ordinal-93 candidates already present before admission. It cannot recover
ordinal 54 or the two absent ordinal-61 sources.

The routing receipts reveal the next additive construction layer. For the
ordinal-54 temporal question, the generated date-aware queries route to the
correct question partition, but the within-partition hybrid search returns
only its top 16 chunks and never constructs the desired episode. Query
planning has therefore solved the partition boundary while local top-k still
loses the fact. The appropriate composition is to reuse the sealed query
routes and run the provider-free exhaustive, source-balanced scan inside those
partitions. This preserves the existing mechanisms in order:

```text
protected S0
  -> query/date/operator planning
  -> global partition routing
  -> exhaustive selected-partition scan
  -> source-balanced packing
  -> exact-cited fact conversion
  -> route-specific answer operation
```

Each arrow is a separately removable stage with its own budget and receipt.

## Claim boundary

Every mapped candidate ID was reconstructed from its immutable database chunk
and checked against its exact source ID, text hash, character coordinates,
timestamp, role, and token count. This proves authentic source provenance.

The target registry labels required source IDs, not answer-bearing character
spans. A hit can therefore be text from the correct conversation while still
omitting the decisive sentence. Source coverage cannot substitute for a
fact-conversion result, a final answer, or an independent semantic judgment.
The forthcoming direct-payload and exact-cited-fact arms are the first tests of
that representation boundary.

## Exact seals

| Artifact or identity | SHA-256 |
| --- | --- |
| frozen retrieval | `e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f` |
| query-expansion preflight | `dc357e4a4e946c541ca5cb278824c376692ba4e4a97a5947c5b18e8da86c5487` |
| query prompt population | `c88a09f1817404d5f29e0cca77fdb260b1479bf004bb8339d543376a3741c02d` |
| query-expansion run / replay | `68f7c0c073c405e33cf019c75e69db1ee5be9b9f3dd84f13cd5a427e6508ba07` |
| query runtime ledger / replay | `16d5ceedee9a86d7c719d3d66538a4d8fa23cf8fbee5763097df69f28afc7c94` |
| source-target analysis / replay | `5fa6ce4931c66c900c42cfc601d2f797b166e9b91b11a3da2de77546bce3a1ec` |
| target plan | `b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff` |
| partition-scan v2-r96 generation | `671f0a3418364f544e61897c42569407805e827ae558980760289dae6b5cf388` |

The query artifacts are under:

```text
eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/matched-eval-spine-v2/s0-plus-query-expansion-v1
```

## Decision

Retain query expansion as a semantic/date/operator planning and routing layer.
Reject its current top-40/greedy admission policy as the final packer. Measure
source-balanced repacking first, then compose its sealed query routes with an
exhaustive partition scan. In parallel, test the already admitted evidence as
raw payload and as exact-cited facts against the same parent and final prompt
cap. Only answer-time positive cells may enter the next locked composition.

The 95% structural source-coverage milestone is useful evidence that the union
of specialized mechanisms spans most desired histories. The 95% semantic QA
gate and the fair Mem0 comparison remain open.
