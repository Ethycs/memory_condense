# Summary passage addresses and preserved user evidence

**Date:** 2026-09-09 (local; execution observations are September 10 UTC)  
**Status:** 30-question evidence audit and preparation passed; fresh answers pending  
**Predecessor:** [141 - Development30 result and omitted user evidence](141%20-%202026-09-09%20-%20Development30%20result%20and%20omitted%20user%20evidence.md)

The latest judged result remains **25/30**, versus 23/30 for its fresh controls.
The new work below measures evidence selection, not improved answer accuracy.
The historical 95/100 already covered roughly 1M-token memories, but used a
cumulative retrieval and answer-repair lineage without demonstrating API-like
end-to-end latency. The paired historical comparison is 95/100 versus 73/100,
with 23 losses and one gain; 18 of the losses are multi-session or temporal.
Those labels must not select runtime routes or become cached answers. See
[132 - Joint target](132%20-%202026-09-09%20-%20Joint%201M%20accuracy%20and%20latency%20target.md).

## Measured routing failures

The complete third-memory frontier audit reproduces all ten frozen overflow
prompts. Each of the three diagnostic witness leaves is absent from the original
route, so additional expansion of already selected leaves cannot recover them.

| Witness | Combined summary dense rank | Whole user summary rank | New passage rank |
| --- | ---: | ---: | ---: |
| Started Disney+ free trial | 77 | 228 | 8 |
| Painting studio and online tutorials | 10 | 5 | 1 |
| Commuter bike tire replacement before April | 10 | 5 | 14 |

Removing generic query words worsened the user-summary ranks to 384, 20 and 9.
That query rewrite is rejected. The new candidate retains the original query
and the previously existing chronology content view only where applicable.

Artifacts under `eval_results/full1m-source-spine-overflow-development30-20260910-r1`:

- Original frontier: `frontier-offset020.json`, SHA
  `f0364d037f429265a67abe568007fc19f60b8ee11821f3c97d9af60a3acafb07`.
- Rejected query view: `focus-frontier-offset020.json`, SHA
  `b9f240b8417296e00e95504cfd15862fdd4e3d652143a8885fd323f2c84a376b`.
- New passage diagnostic: `facet-frontier-offset020.json`, SHA
  `6d2ae790465bd35a60f8cbbe1ddf80c480f5bd31880069d17c6bad0d27a35377`.

Witness annotations are opened after selection in the new diagnostic. They do
not influence embedding, ranking, admission or packet construction. The passage
index includes the entire namespace, without benchmark-question selection.

## Additional addresses and exact hydration

`spine_summary_facets.py` creates exact passages within stored user summaries.
Sentence boundaries and selected conjunction boundaries separate a stated fact
from a subsequent request; long passages use overlapping 48-word windows with
a 24-word stride. The original whole-summary index remains available. These
passages are search addresses, not raw evidence or new asserted facts.

BGE embeds these summary passages at ingest. Query-time selection shares the
original query vector with the existing search and ranks each original leaf
by its strongest passage cosine. The top eight passage leaves and top eight
whole-user-summary leaves are interleaved by rank and deduplicated. Original
leaf descriptors remain unchanged. No Qwen or other completion call is added
at query time. Qwen continues to process summaries only during hierarchy work.

`SourceSpineSupplement` first computes the existing overflow plan, preserves its
entire user prefix, then offers additional user turns selected by the two new
address lists, followed by the previous attached context. The existing hydrator
authenticates whole raw user turns and applies the same 3,072-token/128-span
budget. Additional users can displace assistant context, so evidence retention
does not guarantee answer non-regression.

All three third-memory witnesses now appear in the exact raw candidate packets.
The commuter bike statement is recovered through the wider whole-summary list;
its passage rank 14 is outside the passage cutoff. Every previously hydrated
user statement is retained byte-for-byte on all 30 development questions.
The painting answer also mishandles an already present 30-day challenge, so
recovering its missing turn alone may not resolve the entire answer failure.

All three complete memories have compiled passage indexes:

| Offset | Leaves | Passages | Address manifest SHA |
| --- | ---: | ---: | --- |
| 0 | 2,744 | 5,843 | `3ca328035c6c9e698847e21750a39e5ca678e64862b2124d77f1c125db8de416` |
| 10 | 2,625 | 5,734 | `3f62d6a4adf8e8809a60b4b5bbb8b76dd198c22620a1418ecfc0ce016e313402` |
| 20 | 2,693 | 5,807 | `2fc19c9c9b874b14f8f2a5a6202d7029c20dba30fe2b92b3e8ccbfcc6962b732` |

Roots follow `eval_results/full1m-spine-facet-addresses-offsetNNN-20260910-r1`.
Compilation used zero provider calls and no raw or question inputs. The sealed
compiler and facet implementation hashes must remain unchanged for replay.

## Fresh comparison and continuation

`tools/evaluate_source_spine_facets.py` compares `source_spine_overflow` with
`source_spine_facets`, using the same reader-v2, model, budgets and question
population. Both have adjacent identical-evidence API controls, plus short API
chat, with counterbalanced pair order. Each live memory call recomputes query
embedding, routing and hydration inside its clock. The streamed predictions
are the predictions judged; saved correct benchmark answers are never reused.

Thirty focused tests pass, covering passage identity, exact raw hydration,
retained user evidence, foreign descriptor rejection, matched controls, live
retrieval, incomplete answer rejection and preservation of uncertain requests.

`prepare_facet_comparison.py` in the development30 artifact directory completed
all 30 question audits and prepared 150 fresh requests. Every baseline prompt
matches its original control and every candidate prompt matches its evidence
audit. All previous user evidence is retained exactly within the same budget.
Session **20983** completed successfully. No fresh answer or judge calls have
run for this candidate.

The prepared roots follow
`eval_results/full1m-source-spine-facets-joint-offsetNNN-20260910-r1`:

| Offset | Fresh comparison preflight SHA | Evidence audit SHA |
| --- | --- | --- |
| 0 | `f550b84452ecb59cbb2506ef97645a6e88a67f893c4c9cbdedae9aa7b581de58` | `8ec7b2907ee0fbc32e58b220f3d000079d4279bdd27bf1bfcae816b94fc09ee6` |
| 10 | `f87357338bad2fcbc6591fce9e383aa84bf96823584bda4a02d6b6868653c1ba` | `e6cc583069ec0af253d3b888c7a805b7cc55940b6b6cf885480858729d8a61ea` |
| 20 | `c554f0755be327260c7a6b112287065dd408eb13564b4c0842fe031c4c94f4c1` | `6d2ae790465bd35a60f8cbbe1ddf80c480f5bd31880069d17c6bad0d27a35377` |

`facet-development30-preparation.json` SHA:
`54edbddd4fc3277130335f83789e50e5643c57c7226e4f86fc7949964c62d7d6`.
Its repeat preparation completed with zero provider calls and reproduced the
same audit checks, comparison preflights and preparation report byte-for-byte.
All source files in the new comparison preflights are now frozen. Make any
further serving change in a new version with a new comparison identity.

Fourth-memory raw ingest remains live in session **76404**, using the original
838-request preflight recorded in Log 141. Its strict quote diagnostics are not
final admission verdicts. Preserve all responses and reservations, then audit
the complete namespace before source admission and hierarchy compilation.

Do not run timed API comparisons while bulk provider work or GPU compilation is
active. Once preparation and bulk ingest finish, run and judge the complete
prepared population for both methods. A development improvement still requires
a consistent full100 accuracy-and-latency evaluation and subsequent untouched
confirmation. The joint 95% target remains unmet.
