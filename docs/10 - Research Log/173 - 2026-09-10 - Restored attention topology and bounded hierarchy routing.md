# Restored attention topology and bounded hierarchy routing

Log 172 found that every serving section was a leaf. The compiler had explicitly
deferred parent summaries, while preserving the original attention cuts in each
leaf projection. This successor restores that topology and adds a query path
that actually visits parent summaries before selecting raw-addressable leaves.
It does not change the frozen control or claim an accuracy improvement.

## Implemented path

`spine_parent_hierarchy.py` reconstructs source-local binary parents from the
saved cut positions and the ordered atomic span descriptors. It validates every
cut and the complete leaf partition before any summarization. Original leaf
IDs, summaries, spans and receipts remain identical. Source identities are
ownership constraints, never lexical relevance features. Dates do not reorder
the transcript. Each source retains its own root.

Parent merges consume only the two explicit generated summary channels. User
summaries merge independently; attached material retains its attribution and
is summarized relative to the merged user spine. Inputs with extra fields,
including raw-support-quote fields, are rejected. The compiler has no raw-text
loader, question reader or reference reader. Each channel is bounded to 128
tokens. Existing fitting merges remain lossless; larger merges use the existing
summary-only Qwen journal and bounded recovery policy.

`bounded_spine_hierarchy.py` initially nominates at most eight source roots by
their best eligible descendant's existing dense summary score. Qwen then
selects a beam of four using parent summaries and descends the selected branches.
Each level uses one complete workspace of at most eight summary candidates.
Leaves compete with newly expanded children at subsequent levels. The maximum
depth is sixteen; insufficient depth is rejected before attention, rather than
returning an oversized parent as raw evidence. A leaf-only index is rejected.

The route applies source scope and inclusive question-day eligibility before
root admission and child expansion. Mixed-date leaf summaries can still mention
later events, as in the existing as-of method; a separate exact-span projection
prevents future raw spans from reaching the reader. No raw bytes enter Qwen.
Receipts retain scalar attention results and section bindings, not transformer
token state. This is approximate beam routing and does not certify completeness.

`hierarchical_spine_memory.py` integrates the new path beside the unchanged
relative-reservation control. It reuses the original leaf vectors, computes a
live query embedding, traverses the restored hierarchy, projects eligible spans
and calls the existing flat renderer/hydrator with 3,072 context tokens and 128
raw spans. The caller owns the resident Qwen linker. The new method has not yet
been timed on the complete population, and multiple attention levels may exceed
the latency allowance; that remains an evaluation requirement.

## Complete population preparation

All ten existing complete memories pass topology reconstruction. They retain
27,062 original leaves across 4,805 sources and require 22,257 parent summaries.
The smallest memory remains 1,039,791 token proxies. This preparation did not
read raw transcript text, call a provider, regenerate attention or inspect
benchmark answers. It prepared the first summary-dependency wave for each memory.

The sealed population is
`eval_results/full1m-spine-parent-population-20260910-r1/prepared.json`, SHA
`4db0ba47658ba76e09ccb6e69baddcfd78875f6f84df6de9e88cfcf577a3b8c0`.
Preparation session 12400 completed with exit zero. Parent roots are
`eval_results/full1m-spine-parents-offsetNNN-20260910-r1` for offsets 000–090.

| Offset | Leaves | Parents to compile | Sources |
| --- | ---: | ---: | ---: |
| 000 | 2,744 | 2,245 | 499 |
| 010 | 2,625 | 2,161 | 464 |
| 020 | 2,693 | 2,214 | 479 |
| 030 | 2,751 | 2,278 | 473 |
| 040 | 2,737 | 2,256 | 481 |
| 050 | 2,712 | 2,232 | 480 |
| 060 | 2,612 | 2,155 | 457 |
| 070 | 2,686 | 2,198 | 488 |
| 080 | 2,708 | 2,220 | 488 |
| 090 | 2,794 | 2,298 | 496 |

These are validated parent plans, not generated parent summaries or a deployed
hierarchical serving index. The ten original serving indexes remain unchanged.

## Qwen backend failure

The first memory prepared 452 dependent summary jobs in 57 batches under
preflight `25b7d21f01e6f78946b36c3879763b82fd0bae8253c1fda120dc6425084604b3`.
Execution session 33671 exited one because the local gateway returned HTTP 500:

> Request for unknown model: 'qwen3-8b-gguf' is not found

The advertised `qwen3-8b` route at `https://central-dev.zt:4000/v1` maps to that
unavailable Triton model. A fresh models-list read still advertises the alias
and exposes no other Qwen route. The failed wave contains six request journals
and zero response journals. All original journals are retained; no failed call
was retried. Offset 000's failure is recorded in `gateway-failure.json`, SHA
`fc238bf0f65e01d6d59bfa5bca7199ef9504a30c09bf84bb41b69040cc117642`.
The user has been asked to restore the backend or provide a working Qwen route.
No alternative summarizer was silently substituted.

## Verification and remaining work

Thirty-two focused tests pass (3.39 seconds, tool chunk `bcf142`). They cover
exact reconstruction against the existing full hierarchy implementation,
rejection of changed topology and raw-bearing channel objects, an actual
completion-runtime parent dependency followed by offline replay, parent-to-child
attention traversal, date and scope admission, partial-workspace rejection,
exact raw hydration and the resident adapter with the original renderer.
Model clients in these tests are synthetic; they establish integration and
invariants, not real Qwen relevance or answer accuracy. `git diff --check` passes.

The Qwen backend must become available before complete parent generation can
finish. Preserve the failed offset-000 execution when preparing a recovery.
Then compile all ten parent hierarchies, measure real local attention traversal,
and run a fresh full100 joint accuracy/latency comparison against the unchanged
control and both API baselines. The new path still needs its full100 evaluation
adapter. The target remains at least 95/100 with API-like latency on the same
responses; no target gate has passed. The separate native corpus stays parked.
