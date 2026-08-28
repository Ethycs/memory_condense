# Local-to-global and global-to-local memory connectivity assay

**Date:** 2026-08-27

**Status:** architecture analysis and implementation direction; no provider
calls; no new accuracy result

## Conclusion

The best-supported design for this repository is a **dual-plane, bidirectional
active reconstruction loop**:

```text
local window
  -> provenance-bound events/facts + cues/tags/relations
  -> global associative graph and exact-content store
  -> query-conditioned bounded traversal
  -> compact cited fact subgraph + hydrated exact chunks
  -> local answer window and typed operator
```

A latent bottleneck such as CAV extraction/reinjection is a useful soft routing
plane inside that loop. It is not a sufficient factual memory by itself. Exact
content, event identity, roles, dates, and source provenance must remain on a
separate discrete plane.

The retrieval representation does not have to be the final LLM context. Facts,
tags, and links can stay compact because they select and validate memory; after
selection, the system may hydrate a bounded exact source chunk and use that
chunk as the answer model's substitute context. This avoids making a lossy
summary carry all of the factual burden.

The missing operation in the current final arm is not merely a wider window.
It is the **reverse/forward reconstruction step**:

```text
retrieved content -> new typed cues/relations -> another bounded memory read
```

This is materially different from fixed top-k retrieval, fixed graph expansion,
or reordering the same evidence after CAV reinjection.

## Two directions that must be explicit

### Local window to global memory

The write direction should:

1. retain the raw turn or exact source span as factual authority;
2. segment the local window into event-sized units;
3. emit atomic facts with entity, action/role, status, event time, discourse
   relation, and exact source-span provenance;
4. derive compact cues/tags and typed links from those facts;
5. update only bounded scalar association/heat/decay state; and
6. discard every request-derived token, activation, attention, and routing
   tensor after its receipts have been reduced to bounded IDs, links, and
   scalars.

### Global memory to local window

The read direction should:

1. compile the dated question into entity, role, temporal, and answer-operation
   obligations;
2. retrieve protected direct anchors;
3. activate graph cues and tags from the question and current evidence;
4. retrieve exact content through the selected relations;
5. turn that content into new cues when obligations remain unresolved;
6. repeat for a small fixed number of hops;
7. hydrate and validate exact cited facts and their enclosing source chunks;
8. pack a compact connected subgraph plus the most useful exact chunks under
   independent lane budgets; and
9. run the typed answer operator and answer LLM over that bounded context.

The answer model may see selected exact chunks in addition to facts and
relations. It does not see the complete global memory or persisted hidden
state.

## Technique assay

| Technique | Local -> global operation | Global -> local operation | Strength | Failure mode / boundary | Repository status |
| --- | --- | --- | --- | --- | --- |
| Sparse local/global attention tokens | Global tokens pool local-window tokens | Local tokens attend designated global tokens | Cheap bidirectional communication inside one encoded sequence | No durable random-access factual store; weak exact provenance | Architectural analogue only |
| Perceiver-style latent bottleneck | Latent queries cross-attend to local nodes | Output/local queries cross-attend to latents | Linear `O(NK)` global fusion | Fixed latent capacity can collide facts; latent values are not citations | Same mathematical shape as the CAV router |
| CAV extraction/reinjection | `E: K x N` extracts concepts from selected evidence | `R: N x K` reinjects filled concepts into evidence nodes | Query-conditioned soft linking without an `N x N` matrix | Our answer path has consumed only ordering or a rank-only text guide; enriched `X1` has no production typed-representation consumer | Genuine link receipts and zero-state router exist; final-arm integration incomplete |
| Recurrent/compressive neural memory | A segment updates a neural/associative state | New local tokens query that state | Native streaming and very long sequence modeling | Hard to provide exact source provenance; persisting it would cross this project's transformer-state boundary | Deliberately not the durable authority |
| Classical RAG/source hydration | Local source text is chunked/indexed with locators | Query retrieves and hydrates exact spans | Strong direct recall, cheap replay, exact provenance | One-shot top-k cannot create new cues from intermediate evidence | Strongest current source-discovery base |
| Hierarchical summaries/representatives | Local episodes update coarser representatives | Query descends from representative to detail | Efficient global localization and temporal bridging | Generated summaries can erase role/status/detail; representative retrieval alone cannot execute the answer operation | Episode representatives exist; isolated locked arm did not earn promotion |
| Typed discourse/evidence graph | Events create provenance-bound typed relations | Query follows revision, dependency, temporal, entity, or obligation edges | Exact, inspectable relation closure | Link recall depends on the linker; fixed expansion adds noise | Closure engine exists; final answer path consumes only a limited projection |
| Query-weighted graph diffusion | Local facts seed graph heat; prior access updates bounded weights | Query-conditioned weights propagate to related facts | Soft multi-hop recall and source allocation | Hubs/noise dominate without typed constraints and protected anchors | Heat/Hebbian code exists; current matched tick does not jointly exercise it |
| Active associative reconstruction | Retrieved content exposes new cues/tags | New cues select the next relation/content read | Directly solves evidence-dependent retrieval and local/global connectivity | More latency; needs strict hop, token, and provenance gates | Missing orchestration seam; recommended next layer |
| Typed fact/operator closure | Local content becomes atomic operands/events/members | Complete typed frontier is injected into a deterministic or LLM-assisted operator | Separates retrieval from count/sum/set/timeline logic | Cannot manufacture missing facts or relations | Current final arm is implementing this plane |

## Current external evidence

The neural primitive is well established. Perceiver and Perceiver IO use
asymmetric cross-attention to distill large inputs into a small latent array and
query that latent representation for outputs, giving linear scaling in input
and output size:

- [Perceiver](https://arxiv.org/abs/2103.03206)
- [Perceiver IO](https://arxiv.org/abs/2107.14795)

Recent model-native memories such as Infini-attention, Titans, MIRAS, and ATLAS
combine precise local attention with a learned associative or compressive
long-term state. They support very long streaming contexts but are not a
substitute for exact, source-addressable agent memory:

- [Infini-attention](https://arxiv.org/abs/2404.07143)
- [Titans](https://arxiv.org/abs/2501.00663)
- [MIRAS](https://arxiv.org/abs/2504.13173)
- [ATLAS](https://arxiv.org/abs/2505.23735)

For persistent agent memory, current work is moving from passive retrieval to
multi-granular, provenance-aware graphs and active retrieval. MemORAI uses a
turn-provenance graph, query-focused subgraph construction, and
query-conditioned edge weights. MRAgent makes the bidirectional operation
explicit as `Cue -> Tag -> Content` and `Content -> Cue/Tag`, conditioning each
next read on accumulated evidence:

- [MemORAI](https://arxiv.org/html/2605.01386)
- [MRAgent / active memory reconstruction](https://arxiv.org/html/2606.06036)
- [A-MEM](https://arxiv.org/abs/2502.12110)

MRAgent reports 72.95% LongMemEval judge accuracy with its Gemini retrieval
configuration and 86.76% with Claude used for retrieval, versus roughly
53--55% for the passive baselines in that paper. These numbers are not directly
comparable to this repository's locked population, model routes, prompt cap, or
judge. The relevant result is the ablation direction: associative tags help,
and evidence-conditioned multi-step reconstruction contributes more than a
wider one-shot retrieval budget.

LongMemEval-V2 independently frames memory as context gathering from histories
up to 115M tokens. Its strongest reported method is an active coding-agent
evidence gatherer rather than a fixed RAG query, although it remains expensive:

- [LongMemEval-V2](https://arxiv.org/abs/2605.12493)

## What our prior experiments actually established

| Local result | What it proves | What it does not prove |
| --- | --- | --- |
| CAV-steered order: 6/10 versus original order 5/10 on dev10 | Reinjected `X1` scores can change useful text order | Direct activation injection, graph linking, or held-out accuracy |
| Genuine CAV link guide: linked and unlinked both 10/10 under Sol on dev10 | Rectangular extraction/reinjection receipts and a bounded guide can execute without semantic loss on that slice | A positive CAV accuracy marginal |
| Locked S0+CAV: 53/100 versus S0 57/100 | A fixed three-concept rank-only guide over raw S0 is not safe to promote | That all CAV-based linking is useless |
| Locked S0+EM facts: 60/100 versus S0 57/100 | Post-selection atomic fact representation can help dispersed/numeric questions | A complete local/global reconstruction loop |
| Current adaptive source map: 72/100 | Better source selection and mapped facts are the strongest measured base | The >=95% gate or a composed memory system |

The negative locked CAV arm used three fixed layer-0 concepts over raw S0
membership and exposed a rank-only textual guide. It did not let reinjected
facts generate another retrieval cue, did not operate over the final typed fact
frontier, and did not make `X1` change the operator's input graph. It therefore
tested a lossy consumer, not the complete architecture now indicated by the
local/global failures.

## The six retrieved EM misses through this lens

| Q | First connectivity failure | Needed bridge |
| ---: | --- | --- |
| 14 | Local cuisine events attach to an ambiguous global/source frontier | source/namespace-aware cue-to-event links before distinct count |
| 28 | Correct answer and evidence are disconnected by validation | prediction-to-exact-span proof edge with per-citation salvage |
| 67 | Correct visits share the operative frontier with another namespace | query-conditioned source/event boundary before deduplicated count |
| 69 | Return and replacement-pickup roles collapse under entity deduplication | action-role/status edges that survive selection and dedup |
| 75 | Both prices survive but their approximation qualifiers do not reach comparison | value-to-qualifier edge and bounded comparison semantics |
| 97 | `again` does not activate the previous order | content-to-discourse-cue reverse edge followed by a temporal read |

Q28 and Q75 are proof/operator-boundary cases more than discovery cases. The
other four are direct examples of local content failing to reconstruct the
right global relation.

## Recommended bounded implementation

The first active-reconstruction treatment should be intentionally small:

```text
question-only obligations
  -> protected direct/map facts
  -> provider-free full-store/episode seed read
  -> atomic fact conversion
  -> derive new cues only from admitted, provenance-bound fields
  -> one additional provider-free global read
  -> exact-span and bounded enclosing-chunk hydration and validation
  -> post-selection role-sensitive dedup
  -> CAV/heat optional soft edge weighting
  -> compact linked fact frontier + exact chunk payload
  -> typed operator and one final answer call
```

Required controls and invariants:

- maximum two retrieval rounds in the first arm;
- fixed independent content budget per round and per memory mechanism;
- the second round may use only fields from admitted facts: entity, relation,
  role, status, event time, specificity term, and unresolved slot;
- no question ID, reference answer, judge result, or target-owner registry may
  influence routing;
- every second-round item must bind to an exact raw source span;
- fact-level routing and chunk-level delivery must remain separate receipts, so
  a chunk can be sent to the LLM without pretending that every sentence is an
  extracted atomic fact;
- exact selected chunks may consume a dedicated final-context lane and should
  be preferred over a lossy model summary when they fit the hard budget;
- deduplication occurs after each method selects its candidates and must retain
  distinct action/status/time roles;
- direct-parent evidence and prediction remain protected fallbacks;
- CAV/heat may rank or link candidates but cannot create factual content;
- all latent tensors are transient; durable state is limited to content,
  provenance, typed IDs/edges, scalar scores, and sealed receipts;
- the complete final prompt plus output reserve remains at or below 8,000
  token proxies.

The first matched ablation should compare:

1. current passive typed final composition;
2. active deterministic reverse/forward reconstruction;
3. the same active reconstruction with CAV or query-weighted graph scores; and
4. a wider passive budget control with the same final prompt allowance.

This isolates whether the gain comes from depth and changing cues, latent edge
weighting, or simply more candidates.

## Current code boundary

- `tools/matched_eval/full_store_slot_closure.py` builds one reusable exact
  content-window index and performs a question-only passive scan.
- `tools/matched_eval/typed_memory_final_arm.py` builds exact local story
  overlays and content-coherence links over the admitted frontier, but does not
  issue a new read from those links.
- `src/memory_condense/search/fusion/fixed_cav_router.py` implements the real
  two-pass latent update.
- `src/memory_condense/eval/fast_cav_links.py` seals bounded extraction and
  reinjection links with zero retained token state.
- `src/memory_condense/search/closure/engine.py` and the discourse graph expose
  typed relation traversal.
- `tools/matched_eval/prompt_tick_contracts.py` already defines the ordered
  `discover -> admit -> represent -> link -> answer -> observe` lifecycle.

The minimal missing module is a gold-blind reconstruction controller between
`represent` and final `link`: it derives bounded typed cues from admitted facts,
performs one more indexed read, and records exactly which local fact generated
which global cue and which exact new source span returned to the local frontier.

## Claim boundary

This assay combines current primary literature with repository code, git
history, and sealed local experiments. It does not show that active
reconstruction, CAV reinjection, or their combination already raises the locked
score. It identifies the next causally testable architecture and the controls
needed to distinguish it from a wider passive window. No provider call was made
and no evaluation artifact was changed.
