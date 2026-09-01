# Soft topical boundaries for long-chat memory retrieval

**Date:** 2026-08-30
**Question:** Would better mapping of topical boundaries improve this system's
remaining recall and evidence-use failures?
**Scope:** Primary research on discourse/event segmentation, conversational
memory granularity, hierarchical and graph retrieval, local/global traversal,
evidence density, and downstream reading. Sources were reviewed through
2026-08-30. Vendor or project reports are identified separately from
peer-reviewed papers.

## Executive answer

Yes, but the useful intervention is more specific than "better chunking."

The literature supports a stack with two distinct boundary structures:

1. **chronological event boundaries**, which preserve local setup, role,
   temporal qualifiers, corrections, and neighboring turns; and
2. **a soft, overlapping topical cover**, which links related evidence across
   non-contiguous events and sessions.

Those structures should be routing metadata over immutable exact evidence, not
factual authority and not a hard exclusion gate. A query may activate several
topics; evidence from all specialist routes should be unioned before any
exclusion; only a separately sealed `definitely_irrelevant` decision should
prune a selected leaf. The graph or topic hierarchy should find and connect
evidence, while the final prompt should present exact-cited atomic facts plus
only the narrative context needed to interpret them.

This distinction matters immediately. The current R7 exact-11 diagnostic makes
all 26/26 preregistered answer atoms selected, admitted, visible, and usable
under the hard prompt cap, yet its first terminal answer/judge path scored only
2/11. Two correct raw answers were subsequently shown to have been rejected by
the old validator and are repaired by validator v4, but the remaining gap is
still primarily downstream of source discovery. Relevant evidence occupies
only a small minority of the roughly 7.1--7.2K-token prompts, and many questions
require evidence distributed across multiple components. Better topic maps can
raise density and expose relations, but **post-union evidence interpretation is
the first experiment**, not a corpus resegmentation campaign.

The immediate architectural answer is therefore:

> **soft topic/event routing + typed cross-boundary links + post-union semantic
> sieve + exact fact compilation + obligation-aware closure.**

A hard single-topic partition is the wrong default. It would place a new
false-negative boundary directly in front of the multi-event set, count,
comparison, temporal, and update questions that remain difficult.

## 1. Two different questions were being conflated

### 1.1 Ingest-time boundary formation

This asks where a locally coherent event begins and ends. A good event span
keeps references such as "it," relative time expressions, speaker role, and the
setup/outcome of an action together. Classical TextTiling detects lexical
cohesion valleys and produces contiguous multi-paragraph subtopics
([Hearst, 1997](https://aclanthology.org/J97-1003/)). Event Segmentation Theory
adds a cognitive account: people segment activity hierarchically using both
bottom-up change and top-down goals, and boundary quality affects later memory
([Zacks and Swallow, 2007](https://journals.sagepub.com/doi/10.1111/j.1467-8721.2007.00480.x)).

For this repository, surprise/coherence-derived EM episodes are the analogue.
They are local chronological containers, not the complete topic model.

### 1.2 Query-time topical routing and presentation

This asks which of the already stored evidence should be activated, connected,
compiled, and emphasized for the current question. A topic can recur after many
sessions; one turn can participate in multiple topics; and an aggregate question
can intentionally cross topical boundaries. This layer is better represented as
a weighted cover or graph than as a partition.

The distinction explains why perfect source visibility can coexist with a bad
answer. A retrieval system can place every target span in the prompt while the
reader still fails because the spans are sparse, disconnected, duplicated,
poorly labeled, or surrounded by higher-salience distractors.

## 2. What the literature actually establishes

| Evidence | Main result | What it licenses here | Important qualification |
| --- | --- | --- | --- |
| [TextTiling](https://aclanthology.org/J97-1003/) | Lexical cohesion can identify contiguous subtopic passages. | A cheap, deterministic hard-boundary control. | Expository documents are more linear than long, interleaved dialogue; it does not model recurring or overlapping topics. |
| [Event Segmentation Theory](https://journals.sagepub.com/doi/10.1111/j.1467-8721.2007.00480.x) | Human event models are hierarchical and influenced by prediction change and goals; segmentation affects later memory. | Maintain multiple temporal scales and keep event boundaries distinct from semantic topic identity. | Cognitive evidence motivates structure; it does not specify an IR optimum. |
| [SeCom, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e56f394bbd4f0ec81393d767caa5a31b-Abstract-Conference.html) | Topically coherent segment memories outperform turn-, session-, and summary-level alternatives; compression acts as denoising. | Segment-level routing plus a separate denoising/reading stage is better supported than boundaries alone. | Its segments are contiguous and non-overlapping, and its conversations are far smaller than this project's 1M-token ingest. |
| [EM-LLM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/c05144b635df16ac9bbf8246bbbd55ca-Paper-Conference.pdf) | Bayesian-surprise boundaries, graph refinement, semantic retrieval, and temporal contiguity are complementary; the system operates over up to 10M tokens. | Keep event seeds and a bounded neighbor budget; local coherence repairs a semantic hit that lacks its setup. | Original EM-LLM persists transformer KV memory, which this project forbids. Its topology lesson transfers; its storage substrate does not. |
| [RMM, ACL 2025](https://aclanthology.org/2025.acl-long.413/) | In a 100-case LongMemEval analysis, turn/session/mixed/topic-reflection retrieval scored 47/69/38/78 Recall@5 and 29/34/17/49 QA; a per-instance best granularity reached 86/58. | Adaptive topic organization can beat fixed granularity, while an uncalibrated union of representations can add damaging noise. | The comparison bundles representation and consolidation choices and is only a 100-case analysis. |
| [RAPTOR, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html) | Recursive clustering and summaries support retrieval at several abstraction levels; soft clustering permits membership in more than one cluster. | A precedent for a topic **cover/DAG** and global-to-local descent. | Document QA is not long-chat memory; generated summaries may omit or distort exact facts. |
| [Dense X Retrieval, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.845/) | Self-contained atomic propositions outperform passage units for retrieval and QA under fixed compute budgets. | Compile selected evidence into several exact-cited facts per source handle rather than one lossy summary per chunk. | Wikipedia QA does not test updates, dialogue roles, or 1M-token histories. |
| [LongMemEval, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf) | Long-chat memory separates extraction, multi-session, temporal, update, and abstention abilities; value granularity, fact-augmented keys, and time-aware queries affect recall and QA separately. | Measure retrieval, packing, and reading independently, and route different obligations to different memory views. | Its standard histories, readers, and budgets are not identical to this locked 1M-token protocol. |
| [Associa, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.901/) | An event graph plus connected-subgraph retrieval and iterative missing-clue recall reaches LongMemEval-S R@5/R@10 of .87/.93; removing association drops them to .67/.85, and removing deliberation to .84/.90. | Typed local-to-global links and a bounded closure iteration are separately useful. | Its model, graph construction, and evaluation protocol differ; it does not enforce this project's provenance and prompt invariants. |
| [Zep/Graphiti, 2025 preprint](https://arxiv.org/abs/2501.13956) | Graphiti maintains raw episode nodes, extracted entity/fact edges, and higher-level communities; facts carry valid-time and ingestion-time intervals and retain links back to source episodes. Search combines semantic, lexical, and graph traversal before reranking and text construction. | The closest existing reference architecture for the proposed temporal graph plane: non-lossy episodes below semantic facts/entities below global communities. | An ingested Graphiti episode is normally a message/text/JSON unit, not a learned topical boundary; communities cluster entities, not necessarily overlapping dialogue spans. Its extraction, invalidation, and summaries use LLM judgment and its paper is not peer reviewed. |
| [MemORAI, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1408/) | Removing topic segmentation drops LongMemEval-S turn R@10 from 91.63 to 23.86. Query-focused subgraphs modestly improve over full-graph traversal, and adding relational triplets to turns raises the reported judge score from 61.72 to 75.55. | Strong evidence for topic structure, relevance-bounded traversal, provenance links, and giving the reader relational context rather than isolated nodes. | LLM segmentation/filtering can false-negative; the bundled pipeline and different reader/judge prevent direct score comparison. Dynamic edge weights alone add only 1.88 R@10 points. |
| [Bounded Conversational Memory / CPP, SIGDIAL 2026](https://aclanthology.org/2026.sigdial-1.55/) | On LoCoMo, session retrieval metrics are identical to flat retrieval while QA improves, especially multi-hop. On LongMemEval, CPP reports lower Recall@15 (.881 vs .919) but higher QA (.606 vs .360). Evidence highlighting is the largest directional component. | This is the closest published analogue to R7: organizing and highlighting already retrieved evidence can dominate another retrieval-depth increase. | The LongMemEval component study is cumulative development evidence, not a held-out single-factor ablation, and CPP supplies about 3.4x more context words than its equal-k flat control. |
| [Does Memory Need Graphs?, ACL 2026](https://aclanthology.org/2026.acl-long.1232/) | In a controlled framework, a graph can improve retrieval and session-valued QA, yet graph atomic-key QA can lose to flat cohesive notes. Case studies attribute failures to fragmented metadata and missing narrative anchors, while graphs help long-range cross-session dependencies. | Use the graph as a reachability/index plane, not as the final prompt representation. Preserve a cohesive value with exact facts, relations, and minimal narrative context. | Graph quality is conditional on key/value design, reranking, and reader capacity; "more graph recall" is not an answer-quality guarantee. |
| [QueryLink, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.765/) | Query-memory alignment plus coherent multi-turn chunking and multi-grained retrieval reports at least a 7% reasoning-accuracy improvement. | Query-time alignment should operate across multiple granularities and keep coherent dialogue units available. | LLM-judged headline gains are not comparable to this locked judge or hard budget. |
| [APEX-MEM, ACL 2026](https://aclanthology.org/2026.acl-long.749/) | A temporally grounded event property graph preserves append-only history and resolves conflicts/evolution at query time into compact relevant memory. | Store event-time and update relations, but resolve the currently relevant state late rather than overwriting history. | Its agentic tool loop and reported scores use a different budget and evaluation setup. |

### 2.1 The strongest convergence

Across these otherwise different systems, the same pattern repeats:

- turns are often too small and sessions too noisy;
- one fixed granularity is not uniformly optimal;
- local event context and non-local semantic relations solve different problems;
- graph or hierarchy traversal helps find dispersed evidence;
- atomic facts improve density, but isolated atoms can lose entity and event
  identity;
- summaries help routing and global sensemaking, but are lossy factual values;
- what the reader sees can matter as much as what the retriever found; and
- more candidates without reranking, deduplication, or closure can make answers
  worse.

That is almost a direct description of the repository's progression: R7 repaired
reachability, then exposed evidence-use and operator-connectivity failures.

## 3. Local-to-global and global-to-local are both required

### 3.1 Global to local

RAPTOR and GraphRAG match a query to higher-level descriptions before descending
to exact material. The original
[GraphRAG preprint](https://arxiv.org/abs/2404.16130) builds hierarchical
community summaries and reports gains on global sensemaking over corpora around
1M tokens. Microsoft's later
[DRIFT engineering report](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/)
combines a global community primer with local follow-up exploration. These are
good precedents for using topic/community descriptions to select where to inspect
more deeply.

For this project, global descriptors must remain non-authoritative. A selected
topic or community hydrates exact leaves; its summary cannot itself prove a
count, date, update, or quoted claim.

### 3.2 Local to global

Associa starts with query-relevant nodes, extracts a connected subgraph, then
asks whether relevant evidence is still missing and issues another bounded
retrieval. This is the appropriate shape for a strong local hit whose answer
depends on another session or component.

The allowed cross-boundary edge vocabulary should be typed and provenance-bound:

- `same_entity` or explicit alias;
- `same_event`;
- `temporal_adjacent` within a source;
- `precedes`, `follows`, or normalized event-time relation;
- `supersedes`, `conflicts_with`, or `resolves`;
- shared action, object, status, or obligation; and
- multi-label topic membership and parent-topic membership.

Unrestricted similarity edges or free PPR expansion invite hub drift. Expansion
should be shallow, query-conditioned, separately budgeted, and stopped when
typed obligations are covered or the hard budget is exhausted.

## 4. Proposed representation: a cover, not a partition

The repository's own theory already anticipated this design. In
[Extracted Attention Heads as Recursive Associative Memory](../00%20-%20Theory/01%20-%20Extracted%20Attention%20Heads%20as%20Recursive%20Associative%20Memory.md),
multiple concepts may claim the same tokens, so conceptual chunks form a cover
over the source rather than a single partition. The
[EM-LLM episodic discourse closure theory](../00%20-%20Theory/05%20-%20EM-LLM%20Episodic%20Discourse%20Closure%20for%20Diffuse%20Retrieval.md)
already separates boundaries, temporal contiguity, discourse relations, and
atomic evidence bundles.

The durable representation should be:

```text
immutable exact turn/span/atom
  ├─ chronological event membership (one local timeline, multiple scales)
  ├─ weighted topic memberships (zero-to-many; hierarchical)
  ├─ typed entity/event/time/update edges
  ├─ lexical and semantic routing projections
  └─ exact provenance: source, role, date, coordinates, hashes
```

Recommended boundary rules:

1. An event boundary may split chronology, but never deletes source adjacency.
2. A topic membership may cross events and sessions.
3. A span may hold two or more topic memberships when confidence is ambiguous or
   the span genuinely bridges themes.
4. A recurring topic links events without merging away their timestamps or
   state versions.
5. Low boundary confidence produces overlap or `uncertain`; it does not prune.
6. Topic labels and summaries are routing descriptors, never provenance or
   `definitely_irrelevant` authority.
7. Exact source spans remain available after every summary, fact, or graph hop.

No reviewed long-chat paper cleanly compares a mutually exclusive topic
partition with a soft multi-label topic cover under a matched reader, retrieval
budget, and prompt cap. RAPTOR supplies a document-QA precedent for soft
membership, while most conversational systems use contiguous hard segments.
That missing comparison is a credible original contribution for this project.

### 4.1 Is this Graphiti-like?

Yes. Graphiti is probably the closest named architecture, but it supplies the
**temporal association backbone**, not the whole boundary solution.

The mapping is direct:

| Graphiti | This project's proposed equivalent |
| --- | --- |
| raw episode subgraph | immutable turns, source spans, and chronological EM episodes |
| semantic entity/fact subgraph | exact-cited typed facts plus entity/event/state relations |
| community subgraph | high-level topical hierarchy used for global-to-local routing |
| valid/invalid plus created/expired time | append-only event-time and ingestion-time validity |
| cosine + BM25 + graph traversal | protected semantic + lexical + episodic/graph specialist lanes |
| search -> rerank -> text constructor | union -> R/I/U sieve -> fact compiler -> terminal packet |

The additions are load-bearing. Graphiti begins with externally supplied
episodes—often individual messages—whereas this project needs learned
surprise/coherence event spans. Its communities group connected entities; they
are useful global descriptions but are not a guarantee that a dialogue span has
the right topic boundary or that a span belongs to every relevant topic. The
proposed soft topic cover therefore sits alongside the Graphiti-like graph.

The other deliberate differences are union-before-exclusion, fail-open
`unresolved` leaves, exact quote/hash provenance, deterministic operand closure,
the hard 8K prompt envelope, zero persisted request-derived transformer token
state, and sealed replay. In short: **Graphiti-like underneath, proof-carrying
and soft-boundary above it.**

### 4.2 Existing implementation fit

This does not require a second graph store. The repository's existing discourse
plane already has the necessary lossless substrate:

- `DiscourseUnit` is an evidenced typed node whose factual authority remains in
  exact source spans;
- `DiscourseRelation` is an evidenced n-ary edge, and each `RelationMember`
  already carries a non-negative weight;
- `iter_unit_routes_for_artifact()` exposes text-free routing projections before
  evidence hydration;
- `incident_relations()` supplies bounded local-to-global expansion; and
- `adjacent_episodes()` supplies chronological local closure.

A topic can therefore be a routed discourse unit, with weighted `topic_member`
relations from exact event/turn units and optional `broader_topic` relations
between levels. Multiple relations give a span soft multi-topic membership.
The weights may rank or allocate budget, but must not authorize pre-union
exclusion. This reuses the current immutable evidence, graph receipts, and
storage/query boundaries; the new work is a versioned linker/router plus the
matched hard-versus-soft ablation, not a Graphiti-shaped rewrite.

### 4.3 First implementation evidence

The exact11 A1a assay now supplies a narrow empirical example of that mapping.
A generic semantic classifier pruned a smoker episode because its provider row
omitted the leaf's authenticated date. The dated query asked for an appliance
bought ten days earlier; the text alone was ambiguous, while the temporal edge
was decisive. A separate question-derived temporal fail-open voter restored
the leaf without a graph-database rewrite or a handle-specific rule. The
post-seal retention audit moved from 25/26 to 26/26 semantic atoms while the
runtime still pruned 258/381 selected leaves and stayed under the 8K cap.

The promoted implementation records this rescue as a separate effective
transition layer over immutable A1 and base-classifier construction/replay
pairs. Downstream consumers re-run semantic selection and the temporal policy
before accepting the overlay. This proof-carrying requirement is stricter than
the Graphiti analogy alone: a temporal edge is useful only if the exact source,
base decision, date relation, and I-to-U transition remain jointly verifiable.

This supports the Graphiti analogy at the mechanism level: episode content and
valid time must remain separately addressable, and temporal linkage should be
able to preserve a locally ambiguous episode for downstream fact compilation.
It is not yet evidence of higher end-to-end QA accuracy; Research Logs 91 and
92 keep the rejected and promoted selection arms separate.

## 5. Query-time pipeline implied by the review

```text
query
  -> typed question/obligation analysis
  -> protected specialist lanes
       lexical | semantic | episodic | temporal | graph/topic
  -> UNION FIRST
  -> exact-source deduplication
  -> per-leaf {relevant | definitely irrelevant | unresolved}
  -> bounded bridge/closure retrieval for missing operands
  -> exact-cited atomic fact compilation
  -> deterministic set/count/time/update operator when applicable
  -> compact fact table + minimal narrative anchors + exact excerpts
  -> final LLM
```

This preserves the user's approved ordering: evidence selected by each memory
method is not excluded before the union. Specialist budgets should have protected
minima so a routing error cannot zero out lexical, episodic, temporal, or graph
recall. Any surplus can slew toward the topics and operators activated by the
query, with a separate reserve for cross-topic closure.

The final prompt should not serialize a raw graph neighborhood. It should expose:

- the small set of typed facts that directly bear on the question;
- explicit relations needed to join them;
- dates, roles, state, and uncertainty;
- exact citation handles and short source quotes; and
- a minimal coherent excerpt when atomic facts alone lose the narrative anchor.

This is consistent with Dense X's proposition result, CPP's evidence-highlighting
result, and the graph/flat case studies in *Does Memory Need Graphs?*.

The new provider-free `after_union_fact_closure.py` substrate already encodes
the right safety boundary: optional multi-label topic/boundary metadata and
typed cross-boundary edges can influence grouping and budgets but not pruning;
each selected leaf must end as compiled facts, sealed definitely irrelevant, or
unresolved. The remaining work is to adapt R7 selected leaves into it and test
the treatment without gold-bearing runtime inputs.

## 6. Retrieval style by question type

| Question need | Primary view | Required repair beyond primary retrieval |
| --- | --- | --- |
| One exact personal fact | proposition/turn + local event | enough local context to bind entity, role, and qualifier |
| Multi-event set or count | soft multi-topic activation with protected per-component beams | closure over every operand, deduplication, deterministic set/count operator |
| Temporal comparison | event spans + normalized time index | chronology/update edges and deterministic interval/date operation |
| Knowledge update/current state | entity-state graph over append-only facts | `supersedes/conflicts/resolves` traversal and late state resolution |
| Causal or multi-hop relation | strong local seeds + typed bridge graph | one bounded bridge iteration and sufficiency check |
| Corpus/global theme | hierarchy/community descriptors | global-to-local descent and cited map/reduce synthesis |
| No applicable specialist | broad fail-open union | semantic binary search may exclude only separately certified negative cells |

The union of these routes is the desired memory surface. A question classifier
should change budgets and stopping rules, not assign one exclusive memory owner.

## 7. Controlled ablation plan

The interventions must be tested separately before composition so a density gain
is not misattributed to boundary formation.

| ID | Treatment over the sealed R7 parent | Question answered |
| --- | --- | --- |
| A0 | Current R7 union + validator v4 | Protected reference |
| A1 | Post-union ternary sieve + exact-cited fact compiler only | Is evidence density/interpretation already sufficient to rescue exact-11? |
| A2 | Hard contiguous topical partition metadata | Does ordinary semantic segmentation help, and which cross-boundary targets does it lose? |
| A3 | Soft multi-label topical cover, exact leaves still delivered | Does overlap improve density without losing multi-component closure? |
| A4 | A3 + hierarchical route descriptions, hidden from factual authority | Does global-to-local descent improve selection? |
| A5 | Event boundaries with 0%, 15%, and 30% protected adjacency budgets | How much local context recovers qualifiers before noise dominates? |
| A6 | Typed one-hop entity/event/time/update expansion | Does local-to-global bridging solve the disconnected cases? |
| A7 | Query-adaptive per-lane/topic budgets with protected minima | Does adaptive slew help specialized question types without routing regressions? |
| A8 | A3--A7 composed, then A1 after the union | End-to-end candidate for promotion |

Required factorial controls:

- one hard membership versus at most two and at most three soft memberships;
- event-only, topic-only, and event-plus-topic;
- route summaries hidden from versus shown to the final reader;
- zero versus one bridge hop;
- equal lane budgets versus adaptive budgets with identical total tokens;
- no exclusion versus `definitely_irrelevant` exclusion versus ordinary top-k;
- raw excerpts versus facts plus exact supporting excerpts; and
- one-shot retrieval versus one closure/bridge iteration.

Order of execution:

1. Run A1 on exact-11 because its 26/26 atom visibility isolates the downstream
   question. The provider-free substrate exists; the R7 adapter and terminal
   answer treatment remain to be completed.
2. Build A2 and A3 as metadata-only, zero-provider constructions over the same
   sealed evidence. Neither may change membership in the provider packet yet.
3. Compare hard and soft boundaries on visibility, connectivity, duplicate cost,
   and density before asking for answers.
4. Add A4--A7 one at a time, preserving the protected A0/A1 predictions.
5. Compose only independently positive components into A8.
6. Promote in order: exact-11 diagnostic, locked full100, then confirmation200
   with the previously unexposed 185 rows as the primary confirmation report.

## 8. Metrics and promotion gates

Boundary scores such as Pk or WindowDiff are useful diagnostics, but the target
is proof-carrying QA under budget. Each ablation needs three separate planes.

### Retrieval and topology

- any-target and all-target source visibility;
- semantic-atom visibility and full typed-obligation closure;
- number of evidence components before and after typed linking;
- cross-boundary target retention;
- false-negative routing and false-prune count; and
- event-neighbor gain versus distractor cost.

### Packing and representation

- answer-bearing token density, measured postseal with gold only for diagnosis;
- selected-leaf resolution rate;
- operator-obligation coverage;
- exact-source duplication and tokens recovered by deduplication;
- provenance completeness and exact-quote verification;
- total prompt tokens, including the 768-token answer reserve, never over 8K;
- provider calls, retained transformer-state bytes, p50/p95 latency, and replay
  identity.

### Answering

- locked semantic accuracy overall and by question type;
- exact/set/count/date/operator correctness;
- abstention and incorrect-abstention rates;
- regression count on protected correct questions; and
- exact-11, full100, and confirmation scores reported separately.

A topical treatment should not be promoted unless it improves density or answer
accuracy without reducing 26/26 exact-11 atom visibility, exact provenance, or
typed closure. Runtime policies must remain gold-blind; gold may be opened only
after sealing for diagnostic attribution.

## 9. What the literature does not prove

The evidence is directionally strong but not a license to import headline
scores:

- readers, judges, question populations, context budgets, and provider models
  differ substantially;
- most segmentation work uses document or ordinary multi-session scales, not
  ten independent 1M-token memory namespaces;
- LLM-generated boundaries, summaries, filters, and graph triples can omit facts,
  merge entities, hallucinate relations, or encode benchmark-specific priors;
- community summaries preserve global themes but can erase rare exact details;
- graph expansion can improve recall while injecting sparse metadata that a
  reader cannot use;
- several 2026 results rely on LLM judges and development-style ablations;
- CPP's strongest presentation result lacks a token-matched flat control;
- none of the reviewed systems combines a hard 8K total prompt cap, exact cited
  provenance, sealed replay, and zero persisted request-derived transformer
  token state in the same way this project does.

The 2026 controlled graph study is the most important warning: retrieval recall
and answer accuracy are coupled through the *value* presented to the reader.
Topic boundaries and graph links can improve reachability, but they become a
complete memory system only when they hydrate exact, semantically cohesive,
low-noise evidence.

## Conclusion

Better topical mapping should help, particularly on recurring topics and the
remaining local/global connectivity cases. The literature does not support
replacing the current memory stack with a topic partition. It supports adding a
soft, overlapping topic hierarchy alongside chronological episodes, then using
both to budget and connect evidence after all specialist routes have had a chance
to contribute.

The highest-value next test remains A1: prove whether post-union semantic
sieving and exact fact compilation can convert the already complete exact-11
evidence into correct answers. Then compare hard and soft topic structures as
independent metadata treatments. If soft boundaries reduce distractor load
without losing a single target atom, they become the routing layer for the
composed path to 95%; if not, the failure remains in fact compilation, closure,
or terminal reading rather than corpus organization.
