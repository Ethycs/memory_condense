# Gold-blind ordered-story residual7 repair

Date: 2026-09-06

Status: sealed, byte-replayed provider-free residual7 evidence result. The
ordered-story replacement is admitted on the reduced assay, but neither a
full100 transfer nor answer-model judging has been run. This is an evidence
retrieval result, not a 98% answer-accuracy claim.

## Outcome

The repaired typed-witness successor now recognizes an ordered list's requested
cardinality and audits whether its global candidate set is too ambiguous. Only
when that narrow gate opens does it invoke a source-story selector. The selector
must return exactly the requested number of distinct, chronologically ordered,
first-person completed-event witnesses; otherwise the packet remains identical
to its parent.

On ordinal 86, construction selected the exact three-trip sequence without gold:

1. Muir Woods;
2. Big Sur/Monterey; and
3. Yosemite.

Relative to the sealed repaired typed-witness v2 residual7 parent, the other six
rows were unchanged. Ordinal 86 changed as follows:

| Evidence diagnostic | Repaired v2 parent | Ordered-story replacement | Change |
|---|---:|---:|---:|
| Gold source-ID recall | 2/3 | **3/3** | +1 source |
| Answer components present | 0/3 | **1/3** | +1 component |
| Best evidence F1 | 0.267 | **0.400** | +0.133 |
| Packed evidence rows | 69 | **66** | -3 noisy rows |

Construction did not load a reference answer or gold source IDs. Gold was
opened only by the separate scorer after the construction bytes were sealed,
and replay reproduced the sealed construction. Construction, replay, and
scoring made zero model or provider calls and retained no transformer token
state.

The combined focused regression suite passed **98 tests**. It covers the
ordered-story replacement, the base provider-free assay, typed-witness grammar
and cardinality, typed execution and adapters, temporal insufficiency handling,
activated-turn links, and the incremental graph core/lane.

## What changed

The repair has three deliberately narrow parts:

- ordered-list cardinality distinguishes the requested three trips from the
  three-month time range;
- completed-travel grammar recognizes source-grounded forms such as returning
  from, taking, starting, completing, or finishing a trip while retaining the
  completed-vs-planned distinction; and
- a fail-open ambiguity gate replaces the typed packet only when the story
  proof is exact, source-distinct, chronological, and the requested cardinality
  is complete.

This is not a general instruction to scan every remaining chunk or send the
complement to a model. It is a typed, auditable response to an ordered temporal
question whose global candidate population is larger than its requested answer
set. The reduction from 69 to 66 packed rows is important: the gain came from a
smaller, more coherent proof rather than from adding more context.

## Relationship to the conversational graph

The result reinforces the HippoRAG/Graphiti-shaped direction described in
[Analysis 32](../08%20-%20Analysis/32%20-%20Incremental%20conversational%20association%20graph%20overlay%202026-09-06.md),
but it is not itself a HippoRAG or Graphiti result. The successful selector used
source/story relationships that can be compiled as turns arrive. The intended
production lifecycle remains local and provider-free by default:

```text
T0  atomically capture the raw turn and its pending graph obligation
T1  materialize chunks and commit ordinary BM25/dense searchability
T1g append bounded phrase, typed, sequence, and source-story graph deltas
T2  optionally enrich aliases/OpenIE facts asynchronously with exact citations
```

The current residual assay reconstructs its story inventory from the full
store. That proves the selection policy but is not the desired service cost.
The next structural step is to move the same source-story affinities and event
addresses into persistent T1g deltas, so query time reads a bounded resident
posting/graph neighborhood instead of rescanning the corpus.

## Scope of the result

Source coverage answers a structural question: did the packet include a chunk
from every reference-bearing source? Best F1 and component recall say somewhat
more about the selected text, but neither measures whether a final LLM produced
a correct answer. A Terra response plus a sealed Sol judgment would be required
for an answer-accuracy number. No such calls were made here.

Accordingly, this result supports three claims only:

1. the q86 missing source was reachable without embeddings or a provider;
2. strict story-scoped replacement improved the evidence packet without
   changing the other six residual rows; and
3. ordered story/session connectivity is worth compiling incrementally at
   ingest.

It does not establish a new full100 score, 95% fast-path accuracy, or production
graph latency.

## Latency

The seven-row command took 192.000 seconds end to end because it reconstructed
seven approximately 1M-token namespace indexes for the assay. Ordinal 86 took
2.241 seconds after setup, of which 2.146 seconds was the current full-store
typed/story selection and 72.038 ms was prompt-ready composition. These are
proof-of-policy costs, not the T1g target. Persisting the story inventory during
ingest is required before this path can be described as real-time retrieval.

## Artifact receipts

Canonical root:
`eval_results/longmemeval-1m-hot-v3-ordered-story-residual7-20260906`

| Artifact | SHA-256 |
|---|---|
| `construction.json` | `e060c9b7f3c188541e3a921388870f2247057ee4e1f800186d001eddbf4ed974` |
| `runtime.json` | `f2c98715ec43c614cf44334c71cfe174f67972c2339b0ee3f03d248b6b0c6c7c` |
| `replay.json` | `1ab161b327f01166acb68c874572bbeab9d0477668f7d1bee9ec3a017b854ff0` |
| `scores.json` | `0b28baec1e0a9e3d77989e8d594ea6e430c864488bd61206f43411ae1b2d8bc4` |

## Next gate

Before a full100 run, keep the replacement behind the same strict gate and
verify that construction remains identical for all non-applicable rows. In
parallel, persist the deterministic conversation graph at T1g and re-run its
own reduced assay. The graph lane must demonstrate unique recovery, exact path
receipts, and bounded warm latency; the ordered-story win must not be counted as
graph-specific evidence merely because both use source connectivity.
