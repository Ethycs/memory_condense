# Live concept links and the reachability boundary

**Status:** implemented as an opt-in Qwen/CAV path and measured on the locked
1M-token development stress. The storage mechanism is sound; the first event
concept did not improve final evidence coverage.

## Purpose

The prior 94.7% evidence-source result missed one of five concert sessions and
two of six museum sessions. The question was whether a live concept layer could
reach events that BGE/BM25 never admitted to bounded attention.

```text
new user chunks -> event-sized clauses -> two-layer Qwen residuals
                -> fixed-width CAV margins in SQLite

query -> conversation partitions -> positive event members
      -> TF-ISF object binding -> scalar/concept channel union
      -> bounded QK/OV tournament -> fixed prompt packet
```

No token activations, attention matrices, residual streams, or K/V caches are
stored. The measured artifact has one concept, so each indexed chunk costs one
float32 plus ordinary row/index overhead.

## Implementation

- `QwenMemoryLinker.signatures` batch-compiles pooled residuals without
  constructing attention maps.
- `MemoryCondenser.compile_cav_signatures` max-pools event-sized spans back to
  their durable chunk ID and is idempotent by artifact ID.
- `AssociationStore.put_signatures` writes a batch in one transaction.
- `AssociationStore.concept_members` provides bounded, source-filtered,
  source-unique concept lookup.
- multi-fact graph retrieval activates concept members after partition routing
  and can inject them before the ordinary source cutoff.
- candidate CAV margins participate in recursive QK/OV pruning.
- evaluation now distinguishes raw graph source coverage from packed prompt
  source coverage.

Artifact identity includes model/checkpoint, prefix/head/CAV layers, concept
names, CAV dataset/vector hashes, and pooling policy. Whole-chunk and
conceptual-span coordinates cannot be silently mixed.

## Probe and conceptual chunks

The hand-built `autobiographical_completed_event` development probe passed at
Qwen layers 0 and 1. Layer 0 achieved 93.8% held-out balanced accuracy, 0.772
bootstrap mean cosine, and 48.8% random-label-control accuracy. This is a
development gate, not independent validation.

Whole-chunk pooling produced negative margins for all three hard sources. A
plan or recommendation at the start of the chunk washed out the short event
statement. Clause-level max pooling corrected two:

| Missing source | Whole chunk | Concept spans |
| --- | ---: | ---: |
| Billie Eilish concert | -0.344 | +0.130 |
| Met Egyptian exhibition | -0.391 | +0.242 |
| Modern Art Museum tour | -0.392 | -0.025 |

This validates conceptual chunking as part of the representation, while also
showing that one linear event concept is not a complete semantic parse.

## Locked 1M-token measurements

All rows use 1,039,203 transcript tokens, 5,400 turns, ten locked development
questions, four coarse partitions, and at most eight candidates/1,024 tokens
in any live attention workspace.

| Variant | Evidence coverage | All-source questions | Mean prompt tokens |
| --- | ---: | ---: | ---: |
| protected scalar/Qwen union v25 | 94.7% | 80% | 1,981 |
| candidate CAV utility v26 | 94.7% | 80% | 1,981 |
| 250 streamed candidates v29 | 94.7% | 80% | 2,197 |
| durable whole-chunk concept index v30 | 94.7% | 80% | 1,981 |
| conceptual-span index v31 | 94.7% | 80% | 1,981 |
| direct concept-member attention v32 | 94.7% | 80% | 2,065 |
| scalar/concept set cover v33 | 94.7% | 80% | 2,067 |
| TF-ISF concept-object binding v34 | 94.7% | 80% | 2,063 |

The retrospective span build compiled 2,478 user chunks from 6,450 transient
spans. End-to-end build plus ten-query retrieval took about 134 seconds on the
local machine. Live operation should compile new user chunks asynchronously;
the benchmark backfill is not representative of per-turn steady state.

## Conclusion

The open problem is now narrower. Coarse routing reaches the correct history,
conceptual CAVs can recover events below the lexical/dense cutoff, and the
recursive head workspace remains constant-size. Yet relevance-style top-k and
the final prompt packer still do not guarantee complete sets.

The next experiment should use the new raw-versus-packed metric. If raw graph
coverage is 100% while packed coverage remains 94.7%, work belongs in a
source-aware set packetizer that reserves one compact excerpt per selected
event source. If raw coverage is incomplete, the concept bank needs
object/binding concepts (museum visit, concert attendance,
correction/negation) rather than more candidate breadth.

## Verification

`pixi run -e dev pytest -q` completed with 822 passing tests and one existing
Pydantic settings warning.

Machine-readable measurements are in
`data/longmemeval-million-context-concept-links-development-v1.json`.
