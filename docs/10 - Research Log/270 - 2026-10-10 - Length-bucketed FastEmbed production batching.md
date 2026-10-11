# Length-bucketed FastEmbed production batching

**Date:** 2026-10-10  
**Status:** Integrated; exact-vector admission and scheduler tests pass

## Result

The resident CPU BGE path now sorts pending summaries by their BGE tokenizer
length, embeds them in groups of four, and restores the caller's original
order. Model admission remains recall-preemptible between batches. A live query
still runs alone at batch size one.

This change belongs in the resident scheduler as well as the FastEmbed adapter.
The scheduler previously divided every CPU request into individual calls, so
changing only the adapter would not have affected production ingestion.

## Admission evidence

The production-text assay used 48 cached summaries from the active LongMemEval
compiler workload. Against one FastEmbed call per summary:

| Policy | Elapsed | Throughput | Relative throughput |
| --- | ---: | ---: | ---: |
| Separate batch-one calls | 6.599 s | 7.274 texts/s | 1.000x |
| Token-length order, batch four | 4.762 s | 10.080 texts/s | **1.386x** |

Every FP32 vector was bit-identical. Same-text cosine was 1.0, and every top-ten
retrieval set was unchanged. The sealed runtime admission is
`src/memory_condense/runtime/data/bge-admission/batching.json`; it binds the
ignored full assay artifact by SHA-256.

The wider 512-summary workload explains why ordering matters. Its summaries
range from 3 to 232 BGE tokens. Arrival-order batches of eight perform 4.57x as
much attention-shaped padded work as the unpadded population; length-sorted
batches of eight reduce that ratio to 1.09x. Batch four was fastest in the
measured execution assay.

## Compatibility boundary

The batch policy is operational scheduling rather than vector identity. The
existing embedding identity remains unchanged because the admitted vectors are
bit-identical. Existing indexes, vector caches, and trained BGE tactic
classifiers therefore remain reusable instead of forcing a complete rebuild.

## Verification

- `tests/test_runtime_embedding_batching.py` proves adapter sorting, batch size,
  shape, and output-order restoration.
- `tests/test_resident_embedding_schedule.py` proves the resident scheduler
  forms globally length-bucketed groups of four while preserving summary and
  chunk ownership.
- The two focused modules pass 15 tests.
- Runtime package import and tactic-router checks pass 11 tests.

No additional heavyweight model assay was started while the sealed r12
benchmark was running. The measured model-level assay is reused rather than
duplicated.
