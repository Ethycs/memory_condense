# Confirmation local store and S0-S3 retrieval

Date: 2026-09-03

## Implemented boundary

`tools/confirmation_namespace_store_adapter.py` compiles arbitrary sealed
confirmation populations into content-addressed namespace work.  A namespace
keeps the historical suffix-to-token-target memory construction separate from
its declared question probes.  It materializes each local base store once and
publishes no-clobber, resumable checkpoints.

`tools/confirmation_cumulative_retrieval.py` authenticates those base
checkpoints and runs the production cumulative retrieval interfaces through
S0, S1, S2, and S3.  Every stage preserves its receipt, provider prompt, and
monotone evidence prefix.  The ordered merge is deterministic regardless of
checkpoint-path order and can be projected into the generic `MatchedS0`
population boundary.  Neither module owns a responder, judge, or provider
callable; their physical provider-call count is fixed at zero.

`tools/confirmation_staged_cumulative_coordinator.py` supplies the bounded
population-wide two-phase lifecycle described below.

## Source-acquisition contract

The diffuse base publisher freezes exact anchor rows and therefore requires a
direct-result retrieval mode.  The frozen cumulative path does not pass its
packed `causal_graph` policy into this step.  It first derives
`source_acquisition_config(full_config)`, whose retrieval policy is the default
dense configuration with `k=10`.

The production base adapter now rejects any config that is not this derived
source configuration.  It also requires the frozen BGE-M3 execution identity:
model revision and checkpoint, dimension 1024, device bound to the config,
batch size 32, unnormalized float32 output.  Its artifact carries a sealed
source-treatment contract, and the cumulative consumer binds that contract,
the producing backend identity, the database path, and the database digest.

## Treatment-equivalence limit

The population-neutral namespace database is not byte- or coordinate-identical
to the historical validation current-source database.  It preserves complete
transcripts, exact timestamps, suffix composition, chunking, embedding, index,
and query-freezing mechanics.  It deliberately replaces ordinal/sample-derived
sample and source identifiers with content-addressed namespace coordinates.
Those identifiers participate in corpus, turn, chunk, graph, and tie-break
identities.  Consequently exact historical database identity and permutation /
renumbering neutrality cannot both be claimed.  The contract records the
narrower relation as
`same_transcript_timestamp_and_ingest_semantics_distinct_content_addresses`.

## Staged residency coordinator

The frozen validation lifecycle is two phase:

1. Keep BGE resident while building or opening the combined store and embed all
   held-out raw and dated questions into its `FrozenQueryEmbedder`.
2. Close BGE, load the shared Qwen coverage selector and representative linker,
   then retrieve S0-S3.

The staged coordinator now implements that ordering across an arbitrary
namespace population.  Phase A builds or verifies one combined store at a
time, captures the runtime's exact expanded held-out query batch, seals its
float32 vectors and preparation receipt, and closes the prepared store.  This
keeps memory bounded by one namespace.  Cache-hit verification reopens with
the sealed `FrozenQueryEmbedder`, so it does not recompute BGE vectors.

After every Phase-A checkpoint is present, the coordinator calls `close()` on
BGE and seals one population barrier binding all preparation and vector
digests.  Only then can `load_after_barrier()` construct Qwen.  Phase B uses a
bounded replay embedder that reads one authenticated vector artifact while the
existing production S0-S3 backend opens that namespace.  A missing combined
store cannot be rebuilt after the barrier.  Resident execution remains a
separate explicitly preflighted option; staged execution requires the barrier
and sealed-vector runtime kind.

The production staged Qwen factory is now concrete rather than a protocol-only
boundary.  It authenticates the persisted release barrier before invoking the
historical `_load_shared_qwen` constructor, requires the pinned Qwen3-8B prefix
and Qwen3-0.6B choice-checkpoint identities, and returns that exact shared
coverage-selector/representative-linker pair.  Its owned runtime closes the
choice scorer, drops both Qwen references, collects them, and clears the CUDA
allocator cache after Phase B.  A test-only loader seam is permitted only when
its distinct SHA-256 identity is included in the factory/barrier binding.

## Synthetic verification

The focused suite covers arbitrary namespace schedules, suffix composition,
namespace isolation, checkpoint resume/no-clobber, tamper rejection, stage
order and evidence monotonicity, deterministic merge/replay, population growth,
permutation and renumbering neutrality, packed source-policy rejection, exact
BGE identity enforcement, uncertified staged-lifecycle rejection, and post-checkpoint
database tamper detection.  Staged tests additionally prove the global
BGE-close-before-Qwen-load order, no-clobber resume across multiple namespaces,
rejection of vector tampering before Qwen construction, and concrete production
Qwen construction/closure through an identity-bound fake loader.

## Numeric frontier and frozen terminal overlay

`tools/materialize_confirmation_numeric_v5_overlay.py` is the provider-free
consumer after terminal completion.  Its required inputs are the authenticated
confirmation terminal inputs, the full frozen-v5 plan export, the sealed
terminal-v5 preflight and completion artifact, and the verified staged-store
set.  The full plan export is mandatory because the terminal preflight alone
does not retain the allowed-handle inventory, handle groups, or validation
contract needed to replay validator v5 safely.

For every applicable row, the production backend opens only its declared
cumulative `memory.db`, authenticates the database, index, combined-store
manifest, and preparation/barrier lineage, builds one immutable full-store
window index, and runs the operator-material-v3 census.  The overlay then calls
the authoritative pure numeric and typed-v5 policy functions and revalidates
both receipts.  Arbitration is fixed to supported numeric proof first,
accepted typed-v5 replacement second, and the exact protected parent third;
inapplicable rows are byte-exact parent passthroughs.

Each namespace produces a no-clobber checkpoint.  Supplying its externally
recorded digest enables fast replay without rescanning the namespace, while
all stored policy proofs are revalidated before reuse.  The deterministic
treatment-order merge emits the exact
`memory-condense-confirmation-final-answer-source-v1` schema consumed by the
prediction-plane materializer.  Focused synthetic coverage proves the
three-way priority, non-ordinal arbitrary ordering, one scan per namespace,
sealed resume, tamper failure, and final-source no-clobber replay.
