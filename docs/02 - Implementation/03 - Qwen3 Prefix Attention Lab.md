# Qwen3-8B runs as a bounded seven-layer memory-link inspector, not as the memory store

**Status**: 🟡 EXPERIMENTAL — bounded inspection plus SQLite-persistent compact CAV/QK/OV retrieval is built and locally source-family confirmed; Redis/Chroma backends and a public benchmark remain open
**Date**: 2026-08-16
**Applies to**: the optional local attention-head/CAV research path
**Depends on**: [`00 - Theory/01 - Extracted Attention Heads as Recursive Associative Memory.md`](../00%20-%20Theory/01%20-%20Extracted%20Attention%20Heads%20as%20Recursive%20Associative%20Memory.md)

## 1. What is installed

The experimental runtime uses the official Apache-2.0 [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) checkpoint with CUDA PyTorch inside the normal Pixi environment.

| Property | Value |
| --- | --- |
| Original checkpoint | Qwen3-8B, BF16, 36 decoder layers |
| Original checkpoint size from its index | 16,381,470,720 bytes |
| Cached teacher weights | all five official BF16 safetensors shards |
| Cached file bytes | 16,381,516,776 |
| Indexed tensor bytes | 16,381,470,720 |
| Retained model | embeddings plus complete layers 0–6 |
| Runtime weight shard | `model-00001-of-00005.safetensors` |
| Runtime shard size | 3,996,250,744 bytes |
| Retained parameter count | 1,972,958,976 |
| Query heads per retained layer | 32 |
| KV heads per retained layer | 8 |
| Head dimension | 128 |
| Residual dimension | 4,096 |
| Runtime dtype | BF16; no weight quantization |

The first shard also contains three tensors from layer 7. They are intentionally ignored because the rest of layer 7 is in shard 2. Loading those tensors would create an invalid partial layer rather than a coherent prefix.

Model data lives under `.cache/models/Qwen3-8B/`, which is ignored by Git. The
full teacher was downloaded after the prefix results, when J-Space was
considered. It is not loaded by the prefix, CAV, or live-head smoke tasks.

## 2. Environment setup

`pixi.toml` now resolves conda-forge's CUDA build of PyTorch and pins the CUDA runtime to 12.6. The verified local environment is:

```text
torch 2.7.1
CUDA build 12.6
CUDA available: true
GPU: NVIDIA GeForce RTX 2070 SUPER
```

Install or synchronize it normally:

```powershell
pixi install -e dev
```

Download the metadata, tokenizer, and only the required BF16 shard:

```powershell
pixi run -e dev qwen-download-prefix
```

The task is idempotent. Hugging Face verifies and reuses files already present in the local directory.

The future offline Jacobian-lens experiment requires all downstream layers,
terminal norm, and unembedding. Fetch the four remaining shards separately:

```powershell
pixi run -e dev qwen-download-full-teacher
```

This download is complete locally, but J-lens integration and fitting were
deliberately paused to focus on improving the existing retrieval result.

## 3. Verification

Run the real CUDA smoke path:

```powershell
pixi run -e dev qwen-smoke
```

The command:

1. Reads the complete five-shard index.
2. Proves the locally present shards contain seven complete contiguous layers.
3. Constructs a seven-layer `Qwen3Model` on the meta device.
4. Streams only embeddings and layers 0–6 from safetensors to CUDA.
5. Ignores the incomplete layer-7 tensors.
6. Processes a short text through the prefix.
7. Captures layer-6 residuals, real RoPE-adjusted Q/K tensors, V tensors, attention weights, aggregate attention output, and one head's isolated $W_O$ contribution.

Measured smoke output on this machine:

```text
residual:             [1, 10, 4096]
queries:              [1, 32, 10, 128]
keys:                 [1, 8, 10, 128]
values:               [1, 8, 10, 128]
attention:            [1, 32, 10, 10]
attention output:     [1, 10, 4096]
selected head output: [1, 10, 4096]
peak CUDA allocation: 3,956,855,296 bytes
```

## 4. Python interface

```python
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder, mean_pool_residual

encoder = Qwen3PrefixEncoder(
    ".cache/models/Qwen3-8B",
    layers=7,
    device="cuda",
    dtype="bfloat16",
)

capture = encoder.capture(
    "A live memory system associates concepts through selected heads.",
    layer=6,
)

concept_example = mean_pool_residual(capture.residual)
head_0_output = capture.output_for_head(0)

print(capture.queries.shape)   # [batch, 32, tokens, 128]
print(capture.keys.shape)      # [batch, 8, tokens, 128]
print(capture.attention.shape) # [batch, 32, tokens, tokens]
print(concept_example.shape)   # [batch, 4096]
```

`Qwen3PrefixEncoder.encode()` returns the actual last retained layer residual before the synthetic terminal norm. The full teacher's terminal norm is stored in shard 4 and is neither downloaded nor needed for prefix-layer CAVs. `Qwen3Model.forward()` requires a terminal norm structurally, so the loader materializes a neutral all-ones norm and discards its output in favor of the hooked pre-norm residual.

`capture_layers()` captures several selected layers in one prefix pass. It does
not request an attention matrix from every retained layer; selected-layer QK
maps are reconstructed from their captured RoPE-adjusted queries and keys.
This avoids retaining seven simultaneous quadratic attention maps.

## 5. Runtime rule: inspect memory, never accumulate it inside Qwen

The production candidate is a transient inspector:

```text
external episode store / sparse graph
    -> fetch at most k candidate texts
    -> one capped Qwen workspace computes QK, OV, and a CAV signature
    -> retain compact edges, coordinates, and provenance externally
    -> discard all token activations and K/V
```

Recursive or nested memory layers repeat this operation with a fresh bounded
candidate set. A later hop receives IDs selected by the preceding hop; it does
not receive an ever-growing concatenation of prior transformer context.

`QwenMemoryLinker` enforces candidate and token ceilings (currently 8 and
1,024 by default). `CAVLinkIndex` stores only float32 concept coordinates and
Concept↔Episode membership. `HeadAssociationGraph` stores sparse per-head QK
evidence and scalar OV transport, with a hard degree bound. The older
`QwenLiveHeadMemory` remains a bounded lab workspace and refuses more than 64
items by default. It is not a corpus store.

## 6. Current boundary

**Built and verified:**

- Selective official-shard download
- Complete-prefix detection from the full safetensors index
- Streaming materialization without constructing the 8B model in RAM
- BF16 CUDA execution
- Residual, Q/K/V, QK attention, and per-head output capture
- Batched residual capture across layers 0–6
- Held-out CAV fitting with bootstrap stability and random-label controls
- One-pass multi-layer capture for entry-head and CAV compilation
- A hard-capped, transient joint candidate workspace
- QK association scores and actual OV transport compiled into sparse edges
- Compact CAV signatures, Concept↔Episode links, and CAV coactivation counts
- Degree-bounded graph storage and graph-safe episode removal
- Schema-v6 persistence of source-identified turns, versioned float32 CAV signatures, and sparse QK/OV edges
- Close/reopen-safe, opt-in association expansion under the hybrid result cap
- Bounded two-hop traversal carrying only IDs, scalar scores, and compact paths
- Conserved source-heat diffusion with restart, row-normalized QK/OV edge
  utility, multi-parent support, and a fixed scalar frontier
- Dual-channel association allocation: one ranked-QK exploitation slot plus
  one heat-weighted exploration slot
- Heat-aware context packing with measured per-source text exposure and an
  optional per-source expansion cap
- Safe public-facade admission that protects strong lexical anchors and rejects
  any association result that increases prompt tokens
- Usage-aware persistent edge pruning and synchronized chunk-artifact removal
- Parallel model-free sweep arms with isolated SQLite readers and physical
  prune-copy measurements
- A separate capped GQA-aware K/V laboratory implementation
- CAV-gated residual entry retrieval
- Shared-context QK association-edge compilation and selected-head traversal
- Recursive OV-to-query retrieval as a measured negative control
- Decayed per-memory QK mass and OV transport for live utility/pruning
- A leakage-resistant delayed-feedback policy that learns role-separated head
  gates and sparse edge utility from the following turn, while serializing
  scalars/IDs only
- Pins, importance, turn recency, and utility pruning

**Not integrated or established:**

- Redis live-graph or Chroma document/vector backend implementations (the
  storage contract is intentionally backend-oriented, but SQLite is the only
  implementation currently measured)
- A production conceptual-chunk schema or chunk-boundary learner
- Automatic invocation of Qwen during ordinary ingestion; compilation remains
  an explicitly staged offline/write-time operation
- A true Jacobian lens/J-Space dictionary; that requires the missing downstream
  model layers, terminal norm, and unembedding during offline compilation
- Online CAV refitting or consolidation
- A completed causal next-turn replay showing that learned head gates improve
  held-out retrieval; the policy is implemented but deliberately not admitted
  to reads or pruning before that gate
- A public/common-benchmark comparison and a replicated fresh recall gain

The local smokes establish two useful but narrow signals. On the eight-query
lookup set, layer-5 residual retrieval improved from R@1 0.125/R@3 0.500 to
0.750/0.875 when a bounded positive-CAV type gate was added. On four
held-out-link development chains, calibrated layer-1 association traversal improved R@1/R@3
from 0.000/0.750 to **1.000/1.000** after heads 31, 16, 26, and 8 plus the
forward temporal orientation were selected on four separate links.
Direct cross-sequence QK and recursive QK+OV did not work. These are tiny
synthetic calibration smokes, not a token-saving or general-recall claim.
The reranker was corrected after inspecting those four errors, so a fresh
blind split is required before treating 1.000/1.000 as held-out evidence.

On the real B0 build-session development set, retaining every unique hybrid
top-10 anchor and recycling only exact-duplicate slots with one local QK link
and layer-5 residual candidates moved answer containment from 92.3% to 100.0%
at 1,505 to 1,547 mean context tokens. This is development-tuned evidence,
not a blind result. The important architectural result is that the successful
QK route needed only a small local candidate workspace; it did not require a
corpus-sized transformer K/V store.

The corrected no-storage inspector was then tested on B0's two misses that
have useful local neighbors. It inspected four direct-anchor neighborhoods in
fresh four-candidate workspaces and recursively reduced their finalists. With
a fixed 1,152-token ceiling, neither query used more than 1,056 tokens in any
pass; q0's gold finished rank 1 and q38's gold rank 2. Each query required
seven passes / 28 candidate inspections. No token K/V survived a pass. The
1,024-token control dropped q38 during an extra pairwise reduction, so the
result is sensitive to the explicit workspace budget rather than free.

The first source-family-separated notes slice was also a ceiling case: hybrid
top-10 and compiled-link top-10 both achieved 8/8 containment. Hybrid top-3
alone retained 8/8 at 636 mean tokens versus 2,141 at top-10 (70.3% fewer),
while reserving one of those three slots for a compiled QK/OV link retained
8/8 but increased the mean to 672 tokens. The links did not earn their cost on
these easy probes. Cold compilation of 113 chunks retained zero request-derived token K/V,
904 bytes of float32 CAV payload, and about 43 KB of per-head edge payload.
Pruning 337 directed edges to 225 preserved 8/8. These notes results are now
development data.

A later locked 18-question split first falsified unconditional replacement:
recall fell from 100% to 94.4% when a high-confidence rank-five lexical anchor
was displaced. The resulting safe admission rule protects near-max lexical
anchors (`normalized_score >= 0.90`) and rolls back the entire associative
result when it would add prompt tokens. That rule was frozen before a third,
fresh six-family split. On that locked confirmation, hybrid `k=5` recalled
83.3% at 973.9 mean prompt tokens. Safe CAV, QK, and QK+CAV retained 83.3%
recall at 947.7, 961.1, and 950.1 tokens respectively. Degree-two physical
pruning retained the same recall at 967.9 tokens while reducing 1,204 directed
edges to 812 and per-head payload from 154,112 to 103,936 bytes. Qwen was not
loaded for any read arm and no token K/V was persisted.

This confirms local token saving and recall non-regression, not a recall gain:
none of the three fresh misses was recovered. A three-hop diagnostic also did
not recover a miss and regressed the development split, so two hops remains the
default. Exact protocol, hashes, and artifacts are recorded in
[`10 - Research Log/04 - 2026-08-16 - Safe associative memory confirmation.md`](../10%20-%20Research%20Log/04%20-%202026-08-16%20-%20Safe%20associative%20memory%20confirmation.md).

The next development pass made the graph's scalar scores control source
exposure. A finite heat walk now follows the compact stored QK/OV edges, sums
support from multiple paths, and hydrates text only after selecting a capped
ID frontier. Pure heat reduced development tokens by 19.3% but failed to retain
ranked QK's one recovery. The selected two-hop dual policy therefore reserves
one result slot for ranked QK and one for heat. With degree-two pruning it
replayed at 91.7% recall / 828.7 mean tokens on development, 100% / 927.0 on
v2, and 83.3% / 924.1 on the prior locked-confirmation split. Those are
posthoc replays, not a fresh confirmation. Ordinary reads still load no Qwen
weights and retain zero request-token-state bytes. Static checkpoint and
tokenizer assets are reusable linker machinery, not memory, and are outside
that metric. Full protocol and caveats are in
[`10 - Research Log/05 - 2026-08-16 - Source heat diffusion development.md`](../10%20-%20Research%20Log/05%20-%202026-08-16%20-%20Source%20heat%20diffusion%20development.md).

The delayed turn learner has now received its first transfer test. Exact
next-chunk targets were rarely present in the narrow compiled graph, and the
development-selected global CAV-transition and CAV-velocity controls both made
delta-cosine worse on a separate compiled store. The existing two-dimensional
CAV bank is not rich enough to route live memory. These weights are therefore
not admitted to read ranking or pruning.

A separate local full-Qwen answerer was also wired into the benchmark harness
to keep generation K/V ephemeral. With the available 8 GB GPU and CPU offload,
one question did not complete checkpoint placement within ten minutes. This
does not affect the prefix linker, but it closes the full 36-layer local model
as a practical responder path on this machine.

## Verification block

```powershell
pixi run --frozen -e dev pytest -q tests/test_qwen_prefix.py tests/test_association_store.py tests/test_heat_diffusion.py tests/test_condenser.py
pixi run -e dev qwen-smoke
pixi run -e dev qwen-cav-probe
pixi run -e dev qwen-head-memory-smoke
pixi run -e dev qwen-head-association-smoke
```

The unit tests check shard selection, schema migration, compact artifact
serialization, restart behavior, fixed retrieval caps, pruning, and deletion
synchronization. The Qwen tasks run the real seven-layer BF16 prefix on CUDA.
The next evidence gate is replaying the notes benchmark through the persisted
close/reopen path, followed by a harder untouched source-family comparison
against the hybrid baseline.
