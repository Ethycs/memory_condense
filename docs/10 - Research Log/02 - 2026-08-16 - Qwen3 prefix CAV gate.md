# Qwen3-8B layers 0–5 pass the first controlled CAV gate; layer 6 does not

**Status**: measured local probe; not an end-to-end memory result  
**Cost**: $0; local BF16 CUDA inference  
**Model data**: one official Qwen3-8B shard, ignored by Git  
**Depends on**: [`00 - Theory/01 - Extracted Attention Heads as Recursive Associative Memory.md`](../00%20-%20Theory/01%20-%20Extracted%20Attention%20Heads%20as%20Recursive%20Associative%20Memory.md)

## Result

The seven-layer Qwen3-8B prefix exposes held-out, directionally stable concept
signals for both project-relevant probes. **Layer 5 is the current candidate for
the first live head-memory implementation.** There is no reason yet to download
shard 2: layers 0–5 all pass the predeclared common gate, while layer 6 fails
both concepts.

| Concept | Best layer | Held-out balanced accuracy | Bootstrap mean cosine | Random-label control |
| --- | ---: | ---: | ---: | ---: |
| context dependency | 5 | **1.000** | **0.894** | 0.553 |
| binding constraint | 5 | **0.812** | **0.750** | 0.529 |

The gate required, for the same layer and every concept:

```text
held-out balanced accuracy >= 0.75
bootstrap mean cosine       >= 0.50
random-label mean accuracy  <= 0.65
```

Common passing layers were 0, 1, 2, 3, 4, and 5. This establishes that the
downloaded prefix is sufficient to begin the live-memory prototype. It does
not establish that deeper layers would not be better.

## Method

The committed dataset contains 80 hand-written examples across two concepts.
Each concept has 12 positive and 12 negative fitting examples, plus eight
positive and eight negative held-out examples. Held-out wording deliberately
reduces direct keyword overlap with fitting examples.

For every example, one BF16 forward pass captures the pre-terminal-norm
residual at all seven retained layers. The CAV is the normalized difference
between positive and negative fitting centroids. The decision threshold is the
midpoint between the two fitting projection means.

Two controls address the easy high-dimensional failure mode:

1. bootstrap resampling measures whether fitting-example changes rotate the
   direction; and
2. random fitting-label permutations are scored against the true held-out
   labels.

The method intentionally avoids reporting training separability as evidence.
Every layer has 4,096 dimensions and only 24 fitting examples per concept, so a
training-only linear probe would be nearly meaningless.

## Layer curve

| Layer | Context held-out | Context stability | Constraint held-out | Constraint stability | Common pass? |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1.000 | 0.851 | 0.750 | 0.740 | yes |
| 1 | 1.000 | 0.855 | 0.750 | 0.835 | yes |
| 2 | 1.000 | 0.863 | 0.750 | 0.860 | yes |
| 3 | 0.938 | 0.867 | 0.750 | 0.824 | yes |
| 4 | 0.938 | 0.870 | 0.750 | 0.794 | yes |
| 5 | **1.000** | **0.894** | **0.812** | 0.750 | **yes** |
| 6 | 0.625 | 0.749 | 0.562 | 0.347 | no |

Random-label mean balanced accuracy ranged from 0.443 to 0.582 across all
concept/layer pairs. Layer 6's collapse therefore is not caused by a stricter
control result. This probe does not explain the collapse; interpreting it as a
general property of Qwen would exceed the evidence.

## Artifacts and reproduction

| Artifact | Status |
| --- | --- |
| `examples/cav/live_memory_concepts.json` | committed input dataset |
| `eval_results/qwen3_prefix_cav_probe.json` | generated, Git-ignored detailed report |
| `eval_results/qwen3_prefix_cav_probe.safetensors` | generated, Git-ignored CAV vectors |
| `src/memory_condense/cav_probe.py` | probe, controls, and artifact writer |

Reproduce with:

```powershell
pixi run -e dev qwen-cav-probe
```

The output records the input dataset SHA-256, seed, thresholds, per-layer
metrics, and CAV thresholds. The safetensors artifact stores every fitted
concept/layer vector with the model and dataset hash in metadata.

## Limitations

1. The examples are small, synthetic, English-only, and authored for this
   experiment. Accuracy may reflect writing style as well as the intended
   concept.
2. Mean-difference directions are the simplest CAV family. Logistic probes,
   whitening, and online updates remain untested.
3. This measures concept readout, not memory retrieval, recursion, pruning, or
   token savings.
4. The probe establishes a viable early layer, not an optimal one. No deeper
   layer was measured because only shard 1 is present.
5. The context-dependency concept is suited to deciding when retrieval is
   needed; it is not itself a content-addressing taxonomy.

## Decision

Proceed without another model download. Use layer 5 for the first live-memory
prototype, with layer 2 as an early-layer ablation. The next falsifiable step is
an append-immediate, in-memory per-head K/V store that:

1. writes a new episode from layer-5 normalized residuals;
2. uses the layer's 32 Q heads against eight GQA K/V heads;
3. recursively feeds retrieved OV output into a bounded next lookup;
4. keeps source text as the terminal evidence; and
5. compares direct QK, CAV-gated QK, and recursive QK+OV retrieval.

