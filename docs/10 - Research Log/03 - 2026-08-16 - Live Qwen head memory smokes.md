# Live Qwen head memory shows useful CAV entry and selected-head association signals, while direct QK and OV recursion fail

**Status**: measured local calibration and held-out-link development smokes; not a blind, common, or in-regime benchmark  
**Cost**: $0; local BF16 CUDA inference  
**Model data**: Qwen3-8B embeddings plus layers 0–6 from the first official shard  
**Depends on**: [`02 - Qwen3 prefix CAV gate.md`](02%20-%202026-08-16%20-%20Qwen3%20prefix%20CAV%20gate.md)

## Result

The live prototype writes source-grounded memories immediately, retains
per-token K/V tensors, uses residual and CAV signals for semantic entry, and
compiles sparse episode edges from per-head QK attention observed while
memories share a causal context.

The evidence favors a multi-layer design:

- **Layer 5** supplies the better residual/CAV entry representation.
- **Layer 1**, restricted to heads **31, 16, 26, and 8**, supplies the better
  association graph on this smoke.
- Direct QK between independently encoded queries and stored memory tokens is
  not useful as the primary address here.
- Feeding the QK-weighted V result through W_O into another query step is
  implemented, but does not improve recall here.

## Corrected QK/OV circuit

The first implementation ranked episodes by each head's maximum QK token score
and mixed one winning V token. That was only a retrieval heuristic. After the
QK/OV distinction was challenged, the arm was corrected to perform the actual
external-attention sequence:

```text
query tokens Q × every memory-token K
    → one softmax over all memory-token slots
    → attention-weighted V per query token and head
    → concatenated head result through W_O
    → residual update for the next hop
```

Episode recall scores are the attention probability mass assigned to their
tokens, aggregated over query tokens and the strongest heads. The corrected
arm still fails the lookup smoke, making the negative result more informative.

## Lookup smoke

Eight queries address 16 short constraint/observation memories. All arms use
the same memory texts and return three items.

| Arm | R@1 | R@3 |
| --- | ---: | ---: |
| Layer-5 residual | 0.125 | 0.500 |
| Layer-5 residual + bounded positive CAV gate | **0.750** | **0.875** |
| CAV/residual seed + QK association graph | 0.750 | 0.750 |
| Direct token-level QK→V→O | 0.000 | 0.000 |
| CAV-gated direct QK→V→O | 0.000 | 0.000 |
| Recursive CAV + QK→V→O | 0.000 | 0.000 |

The useful CAV term is a bounded type gate: each concept contributes a binary
above/below-threshold feature. Using the raw CAV margin created hubs and was
discarded because extreme projection magnitude overwhelmed relevance.

## Held-out association smoke

The association set contains eight two-step chains. Four anchor-to-fact links
select the heads; four evaluation queries target different, unseen chains.
The examples share an authored relation template, so this is a held-out link
split, not a held-out domain or author split.

| Layer-1 arm | R@1 | R@3 |
| --- | ---: | ---: |
| Residual seed | 0.000 | 0.750 |
| Residual seed + calibrated selected-head QK graph | **1.000** | **1.000** |
| Direct token-level QK→V→O | 0.250 | 0.500 |
| CAV-gated direct QK→V→O | 0.250 | 0.250 |
| Recursive CAV + QK→V→O | 0.250 | 0.250 |

Calibration selected heads 31, 16, 26, and 8, with link-recovery MRRs 0.8125,
0.7500, 0.7083, and 0.6667 respectively. The compiled graph contained 108
directed edges. The dataset disables its CAV weight, so the association gain
is isolated to residual seeding plus selected-head graph traversal.

The first association implementation found all four answers at R@3 but left
the semantic seed at rank 1. Two corrections converted that into the reported
R@1 result: calibration now selects temporal orientation as well as heads, and
retrieval fuses graph support with semantic scores before sorting. Edges to an
already seeded destination count as corroboration; max fusion prevents cycles
from repeatedly increasing a score. On graphs without direction calibration,
semantic seed order remains authoritative, which preserved the separate lookup
smoke at 0.750/0.750 for its association arm.

Because the reranking correction was designed after inspecting errors on these
four evaluation queries, this set is now a development smoke. Its link
identities remain separate from head/direction calibration, but 1.000/1.000
must replicate on a fresh blind set before it counts as held-out evidence for
the corrected algorithm.

## Association layer sweep

The layer decision was measured rather than inherited from the CAV result.
This preliminary sweep used all eight authored links for head calibration, so
only the subsequent layer-1 run above is the held-out result.

| Layer | Residual R@3 | Association R@3 | Best-head calibration MRR |
| ---: | ---: | ---: | ---: |
| 0 | 0.375 | 0.375 | 0.609 |
| 1 | 0.375 | **0.750** | **0.906** |
| 2 | 0.375 | 0.500 | — |
| 3 | 0.375 | 0.500 | — |
| 4 | 0.375 | 0.500 | — |
| 5 | 0.500 | 0.500 | — |
| 6 | **0.625** | 0.375 | — |

Layer 1 was the only layer in the calibration sweep with a material graph gain.
After calibrated-direction score fusion, it reached R@1/R@3 1.000/1.000 on
the unseen links.

## Live use and pruning

Reads through the direct head circuit now record two decayed per-memory values:

1. QK attention probability mass assigned to that memory's token slots; and
2. the RMS residual contribution caused by that memory after its weighted V
   values pass through W_O.

Pruning uses those live-head quantities with importance and turn recency.
Access count remains diagnostic rather than acting as a permanent popularity
advantage. Pins remain hard constraints. This lifecycle is mechanically tested
but has not yet been evaluated for recall retention under a storage budget.

## J-Space implication

The Jacobian-lens/J-Space result suggests a more principled next compiler:
identify the sparse token-labelled directions that actually affect later
verbalizable content, then retain OV heads that preserve and broadcast those
directions. This could replace generic OV norm with concept-bearing OV utility.

The present prefix cannot produce a true J-lens because the averaged Jacobian
includes all downstream transformer layers, final normalization, and
unembedding. All official BF16 teacher shards are now cached and verified, but
J-lens environment integration/fitting was paused to focus on this measured
retrieval path. A full teacher pass will be required during compilation even
if only the compiled dictionary and selected heads remain at runtime.

## Limitations

1. The lookup set has eight queries and the held-out association set has four.
   A single query changes recalls by 0.125 and 0.250 respectively.
2. Both datasets are synthetic and authored alongside the implementation.
3. The association split changes entity/link identity but not the template.
4. The corrected reranker was informed by the four evaluation errors, so a
   fresh blind split is required.
5. Selected-head stability has not been bootstrapped across corpora or seeds.
6. No prompt token count, answer recall, latency distribution, disk cost, or
   GPU amortization has been compared with the existing hybrid retriever.
7. Pruning is mechanically tested, not utility-benchmarked.
8. OV recursion's failure is local to this formulation; it is not evidence
   against J-Space-aligned transport or every learned recursive update.

## Decision

Preserve direct-QK and OV-recursion as controls, but do not put them in the
default path. Build the next opt-in experiment around layer-5 residual/CAV
entry, layer-1 selected-head association edges, source text as terminal
evidence, live-head utility, and versioned persistence. A full-teacher J-lens
compiler is a separate stage and requires the remaining Qwen weights.

Reproduce with:

```powershell
pixi run -e dev qwen-head-memory-smoke
pixi run -e dev qwen-head-association-smoke
```
