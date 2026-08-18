# Bounded local attention after coarse routing

**Status:** implemented and measured on the locked 1M-token development
stress; promising mechanism, not admitted as the default policy.

## Question

Can Qwen attention heads become a useful retrieval operator after a cheap
coarse search reduces the memory to a bounded local candidate set?

The motivating decomposition is the Neural Storage paper's coarse cue,
locality, fine cue, and data-item path. Its fully connected graph and lossy
data-strength mechanism are not suitable here, but the two-stage cue path is:

```text
question -> conversation partitions -> local event candidates
         -> bounded QK/OV selection -> provenance-preserving prompt
```

Reference: <https://arxiv.org/html/2101.02729v1>

## Fixed evaluation

- LongMemEval-S cleaned dataset, locked development split
- one 1,039,203-token memory containing 5,400 turns from ten namespaced
  histories
- ten questions, no provider calls
- final prompt and consolidation budgets unchanged
- role-aware retrieval retained from the selected v12 arm

## Coarse routing

Partition ranking uses bounded reciprocal-rank heat over the role-adjusted
coarse candidate pool. Diagnostics now retain the selected partition IDs and
text-free ranking evidence in offline CSVs.

The correct history ranked first for seven questions, second for two, and
fourth for one. Consequently a one-history beam was unsafe, while four
histories gave 100% coarse-routing recall on this prefix.

| Arm | Evidence coverage | Literal reachability | Mean context |
| --- | ---: | ---: | ---: |
| role-aware unrestricted v12 | 93.0% | 50.0% | 1,908 |
| role-aware one partition v16 | 78.0% | 40.0% | 1,897 |
| role-aware two partitions v17 | 86.3% | 40.0% | 2,035 |
| role-aware four partitions v19 | **94.7%** | **50.0%** | 2,031 |

The four-partition arm recovered a fourth of six museum sessions. It retained
four of five concert sessions. Eight of ten questions retained every labelled
source.

## Bounded attention

The existing two-layer Qwen3-8B controller inspected at most eight candidates
and 1,024 tokens per forward. It retained only chunk IDs and scalar QK/OV
scores; transformer token state remained transient.

A broad 250-source activation frontier alone regressed evidence coverage to
91.3%. Ordinary Qwen replacement raised it to 93.0% but took 240 seconds for
ten questions. The tournament was then made source-unique for explicit
multi-fact queries: only the strongest candidate per durable source/session
may enter, so repeated chunks from one event cannot consume the attention
reserve.

| Arm | Evidence coverage | Literal reachability | Mean context | Wall time |
| --- | ---: | ---: | ---: | ---: |
| broad scalar a250 v21 | 91.3% | 50.0% | 1,935 | 49 s |
| ordinary QK/OV reserve v22 | 93.0% | 50.0% | 1,961 | 240 s |
| source-unique QK/OV, 6 slots v23 | **94.7%** | **50.0%** | 1,976 | 56 s |
| source-unique QK/OV, 8 slots v24 | 94.7% | 50.0% | 1,976 | 57 s |

Source uniqueness therefore recovered 1.7 coverage points over the identical
broad scalar frontier and cut the observed attention run from 240 to 56
seconds by eliminating duplicate-session candidates. It matched, but did not
beat, the narrower scalar beam.

The museum evidence was complementary: the scalar beam retained sessions 1,
2, 5, and 6, while source-unique attention retained 1, 2, 4, and 6. Their union
would retain five of six. Both arms still missed the earliest Billie Eilish
concert session.

## Decision

- Keep source-unique attention implemented and opt-in.
- Keep six Qwen reserve slots; eight changed no packed evidence.
- Do not replace the scalar default: equal coverage at higher compute is not a
  policy win.
- The next admissible experiment is a protected union: retain the narrow
  scalar event sources and spend a bounded attention reserve only on new
  sources from the broader local frontier. Attention should explore evidence
  omitted by the scalar path, not rerank away its winners.
- This is still a ten-question development retrieval result, not 95% judged
  answer accuracy.

## Verification

`pixi run -e dev pytest -q` completed with 816 passing tests and one existing
Pydantic settings warning.

Machine-readable details are in
`data/longmemeval-million-context-local-attention-development-v1.json`.
