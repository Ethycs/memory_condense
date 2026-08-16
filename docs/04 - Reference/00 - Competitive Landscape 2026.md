# Competitive Landscape — Agent Memory Systems, mid-2026

**Status**: Living Document — last reconciled 2026-08-16
**Scope**: external systems and benchmarks this project is implicitly compared against. All numbers are from the linked sources, not reproduced here.

## The systems

| System | Approach | Headline number | Source |
| --- | --- | --- | --- |
| **SimpleMem** | Entropy-aware filtering → fact extraction (pronoun/timestamp resolution) → 3-layer index (dense + lexical + metadata) → async recursive consolidation → query-complexity-adaptive retrieval depth | 43.24 avg F1 on LoCoMo (GPT-4.1-mini) vs Mem0's 34.20; ~30× fewer inference tokens than full context | [arXiv 2601.02553](https://arxiv.org/pdf/2601.02553), [review](https://bluuewhale.github.io/posts/simple-mem-en/) |
| **Mem0** | LLM extraction + vector, lexical, entity-graph, and temporal retrieval; managed Platform and OSS/library capabilities differ | Vendor-reported 91.6% LoCoMo and 93.4% LongMemEval at 6,956 and 6,787 mean tokens | [current evaluation architecture and results](https://docs.mem0.ai/core-concepts/memory-evaluation), [Platform graph](https://docs.mem0.ai/platform/features/graph-memory) |
| **Zep** | Temporal knowledge graph (Graphiti) | 63.8% LongMemEval | [comparison](https://niteagent.com/blog/ai-agent-memory-comparison-2026/) |
| **Letta** | OS-inspired tiered memory (MemGPT lineage) | 83.2% LongMemEval | [comparison](https://blog.devgenius.io/ai-agent-memory-systems-in-2026-mem0-zep-hindsight-memvid-and-everything-in-between-compared-96e35b818da8) |
| **Hindsight** | (newer entrant) | 91.4% LongMemEval overall | same |

## The benchmarks

- **LoCoMo** — ~300-turn, ~9k-token conversations over up to 35 sessions; QA + event summarization; [site](https://snap-research.github.io/locomo/). Widely used, increasingly criticized.
- **LongMemEval** — up to 1.5M tokens, 500 questions, five temporal-complexity levels; considered the harder, more realistic test.

## The result that matters most to this project

**MemDelta** ([arXiv 2606.29914](https://arxiv.org/pdf/2606.29914)): on LongMemEval-S, Mem0 beat a simple RAG baseline 72.7% vs 61.4% with MiniLM embeddings — but swapping only the embedding model (same code, data, retrieval logic) put the RAG baseline at 73.9%, *ahead* of Mem0 by 1.2pp. The "+11pp memory-architecture gain" was an embedding confound. Where Mem0 does win, the advantage concentrates in a narrow question subset and costs substantially more on the write path.

**Implication**: this repo's current form — good chunking + bge-m3 (a strong embedder) + clean dense retrieval — is essentially the strong baseline that fancy architectures keep failing to beat fairly. Building Phases 2–4 (LLM memory ops, decay, tiering) is *not* obviously the highest-value move; benchmarking the current pipeline on LongMemEval is.

## Mem0 versus memory_condense, as of 2026-08-16

This comparison uses the current Mem0 Platform documentation, not the older
2025 paper as though the implementation were frozen there. Mem0 currently
describes an extraction phase that looks up context, writes deduplicated
ADD-only facts, links entities, and attaches temporal metadata. Reads fuse
semantic, BM25, entity, and temporal signals. Its managed graph connects entity
and memory nodes using co-occurrence; it does not currently expose typed edge
semantics. See the [official evaluation architecture](https://docs.mem0.ai/core-concepts/memory-evaluation)
and [official graph documentation](https://docs.mem0.ai/platform/features/graph-memory).

| Dimension | Mem0 | memory_condense |
| --- | --- | --- |
| Product maturity | Production library, self-hosted server, managed Platform, integrations, and public benchmark claims | Research implementation with local SQLite/HNSW/BM25 and an MCP facade |
| Durable memory | Extracted facts, vector records, entities/graph, temporal metadata, SQL history/audit | Append-only raw transcript plus chunks, typed lifecycle memory, exact provenance, CAV signatures, and sparse QK/OV edges |
| Graph meaning | Entity/memory co-occurrence used to boost retrieval | Learned conceptual/attention associations; a richer native hypergraph is proposed but not yet validated |
| Transformer role | LLMs participate in extraction and query-time memory pipeline | Qwen prefix is a staged write-time linker only; ordinary associative reads are model-free |
| Token state | Conventional memory records, not advertised as a persistent K/V-memory design | Explicit invariant: persist zero transformer token K/V or residual sequences |
| Context control | Fused scoring followed by top-N retrieval | Hard section budgets plus heat-weighted allocation of how much text each source contributes |
| Correction/pruning | Current evaluation pipeline emphasizes ADD-only fact writes plus deduplication | Explicit supersede/delete/pin lifecycle, edge usage/decay, degree pruning, and rebuildable raw provenance |
| Evidence | Strong vendor-reported public results; Platform/OSS parity is not guaranteed | Local containment/token replays only; no public answer-stage result yet |

The novel bet is therefore not "a vector database but smaller." It is that
compiled attention evidence can become a cheap external memory-control plane:
ranked QK preserves rare decisive associations, heat diffuses corroboration,
and the resulting source heat controls the prompt allocation. The risk is
equally clear: Mem0 has already solved far more of the production surface and
currently reports 91.6–93.4% public accuracy. memory_condense has not yet shown
that its extra graph semantics translate into answer accuracy.

The only fair head-to-head is same conversations, same answer and judge models,
same prompt-token cap, same number of questions, and separate write/read cost.
The project target is now **at least 95% judge accuracy over at least 100 locked
long-chat questions under an 8,000-token responder-prompt ceiling**. Until that
gate passes, token reductions are secondary.

## Where memory_condense stands (honest read, 2026-08-16)

1. **Architecture**: differentiated as a memory-control experiment, not yet competitive as a product.
2. **Baseline quality**: credible enough to test — bge-m3, BM25, pooled spans, and bounded compiled associations are all real paths.
3. **Context control**: locally promising — the selected QK/heat replay preserved its development recovery while reducing selected text, but this is posthoc evidence.
4. **Evaluation**: the harness now enforces a 95% judge-accuracy target, minimum question count, and hard prompt cap, but no public answer-stage run has passed it.
