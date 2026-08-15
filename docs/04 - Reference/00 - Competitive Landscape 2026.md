# Competitive Landscape — Agent Memory Systems, mid-2026

**Status**: Living Document — last reconciled 2026-08-14
**Scope**: external systems and benchmarks this project is implicitly compared against. All numbers are from the linked sources, not reproduced here.

## The systems

| System | Approach | Headline number | Source |
| --- | --- | --- | --- |
| **SimpleMem** | Entropy-aware filtering → fact extraction (pronoun/timestamp resolution) → 3-layer index (dense + lexical + metadata) → async recursive consolidation → query-complexity-adaptive retrieval depth | 43.24 avg F1 on LoCoMo (GPT-4.1-mini) vs Mem0's 34.20; ~30× fewer inference tokens than full context | [arXiv 2601.02553](https://arxiv.org/pdf/2601.02553), [review](https://bluuewhale.github.io/posts/simple-mem-en/) |
| **Mem0** | LLM-extracted fact store + graph variant | 34.20 F1 LoCoMo; 49.0% LongMemEval (GPT-4o) | [Mem0 benchmark blog](https://mem0.ai/blog/ai-memory-benchmarks-in-2026) |
| **Zep** | Temporal knowledge graph (Graphiti) | 63.8% LongMemEval | [comparison](https://niteagent.com/blog/ai-agent-memory-comparison-2026/) |
| **Letta** | OS-inspired tiered memory (MemGPT lineage) | 83.2% LongMemEval | [comparison](https://blog.devgenius.io/ai-agent-memory-systems-in-2026-mem0-zep-hindsight-memvid-and-everything-in-between-compared-96e35b818da8) |
| **Hindsight** | (newer entrant) | 91.4% LongMemEval overall | same |

## The benchmarks

- **LoCoMo** — ~300-turn, ~9k-token conversations over up to 35 sessions; QA + event summarization; [site](https://snap-research.github.io/locomo/). Widely used, increasingly criticized.
- **LongMemEval** — up to 1.5M tokens, 500 questions, five temporal-complexity levels; considered the harder, more realistic test.

## The result that matters most to this project

**MemDelta** ([arXiv 2606.29914](https://arxiv.org/pdf/2606.29914)): on LongMemEval-S, Mem0 beat a simple RAG baseline 72.7% vs 61.4% with MiniLM embeddings — but swapping only the embedding model (same code, data, retrieval logic) put the RAG baseline at 73.9%, *ahead* of Mem0 by 1.2pp. The "+11pp memory-architecture gain" was an embedding confound. Where Mem0 does win, the advantage concentrates in a narrow question subset and costs substantially more on the write path.

**Implication**: this repo's current form — good chunking + bge-m3 (a strong embedder) + clean dense retrieval — is essentially the strong baseline that fancy architectures keep failing to beat fairly. Building Phases 2–4 (LLM memory ops, decay, tiering) is *not* obviously the highest-value move; benchmarking the current pipeline on LongMemEval is.

## Where memory_condense stands (honest read, 2026-08-14)

1. **Architecture**: not competitive as a product — SimpleMem already ships a superset of our planned Phases 1–4.
2. **Baseline quality**: plausibly competitive — per MemDelta, embedding quality + retrieval engineering carries most of the published gains, and that is exactly what we have.
3. **Eval methodology**: genuinely differentiated — zero-annotation self-replay on personal conversations (see `01 - Design/01`) is not something the listed systems offer.
4. **Unverifiable claims**: we have no common-benchmark numbers and no token/latency instrumentation, so any external comparison is currently talk. See the roadmap.
