"""
Phase 0 demo: Ingest a conversation and retrieve chunks by similarity.

Usage:
    pixi run python examples/similarity_demo.py
"""

from memory_condense import MemoryCondenser


def main():
    conversation = [
        ("user", "My name is Alex and I'm building a memory system for LLMs."),
        (
            "assistant",
            "Nice to meet you, Alex! That's a fascinating project. "
            "Memory systems for LLMs can dramatically improve "
            "conversation coherence over long sessions.",
        ),
        (
            "user",
            "I prefer Python and I want to use SQLite for storage. "
            "The embedding model should be bge-m3 running locally.",
        ),
        (
            "assistant",
            "Great choices. SQLite is excellent for local storage - "
            "it's reliable, zero-config, and handles concurrent reads "
            "well with WAL mode. bge-m3 is a strong multilingual "
            "embedding model with 1024 dimensions.",
        ),
        (
            "user",
            "For the chunking strategy, I want sentences merged into "
            "120-250 token chunks. Can you help design that?",
        ),
        (
            "assistant",
            "Absolutely. A good approach is sentence boundary detection "
            "with pySBD, followed by a greedy merge pass. Split text "
            "into sentences, then merge consecutive sentences until "
            "the next merge would exceed 250 tokens. This preserves "
            "semantic coherence while staying within budget.",
        ),
    ]

    with MemoryCondenser(data_dir="./demo_data") as mc:
        print("Ingesting conversation...")
        for role, text in conversation:
            turn, chunks = mc.ingest(role, text)
            print(f"  [{role}] {len(chunks)} chunk(s)")

        queries = [
            "What is the user's name?",
            "What storage technology is being used?",
            "How does the chunking strategy work?",
            "What programming language does the user prefer?",
        ]

        for query in queries:
            print(f"\n--- Query: {query} ---")
            results = mc.search(query, k=3)
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r.score:.4f}] {r.chunk.text[:100]}...")


if __name__ == "__main__":
    main()
