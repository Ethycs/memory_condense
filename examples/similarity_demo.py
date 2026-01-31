"""
Phase 0 demo: Ingest real LLM conversations and retrieve chunks by similarity.

Usage:
    pixi run python examples/similarity_demo.py [conversation_dir]

If no directory is given, uses a small built-in sample conversation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from memory_condense import MemoryCondenser
from memory_condense.loader import load_conversation, load_directory


def demo_from_files(convo_dir: Path) -> None:
    """Ingest real conversation files and search across them."""
    conversations = load_directory(convo_dir)
    if not conversations:
        print(f"No .txt or .md conversation files found in {convo_dir}")
        return

    print(f"Found {len(conversations)} conversation files")

    with MemoryCondenser(data_dir="./demo_data") as mc:
        total_turns = 0
        total_chunks = 0
        for name, turns in conversations.items():
            file_turns = 0
            file_chunks = 0
            for role, text in turns:
                _, chunks = mc.ingest(role, text)
                file_turns += 1
                file_chunks += len(chunks)
            total_turns += file_turns
            total_chunks += file_chunks
            print(f"  {name}: {file_turns} turns -> {file_chunks} chunks")

        print(f"\nTotal: {total_turns} turns, {total_chunks} chunks indexed")

        queries = [
            "What is Shannon entropy?",
            "neural network weight initialization",
            "catastrophe theory and singularities",
            "AI safety and alignment",
            "context window and memory management",
        ]

        for query in queries:
            print(f"\n--- Query: {query} ---")
            results = mc.search(query, k=5)
            for i, r in enumerate(results, 1):
                role = r.turn.role if r.turn else "?"
                print(f"  {i}. [{r.score:.4f}] ({role}) {r.chunk.text[:120]}...")


def demo_builtin() -> None:
    """Small built-in demo with sample data."""
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
    ]

    with MemoryCondenser(data_dir="./demo_data") as mc:
        print("Ingesting sample conversation...")
        for role, text in conversation:
            turn, chunks = mc.ingest(role, text)
            print(f"  [{role}] {len(chunks)} chunk(s)")

        queries = [
            "What is the user's name?",
            "What storage technology is being used?",
            "What programming language does the user prefer?",
        ]

        for query in queries:
            print(f"\n--- Query: {query} ---")
            results = mc.search(query, k=3)
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r.score:.4f}] {r.chunk.text[:100]}...")


def main():
    if len(sys.argv) > 1:
        convo_dir = Path(sys.argv[1])
        if not convo_dir.is_dir():
            print(f"Error: {convo_dir} is not a directory")
            sys.exit(1)
        demo_from_files(convo_dir)
    else:
        demo_builtin()


if __name__ == "__main__":
    main()
