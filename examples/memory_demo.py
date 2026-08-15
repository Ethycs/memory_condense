"""
Memory-layer demo: extraction with provenance, decay/tiering, supersede, and
deterministic context packing.

Usage:
    pixi run python examples/memory_demo.py

Downloads bge-m3 on first run (~2.3 GB), same as similarity_demo.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from memory_condense import MemoryCondenser
from memory_condense.decay import item_energy, item_heat
from memory_condense.schemas import CreateOp, MemoryType, PinState, PinOp, Provenance

DATA_DIR = Path("./demo_data_memory")

CONVERSATION = [
    ("user", "I prefer Python for this project and I'd rather avoid heavy frameworks."),
    (
        "assistant",
        "Understood. A small, dependency-light Python package it is.",
    ),
    ("user", "We decided to use SQLite for storage. It must never block on writes."),
    (
        "assistant",
        "SQLite in WAL mode handles that well - readers never block writers.",
    ),
    ("user", "Actually, correction: we're using hnswlib for the vector index, not FAISS."),
    ("assistant", "Noted, hnswlib it is - it's easier to ship on Windows."),
]


def show_memories(mc: MemoryCondenser) -> None:
    items = mc.memory.list_items()
    if not items:
        print("  (no memory items)")
        return
    for item in items:
        energy = item_energy(item)
        heat = item_heat(item).value
        pin = " PINNED" if item.is_pinned else ""
        print(f"  [{item.type.value:<10}] {heat:<4} e={energy:.2f}{pin}  {item.content}")
        for prov in item.provenance:
            print(f'      quoted from turn {prov.turn_id[:8]}: "{prov.quote[:60]}"')


def main() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    with MemoryCondenser(data_dir=DATA_DIR) as mc:
        print("=== 1. Ingesting conversation (memory extracted automatically) ===")
        for role, text in CONVERSATION:
            turn, chunks = mc.ingest(role, text)
            print(f"  [{role}] {len(chunks)} chunk(s)")

        print("\n=== 2. Extracted memory items (every one carries provenance) ===")
        show_memories(mc)
        print(f"\n  heat distribution: {mc.heat_counts()}")

        print("\n=== 3. Provenance is enforced: a fabricated memory is rejected ===")
        fake = CreateOp(
            type=MemoryType.DECISION,
            content="We decided to rewrite everything in Rust.",
            provenance=[
                Provenance(turn_id="does-not-exist", quote="rewrite everything in Rust")
            ],
        )
        from memory_condense.schemas import MemoryOps

        report = mc.validator.validate(MemoryOps(create=[fake]))
        for err in report.rejected:
            print(f"  REJECTED ({err.reason}): {err.detail}")

        print("\n=== 4. Pinning exempts an item from decay ===")
        items = mc.memory.list_items()
        if items:
            target = items[0]
            mc.memory.pin(PinOp(mem_id=target.mem_id, pin=PinState.USER))
            pinned = mc.memory.get(target.mem_id)
            print(f"  pinned: {pinned.content}")
            print(f"  is_pinned={pinned.is_pinned}, energy holds at {pinned.energy:.2f}")

        print("\n=== 5. Ranked memory recall ===")
        for result in mc.recall_memories("what storage are we using?", k=3):
            print(
                f"  [{result.score:.3f}] rel={result.relevance:.2f} "
                f"imp={result.importance:.2f} energy={result.energy:.2f}  "
                f"{result.item.content}"
            )

        print("\n=== 6. Budgeted context assembly ===")
        ctx = mc.build_context(
            "Remind me what we settled on for storage and indexing.",
            system_prompt="You are a helpful engineering assistant.",
        )
        print(f"  messages: {len(ctx.messages)}")
        print(f"  token budget used: {ctx.token_counts}")
        print(f"  dropped for budget: {ctx.dropped}")
        print("\n  --- memory header ---")
        print("  " + (ctx.memory_header or "(empty)").replace("\n", "\n  "))

    shutil.rmtree(DATA_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
