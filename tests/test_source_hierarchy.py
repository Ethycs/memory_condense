from __future__ import annotations

import numpy as np

from memory_condense.source_hierarchy import SourceContractionIndex
from memory_condense.transcript_store import TranscriptStore


def _add_source(db, source_id: str, vector: list[float], label: int) -> None:
    turn = TranscriptStore(db).append("user", source_id, source_id=source_id)
    db.execute(
        "INSERT INTO chunks "
        "(chunk_id, turn_id, text, start_char, end_char, token_count, "
        "embedding, hnsw_label, term_count) VALUES (?, ?, ?, 0, 1, 1, ?, ?, 1)",
        (
            f"chunk-{label}",
            turn.turn_id,
            source_id,
            np.asarray(vector, dtype=np.float32).tobytes(),
            label,
        ),
    )
    db.commit()


def test_hsc_expands_to_the_semantically_paired_source(db):
    _add_source(db, "project::alpha", [1.0, 0.0], 0)
    _add_source(db, "project::beta", [0.9, 0.1], 1)
    _add_source(db, "project::gamma", [-1.0, 0.0], 2)
    _add_source(db, "project::delta", [-0.9, 0.1], 3)
    hierarchy = SourceContractionIndex(db, dim=2, max_levels=3)

    expanded = hierarchy.expand(
        np.asarray([1.0, 0.0], dtype=np.float32),
        ["project::alpha"],
        slots=2,
        hops=1,
    )

    assert expanded == [("project::beta", expanded[0][1])]
    assert expanded[0][1] > 0.9


def test_hsc_never_crosses_an_explicit_top_level_partition(db):
    _add_source(db, "first::alpha", [1.0, 0.0], 0)
    _add_source(db, "first::beta", [0.8, 0.2], 1)
    _add_source(db, "second::near-copy", [1.0, 0.0], 2)
    hierarchy = SourceContractionIndex(db, dim=2)

    expanded = hierarchy.expand(
        np.asarray([1.0, 0.0], dtype=np.float32),
        ["first::alpha"],
        slots=5,
        hops=3,
    )

    assert [source_id for source_id, _score in expanded] == ["first::beta"]


def test_hsc_rebuilds_after_live_invalidation(db):
    _add_source(db, "project::alpha", [1.0, 0.0], 0)
    hierarchy = SourceContractionIndex(db, dim=2)
    assert hierarchy.stats()["sources"] == 1

    _add_source(db, "project::beta", [0.9, 0.1], 1)
    hierarchy.invalidate()

    assert hierarchy.stats()["sources"] == 2
    assert hierarchy.stats()["internal_nodes"] == 1
