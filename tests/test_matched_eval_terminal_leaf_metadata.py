from __future__ import annotations

import copy

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.after_union_fact_closure import SelectedHLeaf
from tools.matched_eval.terminal_leaf_metadata import (
    FORMAT,
    METADATA_AUTHORITY,
    TerminalLeafMetadataError,
    authenticate_selected_leaf_projection,
    compile_terminal_leaf_metadata,
    compile_terminal_leaf_metadata_population,
)


def _leaf(
    handle: str = "H001",
    group: str = "G001",
    *,
    topics: tuple[str, ...] = (
        "kind:direct",
        "status:completed",
        "entity:road-bike",
    ),
    boundaries: tuple[str, ...] = (
        "group:G001",
        "date:2026-08-20",
        "relation:author-by-user-date",
    ),
) -> SelectedHLeaf:
    text = f"Exact selected memory for {handle}."
    return SelectedHLeaf(
        handle,
        group,
        text,
        quote_sha256(text),
        topic_labels=topics,
        boundary_labels=boundaries,
    )


def test_authenticated_labels_become_receipt_bound_provider_metadata() -> None:
    leaf = _leaf()

    metadata = compile_terminal_leaf_metadata(leaf)

    assert metadata.event_date == "2026-08-20"
    assert metadata.source_relation == "author-by-user-date"
    assert metadata.kind == "direct"
    assert metadata.status == "completed"
    assert metadata.entity_label == "road-bike"
    assert metadata.leaf_receipt_sha256 == leaf.receipt_sha256
    assert metadata.source_receipt_sha256 == leaf.source_receipt_sha256
    assert metadata.projection()["retained_transformer_token_state_bytes"] == 0
    assert metadata.provider_projection() == {
        "entity_label": "road-bike",
        "event_date": "2026-08-20",
        "format": FORMAT,
        "group_handle": "G001",
        "handle_id": "H001",
        "kind": "direct",
        "leaf_receipt_sha256": leaf.receipt_sha256,
        "metadata_authority": METADATA_AUTHORITY,
        "metadata_receipt_sha256": metadata.receipt_sha256,
        "source_relation": "author-by-user-date",
        "status": "completed",
    }
    assert "source_receipt_sha256" not in metadata.provider_projection()
    assert "chunk_id" not in metadata.provider_projection()


def test_optional_metadata_is_omitted_but_leaf_binding_remains() -> None:
    leaf = _leaf(topics=(), boundaries=("group:G001",))

    provider = compile_terminal_leaf_metadata(leaf).provider_projection()

    assert set(provider) == {
        "format",
        "group_handle",
        "handle_id",
        "leaf_receipt_sha256",
        "metadata_authority",
        "metadata_receipt_sha256",
    }


def test_exact_persisted_leaf_projection_is_reauthenticated() -> None:
    leaf = _leaf()

    assert authenticate_selected_leaf_projection(leaf.projection()) == leaf

    changed = copy.deepcopy(leaf.projection())
    changed["boundary_labels"][1] = "date:2026-08-21"
    with pytest.raises(ValueError, match="receipt|authenticated projection"):
        authenticate_selected_leaf_projection(changed)


@pytest.mark.parametrize(
    ("topics", "boundaries", "message"),
    [
        (
            ("kind:direct", "kind:source-fact"),
            ("group:G001",),
            "repeats field",
        ),
        (("topic:bike",), ("group:G001",), "unsupported field"),
        ((), ("group:G999",), "missing or foreign"),
        ((), ("date:2026-08-20",), "missing or foreign"),
        ((), ("group:G001", "date:2026-02-30"), "ISO calendar date"),
        (
            ("entity:Road-Bike",),
            ("group:G001",),
            "normalized lowercase",
        ),
        ((), ("group:G001", "source:user"), "unsupported field"),
    ],
)
def test_changed_or_ambiguous_label_grammar_fails_closed(
    topics: tuple[str, ...],
    boundaries: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(TerminalLeafMetadataError, match=message):
        compile_terminal_leaf_metadata(
            _leaf(topics=topics, boundaries=boundaries)
        )


def test_metadata_receipt_changes_with_the_authenticated_leaf() -> None:
    first = compile_terminal_leaf_metadata(_leaf())
    second = compile_terminal_leaf_metadata(
        _leaf(boundaries=(
            "group:G001",
            "date:2026-08-21",
            "relation:author-by-user-date",
        ))
    )

    assert first.receipt_sha256 != second.receipt_sha256
    assert first.provider_projection()["metadata_receipt_sha256"] == (
        first.receipt_sha256
    )


def test_population_preserves_selected_order_and_rejects_repeated_handles() -> None:
    first = _leaf("H001", "G001")
    second = _leaf(
        "H002",
        "G002",
        boundaries=("group:G002",),
    )

    compiled = compile_terminal_leaf_metadata_population((first, second))

    assert tuple(row.handle_id for row in compiled) == ("H001", "H002")
    with pytest.raises(TerminalLeafMetadataError, match="repeats"):
        compile_terminal_leaf_metadata_population((first, first))
