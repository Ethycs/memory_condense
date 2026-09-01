from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from memory_condense.domain.integrity import file_sha256

from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools.matched_eval.protected_parent_contribution import (
    ProtectedParentContributionError,
    rehydrate_protected_parent_contributions,
)
from tools.matched_eval.typed_operator_adapter import FrontierMode
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


_ROOT = Path(__file__).resolve().parents[1]
_PARENT = (
    _ROOT
    / "eval_results"
    / "matched_eval_100"
    / "typed-memory-final-v3-shared-surplus"
    / "typed-memory-final-composition-v1.json"
)


@lru_cache(maxsize=1)
def _parent_payload() -> dict[str, Any]:
    if not _PARENT.is_file():
        pytest.skip("sealed fitted-parent composition is not present")
    value = json.loads(_PARENT.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _row(ordinal: int = 31) -> dict[str, Any]:
    return copy.deepcopy(_parent_payload()["questions"][ordinal])


def _spec(row: dict[str, Any]):
    dated = row["provider_projection"]["provider_input"]["dated_question"]
    return compile_typed_operator_spec(dated)


def _reseal(row: dict[str, Any]) -> None:
    row.pop("composition_row_sha256")
    row["composition_row_sha256"] = identity_sha256(row)


def test_real_fitted_parent_rehydrates_by_mechanism_without_compact_drift() -> None:
    row = _row()
    result = rehydrate_protected_parent_contributions(
        row,
        _spec(row),
        file_sha256(_PARENT),
    )

    assert [value.mechanism_id for value in result.contributions] == [
        "adaptive_parent_map_v1",
        "adaptive_parent_direct_pointer_v1",
        "full_store_slot_closure_v1",
        "active_reconstruction_v1",
    ]
    assert [len(value.bindings) for value in result.contributions] == [1, 1, 23, 8]
    assert len({value.sealed_artifact_sha256 for value in result.contributions}) == 4
    assert all(
        binding.sealed_artifact_sha256 == contribution.sealed_artifact_sha256
        for contribution in result.contributions
        for binding in contribution.bindings
    )

    compact = row["provider_projection"]["provider_input"]["typed_evidence"]
    rebuilt = result.compact_content_projection()
    assert rebuilt == {"handles": compact["handles"], "items": compact["items"]}
    assert result.audit.compact_projection_byte_identical is True
    assert result.audit.parent_frontier_reused is True
    assert result.audit.frontier_mode is FrontierMode.BOUNDED
    assert result.audit.truncated is True

    expected_handles = {value["handle_id"] for value in compact["handles"]}
    assert set(result.exact_span_keys_by_handle) == expected_handles
    assert len(result.local_selection_priority_by_handle) == 31
    assert len(result.source_ids) == 18
    assert all(row.source_ids for row in result.audit.source_provenance)
    for provenance in result.audit.source_provenance:
        original = provenance.original_binding.projection()
        cloned = provenance.cloned_binding.projection()
        changed = {key for key in original if original[key] != cloned[key]}
        assert changed == {"receipt_sha256", "sealed_artifact_sha256"}

    assert_gold_blind(result.audit.projection())
    assert result.audit.receipt_sha256 == identity_sha256(
        result.audit.projection(include_receipt=False)
    )

    replay = rehydrate_protected_parent_contributions(
        row,
        _spec(row),
        file_sha256(_PARENT),
    )
    assert replay.projection() == result.projection()
    assert replay.audit.projection() == result.audit.projection()


def test_malformed_parent_frontier_fails_closed_without_compact_drift() -> None:
    row = _row()
    frontier = row["provider_projection"]["provider_input"]["typed_evidence"][
        "frontier"
    ]
    frontier["mode"] = "unsealed_mode"
    frontier["truncated"] = False
    _reseal(row)

    result = rehydrate_protected_parent_contributions(
        row,
        _spec(row),
        file_sha256(_PARENT),
    )

    assert result.audit.compact_projection_byte_identical is True
    assert result.audit.parent_frontier_reused is False
    assert result.audit.frontier_mode is FrontierMode.BOUNDED
    assert result.audit.truncated is True
    assert all(
        value.frontier_mode is FrontierMode.BOUNDED and value.truncated
        for value in result.contributions
    )


def test_derived_compact_item_tamper_is_detected_after_typed_reparse() -> None:
    row = _row()
    item = row["provider_projection"]["provider_input"]["typed_evidence"][
        "items"
    ][0]
    assert item["content_coherence"] == "match"
    item["content_coherence"] = "conflict"
    _reseal(row)

    with pytest.raises(
        ProtectedParentContributionError,
        match="compact parent projection changed",
    ):
        rehydrate_protected_parent_contributions(
            row,
            _spec(row),
            file_sha256(_PARENT),
        )


def test_operator_spec_must_match_the_sealed_parent_question() -> None:
    row = _row()
    other = _row(61)

    with pytest.raises(
        ProtectedParentContributionError,
        match="operator question diverged",
    ):
        rehydrate_protected_parent_contributions(
            row,
            _spec(other),
            file_sha256(_PARENT),
        )
