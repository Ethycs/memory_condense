from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import assay_hot_v3_typed_operator_full100 as assay


def _diagnostic(index: int) -> dict[str, object]:
    applicable = index == 0
    status = "supported" if applicable else "non_deterministic"
    executor = "numeric" if applicable else "none"
    slots = [{"slot_id": "s"}] if applicable else []
    return {
        "adapter_audit_sha256": "1" * 64,
        "adapter_bundle_sha256": "2" * 64,
        "candidate_arbiter_required": not applicable,
        "deterministic_executor_applicable": applicable,
        "evidence_coverage": {
            "all_surviving_included_items_citation_backed": True,
            "item_count": 1,
            "represented_source_group_count": 1,
        },
        "evidence_packet": {
            "frontier": {"closed": False, "mode": "bounded"},
        },
        "evidence_consensus": {"receipt_sha256": "3" * 64},
        "effective_packed_execution": {
            "prediction": "2" if applicable else "",
            "source": "operator_first_numeric_policy" if applicable else "generic_typed_operator",
            "status": status,
        },
        "escalation_state": {"receipt_sha256": "8" * 64},
        "escalation_anchor": {"authenticated_anchor_count": 1, "basis": "fixture"},
        "local_inventory": {"provider_use_forbidden": True},
        "local_operator_first_numeric": {
            "reason": "fixture",
            "status": status if applicable else "insufficient",
        },
        "local_operator_first_numeric_admissible": applicable,
        "local_vs_capped": {
            "operator_item_delta": 0,
            "packed_eligible_handle_count": 1,
            "provider_capacity_omitted_handle_count": 0,
            "provider_preserved_packed_handle_count": 1,
        },
        "next_action": {
            "next_action": "none" if applicable else "semantic_global",
            "receipt_sha256": "9" * 64,
        },
        "operator_execution": {
            "executor": executor,
            "prediction": "2" if applicable else "",
            "reason": "fixture",
            "status": status,
        },
        "operator_first_numeric": {
            "mode": "fixed_scalar_comparison",
            "reason": "fixture",
            "status": status if applicable else "insufficient",
        },
        "operator_first_numeric_admissible": applicable,
        "operator_spec": {
            "required_slots": slots,
            "requires_complete_frontier": applicable,
            "style": "numeric_reduce" if applicable else "direct_extract",
        },
        "ordinal": index,
        "prompt_question_sha256": f"{index + 4:064x}",
        "provider_input_sha256": "5" * 64,
        "provider_packet_receipt_sha256": "6" * 64,
        "question_id": f"q{index}",
        "retrieval_query_sha256": f"{index + 7:064x}",
        "slot_closure": {
            "bound_slot_ids": ["s"] if applicable else [],
            "conflicted_slot_ids": [],
            "missing_slot_ids": [],
            "sufficient": applicable,
        },
    }


def test_extract_dated_question_uses_frozen_question_envelope() -> None:
    arm = {
        "provider_messages": [
            {"role": "system", "content": "policy"},
            {
                "role": "user",
                "content": "Retrieved excerpts:\ntext\n\nQuestion: [Question asked at 2026-01-01] How many?\nShort answer:",
            },
        ]
    }
    assert assay._extract_dated_question(arm) == (
        "[Question asked at 2026-01-01] How many?"
    )
    arm["provider_messages"][1]["content"] += " changed"
    with pytest.raises(ValueError, match="frozen QA question envelope"):
        assay._extract_dated_question(arm)


def test_aggregate_partitions_applicability_and_obligations() -> None:
    aggregate = assay._aggregate([_diagnostic(0), _diagnostic(1)])
    assert aggregate["question_count"] == 2
    assert aggregate["deterministic_executor_applicable_count"] == 1
    assert aggregate["candidate_arbiter_required_count"] == 1
    assert aggregate["execution_supported_count"] == 1
    assert aggregate["obligation_total_slot_count"] == 1
    assert aggregate["obligation_bound_slot_count"] == 1
    assert aggregate["obligation_closure_rate"] == 1.0
    assert aggregate["bounded_frontier_count"] == 2


def test_numeric_overlay_cannot_close_required_bounded_frontier() -> None:
    supported = SimpleNamespace(status=assay.ExecutionStatus.SUPPORTED)
    bounded = SimpleNamespace(frontier=SimpleNamespace(closed=False))
    assert not assay._numeric_policy_is_admissible(
        spec=SimpleNamespace(requires_complete_frontier=True),
        packet=bounded,
        decision=supported,
        applicable=True,
    )
    assert assay._numeric_policy_is_admissible(
        spec=SimpleNamespace(requires_complete_frontier=False),
        packet=bounded,
        decision=supported,
        applicable=True,
    )


def test_run_and_replay_are_gold_blind_and_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_sha = "a" * 64
    population_sha = "b" * 64
    selection = {
        "bindings": {"population_identity_sha256": population_sha},
        "questions": [{"ordinal": 0}, {"ordinal": 1}],
    }
    monkeypatch.setattr(assay, "EXPECTED_V3_SELECTION_SHA256", selection_sha)
    monkeypatch.setattr(assay, "EXPECTED_POPULATION_SHA256", population_sha)
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 2)
    monkeypatch.setattr(
        assay.v3, "_load_selection", lambda _root: (selection, selection_sha)
    )
    monkeypatch.setattr(
        assay,
        "_diagnostic_row",
        lambda row, *, selection_sha256: _diagnostic(int(row["ordinal"])),
    )

    def forbidden_gold_loader(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("construction/replay opened benchmark gold")

    monkeypatch.setattr(assay.full100, "_load_population", forbidden_gold_loader)
    output_root = tmp_path / "assay"
    construction_sha = assay.run(v3_root=tmp_path, output_root=output_root)
    replay_sha = assay.replay(v3_root=tmp_path, output_root=output_root)
    construction, loaded_sha = assay._load_construction(output_root)
    replay, loaded_replay_sha = assay._load_replay(
        output_root, construction_sha256=construction_sha
    )
    assert loaded_sha == construction_sha
    assert loaded_replay_sha == replay_sha
    assert construction["gold_fields_present"] is False
    assert construction["implementation"]["sha256"] == (
        assay._implementation_identity()["sha256"]
    )
    assert construction["provider_calls"] == 0
    assert frozenset(construction) == assay._CONSTRUCTION_KEYS
    assert replay["byte_identical"] is True
    assert replay["gold_fields_present"] is False

    extra_root = tmp_path / "extra-field"
    altered = dict(construction)
    altered["self_signed_extra"] = True
    assay.hot._atomic_write_json(  # noqa: SLF001
        extra_root / assay.CONSTRUCTION_NAME, altered
    )
    with pytest.raises(ValueError, match="construction is invalid"):
        assay._load_construction(extra_root)
