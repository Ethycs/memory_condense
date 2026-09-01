from __future__ import annotations

import hashlib

from tools.matched_eval.typed_connectivity_ledger import (
    build_typed_connectivity_ledger,
)
from tools.matched_eval.typed_memory_final_arm import fit_typed_final_prompt
from tools.matched_eval.typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    ParsedTypedItems,
    ProvenanceGrade,
    TypedEvidenceContribution,
    merge_typed_evidence_contributions,
    parse_typed_items,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


QUESTION = (
    "[Question asked at 2023/05/30 (Tue) 16:15]\n"
    "Did I receive a higher percentage discount on my first order from "
    "HelloFresh, compared to my first UberEats order?"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contribution(
    mechanism: str,
    handle_number: int,
    summary: str,
    value: float,
) -> TypedEvidenceContribution:
    spec = compile_typed_operator_spec(QUESTION)
    handle = f"H{handle_number:03d}"
    group = f"G{handle_number:03d}"
    binding = EvidenceHandleBinding(
        handle,
        EvidenceOrigin.MAP,
        ProvenanceGrade.EXACT_CITATION,
        group,
        _sha(f"artifact-{mechanism}"),
        _sha(f"parent-{mechanism}"),
        _sha(f"evidence-{mechanism}"),
        _sha(summary),
        _sha(f"citation-{mechanism}"),
        len(summary),
        _sha(f"private-namespace-source-{mechanism}"),
    )
    parsed = parse_typed_items(
        [
            {
                "handle_ids": [handle],
                "kind": "operand",
                "numeric_role": "operand",
                "numeric_value": value,
                "status": "completed",
                "summary": summary,
                "unit": "%",
            }
        ],
        operator_spec=spec,
        bindings=(binding,),
    )
    assert not parsed.rejected_items
    return TypedEvidenceContribution(
        mechanism,
        (binding,),
        ParsedTypedItems(
            parsed.accepted_items,
            (),
            _sha(f"parse-{mechanism}"),
        ),
        _sha(f"artifact-{mechanism}"),
        FrontierMode.BOUNDED,
        False,
    )


def test_connectivity_ledger_separates_local_global_and_operator_stages() -> None:
    left = _contribution(
        "parent_map",
        1,
        "HelloFresh first-order discount was 40 percent.",
        40,
    )
    right = _contribution(
        "full_store",
        500_001,
        "UberEats first-order discount was 20 percent.",
        20,
    )
    locally_retrieved_but_dropped = _contribution(
        "tail_source",
        300_001,
        "HelloFresh repeated first-order discount was 40 percent.",
        40,
    )
    spec = compile_typed_operator_spec(QUESTION)
    packet = merge_typed_evidence_contributions(spec, (left, right))
    fitted = fit_typed_final_prompt(
        dated_question=QUESTION,
        parent_prediction="Yes",
        packet=packet,
        mechanism_by_handle={
            "H001": "parent_map",
            "H500001": "full_store",
        },
    )

    ledger = build_typed_connectivity_ledger(
        (left, right, locally_retrieved_but_dropped),
        fitted,
    )
    by_handle = {row["handle_id"]: row for row in ledger["rows"]}
    assert ledger["retrieved_local_count"] == 3
    assert ledger["globally_bound_count"] == 2
    assert ledger["operator_consumed_count"] == 2
    assert by_handle["H001"]["discourse_slot_ids"]
    assert by_handle["H001"]["source_group_survived"] is True
    assert by_handle["H001"]["provenance_survived"] is True
    assert by_handle["H001"]["operator_consumed"] is True
    dropped = by_handle["H300001"]
    assert dropped["retrieved_local"] is True
    assert dropped["globally_bound"] is False
    assert dropped["operator_consumed"] is False
    assert {
        "item",
        "provenance_binding",
        "source_group",
        "validation_contract",
    } <= set(dropped["disconnection_stages"])
    assert "private-namespace-source" not in repr(ledger)
    assert ledger == build_typed_connectivity_ledger(
        (left, right, locally_retrieved_but_dropped),
        fitted,
    )


def test_connectivity_ledger_labels_intentional_post_selection_dedup() -> None:
    summary = "HelloFresh first-order discount was 40 percent."
    loser = _contribution("full_store", 500_001, summary, 40)
    owner = _contribution("active_reconstruction", 600_001, summary, 40)
    spec = compile_typed_operator_spec(QUESTION)
    packet = merge_typed_evidence_contributions(spec, (owner,))
    fitted = fit_typed_final_prompt(
        dated_question=QUESTION,
        parent_prediction="Yes",
        packet=packet,
        mechanism_by_handle={"H600001": "active_reconstruction"},
    )
    loser_item = loser.parsed.accepted_items[0]
    owner_item = owner.parsed.accepted_items[0]
    exclusion = {
        "duplicate_binding_receipt_sha256s": [
            loser.bindings[0].receipt_sha256
        ],
        "duplicate_item_receipt_sha256": loser_item.receipt_sha256,
        "duplicate_mechanism_id": loser.mechanism_id,
        "owner_item_receipt_sha256": owner_item.receipt_sha256,
        "owner_mechanism_id": owner.mechanism_id,
    }
    ledger = build_typed_connectivity_ledger(
        (loser, owner),
        fitted,
        post_selection_dedup_exclusions=(exclusion,),
    )
    rows = {row["handle_id"]: row for row in ledger["rows"]}
    assert ledger["post_selection_dedup_subsumed_count"] == 1
    assert rows["H500001"]["post_selection_dedup_subsumed"] is True
    assert rows["H500001"]["disconnection_stages"] == [
        "post_selection_dedup_subsumed"
    ]
    assert rows["H500001"]["post_selection_dedup_owner_mechanism_id"] == (
        "active_reconstruction"
    )
    assert rows["H600001"]["globally_bound"] is True
