"""Provider-free local-to-global connectivity audit for typed memory facts.

Retrieval alone is not counted as recovery.  A locally retrieved item becomes
globally connected only when its opaque source group, semantic action/role,
temporal and discourse coordinates (when present), and exact provenance
binding all survive into the fitted provider/operator packet.  The ledger is
prompt-external and contains no question IDs, raw source IDs, or benchmark
targets; mechanism-local audits can join through binding/locator receipts.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .typed_memory_final_arm import (
    FittedTypedFinalPrompt,
    compact_typed_evidence_projection,
)
from .typed_operator_adapter import TypedEvidenceContribution, TypedEvidenceItem
from .typed_operator_spec import normalized_terms


FORMAT = "memory-condense-typed-local-global-connectivity-ledger-v1"
_ACTION_TERMS = frozenset(
    {
        "assist",
        "buy",
        "cancel",
        "clean",
        "donate",
        "join",
        "pick",
        "pickup",
        "receive",
        "return",
        "service",
        "spend",
        "visit",
    }
)


def _require(ok: object, message: str) -> None:
    if not ok:
        raise MatchedEvalContractError(message)


def _action_role(item: TypedEvidenceItem) -> tuple[str, ...]:
    terms = set(normalized_terms(item.summary))
    actions = {term for term in terms if term in _ACTION_TERMS}
    if {"pick", "up"} <= terms:
        actions.add("pickup")
    if {"dry", "clean"} <= terms:
        actions.add("dry_clean")
    if item.relation:
        actions.update(f"relation:{term}" for term in normalized_terms(item.relation))
    return tuple(sorted(actions))


def _story_link_ids_by_group(story: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {}
    for key in ("link_overlays", "group_links"):
        raw = story.get(key, [])
        _require(type(raw) is list, "typed connectivity story links changed type")
        for index, row in enumerate(raw):
            _require(type(row) is dict, "typed connectivity story link changed type")
            groups = row.get("group_handles")
            if groups is None:
                groups = [row.get("left_group"), row.get("right_group")]
            _require(
                type(groups) is list
                and all(type(group) is str and group for group in groups),
                "typed connectivity story groups changed",
            )
            link_id = row.get("link_id")
            if type(link_id) is not str or not link_id:
                link_id = f"{key}:{index + 1}"
            for group in groups:
                result.setdefault(group, []).append(link_id)
    return {
        group: tuple(dict.fromkeys(link_ids))
        for group, link_ids in result.items()
    }


def build_typed_connectivity_ledger(
    original_contributions: tuple[TypedEvidenceContribution, ...],
    fitted: FittedTypedFinalPrompt,
    *,
    post_selection_dedup_exclusions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Audit retrieved-local, globally-bound and operator-consumed stages."""

    _require(
        type(original_contributions) is tuple
        and bool(original_contributions)
        and all(type(row) is TypedEvidenceContribution for row in original_contributions),
        "typed connectivity contributions must be a nonempty exact tuple",
    )
    if type(fitted) is not FittedTypedFinalPrompt:
        raise TypeError("fitted prompt must be exact")
    exclusions = tuple(post_selection_dedup_exclusions)
    _require(
        all(isinstance(row, Mapping) for row in exclusions),
        "typed connectivity dedup exclusions changed type",
    )
    dedup_by_binding_receipt: dict[str, Mapping[str, Any]] = {}
    for exclusion in exclusions:
        duplicate_receipts = exclusion.get(
            "duplicate_binding_receipt_sha256s", []
        )
        _require(
            type(duplicate_receipts) is list
            and bool(duplicate_receipts)
            and all(type(value) is str for value in duplicate_receipts),
            "typed connectivity dedup binding receipts changed",
        )
        for receipt in duplicate_receipts:
            require_sha256(receipt, "typed connectivity dedup binding")
            _require(
                receipt not in dedup_by_binding_receipt,
                "typed connectivity binding was deduped more than once",
            )
            dedup_by_binding_receipt[receipt] = exclusion
    mechanisms = tuple(row.mechanism_id for row in original_contributions)
    _require(
        len(set(mechanisms)) == len(mechanisms),
        "typed connectivity mechanisms repeat",
    )
    original_binding_by_handle = {
        binding.handle_id: (contribution.mechanism_id, binding)
        for contribution in original_contributions
        for binding in contribution.bindings
    }
    _require(
        len(original_binding_by_handle)
        == sum(len(row.bindings) for row in original_contributions),
        "typed connectivity original handles collide",
    )
    final_binding_by_handle = {
        binding.handle_id: binding for binding in fitted.packet.local_bindings
    }
    final_item_by_receipt = {
        item.receipt_sha256: item for item in fitted.packet.items
    }
    allowed = set(fitted.allowed_handle_ids)
    contract_raw = fitted.validation_contract.get("by_handle", {})
    _require(type(contract_raw) is dict, "typed connectivity contract changed")
    contract_by_handle: dict[str, Mapping[str, Any]] = {
        handle: row
        for handle, row in contract_raw.items()
        if type(handle) is str and isinstance(row, Mapping)
    }
    compact = compact_typed_evidence_projection(fitted.packet)
    compact_handle_by_id = {
        row["handle_id"]: row for row in compact["handles"]
    }
    compact_items = tuple(compact["items"])
    _require(
        len(compact_items) == len(fitted.packet.items),
        "typed connectivity compact item order changed",
    )
    compact_item_by_receipt = {
        item.receipt_sha256: projection
        for item, projection in zip(
            fitted.packet.items,
            compact_items,
            strict=True,
        )
    }
    links_by_group = _story_link_ids_by_group(fitted.story_coherence)
    advisory_handles: set[str] = set()
    for key in (
        "deterministic_execution_advisory",
        "scalar_validation_advisory",
    ):
        advisory = fitted.validation_contract.get(key)
        if isinstance(advisory, Mapping):
            raw_handles = advisory.get("used_handle_ids", [])
            _require(
                type(raw_handles) is list
                and all(type(handle) is str for handle in raw_handles),
                "typed connectivity advisory handles changed",
            )
            advisory_handles.update(raw_handles)

    rows: list[dict[str, Any]] = []
    for contribution in original_contributions:
        binding_by_handle = {
            row.handle_id: row for row in contribution.bindings
        }
        for item in contribution.parsed.accepted_items:
            action_role = _action_role(item)
            temporal_required = item.date is not None
            discourse_required = bool(item.supported_slot_ids)
            for handle in item.handle_ids:
                _require(
                    handle in binding_by_handle,
                    "typed connectivity item escaped its local binding",
                )
                binding = binding_by_handle[handle]
                dedup_exclusion = dedup_by_binding_receipt.get(
                    binding.receipt_sha256
                )
                dedup_subsumed = dedup_exclusion is not None
                final_binding = final_binding_by_handle.get(handle)
                final_item = final_item_by_receipt.get(item.receipt_sha256)
                provider_handle = compact_handle_by_id.get(handle)
                provider_item = compact_item_by_receipt.get(item.receipt_sha256)
                item_survived = final_item is not None and provider_item is not None
                binding_survived = (
                    final_binding is not None
                    and final_binding.receipt_sha256 == binding.receipt_sha256
                )
                group_survived = bool(
                    provider_handle is not None
                    and provider_handle.get("group_handle")
                    == binding.source_group_handle
                )
                provenance_survived = bool(
                    provider_handle is not None
                    and provider_handle.get("origin") == binding.origin.value
                    and provider_handle.get("provenance_grade")
                    == binding.provenance_grade.value
                )
                action_role_survived = bool(
                    not action_role
                    or (
                        provider_item is not None
                        and _action_role(final_item) == action_role
                    )
                )
                temporal_survived = bool(
                    not temporal_required
                    or (
                        provider_item is not None
                        and provider_item.get("date") == item.date
                    )
                )
                contract = contract_by_handle.get(handle)
                contract_slots = (
                    ()
                    if contract is None
                    else tuple(contract.get("supported_slot_ids", []))
                )
                discourse_survived = bool(
                    not discourse_required
                    or set(item.supported_slot_ids) <= set(contract_slots)
                )
                globally_bound = bool(
                    item_survived
                    and binding_survived
                    and group_survived
                    and provenance_survived
                    and action_role_survived
                    and temporal_survived
                    and discourse_survived
                )
                contract_item_receipts = (
                    ()
                    if contract is None
                    else tuple(contract.get("usable_item_receipt_sha256s", []))
                )
                contract_bound = bool(
                    contract is not None
                    and item.receipt_sha256 in contract_item_receipts
                )
                operator_consumed = bool(
                    globally_bound and handle in allowed and contract_bound
                )
                failures = (
                    ("post_selection_dedup_subsumed",)
                    if dedup_subsumed
                    else tuple(
                        name
                        for name, passed in (
                            ("item", item_survived),
                            ("provenance_binding", binding_survived),
                            ("source_group", group_survived),
                            ("provider_provenance", provenance_survived),
                            ("action_role", action_role_survived),
                            ("temporal", temporal_survived),
                            ("discourse", discourse_survived),
                            ("allowed_handle", handle in allowed),
                            ("validation_contract", contract_bound),
                        )
                        if not passed
                    )
                )
                body = {
                    "action_role_terms": list(action_role),
                    "action_role_survived": action_role_survived,
                    "advisory_consumed": handle in advisory_handles,
                    "binding_receipt_sha256": binding.receipt_sha256,
                    "disconnection_stages": list(failures),
                    "discourse_slot_ids": list(item.supported_slot_ids),
                    "discourse_survived": discourse_survived,
                    "globally_bound": globally_bound,
                    "handle_id": handle,
                    "item_kind": item.kind.value,
                    "item_receipt_sha256": item.receipt_sha256,
                    "local_source_locator_sha256": (
                        binding.local_source_locator_sha256
                    ),
                    "mechanism_id": contribution.mechanism_id,
                    "operator_consumed": operator_consumed,
                    "post_selection_dedup_owner_item_receipt_sha256": (
                        None
                        if dedup_exclusion is None
                        else dedup_exclusion.get(
                            "owner_item_receipt_sha256"
                        )
                    ),
                    "post_selection_dedup_owner_mechanism_id": (
                        None
                        if dedup_exclusion is None
                        else dedup_exclusion.get("owner_mechanism_id")
                    ),
                    "post_selection_dedup_subsumed": dedup_subsumed,
                    "provenance_grade": binding.provenance_grade.value,
                    "provenance_survived": provenance_survived,
                    "retrieved_local": True,
                    "sealed_artifact_sha256": binding.sealed_artifact_sha256,
                    "source_group_handle": binding.source_group_handle,
                    "source_group_survived": group_survived,
                    "story_link_ids": list(
                        links_by_group.get(binding.source_group_handle, ())
                    ),
                    "temporal_required": temporal_required,
                    "temporal_survived": temporal_survived,
                }
                rows.append({**body, "row_receipt_sha256": identity_sha256(body)})

    failure_counts: dict[str, int] = {}
    for row in rows:
        for reason in row["disconnection_stages"]:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    payload: dict[str, Any] = {
        "failure_count_by_stage": failure_counts,
        "fitted_prompt_receipt_sha256": fitted.receipt_sha256,
        "format": FORMAT,
        "globally_bound_count": sum(row["globally_bound"] for row in rows),
        "gold_loaded": False,
        "operator_consumed_count": sum(row["operator_consumed"] for row in rows),
        "post_selection_dedup_subsumed_count": sum(
            row["post_selection_dedup_subsumed"] for row in rows
        ),
        "provider_prompt_count": 0,
        "retained_transformer_token_state_bytes": 0,
        "retrieved_local_count": len(rows),
        "rows": rows,
    }
    payload["receipt_sha256"] = identity_sha256(payload)
    require_sha256(payload["receipt_sha256"], "typed connectivity receipt")
    assert_gold_blind(payload, path="typed_connectivity_ledger")
    return payload


__all__ = ["FORMAT", "build_typed_connectivity_ledger"]
