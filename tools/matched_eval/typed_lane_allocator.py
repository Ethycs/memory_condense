"""Non-borrowable packing for typed-memory mechanism lanes.

The generic typed-evidence union is intentionally order preserving.  That is
useful for a single mechanism, but it means a large earlier contribution can
consume the packet cap before a later specialist is considered.  This module
adds the prompt-tick boundary required by the cumulative architecture: every
mechanism is assigned to one named lane with its own guaranteed final-content
allowance.  The base allocator remains strictly non-borrowable.  An optional
second phase can preserve every item selected by those lane guarantees and fill
otherwise unused shared capacity from omitted items.  Keeping the phases
separate makes the guarantee and borrowing ledger independently auditable.

The allocator is provider-free and does not inspect question IDs, references,
answers, source locators, or judge outcomes.  It only sees already selected,
typed contributions and their opaque handles.  Exact dropped-item and
dropped-binding receipts remain in the local allocation ledger.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .prompt_tick_contracts import LaneBudget
from .typed_memory_final_arm import compact_evidence_content_projection
from .typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceStatus,
    ParsedTypedItems,
    TypedEvidenceContribution,
    TypedEvidenceItem,
)
from .typed_operator_spec import TypedOperatorSpec, normalized_terms


FORMAT = "memory-condense-typed-lane-allocation-v1"
SURPLUS_FORMAT = "memory-condense-typed-lane-surplus-fill-v1"


def _require(ok: object, message: str) -> None:
    if not ok:
        raise MatchedEvalContractError(message)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _usable_item(
    item: TypedEvidenceItem,
    operator_spec: TypedOperatorSpec | None,
) -> bool:
    return bool(
        item.included
        and not item.content_conflict
        and item.status is not EvidenceStatus.CANCELLED
        and (
            item.status is not EvidenceStatus.PROPOSED
            or operator_spec is None
            or operator_spec.include_proposed
        )
    )


def lane_content_token_proxy(
    items: tuple[TypedEvidenceItem, ...],
    bindings: tuple[EvidenceHandleBinding, ...],
) -> int:
    """Count exactly the opaque handle and typed-item content owned by a lane."""

    if type(items) is not tuple or any(type(row) is not TypedEvidenceItem for row in items):
        raise TypeError("lane items must be an exact typed-item tuple")
    if type(bindings) is not tuple or any(
        type(row) is not EvidenceHandleBinding for row in bindings
    ):
        raise TypeError("lane bindings must be an exact evidence-binding tuple")
    if not items:
        return 0
    projection = compact_evidence_content_projection(items, bindings)
    return count_tokens(_canonical_json(projection))


@dataclass(frozen=True, slots=True)
class TypedLaneAllocationReceipt:
    lane_id: str
    mechanism_ids: tuple[str, ...]
    final_content_token_cap: int
    selected_item_receipt_sha256s: tuple[str, ...]
    selected_binding_receipt_sha256s: tuple[str, ...]
    omitted_item_receipt_sha256s: tuple[str, ...]
    omitted_binding_receipt_sha256s: tuple[str, ...]
    final_content_token_proxy: int
    local_selection_priority_receipt_sha256: str
    non_borrowable: Literal[True] = True
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.lane_id, "typed allocation lane")
        _require(
            type(self.mechanism_ids) is tuple
            and bool(self.mechanism_ids)
            and all(type(value) is str and value for value in self.mechanism_ids)
            and len(set(self.mechanism_ids)) == len(self.mechanism_ids),
            "typed allocation mechanisms must be ordered unique text",
        )
        for values, label in (
            (self.selected_item_receipt_sha256s, "selected item receipts"),
            (self.selected_binding_receipt_sha256s, "selected binding receipts"),
            (self.omitted_item_receipt_sha256s, "omitted item receipts"),
            (self.omitted_binding_receipt_sha256s, "omitted binding receipts"),
        ):
            _require(type(values) is tuple and len(set(values)) == len(values), label)
            for value in values:
                require_sha256(value, label)
        require_sha256(
            self.local_selection_priority_receipt_sha256,
            "typed lane local selection-priority receipt",
        )
        _require(
            not (
                set(self.selected_item_receipt_sha256s)
                & set(self.omitted_item_receipt_sha256s)
            )
            and not (
                set(self.selected_binding_receipt_sha256s)
                & set(self.omitted_binding_receipt_sha256s)
            ),
            "typed lane selected and omitted partitions overlap",
        )
        _require(
            type(self.final_content_token_cap) is int
            and self.final_content_token_cap >= 0
            and type(self.final_content_token_proxy) is int
            and 0 <= self.final_content_token_proxy <= self.final_content_token_cap,
            "typed lane escaped its final-content allowance",
        )
        _require(
            self.non_borrowable is True
            and self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "typed lane allocation must be non-borrowable/provider-free/zero-state",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "typed lane receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="typed_lane_allocation")

    @property
    def unspent_content_tokens(self) -> int:
        return self.final_content_token_cap - self.final_content_token_proxy

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "final_content_token_cap": self.final_content_token_cap,
            "final_content_token_proxy": self.final_content_token_proxy,
            "format": FORMAT,
            "gold_loaded": False,
            "lane_id": self.lane_id,
            "local_selection_priority_receipt_sha256": (
                self.local_selection_priority_receipt_sha256
            ),
            "mechanism_ids": list(self.mechanism_ids),
            "non_borrowable": True,
            "omitted_binding_receipt_sha256s": list(
                self.omitted_binding_receipt_sha256s
            ),
            "omitted_item_receipt_sha256s": list(
                self.omitted_item_receipt_sha256s
            ),
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "selected_binding_receipt_sha256s": list(
                self.selected_binding_receipt_sha256s
            ),
            "selected_item_receipt_sha256s": list(
                self.selected_item_receipt_sha256s
            ),
            "unspent_content_tokens": self.unspent_content_tokens,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class TypedLaneAllocation:
    contributions: tuple[TypedEvidenceContribution, ...]
    receipts: tuple[TypedLaneAllocationReceipt, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.contributions) is tuple
            and bool(self.contributions)
            and all(type(row) is TypedEvidenceContribution for row in self.contributions),
            "typed allocation contributions changed type",
        )
        _require(
            type(self.receipts) is tuple
            and bool(self.receipts)
            and all(type(row) is TypedLaneAllocationReceipt for row in self.receipts),
            "typed allocation receipts changed type",
        )
        mechanisms = tuple(row.mechanism_id for row in self.contributions)
        receipt_mechanisms = tuple(
            mechanism for row in self.receipts for mechanism in row.mechanism_ids
        )
        _require(
            len(set(mechanisms)) == len(mechanisms)
            and set(mechanisms) == set(receipt_mechanisms)
            and len(receipt_mechanisms) == len(mechanisms),
            "typed allocation lost or repeated a mechanism",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "typed allocation changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="typed_lane_allocation_result")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "contribution_receipt_sha256s": [
                row.receipt_sha256 for row in self.contributions
            ],
            "format": f"{FORMAT}-result",
            "gold_loaded": False,
            "lane_receipts": [row.projection() for row in self.receipts],
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _round_robin_items(
    contributions: tuple[TypedEvidenceContribution, ...],
    *,
    operator_spec: TypedOperatorSpec | None,
    local_selection_priority_by_handle: Mapping[str, tuple[int, ...]],
) -> tuple[tuple[TypedEvidenceContribution, TypedEvidenceItem], ...]:
    ordinal: dict[str, int] = {}
    owner: dict[str, TypedEvidenceContribution] = {}
    group_by_handle = {
        binding.handle_id: binding.source_group_handle
        for contribution in contributions
        for binding in contribution.bindings
    }
    flattened: list[TypedEvidenceItem] = []
    for contribution in contributions:
        for item in contribution.parsed.accepted_items:
            ordinal[item.receipt_sha256] = len(flattened)
            owner[item.receipt_sha256] = contribution
            flattened.append(item)

    priority_width = max(
        (len(value) for value in local_selection_priority_by_handle.values()),
        default=0,
    )

    def local_priority(item: TypedEvidenceItem) -> tuple[int, ...]:
        values = tuple(
            local_selection_priority_by_handle.get(
                handle,
                (0,) * priority_width,
            )
            for handle in item.handle_ids
        )
        return max(values, default=(0,) * priority_width)

    def strength(
        item: TypedEvidenceItem,
        *,
        represented_groups: set[str],
    ) -> tuple[Any, ...]:
        terms = set(normalized_terms(item.summary))
        obligation_matches = (
            0
            if operator_spec is None
            else sum(
                term in terms
                for slot in operator_spec.required_slots
                for term in slot.match_terms
            )
        )
        groups = {group_by_handle[row] for row in item.handle_ids}
        return (
            int(bool(groups - represented_groups)),
            int(_usable_item(item, operator_spec)),
            *local_priority(item),
            len(item.supported_slot_ids),
            obligation_matches,
            int(item.date is not None),
            int(item.entity_key is not None or item.group_key is not None),
            -ordinal[item.receipt_sha256],
            item.receipt_sha256,
        )

    result: list[tuple[TypedEvidenceContribution, TypedEvidenceItem]] = []
    represented_groups: set[str] = set()
    remaining = list(flattened)
    # Preserve one strongest candidate per nonempty mechanism before any lane
    # fill, then favor new source groups and question-derived slot gain.
    for contribution in contributions:
        candidates = [
            item
            for item in remaining
            if owner[item.receipt_sha256] is contribution
        ]
        if not candidates:
            continue
        selected = max(
            candidates,
            key=lambda item: strength(item, represented_groups=represented_groups),
        )
        result.append((contribution, selected))
        remaining.remove(selected)
        represented_groups.update(group_by_handle[row] for row in selected.handle_ids)
    while remaining:
        selected = max(
            remaining,
            key=lambda item: strength(item, represented_groups=represented_groups),
        )
        contribution = owner[selected.receipt_sha256]
        result.append((contribution, selected))
        remaining.remove(selected)
        represented_groups.update(group_by_handle[row] for row in selected.handle_ids)
    return tuple(result)


def allocate_typed_contribution_lanes(
    contributions: tuple[TypedEvidenceContribution, ...],
    *,
    lane_budgets: tuple[LaneBudget, ...],
    lane_by_mechanism: Mapping[str, str],
    operator_spec: TypedOperatorSpec | None = None,
    local_selection_priority_by_handle: Mapping[str, Sequence[int]] | None = None,
) -> TypedLaneAllocation:
    """Pack typed contributions inside independent, non-borrowable lanes."""

    if (
        type(contributions) is not tuple
        or not contributions
        or any(type(row) is not TypedEvidenceContribution for row in contributions)
    ):
        raise TypeError("contributions must be a non-empty exact tuple")
    if type(lane_budgets) is not tuple or not lane_budgets or any(
        type(row) is not LaneBudget for row in lane_budgets
    ):
        raise TypeError("lane_budgets must be a non-empty exact tuple")
    if operator_spec is not None and type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be exact when supplied")
    raw_priority = dict(local_selection_priority_by_handle or {})
    priorities: dict[str, tuple[int, ...]] = {}
    priority_width: int | None = None
    for handle, value in raw_priority.items():
        require_text(handle, "typed lane local-priority handle")
        _require(
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and bool(value)
            and all(type(part) is int for part in value),
            "typed lane local priority must be a nonempty integer sequence",
        )
        normalized = tuple(value)
        if priority_width is None:
            priority_width = len(normalized)
        _require(
            len(normalized) == priority_width,
            "typed lane local priorities changed width",
        )
        priorities[handle] = normalized
    mechanisms = tuple(row.mechanism_id for row in contributions)
    _require(
        len(set(mechanisms)) == len(mechanisms),
        "typed allocation contribution mechanisms repeat",
    )
    budgets = {row.lane_id: row for row in lane_budgets}
    _require(
        len(budgets) == len(lane_budgets), "typed allocation lane budgets repeat"
    )
    mapping = dict(lane_by_mechanism)
    _require(
        set(mapping) == set(mechanisms)
        and all(type(value) is str and value in budgets for value in mapping.values()),
        "typed allocation must map every mechanism to one declared lane",
    )
    unused_lanes = set(budgets) - set(mapping.values())
    _require(not unused_lanes, "typed allocation declared an unused lane budget")
    all_handle_ids = {
        binding.handle_id
        for contribution in contributions
        for binding in contribution.bindings
    }
    _require(
        set(priorities) <= all_handle_ids,
        "typed lane local priority names an unknown handle",
    )

    selected_by_mechanism: dict[str, list[TypedEvidenceItem]] = {
        mechanism: [] for mechanism in mechanisms
    }
    receipts: list[TypedLaneAllocationReceipt] = []
    for lane_budget in lane_budgets:
        lane_contributions = tuple(
            row for row in contributions if mapping[row.mechanism_id] == lane_budget.lane_id
        )
        all_bindings = tuple(
            binding for row in lane_contributions for binding in row.bindings
        )
        accepted: list[TypedEvidenceItem] = []
        omitted: list[TypedEvidenceItem] = []
        for contribution, item in _round_robin_items(
            lane_contributions,
            operator_spec=operator_spec,
            local_selection_priority_by_handle=priorities,
        ):
            if not _usable_item(item, operator_spec):
                omitted.append(item)
                continue
            trial = tuple((*accepted, item))
            if lane_content_token_proxy(trial, all_bindings) <= lane_budget.final_content_token_cap:
                accepted.append(item)
                selected_by_mechanism[contribution.mechanism_id].append(item)
            else:
                omitted.append(item)
        used_handles = {handle for item in accepted for handle in item.handle_ids}
        selected_bindings = tuple(
            row for row in all_bindings if row.handle_id in used_handles
        )
        omitted_bindings = tuple(
            row for row in all_bindings if row.handle_id not in used_handles
        )
        receipts.append(
            TypedLaneAllocationReceipt(
                lane_budget.lane_id,
                tuple(row.mechanism_id for row in lane_contributions),
                lane_budget.final_content_token_cap,
                tuple(row.receipt_sha256 for row in accepted),
                tuple(row.receipt_sha256 for row in selected_bindings),
                tuple(row.receipt_sha256 for row in omitted),
                tuple(row.receipt_sha256 for row in omitted_bindings),
                lane_content_token_proxy(tuple(accepted), all_bindings),
                identity_sha256(
                    {
                        "format": f"{FORMAT}-local-selection-priority-v1",
                        "lane_id": lane_budget.lane_id,
                        "priority_by_handle": {
                            handle: list(priorities[handle])
                            for handle in sorted(priorities)
                            if handle in {row.handle_id for row in all_bindings}
                        },
                    }
                ),
            )
        )

    rebuilt: list[TypedEvidenceContribution] = []
    for contribution in contributions:
        selected = tuple(selected_by_mechanism[contribution.mechanism_id])
        used_handles = {handle for item in selected for handle in item.handle_ids}
        bindings = tuple(
            row for row in contribution.bindings if row.handle_id in used_handles
        )
        parsed = ParsedTypedItems(
            selected,
            contribution.parsed.rejected_items,
            identity_sha256(
                {
                    "format": f"{FORMAT}-parse-subset",
                    "lane_id": mapping[contribution.mechanism_id],
                    "mechanism_id": contribution.mechanism_id,
                    "original_parse_receipt_sha256": (
                        contribution.parsed.parse_receipt_sha256
                    ),
                    "selected_item_receipt_sha256s": [
                        row.receipt_sha256 for row in selected
                    ],
                }
            ),
        )
        rebuilt.append(
            TypedEvidenceContribution(
                contribution.mechanism_id,
                bindings,
                parsed,
                contribution.sealed_artifact_sha256,
                contribution.frontier_mode,
                contribution.truncated
                or len(selected) < len(contribution.parsed.accepted_items),
            )
        )
    return TypedLaneAllocation(tuple(rebuilt), tuple(receipts))


def fill_typed_lane_surplus(
    original_contributions: tuple[TypedEvidenceContribution, ...],
    minimum_allocation: TypedLaneAllocation,
    *,
    lane_budgets: tuple[LaneBudget, ...],
    lane_by_mechanism: Mapping[str, str],
    operator_spec: TypedOperatorSpec | None = None,
    local_selection_priority_by_handle: Mapping[str, Sequence[int]] | None = None,
) -> tuple[tuple[TypedEvidenceContribution, ...], dict[str, Any]]:
    """Preserve lane minima, then spend only their aggregate unused capacity.

    The first-phase selections are immutable protected minima. Omitted usable
    items compete provider-free for a shared cap equal to the sum of the active
    lane caps. A lane may exceed its own first-phase cap only when another lane
    donates at least the same amount of unused capacity. Original typed items
    and bindings are reused byte-for-byte, and the audit binds every addition
    and remaining omission.
    """

    if (
        type(original_contributions) is not tuple
        or not original_contributions
        or any(
            type(row) is not TypedEvidenceContribution
            for row in original_contributions
        )
    ):
        raise TypeError("original contributions must be a non-empty exact tuple")
    if type(minimum_allocation) is not TypedLaneAllocation:
        raise TypeError("minimum_allocation must be an exact TypedLaneAllocation")
    minimum_contributions = minimum_allocation.contributions
    if type(lane_budgets) is not tuple or not lane_budgets or any(
        type(row) is not LaneBudget for row in lane_budgets
    ):
        raise TypeError("lane_budgets must be a non-empty exact tuple")
    if operator_spec is not None and type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be exact when supplied")
    allocation_audit = minimum_allocation.projection()
    minimum_allocation_receipt_sha256 = minimum_allocation.receipt_sha256

    original_mechanisms = tuple(row.mechanism_id for row in original_contributions)
    minimum_mechanisms = tuple(row.mechanism_id for row in minimum_contributions)
    _require(
        len(set(original_mechanisms)) == len(original_mechanisms)
        and minimum_mechanisms == original_mechanisms,
        "surplus fill mechanisms changed order, coverage, or uniqueness",
    )
    budgets = {row.lane_id: row for row in lane_budgets}
    _require(
        len(budgets) == len(lane_budgets),
        "surplus fill lane budgets repeat",
    )
    mapping = dict(lane_by_mechanism)
    _require(
        set(mapping) == set(original_mechanisms)
        and set(mapping.values()) == set(budgets)
        and all(type(value) is str for value in mapping.values()),
        "surplus fill must map every mechanism to one active lane",
    )

    raw_priorities = dict(local_selection_priority_by_handle or {})
    priorities: dict[str, tuple[int, ...]] = {}
    priority_width: int | None = None
    for handle, value in raw_priorities.items():
        require_text(handle, "surplus local-priority handle")
        _require(
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and bool(value)
            and all(type(part) is int for part in value),
            "surplus local priority must be a nonempty integer sequence",
        )
        normalized = tuple(value)
        if priority_width is None:
            priority_width = len(normalized)
        _require(
            len(normalized) == priority_width,
            "surplus local priorities changed width",
        )
        priorities[handle] = normalized
    zero_priority = (0,) * (priority_width or 0)

    original_by_mechanism = {
        row.mechanism_id: row for row in original_contributions
    }
    minimum_by_mechanism = {
        row.mechanism_id: row for row in minimum_contributions
    }
    item_by_receipt: dict[str, TypedEvidenceItem] = {}
    owner_by_receipt: dict[str, str] = {}
    ordinal_by_receipt: dict[str, int] = {}
    binding_by_handle: dict[str, EvidenceHandleBinding] = {}
    group_by_handle: dict[str, str] = {}
    for contribution in original_contributions:
        for binding in contribution.bindings:
            _require(
                binding.handle_id not in binding_by_handle,
                "surplus fill original handles collide",
            )
            binding_by_handle[binding.handle_id] = binding
            group_by_handle[binding.handle_id] = binding.source_group_handle
        for item in contribution.parsed.accepted_items:
            _require(
                item.receipt_sha256 not in item_by_receipt,
                "surplus fill original item receipts collide",
            )
            ordinal_by_receipt[item.receipt_sha256] = len(item_by_receipt)
            item_by_receipt[item.receipt_sha256] = item
            owner_by_receipt[item.receipt_sha256] = contribution.mechanism_id
    _require(
        set(priorities) <= set(binding_by_handle),
        "surplus local priority names an unknown handle",
    )

    selected_by_mechanism: dict[str, list[TypedEvidenceItem]] = {}
    minimum_receipts: set[str] = set()
    for mechanism in original_mechanisms:
        original = original_by_mechanism[mechanism]
        minimum = minimum_by_mechanism[mechanism]
        original_items = {
            row.receipt_sha256: row for row in original.parsed.accepted_items
        }
        selected = list(minimum.parsed.accepted_items)
        _require(
            all(
                item.receipt_sha256 in original_items
                and item == original_items[item.receipt_sha256]
                for item in selected
            ),
            "surplus minimum item escaped or changed its original contribution",
        )
        _require(
            all(_usable_item(item, operator_spec) for item in selected),
            "surplus minimum item is not usable under the operator policy",
        )
        represented = {handle for item in selected for handle in item.handle_ids}
        expected_bindings = tuple(
            row for row in original.bindings if row.handle_id in represented
        )
        _require(
            minimum.bindings == expected_bindings,
            "surplus minimum bindings changed original provenance or order",
        )
        selected_by_mechanism[mechanism] = selected
        minimum_receipts.update(item.receipt_sha256 for item in selected)

    _require(
        allocation_audit.get("contribution_receipt_sha256s")
        == [row.receipt_sha256 for row in minimum_contributions],
        "minimum lane allocation audit does not bind supplied contributions",
    )
    raw_lane_receipts = allocation_audit.get("lane_receipts")
    _require(
        type(raw_lane_receipts) is list
        and len(raw_lane_receipts) == len(lane_budgets)
        and all(type(row) is dict for row in raw_lane_receipts),
        "minimum lane allocation audit changed lane coverage",
    )
    ordered_minimum_item_receipts: list[str] = []
    ordered_minimum_binding_receipts: list[str] = []
    for budget, lane_audit in zip(
        lane_budgets,
        raw_lane_receipts,
        strict=True,
    ):
        declared_lane_receipt = require_sha256(
            lane_audit.get("receipt_sha256"),
            "minimum lane receipt",
        )
        unsigned_lane_audit = dict(lane_audit)
        unsigned_lane_audit.pop("receipt_sha256")
        _require(
            identity_sha256(unsigned_lane_audit) == declared_lane_receipt,
            "minimum lane receipt changed",
        )
        lane_mechanisms = tuple(
            mechanism
            for mechanism in original_mechanisms
            if mapping[mechanism] == budget.lane_id
        )
        expected_item_receipts = {
            item.receipt_sha256
            for mechanism in lane_mechanisms
            for item in minimum_by_mechanism[
                mechanism
            ].parsed.accepted_items
        }
        expected_binding_receipts = {
            binding.receipt_sha256
            for mechanism in lane_mechanisms
            for binding in minimum_by_mechanism[mechanism].bindings
        }
        selected_items = lane_audit.get("selected_item_receipt_sha256s")
        selected_bindings = lane_audit.get(
            "selected_binding_receipt_sha256s"
        )
        all_lane_items = tuple(
            item
            for mechanism in lane_mechanisms
            for item in original_by_mechanism[
                mechanism
            ].parsed.accepted_items
        )
        all_lane_bindings = tuple(
            binding
            for mechanism in lane_mechanisms
            for binding in original_by_mechanism[mechanism].bindings
        )
        omitted_items = lane_audit.get("omitted_item_receipt_sha256s")
        omitted_bindings = lane_audit.get(
            "omitted_binding_receipt_sha256s"
        )
        expected_priority_receipt = identity_sha256(
            {
                "format": f"{FORMAT}-local-selection-priority-v1",
                "lane_id": budget.lane_id,
                "priority_by_handle": {
                    handle: list(priorities[handle])
                    for handle in sorted(priorities)
                    if handle
                    in {row.handle_id for row in all_lane_bindings}
                },
            }
        )
        _require(
            lane_audit.get("lane_id") == budget.lane_id
            and lane_audit.get("mechanism_ids") == list(lane_mechanisms)
            and lane_audit.get("final_content_token_cap")
            == budget.final_content_token_cap
            and type(selected_items) is list
            and len(selected_items) == len(set(selected_items))
            and set(selected_items) == expected_item_receipts
            and type(selected_bindings) is list
            and len(selected_bindings) == len(set(selected_bindings))
            and set(selected_bindings) == expected_binding_receipts
            and type(omitted_items) is list
            and len(omitted_items) == len(set(omitted_items))
            and set(omitted_items)
            == {
                item.receipt_sha256 for item in all_lane_items
            }
            - expected_item_receipts
            and type(omitted_bindings) is list
            and len(omitted_bindings) == len(set(omitted_bindings))
            and set(omitted_bindings)
            == {
                binding.receipt_sha256 for binding in all_lane_bindings
            }
            - expected_binding_receipts
            and lane_audit.get("final_content_token_proxy")
            == lane_content_token_proxy(
                tuple(item_by_receipt[receipt] for receipt in selected_items),
                all_lane_bindings,
            )
            and lane_audit.get(
                "local_selection_priority_receipt_sha256"
            )
            == expected_priority_receipt,
            "minimum lane allocation partitions or accounting changed",
        )
        for value in (
            *selected_items,
            *selected_bindings,
            *omitted_items,
            *omitted_bindings,
        ):
            require_sha256(value, "minimum lane selected receipt")
        ordered_minimum_item_receipts.extend(selected_items)
        ordered_minimum_binding_receipts.extend(selected_bindings)
    _require(
        len(ordered_minimum_item_receipts) == len(minimum_receipts)
        and set(ordered_minimum_item_receipts) == minimum_receipts,
        "minimum lane item receipt partition changed",
    )

    def lane_items(lane_id: str) -> tuple[TypedEvidenceItem, ...]:
        return tuple(
            item
            for mechanism in original_mechanisms
            if mapping[mechanism] == lane_id
            for item in selected_by_mechanism[mechanism]
        )

    bindings_by_lane = {
        lane_id: tuple(
            binding
            for contribution in original_contributions
            if mapping[contribution.mechanism_id] == lane_id
            for binding in contribution.bindings
        )
        for lane_id in budgets
    }
    proxy_by_lane = {
        lane_id: lane_content_token_proxy(
            lane_items(lane_id),
            bindings_by_lane[lane_id],
        )
        for lane_id in budgets
    }
    _require(
        all(
            proxy_by_lane[row.lane_id] <= row.final_content_token_cap
            for row in lane_budgets
        ),
        "surplus minimum contribution escaped its original lane cap",
    )
    shared_cap = sum(row.final_content_token_cap for row in lane_budgets)
    base_proxy = sum(proxy_by_lane.values())
    _require(
        base_proxy <= shared_cap,
        "surplus protected minima exceed their aggregate lane cap",
    )

    represented_groups = {
        group_by_handle[handle]
        for receipt in minimum_receipts
        for handle in item_by_receipt[receipt].handle_ids
    }
    remaining = [
        item
        for contribution in original_contributions
        for item in contribution.parsed.accepted_items
        if item.receipt_sha256 not in minimum_receipts
    ]

    def local_priority(item: TypedEvidenceItem) -> tuple[int, ...]:
        return max(
            (priorities.get(handle, zero_priority) for handle in item.handle_ids),
            default=zero_priority,
        )

    def strength(item: TypedEvidenceItem) -> tuple[Any, ...]:
        terms = set(normalized_terms(item.summary))
        obligation_matches = (
            0
            if operator_spec is None
            else sum(
                term in terms
                for slot in operator_spec.required_slots
                for term in slot.match_terms
            )
        )
        groups = {group_by_handle[handle] for handle in item.handle_ids}
        return (
            int(bool(groups - represented_groups)),
            int(_usable_item(item, operator_spec)),
            *local_priority(item),
            len(item.supported_slot_ids),
            obligation_matches,
            int(item.date is not None),
            int(item.entity_key is not None or item.group_key is not None),
            -ordinal_by_receipt[item.receipt_sha256],
            item.receipt_sha256,
        )

    added: list[TypedEvidenceItem] = []
    budget_omitted: list[TypedEvidenceItem] = []
    ineligible: list[TypedEvidenceItem] = []
    while remaining:
        item = max(remaining, key=strength)
        remaining.remove(item)
        if not _usable_item(item, operator_spec):
            ineligible.append(item)
            continue
        mechanism = owner_by_receipt[item.receipt_sha256]
        lane_id = mapping[mechanism]
        selected_by_mechanism[mechanism].append(item)
        trial_proxy = lane_content_token_proxy(
            lane_items(lane_id),
            bindings_by_lane[lane_id],
        )
        trial_total = sum(
            trial_proxy if key == lane_id else value
            for key, value in proxy_by_lane.items()
        )
        if trial_total <= shared_cap:
            proxy_by_lane[lane_id] = trial_proxy
            added.append(item)
            represented_groups.update(
                group_by_handle[handle] for handle in item.handle_ids
            )
        else:
            selected_by_mechanism[mechanism].pop()
            budget_omitted.append(item)

    final_receipts = {
        item.receipt_sha256
        for values in selected_by_mechanism.values()
        for item in values
    }
    _require(
        minimum_receipts <= final_receipts,
        "surplus fill dropped a protected lane minimum",
    )
    original_receipts = {
        item.receipt_sha256
        for contribution in original_contributions
        for item in contribution.parsed.accepted_items
    }
    _require(
        minimum_receipts.isdisjoint(
            item.receipt_sha256 for item in added
        )
        and minimum_receipts.isdisjoint(
            item.receipt_sha256 for item in budget_omitted
        )
        and minimum_receipts.isdisjoint(
            item.receipt_sha256 for item in ineligible
        )
        and not (
            {item.receipt_sha256 for item in added}
            & {item.receipt_sha256 for item in budget_omitted}
        )
        and not (
            {item.receipt_sha256 for item in added}
            & {item.receipt_sha256 for item in ineligible}
        )
        and not (
            {item.receipt_sha256 for item in budget_omitted}
            & {item.receipt_sha256 for item in ineligible}
        )
        and original_receipts
        == minimum_receipts
        | {item.receipt_sha256 for item in added}
        | {item.receipt_sha256 for item in budget_omitted}
        | {item.receipt_sha256 for item in ineligible},
        "surplus item lifecycle partition changed",
    )
    rebuilt: list[TypedEvidenceContribution] = []
    for contribution in original_contributions:
        selected = tuple(selected_by_mechanism[contribution.mechanism_id])
        represented = {handle for item in selected for handle in item.handle_ids}
        bindings = tuple(
            row for row in contribution.bindings if row.handle_id in represented
        )
        parsed = ParsedTypedItems(
            selected,
            contribution.parsed.rejected_items,
            identity_sha256(
                {
                    "added_item_receipt_sha256s": [
                        item.receipt_sha256
                        for item in added
                        if owner_by_receipt[item.receipt_sha256]
                        == contribution.mechanism_id
                    ],
                    "format": f"{SURPLUS_FORMAT}-parse-subset",
                    "mechanism_id": contribution.mechanism_id,
                    "minimum_allocation_receipt_sha256": (
                        minimum_allocation_receipt_sha256
                    ),
                    "original_parse_receipt_sha256": (
                        contribution.parsed.parse_receipt_sha256
                    ),
                    "selected_item_receipt_sha256s": [
                        item.receipt_sha256 for item in selected
                    ],
                }
            ),
        )
        rebuilt.append(
            TypedEvidenceContribution(
                contribution.mechanism_id,
                bindings,
                parsed,
                contribution.sealed_artifact_sha256,
                contribution.frontier_mode,
                contribution.truncated
                or len(selected) < len(contribution.parsed.accepted_items),
            )
        )

    final_proxy = sum(proxy_by_lane.values())
    final_handles = {
        handle
        for contribution in rebuilt
        for item in contribution.parsed.accepted_items
        for handle in item.handle_ids
    }
    minimum_handles = {
        handle
        for receipt in minimum_receipts
        for handle in item_by_receipt[receipt].handle_ids
    }
    priority_body = {
        "format": f"{SURPLUS_FORMAT}-local-selection-priority-v1",
        "priority_by_handle": {
            handle: list(priorities[handle]) for handle in sorted(priorities)
        },
    }
    audit: dict[str, Any] = {
        "added_binding_receipt_sha256s": [
            binding_by_handle[handle].receipt_sha256
            for handle in binding_by_handle
            if handle in final_handles and handle not in minimum_handles
        ],
        "added_item_receipt_sha256s": [item.receipt_sha256 for item in added],
        "base_content_token_proxy": base_proxy,
        "budget_omitted_item_receipt_sha256s": [
            item.receipt_sha256 for item in budget_omitted
        ],
        "contribution_receipt_sha256s": [row.receipt_sha256 for row in rebuilt],
        "final_content_token_proxy": final_proxy,
        "format": SURPLUS_FORMAT,
        "gold_loaded": False,
        "ineligible_item_receipt_sha256s": [
            item.receipt_sha256 for item in ineligible
        ],
        "lane_rows": [
            {
                "borrowed_content_tokens": max(
                    0,
                    proxy_by_lane[row.lane_id] - row.final_content_token_cap,
                ),
                "final_content_token_cap": row.final_content_token_cap,
                "final_content_token_proxy": proxy_by_lane[row.lane_id],
                "lane_id": row.lane_id,
                "remaining_donatable_content_tokens": max(
                    0,
                    row.final_content_token_cap - proxy_by_lane[row.lane_id],
                ),
            }
            for row in lane_budgets
        ],
        "local_selection_priority_receipt_sha256": identity_sha256(priority_body),
        "minimum_allocation_receipt_sha256": minimum_allocation_receipt_sha256,
        "minimum_binding_receipt_sha256s": ordered_minimum_binding_receipts,
        "minimum_item_receipt_sha256s": ordered_minimum_item_receipts,
        "original_contribution_receipt_sha256s": [
            row.receipt_sha256 for row in original_contributions
        ],
        "policy": (
            "preserve_all_non_borrowable_lane_minima_then_new_group_slot_"
            "and_local_priority_fill_within_sum_of_active_lane_caps"
        ),
        "provider_prompt_count": 0,
        "retained_transformer_token_state_bytes": 0,
        "shared_final_content_token_cap": shared_cap,
        "unspent_shared_content_tokens": shared_cap - final_proxy,
    }
    audit["receipt_sha256"] = identity_sha256(audit)
    assert_gold_blind(audit, path="typed_lane_surplus_fill")
    return tuple(rebuilt), audit


__all__ = [
    "FORMAT",
    "SURPLUS_FORMAT",
    "TypedLaneAllocation",
    "TypedLaneAllocationReceipt",
    "allocate_typed_contribution_lanes",
    "fill_typed_lane_surplus",
    "lane_content_token_proxy",
]
