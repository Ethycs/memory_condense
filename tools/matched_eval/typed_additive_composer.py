"""Gold-blind additive composition for independently selected typed evidence.

The composer is the reusable boundary between retrieval mechanisms and the
common final prompt fitter.  Every mechanism selects before this module is
called.  Composition then performs, in order:

1. identity-proven cross-mechanism deduplication;
2. non-borrowable lane allocation;
3. shared-surplus fill without dropping any lane minimum; and
4. a fair compact packet merge that protects those exact minima.

No provider is called and no benchmark answer or verdict is accepted by the
API.  All decisions and lifecycle partitions are bound by SHA-256 receipts.
The returned packet, mechanism map, protected item receipts, and protection
receipt can be passed directly to ``fit_typed_final_prompt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .prompt_tick_contracts import LaneBudget
from .typed_lane_allocator import (
    TypedLaneAllocation,
    allocate_typed_contribution_lanes,
    fill_typed_lane_surplus,
)
from .typed_memory_final_arm import (
    LOCAL_RETENTION_PRIORITY_WIDTH,
    validate_disjoint_contribution_ranges,
)
from .typed_operator_adapter import (
    ConflictPolicy,
    ContentCoherence,
    EvidenceHandleBinding,
    EvidenceStatus,
    FrontierMode,
    ParsedTypedItems,
    ProviderPayloadMode,
    TypedEvidenceContribution,
    TypedEvidenceItem,
    TypedEvidencePacket,
    build_typed_evidence_packet,
)
from .typed_operator_spec import TypedOperatorSpec


FORMAT = "memory-condense-typed-additive-composer-v1"
DEDUP_FORMAT = f"{FORMAT}-post-selection-dedup-v1"
FAIR_MERGE_FORMAT = f"{FORMAT}-fair-premerge-v1"


def _require(ok: object, message: str) -> None:
    if not ok:
        raise MatchedEvalContractError(message)


def _sealed_audit(value: dict[str, Any], *, path: str) -> dict[str, Any]:
    _require("receipt_sha256" not in value, f"{path} audit was already sealed")
    assert_gold_blind(value, path=path)
    result = {**value, "receipt_sha256": identity_sha256(value)}
    assert_gold_blind(result, path=path)
    return result


def _verify_sealed_audit(value: Mapping[str, Any], *, label: str) -> str:
    _require(type(value) is dict, f"{label} must be an exact sealed dict")
    unsigned = dict(value)
    receipt = require_sha256(unsigned.pop("receipt_sha256", None), label)
    _require(identity_sha256(unsigned) == receipt, f"{label} receipt changed")
    assert_gold_blind(value, path=label)
    return receipt


def _usable_item(
    item: TypedEvidenceItem,
    operator_spec: TypedOperatorSpec,
) -> bool:
    return bool(
        item.included
        and not item.content_conflict
        and item.status is not EvidenceStatus.CANCELLED
        and (
            item.status is not EvidenceStatus.PROPOSED
            or operator_spec.include_proposed
        )
    )


def _range_prefix(value: str) -> int:
    _require(len(value) >= 2 and value[0] in {"H", "G"}, "opaque ID changed")
    try:
        return int(value[1:]) // 100_000
    except ValueError as exc:  # pragma: no cover - binding contracts guard it
        raise MatchedEvalContractError("opaque ID changed") from exc


def _validate_input_partitions(
    contributions: tuple[TypedEvidenceContribution, ...],
) -> None:
    """Validate complete input ranges, including items later dropped."""

    _require(
        type(contributions) is tuple
        and bool(contributions)
        and all(type(row) is TypedEvidenceContribution for row in contributions),
        "additive contributions must be a nonempty exact tuple",
    )
    mechanisms = tuple(row.mechanism_id for row in contributions)
    _require(
        len(set(mechanisms)) == len(mechanisms),
        "additive contribution mechanisms repeat",
    )
    all_handles = tuple(
        binding.handle_id
        for contribution in contributions
        for binding in contribution.bindings
    )
    _require(
        len(set(all_handles)) == len(all_handles),
        "additive contribution handles collide",
    )
    item_receipts = tuple(
        item.receipt_sha256
        for contribution in contributions
        for item in contribution.parsed.accepted_items
    )
    _require(
        len(set(item_receipts)) == len(item_receipts),
        "additive contribution item receipts collide",
    )

    handle_range_by_mechanism: dict[str, int] = {}
    group_range_by_mechanism: dict[str, int] = {}
    for contribution in contributions:
        if not contribution.bindings:
            continue
        handle_ranges = {
            _range_prefix(row.handle_id) for row in contribution.bindings
        }
        group_ranges = {
            _range_prefix(row.source_group_handle)
            for row in contribution.bindings
        }
        _require(
            len(handle_ranges) == len(group_ranges) == 1,
            "one additive contribution crossed its opaque H/G range",
        )
        handle_range_by_mechanism[contribution.mechanism_id] = next(
            iter(handle_ranges)
        )
        group_range_by_mechanism[contribution.mechanism_id] = next(
            iter(group_ranges)
        )
    _require(
        len(set(handle_range_by_mechanism.values()))
        == len(handle_range_by_mechanism)
        and len(set(group_range_by_mechanism.values()))
        == len(group_range_by_mechanism),
        "additive contributions do not have globally disjoint H/G ranges",
    )


def _normalized_mechanism_priorities(
    values: Mapping[str, int],
    mechanisms: tuple[str, ...],
    *,
    label: str,
) -> dict[str, int]:
    result = dict(values)
    _require(
        set(result) == set(mechanisms)
        and all(type(key) is str and key for key in result)
        and all(type(value) is int for value in result.values()),
        f"{label} must assign one exact integer to every mechanism",
    )
    return result


def _normalized_local_priorities(
    values: Mapping[str, Sequence[int]] | None,
    *,
    known_handles: set[str],
) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for handle, raw in dict(values or {}).items():
        require_text(handle, "additive local-priority handle")
        _require(
            isinstance(raw, Sequence)
            and not isinstance(raw, (str, bytes))
            and len(raw) == LOCAL_RETENTION_PRIORITY_WIDTH
            and all(type(value) is int for value in raw),
            "additive local priorities must use the final fitter fixed width",
        )
        result[handle] = tuple(raw)
    _require(
        set(result) <= known_handles,
        "additive local priority names an unknown input handle",
    )
    return result


def _provider_semantic_projection(item: TypedEvidenceItem) -> dict[str, Any]:
    """Return every typed semantic that can affect provider context."""

    value = item.projection(include_receipt=False)
    # These two fields are representation identities, not provider semantics.
    value.pop("handle_ids")
    value.pop("item_id")
    return value


def deduplicate_selected_contributions(
    contributions: tuple[TypedEvidenceContribution, ...],
    *,
    owner_priority_by_mechanism: Mapping[str, int],
    exact_span_keys_by_handle: Mapping[str, Sequence[str]] | None = None,
) -> tuple[tuple[TypedEvidenceContribution, ...], dict[str, Any]]:
    """Deduplicate only identity-proven semantic equivalents after selection.

    A lower-priority representation is excluded only when a different
    mechanism already owns an item with the exact same provider semantics and
    the exact same nonempty set of immutable evidence/span receipts.  Equal
    text at different coordinates, partial origin overlap, and richer or
    otherwise semantically different representations all survive.
    """

    _validate_input_partitions(contributions)
    mechanisms = tuple(row.mechanism_id for row in contributions)
    priorities = _normalized_mechanism_priorities(
        owner_priority_by_mechanism,
        mechanisms,
        label="dedup owner priority",
    )
    known_handles = {
        row.handle_id for contribution in contributions for row in contribution.bindings
    }
    span_keys: dict[str, tuple[str, ...]] = {}
    for handle, raw_values in dict(exact_span_keys_by_handle or {}).items():
        require_text(handle, "dedup span handle")
        _require(
            isinstance(raw_values, Sequence)
            and not isinstance(raw_values, (str, bytes)),
            "dedup exact span receipts must be a sequence",
        )
        values = tuple(raw_values)
        _require(
            bool(values) and len(set(values)) == len(values),
            "dedup span binding must be nonempty and unique",
        )
        for value in values:
            require_sha256(value, "dedup exact span receipt")
        span_keys[handle] = values
    _require(
        set(span_keys) <= known_handles,
        "dedup exact span binding names an unknown input handle",
    )

    semantic_records: dict[
        str,
        list[tuple[str, TypedEvidenceItem, frozenset[str], tuple[str, ...]]],
    ] = {}
    rebuilt_by_index: dict[int, TypedEvidenceContribution] = {}
    exclusions: list[dict[str, Any]] = []
    processing_order = tuple(
        sorted(
            enumerate(contributions),
            key=lambda row: (-priorities[row[1].mechanism_id], row[0]),
        )
    )
    for contribution_index, contribution in processing_order:
        binding_by_handle = {row.handle_id: row for row in contribution.bindings}
        accepted: list[TypedEvidenceItem] = []
        local_exclusions: list[str] = []
        for item in contribution.parsed.accepted_items:
            semantic_key = identity_sha256(_provider_semantic_projection(item))
            item_spans = frozenset(
                span
                for handle in item.handle_ids
                for span in span_keys.get(handle, ())
            )
            duplicate = next(
                (
                    record
                    for record in semantic_records.get(semantic_key, ())
                    if record[0] != contribution.mechanism_id
                    and bool(item_spans)
                    and record[2] == item_spans
                ),
                None,
            )
            if duplicate is not None:
                owner_mechanism, owner_item, owner_spans, owner_bindings = duplicate
                row = {
                    "dedup_proof": (
                        "equal_nonempty_immutable_evidence_identity_set_plus_"
                        "exact_provider_semantic_projection"
                    ),
                    "duplicate_binding_receipt_sha256s": [
                        binding_by_handle[handle].receipt_sha256
                        for handle in item.handle_ids
                    ],
                    "duplicate_item_receipt_sha256": item.receipt_sha256,
                    "duplicate_mechanism_id": contribution.mechanism_id,
                    "operation_position": "after_all_mechanism_selection",
                    "owner_binding_receipt_sha256s": list(owner_bindings),
                    "owner_item_receipt_sha256": owner_item.receipt_sha256,
                    "owner_mechanism_id": owner_mechanism,
                    "semantic_dedup_key_sha256": semantic_key,
                    "shared_exact_span_receipt_sha256s": sorted(
                        owner_spans & item_spans
                    ),
                }
                exclusions.append(row)
                local_exclusions.append(item.receipt_sha256)
                continue
            accepted.append(item)
            semantic_records.setdefault(semantic_key, []).append(
                (
                    contribution.mechanism_id,
                    item,
                    item_spans,
                    tuple(
                        binding_by_handle[handle].receipt_sha256
                        for handle in item.handle_ids
                    ),
                )
            )

        represented_handles = {
            handle for item in accepted for handle in item.handle_ids
        }
        retained_bindings = tuple(
            binding
            for binding in contribution.bindings
            if binding.handle_id in represented_handles
        )
        _require(
            {row.handle_id for row in retained_bindings} == represented_handles,
            "post-selection dedup left an unbound handle",
        )
        parsed = ParsedTypedItems(
            tuple(accepted),
            contribution.parsed.rejected_items,
            identity_sha256(
                {
                    "accepted_item_receipt_sha256s": [
                        item.receipt_sha256 for item in accepted
                    ],
                    "format": f"{DEDUP_FORMAT}-parse-subset",
                    "original_parse_receipt_sha256": (
                        contribution.parsed.parse_receipt_sha256
                    ),
                    "post_selection_duplicate_receipt_sha256s": local_exclusions,
                }
            ),
        )
        rebuilt_by_index[contribution_index] = TypedEvidenceContribution(
            contribution.mechanism_id,
            retained_bindings,
            parsed,
            contribution.sealed_artifact_sha256,
            contribution.frontier_mode,
            contribution.truncated,
        )

    rebuilt = tuple(rebuilt_by_index[index] for index in range(len(contributions)))
    audit = _sealed_audit(
        {
            "dedup_owner_priority_rows": [
                {"mechanism_id": mechanism, "priority": priorities[mechanism]}
                for mechanism in mechanisms
            ],
            "exact_span_binding_rows": [
                {
                    "exact_span_receipt_sha256s": list(span_keys[handle]),
                    "handle_id": handle,
                }
                for handle in sorted(span_keys)
            ],
            "exclusions": exclusions,
            "format": DEDUP_FORMAT,
            "gold_loaded": False,
            "input_contribution_receipt_sha256s": [
                row.receipt_sha256 for row in contributions
            ],
            "operation_position": "after_all_mechanism_selection",
            "output_contribution_receipt_sha256s": [
                row.receipt_sha256 for row in rebuilt
            ],
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
        },
        path="typed_additive_post_selection_dedup",
    )
    return rebuilt, audit


def _fair_merge_contributions(
    operator_spec: TypedOperatorSpec,
    contributions: tuple[TypedEvidenceContribution, ...],
    *,
    protected_item_receipt_sha256s: tuple[str, ...],
    minimum_allocation_receipt_sha256: str,
    surplus_fill_audit: Mapping[str, Any],
    mechanism_priority_by_mechanism: Mapping[str, int],
    local_selection_priority_by_handle: Mapping[str, tuple[int, ...]],
) -> tuple[TypedEvidencePacket, dict[str, Any]]:
    """Fairly build the compact pre-fit packet while preserving lane minima."""

    mechanisms = tuple(row.mechanism_id for row in contributions)
    priorities = _normalized_mechanism_priorities(
        mechanism_priority_by_mechanism,
        mechanisms,
        label="fair merge mechanism priority",
    )
    minimum_receipt = require_sha256(
        minimum_allocation_receipt_sha256,
        "fair merge minimum allocation receipt",
    )
    surplus_receipt = _verify_sealed_audit(
        surplus_fill_audit,
        label="fair merge surplus fill audit",
    )
    _require(
        surplus_fill_audit.get("minimum_allocation_receipt_sha256")
        == minimum_receipt
        and surplus_fill_audit.get("minimum_item_receipt_sha256s")
        == list(protected_item_receipt_sha256s)
        and surplus_fill_audit.get("contribution_receipt_sha256s")
        == [row.receipt_sha256 for row in contributions],
        "fair merge inputs do not match the sealed surplus fill",
    )

    binding_by_handle: dict[str, EvidenceHandleBinding] = {}
    owner_by_item_receipt: dict[str, str] = {}
    ordered_items: list[TypedEvidenceItem] = []
    ordinal_by_receipt: dict[str, int] = {}
    for contribution in contributions:
        local_handles = {row.handle_id for row in contribution.bindings}
        for binding in contribution.bindings:
            _require(
                binding.handle_id not in binding_by_handle,
                "fair merge handles collide",
            )
            binding_by_handle[binding.handle_id] = binding
        for item in contribution.parsed.accepted_items:
            _require(
                set(item.handle_ids) <= local_handles
                and item.receipt_sha256 not in owner_by_item_receipt,
                "fair merge item escaped or collided",
            )
            owner_by_item_receipt[item.receipt_sha256] = contribution.mechanism_id
            ordinal_by_receipt[item.receipt_sha256] = len(ordered_items)
            ordered_items.append(item)
    _require(
        set(local_selection_priority_by_handle) <= set(binding_by_handle),
        "fair merge local priority names an unallocated handle",
    )
    zero_local_priority = (0,) * LOCAL_RETENTION_PRIORITY_WIDTH

    def strength(item: TypedEvidenceItem) -> tuple[Any, ...]:
        local_priority = max(
            (
                local_selection_priority_by_handle.get(handle, zero_local_priority)
                for handle in item.handle_ids
            ),
            default=zero_local_priority,
        )
        return (
            int(item.included),
            int(item.content_coherence is not ContentCoherence.CONFLICT),
            local_priority,
            len(item.supported_slot_ids),
            priorities[owner_by_item_receipt[item.receipt_sha256]],
            -ordinal_by_receipt[item.receipt_sha256],
            item.receipt_sha256,
        )

    protected_receipts = tuple(protected_item_receipt_sha256s)
    _require(
        bool(protected_receipts)
        and len(set(protected_receipts)) == len(protected_receipts),
        "fair merge requires ordered unique nonempty lane minima",
    )
    for value in protected_receipts:
        require_sha256(value, "fair merge protected item receipt")
    item_by_receipt = {row.receipt_sha256: row for row in ordered_items}
    _require(
        set(protected_receipts) <= set(item_by_receipt)
        and all(_usable_item(item_by_receipt[row], operator_spec) for row in protected_receipts),
        "fair merge protected lane minimum is missing or unusable",
    )
    protected = [item_by_receipt[row] for row in protected_receipts]
    protected_owner_ids = {
        owner_by_item_receipt[item.receipt_sha256] for item in protected
    }
    _require(
        all(
            not any(_usable_item(item, operator_spec) for item in row.parsed.accepted_items)
            or row.mechanism_id in protected_owner_ids
            for row in contributions
        ),
        "fair merge protected lane minima lost a nonempty mechanism",
    )
    protected_handles = {
        handle for item in protected for handle in item.handle_ids
    }
    protected_binding_receipts = tuple(
        binding.receipt_sha256
        for contribution in contributions
        for binding in contribution.bindings
        if binding.handle_id in protected_handles
    )
    _require(
        set(protected_binding_receipts)
        == set(surplus_fill_audit.get("minimum_binding_receipt_sha256s", ())),
        "fair merge protected binding partition changed",
    )

    selected = list(protected)
    selected_receipts = set(protected_receipts)
    remaining = sorted(
        (
            item
            for item in ordered_items
            if _usable_item(item, operator_spec)
            and item.receipt_sha256 not in selected_receipts
        ),
        key=strength,
        reverse=True,
    )
    rejected = tuple(
        item
        for contribution in contributions
        for item in contribution.parsed.rejected_items
    )
    modes = {row.frontier_mode for row in contributions}
    mode = (
        FrontierMode.OPEN
        if FrontierMode.OPEN in modes
        else FrontierMode.BOUNDED
        if FrontierMode.BOUNDED in modes
        else FrontierMode.EXHAUSTIVE
    )
    artifacts = tuple(
        dict.fromkeys(row.sealed_artifact_sha256 for row in contributions)
    )

    def build(items: Sequence[TypedEvidenceItem]) -> TypedEvidencePacket:
        represented_handles = {
            handle for item in items for handle in item.handle_ids
        }
        bindings = tuple(
            binding
            for contribution in contributions
            for binding in contribution.bindings
            if binding.handle_id in represented_handles
        )
        parsed = ParsedTypedItems(
            tuple(items),
            rejected,
            identity_sha256(
                {
                    "accepted_item_receipt_sha256s": [
                        item.receipt_sha256 for item in items
                    ],
                    "contribution_receipt_sha256s": [
                        row.receipt_sha256 for row in contributions
                    ],
                    "format": f"{FAIR_MERGE_FORMAT}-parse-subset",
                    "rejected_item_receipt_sha256s": [
                        row.rejection_sha256 for row in rejected
                    ],
                }
            ),
        )
        return build_typed_evidence_packet(
            operator_spec,
            bindings,
            parsed,
            sealed_input_artifact_sha256s=artifacts,
            frontier_mode=mode,
            conflict_policy=ConflictPolicy.QUARANTINE,
            output_token_reserve=1,
            truncated=(
                any(row.truncated for row in contributions)
                or len(items) < len(ordered_items)
            ),
            provider_payload_mode=ProviderPayloadMode.COMPACT_FINAL,
        )

    packet = build(selected)
    _require(
        {row.receipt_sha256 for row in packet.items} == set(protected_receipts),
        "protected lane minima exceed the typed packet envelope",
    )
    for item in remaining:
        trial = tuple((*selected, item))
        candidate = build(trial)
        if {row.receipt_sha256 for row in candidate.items} == {
            row.receipt_sha256 for row in trial
        }:
            selected.append(item)
            packet = candidate

    final_receipts = {row.receipt_sha256 for row in packet.items}
    rows: list[dict[str, Any]] = []
    for contribution in contributions:
        candidates = contribution.parsed.accepted_items
        rows.append(
            {
                "accepted_candidate_count": len(candidates),
                "admitted_item_receipt_sha256s": [
                    item.receipt_sha256
                    for item in candidates
                    if item.receipt_sha256 in final_receipts
                ],
                "dropped_item_receipt_sha256s": [
                    item.receipt_sha256
                    for item in candidates
                    if item.receipt_sha256 not in final_receipts
                ],
                "mechanism_id": contribution.mechanism_id,
                "parser_rejected_count": len(contribution.parsed.rejected_items),
                "protected_minimum_item_receipt_sha256s": [
                    item.receipt_sha256
                    for item in protected
                    if owner_by_item_receipt[item.receipt_sha256]
                    == contribution.mechanism_id
                ],
                "usable_candidate_count": sum(
                    _usable_item(item, operator_spec) for item in candidates
                ),
            }
        )
    audit = _sealed_audit(
        {
            "format": FAIR_MERGE_FORMAT,
            "gold_loaded": False,
            "input_contribution_receipt_sha256s": [
                row.receipt_sha256 for row in contributions
            ],
            "local_selection_priority_receipt_sha256": identity_sha256(
                {
                    "fixed_width": LOCAL_RETENTION_PRIORITY_WIDTH,
                    "format": f"{FAIR_MERGE_FORMAT}-local-priority-v1",
                    "rows": [
                        {"handle_id": handle, "priority": list(priority)}
                        for handle, priority in sorted(
                            local_selection_priority_by_handle.items()
                        )
                    ],
                }
            ),
            "mechanism_priority_rows": [
                {"mechanism_id": mechanism, "priority": priorities[mechanism]}
                for mechanism in mechanisms
            ],
            "mechanisms": rows,
            "minimum_allocation_receipt_sha256": minimum_receipt,
            "packet_receipt_sha256": packet.receipt_sha256,
            "policy": (
                "all_exact_non_borrowable_lane_minima_then_local_priority_"
                "and_typed_strength_fill_against_compact_final_projection"
            ),
            "protected_minimum_binding_receipt_sha256s": list(
                protected_binding_receipts
            ),
            "protected_minimum_item_receipt_sha256s": list(protected_receipts),
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "shared_lane_surplus_fill_receipt_sha256": surplus_receipt,
        },
        path="typed_additive_fair_premerge",
    )
    return packet, audit


def retained_mechanism_bindings(
    contributions: tuple[TypedEvidenceContribution, ...],
    packet: TypedEvidencePacket,
) -> tuple[dict[str, str], tuple[dict[str, Any], ...]]:
    """Return retained mechanism ownership and exact dropped provenance."""

    _require(type(packet) is TypedEvidencePacket, "retained packet changed type")
    allocated: dict[str, tuple[str, EvidenceHandleBinding]] = {}
    for contribution in contributions:
        for binding in contribution.bindings:
            _require(
                binding.handle_id not in allocated,
                "allocated mechanism bindings collide",
            )
            allocated[binding.handle_id] = (contribution.mechanism_id, binding)
    packet_handle_ids = {row.handle_id for row in packet.handles}
    _require(
        packet_handle_ids <= set(allocated),
        "fair-merge packet handle escaped allocated contributions",
    )
    retained = {
        handle: allocated[handle][0]
        for handle in allocated
        if handle in packet_handle_ids
    }
    dropped = tuple(
        binding.projection()
        for handle, (_mechanism, binding) in allocated.items()
        if handle not in packet_handle_ids
    )
    _require(
        set(retained) == packet_handle_ids,
        "typed contribution mechanism bindings changed",
    )
    return retained, dropped


@dataclass(frozen=True, slots=True)
class AdditiveTypedComposition:
    """Sealed provider-free output ready for the exact final prompt fitter."""

    packet: TypedEvidencePacket
    contributions: tuple[TypedEvidenceContribution, ...]
    minimum_allocation: TypedLaneAllocation
    mechanism_by_handle: Mapping[str, str]
    protected_item_receipt_sha256s: tuple[str, ...]
    retained_local_priority_by_handle: Mapping[str, tuple[int, ...]]
    post_selection_dedup_audit: Mapping[str, Any]
    surplus_fill_audit: Mapping[str, Any]
    fair_merge_audit: Mapping[str, Any]
    dropped_binding_projections: tuple[Mapping[str, Any], ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.packet) is TypedEvidencePacket, "additive packet changed type")
        _require(
            type(self.contributions) is tuple
            and bool(self.contributions)
            and all(type(row) is TypedEvidenceContribution for row in self.contributions),
            "additive final contributions changed type",
        )
        _require(
            type(self.minimum_allocation) is TypedLaneAllocation,
            "additive minimum allocation changed type",
        )
        mechanism = dict(self.mechanism_by_handle)
        _require(
            set(mechanism) == {row.handle_id for row in self.packet.handles},
            "additive mechanism map changed packet coverage",
        )
        validate_disjoint_contribution_ranges(self.packet, mechanism)
        for audit, label in (
            (self.post_selection_dedup_audit, "additive dedup audit"),
            (self.surplus_fill_audit, "additive surplus audit"),
            (self.fair_merge_audit, "additive fair merge audit"),
        ):
            _verify_sealed_audit(audit, label=label)
        _require(
            tuple(self.fair_merge_audit.get("protected_minimum_item_receipt_sha256s", ()))
            == self.protected_item_receipt_sha256s
            and self.fair_merge_audit.get("packet_receipt_sha256")
            == self.packet.receipt_sha256,
            "additive result escaped its protected minima or packet",
        )
        _require(
            set(self.retained_local_priority_by_handle) <= set(mechanism),
            "additive retained local priority escaped packet handles",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "additive result receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="typed_additive_composition")

    @property
    def protection_source_receipt_sha256(self) -> str:
        """Receipt to bind ``protected_item_receipt_sha256s`` during fitting."""

        return self.receipt_sha256

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "contribution_receipt_sha256s": [
                row.receipt_sha256 for row in self.contributions
            ],
            "dropped_binding_receipt_sha256s": [
                row["receipt_sha256"] for row in self.dropped_binding_projections
            ],
            "fair_merge_audit_receipt_sha256": self.fair_merge_audit[
                "receipt_sha256"
            ],
            "format": FORMAT,
            "gold_loaded": False,
            "mechanism_by_handle": dict(self.mechanism_by_handle),
            "minimum_allocation_receipt_sha256": (
                self.minimum_allocation.receipt_sha256
            ),
            "packet_receipt_sha256": self.packet.receipt_sha256,
            "post_selection_dedup_audit_receipt_sha256": (
                self.post_selection_dedup_audit["receipt_sha256"]
            ),
            "protected_item_receipt_sha256s": list(
                self.protected_item_receipt_sha256s
            ),
            "provider_prompt_count": 0,
            "retained_local_priority_receipt_sha256": identity_sha256(
                {
                    "fixed_width": LOCAL_RETENTION_PRIORITY_WIDTH,
                    "format": f"{FORMAT}-retained-local-priority-v1",
                    "rows": [
                        {"handle_id": handle, "priority": list(priority)}
                        for handle, priority in sorted(
                            self.retained_local_priority_by_handle.items()
                        )
                    ],
                }
            ),
            "retained_transformer_token_state_bytes": 0,
            "surplus_fill_audit_receipt_sha256": self.surplus_fill_audit[
                "receipt_sha256"
            ],
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def compose_additive_typed_evidence(
    operator_spec: TypedOperatorSpec,
    contributions: tuple[TypedEvidenceContribution, ...],
    *,
    lane_budgets: tuple[LaneBudget, ...],
    lane_by_mechanism: Mapping[str, str],
    dedup_owner_priority_by_mechanism: Mapping[str, int],
    exact_span_keys_by_handle: Mapping[str, Sequence[str]] | None = None,
    local_selection_priority_by_handle: Mapping[str, Sequence[int]] | None = None,
    fair_merge_priority_by_mechanism: Mapping[str, int] | None = None,
) -> AdditiveTypedComposition:
    """Compose independently selected mechanism outputs into one typed packet.

    ``dedup_owner_priority_by_mechanism`` is mandatory and higher values own
    proven duplicates.  This makes protected-parent ownership an explicit
    caller policy instead of a mechanism-name heuristic.  When omitted, fair
    merge priorities reuse that same explicit map.
    """

    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be exact")
    _validate_input_partitions(contributions)
    mechanisms = tuple(row.mechanism_id for row in contributions)
    dedup_priorities = _normalized_mechanism_priorities(
        dedup_owner_priority_by_mechanism,
        mechanisms,
        label="dedup owner priority",
    )
    merge_priorities = _normalized_mechanism_priorities(
        fair_merge_priority_by_mechanism or dedup_priorities,
        mechanisms,
        label="fair merge mechanism priority",
    )
    input_handles = {
        row.handle_id for contribution in contributions for row in contribution.bindings
    }
    local_priorities = _normalized_local_priorities(
        local_selection_priority_by_handle,
        known_handles=input_handles,
    )

    deduplicated, dedup_audit = deduplicate_selected_contributions(
        contributions,
        owner_priority_by_mechanism=dedup_priorities,
        exact_span_keys_by_handle=exact_span_keys_by_handle,
    )
    deduplicated_handles = {
        row.handle_id for contribution in deduplicated for row in contribution.bindings
    }
    retained_priorities = {
        handle: priority
        for handle, priority in local_priorities.items()
        if handle in deduplicated_handles
    }
    minimum = allocate_typed_contribution_lanes(
        deduplicated,
        lane_budgets=lane_budgets,
        lane_by_mechanism=lane_by_mechanism,
        operator_spec=operator_spec,
        local_selection_priority_by_handle=retained_priorities,
    )
    minimum_by_mechanism = {
        row.mechanism_id: row for row in minimum.contributions
    }
    _require(
        all(
            not any(_usable_item(item, operator_spec) for item in original.parsed.accepted_items)
            or any(
                _usable_item(item, operator_spec)
                for item in minimum_by_mechanism[
                    original.mechanism_id
                ].parsed.accepted_items
            )
            for original in deduplicated
        ),
        "non-borrowable lane cap starved a nonempty additive mechanism",
    )
    protected_receipts = tuple(
        receipt
        for lane_receipt in minimum.receipts
        for receipt in lane_receipt.selected_item_receipt_sha256s
    )
    _require(
        bool(protected_receipts),
        "additive composition requires at least one usable lane minimum",
    )
    expanded, surplus_audit = fill_typed_lane_surplus(
        deduplicated,
        minimum,
        lane_budgets=lane_budgets,
        lane_by_mechanism=lane_by_mechanism,
        operator_spec=operator_spec,
        local_selection_priority_by_handle=retained_priorities,
    )
    expanded_handles = {
        row.handle_id for contribution in expanded for row in contribution.bindings
    }
    expanded_priorities = {
        handle: priority
        for handle, priority in retained_priorities.items()
        if handle in expanded_handles
    }
    packet, fair_audit = _fair_merge_contributions(
        operator_spec,
        expanded,
        protected_item_receipt_sha256s=protected_receipts,
        minimum_allocation_receipt_sha256=minimum.receipt_sha256,
        surplus_fill_audit=surplus_audit,
        mechanism_priority_by_mechanism=merge_priorities,
        local_selection_priority_by_handle=expanded_priorities,
    )
    mechanism_by_handle, dropped = retained_mechanism_bindings(expanded, packet)
    validate_disjoint_contribution_ranges(packet, mechanism_by_handle)
    packet_priorities = {
        handle: priority
        for handle, priority in expanded_priorities.items()
        if handle in mechanism_by_handle
    }
    return AdditiveTypedComposition(
        packet,
        expanded,
        minimum,
        mechanism_by_handle,
        protected_receipts,
        packet_priorities,
        dedup_audit,
        surplus_audit,
        fair_audit,
        dropped,
    )


__all__ = [
    "AdditiveTypedComposition",
    "DEDUP_FORMAT",
    "FAIR_MERGE_FORMAT",
    "FORMAT",
    "compose_additive_typed_evidence",
    "deduplicate_selected_contributions",
    "retained_mechanism_bindings",
]
