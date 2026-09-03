"""Gold-blind additive composition for independently selected typed evidence.

The composer is the reusable boundary between retrieval mechanisms and the
common final prompt fitter.  Every mechanism selects before this module is
called.  Composition then performs, in order:

1. non-borrowable lane allocation;
2. shared-surplus fill without dropping any lane minimum;
3. identity-proven cross-mechanism deduplication;
4. refill of capacity released by that deduplication; and
5. a fair compact packet merge that protects those exact minima.

No provider is called and no benchmark answer or verdict is accepted by the
API.  All decisions and lifecycle partitions are bound by SHA-256 receipts.
The returned packet, mechanism map, protected item receipts, and protection
receipt can be passed directly to ``fit_typed_final_prompt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

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
    lane_content_token_proxy,
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
POST_DEDUP_BACKFILL_FORMAT = "memory-condense-typed-additive-composer-v2"
POST_DEDUP_BACKFILL_DEDUP_FORMAT = (
    f"{POST_DEDUP_BACKFILL_FORMAT}-post-selection-dedup-v1"
)
DEDUP_BACKFILL_FORMAT = (
    f"{POST_DEDUP_BACKFILL_FORMAT}-post-dedup-capacity-backfill-v1"
)
POST_DEDUP_BACKFILL_FAIR_MERGE_FORMAT = (
    f"{POST_DEDUP_BACKFILL_FORMAT}-fair-premerge-v1"
)

LEGACY_COMPOSITION_MODE = "legacy_v1"
POST_DEDUP_BACKFILL_COMPOSITION_MODE = "post_dedup_backfill_v2"
CompositionMode = Literal["legacy_v1", "post_dedup_backfill_v2"]


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


def _normalized_exact_span_keys(
    contributions: tuple[TypedEvidenceContribution, ...],
    exact_span_keys_by_handle: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    """Validate immutable span identities against one contribution population."""

    known_handles = {
        row.handle_id
        for contribution in contributions
        for row in contribution.bindings
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
    return span_keys


def deduplicate_selected_contributions(
    contributions: tuple[TypedEvidenceContribution, ...],
    *,
    owner_priority_by_mechanism: Mapping[str, int],
    exact_span_keys_by_handle: Mapping[str, Sequence[str]] | None = None,
    operation_position: str = "after_all_mechanism_selection",
    audit_format: str = DEDUP_FORMAT,
) -> tuple[tuple[TypedEvidenceContribution, ...], dict[str, Any]]:
    """Deduplicate only identity-proven semantic equivalents after selection.

    A lower-priority representation is excluded only when a different
    mechanism already owns an item with the exact same provider semantics and
    the exact same nonempty set of immutable evidence/span receipts.  Equal
    text at different coordinates, partial origin overlap, and richer or
    otherwise semantically different representations all survive.
    """

    _validate_input_partitions(contributions)
    require_text(operation_position, "dedup operation position")
    _require(
        audit_format in {DEDUP_FORMAT, POST_DEDUP_BACKFILL_DEDUP_FORMAT},
        "dedup audit format changed",
    )
    mechanisms = tuple(row.mechanism_id for row in contributions)
    priorities = _normalized_mechanism_priorities(
        owner_priority_by_mechanism,
        mechanisms,
        label="dedup owner priority",
    )
    span_keys = _normalized_exact_span_keys(
        contributions,
        exact_span_keys_by_handle,
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
                    "operation_position": operation_position,
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
                    "format": f"{audit_format}-parse-subset",
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
            "format": audit_format,
            "gold_loaded": False,
            "input_contribution_receipt_sha256s": [
                row.receipt_sha256 for row in contributions
            ],
            "operation_position": operation_position,
            "output_contribution_receipt_sha256s": [
                row.receipt_sha256 for row in rebuilt
            ],
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
        },
        path="typed_additive_post_selection_dedup",
    )
    return rebuilt, audit


def _backfill_freed_dedup_capacity(
    operator_spec: TypedOperatorSpec,
    original_contributions: tuple[TypedEvidenceContribution, ...],
    deduplicated_contributions: tuple[TypedEvidenceContribution, ...],
    *,
    lane_budgets: tuple[LaneBudget, ...],
    lane_by_mechanism: Mapping[str, str],
    exact_span_keys_by_handle: Mapping[str, Sequence[str]] | None,
    surplus_fill_audit: Mapping[str, Any],
    initial_dedup_audit: Mapping[str, Any],
) -> tuple[tuple[TypedEvidenceContribution, ...], dict[str, Any]]:
    """Refill only capacity physically released by exact post-admission dedup.

    The surplus allocator has already fixed a gold-blind consideration order
    and recorded every usable item that missed the aggregate lane envelope.
    Deduplication can make that prior capacity decision stale.  This pass
    reconsiders that exact omitted order, admits only globally novel span
    classes, and never exceeds the original aggregate lane cap.
    """

    _validate_input_partitions(original_contributions)
    _validate_input_partitions(deduplicated_contributions)
    surplus_receipt = _verify_sealed_audit(
        surplus_fill_audit,
        label="post-dedup backfill surplus audit",
    )
    initial_dedup_receipt = _verify_sealed_audit(
        initial_dedup_audit,
        label="post-dedup backfill initial dedup audit",
    )
    mechanisms = tuple(row.mechanism_id for row in original_contributions)
    _require(
        tuple(row.mechanism_id for row in deduplicated_contributions)
        == mechanisms,
        "post-dedup backfill mechanisms changed order",
    )
    budgets = {row.lane_id: row for row in lane_budgets}
    mapping = dict(lane_by_mechanism)
    _require(
        len(budgets) == len(lane_budgets)
        and set(mapping) == set(mechanisms)
        and set(mapping.values()) == set(budgets),
        "post-dedup backfill lane policy changed",
    )
    _require(
        surplus_fill_audit.get("original_contribution_receipt_sha256s")
        == [row.receipt_sha256 for row in original_contributions]
        and initial_dedup_audit.get("input_contribution_receipt_sha256s")
        == surplus_fill_audit.get("contribution_receipt_sha256s")
        and initial_dedup_audit.get("output_contribution_receipt_sha256s")
        == [row.receipt_sha256 for row in deduplicated_contributions]
        and initial_dedup_audit.get("operation_position")
        == "after_independent_lane_admission_and_shared_surplus_fill",
        "post-dedup backfill inputs escaped the sealed admission lifecycle",
    )

    original_by_mechanism = {
        row.mechanism_id: row for row in original_contributions
    }
    item_by_receipt: dict[str, TypedEvidenceItem] = {}
    owner_by_receipt: dict[str, str] = {}
    binding_by_handle: dict[str, EvidenceHandleBinding] = {}
    for contribution in original_contributions:
        for binding in contribution.bindings:
            _require(
                binding.handle_id not in binding_by_handle,
                "post-dedup backfill handles collide",
            )
            binding_by_handle[binding.handle_id] = binding
        for item in contribution.parsed.accepted_items:
            _require(
                item.receipt_sha256 not in item_by_receipt,
                "post-dedup backfill item receipts collide",
            )
            item_by_receipt[item.receipt_sha256] = item
            owner_by_receipt[item.receipt_sha256] = contribution.mechanism_id

    selected_by_mechanism: dict[str, list[TypedEvidenceItem]] = {}
    selected_receipts: set[str] = set()
    for contribution in deduplicated_contributions:
        original = original_by_mechanism[contribution.mechanism_id]
        original_items = {
            row.receipt_sha256: row for row in original.parsed.accepted_items
        }
        selected = list(contribution.parsed.accepted_items)
        _require(
            all(
                item.receipt_sha256 in original_items
                and item == original_items[item.receipt_sha256]
                for item in selected
            ),
            "post-dedup backfill retained item changed its original semantics",
        )
        selected_by_mechanism[contribution.mechanism_id] = selected
        for item in selected:
            _require(
                item.receipt_sha256 not in selected_receipts,
                "post-dedup backfill retained item repeats",
            )
            selected_receipts.add(item.receipt_sha256)

    span_keys = _normalized_exact_span_keys(
        original_contributions,
        exact_span_keys_by_handle,
    )

    def equivalence_parts(
        item: TypedEvidenceItem,
    ) -> tuple[str, frozenset[str]]:
        return (
            identity_sha256(_provider_semantic_projection(item)),
            frozenset(
                span
                for handle in item.handle_ids
                for span in span_keys.get(handle, ())
            ),
        )

    semantic_records: dict[
        str,
        list[tuple[str, TypedEvidenceItem, frozenset[str]]],
    ] = {}
    for mechanism in mechanisms:
        for item in selected_by_mechanism[mechanism]:
            semantic_key, item_spans = equivalence_parts(item)
            semantic_records.setdefault(semantic_key, []).append(
                (mechanism, item, item_spans)
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

    def selected_lane_items(lane_id: str) -> tuple[TypedEvidenceItem, ...]:
        return tuple(
            item
            for mechanism in mechanisms
            if mapping[mechanism] == lane_id
            for item in selected_by_mechanism[mechanism]
        )

    proxy_by_lane = {
        lane_id: lane_content_token_proxy(
            selected_lane_items(lane_id),
            bindings_by_lane[lane_id],
        )
        for lane_id in budgets
    }
    shared_cap = sum(row.final_content_token_cap for row in lane_budgets)
    pre_backfill_proxy = sum(proxy_by_lane.values())
    _require(
        pre_backfill_proxy <= shared_cap,
        "post-dedup population exceeds the original shared lane cap",
    )

    raw_omitted = surplus_fill_audit.get(
        "budget_omitted_item_receipt_sha256s"
    )
    _require(
        type(raw_omitted) is list
        and len(raw_omitted) == len(set(raw_omitted))
        and all(
            type(value) is str
            and value in item_by_receipt
            and value not in selected_receipts
            for value in raw_omitted
        ),
        "post-dedup backfill omitted population changed",
    )
    for value in raw_omitted:
        require_sha256(value, "post-dedup omitted item")
        _require(
            _usable_item(item_by_receipt[value], operator_spec),
            "post-dedup budget-omitted item became unusable",
        )

    rows: list[dict[str, Any]] = []
    backfilled_receipts: list[str] = []
    for receipt in raw_omitted:
        item = item_by_receipt[receipt]
        mechanism = owner_by_receipt[receipt]
        semantic_key, item_spans = equivalence_parts(item)
        duplicate = next(
            (
                record
                for record in semantic_records.get(semantic_key, ())
                if record[0] != mechanism
                and bool(item_spans)
                and record[2] == item_spans
            ),
            None,
        )
        before = sum(proxy_by_lane.values())
        if duplicate is not None:
            disposition = "skipped_exact_duplicate"
            owner_item_receipt = duplicate[1].receipt_sha256
            after = before
        else:
            lane_id = mapping[mechanism]
            selected_by_mechanism[mechanism].append(item)
            trial_lane_proxy = lane_content_token_proxy(
                selected_lane_items(lane_id),
                bindings_by_lane[lane_id],
            )
            trial_total = sum(
                trial_lane_proxy if key == lane_id else value
                for key, value in proxy_by_lane.items()
            )
            if trial_total <= shared_cap:
                proxy_by_lane[lane_id] = trial_lane_proxy
                selected_receipts.add(receipt)
                backfilled_receipts.append(receipt)
                semantic_records.setdefault(semantic_key, []).append(
                    (mechanism, item, item_spans)
                )
                disposition = "admitted_unique"
                owner_item_receipt = None
                after = trial_total
            else:
                selected_by_mechanism[mechanism].pop()
                disposition = "shared_capacity_unfit"
                owner_item_receipt = None
                after = before
        rows.append(
            {
                "content_token_proxy_after": after,
                "content_token_proxy_before": before,
                "disposition": disposition,
                "item_receipt_sha256": receipt,
                "mechanism_id": mechanism,
                "owner_item_receipt_sha256": owner_item_receipt,
                "shared_exact_span_receipt_sha256s": sorted(item_spans),
            }
        )

    rebuilt: list[TypedEvidenceContribution] = []
    for mechanism in mechanisms:
        original = original_by_mechanism[mechanism]
        selected = tuple(selected_by_mechanism[mechanism])
        represented = {
            handle for item in selected for handle in item.handle_ids
        }
        bindings = tuple(
            row for row in original.bindings if row.handle_id in represented
        )
        parsed = ParsedTypedItems(
            selected,
            original.parsed.rejected_items,
            identity_sha256(
                {
                    "backfilled_item_receipt_sha256s": [
                        receipt
                        for receipt in backfilled_receipts
                        if owner_by_receipt[receipt] == mechanism
                    ],
                    "format": f"{DEDUP_BACKFILL_FORMAT}-parse-subset",
                    "initial_post_selection_dedup_receipt_sha256": (
                        initial_dedup_receipt
                    ),
                    "mechanism_id": mechanism,
                    "original_parse_receipt_sha256": (
                        original.parsed.parse_receipt_sha256
                    ),
                    "selected_item_receipt_sha256s": [
                        item.receipt_sha256 for item in selected
                    ],
                    "surplus_fill_receipt_sha256": surplus_receipt,
                }
            ),
        )
        rebuilt.append(
            TypedEvidenceContribution(
                mechanism,
                bindings,
                parsed,
                original.sealed_artifact_sha256,
                original.frontier_mode,
                original.truncated
                or len(selected) < len(original.parsed.accepted_items),
            )
        )

    final_proxy = sum(proxy_by_lane.values())
    audit = _sealed_audit(
        {
            "backfill_rows": rows,
            "backfilled_item_receipt_sha256s": backfilled_receipts,
            "exclusions": list(initial_dedup_audit.get("exclusions", ())),
            "final_content_token_proxy": final_proxy,
            "format": DEDUP_BACKFILL_FORMAT,
            "gold_loaded": False,
            "initial_post_selection_dedup_audit": dict(
                initial_dedup_audit
            ),
            "input_contribution_receipt_sha256s": (
                surplus_fill_audit.get("contribution_receipt_sha256s")
            ),
            "operation_position": (
                "after_independent_lane_admission_and_shared_surplus_fill_"
                "then_dedup_and_freed_capacity_backfill"
            ),
            "output_contribution_receipt_sha256s": [
                row.receipt_sha256 for row in rebuilt
            ],
            "pre_backfill_content_token_proxy": pre_backfill_proxy,
            "pre_backfill_contribution_receipt_sha256s": [
                row.receipt_sha256 for row in deduplicated_contributions
            ],
            "provider_prompt_count": 0,
            "reconsidered_budget_omitted_item_receipt_sha256s": list(
                raw_omitted
            ),
            "retained_transformer_token_state_bytes": 0,
            "shared_final_content_token_cap": shared_cap,
            "surplus_fill_receipt_sha256": surplus_receipt,
            "unspent_shared_content_tokens": shared_cap - final_proxy,
        },
        path="typed_additive_post_dedup_capacity_backfill",
    )
    return tuple(rebuilt), audit


def _fair_merge_contributions(
    operator_spec: TypedOperatorSpec,
    contributions: tuple[TypedEvidenceContribution, ...],
    *,
    original_protected_item_receipt_sha256s: tuple[str, ...] | None,
    protected_item_receipt_sha256s: tuple[str, ...],
    minimum_allocation_receipt_sha256: str,
    surplus_fill_audit: Mapping[str, Any],
    post_selection_dedup_audit: Mapping[str, Any] | None,
    mechanism_priority_by_mechanism: Mapping[str, int],
    local_selection_priority_by_handle: Mapping[str, tuple[int, ...]],
    provider_payload_mode: ProviderPayloadMode,
    composition_mode: CompositionMode,
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
    successor = composition_mode == POST_DEDUP_BACKFILL_COMPOSITION_MODE
    _require(
        composition_mode
        in {LEGACY_COMPOSITION_MODE, POST_DEDUP_BACKFILL_COMPOSITION_MODE},
        "fair merge composition mode changed",
    )
    fair_merge_format = (
        POST_DEDUP_BACKFILL_FAIR_MERGE_FORMAT if successor else FAIR_MERGE_FORMAT
    )
    if successor:
        _require(
            type(post_selection_dedup_audit) is dict
            and type(original_protected_item_receipt_sha256s) is tuple,
            "successor fair merge lost its post-dedup authority",
        )
        dedup_receipt = _verify_sealed_audit(
            post_selection_dedup_audit,
            label="fair merge post-admission dedup audit",
        )
        initial_dedup_audit = post_selection_dedup_audit.get(
            "initial_post_selection_dedup_audit"
        )
        _verify_sealed_audit(
            initial_dedup_audit,
            label="fair merge initial post-admission dedup audit",
        )
        original_protected_receipts = tuple(
            original_protected_item_receipt_sha256s
        )
        _require(
            surplus_fill_audit.get("minimum_allocation_receipt_sha256")
            == minimum_receipt
            and surplus_fill_audit.get("minimum_item_receipt_sha256s")
            == list(original_protected_receipts)
            and post_selection_dedup_audit.get(
                "input_contribution_receipt_sha256s"
            )
            == surplus_fill_audit.get("contribution_receipt_sha256s")
            and initial_dedup_audit.get("input_contribution_receipt_sha256s")
            == surplus_fill_audit.get("contribution_receipt_sha256s")
            and initial_dedup_audit.get("output_contribution_receipt_sha256s")
            == post_selection_dedup_audit.get(
                "pre_backfill_contribution_receipt_sha256s"
            )
            and post_selection_dedup_audit.get(
                "output_contribution_receipt_sha256s"
            )
            == [row.receipt_sha256 for row in contributions]
            and post_selection_dedup_audit.get("operation_position")
            == (
                "after_independent_lane_admission_and_shared_surplus_fill_"
                "then_dedup_and_freed_capacity_backfill"
            ),
            "fair merge inputs do not match the sealed surplus fill",
        )
    else:
        _require(
            post_selection_dedup_audit is None
            and original_protected_item_receipt_sha256s is None
            and provider_payload_mode is ProviderPayloadMode.COMPACT_FINAL
            and surplus_fill_audit.get("minimum_allocation_receipt_sha256")
            == minimum_receipt
            and surplus_fill_audit.get("minimum_item_receipt_sha256s")
            == list(protected_item_receipt_sha256s)
            and surplus_fill_audit.get("contribution_receipt_sha256s")
            == [row.receipt_sha256 for row in contributions],
            "legacy v1 fair merge inputs changed",
        )
        dedup_receipt = ""
        initial_dedup_audit = None
        original_protected_receipts = tuple(protected_item_receipt_sha256s)

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
    protection_transfers: list[dict[str, Any]] = []
    exclusion_by_duplicate: dict[str, Mapping[str, Any]] = {}
    if successor:
        _require(type(post_selection_dedup_audit) is dict, "dedup audit changed")
        raw_exclusions = post_selection_dedup_audit.get("exclusions")
        _require(
            type(raw_exclusions) is list
            and all(type(row) is dict for row in raw_exclusions),
            "fair merge dedup exclusions changed schema",
        )
        _require(
            type(initial_dedup_audit) is dict
            and raw_exclusions == initial_dedup_audit.get("exclusions"),
            "fair merge backfill changed the exact dedup authority",
        )
        exclusion_by_duplicate = {
            row["duplicate_item_receipt_sha256"]: row for row in raw_exclusions
        }
        _require(
            len(exclusion_by_duplicate) == len(raw_exclusions),
            "fair merge dedup exclusions repeated a duplicate item",
        )
        expected_effective_protection: list[str] = []
        for original_receipt in original_protected_receipts:
            if original_receipt in item_by_receipt:
                effective_receipt = original_receipt
            else:
                exclusion = exclusion_by_duplicate.get(original_receipt)
                _require(
                    exclusion is not None,
                    "protected lane minimum disappeared without dedup authority",
                )
                effective_receipt = exclusion["owner_item_receipt_sha256"]
                _require(
                    effective_receipt in item_by_receipt,
                    "dedup owner did not survive post-admission composition",
                )
                protection_transfers.append(
                    {
                        "effective_item_receipt_sha256": effective_receipt,
                        "original_item_receipt_sha256": original_receipt,
                        "shared_exact_span_receipt_sha256s": exclusion[
                            "shared_exact_span_receipt_sha256s"
                        ],
                    }
                )
            if effective_receipt not in expected_effective_protection:
                expected_effective_protection.append(effective_receipt)
        _require(
            tuple(expected_effective_protection) == protected_receipts,
            "effective protected minima disagree with dedup authority transfer",
        )
    _require(
        set(protected_receipts) <= set(item_by_receipt)
        and all(_usable_item(item_by_receipt[row], operator_spec) for row in protected_receipts),
        "fair merge protected lane minimum is missing or unusable",
    )
    protected = [item_by_receipt[row] for row in protected_receipts]
    if not successor:
        protected_owner_ids = {
            owner_by_item_receipt[item.receipt_sha256] for item in protected
        }
        _require(
            all(
                not any(
                    _usable_item(item, operator_spec)
                    for item in row.parsed.accepted_items
                )
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
    if successor:
        original_minimum_binding_receipts = set(
            surplus_fill_audit.get("minimum_binding_receipt_sha256s", ())
        )
        effective_binding_receipt_set = set(protected_binding_receipts)
        transferred_duplicate_bindings: set[str] = set()
        for original_receipt in original_protected_receipts:
            exclusion = exclusion_by_duplicate.get(original_receipt)
            if exclusion is None:
                continue
            duplicate_bindings = set(
                exclusion["duplicate_binding_receipt_sha256s"]
            )
            owner_bindings = set(exclusion["owner_binding_receipt_sha256s"])
            transferred_duplicate_bindings.update(duplicate_bindings)
            _require(
                owner_bindings <= effective_binding_receipt_set,
                "dedup protection transfer lost its owner bindings",
            )
        _require(
            original_minimum_binding_receipts
            <= effective_binding_receipt_set | transferred_duplicate_bindings,
            "fair merge protected binding partition changed",
        )
    else:
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
                    "format": f"{fair_merge_format}-parse-subset",
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
            provider_payload_mode=provider_payload_mode,
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
    audit_body = {
            "format": fair_merge_format,
            "gold_loaded": False,
            "input_contribution_receipt_sha256s": [
                row.receipt_sha256 for row in contributions
            ],
            "local_selection_priority_receipt_sha256": identity_sha256(
                {
                    "fixed_width": LOCAL_RETENTION_PRIORITY_WIDTH,
                    "format": f"{fair_merge_format}-local-priority-v1",
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
            "original_protected_minimum_item_receipt_sha256s": list(
                original_protected_receipts
            ),
            "packet_receipt_sha256": packet.receipt_sha256,
            "post_admission_dedup_receipt_sha256": dedup_receipt,
            "policy": (
                "all_exact_non_borrowable_lane_minima_then_local_priority_"
                "and_typed_strength_fill_against_compact_final_projection"
            ),
            "protected_minimum_binding_receipt_sha256s": list(
                protected_binding_receipts
            ),
            "protected_minimum_item_receipt_sha256s": list(protected_receipts),
            "protected_minimum_item_transfer_rows": protection_transfers,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "shared_lane_surplus_fill_receipt_sha256": surplus_receipt,
        }
    if not successor:
        for key in (
            "original_protected_minimum_item_receipt_sha256s",
            "post_admission_dedup_receipt_sha256",
            "protected_minimum_item_transfer_rows",
        ):
            audit_body.pop(key)
    audit = _sealed_audit(
        audit_body,
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
    format_id: str = FORMAT

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
        _require(
            self.format_id in {FORMAT, POST_DEDUP_BACKFILL_FORMAT},
            "additive composition format changed",
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
        expected_dedup_format = (
            DEDUP_FORMAT
            if self.format_id == FORMAT
            else DEDUP_BACKFILL_FORMAT
        )
        expected_fair_format = (
            FAIR_MERGE_FORMAT
            if self.format_id == FORMAT
            else POST_DEDUP_BACKFILL_FAIR_MERGE_FORMAT
        )
        _require(
            self.post_selection_dedup_audit.get("format")
            == expected_dedup_format
            and self.fair_merge_audit.get("format") == expected_fair_format
            and (
                self.format_id == FORMAT
                or "initial_post_selection_dedup_audit"
                in self.post_selection_dedup_audit
            ),
            "additive format does not identify its construction semantics",
        )
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
            "format": self.format_id,
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
                    "format": f"{self.format_id}-retained-local-priority-v1",
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
    provider_payload_mode: ProviderPayloadMode = ProviderPayloadMode.COMPACT_FINAL,
    composition_mode: CompositionMode = LEGACY_COMPOSITION_MODE,
) -> AdditiveTypedComposition:
    """Compose independently selected mechanism outputs into one typed packet.

    ``dedup_owner_priority_by_mechanism`` is mandatory and higher values own
    proven duplicates.  This makes protected-parent ownership an explicit
    caller policy instead of a mechanism-name heuristic.  When omitted, fair
    merge priorities reuse that same explicit map.
    """

    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be exact")
    if type(provider_payload_mode) is not ProviderPayloadMode:
        raise TypeError("provider_payload_mode must be exact")
    if composition_mode not in {
        LEGACY_COMPOSITION_MODE,
        POST_DEDUP_BACKFILL_COMPOSITION_MODE,
    }:
        raise ValueError("unsupported additive composition mode")
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

    if composition_mode == LEGACY_COMPOSITION_MODE:
        if provider_payload_mode is not ProviderPayloadMode.COMPACT_FINAL:
            raise ValueError(
                "legacy_v1 requires the historical compact-final provider mode"
            )
        deduplicated, dedup_audit = deduplicate_selected_contributions(
            contributions,
            owner_priority_by_mechanism=dedup_priorities,
            exact_span_keys_by_handle=exact_span_keys_by_handle,
        )
        deduplicated_handles = {
            row.handle_id
            for contribution in deduplicated
            for row in contribution.bindings
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
                not any(
                    _usable_item(item, operator_spec)
                    for item in original.parsed.accepted_items
                )
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
            row.handle_id
            for contribution in expanded
            for row in contribution.bindings
        }
        expanded_priorities = {
            handle: priority
            for handle, priority in retained_priorities.items()
            if handle in expanded_handles
        }
        packet, fair_audit = _fair_merge_contributions(
            operator_spec,
            expanded,
            original_protected_item_receipt_sha256s=None,
            protected_item_receipt_sha256s=protected_receipts,
            minimum_allocation_receipt_sha256=minimum.receipt_sha256,
            surplus_fill_audit=surplus_audit,
            post_selection_dedup_audit=None,
            mechanism_priority_by_mechanism=merge_priorities,
            local_selection_priority_by_handle=expanded_priorities,
            provider_payload_mode=provider_payload_mode,
            composition_mode=composition_mode,
        )
        mechanism_by_handle, dropped = retained_mechanism_bindings(
            expanded, packet
        )
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
            format_id=FORMAT,
        )

    # Each mechanism first spends its own protected allowance.  Physical
    # deduplication before this point is non-monotonic: a preferred owner can
    # later miss its lane while the alternate representation has already been
    # erased.  Union the admitted populations first, then choose one exact
    # representative and transfer any lane-minimum authority to that survivor.
    minimum = allocate_typed_contribution_lanes(
        contributions,
        lane_budgets=lane_budgets,
        lane_by_mechanism=lane_by_mechanism,
        operator_spec=operator_spec,
        local_selection_priority_by_handle=local_priorities,
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
            for original in contributions
        ),
        "non-borrowable lane cap starved a nonempty additive mechanism",
    )
    original_protected_receipts = tuple(
        receipt
        for lane_receipt in minimum.receipts
        for receipt in lane_receipt.selected_item_receipt_sha256s
    )
    _require(
        bool(original_protected_receipts),
        "additive composition requires at least one usable lane minimum",
    )
    admitted, surplus_audit = fill_typed_lane_surplus(
        contributions,
        minimum,
        lane_budgets=lane_budgets,
        lane_by_mechanism=lane_by_mechanism,
        operator_spec=operator_spec,
        local_selection_priority_by_handle=local_priorities,
    )
    admitted_handles = {
        row.handle_id for contribution in admitted for row in contribution.bindings
    }
    admitted_span_keys = {
        handle: values
        for handle, values in dict(exact_span_keys_by_handle or {}).items()
        if handle in admitted_handles
    }
    initial_expanded, initial_dedup_audit = deduplicate_selected_contributions(
        admitted,
        owner_priority_by_mechanism=dedup_priorities,
        exact_span_keys_by_handle=admitted_span_keys,
        operation_position=(
            "after_independent_lane_admission_and_shared_surplus_fill"
        ),
        audit_format=POST_DEDUP_BACKFILL_DEDUP_FORMAT,
    )
    expanded, dedup_audit = _backfill_freed_dedup_capacity(
        operator_spec,
        contributions,
        initial_expanded,
        lane_budgets=lane_budgets,
        lane_by_mechanism=lane_by_mechanism,
        exact_span_keys_by_handle=exact_span_keys_by_handle,
        surplus_fill_audit=surplus_audit,
        initial_dedup_audit=initial_dedup_audit,
    )
    expanded_handles = {
        row.handle_id for contribution in expanded for row in contribution.bindings
    }
    expanded_priorities = {
        handle: priority
        for handle, priority in local_priorities.items()
        if handle in expanded_handles
    }
    retained_item_receipts = {
        item.receipt_sha256
        for contribution in expanded
        for item in contribution.parsed.accepted_items
    }
    exclusion_by_duplicate = {
        row["duplicate_item_receipt_sha256"]: row
        for row in dedup_audit["exclusions"]
    }
    effective_protected: list[str] = []
    for receipt in original_protected_receipts:
        effective = receipt
        if receipt not in retained_item_receipts:
            exclusion = exclusion_by_duplicate.get(receipt)
            _require(
                exclusion is not None,
                "protected minimum disappeared outside exact dedup",
            )
            effective = exclusion["owner_item_receipt_sha256"]
        _require(
            effective in retained_item_receipts,
            "protected dedup equivalence class lost every representative",
        )
        if effective not in effective_protected:
            effective_protected.append(effective)
    protected_receipts = tuple(effective_protected)
    packet, fair_audit = _fair_merge_contributions(
        operator_spec,
        expanded,
        original_protected_item_receipt_sha256s=(
            original_protected_receipts
        ),
        protected_item_receipt_sha256s=protected_receipts,
        minimum_allocation_receipt_sha256=minimum.receipt_sha256,
        surplus_fill_audit=surplus_audit,
        post_selection_dedup_audit=dedup_audit,
        mechanism_priority_by_mechanism=merge_priorities,
        local_selection_priority_by_handle=expanded_priorities,
        provider_payload_mode=provider_payload_mode,
        composition_mode=composition_mode,
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
        format_id=POST_DEDUP_BACKFILL_FORMAT,
    )


__all__ = [
    "AdditiveTypedComposition",
    "CompositionMode",
    "DEDUP_BACKFILL_FORMAT",
    "DEDUP_FORMAT",
    "FAIR_MERGE_FORMAT",
    "FORMAT",
    "LEGACY_COMPOSITION_MODE",
    "POST_DEDUP_BACKFILL_COMPOSITION_MODE",
    "POST_DEDUP_BACKFILL_DEDUP_FORMAT",
    "POST_DEDUP_BACKFILL_FAIR_MERGE_FORMAT",
    "POST_DEDUP_BACKFILL_FORMAT",
    "compose_additive_typed_evidence",
    "deduplicate_selected_contributions",
    "retained_mechanism_bindings",
]
