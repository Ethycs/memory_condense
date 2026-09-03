"""Rehydrate a fitted parent packet as protected additive contributions.

The fitted compact packet is already the result of independent parent
mechanisms, but its retained bindings still point at several source artifacts.
``TypedEvidenceContribution`` deliberately forbids that shape: every binding
inside one contribution must name the same sealed artifact.  This adapter keeps
the parent H/G identifiers and provider-visible evidence bytes unchanged while
cloning each mechanism's bindings onto a deterministic wrapper artifact.

No source text is rendered and no provider is called.  Original binding
projections, local source joins, exact-span keys, and available local selection
priorities remain in a sealed, gold-blind audit outside the provider packet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from .typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    ProvenanceGrade,
    TypedEvidenceContribution,
    compact_evidence_content_projection,
    parse_typed_items,
)
from .typed_operator_spec import TypedOperatorSpec


MECHANISM_ID = "protected_parent"
RESULT_FORMAT = "memory-condense-protected-parent-contribution-set-v1"
AUDIT_FORMAT = "memory-condense-protected-parent-contribution-audit-v1"
GROUP_AUDIT_FORMAT = f"{AUDIT_FORMAT}-group"
HANDLE_AUDIT_FORMAT = f"{AUDIT_FORMAT}-handle-provenance"
WRAPPER_ARTIFACT_FORMAT = (
    "memory-condense-protected-parent-binding-wrapper-artifact-v1"
)
_PARENT_FINAL_FORMAT = "memory-condense-locked-typed-memory-final-arm-v1"
_DERIVED_COMPACT_ITEM_KEYS = frozenset(
    {"content_coherence", "supported_slot_ids"}
)


class ProtectedParentContributionError(MatchedEvalContractError):
    """Raised when fitted-parent rehydration loses an exact invariant."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ProtectedParentContributionError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact list")
    return value  # type: ignore[return-value]


def _ordered_text(value: object, label: str) -> tuple[str, ...]:
    rows = _exact_list(value, label)
    _require(
        all(type(row) is str and row and row.strip() == row for row in rows)
        and len(rows) == len(set(rows)),
        f"{label} must be ordered unique nonempty text",
    )
    return tuple(rows)


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _rehydrate_handle_binding(raw: object) -> EvidenceHandleBinding:
    """Rebuild one exact typed binding without importing an assay runner."""

    row = _exact_dict(raw, "retained evidence binding")
    binding = EvidenceHandleBinding(
        handle_id=require_text(row.get("handle_id"), "retained handle"),
        origin=EvidenceOrigin(require_text(row.get("origin"), "retained origin")),
        provenance_grade=ProvenanceGrade(
            require_text(row.get("provenance_grade"), "retained provenance")
        ),
        source_group_handle=require_text(
            row.get("source_group_handle"), "retained group"
        ),
        sealed_artifact_sha256=require_sha256(
            row.get("sealed_artifact_sha256"), "retained artifact"
        ),
        parent_receipt_sha256=require_sha256(
            row.get("parent_receipt_sha256"), "retained parent"
        ),
        evidence_receipt_sha256=require_sha256(
            row.get("evidence_receipt_sha256"), "retained evidence"
        ),
        payload_sha256=require_sha256(
            row.get("payload_sha256"), "retained payload"
        ),
        citation_sha256=require_sha256(
            row.get("citation_sha256"), "retained citation"
        ),
        citation_char_count=row.get("citation_char_count"),
        local_source_locator_sha256=require_sha256(
            row.get("local_source_locator_sha256"), "retained locator"
        ),
        receipt_sha256=require_sha256(
            row.get("receipt_sha256"), "retained receipt"
        ),
    )
    _require(binding.projection() == row, "retained evidence binding changed")
    return binding


def _compact_operator_projection(spec: TypedOperatorSpec) -> dict[str, Any]:
    """Mirror the compact provider projection without receipt-bearing fields."""

    return {
        "absence_decision_requires_closed_frontier": (
            spec.absence_decision_requires_closed_frontier
        ),
        "answer_shape": spec.answer_shape.value,
        "cardinality": spec.cardinality,
        "comparison_mode": spec.comparison_mode.value,
        "include_proposed": spec.include_proposed,
        "operation": spec.operation,
        "ordering": spec.ordering,
        "personalization_required": spec.personalization_required,
        "query_timestamp": spec.query_timestamp,
        "required_evidence_role": spec.required_evidence_role,
        "required_slots": [
            {
                "kind": slot.kind.value,
                "label": slot.label,
                "match_terms": list(slot.match_terms),
                "minimum_match_term_count": slot.minimum_match_term_count,
                "relation_constraint": slot.relation_constraint,
                "requires_numeric": slot.requires_numeric,
                "slot_id": slot.slot_id,
            }
            for slot in spec.required_slots
        ],
        "requires_all_slots": spec.requires_all_slots,
        "requires_complete_frontier": spec.requires_complete_frontier,
        "specificity_required": spec.specificity_required,
        "style": spec.style.value,
        "temporal_mode": spec.temporal_mode.value,
        "temporal_window_days": spec.temporal_window_days,
    }


def _handle_prefix(handle_id: str) -> int:
    _require(
        handle_id.startswith("H") and handle_id[1:].isdigit(),
        "protected parent handle prefix changed",
    )
    return int(handle_id[1:]) // 100_000


def _safe_frontier(
    raw: object,
    *,
    handle_ids: tuple[str, ...],
    represented_handle_ids: tuple[str, ...],
) -> tuple[FrontierMode, bool, bool]:
    """Reuse a compact frontier only when its complete bookkeeping is sound."""

    if type(raw) is not dict:
        return FrontierMode.BOUNDED, True, False
    try:
        mode = FrontierMode(raw.get("mode"))
    except (TypeError, ValueError):
        return FrontierMode.BOUNDED, True, False
    truncated = raw.get("truncated")
    closed = raw.get("closed")
    rejected_count = raw.get("rejected_item_count")
    try:
        available = _ordered_text(
            raw.get("available_handle_ids"), "parent frontier available handles"
        )
        represented = _ordered_text(
            raw.get("represented_handle_ids"),
            "parent frontier represented handles",
        )
        omitted = _ordered_text(
            raw.get("omitted_handle_ids"), "parent frontier omitted handles"
        )
        _ordered_text(
            raw.get("unresolved_slot_ids"), "parent frontier unresolved slots"
        )
    except ProtectedParentContributionError:
        return FrontierMode.BOUNDED, True, False
    expected_omitted = tuple(row for row in available if row not in represented)
    safe = (
        type(truncated) is bool
        and type(closed) is bool
        and type(rejected_count) is int
        and rejected_count >= 0
        and available == handle_ids
        and represented == represented_handle_ids
        and omitted == expected_omitted
        and closed == (mode is FrontierMode.EXHAUSTIVE)
    )
    if mode is FrontierMode.EXHAUSTIVE:
        safe = safe and not truncated and not omitted and rejected_count == 0
    if not safe:
        return FrontierMode.BOUNDED, True, False
    return mode, truncated, True


def _wrapper_sha256(
    *,
    parent_artifact_sha256: str,
    composition_row_sha256: str,
    mechanism_id: str,
    handle_ids: tuple[str, ...],
    binding_receipt_sha256s: tuple[str, ...],
    compact_group_sha256: str,
) -> str:
    return identity_sha256(
        {
            "binding_receipt_sha256s": list(binding_receipt_sha256s),
            "compact_group_sha256": compact_group_sha256,
            "composition_row_sha256": composition_row_sha256,
            "format": WRAPPER_ARTIFACT_FORMAT,
            "handle_ids": list(handle_ids),
            "parent_composition_artifact_sha256": parent_artifact_sha256,
            "parent_mechanism_id": mechanism_id,
        }
    )


def _clone_binding(
    binding: EvidenceHandleBinding,
    *,
    wrapper_artifact_sha256: str,
) -> EvidenceHandleBinding:
    return EvidenceHandleBinding(
        handle_id=binding.handle_id,
        origin=binding.origin,
        provenance_grade=binding.provenance_grade,
        source_group_handle=binding.source_group_handle,
        sealed_artifact_sha256=wrapper_artifact_sha256,
        parent_receipt_sha256=binding.parent_receipt_sha256,
        evidence_receipt_sha256=binding.evidence_receipt_sha256,
        payload_sha256=binding.payload_sha256,
        citation_sha256=binding.citation_sha256,
        citation_char_count=binding.citation_char_count,
        local_source_locator_sha256=binding.local_source_locator_sha256,
    )


def _canonical_coordinate_span_key(local: Mapping[str, Any]) -> str | None:
    try:
        namespace_id = require_sha256(local.get("namespace_id"), "span namespace")
        source_id = require_text(local.get("source_id"), "span source")
        quote = require_sha256(local.get("quote_sha256"), "span quote")
        span = _exact_dict(local.get("span"), "local span")
        chunk_id = require_text(span.get("chunk_id"), "span chunk")
        start = span.get("start_char")
        end = span.get("end_char")
        _require(
            type(start) is int
            and type(end) is int
            and 0 <= start < end,
            "span coordinates changed",
        )
    except (MatchedEvalContractError, TypeError):
        return None
    return identity_sha256(
        {
            "chunk_id": chunk_id,
            "end_char": end,
            "format": f"{_PARENT_FINAL_FORMAT}-canonical-coordinate-span-v1",
            "namespace_id": namespace_id,
            "quote_sha256": quote,
            "source_id": source_id,
            "start_char": start,
        }
    )


def _origin_coordinate_span_key(origin: Mapping[str, Any]) -> str | None:
    local = {
        "namespace_id": origin.get("namespace_id"),
        "quote_sha256": origin.get("quote_sha256"),
        "source_id": origin.get("source_id"),
        "span": {
            "chunk_id": origin.get("chunk_id"),
            "end_char": origin.get("quote_end_char"),
            "start_char": origin.get("quote_start_char"),
        },
    }
    return _canonical_coordinate_span_key(local)


def _canonical_native_evidence_key(
    *,
    namespace_id: object,
    source_id: object,
    evidence_id: object,
    quote_sha256: object,
) -> str | None:
    try:
        namespace = require_sha256(namespace_id, "evidence namespace")
        source = require_text(source_id, "evidence source")
        evidence = require_text(evidence_id, "native evidence ID")
        quote = require_sha256(quote_sha256, "evidence quote")
    except (MatchedEvalContractError, TypeError):
        return None
    return identity_sha256(
        {
            "evidence_id": evidence,
            "format": f"{_PARENT_FINAL_FORMAT}-canonical-native-evidence-v1",
            "namespace_id": namespace,
            "quote_sha256": quote,
            "source_id": source,
        }
    )


def _exact_span_keys(
    local_audit: Mapping[str, Any],
    bindings: tuple[EvidenceHandleBinding, ...],
) -> dict[str, tuple[str, ...]]:
    """Conservatively reconstruct only exact keys present in sealed lineage."""

    by_handle = {row.handle_id: row for row in bindings}
    direct_by_id: dict[str, dict[str, Any]] = {}
    exclusions_by_receipt: dict[str, dict[str, Any]] = {}
    fact_by_binding: dict[str, dict[str, Any]] = {}
    namespace_ids: set[str] = set()
    for audit_key in ("adaptive_parent_source", "adaptive_tail_source"):
        raw = local_audit.get(audit_key)
        if type(raw) is not dict:
            continue
        for value in raw.get("direct_evidence", []):
            if type(value) is dict and type(value.get("evidence_id")) is str:
                direct_by_id.setdefault(value["evidence_id"], value)
                if type(value.get("namespace_id")) is str:
                    namespace_ids.add(value["namespace_id"])
        for value in raw.get("direct_exclusions", []):
            if type(value) is dict and type(value.get("receipt_sha256")) is str:
                exclusions_by_receipt.setdefault(value["receipt_sha256"], value)
        for value in raw.get("source_fact_admission_bindings", []):
            if (
                type(value) is dict
                and type(value.get("binding_receipt_sha256")) is str
            ):
                fact_by_binding.setdefault(value["binding_receipt_sha256"], value)
                for origin in value.get("exact_origins", []):
                    if type(origin) is dict and type(origin.get("namespace_id")) is str:
                        namespace_ids.add(origin["namespace_id"])

    full = local_audit.get("full_store_slot_closure")
    if type(full) is dict:
        for value in full.get("local_citation_bindings", []):
            if type(value) is dict:
                local = value.get("local_citation_binding")
                if type(local) is dict and type(local.get("namespace_id")) is str:
                    namespace_ids.add(local["namespace_id"])
    active = local_audit.get("active_reconstruction")
    if type(active) is dict:
        local_result = active.get("local_result")
        if type(local_result) is dict:
            for local in local_result.get("local_bindings", []):
                if type(local) is dict and type(local.get("namespace_id")) is str:
                    namespace_ids.add(local["namespace_id"])
    unique_namespace = next(iter(namespace_ids)) if len(namespace_ids) == 1 else None

    found: dict[str, list[str]] = {}
    map_audit = local_audit.get("adaptive_parent_map")
    if type(map_audit) is dict:
        for value in map_audit.get("exact_item_bindings", []):
            if type(value) is not dict:
                continue
            raw_binding = value.get("binding")
            alias = value.get("payload_alias")
            if type(raw_binding) is not dict or type(alias) is not dict:
                continue
            handle = raw_binding.get("handle_id")
            direct = direct_by_id.get(alias.get("evidence_id"))
            namespace_id = (
                direct.get("namespace_id") if direct is not None else unique_namespace
            )
            if handle not in by_handle or namespace_id is None:
                continue
            key = _canonical_native_evidence_key(
                namespace_id=namespace_id,
                source_id=alias.get("source_id"),
                evidence_id=alias.get("evidence_id"),
                quote_sha256=by_handle[handle].citation_sha256,
            )
            if key is not None:
                found.setdefault(handle, []).append(key)

    for binding in bindings:
        fact = fact_by_binding.get(binding.receipt_sha256)
        if fact is not None:
            for origin in fact.get("exact_origins", []):
                if type(origin) is dict:
                    key = _origin_coordinate_span_key(origin)
                    if key is not None:
                        found.setdefault(binding.handle_id, []).append(key)
        exclusion = exclusions_by_receipt.get(binding.evidence_receipt_sha256)
        if exclusion is not None:
            for evidence_id in exclusion.get("matching_direct_evidence_ids", []):
                direct = direct_by_id.get(evidence_id)
                if direct is None:
                    continue
                key = _canonical_native_evidence_key(
                    namespace_id=direct.get("namespace_id"),
                    source_id=direct.get("source_id"),
                    evidence_id=direct.get("evidence_id"),
                    quote_sha256=direct.get("quote_sha256"),
                )
                if key is not None:
                    found.setdefault(binding.handle_id, []).append(key)

    if type(full) is dict:
        for value in full.get("local_citation_bindings", []):
            if type(value) is not dict:
                continue
            handle = value.get("handle_id")
            local = value.get("local_citation_binding")
            if handle not in by_handle or type(local) is not dict:
                continue
            key = _canonical_coordinate_span_key(local)
            if key is not None:
                found.setdefault(handle, []).append(key)

    if type(active) is dict:
        contribution = active.get("contribution")
        local_result = active.get("local_result")
        if type(contribution) is dict and type(local_result) is dict:
            receipts = contribution.get("binding_receipt_sha256s", [])
            locals_ = local_result.get("local_bindings", [])
            if type(receipts) is list and type(locals_) is list:
                handle_by_receipt = {
                    row.receipt_sha256: row.handle_id for row in bindings
                }
                for receipt, local in zip(receipts, locals_, strict=False):
                    handle = handle_by_receipt.get(receipt)
                    if handle is None or type(local) is not dict:
                        continue
                    key = _canonical_coordinate_span_key(local)
                    if key is not None:
                        found.setdefault(handle, []).append(key)
    return {
        handle: _ordered_unique(values)
        for handle, values in found.items()
        if values
    }


def _local_priorities(
    local_audit: Mapping[str, Any],
    *,
    known_handles: frozenset[str],
) -> dict[str, tuple[int, ...]]:
    audit = local_audit.get("active_full_selection_priority")
    if type(audit) is not dict or type(audit.get("rows")) is not list:
        return {}
    result: dict[str, tuple[int, ...]] = {}
    for value in audit["rows"]:
        if type(value) is not dict:
            continue
        handle = value.get("handle_id")
        raw = value.get("local_selection_priority")
        if (
            handle not in known_handles
            or type(raw) is not list
            or not raw
            or any(type(row) is not int for row in raw)
        ):
            continue
        result[handle] = tuple(raw)
    return result


@dataclass(frozen=True, slots=True)
class ProtectedParentHandleProvenance:
    handle_id: str
    parent_mechanism_id: str
    source_ids: tuple[str, ...]
    original_binding: EvidenceHandleBinding
    cloned_binding: EvidenceHandleBinding
    exact_span_keys: tuple[str, ...] = ()
    local_selection_priority: tuple[int, ...] = ()

    def projection(self) -> dict[str, Any]:
        return {
            "cloned_binding": self.cloned_binding.projection(),
            "exact_span_receipt_sha256s": list(self.exact_span_keys),
            "format": HANDLE_AUDIT_FORMAT,
            "handle_id": self.handle_id,
            "local_selection_priority": list(self.local_selection_priority),
            "original_binding": self.original_binding.projection(),
            "parent_mechanism_id": self.parent_mechanism_id,
            "source_ids": list(self.source_ids),
        }


@dataclass(frozen=True, slots=True)
class ProtectedParentGroupAudit:
    parent_mechanism_id: str
    wrapper_artifact_sha256: str
    handle_ids: tuple[str, ...]
    item_count: int
    original_binding_receipt_sha256s: tuple[str, ...]
    cloned_binding_receipt_sha256s: tuple[str, ...]

    def projection(self) -> dict[str, Any]:
        return {
            "cloned_binding_receipt_sha256s": list(
                self.cloned_binding_receipt_sha256s
            ),
            "format": GROUP_AUDIT_FORMAT,
            "handle_ids": list(self.handle_ids),
            "item_count": self.item_count,
            "original_binding_receipt_sha256s": list(
                self.original_binding_receipt_sha256s
            ),
            "parent_mechanism_id": self.parent_mechanism_id,
            "wrapper_artifact_sha256": self.wrapper_artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProtectedParentAudit:
    question_sha256: str
    parent_composition_artifact_sha256: str
    composition_row_sha256: str
    compact_projection_sha256: str
    reconstructed_compact_projection_sha256: str
    compact_projection_byte_identical: bool
    frontier_mode: FrontierMode
    truncated: bool
    parent_frontier_reused: bool
    source_ids: tuple[str, ...]
    compact_item_receipt_order: tuple[str, ...]
    groups: tuple[ProtectedParentGroupAudit, ...]
    source_provenance: tuple[ProtectedParentHandleProvenance, ...]
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.question_sha256, "protected parent question")
        require_sha256(
            self.parent_composition_artifact_sha256,
            "protected parent composition artifact",
        )
        require_sha256(self.composition_row_sha256, "protected parent row")
        require_sha256(self.compact_projection_sha256, "parent compact projection")
        require_sha256(
            self.reconstructed_compact_projection_sha256,
            "reconstructed compact projection",
        )
        _require(
            self.compact_projection_byte_identical
            and self.compact_projection_sha256
            == self.reconstructed_compact_projection_sha256,
            "protected parent compact bytes changed",
        )
        _require(
            type(self.frontier_mode) is FrontierMode
            and type(self.truncated) is bool
            and type(self.parent_frontier_reused) is bool,
            "protected parent frontier audit changed",
        )
        _require(
            type(self.groups) is tuple
            and all(type(row) is ProtectedParentGroupAudit for row in self.groups)
            and type(self.source_provenance) is tuple
            and all(
                type(row) is ProtectedParentHandleProvenance
                for row in self.source_provenance
            ),
            "protected parent audit rows changed type",
        )
        _require(
            type(self.compact_item_receipt_order) is tuple
            and all(
                require_sha256(row, "protected parent compact item") == row
                for row in self.compact_item_receipt_order
            )
            and len(self.compact_item_receipt_order)
            == len(set(self.compact_item_receipt_order)),
            "protected parent compact item order changed",
        )
        _require(
            self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "protected parent audit must be provider-free, zero-state, and gold-blind",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "protected parent audit changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="protected_parent_audit")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "compact_projection_byte_identical": (
                self.compact_projection_byte_identical
            ),
            "compact_projection_sha256": self.compact_projection_sha256,
            "compact_item_receipt_order": list(
                self.compact_item_receipt_order
            ),
            "composition_row_sha256": self.composition_row_sha256,
            "format": AUDIT_FORMAT,
            "frontier_mode": self.frontier_mode.value,
            "gold_loaded": False,
            "groups": [row.projection() for row in self.groups],
            "mechanism_id": MECHANISM_ID,
            "parent_composition_artifact_sha256": (
                self.parent_composition_artifact_sha256
            ),
            "parent_frontier_reused": self.parent_frontier_reused,
            "provider_prompt_count": 0,
            "question_sha256": self.question_sha256,
            "reconstructed_compact_projection_sha256": (
                self.reconstructed_compact_projection_sha256
            ),
            "retained_transformer_token_state_bytes": 0,
            "source_ids": list(self.source_ids),
            "source_provenance": [
                row.projection() for row in self.source_provenance
            ],
            "truncated": self.truncated,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ProtectedParentContributionSet:
    contributions: tuple[TypedEvidenceContribution, ...]
    audit: ProtectedParentAudit

    def __post_init__(self) -> None:
        _require(
            type(self.contributions) is tuple
            and self.contributions
            and all(
                type(row) is TypedEvidenceContribution
                for row in self.contributions
            ),
            "protected parent contributions changed type",
        )
        _require(
            len({row.mechanism_id for row in self.contributions})
            == len(self.contributions),
            "protected parent mechanisms repeat",
        )

    @property
    def source_ids(self) -> tuple[str, ...]:
        return self.audit.source_ids

    @property
    def exact_span_keys_by_handle(self) -> dict[str, tuple[str, ...]]:
        return {
            row.handle_id: row.exact_span_keys
            for row in self.audit.source_provenance
            if row.exact_span_keys
        }

    @property
    def local_selection_priority_by_handle(self) -> dict[str, tuple[int, ...]]:
        return {
            row.handle_id: row.local_selection_priority
            for row in self.audit.source_provenance
            if row.local_selection_priority
        }

    def compact_content_projection(self) -> dict[str, Any]:
        """Rebuild parent content in its sealed pre-grouping item order."""

        item_by_receipt = {
            item.receipt_sha256: item
            for contribution in self.contributions
            for item in contribution.parsed.accepted_items
        }
        _require(
            set(item_by_receipt) == set(self.audit.compact_item_receipt_order),
            "protected parent compact item order lost contribution items",
        )
        bindings = tuple(
            row.cloned_binding for row in self.audit.source_provenance
        )
        items = tuple(
            item_by_receipt[receipt]
            for receipt in self.audit.compact_item_receipt_order
        )
        return compact_evidence_content_projection(items, bindings)

    def projection(self) -> dict[str, Any]:
        value = {
            "audit_receipt_sha256": self.audit.receipt_sha256,
            "contribution_receipt_sha256s": [
                row.receipt_sha256 for row in self.contributions
            ],
            "format": RESULT_FORMAT,
            "mechanism_id": MECHANISM_ID,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
        }
        assert_gold_blind(value, path="protected_parent_contribution_set")
        return value


def rehydrate_protected_parent_contributions(
    composition_row: Mapping[str, Any],
    operator_spec: TypedOperatorSpec,
    parent_composition_artifact_sha256: str,
    /,
) -> ProtectedParentContributionSet:
    """Return fitted parent evidence partitioned by its original mechanisms.

    Compact items are parsed only after removing the two fields derived by the
    typed parser.  The reassembled compact packet must then be canonical-byte
    identical to the parent packet, including summaries and typed semantics.
    """

    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be exact")
    parent_sha = require_sha256(
        parent_composition_artifact_sha256, "parent composition artifact"
    )
    row = _exact_dict(composition_row, "composition row")
    row_sha = require_sha256(
        row.get("composition_row_sha256"), "composition row"
    )
    unsigned_row = dict(row)
    unsigned_row.pop("composition_row_sha256")
    _require(identity_sha256(unsigned_row) == row_sha, "composition row changed")
    _require(
        row.get("dated_question_sha256") == operator_spec.question_sha256,
        "composition row and operator question diverged",
    )

    provider = _exact_dict(row.get("provider_projection"), "provider projection")
    provider_input = _exact_dict(
        provider.get("provider_input"), "provider input"
    )
    compact = _exact_dict(
        provider_input.get("typed_evidence"), "compact typed evidence"
    )
    _require(
        compact.get("operator_spec") == _compact_operator_projection(operator_spec),
        "compact operator spec changed",
    )
    compact_handles = _exact_list(compact.get("handles"), "compact handles")
    compact_items = _exact_list(compact.get("items"), "compact items")
    handle_ids = tuple(
        require_text(
            _exact_dict(value, "compact handle").get("handle_id"),
            "compact handle ID",
        )
        for value in compact_handles
    )
    _require(
        bool(handle_ids) and len(handle_ids) == len(set(handle_ids)),
        "compact handles must be nonempty and unique",
    )
    allowed = _ordered_text(row.get("allowed_handle_ids"), "allowed handles")
    _require(
        len(allowed) == len(handle_ids) and set(allowed) == set(handle_ids),
        "allowed handles diverged from compact handles",
    )
    mechanism_by_handle = _exact_dict(
        row.get("mechanism_by_handle"), "mechanism by handle"
    )
    group_by_handle = _exact_dict(row.get("handle_group_by_id"), "handle groups")
    _require(
        set(mechanism_by_handle) == set(handle_ids)
        and set(group_by_handle) == set(handle_ids),
        "parent handle metadata coverage changed",
    )

    mechanisms: list[str] = []
    prefix_by_mechanism: dict[str, int] = {}
    mechanism_by_prefix: dict[int, str] = {}
    for handle in handle_ids:
        mechanism = require_text(
            mechanism_by_handle.get(handle), "parent mechanism ID"
        )
        group = require_text(group_by_handle.get(handle), "parent group handle")
        prefix = _handle_prefix(handle)
        _require(0 <= prefix <= 6, "protected parent handle escaped prefixes 0-6")
        _require(
            group.startswith("G")
            and group[1:].isdigit()
            and int(group[1:]) // 100_000 == prefix,
            "protected parent H/G prefixes diverged",
        )
        prior_prefix = prefix_by_mechanism.setdefault(mechanism, prefix)
        prior_mechanism = mechanism_by_prefix.setdefault(prefix, mechanism)
        _require(
            prior_prefix == prefix and prior_mechanism == mechanism,
            "parent mechanisms and H/G prefixes are not one-to-one",
        )
        if mechanism not in mechanisms:
            mechanisms.append(mechanism)

    local_audit = _exact_dict(row.get("local_audit"), "parent local audit")
    retained = _exact_list(
        local_audit.get("retained_fitted_bindings"),
        "retained fitted bindings",
    )
    original_bindings = tuple(
        _rehydrate_handle_binding(value) for value in retained
    )
    _require(
        tuple(row.handle_id for row in original_bindings) == handle_ids,
        "retained binding order diverged from compact handles",
    )
    for raw_handle, binding in zip(compact_handles, original_bindings, strict=True):
        handle = _exact_dict(raw_handle, "compact handle")
        _require(
            handle
            == {
                "group_handle": binding.source_group_handle,
                "handle_id": binding.handle_id,
                "origin": binding.origin.value,
                "provenance_grade": binding.provenance_grade.value,
            },
            "compact handle semantics changed",
        )
        _require(
            binding.source_group_handle == group_by_handle[binding.handle_id],
            "parent group metadata diverged from binding",
        )

    item_mechanisms: list[str] = []
    represented: list[str] = []
    for value in compact_items:
        item = _exact_dict(value, "compact item")
        cited = _ordered_text(item.get("handle_ids"), "compact item handles")
        _require(set(cited) <= set(handle_ids), "compact item escaped parent handles")
        item_groups = {mechanism_by_handle[handle] for handle in cited}
        _require(
            len(item_groups) == 1,
            "compact item crossed protected parent mechanism boundaries",
        )
        item_mechanisms.append(next(iter(item_groups)))
        represented.extend(cited)
    represented_ids = tuple(row for row in handle_ids if row in set(represented))
    frontier_mode, truncated, frontier_reused = _safe_frontier(
        compact.get("frontier"),
        handle_ids=handle_ids,
        represented_handle_ids=represented_ids,
    )

    compact_bytes = canonical_json_bytes(compact)
    item_by_ordinal: dict[int, Any] = {}
    cloned_by_handle: dict[str, EvidenceHandleBinding] = {}
    contributions: list[TypedEvidenceContribution] = []
    group_audits: list[ProtectedParentGroupAudit] = []
    for mechanism in mechanisms:
        group_bindings = tuple(
            row
            for row in original_bindings
            if mechanism_by_handle[row.handle_id] == mechanism
        )
        item_ordinals = tuple(
            index
            for index, item_mechanism in enumerate(item_mechanisms)
            if item_mechanism == mechanism
        )
        raw_group_items = [compact_items[index] for index in item_ordinals]
        compact_group = {
            "handles": [
                compact_handles[handle_ids.index(binding.handle_id)]
                for binding in group_bindings
            ],
            "items": raw_group_items,
        }
        wrapper = _wrapper_sha256(
            parent_artifact_sha256=parent_sha,
            composition_row_sha256=row_sha,
            mechanism_id=mechanism,
            handle_ids=tuple(row.handle_id for row in group_bindings),
            binding_receipt_sha256s=tuple(
                row.receipt_sha256 for row in group_bindings
            ),
            compact_group_sha256=identity_sha256(compact_group),
        )
        cloned = tuple(
            _clone_binding(row, wrapper_artifact_sha256=wrapper)
            for row in group_bindings
        )
        parse_input = [
            {
                key: child
                for key, child in _exact_dict(value, "compact group item").items()
                if key not in _DERIVED_COMPACT_ITEM_KEYS
            }
            for value in raw_group_items
        ]
        parsed = parse_typed_items(
            parse_input,
            operator_spec=operator_spec,
            bindings=cloned,
        )
        _require(
            not parsed.rejected_items
            and len(parsed.accepted_items) == len(raw_group_items),
            f"protected parent items failed typed parsing for {mechanism}",
        )
        contribution = TypedEvidenceContribution(
            mechanism_id=mechanism,
            bindings=cloned,
            parsed=parsed,
            sealed_artifact_sha256=wrapper,
            frontier_mode=frontier_mode,
            truncated=truncated,
        )
        contributions.append(contribution)
        cloned_by_handle.update({row.handle_id: row for row in cloned})
        item_by_ordinal.update(zip(item_ordinals, parsed.accepted_items, strict=True))
        group_audits.append(
            ProtectedParentGroupAudit(
                parent_mechanism_id=mechanism,
                wrapper_artifact_sha256=wrapper,
                handle_ids=tuple(row.handle_id for row in cloned),
                item_count=len(parsed.accepted_items),
                original_binding_receipt_sha256s=tuple(
                    row.receipt_sha256 for row in group_bindings
                ),
                cloned_binding_receipt_sha256s=tuple(
                    row.receipt_sha256 for row in cloned
                ),
            )
        )

    ordered_items = tuple(item_by_ordinal[index] for index in range(len(compact_items)))
    ordered_clones = tuple(cloned_by_handle[handle] for handle in handle_ids)
    rebuilt_content = compact_evidence_content_projection(
        ordered_items, ordered_clones
    )
    reconstructed = {
        "conflict_policy": compact.get("conflict_policy"),
        "format": compact.get("format"),
        "frontier": compact.get("frontier"),
        **rebuilt_content,
        "operator_spec": _compact_operator_projection(operator_spec),
    }
    rebuilt_bytes = canonical_json_bytes(reconstructed)
    _require(rebuilt_bytes == compact_bytes, "compact parent projection changed")

    source_map = reduced_cli._local_source_map(local_audit)
    exact_spans = _exact_span_keys(local_audit, original_bindings)
    priorities = _local_priorities(
        local_audit, known_handles=frozenset(handle_ids)
    )
    provenance: list[ProtectedParentHandleProvenance] = []
    aggregate_sources: list[str] = []
    for original in original_bindings:
        sources = source_map.get(original.local_source_locator_sha256)
        _require(bool(sources), "protected parent binding lost local sources")
        ordered_sources = tuple(sorted(sources))
        aggregate_sources.extend(ordered_sources)
        provenance.append(
            ProtectedParentHandleProvenance(
                handle_id=original.handle_id,
                parent_mechanism_id=mechanism_by_handle[original.handle_id],
                source_ids=ordered_sources,
                original_binding=original,
                cloned_binding=cloned_by_handle[original.handle_id],
                exact_span_keys=exact_spans.get(original.handle_id, ()),
                local_selection_priority=priorities.get(original.handle_id, ()),
            )
        )

    compact_sha = identity_sha256(compact)
    reconstructed_sha = identity_sha256(reconstructed)
    audit = ProtectedParentAudit(
        question_sha256=operator_spec.question_sha256,
        parent_composition_artifact_sha256=parent_sha,
        composition_row_sha256=row_sha,
        compact_projection_sha256=compact_sha,
        reconstructed_compact_projection_sha256=reconstructed_sha,
        compact_projection_byte_identical=True,
        frontier_mode=frontier_mode,
        truncated=truncated,
        parent_frontier_reused=frontier_reused,
        source_ids=_ordered_unique(aggregate_sources),
        compact_item_receipt_order=tuple(
            row.receipt_sha256 for row in ordered_items
        ),
        groups=tuple(group_audits),
        source_provenance=tuple(provenance),
    )
    return ProtectedParentContributionSet(tuple(contributions), audit)
