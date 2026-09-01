"""Provider-free fact-seeded one-hop reads over a resident full-store index.

The typed fact compiler can make a useful local fact explicit without finding
the remote turn that completes its story.  This adapter uses only exact cited
fact surfaces as a second-read seed.  It validates those citations against the
same admitted H/G evidence and prompt-external bindings that produced the fact
packet, performs one bounded read through the trusted active scanner, and then
hydrates the winning exact cached row (or its exact selected window when row
expansion would overlap first-pass evidence or exceed a bound).

No benchmark answer, question ID, provider call, embedding, or persisted
transformer state enters this module.  Physical index coverage is not treated
as semantic completeness, and invalid fact packets materialize as explicit
zero-result records rather than aborting a population assay.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .full_store_slot_closure import (
    FullStoreSlotCandidate,
    FullStoreSlotClosureResult,
    FullStoreWindowIndex,
    LocalCitationBinding,
)
from .typed_active_full_store_scanner import scan_typed_active_full_store
from .typed_active_reconstruction import (
    ActiveReconstructionBudget,
    ActiveReconstructionCandidateScanner,
    ActiveReconstructionCue,
    ActiveReconstructionScanBatch,
    ActiveReconstructionScanRequest,
    active_candidate_id_for_index_span,
    active_supported_slot_ids,
    active_temporal_support,
    candidate_projection_receipt_sha256,
    citation_span_receipt_sha256,
    derive_index_aware_active_reconstruction_cues,
    validate_active_reconstruction_scan_batch,
)
from .typed_fact_compiler import TypedFactPacket
from .typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    ProvenanceGrade,
    TypedEvidenceContribution,
    TypedEvidenceItem,
    conservative_numeric_value,
    parse_typed_items,
)
from .typed_operator_spec import SlotKind, normalized_terms


MECHANISM_ID = "typed_fact_seeded_one_hop_reconstruction_v1"
BUDGET_FORMAT = "memory-condense-typed-fact-seeded-read-budget-v1"
HANDLE_PROOF_FORMAT = "memory-condense-typed-fact-seed-handle-proof-v1"
PROVENANCE_FORMAT = "memory-condense-typed-fact-seed-provenance-v1"
DECISION_FORMAT = "memory-condense-typed-fact-seeded-read-decision-v1"
LINEAGE_FORMAT = "memory-condense-typed-fact-seeded-read-lineage-v1"
RESULT_FORMAT = "memory-condense-typed-fact-seeded-read-result-v1"
HYDRATED_CANDIDATE_FORMAT = (
    "memory-condense-typed-fact-seeded-hydrated-candidate-v1"
)
REINJECTED_PARENT_FORMAT = (
    "memory-condense-typed-fact-seeded-reinjected-parent-v1"
)


class TypedFactSeededReconstructionError(MatchedEvalContractError):
    """Raised when fact-seeded read bounds or provenance change."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TypedFactSeededReconstructionError(message)


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _source_provider_input(source: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(source, Mapping), "fact seed source must be an object")
    candidate: object = source
    if "dated_question" not in source or "typed_evidence" not in source:
        candidate = source.get("provider_input")
        if not isinstance(candidate, Mapping):
            projection = source.get("provider_projection")
            candidate = (
                projection.get("provider_input")
                if isinstance(projection, Mapping)
                else None
            )
    _require(
        isinstance(candidate, Mapping)
        and type(candidate.get("dated_question")) is str
        and isinstance(candidate.get("typed_evidence"), Mapping),
        "fact seed source has no provider input",
    )
    value = dict(candidate)
    assert_gold_blind(value, path="typed_fact_seed_source_provider_input")
    return value


_OPERATOR_DERIVED_FIELDS = frozenset(
    {
        "format",
        "question_sha256",
        "receipt_sha256",
        "retained_transformer_token_state_bytes",
        "route_receipt_sha256",
    }
)
_SLOT_DERIVED_FIELDS = frozenset({"format"})


def _operator_semantic_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only serialization lineage fields from an operator projection.

    Compact final evidence intentionally omits the slot ``format`` marker and
    top-level runtime receipts.  Every field that affects retrieval or answer
    semantics remains and must compare exactly, including slot order.
    """

    raw_slots = value.get("required_slots")
    _require(type(raw_slots) is list, "fact seed operator slots changed type")
    slots: list[dict[str, Any]] = []
    for raw in raw_slots:
        _require(isinstance(raw, Mapping), "fact seed operator slot changed type")
        slots.append(
            {
                key: child
                for key, child in raw.items()
                if key not in _SLOT_DERIVED_FIELDS
            }
        )
    return {
        **{
            key: child
            for key, child in value.items()
            if key not in _OPERATOR_DERIVED_FIELDS
            and key != "required_slots"
        },
        "required_slots": slots,
    }


def _operator_lineage_matches(
    parent_projection: Mapping[str, Any],
    source_projection: Mapping[str, Any],
) -> bool:
    """Compare exact semantics and any serialization fields source retained."""

    if _operator_semantic_projection(parent_projection) != (
        _operator_semantic_projection(source_projection)
    ):
        return False
    for key in _OPERATOR_DERIVED_FIELDS & set(source_projection):
        if source_projection.get(key) != parent_projection.get(key):
            return False
    parent_slots = parent_projection.get("required_slots")
    source_slots = source_projection.get("required_slots")
    if type(parent_slots) is not list or type(source_slots) is not list:
        return False
    for parent_slot, source_slot in zip(parent_slots, source_slots, strict=True):
        if not isinstance(parent_slot, Mapping) or not isinstance(
            source_slot, Mapping
        ):
            return False
        for key in _SLOT_DERIVED_FIELDS & set(source_slot):
            if source_slot.get(key) != parent_slot.get(key):
                return False
    return True


def rematerialize_evidence_handle_bindings(
    rows: Sequence[Mapping[str, Any]], /
) -> tuple[EvidenceHandleBinding, ...]:
    """Rebuild and byte-check retained binding projections from an artifact."""

    _require(
        isinstance(rows, Sequence)
        and not isinstance(rows, (str, bytes)),
        "retained binding projections changed type",
    )
    rebuilt: list[EvidenceHandleBinding] = []
    expected_keys = {
        "citation_char_count",
        "citation_sha256",
        "evidence_receipt_sha256",
        "format",
        "handle_id",
        "local_source_locator_sha256",
        "origin",
        "parent_receipt_sha256",
        "payload_sha256",
        "provenance_grade",
        "receipt_sha256",
        "sealed_artifact_sha256",
        "source_group_handle",
    }
    for raw in rows:
        _require(
            isinstance(raw, Mapping) and set(raw) == expected_keys,
            "retained binding projection changed shape",
        )
        try:
            binding = EvidenceHandleBinding(
                handle_id=str(raw["handle_id"]),
                origin=EvidenceOrigin(str(raw["origin"])),
                provenance_grade=ProvenanceGrade(str(raw["provenance_grade"])),
                source_group_handle=str(raw["source_group_handle"]),
                sealed_artifact_sha256=str(raw["sealed_artifact_sha256"]),
                parent_receipt_sha256=str(raw["parent_receipt_sha256"]),
                evidence_receipt_sha256=str(raw["evidence_receipt_sha256"]),
                payload_sha256=str(raw["payload_sha256"]),
                citation_sha256=str(raw["citation_sha256"]),
                citation_char_count=raw["citation_char_count"],
                local_source_locator_sha256=str(
                    raw["local_source_locator_sha256"]
                ),
                receipt_sha256=str(raw["receipt_sha256"]),
            )
        except (KeyError, TypeError, ValueError, MatchedEvalContractError) as exc:
            raise TypedFactSeededReconstructionError(
                "retained binding projection failed rematerialization"
            ) from exc
        _require(
            binding.projection() == dict(raw),
            "retained binding projection changed bytes",
        )
        rebuilt.append(binding)
    _require(
        len({row.handle_id for row in rebuilt}) == len(rebuilt),
        "retained binding handles repeat",
    )
    return tuple(rebuilt)


@dataclass(frozen=True, slots=True)
class FactSeededReconstructionBudget:
    """Hard bounds for a single fact-derived global read and hydration."""

    max_cues: int = 12
    max_terms_per_cue: int = 16
    max_cue_terms: int = 96
    max_scanner_candidates: int = 8
    max_scanner_tokens: int = 1_024
    max_hydrated_candidates: int = 8
    max_hydrated_tokens: int = 2_048
    max_enclosing_row_tokens: int = 384
    use_coverage_aware_callback_selection: bool = False
    use_cited_parent_provenance_reinjection: bool = False

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name in {
                "use_coverage_aware_callback_selection",
                "use_cited_parent_provenance_reinjection",
            }:
                _require(type(value) is bool, f"{name} must be exact")
                continue
            _require(type(value) is int and value > 0, f"{name} must be positive")
        _require(
            self.max_cues <= 24
            and self.max_terms_per_cue <= 32
            and self.max_cue_terms <= 256
            and self.max_scanner_candidates <= 32
            and self.max_scanner_tokens <= 4_096
            and self.max_hydrated_candidates <= 32
            and self.max_hydrated_tokens <= 4_096
            and self.max_enclosing_row_tokens <= self.max_hydrated_tokens,
            "fact-seeded read budget exceeds its hard envelope",
        )

    def projection(self) -> dict[str, Any]:
        value = {
            "format": BUDGET_FORMAT,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name
                not in {
                    "use_coverage_aware_callback_selection",
                    "use_cited_parent_provenance_reinjection",
                }
            },
        }
        if self.use_coverage_aware_callback_selection:
            value["use_coverage_aware_callback_selection"] = True
        if self.use_cited_parent_provenance_reinjection:
            value["use_cited_parent_provenance_reinjection"] = True
        return value

    @property
    def budget_id(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class FactSeedHandleProof:
    """Exact local proof that one compiled H citation was first-round admitted."""

    handle_id: str
    source_group_handle: str
    binding_receipt_sha256: str
    local_source_locator_sha256: str
    source_ids: tuple[str, ...]
    fact_receipt_sha256s: tuple[str, ...]
    source_summary_sha256s: tuple[str, ...]
    citation_quote_sha256s: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            re.fullmatch(r"H[0-9]{3,6}", self.handle_id) is not None
            and re.fullmatch(r"G[0-9]{3,6}", self.source_group_handle) is not None,
            "fact seed proof lost opaque coordinates",
        )
        for value, label in (
            (self.binding_receipt_sha256, "fact seed binding"),
            (self.local_source_locator_sha256, "fact seed locator"),
        ):
            require_sha256(value, label)
        for values, label in (
            (self.source_ids, "fact seed sources"),
            (self.fact_receipt_sha256s, "fact seed facts"),
            (self.source_summary_sha256s, "fact seed summaries"),
            (self.citation_quote_sha256s, "fact seed quotes"),
        ):
            _require(
                type(values) is tuple and values and len(set(values)) == len(values),
                f"{label} changed",
            )
        for value in self.source_ids:
            require_text(value, "fact seed source")
        for value in (
            *self.fact_receipt_sha256s,
            *self.source_summary_sha256s,
            *self.citation_quote_sha256s,
        ):
            require_sha256(value, "fact seed proof digest")
        expected = identity_sha256(self.local_audit_projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "fact seed proof receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "binding_receipt_sha256": self.binding_receipt_sha256,
            "citation_quote_sha256s": list(self.citation_quote_sha256s),
            "fact_receipt_sha256s": list(self.fact_receipt_sha256s),
            "format": HANDLE_PROOF_FORMAT,
            "handle_id": self.handle_id,
            "source_group_handle": self.source_group_handle,
            "source_summary_sha256s": list(self.source_summary_sha256s),
            "verified_resident_source_count": len(self.source_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    def local_audit_projection(
        self, *, include_receipt: bool = True
    ) -> dict[str, Any]:
        value = {
            **self.projection(include_receipt=False),
            "local_source_locator_sha256": self.local_source_locator_sha256,
            "source_ids": list(self.source_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class FactSeedProvenance:
    index_receipt_sha256: str
    parent_result_receipt_sha256: str
    packet_receipt_sha256: str
    source_provider_input_sha256: str
    admitted_binding_receipt_sha256s: tuple[str, ...]
    handle_proofs: tuple[FactSeedHandleProof, ...]
    provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.index_receipt_sha256, "fact seed index"),
            (self.parent_result_receipt_sha256, "fact seed parent"),
            (self.packet_receipt_sha256, "fact seed packet"),
            (self.source_provider_input_sha256, "fact seed provider input"),
        ):
            require_sha256(value, label)
        _require(
            type(self.admitted_binding_receipt_sha256s) is tuple
            and len(set(self.admitted_binding_receipt_sha256s))
            == len(self.admitted_binding_receipt_sha256s),
            "fact seed admitted bindings changed",
        )
        for value in self.admitted_binding_receipt_sha256s:
            require_sha256(value, "fact seed admitted binding")
        _require(
            type(self.handle_proofs) is tuple
            and all(type(row) is FactSeedHandleProof for row in self.handle_proofs)
            and len({row.handle_id for row in self.handle_proofs})
            == len(self.handle_proofs),
            "fact seed handle proofs changed",
        )
        _require(
            self.provider_calls == 0
            and self.retained_transformer_token_state_bytes == 0,
            "fact seed provenance retained provider state",
        )
        expected = identity_sha256(self.local_audit_projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "fact seed provenance changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="typed_fact_seed_provenance")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "admitted_binding_receipt_sha256s": list(
                self.admitted_binding_receipt_sha256s
            ),
            "format": PROVENANCE_FORMAT,
            "handle_proofs": [row.projection() for row in self.handle_proofs],
            "index_receipt_sha256": self.index_receipt_sha256,
            "packet_receipt_sha256": self.packet_receipt_sha256,
            "parent_result_receipt_sha256": self.parent_result_receipt_sha256,
            "provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
            "source_provider_input_sha256": self.source_provider_input_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    def local_audit_projection(
        self, *, include_receipt: bool = True
    ) -> dict[str, Any]:
        value = {
            **self.projection(include_receipt=False),
            "handle_proofs": [
                row.local_audit_projection() for row in self.handle_proofs
            ],
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _source_ids(
    value: object, *, locator: str
) -> tuple[str, ...]:
    if type(value) is str:
        rows: Sequence[object] = (value,)
    elif isinstance(value, (tuple, list, set, frozenset)):
        rows = tuple(value)
    else:
        raise TypedFactSeededReconstructionError(
            f"source mapping for locator {locator} changed type"
        )
    _require(
        rows
        and all(type(row) is str and bool(row.strip()) for row in rows),
        "fact seed source mapping is empty or invalid",
    )
    return tuple(sorted(set(rows)))  # type: ignore[arg-type]


def build_fact_seed_provenance(
    index: FullStoreWindowIndex,
    parent_result: FullStoreSlotClosureResult,
    source: Mapping[str, Any],
    packet: TypedFactPacket,
    admitted_bindings: tuple[EvidenceHandleBinding, ...],
    /,
    *,
    source_ids_by_local_locator_sha256: Mapping[str, object],
) -> FactSeedProvenance:
    """Verify fact citations against admitted summaries, bindings, and index."""

    _require(type(index) is FullStoreWindowIndex, "fact seed index changed")
    _require(
        type(parent_result) is FullStoreSlotClosureResult
        and parent_result.receipt.window_index_receipt_sha256
        == index.receipt_sha256
        and parent_result.receipt.cache_receipt_sha256
        == index.cache.cache_receipt_sha256,
        "fact seed parent escaped its resident index",
    )
    _require(type(packet) is TypedFactPacket, "fact seed packet changed")
    _require(
        type(admitted_bindings) is tuple
        and all(type(row) is EvidenceHandleBinding for row in admitted_bindings)
        and len({row.handle_id for row in admitted_bindings})
        == len(admitted_bindings),
        "fact seed admitted bindings changed",
    )
    _require(
        isinstance(source_ids_by_local_locator_sha256, Mapping),
        "fact seed source map changed",
    )
    provider = _source_provider_input(source)
    question = provider["dated_question"]
    typed = provider["typed_evidence"]
    _require(isinstance(typed, Mapping), "fact seed typed evidence changed")
    operator = typed.get("operator_spec")
    raw_handles = typed.get("handles")
    raw_items = typed.get("items")
    _require(
        isinstance(operator, Mapping)
        and type(raw_handles) is list
        and type(raw_items) is list,
        "fact seed source evidence shape changed",
    )
    _require(
        question == parent_result.dated_question
        and packet.dated_question_sha256 == quote_sha256(question),
        "fact seed question lineage changed",
    )
    parent_operator = parent_result.operator_spec.projection()
    _require(
        _operator_lineage_matches(parent_operator, operator)
        and packet.operator_spec_receipt_sha256
        == operator.get("receipt_sha256", identity_sha256(operator)),
        "fact seed operator lineage changed",
    )

    source_handles: dict[str, Mapping[str, Any]] = {}
    for raw in raw_handles:
        _require(isinstance(raw, Mapping), "fact seed source handle changed")
        handle = raw.get("handle_id")
        _require(
            type(handle) is str
            and handle not in source_handles,
            "fact seed source handles repeat or are invalid",
        )
        source_handles[handle] = raw
    binding_by_handle = {row.handle_id: row for row in admitted_bindings}
    resident_sources = {row.source_id for row in index.rows}

    facts_by_handle: dict[str, list[str]] = {}
    summaries_by_handle: dict[str, list[str]] = {}
    quotes_by_handle: dict[str, list[str]] = {}
    for fact in packet.facts:
        for citation in fact.citations:
            _require(
                0 <= citation.source_item_index < len(raw_items),
                "fact citation source coordinate escaped admitted evidence",
            )
            item = raw_items[citation.source_item_index]
            _require(isinstance(item, Mapping), "fact citation source item changed")
            item_handles = item.get("handle_ids")
            summary = item.get("summary")
            source_handle = source_handles.get(citation.handle_id)
            binding = binding_by_handle.get(citation.handle_id)
            _require(
                type(item_handles) is list
                and citation.handle_id in item_handles
                and type(summary) is str
                and citation.quote in summary
                and quote_sha256(summary) == citation.source_summary_sha256
                and quote_sha256(citation.quote) == citation.quote_sha256,
                "fact citation is not exact admitted source evidence",
            )
            _require(
                source_handle is not None
                and binding is not None
                and source_handle.get("group_handle", source_handle.get("source_group_handle"))
                == citation.group_handle
                == binding.source_group_handle,
                "fact citation H/G binding changed",
            )
            if source_handle.get("origin") is not None:
                _require(
                    source_handle.get("origin") == binding.origin.value,
                    "fact citation origin changed",
                )
            if source_handle.get("provenance_grade") is not None:
                _require(
                    source_handle.get("provenance_grade")
                    == binding.provenance_grade.value,
                    "fact citation provenance grade changed",
                )
            sources = _source_ids(
                source_ids_by_local_locator_sha256.get(
                    binding.local_source_locator_sha256
                ),
                locator=binding.local_source_locator_sha256,
            )
            _require(
                set(sources) <= resident_sources,
                "fact citation provenance points outside the resident index",
            )
            facts_by_handle.setdefault(citation.handle_id, []).append(
                fact.receipt_sha256
            )
            summaries_by_handle.setdefault(citation.handle_id, []).append(
                citation.source_summary_sha256
            )
            quotes_by_handle.setdefault(citation.handle_id, []).append(
                citation.quote_sha256
            )

    cited_handles = _ordered_unique(
        citation.handle_id
        for fact in packet.facts
        for citation in fact.citations
    )
    _require(
        cited_handles == packet.retained_handle_ids,
        "fact packet retained handles lost exact citation order",
    )
    proofs: list[FactSeedHandleProof] = []
    for handle in cited_handles:
        binding = binding_by_handle[handle]
        sources = _source_ids(
            source_ids_by_local_locator_sha256.get(
                binding.local_source_locator_sha256
            ),
            locator=binding.local_source_locator_sha256,
        )
        proofs.append(
            FactSeedHandleProof(
                handle_id=handle,
                source_group_handle=binding.source_group_handle,
                binding_receipt_sha256=binding.receipt_sha256,
                local_source_locator_sha256=binding.local_source_locator_sha256,
                source_ids=sources,
                fact_receipt_sha256s=_ordered_unique(facts_by_handle[handle]),
                source_summary_sha256s=_ordered_unique(
                    summaries_by_handle[handle]
                ),
                citation_quote_sha256s=_ordered_unique(quotes_by_handle[handle]),
            )
        )
    return FactSeedProvenance(
        index_receipt_sha256=index.receipt_sha256,
        parent_result_receipt_sha256=parent_result.receipt.receipt_sha256,
        packet_receipt_sha256=packet.receipt_sha256,
        source_provider_input_sha256=identity_sha256(provider),
        admitted_binding_receipt_sha256s=tuple(
            row.receipt_sha256 for row in admitted_bindings
        ),
        handle_proofs=tuple(proofs),
    )


def _fact_seed_items(
    packet: TypedFactPacket,
    parent_result: FullStoreSlotClosureResult,
    admitted_bindings: tuple[EvidenceHandleBinding, ...],
) -> tuple[tuple[TypedEvidenceItem, ...], str, Mapping[str, str]]:
    binding_by_handle = {row.handle_id: row for row in admitted_bindings}
    retained_bindings = tuple(binding_by_handle[row] for row in packet.retained_handle_ids)
    raw_items: list[dict[str, Any]] = []
    for fact in packet.facts:
        raw: dict[str, Any] = {
            "handle_ids": list(fact.handle_ids),
            "included": True,
            "kind": fact.kind,
            "numeric_role": "operand" if fact.numeric_value is not None else "none",
            "specificity_terms": list(fact.question_term_hits),
            "summary": fact.text,
            "value_authority": "explicit",
        }
        for key, value in (
            ("entity_key", fact.entity),
            ("numeric_value", fact.numeric_value),
            ("unit", fact.unit),
            ("date", fact.date),
            ("status", fact.status),
        ):
            if value is not None:
                raw[key] = value
        raw_items.append(raw)
    parsed = parse_typed_items(
        raw_items,
        operator_spec=parent_result.operator_spec,
        bindings=retained_bindings,
    )
    _require(
        len(parsed.accepted_items) == len(packet.facts)
        and not parsed.rejected_items,
        "validated compiled facts did not rematerialize as typed cue seeds",
    )
    fact_by_item = {
        item.receipt_sha256: fact.receipt_sha256
        for item, fact in zip(parsed.accepted_items, packet.facts, strict=True)
    }
    return parsed.accepted_items, parsed.parse_receipt_sha256, fact_by_item


def _cited_parent_candidate_pairs(
    index: FullStoreWindowIndex,
    parent_result: FullStoreSlotClosureResult,
    packet: TypedFactPacket,
    provenance: FactSeedProvenance,
    /,
) -> tuple[
    tuple[tuple[FullStoreSlotCandidate, LocalCitationBinding], ...],
    Mapping[str, tuple[str, ...]],
]:
    """Rematerialize exact local CAV anchors for cited H/G provenance.

    Exact cited bytes are preferred.  A compiler citation can be a byte-exact
    substring of an admitted summary rather than the cached raw turn; in that
    case the deterministic highest-overlap window from the already verified
    source is used only as an opaque source/history anchor.  Every returned
    pair is independently verifiable against ``index`` by the active cue
    validator.  Duplicate discoveries are intentionally retained here and are
    collapsed only by the downstream bounded cue derivation.
    """

    proof_by_handle = {row.handle_id: row for row in provenance.handle_proofs}
    relevant_sources = {
        source_id
        for proof in provenance.handle_proofs
        for source_id in proof.source_ids
    }
    window_indices_by_source: dict[str, list[int]] = {
        source_id: [] for source_id in relevant_sources
    }
    for window_index, window in enumerate(index.windows):
        if window.row.source_id in relevant_sources:
            window_indices_by_source[window.row.source_id].append(window_index)

    pairs: list[tuple[FullStoreSlotCandidate, LocalCitationBinding]] = []
    fact_receipts_by_parent: dict[str, list[str]] = {}
    for fact in packet.facts:
        for citation in fact.citations:
            proof = proof_by_handle.get(citation.handle_id)
            _require(
                proof is not None
                and fact.receipt_sha256 in proof.fact_receipt_sha256s
                and citation.quote_sha256 in proof.citation_quote_sha256s,
                "cited parent escaped its verified fact/handle proof",
            )
            group_match = re.fullmatch(r"G([0-9]{3,6})", citation.group_handle)
            _require(group_match is not None, "cited parent group changed")
            source_group_handle = f"G{int(group_match.group(1)):04d}"
            citation_receipt = identity_sha256(citation.projection())
            citation_terms = set(normalized_terms(citation.quote))

            for source_id in proof.source_ids:
                source_window_indices = tuple(
                    window_indices_by_source.get(source_id, ())
                )
                _require(
                    source_window_indices,
                    "verified cited source has no resident index window",
                )
                exact_coordinates: dict[
                    tuple[str, int, int], tuple[int, int, int]
                ] = {}
                seen_chunks: set[str] = set()
                for window_index in source_window_indices:
                    window = index.windows[window_index]
                    row = window.row
                    if row.chunk_id in seen_chunks:
                        continue
                    seen_chunks.add(row.chunk_id)
                    start = row.text.find(citation.quote)
                    while start >= 0:
                        end = start + len(citation.quote)
                        containing = tuple(
                            candidate_index
                            for candidate_index in source_window_indices
                            if index.windows[candidate_index].row.chunk_id
                            == row.chunk_id
                            and index.windows[candidate_index].start_char <= start
                            and end <= index.windows[candidate_index].end_char
                        )
                        if containing:
                            selected_window = min(
                                containing,
                                key=lambda candidate_index: (
                                    index.windows[candidate_index].end_char
                                    - index.windows[candidate_index].start_char,
                                    candidate_index,
                                ),
                            )
                            exact_coordinates[(row.chunk_id, start, end)] = (
                                selected_window,
                                start,
                                end,
                            )
                        start = row.text.find(citation.quote, start + 1)

                if exact_coordinates:
                    anchors = tuple(
                        (*value, "exact_citation_surface")
                        for _key, value in sorted(exact_coordinates.items())
                    )
                else:
                    selected_window = min(
                        source_window_indices,
                        key=lambda candidate_index: (
                            -len(
                                citation_terms
                                & set(index.windows[candidate_index].terms)
                            ),
                            index.windows[candidate_index].token_count,
                            index.windows[candidate_index].row.ordinal,
                            index.windows[candidate_index].start_char,
                            candidate_index,
                        ),
                    )
                    selected = index.windows[selected_window]
                    anchors = (
                        (
                            selected_window,
                            selected.start_char,
                            selected.end_char,
                            "verified_source_window_anchor",
                        ),
                    )

                for window_index, start_char, end_char, anchor_kind in anchors:
                    window = index.windows[window_index]
                    row = window.row
                    quote = row.text[start_char:end_char]
                    text_sha = quote_sha256(quote)
                    span = EvidenceSpan(
                        chunk_id=row.chunk_id,
                        start_char=start_char,
                        end_char=end_char,
                        quote_sha256=text_sha,
                        ordinal=row.ordinal,
                        source_id=row.source_id,
                        turn_start_char=row.turn_start_char,
                        turn_id=row.turn_id,
                        role=row.role,
                        created_at=row.created_at,
                    )
                    candidate_id = active_candidate_id_for_index_span(
                        index,
                        window_index,
                        start_char=start_char,
                        end_char=end_char,
                        text_sha256=text_sha,
                    )
                    local = LocalCitationBinding(
                        candidate_id=candidate_id,
                        source_group_handle=source_group_handle,
                        namespace_id=index.cache.namespace_id,
                        cache_receipt_sha256=index.cache.cache_receipt_sha256,
                        source_database_sha256=index.cache.source_database_sha256,
                        source_store_receipt_sha256=(
                            index.cache.source_store_receipt_sha256
                        ),
                        source_id=row.source_id,
                        partition_id=row.partition_id,
                        span=span,
                        quote_sha256=text_sha,
                    )
                    slots = active_supported_slot_ids(
                        parent_result.operator_spec, quote
                    )
                    distance, temporal = active_temporal_support(
                        window.event_date, parent_result.temporal_target
                    )
                    quote_terms = set(normalized_terms(quote))
                    matched_terms = tuple(
                        term
                        for term in normalized_terms(citation.quote)
                        if term in quote_terms
                    )
                    axes = [
                        "fact_seed_cited_parent_provenance_reinjection",
                        f"fact_seed_citation_receipt:{citation_receipt}",
                        f"fact_seed_parent_anchor:{anchor_kind}",
                    ]
                    if slots:
                        axes.append("original_operator_slot_support")
                    if temporal:
                        axes.append("original_temporal_target_support")
                    candidate = FullStoreSlotCandidate(
                        candidate_id=candidate_id,
                        source_group_handle=source_group_handle,
                        quote=quote,
                        quote_sha256=text_sha,
                        token_count=count_tokens(quote),
                        role=row.role,
                        created_at=row.created_at,
                        event_date=window.event_date,
                        event_date_basis=window.event_date_basis,
                        supported_slot_ids=slots,
                        matched_query_terms=matched_terms,
                        contains_numeric_value=bool(_NUMBER_RE.search(quote)),
                        temporal_distance_days=distance,
                        selection_axes=tuple(axes),
                        citation_binding_receipt_sha256=local.receipt_sha256,
                    )
                    pairs.append((candidate, local))
                    parent_receipt = candidate_projection_receipt_sha256(
                        candidate
                    )
                    fact_receipts_by_parent.setdefault(
                        parent_receipt, []
                    ).append(fact.receipt_sha256)
    return tuple(pairs), {
        parent: _ordered_unique(facts)
        for parent, facts in fact_receipts_by_parent.items()
    }


def _active_budget(budget: FactSeededReconstructionBudget) -> ActiveReconstructionBudget:
    return ActiveReconstructionBudget(
        max_hops=1,
        max_cues_per_hop=budget.max_cues,
        max_terms_per_cue=budget.max_terms_per_cue,
        max_cue_terms_per_hop=budget.max_cue_terms,
        max_selected_candidates_per_hop=budget.max_scanner_candidates,
        max_selected_tokens_per_hop=budget.max_scanner_tokens,
        max_admitted_candidates=budget.max_hydrated_candidates,
        max_admitted_tokens=budget.max_hydrated_tokens,
        use_selected_provenance_affinity=(
            budget.use_cited_parent_provenance_reinjection
        ),
        use_index_aware_cue_ranking=True,
        use_fixed_scan_subchannels=True,
        use_coverage_aware_callback_selection=(
            budget.use_coverage_aware_callback_selection
        ),
    )


DecisionStatus = Literal[
    "admitted_enclosing_row",
    "admitted_exact_window",
    "duplicate_first_pass_span",
    "duplicate_recovered_span",
    "hydration_budget_excluded",
]


@dataclass(frozen=True, slots=True)
class FactSeededRecoveryDecision:
    scan_match_receipt_sha256: str
    status: DecisionStatus
    supporting_fact_receipt_sha256s: tuple[str, ...]
    recovered_candidate_receipt_sha256: str | None
    recovered_local_binding_receipt_sha256: str | None
    reason: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.scan_match_receipt_sha256, "fact seed scan match")
        _require(
            self.status in {
                "admitted_enclosing_row",
                "admitted_exact_window",
                "duplicate_first_pass_span",
                "duplicate_recovered_span",
                "hydration_budget_excluded",
            },
            "fact seed decision status changed",
        )
        _require(
            type(self.supporting_fact_receipt_sha256s) is tuple
            and self.supporting_fact_receipt_sha256s
            and len(set(self.supporting_fact_receipt_sha256s))
            == len(self.supporting_fact_receipt_sha256s),
            "fact seed decision parents changed",
        )
        for value in self.supporting_fact_receipt_sha256s:
            require_sha256(value, "fact seed decision parent")
        admitted = self.status.startswith("admitted_")
        _require(
            admitted
            == (
                self.recovered_candidate_receipt_sha256 is not None
                and self.recovered_local_binding_receipt_sha256 is not None
            ),
            "fact seed decision admission proof changed",
        )
        for value in (
            self.recovered_candidate_receipt_sha256,
            self.recovered_local_binding_receipt_sha256,
        ):
            if value is not None:
                require_sha256(value, "fact seed recovered proof")
        require_text(self.reason, "fact seed decision reason")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "fact seed decision changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "format": DECISION_FORMAT,
            "reason": self.reason,
            "recovered_candidate_receipt_sha256": (
                self.recovered_candidate_receipt_sha256
            ),
            "recovered_local_binding_receipt_sha256": (
                self.recovered_local_binding_receipt_sha256
            ),
            "scan_match_receipt_sha256": self.scan_match_receipt_sha256,
            "status": self.status,
            "supporting_fact_receipt_sha256s": list(
                self.supporting_fact_receipt_sha256s
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class FactSeededRecoveryLineage:
    supporting_fact_receipt_sha256s: tuple[str, ...]
    cue_receipt_sha256s: tuple[str, ...]
    scan_match_receipt_sha256: str
    source_window_span_receipt_sha256: str
    recovered_candidate_receipt_sha256: str
    recovered_local_binding_receipt_sha256: str
    recovered_span_receipt_sha256: str
    cached_row_receipt_sha256: str
    hydration_kind: Literal["enclosing_row", "exact_window"]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for values, label in (
            (self.supporting_fact_receipt_sha256s, "fact seed lineage facts"),
            (self.cue_receipt_sha256s, "fact seed lineage cues"),
        ):
            _require(
                type(values) is tuple and values and len(set(values)) == len(values),
                f"{label} changed",
            )
            for value in values:
                require_sha256(value, label)
        for value, label in (
            (self.scan_match_receipt_sha256, "fact seed lineage match"),
            (self.source_window_span_receipt_sha256, "fact seed source span"),
            (self.recovered_candidate_receipt_sha256, "fact seed child candidate"),
            (self.recovered_local_binding_receipt_sha256, "fact seed child binding"),
            (self.recovered_span_receipt_sha256, "fact seed child span"),
            (self.cached_row_receipt_sha256, "fact seed cached row"),
        ):
            require_sha256(value, label)
        _require(
            self.hydration_kind in {"enclosing_row", "exact_window"},
            "fact seed hydration kind changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "fact seed lineage changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "cached_row_receipt_sha256": self.cached_row_receipt_sha256,
            "cue_receipt_sha256s": list(self.cue_receipt_sha256s),
            "format": LINEAGE_FORMAT,
            "hydration_kind": self.hydration_kind,
            "recovered_candidate_receipt_sha256": (
                self.recovered_candidate_receipt_sha256
            ),
            "recovered_local_binding_receipt_sha256": (
                self.recovered_local_binding_receipt_sha256
            ),
            "recovered_span_receipt_sha256": self.recovered_span_receipt_sha256,
            "scan_match_receipt_sha256": self.scan_match_receipt_sha256,
            "source_window_span_receipt_sha256": (
                self.source_window_span_receipt_sha256
            ),
            "supporting_fact_receipt_sha256s": list(
                self.supporting_fact_receipt_sha256s
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


ResultStatus = Literal["scanned", "packet_invalid", "no_fact_cues"]


@dataclass(frozen=True, slots=True)
class TypedFactSeededReconstructionResult:
    index: FullStoreWindowIndex
    parent_result: FullStoreSlotClosureResult
    packet: TypedFactPacket
    provenance: FactSeedProvenance
    budget: FactSeededReconstructionBudget
    status: ResultStatus
    reason: str
    seed_items: tuple[TypedEvidenceItem, ...]
    seed_parse_receipt_sha256: str | None
    request: ActiveReconstructionScanRequest | None
    batch: ActiveReconstructionScanBatch | None
    decisions: tuple[FactSeededRecoveryDecision, ...]
    candidates: tuple[FullStoreSlotCandidate, ...]
    local_bindings: tuple[LocalCitationBinding, ...]
    lineages: tuple[FactSeededRecoveryLineage, ...]
    cue_derivation_truncated: bool
    hydration_truncated: bool
    provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.index) is FullStoreWindowIndex
            and type(self.parent_result) is FullStoreSlotClosureResult
            and self.parent_result.receipt.window_index_receipt_sha256
            == self.index.receipt_sha256,
            "fact seed result index/parent changed",
        )
        _require(
            type(self.packet) is TypedFactPacket
            and type(self.provenance) is FactSeedProvenance
            and self.provenance.packet_receipt_sha256 == self.packet.receipt_sha256
            and self.provenance.index_receipt_sha256 == self.index.receipt_sha256,
            "fact seed result provenance changed",
        )
        _require(
            tuple(row.handle_id for row in self.provenance.handle_proofs)
            == self.packet.retained_handle_ids,
            "fact seed result handle provenance changed",
        )
        _require(type(self.budget) is FactSeededReconstructionBudget, "fact seed budget changed")
        _require(
            self.status in {"scanned", "packet_invalid", "no_fact_cues"},
            "fact seed result status changed",
        )
        _require(
            (self.status == "packet_invalid") is (not self.packet.valid),
            "fact seed invalid-packet status changed",
        )
        require_text(self.reason, "fact seed result reason")
        _require(
            type(self.seed_items) is tuple
            and all(type(row) is TypedEvidenceItem for row in self.seed_items),
            "fact seed typed seeds changed",
        )
        if self.seed_parse_receipt_sha256 is not None:
            require_sha256(self.seed_parse_receipt_sha256, "fact seed parse")
        _require(
            (self.status == "packet_invalid")
            == (not self.seed_items and self.seed_parse_receipt_sha256 is None),
            "fact seed typed-seed materialization changed",
        )
        scanned = self.status == "scanned"
        _require(
            scanned == (self.request is not None and self.batch is not None),
            "fact seed scan materialization changed",
        )
        if scanned:
            _require(
                self.request is not None
                and self.batch is not None
                and self.request.index is self.index
                and self.request.operator_spec is self.parent_result.operator_spec
                and self.request.temporal_target is self.parent_result.temporal_target
                and self.batch.request_receipt_sha256 == self.request.receipt_sha256
                and len(self.decisions) == len(self.batch.matches),
                "fact seed scan lineage changed",
            )
        else:
            _require(
                not self.decisions
                and not self.candidates
                and not self.local_bindings
                and not self.lineages,
                "fact seed zero-result status carried candidates",
            )
        _require(
            len(self.candidates) == len(self.local_bindings) == len(self.lineages)
            and len(self.candidates) <= self.budget.max_hydrated_candidates
            and sum(row.token_count for row in self.candidates)
            <= self.budget.max_hydrated_tokens,
            "fact seed recovered evidence exceeded its bound",
        )
        for candidate, binding, lineage in zip(
            self.candidates, self.local_bindings, self.lineages, strict=True
        ):
            _require(
                candidate.candidate_id == binding.candidate_id
                and candidate.citation_binding_receipt_sha256
                == binding.receipt_sha256
                and candidate_projection_receipt_sha256(candidate)
                == lineage.recovered_candidate_receipt_sha256
                and binding.receipt_sha256
                == lineage.recovered_local_binding_receipt_sha256
                and citation_span_receipt_sha256(binding)
                == lineage.recovered_span_receipt_sha256,
                "fact seed recovered candidate lineage changed",
            )
        _require(
            type(self.cue_derivation_truncated) is bool
            and type(self.hydration_truncated) is bool,
            "fact seed truncation flags changed",
        )
        _require(
            self.provider_calls == 0
            and self.retained_transformer_token_state_bytes == 0,
            "fact seed result retained provider state",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "fact seed result changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="typed_fact_seeded_reconstruction")
        assert_gold_blind(
            self.provider_projection(),
            path="typed_fact_seeded_reconstruction_provider",
        )

    @property
    def truncated(self) -> bool:
        return bool(
            self.status != "scanned"
            or self.cue_derivation_truncated
            or self.hydration_truncated
            or (self.batch is not None and self.batch.selection_truncated)
        )

    def provider_projection(self) -> dict[str, Any]:
        return {
            "candidates": [row.projection() for row in self.candidates],
            "format": RESULT_FORMAT,
            "frontier_mode": FrontierMode.BOUNDED.value,
            "new_provider_calls": 0,
            "operator_spec": self.parent_result.operator_spec.projection(),
            "retained_transformer_token_state_bytes": 0,
            "semantic_completeness_status": "not_claimed",
            "status": self.status,
            "temporal_target": self.parent_result.temporal_target.projection(),
            "truncated": self.truncated,
        }

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "admitted_candidate_count": len(self.candidates),
            "admitted_candidate_tokens": sum(
                row.token_count for row in self.candidates
            ),
            "batch_receipt_sha256": (
                None if self.batch is None else self.batch.receipt_sha256
            ),
            "budget_id": self.budget.budget_id,
            "cue_derivation_truncated": self.cue_derivation_truncated,
            "decision_receipt_sha256s": [
                row.receipt_sha256 for row in self.decisions
            ],
            "format": RESULT_FORMAT,
            "frontier_mode": FrontierMode.BOUNDED.value,
            "hydration_truncated": self.hydration_truncated,
            "index_receipt_sha256": self.index.receipt_sha256,
            "lineage_receipt_sha256s": [
                row.receipt_sha256 for row in self.lineages
            ],
            "new_provider_calls": 0,
            "packet_receipt_sha256": self.packet.receipt_sha256,
            "parent_result_receipt_sha256": (
                self.parent_result.receipt.receipt_sha256
            ),
            "provenance_receipt_sha256": self.provenance.receipt_sha256,
            "reason": self.reason,
            "request_receipt_sha256": (
                None if self.request is None else self.request.receipt_sha256
            ),
            "retained_transformer_token_state_bytes": 0,
            "seed_item_receipt_sha256s": [
                row.receipt_sha256 for row in self.seed_items
            ],
            "seed_parse_receipt_sha256": self.seed_parse_receipt_sha256,
            "semantic_completeness_status": "not_claimed",
            "status": self.status,
            "truncated": self.truncated,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    def local_audit_projection(self) -> dict[str, Any]:
        return {
            "batch": None if self.batch is None else self.batch.projection(),
            "decisions": [row.projection() for row in self.decisions],
            "lineages": [row.projection() for row in self.lineages],
            "local_bindings": [row.projection() for row in self.local_bindings],
            "provenance": self.provenance.local_audit_projection(),
            "provider_projection_sha256": identity_sha256(
                self.provider_projection()
            ),
            "receipt": self.projection(),
            "request": None if self.request is None else self.request.projection(),
        }


def _row_for_match(
    index: FullStoreWindowIndex, local: LocalCitationBinding
):
    span = local.span
    matches = tuple(
        row
        for row in index.rows
        if row.namespace_id == local.namespace_id
        and row.partition_id == local.partition_id
        and row.source_id == local.source_id
        and row.chunk_id == span.chunk_id
        and row.turn_id == span.turn_id
        and row.ordinal == span.ordinal
        and row.role == span.role
        and row.created_at == span.created_at
        and 0 <= span.start_char < span.end_char <= len(row.text)
        and quote_sha256(row.text[span.start_char : span.end_char])
        == local.quote_sha256
    )
    _require(len(matches) == 1, "fact seed scan match lost its exact cached row")
    return matches[0]


def _overlaps(left: EvidenceSpan, right: EvidenceSpan) -> bool:
    return bool(
        left.source_id == right.source_id
        and left.chunk_id == right.chunk_id
        and left.start_char < right.end_char
        and right.start_char < left.end_char
    )


_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w.])|"
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|dozen|hundred|thousand)\b",
    re.IGNORECASE,
)


def _hydrated_candidate(
    index: FullStoreWindowIndex,
    parent_result: FullStoreSlotClosureResult,
    match: Any,
    parent_spans: tuple[EvidenceSpan, ...],
    budget: FactSeededReconstructionBudget,
) -> tuple[FullStoreSlotCandidate, LocalCitationBinding, str, str]:
    """Hydrate one validated match without exposing a mutable scanner row."""

    row = _row_for_match(index, match.local_binding)
    full_span = EvidenceSpan(
        chunk_id=row.chunk_id,
        start_char=0,
        end_char=len(row.text),
        quote_sha256=row.text_sha256,
        ordinal=row.ordinal,
        source_id=row.source_id,
        turn_start_char=row.turn_start_char,
        turn_id=row.turn_id,
        role=row.role,
        created_at=row.created_at,
    )
    full_row_safe = bool(
        row.token_count <= budget.max_enclosing_row_tokens
        and not any(_overlaps(full_span, parent) for parent in parent_spans)
    )
    selected_span = full_span if full_row_safe else match.local_binding.span
    hydration_kind = "enclosing_row" if full_row_safe else "exact_window"
    quote = row.text[selected_span.start_char : selected_span.end_char]
    span = EvidenceSpan(
        chunk_id=row.chunk_id,
        start_char=selected_span.start_char,
        end_char=selected_span.end_char,
        quote_sha256=quote_sha256(quote),
        ordinal=row.ordinal,
        source_id=row.source_id,
        turn_start_char=row.turn_start_char,
        turn_id=row.turn_id,
        role=row.role,
        created_at=row.created_at,
    )
    row_receipt = identity_sha256(row.receipt_projection())
    candidate_id = identity_sha256(
        {
            "cached_row_receipt_sha256": row_receipt,
            "end_char": span.end_char,
            "format": HYDRATED_CANDIDATE_FORMAT,
            "index_receipt_sha256": index.receipt_sha256,
            "scan_match_receipt_sha256": match.receipt_sha256,
            "start_char": span.start_char,
            "text_sha256": span.quote_sha256,
        }
    )
    local = LocalCitationBinding(
        candidate_id=candidate_id,
        source_group_handle=match.candidate.source_group_handle,
        namespace_id=index.cache.namespace_id,
        cache_receipt_sha256=index.cache.cache_receipt_sha256,
        source_database_sha256=index.cache.source_database_sha256,
        source_store_receipt_sha256=index.cache.source_store_receipt_sha256,
        source_id=row.source_id,
        partition_id=row.partition_id,
        span=span,
        quote_sha256=span.quote_sha256,
    )
    slots = active_supported_slot_ids(parent_result.operator_spec, quote)
    distance, temporal = active_temporal_support(
        match.candidate.event_date, parent_result.temporal_target
    )
    axes = [
        f"fact_seed_support:{match.support_kind.value}",
        f"hydration:{hydration_kind}",
    ]
    if slots:
        axes.append("original_operator_slot_support")
    if temporal:
        axes.append("original_temporal_target_support")
    matched_terms = tuple(
        term
        for term in match.matched_cue_terms
        if term in set(normalized_terms(quote))
    )
    candidate = FullStoreSlotCandidate(
        candidate_id=candidate_id,
        source_group_handle=match.candidate.source_group_handle,
        quote=quote,
        quote_sha256=span.quote_sha256,
        token_count=count_tokens(quote),
        role=row.role,
        created_at=row.created_at,
        event_date=match.candidate.event_date,
        event_date_basis=match.candidate.event_date_basis,
        supported_slot_ids=slots,
        matched_query_terms=matched_terms,
        contains_numeric_value=bool(_NUMBER_RE.search(quote)),
        temporal_distance_days=distance,
        selection_axes=tuple(axes),
        citation_binding_receipt_sha256=local.receipt_sha256,
    )
    return candidate, local, hydration_kind, row_receipt


def run_typed_fact_seeded_reconstruction(
    index: FullStoreWindowIndex,
    parent_result: FullStoreSlotClosureResult,
    source: Mapping[str, Any],
    packet: TypedFactPacket,
    admitted_bindings: tuple[EvidenceHandleBinding, ...],
    /,
    *,
    source_ids_by_local_locator_sha256: Mapping[str, object],
    candidate_scanner: ActiveReconstructionCandidateScanner = (
        scan_typed_active_full_store
    ),
    budget: FactSeededReconstructionBudget = FactSeededReconstructionBudget(),
) -> TypedFactSeededReconstructionResult:
    """Run one exact-cited, index-aware global read without provider calls."""

    _require(
        candidate_scanner is scan_typed_active_full_store,
        "fact seed production read requires the trusted local scanner",
    )
    _require(type(budget) is FactSeededReconstructionBudget, "fact seed budget changed")
    provenance = build_fact_seed_provenance(
        index,
        parent_result,
        source,
        packet,
        admitted_bindings,
        source_ids_by_local_locator_sha256=source_ids_by_local_locator_sha256,
    )
    if not packet.valid:
        return TypedFactSeededReconstructionResult(
            index=index,
            parent_result=parent_result,
            packet=packet,
            provenance=provenance,
            budget=budget,
            status="packet_invalid",
            reason=packet.invalid_reason or "invalid_fact_packet",
            seed_items=(),
            seed_parse_receipt_sha256=None,
            request=None,
            batch=None,
            decisions=(),
            candidates=(),
            local_bindings=(),
            lineages=(),
            cue_derivation_truncated=False,
            hydration_truncated=False,
        )

    items, parse_receipt, fact_by_item = _fact_seed_items(
        packet, parent_result, admitted_bindings
    )
    active_budget = _active_budget(budget)
    candidate_pairs: tuple[
        tuple[FullStoreSlotCandidate, LocalCitationBinding], ...
    ] = ()
    fact_receipts_by_parent: dict[str, tuple[str, ...]] = {
        parent: (fact_receipt,)
        for parent, fact_receipt in fact_by_item.items()
    }
    if budget.use_cited_parent_provenance_reinjection:
        (
            candidate_pairs,
            cited_parent_fact_receipts,
        ) = _cited_parent_candidate_pairs(
            index, parent_result, packet, provenance
        )
        fact_receipts_by_parent.update(cited_parent_fact_receipts)
    cues, cue_truncated = derive_index_aware_active_reconstruction_cues(
        index,
        hop=1,
        items=items,
        candidate_pairs=candidate_pairs,
        operator_spec=parent_result.operator_spec,
        temporal_target=parent_result.temporal_target,
        budget=active_budget,
    )
    if not cues:
        return TypedFactSeededReconstructionResult(
            index=index,
            parent_result=parent_result,
            packet=packet,
            provenance=provenance,
            budget=budget,
            status="no_fact_cues",
            reason="validated_facts_produced_no_bounded_selective_cues",
            seed_items=items,
            seed_parse_receipt_sha256=parse_receipt,
            request=None,
            batch=None,
            decisions=(),
            candidates=(),
            local_bindings=(),
            lineages=(),
            cue_derivation_truncated=cue_truncated,
            hydration_truncated=False,
        )

    request = ActiveReconstructionScanRequest(
        index=index,
        operator_spec=parent_result.operator_spec,
        temporal_target=parent_result.temporal_target,
        hop=1,
        lineage_parent_receipt_sha256=packet.receipt_sha256,
        cues=cues,
        max_selected_candidates=budget.max_scanner_candidates,
        max_selected_tokens=budget.max_scanner_tokens,
        use_fixed_scan_subchannels=active_budget.use_fixed_scan_subchannels,
        use_coverage_aware_callback_selection=(
            active_budget.use_coverage_aware_callback_selection
        ),
    )
    batch = validate_active_reconstruction_scan_batch(
        request, candidate_scanner(request)
    )
    cue_by_receipt: dict[str, ActiveReconstructionCue] = {
        row.receipt_sha256: row for row in cues
    }
    parent_spans = tuple(row.span for row in parent_result.local_bindings)
    parent_span_receipts = {
        citation_span_receipt_sha256(row) for row in parent_result.local_bindings
    }
    seen_source_spans: set[str] = set()
    seen_recovered_spans: set[str] = set()
    decisions: list[FactSeededRecoveryDecision] = []
    candidates: list[FullStoreSlotCandidate] = []
    bindings: list[LocalCitationBinding] = []
    lineages: list[FactSeededRecoveryLineage] = []
    tokens = 0
    hydration_truncated = False
    for match in batch.matches:
        cue_receipts = match.supporting_cue_receipt_sha256s
        facts = _ordered_unique(
            fact_receipt
            for receipt in cue_receipts
            for fact_receipt in fact_receipts_by_parent[
                cue_by_receipt[receipt].parent_receipt_sha256
            ]
        )
        source_span_receipt = citation_span_receipt_sha256(match.local_binding)
        if source_span_receipt in parent_span_receipts:
            status: DecisionStatus = "duplicate_first_pass_span"
            candidate = None
            local = None
            reason = "selected_after_scan_then_excluded_as_exact_first_pass_span"
            hydration_truncated = True
        elif source_span_receipt in seen_source_spans:
            status = "duplicate_recovered_span"
            candidate = None
            local = None
            reason = "selected_after_scan_then_excluded_as_repeated_source_window"
            hydration_truncated = True
        else:
            seen_source_spans.add(source_span_receipt)
            candidate, local, hydration_kind, row_receipt = _hydrated_candidate(
                index, parent_result, match, parent_spans, budget
            )
            recovered_span = citation_span_receipt_sha256(local)
            if recovered_span in seen_recovered_spans:
                status = "duplicate_recovered_span"
                candidate = None
                local = None
                reason = "selected_after_scan_then_excluded_as_repeated_hydrated_span"
                hydration_truncated = True
            elif (
                len(candidates) >= budget.max_hydrated_candidates
                or tokens + candidate.token_count > budget.max_hydrated_tokens
            ):
                status = "hydration_budget_excluded"
                candidate = None
                local = None
                reason = "selected_after_scan_then_excluded_by_hydration_budget"
                hydration_truncated = True
            else:
                seen_recovered_spans.add(recovered_span)
                candidates.append(candidate)
                bindings.append(local)
                tokens += candidate.token_count
                status = (
                    "admitted_enclosing_row"
                    if hydration_kind == "enclosing_row"
                    else "admitted_exact_window"
                )
                reason = (
                    "exact_cached_enclosing_row"
                    if hydration_kind == "enclosing_row"
                    else "exact_scanner_window_preserved_for_overlap_or_row_bound"
                )
                lineage = FactSeededRecoveryLineage(
                    supporting_fact_receipt_sha256s=facts,
                    cue_receipt_sha256s=cue_receipts,
                    scan_match_receipt_sha256=match.receipt_sha256,
                    source_window_span_receipt_sha256=source_span_receipt,
                    recovered_candidate_receipt_sha256=(
                        candidate_projection_receipt_sha256(candidate)
                    ),
                    recovered_local_binding_receipt_sha256=local.receipt_sha256,
                    recovered_span_receipt_sha256=recovered_span,
                    cached_row_receipt_sha256=row_receipt,
                    hydration_kind=hydration_kind,
                )
                lineages.append(lineage)
        decisions.append(
            FactSeededRecoveryDecision(
                scan_match_receipt_sha256=match.receipt_sha256,
                status=status,
                supporting_fact_receipt_sha256s=facts,
                recovered_candidate_receipt_sha256=(
                    None
                    if candidate is None
                    else candidate_projection_receipt_sha256(candidate)
                ),
                recovered_local_binding_receipt_sha256=(
                    None if local is None else local.receipt_sha256
                ),
                reason=reason,
            )
        )
    return TypedFactSeededReconstructionResult(
        index=index,
        parent_result=parent_result,
        packet=packet,
        provenance=provenance,
        budget=budget,
        status="scanned",
        reason="one_bounded_fact_seeded_global_scan_completed",
        seed_items=items,
        seed_parse_receipt_sha256=parse_receipt,
        request=request,
        batch=batch,
        decisions=tuple(decisions),
        candidates=tuple(candidates),
        local_bindings=tuple(bindings),
        lineages=tuple(lineages),
        cue_derivation_truncated=cue_truncated,
        hydration_truncated=hydration_truncated,
    )


def adapt_typed_fact_seeded_reconstruction_to_contribution(
    result: TypedFactSeededReconstructionResult,
    /,
    *,
    handle_start: int,
    group_start: int,
) -> TypedEvidenceContribution:
    """Expose exact hydrated children as a bounded direct-pointer lane."""

    _require(
        type(result) is TypedFactSeededReconstructionResult,
        "fact seed contribution requires an exact result",
    )
    for value, label in (
        (handle_start, "fact seed handle start"),
        (group_start, "fact seed group start"),
    ):
        _require(type(value) is int and 1 <= value <= 999_999, f"{label} changed")
    _require(
        not result.candidates
        or handle_start + len(result.candidates) - 1 <= 999_999,
        "fact seed handle range overflowed",
    )
    source_keys = _ordered_unique(
        f"{local.namespace_id}\0{local.source_id}"
        for local in result.local_bindings
    )
    _require(
        not source_keys or group_start + len(source_keys) - 1 <= 999_999,
        "fact seed group range overflowed",
    )
    groups = {
        key: f"G{group_start + index:03d}"
        for index, key in enumerate(source_keys)
    }
    sealed = identity_sha256(result.local_audit_projection())
    bindings: list[EvidenceHandleBinding] = []
    raw_items: list[dict[str, Any]] = []
    numeric_slots = {
        slot.slot_id
        for slot in result.parent_result.operator_spec.required_slots
        if slot.requires_numeric
    }
    for index, (candidate, local, lineage) in enumerate(
        zip(result.candidates, result.local_bindings, result.lineages, strict=True)
    ):
        handle = f"H{handle_start + index:03d}"
        group = groups[f"{local.namespace_id}\0{local.source_id}"]
        binding = EvidenceHandleBinding(
            handle_id=handle,
            origin=EvidenceOrigin.DIRECT_POINTER,
            provenance_grade=ProvenanceGrade.DIRECT_POINTER,
            source_group_handle=group,
            sealed_artifact_sha256=sealed,
            parent_receipt_sha256=lineage.receipt_sha256,
            evidence_receipt_sha256=local.receipt_sha256,
            payload_sha256=candidate_projection_receipt_sha256(candidate),
            citation_sha256=candidate.quote_sha256,
            citation_char_count=len(candidate.quote),
            local_source_locator_sha256=local.receipt_sha256,
        )
        numeric = (
            conservative_numeric_value(candidate.quote)
            if numeric_slots & set(candidate.supported_slot_ids)
            else None
        )
        if numeric is not None:
            kind = "operand"
        elif result.parent_result.operator_spec.temporal_mode.value != "none":
            kind = "event"
        elif result.parent_result.operator_spec.answer_shape.value == "set_list":
            kind = "member"
        elif result.parent_result.operator_spec.style.value == "state_chain":
            kind = "state"
        else:
            kind = "direct"
        semantic_slots = tuple(
            slot
            for slot in result.parent_result.operator_spec.required_slots
            if slot.slot_id in candidate.supported_slot_ids
            and slot.kind in {SlotKind.OPERAND, SlotKind.COMPARISON_SIDE}
        )
        raw: dict[str, Any] = {
            "handle_ids": [handle],
            "included": True,
            "kind": kind,
            "numeric_role": "operand" if numeric is not None else "none",
            "specificity_terms": [],
            "summary": candidate.quote,
            "value_authority": "explicit",
        }
        if len(semantic_slots) == 1:
            raw["entity_key"] = semantic_slots[0].label
        if candidate.event_date is not None:
            raw["date"] = candidate.event_date
        if candidate.role in {"user", "assistant"}:
            raw["relation"] = f"authored_by_{candidate.role}"
        if numeric is not None:
            raw["numeric_value"] = numeric
        bindings.append(binding)
        raw_items.append(raw)
    exact_bindings = tuple(bindings)
    parsed = parse_typed_items(
        raw_items,
        operator_spec=result.parent_result.operator_spec,
        bindings=exact_bindings,
    )
    return TypedEvidenceContribution(
        mechanism_id=MECHANISM_ID,
        bindings=exact_bindings,
        parsed=parsed,
        sealed_artifact_sha256=sealed,
        frontier_mode=FrontierMode.BOUNDED,
        truncated=result.truncated or result.status != "scanned",
    )


__all__ = [
    "FactSeedHandleProof",
    "FactSeedProvenance",
    "FactSeededReconstructionBudget",
    "FactSeededRecoveryDecision",
    "FactSeededRecoveryLineage",
    "MECHANISM_ID",
    "TypedFactSeededReconstructionError",
    "TypedFactSeededReconstructionResult",
    "adapt_typed_fact_seeded_reconstruction_to_contribution",
    "build_fact_seed_provenance",
    "rematerialize_evidence_handle_bindings",
    "run_typed_fact_seeded_reconstruction",
]
