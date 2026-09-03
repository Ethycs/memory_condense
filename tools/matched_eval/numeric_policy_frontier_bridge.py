"""Sound full-store frontier bridge for the operator-first numeric policy.

The numeric operand specialist proves that it physically visited the complete
resident :class:`FullStoreWindowIndex`, but intentionally labels semantic
completeness ``not_claimed``.  This bridge preserves that contract.  It uses
the specialist receipt only as the physical-scan anchor, then runs the exact
operator-first grammar over every immutable full content row and maps that
policy-specific census back to the sealed provider packet.

A frontier closes only when the question is in the explicitly supported
grammar and provider/census agree bidirectionally on both their non-empty
semantic-key set and every material fact within each key.  Material facts bind
action, entity, event, status, event time and basis, role, numeric value/unit,
contribution, and coherence state.  Every fact also needs an exact source-
surface binding.  Legacy specialist selection truncation is recorded but does
not constrain this independent unbounded census.  A paraphrase without its
exact underlying surface therefore remains open rather than turning physical
exhaustion into an unsupported semantic claim.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from memory_condense.domain.discourse import quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .full_store_slot_closure import FullStoreWindowIndex
from .numeric_operand_specialist import NumericOperandClosureResult
from .operator_first_numeric_policy import (
    NumericCandidateAtom,
    NumericPolicyMode,
    OperatorFirstNumericCompilation,
    RelevantNumericFrontier,
    build_relevant_numeric_frontier,
    compile_operator_first_numeric_candidates,
)
from .typed_action_semantics import completed_action_concepts


BRIDGE_FORMAT = "memory-condense-numeric-policy-frontier-bridge-v2"
CENSUS_ATOM_FORMAT = "memory-condense-numeric-policy-census-atom-v2"
MATERIAL_FACT_FORMAT = "memory-condense-numeric-policy-material-fact-v2"
POLICY_GRAMMAR_ID = "operator-first-numeric-supported-grammar-v2"
SUPPORTED_DOMAINS = frozenset(
    {"bike", "plant", "cuisine", "clothing"}
)
EXTENDED_SUPPORTED_DOMAINS = frozenset(
    {*SUPPORTED_DOMAINS, "jewelry", "museum_gallery"}
)
SUPPORTED_COUNT_MODES = frozenset(
    {
        NumericPolicyMode.DISTINCT_ENTITY_COUNT,
        NumericPolicyMode.ENTITY_EVENT_COUNT,
        NumericPolicyMode.ACTION_OBLIGATION_COUNT,
    }
)


class NumericPolicyFrontierBridgeError(MatchedEvalContractError):
    """Raised when the full-store and specialist lifecycles do not bind."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise NumericPolicyFrontierBridgeError(message)


def _ordered_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(
        type(values) is tuple
        and all(type(value) is str and value for value in values)
        and len(values) == len(set(values)),
        f"{label} must be an ordered unique tuple",
    )
    return values


def _semantic_key(atom: NumericCandidateAtom, mode: NumericPolicyMode) -> str:
    if mode is NumericPolicyMode.ACTION_OBLIGATION_COUNT:
        return f"{atom.action_key}:{atom.entity_key}"
    if mode is NumericPolicyMode.DISTINCT_ENTITY_COUNT:
        return atom.entity_key
    return atom.event_key


def _semantic_key_sha256(value: str) -> str:
    return identity_sha256(
        {"grammar_id": POLICY_GRAMMAR_ID, "semantic_key": value}
    )


def _surface(value: str) -> str:
    return " ".join(value.casefold().split())


def _exact_surface_bound(provider_summary: str, source_row: str) -> bool:
    provider = _surface(provider_summary)
    source = _surface(source_row)
    return bool(provider and source and (provider in source or source in provider))


def _status_for_row(text: str, mode: NumericPolicyMode) -> str:
    if re.search(r"\b(?:cancelled|canceled|abandoned|superseded)\b", text, re.I):
        return "cancelled"
    if mode is NumericPolicyMode.ACTION_OBLIGATION_COUNT and re.search(
        r"\b(?:need|have\s+to|not\s+yet|haven't|have\s+not|awaiting)\b",
        text,
        re.I,
    ):
        return "current"
    if re.search(
        r"\b(?:plan(?:ned)?|intend|want\s+to|need\s+to|time\s+to|"
        r"scheduled?|due\s+to)\b",
        text,
        re.I,
    ):
        return "proposed"
    if completed_action_concepts(text):
        return "completed"
    return "unknown"


def _material_fact_payload(
    *,
    action_key: str,
    content_coherence: str,
    contribution_value: float,
    entity_key: str,
    event_date: str | None,
    event_key: str,
    included: bool,
    numeric_value: float | None,
    source_role: str,
    status: str,
    temporal_basis: str,
    unit: str | None,
) -> dict[str, Any]:
    return {
        "action_key": action_key,
        "content_coherence": content_coherence,
        "contribution_value": contribution_value,
        "entity_key": entity_key,
        "event_date": event_date,
        "event_key": event_key,
        "format": MATERIAL_FACT_FORMAT,
        "included": included,
        "numeric_value": numeric_value,
        "source_role": source_role,
        "status": status,
        "temporal_basis": temporal_basis,
        "unit": unit,
    }


def _material_fact_projection(
    atom: NumericCandidateAtom,
    item: Mapping[str, Any],
    *,
    operator_material_status: bool = False,
) -> dict[str, Any]:
    """Project every answer-material field at the reducer's fact boundary."""

    return _material_fact_payload(
        action_key=atom.action_key,
        content_coherence=str(item.get("content_coherence", "match")),
        contribution_value=atom.contribution_value,
        entity_key=atom.entity_key,
        event_date=atom.event_date,
        event_key=atom.event_key,
        included=item.get("included", True) is True,
        numeric_value=atom.numeric_value,
        source_role=atom.source_role,
        # Once the compiler has admitted an atom, distinctions such as
        # ``unknown`` versus ``completed`` are not material to the current
        # count reducer.  Excluded/cancelled/proposed states have already been
        # handled by the compiler gate.  The successor profile can therefore
        # compare the operator-visible eligibility state while v2 keeps its
        # exact historical status contract for byte replay.
        status=("operator_eligible" if operator_material_status else atom.status),
        temporal_basis=atom.temporal_basis.value,
        unit=atom.unit,
    )


def _material_fact_sha256(
    atom: NumericCandidateAtom,
    item: Mapping[str, Any],
    *,
    operator_material_status: bool = False,
) -> str:
    return identity_sha256(
        _material_fact_projection(
            atom,
            item,
            operator_material_status=operator_material_status,
        )
    )


def _provider_parts(
    provider_input: Mapping[str, Any],
) -> tuple[str, dict[str, Any], tuple[dict[str, Any], ...]]:
    _require(isinstance(provider_input, Mapping), "provider input changed type")
    assert_gold_blind(provider_input, path="numeric_policy_frontier_provider_input")
    question = provider_input.get("dated_question")
    typed = provider_input.get("typed_evidence")
    _require(type(question) is str and bool(question.strip()), "dated question missing")
    _require(isinstance(typed, Mapping), "typed evidence changed type")
    operator = typed.get("operator_spec")
    items = typed.get("items")
    _require(isinstance(operator, Mapping), "typed operator changed type")
    _require(
        type(items) is list and all(isinstance(row, Mapping) for row in items),
        "typed evidence item inventory changed",
    )
    return (
        question,
        dict(operator),
        tuple(dict(row) for row in items),
    )


def _synthetic_full_store_input(
    provider_input: Mapping[str, Any],
    index: FullStoreWindowIndex,
    mode: NumericPolicyMode,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    question, operator, _items = _provider_parts(provider_input)
    _require(
        len(index.rows) <= 999_999,
        "policy census exceeds its opaque handle namespace",
    )
    handles: list[dict[str, str]] = []
    items: list[dict[str, Any]] = []
    row_item_by_sha256: dict[str, dict[str, Any]] = {}
    for offset, row in enumerate(index.rows, start=1):
        handle = f"H{offset:06d}"
        group = f"G{offset:06d}"
        text = row.text
        role = row.role.casefold()
        item: dict[str, Any] = {
            "content_coherence": "match",
            "date": row.created_at,
            "handle_ids": [handle],
            "included": True,
            "kind": "direct",
            "relation": f"authored_by_{role};date_basis=source_created_at",
            "status": _status_for_row(text, mode),
            "summary": text,
            "supported_slot_ids": [],
            "value_authority": "explicit",
        }
        handles.append(
            {
                "group_handle": group,
                "handle_id": handle,
                "origin": "direct_pointer",
            }
        )
        items.append(item)
        row_item_by_sha256[identity_sha256(item)] = item
    synthetic = {
        "dated_question": question,
        "typed_evidence": {
            "conflict_policy": "quarantine",
            "format": "numeric-policy-full-store-census-v2",
            "frontier": {
                "closed": False,
                "mode": "open",
                "truncated": True,
            },
            "handles": handles,
            "items": items,
            "operator_spec": operator,
        },
    }
    return synthetic, row_item_by_sha256


@dataclass(frozen=True, slots=True)
class NumericPolicyCensusAtom:
    semantic_key_sha256: str
    material_fact_sha256: str
    action_key: str
    entity_key: str
    event_key: str
    status: str
    event_date: str | None
    temporal_basis: str
    source_role: str
    numeric_value: float | None
    unit: str | None
    contribution_value: float
    content_coherence: str
    included: bool
    source_content_row_sha256: str
    candidate_atom_receipt_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.semantic_key_sha256, "census semantic key"),
            (self.material_fact_sha256, "census material fact"),
            (self.source_content_row_sha256, "census source content row"),
            (self.candidate_atom_receipt_sha256, "census candidate atom"),
        ):
            require_sha256(value, label)
        for value, label in (
            (self.action_key, "census action"),
            (self.entity_key, "census entity"),
            (self.event_key, "census event"),
            (self.status, "census status"),
            (self.temporal_basis, "census temporal basis"),
            (self.source_role, "census source role"),
            (self.content_coherence, "census conflict state"),
        ):
            require_text(value, label)
        if self.event_date is not None:
            require_text(self.event_date, "census event date")
        if self.unit is not None:
            require_text(self.unit, "census unit")
        if self.numeric_value is not None and (
            type(self.numeric_value) not in {int, float}
            or not math.isfinite(float(self.numeric_value))
        ):
            raise NumericPolicyFrontierBridgeError("census numeric value changed")
        _require(
            type(self.contribution_value) in {int, float}
            and math.isfinite(float(self.contribution_value))
            and float(self.contribution_value) > 0,
            "census contribution changed",
        )
        _require(type(self.included) is bool, "census inclusion state changed")
        expected_fact = identity_sha256(
            _material_fact_payload(
                action_key=self.action_key,
                content_coherence=self.content_coherence,
                contribution_value=self.contribution_value,
                entity_key=self.entity_key,
                event_date=self.event_date,
                event_key=self.event_key,
                included=self.included,
                numeric_value=self.numeric_value,
                source_role=self.source_role,
                status=self.status,
                temporal_basis=self.temporal_basis,
                unit=self.unit,
            )
        )
        _require(
            self.material_fact_sha256 == expected_fact,
            "census material-fact tuple changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise NumericPolicyFrontierBridgeError("census atom changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "action_key": self.action_key,
            "candidate_atom_receipt_sha256": self.candidate_atom_receipt_sha256,
            "entity_key": self.entity_key,
            "event_key": self.event_key,
            "status": self.status,
            "event_date": self.event_date,
            "temporal_basis": self.temporal_basis,
            "source_role": self.source_role,
            "numeric_value": self.numeric_value,
            "unit": self.unit,
            "contribution_value": self.contribution_value,
            "content_coherence": self.content_coherence,
            "included": self.included,
            "format": CENSUS_ATOM_FORMAT,
            "material_fact_sha256": self.material_fact_sha256,
            "semantic_key_sha256": self.semantic_key_sha256,
            "source_content_row_sha256": self.source_content_row_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class NumericPolicyFrontierBridgeResult:
    frontier: RelevantNumericFrontier
    applicable: bool
    policy_grammar_id: Literal[
        "operator-first-numeric-supported-grammar-v2"
    ]
    question_sha256: str
    window_index_receipt_sha256: str
    specialist_receipt_sha256: str
    provider_compilation_receipt_sha256: str
    census_compilation_receipt_sha256: str
    candidate_population_receipt_sha256: str
    physical_content_rows_scanned: int
    physical_sentence_windows_scanned: int
    specialist_selection_truncated: bool
    census_atoms: tuple[NumericPolicyCensusAtom, ...]
    census_semantic_key_sha256s: tuple[str, ...]
    provider_semantic_key_sha256s: tuple[str, ...]
    represented_semantic_key_sha256s: tuple[str, ...]
    census_material_fact_sha256s: tuple[str, ...]
    provider_material_fact_sha256s: tuple[str, ...]
    represented_material_fact_sha256s: tuple[str, ...]
    unresolved_candidate_keys: tuple[str, ...]
    represented_handle_ids: tuple[str, ...]
    specialist_semantic_completeness_status: Literal["not_claimed"] = "not_claimed"
    policy_semantic_completeness_scope: Literal["versioned_supported_grammar"] = (
        "versioned_supported_grammar"
    )
    physical_scan_exhaustive: Literal[True] = True
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.frontier) is RelevantNumericFrontier,
            "bridge frontier changed type",
        )
        _require(type(self.applicable) is bool, "bridge applicability changed")
        for value, label in (
            (self.question_sha256, "bridge question"),
            (self.window_index_receipt_sha256, "bridge window index"),
            (self.specialist_receipt_sha256, "bridge specialist"),
            (self.provider_compilation_receipt_sha256, "bridge provider compilation"),
            (self.census_compilation_receipt_sha256, "bridge census compilation"),
            (self.candidate_population_receipt_sha256, "bridge population"),
        ):
            require_sha256(value, label)
        _require(
            self.policy_grammar_id == POLICY_GRAMMAR_ID
            and self.specialist_semantic_completeness_status == "not_claimed"
            and self.policy_semantic_completeness_scope
            == "versioned_supported_grammar"
            and self.physical_scan_exhaustive is True,
            "bridge confused physical and semantic completeness",
        )
        _require(
            type(self.physical_content_rows_scanned) is int
            and self.physical_content_rows_scanned >= 0
            and type(self.physical_sentence_windows_scanned) is int
            and self.physical_sentence_windows_scanned >= 0,
            "bridge physical scan counts changed",
        )
        _require(
            type(self.specialist_selection_truncated) is bool,
            "specialist selection audit flag changed",
        )
        _require(
            type(self.census_atoms) is tuple
            and all(type(row) is NumericPolicyCensusAtom for row in self.census_atoms),
            "bridge census changed type",
        )
        _ordered_unique(
            self.census_semantic_key_sha256s,
            "census semantic keys",
        )
        _ordered_unique(
            self.provider_semantic_key_sha256s,
            "provider semantic keys",
        )
        _ordered_unique(
            self.represented_semantic_key_sha256s,
            "represented semantic keys",
        )
        _ordered_unique(
            self.census_material_fact_sha256s,
            "census material facts",
        )
        _ordered_unique(
            self.provider_material_fact_sha256s,
            "provider material facts",
        )
        _ordered_unique(
            self.represented_material_fact_sha256s,
            "represented material facts",
        )
        _ordered_unique(self.unresolved_candidate_keys, "unresolved candidates")
        _ordered_unique(self.represented_handle_ids, "represented handles")
        _require(
            self.frontier.candidate_population_receipt_sha256
            == self.candidate_population_receipt_sha256
            and self.frontier.unresolved_candidate_keys
            == self.unresolved_candidate_keys
            and self.frontier.represented_handle_ids == self.represented_handle_ids,
            "bridge frontier lost its census mapping",
        )
        expected_closed = bool(
            self.applicable
            and self.census_semantic_key_sha256s
            and not self.unresolved_candidate_keys
            and set(self.census_semantic_key_sha256s)
            == set(self.provider_semantic_key_sha256s)
            == set(self.represented_semantic_key_sha256s)
            and set(self.census_material_fact_sha256s)
            == set(self.provider_material_fact_sha256s)
            == set(self.represented_material_fact_sha256s)
        )
        _require(
            self.frontier.closed is expected_closed,
            "bridge closure is not justified by exact census/provider equality",
        )
        _require(
            self.provider_prompt_count == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False,
            "bridge must remain provider-free, zero-state, and gold-blind",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise NumericPolicyFrontierBridgeError("numeric frontier bridge changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="numeric_policy_frontier_bridge")

    @property
    def closed(self) -> bool:
        return self.frontier.closed

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "applicable": self.applicable,
            "candidate_population_receipt_sha256": (
                self.candidate_population_receipt_sha256
            ),
            "census_atoms": [row.projection() for row in self.census_atoms],
            "census_compilation_receipt_sha256": (
                self.census_compilation_receipt_sha256
            ),
            "census_semantic_key_sha256s": list(
                self.census_semantic_key_sha256s
            ),
            "census_material_fact_sha256s": list(
                self.census_material_fact_sha256s
            ),
            "format": BRIDGE_FORMAT,
            "frontier": self.frontier.projection(),
            "gold_loaded": False,
            "physical_content_rows_scanned": self.physical_content_rows_scanned,
            "physical_scan_exhaustive": True,
            "physical_sentence_windows_scanned": (
                self.physical_sentence_windows_scanned
            ),
            "policy_grammar_id": self.policy_grammar_id,
            "policy_semantic_completeness_scope": (
                self.policy_semantic_completeness_scope
            ),
            "policy_semantic_census_unit": "full_immutable_content_row",
            "provider_compilation_receipt_sha256": (
                self.provider_compilation_receipt_sha256
            ),
            "provider_prompt_count": 0,
            "provider_semantic_key_sha256s": list(
                self.provider_semantic_key_sha256s
            ),
            "provider_material_fact_sha256s": list(
                self.provider_material_fact_sha256s
            ),
            "question_sha256": self.question_sha256,
            "represented_handle_ids": list(self.represented_handle_ids),
            "represented_semantic_key_sha256s": list(
                self.represented_semantic_key_sha256s
            ),
            "represented_material_fact_sha256s": list(
                self.represented_material_fact_sha256s
            ),
            "retained_transformer_token_state_bytes": 0,
            "specialist_receipt_sha256": self.specialist_receipt_sha256,
            "specialist_selection_truncated": (
                self.specialist_selection_truncated
            ),
            "specialist_semantic_completeness_status": "not_claimed",
            "unresolved_candidate_keys": list(self.unresolved_candidate_keys),
            "window_index_receipt_sha256": self.window_index_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _validate_physical_anchor(
    *,
    question: str,
    index: FullStoreWindowIndex,
    specialist_result: NumericOperandClosureResult,
) -> None:
    _require(
        type(index) is FullStoreWindowIndex,
        "bridge requires an exact immutable full-store index",
    )
    _require(
        type(specialist_result) is NumericOperandClosureResult,
        "bridge requires an exact numeric specialist result",
    )
    receipt = specialist_result.receipt
    _require(
        specialist_result.dated_question == question
        and receipt.question_sha256 == quote_sha256(question),
        "specialist result belongs to another question",
    )
    _require(
        receipt.window_index_receipt_sha256 == index.receipt_sha256
        and receipt.cache_receipt_sha256 == index.cache.cache_receipt_sha256,
        "specialist result belongs to another resident index",
    )
    _require(
        receipt.physical_scan_exhaustive is True
        and receipt.physical_content_rows_scanned == len(index.rows)
        and receipt.physical_sentence_windows_scanned == len(index.windows),
        "specialist physical scan receipt does not cover the resident index",
    )
    _require(
        receipt.semantic_completeness_status == "not_claimed",
        "specialist semantic completeness contract changed",
    )


def _item_records(
    items: tuple[dict[str, Any], ...]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in items:
        summary = item.get("summary")
        _require(type(summary) is str and bool(summary), "typed item summary changed")
        output[identity_sha256(item)] = item
    return output


def operator_first_numeric_frontier_applicable(
    provider_input: Mapping[str, Any],
    /,
    *,
    supported_domains: frozenset[str] = SUPPORTED_DOMAINS,
) -> bool:
    """Return whether the sealed question is covered by this census grammar.

    This is a question/evidence-schema preflight only.  ``True`` does not mean
    the eventual frontier will close; the full index and exact mapping still
    decide that.
    """

    _require(
        type(supported_domains) is frozenset
        and bool(supported_domains)
        and all(type(value) is str and value for value in supported_domains),
        "supported domain policy changed type",
    )
    compilation = compile_operator_first_numeric_candidates(provider_input)
    return bool(
        compilation.mode in SUPPORTED_COUNT_MODES
        and compilation.question_domain in supported_domains
        and compilation.query_action_keys
    )


def build_operator_first_numeric_frontier(
    provider_input: Mapping[str, Any],
    /,
    *,
    index: FullStoreWindowIndex,
    specialist_result: NumericOperandClosureResult,
    supported_domains: frozenset[str] = SUPPORTED_DOMAINS,
    operator_material_status: bool = False,
) -> NumericPolicyFrontierBridgeResult:
    """Build a policy frontier from a complete resident-index census.

    This entry point accepts no question ID, reference answer, prediction,
    provider, or model state.  An open result is a normal, auditable outcome.
    """

    question, _operator, provider_items = _provider_parts(provider_input)
    _require(
        type(operator_material_status) is bool,
        "operator-material status policy changed type",
    )
    _validate_physical_anchor(
        question=question,
        index=index,
        specialist_result=specialist_result,
    )
    provider_compilation = compile_operator_first_numeric_candidates(provider_input)
    synthetic, census_item_by_sha256 = _synthetic_full_store_input(
        provider_input,
        index,
        provider_compilation.mode,
    )
    census_compilation = compile_operator_first_numeric_candidates(synthetic)
    _require(
        census_compilation.mode is provider_compilation.mode
        and census_compilation.question_domain == provider_compilation.question_domain
        and census_compilation.query_action_keys
        == provider_compilation.query_action_keys,
        "full-store census changed the question-derived policy",
    )

    census_pairs: list[tuple[str, str, NumericCandidateAtom, str]] = []
    census_atoms: list[NumericPolicyCensusAtom] = []
    for atom in census_compilation.candidate_atoms:
        source_item = census_item_by_sha256.get(atom.item_sha256)
        _require(source_item is not None, "census atom lost its indexed content row")
        source_row = str(source_item["summary"])
        key = _semantic_key(atom, census_compilation.mode)
        fact_sha256 = _material_fact_sha256(
            atom,
            source_item,
            operator_material_status=operator_material_status,
        )
        census_pairs.append((key, fact_sha256, atom, source_row))
        census_atoms.append(
            NumericPolicyCensusAtom(
                semantic_key_sha256=_semantic_key_sha256(key),
                material_fact_sha256=fact_sha256,
                action_key=atom.action_key,
                entity_key=atom.entity_key,
                event_key=atom.event_key,
                status=(
                    "operator_eligible" if operator_material_status else atom.status
                ),
                event_date=atom.event_date,
                temporal_basis=atom.temporal_basis.value,
                source_role=atom.source_role,
                numeric_value=atom.numeric_value,
                unit=atom.unit,
                contribution_value=atom.contribution_value,
                content_coherence=str(
                    source_item.get("content_coherence", "match")
                ),
                included=source_item.get("included", True) is True,
                source_content_row_sha256=quote_sha256(source_row),
                candidate_atom_receipt_sha256=atom.receipt_sha256,
            )
        )

    provider_item_by_sha256 = _item_records(provider_items)
    provider_by_key: dict[
        str, list[tuple[str, NumericCandidateAtom, str]]
    ] = {}
    for atom in provider_compilation.candidate_atoms:
        provider_item = provider_item_by_sha256.get(atom.item_sha256)
        _require(provider_item is not None, "provider atom lost its typed item")
        summary = str(provider_item["summary"])
        fact_sha256 = _material_fact_sha256(
            atom,
            provider_item,
            operator_material_status=operator_material_status,
        )
        provider_by_key.setdefault(
            _semantic_key(atom, provider_compilation.mode), []
        ).append((fact_sha256, atom, summary))
    census_by_key: dict[
        str, list[tuple[str, NumericCandidateAtom, str]]
    ] = {}
    for key, fact_sha256, atom, source in census_pairs:
        census_by_key.setdefault(key, []).append((fact_sha256, atom, source))

    unresolved: list[str] = []
    supported = operator_first_numeric_frontier_applicable(
        provider_input,
        supported_domains=supported_domains,
    )
    if not supported:
        unresolved.append(
            "unsupported_policy_scope:"
            + identity_sha256(
                {
                    "domain": provider_compilation.question_domain,
                    "mode": provider_compilation.mode.value,
                    "query_action_keys": list(
                        provider_compilation.query_action_keys
                    ),
                }
            )[:16]
        )
    elif not census_by_key:
        # The current reducer intentionally does not prove zero-count answers.
        # Keep its frontier equally conservative even though an all-window
        # grammar pass physically observed no candidate.
        unresolved.append("empty_candidate_census")

    represented_facts: list[str] = []
    for key, occurrences in census_by_key.items():
        providers = provider_by_key.get(key, [])
        if not providers:
            unresolved.append("missing:" + _semantic_key_sha256(key)[:16])
            continue
        census_fact_ids = tuple(dict.fromkeys(row[0] for row in occurrences))
        provider_fact_ids = {row[0] for row in providers}
        for fact_sha256 in census_fact_ids:
            fact_occurrences = tuple(
                row for row in occurrences if row[0] == fact_sha256
            )
            fact_providers = tuple(
                row for row in providers if row[0] == fact_sha256
            )
            if fact_sha256 not in provider_fact_ids:
                unresolved.append("missing_fact:" + fact_sha256[:16])
                continue
            if not any(
                _exact_surface_bound(summary, source)
                for _provider_fact, _provider_atom, summary in fact_providers
                for _census_fact, _census_atom, source in fact_occurrences
            ):
                unresolved.append("surface_unbound_fact:" + fact_sha256[:16])
                continue
            represented_facts.append(fact_sha256)

    for key, providers in provider_by_key.items():
        occurrences = census_by_key.get(key, [])
        if not occurrences:
            unresolved.append("provider_only:" + _semantic_key_sha256(key)[:16])
            continue
        census_fact_ids = {row[0] for row in occurrences}
        for fact_sha256, _provider_atom, _summary in providers:
            if fact_sha256 not in census_fact_ids:
                unresolved.append("provider_only_fact:" + fact_sha256[:16])

    census_fact_ids_by_key = {
        key: {row[0] for row in rows} for key, rows in census_by_key.items()
    }
    provider_fact_ids_by_key = {
        key: {row[0] for row in rows} for key, rows in provider_by_key.items()
    }
    represented_fact_set = set(represented_facts)
    represented_keys = [
        _semantic_key_sha256(key)
        for key, fact_ids in census_fact_ids_by_key.items()
        if fact_ids
        and fact_ids == provider_fact_ids_by_key.get(key, set())
        and fact_ids <= represented_fact_set
    ]

    specialist_truncated = bool(
        specialist_result.receipt.selection_truncated
        or not specialist_result.receipt.all_plausible_operand_groups_reserved
    )

    unresolved_keys = tuple(dict.fromkeys(unresolved))
    represented_handle_ids = tuple(
        dict.fromkeys(
            handle
            for atom in provider_compilation.candidate_atoms
            for handle in atom.handle_ids
        )
    )
    population_projection = {
        "census_atom_receipt_sha256s": [
            row.receipt_sha256 for row in census_atoms
        ],
        "census_material_fact_sha256s": list(
            dict.fromkeys(row[1] for row in census_pairs)
        ),
        "census_compilation_receipt_sha256": (
            census_compilation.receipt_sha256
        ),
        "format": "memory-condense-numeric-policy-candidate-population-v2",
        "policy_semantic_census_unit": "full_immutable_content_row",
        "physical_content_rows_scanned": len(index.rows),
        "physical_sentence_windows_scanned": len(index.windows),
        "policy_grammar_id": POLICY_GRAMMAR_ID,
        "question_sha256": quote_sha256(question),
        "specialist_receipt_sha256": specialist_result.receipt.receipt_sha256,
        "window_index_receipt_sha256": index.receipt_sha256,
    }
    candidate_population_receipt_sha256 = identity_sha256(population_projection)
    frontier = build_relevant_numeric_frontier(
        provider_input,
        candidate_population_receipt_sha256=(
            candidate_population_receipt_sha256
        ),
        represented_handle_ids=represented_handle_ids,
        unresolved_candidate_keys=unresolved_keys,
        # The bridge ledger visits every immutable window and has no candidate
        # cap.  Legacy specialist truncation is preserved below as audit
        # metadata, but its broader/narrower semantic draft set cannot veto
        # this policy-specific population.
        selection_truncated=False,
    )
    return NumericPolicyFrontierBridgeResult(
        frontier=frontier,
        applicable=supported,
        policy_grammar_id=POLICY_GRAMMAR_ID,
        question_sha256=quote_sha256(question),
        window_index_receipt_sha256=index.receipt_sha256,
        specialist_receipt_sha256=specialist_result.receipt.receipt_sha256,
        provider_compilation_receipt_sha256=(
            provider_compilation.receipt_sha256
        ),
        census_compilation_receipt_sha256=census_compilation.receipt_sha256,
        candidate_population_receipt_sha256=(
            candidate_population_receipt_sha256
        ),
        physical_content_rows_scanned=len(index.rows),
        physical_sentence_windows_scanned=len(index.windows),
        specialist_selection_truncated=specialist_truncated,
        census_atoms=tuple(census_atoms),
        census_semantic_key_sha256s=tuple(
            _semantic_key_sha256(key) for key in census_by_key
        ),
        provider_semantic_key_sha256s=tuple(
            _semantic_key_sha256(key) for key in provider_by_key
        ),
        represented_semantic_key_sha256s=tuple(dict.fromkeys(represented_keys)),
        census_material_fact_sha256s=tuple(
            dict.fromkeys(row[1] for row in census_pairs)
        ),
        provider_material_fact_sha256s=tuple(
            dict.fromkeys(
                fact_sha256
                for rows in provider_by_key.values()
                for fact_sha256, _atom, _summary in rows
            )
        ),
        represented_material_fact_sha256s=tuple(
            dict.fromkeys(represented_facts)
        ),
        unresolved_candidate_keys=unresolved_keys,
        represented_handle_ids=represented_handle_ids,
    )


__all__ = [
    "BRIDGE_FORMAT",
    "CENSUS_ATOM_FORMAT",
    "EXTENDED_SUPPORTED_DOMAINS",
    "MATERIAL_FACT_FORMAT",
    "NumericPolicyCensusAtom",
    "NumericPolicyFrontierBridgeError",
    "NumericPolicyFrontierBridgeResult",
    "POLICY_GRAMMAR_ID",
    "SUPPORTED_DOMAINS",
    "build_operator_first_numeric_frontier",
    "operator_first_numeric_frontier_applicable",
]
