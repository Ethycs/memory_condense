"""Cold-safe structural targets for latent-router training.

The builder in this module consumes only an already packed evidence packet and
its exact closure plan.  It emits position-only direct co-bundle supervision;
it never imports a tensor runtime and never accepts answer, source, category,
or scorer labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass
from itertools import chain
from types import MappingProxyType
from typing import Any

from memory_condense.domain._discourse_identity import (
    _sha256,
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
    ClosureReceipt,
    ClosureScopeWitness,
    DiscourseSnapshot,
    EpisodeSeed,
    EvidenceAtom,
    EvidenceBundle,
    EvidenceObligation,
    EvidencePacket,
    EvidenceSpan,
    ObligationResult,
    QueryProgram,
)
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.fusion.models import FusionCaps
from memory_condense.search.fusion.planner import (
    _atom_refs as _canonical_atom_refs,
    _authoritative_hyperedges as _canonical_authoritative_hyperedges,
)
from memory_condense.search.fusion.resident_models import resident_values_sha256
from memory_condense.search.packing.evidence_packet import (
    _proved_obligation_ids,
    _required_proof_ids,
)


_MAX_TRAINING_ATOMS = 64
_MAX_UNORDERED_PAIRS = 2_016
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_ATOM_REF_SEQUENCE_KIND = "latent_router_training_packet_atom_refs"
_HYPEREDGE_SEQUENCE_KIND = "latent_router_training_authoritative_hyperedges"
_POSITIVE_PAIR_SEQUENCE_KIND = "latent_router_training_positive_pairs"
_NEGATIVE_PAIR_SEQUENCE_KIND = "latent_router_training_negative_pairs"
_CANDIDATE_DROP_REASON = "candidate_cap"
_SELECTABLE_DROP_REASONS = frozenset(
    {"hard_budget", "hard_prompt_budget", "lower_utility"}
)
_CLOSURE_STOP_REASONS = frozenset(
    {
        "complete",
        "frontier_exhausted",
        "budget_exhausted",
        "budget_impossible",
        "workspace_cap",
        "conflicted",
        "not_found",
    }
)


def _exact_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _exact_positive_int(value: object, label: str) -> int:
    normalized = _exact_nonnegative_int(value, label)
    if normalized < 1:
        raise ValueError(f"{label} must be positive")
    return normalized


def _exact_sha256(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    return _sha256(value, label)


def _exact_string(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    return value


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact boolean")
    return value


def _exact_float(value: object, label: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{label} must be an exact float")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _validate_seal_field(value: object, label: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if value:
        _sha256(value, label)


def _bind_expected_sha256(obj: object, name: str, expected: str) -> None:
    current = getattr(obj, name)
    _validate_seal_field(current, name)
    if current and current != expected:
        raise ValueError(f"{name} does not match its canonical sequence")
    object.__setattr__(obj, name, expected)


def _field_values(value: object) -> dict[str, Any]:
    return {item.name: getattr(value, item.name) for item in fields(value)}


def _exact_tuple(value: object, label: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact tuple")
    return value


def _exact_string_tuple(value: object, label: str) -> tuple[str, ...]:
    values = _exact_tuple(value, label)
    if any(type(item) is not str for item in values):
        raise TypeError(f"{label} must contain exact strings")
    return values


def _exact_optional(
    value: object,
    validator: Any,
    label: str,
) -> None:
    if value is not None:
        validator(value, label)


def _validate_frozen_json(value: object, label: str) -> None:
    value_type = type(value)
    if value is None or value_type in {str, int, bool}:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain only finite numbers")
        return
    if value_type is tuple:
        for index, item in enumerate(value):
            _validate_frozen_json(item, f"{label}[{index}]")
        return
    if value_type is _MAPPING_PROXY_TYPE:
        for key, item in value.items():
            _exact_string(key, f"{label} key")
            _validate_frozen_json(item, f"{label}.{key}")
        return
    raise TypeError(f"{label} must be an exact frozen JSON value")


def _require_plain_identity_tree(value: object, label: str) -> None:
    """Reject mutable containers and runtime objects from target receipts."""

    value_type = type(value)
    if value is None or value_type in {str, int}:
        return
    if value_type is tuple:
        for index, item in enumerate(value):
            _require_plain_identity_tree(item, f"{label}[{index}]")
        return
    if is_dataclass(value):
        for item in fields(value):
            _require_plain_identity_tree(
                getattr(value, item.name),
                f"{label}.{item.name}",
            )
        return
    raise TypeError(f"{label} contains an unsupported identity value")


@dataclass(frozen=True, slots=True)
class AtomPositionPairTarget(SealedIdentity):
    """One unordered packet-position pair and its exact binary target."""

    _SEAL_FIELD = "pair_sha256"
    _SEAL_MISMATCH = "atom-position pair SHA-256 does not match its contents"

    left_position: int
    right_position: int
    direct_co_bundle_target: int
    pair_sha256: str = ""

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.left_position, "left_position")
        _exact_nonnegative_int(self.right_position, "right_position")
        _exact_nonnegative_int(
            self.direct_co_bundle_target,
            "direct_co_bundle_target",
        )
        if self.left_position >= self.right_position:
            raise ValueError("atom-position pairs require left_position < right_position")
        if self.right_position >= _MAX_TRAINING_ATOMS:
            raise MemoryError("atom-position pair contains a position at or above 64")
        if self.direct_co_bundle_target not in {0, 1}:
            raise ValueError("direct_co_bundle_target must be exactly zero or one")
        _validate_seal_field(self.pair_sha256, "pair_sha256")
        self._seal()


@dataclass(frozen=True, slots=True)
class DirectCoBundleNeighborhood(SealedIdentity):
    """Self plus every direct co-bundle neighbor for one packet position."""

    _SEAL_FIELD = "neighborhood_sha256"
    _SEAL_MISMATCH = "co-bundle neighborhood SHA-256 does not match its contents"

    atom_position: int
    member_positions: tuple[int, ...]
    neighborhood_sha256: str = ""

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.atom_position, "atom_position")
        if self.atom_position >= _MAX_TRAINING_ATOMS:
            raise MemoryError("neighborhood atom_position must be below 64")
        members = _exact_tuple(self.member_positions, "member_positions")
        if len(members) > _MAX_TRAINING_ATOMS:
            raise MemoryError("a co-bundle neighborhood cannot exceed 64 members")
        for member in members:
            _exact_nonnegative_int(member, "member_position")
            if member >= _MAX_TRAINING_ATOMS:
                raise MemoryError("neighborhood member positions must be below 64")
        if not members:
            raise ValueError("a co-bundle neighborhood must contain self")
        if tuple(sorted(set(members))) != members:
            raise ValueError("member_positions must be unique and ascending")
        if self.atom_position not in members:
            raise ValueError("a co-bundle neighborhood must contain self")
        _validate_seal_field(self.neighborhood_sha256, "neighborhood_sha256")
        self._seal()


@dataclass(frozen=True, slots=True)
class LatentRouterStructuralTargets(SealedIdentity):
    """Position-only numeric supervision, independent of packet metadata."""

    _SEAL_FIELD = "target_sha256"
    _SEAL_MISMATCH = "latent-router structural target SHA-256 does not match its contents"

    atom_count: int
    positive_pairs: tuple[AtomPositionPairTarget, ...]
    negative_pairs: tuple[AtomPositionPairTarget, ...]
    neighborhoods: tuple[DirectCoBundleNeighborhood, ...]
    positive_pair_count: int
    negative_pair_count: int
    positive_pair_sequence_sha256: str = ""
    negative_pair_sequence_sha256: str = ""
    target_sha256: str = ""

    def __post_init__(self) -> None:
        atom_count = _exact_positive_int(self.atom_count, "atom_count")
        if atom_count > _MAX_TRAINING_ATOMS:
            raise MemoryError("structural target atom count exceeds 64")
        positive_pairs = _exact_tuple(self.positive_pairs, "positive_pairs")
        negative_pairs = _exact_tuple(self.negative_pairs, "negative_pairs")
        neighborhoods = _exact_tuple(self.neighborhoods, "neighborhoods")
        if len(positive_pairs) + len(negative_pairs) > _MAX_UNORDERED_PAIRS:
            raise MemoryError("structural target pair count exceeds 2016")
        if len(neighborhoods) > _MAX_TRAINING_ATOMS:
            raise MemoryError("structural target neighborhood count exceeds 64")
        _exact_nonnegative_int(self.positive_pair_count, "positive_pair_count")
        _exact_nonnegative_int(self.negative_pair_count, "negative_pair_count")
        if any(type(item) is not AtomPositionPairTarget for item in positive_pairs):
            raise TypeError("positive_pairs must contain exact AtomPositionPairTarget values")
        if any(type(item) is not AtomPositionPairTarget for item in negative_pairs):
            raise TypeError("negative_pairs must contain exact AtomPositionPairTarget values")
        if any(
            type(item) is not DirectCoBundleNeighborhood for item in neighborhoods
        ):
            raise TypeError(
                "neighborhoods must contain exact DirectCoBundleNeighborhood values"
            )
        positive_pairs = tuple(
            AtomPositionPairTarget(**_field_values(item)) for item in positive_pairs
        )
        negative_pairs = tuple(
            AtomPositionPairTarget(**_field_values(item)) for item in negative_pairs
        )
        neighborhoods = tuple(
            DirectCoBundleNeighborhood(**_field_values(item))
            for item in neighborhoods
        )
        object.__setattr__(self, "positive_pairs", positive_pairs)
        object.__setattr__(self, "negative_pairs", negative_pairs)
        object.__setattr__(self, "neighborhoods", neighborhoods)
        if self.positive_pair_count != len(positive_pairs):
            raise ValueError("positive_pair_count disagrees with positive_pairs")
        if self.negative_pair_count != len(negative_pairs):
            raise ValueError("negative_pair_count disagrees with negative_pairs")
        _bind_expected_sha256(
            self,
            "positive_pair_sequence_sha256",
            resident_values_sha256(_POSITIVE_PAIR_SEQUENCE_KIND, positive_pairs),
        )
        _bind_expected_sha256(
            self,
            "negative_pair_sequence_sha256",
            resident_values_sha256(_NEGATIVE_PAIR_SEQUENCE_KIND, negative_pairs),
        )

        positive_coordinates = self._validate_pair_class(
            positive_pairs,
            target=1,
            atom_count=atom_count,
            label="positive_pairs",
        )
        negative_coordinates = self._validate_pair_class(
            negative_pairs,
            target=0,
            atom_count=atom_count,
            label="negative_pairs",
        )
        if positive_coordinates & negative_coordinates:
            raise ValueError("positive and negative atom-position pairs must be disjoint")
        total_pair_count = atom_count * (atom_count - 1) // 2
        if total_pair_count > _MAX_UNORDERED_PAIRS:
            raise MemoryError("structural target pair count exceeds 2016")
        if len(positive_coordinates) + len(negative_coordinates) != total_pair_count:
            raise ValueError("positive and negative pairs must exhaust the unordered complement")
        self._validate_neighborhoods(
            neighborhoods,
            positive_coordinates,
            atom_count=atom_count,
        )
        _validate_seal_field(self.target_sha256, "target_sha256")
        _require_plain_identity_tree(self, "latent-router structural targets")
        self._seal()

    @staticmethod
    def _validate_pair_class(
        pairs: tuple[AtomPositionPairTarget, ...],
        *,
        target: int,
        atom_count: int,
        label: str,
    ) -> set[tuple[int, int]]:
        coordinates = tuple(
            (item.left_position, item.right_position) for item in pairs
        )
        if coordinates != tuple(sorted(coordinates)):
            raise ValueError(f"{label} must use lexicographic position order")
        if len(set(coordinates)) != len(coordinates):
            raise ValueError(f"{label} must not contain duplicate positions")
        for item in pairs:
            if item.left_position < 0 or item.left_position >= item.right_position:
                raise ValueError(f"{label} contains an invalid unordered pair")
            if item.direct_co_bundle_target != target:
                raise ValueError(f"{label} contains the wrong binary target")
            if item.left_position >= atom_count or item.right_position >= atom_count:
                raise ValueError(f"{label} contains an out-of-range position")
        return set(coordinates)

    @staticmethod
    def _validate_neighborhoods(
        neighborhoods: tuple[DirectCoBundleNeighborhood, ...],
        positive_coordinates: set[tuple[int, int]],
        *,
        atom_count: int,
    ) -> None:
        if len(neighborhoods) != atom_count:
            raise ValueError("there must be exactly one neighborhood per atom")
        if tuple(item.atom_position for item in neighborhoods) != tuple(
            range(atom_count)
        ):
            raise ValueError("neighborhoods must use ascending packet position")
        expected = [{position} for position in range(atom_count)]
        for left, right in positive_coordinates:
            expected[left].add(right)
            expected[right].add(left)
        for item in neighborhoods:
            if any(position >= atom_count for position in item.member_positions):
                raise ValueError("a neighborhood contains an out-of-range position")
            if item.member_positions != tuple(sorted(expected[item.atom_position])):
                raise ValueError(
                    "neighborhoods must equal self plus direct co-bundle neighbors"
                )


@dataclass(frozen=True, slots=True)
class LatentRouterStructuralTargetReceipt(SealedIdentity):
    """Text-free outer binding from exact packet/plan inputs to numerics."""

    _SEAL_FIELD = "target_receipt_sha256"
    _SEAL_MISMATCH = "structural target receipt SHA-256 does not match its contents"

    packet_receipt_sha256: str
    closure_plan_sha256: str
    fusion_caps_sha256: str
    ordered_atom_refs_sha256: str
    authoritative_hyperedges_sha256: str
    structural_targets: LatentRouterStructuralTargets
    target_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "packet_receipt_sha256",
            "closure_plan_sha256",
            "fusion_caps_sha256",
            "ordered_atom_refs_sha256",
            "authoritative_hyperedges_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _exact_sha256(getattr(self, name), name),
            )
        if type(self.structural_targets) is not LatentRouterStructuralTargets:
            raise TypeError(
                "structural_targets must be an exact LatentRouterStructuralTargets"
            )
        object.__setattr__(
            self,
            "structural_targets",
            LatentRouterStructuralTargets(**_field_values(self.structural_targets)),
        )
        _validate_seal_field(self.target_receipt_sha256, "target_receipt_sha256")
        _require_plain_identity_tree(self, "structural target receipt")
        self._seal()


def _validate_query_program(program: QueryProgram) -> None:
    for name in ("query", "intent", "ordering"):
        _exact_string(getattr(program, name), f"query_program.{name}")
    _exact_string_tuple(program.subject_terms, "query_program.subject_terms")
    _exact_optional(
        program.as_of_ordinal,
        _exact_nonnegative_int,
        "query_program.as_of_ordinal",
    )
    _exact_optional(
        program.cardinality,
        _exact_positive_int,
        "query_program.cardinality",
    )
    _exact_sha256(program.program_sha256, "query_program.program_sha256")
    obligations = _exact_tuple(program.obligations, "query_program.obligations")
    if any(type(item) is not EvidenceObligation for item in obligations):
        raise TypeError(
            "query_program.obligations must contain exact EvidenceObligation values"
        )
    for index, obligation in enumerate(obligations):
        prefix = f"query_program.obligations[{index}]"
        for name in ("obligation_id", "kind", "temporal_stance"):
            _exact_string(getattr(obligation, name), f"{prefix}.{name}")
        _exact_bool(obligation.required, f"{prefix}.required")
        _exact_float(obligation.weight, f"{prefix}.weight")
        _exact_positive_int(obligation.min_count, f"{prefix}.min_count")
        _exact_optional(
            obligation.max_count,
            _exact_nonnegative_int,
            f"{prefix}.max_count",
        )
        for name in (
            "unit_kinds",
            "relation_types",
            "subject_terms",
            "dependencies",
        ):
            _exact_string_tuple(getattr(obligation, name), f"{prefix}.{name}")


def _validate_policy(policy: ClosurePolicy) -> None:
    for name in (
        "max_hops",
        "max_units",
        "max_relations",
        "max_degree",
        "max_frontier",
        "max_bundles",
        "beam_width",
    ):
        _exact_positive_int(getattr(policy, name), f"closure_policy.{name}")
    _exact_nonnegative_int(
        policy.max_episode_neighbors,
        "closure_policy.max_episode_neighbors",
    )
    _exact_float(
        policy.min_relation_confidence,
        "closure_policy.min_relation_confidence",
    )


def _validate_snapshot(snapshot: DiscourseSnapshot) -> None:
    for name in (
        "max_turn_ordinal",
        "chunk_count",
        "graph_revision",
        "schema_version",
        "source_revision",
        "graph_content_revision",
    ):
        _exact_nonnegative_int(getattr(snapshot, name), f"snapshot.{name}")
    _exact_string_tuple(snapshot.artifact_ids, "snapshot.artifact_ids")
    for name in (
        "source_content_sha256",
        "graph_content_sha256",
        "snapshot_sha256",
    ):
        _exact_sha256(getattr(snapshot, name), f"snapshot.{name}")


def _validate_plan_owned_values(plan: ClosurePlan) -> None:
    _validate_query_program(plan.query_program)
    _validate_policy(plan.policy)
    _validate_snapshot(plan.snapshot)
    seeds = _exact_tuple(plan.seeds, "plan.seeds")
    if any(type(item) is not EpisodeSeed for item in seeds):
        raise TypeError("plan.seeds must contain exact EpisodeSeed values")
    for index, seed in enumerate(seeds):
        prefix = f"plan.seeds[{index}]"
        for name in ("episode_id", "anchor_chunk_id", "route"):
            _exact_string(getattr(seed, name), f"{prefix}.{name}")
        _exact_float(seed.score, f"{prefix}.score")
        _exact_string_tuple(seed.path, f"{prefix}.path")

    results = _exact_tuple(plan.obligation_results, "plan.obligation_results")
    if any(type(item) is not ObligationResult for item in results):
        raise TypeError(
            "plan.obligation_results must contain exact ObligationResult values"
        )
    for index, result in enumerate(results):
        prefix = f"plan.obligation_results[{index}]"
        for name in ("obligation_id", "status"):
            _exact_string(getattr(result, name), f"{prefix}.{name}")
        for name in ("unit_ids", "relation_ids", "bundle_ids"):
            _exact_string_tuple(getattr(result, name), f"{prefix}.{name}")
        _exact_optional(result.reason, _exact_string, f"{prefix}.reason")

    witnesses = _exact_tuple(plan.scope_witnesses, "plan.scope_witnesses")
    if any(type(item) is not ClosureScopeWitness for item in witnesses):
        raise TypeError(
            "plan.scope_witnesses must contain exact ClosureScopeWitness values"
        )
    for index, witness in enumerate(witnesses):
        prefix = f"plan.scope_witnesses[{index}]"
        _exact_string(witness.kind, f"{prefix}.kind")
        _exact_string(witness.subject_id, f"{prefix}.subject_id")
        _exact_optional(
            witness.requested_limit,
            _exact_nonnegative_int,
            f"{prefix}.requested_limit",
        )
        _exact_nonnegative_int(witness.returned_count, f"{prefix}.returned_count")
        _exact_bool(witness.exhaustive, f"{prefix}.exhaustive")
        _validate_frozen_json(witness.detail, f"{prefix}.detail")
        _exact_sha256(witness.witness_sha256, f"{prefix}.witness_sha256")

    for name in (
        "visited_episode_ids",
        "visited_unit_ids",
        "visited_relation_ids",
        "direct_chunk_ids",
    ):
        _exact_string_tuple(getattr(plan, name), f"plan.{name}")
    _exact_string(plan.stopping_reason, "plan.stopping_reason")
    if plan.stopping_reason not in _CLOSURE_STOP_REASONS:
        raise ValueError("plan.stopping_reason is not a supported closed value")
    _exact_bool(plan.complete_claimed, "plan.complete_claimed")
    _exact_optional(
        plan.expansion_receipt_sha256,
        _exact_sha256,
        "plan.expansion_receipt_sha256",
    )
    _exact_optional(plan.artifact_id, _exact_string, "plan.artifact_id")
    _exact_sha256(plan.plan_sha256, "plan.plan_sha256")


def _validate_closure_receipt(receipt: ClosureReceipt) -> None:
    for name in ("plan_sha256", "context_sha256", "receipt_sha256"):
        _exact_sha256(getattr(receipt, name), f"packet_receipt.{name}")
    for name in ("selected_atom_ids", "selected_bundle_ids"):
        _exact_string_tuple(getattr(receipt, name), f"packet_receipt.{name}")
    if type(receipt.dropped_bundle_reasons) is not _MAPPING_PROXY_TYPE:
        raise TypeError(
            "packet_receipt.dropped_bundle_reasons must be an exact frozen mapping"
        )
    for key, value in receipt.dropped_bundle_reasons.items():
        _exact_string(key, "packet_receipt.dropped_bundle_reasons key")
        _exact_string(value, f"packet_receipt.dropped_bundle_reasons.{key}")
    for name in (
        "context_token_proxy",
        "max_context_token_proxy",
        "retained_request_token_state_bytes",
        "responder_output_token_reserve",
    ):
        _exact_nonnegative_int(getattr(receipt, name), f"packet_receipt.{name}")
    for name in (
        "prompt_token_proxy",
        "max_prompt_token_proxy",
        "prompt_workspace_token_proxy",
    ):
        _exact_optional(
            getattr(receipt, name),
            _exact_nonnegative_int,
            f"packet_receipt.{name}",
        )
    for name in ("tokenizer_identity", "stopping_reason"):
        _exact_string(getattr(receipt, name), f"packet_receipt.{name}")
    if receipt.stopping_reason not in _CLOSURE_STOP_REASONS:
        raise ValueError(
            "packet_receipt.stopping_reason is not a supported closed value"
        )
    _exact_bool(receipt.complete_claimed, "packet_receipt.complete_claimed")
    for name in (
        "base_messages_sha256",
        "evidence_prefix_sha256",
        "evidence_suffix_sha256",
        "prompt_messages_sha256",
    ):
        _exact_optional(
            getattr(receipt, name),
            _exact_sha256,
            f"packet_receipt.{name}",
        )
    _exact_optional(
        receipt.evidence_message_role,
        _exact_string,
        "packet_receipt.evidence_message_role",
    )


def _reconstruct_caps(caps: FusionCaps) -> FusionCaps:
    reconstructed = FusionCaps(**_field_values(caps))
    if reconstructed != caps:
        raise ValueError("FusionCaps changes under authoritative reconstruction")
    return reconstructed


def _reconstruct_closure_receipt(receipt: ClosureReceipt) -> ClosureReceipt:
    reconstructed = ClosureReceipt(**_field_values(receipt))
    if reconstructed != receipt:
        raise ValueError(
            "closure receipt changes under authoritative reconstruction"
        )
    return reconstructed


def _reconstruct_plan(plan: ClosurePlan) -> ClosurePlan:
    obligation_values = tuple(
        EvidenceObligation(**_field_values(item))
        for item in plan.query_program.obligations
    )
    query_values = _field_values(plan.query_program)
    query_values["obligations"] = obligation_values
    query_program = QueryProgram(**query_values)
    policy = ClosurePolicy(**_field_values(plan.policy))
    snapshot = DiscourseSnapshot(**_field_values(plan.snapshot))
    seeds = tuple(EpisodeSeed(**_field_values(item)) for item in plan.seeds)
    atoms = []
    for atom in plan.atoms:
        atom_values = _field_values(atom)
        atom_values["span"] = EvidenceSpan(**_field_values(atom.span))
        atoms.append(EvidenceAtom(**atom_values))
    bundles = tuple(EvidenceBundle(**_field_values(item)) for item in plan.bundles)
    obligation_results = tuple(
        ObligationResult(**_field_values(item)) for item in plan.obligation_results
    )
    scope_witnesses = tuple(
        ClosureScopeWitness(**_field_values(item)) for item in plan.scope_witnesses
    )
    plan_values = _field_values(plan)
    plan_values.update(
        {
            "query_program": query_program,
            "policy": policy,
            "snapshot": snapshot,
            "seeds": seeds,
            "atoms": tuple(atoms),
            "bundles": bundles,
            "obligation_results": obligation_results,
            "scope_witnesses": scope_witnesses,
        }
    )
    reconstructed = ClosurePlan(**plan_values)
    if reconstructed != plan:
        raise ValueError("closure plan changes under authoritative reconstruction")
    return reconstructed


def _validate_exact_inputs(
    packet: object,
    plan: object,
    caps: object,
) -> tuple[EvidencePacket, ClosurePlan, FusionCaps]:
    if type(packet) is not EvidencePacket:
        raise TypeError("packet must be an exact EvidencePacket")
    if type(plan) is not ClosurePlan:
        raise TypeError("plan must be an exact ClosurePlan")
    if type(caps) is not FusionCaps:
        raise TypeError("caps must be an exact FusionCaps")
    if type(packet.receipt) is not ClosureReceipt:
        raise TypeError("packet receipt must be an exact ClosureReceipt")
    if type(packet.context) is not str:
        raise TypeError("packet context must be an exact string")
    if type(packet.atoms) is not tuple or type(packet.bundles) is not tuple:
        raise TypeError("packet atom and bundle collections must be exact tuples")
    if type(plan.atoms) is not tuple or type(plan.bundles) is not tuple:
        raise TypeError("plan atom and bundle collections must be exact tuples")
    if type(plan.query_program) is not QueryProgram:
        raise TypeError("closure query_program must be an exact QueryProgram")
    if type(plan.policy) is not ClosurePolicy:
        raise TypeError("closure policy must be an exact ClosurePolicy")
    if type(plan.snapshot) is not DiscourseSnapshot:
        raise TypeError("closure snapshot must be an exact DiscourseSnapshot")

    positive_cap_fields = (
        "max_atoms",
        "max_latents",
        "max_hidden_dim",
        "max_route_cells",
        "max_hyperedges",
        "max_groups",
        "max_group_atoms",
        "max_latent_memberships_per_atom",
    )
    for name in positive_cap_fields:
        _exact_positive_int(getattr(caps, name), f"FusionCaps.{name}")
    _exact_nonnegative_int(
        caps.max_topology_links,
        "FusionCaps.max_topology_links",
    )
    _exact_sha256(caps.caps_sha256, "caps_sha256")
    caps = _reconstruct_caps(caps)

    if not packet.atoms:
        raise ValueError("structural targets require at least one selected atom")
    atom_count = len(packet.atoms)
    if atom_count > caps.max_atoms:
        raise MemoryError("packet atom count exceeds FusionCaps.max_atoms")
    if atom_count > _MAX_TRAINING_ATOMS:
        raise MemoryError("structural target atom count exceeds 64")
    if atom_count * (atom_count - 1) // 2 > _MAX_UNORDERED_PAIRS:
        raise MemoryError("structural target pair count exceeds 2016")
    if len(packet.bundles) > caps.max_hyperedges:
        raise MemoryError("packet bundle count exceeds FusionCaps.max_hyperedges")
    receipt_atom_ids = packet.receipt.selected_atom_ids
    receipt_bundle_ids = packet.receipt.selected_bundle_ids
    if type(receipt_atom_ids) is not tuple:
        raise TypeError("packet receipt selected_atom_ids must be an exact tuple")
    if type(receipt_bundle_ids) is not tuple:
        raise TypeError("packet receipt selected_bundle_ids must be an exact tuple")
    if len(receipt_atom_ids) > caps.max_atoms or len(receipt_atom_ids) > (
        _MAX_TRAINING_ATOMS
    ):
        raise MemoryError("packet receipt selected atom count exceeds training caps")
    if len(receipt_bundle_ids) > caps.max_hyperedges:
        raise MemoryError("packet receipt selected bundle count exceeds training caps")
    if len(receipt_atom_ids) != atom_count:
        raise ValueError("packet receipt selected atom count disagrees with packet")
    if len(receipt_bundle_ids) != len(packet.bundles):
        raise ValueError("packet receipt selected bundle count disagrees with packet")
    if any(type(value) is not str for value in receipt_atom_ids):
        raise TypeError("packet receipt selected_atom_ids must contain exact strings")
    if any(type(value) is not str for value in receipt_bundle_ids):
        raise TypeError("packet receipt selected_bundle_ids must contain exact strings")

    if any(type(atom) is not EvidenceAtom for atom in packet.atoms):
        raise TypeError("packet atoms must be exact EvidenceAtom values")
    if any(type(bundle) is not EvidenceBundle for bundle in packet.bundles):
        raise TypeError("packet bundles must be exact EvidenceBundle values")
    raw_links = 0
    for bundle in packet.bundles:
        if type(bundle.atom_ids) is not tuple:
            raise TypeError("packet bundle atom_ids must be an exact tuple")
        raw_links += len(bundle.atom_ids) * (len(bundle.atom_ids) - 1) // 2
        if raw_links > caps.max_topology_links:
            raise MemoryError(
                "packet co-memberships exceed FusionCaps.max_topology_links"
            )

    if any(type(atom) is not EvidenceAtom for atom in plan.atoms):
        raise TypeError("plan atoms must be exact EvidenceAtom values")
    if any(type(bundle) is not EvidenceBundle for bundle in plan.bundles):
        raise TypeError("plan bundles must be exact EvidenceBundle values")
    _validate_plan_owned_values(plan)
    _validate_closure_receipt(packet.receipt)

    if any(
        type(atom.atom_id) is not str
        or type(atom.span) is not EvidenceSpan
        or type(atom.text) is not str
        or type(atom.label) is not str
        or (atom.role is not None and type(atom.role) is not str)
        or (atom.created_at is not None and type(atom.created_at) is not str)
        for atom in chain(packet.atoms, plan.atoms)
    ):
        raise TypeError("atom bodies must retain exact span and string fields")
    span_string_fields = (
        "chunk_id",
        "quote_sha256",
        "source_id",
        "turn_id",
        "role",
        "created_at",
    )
    span_integer_fields = (
        "start_char",
        "end_char",
        "ordinal",
        "turn_start_char",
    )
    if any(
        any(
            value is not None and type(value) is not str
            for value in (getattr(atom.span, name) for name in span_string_fields)
        )
        or any(type(getattr(atom.span, name)) is not int for name in span_integer_fields)
        for atom in chain(packet.atoms, plan.atoms)
    ):
        raise TypeError("atom spans must retain exact scalar field types")

    bundle_tuple_fields = ("atom_ids", "obligation_ids", "unit_ids", "relation_ids")
    if any(
        type(bundle.bundle_id) is not str
        or type(bundle.required) is not bool
        or type(bundle.utility) is not float
        or not math.isfinite(bundle.utility)
        or any(
            type(getattr(bundle, name)) is not tuple
            or any(type(value) is not str for value in getattr(bundle, name))
            for name in bundle_tuple_fields
        )
        for bundle in chain(packet.bundles, plan.bundles)
    ):
        raise TypeError("bundle bodies must retain exact scalar and tuple/string fields")
    plan = _reconstruct_plan(plan)
    _reconstruct_closure_receipt(packet.receipt)
    return packet, plan, caps


def _verify_packet_plan(
    packet: EvidencePacket,
    plan: ClosurePlan,
    caps: FusionCaps,
) -> None:
    """Compare all receipt joins and selected bodies independently by ID."""

    if quote_sha256(packet.context) != packet.receipt.context_sha256:
        raise ValueError("packet context no longer matches its closure receipt")
    if packet.receipt.plan_sha256 != plan.plan_sha256:
        raise ValueError("packet receipt does not bind the supplied closure plan")
    if packet.receipt.selected_atom_ids != tuple(atom.atom_id for atom in packet.atoms):
        raise ValueError("packet atom order disagrees with its receipt")
    if packet.receipt.selected_bundle_ids != tuple(
        bundle.bundle_id for bundle in packet.bundles
    ):
        raise ValueError("packet bundle order disagrees with its receipt")

    packet_atom_ids = tuple(atom.atom_id for atom in packet.atoms)
    if len(packet_atom_ids) != len(set(packet_atom_ids)):
        raise ValueError("packet atom IDs must be unique")
    plan_atoms = {atom.atom_id: atom for atom in plan.atoms}
    if len(plan_atoms) != len(plan.atoms):
        raise ValueError("closure plan atom IDs must be unique")
    for atom in packet.atoms:
        planned = plan_atoms.get(atom.atom_id)
        if planned is None or identity_sha256(
            atom.identity_payload()
        ) != identity_sha256(planned.identity_payload()):
            raise ValueError("packet atom does not exactly match the closure plan")
        if quote_sha256(atom.text) != atom.span.quote_sha256:
            raise ValueError("packet atom text does not match its source span")

    packet_bundle_ids = tuple(bundle.bundle_id for bundle in packet.bundles)
    if len(packet_bundle_ids) != len(set(packet_bundle_ids)):
        raise ValueError("packet bundle IDs must be unique")
    plan_bundles = {bundle.bundle_id: bundle for bundle in plan.bundles}
    if len(plan_bundles) != len(plan.bundles):
        raise ValueError("closure plan bundle IDs must be unique")
    selected_atom_ids = set(packet_atom_ids)
    for bundle in packet.bundles:
        planned = plan_bundles.get(bundle.bundle_id)
        if planned is None or identity_sha256(
            bundle.identity_payload()
        ) != identity_sha256(planned.identity_payload()):
            raise ValueError("packet bundle does not exactly match the closure plan")
        if len(bundle.atom_ids) != len(set(bundle.atom_ids)):
            raise ValueError("packet bundle atom IDs must be unique")
        if any(atom_id not in selected_atom_ids for atom_id in bundle.atom_ids):
            raise ValueError("packet bundle references an unselected atom")

    ranked_bundles = tuple(
        sorted(
            plan.bundles,
            key=lambda item: (
                not item.required,
                -item.utility,
                item.bundle_id,
            ),
        )
    )
    candidate_bundles = ranked_bundles[: plan.policy.max_bundles]
    candidate_ids = {bundle.bundle_id for bundle in candidate_bundles}
    selected_bundle_ids = set(packet_bundle_ids)
    if any(bundle_id not in candidate_ids for bundle_id in selected_bundle_ids):
        raise ValueError("packet selects a bundle outside the plan candidate cap")
    expected_bundle_order = tuple(
        bundle.bundle_id
        for bundle in candidate_bundles
        if bundle.bundle_id in selected_bundle_ids
    )
    if packet_bundle_ids != expected_bundle_order:
        raise ValueError("packet bundles are not in authoritative packing order")

    selected_bundle_atom_ids = {
        atom_id for bundle in packet.bundles for atom_id in bundle.atom_ids
    }
    if selected_bundle_atom_ids != selected_atom_ids:
        raise ValueError("packet atoms must equal the selected bundle atom union")
    expected_atom_order = tuple(
        atom.atom_id for atom in plan.atoms if atom.atom_id in selected_atom_ids
    )
    if packet_atom_ids != expected_atom_order:
        raise ValueError("packet atoms are not in authoritative plan order")

    dropped = dict(packet.receipt.dropped_bundle_reasons)
    dropped_ids = set(dropped)
    known_bundle_ids = set(plan_bundles)
    if selected_bundle_ids & dropped_ids:
        raise ValueError("a packet bundle cannot be both selected and dropped")
    if selected_bundle_ids | dropped_ids != known_bundle_ids:
        raise ValueError("selected and dropped bundles must partition the closure plan")
    for bundle_id, reason in dropped.items():
        if bundle_id not in candidate_ids:
            if reason != _CANDIDATE_DROP_REASON:
                raise ValueError("a cap-excluded bundle requires candidate_cap")
        elif reason not in _SELECTABLE_DROP_REASONS:
            raise ValueError("a candidate bundle has an unsupported drop reason")

    required_total = _required_proof_ids(plan.query_program)
    proved = _proved_obligation_ids(
        packet_bundle_ids,
        plan=plan,
        bundle_by_id=plan_bundles,
    )
    required_selected = required_total <= proved
    expected_complete = plan.complete_claimed and required_selected
    if not required_selected:
        expected_stopping_reason = "budget_impossible"
    elif plan.stopping_reason == "complete":
        expected_stopping_reason = "complete"
    else:
        expected_stopping_reason = plan.stopping_reason
    if packet.receipt.complete_claimed != expected_complete or (
        packet.receipt.stopping_reason != expected_stopping_reason
    ):
        raise ValueError("packet receipt outcome disagrees with selected plan proof")

    atom_count = len(packet.atoms)
    if atom_count > caps.max_atoms:
        raise MemoryError("packet atom count exceeds FusionCaps.max_atoms")
    if atom_count > _MAX_TRAINING_ATOMS:
        raise MemoryError("structural target atom count exceeds 64")
    pair_count = atom_count * (atom_count - 1) // 2
    if pair_count > _MAX_UNORDERED_PAIRS:
        raise MemoryError("structural target pair count exceeds 2016")
    if len(packet.bundles) > caps.max_hyperedges:
        raise MemoryError("packet bundle count exceeds FusionCaps.max_hyperedges")
    raw_links = sum(
        len(bundle.atom_ids) * (len(bundle.atom_ids) - 1) // 2
        for bundle in packet.bundles
    )
    if raw_links > caps.max_topology_links:
        raise MemoryError("packet co-memberships exceed FusionCaps.max_topology_links")


def _derive_structural_targets(packet: EvidencePacket) -> LatentRouterStructuralTargets:
    atom_positions = {atom.atom_id: index for index, atom in enumerate(packet.atoms)}
    positive_coordinates: set[tuple[int, int]] = set()
    for bundle in packet.bundles:
        positions = tuple(sorted(atom_positions[atom_id] for atom_id in bundle.atom_ids))
        for left_index, left_position in enumerate(positions):
            for right_position in positions[left_index + 1 :]:
                positive_coordinates.add((left_position, right_position))
    ordered_positives = tuple(sorted(positive_coordinates))
    positive_pairs = tuple(
        AtomPositionPairTarget(left, right, 1) for left, right in ordered_positives
    )
    negative_pairs = tuple(
        AtomPositionPairTarget(left, right, 0)
        for left in range(len(packet.atoms))
        for right in range(left + 1, len(packet.atoms))
        if (left, right) not in positive_coordinates
    )

    member_sets = [{position} for position in range(len(packet.atoms))]
    for left, right in ordered_positives:
        member_sets[left].add(right)
        member_sets[right].add(left)
    neighborhoods = tuple(
        DirectCoBundleNeighborhood(position, tuple(sorted(members)))
        for position, members in enumerate(member_sets)
    )
    return LatentRouterStructuralTargets(
        atom_count=len(packet.atoms),
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        neighborhoods=neighborhoods,
        positive_pair_count=len(positive_pairs),
        negative_pair_count=len(negative_pairs),
    )


def build_latent_router_structural_targets(
    packet: EvidencePacket,
    plan: ClosurePlan,
    *,
    caps: FusionCaps,
) -> LatentRouterStructuralTargetReceipt:
    """Build sealed direct co-bundle targets from one exact selected packet."""

    exact_packet, exact_plan, exact_caps = _validate_exact_inputs(packet, plan, caps)
    _verify_packet_plan(exact_packet, exact_plan, exact_caps)
    structural_targets = _derive_structural_targets(exact_packet)
    atom_refs = _canonical_atom_refs(exact_packet)
    hyperedges = _canonical_authoritative_hyperedges(exact_packet)
    return LatentRouterStructuralTargetReceipt(
        packet_receipt_sha256=exact_packet.receipt.receipt_sha256,
        closure_plan_sha256=exact_plan.plan_sha256,
        fusion_caps_sha256=exact_caps.caps_sha256,
        ordered_atom_refs_sha256=resident_values_sha256(
            _ATOM_REF_SEQUENCE_KIND,
            atom_refs,
        ),
        authoritative_hyperedges_sha256=resident_values_sha256(
            _HYPEREDGE_SEQUENCE_KIND,
            hyperedges,
        ),
        structural_targets=structural_targets,
    )


__all__ = [
    "AtomPositionPairTarget",
    "DirectCoBundleNeighborhood",
    "LatentRouterStructuralTargetReceipt",
    "LatentRouterStructuralTargets",
    "build_latent_router_structural_targets",
]
