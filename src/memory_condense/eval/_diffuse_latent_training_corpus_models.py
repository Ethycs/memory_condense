"""Frozen text-free models for the structural latent-training corpus."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Mapping

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.eval._diffuse_latent_training_corpus_codec import (
    DecodedLatentTrainingPayload,
    encode_latent_training_payload,
)
from memory_condense.eval.diffuse_longmemeval_route_v2 import (
    EPISODE_PRIMARY_ANALYSIS_QUERY_V2_FORMAT,
    EpisodePrimaryRouteReceiptV2,
)
from memory_condense.search.fusion.training_targets import (
    AtomPositionPairTarget,
    DirectCoBundleNeighborhood,
    LatentRouterStructuralTargetReceipt,
    LatentRouterStructuralTargets,
)


ANALYSIS_POPULATION_PROJECTION_FORMAT = (
    "memory-condense-latent-training-analysis-population-projection-v1"
)
LATENT_TRAINING_ROW_FORMAT = "memory-condense-latent-training-row-v1"
LATENT_TRAINING_PARTITION_FORMAT = "memory-condense-latent-training-partition-v1"
LATENT_TRAINING_CORPUS_FORMAT = "memory-condense-latent-training-corpus-v1"
LATENT_TRAINING_PUBLICATION_FORMAT = (
    "memory-condense-latent-training-structural-publication-v1"
)
ROOT_MANIFEST_NAME = "manifest.json"
MAX_PAYLOAD_SHARD_BYTES = 16 * 1024 * 1024
MAX_METADATA_FILE_BYTES = 4 * 1024 * 1024
_ATOM_REF_KIND = "latent_router_training_packet_atom_refs"
_HYPEREDGE_KIND = "latent_router_training_authoritative_hyperedges"


class LatentTrainingCorpusError(ValueError):
    """A structural corpus cannot support its bounded generic claim."""


def _sha(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise TypeError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{label} must be an exact non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{label} must be an exact integer >= {minimum}")
    return value


def _tuple(value: object, label: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact tuple")
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    rows = _tuple(value, label)
    if any(type(item) is not str or not item.strip() for item in rows):
        raise TypeError(f"{label} must contain exact non-empty strings")
    return rows


def _ids_sha256(values: tuple[str, ...]) -> str:
    return identity_sha256(list(values))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any, label: str) -> Any:
    kind = type(value)
    if value is None or kind in {str, bool, int}:
        return value
    if kind is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return value
    if kind in {list, tuple}:
        return tuple(
            _freeze(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{label} contains a non-string key")
        return MappingProxyType(
            {key: _freeze(item, f"{label}.{key}") for key, item in value.items()}
        )
    raise TypeError(f"{label} contains unsupported type {kind.__name__}")


def _mapping(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be an exact JSON object")
    if set(value) != keys:
        raise ValueError(f"{label} has a non-closed schema")
    return value


def _list(value: object, label: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{label} must be an exact JSON array")
    return value


def _safe_relative(value: object, label: str) -> str:
    text = _text(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or "\\" in text or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{label} must be a contained POSIX relative path")
    return text


@dataclass(frozen=True, slots=True)
class LatentTrainingPopulationExpectation:
    dataset_sha256: str
    split_manifest_sha256: str
    treatment_file_sha256: str
    sanitized_projection_sha256: str
    fit_count: int
    fit_ordered_question_ids_sha256: str
    validation_count: int
    validation_ordered_question_ids_sha256: str
    analysis_ordered_question_ids_sha256: str
    excluded_confirmation_count: int
    excluded_confirmation_ordered_question_ids_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "dataset_sha256", "split_manifest_sha256", "treatment_file_sha256",
            "sanitized_projection_sha256", "fit_ordered_question_ids_sha256",
            "validation_ordered_question_ids_sha256",
            "analysis_ordered_question_ids_sha256",
            "excluded_confirmation_ordered_question_ids_sha256",
        ):
            _sha(getattr(self, name), name)
        for name in ("fit_count", "validation_count", "excluded_confirmation_count"):
            _integer(getattr(self, name), name, minimum=1)


LOCKED_LATENT_TRAINING_POPULATION = LatentTrainingPopulationExpectation(
    dataset_sha256="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    split_manifest_sha256="8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4",
    treatment_file_sha256="b4d1d34538fdabbd6127c339bff8167293d290eb732afc18a5d8963d12b15001",
    sanitized_projection_sha256="58a1982122d259e046ac5268de8fc3c2857a63d24c859e3bc13e4e6b9aa52ad8",
    fit_count=200,
    fit_ordered_question_ids_sha256="533aa545efb8032f7b181f39264c6d10a49471bd460414f420e37dc840a19c55",
    validation_count=100,
    validation_ordered_question_ids_sha256="7a67aa6f43ffb94d487fb9184f871735bd9edac1974a3154898846d1140c83a1",
    analysis_ordered_question_ids_sha256="cf5e8648b71634e4e22be872881766e37e0dc24a2931d0c63365e075b2742046",
    excluded_confirmation_count=200,
    excluded_confirmation_ordered_question_ids_sha256="6270b044792dbda79cd79a104ab6a519b2f81980c47522c19a196583d8c0d102",
)


@dataclass(frozen=True, slots=True)
class AnalysisPopulationProjection(SealedIdentity):
    _SEAL_FIELD = "projection_sha256"
    _SEAL_MISMATCH = "analysis population projection does not match"

    treatment_file_sha256: str
    sanitized_projection_sha256: str
    dataset_sha256: str
    split_manifest_sha256: str
    ordered_question_ids: tuple[str, ...]
    excluded_confirmation_count: int
    excluded_confirmation_ordered_question_ids_sha256: str
    source_treatment_exact_type_verified: bool = False
    format: str = ANALYSIS_POPULATION_PROJECTION_FORMAT
    projection_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.format) is not str or self.format != ANALYSIS_POPULATION_PROJECTION_FORMAT:
            raise ValueError("unsupported analysis population projection format")
        for name in (
            "treatment_file_sha256", "sanitized_projection_sha256",
            "dataset_sha256", "split_manifest_sha256",
            "excluded_confirmation_ordered_question_ids_sha256",
        ):
            _sha(getattr(self, name), name)
        ids = _strings(self.ordered_question_ids, "ordered_question_ids")
        if len(set(ids)) != len(ids):
            raise ValueError("analysis population IDs must be unique")
        _integer(self.excluded_confirmation_count, "excluded_confirmation_count", minimum=1)
        if self.source_treatment_exact_type_verified is not False:
            raise ValueError("generic src projection cannot verify the tools treatment type")
        self._seal()

    @property
    def ordered_question_ids_sha256(self) -> str:
        return _ids_sha256(self.ordered_question_ids)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        body: dict[str, object] = {
            "format": self.format,
            "treatment_file_sha256": self.treatment_file_sha256,
            "sanitized_projection_sha256": self.sanitized_projection_sha256,
            "dataset_sha256": self.dataset_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "ordered_question_count": len(self.ordered_question_ids),
            "ordered_question_ids_sha256": self.ordered_question_ids_sha256,
            "excluded_confirmation_count": self.excluded_confirmation_count,
            "excluded_confirmation_ordered_question_ids_sha256": self.excluded_confirmation_ordered_question_ids_sha256,
            "source_treatment_exact_type_verified": (
                self.source_treatment_exact_type_verified
            ),
        }
        if include_receipt:
            body["projection_sha256"] = self.projection_sha256
        return body


@dataclass(frozen=True, slots=True)
class AnalysisPopulationRow:
    ordinal: int
    partition: Literal["fit", "validation"]
    partition_ordinal: int
    question_id: str

    def __post_init__(self) -> None:
        _integer(self.ordinal, "population row ordinal")
        _integer(self.partition_ordinal, "population row partition ordinal")
        if type(self.partition) is not str or self.partition not in {"fit", "validation"}:
            raise TypeError("population row partition has the wrong exact literal")
        _text(self.question_id, "population row question_id")


@dataclass(frozen=True, slots=True)
class LatentTrainingRouteEvidence(SealedIdentity):
    _SEAL_FIELD = "evidence_sha256"
    _SEAL_MISMATCH = "latent training route evidence does not match"

    analysis_arm_v2_body: Mapping[str, Any]
    analysis_base_arm_body: Mapping[str, Any]
    episode_policy_body: Mapping[str, Any]
    closure_policy_body: Mapping[str, Any]
    compilation_receipt_body: Mapping[str, Any]
    compilation_snapshot_body: Mapping[str, Any]
    representative_policy_body: Mapping[str, Any]
    anchor_projection_body: Mapping[str, Any]
    inner_analysis_query_receipt_body: Mapping[str, Any]
    inner_diffuse_query_receipt_body: Mapping[str, Any]
    legacy_input_receipt_body: Mapping[str, Any]
    source_scope_body: Mapping[str, Any]
    direct_expansion_body: Mapping[str, Any]
    representative_expansion_body: Mapping[str, Any]
    route_receipt: EpisodePrimaryRouteReceiptV2
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "analysis_arm_v2_body", "analysis_base_arm_body",
            "episode_policy_body", "closure_policy_body",
            "compilation_receipt_body", "compilation_snapshot_body",
            "representative_policy_body",
            "anchor_projection_body",
            "inner_analysis_query_receipt_body",
            "inner_diffuse_query_receipt_body", "legacy_input_receipt_body",
            "source_scope_body", "direct_expansion_body",
            "representative_expansion_body",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a JSON mapping")
            object.__setattr__(self, name, _freeze(value, name))
        if type(self.route_receipt) is not EpisodePrimaryRouteReceiptV2:
            raise TypeError("route_receipt must be exact EpisodePrimaryRouteReceiptV2")
        self.route_receipt._seal()
        self._seal()

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        body: dict[str, object] = {
            name: _plain(getattr(self, name))
            for name in (
                "analysis_arm_v2_body", "analysis_base_arm_body",
                "episode_policy_body", "closure_policy_body",
                "compilation_receipt_body", "compilation_snapshot_body",
                "representative_policy_body",
                "anchor_projection_body",
                "inner_analysis_query_receipt_body",
                "inner_diffuse_query_receipt_body", "legacy_input_receipt_body",
                "source_scope_body", "direct_expansion_body",
                "representative_expansion_body",
            )
        }
        body["route_receipt"] = self.route_receipt.identity_payload()
        if include_receipt:
            body["evidence_sha256"] = self.evidence_sha256
        return body


def _target_body(value: LatentRouterStructuralTargetReceipt) -> dict[str, Any]:
    targets = value.structural_targets
    return {
        "packet_receipt_sha256": value.packet_receipt_sha256,
        "closure_plan_sha256": value.closure_plan_sha256,
        "fusion_caps_sha256": value.fusion_caps_sha256,
        "ordered_atom_refs_sha256": value.ordered_atom_refs_sha256,
        "authoritative_hyperedges_sha256": value.authoritative_hyperedges_sha256,
        "structural_targets": {
            "atom_count": targets.atom_count,
            "positive_pairs": [item.identity_payload() for item in targets.positive_pairs],
            "negative_pairs": [item.identity_payload() for item in targets.negative_pairs],
            "neighborhoods": [item.identity_payload() for item in targets.neighborhoods],
            "positive_pair_count": targets.positive_pair_count,
            "negative_pair_count": targets.negative_pair_count,
            "positive_pair_sequence_sha256": targets.positive_pair_sequence_sha256,
            "negative_pair_sequence_sha256": targets.negative_pair_sequence_sha256,
            "target_sha256": targets.target_sha256,
        },
        "target_receipt_sha256": value.target_receipt_sha256,
    }


def _reconstruct_target(value: LatentRouterStructuralTargetReceipt) -> LatentRouterStructuralTargetReceipt:
    targets = value.structural_targets
    if type(targets) is not LatentRouterStructuralTargets:
        raise TypeError("structural targets have the wrong exact type")
    positive_rows = _tuple(targets.positive_pairs, "positive_pairs")
    negative_rows = _tuple(targets.negative_pairs, "negative_pairs")
    neighborhood_rows = _tuple(targets.neighborhoods, "neighborhoods")
    if any(type(item) is not AtomPositionPairTarget for item in (*positive_rows, *negative_rows)):
        raise TypeError("pair targets have the wrong exact type")
    if any(type(item) is not DirectCoBundleNeighborhood for item in neighborhood_rows):
        raise TypeError("neighborhood targets have the wrong exact type")
    def pair(item: AtomPositionPairTarget) -> AtomPositionPairTarget:
        return AtomPositionPairTarget(
            item.left_position, item.right_position, item.direct_co_bundle_target,
            item.pair_sha256,
        )
    rebuilt_targets = LatentRouterStructuralTargets(
        atom_count=targets.atom_count,
        positive_pairs=tuple(pair(item) for item in positive_rows),
        negative_pairs=tuple(pair(item) for item in negative_rows),
        neighborhoods=tuple(
            DirectCoBundleNeighborhood(
                item.atom_position, item.member_positions, item.neighborhood_sha256
            )
            for item in neighborhood_rows
        ),
        positive_pair_count=targets.positive_pair_count,
        negative_pair_count=targets.negative_pair_count,
        positive_pair_sequence_sha256=targets.positive_pair_sequence_sha256,
        negative_pair_sequence_sha256=targets.negative_pair_sequence_sha256,
        target_sha256=targets.target_sha256,
    )
    rebuilt = LatentRouterStructuralTargetReceipt(
        packet_receipt_sha256=value.packet_receipt_sha256,
        closure_plan_sha256=value.closure_plan_sha256,
        fusion_caps_sha256=value.fusion_caps_sha256,
        ordered_atom_refs_sha256=value.ordered_atom_refs_sha256,
        authoritative_hyperedges_sha256=value.authoritative_hyperedges_sha256,
        structural_targets=rebuilt_targets,
        target_receipt_sha256=value.target_receipt_sha256,
    )
    if _target_body(rebuilt) != _target_body(value):
        raise ValueError("structural target changed during reconstruction")
    return rebuilt


@dataclass(frozen=True, slots=True)
class LatentTrainingCorpusRowManifest(SealedIdentity):
    _SEAL_FIELD = "row_sha256"
    _SEAL_MISMATCH = "latent training row manifest does not match"

    ordinal: int
    partition: Literal["fit", "validation"]
    partition_ordinal: int
    question_id: str
    question_id_sha256: str
    question_probe_sha256: str
    retrieval_query_sha256: str
    prompt_question_sha256: str
    route_record_sha256: str
    route_evidence: LatentTrainingRouteEvidence
    payload_relative_path: str
    payload_sha256: str
    payload_bytes: int
    packet_receipt_sha256: str
    closure_plan_sha256: str
    ordered_atom_refs_sha256: str
    authoritative_hyperedges_sha256: str
    structural_target: LatentRouterStructuralTargetReceipt
    format: str = LATENT_TRAINING_ROW_FORMAT
    row_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.format) is not str or self.format != LATENT_TRAINING_ROW_FORMAT:
            raise ValueError("unsupported latent training row format")
        _integer(self.ordinal, "ordinal")
        _integer(self.partition_ordinal, "partition_ordinal")
        if type(self.partition) is not str or self.partition not in {"fit", "validation"}:
            raise TypeError("row partition has the wrong exact literal")
        _text(self.question_id, "question_id")
        for name in (
            "question_id_sha256", "question_probe_sha256", "retrieval_query_sha256",
            "prompt_question_sha256", "route_record_sha256", "payload_sha256",
            "packet_receipt_sha256", "closure_plan_sha256",
            "ordered_atom_refs_sha256", "authoritative_hyperedges_sha256",
        ):
            _sha(getattr(self, name), name)
        _integer(self.payload_bytes, "payload_bytes", minimum=1)
        _safe_relative(self.payload_relative_path, "payload_relative_path")
        if self.question_id_sha256 != identity_sha256({"question_id": self.question_id}):
            raise ValueError("question ID digest differs from its opaque ID")
        if self.payload_relative_path != f"payloads/{self.payload_sha256}.json":
            raise ValueError("payload path must be content addressed")
        if type(self.route_evidence) is not LatentTrainingRouteEvidence:
            raise TypeError("route_evidence has the wrong exact type")
        self.route_evidence.route_receipt._seal()
        self.route_evidence._seal()
        if type(self.structural_target) is not LatentRouterStructuralTargetReceipt:
            raise TypeError("structural_target has the wrong exact type")
        _reconstruct_target(self.structural_target)
        route = self.route_evidence.route_receipt
        if (
            self.structural_target.packet_receipt_sha256 != self.packet_receipt_sha256
            or self.structural_target.closure_plan_sha256 != self.closure_plan_sha256
            or self.structural_target.ordered_atom_refs_sha256 != self.ordered_atom_refs_sha256
            or self.structural_target.authoritative_hyperedges_sha256 != self.authoritative_hyperedges_sha256
            or route.packet_receipt_sha256 != self.packet_receipt_sha256
            or route.closure_plan_sha256 != self.closure_plan_sha256
            or route.retrieval_query_sha256 != self.retrieval_query_sha256
        ):
            raise ValueError("row packet, plan, route, or target joins disagree")
        expected_record = identity_sha256(
            {
                "format": EPISODE_PRIMARY_ANALYSIS_QUERY_V2_FORMAT,
                "analysis_arm_v2_sha256": route.analysis_arm_v2_sha256,
                "inner_analysis_query_receipt_sha256": route.inner_analysis_query_receipt_sha256,
                "inner_diffuse_query_receipt_sha256": route.inner_diffuse_query_receipt_sha256,
                "route_receipt_sha256": route.receipt_sha256,
            }
        )
        if self.route_record_sha256 != expected_record:
            raise ValueError("route record digest cannot be reconstructed")
        self._seal()

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        body: dict[str, object] = {
            "format": self.format, "ordinal": self.ordinal,
            "partition": self.partition, "partition_ordinal": self.partition_ordinal,
            "question_id": self.question_id, "question_id_sha256": self.question_id_sha256,
            "question_probe_sha256": self.question_probe_sha256,
            "retrieval_query_sha256": self.retrieval_query_sha256,
            "prompt_question_sha256": self.prompt_question_sha256,
            "route_record_sha256": self.route_record_sha256,
            "route_evidence": self.route_evidence.identity_payload(),
            "payload_relative_path": self.payload_relative_path,
            "payload_sha256": self.payload_sha256, "payload_bytes": self.payload_bytes,
            "packet_receipt_sha256": self.packet_receipt_sha256,
            "closure_plan_sha256": self.closure_plan_sha256,
            "ordered_atom_refs_sha256": self.ordered_atom_refs_sha256,
            "authoritative_hyperedges_sha256": self.authoritative_hyperedges_sha256,
            "structural_target": _target_body(self.structural_target),
        }
        if include_receipt:
            body["row_sha256"] = self.row_sha256
        return body


@dataclass(frozen=True, slots=True)
class LatentTrainingCorpusPartitionManifest(SealedIdentity):
    _SEAL_FIELD = "partition_sha256"
    _SEAL_MISMATCH = "latent training partition manifest does not match"

    partition: Literal["fit", "validation"]
    start_ordinal: int
    row_count: int
    ordered_question_ids_sha256: str
    row_relative_paths: tuple[str, ...]
    row_sha256s: tuple[str, ...]
    production_authorized: bool = False
    d1_eligible: bool = False
    format: str = LATENT_TRAINING_PARTITION_FORMAT
    partition_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.format) is not str or self.format != LATENT_TRAINING_PARTITION_FORMAT:
            raise ValueError("unsupported latent training partition format")
        if type(self.partition) is not str or self.partition not in {"fit", "validation"}:
            raise TypeError("partition has the wrong exact literal")
        _integer(self.start_ordinal, "start_ordinal")
        _integer(self.row_count, "row_count", minimum=1)
        _sha(self.ordered_question_ids_sha256, "ordered_question_ids_sha256")
        paths = _strings(self.row_relative_paths, "row_relative_paths")
        hashes = _strings(self.row_sha256s, "row_sha256s")
        if len(paths) != self.row_count or len(hashes) != self.row_count:
            raise ValueError("partition row inventory count changed")
        for path in paths:
            _safe_relative(path, "row_relative_path")
        for digest in hashes:
            _sha(digest, "row_sha256")
        if self.production_authorized is not False or self.d1_eligible is not False:
            raise ValueError("generic structural partitions cannot authorize D1")
        self._seal()

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        body: dict[str, object] = {
            "format": self.format, "partition": self.partition,
            "start_ordinal": self.start_ordinal, "row_count": self.row_count,
            "ordered_question_ids_sha256": self.ordered_question_ids_sha256,
            "row_relative_paths": list(self.row_relative_paths),
            "row_sha256s": list(self.row_sha256s),
            "production_authorized": self.production_authorized,
            "d1_eligible": self.d1_eligible,
        }
        if include_receipt:
            body["partition_sha256"] = self.partition_sha256
        return body


@dataclass(frozen=True, slots=True)
class LatentTrainingFileIdentity:
    relative_path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        _safe_relative(self.relative_path, "inventory relative_path")
        _sha(self.sha256, "inventory sha256")
        _integer(self.bytes, "inventory bytes", minimum=1)

    def identity_payload(self) -> dict[str, object]:
        return {"relative_path": self.relative_path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class LatentTrainingCorpusManifest(SealedIdentity):
    _SEAL_FIELD = "corpus_sha256"
    _SEAL_MISMATCH = "latent training corpus manifest does not match"

    population_projection_sha256: str
    implementation_sha256: str
    treatment_file_sha256: str
    sanitized_projection_sha256: str
    dataset_sha256: str
    split_manifest_sha256: str
    analysis_ordered_question_ids_sha256: str
    fit_partition_sha256: str
    validation_partition_sha256: str
    excluded_confirmation_count: int
    excluded_confirmation_ordered_question_ids_sha256: str
    inventory: tuple[LatentTrainingFileIdentity, ...]
    inventory_sha256: str
    population_status: Literal["locked_projection", "synthetic_projection"]
    episodic_route: Literal["episode_primary"] = "episode_primary"
    closure_routing_scope: Literal["seeded_graph"] = "seeded_graph"
    scorer_labels_present: bool = False
    evaluator_label_schema_present: bool = False
    tensor_or_embedding_payload_present: bool = False
    source_treatment_exact_type_verified: bool = False
    production_authorized: bool = False
    d1_eligible: bool = False
    format: str = LATENT_TRAINING_CORPUS_FORMAT
    corpus_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.format) is not str or self.format != LATENT_TRAINING_CORPUS_FORMAT:
            raise ValueError("unsupported latent training corpus format")
        for name in (
            "population_projection_sha256", "implementation_sha256",
            "treatment_file_sha256", "sanitized_projection_sha256",
            "dataset_sha256", "split_manifest_sha256",
            "analysis_ordered_question_ids_sha256", "fit_partition_sha256",
            "validation_partition_sha256",
            "excluded_confirmation_ordered_question_ids_sha256", "inventory_sha256",
        ):
            _sha(getattr(self, name), name)
        _integer(self.excluded_confirmation_count, "excluded_confirmation_count", minimum=1)
        inventory = _tuple(self.inventory, "inventory")
        if not inventory or any(type(item) is not LatentTrainingFileIdentity for item in inventory):
            raise TypeError("inventory must contain exact file identities")
        inventory = tuple(
            LatentTrainingFileIdentity(item.relative_path, item.sha256, item.bytes)
            for item in inventory
        )
        object.__setattr__(self, "inventory", inventory)
        paths = tuple(item.relative_path for item in inventory)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("inventory must use unique sorted relative paths")
        if self.inventory_sha256 != identity_sha256([item.identity_payload() for item in inventory]):
            raise ValueError("inventory SHA-256 changed")
        if type(self.population_status) is not str or self.population_status not in {
            "locked_projection", "synthetic_projection"
        }:
            raise TypeError("population status has the wrong exact literal")
        if type(self.episodic_route) is not str or self.episodic_route != "episode_primary":
            raise TypeError("corpus requires the exact episode_primary route")
        if type(self.closure_routing_scope) is not str or self.closure_routing_scope != "seeded_graph":
            raise TypeError("corpus requires the exact seeded_graph closure scope")
        for name in (
            "scorer_labels_present", "evaluator_label_schema_present",
            "tensor_or_embedding_payload_present", "source_treatment_exact_type_verified",
            "production_authorized", "d1_eligible",
        ):
            if getattr(self, name) is not False:
                raise ValueError("generic corpus cannot mint scientific authority")
        self._seal()

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        body: dict[str, object] = {
            "format": self.format,
            "population_projection_sha256": self.population_projection_sha256,
            "implementation_sha256": self.implementation_sha256,
            "treatment_file_sha256": self.treatment_file_sha256,
            "sanitized_projection_sha256": self.sanitized_projection_sha256,
            "dataset_sha256": self.dataset_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "analysis_ordered_question_ids_sha256": self.analysis_ordered_question_ids_sha256,
            "fit_partition_sha256": self.fit_partition_sha256,
            "validation_partition_sha256": self.validation_partition_sha256,
            "excluded_confirmation_count": self.excluded_confirmation_count,
            "excluded_confirmation_ordered_question_ids_sha256": self.excluded_confirmation_ordered_question_ids_sha256,
            "inventory": [item.identity_payload() for item in self.inventory],
            "inventory_sha256": self.inventory_sha256,
            "population_status": self.population_status,
            "episodic_route": self.episodic_route,
            "closure_routing_scope": self.closure_routing_scope,
            "scorer_labels_present": self.scorer_labels_present,
            "evaluator_label_schema_present": self.evaluator_label_schema_present,
            "tensor_or_embedding_payload_present": (
                self.tensor_or_embedding_payload_present
            ),
            "source_treatment_exact_type_verified": (
                self.source_treatment_exact_type_verified
            ),
            "production_authorized": self.production_authorized,
            "d1_eligible": self.d1_eligible,
        }
        if include_receipt:
            body["corpus_sha256"] = self.corpus_sha256
        return body


@dataclass(frozen=True, slots=True)
class LatentTrainingCorpusPublicationReceipt(SealedIdentity):
    _SEAL_MISMATCH = "latent training publication receipt does not match"

    corpus_sha256: str
    implementation_sha256: str
    root_manifest_sha256: str
    root_manifest_bytes: int
    inventory_sha256: str
    production_authorized: bool = False
    d1_eligible: bool = False
    qwen_execution_attested: bool = False
    format: str = LATENT_TRAINING_PUBLICATION_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.format) is not str or self.format != LATENT_TRAINING_PUBLICATION_FORMAT:
            raise ValueError("unsupported structural publication format")
        for name in (
            "corpus_sha256", "implementation_sha256",
            "root_manifest_sha256", "inventory_sha256",
        ):
            _sha(getattr(self, name), name)
        _integer(self.root_manifest_bytes, "root_manifest_bytes", minimum=1)
        if any(getattr(self, name) is not False for name in (
            "production_authorized", "d1_eligible", "qwen_execution_attested"
        )):
            raise ValueError("provider-free publication cannot attest D1 execution")
        self._seal()


@dataclass(frozen=True, slots=True)
class DecodedLatentTrainingCorpusRow:
    manifest: LatentTrainingCorpusRowManifest
    payload: DecodedLatentTrainingPayload

    def __post_init__(self) -> None:
        if type(self.manifest) is not LatentTrainingCorpusRowManifest:
            raise TypeError("decoded row manifest has the wrong exact type")
        if type(self.payload) is not DecodedLatentTrainingPayload:
            raise TypeError("decoded row payload has the wrong exact type")
        self.manifest._seal()


def _validate_view(manifest: Any, partition: Any, rows: Any, role: str, authority: Any, eligible: Any) -> None:
    from memory_condense.eval._diffuse_latent_training_corpus_route import (
        live_route_v2_implementation_sha256,
        validate_persisted_route,
    )
    from memory_condense.eval.diffuse_latent_training_corpus import (
        latent_training_corpus_implementation_sha256,
    )

    if type(manifest) is not LatentTrainingCorpusManifest:
        raise TypeError("verified manifest has the wrong exact type")
    if type(partition) is not LatentTrainingCorpusPartitionManifest:
        raise TypeError("verified partition has the wrong exact type")
    manifest._seal()
    partition._seal()
    if manifest.implementation_sha256 != latent_training_corpus_implementation_sha256():
        raise ValueError("verified corpus implementation identity is no longer current")
    if any(
        getattr(manifest, name) is not False
        for name in (
            "scorer_labels_present",
            "evaluator_label_schema_present",
            "tensor_or_embedding_payload_present",
            "source_treatment_exact_type_verified",
            "production_authorized",
            "d1_eligible",
        )
    ) or partition.production_authorized is not False or (
        partition.d1_eligible is not False
    ):
        raise ValueError("verified corpus contains an authority or label flag")
    values = _tuple(rows, "verified rows")
    route_implementation = live_route_v2_implementation_sha256()
    if partition.partition != role or len(values) != partition.row_count or any(
        type(item) is not DecodedLatentTrainingCorpusRow for item in values
    ):
        raise TypeError("verified partition rows have the wrong role/count/type")
    for index, item in enumerate(values):
        item.manifest._seal()
        if item.manifest.partition != role or item.manifest.partition_ordinal != index or (
            item.manifest.row_sha256 != partition.row_sha256s[index]
        ):
            raise ValueError("verified row order differs from its partition")
        payload = encode_latent_training_payload(
            item.payload.retrieval_query,
            item.payload.plan,
            item.payload.packet,
            question_id=item.payload.question_id,
            prompt_question=item.payload.prompt_question,
        )
        if (
            len(payload) != item.manifest.payload_bytes
            or hashlib.sha256(payload).hexdigest() != item.manifest.payload_sha256
        ):
            raise ValueError("verified row payload is no longer canonical/current")
        validate_persisted_route(
            item.manifest,
            item.payload,
            expected_route_implementation_sha256=route_implementation,
        )
    expected = manifest.fit_partition_sha256 if role == "fit" else manifest.validation_partition_sha256
    if partition.partition_sha256 != expected:
        raise ValueError("verified partition differs from its root manifest")
    if authority is not False or eligible is not False:
        raise ValueError("generic verified partition cannot authorize D1")


@dataclass(frozen=True, slots=True)
class VerifiedLatentTrainingFitCorpus:
    manifest: LatentTrainingCorpusManifest
    partition: LatentTrainingCorpusPartitionManifest
    rows: tuple[DecodedLatentTrainingCorpusRow, ...]
    production_authorized: bool = False
    d1_eligible: bool = False

    def __post_init__(self) -> None:
        _validate_view(self.manifest, self.partition, self.rows, "fit", self.production_authorized, self.d1_eligible)


@dataclass(frozen=True, slots=True)
class VerifiedLatentTrainingValidationCorpus:
    manifest: LatentTrainingCorpusManifest
    partition: LatentTrainingCorpusPartitionManifest
    rows: tuple[DecodedLatentTrainingCorpusRow, ...]
    production_authorized: bool = False
    d1_eligible: bool = False

    def __post_init__(self) -> None:
        _validate_view(self.manifest, self.partition, self.rows, "validation", self.production_authorized, self.d1_eligible)


@dataclass(frozen=True, slots=True)
class VerifiedLatentTrainingFullCorpus:
    manifest: LatentTrainingCorpusManifest
    fit: VerifiedLatentTrainingFitCorpus
    validation: VerifiedLatentTrainingValidationCorpus
    production_authorized: bool = False
    d1_eligible: bool = False

    def __post_init__(self) -> None:
        if type(self.manifest) is not LatentTrainingCorpusManifest:
            raise TypeError("verified full manifest has the wrong exact type")
        if type(self.fit) is not VerifiedLatentTrainingFitCorpus or type(
            self.validation
        ) is not VerifiedLatentTrainingValidationCorpus:
            raise TypeError("verified full partition views have the wrong exact types")
        self.manifest._seal()
        if any(
            getattr(self.manifest, name) is not False
            for name in (
                "scorer_labels_present",
                "evaluator_label_schema_present",
                "tensor_or_embedding_payload_present",
                "source_treatment_exact_type_verified",
                "production_authorized",
                "d1_eligible",
            )
        ):
            raise ValueError("verified full corpus contains an authority or label flag")
        self.fit.__post_init__()
        self.validation.__post_init__()
        if self.fit.manifest is not self.manifest or self.validation.manifest is not self.manifest:
            raise ValueError("verified partition views must share the exact root manifest")
        if self.production_authorized is not False or self.d1_eligible is not False:
            raise ValueError("generic verified corpus cannot authorize D1")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json(value).encode("utf-8")


def _file_identity(relative_path: str, payload: bytes) -> LatentTrainingFileIdentity:
    return LatentTrainingFileIdentity(
        relative_path=relative_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )


__all__ = [name for name in globals() if not name.startswith("_")]
