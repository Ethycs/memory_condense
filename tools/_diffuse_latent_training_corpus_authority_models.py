"""Closed, non-authoritative candidate models for the route-v2 corpus.

No genuine output identity is independently pinned yet.  These objects retain
integrity and audit coordinates only; every scientific-authority, execution,
fit, validation, update, and selection claim is forced to literal false.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.eval._diffuse_latent_training_corpus_models import (
    DecodedLatentTrainingCorpusRow,
    LatentTrainingCorpusPartitionManifest,
    LatentTrainingFileIdentity,
    VerifiedLatentTrainingFullCorpus,
    _integer,
    _safe_relative,
    _sha,
    _text,
)


PRODUCTION_EXTERNAL_LOCK_FORMAT = (
    "memory-condense-latent-training-production-external-lock-v1"
)
DECLARED_EXECUTION_FORMAT = (
    "memory-condense-latent-training-declared-execution-candidate-v1"
)
PRODUCTION_CANDIDATE_FORMAT = (
    "memory-condense-latent-training-production-candidate-v1"
)
PHASE_CANDIDATE_FORMAT = (
    "memory-condense-latent-training-phase-candidate-v1"
)
CANDIDATE_PUBLICATION_FORMAT = (
    "memory-condense-latent-training-candidate-publication-v1"
)
PRODUCTION_CANDIDATE_NAME = "production-candidate.json"
PHASE_CANDIDATE_NAME = "phase-candidate.json"
CANDIDATE_RECEIPT_NAME = "candidate-receipt.json"
AUTHORITY_NOT_PINNED_REASON = "genuine_output_identities_not_pinned"
CANDIDATE_EXECUTION_DISABLED_REASON = (
    "unsafe_derived_runtime_lifecycle_not_hardened"
)


class ProductionLatentTrainingCorpusError(ValueError):
    """A candidate corpus failed its closed structural contract."""


class ProductionAuthorityNotPinned(RuntimeError):
    """No audited genuine output identities authenticate a candidate yet."""


class ProductionCandidateExecutionUnavailable(RuntimeError):
    """The candidate builder is closed before any caller input is touched."""


def _false(value: object, label: str) -> None:
    if value is not False:
        raise ValueError(f"{label} must remain false before output pinning")


def _literal(value: object, expected: str, label: str) -> None:
    if type(value) is not str or value != expected:
        raise ValueError(f"{label} must equal {expected!r}")


def _files(
    values: object,
    label: str,
) -> tuple[LatentTrainingFileIdentity, ...]:
    if type(values) is not tuple or not values or any(
        type(item) is not LatentTrainingFileIdentity for item in values
    ):
        raise TypeError(f"{label} must contain exact file identities")
    result = tuple(
        LatentTrainingFileIdentity(item.relative_path, item.sha256, item.bytes)
        for item in values
    )
    paths = tuple(item.relative_path for item in result)
    if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
        raise ValueError(f"{label} paths must be unique and sorted")
    return result


def inventory_sha256(values: tuple[LatentTrainingFileIdentity, ...]) -> str:
    return identity_sha256([item.identity_payload() for item in values])


@dataclass(frozen=True, slots=True)
class ProductionExternalLock(SealedIdentity):
    """The exact documented population/control/checkpoint projection."""

    _SEAL_FIELD = "lock_sha256"
    _SEAL_MISMATCH = "production external lock does not match"

    dataset_sha256: str
    split_manifest_sha256: str
    treatment_file_sha256: str
    sanitized_projection_sha256: str
    analysis_ordered_question_ids_sha256: str
    fit_count: int
    fit_ordered_question_ids_sha256: str
    validation_count: int
    validation_ordered_question_ids_sha256: str
    excluded_confirmation_count: int
    excluded_confirmation_ordered_question_ids_sha256: str
    compilation_policy_sha256: str
    episode_policy_sha256: str
    closure_policy_sha256: str
    base_arm_sha256: str
    episode_primary_arm_sha256: str
    matched_controls_sha256: str
    representative_policy_controls_sha256: str
    retrieval_config_sha256: str
    evaluation_policy_sha256: str
    fusion_caps_sha256: str
    qwen_atom_feature_caps_sha256: str
    bge_model_id: str
    bge_model_revision: str
    bge_checkpoint_sha256: str
    qwen_model_id: str
    qwen_model_revision: str
    qwen_checkpoint_sha256: str
    retrieval_prefix_layers: int
    retrieval_attention_layer: int
    feature_prefix_layers: int
    feature_output_layer: int
    runtime_mode: Literal["resident_bge_qwen"] = "resident_bge_qwen"
    device: Literal["cuda:0"] = "cuda:0"
    episodic_route: Literal["episode_primary"] = "episode_primary"
    closure_routing_scope: Literal["seeded_graph"] = "seeded_graph"
    scorer_labels_present: bool = False
    evaluator_label_schema_present: bool = False
    format: str = PRODUCTION_EXTERNAL_LOCK_FORMAT
    lock_sha256: str = ""

    def __post_init__(self) -> None:
        _literal(self.format, PRODUCTION_EXTERNAL_LOCK_FORMAT, "external-lock format")
        for name in (
            "dataset_sha256", "split_manifest_sha256", "treatment_file_sha256",
            "sanitized_projection_sha256", "analysis_ordered_question_ids_sha256",
            "fit_ordered_question_ids_sha256", "validation_ordered_question_ids_sha256",
            "excluded_confirmation_ordered_question_ids_sha256",
            "compilation_policy_sha256", "episode_policy_sha256",
            "closure_policy_sha256", "base_arm_sha256",
            "episode_primary_arm_sha256", "matched_controls_sha256",
            "representative_policy_controls_sha256", "retrieval_config_sha256",
            "evaluation_policy_sha256", "fusion_caps_sha256",
            "qwen_atom_feature_caps_sha256", "bge_checkpoint_sha256",
            "qwen_checkpoint_sha256",
        ):
            _sha(getattr(self, name), name)
        for name in (
            "fit_count", "validation_count", "excluded_confirmation_count",
            "retrieval_prefix_layers", "feature_prefix_layers",
        ):
            _integer(getattr(self, name), name, minimum=1)
        for name in ("retrieval_attention_layer", "feature_output_layer"):
            _integer(getattr(self, name), name)
        if self.retrieval_attention_layer >= self.retrieval_prefix_layers or (
            self.feature_output_layer >= self.feature_prefix_layers
        ):
            raise ValueError("selected Qwen layer lies outside its retained prefix")
        for name in (
            "bge_model_id", "bge_model_revision", "qwen_model_id",
            "qwen_model_revision",
        ):
            _text(getattr(self, name), name)
        _literal(self.runtime_mode, "resident_bge_qwen", "runtime mode")
        _literal(self.device, "cuda:0", "runtime device")
        _literal(self.episodic_route, "episode_primary", "episodic route")
        _literal(self.closure_routing_scope, "seeded_graph", "closure scope")
        _false(self.scorer_labels_present, "scorer-label presence")
        _false(self.evaluator_label_schema_present, "evaluator-schema presence")
        self._seal()


def locked_production_external_lock() -> ProductionExternalLock:
    """Reconstruct the lock from closed literals, avoiding a mutable singleton."""

    return ProductionExternalLock(
        dataset_sha256="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
        split_manifest_sha256="8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4",
        treatment_file_sha256="b4d1d34538fdabbd6127c339bff8167293d290eb732afc18a5d8963d12b15001",
        sanitized_projection_sha256="58a1982122d259e046ac5268de8fc3c2857a63d24c859e3bc13e4e6b9aa52ad8",
        analysis_ordered_question_ids_sha256="cf5e8648b71634e4e22be872881766e37e0dc24a2931d0c63365e075b2742046",
        fit_count=200,
        fit_ordered_question_ids_sha256="533aa545efb8032f7b181f39264c6d10a49471bd460414f420e37dc840a19c55",
        validation_count=100,
        validation_ordered_question_ids_sha256="7a67aa6f43ffb94d487fb9184f871735bd9edac1974a3154898846d1140c83a1",
        excluded_confirmation_count=200,
        excluded_confirmation_ordered_question_ids_sha256="6270b044792dbda79cd79a104ab6a519b2f81980c47522c19a196583d8c0d102",
        compilation_policy_sha256="b310b8f2abded1e8ce296b8c1dffb0fca99308cff38a742e08f40eb810d704c4",
        episode_policy_sha256="bdcc3b5aef5c961c6229a6b8ee77a19e45056913fb04533880efe86e61837118",
        closure_policy_sha256="c9eea134c827b508fb8092d207ed55be00ea5d9a96b0937ef960cc08b7977461",
        base_arm_sha256="ff4e843ddc6985eb2c97a9a7247881723792e7d4549be90204ab8e424336a6a0",
        episode_primary_arm_sha256="fe7fb2526fb8b8e46ef934d4e2a7cf0b09fa66f8dc617d456383e6df58e0fd25",
        matched_controls_sha256="c7935ad61497f2591a6e2be513a3fe769164cb634a6fb85ec15d6c9678e2a06b",
        representative_policy_controls_sha256="780bf148e69ddbbfa4583ba188b64954778a2582534a4b4d624fe777ac2e77c8",
        retrieval_config_sha256="062f2e52a6500545f35b6e17293a074589c4e68c7c4bdf3aee46dd073ab3f2ed",
        evaluation_policy_sha256="4fbb199b99aa7f60fc042d30cc7e5d09034d92d7a816c304b53471cac3a68634",
        fusion_caps_sha256="e2e453ffd238a87d536c931b57e024e6c69ab6c740bb8e3a2b2c931de7146284",
        qwen_atom_feature_caps_sha256="535b2df0bf8732cca2d0df615cbbb011ea3e5dd30c1a49d8752230a57378b6ad",
        bge_model_id="BAAI/bge-m3",
        bge_model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        bge_checkpoint_sha256="a3d5c49f064ab58d7cf5bba1c2085918f529778e88535aca7de674c9094af0b7",
        qwen_model_id="Qwen/Qwen3-8B",
        qwen_model_revision="b968826d9c46dd6066d109eabc6255188de91218",
        qwen_checkpoint_sha256="76273516aa6924b12344d5e83daa485b66459b663c745cb3b9ef51cc17c7440d",
        retrieval_prefix_layers=2, retrieval_attention_layer=1,
        feature_prefix_layers=1, feature_output_layer=0,
    )


@dataclass(frozen=True, slots=True)
class DeclaredProductionExecutionCoordinates(SealedIdentity):
    """Audit coordinates only; no execution fact is authenticated here."""

    _SEAL_FIELD = "coordinates_sha256"
    _SEAL_MISMATCH = "declared execution coordinates do not match"

    launcher_relative_path: str
    launcher_sha256: str
    source_commit: str
    package_implementation_sha256: str
    corpus_implementation_sha256: str
    route_implementation_sha256: str
    runtime_binding_sha256: str
    ordered_legacy_input_provider_identities_sha256: str
    representative_linker_identity_sha256: str
    representative_policy_factory_identity_sha256: str
    bge_checkpoint_sha256: str
    qwen_retrieval_checkpoint_sha256: str
    qwen_feature_checkpoint_sha256: str
    qwen_retrieval_contract_sha256: str
    qwen_feature_contract_sha256: str
    tracked_worktree_clean_attested: bool = False
    local_checkpoint_bytes_verified_attested: bool = False
    runtime_binding_rederived_attested: bool = False
    retrieval_qwen_execution_attested: bool = False
    feature_qwen_execution_attested: bool = False
    production_authorized: bool = False
    format: str = DECLARED_EXECUTION_FORMAT
    coordinates_sha256: str = ""

    def __post_init__(self) -> None:
        _literal(self.format, DECLARED_EXECUTION_FORMAT, "execution format")
        _safe_relative(self.launcher_relative_path, "launcher relative path")
        for name in (
            "launcher_sha256", "package_implementation_sha256",
            "corpus_implementation_sha256",
            "route_implementation_sha256", "runtime_binding_sha256",
            "ordered_legacy_input_provider_identities_sha256",
            "representative_linker_identity_sha256",
            "representative_policy_factory_identity_sha256",
            "bge_checkpoint_sha256", "qwen_retrieval_checkpoint_sha256",
            "qwen_feature_checkpoint_sha256", "qwen_retrieval_contract_sha256",
            "qwen_feature_contract_sha256",
        ):
            _sha(getattr(self, name), name)
        commit = _text(self.source_commit, "source commit").casefold()
        if commit != self.source_commit or len(commit) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise ValueError("source commit must be a canonical complete object ID")
        for name in (
            "tracked_worktree_clean_attested", "local_checkpoint_bytes_verified_attested",
            "runtime_binding_rederived_attested", "retrieval_qwen_execution_attested",
            "feature_qwen_execution_attested", "production_authorized",
        ):
            _false(getattr(self, name), name)
        self._seal()


@dataclass(frozen=True, slots=True)
class ProductionCorpusCandidateReceipt(SealedIdentity):
    """A full generic package plus non-authenticating audit coordinates."""

    _SEAL_FIELD = "candidate_sha256"
    _SEAL_MISMATCH = "production corpus candidate does not match"

    generic_root_manifest_sha256: str
    generic_root_manifest_bytes: int
    generic_corpus_sha256: str
    generic_inventory_sha256: str
    generic_population_projection_sha256: str
    generic_implementation_sha256: str
    generic_fit_partition_sha256: str
    generic_fit_manifest_file_sha256: str
    generic_fit_manifest_file_bytes: int
    generic_validation_partition_sha256: str
    generic_validation_manifest_file_sha256: str
    generic_validation_manifest_file_bytes: int
    external_lock: ProductionExternalLock
    declared_execution: DeclaredProductionExecutionCoordinates
    source_treatment_exact_type_verified: bool = False
    production_authorized: bool = False
    d1_eligible: bool = False
    validation_eligible: bool = False
    retrieval_qwen_execution_attested: bool = False
    feature_qwen_execution_attested: bool = False
    scorer_labels_present: bool = False
    evaluator_label_schema_present: bool = False
    format: str = PRODUCTION_CANDIDATE_FORMAT
    candidate_sha256: str = ""

    def __post_init__(self) -> None:
        _literal(self.format, PRODUCTION_CANDIDATE_FORMAT, "candidate format")
        for name in (
            "generic_root_manifest_sha256", "generic_corpus_sha256",
            "generic_inventory_sha256", "generic_population_projection_sha256",
            "generic_implementation_sha256", "generic_fit_partition_sha256",
            "generic_fit_manifest_file_sha256", "generic_validation_partition_sha256",
            "generic_validation_manifest_file_sha256",
        ):
            _sha(getattr(self, name), name)
        for name in (
            "generic_root_manifest_bytes", "generic_fit_manifest_file_bytes",
            "generic_validation_manifest_file_bytes",
        ):
            _integer(getattr(self, name), name, minimum=1)
        if type(self.external_lock) is not ProductionExternalLock or type(
            self.declared_execution
        ) is not DeclaredProductionExecutionCoordinates:
            raise TypeError("candidate has the wrong lock/execution type")
        self.external_lock.__post_init__()
        self.declared_execution.__post_init__()
        expected_lock = locked_production_external_lock()
        if (
            self.external_lock.lock_sha256 != expected_lock.lock_sha256
            or self.external_lock.identity_payload() != expected_lock.identity_payload()
        ):
            raise ValueError("candidate binds another external production lock")
        if (
            self.generic_implementation_sha256
            != self.declared_execution.corpus_implementation_sha256
            or self.declared_execution.bge_checkpoint_sha256
            != self.external_lock.bge_checkpoint_sha256
            or self.declared_execution.qwen_retrieval_checkpoint_sha256
            != self.external_lock.qwen_checkpoint_sha256
            or self.declared_execution.qwen_feature_checkpoint_sha256
            != self.external_lock.qwen_checkpoint_sha256
        ):
            raise ValueError("candidate execution/checkpoint joins changed")
        expected_corpus_implementation = identity_sha256(
            {
                "format": (
                    "memory-condense-latent-training-corpus-implementation-v2"
                ),
                "memory_condense_package_sha256": (
                    self.declared_execution.package_implementation_sha256
                ),
            }
        )
        if (
            self.declared_execution.corpus_implementation_sha256
            != expected_corpus_implementation
        ):
            raise ValueError("candidate package/corpus implementation join changed")
        expected_retrieval = identity_sha256(
            {
                "format": "qwen-prefix-layer-contract-v1",
                "model_id": self.external_lock.qwen_model_id,
                "model_revision": self.external_lock.qwen_model_revision,
                "checkpoint_sha256": self.external_lock.qwen_checkpoint_sha256,
                "retained_layers": self.external_lock.retrieval_prefix_layers,
                "selected_layer_kind": "attention",
                "selected_layer": self.external_lock.retrieval_attention_layer,
            }
        )
        expected_feature = identity_sha256(
            {
                "format": "qwen-prefix-layer-contract-v1",
                "model_id": self.external_lock.qwen_model_id,
                "model_revision": self.external_lock.qwen_model_revision,
                "checkpoint_sha256": self.external_lock.qwen_checkpoint_sha256,
                "retained_layers": self.external_lock.feature_prefix_layers,
                "selected_layer_kind": "output",
                "selected_layer": self.external_lock.feature_output_layer,
            }
        )
        if (
            self.declared_execution.qwen_retrieval_contract_sha256 != expected_retrieval
            or self.declared_execution.qwen_feature_contract_sha256 != expected_feature
            or expected_retrieval == expected_feature
        ):
            raise ValueError("candidate Qwen layer contracts changed")
        for name in (
            "source_treatment_exact_type_verified", "production_authorized",
            "d1_eligible", "validation_eligible", "retrieval_qwen_execution_attested",
            "feature_qwen_execution_attested", "scorer_labels_present",
            "evaluator_label_schema_present",
        ):
            _false(getattr(self, name), name)
        self._seal()


@dataclass(frozen=True, slots=True)
class ProductionPhaseCandidateReceipt(SealedIdentity):
    _SEAL_FIELD = "phase_candidate_sha256"
    _SEAL_MISMATCH = "phase candidate does not match"

    phase: Literal["fit", "validation"]
    generic_corpus_sha256: str
    generic_root_manifest_sha256: str
    production_candidate_sha256: str
    production_candidate_file_sha256: str
    production_candidate_file_bytes: int
    partition_sha256: str
    partition_file_sha256: str
    partition_file_bytes: int
    row_count: int
    ordered_question_ids_sha256: str
    inventory: tuple[LatentTrainingFileIdentity, ...]
    inventory_sha256: str
    source_treatment_exact_type_verified: bool = False
    production_authorized: bool = False
    d1_eligible: bool = False
    validation_eligible: bool = False
    optimizer_updates_authorized: bool = False
    validation_diagnostics_authorized: bool = False
    checkpoint_selection_authorized: bool = False
    retrieval_qwen_execution_attested: bool = False
    feature_qwen_execution_attested: bool = False
    sibling_partition_present: bool = False
    scorer_labels_present: bool = False
    evaluator_label_schema_present: bool = False
    format: str = PHASE_CANDIDATE_FORMAT
    phase_candidate_sha256: str = ""

    def __post_init__(self) -> None:
        _literal(self.format, PHASE_CANDIDATE_FORMAT, "phase-candidate format")
        if type(self.phase) is not str or self.phase not in {"fit", "validation"}:
            raise ValueError("phase candidate has an unsupported role")
        for name in (
            "generic_corpus_sha256", "generic_root_manifest_sha256",
            "production_candidate_sha256", "production_candidate_file_sha256",
            "partition_sha256", "partition_file_sha256",
            "ordered_question_ids_sha256", "inventory_sha256",
        ):
            _sha(getattr(self, name), name)
        for name in (
            "production_candidate_file_bytes", "partition_file_bytes", "row_count",
        ):
            _integer(getattr(self, name), name, minimum=1)
        if self.row_count != (200 if self.phase == "fit" else 100):
            raise ValueError("phase candidate row count differs from the lock")
        values = _files(self.inventory, "phase-candidate inventory")
        object.__setattr__(self, "inventory", values)
        if self.inventory_sha256 != inventory_sha256(values):
            raise ValueError("phase-candidate inventory digest changed")
        for name in (
            "source_treatment_exact_type_verified", "production_authorized",
            "d1_eligible", "validation_eligible", "optimizer_updates_authorized",
            "validation_diagnostics_authorized", "checkpoint_selection_authorized",
            "retrieval_qwen_execution_attested", "feature_qwen_execution_attested",
            "sibling_partition_present", "scorer_labels_present",
            "evaluator_label_schema_present",
        ):
            _false(getattr(self, name), name)
        self._seal()


@dataclass(frozen=True, slots=True)
class ProductionCandidatePublicationReceipt(SealedIdentity):
    _SEAL_MISMATCH = "candidate publication receipt does not match"

    generic_corpus_sha256: str
    generic_root_manifest_sha256: str
    production_candidate_sha256: str
    production_candidate_file_sha256: str
    production_candidate_file_bytes: int
    fit_phase_candidate_sha256: str
    validation_phase_candidate_sha256: str
    source_commit: str
    source_treatment_exact_type_verified: bool = False
    production_authorized: bool = False
    d1_eligible: bool = False
    validation_eligible: bool = False
    retrieval_qwen_execution_attested: bool = False
    feature_qwen_execution_attested: bool = False
    scorer_labels_present: bool = False
    evaluator_label_schema_present: bool = False
    format: str = CANDIDATE_PUBLICATION_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _literal(self.format, CANDIDATE_PUBLICATION_FORMAT, "publication format")
        for name in (
            "generic_corpus_sha256", "generic_root_manifest_sha256",
            "production_candidate_sha256", "production_candidate_file_sha256",
            "fit_phase_candidate_sha256", "validation_phase_candidate_sha256",
        ):
            _sha(getattr(self, name), name)
        _integer(self.production_candidate_file_bytes, "candidate bytes", minimum=1)
        commit = _text(self.source_commit, "publication source commit").casefold()
        if commit != self.source_commit or len(commit) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise ValueError("publication source commit is not canonical")
        for name in (
            "source_treatment_exact_type_verified", "production_authorized",
            "d1_eligible", "validation_eligible", "retrieval_qwen_execution_attested",
            "feature_qwen_execution_attested", "scorer_labels_present",
            "evaluator_label_schema_present",
        ):
            _false(getattr(self, name), name)
        self._seal()


@dataclass(frozen=True, slots=True)
class VerifiedLatentTrainingPhaseCandidate:
    partition: LatentTrainingCorpusPartitionManifest
    rows: tuple[DecodedLatentTrainingCorpusRow, ...]
    candidate: ProductionCorpusCandidateReceipt
    phase_candidate: ProductionPhaseCandidateReceipt
    production_authorized: bool = False
    d1_eligible: bool = False
    validation_eligible: bool = False

    def __post_init__(self) -> None:
        if type(self.partition) is not LatentTrainingCorpusPartitionManifest:
            raise TypeError("verified candidate has the wrong partition type")
        if type(self.candidate) is not ProductionCorpusCandidateReceipt or type(
            self.phase_candidate
        ) is not ProductionPhaseCandidateReceipt:
            raise TypeError("verified candidate has the wrong receipt types")
        self.partition.__post_init__()
        self.candidate.__post_init__()
        self.phase_candidate.__post_init__()
        role = self.phase_candidate.phase
        expected_count = (
            self.candidate.external_lock.fit_count
            if role == "fit"
            else self.candidate.external_lock.validation_count
        )
        expected_order = (
            self.candidate.external_lock.fit_ordered_question_ids_sha256
            if role == "fit"
            else self.candidate.external_lock.validation_ordered_question_ids_sha256
        )
        if (
            self.partition.partition != role
            or self.partition.partition_sha256 != self.phase_candidate.partition_sha256
            or self.phase_candidate.production_candidate_sha256
            != self.candidate.candidate_sha256
            or self.phase_candidate.generic_corpus_sha256
            != self.candidate.generic_corpus_sha256
            or self.phase_candidate.generic_root_manifest_sha256
            != self.candidate.generic_root_manifest_sha256
            or self.partition.start_ordinal != (0 if role == "fit" else 200)
            or self.partition.row_count != expected_count
            or self.phase_candidate.row_count != expected_count
            or self.partition.ordered_question_ids_sha256 != expected_order
            or self.phase_candidate.ordered_question_ids_sha256 != expected_order
        ):
            raise ValueError("verified phase-candidate joins changed")
        from memory_condense.eval.diffuse_latent_training_corpus import (
            validate_structural_latent_training_partition_rows,
        )
        validate_structural_latent_training_partition_rows(self.partition, self.rows)
        for name in ("production_authorized", "d1_eligible", "validation_eligible"):
            _false(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class VerifiedLatentTrainingCorpusCandidate:
    generic: VerifiedLatentTrainingFullCorpus
    candidate: ProductionCorpusCandidateReceipt
    fit: VerifiedLatentTrainingPhaseCandidate
    validation: VerifiedLatentTrainingPhaseCandidate
    publication: ProductionCandidatePublicationReceipt
    production_authorized: bool = False
    d1_eligible: bool = False
    validation_eligible: bool = False

    def __post_init__(self) -> None:
        if type(self.generic) is not VerifiedLatentTrainingFullCorpus:
            raise TypeError("verified candidate root has the wrong generic type")
        if type(self.candidate) is not ProductionCorpusCandidateReceipt or type(
            self.publication
        ) is not ProductionCandidatePublicationReceipt:
            raise TypeError("verified candidate root has the wrong receipt types")
        if type(self.fit) is not VerifiedLatentTrainingPhaseCandidate or type(
            self.validation
        ) is not VerifiedLatentTrainingPhaseCandidate:
            raise TypeError("verified candidate root has the wrong phase views")
        self.generic.__post_init__()
        self.candidate.__post_init__()
        self.fit.__post_init__()
        self.validation.__post_init__()
        self.publication.__post_init__()
        manifest = self.generic.manifest
        if (
            self.fit.partition.partition != "fit"
            or self.validation.partition.partition != "validation"
            or manifest.corpus_sha256 != self.candidate.generic_corpus_sha256
            or manifest.inventory_sha256 != self.candidate.generic_inventory_sha256
            or manifest.population_projection_sha256
            != self.candidate.generic_population_projection_sha256
            or manifest.implementation_sha256
            != self.candidate.generic_implementation_sha256
            or self.generic.fit.partition.partition_sha256
            != self.candidate.generic_fit_partition_sha256
            or self.generic.validation.partition.partition_sha256
            != self.candidate.generic_validation_partition_sha256
            or self.fit.partition.partition_sha256
            != self.generic.fit.partition.partition_sha256
            or self.validation.partition.partition_sha256
            != self.generic.validation.partition.partition_sha256
            or self.fit.candidate.candidate_sha256
            != self.candidate.candidate_sha256
            or self.validation.candidate.candidate_sha256
            != self.candidate.candidate_sha256
            or self.fit.phase_candidate.generic_root_manifest_sha256
            != self.candidate.generic_root_manifest_sha256
            or self.validation.phase_candidate.generic_root_manifest_sha256
            != self.candidate.generic_root_manifest_sha256
            or self.publication.generic_corpus_sha256
            != self.candidate.generic_corpus_sha256
            or self.publication.generic_root_manifest_sha256
            != self.candidate.generic_root_manifest_sha256
            or self.publication.source_commit
            != self.candidate.declared_execution.source_commit
            or self.publication.production_candidate_sha256
            != self.candidate.candidate_sha256
            or self.publication.fit_phase_candidate_sha256
            != self.fit.phase_candidate.phase_candidate_sha256
            or self.publication.validation_phase_candidate_sha256
            != self.validation.phase_candidate.phase_candidate_sha256
        ):
            raise ValueError("verified candidate root joins changed")
        for name in ("production_authorized", "d1_eligible", "validation_eligible"):
            _false(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ProductionAuthorityStatus:
    reason: Literal["genuine_output_identities_not_pinned"] = (
        AUTHORITY_NOT_PINNED_REASON
    )
    genuine_output_identities_pinned: bool = False
    production_authorized: bool = False
    d1_eligible: bool = False
    validation_eligible: bool = False

    def __post_init__(self) -> None:
        _literal(self.reason, AUTHORITY_NOT_PINNED_REASON, "authority status reason")
        for name in (
            "genuine_output_identities_pinned", "production_authorized",
            "d1_eligible", "validation_eligible",
        ):
            _false(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ProductionCandidateExecutionStatus:
    reason: Literal["unsafe_derived_runtime_lifecycle_not_hardened"] = (
        CANDIDATE_EXECUTION_DISABLED_REASON
    )
    candidate_execution_enabled: bool = False
    source_runtime_verified: bool = False
    production_authorized: bool = False
    d1_eligible: bool = False
    validation_eligible: bool = False

    def __post_init__(self) -> None:
        _literal(
            self.reason,
            CANDIDATE_EXECUTION_DISABLED_REASON,
            "candidate execution status reason",
        )
        for name in (
            "candidate_execution_enabled", "source_runtime_verified",
            "production_authorized", "d1_eligible", "validation_eligible",
        ):
            _false(getattr(self, name), name)


__all__ = [
    "AUTHORITY_NOT_PINNED_REASON", "CANDIDATE_EXECUTION_DISABLED_REASON",
    "CANDIDATE_PUBLICATION_FORMAT",
    "CANDIDATE_RECEIPT_NAME", "DECLARED_EXECUTION_FORMAT",
    "DeclaredProductionExecutionCoordinates", "PHASE_CANDIDATE_FORMAT",
    "PHASE_CANDIDATE_NAME", "PRODUCTION_CANDIDATE_FORMAT",
    "PRODUCTION_CANDIDATE_NAME", "PRODUCTION_EXTERNAL_LOCK_FORMAT",
    "ProductionAuthorityNotPinned", "ProductionAuthorityStatus",
    "ProductionCandidateExecutionStatus",
    "ProductionCandidateExecutionUnavailable",
    "ProductionCandidatePublicationReceipt", "ProductionCorpusCandidateReceipt",
    "ProductionExternalLock", "ProductionLatentTrainingCorpusError",
    "ProductionPhaseCandidateReceipt", "VerifiedLatentTrainingCorpusCandidate",
    "VerifiedLatentTrainingPhaseCandidate", "inventory_sha256",
    "locked_production_external_lock",
]
