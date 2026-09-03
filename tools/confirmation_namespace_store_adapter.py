#!/usr/bin/env python3
"""Population-neutral, provider-free confirmation namespace adapter.

This module is deliberately downstream of the policy-v5-r3 freeze.  It accepts
only the sanitized :class:`ConfirmationTreatmentInput`, its sealed pipeline
preflight, and the sealed policy freeze.  It never loads benchmark labels and
contains no provider execution path.

The important construction detail is that a namespace's *probe membership* is
not its complete memory.  Matching the historical 1M treatment, each namespace
starts at the first probe in its scheduled block and consumes complete histories
from that suffix until the token target is reached.  Only the scheduled block's
questions become probes.  The two populations receive separate receipts.

The generic checkpoint runner is useful with synthetic backends.  The concrete
``ProductionBaseStoreBackend`` reuses the production ``MemoryCondenser`` ingest,
SQLite/HNSW publication, and direct retrieval-pointer implementation.  Later
policy stages can consume its authenticated store and query artifacts without
rebuilding the corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from tools.plan_confirmation_treatment_pipeline import (
    FORMAT as PREFLIGHT_FORMAT,
    SealedConfirmationPipelinePlan,
    compile_confirmation_pipeline_preflight,
)
from tools.confirmation_canonical import (
    canonical_json_bytes,
    canonical_sha256,
    parse_json_bytes,
    publish_no_clobber,
)
from tools.confirmation_treatment import (
    ConfirmationTreatmentInput,
    TreatmentQuestion,
    TreatmentSample,
)


FREEZE_FORMAT = "memory-condense-policy-v5-r3-confirmation-runtime-policy-v1"
FREEZE_STATUS = "sanitized_prediction_runtime_policy"
WORKSET_FORMAT = "memory-condense-confirmation-namespace-workset-v1"
NAMESPACE_WORK_FORMAT = f"{WORKSET_FORMAT}-namespace-v1"
STORE_ID_FORMAT = f"{WORKSET_FORMAT}-store-identity-v1"
MEMBER_KEY_FORMAT = f"{WORKSET_FORMAT}-member-key-v1"
CHECKPOINT_FORMAT = "memory-condense-confirmation-namespace-checkpoint-v1"
BACKEND_RESULT_FORMAT = f"{CHECKPOINT_FORMAT}-backend-result-v1"
QUESTION_BINDING_FORMAT = f"{BACKEND_RESULT_FORMAT}-question-v1"
SOURCE_TREATMENT_CONTRACT_FORMAT = (
    "memory-condense-confirmation-source-treatment-contract-v1"
)
SOURCE_COORDINATE_SEMANTICS = (
    "population_neutral_content_addressed_namespace_coordinates_v1"
)
FROZEN_CURRENT_SOURCE_EQUIVALENCE = (
    "same_transcript_timestamp_and_ingest_semantics_distinct_content_addresses"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_CONFIRMATION_GUARDS = {
    "confirmation_role_fixed": True,
    "confirmation_tuning_forbidden": True,
    "gold_or_reference_available_during_prediction": False,
    "judge_available_before_all_predictions_freeze": False,
    "policy_change_requires_new_version": True,
    "question_local_gold_blind_routing_only": True,
    "treatment_projection_only_runtime_input": True,
    "validation_artifacts_runtime_use_forbidden": True,
    "validation_ordinals_runtime_use_forbidden": True,
    "validation_question_ids_runtime_use_forbidden": True,
}
_FORBIDDEN_RUNTIME_KEYS = frozenset(
    {
        "answer",
        "answers",
        "category",
        "correct",
        "desired_answer",
        "evidence_sources",
        "gold",
        "gold_answer",
        "ground_truth",
        "judge_verdict",
        "reference",
        "reference_answer",
        "target_owner",
        "verdict",
    }
)


class ConfirmationNamespaceError(ValueError):
    """The confirmation namespace boundary failed closed."""


class ConfirmationNamespaceSealError(ConfirmationNamespaceError):
    """A sealed input or checkpoint is missing, changed, or noncanonical."""


@dataclass(frozen=True, slots=True)
class SealedPayload:
    path: Path
    sha256: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NamespaceMember:
    row_receipt_sha256: str
    content_binding_sha256: str
    content_occurrence: int
    transcript_tokens: int

    @property
    def member_key_sha256(self) -> str:
        return canonical_sha256(
            {
                "format": MEMBER_KEY_FORMAT,
                "content_binding_sha256": self.content_binding_sha256,
                "content_occurrence": self.content_occurrence,
            }
        )

    def projection(self) -> dict[str, Any]:
        return {
            "content_binding_sha256": self.content_binding_sha256,
            "content_occurrence": self.content_occurrence,
            "member_key_sha256": self.member_key_sha256,
            "row_receipt_sha256": self.row_receipt_sha256,
            "transcript_tokens": self.transcript_tokens,
        }


@dataclass(frozen=True, slots=True)
class ProbeBinding:
    question_id: str
    row_receipt_sha256: str
    content_binding_sha256: str

    def projection(self) -> dict[str, Any]:
        return {
            "content_binding_sha256": self.content_binding_sha256,
            "question_id": self.question_id,
            "row_receipt_sha256": self.row_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class ConfirmationNamespaceWork:
    namespace_id: str
    namespace_receipt_sha256: str
    namespace_store_id: str
    target_tokens: int
    actual_tokens: int
    probes: tuple[ProbeBinding, ...]
    haystack: tuple[NamespaceMember, ...]
    work_receipt_sha256: str

    def body(self) -> dict[str, Any]:
        return {
            "actual_tokens": self.actual_tokens,
            "format": NAMESPACE_WORK_FORMAT,
            "haystack": [item.projection() for item in self.haystack],
            "haystack_membership_sha256": canonical_sha256(
                [item.member_key_sha256 for item in self.haystack]
            ),
            "namespace_id": self.namespace_id,
            "namespace_receipt_sha256": self.namespace_receipt_sha256,
            "namespace_store_id": self.namespace_store_id,
            "probe_membership_sha256": canonical_sha256(
                [item.row_receipt_sha256 for item in self.probes]
            ),
            "probes": [item.projection() for item in self.probes],
            "suffix_construction": "complete-histories-from-probe-block-start-until-token-target-v1",
            "target_tokens": self.target_tokens,
        }

    def projection(self) -> dict[str, Any]:
        return {**self.body(), "work_receipt_sha256": self.work_receipt_sha256}


@dataclass(frozen=True, slots=True)
class ConfirmationNamespaceWorkset:
    treatment_file_sha256: str
    sanitized_projection_sha256: str
    dataset_sha256: str
    split_manifest_sha256: str
    preflight_sha256: str
    freeze_sha256: str
    target_tokens: int
    namespaces: tuple[ConfirmationNamespaceWork, ...]
    workset_identity_sha256: str

    def body(self) -> dict[str, Any]:
        return {
            "dataset_sha256": self.dataset_sha256,
            "format": WORKSET_FORMAT,
            "freeze_sha256": self.freeze_sha256,
            "gold_loaded": False,
            "namespace_count": len(self.namespaces),
            "namespace_work_receipt_sha256s": [
                item.work_receipt_sha256 for item in self.namespaces
            ],
            "physical_provider_calls": 0,
            "preflight_sha256": self.preflight_sha256,
            "sanitized_projection_sha256": self.sanitized_projection_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "target_tokens": self.target_tokens,
            "treatment_file_sha256": self.treatment_file_sha256,
        }

    def projection(self) -> dict[str, Any]:
        return {
            **self.body(),
            "namespaces": [item.projection() for item in self.namespaces],
            "workset_identity_sha256": self.workset_identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class QuestionRetrievalBinding:
    question_id: str
    row_receipt_sha256: str
    retrieval_receipt_sha256: str

    def projection(self) -> dict[str, Any]:
        return {
            "format": QUESTION_BINDING_FORMAT,
            "question_id": self.question_id,
            "retrieval_receipt_sha256": self.retrieval_receipt_sha256,
            "row_receipt_sha256": self.row_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class NamespaceBackendResult:
    namespace_id: str
    namespace_store_id: str
    store_receipt_sha256: str
    index_receipt_sha256: str
    query_artifact_receipt_sha256: str
    artifact_projection: Mapping[str, Any]
    questions: tuple[QuestionRetrievalBinding, ...]
    physical_provider_calls: int = 0

    def projection(self) -> dict[str, Any]:
        return {
            "artifact_projection": dict(self.artifact_projection),
            "format": BACKEND_RESULT_FORMAT,
            "index_receipt_sha256": self.index_receipt_sha256,
            "namespace_id": self.namespace_id,
            "namespace_store_id": self.namespace_store_id,
            "physical_provider_calls": self.physical_provider_calls,
            "query_artifact_receipt_sha256": self.query_artifact_receipt_sha256,
            "questions": [item.projection() for item in self.questions],
            "store_receipt_sha256": self.store_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class NamespaceExecutionRequest:
    namespace_root: Path
    work: ConfirmationNamespaceWork
    sample: Any
    probes: tuple[ProbeBinding, ...]
    treatment: ConfirmationTreatmentInput


class NamespaceExecutionBackend(Protocol):
    @property
    def identity_sha256(self) -> str: ...

    def execute(self, request: NamespaceExecutionRequest) -> NamespaceBackendResult: ...

    def verify(
        self,
        request: NamespaceExecutionRequest,
        expected: Mapping[str, Any],
    ) -> NamespaceBackendResult: ...


@dataclass(frozen=True, slots=True)
class ConfirmationNamespaceExecution:
    checkpoint_paths: tuple[Path, ...]
    checkpoint_sha256s: tuple[str, ...]
    created_count: int
    reused_count: int
    physical_provider_calls: int = 0


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationNamespaceError(message)


def _sha256(value: object, label: str) -> str:
    _require(
        type(value) is str and _SHA256.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value  # type: ignore[return-value]


def build_production_source_treatment_contract(
    config: Any,
    embedding_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Certify the only source-acquisition runtime accepted by confirmation.

    The historical cumulative path freezes direct dense query anchors before
    it uses the packed causal-graph policy.  Passing the latter policy into
    the diffuse base publisher is invalid.  The confirmation namespace keeps
    transcript and timestamp semantics but deliberately uses content-addressed
    sample/source coordinates, so it must not claim byte-identical replay of a
    historical current-source database.
    """

    from memory_condense.domain.discourse import identity_sha256
    from memory_condense.eval._diffuse_base_contracts import (
        DiffuseBaseEmbeddingIdentity,
    )
    from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
    from memory_condense.modeling.embedding import (
        BGE_M3_CHECKPOINT_SHA256,
        DEFAULT_MODEL_DIM,
        DEFAULT_MODEL_NAME,
        DEFAULT_MODEL_REVISION,
    )

    _require(isinstance(config, EvalConfig), "source config must be an EvalConfig")
    _require(
        config.embedding_device is not None
        and bool(str(config.embedding_device).strip()),
        "source config requires an explicit embedding device",
    )
    # Applying the production derivation again is idempotent exactly when the
    # caller supplied the dedicated direct source configuration.
    expected_config = config.model_copy(
        update={"retrieval": RetrievalConfig(mode="dense", k=10)}
    )
    _require(
        config.model_dump(mode="json") == expected_config.model_dump(mode="json"),
        "production base requires source_acquisition_config, not a packed policy",
    )
    normalized_embedding = DiffuseBaseEmbeddingIdentity.model_validate(
        embedding_identity
    ).model_dump(mode="json")
    expected_embedding = {
        "backend": "sentence-transformers.encode-v1",
        "model_id": DEFAULT_MODEL_NAME,
        "model_revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "dimension": DEFAULT_MODEL_DIM,
        "device": str(config.embedding_device).casefold(),
        "batch_size": 32,
        "normalize_embeddings": False,
        "output_dtype": "float32",
    }
    _require(
        normalized_embedding == expected_embedding,
        "production base requires the frozen BGE-M3 source embedding identity",
    )
    body = {
        "coordinate_semantics": SOURCE_COORDINATE_SEMANTICS,
        "embedding_identity_sha256": identity_sha256(normalized_embedding),
        "format": SOURCE_TREATMENT_CONTRACT_FORMAT,
        "frozen_current_source_equivalence": FROZEN_CURRENT_SOURCE_EQUIVALENCE,
        "historical_coordinate_or_byte_identity": False,
        "source_acquisition_config_sha256": identity_sha256(
            config.model_dump(mode="json")
        ),
        "source_retrieval_policy_sha256": identity_sha256(
            config.retrieval.model_dump(mode="json")
        ),
        "source_scope": (
            "gold_blind_haystack_store_with_separately_addressed_question_probes"
        ),
        "timestamp_semantics": (
            "exact_longmemeval_dataset_session_timestamps_v1"
        ),
    }
    return MappingProxyType({**body, "contract_sha256": canonical_sha256(body)})


def _exact_mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value  # type: ignore[return-value]


def _sidecar_bytes(path: Path, digest: str) -> bytes:
    return f"{digest}  {path.name}\n".encode("ascii")


def read_sealed_payload(path: str | Path, *, label: str) -> SealedPayload:
    target = Path(path)
    sidecar = target.with_name(target.name + ".sha256")
    if target.is_symlink() or not target.is_file():
        raise ConfirmationNamespaceSealError(f"{label} is not a regular file")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ConfirmationNamespaceSealError(f"{label} sidecar is missing")
    raw = target.read_bytes()
    try:
        payload = parse_json_bytes(raw, label)
    except ValueError as exc:
        raise ConfirmationNamespaceSealError(f"cannot decode {label}") from exc
    if type(payload) is not dict or raw != canonical_json_bytes(payload) + b"\n":
        raise ConfirmationNamespaceSealError(f"{label} is not canonical JSON")
    digest = hashlib.sha256(raw).hexdigest()
    if sidecar.read_bytes() != _sidecar_bytes(target, digest):
        raise ConfirmationNamespaceSealError(f"{label} sidecar is invalid")
    return SealedPayload(target, digest, MappingProxyType(payload))


def _assert_runtime_gold_blind(value: object, path: str = "runtime") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            _require(
                key not in _FORBIDDEN_RUNTIME_KEYS,
                f"label-bearing runtime field is forbidden: {path}.{raw_key}",
            )
            if key == "gold_loaded":
                _require(child is False, f"gold sentinel must be false: {path}")
            _assert_runtime_gold_blind(child, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_runtime_gold_blind(child, f"{path}[{index}]")


def _validate_freeze(
    freeze: SealedPayload,
    treatment: ConfirmationTreatmentInput,
) -> str:
    value = freeze.payload
    _require(
        set(value)
        == {
            "format",
            "runtime_policy_identity_sha256",
            "source_policy_manifest_sha256",
            "status",
            "treatment_policy",
            "treatment_projection_sha256",
        },
        "runtime policy schema changed",
    )
    _require(value.get("format") == FREEZE_FORMAT, "freeze format changed")
    _require(value.get("status") == FREEZE_STATUS, "freeze status changed")
    declared = _sha256(
        value.get("runtime_policy_identity_sha256"), "runtime policy identity"
    )
    body = {
        key: child
        for key, child in value.items()
        if key != "runtime_policy_identity_sha256"
    }
    _require(declared == canonical_sha256(body), "runtime policy self-identity changed")
    source_policy_sha256 = _sha256(
        value.get("source_policy_manifest_sha256"),
        "runtime source policy manifest",
    )

    policy = _exact_mapping(value.get("treatment_policy"), "freeze treatment policy")
    _require(
        _sha256(
            value.get("treatment_projection_sha256"),
            "runtime treatment projection identity",
        )
        == canonical_sha256(policy),
        "runtime treatment projection changed",
    )
    static_root = _exact_mapping(
        policy.get("confirmation_population_static_root"),
        "freeze confirmation static root",
    )
    expected_static = {
        "dataset_sha256": treatment.dataset_sha256,
        "ordered_normalized_sample_bindings_sha256": (
            treatment.ordered_normalized_sample_bindings_sha256
        ),
        "ordered_question_ids_sha256": treatment.ordered_question_ids_sha256,
        "ordered_raw_record_bindings_sha256": (
            treatment.ordered_raw_record_bindings_sha256
        ),
        "sample_count": len(treatment.samples),
        "split_manifest_sha256": treatment.split_manifest_sha256,
    }
    _require(dict(static_root) == expected_static, "freeze static root changed")
    guards = _exact_mapping(
        policy.get("confirmation_guards"), "freeze confirmation guards"
    )
    _require(
        all(guards.get(key) is item for key, item in _REQUIRED_CONFIRMATION_GUARDS.items()),
        "freeze confirmation guards changed",
    )
    return source_policy_sha256


def _sample_tokens(
    sample: TreatmentSample,
    token_counter: Callable[[str], int],
) -> int:
    total = 0
    for _role, text in sample.turns:
        count = token_counter(text)
        _require(type(count) is int and count >= 0, "token counter returned an invalid count")
        total += count
    _require(total > 0, "confirmation history has no countable tokens")
    return total


def _production_token_counter(text: str) -> int:
    from memory_condense.domain._tokenizer import count_tokens

    return count_tokens(text)


def _namespace_store_id(
    namespace: Mapping[str, Any],
    haystack: Sequence[NamespaceMember],
    target_tokens: int,
) -> str:
    return canonical_sha256(
        {
            "format": STORE_ID_FORMAT,
            "namespace_content_population_sha256": namespace[
                "content_population_sha256"
            ],
            "ordered_haystack_member_keys": [
                item.member_key_sha256 for item in haystack
            ],
            "suffix_construction": "complete-histories-from-probe-block-start-until-token-target-v1",
            "target_tokens": target_tokens,
        }
    )


def compile_confirmation_namespace_workset(
    treatment: ConfirmationTreatmentInput,
    *,
    preflight: SealedConfirmationPipelinePlan,
    freeze: SealedPayload,
    target_tokens: int,
    token_counter: Callable[[str], int] | None = None,
) -> ConfirmationNamespaceWorkset:
    """Compile text-free work receipts for arbitrary population/schedule sizes."""

    _require(type(treatment) is ConfirmationTreatmentInput, "treatment changed type")
    _require(type(target_tokens) is int and target_tokens > 0, "target_tokens must be positive")
    _require(
        preflight.sha256
        == hashlib.sha256(
            canonical_json_bytes(preflight.payload) + b"\n"
        ).hexdigest(),
        "preflight file identity changed",
    )
    _require(
        freeze.sha256
        == hashlib.sha256(canonical_json_bytes(freeze.payload) + b"\n").hexdigest(),
        "freeze file identity changed",
    )
    _require(preflight.payload.get("format") == PREFLIGHT_FORMAT, "preflight format changed")
    raw_sizes = preflight.payload.get("namespace_sizes")
    _require(type(raw_sizes) is list and bool(raw_sizes), "preflight namespace schedule changed")
    sizes = tuple(raw_sizes)
    expected_preflight = compile_confirmation_pipeline_preflight(
        treatment, namespace_sizes=sizes
    )
    _require(
        expected_preflight == preflight.payload,
        "preflight does not exactly bind the treatment",
    )
    source_policy_sha256 = _validate_freeze(freeze, treatment)

    rows = preflight.payload["rows"]
    raw_namespaces = preflight.payload["namespaces"]
    _require(
        type(rows) is list
        and type(raw_namespaces) is list
        and len(rows) == len(treatment.samples)
        and len(raw_namespaces) == len(sizes),
        "preflight population shape changed",
    )
    counter = token_counter or _production_token_counter
    counts = tuple(_sample_tokens(sample, counter) for sample in treatment.samples)
    occurrences: dict[str, int] = {}
    members: list[NamespaceMember] = []
    for row, count in zip(rows, counts, strict=True):
        _require(type(row) is dict, "preflight row changed type")
        binding = _sha256(row.get("content_binding_sha256"), "row content binding")
        occurrence = occurrences.get(binding, 0)
        occurrences[binding] = occurrence + 1
        members.append(
            NamespaceMember(
                row_receipt_sha256=_sha256(
                    row.get("row_receipt_sha256"), "preflight row receipt"
                ),
                content_binding_sha256=binding,
                content_occurrence=occurrence,
                transcript_tokens=count,
            )
        )

    compiled: list[ConfirmationNamespaceWork] = []
    cursor = 0
    for size, raw_namespace in zip(sizes, raw_namespaces, strict=True):
        _require(type(size) is int and size > 0, "namespace size changed")
        namespace = _exact_mapping(raw_namespace, "preflight namespace")
        probe_stop = cursor + size
        probe_rows = rows[cursor:probe_stop]
        probe_members = tuple(
            ProbeBinding(
                question_id=str(row["question_id"]),
                row_receipt_sha256=_sha256(
                    row["row_receipt_sha256"], "probe row receipt"
                ),
                content_binding_sha256=_sha256(
                    row["content_binding_sha256"], "probe content binding"
                ),
            )
            for row in probe_rows
        )
        total = 0
        stop = cursor
        while stop < len(members) and total < target_tokens:
            total += members[stop].transcript_tokens
            stop += 1
        _require(
            total >= target_tokens,
            "treatment suffix cannot reach the declared namespace token target",
        )
        # The historical composer stops at the token target.  If that would
        # occur before the declared probe block is complete, this schedule is
        # incompatible rather than silently changing the construction rule.
        _require(
            stop >= probe_stop,
            "token target is reached before the declared probe block is complete",
        )
        haystack = tuple(members[cursor:stop])
        store_id = _namespace_store_id(namespace, haystack, target_tokens)
        body = {
            "actual_tokens": total,
            "format": NAMESPACE_WORK_FORMAT,
            "haystack": [item.projection() for item in haystack],
            "haystack_membership_sha256": canonical_sha256(
                [item.member_key_sha256 for item in haystack]
            ),
            "namespace_id": _sha256(namespace.get("namespace_id"), "namespace ID"),
            "namespace_receipt_sha256": _sha256(
                namespace.get("namespace_receipt_sha256"), "namespace receipt"
            ),
            "namespace_store_id": store_id,
            "probe_membership_sha256": canonical_sha256(
                [item.row_receipt_sha256 for item in probe_members]
            ),
            "probes": [item.projection() for item in probe_members],
            "suffix_construction": "complete-histories-from-probe-block-start-until-token-target-v1",
            "target_tokens": target_tokens,
        }
        compiled.append(
            ConfirmationNamespaceWork(
                namespace_id=str(body["namespace_id"]),
                namespace_receipt_sha256=str(body["namespace_receipt_sha256"]),
                namespace_store_id=store_id,
                target_tokens=target_tokens,
                actual_tokens=total,
                probes=probe_members,
                haystack=haystack,
                work_receipt_sha256=canonical_sha256(body),
            )
        )
        cursor = probe_stop
    _require(cursor == len(rows), "namespace schedule did not consume every probe")

    body = {
        "dataset_sha256": treatment.dataset_sha256,
        "format": WORKSET_FORMAT,
        "freeze_sha256": source_policy_sha256,
        "gold_loaded": False,
        "namespace_count": len(compiled),
        "namespace_work_receipt_sha256s": [
            item.work_receipt_sha256 for item in compiled
        ],
        "physical_provider_calls": 0,
        "preflight_sha256": preflight.sha256,
        "sanitized_projection_sha256": treatment.sanitized_projection_sha256,
        "split_manifest_sha256": treatment.split_manifest_sha256,
        "target_tokens": target_tokens,
        "treatment_file_sha256": treatment.file_sha256,
    }
    workset = ConfirmationNamespaceWorkset(
        treatment_file_sha256=treatment.file_sha256,
        sanitized_projection_sha256=treatment.sanitized_projection_sha256,
        dataset_sha256=treatment.dataset_sha256,
        split_manifest_sha256=treatment.split_manifest_sha256,
        preflight_sha256=preflight.sha256,
        freeze_sha256=source_policy_sha256,
        target_tokens=target_tokens,
        namespaces=tuple(compiled),
        workset_identity_sha256=canonical_sha256(body),
    )
    _assert_runtime_gold_blind(workset.projection())
    return workset


def _rows_by_receipt(
    treatment: ConfirmationTreatmentInput,
    preflight: SealedConfirmationPipelinePlan,
) -> Mapping[str, tuple[TreatmentSample, Mapping[str, Any]]]:
    rows = preflight.payload.get("rows")
    _require(type(rows) is list, "preflight rows changed")
    _require(len(rows) == len(treatment.samples), "preflight row count changed")
    return {
        str(row["row_receipt_sha256"]): (sample, row)
        for sample, row in zip(treatment.samples, rows, strict=True)
    }


def build_namespace_sample(
    treatment: ConfirmationTreatmentInput,
    preflight: SealedConfirmationPipelinePlan,
    work: ConfirmationNamespaceWork,
) -> Any:
    """Materialize at most one suffix-composite sample in process memory."""

    from memory_condense.eval.diffuse_longmemeval_runtime import (
        gold_blind_from_treatment_sample,
    )

    lookup = _rows_by_receipt(treatment, preflight)
    turns: list[tuple[str, str]] = []
    sources: list[str | None] = []
    timestamps: list[Any] = []
    for member in work.haystack:
        _require(member.row_receipt_sha256 in lookup, "haystack member escaped treatment")
        sample, row = lookup[member.row_receipt_sha256]
        _require(
            row["content_binding_sha256"] == member.content_binding_sha256,
            "haystack member content binding changed",
        )
        prefix = member.member_key_sha256
        turns.extend(sample.turns)
        sources.extend(
            None if source_id is None else f"{prefix}::{source_id}"
            for source_id in sample.turn_source_ids
        )
        timestamps.extend(sample.turn_created_at)

    questions: list[TreatmentQuestion] = []
    for probe in work.probes:
        _require(probe.row_receipt_sha256 in lookup, "probe escaped treatment")
        sample, _row = lookup[probe.row_receipt_sha256]
        _require(len(sample.questions) == 1, "probe sample changed question count")
        question = sample.questions[0]
        _require(question.question_id == probe.question_id, "probe identity changed")
        questions.append(question)

    composite = TreatmentSample(
        # Store identity deliberately excludes benchmark/sample numbering.  It
        # is a content address for this exact suffix-composite memory.
        sample_id=f"confirmation-namespace-{work.namespace_store_id}",
        turns=tuple(turns),
        turn_source_ids=tuple(sources),
        turn_created_at=tuple(timestamps),
        questions=tuple(questions),
    )
    blind = gold_blind_from_treatment_sample(composite)
    _require(
        tuple(item.question_id for item in blind.questions)
        == tuple(item.question_id for item in work.probes),
        "composite probe order changed",
    )
    return blind


def _validate_backend_result(
    result: NamespaceBackendResult,
    work: ConfirmationNamespaceWork,
) -> dict[str, Any]:
    _require(type(result) is NamespaceBackendResult, "backend result changed type")
    _require(
        result.namespace_id == work.namespace_id
        and result.namespace_store_id == work.namespace_store_id,
        "backend result escaped its namespace",
    )
    _require(result.physical_provider_calls == 0, "local stage made a provider call")
    for value, label in (
        (result.store_receipt_sha256, "store receipt"),
        (result.index_receipt_sha256, "index receipt"),
        (result.query_artifact_receipt_sha256, "query artifact receipt"),
    ):
        _sha256(value, label)
    _require(
        tuple((item.question_id, item.row_receipt_sha256) for item in result.questions)
        == tuple((item.question_id, item.row_receipt_sha256) for item in work.probes),
        "backend retrieval rows escaped the declared probe membership",
    )
    for item in result.questions:
        _sha256(item.retrieval_receipt_sha256, "retrieval receipt")
    projection = result.projection()
    _assert_runtime_gold_blind(projection)
    # Canonicalization also rejects opaque runtime objects in artifact metadata.
    canonical_json_bytes(projection)
    return projection


def _checkpoint_path(root: Path, work: ConfirmationNamespaceWork) -> Path:
    return root / "checkpoints" / f"{work.namespace_store_id}.json"


def _read_checkpoint(path: Path) -> SealedPayload:
    return read_sealed_payload(path, label="confirmation namespace checkpoint")


def _publish_checkpoint(path: Path, payload: Mapping[str, Any]) -> SealedPayload:
    raw = canonical_json_bytes(payload) + b"\n"
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    if path.exists() or path.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        existing = _read_checkpoint(path)
        if existing.sha256 != digest:
            raise ConfirmationNamespaceSealError(
                f"refusing to replace another namespace checkpoint: {path}"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_no_clobber(path, raw)
    try:
        publish_no_clobber(sidecar, _sidecar_bytes(path, digest))
    except BaseException:
        # Leave the data file in place.  An incomplete pair fails closed and is
        # never silently replaced on resume.
        raise
    return _read_checkpoint(path)


def execute_confirmation_namespaces(
    treatment: ConfirmationTreatmentInput,
    *,
    preflight: SealedConfirmationPipelinePlan,
    workset: ConfirmationNamespaceWorkset,
    output_root: str | Path,
    backend: NamespaceExecutionBackend,
) -> ConfirmationNamespaceExecution:
    """Execute/verify namespaces sequentially with no-clobber checkpoints."""

    _require(workset.preflight_sha256 == preflight.sha256, "workset preflight changed")
    _require(workset.treatment_file_sha256 == treatment.file_sha256, "workset treatment changed")
    backend_identity = _sha256(backend.identity_sha256, "backend identity")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    _require(root.is_dir() and not root.is_symlink(), "output root must be a regular directory")
    paths: list[Path] = []
    digests: list[str] = []
    created = 0
    reused = 0

    for work in workset.namespaces:
        namespace_root = root / "namespaces" / work.namespace_store_id
        if namespace_root.exists() or namespace_root.is_symlink():
            _require(
                namespace_root.is_dir() and not namespace_root.is_symlink(),
                "namespace artifact root must be a regular directory",
            )
        checkpoint_path = _checkpoint_path(root, work)
        sample = build_namespace_sample(treatment, preflight, work)
        request = NamespaceExecutionRequest(
            namespace_root=namespace_root,
            work=work,
            sample=sample,
            probes=work.probes,
            treatment=treatment,
        )
        if checkpoint_path.exists() or checkpoint_path.is_symlink():
            sealed = _read_checkpoint(checkpoint_path)
            checkpoint = sealed.payload
            checkpoint_body = {
                key: child
                for key, child in checkpoint.items()
                if key != "checkpoint_receipt_sha256"
            }
            _require(
                checkpoint.get("format") == CHECKPOINT_FORMAT
                and checkpoint.get("checkpoint_receipt_sha256")
                == canonical_sha256(checkpoint_body)
                and checkpoint.get("workset_identity_sha256")
                == workset.workset_identity_sha256
                and checkpoint.get("namespace_work_receipt_sha256")
                == work.work_receipt_sha256
                and checkpoint.get("backend_identity_sha256") == backend_identity
                and checkpoint.get("physical_provider_calls") == 0
                and checkpoint.get("gold_loaded") is False,
                "namespace checkpoint binding changed",
            )
            raw_execution = _exact_mapping(
                checkpoint.get("execution"), "checkpoint execution"
            )
            verified = backend.verify(request, raw_execution)
            _require(
                _validate_backend_result(verified, work) == raw_execution,
                "backend verification differs from checkpoint",
            )
            reused += 1
        else:
            result = backend.execute(request)
            execution = _validate_backend_result(result, work)
            body = {
                "backend_identity_sha256": backend_identity,
                "execution": execution,
                "format": CHECKPOINT_FORMAT,
                "freeze_sha256": workset.freeze_sha256,
                "gold_loaded": False,
                "namespace_id": work.namespace_id,
                "namespace_store_id": work.namespace_store_id,
                "namespace_work_receipt_sha256": work.work_receipt_sha256,
                "physical_provider_calls": 0,
                "preflight_sha256": workset.preflight_sha256,
                "workset_identity_sha256": workset.workset_identity_sha256,
            }
            payload = {**body, "checkpoint_receipt_sha256": canonical_sha256(body)}
            _assert_runtime_gold_blind(payload)
            sealed = _publish_checkpoint(checkpoint_path, payload)
            created += 1
        paths.append(checkpoint_path)
        digests.append(sealed.sha256)
        del sample, request

    return ConfirmationNamespaceExecution(
        checkpoint_paths=tuple(paths),
        checkpoint_sha256s=tuple(digests),
        created_count=created,
        reused_count=reused,
    )


class ProductionBaseStoreBackend:
    """Production MemoryCondenser ingest/index/direct-retrieval backend.

    The caller supplies the already resolved source-acquisition ``EvalConfig``,
    local embedder, certified condenser factory, and exact embedding identity.
    The class exposes no network/provider callable.
    """

    def __init__(
        self,
        *,
        config: Any,
        embedder: Any,
        embedding_identity: Mapping[str, Any],
        condenser_factory: Callable[[Path], Any],
        implementation_digest: str | None = None,
        environment_digest: str | None = None,
    ) -> None:
        from memory_condense.eval._diffuse_base_store import (
            declared_factory_identity,
        )

        self._config = config
        self._embedder = embedder
        self._embedding_identity = dict(embedding_identity)
        self._source_treatment_contract = build_production_source_treatment_contract(
            config, self._embedding_identity
        )
        self._condenser_factory = condenser_factory
        self._build_runtime = declared_factory_identity(condenser_factory)
        self._implementation_digest = implementation_digest
        self._environment_digest = environment_digest
        body = {
            "backend": "production-diffuse-base-memory-condenser-v1",
            "build_runtime_identity": self._build_runtime.model_dump(mode="json"),
            "embedding_identity": self._embedding_identity,
            "environment_digest": environment_digest,
            "implementation_digest": implementation_digest,
            "retrieval_config": config.model_dump(mode="json"),
            "source_treatment_contract": dict(self._source_treatment_contract),
        }
        self._identity_sha256 = canonical_sha256(body)

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    def _treatment_identity(self, request: NamespaceExecutionRequest) -> Any:
        from memory_condense.eval._diffuse_base_contracts import (
            DiffuseBaseTreatmentIdentity,
        )

        return DiffuseBaseTreatmentIdentity(
            treatment_file_sha256=request.treatment.file_sha256,
            sanitized_projection_sha256=request.work.work_receipt_sha256,
            dataset_sha256=request.treatment.dataset_sha256,
            split_manifest_sha256=request.treatment.split_manifest_sha256,
            ordered_question_ids_sha256=canonical_sha256(
                [probe.question_id for probe in request.probes]
            ),
            sample_count=1,
            sample_ordinal=0,
        )

    def _projection(self, request: NamespaceExecutionRequest, base: Any) -> NamespaceBackendResult:
        from memory_condense.eval._diffuse_base_contracts import STORE_DIRECTORY_NAME

        frozen = tuple(base.frozen_query_inputs)
        _require(len(frozen) == len(request.probes), "base retrieval row count changed")
        questions = tuple(
            QuestionRetrievalBinding(
                question_id=probe.question_id,
                row_receipt_sha256=probe.row_receipt_sha256,
                retrieval_receipt_sha256=row.receipt_sha256,
            )
            for probe, row in zip(request.probes, frozen, strict=True)
        )
        store_dir = Path(base.store_path) / STORE_DIRECTORY_NAME
        artifact = MappingProxyType(
            {
                "base_store_key": base.base_store_key,
                "database_relative_path": str(
                    (store_dir / "memory.db").relative_to(request.namespace_root)
                ).replace("\\", "/"),
                "database_sha256": base.store_manifest.database_sha256,
                "index_sha256": base.store_manifest.index_sha256,
                "query_input_key": base.query_input_key,
                "query_inputs_relative_path": str(
                    Path(base.query_inputs_path).relative_to(request.namespace_root)
                ).replace("\\", "/"),
                "store_relative_path": str(
                    store_dir.relative_to(request.namespace_root)
                ).replace("\\", "/"),
                "source_treatment_contract": dict(
                    self._source_treatment_contract
                ),
                "store_manifest_sha256": base.store_manifest_sha256,
            }
        )
        return NamespaceBackendResult(
            namespace_id=request.work.namespace_id,
            namespace_store_id=request.work.namespace_store_id,
            store_receipt_sha256=base.store_manifest.artifact_sha256,
            index_receipt_sha256=base.store_manifest.index_sha256,
            query_artifact_receipt_sha256=base.query_manifest.artifact_sha256,
            artifact_projection=artifact,
            questions=questions,
        )

    def execute(self, request: NamespaceExecutionRequest) -> NamespaceBackendResult:
        from memory_condense.eval.diffuse_longmemeval_base import (
            publish_diffuse_longmemeval_base,
        )

        base = publish_diffuse_longmemeval_base(
            request.namespace_root,
            treatment_identity=self._treatment_identity(request),
            sample=request.sample,
            config=self._config,
            embedding_identity=self._embedding_identity,
            build_runtime_identity=self._build_runtime,
            embedder=self._embedder,
            condenser_factory=self._condenser_factory,
            implementation_digest=self._implementation_digest,
            environment_digest=self._environment_digest,
        )
        return self._projection(request, base)

    def verify(
        self,
        request: NamespaceExecutionRequest,
        expected: Mapping[str, Any],
    ) -> NamespaceBackendResult:
        from memory_condense.eval.diffuse_longmemeval_base import (
            verify_diffuse_longmemeval_base,
        )

        base = verify_diffuse_longmemeval_base(
            request.namespace_root,
            treatment_identity=self._treatment_identity(request),
            sample=request.sample,
            config=self._config,
            embedding_identity=self._embedding_identity,
            build_runtime_identity=self._build_runtime,
            implementation_digest=self._implementation_digest,
            environment_digest=self._environment_digest,
        )
        result = self._projection(request, base)
        _require(result.projection() == expected, "published base store changed")
        return result


__all__ = [
    "BACKEND_RESULT_FORMAT",
    "CHECKPOINT_FORMAT",
    "ConfirmationNamespaceError",
    "ConfirmationNamespaceExecution",
    "ConfirmationNamespaceSealError",
    "ConfirmationNamespaceWork",
    "ConfirmationNamespaceWorkset",
    "NamespaceBackendResult",
    "NamespaceExecutionBackend",
    "NamespaceExecutionRequest",
    "NamespaceMember",
    "ProbeBinding",
    "ProductionBaseStoreBackend",
    "QuestionRetrievalBinding",
    "SealedPayload",
    "FROZEN_CURRENT_SOURCE_EQUIVALENCE",
    "SOURCE_COORDINATE_SEMANTICS",
    "SOURCE_TREATMENT_CONTRACT_FORMAT",
    "build_namespace_sample",
    "build_production_source_treatment_contract",
    "compile_confirmation_namespace_workset",
    "execute_confirmation_namespaces",
    "read_sealed_payload",
]
