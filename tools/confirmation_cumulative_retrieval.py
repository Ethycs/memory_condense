#!/usr/bin/env python3
"""Provider-free S0--S3 retrieval over sealed confirmation namespaces.

This module is a population-neutral consumer of the namespace/base-store
boundary in :mod:`tools.confirmation_namespace_store_adapter`.  It does not
load benchmark labels and it has no responder, judge, or other provider path.

The runtime processes one namespace at a time.  Each namespace checkpoint
binds the policy freeze, preflight, workset, verified base checkpoint, local
backend, combined-store receipt, and every cumulative evidence prefix.  A
separate pure merge replays checkpoints in the workset's declared order; the
order in which checkpoint paths are supplied is irrelevant.

The production backend is deliberately dependency-injected at the policy
edge.  It reuses ``build/open_recall_guarded_cumulative_store`` and
``retrieve_recall_guarded_cumulative_packet`` without importing any frozen
benchmark campaign or its positional constants.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import (
    identity_sha256 as runtime_identity_sha256,
    quote_sha256,
)
from memory_condense.domain.integrity import file_sha256
from tools.confirmation_cumulative_receipts import (
    CausalCoveragePredecessorReceipt,
    CumulativeRetrievalLadder,
    CumulativeRetrievalStageReceipt,
    RecallGuardedCumulativeReceipt,
)
from tools.confirmation_combined_store_receipt import CombinedCumulativeStoreReceipt
from tools.confirmation_namespace_store_adapter import (
    BACKEND_RESULT_FORMAT as BASE_BACKEND_RESULT_FORMAT,
    CHECKPOINT_FORMAT as BASE_CHECKPOINT_FORMAT,
    FROZEN_CURRENT_SOURCE_EQUIVALENCE,
    QUESTION_BINDING_FORMAT as BASE_QUESTION_BINDING_FORMAT,
    SOURCE_COORDINATE_SEMANTICS,
    SOURCE_TREATMENT_CONTRACT_FORMAT,
    ConfirmationNamespaceExecution,
    ConfirmationNamespaceWork,
    ConfirmationNamespaceWorkset,
    SealedPayload,
    compile_confirmation_namespace_workset,
    read_sealed_payload,
)
from tools.confirmation_prompt_extract import extract_stage_question
from tools.plan_confirmation_treatment_pipeline import (
    SealedConfirmationPipelinePlan,
    compile_confirmation_pipeline_preflight,
)
from tools.confirmation_canonical import (
    canonical_json_bytes,
    canonical_sha256,
    publish_no_clobber,
)
from tools.confirmation_treatment import (
    ConfirmationTreatmentInput,
)


STAGE_IDS = (
    "causal_graph_coverage_predecessor",
    "direct_episode_additions",
    "representative_episode_additions",
    "artifact_global_closure_additions",
)
SOURCE_STAGE_ID = STAGE_IDS[0]

QUESTION_FORMAT = "memory-condense-confirmation-cumulative-question-v1"
STAGE_FORMAT = f"{QUESTION_FORMAT}-stage-v1"
EVIDENCE_FORMAT = f"{QUESTION_FORMAT}-evidence-v1"
BACKEND_RESULT_FORMAT = "memory-condense-confirmation-cumulative-backend-result-v1"
CHECKPOINT_FORMAT = "memory-condense-confirmation-cumulative-checkpoint-v1"
MERGED_ROW_FORMAT = "memory-condense-confirmation-cumulative-merged-row-v1"
MERGED_FORMAT = "memory-condense-confirmation-cumulative-merged-v1"
POPULATION_IDENTITY_FORMAT = f"{MERGED_FORMAT}-population-identity-v1"
RESIDENT_PRODUCTION_MODE = "resident_bge_qwen"
STAGED_PRODUCTION_MODE = "staged_bge_then_qwen"

_SOURCE_TREATMENT_KEYS = frozenset(
    {
        "contract_sha256",
        "coordinate_semantics",
        "embedding_identity_sha256",
        "format",
        "frozen_current_source_equivalence",
        "historical_coordinate_or_byte_identity",
        "source_acquisition_config_sha256",
        "source_retrieval_policy_sha256",
        "source_scope",
        "timestamp_semantics",
    }
)

_SHA256_ALPHABET = frozenset("0123456789abcdef")
_FORBIDDEN_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "category",
        "correct",
        "desired_answer",
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


class ConfirmationCumulativeError(ValueError):
    """A cumulative confirmation input, checkpoint, or receipt failed closed."""


class ConfirmationCumulativeSealError(ConfirmationCumulativeError):
    """A durable cumulative artifact is missing, changed, or noncanonical."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationCumulativeError(message)


def _sha256(value: object, label: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and set(value) <= _SHA256_ALPHABET,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value  # type: ignore[return-value]


def _text(value: object, label: str) -> str:
    _require(type(value) is str and bool(value.strip()), f"{label} must be text")
    return value  # type: ignore[return-value]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(type(value) is dict, f"{label} must be an object")
    return value  # type: ignore[return-value]


def _object_rows(value: object, label: str) -> list[Mapping[str, Any]]:
    _require(
        type(value) is list and all(type(item) is dict for item in value),
        f"{label} must be an array of objects",
    )
    return value  # type: ignore[return-value]


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    _require(set(value) == expected, f"{label} has a non-closed schema")


def _assert_label_free(value: object, path: str = "confirmation_retrieval") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            _require(
                key not in _FORBIDDEN_FIELDS,
                f"label-bearing field is forbidden: {path}.{raw_key}",
            )
            if key == "gold_loaded":
                _require(child is False, f"gold sentinel must be false: {path}")
            _assert_label_free(child, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_label_free(child, f"{path}[{index}]")


def _sidecar_bytes(path: Path, digest: str) -> bytes:
    return f"{digest}  {path.name}\n".encode("ascii")


def _publish_sealed(
    path: Path, payload: Mapping[str, Any], *, label: str
) -> tuple[SealedPayload, bool]:
    raw = canonical_json_bytes(payload) + b"\n"
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    if path.exists() or path.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        existing = read_sealed_payload(path, label=label)
        if existing.sha256 != digest:
            raise ConfirmationCumulativeSealError(
                f"refusing to replace another {label}: {path}"
            )
        return existing, False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        publish_no_clobber(path, raw)
        publish_no_clobber(sidecar, _sidecar_bytes(path, digest))
    except (OSError, ValueError) as exc:
        raise ConfirmationCumulativeSealError(f"cannot publish {label}") from exc
    return read_sealed_payload(path, label=label), True


def _plain_json(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain_json(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(child) for child in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError(f"runtime binding contains a non-JSON value: {type(value)!r}")


@dataclass(frozen=True, slots=True)
class ConfirmationCumulativeInput:
    """All sealed, label-free inputs needed by cumulative retrieval."""

    treatment: ConfirmationTreatmentInput
    preflight: SealedConfirmationPipelinePlan
    policy_freeze: SealedPayload
    workset: ConfirmationNamespaceWorkset
    base_execution: ConfirmationNamespaceExecution

    @property
    def source_policy_sha256(self) -> str:
        """Return the policy identity, distinct from the runtime file seal."""

        return self.workset.freeze_sha256


@dataclass(frozen=True, slots=True)
class ConfirmationQuery:
    question_id: str
    row_receipt_sha256: str
    content_binding_sha256: str
    question: str
    dated_question: str


@dataclass(frozen=True, slots=True)
class VerifiedBaseNamespace:
    checkpoint: SealedPayload
    namespace_root: Path
    backend_identity_sha256: str
    execution: Mapping[str, Any]
    retrieval_receipts_by_row: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class CumulativeEvidence:
    evidence_id: str
    source_id: str
    text: str

    def projection(self) -> dict[str, str]:
        return {
            "evidence_id": self.evidence_id,
            "format": EVIDENCE_FORMAT,
            "source_id": self.source_id,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class CumulativeStage:
    stage_id: str
    stage_receipt: Mapping[str, Any]
    provider_messages: tuple[Mapping[str, str], ...]
    evidence: tuple[CumulativeEvidence, ...]

    def projection(self) -> dict[str, Any]:
        return {
            "evidence": [item.projection() for item in self.evidence],
            "format": STAGE_FORMAT,
            "provider_messages": [dict(item) for item in self.provider_messages],
            "stage_id": self.stage_id,
            "stage_receipt": _plain_json(self.stage_receipt),
        }


@dataclass(frozen=True, slots=True)
class CumulativeQuestion:
    question_id: str
    row_receipt_sha256: str
    content_binding_sha256: str
    question: str
    dated_question: str
    base_retrieval_receipt_sha256: str
    predecessor_receipt: Mapping[str, Any]
    retrieval_receipt: Mapping[str, Any]
    stages: tuple[CumulativeStage, ...]

    def projection(self) -> dict[str, Any]:
        body = {
            "base_retrieval_receipt_sha256": self.base_retrieval_receipt_sha256,
            "content_binding_sha256": self.content_binding_sha256,
            "dated_question": self.dated_question,
            "dated_question_sha256": quote_sha256(self.dated_question),
            "format": QUESTION_FORMAT,
            "physical_provider_calls": 0,
            "predecessor_receipt": _plain_json(self.predecessor_receipt),
            "question": self.question,
            "question_id": self.question_id,
            "question_id_sha256": quote_sha256(self.question_id),
            "question_sha256": quote_sha256(self.question),
            "retrieval_receipt": _plain_json(self.retrieval_receipt),
            "row_receipt_sha256": self.row_receipt_sha256,
            "stage_ids": list(STAGE_IDS),
            "stages": [item.projection() for item in self.stages],
        }
        return {**body, "question_receipt_sha256": canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class CumulativeNamespaceResult:
    namespace_id: str
    namespace_store_id: str
    base_checkpoint_sha256: str
    combined_store_receipt: Mapping[str, Any]
    compilation_receipt_sha256: str
    artifact_projection: Mapping[str, Any]
    questions: tuple[CumulativeQuestion, ...]
    physical_provider_calls: int = 0

    def projection(self) -> dict[str, Any]:
        return {
            "artifact_projection": _plain_json(self.artifact_projection),
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "combined_store_receipt": _plain_json(self.combined_store_receipt),
            "compilation_receipt_sha256": self.compilation_receipt_sha256,
            "format": BACKEND_RESULT_FORMAT,
            "namespace_id": self.namespace_id,
            "namespace_store_id": self.namespace_store_id,
            "physical_provider_calls": self.physical_provider_calls,
            "questions": [item.projection() for item in self.questions],
        }


@dataclass(frozen=True, slots=True)
class CumulativeNamespaceRequest:
    namespace_root: Path
    work: ConfirmationNamespaceWork
    queries: tuple[ConfirmationQuery, ...]
    base: VerifiedBaseNamespace
    policy_freeze_sha256: str
    preflight_sha256: str
    workset_identity_sha256: str


class CumulativeNamespaceBackend(Protocol):
    """Provider-free local backend for one already-materialized namespace."""

    @property
    def identity_sha256(self) -> str: ...

    @property
    def policy_freeze_sha256(self) -> str: ...

    def execute(self, request: CumulativeNamespaceRequest) -> CumulativeNamespaceResult: ...

    def verify(
        self,
        request: CumulativeNamespaceRequest,
        expected: Mapping[str, Any],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ConfirmationCumulativeExecution:
    checkpoint_paths: tuple[Path, ...]
    checkpoint_sha256s: tuple[str, ...]
    backend_identity_sha256: str
    created_count: int
    reused_count: int
    physical_provider_calls: int = 0


def _validate_inputs(
    inputs: ConfirmationCumulativeInput,
    *,
    token_counter: Callable[[str], int] | None,
) -> Mapping[str, ConfirmationQuery]:
    _require(type(inputs) is ConfirmationCumulativeInput, "input boundary changed type")
    treatment = inputs.treatment
    _require(
        type(treatment) is ConfirmationTreatmentInput,
        "confirmation treatment changed type",
    )
    sizes = inputs.preflight.payload.get("namespace_sizes")
    _require(type(sizes) is list and bool(sizes), "preflight schedule changed")
    expected_preflight = compile_confirmation_pipeline_preflight(
        treatment, namespace_sizes=tuple(sizes)
    )
    _require(
        expected_preflight == inputs.preflight.payload,
        "preflight does not bind the treatment",
    )
    preflight_raw = canonical_json_bytes(inputs.preflight.payload) + b"\n"
    _require(
        hashlib.sha256(preflight_raw).hexdigest() == inputs.preflight.sha256,
        "preflight byte identity changed",
    )
    expected_workset = compile_confirmation_namespace_workset(
        treatment,
        preflight=inputs.preflight,
        freeze=inputs.policy_freeze,
        target_tokens=inputs.workset.target_tokens,
        token_counter=token_counter,
    )
    _require(
        expected_workset.projection() == inputs.workset.projection(),
        "namespace workset does not bind the sealed inputs",
    )
    _require(
        inputs.base_execution.physical_provider_calls == 0,
        "base namespace stage contains provider calls",
    )

    raw_rows = inputs.preflight.payload.get("rows")
    _require(
        type(raw_rows) is list and len(raw_rows) == len(treatment.samples),
        "preflight row population changed",
    )
    queries: dict[str, ConfirmationQuery] = {}
    for sample, raw_row in zip(treatment.samples, raw_rows, strict=True):
        row = _mapping(raw_row, "preflight row")
        _require(len(sample.questions) == 1, "treatment sample changed question count")
        question = sample.questions[0]
        receipt = _sha256(row.get("row_receipt_sha256"), "preflight row receipt")
        _require(receipt not in queries, "preflight row receipt is duplicated")
        queries[receipt] = ConfirmationQuery(
            question_id=question.question_id,
            row_receipt_sha256=receipt,
            content_binding_sha256=_sha256(
                row.get("content_binding_sha256"), "preflight content binding"
            ),
            question=question.question,
            dated_question=question.dated_question,
        )
    return MappingProxyType(queries)


def _safe_relative_paths(projection: Mapping[str, Any], root: Path) -> None:
    resolved_root = root.resolve()
    for key, raw in projection.items():
        if not str(key).endswith("_relative_path"):
            continue
        relative = Path(_text(raw, f"{key}"))
        _require(not relative.is_absolute(), f"{key} must be relative")
        candidate = (resolved_root / relative).resolve()
        _require(candidate.is_relative_to(resolved_root), f"{key} escapes its namespace")


def _validate_base_checkpoint(
    sealed: SealedPayload,
    *,
    expected_digest: str,
    inputs: ConfirmationCumulativeInput,
    work: ConfirmationNamespaceWork,
) -> VerifiedBaseNamespace:
    _require(sealed.sha256 == expected_digest, "base checkpoint digest changed")
    checkpoint = sealed.payload
    body = dict(checkpoint)
    declared = _sha256(
        body.pop("checkpoint_receipt_sha256", None), "base checkpoint receipt"
    )
    _require(
        checkpoint.get("format") == BASE_CHECKPOINT_FORMAT
        and canonical_sha256(body) == declared,
        "base checkpoint self-seal changed",
    )
    expected = {
        "freeze_sha256": inputs.workset.freeze_sha256,
        "gold_loaded": False,
        "namespace_id": work.namespace_id,
        "namespace_store_id": work.namespace_store_id,
        "namespace_work_receipt_sha256": work.work_receipt_sha256,
        "physical_provider_calls": 0,
        "preflight_sha256": inputs.workset.preflight_sha256,
        "workset_identity_sha256": inputs.workset.workset_identity_sha256,
    }
    _require(
        all(checkpoint.get(key) == value for key, value in expected.items()),
        "base checkpoint belongs to another namespace or input",
    )
    backend_identity = _sha256(
        checkpoint.get("backend_identity_sha256"), "base backend identity"
    )
    execution = _mapping(checkpoint.get("execution"), "base execution")
    _exact_keys(
        checkpoint,
        frozenset(
            {
                "backend_identity_sha256",
                "checkpoint_receipt_sha256",
                "execution",
                "format",
                "freeze_sha256",
                "gold_loaded",
                "namespace_id",
                "namespace_store_id",
                "namespace_work_receipt_sha256",
                "physical_provider_calls",
                "preflight_sha256",
                "workset_identity_sha256",
            }
        ),
        "base checkpoint",
    )
    _exact_keys(
        execution,
        frozenset(
            {
                "artifact_projection",
                "format",
                "index_receipt_sha256",
                "namespace_id",
                "namespace_store_id",
                "physical_provider_calls",
                "query_artifact_receipt_sha256",
                "questions",
                "store_receipt_sha256",
            }
        ),
        "base execution",
    )
    _require(
        execution.get("format") == BASE_BACKEND_RESULT_FORMAT
        and execution.get("namespace_id") == work.namespace_id
        and execution.get("namespace_store_id") == work.namespace_store_id
        and execution.get("physical_provider_calls") == 0,
        "base execution escaped its namespace",
    )
    for key in (
        "index_receipt_sha256",
        "query_artifact_receipt_sha256",
        "store_receipt_sha256",
    ):
        _sha256(execution.get(key), f"base {key}")
    rows = _object_rows(execution.get("questions"), "base question bindings")
    _require(
        tuple(
            (row.get("question_id"), row.get("row_receipt_sha256")) for row in rows
        )
        == tuple((probe.question_id, probe.row_receipt_sha256) for probe in work.probes),
        "base execution changed probe membership",
    )
    receipts: dict[str, str] = {}
    for row in rows:
        _exact_keys(
            row,
            frozenset(
                {
                    "format",
                    "question_id",
                    "retrieval_receipt_sha256",
                    "row_receipt_sha256",
                }
            ),
            "base question binding",
        )
        _require(
            row.get("format") == BASE_QUESTION_BINDING_FORMAT,
            "base question binding format changed",
        )
        receipt = _sha256(
            row.get("retrieval_receipt_sha256"), "base retrieval receipt"
        )
        receipts[str(row["row_receipt_sha256"])] = receipt
    artifact = _mapping(execution.get("artifact_projection"), "base artifact projection")
    base_root = sealed.path.parent.parent
    namespace_root = base_root / "namespaces" / work.namespace_store_id
    _safe_relative_paths(artifact, namespace_root)
    _assert_label_free(checkpoint, "base_checkpoint")
    return VerifiedBaseNamespace(
        checkpoint=sealed,
        namespace_root=namespace_root,
        backend_identity_sha256=backend_identity,
        execution=MappingProxyType(dict(execution)),
        retrieval_receipts_by_row=MappingProxyType(receipts),
    )


def _base_namespaces(
    inputs: ConfirmationCumulativeInput,
) -> Mapping[str, tuple[Path, str]]:
    execution = inputs.base_execution
    _require(
        len(execution.checkpoint_paths)
        == len(execution.checkpoint_sha256s)
        == len(inputs.workset.namespaces),
        "base checkpoint population changed",
    )
    result: dict[str, tuple[Path, str]] = {}
    for path, expected_digest in zip(
        execution.checkpoint_paths, execution.checkpoint_sha256s, strict=True
    ):
        sealed = read_sealed_payload(path, label="base namespace checkpoint")
        _require(sealed.sha256 == expected_digest, "base execution digest changed")
        store_id = _sha256(
            sealed.payload.get("namespace_store_id"), "base namespace store ID"
        )
        _require(store_id not in result, "base namespace checkpoint is duplicated")
        result[store_id] = (Path(path), expected_digest)
    expected = {work.namespace_store_id for work in inputs.workset.namespaces}
    _require(set(result) == expected, "base namespace checkpoint set changed")
    return MappingProxyType(result)


def _queries_for_work(
    work: ConfirmationNamespaceWork,
    by_receipt: Mapping[str, ConfirmationQuery],
) -> tuple[ConfirmationQuery, ...]:
    queries: list[ConfirmationQuery] = []
    for probe in work.probes:
        _require(
            probe.row_receipt_sha256 in by_receipt,
            "namespace probe escaped the treatment",
        )
        query = by_receipt[probe.row_receipt_sha256]
        _require(
            query.question_id == probe.question_id
            and query.content_binding_sha256 == probe.content_binding_sha256,
            "namespace probe binding changed",
        )
        queries.append(query)
    return tuple(queries)


def _validate_messages(value: object) -> list[dict[str, str]]:
    _require(type(value) is list and bool(value), "stage provider messages are missing")
    result: list[dict[str, str]] = []
    for item in value:  # type: ignore[union-attr]
        message = _mapping(item, "stage provider message")
        _exact_keys(message, frozenset({"role", "content"}), "stage provider message")
        result.append(
            {
                "role": _text(message.get("role"), "provider role"),
                "content": _text(message.get("content"), "provider content"),
            }
        )
    return result


def _validate_question(
    row: Mapping[str, Any],
    *,
    query: ConfirmationQuery,
    base: VerifiedBaseNamespace,
    combined: CombinedCumulativeStoreReceipt,
) -> None:
    _exact_keys(
        row,
        frozenset(
            {
                "base_retrieval_receipt_sha256",
                "content_binding_sha256",
                "dated_question",
                "dated_question_sha256",
                "format",
                "physical_provider_calls",
                "predecessor_receipt",
                "question",
                "question_id",
                "question_id_sha256",
                "question_receipt_sha256",
                "question_sha256",
                "retrieval_receipt",
                "row_receipt_sha256",
                "stage_ids",
                "stages",
            }
        ),
        "cumulative question",
    )
    body = dict(row)
    declared = _sha256(body.pop("question_receipt_sha256"), "question receipt")
    _require(canonical_sha256(body) == declared, "question self-seal changed")
    expected = {
        "base_retrieval_receipt_sha256": base.retrieval_receipts_by_row[
            query.row_receipt_sha256
        ],
        "content_binding_sha256": query.content_binding_sha256,
        "dated_question": query.dated_question,
        "dated_question_sha256": quote_sha256(query.dated_question),
        "format": QUESTION_FORMAT,
        "physical_provider_calls": 0,
        "question": query.question,
        "question_id": query.question_id,
        "question_id_sha256": quote_sha256(query.question_id),
        "question_sha256": quote_sha256(query.question),
        "row_receipt_sha256": query.row_receipt_sha256,
        "stage_ids": list(STAGE_IDS),
    }
    _require(
        all(row.get(key) == value for key, value in expected.items()),
        "cumulative question belongs to another probe",
    )

    stages = _object_rows(row.get("stages"), "cumulative stages")
    _require(
        tuple(stage.get("stage_id") for stage in stages) == STAGE_IDS,
        "cumulative stage order changed",
    )
    typed_stages: list[CumulativeRetrievalStageReceipt] = []
    parent_ids: tuple[str, ...] = ()
    for position, (stage, expected_stage_id) in enumerate(
        zip(stages, STAGE_IDS, strict=True)
    ):
        _exact_keys(
            stage,
            frozenset(
                {"evidence", "format", "provider_messages", "stage_id", "stage_receipt"}
            ),
            "cumulative stage",
        )
        _require(
            stage.get("format") == STAGE_FORMAT
            and stage.get("stage_id") == expected_stage_id,
            "cumulative stage identity changed",
        )
        receipt = CumulativeRetrievalStageReceipt(
            **dict(_mapping(stage.get("stage_receipt"), "stage receipt"))
        )
        _require(receipt.stage_id == expected_stage_id, "stage receipt identity changed")
        evidence_rows = _object_rows(stage.get("evidence"), "stage evidence")
        evidence_ids: list[str] = []
        for evidence in evidence_rows:
            _exact_keys(
                evidence,
                frozenset({"evidence_id", "format", "source_id", "text"}),
                "stage evidence",
            )
            _require(evidence.get("format") == EVIDENCE_FORMAT, "evidence format changed")
            evidence_ids.append(_text(evidence.get("evidence_id"), "evidence ID"))
            _text(evidence.get("source_id"), "evidence source")
            _text(evidence.get("text"), "evidence text")
        ids = tuple(evidence_ids)
        _require(ids == receipt.selected_evidence_ids, "stage evidence coordinates changed")
        if position:
            _require(ids[: len(parent_ids)] == parent_ids, "stage evidence is not cumulative")
        messages = _validate_messages(stage.get("provider_messages"))
        _require(
            runtime_identity_sha256(messages) == receipt.prompt_messages_sha256
            and count_chat_prompt_token_proxy(messages) == receipt.prompt_token_proxy,
            "stage prompt receipt changed",
        )
        _require(
            extract_stage_question({"provider_messages": messages})
            == query.dated_question,
            "stage changed its declared question",
        )
        typed_stages.append(receipt)
        parent_ids = ids

    ladder = CumulativeRetrievalLadder(stages=tuple(typed_stages))
    predecessor = CausalCoveragePredecessorReceipt(
        **dict(_mapping(row.get("predecessor_receipt"), "predecessor receipt"))
    )
    final = RecallGuardedCumulativeReceipt(
        **dict(_mapping(row.get("retrieval_receipt"), "retrieval receipt"))
    )
    root, last = typed_stages[0], typed_stages[-1]
    cross_bindings = (
        predecessor.retrieval_query_sha256
        == runtime_identity_sha256({"query": query.dated_question}),
        predecessor.prompt_question_sha256
        == runtime_identity_sha256({"prompt_question": query.dated_question}),
        predecessor.retrieval_policy_sha256 == combined.retrieval_policy_sha256,
        predecessor.prompt_messages_sha256 == root.prompt_messages_sha256,
        final.predecessor_receipt_sha256 == predecessor.receipt_sha256,
        final.ladder_receipt_sha256 == ladder.receipt_sha256,
        final.protected_evidence_ids == root.selected_evidence_ids,
        final.final_evidence_ids == last.selected_evidence_ids,
        final.added_atom_ids
        == last.selected_evidence_ids[len(root.selected_evidence_ids) :],
        final.prompt_messages_sha256 == last.prompt_messages_sha256,
        final.final_context_sha256 == last.context_sha256,
        final.context_token_proxy == last.context_token_proxy,
        final.prompt_token_proxy == last.prompt_token_proxy,
        final.max_context_token_proxy == root.max_context_token_proxy,
        final.max_prompt_token_proxy == root.max_prompt_token_proxy,
        final.responder_output_token_reserve == root.responder_output_token_reserve,
        final.stage_admission_statuses
        == tuple(stage.admission_status for stage in typed_stages[1:]),
    )
    _require(all(cross_bindings), "cumulative receipts no longer cross-bind")
    _assert_label_free(row, "cumulative_question")


def _validate_backend_projection(
    projection: Mapping[str, Any], request: CumulativeNamespaceRequest
) -> CombinedCumulativeStoreReceipt:
    _exact_keys(
        projection,
        frozenset(
            {
                "artifact_projection",
                "base_checkpoint_sha256",
                "combined_store_receipt",
                "compilation_receipt_sha256",
                "format",
                "namespace_id",
                "namespace_store_id",
                "physical_provider_calls",
                "questions",
            }
        ),
        "cumulative backend result",
    )
    _require(
        projection.get("format") == BACKEND_RESULT_FORMAT
        and projection.get("namespace_id") == request.work.namespace_id
        and projection.get("namespace_store_id") == request.work.namespace_store_id
        and projection.get("base_checkpoint_sha256") == request.base.checkpoint.sha256
        and projection.get("physical_provider_calls") == 0,
        "cumulative backend escaped its namespace",
    )
    combined = CombinedCumulativeStoreReceipt(
        **dict(_mapping(projection.get("combined_store_receipt"), "combined receipt"))
    )
    _require(
        projection.get("compilation_receipt_sha256")
        == combined.compilation_receipt_sha256,
        "combined store changed its compilation receipt",
    )
    artifact = _mapping(projection.get("artifact_projection"), "cumulative artifacts")
    _safe_relative_paths(artifact, request.namespace_root)
    rows = _object_rows(projection.get("questions"), "cumulative questions")
    _require(len(rows) == len(request.queries), "cumulative question count changed")
    for row, query in zip(rows, request.queries, strict=True):
        _validate_question(row, query=query, base=request.base, combined=combined)
    _assert_label_free(projection, "cumulative_backend")
    canonical_json_bytes(projection)
    return combined


def _checkpoint_path(root: Path, work: ConfirmationNamespaceWork) -> Path:
    return root / "checkpoints" / f"{work.namespace_store_id}.json"


def _checkpoint_payload(
    *,
    inputs: ConfirmationCumulativeInput,
    work: ConfirmationNamespaceWork,
    base: VerifiedBaseNamespace,
    backend_identity_sha256: str,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "backend_identity_sha256": backend_identity_sha256,
        "base_backend_identity_sha256": base.backend_identity_sha256,
        "base_checkpoint_receipt_sha256": base.checkpoint.payload[
            "checkpoint_receipt_sha256"
        ],
        "base_checkpoint_sha256": base.checkpoint.sha256,
        "execution": dict(execution),
        "format": CHECKPOINT_FORMAT,
        "freeze_sha256": inputs.workset.freeze_sha256,
        "gold_loaded": False,
        "namespace_id": work.namespace_id,
        "namespace_store_id": work.namespace_store_id,
        "namespace_work_receipt_sha256": work.work_receipt_sha256,
        "physical_provider_calls": 0,
        "preflight_sha256": inputs.workset.preflight_sha256,
        "workset_identity_sha256": inputs.workset.workset_identity_sha256,
    }
    payload = {**body, "checkpoint_receipt_sha256": canonical_sha256(body)}
    _assert_label_free(payload, "cumulative_checkpoint")
    return payload


def _validate_checkpoint(
    sealed: SealedPayload,
    *,
    inputs: ConfirmationCumulativeInput,
    request: CumulativeNamespaceRequest,
    backend_identity_sha256: str,
) -> Mapping[str, Any]:
    checkpoint = sealed.payload
    _exact_keys(
        checkpoint,
        frozenset(
            {
                "backend_identity_sha256",
                "base_backend_identity_sha256",
                "base_checkpoint_receipt_sha256",
                "base_checkpoint_sha256",
                "checkpoint_receipt_sha256",
                "execution",
                "format",
                "freeze_sha256",
                "gold_loaded",
                "namespace_id",
                "namespace_store_id",
                "namespace_work_receipt_sha256",
                "physical_provider_calls",
                "preflight_sha256",
                "workset_identity_sha256",
            }
        ),
        "cumulative checkpoint",
    )
    body = dict(checkpoint)
    declared = _sha256(
        body.pop("checkpoint_receipt_sha256", None), "cumulative checkpoint receipt"
    )
    _require(
        checkpoint.get("format") == CHECKPOINT_FORMAT
        and canonical_sha256(body) == declared,
        "cumulative checkpoint self-seal changed",
    )
    expected = {
        "backend_identity_sha256": backend_identity_sha256,
        "base_backend_identity_sha256": request.base.backend_identity_sha256,
        "base_checkpoint_receipt_sha256": request.base.checkpoint.payload[
            "checkpoint_receipt_sha256"
        ],
        "base_checkpoint_sha256": request.base.checkpoint.sha256,
        "freeze_sha256": inputs.workset.freeze_sha256,
        "gold_loaded": False,
        "namespace_id": request.work.namespace_id,
        "namespace_store_id": request.work.namespace_store_id,
        "namespace_work_receipt_sha256": request.work.work_receipt_sha256,
        "physical_provider_calls": 0,
        "preflight_sha256": inputs.workset.preflight_sha256,
        "workset_identity_sha256": inputs.workset.workset_identity_sha256,
    }
    _require(
        all(checkpoint.get(key) == value for key, value in expected.items()),
        "cumulative checkpoint binding changed",
    )
    execution = _mapping(checkpoint.get("execution"), "cumulative execution")
    _validate_backend_projection(execution, request)
    _assert_label_free(checkpoint, "cumulative_checkpoint")
    return execution


def _request_for_work(
    *,
    inputs: ConfirmationCumulativeInput,
    output_root: Path,
    work: ConfirmationNamespaceWork,
    queries_by_receipt: Mapping[str, ConfirmationQuery],
    base_locations: Mapping[str, tuple[Path, str]],
) -> CumulativeNamespaceRequest:
    base_path, base_digest = base_locations[work.namespace_store_id]
    base_sealed = read_sealed_payload(base_path, label="base namespace checkpoint")
    base = _validate_base_checkpoint(
        base_sealed,
        expected_digest=base_digest,
        inputs=inputs,
        work=work,
    )
    return CumulativeNamespaceRequest(
        namespace_root=output_root / "namespaces" / work.namespace_store_id,
        work=work,
        queries=_queries_for_work(work, queries_by_receipt),
        base=base,
        policy_freeze_sha256=inputs.source_policy_sha256,
        preflight_sha256=inputs.preflight.sha256,
        workset_identity_sha256=inputs.workset.workset_identity_sha256,
    )


def confirmation_cumulative_requests(
    inputs: ConfirmationCumulativeInput,
    *,
    output_root: str | Path,
    token_counter: Callable[[str], int] | None = None,
) -> tuple[CumulativeNamespaceRequest, ...]:
    """Authenticate inputs and construct the declared namespace requests."""

    queries = _validate_inputs(inputs, token_counter=token_counter)
    base_locations = _base_namespaces(inputs)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    _require(root.is_dir() and not root.is_symlink(), "output root must be a directory")
    return tuple(
        _request_for_work(
            inputs=inputs,
            output_root=root,
            work=work,
            queries_by_receipt=queries,
            base_locations=base_locations,
        )
        for work in inputs.workset.namespaces
    )


def execute_confirmation_cumulative_namespaces(
    inputs: ConfirmationCumulativeInput,
    *,
    output_root: str | Path,
    backend: CumulativeNamespaceBackend,
    token_counter: Callable[[str], int] | None = None,
) -> ConfirmationCumulativeExecution:
    """Execute or verify each namespace sequentially with exact zero providers."""

    backend_identity = _sha256(backend.identity_sha256, "cumulative backend identity")
    _require(
        _sha256(backend.policy_freeze_sha256, "backend policy freeze")
        == inputs.source_policy_sha256,
        "cumulative backend binds another policy freeze",
    )
    root = Path(output_root)
    requests = confirmation_cumulative_requests(
        inputs, output_root=root, token_counter=token_counter
    )
    paths: list[Path] = []
    digests: list[str] = []
    created = 0
    reused = 0
    for request in requests:
        work = request.work
        checkpoint_path = _checkpoint_path(root, work)
        if checkpoint_path.exists() or checkpoint_path.is_symlink():
            sealed = read_sealed_payload(
                checkpoint_path, label="cumulative namespace checkpoint"
            )
            expected = _validate_checkpoint(
                sealed,
                inputs=inputs,
                request=request,
                backend_identity_sha256=backend_identity,
            )
            backend.verify(request, expected)
            reused += 1
        else:
            result = backend.execute(request)
            _require(
                type(result) is CumulativeNamespaceResult,
                "cumulative backend result changed type",
            )
            execution = result.projection()
            _validate_backend_projection(execution, request)
            payload = _checkpoint_payload(
                inputs=inputs,
                work=work,
                base=request.base,
                backend_identity_sha256=backend_identity,
                execution=execution,
            )
            sealed, was_created = _publish_sealed(
                checkpoint_path,
                payload,
                label="cumulative namespace checkpoint",
            )
            _require(was_created, "fresh cumulative checkpoint was not created")
            created += 1
        paths.append(checkpoint_path)
        digests.append(sealed.sha256)
        del request
    return ConfirmationCumulativeExecution(
        checkpoint_paths=tuple(paths),
        checkpoint_sha256s=tuple(digests),
        backend_identity_sha256=backend_identity,
        created_count=created,
        reused_count=reused,
    )


def _cumulative_locations(
    execution: ConfirmationCumulativeExecution,
    workset: ConfirmationNamespaceWorkset,
) -> Mapping[str, tuple[Path, str]]:
    _require(execution.physical_provider_calls == 0, "cumulative execution used providers")
    _require(
        len(execution.checkpoint_paths)
        == len(execution.checkpoint_sha256s)
        == len(workset.namespaces),
        "cumulative checkpoint population changed",
    )
    locations: dict[str, tuple[Path, str]] = {}
    for path, digest in zip(
        execution.checkpoint_paths, execution.checkpoint_sha256s, strict=True
    ):
        sealed = read_sealed_payload(path, label="cumulative namespace checkpoint")
        _require(sealed.sha256 == digest, "cumulative execution digest changed")
        store_id = _sha256(
            sealed.payload.get("namespace_store_id"), "cumulative namespace store ID"
        )
        _require(store_id not in locations, "cumulative checkpoint is duplicated")
        locations[store_id] = (Path(path), digest)
    _require(
        set(locations) == {work.namespace_store_id for work in workset.namespaces},
        "cumulative namespace checkpoint set changed",
    )
    return MappingProxyType(locations)


def replay_confirmation_cumulative_merge(
    inputs: ConfirmationCumulativeInput,
    *,
    cumulative_execution: ConfirmationCumulativeExecution,
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Pure deterministic ordered replay of all namespace checkpoints."""

    queries = _validate_inputs(inputs, token_counter=token_counter)
    base_locations = _base_namespaces(inputs)
    locations = _cumulative_locations(cumulative_execution, inputs.workset)
    rows: list[dict[str, Any]] = []
    namespace_refs: list[dict[str, Any]] = []
    for work in inputs.workset.namespaces:
        checkpoint_path, checkpoint_digest = locations[work.namespace_store_id]
        root = checkpoint_path.parent.parent
        request = _request_for_work(
            inputs=inputs,
            output_root=root,
            work=work,
            queries_by_receipt=queries,
            base_locations=base_locations,
        )
        sealed = read_sealed_payload(
            checkpoint_path, label="cumulative namespace checkpoint"
        )
        _require(sealed.sha256 == checkpoint_digest, "cumulative checkpoint changed")
        execution = _validate_checkpoint(
            sealed,
            inputs=inputs,
            request=request,
            backend_identity_sha256=cumulative_execution.backend_identity_sha256,
        )
        question_rows = _object_rows(execution.get("questions"), "namespace questions")
        namespace_refs.append(
            {
                "checkpoint_receipt_sha256": sealed.payload[
                    "checkpoint_receipt_sha256"
                ],
                "checkpoint_sha256": sealed.sha256,
                "namespace_id": work.namespace_id,
                "namespace_store_id": work.namespace_store_id,
                "namespace_work_receipt_sha256": work.work_receipt_sha256,
            }
        )
        for question in question_rows:
            rows.append(
                {
                    "format": MERGED_ROW_FORMAT,
                    "namespace_checkpoint_sha256": sealed.sha256,
                    "namespace_id": work.namespace_id,
                    "namespace_store_id": work.namespace_store_id,
                    "question": dict(question),
                    "source_question_receipt_sha256": question[
                        "question_receipt_sha256"
                    ],
                }
            )
        del request, execution, question_rows

    question_receipts = [
        row["source_question_receipt_sha256"] for row in rows
    ]
    population_body = {
        "dataset_sha256": inputs.workset.dataset_sha256,
        "format": POPULATION_IDENTITY_FORMAT,
        "namespace_store_ids": [
            work.namespace_store_id for work in inputs.workset.namespaces
        ],
        "ordered_row_receipt_sha256s": [
            probe.row_receipt_sha256
            for work in inputs.workset.namespaces
            for probe in work.probes
        ],
        "preflight_sha256": inputs.workset.preflight_sha256,
        "sanitized_projection_sha256": inputs.workset.sanitized_projection_sha256,
        "split_manifest_sha256": inputs.workset.split_manifest_sha256,
        "workset_identity_sha256": inputs.workset.workset_identity_sha256,
    }
    population_identity = {
        **population_body,
        "population_identity_sha256": canonical_sha256(population_body),
    }
    body = {
        "backend_identity_sha256": cumulative_execution.backend_identity_sha256,
        "format": MERGED_FORMAT,
        "freeze_sha256": inputs.workset.freeze_sha256,
        "gold_loaded": False,
        "namespace_checkpoints": namespace_refs,
        "namespace_count": len(namespace_refs),
        "physical_provider_calls": 0,
        "population_identity": population_identity,
        "population_identity_sha256": population_identity[
            "population_identity_sha256"
        ],
        "preflight_sha256": inputs.workset.preflight_sha256,
        "question_count": len(rows),
        "question_order_sha256": canonical_sha256(question_receipts),
        "question_receipt_sha256s": question_receipts,
        "questions": rows,
        "stage_ids": list(STAGE_IDS),
        "workset_identity_sha256": inputs.workset.workset_identity_sha256,
    }
    merged = {**body, "merge_receipt_sha256": canonical_sha256(body)}
    _assert_label_free(merged, "cumulative_merge")
    canonical_json_bytes(merged)
    return merged


def publish_confirmation_cumulative_merge(
    inputs: ConfirmationCumulativeInput,
    *,
    cumulative_execution: ConfirmationCumulativeExecution,
    output_path: str | Path,
    token_counter: Callable[[str], int] | None = None,
) -> tuple[SealedPayload, bool]:
    payload = replay_confirmation_cumulative_merge(
        inputs,
        cumulative_execution=cumulative_execution,
        token_counter=token_counter,
    )
    return _publish_sealed(
        Path(output_path), payload, label="confirmation cumulative merge"
    )


def matched_s0_population_from_confirmation_merge(
    inputs: ConfirmationCumulativeInput,
    *,
    cumulative_execution: ConfirmationCumulativeExecution,
    merged: SealedPayload,
    max_prompt_tokens: int,
    renderer_id: str,
    token_counter: Callable[[str], int] | None = None,
) -> Any:
    """Project the generic merge into the existing typed MatchedS0 API.

    The durable confirmation artifacts contain no ordinals.  Enumeration is
    confined to this compatibility boundary because ``MatchedS0Row`` itself
    requires a presentation-order integer; no retrieval decision consumes it.
    """

    from memory_condense.eval.fast_completion_runtime import (
        preflight_fast_completion_prompts,
    )
    from tools.matched_eval.contracts import (
        ArtifactRef,
        EvaluationMemorySnapshot,
        EvidenceItem,
        MemoryPacket,
        identity_sha256 as matched_identity_sha256,
    )
    from tools.matched_eval.population import (
        MatchedS0Population,
        MatchedS0Row,
    )
    from tools.matched_eval.renderer import render_memory_packet_for_id

    _require(
        type(max_prompt_tokens) is int and max_prompt_tokens > 0,
        "matched prompt cap must be positive",
    )
    expected = replay_confirmation_cumulative_merge(
        inputs,
        cumulative_execution=cumulative_execution,
        token_counter=token_counter,
    )
    _require(merged.payload == expected, "sealed merge differs from deterministic replay")
    raw = canonical_json_bytes(dict(merged.payload)) + b"\n"
    _require(
        hashlib.sha256(raw).hexdigest() == merged.sha256,
        "sealed merge byte identity changed",
    )
    matched_rows: list[Any] = []
    wrappers = _object_rows(merged.payload.get("questions"), "merged questions")
    for presentation_index, wrapper in enumerate(wrappers):
        question = _mapping(wrapper.get("question"), "merged question")
        stages = _object_rows(question.get("stages"), "merged question stages")
        root_stage = stages[0]
        root_receipt = CumulativeRetrievalStageReceipt(
            **dict(_mapping(root_stage.get("stage_receipt"), "S0 stage receipt"))
        )
        evidence = tuple(
            EvidenceItem(
                evidence_id=_text(row.get("evidence_id"), "S0 evidence ID"),
                source_id=_text(row.get("source_id"), "S0 evidence source"),
                text=_text(row.get("text"), "S0 evidence text"),
                token_count=count_tokens(str(row["text"])),
            )
            for row in _object_rows(root_stage.get("evidence"), "S0 evidence")
        )
        packet = MemoryPacket(
            question_id=str(question["question_id"]),
            question_sha256=str(question["question_sha256"]),
            dated_question=str(question["dated_question"]),
            dated_question_sha256=str(question["dated_question_sha256"]),
            stage_id=SOURCE_STAGE_ID,
            protected_evidence=evidence,
        )
        rendered = render_memory_packet_for_id(packet, renderer_id=renderer_id)
        matched_rows.append(
            MatchedS0Row(
                ordinal=presentation_index,
                question_part_sha256=str(question["question_receipt_sha256"]),
                source_stage_receipt_sha256=root_receipt.receipt_sha256,
                packet=packet,
                rendered_prompt=rendered,
            )
        )
    rows_tuple = tuple(matched_rows)
    prompts = preflight_fast_completion_prompts(
        tuple(row.rendered_prompt.messages for row in rows_tuple),
        max_prompt_tokens=max_prompt_tokens,
    )
    snapshot = EvaluationMemorySnapshot(
        population_identity_sha256=str(
            merged.payload["population_identity_sha256"]
        ),
        question_order_sha256=matched_identity_sha256(
            {
                "ordered_question_receipt_sha256s": list(
                    merged.payload["question_receipt_sha256s"]
                )
            }
        ),
        source_artifacts=(
            ArtifactRef(
                role="confirmation_cumulative_retrieval",
                sha256=merged.sha256,
            ),
        ),
        policy_id="policy_v5_r3_confirmation",
        renderer_id=renderer_id,
        implementation_id="confirmation_cumulative_retrieval_v1",
    )
    return MatchedS0Population(
        retrieval_sha256=merged.sha256,
        snapshot=snapshot,
        rows=rows_tuple,
        prompt_population=prompts,
        max_prompt_tokens=max_prompt_tokens,
        renderer_id=renderer_id,
    )


def project_runtime_question(
    result: Any,
    *,
    query: ConfirmationQuery,
    base_retrieval_receipt_sha256: str,
) -> CumulativeQuestion:
    """Detach one production cumulative result into the generic schema."""

    messages_by_stage = result.provider_messages_by_stage()
    evidence: list[CumulativeEvidence] = [
        CumulativeEvidence(
            evidence_id=evidence_id,
            source_id=excerpt.source_id,
            text=excerpt.text,
        )
        for evidence_id, excerpt in zip(
            result.ladder.stages[0].selected_evidence_ids,
            result.predecessor.excerpts,
            strict=True,
        )
    ]
    stages: list[CumulativeStage] = []
    for position, stage in enumerate(result.ladder.stages):
        if position:
            packet = result.addition_packets[position - 1]
            if packet is not None:
                evidence.extend(
                    CumulativeEvidence(
                        evidence_id=evidence_id,
                        source_id=_text(atom.span.source_id, "addition source"),
                        text=atom.text,
                    )
                    for evidence_id, atom in zip(
                        stage.added_evidence_ids, packet.atoms, strict=True
                    )
                )
        _require(
            tuple(item.evidence_id for item in evidence)
            == stage.selected_evidence_ids,
            "runtime stage changed evidence coordinates",
        )
        stages.append(
            CumulativeStage(
                stage_id=stage.stage_id,
                stage_receipt=MappingProxyType(asdict(stage)),
                provider_messages=tuple(
                    MappingProxyType(dict(message))
                    for message in messages_by_stage[stage.stage_id]
                ),
                evidence=tuple(evidence),
            )
        )
    return CumulativeQuestion(
        question_id=query.question_id,
        row_receipt_sha256=query.row_receipt_sha256,
        content_binding_sha256=query.content_binding_sha256,
        question=query.question,
        dated_question=query.dated_question,
        base_retrieval_receipt_sha256=base_retrieval_receipt_sha256,
        predecessor_receipt=MappingProxyType(asdict(result.predecessor.receipt)),
        retrieval_receipt=MappingProxyType(asdict(result.receipt)),
        stages=tuple(stages),
    )


def held_out_queries(queries: Sequence[ConfirmationQuery]) -> tuple[str, ...]:
    """Return the exact deduplicated raw/dated query batch for one namespace."""

    return tuple(
        dict.fromkeys(
            text for query in queries for text in (query.question, query.dated_question)
        )
    )


def validate_production_source_database(
    base: VerifiedBaseNamespace,
    *,
    source_backend_identity_sha256: str,
    source_treatment_contract_sha256: str,
    embedding_identity: Mapping[str, Any],
) -> Path:
    """Authenticate a production base checkpoint and its immutable database."""

    _require(
        base.backend_identity_sha256
        == _sha256(source_backend_identity_sha256, "source backend identity"),
        "base namespace was produced by another source backend",
    )
    artifacts = _mapping(
        base.execution.get("artifact_projection"), "base artifact projection"
    )
    contract = _mapping(
        artifacts.get("source_treatment_contract"),
        "base source treatment contract",
    )
    _exact_keys(contract, _SOURCE_TREATMENT_KEYS, "source treatment contract")
    declared_contract = _sha256(
        contract.get("contract_sha256"), "source treatment contract receipt"
    )
    contract_body = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    _require(
        canonical_sha256(contract_body)
        == declared_contract
        == _sha256(source_treatment_contract_sha256, "source treatment contract"),
        "source treatment contract changed",
    )
    _require(
        contract.get("format") == SOURCE_TREATMENT_CONTRACT_FORMAT
        and contract.get("coordinate_semantics") == SOURCE_COORDINATE_SEMANTICS
        and contract.get("frozen_current_source_equivalence")
        == FROZEN_CURRENT_SOURCE_EQUIVALENCE
        and contract.get("historical_coordinate_or_byte_identity") is False,
        "source treatment made an unsupported equivalence claim",
    )
    for name in (
        "embedding_identity_sha256",
        "source_acquisition_config_sha256",
        "source_retrieval_policy_sha256",
    ):
        _sha256(contract.get(name), f"source treatment {name}")
    _text(contract.get("source_scope"), "source treatment scope")
    _text(contract.get("timestamp_semantics"), "source timestamp semantics")
    normalized_embedding = _plain_json(embedding_identity)
    _require(
        contract.get("embedding_identity_sha256")
        == runtime_identity_sha256(normalized_embedding),
        "combined runtime embedding identity differs from the source",
    )
    relative = Path(
        _text(artifacts.get("database_relative_path"), "base database path")
    )
    _require(not relative.is_absolute(), "base database path must be relative")
    root = base.namespace_root.resolve()
    database = (root / relative).resolve()
    _require(database.is_relative_to(root), "base database escaped its namespace")
    _require(database.is_file(), "base database is missing")
    _require(
        file_sha256(database)
        == _sha256(artifacts.get("database_sha256"), "base database digest"),
        "base database changed after source verification",
    )
    return database


class ProductionCumulativeNamespaceBackend:
    """Production local store/retrieval adapter with caller-sealed policies.

    All policy-bearing objects are supplied by the confirmation entry point.
    ``runtime_policy_binding`` must be the label-free frozen-policy projection
    that certifies those objects; it is included in ``identity_sha256``.

    Resident execution requires a certified simultaneous-residency preflight.
    Staged execution is accepted only with a sealed BGE-release barrier and an
    embedder backed by frozen query artifacts; it can verify/open an existing
    combined store but cannot build one after Qwen has loaded.
    """

    def __init__(
        self,
        *,
        policy_freeze_sha256: str,
        runtime_policy_binding: Mapping[str, Any],
        source_backend_identity_sha256: str,
        source_treatment_contract_sha256: str,
        model_residency_mode: str,
        embedding_runtime_kind: str = "live_bge",
        staged_barrier_receipt_sha256: str | None = None,
        config: Any,
        embedder: Any,
        compilation_policy: Any,
        coverage_selector: Any,
        representative_linker: Any,
        episode_policy_factory: Callable[[str], Any],
        representative_policy_factory: Callable[[str], Any],
        closure_policy: Any,
        max_context_tokens: int,
        max_prompt_tokens: int,
        responder_output_token_reserve: int,
        source_router_max_sources: int,
        source_router_rrf_constant: int,
        embedding_identity: Mapping[str, Any] | None = None,
        build_store: Callable[..., Any] | None = None,
        open_store: Callable[..., Any] | None = None,
        retrieve: Callable[..., Any] | None = None,
    ) -> None:
        from memory_condense.eval._recall_guarded_cumulative_ops import (
            retrieve_recall_guarded_cumulative_packet,
        )
        from memory_condense.eval.recall_guarded_cumulative_runtime import (
            build_recall_guarded_cumulative_store,
            open_recall_guarded_cumulative_store,
        )

        self._policy_freeze_sha256 = _sha256(
            policy_freeze_sha256, "production policy freeze"
        )
        self._runtime_policy_binding = _plain_json(runtime_policy_binding)
        _assert_label_free(self._runtime_policy_binding, "runtime_policy_binding")
        self._source_backend_identity_sha256 = _sha256(
            source_backend_identity_sha256, "source backend identity"
        )
        self._source_treatment_contract_sha256 = _sha256(
            source_treatment_contract_sha256, "source treatment contract"
        )
        _require(
            model_residency_mode in {
                RESIDENT_PRODUCTION_MODE,
                STAGED_PRODUCTION_MODE,
            },
            "production model residency mode is invalid",
        )
        _require(
            self._runtime_policy_binding.get("model_residency_mode")
            == model_residency_mode,
            "runtime policy binding changed model residency mode",
        )
        _require(
            embedding_runtime_kind in {"live_bge", "sealed_frozen_queries"},
            "production embedding runtime kind is invalid",
        )
        if model_residency_mode == RESIDENT_PRODUCTION_MODE:
            _require(
                embedding_runtime_kind == "live_bge"
                and staged_barrier_receipt_sha256 is None,
                "resident mode requires the live BGE runtime",
            )
            self._residency_receipt_sha256 = _sha256(
                self._runtime_policy_binding.get(
                    "resident_preflight_receipt_sha256"
                ),
                "resident BGE-Qwen preflight receipt",
            )
        else:
            _require(
                embedding_runtime_kind == "sealed_frozen_queries",
                "staged retrieval requires sealed frozen query vectors",
            )
            self._residency_receipt_sha256 = _sha256(
                staged_barrier_receipt_sha256, "staged BGE-release barrier receipt"
            )
        self._model_residency_mode = model_residency_mode
        self._embedding_runtime_kind = embedding_runtime_kind
        self._config = config
        self._embedder = embedder
        self._compilation_policy = compilation_policy
        self._coverage_selector = coverage_selector
        self._representative_linker = representative_linker
        self._episode_policy_factory = episode_policy_factory
        self._representative_policy_factory = representative_policy_factory
        self._closure_policy = closure_policy
        self._max_context_tokens = max_context_tokens
        self._max_prompt_tokens = max_prompt_tokens
        self._reserve = responder_output_token_reserve
        self._max_sources = source_router_max_sources
        self._rrf_constant = source_router_rrf_constant
        self._embedding_identity = (
            None if embedding_identity is None else dict(embedding_identity)
        )
        self._build_store = build_store or build_recall_guarded_cumulative_store
        self._open_store = open_store or open_recall_guarded_cumulative_store
        self._retrieve = retrieve or retrieve_recall_guarded_cumulative_packet
        for value, label, minimum in (
            (max_context_tokens, "max context tokens", 1),
            (max_prompt_tokens, "max prompt tokens", 1),
            (responder_output_token_reserve, "responder reserve", 0),
            (source_router_max_sources, "source-router sources", 1),
            (source_router_rrf_constant, "source-router RRF constant", 1),
        ):
            _require(type(value) is int and value >= minimum, f"{label} is invalid")
        retrieval_projection = config.retrieval.model_dump(mode="json")
        identity = {
            "backend": "production-confirmation-cumulative-local-v1",
            "compilation_policy": _plain_json(compilation_policy),
            "embedding_identity": self._embedding_identity,
            "embedding_runtime_kind": self._embedding_runtime_kind,
            "max_context_tokens": max_context_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "policy_freeze_sha256": self._policy_freeze_sha256,
            "model_residency_mode": self._model_residency_mode,
            "responder_output_token_reserve": responder_output_token_reserve,
            "residency_receipt_sha256": self._residency_receipt_sha256,
            "retrieval": retrieval_projection,
            "runtime_policy_binding": self._runtime_policy_binding,
            "source_backend_identity_sha256": (
                self._source_backend_identity_sha256
            ),
            "source_treatment_contract_sha256": (
                self._source_treatment_contract_sha256
            ),
            "source_router_max_sources": source_router_max_sources,
            "source_router_rrf_constant": source_router_rrf_constant,
        }
        self._identity_sha256 = canonical_sha256(identity)

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    @property
    def policy_freeze_sha256(self) -> str:
        return self._policy_freeze_sha256

    def _source_database(self, base: VerifiedBaseNamespace) -> Path:
        _require(self._embedding_identity is not None, "embedding identity is required")
        return validate_production_source_database(
            base,
            source_backend_identity_sha256=self._source_backend_identity_sha256,
            source_treatment_contract_sha256=(
                self._source_treatment_contract_sha256
            ),
            embedding_identity=self._embedding_identity,
        )

    def _open_or_build(self, request: CumulativeNamespaceRequest) -> tuple[Any, str]:
        source = self._source_database(request.base)
        target = request.namespace_root / "combined-store"
        held_out = held_out_queries(request.queries)
        if target.exists():
            prepared = self._open_store(
                target,
                config=self._config,
                embedder=self._embedder,
                held_out_queries=held_out,
                coverage_selector=self._coverage_selector,
            )
            mode = "verified_cache_hit"
        else:
            _require(
                self._model_residency_mode != STAGED_PRODUCTION_MODE,
                "staged retrieval cannot build a missing combined store",
            )
            prepared = self._build_store(
                source,
                target,
                config=self._config,
                embedder=self._embedder,
                held_out_queries=held_out,
                compilation_policy=self._compilation_policy,
                coverage_selector=self._coverage_selector,
                embedding_identity=self._embedding_identity,
            )
            mode = "fresh_atomic_build"
        expected_policy = runtime_identity_sha256(
            self._config.retrieval.model_dump(mode="json")
        )
        _require(
            prepared.receipt.source_database_sha256
            == file_sha256(source)
            and prepared.receipt.retrieval_policy_sha256 == expected_policy,
            "combined store does not bind the verified base or retrieval policy",
        )
        return prepared, mode

    def execute(self, request: CumulativeNamespaceRequest) -> CumulativeNamespaceResult:
        prepared, mode = self._open_or_build(request)
        try:
            prepared.condenser.set_context_candidate_selector(self._coverage_selector)
            artifact_id = prepared.compilation.artifact.artifact_id
            questions: list[CumulativeQuestion] = []
            for query in request.queries:
                result = self._retrieve(
                    prepared.condenser,
                    query=query.question,
                    prompt_question=query.dated_question,
                    retrieval=self._config.retrieval,
                    artifact_id=artifact_id,
                    max_context_tokens=self._max_context_tokens,
                    max_prompt_tokens=self._max_prompt_tokens,
                    responder_output_token_reserve=self._reserve,
                    episode_policy=self._episode_policy_factory(artifact_id),
                    representative_linker=self._representative_linker,
                    representative_policy=self._representative_policy_factory(
                        artifact_id
                    ),
                    source_router_max_sources=self._max_sources,
                    source_router_rrf_constant=self._rrf_constant,
                    closure_policy=self._closure_policy,
                    require_certified_coverage_runtime=True,
                    require_owned_representative_runtime=True,
                )
                questions.append(
                    project_runtime_question(
                        result,
                        query=query,
                        base_retrieval_receipt_sha256=(
                            request.base.retrieval_receipts_by_row[
                                query.row_receipt_sha256
                            ]
                        ),
                    )
                )
            return CumulativeNamespaceResult(
                namespace_id=request.work.namespace_id,
                namespace_store_id=request.work.namespace_store_id,
                base_checkpoint_sha256=request.base.checkpoint.sha256,
                combined_store_receipt=MappingProxyType(asdict(prepared.receipt)),
                compilation_receipt_sha256=prepared.compilation.receipt_sha256,
                artifact_projection=MappingProxyType(
                    {
                        "combined_store_mode": mode,
                        "combined_store_relative_path": "combined-store",
                        "retained_request_token_state_bytes": 0,
                    }
                ),
                questions=tuple(questions),
            )
        finally:
            prepared.close()

    def verify(
        self,
        request: CumulativeNamespaceRequest,
        expected: Mapping[str, Any],
    ) -> None:
        prepared, _mode = self._open_or_build(request)
        try:
            _require(
                _plain_json(asdict(prepared.receipt))
                == expected.get("combined_store_receipt")
                and prepared.compilation.receipt_sha256
                == expected.get("compilation_receipt_sha256"),
                "published cumulative store changed",
            )
        finally:
            prepared.close()


__all__ = [
    "BACKEND_RESULT_FORMAT",
    "CHECKPOINT_FORMAT",
    "EVIDENCE_FORMAT",
    "MERGED_FORMAT",
    "MERGED_ROW_FORMAT",
    "QUESTION_FORMAT",
    "RESIDENT_PRODUCTION_MODE",
    "SOURCE_STAGE_ID",
    "STAGED_PRODUCTION_MODE",
    "STAGE_FORMAT",
    "STAGE_IDS",
    "ConfirmationCumulativeError",
    "ConfirmationCumulativeExecution",
    "ConfirmationCumulativeInput",
    "ConfirmationCumulativeSealError",
    "ConfirmationQuery",
    "CumulativeEvidence",
    "CumulativeNamespaceBackend",
    "CumulativeNamespaceRequest",
    "CumulativeNamespaceResult",
    "CumulativeQuestion",
    "CumulativeStage",
    "ProductionCumulativeNamespaceBackend",
    "VerifiedBaseNamespace",
    "confirmation_cumulative_requests",
    "execute_confirmation_cumulative_namespaces",
    "held_out_queries",
    "matched_s0_population_from_confirmation_merge",
    "project_runtime_question",
    "publish_confirmation_cumulative_merge",
    "replay_confirmation_cumulative_merge",
    "validate_production_source_database",
]
