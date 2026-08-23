"""One-pass feature execution over a sealed fast retrieval artifact.

The session consumes the immutable projection produced by
``recall_guarded_cumulative_fast_artifact``.  It never opens the source
artifact, corpus, or a store.  Exact question and evidence strings are
deduplicated globally, encoded by one ``encode_layers`` call, routed with a
fixed CAV bank, and reduced immediately to tensor-free matched-readout
receipts.

The encoder and router remain caller-owned.  In particular, this function
does not close or persist either runtime and never returns token IDs, hidden
states, feature tensors, or router tensors.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    CAMPAIGN_FORMAT,
    RETRIEVAL_FORMAT,
    STAGE_IDS,
    FastEvidence,
    FastFeatureRow,
    FastProviderMessage,
    FastQuestionParseReceipt,
    FastRetrievalArtifact,
    FastRetrievalQuestion,
    FastRetrievalStage,
)
from memory_condense.search.fusion.fixed_cav_router import FixedCAVForward
from memory_condense.search.fusion.steered_readout import (
    MatchedSteeredReadout,
    matched_steered_readout,
)


FAST_CAV_FEATURE_BACKEND_FORMAT = "qwen3_prefix.encode_layers.mean_pool.v1"
FAST_CAV_STAGE_RECEIPT_FORMAT = "memory-condense-fast-cav-stage-receipt-v1"
FAST_CAV_SESSION_RECEIPT_FORMAT = "memory-condense-fast-cav-session-receipt-v1"
FAST_CAV_MAX_ENCODER_ROWS = 1024
FAST_CAV_MAX_HIDDEN_DIM = 4096
FAST_CAV_MAX_BATCH_SIZE = 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FLOAT_DTYPES = frozenset(
    {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}
)


class FastCAVFeatureSessionError(ValueError):
    """Raised when a fast feature execution cannot preserve provenance."""


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise FastCAVFeatureSessionError(
            f"{label} must be an exact lowercase SHA-256 digest"
        )
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise FastCAVFeatureSessionError(f"{label} must be an exact non-empty string")
    return value


def _exact_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise FastCAVFeatureSessionError(
            f"{label} must be an exact integer >= {minimum}"
        )
    if maximum is not None and value > maximum:
        raise FastCAVFeatureSessionError(f"{label} exceeds {maximum}")
    return value


def _zero(value: object, label: str) -> int:
    if type(value) is not int or value != 0:
        raise FastCAVFeatureSessionError(f"{label} must remain exactly zero")
    return 0


def _quote_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _feature_row_sha256(question: str, evidence_text: str) -> str:
    return identity_sha256(
        {
            "format": "memory-condense-fast-feature-row-v1",
            "question": question,
            "evidence_text": evidence_text,
        }
    )


def _tuple_of_exact_strings(values: object, label: str) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise FastCAVFeatureSessionError(f"{label} must be an exact tuple")
    normalized: list[str] = []
    for index, value in enumerate(values):
        normalized.append(_text(value, f"{label}[{index}]"))
    return tuple(normalized)


def _stage_receipt_payload(receipt: "FastCAVStageReceipt") -> dict[str, Any]:
    return {
        "format": receipt.format,
        "artifact_sha256": receipt.artifact_sha256,
        "placement_ordinal": receipt.placement_ordinal,
        "question_ordinal": receipt.question_ordinal,
        "question_id": receipt.question_id,
        "question_sha256": receipt.question_sha256,
        "dated_question_sha256": receipt.dated_question_sha256,
        "stage_ordinal": receipt.stage_ordinal,
        "stage_id": receipt.stage_id,
        "source_stage_receipt_sha256": receipt.source_stage_receipt_sha256,
        "evidence_projection_sha256": receipt.evidence_projection_sha256,
        "evidence_feature_row_indices": list(receipt.evidence_feature_row_indices),
        "evidence_ids": list(receipt.evidence_ids),
        "source_ids": list(receipt.source_ids),
        "evidence_text_sha256s": list(receipt.evidence_text_sha256s),
        "packet_identity_sha256": receipt.packet_identity_sha256,
        "feature_backend_identity_sha256": receipt.feature_backend_identity_sha256,
        "feature_checkpoint_sha256": receipt.feature_checkpoint_sha256,
        "feature_layer": receipt.feature_layer,
        "feature_hidden_dim": receipt.feature_hidden_dim,
        "feature_encoder_runtime_dtype": receipt.feature_encoder_runtime_dtype,
        "feature_encoder_runtime_device": receipt.feature_encoder_runtime_device,
        "feature_encoder_prefix_layers": receipt.feature_encoder_prefix_layers,
        "router_runtime_identity_sha256": receipt.router_runtime_identity_sha256,
        "router_bank_identity_sha256": receipt.router_bank_identity_sha256,
        "router_call_ordinal": receipt.router_call_ordinal,
        "reused_router_result": receipt.reused_router_result,
        "readout_sha256": receipt.readout.readout_sha256,
        "result_retained_tensor_bytes": receipt.result_retained_tensor_bytes,
        "retained_token_id_count": receipt.retained_token_id_count,
        "persisted_token_state_bytes": receipt.persisted_token_state_bytes,
    }


@dataclass(frozen=True, slots=True)
class FastCAVStageReceipt:
    """One cumulative-stage result with exact, tensor-free provenance."""

    artifact_sha256: str
    placement_ordinal: int
    question_ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    stage_ordinal: int
    stage_id: str
    source_stage_receipt_sha256: str
    evidence_projection_sha256: str
    evidence_feature_row_indices: tuple[int, ...]
    evidence_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    evidence_text_sha256s: tuple[str, ...]
    packet_identity_sha256: str
    feature_backend_identity_sha256: str
    feature_checkpoint_sha256: str
    feature_layer: int
    feature_hidden_dim: int
    feature_encoder_runtime_dtype: str
    feature_encoder_runtime_device: str
    feature_encoder_prefix_layers: int
    router_runtime_identity_sha256: str
    router_bank_identity_sha256: str
    router_call_ordinal: int
    reused_router_result: bool
    readout: MatchedSteeredReadout
    result_retained_tensor_bytes: int = 0
    retained_token_id_count: int = 0
    persisted_token_state_bytes: int = 0
    format: str = FAST_CAV_STAGE_RECEIPT_FORMAT
    stage_output_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_CAV_STAGE_RECEIPT_FORMAT:
            raise FastCAVFeatureSessionError("fast CAV stage format changed")
        for label in (
            "artifact_sha256",
            "question_sha256",
            "dated_question_sha256",
            "source_stage_receipt_sha256",
            "evidence_projection_sha256",
            "packet_identity_sha256",
            "feature_backend_identity_sha256",
            "feature_checkpoint_sha256",
            "router_runtime_identity_sha256",
            "router_bank_identity_sha256",
        ):
            _digest(getattr(self, label), label)
        _exact_int(self.placement_ordinal, "placement_ordinal")
        _exact_int(self.question_ordinal, "question_ordinal")
        _exact_int(self.stage_ordinal, "stage_ordinal")
        _exact_int(self.router_call_ordinal, "router_call_ordinal")
        _exact_int(self.feature_layer, "feature_layer")
        _exact_int(
            self.feature_hidden_dim,
            "feature_hidden_dim",
            minimum=1,
            maximum=FAST_CAV_MAX_HIDDEN_DIM,
        )
        _text(self.feature_encoder_runtime_dtype, "feature_encoder_runtime_dtype")
        _text(self.feature_encoder_runtime_device, "feature_encoder_runtime_device")
        prefix_layers = _exact_int(
            self.feature_encoder_prefix_layers,
            "feature_encoder_prefix_layers",
            minimum=1,
        )
        if self.feature_layer >= prefix_layers:
            raise FastCAVFeatureSessionError(
                "selected feature layer lies outside the encoder prefix"
            )
        _text(self.question_id, "question_id")
        _text(self.stage_id, "stage_id")
        if type(self.evidence_feature_row_indices) is not tuple or any(
            type(value) is not int or value < 0
            for value in self.evidence_feature_row_indices
        ):
            raise FastCAVFeatureSessionError(
                "evidence_feature_row_indices must contain exact non-negative integers"
            )
        evidence_ids = _tuple_of_exact_strings(self.evidence_ids, "evidence_ids")
        source_ids = _tuple_of_exact_strings(self.source_ids, "source_ids")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise FastCAVFeatureSessionError("evidence_ids must remain unique")
        if type(self.evidence_text_sha256s) is not tuple:
            raise FastCAVFeatureSessionError(
                "evidence_text_sha256s must be an exact tuple"
            )
        for index, value in enumerate(self.evidence_text_sha256s):
            _digest(value, f"evidence_text_sha256s[{index}]")
        width = len(evidence_ids)
        if width < 1 or not (
            len(source_ids)
            == len(self.evidence_text_sha256s)
            == len(self.evidence_feature_row_indices)
            == width
        ):
            raise FastCAVFeatureSessionError(
                "stage evidence coordinate projections disagree"
            )
        if type(self.reused_router_result) is not bool:
            raise FastCAVFeatureSessionError("reused_router_result must be an exact bool")
        if type(self.readout) is not MatchedSteeredReadout:
            raise FastCAVFeatureSessionError(
                "readout must be an exact MatchedSteeredReadout"
            )
        if self.readout.original_atom_order != evidence_ids:
            raise FastCAVFeatureSessionError(
                "readout changed the exact ordered evidence IDs"
            )
        if self.readout.hidden_dim != self.feature_hidden_dim:
            raise FastCAVFeatureSessionError(
                "readout hidden width disagrees with the feature backend"
            )
        if self.readout.result_retained_tensor_bytes != 0:
            raise FastCAVFeatureSessionError("readout retained tensors")
        _zero(self.result_retained_tensor_bytes, "result_retained_tensor_bytes")
        _zero(self.retained_token_id_count, "retained_token_id_count")
        _zero(self.persisted_token_state_bytes, "persisted_token_state_bytes")
        expected = identity_sha256(_stage_receipt_payload(self))
        if self.stage_output_sha256:
            if _digest(self.stage_output_sha256, "stage_output_sha256") != expected:
                raise FastCAVFeatureSessionError(
                    "stage_output_sha256 does not match stage receipt contents"
                )
        else:
            object.__setattr__(self, "stage_output_sha256", expected)


def _session_receipt_payload(
    receipt: "FastCAVFeatureSessionReceipt",
) -> dict[str, Any]:
    return {
        "format": receipt.format,
        "artifact_sha256": receipt.artifact_sha256,
        "feature_backend_format": receipt.feature_backend_format,
        "feature_backend_identity_sha256": receipt.feature_backend_identity_sha256,
        "feature_checkpoint_sha256": receipt.feature_checkpoint_sha256,
        "feature_layer": receipt.feature_layer,
        "feature_hidden_dim": receipt.feature_hidden_dim,
        "feature_source_dtype": receipt.feature_source_dtype,
        "feature_encoder_runtime_dtype": receipt.feature_encoder_runtime_dtype,
        "feature_encoder_runtime_device": receipt.feature_encoder_runtime_device,
        "feature_encoder_prefix_layers": receipt.feature_encoder_prefix_layers,
        "router_runtime_identity_sha256": receipt.router_runtime_identity_sha256,
        "router_bank_identity_sha256": receipt.router_bank_identity_sha256,
        "router_num_cavs": receipt.router_num_cavs,
        "stage_ids": list(receipt.stage_ids),
        "question_count": receipt.question_count,
        "stage_placement_count": receipt.stage_placement_count,
        "logical_evidence_placement_count": receipt.logical_evidence_placement_count,
        "per_question_unique_feature_row_count": (
            receipt.per_question_unique_feature_row_count
        ),
        "global_unique_evidence_text_count": (
            receipt.global_unique_evidence_text_count
        ),
        "global_unique_question_text_count": (
            receipt.global_unique_question_text_count
        ),
        "global_unique_text_count": receipt.global_unique_text_count,
        "encoder_input_projection_sha256": receipt.encoder_input_projection_sha256,
        "encoder_api_call_count": receipt.encoder_api_call_count,
        "unique_router_call_count": receipt.unique_router_call_count,
        "batch_size": receipt.batch_size,
        "stage_output_sha256s": [
            item.stage_output_sha256 for item in receipt.stage_receipts
        ],
        "result_retained_tensor_bytes": receipt.result_retained_tensor_bytes,
        "retained_token_id_count": receipt.retained_token_id_count,
        "persisted_token_state_bytes": receipt.persisted_token_state_bytes,
    }


@dataclass(frozen=True, slots=True)
class FastCAVFeatureSessionReceipt:
    """Complete one-call session result; every nested value is tensor-free."""

    artifact_sha256: str
    feature_backend_format: str
    feature_backend_identity_sha256: str
    feature_checkpoint_sha256: str
    feature_layer: int
    feature_hidden_dim: int
    feature_source_dtype: str
    feature_encoder_runtime_dtype: str
    feature_encoder_runtime_device: str
    feature_encoder_prefix_layers: int
    router_runtime_identity_sha256: str
    router_bank_identity_sha256: str
    router_num_cavs: int
    stage_ids: tuple[str, ...]
    question_count: int
    stage_placement_count: int
    logical_evidence_placement_count: int
    per_question_unique_feature_row_count: int
    global_unique_evidence_text_count: int
    global_unique_question_text_count: int
    global_unique_text_count: int
    encoder_input_projection_sha256: str
    encoder_api_call_count: int
    unique_router_call_count: int
    batch_size: int
    stage_receipts: tuple[FastCAVStageReceipt, ...]
    result_retained_tensor_bytes: int = 0
    retained_token_id_count: int = 0
    persisted_token_state_bytes: int = 0
    format: str = FAST_CAV_SESSION_RECEIPT_FORMAT
    session_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_CAV_SESSION_RECEIPT_FORMAT:
            raise FastCAVFeatureSessionError("fast CAV session format changed")
        if self.feature_backend_format != FAST_CAV_FEATURE_BACKEND_FORMAT:
            raise FastCAVFeatureSessionError("feature backend algorithm changed")
        for label in (
            "artifact_sha256",
            "feature_backend_identity_sha256",
            "feature_checkpoint_sha256",
            "router_runtime_identity_sha256",
            "router_bank_identity_sha256",
            "encoder_input_projection_sha256",
        ):
            _digest(getattr(self, label), label)
        _exact_int(self.feature_layer, "feature_layer")
        _exact_int(
            self.feature_hidden_dim,
            "feature_hidden_dim",
            minimum=1,
            maximum=FAST_CAV_MAX_HIDDEN_DIM,
        )
        if self.feature_source_dtype not in _FLOAT_DTYPES:
            raise FastCAVFeatureSessionError("feature_source_dtype is unsupported")
        _text(self.feature_encoder_runtime_dtype, "feature_encoder_runtime_dtype")
        _text(self.feature_encoder_runtime_device, "feature_encoder_runtime_device")
        prefix_layers = _exact_int(
            self.feature_encoder_prefix_layers,
            "feature_encoder_prefix_layers",
            minimum=1,
        )
        if self.feature_layer >= prefix_layers:
            raise FastCAVFeatureSessionError(
                "selected feature layer lies outside the encoder prefix"
            )
        _exact_int(self.router_num_cavs, "router_num_cavs", minimum=1, maximum=16)
        stage_ids = _tuple_of_exact_strings(self.stage_ids, "stage_ids")
        if stage_ids != STAGE_IDS:
            raise FastCAVFeatureSessionError("session changed the ordered S0-S3 stages")
        question_count = _exact_int(self.question_count, "question_count", minimum=1)
        stage_count = _exact_int(
            self.stage_placement_count,
            "stage_placement_count",
            minimum=1,
        )
        if stage_count != question_count * len(stage_ids):
            raise FastCAVFeatureSessionError(
                "stage placement count disagrees with the cumulative ladder"
            )
        for label in (
            "logical_evidence_placement_count",
            "per_question_unique_feature_row_count",
            "global_unique_evidence_text_count",
            "global_unique_question_text_count",
            "global_unique_text_count",
            "unique_router_call_count",
        ):
            _exact_int(getattr(self, label), label, minimum=1)
        if self.global_unique_text_count > (
            self.global_unique_evidence_text_count
            + self.global_unique_question_text_count
        ):
            raise FastCAVFeatureSessionError("global unique text counts disagree")
        if self.encoder_api_call_count != 1:
            raise FastCAVFeatureSessionError("encoder must be called exactly once")
        _exact_int(
            self.batch_size,
            "batch_size",
            minimum=1,
            maximum=FAST_CAV_MAX_BATCH_SIZE,
        )
        if type(self.stage_receipts) is not tuple or any(
            type(item) is not FastCAVStageReceipt for item in self.stage_receipts
        ):
            raise FastCAVFeatureSessionError(
                "stage_receipts must contain exact frozen stage receipts"
            )
        if len(self.stage_receipts) != stage_count:
            raise FastCAVFeatureSessionError("stage receipt count changed")

        observed_packets: dict[str, tuple[int, str]] = {}
        logical_rows = 0
        placements: set[tuple[str, str]] = set()
        for ordinal, item in enumerate(self.stage_receipts):
            if item.placement_ordinal != ordinal:
                raise FastCAVFeatureSessionError("stage placement order changed")
            if item.artifact_sha256 != self.artifact_sha256:
                raise FastCAVFeatureSessionError("stage changed artifact provenance")
            expected_stage_ordinal = ordinal % len(stage_ids)
            if item.stage_ordinal != expected_stage_ordinal or item.stage_id != stage_ids[
                expected_stage_ordinal
            ]:
                raise FastCAVFeatureSessionError("stage ladder order changed")
            if item.question_ordinal != ordinal // len(stage_ids):
                raise FastCAVFeatureSessionError("question placement order changed")
            if (
                item.feature_backend_identity_sha256
                != self.feature_backend_identity_sha256
                or item.feature_checkpoint_sha256 != self.feature_checkpoint_sha256
                or item.feature_layer != self.feature_layer
                or item.feature_hidden_dim != self.feature_hidden_dim
                or item.feature_encoder_runtime_dtype
                != self.feature_encoder_runtime_dtype
                or item.feature_encoder_runtime_device
                != self.feature_encoder_runtime_device
                or item.feature_encoder_prefix_layers
                != self.feature_encoder_prefix_layers
            ):
                raise FastCAVFeatureSessionError("stage changed feature provenance")
            if (
                item.router_runtime_identity_sha256
                != self.router_runtime_identity_sha256
                or item.router_bank_identity_sha256
                != self.router_bank_identity_sha256
            ):
                raise FastCAVFeatureSessionError("stage changed router provenance")
            placement = (item.question_id, item.stage_id)
            if placement in placements:
                raise FastCAVFeatureSessionError("duplicate question/stage placement")
            placements.add(placement)
            logical_rows += len(item.evidence_ids)

            seen = observed_packets.get(item.packet_identity_sha256)
            if seen is None:
                if item.reused_router_result:
                    raise FastCAVFeatureSessionError(
                        "first packet placement cannot claim router reuse"
                    )
                observed_packets[item.packet_identity_sha256] = (
                    item.router_call_ordinal,
                    item.readout.readout_sha256,
                )
            else:
                if not item.reused_router_result or seen != (
                    item.router_call_ordinal,
                    item.readout.readout_sha256,
                ):
                    raise FastCAVFeatureSessionError(
                        "reused packet changed its router/readout receipt"
                    )
        if logical_rows != self.logical_evidence_placement_count:
            raise FastCAVFeatureSessionError("logical evidence placement count changed")
        if len(observed_packets) != self.unique_router_call_count:
            raise FastCAVFeatureSessionError("unique router call count changed")
        if {value[0] for value in observed_packets.values()} != set(
            range(self.unique_router_call_count)
        ):
            raise FastCAVFeatureSessionError("router call ordinals are not contiguous")
        _zero(self.result_retained_tensor_bytes, "result_retained_tensor_bytes")
        _zero(self.retained_token_id_count, "retained_token_id_count")
        _zero(self.persisted_token_state_bytes, "persisted_token_state_bytes")
        expected = identity_sha256(_session_receipt_payload(self))
        if self.session_receipt_sha256:
            if _digest(self.session_receipt_sha256, "session_receipt_sha256") != expected:
                raise FastCAVFeatureSessionError(
                    "session_receipt_sha256 does not match session contents"
                )
        else:
            object.__setattr__(self, "session_receipt_sha256", expected)

    def stage(self, question_id: str, stage_id: str) -> FastCAVStageReceipt:
        """Return one exact question/stage receipt without materializing tensors."""

        for receipt in self.stage_receipts:
            if receipt.question_id == question_id and receipt.stage_id == stage_id:
                return receipt
        raise KeyError((question_id, stage_id))

    def question_stages(self, question_id: str) -> tuple[FastCAVStageReceipt, ...]:
        """Return the ordered S0-S3 receipts for one question."""

        selected = tuple(
            receipt
            for receipt in self.stage_receipts
            if receipt.question_id == question_id
        )
        if not selected:
            raise KeyError(question_id)
        return selected


def _validate_artifact(artifact: FastRetrievalArtifact) -> None:
    """Recheck the adapter projection without reopening its source file."""

    if type(artifact) is not FastRetrievalArtifact:
        raise TypeError("artifact must be an exact FastRetrievalArtifact")
    _digest(artifact.raw_sha256, "artifact.raw_sha256")
    if artifact.format != RETRIEVAL_FORMAT or artifact.campaign_format != CAMPAIGN_FORMAT:
        raise FastCAVFeatureSessionError("artifact format changed")
    if artifact.stage_ids != STAGE_IDS:
        raise FastCAVFeatureSessionError("artifact changed the ordered S0-S3 stages")
    for label in (
        "population_identity_sha256",
        "source_store_receipt_sha256",
        "combined_store_receipt_sha256",
        "retrieval_implementation_sha256",
        "retrieval_policy_sha256",
    ):
        _digest(getattr(artifact, label), f"artifact.{label}")
    _exact_int(artifact.transcript_tokens, "artifact.transcript_tokens", minimum=1)
    _exact_int(artifact.turn_count, "artifact.turn_count", minimum=1)
    _zero(
        artifact.retained_request_token_state_bytes,
        "artifact.retained_request_token_state_bytes",
    )
    if type(artifact.questions) is not tuple or not artifact.questions:
        raise FastCAVFeatureSessionError("artifact.questions must be a non-empty tuple")

    question_ids: set[str] = set()
    for question_ordinal, question in enumerate(artifact.questions):
        if type(question) is not FastRetrievalQuestion:
            raise FastCAVFeatureSessionError("artifact contains a foreign question view")
        if question.ordinal != question_ordinal:
            raise FastCAVFeatureSessionError("artifact question ordinal changed")
        question_id = _text(question.question_id, "question.question_id")
        if question_id in question_ids:
            raise FastCAVFeatureSessionError("artifact contains duplicate question IDs")
        question_ids.add(question_id)
        raw_question = _text(question.question, "question.question")
        dated_question = _text(question.dated_question, "question.dated_question")
        if _digest(question.question_sha256, "question.question_sha256") != (
            _quote_sha256(raw_question)
        ):
            raise FastCAVFeatureSessionError("raw question SHA-256 changed")
        if _digest(
            question.dated_question_sha256,
            "question.dated_question_sha256",
        ) != _quote_sha256(dated_question):
            raise FastCAVFeatureSessionError("dated question SHA-256 changed")
        _digest(
            question.predecessor_receipt_sha256,
            "question.predecessor_receipt_sha256",
        )
        _digest(
            question.retrieval_receipt_sha256,
            "question.retrieval_receipt_sha256",
        )
        _zero(
            question.retained_request_token_state_bytes,
            "question.retained_request_token_state_bytes",
        )
        if type(question.question_parse_receipt) is not FastQuestionParseReceipt:
            raise FastCAVFeatureSessionError("question parse receipt type changed")
        parse = question.question_parse_receipt
        if (
            parse.source_stage_id != STAGE_IDS[-1]
            or parse.question_sha256 != question.question_sha256
            or parse.dated_question_sha256 != question.dated_question_sha256
            or parse.matching_framing_candidates != 1
        ):
            raise FastCAVFeatureSessionError("question parse receipt changed")
        _digest(parse.provider_message_sha256, "parse.provider_message_sha256")
        if type(question.final_user_message) is not FastProviderMessage:
            raise FastCAVFeatureSessionError("final user provider message type changed")
        if question.final_user_message.role != "user" or (
            _quote_sha256(question.final_user_message.content)
            != parse.provider_message_sha256
        ):
            raise FastCAVFeatureSessionError("final user provider message changed")

        if type(question.feature_rows) is not tuple or not question.feature_rows:
            raise FastCAVFeatureSessionError("question feature rows must be non-empty")
        feature_texts: list[str] = []
        for row_index, row in enumerate(question.feature_rows):
            if type(row) is not FastFeatureRow:
                raise FastCAVFeatureSessionError("question contains a foreign feature row")
            if row.question != raw_question:
                raise FastCAVFeatureSessionError("feature row changed its raw question")
            evidence_text = _text(
                row.evidence_text,
                f"question.feature_rows[{row_index}].evidence_text",
            )
            if row.row_sha256 != _feature_row_sha256(raw_question, evidence_text):
                raise FastCAVFeatureSessionError("feature row SHA-256 changed")
            feature_texts.append(evidence_text)
        if len(feature_texts) != len(set(feature_texts)):
            raise FastCAVFeatureSessionError("question feature rows are not deduplicated")

        if type(question.stages) is not tuple or question.stage_ids != STAGE_IDS:
            raise FastCAVFeatureSessionError("question changed its ordered stage ladder")
        prior_evidence: tuple[FastEvidence, ...] = ()
        observed_feature_texts: list[str] = []
        observed_feature_set: set[str] = set()
        for stage_ordinal, stage in enumerate(question.stages):
            if type(stage) is not FastRetrievalStage:
                raise FastCAVFeatureSessionError("question contains a foreign stage view")
            if stage.stage_id != STAGE_IDS[stage_ordinal]:
                raise FastCAVFeatureSessionError("stage identifier order changed")
            for label in (
                "stage_receipt_sha256",
                "matched_controls_sha256",
                "evidence_projection_sha256",
                "context_sha256",
                "prompt_messages_sha256",
            ):
                _digest(getattr(stage, label), f"stage.{label}")
            if type(stage.evidence) is not tuple or not stage.evidence:
                raise FastCAVFeatureSessionError("stage evidence must be non-empty")
            if stage.evidence[: len(prior_evidence)] != prior_evidence:
                raise FastCAVFeatureSessionError(
                    "cumulative evidence is not an exact ordered prefix"
                )
            evidence_ids: list[str] = []
            for evidence_index, evidence in enumerate(stage.evidence):
                if type(evidence) is not FastEvidence:
                    raise FastCAVFeatureSessionError("stage contains foreign evidence")
                evidence_ids.append(
                    _text(evidence.evidence_id, f"stage.evidence[{evidence_index}].id")
                )
                _text(evidence.source_id, f"stage.evidence[{evidence_index}].source_id")
                _text(evidence.text, f"stage.evidence[{evidence_index}].text")
            if len(evidence_ids) != len(set(evidence_ids)):
                raise FastCAVFeatureSessionError("stage evidence IDs are not unique")
            expected_added = tuple(evidence_ids[len(prior_evidence) :])
            if stage.added_evidence_ids != expected_added:
                raise FastCAVFeatureSessionError("stage added-evidence suffix changed")
            if type(stage.feature_row_indices) is not tuple or len(
                stage.feature_row_indices
            ) != len(stage.evidence):
                raise FastCAVFeatureSessionError("stage feature projection changed")
            for evidence, feature_index in zip(
                stage.evidence,
                stage.feature_row_indices,
                strict=True,
            ):
                if type(feature_index) is not int or not 0 <= feature_index < len(
                    question.feature_rows
                ):
                    raise FastCAVFeatureSessionError("stage feature index is out of range")
                if question.feature_rows[feature_index].evidence_text != evidence.text:
                    raise FastCAVFeatureSessionError(
                        "stage feature index changed exact evidence text"
                    )
                if evidence.text not in observed_feature_set:
                    observed_feature_set.add(evidence.text)
                    observed_feature_texts.append(evidence.text)
            if type(stage.provider_messages) is not tuple or not stage.provider_messages:
                raise FastCAVFeatureSessionError("stage provider messages changed")
            for message in stage.provider_messages:
                if type(message) is not FastProviderMessage:
                    raise FastCAVFeatureSessionError("stage contains a foreign message")
                _text(message.role, "provider message role")
                _text(message.content, "provider message content")
            prior_evidence = stage.evidence
        if tuple(observed_feature_texts) != tuple(feature_texts):
            raise FastCAVFeatureSessionError(
                "feature rows changed first-observed evidence order"
            )
        final_stage = question.stages[-1]
        if not 0 <= parse.provider_message_index < len(final_stage.provider_messages):
            raise FastCAVFeatureSessionError("question parse message index changed")
        if (
            final_stage.provider_messages[parse.provider_message_index]
            != question.final_user_message
        ):
            raise FastCAVFeatureSessionError(
                "final user message no longer matches the final stage"
            )


def _encoder_identity(
    encoder: Any,
    layer: int,
) -> tuple[str, str, str, str, int]:
    checkpoint = _digest(
        getattr(encoder, "checkpoint_sha256", None),
        "encoder.checkpoint_sha256",
    )
    runtime_dtype = _text(
        getattr(encoder, "dtype_name", None),
        "encoder.dtype_name",
    )
    runtime_device = _text(str(getattr(encoder, "device", "")), "encoder.device")
    prefix_layers = _exact_int(
        getattr(encoder, "layers", None),
        "encoder.layers",
        minimum=1,
    )
    if layer >= prefix_layers:
        raise FastCAVFeatureSessionError(
            "selected layer lies outside the loaded encoder prefix"
        )
    explicit = getattr(encoder, "feature_backend_identity_sha256", None)
    if explicit is not None:
        return (
            _digest(explicit, "encoder.feature_backend_identity_sha256"),
            checkpoint,
            runtime_dtype,
            runtime_device,
            prefix_layers,
        )
    model_id = _text(getattr(encoder, "model_id", None), "encoder.model_id")
    model_revision = _text(
        getattr(encoder, "model_revision", None),
        "encoder.model_revision",
    )
    return (
        identity_sha256(
            {
                "format": FAST_CAV_FEATURE_BACKEND_FORMAT,
                "model_id": model_id,
                "model_revision": model_revision,
                "checkpoint_sha256": checkpoint,
                "layer": layer,
                "runtime_dtype": runtime_dtype,
                "runtime_device": runtime_device,
                "prefix_layers": prefix_layers,
            }
        ),
        checkpoint,
        runtime_dtype,
        runtime_device,
        prefix_layers,
    )


def _router_identity(router: Any, layer: int) -> tuple[str, str, int, int, int]:
    route_one = getattr(router, "route_one", None)
    if not callable(route_one):
        raise TypeError("router must expose callable route_one(X)")
    router_layer = _exact_int(getattr(router, "layer", None), "router.layer")
    if router_layer != layer:
        raise FastCAVFeatureSessionError("router layer disagrees with selected layer")
    runtime_sha = _digest(
        getattr(router, "runtime_identity_sha256", None),
        "router.runtime_identity_sha256",
    )
    bank_sha = _digest(
        getattr(router, "bank_identity_sha256", None),
        "router.bank_identity_sha256",
    )
    hidden_dim = _exact_int(
        getattr(router, "hidden_dim", None),
        "router.hidden_dim",
        minimum=1,
        maximum=FAST_CAV_MAX_HIDDEN_DIM,
    )
    max_atoms = _exact_int(
        getattr(router, "max_atoms", None),
        "router.max_atoms",
        minimum=1,
        maximum=64,
    )
    num_cavs = _exact_int(
        getattr(router, "num_cavs", None),
        "router.num_cavs",
        minimum=1,
        maximum=16,
    )
    return runtime_sha, bank_sha, hidden_dim, max_atoms, num_cavs


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("fast CAV feature sessions require PyTorch") from exc
    return torch


def _validate_feature_tensor(value: Any, *, torch: Any, rows: int) -> int:
    if type(value) is not torch.Tensor:
        raise TypeError("encoder output must be one exact torch.Tensor")
    if value.ndim != 2 or int(value.shape[0]) != rows:
        raise FastCAVFeatureSessionError(
            "encoder tensor must have exact shape [global_unique_text_count, D]"
        )
    hidden_dim = int(value.shape[1])
    if not 1 <= hidden_dim <= FAST_CAV_MAX_HIDDEN_DIM:
        raise MemoryError("encoder hidden width exceeds the hard D ceiling")
    if str(value.dtype) not in _FLOAT_DTYPES or not bool(value.is_floating_point()):
        raise TypeError("encoder tensor must use a supported floating-point dtype")
    if bool(value.requires_grad) or value.grad_fn is not None:
        raise RuntimeError("encoder tensor retained an autograd graph")
    if not bool(torch.isfinite(value).all().item()):
        raise FastCAVFeatureSessionError("encoder tensor contains non-finite values")
    if getattr(value, "_base", None) is not None:
        raise RuntimeError("encoder tensor is a view that may retain hidden state")
    if not bool(value.is_contiguous()) or int(value.storage_offset()) != 0:
        raise RuntimeError("encoder tensor must own one compact contiguous allocation")
    expected_bytes = int(value.numel()) * int(value.element_size())
    try:
        storage_bytes = int(value.untyped_storage().nbytes())
    except AttributeError:  # pragma: no cover - old optional torch runtimes
        storage_bytes = expected_bytes
    if storage_bytes != expected_bytes:
        raise RuntimeError("encoder tensor retains unrelated storage")
    return hidden_dim


def _validate_routed(
    routed: Any,
    *,
    torch: Any,
    rows: int,
    hidden_dim: int,
    num_cavs: int,
    dtype: Any,
    device: Any,
) -> None:
    if type(routed) is not FixedCAVForward:
        raise TypeError("router must return an exact FixedCAVForward")
    expected = (
        (routed.steered_nodes, (rows, hidden_dim), "steered_nodes"),
        (routed.extraction_attention, (num_cavs, rows), "extraction_attention"),
        (routed.reinjection_attention, (rows, num_cavs), "reinjection_attention"),
    )
    for value, shape, label in expected:
        if type(value) is not torch.Tensor:
            raise TypeError(f"routed.{label} must be an exact torch.Tensor")
        if tuple(int(item) for item in value.shape) != shape:
            raise FastCAVFeatureSessionError(f"routed.{label} has the wrong shape")
        if value.dtype != dtype or value.device != device:
            raise FastCAVFeatureSessionError(
                f"routed.{label} changed feature dtype/device"
            )
        if bool(value.requires_grad) or value.grad_fn is not None:
            raise RuntimeError(f"routed.{label} retained an autograd graph")
        if not bool(torch.isfinite(value).all().item()):
            raise FastCAVFeatureSessionError(f"routed.{label} is non-finite")


def _assert_tensor_free(value: Any, *, torch: Any, path: str = "receipt") -> None:
    if type(value) is torch.Tensor:
        raise RuntimeError(f"{path} retained a tensor")
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_tensor_free(
                getattr(value, field.name),
                torch=torch,
                path=f"{path}.{field.name}",
            )
        return
    if isinstance(value, tuple):
        for index, child in enumerate(value):
            _assert_tensor_free(child, torch=torch, path=f"{path}[{index}]")
        return
    if isinstance(value, (list, dict, set, Mapping)):
        raise RuntimeError(f"{path} contains a mutable container")


def run_fast_cav_feature_session(
    artifact: FastRetrievalArtifact,
    *,
    encoder: Any,
    router: Any,
    layer: int,
    batch_size: int = 8,
) -> FastCAVFeatureSessionReceipt:
    """Encode once, route unique cumulative packets, and return only receipts."""

    _validate_artifact(artifact)
    layer = _exact_int(layer, "layer")
    batch_size = _exact_int(
        batch_size,
        "batch_size",
        minimum=1,
        maximum=FAST_CAV_MAX_BATCH_SIZE,
    )
    encode_layers = getattr(encoder, "encode_layers", None)
    if not callable(encode_layers):
        raise TypeError("encoder must expose callable encode_layers")
    (
        feature_backend_sha,
        checkpoint_sha,
        encoder_runtime_dtype,
        encoder_runtime_device,
        encoder_prefix_layers,
    ) = _encoder_identity(encoder, layer)
    router_identity = _router_identity(router, layer)
    router_runtime_sha, router_bank_sha, router_hidden_dim, max_atoms, num_cavs = (
        router_identity
    )

    evidence_texts = {
        row.evidence_text
        for question in artifact.questions
        for row in question.feature_rows
    }
    question_texts = {question.question for question in artifact.questions}
    all_unique_texts = tuple(
        sorted(evidence_texts | question_texts, key=lambda value: (len(value), value))
    )
    row_count = len(all_unique_texts)
    if row_count < 1:
        raise FastCAVFeatureSessionError("feature input table is empty")
    if row_count > FAST_CAV_MAX_ENCODER_ROWS:
        raise MemoryError("global feature table exceeds the hard M ceiling")
    input_projection_sha = identity_sha256(
        {
            "format": "memory-condense-fast-cav-encoder-input-projection-v1",
            "ordered_text_sha256s": [
                _quote_sha256(value) for value in all_unique_texts
            ],
        }
    )

    torch = _require_torch()
    encoded: Any = None
    feature_tensor: Any = None
    index_tensor: Any = None
    node_features: Any = None
    query_vector: Any = None
    routed: Any = None
    receipt: FastCAVFeatureSessionReceipt | None = None
    packet_cache: dict[
        tuple[str, tuple[tuple[str, str, str], ...]],
        tuple[int, MatchedSteeredReadout, str],
    ] = {}
    try:
        encoded = encode_layers(
            all_unique_texts,
            layers=(layer,),
            batch_size=batch_size,
        )
        if type(encoded) is not dict or tuple(encoded.keys()) != (layer,):
            raise FastCAVFeatureSessionError(
                "encoder must return exactly one selected-layer tensor"
            )
        feature_tensor = encoded[layer]
        hidden_dim = _validate_feature_tensor(
            feature_tensor,
            torch=torch,
            rows=row_count,
        )
        if hidden_dim != router_hidden_dim:
            raise FastCAVFeatureSessionError(
                "encoder hidden width disagrees with fixed CAV router"
            )
        if _encoder_identity(encoder, layer) != (
            feature_backend_sha,
            checkpoint_sha,
            encoder_runtime_dtype,
            encoder_runtime_device,
            encoder_prefix_layers,
        ):
            raise RuntimeError("encoder identity changed during the single feature pass")
        if _router_identity(router, layer) != router_identity:
            raise RuntimeError("router identity changed before feature routing")

        text_index = {value: index for index, value in enumerate(all_unique_texts)}
        stage_receipts: list[FastCAVStageReceipt] = []
        placement_ordinal = 0
        for question in artifact.questions:
            question_index = text_index[question.question]
            for stage_ordinal, stage in enumerate(question.stages):
                evidence_ids = stage.evidence_ids
                if len(evidence_ids) > max_atoms:
                    raise MemoryError("stage evidence exceeds router max_atoms")
                source_ids = stage.source_ids
                evidence_text_sha256s = tuple(
                    _quote_sha256(value) for value in stage.exact_texts
                )
                packet_coordinates = tuple(
                    zip(
                        evidence_ids,
                        source_ids,
                        evidence_text_sha256s,
                        strict=True,
                    )
                )
                packet_key = (question.question_sha256, packet_coordinates)
                packet_sha = identity_sha256(
                    {
                        "format": "memory-condense-fast-cav-question-packet-v1",
                        "question_sha256": question.question_sha256,
                        "ordered_evidence": [
                            {
                                "evidence_id": evidence_id,
                                "source_id": source_id,
                                "text_sha256": text_sha,
                            }
                            for evidence_id, source_id, text_sha in packet_coordinates
                        ],
                    }
                )
                cached = packet_cache.get(packet_key)
                reused = cached is not None
                if cached is None:
                    router_call_ordinal = len(packet_cache)
                    readout: MatchedSteeredReadout | None = None
                    stage_indices = tuple(text_index[value] for value in stage.exact_texts)
                    try:
                        index_tensor = torch.tensor(
                            stage_indices,
                            dtype=torch.long,
                            device=feature_tensor.device,
                        )
                        node_features = feature_tensor.index_select(0, index_tensor)
                        query_vector = feature_tensor[question_index]
                        source_version = int(node_features._version)
                        routed = router.route_one(node_features)
                        if int(node_features._version) != source_version:
                            raise RuntimeError("router mutated its evidence feature input")
                        _validate_routed(
                            routed,
                            torch=torch,
                            rows=len(evidence_ids),
                            hidden_dim=hidden_dim,
                            num_cavs=num_cavs,
                            dtype=feature_tensor.dtype,
                            device=feature_tensor.device,
                        )
                        readout = matched_steered_readout(
                            atom_ids=evidence_ids,
                            node_features=node_features,
                            query_vector=query_vector,
                            routed=routed,
                            max_output_atoms=max_atoms,
                            max_hidden_dim=FAST_CAV_MAX_HIDDEN_DIM,
                        )
                    finally:
                        index_tensor = None
                        node_features = None
                        query_vector = None
                        routed = None
                    if type(readout) is not MatchedSteeredReadout:
                        raise RuntimeError("matched readout did not return its exact receipt")
                    if _router_identity(router, layer) != router_identity:
                        raise RuntimeError("router identity changed during packet routing")
                    cached = (router_call_ordinal, readout, packet_sha)
                    packet_cache[packet_key] = cached
                router_call_ordinal, readout, cached_packet_sha = cached
                if cached_packet_sha != packet_sha:
                    raise RuntimeError("packet identity collision detected")
                stage_receipts.append(
                    FastCAVStageReceipt(
                        artifact_sha256=artifact.raw_sha256,
                        placement_ordinal=placement_ordinal,
                        question_ordinal=question.ordinal,
                        question_id=question.question_id,
                        question_sha256=question.question_sha256,
                        dated_question_sha256=question.dated_question_sha256,
                        stage_ordinal=stage_ordinal,
                        stage_id=stage.stage_id,
                        source_stage_receipt_sha256=stage.stage_receipt_sha256,
                        evidence_projection_sha256=stage.evidence_projection_sha256,
                        evidence_feature_row_indices=stage.feature_row_indices,
                        evidence_ids=evidence_ids,
                        source_ids=source_ids,
                        evidence_text_sha256s=evidence_text_sha256s,
                        packet_identity_sha256=packet_sha,
                        feature_backend_identity_sha256=feature_backend_sha,
                        feature_checkpoint_sha256=checkpoint_sha,
                        feature_layer=layer,
                        feature_hidden_dim=hidden_dim,
                        feature_encoder_runtime_dtype=encoder_runtime_dtype,
                        feature_encoder_runtime_device=encoder_runtime_device,
                        feature_encoder_prefix_layers=encoder_prefix_layers,
                        router_runtime_identity_sha256=router_runtime_sha,
                        router_bank_identity_sha256=router_bank_sha,
                        router_call_ordinal=router_call_ordinal,
                        reused_router_result=reused,
                        readout=readout,
                    )
                )
                placement_ordinal += 1

        if _encoder_identity(encoder, layer) != (
            feature_backend_sha,
            checkpoint_sha,
            encoder_runtime_dtype,
            encoder_runtime_device,
            encoder_prefix_layers,
        ):
            raise RuntimeError("encoder identity changed after feature routing")
        if _router_identity(router, layer) != router_identity:
            raise RuntimeError("router identity changed after feature routing")
        receipt = FastCAVFeatureSessionReceipt(
            artifact_sha256=artifact.raw_sha256,
            feature_backend_format=FAST_CAV_FEATURE_BACKEND_FORMAT,
            feature_backend_identity_sha256=feature_backend_sha,
            feature_checkpoint_sha256=checkpoint_sha,
            feature_layer=layer,
            feature_hidden_dim=hidden_dim,
            feature_source_dtype=str(feature_tensor.dtype),
            feature_encoder_runtime_dtype=encoder_runtime_dtype,
            feature_encoder_runtime_device=encoder_runtime_device,
            feature_encoder_prefix_layers=encoder_prefix_layers,
            router_runtime_identity_sha256=router_runtime_sha,
            router_bank_identity_sha256=router_bank_sha,
            router_num_cavs=num_cavs,
            stage_ids=artifact.stage_ids,
            question_count=artifact.question_count,
            stage_placement_count=len(stage_receipts),
            logical_evidence_placement_count=artifact.logical_feature_row_count,
            per_question_unique_feature_row_count=artifact.unique_feature_row_count,
            global_unique_evidence_text_count=len(evidence_texts),
            global_unique_question_text_count=len(question_texts),
            global_unique_text_count=row_count,
            encoder_input_projection_sha256=input_projection_sha,
            encoder_api_call_count=1,
            unique_router_call_count=len(packet_cache),
            batch_size=batch_size,
            stage_receipts=tuple(stage_receipts),
        )
        _assert_tensor_free(receipt, torch=torch)
        return receipt
    finally:
        encoded = None
        feature_tensor = None
        index_tensor = None
        node_features = None
        query_vector = None
        routed = None
        packet_cache.clear()
        receipt = None


__all__ = [
    "FAST_CAV_FEATURE_BACKEND_FORMAT",
    "FAST_CAV_MAX_BATCH_SIZE",
    "FAST_CAV_MAX_ENCODER_ROWS",
    "FAST_CAV_MAX_HIDDEN_DIM",
    "FAST_CAV_SESSION_RECEIPT_FORMAT",
    "FAST_CAV_STAGE_RECEIPT_FORMAT",
    "FastCAVFeatureSessionError",
    "FastCAVFeatureSessionReceipt",
    "FastCAVStageReceipt",
    "run_fast_cav_feature_session",
]
