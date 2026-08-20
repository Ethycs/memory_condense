"""Gold-firewalled, provider-free scoring of a verified replay package.

Replay reconstruction completes before the supplied label loader is called.
Only hashes and scalar packet-reachability measurements can cross the report
boundary; benchmark answers and source IDs remain transient scorer inputs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._diffuse_base_contracts import (
    VerifiedDiffuseLongMemEvalBase,
    canonical_json_bytes,
    write_new_bytes,
)
from memory_condense.eval._diffuse_replay_packets import (
    VerifiedDiffuseReplayPacket,
)
from memory_condense.eval._diffuse_replay_contracts import (
    DiffuseLongMemEvalReplayReceipt,
)
from memory_condense.eval.diffuse_longmemeval import (
    LongMemEvalDiffuseMetrics,
    measure_longmemeval_diffuse_packet,
)
from memory_condense.eval.diffuse_longmemeval_replay import (
    REPLAY_MANIFEST_NAME,
    _verify_diffuse_longmemeval_replay_package,
)
from memory_condense.eval.reproducibility import file_sha256, implementation_sha256


POSTHOC_REPORT_FORMAT = "memory-condense-longmemeval-replay-posthoc-score-v1"
_DIGEST = r"^[0-9a-f]{64}$"
_MODES = ("fixed_interval", "lexical_embedding", "qwen_head")


class SupportsAnalysisScoringLabel(Protocol):
    """Closed one-record label view supplied only after replay verification."""

    file_sha256: str
    label_record_sha256: str
    dataset_sha256: str
    split_manifest_sha256: str
    analysis_ordered_question_ids_sha256: str
    analysis_sample_count: int
    sample_ordinal: int
    sample_id_sha256: str
    raw_record_sha256: str
    raw_record_span_sha256: str
    question_id: str
    question_id_sha256: str
    question_text_sha256: str
    question_probe_sha256: str
    gold_answer: str
    gold_answer_sha256: str
    evidence_source_ids: tuple[str, ...]
    evidence_source_ids_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedDiffuseReplayPackage:
    """A receipt plus every packet reconstructed by its exact verifier pass."""

    receipt: DiffuseLongMemEvalReplayReceipt
    manifest_file_sha256: str
    packets: tuple[VerifiedDiffuseReplayPacket, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _require_sha256(self.manifest_file_sha256, "replay manifest file SHA-256")
        expected = tuple(
            (
                arm.boundary_mode,
                query.question_ordinal,
                query.question_id_sha256,
                query.question_probe_sha256,
                query.query_receipt.receipt_sha256,
            )
            for arm in self.receipt.arms
            for query in arm.queries
        )
        observed = tuple(
            (
                packet.boundary_mode,
                packet.question_ordinal,
                packet.question_id_sha256,
                packet.question_probe_sha256,
                packet.receipt.receipt_sha256,
            )
            for packet in self.packets
        )
        if observed != expected:
            raise ValueError("reconstructed packets do not exactly cover the replay")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DiffuseReplayPosthocScoreRow(_FrozenModel):
    boundary_mode: Literal["fixed_interval", "lexical_embedding", "qwen_head"]
    question_ordinal: int = Field(ge=0)
    question_id_sha256: str = Field(pattern=_DIGEST)
    question_probe_sha256: str = Field(pattern=_DIGEST)
    retrieval_receipt_sha256: str = Field(pattern=_DIGEST)
    packet_receipt_sha256: str = Field(pattern=_DIGEST)
    context_sha256: str = Field(pattern=_DIGEST)
    prompt_messages_sha256: str = Field(pattern=_DIGEST)
    gold_answer_sha256: str = Field(pattern=_DIGEST)
    expected_source_ids_sha256: str = Field(pattern=_DIGEST)
    retrieved_source_ids_sha256: str = Field(pattern=_DIGEST)
    expected_source_count: int = Field(ge=0)
    retrieved_source_count: int = Field(ge=0)
    answer_present: bool
    best_evidence_f1: float = Field(ge=0.0, le=1.0)
    evidence_source_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    any_evidence_source: bool | None
    all_evidence_sources: bool | None
    selected_atoms: int = Field(ge=0)
    selected_bundles: int = Field(ge=0)
    source_span_hash_valid: bool
    closure_complete_claimed: bool
    closure_scope_exhaustive: bool
    hard_budget_compliant: bool
    context_token_proxy: int = Field(ge=0)
    prompt_token_proxy: int = Field(ge=0)
    prompt_workspace_token_proxy: int = Field(ge=0)

    @model_validator(mode="after")
    def _source_metric_shape(self) -> "DiffuseReplayPosthocScoreRow":
        expected = self.expected_source_count > 0
        if expected != (self.evidence_source_recall is not None) or expected != (
            self.any_evidence_source is not None
        ) or expected != (self.all_evidence_sources is not None):
            raise ValueError("source metrics disagree with expected-source presence")
        return self


class DiffuseReplayPosthocClaimBoundary(_FrozenModel):
    all_packets_reconstructed_before_label_load: Literal[True] = True
    label_loader_calls: Literal[1] = 1
    label_loader_runtime_certified: Literal[False] = False
    scorer_runtime_certified: Literal[False] = False
    independent_rerun_required: Literal[True] = True
    scorer_implementation_identity_scope: Literal[
        "all_memory_condense_python_sources"
    ] = "all_memory_condense_python_sources"
    scorer_post_label_retrieval_calls: Literal[0] = 0
    scorer_post_label_closure_calls: Literal[0] = 0
    scorer_post_label_model_calls: Literal[0] = 0
    scorer_post_label_provider_calls: Literal[0] = 0
    scorer_post_label_responder_calls: Literal[0] = 0
    scorer_post_label_judge_calls: Literal[0] = 0
    raw_gold_persisted_in_report: Literal[False] = False
    raw_source_ids_persisted_in_report: Literal[False] = False
    packet_text_persisted_in_report: Literal[False] = False
    network_transport_audit_performed: Literal[False] = False
    network_calls_proven_zero: Literal[False] = False
    answer_metric_scope: Literal[
        "squad_normalized_substring_containment_per_packet_atom"
    ] = (
        "squad_normalized_substring_containment_per_packet_atom"
    )
    evidence_metric_scope: Literal["best_packet_atom_token_f1"] = (
        "best_packet_atom_token_f1"
    )
    source_metric_scope: Literal["selected_packet_source_recall"] = (
        "selected_packet_source_recall"
    )
    qa_answer_accuracy_claimed: Literal[False] = False
    minimal_evidence_claimed: Literal[False] = False
    exhaustive_retrieval_claimed: Literal[False] = False


class DiffuseReplayPosthocScoreReport(_FrozenModel):
    format: Literal[
        "memory-condense-longmemeval-replay-posthoc-score-v1"
    ] = POSTHOC_REPORT_FORMAT
    replay_receipt_sha256: str = Field(pattern=_DIGEST)
    replay_manifest_file_sha256: str = Field(pattern=_DIGEST)
    scorer_implementation_sha256: str = Field(pattern=_DIGEST)
    runtime_binding_sha256: str = Field(pattern=_DIGEST)
    treatment_identity_sha256: str = Field(pattern=_DIGEST)
    treatment_file_sha256: str = Field(pattern=_DIGEST)
    sanitized_projection_sha256: str = Field(pattern=_DIGEST)
    dataset_sha256: str = Field(pattern=_DIGEST)
    split_manifest_sha256: str = Field(pattern=_DIGEST)
    analysis_ordered_question_ids_sha256: str = Field(pattern=_DIGEST)
    analysis_sample_count: int = Field(ge=1)
    sample_ordinal: int = Field(ge=0)
    sample_id_sha256: str = Field(pattern=_DIGEST)
    question_id_sha256: str = Field(pattern=_DIGEST)
    question_probe_sha256: str = Field(pattern=_DIGEST)
    raw_record_sha256: str = Field(pattern=_DIGEST)
    raw_record_span_sha256: str = Field(pattern=_DIGEST)
    scoring_label_file_sha256: str = Field(pattern=_DIGEST)
    scoring_label_record_sha256: str = Field(pattern=_DIGEST)
    gold_answer_sha256: str = Field(pattern=_DIGEST)
    evidence_source_ids_sha256: str = Field(pattern=_DIGEST)
    rows: tuple[DiffuseReplayPosthocScoreRow, ...]
    claims: DiffuseReplayPosthocClaimBoundary = DiffuseReplayPosthocClaimBoundary()
    receipt_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def _bind_report(self) -> "DiffuseReplayPosthocScoreReport":
        if self.sample_ordinal >= self.analysis_sample_count:
            raise ValueError("score sample ordinal is outside its analysis pool")
        if tuple(row.boundary_mode for row in self.rows) != _MODES:
            raise ValueError("score rows must contain the three canonical arms")
        if any(
            row.question_ordinal != 0
            or row.question_id_sha256 != self.question_id_sha256
            or row.question_probe_sha256 != self.question_probe_sha256
            or row.gold_answer_sha256 != self.gold_answer_sha256
            or row.expected_source_ids_sha256 != self.evidence_source_ids_sha256
            for row in self.rows
        ):
            raise ValueError("score rows disagree with the selected label")
        expected = identity_sha256(
            self.model_dump(mode="json", exclude={"receipt_sha256"})
        )
        if self.receipt_sha256 != expected:
            raise ValueError("post-hoc score report digest differs from its body")
        return self


def verify_and_reconstruct_diffuse_longmemeval_replay_package(
    path: str | Path,
    *,
    base: VerifiedDiffuseLongMemEvalBase,
    expected_runtime_binding_sha256: str,
) -> VerifiedDiffuseReplayPackage:
    """Run the authoritative verifier once and retain its exact packet outputs."""

    packets: list[VerifiedDiffuseReplayPacket] = []
    receipt = _verify_diffuse_longmemeval_replay_package(
        path,
        base=base,
        expected_runtime_binding_sha256=expected_runtime_binding_sha256,
        _packet_sink=packets,
    )
    # The verifier proved canonical equality and stability for its full pass;
    # bind a fresh physical-file read as well, rejecting an intervening change.
    manifest_payload = canonical_json_bytes(receipt.model_dump(mode="json"))
    expected_manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    manifest_file_sha256 = file_sha256(Path(path) / REPLAY_MANIFEST_NAME)
    if manifest_file_sha256 != expected_manifest_sha256:
        raise RuntimeError("replay manifest changed after packet reconstruction")
    return VerifiedDiffuseReplayPackage(
        receipt=receipt,
        manifest_file_sha256=manifest_file_sha256,
        packets=tuple(packets),
    )


def score_diffuse_longmemeval_replay_package(
    path: str | Path,
    *,
    base: VerifiedDiffuseLongMemEvalBase,
    expected_runtime_binding_sha256: str,
    label_loader: Callable[[], SupportsAnalysisScoringLabel],
) -> DiffuseReplayPosthocScoreReport:
    """Reconstruct every packet, then load one label and score in memory."""

    if not callable(label_loader):
        raise TypeError("label_loader must be callable")
    scorer_implementation = implementation_sha256()
    package = verify_and_reconstruct_diffuse_longmemeval_replay_package(
        path,
        base=base,
        expected_runtime_binding_sha256=expected_runtime_binding_sha256,
    )
    receipt = package.receipt
    if receipt.query_manifest.query_count != 1 or len(package.packets) != 3:
        raise ValueError("single-record scoring requires one query in three arms")
    packets = tuple(package.packets)

    # This is the first point at which answer/evidence labels may enter memory.
    label = label_loader()
    _validate_label_binding(label, receipt, packets)
    rows = tuple(_measure_row(packet, label) for packet in packets)
    treatment = receipt.query_manifest.treatment_identity
    values = {
        "replay_receipt_sha256": receipt.receipt_sha256,
        "replay_manifest_file_sha256": package.manifest_file_sha256,
        "scorer_implementation_sha256": scorer_implementation,
        "runtime_binding_sha256": receipt.runtime_binding.identity_sha256,
        "treatment_identity_sha256": receipt.treatment_identity_sha256,
        "treatment_file_sha256": treatment.treatment_file_sha256,
        "sanitized_projection_sha256": treatment.sanitized_projection_sha256,
        "dataset_sha256": label.dataset_sha256,
        "split_manifest_sha256": label.split_manifest_sha256,
        "analysis_ordered_question_ids_sha256": (
            label.analysis_ordered_question_ids_sha256
        ),
        "analysis_sample_count": label.analysis_sample_count,
        "sample_ordinal": label.sample_ordinal,
        "sample_id_sha256": label.sample_id_sha256,
        "question_id_sha256": label.question_id_sha256,
        "question_probe_sha256": label.question_probe_sha256,
        "raw_record_sha256": label.raw_record_sha256,
        "raw_record_span_sha256": label.raw_record_span_sha256,
        "scoring_label_file_sha256": label.file_sha256,
        "scoring_label_record_sha256": label.label_record_sha256,
        "gold_answer_sha256": label.gold_answer_sha256,
        "evidence_source_ids_sha256": label.evidence_source_ids_sha256,
        "rows": rows,
    }
    claims = DiffuseReplayPosthocClaimBoundary()
    unsigned = {
        "format": POSTHOC_REPORT_FORMAT,
        **values,
        "rows": [row.model_dump(mode="json") for row in rows],
        "claims": claims.model_dump(mode="json"),
    }
    report = DiffuseReplayPosthocScoreReport(
        **values,
        claims=claims,
        receipt_sha256=identity_sha256(unsigned),
    )
    if implementation_sha256() != scorer_implementation:
        raise RuntimeError("scorer implementation changed during measurement")
    return report


def publish_diffuse_longmemeval_posthoc_score(
    path: str | Path,
    report: DiffuseReplayPosthocScoreReport,
) -> None:
    """Publish one canonical report without creating or overwriting its path."""

    if not isinstance(report, DiffuseReplayPosthocScoreReport):
        raise TypeError("report must be a post-hoc score report")
    target = Path(path)
    write_new_bytes(target, canonical_json_bytes(report.model_dump(mode="json")))


def _validate_label_binding(
    label: SupportsAnalysisScoringLabel,
    receipt: DiffuseLongMemEvalReplayReceipt,
    packets: Sequence[VerifiedDiffuseReplayPacket],
) -> None:
    treatment = receipt.query_manifest.treatment_identity
    digest_fields = (
        "file_sha256",
        "label_record_sha256",
        "dataset_sha256",
        "split_manifest_sha256",
        "analysis_ordered_question_ids_sha256",
        "sample_id_sha256",
        "raw_record_sha256",
        "raw_record_span_sha256",
        "question_id_sha256",
        "question_text_sha256",
        "question_probe_sha256",
        "gold_answer_sha256",
        "evidence_source_ids_sha256",
    )
    for name in digest_fields:
        _require_sha256(getattr(label, name), name)
    if type(label.analysis_sample_count) is not int or (
        type(label.sample_ordinal) is not int
    ):
        raise TypeError("label population coordinates must be exact integers")
    if (
        label.analysis_sample_count != treatment.sample_count
        or label.sample_ordinal != treatment.sample_ordinal
        or label.dataset_sha256 != treatment.dataset_sha256
        or label.split_manifest_sha256 != treatment.split_manifest_sha256
        or label.analysis_ordered_question_ids_sha256
        != treatment.ordered_question_ids_sha256
        or label.sample_id_sha256 != receipt.sample_id_sha256
    ):
        raise ValueError("scoring label belongs to another treatment selection")
    coordinates = {
        (packet.question_id_sha256, packet.question_probe_sha256)
        for packet in packets
    }
    if coordinates != {(label.question_id_sha256, label.question_probe_sha256)}:
        raise ValueError("scoring label belongs to another replay question")
    if identity_sha256({"question_id": label.question_id}) != label.question_id_sha256:
        raise ValueError("scoring label question ID hash differs")
    if quote_sha256(label.gold_answer) != label.gold_answer_sha256:
        raise ValueError("scoring label gold-answer hash differs")
    sources = tuple(label.evidence_source_ids)
    if identity_sha256(list(sources)) != label.evidence_source_ids_sha256:
        raise ValueError("scoring label evidence-source hash differs")


def _measure_row(
    packet: VerifiedDiffuseReplayPacket,
    label: SupportsAnalysisScoringLabel,
) -> DiffuseReplayPosthocScoreRow:
    metrics = measure_longmemeval_diffuse_packet(
        packet,
        question_id=label.question_id,
        gold_answer=label.gold_answer,
        evidence_source_ids=label.evidence_source_ids,
        hydrate_span=packet.hydrate_span,
    )
    if tuple(metrics.expected_source_ids) != tuple(label.evidence_source_ids):
        raise ValueError("measurement normalized the pinned source labels")
    return _row_from_metrics(packet, label, metrics)


def _row_from_metrics(
    packet: VerifiedDiffuseReplayPacket,
    label: SupportsAnalysisScoringLabel,
    metrics: LongMemEvalDiffuseMetrics,
) -> DiffuseReplayPosthocScoreRow:
    return DiffuseReplayPosthocScoreRow(
        boundary_mode=packet.boundary_mode,
        question_ordinal=packet.question_ordinal,
        question_id_sha256=packet.question_id_sha256,
        question_probe_sha256=packet.question_probe_sha256,
        retrieval_receipt_sha256=metrics.retrieval_receipt_sha256,
        packet_receipt_sha256=packet.packet.receipt.receipt_sha256,
        context_sha256=packet.packet.receipt.context_sha256,
        prompt_messages_sha256=packet.receipt.prompt_messages_sha256,
        gold_answer_sha256=label.gold_answer_sha256,
        expected_source_ids_sha256=label.evidence_source_ids_sha256,
        retrieved_source_ids_sha256=identity_sha256(
            list(metrics.retrieved_source_ids)
        ),
        expected_source_count=len(metrics.expected_source_ids),
        retrieved_source_count=len(metrics.retrieved_source_ids),
        answer_present=metrics.answer_present,
        best_evidence_f1=metrics.best_evidence_f1,
        evidence_source_recall=metrics.evidence_source_recall,
        any_evidence_source=metrics.any_evidence_source,
        all_evidence_sources=metrics.all_evidence_sources,
        selected_atoms=metrics.selected_atoms,
        selected_bundles=metrics.selected_bundles,
        source_span_hash_valid=metrics.source_span_hash_valid,
        closure_complete_claimed=metrics.closure_complete_claimed,
        closure_scope_exhaustive=metrics.closure_scope_exhaustive,
        hard_budget_compliant=metrics.hard_budget_compliant,
        context_token_proxy=metrics.context_token_proxy,
        prompt_token_proxy=metrics.prompt_token_proxy,
        prompt_workspace_token_proxy=metrics.prompt_workspace_token_proxy,
    )


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


__all__ = [
    "POSTHOC_REPORT_FORMAT",
    "DiffuseReplayPosthocClaimBoundary",
    "DiffuseReplayPosthocScoreReport",
    "DiffuseReplayPosthocScoreRow",
    "SupportsAnalysisScoringLabel",
    "VerifiedDiffuseReplayPackage",
    "publish_diffuse_longmemeval_posthoc_score",
    "score_diffuse_longmemeval_replay_package",
    "verify_and_reconstruct_diffuse_longmemeval_replay_package",
]
