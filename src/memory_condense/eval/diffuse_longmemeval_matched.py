"""Receipts and validation for matched diffuse LongMemEval arms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval.diffuse_compilation import BoundaryMode

if TYPE_CHECKING:
    from memory_condense.eval.diffuse_longmemeval_analysis import (
        DiffuseLongMemEvalRetrievalPhase,
        ExactLegacyDiffuseInputs,
    )


DIFFUSE_MATCHED_PROBE_FORMAT = (
    "memory-condense-longmemeval-diffuse-matched-probe-v1"
)
DIFFUSE_MATCHED_SUITE_FORMAT = (
    "memory-condense-longmemeval-diffuse-matched-suite-v1"
)
MATCHED_BOUNDARY_MODES: tuple[BoundaryMode, ...] = (
    "fixed_interval",
    "lexical_embedding",
    "qwen_head",
)


def _digest(value: object, label: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _matched_source_scope_sha256(
    inputs: ExactLegacyDiffuseInputs,
) -> str | None:
    """Bind source routing while excluding arm-specific graph coordinates."""

    scope = inputs.candidates.source_candidate_scope
    if scope is None:
        return None
    return identity_sha256(
        {
            "source_revision": scope.source_revision,
            "source_content_sha256": scope.source_content_sha256,
            "query_sha256": scope.query_sha256,
            "router_policy_sha256": scope.router_policy_sha256,
            "universe_source_ids": list(scope.universe_source_ids),
            "candidates": [
                item.identity_payload() for item in scope.candidates
            ],
            "truncated_source_ids": list(scope.truncated_source_ids),
            "universe_enumerated": scope.universe_enumerated,
        }
    )


@dataclass(frozen=True, slots=True)
class DiffuseLongMemEvalMatchedProbeReceipt:
    """Shared, arm-independent retrieval inputs for one matched probe."""

    question_id: str
    question_probe_sha256: str
    retrieval_query_sha256: str
    retrieval_policy_sha256: str
    anchor_sequence_sha256: str
    anchor_chunk_ids: tuple[str, ...]
    source_candidate_sequence_sha256: str
    source_candidate_ids: tuple[str, ...]
    source_scope_identity_sha256: str
    legacy_input_provider_identity_sha256: str
    representative_linker_identity_sha256: str
    representative_policy_factory_identity_sha256: str
    representative_policy_controls_sha256: str
    episode_policy_sha256: str
    closure_policy_sha256: str
    format: str = DIFFUSE_MATCHED_PROBE_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != DIFFUSE_MATCHED_PROBE_FORMAT:
            raise ValueError("unsupported matched diffuse probe format")
        if not str(self.question_id).strip():
            raise ValueError("question_id must be non-empty")
        for name in (
            "question_probe_sha256",
            "retrieval_query_sha256",
            "retrieval_policy_sha256",
            "anchor_sequence_sha256",
            "source_candidate_sequence_sha256",
            "source_scope_identity_sha256",
            "legacy_input_provider_identity_sha256",
            "representative_linker_identity_sha256",
            "representative_policy_factory_identity_sha256",
            "representative_policy_controls_sha256",
            "episode_policy_sha256",
            "closure_policy_sha256",
        ):
            _digest(getattr(self, name), name)
        object.__setattr__(self, "anchor_chunk_ids", tuple(self.anchor_chunk_ids))
        object.__setattr__(
            self,
            "source_candidate_ids",
            tuple(self.source_candidate_ids),
        )
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("matched diffuse probe receipt does not match")
        object.__setattr__(self, "receipt_sha256", expected)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "receipt_sha256"
        }
        payload["anchor_chunk_ids"] = list(self.anchor_chunk_ids)
        payload["source_candidate_ids"] = list(self.source_candidate_ids)
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True, slots=True)
class DiffuseLongMemEvalMatchedSuiteReceipt:
    """Proof that three whole-pipeline arms form one matched comparison."""

    sample_id: str
    corpus_sha256: str
    deterministic_turn_ids_sha256: str
    evaluation_policy_sha256: str
    matched_controls_sha256: str
    pipeline_modes: tuple[BoundaryMode, ...]
    pipeline_arm_sha256s: tuple[str, ...]
    compilation_receipt_sha256s: tuple[str, ...]
    retrieval_phase_receipt_sha256s: tuple[str, ...]
    probes: tuple[DiffuseLongMemEvalMatchedProbeReceipt, ...]
    qwen_source_signal_receipt_sha256s: tuple[str, ...]
    qwen_owned_representative_runtime: bool
    zero_returned_transformer_state: bool
    zero_persisted_transformer_state: bool
    format: str = DIFFUSE_MATCHED_SUITE_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != DIFFUSE_MATCHED_SUITE_FORMAT:
            raise ValueError("unsupported matched diffuse suite format")
        if not str(self.sample_id).strip():
            raise ValueError("sample_id must be non-empty")
        for name in (
            "corpus_sha256",
            "deterministic_turn_ids_sha256",
            "evaluation_policy_sha256",
            "matched_controls_sha256",
        ):
            _digest(getattr(self, name), name)
        modes = tuple(self.pipeline_modes)
        if modes != MATCHED_BOUNDARY_MODES:
            raise ValueError("matched suite requires the three canonical arms")
        object.__setattr__(self, "pipeline_modes", modes)
        for name in (
            "pipeline_arm_sha256s",
            "compilation_receipt_sha256s",
            "retrieval_phase_receipt_sha256s",
            "qwen_source_signal_receipt_sha256s",
        ):
            values = tuple(getattr(self, name))
            for index, value in enumerate(values):
                _digest(value, f"{name}[{index}]")
            object.__setattr__(self, name, values)
        if not self.qwen_source_signal_receipt_sha256s:
            raise ValueError("matched suite requires Qwen source signal receipts")
        if len(self.pipeline_arm_sha256s) != len(MATCHED_BOUNDARY_MODES):
            raise ValueError("matched suite requires one identity per arm")
        if len(set(self.pipeline_arm_sha256s)) != len(MATCHED_BOUNDARY_MODES):
            raise ValueError("segmentation pipeline identities must be distinct")
        if len(self.compilation_receipt_sha256s) != len(
            MATCHED_BOUNDARY_MODES
        ) or len(self.retrieval_phase_receipt_sha256s) != len(
            MATCHED_BOUNDARY_MODES
        ):
            raise ValueError("matched suite receipts must cover every arm")
        probes = tuple(self.probes)
        if not probes:
            raise ValueError("matched suite requires at least one probe")
        if len({item.question_id for item in probes}) != len(probes):
            raise ValueError("matched suite probes must be unique")
        object.__setattr__(self, "probes", probes)
        for name in (
            "qwen_owned_representative_runtime",
            "zero_returned_transformer_state",
            "zero_persisted_transformer_state",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"{name} must be certified true")
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("matched diffuse suite receipt does not match")
        object.__setattr__(self, "receipt_sha256", expected)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "sample_id": self.sample_id,
            "corpus_sha256": self.corpus_sha256,
            "deterministic_turn_ids_sha256": self.deterministic_turn_ids_sha256,
            "evaluation_policy_sha256": self.evaluation_policy_sha256,
            "matched_controls_sha256": self.matched_controls_sha256,
            "pipeline_modes": list(self.pipeline_modes),
            "pipeline_arm_sha256s": list(self.pipeline_arm_sha256s),
            "compilation_receipt_sha256s": list(
                self.compilation_receipt_sha256s
            ),
            "retrieval_phase_receipt_sha256s": list(
                self.retrieval_phase_receipt_sha256s
            ),
            "probes": [item.identity_payload() for item in self.probes],
            "qwen_source_signal_receipt_sha256s": list(
                self.qwen_source_signal_receipt_sha256s
            ),
            "qwen_owned_representative_runtime": (
                self.qwen_owned_representative_runtime
            ),
            "zero_returned_transformer_state": (
                self.zero_returned_transformer_state
            ),
            "zero_persisted_transformer_state": (
                self.zero_persisted_transformer_state
            ),
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


def validate_matched_diffuse_retrieval_phases(
    phases: Sequence[DiffuseLongMemEvalRetrievalPhase],
) -> DiffuseLongMemEvalMatchedSuiteReceipt:
    """Validate and seal one honest three-arm, gold-blind comparison."""

    from memory_condense.eval.diffuse_longmemeval_analysis import (
        DiffuseLongMemEvalRetrievalPhase,
    )

    supplied = tuple(phases)
    if len(supplied) != len(MATCHED_BOUNDARY_MODES):
        raise ValueError("matched suite requires exactly three retrieval phases")
    if any(
        not isinstance(item, DiffuseLongMemEvalRetrievalPhase)
        for item in supplied
    ):
        raise TypeError("matched suite requires diffuse retrieval phases")
    by_mode = {
        item.arm.compilation.boundary_mode: item for item in supplied
    }
    if set(by_mode) != set(MATCHED_BOUNDARY_MODES) or len(by_mode) != len(
        supplied
    ):
        raise ValueError("matched suite requires each canonical arm exactly once")
    ordered = tuple(by_mode[mode] for mode in MATCHED_BOUNDARY_MODES)
    reference = ordered[0]

    def require_shared(attribute: str, label: str) -> object:
        values = tuple(getattr(item, attribute) for item in ordered)
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"matched suite changed {label}")
        return values[0]

    require_shared("sample_id", "sample identity")
    require_shared("corpus_sha256", "corpus identity")
    require_shared("deterministic_turn_ids", "deterministic ingest")
    require_shared("evaluation_policy_sha256", "evaluation policy")
    control_hashes = tuple(item.arm.matched_controls_sha256 for item in ordered)
    if len(set(control_hashes)) != 1:
        raise ValueError("matched suite changed downstream controls")
    arm_hashes = tuple(item.arm.arm_sha256 for item in ordered)
    if len(set(arm_hashes)) != len(ordered):
        raise ValueError("segmentation pipeline identities are not distinct")
    for phase in ordered:
        if phase.compilation.compilation_policy_sha256 != (
            phase.arm.compilation.policy_sha256
        ):
            raise ValueError(
                "compiled segmentation policy does not match its declared arm"
            )
    compilation_hashes = tuple(
        item.compilation.compilation_policy_sha256 for item in ordered
    )
    if len(set(compilation_hashes)) != len(ordered):
        raise ValueError("compilation pipeline identities are not distinct")

    source_inputs = tuple(
        tuple(
            (
                source.source_id,
                source.source_stream_sha256,
                source.content_chunks,
                source.metadata_chunks,
            )
            for source in phase.compilation.source_receipts
        )
        for phase in ordered
    )
    if any(value != source_inputs[0] for value in source_inputs[1:]):
        raise ValueError("matched suite compiled different source inputs")
    question_ids = tuple(
        item.probe.question_id for item in reference.questions
    )
    if any(
        tuple(item.probe.question_id for item in phase.questions)
        != question_ids
        for phase in ordered[1:]
    ):
        raise ValueError("matched suite changed its probe sequence")

    probe_receipts: list[DiffuseLongMemEvalMatchedProbeReceipt] = []
    for index, question_id in enumerate(question_ids):
        rows = tuple(phase.questions[index] for phase in ordered)

        def same(values: Sequence[object], label: str) -> object:
            frozen = tuple(values)
            if any(value != frozen[0] for value in frozen[1:]):
                raise ValueError(
                    f"matched probe {question_id!r} changed {label}"
                )
            return frozen[0]

        probe_sha256 = same(
            [item.probe.probe_sha256 for item in rows], "question identity"
        )
        query_sha256 = same(
            [item.legacy_inputs.receipt.query_sha256 for item in rows],
            "retrieval query",
        )
        retrieval_policy_sha256 = same(
            [
                item.legacy_inputs.receipt.retrieval_policy_sha256
                for item in rows
            ],
            "legacy retrieval policy",
        )
        anchor_sequence_sha256 = same(
            [
                item.legacy_inputs.receipt.anchor_sequence_sha256
                for item in rows
            ],
            "legacy anchor sequence",
        )
        anchor_chunk_ids = same(
            [item.legacy_inputs.receipt.anchor_chunk_ids for item in rows],
            "legacy anchor coordinates",
        )
        source_sequence_sha256 = same(
            [
                item.legacy_inputs.receipt.source_candidate_sequence_sha256
                for item in rows
            ],
            "source candidate sequence",
        )
        source_candidate_ids = same(
            [
                item.legacy_inputs.receipt.source_candidate_ids
                for item in rows
            ],
            "source candidate coordinates",
        )
        for phase, row in zip(ordered, rows, strict=True):
            scope = row.legacy_inputs.candidates.source_candidate_scope
            if scope is None:
                raise ValueError(
                    f"matched probe {question_id!r} lacks a source-scope receipt"
                )
            compiled_source_ids = tuple(
                source.source_id
                for source in phase.compilation.source_receipts
            )
            if scope.universe_source_ids != compiled_source_ids:
                raise ValueError(
                    f"matched probe {question_id!r} source universe does not "
                    "equal the compiled source universe"
                )
        scope_identities = tuple(
            _matched_source_scope_sha256(item.legacy_inputs) for item in rows
        )
        if any(value is None for value in scope_identities):
            raise ValueError(
                f"matched probe {question_id!r} lacks a source-scope receipt"
            )
        source_scope_sha256 = same(
            scope_identities, "source-scope identity"
        )
        provider_sha256 = same(
            [
                item.receipt.legacy_input_provider_identity_sha256
                for item in rows
            ],
            "legacy input provider",
        )
        linker_sha256 = same(
            [
                item.receipt.representative_linker_identity_sha256
                for item in rows
            ],
            "representative linker",
        )
        policy_factory_sha256 = same(
            [
                item.receipt.representative_policy_factory_identity_sha256
                for item in rows
            ],
            "representative policy factory",
        )
        policy_controls_sha256 = same(
            [
                item.receipt.representative_policy_controls_sha256
                for item in rows
            ],
            "representative policy controls",
        )
        if any(
            value is None
            for value in (
                linker_sha256,
                policy_factory_sha256,
                policy_controls_sha256,
            )
        ):
            raise ValueError(
                f"matched probe {question_id!r} lacks representative identities"
            )
        for phase, row in zip(ordered, rows, strict=True):
            expected_episode_policy = replace(
                phase.arm.episode,
                artifact_id=phase.compilation.artifact.artifact_id,
            ).policy_sha256
            if row.retrieval.receipt.episode_policy_sha256 != (
                expected_episode_policy
            ):
                raise ValueError(
                    f"matched probe {question_id!r} has an unbound "
                    "episode retrieval policy"
                )
        episode_policy_sha256 = same(
            [phase.arm.episode.policy_sha256 for phase in ordered],
            "episode retrieval controls",
        )
        closure_policy_sha256 = same(
            [item.retrieval.receipt.closure_policy_sha256 for item in rows],
            "closure policy",
        )
        probe_receipts.append(
            DiffuseLongMemEvalMatchedProbeReceipt(
                question_id=question_id,
                question_probe_sha256=str(probe_sha256),
                retrieval_query_sha256=str(query_sha256),
                retrieval_policy_sha256=str(retrieval_policy_sha256),
                anchor_sequence_sha256=str(anchor_sequence_sha256),
                anchor_chunk_ids=tuple(anchor_chunk_ids),  # type: ignore[arg-type]
                source_candidate_sequence_sha256=str(
                    source_sequence_sha256
                ),
                source_candidate_ids=tuple(
                    source_candidate_ids  # type: ignore[arg-type]
                ),
                source_scope_identity_sha256=str(source_scope_sha256),
                legacy_input_provider_identity_sha256=str(provider_sha256),
                representative_linker_identity_sha256=str(linker_sha256),
                representative_policy_factory_identity_sha256=str(
                    policy_factory_sha256
                ),
                representative_policy_controls_sha256=str(
                    policy_controls_sha256
                ),
                episode_policy_sha256=str(episode_policy_sha256),
                closure_policy_sha256=str(closure_policy_sha256),
            )
        )

    qwen_phase = ordered[-1]
    qwen_signals = tuple(
        source.surprise_signal_receipt_sha256
        for source in qwen_phase.compilation.source_receipts
        if source.content_chunks > 0
    )
    if not qwen_signals or any(value is None for value in qwen_signals):
        raise ValueError("qwen_head arm lacks per-source signal receipts")
    returned_state_is_zero = all(
        source.returned_signal_transformer_state_bytes == 0
        for phase in ordered
        for source in phase.compilation.source_receipts
    ) and all(
        question.retrieval.receipt.packet_retained_request_token_state_bytes
        == 0
        and question.retrieval.receipt
        .representative_returned_plan_transformer_state_bytes
        == 0
        for phase in ordered
        for question in phase.questions
    )
    persisted_state_is_zero = all(
        phase.compilation.persisted_request_token_state_bytes == 0
        and all(
            question.retrieval.receipt
            .store_retained_request_token_state_bytes
            == 0
            for question in phase.questions
        )
        for phase in ordered
    )
    qwen_owned = all(
        question.retrieval.receipt.representative_runtime_binding_certified
        is True
        for question in qwen_phase.questions
    )
    return DiffuseLongMemEvalMatchedSuiteReceipt(
        sample_id=reference.sample_id,
        corpus_sha256=reference.corpus_sha256,
        deterministic_turn_ids_sha256=identity_sha256(
            list(reference.deterministic_turn_ids)
        ),
        evaluation_policy_sha256=reference.evaluation_policy_sha256,
        matched_controls_sha256=control_hashes[0],
        pipeline_modes=MATCHED_BOUNDARY_MODES,
        pipeline_arm_sha256s=arm_hashes,
        compilation_receipt_sha256s=tuple(
            item.compilation.receipt_sha256 for item in ordered
        ),
        retrieval_phase_receipt_sha256s=tuple(
            item.receipt_sha256 for item in ordered
        ),
        probes=tuple(probe_receipts),
        qwen_source_signal_receipt_sha256s=tuple(
            str(value) for value in qwen_signals
        ),
        qwen_owned_representative_runtime=qwen_owned,
        zero_returned_transformer_state=returned_state_is_zero,
        zero_persisted_transformer_state=persisted_state_is_zero,
    )


__all__ = [
    "DIFFUSE_MATCHED_PROBE_FORMAT",
    "DIFFUSE_MATCHED_SUITE_FORMAT",
    "MATCHED_BOUNDARY_MODES",
    "DiffuseLongMemEvalMatchedProbeReceipt",
    "DiffuseLongMemEvalMatchedSuiteReceipt",
    "validate_matched_diffuse_retrieval_phases",
]
