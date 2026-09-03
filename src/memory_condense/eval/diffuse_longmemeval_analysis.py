"""Fresh-store, provider-free analysis for diffuse LongMemEval retrieval.

This module owns orchestration only.  It deliberately separates the run into
two phases:

1. a sanitized :class:`GoldBlindLongMemEvalSample` is ingested, compiled, and
   retrieved without answers or evidence-source labels; then
2. the frozen packets are measured against the original benchmark questions.

Legacy anchors and source-router candidates enter through one injected seam.
The runner never derives source candidates from the final anchor set, because
doing so would make an independently missed source impossible to recover.
No responder, judge, CLI, provider, checkpoint loader, or network client is
imported here.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain._discourse_identity import _nonempty, normalize_fields
from memory_condense.domain.discourse import (
    ClosurePolicy,
    EvidenceSpan,
    identity_sha256,
    quote_sha256,
)
from memory_condense.eval._retrieval_qa_prompt import (
    RESPONDER_OUTPUT_TOKEN_RESERVE as BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
)
from memory_condense.eval._diffuse_replay_provider_identity import (
    _OPERATIONAL_PROVIDER_IDENTITY_V2_MARKER,
    build_provider_identity_v2,
)
from memory_condense.eval.diffuse_compilation import (
    DiffuseCompilationPolicy,
    DiffuseCompilationReceipt,
    compile_diffuse_artifact,
)
from memory_condense.eval.diffuse_longmemeval import (
    DiffuseEpisodicRoute,
    LongMemEvalDiffuseMetrics,
    LongMemEvalDiffuseRetrieval,
    _anchor_payload as _diffuse_anchor_payload,
    _diffuse_episodic_route,
    measure_longmemeval_diffuse_packet,
    retrieve_longmemeval_diffuse_packet,
)
from memory_condense.eval.diffuse_longmemeval_matched import (
    DIFFUSE_MATCHED_PROBE_FORMAT,
    DIFFUSE_MATCHED_SUITE_FORMAT,
    MATCHED_BOUNDARY_MODES as _MATCHED_BOUNDARY_MODES,
    DiffuseLongMemEvalMatchedProbeReceipt,
    DiffuseLongMemEvalMatchedSuiteReceipt,
    validate_matched_diffuse_retrieval_phases,
)
from memory_condense.eval.diffuse_longmemeval_inputs import (
    DETERMINISTIC_DIFFUSE_INGEST_FORMAT,
    LEGACY_DIFFUSE_INPUT_FORMAT,
    ExactLegacyDiffuseInputs,
    GoldBlindLongMemEvalQuestion,
    GoldBlindLongMemEvalSample,
    LegacyDiffuseCandidates,
    LegacyDiffuseInputReceipt,
    _assert_deterministic_sample_loaded,
    _question_probe,
    capture_legacy_diffuse_inputs,
    gold_blind_longmemeval_sample,
    ingest_gold_blind_sample_deterministically,
)
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.eval._identity import exact_int, sha256_digest
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.search.episodes import (
    EpisodeRepresentativeRetrievalPolicy,
    EpisodeRetrievalPolicy,
    NestedEpisodeLinker,
    QwenAttentionHeadSurpriseScorer,
)
from memory_condense.search.episodes.qwen_episode_signal import (
    _canonical_callable_code,
)


DIFFUSE_ANALYSIS_QUERY_FORMAT = (
    "memory-condense-longmemeval-diffuse-analysis-query-v1"
)
DIFFUSE_ANALYSIS_PHASE_FORMAT = (
    "memory-condense-longmemeval-diffuse-analysis-phase-v1"
)
DIFFUSE_ANALYSIS_FORMAT = "memory-condense-longmemeval-diffuse-analysis-v1"


class MeasurementQuestionLike(Protocol):
    """Structural post-retrieval question view used only by measurement APIs."""

    question_id: str
    question: str
    dated_question: str
    answer: str
    evidence_sources: Sequence[str]


class MeasurementSampleLike(Protocol):
    """Structural post-retrieval sample view; no raw loader is imported."""

    sample_id: str
    turns: Sequence[tuple[str, str]]
    turn_source_ids: Sequence[str | None]
    turn_created_at: Sequence[Any]
    questions: Sequence[MeasurementQuestionLike]


class LegacyDiffuseInputProvider(Protocol):
    """Gold-blind acquisition of exact legacy anchors and source candidates.

    Implementations may use the production legacy retriever plus an
    independent source router.  Only the query and frozen retrieval policy are
    supplied; benchmark answers and evidence labels are structurally absent.
    """

    def __call__(
        self,
        condenser: MemoryCondenser,
        *,
        query: str,
        retrieval: RetrievalConfig,
        artifact_id: str,
    ) -> LegacyDiffuseCandidates: ...


class FreshCondenserFactory(Protocol):
    """Construct one empty condenser; the runner owns deterministic ingest."""

    def __call__(
        self,
        data_dir: Path,
        config: EvalConfig,
    ) -> MemoryCondenser: ...


RepresentativePolicyFactory = Callable[
    [str], EpisodeRepresentativeRetrievalPolicy
]


def analysis_callable_identity_payload(
    value: object,
    label: str,
) -> dict[str, object]:
    """Bind callable code, using operational v2 only for a direct opt-in.

    A callable may expose ``analysis_identity_payload()`` to bind immutable
    instance configuration. Unmarked callables retain the historical v1
    identity exactly. A class that directly owns the private v2 marker is
    instead hashed by the harness from its actual callable code; it cannot
    author its own code digest. Unknown or inherited markers fail closed.
    """

    if not callable(value):
        raise TypeError(f"{label} must be callable")
    declared_method = getattr(value, "analysis_identity_payload", None)
    declared: Mapping[str, object] | None = None
    if declared_method is not None:
        if not callable(declared_method):
            raise TypeError(
                f"{label}.analysis_identity_payload must be callable"
            )
        raw_declared = declared_method()
        if not isinstance(raw_declared, Mapping):
            raise TypeError(
                f"{label}.analysis_identity_payload must return a mapping"
            )
        declared = dict(raw_declared)

    marker_name = "__memory_condense_operational_identity_v2__"
    missing_marker = object()
    callable_type = type(value)
    marker = inspect.getattr_static(
        callable_type,
        marker_name,
        missing_marker,
    )
    direct_marker = vars(callable_type).get(marker_name, missing_marker)
    if marker is _OPERATIONAL_PROVIDER_IDENTITY_V2_MARKER:
        if direct_marker is not _OPERATIONAL_PROVIDER_IDENTITY_V2_MARKER:
            raise TypeError(f"{label} must own its v2 identity marker directly")
        if declared is None:
            raise TypeError(f"{label} v2 identity requires a declaration")
        return build_provider_identity_v2(value, declared, label=label)
    if marker is not missing_marker:
        raise ValueError(f"{label} has an unsupported identity version marker")

    target = getattr(value, "__func__", value)
    if not inspect.isfunction(target):
        target = getattr(type(value), "__call__", target)
    implementation_type = (
        f"{type(value).__module__}.{type(value).__qualname__}"
    )
    implementation = (
        f"{getattr(target, '__module__', type(value).__module__)}."
        f"{getattr(target, '__qualname__', type(value).__qualname__)}"
    )
    code = getattr(target, "__code__", None)
    code_sha256 = None
    if code is not None:
        canonical = _canonical_callable_code(
            code,
            stable_filename=implementation,
        )
        code_sha256 = hashlib.sha256(canonical).hexdigest()
    return {
        "implementation_type": implementation_type,
        "implementation": implementation,
        "python_code_sha256": code_sha256,
        "declared_identity": declared,
    }


def _callable_implementation_sha256(value: object, label: str) -> str:
    return identity_sha256(analysis_callable_identity_payload(value, label))


def _representative_linker_identity_sha256(
    linker: NestedEpisodeLinker | None,
) -> str | None:
    if linker is None:
        return None
    # This is the same provider-free observer used by the representative plan.
    # Importing it does not construct or load a model.
    from memory_condense.search.episodes.representative_retrieval import (
        _linker_identity,
    )

    return identity_sha256(_linker_identity(linker))


def _representative_policy_controls_sha256(
    policy: EpisodeRepresentativeRetrievalPolicy,
) -> str:
    """Normalize the arm-specific artifact coordinate out of one policy."""

    if not isinstance(policy, EpisodeRepresentativeRetrievalPolicy):
        raise TypeError(
            "representative policy factory must return "
            "EpisodeRepresentativeRetrievalPolicy"
        )
    return replace(policy, artifact_id="matched-artifact").policy_sha256


@dataclass(frozen=True, slots=True)
class DiffuseLongMemEvalArm:
    """One segmentation-pipeline arm with downstream controls held frozen.

    Fixed interval, lexical/embedding change, and Qwen-head segmentation are
    complete pipeline arms, not a single-factor surprise-signal ablation.
    """

    arm_id: str
    compilation: DiffuseCompilationPolicy
    episode: EpisodeRetrievalPolicy = field(
        default_factory=EpisodeRetrievalPolicy
    )
    closure: ClosurePolicy = field(default_factory=ClosurePolicy)
    max_context_tokens: int = 4096
    responder_output_token_reserve: int = (
        BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
    )
    require_owned_representative_runtime: bool = False

    def __post_init__(self) -> None:
        normalize_fields(self, arm_id=_nonempty)
        if self.arm_id != self.compilation.boundary_mode:
            raise ValueError(
                "arm_id must equal the declared compilation boundary_mode"
            )
        if self.episode.artifact_id is not None:
            raise ValueError(
                "analysis episode policy must be artifact-agnostic"
            )
        if type(self.require_owned_representative_runtime) is not bool:
            raise ValueError(
                "require_owned_representative_runtime must be boolean"
            )
        if (
            self.compilation.boundary_mode == "qwen_head"
            and not self.require_owned_representative_runtime
        ):
            raise ValueError(
                "qwen_head analysis requires an owned representative runtime"
            )
        object.__setattr__(
            self,
            "max_context_tokens",
            exact_int(self.max_context_tokens, "max_context_tokens", minimum=1),
        )
        object.__setattr__(
            self,
            "responder_output_token_reserve",
            exact_int(
                self.responder_output_token_reserve,
                "responder_output_token_reserve",
                minimum=0,
            ),
        )

    def identity_payload(self, *, include_boundary: bool = True) -> dict[str, Any]:
        compilation = self.compilation.identity_payload()
        if not include_boundary:
            compilation = dict(compilation)
            compilation.pop("boundary_mode")
        return {
            "arm_id": self.arm_id if include_boundary else None,
            "compilation": compilation,
            "episode_policy_sha256": self.episode.policy_sha256,
            "closure_policy_sha256": self.closure.policy_sha256,
            "max_context_tokens": self.max_context_tokens,
            "responder_output_token_reserve": (
                self.responder_output_token_reserve
            ),
            "require_owned_representative_runtime": (
                self.require_owned_representative_runtime
            ),
        }

    @property
    def arm_sha256(self) -> str:
        return identity_sha256(self.identity_payload())

    @property
    def matched_controls_sha256(self) -> str:
        return identity_sha256(self.identity_payload(include_boundary=False))


def matched_diffuse_boundary_arms(
    reference: DiffuseLongMemEvalArm,
) -> tuple[DiffuseLongMemEvalArm, ...]:
    """Build three segmentation pipelines with shared downstream controls."""

    arms = tuple(
        replace(
            reference,
            arm_id=mode,
            # Representative retrieval is a downstream control.  Requiring
            # the same owned runtime in every arm keeps the comparison matched
            # and necessarily certifies the Qwen-head arm.
            require_owned_representative_runtime=True,
            compilation=replace(
                reference.compilation,
                boundary_mode=mode,
            ),
        )
        for mode in _MATCHED_BOUNDARY_MODES
    )
    if len({item.matched_controls_sha256 for item in arms}) != 1:
        raise RuntimeError("matched boundary arms changed a shared control")
    return arms


def _evaluation_policy_sha256(config: EvalConfig) -> str:
    return identity_sha256(
        {
            "chunker": config.chunker.model_dump(mode="json"),
            "retrieval": config.retrieval.model_dump(mode="json"),
            "max_prompt_tokens": config.max_prompt_tokens,
        }
    )


@dataclass(frozen=True, slots=True)
class DiffuseLongMemEvalAnalysisQueryReceipt(SealedIdentity):
    _SEAL_MISMATCH = "diffuse analysis query receipt does not match"

    corpus_sha256: str
    question_probe_sha256: str
    analysis_arm_sha256: str
    matched_controls_sha256: str
    evaluation_policy_sha256: str
    legacy_input_provider_identity_sha256: str
    representative_linker_identity_sha256: str | None
    representative_policy_factory_identity_sha256: str | None
    representative_policy_sha256: str | None
    representative_policy_controls_sha256: str | None
    compilation_receipt_sha256: str
    legacy_input_receipt_sha256: str
    diffuse_query_receipt_sha256: str
    artifact_id: str
    snapshot_sha256: str
    format: str = DIFFUSE_ANALYSIS_QUERY_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != DIFFUSE_ANALYSIS_QUERY_FORMAT:
            raise ValueError("unsupported diffuse analysis query format")
        for name in (
            "corpus_sha256",
            "question_probe_sha256",
            "analysis_arm_sha256",
            "matched_controls_sha256",
            "evaluation_policy_sha256",
            "legacy_input_provider_identity_sha256",
            "compilation_receipt_sha256",
            "legacy_input_receipt_sha256",
            "diffuse_query_receipt_sha256",
            "snapshot_sha256",
        ):
            sha256_digest(getattr(self, name), name)
        representative_fields = (
            "representative_linker_identity_sha256",
            "representative_policy_factory_identity_sha256",
            "representative_policy_sha256",
            "representative_policy_controls_sha256",
        )
        present = tuple(
            getattr(self, name) is not None for name in representative_fields
        )
        if any(present) and not all(present):
            raise ValueError(
                "representative call-time identities must be all present or absent"
            )
        for name in representative_fields:
            value = getattr(self, name)
            if value is not None:
                sha256_digest(value, name)
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty")
        self._seal()


@dataclass(frozen=True, slots=True)
class DiffuseLongMemEvalGoldBlindQuery:
    probe: GoldBlindLongMemEvalQuestion
    legacy_inputs: ExactLegacyDiffuseInputs
    retrieval: LongMemEvalDiffuseRetrieval
    receipt: DiffuseLongMemEvalAnalysisQueryReceipt

    def __post_init__(self) -> None:
        if self.receipt.question_probe_sha256 != self.probe.probe_sha256:
            raise ValueError("query receipt belongs to another question")
        if self.receipt.legacy_input_receipt_sha256 != (
            self.legacy_inputs.receipt.receipt_sha256
        ):
            raise ValueError("query receipt does not bind legacy inputs")
        if self.receipt.diffuse_query_receipt_sha256 != (
            self.retrieval.receipt.receipt_sha256
        ):
            raise ValueError("query receipt does not bind diffuse retrieval")
        if self.legacy_inputs.receipt.artifact_id != self.receipt.artifact_id:
            raise ValueError("legacy inputs belong to another artifact")
        if self.legacy_inputs.receipt.query_sha256 != (
            self.retrieval.receipt.retrieval_query_sha256
        ):
            raise ValueError("legacy inputs belong to another retrieval query")
        exact_retrieval_anchor_sha256 = identity_sha256(
            tuple(
                _diffuse_anchor_payload(item)
                for item in self.legacy_inputs.candidates.anchors
            )
        )
        if exact_retrieval_anchor_sha256 != (
            self.retrieval.receipt.anchor_sequence_sha256
        ):
            raise ValueError("diffuse retrieval changed the exact legacy anchors")
        if self.legacy_inputs.receipt.anchor_chunk_ids != (
            self.retrieval.receipt.input_anchor_chunk_ids
        ):
            raise ValueError("diffuse retrieval changed legacy anchor coordinates")
        if self.receipt.artifact_id != self.retrieval.receipt.artifact_id:
            raise ValueError("query receipt belongs to another artifact")
        if self.receipt.snapshot_sha256 != self.retrieval.receipt.snapshot_sha256:
            raise ValueError("query receipt belongs to another snapshot")
        representative = self.retrieval.representative_expansion
        if representative is None:
            if self.receipt.representative_linker_identity_sha256 is not None:
                raise ValueError(
                    "absent representative retrieval cannot carry runtime identities"
                )
            if (
                self.legacy_inputs.receipt
                .source_candidate_scope_receipt_sha256
                is not None
            ):
                raise ValueError(
                    "source-scope inputs were not used by representative retrieval"
                )
        else:
            if self.receipt.representative_linker_identity_sha256 != (
                representative.linker_identity_sha256
            ):
                raise ValueError(
                    "query receipt does not bind the representative linker"
                )
            if self.receipt.representative_policy_sha256 != (
                representative.policy_sha256
            ):
                raise ValueError(
                    "query receipt does not bind the representative policy"
                )
            if (
                self.legacy_inputs.receipt
                .source_candidate_scope_receipt_sha256
                != representative.source_scope_receipt_sha256
            ):
                raise ValueError(
                    "representative retrieval changed the exact source scope"
                )


@dataclass(frozen=True, slots=True)
class DiffuseLongMemEvalRetrievalPhase(SealedIdentity):
    _SEAL_MISMATCH = "diffuse retrieval phase receipt does not match"

    sample_id: str
    corpus_sha256: str
    deterministic_turn_ids: tuple[str, ...]
    arm: DiffuseLongMemEvalArm
    evaluation_policy_sha256: str
    compilation: DiffuseCompilationReceipt
    questions: tuple[DiffuseLongMemEvalGoldBlindQuery, ...]
    format: str = DIFFUSE_ANALYSIS_PHASE_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != DIFFUSE_ANALYSIS_PHASE_FORMAT:
            raise ValueError("unsupported diffuse retrieval phase format")
        sha256_digest(self.corpus_sha256, "corpus_sha256")
        sha256_digest(self.evaluation_policy_sha256, "evaluation_policy_sha256")
        if not str(self.sample_id).strip():
            raise ValueError("sample_id must be non-empty")
        turn_ids = tuple(str(value).strip() for value in self.deterministic_turn_ids)
        if not turn_ids or any(not value for value in turn_ids):
            raise ValueError("deterministic turn IDs must be non-empty")
        if len(set(turn_ids)) != len(turn_ids):
            raise ValueError("deterministic turn IDs must be unique")
        questions = tuple(self.questions)
        if not questions:
            raise ValueError("retrieval phase requires at least one question")
        if len({item.probe.question_id for item in questions}) != len(questions):
            raise ValueError("retrieval phase question IDs must be unique")
        if self.compilation.artifact.policy_sha256 != (
            self.compilation.policy_sha256
        ):
            raise ValueError("compilation receipt policy identity changed")
        if self.compilation.artifact.metadata.get("boundary_policy_id") != (
            self.arm.compilation.boundary_mode
        ):
            raise ValueError("compilation receipt belongs to another boundary arm")
        if self.arm.compilation.boundary_mode == "qwen_head":
            if not self.arm.require_owned_representative_runtime:
                raise ValueError(
                    "qwen_head phase lacks the owned representative requirement"
                )
            for source in self.compilation.source_receipts:
                if source.returned_signal_transformer_state_bytes != 0:
                    raise ValueError(
                        "qwen_head source signal has no zero-state attestation"
                    )
                if (
                    source.content_chunks > 0
                    and source.surprise_signal_receipt_sha256 is None
                ):
                    raise ValueError(
                        "qwen_head content source lacks a Qwen signal receipt"
                    )
        for item in questions:
            if item.receipt.corpus_sha256 != self.corpus_sha256:
                raise ValueError("query receipt belongs to another corpus")
            if item.receipt.analysis_arm_sha256 != self.arm.arm_sha256:
                raise ValueError("query receipt belongs to another arm")
            if item.receipt.matched_controls_sha256 != (
                self.arm.matched_controls_sha256
            ):
                raise ValueError("query receipt changed a matched control")
            if item.receipt.evaluation_policy_sha256 != (
                self.evaluation_policy_sha256
            ):
                raise ValueError("query receipt changed the evaluation policy")
            if item.receipt.compilation_receipt_sha256 != (
                self.compilation.receipt_sha256
            ):
                raise ValueError("query receipt belongs to another compilation")
            if item.receipt.artifact_id != self.compilation.artifact.artifact_id:
                raise ValueError("query receipt belongs to another artifact")
            if item.receipt.snapshot_sha256 != (
                self.compilation.final_snapshot.snapshot_sha256
            ):
                raise ValueError("query receipt belongs to another snapshot")
            if self.arm.compilation.boundary_mode == "qwen_head":
                representative = item.retrieval.representative_expansion
                if representative is None:
                    raise ValueError(
                        "qwen_head query lacks representative retrieval"
                    )
                if not representative.runtime_binding_certified:
                    raise ValueError(
                        "qwen_head representative runtime is not owned"
                    )
                if representative.returned_plan_transformer_state_bytes != 0:
                    raise ValueError(
                        "qwen_head representative plan retained transformer state"
                    )
                if (
                    item.retrieval.receipt.representative_runtime_binding_certified
                    is not True
                    or item.retrieval.receipt
                    .representative_returned_plan_transformer_state_bytes
                    != 0
                ):
                    raise ValueError(
                        "qwen_head query receipt lacks owned zero-state retrieval"
                    )
        object.__setattr__(self, "questions", questions)
        object.__setattr__(self, "deterministic_turn_ids", turn_ids)
        self._seal()

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "sample_id": self.sample_id,
            "corpus_sha256": self.corpus_sha256,
            "deterministic_ingest_format": DETERMINISTIC_DIFFUSE_INGEST_FORMAT,
            "deterministic_turn_ids": list(self.deterministic_turn_ids),
            "analysis_arm_sha256": self.arm.arm_sha256,
            "matched_controls_sha256": self.arm.matched_controls_sha256,
            "evaluation_policy_sha256": self.evaluation_policy_sha256,
            "compilation_receipt_sha256": self.compilation.receipt_sha256,
            "question_receipt_sha256s": [
                item.receipt.receipt_sha256 for item in self.questions
            ],
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload



@dataclass(frozen=True, slots=True)
class DiffuseLongMemEvalMeasuredQuestion(SealedIdentity):
    _SEAL_MISMATCH = "diffuse measurement receipt does not match"

    gold_blind: DiffuseLongMemEvalGoldBlindQuery
    metrics: LongMemEvalDiffuseMetrics
    gold_answer_sha256: str
    evidence_sources_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        sha256_digest(self.gold_answer_sha256, "gold_answer_sha256")
        sha256_digest(self.evidence_sources_sha256, "evidence_sources_sha256")
        if self.metrics.question_id != self.gold_blind.probe.question_id:
            raise ValueError("metrics belong to another question")
        if self.metrics.retrieval_receipt_sha256 != (
            self.gold_blind.retrieval.receipt.receipt_sha256
        ):
            raise ValueError("metrics belong to another diffuse retrieval")
        self._seal()

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "query_receipt_sha256": self.gold_blind.receipt.receipt_sha256,
            "gold_answer_sha256": self.gold_answer_sha256,
            "evidence_sources_sha256": self.evidence_sources_sha256,
            "metrics": asdict(self.metrics),
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True, slots=True)
class DiffuseLongMemEvalAnalysis(SealedIdentity):
    _SEAL_MISMATCH = "diffuse analysis receipt does not match"

    retrieval_phase: DiffuseLongMemEvalRetrievalPhase
    questions: tuple[DiffuseLongMemEvalMeasuredQuestion, ...]
    format: str = DIFFUSE_ANALYSIS_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != DIFFUSE_ANALYSIS_FORMAT:
            raise ValueError("unsupported diffuse analysis format")
        questions = tuple(self.questions)
        if tuple(item.gold_blind for item in questions) != (
            self.retrieval_phase.questions
        ):
            raise ValueError("measurements do not cover the frozen phase exactly")
        object.__setattr__(self, "questions", questions)
        self._seal()

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "retrieval_phase_receipt_sha256": (
                self.retrieval_phase.receipt_sha256
            ),
            "measurement_receipt_sha256s": [
                item.receipt_sha256 for item in self.questions
            ],
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


def _retrieve_diffuse_longmemeval_sample_with_route(
    condenser: MemoryCondenser,
    sample: GoldBlindLongMemEvalSample,
    *,
    config: EvalConfig,
    arm: DiffuseLongMemEvalArm,
    legacy_input_provider: LegacyDiffuseInputProvider,
    qwen_scorer: QwenAttentionHeadSurpriseScorer | None = None,
    embedding_identity: Mapping[str, object] | None = None,
    representative_linker: NestedEpisodeLinker | None = None,
    representative_policy_factory: RepresentativePolicyFactory | None = None,
    episodic_route: DiffuseEpisodicRoute,
    _packet_retriever: Any = retrieve_longmemeval_diffuse_packet,
    _packet_retriever_guard: Callable[[], None] | None = None,
) -> DiffuseLongMemEvalRetrievalPhase:
    """Route-aware implementation behind the frozen legacy public facade."""

    if not isinstance(condenser, MemoryCondenser):
        raise TypeError("diffuse analysis requires a MemoryCondenser")
    active_episodic_route = _diffuse_episodic_route(episodic_route)
    _assert_deterministic_sample_loaded(condenser, sample)
    if config.max_prompt_tokens is None:
        raise ValueError("diffuse analysis requires an explicit prompt cap")
    if (representative_linker is None) != (
        representative_policy_factory is None
    ):
        raise ValueError(
            "representative linker and policy factory must be supplied together"
        )
    if _packet_retriever_guard is not None and not callable(
        _packet_retriever_guard
    ):
        raise TypeError("_packet_retriever_guard must be callable")
    provider_identity_sha256 = _callable_implementation_sha256(
        legacy_input_provider,
        "legacy_input_provider",
    )
    representative_linker_identity_sha256 = (
        _representative_linker_identity_sha256(representative_linker)
    )
    representative_policy_factory_identity_sha256 = (
        None
        if representative_policy_factory is None
        else _callable_implementation_sha256(
            representative_policy_factory,
            "representative_policy_factory",
        )
    )
    evaluation_policy_sha256 = _evaluation_policy_sha256(config)
    compilation = compile_diffuse_artifact(
        condenser,
        policy=arm.compilation,
        qwen_scorer=qwen_scorer,
        embedding_identity=embedding_identity,
    )
    artifact_id = compilation.artifact.artifact_id
    rows: list[DiffuseLongMemEvalGoldBlindQuery] = []
    for probe in sample.questions:
        candidates = legacy_input_provider(
            condenser,
            query=probe.retrieval_query,
            retrieval=config.retrieval,
            artifact_id=artifact_id,
        )
        if _callable_implementation_sha256(
            legacy_input_provider,
            "legacy_input_provider",
        ) != provider_identity_sha256:
            raise RuntimeError(
                "legacy input provider identity changed during retrieval"
            )
        exact_inputs = capture_legacy_diffuse_inputs(
            query=probe.retrieval_query,
            retrieval=config.retrieval,
            artifact_id=artifact_id,
            candidates=candidates,
        )
        if exact_inputs.candidates.source_candidates and (
            representative_linker is None
            or representative_policy_factory is None
        ):
            raise ValueError(
                "independent source candidates require representative retrieval"
            )
        representative_policy = (
            None
            if representative_policy_factory is None
            else representative_policy_factory(artifact_id)
        )
        if representative_policy_factory is not None:
            if _callable_implementation_sha256(
                representative_policy_factory,
                "representative_policy_factory",
            ) != representative_policy_factory_identity_sha256:
                raise RuntimeError(
                    "representative policy factory identity changed during retrieval"
                )
            if not isinstance(
                representative_policy,
                EpisodeRepresentativeRetrievalPolicy,
            ):
                raise TypeError(
                    "representative policy factory returned an invalid policy"
                )
        representative_policy_sha256 = (
            None
            if representative_policy is None
            else representative_policy.policy_sha256
        )
        representative_policy_controls_sha256 = (
            None
            if representative_policy is None
            else _representative_policy_controls_sha256(
                representative_policy
            )
        )
        # Preserve the legacy downstream call shape.  Historical cross-commit
        # receipt bytes are intentionally not claimed because implementation
        # identities bind the package source tree.
        route_kwargs = (
            {}
            if active_episodic_route == "legacy_union"
            else {"episodic_route": active_episodic_route}
        )
        if _packet_retriever_guard is not None:
            _packet_retriever_guard()
        retrieval = _packet_retriever(
            condenser,
            query=probe.retrieval_query,
            prompt_question=probe.prompt_question,
            anchors=exact_inputs.candidates.anchors,
            artifact_id=artifact_id,
            max_context_tokens=arm.max_context_tokens,
            max_prompt_tokens=config.max_prompt_tokens,
            responder_output_token_reserve=(
                arm.responder_output_token_reserve
            ),
            episode_policy=arm.episode,
            source_candidates=exact_inputs.candidates.source_candidates,
            source_candidate_scope=(
                exact_inputs.candidates.source_candidate_scope
            ),
            representative_linker=representative_linker,
            representative_policy=representative_policy,
            require_owned_representative_runtime=(
                arm.require_owned_representative_runtime
            ),
            closure_policy=arm.closure,
            **route_kwargs,
        )
        receipt = DiffuseLongMemEvalAnalysisQueryReceipt(
            corpus_sha256=sample.corpus_sha256,
            question_probe_sha256=probe.probe_sha256,
            analysis_arm_sha256=arm.arm_sha256,
            matched_controls_sha256=arm.matched_controls_sha256,
            evaluation_policy_sha256=evaluation_policy_sha256,
            legacy_input_provider_identity_sha256=(
                provider_identity_sha256
            ),
            representative_linker_identity_sha256=(
                representative_linker_identity_sha256
            ),
            representative_policy_factory_identity_sha256=(
                representative_policy_factory_identity_sha256
            ),
            representative_policy_sha256=representative_policy_sha256,
            representative_policy_controls_sha256=(
                representative_policy_controls_sha256
            ),
            compilation_receipt_sha256=compilation.receipt_sha256,
            legacy_input_receipt_sha256=(
                exact_inputs.receipt.receipt_sha256
            ),
            diffuse_query_receipt_sha256=retrieval.receipt.receipt_sha256,
            artifact_id=artifact_id,
            snapshot_sha256=retrieval.receipt.snapshot_sha256,
        )
        if receipt.snapshot_sha256 != (
            compilation.final_snapshot.snapshot_sha256
        ):
            raise RuntimeError("query read a different compiled snapshot")
        rows.append(
            DiffuseLongMemEvalGoldBlindQuery(
                probe=probe,
                legacy_inputs=exact_inputs,
                retrieval=retrieval,
                receipt=receipt,
            )
        )
        observed_linker_identity = _representative_linker_identity_sha256(
            representative_linker
        )
        if observed_linker_identity != representative_linker_identity_sha256:
            raise RuntimeError(
                "representative linker identity changed during retrieval"
            )
    return DiffuseLongMemEvalRetrievalPhase(
        sample_id=sample.sample_id,
        corpus_sha256=sample.corpus_sha256,
        deterministic_turn_ids=sample.deterministic_turn_ids,
        arm=arm,
        evaluation_policy_sha256=evaluation_policy_sha256,
        compilation=compilation,
        questions=tuple(rows),
    )


def retrieve_diffuse_longmemeval_sample(
    condenser: MemoryCondenser,
    sample: GoldBlindLongMemEvalSample,
    *,
    config: EvalConfig,
    arm: DiffuseLongMemEvalArm,
    legacy_input_provider: LegacyDiffuseInputProvider,
    qwen_scorer: QwenAttentionHeadSurpriseScorer | None = None,
    embedding_identity: Mapping[str, object] | None = None,
    representative_linker: NestedEpisodeLinker | None = None,
    representative_policy_factory: RepresentativePolicyFactory | None = None,
) -> DiffuseLongMemEvalRetrievalPhase:
    """Compile and retrieve every probe without accepting benchmark gold."""

    return _retrieve_diffuse_longmemeval_sample_with_route(
        condenser,
        sample,
        config=config,
        arm=arm,
        legacy_input_provider=legacy_input_provider,
        qwen_scorer=qwen_scorer,
        embedding_identity=embedding_identity,
        representative_linker=representative_linker,
        representative_policy_factory=representative_policy_factory,
        episodic_route="legacy_union",
        _packet_retriever=retrieve_longmemeval_diffuse_packet,
    )


def measure_diffuse_longmemeval_sample(
    retrieval_phase: DiffuseLongMemEvalRetrievalPhase,
    sample: MeasurementSampleLike,
    *,
    hydrate_span: Callable[[EvidenceSpan], str],
) -> DiffuseLongMemEvalAnalysis:
    """Introduce gold only after every question packet has been frozen."""

    blind = gold_blind_longmemeval_sample(sample)
    if blind.sample_id != retrieval_phase.sample_id or (
        blind.corpus_sha256 != retrieval_phase.corpus_sha256
    ):
        raise ValueError("benchmark gold belongs to another retrieved corpus")
    if blind.deterministic_turn_ids != retrieval_phase.deterministic_turn_ids:
        raise ValueError("benchmark gold belongs to another deterministic ingest")
    questions_by_id = {item.question_id: item for item in sample.questions}
    if len(questions_by_id) != len(sample.questions):
        raise ValueError("benchmark question IDs must be unique")
    if set(questions_by_id) != {
        item.probe.question_id for item in retrieval_phase.questions
    }:
        raise ValueError("benchmark gold does not cover the frozen probes")
    measured: list[DiffuseLongMemEvalMeasuredQuestion] = []
    for frozen in retrieval_phase.questions:
        question = questions_by_id[frozen.probe.question_id]
        if _question_probe(question).probe_sha256 != frozen.probe.probe_sha256:
            raise ValueError("benchmark question text changed after retrieval")
        metrics = measure_longmemeval_diffuse_packet(
            frozen.retrieval,
            question_id=question.question_id,
            gold_answer=question.answer,
            evidence_source_ids=question.evidence_sources,
            hydrate_span=hydrate_span,
        )
        measured.append(
            DiffuseLongMemEvalMeasuredQuestion(
                gold_blind=frozen,
                metrics=metrics,
                gold_answer_sha256=quote_sha256(question.answer),
                evidence_sources_sha256=identity_sha256(
                    list(question.evidence_sources)
                ),
            )
        )
    return DiffuseLongMemEvalAnalysis(
        retrieval_phase=retrieval_phase,
        questions=tuple(measured),
    )


def run_diffuse_longmemeval_analysis(
    sample: MeasurementSampleLike,
    *,
    config: EvalConfig,
    arm: DiffuseLongMemEvalArm,
    data_dir: str | Path,
    condenser_factory: FreshCondenserFactory,
    legacy_input_provider: LegacyDiffuseInputProvider,
    qwen_scorer: QwenAttentionHeadSurpriseScorer | None = None,
    embedding_identity: Mapping[str, object] | None = None,
    representative_linker: NestedEpisodeLinker | None = None,
    representative_policy_factory: RepresentativePolicyFactory | None = None,
) -> DiffuseLongMemEvalAnalysis:
    """Run one sample in a new store, freezing retrieval before measurement."""

    store_path = Path(data_dir)
    if store_path.exists():
        raise FileExistsError(
            "diffuse analysis data_dir must not exist; use a fresh store path"
    )
    blind = gold_blind_longmemeval_sample(sample)
    condenser = condenser_factory(store_path, config)
    if not isinstance(condenser, MemoryCondenser):
        close = getattr(condenser, "close", None)
        if callable(close):
            close()
        raise TypeError("condenser_factory must return a fresh MemoryCondenser")
    try:
        ingest_gold_blind_sample_deterministically(condenser, blind)
        retrieval_phase = retrieve_diffuse_longmemeval_sample(
            condenser,
            blind,
            config=config,
            arm=arm,
            legacy_input_provider=legacy_input_provider,
            qwen_scorer=qwen_scorer,
            embedding_identity=embedding_identity,
            representative_linker=representative_linker,
            representative_policy_factory=representative_policy_factory,
        )
        return measure_diffuse_longmemeval_sample(
            retrieval_phase,
            sample,
            hydrate_span=condenser.discourse.hydrate_span,
        )
    finally:
        condenser.close()


__all__ = [
    "DETERMINISTIC_DIFFUSE_INGEST_FORMAT",
    "DIFFUSE_ANALYSIS_FORMAT",
    "DIFFUSE_ANALYSIS_PHASE_FORMAT",
    "DIFFUSE_ANALYSIS_QUERY_FORMAT",
    "DIFFUSE_MATCHED_PROBE_FORMAT",
    "DIFFUSE_MATCHED_SUITE_FORMAT",
    "LEGACY_DIFFUSE_INPUT_FORMAT",
    "DiffuseLongMemEvalAnalysis",
    "DiffuseLongMemEvalAnalysisQueryReceipt",
    "DiffuseLongMemEvalArm",
    "DiffuseLongMemEvalGoldBlindQuery",
    "DiffuseLongMemEvalMatchedProbeReceipt",
    "DiffuseLongMemEvalMatchedSuiteReceipt",
    "DiffuseLongMemEvalMeasuredQuestion",
    "DiffuseLongMemEvalRetrievalPhase",
    "ExactLegacyDiffuseInputs",
    "FreshCondenserFactory",
    "GoldBlindLongMemEvalQuestion",
    "GoldBlindLongMemEvalSample",
    "LegacyDiffuseCandidates",
    "LegacyDiffuseInputProvider",
    "LegacyDiffuseInputReceipt",
    "RepresentativePolicyFactory",
    "analysis_callable_identity_payload",
    "capture_legacy_diffuse_inputs",
    "gold_blind_longmemeval_sample",
    "ingest_gold_blind_sample_deterministically",
    "matched_diffuse_boundary_arms",
    "measure_diffuse_longmemeval_sample",
    "retrieve_diffuse_longmemeval_sample",
    "run_diffuse_longmemeval_analysis",
    "validate_matched_diffuse_retrieval_phases",
]
