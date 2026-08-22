"""Fail-closed, two-stage runner for one locked Mem0 comparison shard.

Stage A belongs in the isolated Mem0 environment.  It converts the lossless
``CompositeAddBatch`` sequence to the frozen adapter's private prepared-corpus
seam, performs exactly one ingest and ten searches, deletes all owned state,
and atomically publishes a content-addressed retrieval artifact.

Stage B belongs in the frozen memory-condense environment.  It verifies the
Stage-A artifact and trace, independently rebuilds the exact prompts, and only
then permits the separately injected responder and judge callables to perform
the pre-authorized twenty calls.  This module deliberately contains no real
Mem0 configuration and no provider client: those identities must be frozen by
campaign policy before a caller can construct either authorization object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, Sequence

from memory_condense.eval.benchmark import (
    build_judge_prompt,
    exact_match,
    f1_score,
)
from memory_condense.eval.mem0_adapter import (
    MEM0AI_PIN,
    MEM0_ATTRIBUTION_KIND,
    MEM0_BM25_MODEL,
    MEM0_CERTIFIED_RENDERING,
    MEM0_OFFICIAL_THRESHOLD,
    MEM0_OFFICIAL_TOP_K,
    MEM0_PROVIDER_USAGE_STATUS,
    MEM0_SPACY_MODEL,
    Mem0AdapterStats,
    SourceRef,
    _PreparedBatch,
    _PreparedCorpus,
)
from memory_condense.eval.schemas import UsageStats
from .source_compat import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)

from .preflight import tool_implementation_sha256
from .protocol import (
    CompositeAddBatch,
    RawStressShard,
    shard_receipt,
    validate_raw_stress_shard,
)


SHARD_SCHEMA_VERSION = 2
RETRIEVAL_ARTIFACT_FORMAT = "memory-condense-mem0-retrieval-artifact-v2"
RETRIEVAL_TRACE_FORMAT = "memory-condense-mem0-retrieval-trace-v2"
SCORING_TRACE_FORMAT = "memory-condense-mem0-scoring-trace-v2"
SCORING_RECEIPT_FORMAT = "memory-condense-mem0-scoring-receipt-v2"
SHARD_REPORT_TYPE = "mem0_longmemeval_stress_shard"
SHARD_ARM_ID = "mem0_oss_2_0_18_direct_1m_v1"
MEM0_RUNTIME_PROTOCOL = "mem0-oss-2.0.18-certified-local-v1"
INPUT_ORDER_PROTOCOL = (
    "locked-record-order+official-within-record-date-sort+"
    "consecutive-1-or-2-turn-slices-v1"
)
LONGMEMEVAL_QUESTIONS_PER_SHARD = 10

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JUDGE_RE = re.compile(r"^\s*(CORRECT|INCORRECT)\b", re.IGNORECASE)
_FORBIDDEN_SECRET_KEYS = {
    "api_key",
    "api-key",
    "x_api_key",
    "x-api-key",
    "authorization",
    "proxy_authorization",
    "proxy-authorization",
    "auth",
    "cookie",
    "set_cookie",
    "set-cookie",
    "password",
    "passwd",
    "secret",
    "secret_key",
    "token",
    "access_token",
    "refresh_token",
    "sas_token",
    "client_secret",
    "client_key",
    "private_key",
    "signing_key",
    "connection_string",
    "credentials",
}
_SECRET_VALUE_RE = re.compile(
    r"(?:^\s*(?:bearer|basic)\s+\S+|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:sk|ghp|github_pat|xox[baprs]|AIza)[-_][A-Za-z0-9_-]{8,}|"
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})",
    re.IGNORECASE,
)


class Mem0ShardRunError(RuntimeError):
    """A stage failed closed before publishing a successful receipt."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        trace_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.trace_path = trace_path


class AdapterFactory(Protocol):
    def __call__(self, owned_state_dir: Path) -> Any: ...


class ExtractionMeterInstaller(Protocol):
    """Install a logical-call wrapper and return a no-argument restore hook."""

    def __call__(
        self, adapter: Any, meter: "LogicalExtractionCallMeter"
    ) -> Callable[[], Any]: ...


@dataclass(frozen=True, slots=True)
class ProviderCallResult:
    """One and only one provider response plus its exact usage receipt."""

    text: str
    usage: UsageStats


class ProviderInvoker(Protocol):
    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        max_output_tokens: int,
    ) -> ProviderCallResult: ...


REQUEST_TOKEN_STATE_CONTRACT = "stateless-request-token-state-v1"


def _request_token_state_receipt(value: Any, label: str) -> dict[str, Any]:
    reader = getattr(value, "request_token_state_receipt", None)
    if not callable(reader):
        raise TypeError(
            f"{label} must inject a callable request_token_state_receipt contract"
        )
    receipt = reader()
    if not isinstance(receipt, Mapping):
        raise TypeError(f"{label} request-token-state receipt must be a mapping")
    normalized = _strict_json(receipt)
    assert isinstance(normalized, dict)
    required = {
        "contract": REQUEST_TOKEN_STATE_CONTRACT,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
    }
    for field, expected in required.items():
        if normalized.get(field) != expected:
            raise ValueError(
                f"{label} request-token-state receipt {field} mismatch"
            )
    return dict(required)


@dataclass(frozen=True, slots=True)
class RetrievalStageAuthorization:
    """Policy-selected authority for Stage A; every count is fail-closed."""

    sample_offset: int
    sample_sha256: str
    raw_history_bundle_sha256: str
    question_ids: tuple[str, ...]
    authorized_add_operations: int
    authorized_extraction_calls: int
    authorized_search_operations: int
    source_validation_policy_sha256: str
    source_implementation_sha256: str
    source_environment_lock_sha256: str
    mem0_policy_sha256: str
    mem0_tool_implementation_sha256: str
    mem0_environment_lock_sha256: str
    mem0_stable_config_sha256: str
    source_evaluation_identity: Mapping[str, Any]
    mem0_stable_payload: Mapping[str, Any]
    extraction_model_identity: Mapping[str, Any] | None = None
    extraction_model_identity_sha256: str | None = None
    embedder_model_identity: Mapping[str, Any] | None = None
    embedder_model_identity_sha256: str | None = None
    mem0_provider_retries: int = 0


@dataclass(frozen=True, slots=True)
class ScoringStageAuthorization:
    """Policy-selected authority for Stage B and its exact provider budget."""

    sample_offset: int
    sample_sha256: str
    raw_history_bundle_sha256: str
    question_ids: tuple[str, ...]
    retrieval_artifact_sha256: str
    source_validation_policy_sha256: str
    source_implementation_sha256: str
    source_environment_lock_sha256: str
    mem0_policy_sha256: str
    mem0_tool_implementation_sha256: str
    mem0_environment_lock_sha256: str
    mem0_stable_config_sha256: str
    source_evaluation_identity: Mapping[str, Any]
    mem0_stable_payload: Mapping[str, Any]
    scoring_policy_sha256: str
    responder_model: str
    judge_model: str
    responder_model_identity_sha256: str
    judge_model_identity_sha256: str
    extraction_model_identity: Mapping[str, Any] | None = None
    extraction_model_identity_sha256: str | None = None
    embedder_model_identity: Mapping[str, Any] | None = None
    embedder_model_identity_sha256: str | None = None
    authorized_responder_calls: int = LONGMEMEVAL_QUESTIONS_PER_SHARD
    authorized_judge_calls: int = LONGMEMEVAL_QUESTIONS_PER_SHARD
    max_prompt_tokens: int = 8_000
    responder_max_output_tokens: int = 256
    judge_max_output_tokens: int = 1_024
    provider_retries: int = 0


@dataclass(frozen=True, slots=True)
class RetrievalStageResult:
    artifact_path: Path
    trace_path: Path
    artifact_sha256: str
    artifact_bytes: int
    artifact: Mapping[str, Any]
    trace: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ScoringStageResult:
    report_path: Path
    trace_path: Path
    report_sha256: str
    report_bytes: int
    report: Mapping[str, Any]
    trace: Mapping[str, Any]


class ShardProcessGuard:
    """Single-use process gate, injectable so fake tests remain independent."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._claimed = False

    def claim(self) -> None:
        if self._claimed:
            raise Mem0ShardRunError(
                f"{self.label} already ran in this process; launch a fresh process",
                stage=self.label,
            )
        self._claimed = True


class TrustedRuntimeBinding:
    """Opaque capability reserved for a concrete, frozen production launcher.

    This injectable-core module deliberately exposes no issuer.  Adding the
    real provider implementations later must also add a narrow issuer that
    verifies their exact concrete types, policy bytes, runtime probes, and
    environment lock before it can possess the private seal.  Arbitrary test
    callables therefore cannot elevate their own receipts to production.
    """

    __slots__ = (
        "_authorization_sha256",
        "_bound_callables",
        "_launcher",
        "_receipt",
        "_seal",
        "_stage",
    )

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "TrustedRuntimeBinding can only be issued by the frozen production "
            "launcher; direct construction and injected callables are forbidden"
        )


_TRUSTED_RUNTIME_BINDING_SEAL = object()


def _authorization_binding_sha256(
    authorization: RetrievalStageAuthorization | ScoringStageAuthorization,
) -> str:
    if type(authorization) not in {
        RetrievalStageAuthorization,
        ScoringStageAuthorization,
    }:
        raise TypeError("runtime binding requires an exact stage authorization type")
    return canonical_json_sha256(asdict(authorization))


def _issue_trusted_runtime_binding(
    *,
    launcher: Any,
    stage: str,
    authorization: RetrievalStageAuthorization | ScoringStageAuthorization,
    bound_callables: Sequence[Any],
) -> TrustedRuntimeBinding:
    """Private bridge from an exact, independently verified launcher.

    The provider-free injectable runner cannot manufacture a positive binding.
    ``production_binding`` owns the only concrete launcher types and rechecks
    their frozen artifacts, local-model probe, and transport caps before this
    function allocates the opaque capability.  Keeping the import lazy avoids a
    dependency cycle and, importantly, does not make arbitrary callbacks
    production-eligible.
    """

    if stage not in {"retrieval", "scoring"}:
        raise ValueError("trusted runtime binding stage must be retrieval or scoring")
    expected_type = (
        RetrievalStageAuthorization
        if stage == "retrieval"
        else ScoringStageAuthorization
    )
    if type(authorization) is not expected_type:
        raise TypeError("trusted runtime binding authorization/stage mismatch")
    if not isinstance(bound_callables, Sequence) or isinstance(
        bound_callables, (str, bytes, bytearray)
    ):
        raise TypeError("trusted runtime binding callables must be a sequence")
    subjects = tuple(bound_callables)
    if not subjects:
        raise TypeError("trusted runtime binding must bind concrete runtime subjects")

    authorization_sha256 = _authorization_binding_sha256(authorization)
    from .production_binding import (
        _consume_trusted_runtime_claim,
        _validate_public_production_receipt,
    )

    receipt = _consume_trusted_runtime_claim(
        launcher,
        stage=stage,
        authorization_sha256=authorization_sha256,
        bound_callables=subjects,
    )
    if not isinstance(receipt, Mapping):
        raise TypeError("production launcher returned a non-mapping binding receipt")
    normalized = _strict_json(receipt, path="trusted_runtime_binding_receipt")
    assert isinstance(normalized, dict)
    normalized = _validate_public_production_receipt(
        normalized,
        stage=stage,
        authorization_sha256=authorization_sha256,
    )

    binding = object.__new__(TrustedRuntimeBinding)
    binding._seal = _TRUSTED_RUNTIME_BINDING_SEAL
    binding._stage = stage
    binding._authorization_sha256 = authorization_sha256
    binding._bound_callables = subjects
    binding._receipt = MappingProxyType(normalized)
    binding._launcher = launcher
    return binding


def _execution_binding_receipt(
    binding: TrustedRuntimeBinding | None,
    *,
    stage: str | None = None,
    authorization: RetrievalStageAuthorization | ScoringStageAuthorization | None = None,
    bound_callables: Sequence[Any] = (),
) -> dict[str, Any]:
    if binding is None:
        return {
            "kind": "injected_nonproduction",
            "trusted_runtime_binding_receipt_sha256": None,
            "comparison_certified": False,
            "external_http_attempts_certified": False,
            "external_provider_persistence_certified": False,
        }
    if stage is None or authorization is None:
        raise TypeError(
            "positive runtime binding verification requires stage and authorization"
        )
    if type(binding) is not TrustedRuntimeBinding:
        raise TypeError("trusted runtime binding must have the exact opaque type")
    try:
        sealed = binding._seal is _TRUSTED_RUNTIME_BINDING_SEAL
        bound_stage = binding._stage
        bound_authorization = binding._authorization_sha256
        subjects = binding._bound_callables
        receipt = dict(binding._receipt)
        launcher = binding._launcher
    except AttributeError as exc:
        raise TypeError("trusted runtime binding is incomplete or forged") from exc
    if not sealed:
        raise TypeError("trusted runtime binding seal mismatch")
    if bound_stage != stage:
        raise TypeError("trusted runtime binding stage mismatch")
    if bound_authorization != _authorization_binding_sha256(authorization):
        raise TypeError("trusted runtime binding authorization mismatch")
    observed_subjects = tuple(bound_callables)
    if len(subjects) != len(observed_subjects) or any(
        expected is not observed
        for expected, observed in zip(subjects, observed_subjects, strict=True)
    ):
        raise TypeError("trusted runtime binding callable identity mismatch")

    from .production_binding import (
        _recheck_trusted_runtime_claim,
        _validate_public_production_receipt,
    )

    receipt = _validate_public_production_receipt(
        receipt,
        stage=stage,
        authorization_sha256=bound_authorization,
    )

    _recheck_trusted_runtime_claim(
        launcher,
        stage=stage,
        authorization_sha256=bound_authorization,
        bound_callables=subjects,
        receipt=receipt,
    )
    return receipt


_RETRIEVAL_PROCESS_GUARD = ShardProcessGuard("retrieval")
_SCORING_PROCESS_GUARD = ShardProcessGuard("scoring")


@dataclass(slots=True)
class LogicalExtractionCallMeter:
    """Hard-cap calls to Mem0's logical extraction-model boundary.

    This counts calls to ``Memory.llm.generate_response``.  It deliberately
    does not claim HTTP-attempt accounting, which Mem0 OSS does not expose.
    """

    authorized: int
    attempted: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    infer_true_adds_started: int = 0
    infer_true_adds_exactly_one_call: int = 0
    _wrapped: bool = False
    _add_wrapped: bool = False
    _inside_infer_true_add: bool = False
    _active_add_calls: int = 0
    _active_add_rejections: int = 0
    _request_token_state_reader: Any | None = None
    _request_token_state_verified: bool = False

    def __post_init__(self) -> None:
        _require_count(self.authorized, "authorized extraction calls", minimum=1)

    def wrap(self, callback: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``Memory.llm.generate_response`` with the hard call gate."""

        if self._wrapped:
            raise ValueError("logical extraction meter can wrap only once")
        if not callable(callback):
            raise TypeError("logical extraction target must be callable")
        self._wrapped = True

        def metered(*args: Any, **kwargs: Any) -> Any:
            if not self._inside_infer_true_add:
                self.rejected += 1
                raise Mem0ShardRunError(
                    "Mem0 called its extraction model outside an infer=True add",
                    stage="retrieval",
                )
            if self._active_add_calls >= 1:
                self.rejected += 1
                self._active_add_rejections += 1
                raise Mem0ShardRunError(
                    "an infer=True Mem0 add attempted more than one logical "
                    "extraction-model call",
                    stage="retrieval",
                )
            if self.attempted >= self.authorized:
                self.rejected += 1
                self._active_add_rejections += 1
                raise Mem0ShardRunError(
                    "logical Mem0 extraction-call authorization exhausted",
                    stage="retrieval",
                )
            self._active_add_calls += 1
            self.attempted += 1
            try:
                result = callback(*args, **kwargs)
            except BaseException:
                self.failed += 1
                raise
            self.completed += 1
            return result

        return metered

    def bind_request_token_state_contract(self, llm: Any) -> None:
        if self._request_token_state_reader is not None:
            raise ValueError("request-token-state contract can be bound only once")
        _request_token_state_receipt(llm, "Mem0 extraction LLM")
        self._request_token_state_reader = llm
        self._request_token_state_verified = True

    def verify_request_token_state(self) -> None:
        if self._request_token_state_reader is None:
            raise Mem0ShardRunError(
                "Mem0 extraction LLM omitted its stateless request contract",
                stage="retrieval",
            )
        self._request_token_state_verified = False
        _request_token_state_receipt(
            self._request_token_state_reader, "Mem0 extraction LLM"
        )
        self._request_token_state_verified = True

    def wrap_infer_true_add(
        self, callback: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Supervise each public ``Memory.add(..., infer=True)`` operation."""

        if self._add_wrapped:
            raise ValueError("logical extraction meter can wrap Memory.add only once")
        if not callable(callback):
            raise TypeError("Mem0 Memory.add target must be callable")
        self._add_wrapped = True

        def metered_add(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("infer") is not True:
                raise Mem0ShardRunError(
                    "certified Mem0 ingestion requires explicit infer=True",
                    stage="retrieval",
                )
            if self._inside_infer_true_add:
                raise Mem0ShardRunError(
                    "nested Mem0 Memory.add calls are not permitted",
                    stage="retrieval",
                )
            self._inside_infer_true_add = True
            self._active_add_calls = 0
            self._active_add_rejections = 0
            self.infer_true_adds_started += 1
            operation_error: BaseException | None = None
            try:
                return callback(*args, **kwargs)
            except BaseException as exc:
                operation_error = exc
                raise
            finally:
                observed = self._active_add_calls
                rejected = self._active_add_rejections
                self._inside_infer_true_add = False
                self._active_add_calls = 0
                self._active_add_rejections = 0
                policy_error: Mem0ShardRunError | None = None
                if observed != 1 or rejected:
                    policy_error = Mem0ShardRunError(
                        "each infer=True Mem0 add must make exactly one logical "
                        "Memory.llm.generate_response call "
                        f"(observed={observed}, rejected={rejected})",
                        stage="retrieval",
                    )
                else:
                    self.infer_true_adds_exactly_one_call += 1
                if policy_error is not None:
                    if operation_error is None:
                        raise policy_error
                    operation_error.add_note(str(policy_error))

        return metered_add

    def assert_complete(self) -> None:
        self.verify_request_token_state()
        expected = self.authorized
        actual = {
            "attempted": self.attempted,
            "completed": self.completed,
            "failed": self.failed,
            "rejected": self.rejected,
            "infer_true_adds_started": self.infer_true_adds_started,
            "infer_true_adds_exactly_one_call": (
                self.infer_true_adds_exactly_one_call
            ),
        }
        required = {
            "attempted": expected,
            "completed": expected,
            "failed": 0,
            "rejected": 0,
            "infer_true_adds_started": expected,
            "infer_true_adds_exactly_one_call": expected,
        }
        if actual != required or self._inside_infer_true_add:
            raise Mem0ShardRunError(
                "logical Mem0 extraction-call receipt did not close exactly: "
                f"expected={required!r}, observed={actual!r}",
                stage="retrieval",
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "kind": "mem0_memory_llm_generate_response_logical_calls",
            "boundary": "Memory.llm.generate_response",
            "count_semantics": "local_logical_wrapper_calls_not_http_attempts",
            "external_http_attempts_certified": False,
            "authorized_local_wrapper_retries": 0,
            "external_retry_attempts_certified": False,
            "authorized": self.authorized,
            "attempted": self.attempted,
            "completed": self.completed,
            "failed": self.failed,
            "rejected": self.rejected,
            "infer_true_adds_started": self.infer_true_adds_started,
            "infer_true_adds_exactly_one_call": (
                self.infer_true_adds_exactly_one_call
            ),
            "one_logical_call_per_infer_true_add_certified": (
                self._request_token_state_verified
                and not self._inside_infer_true_add
                and self.attempted == self.authorized
                and self.completed == self.authorized
                and self.failed == 0
                and self.rejected == 0
                and self.infer_true_adds_started == self.authorized
                and self.infer_true_adds_exactly_one_call == self.authorized
            ),
            "persisted_request_token_state": (
                False if self._request_token_state_verified else None
            ),
            "retained_request_token_state_bytes": (
                0 if self._request_token_state_verified else None
            ),
            "request_token_state_evidence_kind": (
                "local_injected_request_token_state_contract"
                if self._request_token_state_verified
                else None
            ),
            "external_provider_persistence_certified": False,
        }


def _same_callable(left: Any, right: Any) -> bool:
    if left is right:
        return True
    return (
        getattr(left, "__self__", None) is getattr(right, "__self__", None)
        and getattr(left, "__func__", None) is getattr(right, "__func__", None)
        and getattr(left, "__func__", None) is not None
    )


def install_memory_llm_extraction_meter(
    adapter: Any, meter: LogicalExtractionCallMeter
) -> Callable[[], None]:
    """Patch the real Mem0 ``Memory`` object and return an exact restore hook.

    ``Mem0LongMemEvalAdapter`` owns an ``_OwnedMem0Backend`` whose ``backend``
    attribute is the pinned ``mem0.Memory`` instance.  Resolving that exact
    chain keeps the counter at the model boundary rather than incorrectly
    treating a public ``Memory.add`` operation as a provider call.
    """

    get_backend = getattr(adapter, "_get_backend", None)
    if not callable(get_backend):
        raise TypeError("Mem0 adapter omitted its owned-backend getter")
    owned_backend = get_backend()
    memory = getattr(owned_backend, "backend", None)
    if memory is None:
        raise TypeError("Mem0 owned backend omitted its Memory instance")
    llm = getattr(memory, "llm", None)
    if llm is None:
        raise TypeError("Mem0 Memory omitted its llm instance")
    original_generate = getattr(llm, "generate_response", None)
    original_add = getattr(memory, "add", None)
    if not callable(original_generate):
        raise TypeError("Mem0 Memory.llm omitted generate_response")
    if not callable(original_add):
        raise TypeError("Mem0 Memory omitted add")

    meter.bind_request_token_state_contract(llm)

    wrapped_generate = meter.wrap(original_generate)
    wrapped_add = meter.wrap_infer_true_add(original_add)
    generate_installed = False
    add_installed = False
    try:
        setattr(llm, "generate_response", wrapped_generate)
        generate_installed = True
        if getattr(llm, "generate_response", None) is not wrapped_generate:
            raise TypeError("could not bind the Mem0 logical-call wrapper")
        setattr(memory, "add", wrapped_add)
        add_installed = True
        if getattr(memory, "add", None) is not wrapped_add:
            raise TypeError("could not bind the Mem0 infer=True add supervisor")
    except BaseException:
        if add_installed:
            setattr(memory, "add", original_add)
        if generate_installed:
            setattr(llm, "generate_response", original_generate)
        raise

    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            raise RuntimeError("Mem0 extraction wrappers were already restored")
        errors: list[str] = []
        if getattr(memory, "add", None) is not wrapped_add:
            errors.append("Memory.add wrapper changed before restoration")
        if getattr(llm, "generate_response", None) is not wrapped_generate:
            errors.append(
                "Memory.llm.generate_response wrapper changed before restoration"
            )
        try:
            setattr(memory, "add", original_add)
        except BaseException as exc:
            errors.append(f"Memory.add restoration failed: {type(exc).__name__}")
        try:
            setattr(llm, "generate_response", original_generate)
        except BaseException as exc:
            errors.append(
                "Memory.llm.generate_response restoration failed: "
                f"{type(exc).__name__}"
            )
        if not _same_callable(getattr(memory, "add", None), original_add):
            errors.append("Memory.add restoration could not be verified")
        if not _same_callable(
            getattr(llm, "generate_response", None), original_generate
        ):
            errors.append(
                "Memory.llm.generate_response restoration could not be verified"
            )
        restored = not errors
        if errors:
            raise RuntimeError("; ".join(errors))

    return restore


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value.strip()


def _require_count(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_json(value: Any, *, path: str = "$" ) -> Any:
    """Deep-copy to plain JSON types and reject lossy/default=str encoding."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _strict_json(asdict(value), path=path)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _strict_json(model_dump(mode="json"), path=path)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            out[key] = _strict_json(item, path=f"{path}.{key}")
        return out
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _strict_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _strict_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_secret_material(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if key in _FORBIDDEN_SECRET_KEYS or key.endswith(
                (
                    "_password",
                    "_secret",
                    "_token",
                    "_api_key",
                    "_authorization",
                    "_auth_token",
                    "_secret_key",
                    "_private_key",
                    "_signing_key",
                    "_connection_string",
                )
            ):
                if child not in (None, "", "<redacted>"):
                    raise ValueError(
                        f"{path} contains unredacted secret field {raw_key!r}"
                    )
            _reject_secret_material(child, path=f"{path}.{raw_key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_secret_material(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and any(
        marker in path.casefold()
        for marker in (
            "config",
            "identity",
            "binding",
            "environment",
            "header",
            "credential",
        )
    ):
        if value != "<redacted>" and _SECRET_VALUE_RE.search(value):
            raise ValueError(f"{path} contains credential-shaped secret material")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    # Deliberately self-contained: this module is bootstrapped against the
    # frozen v3 source snapshot, which predates memory_condense.domain.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _environment_lock_snapshot(
    value: str | os.PathLike[str], *, label: str
) -> tuple[Path, str]:
    try:
        path = Path(value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} must name an existing lock file") from exc
    if not path.is_file():
        raise ValueError(f"{label} must name a regular lock file")
    return path, _file_sha256(path)


def _is_equal_to_or_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _render_json_bytes(value: Mapping[str, Any]) -> bytes:
    _reject_secret_material(value)
    return json.dumps(
        _strict_json(value),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"


def _stage_bytes(target: Path, payload: bytes) -> Path:
    """Flush ``payload`` to a same-directory file without publishing it."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to replace existing artifact {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".staging", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _rollback_linked_target(target: Path, staging: Path) -> None:
    """Remove only the destination hard-linked from our still-live staging file."""

    if target.exists() and os.path.samefile(target, staging):
        target.unlink()


def _atomic_create_payloads(
    entries: Sequence[tuple[Path, bytes]],
) -> tuple[tuple[str, int], ...]:
    """Publish one or more files no-clobber, rolling back a partial group.

    Every payload is fully flushed to a same-directory staging file before any
    destination becomes visible.  Hard-link creation is atomic and cannot
    replace a racing writer.  If a later link fails, earlier links are removed
    only after proving that they still name this transaction's staging inode.
    Thus a two-file Stage A/B receipt cannot leave one of this invocation's
    ``complete`` documents orphaned after a handled publication failure.
    """

    normalized = tuple((path.resolve(strict=False), payload) for path, payload in entries)
    if not normalized:
        raise ValueError("atomic publication requires at least one payload")
    targets = [target for target, _payload in normalized]
    if len(set(targets)) != len(targets):
        raise ValueError("atomic publication targets must be distinct")
    for index, target in enumerate(targets):
        if any(
            target in other.parents or other in target.parents
            for other in targets[index + 1 :]
        ):
            raise ValueError("atomic publication targets must not be nested")
    for target in targets:
        if target.exists():
            raise FileExistsError(f"refusing to replace existing artifact {target}")

    staged: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for target, payload in normalized:
            staged.append((target, _stage_bytes(target, payload)))
        for target, temporary in staged:
            # Linking a fully flushed same-directory file creates the
            # destination atomically and, unlike replace/rename, cannot
            # clobber a racing writer.
            os.link(temporary, target)
            published.append((target, temporary))
    except BaseException as publication_error:
        rollback_errors: list[BaseException] = []
        for target, temporary in reversed(published):
            try:
                _rollback_linked_target(target, temporary)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            publication_error.add_note(
                "atomic publication rollback failed for "
                f"{len(rollback_errors)} destination(s)"
            )
        raise
    finally:
        for _target, temporary in staged:
            temporary.unlink(missing_ok=True)

    return tuple(
        (hashlib.sha256(payload).hexdigest(), len(payload))
        for _target, payload in normalized
    )


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> tuple[str, int]:
    """Publish complete JSON without replacing an existing campaign receipt."""

    return _atomic_create_payloads(((path, _render_json_bytes(value)),))[0]


def _reject_protected_outputs(
    outputs: Sequence[Path],
    protected: Sequence[tuple[Path, str]],
    *,
    label: str,
) -> None:
    for output in outputs:
        for protected_path, protected_label in protected:
            if _is_equal_to_or_within(output, protected_path):
                raise ValueError(
                    f"{label} output {output} equals or descends from protected "
                    f"{protected_label} {protected_path}"
                )


def _implementation_roots() -> tuple[tuple[Path, str], ...]:
    repository = Path(__file__).resolve().parents[2]
    return (
        (Path(__file__).resolve().parent, "Mem0 tool implementation root"),
        (
            (repository / "src" / "memory_condense").resolve(strict=False),
            "source implementation root",
        ),
    )


def _parse_json_object_bytes(payload: bytes, *, filename: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Mem0ShardRunError(
            f"cannot parse JSON artifact {filename}", stage="scoring"
        ) from exc
    if not isinstance(value, dict):
        raise Mem0ShardRunError(
            f"JSON artifact {filename} is not an object", stage="scoring"
        )
    return value


def _load_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    value = _parse_json_object_bytes(payload, filename=path.name)
    return value, payload


def _validate_common_authorization(
    authorization: RetrievalStageAuthorization | ScoringStageAuthorization,
) -> None:
    _require_count(authorization.sample_offset, "sample_offset")
    for name in (
        "sample_sha256",
        "raw_history_bundle_sha256",
        "source_validation_policy_sha256",
        "source_implementation_sha256",
        "source_environment_lock_sha256",
        "mem0_policy_sha256",
        "mem0_tool_implementation_sha256",
        "mem0_environment_lock_sha256",
        "mem0_stable_config_sha256",
    ):
        _require_sha256(getattr(authorization, name), name)
    if (
        not isinstance(authorization.question_ids, tuple)
        or len(authorization.question_ids) != LONGMEMEVAL_QUESTIONS_PER_SHARD
        or len(set(authorization.question_ids)) != len(authorization.question_ids)
        or any(not isinstance(value, str) or not value.strip() for value in authorization.question_ids)
    ):
        raise ValueError("question_ids must be ten unique non-empty IDs")
    from .prompt_pack import validate_source_evaluation_identity

    validate_source_evaluation_identity(authorization.source_evaluation_identity)
    _validated_stable_payload(authorization)


def _validated_model_identities(
    authorization: RetrievalStageAuthorization | ScoringStageAuthorization,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specifications = (
        (
            "extraction_model_identity",
            "extraction_model_identity_sha256",
            {
                "provider",
                "model",
                "revision",
                "model_identity_sha256",
                "provider_retries",
                "logical_call_boundary",
                "logical_calls_per_add",
                "http_attempts_certified",
            },
        ),
        (
            "embedder_model_identity",
            "embedder_model_identity_sha256",
            {
                "provider",
                "model",
                "revision",
                "model_identity_sha256",
                "checkpoint_sha256",
                "dimension",
                "device",
                "dtype",
                "execution",
                "network_calls_authorized",
                "runtime_probe_required",
            },
        ),
    )
    for identity_field, digest_field, expected_keys in specifications:
        raw = getattr(authorization, identity_field)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{identity_field} must be a mapping")
        identity = _strict_json(raw, path=identity_field)
        assert isinstance(identity, dict)
        if set(identity) != expected_keys:
            raise ValueError(f"{identity_field} fields mismatch")
        _reject_secret_material(identity, path=identity_field)
        for field in ("provider", "model", "revision"):
            _require_nonempty(identity.get(field), f"{identity_field}.{field}")
        internal_digest = _require_sha256(
            identity.get("model_identity_sha256"),
            f"{identity_field}.model_identity_sha256",
        )
        internal_body = dict(identity)
        del internal_body["model_identity_sha256"]
        if canonical_json_sha256(internal_body) != internal_digest:
            raise ValueError(f"{identity_field} internal digest mismatch")
        supplied_digest = _require_sha256(
            getattr(authorization, digest_field), digest_field
        )
        if canonical_json_sha256(identity) != supplied_digest:
            raise ValueError(f"{digest_field} mismatch")
        rows.append(identity)

    extraction, embedder = rows
    if extraction["provider_retries"] != 0:
        raise ValueError("extraction model retries must be zero")
    if extraction["logical_call_boundary"] != "Memory.llm.generate_response":
        raise ValueError("extraction logical-call boundary mismatch")
    if extraction["logical_calls_per_add"] != 1:
        raise ValueError("extraction logical calls per add must be one")
    if extraction["http_attempts_certified"] is not False:
        raise ValueError("extraction identity must not certify HTTP attempts")
    _require_sha256(
        embedder["checkpoint_sha256"],
        "embedder_model_identity.checkpoint_sha256",
    )
    _require_count(embedder["dimension"], "embedder dimension", minimum=1)
    if embedder["device"] not in {"cpu", "cuda"}:
        raise ValueError("embedder device must be cpu or cuda")
    _require_nonempty(embedder["dtype"], "embedder dtype")
    if embedder["execution"] != "local_offline":
        raise ValueError("embedder execution must be local_offline")
    if embedder["network_calls_authorized"] != 0:
        raise ValueError("embedder network calls must remain unauthorized")
    if embedder["runtime_probe_required"] is not True:
        raise ValueError("embedder runtime probe must be required")
    for field, wanted in {
        "provider": "huggingface",
        "model": DEFAULT_MODEL_NAME,
        "revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "dimension": DEFAULT_MODEL_DIM,
        "dtype": "float32",
    }.items():
        if embedder[field] != wanted:
            raise ValueError(
                f"embedder {field} does not match the frozen local BGE-M3 arm"
            )
    return extraction, embedder


def _validated_stable_payload(
    authorization: RetrievalStageAuthorization | ScoringStageAuthorization,
) -> dict[str, Any]:
    raw = authorization.mem0_stable_payload
    if not isinstance(raw, Mapping):
        raise ValueError("mem0_stable_payload must be a mapping")
    payload = _strict_json(raw, path="mem0_stable_payload")
    assert isinstance(payload, dict)
    if set(payload) != {"protocol", "config", "stack"}:
        raise ValueError("mem0_stable_payload fields mismatch")
    _reject_secret_material(payload, path="mem0_stable_payload")
    if payload.get("protocol") != MEM0_RUNTIME_PROTOCOL:
        raise ValueError("mem0_stable_payload protocol mismatch")
    if canonical_json_sha256(payload) != authorization.mem0_stable_config_sha256:
        raise ValueError("mem0_stable_payload digest mismatch")
    return payload


def _validate_retrieval_authorization(
    shard: RawStressShard,
    authorization: RetrievalStageAuthorization,
    *,
    computed_mem0_environment_lock_sha256: str,
) -> None:
    _validate_common_authorization(authorization)
    _validated_model_identities(authorization)
    _require_sha256(
        computed_mem0_environment_lock_sha256,
        "computed_mem0_environment_lock_sha256",
    )
    expected = {
        "sample_offset": shard.sample_offset,
        "sample_sha256": shard.sample_sha256,
        "raw_history_bundle_sha256": shard.raw_history_bundle_sha256,
        "question_ids": shard.question_ids,
        "authorized_add_operations": shard.add_counts.add_requests,
        "authorized_extraction_calls": shard.add_counts.add_requests,
        "authorized_search_operations": len(shard.question_ids),
        "mem0_tool_implementation_sha256": tool_implementation_sha256(),
        "mem0_environment_lock_sha256": computed_mem0_environment_lock_sha256,
    }
    for field, actual in expected.items():
        if getattr(authorization, field) != actual:
            raise ValueError(f"retrieval authorization {field} mismatch")
    _require_count(
        authorization.authorized_add_operations,
        "authorized_add_operations",
        minimum=1,
    )
    _require_count(
        authorization.authorized_extraction_calls,
        "authorized_extraction_calls",
        minimum=1,
    )
    _require_count(
        authorization.mem0_provider_retries,
        "mem0_provider_retries",
    )
    if authorization.mem0_provider_retries != 0:
        raise ValueError("Mem0 extraction provider/SDK retries must be zero")
    if authorization.authorized_search_operations != LONGMEMEVAL_QUESTIONS_PER_SHARD:
        raise ValueError("retrieval stage requires exactly ten searches")


def _validate_scoring_authorization(
    shard: RawStressShard,
    authorization: ScoringStageAuthorization,
    *,
    computed_root_environment_lock_sha256: str,
) -> None:
    _validate_common_authorization(authorization)
    _validated_model_identities(authorization)
    for name in (
        "retrieval_artifact_sha256",
        "scoring_policy_sha256",
        "responder_model_identity_sha256",
        "judge_model_identity_sha256",
        "computed_root_environment_lock_sha256",
    ):
        value = (
            computed_root_environment_lock_sha256
            if name == "computed_root_environment_lock_sha256"
            else getattr(authorization, name)
        )
        _require_sha256(value, name)
    expected = {
        "sample_offset": shard.sample_offset,
        "sample_sha256": shard.sample_sha256,
        "raw_history_bundle_sha256": shard.raw_history_bundle_sha256,
        "question_ids": shard.question_ids,
        "source_environment_lock_sha256": computed_root_environment_lock_sha256,
        "mem0_tool_implementation_sha256": tool_implementation_sha256(),
    }
    for field, actual in expected.items():
        if getattr(authorization, field) != actual:
            raise ValueError(f"scoring authorization {field} mismatch")
    _require_nonempty(authorization.responder_model, "responder_model")
    _require_nonempty(authorization.judge_model, "judge_model")
    if authorization.responder_model == authorization.judge_model:
        raise ValueError("responder and judge models must be distinct")
    if authorization.authorized_responder_calls != LONGMEMEVAL_QUESTIONS_PER_SHARD:
        raise ValueError("scoring stage requires exactly ten responder calls")
    if authorization.authorized_judge_calls != LONGMEMEVAL_QUESTIONS_PER_SHARD:
        raise ValueError("scoring stage requires exactly ten judge calls")
    if authorization.provider_retries != 0:
        raise ValueError("scoring stage retries must remain frozen at zero")
    _require_count(authorization.max_prompt_tokens, "max_prompt_tokens", minimum=1)
    _require_count(
        authorization.responder_max_output_tokens,
        "responder_max_output_tokens",
        minimum=1,
    )
    _require_count(
        authorization.judge_max_output_tokens,
        "judge_max_output_tokens",
        minimum=1,
    )
    source_identity = _strict_json(authorization.source_evaluation_identity)
    if authorization.responder_model != source_identity["responder_model"]:
        raise ValueError("responder model diverges from source evaluation identity")
    if authorization.judge_model != source_identity["judge_model"]:
        raise ValueError("judge model diverges from source evaluation identity")
    if authorization.max_prompt_tokens != source_identity["max_prompt_tokens"]:
        raise ValueError("prompt cap diverges from source evaluation identity")
    if (
        authorization.responder_max_output_tokens
        != source_identity["responder_output_token_reserve"]
    ):
        raise ValueError("responder reserve diverges from source evaluation identity")


def composite_add_batch_to_prepared(batch: CompositeAddBatch) -> _PreparedBatch:
    """Losslessly cross the adapter seam without re-parsing normalized turns."""

    roles = tuple(role for role, _content in batch.messages)
    return _PreparedBatch(
        ref=SourceRef(
            sample_id=batch.source_sample_id,
            source=batch.source,
            session=batch.source,
            session_index=batch.session_index,
            original_session_index=batch.original_session_index,
            batch_index=batch.batch_index,
            date=batch.date,
            turn_start=batch.turn_start,
            turn_count=len(batch.messages),
            roles=roles,
        ),
        messages=batch.messages,
    )


def build_adapter_prepared_corpus(shard: RawStressShard) -> _PreparedCorpus:
    """Build the one private corpus consumed by ``Mem0Adapter._ingest_prepared``."""

    batches = tuple(composite_add_batch_to_prepared(row) for row in shard.add_batches)
    if len(batches) != shard.add_counts.add_requests:
        raise ValueError("prepared corpus add count diverged from raw protocol")
    return _PreparedCorpus(
        sample_id=shard.parsed_sample.sample_id,
        batches=batches,
        raw_pair_count=shard.add_counts.raw_pairs,
        skipped_empty_pair_count=shard.add_counts.skipped_empty_pairs,
        official_longmemeval_protocol=True,
    )


def _runtime_identity(
    value: Any,
    expected_stable_sha256: str,
    expected_stable_payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Mem0 runtime identity must be a mapping")
    identity = _strict_json(value)
    assert isinstance(identity, dict)
    allowed = {
        "protocol",
        "config",
        "stack",
        "stable_config_sha256",
        "effective_config_sha256",
        "local_owned_state",
        "on_disk",
        "certified",
    }
    if set(identity) != allowed:
        raise ValueError(
            "Mem0 runtime identity fields mismatch: "
            f"missing={sorted(allowed - set(identity))!r}, "
            f"extra={sorted(set(identity) - allowed)!r}"
        )
    _reject_secret_material(identity, path="runtime_identity")
    required = {
        "protocol": MEM0_RUNTIME_PROTOCOL,
        "stable_config_sha256": expected_stable_sha256,
        "local_owned_state": True,
        "on_disk": True,
        "certified": True,
    }
    for field, expected in required.items():
        if identity.get(field) != expected:
            raise ValueError(f"Mem0 runtime identity {field} mismatch")
    observed_payload = {
        "protocol": identity["protocol"],
        "config": identity["config"],
        "stack": identity["stack"],
    }
    if canonical_json_sha256(observed_payload) != expected_stable_sha256:
        raise ValueError("Mem0 runtime stable payload digest mismatch")
    if _canonical_bytes(observed_payload) != _canonical_bytes(
        expected_stable_payload
    ):
        raise ValueError("Mem0 runtime stable payload differs from policy")
    _require_sha256(identity.get("effective_config_sha256"), "effective_config_sha256")
    stack = identity.get("stack")
    if not isinstance(stack, dict):
        raise ValueError("Mem0 runtime identity omitted stack preflight")
    if stack.get("bm25_model") != MEM0_BM25_MODEL:
        raise ValueError("Mem0 BM25 model identity mismatch")
    if stack.get("spacy_model") != MEM0_SPACY_MODEL:
        raise ValueError("Mem0 spaCy model identity mismatch")
    if stack.get("bm25_operational") is not True:
        raise ValueError("Mem0 BM25 operational probe was not certified")
    if stack.get("entity_extraction_operational") is not True:
        raise ValueError("Mem0 entity/lemma operational probe was not certified")
    versions = stack.get("dependency_versions")
    if not isinstance(versions, dict) or versions.get("mem0ai") != MEM0AI_PIN:
        raise ValueError("Mem0 dependency stack does not pin mem0ai==2.0.18")
    return identity


def _stats_snapshot(value: Any) -> dict[str, Any]:
    names = Mem0AdapterStats.__dataclass_fields__.keys()
    return {name: _strict_json(getattr(value, name)) for name in names}


def _assert_initial_adapter(adapter: Any) -> None:
    if getattr(adapter, "active_user_scope", None) is not None:
        raise ValueError("Mem0 adapter must begin without an active scope")
    stats = getattr(adapter, "stats", None)
    if stats is None:
        raise TypeError("Mem0 adapter omitted stats")
    for field in (
        "add_calls",
        "add_attempted_calls",
        "add_completed_calls",
        "add_failed_calls",
        "search_calls",
    ):
        if getattr(stats, field, None) != 0:
            raise ValueError(f"Mem0 adapter initial {field} must be zero")
    if not callable(getattr(adapter, "_ingest_prepared", None)):
        raise TypeError("Mem0 adapter omitted the prepared-corpus seam")
    if not callable(getattr(adapter, "search", None)):
        raise TypeError("Mem0 adapter omitted search")
    if not callable(getattr(adapter, "cleanup", None)):
        raise TypeError("Mem0 adapter omitted cleanup")


def _pre_ingest_runtime_identity(
    adapter: Any,
    authorization: RetrievalStageAuthorization,
) -> dict[str, Any]:
    """Materialize and verify the frozen runtime before the first add call.

    The concrete adapter lazily creates ``Memory.from_config``.  Pulling that
    boundary forward lets the runner reject a config/stack mismatch before
    any metered extraction call or memory write.  This is defense in depth for
    the injected core; a production issuer must additionally prove that its
    constructor and local embedder cannot make unmetered external calls.
    """

    backend_loader = getattr(adapter, "_get_backend", None)
    identity_reader = getattr(adapter, "_runtime_identity_snapshot", None)
    if not callable(backend_loader) or not callable(identity_reader):
        raise TypeError(
            "Mem0 adapter omitted the pre-ingest runtime identity boundary"
        )
    if backend_loader() is None:
        raise ValueError("Mem0 adapter failed to materialize its backend")
    return _runtime_identity(
        identity_reader(),
        authorization.mem0_stable_config_sha256,
        authorization.mem0_stable_payload,
    )


def _post_cleanup_adapter_state(
    adapter: Any, *, state_target: Path
) -> dict[str, bool]:
    try:
        ledger_empty = len(getattr(adapter, "ledger")) == 0
    except BaseException:
        ledger_empty = False
    scopes = getattr(adapter, "_scopes", None)
    scope_protocol = getattr(adapter, "_scope_protocol", None)
    backend = getattr(adapter, "_backend", object())
    return {
        "active_scope_cleared": getattr(adapter, "active_user_scope", None) is None,
        "adapter_closed": getattr(adapter, "_closed", None) is True,
        "ledger_empty": ledger_empty,
        "registered_scopes_empty": isinstance(scopes, list) and len(scopes) == 0,
        "scope_protocol_empty": (
            isinstance(scope_protocol, dict) and len(scope_protocol) == 0
        ),
        "backend_closed_or_cleared": (
            backend is None or getattr(backend, "_closed", None) is True
        ),
        "owned_state_path_absent": not state_target.exists(),
    }


def _candidate_payload(candidate: Any, expected_rank: int) -> dict[str, Any]:
    rank = getattr(candidate, "rank", None)
    if rank != expected_rank:
        raise ValueError("Mem0 raw-pool ranks must be contiguous and stable")
    memory_id = _require_nonempty(getattr(candidate, "memory_id", None), "memory_id")
    text = getattr(candidate, "text", None)
    if not isinstance(text, str):
        raise TypeError("Mem0 candidate text must be str")
    score_value = getattr(candidate, "score", None)
    if score_value is not None:
        if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
            raise TypeError("Mem0 candidate score must be numeric or null")
        score_value = float(score_value)
        if not math.isfinite(score_value):
            raise ValueError("Mem0 candidate score must be finite")
    created_at = _require_nonempty(
        getattr(candidate, "created_at", None), "candidate created_at"
    )
    if getattr(candidate, "attribution_kind", None) != MEM0_ATTRIBUTION_KIND:
        raise ValueError("Mem0 candidate attribution kind mismatch")
    return {
        "rank": rank,
        "memory_id": memory_id,
        "text": text,
        "score": score_value,
        "created_at": created_at,
        "attribution_kind": MEM0_ATTRIBUTION_KIND,
    }


def _default_prompt_packer(
    question: str,
    result: Any,
    *,
    max_prompt_tokens: int,
    evaluation_identity: Mapping[str, Any],
) -> Any:
    from .prompt_pack import pack_mem0_prompt

    return pack_mem0_prompt(
        question,
        result,
        max_prompt_tokens=max_prompt_tokens,
        evaluation_identity=evaluation_identity,
    )


def _message_payload(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise TypeError("packed messages must be a sequence")
    rows: list[dict[str, str]] = []
    for index, value in enumerate(messages):
        if not isinstance(value, Mapping) or set(value) != {"role", "content"}:
            raise ValueError(f"packed message {index} has an invalid shape")
        role = value.get("role")
        content = value.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("packed message role/content must be str")
        rows.append({"role": role, "content": content})
    if len(rows) != 2:
        raise ValueError("the Mem0 QA prompt must contain exactly two messages")
    return rows


def _pack_payload(
    pack: Any,
    raw_pool: list[dict[str, Any]],
    *,
    question_id: str,
    search_latency_s: float,
) -> dict[str, Any]:
    """Use the prompt packer's canonical, self-hashed retrieval-row contract."""

    row_builder = getattr(pack, "to_retrieval_row", None)
    if not callable(row_builder):
        raise TypeError("prompt packer result omitted to_retrieval_row")
    row = _strict_json(
        row_builder(
            question_id=question_id,
            search_latency_s=search_latency_s,
        )
    )
    if not isinstance(row, dict):
        raise TypeError("prompt packer retrieval row must be an object")
    if row.get("raw_pool") != raw_pool:
        raise ValueError("prompt packer changed the sanitized raw pool")
    if row.get("raw_pool_sha256") != canonical_json_sha256(raw_pool):
        raise ValueError("prompt packer raw-pool digest mismatch")
    messages = _message_payload(row.get("messages"))
    if row.get("messages_sha256") != canonical_json_sha256(messages):
        raise ValueError("prompt packer messages digest mismatch")
    prompt_tokens = _require_count(
        row.get("prompt_token_proxy"), "prompt_token_proxy"
    )
    max_tokens = _require_count(
        row.get("max_prompt_token_proxy"),
        "max_prompt_token_proxy",
        minimum=1,
    )
    residual = _require_count(
        row.get("residual_prompt_token_proxy"),
        "residual_prompt_token_proxy",
    )
    if prompt_tokens > max_tokens or residual != max_tokens - prompt_tokens:
        raise ValueError("prompt packer returned inconsistent token accounting")
    from .prompt_pack import (
        MEM0_EFFECTIVE_RECENT_WINDOW,
        MEM0_PROMPT_PACK_PROTOCOL,
        MEM0_RECENT_WINDOW_SEMANTICS,
    )

    source_identity = row.get("source_evaluation_identity")
    if not isinstance(source_identity, Mapping):
        raise TypeError("prompt packer omitted source_evaluation_identity")
    configured_recent_window = _require_count(
        row.get("configured_recent_window"),
        "configured_recent_window",
    )
    if configured_recent_window != source_identity.get("recent_window"):
        raise ValueError(
            "prompt packer configured recent window differs from source policy"
        )
    if row.get("effective_recent_window") != MEM0_EFFECTIVE_RECENT_WINDOW:
        raise ValueError(
            "LongMemEval prompt packer must use no live recent-turn tail"
        )
    if row.get("recent_window_semantics") != MEM0_RECENT_WINDOW_SEMANTICS:
        raise ValueError("prompt packer recent-window semantics mismatch")
    if row.get("prompt_pack_protocol") != MEM0_PROMPT_PACK_PROTOCOL:
        raise ValueError("prompt packer protocol mismatch")
    row_digest = row.pop("retrieval_row_sha256", None)
    _require_sha256(row_digest, "retrieval_row_sha256")
    if canonical_json_sha256(row) != row_digest:
        raise ValueError("prompt packer retrieval-row digest mismatch")
    row["retrieval_row_sha256"] = row_digest
    return row


def _result_protocol_identity(
    result: Any,
    expected_stable: str,
    expected_stable_payload: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "official_longmemeval_protocol": True,
        "official_search_protocol": True,
        "certified_rendering": True,
        "comparison_certified": True,
        "supports_exact_source_provenance": False,
        "rendering_mode": MEM0_CERTIFIED_RENDERING,
        "attribution_kind": MEM0_ATTRIBUTION_KIND,
    }
    for field, expected in required.items():
        if getattr(result, field, None) != expected:
            raise ValueError(f"Mem0 search result {field} mismatch")
    return _runtime_identity(
        getattr(result, "runtime_identity", None),
        expected_stable,
        expected_stable_payload,
    )


def _failure_trace(
    *,
    trace_format: str,
    stage: str,
    shard: RawStressShard,
    events: list[dict[str, Any]],
    started: float,
    error: BaseException,
    cleanup: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": trace_format,
        "status": "failed",
        "stage": stage,
        "sample_offset": shard.sample_offset,
        "sample_id": shard.parsed_sample.sample_id,
        "sample_sha256": shard.sample_sha256,
        "events": events,
        "cleanup": _strict_json(cleanup or {}),
        "error_type": type(error).__name__,
        "elapsed_s": max(0.0, time.perf_counter() - started),
    }


def run_retrieval_stage(
    *,
    shard: RawStressShard,
    authorization: RetrievalStageAuthorization,
    mem0_environment_lock_path: str | os.PathLike[str],
    owned_state_dir: str | os.PathLike[str],
    artifact_path: str | os.PathLike[str],
    trace_path: str | os.PathLike[str],
    adapter_factory: AdapterFactory,
    max_prompt_tokens: int = 8_000,
    prompt_packer: Callable[..., Any] = _default_prompt_packer,
    extraction_meter_installer: ExtractionMeterInstaller = (
        install_memory_llm_extraction_meter
    ),
    trusted_runtime_binding: TrustedRuntimeBinding | None = None,
    process_guard: ShardProcessGuard = _RETRIEVAL_PROCESS_GUARD,
) -> RetrievalStageResult:
    """Execute isolated Mem0 ingest/retrieval and publish only after cleanup."""

    try:
        validate_raw_stress_shard(shard)
    except BaseException as exc:
        raise Mem0ShardRunError(
            "Mem0 retrieval shard failed independent content validation",
            stage="retrieval",
        ) from exc
    try:
        environment_lock_target, environment_lock_before = (
            _environment_lock_snapshot(
                mem0_environment_lock_path,
                label="mem0_environment_lock_path",
            )
        )
        execution_binding = _execution_binding_receipt(
            trusted_runtime_binding,
            stage="retrieval",
            authorization=authorization,
            bound_callables=(
                adapter_factory,
                prompt_packer,
                extraction_meter_installer,
                process_guard,
            ),
        )
    except BaseException as exc:
        raise Mem0ShardRunError(
            "Mem0 retrieval execution boundary could not be verified",
            stage="retrieval",
        ) from exc
    started = time.perf_counter()
    artifact_target = Path(artifact_path).resolve(strict=False)
    trace_target = Path(trace_path).resolve(strict=False)
    state_target = Path(owned_state_dir).resolve(strict=False)
    events: list[dict[str, Any]] = []
    adapter: Any | None = None
    extraction_meter: LogicalExtractionCallMeter | None = None
    restore_extraction_meter: Callable[[], Any] | None = None
    cleanup_receipt: dict[str, Any] = {
        "attempted": False,
        "completed": False,
        "state_absent_before": not state_target.exists(),
        "state_absent_after": False,
        "active_scope_cleared": False,
        "extraction_meter_restore_attempted": False,
        "extraction_meter_restored_before_cleanup": False,
    }
    operation_error: BaseException | None = None
    artifact_payload: dict[str, Any] | None = None
    environment_lock_after: str | None = None
    try:
        _validate_retrieval_authorization(
            shard,
            authorization,
            computed_mem0_environment_lock_sha256=(
                environment_lock_before
            ),
        )
        extraction_identity, embedder_identity = _validated_model_identities(
            authorization
        )
        _require_count(max_prompt_tokens, "max_prompt_tokens", minimum=1)
        if max_prompt_tokens != 8_000:
            raise ValueError("the comparison prompt cap must be exactly 8000")
        if not callable(adapter_factory):
            raise TypeError("adapter_factory must be callable")
        if not callable(prompt_packer):
            raise TypeError("prompt_packer must be callable")
        if not callable(extraction_meter_installer):
            raise TypeError("extraction_meter_installer must be callable")
        if (
            _is_equal_to_or_within(artifact_target, trace_target)
            or _is_equal_to_or_within(trace_target, artifact_target)
        ):
            raise ValueError("artifact and trace paths must be distinct and non-nested")
        _reject_protected_outputs(
            (artifact_target, trace_target),
            (
                (state_target, "owned_state_dir"),
                (environment_lock_target, "Mem0 environment lock input"),
                *_implementation_roots(),
            ),
            label="Stage A",
        )
        if any(
            _is_equal_to_or_within(state_target, output)
            for output in (artifact_target, trace_target)
        ):
            raise ValueError("owned_state_dir cannot descend from a Stage A output")
        if state_target.exists():
            raise FileExistsError("owned_state_dir must not exist before Stage A")
        if artifact_target.exists() or trace_target.exists():
            raise FileExistsError("Stage A outputs must not already exist")
        process_guard.claim()
        events.append({"sequence": 1, "event": "authorization_verified"})

        corpus = build_adapter_prepared_corpus(shard)
        events.append(
            {
                "sequence": 2,
                "event": "prepared_corpus_built",
                "batches": len(corpus.batches),
                "ordered_batch_hashes": [
                    canonical_json_sha256(
                        {
                            "source_ref": batch.ref.metadata,
                            "messages": list(batch.messages),
                        }
                    )
                    for batch in corpus.batches
                ],
            }
        )
        adapter = adapter_factory(state_target)
        _assert_initial_adapter(adapter)
        pre_ingest_runtime_identity = _pre_ingest_runtime_identity(
            adapter, authorization
        )
        extraction_meter = LogicalExtractionCallMeter(
            authorized=authorization.authorized_extraction_calls
        )
        restore_extraction_meter = extraction_meter_installer(
            adapter, extraction_meter
        )
        if not callable(restore_extraction_meter):
            raise TypeError("extraction meter installer omitted its restore hook")
        events.append(
            {
                "sequence": 3,
                "event": "extraction_meter_installed",
                "boundary": "Memory.llm.generate_response",
                "authorized_logical_calls": (
                    authorization.authorized_extraction_calls
                ),
                "authorized_local_wrapper_retries": (
                    authorization.mem0_provider_retries
                ),
                "external_http_attempts_certified": False,
                "external_retry_attempts_certified": False,
            }
        )
        ingest = adapter._ingest_prepared(corpus)
        extraction_meter.assert_complete()
        if getattr(ingest, "comparison_certified", None) is not True:
            raise ValueError("Mem0 ingest was not comparison-certified")
        if getattr(ingest, "official_longmemeval_protocol", None) is not True:
            raise ValueError("Mem0 ingest did not retain the official protocol")
        if getattr(ingest, "supports_exact_source_provenance", None) is not False:
            raise ValueError("Mem0 exact-provenance capability was overstated")
        runtime_identity = _runtime_identity(
            getattr(ingest, "runtime_identity", None),
            authorization.mem0_stable_config_sha256,
            authorization.mem0_stable_payload,
        )
        if canonical_json_sha256(runtime_identity) != canonical_json_sha256(
            pre_ingest_runtime_identity
        ):
            raise ValueError("Mem0 runtime identity changed during ingest")
        ingest_stats = getattr(ingest, "stats", None)
        if ingest_stats is None:
            raise ValueError("Mem0 ingest omitted stats")
        exact_adds = authorization.authorized_add_operations
        for field, expected in {
            "add_calls": exact_adds,
            "add_attempted_calls": exact_adds,
            "add_completed_calls": exact_adds,
            "add_failed_calls": 0,
        }.items():
            if getattr(ingest_stats, field, None) != expected:
                raise ValueError(f"Mem0 ingest {field} mismatch")
        user_scope = _require_nonempty(getattr(ingest, "user_scope", None), "user_scope")
        if getattr(adapter, "active_user_scope", None) != user_scope:
            raise ValueError("Mem0 adapter active scope diverged after ingest")
        if len(getattr(ingest, "batches_added", ())) != exact_adds:
            raise ValueError("Mem0 ingest batch receipt count mismatch")
        events.append(
            {
                "sequence": 4,
                "event": "ingest_complete",
                "add_operations": exact_adds,
                "logical_extraction_calls": extraction_meter.completed,
                "user_scope_sha256": hashlib.sha256(user_scope.encode()).hexdigest(),
            }
        )

        retrieval_rows: list[dict[str, Any]] = []
        previous_search_latency = 0.0
        for question_index, question in enumerate(
            shard.parsed_sample.questions, start=1
        ):
            query = question.dated_question
            # The adapter's own pack is intentionally irrelevant here.  This
            # renderer keeps all official raw candidates available; the
            # independent packer below owns the actual 8k QA prompt.
            search_result = adapter.search(
                query,
                max_prompt_tokens=1_000_000_000,
                prompt_renderer=lambda rendered_query, _context: rendered_query,
                user_scope=user_scope,
                threshold=MEM0_OFFICIAL_THRESHOLD,
                rendering_mode=MEM0_CERTIFIED_RENDERING,
            )
            search_runtime = _result_protocol_identity(
                search_result,
                authorization.mem0_stable_config_sha256,
                authorization.mem0_stable_payload,
            )
            if canonical_json_sha256(search_runtime) != canonical_json_sha256(runtime_identity):
                raise ValueError("Mem0 runtime identity changed during the shard")
            raw_pool = [
                _candidate_payload(candidate, rank)
                for rank, candidate in enumerate(
                    getattr(search_result, "raw_pool", ()), start=1
                )
            ]
            if len({row["memory_id"] for row in raw_pool}) != len(raw_pool):
                raise ValueError("Mem0 raw pool repeated a memory ID")
            current_search_latency = float(
                getattr(
                    getattr(search_result, "stats", None),
                    "search_latency_s",
                    0.0,
                )
            )
            search_latency = max(
                0.0, current_search_latency - previous_search_latency
            )
            previous_search_latency = current_search_latency
            packed = prompt_packer(
                query,
                search_result,
                max_prompt_tokens=max_prompt_tokens,
                evaluation_identity=authorization.source_evaluation_identity,
            )
            row = _pack_payload(
                packed,
                raw_pool,
                question_id=question.question_id,
                search_latency_s=search_latency,
            )
            retrieval_rows.append(row)
            events.append(
                {
                    "sequence": 4 + question_index,
                    "event": "search_complete",
                    "question_id": question.question_id,
                    "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                    "raw_memory_count": row["raw_memory_count"],
                    "raw_pool_sha256": row["raw_pool_sha256"],
                    "retrieval_row_sha256": row["retrieval_row_sha256"],
                }
            )

        final_stats = getattr(adapter, "stats", None)
        if getattr(final_stats, "search_calls", None) != authorization.authorized_search_operations:
            raise ValueError("Mem0 search call count mismatch")
        if getattr(final_stats, "add_attempted_calls", None) != authorization.authorized_add_operations:
            raise ValueError("Mem0 add count changed after ingest")
        # Keep the boundary installed through retrieval. A Mem0 search must
        # not make or swallow an unbudgeted extraction-model call after the
        # post-ingest check already closed the authorized add sequence.
        extraction_meter.assert_complete()
        artifact_payload = {
            "format": RETRIEVAL_ARTIFACT_FORMAT,
            "status": "complete",
            "certification_status": "injected_nonproduction",
            "comparison_certified": execution_binding["comparison_certified"],
            "execution_binding": execution_binding,
            "sample_offset": shard.sample_offset,
            "sample_id": shard.parsed_sample.sample_id,
            "sample_sha256": shard.sample_sha256,
            "raw_history_bundle_sha256": shard.raw_history_bundle_sha256,
            "history_sample_ids_sha256": canonical_json_sha256(
                list(shard.history_sample_ids)
            ),
            "question_ids": list(shard.question_ids),
            "question_ids_sha256": canonical_json_sha256(list(shard.question_ids)),
            "identity": {
                "source_validation_policy_sha256": authorization.source_validation_policy_sha256,
                "source_implementation_sha256": authorization.source_implementation_sha256,
                "source_environment_lock_sha256": authorization.source_environment_lock_sha256,
                "mem0_policy_sha256": authorization.mem0_policy_sha256,
                "mem0_tool_implementation_sha256": authorization.mem0_tool_implementation_sha256,
                "mem0_environment_lock_sha256": authorization.mem0_environment_lock_sha256,
                "mem0_stable_config_sha256": authorization.mem0_stable_config_sha256,
                "extraction_model_identity": extraction_identity,
                "extraction_model_identity_sha256": canonical_json_sha256(
                    extraction_identity
                ),
                "embedder_model_identity": embedder_identity,
                "embedder_model_identity_sha256": canonical_json_sha256(
                    embedder_identity
                ),
                "runtime_model_identity_probe": {
                    "kind": "unavailable_injected_nonproduction",
                    "extraction_model_identity_sha256": canonical_json_sha256(
                        extraction_identity
                    ),
                    "embedder_model_identity_sha256": canonical_json_sha256(
                        embedder_identity
                    ),
                    "before_match": False,
                    "after_match": False,
                    "comparison_certified": False,
                },
                "source_evaluation_identity": _strict_json(
                    authorization.source_evaluation_identity
                ),
                "source_evaluation_identity_sha256": canonical_json_sha256(
                    authorization.source_evaluation_identity
                ),
                "runtime_identity": {
                    **runtime_identity,
                    "persisted_request_token_state": False,
                    "retained_request_token_state_bytes": 0,
                    "request_token_state_evidence_kind": (
                        "local_injected_request_token_state_contract"
                    ),
                    "external_provider_persistence_certified": False,
                },
            },
            "protocol": {
                "input_order": INPUT_ORDER_PROTOCOL,
                "official_longmemeval_protocol": True,
                "official_search_protocol": True,
                "top_k": MEM0_OFFICIAL_TOP_K,
                "threshold": MEM0_OFFICIAL_THRESHOLD,
                "rendering_mode": MEM0_CERTIFIED_RENDERING,
                "max_prompt_tokens": max_prompt_tokens,
            },
            "raw_input_receipt": shard_receipt(shard),
            "ingestion_receipt": {
                "raw_pairs": shard.add_counts.raw_pairs,
                "skipped_empty_pairs": shard.add_counts.skipped_empty_pairs,
                "authorized_add_operations": authorization.authorized_add_operations,
                "attempted_add_operations": getattr(final_stats, "add_attempted_calls"),
                "completed_add_operations": getattr(final_stats, "add_completed_calls"),
                "failed_add_operations": getattr(final_stats, "add_failed_calls"),
                "extraction_model_calls": extraction_meter.receipt(),
                "persisted_request_token_state": False,
                "retained_request_token_state_bytes": 0,
                "request_token_state_evidence_kind": (
                    "local_injected_request_token_state_contract"
                ),
                "external_provider_persistence_certified": False,
                "one_scope": True,
                "user_scope_sha256": hashlib.sha256(user_scope.encode()).hexdigest(),
                "comparison_certified": execution_binding[
                    "comparison_certified"
                ],
            },
            "search_receipt": {
                "authorized_search_operations": authorization.authorized_search_operations,
                "completed_search_operations": getattr(final_stats, "search_calls"),
                "failed_search_operations": 0,
            },
            "mem0_usage": _stats_snapshot(final_stats),
            "provenance": {
                "attribution_kind": MEM0_ATTRIBUTION_KIND,
                "supports_exact_source_provenance": False,
                "source_session_date_exposure": (
                    "diagnostics_only_not_model_input"
                ),
                "retrieved_created_at_exposure": "answer_prompt_date_headings",
                "provider_usage_status": MEM0_PROVIDER_USAGE_STATUS,
                "external_http_attempts_certified": False,
                "external_retry_attempts_certified": False,
                "external_provider_persistence_certified": False,
            },
            "retrieval_rows": retrieval_rows,
        }
    except BaseException as exc:
        operation_error = exc
    finally:
        if extraction_meter is not None:
            try:
                extraction_meter.verify_request_token_state()
            except BaseException as state_exc:
                if operation_error is None:
                    operation_error = state_exc
                else:
                    operation_error.add_note(
                        "Mem0 extraction request-token-state verification also "
                        f"failed: {type(state_exc).__name__}"
                    )
            cleanup_receipt["extraction_model_calls"] = extraction_meter.receipt()
            meter_state = extraction_meter.receipt()
            cleanup_receipt["persisted_request_token_state"] = meter_state[
                "persisted_request_token_state"
            ]
            cleanup_receipt["retained_request_token_state_bytes"] = meter_state[
                "retained_request_token_state_bytes"
            ]
            cleanup_receipt["request_token_state_evidence_kind"] = meter_state[
                "request_token_state_evidence_kind"
            ]
            cleanup_receipt[
                "external_provider_persistence_certified"
            ] = False
        if restore_extraction_meter is not None:
            cleanup_receipt["extraction_meter_restore_attempted"] = True
            try:
                restore_extraction_meter()
                cleanup_receipt[
                    "extraction_meter_restored_before_cleanup"
                ] = True
            except BaseException as restore_exc:
                cleanup_receipt["extraction_meter_restore_error_type"] = type(
                    restore_exc
                ).__name__
                if operation_error is None:
                    operation_error = restore_exc
                else:
                    operation_error.add_note(
                        "Mem0 extraction-meter restoration also failed: "
                        f"{type(restore_exc).__name__}"
                    )
        cleanup_receipt["attempted"] = adapter is not None
        if adapter is not None:
            try:
                adapter.cleanup()
                cleanup_receipt["completed"] = True
            except BaseException as cleanup_exc:
                cleanup_receipt["cleanup_error_type"] = type(cleanup_exc).__name__
                if operation_error is None:
                    operation_error = cleanup_exc
                else:
                    operation_error.add_note(
                        f"Mem0 cleanup also failed: {type(cleanup_exc).__name__}"
                    )
            observed_state = _post_cleanup_adapter_state(
                adapter, state_target=state_target
            )
            cleanup_receipt.update(observed_state)
            if not all(observed_state.values()) and operation_error is None:
                operation_error = RuntimeError(
                    "Mem0 in-process state remained after cleanup"
                )
        cleanup_receipt["state_absent_after"] = not state_target.exists()
        if not cleanup_receipt["state_absent_after"] and operation_error is None:
            operation_error = RuntimeError("owned Mem0 state remained after cleanup")
        if adapter is not None and not cleanup_receipt["active_scope_cleared"] and operation_error is None:
            operation_error = RuntimeError("Mem0 scope remained active after cleanup")
        try:
            observed_lock_target, environment_lock_after = (
                _environment_lock_snapshot(
                    environment_lock_target,
                    label="mem0_environment_lock_path post-stage",
                )
            )
            if observed_lock_target != environment_lock_target:
                raise RuntimeError("Mem0 environment lock path identity changed")
            if environment_lock_after != environment_lock_before:
                raise RuntimeError("Mem0 environment lock bytes changed during Stage A")
        except BaseException as lock_exc:
            if operation_error is None:
                operation_error = lock_exc
            else:
                operation_error.add_note(
                    "Mem0 environment-lock recheck also failed: "
                    f"{type(lock_exc).__name__}"
                )
        cleanup_receipt["environment_lock"] = {
            "filename": environment_lock_target.name,
            "authorized_sha256": authorization.mem0_environment_lock_sha256,
            "sha256_before": environment_lock_before,
            "sha256_after": environment_lock_after,
            "unchanged": environment_lock_after == environment_lock_before,
        }

    if operation_error is not None:
        failed = _failure_trace(
            trace_format=RETRIEVAL_TRACE_FORMAT,
            stage="retrieval",
            shard=shard,
            events=events,
            started=started,
            error=operation_error,
            cleanup=cleanup_receipt,
        )
        written_trace: Path | None = None
        try:
            # An unsafe trace target must not modify state or a frozen input
            # after the validation/cleanup proof that rejected it.
            trace_is_safe = all(
                not _is_equal_to_or_within(trace_target, protected_path)
                for protected_path, _label in (
                    (state_target, "owned_state_dir"),
                    (environment_lock_target, "Mem0 environment lock input"),
                    *_implementation_roots(),
                )
            )
            if trace_is_safe:
                _atomic_create_json(trace_target, failed)
                written_trace = trace_target
        except BaseException as trace_exc:
            operation_error.add_note(
                f"failure trace could not be written: {type(trace_exc).__name__}"
            )
        raise Mem0ShardRunError(
            "Mem0 retrieval stage failed closed",
            stage="retrieval",
            trace_path=written_trace,
        ) from operation_error

    assert artifact_payload is not None
    assert extraction_meter is not None
    artifact_payload["ingestion_receipt"][
        "extraction_model_calls"
    ] = extraction_meter.receipt()
    artifact_payload["environment_lock"] = cleanup_receipt["environment_lock"]
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "cleanup_complete",
            "state_absent_after": cleanup_receipt["state_absent_after"],
        }
    )
    trace = {
        "format": RETRIEVAL_TRACE_FORMAT,
        "status": "complete",
        "certification_status": "injected_nonproduction",
        "comparison_certified": execution_binding["comparison_certified"],
        "execution_binding": execution_binding,
        "stage": "retrieval",
        "sample_offset": shard.sample_offset,
        "sample_id": shard.parsed_sample.sample_id,
        "sample_sha256": shard.sample_sha256,
        "events": events,
        "cleanup": cleanup_receipt,
        "environment_lock": cleanup_receipt["environment_lock"],
        "elapsed_s": max(0.0, time.perf_counter() - started),
    }
    try:
        trace_payload = _render_json_bytes(trace)
        trace_sha = hashlib.sha256(trace_payload).hexdigest()
        trace_bytes = len(trace_payload)
        artifact_payload["retrieval_trace"] = {
            "filename": trace_target.name,
            "sha256": trace_sha,
            "bytes": trace_bytes,
        }
        artifact_payload["content_sha256"] = canonical_json_sha256(artifact_payload)
        artifact_file_payload = _render_json_bytes(artifact_payload)
        receipts = _atomic_create_payloads(
            (
                (trace_target, trace_payload),
                (artifact_target, artifact_file_payload),
            )
        )
        (published_trace_sha, published_trace_bytes), (
            artifact_sha,
            artifact_bytes,
        ) = receipts
        if (published_trace_sha, published_trace_bytes) != (
            trace_sha,
            trace_bytes,
        ):
            raise AssertionError("published Stage A trace receipt changed")
    except BaseException as exc:
        raise Mem0ShardRunError(
            "Mem0 retrieval receipts could not be published atomically",
            stage="retrieval",
            trace_path=None,
        ) from exc
    return RetrievalStageResult(
        artifact_path=artifact_target,
        trace_path=trace_target,
        artifact_sha256=artifact_sha,
        artifact_bytes=artifact_bytes,
        artifact=MappingProxyType(artifact_payload),
        trace=MappingProxyType(trace),
    )


def _verify_content_sha(payload: Mapping[str, Any], *, label: str) -> None:
    expected = payload.get("content_sha256")
    _require_sha256(expected, f"{label}.content_sha256")
    body = dict(payload)
    del body["content_sha256"]
    if canonical_json_sha256(body) != expected:
        raise Mem0ShardRunError(f"{label} content digest mismatch", stage="scoring")


def _frozen_search_result(
    artifact: Mapping[str, Any], row: Mapping[str, Any], query: str
) -> Any:
    candidates = tuple(
        SimpleNamespace(**dict(candidate)) for candidate in row["raw_pool"]
    )
    identity = artifact["identity"]
    return SimpleNamespace(
        query=query,
        raw_pool=candidates,
        official_longmemeval_protocol=True,
        official_search_protocol=True,
        rendering_mode=MEM0_CERTIFIED_RENDERING,
        certified_rendering=True,
        comparison_certified=True,
        runtime_identity=identity["runtime_identity"],
        attribution_kind=MEM0_ATTRIBUTION_KIND,
        supports_exact_source_provenance=False,
    )


def _verify_retrieval_artifact(
    *,
    artifact_path: Path,
    trace_path: Path,
    shard: RawStressShard,
    authorization: ScoringStageAuthorization,
    prompt_packer: Callable[..., Any],
) -> tuple[dict[str, Any], bytes, list[tuple[dict[str, Any], Any]]]:
    artifact, artifact_bytes = _load_json_object(artifact_path)
    file_digest = hashlib.sha256(artifact_bytes).hexdigest()
    if file_digest != authorization.retrieval_artifact_sha256:
        raise Mem0ShardRunError("retrieval artifact file digest mismatch", stage="scoring")
    _verify_content_sha(artifact, label="retrieval artifact")
    required = {
        "format": RETRIEVAL_ARTIFACT_FORMAT,
        "status": "complete",
        "certification_status": "injected_nonproduction",
        "comparison_certified": False,
        "execution_binding": _execution_binding_receipt(None),
        "sample_offset": shard.sample_offset,
        "sample_id": shard.parsed_sample.sample_id,
        "sample_sha256": shard.sample_sha256,
        "raw_history_bundle_sha256": shard.raw_history_bundle_sha256,
        "question_ids": list(shard.question_ids),
    }
    for field, expected in required.items():
        if artifact.get(field) != expected:
            raise Mem0ShardRunError(
                f"retrieval artifact {field} mismatch", stage="scoring"
            )
    identity = artifact.get("identity")
    if not isinstance(identity, dict):
        raise Mem0ShardRunError("retrieval artifact omitted identity", stage="scoring")
    identity_expected = {
        "source_validation_policy_sha256": authorization.source_validation_policy_sha256,
        "source_implementation_sha256": authorization.source_implementation_sha256,
        "source_environment_lock_sha256": authorization.source_environment_lock_sha256,
        "mem0_policy_sha256": authorization.mem0_policy_sha256,
        "mem0_tool_implementation_sha256": authorization.mem0_tool_implementation_sha256,
        "mem0_environment_lock_sha256": authorization.mem0_environment_lock_sha256,
        "mem0_stable_config_sha256": authorization.mem0_stable_config_sha256,
        "extraction_model_identity": _strict_json(
            authorization.extraction_model_identity
        ),
        "extraction_model_identity_sha256": (
            authorization.extraction_model_identity_sha256
        ),
        "embedder_model_identity": _strict_json(
            authorization.embedder_model_identity
        ),
        "embedder_model_identity_sha256": (
            authorization.embedder_model_identity_sha256
        ),
        "source_evaluation_identity": _strict_json(
            authorization.source_evaluation_identity
        ),
        "source_evaluation_identity_sha256": canonical_json_sha256(
            authorization.source_evaluation_identity
        ),
    }
    for field, expected in identity_expected.items():
        if identity.get(field) != expected:
            raise Mem0ShardRunError(
                f"retrieval artifact identity {field} mismatch", stage="scoring"
            )
    if identity.get("runtime_model_identity_probe") != {
        "kind": "unavailable_injected_nonproduction",
        "extraction_model_identity_sha256": (
            authorization.extraction_model_identity_sha256
        ),
        "embedder_model_identity_sha256": (
            authorization.embedder_model_identity_sha256
        ),
        "before_match": False,
        "after_match": False,
        "comparison_certified": False,
    }:
        raise Mem0ShardRunError(
            "retrieval runtime model-identity probe mismatch", stage="scoring"
        )
    runtime_value = identity.get("runtime_identity")
    if not isinstance(runtime_value, dict):
        raise Mem0ShardRunError(
            "retrieval artifact runtime identity is invalid", stage="scoring"
        )
    runtime_identity = dict(runtime_value)
    runtime_base = dict(runtime_identity)
    for field in (
        "persisted_request_token_state",
        "retained_request_token_state_bytes",
        "request_token_state_evidence_kind",
        "external_provider_persistence_certified",
    ):
        runtime_base.pop(field, None)
    _runtime_identity(
        runtime_base,
        authorization.mem0_stable_config_sha256,
        authorization.mem0_stable_payload,
    )
    if (
        runtime_identity.get("persisted_request_token_state") is not False
        or runtime_identity.get("retained_request_token_state_bytes") != 0
        or runtime_identity.get("request_token_state_evidence_kind")
        != "local_injected_request_token_state_contract"
        or runtime_identity.get("external_provider_persistence_certified")
        is not False
    ):
        raise Mem0ShardRunError(
            "retrieval runtime request-token-state proof mismatch",
            stage="scoring",
        )
    environment_lock = artifact.get("environment_lock")
    if not isinstance(environment_lock, dict):
        raise Mem0ShardRunError(
            "retrieval artifact omitted environment-lock receipt", stage="scoring"
        )
    for field, expected in {
        "authorized_sha256": authorization.mem0_environment_lock_sha256,
        "sha256_before": authorization.mem0_environment_lock_sha256,
        "sha256_after": authorization.mem0_environment_lock_sha256,
        "unchanged": True,
    }.items():
        if environment_lock.get(field) != expected:
            raise Mem0ShardRunError(
                f"retrieval environment-lock receipt {field} mismatch",
                stage="scoring",
            )
    _require_nonempty(environment_lock.get("filename"), "environment lock filename")
    ingestion_receipt = artifact.get("ingestion_receipt")
    if not isinstance(ingestion_receipt, dict):
        raise Mem0ShardRunError(
            "retrieval artifact omitted ingestion receipt", stage="scoring"
        )
    extraction_calls = ingestion_receipt.get("extraction_model_calls")
    exact_adds = shard.add_counts.add_requests
    expected_extraction = {
        "kind": "mem0_memory_llm_generate_response_logical_calls",
        "boundary": "Memory.llm.generate_response",
        "count_semantics": "local_logical_wrapper_calls_not_http_attempts",
        "external_http_attempts_certified": False,
        "authorized_local_wrapper_retries": 0,
        "external_retry_attempts_certified": False,
        "authorized": exact_adds,
        "attempted": exact_adds,
        "completed": exact_adds,
        "failed": 0,
        "rejected": 0,
        "infer_true_adds_started": exact_adds,
        "infer_true_adds_exactly_one_call": exact_adds,
        "one_logical_call_per_infer_true_add_certified": True,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
    }
    if extraction_calls != expected_extraction:
        raise Mem0ShardRunError(
            "retrieval extraction-model receipt mismatch", stage="scoring"
        )
    if (
        ingestion_receipt.get("persisted_request_token_state") is not False
        or ingestion_receipt.get("retained_request_token_state_bytes") != 0
        or ingestion_receipt.get("request_token_state_evidence_kind")
        != "local_injected_request_token_state_contract"
        or ingestion_receipt.get("external_provider_persistence_certified")
        is not False
        or ingestion_receipt.get("comparison_certified") is not False
    ):
        raise Mem0ShardRunError(
            "retrieval ingestion request-token-state proof mismatch",
            stage="scoring",
        )
    trace_meta = artifact.get("retrieval_trace")
    if not isinstance(trace_meta, dict):
        raise Mem0ShardRunError("retrieval artifact omitted trace binding", stage="scoring")
    if trace_meta.get("filename") != trace_path.name:
        raise Mem0ShardRunError("retrieval trace filename mismatch", stage="scoring")
    trace_payload = trace_path.read_bytes()
    if trace_meta.get("bytes") != len(trace_payload):
        raise Mem0ShardRunError("retrieval trace byte count mismatch", stage="scoring")
    if trace_meta.get("sha256") != hashlib.sha256(trace_payload).hexdigest():
        raise Mem0ShardRunError("retrieval trace digest mismatch", stage="scoring")
    trace = _parse_json_object_bytes(trace_payload, filename=trace_path.name)
    if (
        trace.get("format") != RETRIEVAL_TRACE_FORMAT
        or trace.get("status") != "complete"
        or trace.get("certification_status") != "injected_nonproduction"
        or trace.get("comparison_certified") is not False
        or trace.get("execution_binding") != _execution_binding_receipt(None)
        or trace.get("sample_sha256") != shard.sample_sha256
        or not isinstance(trace.get("cleanup"), dict)
        or trace["cleanup"].get("state_absent_after") is not True
        or trace["cleanup"].get("active_scope_cleared") is not True
        or trace["cleanup"].get(
            "extraction_meter_restored_before_cleanup"
        )
        is not True
        or trace["cleanup"].get("persisted_request_token_state") is not False
        or trace["cleanup"].get("retained_request_token_state_bytes") != 0
        or trace["cleanup"].get("request_token_state_evidence_kind")
        != "local_injected_request_token_state_contract"
        or trace["cleanup"].get("external_provider_persistence_certified")
        is not False
        or trace["cleanup"].get("adapter_closed") is not True
        or trace["cleanup"].get("ledger_empty") is not True
        or trace["cleanup"].get("registered_scopes_empty") is not True
        or trace["cleanup"].get("scope_protocol_empty") is not True
        or trace["cleanup"].get("backend_closed_or_cleared") is not True
        or trace["cleanup"].get("owned_state_path_absent") is not True
        or trace.get("environment_lock") != environment_lock
        or trace["cleanup"].get("environment_lock") != environment_lock
    ):
        raise Mem0ShardRunError("retrieval trace cleanup proof mismatch", stage="scoring")

    rows = artifact.get("retrieval_rows")
    if not isinstance(rows, list) or len(rows) != len(shard.question_ids):
        raise Mem0ShardRunError("retrieval artifact row count mismatch", stage="scoring")
    verified: list[tuple[dict[str, Any], Any]] = []
    for index, (row_value, question) in enumerate(
        zip(rows, shard.parsed_sample.questions, strict=True), start=1
    ):
        if not isinstance(row_value, dict):
            raise Mem0ShardRunError("retrieval row is not an object", stage="scoring")
        row = dict(row_value)
        row_digest = row.pop("retrieval_row_sha256", None)
        _require_sha256(row_digest, "retrieval_row_sha256")
        if canonical_json_sha256(row) != row_digest:
            raise Mem0ShardRunError("retrieval row digest mismatch", stage="scoring")
        row["retrieval_row_sha256"] = row_digest
        query = question.dated_question
        if row.get("question_id") != question.question_id:
            raise Mem0ShardRunError("retrieval row question order mismatch", stage="scoring")
        if row.get("query") != query:
            raise Mem0ShardRunError("retrieval row query mismatch", stage="scoring")
        raw_pool = row.get("raw_pool")
        if not isinstance(raw_pool, list):
            raise Mem0ShardRunError("retrieval row raw pool missing", stage="scoring")
        normalized = [
            _candidate_payload(SimpleNamespace(**candidate), rank)
            for rank, candidate in enumerate(raw_pool, start=1)
        ]
        if canonical_json_sha256(normalized) != row.get("raw_pool_sha256"):
            raise Mem0ShardRunError("retrieval row raw-pool digest mismatch", stage="scoring")
        frozen = _frozen_search_result(artifact, row, query)
        pack = prompt_packer(
            query,
            frozen,
            max_prompt_tokens=authorization.max_prompt_tokens,
            evaluation_identity=authorization.source_evaluation_identity,
        )
        rebuilt = _pack_payload(
            pack,
            normalized,
            question_id=question.question_id,
            search_latency_s=float(row.get("search_latency_s", -1.0)),
        )
        if row != rebuilt:
            raise Mem0ShardRunError(
                "retrieval row independently rebuilt content mismatch",
                stage="scoring",
            )
        verified.append((row, pack))
    return artifact, artifact_bytes, verified


@dataclass(slots=True)
class _CallBudget:
    label: str
    authorized: int
    attempted: int = 0
    completed: int = 0
    failed: int = 0
    request_token_state_verified: bool = False

    def call(
        self,
        invoker: ProviderInvoker,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        max_output_tokens: int,
    ) -> ProviderCallResult:
        if self.attempted >= self.authorized:
            raise Mem0ShardRunError(
                f"{self.label} provider authorization exhausted", stage="scoring"
            )
        self.attempted += 1
        try:
            _request_token_state_receipt(invoker, self.label)
            self.request_token_state_verified = False
            result = invoker(
                messages,
                model=model,
                max_output_tokens=max_output_tokens,
            )
            if not isinstance(result, ProviderCallResult):
                raise TypeError("provider invoker must return ProviderCallResult")
            if not isinstance(result.text, str):
                raise TypeError("provider response text must be str")
            if not isinstance(result.usage, UsageStats) or result.usage.calls != 1:
                raise ValueError("provider response must account for exactly one call")
            _request_token_state_receipt(invoker, self.label)
            self.request_token_state_verified = True
        except BaseException as exc:
            try:
                _request_token_state_receipt(invoker, self.label)
                self.request_token_state_verified = True
            except BaseException as state_exc:
                self.request_token_state_verified = False
                exc.add_note(
                    f"{self.label} request-token-state verification failed: "
                    f"{type(state_exc).__name__}"
                )
            self.failed += 1
            raise
        self.completed += 1
        return result

    def receipt(self) -> dict[str, int]:
        return {
            "authorized": self.authorized,
            "attempted": self.attempted,
            "completed": self.completed,
            "failed": self.failed,
        }


def _usage_payload(usage: UsageStats) -> dict[str, Any]:
    return _strict_json(usage)


def _sum_usage(rows: Sequence[UsageStats]) -> UsageStats:
    return sum(rows, UsageStats())


def _provider_input_usage_status(rows: Sequence[UsageStats]) -> str:
    """Describe provider input-count availability without inventing usage.

    A completed call whose provider reports zero input tokens has unknown
    input usage.  It remains a completed call, but zero must not be presented
    as an actually empty request.
    """

    available = sum(1 for row in rows if row.input_tokens > 0)
    if available == 0:
        return "unavailable"
    if available == len(rows):
        return "complete"
    return "partial"


def _parse_judge(text: str) -> tuple[bool, str]:
    match = _JUDGE_RE.match(text)
    if match is None:
        raise ValueError("judge did not return a leading CORRECT/INCORRECT verdict")
    return match.group(1).upper() == "CORRECT", text.strip()


def run_scoring_stage(
    *,
    shard: RawStressShard,
    authorization: ScoringStageAuthorization,
    root_environment_lock_path: str | os.PathLike[str],
    retrieval_artifact_path: str | os.PathLike[str],
    retrieval_trace_path: str | os.PathLike[str],
    report_path: str | os.PathLike[str],
    scoring_trace_path: str | os.PathLike[str],
    responder: ProviderInvoker,
    judge: ProviderInvoker,
    prompt_packer: Callable[..., Any] = _default_prompt_packer,
    trusted_runtime_binding: TrustedRuntimeBinding | None = None,
    process_guard: ShardProcessGuard = _SCORING_PROCESS_GUARD,
) -> ScoringStageResult:
    """Verify Stage A, make exactly ten QA plus ten judge calls, and report."""

    try:
        validate_raw_stress_shard(shard)
    except BaseException as exc:
        raise Mem0ShardRunError(
            "Mem0 scoring shard failed independent content validation",
            stage="scoring",
        ) from exc
    try:
        environment_lock_target, environment_lock_before = (
            _environment_lock_snapshot(
                root_environment_lock_path,
                label="root_environment_lock_path",
            )
        )
        execution_binding = _execution_binding_receipt(
            trusted_runtime_binding,
            stage="scoring",
            authorization=authorization,
            bound_callables=(
                responder,
                judge,
                prompt_packer,
                process_guard,
            ),
        )
    except BaseException as exc:
        raise Mem0ShardRunError(
            "Mem0 scoring execution boundary could not be verified",
            stage="scoring",
        ) from exc
    started = time.perf_counter()
    artifact_target = Path(retrieval_artifact_path).resolve(strict=False)
    retrieval_trace_target = Path(retrieval_trace_path).resolve(strict=False)
    report_target = Path(report_path).resolve(strict=False)
    scoring_trace_target = Path(scoring_trace_path).resolve(strict=False)
    events: list[dict[str, Any]] = []
    answer_budget = _CallBudget(
        "responder", authorization.authorized_responder_calls
    )
    judge_budget = _CallBudget("judge", authorization.authorized_judge_calls)
    operation_error: BaseException | None = None
    report: dict[str, Any] | None = None
    request_token_state_verified = False
    stateless_provider_contracts: dict[str, Any] = {}
    environment_lock_after: str | None = None
    try:
        _validate_scoring_authorization(
            shard,
            authorization,
            computed_root_environment_lock_sha256=(
                environment_lock_before
            ),
        )
        _validated_model_identities(authorization)
        if not callable(responder) or not callable(judge):
            raise TypeError("responder and judge must be injected callables")
        stateless_provider_contracts = {
            "responder": _request_token_state_receipt(responder, "responder"),
            "judge": _request_token_state_receipt(judge, "judge"),
        }
        request_token_state_verified = True
        if not callable(prompt_packer):
            raise TypeError("prompt_packer must be callable")
        if (
            _is_equal_to_or_within(report_target, scoring_trace_target)
            or _is_equal_to_or_within(scoring_trace_target, report_target)
        ):
            raise ValueError(
                "report and scoring trace paths must be distinct and non-nested"
            )
        _reject_protected_outputs(
            (report_target, scoring_trace_target),
            (
                (environment_lock_target, "root environment lock input"),
                (artifact_target, "Stage A retrieval artifact input"),
                (retrieval_trace_target, "Stage A retrieval trace input"),
                *_implementation_roots(),
            ),
            label="Stage B",
        )
        if report_target.exists() or scoring_trace_target.exists():
            raise FileExistsError("Stage B outputs must not already exist")
        if not artifact_target.is_file() or not retrieval_trace_target.is_file():
            raise FileNotFoundError("Stage B requires complete Stage A artifact and trace")
        process_guard.claim()
        events.append({"sequence": 1, "event": "authorization_verified"})
        artifact, artifact_bytes, verified_rows = _verify_retrieval_artifact(
            artifact_path=artifact_target,
            trace_path=retrieval_trace_target,
            shard=shard,
            authorization=authorization,
            prompt_packer=prompt_packer,
        )
        events.append(
            {
                "sequence": 2,
                "event": "retrieval_artifact_verified",
                "sha256": authorization.retrieval_artifact_sha256,
                "bytes": len(artifact_bytes),
            }
        )

        question_results: list[dict[str, Any]] = []
        answer_usages: list[UsageStats] = []
        judge_usages: list[UsageStats] = []
        from .prompt_pack import verify_provider_input_tokens

        for index, ((retrieval_row, pack), question) in enumerate(
            zip(verified_rows, shard.parsed_sample.questions, strict=True), start=1
        ):
            messages = _message_payload(pack.messages)
            answer = answer_budget.call(
                responder,
                messages,
                model=authorization.responder_model,
                max_output_tokens=authorization.responder_max_output_tokens,
            )
            provider_prompt_budget_compliant = verify_provider_input_tokens(
                pack, answer.usage.input_tokens
            )
            prediction = answer.text.strip()
            if not prediction:
                raise ValueError("responder returned an empty answer")
            judge_messages = build_judge_prompt(
                question.question, question.answer, prediction
            )
            judge_result = judge_budget.call(
                judge,
                judge_messages,
                model=authorization.judge_model,
                max_output_tokens=authorization.judge_max_output_tokens,
            )
            if judge_result.usage.input_tokens < 0:
                raise ValueError("judge provider input-token usage cannot be negative")
            judged_correct, judge_reasoning = _parse_judge(judge_result.text)
            answer_usages.append(answer.usage)
            judge_usages.append(judge_result.usage)
            result_row = {
                "question_index": index,
                "question_id": question.question_id,
                "question": question.question,
                "dated_question": question.dated_question,
                "gold_answer": question.answer,
                "prediction": prediction,
                "category": question.category,
                "retrieval_row_sha256": retrieval_row["retrieval_row_sha256"],
                "query_sha256": hashlib.sha256(
                    retrieval_row["query"].encode("utf-8")
                ).hexdigest(),
                "prompt_pack_protocol": retrieval_row[
                    "prompt_pack_protocol"
                ],
                "context": retrieval_row["context"],
                "context_sha256": retrieval_row["context_sha256"],
                "context_tokens": retrieval_row["context_tokens"],
                "messages": retrieval_row["messages"],
                "messages_sha256": retrieval_row["messages_sha256"],
                "prompt_token_proxy": retrieval_row["prompt_token_proxy"],
                "max_prompt_tokens": retrieval_row["max_prompt_token_proxy"],
                "residual_prompt_tokens": retrieval_row[
                    "residual_prompt_token_proxy"
                ],
                "prompt_token_proxy_identity": retrieval_row[
                    "prompt_token_proxy_identity"
                ],
                "raw_pool_count": retrieval_row["raw_memory_count"],
                "raw_pool_sha256": retrieval_row["raw_pool_sha256"],
                "raw_memory_tokens": retrieval_row["raw_memory_tokens"],
                "packed_count": retrieval_row["packed_memory_count"],
                "packed_memory_tokens": retrieval_row["packed_memory_tokens"],
                "packed_pool_sha256": retrieval_row["packed_pool_sha256"],
                "search_latency_s": retrieval_row["search_latency_s"],
                "attribution_kind": MEM0_ATTRIBUTION_KIND,
                "supports_exact_source_provenance": False,
                "exact_match": exact_match(prediction, question.answer),
                "f1": f1_score(prediction, question.answer),
                "judge_correct": judged_correct,
                "judge_reasoning": judge_reasoning,
                "provider_prompt_budget_compliant": (
                    provider_prompt_budget_compliant
                ),
                "configured_recent_window": retrieval_row[
                    "configured_recent_window"
                ],
                "effective_recent_window": retrieval_row[
                    "effective_recent_window"
                ],
                "recent_window_semantics": retrieval_row[
                    "recent_window_semantics"
                ],
                "responder_usage": _usage_payload(answer.usage),
                "judge_usage": _usage_payload(judge_result.usage),
            }
            question_results.append(result_row)
            events.append(
                {
                    "sequence": 2 + index,
                    "event": "question_scored",
                    "question_id": question.question_id,
                    "retrieval_row_sha256": retrieval_row[
                        "retrieval_row_sha256"
                    ],
                    "prediction_sha256": hashlib.sha256(
                        prediction.encode("utf-8")
                    ).hexdigest(),
                    "responder_logical_wrapper_calls_completed": (
                        answer_budget.completed
                    ),
                    "judge_logical_wrapper_calls_completed": (
                        judge_budget.completed
                    ),
                }
            )
        if answer_budget.completed != answer_budget.authorized or answer_budget.failed:
            raise ValueError("responder call accounting did not close exactly")
        if judge_budget.completed != judge_budget.authorized or judge_budget.failed:
            raise ValueError("judge call accounting did not close exactly")
        responder_usage = _sum_usage(answer_usages)
        judge_usage = _sum_usage(judge_usages)
        stateless_provider_contracts = {
            "responder": _request_token_state_receipt(responder, "responder"),
            "judge": _request_token_state_receipt(judge, "judge"),
        }
        model_identity = {
            "responder_model": authorization.responder_model,
            "responder_model_identity_sha256": (
                authorization.responder_model_identity_sha256
            ),
            "judge_model": authorization.judge_model,
            "judge_model_identity_sha256": authorization.judge_model_identity_sha256,
        }
        config = {
            "max_prompt_tokens": authorization.max_prompt_tokens,
            "responder_max_output_tokens": authorization.responder_max_output_tokens,
            "judge_max_output_tokens": authorization.judge_max_output_tokens,
            "authorized_local_wrapper_retries": authorization.provider_retries,
            "external_retry_attempts_certified": False,
            "mem0_top_k": MEM0_OFFICIAL_TOP_K,
            "mem0_threshold": MEM0_OFFICIAL_THRESHOLD,
            "rendering_mode": MEM0_CERTIFIED_RENDERING,
        }
        report = {
            "schema_version": SHARD_SCHEMA_VERSION,
            "report_type": SHARD_REPORT_TYPE,
            "arm_id": SHARD_ARM_ID,
            "run_status": "complete",
            "certification_status": "injected_nonproduction",
            "comparison_certified": execution_binding["comparison_certified"],
            "execution_binding": execution_binding,
            "identity": {
                "source_validation_policy_sha256": authorization.source_validation_policy_sha256,
                "source_implementation_sha256": authorization.source_implementation_sha256,
                "source_environment_lock_sha256": authorization.source_environment_lock_sha256,
                "mem0_policy_sha256": authorization.mem0_policy_sha256,
                "mem0_tool_implementation_sha256": authorization.mem0_tool_implementation_sha256,
                "mem0_environment_lock_sha256": authorization.mem0_environment_lock_sha256,
                "mem0_stable_config_sha256": authorization.mem0_stable_config_sha256,
                "extraction_model_identity": _strict_json(
                    authorization.extraction_model_identity
                ),
                "extraction_model_identity_sha256": (
                    authorization.extraction_model_identity_sha256
                ),
                "embedder_model_identity": _strict_json(
                    authorization.embedder_model_identity
                ),
                "embedder_model_identity_sha256": (
                    authorization.embedder_model_identity_sha256
                ),
                "scoring_policy_sha256": authorization.scoring_policy_sha256,
                "source_evaluation_identity_sha256": canonical_json_sha256(
                    authorization.source_evaluation_identity
                ),
            },
            "model_identity": model_identity,
            "model_identity_sha256": canonical_json_sha256(model_identity),
            "config": config,
            "config_sha256": canonical_json_sha256(config),
            "evaluation_protocol": {
                "split": "validation",
                "stress_context_tokens": 1_000_000,
                "stress_questions": LONGMEMEVAL_QUESTIONS_PER_SHARD,
                "stress_question_offset": 0,
                "answer_prompt_calls_per_question": 1,
                "judge_calls_per_question": 1,
                "authorized_local_wrapper_retries": 0,
                "external_retry_attempts_certified": False,
            },
            "sample_offset": shard.sample_offset,
            "samples": [
                {
                    "sample_id": shard.parsed_sample.sample_id,
                    "sample_sha256": shard.sample_sha256,
                    "raw_history_bundle_sha256": shard.raw_history_bundle_sha256,
                    "history_sample_ids": list(shard.history_sample_ids),
                    "question_ids": list(shard.question_ids),
                }
            ],
            "raw_input_receipt": artifact["raw_input_receipt"],
            "ingestion_receipt": artifact["ingestion_receipt"],
            "retrieval_artifact": {
                "filename": artifact_target.name,
                "sha256": authorization.retrieval_artifact_sha256,
                "bytes": len(artifact_bytes),
                "question_ids": list(shard.question_ids),
                "retrieval_trace": artifact["retrieval_trace"],
            },
            "scoring_receipt": {
                "format": SCORING_RECEIPT_FORMAT,
                "retrieval_artifact_sha256": authorization.retrieval_artifact_sha256,
                "source_environment_lock_sha256": environment_lock_before,
                "responder_logical_wrapper_calls": answer_budget.receipt(),
                "judge_logical_wrapper_calls": judge_budget.receipt(),
                "answer_judge_logical_wrapper_calls": (
                    answer_budget.completed + judge_budget.completed
                ),
                "authorized_local_wrapper_retries": 0,
                "external_http_attempts_certified": False,
                "external_retry_attempts_certified": False,
                "persisted_request_token_state": False,
                "retained_request_token_state_bytes": 0,
                "request_token_state_evidence_kind": (
                    "local_injected_request_token_state_contract"
                ),
                "external_provider_persistence_certified": False,
                "stateless_provider_contracts": stateless_provider_contracts,
                "responder_input_usage_status": (
                    _provider_input_usage_status(answer_usages)
                ),
                "judge_input_usage_status": (
                    _provider_input_usage_status(judge_usages)
                ),
                "responder_usage": _usage_payload(responder_usage),
                "judge_usage": _usage_payload(judge_usage),
            },
            "mem0_usage": artifact["mem0_usage"],
            "provenance": artifact["provenance"],
            "question_results": question_results,
        }
    except BaseException as exc:
        operation_error = exc

    try:
        observed_lock_target, environment_lock_after = _environment_lock_snapshot(
            environment_lock_target,
            label="root_environment_lock_path post-stage",
        )
        if observed_lock_target != environment_lock_target:
            raise RuntimeError("root environment lock path identity changed")
        if environment_lock_after != environment_lock_before:
            raise RuntimeError("root environment lock bytes changed during Stage B")
    except BaseException as lock_exc:
        if operation_error is None:
            operation_error = lock_exc
        else:
            operation_error.add_note(
                "root environment-lock recheck also failed: "
                f"{type(lock_exc).__name__}"
            )
    environment_lock_receipt = {
        "filename": environment_lock_target.name,
        "authorized_sha256": authorization.source_environment_lock_sha256,
        "sha256_before": environment_lock_before,
        "sha256_after": environment_lock_after,
        "unchanged": environment_lock_after == environment_lock_before,
    }
    if report is not None:
        report["environment_lock"] = environment_lock_receipt

    if operation_error is not None:
        failure_state_verified = request_token_state_verified and all(
            budget.attempted == 0 or budget.request_token_state_verified
            for budget in (answer_budget, judge_budget)
        )
        failed = _failure_trace(
            trace_format=SCORING_TRACE_FORMAT,
            stage="scoring",
            shard=shard,
            events=events,
            started=started,
            error=operation_error,
            cleanup={
                "mem0_state_touched": False,
                "responder_logical_wrapper_calls": answer_budget.receipt(),
                "judge_logical_wrapper_calls": judge_budget.receipt(),
                "persisted_request_token_state": (
                    False if failure_state_verified else None
                ),
                "retained_request_token_state_bytes": (
                    0 if failure_state_verified else None
                ),
                "request_token_state_evidence_kind": (
                    "local_injected_request_token_state_contract"
                    if failure_state_verified
                    else None
                ),
                "external_provider_persistence_certified": False,
                "external_http_attempts_certified": False,
                "external_retry_attempts_certified": False,
                "environment_lock": environment_lock_receipt,
            },
        )
        written_trace: Path | None = None
        try:
            trace_is_safe = all(
                not _is_equal_to_or_within(scoring_trace_target, protected_path)
                for protected_path, _label in (
                    (environment_lock_target, "root environment lock input"),
                    (artifact_target, "Stage A retrieval artifact input"),
                    (retrieval_trace_target, "Stage A retrieval trace input"),
                    *_implementation_roots(),
                )
            )
            if trace_is_safe:
                _atomic_create_json(scoring_trace_target, failed)
                written_trace = scoring_trace_target
        except BaseException as trace_exc:
            operation_error.add_note(
                f"failure trace could not be written: {type(trace_exc).__name__}"
            )
        raise Mem0ShardRunError(
            "Mem0 scoring stage failed closed",
            stage="scoring",
            trace_path=written_trace,
        ) from operation_error

    assert report is not None
    events.append(
        {
            "sequence": len(events) + 1,
            "event": "call_budgets_closed",
            "responder_logical_wrapper_calls": answer_budget.receipt(),
            "judge_logical_wrapper_calls": judge_budget.receipt(),
        }
    )
    trace = {
        "format": SCORING_TRACE_FORMAT,
        "status": "complete",
        "certification_status": "injected_nonproduction",
        "comparison_certified": execution_binding["comparison_certified"],
        "execution_binding": execution_binding,
        "stage": "scoring",
        "sample_offset": shard.sample_offset,
        "sample_id": shard.parsed_sample.sample_id,
        "sample_sha256": shard.sample_sha256,
        "retrieval_artifact_sha256": authorization.retrieval_artifact_sha256,
        "events": events,
        "mem0_state_touched": False,
        "responder_logical_wrapper_calls": answer_budget.receipt(),
        "judge_logical_wrapper_calls": judge_budget.receipt(),
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
        "external_http_attempts_certified": False,
        "external_retry_attempts_certified": False,
        "stateless_provider_contracts": stateless_provider_contracts,
        "environment_lock": environment_lock_receipt,
        "elapsed_s": max(0.0, time.perf_counter() - started),
    }
    try:
        trace_payload = _render_json_bytes(trace)
        trace_sha = hashlib.sha256(trace_payload).hexdigest()
        trace_bytes = len(trace_payload)
        report["scoring_receipt"]["scoring_trace"] = {
            "filename": scoring_trace_target.name,
            "sha256": trace_sha,
            "bytes": trace_bytes,
        }
        report_payload = _render_json_bytes(report)
        receipts = _atomic_create_payloads(
            (
                (scoring_trace_target, trace_payload),
                (report_target, report_payload),
            )
        )
        (published_trace_sha, published_trace_bytes), (
            report_sha,
            report_bytes,
        ) = receipts
        if (published_trace_sha, published_trace_bytes) != (
            trace_sha,
            trace_bytes,
        ):
            raise AssertionError("published Stage B trace receipt changed")
    except BaseException as exc:
        raise Mem0ShardRunError(
            "Mem0 scoring receipts could not be published atomically",
            stage="scoring",
            trace_path=None,
        ) from exc
    return ScoringStageResult(
        report_path=report_target,
        trace_path=scoring_trace_target,
        report_sha256=report_sha,
        report_bytes=report_bytes,
        report=MappingProxyType(report),
        trace=MappingProxyType(trace),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed two-stage Mem0 shard runner. The real policy/runtime "
            "binding is intentionally not accepted from ad-hoc CLI flags."
        )
    )
    parser.add_argument(
        "--show-contract",
        action="store_true",
        help="print the provider-free runner status and exit non-zero",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv)
    status = {
        "format": "memory-condense-mem0-runner-status-v1",
        "status": "blocked_pending_frozen_runtime_policy",
        "provider_calls_permitted": False,
        "required_entrypoints": [
            "run_retrieval_stage (isolated Mem0 environment)",
            "run_scoring_stage (frozen root environment)",
        ],
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AdapterFactory",
    "ExtractionMeterInstaller",
    "LogicalExtractionCallMeter",
    "Mem0ShardRunError",
    "ProviderCallResult",
    "ProviderInvoker",
    "RetrievalStageAuthorization",
    "RetrievalStageResult",
    "ScoringStageAuthorization",
    "ScoringStageResult",
    "ShardProcessGuard",
    "TrustedRuntimeBinding",
    "build_adapter_prepared_corpus",
    "canonical_json_sha256",
    "composite_add_batch_to_prepared",
    "install_memory_llm_extraction_meter",
    "main",
    "run_retrieval_stage",
    "run_scoring_stage",
]
