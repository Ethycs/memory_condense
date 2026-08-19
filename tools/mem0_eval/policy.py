"""Fail-closed policy binding for the locked Mem0 comparison campaign.

The source validation policy freezes the LongMemEval population and the
answer/judge protocol.  It does not select a Mem0 extraction model, embedder,
runtime stack, or authorize Mem0 calls.  This module verifies a second,
arm-specific manifest and is the only supported bridge from those frozen
artifacts to :mod:`tools.mem0_eval.run_shard` authorization objects.

No provider client or Mem0 package is imported here.  The manifest is parsed
from one immutable byte snapshot, all JSON is finite and secret-free, and the
policy, isolated lock, and tool implementation can be rechecked immediately
before either execution stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from memory_condense.eval.mem0_adapter import (
    MEM0AI_PIN,
    MEM0_API_VERSION,
    MEM0_ATTRIBUTION_KIND,
    MEM0_BM25_MODEL,
    MEM0_CERTIFIED_RENDERING,
    MEM0_OFFICIAL_THRESHOLD,
    MEM0_OFFICIAL_TOP_K,
    MEM0_SPACY_MODEL,
)
from .source_compat import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)

from .preflight import SourceValidationPlan, tool_implementation_sha256
from .protocol import (
    Mem0ComparisonProtocolError,
    RawStressShard,
    shard_receipt,
    validate_raw_stress_shard,
)

if TYPE_CHECKING:
    from .run_shard import RetrievalStageAuthorization, ScoringStageAuthorization


MEM0_POLICY_FORMAT = "memory-condense-mem0-comparison-policy-v1"
MEM0_POLICY_STATUS = "validation_frozen"
MEM0_ARM_ID = "mem0_oss_2_0_18_direct_1m_v1"
MEM0_RUNTIME_PROTOCOL = "mem0-oss-2.0.18-certified-local-v1"
MEM0_INPUT_ORDER_PROTOCOL = (
    "locked-record-order+official-within-record-date-sort+"
    "consecutive-1-or-2-turn-slices-v1"
)
MEM0_EXTRACTION_BOUNDARY = "Memory.llm.generate_response"
MEM0_PROVENANCE_KIND = "request_window_non_evidence"
MEM0_SOURCE_SESSION_DATE_EXPOSURE = "diagnostics_only_not_model_input"
MEM0_RETRIEVED_CREATED_AT_EXPOSURE = "answer_prompt_date_headings"
MEM0_EMBEDDER_EXECUTION = "local_offline"
MEM0_EMBEDDER_PROVIDER = "huggingface"
MEM0_EMBEDDER_MODEL = DEFAULT_MODEL_NAME
MEM0_EMBEDDER_REVISION = DEFAULT_MODEL_REVISION
MEM0_EMBEDDER_CHECKPOINT_SHA256 = BGE_M3_CHECKPOINT_SHA256
MEM0_EMBEDDER_DIMENSION = DEFAULT_MODEL_DIM
MEM0_EMBEDDER_DTYPE = "float32"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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


class Mem0PolicyError(ValueError):
    """The arm-specific manifest cannot authorize the locked campaign."""


def _reject_json_constant(value: str) -> None:
    raise Mem0PolicyError(f"non-finite JSON number {value!r} is not allowed")


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Mem0PolicyError(f"value is not canonical JSON: {exc}") from exc


def canonical_json_sha256(value: Any) -> str:
    """Return the canonical digest used by the policy and shard runner."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Mem0PolicyError(f"{label} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise Mem0PolicyError(
            f"{label} fields mismatch: missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Mem0PolicyError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    digest = _text(value, label)
    if _SHA256_RE.fullmatch(digest) is None:
        raise Mem0PolicyError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Mem0PolicyError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Mem0PolicyError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise Mem0PolicyError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise Mem0PolicyError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise Mem0PolicyError(f"{label} must be <= {maximum}")
    return result


def _must_equal(actual: Any, expected: Any, label: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise Mem0PolicyError(f"{label} does not match the frozen contract")


def _reject_secret_material(value: Any, label: str = "policy") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if lowered in _FORBIDDEN_SECRET_KEYS or lowered.endswith(
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
                if child != "<redacted>":
                    raise Mem0PolicyError(
                        f"{label}.{key} contains forbidden secret material"
                    )
            _reject_secret_material(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_material(child, f"{label}[{index}]")
    elif isinstance(value, str) and _SECRET_VALUE_RE.search(value):
        raise Mem0PolicyError(f"{label} contains credential-shaped material")


def _immutable_json(value: Any) -> Any:
    """Copy JSON values and freeze mappings exposed by the result object."""

    if isinstance(value, dict):
        return MappingProxyType({key: _immutable_json(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_immutable_json(child) for child in value)
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(child) for child in value]
    return value


def expected_shard_policy_rows(
    shards: Sequence[RawStressShard],
) -> tuple[dict[str, Any], ...]:
    """Build the exact text-free shard table a frozen policy must contain."""

    rows: list[dict[str, Any]] = []
    offsets: set[int] = set()
    question_ids: set[str] = set()
    for shard in shards:
        try:
            content_identity = validate_raw_stress_shard(shard)
        except Mem0ComparisonProtocolError as exc:
            raise Mem0PolicyError(f"invalid expected shard: {exc}") from exc
        if shard.sample_offset in offsets:
            raise Mem0PolicyError("expected shards repeat a sample offset")
        offsets.add(shard.sample_offset)
        ids = tuple(shard.question_ids)
        if len(ids) != 10 or len(set(ids)) != 10:
            raise Mem0PolicyError("each expected shard must contain ten unique questions")
        if question_ids.intersection(ids):
            raise Mem0PolicyError("expected shards repeat question IDs")
        question_ids.update(ids)
        receipt = shard_receipt(shard)
        rows.append(
            {
                "sample_offset": receipt["sample_offset"],
                "sample_id": receipt["sample_id"],
                "sample_sha256": receipt["sample_sha256"],
                "raw_history_bundle_sha256": receipt[
                    "raw_history_bundle_sha256"
                ],
                "history_samples": receipt["history_samples"],
                "history_sample_ids_sha256": canonical_json_sha256(
                    list(shard.history_sample_ids)
                ),
                "questions": receipt["questions"],
                "question_ids": receipt["question_ids"],
                "question_ids_sha256": canonical_json_sha256(
                    receipt["question_ids"]
                ),
                "turns": receipt["turns"],
                "transcript_tokens": receipt["transcript_tokens"],
                "raw_pairs": receipt["raw_pairs"],
                "skipped_empty_pairs": receipt["skipped_empty_pairs"],
                "authorized_add_operations": receipt["mem0_add_requests"],
                "authorized_extraction_calls": receipt["mem0_add_requests"],
                "authorized_search_operations": receipt["questions"],
                "add_batches_sha256": content_identity["add_batches_sha256"],
            }
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class Mem0ShardPolicy:
    sample_offset: int
    sample_id: str
    sample_sha256: str
    raw_history_bundle_sha256: str
    question_ids: tuple[str, ...]
    authorized_add_operations: int
    authorized_extraction_calls: int
    authorized_search_operations: int


@dataclass(frozen=True, slots=True)
class Mem0ComparisonPolicy:
    """One verified, immutable authorization contract for all ten shards."""

    path: Path
    sha256: str
    environment_lock_path: Path
    environment_lock_sha256: str
    tool_root: Path
    tool_implementation_sha256: str
    source_plan: SourceValidationPlan
    arm_id: str
    stable_config_sha256: str
    stable_payload: Mapping[str, Any]
    extraction_identity: Mapping[str, Any]
    embedder_identity: Mapping[str, Any]
    scoring: Mapping[str, Any]
    payload: Mapping[str, Any]
    shards: Mapping[int, Mem0ShardPolicy]

    def recheck(self) -> None:
        """Fail if policy, lock, source plan, or tool bytes changed since load."""

        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise Mem0PolicyError("Mem0 comparison policy changed after verification")
        if (
            hashlib.sha256(self.environment_lock_path.read_bytes()).hexdigest()
            != self.environment_lock_sha256
        ):
            raise Mem0PolicyError("Mem0 environment lock changed after verification")
        if tool_implementation_sha256(self.tool_root) != self.tool_implementation_sha256:
            raise Mem0PolicyError("Mem0 tool implementation changed after verification")

    def retrieval_authorization(
        self, shard: RawStressShard
    ) -> RetrievalStageAuthorization:
        """Construct the sole Stage-A authorization for ``shard``."""

        from .run_shard import RetrievalStageAuthorization

        self.recheck()
        row = self._shard(shard)
        return RetrievalStageAuthorization(
            sample_offset=row.sample_offset,
            sample_sha256=row.sample_sha256,
            raw_history_bundle_sha256=row.raw_history_bundle_sha256,
            question_ids=row.question_ids,
            authorized_add_operations=row.authorized_add_operations,
            authorized_extraction_calls=row.authorized_extraction_calls,
            authorized_search_operations=row.authorized_search_operations,
            source_validation_policy_sha256=(
                self.source_plan.policy_manifest_sha256
            ),
            source_implementation_sha256=self.source_plan.implementation_sha256,
            source_environment_lock_sha256=(
                self.source_plan.environment_lock_sha256
            ),
            mem0_policy_sha256=self.sha256,
            mem0_tool_implementation_sha256=self.tool_implementation_sha256,
            mem0_environment_lock_sha256=self.environment_lock_sha256,
            mem0_stable_config_sha256=self.stable_config_sha256,
            source_evaluation_identity=_plain_json(
                self.source_plan.evaluation_identity
            ),
            mem0_stable_payload=_plain_json(self.stable_payload),
            extraction_model_identity=_plain_json(self.extraction_identity),
            extraction_model_identity_sha256=canonical_json_sha256(
                _plain_json(self.extraction_identity)
            ),
            embedder_model_identity=_plain_json(self.embedder_identity),
            embedder_model_identity_sha256=canonical_json_sha256(
                _plain_json(self.embedder_identity)
            ),
            mem0_provider_retries=0,
        )

    def scoring_authorization(
        self,
        shard: RawStressShard,
        *,
        retrieval_artifact_sha256: str,
    ) -> ScoringStageAuthorization:
        """Construct the sole Stage-B authorization for a verified artifact."""

        from .run_shard import ScoringStageAuthorization

        self.recheck()
        row = self._shard(shard)
        artifact_digest = _sha256(
            retrieval_artifact_sha256, "retrieval_artifact_sha256"
        )
        scoring = _plain_json(self.scoring)
        return ScoringStageAuthorization(
            sample_offset=row.sample_offset,
            sample_sha256=row.sample_sha256,
            raw_history_bundle_sha256=row.raw_history_bundle_sha256,
            question_ids=row.question_ids,
            retrieval_artifact_sha256=artifact_digest,
            source_validation_policy_sha256=(
                self.source_plan.policy_manifest_sha256
            ),
            source_implementation_sha256=self.source_plan.implementation_sha256,
            source_environment_lock_sha256=(
                self.source_plan.environment_lock_sha256
            ),
            mem0_policy_sha256=self.sha256,
            mem0_tool_implementation_sha256=self.tool_implementation_sha256,
            mem0_environment_lock_sha256=self.environment_lock_sha256,
            mem0_stable_config_sha256=self.stable_config_sha256,
            source_evaluation_identity=_plain_json(
                self.source_plan.evaluation_identity
            ),
            mem0_stable_payload=_plain_json(self.stable_payload),
            scoring_policy_sha256=self.sha256,
            responder_model=scoring["responder_identity"]["model"],
            judge_model=scoring["judge_identity"]["model"],
            responder_model_identity_sha256=scoring[
                "responder_identity_sha256"
            ],
            judge_model_identity_sha256=scoring["judge_identity_sha256"],
            extraction_model_identity=_plain_json(self.extraction_identity),
            extraction_model_identity_sha256=canonical_json_sha256(
                _plain_json(self.extraction_identity)
            ),
            embedder_model_identity=_plain_json(self.embedder_identity),
            embedder_model_identity_sha256=canonical_json_sha256(
                _plain_json(self.embedder_identity)
            ),
            authorized_responder_calls=scoring["responder_calls_per_shard"],
            authorized_judge_calls=scoring["judge_calls_per_shard"],
            max_prompt_tokens=scoring["max_prompt_tokens"],
            responder_max_output_tokens=scoring[
                "responder_max_output_tokens"
            ],
            judge_max_output_tokens=scoring["judge_max_output_tokens"],
            provider_retries=scoring["provider_retries"],
        )

    def _shard(self, shard: RawStressShard) -> Mem0ShardPolicy:
        row = self.shards.get(shard.sample_offset)
        if row is None:
            raise Mem0PolicyError(
                f"sample offset {shard.sample_offset} is not policy-authorized"
            )
        observed = expected_shard_policy_rows((shard,))[0]
        expected = _plain_json(self.payload)["shards"]
        policy_row = next(
            item for item in expected if item["sample_offset"] == shard.sample_offset
        )
        _must_equal(observed, policy_row, "runtime shard identity")
        return row


def _validate_source_binding(
    value: Any, source_plan: SourceValidationPlan
) -> dict[str, Any]:
    for field in (
        "policy_manifest_sha256",
        "dataset_sha256",
        "split_manifest_sha256",
        "implementation_sha256",
        "environment_lock_sha256",
    ):
        _sha256(getattr(source_plan, field), f"source_plan.{field}")
    evaluation_plain = _plain_json(source_plan.evaluation_identity)
    if source_plan.target_tokens != evaluation_plain.get("stress_context_tokens"):
        raise Mem0PolicyError("source plan stress-token identity is inconsistent")
    if source_plan.questions_per_shard != evaluation_plain.get("stress_questions"):
        raise Mem0PolicyError("source plan shard-size identity is inconsistent")
    if list(source_plan.sample_offsets) != evaluation_plain.get("sample_offsets"):
        raise Mem0PolicyError("source plan offsets are inconsistent")
    source = _mapping(value, "source")
    expected = {
        "validation_policy_sha256": source_plan.policy_manifest_sha256,
        "dataset_sha256": source_plan.dataset_sha256,
        "split_manifest_sha256": source_plan.split_manifest_sha256,
        "implementation_sha256": source_plan.implementation_sha256,
        "environment_lock_sha256": source_plan.environment_lock_sha256,
        "evaluation_identity": evaluation_plain,
        "evaluation_identity_sha256": canonical_json_sha256(
            source_plan.evaluation_identity
        ),
    }
    _exact_keys(source, set(expected), "source")
    _must_equal(source, expected, "source")
    from .prompt_pack import validate_source_evaluation_identity

    validate_source_evaluation_identity(source["evaluation_identity"])
    return source


def _validate_model_identity(value: Any, label: str, *, embedder: bool) -> dict[str, Any]:
    identity = _mapping(value, label)
    expected = {
        "provider",
        "model",
        "revision",
        "model_identity_sha256",
    }
    if embedder:
        expected |= {
            "checkpoint_sha256",
            "dimension",
            "device",
            "dtype",
            "execution",
            "network_calls_authorized",
            "runtime_probe_required",
        }
    else:
        expected |= {
            "provider_retries",
            "logical_call_boundary",
            "logical_calls_per_add",
            "http_attempts_certified",
        }
    _exact_keys(identity, expected, label)
    for field in ("provider", "model", "revision"):
        _text(identity[field], f"{label}.{field}")
    supplied_digest = _sha256(
        identity.pop("model_identity_sha256"), f"{label}.model_identity_sha256"
    )
    if embedder:
        _sha256(identity["checkpoint_sha256"], f"{label}.checkpoint_sha256")
        _integer(identity["dimension"], f"{label}.dimension", minimum=1)
        if identity["device"] not in {"cpu", "cuda"}:
            raise Mem0PolicyError(f"{label}.device must be cpu or cuda")
        _text(identity["dtype"], f"{label}.dtype")
        if identity["execution"] != MEM0_EMBEDDER_EXECUTION:
            raise Mem0PolicyError(
                f"{label}.execution must be {MEM0_EMBEDDER_EXECUTION!r}"
            )
        if identity["network_calls_authorized"] != 0:
            raise Mem0PolicyError(
                f"{label}.network_calls_authorized must be zero"
            )
        if identity["runtime_probe_required"] is not True:
            raise Mem0PolicyError(
                f"{label}.runtime_probe_required must be true"
            )
        exact_embedder = {
            "provider": MEM0_EMBEDDER_PROVIDER,
            "model": MEM0_EMBEDDER_MODEL,
            "revision": MEM0_EMBEDDER_REVISION,
            "checkpoint_sha256": MEM0_EMBEDDER_CHECKPOINT_SHA256,
            "dimension": MEM0_EMBEDDER_DIMENSION,
            "dtype": MEM0_EMBEDDER_DTYPE,
        }
        for field, wanted in exact_embedder.items():
            if identity[field] != wanted:
                raise Mem0PolicyError(
                    f"{label}.{field} does not match the frozen local BGE-M3 arm"
                )
    else:
        if identity["provider_retries"] != 0:
            raise Mem0PolicyError(f"{label}.provider_retries must be zero")
        if identity["logical_call_boundary"] != MEM0_EXTRACTION_BOUNDARY:
            raise Mem0PolicyError(f"{label}.logical_call_boundary mismatch")
        if identity["logical_calls_per_add"] != 1:
            raise Mem0PolicyError(f"{label}.logical_calls_per_add must be one")
        if identity["http_attempts_certified"] is not False:
            raise Mem0PolicyError(
                f"{label}.http_attempts_certified must honestly remain false"
            )
    actual_digest = canonical_json_sha256(identity)
    if supplied_digest != actual_digest:
        raise Mem0PolicyError(f"{label}.model_identity_sha256 mismatch")
    identity["model_identity_sha256"] = supplied_digest
    return identity


def _validate_stable_payload(
    value: Any,
    *,
    extraction: Mapping[str, Any],
    embedder: Mapping[str, Any],
    supplied_digest: str,
) -> dict[str, Any]:
    payload = _mapping(value, "mem0.stable_payload")
    _exact_keys(payload, {"protocol", "config", "stack"}, "mem0.stable_payload")
    if payload["protocol"] != MEM0_RUNTIME_PROTOCOL:
        raise Mem0PolicyError("mem0.stable_payload.protocol mismatch")
    config = _mapping(payload["config"], "mem0.stable_payload.config")
    if config.get("version") != MEM0_API_VERSION:
        raise Mem0PolicyError("Mem0 config API version mismatch")
    if config.get("custom_instructions") is not None:
        raise Mem0PolicyError("Mem0 custom instructions must be null")
    if config.get("reranker") is not None:
        raise Mem0PolicyError("Mem0 reranker must be null")
    if config.get("graph_store") not in (None,):
        raise Mem0PolicyError("the direct comparison arm does not certify graph_store")
    llm = _mapping(config.get("llm"), "mem0 config llm")
    llm_config = _mapping(llm.get("config"), "mem0 config llm.config")
    if llm.get("provider") != extraction["provider"]:
        raise Mem0PolicyError("Mem0 config LLM provider differs from policy")
    if llm_config.get("model") != extraction["model"]:
        raise Mem0PolicyError("Mem0 config LLM model differs from policy")
    embedded = _mapping(config.get("embedder"), "mem0 config embedder")
    embedded_config = _mapping(
        embedded.get("config"), "mem0 config embedder.config"
    )
    if embedded.get("provider") != embedder["provider"]:
        raise Mem0PolicyError("Mem0 config embedder provider differs from policy")
    expected_embedder_config = {
        "model": embedder["model"],
        "embedding_dims": embedder["dimension"],
        "huggingface_base_url": None,
        "model_kwargs": {
            "revision": embedder["revision"],
            "local_files_only": True,
            "trust_remote_code": False,
            "device": embedder["device"],
        },
    }
    _must_equal(
        embedded_config,
        expected_embedder_config,
        "Mem0 local BGE-M3 embedder config",
    )
    vector = _mapping(config.get("vector_store"), "mem0 config vector_store")
    vector_config = _mapping(
        vector.get("config"), "mem0 config vector_store.config"
    )
    if vector.get("provider") != "qdrant":
        raise Mem0PolicyError("Mem0 vector store must be qdrant")
    if vector_config.get("embedding_model_dims") != embedder["dimension"]:
        raise Mem0PolicyError("Mem0 vector dimension differs from embedder")
    if vector_config.get("on_disk") is not True:
        raise Mem0PolicyError("Mem0 Qdrant vectors must be on disk")
    _text(vector_config.get("collection_name"), "Mem0 collection_name")
    vector_path = _text(vector_config.get("path"), "Mem0 Qdrant path")
    history_path = _text(config.get("history_db_path"), "Mem0 history_db_path")
    if not vector_path.startswith("<owned_state>/"):
        raise Mem0PolicyError("Mem0 Qdrant path must be an owned-state placeholder")
    if not history_path.startswith("<owned_state>/"):
        raise Mem0PolicyError("Mem0 history path must be an owned-state placeholder")
    for field in ("url", "host", "port", "api_key", "client"):
        if vector_config.get(field) not in (None, "<redacted>"):
            raise Mem0PolicyError(f"remote/injected Qdrant field {field} is forbidden")
    expected_config = {
        "version": MEM0_API_VERSION,
        "llm": {
            "provider": extraction["provider"],
            "config": {"model": extraction["model"]},
        },
        "embedder": {
            "provider": embedder["provider"],
            "config": expected_embedder_config,
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "longmemeval",
                "embedding_model_dims": embedder["dimension"],
                "on_disk": True,
                "path": "<owned_state>/qdrant",
            },
        },
        "history_db_path": "<owned_state>/history.sqlite",
        "custom_instructions": None,
        "reranker": None,
    }
    _must_equal(config, expected_config, "Mem0 stable config")
    stack = _mapping(payload["stack"], "mem0.stable_payload.stack")
    _exact_keys(
        stack,
        {
            "dependency_versions",
            "bm25_model",
            "spacy_model",
            "bm25_operational",
            "entity_extraction_operational",
        },
        "mem0.stable_payload.stack",
    )
    versions = _mapping(
        stack["dependency_versions"],
        "mem0.stable_payload.stack.dependency_versions",
    )
    required_versions = {
        "mem0ai": "2.0.18",
        "qdrant-client": "1.15.1",
        "fastembed": "0.7.3",
        "spacy": "3.8.7",
        "en-core-web-sm": "3.8.0",
    }
    _must_equal(versions, required_versions, "Mem0 dependency versions")
    if stack["bm25_model"] != MEM0_BM25_MODEL:
        raise Mem0PolicyError("Mem0 BM25 model mismatch")
    if stack["spacy_model"] != MEM0_SPACY_MODEL:
        raise Mem0PolicyError("Mem0 spaCy model mismatch")
    if stack["bm25_operational"] is not True:
        raise Mem0PolicyError("Mem0 BM25 stack must be operational")
    if stack["entity_extraction_operational"] is not True:
        raise Mem0PolicyError("Mem0 entity extraction must be operational")
    if canonical_json_sha256(payload) != supplied_digest:
        raise Mem0PolicyError("mem0.stable_config_sha256 mismatch")
    return payload


def _validate_mem0(value: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    mem0 = _mapping(value, "mem0")
    _exact_keys(
        mem0,
        {
            "runtime_protocol",
            "mem0ai_version",
            "api_version",
            "input_order_protocol",
            "extraction_identity",
            "embedder_identity",
            "search",
            "rendering_mode",
            "storage",
            "provenance",
            "stable_payload",
            "stable_config_sha256",
        },
        "mem0",
    )
    exact = {
        "runtime_protocol": MEM0_RUNTIME_PROTOCOL,
        "mem0ai_version": MEM0AI_PIN,
        "api_version": MEM0_API_VERSION,
        "input_order_protocol": MEM0_INPUT_ORDER_PROTOCOL,
        "rendering_mode": MEM0_CERTIFIED_RENDERING,
    }
    for field, expected in exact.items():
        if mem0[field] != expected:
            raise Mem0PolicyError(f"mem0.{field} mismatch")
    extraction = _validate_model_identity(
        mem0["extraction_identity"], "mem0.extraction_identity", embedder=False
    )
    embedder = _validate_model_identity(
        mem0["embedder_identity"], "mem0.embedder_identity", embedder=True
    )
    search = _mapping(mem0["search"], "mem0.search")
    _must_equal(
        search,
        {"top_k": MEM0_OFFICIAL_TOP_K, "threshold": MEM0_OFFICIAL_THRESHOLD,
         "rerank": False, "explain": False},
        "mem0.search",
    )
    storage = _mapping(mem0["storage"], "mem0.storage")
    _must_equal(
        storage,
        {"provider": "qdrant", "local_owned_state": True, "on_disk": True,
         "fresh_process_per_shard": True, "cleanup_required": True},
        "mem0.storage",
    )
    provenance = _mapping(mem0["provenance"], "mem0.provenance")
    _must_equal(
        provenance,
        {"attribution_kind": MEM0_ATTRIBUTION_KIND,
         "supports_exact_source_provenance": False,
         "source_session_date_exposure": MEM0_SOURCE_SESSION_DATE_EXPOSURE,
         "retrieved_created_at_exposure": MEM0_RETRIEVED_CREATED_AT_EXPOSURE},
        "mem0.provenance",
    )
    stable_digest = _sha256(
        mem0["stable_config_sha256"], "mem0.stable_config_sha256"
    )
    _validate_stable_payload(
        mem0["stable_payload"],
        extraction=extraction,
        embedder=embedder,
        supplied_digest=stable_digest,
    )
    return mem0, extraction, embedder


def _validate_scoring(value: Any, source_plan: SourceValidationPlan) -> dict[str, Any]:
    scoring = _mapping(value, "scoring")
    _exact_keys(
        scoring,
        {
            "responder_identity",
            "responder_identity_sha256",
            "judge_identity",
            "judge_identity_sha256",
            "responder_calls_per_shard",
            "judge_calls_per_shard",
            "provider_retries",
            "max_prompt_tokens",
            "responder_max_output_tokens",
            "judge_max_output_tokens",
        },
        "scoring",
    )
    responder = _mapping(scoring["responder_identity"], "scoring.responder_identity")
    judge = _mapping(scoring["judge_identity"], "scoring.judge_identity")
    _reject_secret_material(responder, "scoring.responder_identity")
    _reject_secret_material(judge, "scoring.judge_identity")
    if responder.get("model") != source_plan.evaluation_identity["responder_model"]:
        raise Mem0PolicyError("scoring responder differs from source policy")
    if judge.get("model") != source_plan.evaluation_identity["judge_model"]:
        raise Mem0PolicyError("scoring judge differs from source policy")
    for name, identity in (("responder", responder), ("judge", judge)):
        supplied = _sha256(
            scoring[f"{name}_identity_sha256"],
            f"scoring.{name}_identity_sha256",
        )
        if supplied != canonical_json_sha256(identity):
            raise Mem0PolicyError(f"scoring.{name}_identity_sha256 mismatch")
    exact = {
        "responder_calls_per_shard": 10,
        "judge_calls_per_shard": 10,
        "provider_retries": 0,
        "max_prompt_tokens": source_plan.evaluation_identity["max_prompt_tokens"],
        "responder_max_output_tokens": source_plan.evaluation_identity[
            "responder_output_token_reserve"
        ],
    }
    for field, expected in exact.items():
        if scoring[field] != expected:
            raise Mem0PolicyError(f"scoring.{field} mismatch")
    _integer(
        scoring["judge_max_output_tokens"],
        "scoring.judge_max_output_tokens",
        minimum=1,
    )
    return scoring


def load_mem0_comparison_policy(
    path: str | Path,
    *,
    source_plan: SourceValidationPlan,
    mem0_environment_lock: str | Path,
    expected_shards: Sequence[RawStressShard],
    tool_root: str | Path | None = None,
) -> Mem0ComparisonPolicy:
    """Load and verify the exact Mem0 model/runtime/call authorization."""

    policy_path = Path(path).resolve()
    lock_path = Path(mem0_environment_lock).resolve()
    root = Path(tool_root).resolve() if tool_root is not None else Path(__file__).parent
    policy_bytes = policy_path.read_bytes()
    lock_bytes = lock_path.read_bytes()
    try:
        value = json.loads(policy_bytes, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Mem0PolicyError(f"cannot parse Mem0 comparison policy: {exc}") from exc
    policy = _mapping(value, "policy")
    _reject_secret_material(policy)
    _exact_keys(
        policy,
        {"format", "status", "arm_id", "source", "tool", "mem0", "scoring", "shards"},
        "policy",
    )
    if policy["format"] != MEM0_POLICY_FORMAT:
        raise Mem0PolicyError("Mem0 policy format mismatch")
    if policy["status"] != MEM0_POLICY_STATUS:
        raise Mem0PolicyError("Mem0 policy status mismatch")
    if policy["arm_id"] != MEM0_ARM_ID:
        raise Mem0PolicyError("Mem0 policy arm mismatch")
    _validate_source_binding(policy["source"], source_plan)
    tool = _mapping(policy["tool"], "tool")
    _exact_keys(
        tool,
        {"implementation_sha256", "environment_lock_sha256"},
        "tool",
    )
    current_tool_digest = tool_implementation_sha256(root)
    current_lock_digest = hashlib.sha256(lock_bytes).hexdigest()
    if tool.get("implementation_sha256") != current_tool_digest:
        raise Mem0PolicyError("Mem0 tool implementation SHA mismatch")
    if tool.get("environment_lock_sha256") != current_lock_digest:
        raise Mem0PolicyError("Mem0 environment lock SHA mismatch")
    mem0, extraction, embedder = _validate_mem0(policy["mem0"])
    scoring = _validate_scoring(policy["scoring"], source_plan)
    expected_rows = list(expected_shard_policy_rows(expected_shards))
    raw_rows = policy["shards"]
    if not isinstance(raw_rows, list):
        raise Mem0PolicyError("policy.shards must be an array")
    _must_equal(raw_rows, expected_rows, "policy.shards")
    if tuple(row["sample_offset"] for row in expected_rows) != source_plan.sample_offsets:
        raise Mem0PolicyError("policy shards do not match source sample offsets")
    shard_map = {
        row["sample_offset"]: Mem0ShardPolicy(
            sample_offset=row["sample_offset"],
            sample_id=row["sample_id"],
            sample_sha256=row["sample_sha256"],
            raw_history_bundle_sha256=row["raw_history_bundle_sha256"],
            question_ids=tuple(row["question_ids"]),
            authorized_add_operations=row["authorized_add_operations"],
            authorized_extraction_calls=row["authorized_extraction_calls"],
            authorized_search_operations=row["authorized_search_operations"],
        )
        for row in expected_rows
    }
    result = Mem0ComparisonPolicy(
        path=policy_path,
        sha256=hashlib.sha256(policy_bytes).hexdigest(),
        environment_lock_path=lock_path,
        environment_lock_sha256=current_lock_digest,
        tool_root=root,
        tool_implementation_sha256=current_tool_digest,
        source_plan=source_plan,
        arm_id=MEM0_ARM_ID,
        stable_config_sha256=mem0["stable_config_sha256"],
        stable_payload=_immutable_json(mem0["stable_payload"]),
        extraction_identity=_immutable_json(extraction),
        embedder_identity=_immutable_json(embedder),
        scoring=_immutable_json(scoring),
        payload=_immutable_json(policy),
        shards=MappingProxyType(shard_map),
    )
    # Repeat all external byte checks after validation.  This catches a policy
    # or lock replacement during parsing before an authorization can escape.
    result.recheck()
    return result


__all__ = [
    "MEM0_ARM_ID",
    "MEM0_EXTRACTION_BOUNDARY",
    "MEM0_POLICY_FORMAT",
    "MEM0_POLICY_STATUS",
    "Mem0ComparisonPolicy",
    "Mem0PolicyError",
    "Mem0ShardPolicy",
    "canonical_json_sha256",
    "expected_shard_policy_rows",
    "load_mem0_comparison_policy",
]
