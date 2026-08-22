"""Locked shard contracts and preparation for cumulative 1M validation.

The historical :mod:`recall_guarded_cumulative_1m` launcher is deliberately
bound to one development concatenation.  This module leaves that contract
untouched and supplies the validation counterpart:

* reconstruct one of the exact validation offsets ``0, 10, ..., 90``;
* use the hash-locked validation manifest only as a retrieval-control source;
* build the current exact-span source and one cumulative store per shard;
* checkpoint the provider-free S0--S3 ladder as canonical JSON; and
* merge exactly ten independently sealed shard artifacts into one ordered
  100-question artifact without inventing a global physical store.

Benchmark answers, labelled evidence sources, and question categories are not
read by the retrieval path.  The population identities expose only hashes of
the question probes; the plaintext questions in retrieval parts are the exact
provider prompts required for the later fixed-stage responder.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import ClosurePolicy, identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.diffuse_compilation import DiffuseCompilationPolicy
from memory_condense.eval.recall_guarded_cumulative import (
    CausalCoveragePredecessorReceipt,
    CumulativeRetrievalLadder,
    CumulativeRetrievalStageReceipt,
    RecallGuardedCumulativeReceipt,
    RecallGuardedCumulativeRetrieval,
    retrieve_recall_guarded_cumulative_packet,
)
from memory_condense.eval.recall_guarded_cumulative_1m import (
    STAGE_IDS,
    _atomic_write_json,
    _canonical_json_bytes,
    _load_shared_qwen,
    _read_canonical_json,
)
from memory_condense.eval.recall_guarded_cumulative_1m_source import (
    CURRENT_SOURCE_FORMAT,
    CURRENT_SOURCE_SCOPE,
    CURRENT_SOURCE_SELECTION_NAME,
    CURRENT_SOURCE_TIMESTAMP_SEMANTICS,
    current_source_binding,
    prepare_current_source_store,
    source_treatment_identity,
    validate_current_source_receipt,
)
from memory_condense.eval.recall_guarded_cumulative_population import (
    LOCKED_100Q_OFFSETS,
    LOCKED_CONTEXT_TARGET_TOKENS,
    LOCKED_LONGMEMEVAL_DATASET_SHA256,
    LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
    LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    LOCKED_QUESTIONS_PER_SHARD,
    LockedCumulativePopulationPlan,
    build_locked_cumulative_population_identity,
    merge_locked_cumulative_shard_identities,
    validate_locked_cumulative_population_identity,
    validate_locked_cumulative_shard_identity,
)
from memory_condense.eval.recall_guarded_cumulative_runtime import (
    CombinedCumulativeStoreReceipt,
    PreparedRecallGuardedCumulativeStore,
    build_recall_guarded_cumulative_store,
    open_recall_guarded_cumulative_store,
)
from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    implementation_sha256,
)
from memory_condense.eval.schemas import ChunkerConfig, EvalConfig, RetrievalConfig
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample
from memory_condense.search.episodes import (
    EpisodeRepresentativeRetrievalPolicy,
    EpisodeRetrievalPolicy,
)


VALIDATION_CAMPAIGN_FORMAT = (
    "memory-condense-recall-guarded-cumulative-1m-validation-campaign-v1"
)
VALIDATION_POLICY_ATTESTATION_FORMAT = (
    "memory-condense-frozen-validation-retrieval-controls-attestation-v1"
)
VALIDATION_EXECUTION_POLICY_FORMAT = (
    "memory-condense-recall-guarded-cumulative-validation-execution-policy-v1"
)
VALIDATION_PREFLIGHT_FORMAT = (
    "memory-condense-recall-guarded-cumulative-validation-preflight-v1"
)
VALIDATION_SHARD_QUESTION_FORMAT = (
    "memory-condense-recall-guarded-cumulative-validation-shard-query-v1"
)
VALIDATION_MERGED_QUESTION_FORMAT = (
    "memory-condense-recall-guarded-cumulative-validation-merged-query-v1"
)
VALIDATION_SHARD_RETRIEVAL_FORMAT = (
    "memory-condense-recall-guarded-cumulative-validation-shard-retrieval-v1"
)
VALIDATION_SHARD_REFERENCE_FORMAT = (
    "memory-condense-recall-guarded-cumulative-validation-shard-reference-v1"
)
VALIDATION_EXTERNAL_RECONSTRUCTION_FORMAT = (
    "memory-condense-recall-guarded-cumulative-validation-reconstruction-v1"
)
VALIDATION_MERGED_RETRIEVAL_FORMAT = (
    "memory-condense-recall-guarded-cumulative-validation-100q-retrieval-v1"
)

LOCKED_VALIDATION_POLICY_MANIFEST_SHA256 = (
    "5263d5afd15298ec4088db9d6381ae243ddb685e9a3cf4d9892fc84e14fb9883"
)
MAX_CONTEXT_TOKENS = 7_000
MAX_PROMPT_TOKENS = 8_000
RESPONDER_OUTPUT_TOKEN_RESERVE = 256
SOURCE_ROUTER_MAX_SOURCES = 64
SOURCE_ROUTER_RRF_CONSTANT = 60

DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
)
DEFAULT_SPLIT = Path(
    "docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"
)
DEFAULT_POLICY = Path(
    "docs/10 - Research Log/data/"
    "longmemeval-qwen-choice-coverage-operational-validation-v3.json"
)
DEFAULT_QWEN_PREFIX = Path(".cache/models/Qwen3-8B")
DEFAULT_QWEN_CHOICE = Path(".cache/models/Qwen3-0.6B")

_SHA256_ALPHABET = frozenset("0123456789abcdef")
_GOLD_FIELD_NAMES = frozenset(
    {
        "answer",
        "answers",
        "gold",
        "gold_answer",
        "category",
        "question_type",
        "evidence_sources",
        "answer_session_ids",
    }
)


class _UnboundCoverageSelector:
    """Construction sentinel replaced before any retrieval is permitted."""

    strict = True
    requires_baseline_ranking = True
    requires_complete_frontier = True

    def select(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("coverage selector was not bound before retrieval")


@dataclass(frozen=True, slots=True)
class FrozenValidationPolicy:
    """Resolved runtime controls plus an explicit legacy-manifest attestation."""

    config: EvalConfig
    attestation: Mapping[str, Any]
    execution_policy: Mapping[str, Any]

    @property
    def attestation_sha256(self) -> str:
        return str(self.attestation["attestation_sha256"])

    @property
    def execution_policy_sha256(self) -> str:
        return identity_sha256(dict(self.execution_policy))

    @property
    def retrieval_policy_sha256(self) -> str:
        return identity_sha256(self.config.retrieval.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class ValidationShardPreflight:
    """In-memory, gold-blind inputs for one exact validation shard."""

    sample: BenchmarkSample
    shard_identity: Mapping[str, Any]
    population_identity: Mapping[str, Any]
    policy: FrozenValidationPolicy
    sample_offset: int
    shard_root: Path
    qwen_prefix_model_dir: Path
    qwen_choice_model_dir: Path
    retrieval_implementation_sha256: str
    environment_lock_sha256: str
    source_embedding_device: str

    def public_report(self) -> dict[str, Any]:
        return {
            "format": VALIDATION_PREFLIGHT_FORMAT,
            "campaign_format": VALIDATION_CAMPAIGN_FORMAT,
            "sample_offset": self.sample_offset,
            "shard_root": str(self.shard_root),
            "shard_identity_sha256": self.shard_identity[
                "shard_identity_sha256"
            ],
            "population_identity_sha256": self.population_identity[
                "population_identity_sha256"
            ],
            "validation_policy_manifest_sha256": (
                LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
            ),
            "validation_policy_attestation_sha256": (
                self.policy.attestation_sha256
            ),
            "validation_execution_policy_sha256": (
                self.policy.execution_policy_sha256
            ),
            "retrieval_policy_sha256": self.policy.retrieval_policy_sha256,
            "retrieval_implementation_sha256": (
                self.retrieval_implementation_sha256
            ),
            "environment_lock_sha256": self.environment_lock_sha256,
            "source_embedding_device": self.source_embedding_device,
            "qwen_prefix_model_dir": str(self.qwen_prefix_model_dir),
            "qwen_choice_model_dir": str(self.qwen_choice_model_dir),
            "question_count": len(self.sample.questions),
            "provider_calls": 0,
            "gold_fields_present": False,
        }


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_ALPHABET for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{label} has an unexpected schema; missing={missing}, extra={extra}")


def _assert_gold_blind_schema(value: object, *, label: str) -> None:
    """Reject benchmark-label field names anywhere in an exported artifact."""

    if isinstance(value, Mapping):
        prohibited = _GOLD_FIELD_NAMES & {str(key) for key in value}
        if prohibited:
            raise ValueError(f"{label} contains gold-bearing fields: {sorted(prohibited)}")
        for key, child in value.items():
            _assert_gold_blind_schema(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_gold_blind_schema(child, label=f"{label}[{index}]")


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(body)
    value[field] = identity_sha256(value)
    return value


def _direct_episode_controls() -> dict[str, Any]:
    return {
        "max_anchor_episodes": 96,
        "previous_episodes": 1,
        "next_episodes": 1,
        "max_episode_seeds": 256,
        "max_direct_fallbacks": 96,
    }


def _representative_episode_controls() -> dict[str, Any]:
    return {
        "max_input_sources": 64,
        "max_source_groups": 64,
        "max_episodes_per_source": 64,
        "max_total_episodes": 256,
        "max_representatives_per_episode": 2,
        "group_size": 8,
        "beam_per_group": 2,
        "top_k": 8,
        "representative_tokens": 96,
        "query_tokens": 96,
        "score_mode": "qk_ov",
    }


def _closure_controls() -> dict[str, Any]:
    return {
        "max_hops": 3,
        "max_units": 1024,
        "max_relations": 2048,
        "max_degree": 32,
        "max_episode_neighbors": 2,
        "max_frontier": 1024,
        "max_bundles": 256,
        "beam_width": 128,
        "min_relation_confidence": 0.5,
    }


def _episode_policy(artifact_id: str) -> EpisodeRetrievalPolicy:
    return EpisodeRetrievalPolicy(artifact_id=artifact_id, **_direct_episode_controls())


def _representative_policy(
    artifact_id: str,
) -> EpisodeRepresentativeRetrievalPolicy:
    return EpisodeRepresentativeRetrievalPolicy(
        artifact_id=artifact_id,
        **_representative_episode_controls(),
    )


def _closure_policy() -> ClosurePolicy:
    return ClosurePolicy(**_closure_controls())


def _validation_execution_policy(
    *,
    policy_attestation_sha256: str,
    retrieval_policy_sha256: str,
) -> dict[str, Any]:
    return {
        "format": VALIDATION_EXECUTION_POLICY_FORMAT,
        "validation_policy_manifest_sha256": (
            LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
        ),
        "validation_policy_attestation_sha256": policy_attestation_sha256,
        "resolved_retrieval_policy_sha256": retrieval_policy_sha256,
        "population_plan": {
            "dataset_sha256": LOCKED_LONGMEMEVAL_DATASET_SHA256,
            "split_manifest_sha256": (
                LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256
            ),
            "split": "validation",
            "target_tokens": LOCKED_CONTEXT_TARGET_TOKENS,
            "questions_per_shard": LOCKED_QUESTIONS_PER_SHARD,
            "ordered_shard_offsets": list(LOCKED_100Q_OFFSETS),
        },
        "source_timestamp_semantics": CURRENT_SOURCE_TIMESTAMP_SEMANTICS,
        "source_scope": "current_exact_span_gold_blind_per_shard",
        "combined_store_compilation": {
            "boundary_mode": "fixed_interval",
            "held_out_queries": "exact retrieval and dated question strings",
            "persisted_request_token_state_bytes": 0,
        },
        "stage_ids": list(STAGE_IDS),
        "direct_episode_controls": _direct_episode_controls(),
        "representative_episode_controls": _representative_episode_controls(),
        "closure_controls": _closure_controls(),
        "source_router": {
            "max_sources": SOURCE_ROUTER_MAX_SOURCES,
            "rrf_constant": SOURCE_ROUTER_RRF_CONSTANT,
        },
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "responder_output_token_reserve": RESPONDER_OUTPUT_TOKEN_RESERVE,
        "require_certified_coverage_runtime": True,
        "require_owned_representative_runtime": True,
        "provider_calls": 0,
        "gold_blind": True,
    }


def load_frozen_validation_policy(
    policy_path: str | Path,
    *,
    device: str,
) -> FrozenValidationPolicy:
    """Load the exact frozen validation controls without claiming old code."""

    path = Path(policy_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen validation policy is missing: {path}")
    observed_sha = file_sha256(path)
    if observed_sha != LOCKED_VALIDATION_POLICY_MANIFEST_SHA256:
        raise ValueError(
            "validation policy SHA-256 mismatch: "
            f"{observed_sha} != {LOCKED_VALIDATION_POLICY_MANIFEST_SHA256}"
        )
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, Mapping):
        raise ValueError("frozen validation policy must be a JSON object")
    expected = {
        "format": "memory-condense-retrieval-policy-v1",
        "status": "validation_frozen",
        "dataset_sha256": LOCKED_LONGMEMEVAL_DATASET_SHA256,
        "split_manifest_sha256": LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
        "split": "validation",
        "claim_profile": "longmemeval-s-1m-100q-95-v1",
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise ValueError("validation policy does not describe the locked 100Q split")
    retrieval_raw = payload.get("retrieval")
    evaluation = payload.get("evaluation")
    if not isinstance(retrieval_raw, Mapping) or not isinstance(evaluation, Mapping):
        raise ValueError("validation policy omitted retrieval/evaluation controls")
    if (
        evaluation.get("min_target_questions") != 100
        or evaluation.get("stress_context_tokens") != LOCKED_CONTEXT_TARGET_TOKENS
        or evaluation.get("stress_questions") != LOCKED_QUESTIONS_PER_SHARD
        or tuple(evaluation.get("sample_offsets", ())) != LOCKED_100Q_OFFSETS
        or evaluation.get("max_prompt_tokens") != MAX_PROMPT_TOKENS
        or retrieval_raw.get("max_prompt_tokens") != MAX_PROMPT_TOKENS
    ):
        raise ValueError("validation policy changed its 100Q stress controls")
    retrieval_body = dict(retrieval_raw)
    min_tokens = _require_exact_int(
        retrieval_body.pop("chunker_min_tokens", None),
        "chunker_min_tokens",
        minimum=1,
    )
    max_tokens = _require_exact_int(
        retrieval_body.pop("chunker_max_tokens", None),
        "chunker_max_tokens",
        minimum=min_tokens,
    )
    manifest_prompt = _require_exact_int(
        retrieval_body.pop("max_prompt_tokens", None),
        "max_prompt_tokens",
        minimum=1,
    )
    retrieval = RetrievalConfig(**retrieval_body)
    config = EvalConfig(
        chunker=ChunkerConfig(min_tokens=min_tokens, max_tokens=max_tokens),
        retrieval=retrieval,
        embedding_device=str(device),
        max_prompt_tokens=manifest_prompt,
        min_target_questions=100,
        accuracy_target=0.95,
    )
    resolved_sha = identity_sha256(retrieval.model_dump(mode="json"))
    raw_controls_sha = identity_sha256(dict(retrieval_raw))
    declared_implementation = _require_sha256(
        payload.get("implementation_sha256"),
        "manifest-declared implementation SHA-256",
    )
    declared_environment = _require_sha256(
        payload.get("environment_lock_sha256"),
        "manifest-declared environment-lock SHA-256",
    )
    attestation_body = {
        "format": VALIDATION_POLICY_ATTESTATION_FORMAT,
        "manifest_sha256": LOCKED_VALIDATION_POLICY_MANIFEST_SHA256,
        "manifest_status": "validation_frozen",
        "manifest_split": "validation",
        "manifest_claim_profile": "longmemeval-s-1m-100q-95-v1",
        "manifest_declared_implementation_sha256": declared_implementation,
        "manifest_declared_environment_lock_sha256": declared_environment,
        "manifest_retrieval_controls_sha256": raw_controls_sha,
        "resolved_retrieval_policy_sha256": resolved_sha,
        "usage": (
            "retrieval controls and evaluation budgets only; current execution "
            "implementation and environment are separately rebound"
        ),
    }
    attestation = _self_hashed(attestation_body, "attestation_sha256")
    execution = _validation_execution_policy(
        policy_attestation_sha256=attestation["attestation_sha256"],
        retrieval_policy_sha256=resolved_sha,
    )
    return FrozenValidationPolicy(
        config=config,
        attestation=attestation,
        execution_policy=execution,
    )


def _validate_policy_binding(policy: FrozenValidationPolicy) -> None:
    attestation = dict(policy.attestation)
    declared_attestation = _require_sha256(
        attestation.pop("attestation_sha256", None),
        "validation policy attestation SHA-256",
    )
    if declared_attestation != identity_sha256(attestation):
        raise ValueError("validation policy attestation seal changed")
    if policy.execution_policy != _validation_execution_policy(
        policy_attestation_sha256=declared_attestation,
        retrieval_policy_sha256=policy.retrieval_policy_sha256,
    ):
        raise ValueError("validation execution policy changed")


def shard_output_root(output_root: str | Path, sample_offset: int) -> Path:
    offset = _require_exact_int(sample_offset, "sample_offset")
    if offset not in LOCKED_100Q_OFFSETS:
        raise ValueError("sample offset must be one of 0,10,...,90")
    return Path(output_root).resolve() / "shards" / f"offset-{offset:03d}"


def preflight_locked_validation_shard(
    *,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    policy_path: str | Path,
    output_root: str | Path,
    sample_offset: int,
    qwen_prefix_model_dir: str | Path,
    qwen_choice_model_dir: str | Path,
    device: str = "cuda",
    require_model_directories: bool = True,
    plan: LockedCumulativePopulationPlan = LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
) -> ValidationShardPreflight:
    """Reconstruct all campaign identities before touching a model or store."""

    if plan != LOCKED_LONGMEMEVAL_VALIDATION_PLAN:
        raise ValueError("validation retrieval requires the exact locked 100Q plan")
    offset = _require_exact_int(sample_offset, "sample_offset")
    if offset not in plan.shard_offsets:
        raise ValueError("sample offset is not part of the locked validation plan")
    samples, shard_identities, population = (
        build_locked_cumulative_population_identity(
            dataset_path,
            split_manifest_path,
            plan=plan,
        )
    )
    index = plan.shard_offsets.index(offset)
    sample = samples[index]
    shard_identity = shard_identities[index]
    policy = load_frozen_validation_policy(policy_path, device=device)
    _validate_policy_binding(policy)
    prefix = Path(qwen_prefix_model_dir).resolve()
    choice = Path(qwen_choice_model_dir).resolve()
    if require_model_directories:
        if not prefix.is_dir():
            raise FileNotFoundError(f"Qwen prefix checkpoint is missing: {prefix}")
        if not choice.is_dir():
            raise FileNotFoundError(f"Qwen choice checkpoint is missing: {choice}")
    return ValidationShardPreflight(
        sample=sample,
        shard_identity=shard_identity,
        population_identity=population,
        policy=policy,
        sample_offset=offset,
        shard_root=shard_output_root(output_root, offset),
        qwen_prefix_model_dir=prefix,
        qwen_choice_model_dir=choice,
        retrieval_implementation_sha256=implementation_sha256(),
        environment_lock_sha256=environment_lock_sha256(),
        source_embedding_device=str(device).casefold(),
    )


def _held_out_queries(sample: BenchmarkSample) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            text
            for question in sample.questions
            for text in (question.question, question.dated_question)
        )
    )


def prepare_validation_source(
    preflight: ValidationShardPreflight,
) -> tuple[Any, Path, dict[str, Any], str]:
    """Prepare or exhaustively verify one shard's exact-span source."""

    source_config, binding = current_source_binding(
        preflight.policy.config,
        qwen_model_dir=preflight.qwen_prefix_model_dir,
    )
    source_root = preflight.shard_root / "source-current"
    try:
        database, receipt, mode = prepare_current_source_store(
            sample=preflight.sample,
            config=source_config,
            treatment_identity=source_treatment_identity(
                preflight.sample,
                dataset_sha256=LOCKED_LONGMEMEVAL_DATASET_SHA256,
                split_manifest_sha256=(
                    LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256
                ),
                sanitized_projection_sha256=str(
                    preflight.shard_identity["shard_identity_sha256"]
                ),
            ),
            binding=binding,
            source_root=source_root,
            selection_path=(
                preflight.shard_root / CURRENT_SOURCE_SELECTION_NAME
            ),
        )
        selected = validate_current_source_receipt(
            receipt,
            sample=preflight.sample,
            expected_device=preflight.source_embedding_device,
        )
    except BaseException:
        binding.embedder.close()
        raise
    return binding, database, selected, mode


def prepare_validation_store(
    preflight: ValidationShardPreflight,
) -> tuple[
    PreparedRecallGuardedCumulativeStore,
    Any,
    str,
    dict[str, Any],
    str,
]:
    """Build once or verify/reopen one shard's cumulative retrieval store."""

    config = preflight.policy.config
    binding, source_database, source_receipt, source_mode = (
        prepare_validation_source(preflight)
    )
    embedder = binding.embedder
    combined_dir = preflight.shard_root / "combined-store"
    sentinel = _UnboundCoverageSelector()
    try:
        if combined_dir.exists():
            prepared = open_recall_guarded_cumulative_store(
                combined_dir,
                config=config,
                embedder=embedder,
                held_out_queries=_held_out_queries(preflight.sample),
                coverage_selector=sentinel,
            )
            mode = "verified_cache_hit"
        else:
            prepared = build_recall_guarded_cumulative_store(
                source_database,
                combined_dir,
                config=config,
                embedder=embedder,
                held_out_queries=_held_out_queries(preflight.sample),
                compilation_policy=DiffuseCompilationPolicy(
                    boundary_mode="fixed_interval"
                ),
                coverage_selector=sentinel,
                embedding_identity={
                    "backend": "sentence-transformers.encode-v1",
                    "model_id": "BAAI/bge-m3",
                    "dimension": 1024,
                },
            )
            mode = "fresh_atomic_build"
        if (
            prepared.receipt.source_database_sha256
            != source_receipt["database_sha256"]
            or prepared.receipt.turn_count != source_receipt["turn_count"]
            or prepared.receipt.chunk_count != source_receipt["chunk_count"]
            or prepared.receipt.retrieval_policy_sha256
            != preflight.policy.retrieval_policy_sha256
        ):
            prepared.close()
            raise RuntimeError(
                "combined store does not bind the selected source and policy"
            )
    except BaseException:
        embedder.close()
        raise
    return prepared, embedder, mode, source_receipt, source_mode


def _validated_messages(value: object) -> list[dict[str, str]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("retrieval stage provider messages are missing")
    messages: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {"role", "content"}:
            raise ValueError(
                f"provider message {index} has a noncanonical shape"
            )
        role, content = raw.get("role"), raw.get("content")
        if not isinstance(role, str) or not role or not isinstance(content, str):
            raise ValueError(f"provider message {index} requires strings")
        messages.append({"role": role, "content": content})
    return messages


def _extract_stage_question(stage: Mapping[str, Any]) -> str:
    messages = _validated_messages(stage.get("provider_messages"))
    users = [item["content"] for item in messages if item["role"] == "user"]
    if not users:
        raise ValueError("retrieval stage prompt has no user message")
    marker, suffix = "\n\nQuestion: ", "\nShort answer:"
    content = users[-1]
    if marker not in content or suffix not in content:
        raise ValueError("cannot recover question from sealed retrieval prompt")
    question = content.rsplit(marker, 1)[1].rsplit(suffix, 1)[0].strip()
    if not question:
        raise ValueError("sealed retrieval prompt contains an empty question")
    return question


def _stage_rows(result: RecallGuardedCumulativeRetrieval) -> list[dict[str, Any]]:
    messages = result.provider_messages_by_stage()
    evidence: list[dict[str, str]] = [
        {
            "evidence_id": evidence_id,
            "source_id": excerpt.source_id,
            "text": excerpt.text,
        }
        for evidence_id, excerpt in zip(
            result.ladder.stages[0].selected_evidence_ids,
            result.predecessor.excerpts,
            strict=True,
        )
    ]
    rows: list[dict[str, Any]] = []
    for index, stage in enumerate(result.ladder.stages):
        if index:
            packet = result.addition_packets[index - 1]
            if packet is not None:
                evidence.extend(
                    {
                        "evidence_id": evidence_id,
                        "source_id": atom.span.source_id,
                        "text": atom.text,
                    }
                    for evidence_id, atom in zip(
                        stage.added_evidence_ids,
                        packet.atoms,
                        strict=True,
                    )
                )
        if tuple(item["evidence_id"] for item in evidence) != (
            stage.selected_evidence_ids
        ):
            raise RuntimeError("stage evidence changed its sealed coordinates")
        rows.append(
            {
                "stage_id": stage.stage_id,
                "stage_receipt": asdict(stage),
                "provider_messages": [
                    dict(item) for item in messages[stage.stage_id]
                ],
                "evidence": [dict(item) for item in evidence],
            }
        )
    return rows


def _question_part(
    result: RecallGuardedCumulativeRetrieval,
    *,
    question: BenchmarkQuestion,
    local_ordinal: int,
    preflight: ValidationShardPreflight,
    source_store_receipt_sha256: str,
    combined_store_receipt_sha256: str,
    compilation_receipt_sha256: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    global_ordinal = preflight.sample_offset + local_ordinal
    probe = preflight.shard_identity["ordered_question_probes"][local_ordinal]
    part = {
        "format": VALIDATION_SHARD_QUESTION_FORMAT,
        "population_identity_sha256": preflight.population_identity[
            "population_identity_sha256"
        ],
        "shard_identity_sha256": preflight.shard_identity[
            "shard_identity_sha256"
        ],
        "shard_offset": preflight.sample_offset,
        "local_ordinal": local_ordinal,
        "ordinal": global_ordinal,
        "question_id": question.question_id,
        "question_id_sha256": identity_sha256(
            {"question_id": question.question_id}
        ),
        "question_sha256": quote_sha256(question.question),
        "dated_question_sha256": quote_sha256(question.dated_question),
        "probe_identity_sha256": probe["probe_identity_sha256"],
        "validation_policy_manifest_sha256": (
            LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
        ),
        "validation_policy_attestation_sha256": (
            preflight.policy.attestation_sha256
        ),
        "validation_execution_policy_sha256": (
            preflight.policy.execution_policy_sha256
        ),
        "retrieval_policy_sha256": preflight.policy.retrieval_policy_sha256,
        "retrieval_implementation_sha256": (
            preflight.retrieval_implementation_sha256
        ),
        "environment_lock_sha256": preflight.environment_lock_sha256,
        "source_store_receipt_sha256": source_store_receipt_sha256,
        "combined_store_receipt_sha256": combined_store_receipt_sha256,
        "compilation_receipt_sha256": compilation_receipt_sha256,
        "retrieval_receipt": asdict(result.receipt),
        "predecessor_receipt": asdict(result.predecessor.receipt),
        "stage_ids": list(STAGE_IDS),
        "stages": _stage_rows(result),
        "elapsed_seconds": elapsed_seconds,
        "provider_calls": 0,
    }
    _assert_gold_blind_schema(part, label="retrieval question")
    return part


_SHARD_QUESTION_FIELDS = frozenset(
    {
        "format",
        "population_identity_sha256",
        "shard_identity_sha256",
        "shard_offset",
        "local_ordinal",
        "ordinal",
        "question_id",
        "question_id_sha256",
        "question_sha256",
        "dated_question_sha256",
        "probe_identity_sha256",
        "validation_policy_manifest_sha256",
        "validation_policy_attestation_sha256",
        "validation_execution_policy_sha256",
        "retrieval_policy_sha256",
        "retrieval_implementation_sha256",
        "environment_lock_sha256",
        "source_store_receipt_sha256",
        "combined_store_receipt_sha256",
        "compilation_receipt_sha256",
        "retrieval_receipt",
        "predecessor_receipt",
        "stage_ids",
        "stages",
        "elapsed_seconds",
        "provider_calls",
    }
)
_MERGED_QUESTION_FIELDS = _SHARD_QUESTION_FIELDS | frozenset(
    {"source_shard_retrieval_sha256", "source_question_part_sha256"}
)


def _validate_sealed_question_payload(part: Mapping[str, Any]) -> None:
    """Validate the store-independent, provider-visible S0--S3 payload."""

    elapsed = part.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ValueError("retrieval question elapsed time must be finite")
    for name in (
        "question_id_sha256",
        "question_sha256",
        "dated_question_sha256",
        "probe_identity_sha256",
        "source_store_receipt_sha256",
        "combined_store_receipt_sha256",
        "compilation_receipt_sha256",
        "retrieval_implementation_sha256",
        "environment_lock_sha256",
    ):
        _require_sha256(part[name], f"retrieval question {name}")
    stages = part.get("stages")
    if not isinstance(stages, list) or tuple(
        stage.get("stage_id") if isinstance(stage, Mapping) else None
        for stage in stages
    ) != STAGE_IDS:
        raise ValueError("retrieval question changed its cumulative stages")
    typed_stages: list[CumulativeRetrievalStageReceipt] = []
    parent_ids: tuple[str, ...] = ()
    for index, (expected_stage, stage) in enumerate(
        zip(STAGE_IDS, stages, strict=True)
    ):
        assert isinstance(stage, Mapping)
        _require_exact_keys(
            stage,
            frozenset(
                {"stage_id", "stage_receipt", "provider_messages", "evidence"}
            ),
            "retrieval stage",
        )
        if stage["stage_id"] != expected_stage:
            raise ValueError("retrieval stage order changed")
        raw_receipt = stage.get("stage_receipt")
        if not isinstance(raw_receipt, Mapping):
            raise ValueError("retrieval stage receipt is missing")
        typed = CumulativeRetrievalStageReceipt(**dict(raw_receipt))
        typed_stages.append(typed)
        evidence = stage.get("evidence")
        if not isinstance(evidence, list):
            raise ValueError("retrieval stage evidence must be a list")
        ids: list[str] = []
        for evidence_row in evidence:
            if not isinstance(evidence_row, Mapping):
                raise ValueError("retrieval evidence row must be an object")
            _require_exact_keys(
                evidence_row,
                frozenset({"evidence_id", "source_id", "text"}),
                "retrieval evidence row",
            )
            if any(
                not isinstance(evidence_row.get(name), str)
                or not str(evidence_row[name]).strip()
                for name in ("evidence_id", "source_id", "text")
            ):
                raise ValueError("retrieval evidence row is incomplete")
            ids.append(str(evidence_row["evidence_id"]))
        if tuple(ids) != typed.selected_evidence_ids:
            raise ValueError("retrieval stage evidence coordinates changed")
        if index and tuple(ids[: len(parent_ids)]) != parent_ids:
            raise ValueError("retrieval stages are no longer cumulative")
        messages = _validated_messages(stage.get("provider_messages"))
        if (
            identity_sha256(messages) != typed.prompt_messages_sha256
            or count_chat_prompt_token_proxy(messages) != typed.prompt_token_proxy
            or typed.max_prompt_token_proxy != MAX_PROMPT_TOKENS
            or typed.responder_output_token_reserve
            != RESPONDER_OUTPUT_TOKEN_RESERVE
            or typed.max_context_token_proxy != MAX_CONTEXT_TOKENS
        ):
            raise ValueError("retrieval stage prompt/budget seal changed")
        if quote_sha256(_extract_stage_question(stage)) != part[
            "dated_question_sha256"
        ]:
            raise ValueError("retrieval stage changed its dated question")
        parent_ids = tuple(ids)
    ladder = CumulativeRetrievalLadder(stages=tuple(typed_stages))
    final = RecallGuardedCumulativeReceipt(
        **dict(part.get("retrieval_receipt", {}))
    )
    predecessor = CausalCoveragePredecessorReceipt(
        **dict(part.get("predecessor_receipt", {}))
    )
    if (
        predecessor.retrieval_query_sha256 != part["question_sha256"]
        or predecessor.prompt_question_sha256
        != part["dated_question_sha256"]
        or predecessor.retrieval_policy_sha256
        != part["retrieval_policy_sha256"]
        or predecessor.prompt_messages_sha256
        != typed_stages[0].prompt_messages_sha256
        or final.ladder_receipt_sha256 != ladder.receipt_sha256
        or final.predecessor_receipt_sha256 != predecessor.receipt_sha256
        or final.prompt_messages_sha256
        != typed_stages[-1].prompt_messages_sha256
    ):
        raise ValueError("retrieval question receipts no longer cross-bind")
    _assert_gold_blind_schema(part, label="retrieval question")


def _validate_question_part(
    part: Mapping[str, Any],
    *,
    question: BenchmarkQuestion,
    local_ordinal: int,
    preflight: ValidationShardPreflight,
    source_store_receipt_sha256: str,
    combined_store_receipt_sha256: str,
    compilation_receipt_sha256: str,
    merged: bool = False,
    source_shard_retrieval_sha256: str | None = None,
    source_question_part_sha256: str | None = None,
) -> None:
    expected_fields = _MERGED_QUESTION_FIELDS if merged else _SHARD_QUESTION_FIELDS
    _require_exact_keys(part, expected_fields, "retrieval question")
    expected_format = (
        VALIDATION_MERGED_QUESTION_FORMAT
        if merged
        else VALIDATION_SHARD_QUESTION_FORMAT
    )
    global_ordinal = preflight.sample_offset + local_ordinal
    probe = preflight.shard_identity["ordered_question_probes"][local_ordinal]
    expected = {
        "format": expected_format,
        "population_identity_sha256": preflight.population_identity[
            "population_identity_sha256"
        ],
        "shard_identity_sha256": preflight.shard_identity[
            "shard_identity_sha256"
        ],
        "shard_offset": preflight.sample_offset,
        "local_ordinal": local_ordinal,
        "ordinal": global_ordinal,
        "question_id": question.question_id,
        "question_id_sha256": identity_sha256(
            {"question_id": question.question_id}
        ),
        "question_sha256": quote_sha256(question.question),
        "dated_question_sha256": quote_sha256(question.dated_question),
        "probe_identity_sha256": probe["probe_identity_sha256"],
        "validation_policy_manifest_sha256": (
            LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
        ),
        "validation_policy_attestation_sha256": (
            preflight.policy.attestation_sha256
        ),
        "validation_execution_policy_sha256": (
            preflight.policy.execution_policy_sha256
        ),
        "retrieval_policy_sha256": preflight.policy.retrieval_policy_sha256,
        "retrieval_implementation_sha256": (
            preflight.retrieval_implementation_sha256
        ),
        "environment_lock_sha256": preflight.environment_lock_sha256,
        "source_store_receipt_sha256": source_store_receipt_sha256,
        "combined_store_receipt_sha256": combined_store_receipt_sha256,
        "compilation_receipt_sha256": compilation_receipt_sha256,
        "stage_ids": list(STAGE_IDS),
        "provider_calls": 0,
    }
    if merged:
        expected.update(
            {
                "source_shard_retrieval_sha256": (
                    source_shard_retrieval_sha256
                ),
                "source_question_part_sha256": source_question_part_sha256,
            }
        )
    if any(part.get(name) != value for name, value in expected.items()):
        raise ValueError("retrieval question belongs to another shard/campaign")
    _validate_sealed_question_payload(part)


_SHARD_RETRIEVAL_FIELDS = frozenset(
    {
        "format",
        "campaign_format",
        "population_identity",
        "population_identity_sha256",
        "shard_identity",
        "shard_identity_sha256",
        "shard_offset",
        "validation_policy_attestation",
        "validation_policy_attestation_sha256",
        "validation_policy_manifest_sha256",
        "validation_execution_policy",
        "validation_execution_policy_sha256",
        "retrieval_policy_sha256",
        "retrieval_implementation_sha256",
        "environment_lock_sha256",
        "source_embedding_device",
        "source_timestamp_semantics",
        "source_store_mode",
        "source_store_receipt",
        "source_store_receipt_sha256",
        "combined_store_mode",
        "combined_store_receipt",
        "combined_store_receipt_sha256",
        "compilation_receipt_sha256",
        "transcript_tokens",
        "turn_count",
        "question_count",
        "stage_ids",
        "question_part_sha256s",
        "questions",
        "provider_calls",
        "gold_fields_present",
    }
)


def _validate_shard_head(
    retrieval: Mapping[str, Any],
    *,
    preflight: ValidationShardPreflight,
) -> tuple[dict[str, Any], CombinedCumulativeStoreReceipt]:
    _require_exact_keys(
        retrieval,
        _SHARD_RETRIEVAL_FIELDS,
        "validation shard retrieval",
    )
    validate_locked_cumulative_shard_identity(preflight.shard_identity)
    validate_locked_cumulative_population_identity(
        preflight.population_identity,
        plan=LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    )
    _validate_policy_binding(preflight.policy)
    expected = {
        "format": VALIDATION_SHARD_RETRIEVAL_FORMAT,
        "campaign_format": VALIDATION_CAMPAIGN_FORMAT,
        "population_identity": dict(preflight.population_identity),
        "population_identity_sha256": preflight.population_identity[
            "population_identity_sha256"
        ],
        "shard_identity": dict(preflight.shard_identity),
        "shard_identity_sha256": preflight.shard_identity[
            "shard_identity_sha256"
        ],
        "shard_offset": preflight.sample_offset,
        "validation_policy_attestation": dict(preflight.policy.attestation),
        "validation_policy_attestation_sha256": (
            preflight.policy.attestation_sha256
        ),
        "validation_policy_manifest_sha256": (
            LOCKED_VALIDATION_POLICY_MANIFEST_SHA256
        ),
        "validation_execution_policy": dict(
            preflight.policy.execution_policy
        ),
        "validation_execution_policy_sha256": (
            preflight.policy.execution_policy_sha256
        ),
        "retrieval_policy_sha256": preflight.policy.retrieval_policy_sha256,
        "retrieval_implementation_sha256": (
            preflight.retrieval_implementation_sha256
        ),
        "environment_lock_sha256": preflight.environment_lock_sha256,
        "source_embedding_device": preflight.source_embedding_device,
        "source_timestamp_semantics": CURRENT_SOURCE_TIMESTAMP_SEMANTICS,
        "transcript_tokens": preflight.shard_identity["transcript_tokens"],
        "turn_count": preflight.shard_identity["turn_count"],
        "question_count": LOCKED_QUESTIONS_PER_SHARD,
        "stage_ids": list(STAGE_IDS),
        "provider_calls": 0,
        "gold_fields_present": False,
    }
    if any(retrieval.get(name) != value for name, value in expected.items()):
        raise ValueError("validation shard retrieval belongs to another campaign")
    if retrieval.get("source_store_mode") not in {
        "verified_cache_hit",
        "fresh_or_recovered_atomic_publication",
    }:
        raise ValueError("validation shard has an invalid source-store mode")
    if retrieval.get("combined_store_mode") not in {
        "verified_cache_hit",
        "fresh_atomic_build",
    }:
        raise ValueError("validation shard has an invalid combined-store mode")
    source_receipt = validate_current_source_receipt(
        retrieval.get("source_store_receipt"),
        sample=preflight.sample,
        expected_device=preflight.source_embedding_device,
    )
    if source_receipt["receipt_sha256"] != retrieval.get(
        "source_store_receipt_sha256"
    ):
        raise ValueError("validation shard changed its source-store receipt")
    raw_combined = retrieval.get("combined_store_receipt")
    if not isinstance(raw_combined, Mapping):
        raise ValueError("validation shard omitted its combined-store receipt")
    combined = CombinedCumulativeStoreReceipt(**dict(raw_combined))
    if (
        combined.receipt_sha256
        != retrieval.get("combined_store_receipt_sha256")
        or combined.compilation_receipt_sha256
        != retrieval.get("compilation_receipt_sha256")
        or combined.source_database_sha256 != source_receipt["database_sha256"]
        or combined.turn_count != source_receipt["turn_count"]
        or combined.chunk_count != source_receipt["chunk_count"]
        or combined.retrieval_policy_sha256
        != preflight.policy.retrieval_policy_sha256
        or combined.retained_request_token_state_bytes != 0
    ):
        raise ValueError("validation shard store receipts no longer cross-bind")
    return source_receipt, combined


def validate_validation_shard_retrieval(
    retrieval: Mapping[str, Any],
    *,
    preflight: ValidationShardPreflight,
) -> None:
    """Validate a shard artifact against freshly reconstructed source inputs."""

    source_receipt, combined = _validate_shard_head(
        retrieval,
        preflight=preflight,
    )
    questions = retrieval.get("questions")
    hashes = retrieval.get("question_part_sha256s")
    if (
        not isinstance(questions, list)
        or not isinstance(hashes, list)
        or len(questions) != LOCKED_QUESTIONS_PER_SHARD
        or len(hashes) != len(questions)
    ):
        raise ValueError("validation shard question population is incomplete")
    observed_hashes = [
        hashlib.sha256(_canonical_json_bytes(question)).hexdigest()
        for question in questions
    ]
    if hashes != observed_hashes:
        raise ValueError("validation shard question-part digests changed")
    seen_question_ids: set[str] = set()
    for local_ordinal, (question, source_question) in enumerate(
        zip(questions, preflight.sample.questions, strict=True)
    ):
        if not isinstance(question, Mapping):
            raise ValueError("validation shard question must be an object")
        _validate_question_part(
            question,
            question=source_question,
            local_ordinal=local_ordinal,
            preflight=preflight,
            source_store_receipt_sha256=source_receipt["receipt_sha256"],
            combined_store_receipt_sha256=combined.receipt_sha256,
            compilation_receipt_sha256=combined.compilation_receipt_sha256,
        )
        question_id = str(question["question_id"])
        if question_id in seen_question_ids:
            raise ValueError("validation shard repeats a question ID")
        seen_question_ids.add(question_id)
    _assert_gold_blind_schema(retrieval, label="validation shard retrieval")
