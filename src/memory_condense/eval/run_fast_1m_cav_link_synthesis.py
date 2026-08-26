"""Run the sealed 1M matched CAV-link synthesis experiment.

The runner has four phases. ``preflight`` is provider-free. ``answer`` creates
the provider only after all matched prompts have been built and preflighted.
``replay`` reopens immutable completion journals without a provider. ``score``
loads gold only after the retrieval, linked feature artifact, prompt
population, answer, replay, strict response parses, and journals all verify.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import ssl
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.benchmark import exact_match, f1_score
from memory_condense.eval._artifact_json import (
    canonical_json_bytes as _canonical_json_bytes,
)
from memory_condense.eval.fast_cav_feature_artifact import (
    FAST_CAV_FEATURE_ARTIFACT_FORMAT,
    FastCAVFeatureArtifact,
    load_fast_cav_feature_artifact,
)
from memory_condense.eval.fast_cav_link_synthesis import (
    FAST_CAV_LINK_GUIDE_PROJECTION_POLICY_SHA256,
    FAST_CAV_LINK_SYNTHESIS_ARM_IDS,
    FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS,
    FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS,
    FAST_CAV_LINK_SYNTHESIS_POLICY_SHA256,
    FAST_CAV_LINK_SYNTHESIS_STAGE_ID,
    FastCAVLinkSynthesisPopulation,
    build_fast_cav_link_synthesis_population,
    parse_fast_cav_link_synthesis,
)
from memory_condense.eval.fast_completion_runtime import (
    FAST_COMPLETION_RUNTIME_FORMAT,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    ORIGINAL_1M_RETRIEVAL_SHA256,
    FastRetrievalArtifact,
    load_fast_retrieval_artifact,
)


PREFLIGHT_FORMAT = "memory-condense-fast-1m-cav-link-synthesis-preflight-v1"
ANSWER_MANIFEST_FORMAT = "memory-condense-fast-1m-cav-link-synthesis-answers-v1"
SCORE_MANIFEST_FORMAT = "memory-condense-fast-1m-cav-link-synthesis-scores-v1"
ANSWER_BINDING_FORMAT = "memory-condense-fast-1m-cav-link-synthesis-binding-v1"
ZERO_STATE_CONTRACT = "tensor-free-fast-1m-cav-link-synthesis-boundary-v1"

DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-"
    "development-20260821/retrieval.json"
)
DEFAULT_FEATURES = Path(
    "eval_results/longmemeval-1m-fast-cav-links-development-20260823/"
    "features.json"
)
DEFAULT_FEATURES_SHA256 = (
    "f7b6552cdfdcb96ef34063d6fbe887b057c137df3515080896bc2a2877cded2f"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-fast-cav-link-synthesis-development-20260823"
)
DEFAULT_SPLIT = Path(
    "docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"
)
DEFAULT_GATEWAY_URL = "https://central-dev.zt:4000/v1"
DEFAULT_GATEWAY_MODEL = "codex_sdk/gpt-5.6-terra"
DEFAULT_CALLER_MODEL = "openai/codex_sdk/gpt-5.6-terra"
DEFAULT_EXPECTED_QUESTION_COUNT = 10

_DIGEST_CHARS = frozenset("0123456789abcdef")
_ANSWER_FIELDS = frozenset(
    {
        "format",
        "mode",
        "experiment_binding",
        "prompt_population",
        "completion_batch",
        "question_count",
        "logical_answer_count",
        "unique_completion_count",
        "answers",
        "gold_fields_present",
        "zero_state",
    }
)
_ANSWER_ROW_FIELDS = frozenset(
    {
        "logical_ordinal",
        "question_ordinal",
        "question_id",
        "stage_id",
        "arm_id",
        "link_exposed",
        "arm_prompt_sha256",
        "stage_receipt_sha256",
        "messages_sha256",
        "evidence_coordinates_sha256",
        "source_link_receipt_sha256",
        "link_guide_projection_sha256",
        "prompt_token_proxy",
        "hard_prompt_token_cap",
        "max_completion_tokens",
        "completion",
        "completion_sha256",
        "parsed_response",
    }
)
_BATCH_FIELDS = frozenset(
    {
        "logical_completions",
        "unique_records",
        "usage",
        "provenance",
        "runtime_identity_sha256",
        "prompt_population",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "format",
        "model",
        "max_new_tokens",
        "max_prompt_token_proxy",
        "max_concurrency",
        "retries",
        "request_options",
        "prompt_population_sha256",
        "prompt_token_proxy_identity",
        "benchmark_provenance",
        "persisted_transformer_token_state",
        "retained_transformer_token_state_bytes",
        "external_provider_persistence_certified",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "call_key_sha256",
        "request_journal_sha256",
        "response_journal_sha256",
        "messages_sha256",
        "completion",
        "completion_sha256",
        "requested_model",
        "response_id",
        "response_model",
        "finish_reason",
        "prompt_token_proxy",
        "completion_token_proxy",
        "reported_prompt_tokens",
        "reported_completion_tokens",
        "reported_total_tokens",
        "provider_elapsed_s",
        "checkpoint_hit",
        "physical_call",
    }
)
_USAGE_FIELDS = frozenset(
    {
        "logical_calls",
        "unique_calls",
        "deduplicated_logical_calls",
        "physical_calls",
        "checkpoint_hits",
        "prompt_token_proxy",
        "completion_token_proxy",
        "recorded_reported_prompt_tokens",
        "recorded_reported_completion_tokens",
        "recorded_reported_total_tokens",
        "reported_prompt_tokens_complete",
        "reported_completion_tokens_complete",
        "reported_total_tokens_complete",
        "recorded_provider_elapsed_s",
    }
)
_RUNTIME_POPULATION_FIELDS = frozenset(
    {
        "format",
        "logical_prompt_count",
        "unique_prompt_count",
        "ordered_rows",
        "prompt_population_sha256",
        "max_prompt_token_proxy",
        "prompt_token_proxy_identity",
    }
)
_RECORD_DISPOSITION_FIELDS = frozenset({"checkpoint_hit", "physical_call"})
_USAGE_DISPOSITION_FIELDS = frozenset({"physical_calls", "checkpoint_hits"})
_FORBIDDEN_SECRET_FIELDS = frozenset(
    {"api_key", "api-key", "authorization", "litellm_key"}
)


@dataclass(frozen=True, slots=True)
class _Experiment:
    retrieval: FastRetrievalArtifact
    feature: FastCAVFeatureArtifact
    prompts: FastCAVLinkSynthesisPopulation


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and set(value).issubset(_DIGEST_CHARS)
    )


def _publish_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FileExistsError(f"refusing to replace a symbolic link: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace another artifact: {path}")
        return
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: object) -> str:
    payload = _canonical_json_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    _publish_bytes(path, payload)
    _publish_bytes(
        path.with_name(path.name + ".sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )
    return digest


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular file: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"artifact is not valid JSON: {path}") from exc
    if type(payload) is not dict or raw != _canonical_json_bytes(payload):
        raise ValueError(f"artifact is not a canonical JSON object: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar.is_symlink() or not sidecar.is_file() or sidecar.read_bytes() != expected:
        raise ValueError(f"artifact digest sidecar is missing or invalid: {path}")
    return payload, digest


def _answers_path(args: argparse.Namespace) -> Path:
    return Path(args.answers or Path(args.output_root) / "answers.json")


def _replay_path(args: argparse.Namespace) -> Path:
    return Path(args.replay or Path(args.output_root) / "replay.json")


def _checkpoint_path(args: argparse.Namespace) -> Path:
    return _answers_path(args).parent / "completion-calls"


def _validate_common_args(args: argparse.Namespace) -> None:
    if not _is_digest(args.expected_retrieval_sha256):
        raise ValueError("--expected-retrieval-sha256 must be an exact digest")
    if not _is_digest(args.expected_features_sha256):
        raise ValueError("--expected-features-sha256 must be an exact digest")
    if type(args.expected_question_count) is not int or args.expected_question_count < 1:
        raise ValueError("--expected-question-count must be a positive integer")
    if type(args.max_concurrency) is not int or args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be a positive integer")
    for name in ("gateway_url", "gateway_model", "caller_model", "api_key_env"):
        value = getattr(args, name)
        if type(value) is not str or not value or value.strip() != value:
            raise ValueError(f"--{name.replace('_', '-')} must be an exact string")


def _load_experiment(args: argparse.Namespace) -> _Experiment:
    _validate_common_args(args)
    retrieval = load_fast_retrieval_artifact(
        Path(args.retrieval),
        expected_sha256=str(args.expected_retrieval_sha256),
    )
    feature = load_fast_cav_feature_artifact(
        Path(args.features),
        retrieval_artifact=retrieval,
        expected_sha256=str(args.expected_features_sha256),
        require_links=True,
    )
    prompts = build_fast_cav_link_synthesis_population(
        retrieval,
        feature.feature_session,
    )
    # This independent runtime preflight is deliberately before any client.
    runtime_preflight = preflight_fast_completion_prompts(
        prompts.logical_message_population,
        max_prompt_tokens=FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS,
    )
    expected_questions = args.expected_question_count
    expected_prompts = expected_questions * len(FAST_CAV_LINK_SYNTHESIS_ARM_IDS)
    if (
        retrieval.question_count != expected_questions
        or feature.question_count != expected_questions
        or prompts.question_count != expected_questions
        or prompts.logical_prompt_count != expected_prompts
        or len(runtime_preflight.ordered_rows) != expected_prompts
        or runtime_preflight.model_dump() != prompts.completion_preflight.model_dump()
    ):
        raise ValueError("experiment did not preflight the exact matched population")
    return _Experiment(retrieval=retrieval, feature=feature, prompts=prompts)


def _stage_bindings(prompts: FastCAVLinkSynthesisPopulation) -> list[dict[str, Any]]:
    return [
        {
            "question_ordinal": row.question_ordinal,
            "question_id": row.question_id,
            "stage_id": row.stage_id,
            "synthesis_stage_receipt_sha256": row.receipt_sha256,
            "source_stage_receipt_sha256": row.source_stage_receipt_sha256,
            "feature_stage_output_sha256": row.feature_stage_output_sha256,
            "evidence_coordinates_sha256": row.evidence_coordinates_sha256,
            "packet_identity_sha256": row.packet_identity_sha256,
            "source_link_receipt_sha256": row.source_link_receipt_sha256,
            "extraction_matrix_sha256": row.extraction_matrix_sha256,
            "reinjection_matrix_sha256": row.reinjection_matrix_sha256,
            "extraction_links_sha256": row.extraction_links_sha256,
            "reinjection_links_sha256": row.reinjection_links_sha256,
            "link_guide_projection_sha256": row.link_guide_projection_sha256,
            "arm_prompt_sha256s": list(row.arm_prompt_sha256s),
        }
        for row in prompts.stage_receipts
    ]


def _experiment_binding(experiment: _Experiment) -> dict[str, Any]:
    retrieval = experiment.retrieval
    feature = experiment.feature
    prompts = experiment.prompts
    router = feature.router_runtime_receipt
    stages = _stage_bindings(prompts)
    return {
        "format": ANSWER_BINDING_FORMAT,
        "retrieval_sha256": retrieval.raw_sha256,
        "population_identity_sha256": retrieval.population_identity_sha256,
        "retrieval_implementation_sha256": retrieval.retrieval_implementation_sha256,
        "retrieval_policy_sha256": retrieval.retrieval_policy_sha256,
        "feature_manifest_sha256": feature.raw_sha256,
        "feature_manifest_format": feature.format,
        "feature_session_receipt_sha256": (
            feature.feature_session.session_receipt_sha256
        ),
        "feature_checkpoint_sha256": feature.feature_session.feature_checkpoint_sha256,
        "router_runtime_identity_sha256": router.runtime_sha256,
        "router_bank_identity_sha256": router.bank_identity_sha256,
        "router_concept_coordinates_sha256": identity_sha256(
            {
                "artifact_file_sha256s": list(router.artifact_file_sha256s),
                "ordered_tensor_keys": list(router.ordered_tensor_keys),
            }
        ),
        "synthesis_population_sha256": prompts.population_sha256,
        "runtime_prompt_population_sha256": (
            prompts.completion_preflight.prompt_population_sha256
        ),
        "prompt_policy_sha256": prompts.prompt_policy_sha256,
        "link_guide_projection_policy_sha256": (
            FAST_CAV_LINK_GUIDE_PROJECTION_POLICY_SHA256
        ),
        "stage_id": FAST_CAV_LINK_SYNTHESIS_STAGE_ID,
        "arm_ids": list(FAST_CAV_LINK_SYNTHESIS_ARM_IDS),
        "question_count": prompts.question_count,
        "logical_prompt_count": prompts.logical_prompt_count,
        "unique_prompt_count": prompts.unique_prompt_count,
        "max_prompt_tokens": FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS,
        "max_completion_tokens": FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS,
        "stage_bindings": stages,
        "stage_binding_population_sha256": identity_sha256(stages),
        "retained_request_token_state_bytes": 0,
    }


def _benchmark_provenance(
    binding: Mapping[str, Any],
    *,
    gateway_model: str,
    caller_model: str,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    return {
        "format": ANSWER_BINDING_FORMAT,
        "experiment_binding_sha256": identity_sha256(dict(binding)),
        "retrieval_sha256": binding["retrieval_sha256"],
        "feature_manifest_sha256": binding["feature_manifest_sha256"],
        "feature_session_receipt_sha256": binding[
            "feature_session_receipt_sha256"
        ],
        "synthesis_population_sha256": binding["synthesis_population_sha256"],
        "runtime_prompt_population_sha256": binding[
            "runtime_prompt_population_sha256"
        ],
        "stage_binding_population_sha256": binding[
            "stage_binding_population_sha256"
        ],
        "router_runtime_identity_sha256": binding[
            "router_runtime_identity_sha256"
        ],
        "router_bank_identity_sha256": binding["router_bank_identity_sha256"],
        "prompt_policy_sha256": binding["prompt_policy_sha256"],
        "link_guide_projection_policy_sha256": binding[
            "link_guide_projection_policy_sha256"
        ],
        "gateway_model": gateway_model,
        "caller_model_alias": caller_model,
        "gateway_url": gateway_url,
        "authorized_unique_calls": binding["unique_prompt_count"],
        "logical_prompt_count": binding["logical_prompt_count"],
        "max_prompt_tokens": FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS,
        "max_completion_tokens": FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS,
        "max_concurrency": max_concurrency,
        "retries": 0,
        "gold_blind": True,
        "retained_request_token_state_bytes": 0,
    }


def _preflight_payload(experiment: _Experiment) -> dict[str, Any]:
    binding = _experiment_binding(experiment)
    preflight = experiment.prompts.completion_preflight
    return {
        "format": PREFLIGHT_FORMAT,
        "experiment_binding": binding,
        "experiment_binding_sha256": identity_sha256(binding),
        "feature_links_required": True,
        "logical_prompt_count": preflight.logical_prompt_count,
        "unique_prompt_count": preflight.unique_prompt_count,
        "maximum_prompt_token_proxy": max(
            row.prompt_token_proxy for row in preflight.ordered_rows
        ),
        "max_prompt_tokens": FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS,
        "max_completion_tokens": FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS,
        "writes": 0,
        "provider_calls": 0,
        "gold_loaded": False,
        "retained_request_token_state_bytes": 0,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    return _preflight_payload(_load_experiment(args))


def _make_provider_client(api_key: str, gateway_url: str) -> Any:
    import httpx
    import truststore
    from openai import OpenAI

    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return OpenAI(
        api_key=api_key,
        base_url=gateway_url,
        http_client=httpx.Client(verify=context),
        max_retries=0,
    )


def _expected_answer_projection(experiment: _Experiment, prompt: Any) -> dict[str, Any]:
    stage = experiment.prompts.stage_receipts[prompt.question_ordinal]
    return {
        "logical_ordinal": prompt.logical_ordinal,
        "question_ordinal": prompt.question_ordinal,
        "question_id": prompt.question_id,
        "stage_id": prompt.stage_id,
        "arm_id": prompt.arm_id,
        "link_exposed": prompt.link_exposed,
        "arm_prompt_sha256": prompt.arm_prompt_sha256,
        "stage_receipt_sha256": stage.receipt_sha256,
        "messages_sha256": prompt.messages_sha256,
        "evidence_coordinates_sha256": prompt.evidence_coordinates_sha256,
        "source_link_receipt_sha256": prompt.source_link_receipt_sha256,
        "link_guide_projection_sha256": prompt.link_guide_projection_sha256,
        "prompt_token_proxy": prompt.prompt_token_proxy,
        "hard_prompt_token_cap": prompt.hard_prompt_token_cap,
        "max_completion_tokens": prompt.max_completion_tokens,
    }


def _parse_completion(experiment: _Experiment, prompt: Any, completion: str) -> Any:
    question = experiment.retrieval.questions[prompt.question_ordinal]
    stage_receipt = experiment.prompts.stage_receipts[prompt.question_ordinal]
    return parse_fast_cav_link_synthesis(
        completion,
        stage=question.stage(FAST_CAV_LINK_SYNTHESIS_STAGE_ID),
        receipt=stage_receipt,
    )


def _answer_rows(
    experiment: _Experiment,
    completions: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt, completion in zip(
        experiment.prompts.prompts,
        completions,
        strict=True,
    ):
        parsed = _parse_completion(experiment, prompt, completion)
        rows.append(
            {
                **_expected_answer_projection(experiment, prompt),
                "completion": completion,
                "completion_sha256": quote_sha256(completion),
                "parsed_response": parsed.identity_payload(),
            }
        )
    return rows


def _answer_artifact(
    *,
    mode: str,
    experiment: _Experiment,
    completion_batch: Any,
) -> dict[str, Any]:
    if mode not in {"answer", "replay"}:
        raise ValueError("answer artifact mode must be answer or replay")
    rows = _answer_rows(experiment, completion_batch.logical_completions)
    return {
        "format": ANSWER_MANIFEST_FORMAT,
        "mode": mode,
        "experiment_binding": _experiment_binding(experiment),
        "prompt_population": experiment.prompts.identity_payload(),
        "completion_batch": completion_batch.model_dump(),
        "question_count": experiment.retrieval.question_count,
        "logical_answer_count": len(rows),
        "unique_completion_count": len(completion_batch.unique_records),
        "answers": rows,
        "gold_fields_present": False,
        "zero_state": {
            "contract": ZERO_STATE_CONTRACT,
            "persisted_transformer_token_state": False,
            "retained_transformer_token_state_bytes": 0,
            "external_provider_persistence_certified": False,
        },
    }


def _contains_secret(value: object) -> bool:
    if type(value) is dict:
        return any(
            str(key).casefold() in _FORBIDDEN_SECRET_FIELDS
            or _contains_secret(item)
            for key, item in value.items()
        )
    if type(value) is list:
        return any(_contains_secret(item) for item in value)
    return False


def _stable_batch_projection(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "logical_completions": batch["logical_completions"],
        "unique_records": [
            {
                name: value
                for name, value in record.items()
                if name not in _RECORD_DISPOSITION_FIELDS
            }
            for record in batch["unique_records"]
        ],
        "usage": {
            name: value
            for name, value in batch["usage"].items()
            if name not in _USAGE_DISPOSITION_FIELDS
        },
        "provenance": batch["provenance"],
        "runtime_identity_sha256": batch["runtime_identity_sha256"],
        "prompt_population": batch["prompt_population"],
    }


def _validate_completion_batch(
    experiment: _Experiment,
    batch: object,
    predictions: Sequence[str],
) -> None:
    if type(batch) is not dict or set(batch) != _BATCH_FIELDS:
        raise ValueError("completion batch has a noncanonical shape")
    if batch["logical_completions"] != list(predictions):
        raise ValueError("answer rows disagree with the completion batch")
    provenance = batch["provenance"]
    if type(provenance) is not dict or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("completion provenance has a noncanonical shape")
    concurrency = provenance["max_concurrency"]
    benchmark = provenance["benchmark_provenance"]
    binding = _experiment_binding(experiment)
    if (
        provenance["format"] != FAST_COMPLETION_RUNTIME_FORMAT
        or type(provenance["model"]) is not str
        or not provenance["model"]
        or provenance["max_new_tokens"]
        != FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS
        or provenance["max_prompt_token_proxy"]
        != FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS
        or type(concurrency) is not int
        or concurrency < 1
        or provenance["retries"] != 0
        or provenance["request_options"] != {}
        or type(provenance["prompt_token_proxy_identity"]) is not dict
        or provenance["persisted_transformer_token_state"] is not False
        or provenance["retained_transformer_token_state_bytes"] != 0
        or provenance["external_provider_persistence_certified"] is not False
        or type(benchmark) is not dict
    ):
        raise ValueError("completion runtime configuration changed")
    caller = benchmark.get("caller_model_alias")
    gateway = benchmark.get("gateway_url")
    if (
        type(caller) is not str
        or not caller
        or type(gateway) is not str
        or not gateway
        or benchmark
        != _benchmark_provenance(
            binding,
            gateway_model=provenance["model"],
            caller_model=caller,
            gateway_url=gateway,
            max_concurrency=concurrency,
        )
    ):
        raise ValueError("completion benchmark provenance changed")
    runtime_population = experiment.prompts.completion_preflight.model_dump()
    if (
        type(batch["prompt_population"]) is not dict
        or set(batch["prompt_population"]) != _RUNTIME_POPULATION_FIELDS
        or batch["prompt_population"] != runtime_population
        or provenance["prompt_population_sha256"]
        != runtime_population["prompt_population_sha256"]
    ):
        raise ValueError("completion runtime prompt population changed")

    records = batch["unique_records"]
    if type(records) is not list or len(records) != experiment.prompts.unique_prompt_count:
        raise ValueError("unique completion population changed")
    by_messages: dict[str, str] = {}
    for record in records:
        if type(record) is not dict or set(record) != _RECORD_FIELDS:
            raise ValueError("completion record has a noncanonical shape")
        messages_sha = record["messages_sha256"]
        completion = record["completion"]
        if (
            not _is_digest(messages_sha)
            or type(completion) is not str
            or not completion
            or record["completion_sha256"] != quote_sha256(completion)
            or record["finish_reason"] != "stop"
            or record["requested_model"] != provenance["model"]
            or type(record["checkpoint_hit"]) is not bool
            or type(record["physical_call"]) is not bool
            or record["checkpoint_hit"] == record["physical_call"]
            or messages_sha in by_messages
        ):
            raise ValueError("completion record does not verify")
        for name in (
            "call_key_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
        ):
            if not _is_digest(record[name]):
                raise ValueError("completion journal seal is invalid")
        elapsed = record["provider_elapsed_s"]
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise ValueError("completion elapsed time is invalid")
        by_messages[messages_sha] = completion
    expected_unique = tuple(
        dict.fromkeys(row.messages_sha256 for row in experiment.prompts.prompts)
    )
    if tuple(row["messages_sha256"] for row in records) != expected_unique:
        raise ValueError("unique completion order changed")
    for prompt, prediction in zip(
        experiment.prompts.prompts,
        predictions,
        strict=True,
    ):
        if by_messages.get(prompt.messages_sha256) != prediction:
            raise ValueError("prediction is not its journaled completion")

    usage = batch["usage"]
    if (
        type(usage) is not dict
        or set(usage) != _USAGE_FIELDS
        or usage["logical_calls"] != experiment.prompts.logical_prompt_count
        or usage["unique_calls"] != experiment.prompts.unique_prompt_count
        or usage["deduplicated_logical_calls"]
        != experiment.prompts.logical_prompt_count
        - experiment.prompts.unique_prompt_count
        or type(usage["physical_calls"]) is not int
        or type(usage["checkpoint_hits"]) is not int
        or usage["physical_calls"] + usage["checkpoint_hits"] != len(records)
        or usage["physical_calls"]
        != sum(bool(row["physical_call"]) for row in records)
        or usage["checkpoint_hits"]
        != sum(bool(row["checkpoint_hit"]) for row in records)
        or not _is_digest(batch["runtime_identity_sha256"])
        or batch["runtime_identity_sha256"] != identity_sha256(provenance)
    ):
        raise ValueError("completion usage or runtime identity changed")


def _read_and_validate_answers(
    experiment: _Experiment,
    path: Path,
    *,
    expected_mode: str,
) -> tuple[dict[str, Any], str]:
    payload, digest = _read_canonical_json(path)
    if set(payload) != _ANSWER_FIELDS:
        raise ValueError("answer manifest has a noncanonical shape")
    if (
        payload["format"] != ANSWER_MANIFEST_FORMAT
        or payload["mode"] != expected_mode
        or payload["gold_fields_present"] is not False
        or payload["experiment_binding"] != _experiment_binding(experiment)
        or payload["prompt_population"] != experiment.prompts.identity_payload()
    ):
        raise ValueError("answer manifest changed format or upstream provenance")
    if payload["zero_state"] != {
        "contract": ZERO_STATE_CONTRACT,
        "persisted_transformer_token_state": False,
        "retained_transformer_token_state_bytes": 0,
        "external_provider_persistence_certified": False,
    }:
        raise ValueError("answer manifest changed the zero-state boundary")
    if (
        type(payload["question_count"]) is not int
        or payload["question_count"] != experiment.retrieval.question_count
        or type(payload["logical_answer_count"]) is not int
        or payload["logical_answer_count"] != experiment.prompts.logical_prompt_count
        or type(payload["unique_completion_count"]) is not int
        or payload["unique_completion_count"] != experiment.prompts.unique_prompt_count
    ):
        raise ValueError("answer manifest changed population cardinality")
    raw_rows = payload["answers"]
    if type(raw_rows) is not list or len(raw_rows) != experiment.prompts.logical_prompt_count:
        raise ValueError("answer row population changed")
    predictions: list[str] = []
    for raw, prompt in zip(raw_rows, experiment.prompts.prompts, strict=True):
        if type(raw) is not dict or set(raw) != _ANSWER_ROW_FIELDS:
            raise ValueError("answer row has a noncanonical shape")
        expected = _expected_answer_projection(experiment, prompt)
        if any(raw[name] != value for name, value in expected.items()):
            raise ValueError("answer row changed prompt, link, or coordinate provenance")
        completion = raw["completion"]
        if (
            type(completion) is not str
            or not completion
            or raw["completion_sha256"] != quote_sha256(completion)
        ):
            raise ValueError("answer completion does not verify")
        parsed = _parse_completion(experiment, prompt, completion)
        if raw["parsed_response"] != parsed.identity_payload():
            raise ValueError("answer strict parsed response changed")
        predictions.append(completion)
    _validate_completion_batch(experiment, payload["completion_batch"], predictions)
    if _contains_secret(payload):
        raise ValueError("answer artifact serialized provider credentials")
    return payload, digest


def _replay_journals(experiment: _Experiment, answers: Mapping[str, Any], path: Path) -> Any:
    batch = answers["completion_batch"]
    provenance = batch["provenance"]
    runtime = FastCompletionRuntime(
        checkpoint_dir=path,
        prompt_population=experiment.prompts.logical_message_population,
        model=provenance["model"],
        client=None,
        max_prompt_tokens=provenance["max_prompt_token_proxy"],
        max_new_tokens=provenance["max_new_tokens"],
        max_concurrency=provenance["max_concurrency"],
        retries=provenance["retries"],
        request_options=provenance["request_options"],
        benchmark_provenance=provenance["benchmark_provenance"],
    )
    with runtime:
        replay = runtime.run()
    if (
        replay.logical_completions
        != tuple(row["completion"] for row in answers["answers"])
        or replay.runtime_identity_sha256 != batch["runtime_identity_sha256"]
        or tuple(row.response_journal_sha256 for row in replay.unique_records)
        != tuple(row["response_journal_sha256"] for row in batch["unique_records"])
        or _stable_batch_projection(batch)
        != _stable_batch_projection(replay.model_dump())
    ):
        raise ValueError("answer manifest disagrees with immutable provider journals")
    return replay


def _validate_answer_replay_pair(
    answers: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> None:
    if (
        answers["experiment_binding"] != replay["experiment_binding"]
        or answers["prompt_population"] != replay["prompt_population"]
        or answers["answers"] != replay["answers"]
        or _stable_batch_projection(answers["completion_batch"])
        != _stable_batch_projection(replay["completion_batch"])
    ):
        raise ValueError("answer and replay artifacts bind different results")
    usage = replay["completion_batch"]["usage"]
    if usage["physical_calls"] != 0 or usage["checkpoint_hits"] != replay[
        "unique_completion_count"
    ]:
        raise ValueError("replay artifact was not a provider-free journal replay")


def run_answer(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    experiment = _load_experiment(args)
    unique_calls = experiment.prompts.unique_prompt_count
    if not args.enable_provider:
        raise ValueError("answer phase requires the explicit --enable-provider gate")
    if args.authorized_provider_calls != unique_calls:
        raise ValueError(
            "--authorized-provider-calls must exactly equal provider-free "
            f"unique prompt count ({args.authorized_provider_calls} != {unique_calls})"
        )
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    if not api_key:
        raise RuntimeError(f"provider API key is empty: {args.api_key_env}")
    client = _make_provider_client(api_key, str(args.gateway_url))
    binding = _experiment_binding(experiment)
    provenance = _benchmark_provenance(
        binding,
        gateway_model=str(args.gateway_model),
        caller_model=str(args.caller_model),
        gateway_url=str(args.gateway_url),
        max_concurrency=args.max_concurrency,
    )
    runtime = FastCompletionRuntime(
        checkpoint_dir=_checkpoint_path(args),
        prompt_population=experiment.prompts.logical_message_population,
        model=str(args.gateway_model),
        client=client,
        max_prompt_tokens=FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS,
        max_new_tokens=FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS,
        max_concurrency=args.max_concurrency,
        retries=0,
        benchmark_provenance=provenance,
    )
    with runtime:
        batch = runtime.run()
    result = _answer_artifact(
        mode="answer",
        experiment=experiment,
        completion_batch=batch,
    )
    return result, _atomic_write_json(_answers_path(args), result)


def run_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("replay phase forbids provider access and authorization")
    experiment = _load_experiment(args)
    answers, _answer_sha = _read_and_validate_answers(
        experiment,
        _answers_path(args),
        expected_mode="answer",
    )
    batch = _replay_journals(experiment, answers, _checkpoint_path(args))
    result = _answer_artifact(
        mode="replay",
        experiment=experiment,
        completion_batch=batch,
    )
    return result, _atomic_write_json(_replay_path(args), result)


def _load_gold_population(dataset: Path, split: Path) -> Any:
    # Deliberately lazy: no earlier phase can import or access benchmark gold.
    from memory_condense.eval.recall_guarded_cumulative_1m import (
        load_original_population,
    )

    return load_original_population(dataset, split)


def _score_artifact(
    *,
    experiment: _Experiment,
    answers: Mapping[str, Any],
    answer_sha256: str,
    replay_sha256: str,
    gold_population: Any,
) -> dict[str, Any]:
    gold_by_id = {row.question_id: row for row in gold_population.questions}
    expected_ids = tuple(row.question_id for row in experiment.retrieval.questions)
    if len(gold_by_id) != len(gold_population.questions) or tuple(gold_by_id) != expected_ids:
        raise RuntimeError("post-hoc gold population changed question order")
    rows: list[dict[str, Any]] = []
    for source in answers["answers"]:
        gold = gold_by_id[source["question_id"]]
        prediction = source["parsed_response"]["answer"]
        rows.append(
            {
                "logical_ordinal": source["logical_ordinal"],
                "question_ordinal": source["question_ordinal"],
                "question_id": source["question_id"],
                "stage_id": source["stage_id"],
                "arm_id": source["arm_id"],
                "category": gold.category,
                "response_sha256": source["parsed_response"]["response_sha256"],
                "prediction_sha256": quote_sha256(prediction),
                "gold_answer_sha256": quote_sha256(gold.answer),
                "citation_count": len(source["parsed_response"]["citations"]),
                "exact_match": exact_match(prediction, gold.answer),
                "f1": f1_score(prediction, gold.answer),
            }
        )
    aggregates: list[dict[str, Any]] = []
    for arm_id in FAST_CAV_LINK_SYNTHESIS_ARM_IDS:
        selected = [row for row in rows if row["arm_id"] == arm_id]
        if len(selected) != experiment.retrieval.question_count:
            raise RuntimeError(f"score population omitted matched arm {arm_id}")
        aggregates.append(
            {
                "stage_id": FAST_CAV_LINK_SYNTHESIS_STAGE_ID,
                "arm_id": arm_id,
                "questions": len(selected),
                "exact_matches": sum(bool(row["exact_match"]) for row in selected),
                "exact_match_rate": statistics.fmean(
                    float(row["exact_match"]) for row in selected
                ),
                "mean_f1": statistics.fmean(float(row["f1"]) for row in selected),
            }
        )
    return {
        "format": SCORE_MANIFEST_FORMAT,
        "experiment_binding": answers["experiment_binding"],
        "answer_manifest_sha256": answer_sha256,
        "replay_manifest_sha256": replay_sha256,
        "gold_loaded_posthoc": True,
        "question_count": experiment.retrieval.question_count,
        "logical_score_count": len(rows),
        "aggregates": aggregates,
        "rows": rows,
        "retained_request_token_state_bytes": 0,
    }


def run_score(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("score phase forbids provider access and authorization")
    if args.dataset is None:
        raise ValueError("score phase requires --dataset")
    experiment = _load_experiment(args)
    answers, answer_sha = _read_and_validate_answers(
        experiment,
        _answers_path(args),
        expected_mode="answer",
    )
    replay, replay_sha = _read_and_validate_answers(
        experiment,
        _replay_path(args),
        expected_mode="replay",
    )
    _validate_answer_replay_pair(answers, replay)
    journal_replay = _replay_journals(
        experiment,
        answers,
        _checkpoint_path(args),
    )
    if replay["completion_batch"] != journal_replay.model_dump():
        raise ValueError("replay artifact differs from immutable provider journals")

    # Gold is first reachable here, after every upstream and journal check.
    gold = _load_gold_population(Path(args.dataset), Path(args.split))
    result = _score_artifact(
        experiment=experiment,
        answers=answers,
        answer_sha256=answer_sha,
        replay_sha256=replay_sha,
        gold_population=gold,
    )
    return result, _atomic_write_json(Path(args.output_root) / "scores.json", result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("preflight", "answer", "replay", "score"),
        default="preflight",
    )
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--expected-retrieval-sha256",
        default=ORIGINAL_1M_RETRIEVAL_SHA256,
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--expected-features-sha256",
        default=DEFAULT_FEATURES_SHA256,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--expected-question-count",
        type=int,
        default=DEFAULT_EXPECTED_QUESTION_COUNT,
    )
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--gateway-model", default=DEFAULT_GATEWAY_MODEL)
    parser.add_argument("--caller-model", default=DEFAULT_CALLER_MODEL)
    parser.add_argument("--api-key-env", default="LITELLM_KEY")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.phase == "preflight":
        result: Any = run_preflight(args)
        print(
            "CAV-link synthesis preflight passed: "
            f"questions={result['experiment_binding']['question_count']}; "
            f"logical={result['logical_prompt_count']}; "
            f"unique={result['unique_prompt_count']}; "
            f"max_prompt={result['maximum_prompt_token_proxy']}/"
            f"{result['max_prompt_tokens']}; provider_calls=0; writes=0",
            flush=True,
        )
        return 0
    elif args.phase == "answer":
        result, digest = run_answer(args)
    elif args.phase == "replay":
        result, digest = run_replay(args)
    else:
        result, digest = run_score(args)
        print(f"CAV-link synthesis scores published ({digest})", flush=True)
        for row in result["aggregates"]:
            print(
                f"  {row['arm_id']}: EM={row['exact_matches']}/"
                f"{row['questions']} mean_F1={row['mean_f1']:.6f}",
                flush=True,
            )
        return 0
    usage = result["completion_batch"]["usage"]
    print(
        f"CAV-link synthesis {args.phase} published ({digest}): "
        f"logical={result['logical_answer_count']}; "
        f"unique={result['unique_completion_count']}; "
        f"physical={usage['physical_calls']}; "
        f"checkpoint_hits={usage['checkpoint_hits']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANSWER_BINDING_FORMAT",
    "ANSWER_MANIFEST_FORMAT",
    "DEFAULT_FEATURES",
    "DEFAULT_FEATURES_SHA256",
    "DEFAULT_OUTPUT_ROOT",
    "PREFLIGHT_FORMAT",
    "SCORE_MANIFEST_FORMAT",
    "ZERO_STATE_CONTRACT",
    "build_parser",
    "main",
    "run_answer",
    "run_preflight",
    "run_replay",
    "run_score",
]
