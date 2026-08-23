"""Run the streamlined matched CAV experiment over the sealed 1M artifact.

The expensive phase loads Qwen once, encodes the globally deduplicated text
population once, and publishes only tensor-free score/order receipts.  Answer
and replay phases consume that receipt without loading Qwen or reopening a
retrieval store.  Gold answers are unavailable until the explicit score phase.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import ssl
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.benchmark import exact_match, f1_score
from memory_condense.eval.fast_cav_feature_session import (
    FastCAVFeatureSessionReceipt,
    FastCAVStageReceipt,
    run_fast_cav_feature_session,
)
from memory_condense.eval.fast_cav_prompts import (
    ARM_IDS,
    TensorFreeStageOrder,
    build_fast_cav_prompt_population,
)
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    ORIGINAL_1M_RETRIEVAL_SHA256,
    STAGE_IDS,
    FastRetrievalArtifact,
    load_fast_retrieval_artifact,
)
from memory_condense.search.fusion.fixed_cav_router import FixedCAVRuntimeReceipt
from memory_condense.search.fusion.steered_readout import MatchedSteeredReadout
FEATURE_MANIFEST_FORMAT = "memory-condense-fast-1m-cav-features-v1"
ANSWER_MANIFEST_FORMAT = "memory-condense-fast-1m-cav-answers-v1"
SCORE_MANIFEST_FORMAT = "memory-condense-fast-1m-cav-scores-v1"
ZERO_STATE_CONTRACT = "tensor-free-fast-1m-cav-phase-boundary-v1"

DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-"
    "development-20260821/retrieval.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-fast-cav-development-20260822"
)
DEFAULT_QWEN_MODEL = Path(".cache/models/Qwen3-8B")
DEFAULT_SPLIT = Path(
    "docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"
)
DEFAULT_GATEWAY_URL = "https://central-dev.zt:4000/v1"
DEFAULT_GATEWAY_MODEL = "codex_sdk/gpt-5.6-terra"
DEFAULT_CALLER_MODEL = "openai/codex_sdk/gpt-5.6-terra"
DEFAULT_EVENT_CAV = Path("eval_results/qwen3_event_membership_cav_probe.safetensors")
DEFAULT_PREFIX_CAV = Path("eval_results/qwen3_prefix_cav_probe.safetensors")
DEFAULT_LAYER = 0

_STAGE_ALIASES = {f"S{index}": stage_id for index, stage_id in enumerate(STAGE_IDS)}
_CAV_SELECTION_KEYS = (
    "autobiographical_completed_event.layer_0",
    "context_dependency.layer_0",
    "binding_constraint.layer_0",
)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: object) -> str:
    """Publish one immutable canonical JSON object and its digest sidecar."""

    payload = _canonical_json_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace another artifact: {path}")
    else:
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
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_payload = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar.exists():
        if sidecar.read_bytes() != sidecar_payload:
            raise FileExistsError(f"refusing to replace another digest: {sidecar}")
    else:
        sidecar.write_bytes(sidecar_payload)
    return digest


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != _canonical_json_bytes(payload):
        raise ValueError(f"artifact is not a canonical JSON object: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if not sidecar.is_file() or sidecar.read_bytes() != expected:
        raise ValueError(f"artifact digest sidecar is missing or invalid: {path}")
    return payload, digest


def _selected_stages(raw: str) -> tuple[str, ...]:
    value = str(raw).strip()
    if value.casefold() == "all":
        return STAGE_IDS
    requested = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not requested:
        raise ValueError("--stages must select S0, S1, S2, S3, or all")
    if len(requested) != len(set(requested)):
        raise ValueError("--stages must not contain duplicates")
    unknown = tuple(item for item in requested if item not in _STAGE_ALIASES)
    if unknown:
        raise ValueError(f"unknown stage alias: {unknown[0]}")
    requested_ids = {_STAGE_ALIASES[item] for item in requested}
    canonical = tuple(item for item in STAGE_IDS if item in requested_ids)
    if tuple(_STAGE_ALIASES[item] for item in requested) != canonical:
        raise ValueError("--stages must preserve cumulative S0-S3 order")
    return canonical


def _load_artifact(path: Path) -> FastRetrievalArtifact:
    return load_fast_retrieval_artifact(
        path,
        expected_sha256=ORIGINAL_1M_RETRIEVAL_SHA256,
    )


def _orders_from_session(
    session: FastCAVFeatureSessionReceipt,
) -> tuple[TensorFreeStageOrder, ...]:
    return tuple(
        TensorFreeStageOrder(
            question_id=row.question_id,
            stage_id=row.stage_id,
            original_evidence_ids=row.readout.original_atom_order,
            base_evidence_ids=row.readout.base_order,
            treatment_evidence_ids=row.readout.treatment_order,
            upstream_receipt_sha256=row.stage_output_sha256,
        )
        for row in session.stage_receipts
    )


def _orders_payload(
    orders: Sequence[TensorFreeStageOrder],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for order in orders:
        row = order.identity_payload()
        row["order_input_sha256"] = order.order_input_sha256
        result.append(row)
    return result


def _for_stages(
    orders: Sequence[TensorFreeStageOrder],
    stage_ids: Sequence[str],
) -> tuple[TensorFreeStageOrder, ...]:
    selected = set(stage_ids)
    return tuple(order for order in orders if order.stage_id in selected)


def _identity_orders(artifact: FastRetrievalArtifact) -> tuple[TensorFreeStageOrder, ...]:
    """Provider-free lower-bound preflight before CAV features exist."""

    return tuple(
        TensorFreeStageOrder(
            question_id=question.question_id,
            stage_id=stage.stage_id,
            original_evidence_ids=stage.evidence_ids,
            base_evidence_ids=stage.evidence_ids,
            treatment_evidence_ids=stage.evidence_ids,
            upstream_receipt_sha256=stage.stage_receipt_sha256,
        )
        for question in artifact.questions
        for stage in question.stages
    )


def _read_feature_orders(
    artifact: FastRetrievalArtifact,
    path: Path,
) -> tuple[tuple[TensorFreeStageOrder, ...], dict[str, Any], str]:
    payload, digest = _read_canonical_json(path)
    if payload.get("format") != FEATURE_MANIFEST_FORMAT:
        raise ValueError("feature manifest has an unsupported format")
    if payload.get("retrieval_sha256") != artifact.raw_sha256:
        raise ValueError("feature manifest belongs to another retrieval artifact")
    zero_state = payload.get("zero_state")
    if not isinstance(zero_state, dict) or zero_state != {
        "contract": ZERO_STATE_CONTRACT,
        "persisted_transformer_token_state": False,
        "retained_transformer_token_state_bytes": 0,
    }:
        raise ValueError("feature manifest changed the zero-state boundary")
    raw_orders = payload.get("stage_orders")
    if not isinstance(raw_orders, list):
        raise ValueError("feature manifest has no stage-order population")

    orders: list[TensorFreeStageOrder] = []
    for index, raw in enumerate(raw_orders):
        if not isinstance(raw, dict):
            raise TypeError(f"feature stage_orders[{index}] must be an object")
        if raw.get("format") != "memory-condense-tensor-free-stage-order-v1":
            raise ValueError("feature stage-order format changed")
        order = TensorFreeStageOrder(
            question_id=raw.get("question_id"),
            stage_id=raw.get("stage_id"),
            original_evidence_ids=tuple(raw.get("original_evidence_ids", ())),
            base_evidence_ids=tuple(raw.get("base_evidence_ids", ())),
            treatment_evidence_ids=tuple(raw.get("treatment_evidence_ids", ())),
            upstream_receipt_sha256=raw.get("upstream_receipt_sha256"),
            retained_tensor_bytes=raw.get("retained_tensor_bytes", -1),
        )
        if raw.get("order_input_sha256") != order.order_input_sha256:
            raise ValueError("feature stage-order receipt does not verify")
        orders.append(order)

    expected_keys = [
        (question.question_id, stage.stage_id)
        for question in artifact.questions
        for stage in question.stages
    ]
    if [(row.question_id, row.stage_id) for row in orders] != expected_keys:
        raise ValueError("feature stage-order population changed")

    raw_session = payload.get("feature_session")
    if not isinstance(raw_session, dict):
        raise ValueError("feature manifest has no session receipt")
    raw_stage_receipts = raw_session.get("stage_receipts")
    if not isinstance(raw_stage_receipts, list):
        raise ValueError("feature session stage receipts changed")
    typed_stage_receipts: list[FastCAVStageReceipt] = []
    for raw_stage in raw_stage_receipts:
        if not isinstance(raw_stage, dict):
            raise TypeError("feature session stage receipt must be an object")
        raw_readout = raw_stage.get("readout")
        if not isinstance(raw_readout, dict):
            raise ValueError("feature session stage has no readout receipt")
        readout_body = dict(raw_readout)
        for key in (
            "original_atom_order",
            "base_scores",
            "treatment_scores",
            "base_order",
            "treatment_order",
        ):
            readout_body[key] = tuple(readout_body.get(key, ()))
        readout = MatchedSteeredReadout(**readout_body)
        stage_body = dict(raw_stage)
        stage_body["readout"] = readout
        for key in (
            "evidence_feature_row_indices",
            "evidence_ids",
            "source_ids",
            "evidence_text_sha256s",
        ):
            stage_body[key] = tuple(stage_body.get(key, ()))
        typed_stage_receipts.append(FastCAVStageReceipt(**stage_body))

    session_body = dict(raw_session)
    session_body["stage_ids"] = tuple(session_body.get("stage_ids", ()))
    session_body["stage_receipts"] = tuple(typed_stage_receipts)
    typed_session = FastCAVFeatureSessionReceipt(**session_body)
    if _canonical_json_bytes(asdict(typed_session)) != _canonical_json_bytes(raw_session):
        raise ValueError("feature session typed receipt projection changed")

    raw_router = payload.get("router_runtime_receipt")
    if not isinstance(raw_router, dict):
        raise ValueError("feature manifest has no router runtime receipt")
    router_body = dict(raw_router)
    router_body["artifact_file_sha256s"] = tuple(
        router_body.get("artifact_file_sha256s", ())
    )
    router_body["ordered_tensor_keys"] = tuple(
        router_body.get("ordered_tensor_keys", ())
    )
    typed_router = FixedCAVRuntimeReceipt(**router_body)
    if _canonical_json_bytes(asdict(typed_router)) != _canonical_json_bytes(raw_router):
        raise ValueError("feature router typed receipt projection changed")
    if (
        typed_session.artifact_sha256 != artifact.raw_sha256
        or typed_session.stage_placement_count != len(expected_keys)
        or typed_session.encoder_api_call_count != 1
        or typed_session.result_retained_tensor_bytes != 0
        or typed_session.retained_token_id_count != 0
        or typed_session.persisted_token_state_bytes != 0
        or typed_session.router_runtime_identity_sha256
        != typed_router.runtime_sha256
        or typed_session.router_bank_identity_sha256
        != typed_router.bank_identity_sha256
        or typed_session.router_num_cavs != typed_router.num_cavs
        or typed_session.feature_layer != typed_router.layer
        or typed_session.feature_hidden_dim != typed_router.hidden_dim
    ):
        raise ValueError("feature session receipt changed")
    if len(typed_stage_receipts) != len(orders):
        raise ValueError("feature session stage receipts changed")
    for order, source in zip(orders, typed_stage_receipts, strict=True):
        readout = source.readout
        if (
            source.question_id != order.question_id
            or source.stage_id != order.stage_id
            or source.stage_output_sha256 != order.upstream_receipt_sha256
            or readout.original_atom_order != order.original_evidence_ids
            or readout.base_order != order.base_evidence_ids
            or readout.treatment_order != order.treatment_evidence_ids
            or readout.result_retained_tensor_bytes != 0
        ):
            raise ValueError("feature stage-order/readout binding changed")
        stage = artifact.question(order.question_id).stage(order.stage_id)
        if order.original_evidence_ids != stage.evidence_ids:
            raise ValueError("feature order changed exact retrieval evidence")
    return tuple(orders), payload, digest


def _prompt_summary(population: Any) -> dict[str, Any]:
    return {
        "selected_stage_ids": list(population.selected_stage_ids),
        "logical_prompt_count": population.logical_prompt_count,
        "unique_prompt_count": population.unique_prompt_count,
        "maximum_prompt_token_proxy": max(
            row.prompt_token_proxy for row in population.unique_prompts
        ),
        "prompt_population_sha256": population.prompt_population_sha256,
        "retained_tensor_bytes": population.retained_tensor_bytes,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    artifact = _load_artifact(Path(args.retrieval))
    stages = _selected_stages(args.stages)
    feature_path = Path(args.features or Path(args.output_root) / "features.json")
    if feature_path.is_file():
        orders, _features, feature_sha = _read_feature_orders(artifact, feature_path)
        kind = "actual_feature_orders"
    else:
        orders = _identity_orders(artifact)
        feature_sha = None
        kind = "identity_order_lower_bound"
    population = build_fast_cav_prompt_population(
        artifact,
        _for_stages(orders, stages),
        stage_ids=stages,
    )
    return {
        "format": "memory-condense-fast-1m-cav-preflight-v1",
        "retrieval_sha256": artifact.raw_sha256,
        "transcript_tokens": artifact.transcript_tokens,
        "turn_count": artifact.turn_count,
        "question_count": artifact.question_count,
        "logical_evidence_placements": artifact.logical_feature_row_count,
        "deduplicated_feature_rows": artifact.unique_feature_row_count,
        "feature_manifest_sha256": feature_sha,
        "order_population_kind": kind,
        "prompt_preflight": _prompt_summary(population),
        "maximum_possible_unique_prompts": population.logical_prompt_count,
        "elapsed_s": time.perf_counter() - started,
        "writes": 0,
        "provider_calls": 0,
    }


def run_features(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
    from memory_condense.search.fusion.fixed_cav_router import FixedCAVRouter

    total_started = time.perf_counter()
    artifact_started = time.perf_counter()
    artifact = _load_artifact(Path(args.retrieval))
    artifact_elapsed = time.perf_counter() - artifact_started
    stages = _selected_stages(args.stages)

    router_started = time.perf_counter()
    selections = (
        (Path(args.event_cav), _CAV_SELECTION_KEYS[0]),
        (Path(args.prefix_cav), _CAV_SELECTION_KEYS[1]),
        (Path(args.prefix_cav), _CAV_SELECTION_KEYS[2]),
    )
    router = FixedCAVRouter.load(
        selections,
        layer=DEFAULT_LAYER,
        device="cpu",
        dtype="float32",
        extraction_temperature=args.extraction_temperature,
        reinjection_temperature=args.reinjection_temperature,
        alpha=args.alpha,
    )
    router_elapsed = time.perf_counter() - router_started

    encoder_started = time.perf_counter()
    encoder = Qwen3PrefixEncoder(
        Path(args.model_dir),
        layers=1,
        device=args.device,
        dtype=args.dtype,
    )
    encoder_elapsed = time.perf_counter() - encoder_started
    session_started = time.perf_counter()
    try:
        session = run_fast_cav_feature_session(
            artifact,
            encoder=encoder,
            router=router,
            layer=DEFAULT_LAYER,
            batch_size=args.batch_size,
        )
    finally:
        encoder = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - optional runtime
            pass
    session_elapsed = time.perf_counter() - session_started
    orders = _orders_from_session(session)
    population = build_fast_cav_prompt_population(
        artifact,
        _for_stages(orders, stages),
        stage_ids=stages,
    )
    manifest = {
        "format": FEATURE_MANIFEST_FORMAT,
        "retrieval_sha256": artifact.raw_sha256,
        "transcript_tokens": artifact.transcript_tokens,
        "turn_count": artifact.turn_count,
        "question_count": artifact.question_count,
        "feature_session": asdict(session),
        "router_runtime_receipt": asdict(router.runtime_receipt),
        "stage_orders": _orders_payload(orders),
        "default_prompt_preflight": _prompt_summary(population),
        "timing_s": {
            "artifact_load": artifact_elapsed,
            "router_load": router_elapsed,
            "encoder_load": encoder_elapsed,
            "feature_session": session_elapsed,
            "total_before_publish": time.perf_counter() - total_started,
        },
        "zero_state": {
            "contract": ZERO_STATE_CONTRACT,
            "persisted_transformer_token_state": False,
            "retained_transformer_token_state_bytes": 0,
        },
    }
    output = Path(args.features or Path(args.output_root) / "features.json")
    digest = _atomic_write_json(output, manifest)
    return manifest, digest


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


def _answer_artifact(
    *,
    mode: str,
    artifact: FastRetrievalArtifact,
    feature_sha256: str,
    feature_payload: Mapping[str, Any],
    prompt_population: Any,
    completion_batch: Any,
) -> dict[str, Any]:
    answers: list[dict[str, Any]] = []
    for prompt, prediction in zip(
        prompt_population.logical_prompts,
        completion_batch.logical_completions,
        strict=True,
    ):
        answers.append(
            {
                "logical_ordinal": prompt.logical_ordinal,
                "question_ordinal": prompt.question_ordinal,
                "question_id": prompt.question_id,
                "stage_id": prompt.stage_id,
                "arm_id": prompt.arm_id,
                "arm_prompt_sha256": prompt.arm_prompt_sha256,
                "messages_sha256": prompt.messages_sha256,
                "unique_prompt_ordinal": prompt.unique_prompt_ordinal,
                "prompt_token_proxy": prompt.prompt_token_proxy,
                "prediction": prediction,
                "prediction_sha256": quote_sha256(prediction),
            }
        )
    return {
        "format": ANSWER_MANIFEST_FORMAT,
        "mode": mode,
        "retrieval_sha256": artifact.raw_sha256,
        "feature_manifest_sha256": feature_sha256,
        "feature_session_receipt_sha256": feature_payload["feature_session"][
            "session_receipt_sha256"
        ],
        "prompt_population": prompt_population.identity_payload(),
        "completion_batch": completion_batch.model_dump(),
        "question_count": artifact.question_count,
        "logical_answer_count": len(answers),
        "unique_completion_count": len(completion_batch.unique_records),
        "answers": answers,
        "gold_fields_present": False,
        "zero_state": {
            "contract": ZERO_STATE_CONTRACT,
            "persisted_transformer_token_state": False,
            "retained_transformer_token_state_bytes": 0,
            "external_provider_persistence_certified": False,
        },
    }


def run_answers(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    mode = str(args.phase)
    if mode not in {"answer", "replay"}:
        raise ValueError("answer runtime phase must be answer or replay")
    artifact = _load_artifact(Path(args.retrieval))
    feature_path = Path(args.features or Path(args.output_root) / "features.json")
    orders, feature_payload, feature_sha = _read_feature_orders(artifact, feature_path)
    stages = _selected_stages(args.stages)
    prompts = build_fast_cav_prompt_population(
        artifact,
        _for_stages(orders, stages),
        stage_ids=stages,
    )
    unique_calls = prompts.unique_prompt_count

    client = None
    if mode == "answer":
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

    provenance = {
        "format": "memory-condense-fast-1m-cav-answer-binding-v1",
        "retrieval_sha256": artifact.raw_sha256,
        "feature_manifest_sha256": feature_sha,
        "feature_session_receipt_sha256": feature_payload["feature_session"][
            "session_receipt_sha256"
        ],
        "prompt_population_sha256": prompts.prompt_population_sha256,
        "selected_stage_ids": list(stages),
        "authorized_unique_calls": unique_calls,
        "caller_model_alias": str(args.caller_model),
        "gateway_url": str(args.gateway_url),
        "gold_blind": True,
    }
    runtime = FastCompletionRuntime(
        checkpoint_dir=Path(args.output_root) / "completion-calls",
        prompt_population=prompts.logical_message_population,
        model=str(args.gateway_model),
        client=client,
        max_prompt_tokens=8_000,
        max_new_tokens=args.max_new_tokens,
        max_concurrency=args.max_concurrency,
        retries=0,
        benchmark_provenance=provenance,
    )
    with runtime:
        batch = runtime.run()
    result = _answer_artifact(
        mode=mode,
        artifact=artifact,
        feature_sha256=feature_sha,
        feature_payload=feature_payload,
        prompt_population=prompts,
        completion_batch=batch,
    )
    filename = "answers.json" if mode == "answer" else "replay.json"
    digest = _atomic_write_json(Path(args.output_root) / filename, result)
    return result, digest


def _read_and_validate_answers(
    artifact: FastRetrievalArtifact,
    path: Path,
    orders: Sequence[TensorFreeStageOrder],
    feature_sha256: str,
) -> tuple[dict[str, Any], str]:
    payload, digest = _read_canonical_json(path)
    if payload.get("format") != ANSWER_MANIFEST_FORMAT:
        raise ValueError("answer manifest has an unsupported format")
    if (
        payload.get("retrieval_sha256") != artifact.raw_sha256
        or payload.get("feature_manifest_sha256") != feature_sha256
        or payload.get("gold_fields_present") is not False
    ):
        raise ValueError("answer manifest provenance changed")
    prompt_receipt = payload.get("prompt_population")
    if not isinstance(prompt_receipt, dict):
        raise ValueError("answer manifest has no prompt population")
    selected = tuple(prompt_receipt.get("selected_stage_ids", ()))
    prompts = build_fast_cav_prompt_population(
        artifact,
        _for_stages(orders, selected),
        stage_ids=selected,
    )
    if prompt_receipt != prompts.identity_payload():
        raise ValueError("answer prompt population does not verify")
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, list) or len(raw_answers) != len(
        prompts.logical_prompts
    ):
        raise ValueError("answer population count changed")
    predictions: list[str] = []
    for source, expected in zip(raw_answers, prompts.logical_prompts, strict=True):
        if not isinstance(source, dict):
            raise TypeError("answer row must be an object")
        prediction = source.get("prediction")
        if not isinstance(prediction, str) or not prediction:
            raise ValueError("answer prediction must be non-empty")
        expected_projection = {
            "logical_ordinal": expected.logical_ordinal,
            "question_ordinal": expected.question_ordinal,
            "question_id": expected.question_id,
            "stage_id": expected.stage_id,
            "arm_id": expected.arm_id,
            "arm_prompt_sha256": expected.arm_prompt_sha256,
            "messages_sha256": expected.messages_sha256,
            "unique_prompt_ordinal": expected.unique_prompt_ordinal,
            "prompt_token_proxy": expected.prompt_token_proxy,
        }
        if any(source.get(key) != value for key, value in expected_projection.items()):
            raise ValueError("answer row changed prompt provenance")
        if source.get("prediction_sha256") != quote_sha256(prediction):
            raise ValueError("answer prediction SHA-256 does not verify")
        predictions.append(prediction)

    completion_batch = payload.get("completion_batch")
    if not isinstance(completion_batch, dict):
        raise ValueError("answer manifest has no completion batch")
    if completion_batch.get("logical_completions") != predictions:
        raise ValueError("answer rows disagree with verified logical completions")
    runtime_prompt_population = completion_batch.get("prompt_population")
    if not isinstance(runtime_prompt_population, dict):
        raise ValueError("answer completion batch has no runtime prompt population")
    ordered_runtime_rows = runtime_prompt_population.get("ordered_rows")
    if not isinstance(ordered_runtime_rows, list) or len(ordered_runtime_rows) != len(
        prompts.logical_prompts
    ):
        raise ValueError("answer runtime prompt population changed")
    for runtime_row, prompt in zip(
        ordered_runtime_rows,
        prompts.logical_prompts,
        strict=True,
    ):
        if not isinstance(runtime_row, dict) or (
            runtime_row.get("ordinal") != prompt.logical_ordinal
            or runtime_row.get("messages_sha256") != prompt.messages_sha256
            or runtime_row.get("prompt_token_proxy") != prompt.prompt_token_proxy
        ):
            raise ValueError("answer runtime prompt row changed")
    provenance = completion_batch.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("answer completion provenance is missing")
    benchmark = provenance.get("benchmark_provenance")
    if not isinstance(benchmark, dict) or (
        benchmark.get("retrieval_sha256") != artifact.raw_sha256
        or benchmark.get("feature_manifest_sha256") != feature_sha256
        or benchmark.get("prompt_population_sha256")
        != prompts.prompt_population_sha256
        or tuple(benchmark.get("selected_stage_ids", ())) != selected
    ):
        raise ValueError("answer completion benchmark provenance changed")
    unique_records = completion_batch.get("unique_records")
    if not isinstance(unique_records, list):
        raise ValueError("answer completion records are missing")
    completions_by_messages: dict[str, str] = {}
    for record in unique_records:
        if not isinstance(record, dict):
            raise TypeError("answer completion record must be an object")
        messages_sha = record.get("messages_sha256")
        completion = record.get("completion")
        if (
            not isinstance(messages_sha, str)
            or not isinstance(completion, str)
            or not completion
            or record.get("completion_sha256") != quote_sha256(completion)
            or record.get("finish_reason") != "stop"
            or messages_sha in completions_by_messages
        ):
            raise ValueError("answer completion record does not verify")
        completions_by_messages[messages_sha] = completion
    if len(completions_by_messages) != prompts.unique_prompt_count:
        raise ValueError("answer unique completion population changed")
    for prompt, prediction in zip(
        prompts.logical_prompts,
        predictions,
        strict=True,
    ):
        if completions_by_messages.get(prompt.messages_sha256) != prediction:
            raise ValueError("answer prediction is not its journaled completion")
    return payload, digest


def _replay_answer_journals(
    *,
    artifact: FastRetrievalArtifact,
    answers: Mapping[str, Any],
    orders: Sequence[TensorFreeStageOrder],
    checkpoint_dir: Path,
) -> None:
    """Reopen every immutable provider journal before gold is reachable."""

    prompt_receipt = answers["prompt_population"]
    selected = tuple(prompt_receipt["selected_stage_ids"])
    prompts = build_fast_cav_prompt_population(
        artifact,
        _for_stages(orders, selected),
        stage_ids=selected,
    )
    batch = answers["completion_batch"]
    provenance = batch["provenance"]
    runtime = FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=prompts.logical_message_population,
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
    expected_predictions = tuple(row["prediction"] for row in answers["answers"])
    if (
        replay.logical_completions != expected_predictions
        or replay.runtime_identity_sha256 != batch.get("runtime_identity_sha256")
        or tuple(row.response_journal_sha256 for row in replay.unique_records)
        != tuple(
            row.get("response_journal_sha256")
            for row in batch.get("unique_records", ())
        )
    ):
        raise ValueError("answer manifest disagrees with immutable provider journals")


def run_score(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.dataset is None:
        raise ValueError("score phase requires --dataset")
    artifact = _load_artifact(Path(args.retrieval))
    feature_path = Path(args.features or Path(args.output_root) / "features.json")
    orders, _features, feature_sha = _read_feature_orders(artifact, feature_path)
    answer_path = Path(args.answers or Path(args.output_root) / "answers.json")
    answers, answer_sha = _read_and_validate_answers(
        artifact,
        answer_path,
        orders,
        feature_sha,
    )

    _replay_answer_journals(
        artifact=artifact,
        answers=answers,
        orders=orders,
        checkpoint_dir=answer_path.parent / "completion-calls",
    )

    # Gold becomes reachable only after every answer/provenance check above.
    from memory_condense.eval.recall_guarded_cumulative_1m import (
        load_original_population,
    )

    sample = load_original_population(Path(args.dataset), Path(args.split))
    gold_by_id = {question.question_id: question for question in sample.questions}
    if tuple(gold_by_id) != tuple(question.question_id for question in artifact.questions):
        raise RuntimeError("post-hoc gold population changed question order")

    scored_rows: list[dict[str, Any]] = []
    for row in answers["answers"]:
        gold = gold_by_id[row["question_id"]]
        prediction = row["prediction"]
        scored_rows.append(
            {
                "logical_ordinal": row["logical_ordinal"],
                "question_ordinal": row["question_ordinal"],
                "question_id": row["question_id"],
                "stage_id": row["stage_id"],
                "arm_id": row["arm_id"],
                "category": gold.category,
                "prediction_sha256": row["prediction_sha256"],
                "gold_answer_sha256": quote_sha256(gold.answer),
                "exact_match": exact_match(prediction, gold.answer),
                "f1": f1_score(prediction, gold.answer),
            }
        )

    aggregates: list[dict[str, Any]] = []
    selected_stages = tuple(answers["prompt_population"]["selected_stage_ids"])
    for stage_id in selected_stages:
        for arm_id in ARM_IDS:
            rows = [
                row
                for row in scored_rows
                if row["stage_id"] == stage_id and row["arm_id"] == arm_id
            ]
            aggregates.append(
                {
                    "stage_id": stage_id,
                    "arm_id": arm_id,
                    "questions": len(rows),
                    "exact_matches": sum(bool(row["exact_match"]) for row in rows),
                    "exact_match_rate": statistics.fmean(
                        float(row["exact_match"]) for row in rows
                    ),
                    "mean_f1": statistics.fmean(float(row["f1"]) for row in rows),
                }
            )
    result = {
        "format": SCORE_MANIFEST_FORMAT,
        "retrieval_sha256": artifact.raw_sha256,
        "feature_manifest_sha256": feature_sha,
        "answer_manifest_sha256": answer_sha,
        "gold_loaded_posthoc": True,
        "question_count": artifact.question_count,
        "logical_score_count": len(scored_rows),
        "aggregates": aggregates,
        "rows": scored_rows,
    }
    digest = _atomic_write_json(Path(args.output_root) / "scores.json", result)
    return result, digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("preflight", "features", "answer", "replay", "score"),
        default="preflight",
    )
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--stages", default="S1")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--event-cav", type=Path, default=DEFAULT_EVENT_CAV)
    parser.add_argument("--prefix-cav", type=Path, default=DEFAULT_PREFIX_CAV)
    parser.add_argument("--extraction-temperature", type=float, default=0.05)
    parser.add_argument("--reinjection-temperature", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--gateway-model", default=DEFAULT_GATEWAY_MODEL)
    parser.add_argument("--caller-model", default=DEFAULT_CALLER_MODEL)
    parser.add_argument("--api-key-env", default="LITELLM_KEY")
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.phase == "preflight":
        result = run_preflight(args)
        prompt = result["prompt_preflight"]
        print(
            "Fast 1M CAV preflight passed: "
            f"questions={result['question_count']}; "
            f"logical_prompts={prompt['logical_prompt_count']}; "
            f"unique_prompts={prompt['unique_prompt_count']}; "
            f"order_kind={result['order_population_kind']}; provider_calls=0",
            flush=True,
        )
        return 0
    if args.phase == "features":
        result, digest = run_features(args)
        session = result["feature_session"]
        prompt = result["default_prompt_preflight"]
        print(
            "Fast 1M CAV features published: "
            f"{Path(args.features or Path(args.output_root) / 'features.json').resolve()} "
            f"({digest}); encoder_calls={session['encoder_api_call_count']}; "
            f"unique_texts={session['global_unique_text_count']}; "
            f"router_calls={session['unique_router_call_count']}; "
            f"unique_prompts={prompt['unique_prompt_count']}",
            flush=True,
        )
        return 0
    if args.phase in {"answer", "replay"}:
        result, digest = run_answers(args)
        usage = result["completion_batch"]["usage"]
        print(
            f"Fast 1M CAV {args.phase} published ({digest}): "
            f"logical={result['logical_answer_count']}; "
            f"unique={result['unique_completion_count']}; "
            f"physical={usage['physical_calls']}; "
            f"checkpoint_hits={usage['checkpoint_hits']}",
            flush=True,
        )
        return 0
    result, digest = run_score(args)
    print(f"Fast 1M CAV scores published ({digest})", flush=True)
    for row in result["aggregates"]:
        print(
            f"  {row['stage_id']} {row['arm_id']}: "
            f"EM={row['exact_matches']}/{row['questions']} "
            f"mean_F1={row['mean_f1']:.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
