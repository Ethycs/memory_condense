#!/usr/bin/env python3
"""Run the sealed reduced-30 Terra-answer and Sol-judge lifecycle.

The answer boundary consumes only a verified, gold-free construction artifact.
Benchmark references are first opened by ``judge-preflight``, after predictions
already exist as an immutable sealed artifact.  Provider execution is a separate
explicit step whose authorization must equal the number of authenticated
checkpoint misses exactly.  Both provider phases use zero retries.

This runner intentionally depends on the reduced-30 harness contract rather
than a historical v6/v7 selection format.  The harness locates the selected
provider arm dynamically and verifies the exact zero-based question lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval._binary_judge_protocol import (  # noqa: E402
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools import assay_hot_reduced30_construction as harness  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
)
from tools.matched_eval.provider_runtime import (  # noqa: E402
    DEFAULT_API_KEY_ENV,
    DEFAULT_GATEWAY_URL,
    make_provider_client,
)


ANSWER_PREFLIGHT_FORMAT = "memory-condense-hot-reduced30-answer-preflight-v1"
ANSWERS_FORMAT = "memory-condense-hot-reduced30-terra-predictions-v1"
JUDGE_PREFLIGHT_FORMAT = "memory-condense-hot-reduced30-judge-preflight-v1"
JUDGMENTS_FORMAT = "memory-condense-hot-reduced30-sol-judgments-v1"

ANSWER_PREFLIGHT_NAME = "answer-preflight.json"
ANSWERS_NAME = "answers.json"
JUDGE_PREFLIGHT_NAME = "judge-preflight.json"
JUDGMENTS_NAME = "judgments.json"
ANSWER_CHECKPOINT_DIR = "answer-checkpoints"
JUDGE_CHECKPOINT_DIR = "judge-checkpoints"

TERRA_MODEL = "codex_sdk/gpt-5.6-terra"
SOL_MODEL = "codex_sdk/gpt-5.6-sol"
ANSWER_MAX_PROMPT_TOKENS = 12_000
ANSWER_MAX_TOKENS = 256
JUDGE_MAX_PROMPT_TOKENS = 4_096
DEFAULT_MAX_CONCURRENCY = 10
LOCKED_FULL100_POPULATION_SHA256 = (
    "9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246"
)


class Reduced30LifecycleError(ValueError):
    """A seal, identity, gold boundary, or call authorization changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise Reduced30LifecycleError(message)


def _mapping(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact list")
    return value  # type: ignore[return-value]


def _positive_int(value: object, label: str) -> int:
    _require(type(value) is int and value > 0, f"{label} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, label: str) -> int:
    _require(
        type(value) is int and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return int(value)


def _text(value: object, label: str) -> str:
    _require(
        type(value) is str and bool(value) and value.strip() == value,
        f"{label} must be non-empty exact text",
    )
    return str(value)


def _digest(value: object, label: str) -> str:
    _require(type(value) is str, f"{label} must be a SHA-256 digest")
    return require_sha256(value, label)


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(body)
    payload["receipt_sha256"] = identity_sha256(dict(body))
    return payload


def _validate_receipt(payload: Mapping[str, Any], label: str) -> None:
    body = dict(payload)
    receipt = body.pop("receipt_sha256", None)
    _require(receipt == identity_sha256(body), f"{label} receipt changed")


def _read_bound(path: Path, expected_sha256: str, label: str) -> SealedArtifact:
    expected = _digest(expected_sha256, f"expected {label} SHA-256")
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == expected,
        f"{label} digest changed ({artifact.sha256} != {expected})",
    )
    return artifact


def _load_selection(path: Path, expected_sha256: str) -> SealedArtifact:
    artifact = _read_bound(path, expected_sha256, "selection")
    harness.validate_selection(artifact.payload)
    assert_gold_blind(artifact.payload, path="reduced30.selection")
    return artifact


def _plain_messages(value: object, label: str) -> list[dict[str, str]]:
    rows = _list(value, label)
    output: list[dict[str, str]] = []
    for index, candidate in enumerate(rows):
        message = _mapping(candidate, f"{label}[{index}]")
        _require(
            set(message) == {"role", "content"}
            and type(message.get("role")) is str
            and type(message.get("content")) is str,
            f"{label}[{index}] must contain exact role/content strings",
        )
        output.append({"role": message["role"], "content": message["content"]})
    _require(bool(output), f"{label} must not be empty")
    return output


def _selection_prompts(
    selection: Mapping[str, Any],
) -> tuple[list[list[dict[str, str]]], list[dict[str, Any]]]:
    rows = _list(selection.get("questions"), "selection questions")
    _require(len(rows) == harness.QUESTION_COUNT, "selection question count changed")
    prompts: list[list[dict[str, str]]] = []
    bindings: list[dict[str, Any]] = []
    for reduced_ordinal, (candidate, locked) in enumerate(
        zip(rows, harness.LOCKED_QUESTIONS, strict=True)
    ):
        row = _mapping(candidate, f"selection question {reduced_ordinal}")
        ordinal, question_id, question_sha = locked
        telemetry = _mapping(row.get("telemetry"), "selection telemetry")
        declared_path = _text(telemetry.get("arm_path"), "selected arm path")
        source_row = _mapping(row.get("source_row"), "selection source row")
        arm_path, arm = harness.find_provider_arm(source_row, declared_path)
        _require(arm_path == declared_path, "selected arm path changed")
        messages = _plain_messages(
            arm.get("provider_messages"),
            f"provider messages at ordinal {ordinal}",
        )
        payload = canonical_json_bytes({"messages": messages})
        payload_sha = hashlib.sha256(payload).hexdigest()
        declared_sha = arm.get("provider_payload_sha256")
        declared_bytes = arm.get("provider_payload_utf8_bytes")
        if declared_sha is not None:
            _require(
                declared_sha == payload_sha,
                f"declared provider payload hash changed at ordinal {ordinal}",
            )
        if declared_bytes is not None:
            _require(
                declared_bytes == len(payload),
                f"declared provider payload byte count changed at ordinal {ordinal}",
            )
        _require(
            row.get("reduced_ordinal") == reduced_ordinal
            and row.get("global_ordinal") == ordinal
            and row.get("question_id") == question_id
            and row.get("prompt_question_sha256") == question_sha,
            f"locked selection binding changed at ordinal {ordinal}",
        )
        prompts.append(messages)
        bindings.append(
            {
                "arm_path": arm_path,
                "global_ordinal": ordinal,
                "provider_payload_sha256": payload_sha,
                "provider_payload_utf8_bytes": len(payload),
                "prompt_question_sha256": question_sha,
                "question_id": question_id,
                "reduced_ordinal": reduced_ordinal,
                "selection_row_receipt_sha256": _digest(
                    row.get("row_receipt_sha256"), "selection row receipt"
                ),
            }
        )
    return prompts, bindings


def build_answer_preflight(
    *,
    selection: Mapping[str, Any],
    selection_sha256: str,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    gateway = _text(gateway_url, "gateway URL")
    concurrency = _positive_int(max_concurrency, "max concurrency")
    prompts, bindings = _selection_prompts(selection)
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=ANSWER_MAX_PROMPT_TOKENS,
    )
    _require(
        population.logical_prompt_count == harness.QUESTION_COUNT
        and population.unique_prompt_count == harness.QUESTION_COUNT,
        "answer prompts must be an exact unique reduced-30 population",
    )
    question_rows: list[dict[str, Any]] = []
    for binding, prompt_row in zip(
        bindings, population.ordered_rows, strict=True
    ):
        question_rows.append(
            {
                **binding,
                "messages_sha256": prompt_row.messages_sha256,
                "prompt_token_proxy": prompt_row.prompt_token_proxy,
            }
        )
    body = {
        "format": ANSWER_PREFLIGHT_FORMAT,
        "status": "sealed_provider_free_answer_preflight",
        "selection_sha256": _digest(selection_sha256, "selection SHA-256"),
        "locked_question_identity_sha256": harness.LOCKED_IDENTITY_SHA256,
        "question_count": harness.QUESTION_COUNT,
        "model": TERRA_MODEL,
        "gateway_url": gateway,
        "max_prompt_token_proxy": ANSWER_MAX_PROMPT_TOKENS,
        "max_completion_tokens": ANSWER_MAX_TOKENS,
        "max_concurrency": concurrency,
        "retries": 0,
        "request_options": {},
        "prompt_population": population.model_dump(),
        "required_authorized_provider_calls": harness.QUESTION_COUNT,
        "provider_calls": 0,
        "questions": question_rows,
        "gold_fields_present": False,
        "retained_request_token_state_bytes": 0,
    }
    payload = _sealed(body)
    assert_gold_blind(payload, path="reduced30.answer_preflight")
    return payload


def validate_answer_preflight(
    payload: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    selection_sha256: str,
) -> None:
    _validate_receipt(payload, "answer preflight")
    expected = build_answer_preflight(
        selection=selection,
        selection_sha256=selection_sha256,
        gateway_url=_text(payload.get("gateway_url"), "answer gateway URL"),
        max_concurrency=_positive_int(
            payload.get("max_concurrency"), "answer max concurrency"
        ),
    )
    _require(dict(payload) == expected, "answer preflight contract changed")


def answer_preflight(
    *,
    selection_path: Path,
    expected_selection_sha256: str,
    output_root: Path,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    selection = _load_selection(selection_path, expected_selection_sha256)
    payload = build_answer_preflight(
        selection=selection.payload,
        selection_sha256=selection.sha256,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
    )
    artifact, created = publish_sealed_json(
        output_root / ANSWER_PREFLIGHT_NAME, payload
    )
    return {
        "answer_preflight_sha256": artifact.sha256,
        "created": created,
        "new_provider_calls": 0,
        "required_authorized_provider_calls": harness.QUESTION_COUNT,
        "selection_sha256": selection.sha256,
    }


def _answer_runtime(
    *,
    output_root: Path,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    preflight: Mapping[str, Any],
    answer_preflight_sha256: str,
    client: Any | None,
) -> FastCompletionRuntime:
    provenance = {
        "phase": "reduced30_terra_answer",
        "selection_sha256": preflight["selection_sha256"],
        "answer_preflight_sha256": answer_preflight_sha256,
        "locked_question_identity_sha256": harness.LOCKED_IDENTITY_SHA256,
        "gateway_url": preflight["gateway_url"],
        "gold_fields_present": False,
    }
    assert_gold_blind(provenance, path="reduced30.answer_runtime")
    return FastCompletionRuntime(
        checkpoint_dir=output_root / ANSWER_CHECKPOINT_DIR,
        prompt_population=prompts,
        model=TERRA_MODEL,
        client=client,
        max_prompt_tokens=ANSWER_MAX_PROMPT_TOKENS,
        max_new_tokens=ANSWER_MAX_TOKENS,
        max_concurrency=int(preflight["max_concurrency"]),
        retries=0,
        request_options={},
        benchmark_provenance=provenance,
    )


@contextmanager
def _phase_lock(output_root: Path, phase: str) -> Iterator[None]:
    output_root.mkdir(parents=True, exist_ok=True)
    _require(
        not output_root.is_symlink() and output_root.is_dir(),
        "output root must be a regular directory",
    )
    path = output_root / f".{phase}-lifecycle.lock"
    with path.open("a+b") as handle:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - Windows is the primary project runtime.
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _authenticated_records(runtime: FastCompletionRuntime) -> dict[str, Any]:
    # The runtime has already validated all journals in its constructor.  Read
    # them once more under its own cross-process guard to compute exact misses.
    with runtime._journal_guard():  # noqa: SLF001
        return runtime._load_all_records()  # noqa: SLF001


def _completion_client(api_key_env: str, gateway_url: str) -> Any:
    from dotenv import load_dotenv

    load_dotenv(override=False)
    key = os.environ.get(api_key_env, "").strip()
    if not key:
        raise RuntimeError(f"provider API key is empty: {api_key_env}")
    return make_provider_client(key, gateway_url)


def _run_exactly_authorized(
    *,
    runtime_factory: Callable[[Any | None], FastCompletionRuntime],
    authorized_provider_calls: int,
    enable_provider: bool,
    client_factory: Callable[[], Any],
) -> tuple[FastCompletionBatch, int, int, float]:
    authorized = _nonnegative_int(
        authorized_provider_calls, "authorized provider calls"
    )
    audit = runtime_factory(None)
    try:
        authenticated = len(_authenticated_records(audit))
        unique = audit.population.unique_prompt_count
    finally:
        audit.close()
    remaining = unique - authenticated
    _require(remaining >= 0, "authenticated checkpoint count exceeds population")
    _require(
        authorized == remaining,
        "authorized provider calls must equal authenticated checkpoint misses "
        f"exactly ({authorized} != {remaining})",
    )
    _require(
        remaining == 0 or enable_provider is True,
        "--enable-provider is required when authorized calls are nonzero",
    )

    client: Any | None = None
    runtime: FastCompletionRuntime | None = None
    try:
        if remaining:
            client = client_factory()
        runtime = runtime_factory(client)
        started = time.perf_counter()
        batch = runtime.run()
        batch_wall_time_s = time.perf_counter() - started
    finally:
        if runtime is not None:
            runtime.close()
        elif client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    _require(
        batch.usage.physical_calls == remaining
        and batch.usage.checkpoint_hits == authenticated,
        "provider call accounting changed during the authorized run",
    )
    return batch, remaining, authenticated, batch_wall_time_s


def _record_map(batch: FastCompletionBatch) -> dict[str, Any]:
    records = {record.messages_sha256: record for record in batch.unique_records}
    _require(
        len(records) == batch.usage.unique_calls,
        "completion record population changed",
    )
    return records


def _stable_usage(batch: FastCompletionBatch) -> dict[str, Any]:
    usage = batch.usage.model_dump()
    usage.pop("physical_calls")
    usage.pop("checkpoint_hits")
    return usage


def _prediction_payload(
    *,
    selection_sha256: str,
    answer_preflight_sha256: str,
    preflight: Mapping[str, Any],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    records = _record_map(batch)
    preflight_rows = _list(preflight.get("questions"), "answer preflight questions")
    rows: list[dict[str, Any]] = []
    for binding, prompt_row, prediction in zip(
        preflight_rows,
        batch.prompt_population.ordered_rows,
        batch.logical_completions,
        strict=True,
    ):
        record = records[prompt_row.messages_sha256]
        row_body = {
            "reduced_ordinal": binding["reduced_ordinal"],
            "global_ordinal": binding["global_ordinal"],
            "question_id": binding["question_id"],
            "prompt_question_sha256": binding["prompt_question_sha256"],
            "arm_path": binding["arm_path"],
            "provider_payload_sha256": binding["provider_payload_sha256"],
            "messages_sha256": prompt_row.messages_sha256,
            "prediction": prediction,
            "prediction_sha256": quote_sha256(prediction),
            "call_key_sha256": record.call_key_sha256,
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "response_id": record.response_id,
            "response_model": record.response_model,
            "finish_reason": record.finish_reason,
            "provider_elapsed_s": record.provider_elapsed_s,
        }
        rows.append({**row_body, "row_receipt_sha256": identity_sha256(row_body)})
    body = {
        "format": ANSWERS_FORMAT,
        "status": "sealed_terra_predictions_without_gold",
        "selection_sha256": selection_sha256,
        "answer_preflight_sha256": answer_preflight_sha256,
        "locked_question_identity_sha256": harness.LOCKED_IDENTITY_SHA256,
        "prompt_population_sha256": batch.prompt_population.prompt_population_sha256,
        "runtime_identity_sha256": batch.runtime_identity_sha256,
        "model": TERRA_MODEL,
        "gateway_url": preflight["gateway_url"],
        "max_prompt_token_proxy": ANSWER_MAX_PROMPT_TOKENS,
        "max_completion_tokens": ANSWER_MAX_TOKENS,
        "max_concurrency": preflight["max_concurrency"],
        "retries": 0,
        "question_count": harness.QUESTION_COUNT,
        "provider_calls_completed": batch.usage.unique_calls,
        "completion_usage": _stable_usage(batch),
        "questions": rows,
        "gold_fields_present": False,
        "retained_request_token_state_bytes": 0,
    }
    payload = _sealed(body)
    assert_gold_blind(payload, path="reduced30.predictions")
    return payload


def validate_predictions(
    payload: Mapping[str, Any],
    *,
    selection_sha256: str,
    answer_preflight: Mapping[str, Any],
    answer_preflight_sha256: str,
) -> None:
    _validate_receipt(payload, "predictions")
    assert_gold_blind(payload, path="reduced30.predictions")
    _require(
        payload.get("format") == ANSWERS_FORMAT
        and payload.get("status") == "sealed_terra_predictions_without_gold"
        and payload.get("selection_sha256") == selection_sha256
        and payload.get("answer_preflight_sha256") == answer_preflight_sha256
        and payload.get("locked_question_identity_sha256")
        == harness.LOCKED_IDENTITY_SHA256
        and payload.get("model") == TERRA_MODEL
        and payload.get("gateway_url") == answer_preflight.get("gateway_url")
        and payload.get("max_prompt_token_proxy") == ANSWER_MAX_PROMPT_TOKENS
        and payload.get("max_completion_tokens") == ANSWER_MAX_TOKENS
        and payload.get("max_concurrency")
        == answer_preflight.get("max_concurrency")
        and payload.get("retries") == 0
        and payload.get("question_count") == harness.QUESTION_COUNT
        and payload.get("provider_calls_completed") == harness.QUESTION_COUNT
        and payload.get("gold_fields_present") is False
        and payload.get("retained_request_token_state_bytes") == 0,
        "prediction header changed",
    )
    _digest(payload.get("prompt_population_sha256"), "prediction prompt population")
    _digest(payload.get("runtime_identity_sha256"), "prediction runtime identity")
    usage = _mapping(payload.get("completion_usage"), "prediction completion usage")
    _require(
        "physical_calls" not in usage
        and "checkpoint_hits" not in usage
        and usage.get("logical_calls") == harness.QUESTION_COUNT
        and usage.get("unique_calls") == harness.QUESTION_COUNT,
        "prediction stable usage changed",
    )
    expected_rows = _list(
        answer_preflight.get("questions"), "answer preflight questions"
    )
    rows = _list(payload.get("questions"), "prediction questions")
    _require(len(rows) == len(expected_rows), "prediction question count changed")
    call_keys: set[str] = set()
    for index, (candidate, binding) in enumerate(
        zip(rows, expected_rows, strict=True)
    ):
        row = _mapping(candidate, f"prediction row {index}")
        body = dict(row)
        receipt = body.pop("row_receipt_sha256", None)
        _require(receipt == identity_sha256(body), f"prediction row {index} changed")
        for name in (
            "reduced_ordinal",
            "global_ordinal",
            "question_id",
            "prompt_question_sha256",
            "arm_path",
            "provider_payload_sha256",
            "messages_sha256",
        ):
            _require(row.get(name) == binding.get(name), f"prediction {name} changed")
        prediction = _text(row.get("prediction"), f"prediction {index}")
        _require(
            row.get("prediction_sha256") == quote_sha256(prediction),
            f"prediction text hash changed at row {index}",
        )
        call_key = _digest(row.get("call_key_sha256"), "prediction call key")
        _digest(row.get("request_journal_sha256"), "prediction request journal")
        _digest(row.get("response_journal_sha256"), "prediction response journal")
        _text(row.get("response_id"), "prediction response ID")
        _text(row.get("response_model"), "prediction response model")
        _require(row.get("finish_reason") == "stop", "prediction was not complete")
        elapsed = row.get("provider_elapsed_s")
        _require(
            type(elapsed) in {int, float}
            and math.isfinite(float(elapsed))
            and float(elapsed) >= 0,
            "prediction elapsed time is invalid",
        )
        _require(call_key not in call_keys, "prediction call keys are not unique")
        call_keys.add(call_key)


def answer_run(
    *,
    selection_path: Path,
    expected_selection_sha256: str,
    output_root: Path,
    expected_answer_preflight_sha256: str,
    authorized_provider_calls: int,
    enable_provider: bool,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    client_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    selection = _load_selection(selection_path, expected_selection_sha256)
    preflight_artifact = _read_bound(
        output_root / ANSWER_PREFLIGHT_NAME,
        expected_answer_preflight_sha256,
        "answer preflight",
    )
    validate_answer_preflight(
        preflight_artifact.payload,
        selection=selection.payload,
        selection_sha256=selection.sha256,
    )
    authorized = _nonnegative_int(
        authorized_provider_calls, "authorized provider calls"
    )
    predictions_path = output_root / ANSWERS_NAME
    with _phase_lock(output_root, "answer"):
        prompts, _bindings = _selection_prompts(selection.payload)
        runtime_factory = lambda client: _answer_runtime(  # noqa: E731
            output_root=output_root,
            prompts=prompts,
            preflight=preflight_artifact.payload,
            answer_preflight_sha256=preflight_artifact.sha256,
            client=client,
        )
        factory = client_factory or (
            lambda: _completion_client(
                api_key_env,
                str(preflight_artifact.payload["gateway_url"]),
            )
        )
        batch, physical, checkpoint_hits, batch_wall_time_s = _run_exactly_authorized(
            runtime_factory=runtime_factory,
            authorized_provider_calls=authorized,
            enable_provider=enable_provider,
            client_factory=factory,
        )
        payload = _prediction_payload(
            selection_sha256=selection.sha256,
            answer_preflight_sha256=preflight_artifact.sha256,
            preflight=preflight_artifact.payload,
            batch=batch,
        )
        validate_predictions(
            payload,
            selection_sha256=selection.sha256,
            answer_preflight=preflight_artifact.payload,
            answer_preflight_sha256=preflight_artifact.sha256,
        )
        if predictions_path.exists():
            _require(
                authorized == 0,
                "an existing sealed prediction artifact requires authorization 0",
            )
            artifact = read_sealed_json(predictions_path)
            validate_predictions(
                artifact.payload,
                selection_sha256=selection.sha256,
                answer_preflight=preflight_artifact.payload,
                answer_preflight_sha256=preflight_artifact.sha256,
            )
            _require(
                artifact.payload == payload,
                "sealed predictions differ from authenticated checkpoints",
            )
            created = False
        else:
            artifact, created = publish_sealed_json(predictions_path, payload)
            _require(created, "prediction artifact appeared during provider execution")
    return {
        "answers_sha256": artifact.sha256,
        "authenticated_checkpoint_hits": checkpoint_hits,
        "created": created,
        "new_provider_calls": physical,
        "completion_batch_wall_time_s": batch_wall_time_s,
        "question_count": harness.QUESTION_COUNT,
    }


def _load_locked_validation_question_population(
    dataset: Path,
    split_manifest: Path,
) -> tuple[str, list[Any]]:
    """Reconstruct the same ordered validation full100 used by retrieval.

    ``LOCKED_QUESTIONS`` stores global ordinals in the ten-shard validation
    population.  The historical ``load_original_population`` helper instead
    reconstructs one ten-question *development* concatenation, so indexing it
    with those ordinals silently addresses a different benchmark population.
    """

    from memory_condense.eval.recall_guarded_cumulative_population import (
        LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
        build_locked_cumulative_population_identity,
    )

    samples, identities, population = build_locked_cumulative_population_identity(
        dataset,
        split_manifest,
        plan=LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    )
    questions = [question for sample in samples for question in sample.questions]
    population_sha = _digest(
        population.get("population_identity_sha256"),
        "locked validation population identity",
    )
    _require(
        len(samples) == 10
        and len(identities) == 10
        and len(questions) == 100
        and population_sha == LOCKED_FULL100_POPULATION_SHA256,
        "locked validation full100 population changed",
    )
    return population_sha, questions


def _load_judge_material(
    dataset: Path,
    split_manifest: Path,
    prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, list[list[dict[str, str]]], list[dict[str, Any]]]:
    # Gold-bearing imports and reads are deliberately isolated behind this
    # function.  Callers must authenticate sealed predictions first.
    from memory_condense.eval.benchmark import build_judge_prompt
    population_sha, questions = _load_locked_validation_question_population(
        dataset, split_manifest
    )
    prompts: list[list[dict[str, str]]] = []
    bindings: list[dict[str, Any]] = []
    for reduced_ordinal, ((ordinal, question_id, dated_sha), prediction) in enumerate(
        zip(harness.LOCKED_QUESTIONS, prediction_rows, strict=True)
    ):
        question = questions[ordinal]
        _require(
            question.question_id == question_id
            and quote_sha256(question.dated_question) == dated_sha,
            f"gold population binding changed at zero-based ordinal {ordinal}",
        )
        prediction_text = _text(
            prediction.get("prediction"), f"sealed prediction {reduced_ordinal}"
        )
        messages = build_judge_prompt(
            question.question,
            question.answer,
            prediction_text,
        )
        prompts.append(messages)
        bindings.append(
            {
                "reduced_ordinal": reduced_ordinal,
                "global_ordinal": ordinal,
                "question_id": question_id,
                "prompt_question_sha256": dated_sha,
                "judge_question_sha256": quote_sha256(question.question),
                "reference_sha256": quote_sha256(question.answer),
                "prediction_sha256": prediction["prediction_sha256"],
                "category": question.category,
            }
        )
    return population_sha, prompts, bindings


def _judge_preflight_payload(
    *,
    selection_sha256: str,
    answer_preflight_sha256: str,
    answers_sha256: str,
    population_identity_sha256: str,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    bindings: Sequence[Mapping[str, Any]],
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    gateway = _text(gateway_url, "judge gateway URL")
    concurrency = _positive_int(max_concurrency, "judge max concurrency")
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=JUDGE_MAX_PROMPT_TOKENS,
    )
    _require(
        population.logical_prompt_count == harness.QUESTION_COUNT
        and population.unique_prompt_count == harness.QUESTION_COUNT,
        "judge prompts must be an exact unique reduced-30 population",
    )
    rows: list[dict[str, Any]] = []
    for binding, messages, prompt_row in zip(
        bindings, prompts, population.ordered_rows, strict=True
    ):
        plain = [dict(message) for message in messages]
        payload = canonical_json_bytes({"messages": plain})
        rows.append(
            {
                **dict(binding),
                "provider_messages": plain,
                "provider_payload_sha256": hashlib.sha256(payload).hexdigest(),
                "provider_payload_utf8_bytes": len(payload),
                "messages_sha256": prompt_row.messages_sha256,
                "prompt_token_proxy": prompt_row.prompt_token_proxy,
            }
        )
    body = {
        "format": JUDGE_PREFLIGHT_FORMAT,
        "status": "sealed_gold_joined_sol_judge_preflight",
        "selection_sha256": selection_sha256,
        "answer_preflight_sha256": answer_preflight_sha256,
        "answers_sha256": answers_sha256,
        "population_identity_sha256": _digest(
            population_identity_sha256, "population identity"
        ),
        "locked_question_identity_sha256": harness.LOCKED_IDENTITY_SHA256,
        "question_count": harness.QUESTION_COUNT,
        "model": SOL_MODEL,
        "gateway_url": gateway,
        "max_prompt_token_proxy": JUDGE_MAX_PROMPT_TOKENS,
        "max_completion_tokens": JUDGE_MAX_TOKENS,
        "max_concurrency": concurrency,
        "retries": 0,
        "request_options": {},
        "prompt_population": population.model_dump(),
        "required_authorized_provider_calls": harness.QUESTION_COUNT,
        "provider_calls": 0,
        "questions": rows,
        "gold_fields_present": True,
        "retained_request_token_state_bytes": 0,
    }
    return _sealed(body)


def validate_judge_preflight(
    payload: Mapping[str, Any],
    *,
    selection_sha256: str,
    answer_preflight_sha256: str,
    answers_sha256: str,
    prediction_rows: Sequence[Mapping[str, Any]],
) -> None:
    _validate_receipt(payload, "judge preflight")
    _require(
        payload.get("format") == JUDGE_PREFLIGHT_FORMAT
        and payload.get("status") == "sealed_gold_joined_sol_judge_preflight"
        and payload.get("selection_sha256") == selection_sha256
        and payload.get("answer_preflight_sha256") == answer_preflight_sha256
        and payload.get("answers_sha256") == answers_sha256
        and payload.get("locked_question_identity_sha256")
        == harness.LOCKED_IDENTITY_SHA256
        and payload.get("question_count") == harness.QUESTION_COUNT
        and payload.get("model") == SOL_MODEL
        and payload.get("max_prompt_token_proxy") == JUDGE_MAX_PROMPT_TOKENS
        and payload.get("max_completion_tokens") == JUDGE_MAX_TOKENS
        and payload.get("retries") == 0
        and payload.get("request_options") == {}
        and payload.get("required_authorized_provider_calls")
        == harness.QUESTION_COUNT
        and payload.get("provider_calls") == 0
        and payload.get("gold_fields_present") is True
        and payload.get("retained_request_token_state_bytes") == 0,
        "judge preflight header changed",
    )
    _digest(payload.get("population_identity_sha256"), "population identity")
    _text(payload.get("gateway_url"), "judge gateway URL")
    _positive_int(payload.get("max_concurrency"), "judge max concurrency")
    rows = _list(payload.get("questions"), "judge preflight questions")
    _require(len(rows) == harness.QUESTION_COUNT, "judge preflight count changed")
    prompts: list[list[dict[str, str]]] = []
    for index, (candidate, locked, prediction) in enumerate(
        zip(rows, harness.LOCKED_QUESTIONS, prediction_rows, strict=True)
    ):
        row = _mapping(candidate, f"judge preflight row {index}")
        ordinal, question_id, dated_sha = locked
        _require(
            row.get("reduced_ordinal") == index
            and row.get("global_ordinal") == ordinal
            and row.get("question_id") == question_id
            and row.get("prompt_question_sha256") == dated_sha
            and row.get("prediction_sha256") == prediction.get("prediction_sha256"),
            f"judge binding changed at zero-based ordinal {ordinal}",
        )
        _digest(row.get("judge_question_sha256"), "judge question hash")
        _digest(row.get("reference_sha256"), "judge reference hash")
        _require(
            row.get("category") is None or type(row.get("category")) is str,
            "judge category must be text or null",
        )
        messages = _plain_messages(row.get("provider_messages"), "judge messages")
        raw = canonical_json_bytes({"messages": messages})
        _require(
            row.get("provider_payload_sha256")
            == hashlib.sha256(raw).hexdigest()
            and row.get("provider_payload_utf8_bytes") == len(raw),
            f"judge provider payload changed at row {index}",
        )
        prompts.append(messages)
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=JUDGE_MAX_PROMPT_TOKENS
    )
    _require(
        population.logical_prompt_count == harness.QUESTION_COUNT
        and population.unique_prompt_count == harness.QUESTION_COUNT
        and payload.get("prompt_population") == population.model_dump(),
        "judge prompt population changed",
    )
    for row, prompt_row in zip(rows, population.ordered_rows, strict=True):
        _require(
            row.get("messages_sha256") == prompt_row.messages_sha256
            and row.get("prompt_token_proxy") == prompt_row.prompt_token_proxy,
            "judge prompt row binding changed",
        )


def judge_preflight(
    *,
    selection_path: Path,
    expected_selection_sha256: str,
    output_root: Path,
    expected_answer_preflight_sha256: str,
    expected_answers_sha256: str,
    dataset: Path,
    split_manifest: Path,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    # Authenticate every gold-free predecessor before the first gold read.
    selection = _load_selection(selection_path, expected_selection_sha256)
    answer_preflight_artifact = _read_bound(
        output_root / ANSWER_PREFLIGHT_NAME,
        expected_answer_preflight_sha256,
        "answer preflight",
    )
    validate_answer_preflight(
        answer_preflight_artifact.payload,
        selection=selection.payload,
        selection_sha256=selection.sha256,
    )
    answers_artifact = _read_bound(
        output_root / ANSWERS_NAME, expected_answers_sha256, "answers"
    )
    validate_predictions(
        answers_artifact.payload,
        selection_sha256=selection.sha256,
        answer_preflight=answer_preflight_artifact.payload,
        answer_preflight_sha256=answer_preflight_artifact.sha256,
    )
    prediction_rows = _list(
        answers_artifact.payload.get("questions"), "prediction questions"
    )

    existing_path = output_root / JUDGE_PREFLIGHT_NAME
    if existing_path.exists():
        artifact = read_sealed_json(existing_path)
        validate_judge_preflight(
            artifact.payload,
            selection_sha256=selection.sha256,
            answer_preflight_sha256=answer_preflight_artifact.sha256,
            answers_sha256=answers_artifact.sha256,
            prediction_rows=prediction_rows,
        )
        return {
            "created": False,
            "judge_preflight_sha256": artifact.sha256,
            "new_provider_calls": 0,
            "required_authorized_provider_calls": harness.QUESTION_COUNT,
        }

    population_sha, prompts, bindings = _load_judge_material(
        dataset, split_manifest, prediction_rows
    )
    payload = _judge_preflight_payload(
        selection_sha256=selection.sha256,
        answer_preflight_sha256=answer_preflight_artifact.sha256,
        answers_sha256=answers_artifact.sha256,
        population_identity_sha256=population_sha,
        prompts=prompts,
        bindings=bindings,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
    )
    validate_judge_preflight(
        payload,
        selection_sha256=selection.sha256,
        answer_preflight_sha256=answer_preflight_artifact.sha256,
        answers_sha256=answers_artifact.sha256,
        prediction_rows=prediction_rows,
    )
    artifact, created = publish_sealed_json(existing_path, payload)
    return {
        "created": created,
        "judge_preflight_sha256": artifact.sha256,
        "new_provider_calls": 0,
        "required_authorized_provider_calls": harness.QUESTION_COUNT,
    }


def _judge_runtime(
    *,
    output_root: Path,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    preflight: Mapping[str, Any],
    judge_preflight_sha256: str,
    client: Any | None,
) -> FastCompletionRuntime:
    provenance = {
        "phase": "reduced30_sol_judge",
        "selection_sha256": preflight["selection_sha256"],
        "answer_preflight_sha256": preflight["answer_preflight_sha256"],
        "answers_sha256": preflight["answers_sha256"],
        "judge_preflight_sha256": judge_preflight_sha256,
        "population_identity_sha256": preflight["population_identity_sha256"],
        "locked_question_identity_sha256": harness.LOCKED_IDENTITY_SHA256,
        "gateway_url": preflight["gateway_url"],
        "gold_fields_present": True,
    }
    return FastCompletionRuntime(
        checkpoint_dir=output_root / JUDGE_CHECKPOINT_DIR,
        prompt_population=prompts,
        model=SOL_MODEL,
        client=client,
        max_prompt_tokens=JUDGE_MAX_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=int(preflight["max_concurrency"]),
        retries=0,
        request_options={},
        benchmark_provenance=provenance,
    )


def _judgment_payload(
    *,
    judge_preflight: Mapping[str, Any],
    judge_preflight_sha256: str,
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    records = _record_map(batch)
    preflight_rows = _list(
        judge_preflight.get("questions"), "judge preflight questions"
    )
    rows: list[dict[str, Any]] = []
    for binding, prompt_row, verdict_text in zip(
        preflight_rows,
        batch.prompt_population.ordered_rows,
        batch.logical_completions,
        strict=True,
    ):
        record = records[prompt_row.messages_sha256]
        correct = parse_binary_judge_verdict(verdict_text)
        row_body = {
            "reduced_ordinal": binding["reduced_ordinal"],
            "global_ordinal": binding["global_ordinal"],
            "question_id": binding["question_id"],
            "category": binding["category"],
            "prompt_question_sha256": binding["prompt_question_sha256"],
            "judge_question_sha256": binding["judge_question_sha256"],
            "reference_sha256": binding["reference_sha256"],
            "prediction_sha256": binding["prediction_sha256"],
            "judge_messages_sha256": prompt_row.messages_sha256,
            "verdict_sha256": quote_sha256(verdict_text),
            "correct": correct,
            "call_key_sha256": record.call_key_sha256,
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "response_id": record.response_id,
            "response_model": record.response_model,
            "finish_reason": record.finish_reason,
            "provider_elapsed_s": record.provider_elapsed_s,
        }
        rows.append({**row_body, "row_receipt_sha256": identity_sha256(row_body)})
    correct_count = sum(row["correct"] is True for row in rows)
    body = {
        "format": JUDGMENTS_FORMAT,
        "status": "sealed_sol_semantic_judgments",
        "selection_sha256": judge_preflight["selection_sha256"],
        "answer_preflight_sha256": judge_preflight["answer_preflight_sha256"],
        "answers_sha256": judge_preflight["answers_sha256"],
        "judge_preflight_sha256": judge_preflight_sha256,
        "population_identity_sha256": judge_preflight[
            "population_identity_sha256"
        ],
        "locked_question_identity_sha256": harness.LOCKED_IDENTITY_SHA256,
        "judge_prompt_population_sha256": (
            batch.prompt_population.prompt_population_sha256
        ),
        "runtime_identity_sha256": batch.runtime_identity_sha256,
        "model": SOL_MODEL,
        "gateway_url": judge_preflight["gateway_url"],
        "max_prompt_token_proxy": JUDGE_MAX_PROMPT_TOKENS,
        "max_completion_tokens": JUDGE_MAX_TOKENS,
        "max_concurrency": judge_preflight["max_concurrency"],
        "retries": 0,
        "question_count": harness.QUESTION_COUNT,
        "provider_calls_completed": batch.usage.unique_calls,
        "correct_count": correct_count,
        "accuracy": correct_count / harness.QUESTION_COUNT,
        "completion_usage": _stable_usage(batch),
        "questions": rows,
        "gold_fields_present": True,
        "retained_request_token_state_bytes": 0,
    }
    return _sealed(body)


def validate_judgments(
    payload: Mapping[str, Any],
    *,
    judge_preflight: Mapping[str, Any],
    judge_preflight_sha256: str,
) -> None:
    _validate_receipt(payload, "judgments")
    _require(
        payload.get("format") == JUDGMENTS_FORMAT
        and payload.get("status") == "sealed_sol_semantic_judgments"
        and payload.get("selection_sha256")
        == judge_preflight.get("selection_sha256")
        and payload.get("answer_preflight_sha256")
        == judge_preflight.get("answer_preflight_sha256")
        and payload.get("answers_sha256") == judge_preflight.get("answers_sha256")
        and payload.get("judge_preflight_sha256") == judge_preflight_sha256
        and payload.get("population_identity_sha256")
        == judge_preflight.get("population_identity_sha256")
        and payload.get("locked_question_identity_sha256")
        == harness.LOCKED_IDENTITY_SHA256
        and payload.get("model") == SOL_MODEL
        and payload.get("gateway_url") == judge_preflight.get("gateway_url")
        and payload.get("max_prompt_token_proxy") == JUDGE_MAX_PROMPT_TOKENS
        and payload.get("max_completion_tokens") == JUDGE_MAX_TOKENS
        and payload.get("max_concurrency")
        == judge_preflight.get("max_concurrency")
        and payload.get("retries") == 0
        and payload.get("question_count") == harness.QUESTION_COUNT
        and payload.get("provider_calls_completed") == harness.QUESTION_COUNT
        and payload.get("gold_fields_present") is True
        and payload.get("retained_request_token_state_bytes") == 0,
        "judgment header changed",
    )
    _digest(payload.get("judge_prompt_population_sha256"), "judge population")
    _digest(payload.get("runtime_identity_sha256"), "judge runtime identity")
    usage = _mapping(payload.get("completion_usage"), "judgment completion usage")
    _require(
        "physical_calls" not in usage
        and "checkpoint_hits" not in usage
        and usage.get("logical_calls") == harness.QUESTION_COUNT
        and usage.get("unique_calls") == harness.QUESTION_COUNT,
        "judgment stable usage changed",
    )
    expected_rows = _list(
        judge_preflight.get("questions"), "judge preflight questions"
    )
    rows = _list(payload.get("questions"), "judgment questions")
    _require(len(rows) == len(expected_rows), "judgment count changed")
    correct_count = 0
    call_keys: set[str] = set()
    for index, (candidate, binding) in enumerate(
        zip(rows, expected_rows, strict=True)
    ):
        row = _mapping(candidate, f"judgment row {index}")
        body = dict(row)
        receipt = body.pop("row_receipt_sha256", None)
        _require(receipt == identity_sha256(body), f"judgment row {index} changed")
        for name in (
            "reduced_ordinal",
            "global_ordinal",
            "question_id",
            "category",
            "prompt_question_sha256",
            "judge_question_sha256",
            "reference_sha256",
            "prediction_sha256",
        ):
            _require(row.get(name) == binding.get(name), f"judgment {name} changed")
        _require(
            row.get("judge_messages_sha256") == binding.get("messages_sha256"),
            "judgment prompt hash changed",
        )
        _digest(row.get("verdict_sha256"), "judge verdict")
        _require(type(row.get("correct")) is bool, "judge verdict must be boolean")
        correct_count += int(row["correct"])
        call_key = _digest(row.get("call_key_sha256"), "judge call key")
        _digest(row.get("request_journal_sha256"), "judge request journal")
        _digest(row.get("response_journal_sha256"), "judge response journal")
        _text(row.get("response_id"), "judge response ID")
        _text(row.get("response_model"), "judge response model")
        _require(row.get("finish_reason") == "stop", "judge completion was not complete")
        elapsed = row.get("provider_elapsed_s")
        _require(
            type(elapsed) in {int, float}
            and math.isfinite(float(elapsed))
            and float(elapsed) >= 0,
            "judge elapsed time is invalid",
        )
        _require(call_key not in call_keys, "judge call keys are not unique")
        call_keys.add(call_key)
    _require(
        payload.get("correct_count") == correct_count
        and payload.get("accuracy") == correct_count / harness.QUESTION_COUNT,
        "judgment aggregate changed",
    )


def judge_run(
    *,
    selection_path: Path,
    expected_selection_sha256: str,
    output_root: Path,
    expected_answer_preflight_sha256: str,
    expected_answers_sha256: str,
    expected_judge_preflight_sha256: str,
    authorized_provider_calls: int,
    enable_provider: bool,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    client_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    selection = _load_selection(selection_path, expected_selection_sha256)
    answer_preflight_artifact = _read_bound(
        output_root / ANSWER_PREFLIGHT_NAME,
        expected_answer_preflight_sha256,
        "answer preflight",
    )
    validate_answer_preflight(
        answer_preflight_artifact.payload,
        selection=selection.payload,
        selection_sha256=selection.sha256,
    )
    answers_artifact = _read_bound(
        output_root / ANSWERS_NAME, expected_answers_sha256, "answers"
    )
    validate_predictions(
        answers_artifact.payload,
        selection_sha256=selection.sha256,
        answer_preflight=answer_preflight_artifact.payload,
        answer_preflight_sha256=answer_preflight_artifact.sha256,
    )
    prediction_rows = _list(
        answers_artifact.payload.get("questions"), "prediction questions"
    )
    judge_preflight_artifact = _read_bound(
        output_root / JUDGE_PREFLIGHT_NAME,
        expected_judge_preflight_sha256,
        "judge preflight",
    )
    validate_judge_preflight(
        judge_preflight_artifact.payload,
        selection_sha256=selection.sha256,
        answer_preflight_sha256=answer_preflight_artifact.sha256,
        answers_sha256=answers_artifact.sha256,
        prediction_rows=prediction_rows,
    )
    authorized = _nonnegative_int(
        authorized_provider_calls, "authorized provider calls"
    )
    judgments_path = output_root / JUDGMENTS_NAME
    with _phase_lock(output_root, "judge"):
        prompts = [
            _plain_messages(row["provider_messages"], "judge messages")
            for row in _list(
                judge_preflight_artifact.payload.get("questions"),
                "judge preflight questions",
            )
        ]
        runtime_factory = lambda client: _judge_runtime(  # noqa: E731
            output_root=output_root,
            prompts=prompts,
            preflight=judge_preflight_artifact.payload,
            judge_preflight_sha256=judge_preflight_artifact.sha256,
            client=client,
        )
        factory = client_factory or (
            lambda: _completion_client(
                api_key_env,
                str(judge_preflight_artifact.payload["gateway_url"]),
            )
        )
        batch, physical, checkpoint_hits, batch_wall_time_s = _run_exactly_authorized(
            runtime_factory=runtime_factory,
            authorized_provider_calls=authorized,
            enable_provider=enable_provider,
            client_factory=factory,
        )
        payload = _judgment_payload(
            judge_preflight=judge_preflight_artifact.payload,
            judge_preflight_sha256=judge_preflight_artifact.sha256,
            batch=batch,
        )
        validate_judgments(
            payload,
            judge_preflight=judge_preflight_artifact.payload,
            judge_preflight_sha256=judge_preflight_artifact.sha256,
        )
        if judgments_path.exists():
            _require(
                authorized == 0,
                "an existing sealed judgment artifact requires authorization 0",
            )
            artifact = read_sealed_json(judgments_path)
            validate_judgments(
                artifact.payload,
                judge_preflight=judge_preflight_artifact.payload,
                judge_preflight_sha256=judge_preflight_artifact.sha256,
            )
            _require(
                artifact.payload == payload,
                "sealed judgments differ from authenticated checkpoints",
            )
            created = False
        else:
            artifact, created = publish_sealed_json(judgments_path, payload)
            _require(created, "judgment artifact appeared during provider execution")
    return {
        "authenticated_checkpoint_hits": checkpoint_hits,
        "correct_count": payload["correct_count"],
        "created": created,
        "judgments_sha256": artifact.sha256,
        "new_provider_calls": physical,
        "completion_batch_wall_time_s": batch_wall_time_s,
        "question_count": harness.QUESTION_COUNT,
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    answer_plan = commands.add_parser(
        "answer-preflight", help="seal the gold-free 30-call Terra plan"
    )
    _common(answer_plan)
    answer_plan.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    answer_plan.add_argument(
        "--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY
    )

    answer = commands.add_parser(
        "answer-run", help="execute exactly the authorized Terra misses"
    )
    _common(answer)
    answer.add_argument("--expected-answer-preflight-sha256", required=True)
    answer.add_argument("--authorized-provider-calls", type=int, required=True)
    answer.add_argument("--enable-provider", action="store_true")
    answer.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)

    judge_plan = commands.add_parser(
        "judge-preflight", help="join gold only after sealed predictions"
    )
    _common(judge_plan)
    judge_plan.add_argument("--expected-answer-preflight-sha256", required=True)
    judge_plan.add_argument("--expected-answers-sha256", required=True)
    judge_plan.add_argument("--dataset", type=Path, required=True)
    judge_plan.add_argument("--split-manifest", type=Path, required=True)
    judge_plan.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    judge_plan.add_argument(
        "--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY
    )

    judge = commands.add_parser(
        "judge-run", help="execute exactly the authorized Sol misses"
    )
    _common(judge)
    judge.add_argument("--expected-answer-preflight-sha256", required=True)
    judge.add_argument("--expected-answers-sha256", required=True)
    judge.add_argument("--expected-judge-preflight-sha256", required=True)
    judge.add_argument("--authorized-provider-calls", type=int, required=True)
    judge.add_argument("--enable-provider", action="store_true")
    judge.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "selection_path": args.selection.resolve(),
        "expected_selection_sha256": args.expected_selection_sha256,
        "output_root": args.output_root.resolve(),
    }
    if args.command == "answer-preflight":
        result = answer_preflight(
            **common,
            gateway_url=args.gateway_url,
            max_concurrency=args.max_concurrency,
        )
    elif args.command == "answer-run":
        result = answer_run(
            **common,
            expected_answer_preflight_sha256=(
                args.expected_answer_preflight_sha256
            ),
            authorized_provider_calls=args.authorized_provider_calls,
            enable_provider=args.enable_provider,
            api_key_env=args.api_key_env,
        )
    elif args.command == "judge-preflight":
        result = judge_preflight(
            **common,
            expected_answer_preflight_sha256=(
                args.expected_answer_preflight_sha256
            ),
            expected_answers_sha256=args.expected_answers_sha256,
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
            gateway_url=args.gateway_url,
            max_concurrency=args.max_concurrency,
        )
    elif args.command == "judge-run":
        result = judge_run(
            **common,
            expected_answer_preflight_sha256=(
                args.expected_answer_preflight_sha256
            ),
            expected_answers_sha256=args.expected_answers_sha256,
            expected_judge_preflight_sha256=(
                args.expected_judge_preflight_sha256
            ),
            authorized_provider_calls=args.authorized_provider_calls,
            enable_provider=args.enable_provider,
            api_key_env=args.api_key_env,
        )
    else:  # pragma: no cover
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANSWER_PREFLIGHT_NAME",
    "ANSWERS_NAME",
    "JUDGE_PREFLIGHT_NAME",
    "JUDGMENTS_NAME",
    "answer_preflight",
    "answer_run",
    "build_answer_preflight",
    "judge_preflight",
    "judge_run",
    "main",
    "validate_answer_preflight",
    "validate_judge_preflight",
    "validate_judgments",
    "validate_predictions",
]
