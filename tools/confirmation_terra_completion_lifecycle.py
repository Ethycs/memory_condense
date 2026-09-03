#!/usr/bin/env python3
"""Generic sealed Terra completion lifecycle for confirmation prompts.

The input is a sidecar-sealed, gold-blind confirmation prompt preflight whose
ordered rows carry self-sealed ``provider_input`` messages.  The lifecycle is:

``preflight -> approve-release -> provider-run -> materialize -> replay``.

Only ``provider-run`` has a client construction path.  It is unreachable until
the source prompt, lifecycle preflight, release, checkpoint population, explicit
provider opt-in, and exact remaining-call authorization all verify.  Incomplete
request/response journal pairs are terminal and are never retried.

The implementation has no benchmark loader and is population-size neutral.
The Sol judge remains a separate evaluator-only lifecycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from memory_condense.domain.discourse import (  # noqa: E402
    identity_sha256,
    quote_sha256,
)
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRecord,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
)
from tools.confirmation_canonical import (  # noqa: E402
    FileSnapshot,
    FirebreakError,
    assert_snapshot_unchanged,
    canonical_json_bytes,
    canonical_sha256,
    exact_keys,
    parse_json_bytes,
    publish_no_clobber,
    read_snapshot,
    require_int,
    require_list,
    require_mapping,
    require_sha256,
    require_text,
)


FORMAT = "memory-condense-confirmation-terra-completion-lifecycle-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
RELEASE_FORMAT = f"{FORMAT}-provider-release-v1"
COMPLETION_FORMAT = f"{FORMAT}-completions-v1"
COMPLETION_ROW_FORMAT = f"{COMPLETION_FORMAT}-row-v1"
PROVIDER_INPUT_FORMAT = "memory-condense-confirmation-terra-provider-input-v1"
S0_PROMPT_FORMAT = "memory-condense-confirmation-matched-s0-terra-preflight-v1"
INTERMEDIATE_PROMPT_FORMAT = (
    "memory-condense-confirmation-intermediate-terra-preflight-v1"
)
INTERMEDIATE_PROMPT_ROW_FORMAT = f"{INTERMEDIATE_PROMPT_FORMAT}-row-v1"
TERMINAL_PROMPT_FORMAT = "memory-condense-confirmation-terminal-policy-preflight-v1"
SUPPORTED_PROMPT_FORMATS = frozenset(
    {S0_PROMPT_FORMAT, INTERMEDIATE_PROMPT_FORMAT, TERMINAL_PROMPT_FORMAT}
)

PREFLIGHT_NAME = "confirmation-terra-completion-preflight-v1.json"
RELEASE_NAME = "confirmation-terra-provider-release-v1.json"
COMPLETION_NAME = "confirmation-terra-completions-v1.json"
REPLAY_NAME = "confirmation-terra-completions-replay-v1.json"
CHECKPOINT_DIR_NAME = "confirmation-terra-completion-calls"

TERRA_MODEL = "codex_sdk/gpt-5.6-terra"
TERRA_GATEWAY_URL = "https://central-dev.zt:4000/v1"
DEFAULT_API_KEY_ENV = "LITELLM_KEY"

_JOURNAL_FILENAME = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)
_MESSAGE_ROLES = frozenset({"system", "user", "assistant"})
_RUNTIME_KEYS = {
    "gateway_url",
    "hard_complete_chat_token_cap",
    "input_token_cap",
    "max_concurrency",
    "model",
    "output_token_reserve",
    "retry_count",
}
_PROVIDER_INPUT_KEYS = {
    "format",
    "messages",
    "messages_sha256",
    "provider_input_receipt_sha256",
}
_LIFECYCLE_PREFLIGHT_KEYS = {
    "format",
    "status",
    "gold_loaded",
    "source_prompt",
    "runtime",
    "population",
    "ordered_rows",
    "execution",
    "lifecycle_preflight_identity_sha256",
}
_RELEASE_KEYS = {
    "format",
    "release_status",
    "approval_opt_in",
    "gold_loaded",
    "source_prompt_artifact_sha256",
    "lifecycle_preflight_sha256",
    "runtime",
    "population",
    "output_root",
    "output_root_sha256",
    "checkpoint_snapshot",
    "required_authorized_provider_calls",
    "unsafe_retry_policy",
    "provider_calls_during_release",
    "release_identity_sha256",
}
_COMPLETION_KEYS = {
    "format",
    "status",
    "gold_loaded",
    "source_prompt_artifact_sha256",
    "lifecycle_preflight_sha256",
    "provider_release_sha256",
    "runtime",
    "population",
    "ordered_rows",
    "completion_batch",
    "physical_provider_calls_during_materialization",
    "completion_artifact_identity_sha256",
}
_COMPLETION_ROW_KEYS = {
    "format",
    "row_index",
    "source_prompt_row_index",
    "question_id",
    "source_prompt_row_receipt_sha256",
    "messages_sha256",
    "completion",
    "completion_sha256",
    "call_key_sha256",
    "request_journal_sha256",
    "response_journal_sha256",
    "completion_row_receipt_sha256",
}


class ConfirmationTerraLifecycleError(ValueError):
    """The sealed Terra completion lifecycle failed closed."""


@dataclass(frozen=True, slots=True)
class SealedArtifact:
    path: Path
    snapshot: FileSnapshot
    sidecar: FileSnapshot
    payload: dict[str, Any]

    @property
    def sha256(self) -> str:
        return self.snapshot.sha256


@dataclass(frozen=True, slots=True)
class VerifiedPromptArtifact:
    artifact: SealedArtifact
    source_format: str
    source_identity_sha256: str
    runtime: dict[str, Any]
    source_question_ids: tuple[str, ...]
    source_row_indexes: tuple[int, ...]
    question_ids: tuple[str, ...]
    row_receipts: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    prompts: tuple[tuple[dict[str, str], ...], ...]
    prompt_population: FastPromptPopulation


ClientFactory = Callable[[str, str], Any]


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationTerraLifecycleError(message)


def _sidecar_bytes(path: Path, digest: str) -> bytes:
    return f"{digest}  {path.name}\n".encode("ascii")


def read_sealed_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> SealedArtifact:
    target = Path(path)
    sidecar_path = target.with_name(target.name + ".sha256")
    _require(target.is_file() and not target.is_symlink(), f"{label} is not a regular file")
    _require(
        sidecar_path.is_file() and not sidecar_path.is_symlink(),
        f"{label} digest sidecar is absent or unsafe",
    )
    try:
        snapshot = read_snapshot(target, label)
        sidecar = read_snapshot(sidecar_path, f"{label} digest sidecar")
        expected = require_sha256(expected_sha256, f"expected {label} SHA-256")
        payload = require_mapping(parse_json_bytes(snapshot.payload, label), label)
    except (FirebreakError, ValueError) as exc:
        raise ConfirmationTerraLifecycleError(str(exc)) from exc
    _require(snapshot.sha256 == expected, f"{label} differs from its external seal")
    _require(
        sidecar.payload == _sidecar_bytes(target, snapshot.sha256),
        f"{label} digest sidecar is invalid",
    )
    _require(
        snapshot.payload == canonical_json_bytes(payload) + b"\n",
        f"{label} is not canonical JSON",
    )
    return SealedArtifact(target.resolve(), snapshot, sidecar, payload)


def publish_sealed_artifact(
    path: str | Path,
    payload: Mapping[str, Any],
) -> tuple[SealedArtifact, bool]:
    target = Path(path)
    sidecar_path = target.with_name(target.name + ".sha256")
    value = dict(payload)
    raw = canonical_json_bytes(value) + b"\n"
    digest = hashlib.sha256(raw).hexdigest()
    if (
        target.exists()
        or target.is_symlink()
        or sidecar_path.exists()
        or sidecar_path.is_symlink()
    ):
        existing = read_sealed_artifact(
            target,
            expected_sha256=digest,
            label="existing lifecycle artifact",
        )
        _require(existing.payload == value, "refusing to replace a different lifecycle artifact")
        return existing, False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        publish_no_clobber(target, raw)
        try:
            publish_no_clobber(sidecar_path, _sidecar_bytes(target, digest))
        except (FirebreakError, OSError):
            if target.is_file() and not target.is_symlink() and target.read_bytes() == raw:
                target.unlink()
            raise
    except (FirebreakError, OSError) as exc:
        raise ConfirmationTerraLifecycleError("cannot publish sealed lifecycle artifact") from exc
    return (
        read_sealed_artifact(
            target,
            expected_sha256=digest,
            label="published lifecycle artifact",
        ),
        True,
    )


def _plain_messages(value: object, label: str) -> tuple[dict[str, str], ...]:
    try:
        raw_messages = require_list(value, label)
    except FirebreakError as exc:
        raise ConfirmationTerraLifecycleError(str(exc)) from exc
    _require(bool(raw_messages), f"{label} cannot be empty")
    messages: list[dict[str, str]] = []
    for index, raw in enumerate(raw_messages):
        try:
            message = require_mapping(raw, f"{label} message {index}")
            exact_keys(message, {"role", "content"}, f"{label} message {index}")
            role = require_text(message["role"], f"{label} message {index} role")
            content = require_text(
                message["content"],
                f"{label} message {index} content",
                allow_empty=True,
            )
        except FirebreakError as exc:
            raise ConfirmationTerraLifecycleError(str(exc)) from exc
        _require(role in _MESSAGE_ROLES, f"{label} message {index} role is unsupported")
        messages.append({"role": role, "content": content})
    return tuple(messages)


def _verify_runtime(value: object) -> dict[str, Any]:
    try:
        runtime = require_mapping(value, "Terra runtime")
        exact_keys(runtime, _RUNTIME_KEYS, "Terra runtime")
        gateway = require_text(runtime["gateway_url"], "Terra gateway")
        model = require_text(runtime["model"], "Terra model")
        hard_cap = require_int(runtime["hard_complete_chat_token_cap"], "hard prompt cap", minimum=1)
        input_cap = require_int(runtime["input_token_cap"], "input prompt cap", minimum=1)
        output_reserve = require_int(runtime["output_token_reserve"], "output reserve", minimum=1)
        concurrency = require_int(runtime["max_concurrency"], "max concurrency", minimum=1)
        retries = require_int(runtime["retry_count"], "retry count")
    except FirebreakError as exc:
        raise ConfirmationTerraLifecycleError(str(exc)) from exc
    _require(gateway == TERRA_GATEWAY_URL, "Terra gateway route changed")
    _require(model == TERRA_MODEL, "Terra model route changed")
    _require(retries == 0, "Terra retries must equal zero")
    _require(input_cap + output_reserve == hard_cap, "Terra input/output budgets do not sum to the hard cap")
    return {
        "gateway_url": gateway,
        "hard_complete_chat_token_cap": hard_cap,
        "input_token_cap": input_cap,
        "max_concurrency": concurrency,
        "model": model,
        "output_token_reserve": output_reserve,
        "retry_count": retries,
    }


def _self_seal(value: Mapping[str, Any], key: str, label: str) -> str:
    try:
        declared = require_sha256(value.get(key), f"{label} receipt")
    except FirebreakError as exc:
        raise ConfirmationTerraLifecycleError(str(exc)) from exc
    body = dict(value)
    body.pop(key, None)
    _require(canonical_sha256(body) == declared, f"{label} self-seal differs")
    return declared


def compile_intermediate_prompt_artifact(
    *,
    stage_id: str,
    ordered_question_ids: Sequence[str],
    source_row_receipts: Sequence[str],
    messages: Sequence[Sequence[Mapping[str, str]]],
    runtime: Mapping[str, Any],
    stage_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile one reusable, all-rows-provider-bound intermediate prompt.

    Stage-specific semantics stay in ``stage_id`` and ``stage_bindings``.  The
    lifecycle owns only the closed provider-input envelope, population order,
    runtime budget, and call-count contract.
    """

    try:
        normalized_stage = require_text(stage_id, "intermediate stage ID")
        question_ids = tuple(
            require_text(value, f"intermediate question ID {index}")
            for index, value in enumerate(ordered_question_ids)
        )
        receipts = tuple(
            require_sha256(value, f"intermediate source row receipt {index}")
            for index, value in enumerate(source_row_receipts)
        )
        detached_bindings = require_mapping(
            parse_json_bytes(
                canonical_json_bytes(dict(stage_bindings)),
                "intermediate stage bindings",
            ),
            "intermediate stage bindings",
        )
    except (FirebreakError, TypeError, ValueError) as exc:
        raise ConfirmationTerraLifecycleError(str(exc)) from exc
    _require(bool(question_ids), "intermediate prompt population is empty")
    _require(
        len(question_ids) == len(receipts) == len(messages),
        "intermediate prompt inputs have unequal populations",
    )
    _require(
        len(question_ids) == len(set(question_ids)),
        "intermediate prompt question IDs repeat",
    )
    verified_runtime = _verify_runtime(runtime)
    normalized_messages = tuple(
        _plain_messages(value, f"intermediate prompt {index}")
        for index, value in enumerate(messages)
    )
    try:
        population = preflight_fast_completion_prompts(
            normalized_messages,
            max_prompt_tokens=verified_runtime["input_token_cap"],
        )
    except ValueError as exc:
        raise ConfirmationTerraLifecycleError(
            f"intermediate prompt population failed validation: {exc}"
        ) from exc

    rows: list[dict[str, Any]] = []
    for index, (question_id, source_receipt, prompt, prompt_row) in enumerate(
        zip(
            question_ids,
            receipts,
            normalized_messages,
            population.ordered_rows,
            strict=True,
        )
    ):
        provider_body = {
            "format": PROVIDER_INPUT_FORMAT,
            "messages": list(prompt),
            "messages_sha256": prompt_row.messages_sha256,
        }
        provider_input = {
            **provider_body,
            "provider_input_receipt_sha256": canonical_sha256(provider_body),
        }
        row_body = {
            "format": INTERMEDIATE_PROMPT_ROW_FORMAT,
            "row_index": index,
            "question_id": question_id,
            "source_row_receipt_sha256": source_receipt,
            "messages_sha256": prompt_row.messages_sha256,
            "prompt_token_proxy": prompt_row.prompt_token_proxy,
            "provider_input": provider_input,
        }
        rows.append(
            {**row_body, "row_receipt_sha256": canonical_sha256(row_body)}
        )
    body: dict[str, Any] = {
        "format": INTERMEDIATE_PROMPT_FORMAT,
        "status": "compiled",
        "gold_loaded": False,
        "stage_id": normalized_stage,
        "bindings": dict(detached_bindings),
        "population": {
            "question_count": len(question_ids),
            "ordered_question_ids_sha256": canonical_sha256(list(question_ids)),
        },
        "runtime": verified_runtime,
        "execution": {
            "logical_prompt_count": len(question_ids),
            "unique_prompt_count": population.unique_prompt_count,
            "would_call_count": population.unique_prompt_count,
            "would_call_count_status": "exact",
            "physical_provider_calls": 0,
            "provider_execution_available": False,
            "authorization_released": False,
        },
        "ordered_rows": rows,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
    }
    try:
        assert_gold_blind(body, path="confirmation_intermediate_prompt")
    except MatchedEvalContractError as exc:
        raise ConfirmationTerraLifecycleError(str(exc)) from exc
    return {**body, "preflight_identity_sha256": canonical_sha256(body)}


def verify_prompt_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
) -> VerifiedPromptArtifact:
    """Verify a self-contained, sealed confirmation Terra prompt preflight."""

    artifact = read_sealed_artifact(
        path,
        expected_sha256=expected_sha256,
        label="confirmation Terra prompt artifact",
    )
    value = artifact.payload
    try:
        source_format = require_text(value.get("format"), "prompt artifact format")
        status = require_text(value.get("status"), "prompt artifact status")
        runtime = _verify_runtime(value.get("runtime"))
        population = require_mapping(value.get("population"), "prompt population binding")
        execution = require_mapping(value.get("execution"), "prompt execution binding")
        raw_rows = require_list(value.get("ordered_rows"), "ordered prompt rows")
    except FirebreakError as exc:
        raise ConfirmationTerraLifecycleError(str(exc)) from exc
    _require(status == "compiled", "prompt artifact is not compiled")
    _require(value.get("gold_loaded") is False, "prompt artifact crossed the gold firewall")
    _require(
        source_format in SUPPORTED_PROMPT_FORMATS,
        "prompt artifact format is not an approved confirmation Terra preflight",
    )
    source_identity = _self_seal(value, "preflight_identity_sha256", "prompt artifact")
    try:
        assert_gold_blind(value, path="confirmation_terra_prompt_artifact")
    except MatchedEvalContractError as exc:
        raise ConfirmationTerraLifecycleError(str(exc)) from exc

    source_question_ids: list[str] = []
    source_row_indexes: list[int] = []
    question_ids: list[str] = []
    row_receipts: list[str] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        try:
            row = require_mapping(raw, f"prompt row {index}")
            question_id = require_text(row.get("question_id"), f"prompt row {index} question ID")
        except FirebreakError as exc:
            raise ConfirmationTerraLifecycleError(str(exc)) from exc
        if "row_index" in row:
            try:
                row_index = require_int(row["row_index"], f"prompt row {index} index")
            except FirebreakError as exc:
                raise ConfirmationTerraLifecycleError(str(exc)) from exc
            _require(row_index == index, f"prompt row {index} position changed")
        row_receipt = _self_seal(row, "row_receipt_sha256", f"prompt row {index}")
        source_question_ids.append(question_id)
        provider_value = row.get("provider_input")
        if provider_value is None:
            _require(
                source_format == TERMINAL_PROMPT_FORMAT
                and row.get("would_call") is False
                and row.get("prompt_token_proxy") is None,
                f"prompt row {index} omits a required sealed provider input",
            )
            continue
        try:
            provider = require_mapping(
                provider_value, f"prompt row {index} provider input"
            )
            exact_keys(provider, _PROVIDER_INPUT_KEYS, f"prompt row {index} provider input")
            messages_sha = require_sha256(
                provider.get("messages_sha256"), f"prompt row {index} messages"
            )
            prompt_tokens = require_int(
                row.get("prompt_token_proxy"), f"prompt row {index} token proxy", minimum=1
            )
        except FirebreakError as exc:
            raise ConfirmationTerraLifecycleError(str(exc)) from exc
        if source_format == TERMINAL_PROMPT_FORMAT:
            _require(row.get("would_call") is True, f"prompt row {index} call disposition changed")
        _require(provider.get("format") == PROVIDER_INPUT_FORMAT, f"prompt row {index} provider input format changed")
        provider_receipt = _self_seal(
            provider,
            "provider_input_receipt_sha256",
            f"prompt row {index} provider input",
        )
        messages = _plain_messages(provider.get("messages"), f"prompt row {index} messages")
        plain = list(messages)
        _require(
            messages_sha == canonical_sha256(plain)
            and row.get("messages_sha256", messages_sha) == messages_sha,
            f"prompt row {index} message identity differs",
        )
        _require(bool(provider_receipt), f"prompt row {index} provider input is not sealed")
        source_row_indexes.append(index)
        question_ids.append(question_id)
        row_receipts.append(row_receipt)
        prompts.append(messages)
        rows.append(dict(row))
        _require(prompt_tokens <= runtime["input_token_cap"], f"prompt row {index} exceeds the input cap")

    _require(bool(source_question_ids), "prompt artifact has no rows")
    _require(bool(rows), "prompt artifact has no provider-bound rows")
    _require(
        len(source_question_ids) == len(set(source_question_ids)),
        "prompt question IDs repeat",
    )
    try:
        rebuilt_population = preflight_fast_completion_prompts(
            prompts,
            max_prompt_tokens=runtime["input_token_cap"],
        )
    except ValueError as exc:
        raise ConfirmationTerraLifecycleError(f"prompt population failed validation: {exc}") from exc
    if "prompt_population" in value or "prompt_population_sha256" in value:
        try:
            declared_population = require_mapping(
                value.get("prompt_population"), "fast prompt population"
            )
            declared_population_sha = require_sha256(
                value.get("prompt_population_sha256"),
                "fast prompt population SHA-256",
            )
        except FirebreakError as exc:
            raise ConfirmationTerraLifecycleError(str(exc)) from exc
        _require(
            rebuilt_population.model_dump() == declared_population
            and rebuilt_population.prompt_population_sha256
            == declared_population_sha,
            "fast prompt population differs from sealed provider inputs",
        )
        _require(
            all(
                row["prompt_token_proxy"] == prompt_row.prompt_token_proxy
                for row, prompt_row in zip(
                    rows, rebuilt_population.ordered_rows, strict=True
                )
            ),
            "prompt-row token counts differ from the sealed prompt population",
        )
    else:
        _require(
            source_format == TERMINAL_PROMPT_FORMAT,
            "S0 prompt artifact omits its fast prompt population",
        )
    logical = len(rows)
    unique = rebuilt_population.unique_prompt_count
    _require(
        population.get("question_count") == len(source_question_ids)
        and population.get("ordered_question_ids_sha256")
        == canonical_sha256(source_question_ids),
        "prompt question population binding differs",
    )
    all_rows_provider_bound = source_format in {
        S0_PROMPT_FORMAT,
        INTERMEDIATE_PROMPT_FORMAT,
    }
    declared_logical = execution.get(
        "logical_prompt_count"
        if all_rows_provider_bound
        else "logical_terminal_prompt_count"
    )
    declared_would_call = unique if all_rows_provider_bound else logical
    provider_execution_available = execution.get(
        "provider_execution_available",
        value.get("provider_execution_available"),
    )
    authorization_released = execution.get(
        "authorization_released",
        value.get("authorization_released"),
    )
    _require(
        declared_logical == logical
        and execution.get("unique_prompt_count", unique) == unique
        and execution.get("would_call_count") == declared_would_call
        and execution.get("would_call_count_status") == "exact"
        and execution.get("physical_provider_calls") == 0
        and provider_execution_available is False
        and authorization_released is False,
        "prompt execution preflight is not exact and provider-free",
    )
    assert_snapshot_unchanged(artifact.snapshot, "confirmation Terra prompt artifact")
    assert_snapshot_unchanged(artifact.sidecar, "confirmation Terra prompt artifact sidecar")
    return VerifiedPromptArtifact(
        artifact=artifact,
        source_format=source_format,
        source_identity_sha256=source_identity,
        runtime=runtime,
        source_question_ids=tuple(source_question_ids),
        source_row_indexes=tuple(source_row_indexes),
        question_ids=tuple(question_ids),
        row_receipts=tuple(row_receipts),
        rows=tuple(rows),
        prompts=tuple(prompts),
        prompt_population=rebuilt_population,
    )


def compile_lifecycle_preflight(source: VerifiedPromptArtifact) -> dict[str, Any]:
    row_bindings = [
        {
            "row_index": index,
            "source_prompt_row_index": source_row_index,
            "question_id": question_id,
            "source_prompt_row_receipt_sha256": receipt,
            "messages_sha256": source.prompt_population.ordered_rows[index].messages_sha256,
            "prompt_token_proxy": source.prompt_population.ordered_rows[index].prompt_token_proxy,
        }
        for index, (source_row_index, question_id, receipt) in enumerate(
            zip(
                source.source_row_indexes,
                source.question_ids,
                source.row_receipts,
                strict=True,
            )
        )
    ]
    body: dict[str, Any] = {
        "format": PREFLIGHT_FORMAT,
        "status": "verified",
        "gold_loaded": False,
        "source_prompt": {
            "format": source.source_format,
            "artifact_sha256": source.artifact.sha256,
            "artifact_identity_sha256": source.source_identity_sha256,
        },
        "runtime": dict(source.runtime),
        "population": {
            "source_question_count": len(source.source_question_ids),
            "logical_prompt_count": len(source.rows),
            "unique_prompt_count": source.prompt_population.unique_prompt_count,
            "ordered_source_question_ids_sha256": canonical_sha256(
                list(source.source_question_ids)
            ),
            "ordered_question_ids_sha256": canonical_sha256(list(source.question_ids)),
            "ordered_source_row_indexes_sha256": canonical_sha256(
                list(source.source_row_indexes)
            ),
            "ordered_prompt_row_receipts_sha256": canonical_sha256(list(source.row_receipts)),
            "prompt_population_sha256": source.prompt_population.prompt_population_sha256,
        },
        "ordered_rows": row_bindings,
        "execution": {
            "required_provider_calls": source.prompt_population.unique_prompt_count,
            "call_count_basis": "unique-sealed-provider-input-messages",
            "physical_provider_calls": 0,
            "provider_execution_available": False,
            "authorization_released": False,
            "retry_count": 0,
        },
    }
    try:
        assert_gold_blind(body, path="confirmation_terra_lifecycle_preflight")
    except MatchedEvalContractError as exc:
        raise ConfirmationTerraLifecycleError(str(exc)) from exc
    return {**body, "lifecycle_preflight_identity_sha256": canonical_sha256(body)}


def publish_lifecycle_preflight(
    *,
    prompt_artifact_path: str | Path,
    expected_prompt_artifact_sha256: str,
    output_root: str | Path,
) -> tuple[SealedArtifact, bool]:
    source = verify_prompt_artifact(
        prompt_artifact_path,
        expected_sha256=expected_prompt_artifact_sha256,
    )
    payload = compile_lifecycle_preflight(source)
    return publish_sealed_artifact(Path(output_root) / PREFLIGHT_NAME, payload)


def _read_lifecycle_preflight(
    *,
    output_root: str | Path,
    expected_sha256: str,
    source: VerifiedPromptArtifact,
) -> SealedArtifact:
    artifact = read_sealed_artifact(
        Path(output_root) / PREFLIGHT_NAME,
        expected_sha256=expected_sha256,
        label="Terra lifecycle preflight",
    )
    _require(set(artifact.payload) == _LIFECYCLE_PREFLIGHT_KEYS, "lifecycle preflight schema changed")
    _require(
        artifact.payload == compile_lifecycle_preflight(source),
        "Terra lifecycle preflight differs from the sealed prompt artifact",
    )
    return artifact


def _canonical_root(path: str | Path) -> str:
    return Path(path).resolve().as_posix()


def _runtime(
    source: VerifiedPromptArtifact,
    preflight: SealedArtifact,
    *,
    output_root: str | Path,
    client: Any | None,
) -> FastCompletionRuntime:
    _require(
        preflight.payload == compile_lifecycle_preflight(source),
        "runtime inputs differ from the lifecycle preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=Path(output_root) / CHECKPOINT_DIR_NAME,
        prompt_population=source.prompts,
        model=source.runtime["model"],
        client=client,
        max_prompt_tokens=source.runtime["input_token_cap"],
        max_new_tokens=source.runtime["output_token_reserve"],
        max_concurrency=source.runtime["max_concurrency"],
        retries=0,
        benchmark_provenance={
            "format": FORMAT,
            "gold_loaded": False,
            "source_prompt_artifact_sha256": source.artifact.sha256,
            "source_prompt_artifact_identity_sha256": source.source_identity_sha256,
            "lifecycle_preflight_sha256": preflight.sha256,
            "output_root_sha256": canonical_sha256(
                {"canonical_root": _canonical_root(output_root)}
            ),
        },
    )


def _checkpoint_structure(
    output_root: str | Path,
    *,
    maximum: int,
) -> tuple[str, ...]:
    root = Path(output_root) / CHECKPOINT_DIR_NAME
    if not root.exists():
        return ()
    _require(root.is_dir() and not root.is_symlink(), "checkpoint root is absent or unsafe")
    requests: set[str] = set()
    responses: set[str] = set()
    for path in root.iterdir():
        _require(path.is_file() and not path.is_symlink(), "checkpoint root contains unsafe state")
        if path.name == ".fast-completion-journal.lock":
            continue
        match = _JOURNAL_FILENAME.fullmatch(path.name)
        _require(match is not None, "checkpoint root contains foreign journal state")
        assert match is not None
        (requests if match.group("kind") == "request" else responses).add(
            match.group("key")
        )
    _require(requests == responses, "checkpoint request/response pair is incomplete; unsafe retry forbidden")
    _require(len(requests) <= maximum, "checkpoint population exceeds the sealed prompt population")
    return tuple(sorted(requests))


def _authenticated_records(
    source: VerifiedPromptArtifact,
    preflight: SealedArtifact,
    *,
    output_root: str | Path,
) -> tuple[FastCompletionRecord, ...]:
    maximum = source.prompt_population.unique_prompt_count
    structural = _checkpoint_structure(output_root, maximum=maximum)
    if not structural:
        return ()
    runtime = _runtime(source, preflight, output_root=output_root, client=None)
    try:
        with runtime._journal_guard():  # noqa: SLF001 - runtime owns journal authentication
            records = runtime._load_all_records()  # noqa: SLF001
    finally:
        runtime.close()
    _require(len(records) == len(structural), "authenticated checkpoint count differs from journal structure")
    ordered_rows: list[FastCompletionRecord] = []
    seen: set[str] = set()
    for row in source.prompt_population.ordered_rows:
        if row.messages_sha256 in records and row.messages_sha256 not in seen:
            ordered_rows.append(records[row.messages_sha256])
            seen.add(row.messages_sha256)
    ordered = tuple(ordered_rows)
    _require(len(ordered) == len(records), "authenticated checkpoint identities repeat")
    return ordered


def _record_binding(record: FastCompletionRecord) -> dict[str, Any]:
    return {
        "messages_sha256": record.messages_sha256,
        "call_key_sha256": record.call_key_sha256,
        "request_journal_sha256": record.request_journal_sha256,
        "response_journal_sha256": record.response_journal_sha256,
    }


def _require_release_record_subset(
    released_records: Sequence[Mapping[str, Any]],
    current_records: Sequence[FastCompletionRecord],
) -> None:
    current = {
        record.messages_sha256: _record_binding(record)
        for record in current_records
    }
    _require(
        len(current) == len(current_records),
        "checkpoint message identities repeat",
    )
    for row in released_records:
        _require(
            current.get(str(row["messages_sha256"])) == dict(row),
            "a checkpoint authenticated at release changed or disappeared",
        )


def approve_provider_release(
    *,
    prompt_artifact_path: str | Path,
    expected_prompt_artifact_sha256: str,
    output_root: str | Path,
    expected_lifecycle_preflight_sha256: str,
    approve_provider_release: bool,
    authorized_provider_calls: int,
) -> tuple[SealedArtifact, bool]:
    _require(approve_provider_release is True, "provider release requires explicit approval")
    source = verify_prompt_artifact(prompt_artifact_path, expected_sha256=expected_prompt_artifact_sha256)
    preflight = _read_lifecycle_preflight(
        output_root=output_root,
        expected_sha256=expected_lifecycle_preflight_sha256,
        source=source,
    )
    records = _authenticated_records(source, preflight, output_root=output_root)
    unique = source.prompt_population.unique_prompt_count
    remaining = unique - len(records)
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == remaining,
        "release authorization must exactly equal remaining unique Terra calls",
    )
    checkpoint_rows = [_record_binding(record) for record in records]
    canonical_root = _canonical_root(output_root)
    body: dict[str, Any] = {
        "format": RELEASE_FORMAT,
        "release_status": "approved_for_provider_execution",
        "approval_opt_in": True,
        "gold_loaded": False,
        "source_prompt_artifact_sha256": source.artifact.sha256,
        "lifecycle_preflight_sha256": preflight.sha256,
        "runtime": dict(source.runtime),
        "population": dict(preflight.payload["population"]),
        "output_root": canonical_root,
        "output_root_sha256": canonical_sha256({"canonical_root": canonical_root}),
        "checkpoint_snapshot": {
            "authenticated_complete_count": len(checkpoint_rows),
            "ordered_records": checkpoint_rows,
            "ordered_records_sha256": canonical_sha256(checkpoint_rows),
        },
        "required_authorized_provider_calls": remaining,
        "unsafe_retry_policy": "refuse-incomplete-request-response-pair-v1",
        "provider_calls_during_release": 0,
    }
    try:
        assert_gold_blind(body, path="confirmation_terra_provider_release")
    except MatchedEvalContractError as exc:
        raise ConfirmationTerraLifecycleError(str(exc)) from exc
    payload = {**body, "release_identity_sha256": canonical_sha256(body)}
    return publish_sealed_artifact(Path(output_root) / RELEASE_NAME, payload)


def _validate_release(
    release: SealedArtifact,
    *,
    source: VerifiedPromptArtifact,
    preflight: SealedArtifact,
    output_root: str | Path,
) -> tuple[dict[str, Any], ...]:
    value = release.payload
    _require(set(value) == _RELEASE_KEYS, "provider release schema changed")
    _require(_self_seal(value, "release_identity_sha256", "provider release"), "provider release is not sealed")
    root = _canonical_root(output_root)
    _require(
        value.get("format") == RELEASE_FORMAT
        and value.get("release_status") == "approved_for_provider_execution"
        and value.get("approval_opt_in") is True
        and value.get("gold_loaded") is False
        and value.get("source_prompt_artifact_sha256") == source.artifact.sha256
        and value.get("lifecycle_preflight_sha256") == preflight.sha256
        and value.get("runtime") == source.runtime
        and value.get("population") == preflight.payload["population"]
        and value.get("output_root") == root
        and value.get("output_root_sha256") == canonical_sha256({"canonical_root": root})
        and value.get("unsafe_retry_policy") == "refuse-incomplete-request-response-pair-v1"
        and value.get("provider_calls_during_release") == 0,
        "provider release bindings changed",
    )
    snapshot = require_mapping(value.get("checkpoint_snapshot"), "release checkpoint snapshot")
    exact_keys(
        snapshot,
        {"authenticated_complete_count", "ordered_records", "ordered_records_sha256"},
        "release checkpoint snapshot",
    )
    raw_rows = require_list(snapshot["ordered_records"], "release checkpoint records")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        row = require_mapping(raw, f"release checkpoint record {index}")
        exact_keys(
            row,
            {"messages_sha256", "call_key_sha256", "request_journal_sha256", "response_journal_sha256"},
            f"release checkpoint record {index}",
        )
        for key, child in row.items():
            require_sha256(child, f"release checkpoint record {index} {key}")
        rows.append(dict(row))
    unique = source.prompt_population.unique_prompt_count
    _require(
        snapshot["authenticated_complete_count"] == len(rows)
        and snapshot["ordered_records_sha256"] == canonical_sha256(rows)
        and len(rows) <= unique
        and value.get("required_authorized_provider_calls") == unique - len(rows),
        "provider release call accounting changed",
    )
    try:
        assert_gold_blind(value, path="confirmation_terra_provider_release")
    except MatchedEvalContractError as exc:
        raise ConfirmationTerraLifecycleError(str(exc)) from exc
    return tuple(rows)


def _read_release(
    *,
    output_root: str | Path,
    expected_sha256: str,
    source: VerifiedPromptArtifact,
    preflight: SealedArtifact,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    release = read_sealed_artifact(
        Path(output_root) / RELEASE_NAME,
        expected_sha256=expected_sha256,
        label="Terra provider release",
    )
    rows = _validate_release(
        release,
        source=source,
        preflight=preflight,
        output_root=output_root,
    )
    return release, rows


def _default_client_factory(gateway_url: str, api_key_env: str) -> Any:
    # Credentials are loaded only when a released provider phase actually
    # constructs its client.  Never load or inspect .env during readiness,
    # treatment opening, provider-free preparation, status, or replay.
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - locked production dependency
        raise ConfirmationTerraLifecycleError(
            "python-dotenv is required for provider execution"
        ) from exc
    load_dotenv(override=False)
    api_key = os.environ.get(api_key_env, "").strip()
    _require(bool(api_key), f"provider API key is empty: {api_key_env}")
    # Keep SDK/provider construction behind every sealed authorization gate.
    from tools.matched_eval.live import _make_provider_client  # noqa: PLC0415

    return _make_provider_client(api_key, gateway_url)


def _load_execution_inputs(
    *,
    prompt_artifact_path: str | Path,
    expected_prompt_artifact_sha256: str,
    output_root: str | Path,
    expected_lifecycle_preflight_sha256: str,
    expected_release_sha256: str,
) -> tuple[VerifiedPromptArtifact, SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    source = verify_prompt_artifact(prompt_artifact_path, expected_sha256=expected_prompt_artifact_sha256)
    preflight = _read_lifecycle_preflight(
        output_root=output_root,
        expected_sha256=expected_lifecycle_preflight_sha256,
        source=source,
    )
    release, release_records = _read_release(
        output_root=output_root,
        expected_sha256=expected_release_sha256,
        source=source,
        preflight=preflight,
    )
    return source, preflight, release, release_records


def run_provider_completion(
    *,
    prompt_artifact_path: str | Path,
    expected_prompt_artifact_sha256: str,
    output_root: str | Path,
    expected_lifecycle_preflight_sha256: str,
    expected_release_sha256: str,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    client_factory: ClientFactory = _default_client_factory,
) -> dict[str, Any]:
    source, preflight, release, released_records = _load_execution_inputs(
        prompt_artifact_path=prompt_artifact_path,
        expected_prompt_artifact_sha256=expected_prompt_artifact_sha256,
        output_root=output_root,
        expected_lifecycle_preflight_sha256=expected_lifecycle_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    _require(enable_provider is True, "provider execution requires explicit opt-in")
    maximum = source.prompt_population.unique_prompt_count
    structural = _checkpoint_structure(output_root, maximum=maximum)
    records = _authenticated_records(source, preflight, output_root=output_root)
    _require(len(records) == len(structural), "checkpoint population changed during authorization")
    _require_release_record_subset(released_records, records)
    remaining = maximum - len(records)
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == remaining,
        "provider authorization must exactly equal remaining unique Terra calls",
    )
    _require(
        maximum - len(released_records)
        == release.payload["required_authorized_provider_calls"]
        and remaining <= release.payload["required_authorized_provider_calls"],
        "current checkpoint state exceeds the sealed release budget",
    )

    client = None
    if remaining:
        client = client_factory(source.runtime["gateway_url"], api_key_env)
    runtime = _runtime(source, preflight, output_root=output_root, client=client)
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.logical_calls == len(source.rows)
        and batch.usage.unique_calls == maximum
        and batch.usage.physical_calls == remaining
        and batch.usage.checkpoint_hits == len(records)
        and batch.usage.physical_calls + batch.usage.checkpoint_hits == maximum,
        "provider call accounting differs from exact authorization",
    )
    return {
        "authorized_remaining_provider_calls": remaining,
        "checkpoint_hits_before_run": len(records),
        "checkpoint_hits_after_run": maximum,
        "logical_prompt_count": len(source.rows),
        "unique_prompt_count": maximum,
        "physical_provider_calls": batch.usage.physical_calls,
        "prompt_artifact_sha256": source.artifact.sha256,
        "lifecycle_preflight_sha256": preflight.sha256,
        "provider_release_sha256": release.sha256,
        "retry_count": 0,
    }


def _complete_checkpoint_batch(
    source: VerifiedPromptArtifact,
    preflight: SealedArtifact,
    *,
    output_root: str | Path,
) -> FastCompletionBatch:
    maximum = source.prompt_population.unique_prompt_count
    _require(
        len(_checkpoint_structure(output_root, maximum=maximum)) == maximum,
        "materialization requires a complete checkpoint population",
    )
    runtime = _runtime(source, preflight, output_root=output_root, client=None)
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.logical_calls == len(source.rows)
        and batch.usage.unique_calls == batch.usage.checkpoint_hits == maximum
        and batch.usage.physical_calls == 0,
        "materialization is not a complete checkpoint-only replay",
    )
    return batch


def _completion_payload(
    source: VerifiedPromptArtifact,
    preflight: SealedArtifact,
    release: SealedArtifact,
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    records = {record.messages_sha256: record for record in batch.unique_records}
    _require(
        len(records) == source.prompt_population.unique_prompt_count,
        "completion record population differs",
    )
    rows: list[dict[str, Any]] = []
    for index, (
        source_row_index,
        question_id,
        source_receipt,
        prompt_row,
        completion,
    ) in enumerate(
        zip(
            source.source_row_indexes,
            source.question_ids,
            source.row_receipts,
            source.prompt_population.ordered_rows,
            batch.logical_completions,
            strict=True,
        )
    ):
        record = records.get(prompt_row.messages_sha256)
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            f"completion checkpoint binding differs at row {index}",
        )
        assert record is not None
        body = {
            "format": COMPLETION_ROW_FORMAT,
            "row_index": index,
            "source_prompt_row_index": source_row_index,
            "question_id": question_id,
            "source_prompt_row_receipt_sha256": source_receipt,
            "messages_sha256": prompt_row.messages_sha256,
            "completion": completion,
            "completion_sha256": quote_sha256(completion),
            "call_key_sha256": record.call_key_sha256,
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
        }
        rows.append({**body, "completion_row_receipt_sha256": canonical_sha256(body)})
    body = {
        "format": COMPLETION_FORMAT,
        "status": "complete",
        "gold_loaded": False,
        "source_prompt_artifact_sha256": source.artifact.sha256,
        "lifecycle_preflight_sha256": preflight.sha256,
        "provider_release_sha256": release.sha256,
        "runtime": dict(source.runtime),
        "population": {
            **dict(preflight.payload["population"]),
            "question_count": len(source.rows),
        },
        "ordered_rows": rows,
        "completion_batch": batch.model_dump(),
        "physical_provider_calls_during_materialization": 0,
    }
    try:
        assert_gold_blind(body, path="confirmation_terra_completions")
    except MatchedEvalContractError as exc:
        raise ConfirmationTerraLifecycleError(str(exc)) from exc
    return {**body, "completion_artifact_identity_sha256": canonical_sha256(body)}


def materialize_completions(
    *,
    prompt_artifact_path: str | Path,
    expected_prompt_artifact_sha256: str,
    output_root: str | Path,
    expected_lifecycle_preflight_sha256: str,
    expected_release_sha256: str,
) -> tuple[SealedArtifact, bool]:
    source, preflight, release, released_records = _load_execution_inputs(
        prompt_artifact_path=prompt_artifact_path,
        expected_prompt_artifact_sha256=expected_prompt_artifact_sha256,
        output_root=output_root,
        expected_lifecycle_preflight_sha256=expected_lifecycle_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    batch = _complete_checkpoint_batch(source, preflight, output_root=output_root)
    _require_release_record_subset(released_records, batch.unique_records)
    payload = _completion_payload(source, preflight, release, batch)
    return publish_sealed_artifact(Path(output_root) / COMPLETION_NAME, payload)


def load_completed_batch(
    *,
    prompt_artifact_path: str | Path,
    expected_prompt_artifact_sha256: str,
    output_root: str | Path,
    expected_lifecycle_preflight_sha256: str,
    expected_release_sha256: str,
    expected_completion_sha256: str,
) -> FastCompletionBatch:
    """Authenticate and return a complete client-free checkpoint batch.

    This is the provider-neutral handoff for intermediate materializers.  It
    reconstitutes the batch from immutable request/response journals, binds it
    to the exact prompt/preflight/release, and requires the sealed completion
    artifact to equal that checkpoint-only reconstruction.
    """

    source, preflight, release, released_records = _load_execution_inputs(
        prompt_artifact_path=prompt_artifact_path,
        expected_prompt_artifact_sha256=expected_prompt_artifact_sha256,
        output_root=output_root,
        expected_lifecycle_preflight_sha256=expected_lifecycle_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    batch = _complete_checkpoint_batch(source, preflight, output_root=output_root)
    _require_release_record_subset(released_records, batch.unique_records)
    expected_payload = _completion_payload(source, preflight, release, batch)
    completion = read_sealed_artifact(
        Path(output_root) / COMPLETION_NAME,
        expected_sha256=expected_completion_sha256,
        label="Terra completion artifact",
    )
    _validate_completion(completion, expected_payload=expected_payload)
    return batch


def _validate_completion(
    artifact: SealedArtifact,
    *,
    expected_payload: Mapping[str, Any],
) -> None:
    value = artifact.payload
    _require(set(value) == _COMPLETION_KEYS, "completion artifact schema changed")
    _require(value == expected_payload, "completion artifact differs from checkpoint-only reconstruction")
    _require(_self_seal(value, "completion_artifact_identity_sha256", "completion artifact"), "completion artifact is not sealed")
    rows = require_list(value["ordered_rows"], "completion rows")
    for index, raw in enumerate(rows):
        row = require_mapping(raw, f"completion row {index}")
        exact_keys(row, _COMPLETION_ROW_KEYS, f"completion row {index}")
        _require(row["row_index"] == index, f"completion row {index} position changed")
        _require(_self_seal(row, "completion_row_receipt_sha256", f"completion row {index}"), f"completion row {index} is not sealed")
        _require(
            quote_sha256(require_text(row["completion"], f"completion row {index} text"))
            == require_sha256(row["completion_sha256"], f"completion row {index} text SHA-256"),
            f"completion row {index} text identity differs",
        )


def replay_completions(
    *,
    prompt_artifact_path: str | Path,
    expected_prompt_artifact_sha256: str,
    output_root: str | Path,
    expected_lifecycle_preflight_sha256: str,
    expected_release_sha256: str,
    expected_completion_sha256: str,
) -> tuple[SealedArtifact, bool]:
    source, preflight, release, released_records = _load_execution_inputs(
        prompt_artifact_path=prompt_artifact_path,
        expected_prompt_artifact_sha256=expected_prompt_artifact_sha256,
        output_root=output_root,
        expected_lifecycle_preflight_sha256=expected_lifecycle_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    batch = _complete_checkpoint_batch(source, preflight, output_root=output_root)
    _require_release_record_subset(released_records, batch.unique_records)
    expected_payload = _completion_payload(source, preflight, release, batch)
    completion = read_sealed_artifact(
        Path(output_root) / COMPLETION_NAME,
        expected_sha256=expected_completion_sha256,
        label="Terra completion artifact",
    )
    _validate_completion(completion, expected_payload=expected_payload)
    replay, created = publish_sealed_artifact(Path(output_root) / REPLAY_NAME, expected_payload)
    _require(replay.sha256 == completion.sha256, "completion replay is not byte-identical")
    return replay, created


def _common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt-artifact", type=Path, required=True)
    parser.add_argument("--expected-prompt-artifact-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def _execution_inputs(parser: argparse.ArgumentParser) -> None:
    _common_inputs(parser)
    parser.add_argument("--expected-lifecycle-preflight-sha256", required=True)
    parser.add_argument("--expected-release-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="verify and bind sealed Terra prompts")
    _common_inputs(preflight)
    release = commands.add_parser("approve-release", help="seal exact remaining-call authority")
    _common_inputs(release)
    release.add_argument("--expected-lifecycle-preflight-sha256", required=True)
    release.add_argument("--approve-provider-release", action="store_true")
    release.add_argument("--authorized-provider-calls", type=int, required=True)
    provider = commands.add_parser("provider-run", help="execute exactly authorized Terra misses")
    _execution_inputs(provider)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    materialize = commands.add_parser("materialize", help="seal checkpoint-only completions")
    _execution_inputs(materialize)
    replay = commands.add_parser("replay", help="rebuild byte-identical completions")
    _execution_inputs(replay)
    replay.add_argument("--expected-completion-sha256", required=True)
    return parser


def _common_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "prompt_artifact_path": args.prompt_artifact,
        "expected_prompt_artifact_sha256": args.expected_prompt_artifact_sha256,
        "output_root": args.output_root,
    }


def _execution_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **_common_kwargs(args),
        "expected_lifecycle_preflight_sha256": args.expected_lifecycle_preflight_sha256,
        "expected_release_sha256": args.expected_release_sha256,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "preflight":
        artifact, created = publish_lifecycle_preflight(**_common_kwargs(args))
        return {
            "created": created,
            "lifecycle_preflight_sha256": artifact.sha256,
            "logical_prompt_count": artifact.payload["population"]["logical_prompt_count"],
            "unique_prompt_count": artifact.payload["population"]["unique_prompt_count"],
            "physical_provider_calls": 0,
        }
    if args.command == "approve-release":
        artifact, created = approve_provider_release(
            **_common_kwargs(args),
            expected_lifecycle_preflight_sha256=args.expected_lifecycle_preflight_sha256,
            approve_provider_release=args.approve_provider_release,
            authorized_provider_calls=args.authorized_provider_calls,
        )
        return {
            "created": created,
            "provider_release_sha256": artifact.sha256,
            "required_authorized_provider_calls": artifact.payload[
                "required_authorized_provider_calls"
            ],
            "physical_provider_calls": 0,
        }
    if args.command == "provider-run":
        return run_provider_completion(
            **_execution_kwargs(args),
            enable_provider=args.enable_provider,
            authorized_provider_calls=args.authorized_provider_calls,
            api_key_env=args.api_key_env,
        )
    if args.command == "materialize":
        artifact, created = materialize_completions(**_execution_kwargs(args))
        return {
            "completion_sha256": artifact.sha256,
            "created": created,
            "question_count": artifact.payload["population"]["question_count"],
            "physical_provider_calls": 0,
        }
    if args.command == "replay":
        artifact, created = replay_completions(
            **_execution_kwargs(args),
            expected_completion_sha256=args.expected_completion_sha256,
        )
        return {
            "byte_identical": True,
            "created": created,
            "completion_replay_sha256": artifact.sha256,
            "physical_provider_calls": 0,
        }
    raise ConfirmationTerraLifecycleError("unknown command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(build_parser().parse_args(argv))
    except (
        ConfirmationTerraLifecycleError,
        FirebreakError,
        MatchedEvalContractError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"confirmation Terra completion lifecycle failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "COMPLETION_FORMAT",
    "COMPLETION_NAME",
    "ConfirmationTerraLifecycleError",
    "INTERMEDIATE_PROMPT_FORMAT",
    "INTERMEDIATE_PROMPT_ROW_FORMAT",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "PROVIDER_INPUT_FORMAT",
    "RELEASE_FORMAT",
    "RELEASE_NAME",
    "REPLAY_NAME",
    "S0_PROMPT_FORMAT",
    "SealedArtifact",
    "VerifiedPromptArtifact",
    "TERMINAL_PROMPT_FORMAT",
    "approve_provider_release",
    "build_parser",
    "compile_intermediate_prompt_artifact",
    "compile_lifecycle_preflight",
    "load_completed_batch",
    "main",
    "materialize_completions",
    "publish_lifecycle_preflight",
    "publish_sealed_artifact",
    "read_sealed_artifact",
    "replay_completions",
    "run_provider_completion",
    "verify_prompt_artifact",
]
