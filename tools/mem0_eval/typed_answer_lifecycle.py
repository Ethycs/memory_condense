"""Authenticated resumable Terra answer lifecycle for the Mem0 common-parent arm.

The common-input campaign owns retrieval adaptation and prompt fitting.  This
module seals that exact prompt population, delegates per-question checkpoint
semantics to :class:`FastCompletionRuntime`, and materializes/replays answers
without loading benchmark gold.  It owns no provider client construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.live import DEFAULT_GATEWAY_URL
from tools.matched_eval.typed_memory_final_arm import (
    HARD_PROMPT_TOKEN_CAP,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    RESULT_ROW_FORMAT,
    VALIDATOR_POLICY_FORMAT,
    judge_row_projection,
    materialize_typed_final_result_row,
)

from .typed_epoch_campaign import (
    COMPARISON_SEMANTICS,
    RESPONDER_MODEL,
    _validate_common_input,
    load_verified_common_input,
)


FORMAT = "memory-condense-mem0-common-parent-terra-answer-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
RUN_FORMAT = f"{FORMAT}-run-v1"
PREFLIGHT_NAME = "mem0-common-parent-terra-answer-preflight-v1.json"
RUN_NAME = "mem0-common-parent-terra-answer-run-v1.json"
REPLAY_NAME = "mem0-common-parent-terra-answer-replay-v1.json"
CHECKPOINT_DIR_NAME = "terra-mem0-common-parent-answer-v1-calls"


class Mem0TypedAnswerLifecycleError(MatchedEvalContractError):
    """A common input, Terra journal, answer row, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise Mem0TypedAnswerLifecycleError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _read_expected(
    path: str | Path,
    expected_sha256: str,
    label: str,
) -> SealedArtifact:
    expected = require_sha256(expected_sha256, f"expected {label}")
    artifact = read_sealed_json(path)
    _require(artifact.sha256 == expected, f"{label} SHA-256 changed")
    return artifact


def _prompt_row(
    source: Mapping[str, Any],
    *,
    common_input_sha256: str,
) -> dict[str, Any]:
    messages = _exact_list(source.get("messages"), "common Terra messages")
    body = {
        "allowed_handle_ids": list(
            _exact_list(source.get("allowed_handle_ids"), "allowed handles")
        ),
        "common_input_sha256": common_input_sha256,
        "common_row_sha256": require_sha256(
            source.get("common_row_sha256"), "common row"
        ),
        "dated_question_sha256": require_sha256(
            source.get("dated_question_sha256"), "dated question"
        ),
        "handle_group_by_id": dict(
            _exact_dict(source.get("handle_group_by_id"), "handle groups")
        ),
        "messages": [dict(_exact_dict(row, "common Terra message")) for row in messages],
        "messages_sha256": require_sha256(
            source.get("messages_sha256"), "common messages"
        ),
        "ordinal": source.get("ordinal"),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "parent_prediction": require_text(
            source.get("parent_prediction"), "common parent prediction"
        ),
        "parent_prediction_sha256": require_sha256(
            source.get("parent_prediction_sha256"), "common parent prediction"
        ),
        "parent_source_row_sha256": require_sha256(
            source.get("parent_source_row_sha256"), "common parent source row"
        ),
        "preservation_requirements": dict(
            _exact_dict(
                source.get("preservation_requirements"),
                "preservation requirements",
            )
        ),
        "prompt_token_proxy": source.get("prompt_token_proxy"),
        "question_id": require_text(source.get("question_id"), "common question ID"),
        "question_sha256": require_sha256(
            source.get("question_sha256"), "common question"
        ),
        "route_id": require_text(source.get("route_id"), "common route"),
        "story_coherence": dict(
            _exact_dict(source.get("story_coherence"), "story coherence")
        ),
        "validation_contract": dict(
            _exact_dict(source.get("validation_contract"), "validation contract")
        ),
    }
    _require(
        type(body["ordinal"]) is int
        and int(body["ordinal"]) >= 0
        and body["messages_sha256"] == identity_sha256(body["messages"])
        and body["parent_prediction_sha256"]
        == quote_sha256(body["parent_prediction"])
        and type(body["prompt_token_proxy"]) is int
        and count_chat_prompt_token_proxy(body["messages"])
        == body["prompt_token_proxy"]
        and body["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE
        <= HARD_PROMPT_TOKEN_CAP
        and set(body["handle_group_by_id"]) == set(body["allowed_handle_ids"]),
        "common Terra prompt row changed",
    )
    row = {**body, "prompt_row_receipt_sha256": identity_sha256(body)}
    assert_gold_blind(row, path="mem0_common_parent_answer_prompt")
    return row


def build_answer_preflight_payload(
    common_input: SealedArtifact,
    *,
    expected_question_count: int = 100,
    model: str = RESPONDER_MODEL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    """Seal every common-parent Terra prompt without creating checkpoints."""

    common = _validate_common_input(
        common_input.payload,
        expected_question_count=expected_question_count,
    )
    _require(
        model == common["model"] == RESPONDER_MODEL
        and gateway_url == DEFAULT_GATEWAY_URL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "Mem0 Terra runtime policy changed",
    )
    rows = tuple(
        _prompt_row(row, common_input_sha256=common_input.sha256)
        for row in common["questions"]
    )
    prompts = tuple(
        tuple(dict(message) for message in row["messages"]) for row in rows
    )
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == expected_question_count
        and all(
            receipt.messages_sha256 == row["messages_sha256"]
            and receipt.prompt_token_proxy == row["prompt_token_proxy"]
            for receipt, row in zip(population.ordered_rows, rows, strict=True)
        ),
        "Mem0 Terra prompt population changed or contains duplicates",
    )
    payload = {
        "common_input_sha256": common_input.sha256,
        "comparison_semantics": COMPARISON_SEMANTICS,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "max_chat_prompt_tokens": MAX_CHAT_PROMPT_TOKENS,
        "max_concurrency": max_concurrency,
        "model": RESPONDER_MODEL,
        "observed_max_complete_envelope_tokens": max(
            row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE for row in rows
        ),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "parent_origin_receipt_sha256": common[
            "parent_origin_receipt_sha256"
        ],
        "physical_provider_calls": 0,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "prompt_rows": list(rows),
        "question_count": expected_question_count,
        "required_authorized_provider_calls": expected_question_count,
        "retained_transformer_token_state_bytes": 0,
        "sdk_retries": 0,
    }
    assert_gold_blind(payload, path="mem0_common_parent_answer_preflight")
    return payload, prompts


def validate_answer_preflight_artifact(
    artifact: SealedArtifact,
    *,
    expected_question_count: int = 100,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    raw_rows = payload.get("prompt_rows")
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("comparison_semantics") == COMPARISON_SEMANTICS
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("sdk_retries") == 0
        and payload.get("hard_prompt_token_cap") == HARD_PROMPT_TOKEN_CAP
        and payload.get("max_chat_prompt_tokens") == MAX_CHAT_PROMPT_TOKENS
        and payload.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and payload.get("model") == RESPONDER_MODEL
        and payload.get("gateway_url") == DEFAULT_GATEWAY_URL
        and type(payload.get("max_concurrency")) is int
        and int(payload["max_concurrency"]) > 0
        and payload.get("question_count") == expected_question_count
        and payload.get("required_authorized_provider_calls")
        == expected_question_count
        and type(raw_rows) is list
        and len(raw_rows) == expected_question_count,
        "Mem0 Terra sealed preflight changed",
    )
    require_sha256(payload.get("common_input_sha256"), "common input")
    require_sha256(payload.get("parent_origin_receipt_sha256"), "parent origin")
    rows: list[dict[str, Any]] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    for ordinal, raw in enumerate(raw_rows):
        row = _exact_dict(raw, "Mem0 Terra prompt row")
        body = dict(row)
        declared = body.pop("prompt_row_receipt_sha256", None)
        messages = _exact_list(row.get("messages"), "Mem0 Terra messages")
        plain = tuple(dict(_exact_dict(message, "Mem0 Terra message")) for message in messages)
        _require(
            declared == identity_sha256(body)
            and row.get("ordinal") == ordinal
            and row.get("common_input_sha256")
            == payload["common_input_sha256"]
            and row.get("messages_sha256") == identity_sha256(list(plain))
            and row.get("prompt_token_proxy")
            == count_chat_prompt_token_proxy(plain)
            and int(row["prompt_token_proxy"]) <= MAX_CHAT_PROMPT_TOKENS
            and row.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
            and quote_sha256(
                require_text(row.get("parent_prediction"), "Mem0 parent")
            )
            == row.get("parent_prediction_sha256"),
            f"Mem0 Terra prompt row {ordinal} changed",
        )
        prompts.append(plain)
        rows.append(dict(row))
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
    )
    _require(
        population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.unique_prompt_count == expected_question_count
        and max(
            row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE for row in rows
        )
        == payload.get("observed_max_complete_envelope_tokens")
        <= HARD_PROMPT_TOKEN_CAP,
        "Mem0 Terra sealed prompt population changed",
    )
    assert_gold_blind(payload, path="mem0_common_parent_answer_preflight")
    return tuple(prompts), tuple(rows)


def load_verified_answer_preflight(
    preflight_path: str | Path,
    expected_preflight_sha256: str,
    *,
    expected_question_count: int = 100,
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    artifact = _read_expected(
        preflight_path,
        expected_preflight_sha256,
        "Mem0 Terra answer preflight",
    )
    prompts, rows = validate_answer_preflight_artifact(
        artifact,
        expected_question_count=expected_question_count,
    )
    return artifact, prompts, rows


def build_answer_runtime(
    preflight: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    checkpoint_dir: str | Path,
    client: Any | None,
    model: str = RESPONDER_MODEL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
    expected_question_count: int = 100,
) -> FastCompletionRuntime:
    validated_prompts, _ = validate_answer_preflight_artifact(
        preflight,
        expected_question_count=expected_question_count,
    )
    plain = tuple(tuple(dict(row) for row in prompt) for prompt in prompts)
    _require(
        plain == validated_prompts
        and model == preflight.payload.get("model") == RESPONDER_MODEL
        and gateway_url
        == preflight.payload.get("gateway_url")
        == DEFAULT_GATEWAY_URL
        and max_concurrency == preflight.payload.get("max_concurrency"),
        "Mem0 Terra runtime differs from sealed preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=plain,
        model=RESPONDER_MODEL,
        client=client,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "arm": "mem0_common_parent",
            "authorized_unique_calls": expected_question_count,
            "common_input_sha256": preflight.payload["common_input_sha256"],
            "comparison_semantics": COMPARISON_SEMANTICS,
            "experiment_format": RUN_FORMAT,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_loaded": False,
            "preflight_artifact_sha256": preflight.sha256,
        },
    )


def run_answer_checkpoint_batch(
    preflight: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    checkpoint_dir: str | Path,
    client: Any | None,
    model: str = RESPONDER_MODEL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
    expected_question_count: int = 100,
) -> FastCompletionBatch:
    """Run exact remaining calls, or replay all completed checkpoints."""

    runtime = build_answer_runtime(
        preflight,
        prompts,
        checkpoint_dir=checkpoint_dir,
        client=client,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
        expected_question_count=expected_question_count,
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


def materialize_answer_payload(
    preflight: SealedArtifact,
    prompt_rows: Sequence[Mapping[str, Any]],
    batch: FastCompletionBatch,
    *,
    expected_question_count: int = 100,
) -> dict[str, Any]:
    _require(
        type(batch) is FastCompletionBatch
        and batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == expected_question_count
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == expected_question_count
        and len(batch.unique_records) == expected_question_count,
        "Mem0 Terra materialization requires complete checkpoint hits",
    )
    records = {record.messages_sha256: record for record in batch.unique_records}
    _require(
        len(records) == expected_question_count,
        "Mem0 Terra completion identities repeat",
    )
    results: list[dict[str, Any]] = []
    for plan, completion in zip(prompt_rows, batch.logical_completions, strict=True):
        record = records.get(plan["messages_sha256"])
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            "Mem0 Terra checkpoint record changed",
        )
        assert record is not None
        results.append(
            materialize_typed_final_result_row(
                plan,
                completion,
                completion_receipt_sha256=record.completion_sha256,
                call_key_sha256=record.call_key_sha256,
                request_journal_sha256=record.request_journal_sha256,
                response_journal_sha256=record.response_journal_sha256,
            )
        )
    judge_rows = [judge_row_projection(row) for row in results]
    payload = {
        "changed_prediction_count": sum(row["changed_from_parent"] for row in results),
        "common_input_sha256": preflight.payload["common_input_sha256"],
        "comparison_semantics": COMPARISON_SEMANTICS,
        "completion_batch": batch.model_dump(),
        "format": RUN_FORMAT,
        "gold_loaded": False,
        "judge_rows": judge_rows,
        "parent_origin_receipt_sha256": preflight.payload[
            "parent_origin_receipt_sha256"
        ],
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": expected_question_count,
        "questions": results,
        "required_authorized_provider_calls": expected_question_count,
        "retained_transformer_token_state_bytes": 0,
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    assert_gold_blind(payload, path="mem0_common_parent_answer_run")
    return payload


def validate_answer_run(
    artifact: SealedArtifact,
    *,
    expected_preflight_sha256: str,
    expected_question_count: int = 100,
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    questions = payload.get("questions")
    judge_rows = payload.get("judge_rows")
    _require(
        payload.get("format") == RUN_FORMAT
        and payload.get("comparison_semantics") == COMPARISON_SEMANTICS
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("preflight_artifact_sha256")
        == require_sha256(expected_preflight_sha256, "Mem0 Terra preflight")
        and payload.get("question_count") == expected_question_count
        and payload.get("required_authorized_provider_calls")
        == expected_question_count
        and type(questions) is list
        and type(judge_rows) is list
        and len(questions) == len(judge_rows) == expected_question_count,
        "Mem0 Terra answer run envelope changed",
    )
    validated: list[dict[str, Any]] = []
    question_ids: list[str] = []
    for ordinal, (source, projected) in enumerate(
        zip(questions, judge_rows, strict=True)
    ):
        row = _exact_dict(source, "Mem0 Terra result row")
        unsigned = dict(row)
        declared = unsigned.pop("source_row_sha256", None)
        prediction = require_text(row.get("prediction"), "Mem0 Terra prediction")
        _require(
            row.get("format") == RESULT_ROW_FORMAT
            and row.get("ordinal") == ordinal
            and declared == identity_sha256(unsigned)
            and row.get("prediction_sha256") == quote_sha256(prediction)
            and row.get("retained_transformer_token_state_bytes") == 0
            and judge_row_projection(row) == projected,
            f"Mem0 Terra result row {ordinal} changed",
        )
        question_ids.append(require_text(row.get("question_id"), "Mem0 question ID"))
        validated.append(dict(projected))
    _require(
        len(set(question_ids)) == expected_question_count,
        "Mem0 Terra answer question identities repeat",
    )
    assert_gold_blind(payload, path="mem0_common_parent_answer_run")
    return tuple(validated)


def materialize_answer_from_checkpoints(
    *,
    preflight_path: str | Path,
    expected_preflight_sha256: str,
    output_root: str | Path,
    expected_question_count: int = 100,
    model: str = RESPONDER_MODEL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
) -> SealedArtifact:
    preflight, prompts, rows = load_verified_answer_preflight(
        preflight_path,
        expected_preflight_sha256,
        expected_question_count=expected_question_count,
    )
    batch = run_answer_checkpoint_batch(
        preflight,
        prompts,
        checkpoint_dir=Path(output_root) / CHECKPOINT_DIR_NAME,
        client=None,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
        expected_question_count=expected_question_count,
    )
    payload = materialize_answer_payload(
        preflight,
        rows,
        batch,
        expected_question_count=expected_question_count,
    )
    artifact, _ = publish_sealed_json(Path(output_root) / RUN_NAME, payload)
    return artifact


def replay_answer_from_checkpoints(
    *,
    preflight_path: str | Path,
    expected_preflight_sha256: str,
    run_path: str | Path,
    expected_run_sha256: str,
    output_root: str | Path,
    expected_question_count: int = 100,
    model: str = RESPONDER_MODEL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
) -> SealedArtifact:
    run = _read_expected(run_path, expected_run_sha256, "Mem0 Terra answer run")
    validate_answer_run(
        run,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_question_count=expected_question_count,
    )
    rebuilt = materialize_answer_from_checkpoints(
        preflight_path=preflight_path,
        expected_preflight_sha256=expected_preflight_sha256,
        output_root=output_root,
        expected_question_count=expected_question_count,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
    )
    _require(
        rebuilt.sha256 == run.sha256 and rebuilt.payload == run.payload,
        "Mem0 Terra answer is not byte-identical on checkpoint replay",
    )
    replay, _ = publish_sealed_json(Path(output_root) / REPLAY_NAME, run.payload)
    _require(replay.sha256 == run.sha256, "Mem0 Terra replay publication changed")
    return replay


def load_verified_answer_run(
    output_root: str | Path,
    *,
    common_input_path: str | Path,
    expected_common_input_sha256: str,
    expected_preflight_sha256: str,
    expected_run_sha256: str,
    expected_replay_sha256: str,
    expected_question_count: int = 100,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    root = Path(output_root)
    common, _common_rows = load_verified_common_input(
        common_input_path,
        expected_common_input_sha256,
        expected_question_count=expected_question_count,
    )
    preflight = _read_expected(
        root / PREFLIGHT_NAME,
        expected_preflight_sha256,
        "Mem0 Terra preflight",
    )
    prompts, prompt_rows = validate_answer_preflight_artifact(
        preflight,
        expected_question_count=expected_question_count,
    )
    rebuilt_preflight, rebuilt_prompts = build_answer_preflight_payload(
        common,
        expected_question_count=expected_question_count,
        model=RESPONDER_MODEL,
        gateway_url=DEFAULT_GATEWAY_URL,
        max_concurrency=int(preflight.payload["max_concurrency"]),
    )
    _require(
        common.sha256
        == require_sha256(expected_common_input_sha256, "Mem0 common input")
        == preflight.payload.get("common_input_sha256")
        and rebuilt_preflight == preflight.payload
        and rebuilt_prompts == prompts,
        "Mem0 Terra preflight is not the exact projection of its common input",
    )
    run = _read_expected(root / RUN_NAME, expected_run_sha256, "Mem0 Terra run")
    replay = _read_expected(
        root / REPLAY_NAME,
        expected_replay_sha256,
        "Mem0 Terra replay",
    )
    _require(
        run.sha256 == replay.sha256 and run.payload == replay.payload,
        "Mem0 Terra run/replay are not byte-identical",
    )
    rows = validate_answer_run(
        run,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_question_count=expected_question_count,
    )
    checkpoint_root = root / CHECKPOINT_DIR_NAME
    _require(
        checkpoint_root.is_dir() and not checkpoint_root.is_symlink(),
        "Mem0 Terra checkpoint directory is missing or unsafe",
    )
    try:
        runtime = build_answer_runtime(
            preflight,
            prompts,
            checkpoint_dir=checkpoint_root,
            client=None,
            model=RESPONDER_MODEL,
            gateway_url=DEFAULT_GATEWAY_URL,
            max_concurrency=int(preflight.payload["max_concurrency"]),
            expected_question_count=expected_question_count,
        )
        try:
            authenticated_batch = runtime.run()
        finally:
            runtime.close()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Mem0TypedAnswerLifecycleError(
            "Mem0 Terra checkpoint journals do not authenticate"
        ) from exc
    _require(
        type(authenticated_batch) is FastCompletionBatch
        and authenticated_batch.model_dump()
        == run.payload.get("completion_batch"),
        "Mem0 Terra stored completion batch differs from checkpoint journals",
    )
    expected_journals = {
        name
        for record in authenticated_batch.unique_records
        for name in (
            f"{record.call_key_sha256}.request.json",
            f"{record.call_key_sha256}.response.json",
        )
    }
    journal_entries = tuple(
        entry
        for entry in checkpoint_root.iterdir()
        if entry.name != ".fast-completion-journal.lock"
    )
    _require(
        {entry.name for entry in journal_entries} == expected_journals
        and all(entry.is_file() and not entry.is_symlink() for entry in journal_entries),
        "Mem0 Terra checkpoint directory contains unbound entries",
    )
    rebuilt_run = materialize_answer_payload(
        preflight,
        prompt_rows,
        authenticated_batch,
        expected_question_count=expected_question_count,
    )
    _require(
        rebuilt_run == run.payload,
        "Mem0 Terra result rows are not the exact checkpoint/preflight projection",
    )
    batch = _exact_dict(run.payload.get("completion_batch"), "completion batch")
    usage = _exact_dict(batch.get("usage"), "completion usage")
    provenance = _exact_dict(batch.get("provenance"), "completion provenance")
    benchmark = _exact_dict(
        provenance.get("benchmark_provenance"),
        "completion benchmark provenance",
    )
    population = _exact_dict(
        batch.get("prompt_population"),
        "completion prompt population",
    )
    unique_records = _exact_list(
        batch.get("unique_records"),
        "completion unique records",
    )
    _require(
        run.payload.get("common_input_sha256")
        == preflight.payload.get("common_input_sha256")
        and run.payload.get("parent_origin_receipt_sha256")
        == preflight.payload.get("parent_origin_receipt_sha256")
        and usage.get("logical_calls") == expected_question_count
        and usage.get("unique_calls") == expected_question_count
        and usage.get("checkpoint_hits") == expected_question_count
        and usage.get("physical_calls") == 0
        and provenance.get("model") == RESPONDER_MODEL
        and provenance.get("max_new_tokens") == OUTPUT_TOKEN_RESERVE
        and provenance.get("max_prompt_token_proxy") == MAX_CHAT_PROMPT_TOKENS
        and provenance.get("retries") == 0
        and provenance.get("persisted_transformer_token_state") is False
        and provenance.get("retained_transformer_token_state_bytes") == 0
        and benchmark.get("comparison_semantics") == COMPARISON_SEMANTICS
        and benchmark.get("preflight_artifact_sha256") == preflight.sha256
        and benchmark.get("common_input_sha256")
        == preflight.payload.get("common_input_sha256")
        and batch.get("runtime_identity_sha256") == identity_sha256(provenance)
        and population == preflight.payload.get("prompt_population")
        and len(unique_records) == expected_question_count,
        "Mem0 Terra completion batch escaped its sealed runtime identity",
    )
    completion_by_response: dict[str, dict[str, Any]] = {}
    for raw in unique_records:
        record = _exact_dict(raw, "completion record")
        response_sha = require_sha256(
            record.get("response_journal_sha256"),
            "completion response journal",
        )
        _require(
            record.get("checkpoint_hit") is True
            and record.get("physical_call") is False
            and record.get("requested_model") == RESPONDER_MODEL
            and record.get("completion_sha256")
            == quote_sha256(
                require_text(record.get("completion"), "completion text")
            )
            and response_sha not in completion_by_response,
            "Mem0 Terra completion record changed",
        )
        completion_by_response[response_sha] = record
    _require(
        all(
            row["response_journal_sha256"] in completion_by_response
            and row["call_key_sha256"]
            == completion_by_response[row["response_journal_sha256"]][
                "call_key_sha256"
            ]
            for row in run.payload["questions"]
        ),
        "Mem0 Terra results escaped their checkpoint completions",
    )
    return run, replay, rows


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "FORMAT",
    "Mem0TypedAnswerLifecycleError",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "REPLAY_NAME",
    "RUN_FORMAT",
    "RUN_NAME",
    "build_answer_preflight_payload",
    "build_answer_runtime",
    "load_verified_answer_preflight",
    "load_verified_answer_run",
    "materialize_answer_from_checkpoints",
    "materialize_answer_payload",
    "replay_answer_from_checkpoints",
    "run_answer_checkpoint_batch",
    "validate_answer_preflight_artifact",
    "validate_answer_run",
]
