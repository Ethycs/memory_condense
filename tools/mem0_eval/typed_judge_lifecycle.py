"""Authenticated resumable Sol judging for the Mem0 common-parent arm.

Gold is joined only after the Terra answer run has passed its common-input,
preflight, checkpoint-journal, and byte-identical replay checks.  This module
does not construct a provider client.  Materialization and verification are
checkpoint-only and use the exact 8K complete-request envelope.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval._binary_judge_protocol import (
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import build_judge_prompt, exact_match, f1_score
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools._routed_repair_routing import route_question
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.live import DEFAULT_GATEWAY_URL
from tools.matched_eval.typed_memory_final_arm import (
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
)
from tools.matched_eval.typed_memory_final_judging import (
    TypedFinalJudgeGoldRow,
    load_locked_typed_final_gold,
)

from .typed_answer_lifecycle import (
    RUN_FORMAT as ANSWER_RUN_FORMAT,
    load_verified_answer_run,
    validate_answer_run,
)
from .typed_epoch_campaign import (
    COMPARISON_SEMANTICS,
    HARD_PROMPT_TOKEN_CAP,
    JUDGE_MODEL,
    JUDGE_OUTPUT_TOKEN_RESERVE,
    RESPONDER_MODEL,
)


FORMAT = "memory-condense-mem0-common-parent-sol-judge-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
JUDGE_FORMAT = f"{FORMAT}-run-v1"
SCORE_FORMAT = f"{FORMAT}-score-v1"
PREFLIGHT_NAME = "mem0-common-parent-sol-judge-preflight-v1.json"
JUDGE_NAME = "mem0-common-parent-sol-judge-run-v1.json"
JUDGE_REPLAY_NAME = "mem0-common-parent-sol-judge-replay-v1.json"
SCORE_NAME = "mem0-common-parent-sol-score-v1.json"
SCORE_REPLAY_NAME = "mem0-common-parent-sol-score-replay-v1.json"
CHECKPOINT_DIR_NAME = "sol-mem0-common-parent-judge-v1-calls"
MAX_JUDGE_PROMPT_TOKENS = HARD_PROMPT_TOKEN_CAP
MAX_JUDGE_COMPLETE_ENVELOPE_TOKENS = (
    MAX_JUDGE_PROMPT_TOKENS + JUDGE_OUTPUT_TOKEN_RESERVE
)

_PREFLIGHT_KEYS = {
    "answer_replay_sha256",
    "answer_run_sha256",
    "common_input_sha256",
    "comparison_semantics",
    "format",
    "gateway_url",
    "gold_loaded",
    "gold_population_sha256",
    "hard_prompt_token_cap",
    "max_concurrency",
    "max_judge_prompt_tokens",
    "max_judge_complete_envelope_tokens",
    "model",
    "observed_max_complete_envelope_tokens",
    "output_token_reserve",
    "parent_origin_receipt_sha256",
    "physical_provider_calls",
    "prompt_population",
    "prompt_population_sha256",
    "prompt_rows",
    "question_count",
    "required_authorized_provider_calls",
    "responder_model",
    "retained_transformer_token_state_bytes",
    "sdk_retries",
}
_PROMPT_ROW_KEYS = {
    "answer_prompt_row_receipt_sha256",
    "answer_source_row_sha256",
    "category",
    "dated_question_sha256",
    "demand_class",
    "messages",
    "messages_sha256",
    "ordinal",
    "output_token_reserve",
    "prediction",
    "prediction_sha256",
    "prediction_source",
    "prompt_row_receipt_sha256",
    "prompt_token_proxy",
    "question_id",
    "question_sha256",
    "reference",
    "reference_sha256",
    "route_id",
}


class Mem0TypedJudgeLifecycleError(MatchedEvalContractError):
    """A Sol prompt, journal, score, or replay escaped its sealed authority."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise Mem0TypedJudgeLifecycleError(message)


def _dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _read_expected(path: str | Path, expected_sha256: str, label: str) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, f"expected {label}"),
        f"{label} SHA-256 changed",
    )
    return artifact


def _validate_answer_source(
    run: SealedArtifact,
    replay: SealedArtifact,
    source_rows: Sequence[Mapping[str, Any]],
    *,
    expected_question_count: int,
) -> tuple[dict[str, Any], ...]:
    _require(
        run.sha256 == replay.sha256
        and run.payload == replay.payload
        and run.payload.get("format") == ANSWER_RUN_FORMAT
        and run.payload.get("comparison_semantics") == COMPARISON_SEMANTICS
        and run.payload.get("gold_loaded") is False,
        "Mem0 Sol source is not a byte-identical common-parent Terra run",
    )
    validated = validate_answer_run(
        run,
        expected_preflight_sha256=require_sha256(
            run.payload.get("preflight_artifact_sha256"), "Terra preflight"
        ),
        expected_question_count=expected_question_count,
    )
    rows = tuple(dict(row) for row in source_rows)
    _require(rows == validated, "Mem0 Sol source rows differ from the Terra seam")
    return rows


def build_judge_preflight_payload(
    *,
    answer_run: SealedArtifact,
    answer_replay: SealedArtifact,
    source_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[TypedFinalJudgeGoldRow],
    gold_population_sha256: str,
    expected_question_count: int = 100,
    model: str = JUDGE_MODEL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    """Build the exact full-population Sol preflight without making calls."""

    sources = _validate_answer_source(
        answer_run,
        answer_replay,
        source_rows,
        expected_question_count=expected_question_count,
    )
    gold = tuple(gold_rows)
    _require(
        len(sources) == len(gold) == expected_question_count
        and model == JUDGE_MODEL
        and gateway_url == DEFAULT_GATEWAY_URL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "Mem0 Sol runtime policy or population changed",
    )
    pending: list[dict[str, Any]] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    for ordinal, (source, target) in enumerate(zip(sources, gold, strict=True)):
        answer_source = _dict(
            answer_run.payload["questions"][ordinal],
            "Mem0 Terra answer source row",
        )
        _require(
            target.ordinal == ordinal == source.get("ordinal")
            and target.question_id == source.get("question_id")
            and target.question_sha256 == source.get("question_sha256")
            and target.dated_question_sha256
            == source.get("dated_question_sha256")
            and answer_source.get("source_row_sha256")
            == source.get("source_row_sha256"),
            f"Mem0 Sol source/gold binding changed at ordinal {ordinal}",
        )
        prediction = require_text(source.get("prediction"), "Mem0 prediction")
        messages = tuple(
            dict(row)
            for row in build_judge_prompt(
                target.question,
                target.reference,
                prediction,
            )
        )
        body = {
            "answer_prompt_row_receipt_sha256": require_sha256(
                answer_source.get("prompt_row_receipt_sha256"),
                "answer prompt row",
            ),
            "answer_source_row_sha256": require_sha256(
                source.get("source_row_sha256"), "answer source row"
            ),
            "category": require_text(target.category, "judge category"),
            "dated_question_sha256": target.dated_question_sha256,
            "demand_class": route_question(target.dated_question).style.value,
            "messages": list(messages),
            "messages_sha256": identity_sha256(list(messages)),
            "ordinal": ordinal,
            "output_token_reserve": JUDGE_OUTPUT_TOKEN_RESERVE,
            "prediction": prediction,
            "prediction_sha256": require_sha256(
                source.get("prediction_sha256"), "prediction"
            ),
            "prediction_source": require_text(
                source.get("prediction_source"), "prediction source"
            ),
            "question_id": target.question_id,
            "question_sha256": target.question_sha256,
            "reference": target.reference,
            "reference_sha256": target.reference_sha256,
            "route_id": require_text(source.get("route_id"), "answer route"),
        }
        _require(
            body["prediction_sha256"] == quote_sha256(prediction)
            and body["reference_sha256"] == quote_sha256(target.reference),
            f"Mem0 Sol text receipt changed at ordinal {ordinal}",
        )
        pending.append(body)
        prompts.append(messages)
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=MAX_JUDGE_PROMPT_TOKENS,
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == expected_question_count,
        "Mem0 Sol prompts are not one unique call per question",
    )
    rows: list[dict[str, Any]] = []
    for body, receipt in zip(pending, population.ordered_rows, strict=True):
        _require(
            body["messages_sha256"] == receipt.messages_sha256,
            "Mem0 Sol prompt receipt changed",
        )
        projected = {**body, "prompt_token_proxy": receipt.prompt_token_proxy}
        rows.append(
            {
                **projected,
                "prompt_row_receipt_sha256": identity_sha256(projected),
            }
        )
    payload = {
        "answer_replay_sha256": answer_replay.sha256,
        "answer_run_sha256": answer_run.sha256,
        "common_input_sha256": require_sha256(
            answer_run.payload.get("common_input_sha256"), "common input"
        ),
        "comparison_semantics": COMPARISON_SEMANTICS,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": DEFAULT_GATEWAY_URL,
        "gold_loaded": True,
        "gold_population_sha256": require_sha256(
            gold_population_sha256, "gold population"
        ),
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "max_concurrency": max_concurrency,
        "max_judge_complete_envelope_tokens": MAX_JUDGE_COMPLETE_ENVELOPE_TOKENS,
        "max_judge_prompt_tokens": MAX_JUDGE_PROMPT_TOKENS,
        "model": JUDGE_MODEL,
        "observed_max_complete_envelope_tokens": max(
            row["prompt_token_proxy"] + JUDGE_OUTPUT_TOKEN_RESERVE for row in rows
        ),
        "output_token_reserve": JUDGE_OUTPUT_TOKEN_RESERVE,
        "parent_origin_receipt_sha256": require_sha256(
            answer_run.payload.get("parent_origin_receipt_sha256"), "parent origin"
        ),
        "physical_provider_calls": 0,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "prompt_rows": rows,
        "question_count": expected_question_count,
        "required_authorized_provider_calls": expected_question_count,
        "responder_model": RESPONDER_MODEL,
        "retained_transformer_token_state_bytes": 0,
        "sdk_retries": 0,
    }
    return payload, tuple(prompts)


def validate_judge_preflight_artifact(
    artifact: SealedArtifact,
    *,
    expected_question_count: int = 100,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    raw_rows = payload.get("prompt_rows")
    _require(
        set(payload) == _PREFLIGHT_KEYS
        and payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("comparison_semantics") == COMPARISON_SEMANTICS
        and payload.get("gateway_url") == DEFAULT_GATEWAY_URL
        and payload.get("gold_loaded") is True
        and payload.get("hard_prompt_token_cap") == HARD_PROMPT_TOKEN_CAP
        and payload.get("max_judge_prompt_tokens") == MAX_JUDGE_PROMPT_TOKENS
        and payload.get("max_judge_complete_envelope_tokens")
        == MAX_JUDGE_COMPLETE_ENVELOPE_TOKENS
        and payload.get("model") == JUDGE_MODEL
        and payload.get("output_token_reserve") == JUDGE_OUTPUT_TOKEN_RESERVE
        and payload.get("responder_model") == RESPONDER_MODEL
        and payload.get("physical_provider_calls") == 0
        and payload.get("sdk_retries") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and type(payload.get("max_concurrency")) is int
        and int(payload["max_concurrency"]) > 0
        and payload.get("question_count") == expected_question_count
        and payload.get("required_authorized_provider_calls")
        == expected_question_count
        and type(raw_rows) is list
        and len(raw_rows) == expected_question_count,
        "Mem0 Sol sealed preflight changed",
    )
    for key in (
        "answer_replay_sha256",
        "answer_run_sha256",
        "common_input_sha256",
        "gold_population_sha256",
        "parent_origin_receipt_sha256",
        "prompt_population_sha256",
    ):
        require_sha256(payload.get(key), f"Mem0 Sol {key}")
    prompts: list[tuple[dict[str, str], ...]] = []
    rows: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_rows):
        row = _dict(raw, "Mem0 Sol prompt row")
        unsigned = dict(row)
        declared = unsigned.pop("prompt_row_receipt_sha256", None)
        messages = _list(row.get("messages"), "Mem0 Sol messages")
        plain = tuple(dict(_dict(message, "Mem0 Sol message")) for message in messages)
        _require(
            set(row) == _PROMPT_ROW_KEYS
            and declared == identity_sha256(unsigned)
            and row.get("ordinal") == ordinal
            and row.get("messages_sha256") == identity_sha256(list(plain))
            and row.get("prompt_token_proxy")
            == count_chat_prompt_token_proxy(plain)
            and type(row.get("prompt_token_proxy")) is int
            and int(row["prompt_token_proxy"]) <= MAX_JUDGE_PROMPT_TOKENS
            and int(row["prompt_token_proxy"]) + JUDGE_OUTPUT_TOKEN_RESERVE
            <= MAX_JUDGE_COMPLETE_ENVELOPE_TOKENS
            and row.get("output_token_reserve") == JUDGE_OUTPUT_TOKEN_RESERVE
            and row.get("prediction_sha256")
            == quote_sha256(require_text(row.get("prediction"), "prediction"))
            and row.get("reference_sha256")
            == quote_sha256(require_text(row.get("reference"), "reference")),
            f"Mem0 Sol prompt row {ordinal} changed",
        )
        prompts.append(plain)
        rows.append(dict(row))
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=MAX_JUDGE_PROMPT_TOKENS,
    )
    _require(
        population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.unique_prompt_count == expected_question_count
        and max(
            row["prompt_token_proxy"] + JUDGE_OUTPUT_TOKEN_RESERVE for row in rows
        )
        == payload.get("observed_max_complete_envelope_tokens")
        <= MAX_JUDGE_COMPLETE_ENVELOPE_TOKENS,
        "Mem0 Sol sealed prompt population changed",
    )
    return tuple(prompts), tuple(rows)


def build_judge_runtime(
    preflight: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    checkpoint_dir: str | Path,
    client: Any | None,
    model: str = JUDGE_MODEL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
    expected_question_count: int = 100,
) -> FastCompletionRuntime:
    validated, _rows = validate_judge_preflight_artifact(
        preflight,
        expected_question_count=expected_question_count,
    )
    plain = tuple(tuple(dict(message) for message in prompt) for prompt in prompts)
    _require(
        plain == validated
        and model == preflight.payload.get("model") == JUDGE_MODEL
        and gateway_url
        == preflight.payload.get("gateway_url")
        == DEFAULT_GATEWAY_URL
        and max_concurrency == preflight.payload.get("max_concurrency"),
        "Mem0 Sol runtime differs from sealed preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=plain,
        model=JUDGE_MODEL,
        client=client,
        max_prompt_tokens=MAX_JUDGE_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "answer_run_sha256": preflight.payload["answer_run_sha256"],
            "arm": "mem0_common_parent_sol_judge_v1",
            "authorized_unique_calls": expected_question_count,
            "common_input_sha256": preflight.payload["common_input_sha256"],
            "comparison_semantics": COMPARISON_SEMANTICS,
            "experiment_format": JUDGE_FORMAT,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_population_sha256": preflight.payload["gold_population_sha256"],
            "preflight_artifact_sha256": preflight.sha256,
        },
    )


def run_judge_checkpoint_batch(
    preflight: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    checkpoint_dir: str | Path,
    client: Any | None,
    model: str = JUDGE_MODEL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
    expected_question_count: int = 100,
) -> FastCompletionBatch:
    runtime = build_judge_runtime(
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


def materialize_judge_payloads(
    preflight: SealedArtifact,
    prompt_rows: Sequence[Mapping[str, Any]],
    batch: FastCompletionBatch,
    *,
    expected_question_count: int = 100,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        type(batch) is FastCompletionBatch
        and batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == expected_question_count
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == expected_question_count
        and len(batch.unique_records) == expected_question_count,
        "Mem0 Sol materialization requires complete checkpoint hits",
    )
    records = {record.messages_sha256: record for record in batch.unique_records}
    _require(
        len(records) == expected_question_count,
        "Mem0 Sol completion identities repeat",
    )
    rows: list[dict[str, Any]] = []
    for prompt, completion in zip(prompt_rows, batch.logical_completions, strict=True):
        record = records.get(prompt["messages_sha256"])
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False
            and record.requested_model == JUDGE_MODEL,
            "Mem0 Sol checkpoint record changed",
        )
        assert record is not None
        try:
            correct = parse_binary_judge_verdict(completion)
        except RuntimeError as exc:
            raise Mem0TypedJudgeLifecycleError(
                "Mem0 Sol returned an invalid binary verdict"
            ) from exc
        body = {
            "answer_prompt_row_receipt_sha256": prompt[
                "answer_prompt_row_receipt_sha256"
            ],
            "answer_source_row_sha256": prompt["answer_source_row_sha256"],
            "call_key_sha256": record.call_key_sha256,
            "category": prompt["category"],
            "correct": correct,
            "dated_question_sha256": prompt["dated_question_sha256"],
            "demand_class": prompt["demand_class"],
            "judge_output": completion,
            "judge_output_sha256": quote_sha256(completion),
            "messages_sha256": prompt["messages_sha256"],
            "normalized_exact_match": exact_match(
                prompt["prediction"], prompt["reference"]
            ),
            "normalized_f1": f1_score(prompt["prediction"], prompt["reference"]),
            "ordinal": prompt["ordinal"],
            "prediction_sha256": prompt["prediction_sha256"],
            "prediction_source": prompt["prediction_source"],
            "prompt_row_receipt_sha256": prompt["prompt_row_receipt_sha256"],
            "question_id": prompt["question_id"],
            "question_sha256": prompt["question_sha256"],
            "reference_sha256": prompt["reference_sha256"],
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "route_id": prompt["route_id"],
        }
        rows.append({**body, "judge_row_sha256": identity_sha256(body)})
    correct_count = sum(row["correct"] for row in rows)
    accounting = {
        "answer_complete_request_token_cap": HARD_PROMPT_TOKEN_CAP,
        "answer_max_prompt_tokens": MAX_CHAT_PROMPT_TOKENS,
        "answer_output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "judge_complete_envelope_token_cap": MAX_JUDGE_COMPLETE_ENVELOPE_TOKENS,
        "judge_max_prompt_tokens": MAX_JUDGE_PROMPT_TOKENS,
        "judge_model": JUDGE_MODEL,
        "judge_output_token_reserve": JUDGE_OUTPUT_TOKEN_RESERVE,
        "responder_model": RESPONDER_MODEL,
        "retained_transformer_token_state_bytes": 0,
        "sdk_retries": 0,
    }
    common = {
        "answer_replay_sha256": preflight.payload["answer_replay_sha256"],
        "answer_run_sha256": preflight.payload["answer_run_sha256"],
        "common_input_sha256": preflight.payload["common_input_sha256"],
        "comparison_semantics": COMPARISON_SEMANTICS,
        "gold_population_sha256": preflight.payload["gold_population_sha256"],
        "model_accounting": accounting,
        "parent_origin_receipt_sha256": preflight.payload[
            "parent_origin_receipt_sha256"
        ],
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": expected_question_count,
    }
    judge = {
        **common,
        "aggregate": {
            "accuracy": correct_count / expected_question_count,
            "correct": correct_count,
            "question_count": expected_question_count,
        },
        "completion_batch": batch.model_dump(),
        "format": JUDGE_FORMAT,
        "gold_loaded": True,
        "physical_provider_calls_during_materialization": 0,
        "questions": rows,
        "retained_transformer_token_state_bytes": 0,
    }
    score_rows = [
        {
            "correct": row["correct"],
            "answer_prompt_row_receipt_sha256": row[
                "answer_prompt_row_receipt_sha256"
            ],
            "judge_row_sha256": row["judge_row_sha256"],
            "ordinal": row["ordinal"],
            "question_id": row["question_id"],
            "question_sha256": row["question_sha256"],
        }
        for row in rows
    ]
    score = {
        **common,
        "accuracy": correct_count / expected_question_count,
        "correct": correct_count,
        "format": SCORE_FORMAT,
        "judge_artifact_sha256": hashlib.sha256(
            canonical_json_bytes(judge)
        ).hexdigest(),
        "score_rows": score_rows,
    }
    return judge, score


def materialize_judge_from_checkpoints(
    *,
    preflight_path: str | Path,
    expected_preflight_sha256: str,
    output_root: str | Path,
    expected_question_count: int = 100,
    model: str = JUDGE_MODEL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
) -> tuple[SealedArtifact, SealedArtifact]:
    preflight = _read_expected(
        preflight_path, expected_preflight_sha256, "Mem0 Sol preflight"
    )
    prompts, rows = validate_judge_preflight_artifact(
        preflight,
        expected_question_count=expected_question_count,
    )
    batch = run_judge_checkpoint_batch(
        preflight,
        prompts,
        checkpoint_dir=Path(output_root) / CHECKPOINT_DIR_NAME,
        client=None,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
        expected_question_count=expected_question_count,
    )
    judge_payload, score_payload = materialize_judge_payloads(
        preflight,
        rows,
        batch,
        expected_question_count=expected_question_count,
    )
    judge, _ = publish_sealed_json(Path(output_root) / JUDGE_NAME, judge_payload)
    _require(
        score_payload["judge_artifact_sha256"] == judge.sha256,
        "Mem0 Sol score/judge binding changed",
    )
    score, _ = publish_sealed_json(Path(output_root) / SCORE_NAME, score_payload)
    return judge, score


def replay_judge_from_checkpoints(
    *,
    preflight_path: str | Path,
    expected_preflight_sha256: str,
    judge_path: str | Path,
    expected_judge_sha256: str,
    score_path: str | Path,
    expected_score_sha256: str,
    output_root: str | Path,
    expected_question_count: int = 100,
    model: str = JUDGE_MODEL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
) -> tuple[SealedArtifact, SealedArtifact]:
    expected_judge = _read_expected(
        judge_path, expected_judge_sha256, "Mem0 Sol judge"
    )
    expected_score = _read_expected(
        score_path, expected_score_sha256, "Mem0 Sol score"
    )
    rebuilt_judge, rebuilt_score = materialize_judge_from_checkpoints(
        preflight_path=preflight_path,
        expected_preflight_sha256=expected_preflight_sha256,
        output_root=output_root,
        expected_question_count=expected_question_count,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
    )
    _require(
        rebuilt_judge.sha256 == expected_judge.sha256
        and rebuilt_judge.payload == expected_judge.payload
        and rebuilt_score.sha256 == expected_score.sha256
        and rebuilt_score.payload == expected_score.payload,
        "Mem0 Sol judge/score are not byte-identical on checkpoint replay",
    )
    judge_replay, _ = publish_sealed_json(
        Path(output_root) / JUDGE_REPLAY_NAME, expected_judge.payload
    )
    score_replay, _ = publish_sealed_json(
        Path(output_root) / SCORE_REPLAY_NAME, expected_score.payload
    )
    return judge_replay, score_replay


def load_verified_judge_score(
    output_root: str | Path,
    *,
    common_input_path: str | Path,
    expected_common_input_sha256: str,
    answer_output_root: str | Path,
    expected_answer_preflight_sha256: str,
    expected_answer_run_sha256: str,
    expected_answer_replay_sha256: str,
    dataset_path: str | Path,
    split_path: str | Path,
    expected_preflight_sha256: str,
    expected_judge_sha256: str,
    expected_judge_replay_sha256: str,
    expected_score_sha256: str,
    expected_score_replay_sha256: str,
    expected_question_count: int = 100,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
]:
    """Reopen every authority and return an authenticated per-question score."""

    answer_run, answer_replay, source_rows = load_verified_answer_run(
        answer_output_root,
        common_input_path=common_input_path,
        expected_common_input_sha256=expected_common_input_sha256,
        expected_preflight_sha256=expected_answer_preflight_sha256,
        expected_run_sha256=expected_answer_run_sha256,
        expected_replay_sha256=expected_answer_replay_sha256,
        expected_question_count=expected_question_count,
    )
    gold_rows, gold_population_sha256 = load_locked_typed_final_gold(
        dataset_path=dataset_path,
        split_path=split_path,
        source_rows=source_rows,
        allow_subset=expected_question_count != 100,
    )
    root = Path(output_root)
    preflight = _read_expected(
        root / PREFLIGHT_NAME, expected_preflight_sha256, "Mem0 Sol preflight"
    )
    prompts, prompt_rows = validate_judge_preflight_artifact(
        preflight,
        expected_question_count=expected_question_count,
    )
    rebuilt_preflight, rebuilt_prompts = build_judge_preflight_payload(
        answer_run=answer_run,
        answer_replay=answer_replay,
        source_rows=source_rows,
        gold_rows=gold_rows,
        gold_population_sha256=gold_population_sha256,
        expected_question_count=expected_question_count,
        model=JUDGE_MODEL,
        gateway_url=DEFAULT_GATEWAY_URL,
        max_concurrency=int(preflight.payload["max_concurrency"]),
    )
    _require(
        rebuilt_preflight == preflight.payload and rebuilt_prompts == prompts,
        "Mem0 Sol preflight is not the exact answer/gold projection",
    )
    judge = _read_expected(root / JUDGE_NAME, expected_judge_sha256, "Mem0 Sol judge")
    judge_replay = _read_expected(
        root / JUDGE_REPLAY_NAME,
        expected_judge_replay_sha256,
        "Mem0 Sol judge replay",
    )
    score = _read_expected(root / SCORE_NAME, expected_score_sha256, "Mem0 Sol score")
    score_replay = _read_expected(
        root / SCORE_REPLAY_NAME,
        expected_score_replay_sha256,
        "Mem0 Sol score replay",
    )
    _require(
        judge.sha256 == judge_replay.sha256
        and judge.payload == judge_replay.payload
        and score.sha256 == score_replay.sha256
        and score.payload == score_replay.payload,
        "Mem0 Sol judge/score replay is not byte-identical",
    )
    checkpoint_root = root / CHECKPOINT_DIR_NAME
    _require(
        checkpoint_root.is_dir() and not checkpoint_root.is_symlink(),
        "Mem0 Sol checkpoint directory is missing or unsafe",
    )
    try:
        runtime = build_judge_runtime(
            preflight,
            prompts,
            checkpoint_dir=checkpoint_root,
            client=None,
            model=JUDGE_MODEL,
            gateway_url=DEFAULT_GATEWAY_URL,
            max_concurrency=int(preflight.payload["max_concurrency"]),
            expected_question_count=expected_question_count,
        )
        try:
            batch = runtime.run()
        finally:
            runtime.close()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Mem0TypedJudgeLifecycleError(
            "Mem0 Sol checkpoint journals do not authenticate"
        ) from exc
    expected_entries = {
        name
        for record in batch.unique_records
        for name in (
            f"{record.call_key_sha256}.request.json",
            f"{record.call_key_sha256}.response.json",
        )
    }
    entries = tuple(
        entry
        for entry in checkpoint_root.iterdir()
        if entry.name != ".fast-completion-journal.lock"
    )
    _require(
        {entry.name for entry in entries} == expected_entries
        and all(entry.is_file() and not entry.is_symlink() for entry in entries),
        "Mem0 Sol checkpoint directory contains unbound entries",
    )
    rebuilt_judge, rebuilt_score = materialize_judge_payloads(
        preflight,
        prompt_rows,
        batch,
        expected_question_count=expected_question_count,
    )
    _require(
        rebuilt_judge == judge.payload
        and rebuilt_score == score.payload
        and judge.payload.get("completion_batch") == batch.model_dump(),
        "Mem0 Sol stored judge/score differ from their authenticated journals",
    )
    rows = tuple(dict(row) for row in _list(judge.payload.get("questions"), "judge rows"))
    _require(
        len(rows) == expected_question_count
        and score.payload.get("judge_artifact_sha256") == judge.sha256
        and score.payload.get("correct") == sum(row["correct"] for row in rows)
        and score.payload.get("accuracy")
        == sum(row["correct"] for row in rows) / expected_question_count,
        "Mem0 Sol score recomputation changed",
    )
    return judge, judge_replay, score, score_replay, rows


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "FORMAT",
    "JUDGE_FORMAT",
    "JUDGE_NAME",
    "JUDGE_REPLAY_NAME",
    "MAX_JUDGE_PROMPT_TOKENS",
    "MAX_JUDGE_COMPLETE_ENVELOPE_TOKENS",
    "Mem0TypedJudgeLifecycleError",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "SCORE_FORMAT",
    "SCORE_NAME",
    "SCORE_REPLAY_NAME",
    "build_judge_preflight_payload",
    "build_judge_runtime",
    "load_verified_judge_score",
    "materialize_judge_from_checkpoints",
    "materialize_judge_payloads",
    "replay_judge_from_checkpoints",
    "run_judge_checkpoint_batch",
    "validate_judge_preflight_artifact",
]
