#!/usr/bin/env python3
"""Run the locked full-100 Sol judge and deterministic score lifecycle.

The replay-verified full100 terminal answer is authenticated before benchmark
gold is opened.  Preflight then seals exactly one judge prompt for every locked
validation question.  Each provider message is the common binary-judge prompt
built from only question, reference, and the sealed prediction; answer evidence
and terminal handles never enter the provider population.

Provider execution requires a separate release artifact, owns a distinct
100-call journal namespace, has zero retries, and accepts only an authorization
equal to the exact number of missing checkpoint pairs.  Materialization and its
replay are checkpoint-only.  Scoring and score replay consume only the sealed,
byte-identical judge artifacts and never open provider state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval._binary_judge_protocol import (  # noqa: E402
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import build_judge_prompt  # noqa: E402
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
)
from tools import (  # noqa: E402
    run_locked_semantic_global_terminal_full100_answer as answer_cli,
)
from tools.matched_eval import judging, live  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.typed_memory_final_judging import (  # noqa: E402
    DEFAULT_MAX_PROMPT_TOKENS,
    JUDGE_FORMAT,
    JUDGE_MAX_TOKENS,
    PREFLIGHT_FORMAT as TYPED_PREFLIGHT_FORMAT,
    SCORE_FORMAT as TYPED_SCORE_FORMAT,
    TypedFinalJudgeGoldRow,
    load_locked_typed_final_gold,
    materialization_projection,
    preflight_projection,
    validate_preflight_artifact,
)
from tools.run_locked_query_answer_judge import DEFAULT_DATASET  # noqa: E402
from tools.run_matched_eval_spine import DEFAULT_SPLIT  # noqa: E402


FORMAT = "memory-condense-locked-semantic-global-terminal-full100-sol-judge-v1"
PREFLIGHT_LIFECYCLE_FORMAT = f"{FORMAT}-preflight-v1"
RELEASE_FORMAT = f"{FORMAT}-provider-release-v1"
JUDGE_LIFECYCLE_FORMAT = f"{FORMAT}-judge-v1"
SCORE_LIFECYCLE_FORMAT = f"{FORMAT}-score-v1"
JUDGE_INPUT_FORMAT = f"{FORMAT}-judge-input-row-v1"
JOURNAL_OWNER_FORMAT = f"{FORMAT}-journal-owner-v1"

PREFLIGHT_NAME = "semantic-global-terminal-full100-sol-judge-preflight-v1.json"
RELEASE_NAME = "semantic-global-terminal-full100-sol-judge-provider-release-v1.json"
JUDGE_NAME = "semantic-global-terminal-full100-sol-judge-v1.json"
JUDGE_REPLAY_NAME = "semantic-global-terminal-full100-sol-judge-replay-v1.json"
SCORE_NAME = "semantic-global-terminal-full100-sol-score-v1.json"
SCORE_REPLAY_NAME = "semantic-global-terminal-full100-sol-score-replay-v1.json"
CHECKPOINT_DIR_NAME = "sol-semantic-global-terminal-full100-judge-v1-calls"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWER_ROOT = answer_cli.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = DEFAULT_ANSWER_ROOT / "sol-judge-v1"
DEFAULT_MODEL = judging.DEFAULT_SOL_GATEWAY_MODEL
DEFAULT_GATEWAY_URL = live.DEFAULT_GATEWAY_URL
DEFAULT_MAX_CONCURRENCY = 4

QUESTION_COUNT = answer_cli.QUESTION_COUNT
ALL_ORDINALS = tuple(range(QUESTION_COUNT))
_JOURNAL_FILENAME_RE = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)

ANSWER_SOURCE_BINDING_KEY_MAP = {
    f"full100_answer_source_{key}": key
    for key in answer_cli.SOURCE_BINDING_KEYS
}
ANSWER_BINDING_KEYS = (
    "full100_answer_preflight_artifact_sha256",
    "full100_answer_release_authorization_artifact_sha256",
    "full100_answer_run_artifact_sha256",
    "full100_answer_replay_artifact_sha256",
    "full100_answer_postseal_audit_artifact_sha256",
    *ANSWER_SOURCE_BINDING_KEY_MAP,
)

SOURCE_ROW_KEYS = {
    "changed_from_parent",
    "dated_question_sha256",
    "format",
    "ordinal",
    "parent_prediction_sha256",
    "prediction",
    "prediction_sha256",
    "prediction_source",
    "question_id",
    "question_sha256",
    "route_id",
    "source_row_sha256",
}
PROMPT_ROW_KEYS = {
    "category",
    "dated_question_sha256",
    "demand_class",
    "judge_input_format",
    "judge_input_receipt_sha256",
    "messages",
    "messages_sha256",
    "ordinal",
    "prediction",
    "prediction_sha256",
    "prediction_source",
    "prompt_row_receipt_sha256",
    "prompt_token_proxy",
    "question",
    "question_id",
    "question_sha256",
    "reference",
    "reference_sha256",
    "route_id",
    "source_row_sha256",
}
PREFLIGHT_KEYS = {
    "answer_source_binding_sha256",
    "format",
    "gateway_url",
    "gold_loaded",
    "gold_population_sha256",
    "judge_input_population_sha256",
    "judge_mode",
    "lifecycle_format",
    "max_concurrency",
    "model",
    "ordinal_cli_routing_available",
    "physical_provider_calls",
    "production_ordinal_routing_enabled",
    "prompt_population",
    "prompt_population_sha256",
    "prompt_rows",
    "question_count",
    "required_authorized_provider_calls",
    "retained_transformer_token_state_bytes",
    "selected_question_count",
    "typed_final_replay_sha256",
    "typed_final_run_sha256",
}.union(ANSWER_BINDING_KEYS)
RELEASE_KEYS = {
    "answer_root",
    "answer_root_sha256",
    "answer_source_binding_sha256",
    "approval_opt_in",
    "checkpoint_root",
    "checkpoint_root_sha256",
    "format",
    "gateway_url",
    "gold_loaded",
    "gold_population_sha256",
    "journal_owner_identity_sha256",
    "journal_owner_format",
    "judge_output_root",
    "judge_output_root_sha256",
    "max_concurrency",
    "model",
    "ordinal_cli_routing_available",
    "preflight_artifact_sha256",
    "production_ordinal_routing_enabled",
    "prompt_population_sha256",
    "provider_calls_during_release",
    "question_count",
    "release_identity_sha256",
    "release_status",
    "required_authorized_provider_calls",
    "retained_transformer_token_state_bytes",
    "retry_count",
    "unsafe_retry_policy",
}.union(ANSWER_BINDING_KEYS)
JUDGE_ROW_KEYS = {
    "call_key_sha256",
    "category",
    "correct",
    "dated_question_sha256",
    "demand_class",
    "judge_output",
    "judge_output_sha256",
    "judge_row_sha256",
    "messages_sha256",
    "normalized_exact_match",
    "normalized_f1",
    "ordinal",
    "prediction_sha256",
    "prediction_source",
    "prompt_row_receipt_sha256",
    "question_id",
    "question_sha256",
    "reference_sha256",
    "request_journal_sha256",
    "response_journal_sha256",
    "route_id",
    "source_row_sha256",
}
JUDGE_KEYS = {
    "aggregate",
    "answer_source_binding_sha256",
    "completion_batch",
    "format",
    "gold_loaded",
    "gold_population_sha256",
    "journal_owner_identity_sha256",
    "judge_mode",
    "lifecycle_format",
    "physical_provider_calls_during_materialization",
    "preflight_artifact_sha256",
    "prompt_population_sha256",
    "question_count",
    "questions",
    "release_authorization_artifact_sha256",
    "retained_transformer_token_state_bytes",
    "selected_question_count",
    "typed_final_run_sha256",
}.union(ANSWER_BINDING_KEYS)
SCORE_KEYS = {
    "accuracy",
    "answer_source_binding_sha256",
    "correct",
    "format",
    "gold_loaded",
    "gold_population_sha256",
    "judge_artifact_sha256",
    "judge_mode",
    "judge_replay_artifact_sha256",
    "lifecycle_format",
    "physical_provider_calls_during_scoring",
    "preflight_artifact_sha256",
    "prompt_population_sha256",
    "question_count",
    "release_authorization_artifact_sha256",
    "retained_transformer_token_state_bytes",
    "selected_accuracy",
    "selected_question_count",
    "typed_final_run_sha256",
}.union(ANSWER_BINDING_KEYS)


class LockedSemanticGlobalTerminalFull100JudgeError(MatchedEvalContractError):
    """A source, gold population, release, journal, verdict, or score changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticGlobalTerminalFull100JudgeError(message)


def _canonical_root(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _answer_binding(
    run: SealedArtifact,
    replay: SealedArtifact,
    *,
    postseal_audit_sha256: str,
) -> dict[str, Any]:
    payload = run.payload
    binding = {
        "full100_answer_preflight_artifact_sha256": payload.get(
            "preflight_artifact_sha256"
        ),
        "full100_answer_release_authorization_artifact_sha256": payload.get(
            "release_authorization_artifact_sha256"
        ),
        "full100_answer_run_artifact_sha256": run.sha256,
        "full100_answer_replay_artifact_sha256": replay.sha256,
        "full100_answer_postseal_audit_artifact_sha256": require_sha256(
            postseal_audit_sha256, "full100 judge postseal audit"
        ),
        **{
            target: payload.get(source)
            for target, source in ANSWER_SOURCE_BINDING_KEY_MAP.items()
        },
    }
    for key, value in binding.items():
        if key.endswith("_sha256"):
            require_sha256(value, f"full100 judge answer binding {key}")
        else:
            _require(
                type(value) is int and value >= 0,
                f"full100 judge answer binding {key} changed scalar type",
            )
    _require(
        replay.payload.get("expected_run_sha256") == run.sha256
        and replay.payload.get("replayed_run_sha256") == run.sha256
        and replay.payload.get("preflight_artifact_sha256")
        == binding["full100_answer_preflight_artifact_sha256"]
        and replay.payload.get("release_authorization_artifact_sha256")
        == binding["full100_answer_release_authorization_artifact_sha256"]
        and all(
            replay.payload.get(key) == payload.get(key)
            for key in answer_cli.SOURCE_BINDING_KEYS
        ),
        "full100 judge answer run/replay binding changed",
    )
    return binding


def _validate_source_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    result = tuple(dict(row) for row in rows)
    _require(
        len(result) == QUESTION_COUNT
        and tuple(row.get("ordinal") for row in result) == ALL_ORDINALS
        and len({row.get("question_id") for row in result}) == QUESTION_COUNT,
        "full100 judge source population/order changed",
    )
    for ordinal, row in enumerate(result):
        prediction = row.get("prediction")
        _require(
            set(row) == SOURCE_ROW_KEYS
            and row.get("ordinal") == ordinal
            and type(row.get("question_id")) is str
            and bool(row["question_id"])
            and type(prediction) is str
            and bool(prediction)
            and row.get("prediction_sha256") == quote_sha256(prediction)
            and type(row.get("changed_from_parent")) is bool
            and type(row.get("prediction_source")) is str
            and bool(row["prediction_source"])
            and type(row.get("route_id")) is str
            and bool(row["route_id"]),
            f"full100 judge source row {ordinal} changed",
        )
        for key in (
            "dated_question_sha256",
            "parent_prediction_sha256",
            "prediction_sha256",
            "question_sha256",
            "source_row_sha256",
        ):
            require_sha256(row.get(key), f"full100 judge source {key}")
    return result


def _judge_input_body(
    row: Mapping[str, Any], messages: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    return {
        "format": JUDGE_INPUT_FORMAT,
        "messages_sha256": identity_sha256([dict(value) for value in messages]),
        "ordinal": row["ordinal"],
        "prediction": row["prediction"],
        "prediction_sha256": row["prediction_sha256"],
        "question": row["question"],
        "question_id": row["question_id"],
        "question_sha256": row["question_sha256"],
        "reference": row["reference"],
        "reference_sha256": row["reference_sha256"],
    }


def build_preflight_payload(
    run: SealedArtifact,
    replay: SealedArtifact,
    source_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[TypedFinalJudgeGoldRow],
    *,
    gold_population_sha256: str,
    postseal_audit_sha256: str,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    source = _validate_source_rows(source_rows)
    gold = tuple(gold_rows)
    binding = _answer_binding(
        run, replay, postseal_audit_sha256=postseal_audit_sha256
    )
    _require(
        model == DEFAULT_MODEL
        and gateway_url == DEFAULT_GATEWAY_URL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "full100 judge runtime policy changed",
    )
    base, prompts = preflight_projection(
        run_artifact=run,
        replay_artifact_sha256=replay.sha256,
        source_rows=source,
        gold_rows=gold,
        gold_population_sha256=gold_population_sha256,
        mode="full100",
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
    )
    raw_rows = base.get("prompt_rows")
    _require(
        type(raw_rows) is list
        and len(raw_rows) == len(gold) == len(prompts) == QUESTION_COUNT,
        "full100 judge base preflight population changed",
    )
    sealed_rows: list[dict[str, Any]] = []
    input_receipts: list[str] = []
    for raw, gold_row, messages in zip(raw_rows, gold, prompts, strict=True):
        body = dict(raw)
        body.pop("prompt_row_receipt_sha256", None)
        body.update(
            {
                "judge_input_format": JUDGE_INPUT_FORMAT,
                "question": gold_row.question,
            }
        )
        expected = tuple(
            dict(value)
            for value in build_judge_prompt(
                gold_row.question, gold_row.reference, body["prediction"]
            )
        )
        _require(
            tuple(dict(value) for value in messages) == expected
            and body.get("messages") == list(expected),
            f"full100 judge input {gold_row.ordinal} contains non-contract data",
        )
        input_receipt = identity_sha256(_judge_input_body(body, expected))
        body["judge_input_receipt_sha256"] = input_receipt
        sealed_rows.append(
            {**body, "prompt_row_receipt_sha256": identity_sha256(body)}
        )
        input_receipts.append(input_receipt)
    source_binding_sha = identity_sha256(binding)
    payload = {
        **base,
        **binding,
        "answer_source_binding_sha256": source_binding_sha,
        "judge_input_population_sha256": identity_sha256(input_receipts),
        "lifecycle_format": PREFLIGHT_LIFECYCLE_FORMAT,
        "ordinal_cli_routing_available": False,
        "production_ordinal_routing_enabled": False,
        "prompt_rows": sealed_rows,
    }
    _require(
        set(payload) == PREFLIGHT_KEYS
        and payload.get("format") == TYPED_PREFLIGHT_FORMAT
        and payload.get("judge_mode") == "full100"
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("selected_question_count") == QUESTION_COUNT
        and payload.get("required_authorized_provider_calls") == QUESTION_COUNT
        and payload.get("typed_final_run_sha256") == run.sha256
        and payload.get("typed_final_replay_sha256") == replay.sha256,
        "full100 judge preflight envelope changed",
    )
    return payload, prompts


def _validate_preflight(
    artifact: SealedArtifact,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    prompts, rows = validate_preflight_artifact(artifact)
    payload = artifact.payload
    binding = {key: payload.get(key) for key in ANSWER_BINDING_KEYS}
    for key, value in binding.items():
        if key.endswith("_sha256"):
            require_sha256(value, f"sealed full100 judge binding {key}")
        else:
            _require(
                type(value) is int and value >= 0,
                f"sealed full100 judge binding {key} changed scalar type",
            )
    _require(
        set(payload) == PREFLIGHT_KEYS
        and payload.get("format") == TYPED_PREFLIGHT_FORMAT
        and payload.get("lifecycle_format") == PREFLIGHT_LIFECYCLE_FORMAT
        and payload.get("gold_loaded") is True
        and payload.get("physical_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("judge_mode") == "full100"
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("gateway_url") == DEFAULT_GATEWAY_URL
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("selected_question_count") == QUESTION_COUNT
        and payload.get("required_authorized_provider_calls") == QUESTION_COUNT
        and payload.get("ordinal_cli_routing_available") is False
        and payload.get("production_ordinal_routing_enabled") is False
        and payload.get("typed_final_run_sha256")
        == payload.get("full100_answer_run_artifact_sha256")
        and payload.get("typed_final_replay_sha256")
        == payload.get("full100_answer_replay_artifact_sha256")
        and payload.get("answer_source_binding_sha256")
        == identity_sha256(binding)
        and len(prompts) == len(rows) == QUESTION_COUNT
        and tuple(row.get("ordinal") for row in rows) == ALL_ORDINALS,
        "sealed full100 judge preflight changed",
    )
    receipts: list[str] = []
    question_ids: list[str] = []
    message_hashes: list[str] = []
    for ordinal, (messages, row) in enumerate(zip(prompts, rows, strict=True)):
        _require(
            set(row) == PROMPT_ROW_KEYS
            and row.get("ordinal") == ordinal
            and row.get("judge_input_format") == JUDGE_INPUT_FORMAT
            and row.get("question_sha256")
            == quote_sha256(require_text(row.get("question"), "judge question"))
            and row.get("reference_sha256")
            == quote_sha256(require_text(row.get("reference"), "judge reference"))
            and row.get("prediction_sha256")
            == quote_sha256(require_text(row.get("prediction"), "judge prediction")),
            f"sealed full100 judge prompt row {ordinal} changed",
        )
        expected = tuple(
            dict(value)
            for value in build_judge_prompt(
                row["question"], row["reference"], row["prediction"]
            )
        )
        input_receipt = identity_sha256(_judge_input_body(row, expected))
        _require(
            tuple(dict(value) for value in messages) == expected
            and row.get("messages") == list(expected)
            and row.get("messages_sha256") == identity_sha256(list(expected))
            and row.get("judge_input_receipt_sha256") == input_receipt,
            f"sealed full100 judge prompt {ordinal} leaked non-contract input",
        )
        receipts.append(input_receipt)
        question_ids.append(row["question_id"])
        message_hashes.append(row["messages_sha256"])
    _require(
        len(set(question_ids)) == len(set(message_hashes)) == QUESTION_COUNT
        and len(set(receipts)) == QUESTION_COUNT
        and payload.get("judge_input_population_sha256")
        == identity_sha256(receipts),
        "sealed full100 judge prompt identities repeat or changed",
    )
    return prompts, rows


def _read_preflight(
    output_root: str | Path, expected_sha256: str
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    artifact = read_sealed_json(Path(output_root) / PREFLIGHT_NAME)
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "full100 judge preflight"),
        "full100 judge preflight artifact changed",
    )
    prompts, rows = _validate_preflight(artifact)
    return artifact, prompts, rows


def _load_answer_source(
    args: argparse.Namespace,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    return answer_cli.load_verified_answer_run(
        args.answer_root,
        expected_preflight_sha256=str(args.expected_answer_preflight_sha256),
        expected_run_sha256=str(args.expected_answer_run_sha256),
        expected_replay_sha256=str(args.expected_answer_replay_sha256),
        postseal_audit=args.postseal_audit,
        expected_postseal_audit_sha256=str(
            args.expected_postseal_audit_sha256
        ),
    )


def _assert_preflight_answer_binding(
    preflight: SealedArtifact,
    run: SealedArtifact,
    replay: SealedArtifact,
    *,
    postseal_audit_sha256: str,
) -> None:
    binding = _answer_binding(
        run, replay, postseal_audit_sha256=postseal_audit_sha256
    )
    _require(
        all(preflight.payload.get(key) == value for key, value in binding.items())
        and preflight.payload.get("answer_source_binding_sha256")
        == identity_sha256(binding),
        "full100 judge preflight differs from authenticated answer source",
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.judge_output_root)
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "full100 judge preflight requires a fresh absent checkpoint root",
    )
    # The complete gold-free answer lifecycle is authenticated first.
    run, replay, source_rows = _load_answer_source(args)
    source = _validate_source_rows(source_rows)
    binding = _answer_binding(
        run,
        replay,
        postseal_audit_sha256=str(args.expected_postseal_audit_sha256),
    )
    gold_rows, gold_sha = load_locked_typed_final_gold(
        dataset_path=args.dataset,
        split_path=args.split,
        source_rows=source,
        allow_subset=False,
    )
    payload, _ = build_preflight_payload(
        run,
        replay,
        source,
        gold_rows,
        gold_population_sha256=gold_sha,
        postseal_audit_sha256=str(args.expected_postseal_audit_sha256),
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(output_root / PREFLIGHT_NAME, payload)
    return {
        "answer_replay_sha256": replay.sha256,
        "answer_run_sha256": run.sha256,
        "answer_source_binding_sha256": identity_sha256(binding),
        "created": created,
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def _journal_owner_body(
    preflight: SealedArtifact, *, output_root: str | Path
) -> dict[str, Any]:
    judge_root = _canonical_root(output_root)
    checkpoint_root = _canonical_root(Path(output_root) / CHECKPOINT_DIR_NAME)
    return {
        "checkpoint_root": checkpoint_root,
        "checkpoint_root_sha256": identity_sha256(
            {"canonical_root": checkpoint_root}
        ),
        "format": JOURNAL_OWNER_FORMAT,
        "judge_output_root": judge_root,
        "judge_output_root_sha256": identity_sha256(
            {"canonical_root": judge_root}
        ),
        "model": preflight.payload["model"],
        "preflight_artifact_sha256": preflight.sha256,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
        "question_count": QUESTION_COUNT,
    }


def _release_payload(
    preflight: SealedArtifact,
    *,
    answer_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    answer = _canonical_root(answer_root)
    owner = _journal_owner_body(preflight, output_root=output_root)
    body = {
        "answer_root": answer,
        "answer_root_sha256": identity_sha256({"canonical_root": answer}),
        "answer_source_binding_sha256": preflight.payload[
            "answer_source_binding_sha256"
        ],
        "approval_opt_in": True,
        "checkpoint_root": owner["checkpoint_root"],
        "checkpoint_root_sha256": owner["checkpoint_root_sha256"],
        "format": RELEASE_FORMAT,
        "gateway_url": preflight.payload["gateway_url"],
        "gold_loaded": True,
        "gold_population_sha256": preflight.payload["gold_population_sha256"],
        "journal_owner_identity_sha256": identity_sha256(owner),
        "journal_owner_format": JOURNAL_OWNER_FORMAT,
        "judge_output_root": owner["judge_output_root"],
        "judge_output_root_sha256": owner["judge_output_root_sha256"],
        "max_concurrency": preflight.payload["max_concurrency"],
        "model": preflight.payload["model"],
        "ordinal_cli_routing_available": False,
        "preflight_artifact_sha256": preflight.sha256,
        "production_ordinal_routing_enabled": False,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
        "provider_calls_during_release": 0,
        "question_count": QUESTION_COUNT,
        "release_status": "approved_for_provider_execution",
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "retry_count": 0,
        "unsafe_retry_policy": "refuse_incomplete_request_response_pair",
        **{key: preflight.payload[key] for key in ANSWER_BINDING_KEYS},
    }
    return {**body, "release_identity_sha256": identity_sha256(body)}


def _validate_release(
    artifact: SealedArtifact,
    *,
    preflight: SealedArtifact,
    output_root: str | Path,
) -> dict[str, Any]:
    payload = artifact.payload
    body = {
        key: value
        for key, value in payload.items()
        if key != "release_identity_sha256"
    }
    owner = _journal_owner_body(preflight, output_root=output_root)
    answer_root = require_text(payload.get("answer_root"), "judge answer root")
    _require(
        set(payload) == RELEASE_KEYS
        and require_sha256(payload.get("release_identity_sha256"), "judge release")
        == identity_sha256(body)
        and payload.get("format") == RELEASE_FORMAT
        and payload.get("release_status") == "approved_for_provider_execution"
        and payload.get("approval_opt_in") is True
        and payload.get("gold_loaded") is True
        and payload.get("provider_calls_during_release") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("retry_count") == 0
        and payload.get("unsafe_retry_policy")
        == "refuse_incomplete_request_response_pair"
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("required_authorized_provider_calls") == QUESTION_COUNT
        and payload.get("ordinal_cli_routing_available") is False
        and payload.get("production_ordinal_routing_enabled") is False
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("model") == preflight.payload.get("model")
        and payload.get("gateway_url") == preflight.payload.get("gateway_url")
        and payload.get("max_concurrency")
        == preflight.payload.get("max_concurrency")
        and payload.get("gold_population_sha256")
        == preflight.payload.get("gold_population_sha256")
        and payload.get("prompt_population_sha256")
        == preflight.payload.get("prompt_population_sha256")
        and payload.get("answer_source_binding_sha256")
        == preflight.payload.get("answer_source_binding_sha256")
        and payload.get("answer_root_sha256")
        == identity_sha256({"canonical_root": answer_root})
        and payload.get("journal_owner_format") == owner.get("format")
        and all(
            payload.get(key) == value
            for key, value in owner.items()
            if key != "format"
        )
        and payload.get("journal_owner_identity_sha256")
        == identity_sha256(owner)
        and all(
            payload.get(key) == preflight.payload.get(key)
            for key in ANSWER_BINDING_KEYS
        ),
        "full100 judge provider release changed",
    )
    return payload


def _read_release(
    output_root: str | Path,
    expected_sha256: str,
    *,
    preflight: SealedArtifact,
) -> SealedArtifact:
    artifact = read_sealed_json(Path(output_root) / RELEASE_NAME)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, "full100 judge release"),
        "full100 judge release artifact changed",
    )
    _validate_release(artifact, preflight=preflight, output_root=output_root)
    return artifact


def run_approve_release(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.judge_output_root)
    _require(
        args.approve_provider_release is True,
        "full100 judge release requires explicit provider approval",
    )
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "full100 judge release requires an absent checkpoint root",
    )
    run, replay, rows = _load_answer_source(args)
    _validate_source_rows(rows)
    preflight, _, _ = _read_preflight(
        output_root, str(args.expected_judge_preflight_sha256)
    )
    _assert_preflight_answer_binding(
        preflight,
        run,
        replay,
        postseal_audit_sha256=str(args.expected_postseal_audit_sha256),
    )
    payload = _release_payload(
        preflight, answer_root=args.answer_root, output_root=output_root
    )
    artifact, created = publish_sealed_json(output_root / RELEASE_NAME, payload)
    return {
        "created": created,
        "journal_owner_identity_sha256": payload[
            "journal_owner_identity_sha256"
        ],
        "physical_provider_calls": 0,
        "preflight_sha256": preflight.sha256,
        "release_sha256": artifact.sha256,
        "required_authorized_provider_calls": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def _runtime(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionRuntime:
    _require(
        str(args.model) == preflight.payload.get("model") == DEFAULT_MODEL
        and str(args.gateway_url)
        == preflight.payload.get("gateway_url")
        == DEFAULT_GATEWAY_URL
        and int(args.max_concurrency) == preflight.payload.get("max_concurrency")
        and release.payload.get("preflight_artifact_sha256") == preflight.sha256
        and release.payload.get("release_status")
        == "approved_for_provider_execution"
        and len(prompts) == QUESTION_COUNT,
        "full100 judge runtime differs from sealed release",
    )
    return FastCompletionRuntime(
        checkpoint_dir=Path(args.judge_output_root) / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=DEFAULT_MODEL,
        client=client,
        max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=int(args.max_concurrency),
        retries=0,
        benchmark_provenance={
            "answer_source_binding_sha256": preflight.payload[
                "answer_source_binding_sha256"
            ],
            "arm": FORMAT,
            "authorized_unique_calls": QUESTION_COUNT,
            "experiment_format": JUDGE_LIFECYCLE_FORMAT,
            "gold_population_sha256": preflight.payload[
                "gold_population_sha256"
            ],
            "journal_owner_identity_sha256": release.payload[
                "journal_owner_identity_sha256"
            ],
            "judge_mode": "full100",
            "preflight_artifact_sha256": preflight.sha256,
            "release_authorization_artifact_sha256": release.sha256,
            "typed_final_run_sha256": preflight.payload[
                "typed_final_run_sha256"
            ],
        },
    )


def _checkpoint_batch(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(preflight, release, prompts, args=args, client=client)
    try:
        return runtime.run()
    finally:
        runtime.close()


def _read_only_checkpoint_count(output_root: str | Path) -> int:
    root = Path(output_root) / CHECKPOINT_DIR_NAME
    if not root.exists():
        return 0
    _require(
        not root.is_symlink() and root.is_dir(),
        "full100 judge checkpoint root must be a regular directory",
    )
    requests: set[str] = set()
    responses: set[str] = set()
    for path in root.iterdir():
        _require(
            not path.is_symlink() and path.is_file(),
            "full100 judge checkpoint root contains foreign state",
        )
        if path.name == ".fast-completion-journal.lock":
            continue
        match = _JOURNAL_FILENAME_RE.fullmatch(path.name)
        _require(
            match is not None,
            "full100 judge checkpoint root contains foreign journal state",
        )
        assert match is not None
        target = requests if match.group("kind") == "request" else responses
        target.add(match.group("key"))
    _require(
        requests == responses,
        "full100 judge checkpoint pair is incomplete; unsafe retry forbidden",
    )
    _require(
        len(requests) <= QUESTION_COUNT,
        "full100 judge checkpoint population exceeds 100 calls",
    )
    return len(requests)


def _validated_checkpoint_hits(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
) -> int:
    root = Path(args.judge_output_root) / CHECKPOINT_DIR_NAME
    if not root.exists():
        return 0
    runtime = _runtime(preflight, release, prompts, args=args, client=None)
    try:
        with runtime._journal_guard():  # noqa: SLF001 - runtime owns journals
            records = runtime._load_all_records()  # noqa: SLF001
    finally:
        runtime.close()
    _require(
        len(records) <= QUESTION_COUNT,
        "full100 judge checkpoint population escaped the sealed prompts",
    )
    return len(records)


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, _ = _read_preflight(
        args.judge_output_root, str(args.expected_judge_preflight_sha256)
    )
    release = _read_release(
        args.judge_output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    _require(
        args.enable_provider is True
        and type(args.authorized_provider_calls) is int
        and 0 <= args.authorized_provider_calls <= QUESTION_COUNT,
        "full100 judge provider requires bounded Sol authorization",
    )
    candidate_hits = _read_only_checkpoint_count(args.judge_output_root)
    remaining = QUESTION_COUNT - candidate_hits
    _require(
        args.authorized_provider_calls == remaining,
        "full100 judge authorization must exactly equal remaining calls",
    )
    checkpoint_hits = _validated_checkpoint_hits(
        preflight, release, prompts, args=args
    )
    _require(
        checkpoint_hits == candidate_hits,
        "full100 judge checkpoint count changed after authorization",
    )
    if remaining == 0:
        batch = _checkpoint_batch(
            preflight, release, prompts, args=args, client=None
        )
        _require(
            batch.usage.logical_calls
            == batch.usage.unique_calls
            == batch.usage.checkpoint_hits
            == QUESTION_COUNT
            and batch.usage.physical_calls == 0,
            "full100 judge completed checkpoint replay changed",
        )
    else:
        load_dotenv()
        api_key = os.environ.get(str(args.api_key_env), "").strip()
        _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
        client = judging._make_provider_client(  # noqa: SLF001
            api_key, str(args.gateway_url)
        )
        try:
            batch = _checkpoint_batch(
                preflight, release, prompts, args=args, client=client
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        _require(
            batch.usage.logical_calls
            == batch.usage.unique_calls
            == QUESTION_COUNT
            and batch.usage.physical_calls + batch.usage.checkpoint_hits
            == QUESTION_COUNT
            and batch.usage.physical_calls <= args.authorized_provider_calls
            and batch.usage.checkpoint_hits >= checkpoint_hits,
            "full100 judge provider population changed",
        )
    return {
        "authorized_remaining_provider_calls": remaining,
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": preflight.sha256,
        "release_sha256": release.sha256,
        "required_authorized_provider_calls": remaining,
        "retained_transformer_token_state_bytes": 0,
    }


def _judge_payload(
    preflight: SealedArtifact,
    release: SealedArtifact,
    rows: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    judge, _unused_score = materialization_projection(preflight, rows, batch)
    return {
        **judge,
        **{key: preflight.payload[key] for key in ANSWER_BINDING_KEYS},
        "answer_source_binding_sha256": preflight.payload[
            "answer_source_binding_sha256"
        ],
        "journal_owner_identity_sha256": release.payload[
            "journal_owner_identity_sha256"
        ],
        "lifecycle_format": JUDGE_LIFECYCLE_FORMAT,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
        "release_authorization_artifact_sha256": release.sha256,
    }


def _validate_judge(
    preflight: SealedArtifact,
    release: SealedArtifact,
    payload: Mapping[str, Any],
    *,
    expected_batch: FastCompletionBatch | None = None,
) -> tuple[dict[str, Any], ...]:
    raw_rows = payload.get("questions")
    aggregate = payload.get("aggregate")
    _require(
        set(payload) == JUDGE_KEYS
        and payload.get("format") == JUDGE_FORMAT
        and payload.get("lifecycle_format") == JUDGE_LIFECYCLE_FORMAT
        and payload.get("gold_loaded") is True
        and payload.get("judge_mode") == "full100"
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("selected_question_count") == QUESTION_COUNT
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("release_authorization_artifact_sha256") == release.sha256
        and payload.get("journal_owner_identity_sha256")
        == release.payload.get("journal_owner_identity_sha256")
        and payload.get("prompt_population_sha256")
        == preflight.payload.get("prompt_population_sha256")
        and payload.get("gold_population_sha256")
        == preflight.payload.get("gold_population_sha256")
        and payload.get("typed_final_run_sha256")
        == preflight.payload.get("typed_final_run_sha256")
        and payload.get("answer_source_binding_sha256")
        == preflight.payload.get("answer_source_binding_sha256")
        and all(
            payload.get(key) == preflight.payload.get(key)
            for key in ANSWER_BINDING_KEYS
        )
        and type(raw_rows) is list
        and len(raw_rows) == QUESTION_COUNT
        and type(aggregate) is dict
        and aggregate.get("question_count") == QUESTION_COUNT,
        "full100 judge materialization envelope changed",
    )
    if expected_batch is not None:
        _require(
            payload.get("completion_batch")
            == judging._stable_batch(expected_batch)  # noqa: SLF001
            and expected_batch.usage.logical_calls
            == expected_batch.usage.unique_calls
            == expected_batch.usage.checkpoint_hits
            == QUESTION_COUNT
            and expected_batch.usage.physical_calls == 0,
            "full100 judge completion batch changed",
        )
    prompt_by_ordinal = {
        row["ordinal"]: row for row in preflight.payload["prompt_rows"]
    }
    rows: list[dict[str, Any]] = []
    call_keys: list[str] = []
    question_ids: list[str] = []
    for ordinal, raw in enumerate(raw_rows):
        _require(type(raw) is dict, "full100 judge verdict changed type")
        body = dict(raw)
        declared = body.pop("judge_row_sha256", None)
        prompt = prompt_by_ordinal.get(ordinal)
        output = raw.get("judge_output")
        _require(
            set(raw) == JUDGE_ROW_KEYS
            and declared == identity_sha256(body)
            and raw.get("ordinal") == ordinal
            and type(raw.get("correct")) is bool
            and type(output) is str
            and bool(output)
            and raw.get("judge_output_sha256") == quote_sha256(output)
            and parse_binary_judge_verdict(output) is raw.get("correct")
            and prompt is not None
            and raw.get("messages_sha256") == prompt.get("messages_sha256")
            and raw.get("prediction_sha256") == prompt.get("prediction_sha256")
            and raw.get("reference_sha256") == prompt.get("reference_sha256")
            and raw.get("source_row_sha256") == prompt.get("source_row_sha256"),
            f"full100 judge verdict row {ordinal} changed",
        )
        for key in (
            "call_key_sha256",
            "messages_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
        ):
            require_sha256(raw.get(key), f"full100 judge verdict {key}")
        call_keys.append(raw["call_key_sha256"])
        question_ids.append(raw["question_id"])
        rows.append(dict(raw))
    correct = sum(bool(row["correct"]) for row in rows)
    _require(
        len(set(call_keys)) == len(set(question_ids)) == QUESTION_COUNT
        and aggregate.get("correct") == correct
        and aggregate.get("accuracy") == correct / QUESTION_COUNT,
        "full100 judge score arithmetic or identities changed",
    )
    return tuple(rows)


def _complete_checkpoint_batch(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
) -> FastCompletionBatch:
    _require(
        _read_only_checkpoint_count(args.judge_output_root) == QUESTION_COUNT,
        "full100 judge materialization requires 100 complete checkpoints",
    )
    return _checkpoint_batch(preflight, release, prompts, args=args, client=None)


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, rows = _read_preflight(
        args.judge_output_root, str(args.expected_judge_preflight_sha256)
    )
    release = _read_release(
        args.judge_output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    batch = _complete_checkpoint_batch(
        preflight, release, prompts, args=args
    )
    payload = _judge_payload(preflight, release, rows, batch)
    _validate_judge(preflight, release, payload, expected_batch=batch)
    artifact, created = publish_sealed_json(
        Path(args.judge_output_root) / JUDGE_NAME, payload
    )
    return {
        "checkpoint_hits": QUESTION_COUNT,
        "created": created,
        "judge_sha256": artifact.sha256,
        "physical_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, rows = _read_preflight(
        args.judge_output_root, str(args.expected_judge_preflight_sha256)
    )
    release = _read_release(
        args.judge_output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    batch = _complete_checkpoint_batch(
        preflight, release, prompts, args=args
    )
    rebuilt = _judge_payload(preflight, release, rows, batch)
    _validate_judge(preflight, release, rebuilt, expected_batch=batch)
    root = Path(args.judge_output_root)
    artifact = read_sealed_json(root / JUDGE_NAME)
    _require(
        artifact.sha256
        == require_sha256(args.expected_judge_sha256, "full100 judge")
        and artifact.payload == rebuilt,
        "full100 judge differs from checkpoint-only replay",
    )
    replay, _ = publish_sealed_json(root / JUDGE_REPLAY_NAME, rebuilt)
    _require(
        replay.sha256 == artifact.sha256,
        "full100 judge replay is not byte-identical",
    )
    return {
        "byte_identical": True,
        "judge_replay_sha256": replay.sha256,
        "judge_sha256": artifact.sha256,
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
    }


def _load_verified_judge_replay(
    output_root: str | Path,
    *,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
    expected_judge_sha256: str,
    expected_judge_replay_sha256: str,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
]:
    root = Path(output_root)
    preflight, _, _ = _read_preflight(root, expected_preflight_sha256)
    release = _read_release(
        root, expected_release_sha256, preflight=preflight
    )
    judge = read_sealed_json(root / JUDGE_NAME)
    replay = read_sealed_json(root / JUDGE_REPLAY_NAME)
    _require(
        judge.sha256 == require_sha256(expected_judge_sha256, "full100 judge")
        and replay.sha256
        == require_sha256(
            expected_judge_replay_sha256, "full100 judge replay"
        )
        == judge.sha256
        and replay.payload == judge.payload,
        "full100 judge/replay artifacts changed",
    )
    rows = _validate_judge(preflight, release, judge.payload)
    return preflight, release, judge, replay, rows


def _score_payload(
    preflight: SealedArtifact,
    release: SealedArtifact,
    judge: SealedArtifact,
    replay: SealedArtifact,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    correct = sum(bool(row["correct"]) for row in rows)
    accuracy = correct / QUESTION_COUNT
    return {
        "accuracy": accuracy,
        "answer_source_binding_sha256": preflight.payload[
            "answer_source_binding_sha256"
        ],
        "correct": correct,
        "format": TYPED_SCORE_FORMAT,
        "gold_loaded": True,
        "gold_population_sha256": preflight.payload[
            "gold_population_sha256"
        ],
        "judge_artifact_sha256": judge.sha256,
        "judge_mode": "full100",
        "judge_replay_artifact_sha256": replay.sha256,
        "lifecycle_format": SCORE_LIFECYCLE_FORMAT,
        "physical_provider_calls_during_scoring": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
        "question_count": QUESTION_COUNT,
        "release_authorization_artifact_sha256": release.sha256,
        "retained_transformer_token_state_bytes": 0,
        "selected_accuracy": accuracy,
        "selected_question_count": QUESTION_COUNT,
        "typed_final_run_sha256": preflight.payload["typed_final_run_sha256"],
        **{key: preflight.payload[key] for key in ANSWER_BINDING_KEYS},
    }


def _validate_score(
    preflight: SealedArtifact,
    release: SealedArtifact,
    judge: SealedArtifact,
    replay: SealedArtifact,
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    expected = _score_payload(preflight, release, judge, replay, rows)
    _require(
        set(payload) == SCORE_KEYS
        and dict(payload) == expected
        and payload.get("format") == TYPED_SCORE_FORMAT
        and payload.get("lifecycle_format") == SCORE_LIFECYCLE_FORMAT
        and payload.get("gold_loaded") is True
        and payload.get("physical_provider_calls_during_scoring") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("selected_question_count") == QUESTION_COUNT,
        "full100 judge score changed",
    )


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    preflight, release, judge, replay, rows = _load_verified_judge_replay(
        args.judge_output_root,
        expected_preflight_sha256=str(args.expected_judge_preflight_sha256),
        expected_release_sha256=str(args.expected_release_sha256),
        expected_judge_sha256=str(args.expected_judge_sha256),
        expected_judge_replay_sha256=str(args.expected_judge_replay_sha256),
    )
    payload = _score_payload(preflight, release, judge, replay, rows)
    _validate_score(preflight, release, judge, replay, payload, rows)
    artifact, created = publish_sealed_json(
        Path(args.judge_output_root) / SCORE_NAME, payload
    )
    return {
        "accuracy": payload["accuracy"],
        "correct": payload["correct"],
        "created": created,
        "physical_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "score_sha256": artifact.sha256,
    }


def run_score_replay(args: argparse.Namespace) -> dict[str, Any]:
    preflight, release, judge, replay, rows = _load_verified_judge_replay(
        args.judge_output_root,
        expected_preflight_sha256=str(args.expected_judge_preflight_sha256),
        expected_release_sha256=str(args.expected_release_sha256),
        expected_judge_sha256=str(args.expected_judge_sha256),
        expected_judge_replay_sha256=str(args.expected_judge_replay_sha256),
    )
    rebuilt = _score_payload(preflight, release, judge, replay, rows)
    root = Path(args.judge_output_root)
    score = read_sealed_json(root / SCORE_NAME)
    _require(
        score.sha256
        == require_sha256(args.expected_score_sha256, "full100 judge score")
        and score.payload == rebuilt,
        "full100 judge score differs from deterministic replay",
    )
    _validate_score(preflight, release, judge, replay, score.payload, rows)
    score_replay, _ = publish_sealed_json(root / SCORE_REPLAY_NAME, rebuilt)
    _require(
        score_replay.sha256 == score.sha256,
        "full100 judge score replay is not byte-identical",
    )
    return {
        "byte_identical": True,
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
        "score_replay_sha256": score_replay.sha256,
        "score_sha256": score.sha256,
    }


def load_verified_judge_score(
    output_root: str | Path,
    *,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
    expected_judge_sha256: str,
    expected_judge_replay_sha256: str,
    expected_score_sha256: str,
    expected_score_replay_sha256: str,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
]:
    """Return the judge and score only after both byte-identical replays."""

    preflight, release, judge, replay, rows = _load_verified_judge_replay(
        output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
        expected_judge_sha256=expected_judge_sha256,
        expected_judge_replay_sha256=expected_judge_replay_sha256,
    )
    root = Path(output_root)
    score = read_sealed_json(root / SCORE_NAME)
    score_replay = read_sealed_json(root / SCORE_REPLAY_NAME)
    _require(
        score.sha256
        == require_sha256(expected_score_sha256, "full100 judge score")
        and score_replay.sha256
        == require_sha256(
            expected_score_replay_sha256, "full100 judge score replay"
        )
        == score.sha256
        and score_replay.payload == score.payload,
        "full100 judge score/replay artifacts changed",
    )
    _validate_score(preflight, release, judge, replay, score.payload, rows)
    return judge, score, rows


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--judge-output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument(
        "--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY
    )


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--judge-output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )


def _add_answer_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--answer-root", type=Path, default=DEFAULT_ANSWER_ROOT
    )
    parser.add_argument("--expected-answer-preflight-sha256", required=True)
    parser.add_argument("--expected-answer-run-sha256", required=True)
    parser.add_argument("--expected-answer-replay-sha256", required=True)
    parser.add_argument("--postseal-audit", type=Path, required=True)
    parser.add_argument("--expected-postseal-audit-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    _add_runtime(preflight)
    _add_answer_source(preflight)
    preflight.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    preflight.add_argument("--split", type=Path, default=DEFAULT_SPLIT)

    approve = commands.add_parser("approve-release")
    _add_runtime(approve)
    _add_answer_source(approve)
    approve.add_argument("--expected-judge-preflight-sha256", required=True)
    approve.add_argument("--approve-provider-release", action="store_true")

    provider = commands.add_parser("provider-run")
    _add_runtime(provider)
    provider.add_argument("--expected-judge-preflight-sha256", required=True)
    provider.add_argument("--expected-release-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser("materialize")
    _add_runtime(materialize)
    materialize.add_argument("--expected-judge-preflight-sha256", required=True)
    materialize.add_argument("--expected-release-sha256", required=True)

    replay = commands.add_parser("replay")
    _add_runtime(replay)
    replay.add_argument("--expected-judge-preflight-sha256", required=True)
    replay.add_argument("--expected-release-sha256", required=True)
    replay.add_argument("--expected-judge-sha256", required=True)

    score = commands.add_parser("score")
    _add_output(score)
    score.add_argument("--expected-judge-preflight-sha256", required=True)
    score.add_argument("--expected-release-sha256", required=True)
    score.add_argument("--expected-judge-sha256", required=True)
    score.add_argument("--expected-judge-replay-sha256", required=True)

    score_replay = commands.add_parser("score-replay")
    _add_output(score_replay)
    score_replay.add_argument("--expected-judge-preflight-sha256", required=True)
    score_replay.add_argument("--expected-release-sha256", required=True)
    score_replay.add_argument("--expected-judge-sha256", required=True)
    score_replay.add_argument("--expected-judge-replay-sha256", required=True)
    score_replay.add_argument("--expected-score-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args)
    elif args.command == "approve-release":
        result = run_approve_release(args)
    elif args.command == "provider-run":
        result = run_provider(args)
    elif args.command == "materialize":
        result = run_materialize(args)
    elif args.command == "replay":
        result = run_replay(args)
    elif args.command == "score":
        result = run_score(args)
    else:
        result = run_score_replay(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALL_ORDINALS",
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_ANSWER_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "FORMAT",
    "JUDGE_NAME",
    "JUDGE_REPLAY_NAME",
    "LockedSemanticGlobalTerminalFull100JudgeError",
    "PREFLIGHT_NAME",
    "QUESTION_COUNT",
    "RELEASE_NAME",
    "SCORE_NAME",
    "SCORE_REPLAY_NAME",
    "build_parser",
    "build_preflight_payload",
    "load_verified_judge_score",
    "main",
    "run_approve_release",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
    "run_score",
    "run_score_replay",
]
