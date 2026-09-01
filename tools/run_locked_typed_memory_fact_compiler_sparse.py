#!/usr/bin/env python3
"""Offline v3 compiler rematerialization and sparse fact-answer lifecycle.

This runner never issues a compiler call.  It verifies the exact sealed v2
compiler preflight, completion journals, run, and replay, then reparses those
same raw completion bytes with the current provider-free compiler parser.

Only valid rematerialized packets enter the answer provider population.
Invalid packets deterministically keep their protected parent locally.  The
sealed answer result therefore always contains the complete fixed 24-row
diagnostic population while its physical prompt population contains exactly
``valid_packet_count`` rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
)
from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools import run_locked_typed_memory_fact_compiler as v2_cli  # noqa: E402
from tools.matched_eval import live  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.typed_fact_compiler import (  # noqa: E402
    ANSWER_OUTPUT_TOKEN_RESERVE,
    COMPILATION_FORMAT,
    DEFAULT_FACT_PACKET_TOKEN_CAP,
    FACT_PACKET_FORMAT,
    MAX_COMPILER_FACTS,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    RESULT_ROW_FORMAT,
    VALIDATOR_POLICY_FORMAT,
    judge_row_projection,
    materialize_typed_final_result_row,
)


FORMAT = "memory-condense-locked-typed-memory-fact-compiler-sparse-v3"
REMATERIALIZED_FORMAT = f"{FORMAT}-compiler-rematerialized-v1"
REMATERIALIZED_REPLAY_FORMAT = f"{FORMAT}-compiler-rematerialized-replay-v1"
ANSWER_PREFLIGHT_FORMAT = f"{FORMAT}-answer-preflight-v1"
ANSWER_RUN_FORMAT = f"{FORMAT}-answer-run-v1"
ANSWER_REPLAY_FORMAT = f"{FORMAT}-answer-replay-v1"
LOCAL_FALLBACK_FORMAT = f"{FORMAT}-local-parent-fallback-v1"

REMATERIALIZED_NAME = "typed-fact-compiler-rematerialized-v3.json"
REMATERIALIZED_REPLAY_NAME = "typed-fact-compiler-rematerialized-replay-v3.json"
ANSWER_PREFLIGHT_NAME = "typed-fact-answer-sparse-preflight-v3.json"
ANSWER_RUN_NAME = "typed-fact-answer-sparse-run-v3.json"
ANSWER_REPLAY_NAME = "typed-fact-answer-sparse-replay-v3.json"
ANSWER_CHECKPOINT_DIR_NAME = "terra-typed-fact-answer-sparse-v3-calls"

EXPECTED_V2_PREFLIGHT_SHA256 = (
    "c020b625011e67a71112b952a60c49f627f817a5c68a4155ff6c780bd8b44fc2"
)
EXPECTED_V2_RUN_SHA256 = (
    "2de0f0d27c6b08510fdc4e799dcfa8914cf5cf53a02de9fce3c1974d202c85b2"
)
EXPECTED_V2_REPLAY_SHA256 = (
    "a35e5c05e1e006bab943a85db4a1f4a89e6bab669354a9021118ebb4c7469720"
)

HARD_PROMPT_TOKEN_CAP = 8_000
REMAINING_ORDINALS = v2_cli.REMAINING_ORDINALS
SUBSET_QUESTION_COUNT = v2_cli.SUBSET_QUESTION_COUNT

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2_ROOT = (
    REPOSITORY_ROOT
    / "eval_results"
    / "matched_eval_100"
    / "typed-memory-final-v3-fact-compiler-remaining24-v2"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "eval_results"
    / "matched_eval_100"
    / "typed-memory-final-v3-fact-compiler-remaining24-v3-sparse"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LockedTypedFactCompilerSparseError(MatchedEvalContractError):
    """A legacy completion, parser result, sparse population, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedTypedFactCompilerSparseError(message)


def _sha256_argument(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def _read_v2_compiler_source(
    root: Path,
    *,
    expected_preflight_sha256: str,
    expected_run_sha256: str,
    expected_replay_sha256: str,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
    FastCompletionBatch,
    tuple[dict[str, Any], ...],
]:
    preflight, prompts, prompt_rows = v2_cli._read_phase_preflight(  # noqa: SLF001
        root,
        name=v2_cli.COMPILER_PREFLIGHT_NAME,
        expected_sha256=expected_preflight_sha256,
        validator=v2_cli._validate_compiler_preflight,  # noqa: SLF001
    )
    batch = v2_cli._checkpoint_batch(  # noqa: SLF001
        preflight,
        prompts,
        output_root=root,
        checkpoint_dir_name=v2_cli.COMPILER_CHECKPOINT_DIR_NAME,
        run_format=v2_cli.COMPILER_RUN_FORMAT,
        phase="compiler",
        client=None,
    )
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == SUBSET_QUESTION_COUNT
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == SUBSET_QUESTION_COUNT
        and len(batch.unique_records) == SUBSET_QUESTION_COUNT,
        "v2 compiler source is not 24 immutable checkpoint hits",
    )
    run = read_sealed_json(root / v2_cli.COMPILER_RUN_NAME)
    replay = read_sealed_json(root / v2_cli.COMPILER_REPLAY_NAME)
    _require(
        preflight.sha256
        == require_sha256(expected_preflight_sha256, "expected v2 preflight")
        and run.sha256 == require_sha256(expected_run_sha256, "expected v2 run")
        and replay.sha256
        == require_sha256(expected_replay_sha256, "expected v2 replay"),
        "v2 compiler lineage SHA-256 changed",
    )
    run_payload = run.payload
    replay_payload = replay.payload
    questions = run_payload.get("questions")
    _require(
        run_payload.get("format") == v2_cli.COMPILER_RUN_FORMAT
        and run_payload.get("gold_loaded") is False
        and run_payload.get("physical_provider_calls_during_materialization") == 0
        and run_payload.get("retained_transformer_token_state_bytes") == 0
        and run_payload.get("question_count") == SUBSET_QUESTION_COUNT
        and run_payload.get("original_ordinals") == list(REMAINING_ORDINALS)
        and run_payload.get("compiler_preflight_artifact_sha256")
        == preflight.sha256
        and type(questions) is list
        and len(questions) == SUBSET_QUESTION_COUNT
        and replay_payload.get("format") == v2_cli.COMPILER_REPLAY_FORMAT
        and replay_payload.get("byte_identical") is True
        and replay_payload.get("gold_loaded") is False
        and replay_payload.get("physical_provider_calls") == 0
        and replay_payload.get("expected_run_sha256") == run.sha256
        and replay_payload.get("replayed_run_sha256") == run.sha256
        and replay_payload.get("compiler_preflight_artifact_sha256")
        == preflight.sha256,
        "v2 compiler run/replay envelope changed",
    )
    assert_gold_blind(run_payload, path="sparse_v3_legacy_v2_run")
    assert_gold_blind(replay_payload, path="sparse_v3_legacy_v2_replay")
    records = {row.messages_sha256: row for row in batch.unique_records}
    _require(
        len(records) == SUBSET_QUESTION_COUNT,
        "v2 compiler checkpoint identities repeat",
    )
    verified_rows: list[dict[str, Any]] = []
    assert type(questions) is list
    for ordinal, prompt, raw, completion in zip(
        REMAINING_ORDINALS,
        prompt_rows,
        questions,
        batch.logical_completions,
        strict=True,
    ):
        _require(type(raw) is dict, "v2 compiler result row changed type")
        assert type(raw) is dict
        unsigned = dict(raw)
        declared = unsigned.pop("compiler_result_row_sha256", None)
        record = records.get(prompt["messages_sha256"])
        _require(
            declared == identity_sha256(unsigned)
            and raw.get("ordinal") == ordinal
            and raw.get("question_id") == prompt.get("question_id")
            and raw.get("question_sha256") == prompt.get("question_sha256")
            and raw.get("compiler_prompt_row_receipt_sha256")
            == prompt.get("prompt_row_receipt_sha256")
            and raw.get("compiler_completion") == completion
            and quote_sha256(completion)
            == raw.get("compiler_completion_sha256")
            and record is not None
            and record.completion == completion
            and record.completion_sha256 == raw.get("compiler_completion_sha256")
            and record.call_key_sha256 == raw.get("call_key_sha256")
            and record.request_journal_sha256
            == raw.get("request_journal_sha256")
            and record.response_journal_sha256
            == raw.get("response_journal_sha256"),
            f"v2 raw compiler completion binding changed at ordinal {ordinal}",
        )
        verified_rows.append(dict(raw))
    return preflight, run, replay, prompt_rows, batch, tuple(verified_rows)


def _rematerialized_projection(
    legacy_preflight: SealedArtifact,
    legacy_run: SealedArtifact,
    legacy_replay: SealedArtifact,
    prompt_rows: Sequence[Mapping[str, Any]],
    batch: FastCompletionBatch,
    legacy_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = {row.messages_sha256: row for row in batch.unique_records}
    results: list[dict[str, Any]] = []
    for ordinal, prompt, completion, legacy in zip(
        REMAINING_ORDINALS,
        prompt_rows,
        batch.logical_completions,
        legacy_rows,
        strict=True,
    ):
        record = records[prompt["messages_sha256"]]
        _compilation, compilation_projection, _packet, packet_projection = (
            v2_cli._parse_compilation(prompt, completion)  # noqa: SLF001
        )
        body = {
            "call_key_sha256": record.call_key_sha256,
            "compiler_completion": completion,
            "compiler_completion_sha256": record.completion_sha256,
            "compiler_prompt_row_receipt_sha256": prompt[
                "prompt_row_receipt_sha256"
            ],
            "compilation": compilation_projection,
            "composition_row_sha256": prompt["composition_row_sha256"],
            "dated_question_sha256": prompt["dated_question_sha256"],
            "fact_packet": packet_projection,
            "fact_packet_sha256": identity_sha256(packet_projection),
            "legacy_compiler_result_row_sha256": legacy[
                "compiler_result_row_sha256"
            ],
            "ordinal": ordinal,
            "question_id": prompt["question_id"],
            "question_sha256": prompt["question_sha256"],
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "retained_transformer_token_state_bytes": 0,
            "route_id": prompt["route_id"],
            "source_prompt_row_receipt_sha256": prompt[
                "source_prompt_row_receipt_sha256"
            ],
        }
        body["compiler_result_row_sha256"] = identity_sha256(body)
        results.append(body)
    valid = sum(row["fact_packet"]["valid"] is True for row in results)
    payload = {
        "compiler_parser_contract": {
            "compilation_format": COMPILATION_FORMAT,
            "fact_packet_format": FACT_PACKET_FORMAT,
            "max_compiler_facts": MAX_COMPILER_FACTS,
            "max_fact_packet_tokens": DEFAULT_FACT_PACKET_TOKEN_CAP,
        },
        "format": REMATERIALIZED_FORMAT,
        "gold_loaded": False,
        "invalid_packet_count": SUBSET_QUESTION_COUNT - valid,
        "legacy_compiler_preflight_artifact_sha256": legacy_preflight.sha256,
        "legacy_compiler_replay_artifact_sha256": legacy_replay.sha256,
        "legacy_compiler_run_artifact_sha256": legacy_run.sha256,
        "new_compiler_provider_calls": 0,
        "original_ordinals": list(REMAINING_ORDINALS),
        "questions": results,
        "question_count": SUBSET_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "selection_is_posthoc_outcome_conditioned": True,
        "valid_packet_count": valid,
    }
    assert_gold_blind(payload, path="typed_fact_sparse_v3_rematerialized")
    return payload


def _legacy_from_args(args: argparse.Namespace):
    return _read_v2_compiler_source(
        Path(args.v2_root),
        expected_preflight_sha256=args.expected_v2_preflight_sha256,
        expected_run_sha256=args.expected_v2_run_sha256,
        expected_replay_sha256=args.expected_v2_replay_sha256,
    )


def _compiler_rematerialize(args: argparse.Namespace) -> dict[str, Any]:
    legacy = _legacy_from_args(args)
    payload = _rematerialized_projection(*legacy)
    artifact, created = publish_sealed_json(
        Path(args.output_root) / REMATERIALIZED_NAME,
        payload,
    )
    return {
        "created": created,
        "invalid_packet_count": payload["invalid_packet_count"],
        "new_compiler_provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "rematerialized_sha256": artifact.sha256,
        "retained_transformer_token_state_bytes": 0,
        "valid_packet_count": payload["valid_packet_count"],
    }


def _validate_rematerialized(
    artifact: SealedArtifact,
    *,
    legacy_preflight: SealedArtifact,
    legacy_run: SealedArtifact,
    legacy_replay: SealedArtifact,
    prompt_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    questions = payload.get("questions")
    parser = payload.get("compiler_parser_contract")
    _require(
        payload.get("format") == REMATERIALIZED_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("new_compiler_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == SUBSET_QUESTION_COUNT
        and payload.get("original_ordinals") == list(REMAINING_ORDINALS)
        and payload.get("selection_is_posthoc_outcome_conditioned") is True
        and payload.get("legacy_compiler_preflight_artifact_sha256")
        == legacy_preflight.sha256
        and payload.get("legacy_compiler_run_artifact_sha256") == legacy_run.sha256
        and payload.get("legacy_compiler_replay_artifact_sha256")
        == legacy_replay.sha256
        and type(parser) is dict
        and parser.get("compilation_format") == COMPILATION_FORMAT
        and parser.get("fact_packet_format") == FACT_PACKET_FORMAT
        and parser.get("max_compiler_facts") == MAX_COMPILER_FACTS
        and parser.get("max_fact_packet_tokens")
        == DEFAULT_FACT_PACKET_TOKEN_CAP
        and type(questions) is list
        and len(questions) == SUBSET_QUESTION_COUNT,
        "rematerialized compiler envelope changed",
    )
    assert_gold_blind(payload, path="verified_typed_fact_sparse_rematerialized")
    valid = 0
    verified: list[dict[str, Any]] = []
    assert type(questions) is list
    for ordinal, raw, prompt in zip(
        REMAINING_ORDINALS,
        questions,
        prompt_rows,
        strict=True,
    ):
        _require(type(raw) is dict, "rematerialized compiler row changed type")
        assert type(raw) is dict
        unsigned = dict(raw)
        declared = unsigned.pop("compiler_result_row_sha256", None)
        packet = raw.get("fact_packet")
        _require(
            declared == identity_sha256(unsigned)
            and raw.get("ordinal") == ordinal
            and raw.get("question_id") == prompt.get("question_id")
            and raw.get("question_sha256") == prompt.get("question_sha256")
            and raw.get("compiler_prompt_row_receipt_sha256")
            == prompt.get("prompt_row_receipt_sha256")
            and type(packet) is dict
            and type(packet.get("valid")) is bool
            and raw.get("fact_packet_sha256") == identity_sha256(packet),
            f"rematerialized compiler row binding changed at ordinal {ordinal}",
        )
        if packet["valid"] is True:
            valid += 1
        verified.append(dict(raw))
    _require(
        payload.get("valid_packet_count") == valid
        and payload.get("invalid_packet_count") == SUBSET_QUESTION_COUNT - valid,
        "rematerialized compiler validity accounting changed",
    )
    return tuple(verified)


def _compiler_rematerialize_replay(args: argparse.Namespace) -> dict[str, Any]:
    legacy = _legacy_from_args(args)
    replayed = _rematerialized_projection(*legacy)
    root = Path(args.output_root)
    observed = read_sealed_json(root / REMATERIALIZED_NAME)
    _require(
        observed.sha256
        == require_sha256(
            args.expected_rematerialized_sha256,
            "expected rematerialized compiler run",
        )
        and observed.payload == replayed,
        "compiler rematerialization differs from checkpoint-only replay",
    )
    _validate_rematerialized(
        observed,
        legacy_preflight=legacy[0],
        legacy_run=legacy[1],
        legacy_replay=legacy[2],
        prompt_rows=legacy[3],
    )
    payload = {
        "byte_identical": True,
        "expected_rematerialized_sha256": observed.sha256,
        "format": REMATERIALIZED_REPLAY_FORMAT,
        "gold_loaded": False,
        "legacy_compiler_preflight_artifact_sha256": legacy[0].sha256,
        "legacy_compiler_replay_artifact_sha256": legacy[2].sha256,
        "legacy_compiler_run_artifact_sha256": legacy[1].sha256,
        "new_compiler_provider_calls": 0,
        "physical_provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "rematerialized_sha256": observed.sha256,
        "retained_transformer_token_state_bytes": 0,
    }
    assert_gold_blind(payload, path="typed_fact_sparse_rematerialized_replay")
    artifact, created = publish_sealed_json(
        root / REMATERIALIZED_REPLAY_NAME,
        payload,
    )
    return {
        "byte_identical": True,
        "created": created,
        "new_compiler_provider_calls": 0,
        "rematerialized_replay_sha256": artifact.sha256,
        "rematerialized_sha256": observed.sha256,
    }


def _read_verified_rematerialization(
    args: argparse.Namespace,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    legacy = _legacy_from_args(args)
    root = Path(args.output_root)
    rematerialized = read_sealed_json(root / REMATERIALIZED_NAME)
    _require(
        rematerialized.sha256
        == require_sha256(
            args.expected_rematerialized_sha256,
            "expected rematerialized compiler run",
        ),
        "rematerialized compiler SHA-256 changed",
    )
    result_rows = _validate_rematerialized(
        rematerialized,
        legacy_preflight=legacy[0],
        legacy_run=legacy[1],
        legacy_replay=legacy[2],
        prompt_rows=legacy[3],
    )
    replay = read_sealed_json(root / REMATERIALIZED_REPLAY_NAME)
    replay_payload = replay.payload
    _require(
        replay.sha256
        == require_sha256(
            args.expected_rematerialized_replay_sha256,
            "expected rematerialized compiler replay",
        )
        and replay_payload.get("format") == REMATERIALIZED_REPLAY_FORMAT
        and replay_payload.get("byte_identical") is True
        and replay_payload.get("gold_loaded") is False
        and replay_payload.get("new_compiler_provider_calls") == 0
        and replay_payload.get("physical_provider_calls") == 0
        and replay_payload.get("rematerialized_sha256")
        == rematerialized.sha256
        and replay_payload.get("legacy_compiler_preflight_artifact_sha256")
        == legacy[0].sha256
        and replay_payload.get("legacy_compiler_run_artifact_sha256")
        == legacy[1].sha256
        and replay_payload.get("legacy_compiler_replay_artifact_sha256")
        == legacy[2].sha256,
        "rematerialized compiler replay binding changed",
    )
    return rematerialized, replay, legacy[3], result_rows


def _validate_prompt_rows(
    raw_rows: object,
    *,
    expected_ordinals: Sequence[int],
    label: str,
) -> tuple[
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
    Any,
]:
    expected = tuple(expected_ordinals)
    _require(
        type(raw_rows) is list and len(raw_rows) == len(expected),
        f"{label} prompt population changed",
    )
    assert type(raw_rows) is list
    prompts: list[tuple[dict[str, str], ...]] = []
    validated: list[dict[str, Any]] = []
    for ordinal, raw in zip(expected, raw_rows, strict=True):
        _require(type(raw) is dict, f"{label} prompt row changed type")
        assert type(raw) is dict
        body = dict(raw)
        declared = body.pop("prompt_row_receipt_sha256", None)
        messages = v2_cli._plain_messages(  # noqa: SLF001
            raw.get("messages"),
            label=label,
        )
        _require(
            declared == identity_sha256(body)
            and raw.get("ordinal") == ordinal
            and raw.get("messages_sha256") == identity_sha256(list(messages))
            and raw.get("prompt_token_proxy")
            == count_chat_prompt_token_proxy(messages)
            and int(raw["prompt_token_proxy"]) + ANSWER_OUTPUT_TOKEN_RESERVE
            <= HARD_PROMPT_TOKEN_CAP,
            f"{label} prompt seal/order/envelope changed at ordinal {ordinal}",
        )
        v2_cli._decoded_user_payloads(messages, label=label)  # noqa: SLF001
        prompts.append(messages)
        validated.append(dict(raw))
    _require(bool(prompts), f"{label} sparse provider population is empty")
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=HARD_PROMPT_TOKEN_CAP - ANSWER_OUTPUT_TOKEN_RESERVE,
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == len(expected),
        f"{label} prompts must be physically unique",
    )
    return tuple(prompts), tuple(validated), population


def _answer_preflight_projection(
    rematerialized: SealedArtifact,
    rematerialized_replay: SealedArtifact,
    compiler_prompt_rows: Sequence[Mapping[str, Any]],
    compiler_result_rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    require_text(model, "sparse answer model")
    require_text(gateway_url, "sparse answer gateway")
    _require(
        type(max_concurrency) is int and max_concurrency > 0,
        "sparse answer concurrency changed",
    )
    all_plans = tuple(
        v2_cli._answer_prompt_row(prompt, result)  # noqa: SLF001
        for prompt, result in zip(
            compiler_prompt_rows,
            compiler_result_rows,
            strict=True,
        )
    )
    _require(
        tuple(row["ordinal"] for row in all_plans) == REMAINING_ORDINALS,
        "sparse answer all-plan ordinals changed",
    )
    selected = tuple(row for row in all_plans if row["fact_packet_valid"] is True)
    invalid = tuple(row for row in all_plans if row["fact_packet_valid"] is False)
    selected_ordinals = tuple(row["ordinal"] for row in selected)
    invalid_ordinals = tuple(row["ordinal"] for row in invalid)
    _require(
        len(selected) == rematerialized.payload.get("valid_packet_count")
        and len(invalid) == rematerialized.payload.get("invalid_packet_count")
        and len(selected) + len(invalid) == SUBSET_QUESTION_COUNT,
        "sparse answer validity partition changed",
    )
    _require(
        0 < len(selected) < SUBSET_QUESTION_COUNT,
        "locked sparse treatment requires both provider and local partitions",
    )
    for row in invalid:
        source = row.get("source_prompt_plan")
        _require(
            type(source) is dict
            and row.get("compiled_handle_authority_enforced") is False
            and row.get("compiled_retained_handle_ids") == []
            and row.get("byte_identical_source_fallback") is True
            and row.get("messages") == source.get("messages")
            and row.get("messages_sha256") == source.get("messages_sha256"),
            f"invalid packet fallback changed at ordinal {row.get('ordinal')}",
        )
    prompts, _validated, population = _validate_prompt_rows(
        list(selected),
        expected_ordinals=selected_ordinals,
        label="sparse fact answer",
    )
    payload = {
        "all_question_plans": list(all_plans),
        "answer_provider_population_count": len(selected),
        "format": ANSWER_PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "invalid_packet_count": len(invalid),
        "invalid_packet_ordinals": list(invalid_ordinals),
        "local_invalid_fallback_count": len(invalid),
        "max_chat_prompt_tokens": HARD_PROMPT_TOKEN_CAP - ANSWER_OUTPUT_TOKEN_RESERVE,
        "max_concurrency": max_concurrency,
        "model": model,
        "new_compiler_provider_calls": 0,
        "observed_max_complete_envelope_tokens": max(
            row["prompt_token_proxy"] + ANSWER_OUTPUT_TOKEN_RESERVE
            for row in selected
        ),
        "original_ordinals": list(REMAINING_ORDINALS),
        "output_token_reserve": ANSWER_OUTPUT_TOKEN_RESERVE,
        "physical_prompt_rows": list(selected),
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "rematerialized_artifact_sha256": rematerialized.sha256,
        "rematerialized_replay_artifact_sha256": rematerialized_replay.sha256,
        "required_authorized_provider_calls": len(selected),
        "retained_transformer_token_state_bytes": 0,
        "selected_provider_ordinals": list(selected_ordinals),
        "selection_is_posthoc_outcome_conditioned": True,
        "valid_packet_count": len(selected),
    }
    assert_gold_blind(payload, path="typed_fact_sparse_answer_preflight")
    return payload, prompts


def _answer_preflight(args: argparse.Namespace) -> dict[str, Any]:
    rematerialized, replay, prompt_rows, result_rows = (
        _read_verified_rematerialization(args)
    )
    payload, _prompts = _answer_preflight_projection(
        rematerialized,
        replay,
        prompt_rows,
        result_rows,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(
        Path(args.output_root) / ANSWER_PREFLIGHT_NAME,
        payload,
    )
    return {
        "answer_preflight_sha256": artifact.sha256,
        "created": created,
        "invalid_packet_count": payload["invalid_packet_count"],
        "new_compiler_provider_calls": 0,
        "physical_provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "required_authorized_provider_calls": payload[
            "required_authorized_provider_calls"
        ],
        "valid_packet_count": payload["valid_packet_count"],
    }


def _validate_answer_preflight(
    artifact: SealedArtifact,
) -> tuple[
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    payload = artifact.payload
    assert_gold_blind(payload, path="typed_fact_sparse_answer_runtime_preflight")
    all_plans = payload.get("all_question_plans")
    selected_ordinals = payload.get("selected_provider_ordinals")
    invalid_ordinals = payload.get("invalid_packet_ordinals")
    _require(
        payload.get("format") == ANSWER_PREFLIGHT_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("new_compiler_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("hard_prompt_token_cap") == HARD_PROMPT_TOKEN_CAP
        and payload.get("max_chat_prompt_tokens")
        == HARD_PROMPT_TOKEN_CAP - ANSWER_OUTPUT_TOKEN_RESERVE
        and payload.get("output_token_reserve") == ANSWER_OUTPUT_TOKEN_RESERVE
        and payload.get("question_count") == SUBSET_QUESTION_COUNT
        and payload.get("original_ordinals") == list(REMAINING_ORDINALS)
        and type(all_plans) is list
        and len(all_plans) == SUBSET_QUESTION_COUNT
        and type(selected_ordinals) is list
        and type(invalid_ordinals) is list
        and len(selected_ordinals) == payload.get("valid_packet_count")
        == payload.get("answer_provider_population_count")
        == payload.get("required_authorized_provider_calls")
        and len(invalid_ordinals) == payload.get("invalid_packet_count")
        == payload.get("local_invalid_fallback_count")
        and len(selected_ordinals) + len(invalid_ordinals)
        == SUBSET_QUESTION_COUNT
        and 0 < len(selected_ordinals) < SUBSET_QUESTION_COUNT
        and sorted((*selected_ordinals, *invalid_ordinals))
        == list(REMAINING_ORDINALS)
        and type(payload.get("model")) is str
        and bool(payload.get("model"))
        and type(payload.get("gateway_url")) is str
        and bool(payload.get("gateway_url"))
        and type(payload.get("max_concurrency")) is int
        and int(payload["max_concurrency"]) > 0,
        "sparse answer preflight firewall/population changed",
    )
    require_sha256(
        payload.get("rematerialized_artifact_sha256"),
        "sparse answer rematerialized artifact",
    )
    require_sha256(
        payload.get("rematerialized_replay_artifact_sha256"),
        "sparse answer rematerialized replay",
    )
    assert type(all_plans) is list
    _all_prompts, all_validated, _all_population = _validate_prompt_rows(
        all_plans,
        expected_ordinals=REMAINING_ORDINALS,
        label="sparse all-question plan",
    )
    selected_by_ordinal = {
        row["ordinal"]: row
        for row in all_validated
        if row.get("fact_packet_valid") is True
    }
    invalid_by_ordinal = {
        row["ordinal"]: row
        for row in all_validated
        if row.get("fact_packet_valid") is False
    }
    _require(
        tuple(selected_by_ordinal) == tuple(selected_ordinals)
        and tuple(invalid_by_ordinal) == tuple(invalid_ordinals),
        "sparse answer validity ordinals changed",
    )
    for ordinal, row in selected_by_ordinal.items():
        packet = row.get("fact_packet")
        retained = packet.get("retained_handle_ids") if type(packet) is dict else None
        validation = row.get("validation_contract")
        preservation = row.get("preservation_requirements")
        _require(
            type(retained) is list
            and bool(retained)
            and row.get("compiled_handle_authority_enforced") is True
            and row.get("compiled_retained_handle_ids") == retained
            and row.get("allowed_handle_ids") == retained
            and set(row.get("handle_group_by_id", {})) == set(retained)
            and type(validation) is dict
            and set(validation.get("by_handle", {})) == set(retained)
            and type(preservation) is dict
            and set(preservation.get("by_handle", {})) <= set(retained),
            f"sparse valid handle authority changed at ordinal {ordinal}",
        )
    for ordinal, row in invalid_by_ordinal.items():
        source = row.get("source_prompt_plan")
        _require(
            type(source) is dict
            and row.get("compiled_handle_authority_enforced") is False
            and row.get("compiled_retained_handle_ids") == []
            and row.get("byte_identical_source_fallback") is True
            and row.get("messages") == source.get("messages")
            and row.get("messages_sha256") == source.get("messages_sha256"),
            f"sparse local fallback changed at ordinal {ordinal}",
        )
    expected_physical = [selected_by_ordinal[row] for row in selected_ordinals]
    _require(
        payload.get("physical_prompt_rows") == expected_physical,
        "sparse physical prompt projection changed",
    )
    prompts, physical_rows, population = _validate_prompt_rows(
        payload.get("physical_prompt_rows"),
        expected_ordinals=selected_ordinals,
        label="sparse fact answer",
    )
    _require(
        population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256"),
        "sparse answer prompt population receipt changed",
    )
    return prompts, physical_rows, all_validated


def _read_answer_preflight(
    root: Path,
    expected_sha256: str,
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    artifact = read_sealed_json(root / ANSWER_PREFLIGHT_NAME)
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "expected sparse answer preflight"),
        "sparse answer preflight SHA-256 changed",
    )
    prompts, physical, all_plans = _validate_answer_preflight(artifact)
    return artifact, prompts, physical, all_plans


def _runtime(
    artifact: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    output_root: Path,
    client: Any | None,
) -> FastCompletionRuntime:
    payload = artifact.payload
    required = payload["required_authorized_provider_calls"]
    return FastCompletionRuntime(
        checkpoint_dir=output_root / ANSWER_CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=payload["model"],
        client=client,
        max_prompt_tokens=payload["max_chat_prompt_tokens"],
        max_new_tokens=payload["output_token_reserve"],
        max_concurrency=payload["max_concurrency"],
        retries=0,
        benchmark_provenance={
            "arm": "locked_typed_fact_sparse_answer_remaining24_v3",
            "authorized_unique_calls": required,
            "experiment_format": ANSWER_RUN_FORMAT,
            "gateway_url": payload["gateway_url"],
            "gold_loaded": False,
            "invalid_packets_are_local_parent_fallbacks": True,
            "new_compiler_provider_calls": 0,
            "preflight_artifact_sha256": artifact.sha256,
            "rematerialized_artifact_sha256": payload[
                "rematerialized_artifact_sha256"
            ],
            "rematerialized_replay_artifact_sha256": payload[
                "rematerialized_replay_artifact_sha256"
            ],
        },
    )


def _checkpoint_batch(
    artifact: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    output_root: Path,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(
        artifact,
        prompts,
        output_root=output_root,
        client=client,
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


def _answer_provider(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root)
    artifact, prompts, _physical, _all = _read_answer_preflight(
        root,
        args.expected_preflight_sha256,
    )
    required = artifact.payload["required_authorized_provider_calls"]
    _require(
        required == artifact.payload["valid_packet_count"]
        and required < SUBSET_QUESTION_COUNT
        and args.enable_provider is True
        and args.authorized_provider_calls == required,
        f"sparse answer provider requires exact authorization for {required} calls",
    )
    load_dotenv()
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
    client = live._make_provider_client(  # noqa: SLF001
        api_key,
        artifact.payload["gateway_url"],
    )
    try:
        batch = _checkpoint_batch(
            artifact,
            prompts,
            output_root=root,
            client=client,
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    _require(
        batch.usage.logical_calls == batch.usage.unique_calls == required
        and batch.usage.physical_calls + batch.usage.checkpoint_hits == required,
        "sparse answer provider population changed",
    )
    return {
        "answer_provider_population_count": required,
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "invalid_local_fallback_count": artifact.payload["invalid_packet_count"],
        "new_compiler_provider_calls": 0,
        "physical_answer_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": required,
        "retained_transformer_token_state_bytes": 0,
    }


def _local_parent_result(plan: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        plan.get("fact_packet_valid") is False
        and plan.get("compiled_handle_authority_enforced") is False
        and plan.get("byte_identical_source_fallback") is True,
        "local parent fallback requires one invalid packet plan",
    )
    parent = require_text(plan.get("parent_prediction"), "local fallback parent")
    completion = json.dumps(
        {
            "decision": "keep_parent",
            "prediction": parent,
            "used_handle_ids": [],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    basis = {
        "compiler_result_row_sha256": require_sha256(
            plan.get("compiler_result_row_sha256"),
            "local fallback compiler result",
        ),
        "fact_packet_receipt_sha256": require_sha256(
            plan.get("fact_packet_receipt_sha256"),
            "local fallback fact packet",
        ),
        "format": LOCAL_FALLBACK_FORMAT,
        "ordinal": plan.get("ordinal"),
        "prompt_row_receipt_sha256": require_sha256(
            plan.get("prompt_row_receipt_sha256"),
            "local fallback plan",
        ),
        "provider_call_performed": False,
    }
    local_receipt = identity_sha256(basis)
    result = materialize_typed_final_result_row(
        plan,
        completion,
        completion_receipt_sha256=quote_sha256(completion),
        call_key_sha256=identity_sha256({**basis, "kind": "local-call"}),
        request_journal_sha256=identity_sha256(
            {**basis, "kind": "local-request-receipt"}
        ),
        response_journal_sha256=identity_sha256(
            {**basis, "kind": "local-response-receipt"}
        ),
    )
    _require(
        result.get("decision") == "keep_parent"
        and result.get("prediction") == parent
        and result.get("used_handle_ids") == [],
        "local invalid packet did not preserve protected parent",
    )
    body = dict(result)
    body.pop("source_row_sha256")
    body.update(
        {
            "local_fallback_receipt_sha256": local_receipt,
            "prediction_source": "typed_fact_invalid_packet_local_keep_parent_v3",
            "provider_call_performed": False,
            "validation_basis": "invalid_fact_packet_protected_parent",
        }
    )
    body["source_row_sha256"] = identity_sha256(body)
    assert_gold_blind(body, path="typed_fact_sparse_local_parent_result")
    return body


def _materialization_projection(
    preflight: SealedArtifact,
    physical_rows: tuple[dict[str, Any], ...],
    all_plans: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    required = preflight.payload["required_authorized_provider_calls"]
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == required
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == required
        and len(batch.unique_records) == required,
        "sparse answer materialization requires every selected checkpoint",
    )
    records = {row.messages_sha256: row for row in batch.unique_records}
    completion_by_ordinal = {
        row["ordinal"]: completion
        for row, completion in zip(
            physical_rows,
            batch.logical_completions,
            strict=True,
        )
    }
    results: list[dict[str, Any]] = []
    authority_rejections = 0
    provider_results = 0
    local_results = 0
    for plan in all_plans:
        ordinal = plan["ordinal"]
        if plan["fact_packet_valid"] is False:
            result = _local_parent_result(plan)
            local_results += 1
        else:
            completion = completion_by_ordinal.get(ordinal)
            record = records.get(plan["messages_sha256"])
            _require(
                type(completion) is str
                and record is not None
                and record.completion == completion
                and record.checkpoint_hit is True
                and record.physical_call is False,
                f"sparse answer checkpoint changed at ordinal {ordinal}",
            )
            assert type(completion) is str and record is not None
            result = materialize_typed_final_result_row(
                plan,
                completion,
                completion_receipt_sha256=record.completion_sha256,
                call_key_sha256=record.call_key_sha256,
                request_journal_sha256=record.request_journal_sha256,
                response_journal_sha256=record.response_journal_sha256,
            )
            retained = plan.get("compiled_retained_handle_ids")
            _require(
                type(retained) is list
                and bool(retained)
                and plan.get("allowed_handle_ids") == retained
                and set(result.get("used_handle_ids", ())) <= set(retained),
                f"sparse materialized answer escaped handles at ordinal {ordinal}",
            )
            declared = v2_cli._declared_completion_handles(completion)  # noqa: SLF001
            if declared is not None and not set(declared) <= set(retained):
                authority_rejections += 1
                _require(
                    result.get("decision") == "invalid_keep_parent"
                    and result.get("parse_error_code") == "unknown_handle"
                    and result.get("used_handle_ids") == [],
                    f"sparse outside-packet handle accepted at ordinal {ordinal}",
                )
            provider_results += 1
        results.append(result)
    _require(
        provider_results == required
        and local_results == preflight.payload["invalid_packet_count"]
        and len(results) == SUBSET_QUESTION_COUNT,
        "sparse answer result partition changed",
    )
    judge_rows = [judge_row_projection(row) for row in results]
    payload = {
        "answer_preflight_artifact_sha256": preflight.sha256,
        "answer_provider_population_count": required,
        "changed_prediction_count": sum(
            bool(row["changed_from_parent"]) for row in results
        ),
        "compiled_handle_authority_rejection_count": authority_rejections,
        "completion_batch": batch.model_dump(),
        "format": ANSWER_RUN_FORMAT,
        "gold_loaded": False,
        "historical_physical_answer_call_count": required,
        "invalid_packet_count": local_results,
        "judge_rows": judge_rows,
        "local_invalid_fallback_count": local_results,
        "new_compiler_provider_calls": 0,
        "original_ordinals": list(REMAINING_ORDINALS),
        "physical_provider_calls_during_materialization": 0,
        "questions": results,
        "question_count": SUBSET_QUESTION_COUNT,
        "rematerialized_artifact_sha256": preflight.payload[
            "rematerialized_artifact_sha256"
        ],
        "rematerialized_replay_artifact_sha256": preflight.payload[
            "rematerialized_replay_artifact_sha256"
        ],
        "required_authorized_provider_calls": required,
        "retained_transformer_token_state_bytes": 0,
        "selected_provider_ordinals": list(
            preflight.payload["selected_provider_ordinals"]
        ),
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    assert_gold_blind(payload, path="typed_fact_sparse_answer_run")
    return payload


def _answer_materialize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root)
    preflight, prompts, physical, all_plans = _read_answer_preflight(
        root,
        args.expected_preflight_sha256,
    )
    batch = _checkpoint_batch(preflight, prompts, output_root=root, client=None)
    payload = _materialization_projection(
        preflight,
        physical,
        all_plans,
        batch,
    )
    artifact, created = publish_sealed_json(root / ANSWER_RUN_NAME, payload)
    return {
        "answer_run_sha256": artifact.sha256,
        "created": created,
        "historical_physical_answer_call_count": payload[
            "historical_physical_answer_call_count"
        ],
        "local_invalid_fallback_count": payload["local_invalid_fallback_count"],
        "new_compiler_provider_calls": 0,
        "physical_provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
    }


def _validate_answer_run(
    artifact: SealedArtifact,
    preflight: SealedArtifact,
    physical_rows: Sequence[Mapping[str, Any]],
    all_plans: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    assert_gold_blind(payload, path="verified_typed_fact_sparse_answer_run")
    questions = payload.get("questions")
    judge_rows = payload.get("judge_rows")
    completion_batch = payload.get("completion_batch")
    logical = (
        completion_batch.get("logical_completions")
        if type(completion_batch) is dict
        else None
    )
    required = preflight.payload["required_authorized_provider_calls"]
    _require(
        payload.get("format") == ANSWER_RUN_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("new_compiler_provider_calls") == 0
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == SUBSET_QUESTION_COUNT
        and payload.get("original_ordinals") == list(REMAINING_ORDINALS)
        and payload.get("answer_preflight_artifact_sha256") == preflight.sha256
        and payload.get("rematerialized_artifact_sha256")
        == preflight.payload["rematerialized_artifact_sha256"]
        and payload.get("rematerialized_replay_artifact_sha256")
        == preflight.payload["rematerialized_replay_artifact_sha256"]
        and payload.get("answer_provider_population_count") == required
        and payload.get("historical_physical_answer_call_count") == required
        and payload.get("required_authorized_provider_calls") == required
        and payload.get("selected_provider_ordinals")
        == preflight.payload["selected_provider_ordinals"]
        and payload.get("invalid_packet_count")
        == payload.get("local_invalid_fallback_count")
        == preflight.payload["invalid_packet_count"]
        and type(questions) is list
        and type(judge_rows) is list
        and len(questions) == len(judge_rows) == SUBSET_QUESTION_COUNT
        and type(logical) is list
        and len(logical) == required,
        "sparse answer run envelope changed",
    )
    completion_by_ordinal = {
        row["ordinal"]: completion
        for row, completion in zip(physical_rows, logical, strict=True)
    }
    verified: list[dict[str, Any]] = []
    authority_rejections = 0
    local_count = 0
    assert type(questions) is list and type(judge_rows) is list
    for ordinal, source, judge, plan in zip(
        REMAINING_ORDINALS,
        questions,
        judge_rows,
        all_plans,
        strict=True,
    ):
        _require(
            type(source) is dict and type(judge) is dict,
            "sparse answer result row changed type",
        )
        assert type(source) is dict and type(judge) is dict
        unsigned = dict(source)
        declared = unsigned.pop("source_row_sha256", None)
        _require(
            declared == identity_sha256(unsigned)
            and source.get("format") == RESULT_ROW_FORMAT
            and source.get("ordinal") == ordinal
            and plan.get("ordinal") == ordinal
            and source.get("question_id") == plan.get("question_id")
            and source.get("question_sha256") == plan.get("question_sha256")
            and source.get("prompt_row_receipt_sha256")
            == plan.get("prompt_row_receipt_sha256")
            and judge_row_projection(source) == judge,
            f"sparse answer result binding changed at ordinal {ordinal}",
        )
        if plan.get("fact_packet_valid") is False:
            local_count += 1
            _require(
                source == _local_parent_result(plan)
                and source.get("provider_call_performed") is False
                and source.get("decision") == "keep_parent"
                and source.get("prediction") == plan.get("parent_prediction")
                and source.get("used_handle_ids") == [],
                f"sparse local parent result changed at ordinal {ordinal}",
            )
        else:
            retained = plan.get("compiled_retained_handle_ids")
            completion = completion_by_ordinal.get(ordinal)
            _require(
                type(retained) is list
                and bool(retained)
                and type(completion) is str
                and set(source.get("used_handle_ids", ())) <= set(retained),
                f"sparse verified answer escaped handles at ordinal {ordinal}",
            )
            assert type(completion) is str
            used = v2_cli._declared_completion_handles(completion)  # noqa: SLF001
            if used is not None and not set(used) <= set(retained):
                authority_rejections += 1
                _require(
                    source.get("decision") == "invalid_keep_parent"
                    and source.get("parse_error_code") == "unknown_handle"
                    and source.get("used_handle_ids") == [],
                    f"sparse verified outside handle accepted at ordinal {ordinal}",
                )
        verified.append(dict(judge))
    _require(
        local_count == payload.get("local_invalid_fallback_count")
        and authority_rejections
        == payload.get("compiled_handle_authority_rejection_count"),
        "sparse result partition/rejection accounting changed",
    )
    return tuple(verified)


def _answer_replay(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root)
    preflight, prompts, physical, all_plans = _read_answer_preflight(
        root,
        args.expected_preflight_sha256,
    )
    batch = _checkpoint_batch(preflight, prompts, output_root=root, client=None)
    replayed = _materialization_projection(
        preflight,
        physical,
        all_plans,
        batch,
    )
    run = read_sealed_json(root / ANSWER_RUN_NAME)
    _require(
        run.sha256 == require_sha256(args.expected_run_sha256, "expected sparse run")
        and run.payload == replayed,
        "sparse answer run differs from checkpoint-only replay",
    )
    _validate_answer_run(run, preflight, physical, all_plans)
    payload = {
        "answer_preflight_artifact_sha256": preflight.sha256,
        "byte_identical": True,
        "expected_run_sha256": run.sha256,
        "format": ANSWER_REPLAY_FORMAT,
        "gold_loaded": False,
        "historical_physical_answer_call_count": preflight.payload[
            "valid_packet_count"
        ],
        "local_invalid_fallback_count": preflight.payload[
            "invalid_packet_count"
        ],
        "new_compiler_provider_calls": 0,
        "physical_provider_calls": 0,
        "question_count": SUBSET_QUESTION_COUNT,
        "replayed_run_sha256": run.sha256,
        "retained_transformer_token_state_bytes": 0,
    }
    assert_gold_blind(payload, path="typed_fact_sparse_answer_replay")
    artifact, created = publish_sealed_json(root / ANSWER_REPLAY_NAME, payload)
    return {
        "answer_replay_sha256": artifact.sha256,
        "answer_run_sha256": run.sha256,
        "byte_identical": True,
        "created": created,
        "new_compiler_provider_calls": 0,
        "physical_provider_calls": 0,
    }


def read_verified_sparse_answer_run(
    output_root: str | Path,
    *,
    expected_preflight_sha256: str,
    expected_run_sha256: str,
    expected_replay_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    root = Path(output_root)
    preflight, _prompts, physical, all_plans = _read_answer_preflight(
        root,
        expected_preflight_sha256,
    )
    run = read_sealed_json(root / ANSWER_RUN_NAME)
    _require(
        run.sha256 == require_sha256(expected_run_sha256, "expected sparse run"),
        "sparse answer run SHA-256 changed",
    )
    judge_rows = _validate_answer_run(run, preflight, physical, all_plans)
    replay = read_sealed_json(root / ANSWER_REPLAY_NAME)
    payload = replay.payload
    _require(
        replay.sha256
        == require_sha256(expected_replay_sha256, "expected sparse replay")
        and payload.get("format") == ANSWER_REPLAY_FORMAT
        and payload.get("byte_identical") is True
        and payload.get("gold_loaded") is False
        and payload.get("new_compiler_provider_calls") == 0
        and payload.get("physical_provider_calls") == 0
        and payload.get("question_count") == SUBSET_QUESTION_COUNT
        and payload.get("answer_preflight_artifact_sha256") == preflight.sha256
        and payload.get("expected_run_sha256") == run.sha256
        and payload.get("replayed_run_sha256") == run.sha256,
        "sparse answer replay binding changed",
    )
    return run, replay, judge_rows


def _add_output_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)


def _add_legacy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument(
        "--expected-v2-preflight-sha256",
        type=_sha256_argument,
        default=EXPECTED_V2_PREFLIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-v2-run-sha256",
        type=_sha256_argument,
        default=EXPECTED_V2_RUN_SHA256,
    )
    parser.add_argument(
        "--expected-v2-replay-sha256",
        type=_sha256_argument,
        default=EXPECTED_V2_REPLAY_SHA256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    rematerialize = commands.add_parser(
        "compiler-rematerialize",
        help="reparse the exact sealed v2 completion journals with zero calls",
    )
    _add_output_root(rematerialize)
    _add_legacy_args(rematerialize)

    rematerialize_replay = commands.add_parser(
        "compiler-rematerialize-replay",
        help="prove v3 parser rematerialization is checkpoint-byte-identical",
    )
    _add_output_root(rematerialize_replay)
    _add_legacy_args(rematerialize_replay)
    rematerialize_replay.add_argument(
        "--expected-rematerialized-sha256",
        type=_sha256_argument,
        required=True,
    )

    answer_preflight = commands.add_parser(
        "answer-preflight",
        help="seal only valid fact packets into the answer provider population",
    )
    _add_output_root(answer_preflight)
    _add_legacy_args(answer_preflight)
    answer_preflight.add_argument(
        "--expected-rematerialized-sha256",
        type=_sha256_argument,
        required=True,
    )
    answer_preflight.add_argument(
        "--expected-rematerialized-replay-sha256",
        type=_sha256_argument,
        required=True,
    )
    answer_preflight.add_argument("--model", default=live.DEFAULT_TERRA_GATEWAY_MODEL)
    answer_preflight.add_argument(
        "--gateway-url", default=live.DEFAULT_GATEWAY_URL
    )
    answer_preflight.add_argument("--max-concurrency", type=int, default=4)

    provider = commands.add_parser(
        "answer-provider-run",
        help="execute exactly valid_packet_count sparse answer calls",
    )
    _add_output_root(provider)
    provider.add_argument(
        "--expected-preflight-sha256",
        type=_sha256_argument,
        required=True,
    )
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser(
        "answer-materialize",
        help="merge selected completions with local invalid-packet parents",
    )
    _add_output_root(materialize)
    materialize.add_argument(
        "--expected-preflight-sha256",
        type=_sha256_argument,
        required=True,
    )

    replay = commands.add_parser(
        "answer-replay",
        help="prove the complete 24-row sparse answer run is byte-identical",
    )
    _add_output_root(replay)
    replay.add_argument(
        "--expected-preflight-sha256",
        type=_sha256_argument,
        required=True,
    )
    replay.add_argument(
        "--expected-run-sha256",
        type=_sha256_argument,
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    commands: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
        "compiler-rematerialize": _compiler_rematerialize,
        "compiler-rematerialize-replay": _compiler_rematerialize_replay,
        "answer-preflight": _answer_preflight,
        "answer-provider-run": _answer_provider,
        "answer-materialize": _answer_materialize,
        "answer-replay": _answer_replay,
    }
    result = commands[args.command](args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANSWER_CHECKPOINT_DIR_NAME",
    "ANSWER_PREFLIGHT_NAME",
    "ANSWER_REPLAY_NAME",
    "ANSWER_RUN_NAME",
    "DEFAULT_OUTPUT",
    "DEFAULT_V2_ROOT",
    "EXPECTED_V2_PREFLIGHT_SHA256",
    "EXPECTED_V2_REPLAY_SHA256",
    "EXPECTED_V2_RUN_SHA256",
    "FORMAT",
    "LockedTypedFactCompilerSparseError",
    "REMATERIALIZED_NAME",
    "REMATERIALIZED_REPLAY_NAME",
    "REMAINING_ORDINALS",
    "SUBSET_QUESTION_COUNT",
    "main",
    "read_verified_sparse_answer_run",
]
