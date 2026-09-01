#!/usr/bin/env python3
"""Judge the replay-verified exact-11 terminal answers with locked Sol calls.

The Terra run and replay are authenticated before benchmark gold is opened.
Only their stable 11-row judge seam is paired with locked question/reference
rows.  Generic typed-final judging owns prompt construction, binary parsing,
scoring, and materialization; this adapter supplies the selected-subset source
boundary and a resumable, exactly authorized checkpoint lifecycle.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
)
from tools import run_locked_semantic_global_terminal_answer as answer_cli  # noqa: E402
from tools import (  # noqa: E402
    revalidate_locked_semantic_global_terminal_answer_v4 as validator_v4_cli,
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
)
from tools.matched_eval.typed_memory_final_judging import (  # noqa: E402
    CHECKPOINT_DIR_NAME,
    DEFAULT_MAX_PROMPT_TOKENS,
    JUDGE_FORMAT,
    JUDGE_MAX_TOKENS,
    JUDGE_NAME,
    PREFLIGHT_FORMAT,
    PREFLIGHT_NAME,
    REPLAY_NAME,
    SCORE_FORMAT,
    SCORE_NAME,
    SCORE_REPLAY_NAME,
    TypedFinalJudgeGoldRow,
    load_locked_typed_final_gold,
    materialization_projection,
    preflight_projection,
    validate_preflight_artifact,
)
from tools.run_locked_query_answer_judge import DEFAULT_DATASET  # noqa: E402
from tools.run_matched_eval_spine import DEFAULT_SPLIT  # noqa: E402


FORMAT = "memory-condense-locked-semantic-global-terminal-sol-judge-v2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWER_ROOT = answer_cli.DEFAULT_OUTPUT_ROOT
DEFAULT_JUDGE_ROOT = DEFAULT_ANSWER_ROOT / "sol-judge-v2"
DEFAULT_MODEL = judging.DEFAULT_SOL_GATEWAY_MODEL
DEFAULT_GATEWAY_URL = live.DEFAULT_GATEWAY_URL
DEFAULT_MAX_CONCURRENCY = 4
EXACT_ORDINALS = answer_cli.EXACT_ORDINALS
SELECTED_QUESTION_COUNT = len(EXACT_ORDINALS)
POSTSEAL_BINDING_KEYS = answer_cli.POSTSEAL_BINDING_KEYS


class LockedSemanticGlobalTerminalJudgeError(MatchedEvalContractError):
    """A Terra source, gold binding, Sol journal, verdict, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticGlobalTerminalJudgeError(message)


def _validated_postseal_binding(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        all(key in payload for key in POSTSEAL_BINDING_KEYS),
        "terminal Sol source omitted promoted post-seal binding",
    )
    binding = {key: payload[key] for key in POSTSEAL_BINDING_KEYS}
    for key in (
        "postseal_promotion_audit_artifact_sha256",
        "postseal_promotion_audit_identity_sha256",
        "postseal_semantic_atom_manifest_artifact_sha256",
        "postseal_semantic_atom_manifest_identity_sha256",
        "postseal_semantic_atom_population_sha256",
    ):
        require_sha256(binding[key], f"terminal Sol {key}")
    _require(
        binding["postseal_semantic_atom_count"]
        == binding["postseal_semantic_atom_final_usable_count"]
        == answer_cli.postseal_cli.SEMANTIC_ATOM_COUNT
        and binding["postseal_semantic_atom_manifest_artifact_sha256"]
        == answer_cli.postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256
        and binding["postseal_semantic_atom_manifest_identity_sha256"]
        == answer_cli.postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_IDENTITY_SHA256
        and binding["postseal_semantic_atom_population_sha256"]
        == answer_cli.postseal_cli.DEFAULT_SEMANTIC_ATOM_POPULATION_SHA256
        and binding["postseal_source_target_count"]
        == answer_cli.postseal_cli.SOURCE_TARGET_COUNT
        and type(binding["postseal_source_final_usable_count"]) is int
        and 0
        <= binding["postseal_source_final_usable_count"]
        <= answer_cli.postseal_cli.SOURCE_TARGET_COUNT
        and binding["postseal_witness_positive_count"]
        == answer_cli.postseal_cli.POSITIVE_WITNESS_COUNT
        and type(binding["postseal_witness_final_usable_count"]) is int
        and 0
        <= binding["postseal_witness_final_usable_count"]
        <= answer_cli.postseal_cli.POSITIVE_WITNESS_COUNT
        and binding["postseal_witness_manifest_artifact_sha256"]
        == answer_cli.postseal_cli.DEFAULT_WITNESS_MANIFEST_SHA256
        and binding["postseal_witness_manifest_identity_sha256"]
        == answer_cli.postseal_cli.DEFAULT_WITNESS_MANIFEST_IDENTITY_SHA256
        and binding["postseal_target_plan_artifact_sha256"]
        == answer_cli.postseal_cli.DEFAULT_TARGET_PLAN_SHA256
        and binding["postseal_target_plan_identity_sha256"]
        == answer_cli.postseal_cli.DEFAULT_TARGET_PLAN_IDENTITY_SHA256,
        "terminal Sol post-seal promotion binding changed",
    )
    return binding


def _validate_source_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    result = tuple(dict(row) for row in rows)
    _require(
        len(result) == SELECTED_QUESTION_COUNT
        and tuple(row.get("ordinal") for row in result) == EXACT_ORDINALS
        and len({row.get("question_id") for row in result})
        == SELECTED_QUESTION_COUNT,
        "terminal Sol source population/order changed",
    )
    for ordinal, row in zip(EXACT_ORDINALS, result, strict=True):
        prediction = row.get("prediction")
        _require(
            row.get("ordinal") == ordinal
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
            f"terminal Sol source row {ordinal} changed",
        )
        for key in (
            "dated_question_sha256",
            "parent_prediction_sha256",
            "prediction_sha256",
            "question_sha256",
            "source_row_sha256",
        ):
            require_sha256(row.get(key), f"terminal Sol source {key}")
    return result


def build_preflight_payload(
    run: SealedArtifact,
    replay: SealedArtifact,
    source_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[TypedFinalJudgeGoldRow],
    *,
    gold_population_sha256: str,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, str], ...], ...]]:
    source = _validate_source_rows(source_rows)
    gold = tuple(gold_rows)
    postseal_binding = _validated_postseal_binding(run.payload)
    _require(
        model == DEFAULT_MODEL
        and gateway_url == DEFAULT_GATEWAY_URL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "terminal Sol runtime policy changed",
    )
    payload, prompts = preflight_projection(
        run_artifact=run,
        replay_artifact_sha256=replay.sha256,
        source_rows=source,
        gold_rows=gold,
        gold_population_sha256=gold_population_sha256,
        mode="selected_subset",
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
    )
    payload = {
        **payload,
        **postseal_binding,
    }
    raw_rows = payload.get("prompt_rows")
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("judge_mode") == "selected_subset"
        and payload.get("selected_question_count") == SELECTED_QUESTION_COUNT
        and payload.get("required_authorized_provider_calls")
        == SELECTED_QUESTION_COUNT
        and type(raw_rows) is list
        and tuple(row.get("ordinal") for row in raw_rows) == EXACT_ORDINALS,
        "terminal Sol selected-subset prompt projection changed",
    )
    return payload, prompts


def _validate_preflight(
    artifact: SealedArtifact,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    prompts, rows = validate_preflight_artifact(artifact)
    payload = artifact.payload
    _validated_postseal_binding(payload)
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("judge_mode") == "selected_subset"
        and payload.get("gold_loaded") is True
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("gateway_url") == DEFAULT_GATEWAY_URL
        and payload.get("selected_question_count") == SELECTED_QUESTION_COUNT
        and payload.get("required_authorized_provider_calls")
        == SELECTED_QUESTION_COUNT
        and len(prompts) == len(rows) == SELECTED_QUESTION_COUNT
        and tuple(row.get("ordinal") for row in rows) == EXACT_ORDINALS,
        "terminal Sol sealed preflight changed",
    )
    return prompts, rows


def _read_preflight(
    root: Path,
    expected_sha256: str,
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    artifact = read_sealed_json(root / PREFLIGHT_NAME)
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "terminal Sol preflight"),
        "terminal Sol preflight artifact changed",
    )
    prompts, rows = _validate_preflight(artifact)
    return artifact, prompts, rows


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.judge_output_root)
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "terminal Sol preflight requires a fresh absent checkpoint root",
    )
    # Terra run/replay authentication deliberately precedes dataset access.
    validator_root = getattr(args, "answer_validator_v4_root", None)
    validator_run_sha = getattr(
        args, "expected_answer_validator_v4_run_sha256", None
    )
    validator_replay_sha = getattr(
        args, "expected_answer_validator_v4_replay_sha256", None
    )
    validator_values = (
        validator_root,
        validator_run_sha,
        validator_replay_sha,
    )
    _require(
        all(value is None for value in validator_values)
        or all(value is not None for value in validator_values),
        "terminal Sol v4 answer source requires root, run, and replay together",
    )
    if validator_root is None:
        run, replay, source_rows = answer_cli.load_verified_answer_run(
            args.answer_root,
            expected_preflight_sha256=str(
                args.expected_answer_preflight_sha256
            ),
            expected_run_sha256=str(args.expected_answer_run_sha256),
            expected_replay_sha256=str(args.expected_answer_replay_sha256),
            postseal_audit=args.postseal_audit,
            expected_postseal_audit_sha256=str(
                args.expected_postseal_audit_sha256
            ),
        )
    else:
        run, replay, source_rows = (
            validator_v4_cli.load_verified_revalidated_answer_run(
                validator_root,
                answer_root=args.answer_root,
                expected_answer_preflight_sha256=str(
                    args.expected_answer_preflight_sha256
                ),
                expected_answer_run_sha256=str(
                    args.expected_answer_run_sha256
                ),
                expected_answer_replay_sha256=str(
                    args.expected_answer_replay_sha256
                ),
                postseal_audit=args.postseal_audit,
                expected_postseal_audit_sha256=str(
                    args.expected_postseal_audit_sha256
                ),
                expected_validator_run_sha256=str(validator_run_sha),
                expected_validator_replay_sha256=str(validator_replay_sha),
            )
        )
    source = _validate_source_rows(source_rows)
    _validated_postseal_binding(run.payload)
    gold_rows, gold_sha = load_locked_typed_final_gold(
        dataset_path=args.dataset,
        split_path=args.split,
        source_rows=source,
        allow_subset=True,
    )
    payload, _ = build_preflight_payload(
        run,
        replay,
        source,
        gold_rows,
        gold_population_sha256=gold_sha,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(output_root / PREFLIGHT_NAME, payload)
    return {
        "answer_replay_sha256": replay.sha256,
        "answer_run_sha256": run.sha256,
        "created": created,
        "judge_mode": "selected_subset",
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": SELECTED_QUESTION_COUNT,
        "selected_ordinals": list(EXACT_ORDINALS),
        "selected_question_count": SELECTED_QUESTION_COUNT,
    }


def _runtime(
    preflight: SealedArtifact,
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
        and len(prompts) == SELECTED_QUESTION_COUNT,
        "terminal Sol runtime differs from sealed preflight",
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
            "arm": "locked_common_typed_memory_final_sol_judge_v1",
            "authorized_unique_calls": len(prompts),
            "experiment_format": JUDGE_FORMAT,
            "judge_mode": preflight.payload["judge_mode"],
            "preflight_artifact_sha256": preflight.sha256,
            **{
                key: preflight.payload[key]
                for key in POSTSEAL_BINDING_KEYS
            },
            "typed_final_run_sha256": preflight.payload[
                "typed_final_run_sha256"
            ],
        },
    )


def _checkpoint_batch(
    preflight: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(preflight, prompts, args=args, client=client)
    try:
        return runtime.run()
    finally:
        runtime.close()


def _validated_checkpoint_hits(
    preflight: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
) -> int:
    runtime = _runtime(preflight, prompts, args=args, client=None)
    try:
        with runtime._journal_guard():  # noqa: SLF001 - runtime owns journals
            records = runtime._load_all_records()  # noqa: SLF001
    finally:
        runtime.close()
    _require(
        len(records) <= SELECTED_QUESTION_COUNT,
        "terminal Sol checkpoint population escaped exact11",
    )
    return len(records)


def _materialization_projection_with_postseal_binding(
    preflight: SealedArtifact,
    rows: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    judge_payload, score_payload = materialization_projection(
        preflight, rows, batch
    )
    binding = _validated_postseal_binding(preflight.payload)
    return ({**judge_payload, **binding}, {**score_payload, **binding})


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, _ = _read_preflight(
        Path(args.judge_output_root), str(args.expected_judge_preflight_sha256)
    )
    _require(
        args.enable_provider is True
        and type(args.authorized_provider_calls) is int
        and 0 <= args.authorized_provider_calls <= SELECTED_QUESTION_COUNT,
        "terminal Sol provider requires a bounded call authorization",
    )
    checkpoint_hits = _validated_checkpoint_hits(
        preflight, prompts, args=args
    )
    remaining = SELECTED_QUESTION_COUNT - checkpoint_hits
    _require(
        args.authorized_provider_calls == remaining,
        "terminal Sol authorization must exactly equal remaining calls",
    )
    if remaining == 0:
        batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
        _require(
            batch.usage.logical_calls
            == batch.usage.unique_calls
            == batch.usage.checkpoint_hits
            == SELECTED_QUESTION_COUNT
            and batch.usage.physical_calls == 0,
            "terminal Sol completed checkpoint replay changed",
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
                preflight, prompts, args=args, client=client
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        _require(
            batch.usage.logical_calls
            == batch.usage.unique_calls
            == SELECTED_QUESTION_COUNT
            and batch.usage.physical_calls + batch.usage.checkpoint_hits
            == SELECTED_QUESTION_COUNT
            and batch.usage.physical_calls <= args.authorized_provider_calls
            and batch.usage.checkpoint_hits >= checkpoint_hits,
            "terminal Sol provider population changed",
        )
    return {
        "authorized_remaining_provider_calls": remaining,
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": preflight.sha256,
        "required_authorized_provider_calls": remaining,
        "retained_transformer_token_state_bytes": 0,
    }


def _validate_materialization(
    preflight: SealedArtifact,
    judge_payload: Mapping[str, Any],
    score_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    raw_rows = judge_payload.get("questions")
    aggregate = judge_payload.get("aggregate")
    _require(
        judge_payload.get("format") == JUDGE_FORMAT
        and judge_payload.get("gold_loaded") is True
        and judge_payload.get("judge_mode") == "selected_subset"
        and judge_payload.get("physical_provider_calls_during_materialization")
        == 0
        and judge_payload.get("retained_transformer_token_state_bytes") == 0
        and judge_payload.get("selected_question_count")
        == SELECTED_QUESTION_COUNT
        and type(raw_rows) is list
        and len(raw_rows) == SELECTED_QUESTION_COUNT
        and type(aggregate) is dict
        and aggregate.get("question_count") == SELECTED_QUESTION_COUNT
        and score_payload.get("format") == SCORE_FORMAT
        and score_payload.get("judge_mode") == "selected_subset"
        and score_payload.get("selected_question_count")
        == SELECTED_QUESTION_COUNT
        and score_payload.get("typed_final_run_sha256")
        == judge_payload.get("typed_final_run_sha256")
        == preflight.payload.get("typed_final_run_sha256")
        and all(
            judge_payload.get(key)
            == score_payload.get(key)
            == preflight.payload.get(key)
            for key in POSTSEAL_BINDING_KEYS
        ),
        "terminal Sol materialization envelope changed",
    )
    prompt_by_ordinal = {
        row["ordinal"]: row for row in preflight.payload["prompt_rows"]
    }
    rows: list[dict[str, Any]] = []
    for ordinal, raw in zip(EXACT_ORDINALS, raw_rows, strict=True):
        _require(type(raw) is dict, "terminal Sol verdict row changed type")
        body = dict(raw)
        declared = body.pop("judge_row_sha256", None)
        prompt = prompt_by_ordinal.get(ordinal)
        _require(
            declared == identity_sha256(body)
            and raw.get("ordinal") == ordinal
            and type(raw.get("correct")) is bool
            and prompt is not None
            and raw.get("messages_sha256") == prompt.get("messages_sha256")
            and raw.get("prediction_sha256") == prompt.get("prediction_sha256")
            and raw.get("reference_sha256") == prompt.get("reference_sha256")
            and raw.get("source_row_sha256") == prompt.get("source_row_sha256"),
            f"terminal Sol verdict row {ordinal} changed",
        )
        rows.append(dict(raw))
    correct = sum(bool(row["correct"]) for row in rows)
    _require(
        aggregate.get("correct") == correct
        and aggregate.get("accuracy") == correct / SELECTED_QUESTION_COUNT
        and score_payload.get("correct") == correct
        and score_payload.get("selected_accuracy")
        == correct / SELECTED_QUESTION_COUNT,
        "terminal Sol score arithmetic changed",
    )
    return tuple(rows)


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, rows = _read_preflight(
        Path(args.judge_output_root), str(args.expected_judge_preflight_sha256)
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    judge_payload, score_payload = _materialization_projection_with_postseal_binding(
        preflight, rows, batch
    )
    _validate_materialization(preflight, judge_payload, score_payload)
    root = Path(args.judge_output_root)
    judge_artifact, judge_created = publish_sealed_json(
        root / JUDGE_NAME, judge_payload
    )
    score_artifact, score_created = publish_sealed_json(
        root / SCORE_NAME, score_payload
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "correct": score_payload["correct"],
        "judge_created": judge_created,
        "judge_sha256": judge_artifact.sha256,
        "physical_provider_calls": 0,
        "score_created": score_created,
        "score_sha256": score_artifact.sha256,
        "selected_question_count": SELECTED_QUESTION_COUNT,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, rows = _read_preflight(
        Path(args.judge_output_root), str(args.expected_judge_preflight_sha256)
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    judge_payload, score_payload = _materialization_projection_with_postseal_binding(
        preflight, rows, batch
    )
    _validate_materialization(preflight, judge_payload, score_payload)
    root = Path(args.judge_output_root)
    judge_artifact = read_sealed_json(root / JUDGE_NAME)
    score_artifact = read_sealed_json(root / SCORE_NAME)
    _require(
        judge_artifact.sha256
        == require_sha256(args.expected_judge_sha256, "terminal Sol judge")
        and score_artifact.sha256
        == require_sha256(args.expected_score_sha256, "terminal Sol score")
        and judge_artifact.payload == judge_payload
        and score_artifact.payload == score_payload,
        "terminal Sol materialization differs from checkpoint replay",
    )
    judge_replay, _ = publish_sealed_json(root / REPLAY_NAME, judge_payload)
    score_replay, _ = publish_sealed_json(
        root / SCORE_REPLAY_NAME, score_payload
    )
    _require(
        judge_replay.sha256 == judge_artifact.sha256
        and score_replay.sha256 == score_artifact.sha256,
        "terminal Sol replay is not byte-identical",
    )
    return {
        "byte_identical": True,
        "judge_replay_sha256": judge_replay.sha256,
        "physical_provider_calls": 0,
        "score_replay_sha256": score_replay.sha256,
    }


def load_verified_judge_run(
    judge_output_root: str | Path,
    *,
    expected_preflight_sha256: str,
    expected_judge_sha256: str,
    expected_score_sha256: str,
    expected_judge_replay_sha256: str,
    expected_score_replay_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    """Return exact verdict rows after byte-identical judge/score replay."""

    root = Path(judge_output_root)
    preflight, _, _ = _read_preflight(root, expected_preflight_sha256)
    judge_artifact = read_sealed_json(root / JUDGE_NAME)
    score_artifact = read_sealed_json(root / SCORE_NAME)
    _require(
        judge_artifact.sha256
        == require_sha256(expected_judge_sha256, "terminal Sol judge")
        and score_artifact.sha256
        == require_sha256(expected_score_sha256, "terminal Sol score"),
        "terminal Sol judge/score artifacts changed",
    )
    rows = _validate_materialization(
        preflight, judge_artifact.payload, score_artifact.payload
    )
    judge_replay = read_sealed_json(root / REPLAY_NAME)
    score_replay = read_sealed_json(root / SCORE_REPLAY_NAME)
    _require(
        judge_replay.sha256
        == require_sha256(
            expected_judge_replay_sha256, "terminal Sol judge replay"
        )
        == judge_artifact.sha256
        and score_replay.sha256
        == require_sha256(
            expected_score_replay_sha256, "terminal Sol score replay"
        )
        == score_artifact.sha256
        and judge_replay.payload == judge_artifact.payload
        and score_replay.payload == score_artifact.payload,
        "terminal Sol judge/score replay changed",
    )
    return judge_artifact, score_artifact, rows


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--judge-output-root", type=Path, default=DEFAULT_JUDGE_ROOT
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument(
        "--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _add_runtime(preflight)
    preflight.add_argument(
        "--answer-root", type=Path, default=DEFAULT_ANSWER_ROOT
    )
    preflight.add_argument("--expected-answer-preflight-sha256", required=True)
    preflight.add_argument("--expected-answer-run-sha256", required=True)
    preflight.add_argument("--expected-answer-replay-sha256", required=True)
    preflight.add_argument("--answer-validator-v4-root", type=Path)
    preflight.add_argument("--expected-answer-validator-v4-run-sha256")
    preflight.add_argument("--expected-answer-validator-v4-replay-sha256")
    preflight.add_argument("--postseal-audit", type=Path, required=True)
    preflight.add_argument("--expected-postseal-audit-sha256", required=True)
    preflight.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    preflight.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    provider = commands.add_parser("provider-run")
    _add_runtime(provider)
    provider.add_argument("--expected-judge-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)
    materialize = commands.add_parser("materialize")
    _add_runtime(materialize)
    materialize.add_argument("--expected-judge-preflight-sha256", required=True)
    replay = commands.add_parser("replay")
    _add_runtime(replay)
    replay.add_argument("--expected-judge-preflight-sha256", required=True)
    replay.add_argument("--expected-judge-sha256", required=True)
    replay.add_argument("--expected-score-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args)
    elif args.command == "provider-run":
        result = run_provider(args)
    elif args.command == "materialize":
        result = run_materialize(args)
    else:
        result = run_replay(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_ANSWER_ROOT",
    "DEFAULT_JUDGE_ROOT",
    "EXACT_ORDINALS",
    "FORMAT",
    "JUDGE_NAME",
    "LockedSemanticGlobalTerminalJudgeError",
    "PREFLIGHT_NAME",
    "REPLAY_NAME",
    "SCORE_NAME",
    "SCORE_REPLAY_NAME",
    "SELECTED_QUESTION_COUNT",
    "build_parser",
    "build_preflight_payload",
    "load_verified_judge_run",
    "main",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
]
