"""Provider-free v4 revalidation of the sealed exact-11 Terra completions.

This lifecycle authenticates the historical v2 preflight, answer run, replay,
response journals, and post-seal evidence gate before applying validator v4.
It writes a distinct run/replay pair; the historical v3-validator answer files
remain immutable and exactly reproducible.  No command has a provider path and
no command loads benchmark gold.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from tools import run_locked_semantic_global_terminal_answer as answer_cli  # noqa: E402
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
)
from tools.matched_eval.typed_memory_final_validator_v4 import (  # noqa: E402
    RESULT_ROW_FORMAT,
    VALIDATOR_POLICY_FORMAT,
    judge_row_projection,
    materialize_typed_final_result_row_v4,
)


FORMAT = "memory-condense-locked-semantic-global-terminal-answer-validator-v4"
RUN_FORMAT = f"{FORMAT}-run-v1"
REPLAY_FORMAT = f"{FORMAT}-replay-v1"
RUN_NAME = "semantic-global-terminal-terra-answer-validator-v4.json"
REPLAY_NAME = "semantic-global-terminal-terra-answer-validator-v4-replay.json"
DEFAULT_OUTPUT_ROOT = answer_cli.DEFAULT_OUTPUT_ROOT / "validator-v4"
EXACT_ORDINALS = answer_cli.EXACT_ORDINALS
QUESTION_COUNT = len(EXACT_ORDINALS)
POSTSEAL_BINDING_KEYS = answer_cli.POSTSEAL_BINDING_KEYS


class LockedSemanticGlobalTerminalValidatorV4Error(MatchedEvalContractError):
    """A sealed v2 answer source or provider-free v4 replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticGlobalTerminalValidatorV4Error(message)


def _read_source(
    args: argparse.Namespace,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
]:
    root = Path(args.answer_root)
    preflight, _prompts, plans = answer_cli._read_preflight(  # noqa: SLF001
        root, str(args.expected_answer_preflight_sha256)
    )
    run, replay, _judge_rows = answer_cli.load_verified_answer_run(
        root,
        expected_preflight_sha256=str(args.expected_answer_preflight_sha256),
        expected_run_sha256=str(args.expected_answer_run_sha256),
        expected_replay_sha256=str(args.expected_answer_replay_sha256),
        postseal_audit=args.postseal_audit,
        expected_postseal_audit_sha256=str(
            args.expected_postseal_audit_sha256
        ),
    )
    _require(
        run.payload.get("preflight_artifact_sha256") == preflight.sha256
        and tuple(row.get("ordinal") for row in plans) == EXACT_ORDINALS,
        "v4 answer source preflight/run binding changed",
    )
    return preflight, run, replay, plans


def _validated_completion_records(
    run: SealedArtifact,
    plans: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    batch = run.payload.get("completion_batch")
    _require(type(batch) is dict, "v4 source completion batch changed type")
    completions = batch.get("logical_completions")
    records = batch.get("unique_records")
    usage = batch.get("usage")
    _require(
        type(completions) is list
        and len(completions) == QUESTION_COUNT
        and all(type(value) is str and bool(value) for value in completions)
        and type(records) is list
        and len(records) == QUESTION_COUNT
        and type(usage) is dict
        and usage.get("logical_calls") == QUESTION_COUNT
        and usage.get("unique_calls") == QUESTION_COUNT
        and usage.get("checkpoint_hits") == QUESTION_COUNT
        and usage.get("physical_calls") == 0,
        "v4 source completion batch is not checkpoint-only exact11",
    )
    by_messages: dict[str, Mapping[str, Any]] = {}
    for record in records:
        _require(type(record) is dict, "v4 source completion record changed type")
        messages_sha = require_sha256(
            record.get("messages_sha256"), "v4 source response messages"
        )
        completion = record.get("completion")
        _require(
            type(completion) is str
            and bool(completion)
            and record.get("completion_sha256") == quote_sha256(completion)
            and record.get("checkpoint_hit") is True
            and record.get("physical_call") is False
            and messages_sha not in by_messages,
            "v4 source completion record changed",
        )
        for key in (
            "call_key_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
        ):
            require_sha256(record.get(key), f"v4 source {key}")
        by_messages[messages_sha] = record
    result: list[tuple[str, Mapping[str, Any]]] = []
    for plan, completion in zip(plans, completions, strict=True):
        messages_sha = require_sha256(
            plan.get("messages_sha256"), "v4 plan messages"
        )
        record = by_messages.get(messages_sha)
        _require(
            record is not None and record.get("completion") == completion,
            "v4 completion differs from its authenticated prompt journal",
        )
        assert record is not None
        result.append((completion, record))
    return tuple(result)


def build_materialization_payload(
    preflight: SealedArtifact,
    source_run: SealedArtifact,
    source_replay: SealedArtifact,
    plans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact provider-free v4 answer artifact."""

    completions = _validated_completion_records(source_run, plans)
    results: list[dict[str, Any]] = []
    for plan, (completion, record) in zip(plans, completions, strict=True):
        results.append(
            materialize_typed_final_result_row_v4(
                plan,
                completion,
                completion_receipt_sha256=str(record["completion_sha256"]),
                call_key_sha256=str(record["call_key_sha256"]),
                request_journal_sha256=str(record["request_journal_sha256"]),
                response_journal_sha256=str(record["response_journal_sha256"]),
            )
        )
    judge_rows = [judge_row_projection(row) for row in results]
    _require(
        tuple(row["ordinal"] for row in results) == EXACT_ORDINALS
        and tuple(row["question_id"] for row in results)
        == tuple(row["question_id"] for row in judge_rows),
        "v4 materialization population/order changed",
    )
    payload = {
        "changed_prediction_count": sum(
            bool(row["changed_from_parent"]) for row in results
        ),
        "exact_ordinals": list(EXACT_ORDINALS),
        "format": RUN_FORMAT,
        "gold_loaded": False,
        "invalid_completion_parent_fallback_count": sum(
            row["prediction_source"] == "typed_final_invalid_keep_parent_v4"
            for row in results
        ),
        "judge_rows": judge_rows,
        "physical_provider_calls_during_revalidation": 0,
        **{key: preflight.payload[key] for key in POSTSEAL_BINDING_KEYS},
        "question_count": QUESTION_COUNT,
        "questions": results,
        "retained_transformer_token_state_bytes": 0,
        "source_answer_preflight_artifact_sha256": preflight.sha256,
        "source_answer_replay_artifact_sha256": source_replay.sha256,
        "source_answer_run_artifact_sha256": source_run.sha256,
        "source_completion_batch_sha256": identity_sha256(
            source_run.payload["completion_batch"]
        ),
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    assert_gold_blind(payload, path="semantic_global_terminal_validator_v4_run")
    return payload


def _validate_run(
    artifact: SealedArtifact,
    *,
    expected_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    questions = payload.get("questions")
    judge_rows = payload.get("judge_rows")
    _require(
        payload == dict(expected_payload)
        and payload.get("format") == RUN_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls_during_revalidation") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("validator_policy_format") == VALIDATOR_POLICY_FORMAT
        and type(questions) is list
        and type(judge_rows) is list
        and len(questions) == len(judge_rows) == QUESTION_COUNT,
        "v4 answer run envelope changed",
    )
    validated: list[dict[str, Any]] = []
    for ordinal, source, projected in zip(
        EXACT_ORDINALS, questions, judge_rows, strict=True
    ):
        _require(
            type(source) is dict and type(projected) is dict,
            "v4 result row changed type",
        )
        unsigned = dict(source)
        declared = unsigned.pop("source_row_sha256", None)
        prediction = source.get("prediction")
        _require(
            source.get("format") == RESULT_ROW_FORMAT
            and source.get("ordinal") == ordinal
            and declared == identity_sha256(unsigned)
            and type(prediction) is str
            and bool(prediction)
            and source.get("prediction_sha256") == quote_sha256(prediction)
            and source.get("validator_policy_format") == VALIDATOR_POLICY_FORMAT
            and judge_row_projection(source) == projected,
            f"v4 answer result row {ordinal} changed",
        )
        validated.append(dict(projected))
    return tuple(validated)


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, source_run, source_replay, plans = _read_source(args)
    payload = build_materialization_payload(
        preflight, source_run, source_replay, plans
    )
    artifact, created = publish_sealed_json(
        Path(args.output_root) / RUN_NAME, payload
    )
    _validate_run(artifact, expected_payload=payload)
    return {
        "changed_prediction_count": payload["changed_prediction_count"],
        "created": created,
        "gold_loaded": False,
        "invalid_completion_parent_fallback_count": payload[
            "invalid_completion_parent_fallback_count"
        ],
        "physical_provider_calls": 0,
        "run_sha256": artifact.sha256,
        "source_answer_run_sha256": source_run.sha256,
    }


def _replay_payload(
    run: SealedArtifact,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "byte_identical": run.payload == dict(expected),
        "expected_run_sha256": run.sha256,
        "format": REPLAY_FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "replayed_run_sha256": run.sha256,
        "retained_transformer_token_state_bytes": 0,
        "source_answer_preflight_artifact_sha256": expected[
            "source_answer_preflight_artifact_sha256"
        ],
        "source_answer_replay_artifact_sha256": expected[
            "source_answer_replay_artifact_sha256"
        ],
        "source_answer_run_artifact_sha256": expected[
            "source_answer_run_artifact_sha256"
        ],
        "source_completion_batch_sha256": expected[
            "source_completion_batch_sha256"
        ],
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    assert_gold_blind(body, path="semantic_global_terminal_validator_v4_replay")
    return body


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    preflight, source_run, source_replay, plans = _read_source(args)
    expected = build_materialization_payload(
        preflight, source_run, source_replay, plans
    )
    run = read_sealed_json(Path(args.output_root) / RUN_NAME)
    _require(
        run.sha256
        == require_sha256(args.expected_validator_run_sha256, "v4 answer run"),
        "v4 answer run artifact changed",
    )
    _validate_run(run, expected_payload=expected)
    replay_payload = _replay_payload(run, expected)
    _require(replay_payload["byte_identical"] is True, "v4 replay is not exact")
    replay, created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, replay_payload
    )
    return {
        "byte_identical": True,
        "created": created,
        "physical_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "run_sha256": run.sha256,
    }


def load_verified_revalidated_answer_run(
    output_root: str | Path,
    *,
    answer_root: str | Path,
    expected_answer_preflight_sha256: str,
    expected_answer_run_sha256: str,
    expected_answer_replay_sha256: str,
    postseal_audit: str | Path,
    expected_postseal_audit_sha256: str,
    expected_validator_run_sha256: str,
    expected_validator_replay_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    """Return the stable judge seam after authenticating both generations."""

    args = argparse.Namespace(
        answer_root=Path(answer_root),
        expected_answer_preflight_sha256=expected_answer_preflight_sha256,
        expected_answer_run_sha256=expected_answer_run_sha256,
        expected_answer_replay_sha256=expected_answer_replay_sha256,
        postseal_audit=Path(postseal_audit),
        expected_postseal_audit_sha256=expected_postseal_audit_sha256,
    )
    preflight, source_run, source_replay, plans = _read_source(args)
    expected = build_materialization_payload(
        preflight, source_run, source_replay, plans
    )
    root = Path(output_root)
    run = read_sealed_json(root / RUN_NAME)
    _require(
        run.sha256
        == require_sha256(expected_validator_run_sha256, "v4 answer run"),
        "v4 answer run artifact changed",
    )
    judge_rows = _validate_run(run, expected_payload=expected)
    replay = read_sealed_json(root / REPLAY_NAME)
    replay_expected = _replay_payload(run, expected)
    _require(
        replay.sha256
        == require_sha256(expected_validator_replay_sha256, "v4 answer replay")
        and replay.payload == replay_expected
        and replay.payload.get("byte_identical") is True,
        "v4 answer replay changed",
    )
    return run, replay, judge_rows


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--answer-root", type=Path, required=True)
    parser.add_argument("--expected-answer-preflight-sha256", required=True)
    parser.add_argument("--expected-answer-run-sha256", required=True)
    parser.add_argument("--expected-answer-replay-sha256", required=True)
    parser.add_argument("--postseal-audit", type=Path, required=True)
    parser.add_argument("--expected-postseal-audit-sha256", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize")
    _add_sources(materialize)
    replay = commands.add_parser("replay")
    _add_sources(replay)
    replay.add_argument("--expected-validator-run-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_materialize(args) if args.command == "materialize" else run_replay(args)
    print(_canonical_output(payload))
    return 0


def _canonical_output(value: Mapping[str, Any]) -> str:
    import json

    return json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
