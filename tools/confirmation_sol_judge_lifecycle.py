#!/usr/bin/env python3
"""Execute the sealed confirmation judge plan through the locked Sol route.

The gold-capable judge scaffold owns dataset access and emits the complete
question/reference/prediction plane.  This module never opens the benchmark;
it authenticates that plane, rebuilds the standard binary-judge prompts, and
runs the closed lifecycle

``preflight -> approve-release -> provider-run -> materialize -> replay``.

Only ``provider-run`` constructs a client.  Releases authorize exactly the
remaining complete-journal misses, retries are zero, and request-only journals
are terminal rather than silently retried.
"""

from __future__ import annotations

import argparse
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

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval._binary_judge_protocol import (  # noqa: E402
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import build_judge_prompt  # noqa: E402
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRecord,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from tools import confirmation_gold_judge_scaffold as judge  # noqa: E402
from tools.v4_population_firebreak.canonical import (  # noqa: E402
    assert_snapshot_unchanged,
    canonical_sha256,
    require_list,
    require_mapping,
    require_sha256,
    require_text,
)


FORMAT = "memory-condense-confirmation-sol-judge-lifecycle-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
RELEASE_FORMAT = f"{FORMAT}-provider-release-v1"
COMPLETION_FORMAT = f"{FORMAT}-completion-plane-v1"
COMPLETION_ROW_FORMAT = f"{COMPLETION_FORMAT}-row-v1"

PREFLIGHT_NAME = "confirmation-sol-judge-preflight-v1.json"
RELEASE_NAME = "confirmation-sol-judge-provider-release-v1.json"
COMPLETION_NAME = "confirmation-sol-judge-completions-v1.json"
RESULTS_NAME = "confirmation-sol-judge-results-v1.json"
REPLAY_NAME = "confirmation-sol-judge-completions-replay-v1.json"
CHECKPOINT_DIR_NAME = "confirmation-sol-judge-calls"

SOL_MODEL = "codex_sdk/gpt-5.6-sol"
SOL_GATEWAY_URL = "https://central-dev.zt:4000/v1"
MAX_PROMPT_TOKENS = 8_000
MAX_CONCURRENCY = 4
DEFAULT_API_KEY_ENV = "LITELLM_KEY"

_JOURNAL = re.compile(r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$")


class ConfirmationSolLifecycleError(ValueError):
    """The sealed Sol judge lifecycle failed closed."""


@dataclass(frozen=True, slots=True)
class VerifiedJudgePlan:
    artifact: judge.SealedJson
    question_ids: tuple[str, ...]
    row_receipts: tuple[str, ...]
    prompts: tuple[tuple[dict[str, str], ...], ...]
    population: FastPromptPopulation


ClientFactory = Callable[[str, str], Any]


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationSolLifecycleError(message)


def _seal(body: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {**dict(body), key: canonical_sha256(dict(body))}


def _self_seal(value: Mapping[str, Any], key: str, label: str) -> str:
    try:
        receipt = require_sha256(value.get(key), f"{label} receipt")
    except ValueError as exc:
        raise ConfirmationSolLifecycleError(str(exc)) from exc
    body = dict(value)
    body.pop(key, None)
    _require(canonical_sha256(body) == receipt, f"{label} self-seal differs")
    return receipt


def verify_judge_plan(path: str | Path, *, expected_sha256: str) -> VerifiedJudgePlan:
    """Authenticate an already-opened complete judge plane and exact prompts."""

    try:
        artifact = judge.read_sealed_json(
            path,
            expected_sha256=expected_sha256,
            label="confirmation Sol judge plan",
        )
        question_ids = judge._validate_judge_plan(artifact)  # noqa: SLF001
    except (ValueError, OSError) as exc:
        raise ConfirmationSolLifecycleError(str(exc)) from exc
    rows = require_list(artifact.payload.get("rows"), "judge plan rows")
    prompts: list[tuple[dict[str, str], ...]] = []
    receipts: list[str] = []
    for index, raw in enumerate(rows):
        row = require_mapping(raw, f"judge plan row {index}")
        messages = tuple(
            dict(message)
            for message in build_judge_prompt(
                require_text(row.get("question"), f"judge row {index} question"),
                require_text(
                    row.get("reference_answer"), f"judge row {index} reference"
                ),
                require_text(row.get("prediction"), f"judge row {index} prediction"),
            )
        )
        prompts.append(messages)
        receipts.append(
            require_sha256(row.get("row_receipt_sha256"), f"judge row {index} receipt")
        )
    try:
        population = preflight_fast_completion_prompts(
            prompts, max_prompt_tokens=MAX_PROMPT_TOKENS
        )
    except (TypeError, ValueError) as exc:
        raise ConfirmationSolLifecycleError(str(exc)) from exc
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == len(question_ids),
        "confirmation requires one distinct Sol prompt per sealed prediction",
    )
    assert_snapshot_unchanged(artifact.snapshot, "confirmation Sol judge plan")
    assert_snapshot_unchanged(
        artifact.sidecar, "confirmation Sol judge plan digest sidecar"
    )
    return VerifiedJudgePlan(
        artifact=artifact,
        question_ids=question_ids,
        row_receipts=tuple(receipts),
        prompts=tuple(prompts),
        population=population,
    )


def _runtime_projection(plan: VerifiedJudgePlan) -> dict[str, Any]:
    return {
        "gateway_url": SOL_GATEWAY_URL,
        "max_concurrency": MAX_CONCURRENCY,
        "max_new_tokens": JUDGE_MAX_TOKENS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "model": SOL_MODEL,
        "retry_count": 0,
    }


def compile_preflight(plan: VerifiedJudgePlan) -> dict[str, Any]:
    rows = [
        _seal(
            {
                "row_index": index,
                "question_id": question_id,
                "judge_plan_row_receipt_sha256": receipt,
                "messages_sha256": prompt.messages_sha256,
                "prompt_token_proxy": prompt.prompt_token_proxy,
            },
            "row_receipt_sha256",
        )
        for index, (question_id, receipt, prompt) in enumerate(
            zip(
                plan.question_ids,
                plan.row_receipts,
                plan.population.ordered_rows,
                strict=True,
            )
        )
    ]
    body = {
        "format": PREFLIGHT_FORMAT,
        "status": "verified",
        "gold_loaded": True,
        "judge_plan_sha256": plan.artifact.sha256,
        "runtime": _runtime_projection(plan),
        "population": {
            "question_count": len(rows),
            "ordered_question_ids_sha256": canonical_sha256(list(plan.question_ids)),
            "prompt_population_sha256": plan.population.prompt_population_sha256,
            "unique_prompt_count": plan.population.unique_prompt_count,
        },
        "ordered_rows": rows,
        "execution": {
            "would_call_count": plan.population.unique_prompt_count,
            "physical_provider_calls": 0,
            "provider_execution_available": False,
            "authorization_released": False,
        },
    }
    return _seal(body, "preflight_identity_sha256")


def publish_preflight(
    *, judge_plan_path: str | Path, expected_judge_plan_sha256: str, output_root: str | Path
) -> tuple[judge.SealedJson, bool]:
    plan = verify_judge_plan(judge_plan_path, expected_sha256=expected_judge_plan_sha256)
    return judge.publish_sealed_json(Path(output_root) / PREFLIGHT_NAME, compile_preflight(plan))


def _read_preflight(
    plan: VerifiedJudgePlan, *, output_root: str | Path, expected_sha256: str
) -> judge.SealedJson:
    try:
        artifact = judge.read_sealed_json(
            Path(output_root) / PREFLIGHT_NAME,
            expected_sha256=expected_sha256,
            label="confirmation Sol lifecycle preflight",
        )
    except (ValueError, OSError) as exc:
        raise ConfirmationSolLifecycleError(str(exc)) from exc
    _require(artifact.payload == compile_preflight(plan), "Sol lifecycle preflight differs")
    return artifact


def _canonical_root(path: str | Path) -> str:
    return str(Path(path).resolve())


def _runtime(
    plan: VerifiedJudgePlan,
    preflight: judge.SealedJson,
    *,
    output_root: str | Path,
    client: Any | None,
) -> FastCompletionRuntime:
    return FastCompletionRuntime(
        checkpoint_dir=Path(output_root) / CHECKPOINT_DIR_NAME,
        prompt_population=plan.prompts,
        model=SOL_MODEL,
        client=client,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=MAX_CONCURRENCY,
        retries=0,
        benchmark_provenance={
            "arm": FORMAT,
            "authorized_unique_calls": plan.population.unique_prompt_count,
            "judge_plan_sha256": plan.artifact.sha256,
            "lifecycle_preflight_sha256": preflight.sha256,
        },
    )


def _checkpoint_keys(output_root: str | Path, *, maximum: int) -> tuple[str, ...]:
    root = Path(output_root) / CHECKPOINT_DIR_NAME
    if not root.exists():
        return ()
    _require(root.is_dir() and not root.is_symlink(), "checkpoint root is unsafe")
    requests: set[str] = set()
    responses: set[str] = set()
    for path in root.iterdir():
        if path.name == ".fast-completion-journal.lock":
            continue
        _require(path.is_file() and not path.is_symlink(), "checkpoint root has foreign state")
        match = _JOURNAL.fullmatch(path.name)
        _require(match is not None, "checkpoint root contains foreign journal state")
        assert match is not None
        (requests if match.group("kind") == "request" else responses).add(match.group("key"))
    _require(requests == responses, "checkpoint request/response pair is incomplete; retry forbidden")
    _require(len(requests) <= maximum, "checkpoint population exceeds sealed prompts")
    return tuple(sorted(requests))


def _authenticated_records(
    plan: VerifiedJudgePlan,
    preflight: judge.SealedJson,
    *,
    output_root: str | Path,
) -> tuple[FastCompletionRecord, ...]:
    structural = _checkpoint_keys(output_root, maximum=plan.population.unique_prompt_count)
    if not structural:
        return ()
    runtime = _runtime(plan, preflight, output_root=output_root, client=None)
    try:
        with runtime._journal_guard():  # noqa: SLF001
            by_message = runtime._load_all_records()  # noqa: SLF001
    finally:
        runtime.close()
    ordered = tuple(
        by_message[row.messages_sha256]
        for row in plan.population.ordered_rows
        if row.messages_sha256 in by_message
    )
    _require(len(ordered) == len(structural), "authenticated checkpoint population differs")
    return ordered


def _record_binding(record: FastCompletionRecord) -> dict[str, Any]:
    return {
        "messages_sha256": record.messages_sha256,
        "call_key_sha256": record.call_key_sha256,
        "request_journal_sha256": record.request_journal_sha256,
        "response_journal_sha256": record.response_journal_sha256,
    }


def approve_provider_release(
    *,
    judge_plan_path: str | Path,
    expected_judge_plan_sha256: str,
    output_root: str | Path,
    expected_preflight_sha256: str,
    approve_provider_release: bool,
    authorized_provider_calls: int,
) -> tuple[judge.SealedJson, bool]:
    _require(approve_provider_release is True, "Sol release requires explicit approval")
    plan = verify_judge_plan(judge_plan_path, expected_sha256=expected_judge_plan_sha256)
    preflight = _read_preflight(plan, output_root=output_root, expected_sha256=expected_preflight_sha256)
    records = _authenticated_records(plan, preflight, output_root=output_root)
    remaining = plan.population.unique_prompt_count - len(records)
    _require(
        type(authorized_provider_calls) is int and authorized_provider_calls == remaining,
        "release authorization must exactly equal remaining Sol calls",
    )
    bound = [_record_binding(record) for record in records]
    body = {
        "format": RELEASE_FORMAT,
        "release_status": "approved_for_provider_execution",
        "approval_opt_in": True,
        "gold_loaded": True,
        "judge_plan_sha256": plan.artifact.sha256,
        "lifecycle_preflight_sha256": preflight.sha256,
        "runtime": _runtime_projection(plan),
        "output_root": _canonical_root(output_root),
        "checkpoint_snapshot": {
            "authenticated_complete_count": len(bound),
            "ordered_records": bound,
            "ordered_records_sha256": canonical_sha256(bound),
        },
        "required_authorized_provider_calls": remaining,
        "unsafe_retry_policy": "refuse-incomplete-request-response-pair-v1",
        "provider_calls_during_release": 0,
    }
    return judge.publish_sealed_json(Path(output_root) / RELEASE_NAME, _seal(body, "release_identity_sha256"))


def _read_release(
    plan: VerifiedJudgePlan,
    preflight: judge.SealedJson,
    *,
    output_root: str | Path,
    expected_sha256: str,
) -> judge.SealedJson:
    try:
        artifact = judge.read_sealed_json(
            Path(output_root) / RELEASE_NAME,
            expected_sha256=expected_sha256,
            label="confirmation Sol provider release",
        )
    except (ValueError, OSError) as exc:
        raise ConfirmationSolLifecycleError(str(exc)) from exc
    value = artifact.payload
    _self_seal(value, "release_identity_sha256", "Sol provider release")
    _require(
        value.get("format") == RELEASE_FORMAT
        and value.get("release_status") == "approved_for_provider_execution"
        and value.get("approval_opt_in") is True
        and value.get("gold_loaded") is True
        and value.get("judge_plan_sha256") == plan.artifact.sha256
        and value.get("lifecycle_preflight_sha256") == preflight.sha256
        and value.get("runtime") == _runtime_projection(plan)
        and value.get("output_root") == _canonical_root(output_root)
        and value.get("unsafe_retry_policy") == "refuse-incomplete-request-response-pair-v1"
        and value.get("provider_calls_during_release") == 0,
        "Sol provider release bindings changed",
    )
    snapshot = require_mapping(value.get("checkpoint_snapshot"), "checkpoint snapshot")
    rows = require_list(snapshot.get("ordered_records"), "release checkpoint records")
    _require(
        snapshot.get("authenticated_complete_count") == len(rows)
        and snapshot.get("ordered_records_sha256") == canonical_sha256(rows)
        and value.get("required_authorized_provider_calls")
        == plan.population.unique_prompt_count - len(rows),
        "Sol release accounting changed",
    )
    return artifact


def _default_client_factory(gateway_url: str, api_key_env: str) -> Any:
    # Load credentials only after the sealed Sol release and exact remaining
    # call authorization have passed.  Readiness, preflight, status, and
    # prediction execution must never inspect the repository environment.
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - locked production dependency
        raise ConfirmationSolLifecycleError(
            "python-dotenv is required for provider execution"
        ) from exc
    load_dotenv(override=False)
    key = os.environ.get(api_key_env, "").strip()
    _require(bool(key), f"provider API key is empty: {api_key_env}")
    from tools.matched_eval.live import _make_provider_client  # noqa: PLC0415

    return _make_provider_client(key, gateway_url)


def _inputs(
    *,
    judge_plan_path: str | Path,
    expected_judge_plan_sha256: str,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
) -> tuple[VerifiedJudgePlan, judge.SealedJson, judge.SealedJson]:
    plan = verify_judge_plan(judge_plan_path, expected_sha256=expected_judge_plan_sha256)
    preflight = _read_preflight(plan, output_root=output_root, expected_sha256=expected_preflight_sha256)
    release = _read_release(
        plan, preflight, output_root=output_root, expected_sha256=expected_release_sha256
    )
    return plan, preflight, release


def run_provider(
    *,
    judge_plan_path: str | Path,
    expected_judge_plan_sha256: str,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    client_factory: ClientFactory = _default_client_factory,
) -> dict[str, Any]:
    plan, preflight, release = _inputs(
        judge_plan_path=judge_plan_path,
        expected_judge_plan_sha256=expected_judge_plan_sha256,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    _require(enable_provider is True, "Sol execution requires explicit opt-in")
    before = _authenticated_records(plan, preflight, output_root=output_root)
    released_rows = require_list(
        require_mapping(release.payload["checkpoint_snapshot"], "checkpoint snapshot")["ordered_records"],
        "release records",
    )
    current = {record.messages_sha256: _record_binding(record) for record in before}
    _require(
        all(current.get(str(row.get("messages_sha256"))) == dict(row) for row in released_rows),
        "a checkpoint authenticated at release changed or disappeared",
    )
    remaining = plan.population.unique_prompt_count - len(before)
    _require(
        type(authorized_provider_calls) is int and authorized_provider_calls == remaining,
        "provider authorization must exactly equal remaining Sol calls",
    )
    _require(
        remaining <= int(release.payload["required_authorized_provider_calls"]),
        "current checkpoint state exceeds the sealed release budget",
    )
    client = client_factory(SOL_GATEWAY_URL, api_key_env) if remaining else None
    runtime = _runtime(plan, preflight, output_root=output_root, client=client)
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.logical_calls == batch.usage.unique_calls == len(plan.question_ids)
        and batch.usage.physical_calls == remaining
        and batch.usage.checkpoint_hits == len(before),
        "Sol provider call accounting differs from authorization",
    )
    return {
        "question_count": len(plan.question_ids),
        "authorized_remaining_provider_calls": remaining,
        "checkpoint_hits_before_run": len(before),
        "physical_provider_calls": batch.usage.physical_calls,
        "retry_count": 0,
    }


def _checkpoint_batch(
    plan: VerifiedJudgePlan, preflight: judge.SealedJson, *, output_root: str | Path
) -> FastCompletionBatch:
    _require(
        len(_checkpoint_keys(output_root, maximum=plan.population.unique_prompt_count))
        == plan.population.unique_prompt_count,
        "materialization requires a complete Sol checkpoint population",
    )
    runtime = _runtime(plan, preflight, output_root=output_root, client=None)
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == len(plan.question_ids)
        and batch.usage.physical_calls == 0,
        "materialization is not checkpoint-only",
    )
    return batch


def _completion_payload(
    plan: VerifiedJudgePlan,
    preflight: judge.SealedJson,
    release: judge.SealedJson,
    batch: FastCompletionBatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = {record.messages_sha256: record for record in batch.unique_records}
    completion_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, str]] = []
    for index, (question_id, source_receipt, prompt_row, output) in enumerate(
        zip(
            plan.question_ids,
            plan.row_receipts,
            plan.population.ordered_rows,
            batch.logical_completions,
            strict=True,
        )
    ):
        try:
            verdict = parse_binary_judge_verdict(output)
        except RuntimeError as exc:
            raise ConfirmationSolLifecycleError(
                f"Sol verdict {index} is invalid"
            ) from exc
        record = records[prompt_row.messages_sha256]
        body = {
            "format": COMPLETION_ROW_FORMAT,
            "row_index": index,
            "question_id": question_id,
            "judge_plan_row_receipt_sha256": source_receipt,
            "messages_sha256": prompt_row.messages_sha256,
            "completion": output,
            "completion_sha256": quote_sha256(output),
            **_record_binding(record),
            "verdict": "correct" if verdict else "incorrect",
        }
        completion_rows.append(_seal(body, "row_receipt_sha256"))
        result_rows.append(
            {"question_id": question_id, "verdict": "correct" if verdict else "incorrect"}
        )
    completion_body = {
        "format": COMPLETION_FORMAT,
        "status": "complete",
        "gold_loaded": True,
        "judge_plan_sha256": plan.artifact.sha256,
        "lifecycle_preflight_sha256": preflight.sha256,
        "provider_release_sha256": release.sha256,
        "population": {
            "question_count": len(plan.question_ids),
            "ordered_question_ids_sha256": canonical_sha256(list(plan.question_ids)),
            "prompt_population_sha256": plan.population.prompt_population_sha256,
        },
        "ordered_rows": completion_rows,
        "completion_batch": batch.model_dump(),
        "physical_provider_calls_during_materialization": 0,
    }
    completion = _seal(completion_body, "completion_identity_sha256")
    results = {
        "format": judge.JUDGE_RESULTS_FORMAT,
        "status": "complete",
        "judge_plan_sha256": plan.artifact.sha256,
        "sample_count": len(plan.question_ids),
        "ordered_question_ids_sha256": canonical_sha256(list(plan.question_ids)),
        "rows": result_rows,
    }
    return completion, results


def materialize(
    *,
    judge_plan_path: str | Path,
    expected_judge_plan_sha256: str,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
) -> tuple[judge.SealedJson, judge.SealedJson]:
    plan, preflight, release = _inputs(
        judge_plan_path=judge_plan_path,
        expected_judge_plan_sha256=expected_judge_plan_sha256,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    batch = _checkpoint_batch(plan, preflight, output_root=output_root)
    completion, results = _completion_payload(plan, preflight, release, batch)
    completion_artifact, _ = judge.publish_sealed_json(
        Path(output_root) / COMPLETION_NAME, completion
    )
    results_artifact, _ = judge.publish_sealed_json(Path(output_root) / RESULTS_NAME, results)
    return completion_artifact, results_artifact


def replay(
    *,
    judge_plan_path: str | Path,
    expected_judge_plan_sha256: str,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_release_sha256: str,
    expected_completion_sha256: str,
    expected_results_sha256: str,
) -> tuple[judge.SealedJson, judge.SealedJson]:
    plan, preflight, release = _inputs(
        judge_plan_path=judge_plan_path,
        expected_judge_plan_sha256=expected_judge_plan_sha256,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    expected_completion, expected_results = _completion_payload(
        plan, preflight, release, _checkpoint_batch(plan, preflight, output_root=output_root)
    )
    completion = judge.read_sealed_json(
        Path(output_root) / COMPLETION_NAME,
        expected_sha256=expected_completion_sha256,
        label="confirmation Sol completion plane",
    )
    results = judge.read_sealed_json(
        Path(output_root) / RESULTS_NAME,
        expected_sha256=expected_results_sha256,
        label="confirmation Sol result plane",
    )
    _require(completion.payload == expected_completion, "Sol completion replay differs")
    _require(results.payload == expected_results, "Sol result replay differs")
    replay_artifact, _ = judge.publish_sealed_json(Path(output_root) / REPLAY_NAME, expected_completion)
    return replay_artifact, results


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--judge-plan", required=True)
    parser.add_argument("--expected-judge-plan-sha256", required=True)
    parser.add_argument("--output-root", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _common(preflight)
    release = commands.add_parser("approve-release")
    _common(release)
    release.add_argument("--expected-preflight-sha256", required=True)
    release.add_argument("--approve-provider-release", action="store_true")
    release.add_argument("--authorized-provider-calls", type=int, required=True)
    provider = commands.add_parser("provider-run")
    _common(provider)
    provider.add_argument("--expected-preflight-sha256", required=True)
    provider.add_argument("--expected-release-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    materialize_parser = commands.add_parser("materialize")
    _common(materialize_parser)
    materialize_parser.add_argument("--expected-preflight-sha256", required=True)
    materialize_parser.add_argument("--expected-release-sha256", required=True)
    replay_parser = commands.add_parser("replay")
    _common(replay_parser)
    replay_parser.add_argument("--expected-preflight-sha256", required=True)
    replay_parser.add_argument("--expected-release-sha256", required=True)
    replay_parser.add_argument("--expected-completion-sha256", required=True)
    replay_parser.add_argument("--expected-results-sha256", required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    common = {
        "judge_plan_path": args.judge_plan,
        "expected_judge_plan_sha256": args.expected_judge_plan_sha256,
        "output_root": args.output_root,
    }
    if args.command == "preflight":
        artifact, created = publish_preflight(**common)
        return {"created": created, "preflight_sha256": artifact.sha256, "physical_provider_calls": 0}
    shared = {
        **common,
        "expected_preflight_sha256": args.expected_preflight_sha256,
    }
    if args.command == "approve-release":
        artifact, created = approve_provider_release(
            **shared,
            approve_provider_release=args.approve_provider_release,
            authorized_provider_calls=args.authorized_provider_calls,
        )
        return {"created": created, "release_sha256": artifact.sha256, "physical_provider_calls": 0}
    execution = {**shared, "expected_release_sha256": args.expected_release_sha256}
    if args.command == "provider-run":
        return run_provider(
            **execution,
            enable_provider=args.enable_provider,
            authorized_provider_calls=args.authorized_provider_calls,
            api_key_env=args.api_key_env,
        )
    if args.command == "materialize":
        completion, results = materialize(**execution)
        return {"completion_sha256": completion.sha256, "results_sha256": results.sha256, "physical_provider_calls": 0}
    replay_artifact, results = replay(
        **execution,
        expected_completion_sha256=args.expected_completion_sha256,
        expected_results_sha256=args.expected_results_sha256,
    )
    return {"replay_sha256": replay_artifact.sha256, "results_sha256": results.sha256, "physical_provider_calls": 0}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "COMPLETION_NAME",
    "ConfirmationSolLifecycleError",
    "PREFLIGHT_NAME",
    "RELEASE_NAME",
    "RESULTS_NAME",
    "SOL_GATEWAY_URL",
    "SOL_MODEL",
    "approve_provider_release",
    "build_parser",
    "materialize",
    "publish_preflight",
    "replay",
    "run_provider",
    "verify_judge_plan",
]
