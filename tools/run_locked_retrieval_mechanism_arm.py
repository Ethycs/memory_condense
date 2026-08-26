#!/usr/bin/env python3
"""Run the matched locked-100 retrieval-mechanism control arm.

``S0_CONTROL`` sends the exact sealed S0 provider messages for every question.
The historical fixed-S1 validator is deliberately run first: no output path is
touched until the pinned merged retrieval and baseline answer artifact pass it.
The benchmark corpus and raw gold answers are never inputs to this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # support ``python tools/run_...py``
    _REPOSITORY = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_REPOSITORY / "src"), str(_REPOSITORY)]

from dotenv import load_dotenv

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._artifact_json import canonical_json_bytes
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.recall_guarded_cumulative_1m import STAGE_IDS
from memory_condense.eval.recall_guarded_cumulative_final_answer import (
    RESPONDER_OUTPUT_TOKEN_RESERVE,
    RESPONDER_PROMPT_CAP,
)
from memory_condense.eval.recall_guarded_cumulative_final_answer_runtime import (
    LOCKED_FINAL_ANSWER_GATEWAY_MODEL,
)
from memory_condense.eval.recall_guarded_cumulative_provider_synthesis_runtime import (
    CENTRAL_DEV_GATEWAY_URL,
)
from memory_condense.eval import run_fast_1m_em_facts as em_runner
from tools._locked_em_repair_adapter import (
    LockedEMRepairPopulation,
    _read_canonical_artifact,
    load_locked_em_repair_population,
)


ARM_LABEL = "S0_CONTROL"
SOURCE_STAGE_ID = STAGE_IDS[0]
PREFLIGHT_FORMAT = "memory-condense-locked-retrieval-mechanism-arm-preflight-v1"
RUN_FORMAT = "memory-condense-locked-retrieval-mechanism-arm-run-v1"
ARM_IDENTITY_FORMAT = "memory-condense-locked-retrieval-mechanism-arm-identity-v1"

DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
    "/retrieval.json"
)
DEFAULT_BASELINE_ANSWERS = Path(
    "eval_results/longmemeval-1m-fixed-s1-validation-20260826/final-answers.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/s0-control-v1"
)
EXPECTED_RETRIEVAL_SHA256 = (
    "e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f"
)
EXPECTED_BASELINE_ANSWERS_SHA256 = (
    "d7fc47b8d1f372f002230c6ffe489dac8cd11bd71b35b8d3008b1255da2a38cd"
)
EXPECTED_QUESTION_COUNT = 100
DEFAULT_MODEL = LOCKED_FINAL_ANSWER_GATEWAY_MODEL
DEFAULT_API_KEY_ENV = "LITELLM_KEY"

_FORBIDDEN_GOLD_KEYS = frozenset(
    {
        "answer_session_ids",
        "category",
        "evidence_sources",
        "gold",
        "gold_answer",
        "reference",
        "reference_answer",
    }
)
_DISPOSITION_FIELDS = frozenset({"checkpoint_hit", "physical_call"})
_USAGE_DISPOSITION_FIELDS = frozenset({"checkpoint_hits", "physical_calls"})

_publish = em_runner._publish
_read = em_runner._read
_make_provider_client = em_runner._make_provider_client


@dataclass(frozen=True, slots=True)
class _ArmRow:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    retrieval_question_part_sha256: str
    source_stage_id: str
    stage_receipt_sha256: str
    evidence_projection_sha256: str
    provider_messages_sha256: str
    prompt_token_proxy: int
    evidence_row_count: int
    messages: tuple[dict[str, str], ...]
    binding_sha256: str


@dataclass(frozen=True, slots=True)
class _ArmPlan:
    population: LockedEMRepairPopulation
    rows: tuple[_ArmRow, ...]
    prompt_population: FastPromptPopulation
    arm_identity: Mapping[str, Any]
    arm_identity_sha256: str
    preflight: Mapping[str, Any]
    preflight_sha256: str

    @property
    def exact_calls(self) -> int:
        return self.prompt_population.unique_prompt_count


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _contains_gold_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _FORBIDDEN_GOLD_KEYS
            or _contains_gold_key(child)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return any(_contains_gold_key(child) for child in value)
    return False


def _messages(value: object, ordinal: int) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"S0 provider messages changed at ordinal {ordinal}")
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"role", "content"}:
            raise ValueError(f"S0 provider message shape changed at ordinal {ordinal}")
        role, content = item.get("role"), item.get("content")
        if role not in {"system", "user"} or not isinstance(content, str):
            raise ValueError(f"S0 provider message value changed at ordinal {ordinal}")
        result.append({"role": str(role), "content": content})
    if [row["role"] for row in result] != ["system", "user"]:
        raise ValueError(f"S0 provider message order changed at ordinal {ordinal}")
    return tuple(result)


def _validated_sources(
    retrieval_path: Path,
    baseline_answers_path: Path,
    *,
    expected_retrieval_sha256: str,
    expected_baseline_answers_sha256: str,
) -> tuple[LockedEMRepairPopulation, Mapping[str, Any]]:
    """Run the historical validator before rereading the same pinned retrieval."""

    population = load_locked_em_repair_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        baseline_final_answers_path=baseline_answers_path,
        expected_baseline_final_answers_sha256=expected_baseline_answers_sha256,
    )
    retrieval, observed_sha = _read_canonical_artifact(
        retrieval_path,
        expected_sha256=expected_retrieval_sha256,
    )
    if observed_sha != population.retrieval_sha256:
        raise ValueError("retrieval changed after historical validation")
    return population, retrieval


def _prepare(
    *,
    retrieval_path: Path,
    baseline_answers_path: Path,
    expected_retrieval_sha256: str = EXPECTED_RETRIEVAL_SHA256,
    expected_baseline_answers_sha256: str = EXPECTED_BASELINE_ANSWERS_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> _ArmPlan:
    population, retrieval = _validated_sources(
        retrieval_path,
        baseline_answers_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_baseline_answers_sha256=expected_baseline_answers_sha256,
    )
    raw_questions = retrieval.get("questions")
    part_hashes = retrieval.get("question_part_sha256s")
    if (
        population.question_count != expected_question_count
        or retrieval.get("question_count") != expected_question_count
        or not isinstance(raw_questions, list)
        or not isinstance(part_hashes, list)
        or len(raw_questions) != expected_question_count
        or len(part_hashes) != expected_question_count
    ):
        raise ValueError("locked S0 question population changed")

    rows: list[_ArmRow] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    for ordinal, (adapter_row, raw, part_sha) in enumerate(
        zip(population.rows, raw_questions, part_hashes, strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise ValueError(f"retrieval question {ordinal} is not an object")
        raw_stages = raw.get("stages")
        if not isinstance(raw_stages, list) or len(raw_stages) != len(STAGE_IDS):
            raise ValueError(f"retrieval stage population changed at ordinal {ordinal}")
        stage = raw_stages[0]
        receipt = stage.get("stage_receipt") if isinstance(stage, Mapping) else None
        evidence = stage.get("evidence") if isinstance(stage, Mapping) else None
        if (
            not isinstance(stage, Mapping)
            or stage.get("stage_id") != SOURCE_STAGE_ID
            or not isinstance(receipt, Mapping)
            or not isinstance(evidence, list)
        ):
            raise ValueError(f"sealed S0 stage changed at ordinal {ordinal}")
        question = adapter_row.question
        adapter_stage = question.stage(SOURCE_STAGE_ID)
        messages = _messages(stage.get("provider_messages"), ordinal)
        messages_sha = identity_sha256(list(messages))
        prompt_tokens = receipt.get("prompt_token_proxy")
        if (
            raw.get("ordinal") != ordinal
            or raw.get("question_id") != question.question_id
            or raw.get("question_sha256") != question.question_sha256
            or raw.get("dated_question_sha256") != question.dated_question_sha256
            or part_sha != question.retrieval_question_part_sha256
            or receipt.get("receipt_sha256")
            != adapter_stage.stage_receipt_sha256
            or receipt.get("evidence_projection_sha256")
            != adapter_stage.evidence_projection_sha256
            or receipt.get("prompt_messages_sha256") != messages_sha
            or type(prompt_tokens) is not int
            or prompt_tokens < 1
            or prompt_tokens > RESPONDER_PROMPT_CAP
            or receipt.get("max_prompt_token_proxy") != RESPONDER_PROMPT_CAP
            or receipt.get("responder_output_token_reserve")
            != RESPONDER_OUTPUT_TOKEN_RESERVE
            or tuple(
                item.get("evidence_id") if isinstance(item, Mapping) else None
                for item in evidence
            )
            != adapter_stage.evidence_ids
        ):
            raise ValueError(f"sealed S0 binding changed at ordinal {ordinal}")
        binding = {
            "ordinal": ordinal,
            "question_id": question.question_id,
            "question_sha256": question.question_sha256,
            "dated_question_sha256": question.dated_question_sha256,
            "retrieval_question_part_sha256": question.retrieval_question_part_sha256,
            "source_stage_id": SOURCE_STAGE_ID,
            "stage_receipt_sha256": adapter_stage.stage_receipt_sha256,
            "evidence_projection_sha256": adapter_stage.evidence_projection_sha256,
            "provider_messages_sha256": messages_sha,
            "prompt_token_proxy": prompt_tokens,
            "evidence_row_count": len(evidence),
        }
        rows.append(
            _ArmRow(
                **binding,
                messages=messages,
                binding_sha256=identity_sha256(binding),
            )
        )
        prompts.append(messages)

    prompt_population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=RESPONDER_PROMPT_CAP,
    )
    if (
        prompt_population.logical_prompt_count != expected_question_count
        or prompt_population.unique_prompt_count != expected_question_count
        or tuple(row.provider_messages_sha256 for row in rows)
        != tuple(row.messages_sha256 for row in prompt_population.ordered_rows)
        or tuple(row.prompt_token_proxy for row in rows)
        != tuple(row.prompt_token_proxy for row in prompt_population.ordered_rows)
    ):
        raise ValueError("S0 prompt population changed after historical validation")
    arm_identity = {
        "format": ARM_IDENTITY_FORMAT,
        "arm_label": ARM_LABEL,
        "parent_arm": None,
        "source_stage_id": SOURCE_STAGE_ID,
        "mechanism": "exact_sealed_s0_provider_messages",
        "retrieval_sha256": population.retrieval_sha256,
        "baseline_final_answers_sha256": population.baseline_final_answers_sha256,
        "population_identity_sha256": population.population_identity_sha256,
        "historical_validator_binding_sha256": population.binding_sha256,
        "question_binding_sha256s": [row.binding_sha256 for row in rows],
        "prompt_population_sha256": prompt_population.prompt_population_sha256,
        "question_count": expected_question_count,
    }
    arm_identity_sha = identity_sha256(arm_identity)
    preflight = {
        "format": PREFLIGHT_FORMAT,
        "arm_identity": arm_identity,
        "arm_identity_sha256": arm_identity_sha,
        "question_count": expected_question_count,
        "prompt_population": prompt_population.model_dump(),
        "required_authorized_provider_calls": prompt_population.unique_prompt_count,
        "provider_settings": {
            "gateway_url": CENTRAL_DEV_GATEWAY_URL,
            "model": DEFAULT_MODEL,
            "max_prompt_tokens": RESPONDER_PROMPT_CAP,
            "max_new_tokens": RESPONDER_OUTPUT_TOKEN_RESERVE,
            "retries": 0,
        },
        "questions": [
            {
                "ordinal": row.ordinal,
                "question_id": row.question_id,
                "question_sha256": row.question_sha256,
                "dated_question_sha256": row.dated_question_sha256,
                "retrieval_question_part_sha256": (
                    row.retrieval_question_part_sha256
                ),
                "source_stage_id": row.source_stage_id,
                "stage_receipt_sha256": row.stage_receipt_sha256,
                "evidence_projection_sha256": row.evidence_projection_sha256,
                "provider_messages_sha256": row.provider_messages_sha256,
                "prompt_token_proxy": row.prompt_token_proxy,
                "evidence_row_count": row.evidence_row_count,
                "binding_sha256": row.binding_sha256,
            }
            for row in rows
        ],
        "provider_calls": 0,
        "gold_loaded": False,
        "output_root_mutated_before_historical_validation": False,
        "retained_request_token_state_bytes": 0,
    }
    if _contains_gold_key(preflight):
        raise RuntimeError("S0 preflight crossed the gold firewall")
    return _ArmPlan(
        population=population,
        rows=tuple(rows),
        prompt_population=prompt_population,
        arm_identity=arm_identity,
        arm_identity_sha256=arm_identity_sha,
        preflight=preflight,
        preflight_sha256=_digest(preflight),
    )


def _stable_batch(batch: FastCompletionBatch) -> dict[str, Any]:
    payload = batch.model_dump()
    return {
        "logical_completions": payload["logical_completions"],
        "unique_records": [
            {
                key: value
                for key, value in record.items()
                if key not in _DISPOSITION_FIELDS
            }
            for record in payload["unique_records"]
        ],
        "usage": {
            key: value
            for key, value in payload["usage"].items()
            if key not in _USAGE_DISPOSITION_FIELDS
        },
        "provenance": payload["provenance"],
        "runtime_identity_sha256": payload["runtime_identity_sha256"],
        "prompt_population": payload["prompt_population"],
    }


def _runtime(
    plan: _ArmPlan,
    *,
    checkpoint_dir: Path,
    client: Any | None,
    max_concurrency: int,
) -> FastCompletionRuntime:
    return FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=[row.messages for row in plan.rows],
        model=DEFAULT_MODEL,
        client=client,
        max_prompt_tokens=RESPONDER_PROMPT_CAP,
        max_new_tokens=RESPONDER_OUTPUT_TOKEN_RESERVE,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "experiment_format": RUN_FORMAT,
            "arm_label": ARM_LABEL,
            "arm_identity_sha256": plan.arm_identity_sha256,
            "preflight_artifact_sha256": plan.preflight_sha256,
            "retrieval_sha256": plan.population.retrieval_sha256,
            "baseline_final_answers_sha256": (
                plan.population.baseline_final_answers_sha256
            ),
            "population_identity_sha256": plan.population.population_identity_sha256,
            "authorized_unique_calls": plan.exact_calls,
            "gateway_url": CENTRAL_DEV_GATEWAY_URL,
            "gold_loaded": False,
        },
    )


def _run_artifact(plan: _ArmPlan, batch: FastCompletionBatch) -> dict[str, Any]:
    records = {
        record.messages_sha256: record for record in batch.unique_records
    }
    questions: list[dict[str, Any]] = []
    for row, prediction in zip(plan.rows, batch.logical_completions, strict=True):
        record = records[row.provider_messages_sha256]
        if quote_sha256(prediction) != record.completion_sha256:
            raise RuntimeError("S0 completion changed after journal verification")
        questions.append(
            {
                "ordinal": row.ordinal,
                "question_id": row.question_id,
                "question_sha256": row.question_sha256,
                "dated_question_sha256": row.dated_question_sha256,
                "retrieval_question_part_sha256": row.retrieval_question_part_sha256,
                "source_stage_id": SOURCE_STAGE_ID,
                "stage_receipt_sha256": row.stage_receipt_sha256,
                "evidence_projection_sha256": row.evidence_projection_sha256,
                "provider_messages_sha256": row.provider_messages_sha256,
                "prompt_token_proxy": row.prompt_token_proxy,
                "source_binding_sha256": row.binding_sha256,
                "prediction": {
                    "text": prediction,
                    "sha256": record.completion_sha256,
                },
                "call_key_sha256": record.call_key_sha256,
                "request_journal_sha256": record.request_journal_sha256,
                "response_journal_sha256": record.response_journal_sha256,
            }
        )
    artifact = {
        "format": RUN_FORMAT,
        "arm_label": ARM_LABEL,
        "arm_identity": plan.arm_identity,
        "arm_identity_sha256": plan.arm_identity_sha256,
        "preflight_artifact_sha256": plan.preflight_sha256,
        "retrieval_sha256": plan.population.retrieval_sha256,
        "baseline_final_answers_sha256": (
            plan.population.baseline_final_answers_sha256
        ),
        "population_identity_sha256": plan.population.population_identity_sha256,
        "historical_validator_binding_sha256": plan.population.binding_sha256,
        "source_stage_id": SOURCE_STAGE_ID,
        "question_count": len(questions),
        "logical_answer_count": batch.usage.logical_calls,
        "unique_provider_prompt_count": batch.usage.unique_calls,
        "completion_batch": _stable_batch(batch),
        "questions": questions,
        "gold_loaded": False,
        "retained_request_token_state_bytes": 0,
    }
    if _contains_gold_key(artifact):
        raise RuntimeError("S0 run crossed the gold firewall")
    return artifact


def _args_plan(args: argparse.Namespace) -> _ArmPlan:
    return _prepare(
        retrieval_path=Path(args.retrieval),
        baseline_answers_path=Path(args.baseline_answers),
        expected_retrieval_sha256=EXPECTED_RETRIEVAL_SHA256,
        expected_baseline_answers_sha256=EXPECTED_BASELINE_ANSWERS_SHA256,
        expected_question_count=int(args.expected_question_count),
    )


def run_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    plan = _args_plan(args)
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("preflight forbids provider access and authorization")
    artifact = dict(plan.preflight)
    return artifact, _publish(Path(args.output_root) / "preflight.json", artifact)


def run_arm(args: argparse.Namespace) -> tuple[dict[str, Any], str, int]:
    plan = _args_plan(args)
    if not args.enable_provider:
        raise ValueError("run requires --enable-provider")
    if args.authorized_provider_calls != plan.exact_calls:
        raise ValueError(
            "--authorized-provider-calls must exactly equal the sealed S0 "
            f"population ({args.authorized_provider_calls} != {plan.exact_calls})"
        )
    output_root = Path(args.output_root)
    _publish(output_root / "preflight.json", plan.preflight)
    run_path = output_root / "run.json"
    if run_path.exists():
        existing, _ = _read(run_path)
        if (
            existing.get("arm_identity_sha256") != plan.arm_identity_sha256
            or existing.get("preflight_artifact_sha256") != plan.preflight_sha256
        ):
            raise FileExistsError("existing S0 run belongs to another sealed arm")
    load_dotenv()
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    if not api_key:
        raise RuntimeError(f"provider API key is empty: {args.api_key_env}")
    client = _make_provider_client(api_key, CENTRAL_DEV_GATEWAY_URL)
    try:
        batch = _runtime(
            plan,
            checkpoint_dir=output_root / "terra-answer-calls",
            client=client,
            max_concurrency=int(args.max_concurrency),
        ).run()
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    observed = batch.usage.physical_calls + batch.usage.checkpoint_hits
    if observed != plan.exact_calls:
        raise RuntimeError("S0 runtime journal population changed")
    artifact = _run_artifact(plan, batch)
    digest = _publish(run_path, artifact)
    return artifact, digest, batch.usage.physical_calls


def load_verified_run(
    run_path: str | Path,
    *,
    expected_run_sha256: str,
    retrieval_path: str | Path = DEFAULT_RETRIEVAL,
    baseline_answers_path: str | Path = DEFAULT_BASELINE_ANSWERS,
    checkpoint_dir: str | Path | None = None,
    max_concurrency: int = 4,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    expected_retrieval_sha256: str = EXPECTED_RETRIEVAL_SHA256,
    expected_baseline_answers_sha256: str = EXPECTED_BASELINE_ANSWERS_SHA256,
) -> tuple[dict[str, Any], str]:
    """Historically validate sources, replay journals, and return one sealed run."""

    plan = _prepare(
        retrieval_path=Path(retrieval_path),
        baseline_answers_path=Path(baseline_answers_path),
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_baseline_answers_sha256=expected_baseline_answers_sha256,
        expected_question_count=expected_question_count,
    )
    source, source_sha = _read(Path(run_path))
    if source_sha != expected_run_sha256:
        raise ValueError("S0 run artifact SHA-256 changed")
    runtime = _runtime(
        plan,
        checkpoint_dir=Path(checkpoint_dir or Path(run_path).parent / "terra-answer-calls"),
        client=None,
        max_concurrency=max_concurrency,
    )
    batch = runtime.run()
    if batch.usage.physical_calls or batch.usage.checkpoint_hits != plan.exact_calls:
        raise RuntimeError("S0 replay did not consume the complete sealed journal set")
    expected = _run_artifact(plan, batch)
    if canonical_json_bytes(source) != canonical_json_bytes(expected):
        raise ValueError("S0 run differs from its immutable runtime journals")
    return source, source_sha


def run_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("replay forbids provider access and authorization")
    run_path = Path(args.run_artifact or Path(args.output_root) / "run.json")
    source, source_sha = _read(run_path)
    if args.expected_run_sha256 and source_sha != args.expected_run_sha256:
        raise ValueError("--expected-run-sha256 differs from the sealed run")
    verified, verified_sha = load_verified_run(
        run_path,
        expected_run_sha256=source_sha,
        retrieval_path=args.retrieval,
        baseline_answers_path=args.baseline_answers,
        checkpoint_dir=Path(args.output_root) / "terra-answer-calls",
        max_concurrency=int(args.max_concurrency),
        expected_question_count=int(args.expected_question_count),
    )
    if verified_sha != source_sha:
        raise RuntimeError("S0 replay changed the source digest")
    replay_path = Path(args.run_replay or Path(args.output_root) / "run-replay.json")
    return verified, _publish(replay_path, verified)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "run", "replay"))
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--baseline-answers", type=Path, default=DEFAULT_BASELINE_ANSWERS
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-artifact", type=Path)
    parser.add_argument("--run-replay", type=Path)
    parser.add_argument("--expected-run-sha256")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    parser.set_defaults(expected_question_count=EXPECTED_QUESTION_COUNT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "preflight":
        artifact, digest = run_preflight(args)
        print(
            f"{ARM_LABEL} preflight: logical={artifact['question_count']}; "
            f"unique={artifact['required_authorized_provider_calls']}; "
            f"provider_calls=0; sha256={digest}"
        )
        return 0
    if args.phase == "run":
        artifact, digest, physical = run_arm(args)
        print(
            f"{ARM_LABEL} run: answers={artifact['logical_answer_count']}; "
            f"physical_calls={physical}; sha256={digest}"
        )
        return 0
    artifact, digest = run_replay(args)
    print(
        f"{ARM_LABEL} replay: answers={artifact['logical_answer_count']}; "
        f"physical_calls=0; sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_LABEL",
    "PREFLIGHT_FORMAT",
    "RUN_FORMAT",
    "load_verified_run",
    "main",
    "run_arm",
    "run_preflight",
    "run_replay",
]
