"""Answer and judge one sealed hot-retrieval raw-chunk packet.

``answer`` reads only the gold-free selection and sends its already-packed A3
messages to Terra. ``judge`` is a separate process that first verifies the
sealed predictions, then joins the pinned benchmark references and sends the
official binary prompts to Sol. Both phases use zero-retry completion journals.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval._binary_judge_protocol import (
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.eval.recall_guarded_cumulative_1m import (
    ORIGINAL_ORDERED_QUESTION_IDS,
    ORIGINAL_QUESTIONS,
    load_original_population,
    population_identity_sha256,
)
try:
    from tools import assay_hot_retrieval_1m as assay
    from tools.matched_eval.provider_runtime import (
        DEFAULT_API_KEY_ENV,
        DEFAULT_GATEWAY_URL,
        make_provider_client,
    )
except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
    import assay_hot_retrieval_1m as assay
    from matched_eval.provider_runtime import (
        DEFAULT_API_KEY_ENV,
        DEFAULT_GATEWAY_URL,
        make_provider_client,
    )


ANSWER_FORMAT = "memory-condense-hot-retrieval-answers-v1"
JUDGE_FORMAT = "memory-condense-hot-retrieval-semantic-judge-v1"
TERRA_MODEL = "codex_sdk/gpt-5.6-terra"
SOL_MODEL = "codex_sdk/gpt-5.6-sol"
ANSWER_MAX_PROMPT_TOKENS = 8_000
ANSWER_MAX_TOKENS = 256
JUDGE_MAX_PROMPT_TOKENS = 4_096
DEFAULT_SELECTION_SHA256 = (
    "cafe769331b36a2500b43d012360a775669e9ffdc4519bf6a115f83c767cba06"
)
DEFAULT_OUTPUT_ROOT = assay.DEFAULT_OUTPUT_ROOT
DEFAULT_ANSWERS_NAME = "answers.json"
DEFAULT_JUDGMENTS_NAME = "answer-judgments.json"


def _completion_client(api_key_env: str, gateway_url: str) -> Any:
    load_dotenv(override=False)
    key = os.environ.get(api_key_env, "").strip()
    if not key:
        raise RuntimeError(f"provider API key is empty: {api_key_env}")
    return make_provider_client(key, gateway_url)


def _load_bound_selection(
    output_root: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    selection, digest = assay._load_selection(output_root)  # noqa: SLF001
    if digest != expected_sha256:
        raise ValueError(
            f"selection digest changed ({digest} != {expected_sha256})"
        )
    rows = selection.get("questions")
    if (
        selection.get("format") != assay.SELECTION_FORMAT
        or selection.get("gold_fields_present") is not False
        or selection.get("provider_calls") != 0
        or not isinstance(rows, list)
        or len(rows) != ORIGINAL_QUESTIONS
    ):
        raise ValueError("selection is not the sealed gold-free dev1M packet")
    for ordinal, (row, question_id) in enumerate(
        zip(rows, ORIGINAL_ORDERED_QUESTION_IDS, strict=True)
    ):
        if row.get("ordinal") != ordinal or row.get("question_id") != question_id:
            raise ValueError("selection question order changed")
        arm = row.get("arms", {}).get("a3_protected_union")
        if not isinstance(arm, Mapping) or arm.get("raw_evidence_only") is not True:
            raise ValueError("selection omitted the raw A3 packet")
        messages = arm.get("provider_messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("A3 packet omitted provider messages")
        payload = assay._canonical_json_bytes({"messages": messages})  # noqa: SLF001
        if (
            arm.get("provider_payload_sha256")
            != hashlib.sha256(payload).hexdigest()
            or arm.get("provider_payload_utf8_bytes") != len(payload)
            or arm.get("prompt_workspace_token_proxy", 0)
            > ANSWER_MAX_PROMPT_TOKENS
        ):
            raise ValueError("A3 provider payload binding changed")
    return selection, digest


def _record_by_messages_sha(batch: Any) -> dict[str, Any]:
    records = {record.messages_sha256: record for record in batch.unique_records}
    if len(records) != batch.usage.unique_calls:
        raise RuntimeError("completion batch record population changed")
    return records


def answer_packet(
    *,
    output_root: Path,
    expected_selection_sha256: str,
    gateway_url: str,
    api_key_env: str,
    max_concurrency: int,
) -> str:
    existing = output_root / DEFAULT_ANSWERS_NAME
    if existing.exists():
        artifact, digest = assay._read_json_artifact(existing)  # noqa: SLF001
        if (
            artifact.get("format") != ANSWER_FORMAT
            or artifact.get("selection_sha256") != expected_selection_sha256
        ):
            raise ValueError("existing answer artifact has another binding")
        print(f"Sealed answers verified: {existing} ({digest})", flush=True)
        return digest

    selection, selection_sha = _load_bound_selection(
        output_root,
        expected_selection_sha256,
    )
    prompts = [
        row["arms"]["a3_protected_union"]["provider_messages"]
        for row in selection["questions"]
    ]
    client = _completion_client(api_key_env, gateway_url)
    runtime = FastCompletionRuntime(
        checkpoint_dir=output_root / "answer-checkpoints",
        prompt_population=prompts,
        model=TERRA_MODEL,
        client=client,
        max_prompt_tokens=ANSWER_MAX_PROMPT_TOKENS,
        max_new_tokens=ANSWER_MAX_TOKENS,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "phase": "hot_raw_chunk_answer",
            "selection_sha256": selection_sha,
            "arm_id": "a3_protected_union",
            "gold_available": False,
            "gateway_url": gateway_url,
        },
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
        close_client = getattr(client, "close", None)
        if callable(close_client):
            close_client()

    records = _record_by_messages_sha(batch)
    rows: list[dict[str, Any]] = []
    for selected, population_row, prediction in zip(
        selection["questions"],
        batch.prompt_population.ordered_rows,
        batch.logical_completions,
        strict=True,
    ):
        record = records[population_row.messages_sha256]
        rows.append(
            {
                "ordinal": int(selected["ordinal"]),
                "question_id": str(selected["question_id"]),
                "arm_id": "a3_protected_union",
                "provider_payload_sha256": selected["arms"][
                    "a3_protected_union"
                ]["provider_payload_sha256"],
                "messages_sha256": population_row.messages_sha256,
                "prediction": prediction,
                "prediction_sha256": quote_sha256(prediction),
                "call_key_sha256": record.call_key_sha256,
                "request_journal_sha256": record.request_journal_sha256,
                "response_journal_sha256": record.response_journal_sha256,
                "provider_elapsed_s": record.provider_elapsed_s,
            }
        )
    body = {
        "format": ANSWER_FORMAT,
        "status": "sealed_terra_predictions_without_gold",
        "selection_sha256": selection_sha,
        "prompt_population_sha256": (
            batch.prompt_population.prompt_population_sha256
        ),
        "runtime_identity_sha256": batch.runtime_identity_sha256,
        "gateway_url": gateway_url,
        "model": TERRA_MODEL,
        "max_completion_tokens": ANSWER_MAX_TOKENS,
        "max_concurrency": max_concurrency,
        "retries": 0,
        "usage": batch.usage.model_dump(),
        "questions": rows,
        "gold_fields_present": False,
        "retained_request_token_state_bytes": 0,
    }
    digest = assay._atomic_write_json(existing, body)  # noqa: SLF001
    print(f"Terra answers published: {existing} ({digest})", flush=True)
    return digest


def judge_answers(
    *,
    output_root: Path,
    expected_selection_sha256: str,
    dataset: Path,
    split_manifest: Path,
    gateway_url: str,
    api_key_env: str,
    max_concurrency: int,
) -> str:
    existing = output_root / DEFAULT_JUDGMENTS_NAME
    if existing.exists():
        artifact, digest = assay._read_json_artifact(existing)  # noqa: SLF001
        if (
            artifact.get("format") != JUDGE_FORMAT
            or artifact.get("selection_sha256") != expected_selection_sha256
        ):
            raise ValueError("existing judgment artifact has another binding")
        print(f"Sealed judgments verified: {existing} ({digest})", flush=True)
        return digest

    _selection, selection_sha = _load_bound_selection(
        output_root,
        expected_selection_sha256,
    )
    answers_path = output_root / DEFAULT_ANSWERS_NAME
    answers, answers_sha = assay._read_json_artifact(answers_path)  # noqa: SLF001
    answer_rows = answers.get("questions")
    if (
        answers.get("format") != ANSWER_FORMAT
        or answers.get("status") != "sealed_terra_predictions_without_gold"
        or answers.get("selection_sha256") != selection_sha
        or answers.get("gold_fields_present") is not False
        or answers.get("retries") != 0
        or not isinstance(answer_rows, list)
        or len(answer_rows) != ORIGINAL_QUESTIONS
        or [row.get("question_id") for row in answer_rows]
        != list(ORIGINAL_ORDERED_QUESTION_IDS)
    ):
        raise ValueError("sealed answer artifact changed")

    sample = load_original_population(dataset, split_manifest)
    population_sha = population_identity_sha256(sample)
    if population_sha != assay.EXPECTED_POPULATION_SHA256:
        raise RuntimeError("judge population identity changed")
    prompts: list[list[dict[str, str]]] = []
    for answer, question in zip(answer_rows, sample.questions, strict=True):
        prediction = str(answer["prediction"])
        if answer.get("prediction_sha256") != quote_sha256(prediction):
            raise ValueError("sealed prediction changed")
        prompts.append(
            build_judge_prompt(question.question, question.answer, prediction)
        )

    client = _completion_client(api_key_env, gateway_url)
    runtime = FastCompletionRuntime(
        checkpoint_dir=output_root / "judge-checkpoints",
        prompt_population=prompts,
        model=SOL_MODEL,
        client=client,
        max_prompt_tokens=JUDGE_MAX_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "phase": "hot_raw_chunk_semantic_judge",
            "selection_sha256": selection_sha,
            "answers_sha256": answers_sha,
            "population_identity_sha256": population_sha,
            "gateway_url": gateway_url,
        },
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
        close_client = getattr(client, "close", None)
        if callable(close_client):
            close_client()

    records = _record_by_messages_sha(batch)
    judged_rows: list[dict[str, Any]] = []
    for answer, question, population_row, verdict_text in zip(
        answer_rows,
        sample.questions,
        batch.prompt_population.ordered_rows,
        batch.logical_completions,
        strict=True,
    ):
        record = records[population_row.messages_sha256]
        correct = parse_binary_judge_verdict(verdict_text)
        judged_rows.append(
            {
                "ordinal": int(answer["ordinal"]),
                "question_id": question.question_id,
                "category": question.category,
                "question_sha256": quote_sha256(question.question),
                "reference_sha256": quote_sha256(question.answer),
                "prediction_sha256": answer["prediction_sha256"],
                "judge_messages_sha256": population_row.messages_sha256,
                "verdict_sha256": quote_sha256(verdict_text),
                "correct": correct,
                "call_key_sha256": record.call_key_sha256,
                "request_journal_sha256": record.request_journal_sha256,
                "response_journal_sha256": record.response_journal_sha256,
                "provider_elapsed_s": record.provider_elapsed_s,
            }
        )
    correct = sum(row["correct"] is True for row in judged_rows)
    body = {
        "format": JUDGE_FORMAT,
        "status": "sealed_sol_semantic_accuracy",
        "selection_sha256": selection_sha,
        "answers_sha256": answers_sha,
        "population_identity_sha256": population_sha,
        "judge_prompt_population_sha256": (
            batch.prompt_population.prompt_population_sha256
        ),
        "runtime_identity_sha256": batch.runtime_identity_sha256,
        "gateway_url": gateway_url,
        "model": SOL_MODEL,
        "max_completion_tokens": JUDGE_MAX_TOKENS,
        "max_concurrency": max_concurrency,
        "retries": 0,
        "questions": len(judged_rows),
        "correct": correct,
        "accuracy": correct / len(judged_rows),
        "usage": batch.usage.model_dump(),
        "rows": judged_rows,
        "gold_fields_present": True,
        "retained_request_token_state_bytes": 0,
    }
    digest = assay._atomic_write_json(existing, body)  # noqa: SLF001
    print(
        f"Sol judgments published: {existing} ({digest}); "
        f"accuracy={correct}/{len(judged_rows)}",
        flush=True,
    )
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--expected-selection-sha256",
        default=DEFAULT_SELECTION_SHA256,
    )
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--max-concurrency", type=int, default=10)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("answer", help="send sealed A3 raw packets to Terra")
    judge = commands.add_parser("judge", help="join gold and judge sealed answers")
    judge.add_argument("--dataset", type=Path, required=True)
    judge.add_argument("--split-manifest", type=Path, default=assay.DEFAULT_SPLIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    if args.command == "answer":
        answer_packet(
            output_root=output_root,
            expected_selection_sha256=args.expected_selection_sha256,
            gateway_url=args.gateway_url,
            api_key_env=args.api_key_env,
            max_concurrency=args.max_concurrency,
        )
    elif args.command == "judge":
        judge_answers(
            output_root=output_root,
            expected_selection_sha256=args.expected_selection_sha256,
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
            gateway_url=args.gateway_url,
            api_key_env=args.api_key_env,
            max_concurrency=args.max_concurrency,
        )
    else:  # pragma: no cover
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
