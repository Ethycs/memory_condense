"""Answer and judge the sealed locked100 hot raw-chunk selection.

``answer`` has no benchmark loader: it sends only the 100 already-packed A3
messages to Terra.  ``judge`` separately authenticates those predictions,
opens the locked references, and sends question + reference + prediction to
Sol.  Both phases are zero-retry, concurrent, checkpointed, and immutable.
"""

from __future__ import annotations

import argparse
import importlib
import os
import statistics
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
from memory_condense.eval.recall_guarded_cumulative_population import (
    LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    build_locked_cumulative_population_identity,
)

try:
    from tools import assay_hot_retrieval_full100 as assay
    from tools.matched_eval.provider_runtime import (
        DEFAULT_API_KEY_ENV,
        DEFAULT_GATEWAY_URL,
        make_provider_client,
    )
except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
    import assay_hot_retrieval_full100 as assay
    from matched_eval.provider_runtime import (
        DEFAULT_API_KEY_ENV,
        DEFAULT_GATEWAY_URL,
        make_provider_client,
    )


ANSWER_FORMAT = "memory-condense-hot-retrieval-full100-answers-v1"
JUDGE_FORMAT = "memory-condense-hot-retrieval-full100-semantic-judge-v1"
TERRA_MODEL = "codex_sdk/gpt-5.6-terra"
SOL_MODEL = "codex_sdk/gpt-5.6-sol"
ANSWER_MAX_PROMPT_TOKENS = 8_000
SPINE_EPISODIC_FACT_ANSWER_MAX_PROMPT_TOKENS = 11_000
SPINE_EPISODIC_FACT_RESERVED_ANSWER_MAX_PROMPT_TOKENS = 12_000
ANSWER_MAX_TOKENS = 256
JUDGE_MAX_PROMPT_TOKENS = 4_096
ANSWERS_NAME = "answers.json"
JUDGMENTS_NAME = "answer-judgments.json"
SPINE_EPISODIC_FACT_PROFILE = "spine-episodic-fact-v3"
SPINE_EPISODIC_FACT_RESERVED_PROFILE = "spine-episodic-fact-reserved-v7"
SPINE_EPISODIC_FACT_R3_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v6-spine-episode-fact-ledger-"
    "full100-20260907-r3"
)
SPINE_EPISODIC_FACT_R3_SELECTION_SHA256 = (
    "9de07c09a8136b402158d05cef282b96d461b7baa02c0f0f239564cf39893965"
)
SELECTION_PROFILES = (
    "frozen-v6",
    "adaptive-v7",
    "source-seed-hybrid-v3",
    "user-envelope-shadow-v1",
    "operation-aware-v1",
    "user-spine-v1",
    SPINE_EPISODIC_FACT_PROFILE,
    SPINE_EPISODIC_FACT_RESERVED_PROFILE,
)


def _selection_assay(profile: str) -> Any:
    """Return the exact receipt validator for the requested selection policy."""

    if profile == "frozen-v6":
        return assay
    if profile == "adaptive-v7":
        try:
            return importlib.import_module(
                "tools.assay_hot_retrieval_adaptive_full100"
            )
        except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
            return importlib.import_module(
                "assay_hot_retrieval_adaptive_full100"
            )
    if profile == "source-seed-hybrid-v3":
        try:
            return importlib.import_module(
                "tools.assay_hot_retrieval_source_seed_hybrid_full100"
            )
        except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
            return importlib.import_module(
                "assay_hot_retrieval_source_seed_hybrid_full100"
            )
    if profile == "user-envelope-shadow-v1":
        try:
            return importlib.import_module(
                "tools.assay_hot_v4_user_envelope_provider_selection"
            )
        except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
            return importlib.import_module(
                "assay_hot_v4_user_envelope_provider_selection"
            )
    if profile == "operation-aware-v1":
        try:
            return importlib.import_module(
                "tools.assay_hot_v5_operation_aware_provider_selection"
            )
        except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
            return importlib.import_module(
                "assay_hot_v5_operation_aware_provider_selection"
            )
    if profile == "user-spine-v1":
        try:
            return importlib.import_module(
                "tools.assay_hot_v5_user_spine_provider_selection"
            )
        except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
            return importlib.import_module(
                "assay_hot_v5_user_spine_provider_selection"
            )
    if profile == SPINE_EPISODIC_FACT_PROFILE:
        try:
            return importlib.import_module(
                "tools.assay_hot_v6_spine_episode_fact_ledger_full100"
            )
        except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
            return importlib.import_module(
                "assay_hot_v6_spine_episode_fact_ledger_full100"
            )
    if profile == SPINE_EPISODIC_FACT_RESERVED_PROFILE:
        try:
            return importlib.import_module(
                "tools.assay_hot_v7_spine_episode_fact_reserved_full100"
            )
        except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
            return importlib.import_module(
                "assay_hot_v7_spine_episode_fact_reserved_full100"
            )
    raise ValueError(f"unknown selection profile: {profile}")


def _answer_max_prompt_tokens(selection_profile: str) -> int:
    if selection_profile == SPINE_EPISODIC_FACT_RESERVED_PROFILE:
        return SPINE_EPISODIC_FACT_RESERVED_ANSWER_MAX_PROMPT_TOKENS
    if selection_profile == SPINE_EPISODIC_FACT_PROFILE:
        return SPINE_EPISODIC_FACT_ANSWER_MAX_PROMPT_TOKENS
    return ANSWER_MAX_PROMPT_TOKENS


def _completion_client(
    api_key_env: str,
    gateway_url: str,
    dotenv_path: Path | None,
) -> Any:
    if dotenv_path is not None:
        load_dotenv(dotenv_path=dotenv_path, override=False)
    else:
        load_dotenv(override=False)
    key = os.environ.get(api_key_env, "").strip()
    if not key:
        raise RuntimeError(f"provider API key is empty: {api_key_env}")
    return make_provider_client(key, gateway_url)


def _load_selection(
    output_root: Path,
    expected_selection_sha256: str,
    *,
    selection_profile: str = "frozen-v6",
) -> tuple[dict[str, Any], str]:
    loader = _selection_assay(selection_profile)
    load_selection = getattr(loader, "load_selection", None)
    if not callable(load_selection):
        load_selection = loader._load_selection  # noqa: SLF001
    selection, digest = load_selection(output_root)
    if digest != expected_selection_sha256:
        raise ValueError(
            f"selection digest changed ({digest} != {expected_selection_sha256})"
        )
    if (
        selection.get("gold_fields_present") is not False
        or selection.get("provider_calls") != 0
        or len(selection.get("questions", [])) != assay.EXPECTED_QUESTION_COUNT
    ):
        raise ValueError("selection is not the sealed gold-free locked100 packet")
    return selection, digest


def _records_by_messages_sha(batch: Any) -> dict[str, Any]:
    records = {record.messages_sha256: record for record in batch.unique_records}
    if len(records) != batch.usage.unique_calls:
        raise RuntimeError("completion record population changed")
    return records


def _latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values = sorted(float(row["provider_elapsed_s"]) for row in rows)
    if not values:
        raise ValueError("cannot summarize empty provider timings")
    return {
        "min_s": values[0],
        "p50_s": statistics.median(values),
        "mean_s": statistics.fmean(values),
        "p95_s": values[max(0, (95 * len(values) + 99) // 100 - 1)],
        "max_s": values[-1],
    }


def answer(
    *,
    output_root: Path,
    expected_selection_sha256: str,
    gateway_url: str,
    api_key_env: str,
    dotenv_path: Path | None,
    max_concurrency: int,
    authorized_provider_calls: int,
    selection_profile: str = "frozen-v6",
) -> str:
    path = output_root / ANSWERS_NAME
    if path.exists():
        _selection, selection_sha = _load_selection(
            output_root,
            expected_selection_sha256,
            selection_profile=selection_profile,
        )
        _body, digest = _load_answers(
            output_root,
            selection_sha,
            selection=_selection,
            selection_profile=selection_profile,
        )
        print(f"Locked100 Terra answers verified: {path} ({digest})", flush=True)
        return digest
    selection, selection_sha = _load_selection(
        output_root,
        expected_selection_sha256,
        selection_profile=selection_profile,
    )
    prompts = [
        row["arms"]["a3_protected_union"]["provider_messages"]
        for row in selection["questions"]
    ]
    client = _completion_client(api_key_env, gateway_url, dotenv_path)
    runtime: FastCompletionRuntime | None = None
    try:
        runtime = FastCompletionRuntime(
            checkpoint_dir=output_root / "answer-checkpoints",
            prompt_population=prompts,
            model=TERRA_MODEL,
            client=client,
            max_prompt_tokens=_answer_max_prompt_tokens(selection_profile),
            max_new_tokens=ANSWER_MAX_TOKENS,
            max_concurrency=max_concurrency,
            retries=0,
            benchmark_provenance={
                "phase": "hot_raw_chunk_locked100_answer",
                "selection_sha256": selection_sha,
                "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
                "arm_id": "a3_protected_union",
                "gold_available": False,
                "gateway_url": gateway_url,
            },
        )
        if (
            runtime.population.logical_prompt_count != assay.EXPECTED_QUESTION_COUNT
            or runtime.population.unique_prompt_count != assay.EXPECTED_QUESTION_COUNT
        ):
            raise ValueError("Terra provider population is not exactly 100 unique prompts")
        with runtime._journal_guard():  # noqa: SLF001
            checkpoint_hits = len(runtime._load_all_records())  # noqa: SLF001
        missing = runtime.population.unique_prompt_count - checkpoint_hits
        if missing != authorized_provider_calls:
            raise ValueError(
                "Terra authorization must equal authenticated missing calls "
                f"({authorized_provider_calls} != {missing})"
            )
        batch = runtime.run()
    finally:
        if runtime is not None:
            runtime.close()
        else:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    if batch.usage.physical_calls != authorized_provider_calls:
        raise RuntimeError("Terra physical-call count differs from authorization")
    records = _records_by_messages_sha(batch)
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
                "shard_offset": int(selected["shard_offset"]),
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
    if len(rows) != assay.EXPECTED_QUESTION_COUNT:
        raise RuntimeError("Terra did not return the complete locked100 population")
    body = {
        "format": ANSWER_FORMAT,
        "status": "sealed_terra_predictions_without_gold",
        "selection_sha256": selection_sha,
        "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
        "prompt_population_sha256": batch.prompt_population.prompt_population_sha256,
        "runtime_identity_sha256": batch.runtime_identity_sha256,
        "gateway_url": gateway_url,
        "model": TERRA_MODEL,
        "max_completion_tokens": ANSWER_MAX_TOKENS,
        "max_concurrency": max_concurrency,
        "authorized_provider_calls": authorized_provider_calls,
        "retries": 0,
        "usage": batch.usage.model_dump(),
        "provider_latency": _latency_summary(rows),
        "questions": rows,
        "gold_fields_present": False,
        "retained_request_token_state_bytes": 0,
    }
    digest = assay.hot._atomic_write_json(path, body)  # noqa: SLF001
    print(f"Locked100 Terra answers published: {path} ({digest})", flush=True)
    return digest


def _load_answers(
    output_root: Path,
    selection_sha256: str,
    *,
    selection: Mapping[str, Any] | None = None,
    selection_profile: str = "frozen-v6",
) -> tuple[dict[str, Any], str]:
    body, digest = assay.hot._read_json_artifact(  # noqa: SLF001
        output_root / ANSWERS_NAME
    )
    rows = body.get("questions")
    if (
        body.get("format") != ANSWER_FORMAT
        or body.get("status") != "sealed_terra_predictions_without_gold"
        or body.get("selection_sha256") != selection_sha256
        or body.get("population_identity_sha256")
        != assay.EXPECTED_POPULATION_SHA256
        or body.get("gold_fields_present") is not False
        or body.get("retries") != 0
        or not isinstance(rows, list)
        or len(rows) != assay.EXPECTED_QUESTION_COUNT
        or [row.get("ordinal") for row in rows]
        != list(range(assay.EXPECTED_QUESTION_COUNT))
    ):
        raise ValueError("sealed Terra answer artifact changed")
    for row in rows:
        prediction = str(row.get("prediction", ""))
        if row.get("prediction_sha256") != quote_sha256(prediction):
            raise ValueError("sealed Terra prediction changed")
    if selection is None:
        selection, observed_sha = _load_selection(
            output_root,
            selection_sha256,
            selection_profile=selection_profile,
        )
        if observed_sha != selection_sha256:
            raise ValueError("Terra answers bind another selection")
    selected_rows = selection.get("questions")
    if not isinstance(selected_rows, list) or len(selected_rows) != len(rows):
        raise ValueError("Terra answer/selection population changed")
    for answer_row, selected in zip(rows, selected_rows, strict=True):
        arm = selected["arms"]["a3_protected_union"]
        if any(
            answer_row.get(key) != selected.get(key)
            for key in ("ordinal", "shard_offset", "question_id")
        ) or answer_row.get("provider_payload_sha256") != arm.get(
            "provider_payload_sha256"
        ):
            raise ValueError("Terra answer row differs from its sealed prompt")
        for field in (
            "messages_sha256",
            "call_key_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
        ):
            value = answer_row.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"Terra answer row has invalid {field}")
    prompts = [
        row["arms"]["a3_protected_union"]["provider_messages"]
        for row in selected_rows
    ]
    concurrency = body.get("max_concurrency")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("Terra answer artifact has invalid concurrency")
    replay_runtime = FastCompletionRuntime(
        checkpoint_dir=output_root / "answer-checkpoints",
        prompt_population=prompts,
        model=TERRA_MODEL,
        client=None,
        max_prompt_tokens=_answer_max_prompt_tokens(selection_profile),
        max_new_tokens=ANSWER_MAX_TOKENS,
        max_concurrency=concurrency,
        retries=0,
        benchmark_provenance={
            "phase": "hot_raw_chunk_locked100_answer",
            "selection_sha256": selection_sha256,
            "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
            "arm_id": "a3_protected_union",
            "gold_available": False,
            "gateway_url": body.get("gateway_url"),
        },
    )
    try:
        replay = replay_runtime.run()
    finally:
        replay_runtime.close()
    if (
        body.get("model") != TERRA_MODEL
        or body.get("max_completion_tokens") != ANSWER_MAX_TOKENS
        or body.get("runtime_identity_sha256") != replay.runtime_identity_sha256
        or body.get("prompt_population_sha256")
        != replay.prompt_population.prompt_population_sha256
        or list(replay.logical_completions)
        != [str(row["prediction"]) for row in rows]
        or replay.usage.physical_calls != 0
        or replay.usage.checkpoint_hits != assay.EXPECTED_QUESTION_COUNT
    ):
        raise ValueError("Terra answer journals do not byte-replay")
    record_by_message = _records_by_messages_sha(replay)
    for answer_row, prompt_row in zip(
        rows, replay.prompt_population.ordered_rows, strict=True
    ):
        record = record_by_message[prompt_row.messages_sha256]
        if (
            answer_row.get("messages_sha256") != prompt_row.messages_sha256
            or answer_row.get("call_key_sha256") != record.call_key_sha256
            or answer_row.get("request_journal_sha256")
            != record.request_journal_sha256
            or answer_row.get("response_journal_sha256")
            != record.response_journal_sha256
        ):
            raise ValueError("Terra answer journal binding changed")
    return body, digest


def _load_judgments(
    output_root: Path,
    *,
    selection_sha256: str,
    answers_sha256: str,
) -> tuple[dict[str, Any], str]:
    body, digest = assay.hot._read_json_artifact(  # noqa: SLF001
        output_root / JUDGMENTS_NAME
    )
    rows = body.get("rows")
    if (
        body.get("format") != JUDGE_FORMAT
        or body.get("status") != "sealed_sol_semantic_accuracy"
        or body.get("selection_sha256") != selection_sha256
        or body.get("answers_sha256") != answers_sha256
        or body.get("population_identity_sha256")
        != assay.EXPECTED_POPULATION_SHA256
        or body.get("questions") != assay.EXPECTED_QUESTION_COUNT
        or body.get("gold_fields_present") is not True
        or body.get("retries") != 0
        or not isinstance(rows, list)
        or len(rows) != assay.EXPECTED_QUESTION_COUNT
        or [row.get("ordinal") for row in rows]
        != list(range(assay.EXPECTED_QUESTION_COUNT))
    ):
        raise ValueError("sealed Sol judgment artifact changed")
    correct = sum(row.get("correct") is True for row in rows)
    if body.get("correct") != correct or body.get("accuracy") != correct / len(rows):
        raise ValueError("sealed Sol score does not reproduce")
    for row in rows:
        if type(row.get("correct")) is not bool:
            raise ValueError("sealed Sol verdict is not binary")
        for field in (
            "judge_messages_sha256",
            "verdict_sha256",
            "call_key_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
        ):
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"Sol judgment row has invalid {field}")
    return body, digest


def _build_judge_prompts(
    answers: Mapping[str, Any],
    questions: Sequence[Any],
) -> list[list[dict[str, str]]]:
    answer_rows = answers.get("questions")
    if not isinstance(answer_rows, list) or len(answer_rows) != len(questions):
        raise ValueError("answer/judge population changed")
    prompts: list[list[dict[str, str]]] = []
    for answer_row, question in zip(answer_rows, questions, strict=True):
        if answer_row.get("question_id") != question.question_id:
            raise ValueError("answer order differs from locked judge population")
        prompts.append(
            build_judge_prompt(
                question.question,
                question.answer,
                str(answer_row["prediction"]),
            )
        )
    return prompts


def _verify_judgment_journals(
    *,
    output_root: Path,
    body: Mapping[str, Any],
    selection_sha256: str,
    answers: Mapping[str, Any],
    answers_sha256: str,
    questions: Sequence[Any],
) -> None:
    rows = body.get("rows")
    if not isinstance(rows, list):
        raise ValueError("sealed Sol rows changed")
    prompts = _build_judge_prompts(answers, questions)
    concurrency = body.get("max_concurrency")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("Sol judgment artifact has invalid concurrency")
    gateway_url = body.get("gateway_url")
    if not isinstance(gateway_url, str) or not gateway_url:
        raise ValueError("Sol judgment artifact has invalid gateway URL")
    replay_runtime = FastCompletionRuntime(
        checkpoint_dir=output_root / "judge-checkpoints",
        prompt_population=prompts,
        model=SOL_MODEL,
        client=None,
        max_prompt_tokens=JUDGE_MAX_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=concurrency,
        retries=0,
        benchmark_provenance={
            "phase": "hot_raw_chunk_locked100_semantic_judge",
            "selection_sha256": selection_sha256,
            "answers_sha256": answers_sha256,
            "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
            "gateway_url": gateway_url,
        },
    )
    try:
        replay = replay_runtime.run()
    finally:
        replay_runtime.close()
    if (
        body.get("model") != SOL_MODEL
        or body.get("max_completion_tokens") != JUDGE_MAX_TOKENS
        or body.get("runtime_identity_sha256") != replay.runtime_identity_sha256
        or body.get("judge_prompt_population_sha256")
        != replay.prompt_population.prompt_population_sha256
        or replay.usage.physical_calls != 0
        or replay.usage.checkpoint_hits != assay.EXPECTED_QUESTION_COUNT
    ):
        raise ValueError("Sol judgment journals do not byte-replay")
    records = _records_by_messages_sha(replay)
    answer_rows = answers["questions"]
    for row, answer_row, question, prompt_row, verdict in zip(
        rows,
        answer_rows,
        questions,
        replay.prompt_population.ordered_rows,
        replay.logical_completions,
        strict=True,
    ):
        record = records[prompt_row.messages_sha256]
        expected = {
            "ordinal": int(answer_row["ordinal"]),
            "shard_offset": int(answer_row["shard_offset"]),
            "question_id": question.question_id,
            "category": question.category,
            "question_sha256": quote_sha256(question.question),
            "reference_sha256": quote_sha256(question.answer),
            "prediction_sha256": answer_row["prediction_sha256"],
            "judge_messages_sha256": prompt_row.messages_sha256,
            "verdict_sha256": quote_sha256(verdict),
            "correct": parse_binary_judge_verdict(verdict),
            "call_key_sha256": record.call_key_sha256,
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "provider_elapsed_s": record.provider_elapsed_s,
        }
        if row != expected:
            raise ValueError("Sol judgment row differs from its sealed journal")
    correct_count = sum(row["correct"] is True for row in rows)
    categories = sorted({str(row["category"]) for row in rows})
    expected_by_category = {
        category: {
            "questions": sum(row["category"] == category for row in rows),
            "correct": sum(
                row["category"] == category and row["correct"] is True
                for row in rows
            ),
        }
        for category in categories
    }
    usage = body.get("usage")
    authorized = body.get("authorized_provider_calls")
    if (
        body.get("correct") != correct_count
        or body.get("accuracy") != correct_count / len(rows)
        or body.get("by_category") != expected_by_category
        or body.get("provider_latency") != _latency_summary(rows)
        or not isinstance(usage, Mapping)
        or isinstance(authorized, bool)
        or not isinstance(authorized, int)
        or authorized < 0
        or usage.get("physical_calls") != authorized
        or usage.get("physical_calls", 0) + usage.get("checkpoint_hits", 0)
        != assay.EXPECTED_QUESTION_COUNT
        or usage.get("logical_calls") != assay.EXPECTED_QUESTION_COUNT
        or usage.get("unique_calls") != assay.EXPECTED_QUESTION_COUNT
    ):
        raise ValueError("Sol judgment aggregate differs from sealed journals")


def judge(
    *,
    output_root: Path,
    expected_selection_sha256: str,
    dataset: Path,
    split_manifest: Path,
    gateway_url: str,
    api_key_env: str,
    dotenv_path: Path | None,
    max_concurrency: int,
    authorized_provider_calls: int,
    selection_profile: str = "frozen-v6",
) -> str:
    path = output_root / JUDGMENTS_NAME
    if path.exists():
        selection, selection_sha = _load_selection(
            output_root,
            expected_selection_sha256,
            selection_profile=selection_profile,
        )
        _answers, answers_sha = _load_answers(
            output_root,
            selection_sha,
            selection=selection,
            selection_profile=selection_profile,
        )
        samples, _identities, population = build_locked_cumulative_population_identity(
            dataset,
            split_manifest,
            plan=LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
        )
        if (
            population.get("population_identity_sha256")
            != assay.EXPECTED_POPULATION_SHA256
        ):
            raise RuntimeError("judge population identity changed")
        questions = [question for sample in samples for question in sample.questions]
        judgment_body, digest = _load_judgments(
            output_root,
            selection_sha256=selection_sha,
            answers_sha256=answers_sha,
        )
        _verify_judgment_journals(
            output_root=output_root,
            body=judgment_body,
            selection_sha256=selection_sha,
            answers=_answers,
            answers_sha256=answers_sha,
            questions=questions,
        )
        print(f"Locked100 Sol judgments verified: {path} ({digest})", flush=True)
        return digest
    _selection, selection_sha = _load_selection(
        output_root,
        expected_selection_sha256,
        selection_profile=selection_profile,
    )
    answers, answers_sha = _load_answers(
        output_root,
        selection_sha,
        selection=_selection,
        selection_profile=selection_profile,
    )
    samples, _identities, population = build_locked_cumulative_population_identity(
        dataset,
        split_manifest,
        plan=LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    )
    if population.get("population_identity_sha256") != assay.EXPECTED_POPULATION_SHA256:
        raise RuntimeError("judge population identity changed")
    questions = [question for sample in samples for question in sample.questions]
    prompts = _build_judge_prompts(answers, questions)
    client = _completion_client(api_key_env, gateway_url, dotenv_path)
    runtime: FastCompletionRuntime | None = None
    try:
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
                "phase": "hot_raw_chunk_locked100_semantic_judge",
                "selection_sha256": selection_sha,
                "answers_sha256": answers_sha,
                "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
                "gateway_url": gateway_url,
            },
        )
        if (
            runtime.population.logical_prompt_count != assay.EXPECTED_QUESTION_COUNT
            or runtime.population.unique_prompt_count != assay.EXPECTED_QUESTION_COUNT
        ):
            raise ValueError("Sol provider population is not exactly 100 unique prompts")
        with runtime._journal_guard():  # noqa: SLF001
            checkpoint_hits = len(runtime._load_all_records())  # noqa: SLF001
        missing = runtime.population.unique_prompt_count - checkpoint_hits
        if missing != authorized_provider_calls:
            raise ValueError(
                "Sol authorization must equal authenticated missing calls "
                f"({authorized_provider_calls} != {missing})"
            )
        batch = runtime.run()
    finally:
        if runtime is not None:
            runtime.close()
        else:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    if batch.usage.physical_calls != authorized_provider_calls:
        raise RuntimeError("Sol physical-call count differs from authorization")
    records = _records_by_messages_sha(batch)
    rows: list[dict[str, Any]] = []
    for answer_row, question, population_row, verdict_text in zip(
        answers["questions"],
        questions,
        batch.prompt_population.ordered_rows,
        batch.logical_completions,
        strict=True,
    ):
        record = records[population_row.messages_sha256]
        correct = parse_binary_judge_verdict(verdict_text)
        rows.append(
            {
                "ordinal": int(answer_row["ordinal"]),
                "shard_offset": int(answer_row["shard_offset"]),
                "question_id": question.question_id,
                "category": question.category,
                "question_sha256": quote_sha256(question.question),
                "reference_sha256": quote_sha256(question.answer),
                "prediction_sha256": answer_row["prediction_sha256"],
                "judge_messages_sha256": population_row.messages_sha256,
                "verdict_sha256": quote_sha256(verdict_text),
                "correct": correct,
                "call_key_sha256": record.call_key_sha256,
                "request_journal_sha256": record.request_journal_sha256,
                "response_journal_sha256": record.response_journal_sha256,
                "provider_elapsed_s": record.provider_elapsed_s,
            }
        )
    correct_count = sum(row["correct"] is True for row in rows)
    categories = sorted({str(row["category"]) for row in rows})
    by_category = {
        category: {
            "questions": sum(row["category"] == category for row in rows),
            "correct": sum(
                row["category"] == category and row["correct"] is True
                for row in rows
            ),
        }
        for category in categories
    }
    body = {
        "format": JUDGE_FORMAT,
        "status": "sealed_sol_semantic_accuracy",
        "selection_sha256": selection_sha,
        "answers_sha256": answers_sha,
        "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
        "judge_prompt_population_sha256": batch.prompt_population.prompt_population_sha256,
        "runtime_identity_sha256": batch.runtime_identity_sha256,
        "gateway_url": gateway_url,
        "model": SOL_MODEL,
        "max_completion_tokens": JUDGE_MAX_TOKENS,
        "max_concurrency": max_concurrency,
        "authorized_provider_calls": authorized_provider_calls,
        "retries": 0,
        "questions": len(rows),
        "correct": correct_count,
        "accuracy": correct_count / len(rows),
        "by_category": by_category,
        "usage": batch.usage.model_dump(),
        "provider_latency": _latency_summary(rows),
        "rows": rows,
        "gold_fields_present": True,
        "retained_request_token_state_bytes": 0,
    }
    digest = assay.hot._atomic_write_json(path, body)  # noqa: SLF001
    print(
        f"Locked100 Sol judgments published: {path} ({digest}); "
        f"accuracy={correct_count}/{len(rows)}",
        flush=True,
    )
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=assay.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--dotenv-path", type=Path)
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--authorized-provider-calls", type=int, required=True)
    parser.add_argument(
        "--selection-profile",
        choices=SELECTION_PROFILES,
        default="frozen-v6",
        help="Receipt contract used to validate selection.json.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("answer")
    judge_parser = commands.add_parser("judge")
    judge_parser.add_argument("--dataset", type=Path, required=True)
    judge_parser.add_argument("--split-manifest", type=Path, default=assay.DEFAULT_SPLIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_concurrency < 1:
        raise ValueError("max concurrency must be positive")
    if args.authorized_provider_calls < 0:
        raise ValueError("authorized provider calls cannot be negative")
    output_root = args.output_root.resolve()
    dotenv_path = None if args.dotenv_path is None else args.dotenv_path.resolve()
    if args.command == "answer":
        answer(
            output_root=output_root,
            expected_selection_sha256=args.expected_selection_sha256,
            gateway_url=args.gateway_url,
            api_key_env=args.api_key_env,
            dotenv_path=dotenv_path,
            max_concurrency=args.max_concurrency,
            authorized_provider_calls=args.authorized_provider_calls,
            selection_profile=args.selection_profile,
        )
    elif args.command == "judge":
        judge(
            output_root=output_root,
            expected_selection_sha256=args.expected_selection_sha256,
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
            gateway_url=args.gateway_url,
            api_key_env=args.api_key_env,
            dotenv_path=dotenv_path,
            max_concurrency=args.max_concurrency,
            authorized_provider_calls=args.authorized_provider_calls,
            selection_profile=args.selection_profile,
        )
    else:  # pragma: no cover
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
