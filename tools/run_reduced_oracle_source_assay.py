#!/usr/bin/env python3
"""Post-hoc reduced-corpus control for the remaining typed-memory misses.

This diagnostic is intentionally outcome conditioned.  It selects the raw
LongMemEval sessions named by ``answer_session_ids`` for the 24 questions that
remain wrong after the miss-only replay.  Terra never sees the reference
answer, question ID, or source IDs.  When a complete labelled source set does
not fit the hard envelope, a deterministic question-only whole-turn fitter is
used and its omissions are sealed explicitly.

The result is an oracle-source diagnostic, never a benchmark score.  Its job
is to distinguish million-token retrieval/localization failures from answer
technique failures without relaxing the 8,000-token complete envelope.
"""

from __future__ import annotations

import argparse
import hashlib
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

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
)
from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval._binary_judge_protocol import (  # noqa: E402
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import (  # noqa: E402
    build_judge_prompt,
    build_qa_prompt,
    exact_match,
    f1_score,
)
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools.matched_eval import live  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
)


FORMAT = "memory-condense-reduced-oracle-source-assay-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-answer-preflight"
AUTHORITY_FORMAT = f"{FORMAT}-judge-authority"
ANSWER_FORMAT = f"{FORMAT}-answer-run"
JUDGE_FORMAT = f"{FORMAT}-judge-run"

PREFLIGHT_NAME = "answer-preflight.json"
AUTHORITY_NAME = "judge-authority.json"
ANSWER_NAME = "answer-run.json"
JUDGE_NAME = "judge-run.json"
ANSWER_CHECKPOINT_DIR = "terra-answer-calls"
JUDGE_CHECKPOINT_DIR = "sol-judge-calls"

EXPECTED_DATASET_SHA256 = (
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
)
EXPECTED_COMPOSITION_SHA256 = (
    "730a437e242174d188ae67484d9414d87c74d8ed926d9e4cdc726c7d5260317f"
)
REMAINING_ORDINALS = (
    6,
    7,
    14,
    16,
    28,
    31,
    36,
    42,
    43,
    49,
    53,
    54,
    61,
    65,
    67,
    69,
    72,
    77,
    79,
    81,
    86,
    93,
    94,
    97,
)
QUESTION_COUNT = len(REMAINING_ORDINALS)

GATEWAY_URL = "https://central-dev.zt:4000/v1"
TERRA_MODEL = "codex_sdk/gpt-5.6-terra"
SOL_MODEL = "codex_sdk/gpt-5.6-sol"
ANSWER_OUTPUT_RESERVE = 256
ANSWER_PROMPT_CAP = 8_000 - ANSWER_OUTPUT_RESERVE
JUDGE_PROMPT_CAP = 8_000 - JUDGE_MAX_TOKENS
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_DATASET = Path(
    r"C:\Users\Keytone\Downloads\memory-condense-rig\datasets"
    r"\longmemeval_s_cleaned.json"
)
DEFAULT_COMPOSITION = Path(
    "eval_results/matched_eval_100/typed-memory-final-v3-shared-surplus/"
    "typed-memory-final-composition-v1.json"
)
DEFAULT_OUTPUT = Path(
    "eval_results/matched_eval_100/reduced-oracle-source-remaining24-v1"
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "at",
        "be",
        "did",
        "do",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "last",
        "many",
        "me",
        "my",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)


class ReducedOracleSourceAssayError(MatchedEvalContractError):
    """A reduced oracle-source input, prompt, journal, or seal changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedOracleSourceAssayError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_dataset(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), "dataset must be a regular file")
    _require(_file_sha256(path) == EXPECTED_DATASET_SHA256, "dataset SHA-256 changed")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReducedOracleSourceAssayError("dataset is not strict JSON") from exc
    _require(type(value) is list and len(value) == 500, "dataset population changed")
    _require(all(type(row) is dict for row in value), "dataset rows changed type")
    return value


def _composition_rows(path: Path) -> tuple[SealedArtifact, dict[int, dict[str, Any]]]:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == EXPECTED_COMPOSITION_SHA256,
        "shared-surplus composition SHA-256 changed",
    )
    questions = artifact.payload.get("questions")
    _require(type(questions) is list and len(questions) == 100, "composition population changed")
    rows: dict[int, dict[str, Any]] = {}
    for raw in questions:
        _require(type(raw) is dict and type(raw.get("ordinal")) is int, "composition row changed")
        ordinal = int(raw["ordinal"])
        rows[ordinal] = raw
    _require(set(REMAINING_ORDINALS) <= set(rows), "remaining ordinals left composition")
    return artifact, rows


def _terms(text: str) -> tuple[str, ...]:
    return tuple(
        word for word in _WORD_RE.findall(text.casefold()) if word not in _STOPWORDS
    )


def _turn_score(question: str, role: str, content: str, index: int) -> tuple[int, int, int]:
    query_terms = _terms(question)
    body = content.casefold()
    overlap = sum(1 for term in set(query_terms) if term in body)
    bigrams = sum(
        1
        for left, right in zip(query_terms, query_terms[1:])
        if f"{left} {right}" in body
    )
    numeric = sum(1 for term in set(query_terms) if term.isdigit() and term in body)
    return (overlap * 4 + bigrams * 7 + numeric * 3 + (2 if role == "user" else 0), -index, -len(content))


def _render_turn(source_rank: int, date: str, role: str, content: str) -> str:
    label = f"Memory session {source_rank}"
    if date:
        label += f" | {date}"
    return f"[{label} | {role.upper()}] {content}"


def _prompt(question: str, turns: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], ...]:
    return tuple(
        build_qa_prompt(question, [str(row["rendered"]) for row in turns])
    )


def _fit_turns(
    question: str,
    source_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    ordinal = 0
    for source_rank, source in enumerate(source_rows, start=1):
        date = str(source.get("date") or "")
        raw_turns = source.get("turns")
        _require(type(raw_turns) is list and bool(raw_turns), "oracle source has no turns")
        for turn_index, raw in enumerate(raw_turns):
            _require(type(raw) is dict, "oracle source turn changed type")
            role = str(raw.get("role") or "").casefold()
            content = raw.get("content")
            _require(role in {"assistant", "system", "user"}, "oracle source role changed")
            _require(type(content) is str and bool(content.strip()), "oracle source text changed")
            turns.append(
                {
                    "content_sha256": quote_sha256(content),
                    "global_order": ordinal,
                    "rendered": _render_turn(source_rank, date, role, content),
                    "role": role,
                    "source_rank": source_rank,
                    "turn_index": turn_index,
                }
            )
            ordinal += 1

    full_prompt = _prompt(question, turns)
    full_tokens = count_chat_prompt_token_proxy(full_prompt)
    if full_tokens <= ANSWER_PROMPT_CAP:
        selected = tuple(turns)
    else:
        by_source: dict[int, list[dict[str, Any]]] = {}
        for row in turns:
            by_source.setdefault(int(row["source_rank"]), []).append(row)
        ranked = sorted(
            turns,
            key=lambda row: _turn_score(
                question,
                str(row["role"]),
                str(row["rendered"]),
                int(row["global_order"]),
            ),
            reverse=True,
        )
        priority: list[dict[str, Any]] = []
        seen: set[int] = set()
        # One strongest user-bearing turn per labelled source prevents a long
        # source from consuming the entire oracle context.
        for source_rank in sorted(by_source):
            source_ranked = [row for row in ranked if row["source_rank"] == source_rank]
            user_rows = [row for row in source_ranked if row["role"] == "user"]
            chosen = (user_rows or source_ranked)[0]
            priority.append(chosen)
            seen.add(int(chosen["global_order"]))
            for neighbor_order in (
                int(chosen["global_order"]) - 1,
                int(chosen["global_order"]) + 1,
            ):
                neighbor = next(
                    (
                        row
                        for row in turns
                        if row["global_order"] == neighbor_order
                        and row["source_rank"] == source_rank
                    ),
                    None,
                )
                if neighbor is not None and neighbor_order not in seen:
                    priority.append(neighbor)
                    seen.add(neighbor_order)
        priority.extend(row for row in ranked if int(row["global_order"]) not in seen)

        admitted: list[dict[str, Any]] = []
        for candidate in priority:
            proposal = sorted([*admitted, candidate], key=lambda row: int(row["global_order"]))
            if count_chat_prompt_token_proxy(_prompt(question, proposal)) <= ANSWER_PROMPT_CAP:
                admitted = proposal
        selected = tuple(admitted)

    selected_orders = {int(row["global_order"]) for row in selected}
    final_prompt_tokens = count_chat_prompt_token_proxy(_prompt(question, selected))
    _require(bool(selected), "oracle source fitter selected no turns")
    _require(
        final_prompt_tokens <= ANSWER_PROMPT_CAP
        and final_prompt_tokens + ANSWER_OUTPUT_RESERVE <= 8_000,
        "oracle source fitter violated the complete envelope",
    )
    audit = {
        "dropped_turn_count": len(turns) - len(selected),
        "full_prompt_token_proxy": full_tokens,
        "full_source_retained": len(selected) == len(turns),
        "fitted_prompt_token_proxy": final_prompt_tokens,
        "selected_content_sha256s": [row["content_sha256"] for row in selected],
        "selected_turn_count": len(selected),
        "source_count": len(source_rows),
        "total_turn_count": len(turns),
        "whole_turns_only": True,
    }
    return selected, audit


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset)
    composition_path = Path(args.composition)
    output = Path(args.output_root)
    dataset = _read_dataset(dataset_path)
    composition, composition_rows = _composition_rows(composition_path)
    records = {str(row.get("question_id")): row for row in dataset}
    _require(len(records) == 500, "dataset question IDs changed")

    prompt_rows: list[dict[str, Any]] = []
    authority_rows: list[dict[str, Any]] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    for ordinal in REMAINING_ORDINALS:
        composition_row = composition_rows[ordinal]
        question_id = str(composition_row.get("question_id") or "")
        record = records.get(question_id)
        _require(record is not None, f"dataset lacks question {question_id}")
        provider_input = composition_row.get("provider_projection", {}).get("provider_input", {})
        dated_question = provider_input.get("dated_question")
        _require(type(dated_question) is str and bool(dated_question), "dated question changed")
        _require(record.get("question") in dated_question, "dataset/composition question changed")
        session_ids = record.get("haystack_session_ids")
        session_dates = record.get("haystack_dates")
        sessions = record.get("haystack_sessions")
        answer_ids = record.get("answer_session_ids")
        _require(
            type(session_ids) is list
            and type(session_dates) is list
            and type(sessions) is list
            and len(session_ids) == len(session_dates) == len(sessions)
            and type(answer_ids) is list
            and bool(answer_ids),
            "oracle source coordinates changed",
        )
        answer_set = set(answer_ids)
        selected_sources = [
            {"date": session_dates[index], "source_id": source_id, "turns": sessions[index]}
            for index, source_id in enumerate(session_ids)
            if source_id in answer_set
        ]
        _require(
            len(selected_sources) == len(answer_set),
            "not every labelled answer source exists exactly once",
        )
        fitted, audit = _fit_turns(dated_question, selected_sources)
        messages = _prompt(dated_question, fitted)
        messages_sha256 = identity_sha256(list(messages))
        prompts.append(messages)
        prompt_body = {
            "dated_question_sha256": quote_sha256(dated_question),
            "fitting_audit": audit,
            "messages": list(messages),
            "messages_sha256": messages_sha256,
            "ordinal": ordinal,
            "oracle_source_ids_sha256": identity_sha256(sorted(answer_set)),
            "question_sha256": quote_sha256(str(record["question"])),
        }
        prompt_rows.append(
            {**prompt_body, "prompt_row_sha256": identity_sha256(prompt_body)}
        )
        authority_body = {
            "dated_question": dated_question,
            "ordinal": ordinal,
            "question": str(record["question"]),
            "question_id": question_id,
            "reference": str(record["answer"]),
        }
        authority_rows.append(
            {**authority_body, "authority_row_sha256": identity_sha256(authority_body)}
        )

    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=ANSWER_PROMPT_CAP,
    )
    _require(
        population.logical_prompt_count == population.unique_prompt_count == QUESTION_COUNT,
        "oracle-source answer prompts are not 24 unique calls",
    )
    preflight_payload = {
        "answer_output_token_reserve": ANSWER_OUTPUT_RESERVE,
        "answer_prompt_cap": ANSWER_PROMPT_CAP,
        "composition_artifact_sha256": composition.sha256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "fitted_question_count": sum(
            not row["fitting_audit"]["full_source_retained"] for row in prompt_rows
        ),
        "format": PREFLIGHT_FORMAT,
        "full_source_question_count": sum(
            row["fitting_audit"]["full_source_retained"] for row in prompt_rows
        ),
        "gold_answer_in_provider_messages": False,
        "gold_source_selection": True,
        "hard_complete_envelope_tokens": 8_000,
        "model": TERRA_MODEL,
        "original_ordinals": list(REMAINING_ORDINALS),
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "prompt_rows": prompt_rows,
        "provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "within_source_fitting_uses_answer_or_reference": False,
    }
    preflight, preflight_created = publish_sealed_json(
        output / PREFLIGHT_NAME,
        preflight_payload,
    )
    authority_payload = {
        "answer_preflight_sha256": preflight.sha256,
        "composition_artifact_sha256": composition.sha256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "format": AUTHORITY_FORMAT,
        "gold_loaded": True,
        "original_ordinals": list(REMAINING_ORDINALS),
        "question_count": QUESTION_COUNT,
        "rows": authority_rows,
    }
    authority, authority_created = publish_sealed_json(
        output / AUTHORITY_NAME,
        authority_payload,
    )
    return {
        "answer_preflight": preflight.path.as_posix(),
        "answer_preflight_created": preflight_created,
        "answer_preflight_sha256": preflight.sha256,
        "fitted_question_count": preflight_payload["fitted_question_count"],
        "full_source_question_count": preflight_payload["full_source_question_count"],
        "judge_authority": authority.path.as_posix(),
        "judge_authority_created": authority_created,
        "judge_authority_sha256": authority.sha256,
        "provider_calls": 0,
        "question_count": QUESTION_COUNT,
    }


def _read_preflight(output: Path, expected_sha256: str) -> tuple[SealedArtifact, tuple[tuple[dict[str, str], ...], ...]]:
    artifact = read_sealed_json(output / PREFLIGHT_NAME)
    _require(artifact.sha256 == require_sha256(expected_sha256, "expected answer preflight"), "answer preflight SHA-256 changed")
    payload = artifact.payload
    rows = payload.get("prompt_rows")
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("original_ordinals") == list(REMAINING_ORDINALS)
        and payload.get("gold_answer_in_provider_messages") is False
        and type(rows) is list
        and len(rows) == QUESTION_COUNT,
        "answer preflight population changed",
    )
    prompts: list[tuple[dict[str, str], ...]] = []
    for ordinal, row in zip(REMAINING_ORDINALS, rows, strict=True):
        _require(type(row) is dict and row.get("ordinal") == ordinal, "answer prompt order changed")
        body = dict(row)
        declared = body.pop("prompt_row_sha256", None)
        messages = row.get("messages")
        _require(
            identity_sha256(body) == declared and type(messages) is list,
            "answer prompt row seal changed",
        )
        plain = tuple(dict(message) for message in messages)
        _require(
            identity_sha256(list(plain)) == row.get("messages_sha256")
            and count_chat_prompt_token_proxy(plain) <= ANSWER_PROMPT_CAP,
            "answer prompt bytes/budget changed",
        )
        prompts.append(plain)
    population = preflight_fast_completion_prompts(prompts, max_prompt_tokens=ANSWER_PROMPT_CAP)
    _require(
        population.prompt_population_sha256 == payload.get("prompt_population_sha256")
        and population.model_dump() == payload.get("prompt_population"),
        "answer prompt population changed",
    )
    return artifact, tuple(prompts)


def _client() -> Any:
    load_dotenv()
    api_key = os.environ.get("LITELLM_KEY", "").strip()
    _require(bool(api_key), "LITELLM_KEY is empty")
    return live._make_provider_client(api_key, GATEWAY_URL)  # noqa: SLF001


def _run_runtime(
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    checkpoint_dir: Path,
    model: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    client: Any | None,
    provenance: Mapping[str, Any],
) -> FastCompletionBatch:
    runtime = FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=prompts,
        model=model,
        client=client,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        max_concurrency=DEFAULT_MAX_CONCURRENCY,
        retries=0,
        benchmark_provenance=provenance,
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


def _answer_projection(preflight: SealedArtifact, batch: FastCompletionBatch) -> dict[str, Any]:
    rows = preflight.payload["prompt_rows"]
    records = {record.messages_sha256: record for record in batch.unique_records}
    questions: list[dict[str, Any]] = []
    for row, prediction in zip(rows, batch.logical_completions, strict=True):
        record = records.get(row["messages_sha256"])
        _require(record is not None and record.completion == prediction, "answer completion binding changed")
        questions.append(
            {
                "call_key_sha256": record.call_key_sha256,
                "completion_receipt_sha256": record.completion_sha256,
                "fitting_audit": row["fitting_audit"],
                "messages_sha256": row["messages_sha256"],
                "ordinal": row["ordinal"],
                "prediction": prediction,
                "prediction_sha256": quote_sha256(prediction),
                "request_journal_sha256": record.request_journal_sha256,
                "response_journal_sha256": record.response_journal_sha256,
            }
        )
    return {
        "answer_preflight_sha256": preflight.sha256,
        "format": ANSWER_FORMAT,
        "gold_loaded": False,
        "original_ordinals": list(REMAINING_ORDINALS),
        "provider_calls_during_materialization": 0,
        "question_count": QUESTION_COUNT,
        "questions": questions,
        "retained_transformer_token_state_bytes": 0,
    }


def _answer(args: argparse.Namespace, *, replay: bool) -> dict[str, Any]:
    output = Path(args.output_root)
    preflight, prompts = _read_preflight(output, args.expected_preflight_sha256)
    if not replay:
        _require(args.enable_provider is True and args.authorized_provider_calls == QUESTION_COUNT, "answer requires exact authorization for 24 calls")
    client = None if replay else _client()
    try:
        batch = _run_runtime(
            prompts,
            checkpoint_dir=output / ANSWER_CHECKPOINT_DIR,
            model=TERRA_MODEL,
            max_prompt_tokens=ANSWER_PROMPT_CAP,
            max_new_tokens=ANSWER_OUTPUT_RESERVE,
            client=client,
            provenance={
                "answer_preflight_sha256": preflight.sha256,
                "authorized_unique_calls": QUESTION_COUNT,
                "experiment_format": FORMAT,
                "gold_loaded": False,
                "oracle_source_selection": True,
            },
        )
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    projection = _answer_projection(preflight, batch)
    artifact, created = publish_sealed_json(output / ANSWER_NAME, projection)
    return {
        "answer_run": artifact.path.as_posix(),
        "answer_run_created": created,
        "answer_run_sha256": artifact.sha256,
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "physical_provider_calls": batch.usage.physical_calls,
        "question_count": QUESTION_COUNT,
        "replay": replay,
    }


def _judge_inputs(output: Path, expected_answer_sha256: str) -> tuple[SealedArtifact, SealedArtifact, tuple[tuple[dict[str, str], ...], ...]]:
    answer = read_sealed_json(output / ANSWER_NAME)
    _require(answer.sha256 == require_sha256(expected_answer_sha256, "expected answer run"), "answer run SHA-256 changed")
    authority = read_sealed_json(output / AUTHORITY_NAME)
    _require(
        authority.payload.get("format") == AUTHORITY_FORMAT
        and authority.payload.get("answer_preflight_sha256") == answer.payload.get("answer_preflight_sha256"),
        "judge authority binding changed",
    )
    answer_rows = answer.payload.get("questions")
    authority_rows = authority.payload.get("rows")
    _require(type(answer_rows) is list and type(authority_rows) is list and len(answer_rows) == len(authority_rows) == QUESTION_COUNT, "judge population changed")
    prompts: list[tuple[dict[str, str], ...]] = []
    for ordinal, predicted, gold in zip(REMAINING_ORDINALS, answer_rows, authority_rows, strict=True):
        _require(predicted.get("ordinal") == gold.get("ordinal") == ordinal, "judge row order changed")
        prompts.append(
            tuple(
                build_judge_prompt(
                    str(gold["dated_question"]),
                    str(gold["reference"]),
                    str(predicted["prediction"]),
                )
            )
        )
    preflight_fast_completion_prompts(prompts, max_prompt_tokens=JUDGE_PROMPT_CAP)
    return answer, authority, tuple(prompts)


def _judge_projection(
    answer: SealedArtifact,
    authority: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    records = {record.messages_sha256: record for record in batch.unique_records}
    questions: list[dict[str, Any]] = []
    for answer_row, authority_row, messages, judge_output in zip(
        answer.payload["questions"], authority.payload["rows"], prompts, batch.logical_completions, strict=True
    ):
        messages_sha256 = identity_sha256([dict(message) for message in messages])
        record = records.get(messages_sha256)
        _require(record is not None and record.completion == judge_output, "judge completion binding changed")
        correct = parse_binary_judge_verdict(judge_output)
        questions.append(
            {
                "correct": correct,
                "f1": f1_score(str(authority_row["reference"]), str(answer_row["prediction"])),
                "full_source_retained": answer_row["fitting_audit"]["full_source_retained"],
                "judge_output": judge_output,
                "judge_output_sha256": quote_sha256(judge_output),
                "ordinal": answer_row["ordinal"],
                "prediction": answer_row["prediction"],
                "reference": authority_row["reference"],
                "normalized_exact_match": exact_match(str(authority_row["reference"]), str(answer_row["prediction"])),
                "request_journal_sha256": record.request_journal_sha256,
                "response_journal_sha256": record.response_journal_sha256,
            }
        )
    full = [row for row in questions if row["full_source_retained"]]
    fitted = [row for row in questions if not row["full_source_retained"]]
    return {
        "aggregate": {
            "accuracy": sum(row["correct"] for row in questions) / QUESTION_COUNT,
            "correct": sum(row["correct"] for row in questions),
            "fitted_accuracy": (sum(row["correct"] for row in fitted) / len(fitted) if fitted else None),
            "fitted_correct": sum(row["correct"] for row in fitted),
            "fitted_question_count": len(fitted),
            "full_source_accuracy": (sum(row["correct"] for row in full) / len(full) if full else None),
            "full_source_correct": sum(row["correct"] for row in full),
            "full_source_question_count": len(full),
            "question_count": QUESTION_COUNT,
        },
        "answer_run_sha256": answer.sha256,
        "format": JUDGE_FORMAT,
        "gold_loaded": True,
        "judge_authority_sha256": authority.sha256,
        "original_ordinals": list(REMAINING_ORDINALS),
        "provider_calls_during_materialization": 0,
        "questions": questions,
        "retained_transformer_token_state_bytes": 0,
    }


def _judge(args: argparse.Namespace, *, replay: bool) -> dict[str, Any]:
    output = Path(args.output_root)
    answer, authority, prompts = _judge_inputs(output, args.expected_answer_run_sha256)
    if not replay:
        _require(args.enable_provider is True and args.authorized_provider_calls == QUESTION_COUNT, "judge requires exact authorization for 24 calls")
    client = None if replay else _client()
    try:
        batch = _run_runtime(
            prompts,
            checkpoint_dir=output / JUDGE_CHECKPOINT_DIR,
            model=SOL_MODEL,
            max_prompt_tokens=JUDGE_PROMPT_CAP,
            max_new_tokens=JUDGE_MAX_TOKENS,
            client=client,
            provenance={
                "answer_run_sha256": answer.sha256,
                "authorized_unique_calls": QUESTION_COUNT,
                "experiment_format": FORMAT,
                "gold_loaded": True,
                "judge_authority_sha256": authority.sha256,
            },
        )
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    projection = _judge_projection(answer, authority, prompts, batch)
    artifact, created = publish_sealed_json(output / JUDGE_NAME, projection)
    return {
        "aggregate": projection["aggregate"],
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "judge_run": artifact.path.as_posix(),
        "judge_run_created": created,
        "judge_run_sha256": artifact.sha256,
        "physical_provider_calls": batch.usage.physical_calls,
        "replay": replay,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--dataset", default=str(DEFAULT_DATASET))
    preflight.add_argument("--composition", default=str(DEFAULT_COMPOSITION))
    preflight.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    for name in ("answer", "answer-replay"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
        command.add_argument("--expected-preflight-sha256", required=True)
        command.add_argument("--enable-provider", action="store_true")
        command.add_argument("--authorized-provider-calls", type=int, default=0)
    for name in ("judge", "judge-replay"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
        command.add_argument("--expected-answer-run-sha256", required=True)
        command.add_argument("--enable-provider", action="store_true")
        command.add_argument("--authorized-provider-calls", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "preflight":
        result = _preflight(args)
    elif args.phase == "answer":
        result = _answer(args, replay=False)
    elif args.phase == "answer-replay":
        result = _answer(args, replay=True)
    elif args.phase == "judge":
        result = _judge(args, replay=False)
    else:
        result = _judge(args, replay=True)
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
