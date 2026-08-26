#!/usr/bin/env python3
"""Run the universal S0-plus-EM-facts mechanism arm.

The treatment is intentionally tool-only so the historical fixed-S1 and S0
artifacts can be validated against the source-package identity they sealed.
It has two provider populations: exactly one EM-v2 compression for every
locked question, followed by answers only for valid, nonempty, bounded fact
packets.  Every rejected packet preserves the exact sealed S0 prediction.

No phase imports benchmark gold, categories, labeled sources, or judge data.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # support ``python tools/run_...py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._artifact_json import canonical_json_bytes
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.fast_em_fact_memory import (
    DEFAULT_EM_STAGE_ID,
    EMFactCompression,
    EMFactMemoryError,
    _fact_block,
    build_em_fact_answer_prompt,
    episodic_neighborhood,
    parse_fact_compression,
)
from memory_condense.eval.recall_guarded_cumulative_1m import STAGE_IDS
from tools._locked_em_repair_adapter import (
    LockedEMRepairPopulation,
    build_compression_prompt_population,
    load_locked_em_repair_population,
)
from tools.run_routed_full_source_repair import (
    _distribution,
    _make_provider_client,
    _publish,
    _read,
    _record_by_messages,
    _stable_batch,
)


ARM_LABEL = "S0_PLUS_EM_FACTS"
PARENT_ARM_LABEL = "S0_CONTROL"
PREFLIGHT_FORMAT = "memory-condense-locked-s0-em-facts-preflight-v1"
COMPRESSION_FORMAT = "memory-condense-locked-s0-em-facts-compression-v1"
ANSWER_PREFLIGHT_FORMAT = (
    "memory-condense-locked-s0-em-facts-answer-preflight-v1"
)
RUN_FORMAT = "memory-condense-locked-retrieval-mechanism-arm-run-v1"
TARGET_LEDGER_FORMAT = "memory-condense-structural-target-ledger-v1"

DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
    "/retrieval.json"
)
DEFAULT_BASELINE_ANSWERS = Path(
    "eval_results/longmemeval-1m-fixed-s1-validation-20260826"
    "/final-answers.json"
)
DEFAULT_S0_RUN = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/s0-control-v1/run.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/s0-plus-em-facts-v1"
)

EXPECTED_RETRIEVAL_SHA256 = (
    "e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f"
)
EXPECTED_BASELINE_ANSWERS_SHA256 = (
    "d7fc47b8d1f372f002230c6ffe489dac8cd11bd71b35b8d3008b1255da2a38cd"
)

DEFAULT_GATEWAY_URL = "https://central-dev.zt:4000/v1"
DEFAULT_MODEL = "codex_sdk/gpt-5.6-terra"
EXPECTED_QUESTION_COUNT = 100
MAX_PROMPT_TOKENS = 8_000
MAX_COMPRESSION_OUTPUT_TOKENS = 1_024
MAX_ANSWER_OUTPUT_TOKENS = 256
MAX_FACTS = 24
MAX_FACT_BLOCK_TOKENS = 1_536

_VALID_COMPRESSION = "valid"
_DIGEST_CHARS = frozenset("0123456789abcdef")
_FORBIDDEN_KEYS = frozenset(
    {
        "answer_session_ids",
        "category",
        "evidence_sources",
        "gold",
        "gold_answer",
        "reference",
        "reference_answer",
        "source_completeness",
    }
)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARS for character in value)
    ):
        raise ValueError(f"{label} must be an exact lowercase SHA-256 digest")
    return value


def _contains_forbidden_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _FORBIDDEN_KEYS
            or _contains_forbidden_key(child)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def _validate_settings(args: argparse.Namespace) -> None:
    if str(args.gateway_url) != DEFAULT_GATEWAY_URL:
        raise ValueError("EM facts arm requires the locked central-dev gateway")
    if str(args.model) != DEFAULT_MODEL:
        raise ValueError("EM facts arm requires the locked Terra model route")
    if type(args.expected_question_count) is not int or (
        args.expected_question_count != EXPECTED_QUESTION_COUNT
    ):
        raise ValueError(
            f"EM facts arm requires exactly {EXPECTED_QUESTION_COUNT} questions"
        )
    if type(args.max_concurrency) is not int or args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be positive")


def _s0_questions(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = value.get("questions")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("S0 run has no ordered question population")
    return rows


def _validate_s0_run_projection(
    value: Mapping[str, Any],
    *,
    artifact_sha256: str,
    population: LockedEMRepairPopulation,
) -> None:
    """Cross-bind the verified S0 run to the locked EM adapter population."""

    if value.get("format") != RUN_FORMAT or value.get("arm_label") != PARENT_ARM_LABEL:
        raise ValueError("parent artifact is not the sealed S0 control arm")
    if (
        value.get("retrieval_sha256") != population.retrieval_sha256
        or value.get("baseline_final_answers_sha256")
        != population.baseline_final_answers_sha256
        or value.get("population_identity_sha256")
        != population.population_identity_sha256
        or value.get("historical_validator_binding_sha256")
        != population.binding_sha256
    ):
        raise ValueError("S0 run changed its locked source binding")
    rows = _s0_questions(value)
    if len(rows) != population.question_count:
        raise ValueError("S0 run question count changed")
    for ordinal, (source, locked) in enumerate(zip(rows, population.rows, strict=True)):
        question = locked.question
        stage = question.stages[0]
        prediction = source.get("prediction")
        if not isinstance(prediction, Mapping):
            raise ValueError("S0 run prediction is missing")
        text = prediction.get("text")
        text_sha = prediction.get("sha256")
        provider_messages_sha = source.get("provider_messages_sha256")
        source_binding_sha = source.get("source_binding_sha256")
        if (
            source.get("ordinal") != ordinal
            or source.get("question_id") != question.question_id
            or source.get("question_sha256") != question.question_sha256
            or source.get("dated_question_sha256") != question.dated_question_sha256
            or source.get("retrieval_question_part_sha256")
            != question.retrieval_question_part_sha256
            or source.get("source_stage_id") != STAGE_IDS[0]
            or source.get("stage_receipt_sha256") != stage.stage_receipt_sha256
            or source.get("evidence_projection_sha256")
            != stage.evidence_projection_sha256
            or type(source.get("prompt_token_proxy")) is not int
            or int(source["prompt_token_proxy"]) < 1
            or not isinstance(text, str)
            or not text.strip()
            or text_sha != quote_sha256(text)
        ):
            raise ValueError(f"S0 run question binding changed at ordinal {ordinal}")
        _require_sha256(
            provider_messages_sha,
            f"S0 provider-messages SHA-256 at ordinal {ordinal}",
        )
        _require_sha256(
            source_binding_sha,
            f"S0 source-binding SHA-256 at ordinal {ordinal}",
        )
    _require_sha256(artifact_sha256, "S0 run SHA-256")


def _load_verified_s0_run(
    args: argparse.Namespace,
    population: LockedEMRepairPopulation,
) -> tuple[dict[str, Any], str]:
    """Use the S0 runner's validator, then apply the EM-specific cross-bind."""

    from tools.run_locked_retrieval_mechanism_arm import load_verified_run

    loaded = load_verified_run(
        Path(args.s0_run),
        expected_run_sha256=str(args.expected_s0_run_sha256),
        retrieval_path=Path(args.retrieval),
        expected_retrieval_sha256=str(args.expected_retrieval_sha256),
        baseline_answers_path=Path(args.baseline_answers),
        expected_baseline_answers_sha256=str(
            args.expected_baseline_answers_sha256
        ),
        max_concurrency=int(args.max_concurrency),
        expected_question_count=int(args.expected_question_count),
    )
    if (
        not isinstance(loaded, tuple)
        or len(loaded) != 2
        or not isinstance(loaded[0], Mapping)
        or not isinstance(loaded[1], str)
    ):
        raise TypeError("S0 load_verified_run returned an unsupported value")
    value = dict(loaded[0])
    digest = loaded[1]
    _validate_s0_run_projection(
        value,
        artifact_sha256=digest,
        population=population,
    )
    return value, digest


@dataclass(frozen=True, slots=True)
class _Inputs:
    population: LockedEMRepairPopulation
    s0_run: Mapping[str, Any]
    s0_run_sha256: str
    binding: Mapping[str, Any]
    compression_prompts: tuple[tuple[dict[str, str], ...], ...]
    compression_preflight: FastPromptPopulation

    @property
    def question_count(self) -> int:
        return self.population.question_count


def _build_inputs(args: argparse.Namespace) -> _Inputs:
    """Validate every historical input before any output path is touched."""

    _validate_settings(args)
    population = load_locked_em_repair_population(
        Path(args.retrieval),
        expected_retrieval_sha256=str(args.expected_retrieval_sha256),
        baseline_final_answers_path=Path(args.baseline_answers),
        expected_baseline_final_answers_sha256=str(
            args.expected_baseline_answers_sha256
        ),
    )
    if population.question_count != args.expected_question_count:
        raise ValueError("locked EM question count changed")
    s0_run, s0_sha = _load_verified_s0_run(args, population)
    prompts = build_compression_prompt_population(population)
    preflight = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
    if (
        preflight.logical_prompt_count != population.question_count
        or preflight.unique_prompt_count != population.question_count
    ):
        raise ValueError("compression prompts are not unique one-per-question")
    question_bindings: list[dict[str, Any]] = []
    for locked, s0 in zip(
        population.rows, _s0_questions(s0_run), strict=True
    ):
        selected_s1 = locked.question.stages[1].evidence
        root, delta = episodic_neighborhood(
            locked.question,  # type: ignore[arg-type]
            stage_id=DEFAULT_EM_STAGE_ID,
        )
        question_bindings.append(
            {
                "ordinal": locked.question.ordinal,
                "question_id": locked.question.question_id,
                "question_binding_sha256": locked.binding_sha256,
                "s0_stage_receipt_sha256": (
                    locked.question.stages[0].stage_receipt_sha256
                ),
                "s1_stage_receipt_sha256": (
                    locked.question.stages[1].stage_receipt_sha256
                ),
                "s1_evidence_projection_sha256": (
                    locked.question.stages[1].evidence_projection_sha256
                ),
                "s0_prediction_sha256": s0["prediction"]["sha256"],
                "s0_source_binding_sha256": s0[
                    "source_binding_sha256"
                ],
                "s0_provider_messages_sha256": s0[
                    "provider_messages_sha256"
                ],
                "s0_prompt_token_proxy": s0["prompt_token_proxy"],
                "s0_evidence_ids_sha256": identity_sha256(
                    [row.evidence_id for row in root]
                ),
                "s1_selected_evidence_ids_sha256": identity_sha256(
                    [row.evidence_id for row in selected_s1]
                ),
                "post_selection_em_delta_ids_sha256": identity_sha256(
                    [row.evidence_id for row in delta]
                ),
                "s0_selected_rows": len(root),
                "s1_selected_rows_before_dedup": len(selected_s1),
                "s0_rows_excluded_from_em_after_selection": (
                    len(selected_s1) - len(delta)
                ),
                "rekeyed_s0_duplicates_excluded_after_selection": (
                    len(selected_s1) - len(root) - len(delta)
                ),
                "post_selection_em_delta_rows": len(delta),
            }
        )
    binding: dict[str, Any] = {
        "format": "memory-condense-locked-s0-em-facts-binding-v1",
        "arm_label": ARM_LABEL,
        "parent_arm_label": PARENT_ARM_LABEL,
        "retrieval_sha256": population.retrieval_sha256,
        "baseline_final_answers_sha256": (
            population.baseline_final_answers_sha256
        ),
        "population_identity_sha256": population.population_identity_sha256,
        "historical_validator_binding_sha256": population.binding_sha256,
        "s0_control_run_sha256": s0_sha,
        "question_bindings": question_bindings,
    }
    binding["binding_sha256"] = identity_sha256(binding)
    if _contains_forbidden_key(binding):
        raise RuntimeError("EM binding crossed the gold firewall")
    return _Inputs(
        population=population,
        s0_run=s0_run,
        s0_run_sha256=s0_sha,
        binding=binding,
        compression_prompts=prompts,
        compression_preflight=preflight,
    )


def _runtime(
    inputs: _Inputs,
    args: argparse.Namespace,
    *,
    kind: str,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    client: Any | None,
) -> FastCompletionRuntime:
    if kind not in {"compression", "answer"}:
        raise ValueError("runtime kind must be compression or answer")
    output_tokens = (
        MAX_COMPRESSION_OUTPUT_TOKENS
        if kind == "compression"
        else MAX_ANSWER_OUTPUT_TOKENS
    )
    return FastCompletionRuntime(
        checkpoint_dir=Path(args.output_root) / f"terra-{kind}-calls",
        prompt_population=prompts,
        model=DEFAULT_MODEL,
        client=client,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens=output_tokens,
        max_concurrency=args.max_concurrency,
        retries=0,
        benchmark_provenance={
            "experiment_format": RUN_FORMAT,
            "arm_label": ARM_LABEL,
            "kind": kind,
            "source_binding_sha256": inputs.binding["binding_sha256"],
            "retrieval_sha256": inputs.population.retrieval_sha256,
            "s0_control_run_sha256": inputs.s0_run_sha256,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_loaded": False,
        },
    )


def _compression_path(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / "compression.json"


def _run_path(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / "run.json"


def _target_ledger_path(args: argparse.Namespace) -> Path:
    return Path(
        args.target_ledger
        or Path(args.output_root) / "structural-target-ledger.json"
    )


def _fact_section(compression: EMFactCompression) -> str:
    return "Compact episodic facts:\n" + _fact_block(compression.facts)


@dataclass(frozen=True, slots=True)
class _AcceptedCompression:
    compression: EMFactCompression
    prompt: Any
    fact_block_token_proxy: int


def _accept_compression(
    inputs: _Inputs,
    ordinal: int,
    completion: str,
) -> tuple[str, _AcceptedCompression | None]:
    question = inputs.population.rows[ordinal].question
    try:
        compression = parse_fact_compression(
            question,  # type: ignore[arg-type]
            completion,
            stage_id=DEFAULT_EM_STAGE_ID,
            max_facts=MAX_FACTS,
        )
    except EMFactMemoryError:
        return "invalid_or_ungrounded", None
    if not compression.facts:
        return "empty", None
    fact_tokens = count_tokens(_fact_section(compression))
    if fact_tokens > MAX_FACT_BLOCK_TOKENS:
        return "fact_block_overflow", None
    try:
        prompt = build_em_fact_answer_prompt(
            question,  # type: ignore[arg-type]
            compression,
            arm="facts",
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            responder_output_token_reserve=MAX_ANSWER_OUTPUT_TOKENS,
            policy="v2",
        )
    except EMFactMemoryError:
        return "answer_prompt_overflow", None
    root, delta = episodic_neighborhood(
        question,  # type: ignore[arg-type]
        stage_id=DEFAULT_EM_STAGE_ID,
    )
    if (
        prompt.root_evidence_ids != tuple(row.evidence_id for row in root)
        or prompt.fact_ids != tuple(row.fact_id for row in compression.facts)
        or prompt.selected_neighborhood_evidence_ids
        or prompt.dropped_neighborhood_evidence_ids
        != tuple(row.evidence_id for row in delta)
        or prompt.prompt_token_proxy + MAX_ANSWER_OUTPUT_TOKENS
        > MAX_PROMPT_TOKENS
    ):
        return "answer_projection_overflow", None
    return (
        _VALID_COMPRESSION,
        _AcceptedCompression(
            compression=compression,
            prompt=prompt,
            fact_block_token_proxy=fact_tokens,
        ),
    )


def _compression_rows(
    inputs: _Inputs,
    batch: FastCompletionBatch,
) -> tuple[list[dict[str, Any]], tuple[_AcceptedCompression | None, ...]]:
    records = _record_by_messages(batch)
    rows: list[dict[str, Any]] = []
    accepted: list[_AcceptedCompression | None] = []
    for ordinal, (prompt_row, completion) in enumerate(
        zip(
            inputs.compression_preflight.ordered_rows,
            batch.logical_completions,
            strict=True,
        )
    ):
        status, candidate = _accept_compression(inputs, ordinal, completion)
        accepted.append(candidate)
        record = records[prompt_row.messages_sha256]
        locked = inputs.population.rows[ordinal]
        selected_s1 = locked.question.stages[1].evidence
        root, delta = episodic_neighborhood(
            locked.question,  # type: ignore[arg-type]
            stage_id=DEFAULT_EM_STAGE_ID,
        )
        rows.append(
            {
                "ordinal": ordinal,
                "question_id": locked.question.question_id,
                "question_sha256": locked.question.question_sha256,
                "dated_question_sha256": (
                    locked.question.dated_question_sha256
                ),
                "retrieval_question_part_sha256": (
                    locked.question.retrieval_question_part_sha256
                ),
                "question_binding_sha256": locked.binding_sha256,
                "s1_stage_receipt_sha256": (
                    locked.question.stages[1].stage_receipt_sha256
                ),
                "s1_evidence_projection_sha256": (
                    locked.question.stages[1].evidence_projection_sha256
                ),
                "compression_status": status,
                "s0_selected_rows": len(root),
                "s1_selected_rows_before_dedup": len(selected_s1),
                "s0_rows_excluded_from_em_after_selection": (
                    len(selected_s1) - len(delta)
                ),
                "rekeyed_s0_duplicates_excluded_after_selection": (
                    len(selected_s1) - len(root) - len(delta)
                ),
                "post_selection_em_delta_rows": len(delta),
                "s1_selected_evidence_ids_sha256": identity_sha256(
                    [row.evidence_id for row in selected_s1]
                ),
                "post_selection_em_delta_ids_sha256": identity_sha256(
                    [row.evidence_id for row in delta]
                ),
                "validated_fact_count": (
                    0 if candidate is None else len(candidate.compression.facts)
                ),
                "fact_block_token_proxy": (
                    None if candidate is None else candidate.fact_block_token_proxy
                ),
                "compression_receipt_sha256": (
                    None
                    if candidate is None
                    else candidate.compression.receipt_sha256
                ),
                "completion_sha256": quote_sha256(completion),
                "compression_prompt_messages_sha256": prompt_row.messages_sha256,
                "call_key_sha256": record["call_key_sha256"],
                "request_journal_sha256": record["request_journal_sha256"],
                "response_journal_sha256": record["response_journal_sha256"],
            }
        )
    return rows, tuple(accepted)


def _compression_artifact(
    inputs: _Inputs,
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    rows, _accepted = _compression_rows(inputs, batch)
    statuses = Counter(str(row["compression_status"]) for row in rows)
    artifact = {
        "format": COMPRESSION_FORMAT,
        "arm_label": ARM_LABEL,
        "source_binding": dict(inputs.binding),
        "question_count": inputs.question_count,
        "required_compression_calls": inputs.question_count,
        "settings": {
            "model": DEFAULT_MODEL,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_output_tokens": MAX_COMPRESSION_OUTPUT_TOKENS,
            "max_facts": MAX_FACTS,
            "fact_block_token_cap": MAX_FACT_BLOCK_TOKENS,
            "memory_policy": "v2",
            "raw_em_rows": 0,
        },
        "completion_batch": _stable_batch(batch),
        "status_counts": dict(sorted(statuses.items())),
        "questions": rows,
        "gold_loaded": False,
        "benchmark_categories_loaded": False,
        "benchmark_source_labels_loaded": False,
        "retained_request_token_state_bytes": 0,
    }
    if _contains_forbidden_key(artifact):
        raise RuntimeError("compression artifact crossed the gold firewall")
    return artifact


def _guard_existing(
    path: Path,
    *,
    format_name: str,
    inputs: _Inputs,
) -> None:
    if not path.exists():
        return
    existing, _sha = _read(path)
    if (
        existing.get("format") != format_name
        or existing.get("arm_label") != ARM_LABEL
        or existing.get("source_binding") != dict(inputs.binding)
    ):
        raise FileExistsError(f"output belongs to another experiment: {path}")


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.enable_provider or args.authorized_provider_calls:
        raise ValueError("preflight forbids provider access and authorization")
    inputs = _build_inputs(args)
    root_counts: list[int] = []
    selected_counts: list[int] = []
    excluded_counts: list[int] = []
    delta_counts: list[int] = []
    for row in inputs.population.rows:
        selected_s1 = row.question.stages[1].evidence
        root, delta = episodic_neighborhood(
            row.question,  # type: ignore[arg-type]
            stage_id=DEFAULT_EM_STAGE_ID,
        )
        root_counts.append(len(root))
        selected_counts.append(len(selected_s1))
        excluded_counts.append(len(selected_s1) - len(delta))
        delta_counts.append(len(delta))
    return {
        "format": PREFLIGHT_FORMAT,
        "arm_label": ARM_LABEL,
        "source_binding": dict(inputs.binding),
        "question_count": inputs.question_count,
        "compression_prompt_population": (
            inputs.compression_preflight.model_dump()
        ),
        "compression_prompt_tokens": _distribution(
            [
                row.prompt_token_proxy
                for row in inputs.compression_preflight.ordered_rows
            ]
        ),
        "s0_selected_rows": _distribution(root_counts),
        "s1_selected_rows_before_dedup": _distribution(selected_counts),
        "s0_rows_excluded_from_em_after_selection": _distribution(
            excluded_counts
        ),
        "post_selection_em_delta_rows": _distribution(delta_counts),
        "required_authorized_provider_calls": inputs.question_count,
        "authorized_call_kind": "terra_em_v2_compression",
        "dependent_answer_calls_require_sealed_compression": True,
        "fact_block_token_cap": MAX_FACT_BLOCK_TOKENS,
        "raw_em_rows": 0,
        "provider_calls": 0,
        "writes": 0,
        "gold_loaded": False,
    }


def run_compression(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    inputs = _build_inputs(args)
    if not args.enable_provider:
        raise ValueError("compression-run requires --enable-provider")
    exact_calls = inputs.compression_preflight.unique_prompt_count
    if args.authorized_provider_calls != exact_calls:
        raise ValueError(
            "--authorized-provider-calls must exactly equal the compression "
            f"population ({args.authorized_provider_calls} != {exact_calls})"
        )
    path = _compression_path(args)
    _guard_existing(
        path,
        format_name=COMPRESSION_FORMAT,
        inputs=inputs,
    )
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    if not api_key:
        raise RuntimeError(f"provider API key is empty: {args.api_key_env}")
    client = _make_provider_client(api_key, DEFAULT_GATEWAY_URL)
    try:
        batch = _runtime(
            inputs,
            args,
            kind="compression",
            prompts=inputs.compression_prompts,
            client=client,
        ).run()
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    if (
        batch.prompt_population.unique_prompt_count != exact_calls
        or batch.usage.physical_calls + batch.usage.checkpoint_hits != exact_calls
    ):
        raise RuntimeError("compression journal population changed")
    artifact = _compression_artifact(inputs, batch)
    return artifact, _publish(path, artifact)


def _verified_compression(
    args: argparse.Namespace,
    inputs: _Inputs,
) -> tuple[dict[str, Any], str, FastCompletionBatch]:
    source, digest = _read(_compression_path(args))
    batch = _runtime(
        inputs,
        args,
        kind="compression",
        prompts=inputs.compression_prompts,
        client=None,
    ).run()
    expected = _compression_artifact(inputs, batch)
    if canonical_json_bytes(source) != canonical_json_bytes(expected):
        raise ValueError("compression artifact differs from immutable journals")
    return source, digest, batch


def run_compression_replay(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls:
        raise ValueError("compression replay forbids provider access")
    inputs = _build_inputs(args)
    source, digest, batch = _verified_compression(args, inputs)
    if batch.usage.physical_calls:
        raise RuntimeError("compression replay unexpectedly made provider calls")
    return source, digest


@dataclass(frozen=True, slots=True)
class _AnswerPlan:
    inputs: _Inputs
    compression_artifact: Mapping[str, Any]
    compression_sha256: str
    compression_batch: FastCompletionBatch
    accepted: tuple[_AcceptedCompression | None, ...]
    valid_ordinals: tuple[int, ...]
    prompts: tuple[Any, ...]
    preflight: FastPromptPopulation | None

    @property
    def unique_calls(self) -> int:
        return 0 if self.preflight is None else self.preflight.unique_prompt_count


def _build_answer_plan(args: argparse.Namespace) -> _AnswerPlan:
    inputs = _build_inputs(args)
    compression, compression_sha, batch = _verified_compression(args, inputs)
    rows, accepted = _compression_rows(inputs, batch)
    if rows != compression.get("questions"):
        raise ValueError("compression question projection changed")
    valid_ordinals = tuple(
        ordinal for ordinal, row in enumerate(accepted) if row is not None
    )
    prompts = tuple(row.prompt for row in accepted if row is not None)
    preflight = None
    if prompts:
        preflight = preflight_fast_completion_prompts(
            [prompt.as_mappings() for prompt in prompts],
            max_prompt_tokens=MAX_PROMPT_TOKENS,
        )
        if (
            preflight.logical_prompt_count != len(valid_ordinals)
            or preflight.unique_prompt_count != len(valid_ordinals)
        ):
            raise RuntimeError("answer prompt population changed")
    return _AnswerPlan(
        inputs=inputs,
        compression_artifact=compression,
        compression_sha256=compression_sha,
        compression_batch=batch,
        accepted=accepted,
        valid_ordinals=valid_ordinals,
        prompts=prompts,
        preflight=preflight,
    )


def run_answer_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.enable_provider or args.authorized_provider_calls:
        raise ValueError("answer preflight forbids provider access")
    plan = _build_answer_plan(args)
    statuses = Counter(
        str(row["compression_status"])
        for row in plan.compression_artifact["questions"]
    )
    answer_tokens = (
        []
        if plan.preflight is None
        else [row.prompt_token_proxy for row in plan.preflight.ordered_rows]
    )
    return {
        "format": ANSWER_PREFLIGHT_FORMAT,
        "arm_label": ARM_LABEL,
        "source_binding": dict(plan.inputs.binding),
        "compression_artifact_sha256": plan.compression_sha256,
        "question_count": plan.inputs.question_count,
        "valid_compression_count": len(plan.valid_ordinals),
        "s0_fallback_count": plan.inputs.question_count - len(plan.valid_ordinals),
        "compression_status_counts": dict(sorted(statuses.items())),
        "answer_prompt_population": (
            None if plan.preflight is None else plan.preflight.model_dump()
        ),
        "answer_prompt_tokens": (
            {"minimum": 0, "mean": 0.0, "maximum": 0, "total": 0}
            if not answer_tokens
            else _distribution(answer_tokens)
        ),
        "required_authorized_provider_calls": plan.unique_calls,
        "authorized_call_kind": "terra_s0_plus_em_facts_answer",
        "fact_block_token_cap": MAX_FACT_BLOCK_TOKENS,
        "raw_em_rows": 0,
        "provider_calls": 0,
        "writes": 0,
        "gold_loaded": False,
    }


def _answer_batch(
    plan: _AnswerPlan,
    args: argparse.Namespace,
    *,
    client: Any | None,
) -> FastCompletionBatch | None:
    if not plan.prompts:
        return None
    return _runtime(
        plan.inputs,
        args,
        kind="answer",
        prompts=[prompt.as_mappings() for prompt in plan.prompts],
        client=client,
    ).run()


def _evidence_target(
    evidence: Any,
    *,
    discovering_method: str,
    selection_role: str,
    disposition: str,
    route_local_receipt_sha256: str,
    duplicate_of_primary_target_id: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "target_id": evidence.evidence_id,
        "target_id_encoding": "raw_sealed_evidence_id",
        "target_kind": "evidence",
        "discovering_method": discovering_method,
        "source_target_id": evidence.source_id,
        "selection_role": selection_role,
        "disposition": disposition,
        "route_local_receipt_sha256": route_local_receipt_sha256,
    }
    if duplicate_of_primary_target_id is not None:
        row["duplicate_of_primary_target_id"] = duplicate_of_primary_target_id
    return row


def _fact_targets(
    compression: EMFactCompression | None,
    *,
    disposition: str,
) -> tuple[dict[str, Any], ...]:
    if compression is None:
        return ()
    rows: list[dict[str, Any]] = []
    for fact in compression.facts:
        evidence_ids = tuple(
            dict.fromkeys(citation.evidence_id for citation in fact.citations)
        )
        source_ids = tuple(
            dict.fromkeys(citation.source_id for citation in fact.citations)
        )
        fact_target_id = identity_sha256(
            {
                "format": "memory-condense-episodic-fact-target-v1",
                "compression_receipt_sha256": compression.receipt_sha256,
                "fact": fact.identity_payload(),
            }
        )
        rows.append(
            {
                "target_id": fact_target_id,
                "target_id_encoding": "sha256",
                "target_kind": "episodic_fact",
                "discovering_method": "post_selection_em_fact_conversion_v2",
                "disposition": disposition,
                "route_local_receipt_sha256": compression.receipt_sha256,
                "fact_id": fact.fact_id,
                "fact_text_sha256": quote_sha256(fact.text),
                "cited_source_target_ids": list(evidence_ids),
                "cited_source_target_ids_sha256": identity_sha256(
                    list(evidence_ids)
                ),
                "cited_source_ids": list(source_ids),
            }
        )
    return tuple(rows)


def _structural_target_ledger(
    plan: _AnswerPlan,
    *,
    source_run_sha256: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ordinal, (locked, completion, accepted) in enumerate(
        zip(
            plan.inputs.population.rows,
            plan.compression_batch.logical_completions,
            plan.accepted,
            strict=True,
        )
    ):
        question = locked.question
        root, delta = episodic_neighborhood(
            question,  # type: ignore[arg-type]
            stage_id=DEFAULT_EM_STAGE_ID,
        )
        root_ids = {row.evidence_id for row in root}
        root_by_coordinate = {(row.source_id, row.text): row for row in root}
        selected = question.stages[1].evidence
        selected_targets: list[dict[str, Any]] = []
        for evidence in selected:
            duplicate = root_by_coordinate.get((evidence.source_id, evidence.text))
            carried_root = evidence.evidence_id in root_ids
            duplicate_root_id = (
                None
                if duplicate is None or carried_root
                else duplicate.evidence_id
            )
            selected_targets.append(
                _evidence_target(
                    evidence,
                    discovering_method=DEFAULT_EM_STAGE_ID,
                    selection_role=(
                        "protected_s0_carry"
                        if carried_root
                        else (
                            "post_selection_s0_duplicate"
                            if duplicate is not None
                            else "post_dedup_em_source"
                        )
                    ),
                    disposition="selected_before_post_selection_dedup",
                    route_local_receipt_sha256=(
                        question.stages[1].stage_receipt_sha256
                    ),
                    duplicate_of_primary_target_id=duplicate_root_id,
                )
            )
        root_targets = [
            _evidence_target(
                evidence,
                discovering_method=STAGE_IDS[0],
                selection_role="protected_s0",
                disposition="protected_s0_unchanged",
                route_local_receipt_sha256=(
                    question.stages[0].stage_receipt_sha256
                ),
            )
            for evidence in root
        ]
        delta_targets = [
            _evidence_target(
                evidence,
                discovering_method=DEFAULT_EM_STAGE_ID,
                selection_role="post_dedup_em_source",
                disposition="admitted_after_post_selection_dedup",
                route_local_receipt_sha256=(
                    question.stages[1].stage_receipt_sha256
                ),
            )
            for evidence in delta
        ]
        try:
            parsed = parse_fact_compression(
                question,  # type: ignore[arg-type]
                completion,
                stage_id=DEFAULT_EM_STAGE_ID,
                max_facts=MAX_FACTS,
            )
        except EMFactMemoryError:
            parsed = None
        candidate_facts = _fact_targets(
            parsed,
            disposition="candidate_before_budget",
        )
        admitted_facts = (
            ()
            if accepted is None
            else _fact_targets(
                accepted.compression,
                disposition="admitted_after_budget",
            )
        )
        body: dict[str, Any] = {
            "ordinal": ordinal,
            "question_id": question.question_id,
            "evidence_targets": root_targets,
            "selected_em_source_targets_before_dedup": selected_targets,
            "post_dedup_em_source_targets": delta_targets,
            "candidate_fact_targets_before_budget": list(candidate_facts),
            "admitted_fact_targets_after_budget": list(admitted_facts),
            "selected_em_source_target_ids_sha256": identity_sha256(
                [row["target_id"] for row in selected_targets]
            ),
            "post_dedup_em_source_target_ids_sha256": identity_sha256(
                [row["target_id"] for row in delta_targets]
            ),
            "candidate_fact_target_ids_sha256": identity_sha256(
                [row["target_id"] for row in candidate_facts]
            ),
            "admitted_fact_target_ids_sha256": identity_sha256(
                [row["target_id"] for row in admitted_facts]
            ),
            "selected_em_source_target_count": len(selected_targets),
            "post_dedup_em_source_target_count": len(delta_targets),
            "candidate_fact_target_count": len(candidate_facts),
            "admitted_fact_target_count": len(admitted_facts),
        }
        body["ledger_row_sha256"] = identity_sha256(body)
        rows.append(body)
    result = {
        "format": TARGET_LEDGER_FORMAT,
        "arm_label": ARM_LABEL,
        "source_run_sha256": source_run_sha256,
        "source_compression_sha256": plan.compression_sha256,
        "source_binding_sha256": plan.inputs.binding["binding_sha256"],
        "population_identity_sha256": (
            plan.inputs.population.population_identity_sha256
        ),
        "question_count": len(rows),
        "target_id_policy": {
            "evidence_targets": "raw_sealed_evidence_id",
            "fact_targets": "sealed_compression_fact_sha256",
            "fact_cited_source_targets": "raw_sealed_evidence_id",
        },
        "ownership_policy": (
            "join-primary-owner-from-posthoc-desired-target-registry"
        ),
        "discovery_projection": (
            "selected_em_source_targets_before_dedup+"
            "candidate_fact_targets_before_budget"
        ),
        "admission_projection": (
            "post_dedup_em_source_targets+admitted_fact_targets_after_budget"
        ),
        "questions": rows,
    }
    result["ledger_sha256"] = identity_sha256(result)
    return result


def _run_artifact(
    plan: _AnswerPlan,
    batch: FastCompletionBatch | None,
) -> dict[str, Any]:
    completions: dict[int, str] = {}
    records: dict[str, Mapping[str, Any]] = {}
    if batch is not None:
        completions = dict(
            zip(plan.valid_ordinals, batch.logical_completions, strict=True)
        )
        records = _record_by_messages(batch)
    s0_rows = _s0_questions(plan.inputs.s0_run)
    compression_rows = plan.compression_artifact["questions"]
    questions: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    for ordinal, (locked, s0, compression_row, accepted) in enumerate(
        zip(
            plan.inputs.population.rows,
            s0_rows,
            compression_rows,
            plan.accepted,
            strict=True,
        )
    ):
        s0_prediction = s0["prediction"]
        if accepted is None:
            prediction = str(s0_prediction["text"])
            prediction_kind = "sealed_s0_control_fallback"
            fallback = str(compression_row["compression_status"])
            prompt_sha = None
            prompt_tokens = None
            answer_call_key = None
            answer_request_journal = None
            answer_journal = None
        else:
            prediction = completions[ordinal]
            prediction_kind = "terra_s0_plus_em_facts"
            fallback = None
            prompt_sha = accepted.prompt.messages_sha256
            prompt_tokens = accepted.prompt.prompt_token_proxy
            answer_record = records[prompt_sha]
            answer_call_key = answer_record["call_key_sha256"]
            answer_request_journal = answer_record["request_journal_sha256"]
            answer_journal = answer_record["response_journal_sha256"]
        question = locked.question
        selected_s1 = question.stages[1].evidence
        root, delta = episodic_neighborhood(
            question,  # type: ignore[arg-type]
            stage_id=DEFAULT_EM_STAGE_ID,
        )
        questions.append(
            {
                "ordinal": ordinal,
                "question_id": question.question_id,
                "question_sha256": question.question_sha256,
                "dated_question_sha256": question.dated_question_sha256,
                "retrieval_question_part_sha256": (
                    question.retrieval_question_part_sha256
                ),
                "arm_label": ARM_LABEL,
                "parent_arm_label": PARENT_ARM_LABEL,
                "s0_stage_receipt_sha256": (
                    question.stages[0].stage_receipt_sha256
                ),
                "s0_evidence_projection_sha256": (
                    question.stages[0].evidence_projection_sha256
                ),
                "s1_stage_receipt_sha256": (
                    question.stages[1].stage_receipt_sha256
                ),
                "s1_evidence_projection_sha256": (
                    question.stages[1].evidence_projection_sha256
                ),
                "question_binding_sha256": locked.binding_sha256,
                "s0_source_binding_sha256": s0[
                    "source_binding_sha256"
                ],
                "s0_provider_messages_sha256": s0[
                    "provider_messages_sha256"
                ],
                "s0_prompt_token_proxy": s0["prompt_token_proxy"],
                "s0_evidence_ids_sha256": identity_sha256(
                    [row.evidence_id for row in root]
                ),
                "s1_selected_evidence_ids_sha256": identity_sha256(
                    [row.evidence_id for row in selected_s1]
                ),
                "post_selection_em_delta_ids_sha256": identity_sha256(
                    [row.evidence_id for row in delta]
                ),
                "s0_control_prediction_sha256": s0_prediction["sha256"],
                "compression_status": compression_row["compression_status"],
                "compression_receipt_sha256": compression_row[
                    "compression_receipt_sha256"
                ],
                "validated_fact_count": compression_row["validated_fact_count"],
                "fact_block_token_proxy": compression_row[
                    "fact_block_token_proxy"
                ],
                "raw_em_rows": 0,
                "prediction_kind": prediction_kind,
                "s0_fallback_reason": fallback,
                "prediction": {
                    "text": prediction,
                    "sha256": quote_sha256(prediction),
                },
                "changed_from_s0": (
                    quote_sha256(prediction) != s0_prediction["sha256"]
                ),
                "compression_response_journal_sha256": compression_row[
                    "response_journal_sha256"
                ],
                "answer_prompt_messages_sha256": prompt_sha,
                "answer_call_key_sha256": answer_call_key,
                "answer_request_journal_sha256": answer_request_journal,
                "answer_response_journal_sha256": answer_journal,
            }
        )
        budget_rows.append(
            {
                "ordinal": ordinal,
                "s0_selected_rows": len(root),
                "s1_selected_rows_before_dedup": len(selected_s1),
                "s0_rows_excluded_from_em_after_selection": (
                    len(selected_s1) - len(delta)
                ),
                "rekeyed_s0_duplicates_excluded_after_selection": (
                    len(selected_s1) - len(root) - len(delta)
                ),
                "post_selection_em_delta_rows": len(delta),
                "validated_fact_count": compression_row["validated_fact_count"],
                "fact_block_token_proxy": compression_row[
                    "fact_block_token_proxy"
                ],
                "fact_block_token_cap": MAX_FACT_BLOCK_TOKENS,
                "raw_em_rows": 0,
                "answer_prompt_token_proxy": prompt_tokens,
                "answer_prompt_token_cap": MAX_PROMPT_TOKENS,
                "answer_output_token_cap": MAX_ANSWER_OUTPUT_TOKENS,
                "s0_fallback": accepted is None,
                "s0_fallback_reason": fallback,
            }
        )
    answer_tokens = [
        int(row["answer_prompt_token_proxy"])
        for row in budget_rows
        if row["answer_prompt_token_proxy"] is not None
    ]
    artifact = {
        "format": RUN_FORMAT,
        "arm_label": ARM_LABEL,
        "parent_arm_label": PARENT_ARM_LABEL,
        "source_binding": dict(plan.inputs.binding),
        "retrieval_sha256": plan.inputs.population.retrieval_sha256,
        "baseline_final_answers_sha256": (
            plan.inputs.population.baseline_final_answers_sha256
        ),
        "population_identity_sha256": (
            plan.inputs.population.population_identity_sha256
        ),
        "historical_validator_binding_sha256": (
            plan.inputs.population.binding_sha256
        ),
        "s0_control_run_sha256": plan.inputs.s0_run_sha256,
        "compression_artifact_sha256": plan.compression_sha256,
        "question_count": plan.inputs.question_count,
        "required_answer_calls": plan.unique_calls,
        "total_sealed_terra_calls": plan.inputs.question_count + plan.unique_calls,
        "settings": {
            "model": DEFAULT_MODEL,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "compression_output_tokens": MAX_COMPRESSION_OUTPUT_TOKENS,
            "answer_output_tokens": MAX_ANSWER_OUTPUT_TOKENS,
            "max_facts": MAX_FACTS,
            "fact_block_token_cap": MAX_FACT_BLOCK_TOKENS,
            "memory_policy": "v2",
            "raw_em_rows": 0,
            "retries": 0,
        },
        "compression_batch_sha256": _digest(
            plan.compression_artifact["completion_batch"]
        ),
        "answer_completion_batch": (
            None if batch is None else _stable_batch(batch)
        ),
        "budget": {
            "s0_non_borrowable": True,
            "deduplicate_s0_after_s1_selection": True,
            "compression_input_is_exact_s1_minus_s0": True,
            "fact_block_token_cap": MAX_FACT_BLOCK_TOKENS,
            "raw_em_rows": 0,
            "shared_residual_tokens": 0,
            "answer_prompt_tokens": (
                {"minimum": 0, "mean": 0.0, "maximum": 0, "total": 0}
                if not answer_tokens
                else _distribution(answer_tokens)
            ),
            "questions": budget_rows,
        },
        "questions": questions,
        "gold_loaded": False,
        "benchmark_categories_loaded": False,
        "benchmark_source_labels_loaded": False,
        "retained_request_token_state_bytes": 0,
    }
    if _contains_forbidden_key(artifact):
        raise RuntimeError("EM facts run crossed the gold firewall")
    return artifact


def run_treatment(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    plan = _build_answer_plan(args)
    if not args.enable_provider:
        raise ValueError("run requires --enable-provider")
    if args.authorized_provider_calls != plan.unique_calls:
        raise ValueError(
            "--authorized-provider-calls must exactly equal the dependent answer "
            f"population ({args.authorized_provider_calls} != {plan.unique_calls})"
        )
    path = _run_path(args)
    _guard_existing(path, format_name=RUN_FORMAT, inputs=plan.inputs)
    client = None
    if plan.unique_calls:
        api_key = os.environ.get(str(args.api_key_env), "").strip()
        if not api_key:
            raise RuntimeError(f"provider API key is empty: {args.api_key_env}")
        client = _make_provider_client(api_key, DEFAULT_GATEWAY_URL)
    try:
        batch = _answer_batch(plan, args, client=client)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    if batch is not None and (
        batch.prompt_population.unique_prompt_count != plan.unique_calls
        or batch.usage.physical_calls + batch.usage.checkpoint_hits
        != plan.unique_calls
    ):
        raise RuntimeError("answer journal population changed")
    artifact = _run_artifact(plan, batch)
    return artifact, _publish(path, artifact)


def run_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls:
        raise ValueError("run replay forbids provider access")
    plan = _build_answer_plan(args)
    source, digest = _read(_run_path(args))
    batch = _answer_batch(plan, args, client=None)
    if batch is not None and batch.usage.physical_calls:
        raise RuntimeError("answer replay unexpectedly made provider calls")
    expected = _run_artifact(plan, batch)
    if canonical_json_bytes(source) != canonical_json_bytes(expected):
        raise ValueError("run artifact differs from immutable journals")
    return source, digest


def _verified_run_for_target_ledger(
    args: argparse.Namespace,
) -> tuple[_AnswerPlan, str]:
    plan = _build_answer_plan(args)
    source, digest = _read(
        _run_path(args),
        expected_sha256=_require_sha256(
            args.expected_run_sha256,
            "sealed EM run expected SHA-256",
        ),
    )
    batch = _answer_batch(plan, args, client=None)
    if batch is not None and batch.usage.physical_calls:
        raise RuntimeError("target-ledger build unexpectedly made provider calls")
    expected = _run_artifact(plan, batch)
    if canonical_json_bytes(source) != canonical_json_bytes(expected):
        raise ValueError("sealed EM run differs from immutable journals")
    return plan, digest


def run_target_ledger(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls:
        raise ValueError("target-ledger build forbids provider access")
    plan, run_sha = _verified_run_for_target_ledger(args)
    ledger = _structural_target_ledger(
        plan,
        source_run_sha256=run_sha,
    )
    path = _target_ledger_path(args)
    if path.exists():
        existing, _existing_sha = _read(path)
        if (
            existing.get("format") != TARGET_LEDGER_FORMAT
            or existing.get("arm_label") != ARM_LABEL
            or existing.get("source_run_sha256") != run_sha
            or existing.get("source_binding_sha256")
            != plan.inputs.binding["binding_sha256"]
        ):
            raise FileExistsError(
                f"target ledger belongs to another experiment: {path}"
            )
    digest = _publish(path, ledger)
    sealed, sealed_sha = _read(path, expected_sha256=digest)
    return sealed, sealed_sha


def run_target_ledger_replay(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls:
        raise ValueError("target-ledger replay forbids provider access")
    plan, run_sha = _verified_run_for_target_ledger(args)
    source, digest = _read(
        _target_ledger_path(args),
        expected_sha256=_require_sha256(
            args.expected_target_ledger_sha256,
            "target ledger expected SHA-256",
        ),
    )
    expected = _structural_target_ledger(
        plan,
        source_run_sha256=run_sha,
    )
    if canonical_json_bytes(source) != canonical_json_bytes(expected):
        raise ValueError("target ledger differs from sealed source journals")
    return source, digest


def load_verified_target_ledger(
    path: str | Path,
    expected_ledger_sha256: str,
    *,
    run_path: str | Path,
    expected_run_sha256: str,
    s0_run_path: str | Path,
    expected_s0_run_sha256: str,
    retrieval_path: str | Path = DEFAULT_RETRIEVAL,
    expected_retrieval_sha256: str = EXPECTED_RETRIEVAL_SHA256,
    baseline_answers_path: str | Path = DEFAULT_BASELINE_ANSWERS,
    expected_baseline_answers_sha256: str = (
        EXPECTED_BASELINE_ANSWERS_SHA256
    ),
    max_concurrency: int = 4,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> tuple[dict[str, Any], str]:
    """Purely verify a standalone EM discovery ledger and all source journals."""

    ledger_path = Path(path)
    sealed_run_path = Path(run_path)
    output_root = sealed_run_path.parent
    if sealed_run_path != output_root / "run.json":
        raise ValueError("verified EM run path must be <output-root>/run.json")
    args = argparse.Namespace(
        phase="target-ledger-replay",
        retrieval=Path(retrieval_path),
        expected_retrieval_sha256=expected_retrieval_sha256,
        baseline_answers=Path(baseline_answers_path),
        expected_baseline_answers_sha256=(
            expected_baseline_answers_sha256
        ),
        s0_run=Path(s0_run_path),
        expected_s0_run_sha256=expected_s0_run_sha256,
        output_root=output_root,
        expected_run_sha256=expected_run_sha256,
        target_ledger=ledger_path,
        expected_target_ledger_sha256=expected_ledger_sha256,
        expected_question_count=expected_question_count,
        gateway_url=DEFAULT_GATEWAY_URL,
        model=DEFAULT_MODEL,
        api_key_env="LITELLM_KEY",
        max_concurrency=max_concurrency,
        enable_provider=False,
        authorized_provider_calls=0,
    )
    return run_target_ledger_replay(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "preflight",
            "compression-run",
            "compression-replay",
            "answer-preflight",
            "run",
            "replay",
            "target-ledger",
            "target-ledger-replay",
        ),
        default="preflight",
    )
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--expected-retrieval-sha256",
        default=EXPECTED_RETRIEVAL_SHA256,
    )
    parser.add_argument(
        "--baseline-answers", type=Path, default=DEFAULT_BASELINE_ANSWERS
    )
    parser.add_argument(
        "--expected-baseline-answers-sha256",
        default=EXPECTED_BASELINE_ANSWERS_SHA256,
    )
    parser.add_argument("--s0-run", type=Path, default=DEFAULT_S0_RUN)
    parser.add_argument("--expected-s0-run-sha256", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-run-sha256")
    parser.add_argument("--target-ledger", type=Path)
    parser.add_argument("--expected-target-ledger-sha256")
    parser.add_argument(
        "--expected-question-count", type=int, default=EXPECTED_QUESTION_COUNT
    )
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="LITELLM_KEY")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.phase == "preflight":
        result: Any = run_preflight(args)
        print(
            "S0_PLUS_EM_FACTS preflight passed: "
            f"questions={result['question_count']}; "
            f"authorized_terra_calls={result['required_authorized_provider_calls']}; "
            "provider_calls=0; writes=0",
            flush=True,
        )
        return 0
    if args.phase == "compression-run":
        result, digest = run_compression(args)
    elif args.phase == "compression-replay":
        result, digest = run_compression_replay(args)
    elif args.phase == "answer-preflight":
        result = run_answer_preflight(args)
        print(
            "S0_PLUS_EM_FACTS answer preflight passed: "
            f"valid={result['valid_compression_count']}; "
            f"fallback={result['s0_fallback_count']}; "
            f"authorized_terra_calls={result['required_authorized_provider_calls']}; "
            "provider_calls=0; writes=0",
            flush=True,
        )
        return 0
    elif args.phase == "run":
        result, digest = run_treatment(args)
    elif args.phase == "target-ledger":
        result, digest = run_target_ledger(args)
    elif args.phase == "target-ledger-replay":
        result, digest = run_target_ledger_replay(args)
    else:
        result, digest = run_replay(args)
    print(
        f"{args.phase} verified {ARM_LABEL} artifact {digest}; "
        f"questions={result['question_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANSWER_PREFLIGHT_FORMAT",
    "ARM_LABEL",
    "COMPRESSION_FORMAT",
    "MAX_FACT_BLOCK_TOKENS",
    "PREFLIGHT_FORMAT",
    "RUN_FORMAT",
    "TARGET_LEDGER_FORMAT",
    "build_parser",
    "load_verified_target_ledger",
    "main",
    "run_answer_preflight",
    "run_compression",
    "run_compression_replay",
    "run_preflight",
    "run_replay",
    "run_target_ledger",
    "run_target_ledger_replay",
    "run_treatment",
]
