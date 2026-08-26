#!/usr/bin/env python3
"""Run a routed, gold-blind repair over the sealed fixed-S1 population.

The provider boundary is deliberately narrow:

* ``plan`` and ``preflight`` route from dated question text and inspect the
  exact prompt/evidence budgets without loading benchmark gold;
* ``run``/``replay`` transform and answer only the selected route, while every
  noneligible prediction is copied from the sealed fixed-S1 campaign;
* ``score`` loads gold only after both Terra journal populations replay; and
* ``judge-*`` sends only changed eligible predictions to Sol.  Every unchanged
  verdict is reused from the independently sealed baseline judge artifact.

This module lives under ``tools`` so it cannot alter the implementation digest
bound into the historical fixed-S1 campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # support ``python tools/run_...py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._artifact_json import canonical_json_bytes
from memory_condense.eval._binary_judge_protocol import (
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import build_judge_prompt, exact_match, f1_score
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.fast_em_fact_memory import (
    DEFAULT_EM_STAGE_ID,
    EMFactMemoryError,
    episodic_neighborhood,
    parse_fact_compression,
)
from memory_condense.eval import run_fast_1m_em_facts as em_runner
from memory_condense.eval.recall_guarded_cumulative_population import (
    LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    build_locked_cumulative_population_identity,
)
from memory_condense.ingest.loader import BenchmarkSample
from tools._locked_em_repair_adapter import (
    LockedEMRepairPopulation,
    load_locked_em_repair_population,
)
from tools._routed_repair_prompts import (
    MAX_ROUTED_PROMPT_TOKENS,
    RoutedAnswerPrompt,
    RoutedCompressionPrompt,
    build_routed_answer_prompt,
    build_routed_fact_compression_prompt,
    numeric_facts_are_quote_grounded,
    normalize_repair_style,
)
from tools._routed_repair_routing import (
    RoutedRepairReceipt,
    RoutedRepairStyle,
    route_question,
)


ROUTE_PLAN_FORMAT = "memory-condense-routed-full-source-route-plan-v1"
PREFLIGHT_FORMAT = "memory-condense-routed-full-source-preflight-v1"
COMPRESSION_FORMAT = "memory-condense-routed-full-source-compression-v1"
ANSWER_PREFLIGHT_FORMAT = "memory-condense-routed-full-source-answer-preflight-v1"
RUN_FORMAT = "memory-condense-routed-full-source-run-v1"
SCORE_FORMAT = "memory-condense-routed-full-source-local-score-v1"
JUDGE_PREFLIGHT_FORMAT = (
    "memory-condense-routed-full-source-sol-judge-preflight-v1"
)
JUDGE_FORMAT = "memory-condense-routed-full-source-sol-judge-v1"

DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
    "/retrieval.json"
)
DEFAULT_BASELINE_ROOT = Path(
    "eval_results/longmemeval-1m-fixed-s1-validation-20260826"
)
DEFAULT_BASELINE_ANSWERS = DEFAULT_BASELINE_ROOT / "final-answers.json"
DEFAULT_BASELINE_JUDGE = (
    DEFAULT_BASELINE_ROOT / "final-answer-semantic-judge-sol.json"
)
DEFAULT_SPLIT = Path(
    "docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-routed-full-source-repair-20260826/numeric-v1"
)

EXPECTED_RETRIEVAL_SHA256 = (
    "e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f"
)
EXPECTED_BASELINE_ANSWERS_SHA256 = (
    "d7fc47b8d1f372f002230c6ffe489dac8cd11bd71b35b8d3008b1255da2a38cd"
)
EXPECTED_BASELINE_JUDGE_SHA256 = (
    "5dc56a240315c5577d1032d40429df7e39adad0f40a098abc371ee2ea2ec77df"
)

DEFAULT_GATEWAY_URL = "https://central-dev.zt:4000/v1"
DEFAULT_TERRA_MODEL = "codex_sdk/gpt-5.6-terra"
DEFAULT_SOL_MODEL = "codex_sdk/gpt-5.6-sol"
DEFAULT_EXPECTED_QUESTION_COUNT = 100
MAX_COMPRESSION_OUTPUT_TOKENS = 1_024
MAX_ANSWER_OUTPUT_TOKENS = 256
MAX_JUDGE_PROMPT_TOKENS = 8_000
MAX_FACTS = 24

_FORBIDDEN_GOLD_KEYS = frozenset(
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
_RECORD_DISPOSITION_FIELDS = frozenset({"checkpoint_hit", "physical_call"})
_USAGE_DISPOSITION_FIELDS = frozenset({"checkpoint_hits", "physical_calls"})
def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be an exact lowercase SHA-256 digest")
    return value


_publish = em_runner._publish


def _read(path: Path, *, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    value, digest = em_runner._read(path)
    if expected_sha256 is not None and digest != _require_sha256(
        expected_sha256, f"{path} expected SHA-256"
    ):
        raise ValueError(f"artifact SHA-256 changed: {path}")
    return value, digest


def _contains_forbidden_gold_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _FORBIDDEN_GOLD_KEYS
            or _contains_forbidden_gold_key(child)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return any(_contains_forbidden_gold_key(child) for child in value)
    return False


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        return {"minimum": 0, "mean": 0.0, "maximum": 0, "total": 0}
    return {
        "minimum": min(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
        "total": sum(values),
    }


def _stable_batch(batch: FastCompletionBatch) -> dict[str, Any]:
    """Remove invocation disposition while retaining immutable journals."""

    payload = batch.model_dump()
    return {
        "logical_completions": payload["logical_completions"],
        "unique_records": [
            {
                key: value
                for key, value in row.items()
                if key not in _RECORD_DISPOSITION_FIELDS
            }
            for row in payload["unique_records"]
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


def _record_by_messages(batch: FastCompletionBatch) -> dict[str, Mapping[str, Any]]:
    return {
        row.messages_sha256: row.model_dump() for row in batch.unique_records
    }


def _route_style(args: argparse.Namespace) -> RoutedRepairStyle:
    return normalize_repair_style(str(args.style))


def _validate_runtime_settings(args: argparse.Namespace) -> None:
    if str(args.gateway_url) != DEFAULT_GATEWAY_URL:
        raise ValueError("routed repair requires the locked central-dev gateway")
    if str(args.terra_model) != DEFAULT_TERRA_MODEL:
        raise ValueError("routed repair requires the locked Terra model route")
    if str(args.sol_model) != DEFAULT_SOL_MODEL:
        raise ValueError("routed repair requires the locked Sol model route")
    if type(args.max_concurrency) is not int or args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be positive")


def _load_population(args: argparse.Namespace) -> LockedEMRepairPopulation:
    population = load_locked_em_repair_population(
        Path(args.retrieval),
        expected_retrieval_sha256=str(args.expected_retrieval_sha256),
        baseline_final_answers_path=Path(args.baseline_answers),
        expected_baseline_final_answers_sha256=str(
            args.expected_baseline_answers_sha256
        ),
    )
    if population.question_count != args.expected_question_count:
        raise ValueError(
            "sealed population question count changed: "
            f"{population.question_count} != {args.expected_question_count}"
        )
    return population


@dataclass(frozen=True, slots=True)
class _TreatmentPlan:
    population: LockedEMRepairPopulation
    style: RoutedRepairStyle
    routes: tuple[RoutedRepairReceipt, ...]
    eligible_ordinals: tuple[int, ...]
    compression_prompts: tuple[RoutedCompressionPrompt, ...]
    compression_preflight: FastPromptPopulation
    route_plan: Mapping[str, Any]
    preflight: Mapping[str, Any]

    @property
    def eligible_count(self) -> int:
        return len(self.eligible_ordinals)

    @property
    def exact_compression_calls(self) -> int:
        return self.eligible_count


def _build_treatment_plan(args: argparse.Namespace) -> _TreatmentPlan:
    _validate_runtime_settings(args)
    population = _load_population(args)
    style = _route_style(args)
    routes = tuple(route_question(row.question.dated_question) for row in population.rows)
    for row, route in zip(population.rows, routes, strict=True):
        if route.question_sha256 != row.question.dated_question_sha256:
            raise RuntimeError("question-only route changed dated-question identity")
    eligible_ordinals = tuple(
        ordinal for ordinal, route in enumerate(routes) if route.style is style
    )
    if not eligible_ordinals:
        raise ValueError(f"selected route {style.value!r} has no eligible questions")
    compression_prompts = tuple(
        build_routed_fact_compression_prompt(
            population.rows[ordinal].question,  # type: ignore[arg-type]
            routes[ordinal],
            stage_id=DEFAULT_EM_STAGE_ID,
            max_prompt_tokens=MAX_ROUTED_PROMPT_TOKENS,
        )
        for ordinal in eligible_ordinals
    )
    compression_preflight = preflight_fast_completion_prompts(
        [prompt.as_mappings() for prompt in compression_prompts],
        max_prompt_tokens=MAX_ROUTED_PROMPT_TOKENS,
    )
    if (
        compression_preflight.logical_prompt_count != len(eligible_ordinals)
        or compression_preflight.unique_prompt_count != len(eligible_ordinals)
    ):
        raise RuntimeError("compression prompts are not unique per eligible question")

    prompt_by_ordinal = dict(zip(eligible_ordinals, compression_prompts, strict=True))
    route_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    for ordinal, (row, route) in enumerate(zip(population.rows, routes, strict=True)):
        s0, em_delta = episodic_neighborhood(
            row.question,  # type: ignore[arg-type]
            stage_id=DEFAULT_EM_STAGE_ID,
        )
        s1 = row.question.stage(DEFAULT_EM_STAGE_ID).evidence
        eligible = ordinal in prompt_by_ordinal
        route_rows.append(
            {
                "ordinal": ordinal,
                "question_id": row.question.question_id,
                "question_sha256": row.question.question_sha256,
                "dated_question_sha256": row.question.dated_question_sha256,
                "retrieval_question_part_sha256": (
                    row.question.retrieval_question_part_sha256
                ),
                "adapter_row_binding_sha256": row.binding_sha256,
                "route": route.identity_payload(),
                "eligible": eligible,
                "baseline_prediction_sha256": row.baseline.text_sha256,
                "s0_stage_receipt_sha256": row.question.stages[0].stage_receipt_sha256,
                "s1_stage_receipt_sha256": row.question.stages[1].stage_receipt_sha256,
                "s1_evidence_projection_sha256": (
                    row.question.stages[1].evidence_projection_sha256
                ),
                "compression_prompt_receipt_sha256": (
                    prompt_by_ordinal[ordinal].receipt_sha256 if eligible else None
                ),
            }
        )
        if eligible:
            prompt = prompt_by_ordinal[ordinal]
            budget_rows.append(
                {
                    "ordinal": ordinal,
                    "question_id": row.question.question_id,
                    "style": route.style.value,
                    "s0_protected_rows": len(s0),
                    "sealed_s1_rows": len(s1),
                    "post_selection_em_delta_rows": len(em_delta),
                    "compression_prompt_token_proxy": prompt.prompt_token_proxy,
                    "compression_prompt_cap": prompt.max_prompt_token_proxy,
                    "compression_output_token_cap": (
                        MAX_COMPRESSION_OUTPUT_TOKENS
                    ),
                    "answer_prompt_preflighted": False,
                    "answer_prompt_cap": MAX_ROUTED_PROMPT_TOKENS,
                    "answer_output_token_cap": MAX_ANSWER_OUTPUT_TOKENS,
                }
            )

    route_counts = Counter(route.style.value for route in routes)
    route_plan: dict[str, Any] = {
        "format": ROUTE_PLAN_FORMAT,
        "treatment_style": style.value,
        "adapter_binding_sha256": population.binding_sha256,
        "retrieval_sha256": population.retrieval_sha256,
        "baseline_final_answers_sha256": (
            population.baseline_final_answers_sha256
        ),
        "population_identity_sha256": population.population_identity_sha256,
        "question_count": population.question_count,
        "eligible_question_count": len(eligible_ordinals),
        "route_counts": dict(sorted(route_counts.items())),
        "questions": route_rows,
        "budget": {
            "selection_policy": "sealed_s1_unchanged",
            "post_selection_exclusion_policy": "exact_s0_prefix_then_s1_minus_s0",
            "shared_unbounded_tail_attached": False,
            "dependent_answer_prompts_preflighted": False,
            "workspace_cap_tokens": MAX_ROUTED_PROMPT_TOKENS,
            "eligible_rows": budget_rows,
            "compression_prompt_tokens": _distribution(
                [prompt.prompt_token_proxy for prompt in compression_prompts]
            ),
        },
        "compression_prompt_population": compression_preflight.model_dump(),
        "required_authorized_compression_calls": len(eligible_ordinals),
        "dependent_answer_calls_require_sealed_compression": True,
        "provider_calls": 0,
        "gold_loaded": False,
        "benchmark_categories_loaded": False,
        "benchmark_source_labels_loaded": False,
    }
    if _contains_forbidden_gold_key(route_plan):
        raise RuntimeError("route plan crossed the gold firewall")
    route_plan_sha = _digest(route_plan)
    preflight: dict[str, Any] = {
        "format": PREFLIGHT_FORMAT,
        "route_plan_sha256": route_plan_sha,
        "treatment_style": style.value,
        "question_count": population.question_count,
        "eligible_question_count": len(eligible_ordinals),
        "compression_prompt_population": compression_preflight.model_dump(),
        "maximum_dependent_answer_logical_calls": len(eligible_ordinals),
        "dependent_answer_calls_preflighted": False,
        "required_authorized_provider_calls": len(eligible_ordinals),
        "authorized_call_kind": "terra_compression",
        "method_budget": route_plan["budget"],
        "provider_calls": 0,
        "writes": 0,
        "gold_loaded": False,
        "benchmark_categories_loaded": False,
        "benchmark_source_labels_loaded": False,
    }
    if _contains_forbidden_gold_key(preflight):
        raise RuntimeError("preflight crossed the gold firewall")
    return _TreatmentPlan(
        population=population,
        style=style,
        routes=routes,
        eligible_ordinals=eligible_ordinals,
        compression_prompts=compression_prompts,
        compression_preflight=compression_preflight,
        route_plan=route_plan,
        preflight=preflight,
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("preflight forbids provider access and authorization")
    return dict(_build_treatment_plan(args).preflight)


def run_plan(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("plan forbids provider access and authorization")
    plan = _build_treatment_plan(args)
    root = Path(args.output_root)
    route_digest = _publish(root / "route-plan.json", plan.route_plan)
    _publish(root / "preflight.json", plan.preflight)
    return dict(plan.route_plan), route_digest


def _runtime(
    plan: _TreatmentPlan,
    args: argparse.Namespace,
    *,
    kind: str,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    client: Any | None,
) -> FastCompletionRuntime:
    if kind not in {"compression", "answer"}:
        raise ValueError("Terra runtime kind must be compression or answer")
    output_tokens = (
        MAX_COMPRESSION_OUTPUT_TOKENS
        if kind == "compression"
        else MAX_ANSWER_OUTPUT_TOKENS
    )
    return FastCompletionRuntime(
        checkpoint_dir=Path(args.output_root) / f"terra-{kind}-calls",
        prompt_population=prompts,
        model=DEFAULT_TERRA_MODEL,
        client=client,
        max_prompt_tokens=MAX_ROUTED_PROMPT_TOKENS,
        max_new_tokens=output_tokens,
        max_concurrency=args.max_concurrency,
        retries=0,
        benchmark_provenance={
            "experiment_format": RUN_FORMAT,
            "kind": kind,
            "route_plan_sha256": _digest(plan.route_plan),
            "adapter_binding_sha256": plan.population.binding_sha256,
            "treatment_style": plan.style.value,
            "eligible_question_count": plan.eligible_count,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_loaded": False,
            "benchmark_categories_loaded": False,
            "benchmark_source_labels_loaded": False,
        },
    )


_make_provider_client = em_runner._make_provider_client


def _compression_path(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / "compression.json"


def _compression_rows(
    plan: _TreatmentPlan, batch: FastCompletionBatch
) -> list[dict[str, Any]]:
    records = _record_by_messages(batch)
    rows: list[dict[str, Any]] = []
    for index, (ordinal, prompt, completion) in enumerate(
        zip(
            plan.eligible_ordinals,
            plan.compression_prompts,
            batch.logical_completions,
            strict=True,
        )
    ):
        try:
            parsed = parse_fact_compression(
                plan.population.rows[ordinal].question,  # type: ignore[arg-type]
                completion,
                stage_id=DEFAULT_EM_STAGE_ID,
                max_facts=MAX_FACTS,
            )
        except EMFactMemoryError:
            status, fact_count, receipt = "invalid", 0, None
        else:
            fact_count, receipt = len(parsed.facts), parsed.receipt_sha256
            if not fact_count:
                status = "empty"
            elif (
                plan.style is RoutedRepairStyle.NUMERIC_REDUCE
                and not numeric_facts_are_quote_grounded(parsed.facts)
            ):
                status = "unsupported_numeric_literal"
            else:
                status = "valid"
        record = records[prompt.messages_sha256]
        rows.append(
            {
                "eligible_index": index,
                "ordinal": ordinal,
                "question_id": plan.population.rows[ordinal].question.question_id,
                "compression_status": status,
                "validated_fact_count": fact_count,
                "compression_receipt_sha256": receipt,
                "completion_sha256": quote_sha256(completion),
                "prompt_receipt_sha256": prompt.receipt_sha256,
                "response_journal_sha256": record["response_journal_sha256"],
            }
        )
    return rows


def _compression_artifact(
    plan: _TreatmentPlan, batch: FastCompletionBatch
) -> dict[str, Any]:
    rows = _compression_rows(plan, batch)
    statuses = Counter(row["compression_status"] for row in rows)
    return {
        "format": COMPRESSION_FORMAT,
        "route_plan_sha256": _digest(plan.route_plan),
        "adapter_binding_sha256": plan.population.binding_sha256,
        "treatment_style": plan.style.value,
        "eligible_question_count": plan.eligible_count,
        "required_authorized_compression_calls": plan.exact_compression_calls,
        "completion_batch": _stable_batch(batch),
        "status_counts": dict(sorted(statuses.items())),
        "questions": rows,
        "gold_loaded": False,
        "benchmark_categories_loaded": False,
        "benchmark_source_labels_loaded": False,
    }


def run_compression(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    plan = _build_treatment_plan(args)
    if not args.enable_provider:
        raise ValueError("compression-run requires --enable-provider")
    if args.authorized_provider_calls != plan.exact_compression_calls:
        raise ValueError(
            "--authorized-provider-calls must exactly equal the compression "
            f"population ({args.authorized_provider_calls} != "
            f"{plan.exact_compression_calls})"
        )
    path = _compression_path(args)
    if path.exists():
        existing, _ = _read(path)
        if existing.get("route_plan_sha256") != _digest(plan.route_plan):
            raise FileExistsError("compression artifact belongs to another plan")
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    if not api_key:
        raise RuntimeError(f"provider API key is empty: {args.api_key_env}")
    client = _make_provider_client(api_key, DEFAULT_GATEWAY_URL)
    try:
        batch = _runtime(
            plan,
            args,
            kind="compression",
            prompts=[prompt.as_mappings() for prompt in plan.compression_prompts],
            client=client,
        ).run()
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    if (
        batch.prompt_population.unique_prompt_count != plan.exact_compression_calls
        or batch.usage.physical_calls + batch.usage.checkpoint_hits
        != plan.exact_compression_calls
    ):
        raise RuntimeError("compression journal population changed")
    artifact = _compression_artifact(plan, batch)
    return artifact, _publish(path, artifact)


def _verified_compression(
    plan: _TreatmentPlan, args: argparse.Namespace
) -> tuple[dict[str, Any], str, FastCompletionBatch]:
    source, digest = _read(_compression_path(args))
    batch = _runtime(
        plan,
        args,
        kind="compression",
        prompts=[prompt.as_mappings() for prompt in plan.compression_prompts],
        client=None,
    ).run()
    expected = _compression_artifact(plan, batch)
    if canonical_json_bytes(source) != canonical_json_bytes(expected):
        raise ValueError("compression artifact differs from immutable journals")
    return source, digest, batch


def run_compression_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("compression replay forbids provider access")
    plan = _build_treatment_plan(args)
    source, _digest_value, batch = _verified_compression(plan, args)
    if batch.usage.physical_calls:
        raise RuntimeError("compression replay unexpectedly called Terra")
    return source, _publish(Path(args.output_root) / "compression-replay.json", source)


@dataclass(frozen=True, slots=True)
class _AnswerPlan:
    treatment: _TreatmentPlan
    compression: Mapping[str, Any]
    compression_sha256: str
    compression_batch: FastCompletionBatch
    valid_eligible_indexes: tuple[int, ...]
    prompts: tuple[RoutedAnswerPrompt, ...]
    answer_fallback_reasons: Mapping[int, str]
    preflight: FastPromptPopulation | None

    @property
    def exact_answer_calls(self) -> int:
        return 0 if self.preflight is None else self.preflight.unique_prompt_count


def _build_answer_plan(args: argparse.Namespace) -> _AnswerPlan:
    treatment = _build_treatment_plan(args)
    compression, compression_sha, compression_batch = _verified_compression(
        treatment, args
    )
    candidates = tuple(
        int(row["eligible_index"])
        for row in compression["questions"]
        if row["compression_status"] == "valid"
    )
    valid_indexes: list[int] = []
    prompts: list[RoutedAnswerPrompt] = []
    answer_fallback_reasons: dict[int, str] = {}
    for index in candidates:
        ordinal = treatment.eligible_ordinals[index]
        prompt = build_routed_answer_prompt(
            treatment.population.rows[ordinal].question,  # type: ignore[arg-type]
            compression_batch.logical_completions[index],
            treatment.routes[ordinal],
            stage_id=DEFAULT_EM_STAGE_ID,
            measured_arm="facts",
            max_prompt_tokens=MAX_ROUTED_PROMPT_TOKENS,
            responder_output_token_reserve=MAX_ANSWER_OUTPUT_TOKENS,
            max_facts=MAX_FACTS,
        )
        if prompt.fallback_reason is not None:
            answer_fallback_reasons[index] = prompt.fallback_reason
            continue
        if prompt.used_raw_s1_fallback or prompt.effective_arm != "facts":
            raise RuntimeError("facts answer prompt changed its measured arm")
        valid_indexes.append(index)
        prompts.append(prompt)
    preflight = (
        None
        if not prompts
        else preflight_fast_completion_prompts(
            [prompt.as_mappings() for prompt in prompts],
            max_prompt_tokens=MAX_ROUTED_PROMPT_TOKENS,
        )
    )
    if preflight is not None and preflight.unique_prompt_count != len(prompts):
        raise RuntimeError("answer prompts are not unique per valid compression")
    return _AnswerPlan(
        treatment=treatment,
        compression=compression,
        compression_sha256=compression_sha,
        compression_batch=compression_batch,
        valid_eligible_indexes=tuple(valid_indexes),
        prompts=tuple(prompts),
        answer_fallback_reasons=answer_fallback_reasons,
        preflight=preflight,
    )


def run_answer_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("answer preflight forbids provider access")
    plan = _build_answer_plan(args)
    fallback = plan.treatment.eligible_count - len(plan.prompts)
    fallback_reasons = Counter(
        str(row["compression_status"])
        for row in plan.compression["questions"]
        if row["compression_status"] != "valid"
    )
    fallback_reasons.update(plan.answer_fallback_reasons.values())
    return {
        "format": ANSWER_PREFLIGHT_FORMAT,
        "route_plan_sha256": _digest(plan.treatment.route_plan),
        "compression_artifact_sha256": plan.compression_sha256,
        "eligible_question_count": plan.treatment.eligible_count,
        "valid_compression_count": len(plan.prompts),
        "baseline_fallback_count": fallback,
        "baseline_fallback_reasons": dict(sorted(fallback_reasons.items())),
        "answer_prompt_population": (
            None if plan.preflight is None else plan.preflight.model_dump()
        ),
        "required_authorized_provider_calls": plan.exact_answer_calls,
        "authorized_call_kind": "terra_answer",
        "provider_calls": 0,
        "writes": 0,
        "gold_loaded": False,
    }


def _answer_runtime(
    plan: _AnswerPlan, args: argparse.Namespace, *, client: Any | None
) -> FastCompletionRuntime:
    if not plan.prompts:
        raise ValueError("zero-call answer plan has no runtime")
    return FastCompletionRuntime(
        checkpoint_dir=Path(args.output_root) / "terra-answer-calls",
        prompt_population=[prompt.as_mappings() for prompt in plan.prompts],
        model=DEFAULT_TERRA_MODEL,
        client=client,
        max_prompt_tokens=MAX_ROUTED_PROMPT_TOKENS,
        max_new_tokens=MAX_ANSWER_OUTPUT_TOKENS,
        max_concurrency=args.max_concurrency,
        retries=0,
        benchmark_provenance={
            "experiment_format": RUN_FORMAT,
            "kind": "answer",
            "route_plan_sha256": _digest(plan.treatment.route_plan),
            "compression_artifact_sha256": plan.compression_sha256,
            "treatment_style": plan.treatment.style.value,
            "valid_compression_count": len(plan.prompts),
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_loaded": False,
        },
    )


def _run_artifact(
    plan: _AnswerPlan,
    args: argparse.Namespace,
    answer_batch: FastCompletionBatch | None,
) -> dict[str, Any]:
    if (answer_batch is None) != (not plan.prompts):
        raise ValueError("answer batch does not match preflight")
    answers_by_index = {
        index: logical
        for index, logical in zip(
            plan.valid_eligible_indexes,
            (() if answer_batch is None else answer_batch.logical_completions),
            strict=True,
        )
    }
    answer_prompt_by_index = dict(
        zip(plan.valid_eligible_indexes, plan.prompts, strict=True)
    )
    answer_records = {} if answer_batch is None else _record_by_messages(answer_batch)
    eligible_index = {
        ordinal: index
        for index, ordinal in enumerate(plan.treatment.eligible_ordinals)
    }
    compression_rows = {
        int(row["eligible_index"]): row for row in plan.compression["questions"]
    }
    questions, budgets = [], []
    for ordinal, (row, route) in enumerate(
        zip(plan.treatment.population.rows, plan.treatment.routes, strict=True)
    ):
        index = eligible_index.get(ordinal)
        prompt = None if index is None else answer_prompt_by_index.get(index)
        if index is None:
            prediction, kind, fallback = (
                row.baseline.text,
                "sealed_baseline_preserved",
                None,
            )
        elif prompt is None:
            status = plan.answer_fallback_reasons.get(
                index, str(compression_rows[index]["compression_status"])
            )
            prediction, kind, fallback = (
                row.baseline.text,
                "sealed_baseline_fallback",
                (
                    status
                    if index in plan.answer_fallback_reasons
                    else f"{status}_compression"
                ),
            )
        else:
            prediction, kind, fallback = (
                answers_by_index[index],
                "routed_terra_candidate",
                None,
            )
        response_journal = None
        if prompt is not None and answer_batch is not None:
            response_journal = answer_records[prompt.messages_sha256][
                "response_journal_sha256"
            ]
        questions.append(
            {
                "ordinal": ordinal,
                "question_id": row.question.question_id,
                "route_style": route.style.value,
                "eligible": index is not None,
                "prediction_kind": kind,
                "baseline_fallback_reason": fallback,
                "baseline_prediction_sha256": row.baseline.text_sha256,
                "prediction": prediction,
                "prediction_sha256": quote_sha256(prediction),
                "changed_from_baseline": quote_sha256(prediction)
                != row.baseline.text_sha256,
                "answer_prompt_receipt_sha256": (
                    None if prompt is None else prompt.receipt_sha256
                ),
                "answer_response_journal_sha256": response_journal,
            }
        )
        if index is not None:
            root, delta = episodic_neighborhood(
                row.question,  # type: ignore[arg-type]
                stage_id=DEFAULT_EM_STAGE_ID,
            )
            compression_row = compression_rows[index]
            budgets.append(
                {
                    "ordinal": ordinal,
                    "question_id": row.question.question_id,
                    "s0_protected_rows": len(root),
                    "sealed_s1_rows": len(row.question.stage(DEFAULT_EM_STAGE_ID).evidence),
                    "post_selection_em_delta_rows": len(delta),
                    "compression_prompt_token_proxy": (
                        plan.treatment.compression_prompts[index].prompt_token_proxy
                    ),
                    "validated_fact_count": compression_row["validated_fact_count"],
                    "compression_status": compression_row["compression_status"],
                    "answer_prompt_token_proxy": (
                        None if prompt is None else prompt.prompt.prompt_token_proxy
                    ),
                    "answer_prompt_cap": MAX_ROUTED_PROMPT_TOKENS,
                    "answer_output_token_cap": MAX_ANSWER_OUTPUT_TOKENS,
                    "baseline_fallback": prompt is None,
                    "baseline_fallback_reason": fallback,
                    "raw_em_tail_attached": False,
                }
            )
    answer_tokens = [
        int(row["answer_prompt_token_proxy"])
        for row in budgets
        if row["answer_prompt_token_proxy"] is not None
    ]
    return {
        "format": RUN_FORMAT,
        "route_plan_sha256": _digest(plan.treatment.route_plan),
        "compression_artifact_sha256": plan.compression_sha256,
        "treatment_style": plan.treatment.style.value,
        "adapter_binding_sha256": plan.treatment.population.binding_sha256,
        "retrieval_sha256": plan.treatment.population.retrieval_sha256,
        "baseline_final_answers_sha256": (
            plan.treatment.population.baseline_final_answers_sha256
        ),
        "population_identity_sha256": (
            plan.treatment.population.population_identity_sha256
        ),
        "question_count": plan.treatment.population.question_count,
        "eligible_question_count": plan.treatment.eligible_count,
        "valid_compression_count": len(plan.prompts),
        "baseline_fallback_count": plan.treatment.eligible_count - len(plan.prompts),
        "required_authorized_answer_calls": plan.exact_answer_calls,
        "total_sealed_terra_calls": (
            plan.treatment.exact_compression_calls + plan.exact_answer_calls
        ),
        "settings": {
            "gateway_url": DEFAULT_GATEWAY_URL,
            "model": DEFAULT_TERRA_MODEL,
            "max_concurrency": args.max_concurrency,
            "max_prompt_tokens": MAX_ROUTED_PROMPT_TOKENS,
            "compression_output_tokens": MAX_COMPRESSION_OUTPUT_TOKENS,
            "answer_output_tokens": MAX_ANSWER_OUTPUT_TOKENS,
            "retries": 0,
        },
        "answer_completion_batch": (
            None if answer_batch is None else _stable_batch(answer_batch)
        ),
        "budget": {
            "selection_policy": "sealed_s1_unchanged",
            "post_selection_exclusion_policy": "exact_s0_prefix_then_s1_minus_s0",
            "shared_unbounded_tail_attached": False,
            "workspace_cap_tokens": MAX_ROUTED_PROMPT_TOKENS,
            "compression_prompt_tokens": _distribution(
                [prompt.prompt_token_proxy for prompt in plan.treatment.compression_prompts]
            ),
            "answer_prompt_tokens": _distribution(answer_tokens),
            "baseline_fallback_questions": plan.treatment.eligible_count
            - len(plan.prompts),
            "eligible_rows": budgets,
        },
        "questions": questions,
        "gold_loaded": False,
        "benchmark_categories_loaded": False,
        "benchmark_source_labels_loaded": False,
        "noneligible_predictions_preserved": True,
        "invalid_or_empty_compressions_preserve_baseline": True,
        "retained_request_token_state_bytes": 0,
    }


def _run_path(args: argparse.Namespace) -> Path:
    return Path(args.run_artifact or Path(args.output_root) / "run.json")


def run_treatment(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    plan = _build_answer_plan(args)
    if not args.enable_provider:
        raise ValueError("run requires --enable-provider")
    if args.authorized_provider_calls != plan.exact_answer_calls:
        raise ValueError(
            "--authorized-provider-calls must exactly equal the sealed answer "
            f"population ({args.authorized_provider_calls} != "
            f"{plan.exact_answer_calls})"
        )
    path = _run_path(args)
    if path.exists():
        existing, _ = _read(path)
        if (
            existing.get("route_plan_sha256") != _digest(plan.treatment.route_plan)
            or existing.get("compression_artifact_sha256")
            != plan.compression_sha256
        ):
            raise FileExistsError("run artifact belongs to another answer plan")
    batch = None
    if plan.prompts:
        api_key = os.environ.get(str(args.api_key_env), "").strip()
        if not api_key:
            raise RuntimeError(f"provider API key is empty: {args.api_key_env}")
        client = _make_provider_client(api_key, DEFAULT_GATEWAY_URL)
        try:
            batch = _answer_runtime(plan, args, client=client).run()
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        if batch.usage.physical_calls + batch.usage.checkpoint_hits != plan.exact_answer_calls:
            raise RuntimeError("answer journal population changed")
    artifact = _run_artifact(plan, args, batch)
    if _contains_forbidden_gold_key(artifact):
        raise RuntimeError("run artifact crossed the gold firewall")
    return artifact, _publish(path, artifact)


def _verified_run(
    args: argparse.Namespace,
) -> tuple[_AnswerPlan, dict[str, Any], str, FastCompletionBatch | None]:
    plan = _build_answer_plan(args)
    source, source_sha = _read(_run_path(args))
    batch = None if not plan.prompts else _answer_runtime(plan, args, client=None).run()
    expected = _run_artifact(plan, args, batch)
    if canonical_json_bytes(source) != canonical_json_bytes(expected):
        raise ValueError("run artifact differs from immutable Terra journals")
    return plan, source, source_sha, batch


def run_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("replay forbids provider access and authorization")
    _plan, source, _source_sha, batch = _verified_run(args)
    if batch is not None and batch.usage.physical_calls:
        raise RuntimeError("Terra replay unexpectedly made provider calls")
    replay_path = Path(args.run_replay or Path(args.output_root) / "run-replay.json")
    return source, _publish(replay_path, source)


def _load_gold_population(
    dataset: Path,
    split: Path,
    population: LockedEMRepairPopulation,
) -> tuple[Any, ...]:
    samples, _shards, identity = build_locked_cumulative_population_identity(
        dataset,
        split,
        plan=LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    )
    questions = tuple(question for sample in samples for question in sample.questions)
    if (
        identity.get("population_identity_sha256")
        != population.population_identity_sha256
        or len(questions) != population.question_count
        or tuple(question.question_id for question in questions)
        != tuple(row.question.question_id for row in population.rows)
    ):
        raise RuntimeError("post-seal gold population changed identity or order")
    return questions


def run_score(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("score forbids provider access and authorization")
    if args.dataset is None:
        raise ValueError("score requires --dataset")
    plan, run, run_sha, answers = _verified_run(args)
    if plan.compression_batch.usage.physical_calls or (
        answers is not None and answers.usage.physical_calls
    ):
        raise RuntimeError("score verification unexpectedly made provider calls")
    population = plan.treatment.population
    gold = _load_gold_population(Path(args.dataset), Path(args.split), population)
    rows: list[dict[str, Any]] = []
    for source, baseline, reference in zip(
        run["questions"], population.rows, gold, strict=True
    ):
        prediction = str(source["prediction"])
        baseline_prediction = baseline.baseline.text
        candidate_exact = exact_match(prediction, reference.answer)
        baseline_exact = exact_match(baseline_prediction, reference.answer)
        rows.append(
            {
                "ordinal": source["ordinal"],
                "question_id": source["question_id"],
                "eligible": source["eligible"],
                "route_style": source["route_style"],
                "prediction_sha256": source["prediction_sha256"],
                "baseline_prediction_sha256": baseline.baseline.text_sha256,
                "gold_answer_sha256": quote_sha256(reference.answer),
                "candidate_exact_match": candidate_exact,
                "baseline_exact_match": baseline_exact,
                "candidate_f1": f1_score(prediction, reference.answer),
                "baseline_f1": f1_score(baseline_prediction, reference.answer),
                "exact_rescue": candidate_exact and not baseline_exact,
                "exact_regression": baseline_exact and not candidate_exact,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    result: dict[str, Any] = {
        "format": SCORE_FORMAT,
        "run_artifact_sha256": run_sha,
        "retrieval_sha256": population.retrieval_sha256,
        "population_identity_sha256": population.population_identity_sha256,
        "question_count": len(rows),
        "eligible_question_count": len(eligible),
        "gold_loaded_post_run_verification": True,
        "aggregate": {
            "baseline_exact_matches": sum(
                bool(row["baseline_exact_match"]) for row in rows
            ),
            "candidate_exact_matches": sum(
                bool(row["candidate_exact_match"]) for row in rows
            ),
            "baseline_mean_f1": statistics.fmean(
                float(row["baseline_f1"]) for row in rows
            ),
            "candidate_mean_f1": statistics.fmean(
                float(row["candidate_f1"]) for row in rows
            ),
            "eligible_exact_rescues": sum(
                bool(row["exact_rescue"]) for row in eligible
            ),
            "eligible_exact_regressions": sum(
                bool(row["exact_regression"]) for row in eligible
            ),
        },
        "rows": rows,
    }
    return result, _publish(Path(args.output_root) / "local-score.json", result)


def _load_baseline_judge(
    args: argparse.Namespace,
    plan: _TreatmentPlan,
) -> tuple[dict[str, Any], str]:
    judge, digest = _read(
        Path(args.baseline_judge),
        expected_sha256=str(args.expected_baseline_judge_sha256),
    )
    questions = judge.get("questions")
    if (
        judge.get("format")
        != "memory-condense-fixed-stage-final-answer-semantic-judge-score-v1"
        or judge.get("retrieval_sha256") != plan.population.retrieval_sha256
        or judge.get("final_answer_artifact_sha256")
        != plan.population.baseline_final_answers_sha256
        or judge.get("population_identity_sha256")
        != plan.population.population_identity_sha256
        or judge.get("question_count") != plan.population.question_count
        or not isinstance(questions, list)
        or len(questions) != plan.population.question_count
    ):
        raise ValueError("baseline semantic judge changed its sealed binding")
    for ordinal, (source, baseline) in enumerate(
        zip(questions, plan.population.rows, strict=True)
    ):
        if (
            not isinstance(source, Mapping)
            or source.get("ordinal") != ordinal
            or source.get("question_id") != baseline.question.question_id
            or source.get("prediction_sha256") != baseline.baseline.text_sha256
            or type(source.get("correct")) is not bool
        ):
            raise ValueError("baseline semantic verdict population changed")
    return judge, digest


@dataclass(frozen=True, slots=True)
class _JudgeRow:
    ordinal: int
    question_id: str
    messages: tuple[dict[str, str], ...]
    messages_sha256: str
    question_sha256: str
    gold_answer_sha256: str
    prediction_sha256: str


@dataclass(frozen=True, slots=True)
class _JudgePlan:
    treatment: _TreatmentPlan
    run: Mapping[str, Any]
    run_sha256: str
    baseline_judge: Mapping[str, Any]
    baseline_judge_sha256: str
    rows: tuple[_JudgeRow, ...]
    preflight: FastPromptPopulation | None
    gold_population_sha256: str

    @property
    def unique_calls(self) -> int:
        return 0 if self.preflight is None else self.preflight.unique_prompt_count


def _build_judge_plan(args: argparse.Namespace) -> _JudgePlan:
    if args.dataset is None:
        raise ValueError("judge phases require --dataset")
    answer_plan, run, run_sha, answers = _verified_run(args)
    if answer_plan.compression_batch.usage.physical_calls or (
        answers is not None and answers.usage.physical_calls
    ):
        raise RuntimeError("judge preflight unexpectedly made Terra calls")
    treatment = answer_plan.treatment
    baseline_judge, baseline_judge_sha = _load_baseline_judge(args, treatment)
    gold = _load_gold_population(
        Path(args.dataset), Path(args.split), treatment.population
    )
    rows: list[_JudgeRow] = []
    gold_projection: list[dict[str, Any]] = []
    for source, reference in zip(run["questions"], gold, strict=True):
        gold_projection.append(
            {
                "ordinal": source["ordinal"],
                "question_id": source["question_id"],
                "question_sha256": quote_sha256(reference.question),
                "dated_question_sha256": quote_sha256(reference.dated_question),
                "gold_answer_sha256": quote_sha256(reference.answer),
            }
        )
        if not source["eligible"] or not source["changed_from_baseline"]:
            continue
        messages = tuple(
            dict(message)
            for message in build_judge_prompt(
                reference.question,
                reference.answer,
                str(source["prediction"]),
            )
        )
        rows.append(
            _JudgeRow(
                ordinal=int(source["ordinal"]),
                question_id=str(source["question_id"]),
                messages=messages,
                messages_sha256=identity_sha256(list(messages)),
                question_sha256=quote_sha256(reference.question),
                gold_answer_sha256=quote_sha256(reference.answer),
                prediction_sha256=str(source["prediction_sha256"]),
            )
        )
    preflight = (
        None
        if not rows
        else preflight_fast_completion_prompts(
            [row.messages for row in rows],
            max_prompt_tokens=MAX_JUDGE_PROMPT_TOKENS,
        )
    )
    if preflight is not None and tuple(row.messages_sha256 for row in rows) != tuple(
        row.messages_sha256 for row in preflight.ordered_rows
    ):
        raise RuntimeError("Sol preflight changed the planned prompt order")
    return _JudgePlan(
        treatment=treatment,
        run=run,
        run_sha256=run_sha,
        baseline_judge=baseline_judge,
        baseline_judge_sha256=baseline_judge_sha,
        rows=tuple(rows),
        preflight=preflight,
        gold_population_sha256=identity_sha256(gold_projection),
    )


def _judge_binding(plan: _JudgePlan) -> dict[str, Any]:
    prompt_population = None if plan.preflight is None else plan.preflight.model_dump()
    return {
        "format": JUDGE_FORMAT + "-campaign",
        "treatment_run_sha256": plan.run_sha256,
        "route_plan_sha256": plan.run["route_plan_sha256"],
        "baseline_judge_sha256": plan.baseline_judge_sha256,
        "retrieval_sha256": plan.treatment.population.retrieval_sha256,
        "population_identity_sha256": (
            plan.treatment.population.population_identity_sha256
        ),
        "gold_population_sha256": plan.gold_population_sha256,
        "question_count": plan.treatment.population.question_count,
        "eligible_question_count": plan.treatment.eligible_count,
        "changed_eligible_prediction_count": len(plan.rows),
        "unique_sol_call_count": plan.unique_calls,
        "judge_prompt_population_sha256": (
            None
            if prompt_population is None
            else prompt_population["prompt_population_sha256"]
        ),
        "gateway_url": DEFAULT_GATEWAY_URL,
        "model": DEFAULT_SOL_MODEL,
        "max_prompt_tokens": MAX_JUDGE_PROMPT_TOKENS,
        "max_new_tokens": JUDGE_MAX_TOKENS,
        "retries": 0,
        "gold_loaded_post_run_verification": True,
        "explicit_gold_answer_field_persisted": False,
    }


def run_judge_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("judge preflight forbids provider access and authorization")
    plan = _build_judge_plan(args)
    tokens = (
        []
        if plan.preflight is None
        else [row.prompt_token_proxy for row in plan.preflight.ordered_rows]
    )
    return {
        "format": JUDGE_PREFLIGHT_FORMAT,
        "campaign_binding": _judge_binding(plan),
        "changed_eligible_prediction_count": len(plan.rows),
        "logical_prompt_count": len(plan.rows),
        "unique_prompt_count": plan.unique_calls,
        "required_authorized_provider_calls": plan.unique_calls,
        "judge_prompt_token_proxy": _distribution(tokens),
        "provider_calls": 0,
        "writes": 0,
        "gold_loaded_post_run_verification": True,
        "explicit_gold_answer_field_persisted": False,
    }


def _judge_runtime(
    plan: _JudgePlan,
    args: argparse.Namespace,
    *,
    client: Any | None,
) -> FastCompletionRuntime:
    if not plan.rows:
        raise ValueError("zero-call judge plan has no completion runtime")
    return FastCompletionRuntime(
        checkpoint_dir=Path(args.output_root) / "sol-judge-calls",
        prompt_population=[row.messages for row in plan.rows],
        model=DEFAULT_SOL_MODEL,
        client=client,
        max_prompt_tokens=MAX_JUDGE_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=args.max_concurrency,
        retries=0,
        benchmark_provenance={
            **_judge_binding(plan),
            "authorized_unique_calls": plan.unique_calls,
        },
    )


def _judge_artifact(
    plan: _JudgePlan,
    batch: FastCompletionBatch | None,
) -> dict[str, Any]:
    if (batch is None) != (not plan.rows):
        raise ValueError("judge batch does not match changed eligible population")
    records = {} if batch is None else _record_by_messages(batch)
    completions = () if batch is None else batch.logical_completions
    new_by_ordinal: dict[int, dict[str, Any]] = {}
    for source, verdict_text in zip(plan.rows, completions, strict=True):
        record = records[source.messages_sha256]
        new_by_ordinal[source.ordinal] = {
            "correct": parse_binary_judge_verdict(verdict_text),
            "judge_messages_sha256": source.messages_sha256,
            "judge_call_key_sha256": record["call_key_sha256"],
            "judge_request_journal_sha256": record["request_journal_sha256"],
            "judge_response_journal_sha256": record["response_journal_sha256"],
            "judge_verdict_sha256": quote_sha256(verdict_text),
        }
    baseline_questions = plan.baseline_judge["questions"]
    rows: list[dict[str, Any]] = []
    for source, baseline in zip(
        plan.run["questions"], baseline_questions, strict=True
    ):
        ordinal = int(source["ordinal"])
        new = new_by_ordinal.get(ordinal)
        baseline_correct = bool(baseline["correct"])
        correct = baseline_correct if new is None else bool(new["correct"])
        rows.append(
            {
                "ordinal": ordinal,
                "question_id": source["question_id"],
                "eligible": source["eligible"],
                "route_style": source["route_style"],
                "prediction_sha256": source["prediction_sha256"],
                "baseline_prediction_sha256": source[
                    "baseline_prediction_sha256"
                ],
                "changed_from_baseline": source["changed_from_baseline"],
                "verdict_source": (
                    "sealed_baseline_judge" if new is None else "new_sol_judge"
                ),
                "baseline_correct": baseline_correct,
                "correct": correct,
                "rescued": correct and not baseline_correct,
                "regressed": baseline_correct and not correct,
                "baseline_judge_response_journal_sha256": baseline[
                    "response_journal_sha256"
                ],
                "judge_messages_sha256": (
                    None if new is None else new["judge_messages_sha256"]
                ),
                "judge_call_key_sha256": (
                    None if new is None else new["judge_call_key_sha256"]
                ),
                "judge_request_journal_sha256": (
                    None if new is None else new["judge_request_journal_sha256"]
                ),
                "judge_response_journal_sha256": (
                    None if new is None else new["judge_response_journal_sha256"]
                ),
                "judge_verdict_sha256": (
                    None if new is None else new["judge_verdict_sha256"]
                ),
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    artifact = {
        "format": JUDGE_FORMAT,
        "campaign_binding": _judge_binding(plan),
        "completion_batch": None if batch is None else _stable_batch(batch),
        "question_count": len(rows),
        "changed_eligible_prediction_count": len(plan.rows),
        "unique_sol_completion_count": plan.unique_calls,
        "aggregate": {
            "baseline_correct": sum(bool(row["baseline_correct"]) for row in rows),
            "candidate_correct": sum(bool(row["correct"]) for row in rows),
            "eligible_rescued": sum(bool(row["rescued"]) for row in eligible),
            "eligible_regressed": sum(bool(row["regressed"]) for row in eligible),
            "eligible_net_marginal": sum(bool(row["rescued"]) for row in eligible)
            - sum(bool(row["regressed"]) for row in eligible),
        },
        "questions": rows,
        "gold_loaded_post_run_verification": True,
        "explicit_gold_answer_field_persisted": False,
        "judge_completions_may_echo_gold": True,
        "unchanged_verdicts_reused_from_sealed_baseline": True,
        "retained_request_token_state_bytes": 0,
    }
    return artifact


def _judge_path(args: argparse.Namespace) -> Path:
    return Path(args.judge_artifact or Path(args.output_root) / "semantic-judge-sol.json")


def run_judge(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    plan = _build_judge_plan(args)
    if not args.enable_provider:
        raise ValueError("judge-run requires the explicit --enable-provider gate")
    if args.authorized_provider_calls != plan.unique_calls:
        raise ValueError(
            "--authorized-provider-calls must exactly equal the changed eligible "
            f"Sol population ({args.authorized_provider_calls} != {plan.unique_calls})"
        )
    existing_path = _judge_path(args)
    if existing_path.exists():
        existing, _ = _read(existing_path)
        if existing.get("campaign_binding") != _judge_binding(plan):
            raise FileExistsError(
                "refusing provider access because semantic-judge-sol.json belongs "
                "to another treatment"
            )
    batch: FastCompletionBatch | None = None
    if plan.rows:
        api_key = os.environ.get(str(args.api_key_env), "").strip()
        if not api_key:
            raise RuntimeError(f"provider API key is empty: {args.api_key_env}")
        client = _make_provider_client(api_key, DEFAULT_GATEWAY_URL)
        try:
            batch = _judge_runtime(plan, args, client=client).run()
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        if (
            batch.prompt_population.unique_prompt_count != plan.unique_calls
            or batch.usage.physical_calls + batch.usage.checkpoint_hits
            != plan.unique_calls
        ):
            raise RuntimeError("Sol journal population changed after authorization")
    artifact = _judge_artifact(plan, batch)
    return artifact, _publish(_judge_path(args), artifact)


def run_judge_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls != 0:
        raise ValueError("judge replay forbids provider access and authorization")
    plan = _build_judge_plan(args)
    source, _source_sha = _read(_judge_path(args))
    batch = None if not plan.rows else _judge_runtime(plan, args, client=None).run()
    if batch is not None and batch.usage.physical_calls:
        raise RuntimeError("Sol replay unexpectedly made provider calls")
    reconstructed = _judge_artifact(plan, batch)
    if canonical_json_bytes(source) != canonical_json_bytes(reconstructed):
        raise ValueError("judge artifact differs from immutable Sol journals")
    replay_path = Path(
        args.judge_replay
        or Path(args.output_root) / "semantic-judge-sol-replay.json"
    )
    return source, _publish(replay_path, source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "preflight",
            "plan",
            "compression-run",
            "compression-replay",
            "answer-preflight",
            "run",
            "replay",
            "score",
            "judge-preflight",
            "judge-run",
            "judge-replay",
        ),
        default="preflight",
    )
    parser.add_argument(
        "--style",
        choices=tuple(style.value for style in RoutedRepairStyle),
        default=RoutedRepairStyle.NUMERIC_REDUCE.value,
    )
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--expected-retrieval-sha256", default=EXPECTED_RETRIEVAL_SHA256
    )
    parser.add_argument(
        "--baseline-answers", type=Path, default=DEFAULT_BASELINE_ANSWERS
    )
    parser.add_argument(
        "--expected-baseline-answers-sha256",
        default=EXPECTED_BASELINE_ANSWERS_SHA256,
    )
    parser.add_argument(
        "--baseline-judge", type=Path, default=DEFAULT_BASELINE_JUDGE
    )
    parser.add_argument(
        "--expected-baseline-judge-sha256",
        default=EXPECTED_BASELINE_JUDGE_SHA256,
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-artifact", type=Path)
    parser.add_argument("--run-replay", type=Path)
    parser.add_argument("--judge-artifact", type=Path)
    parser.add_argument("--judge-replay", type=Path)
    parser.add_argument(
        "--expected-question-count",
        type=int,
        default=DEFAULT_EXPECTED_QUESTION_COUNT,
    )
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--terra-model", default=DEFAULT_TERRA_MODEL)
    parser.add_argument("--sol-model", default=DEFAULT_SOL_MODEL)
    parser.add_argument("--api-key-env", default="LITELLM_KEY")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.phase == "preflight":
        result = run_preflight(args)
        print(
            "Routed repair preflight passed: "
            f"style={result['treatment_style']}; "
            f"eligible={result['eligible_question_count']}/"
            f"{result['question_count']}; "
            f"authorized_terra_calls="
            f"{result['required_authorized_provider_calls']}; "
            "provider_calls=0; writes=0",
            flush=True,
        )
        return 0
    if args.phase == "plan":
        result, digest = run_plan(args)
        print(
            f"Published routed plan ({digest}): "
            f"style={result['treatment_style']}; "
            f"eligible={result['eligible_question_count']}/"
            f"{result['question_count']}; provider_calls=0",
            flush=True,
        )
        return 0
    if args.phase == "run":
        result, digest = run_treatment(args)
    elif args.phase == "compression-run":
        result, digest = run_compression(args)
    elif args.phase == "compression-replay":
        result, digest = run_compression_replay(args)
    elif args.phase == "answer-preflight":
        result = run_answer_preflight(args)
        print(
            "Routed answer preflight passed: "
            f"valid={result['valid_compression_count']}/"
            f"{result['eligible_question_count']}; "
            f"baseline_fallback={result['baseline_fallback_count']}; "
            f"authorized_terra_calls="
            f"{result['required_authorized_provider_calls']}; "
            "provider_calls=0; writes=0",
            flush=True,
        )
        return 0
    elif args.phase == "replay":
        result, digest = run_replay(args)
    elif args.phase == "score":
        result, digest = run_score(args)
    elif args.phase == "judge-preflight":
        result = run_judge_preflight(args)
        print(
            "Routed Sol judge preflight passed: "
            f"changed={result['changed_eligible_prediction_count']}; "
            f"unique_calls={result['unique_prompt_count']}; "
            "provider_calls=0; writes=0",
            flush=True,
        )
        return 0
    elif args.phase == "judge-run":
        result, digest = run_judge(args)
    else:
        result, digest = run_judge_replay(args)
    print(f"Published {args.phase} artifact ({digest})", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "JUDGE_FORMAT",
    "PREFLIGHT_FORMAT",
    "ROUTE_PLAN_FORMAT",
    "RUN_FORMAT",
    "SCORE_FORMAT",
    "build_parser",
    "main",
    "run_judge",
    "run_judge_preflight",
    "run_judge_replay",
    "run_plan",
    "run_preflight",
    "run_compression",
    "run_compression_replay",
    "run_answer_preflight",
    "run_replay",
    "run_score",
    "run_treatment",
]
