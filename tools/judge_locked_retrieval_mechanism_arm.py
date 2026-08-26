#!/usr/bin/env python3
"""Judge a sealed retrieval arm; judge all S0, changed-only descendants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import os
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from dotenv import load_dotenv

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval import run_fast_1m_em_facts as em_runner
from memory_condense.eval._artifact_json import canonical_json_bytes
from memory_condense.eval._binary_judge_protocol import (
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.recall_guarded_cumulative_population import (
    LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    build_locked_cumulative_population_identity,
)
from tools._locked_em_repair_adapter import _read_canonical_artifact
from tools._routed_repair_routing import (
    ROUTED_REPAIR_ROUTING_FORMAT,
    route_question,
)


S0_ARM_LABEL = "S0_CONTROL"
EM_ARM_LABEL = "S0_PLUS_EM_FACTS"
ANSWER_RUN_FORMAT = "memory-condense-locked-retrieval-mechanism-arm-run-v1"
PREFLIGHT_FORMAT = "memory-condense-locked-retrieval-mechanism-sol-judge-preflight-v1"
JUDGE_FORMAT = "memory-condense-locked-retrieval-mechanism-sol-judge-v1"
BINDING_FORMAT = "memory-condense-locked-retrieval-mechanism-sol-judge-binding-v1"
DEFAULT_GATEWAY_URL = "https://central-dev.zt:4000/v1"
DEFAULT_SOL_MODEL = "codex_sdk/gpt-5.6-sol"
DEFAULT_CALLER_MODEL = "openai/codex_sdk/gpt-5.6-sol"
MAX_JUDGE_PROMPT_TOKENS = 8_000
EXPECTED_QUESTION_COUNT = 100

DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
    "/retrieval.json"
)
DEFAULT_BASELINE_ROOT = Path("eval_results/longmemeval-1m-fixed-s1-validation-20260826")
DEFAULT_BASELINE_ANSWERS = DEFAULT_BASELINE_ROOT / "final-answers.json"
DEFAULT_BASELINE_JUDGE = DEFAULT_BASELINE_ROOT / "final-answer-semantic-judge-sol.json"
DEFAULT_ARM_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/s0-control-v1"
)
DEFAULT_SPLIT = Path("docs/10 - Research Log/data/longmemeval-95-target-split-v2.json")
DEFAULT_TOPOLOGY_LEDGER = Path(
    "docs/10 - Research Log/data/longmemeval-locked-100-retrieval-style-ledger-v1.csv"
)
DEFAULT_S0_LOADER = "tools.run_locked_retrieval_mechanism_arm:load_verified_run"
LEGACY_DESCENDANT_LOADER_ARMS = {
    "tools.load_locked_s0_em_facts_arm:load_verified_run": frozenset(
        {EM_ARM_LABEL}
    ),
    "tools.load_locked_s0_cav_links_arm:load_verified_run": frozenset(
        {"S0_PLUS_CAV_LINKS"}
    ),
}

EXPECTED_RETRIEVAL_SHA256 = "e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f"
EXPECTED_BASELINE_ANSWERS_SHA256 = "d7fc47b8d1f372f002230c6ffe489dac8cd11bd71b35b8d3008b1255da2a38cd"
EXPECTED_BASELINE_JUDGE_SHA256 = "5dc56a240315c5577d1032d40429df7e39adad0f40a098abc371ee2ea2ec77df"
EXPECTED_TOPOLOGY_LEDGER_SHA256 = "59e7991ddaf23ec39c8bc1963b0e84b064217b2591d6e9f9774a3707fa10ae07"

_SHA = re.compile(r"^[0-9a-f]{64}$")
_LOADER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN = frozenset(
    {
        "answer_session_ids",
        "benchmark_category",
        "category",
        "evidence_sources",
        "gold",
        "gold_answer",
        "reference",
        "reference_answer",
        "retrieval_topology",
    }
)
_publish = em_runner._publish
_make_provider_client = em_runner._make_provider_client


class RetrievalArmJudgeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _AnswerRow:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    prediction: str
    prediction_sha256: str
    parent_prediction_sha256: str | None
    row_binding_sha256: str


@dataclass(frozen=True, slots=True)
class _AnswerRun:
    payload: Mapping[str, Any]
    sha256: str
    replay_sha256: str
    arm_label: str
    parent_arm_label: str | None
    parent_run_sha256: str | None
    retrieval_sha256: str
    baseline_answers_sha256: str
    population_identity_sha256: str
    historical_validator_binding_sha256: str
    rows: tuple[_AnswerRow, ...]
    loader_spec: str


@dataclass(frozen=True, slots=True)
class _PromptRow:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    gold_answer_sha256: str
    prediction_sha256: str
    baseline_prediction_sha256: str
    changed_from_baseline: bool
    demand_class: str
    demand_receipt_sha256: str
    messages: tuple[dict[str, str], ...] | None
    messages_sha256: str | None


@dataclass(frozen=True, slots=True)
class _Plan:
    candidate: _AnswerRun
    baseline_label: str
    baseline_run_sha256: str | None
    baseline_judge_sha256: str
    baseline_judge_replay_sha256: str | None
    baseline_correct: tuple[bool, ...]
    baseline_judge_row_sha256s: tuple[str, ...]
    prompt_rows: tuple[_PromptRow, ...]
    judged_rows: tuple[_PromptRow, ...]
    preflight: FastPromptPopulation | None
    gold_population_sha256: str
    topology_ledger_sha256: str
    topologies: tuple[str, ...]
    question_order_sha256: str
    prompt_seal_sha256: str

    @property
    def unique_calls(self) -> int:
        return 0 if self.preflight is None else self.preflight.unique_prompt_count


def _require(ok: Any, message: str) -> None:
    if not ok:
        raise RetrievalArmJudgeError(message)


def _sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and _SHA.fullmatch(value), f"invalid {label}")
    return str(value)


def _has_forbidden(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _FORBIDDEN or _has_forbidden(child)
            for key, child in value.items()
        )
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and any(
        _has_forbidden(child) for child in value
    )


def _read(path: Path, expected: str) -> tuple[dict[str, Any], str]:
    return _read_canonical_artifact(path, expected_sha256=_sha(expected, f"{path} SHA-256"))


def _normalize_answer_run(
    payload: Mapping[str, Any],
    *,
    sha256: str,
    replay_sha256: str,
    loader_spec: str,
    expected_question_count: int,
) -> _AnswerRun:
    """Normalize only fields shared by strict mechanism-specific loaders."""

    label = payload.get("arm_label")
    rows = payload.get("questions")
    _require(payload.get("format") == ANSWER_RUN_FORMAT, "answer run format changed")
    _require(payload.get("gold_loaded") is False and not _has_forbidden(payload), "answer run contains posthoc gold or topology fields")
    _require(isinstance(label, str) and label, "answer run omitted arm label")
    _require(payload.get("question_count") == expected_question_count and isinstance(rows, list) and len(rows) == expected_question_count, "answer run question population changed")
    identity = payload.get("arm_identity")
    legacy_descendant = not isinstance(identity, Mapping)
    if not legacy_descendant:
        _require(payload.get("arm_identity_sha256") == identity_sha256(dict(identity)) and identity.get("arm_label") == label, "answer arm identity changed")
    else:
        # Already-sealed EM/CAV descendants predate ``arm_identity``. Admit
        # only the exact arm names explicitly owned by their strict loaders.
        _require(
            label in LEGACY_DESCENDANT_LOADER_ARMS.get(loader_spec, ()),
            "answer arm identity is missing or loader/arm is not allowlisted",
        )
        source_binding = payload.get("source_binding")
        _require(isinstance(source_binding, Mapping), "EM source binding is missing")
        binding_without_digest = dict(source_binding)
        binding_digest = binding_without_digest.pop("binding_sha256", None)
        _require(binding_digest == identity_sha256(binding_without_digest) and source_binding.get("arm_label") == label, "EM source binding changed")

    parent_label = payload.get("parent_arm_label")
    parent_sha = payload.get("parent_run_sha256")
    if label == S0_ARM_LABEL:
        _require(isinstance(identity, Mapping) and identity.get("parent_arm") is None and parent_label is None and parent_sha is None, "S0_CONTROL must not have an arm parent")
    else:
        _require(isinstance(parent_label, str) and parent_label != label, "descendant omitted parent arm label")
        sealed_parent_sha = payload.get("s0_control_run_sha256") if legacy_descendant else None
        if parent_sha is None:
            parent_sha = sealed_parent_sha
        elif sealed_parent_sha is not None:
            _require(parent_sha == sealed_parent_sha, "descendant parent aliases disagree")
        parent_sha = _sha(parent_sha, "parent run SHA-256")

    normalized: list[_AnswerRow] = []
    for ordinal, row in enumerate(rows):
        prediction = row.get("prediction") if isinstance(row, Mapping) else None
        text = prediction.get("text") if isinstance(prediction, Mapping) else None
        prediction_sha = prediction.get("sha256") if isinstance(prediction, Mapping) else None
        _require(isinstance(text, str) and text and prediction_sha == quote_sha256(text), f"prediction changed at ordinal {ordinal}")
        question_id = row.get("question_id") if isinstance(row, Mapping) else None
        _require(row.get("ordinal") == ordinal and isinstance(question_id, str) and question_id, f"answer order changed at ordinal {ordinal}")
        parent_prediction = row.get("parent_prediction_sha256")
        if label == S0_ARM_LABEL:
            _require(parent_prediction is None, f"S0 row {ordinal} names a parent prediction")
        else:
            sealed_parent_prediction = row.get("s0_control_prediction_sha256") if legacy_descendant else None
            if parent_prediction is None:
                parent_prediction = sealed_parent_prediction
            elif sealed_parent_prediction is not None:
                _require(parent_prediction == sealed_parent_prediction, f"parent prediction aliases disagree at {ordinal}")
            parent_prediction = _sha(parent_prediction, f"parent prediction {ordinal}")
        binding = {
            "ordinal": ordinal,
            "question_id": question_id,
            "question_sha256": _sha(row.get("question_sha256"), f"question {ordinal}"),
            "dated_question_sha256": _sha(row.get("dated_question_sha256"), f"dated question {ordinal}"),
            "prediction_sha256": _sha(prediction_sha, f"prediction {ordinal}"),
            "parent_prediction_sha256": parent_prediction,
        }
        normalized.append(_AnswerRow(**binding, prediction=text, row_binding_sha256=identity_sha256(binding)))
    return _AnswerRun(
        payload=payload,
        sha256=sha256,
        replay_sha256=replay_sha256,
        arm_label=label,
        parent_arm_label=parent_label,
        parent_run_sha256=parent_sha,
        retrieval_sha256=_sha(payload.get("retrieval_sha256"), "retrieval SHA-256"),
        baseline_answers_sha256=_sha(payload.get("baseline_final_answers_sha256"), "baseline answer SHA-256"),
        population_identity_sha256=_sha(payload.get("population_identity_sha256"), "population identity SHA-256"),
        historical_validator_binding_sha256=_sha(payload.get("historical_validator_binding_sha256"), "historical validator binding"),
        rows=tuple(normalized),
        loader_spec=loader_spec,
    )


def _prepare_run_from_args(args: argparse.Namespace, *, prefix: str) -> _AnswerRun:
    path = Path(getattr(args, f"{prefix}_run"))
    expected = getattr(args, f"expected_{prefix}_run_sha256")
    _require(expected is not None, f"--expected-{prefix}-run-sha256 is required")
    source, source_sha = _read(path, str(expected))
    replay_path = Path(getattr(args, f"{prefix}_run_replay") or path.with_name("run-replay.json"))
    replay, replay_sha = _read(replay_path, str(expected))
    _require(canonical_json_bytes(source) == canonical_json_bytes(replay), "answer run/replay differ")
    label = source.get("arm_label")
    loader_spec = getattr(args, f"{prefix}_loader") or (DEFAULT_S0_LOADER if label == S0_ARM_LABEL else None)
    _require(isinstance(loader_spec, str) and _LOADER.fullmatch(loader_spec), f"{prefix} requires a strict loader")
    module, name = loader_spec.split(":", 1)
    loader = getattr(importlib.import_module(module), name, None)
    _require(callable(loader), f"invalid strict loader: {loader_spec}")
    checkpoint = getattr(args, f"{prefix}_checkpoint_dir")
    verified, verified_sha = loader(
        path,
        expected_run_sha256=str(expected),
        retrieval_path=Path(args.retrieval),
        baseline_answers_path=Path(args.baseline_answers),
        checkpoint_dir=None if checkpoint is None else Path(checkpoint),
        max_concurrency=int(args.max_concurrency),
        expected_question_count=int(args.expected_question_count),
        expected_retrieval_sha256=EXPECTED_RETRIEVAL_SHA256,
        expected_baseline_answers_sha256=EXPECTED_BASELINE_ANSWERS_SHA256,
    )
    _require(verified_sha == source_sha and canonical_json_bytes(verified) == canonical_json_bytes(source), "strict loader returned another run")
    return _normalize_answer_run(source, sha256=source_sha, replay_sha256=replay_sha, loader_spec=loader_spec, expected_question_count=int(args.expected_question_count))


def _baseline_answer_rows(path: Path, expected: str, candidate: _AnswerRun) -> tuple[tuple[_AnswerRow, ...], str]:
    artifact, digest = _read(path, expected)
    rows = artifact.get("questions")
    _require(artifact.get("gold_fields_present") is False and artifact.get("retrieval_sha256") == candidate.retrieval_sha256 and artifact.get("population_identity_sha256") == candidate.population_identity_sha256 and isinstance(rows, list) and len(rows) == len(candidate.rows), "baseline answer binding changed")
    result: list[_AnswerRow] = []
    for source, row in zip(candidate.rows, rows, strict=True):
        answer = row.get("answer") if isinstance(row, Mapping) else None
        text = answer.get("text") if isinstance(answer, Mapping) else None
        answer_sha = answer.get("sha256") if isinstance(answer, Mapping) else None
        _require(row.get("ordinal") == source.ordinal and row.get("question_id") == source.question_id and row.get("question_sha256") == source.question_sha256 and row.get("dated_question_sha256") == source.dated_question_sha256 and isinstance(text, str) and answer_sha == quote_sha256(text), f"baseline answer changed at {source.ordinal}")
        binding = {
            "ordinal": source.ordinal,
            "question_id": source.question_id,
            "question_sha256": source.question_sha256,
            "dated_question_sha256": source.dated_question_sha256,
            "prediction_sha256": answer_sha,
            "parent_prediction_sha256": None,
        }
        result.append(_AnswerRow(**binding, prediction=text, row_binding_sha256=identity_sha256(binding)))
    return tuple(result), digest


def _load_locked_gold(dataset: Path, split: Path, candidate: _AnswerRun) -> tuple[tuple[Any, ...], str]:
    samples, _shards, identity = build_locked_cumulative_population_identity(dataset, split, plan=LOCKED_LONGMEMEVAL_VALIDATION_PLAN)
    questions = tuple(question for sample in samples for question in sample.questions)
    _require(identity.get("population_identity_sha256") == candidate.population_identity_sha256 and len(questions) == len(candidate.rows), "gold population identity changed")
    projection = []
    for source, question in zip(candidate.rows, questions, strict=True):
        _require(question.question_id == source.question_id and quote_sha256(question.question) == source.question_sha256 and quote_sha256(question.dated_question) == source.dated_question_sha256, f"gold order changed at {source.ordinal}")
        projection.append({"ordinal": source.ordinal, "question_id": source.question_id, "question_sha256": source.question_sha256, "dated_question_sha256": source.dated_question_sha256, "gold_answer_sha256": quote_sha256(question.answer)})
    return questions, identity_sha256(projection)


def _read_topology_ledger(path: Path, expected: str, questions: Sequence[Any]) -> tuple[tuple[str, ...], str]:
    observed = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() and not path.is_symlink() else ""
    _require(observed == _sha(expected, "topology ledger SHA-256"), "topology ledger changed")
    sidecar = path.with_name(path.name + ".sha256")
    _require(sidecar.is_file() and not sidecar.is_symlink() and sidecar.read_bytes() == f"{observed}  {path.name}\n".encode("ascii"), "topology sidecar changed")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == len(questions), "topology population changed")
    result = []
    for ordinal, (row, question) in enumerate(zip(rows, questions, strict=True)):
        topology = row.get("retrieval_topology")
        _require(row.get("ordinal") == str(ordinal) and row.get("question_id") == question.question_id and row.get("question") == question.question and row.get("reference_answer") == question.answer and topology, f"topology binding changed at {ordinal}")
        result.append(str(topology))
    return tuple(result), observed


def _historical_baseline_outcomes(path: Path, expected: str, candidate: _AnswerRun, baseline: Sequence[_AnswerRow]) -> tuple[tuple[bool, ...], tuple[str, ...], str, None]:
    artifact, digest = _read(path, expected)
    rows = artifact.get("questions")
    _require(artifact.get("format") == "memory-condense-fixed-stage-final-answer-semantic-judge-score-v1" and artifact.get("final_answer_artifact_sha256") == candidate.baseline_answers_sha256 and artifact.get("retrieval_sha256") == candidate.retrieval_sha256 and artifact.get("population_identity_sha256") == candidate.population_identity_sha256 and isinstance(rows, list) and len(rows) == len(baseline), "historical judge binding changed")
    correct, hashes = [], []
    for source, row in zip(baseline, rows, strict=True):
        _require(row.get("ordinal") == source.ordinal and row.get("question_id") == source.question_id and row.get("prediction_sha256") == source.prediction_sha256 and type(row.get("correct")) is bool, f"historical verdict changed at {source.ordinal}")
        correct.append(bool(row["correct"])); hashes.append(identity_sha256(dict(row)))
    _require(artifact.get("aggregate", {}).get("correct") == sum(correct), "historical judge aggregate changed")
    return tuple(correct), tuple(hashes), digest, None


def _parent_outcomes(path: Path, replay_path: Path, expected: str, parent: _AnswerRun) -> tuple[tuple[bool, ...], tuple[str, ...], str, str]:
    artifact, digest = _read(path, expected)
    replay, replay_digest = _read(replay_path, expected)
    rows = artifact.get("questions")
    _require(canonical_json_bytes(artifact) == canonical_json_bytes(replay), "parent judge/replay differ")
    binding = artifact.get("campaign_binding")
    _require(artifact.get("format") == JUDGE_FORMAT and artifact.get("arm_label") == parent.arm_label and isinstance(binding, Mapping) and binding.get("arm_run_sha256") == parent.sha256 and isinstance(rows, list) and len(rows) == len(parent.rows), "parent judge binding changed")
    correct, hashes = [], []
    for source, row in zip(parent.rows, rows, strict=True):
        _require(row.get("ordinal") == source.ordinal and row.get("question_id") == source.question_id and row.get("prediction_sha256") == source.prediction_sha256 and type(row.get("correct")) is bool, f"parent verdict changed at {source.ordinal}")
        correct.append(bool(row["correct"])); hashes.append(identity_sha256(dict(row)))
    _require(artifact.get("aggregate", {}).get("candidate_correct") == sum(correct), "parent judge aggregate changed")
    return tuple(correct), tuple(hashes), digest, replay_digest


def _build_plan(args: argparse.Namespace) -> _Plan:
    _require(args.gateway_url == DEFAULT_GATEWAY_URL and args.sol_model == DEFAULT_SOL_MODEL, "judge requires the locked Sol route")
    _require(type(args.expected_question_count) is int and args.expected_question_count > 0 and type(args.max_concurrency) is int and args.max_concurrency > 0, "invalid question/concurrency count")
    _require(args.dataset is not None, "judge requires --dataset")

    # Strict answer replay precedes every read of benchmark gold.
    candidate = _prepare_run_from_args(args, prefix="arm")
    _require(candidate.retrieval_sha256 == EXPECTED_RETRIEVAL_SHA256 and candidate.baseline_answers_sha256 == EXPECTED_BASELINE_ANSWERS_SHA256, "candidate is not locked")
    parent: _AnswerRun | None = None
    if candidate.arm_label == S0_ARM_LABEL:
        baseline, baseline_sha = _baseline_answer_rows(Path(args.baseline_answers), EXPECTED_BASELINE_ANSWERS_SHA256, candidate)
        _require(baseline_sha == candidate.baseline_answers_sha256, "baseline digest changed")
        baseline_label, baseline_run_sha = "FIXED_S1_EXTERNAL_ANCHOR", None
    else:
        _require(args.parent_run is not None, "descendant requires --parent-run")
        parent = _prepare_run_from_args(args, prefix="parent")
        _require(candidate.parent_arm_label == parent.arm_label and candidate.parent_run_sha256 == parent.sha256 and candidate.retrieval_sha256 == parent.retrieval_sha256 and candidate.population_identity_sha256 == parent.population_identity_sha256, "candidate parent binding changed")
        for child, prior in zip(candidate.rows, parent.rows, strict=True):
            _require((child.ordinal, child.question_id, child.question_sha256, child.dated_question_sha256, child.parent_prediction_sha256) == (prior.ordinal, prior.question_id, prior.question_sha256, prior.dated_question_sha256, prior.prediction_sha256), f"parent prediction changed at {child.ordinal}")
        baseline, baseline_label, baseline_run_sha = parent.rows, parent.arm_label, parent.sha256

    questions, gold_sha = _load_locked_gold(Path(args.dataset), Path(args.split), candidate)
    all_rows, judged = [], []
    for source, prior, question in zip(candidate.rows, baseline, questions, strict=True):
        changed = source.prediction_sha256 != prior.prediction_sha256
        judge_now = candidate.arm_label == S0_ARM_LABEL or changed
        route = route_question(question.dated_question)
        messages = tuple(build_judge_prompt(question.question, question.answer, source.prediction)) if judge_now else None
        row = _PromptRow(
            ordinal=source.ordinal,
            question_id=source.question_id,
            question_sha256=source.question_sha256,
            dated_question_sha256=source.dated_question_sha256,
            gold_answer_sha256=quote_sha256(question.answer),
            prediction_sha256=source.prediction_sha256,
            baseline_prediction_sha256=prior.prediction_sha256,
            changed_from_baseline=changed,
            demand_class=route.style.value,
            demand_receipt_sha256=route.receipt_sha256,
            messages=messages,
            messages_sha256=None if messages is None else identity_sha256(list(messages)),
        )
        all_rows.append(row)
        if judge_now:
            judged.append(row)
    preflight = None if not judged else preflight_fast_completion_prompts([row.messages for row in judged if row.messages is not None], max_prompt_tokens=MAX_JUDGE_PROMPT_TOKENS)
    _require(preflight is None or tuple(row.messages_sha256 for row in judged) == tuple(row.messages_sha256 for row in preflight.ordered_rows), "prompt order changed")
    prompt_seal = identity_sha256([{"ordinal": row.ordinal, "question_id": row.question_id, "prediction_sha256": row.prediction_sha256, "gold_answer_sha256": row.gold_answer_sha256, "messages_sha256": row.messages_sha256} for row in judged])

    # Oracle labels/outcomes cannot influence the already sealed prompt set.
    topologies, topology_sha = _read_topology_ledger(Path(args.topology_ledger), str(args.expected_topology_ledger_sha256), questions)
    if parent is None:
        outcomes = _historical_baseline_outcomes(Path(args.baseline_judge), str(args.expected_baseline_judge_sha256), candidate, baseline)
    else:
        _require(args.parent_judge is not None and args.expected_parent_judge_sha256 is not None, "descendant requires parent judge")
        parent_judge = Path(args.parent_judge)
        outcomes = _parent_outcomes(parent_judge, Path(args.parent_judge_replay or parent_judge.with_name("semantic-judge-sol-replay.json")), str(args.expected_parent_judge_sha256), parent)
    correct, verdict_hashes, judge_sha, judge_replay_sha = outcomes
    order_sha = identity_sha256([{"ordinal": row.ordinal, "question_id": row.question_id, "question_sha256": row.question_sha256, "dated_question_sha256": row.dated_question_sha256, "prediction_sha256": row.prediction_sha256, "baseline_prediction_sha256": row.baseline_prediction_sha256, "demand_receipt_sha256": row.demand_receipt_sha256, "topology": topology} for row, topology in zip(all_rows, topologies, strict=True)])
    return _Plan(candidate, baseline_label, baseline_run_sha, judge_sha, judge_replay_sha, correct, verdict_hashes, tuple(all_rows), tuple(judged), preflight, gold_sha, topology_sha, topologies, order_sha, prompt_seal)


def _campaign_binding(plan: _Plan) -> dict[str, Any]:
    return {
        "format": BINDING_FORMAT,
        "arm_label": plan.candidate.arm_label,
        "arm_run_sha256": plan.candidate.sha256,
        "arm_run_replay_sha256": plan.candidate.replay_sha256,
        "arm_loader": plan.candidate.loader_spec,
        "baseline_arm_label": plan.baseline_label,
        "baseline_run_sha256": plan.baseline_run_sha256,
        "baseline_judge_sha256": plan.baseline_judge_sha256,
        "baseline_judge_replay_sha256": plan.baseline_judge_replay_sha256,
        "retrieval_sha256": plan.candidate.retrieval_sha256,
        "baseline_final_answers_sha256": plan.candidate.baseline_answers_sha256,
        "population_identity_sha256": plan.candidate.population_identity_sha256,
        "historical_validator_binding_sha256": plan.candidate.historical_validator_binding_sha256,
        "gold_population_sha256": plan.gold_population_sha256,
        "topology_ledger_sha256": plan.topology_ledger_sha256,
        "ordered_question_binding_sha256": plan.question_order_sha256,
        "sealed_judge_prompt_projection_sha256": plan.prompt_seal_sha256,
        "judge_prompt_population_sha256": None if plan.preflight is None else plan.preflight.prompt_population_sha256,
        "question_count": len(plan.prompt_rows),
        "changed_prediction_count": sum(row.changed_from_baseline for row in plan.prompt_rows),
        "logical_judgment_count": len(plan.judged_rows),
        "unique_judge_call_count": plan.unique_calls,
        "s0_control_all_question_judging": plan.candidate.arm_label == S0_ARM_LABEL,
        "descendant_unchanged_verdict_reuse": plan.candidate.arm_label != S0_ARM_LABEL,
        "question_only_demand_classifier_format": ROUTED_REPAIR_ROUTING_FORMAT,
        "topology_loaded_after_prompt_population_seal": True,
        "arm_or_topology_labels_exposed_to_judge": False,
        "gateway_url": DEFAULT_GATEWAY_URL,
        "gateway_model": DEFAULT_SOL_MODEL,
        "caller_model": DEFAULT_CALLER_MODEL,
        "max_prompt_tokens": MAX_JUDGE_PROMPT_TOKENS,
        "max_new_tokens": JUDGE_MAX_TOKENS,
        "retries": 0,
    }


def _runtime(plan: _Plan, args: argparse.Namespace, client: Any | None) -> FastCompletionRuntime:
    _require(plan.judged_rows, "zero-call plan has no runtime")
    return FastCompletionRuntime(
        checkpoint_dir=Path(args.output_root) / "sol-judge-calls",
        prompt_population=[row.messages for row in plan.judged_rows if row.messages is not None],
        model=DEFAULT_SOL_MODEL,
        client=client,
        max_prompt_tokens=MAX_JUDGE_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=int(args.max_concurrency),
        retries=0,
        benchmark_provenance={
            "experiment_format": JUDGE_FORMAT,
            "campaign_binding_sha256": identity_sha256(_campaign_binding(plan)),
            "arm_run_sha256": plan.candidate.sha256,
            "authorized_unique_calls": plan.unique_calls,
            "arm_or_topology_labels_exposed_to_judge": False,
            "gold_present_only_in_exact_judge_prompt": True,
        },
    )


def _stable_batch(batch: FastCompletionBatch) -> dict[str, Any]:
    value = batch.model_dump()
    return {
        "logical_completions": value["logical_completions"],
        "unique_records": [{key: child for key, child in row.items() if key not in {"checkpoint_hit", "physical_call"}} for row in value["unique_records"]],
        "usage": {key: child for key, child in value["usage"].items() if key not in {"checkpoint_hits", "physical_calls"}},
        "provenance": value["provenance"],
        "runtime_identity_sha256": value["runtime_identity_sha256"],
        "prompt_population": value["prompt_population"],
    }


def _slice(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    result = []
    for values, selected in sorted(groups.items()):
        baseline = sum(bool(row["baseline_correct"]) for row in selected)
        candidate = sum(bool(row["correct"]) for row in selected)
        rescued = sum(bool(row["rescued"]) for row in selected)
        regressed = sum(bool(row["regressed"]) for row in selected)
        result.append({
            **dict(zip(keys, values, strict=True)),
            "questions": len(selected),
            "changed_predictions": sum(bool(row["changed_from_baseline"]) for row in selected),
            "new_sol_judgments": sum(row["verdict_source"] == "new_sol_judge" for row in selected),
            "reused_baseline_verdicts": sum(row["verdict_source"] == "sealed_baseline_judge" for row in selected),
            "baseline_correct": baseline,
            "candidate_correct": candidate,
            "candidate_accuracy": candidate / len(selected),
            "rescued": rescued,
            "regressed": regressed,
            "net_marginal": rescued - regressed,
        })
    return result


def _judge_artifact(plan: _Plan, batch: FastCompletionBatch | None) -> dict[str, Any]:
    _require((batch is None) == (not plan.judged_rows), "judge batch population changed")
    records = {} if batch is None else {row.messages_sha256: row for row in batch.unique_records}
    completions = () if batch is None else batch.logical_completions
    fresh: dict[int, dict[str, Any]] = {}
    for row, verdict in zip(plan.judged_rows, completions, strict=True):
        record = records[row.messages_sha256]
        fresh[row.ordinal] = {
            "correct": parse_binary_judge_verdict(verdict),
            "messages": row.messages_sha256,
            "call": record.call_key_sha256,
            "request": record.request_journal_sha256,
            "response": record.response_journal_sha256,
            "verdict": quote_sha256(verdict),
        }
    rows = []
    for source, topology, baseline_correct, baseline_row_sha in zip(plan.prompt_rows, plan.topologies, plan.baseline_correct, plan.baseline_judge_row_sha256s, strict=True):
        outcome = fresh.get(source.ordinal)
        correct = baseline_correct if outcome is None else bool(outcome["correct"])
        rows.append({
            "ordinal": source.ordinal,
            "question_id": source.question_id,
            "question_sha256": source.question_sha256,
            "dated_question_sha256": source.dated_question_sha256,
            "gold_answer_sha256": source.gold_answer_sha256,
            "question_only_demand_class": source.demand_class,
            "question_only_demand_receipt_sha256": source.demand_receipt_sha256,
            "evidence_topology_class": topology,
            "prediction_sha256": source.prediction_sha256,
            "baseline_prediction_sha256": source.baseline_prediction_sha256,
            "changed_from_baseline": source.changed_from_baseline,
            "verdict_source": "sealed_baseline_judge" if outcome is None else "new_sol_judge",
            "baseline_judge_row_sha256": baseline_row_sha,
            "baseline_correct": baseline_correct,
            "correct": correct,
            "rescued": correct and not baseline_correct,
            "regressed": baseline_correct and not correct,
            "judge_messages_sha256": None if outcome is None else outcome["messages"],
            "judge_call_key_sha256": None if outcome is None else outcome["call"],
            "judge_request_journal_sha256": None if outcome is None else outcome["request"],
            "judge_response_journal_sha256": None if outcome is None else outcome["response"],
            "judge_verdict_sha256": None if outcome is None else outcome["verdict"],
        })
    baseline = sum(bool(row["baseline_correct"]) for row in rows)
    candidate = sum(bool(row["correct"]) for row in rows)
    rescued = sum(bool(row["rescued"]) for row in rows)
    regressed = sum(bool(row["regressed"]) for row in rows)
    return {
        "format": JUDGE_FORMAT,
        "arm_label": plan.candidate.arm_label,
        "campaign_binding": _campaign_binding(plan),
        "completion_batch": None if batch is None else _stable_batch(batch),
        "question_count": len(rows),
        "logical_judgment_count": len(plan.judged_rows),
        "unique_sol_completion_count": plan.unique_calls,
        "aggregate": {
            "baseline_correct": baseline,
            "candidate_correct": candidate,
            "rescued": rescued,
            "regressed": regressed,
            "net_marginal": rescued - regressed,
            "accepted_for_positive_only_composition": rescued - regressed > 0,
        },
        "paired_slices": {
            "by_question_only_demand_class": _slice(rows, ("question_only_demand_class",)),
            "by_evidence_topology_class": _slice(rows, ("evidence_topology_class",)),
            "by_demand_x_topology": _slice(rows, ("question_only_demand_class", "evidence_topology_class")),
        },
        "questions": rows,
        "gold_loaded_only_after_answer_run_replay": True,
        "topology_loaded_only_after_judge_prompt_seal": True,
        "explicit_gold_answer_text_persisted": False,
        "judge_completions_may_echo_gold": True,
        "arm_or_topology_labels_exposed_to_judge": False,
        "unchanged_verdicts_reused_from_sealed_baseline": plan.candidate.arm_label != S0_ARM_LABEL,
        "retained_request_token_state_bytes": 0,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _require(not args.enable_provider and args.authorized_provider_calls == 0, "preflight forbids provider authorization")
    plan = _build_plan(args)
    return {
        "format": PREFLIGHT_FORMAT,
        "campaign_binding": _campaign_binding(plan),
        "logical_prompt_count": len(plan.judged_rows),
        "unique_prompt_count": plan.unique_calls,
        "required_authorized_provider_calls": plan.unique_calls,
        "maximum_prompt_token_proxy": 0 if plan.preflight is None else plan.preflight.max_prompt_token_proxy,
        "provider_calls": 0,
        "writes": 0,
        "gold_loaded_only_after_answer_run_replay": True,
        "topology_loaded_only_after_judge_prompt_seal": True,
        "arm_or_topology_labels_exposed_to_judge": False,
    }


def _judge_path(args: argparse.Namespace) -> Path:
    return Path(args.judge_artifact or Path(args.output_root) / "semantic-judge-sol.json")


def run_judge(args: argparse.Namespace) -> tuple[dict[str, Any], str, int]:
    plan = _build_plan(args)
    _require(args.enable_provider, "run requires --enable-provider")
    _require(args.authorized_provider_calls == plan.unique_calls, f"--authorized-provider-calls must exactly equal {plan.unique_calls}")
    target = _judge_path(args)
    if target.exists():
        existing, _ = em_runner._read(target)
        _require(existing.get("campaign_binding") == _campaign_binding(plan), "existing judge belongs to another campaign")
    batch = None
    if plan.judged_rows:
        load_dotenv()
        api_key = os.environ.get(str(args.api_key_env), "").strip()
        _require(api_key, f"provider API key is empty: {args.api_key_env}")
        client = _make_provider_client(api_key, DEFAULT_GATEWAY_URL)
        try:
            batch = _runtime(plan, args, client).run()
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        _require(batch.usage.physical_calls + batch.usage.checkpoint_hits == plan.unique_calls, "Sol journal population changed")
    artifact = _judge_artifact(plan, batch)
    return artifact, _publish(target, artifact), 0 if batch is None else batch.usage.physical_calls


def run_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    _require(not args.enable_provider and args.authorized_provider_calls == 0, "replay forbids provider authorization")
    plan = _build_plan(args)
    source, _ = em_runner._read(_judge_path(args))
    batch = None if not plan.judged_rows else _runtime(plan, args, None).run()
    _require(batch is None or batch.usage.physical_calls == 0, "Sol replay made provider calls")
    _require(canonical_json_bytes(source) == canonical_json_bytes(_judge_artifact(plan, batch)), "judge differs from journals")
    replay = Path(args.judge_replay or Path(args.output_root) / "semantic-judge-sol-replay.json")
    return source, _publish(replay, source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "run", "replay"))
    parser.add_argument("--arm-run", type=Path, default=DEFAULT_ARM_ROOT / "run.json")
    parser.add_argument("--arm-run-replay", type=Path)
    parser.add_argument("--expected-arm-run-sha256")
    parser.add_argument("--arm-loader")
    parser.add_argument("--arm-checkpoint-dir", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--parent-run-replay", type=Path)
    parser.add_argument("--expected-parent-run-sha256")
    parser.add_argument("--parent-loader")
    parser.add_argument("--parent-checkpoint-dir", type=Path)
    parser.add_argument("--parent-judge", type=Path)
    parser.add_argument("--parent-judge-replay", type=Path)
    parser.add_argument("--expected-parent-judge-sha256")
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--baseline-answers", type=Path, default=DEFAULT_BASELINE_ANSWERS)
    parser.add_argument("--baseline-judge", type=Path, default=DEFAULT_BASELINE_JUDGE)
    parser.add_argument("--expected-baseline-judge-sha256", default=EXPECTED_BASELINE_JUDGE_SHA256)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--topology-ledger", type=Path, default=DEFAULT_TOPOLOGY_LEDGER)
    parser.add_argument("--expected-topology-ledger-sha256", default=EXPECTED_TOPOLOGY_LEDGER_SHA256)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ARM_ROOT)
    parser.add_argument("--judge-artifact", type=Path)
    parser.add_argument("--judge-replay", type=Path)
    parser.add_argument("--expected-question-count", type=int, default=EXPECTED_QUESTION_COUNT)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--sol-model", default=DEFAULT_SOL_MODEL)
    parser.add_argument("--api-key-env", default="LITELLM_KEY")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "preflight":
        result = run_preflight(args)
        print(f"Sol preflight: arm={result['campaign_binding']['arm_label']}; logical={result['logical_prompt_count']}; unique={result['unique_prompt_count']}; calls=0; writes=0")
        return 0
    if args.phase == "run":
        result, digest, physical = run_judge(args)
        print(f"Sol judge {digest}: {result['aggregate']['candidate_correct']}/{result['question_count']}; physical={physical}")
        return 0
    result, digest = run_replay(args)
    print(f"Sol replay {digest}: {result['aggregate']['candidate_correct']}/{result['question_count']}; physical=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["JUDGE_FORMAT", "PREFLIGHT_FORMAT", "build_parser", "main", "run_judge", "run_preflight", "run_replay"]
