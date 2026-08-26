"""Explicit gold-bearing augmentation for :mod:`linear_case_ledger`."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.eval.linear_case_ledger import (
    CAV_SCORE_FORMAT,
    FIXED_JUDGE_FORMAT,
    GOLD_BLIND_LEDGER_FORMAT,
    HEBBIAN_SCORE_FORMAT,
    POSTHOC_LEDGER_FORMAT,
    RETRIEVAL_SCORE_FORMAT,
    SYNTHESIS_SCORE_FORMAT,
    ArtifactSpec,
    CaseLedgerValidationError,
    _identity_sha256,
    _read_sealed_json,
    _require_digest,
    _require_list,
    _require_mapping,
    _require_string,
    _seal_ledger,
    _stage_alias,
)

_SCORE_FORMATS = {
    RETRIEVAL_SCORE_FORMAT,
    SYNTHESIS_SCORE_FORMAT,
    FIXED_JUDGE_FORMAT,
    CAV_SCORE_FORMAT,
    HEBBIAN_SCORE_FORMAT,
}


def _verify_gold_blind_ledger(ledger: Mapping[str, Any]) -> None:
    if ledger.get("format") != GOLD_BLIND_LEDGER_FORMAT:
        raise CaseLedgerValidationError("unsupported gold-blind ledger format")
    if ledger.get("gold_fields_present") is not False or ledger.get(
        "gold_artifacts_read"
    ) != 0:
        raise CaseLedgerValidationError("gold-blind ledger boundary changed")
    digest = _require_digest(ledger.get("ledger_sha256"), "ledger_sha256")
    unsigned = dict(ledger)
    del unsigned["ledger_sha256"]
    if _identity_sha256(unsigned) != digest:
        raise CaseLedgerValidationError("gold-blind ledger SHA-256 does not verify")


def _observation_lookup(
    ledger: Mapping[str, Any],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for raw in _require_list(ledger.get("observations"), "ledger.observations"):
        row = _require_mapping(raw, "ledger observation")
        key = (row["run_id"], row["question_id"], row["stage"], row["arm_id"])
        if key in result:
            raise CaseLedgerValidationError("ledger observation key is duplicated")
        result[key] = row
    return result


def _merge_metrics(
    observation: dict[str, Any],
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    posthoc = observation.setdefault("posthoc", {"sources": [], "metrics": {}})
    metric_target = posthoc["metrics"]
    for key, value in metrics.items():
        if key in metric_target and metric_target[key] != value:
            raise CaseLedgerValidationError(
                f"conflicting post-hoc metric {key!r} for "
                f"{observation['observation_id']}"
            )
        metric_target[key] = value
    source_id = source["score_artifact_sha256"]
    if source_id not in posthoc["sources"]:
        posthoc["sources"].append(source_id)


def _score_binding_field(score_format: str) -> str:
    if score_format == RETRIEVAL_SCORE_FORMAT:
        return "retrieval_artifact_sha256"
    if score_format == SYNTHESIS_SCORE_FORMAT:
        return "synthesis_sha256"
    if score_format == FIXED_JUDGE_FORMAT:
        return "final_answer_artifact_sha256"
    return "answer_manifest_sha256"


def _score_source(
    spec: ArtifactSpec,
    root: Mapping[str, Any],
    digest: str,
    path: Path,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    score_format = _require_string(root.get("format"), "score format")
    expected = (
        run["retrieval_sha256"]
        if score_format == RETRIEVAL_SCORE_FORMAT
        else run["artifact_sha256"]
    )
    if root.get(_score_binding_field(score_format)) != expected:
        raise CaseLedgerValidationError(
            f"post-hoc score does not bind run {spec.run_id!r}"
        )
    if root.get("question_count") != len(
        {
            row.get("question_id")
            for row in root.get("questions", root.get("rows", []))
            if type(row) is dict
        }
    ) and score_format not in {CAV_SCORE_FORMAT, HEBBIAN_SCORE_FORMAT}:
        raise CaseLedgerValidationError("post-hoc score question count changed")
    return {
        "run_id": spec.run_id,
        "artifact_format": score_format,
        "artifact_path": str(path),
        "score_artifact_sha256": digest,
        "answer_or_retrieval_artifact_sha256": expected,
        "gold_loaded_posthoc": True,
    }


def _metric_subset(row: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    return {name: row[name] for name in names if name in row}


def _augment_retrieval_score(
    spec: ArtifactSpec,
    root: Mapping[str, Any],
    source: Mapping[str, Any],
    lookup: Mapping[tuple[str, str, str, str], dict[str, Any]],
    question_order: Sequence[str],
) -> None:
    questions = _require_list(root.get("questions"), "retrieval score questions")
    if [row.get("question_id") for row in questions] != list(question_order):
        raise CaseLedgerValidationError("retrieval score question order changed")
    for question in questions:
        for stage in _require_list(question.get("stages"), "retrieval score stages"):
            alias = _stage_alias(stage.get("stage_id"), "retrieval score stage")
            observation = lookup.get(
                (spec.run_id, question["question_id"], alias, "retrieval")
            )
            if observation is None:
                raise CaseLedgerValidationError("retrieval score row has no observation")
            _merge_metrics(
                observation,
                _metric_subset(
                    stage,
                    (
                        "answer_present",
                        "best_evidence_f1",
                        "evidence_source_recall",
                        "all_evidence_sources",
                        "any_evidence_source",
                        "answer_value_component_recall",
                    ),
                ),
                source,
            )


def _augment_synthesis_score(
    spec: ArtifactSpec,
    root: Mapping[str, Any],
    source: Mapping[str, Any],
    lookup: Mapping[tuple[str, str, str, str], dict[str, Any]],
    question_order: Sequence[str],
) -> None:
    questions = _require_list(root.get("questions"), "synthesis score questions")
    if [row.get("question_id") for row in questions] != list(question_order):
        raise CaseLedgerValidationError("synthesis score question order changed")
    for question in questions:
        for stage in _require_list(question.get("stages"), "synthesis score stages"):
            alias = _stage_alias(stage.get("stage_id"), "synthesis score stage")
            observation = lookup.get(
                (spec.run_id, question["question_id"], alias, "answer")
            )
            if observation is None:
                raise CaseLedgerValidationError("synthesis score row has no observation")
            _merge_metrics(
                observation,
                _metric_subset(
                    stage,
                    (
                        "exact_match",
                        "f1",
                        "answer_sha256",
                        "answer_value_component_recall",
                        "claim_value_component_recall",
                        "cited_expected_source",
                    ),
                ),
                source,
            )


def _augment_fast_score(
    spec: ArtifactSpec,
    root: Mapping[str, Any],
    source: Mapping[str, Any],
    lookup: Mapping[tuple[str, str, str, str], dict[str, Any]],
) -> None:
    for row in _require_list(root.get("rows"), "fast score rows"):
        alias = _stage_alias(row.get("stage_id"), "fast score stage")
        observation = lookup.get(
            (spec.run_id, row.get("question_id"), alias, row.get("arm_id"))
        )
        if observation is None:
            raise CaseLedgerValidationError("fast score row has no observation")
        _merge_metrics(
            observation,
            _metric_subset(
                row,
                ("exact_match", "f1", "category", "gold_answer_sha256"),
            ),
            source,
        )


def _augment_judge_score(
    spec: ArtifactSpec,
    root: Mapping[str, Any],
    source: Mapping[str, Any],
    lookup: Mapping[tuple[str, str, str, str], dict[str, Any]],
    question_order: Sequence[str],
) -> None:
    questions = _require_list(root.get("questions"), "judge questions")
    if [row.get("question_id") for row in questions] != list(question_order):
        raise CaseLedgerValidationError("judge question order changed")
    fixed_stage = _stage_alias(root.get("fixed_stage_id"), "judge fixed stage")
    for row in questions:
        observation = lookup.get(
            (spec.run_id, row.get("question_id"), fixed_stage, "answer")
        )
        if observation is None:
            raise CaseLedgerValidationError("judge row has no answer observation")
        _merge_metrics(
            observation,
            {
                "semantic_correct": row.get("correct"),
                **_metric_subset(row, ("category", "gold_answer_sha256")),
            },
            source,
        )


def _augment_score(
    *,
    spec: ArtifactSpec,
    root: Mapping[str, Any],
    source: Mapping[str, Any],
    lookup: Mapping[tuple[str, str, str, str], dict[str, Any]],
    question_order: Sequence[str],
) -> None:
    score_format = root["format"]
    if score_format == RETRIEVAL_SCORE_FORMAT:
        _augment_retrieval_score(spec, root, source, lookup, question_order)
    elif score_format == SYNTHESIS_SCORE_FORMAT:
        _augment_synthesis_score(spec, root, source, lookup, question_order)
    elif score_format in {CAV_SCORE_FORMAT, HEBBIAN_SCORE_FORMAT}:
        _augment_fast_score(spec, root, source, lookup)
    else:
        _augment_judge_score(spec, root, source, lookup, question_order)


def _numeric_change(left: object, right: object) -> str | None:
    if type(left) not in {int, float} or type(right) not in {int, float}:
        return None
    left_value = float(left)
    right_value = float(right)
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        raise CaseLedgerValidationError("post-hoc metrics must remain finite")
    if math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=1e-12):
        return "preserve"
    return "improve" if right_value > left_value else "regress"


def _metric_outcomes(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, str]:
    left_metrics = left.get("posthoc", {}).get("metrics", {})
    right_metrics = right.get("posthoc", {}).get("metrics", {})
    outcomes: dict[str, str] = {}
    for name in (
        "exact_match",
        "semantic_correct",
        "answer_present",
        "all_evidence_sources",
        "any_evidence_source",
        "cited_expected_source",
    ):
        if name in left_metrics and name in right_metrics:
            left_value = left_metrics[name]
            right_value = right_metrics[name]
            if type(left_value) is bool and type(right_value) is bool:
                if not left_value and right_value:
                    outcomes[name] = "fix"
                elif left_value and not right_value:
                    outcomes[name] = "regress"
                else:
                    outcomes[name] = "preserve"
    for name in (
        "f1",
        "best_evidence_f1",
        "evidence_source_recall",
        "answer_value_component_recall",
        "claim_value_component_recall",
    ):
        if name in left_metrics and name in right_metrics:
            change = _numeric_change(left_metrics[name], right_metrics[name])
            if change is not None:
                outcomes[name] = change
    return outcomes


def _summarize_outcomes(outcomes: Mapping[str, str]) -> str | None:
    values = set(outcomes.values())
    changed = values - {"preserve"}
    if not changed:
        return "preserve" if values else None
    if "regress" in changed and changed != {"regress"}:
        return "mixed"
    if "fix" in changed:
        return "fix"
    if "regress" in changed:
        return "regress"
    return "improve"


def augment_case_ledger_posthoc(
    ledger: Mapping[str, Any],
    score_artifacts: Sequence[ArtifactSpec],
) -> dict[str, Any]:
    """Explicitly attach gold-derived metrics to a sealed gold-blind ledger."""

    _verify_gold_blind_ledger(ledger)
    result = copy.deepcopy(dict(ledger))
    del result["ledger_sha256"]
    result["format"] = POSTHOC_LEDGER_FORMAT
    result["gold_fields_present"] = True
    result["gold_loaded_posthoc"] = True
    result["gold_blind_ledger_sha256"] = ledger["ledger_sha256"]
    result["gold_artifacts_read"] = len(score_artifacts)
    runs = {
        run["run_id"]: run for run in _require_list(result.get("runs"), "ledger.runs")
    }
    lookup = _observation_lookup(result)
    question_order = [
        row["question_id"]
        for row in _require_list(result.get("questions"), "ledger.questions")
    ]
    sources: list[dict[str, Any]] = []
    for spec in score_artifacts:
        run = runs.get(spec.run_id)
        if run is None:
            raise CaseLedgerValidationError(
                f"score run_id {spec.run_id!r} is absent from the gold-blind ledger"
            )
        root, digest, path = _read_sealed_json(spec)
        score_format = root.get("format")
        if score_format not in _SCORE_FORMATS:
            raise CaseLedgerValidationError(
                f"unsupported post-hoc score format: {score_format!r}"
            )
        source = _score_source(spec, root, digest, path, run)
        _augment_score(
            spec=spec,
            root=root,
            source=source,
            lookup=lookup,
            question_order=question_order,
        )
        sources.append(source)
    by_id = {row["observation_id"]: row for row in result["observations"]}
    for comparison in result["comparisons"]:
        left = by_id[comparison["left_observation_id"]]
        right = by_id[comparison["right_observation_id"]]
        metric_outcomes = _metric_outcomes(left, right)
        descriptive = _summarize_outcomes(metric_outcomes)
        if comparison["causal_comparable"]:
            comparison["posthoc_outcome"] = descriptive or "unavailable"
            comparison["posthoc_metric_outcomes"] = metric_outcomes
        else:
            comparison["posthoc_outcome"] = "not_comparable"
            if descriptive is not None:
                comparison["descriptive_outcome"] = descriptive
                comparison["descriptive_metric_outcomes"] = metric_outcomes
    result["score_artifacts"] = sources
    return _seal_ledger(result, "posthoc_ledger_sha256")


__all__ = ["augment_case_ledger_posthoc"]
