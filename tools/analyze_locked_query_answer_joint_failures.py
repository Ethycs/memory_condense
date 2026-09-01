#!/usr/bin/env python3
"""Sealed posthoc taxonomy for query-answer failures shared by two arms.

The answer, runtime, judge, score, and guided-audit checkpoints are verified
before the locked benchmark is opened.  The program is evaluation-only: it
does not import a provider client, rerun retrieval, or mutate answer artifacts.
Posthoc failure labels may use gold.  The separately emitted deployment policy
is keyed only by the already sealed question-only route.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    import sys

    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.locked_split import load_split_manifest, select_locked_split
from memory_condense.eval.recall_guarded_cumulative_population import (
    LOCKED_LONGMEMEVAL_DATASET_SHA256,
    LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
)
from memory_condense.ingest.loader import load_benchmark

from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import identity_sha256, require_sha256
from tools.matched_eval.ledger import _validated_runtime_ledger


ANALYSIS_FORMAT = "memory-condense-query-answer-joint-failure-taxonomy-v1"
ANALYSIS_NAME = "locked-query-answer-joint-failure-taxonomy-v1.json"
EXPECTED_QUESTION_COUNT = 100

DEFAULT_SPINE_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
    "matched-eval-spine-v2"
)
DEFAULT_PARENT_ROOT = DEFAULT_SPINE_ROOT / "s0-control-v2"
DEFAULT_PAYLOAD_ROOT = DEFAULT_SPINE_ROOT / "s0-plus-query-payload-v1"
DEFAULT_FACT_ROOT = (
    DEFAULT_SPINE_ROOT / "s0-plus-query-expansion-routed-fact-answers-v1"
)
DEFAULT_GUIDED_ROOT = DEFAULT_SPINE_ROOT / "s0-plus-query-guided-scan-v1"
DEFAULT_TARGET_PLAN = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
    "target-owner-plan-v1/target-plan.json"
)
DEFAULT_DATASET = Path(
    "C:/Users/Keytone/Downloads/memory-condense-rig/datasets/"
    "longmemeval_s_cleaned.json"
)
DEFAULT_SPLIT = Path("docs/10 - Research Log/data/longmemeval-95-target-split-v2.json")
DEFAULT_OUTPUT = Path("docs/10 - Research Log/data") / ANALYSIS_NAME

EXPECTED_TARGET_PLAN_SHA256 = (
    "b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff"
)
EXPECTED_GUIDED_AUDIT_SHA256 = (
    "329c8490ca2f090fa81c85cbc9999c07f539cc564c84bbaa590300d5f9c4ca34"
)
EXPECTED_GUIDED_RUN_SHA256 = (
    "a544ae9e6e554fcfc9cfc6167018f06b573fcf6546c9c3f3a6e3feda6ed821ff"
)
EXPECTED_GUIDED_LEDGER_SHA256 = (
    "b0edd491ddca674c24728f31cda337226090624db04c63a507eb6188eb802af7"
)

JOINT_FAILURE_ORDINALS = (
    6,
    7,
    14,
    16,
    27,
    28,
    31,
    36,
    37,
    40,
    42,
    43,
    52,
    53,
    54,
    61,
    65,
    67,
    69,
    75,
    77,
    79,
    81,
    82,
    86,
    93,
    94,
    97,
)

CAUSE_ORDER = (
    "source_missing",
    "candidate_reached_but_packing_dropped",
    "partial_multi_source_coverage",
    "operator_failure_despite_source_coverage",
    "answer_shape_or_judge_ambiguity",
    "other",
)

# Gold-informed labels.  Structural assertions below prevent these labels from
# surviving a changed evidence lifecycle unnoticed.
CAUSE_BY_ORDINAL = {
    36: "source_missing",
    37: "source_missing",
    54: "candidate_reached_but_packing_dropped",
    93: "candidate_reached_but_packing_dropped",
    7: "partial_multi_source_coverage",
    31: "partial_multi_source_coverage",
    61: "partial_multi_source_coverage",
    77: "partial_multi_source_coverage",
    86: "partial_multi_source_coverage",
    53: "answer_shape_or_judge_ambiguity",
    75: "answer_shape_or_judge_ambiguity",
    42: "other",
}
for _ordinal in JOINT_FAILURE_ORDINALS:
    CAUSE_BY_ORDINAL.setdefault(
        _ordinal, "operator_failure_despite_source_coverage"
    )

SECONDARY_CAUSES = {
    7: ("candidate_reached_but_packing_dropped",),
    53: ("partial_multi_source_coverage",),
    61: ("source_missing",),
    77: ("candidate_reached_but_packing_dropped",),
}

QUESTION_ONLY_POLICY = {
    "numeric_reduce": {
        "mechanisms": [
            "exhaustive_retrieval",
            "source_balanced_packing",
            "numeric_executor",
        ],
        "trigger": "question-only numeric, duration, amount, count, or comparison route",
    },
    "temporal_timeline": {
        "mechanisms": [
            "exhaustive_retrieval",
            "source_balanced_packing",
            "timeline_event_table",
        ],
        "trigger": "question-only chronology, ordering, recency, or relative-time route",
    },
    "set_join": {
        "mechanisms": [
            "exhaustive_retrieval",
            "source_balanced_packing",
            "set_join",
        ],
        "trigger": "question-only enumeration or complete-set route",
    },
    "synthesize": {
        "mechanisms": [
            "exhaustive_retrieval",
            "source_balanced_packing",
            "synthesis",
        ],
        "trigger": "question-only recommendation, explanation, or synthesis route",
    },
    "direct_extract": {
        "mechanisms": ["exhaustive_retrieval", "packing", "synthesis"],
        "trigger": "question-only point lookup route",
    },
}

POSTHOC_CLASS_REMEDIATION = {
    "source_missing": ["exhaustive_retrieval"],
    "candidate_reached_but_packing_dropped": ["packing"],
    "partial_multi_source_coverage": ["exhaustive_retrieval", "packing"],
    "operator_failure_despite_source_coverage": [
        "numeric_executor",
        "timeline_event_table",
        "set_join",
        "synthesis",
    ],
    "answer_shape_or_judge_ambiguity": [
        "numeric_executor",
        "exact_answer_formatter",
    ],
    "other": ["synthesis", "evidence_sufficiency_check"],
}


class JointFailureAnalysisError(ValueError):
    """Raised when a sealed input or a declared audit invariant changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise JointFailureAnalysisError(message)


@dataclass(frozen=True, slots=True)
class ArmSpec:
    label: str
    root: Path
    answer_sha256: str
    runtime_sha256: str
    judge_preflight_sha256: str
    judge_sha256: str
    score_sha256: str
    correct: int
    answer_preflight_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class VerifiedArm:
    spec: ArmSpec
    answer: SealedArtifact
    runtime: SealedArtifact
    judge: SealedArtifact
    score: SealedArtifact


@dataclass(frozen=True, slots=True)
class VerifiedInputs:
    parent: VerifiedArm
    payload: VerifiedArm
    fact: VerifiedArm
    guided_audit_path: Path
    target_plan_path: Path


def _arm_specs(
    parent_root: Path, payload_root: Path, fact_root: Path
) -> tuple[ArmSpec, ArmSpec, ArmSpec]:
    return (
        ArmSpec(
            label="s0_control_v2",
            root=parent_root,
            answer_sha256="1a2545655d4a5e2061dc1b80efae39c7f8c70f5dc394f36c97d1312f70f39d8a",
            runtime_sha256="f4f6d1a52ceea2b7f65cb66f51bb4925c1db9d20253c7ada7167216285a7d45b",
            judge_preflight_sha256="5ad11d9742cfe1de841c75106c6b434d480280f431d505195ed7c1753bc890d1",
            judge_sha256="05fec9a7f284bb4e95d286f44e7378a8bbc1737a03e7c2ed60aefd50e6ddc689",
            score_sha256="3422ce2825bdcdc347c8307bd3fed5a46de3dff6d33510c8bc3a3ba1c31c56e1",
            correct=53,
        ),
        ArmSpec(
            label="query_payload",
            root=payload_root,
            answer_preflight_sha256="c5c705470259743ce1fb7e07bd72374ada32352f5240e44d06a17cf450f7ac9d",
            answer_sha256="ab271ccb1bb830346fea64c9b11f3c7d504f048cc1ba392da39b177869106c6d",
            runtime_sha256="76150f82d0c6959b52309e0462970fe2c5e7e6fb5c0430a2313d18f423bdd902",
            judge_preflight_sha256="9adbc35c9aebfdbfc06943122ebac97e87b266f44554a75a63a73299de116828",
            judge_sha256="f0460baa796220f9975ab2f4e8250e231ed67da128182f4880f7ac9ef5a4c097",
            score_sha256="41ef567a1d27d4c840489def844372892fb029f7f57ea9f215780e19886d21bb",
            correct=71,
        ),
        ArmSpec(
            label="query_fact",
            root=fact_root,
            answer_preflight_sha256="dc890a923f08f0ee364dd2d39b202d2c2a6e7bd82b8453aee89d8b0379da2877",
            answer_sha256="0ee98720e1ed47658084a2afce3071e8e299f51e15924d7cfffd5c089574d515",
            runtime_sha256="f921ef5e05d6f56bb1957efa1f97d0c954689146ca387fa9b42fc6aa68440fae",
            judge_preflight_sha256="22284e98ccc42df54b467ba3881c4f3b05135843396f21f964aead8e62877da4",
            judge_sha256="78d9195a1510d75e3c1667229c64f91ee991c973e610b4e362a3c05b9b11e77c",
            score_sha256="1136ca8d36310a79b60e6eb53369047dd9f4da4099e4c23a7836aebbf35f109a",
            correct=64,
        ),
    )


def _read_exact(path: Path, expected_sha256: str, label: str) -> SealedArtifact:
    expected = require_sha256(expected_sha256, f"{label} SHA-256")
    artifact = read_sealed_json(path)
    _require(artifact.sha256 == expected, f"{label} checkpoint changed")
    return artifact


def _read_pair(
    root: Path, name: str, replay_name: str, expected_sha256: str, label: str
) -> SealedArtifact:
    artifact = _read_exact(root / name, expected_sha256, label)
    replay = _read_exact(root / replay_name, expected_sha256, f"{label} replay")
    _require(
        replay.payload == artifact.payload,
        f"{label} replay payload differs from the run",
    )
    return artifact


def _verify_self_sealed_rows(
    rows: object, seal_key: str, label: str
) -> tuple[dict[str, Any], ...]:
    _require(type(rows) is list, f"{label} rows must be an exact array")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        _require(type(raw) is dict, f"{label} row {index} must be an object")
        row = dict(raw)
        declared = row.pop(seal_key, None)
        _require(
            type(declared) is str and identity_sha256(row) == declared,
            f"{label} row {index} identity seal changed",
        )
        result.append(raw)
    return tuple(result)


def _verify_arm(spec: ArmSpec) -> VerifiedArm:
    if spec.answer_preflight_sha256 is not None:
        _read_exact(
            spec.root / "answer-preflight.json",
            spec.answer_preflight_sha256,
            f"{spec.label} answer preflight",
        )
    answer = _read_pair(
        spec.root,
        "answer-run.json",
        "answer-run-replay.json",
        spec.answer_sha256,
        f"{spec.label} answer run",
    )
    runtime = _read_pair(
        spec.root,
        "runtime-ledger.json",
        "runtime-ledger-replay.json",
        spec.runtime_sha256,
        f"{spec.label} runtime ledger",
    )
    _read_exact(
        spec.root / "judge-preflight.json",
        spec.judge_preflight_sha256,
        f"{spec.label} judge preflight",
    )
    judge = _read_pair(
        spec.root,
        "semantic-judge-sol.json",
        "semantic-judge-sol-replay.json",
        spec.judge_sha256,
        f"{spec.label} semantic judge",
    )
    score = _read_pair(
        spec.root,
        "score-ledger.json",
        "score-ledger-replay.json",
        spec.score_sha256,
        f"{spec.label} score ledger",
    )

    answer_rows = _verify_self_sealed_rows(
        answer.payload.get("questions"), "source_row_sha256", f"{spec.label} answer"
    )
    judge_rows = _verify_self_sealed_rows(
        judge.payload.get("questions"), "judge_row_sha256", f"{spec.label} judge"
    )
    _require(
        len(answer_rows) == len(judge_rows) == EXPECTED_QUESTION_COUNT,
        f"{spec.label} question count changed",
    )
    _require(
        tuple(row.get("ordinal") for row in answer_rows)
        == tuple(range(EXPECTED_QUESTION_COUNT)),
        f"{spec.label} answer order changed",
    )
    runtime_identity, runtime_row_ids = _validated_runtime_ledger(runtime.payload)
    _require(
        len(runtime_row_ids) == EXPECTED_QUESTION_COUNT,
        f"{spec.label} runtime answer population changed",
    )
    for answer_row, judge_row, runtime_row_id in zip(
        answer_rows, judge_rows, runtime_row_ids, strict=True
    ):
        ordinal = answer_row["ordinal"]
        _require(
            judge_row.get("ordinal") == ordinal
            and judge_row.get("question_id") == answer_row.get("question_id")
            and judge_row.get("question_sha256") == answer_row.get("question_sha256")
            and judge_row.get("dated_question_sha256")
            == answer_row.get("dated_question_sha256")
            and judge_row.get("prediction_sha256")
            == answer_row.get("prediction_sha256")
            and judge_row.get("runtime_row_id") == runtime_row_id,
            f"{spec.label} answer/judge binding changed at {ordinal}",
        )
    _require(
        judge.payload.get("answer_run_sha256") == answer.sha256
        and judge.payload.get("runtime_ledger_sha256") == runtime.sha256
        and judge.payload.get("runtime_ledger_identity_sha256") == runtime_identity,
        f"{spec.label} judge parent bindings changed",
    )
    score_unsigned = dict(score.payload)
    score_identity = score_unsigned.pop("ledger_identity_sha256", None)
    _require(
        type(score_identity) is str and identity_sha256(score_unsigned) == score_identity,
        f"{spec.label} score identity seal changed",
    )
    score_rows = score.payload.get("rows")
    _require(
        type(score_rows) is list and len(score_rows) == EXPECTED_QUESTION_COUNT,
        f"{spec.label} score population changed",
    )
    for judge_row, score_row in zip(judge_rows, score_rows, strict=True):
        _require(
            type(score_row) is dict
            and score_row.get("runtime_row_id") == judge_row.get("runtime_row_id")
            and score_row.get("judge_row_sha256") == judge_row.get("judge_row_sha256")
            and score_row.get("correct") == judge_row.get("correct"),
            f"{spec.label} judge/score row binding changed",
        )
    _require(
        judge.payload.get("aggregate", {}).get("correct") == spec.correct
        and score.payload.get("aggregate", {}).get("candidate_correct") == spec.correct,
        f"{spec.label} locked score changed",
    )
    _require(
        any(
            type(row) is dict
            and row.get("sha256") == judge.sha256
            and str(row.get("role", "")).endswith(":judge")
            for row in score.payload.get("source_artifacts", [])
        ),
        f"{spec.label} score/judge source binding changed",
    )
    return VerifiedArm(spec=spec, answer=answer, runtime=runtime, judge=judge, score=score)


def _verify_child_parent_bindings(child: VerifiedArm, parent: VerifiedArm) -> None:
    _require(
        child.answer.payload.get("parent_answer_run_sha256") == parent.answer.sha256
        and child.answer.payload.get("parent_answer_runtime_ledger_sha256")
        == parent.runtime.sha256
        and child.judge.payload.get("parent_answer_run_sha256")
        == parent.answer.sha256
        and child.judge.payload.get("parent_judge_sha256") == parent.judge.sha256
        and child.judge.payload.get("parent_score_ledger_sha256")
        == parent.score.sha256,
        f"{child.spec.label} exact S0-v2 parent binding changed",
    )
    for child_row, parent_row in zip(
        child.answer.payload["questions"],
        parent.answer.payload["questions"],
        strict=True,
    ):
        _require(
            child_row.get("parent_prediction_sha256")
            == parent_row.get("prediction_sha256"),
            f"{child.spec.label} parent prediction changed at {child_row['ordinal']}",
        )


def _verify_pinned_bytes(path: Path, expected_sha256: str, label: str) -> None:
    expected = require_sha256(expected_sha256, f"{label} SHA-256")
    _require(path.is_file() and not path.is_symlink(), f"{label} is not immutable")
    _require(file_sha256(path) == expected, f"{label} checkpoint changed")
    sidecar = path.with_name(path.name + ".sha256")
    _require(
        sidecar.is_file()
        and not sidecar.is_symlink()
        and sidecar.read_bytes() == f"{expected}  {path.name}\n".encode("ascii"),
        f"{label} sidecar changed",
    )


def verify_sealed_inputs(
    *,
    parent_root: Path = DEFAULT_PARENT_ROOT,
    payload_root: Path = DEFAULT_PAYLOAD_ROOT,
    fact_root: Path = DEFAULT_FACT_ROOT,
    guided_root: Path = DEFAULT_GUIDED_ROOT,
    target_plan_path: Path = DEFAULT_TARGET_PLAN,
) -> VerifiedInputs:
    """Verify all treatment and audit bytes without opening reference answers."""

    parent_spec, payload_spec, fact_spec = _arm_specs(
        parent_root, payload_root, fact_root
    )
    parent = _verify_arm(parent_spec)
    payload = _verify_arm(payload_spec)
    fact = _verify_arm(fact_spec)
    _verify_child_parent_bindings(payload, parent)
    _verify_child_parent_bindings(fact, parent)

    _read_pair(
        guided_root,
        "query-guided-scan-v1-run.json",
        "query-guided-scan-v1-run-replay.json",
        EXPECTED_GUIDED_RUN_SHA256,
        "query-guided scan run",
    )
    guided_ledger = _read_pair(
        guided_root,
        "runtime-ledger.json",
        "runtime-ledger-replay.json",
        EXPECTED_GUIDED_LEDGER_SHA256,
        "query-guided runtime ledger",
    )
    _validated_runtime_ledger(guided_ledger.payload)
    guided_audit_path = guided_root / "source-target-analysis-v1.json"
    _verify_pinned_bytes(
        guided_audit_path, EXPECTED_GUIDED_AUDIT_SHA256, "guided target audit"
    )
    _verify_pinned_bytes(
        target_plan_path, EXPECTED_TARGET_PLAN_SHA256, "target-owner plan"
    )
    return VerifiedInputs(
        parent=parent,
        payload=payload,
        fact=fact,
        guided_audit_path=guided_audit_path,
        target_plan_path=target_plan_path,
    )


def _load_references(dataset_path: Path, split_path: Path) -> tuple[Any, ...]:
    _require(
        file_sha256(dataset_path) == LOCKED_LONGMEMEVAL_DATASET_SHA256,
        "locked LongMemEval dataset SHA-256 changed",
    )
    _require(
        file_sha256(split_path) == LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
        "locked split-manifest SHA-256 changed",
    )
    selected = select_locked_split(
        load_benchmark(dataset_path, "longmemeval"),
        dataset_path=dataset_path,
        manifest=load_split_manifest(split_path),
        split="validation",
    )
    questions = tuple(question for sample in selected for question in sample.questions)
    _require(
        len(selected) == len(questions) == EXPECTED_QUESTION_COUNT,
        "locked validation question population changed",
    )
    return questions


def _answer_rows(arm: VerifiedArm) -> dict[int, dict[str, Any]]:
    return {int(row["ordinal"]): row for row in arm.answer.payload["questions"]}


def _judge_rows(arm: VerifiedArm) -> dict[int, dict[str, Any]]:
    return {int(row["ordinal"]): row for row in arm.judge.payload["questions"]}


def _matches_target(source_id: object, question_id: str, target_id: str) -> bool:
    """Match the target registry's short IDs to exact namespaced source IDs."""

    return type(source_id) is str and source_id in {
        target_id,
        f"{question_id}::{target_id}",
    }


def _stage_hit(
    audit_row: Mapping[str, Any] | None, method: str, stage: str
) -> bool:
    if audit_row is None:
        return False
    value = audit_row.get("hits", {}).get(method, {}).get(stage)
    _require(type(value) is bool, f"guided audit {method}/{stage} changed")
    return value


def _coverage_status(hit_count: int, target_count: int) -> str:
    if hit_count == 0:
        return "none"
    if hit_count == target_count:
        return "full"
    return "partial"


def _validate_cause(
    *, ordinal: int, cause: str, actual_count: int, target_count: int,
    query_candidate_count: int, guided_candidate_count: int,
) -> None:
    if cause == "source_missing":
        _require(
            actual_count == 0
            and query_candidate_count == 0
            and guided_candidate_count == 0,
            f"q{ordinal} no longer supports source_missing",
        )
    elif cause == "candidate_reached_but_packing_dropped":
        _require(
            actual_count == 0
            and query_candidate_count + guided_candidate_count > 0,
            f"q{ordinal} no longer supports a packing drop",
        )
    elif cause == "partial_multi_source_coverage":
        _require(
            target_count > 1 and 0 < actual_count < target_count,
            f"q{ordinal} no longer has partial multi-source coverage",
        )
    elif cause == "operator_failure_despite_source_coverage":
        _require(
            actual_count == target_count,
            f"q{ordinal} no longer has full registered-source coverage",
        )
    elif cause == "answer_shape_or_judge_ambiguity":
        _require(ordinal in {53, 75}, f"q{ordinal} answer-shape audit changed")
    elif cause == "other":
        _require(ordinal == 42, f"q{ordinal} other audit changed")


def _cause_rationale(
    cause: str, actual_count: int, target_count: int, ordinal: int
) -> str:
    if cause == "source_missing":
        return "No registered target source reached either sealed answer packet or the guided candidate audit."
    if cause == "candidate_reached_but_packing_dropped":
        return "The target reached a question-guided candidate stage but was absent from the sealed treatment packet."
    if cause == "partial_multi_source_coverage":
        return f"Only {actual_count}/{target_count} registered sources were present in the sealed treatment packet."
    if cause == "operator_failure_despite_source_coverage":
        return "All registered source IDs were present, but both sealed answer arms remained incorrect."
    if cause == "answer_shape_or_judge_ambiguity":
        if ordinal == 53:
            return "The direct arm produced the correct lower-bound number ('at least 3') but not the exact-count answer required by the judge; fact remained abstaining."
        return "All three predictions used a greater-than hedge around $270 while the reference required exactly $270."
    return "The reference is an evidence-insufficiency answer, while all arms asserted an unsupported university."


def build_analysis_payload(
    *, inputs: VerifiedInputs, references: Sequence[Any],
    target_plan: Mapping[str, Any], guided_audit: Mapping[str, Any],
) -> dict[str, Any]:
    parent_answers, payload_answers, fact_answers = (
        _answer_rows(inputs.parent),
        _answer_rows(inputs.payload),
        _answer_rows(inputs.fact),
    )
    parent_judges, payload_judges, fact_judges = (
        _judge_rows(inputs.parent),
        _judge_rows(inputs.payload),
        _judge_rows(inputs.fact),
    )
    joint = tuple(
        ordinal
        for ordinal in range(EXPECTED_QUESTION_COUNT)
        if not payload_judges[ordinal]["correct"]
        and not fact_judges[ordinal]["correct"]
    )
    _require(joint == JOINT_FAILURE_ORDINALS, "joint-failure population changed")

    desired = target_plan.get("desired_targets")
    _require(type(desired) is list, "target plan desired_targets changed")
    targets_by_ordinal: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in desired:
        if type(raw) is dict and raw.get("target_kind") == "source_id":
            targets_by_ordinal[int(raw["ordinal"])].append(raw)

    missing_block = guided_audit.get("all_30_s0_missing_source_reach", {})
    missing_rows = missing_block.get("rows")
    _require(type(missing_rows) is list, "guided missing-source rows changed")
    audit_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in missing_rows:
        _require(type(raw) is dict, "guided target row must be an object")
        key = (int(raw["ordinal"]), str(raw["source_id"]))
        _require(key not in audit_by_key, "guided target rows are not unique")
        audit_by_key[key] = raw

    rows: list[dict[str, Any]] = []
    cause_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    cause_route: dict[str, Counter[str]] = defaultdict(Counter)
    prospective_status_counts: Counter[str] = Counter()
    for ordinal in joint:
        parent = parent_answers[ordinal]
        payload = payload_answers[ordinal]
        fact = fact_answers[ordinal]
        question = references[ordinal]
        _require(
            question.question_id == payload["question_id"]
            and quote_sha256(question.question) == payload["question_sha256"]
            and quote_sha256(question.dated_question)
            == payload["dated_question_sha256"],
            f"reference question binding changed at {ordinal}",
        )
        reference_sha256 = quote_sha256(question.answer)
        for judge in (parent_judges[ordinal], payload_judges[ordinal], fact_judges[ordinal]):
            _require(
                judge["reference_sha256"] == reference_sha256,
                f"reference/judge binding changed at {ordinal}",
            )
        targets = targets_by_ordinal[ordinal]
        _require(targets, f"q{ordinal} has no registered source targets")
        aliases = payload.get("alias_receipt")
        _require(type(aliases) is list, f"q{ordinal} alias receipt changed")
        target_rows: list[dict[str, Any]] = []
        for target in targets:
            target_id = str(target["target_id"])
            matching = [
                alias
                for alias in aliases
                if type(alias) is dict
                and _matches_target(alias.get("source_id"), payload["question_id"], target_id)
            ]
            matching_tiers = {
                str(alias.get("tier")) for alias in matching if alias.get("tier") is not None
            }
            _require(
                matching_tiers
                <= {"protected_s0", "query_expansion_delta"},
                f"q{ordinal}/{target_id} appears in an unknown payload tier",
            )
            audit = audit_by_key.get((ordinal, target_id))
            actual = bool(matching)
            protected = "protected_s0" in matching_tiers
            query_admitted = not protected and "query_expansion_delta" in matching_tiers
            payload_tier = (
                "protected_s0"
                if protected
                else "query_expansion_delta"
                if query_admitted
                else None
            )
            if audit is None:
                _require(protected, f"q{ordinal}/{target_id} escaped the missing-source audit")
            else:
                _require(
                    _stage_hit(audit, "protected_s0", "protected") is False,
                    f"q{ordinal}/{target_id} missing-source denominator changed",
                )
                _require(
                    query_admitted
                    == _stage_hit(
                        audit, "query_expansion", "admitted_after_s0_dedup"
                    ),
                    f"q{ordinal}/{target_id} payload/audit admission differs",
                )
            query_candidate = _stage_hit(
                audit, "query_expansion", "candidate_reached"
            )
            guided_candidate = _stage_hit(
                audit, "query_guided_scan_v1", "candidate_reached"
            )
            partition_admitted = _stage_hit(
                audit, "partition_scan_v2_r96", "admitted_after_s0_dedup"
            )
            repack_admitted = _stage_hit(
                audit, "query_expansion_repack_v2", "admitted_after_s0_dedup"
            )
            guided_selected = _stage_hit(
                audit, "query_guided_scan_v1", "selected_before_s0_dedup"
            )
            guided_admitted = _stage_hit(
                audit, "query_guided_scan_v1", "admitted_after_s0_dedup"
            )
            union_admitted = actual or partition_admitted or repack_admitted or guided_admitted
            target_rows.append(
                {
                    "source_id": target_id,
                    "primary_owner": target["primary_owner"],
                    "payload_admitted": actual,
                    "payload_tier": payload_tier,
                    "query_v1_candidate": query_candidate,
                    "query_v1_admitted": query_admitted,
                    "guided_candidate": guided_candidate,
                    "guided_selected": guided_selected,
                    "guided_admitted": guided_admitted,
                    "prospective_union_admitted": union_admitted,
                }
            )
        target_count = len(target_rows)
        actual_count = sum(row["payload_admitted"] for row in target_rows)
        query_candidate_count = sum(row["query_v1_candidate"] for row in target_rows)
        guided_candidate_count = sum(row["guided_candidate"] for row in target_rows)
        union_count = sum(row["prospective_union_admitted"] for row in target_rows)
        cause = CAUSE_BY_ORDINAL[ordinal]
        _validate_cause(
            ordinal=ordinal,
            cause=cause,
            actual_count=actual_count,
            target_count=target_count,
            query_candidate_count=query_candidate_count,
            guided_candidate_count=guided_candidate_count,
        )
        route = str(payload["route_style"])
        _require(
            route in QUESTION_ONLY_POLICY and fact.get("route_style") == route,
            f"q{ordinal} route binding changed",
        )
        union_status = _coverage_status(union_count, target_count)
        cause_counts[cause] += 1
        route_counts[route] += 1
        cause_route[cause][route] += 1
        prospective_status_counts[union_status] += 1
        row: dict[str, Any] = {
            "ordinal": ordinal,
            "question_id": payload["question_id"],
            "route_id": route,
            "reference": question.answer,
            "reference_sha256": reference_sha256,
            "predictions": {
                "parent": {
                    "prediction": parent["prediction"],
                    "prediction_sha256": parent["prediction_sha256"],
                    "correct": parent_judges[ordinal]["correct"],
                },
                "query_payload": {
                    "prediction": payload["prediction"],
                    "prediction_sha256": payload["prediction_sha256"],
                    "correct": False,
                },
                "query_fact": {
                    "prediction": fact["prediction"],
                    "prediction_sha256": fact["prediction_sha256"],
                    "correct": False,
                },
            },
            "evidence_target_audit": {
                "registered_source_count": target_count,
                "payload_admitted_source_count": actual_count,
                "payload_coverage": _coverage_status(actual_count, target_count),
                "query_v1_candidate_count": query_candidate_count,
                "guided_candidate_count": guided_candidate_count,
                "prospective_union_admitted_source_count": union_count,
                "prospective_union_coverage": union_status,
                "source_id_reach_is_answer_bearing_span_proof": False,
                "targets": target_rows,
            },
            "posthoc_label": {
                "dominant_cause": cause,
                "secondary_causes": list(SECONDARY_CAUSES.get(ordinal, ())),
                "gold_informed": True,
                "rationale": _cause_rationale(cause, actual_count, target_count, ordinal),
            },
            "deployable_question_only_policy_id": route,
        }
        row["row_sha256"] = identity_sha256(row)
        rows.append(row)

    _require(
        dict(cause_counts)
        == {
            "operator_failure_despite_source_coverage": 16,
            "partial_multi_source_coverage": 5,
            "source_missing": 2,
            "other": 1,
            "answer_shape_or_judge_ambiguity": 2,
            "candidate_reached_but_packing_dropped": 2,
        },
        "dominant-cause aggregate changed",
    )
    _require(
        dict(prospective_status_counts) == {"full": 24, "none": 3, "partial": 1},
        "prospective union coverage sanity check changed",
    )
    payload: dict[str, Any] = {
        "format": ANALYSIS_FORMAT,
        "analysis_kind": "posthoc_gold_informed_evaluation_only",
        "question_count": len(rows),
        "joint_failure_ordinals": list(joint),
        "provider_calls": 0,
        "retrieval_rerun": False,
        "retrieval_mutated": False,
        "answers_mutated": False,
        "transformer_state_bytes": 0,
        "artifact_verification_before_reference_load": True,
        "bindings": {
            "parent_answer_sha256": inputs.parent.answer.sha256,
            "parent_judge_sha256": inputs.parent.judge.sha256,
            "parent_score_sha256": inputs.parent.score.sha256,
            "query_payload_answer_sha256": inputs.payload.answer.sha256,
            "query_payload_judge_sha256": inputs.payload.judge.sha256,
            "query_payload_score_sha256": inputs.payload.score.sha256,
            "query_fact_answer_sha256": inputs.fact.answer.sha256,
            "query_fact_judge_sha256": inputs.fact.judge.sha256,
            "query_fact_score_sha256": inputs.fact.score.sha256,
            "guided_target_audit_sha256": EXPECTED_GUIDED_AUDIT_SHA256,
            "target_plan_sha256": EXPECTED_TARGET_PLAN_SHA256,
            "dataset_sha256": LOCKED_LONGMEMEVAL_DATASET_SHA256,
            "split_sha256": LOCKED_LONGMEMEVAL_SPLIT_MANIFEST_SHA256,
        },
        "locked_scores": {
            "parent": 53,
            "query_payload": 71,
            "query_fact": 64,
        },
        "aggregate": {
            "dominant_cause_counts": {
                cause: cause_counts.get(cause, 0) for cause in CAUSE_ORDER
            },
            "route_counts": dict(sorted(route_counts.items())),
            "cause_by_route": {
                cause: dict(sorted(cause_route[cause].items()))
                for cause in CAUSE_ORDER
            },
            "prospective_union_question_coverage": {
                status: prospective_status_counts.get(status, 0)
                for status in ("full", "partial", "none")
            },
        },
        "interpretation_boundary": {
            "posthoc_labels_are_deployable_online_policy": False,
            "question_only_policy_uses_gold_or_correctness": False,
            "prospective_union_was_run_through_answer_arms": False,
            "source_id_reach_is_answer_bearing_span_proof": False,
            "source_id_reach_is_qa_accuracy": False,
        },
        "posthoc_class_remediation_not_online_routing": POSTHOC_CLASS_REMEDIATION,
        "deployable_question_only_policy": QUESTION_ONLY_POLICY,
        "rows": rows,
    }
    payload["analysis_sha256"] = identity_sha256(payload)
    return payload


def analyze_paths(
    *,
    parent_root: Path = DEFAULT_PARENT_ROOT,
    payload_root: Path = DEFAULT_PAYLOAD_ROOT,
    fact_root: Path = DEFAULT_FACT_ROOT,
    guided_root: Path = DEFAULT_GUIDED_ROOT,
    target_plan_path: Path = DEFAULT_TARGET_PLAN,
    dataset_path: Path = DEFAULT_DATASET,
    split_path: Path = DEFAULT_SPLIT,
) -> dict[str, Any]:
    inputs = verify_sealed_inputs(
        parent_root=parent_root,
        payload_root=payload_root,
        fact_root=fact_root,
        guided_root=guided_root,
        target_plan_path=target_plan_path,
    )
    # This is the explicit boundary: references and gold-tagged target payloads
    # are not parsed until every answer/judge/score/runtime byte is verified.
    references = _load_references(dataset_path, split_path)
    target_plan = read_sealed_json(inputs.target_plan_path)
    guided_audit = read_sealed_json(inputs.guided_audit_path)
    _require(
        target_plan.sha256 == EXPECTED_TARGET_PLAN_SHA256
        and guided_audit.sha256 == EXPECTED_GUIDED_AUDIT_SHA256,
        "post-verification audit bytes changed",
    )
    return build_analysis_payload(
        inputs=inputs,
        references=references,
        target_plan=target_plan.payload,
        guided_audit=guided_audit.payload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--payload-root", type=Path, default=DEFAULT_PAYLOAD_ROOT)
    parser.add_argument("--fact-root", type=Path, default=DEFAULT_FACT_ROOT)
    parser.add_argument("--guided-root", type=Path, default=DEFAULT_GUIDED_ROOT)
    parser.add_argument("--target-plan", type=Path, default=DEFAULT_TARGET_PLAN)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = analyze_paths(
        parent_root=args.parent_root,
        payload_root=args.payload_root,
        fact_root=args.fact_root,
        guided_root=args.guided_root,
        target_plan_path=args.target_plan,
        dataset_path=args.dataset,
        split_path=args.split,
    )
    artifact, created = publish_sealed_json(args.output, payload)
    print(
        {
            "artifact": str(artifact.path),
            "sha256": artifact.sha256,
            "analysis_sha256": payload["analysis_sha256"],
            "created": created,
            "joint_failures": payload["question_count"],
            "causes": payload["aggregate"]["dominant_cause_counts"],
            "provider_calls": 0,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
