#!/usr/bin/env python3
"""Seal a provider-free execution plan for a confirmation treatment.

The compiler accepts only the immutable, label-free ``ConfirmationTreatmentInput``
created by the v4 population firebreak.  It hashes (rather than republishes)
question and history content, partitions the ordered treatment into caller-declared
contiguous namespaces, and reports only call counts that are knowable before the
question-local runtime gates exist.

There is deliberately no provider execution path in this module.  In particular,
an unresolved synthesis count is not an authorization and is never approximated
from another population's admission rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from tools.v4_population_firebreak.canonical import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
    parse_json_bytes,
    publish_no_clobber,
)
from tools.v4_population_firebreak.treatment import (  # noqa: E402
    ConfirmationTreatmentInput,
    TreatmentQuestion,
    TreatmentSample,
    load_confirmation_treatment_input,
)


FORMAT = "memory-condense-confirmation-treatment-pipeline-preflight-v1"
ROW_FORMAT = f"{FORMAT}-row-v1"
NAMESPACE_FORMAT = f"{FORMAT}-namespace-v1"
NAMESPACE_KEY_FORMAT = f"{FORMAT}-namespace-key-v1"
STAGE_FORMAT = f"{FORMAT}-stage-v1"
POLICY_FORMAT = f"{FORMAT}-population-neutral-policy-v1"
INPUT_BINDING_FORMAT = f"{FORMAT}-input-binding-v1"

DEFAULT_QUESTIONS_PER_NAMESPACE = 10


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PLAN_KEYS = frozenset(
    {
        "answer",
        "answers",
        "baseline_correct",
        "baseline_judge_row_sha256",
        "benchmark_category",
        "category",
        "correct",
        "desired_answer",
        "evidence_topology_class",
        "expected_answer",
        "gold",
        "gold_answer",
        "gold_answer_sha256",
        "ground_truth",
        "ground_truth_answer",
        "judge_row_sha256",
        "judge_verdict",
        "judge_verdict_sha256",
        "primary_target_count",
        "primary_target_recalled",
        "question_only_demand_class",
        "reference",
        "reference_answer",
        "reference_answer_sha256",
        "regressed",
        "rescued",
        "target_owner",
        "verdict",
    }
)
_FALSE_LABEL_SENTINELS = frozenset(
    {
        "benchmark_categories_loaded",
        "benchmark_source_labels_loaded",
        "gold_fields_present",
        "gold_loaded",
    }
)


class ConfirmationPipelinePlanError(ValueError):
    """The label-free input or requested namespace schedule is not exact."""


class ConfirmationPipelineSealError(ConfirmationPipelinePlanError):
    """A sealed plan or its digest sidecar is missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class SealedConfirmationPipelinePlan:
    path: Path
    sha256: str
    payload: dict[str, Any]


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationPipelinePlanError(message)


def identity_sha256(value: object) -> str:
    """Hash the strict canonical JSON identity used by runtime receipts."""

    return canonical_sha256(value)


def quote_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_sha256(value: object, label: str) -> str:
    _require(
        type(value) is str and _SHA256.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value  # type: ignore[return-value]


def assert_plan_gold_blind(value: object, path: str = "preflight") -> None:
    """Fail if scorer labels enter the provider-free planning projection."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            child_path = f"{path}.{name}"
            if name.casefold() in _FALSE_LABEL_SENTINELS:
                _require(child is False, f"label sentinel must be false: {child_path}")
                continue
            _require(
                name.casefold() not in _FORBIDDEN_PLAN_KEYS,
                f"label-bearing field is forbidden: {child_path}",
            )
            assert_plan_gold_blind(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_plan_gold_blind(child, f"{path}[{index}]")


def _sidecar_bytes(path: Path, digest: str) -> bytes:
    return f"{digest}  {path.name}\n".encode("ascii")


def read_sealed_confirmation_pipeline_plan(
    path: str | Path,
) -> SealedConfirmationPipelinePlan:
    target = Path(path)
    sidecar = target.with_name(target.name + ".sha256")
    if target.is_symlink() or not target.is_file():
        raise ConfirmationPipelineSealError(
            f"plan must be a regular file: {target}"
        )
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ConfirmationPipelineSealError(
            f"plan digest sidecar is invalid: {sidecar}"
        )
    try:
        raw = target.read_bytes()
        payload = parse_json_bytes(raw, "confirmation pipeline plan")
    except (OSError, ValueError) as exc:
        raise ConfirmationPipelineSealError("cannot read sealed pipeline plan") from exc
    if type(payload) is not dict or raw != canonical_json_bytes(payload) + b"\n":
        raise ConfirmationPipelineSealError("pipeline plan is not canonical JSON")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        sidecar_raw = sidecar.read_bytes()
    except OSError as exc:
        raise ConfirmationPipelineSealError(
            "cannot read pipeline plan sidecar"
        ) from exc
    if sidecar_raw != _sidecar_bytes(target, digest):
        raise ConfirmationPipelineSealError("pipeline plan digest sidecar is invalid")
    return SealedConfirmationPipelinePlan(target, digest, payload)


def publish_sealed_confirmation_pipeline_plan(
    path: str | Path, payload: dict[str, Any]
) -> tuple[SealedConfirmationPipelinePlan, bool]:
    """Publish once or reuse only a byte-identical plan and exact sidecar."""

    target = Path(path)
    sidecar = target.with_name(target.name + ".sha256")
    raw = canonical_json_bytes(payload) + b"\n"
    digest = hashlib.sha256(raw).hexdigest()
    if (
        target.exists()
        or target.is_symlink()
        or sidecar.exists()
        or sidecar.is_symlink()
    ):
        existing = read_sealed_confirmation_pipeline_plan(target)
        if existing.sha256 != digest:
            raise ConfirmationPipelineSealError(
                f"refusing to replace a different sealed plan: {target}"
            )
        return existing, False

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfirmationPipelineSealError("cannot create plan directory") from exc
    target_created = False
    try:
        publish_no_clobber(target, raw)
        target_created = True
        publish_no_clobber(sidecar, _sidecar_bytes(target, digest))
    except (OSError, ValueError) as exc:
        # Roll back only the exact file created by this call.  Never touch a
        # pre-existing or different path, even during an interrupted publish.
        if target_created and not sidecar.exists():
            try:
                if (
                    target.is_file()
                    and not target.is_symlink()
                    and target.read_bytes() == raw
                ):
                    target.unlink()
            except OSError:
                pass
        raise ConfirmationPipelineSealError(
            "cannot publish sealed pipeline plan"
        ) from exc
    return read_sealed_confirmation_pipeline_plan(target), True


def _with_receipt(
    body: Mapping[str, Any], *, key: str = "receipt_sha256"
) -> dict[str, Any]:
    projection = dict(body)
    return {**projection, key: identity_sha256(projection)}


def _exact_text(value: object, label: str) -> str:
    _require(type(value) is str and bool(value), f"{label} must be non-empty text")
    return value


def _canonical_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    _require(
        type(value) is datetime and value.tzinfo is not None,
        "treatment timestamps must be timezone-aware datetimes",
    )
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z")


def _sample_projection(sample: TreatmentSample) -> dict[str, Any]:
    """Recreate the exact sanitized firebreak projection from decoded values."""

    _require(type(sample) is TreatmentSample, "treatment sample changed type")
    _require(
        type(sample.turns) is tuple
        and type(sample.turn_source_ids) is tuple
        and type(sample.turn_created_at) is tuple,
        "treatment sample collections must be immutable tuples",
    )
    _require(
        len(sample.turns)
        == len(sample.turn_source_ids)
        == len(sample.turn_created_at),
        "treatment turn coordinates are misaligned",
    )
    turns: list[list[str]] = []
    for role, text in sample.turns:
        _require(
            type(role) is str
            and role in {"user", "assistant", "system"}
            and type(text) is str
            and bool(text),
            "treatment turn changed schema",
        )
        turns.append([role, text])
    sources: list[str | None] = []
    for source_id in sample.turn_source_ids:
        _require(
            source_id is None or (type(source_id) is str and bool(source_id)),
            "treatment source coordinate changed schema",
        )
        sources.append(source_id)
    _require(
        type(sample.questions) is tuple and len(sample.questions) == 1,
        "treatment sample must contain one immutable question",
    )
    question = sample.questions[0]
    _require(type(question) is TreatmentQuestion, "treatment question changed type")
    sample_id = _exact_text(sample.sample_id, "treatment sample ID")
    question_id = _exact_text(question.question_id, "treatment question ID")
    _require(question_id == sample_id, "treatment question and sample IDs differ")
    question_text = _exact_text(question.question, "treatment question")
    _require(
        question.question_date is None
        or (type(question.question_date) is str and bool(question.question_date)),
        "treatment question date changed schema",
    )
    return {
        "sample_id": sample_id,
        "turns": turns,
        "turn_source_ids": sources,
        "turn_created_at": [
            _canonical_timestamp(value) for value in sample.turn_created_at
        ],
        "questions": [
            {
                "question_id": question_id,
                "question": question_text,
                "question_date": question.question_date,
            }
        ],
    }


def _validate_treatment(
    treatment: ConfirmationTreatmentInput,
) -> tuple[dict[str, Any], ...]:
    _require(
        type(treatment) is ConfirmationTreatmentInput,
        "pipeline input must be a ConfirmationTreatmentInput",
    )
    for value, label in (
        (treatment.file_sha256, "treatment file"),
        (treatment.sanitized_projection_sha256, "treatment projection"),
        (treatment.dataset_sha256, "treatment dataset"),
        (treatment.split_manifest_sha256, "treatment split"),
        (treatment.ordered_question_ids_sha256, "treatment ordered IDs"),
        (
            treatment.ordered_normalized_sample_bindings_sha256,
            "treatment normalized bindings",
        ),
        (treatment.ordered_raw_record_bindings_sha256, "treatment raw bindings"),
    ):
        require_sha256(value, label)
    _require(
        type(treatment.samples) is tuple and bool(treatment.samples),
        "treatment samples must be a non-empty immutable tuple",
    )
    projections = tuple(_sample_projection(sample) for sample in treatment.samples)
    question_ids = tuple(str(row["sample_id"]) for row in projections)
    _require(
        len(question_ids) == len(set(question_ids)),
        "treatment question IDs repeat",
    )
    _require(
        canonical_sha256(list(question_ids)) == treatment.ordered_question_ids_sha256,
        "treatment ordered question IDs changed",
    )
    _require(
        canonical_sha256(list(projections)) == treatment.sanitized_projection_sha256,
        "treatment sanitized projection changed",
    )
    return projections


def uniform_namespace_sizes(
    question_count: int, questions_per_namespace: int
) -> tuple[int, ...]:
    """Return contiguous shard sizes, allowing a smaller final namespace."""

    _require(
        type(question_count) is int and question_count > 0,
        "question count must be a positive integer",
    )
    _require(
        type(questions_per_namespace) is int and questions_per_namespace > 0,
        "questions per namespace must be a positive integer",
    )
    full, remainder = divmod(question_count, questions_per_namespace)
    return (questions_per_namespace,) * full + ((remainder,) if remainder else ())


def _validate_namespace_sizes(
    namespace_sizes: Sequence[int], question_count: int
) -> tuple[int, ...]:
    _require(
        isinstance(namespace_sizes, Sequence)
        and not isinstance(namespace_sizes, (str, bytes))
        and bool(namespace_sizes),
        "namespace schedule must be a non-empty sequence",
    )
    sizes = tuple(namespace_sizes)
    _require(
        all(type(size) is int and size > 0 for size in sizes),
        "namespace sizes must be positive exact integers",
    )
    _require(
        sum(sizes) == question_count,
        "namespace schedule must cover the treatment exactly once",
    )
    return sizes


def _row_projection(sample: Mapping[str, Any]) -> dict[str, Any]:
    question_id = str(sample["sample_id"])
    question = sample["questions"][0]
    question_text = str(question["question"])
    question_date = question["question_date"]
    dated_question = (
        question_text
        if question_date is None
        else f"[Question asked at {question_date}]\n{question_text}"
    )
    content_body = {
        # IDs are intentionally absent.  Renumbering can change identity
        # receipts without changing the memory/question content binding.
        "question": question_text,
        "question_date": question_date,
        "turn_created_at": sample["turn_created_at"],
        "turn_source_ids": sample["turn_source_ids"],
        "turns": sample["turns"],
    }
    body = {
        "format": ROW_FORMAT,
        "question_id": question_id,
        "question_sha256": quote_sha256(question_text),
        "dated_question_sha256": quote_sha256(dated_question),
        "turn_count": len(sample["turns"]),
        "turns_sha256": identity_sha256(sample["turns"]),
        "turn_source_ids_sha256": identity_sha256(sample["turn_source_ids"]),
        "turn_created_at_sha256": identity_sha256(sample["turn_created_at"]),
        "content_binding_sha256": identity_sha256(content_body),
    }
    return _with_receipt(body, key="row_receipt_sha256")


def _namespace_projection(
    rows: Sequence[Mapping[str, Any]], *, content_occurrence: int
) -> dict[str, Any]:
    ordered_receipts = [str(row["row_receipt_sha256"]) for row in rows]
    content_receipts = sorted(str(row["content_binding_sha256"]) for row in rows)
    content_population_sha256 = identity_sha256(content_receipts)
    namespace_id = identity_sha256(
        {
            "format": NAMESPACE_KEY_FORMAT,
            "content_population_sha256": content_population_sha256,
            "content_occurrence": content_occurrence,
        }
    )
    body = {
        "format": NAMESPACE_FORMAT,
        "namespace_id": namespace_id,
        "content_occurrence": content_occurrence,
        "question_count": len(rows),
        "question_ids": [str(row["question_id"]) for row in rows],
        "ordered_row_receipts_sha256": identity_sha256(ordered_receipts),
        "content_population_sha256": content_population_sha256,
    }
    return _with_receipt(body, key="namespace_receipt_sha256")


def _policy_projection() -> dict[str, Any]:
    body = {
        "format": POLICY_FORMAT,
        "namespace_membership_rule": (
            "caller-declared-positive-contiguous-sizes-over-verified-treatment-order-v1"
        ),
        "namespace_identity_rule": "unordered-content-bindings-v1",
        "preflight_routing_rule": "none",
        "runtime_route_authority": "sealed-question-local-state-only",
        "population_size_constant": None,
        "runtime_route_sample_id_branching": False,
        "runtime_route_position_branching": False,
        "cross_population_admission_rate_reuse": False,
    }
    return _with_receipt(body, key="policy_receipt_sha256")


def _stage(
    *,
    stage_id: str,
    logical_question_count: int,
    count_status: str,
    would_call_count: int | None,
    provider_class: str | None,
    count_basis: str,
    upper_bound: int | None,
) -> dict[str, Any]:
    _require(count_status in {"exact", "deferred"}, "stage count status changed")
    _require(
        (count_status == "exact" and type(would_call_count) is int)
        or (count_status == "deferred" and would_call_count is None),
        "stage call count disagrees with its status",
    )
    body = {
        "format": STAGE_FORMAT,
        "stage_id": stage_id,
        "logical_question_count": logical_question_count,
        "provider_class": provider_class,
        "call_count_status": count_status,
        "would_call_count": would_call_count,
        "would_call_count_upper_bound": upper_bound,
        "count_basis": count_basis,
        "provider_execution_enabled": False,
        "authorization_released": False,
    }
    return _with_receipt(body, key="stage_receipt_sha256")


def _stages(question_count: int) -> tuple[dict[str, Any], ...]:
    """Describe the minimal lifecycle without inventing unresolved budgets."""

    return (
        _stage(
            stage_id="treatment_verification",
            logical_question_count=question_count,
            count_status="exact",
            would_call_count=0,
            provider_class=None,
            count_basis="provider-free-firebreak-load",
            upper_bound=0,
        ),
        _stage(
            stage_id="memory_materialization_and_retrieval",
            logical_question_count=question_count,
            count_status="exact",
            would_call_count=0,
            provider_class=None,
            count_basis="provider-free-local-memory-pipeline",
            upper_bound=0,
        ),
        _stage(
            stage_id="upstream_parent_synthesis",
            logical_question_count=question_count,
            count_status="deferred",
            would_call_count=None,
            provider_class="terra",
            count_basis="requires-sealed-question-local-upstream-gate-receipts",
            upper_bound=None,
        ),
        _stage(
            stage_id="terminal_synthesis",
            logical_question_count=question_count,
            count_status="deferred",
            would_call_count=None,
            provider_class="terra",
            count_basis="requires-sealed-question-local-terminal-eligibility-receipts",
            upper_bound=question_count,
        ),
        _stage(
            stage_id="official_full_population_judge",
            logical_question_count=question_count,
            count_status="exact",
            would_call_count=question_count,
            provider_class="sol",
            count_basis="one-call-per-frozen-prediction-after-label-opening",
            upper_bound=question_count,
        ),
    )


def compile_confirmation_pipeline_preflight(
    treatment: ConfirmationTreatmentInput,
    *,
    namespace_sizes: Sequence[int],
) -> dict[str, Any]:
    """Compile one deterministic, label-free, provider-free execution plan."""

    samples = _validate_treatment(treatment)
    sizes = _validate_namespace_sizes(namespace_sizes, len(samples))
    rows = tuple(_row_projection(sample) for sample in samples)

    namespaces: list[dict[str, Any]] = []
    content_occurrences: dict[str, int] = {}
    cursor = 0
    for size in sizes:
        member_rows = rows[cursor : cursor + size]
        content_root = identity_sha256(
            sorted(str(row["content_binding_sha256"]) for row in member_rows)
        )
        occurrence = content_occurrences.get(content_root, 0)
        content_occurrences[content_root] = occurrence + 1
        namespace = _namespace_projection(
            member_rows, content_occurrence=occurrence
        )
        namespaces.append(namespace)
        cursor += size
    _require(cursor == len(rows), "namespace schedule did not consume every row")

    policy = _policy_projection()
    stages = _stages(len(rows))
    known_calls = sum(
        int(stage["would_call_count"])
        for stage in stages
        if stage["call_count_status"] == "exact"
    )
    input_binding = _with_receipt(
        {
            "format": INPUT_BINDING_FORMAT,
            "dataset_sha256": treatment.dataset_sha256,
            "split_manifest_sha256": treatment.split_manifest_sha256,
            "treatment_file_sha256": treatment.file_sha256,
            "sanitized_projection_sha256": treatment.sanitized_projection_sha256,
            "ordered_question_ids_sha256": treatment.ordered_question_ids_sha256,
            "ordered_normalized_sample_bindings_sha256": (
                treatment.ordered_normalized_sample_bindings_sha256
            ),
            "ordered_raw_record_bindings_sha256": (
                treatment.ordered_raw_record_bindings_sha256
            ),
        },
        key="input_binding_receipt_sha256",
    )
    body: dict[str, Any] = {
        "format": FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "provider_execution_available": False,
        "question_count": len(rows),
        "namespace_count": len(namespaces),
        "namespace_sizes": list(sizes),
        "input_binding": input_binding,
        "policy": policy,
        "rows": list(rows),
        "namespaces": namespaces,
        "stages": list(stages),
        "known_would_call_count": known_calls,
        "total_would_call_count_exact": all(
            stage["call_count_status"] == "exact" for stage in stages
        ),
        "deferred_stage_ids": [
            str(stage["stage_id"])
            for stage in stages
            if stage["call_count_status"] == "deferred"
        ],
    }
    assert_plan_gold_blind(body)
    return {
        **body,
        "preflight_identity_sha256": identity_sha256(body),
    }


def compile_uniform_confirmation_pipeline_preflight(
    treatment: ConfirmationTreatmentInput,
    *,
    questions_per_namespace: int = DEFAULT_QUESTIONS_PER_NAMESPACE,
) -> dict[str, Any]:
    return compile_confirmation_pipeline_preflight(
        treatment,
        namespace_sizes=uniform_namespace_sizes(
            len(treatment.samples), questions_per_namespace
        ),
    )


def publish_confirmation_pipeline_preflight(
    output_path: str | Path,
    treatment: ConfirmationTreatmentInput,
    *,
    namespace_sizes: Sequence[int],
) -> tuple[SealedConfirmationPipelinePlan, bool]:
    payload = compile_confirmation_pipeline_preflight(
        treatment, namespace_sizes=namespace_sizes
    )
    return publish_sealed_confirmation_pipeline_plan(output_path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment-input", type=Path, required=True)
    parser.add_argument("--expected-treatment-file-sha256", required=True)
    parser.add_argument("--expected-sanitized-projection-sha256", required=True)
    parser.add_argument(
        "--questions-per-namespace",
        type=int,
        default=DEFAULT_QUESTIONS_PER_NAMESPACE,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    treatment = load_confirmation_treatment_input(
        args.treatment_input,
        expected_file_sha256=str(args.expected_treatment_file_sha256),
        expected_sanitized_projection_sha256=(
            str(args.expected_sanitized_projection_sha256)
        ),
    )
    sizes = uniform_namespace_sizes(
        len(treatment.samples), int(args.questions_per_namespace)
    )
    artifact, created = publish_confirmation_pipeline_preflight(
        args.output,
        treatment,
        namespace_sizes=sizes,
    )
    return {
        "created": created,
        "preflight_sha256": artifact.sha256,
        "question_count": artifact.payload["question_count"],
        "namespace_count": artifact.payload["namespace_count"],
        "known_would_call_count": artifact.payload["known_would_call_count"],
        "deferred_stage_ids": artifact.payload["deferred_stage_ids"],
        "physical_provider_calls": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
