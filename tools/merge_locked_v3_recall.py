"""Merge the ten frozen-v3 provider-free answer-recall CSV shards.

This is deliberately separate from the scored benchmark campaign merger.  It
reconstructs the locked validation population from the exact dataset and split
bytes, validates the historical v3 CSV wire format, and reports retrieval
reachability only.  It never reads or emits held-out answers or candidate text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.eval.context_stress import compose_context_stress_sample
from memory_condense.eval.locked_split import load_split_manifest, select_locked_split
from memory_condense.eval.reproducibility import file_sha256, implementation_sha256
from memory_condense.ingest.loader import load_benchmark

if __package__:
    from . import merge_locked_v3_recall_json as _json_support
    from . import merge_locked_v3_recall_rows as _rows
    from . import merge_locked_v3_recall_schema as _schema
else:  # Preserve ``python tools/merge_locked_v3_recall.py``.
    import merge_locked_v3_recall_json as _json_support
    import merge_locked_v3_recall_rows as _rows
    import merge_locked_v3_recall_schema as _schema


REPORT_FORMAT = _schema.REPORT_FORMAT
FROZEN_OFFSETS = _schema.FROZEN_OFFSETS
MAX_CSV_FIELD_CHARS = _schema.MAX_CSV_FIELD_CHARS
RecallCampaignError = _schema.RecallCampaignError
FrozenV3Anchors = _schema.FrozenV3Anchors
FROZEN_V3_ANCHORS = _schema.FROZEN_V3_ANCHORS
RecallProtocol = _schema.RecallProtocol
FROZEN_V3_PROTOCOL = _schema.FROZEN_V3_PROTOCOL
ExpectedRecallQuestion = _schema.ExpectedRecallQuestion
ExpectedRecallShard = _schema.ExpectedRecallShard
SourceRecheck = _schema.SourceRecheck
LockedV3RecallPlan = _schema.LockedV3RecallPlan
RecallCsvShard = _schema.RecallCsvShard
CSV_SCHEMA = _schema.CSV_SCHEMA
_REQUIRED_BINARY_FIELDS = _schema._REQUIRED_BINARY_FIELDS
_OPTIONAL_BINARY_FIELDS = _schema._OPTIONAL_BINARY_FIELDS
_REQUIRED_INT_FIELDS = _schema._REQUIRED_INT_FIELDS
_OPTIONAL_INT_FIELDS = _schema._OPTIONAL_INT_FIELDS
_REQUIRED_FIXED_FLOAT_FIELDS = _schema._REQUIRED_FIXED_FLOAT_FIELDS
_OPTIONAL_FIXED_FLOAT_FIELDS = _schema._OPTIONAL_FIXED_FLOAT_FIELDS
_JSON_FIELDS = _schema._JSON_FIELDS
_PIPE_FIELDS = _schema._PIPE_FIELDS
_TEXT_FIELDS = _schema._TEXT_FIELDS
_SHA256_RE = _schema._SHA256_RE
_NONNEGATIVE_INT_RE = _schema._NONNEGATIVE_INT_RE
_FIXED_FLOAT_RE = _schema._FIXED_FLOAT_RE

_canonical_json_bytes = _json_support._canonical_json_bytes
canonical_sha256 = _json_support.canonical_sha256
_reject_json_constant = _json_support._reject_json_constant
_unique_object = _json_support._unique_object
_strict_json = _json_support._strict_json
_assert_finite_json = _json_support._assert_finite_json
_require_object = _json_support._require_object
_require_sha256 = _json_support._require_sha256
_safe_relative_file = _json_support._safe_relative_file

_required_integer = _rows._required_integer
_optional_integer = _rows._optional_integer
_required_binary = _rows._required_binary
_optional_binary = _rows._optional_binary
_required_fixed_float = _rows._required_fixed_float
_optional_fixed_float = _rows._optional_fixed_float
_pipe_values = _rows._pipe_values
_parse_csv_row = _rows._parse_csv_row
_same_four_decimals = _rows._same_four_decimals
_validate_answer_value = _rows._validate_answer_value
_validate_retrieval_identity = _rows._validate_retrieval_identity
_validate_question_row = _rows._validate_question_row
_parse_shard = _rows._parse_shard
_mean = _rows._mean
_rate = _rows._rate
_aggregate_rows = _rows._aggregate_rows
_category_metrics = _rows._category_metrics




def _file_digest(path: Path, label: str) -> str:
    try:
        return file_sha256(path)
    except OSError as exc:
        raise RecallCampaignError(f"cannot hash {label}: {exc}") from exc




def _validate_protocol(protocol: RecallProtocol) -> None:
    if protocol.split != "validation":
        raise RecallCampaignError("locked recall protocol must use validation")
    if protocol.sample_offsets != FROZEN_OFFSETS:
        raise RecallCampaignError("locked recall offsets must be 0,10,...,90")
    if protocol.questions_per_shard != 10:
        raise RecallCampaignError("locked recall shards must contain ten questions")
    if protocol.population_questions != 100:
        raise RecallCampaignError("locked recall population must contain 100 questions")
    if protocol.stress_context_tokens < 1:
        raise RecallCampaignError("stress_context_tokens must be positive")


def _validate_policy(
    policy: dict[str, Any],
    *,
    policy_path: Path,
    split_path: Path,
    anchors: FrozenV3Anchors,
    protocol: RecallProtocol,
) -> dict[str, Any]:
    expected_top = {
        "format": "memory-condense-retrieval-policy-v1",
        "status": "validation_frozen",
        "split": protocol.split,
        "dataset_sha256": anchors.dataset_sha256,
        "split_manifest_sha256": anchors.split_manifest_sha256,
        "implementation_sha256": anchors.implementation_sha256,
        "environment_lock_sha256": anchors.environment_lock_sha256,
        "selection_artifact_sha256": anchors.selection_artifact_sha256,
    }
    for key, expected in expected_top.items():
        if policy.get(key) != expected:
            raise RecallCampaignError(f"frozen policy {key} mismatch")
    if policy.get("split_manifest") != split_path.name:
        raise RecallCampaignError("frozen policy split_manifest filename mismatch")
    if policy.get("selection_artifact_required") is not True:
        raise RecallCampaignError("frozen policy must require its selection artifact")
    retrieval = _require_object(policy.get("retrieval"), "policy.retrieval")
    if not retrieval:
        raise RecallCampaignError("policy.retrieval must not be empty")
    evaluation = _require_object(policy.get("evaluation"), "policy.evaluation")
    expected_evaluation = {
        "benchmark_format": "longmemeval",
        "stress_context_tokens": protocol.stress_context_tokens,
        "stress_questions": protocol.questions_per_shard,
        "stress_question_offset": 0,
        "max_samples": 1,
        "min_target_questions": protocol.population_questions,
        "sample_offsets": list(protocol.sample_offsets),
    }
    for key, expected in expected_evaluation.items():
        if evaluation.get(key) != expected:
            raise RecallCampaignError(f"frozen policy evaluation.{key} mismatch")
    if policy_path.name != "longmemeval-qwen-choice-coverage-operational-validation-v3.json":
        raise RecallCampaignError("unexpected frozen-v3 policy filename")
    return retrieval


def build_locked_v3_recall_plan(
    *,
    dataset: str | Path,
    split_manifest: str | Path,
    policy_manifest: str | Path,
    frozen_source_root: str | Path,
    environment_lock: str | Path,
    frozen_repository_root: str | Path,
    anchors: FrozenV3Anchors = FROZEN_V3_ANCHORS,
    protocol: RecallProtocol = FROZEN_V3_PROTOCOL,
) -> LockedV3RecallPlan:
    """Reconstruct the exact validation question population without answers."""

    _validate_protocol(protocol)
    dataset_path = Path(dataset).resolve()
    split_path = Path(split_manifest).resolve()
    policy_path = Path(policy_manifest).resolve()
    source_root = Path(frozen_source_root).resolve()
    environment_path = Path(environment_lock).resolve()
    repository_root = Path(frozen_repository_root).resolve()
    for path, label in (
        (dataset_path, "dataset"),
        (split_path, "split manifest"),
        (policy_path, "policy manifest"),
        (environment_path, "environment lock"),
    ):
        if not path.is_file():
            raise RecallCampaignError(f"{label} does not exist: {path}")
    if not source_root.is_dir():
        raise RecallCampaignError(f"frozen source root does not exist: {source_root}")
    if not repository_root.is_dir():
        raise RecallCampaignError(
            f"frozen repository root does not exist: {repository_root}"
        )

    dataset_digest = _file_digest(dataset_path, "dataset")
    split_digest = _file_digest(split_path, "split manifest")
    policy_payload = policy_path.read_bytes()
    policy_digest = hashlib.sha256(policy_payload).hexdigest()
    environment_digest = _file_digest(environment_path, "environment lock")
    source_digest = implementation_sha256(source_root)
    actual_anchors = FrozenV3Anchors(
        dataset_sha256=dataset_digest,
        split_manifest_sha256=split_digest,
        policy_manifest_sha256=policy_digest,
        implementation_sha256=source_digest,
        environment_lock_sha256=environment_digest,
        selection_artifact_sha256=anchors.selection_artifact_sha256,
    )
    for field_name in (
        "dataset_sha256",
        "split_manifest_sha256",
        "policy_manifest_sha256",
        "implementation_sha256",
        "environment_lock_sha256",
    ):
        if getattr(actual_anchors, field_name) != getattr(anchors, field_name):
            raise RecallCampaignError(f"frozen {field_name} mismatch")

    policy = _require_object(_strict_json(policy_payload, "policy manifest"), "policy")
    retrieval = _validate_policy(
        policy,
        policy_path=policy_path,
        split_path=split_path,
        anchors=anchors,
        protocol=protocol,
    )
    selection_path = _safe_relative_file(
        repository_root,
        policy.get("selection_artifact"),
        "policy.selection_artifact",
    )
    selection_digest = _file_digest(selection_path, "selection artifact")
    if selection_digest != anchors.selection_artifact_sha256:
        raise RecallCampaignError("frozen selection artifact SHA-256 mismatch")

    try:
        samples = load_benchmark(dataset_path, "longmemeval")
        manifest = load_split_manifest(split_path)
        validation = select_locked_split(
            samples,
            dataset_path=dataset_path,
            manifest=manifest,
            split=protocol.split,
        )
    except (OSError, ValueError) as exc:
        raise RecallCampaignError(
            f"cannot reconstruct locked validation population: {exc}"
        ) from exc

    shards: list[ExpectedRecallShard] = []
    population_ids: list[str] = []
    for offset in protocol.sample_offsets:
        if offset >= len(validation):
            raise RecallCampaignError(
                f"validation sample offset {offset} is outside the locked split"
            )
        try:
            stress = compose_context_stress_sample(
                validation[offset:],
                target_tokens=protocol.stress_context_tokens,
                max_questions=protocol.questions_per_shard,
                question_offset=0,
            )
        except ValueError as exc:
            raise RecallCampaignError(
                f"cannot reconstruct recall shard {offset}: {exc}"
            ) from exc
        if len(stress.questions) != protocol.questions_per_shard:
            raise RecallCampaignError(
                f"recall shard {offset} has {len(stress.questions)} questions; "
                f"expected {protocol.questions_per_shard}"
            )
        questions: list[ExpectedRecallQuestion] = []
        for question in stress.questions:
            question_id = str(question.question_id).strip()
            if not question_id:
                raise RecallCampaignError("locked validation question ID is empty")
            evidence_sources = tuple(str(value) for value in question.evidence_sources)
            if (
                not evidence_sources
                or any(not value.strip() or "|" in value for value in evidence_sources)
                or len(evidence_sources) != len(set(evidence_sources))
            ):
                raise RecallCampaignError(
                    f"question {question_id!r} has invalid evidence-source labels"
                )
            questions.append(
                ExpectedRecallQuestion(
                    question_id=question_id,
                    category=question.category or "",
                    evidence_sources=evidence_sources,
                )
            )
            population_ids.append(question_id)
        shards.append(ExpectedRecallShard(offset, tuple(questions)))

    full_validation_ids = [
        str(question.question_id)
        for sample in validation
        for question in sample.questions
    ]
    if len(population_ids) != protocol.population_questions:
        raise RecallCampaignError("locked recall population size mismatch")
    if len(population_ids) != len(set(population_ids)):
        raise RecallCampaignError("locked recall population repeats question IDs")
    if set(population_ids) != set(full_validation_ids):
        raise RecallCampaignError(
            "locked recall shards do not cover the exact validation population"
        )

    # Recheck files consumed more than once so a concurrent mutation cannot be
    # hidden behind an earlier digest.
    rechecks = (
        SourceRecheck("dataset", dataset_path, dataset_digest),
        SourceRecheck("split manifest", split_path, split_digest),
        SourceRecheck("policy manifest", policy_path, policy_digest),
        SourceRecheck("environment lock", environment_path, environment_digest),
        SourceRecheck("selection artifact", selection_path, selection_digest),
        SourceRecheck("frozen implementation", source_root, source_digest, "source"),
    )
    plan = LockedV3RecallPlan(
        dataset_sha256=dataset_digest,
        split_manifest_sha256=split_digest,
        policy_manifest_sha256=policy_digest,
        implementation_sha256=source_digest,
        environment_lock_sha256=environment_digest,
        selection_artifact_sha256=selection_digest,
        retrieval_identity_sha256=canonical_sha256(retrieval),
        retrieval=json.loads(_canonical_json_bytes(retrieval)),
        protocol=protocol,
        shards=tuple(shards),
        source_rechecks=rechecks,
    )
    assert_locked_v3_sources_unchanged(plan)
    return plan


def assert_locked_v3_sources_unchanged(plan: LockedV3RecallPlan) -> None:
    for item in plan.source_rechecks:
        try:
            digest = (
                implementation_sha256(item.path)
                if item.kind == "source"
                else file_sha256(item.path)
            )
        except OSError as exc:
            raise RecallCampaignError(
                f"cannot recheck {item.label}: {exc}"
            ) from exc
        if digest != item.sha256:
            raise RecallCampaignError(f"{item.label} changed during recall merge")


def _validate_plan(plan: LockedV3RecallPlan) -> dict[int, ExpectedRecallShard]:
    _validate_protocol(plan.protocol)
    identities = (
        plan.dataset_sha256,
        plan.split_manifest_sha256,
        plan.policy_manifest_sha256,
        plan.implementation_sha256,
        plan.environment_lock_sha256,
        plan.selection_artifact_sha256,
        plan.retrieval_identity_sha256,
    )
    if any(_SHA256_RE.fullmatch(value) is None for value in identities):
        raise RecallCampaignError("plan contains a malformed identity digest")
    by_offset: dict[int, ExpectedRecallShard] = {}
    ids: list[str] = []
    for shard in plan.shards:
        if shard.sample_offset in by_offset:
            raise RecallCampaignError(
                f"plan repeats sample_offset {shard.sample_offset}"
            )
        by_offset[shard.sample_offset] = shard
        if len(shard.questions) != plan.protocol.questions_per_shard:
            raise RecallCampaignError(
                f"plan shard {shard.sample_offset} does not contain ten questions"
            )
        for question in shard.questions:
            if not question.question_id.strip():
                raise RecallCampaignError("plan contains an empty question ID")
            if not question.evidence_sources:
                raise RecallCampaignError(
                    f"plan question {question.question_id!r} lacks evidence sources"
                )
            if any(
                not source.strip() or "|" in source
                for source in question.evidence_sources
            ):
                raise RecallCampaignError(
                    f"plan question {question.question_id!r} has an invalid "
                    "evidence source"
                )
            ids.append(question.question_id)
    if tuple(sorted(by_offset)) != plan.protocol.sample_offsets:
        raise RecallCampaignError("plan offsets are not exactly 0,10,...,90")
    if len(ids) != plan.protocol.population_questions:
        raise RecallCampaignError("plan does not contain exactly 100 questions")
    if len(ids) != len(set(ids)):
        raise RecallCampaignError("plan contains a duplicate question population")
    if canonical_sha256(plan.retrieval) != plan.retrieval_identity_sha256:
        raise RecallCampaignError("plan retrieval identity does not match its payload")
    return by_offset




def merge_locked_v3_recall(
    plan: LockedV3RecallPlan,
    shards: Sequence[RecallCsvShard],
) -> dict[str, Any]:
    """Purely validate in-memory CSV bytes and return a canonical report value."""

    expected_by_offset = _validate_plan(plan)
    supplied: dict[int, RecallCsvShard] = {}
    for shard in shards:
        if shard.sample_offset in supplied:
            raise RecallCampaignError(
                f"duplicate recall CSV sample_offset {shard.sample_offset}"
            )
        supplied[shard.sample_offset] = shard
    if tuple(sorted(supplied)) != plan.protocol.sample_offsets:
        missing = sorted(set(plan.protocol.sample_offsets) - set(supplied))
        extra = sorted(set(supplied) - set(plan.protocol.sample_offsets))
        raise RecallCampaignError(
            f"recall CSV offsets must be exactly 0,10,...,90; missing={missing}, extra={extra}"
        )

    all_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    shard_reports: list[dict[str, Any]] = []
    population_ids: list[str] = []
    for offset in plan.protocol.sample_offsets:
        parsed, digest, question_ids_digest = _parse_shard(
            supplied[offset], expected_by_offset[offset], plan.retrieval
        )
        ids = [row["question_id"] for row in parsed]
        population_ids.extend(ids)
        all_rows.extend(parsed)
        portable_name = Path(supplied[offset].name).name
        if not portable_name:
            raise RecallCampaignError(f"recall CSV at offset {offset} has no name")
        input_rows.append(
            {
                "sample_offset": offset,
                "name": portable_name,
                "sha256": digest,
                "rows": len(parsed),
                "question_ids_sha256": question_ids_digest,
            }
        )
        shard_reports.append(
            {
                "sample_offset": offset,
                "question_ids_sha256": question_ids_digest,
                "metrics": _aggregate_rows(parsed),
            }
        )
    if len(population_ids) != len(set(population_ids)):
        raise RecallCampaignError("recall CSVs contain a duplicate question population")
    expected_population = [
        question.question_id
        for offset in plan.protocol.sample_offsets
        for question in expected_by_offset[offset].questions
    ]
    if population_ids != expected_population:
        raise RecallCampaignError("recall CSV population order is not locked")

    body: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "status": "provider_free_recall_only_not_answer_accuracy",
        "identities": {
            "dataset_sha256": plan.dataset_sha256,
            "split_manifest_sha256": plan.split_manifest_sha256,
            "policy_manifest_sha256": plan.policy_manifest_sha256,
            "implementation_sha256": plan.implementation_sha256,
            "environment_lock_sha256": plan.environment_lock_sha256,
            "selection_artifact_sha256": plan.selection_artifact_sha256,
            "retrieval_identity_sha256": plan.retrieval_identity_sha256,
            "csv_schema_sha256": canonical_sha256(list(CSV_SCHEMA)),
        },
        "protocol": {
            "split": plan.protocol.split,
            "sample_offsets": list(plan.protocol.sample_offsets),
            "questions_per_shard": plan.protocol.questions_per_shard,
            "population_questions": plan.protocol.population_questions,
            "stress_context_tokens": plan.protocol.stress_context_tokens,
            "provider_calls": 0,
        },
        "population": {
            "questions": len(population_ids),
            "question_ids_sha256": canonical_sha256(population_ids),
            "unique": len(population_ids) == len(set(population_ids)),
        },
        "inputs": input_rows,
        "input_set_sha256": canonical_sha256(
            [
                {"sample_offset": row["sample_offset"], "sha256": row["sha256"]}
                for row in input_rows
            ]
        ),
        "metrics": _aggregate_rows(all_rows),
        "by_category": _category_metrics(all_rows),
        "shards": shard_reports,
        "claims": {
            "answer_accuracy_scored": False,
            "provider_free": True,
            "held_out_answers_emitted": False,
            "candidate_trace_emitted": False,
            "reported_zero_retained_state_consistent": True,
            "zero_retained_state_independently_verified": False,
        },
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def render_canonical_report(report: Mapping[str, Any]) -> bytes:
    """Render the report as canonical JSON with one terminating newline."""

    return _canonical_json_bytes(dict(report)) + b"\n"


def _write_temp(parent: Path, prefix: str, payload: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix, dir=parent)
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def save_locked_v3_recall_report(
    report: Mapping[str, Any], output: str | Path
) -> tuple[Path, Path, str]:
    """Atomically create canonical JSON and its checksum without clobbering."""

    output_path = Path(output)
    checksum_path = output_path.parent / f"{output_path.name}.sha256"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or checksum_path.exists():
        raise RecallCampaignError("output or checksum already exists; refusing to clobber")
    payload = render_canonical_report(report)
    digest = hashlib.sha256(payload).hexdigest()
    checksum = f"{digest}  {output_path.name}\n".encode("ascii")
    json_temp = _write_temp(output_path.parent, f".{output_path.name}.", payload)
    checksum_temp = _write_temp(
        output_path.parent, f".{checksum_path.name}.", checksum
    )
    linked: list[Path] = []
    try:
        os.link(json_temp, output_path)
        linked.append(output_path)
        os.link(checksum_temp, checksum_path)
        linked.append(checksum_path)
    except FileExistsError as exc:
        for path in reversed(linked):
            path.unlink(missing_ok=True)
        raise RecallCampaignError(
            "output or checksum appeared during publication; refusing to clobber"
        ) from exc
    except BaseException:
        for path in reversed(linked):
            path.unlink(missing_ok=True)
        raise
    finally:
        json_temp.unlink(missing_ok=True)
        checksum_temp.unlink(missing_ok=True)
    return output_path, checksum_path, digest


def _parse_shard_argument(value: str) -> tuple[int, Path]:
    raw_offset, separator, raw_path = value.partition("=")
    if not separator or not raw_path:
        raise argparse.ArgumentTypeError("--shard must be OFFSET=CSV_PATH")
    if _NONNEGATIVE_INT_RE.fullmatch(raw_offset) is None:
        raise argparse.ArgumentTypeError("--shard OFFSET must be a non-negative integer")
    return int(raw_offset), Path(raw_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge the ten provider-free frozen-v3 answer-recall CSV shards"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--frozen-repository-root", type=Path, required=True)
    parser.add_argument(
        "--shard",
        action="append",
        type=_parse_shard_argument,
        required=True,
        metavar="OFFSET=CSV_PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_locked_v3_recall_plan(
            dataset=args.dataset,
            split_manifest=args.split_manifest,
            policy_manifest=args.policy_manifest,
            frozen_source_root=args.frozen_source_root,
            environment_lock=args.environment_lock,
            frozen_repository_root=args.frozen_repository_root,
        )
        inputs: list[RecallCsvShard] = []
        seen_paths: set[Path] = set()
        for offset, path in args.shard:
            resolved = path.resolve()
            if resolved in seen_paths:
                raise RecallCampaignError(f"recall CSV path repeated: {resolved}")
            seen_paths.add(resolved)
            try:
                payload = resolved.read_bytes()
            except OSError as exc:
                raise RecallCampaignError(
                    f"cannot read recall CSV {resolved}: {exc}"
                ) from exc
            inputs.append(RecallCsvShard(offset, resolved.name, payload))
        report = merge_locked_v3_recall(plan, inputs)
        assert_locked_v3_sources_unchanged(plan)
        output, checksum, digest = save_locked_v3_recall_report(
            report, args.output
        )
    except (RecallCampaignError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"merged {report['population']['questions']} recall questions; "
        f"saved {output} and {checksum} ({digest})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
