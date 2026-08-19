from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

import tools.merge_locked_v3_recall as merger
from memory_condense.eval.context_stress import transcript_tokens
from memory_condense.eval.locked_split import load_split_manifest, select_locked_split
from memory_condense.eval.reproducibility import file_sha256, implementation_sha256
from memory_condense.ingest.loader import load_benchmark


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _retrieval() -> dict[str, object]:
    return {
        "allow_selected_scope_fixed_k_closure": True,
        "coverage_selector_prefix_model_id": "prefix/model",
        "coverage_selector_prefix_revision": "prefix-revision",
        "coverage_selector_prefix_checkpoint_sha256": _digest("prefix"),
        "coverage_selector_prefix_device": "cpu",
        "coverage_selector_prefix_dtype": "float32",
        "coverage_selector_prefix_layers": 2,
        "coverage_selector_attention_layer": 1,
        "coverage_selector_choice_model_id": "choice/model",
        "coverage_selector_choice_revision": "choice-revision",
        "coverage_selector_choice_checkpoint_sha256": _digest("choice"),
        "coverage_selector_choice_device": "cpu",
        "coverage_selector_choice_dtype": "float32",
    }


def _plan() -> merger.LockedV3RecallPlan:
    retrieval = _retrieval()
    shards = []
    for offset in merger.FROZEN_OFFSETS:
        questions = tuple(
            merger.ExpectedRecallQuestion(
                question_id=f"q-{offset + index:03d}",
                category="counting" if index % 2 == 0 else "single-session-user",
                evidence_sources=(f"source-{offset + index:03d}",),
            )
            for index in range(10)
        )
        shards.append(merger.ExpectedRecallShard(offset, questions))
    return merger.LockedV3RecallPlan(
        dataset_sha256=_digest("dataset"),
        split_manifest_sha256=_digest("split"),
        policy_manifest_sha256=_digest("policy"),
        implementation_sha256=_digest("implementation"),
        environment_lock_sha256=_digest("environment"),
        selection_artifact_sha256=_digest("selection"),
        retrieval_identity_sha256=merger.canonical_sha256(retrieval),
        retrieval=retrieval,
        protocol=merger.FROZEN_V3_PROTOCOL,
        shards=tuple(shards),
    )


def _blank_row() -> dict[str, str]:
    row = {field: "" for field in merger.CSV_SCHEMA}
    for field in merger._REQUIRED_BINARY_FIELDS:
        row[field] = "0"
    for field in merger._REQUIRED_INT_FIELDS:
        row[field] = "0"
    for field in merger._REQUIRED_FIXED_FLOAT_FIELDS:
        row[field] = "0.0000"
    for field in merger._JSON_FIELDS:
        row[field] = "[]"
    return row


def _csv_row(
    expected: merger.ExpectedRecallQuestion,
    retrieval: dict[str, object],
    *,
    local_index: int,
    global_index: int,
) -> dict[str, str]:
    row = _blank_row()
    row.update(
        {
            "question_id": expected.question_id,
            "category": expected.category,
            "in_haystack": "1",
            "in_context": "1" if global_index % 2 == 0 else "0",
            "best_f1": "0.5000",
            "in_expansions": "1" if global_index % 2 == 0 else "0",
            "context_tokens": "100",
            "evidence_source_recall": "1.0000",
            "all_evidence_sources": "1",
            "retrieved_source_ids": expected.evidence_sources[0],
            "raw_evidence_source_recall": "1.0000",
            "raw_all_evidence_sources": "1",
            "raw_retrieved_source_ids": expected.evidence_sources[0],
            "coverage_selector_allow_selected_scope_fixed_k_closure": "1",
            "coverage_selector_prefix_model_id": str(
                retrieval["coverage_selector_prefix_model_id"]
            ),
            "coverage_selector_prefix_model_revision": str(
                retrieval["coverage_selector_prefix_revision"]
            ),
            "coverage_selector_prefix_checkpoint_sha256": str(
                retrieval["coverage_selector_prefix_checkpoint_sha256"]
            ),
            "coverage_selector_prefix_device": str(
                retrieval["coverage_selector_prefix_device"]
            ),
            "coverage_selector_prefix_dtype": str(
                retrieval["coverage_selector_prefix_dtype"]
            ),
            "coverage_selector_prefix_layers": str(
                retrieval["coverage_selector_prefix_layers"]
            ),
            "coverage_selector_prefix_attention_layer": str(
                retrieval["coverage_selector_attention_layer"]
            ),
        }
    )
    if local_index == 0:
        row.update(
            {
                "answer_value_components_expected": "2",
                "answer_value_components_found": "1",
                "answer_value_component_recall": "0.5000",
                "all_answer_value_components": "0",
                "answer_value_component_hit_mask": "1|0",
                "answer_value_metric_kind": "literal_component",
                "coverage_selector_inspected": "1",
                "coverage_selector_classified": "1",
                "coverage_selector_status": "applied",
                "closure_applied": "1",
                "closure_scope": "selected_scope_policy",
                "closure_global_recall_guaranteed": "0",
                "coverage_selector_score_provider_model_id": str(
                    retrieval["coverage_selector_choice_model_id"]
                ),
                "coverage_selector_score_provider_model_revision": str(
                    retrieval["coverage_selector_choice_revision"]
                ),
                "coverage_selector_score_provider_checkpoint_sha256": str(
                    retrieval["coverage_selector_choice_checkpoint_sha256"]
                ),
                "coverage_selector_score_provider_device": str(
                    retrieval["coverage_selector_choice_device"]
                ),
                "coverage_selector_score_provider_dtype": str(
                    retrieval["coverage_selector_choice_dtype"]
                ),
                "coverage_selector_score_provider_forward_passes": "1",
            }
        )
    else:
        row["coverage_selector_status"] = "bypassed"
        row["coverage_selector_bypass_reason"] = "test_bypass"
    return row


def _payload(
    plan: merger.LockedV3RecallPlan,
    offset: int,
) -> bytes:
    expected = next(shard for shard in plan.shards if shard.sample_offset == offset)
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=merger.CSV_SCHEMA,
        lineterminator="\n",
    )
    writer.writeheader()
    for local_index, question in enumerate(expected.questions):
        writer.writerow(
            _csv_row(
                question,
                dict(plan.retrieval),
                local_index=local_index,
                global_index=offset + local_index,
            )
        )
    return output.getvalue().encode("utf-8")


def _shards(plan: merger.LockedV3RecallPlan) -> list[merger.RecallCsvShard]:
    return [
        merger.RecallCsvShard(offset, f"recall-{offset}.csv", _payload(plan, offset))
        for offset in merger.FROZEN_OFFSETS
    ]


def _mutate_csv(
    payload: bytes,
    *,
    row: int,
    field: str,
    value: str,
) -> bytes:
    records = list(csv.reader(io.StringIO(payload.decode("utf-8"))))
    records[row][records[0].index(field)] = value
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(records)
    return output.getvalue().encode("utf-8")


def test_merge_recomputes_metrics_and_emits_no_sensitive_rows() -> None:
    plan = _plan()

    report = merger.merge_locked_v3_recall(plan, _shards(plan))

    assert report["population"] == {
        "questions": 100,
        "question_ids_sha256": merger.canonical_sha256(
            [f"q-{index:03d}" for index in range(100)]
        ),
        "unique": True,
    }
    metrics = report["metrics"]
    assert metrics["literal"]["haystack_recall"] == 1.0
    assert metrics["literal"]["context_recall"] == 0.5
    assert metrics["literal"]["expansion_recall"] == 0.5
    assert metrics["evidence"]["mean_source_recall"] == 1.0
    assert metrics["raw_evidence"]["mean_source_recall"] == 1.0
    assert metrics["answer_value"] == {
        "scored_questions": 10,
        "components_expected": 20,
        "components_found": 10,
        "component_weighted_recall": 0.5,
        "macro_component_recall": 0.5,
        "all_component_questions": 0,
        "all_component_question_recall": 0.0,
    }
    diagnostics = metrics["selector_diagnostics"]
    assert diagnostics["calls"] == 100
    assert diagnostics["bypasses"] == 90
    assert diagnostics["closure_calls"] == 10
    assert diagnostics["score_provider_forward_passes"] == 10
    assert diagnostics["max_retained_state_bytes"] == 0
    assert metrics["reported_zero_state_consistency"] == {
        "evidence_scope": "self_reported_csv_fields_only",
        "independently_verified": False,
        "questions_checked": 100,
        "score_provider_zero_state_questions": 100,
        "selector_zero_state_questions": 100,
        "retained_state_violation_questions": 0,
        "score_provider_retained_state_bytes_total": 0,
        "selector_retained_state_bytes_total": 0,
        "max_retained_state_bytes": 0,
    }
    assert report["claims"]["reported_zero_retained_state_consistent"] is True
    assert (
        report["claims"]["zero_retained_state_independently_verified"] is False
    )
    assert "zero_retained_state_verified" not in report["claims"]
    assert report["receipt_sha256"] == merger.canonical_sha256(
        {key: value for key, value in report.items() if key != "receipt_sha256"}
    )
    assert report["inputs"][0]["sha256"] == hashlib.sha256(
        _payload(plan, 0)
    ).hexdigest()
    assert report["input_set_sha256"] == merger.canonical_sha256(
        [
            {"sample_offset": row["sample_offset"], "sha256": row["sha256"]}
            for row in report["inputs"]
        ]
    )
    rendered = merger.render_canonical_report(report)
    assert b"coverage_candidate_trace" not in rendered
    assert b"source-000" not in rendered
    assert rendered.endswith(b"\n")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("best_f1", "NaN", "finite four-decimal"),
        ("coverage_candidate_trace", "[NaN]", "non-finite JSON"),
        ("question_id", "substituted", "order/substitution"),
        ("coverage_selector_retained_state_bytes", "1", "retained selector"),
        (
            "coverage_selector_score_provider_retained_state_bytes",
            "1",
            "retained score-provider",
        ),
        ("evidence_source_recall", "0.0000", "not recomputable"),
    ],
)
def test_merge_rejects_invalid_row_values(
    field: str,
    value: str,
    message: str,
) -> None:
    plan = _plan()
    shards = _shards(plan)
    shards[0] = replace(
        shards[0], payload=_mutate_csv(shards[0].payload, row=1, field=field, value=value)
    )

    with pytest.raises(merger.RecallCampaignError, match=message):
        merger.merge_locked_v3_recall(plan, shards)


def test_merge_rejects_extra_schema_rows_and_reordering() -> None:
    plan = _plan()

    extra_schema = _shards(plan)
    records = list(csv.reader(io.StringIO(extra_schema[0].payload.decode("utf-8"))))
    records[0].append("extra")
    for row in records[1:]:
        row.append("value")
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(records)
    extra_schema[0] = replace(extra_schema[0], payload=output.getvalue().encode())
    with pytest.raises(merger.RecallCampaignError, match="non-canonical schema"):
        merger.merge_locked_v3_recall(plan, extra_schema)

    extra_row = _shards(plan)
    extra_row[0] = replace(
        extra_row[0],
        payload=extra_row[0].payload + extra_row[0].payload.splitlines(keepends=True)[1],
    )
    with pytest.raises(merger.RecallCampaignError, match="has 11 rows"):
        merger.merge_locked_v3_recall(plan, extra_row)

    reordered = _shards(plan)
    records = list(csv.reader(io.StringIO(reordered[0].payload.decode("utf-8"))))
    records[1], records[2] = records[2], records[1]
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(records)
    reordered[0] = replace(reordered[0], payload=output.getvalue().encode())
    with pytest.raises(merger.RecallCampaignError, match="order/substitution"):
        merger.merge_locked_v3_recall(plan, reordered)


def test_merge_accepts_candidate_trace_above_default_csv_limit() -> None:
    plan = _plan()
    shards = _shards(plan)
    shards[0] = replace(
        shards[0],
        payload=_mutate_csv(
            shards[0].payload,
            row=1,
            field="coverage_candidate_trace",
            value=json.dumps(["x" * 200_000], separators=(",", ":")),
        ),
    )

    report = merger.merge_locked_v3_recall(plan, shards)

    assert report["population"]["questions"] == 100


def test_merge_requires_exact_offsets_and_unique_plan_population() -> None:
    plan = _plan()
    shards = _shards(plan)
    with pytest.raises(merger.RecallCampaignError, match="duplicate recall CSV"):
        merger.merge_locked_v3_recall(plan, [*shards, shards[0]])
    with pytest.raises(merger.RecallCampaignError, match=r"missing=\[90\]"):
        merger.merge_locked_v3_recall(plan, shards[:-1])
    substituted_offset = [*shards[:-1], replace(shards[-1], sample_offset=100)]
    with pytest.raises(
        merger.RecallCampaignError,
        match=r"missing=\[90\], extra=\[100\]",
    ):
        merger.merge_locked_v3_recall(plan, substituted_offset)

    first, second, *rest = plan.shards
    duplicate_question = replace(
        second.questions[0], question_id=first.questions[0].question_id
    )
    duplicate_shard = replace(
        second, questions=(duplicate_question, *second.questions[1:])
    )
    duplicate_plan = replace(plan, shards=(first, duplicate_shard, *rest))
    with pytest.raises(merger.RecallCampaignError, match="duplicate question population"):
        merger.merge_locked_v3_recall(duplicate_plan, shards)


def test_save_is_canonical_no_clobber_and_writes_checksum(tmp_path: Path) -> None:
    report = merger.merge_locked_v3_recall(_plan(), _shards(_plan()))
    output = tmp_path / "campaign.json"

    report_path, checksum_path, digest = merger.save_locked_v3_recall_report(
        report, output
    )

    assert report_path.read_bytes() == merger.render_canonical_report(report)
    assert digest == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert checksum_path.read_text(encoding="ascii") == (
        f"{digest}  campaign.json\n"
    )
    before = report_path.read_bytes(), checksum_path.read_bytes()
    with pytest.raises(merger.RecallCampaignError, match="refusing to clobber"):
        merger.save_locked_v3_recall_report(report, output)
    assert (report_path.read_bytes(), checksum_path.read_bytes()) == before


def test_build_plan_derives_population_from_hashed_dataset_and_split(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "frozen"
    repository.mkdir()
    source_root = repository / "src" / "memory_condense"
    source_root.mkdir(parents=True)
    (source_root / "frozen.py").write_text("FROZEN = True\n", encoding="utf-8")
    environment = repository / "pixi.lock"
    environment.write_text("frozen environment\n", encoding="utf-8")
    selection = repository / "selection.json"
    selection.write_text('{"selected":true}\n', encoding="utf-8")

    records = [
        {
            "question_id": f"locked-{index:03d}",
            "question_type": "single-session-user",
            "question": f"What was marker {index}?",
            "answer": f"SECRET-{index}",
            "haystack_session_ids": [f"session-{index}"],
            "haystack_sessions": [
                [{"role": "user", "content": "repeatable token payload"}]
            ],
            "answer_session_ids": [f"session-{index}"],
        }
        for index in range(100)
    ]
    dataset = repository / "dataset.json"
    dataset.write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")
    split = repository / "split.json"
    split.write_text(
        json.dumps(
            {
                "format": "memory-condense-locked-benchmark-split-v1",
                "dataset_sha256": file_sha256(dataset),
                "salt": "recall-merger-test",
                "splits": {"validation": 100},
                "algorithm": "stratified-largest-remainder-v1",
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    samples = load_benchmark(dataset, "longmemeval")
    stress_tokens = transcript_tokens(samples[0]) * 10
    protocol = replace(merger.FROZEN_V3_PROTOCOL, stress_context_tokens=stress_tokens)
    retrieval = _retrieval()
    policy = (
        repository
        / "longmemeval-qwen-choice-coverage-operational-validation-v3.json"
    )
    policy.write_text(
        json.dumps(
            {
                "format": "memory-condense-retrieval-policy-v1",
                "status": "validation_frozen",
                "split": "validation",
                "dataset_sha256": file_sha256(dataset),
                "split_manifest": split.name,
                "split_manifest_sha256": file_sha256(split),
                "selection_artifact": selection.name,
                "selection_artifact_sha256": file_sha256(selection),
                "selection_artifact_required": True,
                "implementation_sha256": implementation_sha256(source_root),
                "environment_lock_sha256": file_sha256(environment),
                "retrieval": retrieval,
                "evaluation": {
                    "benchmark_format": "longmemeval",
                    "stress_context_tokens": stress_tokens,
                    "stress_questions": 10,
                    "stress_question_offset": 0,
                    "max_samples": 1,
                    "min_target_questions": 100,
                    "sample_offsets": list(merger.FROZEN_OFFSETS),
                },
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    anchors = merger.FrozenV3Anchors(
        dataset_sha256=file_sha256(dataset),
        split_manifest_sha256=file_sha256(split),
        policy_manifest_sha256=file_sha256(policy),
        implementation_sha256=implementation_sha256(source_root),
        environment_lock_sha256=file_sha256(environment),
        selection_artifact_sha256=file_sha256(selection),
    )

    plan = merger.build_locked_v3_recall_plan(
        dataset=dataset,
        split_manifest=split,
        policy_manifest=policy,
        frozen_source_root=source_root,
        environment_lock=environment,
        frozen_repository_root=repository,
        anchors=anchors,
        protocol=protocol,
    )

    manifest = load_split_manifest(split)
    validation = select_locked_split(
        samples,
        dataset_path=dataset,
        manifest=manifest,
        split="validation",
    )
    expected_ids = [sample.sample_id for sample in validation]
    actual_ids = [
        question.question_id for shard in plan.shards for question in shard.questions
    ]
    assert actual_ids == expected_ids
    assert len(actual_ids) == len(set(actual_ids)) == 100
    assert "SECRET-" not in repr(plan)
    assert plan.dataset_sha256 == hashlib.sha256(dataset.read_bytes()).hexdigest()
