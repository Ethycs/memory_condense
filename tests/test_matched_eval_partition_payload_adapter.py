from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tests.test_matched_eval_closure_live import _parent_plane
from tests.test_matched_eval_partition_scan_v2 import _sha, _write_store
from tests.test_matched_eval_query_expansion import _population
from tools import run_locked_query_answer_judge as judge_cli
from tools import run_locked_partition_payload_answers as cli
from tools.matched_eval.artifacts import publish_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.partition_payload_adapter import (
    DELTA_TIER,
    PartitionPayloadAdapterError,
    load_partition_payload_adapter,
)
from tools.matched_eval.partition_scan_v2 import (
    PartitionScanV2Generation,
    construct_partition_scan_v2_question,
)
from tools.matched_eval.query_payload_live import (
    build_query_payload_answer_plan,
    preflight_query_payload_answers,
)
from memory_condense.persistence.db import Database


def _sealed_arm(tmp_path: Path, *, eligible: bool = True):
    source, _namespace, _query_population = _population(tmp_path)
    protected = source.rows[0].packet.protected_evidence[0]
    database = _write_store(
        tmp_path / "partition.db",
        [
            (protected.source_id, protected.text),
            (
                "unrelated-history::episode-7",
                "Choice 0 was green yesterday, according to the later record.",
            ),
        ],
    )
    with Database(database, read_only=True) as store:
        question = construct_partition_scan_v2_question(
            store,
            ordinal=0,
            shard_offset=0,
            packet=source.rows[0].packet,
            eligible=eligible,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=_sha("store"),
        )
    generation = PartitionScanV2Generation(
        retrieval_sha256=source.retrieval_sha256,
        eligibility_manifest_sha256=_sha("eligibility"),
        population_identity_sha256=source.snapshot.population_identity_sha256,
        questions=(question,),
    )
    artifact, _created = publish_sealed_json(
        tmp_path / "generation.json",
        generation.projection(),
    )
    adapter = load_partition_payload_adapter(
        tmp_path / "retrieval.json",
        generation_path=artifact.path,
        expected_retrieval_sha256=source.retrieval_sha256,
        expected_source_population_id=source.population_id,
        expected_generation_sha256=artifact.sha256,
        expected_eligibility_manifest_sha256=_sha("eligibility"),
        expected_question_count=1,
    )
    return source, question, artifact, adapter


def test_partition_adapter_preserves_exact_post_selection_delta_and_shared_plane(
    tmp_path: Path,
) -> None:
    source, question, artifact, adapter = _sealed_arm(tmp_path)
    row = adapter.rows[0]

    assert adapter.query_preflight_sha256 == artifact.sha256
    assert adapter.query_run_sha256 == artifact.sha256
    assert row.selected_before_dedup_ids == question.trace.selected_before_dedup_ids
    assert row.dedup_excluded_ids == question.trace.dedup_excluded_ids
    assert row.admitted_ids == question.trace.admitted_ids
    assert len(row.dedup_excluded_ids) == 1
    assert len(row.admitted_delta) == 1
    assert row.admitted_delta[0].source_id == "unrelated-history::episode-7"
    assert row.question.stage_ids == (
        source.rows[0].packet.stage_id,
        "partition_scan_balanced_source_additions",
    )

    plan = build_query_payload_answer_plan(
        adapter,
        _parent_plane(source),
        delta_tier=DELTA_TIER,
    )
    packed = plan.rows[0]
    assert plan.required_calls == 1
    assert packed.messages is not None
    prompt = "\n".join(message.content for message in packed.messages)
    assert DELTA_TIER in prompt
    assert "Choice 0 was blue." in prompt
    assert "Choice 0 was green yesterday" in prompt
    assert all(alias.tier == DELTA_TIER for alias in packed.aliases[1:])

    preflight = preflight_query_payload_answers(
        plan,
        output_root=tmp_path / "answer",
    )
    assert preflight.payload["provider_calls"] == 0
    assert preflight.payload["gold_loaded"] is False
    assert preflight.payload["required_authorized_provider_calls"] == 1
    assert preflight.payload["retained_request_token_state_bytes"] == 0
    assert preflight.payload["output_token_reserve"] == 256


def test_partition_adapter_rejects_resealed_false_s0_dedup_alias(
    tmp_path: Path,
) -> None:
    source, question, artifact, _adapter = _sealed_arm(tmp_path)
    assert question.trace.dedup_excluded_ids
    changed = copy.deepcopy(artifact.payload)
    raw = changed["questions"][0]
    raw["dedup_alias_bindings"][0][1] = "unknown-protected-evidence"
    row_body = dict(raw)
    row_body.pop("question_identity_sha256")
    raw["question_identity_sha256"] = identity_sha256(row_body)
    generation_body = dict(changed)
    generation_body.pop("artifact_identity_sha256")
    changed["artifact_identity_sha256"] = identity_sha256(generation_body)
    tampered, _created = publish_sealed_json(
        tmp_path / "tampered-generation.json",
        changed,
    )

    with pytest.raises(
        PartitionPayloadAdapterError,
        match="exact protected-S0 binding",
    ):
        load_partition_payload_adapter(
            tmp_path / "retrieval.json",
            generation_path=tampered.path,
            expected_retrieval_sha256=source.retrieval_sha256,
            expected_source_population_id=source.population_id,
            expected_generation_sha256=tampered.sha256,
            expected_eligibility_manifest_sha256=_sha("eligibility"),
            expected_question_count=1,
        )


def test_ineligible_partition_row_is_exact_parent_fallback(tmp_path: Path) -> None:
    source, _question, _artifact, adapter = _sealed_arm(tmp_path, eligible=False)
    plan = build_query_payload_answer_plan(
        adapter,
        _parent_plane(source),
        delta_tier=DELTA_TIER,
    )

    assert plan.required_calls == 0
    assert plan.rows[0].submitted is False
    assert plan.rows[0].reason == "no_usable_adapter_delta"
    assert plan.rows[0].parent.prediction == plan.parent_plane.rows[0].prediction


def test_partition_runner_defaults_are_locked_and_output_isolated() -> None:
    args = cli._parser().parse_args(["preflight"])

    assert args.generation == cli.DEFAULT_GENERATION
    assert args.expected_generation_sha256 == cli.EXPECTED_GENERATION_SHA256
    assert args.expected_eligibility_sha256 == cli.EXPECTED_ELIGIBILITY_SHA256
    assert args.output_root == cli.DEFAULT_OUTPUT
    assert args.output_root != cli.DEFAULT_PARTITION_ROOT
    assert args.output_root != cli.DEFAULT_PARENT_ROOT


def test_consolidated_judge_cli_exposes_partition_payload_plane() -> None:
    args = judge_cli._parser().parse_args(
        [
            "preflight",
            "--arm",
            "partition-payload",
            "--expected-answer-preflight-sha256",
            "a" * 64,
            "--expected-answer-run-sha256",
            "b" * 64,
        ]
    )
    plan_args = judge_cli._plan_namespace(args)

    assert judge_cli._answer_root(args) == cli.DEFAULT_OUTPUT
    assert plan_args.generation == cli.DEFAULT_GENERATION
    assert plan_args.expected_generation_sha256 == cli.EXPECTED_GENERATION_SHA256
    assert plan_args.expected_eligibility_sha256 == cli.EXPECTED_ELIGIBILITY_SHA256
