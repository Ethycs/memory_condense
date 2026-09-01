from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.persistence.db import Database

from tests.test_matched_eval_query_expansion import _population
from tests.test_matched_eval_query_guided_scan import _write_store
from tests.test_matched_eval_query_payload_live import _plan
from tools import run_locked_query_guided_payload_answers as cli
from tools import run_locked_query_answer_judge as judge_cli
from tools.matched_eval.artifacts import publish_sealed_json
from tools.matched_eval.query_guided_payload_adapter import (
    DELTA_TIER,
    QueryGuidedPayloadAdapterError,
    _candidate_from_projection,
    _load_exact_artifact_payload,
    _project_row,
    _verify_replay_file,
)
from tools.matched_eval.query_guided_scan import (
    QueryGuidedScanBudget,
    _construct_row,
    cache_namespace_partitions,
)
from tools.matched_eval.query_payload_live import build_query_payload_answer_plan
from tools.matched_eval.query_expansion import PartitionRoutingReceipt


def _guided_row(tmp_path: Path):
    source, _old_namespace, _query_population = _population(tmp_path)
    database_path, namespace = _write_store(tmp_path / "guided-memory.db")
    with Database(database_path, read_only=True) as database:
        cache = cache_namespace_partitions(
            database,
            namespace,
            source_database_sha256="d" * 64,
            source_store_receipt_sha256=namespace.combined_store_receipt_sha256,
        )
    query = "orchid launch needle color"
    route = PartitionRoutingReceipt.create(
        query=query,
        namespace=namespace,
        selected_partitions=namespace.partition_ids[:4],
        routed_source_count=len(namespace.sources),
    )
    prompt = SimpleNamespace(source=source.rows[0], namespace=namespace)
    parent = {
        "materialized_queries": [query],
        "receipt_sha256": "e" * 64,
        "routing_receipts": [route.projection()],
    }
    raw = _construct_row(
        prompt,
        parent,
        cache,
        budget=QueryGuidedScanBudget(),
    )
    return prompt, parent, raw


def test_guided_row_rehydrates_every_candidate_before_adapter_projection(
    tmp_path: Path,
) -> None:
    prompt, parent, raw = _guided_row(tmp_path)

    projected = _project_row(
        prompt,
        raw,
        parent,
        max_prompt_tokens=8_000,
    )

    assert projected.query_row_receipt_sha256 == raw["receipt_sha256"]
    assert projected.selected_before_dedup_ids == tuple(
        raw["selected_before_dedup_candidate_ids"]
    )
    assert projected.admitted_ids == tuple(raw["admitted_candidate_ids"])
    assert tuple(row.text for row in projected.admitted_delta) == tuple(
        row["text"] for row in raw["admitted_candidates"]
    )
    assert projected.not_admitted_ids == ()
    assert projected.compression_prompt.source_stage_id.endswith("_v1")


def test_guided_candidate_rejects_resealed_surface_provenance_change(
    tmp_path: Path,
) -> None:
    _prompt, _parent, raw = _guided_row(tmp_path)
    candidate = raw["candidates"][0]
    assert _candidate_from_projection(candidate).projection() == candidate

    changed = copy.deepcopy(candidate)
    changed["text"] += " changed"
    with pytest.raises(
        QueryGuidedPayloadAdapterError,
        match="cannot be reconstructed exactly",
    ):
        _candidate_from_projection(changed)


def test_guided_delta_uses_shared_payload_plane_with_exact_tier(
    tmp_path: Path,
) -> None:
    source_plan, parent = _plan(tmp_path)
    plan = build_query_payload_answer_plan(
        source_plan.adapter_population,
        parent,
        delta_tier=DELTA_TIER,
    )

    row = plan.rows[0]
    assert plan.delta_tier == DELTA_TIER
    assert any(alias.tier == DELTA_TIER for alias in row.aliases)
    assert row.messages is not None
    joined = "\n".join(message.content for message in row.messages)
    assert '"query_guided_scan_delta"' in joined
    assert '"query_expansion_delta"' not in joined


def test_replay_hash_check_accepts_only_exact_file_and_named_sidecar(
    tmp_path: Path,
) -> None:
    source, _created = publish_sealed_json(
        tmp_path / "source.json",
        {"format": "test", "gold_loaded": False},
    )
    replay, _created = publish_sealed_json(
        tmp_path / "replay.json",
        source.payload,
    )
    assert replay.sha256 == source.sha256

    _verify_replay_file(replay.path, expected_sha256=source.sha256)

    replay.path.with_name(replay.path.name + ".sha256").write_text(
        f"{source.sha256}  wrong-name.json\n",
        encoding="ascii",
    )
    with pytest.raises(QueryGuidedPayloadAdapterError, match="sidecar changed"):
        _verify_replay_file(replay.path, expected_sha256=source.sha256)


def test_hash_pinned_guided_loader_decodes_exact_questions(
    tmp_path: Path,
) -> None:
    artifact, _created = publish_sealed_json(
        tmp_path / "guided.json",
        {
            "aggregate": {"candidate_count": 2},
            "arm_label": "guided",
            "question_count": 2,
            "questions": [
                {"ordinal": 0, "text": "brace } and quote \\\" stay data"},
                {"nested": {"items": [{"value": 1}]}, "ordinal": 1},
            ],
            "retained_transformer_token_state_bytes": 0,
        },
    )
    payload = _load_exact_artifact_payload(artifact.path)

    assert len(payload["questions"]) == 2
    assert payload["questions"][0]["ordinal"] == 0
    assert payload["questions"][1]["nested"]["items"][0]["value"] == 1
    assert payload["question_count"] == 2
    assert payload["retained_transformer_token_state_bytes"] == 0


def test_runner_is_isolated_and_preflight_has_no_provider_switch() -> None:
    preflight = cli._parser().parse_args(["preflight"])
    provider = cli._parser().parse_args(
        [
            "provider-run",
            "--expected-answer-preflight-sha256",
            "f" * 64,
            "--enable-provider",
            "--authorized-provider-calls",
            "100",
        ]
    )

    assert preflight.output_root == cli.DEFAULT_OUTPUT
    assert preflight.output_root not in {
        preflight.query_parent_root,
        preflight.guided_root,
        preflight.parent_root,
    }
    assert not hasattr(preflight, "enable_provider")
    assert provider.authorized_provider_calls == 100
    assert provider.expected_guided_run_sha256 == cli.EXPECTED_GUIDED_RUN_SHA256
    assert provider.expected_guided_runtime_ledger_sha256 == (
        cli.EXPECTED_GUIDED_RUNTIME_LEDGER_SHA256
    )


def test_consolidated_judge_cli_replays_guided_shared_payload_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = judge_cli._parser().parse_args(
        [
            "preflight",
            "--arm",
            "query-guided-payload",
            "--expected-answer-preflight-sha256",
            "a" * 64,
            "--expected-answer-run-sha256",
            "b" * 64,
        ]
    )
    plan_args = judge_cli._plan_namespace(args)
    sentinel_plan = object()
    sentinel_plane = object()
    observed = {}

    def load_plan(value):
        observed["plan_args"] = value
        return sentinel_plan

    def replay(plan, **kwargs):
        observed["plan"] = plan
        observed["replay"] = kwargs
        return sentinel_plane

    monkeypatch.setattr(judge_cli.guided_payload_cli, "_load_plan", load_plan)
    monkeypatch.setattr(judge_cli, "replay_query_payload_answers", replay)
    monkeypatch.setattr(
        judge_cli,
        "load_verified_payload_semantic_arm_binding",
        lambda *_args, **_kwargs: object(),
    )

    assert judge_cli._answer_root(args) == cli.DEFAULT_OUTPUT
    assert plan_args.query_parent_root == cli.DEFAULT_QUERY_PARENT_ROOT
    assert plan_args.guided_root == cli.DEFAULT_GUIDED_ROOT
    assert plan_args.expected_guided_run_sha256 == cli.EXPECTED_GUIDED_RUN_SHA256
    assert plan_args.expected_guided_runtime_ledger_sha256 == (
        cli.EXPECTED_GUIDED_RUNTIME_LEDGER_SHA256
    )
    assert judge_cli._load_answer_plane(args) is sentinel_plane
    assert observed["plan"] is sentinel_plan
    assert observed["replay"]["output_root"] == cli.DEFAULT_OUTPUT
    assert observed["replay"]["expected_preflight_sha256"] == "a" * 64
    assert observed["replay"]["expected_run_sha256"] == "b" * 64
