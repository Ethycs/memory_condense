from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain._tokenizer import tokenizer_proxy_identity
from memory_condense.eval.benchmark import BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
from memory_condense.eval.mem0_adapter import (
    MEM0AI_PIN,
    MEM0_ATTRIBUTION_KIND,
    MEM0_BM25_MODEL,
    MEM0_CERTIFIED_RENDERING,
    MEM0_SPACY_MODEL,
)
from tools.matched_eval.contracts import identity_sha256
from tools.mem0_eval.prompt_pack import (
    MEM0_MAX_PROMPT_TOKEN_PROXY,
    MEM0_PROMPT_CAP_SEMANTICS,
    MEM0_REQUEST_WINDOW_SEMANTICS,
    MEM0_SOURCE_JUDGE_MODEL,
    MEM0_SOURCE_RESPONDER_MODEL,
    MEM0_TYPED_RETRIEVAL_ROW_FORMAT,
    PromptRequestWindowRef,
)
from tools.mem0_eval import typed_source_bridge as bridge
from tools.mem0_eval.typed_epoch_campaign import (
    Mem0TypedCampaignError,
    _aggregate_write_cost,
    _validate_write_observation,
    preflight_campaign,
)


def _evaluation_identity() -> dict[str, object]:
    return {
        "responder_model": MEM0_SOURCE_RESPONDER_MODEL,
        "judge_model": MEM0_SOURCE_JUDGE_MODEL,
        "use_judge": True,
        "provider_retries": 0,
        "max_provider_calls_per_shard": 20,
        "max_prompt_tokens": MEM0_MAX_PROMPT_TOKEN_PROXY,
        "prompt_cap_semantics": MEM0_PROMPT_CAP_SEMANTICS,
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "responder_output_token_reserve": (
            BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
        ),
        "recent_window": 4,
        "accuracy_target": 0.95,
        "min_target_questions": 100,
        "stress_context_tokens": 1_000_000,
        "stress_questions": 10,
        "stress_question_offset": 0,
        "max_samples": 1,
        "sample_offsets": list(range(0, 100, 10)),
    }


def _runtime_identity() -> dict[str, object]:
    return {
        "protocol": "mem0-oss-2.0.18-certified-local-v1",
        "certified": True,
        "local_owned_state": True,
        "on_disk": True,
        "stable_config_sha256": "a" * 64,
        "effective_config_sha256": "b" * 64,
        "stack": {
            "dependency_versions": {"mem0ai": MEM0AI_PIN},
            "bm25_model": MEM0_BM25_MODEL,
            "spacy_model": MEM0_SPACY_MODEL,
            "bm25_operational": True,
            "entity_extraction_operational": True,
        },
    }


def _typed_fixture() -> tuple[dict[str, object], SimpleNamespace, dict[str, tuple[PromptRequestWindowRef, ...]]]:
    questions = tuple(
        SimpleNamespace(
            dated_question=(
                f"[Question asked at 2026/08/{index + 1:02d} (Sat) 12:00] "
                f"What was remembered for item {index}?"
            ),
            question_id=f"q-{index:02d}",
        )
        for index in range(10)
    )
    shard = SimpleNamespace(
        parsed_sample=SimpleNamespace(questions=questions),
        question_ids=tuple(row.question_id for row in questions),
    )
    ledger: dict[str, tuple[PromptRequestWindowRef, ...]] = {}
    retrieval_rows = []
    for index in range(10):
        memory_id = f"memory-{index:02d}"
        ledger[memory_id] = (
            PromptRequestWindowRef(
                sample_id="context-stress-1000000",
                source=f"source-{index:02d}",
                session=f"session-{index:02d}",
                session_index=index,
                original_session_index=index,
                batch_index=index + 1,
                date=f"2026-07-{index + 1:02d}",
                turn_start=index * 2,
                turn_count=2,
                roles=("user", "assistant"),
            ),
        )
        retrieval_rows.append(
            {
                "raw_pool": [
                    {
                        "rank": 1,
                        "memory_id": memory_id,
                        "text": f"Remembered value {index}.",
                        "score": 0.9,
                        "created_at": f"2026-07-{index + 1:02d}T12:00:00+00:00",
                        "attribution_kind": MEM0_ATTRIBUTION_KIND,
                    }
                ],
                "search_latency_s": index / 1000,
            }
        )
    artifact: dict[str, object] = {
        "identity": {
            "runtime_identity": _runtime_identity(),
            "source_evaluation_identity": _evaluation_identity(),
        },
        "protocol": {"rendering_mode": MEM0_CERTIFIED_RENDERING},
        "retrieval_rows": retrieval_rows,
    }
    return artifact, shard, ledger


def test_provider_free_adapter_joins_authenticated_windows_without_promoting_provenance():
    artifact, shard, ledger = _typed_fixture()

    rows, window_count = bridge._typed_rows(
        artifact=artifact,
        shard=shard,
        ledger=ledger,
    )

    assert len(rows) == 10
    assert window_count == 10
    for index, row in enumerate(rows):
        assert row["format"] == MEM0_TYPED_RETRIEVAL_ROW_FORMAT
        assert row["question_id"] == f"q-{index:02d}"
        assert row["request_window_attribution_preserved"] is True
        assert row["request_window_semantics"] == MEM0_REQUEST_WINDOW_SEMANTICS
        assert row["created_at_source_event_time_authoritative"] is False
        assert row["provenance"] == {
            "kind": MEM0_ATTRIBUTION_KIND,
            "supports_exact_source_provenance": False,
        }
        window = row["raw_pool"][0]["request_window_attribution"][0]
        assert window["source"] == f"source-{index:02d}"
        rendered_messages = json.dumps(row["messages"], sort_keys=True)
        assert window["source"] not in rendered_messages
        assert window["session"] not in rendered_messages

    missing = dict(ledger)
    del missing["memory-07"]
    with pytest.raises(
        bridge.Mem0TypedSourceBridgeError,
        match="absent from the authenticated journal ledger",
    ):
        bridge._typed_rows(artifact=artifact, shard=shard, ledger=missing)


def test_complete_write_metering_is_authenticated_and_cost_composable():
    observed = {
        "add_attempted": 7,
        "add_completed": 7,
        "add_failed": 0,
        "extraction_attempted": 7,
        "extraction_completed": 7,
        "extraction_failed": 0,
        "extraction_raw_message_token_proxy": 123,
        "extraction_provider_input_tokens": 80,
        "extraction_provider_output_tokens": 20,
        "extraction_usage_status": "provider_reported_exact",
        "embedding_operations": 11,
        "embedding_input_token_proxy": 44,
        "returned_memory_count": 5,
        "persisted_memory_count": 4,
        "persisted_storage_bytes": 1_024,
        "add_latency_s": 0.75,
        "extraction_latency_s": 0.5,
        "embedding_latency_s": 0.2,
        "storage_latency_s": 0.1,
    }
    attestation_body = {
        "format": "memory-condense-mem0-complete-write-usage-attestation-v1",
        "observed": observed,
        "observed_sha256": bridge.canonical_json_sha256(observed),
    }
    attestation = {
        **attestation_body,
        "receipt_sha256": bridge.canonical_json_sha256(attestation_body),
    }
    artifact = {
        "write_usage_attestation": attestation,
        "resumable_closure": {
            "write_usage_attestation": attestation,
            "write_usage_attestation_sha256": attestation["receipt_sha256"],
        },
        "execution_binding": {
            "write_usage_attestation_sha256": attestation["receipt_sha256"]
        },
    }

    observation = bridge._write_observation(artifact)
    assert _validate_write_observation(observation) == observation == observed
    aggregate = _aggregate_write_cost(
        {
            "population_identity_sha256": "0" * 64,
            "write_observations": [observation],
        }
    )
    assert aggregate.add_attempted == 7
    assert aggregate.embedding_operations == 11
    assert aggregate.extraction_provider_input_tokens == 80

    forged = copy.deepcopy(artifact)
    forged["write_usage_attestation"]["observed"]["embedding_operations"] = 0
    with pytest.raises(
        bridge.Mem0TypedSourceBridgeError, match="write-usage chain changed"
    ):
        bridge._write_observation(forged)

    forged = copy.deepcopy(artifact)
    forged["execution_binding"]["write_usage_attestation_sha256"] = "0" * 64
    with pytest.raises(
        bridge.Mem0TypedSourceBridgeError, match="write-usage chain changed"
    ):
        bridge._write_observation(forged)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _plain_source_inputs(tmp_path: Path) -> bridge.LockedSourceInputs:
    source_root = tmp_path / "source-root"
    tool_root = tmp_path / "tool-root"
    source_root.mkdir()
    tool_root.mkdir()
    return bridge.LockedSourceInputs(
        benchmark_file=_write(tmp_path / "benchmark.json", "benchmark\n"),
        split_manifest=_write(tmp_path / "split.json", "split\n"),
        policy_manifest=_write(tmp_path / "source-policy.json", "source-policy\n"),
        repository_root=source_root,
        mem0_policy_manifest=_write(tmp_path / "mem0-policy.json", "mem0-policy\n"),
        mem0_environment_lock=_write(tmp_path / "mem0.lock", "mem0-lock\n"),
        mem0_tool_root=tool_root,
    )


def test_locked_context_rejects_self_consistent_replacement_source_and_shards(
    tmp_path,
    monkeypatch,
):
    source = _plain_source_inputs(tmp_path)
    plan = SimpleNamespace(
        dataset_sha256="0" * 64,
        split_manifest_sha256=bridge.FROZEN_SPLIT_MANIFEST_SHA256,
        policy_manifest_sha256=bridge.FROZEN_SOURCE_POLICY_SHA256,
        implementation_sha256=bridge.FROZEN_SOURCE_IMPLEMENTATION_SHA256,
        environment_lock_sha256=bridge.FROZEN_SOURCE_ENVIRONMENT_LOCK_SHA256,
        sample_offsets=bridge.FROZEN_OFFSETS,
        target_tokens=bridge.FROZEN_TARGET_TOKENS,
        questions_per_shard=bridge.FROZEN_QUESTIONS_PER_SHARD,
        evaluation_identity=_evaluation_identity(),
    )
    monkeypatch.setattr(
        bridge,
        "load_source_validation_plan",
        lambda **_kwargs: plan,
    )
    with pytest.raises(
        bridge.Mem0TypedSourceBridgeError,
        match="exact validation-v3 freeze",
    ):
        bridge._locked_context(source)

    plan.dataset_sha256 = bridge.FROZEN_DATASET_SHA256
    shards = []
    for shard_index, offset in enumerate(bridge.FROZEN_OFFSETS):
        raw_pairs = 2_492 if shard_index < 9 else 2_500
        skipped = 0 if shard_index < 9 else 5
        questions = tuple(
            f"replacement-q-{offset + index:03d}" for index in range(10)
        )
        shards.append(
            SimpleNamespace(
                add_counts=SimpleNamespace(
                    raw_pairs=raw_pairs,
                    skipped_empty_pairs=skipped,
                    add_requests=raw_pairs - skipped,
                ),
                history_sample_ids=questions,
                parsed_sample=SimpleNamespace(sample_id="replacement-sample"),
                question_ids=questions,
                raw_history_bundle_sha256=f"{shard_index + 1:064x}",
                sample_offset=offset,
                sample_sha256=f"{shard_index + 101:064x}",
            )
        )
    monkeypatch.setattr(
        bridge,
        "build_raw_stress_shards",
        lambda **_kwargs: tuple(shards),
    )
    monkeypatch.setattr(
        bridge,
        "load_mem0_comparison_policy",
        lambda *_args, **_kwargs: SimpleNamespace(
            environment_lock_sha256="a" * 64,
            sha256="b" * 64,
            tool_implementation_sha256="c" * 64,
        ),
    )

    def fake_shard_receipt(shard):
        index = shard.sample_offset // 10
        return {
            "add_batches_sha256": f"{index + 201:064x}",
            "transcript_tokens": (
                1_000_000 if index < 9 else 1_441_617
            ),
        }

    monkeypatch.setattr(bridge, "shard_receipt", fake_shard_receipt)
    with pytest.raises(
        bridge.Mem0TypedSourceBridgeError,
        match="per-shard add identity changed",
    ):
        bridge._locked_context(source)


def _provider_free_bridge_fixture(tmp_path: Path, monkeypatch):
    source = _plain_source_inputs(tmp_path)
    source_root = source.repository_root
    tool_root = source.mem0_tool_root
    source_projection = {
        "benchmark_file": bridge._file_receipt(source.benchmark_file, "benchmark"),
        "format": bridge.SOURCE_FORMAT,
        "mem0_environment_lock": bridge._file_receipt(
            source.mem0_environment_lock, "Mem0 lock"
        ),
        "mem0_policy": bridge._file_receipt(source.mem0_policy_manifest, "Mem0 policy"),
        "mem0_tool_implementation_sha256": "a" * 64,
        "mem0_tool_root": str(tool_root.resolve()),
        "policy_manifest": bridge._file_receipt(source.policy_manifest, "source policy"),
        "repository_root": str(source_root.resolve()),
        "source_environment_lock_sha256": "b" * 64,
        "source_implementation_sha256": "c" * 64,
        "split_manifest": bridge._file_receipt(source.split_manifest, "split"),
    }
    shards = tuple(
        SimpleNamespace(
            sample_offset=offset,
            question_ids=tuple(f"q-{offset + index:03d}" for index in range(10)),
        )
        for offset in bridge.FROZEN_OFFSETS
    )
    context = bridge._LockedContext(
        source=source,
        source_plan=SimpleNamespace(),
        policy=SimpleNamespace(),
        shards=shards,
        population_identity_sha256="d" * 64,
        population_projection={
            "format": "provider-free-fixture-population-v1",
            "question_count": 100,
        },
        source_projection=source_projection,
    )
    monkeypatch.setattr(bridge, "_locked_context", lambda _source: context)

    terminals = []
    for offset in bridge.FROZEN_OFFSETS:
        artifact = _write(
            tmp_path / f"terminal-{offset:03d}.json",
            json.dumps({"sample_offset": offset}, sort_keys=True),
        )
        trace = _write(
            tmp_path / f"trace-{offset:03d}.json",
            json.dumps({"trace_offset": offset}, sort_keys=True),
        )
        journal = _write(
            tmp_path / f"journal-{offset:03d}.jsonl",
            f"journal-offset-{offset}\n",
        )
        terminals.append(bridge.ResumableTerminalInput(artifact, trace, journal))

    def verified_terminal(*, terminal, shard, context):
        artifact_receipt = bridge._file_receipt(terminal.artifact_path, "artifact")
        trace_receipt = bridge._file_receipt(terminal.trace_path, "trace")
        journal_receipt = bridge._file_receipt(terminal.journal_path, "journal")
        journal_receipt.update(
            {
                "checkpoint_authority_sha256": "2" * 64,
                "cleanup_entry_sha256": "3" * 64,
                "commit_population_sha256": "4" * 64,
                "journal_chain_sha256": "5" * 64,
                "plan_sha256": "6" * 64,
            }
        )
        export = {
            "format": "provider-free-fixture-export-v1",
            "population_identity_sha256": context.population_identity_sha256,
            "question_count": 10,
            "question_ids": list(shard.question_ids),
            "question_offset": shard.sample_offset,
        }
        row = {
            "artifact": artifact_receipt,
            "diagnostic_request_window_count": 10,
            "export_payload_sha256": hashlib.sha256(
                bridge.canonical_json_bytes(export)
            ).hexdigest(),
            "format": bridge.ROW_FORMAT,
            "journal": journal_receipt,
            "population_identity_sha256": context.population_identity_sha256,
            "provenance": {
                "attribution_kind": MEM0_ATTRIBUTION_KIND,
                "request_windows_are_fact_evidence": False,
                "supports_exact_source_provenance": False,
            },
            "question_count": 10,
            "question_ids": list(shard.question_ids),
            "question_offset": shard.sample_offset,
            "raw_history_bundle_sha256": "e" * 64,
            "sample_id": "context-stress-1000000",
            "sample_offset": shard.sample_offset,
            "sample_sha256": "f" * 64,
            "source_add_operations": 1,
            "source_raw_pairs": 1,
            "source_skipped_empty_pairs": 0,
            "trace": trace_receipt,
            "typed_retrieval_rows_sha256": "1" * 64,
            "zero_persisted_transformer_token_state": True,
        }
        return export, {**row, "row_receipt_sha256": identity_sha256(row)}

    monkeypatch.setattr(bridge, "_verify_terminal_shard", verified_terminal)
    return source, tuple(terminals)


def test_provider_free_ten_shard_bridge_publishes_and_reopens_all_authorities(
    tmp_path,
    monkeypatch,
):
    source, terminals = _provider_free_bridge_fixture(tmp_path, monkeypatch)
    output = tmp_path / "bridge"

    manifest, exports = bridge.build_source_bridge(
        source=source,
        terminals=terminals,
        output_root=output,
    )
    reopened = bridge.reopen_source_bridge(output / bridge.MANIFEST_NAME)

    assert len(exports) == len(reopened.exports) == 10
    assert manifest["sample_offsets"] == list(range(0, 100, 10))
    assert manifest["question_count"] == 100
    assert manifest["physical_provider_calls"] == 0
    assert manifest["gold_loaded"] is False
    assert manifest["provenance"]["supports_exact_source_provenance"] is False
    assert [row.payload["question_offset"] for row in reopened.exports] == list(
        range(0, 100, 10)
    )

    terminals[4].trace_path.write_text("forged trace\n", encoding="utf-8")
    with pytest.raises(bridge.Mem0TypedSourceBridgeError, match="receipt changed"):
        bridge.reopen_source_bridge(output / bridge.MANIFEST_NAME)


def test_bridge_rejects_synthetic_population_duplicate_authority_and_sha_api(
    tmp_path,
    monkeypatch,
):
    source, terminals = _provider_free_bridge_fixture(tmp_path, monkeypatch)

    with pytest.raises(
        bridge.Mem0TypedSourceBridgeError,
        match="exactly ten terminals",
    ):
        bridge.build_source_bridge(
            source=source,
            terminals=terminals[:1],
            output_root=tmp_path / "one-export",
            dry_run=True,
        )

    duplicated = list(terminals)
    duplicated[1] = bridge.ResumableTerminalInput(
        duplicated[1].artifact_path,
        duplicated[1].trace_path,
        duplicated[0].journal_path,
    )
    with pytest.raises(
        bridge.Mem0TypedSourceBridgeError,
        match="pairwise distinct",
    ):
        bridge.build_source_bridge(
            source=source,
            terminals=duplicated,
            output_root=tmp_path / "duplicate",
            dry_run=True,
        )

    assert "sha256" not in inspect.signature(bridge.build_source_bridge).parameters
    assert "retrieval_export_paths" not in inspect.signature(preflight_campaign).parameters
    with pytest.raises(TypeError, match="unexpected keyword"):
        bridge.build_source_bridge(
            source=source,
            terminals=terminals,
            output_root=tmp_path / "caller-sha",
            expected_terminal_sha256s=["0" * 64] * 10,
            dry_run=True,
        )

    manifest, _ = bridge.build_source_bridge(
        source=source,
        terminals=terminals,
        output_root=tmp_path / "dry-run",
        dry_run=True,
    )
    forged_manifest = copy.deepcopy(manifest)
    forged_manifest["shards"] = forged_manifest["shards"][:1]
    forged_manifest["export_count"] = 1
    forged_manifest["question_count"] = 10
    forged_manifest["sample_offsets"] = [0]
    with pytest.raises(
        bridge.Mem0TypedSourceBridgeError,
        match="fixed population contract",
    ):
        bridge._manifest_value(forged_manifest)
