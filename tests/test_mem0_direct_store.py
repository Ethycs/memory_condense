from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.eval.sample_identity import canonical_sha256, sample_sha256
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample
from tools.mem0_eval import direct_store_cli
from tools.mem0_eval import direct_store_report as report_module
from tools.mem0_eval.direct_store import (
    DIRECT_STORE_ARM_ID,
    DIRECT_STORE_CLEANUP_FORMAT,
    DIRECT_STORE_RUNTIME_FORMAT,
    DIRECT_STORE_SCHEMA_VERSION,
    DIRECT_STORE_SHARD_FORMAT,
    DirectStoreError,
    FROZEN_SAMPLE_OFFSETS,
    run_injected_direct_store_shard,
    save_direct_store_artifact,
    validate_direct_store_shard_artifact,
)
from tools.mem0_eval.direct_store_report import (
    DIRECT_STORE_CAMPAIGN_FORMAT,
    DIRECT_STORE_POPULATION_FORMAT,
    FROZEN_ADD_OPERATIONS,
    FROZEN_DATASET_SHA256,
    FROZEN_ORDERED_QUESTION_IDS_SHA256,
    FROZEN_QUESTION_COUNT,
    FROZEN_RAW_PAIRS,
    FROZEN_SEARCH_OPERATIONS,
    FROZEN_SPLIT_MANIFEST_SHA256,
    FROZEN_SKIPPED_EMPTY_PAIRS,
    FROZEN_SORTED_UNIQUE_QUESTION_IDS_SHA256,
    build_frozen_direct_store_population_preflight,
    merge_direct_store_retrieval_shards,
    validate_direct_store_campaign_report,
    validate_frozen_direct_store_population,
)
from tools.mem0_eval.protocol import (
    RawStressShard,
    build_composite_add_batches,
    compose_raw_stress_record,
    count_official_add_requests,
)
from tools.mem0_eval.preflight import tool_implementation_sha256


def _record(question_id: str, *, batches: int) -> dict[str, Any]:
    sessions = [
        [
            {"role": "user", "content": f"{question_id} user {index}"},
            {
                "role": "assistant",
                "content": f"{question_id} assistant {index}",
            },
        ]
        for index in range(batches)
    ]
    return {
        "question_id": question_id,
        "haystack_sessions": sessions,
        "haystack_session_ids": [
            f"session-{index:05d}" for index in range(batches)
        ],
        "haystack_dates": [
            f"2025-01-{(index % 28) + 1:02d}" for index in range(batches)
        ],
    }


def _shard(
    offset: int,
    *,
    adds: int = 2,
    reverse_questions: bool = False,
    id_prefix: str = "q",
) -> RawStressShard:
    indexes = list(range(10))
    if reverse_questions:
        indexes.reverse()
    question_ids = [
        f"{id_prefix}-{offset + index:03d}" for index in indexes
    ]
    questions = [
        BenchmarkQuestion(
            question_id=question_id,
            question=f"What was stored for {question_id}?",
            answer=f"answer {question_id}",
            category="single-session-user",
            question_date="2025-02-01",
        )
        for question_id in question_ids
    ]
    sample = BenchmarkSample(
        sample_id=f"mem0-context-stress-1000000-offset-{offset:03d}",
        turns=[("user", f"sample {offset}"), ("assistant", "ack")],
        turn_source_ids=[f"{question_ids[0]}::s", f"{question_ids[0]}::s"],
        questions=questions,
    )
    records = [_record(question_ids[0], batches=adds)]
    records.extend(_record(question_id, batches=0) for question_id in question_ids[1:])
    raw_bundle = compose_raw_stress_record(records, sample_id=sample.sample_id)
    batches = build_composite_add_batches(records)
    return RawStressShard(
        sample_offset=offset,
        parsed_sample=sample,
        sample_sha256=sample_sha256(sample),
        history_sample_ids=tuple(question_ids),
        raw_history_bundle=raw_bundle,
        raw_history_bundle_sha256=canonical_sha256(raw_bundle),
        add_batches=batches,
        add_counts=count_official_add_requests(records),
    )


class _FakeDirectStore:
    def __init__(
        self,
        *,
        attempt_extraction: bool = False,
        swallow_extraction: bool = False,
        return_unknown_search_id: bool = False,
    ) -> None:
        self.attempt_extraction = attempt_extraction
        self.swallow_extraction = swallow_extraction
        self.return_unknown_search_id = return_unknown_search_id
        self.provider_delegate_calls = 0
        self.add_calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []
        self.search_calls: list[tuple[str, dict[str, Any]]] = []
        self.memory_ids: list[str] = []
        self.closed = False
        self.restored_before_close = False

        def generate_response(*_args: Any, **_kwargs: Any) -> str:
            self.provider_delegate_calls += 1
            return '{"facts": []}'

        self.original_generate = generate_response
        self.llm = SimpleNamespace(generate_response=generate_response)

    def add(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, list[dict[str, str]]]:
        self.add_calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        if self.attempt_extraction:
            try:
                self.llm.generate_response(messages=messages)
            except DirectStoreError:
                if not self.swallow_extraction:
                    raise
        memory_id = f"memory-{len(self.memory_ids):05d}"
        self.memory_ids.append(memory_id)
        return {"results": [{"id": memory_id}]}

    def search(self, query: str, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        self.search_calls.append((query, copy.deepcopy(kwargs)))
        memory_id = (
            "outside-ledger"
            if self.return_unknown_search_id
            else self.memory_ids[len(self.search_calls) % len(self.memory_ids)]
        )
        return {
            "results": [
                {
                    "id": memory_id,
                    "memory": f"retrieved for {query}",
                    "score": 0.9,
                    "created_at": "2025-01-01T00:00:00Z",
                }
            ]
        }

    def runtime_receipt(self) -> dict[str, Any]:
        return {
            "format": DIRECT_STORE_RUNTIME_FORMAT,
            "execution_kind": "injected_test_double",
            "backend_label": "deterministic-fake-direct-store-v1",
            "actual_mem0_executed": False,
            "dependency_versions": {"mem0ai": "not-installed-test-double"},
            "local_only": True,
            "network_calls_authorized": 0,
            "network_calls_observed": 0,
            "provider_calls_authorized": 0,
            "provider_calls_observed": 0,
        }

    def close(self) -> None:
        self.restored_before_close = (
            getattr(self.add, "__func__", None) is _FakeDirectStore.add
            and self.llm.generate_response is self.original_generate
        )
        self.closed = True

    def cleanup_receipt(self) -> dict[str, Any]:
        return {
            "format": DIRECT_STORE_CLEANUP_FORMAT,
            "closed": self.closed,
            "owned_state_removed": self.closed,
            "network_calls_observed": 0,
            "provider_calls_observed": 0,
        }


def _clock() -> Any:
    value = 0.0

    def read() -> float:
        nonlocal value
        value += 0.01
        return value

    return read


def test_injected_runner_preserves_batches_and_question_order() -> None:
    shard = _shard(0, adds=2)
    backend = _FakeDirectStore()

    artifact = run_injected_direct_store_shard(
        shard,
        backend_factory=lambda: backend,
        clock=_clock(),
    )

    assert artifact["arm"]["arm_id"] == DIRECT_STORE_ARM_ID
    assert artifact["arm"]["official_mem0_comparison"] is False
    assert artifact["arm"]["infer"] is False
    assert artifact["arm"]["benchmark_result_eligible"] is False
    assert artifact["format"] == DIRECT_STORE_SHARD_FORMAT
    assert artifact["schema_version"] == 2
    assert "question_ids_sha256" not in artifact["sample"]
    assert artifact["sample"]["ordered_question_ids"] == list(
        shard.question_ids
    )
    assert artifact["extraction"]["authorized"] == 0
    assert artifact["extraction"]["attempted"] == 0
    assert artifact["extraction"]["zero_extraction_calls_certified"] is True
    assert [row[1]["infer"] for row in backend.add_calls] == [False, False]
    assert [row[0] for row in backend.add_calls] == [
        [
            {"role": role, "content": content}
            for role, content in batch.messages
        ]
        for batch in shard.add_batches
    ]
    assert [query for query, _kwargs in backend.search_calls] == [
        question.question for question in shard.parsed_sample.questions
    ]
    assert all(
        kwargs
        == {
            "top_k": 200,
            "filters": {"user_id": artifact["ingestion"]["scope"]},
            "threshold": 0.1,
            "rerank": False,
            "explain": False,
        }
        for _query, kwargs in backend.search_calls
    )
    first_source = artifact["retrieval"]["searches"][0]["candidates"][0][
        "source"
    ]
    assert first_source["batch_id"] in {"add-00001", "add-00002"}
    assert first_source["messages_sha256"]
    assert backend.provider_delegate_calls == 0
    assert backend.restored_before_close is True
    assert backend.closed is True
    assert validate_direct_store_shard_artifact(artifact, shard) == artifact


def test_zero_extraction_meter_detects_swallowed_attempt_and_cleans_up() -> None:
    shard = _shard(0, adds=1)
    backend = _FakeDirectStore(
        attempt_extraction=True,
        swallow_extraction=True,
    )

    with pytest.raises(DirectStoreError, match="attempted extraction"):
        run_injected_direct_store_shard(
            shard,
            backend_factory=lambda: backend,
            clock=_clock(),
        )

    assert backend.provider_delegate_calls == 0
    assert backend.restored_before_close is True
    assert backend.closed is True


def test_unknown_search_memory_fails_closed_after_exact_cleanup() -> None:
    shard = _shard(0, adds=1)
    backend = _FakeDirectStore(return_unknown_search_id=True)

    with pytest.raises(DirectStoreError, match="outside the exact add ledger"):
        run_injected_direct_store_shard(
            shard,
            backend_factory=lambda: backend,
            clock=_clock(),
        )

    assert backend.restored_before_close is True
    assert backend.closed is True


def test_invalid_runtime_receipt_still_closes_owned_backend() -> None:
    shard = _shard(0, adds=1)
    backend = _FakeDirectStore()
    original_reader = backend.runtime_receipt

    def invalid_receipt() -> dict[str, Any]:
        value = original_reader()
        value["local_only"] = False
        return value

    backend.runtime_receipt = invalid_receipt  # type: ignore[method-assign]

    with pytest.raises(DirectStoreError, match="local_only mismatch"):
        run_injected_direct_store_shard(
            shard,
            backend_factory=lambda: backend,
            clock=_clock(),
        )

    assert backend.add_calls == []
    assert backend.closed is True


def test_rehashed_add_ledger_tamper_still_fails_source_reconstruction() -> None:
    shard = _shard(0, adds=1)
    artifact = run_injected_direct_store_shard(
        shard,
        backend_factory=_FakeDirectStore,
        clock=_clock(),
    )
    tampered = copy.deepcopy(artifact)
    row = tampered["ingestion"]["add_ledger"][0]
    row["messages_sha256"] = "f" * 64
    body = dict(row)
    body.pop("ledger_row_sha256")
    row["ledger_row_sha256"] = canonical_sha256(body)
    tampered["ingestion"]["add_ledger_sha256"] = canonical_sha256(
        tampered["ingestion"]["add_ledger"]
    )
    body = dict(tampered)
    body.pop("artifact_sha256")
    tampered["artifact_sha256"] = canonical_sha256(body)

    with pytest.raises(DirectStoreError, match="add ledger row 1 mismatch"):
        validate_direct_store_shard_artifact(tampered, shard)


def test_frozen_constants_expose_exact_100q_direct_store_shape() -> None:
    assert DIRECT_STORE_SCHEMA_VERSION == 2
    assert DIRECT_STORE_SHARD_FORMAT.endswith("-v2")
    assert DIRECT_STORE_POPULATION_FORMAT.endswith("-v2")
    assert DIRECT_STORE_CAMPAIGN_FORMAT.endswith("-v2")
    assert FROZEN_SAMPLE_OFFSETS == tuple(range(0, 100, 10))
    assert FROZEN_QUESTION_COUNT == 100
    assert FROZEN_RAW_PAIRS == 24_928
    assert FROZEN_SKIPPED_EMPTY_PAIRS == 5
    assert FROZEN_ADD_OPERATIONS == 24_923
    assert FROZEN_SEARCH_OPERATIONS == 100
    assert FROZEN_DATASET_SHA256 == (
        "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
    )
    assert FROZEN_SPLIT_MANIFEST_SHA256 == (
        "8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4"
    )
    assert FROZEN_ORDERED_QUESTION_IDS_SHA256 == (
        "7a67aa6f43ffb94d487fb9184f871735bd9edac1974a3154898846d1140c83a1"
    )
    assert FROZEN_SORTED_UNIQUE_QUESTION_IDS_SHA256 == (
        "dd8addf6bba1bd83d7ce4c9427e2e8a86cf0eacbf04d13b3cf13cc8d287dd99c"
    )


def test_ordered_and_sorted_set_population_hashes_cannot_be_confused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards = tuple(
        _shard(offset, adds=1, reverse_questions=True)
        for offset in FROZEN_SAMPLE_OFFSETS
    )
    ordered_ids = [
        question_id for shard in shards for question_id in shard.question_ids
    ]
    ordered_hash = canonical_sha256(ordered_ids)
    sorted_unique_hash = canonical_sha256(sorted(set(ordered_ids)))
    assert ordered_hash != sorted_unique_hash
    monkeypatch.setattr(report_module, "FROZEN_RAW_PAIRS", 10)
    monkeypatch.setattr(report_module, "FROZEN_SKIPPED_EMPTY_PAIRS", 0)
    monkeypatch.setattr(report_module, "FROZEN_ADD_OPERATIONS", 10)
    monkeypatch.setattr(
        report_module,
        "FROZEN_ORDERED_QUESTION_IDS_SHA256",
        ordered_hash,
    )
    monkeypatch.setattr(
        report_module,
        "FROZEN_SORTED_UNIQUE_QUESTION_IDS_SHA256",
        sorted_unique_hash,
    )

    receipt = validate_frozen_direct_store_population(shards)
    assert receipt["ordered_question_ids"] == ordered_ids
    assert receipt["ordered_question_ids_sha256"] == ordered_hash
    assert receipt["sorted_unique_question_ids_sha256"] == sorted_unique_hash
    assert "question_ids_sha256" not in receipt
    assert receipt["source_coordinates"] == {
        "dataset_sha256": FROZEN_DATASET_SHA256,
        "split_manifest_sha256": FROZEN_SPLIT_MANIFEST_SHA256,
    }
    assert receipt["source_file_verification"] == {
        "verified_before_population": False,
        "rechecked_after_population": False,
    }
    assert receipt["tool_identity"] == {
        "kind": "tools-mem0-eval-python-tree-v1",
        "root": "tools/mem0_eval",
        "scope": "recursive-*.py",
        "hash_protocol": "length-prefixed-relative-path-and-bytes-sha256-v1",
        "tool_implementation_sha256": tool_implementation_sha256(),
        "rechecked_after_population": False,
    }
    assert receipt["actual_mem0_runtime_environment"] == {
        "status": "unresolved",
        "isolated_environment_lock": "tools/mem0_eval/pixi.lock",
        "isolated_environment_lock_present": False,
        "isolated_environment_lock_sha256": None,
        "actual_mem0_runtime_environment_frozen": False,
        "actual_mem0_runtime_verified": False,
        "actual_mem0_executed": False,
        "retrieval_execution_authorized": False,
        "resource_matching_completed": False,
    }

    reordered = tuple(
        _shard(offset, adds=1, reverse_questions=False)
        for offset in FROZEN_SAMPLE_OFFSETS
    )
    with pytest.raises(DirectStoreError, match="ordered-question population"):
        validate_frozen_direct_store_population(reordered)

    changed_set = tuple(
        _shard(offset, adds=1, reverse_questions=True, id_prefix="changed")
        for offset in FROZEN_SAMPLE_OFFSETS
    )
    changed_order = [
        question_id
        for shard in changed_set
        for question_id in shard.question_ids
    ]
    monkeypatch.setattr(
        report_module,
        "FROZEN_ORDERED_QUESTION_IDS_SHA256",
        canonical_sha256(changed_order),
    )
    with pytest.raises(DirectStoreError, match="sorted-unique-question population"):
        validate_frozen_direct_store_population(changed_set)


def test_provider_free_preflight_binds_source_tool_and_unresolved_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset.json"
    split = tmp_path / "split.json"
    dataset.write_bytes(b"synthetic-dataset\n")
    split.write_bytes(b"synthetic-split\n")
    shards = tuple(
        _shard(offset, adds=1, reverse_questions=True)
        for offset in FROZEN_SAMPLE_OFFSETS
    )
    ordered_ids = [
        question_id for shard in shards for question_id in shard.question_ids
    ]
    monkeypatch.setattr(
        report_module,
        "FROZEN_DATASET_SHA256",
        hashlib.sha256(dataset.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        report_module,
        "FROZEN_SPLIT_MANIFEST_SHA256",
        hashlib.sha256(split.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(report_module, "FROZEN_RAW_PAIRS", 10)
    monkeypatch.setattr(report_module, "FROZEN_SKIPPED_EMPTY_PAIRS", 0)
    monkeypatch.setattr(report_module, "FROZEN_ADD_OPERATIONS", 10)
    monkeypatch.setattr(
        report_module,
        "FROZEN_ORDERED_QUESTION_IDS_SHA256",
        canonical_sha256(ordered_ids),
    )
    monkeypatch.setattr(
        report_module,
        "FROZEN_SORTED_UNIQUE_QUESTION_IDS_SHA256",
        canonical_sha256(sorted(set(ordered_ids))),
    )
    monkeypatch.setattr(
        report_module,
        "build_raw_stress_shards",
        lambda **_kwargs: shards,
    )
    monkeypatch.setattr(
        report_module,
        "tool_implementation_sha256",
        lambda: "a" * 64,
    )

    observed_shards, receipt = build_frozen_direct_store_population_preflight(
        benchmark_file=dataset,
        split_manifest=split,
    )

    assert observed_shards == shards
    assert receipt["status"] == (
        "ready_provider_free_population_and_tool_receipt"
    )
    assert receipt["source_file_verification"] == {
        "verified_before_population": True,
        "rechecked_after_population": True,
    }
    assert receipt["tool_identity"] == {
        "kind": "tools-mem0-eval-python-tree-v1",
        "root": "tools/mem0_eval",
        "scope": "recursive-*.py",
        "hash_protocol": "length-prefixed-relative-path-and-bytes-sha256-v1",
        "tool_implementation_sha256": "a" * 64,
        "rechecked_after_population": True,
    }
    assert receipt["actual_mem0_runtime_environment"]["status"] == "unresolved"
    assert (
        receipt["actual_mem0_runtime_environment"][
            "isolated_environment_lock_present"
        ]
        is False
    )
    assert (
        receipt["actual_mem0_runtime_environment"][
            "actual_mem0_runtime_environment_frozen"
        ]
        is False
    )
    assert (
        receipt["actual_mem0_runtime_environment"][
            "retrieval_execution_authorized"
        ]
        is False
    )
    body = dict(receipt)
    preflight_sha = body.pop("preflight_sha256")
    assert preflight_sha == canonical_sha256(body)


@pytest.mark.parametrize("mismatch", ["dataset", "split"])
def test_provider_free_preflight_rejects_wrong_source_before_population_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    dataset = tmp_path / "dataset.json"
    split = tmp_path / "split.json"
    dataset.write_bytes(b"wrong-dataset\n")
    split.write_bytes(b"wrong-split\n")
    if mismatch == "split":
        monkeypatch.setattr(
            report_module,
            "FROZEN_DATASET_SHA256",
            hashlib.sha256(dataset.read_bytes()).hexdigest(),
        )
    population_builds = 0

    def unexpected_build(**_kwargs: Any) -> tuple[RawStressShard, ...]:
        nonlocal population_builds
        population_builds += 1
        return ()

    monkeypatch.setattr(
        report_module,
        "build_raw_stress_shards",
        unexpected_build,
    )

    with pytest.raises(DirectStoreError, match=f"frozen .*{mismatch}.*SHA-256"):
        build_frozen_direct_store_population_preflight(
            benchmark_file=dataset,
            split_manifest=split,
        )

    assert population_builds == 0


def test_ten_shard_merge_preserves_locked_order_and_zero_call_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exercise the complete ten-shard/100-question merge cheaply. The separate
    # constant test above pins the real 24,923-add population shape.
    monkeypatch.setattr(report_module, "FROZEN_RAW_PAIRS", 10)
    monkeypatch.setattr(report_module, "FROZEN_SKIPPED_EMPTY_PAIRS", 0)
    monkeypatch.setattr(report_module, "FROZEN_ADD_OPERATIONS", 10)
    shards = tuple(
        _shard(offset, adds=1, reverse_questions=True)
        for offset in FROZEN_SAMPLE_OFFSETS
    )
    expected_ids = [
        question_id for shard in shards for question_id in shard.question_ids
    ]
    monkeypatch.setattr(
        report_module,
        "FROZEN_ORDERED_QUESTION_IDS_SHA256",
        canonical_sha256(expected_ids),
    )
    monkeypatch.setattr(
        report_module,
        "FROZEN_SORTED_UNIQUE_QUESTION_IDS_SHA256",
        canonical_sha256(sorted(set(expected_ids))),
    )
    artifacts = [
        run_injected_direct_store_shard(
            shard,
            backend_factory=_FakeDirectStore,
            clock=_clock(),
        )
        for shard in shards
    ]

    campaign = merge_direct_store_retrieval_shards(
        artifacts,
        expected_shards=shards,
    )

    assert campaign["ordered_question_ids"] == expected_ids
    assert campaign["ordered_question_ids_sha256"] == canonical_sha256(
        expected_ids
    )
    assert campaign["sorted_unique_question_ids_sha256"] == canonical_sha256(
        sorted(set(expected_ids))
    )
    assert "question_ids_sha256" not in campaign
    assert campaign["totals"]["add_operations"] == 10
    assert campaign["totals"]["search_operations"] == 100
    assert campaign["totals"]["extraction_calls"] == 0
    assert campaign["totals"]["provider_calls"] == 0
    assert campaign["limitations"]["official_mem0_comparison"] is False
    assert campaign["limitations"]["resource_matching_completed"] is False
    assert campaign["limitations"]["benchmark_result_eligible"] is False
    assert (
        validate_direct_store_campaign_report(
            campaign,
            shard_artifacts=artifacts,
            expected_shards=shards,
        )
        == campaign
    )


def test_population_rejects_synthetic_totals_under_real_contract() -> None:
    shards = tuple(_shard(offset, adds=1) for offset in FROZEN_SAMPLE_OFFSETS)
    with pytest.raises(DirectStoreError, match="population SHA-256 mismatch"):
        validate_frozen_direct_store_population(shards)


def test_artifact_save_is_atomic_and_no_clobber(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    payload = {"safe": True}
    save_direct_store_artifact(payload, target)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    with pytest.raises(DirectStoreError, match="refusing to overwrite"):
        save_direct_store_artifact(payload, target)


def test_cli_labels_arm_and_refuses_unbound_execution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert direct_store_cli.main(["status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["official_mem0_comparison"] is False
    assert status["actual_mem0_executed"] is False
    assert status["schema_version"] == 2
    assert status["cli_runtime_binding"] == "blocked"

    with pytest.raises(SystemExit) as exc:
        direct_store_cli.main(["run-shard"])
    assert exc.value.code == 2
    assert "no concrete Mem0 runtime is bound" in capsys.readouterr().err
