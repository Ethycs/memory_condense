"""Focused checks for the reproducible ingest-throughput rig."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_RIG_PATH = (
    Path(__file__).parents[1] / "tools" / "performance_rig" / "ingest_throughput.py"
)
_SPEC = importlib.util.spec_from_file_location("ingest_throughput_rig", _RIG_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RIG = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RIG
_SPEC.loader.exec_module(_RIG)


def test_fake_rig_separates_durable_and_searchable_boundaries(tmp_path: Path) -> None:
    config = _RIG.RigConfig(
        turns=6,
        tokens_per_turn=64,
        capture_batch_size=2,
        drain_max_manifests=2,
        durable_p95_limit_seconds=10.0,
        searchable_min_tokens_per_second=0.01,
        searchable_base_deadline_seconds=10.0,
        offered_generation_tokens_per_second=0.01,
    )

    report = _RIG.run_benchmark(
        config,
        data_dir=tmp_path / "store",
        embedder=_RIG.DeterministicFakeEmbedder(),
        embedder_mode="fake",
    )

    assert report["format"] == "memory-condense-ingest-throughput-v3"
    assert report["profile"] == "production_capture_through_t1"
    assert report["population"]["turns"] == 6
    assert report["population"]["captured_chunks"] > 0
    assert report["capture_to_durable"]["embedding_calls"] == 0
    assert report["capture_to_durable"]["embedding_chunk_calls"] == 0
    assert report["capture_to_durable"]["embedding_query_calls"] == 0
    assert report["capture_to_durable"]["capture_batches"] == 3
    assert report["capture_to_durable"]["batch_latency_p50_seconds"] >= 0.0
    assert report["capture_to_durable"]["batch_latency_p95_seconds"] >= 0.0
    assert report["capture_to_durable"]["batch_latency_max_seconds"] >= 0.0
    assert report["capture_to_durable"]["proxy_queue_exercised"] is False
    assert report["capture_to_durable"]["pending_depth_after_capture"] == 6
    searchable = report["capture_to_searchable"]
    assert searchable["drain_batches"] == 3
    assert searchable["drain_batch_latency_p50_seconds"] >= 0.0
    assert searchable["drain_batch_latency_p95_seconds"] >= 0.0
    assert searchable["drain_batch_latency_max_seconds"] >= 0.0
    assert searchable["embedding_chunk_calls"] == 3
    assert searchable["embedding_query_calls"] == 1
    assert searchable["chunk_tokens_per_end_to_end_second"] > 0.0
    assert searchable["generation_headroom_ratio"] == (
        searchable["chunk_tokens_per_end_to_end_second"] / 0.01
    )
    assert searchable["drain_generation_headroom_ratio"] == (
        searchable["chunk_tokens_per_drain_second"] / 0.01
    )
    assert searchable["pending_depth_final"] == 0
    assert searchable["lexical_verification_passed"] is True
    assert searchable["dense_verification_passed"] is True
    assert searchable["verification_passed"] is True
    assert searchable["pending_depth_trace"][-1] == 0
    assert report["sla"]["tier_0_core_capture_latency_passed"] is True
    assert report["sla"]["tier_0_core_no_model_passed"] is True
    assert report["sla"]["tier_0_proxy_loss_status"] == (
        "not_measured_by_this_core_ingest_rig"
    )
    assert report["sla"]["tier_1_searchable_throughput_passed"] is True
    assert report["sla"]["tier_1_generation_headroom_passed"] is True
    assert report["sla"]["tier_1_searchable_latency_passed"] is True
    assert report["sla"]["tier_1_generation_interval_latency_passed"] is True
    assert report["sla"]["tier_1_pending_drained_passed"] is True
    assert report["tier_2_enrichment"] == {
        "executed": False,
        "status": "deferred_by_profile",
        "pending_turn_count": 6,
        "ready_turn_count": 6,
        "failed_turn_count": 0,
    }
    assert report["sla"]["tier_2_enrichment_status"] == (
        "deferred_with_durable_receipts"
    )


def test_dense_smoke_does_not_treat_approximate_target_miss_as_ingest_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_search = _RIG.MemoryCondenser.search

    def omit_exact_query_chunk(condenser, query, **kwargs):
        return [
            result
            for result in original_search(condenser, query, **kwargs)
            if result.chunk.text != query
        ]

    monkeypatch.setattr(
        _RIG.MemoryCondenser, "search", omit_exact_query_chunk
    )
    report = _RIG.run_benchmark(
        _RIG.RigConfig(
            turns=6,
            tokens_per_turn=64,
            capture_batch_size=2,
            drain_max_manifests=2,
            durable_p95_limit_seconds=10.0,
            searchable_min_tokens_per_second=0.01,
            searchable_base_deadline_seconds=10.0,
            offered_generation_tokens_per_second=0.01,
        ),
        data_dir=tmp_path / "store",
        embedder=_RIG.DeterministicFakeEmbedder(),
        embedder_mode="fake",
    )

    assert report["capture_to_searchable"]["dense_verification_passed"] is True


def test_t0_gate_counts_capture_time_query_embedding(tmp_path, monkeypatch) -> None:
    original_capture_many = _RIG.MemoryCondenser.capture_many

    def query_during_capture(condenser, records):
        condenser._embedder.embed_query("forbidden capture-time model work")
        return original_capture_many(condenser, records)

    monkeypatch.setattr(
        _RIG.MemoryCondenser, "capture_many", query_during_capture
    )
    with pytest.raises(RuntimeError, match="capture_many invoked the embedder"):
        _RIG.run_benchmark(
            _RIG.RigConfig(
                turns=2,
                tokens_per_turn=16,
                capture_batch_size=2,
                drain_max_manifests=2,
                durable_p95_limit_seconds=10.0,
                searchable_min_tokens_per_second=0.01,
                searchable_base_deadline_seconds=10.0,
                offered_generation_tokens_per_second=0.01,
            ),
            data_dir=tmp_path / "capture-query-work",
            embedder=_RIG.DeterministicFakeEmbedder(),
            embedder_mode="fake",
        )


def test_fake_text_and_embeddings_are_deterministic() -> None:
    first_text, first_marker = _RIG._sized_text(7, 128)
    second_text, second_marker = _RIG._sized_text(7, 128)
    embedder = _RIG.DeterministicFakeEmbedder()

    assert first_text == second_text
    assert first_marker == second_marker
    assert _RIG.count_tokens(first_text) >= 128
    assert (embedder.embed_query(first_text) == embedder.embed_query(second_text)).all()


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -float("inf")])
def test_config_rejects_non_finite_or_boolean_rates(value: object) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        _RIG.RigConfig(offered_generation_tokens_per_second=value)


def test_run_rejects_an_unrecognized_embedder_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="embedder_mode"):
        _RIG.run_benchmark(
            _RIG.RigConfig(turns=1, tokens_per_turn=8),
            data_dir=tmp_path / "store",
            embedder=_RIG.DeterministicFakeEmbedder(),
            embedder_mode="mystery",
        )


def test_capture_percentiles_weight_batches_not_turns(tmp_path: Path) -> None:
    # Four batch durations are 10, 20, 30, and 1 seconds. The last batch has
    # only one turn; copying durations onto turns would incorrectly make p50
    # 20 seconds rather than the true batch p50 of 10 seconds.
    clock_values = iter(
        [
            0.0,
            1.0,
            11.0,
            12.0,
            32.0,
            33.0,
            63.0,
            64.0,
            65.0,
            66.0,
            67.0,
            68.0,
            69.0,
            70.0,
            71.0,
            72.0,
        ]
    )
    report = _RIG.run_benchmark(
        _RIG.RigConfig(
            turns=10,
            tokens_per_turn=16,
            capture_batch_size=3,
            drain_max_manifests=10,
            offered_generation_tokens_per_second=5.0,
        ),
        data_dir=tmp_path / "batch-percentiles",
        embedder=_RIG.DeterministicFakeEmbedder(),
        embedder_mode="fake",
        clock=lambda: next(clock_values),
    )

    durable = report["capture_to_durable"]
    assert durable["capture_batches"] == 4
    assert durable["batch_latency_p50_seconds"] == 10.0
    assert durable["batch_latency_p95_seconds"] == 30.0
    assert durable["batch_latency_max_seconds"] == 30.0
    assert report["sla"]["tier_0_core_capture_latency_passed"] is False
    searchable = report["capture_to_searchable"]
    assert searchable["drain_generation_headroom_ratio"] >= 2.0
    assert searchable["generation_headroom_ratio"] < 2.0
    assert report["sla"]["tier_1_generation_headroom_passed"] is False


def test_main_closes_owned_embedder_when_data_dir_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CloseTrackingEmbedder(_RIG.DeterministicFakeEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    delegate = CloseTrackingEmbedder()
    monkeypatch.setattr(_RIG, "EmbeddingService", lambda **_kwargs: delegate)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("do not reuse", encoding="utf-8")

    with pytest.raises(FileExistsError, match="absent or empty"):
        _RIG.main(["--embedder", "real", "--data-dir", str(occupied)])

    assert delegate.closed is True


def test_main_closes_owned_embedder_and_enforces_failed_sla(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class CloseTrackingEmbedder(_RIG.DeterministicFakeEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    delegate = CloseTrackingEmbedder()
    monkeypatch.setattr(_RIG, "EmbeddingService", lambda **_kwargs: delegate)

    exit_code = _RIG.main(
        [
            "--embedder",
            "real",
            "--turns",
            "1",
            "--tokens-per-turn",
            "16",
            "--offered-generation-tokens-per-second",
            "1000000000000",
            "--data-dir",
            str(tmp_path / "main-store"),
            "--enforce-sla",
        ]
    )

    assert exit_code == 1
    assert delegate.closed is True
    assert (
        json.loads(capsys.readouterr().out)["sla"]
        ["tier_1_generation_interval_latency_passed"]
        is False
    )
