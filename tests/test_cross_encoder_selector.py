from __future__ import annotations

import hashlib

import numpy as np
import pytest

import memory_condense.search.selectors.cross_encoder_selector as cross_encoder_module
from memory_condense.search.selectors.coverage_selector import CoverageSelectionReport
from memory_condense.search.selectors.cross_encoder_selector import (
    MS_MARCO_MODEL_ID,
    MS_MARCO_MODEL_REVISION,
    MS_MARCO_WEIGHTS_SHA256,
    MSMarcoCrossEncoderSelector,
    verify_ms_marco_checkpoint,
)
from memory_condense.domain.schemas import Chunk, RetrievalResult, Turn


def _result(index: int, text: str | None = None) -> RetrievalResult:
    value = text or f"candidate {index}"
    turn = Turn(
        turn_id=f"turn-{index}",
        source_id=f"source-{index}",
        role="user",
        text=value,
    )
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=f"chunk-{index}",
            turn_id=turn.turn_id,
            text=value,
            start_char=0,
            end_char=len(value),
            token_count=max(1, len(value.split())),
        ),
        turn=turn,
        score=1.0 - index / 10.0,
        route="causal_graph",
    )


class _FakeEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((list(pairs), dict(kwargs)))
        if isinstance(self.scores, BaseException):
            raise self.scores
        return np.asarray(self.scores)


class _FakeGrouper:
    def __init__(self):
        self.seen = []
        self.semantic_scores = None
        self.active_partition_total = None
        self.active_partition_inspected = None
        self.closed = False
        self.last_report = None
        self.last_candidate_trace = []

    def select(
        self,
        query,
        candidates,
        *,
        max_results=None,
        source_timestamps=None,
        semantic_scores=None,
        active_partition_total=None,
        active_partition_inspected=None,
    ):
        del query, source_timestamps
        self.seen = list(candidates)
        self.semantic_scores = semantic_scores
        self.active_partition_total = active_partition_total
        self.active_partition_inspected = active_partition_inspected
        self.last_candidate_trace = [
            {
                "chunk_id": candidate.chunk.chunk_id,
                "source_id": candidate.turn.source_id,
                "group_id": "event-1" if index < 2 else "event-2",
                "group_role": "representative" if index != 1 else "support",
            }
            for index, candidate in enumerate(candidates)
        ]
        selected = [candidates[0], *candidates[2:], candidates[1]]
        if max_results is not None:
            selected = selected[:max_results]
        self.last_report = CoverageSelectionReport(
            operator="all",
            cardinality=None,
            requires_completeness=True,
            input_candidates=len(candidates),
            inspected_candidates=len(candidates),
            classified_candidates=len(candidates),
            event_clusters=2,
            new_assignments=2,
            existing_assignments=1,
            null_assignments=0,
            uncertain_assignments=0,
            output_candidates=len(selected),
            representatives=2,
            supporting_candidates=1,
            workspace_tokens=1024,
            elapsed_s=0.01,
            frontier_candidates=len(candidates),
            frontier_attempted=len(candidates),
            routed_frontier_exhaustive=True,
            active_partition_total=active_partition_total,
            active_partition_inspected=active_partition_inspected,
            active_partition_exhaustive=(
                active_partition_inspected >= active_partition_total
                if active_partition_total is not None
                and active_partition_inspected is not None
                else None
            ),
            cardinality_deficit=1,
            prefix_model_id="Qwen/Qwen3-8B",
            prefix_model_revision="prefix-revision",
            prefix_checkpoint_sha256="b" * 64,
            prefix_device="cuda:0",
            prefix_dtype="float16",
            prefix_layers=2,
            prefix_attention_layer=1,
        )
        return selected

    def close(self):
        self.closed = True


class _BoundedFakeGrouper:
    def __init__(self, candidate_pool: int):
        self.candidate_pool = candidate_pool
        self.seen = []
        self.last_report = None
        self.last_candidate_trace = []

    def select(
        self,
        query,
        candidates,
        *,
        max_results=None,
        source_timestamps=None,
    ):
        del query, source_timestamps
        self.seen = list(candidates)
        inspected = min(len(candidates), self.candidate_pool)
        self.last_candidate_trace = [
            {
                "chunk_id": candidate.chunk.chunk_id,
                "source_id": candidate.turn.source_id,
                "group_role": (
                    "representative" if index < inspected else "uninspected"
                ),
            }
            for index, candidate in enumerate(candidates)
        ]
        selected = list(candidates)
        if max_results is not None:
            selected = selected[:max_results]
        self.last_report = CoverageSelectionReport(
            operator="all",
            cardinality=None,
            requires_completeness=True,
            input_candidates=len(candidates),
            inspected_candidates=inspected,
            classified_candidates=inspected,
            event_clusters=inspected,
            new_assignments=inspected,
            existing_assignments=0,
            null_assignments=0,
            uncertain_assignments=len(candidates) - inspected,
            output_candidates=len(selected),
            representatives=inspected,
            supporting_candidates=0,
            workspace_tokens=1024,
            elapsed_s=0.01,
        )
        return selected

    def close(self):
        pass


def test_cross_encoder_reranks_without_dropping_and_keeps_ties_stable():
    candidates = [_result(index) for index in range(4)]
    encoder = _FakeEncoder([0.2, 0.9, 0.9])
    selector = MSMarcoCrossEncoderSelector(
        encoder,
        candidate_pool=3,
        batch_size=64,
        max_length=256,
        max_workspace_tokens=8192,
    )

    selected = selector.select("Which candidates matter?", candidates)

    assert [row.chunk.chunk_id for row in selected] == [
        "chunk-1",
        "chunk-2",
        "chunk-0",
        "chunk-3",
    ]
    assert encoder.calls[0][1]["batch_size"] == 32
    assert encoder.calls[0][1]["show_progress_bar"] is False
    assert selector.last_report is not None
    assert selector.last_report.semantic_model_id == MS_MARCO_MODEL_ID
    assert selector.last_report.semantic_model_revision == MS_MARCO_MODEL_REVISION
    assert selector.last_report.semantic_checkpoint_sha256 == MS_MARCO_WEIGHTS_SHA256
    assert selector.last_report.output_candidates == 4
    assert selector.last_report.workspace_tokens == 8192
    assert selector.last_report.retained_transformer_state_bytes == 0
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace["chunk-1"]["cross_encoder_score"] == pytest.approx(0.9)
    assert trace["chunk-1"]["cross_encoder_rank"] == 1
    assert trace["chunk-2"]["cross_encoder_rank"] == 2
    assert trace["chunk-3"]["cross_encoder_score"] is None


def test_cross_encoder_failure_is_recall_safe_unless_strict():
    candidates = [_result(index) for index in range(3)]
    safe = MSMarcoCrossEncoderSelector(_FakeEncoder(RuntimeError("failed")))

    assert safe.select("query", candidates) == candidates
    assert safe.last_report is not None
    assert safe.last_report.fallback_reason == "RuntimeError: failed"
    assert all(
        row["cross_encoder_score"] is None
        for row in safe.last_candidate_trace
    )

    strict = MSMarcoCrossEncoderSelector(
        _FakeEncoder([float("nan")] * 3),
        strict=True,
    )
    with pytest.raises(ValueError, match="non-finite"):
        strict.select("query", candidates)


def test_cross_encoder_order_feeds_optional_qwen_duplicate_grouper():
    candidates = [_result(index) for index in range(3)]
    grouper = _FakeGrouper()
    selector = MSMarcoCrossEncoderSelector(
        _FakeEncoder([0.1, 0.8, 0.5]),
        duplicate_grouper=grouper,
    )

    selected = selector.select(
        "List all candidates",
        candidates,
        active_partition_total=9,
        active_partition_inspected=3,
    )

    assert [row.chunk.chunk_id for row in grouper.seen] == [
        "chunk-1",
        "chunk-2",
        "chunk-0",
    ]
    assert [row.chunk.chunk_id for row in selected] == [
        "chunk-1",
        "chunk-0",
        "chunk-2",
    ]
    assert grouper.semantic_scores == pytest.approx(
        {"chunk-1": 0.8, "chunk-2": 0.5, "chunk-0": 0.1}
    )
    assert grouper.active_partition_total == 9
    assert grouper.active_partition_inspected == 3
    assert selector.last_report is not None
    assert selector.last_report.event_clusters == 2
    assert selector.last_report.duplicate_grouping is not None
    assert selector.last_report.routed_frontier_exhaustive is True
    assert selector.last_report.active_partition_total == 9
    assert selector.last_report.active_partition_inspected == 3
    assert selector.last_report.active_partition_exhaustive is False
    assert selector.last_report.cardinality_deficit == 1
    assert selector.last_report.prefix_model_id == "Qwen/Qwen3-8B"
    assert selector.last_report.prefix_model_revision == "prefix-revision"
    assert selector.last_report.prefix_checkpoint_sha256 == "b" * 64
    assert selector.last_report.prefix_device == "cuda:0"
    assert selector.last_report.prefix_dtype == "float16"
    assert selector.last_report.prefix_layers == 2
    assert selector.last_report.prefix_attention_layer == 1
    assert selector.last_report.retained_transformer_state_bytes == 0
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace["chunk-1"]["group_role"] == "representative"
    assert trace["chunk-2"]["group_role"] == "support"

    selector.close()
    assert grouper.closed is True
    assert selector.encoder is None
    assert selector.last_candidate_trace == []


def test_cross_encoder_scores_full_frontier_before_bounded_qwen_grouping():
    candidates = [_result(index) for index in range(75)]
    scores = [0.0] * len(candidates)
    scores[72] = 10.0
    grouper = _BoundedFakeGrouper(candidate_pool=64)
    selector = MSMarcoCrossEncoderSelector(
        _FakeEncoder(scores),
        candidate_pool=128,
        duplicate_grouper=grouper,
    )

    selected = selector.select("List all matching events", candidates)

    assert len(selected) == len(candidates)
    assert {row.chunk.chunk_id for row in selected} == {
        row.chunk.chunk_id for row in candidates
    }
    assert grouper.seen[0].chunk.chunk_id == "chunk-72"
    assert selector.last_report is not None
    assert selector.last_report.semantic_inspected_candidates == 75
    assert selector.last_report.duplicate_grouping is not None
    assert selector.last_report.duplicate_grouping["inspected_candidates"] == 64
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace["chunk-72"]["cross_encoder_score"] == pytest.approx(10.0)
    assert trace["chunk-72"]["cross_encoder_rank"] == 1
    assert trace["chunk-72"]["group_role"] == "representative"
    assert trace["chunk-74"]["cross_encoder_score"] == pytest.approx(0.0)
    assert trace["chunk-74"]["group_role"] == "uninspected"


def test_companion_only_mode_preserves_global_order_without_scoring():
    candidates = [_result(index) for index in range(4)]
    encoder = _FakeEncoder([1.0, 0.0, 0.0, 0.0])
    selector = MSMarcoCrossEncoderSelector(
        encoder,
        semantic_rerank=False,
    )

    assert selector.select("query", candidates) == candidates
    assert encoder.calls == []
    assert selector.last_report is not None
    assert selector.last_report.semantic_rerank_enabled is False
    assert selector.last_report.semantic_inspected_candidates == 0
    assert selector.last_report.semantic_workspace_tokens == 0
    assert selector.last_report.uncertain_assignments == len(candidates)
    assert all(
        row["cross_encoder_score"] is None
        for row in selector.last_candidate_trace
    )


def test_score_only_mode_exposes_logits_to_grouper_without_reordering():
    candidates = [_result(index) for index in range(3)]
    grouper = _FakeGrouper()
    selector = MSMarcoCrossEncoderSelector(
        _FakeEncoder([0.1, 0.9, 0.5]),
        semantic_rerank=False,
        semantic_score_only=True,
        duplicate_grouper=grouper,
    )

    selector.select("List all candidates", candidates)

    assert [row.chunk.chunk_id for row in grouper.seen] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]
    assert grouper.semantic_scores == pytest.approx(
        {"chunk-0": 0.1, "chunk-1": 0.9, "chunk-2": 0.5}
    )
    assert selector.last_report is not None
    assert selector.last_report.semantic_rerank_enabled is False
    assert selector.last_report.semantic_score_only_enabled is True
    assert selector.last_report.semantic_inspected_candidates == 3
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace["chunk-1"]["cross_encoder_rank"] == 1


def test_cross_encoder_rejects_conflicting_semantic_modes():
    with pytest.raises(ValueError, match="mutually exclusive"):
        MSMarcoCrossEncoderSelector(
            _FakeEncoder([]),
            semantic_rerank=True,
            semantic_score_only=True,
        )


def test_cross_encoder_selects_one_companion_per_source_in_one_bounded_call():
    candidates_by_source = {
        "source-a": [_result(0), _result(1)],
        "source-b": [_result(2), _result(3)],
        "source-c": [_result(4), _result(5)],
    }
    encoder = _FakeEncoder([0.1, 0.8, 0.3, 0.9])
    selector = MSMarcoCrossEncoderSelector(encoder, candidate_pool=4)

    selected = selector.select_source_companions("query", candidates_by_source)

    assert [pair[1] for pair in encoder.calls[0][0]] == [
        "candidate 0",
        "candidate 2",
        "candidate 4",
        "candidate 1",
    ]
    assert {source_id: row.chunk.chunk_id for source_id, row in selected.items()} == {
        "source-a": "chunk-1",
        "source-b": "chunk-2",
        "source-c": "chunk-4",
    }
    assert selector.last_source_companion_report is not None
    assert selector.last_source_companion_report.input_candidates == 6
    assert selector.last_source_companion_report.inspected_candidates == 4
    assert selector.last_source_companion_report.selected_sources == 3
    assert selector.last_source_companion_report.retained_transformer_state_bytes == 0


def test_cross_encoder_companion_failure_keeps_first_raw_row_per_source():
    first = _result(0)
    candidates_by_source = {
        "source-a": [first, _result(1)],
        "source-b": [_result(2)],
    }
    selector = MSMarcoCrossEncoderSelector(_FakeEncoder(RuntimeError("failed")))

    selected = selector.select_source_companions("query", candidates_by_source)

    assert selected["source-a"] is first
    assert selected["source-b"] is candidates_by_source["source-b"][0]
    assert selector.last_source_companion_report is not None
    assert selector.last_source_companion_report.fallback_reason == (
        "RuntimeError: failed"
    )


def test_cross_encoder_rejects_wrong_score_count_without_dropping():
    candidates = [_result(index) for index in range(3)]
    selector = MSMarcoCrossEncoderSelector(_FakeEncoder([0.1, 0.2]))

    assert selector.select("query", candidates) == candidates
    assert selector.last_report is not None
    assert "2 scores for 3 candidates" in selector.last_report.fallback_reason


def test_pinned_checkpoint_verification_is_offline_and_exact(tmp_path, monkeypatch):
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    weights = b"safe weights"
    (tmp_path / "model.safetensors").write_bytes(weights)
    expected = hashlib.sha256(weights).hexdigest()
    monkeypatch.setattr(
        cross_encoder_module,
        "MS_MARCO_WEIGHTS_SHA256",
        expected,
    )

    assert verify_ms_marco_checkpoint(tmp_path) == expected


def test_pinned_checkpoint_verification_rejects_incomplete_or_wrong(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        verify_ms_marco_checkpoint(tmp_path)

    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="unexpected.*sha256"):
        verify_ms_marco_checkpoint(tmp_path)
