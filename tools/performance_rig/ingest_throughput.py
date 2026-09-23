"""Measure durable-capture and searchable-ingest throughput separately.

The default fake embedder is deterministic and suitable for quick CI.  Passing
``--embedder real`` opts into the repository's pinned BGE-M3 implementation and
an explicitly selected device.  The rig never contacts an answer provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Chunk
from memory_condense.modeling.embedding import DEFAULT_MODEL_NAME, EmbeddingService


FORMAT = "memory-condense-ingest-throughput-v3"


@dataclass(frozen=True, slots=True)
class RigConfig:
    turns: int = 64
    tokens_per_turn: int = 500
    capture_batch_size: int = 2
    drain_max_manifests: int | None = 16
    drain_max_chunks: int | None = None
    drain_max_tokens: int | None = None
    chunk_min_tokens: int = 120
    chunk_max_tokens: int = 250
    durable_p95_limit_seconds: float = 0.250
    searchable_min_tokens_per_second: float = 500.0
    searchable_base_deadline_seconds: float = 2.0
    offered_generation_tokens_per_second: float = 100.0

    def __post_init__(self) -> None:
        positive_ints = {
            "turns": self.turns,
            "tokens_per_turn": self.tokens_per_turn,
            "capture_batch_size": self.capture_batch_size,
            "chunk_min_tokens": self.chunk_min_tokens,
            "chunk_max_tokens": self.chunk_max_tokens,
        }
        for name, value in positive_ints.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.chunk_max_tokens < self.chunk_min_tokens:
            raise ValueError("chunk_max_tokens must be at least chunk_min_tokens")
        for name, value in (
            ("drain_max_manifests", self.drain_max_manifests),
            ("drain_max_chunks", self.drain_max_chunks),
            ("drain_max_tokens", self.drain_max_tokens),
        ):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        positive_floats = {
            "durable_p95_limit_seconds": self.durable_p95_limit_seconds,
            "searchable_min_tokens_per_second": (
                self.searchable_min_tokens_per_second
            ),
            "searchable_base_deadline_seconds": (
                self.searchable_base_deadline_seconds
            ),
            "offered_generation_tokens_per_second": (
                self.offered_generation_tokens_per_second
            ),
        }
        for name, value in positive_floats.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")


class DeterministicFakeEmbedder:
    """Stable hashing embedder with no model or network dependency."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "big") % self._dim
            vector[slot] += 1.0
        if not vector.any():
            vector[0] = 1.0
        return vector

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk.model_copy(update={"embedding": self._vector(chunk.text).tolist()})
            for chunk in chunks
        ]

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(query) for query in queries])

    def close(self) -> None:
        return None


class CountingEmbedder:
    """Record model work without changing the wrapped embedder's output."""

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.chunk_calls = 0
        self.chunk_inputs = 0
        self.query_calls = 0

    @property
    def dim(self) -> int:
        return int(getattr(self._delegate, "dim"))

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        self.chunk_calls += 1
        self.chunk_inputs += len(chunks)
        return getattr(self._delegate, "embed_chunks")(chunks)

    def embed_query(self, query: str) -> np.ndarray:
        self.query_calls += 1
        return np.asarray(getattr(self._delegate, "embed_query")(query))

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        self.query_calls += 1
        method = getattr(self._delegate, "embed_queries", None)
        if method is not None:
            return np.asarray(method(queries))
        return np.stack([getattr(self._delegate, "embed_query")(q) for q in queries])

    def close(self) -> None:
        close = getattr(self._delegate, "close", None)
        if close is not None:
            close()


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _sized_text(ordinal: int, target_tokens: int) -> tuple[str, str]:
    marker = f"ingestspeedunique{ordinal:08d}"
    sentence = (
        "This deterministic memory observation records an ordinary completed "
        "event with stable source provenance and searchable evidence. "
    )
    text = f"{marker}. "
    while count_tokens(text) < target_tokens:
        text += sentence
    return text, marker


def _batches(values: Sequence[Any], size: int) -> Sequence[Sequence[Any]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _positive_rate(numerator: int, seconds: float) -> float:
    return float(numerator) / max(seconds, 1e-12)


def run_benchmark(
    config: RigConfig,
    *,
    data_dir: Path,
    embedder: object,
    embedder_mode: str,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Run one write-path assay and return a JSON-serializable report.

    The caller owns ``embedder`` and remains responsible for closing it.
    """
    if embedder_mode not in {"fake", "real"}:
        raise ValueError("embedder_mode must be 'fake' or 'real'")
    records: list[tuple[str, str, str, None, str]] = []
    markers: dict[str, str] = {}
    input_tokens_by_turn: dict[str, int] = {}
    for ordinal in range(config.turns):
        text, marker = _sized_text(ordinal, config.tokens_per_turn)
        turn_id = f"ingest-speed-turn-{ordinal:08d}"
        records.append(("assistant", text, "ingest-speed-source", None, turn_id))
        markers[turn_id] = marker
        input_tokens_by_turn[turn_id] = count_tokens(text)

    counted = CountingEmbedder(embedder)
    capture_started_by_turn: dict[str, float] = {}
    searchable_latencies: dict[str, float] = {}
    drained_chunk_ids: set[str] = set()
    capture_batch_seconds: list[float] = []
    pending_depth_trace: list[int] = []
    drain_batch_seconds: list[float] = []
    captured_chunk_count = 0
    captured_chunk_tokens = 0
    last_turn_id = records[-1][4]
    verification_chunk: Chunk | None = None

    with MemoryCondenser(
        data_dir=data_dir,
        embedder=counted,
        # Production capture claims both its T1 manifest and T2 obligation in
        # the T0 transaction. This rig times T0/T1 and deliberately leaves T2
        # pending so extraction-model latency cannot contaminate base ingest.
        auto_extract=True,
        chunker_min_tokens=config.chunk_min_tokens,
        chunker_max_tokens=config.chunk_max_tokens,
    ) as condenser:
        phase_started = clock()
        for batch in _batches(records, config.capture_batch_size):
            started = clock()
            captured = condenser.capture_many(batch)
            ended = clock()
            capture_batch_seconds.append(ended - started)
            for turn, chunks in captured:
                capture_started_by_turn[turn.turn_id] = started
                captured_chunk_count += len(chunks)
                captured_chunk_tokens += sum(chunk.token_count for chunk in chunks)
            pending_depth_trace.append(condenser.pending_ingest_count())
        capture_phase_ended = clock()
        capture_embedding_chunk_calls = counted.chunk_calls
        capture_embedding_query_calls = counted.query_calls
        capture_embedding_calls = (
            capture_embedding_chunk_calls + capture_embedding_query_calls
        )
        pending_after_capture = condenser.pending_ingest_count()

        drain_started = clock()
        drain_batches = 0
        drained_chunk_count = 0
        drained_chunk_tokens = 0
        while condenser.pending_ingest_count():
            batch_started = clock()
            drained = condenser.drain_pending_ingests(
                max_manifests=config.drain_max_manifests,
                max_chunks=config.drain_max_chunks,
                max_tokens=config.drain_max_tokens,
                enrich=False,
            )
            batch_ended = clock()
            drain_batch_seconds.append(batch_ended - batch_started)
            if not drained:
                raise RuntimeError("pending journal made no forward progress")
            drained_at = batch_ended
            drain_batches += 1
            for turn, chunks in drained:
                drained_chunk_ids.update(chunk.chunk_id for chunk in chunks)
                searchable_latencies[turn.turn_id] = (
                    drained_at - capture_started_by_turn[turn.turn_id]
                )
                drained_chunk_count += len(chunks)
                drained_chunk_tokens += sum(chunk.token_count for chunk in chunks)
                if turn.turn_id == last_turn_id:
                    marker = markers[last_turn_id]
                    verification_chunk = next(
                        (chunk for chunk in chunks if marker in chunk.text),
                        verification_chunk,
                    )
            pending_depth_trace.append(condenser.pending_ingest_count())
        drain_ended = clock()

        if verification_chunk is None or verification_chunk.embedding is None:
            raise RuntimeError("final marker chunk was not returned by the drain")
        probe_started = clock()
        lexical_hits = condenser.retriever.lexical.search(
            markers[last_turn_id], limit=10
        )
        # Exercise the public dense path end to end: query embedding, ANN
        # lookup, label mapping, durable hydration, and result projection.
        # This is an operational smoke check, not a recall assertion. Exact
        # marker presence is proved by lexical retrieval; the approximate ANN
        # path need only return durably hydrated members of this fresh corpus.
        dense_hits = condenser.search(
            verification_chunk.text,
            k=10,
            ef_search=50,
        )
        probe_ended = clock()
        lexical_probe_verified = any(
            chunk_id == verification_chunk.chunk_id
            for chunk_id, _score in lexical_hits
        )
        dense_probe_verified = bool(dense_hits) and all(
            result.chunk.chunk_id in drained_chunk_ids for result in dense_hits
        )
        probe_verified = lexical_probe_verified and dense_probe_verified
        if not probe_verified:
            raise RuntimeError(
                "post-T1 index probes failed: "
                f"lexical={lexical_probe_verified}, dense={dense_probe_verified}"
            )

        pending_final = condenser.pending_ingest_count()
        enrichment_final = condenser.pending_enrichment_stats()

    if capture_embedding_calls != 0:
        raise RuntimeError("capture_many invoked the embedder")
    if set(searchable_latencies) != set(input_tokens_by_turn):
        raise RuntimeError("not every captured turn became searchable")
    if captured_chunk_count != drained_chunk_count:
        raise RuntimeError("captured and drained chunk counts differ")
    if captured_chunk_tokens != drained_chunk_tokens:
        raise RuntimeError("captured and drained token counts differ")

    capture_seconds = capture_phase_ended - phase_started
    drain_seconds = drain_ended - drain_started
    searchable_phase_seconds = drain_ended - phase_started
    verified_seconds = probe_ended - phase_started
    searchable_values = list(searchable_latencies.values())
    per_turn_deadline_passed = all(
        searchable_latencies[turn_id]
        <= max(
            config.searchable_base_deadline_seconds,
            input_tokens
            / config.searchable_min_tokens_per_second,
        )
        for turn_id, input_tokens in input_tokens_by_turn.items()
    )
    generation_interval_latency_passed = all(
        searchable_latencies[turn_id]
        <= input_tokens / config.offered_generation_tokens_per_second
        for turn_id, input_tokens in input_tokens_by_turn.items()
    )
    searchable_rate = _positive_rate(captured_chunk_tokens, drain_seconds)
    end_to_end_searchable_rate = _positive_rate(
        captured_chunk_tokens, searchable_phase_seconds
    )
    generation_headroom = (
        end_to_end_searchable_rate
        / config.offered_generation_tokens_per_second
    )
    drain_generation_headroom = (
        searchable_rate / config.offered_generation_tokens_per_second
    )

    return {
        "format": FORMAT,
        "embedder_mode": embedder_mode,
        "profile": "production_capture_through_t1",
        "config": asdict(config),
        "population": {
            "turns": config.turns,
            "input_token_proxy": sum(input_tokens_by_turn.values()),
            "captured_chunks": captured_chunk_count,
            "captured_chunk_token_proxy": captured_chunk_tokens,
        },
        "capture_to_durable": {
            "phase_elapsed_seconds": capture_seconds,
            "capture_batches": len(capture_batch_seconds),
            "turns_per_second": _positive_rate(config.turns, capture_seconds),
            "input_tokens_per_second": _positive_rate(
                sum(input_tokens_by_turn.values()), capture_seconds
            ),
            "batch_latency_p50_seconds": _percentile(
                capture_batch_seconds, 0.50
            ),
            "batch_latency_p95_seconds": _percentile(
                capture_batch_seconds, 0.95
            ),
            "batch_latency_max_seconds": max(capture_batch_seconds),
            "embedding_calls": capture_embedding_calls,
            "embedding_chunk_calls": capture_embedding_chunk_calls,
            "embedding_query_calls": capture_embedding_query_calls,
            "pending_depth_after_capture": pending_after_capture,
            "proxy_queue_exercised": False,
        },
        "capture_to_searchable": {
            "drain_elapsed_seconds": drain_seconds,
            "phase_elapsed_seconds": searchable_phase_seconds,
            "verified_elapsed_seconds": verified_seconds,
            "drain_batches": drain_batches,
            "drain_batch_latency_p50_seconds": _percentile(
                drain_batch_seconds, 0.50
            ),
            "drain_batch_latency_p95_seconds": _percentile(
                drain_batch_seconds, 0.95
            ),
            "drain_batch_latency_max_seconds": max(drain_batch_seconds),
            "chunk_tokens_per_drain_second": searchable_rate,
            "chunk_tokens_per_end_to_end_second": end_to_end_searchable_rate,
            "offered_generation_tokens_per_second": (
                config.offered_generation_tokens_per_second
            ),
            "generation_headroom_ratio": generation_headroom,
            "drain_generation_headroom_ratio": drain_generation_headroom,
            "chunks_per_drain_second": _positive_rate(
                drained_chunk_count, drain_seconds
            ),
            "latency_p50_seconds": _percentile(searchable_values, 0.50),
            "latency_p95_seconds": _percentile(searchable_values, 0.95),
            "latency_max_seconds": max(searchable_values),
            "verification_query_seconds": probe_ended - probe_started,
            "lexical_verification_passed": lexical_probe_verified,
            "dense_verification_passed": dense_probe_verified,
            "verification_passed": probe_verified,
            "embedding_chunk_calls": counted.chunk_calls,
            "embedding_chunk_inputs": counted.chunk_inputs,
            "embedding_query_calls": counted.query_calls,
            "pending_depth_peak": max(pending_depth_trace, default=0),
            "pending_depth_final": pending_final,
            "pending_depth_trace": pending_depth_trace,
        },
        "tier_2_enrichment": {
            "executed": False,
            "status": "deferred_by_profile",
            "pending_turn_count": int(enrichment_final["turn_count"]),
            "ready_turn_count": int(enrichment_final["ready_count"]),
            "failed_turn_count": int(enrichment_final["failed_count"]),
        },
        "sla": {
            "tier_0_core_capture_latency_passed": (
                _percentile(capture_batch_seconds, 0.95)
                <= config.durable_p95_limit_seconds
            ),
            "tier_0_core_no_model_passed": capture_embedding_calls == 0,
            "tier_0_proxy_loss_status": (
                "not_measured_by_this_core_ingest_rig"
            ),
            "tier_1_searchable_throughput_passed": (
                end_to_end_searchable_rate
                >= config.searchable_min_tokens_per_second
            ),
            "tier_1_generation_headroom_passed": generation_headroom >= 2.0,
            "tier_1_searchable_latency_passed": per_turn_deadline_passed,
            "tier_1_generation_interval_latency_passed": (
                generation_interval_latency_passed
            ),
            "tier_1_pending_drained_passed": pending_final == 0,
            "tier_2_enrichment_status": "deferred_with_durable_receipts",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", choices=("fake", "real"), default="fake")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--turns", type=int, default=64)
    parser.add_argument("--tokens-per-turn", type=int, default=500)
    parser.add_argument("--capture-batch-size", type=int, default=2)
    parser.add_argument("--drain-max-manifests", type=int, default=16)
    parser.add_argument("--drain-max-chunks", type=int)
    parser.add_argument("--drain-max-tokens", type=int)
    parser.add_argument(
        "--offered-generation-tokens-per-second", type=float, default=100.0
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--source-revision",
        help="exact tested source revision recorded in the JSON receipt",
    )
    parser.add_argument("--enforce-sla", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = RigConfig(
        turns=args.turns,
        tokens_per_turn=args.tokens_per_turn,
        capture_batch_size=args.capture_batch_size,
        drain_max_manifests=args.drain_max_manifests,
        drain_max_chunks=args.drain_max_chunks,
        drain_max_tokens=args.drain_max_tokens,
        offered_generation_tokens_per_second=(
            args.offered_generation_tokens_per_second
        ),
    )
    delegate: object = (
        DeterministicFakeEmbedder()
        if args.embedder == "fake"
        else EmbeddingService(
            model_name=args.model_name,
            device=args.device,
            batch_size=args.embedding_batch_size,
        )
    )

    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        data_dir = args.data_dir
        if data_dir is None:
            temporary = tempfile.TemporaryDirectory(
                prefix="memory-condense-ingest-speed-"
            )
            data_dir = Path(temporary.name)
        elif data_dir.exists() and any(data_dir.iterdir()):
            raise FileExistsError(
                "--data-dir must be absent or empty; the rig never deletes or "
                "reuses a prior store"
            )
        report = run_benchmark(
            config,
            data_dir=data_dir,
            embedder=delegate,
            embedder_mode=args.embedder,
        )
        report["source_revision"] = args.source_revision
    finally:
        close = getattr(delegate, "close", None)
        try:
            if close is not None:
                close()
        finally:
            if temporary is not None:
                temporary.cleanup()

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.enforce_sla:
        passed = all(
            value is True
            for key, value in report["sla"].items()
            if key.endswith("_passed")
        )
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
