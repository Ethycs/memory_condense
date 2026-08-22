from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import zlib
from datetime import datetime, timezone

import numpy as np
import pytest

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.eval.consolidation_replay import (
    FrozenQueryEmbedder,
    _source_rows,
    _comparison,
    stage_causal_store,
)
from memory_condense.eval.causal_benchmark import (
    CAUSAL_BUILD_PROTOCOL,
    CAUSAL_CACHE_REVISION,
    CAUSAL_MANIFEST_NAME,
    _held_out_query_batch,
    causal_consolidation_ingest_fn,
)
from memory_condense.eval.compiled_cache import compiled_store_ingest_fn
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample
from memory_condense.domain.schemas import Chunk


class ReplayEmbedder:
    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(token.encode()) % self._dim] += 1.0
        if not vector.any():
            vector[0] = 1.0
        return vector

    def embed_queries(self, texts):
        return np.stack([self.embed_query(text) for text in texts])

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk.model_copy(update={"embedding": self.embed_query(chunk.text).tolist()})
            for chunk in chunks
        ]


def test_frozen_query_embedder_refuses_unbatched_queries():
    embedder = FrozenQueryEmbedder({"known": [1.0, 0.0]})
    assert embedder.dim == 2
    assert embedder.embed_query("known").tolist() == [1.0, 0.0]
    with pytest.raises(KeyError, match="frozen batch"):
        embedder.embed_query("unknown")


def test_causal_query_batch_includes_enabled_retrieval_facets():
    sample = BenchmarkSample(
        sample_id="facets",
        questions=[
            BenchmarkQuestion(
                question_id="q",
                question=(
                    "Which happened first: I prepared the nursery, "
                    "I picked baby shower gifts, and I ordered a phone case?"
                ),
                answer="nursery",
            )
        ],
    )
    config = EvalConfig(
        retrieval=RetrievalConfig(
            mode="causal_graph",
            source_slots=6,
            query_facet_retrieval=True,
            query_facet_slots=6,
        )
    )

    assert _held_out_query_batch(sample, config) == [
        sample.questions[0].dated_question,
        "I prepared the nursery",
        "I picked baby shower gifts",
        "I ordered a phone case",
    ]


def test_source_reader_accepts_pre_source_id_historical_store(tmp_path):
    database = tmp_path / "historical.db"
    embedding = np.asarray([1.0, 0.0], dtype=np.float32).tobytes()
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE turns (
                turn_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                turn_id TEXT NOT NULL,
                text TEXT NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                embedding BLOB,
                lexical_weights TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO turns VALUES ('t1', 'user', 'old prompt', 'then')"
        )
        connection.execute(
            "INSERT INTO chunks VALUES ('c1', 't1', 'old prompt', 0, 10, 2, ?, NULL)",
            (embedding,),
        )

    rows = _source_rows(database)
    assert rows[0][:4] == (1, "user", "old prompt", None)
    assert rows[0][4][0].chunk_id == "c1"

    target = tmp_path / "uncertifiable-replay"
    with pytest.raises(ValueError, match="invalid created_at: t1"):
        stage_causal_store(database, target, ReplayEmbedder())
    assert not target.exists()


def test_causal_staging_never_retrieves_the_current_or_future_turn(tmp_path):
    source = tmp_path / "source"
    live = ReplayEmbedder()
    turns = [
        ("assistant", "alpha evidence establishes the first durable fact"),
        ("user", "What does alpha establish?"),
        ("assistant", "beta evidence is revealed only after that question"),
        ("user", "How do alpha and beta relate?"),
    ]
    with MemoryCondenser(
        data_dir=source,
        embedder=live,
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=40,
    ) as condenser:
        for role, text in turns:
            condenser.ingest(role, text)

    queries = [text for role, text in turns if role == "user"]
    frozen = FrozenQueryEmbedder(
        {query: live.embed_query(query) for query in queries}
    )
    events, stats = stage_causal_store(
        source / "memory.db",
        tmp_path / "staged",
        frozen,
        retrieval_k=3,
        max_event_nodes=3,
        new_event_nodes=1,
        max_prompt_tokens=128,
    )

    # The completed first user/assistant episode binds its one prior anchor to
    # the newly revealed beta response. The second user has no response, so it
    # cannot manufacture a future member or teach from its own query text.
    assert stats["events"] == 1
    assert events[0].event_id == "causal-user:2:part:0"
    assert len(events[0].causal_chunk_ids) == 1
    with sqlite3.connect(tmp_path / "staged" / "memory.db") as connection:
        selected_text = [
            connection.execute(
                "SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()[0]
            for chunk_id in events[0].chunk_ids
        ]
    assert any("alpha evidence" in text for text in selected_text)
    assert any("beta evidence" in text for text in selected_text)
    assert all("How do alpha and beta relate?" not in text for text in selected_text)


def test_causal_staging_closes_episode_before_next_source_timestamp(tmp_path):
    source = tmp_path / "source"
    live = ReplayEmbedder()
    turns = [
        ("system", "[session-a took place on Monday]", "session-a"),
        ("user", "What color is the first marker?", "session-a"),
        ("assistant", "The first marker is amber.", "session-a"),
        ("system", "[session-b took place on Friday]", "session-b"),
        ("user", "What color is the second marker?", "session-b"),
        ("assistant", "The second marker is violet.", "session-b"),
    ]
    with MemoryCondenser(
        data_dir=source,
        embedder=live,
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=40,
    ) as condenser:
        for role, text, source_id in turns:
            condenser.ingest(role, text, source_id=source_id)

    queries = [text for role, text, _source_id in turns if role == "user"]
    frozen = FrozenQueryEmbedder(
        {query: live.embed_query(query) for query in queries}
    )
    events, _stats = stage_causal_store(
        source / "memory.db",
        tmp_path / "staged",
        frozen,
        retrieval_k=3,
        max_event_nodes=4,
        new_event_nodes=2,
        max_prompt_tokens=128,
    )

    first_event = next(event for event in events if "causal-user:2" in event.event_id)
    with sqlite3.connect(tmp_path / "staged" / "memory.db") as connection:
        selected_text = [
            connection.execute(
                "SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()[0]
            for chunk_id in first_event.chunk_ids
        ]
    assert any("amber" in text for text in selected_text)
    assert all("session-b" not in text for text in selected_text)
    assert all("violet" not in text for text in selected_text)


def test_causal_staging_preserves_turn_chunk_and_timestamp_identity(tmp_path):
    source = tmp_path / "identity-source"
    live = ReplayEmbedder()
    first_at = datetime(2026, 8, 20, 9, 30, tzinfo=timezone.utc)
    second_at = datetime(2026, 8, 20, 9, 31, tzinfo=timezone.utc)
    with MemoryCondenser(
        data_dir=source,
        embedder=live,
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=40,
    ) as condenser:
        _first_turn, first_chunks = condenser.ingest(
            "assistant",
            "The preserved identity marker is amber.",
            source_id="identity-source",
            created_at=first_at,
            turn_id="stable-turn-1",
        )
        condenser.ingest(
            "user",
            "What is the preserved identity marker?",
            source_id="identity-source",
            created_at=second_at,
            turn_id="stable-turn-2",
        )

    query = "What is the preserved identity marker?"
    stage_causal_store(
        source / "memory.db",
        tmp_path / "identity-staged",
        FrozenQueryEmbedder({query: live.embed_query(query)}),
        retrieval_k=1,
        max_event_nodes=3,
        new_event_nodes=1,
        max_prompt_tokens=128,
    )

    with sqlite3.connect(tmp_path / "identity-staged" / "memory.db") as connection:
        turns = connection.execute(
            "SELECT turn_id, created_at, source_id FROM turns ORDER BY ordinal"
        ).fetchall()
        chunk_ids = {
            row[0] for row in connection.execute("SELECT chunk_id FROM chunks")
        }
    assert [row[0] for row in turns] == ["stable-turn-1", "stable-turn-2"]
    assert datetime.fromisoformat(turns[0][1]) == first_at
    assert datetime.fromisoformat(turns[1][1]) == second_at
    assert [row[2] for row in turns] == ["identity-source", "identity-source"]
    assert {item.chunk_id for item in first_chunks} <= chunk_ids


def test_comparison_reports_gains_and_losses_without_question_text():
    baseline = {
        "literal_recall": 0.5,
        "mean_context_tokens": 100.0,
        "rows": [
            {"question_id": "q0", "hit": True},
            {"question_id": "q1", "hit": False},
        ],
    }
    treatment = {
        "literal_recall": 0.5,
        "mean_context_tokens": 90.0,
        "rows": [
            {"question_id": "q0", "hit": False},
            {"question_id": "q1", "hit": True},
        ],
    }
    result = _comparison(treatment, baseline)
    assert result["gained_question_ids"] == ["q1"]
    assert result["lost_question_ids"] == ["q0"]
    assert result["mean_context_token_delta"] == -10.0


def test_causal_benchmark_ingest_learns_only_in_scratch_store(tmp_path):
    sample = BenchmarkSample(
        sample_id="sample-a",
        turns=[
            ("assistant", "The durable project marker is amber."),
            ("user", "Which marker did the project use?"),
            ("assistant", "It used the amber marker."),
        ],
        turn_source_ids=["session-a", "session-a", "session-a"],
        questions=[
            BenchmarkQuestion(
                question_id="q1",
                question="What color marker did the project use?",
                answer="amber",
            )
        ],
    )
    config = EvalConfig(
        retrieval=RetrievalConfig(
            mode="causal_consolidation",
            k=2,
            consolidation_training_k=2,
            consolidation_expansion_tokens=200,
        )
    )
    causal_cache = tmp_path / "causal-cache"
    ingest = causal_consolidation_ingest_fn(
        embedder=ReplayEmbedder(),
        causal_cache_root=causal_cache,
    )
    store = ingest(sample, config, tmp_path / "learned")
    cached_database = store.database_path
    try:
        stats = store.causal_consolidation_stats
        assert stats["staging"]["events"] == 1
        assert stats["learning"]["events_applied"] == 1
        assert stats["learning"]["graph"]["edges"] > 0
        packed = store.build_context(
            sample.questions[0].dated_question,
            recent_turns=0,
            k_memories=0,
            k_expansions=2,
            reheat_memories=False,
            learn_consolidation=False,
            consolidation_chunk_slots=3,
            consolidation_hops=2,
            consolidation_candidates=128,
            consolidation_diffusion_width=32,
        )
        assert any("amber" in excerpt for excerpt in packed.expansions)
        with pytest.raises(sqlite3.OperationalError):
            store.ingest("user", "This cache must reject writes.")
    finally:
        store.close()

    cached_database_bytes = cached_database.read_bytes()

    read_variant = config.model_copy(
        update={
            "retrieval": config.retrieval.model_copy(
                update={"consolidation_expansion_tokens": 300}
            )
        }
    )
    reopened = ingest(sample, read_variant, tmp_path / "unused-on-hit")
    try:
        assert reopened.database_path == cached_database
        assert reopened.causal_consolidation_stats["learning"][
            "events_applied"
        ] == 1
        assert len(list(causal_cache.glob("*/causal-store.json"))) == 1
        packed = reopened.build_context(
            sample.questions[0].dated_question,
            recent_turns=0,
            k_memories=0,
            k_expansions=2,
            reheat_memories=False,
            learn_consolidation=False,
            consolidation_chunk_slots=3,
            consolidation_hops=2,
            consolidation_candidates=128,
            consolidation_diffusion_width=32,
        )
        assert any("amber" in excerpt for excerpt in packed.expansions)
        with pytest.raises(sqlite3.OperationalError):
            reopened.ingest("user", "A cache hit must reject writes too.")
    finally:
        reopened.close()

    assert cached_database.read_bytes() == cached_database_bytes
    assert not cached_database.with_name(f"{cached_database.name}-wal").exists()
    assert not cached_database.with_name(f"{cached_database.name}-shm").exists()


def test_causal_benchmark_does_not_embed_prompts_rejected_by_write_bound(tmp_path):
    class CapturingEmbedder(ReplayEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.query_batches: list[list[str]] = []

        def embed_queries(self, texts):
            self.query_batches.append(list(texts))
            return super().embed_queries(texts)

    oversized = " ".join(["oversized"] * 200)
    sample = BenchmarkSample(
        sample_id="bounded-sample",
        turns=[
            ("assistant", "A compact prior fact."),
            ("user", oversized),
            ("assistant", "The oversized prompt has an outcome."),
            ("user", "Recall the compact fact?"),
            ("assistant", "The compact fact is recalled."),
        ],
        turn_source_ids=["s1"] * 5,
        questions=[
            BenchmarkQuestion(
                question_id="bounded-q",
                question="What fact was compact?",
                answer="fact",
            )
        ],
    )
    config = EvalConfig(
        retrieval=RetrievalConfig(
            mode="causal_consolidation",
            consolidation_max_training_prompt_tokens=10,
        )
    )
    embedder = CapturingEmbedder()
    store = causal_consolidation_ingest_fn(embedder=embedder)(
        sample,
        config,
        tmp_path / "bounded",
    )
    try:
        assert len(embedder.query_batches) == 2
        training_batch, held_out_batch = embedder.query_batches
        assert oversized not in training_batch
        assert training_batch == ["Recall the compact fact?"]
        assert held_out_batch == [sample.questions[0].dated_question]
        assert sample.questions[0].dated_question not in training_batch
        assert store.causal_consolidation_stats["staging"][
            "skipped_large_prompt"
        ] == 1
    finally:
        store.close()


def test_blind_causal_cache_prepare_never_embeds_held_out_questions(
    tmp_path,
    monkeypatch,
):
    class CapturingEmbedder(ReplayEmbedder):
        model_name = "test/blind-prepare"

        def __init__(self) -> None:
            super().__init__()
            self.query_batches: list[list[str]] = []

        def embed_queries(self, texts):
            self.query_batches.append(list(texts))
            return super().embed_queries(texts)

    held_out = "What is the secret held-out answer?"
    sample = BenchmarkSample(
        sample_id="blind-sample",
        turns=[
            ("assistant", "The durable marker is amber."),
            ("user", "Which durable marker was used?"),
            ("assistant", "The amber marker was used."),
        ],
        turn_source_ids=["s1"] * 3,
        questions=[
            BenchmarkQuestion(
                question_id="blind-q",
                question=held_out,
                answer="secret gold value",
            )
        ],
    )
    config = EvalConfig(
        retrieval=RetrievalConfig(mode="causal_graph", source_slots=2)
    )
    embedder = CapturingEmbedder()

    def held_out_forbidden(*_args, **_kwargs):
        raise AssertionError("blind preparation must not inspect QA probes")

    monkeypatch.setattr(
        "memory_condense.eval.causal_benchmark._held_out_query_batch",
        held_out_forbidden,
    )
    ingest = causal_consolidation_ingest_fn(
        embedder=embedder,
        causal_cache_root=tmp_path / "causal-cache",
        prepare_only=True,
    )

    first = ingest(sample, config, tmp_path / "first")
    first.close()
    batches_after_build = len(embedder.query_batches)
    second = ingest(sample, config, tmp_path / "second")
    second.close()

    assert ingest.prepare_only is True
    assert batches_after_build == 1
    assert len(embedder.query_batches) == batches_after_build
    assert "Which durable marker was used?" in embedder.query_batches[0]
    assert held_out not in embedder.query_batches[0]
    assert "secret gold value" not in embedder.query_batches[0]


def test_normal_and_blind_cold_builds_share_the_exact_training_batch(tmp_path):
    class CapturingEmbedder(ReplayEmbedder):
        model_name = "test/training-batch-parity"

        def __init__(self) -> None:
            super().__init__()
            self.query_batches: list[list[str]] = []

        def embed_queries(self, texts):
            self.query_batches.append(list(texts))
            return super().embed_queries(texts)

    sample = BenchmarkSample(
        sample_id="training-parity",
        turns=[
            ("assistant", "The first durable marker is amber."),
            ("user", "Which first marker was used?"),
            ("assistant", "The amber marker was used."),
            ("user", "Which marker should I remember now?"),
            ("assistant", "Remember amber."),
        ],
        turn_source_ids=["s1"] * 5,
        questions=[
            BenchmarkQuestion(
                question_id="training-parity-q",
                question="What was the durable marker?",
                answer="amber",
            )
        ],
    )
    config = EvalConfig(retrieval=RetrievalConfig(mode="causal_consolidation"))
    blind_embedder = CapturingEmbedder()
    normal_embedder = CapturingEmbedder()

    blind = causal_consolidation_ingest_fn(
        embedder=blind_embedder,
        causal_cache_root=tmp_path / "blind-cache",
        prepare_only=True,
    )(sample, config, tmp_path / "blind")
    normal = causal_consolidation_ingest_fn(
        embedder=normal_embedder,
        causal_cache_root=tmp_path / "normal-cache",
    )(sample, config, tmp_path / "normal")
    try:
        assert blind_embedder.query_batches == [
            [
                "Which first marker was used?",
                "Which marker should I remember now?",
            ]
        ]
        assert normal_embedder.query_batches[0] == blind_embedder.query_batches[0]
        assert normal_embedder.query_batches[1] == [
            sample.questions[0].dated_question
        ]
        for section in ("staging", "learning"):
            for field in (
                set(blind.causal_consolidation_stats[section])
                & set(normal.causal_consolidation_stats[section])
                - {"elapsed_s"}
            ):
                assert (
                    blind.causal_consolidation_stats[section][field]
                    == normal.causal_consolidation_stats[section][field]
                )
    finally:
        blind.close()
        normal.close()


def test_rebuilt_compiled_artifact_builds_identity_matched_causal_cache(
    tmp_path,
):
    class CapturingEmbedder(ReplayEmbedder):
        model_name = "test/blind-hit"

        def __init__(self) -> None:
            super().__init__()
            self.query_batches: list[list[str]] = []

        def embed_queries(self, texts):
            self.query_batches.append(list(texts))
            return super().embed_queries(texts)

    held_out = "What held-out marker was used?"
    sample = BenchmarkSample(
        sample_id="blind-hit",
        turns=[
            ("assistant", "The durable marker is amber."),
            ("user", "Which marker was used?"),
            ("assistant", "The amber marker was used."),
        ],
        turn_source_ids=["s1"] * 3,
        questions=[
            BenchmarkQuestion(
                question_id="blind-hit-q",
                question=held_out,
                answer="amber",
            )
        ],
    )
    config = EvalConfig(
        retrieval=RetrievalConfig(mode="causal_graph", source_slots=2)
    )
    causal_cache = tmp_path / "causal-cache"

    # Seed both linked layers, then remove the compiled root to reproduce an
    # otherwise valid causal hit whose source cache was lost.
    compiled_cache = tmp_path / "compiled-cache"
    seed_embedder = CapturingEmbedder()
    seed_ingest = causal_consolidation_ingest_fn(
        compiled_cache,
        embedder=seed_embedder,
        causal_cache_root=causal_cache,
        prepare_only=True,
    )
    seed_ingest(sample, config, tmp_path / "seed").close()
    first_causal_manifest = next(
        causal_cache.glob(f"*/{CAUSAL_MANIFEST_NAME}")
    )
    first_compiled_manifest_sha256 = json.loads(
        first_causal_manifest.read_text(encoding="utf-8")
    )["compiled_manifest_sha256"]
    shutil.rmtree(compiled_cache)

    hit_embedder = CapturingEmbedder()
    hit_ingest = causal_consolidation_ingest_fn(
        compiled_cache,
        embedder=hit_embedder,
        causal_cache_root=causal_cache,
        prepare_only=True,
    )
    store = hit_ingest(sample, config, tmp_path / "hit")
    try:
        receipts = store.blind_cache_receipts
        assert len(receipts["compiled"]) == 1
        assert len(receipts["causal"]) == 1
        assert list(compiled_cache.glob("*/compiled-store.json"))
        assert len(list(causal_cache.glob(f"*/{CAUSAL_MANIFEST_NAME}"))) == 2
        assert receipts["causal"][0]["compiled_manifest_sha256"] == (
            receipts["compiled"][0]["manifest_sha256"]
        )
        assert receipts["causal"][0]["compiled_manifest_sha256"] != (
            first_compiled_manifest_sha256
        )
        compiled_database = next(compiled_cache.glob("*/memory.db"))
        with (
            sqlite3.connect(compiled_database) as compiled_connection,
            sqlite3.connect(store.database_path) as causal_connection,
        ):
            compiled_turn_ids = compiled_connection.execute(
                "SELECT turn_id FROM turns ORDER BY ordinal"
            ).fetchall()
            causal_turn_ids = causal_connection.execute(
                "SELECT turn_id FROM turns ORDER BY ordinal"
            ).fetchall()
            compiled_chunk_ids = compiled_connection.execute(
                "SELECT chunk_id FROM chunks ORDER BY rowid"
            ).fetchall()
            causal_chunk_ids = causal_connection.execute(
                "SELECT chunk_id FROM chunks ORDER BY rowid"
            ).fetchall()
        assert causal_turn_ids == compiled_turn_ids
        assert causal_chunk_ids == compiled_chunk_ids
        assert held_out not in {
            text for batch in hit_embedder.query_batches for text in batch
        }
    finally:
        store.close()


def test_required_causal_cache_hit_is_read_only_and_reports_exact_pair(tmp_path):
    class CapturingEmbedder(ReplayEmbedder):
        model_name = "test/strict-causal-hit"

        def __init__(self) -> None:
            super().__init__()
            self.query_batches: list[list[str]] = []

        def embed_queries(self, texts):
            self.query_batches.append(list(texts))
            return super().embed_queries(texts)

    sample = BenchmarkSample(
        sample_id="strict-causal-hit",
        turns=[
            ("assistant", "The durable marker is amber."),
            ("user", "Which marker was used?"),
            ("assistant", "The amber marker was used."),
        ],
        turn_source_ids=["s1"] * 3,
        questions=[
            BenchmarkQuestion(
                question_id="strict-q",
                question="What marker was used?",
                answer="amber",
            )
        ],
    )
    config = EvalConfig(retrieval=RetrievalConfig(mode="causal_graph"))
    compiled_root = tmp_path / "compiled"
    causal_root = tmp_path / "causal"
    seed = CapturingEmbedder()
    causal_consolidation_ingest_fn(
        compiled_root,
        causal_cache_root=causal_root,
        embedder=seed,
        prepare_only=True,
    )(sample, config, tmp_path / "prepare").close()
    before = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in (compiled_root, causal_root)
        for path in root.rglob("*")
        if path.is_file()
    }

    scored = CapturingEmbedder()
    ingest = causal_consolidation_ingest_fn(
        compiled_root,
        causal_cache_root=causal_root,
        embedder=scored,
        require_cache_hit=True,
    )
    store = ingest(sample, config, tmp_path / "must-remain-unused")
    try:
        receipts = store.blind_cache_receipts
        assert len(receipts["compiled"]) == 1
        assert len(receipts["causal"]) == 1
        assert receipts["causal"][0]["compiled_cache_key"] == receipts["compiled"][0][
            "cache_key"
        ]
        assert receipts["causal"][0]["compiled_manifest_sha256"] == (
            receipts["compiled"][0]["manifest_sha256"]
        )
        assert scored.query_batches == [[sample.questions[0].dated_question]]
    finally:
        store.close()

    after = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in (compiled_root, causal_root)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not (tmp_path / "must-remain-unused").exists()


def test_required_causal_cache_miss_fails_before_query_or_write(tmp_path):
    class CapturingEmbedder(ReplayEmbedder):
        model_name = "test/strict-causal-miss"

        def __init__(self) -> None:
            super().__init__()
            self.query_batches: list[list[str]] = []

        def embed_queries(self, texts):
            self.query_batches.append(list(texts))
            return super().embed_queries(texts)

    sample = BenchmarkSample(
        sample_id="strict-causal-miss",
        turns=[("user", "Historical text")],
        questions=[
            BenchmarkQuestion(
                question_id="strict-miss-q",
                question="Held-out question?",
                answer="answer",
            )
        ],
    )
    config = EvalConfig(retrieval=RetrievalConfig(mode="causal_graph"))
    compiled_root = tmp_path / "compiled"
    causal_root = tmp_path / "causal"
    causal_root.mkdir()
    embedder = CapturingEmbedder()
    compiled_store_ingest_fn(compiled_root, embedder=embedder)(
        sample,
        config,
        tmp_path / "compiled-build",
    ).close()
    compiled_before = {
        path.relative_to(compiled_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in compiled_root.rglob("*")
        if path.is_file()
    }
    ingest = causal_consolidation_ingest_fn(
        compiled_root,
        causal_cache_root=causal_root,
        embedder=embedder,
        require_cache_hit=True,
    )

    with pytest.raises(RuntimeError, match="required causal-store cache entry"):
        ingest(sample, config, tmp_path / "unused")
    assert embedder.query_batches == []
    assert {
        path.relative_to(compiled_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in compiled_root.rglob("*")
        if path.is_file()
    } == compiled_before
    assert list(causal_root.iterdir()) == []
    assert not (tmp_path / "unused").exists()


def test_causal_manifest_attests_training_only_build_protocol(tmp_path):
    sample = BenchmarkSample(
        sample_id="protocol",
        turns=[
            ("assistant", "The durable marker is amber."),
            ("user", "Which marker was used?"),
            ("assistant", "The amber marker was used."),
        ],
        turn_source_ids=["s1"] * 3,
        questions=[
            BenchmarkQuestion(
                question_id="protocol-q",
                question="What marker was used?",
                answer="amber",
            )
        ],
    )
    config = EvalConfig(retrieval=RetrievalConfig(mode="causal_consolidation"))
    causal_cache = tmp_path / "causal-cache"
    ingest = causal_consolidation_ingest_fn(
        embedder=ReplayEmbedder(),
        causal_cache_root=causal_cache,
        prepare_only=True,
    )

    store = ingest(sample, config, tmp_path / "build")
    store.close()
    manifest_path = next(causal_cache.glob(f"*/{CAUSAL_MANIFEST_NAME}"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["cache_revision"] == CAUSAL_CACHE_REVISION
    assert manifest["build_protocol"] == CAUSAL_BUILD_PROTOCOL

    manifest["build_protocol"] = "legacy-held-out-cobatched"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="build protocol mismatch"):
        ingest(sample, config, tmp_path / "must-not-open")


@pytest.mark.parametrize(
    "tampered_field",
    ("compiled_cache_key", "compiled_manifest_sha256"),
)
def test_blind_prepare_rejects_causal_to_compiled_identity_mismatch(
    tmp_path,
    tampered_field,
):
    sample = BenchmarkSample(
        sample_id="compiled-link",
        turns=[
            ("assistant", "The durable marker is amber."),
            ("user", "Which marker was used?"),
            ("assistant", "The amber marker was used."),
        ],
        turn_source_ids=["s1"] * 3,
        questions=[
            BenchmarkQuestion(
                question_id="compiled-link-q",
                question="What marker was used?",
                answer="amber",
            )
        ],
    )
    config = EvalConfig(retrieval=RetrievalConfig(mode="causal_consolidation"))
    causal_cache = tmp_path / "causal-cache"
    ingest = causal_consolidation_ingest_fn(
        tmp_path / "compiled-cache",
        embedder=ReplayEmbedder(),
        causal_cache_root=causal_cache,
        prepare_only=True,
    )
    ingest(sample, config, tmp_path / "build").close()
    manifest_path = next(causal_cache.glob(f"*/{CAUSAL_MANIFEST_NAME}"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[tampered_field] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"compiled .*identity mismatch"):
        ingest(sample, config, tmp_path / "must-not-open")


def test_causal_factory_closes_owned_embedder_on_setup_error(
    tmp_path,
    monkeypatch,
):
    events: list[str] = []

    class TrackingEmbedder(ReplayEmbedder):
        model_name = "test/factory-close"

        def close(self):
            events.append("close")

    def fail_compiled_factory(*_args, **_kwargs):
        raise OSError("simulated cache-root failure")

    monkeypatch.setattr(
        "memory_condense.eval.causal_benchmark.EmbeddingService",
        lambda **_kwargs: TrackingEmbedder(),
    )
    monkeypatch.setattr(
        "memory_condense.eval.causal_benchmark.compiled_store_ingest_fn",
        fail_compiled_factory,
    )

    with pytest.raises(OSError, match="cache-root failure"):
        causal_consolidation_ingest_fn(
            tmp_path / "compiled",
            causal_cache_root=tmp_path / "causal",
        )

    assert events == ["close"]
