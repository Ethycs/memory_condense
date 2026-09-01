from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.domain.schemas import Chunk, RetrievalResult, Turn
from tests.test_matched_eval_population import _publish, _retrieval
from tools.matched_eval.artifacts import read_sealed_json
from tools.matched_eval.contracts import MatchedEvalContractError
from tools.matched_eval.population import load_s0_population
from tools.matched_eval.query_expansion import (
    CHECKPOINT_DIR_NAME,
    PREFLIGHT_NAME,
    RUNTIME_LEDGER_NAME,
    RUNTIME_LEDGER_REPLAY_NAME,
    RUN_NAME,
    RUN_REPLAY_NAME,
    ExistingPartitionHybridSearch,
    FrozenSourceMembership,
    FrozenSourceNamespace,
    LockedQueryExpansionContext,
    PartitionRoutingReceipt,
    QueryExpansionBudget,
    QuerySearchResult,
    build_query_expansion_population,
    load_preflighted_query_expansion_population,
    materialize_search_queries,
    parse_query_plan,
    preflight_query_expansion,
    replay_query_expansion,
    run_query_expansion,
)
from tools import run_locked_query_expansion as query_expansion_cli
from tools.matched_eval import query_expansion as query_expansion_module


def _membership(
    source_id: str,
    *chunk_ids: str,
    metadata: tuple[str, ...] = (),
) -> FrozenSourceMembership:
    content = tuple(value for value in chunk_ids if value not in metadata)
    return FrozenSourceMembership(
        source_id=source_id,
        content_chunk_ids=content,
        metadata_chunk_ids=metadata,
        stream_sha256=identity_sha256(
            {
                "source_id": source_id,
                "content": list(content),
                "metadata": list(metadata),
            }
        ),
    )


def _namespace(population, *, extra_chunks: tuple[str, ...] = ()):
    sources = [
        _membership(f"turn-{row.ordinal}", f"chunk-{row.ordinal}")
        for row in population.rows
    ]
    if extra_chunks:
        sources.append(_membership("unrelated-history::episode-7", *extra_chunks))
    return FrozenSourceNamespace(
        snapshot_id=population.snapshot.snapshot_id,
        combined_store_receipt_sha256="c" * 64,
        sources=tuple(sources),
    )


def _candidate(
    *,
    chunk_id: str,
    source_id: str,
    text: str,
    score: float,
    created_at: datetime | None = None,
) -> RetrievalResult:
    turn_id = f"turn-for-{chunk_id}"
    turn = Turn(
        turn_id=turn_id,
        role="user",
        text=text,
        source_id=source_id,
        created_at=created_at or datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            turn_id=turn_id,
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=count_tokens(text),
        ),
        score=score,
        turn=turn,
        dense_score=score,
        lexical_score=max(0.0, score - 0.1),
        route="hybrid_partition",
    )


class _StructuredCompletions:
    def __init__(self, completion: str) -> None:
        self.completion = completion
        self.requests: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def create(self, **request):
        with self._lock:
            self.requests.append(request)
            ordinal = len(self.requests)
        return SimpleNamespace(
            id=f"structured-{ordinal}",
            model="fake-terra-query-v1",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.completion),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=32,
                completion_tokens=24,
                total_tokens=56,
            ),
        )


class _StructuredClient:
    def __init__(self, completion: str) -> None:
        self.max_retries = 0
        self.chat = SimpleNamespace(
            completions=_StructuredCompletions(completion)
        )
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakePartitionSearch:
    def __init__(
        self,
        namespace: FrozenSourceNamespace,
        hits: tuple[RetrievalResult, ...],
        *,
        fail: BaseException | None = None,
    ) -> None:
        self.namespace = namespace
        self.hits = hits
        self.fail = fail
        self.calls: list[tuple[str, ...]] = []

    def search_many(self, queries, *, budget):
        del budget
        query_tuple = tuple(queries)
        self.calls.append(query_tuple)
        if self.fail is not None:
            raise self.fail
        return tuple(
            QuerySearchResult(
                query_sha256=quote_sha256(query),
                hits=self.hits,
                routing_receipt=PartitionRoutingReceipt.create(
                    query=query,
                    namespace=self.namespace,
                    selected_partitions=("unrelated-history",),
                    routed_source_count=1,
                    active_partition_scan_status="applied",
                    active_partition_scan_contract="synthetic_complete_scan_v1",
                    active_partition_exhaustive=True,
                ),
            )
            for query in query_tuple
        )


def _population(tmp_path: Path, *, count: int = 1, include_s0: bool = False):
    source = load_s0_population(
        _publish(tmp_path, _retrieval(count)),
        expected_retrieval_sha256=None,
        expected_question_count=count,
    )
    namespace = _namespace(source, extra_chunks=("cross-prefix-chunk",))
    population = build_query_expansion_population(
        source,
        namespaces_by_question={
            row.packet.question_id: namespace for row in source.rows
        },
        include_s0_evidence=include_s0,
    )
    return source, namespace, population


def _valid_completion() -> str:
    return json.dumps(
        {
            "queries": [
                "gardening activities two weeks ago",
                "plants acquired during the previous month",
            ],
            "entities": ["garden", "plants"],
            "dates": ["two weeks ago", "last month"],
            "operators": ["enumerate_repeated_events", "timeline"],
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def test_preflight_is_gold_blind_and_seals_store_wide_non_borrowing_budget(
    tmp_path: Path,
) -> None:
    source, namespace, population = _population(tmp_path, count=2)

    artifact = preflight_query_expansion(population, output_root=tmp_path / "arm")

    payload = artifact.payload
    assert payload["question_count"] == 2
    assert payload["required_authorized_provider_calls"] == 2
    assert payload["provider_calls"] == 0
    assert payload["gold_loaded"] is False
    assert payload["scope_policy"] == "entire_frozen_combined_shard_store"
    assert payload["partition_route"] == (
        "global_coarse_rank_then_top4_complete_partition_search"
    )
    assert payload["source_prefix_filter_used"] is False
    assert payload["known_history_filter_used"] is False
    assert payload["question_id_filter_used"] is False
    assert payload["budget"]["partition_slots"] == 4
    assert payload["budget"]["candidate_token_cap"] == 2_400
    assert payload["budget_id"] == population.budget.budget_id
    assert payload["namespaces"][0]["namespace_id"] == namespace.namespace_id
    assert payload["namespaces"][0]["source_count"] == 3
    assert payload["namespaces"][0]["total_chunk_count"] == 3
    assert payload["source_population_id"] == source.population_id
    assert read_sealed_json(tmp_path / "arm" / PREFLIGHT_NAME).sha256 == (
        artifact.sha256
    )
    # Namespace coordinates are sealed in preflight, never placed in prompts.
    for row in population.rows:
        prompt_text = "\n".join(message["content"] for message in row.messages)
        assert "unrelated-history::episode-7" not in prompt_text
        assert "Choice 0 was blue." not in prompt_text


def test_optional_s0_context_is_bounded_and_explicit(tmp_path: Path) -> None:
    _source, _namespace_value, population = _population(
        tmp_path,
        count=1,
        include_s0=True,
    )

    row = population.rows[0]
    assert row.s0_evidence_included is True
    assert row.s0_context_token_proxy <= population.budget.max_s0_context_tokens
    prompt_text = "\n".join(message["content"] for message in row.messages)
    assert "Choice 0 was blue." in prompt_text
    assert "reference" not in prompt_text.casefold()


def test_provider_population_rebuild_uses_only_retrieval_and_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _namespace_value, population = _population(tmp_path, count=2)
    output = tmp_path / "arm"
    sealed = preflight_query_expansion(population, output_root=output)

    def forbidden_database(*_args, **_kwargs):
        raise AssertionError("provider population must not open memory.db")

    monkeypatch.setattr(query_expansion_module, "Database", forbidden_database)
    rebuilt, observed = load_preflighted_query_expansion_population(
        tmp_path / "retrieval.json",
        output_root=output,
        expected_retrieval_sha256=source.retrieval_sha256,
        expected_question_count=2,
    )

    assert observed.sha256 == sealed.sha256
    assert rebuilt.preflight_projection() == population.preflight_projection()


def test_query_plan_is_strict_bounded_and_materializes_operator_hints() -> None:
    budget = QueryExpansionBudget()
    plan = parse_query_plan(_valid_completion(), budget=budget)

    queries = materialize_search_queries(
        plan,
        dated_question="[Question asked at 2026/08/01]\nWhat did I do?",
        budget=budget,
    )

    assert queries[:2] == plan.queries
    assert len(queries) <= budget.max_materialized_queries
    assert any("two weeks ago" in query for query in queries)
    assert any("every repeated event" in query for query in queries)
    with pytest.raises(MatchedEvalContractError, match="schema"):
        parse_query_plan(
            '{"queries":["x"],"entities":[],"dates":[],"operators":[],"extra":1}',
            budget=budget,
        )
    with pytest.raises(MatchedEvalContractError, match="unknown operator"):
        parse_query_plan(
            '{"queries":["x"],"entities":[],"dates":[],"operators":["oracle"]}',
            budget=budget,
        )


def test_run_accepts_cross_prefix_hit_dedups_s0_after_selection_and_replays(
    tmp_path: Path,
) -> None:
    source, namespace, population = _population(tmp_path)
    output = tmp_path / "arm"
    preflight_query_expansion(population, output_root=output)
    protected = source.rows[0].packet.protected_evidence[0]
    duplicate = _candidate(
        chunk_id="chunk-0",
        source_id=protected.source_id,
        text=protected.text,
        score=0.99,
    )
    cross_prefix = _candidate(
        chunk_id="cross-prefix-chunk",
        source_id="unrelated-history::episode-7",
        text="I planted rosemary and mint two weeks ago.",
        score=0.91,
    )
    search = _FakePartitionSearch(namespace, (duplicate, cross_prefix))
    client = _StructuredClient(_valid_completion())

    result = run_query_expansion(
        population,
        output_root=output,
        retrievers_by_namespace={namespace.namespace_id: search},
        enable_provider=True,
        authorized_provider_calls=1,
        client=client,
        max_concurrency=1,
    )

    assert result.physical_provider_calls == 1
    assert len(client.chat.completions.requests) == 1
    assert result.run_artifact.payload["provider_completion_batch"]["provenance"][
        "benchmark_provenance"
    ]["gateway_url"] == "https://central-dev.zt:4000/v1"
    row = result.run_artifact.payload["questions"][0]
    assert row["disposition"] == "added"
    assert row["source_prefix_filter_used"] is False
    assert row["selected_before_dedup_candidate_ids"] == row["candidate_ids"]
    assert len(row["dedup_excluded_candidate_ids"]) == 1
    assert len(row["admitted_candidate_ids"]) == 1
    assert row["admitted_candidates"][0]["source_id"] == (
        "unrelated-history::episode-7"
    )
    assert row["admitted_candidates"][0]["text"] == cross_prefix.chunk.text
    assert row["routing_receipts"][0]["partition_slots"] == 4
    assert row["routing_receipts"][0]["source_prefix_filter_used"] is False
    ledger = result.runtime_ledger_artifact.payload
    assert ledger["total_provider_calls"] == 1
    assert ledger["rows"][0]["disposition"] == "added"
    assert ledger["rows"][0]["dedup_excluded_ids"] == (
        row["dedup_excluded_candidate_ids"]
    )

    replay = replay_query_expansion(
        population,
        output_root=output,
        retrievers_by_namespace={namespace.namespace_id: search},
        expected_run_sha256=result.run_artifact.sha256,
        max_concurrency=1,
    )
    assert replay.physical_provider_calls == 0
    assert replay.checkpoint_hits == 1
    assert replay.run_artifact.sha256 == result.run_artifact.sha256
    assert replay.runtime_ledger_artifact.sha256 == (
        result.runtime_ledger_artifact.sha256
    )
    assert read_sealed_json(output / RUN_NAME).sha256 == read_sealed_json(
        output / RUN_REPLAY_NAME
    ).sha256
    assert read_sealed_json(output / RUNTIME_LEDGER_NAME).sha256 == (
        read_sealed_json(output / RUNTIME_LEDGER_REPLAY_NAME).sha256
    )
    assert len(list((output / CHECKPOINT_DIR_NAME).glob("*.request.json"))) == 1
    assert len(list((output / CHECKPOINT_DIR_NAME).glob("*.response.json"))) == 1


@pytest.mark.parametrize(
    ("completion", "bad_source", "reason_prefix"),
    (
        ("not json", False, "invalid_query_plan"),
        (_valid_completion(), True, "retrieval_failed_closed"),
    ),
)
def test_invalid_plan_or_out_of_namespace_hit_fails_closed_to_noop(
    tmp_path: Path,
    completion: str,
    bad_source: bool,
    reason_prefix: str,
) -> None:
    _source, namespace, population = _population(tmp_path)
    output = tmp_path / "arm"
    preflight_query_expansion(population, output_root=output)
    outside = _candidate(
        chunk_id="outside-chunk",
        source_id="secret-known-history::episode",
        text="This chunk is outside the frozen store.",
        score=1.0,
    )
    search = _FakePartitionSearch(namespace, (outside,) if bad_source else ())

    result = run_query_expansion(
        population,
        output_root=output,
        retrievers_by_namespace={namespace.namespace_id: search},
        enable_provider=True,
        authorized_provider_calls=1,
        client=_StructuredClient(completion),
        max_concurrency=1,
    )

    row = result.run_artifact.payload["questions"][0]
    assert row["disposition"] == "no_op"
    assert row["candidate_ids"] == []
    assert row["admitted_candidates"] == []
    assert row["reason"].startswith(reason_prefix)
    if completion == "not json":
        assert search.calls == []
    assert result.runtime_ledger_artifact.payload["rows"][0]["admitted_ids"] == []


def test_authorization_fails_before_checkpoint_creation(tmp_path: Path) -> None:
    _source, namespace, population = _population(tmp_path)
    output = tmp_path / "arm"
    preflight_query_expansion(population, output_root=output)

    with pytest.raises(MatchedEvalContractError, match="exactly equal 1"):
        run_query_expansion(
            population,
            output_root=output,
            retrievers_by_namespace={
                namespace.namespace_id: _FakePartitionSearch(namespace, ())
            },
            enable_provider=True,
            authorized_provider_calls=0,
            client=_StructuredClient(_valid_completion()),
        )

    assert not (output / CHECKPOINT_DIR_NAME).exists()
    assert not (output / RUN_NAME).exists()


@pytest.mark.parametrize(
    ("enabled", "authorized", "message"),
    (
        (False, 100, "requires --enable-provider"),
        (True, 99, "must exactly equal 100"),
    ),
)
def test_locked_cli_authorization_fails_before_loading_context(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    authorized: int,
    message: str,
) -> None:
    def forbidden(_args):
        raise AssertionError("context must not load before exact authorization")

    monkeypatch.setattr(query_expansion_cli, "_load_preflight_population", forbidden)
    args = SimpleNamespace(
        enable_provider=enabled,
        authorized_provider_calls=authorized,
    )

    with pytest.raises(MatchedEvalContractError, match=message):
        query_expansion_cli._provider_locked(args)


def test_split_cli_keeps_provider_and_store_capabilities_disjoint() -> None:
    provider = query_expansion_cli._parser().parse_args(
        [
            "provider-run",
            "--enable-provider",
            "--authorized-provider-calls",
            "100",
        ]
    )
    materialize = query_expansion_cli._parser().parse_args(["materialize"])

    assert not any(
        hasattr(provider, name)
        for name in ("store_root", "policy", "qwen_prefix", "device")
    )
    assert not any(
        hasattr(materialize, name)
        for name in ("enable_provider", "authorized_provider_calls", "api_key_env")
    )


def test_locked_context_revalidates_database_and_index_bytes(tmp_path: Path) -> None:
    source, namespace, population = _population(tmp_path)
    store = tmp_path / "combined-store"
    store.mkdir()
    database_path = store / "memory.db"
    index_path = store / "hnsw_index.bin"
    database_path.write_bytes(b"sealed database")
    index_path.write_bytes(b"sealed index")
    namespace_id = namespace.namespace_id
    context = LockedQueryExpansionContext(
        population=population,
        store_dirs_by_namespace={namespace_id: store},
        database_sha256_by_namespace={
            namespace_id: file_sha256(database_path)
        },
        index_sha256_by_namespace={namespace_id: file_sha256(index_path)},
        shard_offsets_by_question={source.rows[0].packet.question_id: 0},
    )

    context.revalidate_store_bytes()
    index_path.write_bytes(b"changed index")
    with pytest.raises(MatchedEvalContractError, match="index changed"):
        context.revalidate_store_bytes()


def test_existing_wrapper_uses_global_coarse_top4_without_question_filter(
    tmp_path: Path,
) -> None:
    _source, namespace, _population_value = _population(tmp_path)
    hit = _candidate(
        chunk_id="cross-prefix-chunk",
        source_id="unrelated-history::episode-7",
        text="A globally routed exact span.",
        score=0.8,
    )

    class Condenser:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []
            self.last_partition_routing_report = {}

        def search_hybrid_graph(self, query: str, **kwargs):
            self.calls.append((query, kwargs))
            self.last_partition_routing_report = {
                "selected_partitions": ["unrelated-history"],
                "routed_sources": 1,
                "active_partition_scan_status": "bypassed",
                "active_partition_scan_contract": "",
                "active_partition_exhaustive": None,
            }
            return [hit]

    condenser = Condenser()
    search = ExistingPartitionHybridSearch(condenser, namespace)

    rows = search.search_many(
        ("business milestone four weeks ago",),
        budget=QueryExpansionBudget(),
    )

    assert rows[0].hits == (hit,)
    query, kwargs = condenser.calls[0]
    assert query == "business milestone four weeks ago"
    assert kwargs["source_partition_routing"] is True
    assert kwargs["source_partition_slots"] == 4
    assert kwargs["source_slots"] == 0
    assert "source_ids" not in kwargs
    assert "question_id" not in kwargs
    receipt = rows[0].routing_receipt.projection()
    assert receipt["scope_policy"] == "entire_frozen_combined_shard_store"
    assert receipt["source_prefix_filter_used"] is False
    assert receipt["question_id_filter_used"] is False
