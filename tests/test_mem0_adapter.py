from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense._tokenizer import count_tokens, tokenizer_proxy_identity
from memory_condense.eval.mem0_adapter import (
    MEM0AI_PIN,
    MEM0_API_VERSION,
    MEM0_ATTRIBUTION_KIND,
    MEM0_BM25_MODEL,
    MEM0_CERTIFIED_RENDERING,
    MEM0_CONTEXT_SEPARATOR,
    MEM0_DATE_EXPOSURE_KIND,
    MEM0_ENRICHED_RENDERING,
    MEM0_PROVIDER_USAGE_STATUS,
    MEM0_SPACY_MODEL,
    Mem0Adapter,
    Mem0AttributionError,
    Mem0ConfigurationError,
    Mem0DependencyError,
    Mem0OSSBackendFactory,
    Mem0PoisonedError,
    Mem0PromptBudgetError,
    Mem0ProtocolError,
    Mem0StackIdentity,
)
from memory_condense.loader import BenchmarkSample


class _StepClock:
    def __init__(self, step: float = 0.25) -> None:
        self.value = -step
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


class _FakeBackend:
    def __init__(
        self,
        *,
        add_responses=(),
        search_response=None,
        events=None,
        delete_error: BaseException | None = None,
        close_error: BaseException | None = None,
        add_error: BaseException | None = None,
        runtime_identity=None,
    ):
        self.add_responses = list(add_responses)
        self.search_response = (
            {"results": []} if search_response is None else search_response
        )
        self.add_calls = []
        self.search_calls = []
        self.delete_calls = []
        self.events = events if events is not None else []
        self.delete_error = delete_error
        self.close_error = close_error
        self.add_error = add_error
        self.runtime_identity = {} if runtime_identity is None else runtime_identity

    def add(self, messages, **kwargs):
        self.add_calls.append((messages, kwargs))
        if self.add_error is not None:
            raise self.add_error
        if self.add_responses:
            return self.add_responses.pop(0)
        return {"results": []}

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return self.search_response

    def delete_all(self, **kwargs):
        self.delete_calls.append(kwargs)
        self.events.append(("delete_all", kwargs["user_id"]))
        if self.delete_error is not None:
            raise self.delete_error

    def close(self):
        self.events.append(("backend_close", None))
        if self.close_error is not None:
            raise self.close_error


class _FakeVectorClient:
    def __init__(self, events, error: BaseException | None = None):
        self.events = events
        self.error = error

    def close(self):
        self.events.append(("vector_close", None))
        if self.error is not None:
            raise self.error


def _prompt(query: str, context: str) -> str:
    return f"SYSTEM\nQuestion: {query}\nContext:\n{context}\nAnswer:"


def _single_batch_sample(
    sample_id: str = "sample",
    *,
    source: str = "session_1",
    user: str = "fact",
) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=sample_id,
        turns=[("user", user), ("assistant", "ack")],
        turn_source_ids=[source, source],
    )


def _config(state_root: Path, *, remote: bool = False) -> dict:
    vector_config: dict = {
        "collection_name": "mem0-longmemeval",
        "embedding_model_dims": 8,
        "on_disk": True,
    }
    if remote:
        vector_config["url"] = "http://qdrant.invalid:6333"
    else:
        vector_config["path"] = str(state_root / "qdrant")
    return {
        "version": MEM0_API_VERSION,
        "custom_instructions": None,
        "reranker": None,
        "llm": {"provider": "fake-llm", "config": {"model": "llm-1"}},
        "embedder": {
            "provider": "fake-embedder",
            "config": {"model": "embed-1"},
        },
        "vector_store": {"provider": "qdrant", "config": vector_config},
        "history_db_path": str(state_root / "history.db"),
    }


def _stack_identity() -> Mem0StackIdentity:
    return Mem0StackIdentity(
        dependency_versions={
            "mem0ai": MEM0AI_PIN,
            "qdrant-client": "1.15.1",
            "fastembed": "0.7.3",
            "spacy": "3.8.7",
            "en-core-web-sm": "3.8.0",
        },
        bm25_model=MEM0_BM25_MODEL,
        spacy_model=MEM0_SPACY_MODEL,
        bm25_operational=True,
        entity_extraction_operational=True,
    )


_CREATED = "2024-01-01T09:00:00+00:00"


def test_ingest_matches_official_date_sort_and_blind_one_or_two_turn_slices():
    sample = BenchmarkSample(
        sample_id="chronology",
        turns=[
            ("system", "[late took place at 2025/03/04 (Tue) 11:00]"),
            ("assistant", "late-a"),
            ("user", "late-u"),
            ("assistant", "late-singleton"),
            ("system", "[early took place at 2024/01/01 (Mon) 09:00]"),
            ("user", "early-u"),
            ("assistant", "early-a"),
            ("system", "[middle took place at 2024/06/02 (Sun) 10:30]"),
            ("user", "middle-u1"),
            ("user", "middle-u2"),
            ("assistant", "middle-a1"),
            ("assistant", "middle-a2"),
        ],
        turn_source_ids=[
            "late",
            "late",
            "late",
            "late",
            "early",
            "early",
            "early",
            "middle",
            "middle",
            "middle",
            "middle",
            "middle",
        ],
    )
    backend = _FakeBackend(
        add_responses=[
            {"results": [{"id": "early-memory"}]},
            {"results": []},
            {"results": []},
            {"results": []},
            {"results": [{"id": "late-memory"}]},
        ]
    )
    adapter = Mem0Adapter(
        backend=backend,
        token_counter=len,
        clock=_StepClock(),
        user_scope_factory=lambda _: "scope",
    )

    result = adapter.ingest_sample(sample)

    assert [call[0] for call in backend.add_calls] == [
        [
            {"role": "user", "content": "early-u"},
            {"role": "assistant", "content": "early-a"},
        ],
        [
            {"role": "user", "content": "middle-u1"},
            {"role": "user", "content": "middle-u2"},
        ],
        [
            {"role": "assistant", "content": "middle-a1"},
            {"role": "assistant", "content": "middle-a2"},
        ],
        [
            {"role": "assistant", "content": "late-a"},
            {"role": "user", "content": "late-u"},
        ],
        [{"role": "assistant", "content": "late-singleton"}],
    ]
    assert [ref.source for ref in result.batches_added] == [
        "early",
        "middle",
        "middle",
        "late",
        "late",
    ]
    assert [ref.session_index for ref in result.batches_added] == [1, 2, 2, 3, 3]
    assert [ref.original_session_index for ref in result.batches_added] == [2, 3, 3, 1, 1]
    assert result.batches_added[-1].turn_count == 1
    assert result.batches_added[-1].roles == ("assistant",)
    assert not hasattr(result.batches_added[0], "user_text")
    assert not hasattr(result.batches_added[0], "assistant_text")
    assert all("metadata" not in kwargs for _messages, kwargs in backend.add_calls)
    assert result.date_exposure_kind == MEM0_DATE_EXPOSURE_KIND
    assert result.attribution_kind == MEM0_ATTRIBUTION_KIND
    assert result.supports_exact_source_provenance is False
    assert result.stats.add_calls == 5
    assert result.stats.add_attempted_calls == 5
    assert result.stats.add_completed_calls == 5
    assert result.stats.add_failed_calls == 0
    assert result.official_longmemeval_protocol is False
    assert result.comparison_certified is False
    assert result.stats.add_latency_s == pytest.approx(1.25)
    assert result.stats.add_raw_message_tokens == sum(
        len("\n".join(f"{row['role']}: {row['content']}" for row in messages))
        for messages, _kwargs in backend.add_calls
    )


def test_request_window_attribution_uses_only_prior_ten_messages_and_current_batch():
    turns = []
    sources = []
    for index in range(1, 8):
        turns.extend([("user", f"u{index}"), ("assistant", f"a{index}")])
        sources.extend(["s", "s"])
    backend = _FakeBackend(
        add_responses=[
            *({"results": []} for _ in range(6)),
            {"results": [{"id": "last"}]},
        ]
    )
    adapter = Mem0Adapter(
        backend=backend,
        token_counter=len,
        user_scope_factory=lambda _: "scope",
    )

    result = adapter.ingest(
        BenchmarkSample(
            sample_id="window", turns=turns, turn_source_ids=sources
        )
    )

    refs = result.ledger[("scope", "last")]
    assert [ref.batch_index for ref in refs] == [2, 3, 4, 5, 6, 7]
    assert all(not hasattr(ref, "messages") for ref in refs)
    with pytest.raises(Mem0AttributionError, match="Only request-window"):
        adapter.require_exact_source_provenance()


def test_raw_longmemeval_pairing_skips_empty_pair_without_shifting_turns():
    backend = _FakeBackend(
        add_responses=[
            {"results": [{"id": "early-memory"}]},
            {"results": [{"id": "late-memory"}]},
        ],
        runtime_identity={"certified": True, "stack": "fake-certified"},
    )
    adapter = Mem0Adapter(
        backend=backend,
        token_counter=len,
        user_scope_factory=lambda _: "raw-scope",
    )
    record = {
        "question_id": "raw-empty",
        "haystack_session_ids": ["late", "early"],
        "haystack_dates": ["2025/03/04 (Tue) 11:00", "2024/01/01 (Mon) 09:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "discard-with-empty"},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "late-u"},
                {"role": "assistant", "content": "late-a"},
            ],
            [
                {"role": "user", "content": " "},
                {"role": "assistant", "content": "early-a"},
            ],
        ],
    }

    result = adapter.ingest_longmemeval_record(record)

    assert [messages for messages, _kwargs in backend.add_calls] == [
        [
            {"role": "user", "content": " "},
            {"role": "assistant", "content": "early-a"},
        ],
        [
            {"role": "user", "content": "late-u"},
            {"role": "assistant", "content": "late-a"},
        ],
    ]
    assert all(
        set(kwargs) == {"user_id", "infer"}
        for _messages, kwargs in backend.add_calls
    )
    assert result.raw_pair_count == 3
    assert result.skipped_empty_pair_count == 1
    assert [ref.turn_start for ref in result.batches_added] == [0, 2]
    assert result.official_longmemeval_protocol is True
    # Direct backend injection is test-only and cannot certify owned storage.
    assert result.comparison_certified is False
    assert result.runtime_identity["certified"] is True


def test_official_rendering_uses_only_memory_text_and_returned_created_at():
    backend = _FakeBackend(
        add_responses=[{"results": [{"id": "id-new"}, {"id": "id-old"}]}],
        search_response={
            "results": [
                {
                    "id": "id-new",
                    "memory": "newer fact",
                    "created_at": "2024-02-02T18:00:00-08:00",
                    "metadata": {"source": "untrusted"},
                },
                {
                    "id": "id-old",
                    "memory": "older fact",
                    "created_at": "2024-01-01T09:00:00+00:00",
                },
            ]
        },
        runtime_identity={"certified": True},
    )
    adapter = Mem0Adapter(
        backend=backend,
        token_counter=len,
        user_scope_factory=lambda _: "scope",
    )
    ingest = adapter.ingest_longmemeval_record(
        {
            "question_id": "official",
            "haystack_session_ids": ["secret-session"],
            "haystack_dates": ["2024/01/01 (Mon) 09:00"],
            "haystack_sessions": [[
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ]],
        }
    )

    result = adapter.search(
        "q",
        user_scope=ingest.user_scope,
        max_prompt_tokens=10_000,
        prompt_renderer=_prompt,
    )

    assert [candidate.memory_id for candidate in result.packed] == ["id-new", "id-old"]
    assert result.context == (
        "--- Monday, January 01, 2024 ---\n- older fact\n"
        "--- Saturday, February 03, 2024 ---\n- newer fact"
    )
    assert all(value not in result.context for value in (
        "id-new", "id-old", "secret-session", "untrusted", MEM0_ATTRIBUTION_KIND
    ))
    assert "secret-session" in result.diagnostics[0].audit_rendered
    assert backend.search_calls[0][1]["explain"] is False
    assert result.official_search_protocol is True
    assert result.comparison_certified is False

    ablation = adapter.search(
        "q",
        user_scope=ingest.user_scope,
        max_prompt_tokens=10_000,
        prompt_renderer=_prompt,
        threshold=0.2,
    )
    assert ablation.official_search_protocol is False
    assert ablation.comparison_certified is False


def test_enriched_attribution_rendering_is_explicitly_noncertifying():
    backend = _FakeBackend(
        add_responses=[{"results": [{"id": "m1"}]}],
        search_response={
            "results": [{"id": "m1", "memory": "fact", "created_at": _CREATED}]
        },
        runtime_identity={"certified": True},
    )
    adapter = Mem0Adapter(
        backend=backend,
        token_counter=len,
        user_scope_factory=lambda _: "scope",
    )
    ingest = adapter.ingest_longmemeval_record(
        {
            "question_id": "ablation",
            "haystack_session_ids": ["source-a"],
            "haystack_dates": ["2024/01/01 (Mon) 09:00"],
            "haystack_sessions": [[
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ]],
        }
    )

    result = adapter.search(
        "q",
        user_scope=ingest.user_scope,
        max_prompt_tokens=10_000,
        prompt_renderer=_prompt,
        rendering_mode=MEM0_ENRICHED_RENDERING,
    )

    assert "source=source-a" in result.context
    assert result.rendering_mode == MEM0_ENRICHED_RENDERING
    assert result.certified_rendering is False
    assert result.comparison_certified is False


def test_scoped_ledger_isolates_identical_memory_ids():
    backend = _FakeBackend(
        add_responses=[
            {"results": [{"id": "same-id"}]},
            {"results": [{"id": "same-id"}]},
        ],
        search_response=[
            {"id": "same-id", "memory": "remembered", "created_at": _CREATED}
        ],
    )
    scopes = iter(["scope-a", "scope-b"])
    adapter = Mem0Adapter(
        backend=backend,
        token_counter=len,
        user_scope_factory=lambda _: next(scopes),
    )
    first = adapter.ingest(_single_batch_sample("one", source="source-a"))
    second = adapter.ingest(_single_batch_sample("two", source="source-b"))

    first_search = adapter.search(
        "q",
        user_scope=first.user_scope,
        max_prompt_tokens=10_000,
        prompt_renderer=_prompt,
    )
    second_search = adapter.search(
        "q",
        user_scope=second.user_scope,
        max_prompt_tokens=10_000,
        prompt_renderer=_prompt,
    )

    assert set(adapter.ledger) == {
        ("scope-a", "same-id"),
        ("scope-b", "same-id"),
    }
    assert set(first.ledger) == {("scope-a", "same-id")}
    assert set(second.ledger) == {("scope-b", "same-id")}
    assert first_search.raw_pool[0].request_window_attribution[0].source == "source-a"
    assert second_search.raw_pool[0].request_window_attribution[0].source == "source-b"


@pytest.mark.parametrize(
    "search_row",
    [
        {"memory": "missing id"},
        {"id": "never-added", "memory": "unknown id"},
    ],
)
def test_search_refuses_unattributed_rows(search_row):
    backend = _FakeBackend(
        add_responses=[{"results": [{"id": "known"}]}],
        search_response={"results": [search_row]},
    )
    adapter = Mem0Adapter(
        backend=backend,
        token_counter=len,
        user_scope_factory=lambda _: "scope",
    )
    adapter.ingest(_single_batch_sample())

    with pytest.raises(Mem0AttributionError):
        adapter.search(
            "q", max_prompt_tokens=10_000, prompt_renderer=_prompt
        )


def test_return_shape_variants_are_normalized_without_trusting_metadata():
    backend = _FakeBackend(
        add_responses=[{"memory_id": "m1"}],
        search_response=[
            {
                "memory_id": "m1",
                "text": "safe memory",
                "created_at": _CREATED,
                "metadata": {"source": "spoofed-source", "date": "spoofed-date"},
            }
        ],
    )
    adapter = Mem0Adapter(
        backend=backend,
        token_counter=len,
        user_scope_factory=lambda _: "scope",
    )
    adapter.ingest(_single_batch_sample(source="audited-source"))

    result = adapter.search(
        "q", max_prompt_tokens=10_000, prompt_renderer=_prompt
    )

    assert result.raw_pool[0].metadata["source"] == "spoofed-source"
    assert result.context == "--- Monday, January 01, 2024 ---\n- safe memory"
    assert "spoofed-source" not in result.context
    assert "audited-source" not in result.context
    assert f"attribution={MEM0_ATTRIBUTION_KIND}" in result.diagnostics[0].audit_rendered
    assert "source=audited-source" in result.diagnostics[0].audit_rendered
    assert result.rendering_mode == MEM0_CERTIFIED_RENDERING
    assert result.certified_rendering is True


@pytest.mark.parametrize(
    "response",
    [None, {"results": None}, {"results": ["not-a-row"]}],
)
def test_unsupported_response_shapes_fail_closed(response):
    backend = _FakeBackend(add_responses=[response])
    adapter = Mem0Adapter(
        backend=backend,
        token_counter=len,
        user_scope_factory=lambda _: "scope",
    )
    with pytest.raises(Mem0ProtocolError):
        adapter.ingest(_single_batch_sample())

    assert adapter.stats.add_attempted_calls == 1
    assert adapter.stats.add_completed_calls == 0
    assert adapter.stats.add_failed_calls == 1
    with pytest.raises(Mem0PoisonedError, match="ambiguous mutation"):
        adapter.ingest(_single_batch_sample("second"))


def test_add_exception_poisons_workload_but_cleanup_remains_safe():
    events = []
    backend = _FakeBackend(
        add_error=RuntimeError("ambiguous write"),
        events=events,
    )
    adapter = Mem0Adapter(
        backend=backend,
        token_counter=len,
        user_scope_factory=lambda _: "scope",
    )

    with pytest.raises(RuntimeError, match="ambiguous write"):
        adapter.ingest(_single_batch_sample())
    assert adapter.stats.add_calls == 1
    assert adapter.stats.add_attempted_calls == 1
    assert adapter.stats.add_completed_calls == 0
    assert adapter.stats.add_failed_calls == 1
    with pytest.raises(Mem0PoisonedError):
        adapter.search("q", max_prompt_tokens=100, prompt_renderer=_prompt)

    adapter.cleanup()
    assert events == [("delete_all", "scope"), ("backend_close", None)]


def test_release_scope_bounds_live_state_and_avoids_duplicate_cleanup():
    backend = _FakeBackend(
        add_responses=[
            {"results": [{"id": "m1"}]},
            {"results": [{"id": "m2"}]},
        ]
    )
    scopes = iter(["scope-1", "scope-2"])
    adapter = Mem0Adapter(
        backend=backend,
        token_counter=len,
        user_scope_factory=lambda _: next(scopes),
    )
    first = adapter.ingest(_single_batch_sample("one"))
    second = adapter.ingest(_single_batch_sample("two"))

    adapter.release_scope(first.user_scope)

    assert set(adapter.ledger) == {(second.user_scope, "m2")}
    assert adapter.stats.released_scopes == 1
    assert adapter.stats.unique_ledger_memories == 1
    with pytest.raises(ValueError, match="requires a user scope"):
        adapter.search(
            "q",
            user_scope=first.user_scope,
            max_prompt_tokens=100,
            prompt_renderer=_prompt,
        )

    adapter.cleanup()
    assert backend.delete_calls == [
        {"user_id": "scope-1"},
        {"user_id": "scope-2"},
    ]


def test_full_prompt_budget_includes_labels_separators_and_declared_overhead():
    backend = _FakeBackend(
        add_responses=[
            {"results": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]}
        ],
        search_response={
            "results": [
                {"id": "m1", "memory": "alpha", "created_at": _CREATED},
                {"id": "m2", "memory": "beta", "created_at": _CREATED},
                {"id": "m3", "memory": "gamma", "created_at": _CREATED},
            ]
        },
    )
    adapter = Mem0Adapter(
        backend=backend,
        token_counter=len,
        clock=_StepClock(0.5),
        user_scope_factory=lambda _: "scope",
    )
    adapter.ingest(_single_batch_sample())
    all_result = adapter.search(
        "where?",
        max_prompt_tokens=100_000,
        prompt_renderer=_prompt,
        prompt_token_overhead=7,
    )
    first_two = "--- Monday, January 01, 2024 ---\n- alpha\n- beta"
    exact_cap = len(_prompt("where?", first_two)) + 7

    result = adapter.search(
        "where?",
        max_prompt_tokens=exact_cap,
        prompt_renderer=_prompt,
        prompt_token_overhead=7,
    )

    assert [candidate.memory_id for candidate in result.packed] == ["m1", "m2"]
    assert result.context == first_two
    assert result.context_tokens == len(first_two)
    assert result.prompt == _prompt("where?", first_two)
    assert result.prompt_token_proxy == exact_cap
    assert result.max_prompt_token_proxy == exact_cap
    assert result.prompt_token_proxy_overhead == 7
    assert result.residual_prompt_token_proxy == 0
    assert result.prompt_token_proxy_budget_compliant is True
    assert result.token_counter_identity == "callable:builtins.len:unverified"
    assert result.token_counter_identity_verified is False
    assert result.prompt_tokens == exact_cap
    assert result.residual_prompt_tokens == 0
    assert result.prompt_budget_certified is True
    assert [item.reason for item in result.diagnostics] == [
        "selected",
        "selected",
        "prompt_token_budget",
    ]
    assert result.stats.search_raw_memory_tokens == 2 * len("alphabetagamma")
    assert result.stats.search_prompt_tokens == all_result.prompt_tokens + exact_cap
    assert (
        result.stats.search_prompt_token_proxy
        == all_result.prompt_token_proxy + exact_cap
    )
    assert result.stats.provider_prompt_tokens is None
    assert result.stats.provider_completion_tokens is None
    assert result.stats.provider_usage_status == MEM0_PROVIDER_USAGE_STATUS


def test_recognized_mem0_proxy_counter_binds_exact_vocabulary_identity():
    adapter = Mem0Adapter(
        backend=_FakeBackend(),
        token_counter=count_tokens,
        user_scope_factory=lambda _: "scope",
    )
    adapter.ingest(BenchmarkSample(sample_id="empty"))
    result = adapter.search(
        "q",
        max_prompt_tokens=100,
        prompt_renderer=_prompt,
        prompt_token_overhead=24,
    )

    identity = tokenizer_proxy_identity()
    assert result.token_counter_identity == (
        f"{identity['schema']}:{identity['encoding']}:"
        f"{identity['vocabulary_sha256']}"
    )
    assert result.token_counter_identity_verified is True
    assert result.prompt_token_proxy_budget_compliant is True


def test_context_cap_is_secondary_and_baseline_prompt_must_fit_before_search():
    backend = _FakeBackend(
        add_responses=[{"results": [{"id": "m1"}]}],
        search_response={
            "results": [{"id": "m1", "memory": "alpha", "created_at": _CREATED}]
        },
    )
    adapter = Mem0Adapter(
        backend=backend,
        token_counter=len,
        user_scope_factory=lambda _: "scope",
    )
    adapter.ingest(_single_batch_sample())

    result = adapter.search(
        "q",
        max_prompt_tokens=10_000,
        context_token_budget=0,
        prompt_renderer=_prompt,
    )
    assert result.packed == ()
    assert result.diagnostics[0].reason == "context_token_budget"

    calls_before = len(backend.search_calls)
    with pytest.raises(Mem0PromptBudgetError, match="without retrieved context"):
        adapter.search("q", max_prompt_tokens=1, prompt_renderer=_prompt)
    assert len(backend.search_calls) == calls_before


def test_default_threshold_is_point_one_and_all_thresholds_are_validated():
    backend = _FakeBackend()
    adapter = Mem0Adapter(
        backend=backend,
        token_counter=len,
        user_scope_factory=lambda _: "scope",
    )
    adapter.ingest(BenchmarkSample(sample_id="empty"))
    adapter.search("q", max_prompt_tokens=100, prompt_renderer=_prompt)
    assert backend.search_calls[0][1]["threshold"] == 0.1

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        adapter.search(
            "q", max_prompt_tokens=100, prompt_renderer=_prompt, threshold=1.1
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Mem0Adapter(backend=_FakeBackend(), token_counter=len, threshold=float("nan"))


def test_turn_source_ids_must_be_parallel_and_dates_must_be_parseable():
    adapter = Mem0Adapter(
        backend=_FakeBackend(),
        token_counter=len,
        user_scope_factory=lambda _: "scope-a",
    )
    with pytest.raises(Mem0ProtocolError, match="parallel"):
        adapter.ingest(
            BenchmarkSample(
                sample_id="bad-sources",
                turns=[("user", "u"), ("assistant", "a")],
                turn_source_ids=["s"],
            )
        )

    other = Mem0Adapter(
        backend=_FakeBackend(),
        token_counter=len,
        user_scope_factory=lambda _: "scope-b",
    )
    with pytest.raises(Mem0ProtocolError, match="chronology cannot be certified"):
        other.ingest(
            BenchmarkSample(
                sample_id="bad-date",
                turns=[
                    ("system", "[s took place at someday]"),
                    ("user", "u"),
                ],
                turn_source_ids=["s", "s"],
            )
        )


def test_cleanup_deletes_scopes_then_closes_resources_and_clears_ledger():
    events = []
    backend = _FakeBackend(
        add_responses=[
            {"results": [{"id": "m1"}]},
            {"results": [{"id": "m2"}]},
        ],
        events=events,
    )
    vector = _FakeVectorClient(events)
    scopes = iter(["scope-1", "scope-2"])
    adapter = Mem0Adapter(
        backend=backend,
        vector_client=vector,
        token_counter=len,
        user_scope_factory=lambda _: next(scopes),
    )
    adapter.ingest(_single_batch_sample("one"))
    adapter.ingest(_single_batch_sample("two"))

    adapter.cleanup()
    adapter.cleanup()

    assert events == [
        ("delete_all", "scope-1"),
        ("delete_all", "scope-2"),
        ("backend_close", None),
        ("vector_close", None),
    ]
    assert dict(adapter.ledger) == {}
    assert adapter.active_user_scope is None


def test_cleanup_aggregates_failures_without_skipping_later_resources():
    events = []
    adapter = Mem0Adapter(
        backend=_FakeBackend(
            events=events,
            delete_error=RuntimeError("delete failed"),
            close_error=RuntimeError("backend close failed"),
        ),
        vector_client=_FakeVectorClient(
            events, error=RuntimeError("vector close failed")
        ),
        token_counter=len,
        user_scope_factory=lambda _: "scope",
    )
    adapter.ingest(BenchmarkSample(sample_id="empty"))

    with pytest.raises(BaseExceptionGroup) as raised:
        adapter.cleanup()

    assert len(raised.value.exceptions) == 3
    assert events == [
        ("delete_all", "scope"),
        ("backend_close", None),
        ("vector_close", None),
    ]
    assert dict(adapter.ledger) == {}


def test_context_manager_preserves_workload_error_when_cleanup_also_fails():
    adapter = Mem0Adapter(
        backend=_FakeBackend(delete_error=RuntimeError("cleanup boom")),
        token_counter=len,
        user_scope_factory=lambda _: "scope",
    )

    with pytest.raises(ValueError, match="workload boom") as raised:
        with adapter:
            adapter.ingest(BenchmarkSample(sample_id="empty"))
            raise ValueError("workload boom")

    assert any("cleanup also failed" in note for note in raised.value.__notes__)


def test_injected_backend_factory_is_lazy():
    calls = []
    backend = _FakeBackend()

    def factory():
        calls.append("factory")
        return backend

    adapter = Mem0Adapter(
        backend_factory=factory,
        token_counter=len,
        user_scope_factory=lambda _: "scope",
    )
    assert calls == []
    adapter.ingest(BenchmarkSample(sample_id="s"))
    assert calls == ["factory"]


def test_real_factory_restores_telemetry_env_when_dependency_is_missing(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MEM0_TELEMETRY", "true")
    monkeypatch.setenv("MEM0_DIR", "original-mem0-dir")
    state = tmp_path / "state"

    def missing_version(distribution):
        assert distribution == "mem0ai"
        assert os.environ["MEM0_TELEMETRY"] == "false"
        raise importlib.metadata.PackageNotFoundError(distribution)

    factory = Mem0OSSBackendFactory(
        config=_config(state),
        llm_model_id="llm-1",
        embedder_model_id="embed-1",
        owned_state_dir=state,
        _version_reader=missing_version,
        _stack_preflight=_stack_identity,
        _module_importer=lambda _: pytest.fail("missing package must not import"),
    )

    with pytest.raises(Mem0DependencyError, match=r"mem0ai==2\.0\.18"):
        factory()
    assert os.environ["MEM0_TELEMETRY"] == "true"
    assert os.environ["MEM0_DIR"] == "original-mem0-dir"
    assert not state.exists()


def test_real_factory_rejects_preimported_enabled_telemetry(
    monkeypatch, tmp_path
):
    state = tmp_path / "state"
    monkeypatch.setitem(
        sys.modules,
        "mem0.memory.telemetry",
        SimpleNamespace(MEM0_TELEMETRY=True),
    )
    factory = Mem0OSSBackendFactory(
        config=_config(state),
        llm_model_id="llm-1",
        embedder_model_id="embed-1",
        owned_state_dir=state,
        _version_reader=lambda _: MEM0AI_PIN,
        _stack_preflight=_stack_identity,
        _module_importer=lambda _: pytest.fail("must fail before import"),
    )
    with pytest.raises(Mem0DependencyError, match="already imported"):
        factory()
    assert not state.exists()


def test_real_factory_rejects_preimported_foreign_mem0_state(
    monkeypatch, tmp_path
):
    state = tmp_path / "state"
    monkeypatch.setitem(
        sys.modules,
        "mem0.memory.setup",
        SimpleNamespace(mem0_dir=str(tmp_path / "foreign")),
    )
    factory = Mem0OSSBackendFactory(
        config=_config(state),
        llm_model_id="llm-1",
        embedder_model_id="embed-1",
        owned_state_dir=state,
        _version_reader=lambda _: MEM0AI_PIN,
        _stack_preflight=_stack_identity,
        _module_importer=lambda _: pytest.fail("must fail before import"),
    )
    with pytest.raises(Mem0DependencyError, match="different MEM0_DIR"):
        factory()
    assert not state.exists()


def test_real_factory_rejects_wrong_distribution_or_module_version(tmp_path):
    state_a = tmp_path / "state-a"
    factory_a = Mem0OSSBackendFactory(
        config=_config(state_a),
        llm_model_id="llm-1",
        embedder_model_id="embed-1",
        owned_state_dir=state_a,
        _version_reader=lambda _: "2.0.17",
        _stack_preflight=_stack_identity,
        _module_importer=lambda _: pytest.fail("wrong version must not import"),
    )
    with pytest.raises(Mem0DependencyError, match="expected mem0ai==2.0.18"):
        factory_a()

    state_b = tmp_path / "state-b"
    factory_b = Mem0OSSBackendFactory(
        config=_config(state_b),
        llm_model_id="llm-1",
        embedder_model_id="embed-1",
        owned_state_dir=state_b,
        _version_reader=lambda _: MEM0AI_PIN,
        _stack_preflight=_stack_identity,
        _module_importer=lambda _: SimpleNamespace(
            __version__="9.9.9", Memory=SimpleNamespace(from_config=lambda **_: None)
        ),
    )
    with pytest.raises(Mem0DependencyError, match="does not match"):
        factory_b()
    assert not state_b.exists()


def test_real_factory_passes_frozen_unique_config_and_owns_cleanup(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("MEM0_TELEMETRY", raising=False)
    state = tmp_path / "state"
    events = []
    captured = []

    class Client:
        def close(self):
            events.append("client_close")

    class Store:
        client = Client()
        _has_bm25_slot = True

        def delete_col(self):
            events.append("delete_col")

    class Backend:
        vector_store = Store()
        _entity_store = None

        def close(self):
            events.append("backend_close")

        def add(self, *args, **kwargs):
            return {"results": []}

        def search(self, *args, **kwargs):
            return {"results": []}

        def delete_all(self, *args, **kwargs):
            return None

    class FakeMemory:
        @classmethod
        def from_config(cls, **kwargs):
            assert os.environ["MEM0_TELEMETRY"] == "false"
            assert os.environ["MEM0_DIR"] == str(state.resolve())
            captured.append(kwargs)
            return Backend()

    factory = Mem0OSSBackendFactory(
        config=_config(state),
        llm_model_id="llm-1",
        embedder_model_id="embed-1",
        owned_state_dir=state,
        _version_reader=lambda _: MEM0AI_PIN,
        _stack_preflight=_stack_identity,
        _module_importer=lambda _: SimpleNamespace(
            __version__=MEM0AI_PIN, Memory=FakeMemory
        ),
    )
    wrapped = factory()

    assert state.exists()
    assert "MEM0_TELEMETRY" not in os.environ
    assert "MEM0_DIR" not in os.environ
    effective = captured[0]["config_dict"]
    assert effective["history_db_path"] == str(state / "history.db")
    assert effective["vector_store"]["config"]["collection_name"].startswith(
        "mem0-longmemeval-"
    )
    assert wrapped.collection_name == effective["vector_store"]["config"][
        "collection_name"
    ]
    assert len(wrapped.config_fingerprint) == 64

    adapter = Mem0Adapter(
        backend=wrapped,
        token_counter=count_tokens,
        user_scope_factory=lambda _: "owned-scope",
    )
    ingest = adapter.ingest_longmemeval_record(
        {
            "question_id": "owned",
            "haystack_session_ids": ["s"],
            "haystack_dates": ["2024/01/01 (Mon) 09:00"],
            "haystack_sessions": [[
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ]],
        }
    )
    search = adapter.search(
        "q", max_prompt_tokens=100, prompt_renderer=_prompt
    )
    assert ingest.comparison_certified is True
    assert search.comparison_certified is True

    adapter.cleanup()
    adapter.cleanup()

    assert events == ["delete_col", "backend_close", "client_close"]
    assert not state.exists()


def test_factory_identity_is_stable_redacted_and_independent_of_owned_root(tmp_path):
    class Store:
        _has_bm25_slot = True

        def delete_col(self):
            return None

    class Backend:
        vector_store = Store()
        _entity_store = None

        def close(self):
            return None

    class Memory:
        @classmethod
        def from_config(cls, **_kwargs):
            return Backend()

    wrapped = []
    for name in ("state-a", "state-b"):
        state = tmp_path / name
        config = _config(state)
        config["llm"]["config"]["api_key"] = "never-export-this"
        factory = Mem0OSSBackendFactory(
            config=config,
            llm_model_id="llm-1",
            embedder_model_id="embed-1",
            owned_state_dir=state,
            _version_reader=lambda _: MEM0AI_PIN,
            _module_importer=lambda _: SimpleNamespace(
                __version__=MEM0AI_PIN, Memory=Memory
            ),
            _stack_preflight=_stack_identity,
        )
        wrapped.append(factory())

    first, second = wrapped
    assert first.stable_config_fingerprint == second.stable_config_fingerprint
    assert first.effective_config_fingerprint != second.effective_config_fingerprint
    identity_text = repr(dict(first.runtime_identity))
    assert "never-export-this" not in identity_text
    assert str(first.state_root) not in identity_text
    assert first.runtime_identity["config"]["llm"]["config"]["api_key"] == (
        "<redacted>"
    )
    assert first.runtime_identity["certified"] is True

    first.close()
    second.close()


def test_factory_fails_closed_before_state_when_hybrid_stack_is_incomplete(tmp_path):
    state = tmp_path / "state"
    incomplete = Mem0StackIdentity(
        dependency_versions={"mem0ai": MEM0AI_PIN},
        bm25_model=MEM0_BM25_MODEL,
        spacy_model=MEM0_SPACY_MODEL,
        bm25_operational=False,
        entity_extraction_operational=True,
    )
    factory = Mem0OSSBackendFactory(
        config=_config(state),
        llm_model_id="llm-1",
        embedder_model_id="embed-1",
        owned_state_dir=state,
        _version_reader=lambda _: MEM0AI_PIN,
        _module_importer=lambda _: pytest.fail("preflight must fail before import"),
        _stack_preflight=lambda: incomplete,
    )

    with pytest.raises(Mem0DependencyError, match="hybrid/entity stack"):
        factory()
    assert not state.exists()


def test_real_factory_removes_owned_state_when_mem0_initialization_fails(tmp_path):
    state = tmp_path / "state"

    class FailingMemory:
        @classmethod
        def from_config(cls, **kwargs):
            assert Path(kwargs["config_dict"]["history_db_path"]).parent == state
            raise RuntimeError("initialization failed")

    factory = Mem0OSSBackendFactory(
        config=_config(state),
        llm_model_id="llm-1",
        embedder_model_id="embed-1",
        owned_state_dir=state,
        _version_reader=lambda _: MEM0AI_PIN,
        _stack_preflight=_stack_identity,
        _module_importer=lambda _: SimpleNamespace(
            __version__=MEM0AI_PIN, Memory=FailingMemory
        ),
    )
    with pytest.raises(RuntimeError, match="initialization failed"):
        factory()
    assert not state.exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config["llm"]["config"].pop("model"), "llm.config.model"),
        (lambda config: config["embedder"].pop("provider"), "embedder.provider"),
        (lambda config: config.pop("history_db_path"), "history_db_path"),
        (lambda config: config.pop("custom_instructions"), "custom_instructions"),
        (lambda config: config.pop("reranker"), "reranker"),
        (lambda config: config.update(version="v1.0"), "API version"),
        (
            lambda config: config["vector_store"]["config"].pop(
                "embedding_model_dims"
            ),
            "embedding_model_dims",
        ),
        (
            lambda config: config["vector_store"].update(provider="chroma"),
            "qdrant",
        ),
        (
            lambda config: config["vector_store"]["config"].update(on_disk=False),
            "on_disk=true",
        ),
    ],
)
def test_real_factory_requires_complete_frozen_config(
    tmp_path, mutate, message
):
    state = tmp_path / "state"
    config = _config(state)
    mutate(config)
    with pytest.raises(Mem0ConfigurationError, match=message):
        Mem0OSSBackendFactory(
            config=config,
            llm_model_id="llm-1",
            embedder_model_id="embed-1",
            owned_state_dir=state,
        )


def test_real_factory_rejects_shared_or_ambiguous_state(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(Mem0ConfigurationError, match="must not already exist"):
        Mem0OSSBackendFactory(
            config=_config(existing),
            llm_model_id="llm-1",
            embedder_model_id="embed-1",
            owned_state_dir=existing,
        )

    state = tmp_path / "state"
    outside_history = _config(state)
    outside_history["history_db_path"] = str(tmp_path / "shared.db")
    with pytest.raises(Mem0ConfigurationError, match="inside owned_state_dir"):
        Mem0OSSBackendFactory(
            config=outside_history,
            llm_model_id="llm-1",
            embedder_model_id="embed-1",
            owned_state_dir=state,
        )

    ambiguous = _config(state)
    ambiguous["vector_store"]["config"]["url"] = "http://also-remote"
    with pytest.raises(Mem0ConfigurationError, match="only a local Qdrant path"):
        Mem0OSSBackendFactory(
            config=ambiguous,
            llm_model_id="llm-1",
            embedder_model_id="embed-1",
            owned_state_dir=state,
        )


def test_real_factory_rejects_remote_target_and_model_mismatch(
    tmp_path,
):
    state = tmp_path / "remote-state"
    with pytest.raises(Mem0ConfigurationError, match="only a local Qdrant path"):
        Mem0OSSBackendFactory(
            config=_config(state, remote=True),
            llm_model_id="llm-1",
            embedder_model_id="embed-1",
            owned_state_dir=state,
        )

    mismatch_state = tmp_path / "mismatch"
    with pytest.raises(Mem0ConfigurationError, match="llm_model_id does not match"):
        Mem0OSSBackendFactory(
            config=_config(mismatch_state),
            llm_model_id="other",
            embedder_model_id="embed-1",
            owned_state_dir=mismatch_state,
        )
