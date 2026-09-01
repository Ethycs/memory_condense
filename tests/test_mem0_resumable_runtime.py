from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.mem0_eval import production_binding as binding
from tools.mem0_eval.policy import (
    MEM0_EXTRACTION_GATEWAY_URL,
    MEM0_EXTRACTION_MODEL,
    MEM0_EXTRACTION_PROVIDER,
    MEM0_EXTRACTION_REVISION,
    MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
)
from tools.mem0_eval.resumable import (
    AppendOnlyResumeJournal,
    ResumePlan,
    append_commit,
    append_intent,
    append_send_attempt,
    canonical_json_sha256,
    deterministic_user_scope,
    rehydration_material,
)
from tools.mem0_eval.resumable_runner import (
    DurableHTTPJournalBoundary,
    _adapter_ledger_projection,
    _cumulative_receipt,
    _http_request_sha256,
    _primer_start,
    _rehydrate_terminal_adapter,
    _source_ref_dict,
    _stats_dict,
    prepared_batch_sha256,
)
from tools.mem0_eval.resumable_runtime import (
    RESUMABLE_TRANSPORT_CLOSURE_FORMAT,
    Mem0WriteActivityMeter,
    ZeroCallExtractionTransport,
    suspend_resumable_adapter,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _plan() -> ResumePlan:
    authorization = _sha("authorization")
    return ResumePlan(
        authorization_sha256=authorization,
        mem0_policy_sha256=_sha("policy"),
        source_validation_policy_sha256=_sha("source-policy"),
        source_implementation_sha256=_sha("source-code"),
        source_environment_lock_sha256=_sha("source-lock"),
        mem0_tool_implementation_sha256=_sha("tool"),
        mem0_environment_lock_sha256=_sha("tool-lock"),
        sample_offset=0,
        sample_sha256=_sha("sample"),
        raw_history_bundle_sha256=_sha("history"),
        ordered_batch_sha256s=(_sha("batch"),),
        authorized_add_operations=1,
        authorized_extraction_calls=1,
        authorized_search_operations=1,
        user_scope=deterministic_user_scope(authorization),
    )


def _length_tokens(text: str) -> int:
    return len(text)


def _exact_segment_transport(*, authorized: int, attempted: int) -> dict:
    return {
        "kind": "local_transport_send_cap",
        "role": "extraction",
        "authorized": authorized,
        "attempted": attempted,
        "completed": attempted,
        "failed": 0,
        "rejected": 0,
        "retries_authorized": 0,
        "provider_usage_status": "provider_reported_exact",
        "provider_usage_records": attempted,
        "provider_input_tokens": attempted * 2,
        "provider_output_tokens": attempted,
        "provider_total_tokens": attempted * 3,
        "provider_latency_s": float(attempted),
        "production_eligible": True,
        "provider": MEM0_EXTRACTION_PROVIDER,
        "model": MEM0_EXTRACTION_MODEL,
        "revision": MEM0_EXTRACTION_REVISION,
        "route_identity_sha256": MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
        "request_identity_sha256": canonical_json_sha256(
            {
                "format": "memory-condense-mem0-extraction-request-v1",
                "route_identity_sha256": MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
                "response_format": {"type": "json_object"},
                "max_completion_tokens": 2_000,
                "sampling_parameters": "omitted",
                "sdk_retries": 0,
                "http_transport_retries": 0,
                "follow_redirects": False,
                "trust_env": False,
                "timeout_seconds": 600.0,
                "connect_timeout_seconds": 30.0,
                "cap_boundary": "httpx.BaseTransport.handle_request",
            }
        ),
        "gateway_url": MEM0_EXTRACTION_GATEWAY_URL,
        "max_completion_tokens": 2_000,
        "sampling_parameters_omitted": True,
        "sdk_retries": 0,
        "http_transport_retries": 0,
        "follow_redirects": False,
        "trust_env": False,
        "cap_boundary": "httpx.BaseTransport.handle_request",
        "external_http_attempts_certified": True,
        "external_provider_persistence_certified": False,
    }


class _QuarterStepClock:
    def __init__(self) -> None:
        self.value = -0.25

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


def test_hard_cap_runs_durable_callback_before_increment_and_send() -> None:
    cap = binding.HardTransportAttemptCap(role="extraction", authorized=1)
    events: list[tuple[str, int]] = []

    def before() -> None:
        events.append(("before", cap.receipt()["attempted"]))

    def send() -> str:
        events.append(("send", cap.receipt()["attempted"]))
        return "ok"

    assert cap.call(send, _before_increment=before) == "ok"
    assert events == [("before", 0), ("send", 1)]
    cap.assert_closed()


def test_hard_cap_callback_failure_prevents_attempt_and_inner_send() -> None:
    cap = binding.HardTransportAttemptCap(role="extraction", authorized=1)
    sent: list[bool] = []

    def fail() -> None:
        raise RuntimeError("journal fsync failed")

    with pytest.raises(RuntimeError, match="journal fsync"):
        cap.call(lambda: sent.append(True), _before_increment=fail)
    assert sent == []
    assert cap.receipt()["attempted"] == 0


def test_durable_http_boundary_publishes_one_send_attempt(tmp_path: Path) -> None:
    plan = _plan()
    journal = AppendOnlyResumeJournal(tmp_path / "resume.jsonl", plan)
    state = journal.create(
        owned_state_path="state/working",
        snapshot_root_path="state/snapshots",
    )
    state = append_intent(
        journal, state, ordinal=0, session_sha256=_sha("session")
    )
    boundary = DurableHTTPJournalBoundary(journal)
    boundary.arm(state, ordinal=0)
    request = SimpleNamespace(
        method="POST", url="https://central-dev.zt:9000/v1/chat", content=b"{}"
    )
    boundary.before_http_send(request)
    marked = boundary.consume_after_response()
    assert marked.pending_send_attempt is not None
    assert marked.pending_send_attempt["request_sha256"] == _http_request_sha256(
        request
    )
    with pytest.raises(Exception, match="unarmed"):
        boundary.before_http_send(request)


def test_zero_call_transport_is_provider_free_and_fail_closed() -> None:
    transport = ZeroCallExtractionTransport()
    receipt = transport.transport_receipt()
    assert receipt["authorized"] == receipt["attempted"] == 0
    transport.assert_call_budget_closed()
    with pytest.raises(binding.TransportAttemptLimitExceeded):
        transport.generate_response([])
    assert transport.transport_receipt()["rejected"] == 1
    with pytest.raises(binding.ProductionBindingError):
        transport.assert_call_budget_closed()


def test_write_activity_meter_covers_embedding_and_all_persistence_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Embedder:
        def embed(self, text: str) -> list[float]:
            return [float(len(text))]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text))] for text in texts]

    class Mutations:
        def insert(self, *args, **kwargs):
            return (args, kwargs)

        def update(self, *args, **kwargs):
            return (args, kwargs)

        def delete(self, *args, **kwargs):
            return (args, kwargs)

    class History:
        def save_messages(self, *args, **kwargs):
            return (args, kwargs)

        def add_history(self, *args, **kwargs):
            return (args, kwargs)

        def batch_add_history(self, *args, **kwargs):
            return (args, kwargs)

    embedder = Embedder()
    stores = (Mutations(), Mutations())
    history = History()
    memory = SimpleNamespace(embedding_model=embedder, db=history)
    adapter = SimpleNamespace(_backend=SimpleNamespace(backend=memory))
    monkeypatch.setattr(
        binding, "_materialize_exact_qdrant_stores", lambda _memory: stores
    )
    original_embed = embedder.embed
    original_batch = embedder.embed_batch
    meter = Mem0WriteActivityMeter(token_counter=len)
    restore = meter.install(adapter)
    assert embedder.embed("abc") == [3.0]
    assert embedder.embed_batch((value for value in ("d", "ef"))) == [
        [1.0],
        [2.0],
    ]
    for store in stores:
        store.insert("x")
        store.update("x")
        store.delete("x")
    history.save_messages([])
    history.add_history({})
    history.batch_add_history([])
    restore()
    meter.assert_closed()
    receipt = meter.receipt()
    assert receipt["embedding_attempted"] == receipt["embedding_completed"] == 3
    assert receipt["embedding_input_token_proxy"] == 6
    assert receipt["storage_attempted"] == receipt["storage_completed"] == 9
    assert receipt["embedding_failed"] == receipt["storage_failed"] == 0
    assert embedder.embed == original_embed
    assert embedder.embed_batch == original_batch


def test_write_activity_meter_fails_closed_on_partial_write_and_wrapper_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Embedder:
        def embed(self, text: str) -> list[float]:
            return [1.0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

    class Mutations:
        def insert(self, *_args, **_kwargs):
            raise RuntimeError("partial write")

        def update(self, *_args, **_kwargs):
            return None

        def delete(self, *_args, **_kwargs):
            return None

    class History:
        save_messages = Mutations.update
        add_history = Mutations.update
        batch_add_history = Mutations.update

    store = Mutations()
    embedder = Embedder()
    memory = SimpleNamespace(embedding_model=embedder, db=History())
    adapter = SimpleNamespace(_backend=SimpleNamespace(backend=memory))
    monkeypatch.setattr(
        binding, "_materialize_exact_qdrant_stores", lambda _memory: (store,)
    )
    meter = Mem0WriteActivityMeter()
    meter.install(adapter)
    with pytest.raises(RuntimeError, match="partial write"):
        store.insert("x")
    meter.restore()
    with pytest.raises(binding.ProductionBindingError, match="did not close"):
        meter.assert_closed()

    second = Mem0WriteActivityMeter()
    second.install(adapter)
    embedder.embed = lambda _text: [9.0]
    with pytest.raises(binding.ProductionBindingError, match="wrapper changed"):
        second.restore()


def test_cumulative_receipt_seeds_prefix_without_hiding_segment_counts() -> None:
    logical = _cumulative_receipt(
        {
            "authorized": 2,
            "attempted": 1,
            "completed": 1,
            "failed": 0,
            "rejected": 0,
            "infer_true_adds_started": 1,
            "infer_true_adds_exactly_one_call": 1,
        },
        prefix=256,
        full_authorized=2_548,
        logical=True,
    )
    assert logical["attempted"] == logical["completed"] == 257
    assert logical["infer_true_adds_started"] == 257
    assert logical["authorized"] == 2_548
    assert logical["segment_authorized"] == 2


def test_primer_is_shortest_suffix_covering_ten_messages() -> None:
    batches = [
        SimpleNamespace(ref=SimpleNamespace(turn_count=count))
        for count in (2, 1, 2, 2, 1, 2, 1, 2)
    ]
    start = _primer_start(batches, len(batches))
    assert sum(row.ref.turn_count for row in batches[start:]) >= 10
    assert sum(row.ref.turn_count for row in batches[start + 1 :]) < 10


@pytest.mark.parametrize("shared_client", [False, True])
def test_suspend_closes_both_stores_without_deleting_owned_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_client: bool
) -> None:
    from tools.mem0_eval.source_compat import _OwnedMem0Backend

    state = tmp_path / "owned"
    state.mkdir()
    (state / ".memory-condense-owned-state").write_text("a" * 32)
    (state / "history.sqlite").write_bytes(b"history")

    class Collection:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class Local:
        def __init__(self, name: str) -> None:
            self.closed = False
            self.collections = {name: Collection()}

    class Client:
        def __init__(self, local: Local) -> None:
            self._client = local

        def close(self) -> None:
            self._client.closed = True

    class Store:
        def __init__(self, client: Client) -> None:
            self.client = client
            self.delete_calls = 0

        def delete_col(self) -> None:
            self.delete_calls += 1

    local_a = Local("memory")
    local_b = local_a if shared_client else Local("entity")
    client_a = Client(local_a)
    client_b = client_a if shared_client else Client(local_b)
    stores = (Store(client_a), Store(client_b))

    class Memory:
        def __init__(self) -> None:
            self.db = SimpleNamespace(connection=object())

        def close(self) -> None:
            self.db.connection = None
            self.db = None

    memory = Memory()
    owned = _OwnedMem0Backend(
        backend=memory,
        state_root=state,
        ownership_token="a" * 32,
        collection_name="collection-aaaaaaaaaaaa",
        stable_config_fingerprint=_sha("stable"),
        effective_config_fingerprint=_sha("effective"),
        runtime_identity={},
    )
    transport = ZeroCallExtractionTransport()
    transport_receipt = transport.transport_receipt()
    closure_body = {
        "format": RESUMABLE_TRANSPORT_CLOSURE_FORMAT,
        "segment_authorized_calls": 0,
        "transport_closed": True,
        "budget_closed_exactly": True,
        "provider_usage_complete": True,
        "sdk_retries": 0,
        "http_transport_retries": 0,
        "transport_receipt": transport_receipt,
        "transport_receipt_sha256": canonical_json_sha256(transport_receipt),
    }
    factory = SimpleNamespace(
        _scope="longmemeval:resumable:test",
        transport_closure_receipt=lambda: {
            **closure_body,
            "receipt_sha256": canonical_json_sha256(closure_body),
        },
    )
    adapter = SimpleNamespace(
        _backend=owned,
        _closed=False,
        _production_extraction_transport=transport,
        _resumable_factory=factory,
        active_user_scope=factory._scope,
    )
    monkeypatch.setattr(
        binding, "_materialize_exact_qdrant_stores", lambda _memory: stores
    )
    monkeypatch.setattr(
        "tools.mem0_eval.resumable_runtime._namespace_persisted_memory_count",
        lambda _memory, _scope: 0,
    )
    receipt = suspend_resumable_adapter(adapter)
    assert receipt["owned_state_retained"] is True
    assert state.is_dir()
    assert all(store.delete_calls == 0 for store in stores)
    assert all(collection.closed for local in {local_a, local_b} for collection in local.collections.values())
    assert local_a.closed is True and local_b.closed is True
    assert receipt["qdrant_local_collections_closed"] == (1 if shared_client else 2)
    assert receipt["qdrant_clients_closed"] == (1 if shared_client else 2)
    assert transport._closed is True


@pytest.mark.parametrize("prefix", range(1, 9))
def test_window_ten_rehydration_matches_uninterrupted_adapter_at_every_prefix(
    tmp_path: Path, prefix: int
) -> None:
    from memory_condense.eval.mem0_adapter import (
        Mem0AdapterStats,
        Mem0LongMemEvalAdapter,
        SourceRef,
        _PreparedBatch,
        _PreparedCorpus,
    )

    counts = (2, 1, 2, 2, 1, 2, 1, 2)
    ids = ("m0", "shared", "m2", "shared", "m4", "m5", "shared", "m7")
    scope = "longmemeval:resumable:equivalence"
    batches = []
    for ordinal, count in enumerate(counts[:prefix]):
        roles = ("user", "assistant")[:count]
        ref = SourceRef(
            sample_id="sample-0",
            source=f"session-{ordinal}",
            session=f"Session {ordinal}",
            session_index=ordinal,
            original_session_index=ordinal,
            batch_index=ordinal,
            date=f"2026/08/{ordinal + 1:02d}",
            turn_start=sum(counts[:ordinal]),
            turn_count=count,
            roles=roles,
        )
        messages = tuple(
            (role, f"message-{ordinal}-{turn}")
            for turn, role in enumerate(roles)
        )
        batches.append(_PreparedBatch(ref=ref, messages=messages))

    class Backend:
        def __init__(self, *, preloaded: tuple[str, ...] = ()) -> None:
            self.calls = 0
            self.rows: dict[str, dict[str, object]] = {}
            for memory_id in preloaded:
                self.rows.setdefault(
                    memory_id,
                    {
                        "id": memory_id,
                        "memory": f"fact-{memory_id}",
                        "score": 0.9,
                        "created_at": "2026-08-01T00:00:00Z",
                    },
                )

        def add(self, _messages: object, **_kwargs: object) -> dict[str, object]:
            memory_id = ids[self.calls]
            self.calls += 1
            self.rows.setdefault(
                memory_id,
                {
                    "id": memory_id,
                    "memory": f"fact-{memory_id}",
                    "score": 0.9,
                    "created_at": "2026-08-01T00:00:00Z",
                },
            )
            return {"results": [{"id": memory_id}]}

        def search(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"results": list(self.rows.values())}

    corpus = _PreparedCorpus(
        sample_id="sample-0",
        batches=tuple(batches),
        raw_pair_count=len(batches),
        skipped_empty_pair_count=0,
        official_longmemeval_protocol=True,
    )
    reference = Mem0LongMemEvalAdapter(
        backend=Backend(),
        token_counter=_length_tokens,
        clock=_QuarterStepClock(),
        user_scope_factory=lambda _sample: scope,
    )
    reference._ingest_prepared(corpus)

    authorization_sha = _sha(f"equivalence-{prefix}")
    plan = ResumePlan(
        authorization_sha256=authorization_sha,
        mem0_policy_sha256=_sha("policy"),
        source_validation_policy_sha256=_sha("source-policy"),
        source_implementation_sha256=_sha("source-code"),
        source_environment_lock_sha256=_sha("source-lock"),
        mem0_tool_implementation_sha256=_sha("tool"),
        mem0_environment_lock_sha256=_sha("tool-lock"),
        sample_offset=0,
        sample_sha256=_sha("sample"),
        raw_history_bundle_sha256=_sha("history"),
        ordered_batch_sha256s=tuple(
            prepared_batch_sha256(batch) for batch in batches
        ),
        authorized_add_operations=prefix,
        authorized_extraction_calls=prefix,
        authorized_search_operations=1,
        user_scope=scope,
    )
    journal = AppendOnlyResumeJournal(tmp_path / "resume.jsonl", plan)
    state = journal.create(
        owned_state_path="state/working",
        snapshot_root_path="state/snapshots",
    )
    cumulative_tokens = 0
    seen_ids: set[str] = set()
    for ordinal, batch in enumerate(batches):
        state = append_intent(
            journal,
            state,
            ordinal=ordinal,
            session_sha256=_sha(f"session-{ordinal}"),
        )
        state = append_send_attempt(
            journal, state, request_sha256=_sha(f"request-{ordinal}")
        )
        prior = rehydration_material(state)
        prior_unique: list[dict[str, object]] = []
        for ref in prior["request_window_deque"]:
            if ref not in prior_unique:
                prior_unique.append(ref)
        source = _source_ref_dict(batch.ref)
        window = list(prior_unique)
        if source not in window:
            window.append(source)
        raw_tokens = len("\n".join(f"{role}: {text}" for role, text in batch.messages))
        cumulative_tokens += raw_tokens
        seen_ids.add(ids[ordinal])
        stats = Mem0AdapterStats(
            add_calls=ordinal + 1,
            add_attempted_calls=ordinal + 1,
            add_completed_calls=ordinal + 1,
            add_latency_s=0.25 * (ordinal + 1),
            add_raw_message_tokens=cumulative_tokens,
            add_returned_memories=ordinal + 1,
            unique_ledger_memories=len(seen_ids),
            token_counter_identity=reference.stats.token_counter_identity,
            token_counter_identity_verified=(
                reference.stats.token_counter_identity_verified
            ),
        )
        cumulative = ordinal + 1
        logical = {
            "attempted": cumulative,
            "completed": cumulative,
            "failed": 0,
            "rejected": 0,
            "infer_true_adds_started": cumulative,
            "infer_true_adds_exactly_one_call": cumulative,
        }
        segment_transport = _exact_segment_transport(
            authorized=prefix, attempted=cumulative
        )
        transport = _cumulative_receipt(
            segment_transport,
            prefix=0,
            full_authorized=prefix,
            logical=False,
        )
        state = append_commit(
            journal,
            state,
            response_sha256=_sha(f"response-{ordinal}"),
            returned_memory_ids=(ids[ordinal],),
            source_ref=source,
            request_window_refs=window,
            adapter_stats=_stats_dict(stats),
            logical_meter_receipt=logical,
            transport_receipt=transport,
            add_latency_s=0.25,
            raw_message_tokens=raw_tokens,
            scope_protocol=True,
        )

    resumed = Mem0LongMemEvalAdapter(
        backend=Backend(preloaded=ids[:prefix]),
        token_counter=_length_tokens,
        clock=_QuarterStepClock(),
        user_scope_factory=lambda _sample: scope,
    )
    _rehydrate_terminal_adapter(
        adapter=resumed, state=state, batches=tuple(batches)
    )
    assert _stats_dict(resumed.stats) == _stats_dict(reference.stats)
    assert _adapter_ledger_projection(resumed, state) == _adapter_ledger_projection(
        reference, state
    )

    search_kwargs = {
        "max_prompt_tokens": 1_000_000,
        "prompt_renderer": lambda query, context: f"{query}\n{context}",
        "user_scope": scope,
    }
    uninterrupted_search = reference.search("what?", **search_kwargs)
    resumed_search = resumed.search("what?", **search_kwargs)
    assert uninterrupted_search.context == resumed_search.context
    assert [row.memory_id for row in uninterrupted_search.raw_pool] == [
        row.memory_id for row in resumed_search.raw_pool
    ]
    assert [
        [_source_ref_dict(ref) for ref in row.request_window_attribution]
        for row in uninterrupted_search.raw_pool
    ] == [
        [_source_ref_dict(ref) for ref in row.request_window_attribution]
        for row in resumed_search.raw_pool
    ]
    assert _stats_dict(uninterrupted_search.stats) == _stats_dict(
        resumed_search.stats
    )
