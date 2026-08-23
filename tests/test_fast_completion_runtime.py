from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain.discourse import canonical_json, identity_sha256
from memory_condense.eval import fast_completion_runtime as fast_runtime


def _messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Answer from the supplied evidence."},
        {"role": "user", "content": question},
    ]


class _FakeCompletions:
    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        fail: BaseException | None = None,
        require_overlap: bool = False,
        reported_prompt_tokens: object = 41,
        delay_s: float = 0.02,
        finish_reason: str = "stop",
        fail_request_numbers: frozenset[int] | None = None,
        start_barrier: threading.Barrier | None = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.fail = fail
        self.require_overlap = require_overlap
        self.reported_prompt_tokens = reported_prompt_tokens
        self.delay_s = delay_s
        self.finish_reason = finish_reason
        self.fail_request_numbers = fail_request_numbers or frozenset()
        self.start_barrier = start_barrier
        self.requests: list[dict[str, object]] = []
        self.active = 0
        self.max_active = 0
        self._counter = 0
        self._lock = threading.Lock()
        self._overlap = threading.Event()

    def create(self, **request):
        messages = request["messages"]
        messages_sha = identity_sha256(messages)
        matching_requests: list[dict[str, object]] = []
        for path in self.checkpoint_dir.glob("*.request.json"):
            matching_requests.append(
                json.loads(path.read_text(encoding="utf-8"))
            )
        matching_requests = [
            row for row in matching_requests if row["messages_sha256"] == messages_sha
        ]
        assert len(matching_requests) == 1
        call_key = str(matching_requests[0]["call_key_sha256"])
        assert not (self.checkpoint_dir / f"{call_key}.response.json").exists()

        with self._lock:
            self.requests.append(request)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self._counter += 1
            response_number = self._counter
            if self.active >= 2:
                self._overlap.set()
        try:
            if self.start_barrier is not None:
                self.start_barrier.wait(timeout=2)
            if response_number in self.fail_request_numbers:
                raise RuntimeError("scripted provider failure")
            if self.require_overlap and not self._overlap.wait(timeout=2):
                raise AssertionError("provider calls did not overlap")
            if self.fail is not None:
                raise self.fail
            time.sleep(self.delay_s)
            return SimpleNamespace(
                id=f"response-{response_number}",
                model="fake-provider-model-v1",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=f"answer-{messages_sha[:8]}"
                        ),
                        finish_reason=self.finish_reason,
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=self.reported_prompt_tokens,
                    completion_tokens=3,
                    total_tokens=44,
                ),
            )
        finally:
            with self._lock:
                self.active -= 1


class _FakeClient:
    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        fail: BaseException | None = None,
        require_overlap: bool = False,
        reported_prompt_tokens: object = 41,
        delay_s: float = 0.02,
        finish_reason: str = "stop",
        fail_request_numbers: frozenset[int] | None = None,
        start_barrier: threading.Barrier | None = None,
    ) -> None:
        self.max_retries = 0
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(
                checkpoint_dir=checkpoint_dir,
                fail=fail,
                require_overlap=require_overlap,
                reported_prompt_tokens=reported_prompt_tokens,
                delay_s=delay_s,
                finish_reason=finish_reason,
                fail_request_numbers=fail_request_numbers,
                start_barrier=start_barrier,
            )
        )
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _runtime(
    checkpoint_dir: Path,
    client: _FakeClient | None,
    *,
    population=None,
    max_concurrency: int = 2,
    max_prompt_tokens: int = 256,
    **kwargs,
) -> fast_runtime.FastCompletionRuntime:
    return fast_runtime.FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=population
        or [_messages("one"), _messages("two"), _messages("one")],
        model="codex_sdk/fake-model",
        client=client,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=32,
        max_concurrency=max_concurrency,
        benchmark_provenance={"arm": "fixed-cav", "artifact_sha256": "a" * 64},
        **kwargs,
    )


def _reseal(path: Path, **changes: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    body = dict(payload)
    body.pop("journal_sha256")
    payload["journal_sha256"] = identity_sha256(body)
    path.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))


def test_unique_prompts_overlap_with_bounded_concurrency_and_deduplicate(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    population = [
        _messages("one"),
        _messages("two"),
        _messages("one"),
        _messages("three"),
        _messages("four"),
    ]
    client = _FakeClient(checkpoint_dir, require_overlap=True)
    runtime = _runtime(
        checkpoint_dir,
        client,
        population=population,
        max_concurrency=2,
        request_options={"temperature": 0.0},
    )

    batch = runtime.run()

    assert len(batch.logical_completions) == 5
    assert batch.logical_completions[0] == batch.logical_completions[2]
    assert len(batch.unique_records) == 4
    assert batch.usage.logical_calls == 5
    assert batch.usage.unique_calls == 4
    assert batch.usage.deduplicated_logical_calls == 1
    assert batch.usage.physical_calls == 4
    assert batch.usage.checkpoint_hits == 0
    assert batch.usage.recorded_reported_prompt_tokens == 4 * 41
    assert batch.usage.recorded_reported_completion_tokens == 4 * 3
    assert batch.usage.recorded_reported_total_tokens == 4 * 44
    assert batch.usage.reported_total_tokens_complete is True
    assert client.chat.completions.max_active == 2
    assert len(client.chat.completions.requests) == 4
    assert all(
        request["temperature"] == 0.0
        and request["model"] == "codex_sdk/fake-model"
        and request["max_tokens"] == 32
        for request in client.chat.completions.requests
    )
    assert len(list(checkpoint_dir.glob("*.request.json"))) == 4
    assert len(list(checkpoint_dir.glob("*.response.json"))) == 4
    assert batch.runtime_identity_sha256 == identity_sha256(
        batch.provenance.model_dump()
    )
    assert [row.messages_sha256 for row in batch.prompt_population.ordered_rows] == [
        identity_sha256(messages) for messages in population
    ]


def test_verified_responses_resume_without_provider_calls(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "calls"
    first_client = _FakeClient(checkpoint_dir, require_overlap=True)
    first = _runtime(checkpoint_dir, first_client)
    expected = first.run()
    first.close()

    replay = _runtime(checkpoint_dir, None)
    observed = replay.run()

    assert observed.logical_completions == expected.logical_completions
    assert observed.usage.physical_calls == 0
    assert observed.usage.checkpoint_hits == 2
    assert all(record.checkpoint_hit for record in observed.unique_records)
    replay.close()


def test_request_without_response_is_fail_closed_and_never_retried(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    population = [_messages("only")]
    client = _FakeClient(checkpoint_dir, fail=RuntimeError("gateway disconnected"))
    first = _runtime(checkpoint_dir, client, population=population)

    with pytest.raises(RuntimeError, match="gateway disconnected"):
        first.run()

    assert len(client.chat.completions.requests) == 1
    assert len(list(checkpoint_dir.glob("*.request.json"))) == 1
    assert list(checkpoint_dir.glob("*.response.json")) == []
    second_client = _FakeClient(checkpoint_dir)
    with pytest.raises(RuntimeError, match="refusing an unsafe retry"):
        _runtime(checkpoint_dir, second_client, population=population)
    assert second_client.chat.completions.requests == []


def test_nonterminal_finish_reason_is_not_published_as_an_answer(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    population = [_messages("one")]
    client = _FakeClient(checkpoint_dir, finish_reason="length")
    runtime = _runtime(checkpoint_dir, client, population=population)

    with pytest.raises(RuntimeError, match="not terminally complete"):
        runtime.run()

    assert len(list(checkpoint_dir.glob("*.request.json"))) == 1
    assert len(list(checkpoint_dir.glob("*.response.json"))) == 0


def test_late_over_cap_prompt_rejects_before_path_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "must-not-exist"
    client = _FakeClient(checkpoint_dir)
    counts: list[str] = []

    def count(messages) -> int:
        value = str(messages[-1]["content"])
        counts.append(value)
        return 33 if value == "fits" else 65

    monkeypatch.setattr(fast_runtime, "count_chat_prompt_token_proxy", count)

    with pytest.raises(ValueError, match=r"hard token cap.*ordinal 1: 65 > 64"):
        _runtime(
            checkpoint_dir,
            client,
            population=[_messages("fits"), _messages("late over cap")],
            max_prompt_tokens=64,
        )

    assert counts == ["fits", "late over cap"]
    assert not checkpoint_dir.exists()
    assert client.chat.completions.requests == []


@pytest.mark.parametrize("reported_prompt_tokens", [65, -1, True, 2.5])
def test_invalid_provider_prompt_usage_leaves_only_request_reservation(
    tmp_path: Path,
    reported_prompt_tokens: object,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    population = [_messages("only")]
    client = _FakeClient(
        checkpoint_dir,
        reported_prompt_tokens=reported_prompt_tokens,
    )
    runtime = _runtime(
        checkpoint_dir,
        client,
        population=population,
        max_prompt_tokens=64,
    )

    with pytest.raises(RuntimeError, match="violate the hard token cap"):
        runtime.run()

    assert len(client.chat.completions.requests) == 1
    assert len(list(checkpoint_dir.glob("*.request.json"))) == 1
    assert list(checkpoint_dir.glob("*.response.json")) == []


def test_zero_provider_usage_is_normalized_to_unavailable(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "calls"
    client = _FakeClient(checkpoint_dir, reported_prompt_tokens=0)
    runtime = _runtime(
        checkpoint_dir,
        client,
        population=[_messages("only")],
        max_prompt_tokens=64,
    )

    batch = runtime.run()

    assert batch.unique_records[0].reported_prompt_tokens is None
    assert batch.usage.reported_prompt_tokens_complete is False
    assert batch.usage.recorded_reported_prompt_tokens == 0


@pytest.mark.parametrize("reported_prompt_tokens", [0, 257])
def test_canonical_replay_rejects_provider_prompt_usage_outside_cap(
    tmp_path: Path,
    reported_prompt_tokens: int,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    population = [_messages("only")]
    first = _runtime(
        checkpoint_dir,
        _FakeClient(checkpoint_dir),
        population=population,
        max_prompt_tokens=256,
    )
    first.run()
    first.close()
    response_path = next(checkpoint_dir.glob("*.response.json"))
    _reseal(response_path, reported_prompt_tokens=reported_prompt_tokens)

    with pytest.raises(ValueError, match="violates the hard token cap"):
        _runtime(checkpoint_dir, None, population=population)


@pytest.mark.parametrize("invalid_count", [0, -1, True, 1.5])
def test_invalid_local_prompt_count_rejects_before_path_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_count: object,
) -> None:
    checkpoint_dir = tmp_path / "must-not-exist"
    client = _FakeClient(checkpoint_dir)
    monkeypatch.setattr(
        fast_runtime,
        "count_chat_prompt_token_proxy",
        lambda _messages: invalid_count,
    )

    with pytest.raises(ValueError, match="must return a positive integer"):
        _runtime(
            checkpoint_dir,
            client,
            population=[_messages("only")],
            max_prompt_tokens=64,
        )

    assert not checkpoint_dir.exists()
    assert client.chat.completions.requests == []


def test_nested_identity_inputs_are_detached_and_recursively_immutable(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    options = {"extra_body": {"mode": "original", "layers": [1, 2]}}
    provenance = {
        "arm": "fixed-cav",
        "source": {"artifact_sha256": "a" * 64},
    }
    client = _FakeClient(checkpoint_dir)
    runtime = fast_runtime.FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=[_messages("only")],
        model="codex_sdk/fake-model",
        client=client,
        max_prompt_tokens=256,
        max_new_tokens=32,
        request_options=options,
        benchmark_provenance=provenance,
    )
    runtime_sha = runtime.runtime_identity_sha256

    options["extra_body"]["mode"] = "source-mutated"
    provenance["source"]["artifact_sha256"] = "b" * 64
    with pytest.raises(TypeError):
        runtime.provenance.request_options["extra_body"]["mode"] = "mutated"
    with pytest.raises(TypeError):
        runtime.provenance.benchmark_provenance["source"][
            "artifact_sha256"
        ] = "c" * 64

    batch = runtime.run()

    assert runtime.runtime_identity_sha256 == runtime_sha
    assert batch.provenance.model_dump()["request_options"] == {
        "extra_body": {"mode": "original", "layers": [1, 2]}
    }
    assert batch.provenance.model_dump()["benchmark_provenance"] == {
        "arm": "fixed-cav",
        "source": {"artifact_sha256": "a" * 64},
    }
    assert client.chat.completions.requests[0]["extra_body"] == {
        "mode": "original",
        "layers": [1, 2],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("benchmark_provenance", {"nested": {"hidden_states": [[1.0]]}}),
        ("request_options", {"extra_body": {"past_key_values": [1]}}),
    ],
)
def test_transformer_state_metadata_is_rejected_before_path_mutation(
    tmp_path: Path,
    field: str,
    value: dict[str, object],
) -> None:
    checkpoint_dir = tmp_path / "must-not-exist"
    kwargs = {field: value}

    with pytest.raises(ValueError, match="must not persist transformer state"):
        fast_runtime.FastCompletionRuntime(
            checkpoint_dir=checkpoint_dir,
            prompt_population=[_messages("only")],
            model="codex_sdk/fake-model",
            client=_FakeClient(checkpoint_dir),
            max_prompt_tokens=256,
            max_new_tokens=32,
            **kwargs,
        )

    assert not checkpoint_dir.exists()


def test_concurrent_run_is_rejected_without_expanding_provider_concurrency(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    client = _FakeClient(
        checkpoint_dir,
        require_overlap=True,
        delay_s=0.2,
    )
    runtime = _runtime(checkpoint_dir, client, max_concurrency=2)
    batches: list[fast_runtime.FastCompletionBatch] = []
    failures: list[BaseException] = []

    def first_run() -> None:
        try:
            batches.append(runtime.run())
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            failures.append(exc)

    worker = threading.Thread(target=first_run)
    worker.start()
    assert client.chat.completions._overlap.wait(timeout=2)

    with pytest.raises(RuntimeError, match="already running"):
        runtime.run()

    worker.join(timeout=3)
    assert not worker.is_alive()
    assert failures == []
    assert len(batches) == 1
    assert len(client.chat.completions.requests) == 2
    assert client.chat.completions.max_active == 2


def test_two_runtimes_cannot_duplicate_an_inflight_reserved_call(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    first_client = _FakeClient(
        checkpoint_dir,
        require_overlap=True,
        delay_s=0.2,
    )
    second_client = _FakeClient(checkpoint_dir)
    first = _runtime(checkpoint_dir, first_client, max_concurrency=2)
    second = _runtime(checkpoint_dir, second_client, max_concurrency=2)
    failures: list[BaseException] = []

    def first_run() -> None:
        try:
            first.run()
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            failures.append(exc)

    worker = threading.Thread(target=first_run)
    worker.start()
    assert first_client.chat.completions._overlap.wait(timeout=2)

    with pytest.raises(RuntimeError, match="refusing an unsafe retry"):
        second.run()

    worker.join(timeout=3)
    assert not worker.is_alive()
    assert failures == []
    assert second_client.chat.completions.requests == []
    replay = second.run()
    assert replay.usage.physical_calls == 0
    assert replay.usage.checkpoint_hits == 2


def test_many_logical_prompts_stay_within_unique_call_and_worker_bounds(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    population = [_messages(f"question-{index % 12}") for index in range(36)]
    client = _FakeClient(checkpoint_dir, delay_s=0.005)
    runtime = _runtime(
        checkpoint_dir,
        client,
        population=population,
        max_concurrency=4,
    )

    batch = runtime.run()

    assert batch.usage.logical_calls == 36
    assert batch.usage.unique_calls == 12
    assert batch.usage.physical_calls == 12
    assert batch.usage.deduplicated_logical_calls == 24
    assert 2 <= client.chat.completions.max_active <= 4
    assert len(list(checkpoint_dir.glob("*.request.json"))) == 12
    assert len(list(checkpoint_dir.glob("*.response.json"))) == 12


def test_first_provider_failure_does_not_submit_beyond_concurrency_window(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    workers = 4
    population = [_messages(f"question-{index}") for index in range(20)]
    client = _FakeClient(
        checkpoint_dir,
        delay_s=0.1,
        fail_request_numbers=frozenset({1}),
        start_barrier=threading.Barrier(workers),
    )
    runtime = _runtime(
        checkpoint_dir,
        client,
        population=population,
        max_concurrency=workers,
    )

    with pytest.raises(RuntimeError, match="scripted provider failure"):
        runtime.run()

    assert len(client.chat.completions.requests) == workers
    assert client.chat.completions.max_active == workers
    assert len(list(checkpoint_dir.glob("*.request.json"))) == workers
    assert len(list(checkpoint_dir.glob("*.response.json"))) == workers - 1


def test_atomic_request_publish_failure_cannot_reach_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    population = [_messages("only")]
    client = _FakeClient(checkpoint_dir)
    runtime = _runtime(checkpoint_dir, client, population=population)
    real_replace = fast_runtime.os.replace

    def fail_request(source, destination) -> None:
        if str(destination).endswith(".request.json"):
            raise OSError("simulated atomic request publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(fast_runtime.os, "replace", fail_request)

    with pytest.raises(OSError, match="request publication failure"):
        runtime.run()

    assert client.chat.completions.requests == []
    assert list(checkpoint_dir.glob("*.request.json")) == []
    assert list(checkpoint_dir.glob("*.response.json")) == []
    assert list(checkpoint_dir.glob("*.tmp")) == []


def test_atomic_response_publish_failure_leaves_terminal_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    population = [_messages("only")]
    client = _FakeClient(checkpoint_dir)
    runtime = _runtime(checkpoint_dir, client, population=population)
    real_replace = fast_runtime.os.replace

    def fail_response(source, destination) -> None:
        if str(destination).endswith(".response.json"):
            raise OSError("simulated atomic response publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(fast_runtime.os, "replace", fail_response)

    with pytest.raises(OSError, match="response publication failure"):
        runtime.run()

    assert len(client.chat.completions.requests) == 1
    assert len(list(checkpoint_dir.glob("*.request.json"))) == 1
    assert list(checkpoint_dir.glob("*.response.json")) == []
    assert list(checkpoint_dir.glob("*.tmp")) == []
    with pytest.raises(RuntimeError, match="refusing an unsafe retry"):
        _runtime(checkpoint_dir, None, population=population)


def test_tampered_completed_response_is_not_resumed(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "calls"
    population = [_messages("only")]
    first = _runtime(
        checkpoint_dir, _FakeClient(checkpoint_dir), population=population
    )
    first.run()
    first.close()
    response_path = next(checkpoint_dir.glob("*.response.json"))
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    payload["completion"] = "tampered"
    response_path.write_text(
        canonical_json(payload) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"not canonical|receipt does not verify"):
        _runtime(
            checkpoint_dir, _FakeClient(checkpoint_dir), population=population
        )


def test_canonical_but_changed_request_is_not_resumed(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "calls"
    population = [_messages("only")]
    first = _runtime(
        checkpoint_dir, _FakeClient(checkpoint_dir), population=population
    )
    first.run()
    first.close()
    request_path = next(checkpoint_dir.glob("*.request.json"))
    _reseal(request_path, max_new_tokens=31)

    with pytest.raises(ValueError, match="request provenance changed"):
        _runtime(checkpoint_dir, None, population=population)


def test_prompt_hash_collision_fails_before_path_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "must-not-exist"
    original = fast_runtime.identity_sha256

    def collide_messages(value):
        return "d" * 64 if isinstance(value, list) else original(value)

    monkeypatch.setattr(fast_runtime, "identity_sha256", collide_messages)

    with pytest.raises(RuntimeError, match="prompt SHA-256 collision"):
        _runtime(
            checkpoint_dir,
            _FakeClient(checkpoint_dir),
            population=[_messages("one"), _messages("two")],
        )

    assert not checkpoint_dir.exists()


def test_journals_expose_provenance_without_prompt_or_transformer_state(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    secret_prompt = "PRIVATE-PROMPT-MARKER"
    runtime = _runtime(
        checkpoint_dir,
        _FakeClient(checkpoint_dir),
        population=[_messages(secret_prompt)],
    )

    batch = runtime.run()
    serialized = "".join(
        path.read_text(encoding="utf-8")
        for path in checkpoint_dir.glob("*.json")
    )

    assert secret_prompt not in serialized
    assert "input_ids" not in serialized
    assert "hidden_states" not in serialized
    assert "past_key_values" not in serialized
    assert batch.unique_records[0].messages_sha256 == identity_sha256(
        _messages(secret_prompt)
    )
    assert batch.unique_records[0].request_journal_sha256
    assert batch.unique_records[0].response_journal_sha256
    assert runtime.request_token_state_receipt() == {
        "contract": fast_runtime.FAST_COMPLETION_TOKEN_STATE_CONTRACT,
        "persisted_transformer_token_state": False,
        "retained_transformer_token_state_bytes": 0,
        "journal_payload_kinds": (
            "prompt_hashes_response_text_scalar_usage_and_provenance"
        ),
        "external_provider_persistence_certified": False,
    }


def test_retries_are_zero_and_runtime_owned_options_are_rejected_before_path(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "must-not-exist"
    client = _FakeClient(checkpoint_dir)

    with pytest.raises(ValueError, match="retries must be zero"):
        _runtime(checkpoint_dir, client, retries=1)
    assert not checkpoint_dir.exists()

    with pytest.raises(ValueError, match="runtime-owned key: messages"):
        _runtime(checkpoint_dir, client, request_options={"messages": []})
    assert not checkpoint_dir.exists()

    client.max_retries = 2
    with pytest.raises(ValueError, match="must expose max_retries=0"):
        _runtime(checkpoint_dir, client)
    assert not checkpoint_dir.exists()

    client.max_retries = None
    with pytest.raises(ValueError, match="must expose max_retries=0"):
        _runtime(checkpoint_dir, client)
    assert not checkpoint_dir.exists()
