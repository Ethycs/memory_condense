from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from memory_condense.eval import (
    recall_guarded_cumulative_provider_synthesis_runtime as runtime_module,
)
from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.recall_guarded_cumulative_provider_synthesis_runtime import (
    CENTRAL_DEV_GATEWAY_URL,
    DEFAULT_CALLER_MODEL,
    RecallGuardedCumulativeProviderSynthesisRuntime,
)


class _FakeModel:
    pass


class _FakeScorer:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.model_id = "Qwen/Qwen3-0.6B"
        self.model_revision = "c1899de289a04d12100db370d81485cdf75e47ca"
        self.checkpoint_sha256 = "a" * 64
        self.device = "cuda:0"
        self.dtype_name = "float16"
        self.max_candidates = 128
        self.requested_batch_size = 8
        self.batch_size = 8
        self.query_tokens = 192
        self.candidate_tokens = 256
        self.max_prompt_tokens = 768
        self.max_workspace_tokens = 8192
        self._choices = (" A", " B")
        self._choice_ids = ((32,), (33,))
        self.strict = True


class _FakeIdentity:
    def model_dump(self) -> dict[str, object]:
        return {
            "format": "memory-condense-recall-guarded-synthesis-runtime-v1",
            "model_id": "Qwen/Qwen3-0.6B",
            "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "checkpoint_sha256": "a" * 64,
            "runtime": "transformers.Qwen3ForCausalLM",
            "device": "cuda:0",
            "dtype": "float16",
            "max_position_embeddings": 40960,
            "default_max_new_tokens": 2048,
            "generation_do_sample": False,
            "generation_thinking": False,
            "generation_kv_cache": True,
            "scoring_kv_cache": False,
        }


class _FakeLocalRuntime:
    def __init__(self) -> None:
        self.identity = _FakeIdentity()
        self._scorer = _FakeScorer()
        self.last_score_report = SimpleNamespace(inspected_candidates=2)
        self.usage = SimpleNamespace(
            completion_calls=0,
            score_calls=1,
            score_forward_passes=1,
        )
        self.score_requests: list[tuple[object, object, object]] = []
        self.close_calls = 0

    def score_candidates(
        self,
        query,
        candidates,
        *,
        source_timestamps=None,
    ):
        self.score_requests.append((query, candidates, source_timestamps))
        return {"E1": SimpleNamespace(answerability=0.75)}

    def close(self) -> None:
        self.close_calls += 1


class _FakeCompletions:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _FakeClient:
    def __init__(self, *responses) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _response(
    text: str = "  structured result  ",
    *,
    usage=SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=3,
        total_tokens=15,
    ),
):
    return SimpleNamespace(
        id="chatcmpl-test",
        model="codex_sdk/gpt-5.6-terra-2026-08-01",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )


def _runtime(*responses, **kwargs):
    local = _FakeLocalRuntime()
    client = _FakeClient(*(responses or (_response(),)))
    runtime = RecallGuardedCumulativeProviderSynthesisRuntime(
        api_key="test-secret-that-must-not-be-retained",
        local_runtime=local,
        client=client,
        **kwargs,
    )
    return runtime, local, client


def test_default_codex_request_uses_gateway_model_and_omits_temperature() -> None:
    runtime, _local, client = _runtime(_response(), _response("second"))
    messages = [
        {"role": "system", "content": "Use only cited evidence."},
        {"role": "user", "content": "Synthesize S1 through S3."},
    ]

    assert runtime.complete(messages, max_new_tokens=321) == "structured result"
    assert client.chat.completions.requests == [
        {
            "model": "codex_sdk/gpt-5.6-terra",
            "messages": messages,
            "max_tokens": 321,
        }
    ]
    assert runtime.identity.gateway_url == CENTRAL_DEV_GATEWAY_URL
    assert runtime.identity.caller_model == DEFAULT_CALLER_MODEL
    assert runtime.identity.gateway_model == "codex_sdk/gpt-5.6-terra"
    assert runtime.identity.temperature is None
    assert runtime.identity.retries == 0

    assert runtime.complete(messages) == "second"
    assert runtime.completion_calls == 2
    assert runtime.last_completion_report.cumulative_completion_calls == 2
    usage = runtime.usage
    assert usage["completion_calls"] == 2
    assert usage["completion_reported_input_tokens"] == 24
    assert usage["completion_reported_output_tokens"] == 6
    assert usage["completion_reported_total_tokens"] == 30
    assert usage["completion_input_token_proxy"] == (
        2 * count_chat_prompt_token_proxy(messages)
    )
    assert usage["completion_output_token_proxy"] == (
        count_tokens("structured result") + count_tokens("second")
    )
    assert usage["completion_elapsed_s"] >= 0.0
    runtime.close()


def test_completion_report_captures_canonical_hashes_usage_and_provenance() -> None:
    runtime, _local, _client = _runtime(_response())
    messages = [
        {"role": "system", "content": "Use only cited evidence."},
        {"role": "user", "content": "Synthesize S1 through S3."},
    ]

    output = runtime.complete(messages)
    report = runtime.last_completion_report

    assert output == "structured result"
    assert report is not None
    assert report.gateway_url == CENTRAL_DEV_GATEWAY_URL
    assert report.caller_model == DEFAULT_CALLER_MODEL
    assert report.gateway_model == "codex_sdk/gpt-5.6-terra"
    assert report.messages_sha256 == identity_sha256(messages)
    assert report.completion_sha256 == quote_sha256(output)
    assert report.response_id == "chatcmpl-test"
    assert report.response_model == "codex_sdk/gpt-5.6-terra-2026-08-01"
    assert report.finish_reason == "stop"
    assert report.max_new_tokens == 2048
    assert report.reported_usage_available is True
    assert report.reported_input_tokens == 12
    assert report.reported_output_tokens == 3
    assert report.reported_total_tokens == 15
    assert report.reported_input_tokens_available is True
    assert report.reported_output_tokens_available is True
    assert report.reported_total_tokens_available is True
    assert report.input_token_proxy == count_chat_prompt_token_proxy(messages)
    assert report.output_token_proxy == count_tokens(output)
    assert report.elapsed_s >= 0.0
    assert report.retries == 0
    assert report.cumulative_completion_calls == 1

    identity = runtime.identity.model_dump()
    local = identity["local_scorer"]
    assert local["runtime_identity"] == _FakeIdentity().model_dump()
    assert local["score_provider"] == {
        "model_id": "Qwen/Qwen3-0.6B",
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "checkpoint_sha256": "a" * 64,
        "runtime": f"{_FakeModel.__module__}.{_FakeModel.__name__}",
        "device": "cuda:0",
        "dtype": "float16",
        "max_candidates": 128,
        "requested_batch_size": 8,
        "effective_batch_size": 8,
        "query_tokens": 192,
        "candidate_tokens": 256,
        "max_prompt_tokens": 768,
        "max_workspace_tokens": 8192,
        "choices": [" A", " B"],
        "choice_token_ids": [[32], [33]],
        "single_token_labels": True,
        "strict": True,
        "generation": False,
        "kv_cache": False,
    }
    assert len(local["identity_sha256"]) == 64
    assert "test-secret-that-must-not-be-retained" not in repr(identity)
    assert "test-secret-that-must-not-be-retained" not in repr(
        report.model_dump()
    )
    runtime.close()


def test_missing_provider_usage_is_explicitly_zero_and_unavailable() -> None:
    runtime, _local, _client = _runtime(_response(usage=None))

    runtime.complete([{"role": "user", "content": "question"}])
    report = runtime.last_completion_report

    assert report.reported_usage_available is False
    assert report.reported_input_tokens == 0
    assert report.reported_output_tokens == 0
    assert report.reported_total_tokens == 0
    assert report.reported_input_tokens_available is False
    assert report.reported_output_tokens_available is False
    assert report.reported_total_tokens_available is False
    assert report.input_token_proxy > 0
    assert report.output_token_proxy > 0
    runtime.close()


def test_gateway_zero_usage_is_explicitly_unavailable() -> None:
    zero_usage = SimpleNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )
    runtime, _local, _client = _runtime(_response(usage=zero_usage))

    runtime.complete([{"role": "user", "content": "question"}])
    report = runtime.last_completion_report

    assert report.reported_usage_available is False
    assert report.reported_input_tokens == 0
    assert report.reported_output_tokens == 0
    assert report.reported_total_tokens == 0
    assert report.reported_input_tokens_available is False
    assert report.reported_output_tokens_available is False
    assert report.reported_total_tokens_available is False
    runtime.close()


def test_non_codex_model_strips_only_one_prefix_and_sends_temperature() -> None:
    runtime, _local, client = _runtime(
        _response(),
        caller_model="openai/openai/example-model",
    )

    runtime.complete([{"role": "user", "content": "question"}])

    assert runtime.identity.gateway_model == "openai/example-model"
    assert client.chat.completions.requests[0]["model"] == (
        "openai/example-model"
    )
    assert client.chat.completions.requests[0]["temperature"] == 0.0
    runtime.close()


def test_score_candidates_delegates_to_the_only_local_runtime() -> None:
    runtime, local, client = _runtime(_response())
    candidates = {"E1": "direct evidence"}
    timestamps = {"E1": "2026-08-21T00:00:00Z"}

    scores = runtime.score_candidates(
        "Which source answers it?",
        candidates,
        source_timestamps=timestamps,
    )

    assert scores["E1"].answerability == 0.75
    assert local.score_requests == [
        ("Which source answers it?", candidates, timestamps)
    ]
    assert runtime.last_score_report is local.last_score_report
    assert runtime.usage["completion_calls"] == 0
    assert runtime.usage["score_calls"] == 1
    assert runtime.usage["score_forward_passes"] == 1
    assert client.chat.completions.requests == []
    runtime.close()


def test_close_closes_provider_and_local_once_and_prevents_reuse() -> None:
    runtime, local, client = _runtime(_response())

    runtime.close()
    runtime.close()

    assert client.close_calls == 1
    assert local.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        runtime.complete([{"role": "user", "content": "question"}])
    with pytest.raises(RuntimeError, match="closed"):
        runtime.score_candidates("question", {"E1": "evidence"})


def test_constructor_requires_explicit_nonempty_key() -> None:
    local = _FakeLocalRuntime()
    client = _FakeClient(_response())

    with pytest.raises(ValueError, match="api_key must be supplied explicitly"):
        RecallGuardedCumulativeProviderSynthesisRuntime(
            api_key="   ",
            local_runtime=local,
            client=client,
        )

    assert local.close_calls == 0
    assert client.close_calls == 0


def test_constructor_can_build_one_local_runtime_and_gateway_client(
    monkeypatch,
    tmp_path,
) -> None:
    constructed: dict[str, object] = {}
    local = _FakeLocalRuntime()
    client = _FakeClient(_response())

    def make_local(model_dir, **kwargs):
        constructed["model_dir"] = model_dir
        constructed["local_kwargs"] = kwargs
        return local

    def make_client(api_key):
        constructed["api_key"] = api_key
        return client

    monkeypatch.setattr(
        runtime_module,
        "RecallGuardedCumulativeSynthesisRuntime",
        make_local,
    )
    monkeypatch.setattr(runtime_module, "_new_gateway_client", make_client)

    runtime = RecallGuardedCumulativeProviderSynthesisRuntime(
        tmp_path / "qwen",
        api_key="ephemeral-secret",
        gpu_memory="7GiB",
        cpu_memory="25GiB",
    )

    assert constructed == {
        "model_dir": tmp_path / "qwen",
        "local_kwargs": {
            "max_new_tokens": 2048,
            "gpu_memory": "7GiB",
            "cpu_memory": "25GiB",
        },
        "api_key": "ephemeral-secret",
    }
    assert not hasattr(runtime, "api_key")
    assert "ephemeral-secret" not in repr(runtime.identity.model_dump())
    runtime.close()
    assert client.close_calls == 1
    assert local.close_calls == 1


def test_durable_journal_replays_invalid_output_without_another_call(
    tmp_path,
) -> None:
    checkpoint_dir = tmp_path / "provider-calls"
    messages = [
        {"role": "system", "content": "Return structured JSON."},
        {"role": "user", "content": "Synthesize the evidence."},
    ]
    first_local = _FakeLocalRuntime()
    first_client = _FakeClient(_response("not valid JSON"))
    first = RecallGuardedCumulativeProviderSynthesisRuntime(
        api_key="never-write-this-secret",
        local_runtime=first_local,
        client=first_client,
        checkpoint_dir=checkpoint_dir,
        campaign_binding={"retrieval_sha256": "b" * 64, "question_count": 10},
        authorized_completion_calls=1,
    )

    assert first.complete(messages) == "not valid JSON"
    assert len(first_client.chat.completions.requests) == 1
    first_report = first.last_completion_report
    assert first_report.cache_hit is False
    assert first_report.physical_call is True
    assert first.usage["completion_logical_calls"] == 1
    assert first.usage["completion_unique_calls"] == 1
    assert first.usage["completion_physical_calls"] == 1
    assert first.usage["completion_checkpoint_hits"] == 0
    request_paths = list(checkpoint_dir.glob("*.request.json"))
    response_paths = list(checkpoint_dir.glob("*.response.json"))
    assert len(request_paths) == len(response_paths) == 1
    assert list(checkpoint_dir.glob("*.tmp")) == []
    serialized = request_paths[0].read_text() + response_paths[0].read_text()
    assert "never-write-this-secret" not in serialized

    request_payload = json.loads(request_paths[0].read_text(encoding="utf-8"))
    key_payload = request_payload["call_key_payload"]
    call_key = request_payload["call_key_sha256"]
    assert call_key == identity_sha256(key_payload)
    assert request_paths[0].name == f"{call_key}.request.json"
    assert key_payload == {
        "messages_sha256": identity_sha256(messages),
        "runtime_identity_sha256": identity_sha256(
            first.identity.model_dump()
        ),
        "max_new_tokens": 2048,
        "campaign_binding_sha256": identity_sha256(
            {"retrieval_sha256": "b" * 64, "question_count": 10}
        ),
    }
    first.close()

    resumed_local = _FakeLocalRuntime()
    resumed_client = _FakeClient(_response("must not be used"))
    resumed = RecallGuardedCumulativeProviderSynthesisRuntime(
        api_key="a-different-ephemeral-key",
        local_runtime=resumed_local,
        client=resumed_client,
        checkpoint_dir=checkpoint_dir,
        campaign_binding={"question_count": 10, "retrieval_sha256": "b" * 64},
        authorized_completion_calls=1,
    )

    assert resumed.complete(messages) == "not valid JSON"
    assert resumed_client.chat.completions.requests == []
    replay_report = resumed.last_completion_report
    assert replay_report.cache_hit is True
    assert replay_report.physical_call is False
    assert replay_report.call_key_sha256 == first_report.call_key_sha256
    assert replay_report.completion_sha256 == first_report.completion_sha256
    assert replay_report.response_id == first_report.response_id
    assert replay_report.cumulative_logical_completion_calls == 1
    assert replay_report.cumulative_unique_completion_calls == 1
    assert replay_report.cumulative_physical_completion_calls == 0
    assert replay_report.cumulative_checkpoint_hits == 1
    resumed_usage = resumed.usage
    # The historical journaled response remains the one authoritative unique
    # cost record; this resumed process made no physical gateway call.
    assert resumed_usage["completion_calls"] == 1
    assert resumed_usage["completion_logical_calls"] == 1
    assert resumed_usage["completion_unique_calls"] == 1
    assert resumed_usage["completion_physical_calls"] == 0
    assert resumed_usage["completion_checkpoint_hits"] == 1
    assert resumed_usage["completion_reported_input_tokens"] == 12
    assert resumed_usage["completion_reported_output_tokens"] == 3
    assert resumed_usage["completion_reported_total_tokens"] == 15
    resumed.close()


def test_unique_call_budget_allows_hits_and_blocks_a_new_key_before_request(
    tmp_path,
) -> None:
    client = _FakeClient(_response("first"), _response("forbidden"))
    runtime = RecallGuardedCumulativeProviderSynthesisRuntime(
        api_key="ephemeral-secret",
        local_runtime=_FakeLocalRuntime(),
        client=client,
        checkpoint_dir=tmp_path / "calls",
        campaign_binding={"campaign": "sealed-test"},
        authorized_completion_calls=1,
    )
    first_messages = [{"role": " user ", "content": "same question"}]

    assert runtime.complete(first_messages, max_new_tokens=17) == "first"
    assert runtime.complete(
        [{"role": "user", "content": "same question"}],
        max_new_tokens=17,
    ) == "first"
    with pytest.raises(RuntimeError, match="budget exhausted"):
        runtime.complete(
            [{"role": "user", "content": "different question"}],
            max_new_tokens=17,
        )

    assert len(client.chat.completions.requests) == 1
    assert runtime.usage["completion_calls"] == 2
    assert runtime.usage["completion_logical_calls"] == 2
    assert runtime.usage["completion_unique_calls"] == 1
    assert runtime.usage["completion_physical_calls"] == 1
    assert runtime.usage["completion_checkpoint_hits"] == 1
    runtime.close()


def test_budget_is_enforced_against_verified_existing_journal(
    tmp_path,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    messages = [{"role": "user", "content": "first question"}]
    first, _local, _client = _runtime(
        _response("first"),
        checkpoint_dir=checkpoint_dir,
        authorized_completion_calls=1,
    )
    first.complete(messages)
    first.close()

    resumed_client = _FakeClient(_response("forbidden"))
    resumed = RecallGuardedCumulativeProviderSynthesisRuntime(
        api_key="ephemeral-secret",
        local_runtime=_FakeLocalRuntime(),
        client=resumed_client,
        checkpoint_dir=checkpoint_dir,
        authorized_completion_calls=1,
    )
    assert resumed.complete(messages) == "first"
    with pytest.raises(RuntimeError, match="budget exhausted"):
        resumed.complete([{"role": "user", "content": "second question"}])
    assert resumed_client.chat.completions.requests == []
    resumed.close()


def test_rehashed_response_tampering_fails_closed_on_construction(tmp_path) -> None:
    checkpoint_dir = tmp_path / "calls"
    first, _local, _client = _runtime(
        _response("original"),
        checkpoint_dir=checkpoint_dir,
    )
    first.complete([{"role": "user", "content": "question"}])
    first.close()
    response_path = next(checkpoint_dir.glob("*.response.json"))
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    payload["completion"] = "tampered but re-sealed"
    body = dict(payload)
    body.pop("journal_sha256")
    payload["journal_sha256"] = identity_sha256(body)
    response_path.write_bytes(
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    )
    rejected_local = _FakeLocalRuntime()

    with pytest.raises(ValueError, match="report binding changed"):
        RecallGuardedCumulativeProviderSynthesisRuntime(
            api_key="ephemeral-secret",
            local_runtime=rejected_local,
            client=_FakeClient(_response()),
            checkpoint_dir=checkpoint_dir,
        )

    assert rejected_local.close_calls == 1


def test_failed_physical_call_leaves_reservation_and_resume_fails_closed(
    tmp_path,
) -> None:
    checkpoint_dir = tmp_path / "calls"
    client = _FakeClient(RuntimeError("gateway disconnected"))
    first = RecallGuardedCumulativeProviderSynthesisRuntime(
        api_key="ephemeral-secret",
        local_runtime=_FakeLocalRuntime(),
        client=client,
        checkpoint_dir=checkpoint_dir,
        authorized_completion_calls=1,
    )

    with pytest.raises(RuntimeError, match="gateway disconnected"):
        first.complete([{"role": "user", "content": "question"}])

    assert len(client.chat.completions.requests) == 1
    assert first.usage["completion_physical_calls"] == 1
    assert first.usage["completion_logical_calls"] == 0
    assert len(list(checkpoint_dir.glob("*.request.json"))) == 1
    assert list(checkpoint_dir.glob("*.response.json")) == []
    first.close()

    with pytest.raises(RuntimeError, match="refusing an unsafe retry"):
        RecallGuardedCumulativeProviderSynthesisRuntime(
            api_key="ephemeral-secret",
            local_runtime=_FakeLocalRuntime(),
            client=_FakeClient(_response()),
            checkpoint_dir=checkpoint_dir,
            authorized_completion_calls=1,
        )


@pytest.mark.parametrize(
    "binding",
    [
        {"api_key": "different-value"},
        {"nested": {"authorization": "Bearer anything"}},
        {"label": "prefix-ephemeral-secret-suffix"},
    ],
)
def test_campaign_binding_rejects_credential_fields_and_api_key_values(
    binding,
) -> None:
    with pytest.raises(ValueError, match="campaign_binding"):
        RecallGuardedCumulativeProviderSynthesisRuntime(
            api_key="ephemeral-secret",
            local_runtime=_FakeLocalRuntime(),
            client=_FakeClient(_response()),
            campaign_binding=binding,
        )
