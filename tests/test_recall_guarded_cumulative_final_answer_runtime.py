from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.eval import (
    recall_guarded_cumulative_final_answer_runtime as final_runtime,
)


def test_final_answer_runtime_import_is_provider_free_with_sockets_blocked() -> None:
    code = r"""
import socket
import sys

def blocked_connect(*args, **kwargs):
    raise AssertionError("final-answer runtime import attempted network access")

socket.socket.connect = blocked_connect
import memory_condense.eval.recall_guarded_cumulative_final_answer_runtime
assert "litellm" not in sys.modules
assert "openai" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


class _FakeCompletions:
    def __init__(self, *responses, checkpoint_dir: Path | None = None) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []
        self.checkpoint_dir = checkpoint_dir

    def create(self, **request):
        self.requests.append(request)
        if self.checkpoint_dir is not None:
            assert len(list(self.checkpoint_dir.glob("*.request.json"))) == 1
            assert list(self.checkpoint_dir.glob("*.response.json")) == []
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _FakeClient:
    def __init__(self, *responses, checkpoint_dir: Path | None = None) -> None:
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(
                *responses,
                checkpoint_dir=checkpoint_dir,
            )
        )
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _response(
    text: str = "blue",
    *,
    prompt_tokens: int = 31,
    completion_tokens: int = 2,
    total_tokens: int = 33,
):
    return SimpleNamespace(
        id="answer-test",
        model="codex_sdk/gpt-5.6-terra-test",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


def _messages(question: str = "What color?") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Answer only from the excerpts."},
        {"role": "user", "content": f"Question: {question}\nShort answer:"},
    ]


def _runtime(
    checkpoint_dir: Path,
    client,
    *,
    population=None,
    cap: int = 1,
    replay_only: bool = False,
    **kwargs,
):
    return final_runtime.RecallGuardedCumulativeFinalAnswerRuntime(
        checkpoint_dir=checkpoint_dir,
        campaign_binding={
            "retrieval_sha256": "a" * 64,
            "fixed_stage_id": "direct_episode_additions",
        },
        prompt_population=population or [_messages()],
        authorized_unique_calls=cap,
        api_key=None if replay_only else "ephemeral-test-secret",
        replay_only=replay_only,
        client=None if replay_only else client,
        **kwargs,
    )


def test_live_call_is_locked_and_request_is_journaled_before_network(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "final-answer-calls"
    client = _FakeClient(_response(), checkpoint_dir=checkpoint_dir)
    runtime = _runtime(checkpoint_dir, client)

    answer = runtime.complete(_messages())

    assert answer == "blue"
    assert client.chat.completions.requests == [
        {
            "model": "codex_sdk/gpt-5.6-terra",
            "messages": _messages(),
            "max_tokens": 256,
        }
    ]
    assert runtime.identity.format == (
        "memory-condense-recall-guarded-fixed-stage-final-answer-runtime-v1"
    )
    assert runtime.identity.caller_model == final_runtime.LOCKED_FINAL_ANSWER_MODEL
    assert runtime.identity.gateway_model == (
        final_runtime.LOCKED_FINAL_ANSWER_GATEWAY_MODEL
    )
    assert runtime.identity.default_max_new_tokens == 256
    assert runtime.identity.max_prompt_token_proxy == 8_000
    assert runtime.identity.retries == 0
    assert runtime.identity.temperature is None
    assert runtime.usage["physical_calls"] == 1
    state = runtime.request_token_state_receipt()
    assert state == {
        "contract": final_runtime.FINAL_ANSWER_REQUEST_TOKEN_STATE_CONTRACT,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "stateless_journaled_provider_completion_runtime"
        ),
        "external_provider_persistence_certified": False,
    }
    runtime.close()
    assert client.close_calls == 1


def test_fresh_replay_has_no_client_and_preserves_immutable_record(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "final-answer-calls"
    client = _FakeClient(_response("Miss Bee Providore"))
    first = _runtime(checkpoint_dir, client)
    expected = first.complete(_messages())
    identity = first.identity.model_dump()
    record = dict(first.last_journal_record or {})
    response_path = next(checkpoint_dir.glob("*.response.json"))
    response_bytes = response_path.read_bytes()
    first.close()

    replay = _runtime(checkpoint_dir, None, replay_only=True)
    observed = replay.complete(_messages())

    assert observed == expected
    assert replay.identity.model_dump() == identity
    assert replay.last_journal_record == record
    assert replay.last_completion_report is not None
    assert replay.last_completion_report.cache_hit is True
    assert replay.last_completion_report.physical_call is False
    assert replay.usage["logical_calls"] == 1
    assert replay.usage["unique_calls"] == 1
    assert replay.usage["physical_calls"] == 0
    assert replay.usage["checkpoint_hits"] == 1
    assert response_path.read_bytes() == response_bytes
    replay.close()


def test_complete_population_is_preflighted_before_directory_or_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "must-not-exist"
    original = final_runtime.count_chat_prompt_token_proxy

    def count(messages) -> int:
        if "late over-cap" in messages[-1]["content"]:
            return 8_001
        return original(messages)

    monkeypatch.setattr(final_runtime, "count_chat_prompt_token_proxy", count)
    client = _FakeClient(_response(), _response())

    with pytest.raises(ValueError, match="exceeds.*8000-token cap.*ordinal 1"):
        _runtime(
            checkpoint_dir,
            client,
            population=[_messages(), _messages("late over-cap")],
            cap=2,
        )

    assert not checkpoint_dir.exists()
    assert client.chat.completions.requests == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"caller_model": "openai/codex_sdk/gpt-5.6-sol"}, "locked Terra"),
        ({"max_new_tokens": 255}, "locked 256-token"),
        ({"max_new_tokens": True}, "locked 256-token"),
        ({"max_prompt_tokens": 7_999}, "locked 8000-token"),
        ({"max_prompt_tokens": 8_000.0}, "locked 8000-token"),
    ],
)
def test_route_and_budgets_are_hard_locked_before_path_mutation(
    tmp_path: Path,
    kwargs: dict[str, object],
    match: str,
) -> None:
    checkpoint_dir = tmp_path / "must-not-exist"
    client = _FakeClient(_response())

    with pytest.raises(ValueError, match=match):
        _runtime(checkpoint_dir, client, **kwargs)

    assert not checkpoint_dir.exists()
    assert client.chat.completions.requests == []


@pytest.mark.parametrize("authorized", [True, 2])
def test_authorization_must_exactly_match_unique_population(
    authorized: object,
) -> None:
    with pytest.raises(ValueError, match="authorized"):
        final_runtime.preflight_final_answer_prompt_population(
            [_messages()],
            authorized_unique_calls=authorized,  # type: ignore[arg-type]
        )


def test_duplicate_logical_prompts_share_one_journaled_physical_call(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "final-answer-calls"
    client = _FakeClient(_response())
    runtime = _runtime(
        checkpoint_dir,
        client,
        population=[_messages(), _messages()],
        cap=1,
    )

    assert runtime.prompt_population.logical_prompt_count == 2
    assert runtime.prompt_population.unique_prompt_count == 1
    assert runtime.complete(_messages()) == "blue"
    assert runtime.complete(_messages()) == "blue"
    assert runtime.usage["logical_calls"] == 2
    assert runtime.usage["unique_calls"] == 1
    assert runtime.usage["physical_calls"] == 1
    assert runtime.usage["checkpoint_hits"] == 1
    assert len(client.chat.completions.requests) == 1
    runtime.close()


def test_prompt_outside_population_fails_before_journal_or_network(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "final-answer-calls"
    client = _FakeClient(_response())
    runtime = _runtime(checkpoint_dir, client)

    with pytest.raises(ValueError, match="outside the preflighted population"):
        runtime.complete(_messages("Different question?"))

    assert list(checkpoint_dir.glob("*.json")) == []
    assert client.chat.completions.requests == []
    runtime.close()


@pytest.mark.parametrize("maximum", [255, 257, True, 256.0])
def test_per_call_output_override_cannot_escape_locked_allowance(
    tmp_path: Path,
    maximum: object,
) -> None:
    checkpoint_dir = tmp_path / "final-answer-calls"
    client = _FakeClient(_response())
    runtime = _runtime(checkpoint_dir, client)

    with pytest.raises(ValueError, match="locked 256-token allowance"):
        runtime.complete(
            _messages(),
            max_new_tokens=maximum,  # type: ignore[arg-type]
        )

    assert list(checkpoint_dir.glob("*.json")) == []
    assert client.chat.completions.requests == []
    runtime.close()


def test_interrupted_request_is_terminal_and_never_retried(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "final-answer-calls"
    client = _FakeClient(RuntimeError("gateway disconnected"))
    first = _runtime(checkpoint_dir, client)

    with pytest.raises(RuntimeError, match="gateway disconnected"):
        first.complete(_messages())
    first.close()

    assert len(client.chat.completions.requests) == 1
    assert len(list(checkpoint_dir.glob("*.request.json"))) == 1
    assert list(checkpoint_dir.glob("*.response.json")) == []
    with pytest.raises(RuntimeError, match="refusing an unsafe retry"):
        _runtime(checkpoint_dir, None, replay_only=True)


def test_zero_filled_provider_usage_is_recorded_as_unavailable_and_secret_free(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "final-answer-calls"
    client = _FakeClient(
        _response(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
    )
    runtime = _runtime(checkpoint_dir, client)
    runtime.complete(_messages())
    runtime.close()

    report = runtime.last_journal_record["completion_report"]
    assert report["reported_usage_available"] is False
    assert report["reported_input_tokens_available"] is False
    assert report["reported_output_tokens_available"] is False
    assert report["reported_total_tokens_available"] is False
    serialized = "".join(
        path.read_text(encoding="utf-8")
        for path in checkpoint_dir.glob("*.json")
    )
    assert "ephemeral-test-secret" not in serialized


def test_noncanonical_or_tampered_journal_is_rejected_on_replay(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "final-answer-calls"
    first = _runtime(checkpoint_dir, _FakeClient(_response()))
    first.complete(_messages())
    first.close()
    response_path = next(checkpoint_dir.glob("*.response.json"))
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    response_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not canonical JSON"):
        _runtime(checkpoint_dir, None, replay_only=True)
