from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval.recall_guarded_cumulative_semantic_judge_runtime import (
    DEFAULT_JUDGE_MODEL,
    RecallGuardedCumulativeSemanticJudgeRuntime,
)


def test_semantic_preflight_import_is_provider_free_with_sockets_blocked() -> None:
    code = r"""
import socket
import sys

def blocked_connect(*args, **kwargs):
    raise AssertionError("semantic preflight import attempted network access")

socket.socket.connect = blocked_connect
import memory_condense.eval.recall_guarded_cumulative_semantic_judge
import memory_condense.eval.recall_guarded_cumulative_semantic_judge_runtime
assert "litellm" not in sys.modules
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
    def __init__(self, *responses) -> None:
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
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(*responses)
        )
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _response(text: str = "CORRECT: same fact"):
    return SimpleNamespace(
        id="judge-test",
        model="codex_sdk/gpt-5.6-sol-test",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=31,
            completion_tokens=5,
            total_tokens=36,
        ),
    )


def _messages(prediction: str = "blue") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Return a binary verdict."},
        {"role": "user", "content": f"Prediction: {prediction}"},
    ]


def _runtime(tmp_path, client, *, replay_only: bool = False, cap: int = 2):
    return RecallGuardedCumulativeSemanticJudgeRuntime(
        checkpoint_dir=tmp_path / "judge-calls",
        campaign_binding={
            "synthesis_sha256": "a" * 64,
            "population_identity_sha256": "b" * 64,
        },
        authorized_unique_calls=cap,
        api_key=None if replay_only else "ephemeral-test-secret",
        replay_only=replay_only,
        client=None if replay_only else client,
    )


def test_live_call_journals_then_fresh_process_replays_without_client(
    tmp_path,
) -> None:
    client = _FakeClient(_response())
    first = _runtime(tmp_path, client, cap=1)

    verdict, text = first.judge(_messages())

    assert verdict is True
    assert text == "CORRECT: same fact"
    assert client.chat.completions.requests == [
        {
            "model": "codex_sdk/gpt-5.6-sol",
            "messages": _messages(),
            "max_tokens": 1024,
        }
    ]
    assert first.identity.caller_model == DEFAULT_JUDGE_MODEL
    assert first.identity.retries == 0
    assert first.identity.temperature is None
    assert not hasattr(first.identity, "replay_only")
    first_identity = first.identity.model_dump()
    first_record = dict(first.last_journal_record or {})
    first.close()

    request_path = next((tmp_path / "judge-calls").glob("*.request.json"))
    response_path = next((tmp_path / "judge-calls").glob("*.response.json"))
    serialized = request_path.read_text() + response_path.read_text()
    assert "ephemeral-test-secret" not in serialized
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert request_payload["call_key_sha256"] == identity_sha256(
        request_payload["call_key_payload"]
    )

    replay = _runtime(tmp_path, None, replay_only=True, cap=1)
    assert replay.identity.model_dump() == first_identity
    replay_verdict, replay_text = replay.judge(_messages())
    assert (replay_verdict, replay_text) == (verdict, text)
    assert replay.last_journal_record == first_record
    assert replay.last_completion_report.cache_hit is True
    assert replay.last_completion_report.physical_call is False
    assert replay.usage["physical_calls"] == 0
    assert replay.usage["checkpoint_hits"] == 1
    replay.close()


def test_malformed_verdict_is_durably_replayed_without_retry(tmp_path) -> None:
    client = _FakeClient(_response("maybe"))
    first = _runtime(tmp_path, client, cap=1)
    with pytest.raises(RuntimeError, match="malformed verdict"):
        first.judge(_messages())
    first.close()
    assert len(client.chat.completions.requests) == 1
    assert len(list((tmp_path / "judge-calls").glob("*.response.json"))) == 1

    replay = _runtime(tmp_path, None, replay_only=True, cap=1)
    with pytest.raises(RuntimeError, match="malformed verdict"):
        replay.judge(_messages())
    assert replay.usage["physical_calls"] == 0
    replay.close()


def test_budget_blocks_new_unique_prompt_before_request(tmp_path) -> None:
    client = _FakeClient(_response(), _response("INCORRECT: forbidden"))
    runtime = _runtime(tmp_path, client, cap=1)
    assert runtime.judge(_messages())[0] is True
    assert runtime.judge(_messages())[0] is True
    with pytest.raises(RuntimeError, match="budget exhausted"):
        runtime.judge(_messages("red"))
    assert len(client.chat.completions.requests) == 1
    runtime.close()


def test_interrupted_request_is_never_retried(tmp_path) -> None:
    client = _FakeClient(RuntimeError("gateway disconnected"))
    first = _runtime(tmp_path, client, cap=1)
    with pytest.raises(RuntimeError, match="gateway disconnected"):
        first.judge(_messages())
    first.close()
    assert len(list((tmp_path / "judge-calls").glob("*.request.json"))) == 1
    assert list((tmp_path / "judge-calls").glob("*.response.json")) == []

    with pytest.raises(RuntimeError, match="refusing an unsafe retry"):
        _runtime(tmp_path, None, replay_only=True, cap=1)
