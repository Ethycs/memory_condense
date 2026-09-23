import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.search.native_spine_batch import messages
from memory_condense.search.native_spine_summary import fragment_body
from tools import compile_native_spine as compiler
from tools import run_native_spine_batches as runner
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


MODEL = "codex_sdk/gpt-5.6-terra"


def prepare(root, *, missing_last=False):
    bindings = []
    digest = hashlib.sha256()
    for ordinal in range(3):
        fs = fragment_body({"turns": [{"role": "user", "text": f"I bought {ordinal + 1} books yesterday."}]})
        prompt = messages(fs)
        request, _ = publish_sealed_json(root / "requests" / f"{ordinal:06}.json", {
            "sources_sha256": "fixture-source", "ordinal": ordinal,
            "messages": prompt, "messages_sha256": identity_sha256(prompt),
            "pointers": [f.pointer() for f in fs],
            "prompt_token_proxy": count_chat_prompt_token_proxy(prompt),
        })
        bindings.append({"path": str(request.path.relative_to(root)), "sha256": request.sha256,
                         "atoms": len(fs)})
        for f in fs:
            compiler.add_pointer(digest, f)
    publish_sealed_json(root / "preflight.json", {
        "implementation": compiler.implementation(), "models": [MODEL],
        "gateway": compiler.GATEWAY, "retries": 0, "raw_inputs_to_qwen": False,
        "question_or_gold_inputs": False, "mode": "full", "sources_sha256": "fixture-source",
        "requests": bindings[:-1] if missing_last else bindings,
        "max_atoms": 24, "max_prompt_tokens": 7000, "max_new_tokens": 4096,
        "fragment_count": 3, "body_count": 3, "ordered_pointer_sha256": digest.hexdigest(),
        "timeout_s": 240, "concurrency": 1,
    })


class FakeClient:
    max_retries = 0

    def __init__(self, *, invalid=False, fail=False):
        self.invalid, self.fail, self.calls = invalid, fail, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **kwargs):
        assert kwargs == {"timeout": 240, "max_retries": 0}
        return self

    def close(self):
        pass

    def create(self, **kwargs):
        self.calls += 1
        assert kwargs["model"] == MODEL
        if self.fail:
            raise ConnectionError("simulated provider failure")
        rows = json.loads(kwargs["messages"][1]["content"])["transcripts"]
        atoms = [{"label": f["label"], "summary": "User reported a book purchase yesterday."}
                 for body in rows for f in body["fragments"]]
        if self.invalid and self.calls == 1:
            atoms.pop()
        return SimpleNamespace(id=f"fixture-{self.calls}", model=MODEL, usage=None, choices=[
            SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps({"atoms": atoms})))
        ])


def test_dispatch_drains_running_work_and_never_submits_after_observed_failure():
    entered = []
    barrier = threading.Barrier(2)

    def one(n):
        entered.append(n)
        barrier.wait(timeout=5)
        if n == 0:
            raise ConnectionError("provider unavailable")
        time.sleep(0.03)
        return n

    completed, failures = runner.bounded_dispatch(range(20), one, workers=2)
    assert set(entered) == {0, 1}
    assert completed == {1: 1}
    assert failures == {0: {"error_type": "ConnectionError", "http_status": None}}


def test_dispatch_completes_whole_population_with_bounded_concurrency():
    lock = threading.Lock()
    active = peak = 0

    def one(n):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.002)
        with lock:
            active -= 1
        return n * 2

    completed, failures = runner.bounded_dispatch(range(11), one, workers=3)
    assert completed == {n: n * 2 for n in range(11)}
    assert failures == {}
    assert peak <= 3


def test_complete_run_replays_real_journals_without_creating_a_provider(tmp_path, monkeypatch):
    prepare(tmp_path)
    client = FakeClient()
    monkeypatch.setattr(runner, "_completion_client", lambda *args: client)
    first = runner.execute(tmp_path, MODEL, True).payload
    assert first["complete_source_compilation"] is True
    assert first["accepted_atoms"] == 3
    assert first["new_completed_provider_calls"] == client.calls == 3
    assert first["full100_target_passed"] is False
    path = tmp_path / f"bounded-result-{identity_sha256(MODEL)}.json"
    stable = read_sealed_json(path).sha256

    def forbidden(*args):
        raise AssertionError("replay created a provider")

    monkeypatch.setattr(runner, "_completion_client", forbidden)
    replay = runner.execute(tmp_path, MODEL, False).payload
    assert replay["complete_source_compilation"] is True
    assert replay["new_completed_provider_calls"] == 0
    assert replay["replay_hits"] == 3
    assert read_sealed_json(path).sha256 == stable


def test_invalid_summary_keeps_other_batches_and_never_claims_complete(tmp_path, monkeypatch):
    prepare(tmp_path)
    client = FakeClient(invalid=True)
    monkeypatch.setattr(runner, "_completion_client", lambda *args: client)
    result = runner.execute(tmp_path, MODEL, True).payload
    assert result["completed_batches"] == 3
    assert result["accepted_atoms"] == 2
    assert result["complete_source_compilation"] is False
    assert result["failures"] == []
    replay = runner.execute(tmp_path, MODEL, False).payload
    assert replay["accepted_atoms"] == 2
    assert replay["replay_hits"] == 3
    assert client.calls == 3


def test_unanswered_request_stops_dispatch_and_cannot_be_retried(tmp_path, monkeypatch):
    prepare(tmp_path)
    client = FakeClient(fail=True)
    monkeypatch.setattr(runner, "_completion_client", lambda *args: client)
    first = runner.execute(tmp_path, MODEL, True).payload
    assert client.calls == 1
    assert first["not_dispatched_batches"] == 2
    assert first["failed_jobs_may_have_unacknowledged_provider_calls"] is True
    assert first["complete_source_compilation"] is False
    assert len(list((tmp_path / "checkpoints").glob("*/*.request.json"))) == 1
    assert not list((tmp_path / "checkpoints").glob("*/*.response.json"))
    replay = runner.execute(tmp_path, MODEL, True).payload
    assert client.calls == 1
    assert replay["failures"][0]["error_type"] == "RuntimeError"
    assert replay["not_dispatched_batches"] == 2


@pytest.mark.parametrize("defect", ["missing_last", "mutated_last", "qwen"])
def test_entire_population_is_admitted_before_any_provider_call(tmp_path, monkeypatch, defect):
    prepare(tmp_path, missing_last=defect == "missing_last")
    if defect == "mutated_last":
        path = tmp_path / "requests" / "000002.json"
        path.write_bytes(path.read_bytes().replace(b"books", b"coats"))
    client = FakeClient()
    monkeypatch.setattr(runner, "_completion_client", lambda *args: client)
    with pytest.raises(ValueError):
        runner.execute(tmp_path, "qwen3-8b" if defect == "qwen" else MODEL, True)
    assert client.calls == 0
    assert not (tmp_path / "checkpoints").exists()
