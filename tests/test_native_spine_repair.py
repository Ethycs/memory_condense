import hashlib
import json
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.search.native_spine_batch import messages
from memory_condense.search.native_spine_repair import partition, reconcile, repair_messages
from memory_condense.search.native_spine_summary import fragment_body
from tools import compile_native_spine as compiler
from tools import repair_native_spine_batches as repair
from tools import run_native_spine_batches as runner
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def inputs():
    return fragment_body({"turns": [
        {"role": "user", "text": "I plan to buy a blue chair tomorrow."},
        {"role": "assistant", "text": "Try the independent furniture store."},
    ]})


def original_response():
    return canonical_json({"atoms": [
        {"label": "T0", "summary": "User plans to buy a blue chair tomorrow. " * 20},
        {"label": "T1", "summary": "Assistant suggests an independent furniture store."},
    ]})


def replacement():
    return canonical_json({"atoms": [{"label": "T0", "summary": "User plans to buy a blue chair tomorrow."}]})


def test_repair_keeps_valid_text_and_every_raw_pointer_exact():
    fs = inputs()
    valid, bad = partition(original_response(), fs)
    assert set(valid) == {1} and bad == (0,)
    repaired = reconcile(original_response(), fs, {0: replacement()})
    assert repaired[1] == valid[1]
    assert [a["pointer"] for a in repaired] == [f.pointer() for f in fs]
    prompt = repair_messages(fs[0])
    assert fs[0].text in prompt[1]["content"]
    assert fs[1].text not in prompt[1]["content"]
    assert fs[0].body_sha256 not in prompt[1]["content"]
    assert "90 tokens" in prompt[0]["content"]


@pytest.mark.parametrize("defect", ["missing", "reordered", "empty", "duplicate_key"])
def test_repair_cannot_guess_missing_original_content_or_attribution(defect):
    value = json.loads(original_response())
    if defect == "missing":
        value["atoms"].pop()
    elif defect == "reordered":
        value["atoms"].reverse()
    elif defect == "empty":
        value["atoms"][0]["summary"] = ""
    text = json.dumps(value)
    if defect == "duplicate_key":
        text = text.replace('"summary":', '"summary": "ambiguous", "summary":', 1)
    with pytest.raises(ValueError):
        partition(text, inputs())


@pytest.mark.parametrize("replacements", [{}, {1: replacement()}, {0: replacement(), 1: replacement()}])
def test_repair_must_cover_only_and_all_failed_atoms(replacements):
    with pytest.raises(ValueError):
        reconcile(original_response(), inputs(), replacements)


class Client:
    max_retries = 0

    def __init__(self, response, fail=False):
        self.response, self.fail, self.calls = response, fail, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **kwargs):
        assert kwargs["max_retries"] == 0
        return self

    def close(self):
        pass

    def create(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise ConnectionError("simulated transport failure")
        return SimpleNamespace(id="fixture", model=repair.MODEL, usage=None, choices=[
            SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=self.response))
        ])


def original_checkpoint(root, monkeypatch):
    fs = inputs()
    prompt = messages(fs)
    request, _ = publish_sealed_json(root / "requests" / "000000.json", {
        "ordinal": 0, "sources_sha256": "fixture", "messages": prompt,
        "messages_sha256": identity_sha256(prompt), "pointers": [f.pointer() for f in fs],
        "prompt_token_proxy": count_chat_prompt_token_proxy(prompt),
    })
    digest = hashlib.sha256()
    for f in fs:
        compiler.add_pointer(digest, f)
    publish_sealed_json(root / "preflight.json", {
        "implementation": compiler.implementation(), "models": [repair.MODEL],
        "gateway": compiler.GATEWAY, "retries": 0, "raw_inputs_to_qwen": False,
        "question_or_gold_inputs": False, "mode": "full", "sources_sha256": "fixture",
        "requests": [{"path": "requests/000000.json", "sha256": request.sha256, "atoms": 2}],
        "max_atoms": 24, "max_prompt_tokens": 7000, "max_new_tokens": 4096,
        "fragment_count": 2, "body_count": 1, "ordered_pointer_sha256": digest.hexdigest(),
        "timeout_s": 240, "concurrency": 1,
    })
    client = Client(original_response())
    monkeypatch.setattr(runner, "_completion_client", lambda *args: client)
    original = runner.execute(root, repair.MODEL, True)
    assert original.payload["accepted_atoms"] == 0 and client.calls == 1
    return read_sealed_json(next((root / "validated").glob("*.json")))


def test_real_journal_repair_and_replay_leave_original_failure_unchanged(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "repair"
    original = original_checkpoint(source, monkeypatch)
    plan = repair.prepare(source, output, [0])
    assert plan.payload["maximum_new_provider_calls"] == 1
    client = Client(replacement())
    monkeypatch.setattr(repair, "_completion_client", lambda *args: client)
    result = repair.execute(output, True)
    assert result.payload["repaired_atoms"] == 1
    assert result.payload["unchanged_valid_atoms"] == 1
    assert result.payload["full_source_compilation_complete"] is False
    merged = read_sealed_json(output / "repaired-batches" / "000000.json").payload
    assert merged["summaries"][1]["summary"] == json.loads(original_response())["atoms"][1]["summary"]
    assert read_sealed_json(original.path).sha256 == original.sha256

    def forbidden(*args):
        raise AssertionError("replay created a provider")

    monkeypatch.setattr(repair, "_completion_client", forbidden)
    replay = repair.execute(output, False)
    assert replay.sha256 == result.sha256 and client.calls == 1


def test_failed_repair_transport_cannot_repeat_an_unanswered_call(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "repair"
    original_checkpoint(source, monkeypatch)
    repair.prepare(source, output, [0])
    client = Client(replacement(), fail=True)
    monkeypatch.setattr(repair, "_completion_client", lambda *args: client)
    with pytest.raises(ConnectionError):
        repair.execute(output, True)
    with pytest.raises(RuntimeError, match="refusing an unsafe retry"):
        repair.execute(output, True)
    assert client.calls == 1
    assert not (output / "result.json").exists()
