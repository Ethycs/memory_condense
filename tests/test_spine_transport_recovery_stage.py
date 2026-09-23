from types import SimpleNamespace

import pytest

from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from tools.matched_eval.artifacts import publish_sealed_json
from tools.stage_spine_transport_recovery import copy_completed_checkpoint, copy_exact, journal_state


class Client:
    max_retries = 0

    def __init__(self, *, fail=False):
        self.fail = fail
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **kwargs):
        assert kwargs["max_retries"] == 0
        return self

    def create(self, **kwargs):
        if self.fail:
            raise TimeoutError("fixture transport timeout")
        return SimpleNamespace(id="recorded-response", model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content="Saved user summary."), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=4, total_tokens=8))


def fixture(tmp_path, *, fail=False, prompt="Original raw fragment."):
    request, _ = publish_sealed_json(tmp_path / "request.json", {
        "model": "codex_sdk/gpt-5.6-terra", "messages": [{"role": "user", "content": prompt}]})
    checkpoint = tmp_path / "original"
    runtime = FastCompletionRuntime(checkpoint_dir=checkpoint,
        prompt_population=[request.payload["messages"]], model=request.payload["model"],
        client=Client(fail=fail), max_prompt_tokens=7000, max_new_tokens=3072,
        max_concurrency=1, retries=0, benchmark_provenance={"raw_request_sha256": request.sha256})
    try:
        if fail:
            with pytest.raises(TimeoutError):
                runtime.run()
        else:
            runtime.run()
    finally:
        runtime.close()
    return request, checkpoint


def test_completed_evidence_copies_and_replays_without_changing_original(tmp_path):
    request, original = fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in original.glob("*.json")}
    target = tmp_path / "successor"
    copied = copy_completed_checkpoint(original, target, request)
    assert len(copied) == 2
    assert {p.name: p.read_bytes() for p in target.glob("*.json")} == before
    assert {p.name: p.read_bytes() for p in original.glob("*.json")} == before
    assert journal_state(original) == journal_state(target)
    # A further read-only copy exercises the real runtime's replay contract.
    assert copy_completed_checkpoint(target, tmp_path / "replayed", request) == copied


def test_unresolved_request_cannot_be_imported_as_completed_or_cleared(tmp_path):
    request, original = fixture(tmp_path, fail=True)
    before = {p.name: p.read_bytes() for p in original.glob("*.json")}
    target = tmp_path / "successor"
    with pytest.raises(RuntimeError, match="no response"):
        copy_completed_checkpoint(original, target, request)
    assert not target.exists()
    assert {p.name: p.read_bytes() for p in original.glob("*.json")} == before
    assert journal_state(original)["state"] == "reserved_without_response"


def test_valid_journals_for_a_different_prompt_cannot_be_imported(tmp_path):
    expected, _ = fixture(tmp_path / "expected")
    _, foreign = fixture(tmp_path / "foreign", prompt="Different raw fragment.")
    with pytest.raises(ValueError):
        copy_completed_checkpoint(foreign, tmp_path / "successor", expected)
    assert not (tmp_path / "successor").exists()


def test_copy_refuses_to_replace_different_saved_evidence(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"original")
    target.write_bytes(b"different existing evidence")
    with pytest.raises(ValueError, match="differs"):
        copy_exact(source, target)
    assert source.read_bytes() == b"original"
    assert target.read_bytes() == b"different existing evidence"
