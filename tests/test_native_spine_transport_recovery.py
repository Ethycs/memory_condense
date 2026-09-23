from pathlib import Path

import pytest

from tools import recover_native_spine_transport as recovery
from tools import run_native_spine_batches as runner
from tests.test_native_spine_dispatch import FakeClient, MODEL, prepare


def interrupted(tmp_path, monkeypatch):
    source = tmp_path/"source"
    prepare(source)
    client = FakeClient(fail=True)
    monkeypatch.setattr(runner, "_completion_client", lambda *args: client)
    report = runner.execute(source, MODEL, True)
    assert client.calls == 1 and len(report.payload["failures"]) == 1
    return source, report


def journals(source):
    return {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*.json")}


def test_replacement_is_separate_preserves_original_and_replays_without_provider(tmp_path, monkeypatch):
    source, report = interrupted(tmp_path, monkeypatch)
    before = journals(source)
    root = tmp_path/"recovery"
    plan = recovery.prepare(source, report.path, root, [0])
    assert plan.payload["maximum_combined_original_and_recovery_attempts"] == 2
    client = FakeClient()
    monkeypatch.setattr(recovery, "_completion_client", lambda *args: client)
    result = recovery.execute(root, True)
    assert client.calls == 1 and result.payload["all_recovery_summaries_accepted"]
    assert journals(source) == before
    monkeypatch.setattr(recovery, "_completion_client", lambda *args: pytest.fail("provider during replay"))
    assert recovery.execute(root, False).sha256 == result.sha256
    assert journals(source) == before
    parent, bindings = runner.load_requests(source, MODEL)
    with pytest.raises(RuntimeError, match="no response"):
        runner.run_one(source, parent, MODEL, bindings[0], False)


@pytest.mark.parametrize("ordinals", [[1], [0, 0], [-1]])
def test_only_explicit_failed_original_population_is_eligible(tmp_path, monkeypatch, ordinals):
    source, report = interrupted(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="explicit distinct failed"):
        recovery.prepare(source, report.path, tmp_path/"recovery", ordinals)


def test_failed_replacement_remains_terminal_without_an_implicit_second_recovery(tmp_path, monkeypatch):
    source, report = interrupted(tmp_path, monkeypatch)
    root = tmp_path/"recovery"
    recovery.prepare(source, report.path, root, [0])
    client = FakeClient(fail=True)
    monkeypatch.setattr(recovery, "_completion_client", lambda *args: client)
    with pytest.raises(ConnectionError):
        recovery.execute(root, True)
    with pytest.raises(RuntimeError, match="no response"):
        recovery.execute(root, True)
    assert client.calls == 1 and not (root/"result.json").exists()


def test_invalid_replacement_does_not_admit_source_atoms(tmp_path, monkeypatch):
    source, report = interrupted(tmp_path, monkeypatch)
    root = tmp_path/"recovery"
    recovery.prepare(source, report.path, root, [0])
    monkeypatch.setattr(recovery, "_completion_client", lambda *args: FakeClient(invalid=True))
    result = recovery.execute(root, True)
    assert not result.payload["all_recovery_summaries_accepted"]
    assert result.payload["rows"][0]["status"] == "invalid_summary"


def test_corrupt_unanswered_journal_is_not_replaced(tmp_path, monkeypatch):
    source, report = interrupted(tmp_path, monkeypatch)
    journal, = (source/"checkpoints").rglob("*.request.json")
    journal.write_bytes(journal.read_bytes().replace(b'"max_new_tokens":4096', b'"max_new_tokens":4095'))
    with pytest.raises(ValueError):
        recovery.prepare(source, report.path, tmp_path/"recovery", [0])
