from types import SimpleNamespace

import pytest

from tools import benchmark_hot_api_latency as bench
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def fixture(tmp_path, monkeypatch, *, fail_at=None):
    calls = [{"call_index": i, "question_id": str(i // 2), "ordinal": i // 2,
              "arm": "short_api" if i % 2 == 0 else "packet_api",
              "messages": [{"role": "user", "content": str(i)}], "prompt_token_proxy": 4}
             for i in range(4)]
    preflight, _ = publish_sealed_json(tmp_path / "preflight.json", {"calls": calls})
    monkeypatch.setattr(bench, "load", lambda root: preflight)
    sent = []

    def create(**request):
        sent.append(request)
        if fail_at == len(sent):
            raise RuntimeError("interrupted request")
        class Stream:
            def __iter__(self):
                yield {"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}]}
            def close(self):
                pass
        return Stream()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)
    monkeypatch.setattr(bench, "_completion_client", lambda *args: client)
    return preflight, sent


def test_resume_sends_only_new_requests_and_report_reuses_recorded_measurements(tmp_path, monkeypatch):
    _, sent = fixture(tmp_path, monkeypatch)
    bench.run(tmp_path, 2)
    assert len(sent) == 2
    first = read_sealed_json(tmp_path / "report-002.json")
    bench.report(tmp_path)
    assert len(sent) == 2 and read_sealed_json(tmp_path / "report-002.json").sha256 == first.sha256
    assert first.payload["complete_pairs"] == 1
    assert not first.payload["target_gate_eligible"] and not first.payload["live_retrieval_measured"]
    bench.run(tmp_path, 2)
    assert [r["messages"][0]["content"] for r in sent] == ["0", "1", "2", "3"]
    assert read_sealed_json(tmp_path / "report-004.json").payload["complete_pairs"] == 2


def test_uncertain_delivery_reservation_cannot_be_retried(tmp_path, monkeypatch):
    _, sent = fixture(tmp_path, monkeypatch, fail_at=1)
    with pytest.raises(RuntimeError, match="interrupted"):
        bench.run(tmp_path, 2)
    with pytest.raises(ValueError, match="unacknowledged"):
        bench.run(tmp_path, 2)
    assert len(sent) == 1
    assert (tmp_path / "journal/000.failure.json").exists()


def test_call_allowance_checked_before_network(tmp_path, monkeypatch):
    _, sent = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="exceeds remaining"):
        bench.run(tmp_path, 5)
    assert not sent
