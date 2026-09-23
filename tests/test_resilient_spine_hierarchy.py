import json
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from tools import build_spine_corpus_hierarchy_resilient as builder
from tools.matched_eval.artifacts import publish_sealed_json
from tests.test_attention_summary_sections import SummaryLinker
from tests.test_spine_merge_batch import request, response


def test_preserves_valid_slots_and_recovers_only_invalid_slot():
    jobs = tuple(request(str(i)) for i in range(8))
    replies = [f"User grew variety {i}." for i in range(8)]
    replies[5] = "too long " * 200
    recovered = []
    def recover(job, response_sha, slot):
        recovered.append((job, slot))
        return "Recovered user summary."
    cache = builder.RecoveringCache(SimpleNamespace(recover=recover))
    cache.accept(jobs, response(replies))
    assert recovered == [(jobs[5], 5)]
    assert [cache(job) for job in jobs] == [
        "Recovered user summary." if i == 5 else replies[i] for i in range(8)]


@pytest.mark.parametrize("bad", ["null", "[]", "invalid json", json.dumps({"summaries": [
    {"label": "S0", "summary": "First"}, {"label": "S0", "summary": "Second"}]})])
def test_ambiguous_labels_never_salvage_a_guessed_attribution(bad):
    assert builder.recoverable_slots(bad, (request(), request("other"))) == ([None, None], [0, 1])


def test_repair_replays_without_client_and_never_exceeds_two_completed_attempts(tmp_path, monkeypatch):
    preflight, _ = publish_sealed_json(tmp_path / "preflight.json", {"fixture": True})
    sent = []
    outputs = iter(("too long " * 200, "User planted orchards."))
    class Client:
        max_retries = 0
        chat = property(lambda self: SimpleNamespace(completions=SimpleNamespace(create=self.complete)))
        def with_options(self, **kwargs):
            return self
        def close(self):
            pass
        def complete(self, **kwargs):
            sent.append(kwargs)
            return SimpleNamespace(id=f"repair-{len(sent)}", model=builder.MODEL,
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"summary": next(outputs)})),
                                         finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=500, completion_tokens=10, total_tokens=510))
    monkeypatch.setattr(builder, "_completion_client", lambda *args: Client())
    journal = builder.RecoveryJournal(tmp_path, preflight, True, 2)
    job = request()
    assert journal.recover(job, "a" * 64, 0) == "User planted orchards."
    assert journal.calls == journal.scheduled_calls == len(sent) == 2
    assert all(json.loads(row["messages"][1]["content"]) == json.loads(job.messages[1]["content"]) for row in sent)
    assert "48 words" in sent[0]["messages"][0]["content"]
    assert "24 words" in sent[1]["messages"][0]["content"]
    monkeypatch.setattr(builder, "_completion_client", lambda *args: pytest.fail("replay created a client"))
    replay = builder.RecoveryJournal(tmp_path, preflight, False, 0)
    assert replay.recover(job, "a" * 64, 0) == "User planted orchards."
    assert replay.calls == replay.scheduled_calls == 0 and replay.hits == 2
    assert replay.recoveries == journal.recoveries


def test_regular_wave_reserves_all_calls_before_any_recovery(tmp_path, monkeypatch):
    preflight, _ = publish_sealed_json(tmp_path / "preflight.json", {"fixture": True})
    journal = builder.RecoveryJournal(tmp_path, preflight, True, 2)
    monkeypatch.setattr(journal, "remaining", lambda request: 1)
    def completed(**kwargs):
        runtime = kwargs["runtime_factory"](None)
        # No provider I/O; only exercise scheduling and parser sequencing here.
        runtime.close()
        return SimpleNamespace(logical_completions=["null"]), 1, 0, 0
    monkeypatch.setattr(builder, "_run_exactly_authorized", completed)
    def recover(*args):
        assert journal.calls == journal.scheduled_calls == 2
        journal.reserve(1)
        pytest.fail("recovery exceeded reserved wave allowance")
    monkeypatch.setattr(journal, "recover", recover)
    jobs = tuple(request(str(i)) for i in range(9))
    with pytest.raises(builder.NeedsProviderWork):
        journal.resolve({r.prompt_sha256: r for r in jobs}, "fixture", 0)
    assert journal.scheduled_calls == 2


def test_attention_reuses_only_identical_summary_inputs_without_loading_model(tmp_path, monkeypatch):
    parent_root, root = tmp_path / "parent", tmp_path / "current"
    parent, _ = publish_sealed_json(parent_root / "preflight.json", {"fixture": "parent"})
    preflight, _ = publish_sealed_json(root / "preflight.json", {"fixture": "current"})
    texts = ("orchard harvest", "observatory reservations")
    signal = QwenAttentionHeadSurpriseScorer(SummaryLinker(), max_spans=3, span_token_cap=64).score_sequence(texts)
    key = identity_sha256({"preflight_sha256": parent.sha256, "texts": list(texts)})
    publish_sealed_json(parent_root / "attention" / (key + ".json"), {
        "preflight_sha256": parent.sha256, "scores": signal.scores, "similarities": signal.similarities,
        "receipt": signal.receipt.identity_payload()})
    cache = builder.ReusingAttentionCache(root, preflight, [(parent_root, parent)])
    assert cache.score_sequence(texts) == signal and cache.scorer is None
    replay = builder.ReusingAttentionCache(root, preflight, [])
    assert replay.score_sequence(texts) == signal and replay.scorer is None
    # Changed input must not hit either saved artifact.
    class RejectScorer:
        def score_sequence(self, inputs):
            raise ValueError("changed input reached the scorer")
    replay.scorer = RejectScorer()
    with pytest.raises(ValueError, match="changed input"):
        replay.score_sequence(tuple(reversed(texts)))
