from __future__ import annotations

import memory_condense.eval.transition_trace as transition_trace
from tools.matched_eval.source_gate_controller import _SealedRecord


def test_transition_score_cache_is_bounded_to_one_explicit_lifecycle(monkeypatch):
    calls = {"contains": 0, "f1": 0, "tokens": 0}

    def contains(texts, answer):
        calls["contains"] += 1
        return answer in texts[0]

    def f1(text, answer):
        calls["f1"] += 1
        return float(answer in text)

    def tokens(text):
        calls["tokens"] += 1
        return len(text)

    monkeypatch.setattr(transition_trace, "contains_answer", contains)
    monkeypatch.setattr(transition_trace, "f1_score", f1)
    monkeypatch.setattr(transition_trace, "count_tokens", tokens)
    cache = transition_trace._TransitionScoreCache()

    assert cache.candidate_contains("same text", "text") is True
    assert cache.candidate_contains("same text", "text") is True
    assert cache.candidate_f1("same text", "text") == 1.0
    assert cache.candidate_f1("same text", "text") == 1.0
    assert cache.rendered_token_count("same text") == 9
    assert cache.rendered_token_count("same text") == 9
    assert cache.stats() == {"entries": 3, "hits": 3, "misses": 3}
    assert calls == {"contains": 1, "f1": 1, "tokens": 1}

    cache.clear()

    assert cache.stats() == {"entries": 0, "hits": 0, "misses": 0}
    assert cache.rendered_token_count("same text") == 9
    assert calls["tokens"] == 2


def test_sealed_record_base_does_not_allocate_an_instance_dictionary():
    assert not hasattr(_SealedRecord(), "__dict__")
