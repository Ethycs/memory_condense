from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from tools import evaluate_spine_combined_reader as evaluation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def setup_runtime(tmp_path, monkeypatch, *, changed_live=False):
    question = {"ordinal": 0, "question_id": "fixture", "retrieval_query": "Which adapter?",
                "prompt_question": "Which adapter?"}
    contexts = {"source_spine_diverse": "EXACT_RAW_CANARY",
                "source_spine_combined": "EXACT_RAW_CANARY\nADDITIONAL_USER_EVIDENCE"}
    calls = []
    for arm in evaluation.ARMS:
        policy = arm.removesuffix("_short_api").removesuffix("_api")
        hydrated = None if arm.endswith("_short_api") else SimpleNamespace(
            render_context=lambda policy=policy: contexts[evaluation.ROUTES[policy]])
        messages = evaluation.answer_messages(question, hydrated, policy=policy)
        calls.append({"call_index": len(calls), "arm": arm, "question": question,
                      "messages": messages, "messages_sha256": identity_sha256(messages)})
    preflight, _ = publish_sealed_json(tmp_path / "preflight.json", {"calls": calls,
        "index_root": str(tmp_path), "index_manifest_sha256": "fixture", "addresses_root": str(tmp_path),
        "addresses_sha256": "fixture", "atoms_path": str(tmp_path), "atoms_sha256": "fixture",
        "facets_root": str(tmp_path), "facets_sha256": "fixture", "role_partition_sha256": "fixture"})
    monkeypatch.setattr(evaluation, "load_preflight", lambda _: preflight)
    events = []

    class Memory:
        encoder = SimpleNamespace(close=lambda: None)
        source_spine = SimpleNamespace(index=SimpleNamespace(receipt_sha256="fixture"))

        def __init__(self, *args):
            pass

        def retrieve(self, query, route, dated_question):
            events.append(("retrieve", route))
            context = contexts[route]
            if changed_live and route == "source_spine_combined":
                context = "CHANGED_RAW_EVIDENCE"
            return SimpleNamespace(render_context=lambda: context, diagnostics=(),
                identity_payload=lambda: {"exact_hydration": True, "route": route})

    class Stream:
        def __iter__(self):
            yield {"choices": [{"delta": {"content": "Compatible adapter"}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        def close(self):
            pass

    def create(**kwargs):
        events.append(("api", kwargs["messages"]))
        return Stream()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)
    monkeypatch.setattr(evaluation, "ResidentMemory", Memory)
    monkeypatch.setattr(evaluation, "_completion_client", lambda *args: client)
    return preflight, events


def test_different_candidate_evidence_uses_live_route_and_shared_v3_short_control(tmp_path, monkeypatch):
    preflight, events = setup_runtime(tmp_path, monkeypatch)
    evaluation.run(tmp_path, 8)
    prompts = [e[1] for e in events if e[0] == "api"]
    assert prompts[2] == prompts[3] and prompts[4] == prompts[5] and prompts[6] == prompts[7]
    assert prompts[2][1:] == prompts[4][1:] != prompts[6][1:]
    assert prompts[4][0] == prompts[6][0] == prompts[1][0] != prompts[0][0]
    assert evaluation.short_control("combined_reader") == "reader_short_api"
    assert [e[1] for e in events if e[0] == "retrieve"] == [
        "source_spine_diverse", "source_spine_diverse", "source_spine_combined"]
    assert len(evaluation.recorded(tmp_path, preflight)) == 8


def test_changed_candidate_stops_before_its_api_call_and_is_not_retried(tmp_path, monkeypatch):
    preflight, events = setup_runtime(tmp_path, monkeypatch, changed_live=True)
    with pytest.raises(ValueError, match="live retrieval changed"):
        evaluation.run(tmp_path, 8)
    assert len([e for e in events if e[0] == "api"]) == 6
    assert (tmp_path / "journal/006.reserved").exists()
    assert read_sealed_json(tmp_path / "journal/006.failure.json").payload["retry_performed"] is False
    assert not (tmp_path / "answers.json").exists()
    with pytest.raises(ValueError, match="unacknowledged"):
        evaluation.run(tmp_path, 2)
    assert len([e for e in events if e[0] == "api"]) == 6


def test_candidate_cannot_silently_change_reader_with_its_matched_api(tmp_path, monkeypatch):
    preflight, _ = setup_runtime(tmp_path, monkeypatch)
    calls = preflight.payload["calls"]
    for call in calls:
        if call["arm"] in ("combined_reader", "combined_reader_api"):
            call["messages"][0]["content"] = evaluation.QA_SYSTEM_PROMPT
            call["messages_sha256"] = identity_sha256(call["messages"])
    with pytest.raises(ValueError, match="another reader policy"):
        evaluation.validate_matched_calls(calls)
