from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from tools import evaluate_spine_as_of as evaluation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def setup_runtime(tmp_path, monkeypatch, *, changed_live=False):
    question = {"ordinal": 0, "question_id": "fixture", "retrieval_query": "Which adapter?",
        "prompt_question": "Which adapter?"}
    contexts = {"source_spine_semantic_seeds": "EXACT_RAW_CONTROL",
        "source_spine_as_of": "EXACT_RAW_DATE_SELECTION"}
    calls = []
    for arm in evaluation.ARMS:
        policy = "semantic_seeds" if arm == "short_api" else arm.removesuffix("_api")
        hydrated = None if arm == "short_api" else SimpleNamespace(
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
            context = "CHANGED_EVIDENCE" if changed_live and route == "source_spine_as_of" else contexts[route]
            return SimpleNamespace(render_context=lambda: context, diagnostics=(),
                identity_payload=lambda: {"exact_hydration": True})
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


def test_live_candidate_and_control_have_distinct_evidence_and_one_matched_reader(tmp_path, monkeypatch):
    preflight, events = setup_runtime(tmp_path, monkeypatch)
    evaluation.run(tmp_path, 5)
    prompts = [event[1] for event in events if event[0] == "api"]
    assert prompts[1] == prompts[2] and prompts[3] == prompts[4] and prompts[1][1:] != prompts[3][1:]
    assert all(prompt[0] == prompts[0][0] for prompt in prompts)
    assert "EXACT_RAW" not in str(prompts[0])
    assert [event[1] for event in events if event[0] == "retrieve"] == list(evaluation.ROUTES.values())
    assert len(evaluation.recorded(tmp_path, preflight)) == 5


def test_changed_live_candidate_fails_before_api_and_preserves_unacknowledged_request(tmp_path, monkeypatch):
    _, events = setup_runtime(tmp_path, monkeypatch, changed_live=True)
    with pytest.raises(ValueError, match="live retrieval changed"):
        evaluation.run(tmp_path, 5)
    assert len([event for event in events if event[0] == "api"]) == 3
    assert (tmp_path / "journal/003.reserved").exists()
    assert read_sealed_json(tmp_path / "journal/003.failure.json").payload["retry_performed"] is False
    with pytest.raises(ValueError, match="unacknowledged"):
        evaluation.run(tmp_path, 2)
    assert len([event for event in events if event[0] == "api"]) == 3


def test_candidate_and_its_api_cannot_change_reader_together(tmp_path, monkeypatch):
    preflight, _ = setup_runtime(tmp_path, monkeypatch)
    for call in preflight.payload["calls"]:
        if call["arm"].startswith("as_of"):
            call["messages"][0]["content"] = "CHANGED_READER"
            call["messages_sha256"] = identity_sha256(call["messages"])
    with pytest.raises(ValueError, match="another reader policy"):
        evaluation.validate_matched_calls(preflight.payload["calls"])


def test_incomplete_population_cannot_open_gold(tmp_path, monkeypatch):
    preflight, _ = setup_runtime(tmp_path, monkeypatch)
    evaluation.run(tmp_path, 1)
    publish_sealed_json(tmp_path / "answers.json", {"preflight_sha256": preflight.sha256, "rows": []})
    from tools import run_hot_reduced30_answer_judge as helpers
    monkeypatch.setattr(helpers, "_load_locked_validation_question_population", lambda *args: pytest.fail("gold opened"))
    with pytest.raises(ValueError, match="complete sealed answer population"):
        evaluation.judge(tmp_path, False)


def test_memory_api_pairs_are_adjacent_and_counterbalanced_with_one_short_control():
    orders = [evaluation.call_arm_order(i) for i in range(10)]
    for order in orders:
        assert len(order) == len(set(order)) == 5 and set(order) == set(evaluation.ARMS)
    for arm in evaluation.MEMORY_ARMS:
        assert sum(order.index(arm) < order.index(arm + "_api") for order in orders) == 5
        assert all(abs(order.index(arm) - order.index(arm + "_api")) == 1 for order in orders)


