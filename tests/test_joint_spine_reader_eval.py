from types import SimpleNamespace
import sys

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from tools import evaluate_spine_reader as evaluation
from tools import evaluate_spine_reader_v2
from tools import evaluate_spine_reader_v3
from tools import evaluate_spine_combined_reader
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


@pytest.fixture(autouse=True, params=(evaluation, evaluate_spine_reader_v2, evaluate_spine_reader_v3,
                                   evaluate_spine_combined_reader))
def reader_version(request, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "evaluation", request.param)


def setup_runtime(tmp_path, monkeypatch):
    question = {"ordinal": 0, "question_id": "fixture", "retrieval_query": "Which adapter?", "prompt_question": "Which adapter?"}
    hydrated = SimpleNamespace(render_context=lambda: "EXACT_RAW_CANARY")
    calls = []
    for i, arm in enumerate(evaluation.ARMS):
        policy = "base" if arm.startswith("base") else "reader"
        messages = evaluation.answer_messages(question, None if arm.endswith("_short_api") else hydrated, policy=policy)
        calls.append({"call_index": i, "arm": arm, "question": question,
                      "messages": messages, "messages_sha256": identity_sha256(messages)})
    preflight, _ = publish_sealed_json(tmp_path / "preflight.json", {"calls": calls,
        "index_root": str(tmp_path), "index_manifest_sha256": "fixture", "addresses_root": str(tmp_path),
        "addresses_sha256": "fixture", "atoms_path": str(tmp_path), "atoms_sha256": "fixture",
        "facets_root": str(tmp_path), "facets_sha256": "fixture",
        "role_partition_sha256": "fixture"})
    monkeypatch.setattr(evaluation, "load_preflight", lambda _: preflight)
    events, prompts = [], []
    class Memory:
        encoder = SimpleNamespace(close=lambda: None)
        source_spine = SimpleNamespace(index=SimpleNamespace(receipt_sha256="fixture"))
        def __init__(self, *args):
            pass
        def retrieve(self, query, arm, dated_question):
            events.append(("retrieve", arm))
            return SimpleNamespace(render_context=lambda: "EXACT_RAW_CANARY", diagnostics=(),
                                   identity_payload=lambda: {"exact_hydration": True})
    class Stream:
        def __iter__(self):
            yield {"choices": [{"delta": {"content": "Compatible adapter"}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        def close(self):
            pass
    def create(**kwargs):
        events.append(("api",))
        prompts.append(kwargs["messages"])
        return Stream()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)
    monkeypatch.setattr(evaluation, "ResidentMemory", Memory)
    monkeypatch.setattr(evaluation, "_completion_client", lambda *args: client)
    return preflight, events, prompts


def test_both_policies_retrieve_live_and_have_policy_matched_api_controls(tmp_path, monkeypatch):
    preflight, events, prompts = setup_runtime(tmp_path, monkeypatch)
    evaluation.run(tmp_path, len(evaluation.ARMS))
    expected_events = []
    for arm in evaluation.ARMS:
        if arm in evaluation.MEMORY_ARMS:
            expected_events.append("retrieve")
        expected_events.append("api")
    assert [e[0] for e in events] == expected_events
    route = "source_spine_facets" if evaluation is evaluate_spine_reader_v3 else "source_spine"
    routes = (list(evaluation.ROUTES.values()) if evaluation is evaluate_spine_combined_reader
              else [route, route])
    assert [e[1] for e in events if e[0] == "retrieve"] == routes
    assert "EXACT_RAW_CANARY" not in str(prompts[:2])
    assert prompts[2] == prompts[3] and prompts[4] == prompts[5]
    assert prompts[2][1:] == prompts[4][1:] and prompts[2][0] != prompts[4][0]
    assert prompts[0][0] == prompts[2][0] and prompts[1][0] == prompts[4][0]
    if evaluation is evaluate_spine_combined_reader:
        assert prompts[6] == prompts[7]
        assert prompts[6][0] == prompts[1][0]
    assert read_sealed_json(tmp_path / "answers.json").payload["rows"] == evaluation.answer_rows(evaluation.recorded(tmp_path, preflight))


def test_comparison_rejects_changed_evidence_or_wrong_short_policy(tmp_path, monkeypatch):
    preflight, _, _ = setup_runtime(tmp_path, monkeypatch)
    calls = preflight.payload["calls"]
    for call in calls:
        if call["arm"] in ("reader", "reader_api"):
            call["messages"][1]["content"] = "OTHER_EVIDENCE"
            call["messages_sha256"] = identity_sha256(call["messages"])
    with pytest.raises(ValueError, match="changed the evidence"):
        evaluation.validate_matched_calls(calls)
    short = next(c for c in calls if c["arm"] == "reader_short_api")
    short["messages"][0]["content"] = "OTHER_POLICY"
    short["messages_sha256"] = identity_sha256(short["messages"])
    with pytest.raises(ValueError, match="another reader policy"):
        evaluation.validate_matched_calls(calls)


def test_interrupted_answer_is_not_retried_and_incomplete_answers_cannot_open_gold(tmp_path, monkeypatch):
    preflight, events, _ = setup_runtime(tmp_path, monkeypatch)
    journal = tmp_path / "journal"
    journal.mkdir()
    (journal / "000.reserved").write_text("interrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="unacknowledged"):
        evaluation.run(tmp_path, 6)
    assert not events
    publish_sealed_json(tmp_path / "answers.json", {"preflight_sha256": preflight.sha256, "rows": []})
    from tools import run_hot_reduced30_answer_judge as helpers
    monkeypatch.setattr(helpers, "_load_locked_validation_question_population",
                        lambda *args: (_ for _ in ()).throw(AssertionError("gold opened")))
    with pytest.raises(ValueError, match="unacknowledged"):
        evaluation.judge(tmp_path, False)


def test_reader_pairs_are_adjacent_and_counterbalanced():
    orders = [evaluation.call_arm_order(i) for i in range(10)]
    for arm in evaluation.MEMORY_ARMS:
        assert sum(order.index(arm) < order.index(arm + "_api") for order in orders) == 5
        assert all(abs(order.index(arm) - order.index(arm + "_api")) == 1 for order in orders)
    assert all(set(order) == set(evaluation.ARMS) for order in orders)
