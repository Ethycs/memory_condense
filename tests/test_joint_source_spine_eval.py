from types import SimpleNamespace
import sys

import pytest

from tools import evaluate_source_spine as evaluation
from tools import evaluate_source_spine_overflow
from tools import evaluate_source_spine_facets
from tools import evaluate_spine_term_coverage
from tools import evaluate_spine_source_coverage
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from memory_condense.domain._discourse_identity import identity_sha256


@pytest.fixture(autouse=True, params=(evaluation, evaluate_source_spine_overflow, evaluate_source_spine_facets, evaluate_spine_term_coverage, evaluate_spine_source_coverage))
def evaluation_version(request, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "evaluation", request.param)


def setup_runtime(tmp_path, monkeypatch):
    calls = [{"call_index": i, "arm": arm, "question": {"ordinal": 0, "question_id": "id",
             "retrieval_query": "Which bulb?", "prompt_question": "[Today] Which bulb?"}}
             for i, arm in enumerate(evaluation.ARMS)]
    for call in calls:
        hydrated = None if call["arm"] == "short_api" else SimpleNamespace(render_context=lambda: "EXACT_RAW_CANARY")
        call["messages"] = evaluation.answer_messages(call["question"], hydrated)
        call["messages_sha256"] = identity_sha256(call["messages"])
    preflight, _ = publish_sealed_json(tmp_path / "preflight.json", {"calls": calls, "index_root": str(tmp_path),
                                                                "index_manifest_sha256": "fixture", "addresses_root": str(tmp_path), "addresses_sha256": "fixture", "atoms_path": str(tmp_path), "atoms_sha256": "fixture", "role_partition_sha256": "fixture", "facets_root": str(tmp_path), "facets_sha256": "fixture"})
    monkeypatch.setattr(evaluation, "load_preflight", lambda root: preflight)
    events, prompts = [], []
    class Memory:
        encoder = SimpleNamespace(close=lambda: None)
        source_spine = SimpleNamespace(index=SimpleNamespace(receipt_sha256="fixture"))
        def __init__(self, *args):
            pass
        def retrieve(self, query, arm, dated_question):
            events.append(("retrieve", arm, query))
            return SimpleNamespace(render_context=lambda: "EXACT_RAW_CANARY", diagnostics=(),
                                   identity_payload=lambda: {"exact_hydration": True})
    class Stream:
        def __iter__(self):
            yield {"choices": [{"delta": {"content": "LED bulb"}, "finish_reason": None}]}
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


def test_streamed_predictions_are_scored_and_each_memory_query_retrieves_live(tmp_path, monkeypatch):
    preflight, events, prompts = setup_runtime(tmp_path, monkeypatch)
    evaluation.run(tmp_path, len(evaluation.ARMS))
    expected = [event for arm in evaluation.ARMS
                for event in (("retrieve", "api") if arm in evaluation.MEMORY_ARMS else ("api",))]
    assert [e[0] for e in events] == expected
    assert "EXACT_RAW_CANARY" not in str(prompts[0])
    assert all("EXACT_RAW_CANARY" in str(p) for p in prompts[1:])
    for arm in evaluation.MEMORY_ARMS:
        assert prompts[evaluation.ARMS.index(arm)] == prompts[evaluation.ARMS.index(arm + "_api")]
    answers = read_sealed_json(tmp_path / "answers.json")
    observations = evaluation.recorded(tmp_path, preflight)
    assert answers.payload["rows"] == evaluation.answer_rows(observations)
    assert all(row["prediction"] == "LED bulb" for row in answers.payload["rows"])
    assert all(r.payload["measurement"]["e2e_ttft_s"] >= r.payload["measurement"]["api_ttft_s"] for _, r in observations)


def test_unacknowledged_stream_is_never_retried(tmp_path, monkeypatch):
    _, events, _ = setup_runtime(tmp_path, monkeypatch)
    journal = tmp_path / "journal"
    journal.mkdir()
    (journal / "000.reserved").write_text("interrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="unacknowledged"):
        evaluation.run(tmp_path, 3)
    assert not events


def test_judge_refuses_incomplete_answers_before_opening_gold(tmp_path, monkeypatch):
    preflight, _, _ = setup_runtime(tmp_path, monkeypatch)
    publish_sealed_json(tmp_path / "answers.json", {"preflight_sha256": preflight.sha256, "rows": []})
    from tools import run_hot_reduced30_answer_judge as helpers
    monkeypatch.setattr(helpers, "_load_locked_validation_question_population",
                        lambda *args: (_ for _ in ()).throw(AssertionError("gold opened")))
    with pytest.raises(ValueError, match="complete sealed"):
        evaluation.judge(tmp_path, False)


def test_changed_control_prompt_cannot_be_reported_as_matched(tmp_path, monkeypatch):
    preflight, _, _ = setup_runtime(tmp_path, monkeypatch)
    calls = preflight.payload["calls"]
    control = next(c for c in calls if c["arm"] == evaluation.MEMORY_ARMS[0] + "_api")
    control["messages"][1]["content"] = "Different evidence"
    control["messages_sha256"] = identity_sha256(control["messages"])
    with pytest.raises(ValueError, match="prompts differ"):
        evaluation.validate_matched_calls(calls)


def test_changed_live_route_stops_before_sending_the_unmatched_answer(tmp_path, monkeypatch):
    _, events, _ = setup_runtime(tmp_path, monkeypatch)
    class ChangedMemory:
        encoder = SimpleNamespace(close=lambda: None)
        source_spine = SimpleNamespace(index=SimpleNamespace(receipt_sha256="fixture"))
        def __init__(self, *args):
            pass
        def retrieve(self, *args):
            return SimpleNamespace(render_context=lambda: "CHANGED_RAW_EVIDENCE")
    monkeypatch.setattr(evaluation, "ResidentMemory", ChangedMemory)
    with pytest.raises(ValueError, match="changed the frozen"):
        evaluation.run(tmp_path, 5)
    assert events == [("api",)]  # Only the earlier short API control ran.


def test_each_matched_pair_is_adjacent_and_counterbalanced():
    orders = [evaluation.call_arm_order(i) for i in range(10)]
    for arm in evaluation.MEMORY_ARMS:
        assert sum(order.index(arm) < order.index(arm + "_api") for order in orders) == 5
        assert all(abs(order.index(arm) - order.index(arm + "_api")) == 1 for order in orders)
    assert all(set(order) == set(evaluation.ARMS) for order in orders)
