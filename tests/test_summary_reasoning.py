from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from memory_condense.application.retrieval_workflow import RetrievalWorkflowMixin
from memory_condense.domain.schemas import Turn
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.summary_reasoning import (
    QwenSummaryReasoner, SummaryChoiceRequest, parse_summary_choice, reason_over_summary_hierarchy,
)


class Reasoner:
    model = "qwen3-8b"
    gateway_url = "https://local.test/v1"

    def __init__(self, response=None):
        self.requests = []
        self.response = response

    def complete(self, request):
        self.requests.append(request)
        if self.response is not None:
            return self.response
        labels = [i for i, summary in enumerate(request.summaries) if "target" in summary]
        return json.dumps({"selected_labels": labels[:request.max_choices]})


def section(i, summary, source="source"):
    turn = Turn(turn_id=f"turn-{i}", source_id=source, role="user",
                text=f" RAW_SECRET_{i} τ\r\n", created_at=datetime(2026, 9, 8, tzinfo=timezone.utc))
    return turn, SectionSummary(f"section-{i}", source, summary, (RawSectionSpan.from_turn(turn),), "test")


def tree():
    first, a = section(0, "unrelated garden")
    second, b = section(1, "target astronomy")
    parent = SectionSummary("parent", "source", "target leisure topics", a.spans + b.spans, "test",
                            child_section_ids=(a.section_id, b.section_id))
    return [first, second], SectionSummaryIndex([a, b, parent])


def facade(records, reads):
    app = RetrievalWorkflowMixin()
    def load(identity):
        reads.append(identity)
        return records[identity]
    app._transcript = SimpleNamespace(get_turn=load)
    return app


def test_full_decoder_routes_summaries_then_hydrates_exact_selected_leaf():
    turns, index = tree()
    reasoner, reads = Reasoner(), []
    result = facade({t.turn_id:t for t in turns}, reads).search_reasoned_summary_sections(
        "target question", index, reasoner=reasoner, max_sections=1)
    assert reads == ["turn-1"]
    assert result.sections[0].evidence[0].text == turns[1].text
    assert len(reasoner.requests) == 2
    assert "RAW_SECRET" not in repr([r.messages for r in reasoner.requests])
    assert "section-" not in repr([r.messages for r in reasoner.requests])
    assert "target astronomy" not in result.render_context()
    assert result.plan.routing_backend == "qwen_summary_reasoning"
    assert result.plan.attention_receipt is None
    assert result.plan.reasoning_receipt.raw_content_inspections == 0
    assert not result.plan.frontier_closed


@pytest.mark.parametrize("response", [
    "not json", "[]", '{"selected_labels":[true]}', '{"selected_labels":[0.0]}',
    '{"selected_labels":[-1]}', '{"selected_labels":[2]}', '{"selected_labels":[0,0]}',
    '{"selected_labels":[0,1]}', '{"selected_labels":[0],"answer":"invented"}',
])
def test_invalid_or_overfull_model_response_cannot_trigger_raw_hydration(response):
    turns, index = tree()
    reasoner, reads = Reasoner(response), []
    with pytest.raises(ValueError, match="Qwen summary choice"):
        facade({t.turn_id:t for t in turns}, reads).search_reasoned_summary_sections(
            "target", index, reasoner=reasoner, max_sections=1)
    assert reads == []


def test_empty_selection_and_exact_scope_make_no_raw_claim():
    turns, index = tree()
    reasoner, reads = Reasoner('{"selected_labels":[]}'), []
    app = facade({t.turn_id:t for t in turns}, reads)
    result = app.search_reasoned_summary_sections("question", index, reasoner=reasoner)
    assert result.requires_raw_fallback and not result.sections and not reads
    reasoner.requests.clear()
    scoped = app.search_reasoned_summary_sections("question", index, reasoner=reasoner,
                                                 eligible_source_ids=["source-prefix"])
    assert not scoped.sections and reasoner.requests == []


def test_large_frontier_reduces_bounded_groups_without_losing_last_group():
    records = [section(f"{i:02}", "target" if i == 19 else "unrelated") for i in range(20)]
    reasoner = Reasoner()
    plan = reason_over_summary_hierarchy("target", SectionSummaryIndex([s for _, s in records]),
                                         reasoner=reasoner, max_sections=1, group_size=4, max_calls=6)
    assert [r.section.section_id for r in plan.routes] == ["section-19"]
    assert len(reasoner.requests) == 6
    assert all(len(request.summaries) <= 4 for request in reasoner.requests)


def test_depth_cap_returns_whole_parent_but_call_cap_fails_before_raw_read():
    turns, index = tree()
    reasoner, reads = Reasoner(), []
    app = facade({t.turn_id:t for t in turns}, reads)
    parent = app.search_reasoned_summary_sections("target", index, reasoner=reasoner,
                                                 max_sections=1, max_depth=1)
    assert parent.plan.routes[0].section.section_id == "parent"
    assert reads == ["turn-0", "turn-1"]
    reads.clear()
    with pytest.raises(ValueError, match="call budget"):
        app.search_reasoned_summary_sections("target", index, reasoner=reasoner,
                                             max_sections=1, max_calls=1)
    assert not reads


def test_prompt_budget_rejects_before_provider_and_does_not_truncate():
    turns, index = tree()
    reasoner = Reasoner()
    with pytest.raises(ValueError, match="token budget"):
        reason_over_summary_hierarchy("target " * 500, index, reasoner=reasoner, max_prompt_tokens=150)
    assert reasoner.requests == []
    request = SummaryChoiceRequest("literal question", ("first summary", "second summary"), 2)
    body = json.loads(request.messages[1]["content"])
    assert body["summaries"] == [{"label":0,"summary":"first summary"}, {"label":1,"summary":"second summary"}]


def test_local_gateway_adapter_disables_retries_and_rejects_truncated_output():
    class Client:
        base_url = "https://local.test/v1"
        finish = "stop"
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def with_options(self, **kwargs):
            assert kwargs == {"max_retries":0}
            return self
        def create(self, **kwargs):
            assert kwargs["model"] == "qwen3-8b"
            assert kwargs["extra_body"] == {"enable_thinking":False}
            assert kwargs["temperature"] == 0
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason=self.finish,
                message=SimpleNamespace(content='{"selected_labels":[0]}'))])
    client = Client()
    reasoner = QwenSummaryReasoner(client)
    request = SummaryChoiceRequest("query", ("summary",), 1)
    assert parse_summary_choice(reasoner.complete(request), candidate_count=1, max_choices=1) == (0,)
    client.finish = "length"
    with pytest.raises(ValueError, match="finish normally"):
        reasoner.complete(request)
    with pytest.raises(ValueError, match="Qwen model"):
        QwenSummaryReasoner(client, model="non-Q-model")
