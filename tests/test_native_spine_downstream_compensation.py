from types import SimpleNamespace

import pytest

from memory_condense.application.threaded_section_context import TranscriptOrder
from memory_condense.application.user_evidence_projection import render_user_spine_sections
from memory_condense.eval.spine_reader_policy_v7 import SPINE_READER_SYSTEM_PROMPT_V7
from tests.test_threaded_section_context import fixture
from tools.assess_native_spine_downstream_compensation import current, neutral_messages
from tools.evaluate_native_spine_downstream_compensation import token_statistics


def test_neutral_presentation_restores_all_roles_in_order_without_changing_reader():
    turns, hydrated = fixture()
    order = TranscriptOrder(turns)
    old = render_user_spine_sections(hydrated, order)
    question = {'retrieval_query': 'What was said?', 'prompt_question': 'What was said?'}
    messages = current.reader.apply_reader(current.context_policy.messages(question,
        SimpleNamespace(render_context=lambda: old.text)))
    saved = {'hydration': hydrated.identity_payload(), 'rendered': old.identity_payload(),
             'question': question, 'messages': messages}
    result, rendered = neutral_messages(saved, order, SPINE_READER_SYSTEM_PROMPT_V7)
    assert turns[1].text not in old.text
    assert all(t.text in rendered['text'] for t in turns)
    assert rendered['text'].index(turns[0].text) < rendered['text'].index(turns[1].text) < rendered['text'].index(turns[2].text)
    assert result[0] == messages[0]
    assert rendered['token_count'] <= hydrated.max_context_tokens
    assert len(rendered['placements']) == sum(len(s.evidence) for s in hydrated.sections)
    with pytest.raises(ValueError, match='reader or question'):
        neutral_messages(saved, order, 'Changed reader instructions')


def test_missing_provider_usage_remains_missing_and_proxy_is_separate():
    rows = [{'reported_prompt_tokens': None, 'prompt_token_proxy': 200, 'projected_prompt_token_proxy': 100},
            {'reported_prompt_tokens': 280, 'prompt_token_proxy': 300, 'projected_prompt_token_proxy': 150}]
    result = token_statistics(rows)
    assert result['mean_reported_prompt_tokens'] is None
    assert result['reported_prompt_tokens_available'] == 1
    assert result['mean_prompt_token_proxy'] == 250
    assert result['mean_projected_prompt_token_proxy'] == 125
