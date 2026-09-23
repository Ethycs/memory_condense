import json

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest
from tools.build_user_spine_pilot_hierarchy_v4 import bounded_attached_prefix, OMISSION
from tools.build_user_spine_pilot_hierarchy_v3 import MergeJournal


def request(kind='attached_context', cap=30):
    return SpineSummaryRequest(kind, (SpineSummaryFragment('user' if kind == 'user_spine' else 'assistant',
        '2026-09-09', 'Existing summary.'),), max_output_tokens=cap)


def test_overflow_retains_complete_attributed_negation_and_marks_loss():
    summary = 'Assistant did not recommend camping. ' + 'Assistant suggested considering alternatives. ' * 12
    result = bounded_attached_prefix(json.dumps({'summary': summary}), request())
    assert result.startswith('Assistant did not recommend camping.')
    assert result.endswith(OMISSION)
    assert summary.startswith(result.removesuffix(OMISSION))
    assert count_tokens(result) <= 30


def test_user_spine_must_not_be_cut():
    with pytest.raises(ValueError, match='forbidden'):
        bounded_attached_prefix(json.dumps({'summary': 'User did not go. ' * 30}), request('user_spine'))


def test_single_oversized_sentence_fails_without_partial_claim():
    with pytest.raises(ValueError, match='no complete'):
        bounded_attached_prefix(json.dumps({'summary': 'Assistant ' + 'possibly ' * 70 + 'recommended it.'}), request())


def test_valid_text_reused_and_malformed_json_not_repaired():
    assert bounded_attached_prefix('{"summary":"Assistant declined."}', request()) == 'Assistant declined.'
    with pytest.raises(ValueError, match='JSON'):
        bounded_attached_prefix('Assistant declined.', request())


def test_singleton_reuse_never_invokes_gateway():
    journal = object.__new__(MergeJournal)
    journal.reused = 0
    journal.complete_messages = lambda *args, **kwargs: pytest.fail('unnecessary provider call')
    assert journal.summarize(request()) == 'Existing summary.'
    assert journal.reused == 1
