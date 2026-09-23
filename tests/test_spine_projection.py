import json
from dataclasses import replace

import pytest

from memory_condense.search.section_routing import SectionSummaryIndex
from tests.test_summary_reasoning import section
from tools.evaluate_user_spine_projection import project_user_channel


def test_user_projection_preserves_negation_raw_locators_and_empty_prelude():
    _, original = section(0, json.dumps({'user_spine': 'User did not purchase it.',
        'attached_context_not_user_assertions': 'ASSISTANT_CANARY recommended purchasing it.',
        'transcript_date_range': ['2026-09-09', '2026-09-09']}))
    before = SectionSummaryIndex([original])
    after = project_user_channel(before)
    assert 'ASSISTANT_CANARY' not in after.sections[0].summary
    assert 'User did not purchase it.' in after.sections[0].summary
    assert after.sections[0].spans == original.spans
    assert before.receipt_sha256 != after.receipt_sha256
    body = json.loads(original.summary)
    body['user_spine'] = None
    prelude = replace(original, summary=json.dumps(body), receipt_sha256='')
    projected = project_user_channel(SectionSummaryIndex([prelude]))
    assert json.loads(projected.sections[0].summary)['user_spine'] is None
    assert 'ASSISTANT_CANARY' not in projected.sections[0].summary


def test_untyped_summary_cannot_be_assumed_to_be_user_speech():
    _, original = section(0, 'An assistant recommendation.')
    with pytest.raises(ValueError):
        project_user_channel(SectionSummaryIndex([original]))
