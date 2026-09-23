from copy import deepcopy

import pytest

from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from tests.test_native_spine_context_routing import parent_fixture
from tests.test_native_spine_routing import QUERY, DATED
from tests.test_frozen_native_spine_audit import packet
from tools import native_spine_context_policy as policy


def test_dense_parent_policy_uses_fresh_summary_query_and_excludes_future_raw():
    history, semantic, hierarchy, encoder = parent_fixture(future=True)
    memory = ResidentNativeSpineContextMemory(semantic, hierarchy, encoder=encoder, load_turn=history.get_turn)
    messages, hydration, routing = policy.build(memory,
        {'retrieval_query': QUERY, 'prompt_question': DATED}, policy.DENSE_PARENT_2048)
    assert encoder.calls == [QUERY]
    assert routing['ancestor_hops'] == 2
    assert routing['baseline']['routing_backend'] == 'summary_dense'
    assert routing['query_qwen_passes'] == routing['raw_reads_during_routing'] == 0
    assert hydration['max_context_tokens'] == 2048
    assert hydration['context_token_count'] <= 2048
    assert all(s['section']['source_id'] == history.atoms[0].source_id for s in hydration['sections'])
    assert 'I visited the botanical garden.' in messages[1]['content']
    assert 'I also enjoy watching birds.' in messages[1]['content']
    assert 'Stored attention parent summary.' not in messages[1]['content']


@pytest.mark.parametrize('damage', [{'ancestor_hops': 3}, {'max_context_tokens': 1000000},
                                   {'lexical_reserve': True}, {'question_id': 'target'}])
def test_unbounded_or_question_specific_policy_rejected_before_serving(damage):
    class Memory:
        def retrieve(self, *args, **kwargs):
            pytest.fail('invalid policy reached retrieval')
    with pytest.raises(ValueError, match='bounded, question-independent'):
        policy.build(Memory(), {'retrieval_query': QUERY, 'prompt_question': DATED},
                     {**policy.DENSE_PARENT_2048, **damage})


def test_policy_audit_reconstructs_source_and_rejects_changed_source_bytes():
    body, sessions, q, h, r = packet()
    limits = {**policy.DENSE_PARENT_2048, 'max_context_tokens': 1024, 'ancestor_hops': 1,
              'max_direct': h['plan']['max_sections'], 'context_seed_limit': 1}
    r.update(baseline=h['plan'], context_atomic_ids=[], consulted_chunk_ids=[])
    messages, count = policy.verify_packet(q, h, r, sessions, lambda _: body, limits)
    assert count > 0 and 'café' in messages[1]['content'] and '🌍' in messages[1]['content']
    altered = deepcopy(h)
    altered['sections'][0]['evidence'][0]['text'] += ' forged'
    with pytest.raises(ValueError, match='exact original raw'):
        policy.verify_packet(q, altered, r, sessions, lambda _: body, limits)
    with pytest.raises(ValueError, match='sealed context policy'):
        policy.verify_packet(q, h, r, sessions, lambda _: body, {**limits, 'max_context_tokens': 2048})
