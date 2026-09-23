from copy import deepcopy

import pytest

from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from memory_condense.application.threaded_section_context import TranscriptOrder
from memory_condense.domain._discourse_identity import quote_sha256
from tests.test_native_spine_context_routing import parent_fixture
from tests.test_native_spine_routing import QUERY, DATED
from tools import evaluate_native_spine_application_reader100 as evaluation
from tools import native_spine_context_policy as context


def instructions():
    return {'format': 'native-spine-reader-instructions-v1', 'name': 'v7',
        'baseline_system_sha256': quote_sha256(evaluation.reader.SPINE_READER_SYSTEM_PROMPT_V6),
        'system_prompt': evaluation.reader.SPINE_READER_SYSTEM_PROMPT_V7}


def test_reader_change_preserves_live_retrieval_rendering_and_full_question():
    history, semantic, hierarchy, encoder = parent_fixture(future=True)
    memory = ResidentNativeSpineContextMemory(semantic, hierarchy,
        encoder=encoder, load_turn=history.get_turn)
    question = {'retrieval_query': QUERY, 'prompt_question': DATED}
    order = TranscriptOrder(history.turns.values())
    original = evaluation.presentation.build(memory, question, context.DENSE_PARENT_2048, order)
    before = deepcopy(original)
    changed = evaluation.build(memory, question, context.DENSE_PARENT_2048, order, instructions())
    assert original == before
    assert changed[1:] == original[1:]
    assert changed[0][1:] == original[0][1:]
    assert changed[0][0] != original[0][0]
    assert encoder.calls == [QUERY, QUERY]
    assert '<C1' in changed[0][-1]['content']
    assert all(e['span']['created_at'] < '2026-09-13' for section in changed[1]['sections']
               for e in section['evidence'])


@pytest.mark.parametrize('defect', ['gold_field', 'source_policy', 'blank', 'unbounded'])
def test_policy_rejects_unsealed_extra_inputs_or_invalid_instructions(defect):
    policy = instructions()
    if defect == 'gold_field':
        policy['references'] = ['do not admit answer data']
    elif defect == 'source_policy':
        policy['baseline_system_sha256'] = '0' * 64
    elif defect == 'blank':
        policy['system_prompt'] = ' '
    else:
        policy['system_prompt'] = 'word ' * 3000
    with pytest.raises(ValueError, match='sealed reader instructions'):
        evaluation.validate_reader_policy(policy)


def test_reader_adapter_rejects_changed_baseline_system_message():
    with pytest.raises(ValueError, match='unchanged v6'):
        evaluation.reader.apply_reader([{'role': 'system', 'content': 'different reader'}])
