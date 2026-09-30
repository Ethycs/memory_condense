from copy import deepcopy

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval.spine_reader_policy_v7 import SPINE_READER_SYSTEM_PROMPT_V7
from tools import evaluate_native_spine_answer_model_cap8 as evaluation


def source():
    messages = [{'role': 'system', 'content': SPINE_READER_SYSTEM_PROMPT_V7},
                {'role': 'user', 'content': 'Exact evidence.\nQuestion: What did I choose?'}]
    return {'messages': messages, 'measurement': {'messages_sha256': identity_sha256(messages)}}


def test_model_comparison_preserves_every_prompt_byte_and_does_not_mutate_source():
    packet = source()
    original = deepcopy(packet)
    messages = evaluation.changed_messages(packet)
    assert messages == packet['messages']
    messages[-1]['content'] = 'changed'
    assert packet == original


def test_rejects_a_modified_baseline_request_or_reader():
    packet = source()
    packet['messages'][-1]['content'] += ' Extra content.'
    with pytest.raises(ValueError, match='actual answer request'):
        evaluation.changed_messages(packet)
    packet = source()
    packet['messages'][0]['content'] = 'Different reader'
    packet['measurement']['messages_sha256'] = identity_sha256(packet['messages'])
    with pytest.raises(ValueError, match='unchanged v7'):
        evaluation.changed_messages(packet)


@pytest.mark.parametrize('defect', ['baseline_model', 'missing_model', 'different_gateway'])
def test_model_cannot_silently_fall_back_or_leave_authorized_gateway(defect):
    model = evaluation.MODEL
    inventory = {'model_ids': [model, evaluation.previous.MODEL], 'gateway': evaluation.frozen.GATEWAY}
    evaluation.validate_model(model, inventory)
    if defect == 'baseline_model':
        model = evaluation.previous.MODEL
    elif defect == 'missing_model':
        inventory['model_ids'].remove(model)
    else:
        inventory['gateway'] = 'https://example.invalid/v1'
    with pytest.raises(ValueError, match='alternative answer model'):
        evaluation.validate_model(model, inventory)
