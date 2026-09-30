import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval.spine_reader_policy_v7 import SPINE_READER_SYSTEM_PROMPT_V7
from memory_condense.eval.spine_reader_policy_v9 import apply_reader
from tools.evaluate_native_spine_reader9 import changed_messages


def test_reader_changes_only_system_instructions_without_mutating_evidence():
    messages = [{'role': 'system', 'content': SPINE_READER_SYSTEM_PROMPT_V7},
                {'role': 'user', 'content': 'USER_STATEMENTS\nExact evidence.\nQuestion: Why?'}]
    source = {'messages': messages, 'measurement': {'messages_sha256': identity_sha256(messages)}}
    result = changed_messages(source)
    assert result[0] != messages[0]
    assert result[1:] == messages[1:]
    assert messages[0]['content'] == SPINE_READER_SYSTEM_PROMPT_V7
    result[1]['content'] = 'changed'
    assert source['messages'][1]['content'].startswith('USER_STATEMENTS')


def test_reader_rejects_changed_baseline_or_measurement():
    with pytest.raises(ValueError, match='unchanged v7'):
        apply_reader([{'role': 'system', 'content': 'wrong policy'}])
    with pytest.raises(ValueError, match='actual answer request'):
        changed_messages({'messages': [], 'measurement': {'messages_sha256': 'incorrect'}})
