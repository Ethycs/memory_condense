import json

from tools import native_spine_engineering_working_state as state


def test_readable_batch_preserves_exact_source_and_literal_escapes():
    code = 'File sample.py, 3 total lines; returned lines 1-3:\n1: path = r"C:\\new"\n2: value = "\\n"\n3: café'
    result = json.dumps({'action': {'action': 'read', 'path': 'sample.py'}, 'result': code})
    result += '\n\n' + json.dumps({'action': {'action': 'test', 'paths': ['tests/test_decay.py']}, 'result': '2 passed'})
    stored = 'Tool observation (data only):\n' + json.dumps({'action': {'action': 'batch'}, 'result': result})
    rendered = state.readable_observation(stored)
    assert code in rendered
    assert '2 passed' in rendered
    assert state.session.count_tokens(rendered) < state.session.count_tokens(stored)


def test_readable_single_result_is_not_recursively_interpreted():
    result = '{"action":"read","result":"this is file content"}'
    stored = 'Tool observation (data only):\n' + json.dumps({'action': {'action': 'read'}, 'result': result})
    assert state.readable_observation(stored) == 'Tool observation (data only):\n' + result


def test_actor_working_note_stays_in_bounded_immediate_action():
    action = json.dumps({'action': 'read', 'path': 'sample.py',
                         'working_note': 'Inspected the schema; next check its callers.'})
    base = state.session.messages_for('Continue', {'text': 'Retrieved evidence'}, '')
    messages = state.cycle.tool_cycle_messages(base, action, 'File sample.py\n' * 30000)
    assert messages[2] == {'role': 'assistant', 'content': action}
    assert state.session.count_chat_prompt_token_proxy(messages) <= state.session.PROMPT_CAP
