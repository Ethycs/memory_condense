import pytest

from tools import paired_engineering_quality as pair


def test_full_context_retains_old_instruction_and_exact_complete_tool_pairs():
    rows = [{'role': 'user', 'kind': 'seed', 'text': 'Per turn, never wall time.'},
            {'role': 'assistant', 'kind': 'action', 'text': '{"action":"read","path":"x.py"}'},
            {'role': 'system', 'kind': 'tool', 'text': 'first\nsecond\nthird'},
            {'role': 'system', 'kind': 'activity', 'text': 'internal duplicate receipt'},
            {'role': 'user', 'kind': 'prompt', 'text': 'Go'}]
    messages = pair.full_messages(rows)
    assert messages == [{'role': 'system', 'content': pair.SYSTEM},
                        {'role': 'user', 'content': 'Per turn, never wall time.'},
                        {'role': 'assistant', 'content': rows[1]['text']},
                        {'role': 'user', 'content': 'Tool result:\nfirst\nsecond\nthird'},
                        {'role': 'user', 'content': 'Go'}]


def test_both_context_policies_use_identical_tool_execution_and_no_hidden_test_access(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'x.py').write_text('\n'.join('value = ' + str(i) for i in range(260)))
    result = pair.execute(tmp_path, {'action': 'read', 'path': 'x.py', 'line_count': 999},
                          tmp_path / 'action', None, 'u', {})
    assert '240: value = 239' in result and '241: value = 240' not in result
    result = pair.execute(tmp_path, {'action': 'find', 'path': 'x.py', 'text': 'value'},
                          tmp_path / 'action', None, 'u', {})
    assert len(result.splitlines()) == 260 and 'value = 259' in result
    with pytest.raises(ValueError):
        pair.execute(tmp_path, {'action': 'read', 'path': '../acceptance/test_paired_behavior.py'},
                     tmp_path / 'action', None, 'u', {})
