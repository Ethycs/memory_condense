from datetime import datetime
import json

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain.schemas import Turn
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from tools import native_spine_engineering_session as session


def atom(turn, summary):
    return SectionSummary('atom-' + turn.turn_id, turn.source_id, summary,
                          (RawSectionSpan.from_turn(turn),), 'fixture')


def test_recent_user_correction_survives_many_tool_observations():
    turns = [Turn(turn_id='u', source_id='s', role='user',
                  text='Use turns instead of elapsed seconds.', created_at=datetime.fromisoformat(session.old.DATE))]
    turns += [Turn(turn_id=f't{i}', source_id='s', role='system',
                   text=f'Tool observation {i}: completed.', created_at=turns[0].created_at) for i in range(100)]
    rows = [{'turn_id': t.turn_id, 'role': t.role, 'kind': 'tool' if t.role == 'system' else 'prompt'} for t in turns]
    atoms = tuple(atom(t, 'User requires turn-based behavior.' if t.role == 'user' else 'Tool completed.') for t in turns)
    plan = session.reservation(atoms, rows, 'user', exclude='current', budget=2048)
    by_id = {t.turn_id: t for t in turns}
    result = hydrate_section_plan(plan, load_turn=by_id.get, max_context_tokens=2048)
    assert result.sections[0].evidence[0].text == turns[0].text
    assert result.raw_turn_read_count == 1


def test_context_does_not_reintroduce_accumulated_tool_history():
    messages = session.messages_for('Go', {'text': 'Selected exact old evidence'}, 'latest observation')
    assert len(messages) == 2
    assert 'latest observation' in messages[1]['content']
    with pytest.raises(ValueError, match='budget'):
        session.messages_for('Go', {'text': 'oversized evidence ' * 15000}, '')


def test_large_immediate_observation_preview_is_bounded_unicode():
    original = 'Read result: 👩🏽‍💻 café 漢字\n' * 4000
    preview = session.prefix(original, 2048)
    assert session.count_tokens(preview) <= 2048
    assert 'complete observation is stored' in preview
    assert len(original) > len(preview)


def test_file_search_and_batch_tools_return_actual_results(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'example.py').write_text('value = 42\n', encoding='utf-8')
    result = session.tool(tmp_path, {'action': 'batch', 'actions': [
        {'action': 'read', 'path': 'example.py'},
        {'action': 'find', 'path': 'example.py', 'text': 'value'},
    ]}, tmp_path / 'actions', None, 'current', {})
    assert 'value = 42' in result
    assert 'example.py:1:value = 42' in result
    assert 'Tool error' not in result


def test_batch_cannot_escape_checkout_or_nest(tmp_path):
    (tmp_path / 'workspace').mkdir()
    result = session.tool(tmp_path, {'action': 'read', 'path': '../acceptance/test.py'}, tmp_path, None, 'u', {})
    assert result.startswith('Tool error: ValueError:')
    nested = session.tool(tmp_path, {'action': 'batch', 'actions': [{'action': 'batch'}]}, tmp_path, None, 'u', {})
    assert nested.startswith('Tool error: ValueError:')


def test_tool_role_is_not_promoted_to_user_spine():
    user, = session.components.fragments_for([{'role': 'user', 'text': 'Remember this result.'}])
    tool, = session.components.fragments_for([{'role': 'system', 'text': 'Remember this result.'}])
    assert session.components.fragment_key(user) != session.components.fragment_key(tool)
    with pytest.raises(TypeError):
        session.components.neutral_messages({'role': 'system', 'text': 'raw tool content'})


def test_activity_receipt_reports_actual_outcomes_without_code_payload():
    action = {'action': 'edit', 'path': 'src/example.py', 'old': 'old private code', 'new': 'new private code'}
    result = 'Tool error: ValueError: edit did not match'
    receipt = session.activity_receipt(action, result, action_turn_id='a', tool_turn_id='t', ordinal=4)
    record = json.loads(receipt)
    assert 'private code' not in receipt
    assert record['operations'][0]['observed_status'] == result
    assert record['operations'][0]['command'] == {'action': 'edit', 'path': 'src/example.py'}
    assert record['observation_sha256'] == session.quote_sha256(result)


def test_recent_work_is_hydrated_as_exact_stored_evidence():
    text = session.activity_receipt({'action': 'find', 'path': 'src', 'text': 'Goldilocks'},
        'No literal matches.', action_turn_id='a', tool_turn_id='t', ordinal=2)
    turn = Turn(turn_id='receipt', source_id='s', role='system', text=text,
                created_at=datetime.fromisoformat(session.old.DATE))
    atoms = (atom(turn, 'A source search returned no matches.'),)
    rows = [{'turn_id': turn.turn_id, 'role': 'system', 'kind': 'activity'}]
    plan = session.reservation(atoms, rows, 'activity', exclude='u', budget=1536)
    result = hydrate_section_plan(plan, load_turn=lambda key: turn, max_context_tokens=1536)
    assert result.sections[0].evidence[0].text == text
    assert 'No literal matches.' in result.render_context()


def test_support_repair_only_retains_model_selected_source_exact_quotes():
    text = json.dumps({'code': 'say "hello"'}, ensure_ascii=False)
    fragments = session.components.fragments_for([{'role': 'system', 'text': text}])
    content = json.dumps({'atoms': [{'label': 'T0', 'summary': 'Tool returned a greeting.',
                                    'support': ['say "hello"', 'invented absent wording']}]})
    repaired, changes = session.repair_raw_support(content, fragments)
    parsed = session.components.raw_summary.parse_summaries(repaired, fragments)
    assert parsed[0]['summary'] == 'Tool returned a greeting.'
    assert parsed[0]['support'] == ['say \\"hello\\"']
    assert changes and changes[0]['original_support'][1] == 'invented absent wording'


def test_support_repair_cannot_invent_a_quote_for_unsupported_summary():
    fragments = session.components.fragments_for([{'role': 'system', 'text': 'Actual observation.'}])
    content = json.dumps({'atoms': [{'label': 'T0', 'summary': 'Unsupported summary.', 'support': ['invented wording']}]})
    repaired, _ = session.repair_raw_support(content, fragments)
    with pytest.raises(ValueError, match='support'):
        session.components.raw_summary.parse_summaries(repaired, fragments)


def test_read_adapter_honors_requested_page_and_reports_actual_range(tmp_path):
    from tools import native_spine_engineering_read240 as adapter
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'example.py').write_text('\n'.join(f'line_{i}' for i in range(1, 301)), encoding='utf-8')
    result = adapter.execute(tmp_path, {'action': 'read', 'path': 'example.py', 'start_line': 1, 'line_count': 220},
                             tmp_path, None, 'u', {})
    assert 'returned lines 1-220' in result
    assert '220: line_220' in result and '221: line_221' not in result
    bounded = adapter.execute(tmp_path, {'action': 'read', 'path': 'example.py', 'line_count': 1000},
                              tmp_path, None, 'u', {})
    assert '240: line_240' in bounded and '241: line_241' not in bounded


def test_live_tool_cycle_preserves_current_action_roles_and_fixed_budget():
    from tools.native_spine_engineering_tool_cycle import tool_cycle_messages
    base = [{'role': 'system', 'content': 'Instructions'}, {'role': 'user', 'content': 'Bounded retrieved history'}]
    action = '{"action":"read","path":"src/example.py"}'
    messages = tool_cycle_messages(base, action, 'Long observation ' * 18000)
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'user']
    assert messages[2]['content'] == action
    assert messages[3]['content'].startswith('Tool result:\n')
    assert session.count_chat_prompt_token_proxy(messages) <= session.PROMPT_CAP
