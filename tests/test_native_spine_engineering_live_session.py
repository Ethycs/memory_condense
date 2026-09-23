from tools import native_spine_engineering_live_session as live


def history():
    rows = [{'turn_id': 'old-user', 'kind': 'prompt', 'step': 0, 'text': 'old'}]
    rows += [{'turn_id': 'old-action', 'kind': 'action', 'step': 0, 'text': 'old action'},
             {'turn_id': 'old-tool', 'kind': 'tool', 'step': 0, 'text': 'old observation'}]
    rows += [{'turn_id': 'user', 'kind': 'prompt', 'step': 1, 'text': 'new'}]
    for i in range(5):
        rows += [{'turn_id': f'a{i}', 'kind': 'action', 'step': 1, 'text': f'action{i}'},
                 {'turn_id': f't{i}', 'kind': 'tool', 'step': 1, 'text': 'result ' * 100},
                 {'turn_id': f'r{i}', 'kind': 'activity', 'step': 1, 'text': 'receipt'}]
    return rows


def test_bounded_working_history_keeps_complete_latest_pairs_and_no_previous_user_turn():
    rows = history()
    messages, selected = live.working_pairs(rows, budget=300)
    assert len(selected) == 2
    assert [p['action_turn_id'] for p in selected] == ['a3', 'a4']
    assert len(messages) == 4
    assert messages[2]['content'] == 'action4'
    assert all('old' not in m['content'] for m in messages)
    assert live.s.count_chat_prompt_token_proxy(messages) <= 300


def test_ingestion_is_required_before_unstored_work_is_evicted():
    rows = history()
    _, selected = live.working_pairs(rows, budget=300)
    assert live.needs_ingestion(rows, rows[:4], selected)
    assert not live.needs_ingestion(rows, rows[:13], selected)
    assert live.needs_ingestion(rows, rows[:3], selected)  # New user prompt.


def test_full_available_current_work_is_preserved_when_it_fits():
    messages, selected = live.working_pairs(history())
    assert len(selected) == 5
    assert len(messages) == 10
    assert messages[0]['content'] == 'action0'
