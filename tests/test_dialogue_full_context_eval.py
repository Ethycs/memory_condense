from types import SimpleNamespace

from tools.evaluate_dialogue_full_context import render_full_context
from tools.evaluate_dialogue_full_context_batch import batch_messages, parse_answers
import pytest


def test_full_control_preserves_all_roles_order_and_raw_characters():
    turns = [SimpleNamespace(role='user', text='Earlier question\n漢字'),
             SimpleNamespace(role='system', text='tool dump\n<|endoftext|>\n'*10000),
             SimpleNamespace(role='assistant', text='Later correction')]
    rendered = render_full_context(turns)
    expected = '\n\n'.join(
        f'[T{i} role={t.role} chars=0:{len(t.text)}]\n'+t.text
        for i,t in enumerate(turns))
    assert rendered == expected
    assert all(t.text in rendered for t in turns)
    assert rendered.index('Earlier question') < rendered.index('tool dump') < rendered.index('Later correction')


def test_batch_contains_one_archive_and_every_original_question_without_reference():
    questions=[dict(id=f'Q{i:02}',question=f'Question {i}?') for i in range(1,16)]
    messages=batch_messages(dict(reader_system='Historical evidence only.'),dict(text='FULL_ARCHIVE'),questions)
    assert messages[1]['content'].count('FULL_ARCHIVE')==1
    assert all(q['question'] in messages[1]['content'] for q in questions)
    assert '450 words per answer' in messages[0]['content']
    with pytest.raises(ValueError,match='omitted or duplicated'):
        parse_answers('{"answers":[{"id":"Q01","content":"one only"}]}',questions)
