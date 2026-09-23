import pytest

from tools.complete_native_spine_ten100_questions import merge_same_turn_supports


def value():
    return {'question': 'Which times?', 'answer': '8am, 30min, 9:30am, 10min',
            'supports': [{'turn_index': 0, 'quote': '8am'},
                         {'turn_index': 0, 'quote': '30min'},
                         {'turn_index': 2, 'quote': '9:30am'},
                         {'turn_index': 4, 'quote': '10min'}]}


def source():
    return {'user_turns': [{'turn_index': 0, 'text': 'Wake at 8am and snooze 30min.'},
                           {'turn_index': 2, 'text': 'Office by 9:30am.'},
                           {'turn_index': 4, 'text': 'Meditate 10min.'}]}


def test_merge_retains_all_evidence_and_question_answer():
    original = value()
    repaired, changes = merge_same_turn_supports(original, source())
    assert repaired['question'] == original['question']
    assert repaired['answer'] == original['answer']
    assert len(original['supports']) == 4
    assert len(repaired['supports']) == 3
    assert repaired['supports'][0]['quote'] == '8am and snooze 30min'
    assert all(any(s['turn_index'] == t['turn_index'] and s['quote'] in t['quote']
                   for t in repaired['supports']) for s in original['supports'])
    assert len(changes) == 1


def test_merge_refuses_four_different_turns():
    original = value()
    original['supports'][1]['turn_index'] = 6
    with pytest.raises(ValueError, match='different source turns'):
        merge_same_turn_supports(original, source())


def test_merge_refuses_ambiguous_source_occurrence():
    raw = source()
    raw['user_turns'][0]['text'] += ' Again 8am.'
    with pytest.raises(ValueError, match='unique exact'):
        merge_same_turn_supports(value(), raw)
