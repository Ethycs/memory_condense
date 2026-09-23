import pytest

from tools.complete_native_spine_five100_questions import exact_supports


def case(quote, text):
    value = {'question': 'What did I order?', 'answer': 'A lamp', 'category': 'fact',
             'supports': [{'turn_index': 2, 'quote': quote}]}
    source = {'user_turns': [{'turn_index': 2, 'text': text}]}
    return value, source


def test_case_correction_preserves_question_answer_and_exact_source_slice():
    value, source = case('Ordered a lamp', 'Yesterday I ordered a lamp.')
    corrected, changes = exact_supports(value, source)
    assert corrected['question'] == value['question']
    assert corrected['answer'] == value['answer']
    assert corrected['supports'][0]['quote'] == 'ordered a lamp'
    change, = changes
    assert source['user_turns'][0]['text'][change['start_char']:change['end_char']] == change['exact_quote']
    assert value['supports'][0]['quote'] == 'Ordered a lamp'


@pytest.mark.parametrize('quote,text', [
    ('Ordered a lamp', 'I ordered a desk.'),
    ('Ordered a lamp', 'I ordered a lamp; then ordered a lamp again.'),
    ('I ordered a lamp', 'I did not order a lamp.'),
])
def test_correction_refuses_changed_facts_or_ambiguous_matches(quote, text):
    with pytest.raises(ValueError):
        exact_supports(*case(quote, text))
