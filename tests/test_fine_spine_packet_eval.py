from copy import deepcopy

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from tools import evaluate_fine_spine_packets as evaluation


def population():
    cases = []
    for ordinal in range(100):
        messages = [{'role': 'system', 'content': evaluation.previous.source.QA_SYSTEM_PROMPT},
                    {'role': 'user', 'content': f'Exact evidence and question {ordinal}.'}]
        cases.append({'question': {'ordinal': ordinal, 'question_id': f'question-{ordinal}'},
            'messages': {arm: deepcopy(messages) for arm in evaluation.ARMS},
            'messages_sha256': {arm: identity_sha256(messages) for arm in evaluation.ARMS}})
    return cases


def test_all_three_full100_prediction_populations_have_exact_question_and_arm_bindings():
    rows = evaluation.reader_rows(population(), [f'prediction-{i}' for i in range(300)])
    for arm in evaluation.ARMS:
        assert [r['ordinal'] for r in rows if r['arm'] == arm] == list(range(100))
    assert rows[5]['prediction'] == 'prediction-5'
    assert rows[5]['question_id'] == 'question-1'
    assert rows[5]['arm'] == 'fine_spine_compact'


@pytest.mark.parametrize('corruption', ['question', 'prompt', 'policy', 'arm', 'partial'])
def test_invalid_reader_population_is_rejected(corruption):
    cases = population()
    if corruption == 'question':
        cases[-1]['question']['ordinal'] = 98
    elif corruption == 'prompt':
        cases[0]['messages']['fine_spine'][1]['content'] = 'changed'
    elif corruption == 'policy':
        cases[0]['messages']['fine_spine'][0]['content'] = 'changed reader'
        cases[0]['messages_sha256']['fine_spine'] = identity_sha256(cases[0]['messages']['fine_spine'])
    elif corruption == 'arm':
        cases[0]['messages'].pop('relative_reservation')
    else:
        cases.pop()
    with pytest.raises(ValueError):
        evaluation.validate_cases(cases)


def test_two_complete_arms_cannot_substitute_for_the_required_third_arm():
    with pytest.raises(ValueError, match='all300'):
        evaluation.reader_rows(population(), ['answer'] * 200)


def test_failed_answer_replay_prevents_reference_loading(monkeypatch, tmp_path):
    def incomplete(*args):
        raise ValueError('incomplete answers')
    def forbidden():
        pytest.fail('reference data opened before all reader answers')
    monkeypatch.setattr(evaluation, 'answers', incomplete)
    monkeypatch.setattr(evaluation, 'load_references', forbidden)
    with pytest.raises(ValueError, match='incomplete'):
        evaluation.judge(tmp_path)
