from copy import deepcopy

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from tools import evaluate_spine_relative_reservation as evaluation


def population():
    cases = []
    for ordinal in range(100):
        messages = [{'role': 'system', 'content': evaluation.source.QA_SYSTEM_PROMPT},
                    {'role': 'user', 'content': f'Evidence and question {ordinal}'}]
        cases.append({'question': {'ordinal': ordinal},
            'messages': {arm: deepcopy(messages) for arm in evaluation.ARMS},
            'messages_sha256': {arm: identity_sha256(messages) for arm in evaluation.ARMS}})
    return cases


def test_full_population_includes_unchanged_prompts_for_both_arms():
    evaluation.validate_cases(population())


@pytest.mark.parametrize('corruption', ['partial', 'duplicate', 'one_arm', 'prompt', 'policy'])
def test_partial_or_changed_reader_population_is_rejected(corruption):
    cases = population()
    if corruption == 'partial':
        cases.pop()
    elif corruption == 'duplicate':
        cases[-1]['question']['ordinal'] = 98
    elif corruption == 'one_arm':
        cases[-1]['messages'].pop('as_of')
    elif corruption == 'prompt':
        cases[-1]['messages']['relative_reservation'][1]['content'] = 'replacement'
    else:
        cases[-1]['messages']['as_of'][0]['content'] = 'replacement reader'
        cases[-1]['messages_sha256']['as_of'] = identity_sha256(cases[-1]['messages']['as_of'])
    with pytest.raises(ValueError):
        evaluation.validate_cases(cases)


def test_incomplete_answers_prevent_reference_loading(monkeypatch, tmp_path):
    def incomplete(*args):
        raise ValueError('reader population incomplete')
    def forbidden():
        pytest.fail('references loaded before complete answers')
    monkeypatch.setattr(evaluation, 'answers', incomplete)
    monkeypatch.setattr(evaluation, 'load_references', forbidden)
    with pytest.raises(ValueError, match='incomplete'):
        evaluation.judge(tmp_path)
