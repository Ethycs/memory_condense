from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from tools import evaluate_threaded_spine_full100 as evaluation


def calls():
    result = []
    for ordinal in range(100):
        question = {'ordinal': ordinal, 'prompt_question': f'Question {ordinal}'}
        messages = {'threaded': evaluation.threaded_messages(question, 'Exact raw evidence.'),
                    'short_api': evaluation.threaded_messages(question)}
        messages['threaded_api'] = messages['threaded']
        messages['flat'] = [{'role': 'system', 'content': evaluation.previous.source.QA_SYSTEM_PROMPT},
                            {'role': 'user', 'content': f'Flat exact evidence and question {ordinal}'}]
        for arm in evaluation.call_order(ordinal):
            result.append({'call_index': len(result), 'question': question, 'arm': arm,
                          'messages': messages[arm], 'messages_sha256': identity_sha256(messages[arm])})
    return result


def observations(correct=95, candidate_latency=5.4):
    measured, judged = [], []
    for call in calls():
        ordinal, arm = call['question']['ordinal'], call['arm']
        prediction_sha = quote_sha256(f'{ordinal}-{arm}')
        response = SimpleNamespace(sha256=quote_sha256(f'response-{ordinal}-{arm}'), payload={'measurement': {
            'prediction_sha256': prediction_sha, 'prepare_s': .2 if arm in evaluation.MEMORY_ARMS else 0,
            'e2e_ttft_s': candidate_latency if arm == 'threaded' else 5.,
            'e2e_total_s': candidate_latency if arm == 'threaded' else 5., 'finish_reason': 'stop'}})
        measured.append((call, response))
        if arm in evaluation.MEMORY_ARMS:
            judged.append({'ordinal': ordinal, 'arm': arm, 'prediction_sha256': prediction_sha,
                           'response_sha256': response.sha256, 'correct': ordinal < correct})
    return measured, judged


def test_all_questions_have_adjacent_counterbalanced_identical_evidence_pairs():
    population = calls()
    evaluation.validate_calls(population)
    first = 0
    for ordinal in range(100):
        arms = [c['arm'] for c in population[ordinal * 4:ordinal * 4 + 4]]
        assert abs(arms.index('threaded') - arms.index('threaded_api')) == 1
        first += arms.index('threaded') < arms.index('threaded_api')
    assert first == 50


@pytest.mark.parametrize('corruption', ['partial', 'question', 'api_evidence', 'short_reader', 'order'])
def test_invalid_outbound_populations_are_rejected(corruption):
    population = deepcopy(calls())
    if corruption == 'partial':
        population.pop()
    elif corruption == 'question':
        population[-1]['question'] = {'ordinal': 0, 'prompt_question': 'Replacement'}
    elif corruption in ('api_evidence', 'short_reader'):
        arm = 'threaded_api' if corruption == 'api_evidence' else 'short_api'
        call = next(c for c in population if c['arm'] == arm)
        call['messages'] = deepcopy(call['messages'])
        call['messages'][0 if arm == 'short_api' else 1]['content'] = 'Changed'
        call['messages_sha256'] = identity_sha256(call['messages'])
    else:
        population[0], population[1] = population[1], population[0]
    with pytest.raises(ValueError):
        evaluation.validate_calls(population)


def test_joint_gate_requires_quality_and_latency_on_the_same_predictions():
    measured, judged = observations()
    assert evaluation.joint_statistics(measured, judged)['target_gate_passed']
    measured, judged = observations(correct=94)
    assert not evaluation.joint_statistics(measured, judged)['target_gate_passed']
    measured, judged = observations(candidate_latency=5.6)
    assert not evaluation.joint_statistics(measured, judged)['target_gate_passed']


def test_95_percent_with_bad_p95_still_fails_the_joint_gate():
    measured, judged = observations()
    for call, response in measured:
        if call['arm'] == 'threaded' and call['question']['ordinal'] >= 94:
            response.payload['measurement']['e2e_total_s'] = 20.
    result = evaluation.joint_statistics(measured, judged)
    assert result['candidate_accuracy_passed']
    assert not result['candidate_latency_passed']
    assert not result['target_gate_passed']


@pytest.mark.parametrize('corruption', ['missing_control', 'other_prediction', 'other_response', 'truncated'])
def test_incomplete_mismatched_or_truncated_measurements_cannot_pass(corruption):
    measured, judged = observations()
    if corruption == 'missing_control':
        measured.pop()
    elif corruption == 'other_prediction':
        judged[0]['prediction_sha256'] = '0' * 64
    elif corruption == 'other_response':
        judged[0]['response_sha256'] = '0' * 64
    else:
        measured[0][1].payload['measurement']['finish_reason'] = 'length'
        assert not evaluation.joint_statistics(measured, judged)['target_gate_passed']
        return
    with pytest.raises(ValueError):
        evaluation.joint_statistics(measured, judged)


def test_partial_answer_population_blocks_reference_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, 'load_preflight', lambda _: object())
    def incomplete(*args):
        raise ValueError('incomplete answer population')
    def forbidden():
        pytest.fail('references opened before all400 answers')
    monkeypatch.setattr(evaluation, 'seal_answers', incomplete)
    monkeypatch.setattr(evaluation, 'load_references', forbidden)
    with pytest.raises(ValueError, match='incomplete'):
        evaluation.judge(tmp_path)
