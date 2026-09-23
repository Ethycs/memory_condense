from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from tools import evaluate_additive_spine_full100 as evaluation


def calls():
    result = []
    for ordinal in range(100):
        question = {'ordinal': ordinal, 'question_id': str(ordinal),
                    'retrieval_query': f'Question {ordinal}', 'prompt_question': f'Question {ordinal}'}
        messages = {arm: evaluation.answer_messages(question, f'Exact {arm} evidence') for arm in evaluation.MEMORY_ARMS}
        for arm in evaluation.CANDIDATES:
            messages[arm + '_api'] = messages[arm]
        messages['short_api'] = evaluation.answer_messages(question)
        for arm in evaluation.call_order(ordinal):
            result.append({'call_index': len(result), 'question': question, 'arm': arm,
                           'messages': messages[arm], 'messages_sha256': identity_sha256(messages[arm])})
    return result


def observations():
    measured, judged = [], []
    for call in calls():
        ordinal, arm = call['question']['ordinal'], call['arm']
        prediction = f'Answer {ordinal} {arm}'
        response = SimpleNamespace(sha256=quote_sha256(f'response-{ordinal}-{arm}'), payload={'measurement': {
            'prediction': prediction, 'prediction_sha256': quote_sha256(prediction),
            'prepare_s': .2 if arm in evaluation.MEMORY_ARMS else 0.,
            'e2e_ttft_s': 5., 'e2e_total_s': 5., 'finish_reason': 'stop'}})
        measured.append((call, response))
        if arm in evaluation.MEMORY_ARMS:
            judged.append({'ordinal': ordinal, 'arm': arm, 'prediction_sha256': quote_sha256(prediction),
                           'response_sha256': response.sha256, 'correct': ordinal < 95})
    return measured, judged


def test_both_candidates_have_complete_adjacent_counterbalanced_api_pairs():
    population = calls()
    evaluation.validate_calls(population)
    for candidate in evaluation.CANDIDATES:
        first = 0
        for ordinal in range(100):
            arms = [c['arm'] for c in population[ordinal * 6:ordinal * 6 + 6]]
            assert abs(arms.index(candidate) - arms.index(candidate + '_api')) == 1
            first += arms.index(candidate) < arms.index(candidate + '_api')
        assert first == 50


@pytest.mark.parametrize('corruption', ['partial', 'question', 'reader', 'api', 'order'])
def test_corrupted_full100_request_population_is_rejected(corruption):
    population = deepcopy(calls())
    if corruption == 'partial':
        population.pop()
    elif corruption == 'question':
        population[-1]['question'] = {'ordinal': 0, 'prompt_question': 'Changed'}
    elif corruption in ('reader', 'api'):
        call = next(c for c in population if c['arm'] == 'supplemented_api')
        call['messages'] = deepcopy(call['messages'])
        call['messages'][0 if corruption == 'reader' else 1]['content'] = 'Changed'
        call['messages_sha256'] = identity_sha256(call['messages'])
    else:
        population[0], population[1] = population[1], population[0]
    with pytest.raises(ValueError):
        evaluation.validate_calls(population)


def test_each_candidate_requires_its_own_95_percent_and_latency_pass():
    measured, judged = observations()
    assert evaluation.joint_statistics(measured, judged)['target_gate_passed'] == {'grouped': True, 'supplemented': True}
    next(r for r in judged if r['ordinal'] == 0 and r['arm'] == 'supplemented')['correct'] = False
    assert evaluation.joint_statistics(measured, judged)['target_gate_passed'] == {'grouped': True, 'supplemented': False}
    measured, judged = observations()
    for call, response in measured:
        if call['arm'] == 'grouped' and call['question']['ordinal'] >= 94:
            response.payload['measurement']['e2e_total_s'] = 20.
    assert evaluation.joint_statistics(measured, judged)['target_gate_passed'] == {'grouped': False, 'supplemented': True}


@pytest.mark.parametrize('corruption', ['partial', 'prediction', 'response', 'truncated'])
def test_incomplete_or_mismatched_streams_cannot_pass(corruption):
    measured, judged = observations()
    if corruption == 'partial':
        measured.pop()
    elif corruption in ('prediction', 'response'):
        judged[0][corruption + '_sha256'] = '0' * 64
    else:
        measured[0][1].payload['measurement']['finish_reason'] = 'length'
        assert not evaluation.joint_statistics(measured, judged)['any_target_gate_passed']
        return
    with pytest.raises(ValueError):
        evaluation.joint_statistics(measured, judged)


def test_partial_answers_block_reference_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, 'load_preflight', lambda _: object())
    def incomplete(*args):
        raise ValueError('all600 required')
    monkeypatch.setattr(evaluation, 'seal_answers', incomplete)
    monkeypatch.setattr(evaluation, 'load_references', lambda: pytest.fail('references opened early'))
    with pytest.raises(ValueError, match='all600'):
        evaluation.judge(tmp_path)


def test_complete_judging_and_replay_use_actual_runtime_and_isolated_client_contract(monkeypatch, tmp_path):
    measured, _ = observations()
    events, requests = [], []
    monkeypatch.setattr(evaluation, 'load_preflight', lambda _: SimpleNamespace(sha256='a' * 64))
    def seal(*args):
        events.append('sealed')
        return SimpleNamespace(sha256='b' * 64), measured
    def references():
        assert events[-1] == 'sealed'
        events.append('references')
        return None, [SimpleNamespace(question_id=str(i), answer=f'Reference {i}') for i in range(100)]
    def client_factory(*args):
        def create(**request):
            assert events[-1] == 'references'
            requests.append(request)
            return SimpleNamespace(id='synthetic-judge', model=request['model'], usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content='CORRECT'), finish_reason='stop')])
        return SimpleNamespace(max_retries=0, close=lambda: None,
                               chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(evaluation, 'seal_answers', seal)
    monkeypatch.setattr(evaluation, 'load_references', references)
    monkeypatch.setattr(evaluation, '_completion_client', client_factory)
    report = evaluation.judge(tmp_path, True)
    assert len(requests) == 300 and len(report.payload['rows']) == 300
    assert report.payload['accuracy'] == {'flat': 100, 'grouped': 100, 'supplemented': 100}
    assert report.payload['any_target_gate_passed']
    monkeypatch.setattr(evaluation, '_completion_client', lambda *args: pytest.fail('replay opened a provider client'))
    assert evaluation.judge(tmp_path, False).sha256 == report.sha256
