from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import quote_sha256
from tools import audit_native_spine_paired_answers as audit


def population(tmp_path):
    cases, refs, observations, inputs, grades = [], [], [], [], []
    for ordinal in range(100):
        answer = f'Fact {ordinal}'
        case = {'ordinal': ordinal, 'question_id': f'q-{ordinal}', 'namespace_id': 'one-history',
                'question': f'What is fact {ordinal}?', 'question_date': '2026/09/15 (Tue) 12:00',
                'reference_sha256': quote_sha256(answer)}
        question = audit.frozen.question(case)
        cases.append(case)
        refs.append({'question_id': case['question_id'], 'answer': answer})
        memory = None
        for arm in (audit.evaluation.ARMS if ordinal % 2 == 0 else audit.evaluation.ARMS[::-1]):
            prediction = answer if arm == 'parent_context' or ordinal else 'Different fact'
            response = SimpleNamespace(path=tmp_path / f'{ordinal}-{arm}.json',
                sha256=quote_sha256(f'{ordinal}-{arm}-{prediction}'),
                payload={'measurement': {'prediction': prediction, 'prediction_sha256': quote_sha256(prediction)}})
            call = {'call_index': len(observations), 'arm': arm, 'question': deepcopy(question),
                    'messages': [{'role': 'user', 'content': f'Identical raw packet for {ordinal}'}]}
            observations.append((call, response))
            if arm == 'parent_context':
                memory = response
        identity = {'ordinal': ordinal, 'question_id': case['question_id'],
            'reference_sha256': quote_sha256(answer), 'prediction_sha256': quote_sha256(answer),
            'response_sha256': memory.sha256}
        inputs.append({**identity, 'messages': audit.frozen.build_judge_prompt(case['question'], answer, answer)})
        grades.append({**identity, 'correct': True, 'verdict': 'CORRECT'})
    return observations, cases, refs, inputs, grades


def test_all_controls_use_original_question_reference_and_grader(tmp_path):
    args = population(tmp_path)
    before = deepcopy(args)
    rows = audit.paired_rows(*args)
    assert args == before
    assert len(rows) == 100
    assert sum(r['identical_prediction'] for r in rows) == 99
    assert rows[0]['messages'] == audit.frozen.build_judge_prompt('What is fact 0?', 'Fact 0', 'Different fact')
    assert all('memory_correct' not in message['content'] for r in rows for message in r['messages'])


@pytest.mark.parametrize('defect', ['partial', 'arm', 'question', 'prompt', 'reference',
                                   'duplicate_ref', 'memory_grade', 'grader_prompt', 'order'])
def test_incomplete_or_changed_original_population_is_rejected(tmp_path, defect):
    observations, cases, refs, inputs, grades = population(tmp_path)
    if defect == 'partial':
        observations.pop()
    elif defect == 'arm':
        observations[1][0]['arm'] = 'parent_context'
    elif defect == 'question':
        observations[1][0]['question']['retrieval_query'] += ' altered'
    elif defect == 'prompt':
        observations[1][0]['messages'][0]['content'] += ' hint'
    elif defect == 'reference':
        refs[0]['answer'] = 'Altered reference'
    elif defect == 'duplicate_ref':
        refs[1] = refs[0]
    elif defect == 'memory_grade':
        grades[0]['correct'] = False
    elif defect == 'grader_prompt':
        inputs[0]['messages'][0]['content'] += ' Altered grading'
    else:
        cases.reverse()
    with pytest.raises(ValueError):
        audit.paired_rows(observations, cases, refs, inputs, grades)


def test_scores_remain_separate_and_identical_answer_disagreement_is_explicit():
    rows = [{'ordinal': n, 'memory_correct': correct, 'identical_prediction': identical,
             'messages': []} for n, correct, identical in [(0, True, False), (1, False, False), (2, True, True)]]
    result = audit.summarize(rows, ['INCORRECT', 'CORRECT', 'INCORRECT'])
    assert result['memory_accuracy'] == {'correct': 2, 'questions': 3}
    assert result['api_control_accuracy'] == {'correct': 1, 'questions': 3}
    assert result['api_only_correct_ordinals'] == [1]
    assert result['memory_only_correct_ordinals'] == [0, 2]
    assert result['identical_prediction_grade_disagreements'] == [2]


def test_partial_grading_cannot_publish_a_score():
    with pytest.raises(ValueError, match='every paired answer'):
        audit.summarize([{'ordinal': 0}], [])
