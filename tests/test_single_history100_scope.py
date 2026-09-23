from types import SimpleNamespace

import pytest

from tools import evaluate_native_spine_single_history100 as evaluation
from tools.prepare_native_spine_single_history100 import validate_authored
from tools.matched_eval.artifacts import publish_sealed_json
from tools.prepare_native_spine_design_slice import binding


def population(tmp_path):
    case = {'namespace_id': 'one-history', 'namespace_sha256': 'same-source', 'question_date': '2023/05/30'}
    scope, _ = publish_sealed_json(tmp_path/'scope.json', {'case': case,
        'actual_body_tokens': 1_098_417, 'through_question_day_body_tokens': 1_098_417})
    questions = SimpleNamespace(payload={'history_count': 1, 'question_count': 100,
        'scope': binding(scope), 'actual_body_tokens': 1_098_417,
        'questions': [{**case, 'ordinal': i, 'question_id': str(i), 'question': f'Question {i}?'} for i in range(100)]})
    return scope, questions


def test_exactly_one_history_and_100_questions_required(tmp_path):
    scope, questions = population(tmp_path)
    assert len(evaluation.validate_population(questions, scope)) == 100
    questions.payload['questions'][99]['namespace_id'] = 'second-history'
    with pytest.raises(ValueError, match='one unchanged'):
        evaluation.validate_population(questions, scope)


def test_duplicate_questions_cannot_inflate_accuracy_population(tmp_path):
    scope, questions = population(tmp_path)
    questions.payload['questions'][99]['question'] = questions.payload['questions'][0]['question']
    with pytest.raises(ValueError, match='100 unique'):
        evaluation.validate_population(questions, scope)


def test_generated_reference_must_quote_the_actual_user_turn():
    source = {'user_turns': [{'turn_index': 2, 'text': 'I requested PostgreSQL for the database.'}]}
    row = {'question': 'Which database did I request?', 'answer': 'PostgreSQL', 'category': 'requirement',
        'supports': [{'turn_index': 2, 'quote': 'requested PostgreSQL'}]}
    assert validate_authored(row, source) is row
    row['supports'][0]['quote'] = 'requested MySQL'
    with pytest.raises(ValueError, match='exact source user text'):
        validate_authored(row, source)
