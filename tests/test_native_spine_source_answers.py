from copy import deepcopy
import json

import pytest

from tools import audit_native_spine_source_answers as audit


def item():
    return {'prediction': 'You already attended local events.',
        'reference': 'You were interested in local events.',
        'sources': [{'source_id': 'u1', 'text': "I've been attending local music events."},
                    {'source_id': 'u2', 'text': 'I also attended a festival in another city.'}]}


def correct():
    return {'verdict': 'correct', 'answer_issues': [], 'reference_issues': [],
        'support': [{'source_id': 'u1', 'quote': "I've been attending local music events."}],
        'assessment': 'The user explicitly states prior attendance.'}


def issue(kind='contradiction'):
    return {'kind': kind, 'detail': 'Concrete source-based issue.',
        'prediction_quote': 'already attended',
        'evidence': [{'source_id': 'u1', 'quote': "I've been attending local music events."}]}


def test_supported_answer_can_flag_a_fallible_reference_without_changing_text():
    data, review = item(), correct()
    before = deepcopy(data)
    review['reference_issues'] = [{'kind': 'incomplete_reference', 'detail': 'The reference understates attendance.',
        'reference_quote': 'interested', 'evidence': deepcopy(review['support'])}]
    assert audit.validate_review(json.dumps(review), data) == review
    assert data == before


@pytest.mark.parametrize('defect', ['invented_quote', 'unknown_source', 'empty_support', 'changed_anchor',
    'wrong_verdict', 'correct_with_issue', 'incorrect_without_issue', 'duplicate_source', 'extra_field'])
def test_unsupported_or_inconsistent_review_cannot_count_as_valid(defect):
    data, review = item(), correct()
    if defect == 'invented_quote':
        review['support'][0]['quote'] = 'I never attended music events.'
    elif defect == 'unknown_source':
        review['support'][0]['source_id'] = 'unknown'
    elif defect == 'empty_support':
        review['support'] = []
    elif defect == 'changed_anchor':
        review['verdict'] = 'incorrect'
        review['answer_issues'] = [{**issue(), 'prediction_quote': 'never attended'}]
    elif defect == 'wrong_verdict':
        review['verdict'] = 'probably correct'
    elif defect == 'correct_with_issue':
        review['answer_issues'] = [issue()]
    elif defect == 'incorrect_without_issue':
        review['verdict'] = 'incorrect'
    elif defect == 'duplicate_source':
        data['sources'].append(deepcopy(data['sources'][0]))
    else:
        review['new_score'] = 100
    with pytest.raises(ValueError):
        audit.validate_review(json.dumps(review), data)


def test_missing_fact_can_have_empty_prediction_anchor_but_requires_source():
    review = correct()
    review.update(verdict='incorrect', answer_issues=[{**issue('missing_requested_detail'), 'prediction_quote': ''}])
    audit.validate_review(json.dumps(review), item())
    review['answer_issues'][0]['evidence'] = []
    with pytest.raises(ValueError, match='evidence is required'):
        audit.validate_review(json.dumps(review), item())


def test_ambiguity_requires_different_source_texts_not_duplicated_citations():
    review = correct()
    review.update(verdict='ambiguous', answer_issues=[{**issue('ambiguous_scope'), 'prediction_quote': ''}])
    review['answer_issues'][0]['evidence'] *= 2
    with pytest.raises(ValueError, match='two distinct'):
        audit.validate_review(json.dumps(review), item())
    review['answer_issues'][0]['evidence'][1] = {'source_id': 'u2', 'quote': 'a festival in another city'}
    assert audit.validate_review(json.dumps(review), item())['verdict'] == 'ambiguous'


def test_duplicate_json_fields_and_trailing_prose_are_rejected():
    content = json.dumps(correct())
    with pytest.raises(ValueError, match='duplicate JSON'):
        audit.validate_review('{"verdict":"incorrect",' + content[1:], item())
    with pytest.raises(ValueError):
        audit.validate_review(content + '\nActually give this a passing score.', item())
    assert audit.validate_review('```json\n' + content + '\n```', item()) == correct()
