from copy import deepcopy
import json

import pytest

from tools.review_native_spine_model_pairs import validate_pair


def fixture():
    item = {'data': {'predictions': {'A': 'I chose the blue bike.', 'B': 'I chose the green bike.'},
                    'reference': 'A blue bike.',
                    'sources': [{'source_id': 'served-context', 'text': 'I chose the blue bike.'}]}}
    good = {'verdict': 'correct', 'answer_issues': [], 'reference_issues': [],
            'support': [{'source_id': 'served-context', 'quote': 'I chose the blue bike.'}],
            'assessment': 'Matches the source.'}
    bad = {'verdict': 'incorrect', 'answer_issues': [{'kind': 'contradiction',
           'detail': 'The selected color differs.', 'prediction_quote': 'green',
           'evidence': [{'source_id': 'served-context', 'quote': 'blue bike'}]}],
           'reference_issues': [], 'support': [], 'assessment': 'Wrong color.'}
    return item, {'A': good, 'B': bad}


def test_validates_both_answers_against_their_own_text_and_exact_sources():
    item, pair = fixture()
    assert validate_pair(json.dumps(pair), item) == pair


@pytest.mark.parametrize('defect', ['invented_quote', 'swapped_answers', 'missing_label', 'duplicate_label'])
def test_rejects_unsupported_or_misassigned_paired_reviews(defect):
    item, pair = fixture()
    pair = deepcopy(pair)
    if defect == 'invented_quote':
        pair['A']['support'][0]['quote'] = 'I chose the red bike.'
    elif defect == 'swapped_answers':
        item['data']['predictions'] = dict(A=item['data']['predictions']['B'], B=item['data']['predictions']['A'])
    elif defect == 'missing_label':
        del pair['B']
    text = json.dumps(pair)
    if defect == 'duplicate_label':
        text = '{"A":' + json.dumps(pair['A']) + ',"A":' + json.dumps(pair['A']) + ',"B":' + json.dumps(pair['B']) + '}'
    with pytest.raises(ValueError):
        validate_pair(text, item)
