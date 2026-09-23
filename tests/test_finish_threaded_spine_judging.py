from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.finish_threaded_spine_judging import validate_judge_rows


def population():
    rows, observations = [], []
    for ordinal in range(100):
        for arm in ('flat', 'threaded', 'threaded_api', 'short_api'):
            q = {'ordinal': ordinal, 'question_id': str(ordinal)}
            m = {'prediction': f'answer {ordinal}', 'prediction_sha256': f'prediction {ordinal}'}
            response = SimpleNamespace(sha256=f'response {ordinal} {arm}', payload={'measurement': m})
            observations.append(({'question': q, 'arm': arm}, response))
            if arm in ('flat', 'threaded'):
                rows.append({**q, 'arm': arm, **m, 'response_sha256': response.sha256})
    return rows, observations


def test_recovery_requires_all_predictions_from_the_exact_timed_responses():
    rows, observations = population()
    validate_judge_rows(rows, observations)
    with pytest.raises(ValueError, match='all200'):
        validate_judge_rows(rows[:-1], observations)
    for field in ('prediction', 'prediction_sha256', 'response_sha256', 'question_id', 'arm', 'ordinal'):
        corrupted = deepcopy(rows)
        corrupted[0][field] = 'changed'
        with pytest.raises(ValueError, match='timed prediction'):
            validate_judge_rows(corrupted, observations)
