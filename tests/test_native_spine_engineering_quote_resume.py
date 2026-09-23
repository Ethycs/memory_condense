import json
from types import SimpleNamespace

import pytest

from tools import native_spine_engineering_quote_resume as repair


def response(quote):
    return json.dumps({'atoms': [{'label': 'T0', 'summary': 'A file was written.', 'support': [quote]}]})


@pytest.mark.parametrize('quote', [
    'Wrote tests/test_decay.py; sha256=' + 'b93bb282514e9bf74d0a8ab66f8c8ae56aea35b9f22a11ed3036f6505ee978ca',
    'Observed ' + '\U0001f9e0\u03bb' * 40,
])
def test_oversize_literal_quote_is_a_bounded_unicode_source_prefix(quote):
    result, changes = repair.repair_raw_support(response(quote), [SimpleNamespace(text=quote)])
    atom = json.loads(result)['atoms'][0]
    support, = atom['support']
    assert support and quote.startswith(support)
    assert repair.s.count_tokens(support) <= 32
    assert atom['summary'] == 'A file was written.'
    assert changes[-1]['repair'] == 'bounded_prefix_of_model_selected_exact_quote'


def test_nonliteral_quote_is_not_rescued_by_a_matching_prefix():
    quote = 'A shared phrase ' + 'invented ' * 40
    result, changes = repair.repair_raw_support(response(quote), [SimpleNamespace(text='A shared phrase')])
    assert json.loads(result)['atoms'][0]['support'] == [quote]
    assert not changes


def test_valid_short_evidence_stays_unchanged():
    result, changes = repair.repair_raw_support(response('Wrote file'), [SimpleNamespace(text='Wrote file')])
    assert json.loads(result) == json.loads(response('Wrote file'))
    assert not changes
