import json

import pytest

from memory_condense.search.spine_quote_json_repair_v4 import repair_support_list_closures


def document(summary='A preserved summary.', support=None):
    return json.dumps({"atoms": [{"label": "T0", "summary": summary,
        "support": support or ['“She said "hello" and "goodbye".”']}]}, ensure_ascii=False)


def test_escapes_only_internal_support_quotes_and_reverses_exactly():
    original = document('An untouched summary with "quotes".').replace('\\"hello\\"', '"hello"').replace('\\"goodbye\\"', '"goodbye"')
    repaired, audit = repair_support_list_closures(original)
    assert json.loads(repaired) == json.loads(document('An untouched summary with "quotes".'))
    assert audit['summary_texts_unchanged'] and audit['original_characters_preserved']
    assert len(audit['original_insertion_offsets']) == 4
    restored = repaired
    for i, offset in reversed(list(enumerate(audit['original_insertion_offsets']))):
        restored = restored[:offset + i] + restored[offset + i + 1:]
    assert restored == original


@pytest.mark.parametrize('support', [
    ['“"hello" is a greeting”'], ['“Already escaped "hello".”', '“Another "greeting".”'],
    ['“A path C:\\folder and "hello".”'],
])
def test_preserves_valid_json_without_normalization(support):
    original = document(support=support)
    assert repair_support_list_closures(original) == (original, None)


def test_multiple_atoms_keep_all_summaries_and_support_values():
    body = {'atoms': [{'label':f'T{i}', 'summary':f'Original summary {i}.',
        'support':['“They said "yes".”','“"no" was another option”']} for i in range(3)]}
    valid = json.dumps(body, ensure_ascii=False)
    broken = valid.replace('\\"', '"')
    repaired, audit = repair_support_list_closures(broken)
    assert json.loads(repaired) == body and len(audit['original_insertion_offsets']) == 12


@pytest.mark.parametrize('broken', [
    '{"atoms":[{"label":"T0","summary":"“Broken "summary"”","support":["safe"]}]}',
    '{"atoms":[{"label":"T0","summary":"safe","support":["No curly "quotes" here"]}]}',
    '{"atoms":[{"label":"T1","summary":"safe","support":["“Broken "quotes"”"]}]}',
    '{"atoms":[{"label":"T0","summary":"safe","support":["“Broken "quotes"”"another"]}]}',
    '{"atoms":[{"label":"T0","summary":"safe","support":["“Broken "quotes"”"],"extra":1}]}',
])
def test_refuses_summary_changes_ambiguous_unquoted_support_or_schema_changes(broken):
    with pytest.raises(ValueError):
        repair_support_list_closures(broken)
