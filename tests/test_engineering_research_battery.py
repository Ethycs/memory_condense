import json
from copy import deepcopy

import pytest

from tools import engineering_research_battery as battery


TRANSCRIPT = ('Unattributed source π\r\nUser:\r\nKeep exact code.\r\n'
              'Assistant:\r\n```text\r\nUser:\r\nnot a role\r\n```\r\n'
              'User:\r\nBuild it now.\r\nAssistant:\r\nFUTURE_SECRET\r\n')


@pytest.fixture
def case_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(battery, 'token_count', lambda text: len(text.split()))
    root = tmp_path / 'sources'
    root.mkdir()
    name = 'Session_abcdef12_2025-01-01T00-00-00-000Z.txt'
    (root / name).write_bytes(TRANSCRIPT.encode('utf-8'))
    case = {'id': 'E01', 'domain': 'engineering', 'split': 'development', 'source': name,
            'cutoff_turn': 3, 'request_anchor': 'Build it now.', 'title': 'Fixture',
            'task_kind': 'python_component', 'task': 'Build the requested component.',
            'deliverables': ['result.py'], 'failure_classes': ['implementation'],
            'criteria': [{'id': f'C{i}', 'supports': [{'turn': 1, 'quote': 'Keep exact code.'}],
                          'criterion': 'Preserve exact code', 'weight': 1} for i in range(3)]}
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'protocol': {'arms': ['memory', 'full_context']}, 'cases': [case]}))
    return root, case, spec, tmp_path / 'out'


def test_parser_preserves_preamble_crlf_unicode_and_ignores_fenced_headers():
    turns = battery.parse_transcript(TRANSCRIPT)
    assert [t.role for t in turns] == ['source', 'user', 'assistant', 'user', 'assistant']
    assert turns[0].text == 'Unattributed source π\r\n'
    assert 'User:\r\nnot a role' in turns[2].text
    for turn in turns:
        assert TRANSCRIPT[turn.text_start:turn.text_end] == turn.text
        assert battery.sha(turn.text.encode('utf-8')) == turn.text_sha256


@pytest.mark.parametrize('text', ['User:\n```\nunfinished', 'plain notes', 'Assistant:\nno user'])
def test_parser_refuses_unreliable_boundaries(text):
    with pytest.raises(ValueError):
        battery.parse_transcript(text)


def test_prefix_never_contains_current_or_future(case_fixture):
    root, case, _, _ = case_fixture
    actor, rubric, metadata = battery.compile_case(case, root)
    history = battery.render_history(actor['history'])
    assert 'FUTURE_SECRET' not in json.dumps(actor)
    assert 'Build it now.' not in history
    assert actor['original_request'] == 'Build it now.\r\n'
    assert actor['history'][0]['role'] == 'source'
    assert metadata['future_original_turns_excluded'] == 1
    assert metadata['export_timestamp'] == '2025-01-01T00:00:00+00:00'
    assert 'criteria' not in actor and len(rubric['criteria']) == 3


@pytest.mark.parametrize('turn,quote', [(4, 'FUTURE_SECRET'), (-1, 'FUTURE_SECRET'), (1, 'fabricated')])
def test_rubric_rejects_future_negative_and_fabricated_anchors(case_fixture, turn, quote):
    root, case, _, _ = case_fixture
    case['criteria'][0]['supports'] = [{'turn': turn, 'quote': quote}]
    with pytest.raises(ValueError, match='Rubric evidence'):
        battery.compile_case(case, root)


def test_current_request_can_support_rubric(case_fixture):
    root, case, _, _ = case_fixture
    case['criteria'][0]['supports'] = [{'turn': 3, 'quote': 'Build it now.'}]
    battery.compile_case(case, root)


def test_both_arms_share_exact_task_and_instructions(case_fixture):
    root, case, _, _ = case_fixture
    actor, _, _ = battery.compile_case(case, root)
    full = battery.actor_messages(actor, battery.render_history(actor['history']))
    memory = battery.actor_messages(actor, 'RETRIEVED_ONLY')
    assert full[0] == memory[0]
    assert full[1]['content'].split('\n\nCurrent task:')[1] == memory[1]['content'].split('\n\nCurrent task:')[1]
    assert 'Keep exact code.' not in memory[1]['content']


@pytest.mark.parametrize('same_family', [True, False])
def test_split_leakage_fails_before_writing_any_artifact(case_fixture, same_family):
    root, case, spec, output = case_fixture
    second = deepcopy(case)
    second.update(id='E02', split='validation')
    if not same_family:
        second['source'] = case['source'].replace('abcdef12', '12345678')
        (root / second['source']).write_bytes((root / case['source']).read_bytes())
    spec.write_text(json.dumps({'protocol': {}, 'cases': [case, second]}))
    with pytest.raises(ValueError, match='leakage'):
        battery.build(spec, root, output)
    assert not output.exists()


def test_build_audit_and_source_tamper(case_fixture):
    root, case, spec, output = case_fixture
    prepared = battery.build(spec, root, output)
    assert prepared['case_count'] == 1
    assert battery.audit(output)['cases_verified'] == 1
    (root / case['source']).write_bytes((TRANSCRIPT + '\r\nchanged').encode('utf-8'))
    with pytest.raises(ValueError, match='Actor/rubric'):
        battery.audit(output)


@pytest.mark.parametrize('target', ['actor', 'rubric', 'inventory', 'manifest'])
def test_sealed_artifact_tampering_detected(case_fixture, target):
    root, _, spec, output = case_fixture
    battery.build(spec, root, output)
    manifest = json.loads((output / 'battery.json').read_text())
    if target == 'manifest':
        path = output / 'battery.json'
    elif target == 'inventory':
        path = battery.Path(manifest['inventory']['path'])
    else:
        path = battery.Path(manifest['cases'][0][target]['path'])
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='binding changed'):
        battery.audit(output)


def test_frozen_artifact_cannot_be_overwritten(tmp_path):
    battery.publish(tmp_path / 'a.json', {'x': 1})
    with pytest.raises(ValueError, match='Frozen artifact'):
        battery.publish(tmp_path / 'a.json', {'x': 2})


@pytest.mark.parametrize('path', ['../escape.txt', '.hidden/a.txt'])
def test_source_path_must_stay_in_visible_archive(case_fixture, path):
    root, _, _, _ = case_fixture
    with pytest.raises(ValueError):
        battery.safe_source(root, path)


@pytest.mark.parametrize('name', ['../x', 'C:/x', 'a\\x', '/x', '.'])
def test_artifact_escape_rejected(case_fixture, name):
    root, case, _, _ = case_fixture
    actor, _, _ = battery.compile_case(case, root)
    with pytest.raises(ValueError, match='escapes'):
        battery.validate_result({'case_id': 'E01', 'arm': 'memory', 'artifacts': {name: 'x'}}, actor)


def research_result(case_fixture):
    root, case, _, _ = case_fixture
    case.update(domain='research', deliverables=['analysis.md', 'claims.json'])
    actor, _, _ = battery.compile_case(case, root)
    claim = {'claim': 'The user asked for exact code.', 'status': 'user_requirement',
             'evidence': [{'turn_id': 'abcdef12:T0001', 'quote': 'Keep exact code.'}]}
    result = {'case_id': 'E01', 'arm': 'memory', 'artifacts': {
        'analysis.md': 'Source-grounded analysis.', 'claims.json': json.dumps([claim])}}
    return actor, result, claim


def test_research_claims_check_sources_but_do_not_claim_quality_score(case_fixture):
    actor, result, _ = research_result(case_fixture)
    report = battery.validate_result(result, actor)
    assert report['claims_checked'] == report['valid_citations'] == 1
    assert report['structurally_complete'] and not report['quality_scored']


@pytest.mark.parametrize('mutation', ['future', 'inexact', 'status', 'empty'])
def test_invalid_research_claims_rejected(case_fixture, mutation):
    actor, result, claim = research_result(case_fixture)
    if mutation == 'future':
        claim['evidence'] = [{'turn_id': 'abcdef12:T0004', 'quote': 'FUTURE_SECRET'}]
    elif mutation == 'inexact':
        claim['evidence'][0]['quote'] = 'Never stated'
    elif mutation == 'status':
        claim['status'] = 'proven'
    else:
        claim['evidence'] = []
    result['artifacts']['claims.json'] = json.dumps([claim])
    with pytest.raises(ValueError):
        battery.validate_result(result, actor)


def test_missing_deliverables_remain_incomplete(case_fixture):
    actor, result, _ = research_result(case_fixture)
    del result['artifacts']['claims.json']
    checked = battery.validate_result(result, actor)
    assert checked['missing_deliverables'] == ['claims.json'] and not checked['structurally_complete']
