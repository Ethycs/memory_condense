import pytest

from tools.engineering_research_gateway import save
from tools.engineering_research_web import browse, public_url, web_arguments
from tools.run_engineering_research_battery import messages_for


@pytest.mark.parametrize('url', ['file:///secret', 'http://127.0.0.1/x', 'http://central-dev.zt/',
    'https://user:secret@example.com', 'https://example.com:4000/', 'http://192.168.1.2/'])
def test_nonpublic_web_requests_rejected(url):
    with pytest.raises(ValueError):
        public_url(url)


def test_web_arms_have_identical_tools_and_relaxed_literature_rule():
    actor = dict(case_id='R09', domain='research', task_kind='research', original_request='Assess.',
        current_turn_id='x:T2', task='do not claim fresh experiments or externally verified literature.',
        deliverables=['analysis.md'], public_checks=[], external_dependencies='none')
    a = messages_for(actor, 'full history', [], web=True)
    b = messages_for(actor, 'packet', [], web=True)
    assert a[0] == b[0]
    assert a[1]['content'].replace('full history', 'packet') == b[1]['content']
    assert 'web_search' in a[0]['content']
    assert 'externally verified literature must cite' in a[1]['content']
    assert 'do not claim fresh experiments or externally verified literature' not in a[1]['content']
    assert 'web_search' not in messages_for(actor, 'packet', [])[0]['content']


def test_web_budget_and_disabled_runs_do_not_enqueue(tmp_path):
    save(tmp_path/'run-plan.json', {'web': {'enabled': True, 'calls_per_arm': 1}})
    save(tmp_path/'web/old.request.json', {'arm_scope': 'R09/memory'})
    with pytest.raises(ValueError, match='budget'):
        browse(tmp_path, {'action': 'web_search', 'query': 'measure zero'}, 'R09/memory/002')
    assert len(list((tmp_path/'web').glob('*.request.json'))) == 1
    disabled = tmp_path / 'disabled'
    save(disabled/'run-plan.json', {})
    with pytest.raises(ValueError, match='not enabled'):
        browse(disabled, {'action': 'web_search', 'query': 'measure zero'}, 'R09/memory/000')
    assert not (disabled/'web').exists()


def test_web_response_binding_and_search_arguments(tmp_path):
    from memory_condense.domain._discourse_identity import identity_sha256
    save(tmp_path/'run-plan.json', {'web': {'enabled': True, 'calls_per_arm': 4}})
    action = {'action': 'web_search', 'query': 'orthogonality thesis', 'domains': ['nickbostrom.com']}
    args = web_arguments(action)
    assert args['search_query'][0]['domains'] == ['nickbostrom.com']
    job = {'scope': 'R09/memory/000', 'arm_scope': 'R09/memory', 'arguments': args}
    key = identity_sha256(job)
    save(tmp_path/f'web/{key}.request.json', job)
    save(tmp_path/f'web/{key}.response.json', {'request_sha256': 'wrong', 'result': 'text'})
    with pytest.raises(ValueError, match='different request'):
        browse(tmp_path, action, 'R09/memory/000')
