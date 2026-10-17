"""Codex transport uses the shared durable memory/reduction pipeline."""
import gzip
import json

import httpx
import pytest
from starlette.testclient import TestClient

from memory_condense.interfaces.proxy_codex import CodexMemoryProxy, conversation_headers
from memory_condense.interfaces.proxy_server import ProxyConfig, build_app
from memory_condense.interfaces import proxy_memory_responses as responses
from memory_condense.interfaces import proxy_memory_wire as wire
from tests.test_proxy_memory import Factory
from tests.test_proxy_responses import message_item, response_body, events
from tests.test_proxy_memory import completion


HEADERS = {'authorization': 'Bearer first-token', 'chatgpt-account-id': 'account-one'}


def payload(items, turn='turn-one'):
    return dict(model='test', stream=True, instructions='Preserve these instructions.',
                input=items, tools=[dict(type='custom', name='apply_patch')],
                client_metadata=dict(installation_id='installation', thread_id='thread',
                                     turn_id=turn, request_kind='turn'))


def app_for(tmp_path, upstream):
    memory = CodexMemoryProxy(tmp_path, Factory(), inline_memory=False, summary_api_fallback=False)
    return build_app(config=ProxyConfig(mode='augment', codex_chatgpt=True,
        upstreams={'openai': 'https://api.openai.com', 'anthropic': 'https://api.anthropic.com'},
        default_provider='openai'),
        memory=memory, client=httpx.AsyncClient(transport=httpx.MockTransport(upstream))), memory


def test_custom_and_namespaced_calls_survive_replay_reduction_token_refresh_and_restart(tmp_path):
    outputs = [dict(type='message', role='assistant', phase='commentary',
                    content=[dict(type='output_text', text='I will inspect it.')]),
               dict(type='function_call', call_id='read', name='exec_command',
                    namespace='functions', arguments='{"cmd":"pwd"}'),
               dict(type='custom_tool_call', call_id='patch', name='apply_patch',
                    input='*** Begin Patch\n*** End Patch')]
    results = [dict(type='function_call_output', call_id='read', output='C:/project'),
               dict(type='custom_tool_call_output', call_id='patch', output='Applied')]
    seen = []

    def upstream(request):
        assert request.url.path == '/backend-api/codex/responses'
        assert 'x-memory-conversation-id' not in request.headers
        seen.append(json.loads(request.content))
        value = response_body(output=outputs) if len(seen) == 1 else response_body('Done.')
        return httpx.Response(200, content=events(value), headers={'content-type': 'text/event-stream'})

    old = [item for i in range(6) for item in
           (message_item('user', f'old question {i}'), message_item('assistant', f'old answer {i}', kind='output_text'))]
    initial = payload(old + [message_item('user', 'Fix the file.')])
    app, memory = app_for(tmp_path, upstream)
    with TestClient(app) as client:
        first = client.post('/responses', headers=HEADERS, json=initial)
        assert first.status_code == 200, first.text
        assert first.content == events(response_body(output=outputs))
        next(iter(memory.entries.values())).session.flush()
        continuation = payload(initial['input'] + outputs + results)
        second = client.post('/responses', headers=dict(HEADERS, authorization='Bearer refreshed'), json=continuation)
        assert second.status_code == 200, second.text
        assert len(memory.entries) == 1
        assert seen[-1]['input'][-5:] == outputs + results
        assert 'old question 0' not in json.dumps(seen[-1]['input'])
        assert 'Remember the original design.' in json.dumps(seen[-1]['input'])
        assert seen[-1]['instructions'] == initial['instructions']
        assert seen[-1]['tools'] == initial['tools']
        assert seen[-1]['store'] is False
        assert len(wire.encoded(seen[-1]['input'])) < len(wire.encoded(continuation['input']))
        assert client.post('/responses', headers=HEADERS, json=continuation).content == second.content
    app, _ = app_for(tmp_path, upstream)
    with TestClient(app) as client:
        assert client.post('/responses', headers=HEADERS, json=continuation).content == second.content
    assert len(seen) == 2


def test_multiple_assistant_phases_are_not_collapsed_in_history():
    output = [dict(message_item('assistant', 'Working', kind='output_text'), phase='commentary'),
              dict(message_item('assistant', 'Finished', kind='output_text'), phase='final_answer')]
    message = responses.response_message(response_body(output=output))
    assert responses.input_item(message) == output
    incoming = responses.dialogue(payload(output + [message_item('user', 'Next')]))
    assert CodexMemoryProxy._new_messages([('answer', message)], incoming, 'responses') == incoming[-1:]


def test_responses_lite_tool_catalog_survives_reduction_updates_and_restart(tmp_path):
    catalog = dict(type='additional_tools', id='at_tools', role='developer', tools=[
        dict(type='namespace', name='functions', description='', tools=[
            dict(type='custom', name='exec', description='Execute code',
                 format=dict(type='text'))])])
    seen = []

    def upstream(request):
        seen.append(json.loads(request.content))
        assert request.headers['x-openai-internal-codex-responses-lite'] == 'true'
        return httpx.Response(200, content=events(response_body('Done.')),
                              headers={'content-type': 'text/event-stream'})

    history = [item for i in range(6) for item in (
        message_item('user', f'Question {i}'), message_item('assistant', f'Answer {i}', kind='output_text'))]
    body = payload([catalog, message_item('developer', 'Keep the code intact.'),
                    *history, message_item('user', 'Verify the package.')])
    body.pop('tools')
    body.pop('instructions')
    assert not responses.use_inline(body)
    app, memory = app_for(tmp_path, upstream)
    headers = dict(HEADERS, **{'x-openai-internal-codex-responses-lite': 'true'})
    with TestClient(app) as client:
        first = client.post('/responses', headers=headers, json=body)
        assert first.status_code == 200, first.text
        session = next(iter(memory.entries.values())).session
        session.flush()
        updated = dict(catalog, id='at_updated', tools=[])
        followup = payload([updated, body['input'][1], *history,
            body['input'][-1], *response_body('Done.')['output'], message_item('user', 'Continue.')], 'next')
        followup.pop('tools')
        followup.pop('instructions')
        second = client.post('/responses', headers=headers, json=followup)
        assert second.status_code == 200, second.text
        assert seen[0]['input'][0] == catalog
        assert seen[-1]['input'][0] == updated
        assert seen[-1]['input'][1] == body['input'][1]
        assert 'tools' not in seen[-1]
        assert 'Question 0' not in json.dumps(seen[-1]['input'])
        assert 'Remember the original design.' in json.dumps(seen[-1]['input'])
        assert all('responses_tools' not in e.metadata.get('proxy_message', {}) for e in session.events())
        assert client.post('/responses', headers=headers, json=followup).content == second.content
    app, _ = app_for(tmp_path, upstream)
    with TestClient(app) as client:
        assert client.post('/responses', headers=headers, json=followup).content == second.content
    assert len(seen) == 2


@pytest.mark.parametrize('item', [
    dict(type='additional_tools', role='user', tools=[]),
    dict(type='additional_tools', role='developer', tools={}),
])
def test_invalid_responses_lite_catalog_is_rejected(item):
    with pytest.raises(ValueError, match='additional_tools requires'):
        responses.canonical_item(item)


def test_codex_terminal_envelope_without_output_uses_completed_items():
    item = dict(type='custom_tool_call', call_id='patch', name='apply_patch', input='patch')
    frames = [dict(type='response.output_item.done', output_index=0, item=item),
              dict(type='response.completed', response=dict(id='r', status='completed'))]
    assert responses.assemble_stream(frames)['output'] == [item]
    with pytest.raises(ValueError, match='omitted'):
        responses.assemble_stream([dict(frames[0], output_index=1), frames[1]])


def test_images_survive_history_reduction_without_entering_text_index(tmp_path):
    image = dict(type='message', role='user', content=[dict(type='input_text', text='Screenshot'),
                 dict(type='input_image', image_url='data:image/png;base64,TEST-IMAGE')])
    seen = []
    def upstream(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=events(response_body('Done.')), headers={'content-type':'text/event-stream'})
    app, memory = app_for(tmp_path, upstream)
    history = [image]
    for i in range(5):
        history.extend([message_item('assistant', str(i), kind='output_text'), message_item('user', str(i))])
    body = payload(history)
    with TestClient(app) as client:
        first = client.post('/responses', headers=HEADERS, json=body)
        assert first.status_code == 200, first.text
        session = next(iter(memory.entries.values())).session
        session.flush()
        history += response_body('Done.')['output'] + [message_item('user','Next')]
        second = client.post('/responses', headers=HEADERS, json=payload(history,'next'))
        assert second.status_code == 200, second.text
        blocks = [block for item in seen[-1]['input'] for block in item.get('content',[])]
        assert image['content'][1] in blocks
        assert not any('TEST-IMAGE' in event.text for event in session.events()
                       if event.event_id.endswith(':input:0'))


def test_account_and_thread_isolation_and_retry_identity():
    body = payload([message_item('user', 'Hello')])
    headers = conversation_headers(body, HEADERS)
    ns = CodexMemoryProxy.namespace('openai', headers)
    assert CodexMemoryProxy.namespace('openai', dict(headers, authorization='Bearer refresh')) == ns
    assert CodexMemoryProxy.namespace('openai', dict(headers, **{'chatgpt-account-id': 'other'})) != ns
    other = payload([message_item('user', 'Hello')])
    other['client_metadata']['thread_id'] = 'other-thread'
    assert CodexMemoryProxy.namespace('openai', conversation_headers(other, HEADERS)) != ns
    other = payload([message_item('user', 'Different input in the same turn')])
    assert conversation_headers(other, HEADERS)['x-memory-request-id'] != headers['x-memory-request-id']


def test_compaction_matches_retained_suffix_without_reimporting_summary():
    history = [('u',dict(role='user',content='Fix the readiness status.')),
               ('a',dict(role='assistant',content='Changed it to 204.'))]
    incoming = [dict(role='user',content='Compacted conversation summary.'),
                *[dict(message) for _,message in history],dict(role='user',content='Verify it.')]
    assert CodexMemoryProxy._new_messages(history,incoming,'responses') == incoming[-1:]
    incoming[2]['content'] = 'A different branch.'
    from memory_condense.interfaces.proxy_memory import MemoryProxyError
    with pytest.raises(MemoryProxyError,match='History differs'):
        CodexMemoryProxy._new_messages(history,incoming,'responses')


def test_steering_can_omit_a_committed_prose_reply_across_later_replays(tmp_path):
    seen = []
    def upstream(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,content=events(response_body(f'Reply {len(seen)}')),
                              headers={'content-type':'text/event-stream'})
    body = payload([message_item('user','Earlier question'),
                    message_item('assistant','Earlier answer',kind='output_text'),
                    message_item('user','Hello')])
    app,_=app_for(tmp_path,upstream)
    with TestClient(app) as client:
        assert client.post('/responses',headers=HEADERS,json=body).status_code == 200
        # Codex steered before recording Reply 1. Its user/tool history is intact.
        steered=payload(body['input']+[message_item('user','Hello?')])
        second=client.post('/responses',headers=HEADERS,json=steered)
        assert second.status_code == 200,second.text
        assert client.post('/responses',headers=HEADERS,json=steered).content == second.content
    app,_=app_for(tmp_path,upstream)
    with TestClient(app) as client:
        continuation=payload(steered['input']+response_body('Reply 2')['output']+
                             [message_item('user','Continue')],'next')
        third=client.post('/responses',headers=HEADERS,json=continuation)
        assert third.status_code == 200,third.text
    assert len(seen) == 3


@pytest.mark.parametrize('change',['user','assistant','tool_call'])
def test_omitted_prose_matching_still_rejects_changed_history_or_missing_calls(change):
    from memory_condense.interfaces.proxy_memory import MemoryProxyError
    history=[('u',dict(role='user',content='Original request')),
             ('a',dict(role='assistant',content='Original reply')),
             ('u2',dict(role='user',content='Follow up')),
             ('a2',dict(role='assistant',content='Unreceived reply'))]
    incoming=[dict(m) for _,m in history[:-1]]+[dict(role='user',content='New request')]
    if change=='tool_call':
        history[-1][1]['tool_calls']=[dict(id='call',type='function',
            function=dict(name='run',arguments='{}'))]
    else:
        incoming[0 if change=='user' else 1]['content']='Changed history'
    with pytest.raises(MemoryProxyError,match='History differs'):
        CodexMemoryProxy._new_messages(history,incoming,'responses')


def test_compressed_input_and_nonconversation_routes(tmp_path):
    seen = []
    def upstream(request):
        seen.append(request)
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={'models': []})
        return httpx.Response(200, content=events(response_body('Hi')), headers={'content-type': 'text/event-stream'})
    app, _ = app_for(tmp_path, upstream)
    body = payload([message_item('user', 'Hello')])
    with TestClient(app) as client:
        assert client.get('/models', headers=HEADERS).status_code == 200
        reply = client.post('/responses', content=gzip.compress(wire.encoded(body)),
                            headers=dict(HEADERS, **{'content-encoding': 'gzip'}))
        assert reply.status_code == 200, reply.text
        assert 'content-encoding' not in seen[-1].headers
        body['client_metadata']['request_kind'] = 'title'
        assert client.post('/responses', json=body, headers=HEADERS).status_code == 200
        assert json.loads(seen[-1].content) == body


@pytest.mark.parametrize('metadata', [{}, {'thread_id': 'only-thread'}, {'x-codex-turn-metadata': 'invalid'}])
def test_missing_identity_fails_before_provider_call(tmp_path, metadata):
    def upstream(request):
        pytest.fail('Invalid identity must not reach upstream')
    app, _ = app_for(tmp_path, upstream)
    with TestClient(app) as client:
        body = payload([message_item('user', 'Hello')])
        body['client_metadata'] = metadata
        assert client.post('/responses', headers=HEADERS, json=body).status_code == 400


@pytest.mark.parametrize('provider,path,auth', [
    ('openai','/v1/chat/completions',{'authorization':'Bearer api-key'}),
    ('anthropic','/v1/messages',{'x-api-key':'anthropic-key'}),
])
def test_standard_provider_clients_remain_available_with_codex_enabled(tmp_path,provider,path,auth):
    seen=[]
    def upstream(request):
        seen.append(request)
        assert request.url.host == ('api.openai.com' if provider=='openai' else 'api.anthropic.com')
        return httpx.Response(200,json=completion(provider,'Recorded.'))
    app,memory=app_for(tmp_path,upstream)
    with TestClient(app) as client:
        headers=dict(auth,**{'x-memory-conversation-id':'normal-api-client'})
        body=dict(model='test',messages=[dict(role='user',content='Remember this.')],max_tokens=128)
        reply=client.post(path,headers=headers,json=body)
        assert reply.status_code == 200,reply.text
        assert len(memory.entries) == 1
        assert seen[-1].url.path == path
        for key,value in auth.items():
            assert seen[-1].headers[key] == value
