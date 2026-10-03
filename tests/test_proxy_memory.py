"""Exercise real HTTP routes and durable ChatSession with a lightweight backend."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time

import httpx
import pytest
from starlette.testclient import TestClient

from memory_condense.application.chat_session import ChatSession,ChatEvent
from memory_condense.interfaces.proxy_memory import MemoryProxy
from memory_condense.interfaces.proxy_server import ProxyConfig,build_app
from memory_condense.interfaces import proxy_memory_wire as wire
from tests.test_chat_streaming import StreamingBackend


HEADERS={'x-memory-conversation-id':'engineering','x-memory-request-id':'turn1','authorization':'Bearer test-secret'}


class Factory:
    def __init__(self): self.backends=[]
    def __call__(self, namespace,directory):
        backend=StreamingBackend()
        self.backends.append(backend)
        return ChatSession(directory,namespace,backend,streaming=True,recent_exchanges=2,recent_token_budget=8192)


def completion(provider,content):
    if provider=='anthropic':
        return dict(id='msg_test',type='message',role='assistant',model='test',
            content=[dict(type='text',text=content)],stop_reason='end_turn',stop_sequence=None,
            usage=dict(input_tokens=42,output_tokens=12))
    return dict(id='chatcmpl_test',object='chat.completion',created=1,model='test',
        choices=[dict(index=0,message=dict(role='assistant',content=content),finish_reason='stop')],
        usage=dict(prompt_tokens=42,completion_tokens=12,total_tokens=54))


def inline_content(user='Remember release r18.',answer='The release is r18.'):
    return json.dumps(dict(answer=answer,memory=dict(
        user=dict(summary=user,support=[user]),assistant=dict(summary=answer,support=[answer]))))


def app_for(tmp_path,handler,**options):
    factory=Factory()
    memory=MemoryProxy(tmp_path,factory,**options)
    app=build_app(config=ProxyConfig(mode='augment',upstreams={'openai':'https://example.test/v1','anthropic':'https://example.test'}),
                  memory=memory,client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return app,memory,factory


@pytest.mark.parametrize('provider,path',[('openai','/v1/chat/completions'),('anthropic','/v1/messages')])
@pytest.mark.parametrize('stream',[False,True])
def test_inline_is_hidden_and_response_protocol_and_retry_survive_reopen(tmp_path,provider,path,stream):
    sent=[]
    def upstream(request):
        sent.append(json.loads(request.content))
        assert request.url.path==path
        assert request.headers['authorization']=='Bearer test-secret'
        assert 'x-memory-conversation-id' not in request.headers
        value=completion(provider,inline_content())
        return httpx.Response(200,content=wire.answer_stream(provider,value) if stream else wire.encoded(value),
            headers={'content-type':'text/event-stream' if stream else 'application/json','set-cookie':'secret=hidden'})
    payload=dict(model='test',messages=[dict(role='user',content='Remember release r18.')],stream=stream,max_tokens=256)
    for reopening in range(2):
        app,memory,_=app_for(tmp_path,upstream)
        with TestClient(app) as client:
            response=client.post(path,headers=HEADERS,json=payload)
            assert response.status_code==200,response.text
            public=wire.assemble_stream(provider,response.content) if stream else response.json()
            assert wire.text_content(wire.response_message(provider,public))=='The release is r18.'
            assert b'"memory"' not in response.content and 'set-cookie' not in response.headers
            entry=next(iter(memory.entries.values()))
            answers=[e for e in entry.session.events() if e.role=='assistant']
            assert len(answers)==1 and answers[0].text=='The release is r18.'
            assert answers[0].metadata['inline_generation']['status']=='accepted'
            assert entry.session.flush()['pending_events']==0
            assert client.post(path,headers=HEADERS,json=payload).content==response.content
    assert len(sent)==1
    assert sent[0]['max_tokens']==768


def test_old_history_is_recalled_not_resent_and_feedback_is_applied(tmp_path):
    sent=[]
    def upstream(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200,json=completion('openai','Done.'))
    app,memory,factory=app_for(tmp_path,upstream,inline_memory=False)
    history=[]
    for i in range(4):
        history.extend([dict(role='user',content=f'old user {i}'),dict(role='assistant',content=f'old answer {i}')])
    policies=[dict(role='system',content='Keep system instructions.'),dict(role='developer',content='Keep developer instructions.')]
    initial=history+[dict(role='user',content='First new question')]
    with TestClient(app) as client:
        first=client.post('/v1/chat/completions',headers=HEADERS,json=dict(model='test',messages=policies+initial))
        assert first.status_code==200
        assert client.post('/_memory/flush',headers=HEADERS).status_code==200
        more=initial+[first.json()['choices'][0]['message'],dict(role='user',content='Follow up')]
        second=client.post('/v1/chat/completions',headers=dict(HEADERS,**{'x-memory-request-id':'turn2'}),json=dict(model='test',messages=policies+more))
        assert second.status_code==200,second.text
        assert sent[1]['messages'][:2]==policies
        text=json.dumps(sent[1]['messages'])
        assert 'old user 0' not in text and 'Remember the original design.' in text
        assert 'Follow up' in text and 'First new question' in text
        assert client.post('/_memory/flush',headers=HEADERS).json()['pending_feedback']==0
        learned=factory.backends[0].learned
        assert len(learned)>=1 and len({event_id for _,event_id in learned})==len(learned)
        assert learned[-1][0].query=='Follow up'
        events=next(iter(memory.entries.values())).session.events()
        assert sum(e.text=='old user 0' for e in events)==1
        assert sum(e.text=='Follow up' for e in events)==1


@pytest.mark.parametrize('stream',[False,True])
def test_native_openai_tool_calls_and_results_are_preserved(tmp_path,stream):
    requests=[]
    call=dict(id='call_1',type='function',function=dict(name='read_file',arguments='{"path":"app.py"}'))
    tool_message=dict(role='assistant',content='',tool_calls=[call])
    def upstream(request):
        payload=json.loads(request.content)
        requests.append(payload)
        assert 'Exchange input to summarize' not in json.dumps(payload)
        result=completion('openai','Fixed.')
        if len(requests)==1:
            result['choices'][0].update(message=tool_message,finish_reason='tool_calls')
        return httpx.Response(200,content=wire.answer_stream('openai',result) if stream else wire.encoded(result),
            headers={'content-type':'text/event-stream' if stream else 'application/json'})
    app,memory,_=app_for(tmp_path,upstream)
    tools=[dict(type='function',function=dict(name='read_file',parameters=dict(type='object',properties={}))) ]
    user=dict(role='user',content='Fix app.py')
    with TestClient(app) as client:
        result=client.post('/v1/chat/completions',headers=HEADERS,json=dict(model='test',messages=[user],tools=tools,stream=stream))
        assert result.status_code==200,result.text
        decoded=wire.assemble_stream('openai',result.content) if stream else result.json()
        assert decoded['choices'][0]['message']['tool_calls']==[call]
        tool=dict(role='tool',tool_call_id='call_1',content='print("broken")')
        result=client.post('/v1/chat/completions',headers=dict(HEADERS,**{'x-memory-request-id':'turn2'}),
            json=dict(model='test',messages=[user,tool_message,tool],tools=tools,stream=stream))
        assert result.status_code==200,result.text
        assert requests[1]['messages'][-2:]==[tool_message,tool]
        assert requests[1]['tools']==tools
        entry=next(iter(memory.entries.values()))
        assert any(e.role=='tool' and 'print' in e.text for e in entry.session.events())


def test_anthropic_tool_blocks_preserved(tmp_path):
    sent=[]
    blocks=[dict(type='tool_use',id='tool1',name='read_file',input={'path':'app.py'})]
    def upstream(request):
        sent.append(json.loads(request.content))
        value=completion('anthropic','Finished')
        if len(sent)==1:
            value.update(content=blocks,stop_reason='tool_use')
        return httpx.Response(200,json=value)
    app,memory,_=app_for(tmp_path,upstream)
    tools=[dict(name='read_file',input_schema=dict(type='object',properties={}))]
    with TestClient(app) as client:
        user=dict(role='user',content='Read app.py')
        result=client.post('/v1/messages',headers=HEADERS,json=dict(model='test',system='System',messages=[user],tools=tools,max_tokens=256))
        assert result.json()['content']==blocks
        tool=dict(role='user',content=[dict(type='tool_result',tool_use_id='tool1',content='Code')])
        result=client.post('/v1/messages',headers=dict(HEADERS,**{'x-memory-request-id':'turn2'}),
            json=dict(model='test',system='System',messages=[user,dict(role='assistant',content=blocks),tool],tools=tools,max_tokens=256))
        assert result.status_code==200,result.text
        assert sent[-1]['messages'][-1]==tool and sent[-1]['system']=='System'
        assert any(e.role=='tool' for e in next(iter(memory.entries.values())).session.events())


def test_empty_recovery_records_every_attempt_and_never_teaches_empty_reply(tmp_path):
    sent=[]
    def upstream(request):
        sent.append(request)
        return httpx.Response(200,json=completion('openai','' if len(sent)<3 else inline_content()))
    app,memory,_=app_for(tmp_path,upstream)
    with TestClient(app) as client:
        response=client.post('/v1/chat/completions',headers=HEADERS,json=dict(model='test',messages=[dict(role='user',content='Remember release r18.')]))
        assert response.status_code==200,response.text
        entry=next(iter(memory.entries.values()))
        assert len([e for e in entry.session.events() if e.role=='assistant'])==1
        with entry.session._connect() as db:
            assert db.execute('SELECT outcome FROM proxy_attempts ORDER BY attempt').fetchall()==[('empty_completed',),('empty_completed',),('complete',)]
    assert len(sent)==3


@pytest.mark.parametrize('failure',['timeout','invalid_json','empty','truncated_stream','token_limit'])
def test_failed_request_is_durable_and_replay_never_resends_unknown_outcome(tmp_path,failure):
    calls=[]
    def upstream(request):
        calls.append(request)
        if failure=='timeout': raise httpx.ReadTimeout('timeout')
        if failure=='truncated_stream':
            return httpx.Response(200,content=b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',headers={'content-type':'text/event-stream'})
        value=completion('openai','' if failure=='empty' else 'broken JSON')
        if failure=='token_limit': value['choices'][0]['finish_reason']='length'
        return httpx.Response(200,json=value)
    app,memory,_=app_for(tmp_path,upstream,inline_memory=failure!='token_limit')
    with TestClient(app) as client:
        payload=dict(model='test',messages=[dict(role='user',content='Question')],stream=failure=='truncated_stream')
        for _ in range(2):
            assert client.post('/v1/chat/completions',headers=HEADERS,json=payload).status_code==502
        session=next(iter(memory.entries.values())).session
        assert not any(e.role=='assistant' for e in session.events())
        assert len([e for e in session.events() if ':error:' in e.event_id])==1
        with session._connect() as db:
            assert db.execute('SELECT COUNT(*) FROM feedback').fetchone()[0]==0
    assert len(calls)==(3 if failure=='empty' else 1)


def test_namespaces_id_conflicts_and_concurrent_duplicate_requests(tmp_path):
    calls=[]
    def upstream(request):
        calls.append(request)
        time.sleep(.02)
        return httpx.Response(200,json=completion('openai','Answer'))
    app,memory,_=app_for(tmp_path,upstream,inline_memory=False)
    with TestClient(app) as client:
        payload=dict(model='test',messages=[dict(role='user',content='Question')])
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies=list(pool.map(lambda _:client.post('/v1/chat/completions',headers=HEADERS,json=payload),range(2)))
        assert all(r.status_code==200 for r in replies) and len(calls)==1
        conflict=client.post('/v1/chat/completions',headers=HEADERS,json=dict(payload,messages=[dict(role='user',content='Changed')]))
        assert conflict.status_code==409 and len(calls)==1
        assert client.post('/v1/chat/completions',headers={'authorization':'Bearer test-secret'},json=payload).status_code==400
        assert client.post('/v1/chat/completions',headers=dict(HEADERS,authorization='Bearer second-user'),json=payload).status_code==200
        assert len(memory.entries)==2 and len(calls)==2
        assert client.get('/_memory/health').json()['active_sessions']==2


def test_tool_sse_fragments_and_truncated_anthropic_are_not_lost():
    def frame(x): return b'data: '+wire.encoded(x)+b'\n\n'
    stream=b''.join([frame({'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'id':'c','type':'function','function':{'name':'read','arguments':'{"p":'}}]}}]}),
        frame({'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'function':{'arguments':'"a"}'}}]},'finish_reason':'tool_calls'}]}),b'data: [DONE]\n\n'])
    assert wire.assemble_stream('openai',stream)['choices'][0]['message']['tool_calls'][0]['function']['arguments']=='{"p":"a"}'
    with pytest.raises(ValueError,match='before completion'):
        wire.assemble_stream('anthropic',frame({'type':'message_start','message':{'id':'a','content':[]}}))


@pytest.mark.parametrize('state',['captured','calling','output_captured'])
def test_crash_boundaries_recover_without_duplicate_generation(tmp_path,state):
    calls=[]
    def upstream(request):
        calls.append(request)
        return httpx.Response(200,json=completion('openai','Answer'))
    app,memory,_=app_for(tmp_path,upstream,inline_memory=False)
    payload=dict(model='test',messages=[dict(role='user',content='Question')])
    body=wire.encoded(payload)
    request_id='proxy:'+hashlib.sha256(b'turn1').hexdigest()
    entry=memory.entry(memory.namespace('openai',HEADERS))
    if state=='captured':
        entry.session.ingest(ChatEvent(request_id+':input:0','user','Question',metadata={
            'proxy_message':payload['messages'][0],'proxy_request_sha256':hashlib.sha256(body).hexdigest()}))
    elif state=='calling':
        with entry.session._connect() as db:
            db.execute('INSERT INTO proxy_requests(request_id,body_sha,state) VALUES(?,?,?)',
                (request_id,hashlib.sha256(body).hexdigest(),'calling'))
    else:
        with TestClient(app) as client:
            assert client.post('/v1/chat/completions',headers=HEADERS,content=body).status_code==200
            with entry.session._connect() as db:
                db.execute("UPDATE proxy_requests SET state='calling'")
        app,memory,_=app_for(tmp_path,upstream,inline_memory=False)
    # Reopening also reconstructs the native conversation from captured IO.
    if state!='output_captured':
        memory.close()
        app,memory,_=app_for(tmp_path,upstream,inline_memory=False)
    with TestClient(app) as client:
        reply=client.post('/v1/chat/completions',headers=HEADERS,content=body)
        assert reply.status_code==(409 if state=='calling' else 200),reply.text
        entry=next(iter(memory.entries.values()))
        if state!='calling':
            assert sum(e.role=='user' for e in entry.session.events())==1
            with entry.session._connect() as db:
                assert db.execute('SELECT state FROM proxy_requests').fetchone()[0]=='complete'
    assert len(calls)==(0 if state=='calling' else 1)


def test_provider_error_is_replayed_and_audited_without_credentials(tmp_path):
    calls=[]
    def upstream(request):
        calls.append(request)
        return httpx.Response(429,json={'error':{'message':'Busy'}},headers={'retry-after':'10'})
    app,memory,_=app_for(tmp_path,upstream)
    with TestClient(app) as client:
        payload=dict(model='test',messages=[dict(role='user',content='Question')])
        first=client.post('/v1/chat/completions',headers=HEADERS,json=payload)
        again=client.post('/v1/chat/completions',headers=HEADERS,json=payload)
        assert first.status_code==again.status_code==429 and first.content==again.content
        assert first.headers['retry-after']==again.headers['retry-after']=='10'
        entry=next(iter(memory.entries.values()))
        with entry.session._connect() as db:
            served=db.execute('SELECT served_body FROM proxy_requests').fetchone()[0]
            response=db.execute('SELECT response_body FROM proxy_attempts').fetchone()[0]
            assert json.loads(served)['messages'][-1]['role']=='user'
            assert response==first.content
            assert 'test-secret' not in '\n'.join(db.iterdump())
    assert len(calls)==1


@pytest.mark.parametrize('option',[{'response_format':{'type':'json_object'}},{'logprobs':True}])
def test_client_output_contract_disables_inline_summaries(tmp_path,option):
    def upstream(request):
        sent=json.loads(request.content)
        assert all(sent[k]==v for k,v in option.items())
        assert sent['max_tokens']==100
        assert 'Exchange input to summarize' not in json.dumps(sent)
        value=completion('openai','{"release":"r18"}')
        value['choices'][0]['logprobs']={'content':[]}
        return httpx.Response(200,json=value)
    app,_,_=app_for(tmp_path,upstream)
    with TestClient(app) as client:
        response=client.post('/v1/chat/completions',headers=HEADERS,
            json=dict(model='test',messages=[dict(role='user',content='Release?')],max_tokens=100,**option))
        assert response.status_code==200,response.text
        assert response.json()['choices'][0]['logprobs']=={'content':[]}
