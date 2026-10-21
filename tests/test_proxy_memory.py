"""Exercise real HTTP routes and durable ChatSession with a lightweight backend."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import threading
import time

import httpx
import pytest
from starlette.testclient import TestClient

from memory_condense.application.chat_session import ChatSession,ChatEvent
from memory_condense.control.snapshots import SessionSnapshotStore
from memory_condense.interfaces.proxy_memory import MemoryProxy, MemoryProxyError
from memory_condense.interfaces.proxy_server import ProxyConfig,build_app
from memory_condense.interfaces import proxy_memory_wire as wire
from tests.test_chat_streaming import StreamingBackend


HEADERS={'x-memory-conversation-id':'engineering','x-memory-request-id':'turn1','authorization':'Bearer test-secret'}


class Factory:
    def __init__(self): self.backends,self.lifecycle,self.forgotten=[],{},[]
    def __call__(self, namespace,directory):
        backend=StreamingBackend()
        self.backends.append(backend)
        self.lifecycle[namespace]=dict(stage='ready')
        return ChatSession(directory,namespace,backend,streaming=True,recent_exchanges=2,recent_token_budget=8192)
    def startup_status(self):
        return {namespace:dict(record) for namespace,record in self.lifecycle.items()}
    def forget(self, namespaces=None):
        # NativeProxySessions.forget: per-namespace state over trees a swap replaced.
        self.forgotten.append(namespaces)
        if namespaces is None:
            self.lifecycle.clear()
        else:
            for namespace in namespaces:
                self.lifecycle.pop(namespace,None)


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
        # A client without a conversation header (Claude Code, Codex) is
        # attributed by request digest and served in its own namespace, not refused.
        headerless=client.post('/v1/chat/completions',headers={'authorization':'Bearer test-secret'},json=payload)
        assert headerless.status_code==200 and len(memory.entries)==2 and len(calls)==2
        # Credentials do not partition memory: this system assumes a
        # privileged environment, and a conversation keeps its store when the
        # provider token rotates. Previously this minted a third namespace,
        # which meant an OAuth refresh silently emptied a live conversation's
        # memory and consumed one of max_sessions. Same conversation and same
        # request id, so it replays rather than calling upstream again.
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


def _history(*turns):
    """``messages`` for a replayed conversation: user/assistant/user/..."""
    return [dict(role='assistant' if i%2 else 'user',content=text) for i,text in enumerate(turns)]


def test_swap_closes_every_session_and_a_snapshot_round_trips(tmp_path):
    """Augment save -> mutate -> load, proving the restored rows come back."""
    def upstream(request):
        return httpx.Response(200,json=completion('openai','Answer'))
    app,memory,factory=app_for(tmp_path,upstream,inline_memory=False)
    snapshots=SessionSnapshotStore(tmp_path)
    with TestClient(app) as client:
        def ask(request_id,*turns):
            reply=client.post('/v1/chat/completions',headers=dict(HEADERS,**{'x-memory-request-id':request_id}),
                json=dict(model='test',messages=_history(*turns)))
            assert reply.status_code==200,reply.text
        ask('turn1','First question')
        base=snapshots.save('base')
        assert base['file']=='base' and base['file_count']>0
        ask('turn2','First question','Answer','Second question')
        entry=next(iter(memory.entries.values()))
        session=entry.session
        assert [e.text for e in session.events() if e.role=='user']==['First question','Second question']

        assert memory.swap(lambda: snapshots.restore('base'))==tmp_path/'sessions'
        assert memory.entries=={} and memory.generation==1 and memory.last_swap_at
        assert entry.stale and session.status()['closed']
        # The factory's per-namespace state described the replaced tree.
        assert factory.forgotten==[None] and factory.startup_status()=={}

        # The next request re-admits the conversation over the restored tree.
        ask('turn3','First question','Answer','Third question')
        entry=next(iter(memory.entries.values()))
        assert entry.session is not session
        assert [e.text for e in entry.session.events() if e.role=='user']==['First question','Third question']
        assert list(factory.startup_status())==[memory.namespace('openai',HEADERS)]


def test_a_swap_holds_off_a_concurrent_request_and_never_half_serves(tmp_path):
    def upstream(request):
        return httpx.Response(200,json=completion('openai','Answer'))
    app,memory,_=app_for(tmp_path,upstream,inline_memory=False)
    snapshots=SessionSnapshotStore(tmp_path)
    with TestClient(app) as client:
        def ask(request_id,*turns):
            reply=client.post('/v1/chat/completions',headers=dict(HEADERS,**{'x-memory-request-id':request_id}),
                json=dict(model='test',messages=_history(*turns)))
            assert reply.status_code==200,reply.text
        ask('turn1','First question')
        snapshots.save('base')
        ask('turn2','First question','Answer','Second question')

        order,gate,entered=[],threading.Event(),threading.Event()
        def mutate():
            entered.set()
            assert gate.wait(20)
            result=snapshots.restore('base')
            order.append('swap')
            return result
        with ThreadPoolExecutor(1) as pool:
            swap=pool.submit(memory.swap,mutate)
            assert entered.wait(20)
            threading.Timer(0.5,gate.set).start()
            started=time.perf_counter()
            ask('turn3','First question','Answer','Third question')
            elapsed=time.perf_counter()-started
            order.append('request')
            swap.result(30)
        assert order==['swap','request'] and elapsed>=0.4
        # Served from the restored tree, never from a half-swapped one.
        entry=next(iter(memory.entries.values()))
        assert [e.text for e in entry.session.events() if e.role=='user']==['First question','Third question']


def test_a_swap_waits_for_in_flight_io_and_a_closed_entry_is_re_resolved(tmp_path):
    def upstream(request):
        return httpx.Response(200,json=completion('openai','Answer'))
    app,memory,_=app_for(tmp_path,upstream,inline_memory=False)
    with TestClient(app) as client:
        assert client.post('/v1/chat/completions',headers=HEADERS,
            json=dict(model='test',messages=_history('First question'))).status_code==200
        namespace=memory.namespace('openai',HEADERS)
        entry=memory.entries[namespace]

        # An in-flight exchange owns the session IO lock; the swap must drain
        # it before it touches a single file.
        entry.session._io_lock.acquire()
        released=False
        try:
            with ThreadPoolExecutor(1) as pool:
                swap=pool.submit(memory.swap,lambda:'mutated')
                time.sleep(0.3)
                assert not swap.done(),'the swap must wait for in-flight IO'
                entry.session._io_lock.release()
                released=True
                assert swap.result(30)=='mutated'
        finally:
            if not released:
                entry.session._io_lock.release()
        assert entry.stale and memory.entries=={}

        # A request that resolved its entry before the swap must not use it.
        resolve,calls=memory.entry,[]
        def once(namespace,*,summary_api=None):
            calls.append(namespace)
            return entry if len(calls)==1 else resolve(namespace,summary_api=summary_api)
        memory.entry=once
        try:
            with memory._io(namespace) as fresh:
                assert fresh is not entry and not fresh.stale
        finally:
            memory.entry=resolve
        assert len(calls)==2


def test_a_failed_swap_mutation_leaves_the_proxy_serving(tmp_path):
    """Finding 4 for augment: sessions are re-admitted lazily, so a mutation
    that failed before touching the tree cannot wedge the proxy."""
    def boom():
        raise PermissionError('sharing violation')
    def upstream(request):
        return httpx.Response(200,json=completion('openai','Answer'))
    app,memory,_=app_for(tmp_path,upstream,inline_memory=False)
    with TestClient(app) as client:
        assert client.post('/v1/chat/completions',headers=HEADERS,
            json=dict(model='test',messages=_history('First question'))).status_code==200
        with pytest.raises(PermissionError):
            memory.swap(boom)
        assert not memory.closed and memory.entries=={}
        reply=client.post('/v1/chat/completions',headers=dict(HEADERS,**{'x-memory-request-id':'turn2'}),
            json=dict(model='test',messages=_history('First question','Answer','Second question')))
        assert reply.status_code==200,reply.text
        entry=next(iter(memory.entries.values()))
        assert [e.text for e in entry.session.events() if e.role=='user']==['First question','Second question']


@pytest.mark.parametrize('status,tail',[(200,'complete'),(429,'http_error')])
def test_a_guarded_passthrough_records_the_body_it_actually_sent(tmp_path,monkeypatch,status,tail):
    """Finding 8: served_body is what went upstream, and both facts survive.

    What the guard decides is unchanged -- passthrough on a broken tool
    pairing is the sanctioned degraded mode -- only what the ledger says.
    """
    monkeypatch.setattr('memory_condense.interfaces.proxy_memory.guard_served_body',
                        lambda fmt,served:['tool_result without its tool_use'])
    sent=[]
    def upstream(request):
        sent.append(request.content)
        if status>=400:
            return httpx.Response(status,json={'error':{'message':'Busy'}})
        return httpx.Response(200,json=completion('openai','Answer'))
    app,memory,_=app_for(tmp_path,upstream,inline_memory=False)
    body=wire.encoded(dict(model='test',messages=_history('First question')))
    with TestClient(app) as client:
        reply=client.post('/v1/chat/completions',headers=HEADERS,content=body)
        assert reply.status_code==status,reply.text
        # Passthrough: the client own bytes went upstream verbatim.
        assert sent==[body]
        entry=next(iter(memory.entries.values()))
        with entry.session._connect() as db:
            served,guard=db.execute('SELECT served_body,served_guard FROM proxy_requests').fetchone()
            outcomes=[r[0] for r in db.execute('SELECT outcome FROM proxy_attempts ORDER BY attempt')]
        assert bytes(served)==body,'the ledger records the body that was sent'
        assert json.loads(guard)==['tool_result without its tool_use']
        assert outcomes==['invalid_served_body+'+tail],'both facts, neither overwritten'


def test_a_guarded_passthrough_keeps_its_verdict_through_an_envelope_failure(tmp_path,monkeypatch):
    monkeypatch.setattr('memory_condense.interfaces.proxy_memory.guard_served_body',
                        lambda fmt,served:['tool_result without its tool_use'])
    def upstream(request):
        return httpx.Response(200,json=completion('openai','not an inline envelope'))
    app,memory,_=app_for(tmp_path,upstream)  # inline memory on: the envelope is required
    with TestClient(app) as client:
        reply=client.post('/v1/chat/completions',headers=HEADERS,
            json=dict(model='test',messages=_history('First question')))
        assert reply.status_code==502,reply.text
        entry=next(iter(memory.entries.values()))
        with entry.session._connect() as db:
            outcomes=[r[0] for r in db.execute('SELECT outcome FROM proxy_attempts ORDER BY attempt')]
        assert outcomes==['invalid_served_body+invalid_inline_envelope']


def test_a_rotated_credential_keeps_the_conversations_memory():
    """A refreshed provider token must not fork the store.

    The namespace used to mix `sha256(authorization, x-api-key, ...)` in for
    tenant isolation. Provider OAuth tokens rotate -- and `proxy --print-env`
    tells Claude Code users to keep their claude.ai login, which is exactly
    such a token -- so every refresh handed a live conversation a fresh empty
    namespace and burned one of `max_sessions`. Access control over a network
    is a transport concern, not a storage-key concern.
    """
    conversation = {'x-memory-conversation-id': 'engineering'}
    rotated = [MemoryProxy.namespace('openai', dict(conversation, authorization=f'Bearer token-v{n}'))
               for n in range(3)]
    assert len(set(rotated)) == 1
    # No credential at all is the same conversation, too.
    assert MemoryProxy.namespace('openai', conversation) == rotated[0]
    # What still partitions: the conversation, and the provider.
    assert MemoryProxy.namespace('openai', {'x-memory-conversation-id': 'other'}) not in rotated
    assert MemoryProxy.namespace('anthropic', conversation) not in rotated
