"""Small real-model proxy acceptance test; uses the shipping HTTP handler."""
import argparse
from copy import deepcopy
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import ssl
import time

import httpx
import truststore
from dotenv import load_dotenv
from starlette.testclient import TestClient

from memory_condense.interfaces.proxy_memory import MemoryProxy
from memory_condense.interfaces.proxy_server import ProxyConfig,build_app
from memory_condense.interfaces import proxy_memory_wire as wire
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.persistence import native_spine_incremental_store
from memory_condense.search.section_summary import RawSectionSpan
from tools.engineering_research_gateway import read,save,emit
from tools.native_proxy_runtime import NativeProxySessions


def audit(root,expected_completed=4,expected_failed=0):
    plan=read(root/'plan.json')
    directories=list((Path(plan['runtime_root'])/'sessions').iterdir())
    assert len(directories)==1
    directory=directories[0]
    with closing(sqlite3.connect(directory/'chat/chat-events.sqlite')) as journal, Database(directory/'store/memory/memory.db',read_only=True) as db:
        rows=journal.execute('SELECT event_id,role,text,metadata FROM events ORDER BY sequence').fetchall()
        turns=TranscriptStore(db).get_all()
        by_id={t.turn_id:t for t in turns}
        assert len(rows)==len(turns)
        for event_id,role,text,_ in rows:
            assert by_id[event_id].text==text
            assert by_id[event_id].role==(role if role in ('user','assistant','system') else 'system')
        pointers=0
        for (refs,) in journal.execute('SELECT refs FROM packets'):
            for ref in json.loads(refs):
                span=RawSectionSpan(**ref['span'])
                assert span==RawSectionSpan.from_turn(by_id[span.turn_id],start_char=span.start_char,end_char=span.end_char)
                pointers+=1
        applied=journal.execute('SELECT COUNT(*) FROM feedback WHERE successful=1 AND applied=1').fetchone()[0]
        learned=db.execute("SELECT COUNT(*) FROM consolidation_access_events WHERE event_id LIKE '_chat:feedback:%'").fetchone()[0]
        completed_outputs=[(event_id,json.loads(metadata)) for event_id,role,_,metadata in rows
            if role=='assistant' and json.loads(metadata).get('response',{}).get('proxy_wire')]
        expected_learning=sum(bool(json.loads(journal.execute('SELECT refs FROM packets WHERE packet_id=?',
            (metadata['io']['packet_id'],)).fetchone()[0])) for _,metadata in completed_outputs)
        assert len(completed_outputs)==expected_completed
        assert applied==learned==expected_learning and learned>0
        assert journal.execute('SELECT COUNT(*) FROM feedback WHERE applied=0').fetchone()[0]==0
        reopened=native_spine_incremental_store.load(directory/'store/memory'/native_spine_incremental_store.FILENAME,turns=turns)
        assert reopened.native.receipt['turn_count']==len(rows)
        states=journal.execute('SELECT state,COUNT(*) FROM proxy_requests GROUP BY state').fetchall()
        expected_states=[('complete',expected_completed)]+([('failed',expected_failed)] if expected_failed else [])
        assert states==expected_states
        committed,target=journal.execute('SELECT committed,target FROM ingestion_state').fetchone()
        assert committed==len(rows) and target is None
        # The source requirement is older than the recent window. Its exact
        # quote must be supplied by recall, rather than a resent old message.
        served=json.loads(journal.execute('SELECT served_body FROM proxy_requests WHERE request_id=?',
            ('proxy:'+hashlib.sha256(b'turn-2').hexdigest(),)).fetchone()[0])
        second_packet=journal.execute('SELECT text,refs FROM packets WHERE packet_id=?',
            ('proxy:'+hashlib.sha256(b'turn-2').hexdigest(),)).fetchone()
        assert '/ready' in second_packet[0] and '204' in second_packet[0]
        original='The Cedar service readiness contract is GET /ready, HTTP 204, with an empty body. Use Python standard library only.'
        assert not any(m.get('content')==original for m in served['messages'])
        assert any(m['role']=='system' and 'recorded user requirements' in m['content'] for m in served['messages'])
        inline_accepted=sum(metadata.get('inline_generation',{}).get('status')=='accepted' for _,metadata in completed_outputs)
    save(root/'audit.json',dict(events=len(rows),learning_updates=learned,source_pointers=pointers,
        native_and_parent_receipts_reopened=True,request_states=states,separate_process=True,
        completed_requests=expected_completed,failed_requests=expected_failed,
        probe_completed=expected_completed==plan['requests'] and expected_failed==0,
        pending_events=0,pending_feedback=0,older_contract_recalled=True,inline_pairs_accepted=inline_accepted,
        audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    emit(phase='proxy_audit_complete',events=len(rows),learning_updates=learned,pointers=pointers)


def live(root,runtime_root,gateway,model,tool_model=None):
    if root.exists() or runtime_root.exists(): raise ValueError('Use fresh probe directories')
    load_dotenv()
    key=os.environ['LITELLM_KEY']
    save(root/'plan.json',dict(runtime_root=str(runtime_root),gateway=gateway,model=model,tool_model=tool_model or model,
        real_local_models=True,real_provider=True,requests=4,seed_exchanges=14,
        scope='HTTP handler through Starlette TestClient; real outbound HTTPS; scripted read_file result; no performance benchmark',
        implementation={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__),Path('tools/native_proxy_runtime.py'),*Path('src/memory_condense/interfaces').glob('proxy_memory*.py'),
            Path('src/memory_condense/interfaces/proxy_server.py')]}))
    factory=NativeProxySessions(runtime_root)
    memory=MemoryProxy(runtime_root,factory)
    remote=httpx.AsyncClient(timeout=180,verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    app=build_app(config=ProxyConfig(mode='augment',upstreams={'openai':gateway,'anthropic':gateway}),memory=memory,client=remote)
    headers={'authorization':'Bearer '+key,'x-memory-conversation-id':'proxy-engineering-acceptance'}
    messages=[dict(role='system',content='You are implementing a small Python HTTP service. Follow the recorded user requirements.')]
    messages.extend([dict(role='user',content='The Cedar service readiness contract is GET /ready, HTTP 204, with an empty body. Use Python standard library only.'),
                     dict(role='assistant',content='Recorded: GET /ready returns HTTP 204 with an empty body, using the Python standard library.')])
    for i in range(13):
        messages.extend([dict(role='user',content=f'Engineering checkpoint {i+1}: keep changes small and preserve existing behavior.'),
                         dict(role='assistant',content=f'Checkpoint {i+1} acknowledged. I will preserve existing behavior.')])
    results=[]
    with TestClient(app) as client:
        def ask(text,ordinal,*,tools=None,stream=True):
            if text is not None: messages.append(dict(role='user',content=text))
            selected=tool_model if tools and tool_model else model
            payload=dict(model=selected,messages=deepcopy(messages),max_tokens=1024,stream=stream)
            if selected.startswith('codex_sdk/'):
                payload['reasoning_effort']='none'
            if tools: payload['tools']=tools
            before=time.perf_counter()
            reply=client.post('/v1/chat/completions',headers=dict(headers,**{'x-memory-request-id':f'turn-{ordinal}'}),json=payload)
            elapsed=time.perf_counter()-before
            save(root/f'response-{ordinal}.json',dict(status=reply.status_code,body=reply.text,elapsed_s=elapsed))
            assert reply.status_code==200,reply.text
            value=wire.assemble_stream('openai',reply.content) if stream else reply.json()
            message=wire.response_message('openai',value)
            assert '"memory"' not in wire.text_content(message)
            messages.append(message)
            results.append(dict(ordinal=ordinal,elapsed_s=elapsed,finish_reason=wire.finish_reason('openai',value),message=message))
            emit(phase='proxy_answered',ordinal=ordinal,elapsed_s=elapsed,finish_reason=wire.finish_reason('openai',value))
            return payload,reply,message
        ask('What readiness route, status code, and response body did we agree?',1)
        flushed=client.post('/_memory/flush',headers=headers)
        assert flushed.status_code==200,flushed.text
        save(root/'initial-flush.json',flushed.json())
        ask('Implement the agreed readiness handler as a Python BaseHTTPRequestHandler subclass. Return the code.',2)
        answer=results[-1]['message']['content']
        assert '/ready' in answer and ('204' in answer or 'HTTPStatus.NO_CONTENT' in answer)
        tools=[dict(type='function',function=dict(name='read_file',description='Read a project file.',
            parameters=dict(type='object',properties={'path':{'type':'string'}},required=['path'],additionalProperties=False)))]
        _,_,message=ask('Before proposing another change, call read_file for server.py.',3,tools=tools)
        calls=message.get('tool_calls',[])
        assert len(calls)==1 and calls[0]['function']['name']=='read_file'
        fixture='from http.server import BaseHTTPRequestHandler\nclass Handler(BaseHTTPRequestHandler):\n    def do_GET(self):\n        self.send_error(404)\n'
        messages.append(dict(role='tool',tool_call_id=calls[0]['id'],content=fixture))
        payload,reply,_=ask(None,4,tools=tools)
        answer=results[-1]['message']['content']
        assert '/ready' in answer and ('204' in answer or 'HTTPStatus.NO_CONTENT' in answer)
        repeated=client.post('/v1/chat/completions',headers=dict(headers,**{'x-memory-request-id':'turn-4'}),json=payload)
        assert repeated.content==reply.content
        flushed=client.post('/_memory/flush',headers=headers)
        assert flushed.status_code==200,flushed.text
        save(root/'final-flush.json',flushed.json())
        status=client.get('/_memory/health').json()
        assert status['pending_events']==status['pending_feedback']==status['failed_sessions']==0
        entry=next(iter(memory.entries.values()))
        assert len(entry.messages[-1][1]['content'])>0
        with entry.session._connect() as db:
            attempts=db.execute('SELECT outcome,COUNT(*) FROM proxy_attempts GROUP BY outcome').fetchall()
            assert attempts==[('complete',4)]
        save(root/'report.json',dict(requests=results,status=status,attempts=attempts,cached_retry_byte_identical=True,
            hidden_summaries=True,native_tool_cycle=True,older_contract_recalled=True,runtime_metrics=dict(factory.runtime.metrics)))
    assert factory.runtime.process is None or factory.runtime.process.poll() is not None
    emit(phase='proxy_probe_complete',provider_requests=4,models_stopped=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('live','audit'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--runtime-root',type=Path)
    parser.add_argument('--gateway',default='https://central-dev.zt:4000/v1')
    parser.add_argument('--model',default='codex_sdk/gpt-5.6-sol')
    parser.add_argument('--tool-model')
    parser.add_argument('--expected-completed',type=int,default=4)
    parser.add_argument('--expected-failed',type=int,default=0)
    args=parser.parse_args()
    if args.phase=='audit': audit(args.root,args.expected_completed,args.expected_failed)
    else: live(args.root,args.runtime_root,args.gateway,args.model,args.tool_model)
