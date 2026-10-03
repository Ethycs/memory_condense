"""Exercise the installed package over HTTP with real local models and a fixture provider."""
import argparse
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.abc
import json
from pathlib import Path
import socket
import sys
import threading
import time


class NoResearch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('tools','tests'):
            raise AssertionError('Installed runtime tried to import '+fullname)


sys.meta_path.insert(0,NoResearch())
import httpx
import uvicorn
import memory_condense
from memory_condense.interfaces.proxy_memory import MemoryProxy
from memory_condense.interfaces.proxy_server import ProxyConfig,build_app
from memory_condense.runtime.config import RuntimeAssets
from memory_condense.runtime.sessions import NativeProxySessions


ANSWER='GET /ready returns HTTP 204 with an empty body.'


def run(assets_dir,root,expected_package_root):
    installed=Path(memory_condense.__file__).resolve()
    assert installed.is_relative_to(expected_package_root.resolve()),installed
    root.mkdir(parents=True,exist_ok=False)
    sent=[]
    class Provider(BaseHTTPRequestHandler):
        def log_message(self,*_): pass
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            sent.append(request)
            answer='Ready for the next step.' if len(sent)==1 else ANSWER
            message=dict(role='assistant',content=answer)
            finish='stop'
            if request.get('tools') and request['messages'][-1]['role']=='user':
                message.update(content=None,tool_calls=[dict(id='read1',type='function',
                    function=dict(name='read_file',arguments='{"path":"server.py"}'))])
                finish='tool_calls'
            elif not request.get('tools'):
                user=json.loads(request['messages'][-1]['content'].split('JSON string):\n')[-1])
                message['content']=json.dumps(dict(answer=answer,memory={
                    'user':dict(summary=user,support=[user]),
                    'assistant':dict(summary=answer,support=[answer])}))
            body=json.dumps(dict(id='fixture',object='chat.completion',model='fixture',
                choices=[dict(index=0,message=message,finish_reason=finish)])).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    provider=ThreadingHTTPServer(('127.0.0.1',0),Provider)
    provider_thread=threading.Thread(target=provider.serve_forever,daemon=True);provider_thread.start()
    factory=NativeProxySessions(root/'runtime',assets=RuntimeAssets.resolve(assets_dir))
    memory=MemoryProxy(root/'runtime',factory)
    upstream=f'http://127.0.0.1:{provider.server_port}'
    app=build_app(config=ProxyConfig(mode='augment',upstreams={'openai':upstream,'anthropic':upstream}),memory=memory)
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='warning'))
    service=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);service.start()
    messages=[dict(role='system',content='Preserve recorded user requirements.'),
              dict(role='user',content=ANSWER),dict(role='assistant',content='Recorded.')]
    for i in range(13):
        messages.extend([dict(role='user',content=f'Checkpoint {i}: preserve behavior.'),
                         dict(role='assistant',content='Acknowledged.')])
    outputs=[]
    headers={'x-memory-conversation-id':'installed-acceptance','authorization':'Bearer fixture'}
    try:
        deadline=time.monotonic()+30
        while not server.started:
            if time.monotonic()>deadline or not service.is_alive(): raise RuntimeError('Proxy failed to start')
            time.sleep(.1)
        with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=600,trust_env=False) as client:
            def ask(text,ordinal,tools=None):
                if text is not None: messages.append(dict(role='user',content=text))
                payload=dict(model='fixture',messages=deepcopy(messages),max_tokens=512)
                if tools: payload['tools']=tools
                response=client.post('/v1/chat/completions',headers=dict(headers,**{'x-memory-request-id':str(ordinal)}),json=payload)
                assert response.status_code==200,response.text
                message=response.json()['choices'][0]['message']
                assert '"memory"' not in (message.get('content') or '')
                messages.append(message);outputs.append(message)
                print(json.dumps({'phase':'installed_answer','ordinal':ordinal}),flush=True)
                return payload,response,message
            ask('Start the next engineering checkpoint.',1)
            response=client.post('/_memory/flush',headers=headers)
            assert response.status_code==200,response.text
            ask('Recall the original readiness route and status.',2)
            assert any('Recalled conversation evidence' in str(m.get('content','')) and '/ready' in str(m.get('content',''))
                       for m in sent[1]['messages'])
            assert not any(m.get('content')==ANSWER for m in sent[1]['messages'])
            tools=[dict(type='function',function=dict(name='read_file',parameters={'type':'object','properties':{'path':{'type':'string'}}}))]
            _,_,call=ask('Read server.py before continuing.',3,tools)
            assert call['tool_calls'][0]['id']=='read1'
            messages.append(dict(role='tool',tool_call_id='read1',content='class Handler: pass'))
            payload,response,_=ask(None,4,tools)
            assert sent[-1]['messages'][-1]['role']=='tool'
            duplicate=client.post('/v1/chat/completions',headers=dict(headers,**{'x-memory-request-id':'4'}),json=payload)
            assert duplicate.content==response.content and len(sent)==4
            flushed=client.post('/_memory/flush',headers=headers)
            assert flushed.status_code==200,flushed.text
            health=client.get('/_memory/health').json()
            assert health['pending_events']==health['pending_feedback']==health['failed_sessions']==0
            report=dict(installed_module=str(installed),real_http=True,real_local_models=True,
                provider='controlled loopback fixture; not an answer-quality evaluation',
                provider_requests=len(sent),native_tool_cycle=True,duplicate_replayed=True,
                older_contract_recalled=True,health=health,outputs=outputs)
            (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:
        server.should_exit=True;service.join(600)
        provider.shutdown();provider.server_close();sock.close()
        assert not service.is_alive(),'Proxy did not drain and close'
        assert factory.runtime.process is None or factory.runtime.process.poll() is not None
    print(json.dumps({'phase':'installed_proxy_complete','models_stopped':True}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--expected-package-root',type=Path,required=True)
    args=parser.parse_args()
    run(args.assets_dir,args.root,args.expected_package_root)
