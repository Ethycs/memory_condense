"""Failed local jobs escalate without losing source ownership or tenant isolation."""
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event
from types import SimpleNamespace

import httpx
import pytest
from starlette.testclient import TestClient

from memory_condense.application.chat_session import ChatSession
from memory_condense.interfaces.proxy_memory import MemoryProxy
from memory_condense.interfaces.proxy_server import ProxyConfig, build_app
from memory_condense.interfaces import proxy_memory_wire as wire
from memory_condense.runtime.artifacts import read
from memory_condense.runtime.summary_fallback import (
    LocalFirstSummaries, SummaryEscalationError, escalation_job,
)
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest, parse_spine_summary
from tests.test_chat_streaming import StreamingBackend
from tests.test_proxy_memory import completion


def raw_messages(text='Release r18 is planned, not deployed.'):
    return [dict(role='system',content='Compile source summaries.'),
            dict(role='user',content=json.dumps(dict(fragments=[dict(label='T0',role='user',fragment=text)])))]


def response(content, **extra):
    return dict(content=content, finish_reason='stop', request_sha256='local-receipt', **extra)


class Local:
    def __init__(self, result=None):
        self.result = result or response('invalid JSON')
        self.calls = self.starts = self.stops = 0

    def call(self, *args, **kwargs):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def start_reader(self):
        self.starts += 1

    def stop_reader(self):
        self.stops += 1


class API:
    identity = dict(provider='openai',model='frontier')

    def __init__(self, result=None):
        self.calls = []
        self.result = result or response(json.dumps(dict(atoms=[dict(label='T0',summary='Release r18 is planned')])) )

    def __call__(self, messages, **kwargs):
        self.calls.append((messages,kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_valid_local_generation_never_calls_api(tmp_path):
    value=response(json.dumps(dict(atoms=[dict(label='T0',summary='Release r18 is planned',
                                               support=['Release r18 is planned'])])))
    local,api=Local(value),API()
    runtime=LocalFirstSummaries(local,tmp_path,api)
    assert runtime.call('raw',raw_messages(),scope='live') is value
    assert local.calls==1 and not api.calls
    assert not list(tmp_path.rglob('reservation.json'))


@pytest.mark.parametrize('failure',[
    response('invalid JSON'),
    response('{"atoms":[]}',generated_finish_reason='length'),
    response('{"atoms":[{"label":"T0","summary":"Release r18 is planned"}]}',local_extractive_fallbacks=1),
    httpx.ReadTimeout('local server timed out'),
])
def test_failed_local_output_or_transport_escalates_and_survives_reopen(tmp_path,failure):
    local,api=Local(failure),API()
    runtime=LocalFirstSummaries(local,tmp_path,api)
    result=runtime.call('raw',raw_messages(),scope='live')
    assert result['summary_route']=='api_escalation'
    assert json.loads(result['content'])['atoms'][0]['support']==['Release r18 is planned']
    assert len(api.calls)==local.calls==1
    reopened=LocalFirstSummaries(local,tmp_path,api)
    assert reopened.call('raw',raw_messages(),scope='reopen')==result
    assert len(api.calls)==local.calls==1
    # A new source cannot reuse the old result.
    changed=raw_messages('Release r18 is planned, not deployed. Revised scope.')
    reopened.call('raw',changed,scope='changed')
    assert len(api.calls)==2


def test_raw_repair_escalates_only_residual_fragments(tmp_path):
    messages=[dict(role='system',content='Compile source summaries.'),
        dict(role='user',content=json.dumps(dict(fragments=[
            dict(label='T0',role='user',fragment='Alpha plan remains pending.'),
            dict(label='T1',role='user',fragment='Beta release shipped.')])))]
    class Sequence(Local):
        def __init__(self):
            super().__init__()
            self.results=iter((
                response(json.dumps(dict(atoms=[dict(label='T0',summary='Alpha plan remains pending.')]))),
                response('invalid retry'),
            ))
        def call(self,*args,**kwargs):
            self.calls+=1
            return next(self.results)
    local=Sequence()
    api=API(response(json.dumps(dict(atoms=[dict(label='T1',summary='Beta release shipped.')]))))
    result=LocalFirstSummaries(local,tmp_path,api).call('raw',messages,scope='live')
    atoms=json.loads(result['content'])['atoms']
    assert [(a['label'],a['summary']) for a in atoms]==[
        ('T0','Alpha plan remains pending.'),('T1','Beta release shipped.')]
    sent=json.loads(api.calls[0][0][1]['content'])['fragments']
    assert [f['label'] for f in sent]==['T1']
    assert local.calls==2 and len(api.calls)==1
    assert next(tmp_path.rglob('request.json')) and read(next(tmp_path.rglob('request.json')))['residual_labels']==['T1']


def test_raw_repair_stays_local_when_targeted_retry_succeeds(tmp_path):
    messages=[dict(role='system',content='Compile source summaries.'),
        dict(role='user',content=json.dumps(dict(fragments=[
            dict(label='T0',role='user',fragment='Alpha plan remains pending.'),
            dict(label='T1',role='user',fragment='Beta release shipped.')])))]
    class Sequence(Local):
        def __init__(self):
            super().__init__()
            self.results=iter((
                response(json.dumps(dict(atoms=[dict(label='T0',summary='Alpha plan remains pending.')]))),
                response(json.dumps(dict(atoms=[dict(label='T1',summary='Beta release shipped.')]))),
            ))
        def call(self,*args,**kwargs):
            self.calls+=1
            return next(self.results)
    local,api=Sequence(),API()
    result=LocalFirstSummaries(local,tmp_path,api).call('raw',messages,scope='live')
    assert result['summary_route']=='local_targeted_repair'
    assert [a['label'] for a in json.loads(result['content'])['atoms']]==['T0','T1']
    assert local.calls==2 and not api.calls
    assert list(tmp_path.rglob('local-repair.json')) and not list(tmp_path.rglob('reservation.json'))


def test_concurrent_repair_is_single_flight(tmp_path):
    local,api=Local(),API()
    runtime=LocalFirstSummaries(local,tmp_path,api)
    with ThreadPoolExecutor(max_workers=3) as pool:
        values=list(pool.map(lambda _:runtime.call('raw',raw_messages(),scope='live'),range(3)))
    assert values[0]==values[1]==values[2]
    assert local.calls==len(api.calls)==1


def test_corrupted_cached_summary_cannot_bypass_source_validation(tmp_path):
    local,api=Local(),API()
    runtime=LocalFirstSummaries(local,tmp_path,api)
    runtime.call('raw',raw_messages(),scope='live')
    accepted=next(tmp_path.rglob('accepted.json'))
    cached=read(accepted)
    cached['content']=json.dumps(dict(atoms=[dict(label='T0',summary='Release r99 deployed',support=[])]))
    accepted.write_text(json.dumps(cached),encoding='utf-8')
    with pytest.raises(SummaryEscalationError,match='Cached API summary needs review'):
        runtime.call('raw',raw_messages(),scope='reopen')
    assert local.calls==len(api.calls)==1


@pytest.mark.parametrize('failure',[
    response('{"atoms":[{"label":"T0","summary":"Release r99 was deployed"}]}'),
    response('{"atoms":[]}'),
    response('valid-looking but truncated',generated_finish_reason='length'),
    httpx.ReadTimeout('ambiguous upstream outcome'),
])
def test_failed_api_is_retained_and_never_silently_replaced_or_resent(tmp_path,failure):
    api=API(failure)
    with pytest.raises(SummaryEscalationError):
        LocalFirstSummaries(Local(),tmp_path,api).call('raw',raw_messages(),scope='live')
    with pytest.raises(SummaryEscalationError,match='Prior API summary attempt'):
        LocalFirstSummaries(Local(),tmp_path,api).call('raw',raw_messages(),scope='retry')
    assert len(api.calls)==1 and not list(tmp_path.rglob('accepted.json'))
    assert len(list(tmp_path.rglob('reservation.json')))==len(list(tmp_path.rglob('error.json')))==1


def test_merge_escalation_uses_only_typed_summaries_and_keeps_roles(tmp_path):
    request=SpineSummaryRequest('attached_context',(
        SpineSummaryFragment('assistant','2026-10-03','Proposed deployment; not executed.'),
        SpineSummaryFragment('system','2026-10-03','Tool reports two failed checks.')),
        user_spine='User requests a plan.',max_output_tokens=64)
    api=API(response(json.dumps(dict(extracts=[dict(index=0,quote='not executed.'),
                                              dict(index=1,quote='two failed checks.')]))))
    runtime=LocalFirstSummaries(Local(),tmp_path,api)
    result=runtime.call('merge',request.messages,scope='live',typed_request=request)
    summary=parse_spine_summary(result['content'],request)
    assert summary=='[assistant] not executed. [system] two failed checks.'
    sent=json.loads(api.calls[0][0][1]['content'])
    assert [f['summary'] for f in sent['fragments']]==[f.summary for f in request.fragments]
    assert sent['user_spine']==request.user_spine


def test_merge_api_near_quote_is_hydrated_to_an_exact_source_span(tmp_path):
    source = ('Hay dos tipos de "th" en inglés: el sonido "voiced th" como en "the", '
              '"this" y "those", y el sonido "unvoiced th" como en "think", "thought" y "bath".')
    request = SpineSummaryRequest('attached_context', (
        SpineSummaryFragment('assistant', '2026-10-11', source),),
        user_spine='How is th pronounced?', max_output_tokens=64)
    paraphrase = ('Hay dos tipos de "th" en inglés: el sonido "voiced th" y el sonido '
                  '"unvoiced th"')
    api = API(response('0\t'+paraphrase))
    result = LocalFirstSummaries(Local(), tmp_path, api).call(
        'merge', request.messages, scope='live', typed_request=request)
    summary = parse_spine_summary(result['content'], request)
    exact = summary.removeprefix('[assistant] ')
    assert exact in source
    assert 'voiced th' in exact and 'unvoiced th' in exact
    assert result['api_source_grounding'] is True


def test_bad_caller_job_does_not_trigger_api(tmp_path):
    api=API()
    runtime=LocalFirstSummaries(Local(),tmp_path,api)
    with pytest.raises(ValueError,match='typed fragments'):
        runtime.call('merge',[],scope='bad')
    assert not api.calls


def test_merge_contract_includes_metadata_and_rejects_duplicate_index_substitution(tmp_path):
    request=SpineSummaryRequest('attached_context',(
        SpineSummaryFragment('assistant','2026-10-03','Camera and lens; plans to digitize negatives.'),
        SpineSummaryFragment('system','2026-10-03','{"kind": "recall_feedback", "successful": true}')),
        user_spine='User asks about cameras.',max_output_tokens=64)
    api=API(response(json.dumps(dict(extracts=[dict(index=0,quote='Camera and lens'),
                                              dict(index=0,quote='plans to digitize negatives.')]))))
    with pytest.raises(SummaryEscalationError):
        LocalFirstSummaries(Local(),tmp_path,api).call('merge',request.messages,scope='live',typed_request=request)
    prompt=api.calls[0][0][0]['content']
    assert 'required output indices are [0, 1]' in prompt
    assert 'including JSON metadata, recall receipts and feedback' in prompt
    assert not list(tmp_path.rglob('accepted.json'))


def test_merge_escalation_prompt_uses_indexed_lines_without_json_contradiction():
    request = SpineSummaryRequest('attached_context',
        (SpineSummaryFragment('assistant', '2026-10-10', 'Please share the exact show title.'),),
        user_spine='Recommend similar shows.')
    messages, _budget = escalation_job('merge', request.messages, 128, request)
    instruction = messages[0]['content']
    assert 'returning only the indexed lines' in instruction
    assert 'returning only the JSON object' not in instruction


def test_raw_api_near_quote_ranks_to_an_exact_transcript_line(tmp_path):
    source = ('the fundamental idea is our\n1:32\nouter world is a reflection\n1:35\n'
              'the outer world\n1:50\nis merely a reflection formed by the\n1:54\nimage')
    messages = raw_messages(source)
    proposed = json.dumps(dict(atoms=[dict(
        label='T0', summary='the outer world is merely a reflection formed by the')]))
    api = API(response(proposed))
    result = LocalFirstSummaries(Local(), tmp_path, api).call('raw', messages, scope='live')
    atom = json.loads(result['content'])['atoms'][0]
    assert atom['summary'] == 'is merely a reflection formed by the'
    assert atom['summary'] in source
    assert result['api_source_grounding'] is True


def test_raw_api_near_quote_ranks_across_timestamped_lines(tmp_path):
    source = ('lows and you cannot escape it so the\n11:00\n'
              'key is you must always stay with the\n11:03\n'
              "identity and the Paradigm that you've\n11:05\n"
              'created for yourself and that is the best')
    messages = raw_messages(source)
    proposed = json.dumps(dict(atoms=[dict(
        label='T0', summary=("the key is you must always stay with the\n11:03\n"
                            "identity and the Paradigm that you've\n11:05\n"
                            'created for yourself'))]))
    result = LocalFirstSummaries(Local(), tmp_path, API(response(proposed))).call(
        'raw', messages, scope='live')
    atom = json.loads(result['content'])['atoms'][0]
    assert atom['summary'] in source
    assert 'key is you must always stay with the' in atom['summary']
    assert "identity and the Paradigm that you've" in atom['summary']
    assert result['api_source_grounding'] is True


def test_failed_local_proposal_is_reviewed_then_exactly_grounded(tmp_path):
    source = 'Release r18 remains planned, not deployed.'
    messages = raw_messages(source)
    local = Local(response(json.dumps(dict(atoms=[dict(
        label='T0', summary='Release r18 is still planned and has not deployed.')]))) )
    api = API(response(json.dumps(dict(atoms=[dict(
        label='T0', decision='supported', summary='Release r18 remains planned')]))) )

    result = LocalFirstSummaries(local, tmp_path, api).call('raw', messages, scope='live')

    atom = json.loads(result['content'])['atoms'][0]
    sent = json.loads(api.calls[0][0][1]['content'])['fragments'][0]
    assert result['summary_route'] == 'api_adjudication'
    assert atom['summary'] == 'Release r18 remains planned'
    assert atom['summary'] in source
    assert sent['rejected_summary'] == 'Release r18 is still planned and has not deployed.'
    assert 'review of excerpts rejected by the deterministic validator' in api.calls[0][0][0]['content']
    request = read(next(tmp_path.rglob('request.json')))
    assert request['rejected_proposal_labels'] == ['T0']


def test_local_lexical_grounding_avoids_bge_and_api(tmp_path):
    source = ("key is you must always stay with the\n11:03\n"
              "identity and the Paradigm that you've\n11:05\ncreated for yourself")
    messages = raw_messages(source)
    local = Local(response(json.dumps(dict(atoms=[dict(
        label='T0', summary=("key is you must always stay with the identity and the Paradigm "
                             "that you've created for yourself"))]))) )
    api = API()
    runtime = LocalFirstSummaries(local, tmp_path, api)
    runtime._grounder = lambda *_args, **_kwargs: pytest.fail('BGE ran before review')
    result = runtime.call('raw', messages, scope='live')
    atom = json.loads(result['content'])['atoms'][0]
    assert result['summary_route'] == 'local_lexical_grounding'
    assert result['lexical_source_grounding'] is True
    assert atom['summary'] in source
    assert not api.calls


def test_bge_uses_reviewed_cutoff_after_llm_adjudication(tmp_path):
    source = 'For example, creating short-form videos could appeal to younger viewers.'
    messages = raw_messages(source)
    proposal = 'create short-form videos'
    local = Local(response(json.dumps(dict(atoms=[dict(label='T0', summary=proposal)]))) )
    api = API(response(json.dumps(dict(atoms=[dict(
        label='T0', decision='supported', summary=proposal)]))) )
    reviewed_calls = []

    def ground(_raw, _proposal, *, reviewed=False):
        reviewed_calls.append(reviewed)
        return ('creating short-form videos could appeal to younger viewers'
                if reviewed else None)

    runtime = LocalFirstSummaries(local, tmp_path, api)
    runtime._grounder = ground
    result = runtime.call('raw', messages, scope='live')
    atom = json.loads(result['content'])['atoms'][0]
    assert result['summary_route'] == 'api_adjudication'
    assert atom['summary'] in source
    assert True in reviewed_calls and len(api.calls) == 1


def test_adjudication_prefilter_preserves_valid_later_batch_rows(tmp_path):
    messages = [dict(role='system', content='Compile source summaries.'),
        dict(role='user', content=json.dumps(dict(fragments=[
            dict(label='T0', role='user', fragment='Alpha remains planned.'),
            dict(label='T1', role='assistant', fragment='Beta shipped successfully.')]))) ]

    class Sequence(Local):
        def __init__(self):
            super().__init__()
            self.results = iter((
                response(json.dumps(dict(atoms=[
                    dict(label='T0', summary='Alpha is still being planned.'),
                    dict(label='T1', summary='Beta shipped successfully.')] ))),
                response(json.dumps(dict(atoms=[
                    dict(label='T0', summary='Alpha is still being planned.')] ))),
            ))

        def call(self, *args, **kwargs):
            self.calls += 1
            return next(self.results)

    local = Sequence()
    api = API(response(json.dumps(dict(atoms=[dict(
        label='T0', decision='supported', summary='Alpha remains planned.')]))) )
    result = LocalFirstSummaries(local, tmp_path, api).call('raw', messages, scope='live')

    atoms = json.loads(result['content'])['atoms']
    sent = json.loads(api.calls[0][0][1]['content'])['fragments']
    assert [(atom['label'], atom['summary']) for atom in atoms] == [
        ('T0', 'Alpha remains planned.'), ('T1', 'Beta shipped successfully.')]
    assert [fragment['label'] for fragment in sent] == ['T0']
    assert local.calls == 2 and len(api.calls) == 1


def test_adjudication_requires_decision_and_exact_support(tmp_path):
    source = 'Release r18 remains planned, not deployed.'
    messages = raw_messages(source)
    local = Local(response(json.dumps(dict(atoms=[dict(
        label='T0', summary='Release r18 is still planned and has not deployed.')]))) )
    # Even an exact quote is not admitted as an adjudication unless the remote
    # model records what it decided about the rejected proposal.
    api = API(response(json.dumps(dict(atoms=[dict(
        label='T0', summary='Release r18 remains planned')]))) )
    with pytest.raises(SummaryEscalationError):
        LocalFirstSummaries(local, tmp_path, api).call('raw', messages, scope='live')
    assert len(api.calls) == 1
    assert not list(tmp_path.rglob('accepted.json'))


def test_adjudication_cannot_accept_a_non_source_quote(tmp_path):
    source = 'Release r18 remains planned, not deployed.'
    messages = raw_messages(source)
    local = Local(response(json.dumps(dict(atoms=[dict(
        label='T0', summary='Release r18 is still planned and has not deployed.')]))) )
    api = API(response(json.dumps(dict(atoms=[dict(
        label='T0', decision='supported', summary='Release r18 deployed successfully')]))) )
    with pytest.raises(SummaryEscalationError):
        LocalFirstSummaries(local, tmp_path, api).call('raw', messages, scope='live')
    assert len(api.calls) == 1
    assert not list(tmp_path.rglob('accepted.json'))


def test_failed_paid_receipt_is_revalidated_without_a_second_api_call(tmp_path, monkeypatch):
    import memory_condense.runtime.local as local_module

    source = ('key is you must always stay with the\n11:03\n'
              "identity and the Paradigm that you've\n11:05\ncreated for yourself")
    messages = raw_messages(source)
    proposed = json.dumps(dict(atoms=[dict(
        label='T0', summary=("the key is you must always stay with the\n11:03\n"
                            "identity and the Paradigm that you've\n11:05\n"
                            'created for yourself'))]))
    api = API(response(proposed))
    original = local_module._ranked_exact_source_span
    monkeypatch.setattr(local_module, '_ranked_exact_source_span', lambda *_args: None)
    runtime = LocalFirstSummaries(Local(), tmp_path, api)
    with pytest.raises(SummaryEscalationError):
        runtime.call('raw', messages, scope='first')
    monkeypatch.setattr(local_module, '_ranked_exact_source_span', original)
    result = runtime.call('raw', messages, scope='resume')
    atom = json.loads(result['content'])['atoms'][0]
    assert atom['summary'] in source
    assert result['summary_route'] == 'api_receipt_revalidated'
    assert len(api.calls) == 1


def test_unavailable_local_summarizer_keeps_other_models_and_uses_api(tmp_path):
    class Missing(Local):
        def start_reader(self):
            self.starts+=1
            raise FileNotFoundError('missing local checkpoint')
    local,api=Missing(),API()
    runtime=LocalFirstSummaries(local,tmp_path,api)
    runtime.start_reader()
    runtime.start_reader()
    result=runtime.call('raw',raw_messages(),scope='live')
    assert local.starts==local.stops==1 and local.calls==0
    assert result['summary_route']=='api_escalation' and len(api.calls)==1


@pytest.mark.parametrize('provider,path',[('openai','/v1/chat/completions'),('anthropic','/v1/messages')])
def test_background_escalation_uses_conversation_model_and_credentials_after_reply(tmp_path,provider,path):
    release=Event()
    calls=[]

    class Factory:
        def __init__(self): self.runtimes={}
        def configure_summary_provider(self,namespace,api):
            if namespace in self.runtimes:
                self.runtimes[namespace].configure(api)
            else:
                self.runtimes[namespace]=LocalFirstSummaries(Local(),tmp_path/'summaries'/namespace,api)
        def __call__(self,namespace,directory):
            runtime=self.runtimes[namespace]
            class Backend(StreamingBackend):
                def prepare_exchange(self,events):
                    result=super().prepare_exchange(events)
                    assert release.wait(10)
                    for event in events:
                        if event.role=='user':
                            runtime.call('raw',raw_messages(event.text),scope='background')
                    return result
            return ChatSession(directory,namespace,Backend(),streaming=True)

    def upstream(request):
        payload=json.loads(request.content)
        is_summary=('Select routing excerpts' in json.dumps(payload))
        calls.append((request.headers.get('authorization'),payload,is_summary))
        assert request.url.path==path
        if is_summary:
            assert 'idempotency-key' not in request.headers
            fragments=json.loads(payload['messages'][-1]['content'])['fragments']
            content=json.dumps(dict(atoms=[dict(label=f['label'],summary=f['fragment']) for f in fragments]))
        else:
            assert request.headers['idempotency-key']=='visible-request'
            content='Recorded.'
        value=completion(provider,content)
        return httpx.Response(200,content=wire.answer_stream(provider,value) if payload.get('stream') else wire.encoded(value),
            headers={'content-type':'text/event-stream' if payload.get('stream') else 'application/json'})

    factory=Factory()
    memory=MemoryProxy(tmp_path/'memory',factory,inline_memory=False)
    app=build_app(config=ProxyConfig(mode='augment',upstreams={'openai':'https://example.test/v1','anthropic':'https://example.test'}),
        memory=memory,client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
    with TestClient(app) as client:
        try:
            for i in range(2):
                release.clear()
                headers={'authorization':f'Bearer private-key-{i}','x-memory-conversation-id':'same-name',
                         'x-memory-request-id':'same-request','idempotency-key':'visible-request',
                         'x-memory-provider':provider}
                payload=dict(model=f'frontier-{i}',messages=[dict(role='user',content=f'Release r{i} remains a plan.')],max_tokens=256)
                result=client.post(path,headers=headers,json=payload)
                assert result.status_code==200,result.text
                release.set()
                assert client.post('/_memory/flush',headers=headers).json()['pending_events']==0
        finally:
            release.set()
    assert len(calls)==4 and sum(c[2] for c in calls)==2
    for credential,payload,_ in calls:
        assert payload['model']=='frontier-'+credential[-1]
    assert len(factory.runtimes)==2
    for path in tmp_path.rglob('*'):
        if path.is_file():
            assert b'private-key-' not in path.read_bytes()


def test_native_factory_keeps_provider_binding_per_namespace(tmp_path,monkeypatch):
    from memory_condense.runtime import sessions
    monkeypatch.setattr(sessions,'LocalRuntime',lambda *a,**k:Local())
    monkeypatch.setattr(sessions,'ChatSession',lambda path,name,backend,**k:SimpleNamespace(backend=backend))
    factory=sessions.NativeProxySessions(tmp_path)
    a,b=API(),API()
    factory.configure_summary_provider('a',a)
    first=factory('a',tmp_path/'a')
    second=factory('b',tmp_path/'b')
    factory.configure_summary_provider('b',b)
    assert first.backend.runtime.api is a
    assert second.backend.runtime.api is b
    assert first.backend.runtime.local is second.backend.runtime.local


@pytest.mark.parametrize('enabled',[False,True])
def test_benchmark_routes_summary_failures_through_same_policy(tmp_path,monkeypatch,enabled):
    from memory_condense.runtime.local import LocalRuntime
    from tools.engineering_research_gateway_reader import GatewayReaderRuntime
    local,api=Local(),API()
    monkeypatch.setattr(LocalRuntime,'call',lambda self,*a,**k:local.call(*a,**k))
    runtime=GatewayReaderRuntime(tmp_path,gateway='https://example.test/v1',reader_model='frontier',
        judge_model='judge',summary_api_fallback=enabled)
    def remote(kind,messages,*,scope,max_tokens):
        assert kind=='summary_api'
        return api(messages,max_tokens=max_tokens)
    monkeypatch.setattr(runtime,'_remote_call',remote)
    result=runtime.call('raw',raw_messages(),scope='live')
    assert local.calls==1
    assert len(api.calls)==int(enabled)
    assert (result.get('summary_route')=='api_escalation') is enabled


def test_benchmark_summary_policy_reuses_local_embedding_and_router(tmp_path, monkeypatch):
    from memory_condense.runtime.local import LocalRuntime
    from tools.engineering_research_gateway_reader import GatewayReaderRuntime

    embedding, router = object(), object()
    monkeypatch.setattr(LocalRuntime, 'embedding', lambda self: embedding)
    monkeypatch.setattr(LocalRuntime, 'tactic_router', lambda self: router)
    runtime = GatewayReaderRuntime(tmp_path, gateway='https://example.test/v1',
        reader_model='frontier', judge_model='judge', summary_api_fallback=True)
    assert runtime.summary_runtime.embedding() is embedding
    assert runtime.summary_runtime.tactic_router() is router


def test_unconfigured_local_policy_remains_offline(tmp_path):
    local=Local()
    runtime=LocalFirstSummaries(local,tmp_path,None)
    runtime.start_reader()
    assert runtime.call('raw',raw_messages(),scope='offline') is local.result
    assert local.calls==local.starts==1 and not list(tmp_path.rglob('reservation.json'))


def test_reopened_proxy_cannot_silently_disable_escalation_before_provider_binding(tmp_path):
    local,api=Local(),API()
    runtime=LocalFirstSummaries(local,tmp_path,None,require_provider=True)
    with pytest.raises(SummaryEscalationError,match='bind the provider/model'):
        runtime.call('raw',raw_messages(),scope='flush-after-restart')
    assert not local.calls and not list(tmp_path.rglob('reservation.json'))
    runtime.configure(api)
    assert runtime.call('raw',raw_messages(),scope='resumed')['summary_route']=='api_escalation'
    assert len(api.calls)==1
