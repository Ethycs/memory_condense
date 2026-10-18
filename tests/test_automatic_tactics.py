"""Automatic tactic dispatch, bounded hierarchy context and durable proxy IO."""
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from starlette.testclient import TestClient

from memory_condense.application.tactic_router import BGETacticRouter
from memory_condense.application.native_spine_policy import retrieval_limits
from memory_condense.application.chat_session import ChatEvent, ChatSession
from memory_condense.application.dialogue_retrieval import DialogueRetrievalPolicy
from memory_condense.domain._tokenizer import count_tokens
from tests.test_tactic_router import model
from tests.test_dialogue_retrieval import prepared, QUERY, DATED


def head(encoder, label):
    intercept = [-10., -10., -10.]
    intercept[('focused','overview','uncertain').index(label)] = 10.
    return BGETacticRouter(encoder, model(encoder, weights=[[0.,0.]]*3, intercept=intercept))


@pytest.mark.parametrize('label,expected,budget',[
    ('focused','cap8',3072),('overview','hierarchy_neighborhoods',6144),('uncertain','cap8',3072)])
def test_auto_dispatch_reuses_query_vector_and_preserves_scope(label,expected,budget):
    history,memory,encoder = prepared(future=True)
    memory.configure_tactics(head(encoder,label))
    result = memory.retrieve(QUERY,DATED,**retrieval_limits(memory,{}))
    assert result.tactic['selected_tactic']==expected
    assert result.hydration.max_context_tokens==budget
    assert encoder.calls==[QUERY]
    if expected=='hierarchy_neighborhoods':
        assert result.hierarchy_context.eligible_sections==3
        assert result.hierarchy_context.omitted_sections==0
        assert 'STORED HIERARCHY SUMMARIES' in result.hierarchy_context.text
        assert {r['source_id'] for r in result.hierarchy_context.references}<=set(result.routing.baseline.eligible_source_ids)
    else:
        assert result.hierarchy_context is None
    for section in result.hydration.sections:
        for evidence in section.evidence:
            s=evidence.span
            assert evidence.text==history.get_turn(s.turn_id).text[s.start_char:s.end_char]


def test_ambiguous_followup_uses_context_once_but_clear_question_does_not(monkeypatch):
    _,memory,encoder=prepared()
    # The question vector [1,0] is uncertain; contextual [-1,0] is overview.
    classifier=BGETacticRouter(encoder,model(encoder,weights=[[0,0],[-5,0],[5,0]]))
    memory.configure_tactics(classifier)
    calls=[]
    def embed(text):
        calls.append(text)
        return np.array([-1.,0.] if text.startswith('Recent conversation:') else [1.,0.])
    monkeypatch.setattr(encoder,'embed_query',embed)
    result=memory.retrieve(QUERY,DATED,recent=['Assess the full project history.'])
    assert result.tactic['initial_decision']['abstained']
    assert result.tactic['selected_tactic']=='hierarchy_neighborhoods'
    assert result.tactic['contextualized']
    assert len(calls)==2
    memory.configure_tactics(head(encoder,'focused'))
    calls.clear()
    assert memory.retrieve(QUERY,DATED,recent=['Broad earlier discussion.']).tactic['selected_tactic']=='cap8'
    assert calls==[QUERY]


def test_hierarchy_budget_omits_whole_records_and_reports_coverage():
    _,memory,_=prepared()
    context=memory.dialogue.hierarchy_context
    scope=tuple(s.source_id for s in memory.router.hierarchy.sections)
    for budget in (0,10,200,500,24576):
        result=context.render(scope,(),max_tokens=budget)
        assert count_tokens(result.text)==result.token_count<=budget
        assert len(result.references)+result.omitted_sections==result.eligible_sections
        for ref in result.references:
            section=next(s for s in memory.router.hierarchy.sections if s.section_id==ref['section_id'])
            assert section.summary in result.text
    assert context.render((),(),max_tokens=24576).references==()
    with pytest.raises(ValueError):
        context.render(scope,(),max_tokens=-1)


def test_summary_context_uses_transcript_positions_instead_of_hashed_id_order():
    from memory_condense.application.hierarchy_context import PreparedHierarchyContext
    history,memory,_=prepared()
    positions={a.spans[0].turn_id:i for i,a in enumerate(history.atoms)}
    context=PreparedHierarchyContext(memory.router,turn_positions=positions)
    result=context.render(tuple(s.source_id for s in memory.router.hierarchy.sections),(),max_tokens=24576)
    assert result.chronological
    assert [r['first_turn_position'] for r in result.references]==sorted(r['first_turn_position'] for r in result.references)
    assert all(r['first_turn_position']==positions[r['first_turn_id']] for r in result.references)
    with pytest.raises(ValueError,match='positions'):
        PreparedHierarchyContext(memory.router,turn_positions={})


def test_forced_baseline_and_oversized_request_do_not_invoke_classifier():
    _,memory,encoder=prepared()
    memory.configure_tactics(head(encoder,'overview'))
    result=memory.retrieve(QUERY,DATED,augment=False)
    assert result.tactic is None and result.hierarchy_context is None
    query='word '*1000
    dated='[Question asked at 2026/09/12 (Saturday) 12:00] '+query
    result=memory.retrieve(query,dated)
    assert result.tactic['reason']=='query_budget'
    assert result.tactic['selected_tactic']=='cap8' and result.hierarchy_context is None
    assert encoder.calls==[QUERY,query]


class AutoBackend:
    """Real native ingest, hydration and learning with a deterministic scope head."""
    def __new__(cls,path,*,history=False):
        from tests.test_chat_native import NativeFixtureBackend
        class Backend(NativeFixtureBackend):
            recall_accepts_recent=True
            def sync(self,events):
                super().sync(events)
                memory=self.app._load_native_spine()[2]
                matrix=np.asarray([[1.,0.]]*len(memory.router.leaf_atoms),dtype=np.float32)
                memory.configure_dialogue(DialogueRetrievalPolicy(),matrix,embedding_identity=memory._query_embedding_identity,
                    turn_positions={t.turn_id:i for i,t in enumerate(self.app.transcript.native_snapshot())})
                if history:
                    from tests.test_decision_history import history_router
                    memory.configure_tactics(history_router(self.encoder))
                else:
                    memory.configure_tactics(head(self.encoder,'overview'))
            def recall(self,query,*,recent=()):
                self.last_recent=recent
                from memory_condense.application.chat_native import native_packet
                return native_packet(self.app,query,'[Question asked at 2099/01/01 (Thu) 12:00] '+query,self.events,recent=recent)
        return Backend(path)


@pytest.mark.parametrize('history',[False,True])
def test_real_chat_journals_summary_pointers_and_only_exact_spans_get_learning_credit(tmp_path,history):
    from tests.test_chat_native import graph
    backend=AutoBackend(tmp_path/'memory',history=history)
    with ChatSession(tmp_path/'chat','one',backend) as chat:
        chat.ingest_many([ChatEvent('u1','user','Use one IO stream.'),
                          ChatEvent('a1','assistant','Preserve original pointers.'),
                          ChatEvent('now','user','What have we tried?')])
        first=chat.recall('What have we tried?',packet_id='p',input_event_id='now')
        assert backend.last_recent==('Use one IO stream.','Preserve original pointers.')
        assert 'STORED HIERARCHY SUMMARIES' in first.text and '<MEMORY ' in first.text
        trace=chat.event('_chat:recall:p').metadata['_chat']['retrieval']
        assert trace['mode']==('decision_history' if history else 'hierarchy_neighborhoods')
        assert trace['tactic']['selected_tactic']==trace['mode']
        if history:
            assert trace['decision_history']['annotations']
        assert trace['hierarchy_context']['references']
        assert trace['continuity']['evidence']
        assert 'Fresh-chat handoff:' in first.text
        assert all('span' in r and 'label' not in r for r in first.references)
        assert chat.recall('What have we tried?',packet_id='p',input_event_id='now')==first
        chat.feedback('p',successful=True)
        chat.flush()
        assert set(dict(graph(tmp_path/'memory')['nodes']))<={'u1','a1','now'}


@pytest.mark.parametrize('provider,path',[('openai','/v1/chat/completions'),('anthropic','/v1/messages')])
@pytest.mark.parametrize('history',[False,True])
def test_proxy_delivers_combined_packet_and_replays_without_new_recall(tmp_path,provider,path,history):
    from memory_condense.interfaces.proxy_memory import MemoryProxy
    from memory_condense.interfaces.proxy_server import ProxyConfig,build_app
    from tests.test_proxy_memory import completion,HEADERS
    sent=[]
    def factory(namespace,directory):
        return ChatSession(directory/'chat',namespace,AutoBackend(directory/'memory',history=history))
    def upstream(request):
        body=json.loads(request.content)
        sent.append(body)
        text=json.dumps(body)
        assert 'STORED HIERARCHY SUMMARIES' in text
        assert 'CONTINUITY OUTPUT CONVENTIONS' in text
        assert ('EXACT HISTORY EVIDENCE' if history else 'EXACT NEIGHBORHOOD EVIDENCE') in text
        return httpx.Response(200,json=completion(provider,'The evidence distinguishes plans and completed work.'))
    memory=MemoryProxy(tmp_path/'proxy',factory,inline_memory=False)
    app=build_app(config=ProxyConfig(mode='augment',upstreams={'openai':'https://example.test/v1','anthropic':'https://example.test'}),
        memory=memory,client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
    payload=dict(model='test',max_tokens=256,messages=[dict(role='user',content='Use one IO stream.'),
        dict(role='assistant',content='Preserve original pointers.'),dict(role='user',content='What have we tried?')])
    with TestClient(app) as client:
        response=client.post(path,headers=HEADERS,json=payload)
        assert response.status_code==200,response.text
        assert client.post(path,headers=HEADERS,json=payload).content==response.content
        assert len(sent)==1


@pytest.mark.parametrize('mode',['auto','cap8','dialogue','hierarchy_neighborhoods','decision_history'])
def test_proxy_default_and_forced_modes_reach_native_sessions(tmp_path,monkeypatch,mode):
    import uvicorn
    from memory_condense.cli import main
    from memory_condense.runtime import sessions
    from memory_condense.runtime.config import RuntimeAssets
    from memory_condense.interfaces import proxy_server
    monkeypatch.setattr(RuntimeAssets,'require_files',lambda *a,**k:None)
    monkeypatch.setattr(sessions,'LocalRuntime',lambda *a,**k:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(sessions,'ChatSession',lambda path,name,backend,**k:SimpleNamespace(backend=backend))
    observed=[]
    def serve(config,memory,locator=None,**_):
        assert locator is not None
        assert memory.status()['retrieval']['mode']==mode
        backend=memory.factory('one',tmp_path/'one').backend
        assert backend.retrieval_policy==mode
        assert (backend.dialogue_policy is None)==(mode=='cap8')
        observed.append(True)
    monkeypatch.setattr(proxy_server,'build_app',serve)
    monkeypatch.setattr(uvicorn,'run',lambda *a,**k:None)
    args=['proxy','--data-dir',str(tmp_path/'service')]
    if mode!='auto':
        args+=['--memory-retrieval-policy',mode]
    assert main(args)==0 and observed==[True]


def test_automatic_published_snapshot_survives_append_and_failed_preparation(tmp_path,monkeypatch):
    from tests.test_engineering_research_resident import setup,STAMP
    from memory_condense.application import tactic_router
    backend,events,_,_=setup(tmp_path,monkeypatch)
    backend.retrieval_policy='auto'
    backend.dialogue_policy=DialogueRetrievalPolicy()
    monkeypatch.setattr(tactic_router,'BGETacticRouter',lambda encoder:head(encoder,'overview'))
    try:
        backend.sync(events)
        first=backend.recall_published('What have we tried?')
        assert first['retrieval']['mode']=='hierarchy_neighborhoods'
        classifier=backend._tactic_classifier
        prepare=backend._prepare_dialogue
        def fail(memory):
            raise ValueError('simulated preparation failure')
        monkeypatch.setattr(backend,'_prepare_dialogue',fail)
        events.append(ChatEvent('next','user','Keep the deployment a proposal.',STAMP))
        with pytest.raises(ValueError,match='preparation failure'):
            backend.sync(events)
        during=backend.recall_published('What have we tried?')
        assert during['text']==first['text']
        assert during['published_events']==len(events)-1
        assert during['retrieval']['hierarchy_context']==first['retrieval']['hierarchy_context']
        monkeypatch.setattr(backend,'_prepare_dialogue',prepare)
        backend.sync(events)
        after=backend.recall_published('What have we tried?')
        assert after['published_events']==len(events)
        assert after['retrieval']['hierarchy_sha256']!=first['retrieval']['hierarchy_sha256']
        assert backend._tactic_classifier is classifier
    finally:
        backend.close()
