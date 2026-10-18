"""Experimental mechanisms cannot weaken the shipping native source boundary."""
from dataclasses import replace
import json
from types import MappingProxyType, SimpleNamespace

import pytest

from memory_condense.application.native_experiments import (
    NativeExperimentPolicy, PreparedNativeExperiments, SummaryGraph)
from memory_condense.application.native_spine_retrieval import ResidentNativeSpineMemory
from memory_condense.application.chat_native import native_packet, learn_native_packet
from memory_condense.application.chat_session import ChatEvent
from memory_condense.associations.heat_diffusion import diffuse_scalar_heat
from tests.test_native_spine_routing import fixture, QUERY, DATED


LIMITS = dict(max_direct=1,lexical_reserve=0,max_additions=0)


def prepared(*modules, future=False, attention=None, reinjection=None, spectral=None, **policy):
    history,semantic,hierarchy,encoder = fixture(future=future)
    memory = ResidentNativeSpineMemory(semantic,hierarchy,encoder=encoder,load_turn=history.get_turn)
    a,b = history.atoms[0].section_id,history.atoms[2].section_id
    edges = {a:((b,.9),),b:((a,.9),)}
    if future:
        edges[a] += ((history.atoms[3].section_id,1.),)
    graph = SummaryGraph(MappingProxyType(edges),MappingProxyType(dict(edges=len(edges))))
    memory.experiments = PreparedNativeExperiments(memory.router,
        NativeExperimentPolicy(modules=modules,**policy),graph=graph,attention=attention,
        reinjection=reinjection,spectral=spectral)
    return history,memory,encoder


@pytest.mark.parametrize('value',[[],{'modules':'heat_diffusion'},{'modules':['missing']},
    {'modules':['heat_diffusion','heat_diffusion']},{'extra_context_tokens':True},
    {'candidate_limit':100},{'max_qwen_passes':0},{'qwen_layer':5},
    {'modules':['cav_associations']},{'modules':['cav_reinjection']},{'unexpected':False}])
def test_policy_rejects_invalid_activation(value):
    with pytest.raises(ValueError):
        NativeExperimentPolicy.from_mapping(value)


@pytest.mark.parametrize('module',['learned_associations','heat_diffusion'])
def test_real_mechanism_adds_evidence_preserves_control_and_date_scope(module):
    history,memory,encoder = prepared(module,future=True)
    experiment = memory.experiments
    memory.experiments = None
    control = memory.retrieve(QUERY,DATED,**LIMITS)
    memory.experiments = experiment
    result = memory.retrieve(QUERY,DATED,**LIMITS)
    assert result.hydration == control.hydration
    assert result.routing == control.routing
    assert encoder.calls==[QUERY,QUERY]
    extra = result.experiments.hydration
    assert [s.section for s in extra.sections]==[history.atoms[2]]
    assert extra.sections[0].evidence[0].text==history.get_turn(history.atoms[2].spans[0].turn_id).text
    assert result.experiments.receipt['query_qwen_passes']==0
    assert extra.context_token_count<=experiment.policy.extra_context_tokens


def test_routing_has_no_raw_access_and_addition_budget_cannot_backfill_control():
    _,memory,_ = prepared('learned_associations',extra_context_tokens=0)
    old = memory.load_turn
    calls=[]
    memory.load_turn=lambda tid:(calls.append(tid),old(tid))[1]
    route = memory.router.route_vector(QUERY,DATED,[1.,0.],
        embedding_identity=memory.router.semantic.embedding_identity,**LIMITS)
    selection = memory.experiments.select(QUERY,[1.,0.],route)
    assert calls==[] and selection.plan.routes
    result = memory.retrieve(QUERY,DATED,**LIMITS)
    assert not result.experiments.hydration.sections
    assert result.hydration.sections


def test_optional_gpu_failure_is_visible_and_does_not_change_control():
    class Attention:
        def experiment(self,module,query,summaries,**kwargs):
            assert all(s.summary.startswith('Stored summary') for s in summaries)
            return [],dict(status='model_error',error_type='RuntimeError',model_passes=1)
    _,memory,_ = prepared('qwen_associations',attention=Attention())
    result = memory.retrieve(QUERY,DATED,**LIMITS)
    assert result.experiments.receipt['arms'][0]['status']=='model_error'
    assert result.experiments.receipt['query_qwen_passes']==1
    assert result.hydration.sections and not result.experiments.hydration.sections


def test_foreign_gpu_proposal_is_rejected_before_hydration():
    class Attention:
        def experiment(self,*args,**kwargs):
            return [('foreign',1.)],dict(status='selected',model_passes=1)
    _,memory,_ = prepared('qwen_associations',attention=Attention())
    memory.load_turn=lambda _:pytest.fail('Foreign proposal reached raw hydration')
    with pytest.raises(ValueError,match='namespace'):
        memory.retrieve(QUERY,DATED,**LIMITS)


def test_precomputed_cav_reinjection_uses_bge_seeds_without_qwen():
    class Reinjection:
        def select(self,seeds,*,allowed,candidates,limit):
            assert seeds and limit==16
            assert candidates
            key=candidates[0][0]
            return [(key,.4)],dict(status='selected',model_passes=0,raw_reads=0,
                query_qwen_passes=0,retained_transformer_token_state_bytes=0,
                query_compute_cells=12,dense_attention_cells=16,compute_cell_ratio=.75)
    _,memory,_=prepared('cav_reinjection',reinjection=Reinjection(),cav_bundle='fixture.json')
    result=memory.retrieve(QUERY,DATED,**LIMITS)
    arm=result.experiments.receipt['arms'][0]
    assert arm['status']=='selected'
    assert arm['model_passes']==0
    assert result.experiments.receipt['query_qwen_passes']==0
    assert result.experiments.hydration.sections


def test_spectral_reinjection_uses_precomputed_graph_without_qwen():
    class Spectral:
        def select(self,seeds,*,allowed,candidates,limit):
            assert seeds and candidates and limit>=16
            key=candidates[0][0]
            return [(key,.4)],dict(status='selected',model_passes=0,raw_reads=0,
                query_qwen_passes=0,retained_transformer_token_state_bytes=0,
                query_compute_cells=64,dense_attention_cells=0)
    _,memory,_=prepared('spectral_reinjection',spectral=Spectral())
    result=memory.retrieve(QUERY,DATED,**LIMITS)
    arm=result.experiments.receipt['arms'][0]
    assert arm['status']=='selected' and arm['model_passes']==0
    assert result.experiments.receipt['query_qwen_passes']==0
    assert result.experiments.hydration.sections


def test_stale_experimental_raw_is_rejected_independently():
    history,memory,_ = prepared('learned_associations')
    tid=history.atoms[2].spans[0].turn_id
    memory.load_turn=lambda key:history.get_turn(key).model_copy(update={'text':'tampered'}) if key==tid else history.get_turn(key)
    result=memory.retrieve(QUERY,DATED,**LIMITS)
    assert result.hydration.sections and not result.experiments.hydration.sections
    assert result.experiments.hydration.diagnostics


def test_packet_marks_real_originals_ineligible_for_recursive_learning():
    history,memory,_=prepared('learned_associations')
    turns=[history.get_turn(a.spans[0].turn_id) for a in history.atoms]
    events=[ChatEvent(t.turn_id,t.role,t.text,t.created_at.isoformat()) for t in turns]
    view=SimpleNamespace(transcript=SimpleNamespace(get_turn=history.get_turn),
        retrieve_native_spine=lambda q,d,**kw:memory.retrieve(q,d,**LIMITS))
    packet=native_packet(view,QUERY,DATED,events)
    extra=[r for r in packet['references'] if not r.get('learning_eligible',True)]
    assert len(extra)==1 and extra[0]['independent']
    assert extra[0]['experimental_route']=='learned_associations'
    view._db=SimpleNamespace(execute=lambda *_:pytest.fail('Graph-derived evidence earned learning credit'))
    assert learn_native_packet(view,SimpleNamespace(references=extra),events,access_event_id='no-loop') is None


def test_heat_conserves_mass_and_keeps_dangling_nodes():
    heat=diffuse_scalar_heat({'a':1.},{'a':(('b',.8),('c',.2))},hops=3,max_nodes=3)
    assert heat.total_heat==pytest.approx(1.)
    assert {n.chunk_id for n in heat.nodes}=={'a','b','c'}
    assert next(n.heat for n in heat.nodes if n.chunk_id=='b')>next(n.heat for n in heat.nodes if n.chunk_id=='c')


def test_existing_scalar_store_projects_to_current_summary_addresses(tmp_path):
    from tests.test_chat_native import NativeFixtureBackend
    backend=NativeFixtureBackend(tmp_path/'memory')
    try:
        backend.sync((ChatEvent('a','user','Use incremental publication.'),
                      ChatEvent('b','assistant','Keep the exact source pointers.')))
        chunks=[r[0] for r in backend.app._db.execute('SELECT chunk_id FROM chunks').fetchall()]
        for event in ('access1','access2'):
            backend.app.observe_context_access([],chunks,access_event_id=event)
        memory=backend.app._load_native_spine()[2]
        policy=NativeExperimentPolicy(modules=('learned_associations',))
        snapshot=SummaryGraph.capture(backend.app,memory.router,policy)
        assert snapshot.receipt['edges']==1
        assert set(snapshot.adjacency)==set(memory.router.atoms)
        # A new observation changes a new snapshot, never the admitted one.
        old=dict(snapshot.receipt)
        backend.app.observe_context_access([],chunks,access_event_id='access3')
        newer=SummaryGraph.capture(backend.app,memory.router,policy)
        assert dict(snapshot.receipt)==old
        assert newer is not snapshot
    finally:
        backend.close()


def test_profile_parser_and_cli_enable_only_explicit_modules(tmp_path):
    from memory_condense.runtime.config import experiment_policy
    assert experiment_policy() is None
    path=tmp_path/'experiments.json'
    policy=NativeExperimentPolicy(modules=('heat_diffusion',))
    path.write_text(json.dumps(policy.identity_payload()),encoding='utf-8')
    assert experiment_policy(path)==policy


def test_qwen_capture_stops_before_unused_tail_and_cleans_hooks():
    import torch
    from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
    encoder=object.__new__(Qwen3PrefixEncoder)
    first=torch.nn.Identity()
    class Unused(torch.nn.Module):
        def forward(self,*args,**kwargs):
            pytest.fail('Executed unused Qwen tail')
    model=SimpleNamespace(layers=[first,Unused()])
    encoder.model=model
    def forward(**inputs):
        value=inputs['value']
        for layer in model.layers:
            value=layer(value)
    encoder._forward=forward
    captured=[]
    handle=first.register_forward_hook(lambda _m,_a,o:captured.append(o))
    value=torch.tensor([3.])
    try:
        encoder._forward_through_layer(0,value=value)
        assert captured==[value] and len(first._forward_hooks)==1
    finally:
        handle.remove()


def test_qwen_attention_values_are_mask_pooled_and_release_hooks():
    import torch
    from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder

    class ToyTokenizer:
        padding_side = 'left'

        def __call__(self, texts, **_kwargs):
            width=max(len(text) for text in texts)
            rows=[]
            masks=[]
            for row,text in enumerate(texts,1):
                values=[float(row*10+column) for column in range(len(text))]
                rows.append(values+[0.]*(width-len(text)))
                masks.append([1]*len(text)+[0]*(width-len(text)))
            return {
                'input_ids':torch.tensor(rows),
                'attention_mask':torch.tensor(masks),
            }

    class ToyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim=2
            self.o_proj=torch.nn.Identity()

        def forward(self,value):
            return self.o_proj(value)

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn=ToyAttention()

        def forward(self,value):
            return self.self_attn(value)

    encoder=object.__new__(Qwen3PrefixEncoder)
    encoder._torch=torch
    encoder.device=torch.device('cpu')
    encoder.layers=1
    encoder.config=SimpleNamespace(num_attention_heads=2)
    encoder.tokenizer=ToyTokenizer()
    layer=ToyLayer()
    encoder.model=SimpleNamespace(layers=[layer])

    def forward(**inputs):
        value=inputs['input_ids'].unsqueeze(-1).repeat(1,1,4)
        return layer(value)

    encoder._forward=forward
    result=encoder.encode_attention_values(['ab','c'],layers=(0,),batch_size=2)[0]
    assert result.dtype==torch.float32
    assert result.device.type=='cpu'
    assert torch.equal(result,torch.tensor([[10.5]*4,[20.]*4]))
    assert encoder.tokenizer.padding_side=='left'
    assert len(layer.self_attn.o_proj._forward_pre_hooks)==0
    assert len(layer._forward_hooks)==0


def test_publication_refreshes_learned_graph_without_mutating_active_readers(tmp_path,monkeypatch):
    from tests.test_engineering_research_resident import setup
    backend,events,_,_=setup(tmp_path,monkeypatch)
    backend.experiment_policy=NativeExperimentPolicy(modules=('learned_associations',))
    try:
        backend.sync(events)
        previous=backend._published[0]
        assert previous.experiments.graph.receipt['edges']==0
        packet=backend.recall_published('Deployment?')
        assert packet['retrieval']['experiments']['arms'][0]['status']=='no_learned_links'
        feedback=SimpleNamespace(references=packet['references'],input_event_id=events[-1].event_id)
        backend.learn(feedback,access_event_id='observation1')
        backend.learn(feedback,access_event_id='observation2')
        current=backend._published[0]
        assert current is not previous
        assert current.experiments.graph.receipt['edges']>0
        assert previous.experiments.graph.receipt['edges']==0
        assert backend.recall('Deployment?')==backend.recall_published('Deployment?')
    finally:
        backend.close()


def test_publication_builds_sparse_spectral_state_and_recall_uses_it(tmp_path,monkeypatch):
    from tests.test_engineering_research_resident import setup
    backend,events,_,_=setup(tmp_path,monkeypatch)
    backend.experiment_policy=NativeExperimentPolicy(modules=('spectral_reinjection',))
    try:
        backend.sync(events)
        experiment=backend._published[0].experiments
        assert experiment.spectral.receipt['algorithm'].endswith('spectral-reinjection-v1')
        assert experiment.spectral.receipt['retained_state_bytes']>0
        packet=backend.recall_published('Which deployment choice did we make?')
        arm=packet['retrieval']['experiments']['arms'][0]
        assert arm['status']=='selected' and arm['query_qwen_passes']==0
    finally:
        backend.close()


def test_optional_qwen_learning_receives_only_stored_summary_text(tmp_path):
    from tests.test_chat_native import NativeFixtureBackend
    from memory_condense.application.native_experiment_learning import summary_weights
    backend=NativeFixtureBackend(tmp_path/'memory')
    try:
        backend.sync((ChatEvent('a','user','Use incremental publication.'),
                      ChatEvent('b','assistant','Keep the exact source pointers.'),
                      ChatEvent('q','user','Which choice did we make?')))
        memory=backend.app._load_native_spine()[2]
        atoms=list(memory.router.atoms.values())
        packet=SimpleNamespace(input_event_id='q',references=[dict(section_id=a.section_id,
            independent=True) for a in atoms if a.spans[0].turn_id!='q'])
        chunks=[r[0] for r in backend.app._db.execute('SELECT chunk_id FROM chunks').fetchall()]
        calls=[]
        class Attention:
            def experiment(self,module,query,summaries,**kwargs):
                assert module=='qwen_consolidation'
                assert query==' '.join(a.summary for a in atoms if a.spans[0].turn_id=='q')
                assert all(s.summary in {a.summary for a in atoms} for s in summaries)
                calls.append(summaries)
                return [(s.section_id,.25) for s in summaries],dict(status='selected',model_passes=1)
        weights,receipt=summary_weights(backend.app,packet,chunks,memory.router,Attention(),
            NativeExperimentPolicy(modules=('qwen_consolidation',)))
        assert calls and weights and receipt['model_passes']==1
        assert sum(v==.25 for v in weights.values())==2
    finally:
        backend.close()


def test_cav_artifacts_require_training_checkpoint_and_unchanged_bytes(tmp_path):
    import hashlib
    from memory_condense.runtime.experimental_attention import cav_bundle
    report=tmp_path/'report.json'; vectors=tmp_path/'vectors.safetensors'
    report.write_text(json.dumps(dict(checkpoint_sha256='a'*64,runtime_dtype='float16')),encoding='utf-8')
    vectors.write_bytes(b'not loaded by provenance validation')
    value=dict(checkpoint_sha256='a'*64,layer=4,report=report.name,vectors=vectors.name,
        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        vectors_sha256=hashlib.sha256(vectors.read_bytes()).hexdigest())
    path=tmp_path/'bundle.json'; path.write_text(json.dumps(value),encoding='utf-8')
    assert cav_bundle(path,layer=4,checkpoint_sha256='a'*64)['vectors']==vectors
    with pytest.raises(ValueError,match='training checkpoint'):
        cav_bundle(path,layer=4,checkpoint_sha256='b'*64)
    vectors.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        cav_bundle(path,layer=4)


def test_installed_proxy_cli_propagates_experiments_to_all_sessions(tmp_path,monkeypatch):
    import uvicorn
    from memory_condense.cli import main
    from memory_condense.interfaces import proxy_server
    from memory_condense.runtime import sessions
    from memory_condense.runtime.config import RuntimeAssets
    policy=NativeExperimentPolicy(modules=('learned_associations','heat_diffusion'))
    path=tmp_path/'experiments.json'
    path.write_text(json.dumps(policy.identity_payload()),encoding='utf-8')
    monkeypatch.setattr(RuntimeAssets,'require_files',lambda *a,**k:None)
    monkeypatch.setattr(sessions,'LocalRuntime',lambda *a,**k:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(sessions,'ChatSession',lambda path,name,backend,**k:SimpleNamespace(backend=backend))
    observed=[]
    def serve(config,memory,locator=None,**_):
        assert locator is not None
        assert memory.status()['retrieval']['experiments']==policy.identity_payload()
        for name in ('a','b'):
            session=memory.factory(name,tmp_path/name)
            assert session.backend.experiment_policy==policy
        observed.append(True)
        return object()
    monkeypatch.setattr(proxy_server,'build_app',serve)
    monkeypatch.setattr(uvicorn,'run',lambda *a,**k:None)
    assert main(['proxy','--data-dir',str(tmp_path/'service'),'--memory-experiment-config',str(path)])==0
    assert observed
    with pytest.raises(SystemExit) as exc:
        main(['proxy','--mode','observe','--memory-experiment-config',str(path)])
    assert exc.value.code==2
