from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from tools import evaluate_hierarchical_spine_full100 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json


def question(ordinal):
    return {'ordinal':ordinal, 'question_id':f'q-{ordinal}', 'retrieval_query':f'question-{ordinal}',
            'prompt_question':f'[Question asked at 2026/09/10 (Thu) 12:00]\nquestion-{ordinal}',
            'shard_offset':10*(ordinal//10)}


def audit(q):
    return {'parent_hierarchy_sha256':'a'*64,'raw_inputs_to_qwen':False,'live_query_embedding':True,
        'projection':{'projected_plan_sha256':identity_sha256(q)}, 'attention_plan':{
            'query_sha256':quote_sha256(q['retrieval_query']), 'routes':[{'section_id':'leaf'}],
            'attention_receipt':{'linker_identity_json':'fixture-linker', 'raw_content_inspections':0,
                'retained_transformer_token_state_bytes':0,'rounds':[{
                    'candidate_section_ids':['parent'], 'selected_section_ids':['parent'],
                    'selected_qk_scores':[.1], 'selected_ov_transport':[.2], 'model_passes':1,
                    'max_workspace_candidates':1,'max_workspace_tokens':100,'candidate_inspections':1}]}}}


def population(root=None):
    calls=[]
    for ordinal in range(100):
        q=question(ordinal)
        prompts={a:evaluation.messages(q) for a in evaluation.ARMS}
        payload={'question':q,'messages':prompts,'hydration':{a:{'fixture':a,'ordinal':ordinal} for a in evaluation.MEMORY_ARMS},
                 'hierarchy_audit':audit(q)}
        sha=identity_sha256(payload)
        if root is not None:
            artifact,_=publish_sealed_json(root/'evidence'/f'{ordinal:03}.json',payload)
            sha=artifact.sha256
        for arm in evaluation.call_order(ordinal):
            calls.append({'call_index':len(calls),'question':q,'arm':arm,
                'messages':prompts[arm],'messages_sha256':identity_sha256(prompts[arm]),'evidence_sha256':sha})
    return calls


def observations(correct=95, candidate_time=1., api_time=1.):
    measured,judged=[],[]
    for c in population():
        arm=c['arm']
        m={'prediction':'fact','prediction_sha256':quote_sha256('fact'),'prepare_s':.1,
           'e2e_ttft_s':candidate_time if arm=='hierarchy' else api_time,
           'e2e_total_s':candidate_time if arm=='hierarchy' else api_time,'finish_reason':'stop'}
        response=SimpleNamespace(sha256=identity_sha256(c),payload={'measurement':m})
        measured.append((c,response))
        if arm in evaluation.MEMORY_ARMS:
            judged.append({'ordinal':c['question']['ordinal'],'arm':arm,'prediction_sha256':m['prediction_sha256'],
                           'response_sha256':response.sha256,'correct':c['question']['ordinal']<correct})
    return measured,judged


def test_all100_candidate_api_pairs_are_adjacent_and_counterbalanced():
    calls=population()
    evaluation.validate_calls(calls)
    orders=[evaluation.call_order(i) for i in range(100)]
    assert sum(o.index('hierarchy')<o.index('hierarchy_api') for o in orders)==50
    assert all(abs(o.index('hierarchy')-o.index('hierarchy_api'))==1 for o in orders)


@pytest.mark.parametrize('change',['missing','order','api_context','reader'])
def test_request_validation_rejects_broken_population_or_matched_control(change):
    calls=population()
    if change=='missing':
        calls.pop()
    elif change=='order':
        calls[0],calls[1]=calls[1],calls[0]
    else:
        row=next(c for c in calls if c['arm']=='hierarchy_api')
        row['messages']=deepcopy(row['messages'])
        row['messages'][1 if change=='api_context' else 0]['content']+=' changed'
        row['messages_sha256']=identity_sha256(row['messages'])
    with pytest.raises(ValueError):
        evaluation.validate_calls(calls)


@pytest.mark.parametrize('correct,time,passed',[(95,1.1,True),(94,1.,False),(100,1.11,False)])
def test_quality_and_latency_must_pass_together(correct,time,passed):
    assert evaluation.joint_statistics(*observations(correct,time))['target_gate_passed'] is passed


def test_tail_ttft_and_short_api_are_independent_required_gates():
    measured,judged=observations(100)
    for c,r in measured:
        if c['arm']=='hierarchy' and c['question']['ordinal']>=94:
            r.payload['measurement']['e2e_ttft_s']=1.2
    assert not evaluation.joint_statistics(measured,judged)['target_gate_passed']
    measured,judged=observations(100)
    for c,r in measured:
        if c['arm']=='short_api':
            r.payload['measurement']['e2e_ttft_s']=.5
            r.payload['measurement']['e2e_total_s']=.5
    result=evaluation.joint_statistics(measured,judged)
    assert result['candidate_latency_ratios']['hierarchy_api']['e2e_total_s']['median_s']==1.
    assert not result['target_gate_passed']


@pytest.mark.parametrize('change',['prediction','response','missing'])
def test_cannot_combine_accuracy_with_other_response_timings(change):
    measured,judged=observations()
    if change=='missing':
        measured.pop()
    else:
        judged[0][change+'_sha256']='f'*64
    with pytest.raises(ValueError):
        evaluation.joint_statistics(measured,judged)


def test_truncated_stream_fails_joint_gate_even_with_full_quality():
    measured,judged=observations(100)
    measured[0][1].payload['measurement']['finish_reason']='length'
    assert not evaluation.joint_statistics(measured,judged)['target_gate_passed']


def test_attention_roundoff_can_vary_but_not_selected_path_or_workspace():
    a=audit(question(0))
    changed=deepcopy(a)
    changed['attention_plan']['attention_receipt']['rounds'][0]['selected_qk_scores']=[.100000001]
    assert evaluation.attention_signature(a)==evaluation.attention_signature(changed)
    changed['attention_plan']['attention_receipt']['rounds'][0]['selected_section_ids']=['foreign']
    assert evaluation.attention_signature(a)!=evaluation.attention_signature(changed)
    changed['attention_plan']['attention_receipt']['rounds'][0]['model_passes']=2
    with pytest.raises(ValueError):
        evaluation.attention_signature(changed)


def test_incomplete_answers_never_open_references(tmp_path,monkeypatch):
    preflight=SimpleNamespace(payload={'calls':population()},sha256='f'*64)
    monkeypatch.setattr(evaluation,'load_preflight',lambda root:preflight)
    monkeypatch.setattr(evaluation,'load_references',lambda:pytest.fail('references opened before all answers'))
    with pytest.raises(ValueError,match='all400'):
        evaluation.judge(tmp_path)


def test_entire_400_stream_execution_and_200_judgments_then_zero_call_replay(tmp_path,monkeypatch):
    calls=population(tmp_path)
    preflight,_=publish_sealed_json(tmp_path/'preflight.json',{'calls':calls,'qwen_identity':{'fixture':True},
        'namespaces':[{'root':'fixture','offset':i,'parent':{}} for i in range(0,100,10)]})
    monkeypatch.setattr(evaluation,'load_preflight',lambda root:preflight)
    monkeypatch.setattr(evaluation,'load_as_of',lambda root:SimpleNamespace(payload={}))
    monkeypatch.setattr(evaluation,'require_idle',lambda:None)
    monkeypatch.setattr(evaluation,'new_linker',lambda:object())
    monkeypatch.setattr(evaluation,'qwen_linker_identity',lambda *args,**kwargs:{'fixture':True})
    monkeypatch.setattr(evaluation,'resident',lambda *args:SimpleNamespace(encoder=SimpleNamespace(close=lambda:None)))
    builds=[]
    def build(memory,q,arm):
        builds.append((q['ordinal'],arm))
        return evaluation.messages(q),{'fixture':arm,'ordinal':q['ordinal']},audit(q) if arm=='hierarchy' else None
    monkeypatch.setattr(evaluation,'build',build)
    sent=[]
    class Stream:
        def __iter__(self):
            yield {'model':evaluation.MODEL,'choices':[{'index':0,'delta':{'role':'assistant'},'finish_reason':None}]}
            yield {'model':evaluation.MODEL,'choices':[{'index':0,'delta':{'content':'fact'},'finish_reason':'stop'}]}
        def close(self):
            pass
    class Client:
        max_retries=0
        chat=property(lambda self:SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        def with_options(self,**kwargs):
            return self
        def close(self):
            pass
        def create(self,**kwargs):
            sent.append(kwargs)
            if kwargs.get('stream'):
                return Stream()
            assert (tmp_path/'answers.json').is_file()
            return SimpleNamespace(id=f'judge-{len(sent)}',model='codex_sdk/gpt-5.6-sol',usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content='CORRECT'),finish_reason='stop')])
    monkeypatch.setattr(evaluation,'_completion_client',lambda *args:Client())
    def references():
        assert len(list((tmp_path/'journal').glob('*.response.json')))==400
        assert (tmp_path/'answers.json').is_file()
        return None,[SimpleNamespace(question_id=f'q-{i}',answer='fact') for i in range(100)]
    monkeypatch.setattr(evaluation,'load_references',references)
    evaluation.run(tmp_path,True)
    assert len(builds)==200 and sum(c.get('stream',False) for c in sent)==400
    assert sum(not c.get('stream',False) for c in sent)==100  # Identical predictions share judge prompts.
    from tools.matched_eval.artifacts import read_sealed_json
    report=read_sealed_json(tmp_path/'joint-report.json')
    assert report.payload['accuracy']=={'flat':100,'hierarchy':100}
    monkeypatch.setattr(evaluation,'_completion_client',lambda *args:pytest.fail('replay attempted provider'))
    replay=evaluation.judge(tmp_path,False)
    assert replay.sha256==report.sha256
