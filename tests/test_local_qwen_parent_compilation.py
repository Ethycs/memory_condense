from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.search.episodes.user_spine_hierarchy import _render_channels
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.spine_parent_hierarchy import SourceSpineParentPlan
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest
from tools.build_spine_corpus_hierarchy_resilient import NeedsProviderWork
from tools.local_qwen_spine_backend import decode_rows, job_messages
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.restore_spine_parent_hierarchy_cached import compile_available
from tools.restore_spine_parent_hierarchy_local import LocalJournal, run


def job(topic):
    return SpineSummaryRequest('user_spine',(
        SpineSummaryFragment('user_summary','2026-09-10',f'User enjoys {topic}. '*20),
        SpineSummaryFragment('user_summary','2026-09-10',f'User studies {topic}. '*20)),
        max_output_tokens=128)


def plan(topic):
    leaves,spans = [],[]
    for i in range(2):
        turn = Turn(turn_id=f'{topic}-{i}',source_id=topic,role='user',text=f'RAW_CANARY_{i}',
                    created_at=datetime(2026,9,10,tzinfo=timezone.utc))
        span = RawSectionSpan.from_turn(turn)
        spans.append(span)
        leaves.append(SectionSummary('spine-section-'+identity_sha256([span.receipt_sha256]),topic,
            _render_channels((f'User enjoys {topic} activities. '*20)+str(i),None,(span,)),(span,),'fixture'))
    cuts = [{'section_id':'spine-section-'+identity_sha256([s.receipt_sha256 for s in spans]),
             'split_atom':1,'attention_change':.2}]
    return SourceSpineParentPlan(leaves,spans,cuts),cuts,spans


class Backend:
    max_batch_size = 2
    identity = {'fixture':'local Qwen'}
    identity_sha256 = identity_sha256(identity)

    def __init__(self, invalid_before=0):
        self.calls = []
        self.invalid_before = invalid_before

    def generate(self, jobs, attempt):
        self.calls.append((jobs,attempt))
        return {'backend_sha256':self.identity_sha256,'rows':[
            {'job_sha256':j.prompt_sha256,'response':('bad JSON' if attempt<self.invalid_before
                 else '{"summary":"User enjoys a named activity."}'),
             'stopped':True,'output_tokens':10,'input_tokens':100} for j in jobs],
            'elapsed_s':1.0,'tokens_per_second':10.0,'peak_gpu_allocated_GiB':1.0,
            'raw_inputs_to_qwen':False,'remote_provider_calls':0}


def journal(root,backend,budget):
    preflight,_ = publish_sealed_json(root/'preflight.json',{'fixture':'test'})
    return LocalJournal(root,preflight,backend,budget)


def test_padding_is_removed_but_token_cap_is_rejected():
    class Tokenizer:
        def decode(self,tokens,skip_special_tokens):
            return ','.join(str(t) for t in tokens if t not in (8,9))
    rows = decode_rows(Tokenizer(),[[0,1,3,9,9],[1,2,4,5,6],[0,2,7,8,9]],2,{8,9})
    assert [(r['response'],r['output_tokens'],r['stopped']) for r in rows] == [
        ('3',2,True),('4,5,6',3,False),('7',2,True)]


def test_raw_job_shapes_and_unbounded_recovery_are_rejected():
    with pytest.raises(ValueError,match='typed summary'):
        job_messages({'raw_support_quote':'CANARY'},0)
    with pytest.raises(ValueError,match='bounded attempt'):
        job_messages(job('hiking'),3)
    assert 'at most 48 words' in job_messages(job('hiking'),1)[0]['content']
    assert 'at most 24 words' in job_messages(job('hiking'),2)[0]['content']


def test_budget_retains_completed_jobs_and_replay_needs_no_generation(tmp_path):
    backend = Backend()
    current = journal(tmp_path,backend,2)
    jobs = [job(t) for t in ('hiking','orchards','observatories')]
    pending = {j.prompt_sha256:j for j in jobs}
    current.resolve(pending,'parents',0)
    assert current.jobs==2 and len(current.cache.values)==2
    with pytest.raises(NeedsProviderWork):
        current.resolve(pending,'parents',1)
    resumed = journal(tmp_path,backend,1)
    resumed.replay()
    assert len(backend.calls)==1 and len(resumed.cache.values)==2
    resumed.resolve(pending,'parents',0)
    assert resumed.jobs==1 and len(resumed.cache.values)==3
    offline = journal(tmp_path,Backend(),0)
    offline.replay()
    offline.resolve(pending,'parents',0)
    assert not offline.backend.calls and offline.cache.values==resumed.cache.values


def test_invalid_attempt_is_not_repeated_when_batch_membership_changes(tmp_path):
    backend = Backend(invalid_before=1)
    current = journal(tmp_path,backend,2)
    jobs = [job(t) for t in ('hiking','orchards')]
    pending = {j.prompt_sha256:j for j in jobs}
    with pytest.raises(NeedsProviderWork):
        current.resolve(pending,'parents',0)
    resumed = journal(tmp_path,backend,2)
    resumed.replay()
    # Only one of the previously invalid jobs is requested on this traversal.
    resumed.resolve({jobs[1].prompt_sha256:jobs[1]},'parents',0)
    assert [attempt for _,attempt in backend.calls]==[0,1]
    assert len(resumed.cache.values)==1


def test_recovery_is_bounded_and_invalid_results_cannot_enter_cache(tmp_path):
    backend = Backend(invalid_before=3)
    current = journal(tmp_path,backend,6)
    j = job('hiking')
    with pytest.raises(ValueError,match='exhausted'):
        current.resolve({j.prompt_sha256:j},'parents',0)
    assert len(backend.calls)==3 and not current.cache.values
    resumed = journal(tmp_path,backend,100)
    resumed.replay()
    with pytest.raises(ValueError,match='exhausted'):
        resumed.resolve({j.prompt_sha256:j},'parents',0)
    assert len(backend.calls)==3


def test_uncertain_execution_is_not_retried(tmp_path):
    current = journal(tmp_path,Backend(),0)
    j = job('hiking')
    with pytest.raises(NeedsProviderWork):
        current.resolve({j.prompt_sha256:j},'parents',0)
    request = read_sealed_json(next((tmp_path/'requests').glob('*.json')))
    reservation = tmp_path/'executions'/(request.sha256+'.reserved')
    reservation.parent.mkdir()
    reservation.write_text(request.sha256,encoding='utf-8')
    with pytest.raises(ValueError,match='implicit retry'):
        current.replay()
    assert not current.backend.calls


def test_wrong_job_binding_and_missing_eos_are_rejected(tmp_path):
    current = journal(tmp_path,Backend(),0)
    j = job('hiking')
    body = {'preflight_sha256':current.preflight.sha256,'backend_sha256':current.backend.identity_sha256,
        'attempt':0,'jobs':[asdict(j)],'messages':[job_messages(j,0)],'raw_inputs_to_qwen':False}
    request,_ = publish_sealed_json(tmp_path/'request.json',body)
    result = current.backend.generate([j],0)
    result['rows'][0]['stopped'] = False
    response,_ = publish_sealed_json(tmp_path/'response.json',{'request_sha256':request.sha256,**result})
    current.accept(request,response)
    assert not current.cache.values
    result['rows'][0]['job_sha256'] = 'wrong'
    response,_ = publish_sealed_json(tmp_path/'wrong.json',{'request_sha256':request.sha256,**result})
    with pytest.raises(ValueError,match='attribution'):
        current.accept(request,response)


def test_real_parent_plan_completes_and_replays_without_raw_inputs(tmp_path):
    plans = {topic:plan(topic)[0] for topic in ('orchard','observatory')}
    current = journal(tmp_path,Backend(),2)
    parts,progress = compile_available(plans,current,tmp_path)
    assert len(parts)==2 and progress.payload['completed_parent_count']==2
    assert progress.payload['complete_namespace'] is True
    assert all('RAW_CANARY' not in str(j.messages) for jobs,_ in current.backend.calls for j in jobs)
    resumed = journal(tmp_path,Backend(),0)
    resumed.replay()
    replayed,again = compile_available(plans,resumed,tmp_path)
    assert again.sha256==progress.sha256 and not resumed.backend.calls
    assert {s:p.to_json() for s,p in parts.items()}=={s:p.to_json() for s,p in replayed.items()}


def test_local_compiler_publishes_complete_hierarchy_then_replays(tmp_path):
    source,cuts,spans = plan('orchard')
    index = SectionSummaryIndex(source.leaves)
    raw_sha = identity_sha256([s.receipt_sha256 for s in spans])
    atoms,_ = publish_sealed_json(tmp_path/'atoms.json',{'complete_namespace':True,
        'atoms':[s.identity_payload() for s in source.leaves],'raw_span_population_sha256':raw_sha})
    leaf_root,index_root,old = (tmp_path/k for k in ('leaves','index','cached'))
    leaf,_ = publish_sealed_json(leaf_root/'hierarchy.json',{'leaf_projection_complete':True,
        'complete_namespace':True,'raw_inputs_to_qwen':False,'atoms_sha256':atoms.sha256,
        'index_json':index.to_json(),'raw_span_population_sha256':raw_sha,
        'source_cuts':{'orchard':cuts},'source_attention_receipts':{'orchard':['fixture-attention']}})
    manifest,_ = publish_sealed_json(index_root/'index.json',{'complete_namespace':True,
        'hierarchy_sha256':leaf.sha256,'raw_span_population_sha256':raw_sha,
        'index_json':index.to_json(),'raw_token_proxy':sum(s.token_count for s in spans)})
    cache,_ = publish_sealed_json(old/'input-cache.json',{'atoms_sha256':atoms.sha256,'summaries':{}})
    publish_sealed_json(old/'preflight.json',{'implementation':{},'raw_inputs_to_qwen':False,
        'leaf_root':str(leaf_root),'atoms_path':str(tmp_path/'atoms.json'),'index_root':str(index_root),
        'input_cache_sha256':cache.sha256,'leaf_projection_sha256':leaf.sha256,'atoms_sha256':atoms.sha256,
        'serving_index_sha256':manifest.sha256,'leaf_index_sha256':index.receipt_sha256,
        'raw_span_population_sha256':raw_sha,'raw_token_proxy':manifest.payload['raw_token_proxy']})
    backend = Backend()
    partial = run(old,tmp_path/'local',backend,0)
    assert partial['complete_namespace'] is False and not (tmp_path/'local'/'hierarchy.json').exists()
    result = run(old,tmp_path/'local',backend,1)
    assert result['complete_namespace'] is True and result['new_local_jobs']==1
    full = read_sealed_json(tmp_path/'local'/'hierarchy.json')
    restored = SectionSummaryIndex.from_json(full.payload['index_json'])
    assert tuple(s for s in restored.sections if not s.child_section_ids)==index.sections
    assert full.payload['parent_summary_compilation_complete'] is True
    again = run(old,tmp_path/'local',backend,0)
    assert again['hierarchy_sha256']==full.sha256 and again['new_local_jobs']==0
    assert len(backend.calls)==1
