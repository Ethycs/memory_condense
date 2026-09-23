from datetime import datetime, timezone

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.search.episodes.user_spine_hierarchy import _render_channels
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.spine_merge_batch import PendingMerge, SummaryMergeCache
from memory_condense.search.spine_parent_hierarchy import SourceSpineParentPlan
from tools.build_spine_corpus_hierarchy import MODEL, GATEWAY
from tools.build_spine_corpus_hierarchy_resilient import RecoveryJournal
from tools.compile_spine_leaf_projection import frozen_cache
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.restore_spine_parent_hierarchy_cached import compile_available, run


def make_plan(topic):
    leaves,spans=[],[]
    for i in range(2):
        turn=Turn(turn_id=f'{topic}-{i}',source_id=topic,role='user',text=f'RAW_CANARY_{i}',
                  created_at=datetime(2026,9,10,tzinfo=timezone.utc))
        span=RawSectionSpan.from_turn(turn)
        spans.append(span)
        leaves.append(SectionSummary('spine-section-'+identity_sha256([span.receipt_sha256]),topic,
            _render_channels((f'User enjoys {topic} activities. '*20)+str(i),None,(span,)),(span,),'fixture'))
    cuts=[{'section_id':'spine-section-'+identity_sha256([s.receipt_sha256 for s in spans]),
           'split_atom':1,'attention_change':.2}]
    plan=SourceSpineParentPlan(leaves,spans,cuts)
    with pytest.raises(PendingMerge) as missing:
        plan.compile(summarize=SummaryMergeCache(),summarizer_identity='fixture')
    return plan,missing.value.request,cuts,spans


def test_complete_source_is_persisted_before_missing_dependency_and_reused_unchanged(tmp_path):
    a,job_a,_,_=make_plan('orchard')
    b,job_b,_,_=make_plan('observatory')
    preflight,_=publish_sealed_json(tmp_path/'preflight.json',{'fixture':True})
    journal=RecoveryJournal(tmp_path,preflight,False,0)
    journal.cache.values[job_a.prompt_sha256]='User enjoys orchard activities.'
    parts,progress=compile_available({'orchard':a,'observatory':b},journal,tmp_path)
    assert set(parts)=={'orchard'} and progress.payload['complete_namespace'] is False
    assert progress.payload['completed_parent_count']==1
    assert progress.payload['next_dependency_jobs']==[job_b.prompt_sha256]
    saved=read_sealed_json(tmp_path/'source-parts'/(identity_sha256('orchard')+'.json'))
    assert saved.payload['complete_source'] is True and saved.payload['complete_namespace'] is False
    assert journal.calls==0 and not (tmp_path/'hierarchy.json').exists()
    journal.cache.values[job_b.prompt_sha256]='User enjoys observatory activities.'
    completed,final=compile_available({'orchard':a,'observatory':b},journal,tmp_path)
    assert len(completed)==2 and final.payload['complete_namespace'] is True
    assert final.payload['completed_parent_count']==2
    assert read_sealed_json(tmp_path/'source-parts'/(identity_sha256('orchard')+'.json')).sha256==saved.sha256
    assert completed['orchard'].to_json()==parts['orchard'].to_json()


def inherited(root,atoms,values,*,wrong_atoms=False):
    cached,_=publish_sealed_json(root/'input-cache.json',{'atoms_sha256':'wrong' if wrong_atoms else atoms.sha256,
        'parents':[],'summaries':values,'authenticated_inputs':[{'fixture':True}],'new_calls':0})
    preflight,_=publish_sealed_json(root/'preflight.json',{'atoms_sha256':atoms.sha256,
        'complete_namespace':True,'model':MODEL,'gateway':GATEWAY,'raw_inputs_to_qwen':False,
        'max_channel_tokens':128,'implementation':{},
        'parent_caches':{'input_snapshot_sha256':cached.sha256}})
    return preflight,cached


def test_frozen_import_binds_parent_provenance_and_replays_without_provider(tmp_path):
    atoms,_=publish_sealed_json(tmp_path/'atoms.json',{'fixture':'atoms'})
    prior,cache=inherited(tmp_path/'prior',atoms,{'a'*64:'Completed user summary.'})
    imported,_=frozen_cache(tmp_path/'current',atoms,[tmp_path/'prior'])
    assert imported.payload['summaries']==cache.payload['summaries']
    assert imported.payload['authenticated_inputs']==[{'inherited_cache_sha256':cache.sha256}]
    assert imported.payload['parents'][0]['preflight_sha256']==prior.sha256
    replay,_=frozen_cache(tmp_path/'current',atoms,[tmp_path/'prior'])
    assert replay.sha256==imported.sha256 and replay.payload['new_calls']==0


def test_wrong_corpus_cache_cannot_supply_parent_summaries(tmp_path):
    atoms,_=publish_sealed_json(tmp_path/'atoms.json',{'fixture':'atoms'})
    inherited(tmp_path/'prior',atoms,{'a'*64:'Changed corpus.'},wrong_atoms=True)
    with pytest.raises(ValueError,match='inherited frozen summary cache'):
        frozen_cache(tmp_path/'current',atoms,[tmp_path/'prior'])


def test_conflicting_completed_summaries_are_not_overwritten(tmp_path):
    atoms,_=publish_sealed_json(tmp_path/'atoms.json',{'fixture':'atoms'})
    inherited(tmp_path/'a',atoms,{'a'*64:'First.'})
    inherited(tmp_path/'b',atoms,{'a'*64:'Second.'})
    with pytest.raises(ValueError,match='disagree'):
        frozen_cache(tmp_path/'current',atoms,[tmp_path/'a',tmp_path/'b'])


def test_cached_compiler_publishes_complete_index_only_after_all_source_parents(tmp_path):
    plan,job,cuts,spans=make_plan('orchard')
    index=SectionSummaryIndex(plan.leaves)
    raw_sha=identity_sha256([s.receipt_sha256 for s in spans])
    atoms,_=publish_sealed_json(tmp_path/'atoms.json',{'complete_namespace':True,
        'atoms':[s.identity_payload() for s in plan.leaves], 'raw_span_population_sha256':raw_sha})
    leaf_root=tmp_path/'leaves'
    projection,_=inherited(leaf_root,atoms,{job.prompt_sha256:'User enjoys orchard activities.'})
    leaf,_=publish_sealed_json(leaf_root/'hierarchy.json',{'leaf_projection_complete':True,
        'complete_namespace':True,'raw_inputs_to_qwen':False,'atoms_sha256':atoms.sha256,
        'preflight_sha256':projection.sha256,'index_json':index.to_json(),'raw_span_population_sha256':raw_sha,
        'source_cuts':{'orchard':cuts},'source_attention_receipts':{'orchard':['fixture-attention']}})
    index_root=tmp_path/'index'
    manifest,_=publish_sealed_json(index_root/'index.json',{'complete_namespace':True,'hierarchy_sha256':leaf.sha256,
        'raw_span_population_sha256':raw_sha,'index_json':index.to_json(),'raw_token_proxy':sum(s.token_count for s in spans)})
    old=tmp_path/'old-plan'
    publish_sealed_json(old/'preflight.json',{'implementation':{},'leaf_root':str(leaf_root),
        'atoms_path':str(tmp_path/'atoms.json'),'index_root':str(index_root),
        'leaf_projection_sha256':leaf.sha256,'leaf_projection_preflight_sha256':projection.sha256,
        'atoms_sha256':atoms.sha256,'serving_index_sha256':manifest.sha256,
        'leaf_index_sha256':index.receipt_sha256,'raw_span_population_sha256':raw_sha,
        'raw_token_proxy':manifest.payload['raw_token_proxy']})
    result=run(old,tmp_path/'restored')
    assert result['complete_namespace'] is True and result['new_calls']==0 and result['complete_parents']==1
    full=read_sealed_json(tmp_path/'restored'/'hierarchy.json')
    restored=SectionSummaryIndex.from_json(full.payload['index_json'])
    assert tuple(s for s in restored.sections if not s.child_section_ids)==index.sections
    assert full.payload['parent_summary_compilation_complete'] is True
    assert run(old,tmp_path/'restored')['hierarchy_sha256']==full.sha256
