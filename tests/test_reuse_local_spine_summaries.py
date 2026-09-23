from dataclasses import asdict

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.spine_summary import SpineSummaryFragment,SpineSummaryRequest
from tools.local_qwen_spine_backend import job_messages
from tools.matched_eval.artifacts import publish_sealed_json,read_sealed_json
from tools.reuse_local_spine_parent_summaries import freeze_namespace,INPUT_KEYS


def fixtures(root,*,wrong_corpus=False,raw=False):
    fields = {key:'fixture-'+key for key in INPUT_KEYS}
    original,_ = publish_sealed_json(root/'cached'/'input-cache.json',{
        'atoms_sha256':fields['atoms_sha256'],'summaries':{'a'*64:'Prior completed summary.'}})
    cached,_ = publish_sealed_json(root/'cached'/'preflight.json',{
        **fields,'input_cache_sha256':original.sha256,'implementation':{},'raw_inputs_to_qwen':False})
    local,_ = publish_sealed_json(root/'local'/'preflight.json',{
        **fields,'atoms_sha256':'different' if wrong_corpus else fields['atoms_sha256'],
        'input_cache_sha256':original.sha256,'cached_preflight_sha256':cached.sha256,
        'backend_sha256':'b'*64,'raw_inputs_to_qwen':raw,'implementation':{}})
    return cached,local


def response(root,preflight,topic,*,completed=True,valid=True,changed_job=False):
    job = SpineSummaryRequest('user_spine',(
        SpineSummaryFragment('user_summary','2026-09-10',f'User enjoys {topic}.'),),max_output_tokens=128)
    body = {'preflight_sha256':preflight.sha256,'backend_sha256':'b'*64,'attempt':0,
            'jobs':[asdict(job)],'messages':[job_messages(job,0)],'raw_inputs_to_qwen':False}
    request,_ = publish_sealed_json(root/'requests'/(identity_sha256(body)+'.json'),body)
    if completed:
        publish_sealed_json(root/'responses'/(request.sha256+'.json'),{
            'request_sha256':request.sha256,'backend_sha256':'b'*64,
            'raw_inputs_to_qwen':False,'remote_provider_calls':0,'rows':[
                {'job_sha256':'changed' if changed_job else job.prompt_sha256,'stopped':True,
                 'response':'{"summary":"User enjoys outdoor activities."}' if valid else 'not JSON'}]})
    else:
        reservation = root/'executions'/(request.sha256+'.reserved')
        reservation.parent.mkdir(parents=True,exist_ok=True)
        reservation.write_text(request.sha256,encoding='utf-8')
    return job,request


def test_snapshot_reuses_only_valid_completed_outputs_and_preserves_incomplete_execution(tmp_path):
    _,local = fixtures(tmp_path)
    valid,_ = response(tmp_path/'local',local,'hiking')
    invalid,_ = response(tmp_path/'local',local,'cycling',valid=False)
    incomplete,request = response(tmp_path/'local',local,'rowing',completed=False)
    result = freeze_namespace(tmp_path/'cached',tmp_path/'local',tmp_path/'snapshot')
    cache = read_sealed_json(tmp_path/'snapshot'/'input-cache.json')
    assert result['additional_local_jobs']==1 and result['completed_local_batches']==2
    assert set(cache.payload['summaries'])=={'a'*64,valid.prompt_sha256}
    assert invalid.prompt_sha256 not in cache.payload['summaries']
    assert incomplete.prompt_sha256 not in cache.payload['summaries']
    assert cache.payload['incomplete_requests_excluded']==[request.sha256]
    assert (tmp_path/'local'/'executions'/(request.sha256+'.reserved')).exists()
    again = freeze_namespace(tmp_path/'cached',tmp_path/'local',tmp_path/'snapshot')
    assert again==result and cache.payload['new_calls']==0


@pytest.mark.parametrize('change',['corpus','raw','job'])
def test_changed_local_bindings_cannot_supply_snapshot(tmp_path,change):
    _,local = fixtures(tmp_path,wrong_corpus=change=='corpus',raw=change=='raw')
    response(tmp_path/'local',local,'hiking',changed_job=change=='job')
    with pytest.raises(ValueError,match='different memory|attribution'):
        freeze_namespace(tmp_path/'cached',tmp_path/'local',tmp_path/'snapshot')
    assert not (tmp_path/'snapshot'/'input-cache.json').exists()


def test_namespace_without_local_run_reuses_original_cache(tmp_path):
    fixtures(tmp_path)
    result = freeze_namespace(tmp_path/'cached',tmp_path/'absent',tmp_path/'snapshot')
    assert result['additional_local_jobs']==0 and result['cached_jobs']==1
    cache = read_sealed_json(tmp_path/'snapshot'/'input-cache.json')
    assert cache.payload['local_preflight_sha256'] is None


def test_snapshot_cannot_overwrite_a_source_root(tmp_path):
    fixtures(tmp_path)
    with pytest.raises(ValueError,match='separate snapshot'):
        freeze_namespace(tmp_path/'cached',tmp_path/'local',tmp_path/'local')


@pytest.mark.parametrize('elapsed,peak,valid,passed',[(10.,5.,True,True),(21.,5.,True,False),
    (10.,5.4,True,False),(10.,5.,False,False)])
def test_batch_release_requires_valid_outputs_measured_speed_and_memory_headroom(tmp_path,monkeypatch,elapsed,peak,valid,passed):
    from tools import run_local_spine_parent_batch4 as runner
    jobs = tuple(SpineSummaryRequest('user_spine',(
        SpineSummaryFragment('user_summary','2026-09-10',f'User enjoys activity {i}.'),)) for i in range(4))
    monkeypatch.setattr(runner,'baseline',lambda _: (jobs,20.))
    class Backend:
        identity = {'fixture':True}
        identity_sha256 = identity_sha256(identity)
        def generate(self,requested,attempt):
            assert requested==jobs and attempt==0
            return {'backend_sha256':self.identity_sha256,'rows':[
                {'job_sha256':j.prompt_sha256,'stopped':True,
                 'response':'{"summary":"User enjoys a named activity."}' if valid else 'not JSON'} for j in jobs],
                'elapsed_s':elapsed,'peak_gpu_allocated_GiB':peak,'remote_provider_calls':0,
                'raw_inputs_to_qwen':False}
    result = runner.benchmark(tmp_path/'probe',tmp_path/'local',Backend())
    assert result.payload['batch_release_passed'] is passed
    assert result.payload['target_gate_passed'] is False and result.payload['accuracy_measured'] is False
