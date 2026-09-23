"""Freeze completed local summary outputs for a new ingest execution policy."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from memory_condense.domain._discourse_identity import identity_sha256
from tools.matched_eval.artifacts import publish_sealed_json,read_sealed_json
from tools.restore_spine_parent_hierarchy_local import LocalJournal


INPUT_KEYS = ('leaf_projection_sha256','atoms_sha256','serving_index_sha256',
    'leaf_index_sha256','raw_span_population_sha256','raw_token_proxy','leaf_root','atoms_path','index_root')


def verify_implementation(preflight):
    for path,sha in preflight.payload['implementation'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError('source summary implementation changed')


def freeze_namespace(cached_root,local_root,root):
    if root.resolve() in (cached_root.resolve(),local_root.resolve()):
        raise ValueError('summary reuse requires a separate snapshot root')
    cached = read_sealed_json(cached_root/'preflight.json')
    verify_implementation(cached)
    original = read_sealed_json(cached_root/'input-cache.json')
    if (original.sha256 != cached.payload['input_cache_sha256']
        or original.payload['atoms_sha256'] != cached.payload['atoms_sha256']
        or cached.payload['raw_inputs_to_qwen'] is not False):
        raise ValueError('cached summary corpus binding changed')
    values = dict(original.payload['summaries'])
    responses,incomplete,local_sha = [],[],None
    if (local_root/'preflight.json').exists():
        local = read_sealed_json(local_root/'preflight.json')
        verify_implementation(local)
        p = local.payload
        if (p['cached_preflight_sha256'] != cached.sha256
            or p['input_cache_sha256'] != original.sha256
            or any(p[k] != cached.payload[k] for k in INPUT_KEYS)
            or p['raw_inputs_to_qwen'] is not False):
            raise ValueError('local summaries belong to a different memory')
        local_sha = local.sha256
        journal = LocalJournal(local_root,local,SimpleNamespace(identity_sha256=p['backend_sha256']),0)
        journal.cache.values.update(values)
        for path in sorted((local_root/'requests').glob('*.json')):
            request = read_sealed_json(path)
            if request.payload['preflight_sha256'] != local.sha256:
                raise ValueError('local request belongs to a different preflight')
            response_path = local_root/'responses'/(request.sha256+'.json')
            if not response_path.exists():
                incomplete.append(request.sha256)
                continue
            response = read_sealed_json(response_path)
            journal.accept(request,response)
            responses.append({'request_sha256':request.sha256,'response_sha256':response.sha256})
        values = journal.cache.values
    cache,_ = publish_sealed_json(root/'input-cache.json',{
        'atoms_sha256':cached.payload['atoms_sha256'],'summaries':values,
        'original_cache_sha256':original.sha256,'original_cached_preflight_sha256':cached.sha256,
        'local_preflight_sha256':local_sha,'completed_local_responses':responses,
        'incomplete_requests_excluded':incomplete,'new_calls':0,'raw_inputs_to_qwen':False})
    preflight,_ = publish_sealed_json(root/'preflight.json',{
        'format':'memory-condense-completed-local-summary-snapshot-v1',
        **{k:cached.payload[k] for k in INPUT_KEYS},'input_cache_sha256':cache.sha256,
        'cached_root':str(cached_root.resolve()),'cached_preflight_sha256':cached.sha256,
        'local_root':str(local_root.resolve()),'local_preflight_sha256':local_sha,
        'raw_inputs_to_qwen':False,'implementation':{
            **cached.payload['implementation'],
            **(local.payload['implementation'] if local_sha else {}),
            'tools/reuse_local_spine_parent_summaries.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})
    return {'output_root':str(root.resolve()),'preflight_sha256':preflight.sha256,
        'cached_jobs':len(values),'additional_local_jobs':len(values)-len(original.payload['summaries']),
        'completed_local_batches':len(responses),'incomplete_requests_excluded':len(incomplete)}


def freeze_population(source_population,local_root,root):
    source = read_sealed_json(source_population)
    if [r['offset'] for r in source.payload['records']] != list(range(0,100,10)):
        raise ValueError('summary snapshot requires all ten bound memories')
    records = []
    for row in source.payload['records']:
        cached_root = Path(row['output_root'])
        if read_sealed_json(cached_root/'preflight.json').sha256 != row['preflight_sha256']:
            raise ValueError('source population changed')
        record = freeze_namespace(cached_root,local_root/f"offset-{row['offset']:03}",root/f"offset-{row['offset']:03}")
        records.append({'offset':row['offset'],**record})
    artifact,_ = publish_sealed_json(root/'population.json',{
        'format':'memory-condense-completed-local-summary-population-v1',
        'source_population_sha256':source.sha256,'records':records,'new_calls':0,
        'complete_namespace_population':False,'raw_inputs_to_qwen':False})
    print({'summary_snapshot_sha256':artifact.sha256,
           'additional_local_jobs':sum(r['additional_local_jobs'] for r in records),
           'incomplete_requests_excluded':sum(r['incomplete_requests_excluded'] for r in records)},flush=True)
    return artifact
