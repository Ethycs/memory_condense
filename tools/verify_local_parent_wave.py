"""Replay a completed parent-merge wave without model calls or raw text reads."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from memory_condense.domain._discourse_identity import identity_sha256,quote_sha256
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.spine_summary import parse_spine_summary
from tools.build_spine_corpus_hierarchy import restore_request
from tools.matched_eval.artifacts import read_sealed_json,publish_sealed_json
from tools.restore_spine_parent_hierarchy_local import LocalJournal


def verify(root,before_sha,after_sha,output):
    preflight = read_sealed_json(root/'preflight.json')
    progress = {a.sha256:a for a in (read_sealed_json(p) for p in (root/'progress').glob('*.json'))}
    before,after = progress[before_sha],progress[after_sha]
    if (before.payload['preflight_sha256']!=preflight.sha256
        or after.payload['preflight_sha256']!=preflight.sha256
        or after.payload['completed_source_count']<before.payload['completed_source_count']):
        raise ValueError('wave progress binding changed')
    frontier = set(before.payload['next_dependency_jobs'])
    if not frontier or len(frontier)!=len(before.payload['next_dependency_jobs']):
        raise ValueError('wave requires a nonempty unique initial dependency frontier')
    cache = read_sealed_json(Path(preflight.payload['cached_root'])/'input-cache.json')
    if cache.sha256!=preflight.payload['input_cache_sha256']:
        raise ValueError('input cache changed')
    journal = LocalJournal(root,preflight,SimpleNamespace(identity_sha256=preflight.payload['backend_sha256']),0)
    journal.cache.values.update(cache.payload['summaries'])
    counts,bindings = Counter(),[]
    elapsed,tokens,peak = 0.,0,0.
    for path in sorted((root/'requests').glob('*.json')):
        request = read_sealed_json(path)
        jobs = tuple(restore_request(j) for j in request.payload['jobs'])
        if not any(j.prompt_sha256 in frontier for j in jobs):
            continue
        if any(j.prompt_sha256 not in frontier for j in jobs):
            raise ValueError('wave populations mixed')
        response = read_sealed_json(root/'responses'/(request.sha256+'.json'))
        journal.accept(request,response)
        bindings.append({'request_sha256':request.sha256,'response_sha256':response.sha256})
        counts['completed_batches']+=1
        elapsed+=response.payload['elapsed_s']
        tokens+=sum(row['output_tokens'] for row in response.payload['rows'])
        peak=max(peak,response.payload['peak_gpu_allocated_GiB'])
        for job,row in zip(jobs,response.payload['rows'],strict=True):
            attempt=request.payload['attempt']
            counts[f'attempt_{attempt}_jobs']+=1
            try:
                if row['stopped'] is not True:
                    raise ValueError('no EOS')
                parse_spine_summary(row['response'],job)
            except ValueError as exc:
                counts[str(exc)]+=1
            else:
                counts[f'attempt_{attempt}_valid']+=1
    if elapsed<=0 or counts['attempt_0_jobs']!=len(frontier) or any(sha not in journal.cache.values for sha in frontier):
        raise ValueError('wave does not have one initial generation and an accepted result per dependency')
    leaf=read_sealed_json(Path(preflight.payload['leaf_root'])/'hierarchy.json')
    if leaf.sha256!=preflight.payload['leaf_projection_sha256']:
        raise ValueError('original leaf population changed')
    original=SectionSummaryIndex.from_json(leaf.payload['index_json'])
    parents=leaves=0
    for binding in after.payload['completed_source_parts']:
        part=read_sealed_json(root/'source-parts'/(identity_sha256(binding['source_id'])+'.json'))
        if part.sha256!=binding['sha256'] or part.payload['preflight_sha256']!=preflight.sha256:
            raise ValueError('completed source changed')
        index=SectionSummaryIndex.from_json(part.payload['index_json'])
        retained=tuple(s for s in index.sections if not s.child_section_ids)
        if retained!=tuple(s for s in original.sections if s.source_id==binding['source_id']):
            raise ValueError('original leaf or raw span descriptors changed')
        parents+=sum(bool(s.child_section_ids) for s in index.sections)
        leaves+=len(retained)
    if parents!=after.payload['completed_parent_count']:
        raise ValueError('published parent count changed')
    result,_=publish_sealed_json(output,{
        'preflight_sha256':preflight.sha256,'before_progress_sha256':before.sha256,'after_progress_sha256':after.sha256,
        'initial_dependency_jobs':len(frontier),'all_initial_dependencies_resolved':True,'counts':dict(counts),
        'generation_seconds':elapsed,'aggregate_tokens_per_second':tokens/elapsed,'peak_gpu_allocated_GiB':peak,
        'completed_source_count':after.payload['completed_source_count'],'completed_parent_count':parents,
        'additional_published_parents':parents-before.payload['completed_parent_count'],'verified_unchanged_leaf_count':leaves,
        'completed_request_responses':bindings,
        'accepted_summary_sha256s':{sha:quote_sha256(journal.cache.values[sha]) for sha in sorted(frontier)},
        'verification_new_model_calls':0,'raw_text_reads':0,
        'complete_namespace':after.payload['complete_namespace'],'target_gate_passed':False})
    print({'verification_sha256':result.sha256,**{k:v for k,v in result.payload.items()
        if k not in ('completed_request_responses','accepted_summary_sha256s')}},flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace-root',type=Path,required=True)
    parser.add_argument('--before-sha',required=True)
    parser.add_argument('--after-sha',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    verify(args.namespace_root,args.before_sha,args.after_sha,args.output)
