"""Complete saved attention trees using cached merges and resident local Qwen."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.spine_merge_batch import SummaryMergeCache
from memory_condense.search.spine_summary import parse_spine_summary
from tools.build_spine_corpus_hierarchy import restore_request
from tools.build_spine_corpus_hierarchy_resilient import NeedsProviderWork
from tools.local_qwen_spine_backend import LocalQwenBackend, job_messages
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.restore_spine_parent_hierarchy import prepare_sources
from tools.restore_spine_parent_hierarchy_cached import IMPLEMENTATION as CACHED_IMPLEMENTATION, compile_available


IMPLEMENTATION = (*CACHED_IMPLEMENTATION, 'tools/local_qwen_spine_backend.py',
    'tools/restore_spine_parent_hierarchy_local.py', 'tools/probe_local_qwen_parent_summaries.py')


class LocalJournal:
    def __init__(self, root, preflight, backend, budget):
        self.root, self.preflight, self.backend, self.budget = root, preflight, backend, budget
        self.cache = SummaryMergeCache()
        self.calls = self.jobs = 0
        self.completed_attempts = set()

    def accept(self, request, response):
        p, r = request.payload, response.payload
        jobs = tuple(restore_request(row) for row in p['jobs'])
        if (p['preflight_sha256'] != self.preflight.sha256 or p['raw_inputs_to_qwen'] is not False
            or p['messages'] != [job_messages(j,p['attempt']) for j in jobs]
            or p['backend_sha256'] != self.backend.identity_sha256
            or r['request_sha256'] != request.sha256 or r['backend_sha256'] != self.backend.identity_sha256
            or r['raw_inputs_to_qwen'] is not False or r['remote_provider_calls'] != 0
            or len(r['rows']) != len(jobs)):
            raise ValueError('local summary request/response binding changed')
        for job, row in zip(jobs, r['rows'], strict=True):
            if row['job_sha256'] != job.prompt_sha256:
                raise ValueError('local batch changed job attribution')
            self.completed_attempts.add((job.prompt_sha256,p['attempt']))
            try:
                if row['stopped'] is not True:
                    raise ValueError('generation did not reach EOS')
                summary = parse_spine_summary(row['response'], job)
            except (ValueError, TypeError):
                continue
            old = self.cache.values.get(job.prompt_sha256)
            if old is not None and old != summary:
                raise ValueError('completed local summary changed')
            self.cache.values[job.prompt_sha256] = summary

    def replay(self):
        for path in sorted((self.root/'requests').glob('*.json')):
            request = read_sealed_json(path)
            response_path = self.root/'responses'/(request.sha256+'.json')
            reservation = self.root/'executions'/(request.sha256+'.reserved')
            if response_path.exists():
                self.accept(request, read_sealed_json(response_path))
            elif reservation.exists():
                raise ValueError('local execution lacks a completed response; refusing an implicit retry')

    def resolve(self, pending, stage, wave):
        advanced = False
        for attempt in range(3):
            missing = [j for sha,j in pending.items() if sha not in self.cache.values
                       and (sha,attempt) not in self.completed_attempts]
            for start in range(0,len(missing),self.backend.max_batch_size):
                jobs = tuple(missing[start:start+self.backend.max_batch_size])
                body = {'preflight_sha256':self.preflight.sha256,
                    'backend_sha256':self.backend.identity_sha256, 'attempt':attempt,
                    'jobs':[asdict(j) for j in jobs], 'messages':[job_messages(j,attempt) for j in jobs],
                    'raw_inputs_to_qwen':False}
                request,_ = publish_sealed_json(self.root/'requests'/(identity_sha256(body)+'.json'),body)
                response_path = self.root/'responses'/(request.sha256+'.json')
                if response_path.exists():
                    self.accept(request,read_sealed_json(response_path))
                    continue
                if self.jobs+len(jobs) > self.budget:
                    if advanced:
                        return  # Let the compiler publish newly completed sources first.
                    raise NeedsProviderWork('local generation allowance exhausted')
                reservation = self.root/'executions'/(request.sha256+'.reserved')
                reservation.parent.mkdir(parents=True,exist_ok=True)
                with reservation.open('x',encoding='utf-8') as handle:
                    handle.write(request.sha256+'\n')
                self.jobs += len(jobs)
                self.calls += 1
                result = self.backend.generate(jobs,attempt)
                response,_ = publish_sealed_json(response_path,{'request_sha256':request.sha256,**result})
                before = len(self.cache.values)
                self.accept(request,response)
                advanced |= len(self.cache.values) > before
                print({'local_batches':self.calls,'local_jobs':self.jobs,'attempt':attempt,
                    'accepted_jobs':len(self.cache.values)-before,'elapsed_s':result['elapsed_s'],
                    'tokens_per_second':result['tokens_per_second'],
                    'peak_gpu_allocated_GiB':result['peak_gpu_allocated_GiB'],
                    'response_sha256':response.sha256},flush=True)
        if any(sha not in self.cache.values for sha in pending):
            raise ValueError('local summary recovery exhausted two bounded attempts')


def run(cached_root, root, backend, budget=0):
    if cached_root.resolve() == root.resolve():
        raise ValueError('local compilation requires a separate successor root')
    prior = read_sealed_json(cached_root/'preflight.json')
    p = prior.payload
    for name,digest in p['implementation'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('cached parent compiler changed')
    cache = read_sealed_json(cached_root/'input-cache.json')
    leaf = read_sealed_json(Path(p['leaf_root'])/'hierarchy.json')
    atoms = read_sealed_json(Path(p['atoms_path']))
    manifest = read_sealed_json(Path(p['index_root'])/'index.json')
    if (cache.sha256 != p['input_cache_sha256'] or cache.payload['atoms_sha256'] != atoms.sha256
        or leaf.sha256 != p['leaf_projection_sha256'] or atoms.sha256 != p['atoms_sha256']
        or manifest.sha256 != p['serving_index_sha256'] or p['raw_inputs_to_qwen'] is not False):
        raise ValueError('cached parent inputs changed')
    index,plans = prepare_sources(leaf,atoms,manifest)
    method = {'format':'memory-condense-local-spine-parents-v1',
        'backend':backend.identity,'backend_sha256':backend.identity_sha256,
        'max_channel_tokens':128,'max_batch_size':backend.max_batch_size,
        'max_recovery_attempts':2,'recovery_words':[48,24],'automatic_retries':0,
        'raw_inputs_to_qwen':False,'query_independent':True,
        'topology':'restore every authenticated source-local attention cut',
        'leaf_policy':'preserve original leaf descriptors byte for byte',
        'cache_policy':'reuse bound completed gateway Qwen merges; new merges use local NF4 Qwen',
        'implementation':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}}
    preflight,_ = publish_sealed_json(root/'preflight.json',{**method,
        'method_policy_sha256':identity_sha256(method),'cached_root':str(cached_root.resolve()),
        'cached_preflight_sha256':prior.sha256,'input_cache_sha256':cache.sha256,
        **{k:p[k] for k in ('leaf_projection_sha256','atoms_sha256','serving_index_sha256',
            'leaf_index_sha256','raw_span_population_sha256','raw_token_proxy','leaf_root','atoms_path','index_root')}})
    topology,_ = publish_sealed_json(root/'topology.json',{'preflight_sha256':preflight.sha256,
        'leaf_count':len(index.sections),'parent_count':sum(len(plan.parents) for plan in plans.values()),
        'sources':[{'source_id':sid,'root_section_id':plan.root_section_id,
            'plan_sha256':plan.receipt_sha256,'parent_count':len(plan.parents)} for sid,plan in plans.items()],
        'raw_text_reads':0,'new_attention_calls':0})
    journal = LocalJournal(root,preflight,backend,budget)
    journal.cache.values.update(cache.payload['summaries'])
    journal.replay()
    parts,progress = compile_available(plans,journal,root)
    result = {'output_root':str(root.resolve()),'preflight_sha256':preflight.sha256,
        'topology_sha256':topology.sha256,'progress_sha256':progress.sha256,
        'complete_sources':len(parts),'source_count':len(plans),
        'complete_parents':progress.payload['completed_parent_count'],
        'parent_count':topology.payload['parent_count'],'leaf_count':len(index.sections),
        'next_dependency_jobs':len(progress.payload['next_dependency_jobs']),
        'raw_token_proxy':p['raw_token_proxy'],'new_local_jobs':journal.jobs,
        'new_local_batches':journal.calls,'remote_provider_calls':0,'complete_namespace':len(parts)==len(plans)}
    if len(parts)==len(plans):
        restored = SectionSummaryIndex(tuple(s for sid in plans for s in parts[sid].sections))
        if tuple(s for s in restored.sections if not s.child_section_ids) != index.sections:
            raise ValueError('local restoration changed original leaves')
        artifact,_ = publish_sealed_json(root/'hierarchy.json',{
            'preflight_sha256':preflight.sha256,'topology_sha256':topology.sha256,
            'input_cache_sha256':cache.sha256,'progress_sha256':progress.sha256,
            'leaf_projection_sha256':leaf.sha256,'leaf_index_sha256':index.receipt_sha256,
            'index_json':restored.to_json(),'root_section_ids':[p.root_section_id for p in plans.values()],
            'leaf_count':len(index.sections),'parent_count':topology.payload['parent_count'],
            'raw_span_population_sha256':leaf.payload['raw_span_population_sha256'],
            'complete_namespace':True,'parent_summary_compilation_complete':True,
            'raw_inputs_to_qwen':False,'summary_merge_jobs':len(journal.cache.values),'target_gate_passed':False})
        result['hierarchy_sha256'] = artifact.sha256
    print(result,flush=True)
    return result


def population(source_population,root,backend,budget=0,offsets=None):
    source = read_sealed_json(source_population)
    if [r['offset'] for r in source.payload['records']] != list(range(0,100,10)):
        raise ValueError('local compilation requires the ten bound cached memory plans')
    selected = list(range(0,100,10)) if offsets is None else list(offsets)
    if not selected or len(set(selected)) != len(selected) or any(o not in range(0,100,10) for o in selected):
        raise ValueError('invalid local namespace selection')
    records = []
    for row in source.payload['records']:
        if row['offset'] not in selected:
            continue
        old = Path(row['output_root'])
        if read_sealed_json(old/'preflight.json').sha256 != row['preflight_sha256']:
            raise ValueError('cached population binding changed')
        records.append({'offset':row['offset'],**run(old,root/f"offset-{row['offset']:03}",backend,budget)})
    body = {'format':'memory-condense-local-parent-population-v1',
        'source_population_sha256':source.sha256,'records':records,
        'complete_namespace_population':len(records)==10 and all(r['complete_namespace'] for r in records),
        'raw_inputs_to_qwen':False,'remote_provider_calls':0,'target_gate_passed':False}
    artifact,_ = publish_sealed_json(root/'populations'/(identity_sha256(body)+'.json'),body)
    print({'population_path':str(artifact.path),'population_sha256':artifact.sha256,
           'complete_namespace_population':body['complete_namespace_population']},flush=True)
    return artifact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-population',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--probe-root',type=Path,default=Path('eval_results/local-qwen-parent-summary-probe-20260910-r1'))
    parser.add_argument('--dependency-root',type=Path,default=Path('.cache/local-qwen-runtime/site-packages'))
    parser.add_argument('--model-root',type=Path,default=Path('../../.cache/models/Qwen3-8B'))
    parser.add_argument('--max-new-jobs-per-namespace',type=int,default=0)
    parser.add_argument('--offset',type=int,action='append')
    args = parser.parse_args()
    if args.max_new_jobs_per_namespace < 0:
        parser.error('local job allowance must be nonnegative')
    backend = LocalQwenBackend(args.probe_root,args.dependency_root,args.model_root)
    population(args.source_population,args.output_root,backend,args.max_new_jobs_per_namespace,args.offset)
