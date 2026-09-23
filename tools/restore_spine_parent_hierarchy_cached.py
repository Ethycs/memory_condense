"""Reuse authenticated Qwen merges and retain completed source parent trees."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.spine_merge_batch import PendingMerge
from tools.build_spine_corpus_hierarchy import GATEWAY, MODEL
from tools.build_spine_corpus_hierarchy_resilient import NeedsProviderWork, RecoveryJournal
from tools.compile_spine_leaf_projection import frozen_cache
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.restore_spine_parent_hierarchy import IMPLEMENTATION as BASE_IMPLEMENTATION, prepare_sources


IMPLEMENTATION = (*BASE_IMPLEMENTATION, 'tools/restore_spine_parent_hierarchy_cached.py',
                  'tools/compile_spine_leaf_projection.py')


def compile_available(plans, journal, root):
    """Persist complete sources before waiting on another summary dependency."""
    done, bindings, wave = {}, {}, 0
    while len(done) < len(plans):
        pending = {}
        for source, plan in plans.items():
            if source in done:
                continue
            try:
                part = plan.compile(summarize=journal.cache, summarizer_identity=journal.preflight.sha256)
            except PendingMerge as missing:
                pending.setdefault(missing.request.prompt_sha256, missing.request)
                continue
            path = root/'source-parts'/(identity_sha256(source)+'.json')
            artifact,_ = publish_sealed_json(path, {
                'preflight_sha256':journal.preflight.sha256, 'source_id':source,
                'plan_sha256':plan.receipt_sha256, 'root_section_id':plan.root_section_id,
                'parent_count':len(plan.parents), 'index_json':part.to_json(),
                'complete_source':True, 'complete_namespace':False, 'raw_inputs_to_qwen':False})
            done[source], bindings[source] = part, artifact.sha256
        body = {'preflight_sha256':journal.preflight.sha256,
            'completed_source_parts':[{'source_id':s,'sha256':bindings[s]} for s in plans if s in done],
            'completed_source_count':len(done),
            'completed_parent_count':sum(len(plans[s].parents) for s in done),
            'total_source_count':len(plans), 'total_parent_count':sum(len(p.parents) for p in plans.values()),
            'next_dependency_jobs':list(pending), 'complete_namespace':len(done)==len(plans),
            'raw_inputs_to_qwen':False}
        progress,_ = publish_sealed_json(root/'progress'/(identity_sha256(body)+'.json'),body)
        print({'progress_sha256':progress.sha256,'complete_sources':len(done),
               'complete_parents':body['completed_parent_count'],'next_dependency_jobs':len(pending)},flush=True)
        if not pending:
            return done, progress
        try:
            journal.resolve(pending,'cached_source_parents',wave)
        except NeedsProviderWork:
            return done, progress
        wave += 1
    raise AssertionError('parent progress loop exited without a final state')


def run(source_plan_root, root, enable=False, budget=0):
    if source_plan_root.resolve() == root.resolve():
        raise ValueError('cached parent restoration requires a separate successor root')
    prior = read_sealed_json(source_plan_root/'preflight.json')
    p = prior.payload
    for name,digest in p['implementation'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('source parent compiler changed')
    leaf_root, atoms_path, index_root = (Path(p[k]) for k in ('leaf_root','atoms_path','index_root'))
    leaf = read_sealed_json(leaf_root/'hierarchy.json')
    projection = read_sealed_json(leaf_root/'preflight.json')
    atoms = read_sealed_json(atoms_path)
    manifest = read_sealed_json(index_root/'index.json')
    if (leaf.sha256 != p['leaf_projection_sha256'] or projection.sha256 != p['leaf_projection_preflight_sha256']
        or atoms.sha256 != p['atoms_sha256'] or manifest.sha256 != p['serving_index_sha256']
        or projection.sha256 != leaf.payload['preflight_sha256']):
        raise ValueError('source parent plan input binding changed')
    index,plans = prepare_sources(leaf,atoms,manifest)
    # The existing importer verifies frozen inherited caches, summary-only
    # request reconstruction, completed response journals and accepted repairs.
    # It opens no raw corpus and never retries incomplete parent calls.
    cache,_ = frozen_cache(root,atoms,[leaf_root])
    method = {'format':'memory-condense-cached-spine-parents-v1','model':MODEL,'gateway':GATEWAY,
        'max_channel_tokens':128,'max_jobs_per_batch':8,'max_concurrency':4,'retries':0,
        'max_recovery_calls_per_failed_job':2,'recovery_words':[48,24],
        'raw_inputs_to_qwen':False,'query_independent':True,
        'topology':'restore every authenticated source-local attention cut',
        'leaf_policy':'preserve original leaf descriptors byte for byte',
        'cache_policy':'frozen completed summary merges from the bound leaf compiler',
        'source_checkpoint_policy':'publish each complete source before resolving further dependencies',
        'implementation':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}}
    preflight,_ = publish_sealed_json(root/'preflight.json',{**method,
        'method_policy_sha256':identity_sha256(method),'source_plan_root':str(source_plan_root.resolve()),
        'source_plan_sha256':prior.sha256,'input_cache_sha256':cache.sha256,
        **{k:p[k] for k in ('leaf_projection_sha256','leaf_projection_preflight_sha256','atoms_sha256',
            'serving_index_sha256','leaf_index_sha256','raw_span_population_sha256','raw_token_proxy',
            'leaf_root','atoms_path','index_root')},'complete_namespace':True})
    topology,_ = publish_sealed_json(root/'topology.json',{
        'preflight_sha256':preflight.sha256,'leaf_count':len(index.sections),
        'parent_count':sum(len(plan.parents) for plan in plans.values()),
        'sources':[{'source_id':sid,'root_section_id':plan.root_section_id,
                    'plan_sha256':plan.receipt_sha256,'parent_count':len(plan.parents)} for sid,plan in plans.items()],
        'raw_text_reads':0,'new_attention_calls':0})
    journal = RecoveryJournal(root,preflight,enable,budget)
    journal.cache.values.update(cache.payload['summaries'])
    journal.replay()
    parts,progress = compile_available(plans,journal,root)
    result = {'output_root':str(root.resolve()),'preflight_sha256':preflight.sha256,
        'topology_sha256':topology.sha256,'progress_sha256':progress.sha256,
        'input_cache_sha256':cache.sha256,'cached_jobs':len(cache.payload['summaries']),
        'leaf_count':len(index.sections),'parent_count':topology.payload['parent_count'],
        'source_count':len(plans),'complete_sources':len(parts),
        'complete_parents':progress.payload['completed_parent_count'],
        'next_dependency_jobs':len(progress.payload['next_dependency_jobs']),
        'raw_token_proxy':p['raw_token_proxy'],'new_calls':journal.calls,
        'complete_namespace':len(parts)==len(plans)}
    if len(parts) != len(plans):
        print(result,flush=True)
        return result
    restored = SectionSummaryIndex(tuple(s for sid in plans for s in parts[sid].sections))
    if tuple(s for s in restored.sections if not s.child_section_ids) != index.sections:
        raise ValueError('cached restoration changed the complete original leaf population')
    artifact,_ = publish_sealed_json(root/'hierarchy.json',{
        'preflight_sha256':preflight.sha256,'topology_sha256':topology.sha256,
        'input_cache_sha256':cache.sha256,'progress_sha256':progress.sha256,
        'leaf_projection_sha256':leaf.sha256,'leaf_index_sha256':index.receipt_sha256,
        'index_json':restored.to_json(),'root_section_ids':[plan.root_section_id for plan in plans.values()],
        'leaf_count':len(index.sections),'parent_count':topology.payload['parent_count'],
        'raw_span_population_sha256':leaf.payload['raw_span_population_sha256'],
        'complete_namespace':True,'parent_summary_compilation_complete':True,
        'raw_inputs_to_qwen':False,'summary_merge_jobs':len(journal.cache.values),'target_gate_passed':False})
    result['hierarchy_sha256'] = artifact.sha256
    print(result,flush=True)
    return result


def population(source_population,root,enable=False,budget=0):
    source = read_sealed_json(source_population)
    if [r['offset'] for r in source.payload['records']] != list(range(0,100,10)):
        raise ValueError('cached restoration requires the ten complete source plans')
    records=[]
    for row in source.payload['records']:
        old = Path(row['output_root'])
        if read_sealed_json(old/'preflight.json').sha256 != row['preflight_sha256']:
            raise ValueError('source population parent plan changed')
        record=run(old,root/f"offset-{row['offset']:03}",enable,budget)
        records.append({'offset':row['offset'],**record})
    payload={'format':'memory-condense-cached-parent-population-v1','source_population_sha256':source.sha256,
        'source_population_path':str(source_population.resolve()),'records':records,
        'complete_namespace_population':all(r['complete_namespace'] for r in records),
        'new_calls':sum(r['new_calls'] for r in records),'raw_inputs_to_qwen':False,'target_gate_passed':False}
    artifact,_=publish_sealed_json(root/'populations'/(identity_sha256(payload)+'.json'),payload)
    print({'population_path':str(artifact.path) if hasattr(artifact,'path') else str(root/'populations'/(identity_sha256(payload)+'.json')),
        'population_sha256':artifact.sha256,'complete_sources':sum(r['complete_sources'] for r in records),
        'complete_parents':sum(r['complete_parents'] for r in records),
        'next_dependency_jobs':sum(r['next_dependency_jobs'] for r in records),
        'new_calls':payload['new_calls'],'complete_namespace_population':payload['complete_namespace_population']},flush=True)
    return artifact


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','run'))
    parser.add_argument('--source-population',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--enable-provider',action='store_true')
    parser.add_argument('--max-new-calls-per-namespace',type=int,default=0)
    args=parser.parse_args()
    if args.max_new_calls_per_namespace<0 or (args.enable_provider and args.phase!='run'):
        parser.error('provider execution requires run and a nonnegative allowance')
    population(args.source_population,args.output_root,args.enable_provider,args.max_new_calls_per_namespace)
