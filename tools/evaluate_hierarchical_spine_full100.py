"""Fresh full100 joint evaluation of summary-tree Qwen routing and flat control."""
import argparse
import hashlib
from pathlib import Path
import time

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS, parse_binary_judge_verdict
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.eval.thread_local_provider_v2 import ThreadLocalProvider
from memory_condense.eval.streaming_latency import measure_streaming_answer, latency_distribution
from memory_condense.search.episodes.qwen_episode_signal import qwen_linker_identity
from tools import evaluate_spine_relative_reservation as previous
from tools.evaluate_spine_as_of import answer_messages as flat_messages, load_preflight as load_as_of
from tools.evaluate_spine_reader_residual import load_references
from tools.hierarchical_spine_memory import ResidentMemory
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized
from tools.run_spine_reader_after_timeout import require_idle


MODEL, GATEWAY = 'codex_sdk/gpt-5.6-terra', 'https://central-dev.zt:4000/v1'
MEMORY_ARMS = ('flat', 'hierarchy')
ARMS = (*MEMORY_ARMS, 'hierarchy_api', 'short_api')
POLICY = {'question_count':100, 'answer_calls':400, 'logical_judgments':200,
    'max_output_tokens':256, 'max_prompt_tokens':5500, 'max_context_tokens':3072, 'max_raw_spans':128,
    'timed_concurrency':1, 'automatic_retries':0, 'cached_query_vectors':False, 'cached_predictions':False,
    'all_answers_before_judging':True, 'accuracy_threshold':.95, 'latency_ratio_limit':1.10,
    'root_shortlist':8, 'beam':4, 'max_depth':16, 'max_qwen_workspace_tokens':4096,
    'raw_inputs_to_qwen':False, 'qwen_prefix_layers':6, 'qwen_attention_layer':5,
    'qwen_precision':'FP16 weights/forward; FP32 softmax and pooled readout',
    'reader':'unchanged v2', 'reader_temperature':'omitted', 'raw_renderer':'unchanged flat sections',
    'cold_setup_excluded_from_warm_latency':True, 'timed_qwen_attention_recomputed':True,
    'live_attention_selection_must_match_preparation':True, 'scalar_roundoff_may_differ':True}
IMPLEMENTATION = tuple(dict.fromkeys((*previous.IMPLEMENTATION,
    'tools/evaluate_hierarchical_spine_full100.py', 'tools/hierarchical_spine_memory.py',
    'src/memory_condense/search/bounded_spine_hierarchy.py',
    'src/memory_condense/search/summary_shortlist_attention.py',
    'src/memory_condense/search/episodes/qwen_episode_signal.py',
    'src/memory_condense/modeling/qwen_prefix.py',
    'src/memory_condense/associations/qwen_memory_linker.py',
    'src/memory_condense/eval/streaming_latency.py',
    'src/memory_condense/eval/thread_local_provider.py',
    'src/memory_condense/eval/thread_local_provider_v2.py',
    'src/memory_condense/eval/fast_completion_runtime.py',
    'tools/run_spine_reader_after_timeout.py')))


def implementation():
    return {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}


def call_order(ordinal):
    pair = ('hierarchy', 'hierarchy_api') if ordinal % 2 == 0 else ('hierarchy_api', 'hierarchy')
    groups = [('flat',), pair, ('short_api',)]
    offset = ordinal % 3
    return tuple(a for group in groups[offset:] + groups[:offset] for a in group)


def messages(question, hydrated=None):
    result = flat_messages(question, hydrated, policy='as_of')
    if count_chat_prompt_token_proxy(result) > POLICY['max_prompt_tokens']:
        raise ValueError('hierarchy reader exceeds the original prompt budget')
    return result


def new_linker():
    from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
    from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
    print('Loading resident Qwen prefix for summary-only hierarchy traversal...', flush=True)
    encoder = Qwen3PrefixEncoder(Path('../../.cache/models/Qwen3-8B').resolve(),
                                layers=6, device='cuda', dtype='float16')
    return QwenMemoryLinker(encoder, layer=5, max_candidates=8, max_workspace_tokens=4096)


def resident(payload, parent, linker):
    return ResidentMemory(Path(payload['index_root']), payload['index_manifest_sha256'],
        Path(payload['addresses_root']), Path(payload['atoms_path']), Path(payload['facets_root']),
        payload['addresses_sha256'], payload['atoms_sha256'], payload['facets_sha256'],
        parent_root=Path(parent['root']), parent_sha256=parent['sha256'], linker=linker)


def admit_parents(population_path, original):
    population = read_sealed_json(population_path)
    rows = population.payload['records']
    if [r['offset'] for r in rows] != list(range(0,100,10)):
        raise ValueError('hierarchy comparison requires all ten complete memories')
    namespaces, methods = [], set()
    for row, binding in zip(rows, original.payload['source_bindings'], strict=True):
        source = load_as_of(Path(binding['root']))
        if source.sha256 != binding['sha256'] or source.payload['shard_offset'] != row['offset']:
            raise ValueError('source namespace changed')
        p = source.payload
        root = Path(row['output_root'])
        # All admission precedes loading Qwen or creating any answer requests.
        if not (root / 'hierarchy.json').is_file():
            raise ValueError(f"parent generation incomplete at offset {row['offset']:03}")
        artifact = read_sealed_json(root / 'hierarchy.json')
        preflight = read_sealed_json(root / 'preflight.json')
        topology = read_sealed_json(root / 'topology.json')
        a, f = artifact.payload, preflight.payload
        if (a['preflight_sha256'] != preflight.sha256 or a['topology_sha256'] != topology.sha256
            or f['serving_index_sha256'] != p['index_manifest_sha256']
            or a['parent_summary_compilation_complete'] is not True or a['complete_namespace'] is not True
            or a['raw_inputs_to_qwen'] is not False or a['parent_count'] <= 0
            or p['raw_token_proxy'] < 1_000_000 or f['raw_token_proxy'] != p['raw_token_proxy']):
            raise ValueError('parent hierarchy is not complete and bound to the evaluated memory')
        methods.add(f['method_policy_sha256'])
        namespaces.append({**binding, 'offset':row['offset'], 'raw_token_proxy':p['raw_token_proxy'],
            'parent':{'root':str(root.resolve()), 'sha256':artifact.sha256,
                      'preflight_sha256':preflight.sha256, 'topology_sha256':topology.sha256}})
    if len(methods) != 1:
        raise ValueError('all ten memories must use the same parent compilation method')
    return population, namespaces


def attention_signature(audit):
    plan = audit['attention_plan']
    receipt = plan['attention_receipt']
    rounds = receipt['rounds']
    if (not rounds or len(rounds) > 16 or receipt['raw_content_inspections'] != 0
        or receipt['retained_transformer_token_state_bytes'] != 0
        or audit['raw_inputs_to_qwen'] is not False or audit['live_query_embedding'] is not True
        or any(r['model_passes'] != 1 or r['max_workspace_candidates'] > 8
               or r['max_workspace_tokens'] > 4096 for r in rounds)):
        raise ValueError('live hierarchy lacks complete bounded summary-only attention')
    return {'parent':audit['parent_hierarchy_sha256'], 'query':plan['query_sha256'],
        'routes':plan['routes'], 'linker':receipt['linker_identity_json'],
        'projected_plan':audit['projection']['projected_plan_sha256'],
        'rounds':[{key:r[key] for key in ('candidate_section_ids','selected_section_ids',
                   'model_passes','max_workspace_candidates','max_workspace_tokens','candidate_inspections')} for r in rounds]}


def build(memory, question, arm):
    audit = None
    if arm == 'hierarchy':
        plan, audit = memory.hierarchical_plan(question['retrieval_query'], question['prompt_question'])
        hydrated = hydrate_section_plan(plan, load_turn=memory.turns.get,
                                       max_context_tokens=3072, max_raw_spans=128)
    elif arm == 'flat':
        hydrated = memory.retrieve(question['retrieval_query'], 'source_spine_relative_reservation', question['prompt_question'])
    else:
        raise ValueError('build requires a live memory arm')
    return messages(question, hydrated), hydrated.identity_payload(), audit


def validate_calls(calls):
    if len(calls) != 400:
        raise ValueError('joint comparison requires all400 requests')
    for ordinal in range(100):
        group = calls[4*ordinal:4*ordinal+4]
        if [c['arm'] for c in group] != list(call_order(ordinal)):
            raise ValueError('counterbalanced call order changed')
        q = group[0]['question']
        by_arm = {c['arm']:c for c in group}
        for i,c in enumerate(group,4*ordinal):
            if (c['question'] != q or q['ordinal'] != ordinal or c['call_index'] != i
                or identity_sha256(c['messages']) != c['messages_sha256']
                or c['messages'][0] != messages(q)[0]
                or count_chat_prompt_token_proxy(c['messages']) > 5500):
                raise ValueError('question, reader, prompt binding or budget changed')
        if by_arm['hierarchy']['messages'] != by_arm['hierarchy_api']['messages']:
            raise ValueError('identical-evidence API prompt changed')
        if by_arm['short_api']['messages'] != messages(q):
            raise ValueError('short API reader or question changed')


def prepare(root, source_root, population_path):
    original = previous.load_preflight(source_root)
    population, namespaces = admit_parents(population_path, original)
    code, calls = implementation(), []
    linker = new_linker()
    linker_identity = qwen_linker_identity(linker, strict=True)
    for namespace in namespaces:
        memory = resident(load_as_of(Path(namespace['root'])).payload, namespace['parent'], linker)
        try:
            for old in original.payload['cases'][namespace['offset']:namespace['offset']+10]:
                q = old['question']
                values = {arm:build(memory,q,arm) for arm in MEMORY_ARMS}
                if values['flat'][0] != old['messages']['relative_reservation']:
                    raise ValueError('live flat control changed')
                attention_signature(values['hierarchy'][2])
                prompts = {arm:values[arm][0] for arm in MEMORY_ARMS}
                prompts.update(hierarchy_api=prompts['hierarchy'], short_api=messages(q))
                evidence,_ = publish_sealed_json(root/'evidence'/f"{q['ordinal']:03}.json", {
                    'question':q, 'messages':prompts, 'hydration':{a:values[a][1] for a in MEMORY_ARMS},
                    'hierarchy_audit':values['hierarchy'][2]})
                for arm in call_order(q['ordinal']):
                    calls.append({'call_index':len(calls), 'question':q, 'arm':arm,
                        'messages':prompts[arm], 'messages_sha256':identity_sha256(prompts[arm]),
                        'evidence_sha256':evidence.sha256})
                print({'prepared_ordinal':q['ordinal'], 'attention_levels':len(values['hierarchy'][2]['attention_plan']['attention_receipt']['rounds'])},flush=True)
        finally:
            memory.encoder.close()
    validate_calls(calls)
    if code != implementation() or qwen_linker_identity(linker,strict=True) != linker_identity:
        raise ValueError('implementation or Qwen identity changed during preparation')
    preflight,_ = publish_sealed_json(root/'preflight.json', {
        'format':'memory-condense-hierarchical-spine-joint-full100-v1',
        'source_root':str(source_root.resolve()), 'source_preflight_sha256':original.sha256,
        'parent_population_path':str(population_path.resolve()), 'parent_population_sha256':population.sha256,
        'namespaces':namespaces, 'calls':calls, 'implementation':code, 'policy':POLICY,
        'model':MODEL, 'gateway':GATEWAY, 'gold_loaded':False, 'qwen_identity':linker_identity})
    print({'preflight_sha256':preflight.sha256,'fresh_answer_calls_required':400},flush=True)


def load_preflight(root):
    artifact = read_sealed_json(root/'preflight.json')
    p = artifact.payload
    if p['implementation'] != implementation() or p['policy'] != POLICY or p['model'] != MODEL or p['gateway'] != GATEWAY or p['gold_loaded'] is not False:
        raise ValueError('frozen hierarchy experiment changed')
    original = previous.load_preflight(Path(p['source_root']))
    population,namespaces = admit_parents(Path(p['parent_population_path']),original)
    if original.sha256 != p['source_preflight_sha256'] or population.sha256 != p['parent_population_sha256'] or namespaces != p['namespaces']:
        raise ValueError('complete source or parent population changed')
    validate_calls(p['calls'])
    for ordinal,old in enumerate(original.payload['cases']):
        evidence = read_sealed_json(root/'evidence'/f'{ordinal:03}.json')
        e = evidence.payload
        if e['question'] != old['question'] or e['messages']['flat'] != old['messages']['relative_reservation']:
            raise ValueError('source question or flat control changed')
        attention_signature(e['hierarchy_audit'])
        for call in p['calls'][4*ordinal:4*ordinal+4]:
            if call['question'] != e['question'] or call['evidence_sha256'] != evidence.sha256 or call['messages'] != e['messages'][call['arm']]:
                raise ValueError('prepared evidence changed')
    return artifact


def recorded(root, preflight):
    results, evidence_cache = [], {}
    for call in preflight.payload['calls']:
        prefix = root/'journal'/f"{call['call_index']:03}"
        if not prefix.with_suffix('.response.json').exists():
            if prefix.with_suffix('.reserved').exists() or prefix.with_suffix('.request.json').exists():
                raise ValueError('unacknowledged stream; preserve its root')
            continue
        request = read_sealed_json(prefix.with_suffix('.request.json'))
        response = read_sealed_json(prefix.with_suffix('.response.json'))
        r,m = response.payload,response.payload['measurement']
        if (request.payload != {'preflight_sha256':preflight.sha256,'call':call}
            or r['request_sha256'] != request.sha256 or r['messages'] != call['messages']
            or m['messages_sha256'] != call['messages_sha256'] or m['prediction_sha256'] != quote_sha256(m['prediction'])
            or r['evidence_sha256'] != call['evidence_sha256'] or m['model'] != MODEL or m['max_tokens'] != 256):
            raise ValueError('streamed request or response binding changed')
        ordinal = call['question']['ordinal']
        if ordinal not in evidence_cache:
            evidence_cache[ordinal] = read_sealed_json(root/'evidence'/f'{ordinal:03}.json')
        evidence = evidence_cache[ordinal]
        if evidence.sha256 != call['evidence_sha256']:
            raise ValueError('response lost its prepared evidence binding')
        expected = evidence.payload['hydration'][call['arm']] if call['arm'] in MEMORY_ARMS else None
        if r['hydration'] != expected:
            raise ValueError('live response changed the exact raw evidence')
        if call['arm'] == 'hierarchy':
            if attention_signature(r['hierarchy_audit']) != attention_signature(evidence.payload['hierarchy_audit']):
                raise ValueError('live hierarchical attention changed its selected path')
        elif r['hierarchy_audit'] is not None:
            raise ValueError('API or flat arm unexpectedly claims hierarchy attention')
        results.append((call,response))
    if [c['call_index'] for c,_ in results] != list(range(len(results))):
        raise ValueError('stream journal has a gap')
    return results


def seal_answers(root,preflight):
    observations = recorded(root,preflight)
    if len(observations) != 400:
        raise ValueError('all400 fresh answers must seal before references open')
    artifact,_ = publish_sealed_json(root/'answers.json',{'preflight_sha256':preflight.sha256,
        'rows':[{'call_index':c['call_index'],'ordinal':c['question']['ordinal'],'arm':c['arm'],
                 'response_sha256':r.sha256,'prediction':r.payload['measurement']['prediction'],
                 'prediction_sha256':r.payload['measurement']['prediction_sha256']} for c,r in observations]})
    return artifact,observations


def joint_statistics(observations,judged):
    if (len(observations) != 400 or len(judged) != 200
        or any(sorted(c['question']['ordinal'] for c,_ in observations if c['arm']==a) != list(range(100)) for a in ARMS)
        or any(sorted(r['ordinal'] for r in judged if r['arm']==a) != list(range(100)) for a in MEMORY_ARMS)):
        raise ValueError('joint gate requires complete matched full100 populations')
    lookup = {(c['question']['ordinal'],c['arm']):r for c,r in observations}
    for row in judged:
        response = lookup[row['ordinal'],row['arm']]
        if row['prediction_sha256'] != response.payload['measurement']['prediction_sha256'] or row['response_sha256'] != response.sha256:
            raise ValueError('quality and latency refer to different responses')
    scores = {a:sum(r['correct'] for r in judged if r['arm']==a) for a in MEMORY_ARMS}
    timing = {a:{metric:latency_distribution([r.payload['measurement'][metric] for c,r in observations if c['arm']==a])
                 for metric in ('prepare_s','e2e_ttft_s','e2e_total_s')} for a in ARMS}
    if any(timing[a][metric][stat] <= 0 for a in ('hierarchy_api','short_api')
           for metric in ('e2e_ttft_s','e2e_total_s') for stat in ('median_s','p95_s')):
        raise ValueError('API latency denominators must be positive')
    ratios = {a:{metric:{stat:timing['hierarchy'][metric][stat]/timing[a][metric][stat]
                        for stat in ('median_s','p95_s')} for metric in ('e2e_ttft_s','e2e_total_s')}
              for a in ('hierarchy_api','short_api')}
    quality = scores['hierarchy'] >= 95
    latency = all(v <= 1.10 for metrics in ratios.values() for stats in metrics.values() for v in stats.values())
    finished = all(r.payload['measurement']['finish_reason']=='stop' for _,r in observations)
    return {'accuracy':scores,'latency':timing,'candidate_latency_ratios':ratios,
        'candidate_accuracy_passed':quality,'candidate_latency_passed':latency,
        'all_streams_finished_normally':finished,'same_streamed_answers_scored':True,
        'target_gate_passed':quality and latency and finished,
        'flat_control_has_no_independent_joint_gate':True}


def judge(root,enable=False):
    preflight = load_preflight(root)
    answers,observations = seal_answers(root,preflight)
    _,references = load_references()
    rows=[]
    for call,response in observations:
        if call['arm'] not in MEMORY_ARMS:
            continue
        q = references[call['question']['ordinal']]
        if q.question_id != call['question']['question_id']:
            raise ValueError('reference question changed')
        m=response.payload['measurement']
        rows.append({'ordinal':call['question']['ordinal'],'question_id':q.question_id,'arm':call['arm'],
            'prediction':m['prediction'],'prediction_sha256':m['prediction_sha256'],'response_sha256':response.sha256,
            'reference_sha256':quote_sha256(q.answer),
            'messages':build_judge_prompt(call['question']['retrieval_query'],q.answer,m['prediction'])})
    inputs,_=publish_sealed_json(root/'judge-preflight.json',{'answers_sha256':answers.sha256,'rows':rows})
    def factory(client):
        return FastCompletionRuntime(checkpoint_dir=root/'judge-checkpoints',
            prompt_population=[r['messages'] for r in rows],model='codex_sdk/gpt-5.6-sol',client=client,
            max_prompt_tokens=4096,max_new_tokens=JUDGE_MAX_TOKENS,max_concurrency=8,retries=0,
            request_options={'temperature':0},benchmark_provenance={'binding_sha256':inputs.sha256,'phase':'judge'})
    audit=factory(None)
    try:
        remaining=audit.population.unique_prompt_count-len(_authenticated_records(audit))
    finally:
        audit.close()
    batch,calls,hits,_=_run_exactly_authorized(runtime_factory=factory,authorized_provider_calls=remaining,
        enable_provider=enable,client_factory=lambda:ThreadLocalProvider(lambda:_completion_client('LITELLM_KEY',GATEWAY)))
    judged=[{**{k:v for k,v in row.items() if k!='messages'},'verdict':verdict,
             'correct':parse_binary_judge_verdict(verdict)} for row,verdict in zip(rows,batch.logical_completions,strict=True)]
    stats=joint_statistics(observations,judged)
    result,_=publish_sealed_json(root/'joint-report.json',{'preflight_sha256':preflight.sha256,
        'answers_sha256':answers.sha256,'judge_preflight_sha256':inputs.sha256,'rows':judged,**stats,
        'judge_response_journal_shas':[r.response_journal_sha256 for r in batch.unique_records]})
    print({'report_sha256':result.sha256,'accuracy':stats['accuracy'],'target_gate_passed':stats['target_gate_passed'],
           'new_judge_calls':calls,'replay_hits':hits},flush=True)
    return result


def run(root,enable):
    if not enable:
        raise ValueError('provider execution flag required')
    preflight=load_preflight(root)
    if recorded(root,preflight):
        raise ValueError('a started experiment cannot receive another release')
    require_idle()
    with (root/'execution.reserved').open('x',encoding='utf-8') as handle:
        handle.write(preflight.sha256+'\n')
    publish_sealed_json(root/'release.json',{'preflight_sha256':preflight.sha256,
        'maximum_answer_calls':400,'timed_concurrency':1,'automatic_retries':0})
    started=time.perf_counter()
    linker=new_linker()
    qwen_setup=time.perf_counter()-started
    if qwen_linker_identity(linker,strict=True) != preflight.payload['qwen_identity']:
        raise ValueError('resident Qwen identity changed')
    client=_completion_client('LITELLM_KEY',GATEWAY)
    try:
        for namespace in preflight.payload['namespaces']:
            started=time.perf_counter()
            memory=resident(load_as_of(Path(namespace['root'])).payload,namespace['parent'],linker)
            setup=time.perf_counter()-started
            try:
                for call in preflight.payload['calls'][4*namespace['offset']:4*(namespace['offset']+10)]:
                    prefix=root/'journal'/f"{call['call_index']:03}"
                    prefix.parent.mkdir(parents=True,exist_ok=True)
                    with prefix.with_suffix('.reserved').open('x',encoding='utf-8') as handle:
                        handle.write(preflight.sha256+'\n')
                    request,_=publish_sealed_json(prefix.with_suffix('.request.json'),{'preflight_sha256':preflight.sha256,'call':call})
                    prepared={}
                    def prompt():
                        if call['arm'] in MEMORY_ARMS:
                            m,h,a=build(memory,call['question'],call['arm'])
                        else:
                            m,h,a=[dict(row) for row in call['messages']],None,None
                        if m != call['messages']:
                            raise ValueError('live retrieval does not reproduce its exact API control')
                        prepared.update(messages=m,hydration=h,hierarchy_audit=a)
                        return m
                    try:
                        measurement=measure_streaming_answer(client=client,model=MODEL,prepare_prompt=prompt,max_tokens=256)
                    except Exception as error:
                        publish_sealed_json(prefix.with_suffix('.failure.json'),{'request_sha256':request.sha256,
                            'exception_type':type(error).__name__,'retry_performed':False})
                        raise
                    publish_sealed_json(prefix.with_suffix('.response.json'),{'request_sha256':request.sha256,
                        'measurement':measurement,**prepared,'evidence_sha256':call['evidence_sha256'],
                        'resident_setup_s_excluded_from_warm_latency':setup,
                        'shared_qwen_setup_s_excluded_from_warm_latency':qwen_setup})
                    if (call['call_index']+1)%4==0:
                        print({'completed_answer_calls':call['call_index']+1,'ordinal':call['question']['ordinal']},flush=True)
            finally:
                memory.encoder.close()
    finally:
        client.close()
    answers,_=seal_answers(root,preflight)
    print({'answers_sha256':answers.sha256,'fresh_streamed_responses':400},flush=True)
    result=judge(root,True)
    publish_sealed_json(root/'complete.json',{'joint_report_sha256':result.sha256,'target_gate_passed':result.payload['target_gate_passed']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','run','replay'))
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--parent-population',type=Path)
    parser.add_argument('--enable-provider',action='store_true')
    args=parser.parse_args()
    if args.phase=='prepare':
        if args.source_root is None or args.parent_population is None:
            parser.error('prepare requires --source-root and --parent-population')
        prepare(args.output_root,args.source_root,args.parent_population)
    elif args.phase=='run':
        run(args.output_root,args.enable_provider)
    else:
        judge(args.output_root,False)
