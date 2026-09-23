"""Matched real-transcript answers: summary BM25, narrow Qwen and wider exploration.

The locked benchmark question and the seven source-derived development probes
are reported separately. Compilation and answer construction do not load the
reference file or benchmark answers. Gold is joined only after answers seal.
"""

import argparse
import json
from pathlib import Path
import time

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval._retrieval_qa_prompt import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE, QA_NO_CONTEXT
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.spine_routing import route_user_spine_hierarchy
from memory_condense.search.summary_reasoning import reason_over_summary_hierarchy
from tools.assay_user_spine_hierarchy import Journal, PROBES, _turns
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized


QUERIES = Path('tests/fixtures/user_spine_real_queries_v1.json')
REFERENCES = Path('tests/fixtures/user_spine_real_references_v1.json')
MODELS = {'answers':'codex_sdk/gpt-5.6-terra','judge':'codex_sdk/gpt-5.6-sol'}


def prepare(root, source_root):
    source=read_sealed_json(source_root/'preflight.json')
    hierarchy=read_sealed_json(source_root/'hierarchy-r4.json')
    query_fixture=json.loads(QUERIES.read_text(encoding='utf-8'))
    assert query_fixture['source_turn_population_sha256']==source.payload['binding']['turn_population_sha256']
    probes=read_sealed_json(PROBES)
    original=next(q for q in probes.payload['questions'] if q['ordinal']==source.payload['global_ordinal'])
    questions=[{'id':'locked_q86','group':'locked_benchmark_development','question':original['retrieval_query'],
                'dated_question':original['prompt_question'],'question_id':original['question_id'],'global_ordinal':86}]
    questions.extend({'id':q['id'],'group':'source_derived_development','question':q['question'],
                      'dated_question':'[Question asked at 2023/06/01 (Thu) 03:56]\n'+q['question']} for q in query_fixture['cases'])
    result,_=publish_sealed_json(root/'preflight.json',{
        'source_root':str(source_root.resolve()),'source_preflight_sha256':source.sha256,'hierarchy_sha256':hierarchy.sha256,
        'implementation':{name:file_sha256(Path(name)) for name in (str(Path(__file__)),
            'src/memory_condense/search/spine_routing.py','src/memory_condense/search/summary_reasoning.py')},
        'query_fixture_sha256':file_sha256(QUERIES),'questions':questions,'qwen_model':'qwen3-8b',
        'gateway_url':source.payload['gateway_url'],'max_qwen_completion_calls':160,
        'arms':['summary_bm25','qwen_narrow','qwen_beam8'],'max_sections':3,'max_context_tokens':4096,'max_raw_spans':128,
        'beam_sections':8,'group_size':12,'max_route_prompt_tokens':6000,'max_answer_prompt_tokens':5500,
        'answer_max_tokens':256,'judge_max_tokens':32,'gold_loaded':False,'single_query_family_is_not_full100':True})
    print({'preflight_sha256':result.sha256,'questions':len(questions),'arms':3},flush=True)


def construct(root, enable):
    preflight=read_sealed_json(root/'preflight.json')
    p=preflight.payload
    if any(file_sha256(Path(name))!=sha for name,sha in p['implementation'].items()):raise ValueError('evaluation implementation changed')
    source_root=Path(p['source_root'])
    source=read_sealed_json(source_root/'preflight.json')
    hierarchy=read_sealed_json(source_root/'hierarchy-r4.json')
    assert source.sha256==p['source_preflight_sha256'] and hierarchy.sha256==p['hierarchy_sha256']
    index=SectionSummaryIndex.from_json(hierarchy.payload['index_json'])
    journal=Journal(root,preflight,enable)
    routes=[]
    for question in p['questions']:
        for arm in p['arms']:
            before=time.perf_counter()
            try:
                audit=None
                if arm=='summary_bm25':plan=index.route(question['question'],max_sections=3)
                elif arm=='qwen_narrow':plan=reason_over_summary_hierarchy(question['question'],index,reasoner=journal,
                    max_sections=3,group_size=8,max_calls=32,max_prompt_tokens=6000)
                else:
                    audit=route_user_spine_hierarchy(question['question'],index,reasoner=journal,max_sections=3,
                        beam_sections=8,group_size=12,max_calls=64,max_prompt_tokens=6000)
                    plan=audit.plan
                error=None
            except ValueError as exc:
                plan,audit,error=None,None,str(exc)
            routes.append((question,arm,plan,audit,error,time.perf_counter()-before))
    # Raw storage opens after all summary-routing calls are finished.
    turns,_=_turns(source.payload['binding'])
    records={t.turn_id:t for t in turns}
    rows=[]
    for question,arm,plan,audit,error,elapsed in routes:
        result=hydrate_section_plan(plan,load_turn=records.get,max_raw_spans=128,max_context_tokens=4096) if plan else None
        context=result.render_context() if result else ''
        messages=[{'role':'system','content':QA_SYSTEM_PROMPT},{'role':'user','content':QA_USER_TEMPLATE.format(
            context=context or QA_NO_CONTEXT,question=question['dated_question'])}]
        tokens=count_chat_prompt_token_proxy(messages)
        if tokens>5500:raise ValueError('answer prompt exceeded matched budget')
        rows.append({'case_id':question['id'],'group':question['group'],'arm':arm,'question':question['question'],
            'messages':messages,'prompt_tokens':tokens,'route_error':error,'hydration':result.identity_payload() if result else None,
            'beam_audit':audit.identity_payload() if audit else None})
    result,_=publish_sealed_json(root/'selection.json',{'preflight_sha256':preflight.sha256,'rows':rows,
        'qwen_request_artifact_shas':journal.requests,'gold_loaded':False,'raw_qwen_inputs':0})
    print({'selection_sha256':result.sha256,'rows':len(rows),'new_provider_calls':journal.calls,
           'checkpoint_hits':journal.hits,'route_errors':sum(r['route_error'] is not None for r in rows)},flush=True)


def _batch(root, phase, prompts, binding_sha, model, cap, output_cap, gateway, enable):
    def factory(client):
        return FastCompletionRuntime(checkpoint_dir=root/(phase+'-checkpoints'),prompt_population=prompts,model=model,
            client=client,max_prompt_tokens=cap,max_new_tokens=output_cap,max_concurrency=8,retries=0,
            request_options={'temperature':0},benchmark_provenance={'binding_sha256':binding_sha,'phase':phase})
    audit=factory(None)
    try:remaining=audit.population.unique_prompt_count-len(_authenticated_records(audit))
    finally:audit.close()
    batch,calls,hits,elapsed=_run_exactly_authorized(runtime_factory=factory,authorized_provider_calls=remaining,
        enable_provider=enable,client_factory=lambda:_completion_client('LITELLM_KEY',gateway))
    return batch,calls,hits,elapsed


def answers(root, enable):
    selection=read_sealed_json(root/'selection.json')
    preflight=read_sealed_json(root/'preflight.json')
    assert selection.payload['preflight_sha256']==preflight.sha256
    rows=selection.payload['rows']
    batch,calls,hits,elapsed=_batch(root,'answers',[r['messages'] for r in rows],selection.sha256,MODELS['answers'],5500,256,
                                  preflight.payload['gateway_url'],enable)
    predictions=[{'case_id':r['case_id'],'group':r['group'],'arm':r['arm'],'question':r['question'],
                  'prediction':text,'prediction_sha256':quote_sha256(text)} for r,text in zip(rows,batch.logical_completions,strict=True)]
    result,_=publish_sealed_json(root/'answers.json',{'selection_sha256':selection.sha256,'preflight_sha256':preflight.sha256,
        'rows':predictions,'gold_loaded':False,'response_journal_shas':[r.response_journal_sha256 for r in batch.unique_records]})
    print({'answers_sha256':result.sha256,'new_provider_calls':calls,'checkpoint_hits':hits,'elapsed_seconds':elapsed},flush=True)


def judge_preflight(root):
    # First access to benchmark references or the development-reference file.
    answers=read_sealed_json(root/'answers.json')
    preflight=read_sealed_json(root/'preflight.json')
    assert answers.payload['preflight_sha256']==preflight.sha256
    from tools.run_hot_reduced30_answer_judge import _load_locked_validation_question_population
    from memory_condense.eval.benchmark import build_judge_prompt
    population_sha,questions=_load_locked_validation_question_population(
        Path('C:/Users/Keytone/Downloads/memory-condense-rig/datasets/longmemeval_s_cleaned.json'),
        Path('docs/10 - Research Log/data/longmemeval-95-target-split-v2.json'))
    original=questions[86]
    assert original.question_id==preflight.payload['questions'][0]['question_id']
    refs=json.loads(REFERENCES.read_text(encoding='utf-8'))
    source=read_sealed_json(Path(preflight.payload['source_root'])/'preflight.json')
    turns,digest=_turns(source.payload['binding'])
    assert digest==refs['source_turn_population_sha256']
    references={'locked_q86':original.answer}
    for ref in refs['cases']:
        assert turns[ref['turn_ordinal']].role=='user' and ref['quote'] in turns[ref['turn_ordinal']].text
        references[ref['id']]=ref['answer']
    rows=[{**r,'messages':build_judge_prompt(r['question'],references[r['case_id']],r['prediction']),
           'reference_sha256':quote_sha256(references[r['case_id']])} for r in answers.payload['rows']]
    result,_=publish_sealed_json(root/'judge-preflight.json',{'answers_sha256':answers.sha256,'rows':rows,
        'population_sha256':population_sha,'reference_fixture_sha256':file_sha256(REFERENCES),'gold_loaded':True})
    print({'judge_preflight_sha256':result.sha256,'judgments':len(rows)},flush=True)


def judge(root, enable):
    from memory_condense.eval._binary_judge_protocol import parse_binary_judge_verdict
    preflight=read_sealed_json(root/'preflight.json')
    inputs=read_sealed_json(root/'judge-preflight.json')
    assert inputs.payload['answers_sha256']==read_sealed_json(root/'answers.json').sha256
    rows=inputs.payload['rows']
    batch,calls,hits,elapsed=_batch(root,'judge',[r['messages'] for r in rows],inputs.sha256,MODELS['judge'],4096,32,
                                  preflight.payload['gateway_url'],enable)
    judged=[{k:r[k] for k in ('case_id','group','arm','prediction','prediction_sha256','reference_sha256')} |
            {'verdict':verdict,'correct':parse_binary_judge_verdict(verdict)} for r,verdict in zip(rows,batch.logical_completions,strict=True)]
    aggregates={group:{arm:{'correct':sum(r['correct'] for r in judged if r['group']==group and r['arm']==arm),
        'count':sum(r['group']==group and r['arm']==arm for r in judged)} for arm in preflight.payload['arms']}
        for group in sorted({r['group'] for r in judged})}
    result,_=publish_sealed_json(root/'judgments.json',{'judge_preflight_sha256':inputs.sha256,'rows':judged,'aggregates':aggregates,
        'response_journal_shas':[r.response_journal_sha256 for r in batch.unique_records],'promotion':False})
    print(json.dumps({'judgments_sha256':result.sha256,'new_provider_calls':calls,'checkpoint_hits':hits,
        'elapsed_seconds':elapsed,'aggregates':aggregates},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','construct','answers','judge-preflight','judge'))
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--enable-provider',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.output_root,args.source_root)
    elif args.command=='construct':construct(args.output_root,args.enable_provider)
    elif args.command=='answers':answers(args.output_root,args.enable_provider)
    elif args.command=='judge-preflight':judge_preflight(args.output_root)
    else:judge(args.output_root,args.enable_provider)
