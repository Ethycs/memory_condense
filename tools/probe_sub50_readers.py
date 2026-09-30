"""Nine bounded calls on identical saved packets; no history reconstruction."""
from contextlib import closing
from pathlib import Path
import json
import statistics
import argparse

from tools.engineering_research_gateway import read, save, emit, generate
from tools.run_hot_reduced30_answer_judge import _completion_client
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy


def main(root, models):
    root.mkdir(exist_ok=False)
    source=Path('eval_results/chat-io-batch12-20260929-r4')
    base=read(source/'run-plan.json')
    scopes=['batch12-a07','batch12-a10','batch12-a12']
    jobs={j['scope']:j for p in (source/'gateway').glob('*.request.json')
          if (j:=read(p))['kind']=='actor' and j['scope'] in scopes}
    cases=read(source/'cases.json')['cases']
    expected={f'batch12-a{i:02d}':c['expected'] for i,c in enumerate(cases,1)}
    save(root/'plan.json',dict(models=models,scopes=scopes,maximum_calls=3*len(models),gateway=base['gateway'],
        reasoning_effort='none for codex_sdk variants; omitted for Claude',same_saved_packets=True))
    results=[]
    with closing(_completion_client('LITELLM_KEY',base['gateway']).with_options(timeout=60,max_retries=0)) as client:
        for i,scope in enumerate(scopes):
            for model in models[i:]+models[:i]:
                job=jobs[scope]
                name=str(len(results))
                (root/(name+'.reserved')).touch(exist_ok=False)
                plan=dict(models={'actor':model},reasoning_effort={'actor':'none'} if model.startswith('codex_sdk/') else {})
                result=generate(client,plan,job,identity_sha256(job),count_chat_prompt_token_proxy(job['messages']))
                try: correct=json.loads(result.get('content',''))==expected[scope] and result.get('finish_reason')=='stop'
                except ValueError: correct=False
                result.update(model=model,scope=scope,correct=correct)
                save(root/(name+'.json'),result)
                results.append(result)
                emit(**{k:result.get(k) for k in ('model','scope','correct','elapsed_s','error_type')})
    stats={model:dict(correct=sum(r['correct'] for r in results if r['model']==model),total=3,
        mean_s=statistics.mean(r.get('elapsed_s',60) for r in results if r['model']==model)) for model in models}
    save(root/'report.json',dict(models=stats,calls=len(results),history_ingestions=0,
                               broad_engineering_quality_equivalence=False))
    emit(phase='complete',models=stats)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('eval_results/sub50-reader-probe-20260930-r1'))
    parser.add_argument('--models',nargs='+',default=['codex_sdk/gpt-5.6-sol','codex_sdk/gpt-5.6-luna','claude-haiku-4-5'])
    args=parser.parse_args()
    main(args.root,args.models)
