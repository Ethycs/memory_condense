"""Bounded resident CPU comparison on saved raw jobs and new engineering IO."""
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
from copy import deepcopy
import json
import socket
import subprocess
import time

import httpx
import psutil
from openai import OpenAI

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search import native_spine_summary as raw
from tools.engineering_research_memory import RAW_SYSTEM
from tools.engineering_research_gateway import read, save, emit
from tools.native_spine_engineering_session import repair_raw_support


ROOT = Path('eval_results/small-summarizer-20260929-r1')
MODELS = {'transcript': 'LFM2-2.6B-Transcript', 'small': 'LFM2.5-1.2B-Instruct'}
SCHEMA = dict(type='object', properties=dict(atoms=dict(type='array', items=dict(type='object',
    properties=dict(label=dict(type='string'), summary=dict(type='string'),
                    support=dict(type='array', items=dict(type='string'))),
    required=['label','summary','support'], additionalProperties=False))),
    required=['atoms'], additionalProperties=False)


def prepare():
    jobs = []
    for path in sorted(Path('eval_results/chat-io-batch12-20260929-r3/gateway').glob('*.request.json')):
        job = read(path)
        if job['kind']!='raw': continue
        payload = json.loads(job['messages'][1]['content'])['fragments']
        digest = identity_sha256(payload)
        fragments = [raw.BodyFragment(digest, i, f['speaker'], 0, len(f['fragment']),
                     quote_sha256(f['fragment']), f['fragment']) for i,f in enumerate(payload)]
        jobs.append(dict(name=path.stem, messages=job['messages'], fragments=[asdict(f) for f in fragments],
                         source=str(path), synthetic=False))
    filler = ('The review separates operator intent from observed execution. Retain the exact labels when writing the handoff. '
              'The rollout worksheet has columns for environment, ownership, evidence, and recovery. '
              'Do not treat a recommendation as confirmation from an operator. ')*5
    texts = [
        ('user', 'Project Cedar: plan migration cedar-812 to staging cluster amber-563 using rollback tag spruce-447. '
         'Do not deploy. Correction: use release r29 instead of r28. Keep region eu-west-3. '+filler,
         ['cedar-812','amber-563','spruce-447','r29','r28','eu-west-3']),
        ('assistant', 'I propose changing retry_delay_ms to 750 in src/queue.py and adding test_cancel_after_flush. '
         'This is a proposal; I have not modified files or run tests. The user has not approved this proposal. '+filler,
         ['retry_delay_ms','750','src/queue.py','test_cancel_after_flush']),
        ('system', 'Tool observation: pytest tests/test_queue.py completed with 17 passed and 2 failed. '
         'The failure is test_cancel_after_flush with expected 0 pending jobs, actual 1. '
         'The command exited with code 1. No deployment command ran. '+filler,
         ['17','2','test_cancel_after_flush','0','1'])]
    fragments = raw.fragment_body(dict(turns=[dict(role=r,text=t) for r,t,_ in texts]))
    assert len(fragments)==3 and all(count_tokens(f.text)>=192 for f in fragments)
    messages = raw.summary_messages(fragments)
    messages[0]['content'] = RAW_SYSTEM
    jobs.append(dict(name='new-engineering', messages=messages, fragments=[asdict(f) for f in fragments],
                     synthetic=True, required_identifiers=[g for _,_,g in texts]))
    save(ROOT/'plan.json', dict(jobs=jobs, models=MODELS, quantization='Q4_K_M', device='cpu',
        threads=6, server_slots=3, concurrency=3, max_output_tokens=1024, retries=0,
        structured_output=True, history_ingestions=0, maximum_requests_per_model=len(jobs)))
    emit(phase='prepared', jobs=len(jobs), saved_jobs=len(jobs)-1)


def run(kind, *, bounded=False):
    plan = read(ROOT/'plan.json')
    name = MODELS[kind]
    output = ROOT/(kind+'-bounded' if bounded else kind)
    output.mkdir(exist_ok=False)
    port = 18592
    with socket.socket() as sock: sock.bind(('127.0.0.1',port))
    exe = Path('.cache/runtimes/llama-b11272-bin-win-cpu-x64/llama-server.exe').resolve()
    model = Path('.cache/models')/(name+'-GGUF')/(name+'-Q4_K_M.gguf')
    workers = 1 if bounded else 3
    args = [str(exe), '-m', str(model.resolve()), '--host','127.0.0.1','--port',str(port),
            '-ngl','0','-t','6','-tb','6','-c',str(8192*workers),'-np',str(workers),'--cache-ram','0','--metrics']
    jobs = plan['jobs']
    if bounded:
        novel = jobs[-1]
        longest = max((f for j in jobs[:-1] for f in j['fragments']),key=lambda f:count_tokens(f['text']))
        jobs = []
        for i,f in enumerate([*novel['fragments'],longest]):
            fragment = raw.BodyFragment(**f)
            messages = raw.summary_messages((fragment,))
            messages[0]['content'] = RAW_SYSTEM
            if kind=='transcript':
                messages = [dict(role='system',content='You are an expert meeting analyst. Analyze the transcript carefully and provide clear, accurate information based on the content.'),
                    dict(role='user',content=RAW_SYSTEM+' Return exactly one atom, with label T0.\n\n'
                        'Title: Engineering handoff\nDate: Not provided\nTime: Not provided\n'
                        'Duration: Not provided\nParticipants: '+fragment.role+'\n----------\n'
                        '**'+fragment.role+'**: '+fragment.text)]
            job = dict(name=f'bounded-{i}',messages=messages,fragments=[asdict(fragment)],synthetic=i<3)
            if i<3: job['required_identifiers']=[novel['required_identifiers'][i]]
            jobs.append(job)
    save(output/'plan.json',dict(jobs=jobs,concurrency=workers,max_output_tokens=512 if bounded else 1024,
        native_transcript_format=bounded and kind=='transcript',temperature=.3 if bounded and kind=='transcript' else 0))
    save(output/'runtime.json', dict(args=args, model_sha256=__import__('hashlib').file_digest(model.open('rb'),'sha256').hexdigest()))
    started = time.perf_counter()
    with (output/'server.log').open('w',encoding='utf-8') as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            with httpx.Client(timeout=2,trust_env=False) as health:
                for _ in range(120):
                    if process.poll() is not None: raise RuntimeError('Local server exited; inspect server.log')
                    try:
                        if health.get(f'http://127.0.0.1:{port}/health').status_code==200: break
                    except httpx.HTTPError: pass
                    time.sleep(.5)
                else: raise TimeoutError('Local server did not become ready')
            load_s = time.perf_counter()-started
            emit(phase='model_ready', model=name, cold_load_s=load_s)
            with OpenAI(base_url=f'http://127.0.0.1:{port}/v1', api_key='local-benchmark', timeout=240,
                        max_retries=0, http_client=httpx.Client(trust_env=False,timeout=240)) as client:
                def invoke(job):
                    (output/(job['name']+'.reserved')).touch(exist_ok=False)
                    started = time.perf_counter()
                    content, first, finish, usage, timings = '', None, None, None, None
                    result = dict(name=job['name'], synthetic=job['synthetic'])
                    try:
                        schema=deepcopy(SCHEMA)
                        schema['properties']['atoms'].update(minItems=len(job['fragments']),maxItems=len(job['fragments']))
                        schema['properties']['atoms']['items']['properties']['label']['enum']=[f'T{i}' for i in range(len(job['fragments']))]
                        with client.chat.completions.create(model=name, messages=job['messages'], temperature=.3 if bounded and kind=='transcript' else 0,
                            max_tokens=512 if bounded else plan['max_output_tokens'], stream=True, stream_options={'include_usage':True},
                            response_format=dict(type='json_schema',json_schema=dict(name='routing_atoms',strict=True,schema=schema))) as stream:
                            for chunk in stream:
                                timings = (chunk.model_extra or {}).get('timings',timings)
                                if chunk.usage: usage=chunk.usage.model_dump()
                                for choice in chunk.choices:
                                    if choice.delta.content:
                                        first = first if first is not None else time.perf_counter()-started
                                        content += choice.delta.content
                                    finish = choice.finish_reason or finish
                        fragments = tuple(raw.BodyFragment(**f) for f in job['fragments'])
                        repaired, changes = repair_raw_support(content,fragments)
                        parsed = raw.parse_summaries(repaired,fragments)
                        result.update(valid=finish=='stop',parsed=parsed,support_escape_repairs=changes)
                        if job.get('required_identifiers'):
                            result['identifier_coverage'] = [dict(required=required,
                                missing=[s for s in required if s not in item['summary']])
                                for required,item in zip(job['required_identifiers'],parsed,strict=True)]
                    except Exception as exc:
                        result.update(valid=False,error_type=type(exc).__name__,error=str(exc)[:400])
                    result.update(elapsed_s=time.perf_counter()-started,ttft_s=first,content=content,
                        finish_reason=finish,usage=usage,timings=timings,output_tokens_proxy=count_tokens(content))
                    save(output/(job['name']+'.json'),result)
                    emit(phase='summary',model=name,**{k:result.get(k) for k in ('name','valid','elapsed_s','ttft_s','error')})
                    return result
                measured=time.perf_counter()
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results=list(pool.map(invoke,jobs))
                elapsed=time.perf_counter()-measured
            report=dict(model=name,device='cpu',quantization='Q4_K_M',concurrency=workers,
                cold_load_s=load_s,wall_s=elapsed,valid=sum(r['valid'] for r in results),total=len(results),
                mean_request_s=sum(r['elapsed_s'] for r in results)/len(results),
                memory=psutil.Process(process.pid).memory_info()._asdict(),
                semantic_accuracy_proven=False,history_ingestions=0)
            save(output/'report.json',report)
            emit(phase='complete',**report)
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','transcript','small'))
    parser.add_argument('--bounded',action='store_true')
    args=parser.parse_args()
    prepare() if args.phase=='prepare' else run(args.phase,bounded=args.bounded)
