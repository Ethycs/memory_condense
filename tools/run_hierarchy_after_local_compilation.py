"""Wait for the bound local compiler, then run the complete joint full100 eval."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from memory_condense.domain._discourse_identity import quote_sha256
from tools import evaluate_hierarchical_spine_full100 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json,read_sealed_json
from tools.run_hot_reduced30_answer_judge import _completion_client
from tools.run_spine_reader_after_timeout import require_idle


MODELS = ('codex_sdk/gpt-5.6-terra','codex_sdk/gpt-5.6-sol')
READINESS_MESSAGES = [{'role':'user','content':'Readiness check. Reply only with OK.'}]


def implementation():
    return {**evaluation.implementation(),__file__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def dependency_alive(row,process_factory=psutil.Process):
    try:
        process = process_factory(row['pid'])
        return process.is_running() and process.create_time()==row['create_time']
    except psutil.NoSuchProcess:
        return False


def wait_for_terminal(row,maximum_seconds,*,sleep=time.sleep,now=time.monotonic,alive=dependency_alive):
    started,last_report = now(),float('-inf')
    while alive(row):
        elapsed = now()-started
        if elapsed>=maximum_seconds:
            raise TimeoutError('compiler is still live at the handoff wait deadline; it was not restarted')
        if elapsed-last_report>=60:
            print({'waiting_for_compiler_pid':row['pid'],'elapsed_wait_s':round(elapsed)},flush=True)
            last_report = elapsed
        sleep(min(10,maximum_seconds-elapsed))


def prepare(root,compiler_root,evaluation_root,source_root,pid,create_time):
    workspace = Path.cwd().resolve()
    for path in (root,compiler_root,evaluation_root,source_root):
        path.resolve().relative_to(workspace)
    if len({p.resolve() for p in (root,compiler_root,evaluation_root,source_root)})!=4:
        raise ValueError('handoff requires separate input and output roots')
    if evaluation_root.exists() and any(evaluation_root.iterdir()):
        raise ValueError('hierarchy evaluation root must remain unstarted')
    process = psutil.Process(pid)
    command = process.cmdline()
    if (not process.is_running() or process.create_time()!=create_time
        or Path(process.cwd()).resolve()!=workspace
        or Path(process.exe()).resolve()!=Path(sys.executable).resolve()
        or 'tools.run_local_spine_parent_batch4' not in command
        or '--output-root' not in command
        or Path(command[command.index('--output-root')+1]).resolve()!=compiler_root.resolve()):
        raise ValueError('handoff requires the live bound four-summary compiler process')
    batch = read_sealed_json(compiler_root/'batch-result.json')
    batch_preflight = read_sealed_json(compiler_root/'batch-preflight.json')
    inputs = read_sealed_json(compiler_root/'input-snapshots'/'population.json')
    first = read_sealed_json(compiler_root/'parents'/'offset-000'/'preflight.json')
    source = evaluation.previous.load_preflight(source_root)
    if (batch.payload['batch_release_passed'] is not True
        or batch.payload['preflight_sha256']!=batch_preflight.sha256
        or batch_preflight.payload['backend']['batch_adapter_sha256']!=hashlib.sha256(
            Path('tools/run_local_spine_parent_batch4.py').read_bytes()).hexdigest()
        or [r['offset'] for r in inputs.payload['records']]!=list(range(0,100,10))
        or first.payload['backend_sha256']!=batch.payload['backend_sha256']):
        raise ValueError('local compiler did not pass its bound batch release')
    preflight,_ = publish_sealed_json(root/'preflight.json',{
        'format':'memory-condense-hierarchy-after-local-v1','workspace':str(workspace),
        'dependency':{'pid':pid,'create_time':create_time},
        'compiler_root':str(compiler_root.resolve()),'batch_result_sha256':batch.sha256,
        'input_population_sha256':inputs.sha256,'parent_method_sha256':first.payload['method_policy_sha256'],
        'compiler_implementation':{**first.payload['implementation'],**{
            path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (
                'tools/run_local_spine_parent_batch4.py','tools/reuse_local_spine_parent_summaries.py')}},
        'evaluation_root':str(evaluation_root.resolve()),'source_root':str(source_root.resolve()),
        'source_preflight_sha256':source.sha256,'implementation':implementation(),
        'maximum_wait_seconds':72*60*60,'poll_seconds':10,
        'maximum_readiness_calls':2,'readiness_models':list(MODELS),
        'maximum_answer_calls':400,'maximum_logical_judgments':200,
        'all_answers_before_judging':True,'timed_concurrency':1,'automatic_retries':0,
        'raw_inputs_to_qwen':False,'target_gate_passed':False})
    print({'handoff_preflight_sha256':preflight.sha256,'waiting_for_pid':pid,
           'maximum_answer_calls':400,'new_provider_calls':0},flush=True)
    return preflight


def validate_preflight(root):
    preflight = read_sealed_json(root/'preflight.json')
    p = preflight.payload
    if p['implementation']!=implementation() or p['workspace']!=str(Path.cwd().resolve()):
        raise ValueError('handoff evaluation implementation changed')
    for path,sha in p['compiler_implementation'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
            raise ValueError('bound local compiler implementation changed')
    compiler_root = Path(p['compiler_root'])
    if (read_sealed_json(compiler_root/'batch-result.json').sha256!=p['batch_result_sha256']
        or read_sealed_json(compiler_root/'input-snapshots'/'population.json').sha256!=p['input_population_sha256']
        or evaluation.previous.load_preflight(Path(p['source_root'])).sha256!=p['source_preflight_sha256']):
        raise ValueError('handoff input binding changed')
    return preflight


def completed_population(preflight):
    p = preflight.payload
    if dependency_alive(p['dependency']):
        raise ValueError('local compiler is still running')
    compiler_root = Path(p['compiler_root'])
    candidates = []
    for path in (compiler_root/'parents'/'populations').glob('*.json'):
        artifact = read_sealed_json(path)
        a = artifact.payload
        if a['complete_namespace_population'] is not True:
            continue
        if (a['source_population_sha256']!=p['input_population_sha256']
            or a['raw_inputs_to_qwen'] is not False or a['remote_provider_calls']!=0
            or [r['offset'] for r in a['records']]!=list(range(0,100,10))):
            raise ValueError('completed parent population binding changed')
        for row in a['records']:
            expected = compiler_root/'parents'/f"offset-{row['offset']:03}"
            parent = read_sealed_json(expected/'preflight.json')
            if (Path(row['output_root']).resolve()!=expected.resolve()
                or row['preflight_sha256']!=parent.sha256
                or parent.payload['method_policy_sha256']!=p['parent_method_sha256']
                or row['complete_namespace'] is not True):
                raise ValueError('completed population changed the bound parent method')
        # The existing full100 evaluator authenticates each completed hierarchy
        # against the original million-token memory and refuses incomplete trees.
        evaluation.admit_parents(path,evaluation.previous.load_preflight(Path(p['source_root'])))
        candidates.append(artifact)
    if len(candidates)!=1:
        raise ValueError('handoff requires exactly one complete bound ten-memory population')
    return candidates[0]


def readiness(root,preflight,client_factory=None):
    if client_factory is None:
        client_factory = lambda:_completion_client('LITELLM_KEY',evaluation.GATEWAY).with_options(timeout=30,max_retries=0)
    client = client_factory()
    rows = []
    try:
        for ordinal,model in enumerate(MODELS):
            prefix = root/'readiness'/f'{ordinal:02}'
            prefix.parent.mkdir(parents=True,exist_ok=True)
            with prefix.with_suffix('.reserved').open('x',encoding='utf-8') as handle:
                handle.write(preflight.sha256+'\n')
            request,_ = publish_sealed_json(prefix.with_suffix('.request.json'),{
                'preflight_sha256':preflight.sha256,'model':model,'messages':READINESS_MESSAGES,
                'max_tokens':64,'timeout_seconds':30,'automatic_retries':0,
                'benchmark_questions_used':False,'raw_corpus_used':False})
            response = client.chat.completions.create(model=model,messages=READINESS_MESSAGES,max_tokens=64)
            if len(response.choices)!=1:
                raise ValueError('readiness returned an unexpected number of choices')
            choice = response.choices[0]
            text = choice.message.content
            artifact,_ = publish_sealed_json(prefix.with_suffix('.response.json'),{
                'request_sha256':request.sha256,'model':model,'finish_reason':choice.finish_reason,
                'response':text,'response_sha256':quote_sha256(text or ''),
                'observed_utc':datetime.now(timezone.utc).isoformat()})
            if choice.finish_reason!='stop' or not text or text.strip().casefold()!='ok':
                raise ValueError('reader or judge inference readiness failed')
            rows.append({'model':model,'response_sha256':artifact.sha256})
    finally:
        client.close()
    artifact,_ = publish_sealed_json(root/'readiness'/'complete.json',{
        'preflight_sha256':preflight.sha256,'rows':rows,'new_provider_calls':2})
    return artifact


def run(root,enable=False):
    if not enable:
        raise ValueError('the authorized evaluation handoff requires the provider flag')
    preflight = validate_preflight(root)
    p = preflight.payload
    with (root/'execution.reserved').open('x',encoding='utf-8') as handle:
        handle.write(preflight.sha256+'\n')
    publish_sealed_json(root/'release.json',{'preflight_sha256':preflight.sha256,
        'executor_pid':os.getpid(),'executor_create_time':psutil.Process().create_time(),
        'maximum_readiness_calls':2,'maximum_answer_calls':400,'maximum_logical_judgments':200,
        'automatic_retries':0})
    phase = 'wait for bound compiler'
    try:
        wait_for_terminal(p['dependency'],p['maximum_wait_seconds'])
        validate_preflight(root)
        population = completed_population(preflight)
        publish_sealed_json(root/'dependency-complete.json',{
            'preflight_sha256':preflight.sha256,'parent_population_sha256':population.sha256,
            'parent_population_path':str(population.path.resolve()),'dependency_observed_terminal':True})
        phase = 'wait for idle evaluation workspace'
        started = time.monotonic()
        while True:
            try:
                require_idle()
                break
            except ValueError:
                if time.monotonic()-started>=3600:
                    raise TimeoutError('evaluation workspace remained busy; no timed calls were started')
                time.sleep(10)
        phase = 'prepare all100 prompts and exact evidence'
        command = [sys.executable,'-X','utf8','-m','tools.evaluate_hierarchical_spine_full100']
        subprocess.run([*command,'prepare','--output-root',p['evaluation_root'],
            '--source-root',p['source_root'],'--parent-population',str(population.path)],check=True)
        phase = 'fresh reader and judge readiness'
        readiness(root,preflight)
        phase = '400 fresh streams then 200 logical judgments'
        subprocess.run([*command,'run','--output-root',p['evaluation_root'],'--enable-provider'],check=True)
        phase = 'zero-provider judge replay'
        subprocess.run([*command,'replay','--output-root',p['evaluation_root']],check=True)
        complete = read_sealed_json(Path(p['evaluation_root'])/'complete.json')
        report = read_sealed_json(Path(p['evaluation_root'])/'joint-report.json')
        if complete.payload['joint_report_sha256']!=report.sha256:
            raise ValueError('evaluation completion report binding changed')
        result,_ = publish_sealed_json(root/'complete.json',{
            'preflight_sha256':preflight.sha256,'evaluation_complete_sha256':complete.sha256,
            'joint_report_sha256':report.sha256,'judge_replay_completed':True,
            'target_gate_passed':report.payload['target_gate_passed']})
        print({'handoff_complete_sha256':result.sha256,'target_gate_passed':result.payload['target_gate_passed']},flush=True)
    except Exception as error:
        publish_sealed_json(root/'failure.json',{'preflight_sha256':preflight.sha256,'phase':phase,
            'exception_type':type(error).__name__,'message':str(error),'automatic_retry_performed':False})
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','run'))
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--compiler-root',type=Path)
    parser.add_argument('--evaluation-root',type=Path)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--dependency-pid',type=int)
    parser.add_argument('--dependency-create-time',type=float)
    parser.add_argument('--enable-provider',action='store_true')
    args = parser.parse_args()
    if args.phase=='prepare':
        if any(v is None for v in (args.compiler_root,args.evaluation_root,args.source_root,args.dependency_pid,args.dependency_create_time)):
            parser.error('preparation requires compiler, evaluation, source roots and the exact live process identity')
        prepare(args.output_root,args.compiler_root,args.evaluation_root,args.source_root,args.dependency_pid,args.dependency_create_time)
    else:
        run(args.output_root,args.enable_provider)
