"""Matched engineering continuation using live inline memory and the local stack."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

from tools import engineering_research_battery as battery
from tools import run_engineering_research_battery as legacy
from tools.engineering_research_gateway import emit, read, save

REPO=Path(__file__).resolve().parents[1]


def prepare(root,bundle):
    if root.exists():
        raise ValueError('Use a fresh engineering run directory')
    audit=battery.audit(bundle)
    # Battery manifests use the builder's checksum-with-filename sidecar;
    # audit above validates that format and the complete source bindings.
    manifest=json.loads((bundle/'battery.json').read_text(encoding='utf-8'))
    if len(manifest['cases'])!=15 or any(c['domain']!='engineering' for c in manifest['cases']):
        raise ValueError('Exactly fifteen engineering cases required')
    files=[*Path('src/memory_condense').rglob('*.py'),*Path('tools').glob('engineering_research_*.py'),
        Path(__file__),Path(legacy.__file__),Path('tools/check_inline_engineering15.py'),
        Path('tools/inline_battery_web.py'),Path('tools/build_inline_engineering15.py'),
        Path('tools/fastembed_bge.py'),Path('tools/probe_llama32_local.py'),
        Path('tools/compile_native_spine_attention.py'),Path('tools/build_spine_corpus_hierarchy.py')]
    save(root/'run-plan.json',dict(schema='inline-engineering15-v1',battery_sha256=audit['battery_sha256'],
        bundle=str(bundle.resolve()),cases=manifest['cases'],pilot=['E01','E11'],
        gateway='https://central-dev.zt:4000/v1',
        models=dict(actor='codex_sdk/gpt-5.6-sol',judge='codex_sdk/gpt-5.6-terra',
                    raw='local Llama-3.2-3B-Instruct-Q4_K_M',merge='local Llama exact extracts'),
        reasoning_effort=dict(actor='none',judge='none'),
        budgets=dict(actor=dict(calls=720,prompt_cap=131072,output_cap=4096,input_token_budget=45_000_000),
                     judge=dict(calls=30,prompt_cap=131072,output_cap=4096,input_token_budget=4_000_000),
                     raw=dict(calls=0,prompt_cap=1,output_cap=1,input_token_budget=0),
                     merge=dict(calls=0,prompt_cap=1,output_cap=1,input_token_budget=0)),
        inline_memory=True,chat_ingestion=True,streaming=True,compiler_concurrency=3,
        recent_exchanges=12,working_tokens=8192,actor_calls_per_arm_case=24,
        memory_prompt_cap=24576,full_context_prompt_cap=131072,
        web=dict(enabled=True,calls_per_arm=4,bridge_timeout_s=240,backend='bounded public Bing RSS/HTML'),
        local_stack='FP32 BGE FastEmbed CPU; six-layer lossless Qwen attention; local Llama source compiler',
        qwen_inputs='summaries only',same_actor_protocol_both_arms=True,
        historical_future_answers_excluded=True,private_checks_excluded_from_actor=True,
        early_stop='Any provider/lifecycle/harness error; five rejected summary pairs consecutively; two consecutive pairs with incomplete or mechanically failing memory output',
        automatic_retries=0,implementation={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    # Preserve exactly what was executed, including files already uncommitted.
    for file in files:
        relative=file.resolve().relative_to(REPO)
        destination=root/'implementation'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(file.read_bytes())
    emit(phase='engineering_prepared',cases=15,paired_actor_arms=30)


def verify(root):
    plan=read(root/'run-plan.json')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in plan['implementation'].items()):
        raise ValueError('Frozen engineering implementation changed')
    return plan


def added_checks(workspace,case_id):
    """Same restricted executor and Windows limits, with the new frozen checker."""
    from tools.engineering_research_execution import attach_job
    sandbox=REPO/'tools/engineering_research_sandbox.py'
    checker=REPO/'tools/check_inline_engineering15.py'
    environment={k:v for k,v in os.environ.items() if k.upper() in ('SYSTEMROOT','WINDIR','COMSPEC')}
    environment.update(PYTHONUTF8='1',TEMP=str(workspace),TMP=str(workspace),
                       OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    child=subprocess.Popen([sys.executable,'-I','-B',str(sandbox),str(workspace),'acceptance',
                            str(checker),case_id],cwd=workspace,env=environment,
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW)
    close_job=None
    started=time.perf_counter()
    try:
        close_job=attach_job(child)
        try:
            output,_=child.communicate('RUN\n',timeout=75)
            timed_out=False
        except subprocess.TimeoutExpired:
            child.kill()
            output,_=child.communicate()
            timed_out=True
        return dict(exit_code=child.returncode,output=output[:100000],timed_out=timed_out,
                    elapsed_s=time.perf_counter()-started,environment_credentials_removed=True,
                    checker_sha256=hashlib.sha256(checker.read_bytes()).hexdigest())
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        if close_job:
            close_job()


def arm(root,case_id,arm_name):
    from tools.engineering_research_chat import open_chat
    from tools.engineering_research_local import LocalRuntime
    started=time.perf_counter()
    plan=verify(root)
    record=next(c for c in plan['cases'] if c['id']==case_id)
    actor=battery.read_binding(record['actor'])
    folder=root/'cases'/case_id/arm_name
    runtime=LocalRuntime(folder/'local-runtime') if arm_name=='memory' else None
    try:
        with open_chat(root,folder,actor,arm_name,runtime=runtime,streaming=arm_name=='memory',
                       recent_exchanges=12,recent_token_budget=8192) as chat:
            result=legacy._run_arm(root,record,arm_name,chat=chat,
                                   inline_memory=True,recall_each_action=True)
            outputs=[e for e in chat.events() if e.metadata.get('inline_generation')]
            accepted=sum(e.metadata['inline_generation']['status']=='accepted' for e in outputs)
            save(folder/'inline-status.json',dict(accepted=accepted,fallback=len(outputs)-accepted,
                source_bound_outputs=len(outputs),runtime_metrics=dict(runtime.metrics) if runtime else {}))
            if len(outputs)>=5 and all(e.metadata['inline_generation']['status']=='fallback' for e in outputs[-5:]):
                raise RuntimeError('Five consecutive rejected inline summary pairs')
        if case_id in ('E11','E12','E13','E14','E15'):
            save(folder/'added-checks.json',added_checks(Path(result['workspace']).resolve(),case_id))
        if result.get('lifecycle_error'):
            raise RuntimeError(result['lifecycle_error'])
    except Exception as exc:
        save(folder/'operational-failure.json',dict(error_type=type(exc).__name__,reason=str(exc)))
        raise
    finally:
        if runtime is not None:
            runtime.close()
            save(folder/'shutdown.json',dict(local_server_stopped=runtime.process is None or runtime.process.poll() is not None))
        save(folder/'full-cycle.json',dict(wall_s=time.perf_counter()-started,
            includes='Initial ingest, model startup, actor/tool loop, final learning/ingest, checks and shutdown; excludes paired grading'))


def audit(root,case_id):
    from memory_condense.persistence.db import Database
    from memory_condense.persistence.transcript_store import TranscriptStore
    from memory_condense.persistence import native_spine_incremental_store
    from memory_condense.search.section_summary import RawSectionSpan
    from memory_condense.domain.inline_memory import summaries_for_rows
    folder=root/'cases'/case_id/'memory'
    result=read(folder/'result.json')
    with closing(sqlite3.connect(folder/'chat/chat-events.sqlite')) as journal, \
            Database(folder/'store/memory/memory.db',read_only=True) as db:
        turns={t.turn_id:t for t in TranscriptStore(db).get_all()}
        rows=[dict(turn_id=i,role=r,text=t,metadata=json.loads(m))
              for i,r,t,m in journal.execute('SELECT event_id,role,text,metadata FROM events')]
        assert len(rows)==len(turns)
        for row in rows:
            assert turns[row['turn_id']].text==row['text']
            assert turns[row['turn_id']].role==(row['role'] if row['role'] in ('user','assistant','system') else 'system')
        packets=list(journal.execute('SELECT packet_id,refs FROM packets'))
        pointers=0
        for packet_id,refs in packets:
            for ref in json.loads(refs):
                span=RawSectionSpan(**ref['span'])
                assert span==RawSectionSpan.from_turn(turns[span.turn_id],start_char=span.start_char,end_char=span.end_char)
                pointers+=1
        assert packets and pointers,'No original memory evidence delivered'
        pending=journal.execute('SELECT COUNT(*) FROM feedback WHERE successful=1 AND applied=0').fetchone()[0]
        learned=db.execute("SELECT COUNT(*) FROM consolidation_access_events WHERE event_id LIKE '_chat:feedback:%'").fetchone()[0]
        assert pending==0 and learned>0
        snapshot=native_spine_incremental_store.load(folder/'store/memory'/native_spine_incremental_store.FILENAME,
                                                     turns=tuple(turns.values()))
        assert snapshot.native.receipt==result['final_reopen']['snapshot']
        assert snapshot.parent_receipt==result['final_reopen']['parent_snapshot']
        accepted=[r for r in rows if r['metadata'].get('inline_generation',{}).get('status')=='accepted']
        summaries=summaries_for_rows(rows)
        assert all(r['turn_id'] in summaries for r in accepted)
    save(folder/'reopen-audit.json',dict(separate_process=True,events=len(rows),packets=len(packets),
        source_pointers=pointers,hebbian_updates=learned,accepted_inline_outputs=len(accepted),
        native_and_parent_receipts_reconstructed=True))


def web_worker(root,stop):
    from tools.inline_battery_web import execute
    while not stop.is_set():
        for path in sorted((root/'web').glob('*.request.json')):
            response=path.with_name(path.name.replace('.request.json','.response.json'))
            if response.with_suffix('.json.sha256').exists():
                continue
            request=read(path)
            try:
                result=execute(request['arguments'])
            except Exception as exc:
                result=dict(tool_error=type(exc).__name__+': '+str(exc))
            save(response,dict(request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),result=result))
            emit(phase='web',scope=request['scope'],error=result.get('tool_error'))
        stop.wait(.2)


def child(root,label,args):
    folder=root/'logs'
    folder.mkdir(exist_ok=True)
    with (folder/(label+'.stdout.log')).open('w',encoding='utf-8') as out, \
            (folder/(label+'.stderr.log')).open('w',encoding='utf-8') as err:
        process=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),*args,'--root',str(root)],
                               cwd=REPO,stdout=out,stderr=err)
    if process.returncode:
        raise RuntimeError(label+f' exited {process.returncode}; inspect saved stderr')


def run(root,campaign):
    from tools.engineering_research_gateway import Gateway,worker
    plan=verify(root)
    (root/'run.reserved').touch(exist_ok=False)
    stop=threading.Event()
    worker_errors=[]
    def provider():
        try:
            worker(root)
        except BaseException as exc:
            worker_errors.append(str(exc))
            (root/'STOP').touch()
    provider_thread=threading.Thread(target=provider,daemon=True)
    browser=threading.Thread(target=web_worker,args=(root,stop),daemon=True)
    provider_thread.start()
    browser.start()
    records=sorted(plan['cases'],key=lambda c:(c['id'] not in plan['pilot'],plan['pilot'].index(c['id']) if c['id'] in plan['pilot'] else c['id']))
    consecutive_bad=0
    completed=[]
    try:
        for record in records:
            if (campaign/'STOP').exists() or (root/'STOP').exists() or worker_errors:
                raise RuntimeError('Stop requested or provider worker failed: '+str(worker_errors))
            case_id=record['id']
            order=('memory','full_context') if int(case_id[1:])%2 else ('full_context','memory')
            results={}
            for arm_name in order:
                label=case_id+'-'+arm_name
                emit(phase='engineering_arm_start',case=case_id,arm=arm_name)
                child(root,label,['arm','--case',case_id,'--arm',arm_name])
                results[arm_name]=read(root/'cases'/case_id/arm_name/'result.json')
                if arm_name=='memory':
                    child(root,case_id+'-audit',['audit','--case',case_id])
            legacy.grade_pair(root,record,results)
            completed.append(case_id)
            added={arm:read(root/'cases'/case_id/arm/'added-checks.json')
                   for arm in order if (root/'cases'/case_id/arm/'added-checks.json').exists()}
            memory=results['memory']
            bad=(not memory['finished'] or not memory['structural']['structurally_complete']
                 or any(memory[k]['exit_code']!=0 for k in ('unit_tests','behavioral_checks') if k in memory)
                 or added.get('memory',{}).get('exit_code',0)!=0)
            consecutive_bad=consecutive_bad+1 if bad else 0
            save(root/f'progress-{len(completed):02}.json',dict(completed=completed,latest_case=case_id,
                added_behavioral_checks=added,consecutive_memory_execution_failures=consecutive_bad))
            emit(phase='engineering_pair_complete',case=case_id,completed_pairs=len(completed))
            if consecutive_bad>=2:
                raise RuntimeError('Two consecutive memory engineering execution failures')
        report=legacy.summarize(root)
        # Include the five newly frozen behavioral contracts in the final
        # success status as well as retaining the original grader report.
        for row in report['rows']:
            for arm_name in ('memory','full_context'):
                path=root/'cases'/row['case_id']/arm_name/'added-checks.json'
                if path.exists() and read(path)['exit_code']!=0:
                    row['arms'][arm_name].update(mechanical=False,status='fail')
        from collections import Counter
        report['per_arm']={a:dict(Counter(r['arms'][a]['status'] for r in report['rows']))
                           for a in ('memory','full_context')}
        report['note']='15 engineering checkpoints from ten session families; related checkpoints are correlated. Includes independent behavioral checks for all eleven fixed-interface components.'
        save(root/'complete.json',dict(completed_pairs=15,matched_arms=30,
            report=report,added_behavioral_checks={c['id']:{a:read(root/'cases'/c['id']/a/'added-checks.json')
                for a in ('memory','full_context')} for c in records if c['id'] in ('E11','E12','E13','E14','E15')}))
    except Exception as exc:
        save(root/'stopped.json',dict(reason=str(exc),completed_pairs=completed,no_automatic_retry=True))
        raise
    finally:
        (root/'STOP').touch()
        stop.set()
        browser.join(timeout=1)
        provider_thread.join(timeout=5)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','run','arm','audit'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--bundle',type=Path)
    parser.add_argument('--campaign',type=Path)
    parser.add_argument('--case')
    parser.add_argument('--arm',choices=('memory','full_context'))
    args=parser.parse_args()
    root=args.root.resolve()
    if args.phase=='prepare': prepare(root,args.bundle.resolve())
    elif args.phase=='run': run(root,args.campaign.resolve())
    elif args.phase=='arm': arm(root,args.case,args.arm)
    else: audit(root,args.case)
