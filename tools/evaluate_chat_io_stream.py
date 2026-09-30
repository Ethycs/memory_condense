"""One saved million-token history, 24 sequential replies, bounded streaming IO."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from pathlib import Path
import shutil
import statistics
import time

from memory_condense.application.chat_io import ChatIO
from memory_condense.application.chat_session import ChatEvent
from tools.engineering_research_chat import open_chat
from tools.engineering_research_gateway import read, save, emit, worker
from tools.evaluate_chat_io_batch12 import SOURCE, cases, answer_json
from tools.evaluate_chat_io_complete_turn import warm_existing
from tools.evaluate_chat_io_live import reader


def prepare(root,runtime_root):
    if root.exists() or runtime_root.exists():
        raise ValueError('Use fresh run and runtime directories')
    source=read(SOURCE/'run-plan.json')
    population=cases()
    for i in range(12):
        item=cases()[[1,3,7,9,11,6][i%6]]
        population.append(dict(item,query=f'Continuation check {13+i}. '+item['query']))
    files=[Path(__file__),Path('tools/evaluate_chat_io_batch12.py'),Path('tools/evaluate_chat_io_live.py'),
           Path('tools/evaluate_chat_io_complete_turn.py'),Path('tools/compile_native_spine_attention.py'),
           *Path('tools').glob('engineering_research_*.py'),*Path('src/memory_condense').rglob('*.py')]
    save(root/'run-plan.json',dict(gateway=source['gateway'],models=dict(source['models'],actor='claude_code/claude-haiku-4-5'),
        reasoning_effort=dict(raw='none'),compiler_concurrency=3,streaming=True,recent_exchanges=12,recent_token_budget=8192,
        preparation_workers=3,sealed_hierarchy_exchanges=8,ordered_publication=True,reader_lane_reserved=True,
        runtime_root=str(runtime_root),runtime_storage_override=True,live_exchanges=24,
        budgets={'raw':dict(calls=36,prompt_cap=7000,output_cap=4096,input_token_budget=252000),
                 'merge':dict(calls=64,prompt_cap=2048,output_cap=768,input_token_budget=131072),
                 'actor':dict(calls=24,prompt_cap=16384,output_cap=256,input_token_budget=393216)},
        implementation={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        source=str(SOURCE),history_count=1,history_reingestions=0,synthetic_continuation=True,
        broad_accuracy_benchmark=False,inter_turn_delays=False,final_flush=True,startup_excluded=True,
        qwen_inputs='typed routing summaries; short entries may be verbatim',
        source_files={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('live/store/memory','live/chat') for p in (SOURCE/folder).iterdir() if p.is_file()}))
    save(root/'cases.json',dict(cases=population))
    save(root/'actor.json',read(SOURCE/'actor.json'))
    for folder in ('live/store/memory','live/chat','cache'):
        shutil.copytree(SOURCE/folder,runtime_root/folder.removeprefix('live/') if folder.startswith('live/') else root/folder)
    emit(phase='prepared',exchanges=24,history_count=1,history_reingestions=0,runtime_root=str(runtime_root))


def live(root):
    (root/'live.reserved').touch(exist_ok=False)
    plan=read(root/'run-plan.json')
    started=time.perf_counter()
    results=[]
    try:
        with open_chat(root,Path(plan['runtime_root']),read(root/'actor.json'),streaming=True,
                       recent_exchanges=12,recent_token_budget=8192) as chat:
            chat.flush()
            warm=chat._call(lambda:warm_existing(chat))
            initial=dict(events=len(chat.events()),tokens=chat.backend.last_reopen['snapshot']['body_tokens'])
            startup_s=time.perf_counter()-started
            save(root/'bootstrap.json',dict(startup_s=startup_s,initial=initial,warmup=warm))
            emit(phase='warm_ready',startup_s=startup_s,**initial)
            while not (root/'generation-enabled').exists():
                if (root/'STOP').exists(): raise RuntimeError('Stopped before generation')
                time.sleep(.1)
            measured=time.perf_counter()
            first_timing=len(chat.backend.timings)
            io=ChatIO(chat)
            for ordinal,case in enumerate(read(root/'cases.json')['cases'],1):
                before=time.perf_counter()
                result=io.exchange(ChatEvent(f'batch12-u{ordinal:02}','user',case['query']),
                    request_id=f'batch12-a{ordinal:02}',reader=reader(root,f'batch12-a{ordinal:02}'))
                elapsed=time.perf_counter()-before
                try: answer=answer_json(result['response']['content'])
                except ValueError: answer=None
                recalled=next(t for t in reversed(chat.backend.timings) if t['operation']=='recall')
                published=recalled['published_events']
                lag=max(0,ordinal-1-max(0,(published-initial['events'])//4))
                row=dict(ordinal=ordinal,kind=case['kind'],expected=case['expected'],answer=answer,
                    correct=answer==case['expected'],answer_s=elapsed,reader_s=result['response']['elapsed_s'],
                    end_s=time.perf_counter()-measured,published_events=published,backlog_completed_exchanges=lag,
                    within_recent_window=lag<=12,prompt_tokens_proxy=result['response']['prompt_tokens_proxy'],
                    status=chat.status(),result=result)
                results.append(row)
                save(root/f'turn-{ordinal:02}.json',row)
                emit(phase='answer',**{k:row[k] for k in ('ordinal','correct','answer_s','backlog_completed_exchanges','prompt_tokens_proxy')})
            last_answer=time.perf_counter()-measured
            status=chat.flush()
            full_s=time.perf_counter()-measured
            timings=chat.backend.timings[first_timing:]
            events=chat.events()
            verification=chat.backend.recall_published(cases()[7]['query'])
            save(root/'post-drain-recall.json',verification)
            save(root/'final-events.json',dict(events=[asdict(e) for e in events]))
            original_ids={r['span']['turn_id'] for r in verification['references'] if r['independent']}
            syncs=[t for t in timings if t['operation']=='sync']
            checks=dict(all_answers_correct=all(r['correct'] for r in results),recalls_every_question=sum(t['operation']=='recall' for t in timings)==24,
                learned_all_twenty_four=sum(t['operation']=='learn' for t in timings)==24,
                all_io_captured=len(events)==initial['events']+96,all_io_indexed=status['indexed_events']==len(events),
                no_pending_events=status['pending_events']==0,no_pending_feedback=status['pending_feedback']==0,
                no_errors=status['last_error'] is None,preparation_bounded=all(r['status']['preparation_jobs']<=3 for r in results),
                stayed_within_recent_window=all(r['within_recent_window'] for r in results),
                post_drain_original_input_hydrated='batch12-u01' in original_ids,
                post_drain_exact_facts_hydrated=all(t in verification['text'] for t in ('juniper-641','orchid-927')),
                original_store_unchanged=all(hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()==sha for p,sha in plan['source_files'].items()))
            reservations=[read(p) for p in (root/'gateway').glob('*.reservation.json')]
            report=dict(checks=checks,all_passed=all(checks.values()),initial=initial,
                final=dict(events=len(events),tokens=chat.backend.last_reopen['snapshot']['body_tokens']),
                runtime_root=plan['runtime_root'],runtime_storage_override=True,startup_s=startup_s,startup_excluded=True,
                total_cycle_s=full_s,answer_window_s=last_answer,final_drain_s=full_s-last_answer,
                answer_mean_s=statistics.mean(r['answer_s'] for r in results),answer_max_s=max(r['answer_s'] for r in results),
                reader_mean_s=statistics.mean(r['reader_s'] for r in results),first_twelve_answer_window_s=results[11]['end_s'],
                prompt_tokens_mean=statistics.mean(r['prompt_tokens_proxy'] for r in results),
                max_backlog_completed_exchanges=max(r['backlog_completed_exchanges'] for r in results),
                correct=sum(r['correct'] for r in results),total=len(results),status=status,
                provider_calls=dict(Counter(r['kind'] for r in reservations)),publication_count=len(syncs),
                publication_new_events=[t['new_turns'] for t in syncs],
                backend_timings=timings,turns=[{k:v for k,v in r.items() if k!='result'} for r in results],
                history_reingestions=0,broad_accuracy_benchmark=False)
            save(root/'report.json',report)
            emit(phase='complete',**{k:v for k,v in report.items() if k not in ('turns','backend_timings','status')})
    finally:
        save(root/'progress.json',dict(completed_turns=len(results)))
        (root/'STOP').touch()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','live','gateway'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--runtime-root',type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    if args.phase=='prepare':
        if args.runtime_root is None: parser.error('--runtime-root is required for prepare')
        prepare(root,args.runtime_root.resolve())
    elif args.phase=='gateway':
        (root/'generation-enabled').touch(exist_ok=False)
        worker(root)
    else: live(root)
