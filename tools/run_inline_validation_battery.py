"""Sequential ten-history QA and fifteen-pair engineering campaign with stop gates."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.engineering_research_gateway import emit, read, save

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'eval_results/native-spine-ten100-20260922-r1'


def qa_gate(report):
    if report['invalid_grades'] > 2:
        return 'More than two invalid grades in one history'
    if report['support_complete'] < 90:
        return 'Required support coverage below 90/100'
    if report['correct'] < 75:
        return 'Automatic accuracy below 75/100; inspect before spending on remaining histories'
    if report['inline_memory']['accepted'] < 80:
        return 'Fewer than 80/100 accepted inline summary pairs'
    return None


def command(root, label, arguments):
    if (root/'STOP').exists():
        raise RuntimeError('Campaign STOP requested')
    folder = root/'logs'
    folder.mkdir(parents=True, exist_ok=True)
    emit(phase='starting', job=label)
    with (folder/(label+'.stdout.log')).open('w', encoding='utf-8') as out, \
            (folder/(label+'.stderr.log')).open('w', encoding='utf-8') as err:
        child = subprocess.Popen([sys.executable, '-u', *arguments], cwd=REPO, stdout=out, stderr=err)
        save(folder/(label+'.process.json'), dict(pid=child.pid, arguments=arguments))
        while child.poll() is None:
            # Stop only the process started by this controller; its normal
            # exceptions close the model runtime. External STOP waits for the
            # active bounded phase and prevents subsequent work.
            time.sleep(.5)
        code = child.returncode
    if code:
        raise RuntimeError(f'{label} exited {code}; see its saved stderr')
    if (root/'STOP').exists():
        raise RuntimeError('Campaign STOP requested')
    emit(phase='finished', job=label)


def run(root, runtime_root):
    root.mkdir(parents=True, exist_ok=True)
    (root/'controller.reserved').touch(exist_ok=False)
    sources=[]
    for index in range(1,11):
        source=SOURCE/f'history-{index:02}'
        scope=read(source/'scope.json')
        questions=read(source/'questions/questions.json')['questions']
        if scope['through_question_day_body_tokens']<1_000_000 or len(questions)!=100:
            raise ValueError('Unexpected history population')
        if not (source/'application/memory.db').is_file():
            raise ValueError('Existing memory store is missing')
        sources.append(dict(directory=str(source), body_tokens=scope['actual_body_tokens'],
            question_ids=[q['question_id'] for q in questions]))
    save(root/'campaign.json', dict(history_count=10, questions_per_history=100, total_questions=1000,
        engineering_pairs=15, actor='codex_sdk/gpt-5.6-sol', reasoning_effort='none',
        histories=sources, history_reingestions=0, local_gpu_concurrency=1,
        engineering_order='After QA; same-model full-context and memory arms',
        stop_policy=dict(operational='Any failed phase or audit; empty answer/evidence; five consecutive summary fallbacks; sustained backlog >12 exchanges',
                         after_history='Accuracy <75, support <90, invalid grades >2, accepted inline pairs <80 per100; pause for diagnosis, never silently retry'),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    reports=[]
    try:
        for index,source in enumerate(sources,1):
            name=f'history-{index:02}'
            case=root/'qa'/name
            common=['--root',str(case)]
            command(root,name+'-prepare',['tools/evaluate_chat_io_local100.py','prepare',*common,
                '--runtime-root',str(runtime_root/name),'--source',source['directory'],
                '--reader-gateway','https://central-dev.zt:4000/v1',
                '--reader-model','codex_sdk/gpt-5.6-sol','--judge-model','codex_sdk/gpt-5.6-sol',
                '--inline-memory','--stop-on-obvious-problems'])
            command(root,name+'-live',['tools/evaluate_chat_io_local100.py','live',*common])
            command(root,name+'-audit',['tools/evaluate_chat_io_local100.py','audit',*common])
            report=read(case/'report.json')
            reports.append(dict(history=name,correct=report['correct'],support=report['support_complete'],
                inline=report['inline_memory'],mean_answer_s=report['answer_latency']['mean_s']))
            save(root/f'qa-progress-{index:02}.json',dict(completed_histories=index,results=reports))
            reason=qa_gate(report)
            if reason:
                raise RuntimeError(name+': '+reason)
        save(root/'qa-complete.json',dict(histories=10,questions=1000,correct=sum(r['correct'] for r in reports),results=reports))
        ready=root/'engineering-ready.json'
        if not ready.with_suffix('.json.sha256').exists():
            raise RuntimeError('Engineering preparation is not sealed; QA retained and engineering not launched')
        config=read(ready)
        command(root,'engineering',['tools/run_inline_engineering15.py','run','--root',config['root'],
                                    '--campaign',str(root)])
        save(root/'complete.json',dict(qa_histories=10,qa_questions=1000,engineering_pairs=15))
        emit(phase='campaign_complete')
    except Exception as exc:
        save(root/'stopped.json',dict(error_type=type(exc).__name__,reason=str(exc),
                                    completed_qa_histories=len(reports),no_automatic_retry=True))
        (root/'STOP').touch()
        emit(phase='campaign_stopped',reason=str(exc))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--runtime-root',type=Path,required=True)
    args=parser.parse_args()
    run(args.root.resolve(),args.runtime_root.resolve())
