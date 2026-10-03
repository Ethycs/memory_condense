"""Complete one preserved partial history and two fresh 100-question histories."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tools.engineering_research_gateway import emit, read, save
from tools.run_inline_validation_battery import SOURCE, command, qa_gate


def run(root, runtime_root, predecessor):
    root.mkdir(parents=True, exist_ok=False)
    save(root/'campaign.json', dict(history_count=3, questions_per_history=100,
        predecessor=str(predecessor), history_reingestions=0, local_gpu_concurrency=1,
        empty_response_retries=2, retry_scope='Only completed empty responses; no timeout or ambiguous-outcome retry',
        accuracy='All original 100 questions per history; raw automated grades; attempts reported separately',
        timing='History 01 continuation only, with retained answer latencies; histories 02 and 03 complete cycles',
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    reports=[]
    try:
        for index in range(1,4):
            name=f'history-{index:02}'
            case=root/'qa'/name
            common=['--root',str(case)]
            if index==1:
                args=['resume',*common,'--runtime-root',str(runtime_root/name),
                      '--predecessor',str(predecessor),'--empty-response-retries','2']
            else:
                args=['prepare',*common,'--runtime-root',str(runtime_root/name),'--source',str(SOURCE/name),
                      '--reader-gateway','https://central-dev.zt:4000/v1',
                      '--reader-model','codex_sdk/gpt-5.6-sol','--judge-model','codex_sdk/gpt-5.6-sol',
                      '--inline-memory','--stop-on-obvious-problems','--empty-response-retries','2']
            command(root,name+'-prepare',['tools/evaluate_chat_io_local100.py',*args])
            command(root,name+'-live',['tools/evaluate_chat_io_local100.py','live',*common])
            command(root,name+'-audit',['tools/evaluate_chat_io_local100.py','audit',*common])
            report=read(case/'report.json')
            audit=read(case/'reopen-audit.json')
            reports.append(dict(history=name,correct=report['correct'],support=report['support_complete'],
                invalid_grades=report['invalid_grades'],inline=report['inline_memory'],
                answer_latency=report['answer_latency'],cycle_s=report['cycle']['total_cycle_s'],
                timing_scope=report['cycle']['timing_scope'],recovery=report['gateway_recovery'],audit=audit))
            save(root/f'progress-{index:02}.json',dict(completed_histories=index,results=reports))
            reason=qa_gate(report)
            if reason:
                raise RuntimeError(name+': '+reason)
        save(root/'complete.json',dict(histories=3,questions=300,correct=sum(r['correct'] for r in reports),
            results=reports,predecessor_failure_retained=True,retained_answers=87))
        emit(phase='three_histories_complete',correct=sum(r['correct'] for r in reports),questions=300)
    except Exception as exc:
        save(root/'stopped.json',dict(error_type=type(exc).__name__,reason=str(exc),completed_histories=len(reports)))
        emit(phase='campaign_stopped',reason=str(exc))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--runtime-root',type=Path,required=True)
    parser.add_argument('--predecessor',type=Path,required=True)
    args=parser.parse_args()
    run(args.root.resolve(),args.runtime_root.resolve(),args.predecessor.resolve())
