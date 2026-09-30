"""Compare measured twelve-exchange cycles without excluding the final drain."""
from pathlib import Path
from tools.engineering_research_gateway import read, save
from tools.evaluate_chat_io_batch12 import answer_json


def assess(root):
    report,plan=read(root/'report.json'),read(root/'run-plan.json')
    turns=[read(root/f'turn-{i:02}.json') for i in range(1,13)]
    correct=0
    for row in turns:
        try:
            correct+=answer_json(row['result']['response']['content'])==row['expected']
        except ValueError:
            pass
    syncs=[t for t in report['backend_timings'] if t['operation']=='sync']
    preparations=[t for t in report['backend_timings'] if t['operation']=='prepare']
    return dict(root=str(root),reader=plan['models']['actor'],
        runtime_root=report.get('runtime_root',str(root/'live')),
        runtime_storage_override=report.get('runtime_storage_override',False),
        full_cycle_s=report['total_cycle_s'],under_50_s=report['total_cycle_s']<50,
        answer_window_s=report['total_cycle_s']-report['final_drain_s'],
        answer_mean_s=report['answer_mean_s'],answer_max_s=report['answer_max_s'],
        reader_mean_s=report['reader_mean_s'],final_drain_s=report['final_drain_s'],
        facts_correct=correct,total=12,original_report_correct=report['correct'],
        fenced_json_answers=sum(r['result']['response']['content'].strip().startswith('```') for r in turns),
        lifecycle_checks={k:v for k,v in report['checks'].items() if k!='all_turn_answers_correct'},
        batches_s=report['batch_durations_s'],calls=report['provider_calls'],
        compile_wall_s=sum(t['compile_s'] for t in syncs),
        preparation_jobs=len(preparations),preparation_wall_s=sum(t['elapsed_s'] for t in preparations),
        hierarchy_wall_s=sum(t['compile_timings']['hierarchy_s'] for t in syncs),
        raw_ingest_s=sum(t['raw_ingest_s'] for t in syncs),
        publish_s=sum(t['publish_s'] for t in syncs),startup_s=report['startup_s'])


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('eval_results/sub50-cycle-20260930-assessment.json'))
    args=parser.parse_args()
    roots=[Path('eval_results/chat-io-batch12-20260929-r4')]
    roots.extend(sorted(Path('eval_results').glob('chat-io-batch12-20260930-r*'),
                        key=lambda p:int(p.name.rsplit('-r',1)[1])))
    rows=[assess(root) for root in roots if (root/'report.json').exists()]
    result=dict(runs=rows,baseline_full_cycle_s=rows[0]['full_cycle_s'],
        startup_excluded=True,final_ingestion_and_learning_drain_included=True,
        one_existing_history_per_run=True,history_reingestions=0,
        broad_accuracy_benchmark=False,
        decoder='JSON or one exact Markdown JSON fence; facts and captured outputs unchanged',
        caveats=['One timing sample per configuration; shared-provider latency varies.',
                 'The r15 active memory and chat journal use C: temporary storage; earlier runs use F: project storage.',
                 'Twelve synthetic continuation exchanges are not the hundred-question or engineering benchmark.',
                 'Compilation overlaps raw indexing; their durations must not be summed as elapsed wall time.'])
    save(args.output,result)
    print(json.dumps(result,indent=2))
