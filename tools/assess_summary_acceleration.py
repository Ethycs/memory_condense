"""Compare completed live runs without sending data or rerunning histories."""
from pathlib import Path
from tools.engineering_research_gateway import read, save, emit


def main():
    results=[]
    for run in ('r3','r4'):
        root=Path('eval_results/chat-io-batch12-20260929-'+run)
        report=read(root/'report.json')
        syncs=[t for t in report['backend_timings'] if t['operation']=='sync']
        models=[]
        for path in (root/'gateway').glob('*.request.json'):
            request=read(path)
            response=read(path.with_name(path.name.replace('.request.','.response.')))
            models.append((request['kind'],response.get('elapsed_s',0)))
        results.append(dict(run=run,correct=report['correct'],total=report['total'],
            answer_window_s=max(t['end_s'] for t in report['turns']),
            answer_mean_s=report['answer_mean_s'], full_cycle_s=report['total_cycle_s'],
            batch_service_s=sum(report['batch_durations_s']),
            summary_stage_wall_s=sum(t['compile_timings']['atoms_s']+t['compile_timings']['exchanges_s']
                                     +t['compile_timings']['hierarchy_s'] for t in syncs),
            raw_generation_calls=sum(k=='raw' for k,_ in models),
            raw_generation_service_s=sum(s for k,s in models if k=='raw'),
            merge_generation_calls=sum(k=='merge' for k,_ in models),
            merge_generation_service_s=sum(s for k,s in models if k=='merge'),
            raw_index_s=sum(t['raw_ingest_s'] for t in syncs),
            summary_embedding_s=sum(t['summary_embedding_s'] for t in syncs),
            publish_s=sum(t['publish_s'] for t in syncs),checks=report['checks']))
    result=dict(runs=results,full_cycle_reduction_pct=100*(1-results[1]['full_cycle_s']/results[0]['full_cycle_s']),
        model_service_times_overlap_under_concurrency=True,
        same_source_cases_and_reader=True, synthetic_continuation_on_saved_real_history=True,
        independent_replications=1, performance_attribution='combined reuse plus concurrency; not isolated ablation')
    save(Path('eval_results/chat-io-batch12-20260929-r4/comparison.json'),result)
    emit(**result)


if __name__=='__main__': main()
