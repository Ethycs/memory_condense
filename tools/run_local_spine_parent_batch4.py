"""Measure four independent Qwen sequences, then continue all parent trees."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
from types import SimpleNamespace

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.spine_summary import parse_spine_summary
from tools.build_spine_corpus_hierarchy import restore_request
from tools.local_qwen_spine_backend import LocalQwenBackend
from tools.matched_eval.artifacts import publish_sealed_json,read_sealed_json
from tools.restore_spine_parent_hierarchy_local import LocalJournal,population
from tools.reuse_local_spine_parent_summaries import freeze_population,verify_implementation


BASELINE_RESPONSES = (
    'a71cf58614e1f582c2fc3765812b839a8ad108ceb8b4d2e09fe3931913e56a78',
    '2d4fb9b836d164ee2d891fb54c308152c90d2ea1b4f9549cc8ba7f8e9bc37dc4')


class BatchFourQwen(LocalQwenBackend):
    max_batch_size = 4

    def __init__(self,*args):
        super().__init__(*args)
        self.identity['batch_adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.identity_sha256 = identity_sha256(self.identity)


def baseline(local_root):
    preflight = read_sealed_json(local_root/'preflight.json')
    verify_implementation(preflight)
    journal = LocalJournal(local_root,preflight,
        SimpleNamespace(identity_sha256=preflight.payload['backend_sha256']),0)
    requests = {read_sealed_json(p).sha256:read_sealed_json(p) for p in (local_root/'requests').glob('*.json')}
    responses = {read_sealed_json(p).sha256:read_sealed_json(p) for p in (local_root/'responses').glob('*.json')}
    jobs,seconds = [],0.
    for sha in BASELINE_RESPONSES:
        response = responses[sha]
        request = requests[response.payload['request_sha256']]
        journal.accept(request,response)
        if request.payload['attempt'] != 0:
            raise ValueError('batch benchmark requires original generation attempts')
        jobs.extend(restore_request(j) for j in request.payload['jobs'])
        seconds += response.payload['elapsed_s']
    if len(jobs)!=4 or any(j.prompt_sha256 not in journal.cache.values for j in jobs):
        raise ValueError('batch benchmark requires four valid bound real summaries')
    return tuple(jobs),seconds


def benchmark(root,local_root,backend):
    jobs,baseline_s = baseline(local_root)
    preflight,_ = publish_sealed_json(root/'batch-preflight.json',{
        'backend':backend.identity,'backend_sha256':backend.identity_sha256,
        'jobs':[asdict(j) for j in jobs],'baseline_response_sha256s':list(BASELINE_RESPONSES),
        'baseline_two_batches_s':baseline_s,'raw_inputs_to_qwen':False,
        'gate':'all four summaries valid, elapsed below two-batch baseline, peak allocation below 5.3 GiB'})
    with (root/'batch-execution.reserved').open('x',encoding='utf-8') as handle:
        handle.write(preflight.sha256+'\n')
    result = backend.generate(jobs,0)
    if (result['backend_sha256'] != backend.identity_sha256 or result['raw_inputs_to_qwen'] is not False
        or result['remote_provider_calls'] != 0 or len(result['rows']) != 4
        or any(row['job_sha256'] != job.prompt_sha256 for job,row in zip(jobs,result['rows'],strict=True))):
        raise ValueError('four-row benchmark response attribution changed')
    validations = []
    for job,row in zip(jobs,result['rows'],strict=True):
        error = None
        try:
            if not row['stopped']:
                raise ValueError('no EOS')
            parse_spine_summary(row['response'],job)
        except ValueError as exc:
            error = str(exc)
        validations.append({'job_sha256':job.prompt_sha256,'valid':error is None,'error':error})
    passed = (all(v['valid'] for v in validations) and result['elapsed_s']<baseline_s
              and result['peak_gpu_allocated_GiB']<5.3)
    artifact,_ = publish_sealed_json(root/'batch-result.json',{
        'preflight_sha256':preflight.sha256,**result,'validations':validations,
        'baseline_two_batches_s':baseline_s,'wall_time_speedup':baseline_s/result['elapsed_s'],
        'batch_release_passed':passed,'accuracy_measured':False,'target_gate_passed':False})
    print({'batch_result_sha256':artifact.sha256,'elapsed_s':result['elapsed_s'],
        'wall_time_speedup':artifact.payload['wall_time_speedup'],
        'peak_gpu_allocated_GiB':result['peak_gpu_allocated_GiB'],'batch_release_passed':passed},flush=True)
    return artifact


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-population',type=Path,required=True)
    parser.add_argument('--local-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--max-new-jobs-per-namespace',type=int,default=10000)
    parser.add_argument('--probe-root',type=Path,default=Path('eval_results/local-qwen-parent-summary-probe-20260910-r1'))
    parser.add_argument('--dependency-root',type=Path,default=Path('.cache/local-qwen-runtime/site-packages'))
    parser.add_argument('--model-root',type=Path,default=Path('../../.cache/models/Qwen3-8B'))
    args = parser.parse_args()
    if args.max_new_jobs_per_namespace<0:
        parser.error('local job allowance must be nonnegative')
    backend = BatchFourQwen(args.probe_root,args.dependency_root,args.model_root)
    result = benchmark(args.output_root,args.local_root/'offset-000',backend)
    if not result.payload['batch_release_passed']:
        raise ValueError('four-row policy did not pass the measured ingest release gate')
    snapshot = freeze_population(args.source_population,args.local_root,args.output_root/'input-snapshots')
    population(snapshot.path,args.output_root/'parents',backend,args.max_new_jobs_per_namespace)
