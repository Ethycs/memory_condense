"""Compare four identical parent-summary jobs with the saved local Qwen batch."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval.thread_local_provider_v2 import ThreadLocalProvider
from memory_condense.search.spine_summary import parse_spine_summary
from tools.build_spine_corpus_hierarchy import restore_request
from tools.local_qwen_spine_backend import job_messages
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _completion_client
from tools.run_local_spine_parent_batch4 import baseline


MODEL = 'codex_sdk/gpt-5.6-terra'
GATEWAY = 'https://central-dev.zt:4000/v1'
LOCAL_ROOT = Path('eval_results/full1m-spine-parents-local-20260910-r1/offset-000')
BATCH_ROOT = Path('eval_results/full1m-spine-parents-local-batch4-20260910-r1')


def prepare(root):
    jobs, _ = baseline(LOCAL_ROOT)
    saved = read_sealed_json(BATCH_ROOT/'batch-result.json')
    saved_preflight = read_sealed_json(BATCH_ROOT/'batch-preflight.json')
    if (saved.payload['preflight_sha256'] != saved_preflight.sha256
            or saved.payload['batch_release_passed'] is not True
            or identity_sha256(saved_preflight.payload['jobs']) != identity_sha256([asdict(job) for job in jobs])
            or len(jobs) != 4):
        raise ValueError('requires the exact four jobs from the valid local Qwen batch')
    calls = [{'ordinal':i, 'job':asdict(job), 'messages':job_messages(job, 0)}
             for i, job in enumerate(jobs)]
    artifact, _ = publish_sealed_json(root/'preflight.json', {
        'format':'memory-condense-terra-parent-throughput-probe-v1',
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'local_batch_sha256':saved.sha256, 'local_batch_elapsed_s':saved.payload['elapsed_s'],
        'calls':calls, 'model':MODEL, 'gateway':GATEWAY, 'max_tokens':256,
        'concurrency':4, 'physical_call_cap':4, 'automatic_retries':0,
        'raw_inputs_sent':False, 'inputs':'authenticated child summaries only',
        'client_setup_included':True, 'answer_accuracy_measured':False,
        'target_gate_passed':False})
    print({'preflight_sha256':artifact.sha256, 'calls':4,
           'local_batch_elapsed_s':saved.payload['elapsed_s'], 'new_provider_calls':0}, flush=True)
    return artifact


def run(root):
    preflight = prepare(root)
    with (root/'execution.reserved').open('x', encoding='utf-8') as handle:
        handle.write(preflight.sha256+'\n')
    provider = ThreadLocalProvider(lambda: _completion_client('LITELLM_KEY', GATEWAY))

    def complete(call):
        job = restore_request(call['job'])
        if call['messages'] != job_messages(job, 0):
            raise ValueError('summary messages changed')
        request, _ = publish_sealed_json(root/'requests'/f"{call['ordinal']:02}.json", {
            'preflight_sha256':preflight.sha256, 'call':call,
            'messages_sha256':identity_sha256(call['messages'])})
        started = time.perf_counter()
        row = {'ordinal':call['ordinal'], 'job_sha256':job.prompt_sha256,
               'request_sha256':request.sha256, 'valid':False}
        try:
            response = provider.chat.completions.create(model=MODEL,
                messages=call['messages'], max_tokens=256, timeout=60)
            choice = response.choices[0]
            row.update({'response':choice.message.content, 'finish_reason':choice.finish_reason,
                        'reported_model':response.model,
                        'usage':response.usage.model_dump() if response.usage else None})
            raw, _ = publish_sealed_json(root/'responses'/f"{call['ordinal']:02}.json", {
                'request_sha256':request.sha256, 'response':response.model_dump(mode='json')})
            row['response_sha256'] = raw.sha256
            if choice.finish_reason != 'stop':
                raise ValueError('summary did not finish normally')
            parse_spine_summary(choice.message.content, job)
            row['valid'] = True
        except Exception as exc:
            row.update({'error_type':type(exc).__name__, 'http_status':getattr(exc, 'status_code', None)})
            if isinstance(exc, ValueError):
                row['validation_error'] = str(exc)
        row['elapsed_s'] = time.perf_counter()-started
        publish_sealed_json(root/'observations'/f"{call['ordinal']:02}.json", row)
        print({k:row[k] for k in ('ordinal', 'valid', 'elapsed_s')}, flush=True)
        return row

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(complete, preflight.payload['calls']))
    finally:
        provider.close()
    elapsed = time.perf_counter()-started
    result, _ = publish_sealed_json(root/'result.json', {
        'preflight_sha256':preflight.sha256, 'rows':rows, 'elapsed_s':elapsed,
        'wall_time_speedup':preflight.payload['local_batch_elapsed_s']/elapsed,
        'all_four_valid':all(row['valid'] for row in rows),
        'faster_than_saved_qwen_batch':elapsed < preflight.payload['local_batch_elapsed_s'],
        'provider_attempts':4, 'saved_provider_responses':sum('response_sha256' in row for row in rows),
        'automatic_retries':0, 'raw_inputs_sent':False,
        'answer_accuracy_measured':False, 'target_gate_passed':False})
    print({'result_sha256':result.sha256, **{k:v for k,v in result.payload.items() if k!='rows'}}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if args.action == 'run' and not args.enable_provider:
        parser.error('run requires --enable-provider for exactly four summary-only calls')
    (prepare if args.action == 'prepare' else run)(args.output_root)
