"""Replay saved reader and summary requests on a temporary local CUDA server."""
from pathlib import Path
from statistics import mean, median
import argparse
import hashlib
import json
import os
import socket
import subprocess
import time

import httpx
from openai import OpenAI

from tools.engineering_research_gateway import read, save, emit
from tools.evaluate_chat_io_batch12 import answer_json
from memory_condense.search.spine_summary import (
    SpineSummaryRequest, SpineSummaryFragment, parse_spine_summary,
)


SOURCE = Path('eval_results/chat-io-stream24-20260930-r1')
MODEL = Path('.cache/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf')
MODEL_SHA = '6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff'
RUNTIME = Path('.cache/runtimes/llama-b11272-bin-win-cuda-12.4-x64')
CUDA = Path('.cache/runtimes/cudart-llama-bin-win-cuda-12.4-x64')


from memory_condense.runtime.llama import gpu, invoke





def stats(rows):
    durations = sorted(r['elapsed_s'] for r in rows)
    timing_rows = [r['timings'] for r in rows if r.get('timings')]
    result = dict(calls=len(rows),
                  mean_s=mean(durations), median_s=median(durations), max_s=max(durations),
                  mean_ttft_s=mean(r['ttft_s'] for r in rows if r.get('ttft_s') is not None),
                  below_5s=sum(t < 5 for t in durations))
    if all(r['kind']=='actor' for r in rows):
        result['correct'] = sum(r.get('correct', False) for r in rows)
    if timing_rows:
        for tokens, milliseconds, label in [('prompt_n','prompt_ms','prefill_tokens_per_s'),
                                            ('predicted_n','predicted_ms','decode_tokens_per_s')]:
            elapsed = sum(t.get(milliseconds, 0) for t in timing_rows)
            if elapsed:
                # llama.cpp counts the first generated token in prompt evaluation.
                offset = 1 if tokens == 'predicted_n' else 0
                result[label] = 1000 * sum(max(t.get(tokens, 0)-offset, 0) for t in timing_rows) / elapsed
        result['mean_prompt_tokens'] = mean(t.get('prompt_n', 0) for t in timing_rows)
    return result


def main(root):
    root.mkdir(parents=True, exist_ok=False)
    with MODEL.open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != MODEL_SHA:
        raise ValueError('Model checksum mismatch')
    cases = read(SOURCE/'cases.json')['cases']
    jobs = []
    for path in (SOURCE/'gateway').glob('*.request.json'):
        job = read(path)
        if job['kind'] not in ('actor', 'merge'):
            continue
        baseline = read(path.with_name(path.name.replace('.request.', '.response.')))
        jobs.append(dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         job=job, baseline=baseline))
    jobs.sort(key=lambda row: (row['job']['kind'], row['job']['scope'], row['source']))
    if sum(r['job']['kind']=='actor' for r in jobs) != 24 or len(jobs) != 30:
        raise ValueError('Expected 24 reader and six summary requests')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    args = [str((RUNTIME/'llama-server.exe').resolve()), '-m', str(MODEL.resolve()),
            '--host', '127.0.0.1', '--port', str(port), '-ngl', '99',
            '-c', '16384', '-np', '1', '-t', '6', '-tb', '6',
            '--flash-attn', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0',
            '--cache-ram', '0', '--fit', 'off', '--metrics']
    save(root/'plan.json', dict(model=str(MODEL), model_sha256=actual, args=args,
        quantization='Q4_K_M', kv_cache='Q8_0', prompt_cache=False,
        source=str(SOURCE), jobs=jobs, maximum_calls=31, warmup_calls=1,
        history_ingestions=0, full_pipeline=False, network='loopback only',
        comparison='Same saved messages; historical provider timing, not a simultaneous control',
        summary_semantic_accuracy_proven=False, gpu_before=gpu()))
    environment = dict(os.environ)
    environment['PATH'] = str(CUDA.resolve()) + os.pathsep + environment.get('PATH', '')
    results = []
    started = time.perf_counter()
    with (root/'server.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            with httpx.Client(timeout=2, trust_env=False) as health:
                for _ in range(240):
                    if process.poll() is not None:
                        raise RuntimeError('Server exited; inspect server.log')
                    try:
                        if health.get(f'http://127.0.0.1:{port}/health').status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(.5)
                else:
                    raise TimeoutError('Local model did not load within 120 seconds')
            load_s = time.perf_counter() - started
            save(root/'runtime.json', dict(load_s=load_s, gpu_loaded=gpu(), pid=process.pid))
            emit(phase='ready', load_s=load_s)
            with OpenAI(base_url=f'http://127.0.0.1:{port}/v1', api_key='local-only',
                        max_retries=0, timeout=120,
                        http_client=httpx.Client(timeout=120, trust_env=False)) as client:
                warm = invoke(client, dict(messages=[dict(role='user',content='Reply with OK.')], max_tokens=8))
                save(root/'warmup.json', warm)
                for ordinal, row in enumerate(jobs, 1):
                    job = row['job']
                    prefix = root/f'{ordinal:02d}-{job["kind"]}'
                    prefix.with_suffix('.reserved').touch(exist_ok=False)
                    result = invoke(client, job)
                    result.update(kind=job['kind'], scope=job['scope'], source=row['source'],
                                  baseline_elapsed_s=row['baseline']['elapsed_s'])
                    try:
                        if job['kind'] == 'actor':
                            index = int(job['scope'].rsplit('a', 1)[1])-1
                            result['case_kind'] = cases[index]['kind']
                            result['expected'] = cases[index]['expected']
                            result['parsed'] = answer_json(result['content'])
                            result['correct'] = result['parsed'] == result['expected'] and result['finish_reason']=='stop'
                        else:
                            typed = dict(job['typed_request'])
                            typed['fragments'] = tuple(SpineSummaryFragment(**f) for f in typed['fragments'])
                            result['parsed'] = parse_spine_summary(result['content'], SpineSummaryRequest(**typed))
                            result['valid'] = result['finish_reason']=='stop'
                    except ValueError as exc:
                        result.update(correct=False, valid=False, validation_error=str(exc))
                    save(prefix.with_suffix('.json'), result)
                    results.append(result)
                    emit(phase='request', ordinal=ordinal, **{k:result.get(k) for k in
                         ('kind','scope','elapsed_s','ttft_s','correct','valid')})
            reader = [r for r in results if r['kind']=='actor']
            summaries = [r for r in results if r['kind']=='merge']
            report = dict(reader=stats(reader), summaries=stats(summaries),
                summary_valid=sum(r.get('valid',False) for r in summaries),
                haiku_historical_mean_s=mean(r['baseline_elapsed_s'] for r in reader),
                qwen_historical_mean_s=mean(r['baseline_elapsed_s'] for r in summaries),
                cold_load_s=load_s, first_generation_warmup_s=warm['elapsed_s'],
                startup_plus_first_generation_s=load_s+warm['elapsed_s'],
                gpu_after=gpu(), history_ingestions=0,
                full_pipeline=False, broad_accuracy_proven=False,
                summary_semantic_accuracy_proven=False, generation_calls=len(results)+1)
            save(root/'report.json',report)
            emit(phase='complete',**report)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            save(root/'shutdown.json',dict(server_stopped=process.poll() is not None, gpu=gpu()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('eval_results/llama32-local-20260930-r1'))
    main(parser.parse_args().root)
