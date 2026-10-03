"""Compare resident FP32 BGE on CPU/GPU using saved live-chat inputs only."""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, median
import time
import winreg

import numpy as np

from tools.engineering_research_gateway import emit, read, save


def stats(values):
    return dict(calls=len(values), mean_s=mean(values), median_s=median(values),
                p95_s=float(np.percentile(values, 95)), max_s=max(values))


def fidelity(before, after):
    first = np.asarray(before, dtype=np.float64)
    second = np.asarray(after, dtype=np.float64)
    first = first.reshape(-1, first.shape[-1])
    second = second.reshape(-1, second.shape[-1])
    cosine = np.sum(first*second, axis=1)/(np.linalg.norm(first, axis=1)*np.linalg.norm(second, axis=1))
    return dict(vectors=len(first), all_finite=bool(np.isfinite(second).all()),
                exact=bool(np.array_equal(first, second)), cosine_min=float(cosine.min()),
                cosine_mean=float(cosine.mean()), max_absolute_error=float(np.max(np.abs(first-second))))


def run(root: Path, baseline_root: Path | None = None):
    import psutil
    import torch
    from threadpoolctl import threadpool_info
    from memory_condense.ingest.chunker import Chunker
    from memory_condense.modeling.embedding import EmbeddingService

    root.mkdir(parents=True, exist_ok=False)
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    source = Path('eval_results/chat-io-stream24-20260930-r1')
    query_source = source/'cases.json'
    summary_source = Path('eval_results/qwen-unused-memory-20260930-r2/plan.json')
    event_source = source/'final-events.json'
    queries = [c['query'] for c in read(query_source)['cases']]
    summary_batches = [c['texts'] for c in read(summary_source)['cases']]
    chunker = Chunker()
    chunks = []
    for event in read(event_source)['events'][-96:]:
        chunks.extend(chunker.chunk_turn(event['event_id'], event['text']))
        if len(chunks) >= 16:
            break
    chunk_batches = [chunks[:8], chunks[8:16]]
    assert len(queries) == 24 and all(len(b) == 8 for b in chunk_batches)
    plan = dict(cpu=cpu, physical_cores=psutil.cpu_count(logical=False), logical_cores=os.cpu_count(),
        ram_total=psutil.virtual_memory().total, ram_available=psutil.virtual_memory().available,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        input_sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                       for p in (query_source, summary_source, event_source)],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        queries=queries, summary_batches=summary_batches,
        chunk_batches=[[c.model_dump(mode='json') for c in b] for b in chunk_batches],
        query_repetitions=2, batch_repetitions=2, aten_threads=4, mkl_thread_candidates=[1,4,8,12],
        baseline_root=str(baseline_root) if baseline_root else None,
        dtype='float32', provider_calls=0, history_reingestions=0, production_changes=False,
        placement='One model moved between diagnostic arms only; no per-request transfers.',
        limits='Component replay, not end-to-end chat or answer accuracy. Query priority/queue wait and concurrent Qwen/Llama not exercised.')
    save(root/'plan.json', plan)
    torch.set_num_threads(4)
    service = EmbeddingService(device='cpu' if baseline_root else 'cuda', batch_size=8)
    started = time.perf_counter()
    model = service._load_model()
    load_s = time.perf_counter()-started
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    weight_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
    tokenizer = model.tokenizer
    token_stats = dict(query_lengths=[len(tokenizer.encode(q)) for q in queries],
        summary_batch_lengths=[[len(tokenizer.encode(t)) for t in b] for b in summary_batches],
        chunk_batch_lengths=[[len(tokenizer.encode(c.text)) for c in b] for b in chunk_batches])
    save(root/'runtime.json', dict(load_s=load_s, model_weight_bytes=weight_bytes,
         checkpoint_sha256=service._verified_checkpoint_sha256, token_lengths=token_stats))
    emit(phase='loaded', load_s=load_s, weight_bytes=weight_bytes)

    def call(kind, index):
        if kind == 'query':
            return service.embed_query(queries[index])
        if kind == 'summary':
            return service.embed_queries(summary_batches[index])
        return np.asarray([c.embedding for c in service.embed_chunks(chunk_batches[index])], dtype=np.float32)

    def measure(name, device, mkl_threads):
        for kind in ('query', 'summary', 'chunk'):
            call(kind, 0)
        gc.collect()
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        records, outputs = {}, {}
        for kind, count in (('query', len(queries)), ('summary', len(summary_batches)), ('chunk', len(chunk_batches))):
            times, values = [], []
            for repetition in range(2):
                for index in range(count):
                    if device == 'cuda': torch.cuda.synchronize()
                    started = time.perf_counter()
                    value = call(kind, index)
                    if device == 'cuda': torch.cuda.synchronize()
                    elapsed = time.perf_counter()-started
                    times.append(dict(index=index, repetition=repetition, elapsed_s=elapsed))
                    if repetition == 0: values.append(value)
            records[kind] = dict(**stats([t['elapsed_s'] for t in times]), timings=times)
            outputs[kind] = np.concatenate([np.asarray(v).reshape(-1, 1024) for v in values], axis=0)
            emit(phase=name, workload=kind, **stats([t['elapsed_s'] for t in times]))
        result = dict(device=device, aten_threads=4, mkl_threads=mkl_threads, workloads=records,
            gpu_allocated_bytes=torch.cuda.memory_allocated(),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
            process_rss_bytes=psutil.Process().memory_info().rss)
        np.savez(root/f'{name}-vectors.npz', **outputs)
        result['vectors_sha256'] = hashlib.sha256((root/f'{name}-vectors.npz').read_bytes()).hexdigest()
        save(root/f'{name}.json', result)
        return result, outputs

    try:
        if baseline_root:
            prior_plan = read(baseline_root/'plan.json')
            assert prior_plan['queries'] == queries and prior_plan['summary_batches'] == summary_batches
            assert [[c['text'] for c in b] for b in prior_plan['chunk_batches']] == [[c.text for c in b] for b in chunk_batches]
            gpu = read(baseline_root/'gpu_fp32.json')
            vectors_path = baseline_root/'gpu_fp32-vectors.npz'
            assert hashlib.sha256(vectors_path.read_bytes()).hexdigest() == gpu['vectors_sha256']
            with np.load(vectors_path) as arrays:
                reference = {kind:arrays[kind].copy() for kind in ('query','summary','chunk')}
        else:
            gpu, reference = measure('gpu_fp32', 'cuda', None)
        started = time.perf_counter()
        model.to('cpu')
        service._device = 'cpu'
        gc.collect()
        torch.cuda.empty_cache()
        park_s = time.perf_counter()-started
        # Initialize native worker state, then override MKL on this execution
        # thread. This build otherwise uses one MKL thread despite ATen=4.
        call('query', 0)
        library = ctypes.CDLL(next(p['filepath'] for p in threadpool_info() if p['internal_api'] == 'mkl'))
        set_mkl = library.MKL_Set_Num_Threads_Local
        set_mkl.argtypes, set_mkl.restype = [ctypes.c_int], ctypes.c_int
        original_mkl_threads = set_mkl(1)
        pilots = []
        for threads in (1,4,8,12):
            set_mkl(threads)
            parallel_info = torch.__config__.parallel_info()
            assert f'mkl_get_max_threads() : {threads}' in parallel_info
            call('query', 0)
            query_times, batch_times = [], []
            for repeat in range(1):
                for index in (1,5,9,13):
                    started = time.perf_counter()
                    call('query', index)
                    query_times.append(time.perf_counter()-started)
                started = time.perf_counter()
                call('summary', 1)
                batch_times.append(time.perf_counter()-started)
            pilot = dict(threads=threads, aten_threads=4, parallel_info=parallel_info,
                         query=stats(query_times), summary_batch8=stats(batch_times),
                         selection_score_s=4*mean(query_times)+mean(batch_times))
            pilots.append(pilot)
            save(root/f'pilot-mkl{threads}.json', pilot)
            emit(phase='cpu_thread_pilot', threads=threads, query=pilot['query'], summary_batch8=pilot['summary_batch8'])
        save(root/'thread-pilot.json', dict(results=pilots,
             selection='Lowest 4*mean single-query latency + mean eight-summary batch latency.'))
        selected = min(pilots, key=lambda p:p['selection_score_s'])['threads']
        cpu_arms = {}
        for threads in (selected,):
            set_mkl(threads)
            arm, outputs = measure(f'cpu_fp32_mkl{threads}', 'cpu', threads)
            cpu_arms[str(threads)] = dict(measurement=arm,
                 fidelity={kind:fidelity(reference[kind], outputs[kind]) for kind in outputs},
                 added_mean_s={kind:arm['workloads'][kind]['mean_s']-gpu['workloads'][kind]['mean_s'] for kind in outputs})
        longest = sorted(chunk_batches[0], key=lambda c:len(tokenizer.encode(c.text)), reverse=True)
        microbatches = []
        for size in (1,2,4):
            samples=[]
            for repeat in range(2):
                started=time.perf_counter()
                service.embed_chunks(longest[:size])
                samples.append(time.perf_counter()-started)
            row=dict(chunks=size, **stats(samples))
            microbatches.append(row)
            emit(phase='cpu_chunk_microbatch', **row)
        set_mkl(original_mkl_threads)
        save(root/'report.json', dict(gpu=gpu, cpu=cpu_arms, selected_cpu_threads=selected,
             cpu_thread_pilot=pilots, chunk_microbatches=microbatches, aten_threads=4,
             selected_setting='MKL per-calling-thread override, not ATen intraop count',
             load_s=load_s, diagnostic_gpu_to_cpu_s=park_s,
             model_weight_bytes=weight_bytes, provider_calls=0, production_changes=False,
             latency_scope='Warm component calls; model loading, one-time placement, and queue wait excluded.',
             limitations=plan['limits']))
        emit(phase='complete', selected_cpu_threads=selected)
    finally:
        model = None
        service.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='Reuse a sealed GPU control with identical inputs.')
    args = parser.parse_args()
    run(args.output, args.baseline)
