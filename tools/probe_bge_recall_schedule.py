"""Exercise real CPU BGE recalls against three queued ingestion jobs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
from pathlib import Path
import threading
import time

import numpy as np

from tools.engineering_research_gateway import emit, read, save
from tools.engineering_research_resident import SharedEmbedding
from tools.probe_bge_cpu_placement import fidelity, stats


def run(root):
    import torch
    from threadpoolctl import threadpool_info
    from memory_condense.domain.schemas import Chunk
    from memory_condense.modeling.embedding import EmbeddingService

    root.mkdir(parents=True,exist_ok=False)
    source=Path('eval_results/bge-cpu-placement-20260930-r2')
    inputs=read(source/'plan.json')
    baseline=read(source/'cpu_fp32_mkl12.json')
    vectors_path=source/'cpu_fp32_mkl12-vectors.npz'
    assert hashlib.sha256(vectors_path.read_bytes()).hexdigest()==baseline['vectors_sha256']
    with np.load(vectors_path) as vectors:
        reference={k:vectors[k].copy() for k in vectors}
    queries=inputs['queries'][:12]
    chunks=[[Chunk(**c) for c in batch] for batch in inputs['chunk_batches']]
    summaries=[s for batch in inputs['summary_batches'] for s in batch]
    files=[Path(__file__),Path('tools/engineering_research_resident.py')]
    save(root/'plan.json',dict(input_plan=str(source/'plan.json'),
        input_sha256=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest(),
        implementation={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        query_count=len(queries),chunks=sum(map(len,chunks)),summaries=len(summaries),
        background_workers=3,background_batch_size=1,mkl_threads_per_worker=12,aten_threads=4,
        provider_calls=0,history_reingestions=0,
        scope='Warm component contention test; no answers, full-cycle or sustained arrival-rate claim.'))
    torch.set_num_threads(4)
    base=EmbeddingService(device='cpu',batch_size=8)
    base._load_model()
    library=ctypes.CDLL(next(p['filepath'] for p in threadpool_info() if p['internal_api']=='mkl'))
    set_mkl=library.MKL_Set_Num_Threads_Local
    set_mkl.argtypes,set_mkl.restype=[ctypes.c_int],ctypes.c_int
    shared=SharedEmbedding(base)
    assert shared.background_batch_size==1
    def initialize():
        # This Windows build needs the override on every model-calling thread,
        # after its first native parallel operation has initialized worker state.
        with shared.model_lock:
            base.embed_query('Warm resident CPU BGE.')
            set_mkl(12)
            assert 'mkl_get_max_threads() : 12' in torch.__config__.parallel_info()

    entered=threading.Event()
    original=base.embed_chunks
    def observe(values):
        entered.set()
        return original(values)
    base.embed_chunks=observe
    try:
        with ThreadPoolExecutor(max_workers=4,initializer=initialize) as pool:
            barrier=threading.Barrier(5)
            warm=[pool.submit(barrier.wait,timeout=90) for _ in range(4)]
            barrier.wait(timeout=90)
            for future in warm: future.result()
            emit(phase='warm_ready',device='cpu',batch_size=shared.background_batch_size)
            started=time.perf_counter()
            jobs=[pool.submit(shared.embed_chunks,batch) for batch in chunks]
            jobs.append(pool.submit(shared.embed_queries,summaries))
            assert entered.wait(10)
            def recall():
                outputs,times=[],[]
                for ordinal,query in enumerate(queries):
                    before=time.perf_counter()
                    previous_wait=shared.metrics['query_lock_wait_s']
                    outputs.append(shared.embed_query(query))
                    row=dict(ordinal=ordinal,elapsed_s=time.perf_counter()-before,
                             wait_s=shared.metrics['query_lock_wait_s']-previous_wait)
                    times.append(row)
                    emit(phase='query',**row)
                return np.stack(outputs),times
            outputs,times=pool.submit(recall).result(timeout=180)
            answered=time.perf_counter()-started
            chunk_outputs=[c for job in jobs[:2] for c in job.result(timeout=180)]
            summary_outputs=jobs[2].result(timeout=180)
            full=time.perf_counter()-started
        compared=dict(query=fidelity(reference['query'][:12],outputs),
                      chunk=fidelity(reference['chunk'],[c.embedding for c in chunk_outputs]),
                      summary=fidelity(reference['summary'],summary_outputs))
        checks=dict(all_vectors_finite=all(v['all_finite'] for v in compared.values()),
                    numerical_agreement=all(v['cosine_min']>0.99999999 for v in compared.values()),
                    all_queries=shared.metrics['query_calls']==12,
                    one_text_batches=shared.metrics['chunk_batches']==16 and shared.metrics['summary_batches']==17,
                    no_gpu_allocation=torch.cuda.memory_allocated()==0)
        save(root/'report.json',dict(checks=checks,all_passed=all(checks.values()),
            query_latency=stats([r['elapsed_s'] for r in times]),
            query_wait=stats([r['wait_s'] for r in times]),queries=times,
            metrics=dict(shared.metrics),fidelity=compared,
            query_window_s=answered,all_background_drained_s=full,final_drain_s=full-answered,
            provider_calls=0,history_reingestions=0))
        emit(phase='complete',all_passed=all(checks.values()),query_window_s=answered,
             all_background_drained_s=full,query_wait_max_s=max(r['wait_s'] for r in times))
        if not all(checks.values()): raise AssertionError(checks)
    finally:
        shared.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
