"""Bounded local placement check with saved summaries; no provider calls."""
from pathlib import Path
import gc
import time

import numpy as np

from tools.engineering_research_gateway import read, save, emit
from tools.engineering_research_resident import ResidentAttention
from tools.native_spine_engineering_session import StagedEmbedding
from tools.compile_native_spine_attention import cache_method


ROOT = Path('eval_results/resident-model-placement-20260929-r2')
SOURCE = Path('eval_results/chat-io-compilation-profile-20260929-r1')


def run():
    import torch
    torch.set_num_threads(4)
    if ROOT.exists():
        raise ValueError('Use a fresh probe directory')
    ROOT.mkdir(parents=True)
    windows = list(read(SOURCE/'cpu.json')['new_windows'].values())
    assert len(windows) == 1
    texts = windows[0]
    encoder = StagedEmbedding(device='cuda', batch_size=8)
    attention = None
    try:
        before = encoder.embed_queries(texts)
        attention = ResidentAttention(ROOT/'staged', cache_method(ROOT/'staged', host_embeddings=False), encoder, retain_gpu=False)
        staged = attention.score_sequence(texts)
        attention.park()
        attention.scorer = None
        del attention
        gc.collect()
        torch.cuda.empty_cache()
        attention = ResidentAttention(ROOT/'resident', cache_method(ROOT/'resident', host_embeddings=True), encoder)
        attention.prepare()
        if not attention.retain_gpu:
            raise RuntimeError('Insufficient GPU headroom for the resident check')
        free, _ = torch.cuda.mem_get_info()
        emit(phase='resident_models_ready', allocated_bytes=torch.cuda.memory_allocated(), free_gpu_bytes=free)
        if free < 512 * 1024**2:
            raise RuntimeError('Resident models leave insufficient attention workspace; stop before forward')
        results = []
        for ordinal in range(2):
            attention.root = ROOT/f'uncached-{ordinal}'
            attention.values.clear()
            attention.metrics.clear()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            signal = attention.score_sequence(texts)
            score_s = time.perf_counter()-started
            attention.park()  # Intentional no-op under the retained placement.
            started = time.perf_counter()
            after = encoder.embed_queries(texts)
            embedding_s = time.perf_counter()-started
            model = attention.scorer.linker.encoder.model
            row = dict(attention_s=score_s, embeddings_s=embedding_s, metrics=dict(attention.metrics),
                attention_matches=bool(np.allclose(signal.similarities, staged.similarities, atol=1e-6)),
                scores_match=bool(np.allclose(signal.scores, staged.scores, atol=1e-6)),
                embeddings_identical=bool(np.array_equal(before, after)),
                qwen_embedding_device=str(model.embed_tokens.weight.device),
                qwen_layer_device=str(next(model.layers[0].parameters()).device),
                bge_device=str(next(encoder._model.parameters()).device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                free_gpu_bytes=torch.cuda.mem_get_info()[0])
            results.append(row)
            save(ROOT/f'pass-{ordinal}.json', row)
            emit(phase='resident_pass', ordinal=ordinal, **row)
        assert all(r['attention_matches'] and r['scores_match'] and r['embeddings_identical'] for r in results)
        save(ROOT/'report.json', dict(results=results, provider_calls=0, model_precision_changed=False))
    finally:
        if attention is not None:
            attention.scorer = None
        encoder.close()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    run()
