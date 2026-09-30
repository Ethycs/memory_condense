"""Replay one saved compilation locally; never generate, ingest, or publish memory."""
from pathlib import Path
import argparse
import cProfile
import gc
import shutil
import time

from memory_condense.application.chat_session import ChatEvent
from memory_condense.domain._discourse_identity import identity_sha256
from tools.engineering_research_gateway import read, save, emit
from tools.engineering_research_memory import Compiler, LocalAttention, storage_rows
from tools.engineering_research_resident import ResidentNativeBackend, ResidentAttention
from tools.engineering_research_seed import load_seed
from tools.matched_eval.artifacts import read_sealed_json


SOURCE = Path('eval_results/chat-io-optimization-20260929-r1')
ROOT = Path('eval_results/chat-io-compilation-profile-20260929-r1')


class NoGateway:
    def call(self, *args, **kwargs):
        raise RuntimeError('Offline replay must never call a gateway')


class CachedAttention(LocalAttention):
    def __init__(self, *args):
        super().__init__(*args)
        self.windows = {}

    def load_scorer(self):
        raise RuntimeError('CPU replay requires every attention result already cached')

    def score_sequence(self, texts):
        key = identity_sha256(dict(preflight_sha256=self.preflight.sha256, texts=list(texts)))
        self.windows[key] = list(texts)
        return super().score_sequence(texts)

    def park(self):
        pass


def prepare():
    if ROOT.exists():
        raise ValueError('Use a fresh profile directory')
    ROOT.mkdir(parents=True)
    actor = read(SOURCE/'actor.json')
    events = [ChatEvent(**e) for e in read(SOURCE/'final-events.json')['events']]
    stop = next(i+1 for i, e in enumerate(events) if e.event_id == 'optimized-u2-fresh')
    if stop != 5438:
        raise ValueError('Unexpected saved preparation prefix')
    rows = storage_rows([e.row(actor['source']['family']) for e in events[:stop]])
    for kind in ('atoms', 'merges', 'attention'):
        shutil.copytree(SOURCE/'cache'/kind, ROOT/'cache'/kind)
    compiler = Compiler(ROOT, 'offline-profile', report=lambda **_: None)
    compiler.gateway = NoGateway()
    attention_root = ROOT/'cache/attention'
    attention = CachedAttention(attention_root, read_sealed_json(attention_root/'method.json'))
    backend = ResidentNativeBackend(ROOT, ROOT/'unused', actor)
    backend.compiler, backend.attention = compiler, attention
    started = time.perf_counter()
    backend.seed = load_seed(actor['native_seed'], rows)
    seed_s = time.perf_counter()-started
    emit(phase='seed_loaded_read_only', elapsed_s=seed_s, history_turns=len(backend.seed.turns))
    live = rows[len(backend.seed.turns):]
    backend._compile(live[:-1])
    old_windows = set(attention.windows)
    attention.windows.clear()
    atomic, hierarchy, phases = backend._compile(live)
    receipts = dict(atomic=atomic.receipt_sha256, hierarchy=hierarchy.receipt_sha256)
    new_windows = {k:v for k,v in attention.windows.items() if k not in old_windows}
    if not 1 <= len(new_windows) <= 2:
        raise ValueError('Expected at most two new summary windows')
    # Cached replay again with profiler: profiling overhead is kept separate.
    profiler = cProfile.Profile()
    a2, h2, profiled = profiler.runcall(backend._compile, live)
    profiler.dump_stats(str(ROOT/'cached-compile.prof'))
    assert (a2.receipt_sha256, h2.receipt_sha256) == (atomic.receipt_sha256, hierarchy.receipt_sha256)
    save(ROOT/'cpu.json', dict(history_turns=stop, live_turns=len(live), cold_seed_load_s=seed_s,
        cached_compile=phases, profiled_compile=profiled, receipts=receipts,
        new_windows=new_windows, raw_generation_calls=0, full_history_ingestions=0,
        source_mutations=0, summary_windows_only=True))
    emit(phase='cpu_complete', live_turns=len(live), timings=phases, new_windows=len(new_windows))


def gpu():
    import torch
    from tools.native_spine_engineering_session import StagedEmbedding
    from memory_condense.search.episodes.surprise_models import ScoredSurpriseSequence, AttentionHeadSurpriseReceipt
    torch.set_num_threads(4)
    plan = read(ROOT/'cpu.json')
    marker = ROOT/'gpu.reserved'
    marker.touch(exist_ok=False)
    method = read_sealed_json(ROOT/'cache/attention/method.json')
    encoder = StagedEmbedding(device='cuda', batch_size=8)
    attention = ResidentAttention(ROOT/'cold', method, encoder)
    results = []
    try:
        for arm in ('cold', 'warm'):
            # Match entry to compilation: embedding model on GPU, Qwen absent/parked.
            encoder.embed_queries(['Local diagnostic warmup.'])
            attention.root = ROOT/arm
            attention.values.clear()
            attention.metrics.clear()
            started = time.perf_counter()
            profiler = cProfile.Profile()
            comparisons = []
            for key, texts in plan['new_windows'].items():
                result = profiler.runcall(attention.score_sequence, texts) if arm == 'cold' else attention.score_sequence(texts)
                expected = read(ROOT/'cache/attention/attention'/f'{key}.json')
                reference = ScoredSurpriseSequence(expected['scores'], expected['similarities'],
                    AttentionHeadSurpriseReceipt(**expected['receipt']))
                reference.validate_inputs(texts)
                import numpy as np
                comparisons.append(dict(key=key, scores_match=bool(np.allclose(result.scores, reference.scores, atol=1e-6)),
                    similarities_match=bool(np.allclose(result.similarities, reference.similarities, atol=1e-6))))
            scoring_s = time.perf_counter()-started
            parked = time.perf_counter()
            attention.park()
            park_s = time.perf_counter()-parked
            if arm == 'cold':
                profiler.dump_stats(str(ROOT/'cold-attention.prof'))
            record = dict(arm=arm, scoring_s=scoring_s, park_s=park_s,
                metrics=dict(attention.metrics), comparisons=comparisons,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), cold_has_profiling_overhead=arm=='cold')
            results.append(record)
            save(ROOT/f'{arm}.json', record)
            emit(phase='attention_complete', **record)
        save(ROOT/'gpu.json', dict(results=results, raw_inputs_to_qwen=False, network_calls=0,
            caveat='One changed saved summary window, replayed cold and warm; not a new end-to-end latency measurement.'))
        cold, warm = results
        metrics = warm['metrics']
        save(ROOT/'report.json', dict(
            original_uninstrumented_compilation_remainder_s=11.159091099863872,
            replay_cpu=plan['cached_compile'],
            cold_model_initialization_s=cold['metrics']['model_load_s'],
            warm_model_transfer_s=metrics['embedding_park_s']+metrics['model_resume_s']+warm['park_s'],
            warm_attention_and_cache_s=metrics['score_sequence_s']-metrics['embedding_park_s']-metrics['model_resume_s'],
            attention_matches_saved=all(c['scores_match'] and c['similarities_match']
                                        for r in results for c in r['comparisons']),
            generation_calls=0, full_history_ingestions=0, memory_publications=0,
            limitations=['Cold attention was profiled and includes profiling overhead.',
                         'CPU and GPU components were replayed separately; do not sum them into a measured end-to-end result.',
                         'Original compilation has no fine-grained trace; the old 11.16 seconds cannot be exactly apportioned.']))
    finally:
        attention.park()
        attention.scorer = None
        encoder.close()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','gpu'))
    args = parser.parse_args()
    globals()[args.phase]()
