"""Measure unused coverage-path weights without changing production defaults.

Replay saved summary inputs through the same linker before and after removing
the final MLP and norms that coverage's early exit never executes. This is a
memory/equality probe, not an answer-quality or two-versus-six-layer ablation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
from statistics import mean
import time

from tools.engineering_research_gateway import emit, read, save


def inventory(model):
    result = {}
    for name, parameter in model.named_parameters():
        group = 'embeddings' if name.startswith('embed_tokens.') else 'transformer_and_norm'
        key = f'{group}/{parameter.device.type}/{parameter.dtype}'
        entry = result.setdefault(key, dict(parameters=0, bytes=0))
        entry['parameters'] += parameter.numel()
        entry['bytes'] += parameter.numel() * parameter.element_size()
    return result


def parameter_bytes(module):
    return sum(p.numel() * p.element_size() for p in module.parameters())


def run(root: Path):
    import torch
    from memory_condense.associations.head_memory_models import AssociativeMemoryCandidate
    from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
    from memory_condense.domain._tokenizer import truncate_to_tokens_lossless
    from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
    from memory_condense.search.episodes.surprise_models import EPISODIC_SURPRISE_PROBE

    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    source = Path('eval_results/chat-io-compilation-profile-20260929-r1/cpu.json')
    cases = [dict(name='saved_placement_window', texts=next(iter(read(source)['new_windows'].values())))]
    summary_source = Path('eval_results/llama32-local-20260930-r1/plan.json')
    fragments = []
    for row in read(summary_source)['jobs']:
        typed = row['job'].get('typed_request')
        if typed:
            for fragment in typed['fragments']:
                text = truncate_to_tokens_lossless(fragment['summary'], 128)
                if text and text not in fragments:
                    fragments.append(text)
    for start in range(0, min(len(fragments), 16), 8):
        cases.append(dict(name=f'saved_summary_fragments_{start}', texts=fragments[start:start+8]))
    save(root/'plan.json', dict(
        source_paths=[str(source), str(summary_source)],
        source_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, summary_source)],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=cases, prefix_layers=6, attention_layer=5, dtype='float16',
        host_embeddings=True, provider_calls=0, production_changes=False,
        bge_loaded=False, llama_loaded=False, repeats=3,
    ))

    def memory():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(), free=free, total=total)

    emit(phase='loading', cases=len(cases), gpu=memory())
    started = time.perf_counter()
    encoder = Qwen3PrefixEncoder('.cache/models/Qwen3-8B', layers=6, device='cuda',
                                 dtype='float16', host_embeddings=True)
    linker = QwenMemoryLinker(encoder, layer=5, max_candidates=8, max_workspace_tokens=4096)
    candidates = [tuple(AssociativeMemoryCandidate(f'{c["name"]}-{i}', text)
                        for i, text in enumerate(c['texts'])) for c in cases]
    load_s = time.perf_counter()-started
    emit(phase='loaded', load_s=load_s, gpu=memory())

    def measure(name):
        # Warm the exact shapes before timing; CPU signatures retain no GPU state.
        for rows in candidates:
            linker.inspect_coverage(EPISODIC_SURPRISE_PROBE, rows)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = memory()
        outputs, times = [], []
        for ordinal, rows in enumerate(candidates):
            durations = []
            output = None
            for _ in range(3):
                torch.cuda.synchronize()
                started = time.perf_counter()
                output = linker.inspect_coverage(EPISODIC_SURPRISE_PROBE, rows)
                torch.cuda.synchronize()
                durations.append(time.perf_counter()-started)
            outputs.append(dict(workspace_tokens=output.workspace_tokens,
                                workspace_candidates=output.workspace_candidates,
                                hits=[dict(episode_id=h.episode_id, qk=h.qk_score, ov=h.ov_transport,
                                           heads=list(h.head_weights),
                                           signature=h.transport_signature.tolist()) for h in output.hits]))
            times.append(dict(case=cases[ordinal]['name'], elapsed_s=durations, mean_s=mean(durations)))
        result = dict(inventory=inventory(encoder.model), before=before, after=memory(), times=times)
        save(root/f'{name}-outputs.json', dict(cases=outputs))
        save(root/f'{name}.json', result)
        emit(phase=name, memory=result['after'], mean_s=mean(t['mean_s'] for t in times))
        return result, outputs

    try:
        # These hooks establish that the baseline never executes the modules.
        unused = dict(final_mlp=encoder.model.layers[5].mlp,
                      final_post_attention_norm=encoder.model.layers[5].post_attention_layernorm,
                      synthetic_final_norm=encoder.model.norm)
        unused_bytes = {name: parameter_bytes(module) for name, module in unused.items()}
        two_layer_full_bytes = sum(parameter_bytes(layer) for layer in encoder.model.layers[:2])
        two_layer_coverage_bytes = (two_layer_full_bytes - parameter_bytes(encoder.model.layers[1].mlp)
                                   - parameter_bytes(encoder.model.layers[1].post_attention_layernorm))

        def unexpected(*_args, **_kwargs):
            raise RuntimeError('Coverage executed a supposedly unused module')

        handles = [module.register_forward_pre_hook(unexpected) for module in unused.values()]
        try:
            baseline, expected = measure('baseline')
        finally:
            for handle in handles:
                handle.remove()
        unused.clear()
        handles.clear()

        class UnusedCoverageModule(torch.nn.Module):
            def forward(self, *_args, **_kwargs):
                raise RuntimeError('Pruned module requested outside the coverage path')

        encoder.model.layers[5].mlp = UnusedCoverageModule()
        encoder.model.layers[5].post_attention_layernorm = UnusedCoverageModule()
        encoder.model.norm = UnusedCoverageModule()
        gc.collect()
        torch.cuda.empty_cache()
        reduced, actual = measure('unused_weights_removed')
        exact = actual == expected
        report = dict(
            exact_all_outputs=exact, cases=len(cases), candidates=sum(len(c) for c in candidates),
            unused_parameter_bytes=unused_bytes,
            allocated_reduction_bytes=baseline['before']['allocated']-reduced['before']['allocated'],
            peak_reduction_bytes=baseline['after']['peak_allocated']-reduced['after']['peak_allocated'],
            baseline=baseline, reduced=reduced, load_s=load_s,
            two_layer_weight_estimate=dict(full_layers_bytes=two_layer_full_bytes,
                coverage_layers_bytes=two_layer_coverage_bytes,
                embedding_cpu_bytes=parameter_bytes(encoder.model.embed_tokens),
                measured_quality=False, measured_runtime=False),
            precision_changed=False, early_layer_weights_changed=False,
            provider_calls=0, production_changes=False,
            limitations='Saved summaries only; no full memory pipeline, Llama co-residency, or depth-quality comparison.',
        )
        save(root/'report.json', report)
        emit(phase='complete', exact_all_outputs=exact,
             saved_bytes=report['allocated_reduction_bytes'], peak_saved_bytes=report['peak_reduction_bytes'])
        if not exact:
            raise AssertionError('Unused-weight removal changed coverage outputs')
    finally:
        linker = encoder = None
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
