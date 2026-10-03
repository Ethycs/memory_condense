"""Losslessly compress Qwen's used GPU weights and restore one matrix per call.

Every restored FP16 byte is checked during preparation. Paired linker outputs
must also match exactly. nvCOMP is imported from an isolated experiment cache;
production construction and precision remain unchanged.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import time

from tools.engineering_research_gateway import emit, read, save


def run(root: Path, package: Path):
    sys.path.insert(0, str(package.resolve()))
    import torch
    import torch.nn.functional as functional
    from nvidia import nvcomp
    from memory_condense.associations.head_memory_models import AssociativeMemoryCandidate
    from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
    from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
    from memory_condense.search.episodes.surprise_models import EPISODIC_SURPRISE_PROBE

    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    stream = torch.cuda.current_stream().cuda_stream

    class TorchBuffer:
        def __init__(self, nbytes, _stream):
            # This probe confines codec and GEMMs to the current Torch stream.
            self.tensor = torch.empty(nbytes, dtype=torch.uint8, device='cuda')
            self.ptr = self.tensor.data_ptr()

    nvcomp.set_device_allocator(TorchBuffer)
    source = Path('eval_results/qwen-unused-memory-20260930-r2/plan.json')
    cases = read(source)['cases']
    save(root/'plan.json', dict(cases=cases, source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        package_source=json.loads((package.parent/'source.json').read_text()),
        nvcomp=nvcomp.__version__, nvcomp_cuda=nvcomp.__cuda_version__, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), arms=['fp16_pruned', 'ans_plain', 'ans_byteplanes'],
        dtype='unchanged FP16', reductions='unchanged FP32', quantization=False,
        production_changes=False, provider_calls=0, repeats=3, bge_loaded=False, llama_loaded=False))

    class Unused(torch.nn.Module):
        def forward(self, *_args, **_kwargs):
            raise RuntimeError('Coverage unexpectedly executed a removed module')

    encode_codec = None
    decode_codec = nvcomp.Codec(algorithm='ANS', cuda_stream=stream)

    def wrap(tensor):
        return nvcomp.as_array(tensor, cuda_stream=stream)

    class LosslessLinear(torch.nn.Module):
        def __init__(self, weight, layout):
            super().__init__()
            self.shape = tuple(weight.shape)
            self.layout = layout
            self.byte_shape = (2, weight.numel()) if layout == 'ans_byteplanes' else (weight.numel(), 2)
            original = weight.view(torch.uint8).reshape(-1, 2)
            source_bytes = original.T.contiguous() if layout == 'ans_byteplanes' else original
            encoded = encode_codec.encode(wrap(source_bytes))
            # DLPack exposes allocation capacity, which exceeds encoded length.
            # Compact it once; retaining capacity would erase the memory saving.
            packed = torch.from_dlpack(encoded).reshape(-1)[:encoded.buffer_size].clone()
            self.register_buffer('packed', packed)
            self.config = decode_codec.decompression_config(wrap(self.packed))
            with torch.inference_mode():
                restored = self.restore()
                if not torch.equal(weight.view(torch.uint8), restored.view(torch.uint8)):
                    raise AssertionError('Lossless weight restoration changed a bit')

        def restore(self):
            if torch.cuda.current_stream().cuda_stream != stream:
                raise RuntimeError('Probe codec must remain on its original stream')
            expanded = torch.empty(self.byte_shape, device='cuda', dtype=torch.uint8)
            decode_codec.decode(wrap(self.packed), out=wrap(expanded), decompression_config=self.config)
            if self.layout == 'ans_byteplanes':
                expanded = expanded.T.contiguous()
            return expanded.view(torch.float16).reshape(self.shape)

        def forward(self, inputs):
            if torch.is_grad_enabled() or inputs.dtype != torch.float16 or inputs.device != self.packed.device:
                raise RuntimeError('Lossless window requires same-GPU FP16 inference')
            expanded = self.restore()
            try:
                return functional.linear(inputs, expanded)
            finally:
                del expanded

    def memory():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(), free=free, total=total)

    emit(phase='loading', gpu=memory())
    encoder = Qwen3PrefixEncoder('.cache/models/Qwen3-8B', layers=6, device='cuda',
                                 dtype='float16', host_embeddings=True)
    encoder.model.layers[5].mlp = Unused()
    encoder.model.layers[5].post_attention_layernorm = Unused()
    encoder.model.norm = Unused()
    linker = QwenMemoryLinker(encoder, layer=5, max_candidates=8, max_workspace_tokens=4096)
    candidates = [tuple(AssociativeMemoryCandidate(f'{c["name"]}-{i}', text)
                        for i, text in enumerate(c['texts'])) for c in cases]
    # CPU copies are experimental controls, never read during timed forwards.
    originals = {name: module.weight.detach().cpu().clone()
                 for name, module in encoder.model.named_modules() if isinstance(module, torch.nn.Linear)}
    fp16_bytes = sum(w.nbytes for w in originals.values())
    largest_matrix = max(w.nbytes for w in originals.values())
    emit(phase='loaded', linear_matrices=len(originals), fp16_linear_bytes=fp16_bytes,
         largest_expanded_matrix_bytes=largest_matrix)

    def measure(name):
        for rows in candidates:
            linker.inspect_coverage(EPISODIC_SURPRISE_PROBE, rows)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = memory()
        outputs, times = [], []
        for case, rows in zip(cases, candidates, strict=True):
            durations = []
            for _ in range(3):
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = linker.inspect_coverage(EPISODIC_SURPRISE_PROBE, rows)
                torch.cuda.synchronize()
                durations.append(time.perf_counter()-started)
            outputs.append(dict(name=case['name'], workspace_tokens=result.workspace_tokens,
                workspace_candidates=result.workspace_candidates,
                hits=[dict(episode_id=h.episode_id, qk=h.qk_score, ov=h.ov_transport,
                           heads=list(h.head_weights), signature=h.transport_signature.tolist()) for h in result.hits]))
            times.append(dict(name=case['name'], elapsed_s=durations, mean_s=mean(durations)))
        stats = dict(before=before, after=memory(), times=times, mean_s=mean(t['mean_s'] for t in times))
        save(root/f'{name}-outputs.json', dict(cases=outputs))
        save(root/f'{name}.json', stats)
        emit(phase=name, gpu=stats['after'], mean_s=stats['mean_s'])
        return stats, outputs

    try:
        baseline, reference = measure('fp16_pruned')
        arms = {}
        for layout in ('ans_plain', 'ans_byteplanes'):
            encode_codec = nvcomp.Codec(algorithm='ANS', cuda_stream=stream)
            started = time.perf_counter()
            weights = []
            for ordinal, (name, original) in enumerate(originals.items()):
                parent_name, child_name = name.rsplit('.', 1)
                parent = encoder.model.get_submodule(parent_name)
                replacement = LosslessLinear(original.to('cuda'), layout)
                parent._modules[child_name] = replacement
                weights.append(dict(name=name, original_bytes=original.nbytes,
                                    compressed_bytes=replacement.packed.nbytes, exact=True))
                if (ordinal+1) % 10 == 0:
                    emit(phase='compressed', arm=layout, matrices=ordinal+1)
            del parent, replacement
            torch.cuda.synchronize()
            conversion_s = time.perf_counter()-started
            # Compression has a larger workspace than decompression. The
            # resident reader must not retain one-time preparation workspace.
            encode_codec = None
            gc.collect()
            torch.cuda.empty_cache()
            stats, outputs = measure(layout)
            exact = outputs == reference
            arms[layout] = dict(stats=stats, weights=weights, all_restored_weights_exact=True,
                exact_linker_outputs=exact, preparation_s=conversion_s,
                compressed_weight_bytes=sum(w['compressed_bytes'] for w in weights),
                allocated_saved_bytes=baseline['before']['allocated']-stats['before']['allocated'],
                peak_saved_bytes=baseline['after']['peak_allocated']-stats['after']['peak_allocated'],
                device_free_gain_bytes=stats['after']['free']-baseline['after']['free'])
            save(root/f'{layout}-comparison.json', arms[layout])
            emit(phase='comparison', arm=layout, exact=exact,
                 compressed_bytes=arms[layout]['compressed_weight_bytes'],
                 peak_saved_bytes=arms[layout]['peak_saved_bytes'])
            if not exact:
                raise AssertionError('Lossless arm changed linker outputs')
        save(root/'report.json', dict(baseline=baseline, arms=arms,
            candidates=sum(len(c) for c in candidates), cases=len(cases),
            linear_matrices=len(originals), fp16_linear_bytes=fp16_bytes,
            largest_expanded_matrix_bytes=largest_matrix, provider_calls=0,
            production_changes=False, quantization=False, per_forward_host_weight_transfers=0,
            limitations='Three saved batches. Peak is PyTorch-tracked with nvCOMP array allocation routed to PyTorch; native codec allocations may be separate. Device-free snapshots also reported. No BGE/Llama co-residency test.'))
        emit(phase='complete')
    finally:
        linker = encoder = None
        originals.clear()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=Path('.cache/experiments/nvcomp-5.3.0.16/package'))
    args = parser.parse_args()
    run(args.output, args.package)
