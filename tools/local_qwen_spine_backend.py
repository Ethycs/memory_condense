"""Resident local Qwen generation with independent, summary-only batch rows."""
from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import sys
import time

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.spine_summary import SpineSummaryRequest
from tools.matched_eval.artifacts import read_sealed_json
from tools.probe_local_qwen_parent_summaries import DEVICE_MAP, verify_checkpoint


def job_messages(job, attempt):
    if type(job) is not SpineSummaryRequest or attempt not in (0, 1, 2):
        raise ValueError('local generation requires a typed summary job and bounded attempt')
    messages = [dict(m) for m in job.messages]
    if attempt:
        words = (48, 24)[attempt - 1]
        messages[0]['content'] += (
            f' Merge ALL supplied fragments into ONE summary of at most {words} words. '
            'Return exactly one JSON object with only the key summary. '
            'Do not return a list or one output per fragment.')
    return messages


def decode_rows(tokenizer, output, prompt_width, eos_ids):
    """Discard batch padding after the first EOS; never accept a token-cap exit."""
    rows = []
    for sequence in output:
        generated = [int(v) for v in sequence[prompt_width:]]
        stop = next((i for i, token in enumerate(generated) if token in eos_ids), None)
        tokens = generated if stop is None else generated[:stop + 1]
        rows.append({'response': tokenizer.decode(tokens, skip_special_tokens=True),
                     'output_tokens': len(tokens), 'stopped': stop is not None})
    return rows


class LocalQwenBackend:
    max_batch_size = 2

    def __init__(self, probe_root, dependency_root, model_root):
        self.dependency_root, self.model_root = Path(dependency_root), Path(model_root)
        sys.path.insert(0, str(self.dependency_root.resolve()))
        probe = read_sealed_json(Path(probe_root)/'preflight.json')
        result = read_sealed_json(Path(probe_root)/'result.json')
        p, r = probe.payload, result.payload
        if (r['preflight_sha256'] != probe.sha256 or not r['local_generation_worked']
            or r['raw_inputs_to_qwen'] is not False or r['remote_provider_calls'] != 0
            or p['implementation_sha256'] != hashlib.sha256(
                Path('tools/probe_local_qwen_parent_summaries.py').read_bytes()).hexdigest()
            or str(self.model_root.resolve()) != p['model_root']):
            raise ValueError('local backend requires the completed bound fit probe')
        if {name:metadata.version(name) for name in p['versions']} != p['versions']:
            raise ValueError('local runtime versions changed since the fit probe')
        if hashlib.sha256((self.model_root/'generation_config.json').read_bytes()).hexdigest() != p['generation_config_sha256']:
            raise ValueError('local generation configuration changed')
        self.identity = {key:p[key] for key in (
            'model', 'revision', 'quantization', 'compute_dtype', 'non_quantized_dtype_requested',
            'device_map', 'attention_implementation', 'embedding_execution_device',
            'embedding_outputs_return_to_input_device', 'generation_config_sha256',
            'max_new_tokens', 'do_sample', 'thinking_enabled', 'local_files_only', 'versions')}
        self.identity.update({'checkpoint_sha256':r['checkpoint_sha256'],
            'fit_probe_result_sha256':result.sha256, 'max_batch_size':self.max_batch_size,
            'batch_policy':'independent left-padded sequences', 'max_input_tokens':2048,
            'raw_inputs_to_qwen':False, 'query_attention_changed':False})
        self.identity_sha256 = identity_sha256(self.identity)
        self.model = self.tokenizer = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if verify_checkpoint(self.model_root) != self.identity['checkpoint_sha256']:
            raise ValueError('full checkpoint identity changed')
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_root, local_files_only=True,
                                                       padding_side='left')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_root, local_files_only=True,
            quantization_config=config, device_map=DEVICE_MAP, torch_dtype=torch.float16,
            attn_implementation='sdpa', low_cpu_mem_usage=True).eval()
        embedding = self.model.get_input_embeddings()
        hook = embedding._hf_hook
        if not hook.offload or not hasattr(hook, 'weights_map'):
            raise ValueError('expected pinned Accelerate embedding offload hook')
        hook.execution_device, hook.io_same_device = 'cpu', True
        forward = embedding._old_forward
        def checked_forward(*args, **kwargs):
            if embedding.weight.device.type != 'cpu':
                raise ValueError('embedding weights unexpectedly moved to GPU')
            return forward(*args, **kwargs)
        embedding._old_forward = checked_forward
        print({'local_qwen_loaded':True, 'cold_load_s':time.perf_counter()-started,
               'gpu_allocated_GiB':torch.cuda.memory_allocated()/2**30}, flush=True)

    def generate(self, jobs, attempt):
        jobs = tuple(jobs)
        if not 1 <= len(jobs) <= self.max_batch_size:
            raise ValueError('local batch exceeds the two-row policy')
        messages = [job_messages(job, attempt) for job in jobs]
        self.load()
        import torch
        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False,
            add_generation_prompt=True, enable_thinking=False) for m in messages]
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')
        width = inputs['input_ids'].shape[1]
        if width > self.identity['max_input_tokens']:
            raise ValueError('summary job exceeds the local model input budget')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=256, do_sample=False,
                temperature=None, top_p=None, top_k=None, use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        eos = self.model.generation_config.eos_token_id
        rows = decode_rows(self.tokenizer, output, width, {eos} if isinstance(eos,int) else set(eos))
        for row, job, tokens in zip(rows, jobs, inputs['attention_mask'].sum(dim=1), strict=True):
            row.update({'job_sha256':job.prompt_sha256, 'input_tokens':int(tokens)})
        return {'backend_sha256':self.identity_sha256, 'rows':rows, 'elapsed_s':elapsed,
                'tokens_per_second':sum(r['output_tokens'] for r in rows)/elapsed,
                'peak_gpu_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
                'raw_inputs_to_qwen':False, 'remote_provider_calls':0}
