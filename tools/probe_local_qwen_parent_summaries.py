"""Test full local Qwen generation on synthetic and prepared summary-only jobs.

This offline ingest probe does not change query-time attention precision or
claim reader accuracy. Dependencies live in an isolated workspace directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
from pathlib import Path
import sys
import time

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.modeling.qwen_prefix import (
    DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, QWEN3_8B_FILE_SHA256,
)
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest, parse_spine_summary
from tools.build_spine_corpus_hierarchy import restore_request
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


FIFTH_SHARD_SHA256 = '20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff'
DEVICE_MAP = {'model.embed_tokens':'cpu','model.layers':0,'model.norm':0,'lm_head':0}


def verify_checkpoint(model_root):
    hashes = {**QWEN3_8B_FILE_SHA256, 'model-00005-of-00005.safetensors':FIFTH_SHARD_SHA256}
    for name,expected in hashes.items():
        with (model_root/name).open('rb') as handle:
            observed = hashlib.file_digest(handle,'sha256').hexdigest()
        if observed != expected:
            raise ValueError(f'full Qwen checkpoint file changed: {name}')
        print({'verified_checkpoint_file':name},flush=True)
    return identity_sha256({'model':DEFAULT_MODEL_ID,'revision':DEFAULT_MODEL_REVISION,'files':hashes})


def run(root,dependency_root,model_root,prepared_request):
    sys.path.insert(0,str(dependency_root.resolve()))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    job = SpineSummaryRequest('user_spine',(
        SpineSummaryFragment('user_summary','2026-09-10','User prefers quiet hiking trails.'),
        SpineSummaryFragment('user_summary','2026-09-10','User plans short weekend hikes.')),
        max_output_tokens=64)
    jobs=[job]
    binding=None
    if prepared_request is not None:
        request=read_sealed_json(prepared_request)
        if request.payload.get('raw_inputs') is not False:
            raise ValueError('probe accepts only frozen summary-only merge requests')
        jobs.append(restore_request(request.payload['jobs'][0]))
        binding=request.sha256
    preflight,_=publish_sealed_json(root/'preflight.json',{
        'format':'local-qwen-summary-generation-probe-v1','model':DEFAULT_MODEL_ID,
        'revision':DEFAULT_MODEL_REVISION,'model_root':str(model_root.resolve()),
        'quantization':'NF4 with double quantization','compute_dtype':'float16',
        'non_quantized_dtype_requested':'float16','device_map':DEVICE_MAP,'attention_implementation':'sdpa',
        'embedding_execution_device':'cpu','embedding_outputs_return_to_input_device':True,
        'generation_config_sha256':hashlib.sha256((model_root/'generation_config.json').read_bytes()).hexdigest(),
        'max_new_tokens':256,'do_sample':False,'thinking_enabled':False,'local_files_only':True,
        'prepared_request_sha256':binding,'jobs':[asdict(j) for j in jobs],
        'raw_inputs_to_qwen':False,'query_attention_changed':False,'accuracy_measured':False,
        'versions':{name:metadata.version(name) for name in ('torch','transformers','accelerate','bitsandbytes')},
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    with (root/'execution.reserved').open('x',encoding='utf-8') as handle:
        handle.write(preflight.sha256+'\n')
    try:
        checkpoint_sha=verify_checkpoint(model_root)
        config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True)
        print('Loading full local Qwen with CPU token embeddings and quantized GPU layers...',flush=True)
        started=time.perf_counter()
        tokenizer=AutoTokenizer.from_pretrained(model_root,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(model_root,local_files_only=True,
            quantization_config=config,device_map=DEVICE_MAP,torch_dtype=torch.float16,
            attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
        # Accelerate normally stages CPU-offloaded weights on the main GPU.
        # Execute this lookup on CPU and transfer only its small output tensor.
        embedding=model.get_input_embeddings()
        hook=embedding._hf_hook
        if not hook.offload or not hasattr(hook,'weights_map'):
            raise ValueError('expected the pinned Accelerate embedding offload hook')
        hook.execution_device='cpu'
        hook.io_same_device=True
        embedding_dtype=str(hook.weights_map['weight'].dtype)
        original_forward=embedding._old_forward
        embedding_calls=[]
        def checked_embedding(*args,**kwargs):
            if embedding.weight.device.type!='cpu':
                raise ValueError('embedding weights unexpectedly moved to GPU')
            embedding_calls.append(True)
            return original_forward(*args,**kwargs)
        embedding._old_forward=checked_embedding
        cold=time.perf_counter()-started
        print({'cold_load_s':cold,'gpu_allocated_GiB':torch.cuda.memory_allocated()/2**30,
               'device_map':model.hf_device_map,'embedding_storage_dtype':embedding_dtype},flush=True)
        results=[]
        for ordinal,request in enumerate(jobs):
            prompt=tokenizer.apply_chat_template(request.messages,tokenize=False,
                add_generation_prompt=True,enable_thinking=False)
            inputs=tokenizer(prompt,return_tensors='pt').to('cuda')
            if inputs['input_ids'].shape[1] > 2048:
                raise ValueError('single summary job exceeds the local probe input budget')
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=256,do_sample=False,
                    temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            tokens=output[0,inputs['input_ids'].shape[1]:]
            text=tokenizer.decode(tokens,skip_special_tokens=True)
            eos=model.generation_config.eos_token_id
            eos={eos} if isinstance(eos,int) else set(eos)
            stopped=int(tokens[-1]) in eos
            try:
                summary=parse_spine_summary(text,request) if stopped else None
                valid=summary is not None
                error=None if valid else 'generation did not reach EOS'
            except ValueError as exc:
                summary,valid,error=None,False,str(exc)
            row={'ordinal':ordinal,'request_sha256':request.prompt_sha256,'response':text,
                'response_sha256':quote_sha256(text),'summary':summary,'valid':valid,'validation_error':error,
                'input_tokens':int(inputs['input_ids'].shape[1]),'output_tokens':int(len(tokens)),
                'elapsed_s':elapsed,'tokens_per_second':len(tokens)/elapsed,'stopped':stopped,
                'peak_gpu_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
            artifact,_=publish_sealed_json(root/f'response-{ordinal:02}.json',row)
            results.append({'response_sha256':artifact.sha256,**{k:v for k,v in row.items() if k not in ('response','summary')}})
            print(results[-1],flush=True)
        result,_=publish_sealed_json(root/'result.json',{'preflight_sha256':preflight.sha256,
            'checkpoint_sha256':checkpoint_sha,'cold_load_s':cold,'rows':results,
            'embedding_storage_dtype':embedding_dtype,'verified_cpu_embedding_calls':len(embedding_calls),
            'local_generation_worked':all(r['valid'] for r in results),'raw_inputs_to_qwen':False,
            'remote_provider_calls':0,'query_attention_changed':False,'target_gate_passed':False})
        print({'result_sha256':result.sha256,'local_generation_worked':result.payload['local_generation_worked']},flush=True)
    except Exception as error:
        failure,_=publish_sealed_json(root/'failure.json',{'preflight_sha256':preflight.sha256,
            'exception_type':type(error).__name__,'message':str(error),'retry_performed':False})
        print({'failure_sha256':failure.sha256,'exception_type':type(error).__name__},flush=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--dependency-root',type=Path,default=Path('.cache/local-qwen-runtime/site-packages'))
    parser.add_argument('--model-root',type=Path,default=Path('../../.cache/models/Qwen3-8B'))
    parser.add_argument('--prepared-request',type=Path)
    args=parser.parse_args()
    run(args.output_root,args.dependency_root,args.model_root,args.prepared_request)
