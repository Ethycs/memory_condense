"""Compare FP32 BGE-M3 FastEmbed CPU inference with the sealed PyTorch replay."""
from pathlib import Path
from collections import Counter
import argparse
import gc
import hashlib
import json
import sys
import time

import numpy as np

from tools.engineering_research_gateway import read, save, emit
from tools.probe_bge_cpu_placement import fidelity, stats

PACKAGE=Path('.cache/experiments/fastembed/package')
MODEL=Path('.cache/models/bge-m3-onnx-fp32')
NAME='local/bge-m3-fp32-pinned'


def seal_admission(root):
    plan,result=read(root/'plan.json'),read(root/'report.json')
    names=['onnx/model.onnx','onnx/model.onnx_data','tokenizer.json','tokenizer_config.json',
           'config.json','special_tokens_map.json']
    files={}
    for name in names:
        with (MODEL/name).open('rb') as handle:
            files[name]=hashlib.file_digest(handle,'sha256').hexdigest()
    if not result['faithful'] or any(files[k]!=v for k,v in plan['model']['files'].items()):
        raise ValueError('Failed FastEmbed admission')
    save(root/'admission.json',dict(report_sha256=hashlib.sha256((root/'report.json').read_bytes()).hexdigest(),
        files=files,model_revision=plan['model']['revision'],files_frozen_after_component_assay=True))


def load(threads):
    sys.path.insert(0,str(PACKAGE.resolve()))
    from fastembed import TextEmbedding
    from fastembed.common.model_description import PoolingType, ModelSource
    if not any(m['model']==NAME for m in TextEmbedding.list_supported_models()):
        TextEmbedding.add_custom_model(model=NAME,pooling=PoolingType.CLS,normalization=True,
            sources=ModelSource(hf='onnx-community/bge-m3-ONNX'),dim=1024,
            model_file='onnx/model.onnx',additional_files=['onnx/model.onnx_data'])
    return TextEmbedding(NAME,threads=threads,providers=['CPUExecutionProvider'],
        specific_model_path=str(MODEL.resolve()),local_files_only=True)


def run(root):
    sys.path.insert(0,str(PACKAGE.resolve()))
    import fastembed, onnx, onnxruntime, psutil
    root.mkdir(parents=True,exist_ok=False)
    source=Path('eval_results/bge-cpu-placement-20260930-r2')
    prior=read(source/'plan.json')
    reference_meta=read(source/'cpu_fp32_mkl12.json')
    reference_path=source/'cpu_fp32_mkl12-vectors.npz'
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest()==reference_meta['vectors_sha256']
    with np.load(reference_path) as stored:
        reference={k:stored[k].copy() for k in stored}
    model_source=json.loads((MODEL/'source.json').read_text())
    for name,expected in model_source['files'].items():
        with (MODEL/name).open('rb') as handle:
            assert hashlib.file_digest(handle,'sha256').hexdigest()==expected
    graph=onnx.load(MODEL/'onnx/model.onnx',load_external_data=False)
    types=dict(Counter(t.data_type for t in graph.graph.initializer))
    assert types.get(1) and set(types)<={1,6,7,9},types
    assert not any('Quantiz' in node.op_type or node.op_type.startswith('QLinear') for node in graph.graph.node)
    save(root/'plan.json',dict(source=str(source),model=model_source,initializer_types=types,
        fastembed=fastembed.__version__,onnxruntime=onnxruntime.__version__,
        normalization=True,pooling='CLS',precision='FP32',provider='CPUExecutionProvider',
        query_repetitions=2,batch_repetitions=2,thread_candidates=[4,8,12],
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        provider_calls=0,history_reingestions=0))
    groups=dict(query=[[q] for q in prior['queries']],summary=prior['summary_batches'],
                chunk=[[c['text'] for c in b] for b in prior['chunk_batches']])
    pilots=[]
    for threads in (4,8,12):
        model=load(threads)
        list(model.embed(groups['query'][0]))
        durations=[]
        for texts in (groups['query'][1],groups['query'][5],groups['query'][9],groups['query'][13],groups['summary'][1]):
            started=time.perf_counter()
            list(model.embed(texts,batch_size=8))
            durations.append(time.perf_counter()-started)
        pilot=dict(threads=threads,query_mean_s=sum(durations[:4])/4,summary_batch8_s=durations[4],
                   selection_score_s=sum(durations))
        pilots.append(pilot)
        save(root/f'pilot-{threads}.json',pilot)
        emit(phase='pilot',**pilot)
        del model
        gc.collect()
    selected=min(pilots,key=lambda p:p['selection_score_s'])['threads']
    model=load(selected)
    outputs,results={},{}
    for kind,batches in groups.items():
        list(model.embed(batches[0],batch_size=8))
        durations,values=[],[]
        for repetition in range(2):
            for texts in batches:
                started=time.perf_counter()
                vectors=np.asarray(list(model.embed(texts,batch_size=8)),dtype=np.float32)
                durations.append(time.perf_counter()-started)
                if not repetition: values.append(vectors)
        outputs[kind]=np.concatenate(values)
        results[kind]=dict(**stats(durations),fidelity=fidelity(reference[kind],outputs[kind]),
            pytorch_mean_s=reference_meta['workloads'][kind]['mean_s'])
        emit(phase='measured',kind=kind,**results[kind])
    micro=[]
    for text in groups['chunk'][0][:4]:
        started=time.perf_counter()
        list(model.embed([text],batch_size=1))
        micro.append(time.perf_counter()-started)
    np.savez(root/'vectors.npz',**outputs)
    save(root/'report.json',dict(results=results,pilots=pilots,selected_threads=selected,
        chunk_single=stats(micro),process_rss_bytes=psutil.Process().memory_info().rss,
        vector_sha256=hashlib.sha256((root/'vectors.npz').read_bytes()).hexdigest(),
        tokenizer_max_length=model.model.tokenizer.truncation,
        faithful=all(r['fidelity']['cosine_min']>0.99999999 for r in results.values()),
        provider_calls=0,history_reingestions=0))
    emit(phase='complete',selected_threads=selected)
    seal_admission(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
