"""Local runtime preserves model identity and rejects invented routing extracts."""
import json
from types import SimpleNamespace

import pytest

from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.summary_semantic_index import compatible_summary_embedding, summary_embedding_identity
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest, parse_spine_summary
from tools.engineering_research_local import exact_prefix, raw_extracts, merge_extracts


def test_cpu_cuda_compatibility_requires_opt_in_and_same_pinned_fp32_model():
    cpu, gpu = EmbeddingService(device='cpu',batch_size=8), EmbeddingService(device='cuda',batch_size=8)
    stored = summary_embedding_identity(gpu)
    assert not compatible_summary_embedding(cpu,stored)
    cpu.allow_fp32_device_compatibility = True
    assert compatible_summary_embedding(cpu,stored)
    assert cpu.execution_identity['device']=='cpu'
    for field,value in [('model_revision','different'),('checkpoint_sha256','0'*64),
                         ('model_id','different')]:
        changed=json.loads(stored)
        changed[field]=value
        assert not compatible_summary_embedding(cpu,json.dumps(changed))
    for field,value in [('output_dtype','float16'),('normalize_embeddings',True),
                         ('batch_size',32),('device','mps'),('backend','other')]:
        changed=json.loads(stored)
        changed['execution'][field]=value
        assert not compatible_summary_embedding(cpu,json.dumps(changed))


def test_raw_model_cannot_move_a_fact_between_fragments():
    fragments=[dict(label='T0',fragment='Arden uses saffron-842, with three checks.'),
               dict(label='T1',fragment='Larch uses juniper-641, still a plan.')]
    proposed=json.dumps(dict(atoms=[dict(label='T0',summary='Larch uses juniper-641'),
                                   dict(label='T1',summary='Larch uses juniper-641')]))
    result,fallbacks=raw_extracts(fragments,proposed)
    assert fallbacks==1
    for item,fragment in zip(result['atoms'],fragments):
        assert item['summary'] in fragment['fragment']
        assert all(q in fragment['fragment'] for q in item['support'])


def test_export_adapter_keeps_its_actual_identity_while_admitting_the_pinned_source():
    source=EmbeddingService(device='cpu',batch_size=8)
    export=SimpleNamespace(model_name=source.model_name,model_revision='onnx-export',
        checkpoint_sha256='export-hash',execution_identity={'backend':'fastembed','device':'cpu'},
        allow_fp32_device_compatibility=True,
        validated_source_embedding_identity=summary_embedding_identity(source))
    actual=summary_embedding_identity(export)
    stored=summary_embedding_identity(EmbeddingService(device='cuda',batch_size=8))
    assert compatible_summary_embedding(export,stored)
    assert summary_embedding_identity(export)==actual and actual!=stored
    assert compatible_summary_embedding(export,actual)
    export.validated_source_embedding_identity=stored.replace('float32','float16')
    assert not compatible_summary_embedding(export,stored)


def test_local_benchmark_recall_preserves_streaming_publication_contract(monkeypatch):
    import threading
    from tools.evaluate_chat_io_local100 import QuestionBackend, historical
    backend=QuestionBackend.__new__(QuestionBackend)
    backend.questions={'question':{'question_id':'q1'}}
    backend.plan={'policy':{},'reader':{}}
    backend.timings=[]
    backend._publication_lock=threading.Lock()
    backend._published=(SimpleNamespace(),('event1','event2'),{'snapshot':{'turn_count':2}}, {})
    monkeypatch.setattr(historical.old.policy_tool,'build',lambda *args:([],{'sections':[]},
        {'raw_reads_during_routing':0,'query_qwen_passes':0},{'placements':[],'text':'packet'}))
    monkeypatch.setattr(historical.old.current,'validate_reader_policy',lambda value:value)
    monkeypatch.setattr(historical.old.current.reader,'apply_reader',lambda messages,policy:messages)
    result=backend.recall_published('question')
    assert result=={'text':'','references':[],'published_events':2,'snapshot':{'turn_count':2}}


@pytest.mark.parametrize('envelope',['object','list','fenced'])
def test_local_extract_formats_still_validate_source_ownership(envelope):
    rows=[{'label':'T0','summary':'Exact source fact.'}]
    content=json.dumps({'atoms':rows} if envelope=='object' else rows)
    if envelope=='fenced': content='```json\n'+content+'\n```'
    value,fallback=raw_extracts([{'label':'T0','fragment':'Exact source fact. More text.'}],content)
    assert fallback==0 and value['atoms'][0]['summary']=='Exact source fact.'


def test_merge_never_reattributes_an_extracted_fact():
    request=SpineSummaryRequest('attached_context',(
        SpineSummaryFragment('assistant','2026-09-30','I propose migration orchid; not executed.'),
        SpineSummaryFragment('system','2026-09-30','The tool reports three checks passed.')),
        user_spine='User requested a plan.',max_output_tokens=64)
    proposed=json.dumps(dict(extracts=[dict(index=0,quote='The tool reports three checks passed.'),
                                     dict(index=1,quote='three checks passed.')]))
    value,fallbacks=merge_extracts(request,proposed)
    assert fallbacks==1
    assert '[assistant] I propose migration' in value['summary']
    assert '[system] three checks passed.' in value['summary']
    parse_spine_summary(json.dumps(value),request)


@pytest.mark.parametrize('limit',[1,4,16,64])
def test_exact_fallback_is_bounded_without_unicode_replacement(limit):
    from memory_condense.domain._tokenizer import count_tokens
    text='计划部署；未执行。 Cluster juniper-641. 🐈'*20
    prefix=exact_prefix(text,limit)
    assert prefix and prefix in text and count_tokens(prefix)<=limit
