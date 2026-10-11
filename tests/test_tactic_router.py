import json
from pathlib import Path

import numpy as np
import pytest

from memory_condense.application.tactic_router import (
    BGETacticRouter, DEFAULT_MODEL, FORMAT, LABELS, render_input,
)
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.summary_semantic_index import summary_embedding_identity


class Encoder:
    model_name = 'test-bge'
    model_revision = 'test'
    checkpoint_sha256 = 'a'*64
    execution_identity = {'device':'cpu'}

    def __init__(self,vector=(1.,0.)):
        self.vector = vector
        self.calls = []

    def embed_query(self,text):
        self.calls.append(text)
        return np.asarray(self.vector)


def model(encoder,**changes):
    value = dict(format=FORMAT,labels=LABELS,embedding_identity=summary_embedding_identity(encoder),
        weights=[[4.,0.],[-4.,0.],[0.,4.]],intercept=[0.,0.,0.],dimension=2,
        min_score=.6,min_margin=.15,input_format='bounded-recent-v1',
        max_query_tokens=512,max_context_tokens=256)
    value.update(changes)
    value['receipt_sha256'] = identity_sha256(value)
    return value


@pytest.mark.parametrize('vector,tactic,scope',[
    ((1,0),'cap8','focused'),((-1,0),'hierarchy_neighborhoods','overview'),
    ((0,1),None,'uncertain'),
])
def test_one_shared_encoder_call_prioritizes_without_dispatch(vector,tactic,scope):
    encoder = Encoder(vector)
    router = BGETacticRouter(encoder,model(encoder))
    result = router.prioritize('Is this working?',recent=['The preceding user concern.'])
    assert result.preferred_tactic == tactic
    assert result.predicted_scope == scope
    assert result.abstained == (tactic is None)
    assert result.embedding_calls == len(encoder.calls) == 1
    assert encoder.calls == [render_input('Is this working?',['The preceding user concern.'])[0]]
    assert set(result.priorities) == {'cap8','hierarchy_neighborhoods'}
    assert result.contextualized and result.classifier_sha256 == router.receipt_sha256
    assert not router.weights.flags.writeable


def test_low_margin_abstains_and_does_not_manufacture_a_confident_default():
    encoder = Encoder((1,1))
    result = BGETacticRouter(encoder,model(encoder)).prioritize('Question')
    assert result.abstained and result.preferred_tactic is None
    assert result.margin == 0 and result.reason in ('low_score','low_margin')


def test_vector_reuse_requires_exact_context_and_encoder_binding():
    encoder = Encoder()
    router = BGETacticRouter(encoder,model(encoder))
    bound = dict(vector=[1.,0.],text_sha256=quote_sha256('Question'),
        embedding_identity=summary_embedding_identity(encoder))
    assert router.prioritize('Question',input_vector=bound).embedding_calls == 0
    assert not encoder.calls
    with pytest.raises(ValueError,match='complete classifier input'):
        router.prioritize('Question',recent=['Different meaning'],input_vector=bound)
    with pytest.raises(ValueError,match='complete classifier input'):
        router.prioritize('Question',input_vector={**bound,'embedding_identity':'changed'})
    for vector in ([0.,0.],[float('nan'),0.],[1.,0.,0.]):
        with pytest.raises(ValueError,match='vector'):
            router.prioritize('Question',input_vector={**bound,'vector':vector})
    assert not encoder.calls


def test_encoder_drift_and_artifact_corruption_fail_before_inference():
    encoder = Encoder()
    value = model(encoder)
    corrupted = dict(value,min_score=.1)
    with pytest.raises(ValueError,match='identity changed'):
        BGETacticRouter(encoder,corrupted)
    router = BGETacticRouter(encoder,value)
    encoder.execution_identity = {'device':'other'}
    with pytest.raises(ValueError,match='identity changed'):
        router.prioritize('Question')
    with pytest.raises(ValueError,match='differs from training'):
        BGETacticRouter(encoder,value)
    assert not encoder.calls


def test_context_is_bounded_but_question_is_not_silently_truncated():
    recent = ['discard-this-old-message']+['old '*300]*5+['Most recent clarification.']
    text, contextualized = render_input('Exact current question',recent)
    assert contextualized and 'discard-this-old-message' not in text
    assert 'Most recent clarification.' in text
    assert text.endswith('Current request:\nExact current question')
    context = text.split('Recent conversation:\n')[1].split('\n\nCurrent request:')[0]
    assert count_tokens(context) <= 256
    with pytest.raises(ValueError,match='budget'):
        render_input('word '*1000)
    with pytest.raises(TypeError,match='sequence'):
        render_input('Question','Not a sequence of messages')
    with pytest.raises(ValueError,match='nonempty'):
        render_input(' ')


def test_packaged_model_loads_without_training_dependencies_or_a_model_download():
    value = json.loads(DEFAULT_MODEL.read_text(encoding='utf-8'))
    identity = json.loads(value['embedding_identity'])
    encoder = Encoder(np.ones(value['dimension']))
    encoder.model_name = identity['model_id']
    encoder.model_revision = identity['model_revision']
    encoder.checkpoint_sha256 = identity['checkpoint_sha256']
    encoder.execution_identity = identity['execution']
    router = BGETacticRouter(encoder)
    assert router.prioritize('A bounded classifier smoke query').embedding_calls == 1
    assert value['training']['test_used_for_selection'] is False
    assert router.weights.shape == (3,1024)


def test_jsonl_command_keeps_one_encoder_for_multiple_requests(tmp_path,monkeypatch,capsys):
    import io
    import sys
    from memory_condense.cli import main
    from memory_condense.runtime import embedding

    encoder = Encoder()
    closed = []
    encoder.close = lambda:closed.append(True)
    constructed = []
    def create(**kwargs):
        constructed.append(kwargs)
        return encoder
    monkeypatch.setattr(embedding,'FastEmbedBGE',create)
    path = tmp_path/'head.json'
    path.write_text(json.dumps(model(encoder)),encoding='utf-8')
    monkeypatch.setattr(sys,'stdin',io.StringIO(
        '{"query":"First question"}\n'
        '{"query":"Follow-up","recent":["Prior context"]}\n'
        '{"query":"Bad request","recent":"wrong type"}\n'))
    assert main(['route','--jsonl','--classifier',str(path),'--assets-dir',str(tmp_path)]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(constructed) == 1 and len(encoder.calls) == 2 and closed == [True]
    assert lines[0]['preferred_tactic'] == lines[1]['preferred_tactic'] == 'cap8'
    assert lines[1]['contextualized'] and 'error' in lines[2]


def test_runtime_prioritizer_reuses_existing_encoder_without_starting_gpu_models(tmp_path,monkeypatch):
    from memory_condense.runtime.local import LocalRuntime
    value = json.loads(DEFAULT_MODEL.read_text(encoding='utf-8'))
    identity = json.loads(value['embedding_identity'])
    encoder = Encoder(np.ones(1024))
    encoder.model_name,encoder.model_revision = identity['model_id'],identity['model_revision']
    encoder.checkpoint_sha256,encoder.execution_identity = identity['checkpoint_sha256'],identity['execution']
    runtime = LocalRuntime(tmp_path/'runtime')
    embedding_accesses = []
    def existing():
        embedding_accesses.append(True)
        return encoder
    monkeypatch.setattr(runtime,'embedding',existing)
    first = runtime.tactic_router()
    assert runtime.tactic_router() is first
    first.prioritize('Which memory tactic fits?')
    assert len(embedding_accesses) == 1 and len(encoder.calls) == 1
    assert runtime.scorer is runtime.client is runtime.process is None
