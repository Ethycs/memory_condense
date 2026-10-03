import numpy as np
import pytest

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain.schemas import Chunk
from tools.run_memory_scale_stress import CachedSourceEmbedding, gate


def test_scaled_gates_use_rates_instead_of_a_100_question_threshold():
    report=dict(question_count=1000,invalid_grades=20,support_complete=900,correct=750,
                inline_memory={'accepted':800})
    assert gate(report) is None
    assert 'accuracy' in gate(dict(report,correct=749))
    assert 'coverage' in gate(dict(report,support_complete=899))
    assert 'Invalid' in gate(dict(report,invalid_grades=21))


def test_cached_ingestion_never_supplies_an_embedding_for_changed_raw_text():
    encoder=CachedSourceEmbedding()
    chunk=Chunk(chunk_id='c',turn_id='t',text='original',start_char=0,end_char=8,token_count=1)
    encoder.chunks[('t',0,8,quote_sha256('original'))]=(np.ones(1024,dtype=np.float32),None)
    assert encoder.embed_chunks([chunk])[0].embedding==[1.0]*1024
    with pytest.raises(KeyError):
        encoder.embed_chunks([chunk.model_copy(update={'text':'modified'})])
    with pytest.raises(RuntimeError):
        encoder.embed_query('must not use a source vector as a query')
