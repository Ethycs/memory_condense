"""Background embedding yields to queries and reuses text without source aliases."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest

from memory_condense.domain.schemas import Chunk
from tools.engineering_research_resident import SharedEmbedding
from tools.evaluate_chat_io_batch12 import answer_json


def encoder():
    return SimpleNamespace(model_name='fixture',model_revision='v1',checkpoint_sha256='test',execution_identity={})


def chunks(count):
    return [Chunk(chunk_id=f'c{i}',turn_id=f't{i}',text=f'word{i}',start_char=0,end_char=len(f'word{i}'),token_count=1)
            for i in range(count)]


def test_query_runs_before_next_background_chunk_batch():
    base=encoder()
    entered, query_arrived=Event(),Event()
    order=[]
    def embed(values):
        order.append('batch')
        if len(order)==1:
            entered.set()
            assert query_arrived.wait(5)
        return [c.model_copy(update={'embedding':[1.,2.]}) for c in values]
    base.embed_chunks=embed
    base.embed_query=lambda text:order.append('query') or np.asarray([1.,2.])
    shared=SharedEmbedding(base)
    def query():
        with shared._query_condition:
            shared._query_waiters+=1
            query_arrived.set()
        try:
            with shared.model_lock:
                return base.embed_query('question')
        finally:
            with shared._query_condition:
                shared._query_waiters-=1
                shared._query_condition.notify_all()
    with ThreadPoolExecutor(max_workers=2) as pool:
        background=pool.submit(shared.embed_chunks,chunks(16))
        assert entered.wait(5)
        reader=pool.submit(query)
        reader.result(timeout=5)
        assert len(background.result(timeout=5))==16
    assert order==['batch','query','batch']


def test_identical_text_reuses_vector_but_keeps_every_source_coordinate():
    base=encoder()
    calls=[]
    base.embed_chunks=lambda values:calls.extend(values) or [c.model_copy(update={'embedding':[1.,2.]}) for c in values]
    shared=SharedEmbedding(base)
    first=chunks(1)[0]
    second=first.model_copy(update={'chunk_id':'copy','turn_id':'other','start_char':10,'end_char':15})
    a,b=shared.embed_chunks([first,second])
    assert len(calls)==1 and a.turn_id!=b.turn_id and b.start_char==10
    a.embedding[0]=99
    assert shared.embed_chunks([second])[0].embedding==[1.,2.]
    base.model_revision='v2'
    shared.embed_chunks([second])
    assert len(calls)==2


def test_json_scoring_accepts_only_one_exact_fence_and_keeps_values():
    assert answer_json('```json\n{"status":"planned"}\n```')=={'status':'planned'}
    for value in ('Prose\n```json\n{}\n```','```json\n{}\n```\nextra','```json\n{}\n```\n```json\n{}\n```'):
        with pytest.raises(ValueError): answer_json(value)
