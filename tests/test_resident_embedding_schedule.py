"""Background embedding yields to queries and reuses text without source aliases."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest

from memory_condense.domain.schemas import Chunk
from tools.engineering_research_resident import RecallPriorityLock, SharedEmbedding
from tools.evaluate_chat_io_batch12 import answer_json


def encoder(device='cuda'):
    return SimpleNamespace(model_name='fixture',model_revision='v1',checkpoint_sha256='test',
                           execution_identity={'device':device})


def chunks(count):
    return [Chunk(chunk_id=f'c{i}',turn_id=f't{i}',text=f'word{i}',start_char=0,end_char=len(f'word{i}'),token_count=1)
            for i in range(count)]


def wait_for_queue(lock, *, foreground=0, background=0):
    # Observe real requests; do not synthesize query priority in the test.
    with lock._condition:
        assert lock._condition.wait_for(lambda: len(lock._foreground)==foreground
                                       and len(lock._background)==background, timeout=5)


@pytest.mark.parametrize('device,batch_size', [('cpu',1),('cuda',8)])
def test_query_runs_before_already_queued_background_jobs(device,batch_size):
    base=encoder(device)
    entered, release=Event(),Event()
    order=[]
    def embed(values):
        order.append('batch')
        if len(order)==1:
            entered.set()
            assert release.wait(5)
        return [c.model_copy(update={'embedding':[1.,2.]}) for c in values]
    base.embed_chunks=embed
    base.embed_query=lambda text:order.append('query') or np.asarray([1.,2.])
    shared=SharedEmbedding(base)
    population=chunks(batch_size*4)
    with ThreadPoolExecutor(max_workers=4) as pool:
        first=pool.submit(shared.embed_chunks,population[:batch_size*2])
        try:
            assert entered.wait(5)
            second=pool.submit(shared.embed_chunks,population[batch_size*2:batch_size*3])
            third=pool.submit(shared.embed_chunks,population[batch_size*3:])
            wait_for_queue(shared.model_lock,background=2)
            reader=pool.submit(shared.embed_query,'question')
            wait_for_queue(shared.model_lock,foreground=1,background=2)
        finally:
            release.set()
        assert reader.result(timeout=5).tolist()==[1.,2.]
        assert sum(len(f.result(timeout=5)) for f in (first,second,third))==len(population)
    assert order==['batch','query','batch','batch','batch']
    assert shared.metrics['query_calls']==1 and shared.metrics['chunk_batches']==4


def test_foreground_exception_releases_model_for_waiting_background():
    base=encoder()
    entered,release=Event(),Event()
    order=[]
    def fail(query):
        entered.set()
        assert release.wait(5)
        raise RuntimeError('encoder failure')
    base.embed_query=fail
    base.embed_queries=lambda values:order.append('summary') or np.asarray([[1.,2.] for _ in values])
    shared=SharedEmbedding(base)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reader=pool.submit(shared.embed_query,'question')
        try:
            assert entered.wait(5)
            background=pool.submit(shared.embed_queries,['summary'])
            wait_for_queue(shared.model_lock,background=1)
        finally:
            release.set()
        with pytest.raises(RuntimeError,match='encoder failure'):
            reader.result(timeout=5)
        assert background.result(timeout=5).tolist()==[[1.,2.]]
    assert order==['summary']


def test_nested_model_operation_finishes_when_recall_is_waiting():
    lock=RecallPriorityLock()
    entered,release=Event(),Event()
    order=[]
    def background():
        with lock:
            entered.set()
            assert release.wait(5)
            with lock:
                order.append('nested')
    def query():
        with lock.foreground(): order.append('query')
    with ThreadPoolExecutor(max_workers=2) as pool:
        worker=pool.submit(background)
        try:
            assert entered.wait(5)
            reader=pool.submit(query)
            wait_for_queue(lock,foreground=1)
        finally:
            release.set()
        worker.result(timeout=5)
        reader.result(timeout=5)
    assert order==['nested','query']


@pytest.mark.parametrize('device,batch_size', [('cpu',1),('cuda',8),('auto',1)])
def test_background_batches_preserve_summary_and_chunk_order(device,batch_size):
    base=encoder(device)
    calls=[]
    def embed(values):
        calls.append(len(values))
        return np.asarray([[int(t),2.] for t in values],dtype=np.float32)
    base.embed_queries=embed
    base.embed_chunks=lambda values:[c.model_copy(update={'embedding':v.tolist()})
        for c,v in zip(values,embed([c.text.removeprefix('word') for c in values]))]
    shared=SharedEmbedding(base)
    expected=[[float(i),2.] for i in range(10)]
    assert shared.embed_queries([str(i) for i in range(10)]).tolist()==expected
    assert [c.embedding for c in shared.embed_chunks(chunks(10))]==expected
    assert max(calls)==batch_size
    assert base.execution_identity=={'device':device}


@pytest.mark.parametrize('size', [0,-1,True,1.5])
def test_invalid_background_batch_size_rejected(size):
    with pytest.raises(ValueError,match='positive integer'):
        SharedEmbedding(encoder(),background_batch_size=size)


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
