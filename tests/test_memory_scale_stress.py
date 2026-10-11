import numpy as np
import pytest
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.persistence.db import Database
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


@pytest.mark.parametrize('defer_checkpoint', [False, True])
def test_cached_topology_preserves_real_ingestion_and_rejects_changed_source(tmp_path, defer_checkpoint):
    class Encoder:
        dim = 1024
        def embed_chunks(self, chunks):
            return [c.model_copy(update={'embedding':np.ones(1024,dtype=np.float32).tolist()}) for c in chunks]

    records = [('user', 'First sentence. Another sentence with facts. '*100, 'source',
                datetime(2023,1,1,tzinfo=timezone.utc), 'turn-1'),
               ('assistant', 'Second turn with an exact answer.', 'source',
                datetime(2023,1,2,tzinfo=timezone.utc), 'turn-2')]
    source = tmp_path/'source'
    with MemoryCondenser(source, embedder=Encoder(), auto_extract=False) as app:
        app.ingest_many(records)
    encoder = CachedSourceEmbedding()
    encoder.add(source)
    target = tmp_path/'replayed'
    with MemoryCondenser(target, embedder=encoder, auto_extract=False) as app:
        app._chunker = encoder
        if defer_checkpoint:
            app._db.connection.execute('PRAGMA synchronous=NORMAL')
            app._db.connection.execute('PRAGMA wal_autocheckpoint=0')
        app.ingest_many(records)
        assert app.pending_ingest_count() == 0
        if defer_checkpoint:
            app._db.connection.execute('PRAGMA synchronous=FULL')
            assert app._db.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone() == (0,0,0)
            app._db.connection.execute('PRAGMA wal_autocheckpoint=1000')
    def rows(directory):
        with closing(sqlite3.connect(directory/'memory.db')) as db:
            return db.execute('SELECT chunk_id,turn_id,start_char,end_char,text,token_count,embedding,lexical_weights '
                              'FROM chunks ORDER BY chunk_id').fetchall()
    assert rows(source) == rows(target)
    assert encoder.chunk_cache_hits == 2
    assert encoder.hits == len(rows(source)) > 2
    with pytest.raises(ValueError, match='turn changed'):
        encoder.chunk_turn('turn-1', records[0][1]+' changed')
    with Database(source/'memory.db') as db:
        db.execute("UPDATE chunks SET text='tampered' WHERE chunk_id=(SELECT chunk_id FROM chunks LIMIT 1)")
        db.commit()
    with pytest.raises(ValueError, match='raw span'):
        CachedSourceEmbedding().add(source)
