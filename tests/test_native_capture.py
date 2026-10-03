"""Native IO is durable and learnable without claiming legacy search completion."""
from datetime import datetime, timezone

import pytest

from memory_condense.application.condenser import MemoryCondenser
from tests.test_native_spine_routing import Encoder


def encoder():
    value = Encoder()
    value.dim = 2
    def forbidden(*args, **kwargs):
        pytest.fail('Native capture must not embed raw IO')
    value.embed_chunks = forbidden
    return value


def record(tid, text='The deployment tag is orchid-927.'):
    return ('user', text, 'native', datetime(2026, 10, 3, tzinfo=timezone.utc), tid)


def test_capture_is_atomic_idempotent_and_preserves_unembedded_learning_nodes(tmp_path):
    with MemoryCondenser(tmp_path, embedder=encoder(), auto_extract=False) as app:
        first = app.capture_native_many([record('u1'), record('u2')])
        assert app.capture_native_many([record('u1')]) == first[:1]
        with pytest.raises(ValueError, match='different content'):
            app.capture_native_many([record('u3'), record('u1', 'conflicting')])
        assert app.transcript.get_turn('u3') is None
        assert app.pending_ingest_count() == 0
        assert app._db.execute('SELECT COUNT(*) FROM pending_ingests').fetchone()[0] == 0
        assert app._db.execute('SELECT COUNT(*) FROM chunk_terms').fetchone()[0] == 0
        chunks = app._db.execute('SELECT chunk_id,embedding,hnsw_label FROM chunks').fetchall()
        assert len(chunks) == 2 and all(vector is None and label is None for _,vector,label in chunks)
        # Capture alone is not publication and must not authorize learning.
        with pytest.raises(ValueError, match='active retrievable state'):
            app.observe_context_access([], [c[0] for c in chunks], access_event_id='native-learn')
        expected = app.transcript.get_all()
    with MemoryCondenser(tmp_path, embedder=encoder(), auto_extract=False) as app:
        assert app.transcript.get_all() == expected
        assert app.capture_native_many([record('u1')]) == first[:1]
        assert app._db.execute('SELECT COUNT(*) FROM consolidation_edges').fetchone()[0] == 0


def test_native_snapshot_reuses_appends_but_detects_source_edits(tmp_path, monkeypatch):
    with MemoryCondenser(tmp_path, embedder=encoder(), auto_extract=False) as app:
        app.capture_native_many([record('u1')])
        old = app.transcript.native_snapshot()
        get_all = app.transcript.get_all
        def forbidden():
            pytest.fail('Warm capture must not reload the complete transcript')
        monkeypatch.setattr(app.transcript, 'get_all', forbidden)
        app.capture_native_many([record('u2')])
        assert app.transcript.native_snapshot()[0] is old[0]
        monkeypatch.setattr(app.transcript, 'get_all', get_all)
        app._db.execute("UPDATE turns SET text='edited' WHERE turn_id='u1'")
        app._db.commit()
        assert app.transcript.native_snapshot()[0].text == 'edited'


def test_native_retry_rejects_missing_learning_topology(tmp_path):
    with MemoryCondenser(tmp_path, embedder=encoder(), auto_extract=False) as app:
        app.capture_native_many([record('u1')])
        app._db.execute("DELETE FROM chunks WHERE turn_id='u1'")
        app._db.commit()
        with pytest.raises(ValueError, match='chunks differ'):
            app.capture_native_many([record('u1')])
