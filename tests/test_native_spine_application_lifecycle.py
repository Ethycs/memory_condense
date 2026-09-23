from contextlib import closing
import sqlite3

import pytest

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from memory_condense.persistence.db import CURRENT_SCHEMA_VERSION
from memory_condense.search.section_routing import SectionSummaryIndex
from tests.test_native_spine_context_routing import parent_fixture
from tests.test_native_spine_routing import QUERY, DATED
from tools.native_spine_context_policy import DENSE_PARENT_2048


def inputs():
    history, semantic, hierarchy, encoder = parent_fixture(future=True)
    encoder.dim = 2
    encoder.ingest_calls = []
    def embed_chunks(chunks):
        encoder.ingest_calls.append([c.text for c in chunks])
        return [c.model_copy(update={'embedding': [1.0, 0.0]}) for c in chunks]
    encoder.embed_chunks = embed_chunks
    matrix = semantic._dense._matrix.copy()
    records = [(t.role, t.text, t.source_id, t.created_at, t.turn_id) for t in history.turns.values()]
    return history, semantic, hierarchy, encoder, matrix, records


def application(path, encoder, **kwargs):
    return MemoryCondenser(path, embedder=encoder, auto_extract=False, **kwargs)


def install(app, semantic, hierarchy, matrix):
    return app.install_native_spine(semantic.hierarchy, hierarchy, matrix,
                                   embedding_identity=semantic.embedding_identity)


def test_normal_ingest_persist_close_reopen_hydrates_identical_raw_from_application(tmp_path):
    history, semantic, hierarchy, encoder, matrix, records = inputs()
    expected = ResidentNativeSpineContextMemory(semantic, hierarchy, encoder=encoder,
        load_turn=history.get_turn).retrieve(QUERY, DATED, **DENSE_PARENT_2048)
    with application(tmp_path, encoder) as app:
        app.ingest_many(records)
        assert app.pending_ingest_count() == 0 and encoder.ingest_calls
        receipt = install(app, semantic, hierarchy, matrix)
    with application(tmp_path, encoder, read_only=True) as reopened:
        assert reopened.native_spine_receipt() == receipt
        reads = []
        original = reopened.transcript.get_turn
        def load(turn_id):
            reads.append(turn_id)
            return original(turn_id)
        reopened.transcript.get_turn = load
        # Force the resident loader to bind the observed application DB read.
        reopened._native_spine_loaded = None
        result = reopened.retrieve_native_spine(QUERY, DATED, **DENSE_PARENT_2048)
        assert result.routing.identity_payload() == expected.routing.identity_payload()
        assert result.hydration.identity_payload() == expected.hydration.identity_payload()
        assert reads and encoder.calls == [QUERY, QUERY]
        assert '2026-09-13' not in result.hydration.render_context()
        assert 'Stored attention parent' not in result.hydration.render_context()
    with pytest.raises(sqlite3.ProgrammingError):
        reopened.retrieve_native_spine(QUERY, DATED)


@pytest.mark.parametrize('defect', ['pending', 'missing_atom', 'missing_hierarchy', 'foreign_raw'])
def test_install_rejects_partial_or_mismatched_ingestion(tmp_path, defect):
    _, semantic, hierarchy, encoder, matrix, records = inputs()
    with application(tmp_path, encoder) as app:
        if defect == 'pending':
            app.capture_many(records)
        else:
            if defect == 'foreign_raw':
                records[0] = (records[0][0], records[0][1] + ' changed', *records[0][2:])
            app.ingest_many(records)
        if defect == 'missing_hierarchy':
            hierarchy = SectionSummaryIndex(())
        if defect == 'missing_atom':
            atomic = SectionSummaryIndex(semantic.sections[:-1])
            with pytest.raises(ValueError):
                app.install_native_spine(atomic, hierarchy, matrix[:-1],
                                         embedding_identity=semantic.embedding_identity)
        else:
            with pytest.raises(ValueError):
                install(app, semantic, hierarchy, matrix)
        assert not (tmp_path / 'native-spine.sqlite').exists()


@pytest.mark.parametrize('defect', ['vectors', 'raw', 'source', 'summary'])
def test_reopen_rejects_changed_persisted_data(tmp_path, defect):
    _, semantic, hierarchy, encoder, matrix, records = inputs()
    with application(tmp_path, encoder) as app:
        app.ingest_many(records)
        install(app, semantic, hierarchy, matrix)
    filename = 'memory.db' if defect in ('raw', 'source') else 'native-spine.sqlite'
    with closing(sqlite3.connect(tmp_path / filename)) as db, db:
        # Deliberately inject corruption through a schema-compatible writer.
        db.create_function('memory_condense_writer_schema_version', 0, lambda: CURRENT_SCHEMA_VERSION)
        if defect == 'vectors':
            db.execute('UPDATE snapshot SET vectors=?', (b'changed',))
        elif defect == 'summary':
            db.execute("UPDATE snapshot SET payload=replace(payload, 'Stored summary', 'Altered summary')")
        elif defect == 'raw':
            db.execute("UPDATE turns SET text=text || ' changed' WHERE ordinal=1")
        else:
            db.execute("UPDATE turns SET source_id='foreign' WHERE ordinal=1")
    with application(tmp_path, encoder, read_only=True) as app:
        with pytest.raises(ValueError, match='changed'):
            app.native_spine_receipt()


def test_new_turn_invalidates_resident_snapshot_and_readonly_cannot_publish(tmp_path):
    _, semantic, hierarchy, encoder, matrix, records = inputs()
    with application(tmp_path, encoder) as app:
        app.ingest_many(records)
        install(app, semantic, hierarchy, matrix)
        app.native_spine_receipt()
        app.ingest('user', 'Another new fact.', source_id='new')
        with pytest.raises(ValueError, match='advanced'):
            app.retrieve_native_spine(QUERY, DATED)
    with application(tmp_path, encoder, read_only=True) as app:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            install(app, semantic, hierarchy, matrix)


def test_reopen_requires_same_query_encoder(tmp_path):
    _, semantic, hierarchy, encoder, matrix, records = inputs()
    with application(tmp_path, encoder) as app:
        app.ingest_many(records)
        install(app, semantic, hierarchy, matrix)
    encoder.model_revision = 'different'
    with application(tmp_path, encoder, read_only=True) as app:
        with pytest.raises(ValueError, match='encoder differs'):
            app.native_spine_receipt()
