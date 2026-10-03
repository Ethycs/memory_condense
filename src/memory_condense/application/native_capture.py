"""Exact native IO and learning topology, without the legacy raw search index.

Native publication is the searchable completion receipt. These turns never
claim that a dense/lexical ingest completed and never enter its work queue.
The same source chunks remain available to the existing Hebbian graph.
"""
import json

from memory_condense.application.ingest_workflow import _bind_explicit_chunk_ids
from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.persistence.pending_ingest_store import PendingIngestManifest


def capture_native_many(app, records):
    if app._auto_extract:
        raise ValueError('Native capture requires explicit summary compilation')
    records, explicit_times = app._normalize_ingest_records(records)
    connection = app._db.connection
    output = []
    try:
        connection.execute('BEGIN IMMEDIATE')
        previous = app.transcript.native_snapshot()
        appended = []
        connection.execute('CREATE TABLE IF NOT EXISTS native_captures ('
            'turn_id TEXT PRIMARY KEY REFERENCES turns(turn_id), '
            'payload TEXT NOT NULL, sha TEXT NOT NULL)')
        for role, text, source_id, created_at, turn_id in records:
            if turn_id is None:
                raise ValueError('Native capture requires stable turn identities')
            turn = app.transcript.stage(role, text, source_id=source_id,
                created_at=explicit_times.get(turn_id, created_at), turn_id=turn_id)
            turn, inserted = app.transcript.publish_turn(turn,
                compare_created_at=turn_id in explicit_times, commit=False)
            saved = connection.execute('SELECT payload,sha FROM native_captures WHERE turn_id=?',
                                       (turn_id,)).fetchone()
            if saved is not None:
                payload = json.loads(saved[0])
                if (identity_sha256(payload) != saved[1]
                        or payload.get('format') != 'native-capture-v1'
                        or payload.get('turn_sha256') != identity_sha256(turn.model_dump(mode='json'))):
                    raise ValueError('Native capture receipt differs from its source')
                chunks = PendingIngestManifest.from_json(payload['topology']).reconstruct(turn)
                stored = connection.execute('SELECT chunk_id,turn_id,text,start_char,end_char,token_count '
                    'FROM chunks WHERE turn_id=? ORDER BY chunk_id', (turn_id,)).fetchall()
                expected = sorted((c.chunk_id,c.turn_id,c.text,c.start_char,c.end_char,c.token_count) for c in chunks)
                if stored != expected:
                    raise ValueError('Native capture chunks differ from their receipt')
            elif not inserted:
                # An older store may already own a different chunk topology.
                # Do not replace those nodes or reinterpret its pending work.
                raise ValueError('Existing turn has no native capture receipt')
            else:
                appended.append(turn)
                chunks = _bind_explicit_chunk_ids(turn_id, app._chunker.chunk_turn(turn_id, text))
                payload = dict(format='native-capture-v1',
                    turn_sha256=identity_sha256(turn.model_dump(mode='json')),
                    topology=PendingIngestManifest.build(turn, chunks).canonical_json)
                connection.executemany('INSERT INTO chunks '
                    '(chunk_id,turn_id,text,start_char,end_char,token_count) VALUES (?,?,?,?,?,?)',
                    [(c.chunk_id,c.turn_id,c.text,c.start_char,c.end_char,c.token_count) for c in chunks])
                connection.execute('INSERT INTO native_captures VALUES (?,?,?)',
                                   (turn_id, canonical_json(payload), identity_sha256(payload)))
            output.append((turn, chunks))
        revision = app.transcript.source_revision()
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    app.transcript._remember_native_snapshot((*previous, *appended), revision)
    return output
