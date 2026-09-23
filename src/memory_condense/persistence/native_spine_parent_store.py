"""Persist derived parent-summary vectors against an existing native snapshot."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.search.native_spine_parent_users import project_parent_users
from memory_condense.search.summary_semantic_index import SemanticSectionIndex


FORMAT = 'persisted-native-spine-parent-users-v1'
FILENAME = 'native-spine-parent-users-v1.sqlite'


def _receipt(p, sha):
    return {'snapshot_sha256': sha, **{k: p[k] for k in ('native_snapshot_sha256',
        'projection_sha256', 'semantic_sha256', 'embedding_identity', 'matrix_shape')}}


def publish(path, *, hierarchy, matrix, native_receipt):
    """Publish once; keep the ingested transcript and original indexes unchanged."""
    path = Path(path)
    if path.exists():
        raise ValueError('parent summary snapshot publication requires a fresh path')
    if hierarchy.receipt_sha256 != native_receipt['hierarchy_sha256']:
        raise ValueError('parent summaries differ from the native snapshot hierarchy')
    projection = project_parent_users(hierarchy)
    matrix = np.asarray(matrix)
    if matrix.dtype != np.float32:
        raise ValueError('parent summary vectors must be stored as FP32')
    semantic = SemanticSectionIndex(projection, matrix, embedding_identity=native_receipt['embedding_identity'])
    raw = matrix.tobytes(order='C')
    p = {'format': FORMAT, 'native_snapshot_sha256': native_receipt['snapshot_sha256'],
         'projection_sha256': projection.receipt_sha256, 'semantic_sha256': semantic.receipt_sha256,
         'embedding_identity': semantic.embedding_identity, 'matrix_shape': list(matrix.shape),
         'matrix_dtype': 'float32', 'matrix_sha256': hashlib.sha256(raw).hexdigest()}
    sha = identity_sha256(p)
    # Exclusive path reservation avoids accidentally overwriting a historical cache.
    with path.open('xb'):
        pass
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE snapshot (id INTEGER PRIMARY KEY CHECK(id=1), '
                   'payload TEXT NOT NULL, vectors BLOB NOT NULL, receipt_sha256 TEXT NOT NULL)')
        db.execute('INSERT INTO snapshot VALUES (1, ?, ?, ?)', (canonical_json(p), raw, sha))
    return _receipt(p, sha)


def load(path, *, hierarchy, native_receipt):
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        row = db.execute('SELECT payload, vectors, receipt_sha256 FROM snapshot WHERE id=1').fetchone()
    if row is None:
        raise ValueError('parent summary snapshot is absent')
    serialized, raw, sha = row
    p = json.loads(serialized)
    if (p['format'] != FORMAT or identity_sha256(p) != sha
            or p['native_snapshot_sha256'] != native_receipt['snapshot_sha256']
            or p['embedding_identity'] != native_receipt['embedding_identity']
            or hierarchy.receipt_sha256 != native_receipt['hierarchy_sha256']
            or p['matrix_dtype'] != 'float32' or hashlib.sha256(raw).hexdigest() != p['matrix_sha256']):
        raise ValueError('parent vectors or their native snapshot binding changed')
    projection = project_parent_users(hierarchy)
    matrix = np.frombuffer(raw, dtype=np.float32).reshape(p['matrix_shape'])
    semantic = SemanticSectionIndex(projection, matrix, embedding_identity=p['embedding_identity'])
    if projection.receipt_sha256 != p['projection_sha256'] or semantic.receipt_sha256 != p['semantic_sha256']:
        raise ValueError('stored parent projection changed during reconstruction')
    return semantic, _receipt(p, sha)
