"""Transactional per-section native snapshots, with stable parent addresses.

Only changed section/vector rows are written on a warm update. The manifest
retains complete content hashes; cold admission reconstructs and checks them.
Legacy native and parent files remain readable as the pre-migration checkpoint.
"""
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256, quote_sha256
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.summary_semantic_index import SemanticSectionIndex
from memory_condense.search.native_spine_parent_users import project_parent_users
from memory_condense.persistence import native_spine_store as native
from memory_condense.persistence.native_source_validation import validate as validate_sources


FILENAME = 'native-spine-live-v1.sqlite'
FORMAT = 'memory-condense-native-section-store-v1'


@dataclass(frozen=True)
class IncrementalSnapshot:
    native: native.NativeSpineSnapshot
    parents: SemanticSectionIndex
    parent_receipt: dict
    manifest: dict
    # Per-row identities avoid comparing/serializing whole matrices on update.
    row_hashes: dict
    write_counts: dict
    source_validation: object


def _entries(index, matrix=None, *, old_index=None, old_matrix=None, old_hashes=None):
    if matrix is not None:
        matrix = np.asarray(matrix)
        if matrix.dtype != np.float32 or matrix.ndim != 2 or len(matrix) != len(index.sections):
            raise ValueError('incremental vectors must be aligned FP32 rows')
    old_positions = {s.section_id: i for i,s in enumerate(old_index.sections)} if old_index is not None else {}
    for i, section in enumerate(index.sections):
        payload = index._section_json[section.section_id]
        position = old_positions.get(section.section_id)
        if (position is not None and payload == old_index._section_json[section.section_id]
                and (matrix is None or (matrix.shape[1:] == old_matrix.shape[1:]
                     and np.array_equal(matrix[i].view(np.uint8), old_matrix[position].view(np.uint8))))):
            # An unchanged row needs no vector copy, serialization or hash.
            # Its admitted identity is retained; it will not enter SQL upserts.
            yield section.section_id, payload, None, old_hashes[section.section_id]
            continue
        vector = None if matrix is None else matrix[i].tobytes(order='C')
        sha = quote_sha256(payload) if vector is None else identity_sha256(
            {'section': section.receipt_sha256, 'vector': hashlib.sha256(vector).hexdigest()})
        yield section.section_id, payload, vector, sha


def _assemble(atomic, hierarchy, matrix, projection, parent_matrix, embedding_identity, turns, previous=None):
    def semantic_index(index, vectors, old):
        if (old is not None and index is old.hierarchy and old.embedding_identity == embedding_identity
                and vectors.shape == old._dense._matrix.shape
                and np.array_equal(vectors.view(np.uint8), old._dense._matrix.view(np.uint8))):
            return old
        return SemanticSectionIndex(index, vectors, embedding_identity=embedding_identity)
    semantic = semantic_index(atomic, matrix, previous.native.semantic if previous else None)
    parents = semantic_index(projection, parent_matrix, previous.parents if previous else None)
    sources = validate_sources(semantic, hierarchy, turns, previous)
    if (previous is not None and semantic is previous.native.semantic and parents is previous.parents
            and hierarchy.receipt_sha256 == previous.native.hierarchy.receipt_sha256
            and sources.transcript_sha256 == previous.source_validation.transcript_sha256):
        return previous.native, parents, previous.parent_receipt, previous.manifest, sources
    raw = np.asarray(matrix).tobytes(order='C')
    # The native receipt remains identical to the v1 full rebuild. Cached
    # canonical section strings avoid reflective serialization of old sections.
    metadata = dict(format=native.FORMAT, embedding_identity=embedding_identity,
        matrix_shape=list(matrix.shape), matrix_sha256=hashlib.sha256(raw).hexdigest(),
        matrix_dtype='float32', semantic_sha256=semantic.receipt_sha256,
        hierarchy_sha256=hierarchy.receipt_sha256, transcript_sha256=sources.transcript_sha256,
        turn_count=len(turns), body_tokens=sources.body_tokens)
    fields = {k: canonical_json(v) for k, v in metadata.items()}
    fields.update(atomic_sections=atomic.sections_json(), hierarchy_sections=hierarchy.sections_json())
    serialized = '{' + ','.join(canonical_json(k)+':'+fields[k] for k in sorted(fields)) + '}'
    receipt = native._receipt(metadata, quote_sha256(serialized))
    parent_payload = dict(format='persisted-native-spine-parent-users-v2',
        native_snapshot_sha256=receipt['snapshot_sha256'], projection_sha256=projection.receipt_sha256,
        semantic_sha256=parents.receipt_sha256, embedding_identity=embedding_identity,
        matrix_shape=list(parent_matrix.shape), matrix_dtype='float32',
        matrix_sha256=hashlib.sha256(np.asarray(parent_matrix).tobytes(order='C')).hexdigest())
    parent_receipt = dict(snapshot_sha256=identity_sha256(parent_payload), **{k: parent_payload[k] for k in
        ('native_snapshot_sha256', 'projection_sha256', 'semantic_sha256', 'embedding_identity', 'matrix_shape')})
    manifest = dict(format=FORMAT, native=metadata, native_receipt=receipt,
                    parent=parent_payload, parent_receipt=parent_receipt)
    return native.NativeSpineSnapshot(semantic, hierarchy, receipt), parents, parent_receipt, manifest, sources


def publish(path, *, atomic_index, hierarchy, matrix, projection, parent_matrix, embedding_identity,
            turns, previous=None, before_commit=None):
    """Validate first, then atomically commit both indexes and their manifest."""
    turns = tuple(turns)
    matrix, parent_matrix = np.ascontiguousarray(matrix), np.ascontiguousarray(parent_matrix)
    if (previous is None or hierarchy.receipt_sha256 != previous.native.hierarchy.receipt_sha256
            or projection.receipt_sha256 != previous.parents.hierarchy.receipt_sha256):
        expected_projection = project_parent_users(hierarchy, stable_ids=True, previous=projection)
        if expected_projection.receipt_sha256 != projection.receipt_sha256:
            raise ValueError('incremental parent projection differs from hierarchy')
    snapshot, parents, parent_receipt, manifest, sources = _assemble(atomic_index, hierarchy, matrix,
        projection, parent_matrix, embedding_identity, turns, previous)
    populations = {'atomic': (atomic_index, matrix), 'hierarchy': (hierarchy, None),
                   'parents': (projection, parent_matrix)}
    hashes, changed, removed = {}, {}, {}
    for kind, (index, vectors) in populations.items():
        old = previous.row_hashes.get(kind, {}) if previous is not None else {}
        if previous is not None and manifest is previous.manifest:
            hashes[kind], changed[kind], removed[kind] = old, [], set()
            continue
        if previous is None:
            old_index = old_matrix = None
        elif kind == 'hierarchy':
            old_index, old_matrix = previous.native.hierarchy, None
        else:
            semantic = previous.native.semantic if kind == 'atomic' else previous.parents
            old_index, old_matrix = semantic.hierarchy, semantic._dense._matrix
        rows = list(_entries(index, vectors, old_index=old_index, old_matrix=old_matrix, old_hashes=old))
        hashes[kind] = {sid: sha for sid, _, _, sha in rows}
        changed[kind] = [(sid, payload, vector, sha) for sid, payload, vector, sha in rows if old.get(sid) != sha]
        removed[kind] = set(old) - set(hashes[kind])
    path = Path(path)
    with closing(sqlite3.connect(path)) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        with db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('CREATE TABLE IF NOT EXISTS manifest (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL, sha TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS sections (kind TEXT NOT NULL, sid TEXT NOT NULL, payload TEXT NOT NULL, vector BLOB, sha TEXT NOT NULL, PRIMARY KEY(kind,sid))')
            stored = db.execute('SELECT sha FROM manifest WHERE id=1').fetchone()
            expected = identity_sha256(previous.manifest) if previous is not None else None
            if (stored[0] if stored else None) != expected:
                raise ValueError('incremental snapshot changed since admission')
            for kind in populations:
                db.executemany('DELETE FROM sections WHERE kind=? AND sid=?', ((kind,sid) for sid in removed[kind]))
                db.executemany('INSERT OR REPLACE INTO sections VALUES (?,?,?,?,?)',
                               ((kind, *row) for row in changed[kind]))
            db.execute('INSERT OR REPLACE INTO manifest VALUES (1,?,?)',
                       (canonical_json(manifest), identity_sha256(manifest)))
            if before_commit is not None:
                before_commit()
    return IncrementalSnapshot(snapshot, parents, parent_receipt, manifest, hashes,
        {k: dict(upserted=len(changed[k]), deleted=len(removed[k])) for k in populations}, sources)


def saved_turn_count(path):
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)) as db:
        row = db.execute('SELECT payload,sha FROM manifest WHERE id=1').fetchone()
    if row is None:
        raise ValueError('incremental memory has no committed snapshot')
    p = json.loads(row[0])
    if identity_sha256(p) != row[1] or p['format'] != FORMAT:
        raise ValueError('incremental manifest changed')
    return p['native']['turn_count']


def load(path, *, turns):
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)) as db, db:
        db.execute('BEGIN')
        row = db.execute('SELECT payload,sha FROM manifest WHERE id=1').fetchone()
        stored = db.execute('SELECT kind,sid,payload,vector,sha FROM sections ORDER BY kind,sid').fetchall()
    if row is None:
        raise ValueError('incremental memory has no committed snapshot')
    manifest = json.loads(row[0])
    if manifest.get('format') != FORMAT or identity_sha256(manifest) != row[1]:
        raise ValueError('incremental manifest changed')
    groups = {k: [] for k in ('atomic','hierarchy','parents')}
    hashes = {k: {} for k in groups}
    vectors = {k: [] for k in ('atomic','parents')}
    for kind, sid, payload, vector, sha in stored:
        if kind not in groups:
            raise ValueError('unknown incremental section kind')
        section = SectionSummary.from_dict(json.loads(payload))
        expected = quote_sha256(payload) if vector is None else identity_sha256(
            {'section': section.receipt_sha256, 'vector': hashlib.sha256(vector).hexdigest()})
        if section.section_id != sid or expected != sha or (vector is None) != (kind == 'hierarchy'):
            raise ValueError('incremental section or vector changed')
        groups[kind].append(section)
        hashes[kind][sid] = sha
        if vector is not None:
            vectors[kind].append(np.frombuffer(vector, dtype=np.float32))
    atomic, hierarchy, projection = (SectionSummaryIndex(groups[k]) for k in ('atomic','hierarchy','parents'))
    if project_parent_users(hierarchy, stable_ids=True).receipt_sha256 != projection.receipt_sha256:
        raise ValueError('persisted parent projection changed')
    snapshot, parents, parent_receipt, reconstructed, sources = _assemble(atomic, hierarchy,
        np.asarray(vectors['atomic']), projection, np.asarray(vectors['parents']),
        manifest['native']['embedding_identity'], tuple(turns))
    if reconstructed != manifest:
        raise ValueError('incremental snapshot differs from its transcript or manifest')
    return IncrementalSnapshot(snapshot, parents, parent_receipt, manifest, hashes, {}, sources)
