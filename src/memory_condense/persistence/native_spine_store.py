"""Durable native summary indexes bound to the application's exact transcript.

Raw turns remain in TranscriptStore. This derived snapshot stores summaries,
raw addresses and FP32 summary vectors, never questions or reference answers.
"""
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from memory_condense.application.section_retrieval import HydratedSectionSpan
from memory_condense.domain._discourse_identity import canonical_json, identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.native_spine_context_routing import NativeSpineContextRouter
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.summary_semantic_index import SemanticSectionIndex


FORMAT = 'memory-condense-persisted-native-spine-v1'


def transcript_identity(turns):
    return identity_sha256([{'turn_id': t.turn_id, 'source_id': t.source_id,
        'role': t.role, 'created_at': t.created_at.isoformat(), 'text_sha256': quote_sha256(t.text)}
        for t in turns])


def validate_sources(semantic, hierarchy, turns):
    """Require full, gap-free coverage of this application's ordered raw turns."""
    raw = {t.turn_id: t for t in turns}
    if not raw or len(raw) != len(turns):
        raise ValueError('native snapshot requires unique ingested raw turns')
    router = NativeSpineContextRouter(semantic, hierarchy)
    if len(router.owners) != len(semantic.sections):
        raise ValueError('native snapshot requires a complete attention hierarchy')
    by_turn = defaultdict(list)
    for section in semantic.sections:
        span, = section.spans
        turn = raw.get(span.turn_id)
        if (turn is None or span.source_id != turn.source_id or span.role != turn.role
                or span.created_at != turn.created_at.isoformat()
                or span.turn_text_sha256 != quote_sha256(turn.text)
                or span.end_char > len(turn.text)):
            raise ValueError('native snapshot differs from ingested raw provenance')
        HydratedSectionSpan(span, turn.text[span.start_char:span.end_char])
        by_turn[turn.turn_id].append(span)
    if set(by_turn) != set(raw):
        raise ValueError('native snapshot does not cover every ingested turn')
    for turn_id, spans in by_turn.items():
        cursor = 0
        for span in sorted(spans, key=lambda s: s.start_char):
            if span.start_char != cursor:
                raise ValueError('native snapshot raw coverage has a gap or overlap')
            cursor = span.end_char
        if cursor != len(raw[turn_id].text):
            raise ValueError('native snapshot raw coverage is incomplete')


@dataclass(frozen=True)
class NativeSpineSnapshot:
    semantic: SemanticSectionIndex
    hierarchy: SectionSummaryIndex
    receipt: dict


def publish(path, *, atomic_index, hierarchy, matrix, embedding_identity, turns):
    """Commit one complete derived snapshot after raw ingestion has completed."""
    turns = tuple(turns)
    matrix = np.asarray(matrix)
    semantic = SemanticSectionIndex(atomic_index, matrix, embedding_identity=embedding_identity)
    validate_sources(semantic, hierarchy, turns)
    vector_bytes = matrix.tobytes(order='C')
    payload = {'format': FORMAT, 'atomic_sections': [s.identity_payload() for s in atomic_index.sections],
        'hierarchy_sections': [s.identity_payload() for s in hierarchy.sections],
        'embedding_identity': embedding_identity, 'matrix_shape': list(matrix.shape),
        'matrix_sha256': hashlib.sha256(vector_bytes).hexdigest(), 'matrix_dtype': 'float32',
        'semantic_sha256': semantic.receipt_sha256, 'hierarchy_sha256': hierarchy.receipt_sha256,
        'transcript_sha256': transcript_identity(turns), 'turn_count': len(turns),
        'body_tokens': sum(count_tokens(t.text) for t in turns)}
    sha = identity_sha256(payload)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS snapshot (id INTEGER PRIMARY KEY CHECK(id=1), '
                   'payload TEXT NOT NULL, vectors BLOB NOT NULL, receipt_sha256 TEXT NOT NULL)')
        db.execute('INSERT OR REPLACE INTO snapshot VALUES (1, ?, ?, ?)',
                   (canonical_json(payload), vector_bytes, sha))
    return _receipt(payload, sha)


def _receipt(p, sha):
    return {'snapshot_sha256': sha, **{k: p[k] for k in ('transcript_sha256', 'turn_count',
        'body_tokens', 'semantic_sha256', 'hierarchy_sha256', 'embedding_identity')}}


def load(path, *, turns):
    """Reconstruct only from the persisted snapshot and the application store."""
    turns = tuple(turns)
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        row = db.execute('SELECT payload, vectors, receipt_sha256 FROM snapshot WHERE id=1').fetchone()
    if row is None:
        raise ValueError('native memory has no published snapshot')
    serialized, vector_bytes, sha = row
    p = json.loads(serialized)
    if (p['format'] != FORMAT or identity_sha256(p) != sha
            or p['matrix_dtype'] != 'float32'
            or hashlib.sha256(vector_bytes).hexdigest() != p['matrix_sha256']
            or transcript_identity(turns) != p['transcript_sha256']
            or len(turns) != p['turn_count']
            or sum(count_tokens(t.text) for t in turns) != p['body_tokens']):
        raise ValueError('persisted native memory or its ingested transcript changed')
    atomic = SectionSummaryIndex([SectionSummary.from_dict(s) for s in p['atomic_sections']])
    hierarchy = SectionSummaryIndex([SectionSummary.from_dict(s) for s in p['hierarchy_sections']])
    matrix = np.frombuffer(vector_bytes, dtype=np.float32).reshape(p['matrix_shape'])
    semantic = SemanticSectionIndex(atomic, matrix, embedding_identity=p['embedding_identity'])
    if semantic.receipt_sha256 != p['semantic_sha256'] or hierarchy.receipt_sha256 != p['hierarchy_sha256']:
        raise ValueError('persisted native indexes changed during reconstruction')
    validate_sources(semantic, hierarchy, turns)
    return NativeSpineSnapshot(semantic, hierarchy, _receipt(p, sha))
