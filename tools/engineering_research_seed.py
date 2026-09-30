"""Reuse an authenticated native history when continuing it through live chat."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.persistence import native_spine_store, native_spine_parent_store
from memory_condense.search.native_spine_parent_users import project_parent_users
from tools.engineering_research_gateway import read


@dataclass
class NativeSeed:
    turns: tuple
    atoms: tuple
    hierarchy: tuple
    embedding_identity: str
    vectors: dict
    atomic_index: object = None
    hierarchy_index: object = None


def load_seed(binding, rows):
    """Keep original addresses and vectors; new source turns use the usual compiler.

    A seed is immutable historical material, not the active recall index. Every
    refresh publishes a combined index covering both this prefix and live I/O.
    """
    directory = Path(binding['directory'])
    expected = read(Path(binding['receipt']))
    for name, sha in expected['application_files'].items():
        path = directory / name
        if path.resolve().parent != directory.resolve():
            raise ValueError('Invalid seed file path')
        with path.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != sha:
                raise ValueError('Native seed application changed')
    with Database(directory / 'memory.db', read_only=True) as db:
        turns = tuple(TranscriptStore(db).get_all())
    if len(rows) < len(turns):
        raise ValueError('Chat prefix does not contain the complete native seed')
    for turn, row in zip(turns, rows):
        if ((turn.turn_id, turn.source_id, turn.role, turn.text, turn.created_at.isoformat()) !=
                (row['turn_id'], row['source_id'], row['role'], row['text'], row['created_at'])):
            raise ValueError('Chat prefix differs from native seed provenance')
    sources = {t.source_id for t in turns}
    if any(r['source_id'] in sources for r in rows[len(turns):]):
        raise ValueError('A continuation must have its own live source identity')
    snapshot = native_spine_store.load(directory / 'native-spine.sqlite', turns=turns)
    parents, receipt = native_spine_parent_store.load(directory / native_spine_parent_store.FILENAME,
        hierarchy=snapshot.hierarchy, native_receipt=snapshot.receipt)
    if snapshot.receipt != expected['snapshot'] or receipt != expected['parent_snapshot']:
        raise ValueError('Native seed receipt mismatch')
    vectors = {}
    for semantic in (snapshot.semantic, parents):
        for section, vector in zip(semantic.sections, semantic._dense._matrix, strict=True):
            vectors[section.section_id] = (section.summary, vector)
    parent_by_root = {json.loads(s.summarizer_identity)['original_root_sha256']: (s.summary, v)
                      for s, v in zip(parents.sections, parents._dense._matrix, strict=True)}
    for section in project_parent_users(snapshot.hierarchy, stable_ids=True).sections:
        vectors[section.section_id] = parent_by_root[json.loads(section.summarizer_identity)['original_root_sha256']]
    return NativeSeed(turns, snapshot.semantic.sections, snapshot.hierarchy.sections,
                      snapshot.semantic.embedding_identity, vectors, snapshot.semantic.hierarchy, snapshot.hierarchy)
