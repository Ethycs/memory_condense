from dataclasses import replace
import json
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from memory_condense.persistence import native_spine_parent_store as store
from memory_condense.search.native_spine_parent_users import project_parent_users
from memory_condense.search.section_routing import SectionSummaryIndex
from tests.test_native_spine_context_routing import parent_fixture


def fixture():
    h, semantic, hierarchy, encoder = parent_fixture(future=True)
    hierarchy = SectionSummaryIndex([replace(s, summary=json.dumps({
        'user_spine': 'User describes a visit and birdwatching.',
        'attached_context_not_user_assertions': 'SECRET_ASSISTANT_ONLY'}), receipt_sha256='')
        for s in hierarchy.sections])
    receipt = {'snapshot_sha256': 'a' * 64, 'hierarchy_sha256': hierarchy.receipt_sha256,
               'embedding_identity': semantic.embedding_identity}
    return h, hierarchy, receipt


def test_roots_keep_all_exact_user_addresses_and_exclude_assistant_text():
    h, hierarchy, _ = fixture()
    projected = project_parent_users(hierarchy)
    assert len(projected.sections) == 2
    assert all(not s.child_section_ids and 'SECRET' not in s.summary for s in projected.sections)
    assert {span for s in projected.sections for span in s.spans} == {
        a.spans[0] for a in h.atoms if a.spans[0].role == 'user'}
    changed = SectionSummaryIndex([replace(s, summary='not structured', receipt_sha256='')
                                   for s in hierarchy.sections])
    with pytest.raises(ValueError, match='stored user-spine'):
        project_parent_users(changed)


def test_persist_reopen_and_reject_wrong_snapshot_or_corrupt_vectors(tmp_path):
    _, hierarchy, native = fixture()
    path = tmp_path / store.FILENAME
    matrix = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    receipt = store.publish(path, hierarchy=hierarchy, matrix=matrix, native_receipt=native)
    loaded, same = store.load(path, hierarchy=hierarchy, native_receipt=native)
    assert same == receipt and loaded.hierarchy.receipt_sha256 == project_parent_users(hierarchy).receipt_sha256
    with pytest.raises(ValueError, match='fresh path'):
        store.publish(path, hierarchy=hierarchy, matrix=matrix, native_receipt=native)
    with pytest.raises(ValueError, match='binding changed'):
        store.load(path, hierarchy=hierarchy, native_receipt={**native, 'snapshot_sha256': 'b' * 64})
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('UPDATE snapshot SET vectors=?', (b'changed',))
    with pytest.raises(ValueError, match='binding changed'):
        store.load(path, hierarchy=hierarchy, native_receipt=native)
