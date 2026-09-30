"""Real SQLite transactions, changed-row writes, corruption and cold parity."""
from dataclasses import replace
import json
import sqlite3

import numpy as np
import pytest

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.native_spine_parent_users import ParentUserMemoryCondenser
from memory_condense.application.native_spine_user_completion import UserCompletionMemoryCondenser
from memory_condense.persistence import native_spine_incremental_store as store
from memory_condense.search.native_spine_parent_users import project_parent_users
from memory_condense.search.section_routing import SectionSummaryIndex
from tests.test_native_spine_default_policy import persisted
from tests.test_native_spine_user_completion import QUERY, DATED


def initial(tmp_path):
    encoder, _ = persisted(tmp_path)
    with MemoryCondenser(tmp_path, embedder=encoder, auto_extract=False) as app:
        native = app._load_native_spine()[1]
        prior = app._load_native_spine()[2].router.parent_semantic
        by_root = {json.loads(s.summarizer_identity)['original_root_sha256']: v
                   for s, v in zip(prior.sections, prior._dense._matrix, strict=True)}
        projection = project_parent_users(native.hierarchy, stable_ids=True)
        matrix = np.asarray([by_root[json.loads(s.summarizer_identity)['original_root_sha256']]
                             for s in projection.sections], dtype=np.float32)
        args = dict(atomic_index=native.semantic.hierarchy, hierarchy=native.hierarchy,
                    matrix=native.semantic._dense._matrix, projection=projection, parent_matrix=matrix,
                    embedding_identity=native.semantic.embedding_identity, turns=app.transcript.get_all())
        old_packet = app.retrieve_native_spine(QUERY, DATED)
        state = store.publish(tmp_path/store.FILENAME, **args)
    return encoder, args, state, old_packet


def test_changed_rows_match_fresh_store_and_transaction_rollback(tmp_path):
    encoder, args, state, _ = initial(tmp_path)
    path = tmp_path/store.FILENAME
    same = store.publish(path, previous=state, **args)
    assert all(v == {'upserted': 0, 'deleted': 0} for v in same.write_counts.values())
    sections = args['atomic_index'].sections
    changed = replace(sections[0], summary=sections[0].summary+' clarified', receipt_sha256='')
    update = dict(args, atomic_index=args['atomic_index'].updated([changed, *sections[1:]]))
    def crash():
        raise RuntimeError('crash before commit')
    with pytest.raises(RuntimeError, match='crash before commit'):
        store.publish(path, previous=same, before_commit=crash, **update)
    assert store.load(path, turns=args['turns']).manifest == state.manifest
    actual = store.publish(path, previous=same, **update)
    with pytest.raises(ValueError, match='changed since admission'):
        store.publish(path, previous=same, **args)
    assert actual.write_counts == {'atomic': {'upserted': 1, 'deleted': 0},
        'hierarchy': {'upserted': 0, 'deleted': 0}, 'parents': {'upserted': 0, 'deleted': 0}}
    fresh_args = dict(update, atomic_index=SectionSummaryIndex(update['atomic_index'].sections),
                     hierarchy=SectionSummaryIndex(update['hierarchy'].sections),
                     projection=project_parent_users(update['hierarchy'], stable_ids=True))
    fresh = store.publish(tmp_path/'fresh.sqlite', **fresh_args)
    loaded = store.load(path, turns=args['turns'])
    assert actual.manifest == fresh.manifest == loaded.manifest
    for q in (QUERY, 'clarified', 'pine', 'nonexistent'):
        assert actual.native.semantic.hierarchy.route(q) == fresh.native.semantic.hierarchy.route(q)
    with MemoryCondenser(tmp_path, embedder=encoder, auto_extract=False, read_only=True) as app:
        assert app.native_spine_receipt() == actual.native.receipt
        assert app.native_parent_user_receipt() == actual.parent_receipt
        assert app.retrieve_native_spine(QUERY, DATED).hydration


@pytest.mark.parametrize('defect', ['atomic_vector','parent_vector','summary','missing_row','manifest','raw'])
def test_cold_load_rejects_changed_rows_or_transcript(tmp_path, defect):
    _, args, _, _ = initial(tmp_path)
    path = tmp_path/store.FILENAME
    if defect == 'raw':
        turns = list(args['turns'])
        turns[0] = turns[0].model_copy(update={'text': turns[0].text+' changed'})
    else:
        turns = args['turns']
        with sqlite3.connect(path) as db:
            if defect.endswith('_vector'):
                kind = 'atomic' if defect == 'atomic_vector' else 'parents'
                db.execute('UPDATE sections SET vector=? WHERE kind=?', (b'changed',kind))
            elif defect == 'summary':
                db.execute("UPDATE sections SET payload=replace(payload, 'summary', 'changed') WHERE kind='hierarchy'")
            elif defect == 'missing_row':
                db.execute("DELETE FROM sections WHERE kind='atomic' AND sid=(SELECT min(sid) FROM sections WHERE kind='atomic')")
            else:
                db.execute("UPDATE manifest SET sha=?", ('0'*64,))
    with pytest.raises((ValueError, TypeError)):
        store.load(path, turns=turns)


def test_stable_parent_rejects_forged_content_under_correct_id(tmp_path):
    _, args, state, _ = initial(tmp_path)
    parent, *rest = args['projection'].sections
    forged = replace(parent, summary='unsupported changed content', receipt_sha256='')
    with pytest.raises(ValueError, match='projection differs'):
        store.publish(tmp_path/'forged.sqlite', **dict(args, projection=SectionSummaryIndex([forged,*rest])))


@pytest.mark.parametrize('facade', [ParentUserMemoryCondenser, UserCompletionMemoryCondenser])
def test_explicit_facades_admit_the_incremental_parent_snapshot(tmp_path, facade):
    encoder, _, state, _ = initial(tmp_path)
    with facade(tmp_path, embedder=encoder, auto_extract=False, read_only=True) as app:
        assert app.native_spine_receipt() == state.native.receipt
        assert app.native_parent_user_receipt() == state.parent_receipt
        assert app.retrieve_native_spine(QUERY, DATED).hydration.sections
    assert app._native_spine_incremental is None
