"""Exercise the public API default against the previously explicit cap-8 path."""
import sqlite3

import pytest

from memory_condense import MemoryCondenser
from memory_condense.application.native_spine_parent_users import ParentUserMemoryCondenser
from memory_condense.application.native_spine_policy import CAP8_LIMITS
from memory_condense.application.native_spine_user_completion import UserCompletionMemoryCondenser
from memory_condense.persistence import native_spine_parent_store as store
from memory_condense.search.native_spine_user_completion import NativeSpineUserCompletionRoute
from tests.test_native_spine_application_lifecycle import install
from tests.test_native_spine_user_completion import fixture, QUERY, DATED


def persisted(path):
    history, semantic, hierarchy, encoder, parents, _ = fixture()
    encoder.dim = 2
    encoder.embed_chunks = lambda chunks: [c.model_copy(update={'embedding': [1., 0.]}) for c in chunks]
    records = [(t.role, t.text, t.source_id, t.created_at, t.turn_id) for t in history.turns.values()]
    with MemoryCondenser(path, embedder=encoder, auto_extract=False) as app:
        app.ingest_many(records)
        native = install(app, semantic, hierarchy, semantic._dense._matrix.copy())
    receipt = store.publish(path / store.FILENAME, hierarchy=hierarchy,
        matrix=parents._dense._matrix, native_receipt=native)
    return encoder, receipt


def test_default_public_api_matches_explicit_tested_policy_after_reopen(tmp_path):
    encoder, receipt = persisted(tmp_path)
    with UserCompletionMemoryCondenser(tmp_path, embedder=encoder, auto_extract=False, read_only=True) as old:
        expected = old.retrieve_native_spine(QUERY, DATED, **CAP8_LIMITS)
    with MemoryCondenser(tmp_path, embedder=encoder, auto_extract=False, read_only=True) as app:
        assert app.native_parent_user_receipt() == receipt
        actual = app.retrieve_native_spine(QUERY, DATED)
        assert actual == expected
        assert actual.routing.user_completion_atoms == 8
        assert actual.routing.raw_reads_during_routing == actual.routing.query_qwen_passes == 0
        assert actual.hydration.context_token_count <= 2048
        overridden = app.retrieve_native_spine(QUERY, DATED, user_completion_atoms=0, max_context_tokens=128)
        assert overridden.routing.user_completion_atoms == 0
        assert not overridden.routing.completion_added_atomic_ids
        assert overridden.hydration.context_token_count <= 128
    assert encoder.calls == [QUERY] * 3


def test_explicit_historical_parent_facade_keeps_original_routing(tmp_path):
    encoder, _ = persisted(tmp_path)
    with ParentUserMemoryCondenser(tmp_path, embedder=encoder, auto_extract=False, read_only=True) as app:
        limits = {k: v for k, v in CAP8_LIMITS.items() if k != 'user_completion_atoms'}
        assert not isinstance(app.retrieve_native_spine(QUERY, DATED, **limits).routing,
                              NativeSpineUserCompletionRoute)


def test_corrupt_parent_index_cannot_silently_disable_completion(tmp_path):
    encoder, _ = persisted(tmp_path)
    with sqlite3.connect(tmp_path / store.FILENAME) as db:
        db.execute('UPDATE snapshot SET vectors=?', (b'corrupt',))
    with MemoryCondenser(tmp_path, embedder=encoder, auto_extract=False, read_only=True) as app:
        for _ in range(2):
            with pytest.raises(ValueError, match='binding changed'):
                app.retrieve_native_spine(QUERY, DATED)
        assert not encoder.calls
