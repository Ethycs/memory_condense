from dataclasses import replace
import json

import numpy as np
import pytest

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.native_spine_parent_users import ParentUserMemoryCondenser
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.persistence import native_spine_parent_store as store
from memory_condense.search.native_spine_additive_lexical import NativeSpineAdditiveLexicalRouter
from memory_condense.search.native_spine_parent_user_routing import (
    NativeSpineParentUserRoute, NativeSpineParentUserRouter, route_from_payload,
)
from memory_condense.search.native_spine_parent_users import project_parent_users
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.summary_semantic_index import SemanticSectionIndex
from tests.test_native_spine_application_lifecycle import inputs, install
from tests.test_native_spine_routing import QUERY, DATED


OPTIONS = dict(max_direct=1, lexical_reserve=0, context_seed_limit=1,
               max_additions=1, protected_direct=0, ancestor_hops=0)


def fixture():
    h, s, hierarchy, encoder, matrix, records = inputs()
    hierarchy = SectionSummaryIndex([replace(r, summary=json.dumps({
        'user_spine': 'The user described the visit and birdwatching.',
        'attached_context_not_user_assertions': 'Assistant suggested an unrelated option.'}),
        receipt_sha256='') for r in hierarchy.sections])
    projection = project_parent_users(hierarchy)
    parents = SemanticSectionIndex(projection, np.asarray([[1, 0]] * len(projection.sections), dtype=np.float32),
                                    embedding_identity=s.embedding_identity)
    return h, s, hierarchy, encoder, matrix, records, parents


def routes(options=None):
    h, s, hierarchy, _, _, _, parents = fixture()
    kwargs = dict(embedding_identity=s.embedding_identity, **(options or OPTIONS))
    old = NativeSpineAdditiveLexicalRouter(s, hierarchy).route_vector(QUERY, DATED, np.array([1, 0]), **kwargs)
    new = NativeSpineParentUserRouter(s, hierarchy, parents).route_vector(QUERY, DATED, np.array([1, 0]), **kwargs)
    return h, old, new


def test_complete_parent_reaches_sibling_outside_selected_neighborhood_without_displacement():
    h, old, new = routes()
    assert isinstance(new, NativeSpineParentUserRoute)
    assert new.parent_base == old
    assert new.expanded.routes[:len(old.expanded.routes)] == old.expanded.routes
    assert new.parent_added_atomic_ids == (h.atoms[2].section_id,)
    assert new.context_atomic_ids == old.context_atomic_ids
    assert new.consulted_chunk_ids == old.consulted_chunk_ids
    assert all(s.role == 'user' for r in new.parent_plan.routes for s in r.section.spans)
    assert all(r.section.source_id == h.atoms[0].source_id for r in new.expanded.routes)
    assert new.raw_reads_during_routing == new.query_qwen_passes == 0
    assert route_from_payload(new.identity_payload()) == new


def test_complete_existing_parent_evidence_returns_original_receipt():
    _, old, new = routes({**OPTIONS, 'max_additions': 3, 'ancestor_hops': 2})
    assert new == old
    assert not isinstance(new, NativeSpineParentUserRoute)


@pytest.mark.parametrize('limit', ['ample', 'tokens', 'reads'])
def test_original_raw_evidence_survives_identical_hydration_caps(limit):
    h, old, new = routes()
    before = hydrate_section_plan(old.expanded, load_turn=h.get_turn)
    caps = {'max_context_tokens': 512, 'max_raw_spans': 128}
    if limit == 'tokens': caps['max_context_tokens'] = before.context_token_count
    if limit == 'reads': caps['max_raw_spans'] = before.raw_turn_read_count
    after = hydrate_section_plan(new.expanded, load_turn=h.get_turn, **caps)
    assert after.sections[:len(before.sections)] == before.sections
    assert after.context_token_count <= caps['max_context_tokens']
    assert after.raw_turn_read_count <= caps['max_raw_spans']
    if limit == 'ample':
        assert after.sections[-1].evidence[0].text == h.get_turn(h.atoms[2].spans[0].turn_id).text
    else:
        assert after.sections == before.sections


@pytest.mark.parametrize('defect', ['scope', 'prefix', 'context'])
def test_sealed_parent_supplement_rejects_changed_scope_or_original_context(defect):
    _, _, new = routes()
    with pytest.raises(ValueError, match='parent supplement'):
        if defect == 'scope':
            parent = replace(new.parent_plan, eligible_source_ids=None, receipt_sha256='')
            replace(new, parent_plan=parent, receipt_sha256='')
        elif defect == 'prefix':
            expanded = replace(new.expanded, routes=new.expanded.routes[::-1], receipt_sha256='')
            replace(new, expanded=expanded, receipt_sha256='')
        else:
            replace(new, context_atomic_ids=(), receipt_sha256='')


def test_persist_both_indexes_reopen_application_and_embed_only_one_live_query(tmp_path):
    h, s, hierarchy, encoder, matrix, records, parents = fixture()
    with MemoryCondenser(tmp_path, embedder=encoder, auto_extract=False) as app:
        app.ingest_many(records)
        native = install(app, s, hierarchy, matrix)
    saved = store.publish(tmp_path / store.FILENAME, hierarchy=hierarchy,
                          matrix=parents._dense._matrix, native_receipt=native)
    with ParentUserMemoryCondenser(tmp_path, embedder=encoder, auto_extract=False, read_only=True) as app:
        assert app.native_spine_receipt() == native
        assert app.native_parent_user_receipt() == saved
        result = app.retrieve_native_spine(QUERY, DATED, **OPTIONS)
        assert encoder.calls == [QUERY]
        assert result.routing.parent_added_atomic_ids == (h.atoms[2].section_id,)
        assert result.hydration.sections[-1].evidence[0].text == h.get_turn(h.atoms[2].spans[0].turn_id).text
    encoder.model_revision = 'changed'
    with ParentUserMemoryCondenser(tmp_path, embedder=encoder, auto_extract=False, read_only=True) as app:
        with pytest.raises(ValueError, match='encoder differs'):
            app.native_spine_receipt()


def test_parent_index_cannot_use_a_different_query_encoder():
    _, s, hierarchy, _, _, _, parents = fixture()
    wrong = SemanticSectionIndex(parents.hierarchy, parents._dense._matrix, embedding_identity='different')
    with pytest.raises(ValueError, match='query encoder'):
        NativeSpineParentUserRouter(s, hierarchy, wrong)
