from dataclasses import replace

import numpy as np
import pytest

from memory_condense.application.native_spine_additive_lexical import AdditiveLexicalMemoryCondenser
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.native_spine_additive_lexical import (
    NativeSpineAdditiveLexicalRoute, NativeSpineAdditiveLexicalRouter, route_from_payload,
)
from memory_condense.search.native_spine_context_routing import NativeSpineContextRouter
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.summary_semantic_index import SemanticSectionIndex
from tests.test_native_spine_application_lifecycle import inputs, install


QUERY = 'Where did I watch kingfishers?'
DATED = '[Question asked at 2026/09/12 (Saturday) 12:00] ' + QUERY
OPTIONS = dict(max_direct=1, lexical_reserve=0, context_seed_limit=1,
               max_additions=1, protected_direct=0, ancestor_hops=2)


def setup():
    h, semantic, hierarchy, encoder, matrix, records = inputs()
    # Dense vectors miss the sibling; summaries alone can recover it. A more
    # repetitive future summary must remain outside the query's dated scope.
    summaries = []
    for a in semantic.sections:
        text = 'An ordinary visit.'
        if a == h.atoms[2]:
            text = 'The user watched kingfishers.'
        elif a == h.atoms[5]:
            text = 'kingfishers kingfishers kingfishers'
        summaries.append(replace(a, summary=text, receipt_sha256=''))
    semantic = SemanticSectionIndex(SectionSummaryIndex(summaries), matrix,
                                   embedding_identity=semantic.embedding_identity)
    return h, semantic, hierarchy, encoder, matrix, records


def routes(options=None):
    h, s, hierarchy, _, _, _ = setup()
    kwargs = dict(embedding_identity=s.embedding_identity, **(options or OPTIONS))
    old = NativeSpineContextRouter(s, hierarchy).route_vector(QUERY, DATED, np.array([1, 0]), **kwargs)
    new = NativeSpineAdditiveLexicalRouter(s, hierarchy).route_vector(QUERY, DATED, np.array([1, 0]), **kwargs)
    return h, old, new


def test_summary_only_addition_preserves_seeds_context_and_route_prefix():
    h, old, new = routes()
    assert isinstance(new, NativeSpineAdditiveLexicalRoute)
    assert new.context_route == old
    assert new.expanded.routes[:-1] == old.expanded.routes
    assert new.lexical_atomic_ids == (h.atoms[2].section_id,)
    assert new.consulted_chunk_ids == old.consulted_chunk_ids
    assert new.context_atomic_ids == old.context_atomic_ids
    assert new.raw_reads_during_routing == new.query_qwen_passes == 0
    assert all(r.section.source_id != h.atoms[5].source_id for r in new.expanded.routes)
    assert route_from_payload(new.identity_payload()) == new


def test_already_selected_lexical_match_keeps_original_receipt():
    _, old, new = routes({**OPTIONS, 'max_additions': 3})
    assert new == old
    assert not isinstance(new, NativeSpineAdditiveLexicalRoute)
    assert route_from_payload(new.identity_payload()) == old


@pytest.mark.parametrize('limit', ['full', 'tokens', 'reads'])
def test_same_hydration_budget_keeps_all_prior_evidence_and_never_truncates(limit):
    h, old, new = routes()
    original = hydrate_section_plan(old.expanded, load_turn=h.get_turn)
    caps = dict(max_context_tokens=2048, max_raw_spans=128)
    if limit == 'tokens':
        caps['max_context_tokens'] = original.context_token_count
    if limit == 'reads':
        caps['max_raw_spans'] = original.raw_turn_read_count
    before = hydrate_section_plan(old.expanded, load_turn=h.get_turn, **caps)
    after = hydrate_section_plan(new.expanded, load_turn=h.get_turn, **caps)
    assert after.sections[:len(before.sections)] == before.sections
    assert after.context_token_count <= caps['max_context_tokens']
    assert after.raw_turn_read_count <= caps['max_raw_spans']
    if limit == 'full':
        assert after.sections[-1].evidence[0].text == h.get_turn(h.atoms[2].spans[0].turn_id).text
    else:
        assert after.sections == before.sections


@pytest.mark.parametrize('defect', ['prefix', 'scope', 'seeds', 'limit', 'identity'])
def test_receipt_rejects_changed_original_plan_or_unbounded_lexical_stage(defect):
    _, _, new = routes()
    with pytest.raises(ValueError):
        if defect == 'prefix':
            plan = replace(new.expanded, routes=new.expanded.routes[::-1], receipt_sha256='')
            replace(new, expanded=plan, receipt_sha256='')
        elif defect == 'scope':
            plan = replace(new.lexical_plan, eligible_source_ids=None, receipt_sha256='')
            replace(new, lexical_plan=plan, receipt_sha256='')
        elif defect == 'seeds':
            replace(new, consulted_chunk_ids=(), receipt_sha256='')
        elif defect == 'limit':
            plan = replace(new.lexical_plan, max_sections=2, receipt_sha256='')
            replace(new, lexical_plan=plan, receipt_sha256='')
        else:
            replace(new, lexical_atomic_ids=('foreign',))


def test_application_ingest_close_reopen_uses_same_snapshot_and_one_query_embedding(tmp_path):
    h, s, hierarchy, encoder, matrix, records = setup()
    with AdditiveLexicalMemoryCondenser(tmp_path, embedder=encoder, auto_extract=False) as app:
        app.ingest_many(records)
        receipt = install(app, s, hierarchy, matrix)
    with AdditiveLexicalMemoryCondenser(tmp_path, embedder=encoder,
                                      auto_extract=False, read_only=True) as app:
        assert app.native_spine_receipt() == receipt
        reads = []
        load = app.transcript.get_turn
        def observe(tid):
            reads.append(tid)
            return load(tid)
        app.transcript.get_turn = observe
        app._native_spine_loaded = None
        result = app.retrieve_native_spine(QUERY, DATED, **OPTIONS)
        assert encoder.calls == [QUERY]
        assert result.routing.lexical_atomic_ids == (h.atoms[2].section_id,)
        assert reads == [e.span.turn_id for s in result.hydration.sections for e in s.evidence]
        assert all(e.text == h.get_turn(e.span.turn_id).text
                   for s in result.hydration.sections for e in s.evidence)
    with pytest.raises(Exception, match='closed'):
        app.retrieve_native_spine(QUERY, DATED, **OPTIONS)


def test_application_retains_encoder_and_revision_gates(tmp_path):
    _, s, hierarchy, encoder, matrix, records = setup()
    with AdditiveLexicalMemoryCondenser(tmp_path, embedder=encoder, auto_extract=False) as app:
        app.ingest_many(records)
        install(app, s, hierarchy, matrix)
    encoder.model_revision = 'wrong'
    with AdditiveLexicalMemoryCondenser(tmp_path, embedder=encoder, auto_extract=False) as app:
        with pytest.raises(ValueError, match='encoder differs'):
            app.native_spine_receipt()
        encoder.model_revision = 'test'
        app.native_spine_receipt()
        app.ingest('user', 'A new fact.', source_id='new')
        with pytest.raises(ValueError, match='advanced'):
            app.retrieve_native_spine(QUERY, DATED, **OPTIONS)
