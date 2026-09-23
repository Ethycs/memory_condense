from dataclasses import replace
import json

import numpy as np
import pytest

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.native_spine_user_completion import UserCompletionMemoryCondenser
from memory_condense.application.section_retrieval import _render, hydrate_section_plan
from memory_condense.application.user_evidence_projection import TranscriptOrder, render_user_spine_sections
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.persistence import native_spine_parent_store as store
from memory_condense.search.native_spine_memory import materialize_history
from memory_condense.search.native_spine_parent_user_routing import NativeSpineParentUserRouter
from memory_condense.search.native_spine_parent_users import project_parent_users
from memory_condense.search.native_spine_summary import body_identity, fragment_body
from memory_condense.search.native_spine_user_completion import (
    NativeSpineUserCompletionRoute, NativeSpineUserCompletionRouter, is_user, route_from_payload,
    source_rank,
)
from memory_condense.search.section_routing import SectionRoute, SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.summary_semantic_index import SemanticSectionIndex, summary_embedding_identity
from tests.test_native_spine_application_lifecycle import install
from tests.test_native_spine_routing import DATED, QUERY, Encoder


# Three direct routes: the lead user turn, one long assistant slice, and the
# second conversation's lead. Two context slots re-admit the seed and one
# sibling, so later user turns of the seed conversation stay unrouted.
OPTIONS = dict(max_direct=3, lexical_reserve=0, context_seed_limit=1,
               max_additions=2, protected_direct=0, ancestor_hops=1)
ASSISTANT = ' '.join(f'Detail number {i} of a long recommendation list.' for i in range(40))
BODY = {'turns': [
    {'role': 'user', 'text': 'I visited the botanical garden.'},
    {'role': 'assistant', 'text': ASSISTANT},
    {'role': 'user', 'text': 'I also enjoy watching birds.'},
    {'role': 'assistant', 'text': 'Kingfishers are common there.'},
    {'role': 'user', 'text': 'I chose the riverside trail.'},
    {'role': 'user', 'text': 'Actually, make that the hill trail.'},
    {'role': 'user', 'text': 'I will bring the field guide.'},
    {'role': 'user', 'text': 'And the new binoculars.'},
]}
DAYS = ('2026-09-01', '2026-09-02', '2026-09-03')


def fixture():
    cached = [{'pointer': f.pointer(), 'summary': f'Stored summary {i}.'}
              for i, f in enumerate(fragment_body(BODY, token_cap=128))]
    sessions = []
    for i, day in enumerate(DAYS):
        s = {'original_session_ordinal': i, 'session_id': f'fixture-{i}', 'created_at': day + 'T00:00:00+00:00',
             'metadata_text': 'source', 'body_sha256': body_identity(BODY), 'dataset_origin': 'M'}
        s['occurrence_id'] = identity_sha256(s)
        sessions.append(s)
    history = materialize_history(sessions, load_body=lambda _: BODY, load_summaries=lambda _: cached,
                                  compiler_identity='test')
    by_source = {}
    for atom in history.atoms:
        by_source.setdefault(atom.source_id, []).append(atom)
    hierarchy = []
    for n, (source, atoms) in enumerate(by_source.items()):
        spans = tuple(a.spans[0] for a in atoms)
        leaves = [SectionSummary(f'leaf-{n}-{i:02d}', source, 'An exchange.', spans[i:i + 2], 'test')
                  for i in range(0, len(spans), 2)]
        hierarchy.extend(leaves)
        hierarchy.append(SectionSummary(f'root-{n}', source, json.dumps({
            'user_spine': 'The user described a garden visit, birds and a trail choice.',
            'attached_context_not_user_assertions': 'Assistant listed details.'}),
            spans, 'test', tuple(l.section_id for l in leaves)))
    hierarchy = SectionSummaryIndex(hierarchy)
    index = SectionSummaryIndex(history.atoms)
    encoder = Encoder()
    identity = summary_embedding_identity(encoder)
    lead = {source: w for source, w in zip(by_source, ((1.0, 0.0), (0.6, 0.8), (0.0, 1.0)))}
    first_source = next(iter(by_source))
    def vector(atom):
        atoms = by_source[atom.source_id]
        if atom == atoms[0]:
            return lead[atom.source_id]
        if atom.source_id == first_source and atom == atoms[1]:
            return (0.8, 0.6)  # the first slice of the long assistant reply is a direct match
        return (0.0, 1.0)
    vectors = np.array([vector(a) for a in index.sections], dtype=np.float32)
    semantic = SemanticSectionIndex(index, vectors, embedding_identity=identity)
    projection = project_parent_users(hierarchy)
    parent_weights = {source: w for source, w in zip(by_source, ((0.0, 1.0), (1.0, 0.0), (0.0, 1.0)))}
    parents = SemanticSectionIndex(projection, np.array([parent_weights[p.source_id] for p in projection.sections],
                                                        dtype=np.float32), embedding_identity=identity)
    return history, semantic, hierarchy, encoder, parents, by_source


def routes(limit, options=None):
    h, s, hierarchy, _, parents, by_source = fixture()
    kwargs = dict(embedding_identity=s.embedding_identity, **(options or OPTIONS))
    base = NativeSpineParentUserRouter(s, hierarchy, parents).route_vector(QUERY, DATED, np.array([1, 0]), **kwargs)
    new = NativeSpineUserCompletionRouter(s, hierarchy, parents).route_vector(
        QUERY, DATED, np.array([1, 0]), user_completion_atoms=limit, **kwargs)
    return h, base, new, by_source


def test_prior_route_is_preserved_with_user_routes_before_assistant_context():
    h, base, new, _ = routes(0)
    assert isinstance(new, NativeSpineUserCompletionRoute)
    assert new.completion_base == base and new.completion_added_atomic_ids == ()
    assert new.baseline == base.baseline and new.context_atomic_ids == base.context_atomic_ids
    assert {r.section for r in new.expanded.routes} == {r.section for r in base.expanded.routes}
    roles = [is_user(r) for r in new.expanded.routes]
    assert roles == sorted(roles, reverse=True) and False in roles
    for role in (True, False):
        assert [r for r in new.expanded.routes if is_user(r) == role] == [r for r in base.expanded.routes if is_user(r) == role]
    assert new.raw_reads_during_routing == new.query_qwen_passes == 0
    assert route_from_payload(new.identity_payload()) == new
    assert route_from_payload(base.identity_payload()) == base


def test_completion_adds_every_remaining_user_turn_of_routed_conversations_in_order():
    h, base, new, by_source = routes(64)
    routed = source_rank(base)
    unrouted = [s for s in by_source if s not in routed]
    assert len(routed) == 2 and len(unrouted) == 1
    prior = {r.section.section_id for r in base.expanded.routes}
    added = [r.section for r in new.expanded.routes if r.section.section_id not in prior]
    assert tuple(a.section_id for a in added) == new.completion_added_atomic_ids
    assert added and all(a.spans[0].role == 'user' and a.source_id in routed for a in added)
    served = {r.section.section_id for r in new.expanded.routes}
    for source in routed:
        expected = [a for a in by_source[source] if a.spans[0].role == 'user']
        assert all(a.section_id in served for a in expected)
        positions = [expected.index(a) for a in added if a.source_id == source]
        assert positions == sorted(positions)
    assert all(a.source_id in routed for r in new.expanded.routes for a in (r.section,))
    assert new.expanded.routes[:len(base.baseline.routes)] == base.expanded.routes[:len(base.baseline.routes)]


def test_completion_is_bounded_and_round_robin_starts_at_the_best_ranked_conversation():
    h, base, one, _ = routes(1)
    assert len(one.completion_added_atomic_ids) == 1
    first = [r.section for r in one.expanded.routes if r.section.section_id == one.completion_added_atomic_ids[0]][0]
    remaining = [s for s in source_rank(base) if any(
        a.source_id == s and a.spans[0].role == 'user' and a.section_id not in {r.section.section_id for r in base.expanded.routes}
        for a in h.atoms)]
    assert first.source_id == remaining[0]
    _, _, two, _ = routes(2)
    assert len(two.completion_added_atomic_ids) == 2
    assert two.completion_added_atomic_ids[0] == one.completion_added_atomic_ids[0]
    if len(remaining) > 1:
        second = [a for a in h.atoms if a.section_id == two.completion_added_atomic_ids[1]][0]
        assert second.source_id == remaining[1]


def test_late_user_decision_survives_the_budget_that_previously_kept_assistant_context():
    h, base, new, _ = routes(0)
    late = [r for r in base.expanded.routes[len(base.baseline.routes):]]
    assistant_at = max(i for i, r in enumerate(base.expanded.routes) if not is_user(r))
    assert any(is_user(r) for r in base.expanded.routes[assistant_at:]), 'fixture needs a user route after assistant context'
    full = hydrate_section_plan(base.expanded, load_turn=h.get_turn, max_context_tokens=100_000)
    budget = count_tokens(_render(full.sections[:assistant_at + 1])) + 5
    before = hydrate_section_plan(base.expanded, load_turn=h.get_turn, max_context_tokens=budget)
    after = hydrate_section_plan(new.expanded, load_turn=h.get_turn, max_context_tokens=budget)
    dropped_before = {d.section_id for d in before.diagnostics}
    dropped_after = {d.section_id for d in after.diagnostics}
    users = {r.section.section_id for r in base.expanded.routes if is_user(r)}
    assert dropped_before & users, 'baseline order loses a routed user turn to the budget'
    assert not (dropped_after & users)
    assert dropped_after and all(not is_user(r) for r in base.expanded.routes if r.section.section_id in dropped_after)
    assert {s.section.section_id for s in before.sections if s.section.spans[0].role == 'user'} <= {
        s.section.section_id for s in after.sections}
    order = TranscriptOrder(list(h.turns.values()))
    rendered = render_user_spine_sections(after, order)
    assert all(h.get_turn(r.section.spans[0].turn_id).text in rendered.text for r in base.expanded.routes if is_user(r))
    assert after.context_token_count <= budget


@pytest.mark.parametrize('defect', ['order', 'foreign', 'limit', 'base'])
def test_sealed_completion_rejects_reordering_foreign_atoms_or_exceeded_limits(defect):
    h, base, new, by_source = routes(2)
    with pytest.raises(ValueError, match='user completion'):
        if defect == 'order':
            expanded = replace(new.expanded, routes=new.expanded.routes[::-1], receipt_sha256='')
            replace(new, expanded=expanded, receipt_sha256='')
        elif defect == 'foreign':
            outside = next(a for s, atoms in by_source.items() if s not in source_rank(base) for a in atoms
                           if a.spans[0].role == 'user')
            extra = SectionRoute(outside, 0.01, ())
            expanded = replace(new.expanded, routes=(*new.expanded.routes[:-1], extra, new.expanded.routes[-1]),
                               max_sections=len(new.expanded.routes) + 1,
                               matched_section_count=len(new.expanded.routes) + 1, receipt_sha256='')
            replace(new, expanded=expanded, completion_added_atomic_ids=(*new.completion_added_atomic_ids,
                                                                          outside.section_id), receipt_sha256='')
        elif defect == 'limit':
            replace(new, user_completion_atoms=1, receipt_sha256='')
        else:
            _, other, _, _ = routes(2, {**OPTIONS, 'max_direct': 1})
            assert other != base
            replace(new, completion_base=other, receipt_sha256='')


def test_persisted_application_reopens_with_completion_and_one_live_query(tmp_path):
    h, s, hierarchy, encoder, parents, _ = fixture()
    encoder.dim = 2
    encoder.embed_chunks = lambda chunks: [c.model_copy(update={'embedding': [1.0, 0.0]}) for c in chunks]
    records = [(t.role, t.text, t.source_id, t.created_at, t.turn_id) for t in h.turns.values()]
    with MemoryCondenser(tmp_path, embedder=encoder, auto_extract=False) as app:
        app.ingest_many(records)
        native = install(app, s, hierarchy, s._dense._matrix.copy())
    store.publish(tmp_path / store.FILENAME, hierarchy=hierarchy, matrix=parents._dense._matrix, native_receipt=native)
    with UserCompletionMemoryCondenser(tmp_path, embedder=encoder, auto_extract=False, read_only=True) as app:
        assert app.native_spine_receipt() == native
        result = app.retrieve_native_spine(QUERY, DATED, user_completion_atoms=64, **OPTIONS)
        assert encoder.calls == [QUERY]
        assert isinstance(result.routing, NativeSpineUserCompletionRoute)
        assert result.routing.completion_added_atomic_ids
        assert result.routing.raw_reads_during_routing == 0
        texts = [e.text for section in result.hydration.sections for e in section.evidence]
        assert BODY['turns'][5]['text'] in texts
        assert route_from_payload(result.routing.identity_payload()) == result.routing
