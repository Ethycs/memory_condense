from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.search.as_of_spine_routing import _plan
from memory_condense.search.fine_spine_routing import FineSpineRouter
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.source_spine_hydration import SourceSpineHydrationIndex
from memory_condense.search.summary_semantic_index import SemanticSectionIndex


def fixture():
    turns, atoms, leaves = {}, [], []
    for i in range(6):
        spans = []
        for role in ('user', 'assistant'):
            turn = Turn(turn_id=f't{i}-{role}', source_id=f'opaque-{i}', role=role,
                text=f'First half {i}. Second exact half {i}.',
                created_at=datetime(2026, 9, 20 if i == 5 else 10, tzinfo=timezone.utc))
            turns[turn.turn_id] = turn
            cut = turn.text.index('Second')
            for j, (start, end) in enumerate(((0, cut), (cut, len(turn.text)))):
                span = RawSectionSpan.from_turn(turn, start_char=start, end_char=end)
                atoms.append(SectionSummary(f'a{i}-{role}-{j}', turn.source_id, f'Summary {i} part {j}.', (span,), 'fixture'))
                spans.append(span)
        leaves.append(SectionSummary(f'l{i}', f'opaque-{i}',
            json.dumps({'attached_context_not_user_assertions': f'Attached context {i}.'}), tuple(spans), 'fixture'))
    hierarchy = SectionSummaryIndex(leaves)
    source = SourceSpineHydrationIndex(hierarchy, atoms)
    user_index = SectionSummaryIndex(tuple(a for a in atoms if a.spans[0].role == 'user'))
    matrix = np.array([[i + 1, 1] for i in range(len(user_index.sections))], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    fine = SemanticSectionIndex(user_index, matrix, embedding_identity='fixture')
    router = FineSpineRouter(fine, source)
    query = 'Which events did I report?'
    dated = '[Question asked at 2026/09/19 (Sat) 12:00]\n' + query
    context = source.attachments['l0']
    prior = _plan(source.index.receipt_sha256, quote_sha256(query), (context,))
    return router, source, turns, query, dated, prior


def test_direct_summary_matches_recover_whole_turns_outside_old_packet_without_future_reads():
    router, _, turns, query, dated, prior = fixture()
    plan, audit = router.route_vector(query, dated, [1, 0], embedding_identity='fixture', prior=prior)
    assert audit['raw_reads_during_routing'] == 0
    loaded = []
    def load(turn_id):
        loaded.append(turn_id)
        return turns[turn_id]
    hydrated = hydrate_section_plan(plan, load_turn=load, max_context_tokens=3072, max_raw_spans=128)
    assert set(loaded) == {f't{i}-user' for i in range(5)} | {'t0-assistant'}
    assert len(audit['fine_user_sections']) == 5
    assert all(len(s.evidence) == 2 for s in hydrated.sections)
    assert all('First half' in s.render_raw('X') and 'Second exact half' in s.render_raw('X') for s in hydrated.sections)


def test_date_reservation_stays_first_and_does_not_duplicate_raw_turns():
    router, source, _, query, dated, prior = fixture()
    reserved = source.by_source['opaque-0'][0]
    prior = _plan(source.index.receipt_sha256, prior.query_sha256, (reserved, *(r.section for r in prior.routes)))
    plan, _ = router.route_vector(query, dated, [1, 0], embedding_identity='fixture', prior=prior,
                                  reserved_ids=(reserved.section_id,))
    assert plan.routes[0].section == reserved
    spans = [p.receipt_sha256 for r in plan.routes for p in r.section.spans]
    assert len(spans) == len(set(spans))


def test_missing_fine_fragment_cannot_silently_reduce_user_coverage():
    router, source, *_ = fixture()
    index = SectionSummaryIndex(router.fine.sections[:-1])
    matrix = np.array([[1, 0]] * len(index.sections), dtype=np.float32)
    with pytest.raises(ValueError, match='partition'):
        FineSpineRouter(SemanticSectionIndex(index, matrix, embedding_identity='fixture'), source)


@pytest.mark.parametrize('corruption', ['encoder', 'query', 'reserved', 'future'])
def test_changed_bindings_and_future_context_are_rejected(corruption):
    router, source, _, query, dated, prior = fixture()
    options = {'embedding_identity': 'fixture', 'prior': prior}
    if corruption == 'encoder':
        options['embedding_identity'] = 'foreign'
    elif corruption == 'query':
        query += ' modified'
    elif corruption == 'reserved':
        options['reserved_ids'] = ('foreign',)
    else:
        options['prior'] = _plan(prior.index_sha256, prior.query_sha256, (source.attachments['l5'],))
    with pytest.raises(ValueError):
        router.route_vector(query, dated, [1, 0], **options)


def test_compiler_embeds_only_stored_user_summaries_and_preserves_complete_fragment_population(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from tools import compile_fine_spine_addresses as compiler
    from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json

    router, source, *_ = fixture()
    leaves = tuple(source.originals.values())
    user_atoms = list(router.fine.sections)
    assistant_atoms = [SectionSummary('assistant-' + p.receipt_sha256, p.source_id,
        'Assistant summary must not be embedded.', (p,), 'fixture')
        for leaf in leaves for p in leaf.spans if p.role != 'user']
    atoms_path = tmp_path / 'atoms.json'
    atoms, _ = publish_sealed_json(atoms_path, {'complete_namespace': True,
        'atoms': [a.identity_payload() for a in (*user_atoms, *assistant_atoms)]})
    manifest, _ = publish_sealed_json(tmp_path / 'base.json', {'complete_namespace': True})
    for offset in range(0, 100, 10):
        publish_sealed_json(tmp_path / 'source' / 'namespaces' / f'offset-{offset:03}' / 'preflight.json', {
            'index_root': str(tmp_path / 'original-index'), 'index_manifest_sha256': manifest.sha256,
            'atoms_path': str(atoms_path), 'atoms_sha256': atoms.sha256,
            'question': 'QUESTION SECRET', 'reference': 'GOLD SECRET'})
    allowed = {a.summary for a in user_atoms}
    seen = []
    class Encoder:
        def __init__(self, **kwargs):
            pass
        def embed_queries(self, texts):
            assert set(texts) <= allowed
            seen.extend(texts)
            return [[1.0, 0.0] for _ in texts]
        def close(self):
            pass
    monkeypatch.setattr(compiler, 'EmbeddingService', Encoder)
    monkeypatch.setattr(compiler, 'summary_embedding_identity', lambda _: 'fixture')
    monkeypatch.setattr(compiler, 'load_index', lambda _: (manifest,
        SimpleNamespace(hierarchy=SectionSummaryIndex(leaves), embedding_identity='fixture')))
    compiler.compile_all(tmp_path / 'source', tmp_path / 'compiled')
    assert set(seen) == allowed
    complete = read_sealed_json(tmp_path / 'compiled' / 'complete.json')
    assert len(complete.payload['indexes']) == 10
    for row in complete.payload['indexes']:
        artifact = read_sealed_json(Path(row['root']) / 'index.json')
        assert artifact.payload['user_fragment_count'] == len(user_atoms)
        index = SectionSummaryIndex.from_json(artifact.payload['index_json'])
        assert {p.receipt_sha256 for s in index.sections for p in s.spans} == {
            p.receipt_sha256 for a in user_atoms for p in a.spans}
