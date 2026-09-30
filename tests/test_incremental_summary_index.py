"""Incremental summaries preserve full-rebuild scoring and source bindings."""
from dataclasses import replace
import json

import pytest

from memory_condense.search import section_routing as routing
from memory_condense.search.native_spine_parent_users import project_parent_users
from tests.test_semantic_section_index import hierarchy


def test_updates_reuse_unchanged_terms_and_match_full_rebuild(monkeypatch):
    _, old = hierarchy()
    before = old.to_json()
    changed = replace(old._by_id['1'], summary='coastal pine drive', receipt_sha256='')
    added = replace(old._by_id['2'], section_id='-sorts-before', summary='rare new phrase', receipt_sha256='')
    population = [s if s.section_id != '1' else changed for s in old.sections] + [added]
    fresh = routing.SectionSummaryIndex(population)
    calls = []
    tokenize = routing.tokenize
    def counted(text):
        calls.append(text)
        return tokenize(text)
    monkeypatch.setattr(routing, 'tokenize', counted)
    updated = old.updated(population)
    assert sorted(calls) == sorted([changed.summary, added.summary])
    assert updated.to_json() == fresh.to_json()
    assert old.to_json() == before
    for query in ('pine', 'coastal drive', 'rare phrase', 'missing', 'lossy overview'):
        for scope in (None, (), ('a',), ('b',)):
            assert updated.route(query, max_sections=8, eligible_source_ids=scope) == fresh.route(
                query, max_sections=8, eligible_source_ids=scope)
    trimmed = updated.updated([s for s in population if s.section_id not in ('2', '-sorts-before')])
    clean = routing.SectionSummaryIndex(trimmed.sections)
    assert trimmed.to_json() == clean.to_json()
    assert trimmed.route('coastal pine') == clean.route('coastal pine')
    with pytest.raises(ValueError, match='missing child'):
        old.updated([s for s in old.sections if s.section_id != '0'])


def test_parent_id_survives_changes_elsewhere():
    _, old = hierarchy()
    roots = [replace(s, summary=json.dumps({'user_spine': s.summary}), receipt_sha256='')
             if s.section_id in old._roots else s for s in old.sections]
    old = routing.SectionSummaryIndex(roots)
    parents = project_parent_users(old, stable_ids=True)
    updated = old.updated([replace(s, summary=json.dumps({'user_spine': 'changed other source'}), receipt_sha256='')
                           if s.section_id == '2' else s for s in old.sections])
    new = project_parent_users(updated, stable_ids=True, previous=parents)
    unchanged, = [s for s in parents.sections if s.source_id == 'a']
    assert new._by_id[unchanged.section_id] is unchanged
    assert new.to_json() == project_parent_users(updated, stable_ids=True).to_json()
