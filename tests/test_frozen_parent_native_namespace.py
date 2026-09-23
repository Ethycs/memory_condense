from dataclasses import replace

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.parent_budget_hierarchy_occurrence import bind_parent_budget_hierarchy
from memory_condense.search.section_routing import SectionSummaryIndex
from tests.test_native_hierarchy_occurrence import fixture
from tools import frozen_parent_native_spine_namespace as reader
from tools.matched_eval.artifacts import publish_sealed_json
from tools.prepare_native_spine_design_slice import binding


def saved(root):
    template, occurrence = fixture()
    tree = SectionSummaryIndex.from_json(template['index_json'])
    counts = {'atomic_count': len(SectionSummaryIndex.from_json(template['atomic_index_json']).sections),
        'leaf_count': sum(not s.child_section_ids for s in tree.sections),
        'parent_count': sum(bool(s.child_section_ids) for s in tree.sections)}
    artifact, _ = publish_sealed_json(root/'parent.json', {
        **template, **counts, 'preflight_sha256': 'existing-producer', 'raw_inputs_to_qwen': False})
    row = {'body_sha256': template['body_sha256'], 'artifact': binding(artifact),
        'parent_preflight_sha256': 'existing-producer', **counts}
    return row, artifact, occurrence


def test_lazy_template_preserves_exact_evidence_at_distinct_source_dates(tmp_path, monkeypatch):
    row, artifact, occurrence = saved(tmp_path)
    calls = []
    original = reader.bound
    def tracked(ref):
        calls.append(ref['sha256'])
        return original(ref)
    monkeypatch.setattr(reader, 'bound', tracked)
    templates = reader.LazyParentTemplates([row])
    assert row['body_sha256'] in templates and len(templates) == 1 and not calls
    for day in ('2026-01-02', '2026-09-12'):
        source, history = occurrence(day)
        template = templates.get(source['body_sha256'])
        tree, _, _ = bind_parent_budget_hierarchy(template, source, history.atoms)
        packet = hydrate_section_plan(tree.route('visits tomorrow May 4', max_sections=len(tree.sections)),
            load_turn=history.get_turn, max_raw_spans=128, max_context_tokens=4096)
        assert not packet.diagnostics
        for section in packet.sections:
            for evidence in section.evidence:
                span = evidence.span
                assert evidence.text == history.get_turn(span.turn_id).text[span.start_char:span.end_char]
                assert span.created_at == source['created_at']
    assert calls == [artifact.sha256]


@pytest.mark.parametrize('defect', ['artifact_hash', 'producer', 'body'])
def test_lazy_template_rejects_foreign_bindings(tmp_path, defect):
    row, _, _ = saved(tmp_path)
    if defect == 'artifact_hash':
        row = dict(row, artifact=dict(row['artifact'], sha256='0'*64))
    elif defect == 'producer':
        row = dict(row, parent_preflight_sha256='foreign-producer')
    else:
        row = dict(row, body_sha256='foreign-body')
    with pytest.raises(ValueError):
        reader.LazyParentTemplates([row])[row['body_sha256']]


def test_evicted_tree_is_authenticated_again_before_reuse(tmp_path):
    rows = []
    for key in ('one', 'two'):
        artifact, _ = publish_sealed_json(tmp_path/f'{key}.json', {
            'body_sha256': key, 'preflight_sha256': 'producer', 'raw_inputs_to_qwen': False,
            'original_atomic_addresses_preserved': True, 'atomic_count': 1, 'leaf_count': 1, 'parent_count': 0})
        rows.append({'body_sha256': key, 'artifact': binding(artifact), 'parent_preflight_sha256': 'producer',
            'atomic_count': 1, 'leaf_count': 1, 'parent_count': 0})
    templates = reader.LazyParentTemplates(rows, capacity=1)
    templates['one']
    templates['two']
    assert list(templates.cache) == ['two']
    path = tmp_path/'one.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):
        templates['one']


def test_combined_admission_rejects_incomplete_or_replaced_cached_trees(tmp_path):
    row, _, _ = saved(tmp_path)
    scope, _ = publish_sealed_json(tmp_path/'scope.json', {
        'format': reader.preparation.FORMAT, 'implementation_sha256': reader.digest(reader.preparation.__file__),
        'body_count': 1, 'bodies': [{'body_sha256': row['body_sha256'], 'parent': row['artifact'],
            'parent_preflight_sha256': row['parent_preflight_sha256']}]})
    plan, _ = publish_sealed_json(tmp_path/'preflight.json', {
        'format': 'native-spine-parent-budgeted-hierarchy-v1', 'max_exchange_channel_tokens': 128,
        'max_parent_channel_tokens': 512, 'leaf_token_cap': 512, 'max_leaf_exchanges': 2,
        'window_exchange_cap': 8, 'max_prompt_tokens': 2048, 'raw_inputs_to_qwen': False,
        'timestamp_metadata_in_model_inputs': False, 'original_atomic_addresses_preserved': True,
        'producer_format': reader.compiler.FORMAT, 'producer_implementation': reader.compiler.implementation(),
        'implementation': reader.compiler.parent.implementation(), 'scope': binding(scope)})
    report, _ = publish_sealed_json(tmp_path/'result.json', {
        'producer_format': reader.compiler.FORMAT, 'preflight_sha256': plan.sha256, 'scope': binding(scope),
        'complete_available_body_hierarchies': True, 'complete_native_hierarchies': True,
        'complete_source_compilation': True, 'raw_inputs_to_qwen': False, 'original_atomic_addresses_preserved': True,
        'body_count': 1, 'prepared_body_count': 1, 'templates': [row]})
    reader.validate_complete_result(report, plan, scope)
    with pytest.raises(ValueError, match='complete matching'):
        reader.validate_complete_result(replace(report, payload=dict(report.payload, complete_native_hierarchies=False)), plan, scope)
    with pytest.raises(ValueError, match='replaced an existing'):
        reader.validate_complete_result(replace(report, payload=dict(report.payload,
            templates=[dict(row, artifact=dict(row['artifact'], sha256='0'*64))])), plan, scope)
