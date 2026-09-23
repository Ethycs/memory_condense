import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from tests.test_native_spine_exchanges import Backend, prepared
from tools import compile_pending_native_spine_exchanges as compiler
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


def stopped(prepared):
    root, history = prepared
    original = read_sealed_json(root/'inputs.json')
    source = root/'stopped'
    body_row = original.payload['bodies'][0]
    body = read_sealed_json(root/body_row['path'])
    copied, _ = publish_sealed_json(source/body_row['path'], body.payload)
    completed_id = 'f'*64
    # Intentionally unavailable prior body files: this stage must only retain
    # their checkpoint bindings; complete readmission remains a later boundary.
    inputs, _ = publish_sealed_json(source/'inputs.json', {
        **original.payload, 'sources_sha256': 'fixture-sources', 'body_count': 2,
        'bodies': [dict(body_row, sha256=copied.sha256),
            {'body_sha256': completed_id, 'path': 'bodies/completed-must-not-be-read.json', 'sha256': '0'*64}]})
    backend = Backend()
    plan, _ = publish_sealed_json(source/'preflight.json', {
        'producer_format': compiler.previous.FORMAT, 'producer_implementation': compiler.previous.implementation(),
        'inputs_sha256': inputs.sha256, 'backend_sha256': backend.identity_sha256})
    report, _ = publish_sealed_json(source/'partial.json', {
        'preflight_sha256': plan.sha256, 'body_count': 1, 'prepared_body_count': 2,
        'raw_inputs_to_qwen': False,
        'compiled_bodies': [{'path': f'exchanges/{completed_id}.json', 'sha256': '1'*64}]})
    cache, _ = publish_sealed_json(root/'cache.json', {
        'inputs': binding(inputs), 'source_receipt': {'root': str(source.resolve()),
            'preflight_sha256': plan.sha256, 'merge_cache_sha256': identity_sha256({}), 'accepted_merge_count': 0},
        'backend_sha256': backend.identity_sha256,
        'implementation_sha256': compiler.digest(compiler.cache_reader.__file__),
        'values': {}, 'attempted': [], 'new_model_calls': 0, 'body_compilations': 0, 'body_files_copied': 0})
    return root, history, backend, report, cache


def test_only_pending_body_compiles_and_replays_with_exact_raw_coverage(prepared):
    root, history, backend, report, cache = stopped(prepared)
    target = root/'pending'
    plan = compiler.prepare(target, report.path, cache.path)
    assert len(plan.payload['pending_bodies']) == 1
    assert not (target/'bodies').exists()
    result = compiler.execute(target, backend, 128)
    assert result.payload['complete_pending_body_exchanges'] is True
    assert result.payload['combined_body_binding_count'] == 2
    assert result.payload['full_population_readmission_complete'] is False
    assert result.payload['completed_body_recompilations'] == 0
    row = bound(result.payload['compiled_bodies'][0]['artifact'])
    index = SectionSummaryIndex(tuple(SectionSummary.from_dict(e['section']) for e in row.payload['exchanges']))
    hydrated = hydrate_section_plan(index.route('furniture', max_sections=8), load_turn=history.get_turn,
        max_context_tokens=4096, max_raw_spans=128)
    assert hydrated.sections
    for section in hydrated.sections:
        for evidence in section.evidence:
            original = history.get_turn(evidence.span.turn_id)
            assert evidence.text == original.text[evidence.span.start_char:evidence.span.end_char]
    calls = backend.calls
    assert calls > 0
    assert compiler.execute(target, backend, 0).sha256 == result.sha256
    assert backend.calls == calls


def test_budget_zero_prepares_without_generation_and_then_continues(prepared):
    root, _, backend, report, cache = stopped(prepared)
    target = root/'pending'
    compiler.prepare(target, report.path, cache.path)
    partial = compiler.execute(target, backend, 0)
    assert backend.calls == 0 and partial.payload['body_count'] == 0
    assert compiler.execute(target, backend, 128).payload['complete_pending_body_exchanges'] is True


def test_pending_resume_does_not_retry_unacknowledged_generation(prepared):
    root, _, backend, report, cache = stopped(prepared)
    target = root/'pending'
    compiler.prepare(target, report.path, cache.path)
    backend.fail = True
    with pytest.raises(RuntimeError, match='simulated stopped'):
        compiler.execute(target, backend, 128)
    calls = backend.calls
    with pytest.raises(ValueError, match='refusing an implicit retry'):
        compiler.execute(target, backend, 128)
    assert backend.calls == calls


def test_modified_pending_source_rejects_before_generation(prepared):
    root, _, backend, report, cache = stopped(prepared)
    target = root/'pending'
    plan = compiler.prepare(target, report.path, cache.path)
    path = bound(plan.payload['pending_bodies'][0]['atoms']).path
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):
        compiler.execute(target, backend, 128)
    assert backend.calls == 0
