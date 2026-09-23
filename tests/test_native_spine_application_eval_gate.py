from copy import deepcopy

import pytest

from tools import evaluate_native_spine_threaded100 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json
from tools.prepare_native_spine_design_slice import binding


def verification(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation.app_lifecycle, 'implementation', lambda: {'fixture': 'implementation'})
    plan, _ = publish_sealed_json(tmp_path / 'plan.json', {
        'implementation': {'fixture': 'implementation'}, 'scope': {'fixture': 'one-history'}})
    ingested, _ = publish_sealed_json(tmp_path / 'ingested.json', {
        'ingest_plan': binding(plan), 'snapshot': {'fixture': 'application-snapshot'}})
    payload = {'ingest_complete': binding(ingested), 'history_count': 1, 'question_count': 100,
        'new_process_reopened': True, 'identical_baseline_packets': True,
        'source_namespace_loaded': False, 'source_vector_cache_loaded': False,
        'snapshot': ingested.payload['snapshot']}
    return payload, ingested, plan


def test_answer_admission_requires_reopened_application_without_source_cache(tmp_path, monkeypatch):
    payload, ingested, plan = verification(tmp_path, monkeypatch)
    good, _ = publish_sealed_json(tmp_path / 'verified.json', payload)
    assert evaluation.application_admission(good) == (ingested, plan)
    for field, value in [('history_count', 100), ('question_count', 99),
                         ('new_process_reopened', False), ('identical_baseline_packets', False),
                         ('source_namespace_loaded', True), ('source_vector_cache_loaded', True),
                         ('snapshot', {'fixture': 'different-memory'})]:
        bad = deepcopy(payload)
        bad[field] = value
        artifact, _ = publish_sealed_json(tmp_path / f'{field}.json', bad)
        with pytest.raises(ValueError, match='independent reopen verification'):
            evaluation.application_admission(artifact)
